"""参考训练循环（torch 后端）与门禁输入装配（plan 07 §4.1 / §4.3 / M7.2）。

分工：

- `beatmorph/infra/sanity.py`：**判据**（范式中立，只吃 step_fn）；
- 本模块：把「模型 + 批次 + 优化器」接成 `step_fn`，装配 G1-G4 的输入，并提供训练循环；
- `beatmorph/infra/gates.py`：落盘与 fail-closed；
- `beatmorph/infra/lightning_module.py`：plan 07 §4.1 的目标训练栈（可选依赖）。

为什么默认后端是 torch 而不是 Lightning：`pytorch-lightning` 属 `train` extra，而
**门禁必须在默认 CI 里能跑**（CLAUDE.md §4：契约级与门禁级断言不得依赖权重/GPU/可选依赖）。
Lightning 后端仍然提供，并在缺失时给出可操作的报错（而不是静默降级）。

两个同名的打乱助手（都在 `infra/smoke.py`，**别混用**）：

- `shuffle_hidden_counts`：**G2 门禁用的严格口径**——只在被遮盖格子内置换标签，
  可见场逐位不变 ⇒ 满足 plan 07 §4.3 的「同模型同输入，真标签 / 打乱标签」；
- `shuffled_counts`：整体置换（`BatchSource.batch(shuffled=True)` 的历史口径），
  会连输入一起换掉，只适合「目标结构被破坏」那类对照，**门禁已不再使用**。
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import torch
from torch import nn

from beatmorph.core.logging import get_logger
from beatmorph.field.grid import FieldGrid
from beatmorph.field.integrate import omega
from beatmorph.field.loss import constant_baseline_nll
from beatmorph.generation.batch import FieldBatch
from beatmorph.generation.model import MaskedFieldModel
from beatmorph.infra.artifacts import RunArtifacts, git_rev
from beatmorph.infra.checkpoint import (
    BEST_CHECKPOINT_NAME,
    load_checkpoint,
    rotate_checkpoints,
    save_checkpoint,
)
from beatmorph.infra.config.schema import TrainConfig
from beatmorph.infra.gates import GateInputs
from beatmorph.infra.sanity import StepFn
from beatmorph.infra.smoke import shuffle_hidden_counts, shuffled_counts

__all__ = [
    "BatchSource",
    "ManifestBatchSource",
    "TrainReport",
    "autocast_context",
    "autocast_dtype",
    "betas_of",
    "build_gate_inputs",
    "constant_baseline_for",
    "draw_until_min_events",
    "event_total",
    "make_step_fn",
    "model_from_config",
    "set_initial_head_bias",
    "train",
    "write_scalars",
]

if TYPE_CHECKING:
    from beatmorph.data.dataset import ChartPairDataset
    from beatmorph.data.plan import WindowPlan

logger = get_logger("infra.train_loop")

#: 过程标量的落盘文件名（jsonl；健康检查脚本与 TB 都读它）。
HISTORY_FILENAME: str = "loss_history.jsonl"


class BatchSource(Protocol):
    """批次来源协议：冒烟（`infra.smoke`）与真实清单（`ManifestBatchSource`）都满足它。"""

    def batch(
        self,
        *,
        masked: bool,
        shuffled: bool = False,
        samples: int | None = None,
        min_events: int = 0,
    ) -> FieldBatch:
        """取一个 batch；`masked` 控制遮盖路径，`samples` 覆盖批大小（None = 配置值）。

        `min_events > 0` 时**必须**取到含至少这么多事件的批（不够就重抽，抽不到即抛）：
        门禁的判据在空批上会**空过**（真实数据实测 G1 批 K=2 / 0 事件 ⇒ PASS 无内容），
        因此这是门禁的可信度前提，不是可选优化（plan 07 §9-23）。
        """
        ...

    def describe(self) -> str:
        """一行来源说明（落进 gates.txt 上下文）。"""
        ...

    def coverage(self) -> dict[str, float]:
        """本 run 的**数据覆盖**（在线标量的必备项，plan 07 §9-38）。

        为什么它是协议的一部分：2026-09-27 发现 `ManifestBatchSource` 的共享游标在
        **1000 步后彻底饱和**——此后无论 max_steps 加到多少，模型永远只看同一批
        993 个窗口（全库 0.156%）/ 674 张谱面。当时没有任何指标在问「数据走了多少」，
        因此这个静默失效活了整轮。覆盖率是那条缺陷的**可观测性前提**，不是可选项。

        Returns:
            至少含 `epoch` / `windows_seen` / `windows_total` / `charts_seen` /
            `charts_total` 五个键；无有限数据集（合成来源）时 total 记 0。
        """
        ...

    def batches(
        self, *, start_step: int = 1, first: FieldBatch | None = None
    ) -> Iterator[FieldBatch]:
        """**流式**产出批次（训练主循环的唯一取批入口；`batch()` 留给门禁的重抽循环）。

        为什么协议里要有它：把「一批一批拉」和「一次性拉一批」分开之后，真实来源才可以在
        后台**预取**下一批（RFC-0034），而冒烟来源照旧逐批合成。`start_step` / `first`
        的口径见 `ManifestBatchSource.batches`。
        """
        ...


def event_total(batch: FieldBatch) -> float:
    """批内事件总数（`counts` 缺失时视为 0）。"""
    return 0.0 if batch.counts is None else float(batch.counts.sum())


def draw_until_min_events(
    draw: Callable[[], FieldBatch],
    *,
    min_events: int,
    attempts: int,
) -> FieldBatch:
    """重复 `draw()` 直到批内事件数 >= `min_events`（最多 `attempts` 次）。

    **为什么必须有**：门禁判据在空批上会**空过**——真实 200 行切片上 G1 抽到的窗口实测
    K=2 / 0 事件，`[PASS] G1 loss 7819.5 -> 0.0014` 其实没有任何事件可过拟合（plan 07 §9-23）。
    取不到就抛（fail-closed），不静默降级成「空批也算过」。

    Raises:
        ValueError: `attempts` 次都取不到。
    """
    if min_events <= 0:
        return draw()
    seen = 0.0
    for _ in range(max(1, int(attempts))):
        batch = draw()
        seen = event_total(batch)
        if seen >= min_events:
            return batch
    raise ValueError(
        f"连续 {max(1, int(attempts))} 次取批都取不到 {min_events} 个事件（末次 {seen}）："
        "门禁需要非空批；检查 split / 清单 / 特征缓存",
    )


def betas_of(cfg: TrainConfig) -> tuple[float, float]:
    """AdamW 的 betas（把配置里的二元组显式转成 float 二元组，避免 mypy 的类型漂移）。"""
    first, second = cfg.optim.betas
    return float(first), float(second)


def model_from_config(
    cfg: TrainConfig,
    grid: FieldGrid,
    *,
    seed: int = 0,
    initial_head_bias: float = 0.0,
) -> MaskedFieldModel:
    """按配置建模型（固定初始化种子；可选抬高累积强度偏置）。"""
    torch.manual_seed(seed)
    model = MaskedFieldModel(cfg.model.to_model_config(), grid)
    if initial_head_bias:
        set_initial_head_bias(model, initial_head_bias)
    return model


def set_initial_head_bias(model: MaskedFieldModel, value: float) -> None:
    """把累积强度头的 bias 初始化到 `value`（G1/G2 的相对判据需要首步远离最优）。

    Raises:
        AttributeError: 输出头结构不含 `cum_head`（换头之后必须同步这里的口径）。
    """
    head = getattr(model.head, "cum_head", None)
    if head is None:  # pragma: no cover - 结构性改动才会走到
        raise AttributeError("输出头没有 cum_head：initial_head_bias 的口径需要同步更新")
    with torch.no_grad():
        head.bias.fill_(float(value))


def autocast_dtype(precision: str, device: torch.device) -> torch.dtype | None:
    """精度字符串 -> autocast dtype；`None` = 不启用（CPU，或非 bf16 精度）。

    抽成纯函数是为了能在**没有 GPU 的默认 CI** 里断言这个决策（见
    `tests/unit/infra/test_precision.py`）——否则这条路径只有真机才走到。
    """
    if device.type != "cuda" or str(precision) not in {"bf16", "bf16-mixed"}:
        return None
    return torch.bfloat16


def autocast_context(precision: str, device: torch.device) -> AbstractContextManager[None]:
    """按 `optim.precision` 给出 autocast 上下文（plan 07 §9-36）。

    **为什么需要**：`optim.precision` 此前**只被声明、没有人读**——训练一直跑 fp32。
    8 GB 卡上实测（K=32、`t_window=192`）：bf16 把峰值显存 5.05 → 2.92 GiB（0.58x）、
    步时 0.550 → 0.269 s（0.49x），K 上限从 ≈31 抬到 ≈42-47。

    **只在 CUDA 上启用**：CPU 路径保持 fp32，于是默认 CI（CPU）的数值逐位不变。
    `fp16` 不在支持范围内——它需要 GradScaler，静默按 fp32 跑比假装支持更危险，
    因此 `validate_config` 会直接拒掉（fail-closed）。

    ⚠️ bf16 的**首步**要 1.4-3.0 s（一次性内核编译 / autotune），别把首步当稳态。
    """
    dtype = autocast_dtype(precision, device)
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def make_step_fn(
    model: nn.Module,
    batch: FieldBatch,
    optimizer: torch.optim.Optimizer,
    *,
    grad_clip_norm: float | None = None,
    chunks: int = 1,
    precision: str = "fp32",
) -> StepFn:
    """把「前向 + 反传 + 一步优化」包成 `sanity.StepFn`（返回标量 loss）。

    `chunks > 1` 时**分批前向/反传**（plan 07 §9-15 的内存墙）：门禁的 G2/G3 需要
    足够多的样本（样本太少时打乱臂会直接背样本，plan 07 §9-12），而真实窗口的
    激活显存/内存随批大小线性增长 ⇒ 一次性前向会在 16 个样本上直接 OOM。

    等价性口径（**必须成立，否则分批会悄悄改掉门禁判据**）：

    - 本函数依赖损失是**按样本求和**的（`full_poisson_loss` / `masked_poisson_loss`
      的默认 `reduction="sum"`）。此时逐段 `loss.backward()` 累加得到的梯度与一次性
      反传**数值等价**（只差 float32 的求和顺序：实测 loss 相对差 2.3e-8、梯度最大绝对差
      1.5e-5 ≈ 尺度的 1e-7），返回的标量 loss 是全批 loss（**不是**段均值）。
    - 对 `reduction="mean"` 的损失（或任何在批内重算统计量的损失，例如
      `masked_poisson_loss` 的遮盖比例 `r`）分批**不等价**——那种臂必须自己
      保证批内统计口径，本函数不替它决定。

    Args:
        model: 前向模块（`output.loss` 必须是标量）。
        batch: 全批；`chunks > 1` 时按样本维切段。
        optimizer: 优化器。
        grad_clip_norm: 梯度裁剪阈值（在全批梯度累加**之后**裁剪，与不分批一致）。
        chunks: 前向分段数（<= 1 或 >= B 时退化为一段）。
    """
    parts = batch.split_samples(chunks)
    amp = autocast_context(precision, batch.line_mask.device)

    def step() -> float:
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for part in parts:
            with amp:
                output = model(part)
            loss = output.loss
            if loss is None:  # pragma: no cover - forward(compute_loss=True) 保证非 None
                raise RuntimeError("前向没有返回 loss：门禁需要 compute_loss=True")
            loss.backward()
            total += float(loss.detach())
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        return total

    return step


def _release_cuda() -> None:
    """把已释放的显存还给驱动（门禁装配的显存卫生，见 :func:`build_gate_inputs`）。

    为什么需要：8 GB 卡上「4 批 + 4 模型」同时在场会让分配器滑进 **Windows 共享内存**，
    同一个 G3 步从 **0.57 s 放大到 8.8 s（15×）**（2026-09-27 第五轮实测）。
    释放之后显存回到「3 模型 + 3 批」量级；CPU 上本函数是空操作。
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _hidden_event_total(batch: FieldBatch) -> float:
    """被遮盖格子内的事件数（G2 的监督质量诊断量）。"""
    if batch.counts is None or batch.occlusion is None:
        return 0.0
    hidden = batch.occlusion_bool()
    return float(batch.counts.to(dtype=torch.float32).masked_select(hidden).sum())


def constant_baseline_for(batch: FieldBatch) -> float:
    """G3 的常数基线：`λ = N/|Ω|` 的闭式泊松 NLL（**不是** λ ≡ 0）。"""
    if batch.counts is None:
        raise ValueError("常数基线需要 batch.counts")
    n_events = float(batch.counts.to(dtype=torch.float32).sum())
    return float(constant_baseline_nll(n_events, omega(batch.grid, n_lines=batch.n_lines())))


def build_gate_inputs(
    cfg: TrainConfig,
    source: BatchSource,
    *,
    seed: int | None = None,
    device: str | None = None,
) -> tuple[GateInputs, dict[str, float]]:
    """装配 G1-G4 的输入（各臂**同起点**：同一 seed 建模型）。

    Args:
        cfg: 训练配置。
        source: 批次来源。
        seed: 建模型的种子（默认取 `cfg.optim.seed`）。
        device: 门禁跑在哪个设备上（None = 保持在来源给的设备/CPU）。真实窗口在 CPU 上
            一步约 12 s（实测 d_model=256/6 层的真实批次），而判据需要 100 步 x 多臂
            ⇒ **不给设备就等于把门禁的时间预算拉到小时级**；GPU 可用时传 "cuda"。

    Returns:
        (inputs, stats)：stats 里是落盘用的诊断量（事件数 / 基线 / 帧数 / 时长）。
    """
    seed = cfg.optim.seed if seed is None else seed
    gates = cfg.gates
    target_device = None if device is None else torch.device(device)

    def to_device(batch: FieldBatch) -> FieldBatch:
        return batch if target_device is None else batch.to(target_device)

    # 门禁批必须**非空**：空批上的 G1/G2/G3 都是空过（见 `BatchSource.batch` 的说明）。
    min_events = max(0, int(gates.batch_min_events))
    g1_batch = to_device(source.batch(masked=True, min_events=min_events))
    # G2 的样本数显式给足：样本太少时打乱臂可以直接背下样本，对照会退化成空转
    g2_samples = max(1, int(gates.shuffle_samples))
    # G2/G3 的前向按样本维分段（plan 07 §9-15）：真实窗口上 samples=16 一次性前向会 OOM，
    # 分段后内存回到「一段样本」的量级，而梯度与 loss 与不分段一致（按样本求和的损失）。
    g2_chunks = max(1, int(gates.shuffle_chunks))
    # G2 的两臂：**同一批、同一损失、同一起点**，只有「待补全的标签」不同
    # （plan 07 §4.3 字面口径「同模型同输入，真标签 / 打乱标签」）。
    #
    # 为什么走遮盖路径而不是无遮盖：无遮盖时 observed_counts() 把 counts 原样喂回去，
    # **输入就是目标**——此时打乱目标会连输入一起打乱，对照退化成「拟合真场 vs 拟合乱场」，
    # 测不出「输入对目标有没有信息」（那个命题只有把输入钉住才成立）。
    # 遮盖路径下两臂的可见场逐位相同（shuffle_hidden_counts 只在遮盖集合内置换），
    # 于是差异恰好是「上下文能否预测被遮盖事件」——帧率/对齐类 bug 会在此当场现形。
    real_batch = to_device(source.batch(masked=True, samples=g2_samples, min_events=min_events))
    shuffled_batch = to_device(shuffle_hidden_counts(real_batch, seed=seed + 991))
    # G3 的基线是**全事件**口径的闭式常数基线（`N·(1+log(|Ω|/N))`，plan 03 的 `constant_baseline_nll`），
    # 因此 G3 必须在**无遮盖**批上比：遮盖路径的 loss 是「只监督被遮盖事件 + 重标定 1/r」，
    # 与全事件基线不在同一测度上（同 §4.3 的配对纪律，别再犯 G2 那个错）。
    g3_batch = to_device(source.batch(masked=False, samples=g2_samples, min_events=min_events))

    # 门禁优化器的学习率**独立于训练 lr**：门禁的预算只有 steps 步，必须让模型
    # 真的收敛（否则 G2/G3 两臂都停在初始点附近，对照与基线判据都失去意义）。
    gate_lr = cfg.optim.lr if gates.gate_optimizer_lr is None else float(gates.gate_optimizer_lr)

    def optimizer_for(model: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            model.parameters(),
            lr=gate_lr,
            weight_decay=cfg.optim.weight_decay,
            betas=betas_of(cfg),
        )

    def build_model(grid: FieldGrid, *, model_seed: int, head_bias: float) -> MaskedFieldModel:
        model = model_from_config(cfg, grid, seed=model_seed, initial_head_bias=head_bias)
        return model if target_device is None else model.to(target_device)

    # ⚠️ G3 的预计算**先跑、且跑完立即释放**（本轮实测，2026-09-27 第五轮）：
    # 四个批 + 四个模型同时在场时，本机 8 GB 卡会滑进 **Windows 共享内存**
    # （`nvidia-smi` 实测 7.88 / 8.15 GB、功耗从 94 W 掉到 88 W），同一个 G3 步从
    # **0.57 s 放大到 8.8 s（15×）**，整轮门禁从分钟级变成小时级且迟迟不结束。
    # G3 是独立臂（它只要 `model_loss` / 基线 / 帧数 / 时长这四个标量），因此没有理由
    # 和其余三个模型共处；释放后 G1/G2 的显存回到「三个模型 + 三个批」的量级。
    # 门禁是**分钟级静默**的（此前整轮跑完才写第一行日志）⇒ 外部无法区分「慢」与「卡死」。
    # 这三个批的形状（尤其是 K）**就是**门禁耗时的决定量：步时 ∝ (K·T)²，G2/G3 还要 ×16 段。
    logger.info(
        "门禁批：G1 K=%d samples=%d events=%.0f｜G2 K=%d samples=%d events=%.0f（遮盖内 %.0f）"
        "｜G3 K=%d samples=%d events=%.0f；chunks=%d precision=%s",
        g1_batch.n_lines(),
        g1_batch.batch_size(),
        event_total(g1_batch),
        real_batch.n_lines(),
        real_batch.batch_size(),
        event_total(real_batch),
        event_total(shuffled_batch),
        g3_batch.n_lines(),
        g3_batch.batch_size(),
        event_total(g3_batch),
        g2_chunks,
        cfg.optim.precision,
    )
    g3_model = build_model(
        g3_batch.grid, model_seed=seed + 1, head_bias=gates.contrast_initial_head_bias
    )
    g3_step = make_step_fn(
        g3_model,
        g3_batch,
        optimizer_for(g3_model),
        grad_clip_norm=cfg.optim.grad_clip_norm,
        chunks=g2_chunks,
        precision=cfg.optim.precision,
    )
    logger.info(
        "门禁 G3 预计算开始：%d 步 x %d 段（K=%d）——这一项在正常 K 下就要十几分钟",
        max(1, gates.shuffle_steps),
        g2_chunks,
        g3_batch.n_lines(),
    )
    model_loss = g3_step()
    for _ in range(max(1, gates.shuffle_steps) - 1):
        model_loss = g3_step()
    logger.info("门禁 G3 预计算完成：model_loss=%.4f", model_loss)
    g3_events = event_total(g3_batch)
    g3_lines = float(g3_batch.n_lines())
    g3_samples_used = float(g3_batch.batch_size())
    baseline_loss = constant_baseline_for(g3_batch)
    frames = int(g3_batch.audio_emb.shape[1])
    duration = float(g3_batch.grid.total_seconds)
    del g3_step, g3_model, g3_batch
    _release_cuda()

    # ⚠️ 抬高初值只服务 G1 的**相对**判据；G2/G3 是对照/基线口径，必须从模型的自然初始化
    # 出发（否则它们在预算内先花掉一半步数去压一个人为抬高的、与任务无关的积分项）。
    # 见 `GatesConfig.contrast_initial_head_bias` 的实测依据。
    g1_model = build_model(g1_batch.grid, model_seed=seed, head_bias=gates.initial_head_bias)
    # G2 的两臂必须**同批、同损失、同起点**（plan 07 §4.3）：目标只在「真 / 打乱」上不同。
    # 二者都用同一 seed 建模型，因此初始权重逐位相同。
    g2_real_model = build_model(
        real_batch.grid, model_seed=seed, head_bias=gates.contrast_initial_head_bias
    )
    shuffled_model = build_model(
        shuffled_batch.grid, model_seed=seed, head_bias=gates.contrast_initial_head_bias
    )

    inputs = GateInputs(
        step_fn_real=make_step_fn(
            g1_model,
            g1_batch,
            optimizer_for(g1_model),
            grad_clip_norm=cfg.optim.grad_clip_norm,
            precision=cfg.optim.precision,
        ),
        step_fn_g2_real=make_step_fn(
            g2_real_model,
            real_batch,
            optimizer_for(g2_real_model),
            grad_clip_norm=cfg.optim.grad_clip_norm,
            chunks=g2_chunks,
            precision=cfg.optim.precision,
        ),
        step_fn_g2_shuffled=make_step_fn(
            shuffled_model,
            shuffled_batch,
            optimizer_for(shuffled_model),
            grad_clip_norm=cfg.optim.grad_clip_norm,
            chunks=g2_chunks,
            precision=cfg.optim.precision,
        ),
        model_loss=float(model_loss),
        baseline_loss=float(baseline_loss),
        frames=frames,
        duration_s=duration,
        frame_rate=float(real_batch.frame_rate),
    )
    stats = {
        "g1_events": event_total(g1_batch),
        "g1_lines": float(g1_batch.n_lines()),
        "g1_samples": float(g1_batch.batch_size()),
        "g2_events": event_total(real_batch),
        # G2 的监督质量：被遮盖格子里的**事件数**（两臂逐位相同，只有分布不同）。
        "g2_hidden_events": _hidden_event_total(real_batch),
        "g3_events": g3_events,
        "g3_lines": g3_lines,
        "g3_samples": g3_samples_used,
        "gate_min_batch_events": float(min_events),
        # global 层的自注意力长度 L = K * T：成本 ∝ L^2（每样本独立，但 L^2 缓冲区随
        # 分段内样本数线性叠加）。（2026-09-27 第四轮实测：这是真实门禁墙钟的量级来源。）
        "gate_optimizer_lr": float(gate_lr),
        "g1_initial_head_bias": float(gates.initial_head_bias),
        "contrast_initial_head_bias": float(gates.contrast_initial_head_bias),
        "g2_lines": float(real_batch.n_lines()),
        "g2_attn_len": float(real_batch.n_lines() * real_batch.grid.t_bins),
        "field_t_bins": float(real_batch.grid.t_bins),
        # G2 两臂的同批性落进 gates.txt：样本数一旦不等，比的就是「批大小」而不是「信息」。
        "g2_real_samples": float(real_batch.batch_size()),
        "g2_shuffled_samples": float(shuffled_batch.batch_size()),
        "g2_chunks": float(g2_chunks),
        "g3_baseline_loss": float(inputs.baseline_loss),
        "g3_model_loss": float(inputs.model_loss),
        "audio_frames": float(frames),
        "duration_s": duration,
    }
    return inputs, stats


@dataclass(slots=True)
class TrainReport:
    """一次训练的摘要（落 `metrics.json` 与文本日志）。"""

    steps: int
    first_loss: float
    last_loss: float
    best_loss: float
    loss_history: list[tuple[int, float]] = field(default_factory=list)
    checkpoints: list[str] = field(default_factory=list)
    data_source: str = ""
    #: 续训起点（checkpoint 路径与已完成步数）；None = 从头训练。
    resumed_from: str | None = None
    resumed_step: int = 0

    def to_metrics(self) -> dict[str, Any]:
        """metrics.json 的载荷（**不含**权重；plan 07 §3.2）。"""
        return {
            "steps": self.steps,
            "first_loss": self.first_loss,
            "last_loss": self.last_loss,
            "best_loss": self.best_loss,
            "checkpoints": list(self.checkpoints),
            "data_source": self.data_source,
            "resumed_from": self.resumed_from,
            "resumed_step": self.resumed_step,
            "loss_history": [[step, loss] for step, loss in self.loss_history],
        }

    def format(self) -> str:
        """人类可读摘要（贴训练日志用）。"""
        return (
            f"训练摘要：steps={self.steps}，loss {self.first_loss:.6f} -> {self.last_loss:.6f}"
            f"（best {self.best_loss:.6f}），checkpoints={self.checkpoints or '无'}"
        )


def write_scalars(log_dir: Path, history: Sequence[tuple[int, float]]) -> bool:
    """写 TB 标量 `train/loss`（tensorboard 缺失时只告警，不影响训练结论）。"""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        logger.warning("tensorboard 不可用：train/* 标量未写入")
        return False
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    try:
        for step, value in history:
            writer.add_scalar("train/loss", float(value), int(step))
    finally:
        writer.close()
    return True


class ScalarWriter(Protocol):
    """TB SummaryWriter 的最小接口（不在签名里写 Any：ANN401）。"""

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        """写一个标量。"""
        ...

    def flush(self) -> None:
        """把缓冲刷到磁盘（长跑期间人力监控靠它看到最新点）。"""
        ...

    def close(self) -> None:
        """收尾。"""
        ...


def _peak_vram_gib() -> float:
    """当前进程在 CUDA 上的峰值已分配显存（GiB）；CPU 上恒为 0。

    为什么必须进 TB：8 GB 卡上「显存贴顶 ⇒ 驱动滑进共享内存 ⇒ 步时放大一个量级」是
    静默的（利用率仍 100%），唯一的前兆就是这条曲线与 power.draw。
    """
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated()) / 2**30


def _append_history(path: Path, rows: Sequence[Mapping[str, float]]) -> None:
    """把标量行追加进 logs/loss_history.jsonl（追加，因此续训后仍是同一条曲线）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def open_scalar_writer(log_dir: Path) -> ScalarWriter | None:
    """打开 TB SummaryWriter；tensorboard 缺失时返回 None 并告警（不改变训练结论）。

    权威记录仍是 gates.txt / metrics.json；TB 只服务人力在线监控。
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        logger.warning("tensorboard 不可用：TB 标量不写（loss_history.jsonl 仍有完整曲线）")
        return None
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer: ScalarWriter = SummaryWriter(log_dir=str(log_dir))
    return writer


def _flush_scalars(
    writer: ScalarWriter | None,
    history_path: Path,
    rows: Sequence[Mapping[str, float]],
) -> None:
    """把一批标量同时写进 jsonl（权威、健康检查脚本读它）与 TB（人力在线监控）。"""
    _append_history(history_path, rows)
    if writer is None:
        return
    for row in rows:
        step = int(row["step"])
        writer.add_scalar("train/loss", float(row["loss"]), step)
        writer.add_scalar("train/step_time_s", float(row["step_time_s"]), step)
        # 步时拆分（plan 07 §9-42）：data_share 连续攀升 = GPU 在等数据，
        # 这比「吞吐掉下来」早得多地可见。
        if "data_time_s" in row and "compute_time_s" in row:
            data_s = float(row["data_time_s"])
            compute_s = float(row["compute_time_s"])
            writer.add_scalar("train/data_time_s", data_s, step)
            writer.add_scalar("train/compute_time_s", compute_s, step)
            span = data_s + compute_s
            if span > 0.0:
                writer.add_scalar("perf/data_share", data_s / span, step)
        writer.add_scalar("train/lr", float(row["lr"]), step)
        grad = float(row["grad_norm"])
        if math.isfinite(grad):
            writer.add_scalar("train/grad_norm", grad, step)
        writer.add_scalar("sys/peak_vram_gib", float(row["peak_vram_gib"]), step)
        # 数据覆盖（plan 07 §9-38）：没有这三条曲线，「采样器饱和」这类静默失效看不见。
        if "windows_total" in row and float(row["windows_total"]) > 0.0:
            writer.add_scalar("coverage/epoch", float(row["epoch"]), step)
            writer.add_scalar(
                "coverage/windows_seen",
                float(row["windows_seen"]) / float(row["windows_total"]),
                step,
            )
            if float(row["charts_total"]) > 0.0:
                writer.add_scalar(
                    "coverage/charts_seen",
                    float(row["charts_seen"]) / float(row["charts_total"]),
                    step,
                )
    writer.flush()


def _apply_resume(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    *,
    data_rev: str,
    resume_from: Path,
    report: TrainReport,
) -> int:
    """把 checkpoint 的状态装回模型与优化器，返回下一步的步号。

    载入经 load_checkpoint 逐项校验（配置指纹 / 门禁全绿 / data_rev），不一致即抛——
    宁可重跑，也不用「环境已变」的实验续训出不可信结论。
    """
    loaded = load_checkpoint(
        Path(resume_from),
        cfg=cfg,
        data_rev=data_rev,
        require_gates_green=cfg.gates.required,
    )
    model.load_state_dict(dict(loaded.model_state))
    if loaded.optimizer_state is not None:
        try:
            optimizer.load_state_dict(dict(loaded.optimizer_state))
        except (KeyError, ValueError) as exc:
            raise RuntimeError(f"优化器状态无法恢复（{exc}）：拒绝从半个状态续训") from exc
    report.resumed_from = str(resume_from)
    report.resumed_step = loaded.meta.step
    start_step = loaded.meta.step + 1
    logger.info(
        "续训：%s 已完成 %d 步，从第 %d 步跑到 %d",
        resume_from,
        loaded.meta.step,
        start_step,
        cfg.optim.max_steps,
    )
    return start_step


def _save_step(
    artifacts: RunArtifacts,
    cfg: TrainConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    value: float,
    best: float,
    data_rev: str,
    gates_green: bool,
    report: TrainReport,
) -> Path:
    """存一个步级 checkpoint（先写后删），必要时刷新最优快照，然后旋转旧文件。"""
    path = save_checkpoint(
        artifacts.checkpoints / f"step-{step:07d}.pt",
        model=model,
        cfg=cfg,
        step=step,
        data_rev=data_rev,
        git_rev=git_rev(artifacts.root),
        gates_green=gates_green,
        optimizer=optimizer,
    )
    report.checkpoints.append(path.name)
    best_path = artifacts.checkpoints / BEST_CHECKPOINT_NAME
    if value <= best:
        save_checkpoint(
            best_path,
            model=model,
            cfg=cfg,
            step=step,
            data_rev=data_rev,
            git_rev=git_rev(artifacts.root),
            gates_green=gates_green,
        )
    rotate_checkpoints(artifacts.root, keep_last=cfg.run.keep_last, keep_paths=[best_path])
    logger.info("checkpoint @step %d -> %s", step, path.name)
    return path


def train(  # noqa: PLR0915 - 训练循环的语句数靠注释说明更清楚，拆函数会把状态切碎
    cfg: TrainConfig,
    *,
    source: BatchSource,
    artifacts: RunArtifacts,
    data_rev: str,
    gates_green: bool,
    device: str = "cpu",
    resume_from: Path | None = None,
) -> TrainReport:
    """参考训练循环（默认 CPU 可跑；极小配置用于 CI 与冒烟）。

    长跑的四个运维要点（plan 07 §4.5/§4.6，2026-09-27 第六轮）：

    1. 断点续训：resume_from 指向上一次的 step-*.pt；载入前经 load_checkpoint 逐项校验，
       不一致拒绝恢复，优化器状态（AdamW 动量）一并恢复；
    2. 增量落盘：每 run.log_every 步把标量追加进 logs/loss_history.jsonl 并 flush 到 TB
       （train/loss、train/step_time_s、train/data_time_s、train/compute_time_s、
       perf/data_share、train/grad_norm、sys/peak_vram_gib、coverage/*）——长跑期间必须能
       在线看到进度与**瓶颈在哪一侧**，而不是等训练结束才第一次写盘；
    3. 旋转：每次存盘后只保留最新 run.keep_last 个步级 checkpoint（外加 best.pt），先写后删；
    4. 降速可见：步时与峰值显存逐步进 TB/jsonl（判据见 docs/TRAINING.md §7.5）。

    Args:
        cfg: 训练配置。
        source: 批次来源（冒烟或真实清单）。
        artifacts: 实验产物目录（checkpoint 与 TB 标量写到这里）。
        data_rev: 数据版本（写进 checkpoint 元数据，恢复时校验）。
        gates_green: 本次运行的门禁是否全绿（写进 checkpoint 元数据）。
        device: 设备字符串。
        resume_from: 续训起点 checkpoint；None = 从头训练。

    Returns:
        TrainReport。
    """
    report = TrainReport(
        steps=0, first_loss=float("nan"), last_loss=float("nan"), best_loss=float("nan")
    )
    report.data_source = source.describe()
    first_batch = source.batch(masked=True)
    target_device = torch.device(device)
    # 注意：--device cuda 必须同时搬批次（plan 07 §9-22）；只搬模型会让训练路径直接失败。
    model = model_from_config(cfg, first_batch.grid, seed=cfg.optim.seed).to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        betas=betas_of(cfg),
    )
    start_step = 1
    if resume_from is not None:
        start_step = _apply_resume(
            model, optimizer, cfg, data_rev=data_rev, resume_from=resume_from, report=report
        )
    model.train()
    # 取批入口：同步或 DataLoader（见 `ManifestBatchSource.batches`）。探测批在 start_step==1
    # 时就是第一步的批（不白抽一个窗口）；续训时它只用来拿 grid，游标由 start_step 定位。
    stream = source.batches(start_step=start_step, first=first_batch if start_step == 1 else None)
    history: list[tuple[int, float]] = []
    best = float("inf")
    log_every = max(1, int(cfg.run.log_every))
    writer = open_scalar_writer(artifacts.logs)
    pending: list[dict[str, float]] = []
    try:
        for step in range(start_step, cfg.optim.max_steps + 1):
            started = time.perf_counter()
            raw = next(stream)
            # 数据侧到此为止（选桶 → 取窗口 → 解析谱面 → 建场 → collate）。这一段全在
            # **主进程单线程关键路径**上，此刻 GPU 是空的（plan 07 §9-42）。
            drawn = time.perf_counter()
            batch = raw.to(target_device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(cfg.optim.precision, target_device):
                output = model(batch)
            loss = output.loss
            if loss is None:  # pragma: no cover - compute_loss 默认 True
                raise RuntimeError("训练前向没有返回 loss")
            loss.backward()
            grad_norm = float("nan")
            if cfg.optim.grad_clip_norm is not None:
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip_norm)
                )
            optimizer.step()
            # loss.detach() 取标量会**强制同步**：到这里 GPU 上的搬运/前向/反向/裁剪/更新
            # 都已落地，因此下面的 compute 才是真计算时间，而不是 kernel 排队时间。
            value = float(loss.detach())
            computed = time.perf_counter()
            elapsed = computed - started
            data_time = drawn - started
            compute_time = computed - drawn
            history.append((step, value))
            best = min(best, value)
            if step == start_step:
                report.first_loss = value
            coverage = source.coverage()
            pending.append(
                {
                    "step": float(step),
                    "loss": value,
                    "step_time_s": elapsed,
                    # 步时**拆开**（plan 07 §9-42）：GPU 利用率偏低时，必须能区分
                    # 「数据侧供给不足（GPU 在等）」与「计算侧本身慢」。混成一个数只能
                    # 靠功率/利用率反推，而这两者都看不出瓶颈在哪一侧。
                    "data_time_s": data_time,
                    "compute_time_s": compute_time,
                    "grad_norm": grad_norm,
                    "peak_vram_gib": _peak_vram_gib(),
                    "lr": float(cfg.optim.lr),
                    # 批的 K 与事件数：显存墙与「空批」两个老问题都靠它在线可见
                    # （plan 07 §9-23 / §9-35；步时 ∝ K²，K 必须和 loss 一起看）。
                    "batch_n_lines": float(batch.n_lines()),
                    "batch_events": event_total(batch),
                    # 数据覆盖：epoch 内进度与累计「见过多少窗口/谱面」。采样器一旦饱和，
                    # loss 曲线**看不出来**（它只是反复拟合同一小撮样本）——只有覆盖率能。
                    "epoch": coverage["epoch"],
                    "windows_seen": coverage["windows_seen"],
                    "windows_total": coverage["windows_total"],
                    "charts_seen": coverage["charts_seen"],
                    "charts_total": coverage["charts_total"],
                }
            )
            if step % log_every == 0 or step == cfg.optim.max_steps:
                _flush_scalars(writer, artifacts.logs / HISTORY_FILENAME, pending)
                pending.clear()
            if cfg.run.save_every > 0 and step % cfg.run.save_every == 0:
                _save_step(
                    artifacts,
                    cfg,
                    model,
                    optimizer,
                    step=step,
                    value=value,
                    best=best,
                    data_rev=data_rev,
                    gates_green=gates_green,
                    report=report,
                )
    finally:
        if pending:
            _flush_scalars(writer, artifacts.logs / HISTORY_FILENAME, pending)
        if writer is not None:
            writer.close()
    report.steps = max(0, cfg.optim.max_steps - start_step + 1)
    report.last_loss = history[-1][1] if history else float("nan")
    report.best_loss = best if history else float("nan")
    report.loss_history = history
    logger.info("%s", report.format())
    return report


@dataclass(slots=True)
class ManifestBatchSource:
    """真实清单批次来源（包 `beatmorph.data.dataset`；**延迟导入**避免默认 CI 被拖慢）。

    说明：`FieldBatch` 只携带一个 `FieldGrid`，因此同一批样本必须网格同身份
    （`x_bins` / `t_window` / `bpm_eff` 三者相同，见 `ChartPairDataset.grid_key`），
    该约束由 `collate_field_batch` 强制（不一致即抛）。**因此本模块按网格身份分桶组批**，
    而不是连续取 index——真实语料里连续窗口很容易跨越 BPM 变更点或行边界，那时 J 会静默错掉。
    """

    cfg: TrainConfig
    split: str = "train"
    seed: int = 0
    _dataset: ChartPairDataset | None = None
    #: 当前 epoch 的取批计划（**纯函数产物**）：顺序与覆盖率全部由它回答，本对象不记账。
    _plan: WindowPlan | None = None
    #: 当前 plan 实际生效的块长（请求更大时按需重建，见 `plan`）。
    _plan_chunk: int = 0
    #: 本 epoch 已消费的槽位数。**两条路径（同步 / DataLoader）共用同一口径**
    #: ⇒ `coverage()` 与 `data.workers` 无关。
    _cursor: int = 0
    #: 当前 epoch 序号（0 起；走完全部窗口后 +1）。
    _epoch: int = 0

    def _ensure(self) -> ChartPairDataset:
        if self._dataset is None:
            from beatmorph.data.dataset import ChartPairDataset, DatasetConfig

            data = self.cfg.data
            self._dataset = ChartPairDataset(
                DatasetConfig(
                    manifest_path=Path(data.manifest_path),
                    chart_dir=Path(data.chart_dir),
                    feature_dir=Path(data.feature_dir),
                    split=self.split,
                    t_window=data.t_window,
                    tau_end_s=data.tau_end_s,
                    tau_end_policy=data.tau_end_policy,
                    x_bins=data.x_bins,
                    k_max=data.k_max,
                    occlusion_ratio=data.occlusion_ratio,
                    seed=self.seed,
                    limit=data.max_samples,
                    chart_cache_size=data.chart_cache_size,
                    feature_cache_size=data.feature_cache_size,
                ),
            )
        return self._dataset

    def plan(self, *, chunk: int | None = None) -> WindowPlan:
        """当前 epoch 的取批计划（惰性构建；`chunk` 变大时重建并从头开始）。

        为什么把顺序搬进一个**不可变对象**：修复前顺序由 `self._positions` 决定、覆盖率在
        取批时**就地累加**，于是 worker 各持副本既不可复现、覆盖率还会静默失真（RFC-0033
        的那类失效）。现在两者都是 `(seed, epoch, 数据集索引)` 的纯函数
        ⇒ `data.workers` 取任何值都给出同一顺序、同一串覆盖率数字（RFC-0034 §5）。
        """
        need = max(int(chunk) if chunk is not None else self.cfg.optim.batch_size, 1)
        if self._plan is None or need > self._plan_chunk:
            from beatmorph.data.plan import plan_epoch

            self._plan = plan_epoch(
                self._ensure(),
                seed=self.seed,
                epoch=self._epoch,
                chunk=need,
                batch_size=self.cfg.optim.batch_size,
            )
            self._plan_chunk = self._plan.chunk
            # 重建会换一套顺序 ⇒ 游标必须回到 epoch 开头（否则会跳过或重复窗口）。
            self._cursor = 0
        return self._plan

    #: 为满足 `min_events` 而重抽的上限（超过即抛，不无限重试）。
    max_event_draws: int = 64

    def batch(
        self,
        *,
        masked: bool,
        shuffled: bool = False,
        samples: int | None = None,
        min_events: int = 0,
    ) -> FieldBatch:
        """取若干**同网格身份**的窗口并 collate（`shuffled` 时置换目标）。

        `min_events > 0`：逐次重抽直到批内事件数达标（最多 :attr:`max_event_draws` 次）。
        **为什么必须**：真实切片上第一个桶的第一个窗口实测 0 事件，门禁因此空过
        （plan 07 §9-23；G1 报了 0.0014 的「过拟合」却没有任何事件可过拟合）。

        Raises:
            ValueError: 清单里没有可用样本，或连续重抽都取不到 `min_events` 个事件。
            GridMismatchError: 分桶逻辑失效（不应发生；由 collate 兜底）。
        """
        return draw_until_min_events(
            lambda: self._draw(masked=masked, shuffled=shuffled, samples=samples),
            min_events=min_events,
            attempts=self.max_event_draws,
        )

    def _start_next_epoch(self) -> None:
        """一个 epoch 抽完（每个窗口**恰好**抽过一次）：进入下一轮并重建计划。"""
        self._epoch += 1
        self._plan = None
        self._plan_chunk = 0
        self._cursor = 0

    def _consume(self, *, size: int) -> list[int]:
        """从计划里取**同一网格身份**的至多 `size` 个窗口下标（epoch 走完自动翻页）。"""
        plan = self.plan(chunk=size)
        if self._cursor >= plan.n_windows:
            self._start_next_epoch()
            plan = self.plan(chunk=size)
        start = self._cursor
        stop = min(start + size, int(plan.run_end[start]))
        self._cursor = stop
        return [int(value) for value in plan.order[start:stop]]

    def coverage(self) -> dict[str, float]:
        """本 run 的数据覆盖（口径见 `BatchSource.coverage`）。

        **完全由计划前缀回答**，不做任何就地累加 ⇒ 与 worker 数无关（RFC-0034 §5）。
        """
        plan = self._plan
        if plan is None:
            return {
                "epoch": 0.0,
                "windows_seen": 0.0,
                "windows_total": 0.0,
                "charts_seen": 0.0,
                "charts_total": 0.0,
            }
        seen, charts = plan.coverage_at(self._cursor)
        total = float(plan.n_windows)
        return {
            "epoch": float(self._epoch) + (seen / total if total else 0.0),
            "windows_seen": seen,
            "windows_total": total,
            "charts_seen": charts,
            "charts_total": float(plan.charts_total),
        }

    def batches(
        self, *, start_step: int = 1, first: FieldBatch | None = None
    ) -> Iterator[FieldBatch]:
        """按计划顺序**流式**产出批次（训练主循环的唯一取批入口）。

        `data.workers > 0` 时走 `torch.utils.data.DataLoader`（样本构造在 worker 进程里
        并行，与 GPU 计算重叠）；否则走同步路径。**两条路径消费同一份 plan、同一 `_cursor`
        口径** ⇒ 样本顺序、覆盖率、续训定位逐位一致（`data.workers` 因此是语义无关字段，
        见 RFC-0034 §5）。

        Args:
            start_step: 从第几步开始（1 起）。`first is None` 且 `start_step > 1` 时把游标
                直接定位到 `start_step - 1`（续训 O(1)，不重放前面的批次）。
            first: 调用方已经先取过一批（建模型要 `FieldBatch.grid`）时交回来复用；
                仅当 `start_step == 1` 时允许——续训必须从计划里定位，不能白抽一个窗口。

        Raises:
            ValueError: `first` 与 `start_step > 1` 同时给出。
        """
        if first is not None and start_step != 1:
            raise ValueError(f"first 只在 start_step == 1 时可用（得到 start_step={start_step}）")
        workers = max(0, int(self.cfg.data.workers))
        if workers == 0:
            yield from self._batches_sync(start_step=start_step, first=first)
        else:
            yield from self._batches_parallel(start_step=start_step, first=first, workers=workers)

    def _seek(self, consumed: int) -> None:
        """把游标定位到本 epoch 的第 `consumed` 个槽位（续训 O(1)）。"""
        plan = self.plan()
        if consumed >= plan.n_windows:  # pragma: no cover - 续训步数不会超过一个 epoch
            raise ValueError(f"续训定位超出本 epoch（consumed={consumed} >= {plan.n_windows}）")
        self._cursor = max(0, int(consumed))

    def _batches_sync(self, *, start_step: int, first: FieldBatch | None) -> Iterator[FieldBatch]:
        """同步路径（workers=0；门禁、冒烟与默认 CI 走它，行为与重构前一致）。"""
        if first is None and start_step > 1:
            self._seek(start_step - 1)
        if first is not None:
            yield first
        while True:
            yield self._draw(masked=True)

    def _batches_parallel(
        self, *, start_step: int, first: FieldBatch | None, workers: int
    ) -> Iterator[FieldBatch]:
        """DataLoader 路径：worker 只做**纯函数**的样本构造；顺序与记账都在主进程。

        显存纪律（RFC-0034 §4）：worker **不建 CUDA context**（只在 CPU 上构造样本），
        预取队列只落在**主机内存**（`prefetch_factor=2` ⇒ 在飞样本数有界），
        GPU 上同时只有一个 batch（H2D 与计算仍由主循环串行驱动）。
        """
        from torch.utils.data import DataLoader

        from beatmorph.data.dataset import collate_field_batch
        from beatmorph.data.plan import PlanBatchSampler

        batch_size = max(1, int(self.cfg.optim.batch_size))
        if workers > 1 and not os.environ.get("OMP_NUM_THREADS"):
            # 不在这里**替**调用方设环境变量：本进程的 BLAS 线程池早已初始化，改 os.environ
            # 只对之后 spawn 的子进程有效，属于「看起来生效、实际半生效」的写法。
            # 因此改为显式告警——过度订阅会让并行取批比串行更慢（RFC-0034 §4）。
            logger.warning(
                "data.workers=%d 但未设 OMP_NUM_THREADS：每个 worker 的 BLAS 可能各开满核"
                "（本机 24 逻辑核，串行时单进程已用约 6.45 核）⇒ 建议 "
                "OMP_NUM_THREADS=2 MKL_NUM_THREADS=2，否则并行取批可能比串行更慢。",
                workers,
            )
        dataset = self._ensure()
        step = int(start_step)
        if first is not None:
            # 探测批已在主进程取走 ⇒ 计划槽位 0 已消费，本轮从槽位 1 所在的批次继续。
            self._cursor = 1
            yield first
            step += 1
        else:
            self._cursor = step - 1
        plan = self.plan(chunk=batch_size)
        sampler = PlanBatchSampler(plan, batch_size=batch_size, start_step=step)
        # 批次切分在**主进程**完成（只有 dataset 与 collate_fn 进 worker）。槽位终点由
        # batch_sampler 自己上报，**不**从 collate 出来的批次反推——记账不依赖物化。
        pending: deque[int] = deque()
        order = plan.order

        def _indices() -> Iterator[list[int]]:
            for start, stop in sampler.spans():
                pending.append(stop)
                yield [int(value) for value in order[start:stop]]

        loader = DataLoader(
            dataset,
            batch_sampler=_indices(),
            collate_fn=collate_field_batch,
            num_workers=workers,
            persistent_workers=True,
            prefetch_factor=2,
            pin_memory=torch.cuda.is_available(),
        )
        # 生成器被提前关闭时由 GC 回收 loader（DataLoader.__del__ 会收掉 worker 进程）；
        # 长跑里 generator 与训练同寿命，不必手动收尾。
        for batch in loader:
            self._cursor = pending.popleft()
            yield batch

    def _draw(
        self,
        *,
        masked: bool,
        shuffled: bool = False,
        samples: int | None = None,
    ) -> FieldBatch:
        """取一批（`batch` 的重抽循环用它）。

        **一个 epoch = 走遍全库、每个窗口恰好抽一次**：顺序由 `beatmorph.data.plan` 一次性
        物化（桶内按谱面分层轮转发牌、桶间按窗口数成比例交错）。修复前的实现用**一个全局
        共享游标** `(start + take) % len(bucket)`，而桶长最小为 1 ⇒ 碰到就被清零、
        **1000 步后饱和在 993 个窗口（全库 0.156%）/ 674 张谱面（10%）**
        （2026-09-27 实测，见 [RFC-0033](../../docs/decisions/RFC-0033-sampler-coverage-and-epoch.md)）。

        Raises:
            ValueError: 清单里没有可用样本（split / max_samples / 特征缓存 全空）。
        """
        from beatmorph.data.dataset import collate_field_batch

        dataset = self._ensure()
        size = max(1, int(self.cfg.optim.batch_size if samples is None else samples))
        indices = self._consume(size=size)
        if not indices:  # pragma: no cover - 新 epoch 必然有窗口可抽
            raise ValueError("索引里有 0 个窗口：无法取批")
        windows = [dataset[index] for index in indices]
        batch = collate_field_batch(windows)
        if not masked:
            batch = _drop_occlusion(batch)
        if shuffled:
            batch = _shuffle_targets(batch, seed=self.seed + 991)
        return batch

    def describe(self) -> str:
        """一行来源说明（含清单路径与切分）。"""
        buckets = "" if self._plan is None else f"，网格桶={int(self._plan.bucket_id.max()) + 1}"
        return (
            f"manifest({self.split})：{self.cfg.data.manifest_path}"
            f"（max_samples={self.cfg.data.max_samples}，t_window={self.cfg.data.t_window}{buckets}）"
        )


def _drop_occlusion(batch: FieldBatch) -> FieldBatch:
    """去掉遮盖（G2/G3 的无遮盖口径）。"""
    from dataclasses import replace

    return replace(batch, occlusion=None)


def _shuffle_targets(batch: FieldBatch, *, seed: int) -> FieldBatch:
    """置换目标计数（G2 的对照臂；事件数不变、输入与目标的对应被破坏）。"""
    from dataclasses import replace

    if batch.counts is None:
        raise ValueError("打乱对照需要 batch.counts")
    shuffled = shuffled_counts(batch.counts, seed=seed)
    return replace(batch, counts=shuffled, occlusion=None)
