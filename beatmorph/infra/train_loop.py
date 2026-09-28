"""参考训练循环（torch 后端）与门禁输入装配（plan 07 §4.1 / §4.3 / M7.2）。

分工：

- `beatmorph/infra/sanity.py`：**判据**（范式中立，只吃 step_fn）；
- 本模块：把「模型 + 批次 + 优化器」接成 `step_fn`，装配 G1-G4 的输入，并提供训练循环；
- `beatmorph/infra/gates.py`：落盘与 fail-closed；
- `beatmorph/infra/lightning_module.py`：plan 07 §4.1 的目标训练栈（可选依赖）。

为什么默认后端是 torch 而不是 Lightning：`pytorch-lightning` 属 `train` extra，而
**门禁必须在默认 CI 里能跑**（CLAUDE.md §4：契约级与门禁级断言不得依赖权重/GPU/可选依赖）。
Lightning 后端仍然提供，并在缺失时给出可操作的报错（而不是静默降级）。

置换对照（RFC-0037：门禁 G2 已删除）：`shuffle_counts_within_line`（在 `infra/smoke.py`）
给 val 的 `nll_shuffled` 对照用——在**每条判定线各自的被遮盖集合内**置换计数：
可见场逐位不变、每线事件数不变（修掉旧全局置换连「每线事件预算」一起改的
 nuisance，RFC-0036 §2.5 实证）。
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
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

import torch
from torch import nn

from beatmorph.core.logging import get_logger
from beatmorph.eval.val_metrics import (
    ReweightMode,
    ValAccumulator,
    full_event_nll,
    masked_nll,
    masked_readout,
)
from beatmorph.field.grid import FieldGrid
from beatmorph.field.integrate import omega
from beatmorph.field.loss import constant_baseline_nll
from beatmorph.generation.batch import FieldBatch, FieldOutput
from beatmorph.generation.losses import (
    event_normalizer,
    integral_term,
    masked_poisson_loss,
)
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
from beatmorph.infra.smoke import shuffle_counts_within_line

__all__ = [
    "SCALAR_TAGS",
    "BatchSource",
    "ManifestBatchSource",
    "ManifestValSource",
    "TrainReport",
    "ValBatchSource",
    "autocast_context",
    "autocast_dtype",
    "betas_of",
    "build_gate_inputs",
    "condition_contrasts",
    "constant_baseline_for",
    "draw_until_min_events",
    "evaluate_val",
    "event_total",
    "make_step_fn",
    "model_from_config",
    "set_initial_head_bias",
    "stratified_step_loss",
    "train",
    "write_scalars",
]

if TYPE_CHECKING:
    from beatmorph.data.dataset import ChartPairDataset
    from beatmorph.data.plan import WindowPlan

logger = get_logger("infra.train_loop")

#: 过程标量的落盘文件名（jsonl；健康检查脚本与 TB 都读它）。
HISTORY_FILENAME: str = "loss_history.jsonl"

#: jsonl 键 -> TB 标签的**唯一**映射（val 指标 / 分层损失 / 条件对照臂 / 积分比）。
#:
#: 为什么集中在一处：这两条落盘路径（jsonl 是权威、TB 服务人力监控）一旦各写一份名字，
#: 就会漂成两套口径——而「看的曲线不是记的数」是本项目已经付过代价的那类失效。
#: 键的**产出方**是 `ValAccumulator.scalars()`（`val_*` / `cond_*`）与
#: :func:`stratified_step_loss`（`loss_empty` 等）；`tests/unit/infra/test_val_path.py`
#: 断言产出方给出的每个键都在本表里（漂移即测试失败）。
SCALAR_TAGS: Mapping[str, str] = {
    # ── 训练侧分层（plan 07 §9-46 / §9-47 I；RFC-0037 R3 增补归一化标量）────────
    "integral_per_event": "train/integral_per_event",
    "loss_empty": "train/loss_empty",
    "loss_nonempty": "train/loss_nonempty",
    "loss_nonempty_per_event": "train/loss_nonempty_per_event",
    "loss_sum_raw": "train/loss_sum_raw",
    "clip_active": "train/clip_active",
    "empty_share": "train/empty_share",
    "nonempty_share": "train/nonempty_share",
    # ── val（plan 07 §9-47 C/D）────────────────────────────────────────────
    "val_time_s": "val/time_s",
    "val_windows": "val/windows",
    "val_lines": "val/lines",
    "val_nll": "val/nll",
    "val_nll_constant": "val/nll_masked_constant",
    "val_ratio": "val/ratio",
    "val_nll_empty": "val/nll_empty",
    "val_nll_nonempty": "val/nll_nonempty",
    "val_integral": "val/integral",
    "val_events": "val/events",
    "val_pred_over_true": "val/pred_over_true",
    "val_empty_share": "val/empty_share",
    "val_r0_share": "val/r0_share",
    "val_nll_shuffled": "val/nll_shuffled",
    "val_nll_shuffled_delta": "val/nll_shuffled_delta",
    "val_nll_full_event": "val/nll_full_event",
    # ── 条件干预三元组（plan 07 §9-46 ③：唯一能回答「音频条件有没有被用上」的手段）──
    "cond_audio_zero_delta": "cond/audio_zero_delta",
    "cond_audio_perm_delta": "cond/audio_perm_delta",
    "cond_track_zero_delta": "cond/track_zero_delta",
}

#: val 批内**线内置换**标签的种子偏移（RFC-0037 R5；沿用被删 G2 的常量便于历史对照）。
VAL_SHUFFLE_SEED_OFFSET: int = 991


@dataclass(frozen=True, slots=True)
class SlotTag:
    """一个批次的**计划槽位终点**标签（随批穿过 DataLoader 的 worker 边界）。

    为什么需要它：`DataLoader(in_order=False)` 是修掉队头阻塞的唯一开关（见
    :meth:`ManifestBatchSource._batches_parallel`），但它同时让**交付顺序变成任意排列**
    ——而 `_cursor` 是「计划前缀长度」这一个标量（覆盖率 / 续训定位 / `data.workers`
    语义中性全由它保证）。没有标签就只能假设按序交付，而那正是要拆掉的假设。
    """

    stop: int


class _SlotTaggedDataset(torch.utils.data.Dataset[object]):
    """把 `batch_sampler` 挂在批首的**槽位令牌**翻译出来，其余下标原样转发。

    `_indices()` 产出的每个 index 列表形如 `[-stop-1, 窗口下标...]`：首元素用**负编码**
    表示本批的槽位终点（`-stop-1` 与合法的非负窗口下标不可能冲突），本类把它翻译成
    :class:`SlotTag`；其余元素直接落到真正的数据集上。

    为什么要绕这一圈：`DataLoader` 的 worker **只**能把 `dataset[...]` 的返回值带回
    主进程，`batch_sampler` 本身不随批次返回（torch 的 `_task_info` 私有且不暴露下标）。
    把槽位挂在样本列表首位，是让「批 ↔ 槽位」穿过进程边界的最短路径。

    Note:
        内层数据集的 `__getstate__` / `__setstate__`（`ChartPairDataset` 靠它剥掉 worker
        不该拿的派生状态）由 pickle **递归**调用，本类不需要转发。
    """

    def __init__(self, inner: torch.utils.data.Dataset[Any]) -> None:
        self._inner = inner

    def __len__(self) -> int:
        return len(self._inner)  # type: ignore[arg-type]

    def __getitem__(self, token: int) -> object:
        if token < 0:
            return SlotTag(stop=-token - 1)
        return self._inner[token]


def _collate_with_slot(
    base_collate: Callable[[Sequence[Any]], Any], items: Sequence[Any]
) -> tuple[int, Any]:
    """摘下批首的 :class:`SlotTag`，把其余样本交给真正的 collate。

    为什么用 `functools.partial` 而不是闭包：`collate_fn` 要被子进程按「模块 + 限定名」
    **重新导入**（Windows 上 `DataLoader` 走 spawn），闭包不可 pickle；`partial` 会把
    `base_collate` 也按引用 pickle，因此测试里 monkeypatch 的 stub collate 同样能在
    worker 进程里被正确还原（`tests/unit/infra/test_plan_batches.py`）。
    """
    if not items or not isinstance(items[0], SlotTag):
        raise TypeError(
            "取批路径缺少 SlotTag：batch_sampler 与 _SlotTaggedDataset 不配套"
            f"（首个元素是 {type(items[0]).__name__ if items else '空序列'}）"
        )
    return int(items[0].stop), base_collate(items[1:])


class BatchSource(Protocol):
    """批次来源协议：冒烟（`infra.smoke`）与真实清单（`ManifestBatchSource`）都满足它。"""

    def batch(
        self,
        *,
        masked: bool,
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


class ValBatchSource(Protocol):
    """验证集来源协议：**固定的一批窗口**，每次调用重放出逐位一致的内容。

    与 :class:`BatchSource` 分开的理由：训练来源是**无限流**、带覆盖率记账、每步不同；
    验证集是**有限的、固定的、可重放**的一批（plan 07 §9-47 A）。把两者混在一个协议里，
    「val 不得变成训练信号」（§9-47 G-④）就会退化成一句口号——val 来源**没有**任何写入口。
    """

    def batches(self) -> Iterator[FieldBatch]:
        """按计划顺序产出验证批（每次调用**重放同一批**：同窗口、同遮盖、逐位一致）。"""
        ...

    def describe(self) -> str:
        """一行来源说明（落进训练日志，交代 val 集是什么）。"""
        ...

    def windows(self) -> int:
        """验证集的窗口数（成本算式里的 N；见 `OptimConfig.val_windows`）。"""
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
    """把累积强度头的 bias 初始化到 `value`（G1 的相对判据需要首步远离最优）。

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

    `chunks > 1` 时**分批前向/反传**（plan 07 §9-15 的内存墙）：门禁的 G3 需要
    足够多的样本（样本太少时打乱臂会直接背样本，plan 07 §9-12），而真实窗口的
    激活显存/内存随批大小线性增长 ⇒ 一次性前向会在 16 个样本上直接 OOM。

    等价性口径（**必须成立，否则分批会悄悄改掉门禁判据**）：

    - 训练/门禁损失是 `reduction="per_event"`（RFC-0037 R2）：整式除以**批级**除子
      `D = max(E_total, 1)`。分段前向时各段只见到自己的 `D_part`，直接累加会按
      `1/D_part` 加权 ≠ `1/D_full` ⇒ 本函数对每段 loss 乘**修正因子** `D_part/D_full`
      （恒等式：`part_sum/D_part × D_part/D_full = part_sum/D_full`）。修正后逐段
      `loss.backward()` 累加的梯度与标量与不分段**一致**（只差 float 求和顺序；
      旧 sum 口径时代实测 loss 相对差 2.3e-8、梯度最大绝对差 1.5e-5），返回的标量
      是全批 per_event loss（**不是**段均值）。
    - 对 `reduction="mean"` 的损失（或任何在批内重算统计量的损失）分批**不等价**——
      那种臂必须自己保证批内统计口径，本函数不替它决定。

    Args:
        model: 前向模块（`output.loss` 必须是标量）。
        batch: 全批；`chunks > 1` 时按样本维切段。
        optimizer: 优化器。
        grad_clip_norm: 梯度裁剪阈值（在全批梯度累加**之后**裁剪，与不分批一致）。
        chunks: 前向分段数（<= 1 或 >= B 时退化为一段）。
    """
    parts = batch.split_samples(chunks)
    amp = autocast_context(precision, batch.line_mask.device)
    # per_event 的分段修正（见 docstring）：loss_part × D_part/D_full == part_sum/D_full。
    full_norm = event_normalizer(batch)
    part_norms = [event_normalizer(part) for part in parts]

    def step() -> float:
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for part, part_norm in zip(parts, part_norms, strict=True):
            with amp:
                output = model(part)
            loss = output.loss
            if loss is None:  # pragma: no cover - forward(compute_loss=True) 保证非 None
                raise RuntimeError("前向没有返回 loss：门禁需要 compute_loss=True")
            if part_norm != full_norm:
                loss = loss * (part_norm / full_norm)
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


def constant_baseline_for(batch: FieldBatch) -> float:
    """G3 的常数基线：`λ = N/|Ω|` 的闭式泊松 NLL（**不是** λ ≡ 0）。

    RFC-0037 §2.3：模型臂经 forward 的 per_event 归一化，基线**同除一个除子**
    `D = max(E_total, 1)` ⇒ 判据不等式 `model <= (1-x)·baseline` 与归一化前逐位等价。
    """
    if batch.counts is None:
        raise ValueError("常数基线需要 batch.counts")
    n_events = float(batch.counts.to(dtype=torch.float32).sum())
    baseline = float(constant_baseline_nll(n_events, omega(batch.grid, n_lines=batch.n_lines())))
    return baseline / event_normalizer(batch)


#: 条件干预臂的名字（落进 `cond_*_delta`；§9-46 ③ 的三元组，逐字对应）
CONDITION_ARMS: tuple[str, ...] = ("audio_zero", "audio_perm", "track_zero")


def condition_contrasts(batch: FieldBatch, *, seed: int) -> list[tuple[str, FieldBatch]]:
    """条件干预三元组：**同批、同权重**的前向差分（plan 07 §9-46 ③）。

    为什么必须有它：TB 里此前的**全部**比较都是「条件在场 vs 条件在场」，因此
    「音频条件有没有被用上」在这个项目里从来没有过证据（§9-46 的结论）。三个臂分别是：

    - `audio_zero`：把音频特征置零（模型只能靠事件轨与遮盖几何）；
    - `audio_perm`：把音频**时间轴**置换（保留边缘分布、破坏时间对齐）——
      它比置零更狠：置零可以被「没有音频」这一分布外输入掩盖，置换则始终在分布内；
    - `track_zero`：把判定线事件轨置零（模型只能靠音频）。

    三者都是**同一批窗口**上的 `no_grad` 前向，Δ = NLL(干预) - NLL(基线)：
    Δ > 0 表示该条件**降低了**损失（被用上了）；Δ ≈ 0 表示模型没在看它。

    Note:
        置换只动**帧轴**（`dim=1`），τ 轴与目标一格不动 ⇒ 差分只反映条件的作用
        （秒/τ 换算不在此处实现，红线 7）。
    """
    from dataclasses import replace

    zero_audio = replace(batch, audio_emb=torch.zeros_like(batch.audio_emb))
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(int(batch.audio_emb.shape[1]), generator=generator)
    permuted = replace(batch, audio_emb=batch.audio_emb[:, order, :].contiguous())
    extended = None if batch.extended_tracks is None else torch.zeros_like(batch.extended_tracks)
    zero_tracks = replace(
        batch,
        line_tracks=torch.zeros_like(batch.line_tracks),
        extended_tracks=extended,
    )
    return list(zip(CONDITION_ARMS, (zero_audio, permuted, zero_tracks), strict=True))


def stratified_step_loss(output: FieldOutput, batch: FieldBatch) -> dict[str, float]:
    """把**一步的损失**分到「空窗 / 非空窗」两个总体（plan 07 §9-47 B2）。

    为什么这是**量具修复**而不是目标函数修改：loss 是双峰而不是一条曲线——
    43.8% 的空窗中位 **6.5e-4**、56.2% 的非空窗中位 **96**，差五个数量级，而空窗的
    loss **恒等于积分项** ∫λdV（§9-46 实测）。一个把两个总体混在一起的单步数字
    不携带趋势信息，`best.pt` 因此选中过 step 6234 那个 `events=0, K=1` 的空窗。

    本函数**只读**：它调用的是训练损失**本身**（`reduction="none"`，逐窗口求和后按桶取均值），
    不改变任何梯度路径（CLAUDE.md §3.1：`masked_poisson_loss` 的语义一个字都不许动）。
    空桶的键**不出现**（缺失 != 0）：`batch_size=1` 时每一步只有一支有值。

    Returns:
        形如 `{"loss_empty": ..., "empty_share": ...}` 的标量（键见 :data:`SCALAR_TAGS`）。
    """
    if batch.counts is None:
        return {}
    per_window = masked_poisson_loss(output, batch, reduction="none").sum(dim=1)
    events = batch.counts.to(dtype=torch.float32).sum(dim=(1, 2, 3, 4, 5))
    empty = events <= 0.0
    windows = max(int(batch.batch_size()), 1)
    out: dict[str, float] = {
        "empty_share": float(empty.sum().item()) / windows,
        "nonempty_share": float((~empty).sum().item()) / windows,
    }
    if bool(empty.any()):
        out["loss_empty"] = float(per_window[empty].mean().item())
    if bool((~empty).any()):
        out["loss_nonempty"] = float(per_window[~empty].mean().item())
        # RFC-0037 R3：非空窗的**每事件**归一损失（跨步直接可比，主趋势曲线）。
        out["loss_nonempty_per_event"] = float((per_window[~empty] / events[~empty]).mean().item())
    return out


def evaluate_val(
    model: nn.Module,
    val_source: ValBatchSource,
    *,
    device: torch.device,
    precision: str = "fp32",
    reweight: ReweightMode = "hidden",
    seed: int = 0,
    contrast: bool = True,
) -> tuple[dict[str, float], float | None]:
    """跑一遍固定验证集，返回 (落盘标量, `val/ratio`)。

    设计要求（每一条都对应 §9-47 的一条）：

    - **`model.eval()` + `torch.no_grad()`**（A/②）：`dropout=0.1` 必须关掉，否则同一批
      两次读数不同，「跨步可比」当场失效；结束后**无条件**恢复原来的 train/eval 状态。
    - **同一批、同一权重**（D）：模型臂与全部对照臂都在**同一批**上跑；对照臂只改
      **输入**（条件干预）或**标签**（遮盖内置换），不碰权重、不碰优化器。
    - **不在训练图上**：`no_grad` + `compute_loss=False` ⇒ 没有梯度、没有优化器状态改动，
      val 因此**不可能**变成训练信号（G-④）。
    - **不缓存张量**：批由 `val_source` 每次重放（同窗口、同遮盖种子 ⇒ 逐位一致），
      内存占用与训练同量级，不需要把 128 个窗口的场常驻在内存里。

    Returns:
        (标量字典, ratio)。ratio 为 None 表示本批没有可比的常数基线（例如全空窗）。
    """
    accumulator = ValAccumulator()
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        with torch.no_grad():
            for raw in val_source.batches():
                batch = raw.to(device)
                with autocast_context(precision, device):
                    output = model(batch, compute_loss=False)
                readout = masked_readout(batch, output.lam, reweight=reweight)
                shuffled_nll: float | None = None
                if batch.counts is not None and batch.occlusion is not None:
                    # RFC-0037 R5：线内置换（每线事件数不变；旧全局置换的 nuisance 见 RFC-0036 §2.5）。
                    shuffled = shuffle_counts_within_line(
                        batch, seed=int(seed) + VAL_SHUFFLE_SEED_OFFSET
                    )
                    with autocast_context(precision, device):
                        shuffled_output = model(shuffled, compute_loss=False)
                    shuffled_nll = masked_nll(shuffled, shuffled_output.lam, reweight=reweight)
                contrasts: dict[str, float] | None = None
                if contrast:
                    contrasts = {}
                    for name, perturbed in condition_contrasts(batch, seed=int(seed)):
                        with autocast_context(precision, device):
                            perturbed_output = model(perturbed, compute_loss=False)
                        contrasts[name] = masked_nll(
                            perturbed, perturbed_output.lam, reweight=reweight
                        )
                accumulator.add(
                    readout,
                    nll_shuffled=shuffled_nll,
                    nll_full_event=full_event_nll(batch, output.lam),
                    contrasts=contrasts,
                )
    finally:
        model.train(was_training)
    return accumulator.scalars(), accumulator.ratio


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

    # 门禁批必须**非空**：空批上的 G1/G3 都是空过（见 `BatchSource.batch` 的说明）。
    min_events = max(0, int(gates.batch_min_events))
    g1_batch = to_device(source.batch(masked=True, min_events=min_events))
    # G3 的样本数显式给足：样本太少时模型可以在预算内背下小批，基线对照失效
    g3_samples = max(1, int(gates.baseline_samples))
    # G3 的前向按样本维分段（plan 07 §9-15）：真实窗口上 samples=16 一次性前向会 OOM，
    # 分段后内存回到「一段样本」的量级，而梯度与 loss 与不分段一致（per_event 的
    # 分段修正见 `make_step_fn`）。
    g3_chunks = max(1, int(gates.baseline_chunks))
    # G3 的基线是**全事件**口径的闭式常数基线（`N·(1+log(|Ω|/N))`，plan 03 的 `constant_baseline_nll`），
    # 因此 G3 必须在**无遮盖**批上比：遮盖路径的 loss 是「只监督被遮盖事件 + 重标定 1/r」，
    # 与全事件基线不在同一测度上（§4.3 的配对纪律）。
    g3_batch = to_device(source.batch(masked=False, samples=g3_samples, min_events=min_events))
    # RFC-0037 R4（原 P0）：门禁臂**固定 fp32**——bf16 下同臂重复运行噪声 2.8×
    # （RFC-0036 §2.3 实测），任何阈值化读数都不可解释；fp32 同进程逐位一致。
    # 训练本身仍用 cfg.optim.precision，不受影响。
    gate_precision = "fp32"

    # 门禁优化器的学习率**独立于训练 lr**：门禁的预算只有 steps 步，必须让模型
    # 真的收敛（否则 G3 停在初始点附近，基线判据失去意义）。
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
    # 和 G1 的模型共处；释放后 G1 的显存回到「一个模型 + 一个批」的量级。
    # 门禁是**分钟级静默**的（此前整轮跑完才写第一行日志）⇒ 外部无法区分「慢」与「卡死」。
    # 批的形状（尤其是 K）**就是**门禁耗时的决定量：步时 ∝ (K·T)²，G3 还要 ×16 段。
    logger.info(
        "门禁批：G1 K=%d samples=%d events=%.0f｜G3 K=%d samples=%d events=%.0f；"
        "chunks=%d gate_precision=%s（训练=%s）",
        g1_batch.n_lines(),
        g1_batch.batch_size(),
        event_total(g1_batch),
        g3_batch.n_lines(),
        g3_batch.batch_size(),
        event_total(g3_batch),
        g3_chunks,
        gate_precision,
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
        chunks=g3_chunks,
        precision=gate_precision,
    )
    logger.info(
        "门禁 G3 预计算开始：%d 步 x %d 段（K=%d）——这一项在正常 K 下就要十几分钟",
        max(1, gates.baseline_steps),
        g3_chunks,
        g3_batch.n_lines(),
    )
    model_loss = g3_step()
    for _ in range(max(1, gates.baseline_steps) - 1):
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

    # ⚠️ 抬高初值只服务 G1 的**相对**判据；G3 是基线口径，必须从模型的自然初始化
    # 出发（否则它在预算内先花掉一半步数去压一个人为抬高的、与任务无关的积分项）。
    # 见 `GatesConfig.contrast_initial_head_bias` 的实测依据。
    g1_model = build_model(g1_batch.grid, model_seed=seed, head_bias=gates.initial_head_bias)

    inputs = GateInputs(
        step_fn_real=make_step_fn(
            g1_model,
            g1_batch,
            optimizer_for(g1_model),
            grad_clip_norm=cfg.optim.grad_clip_norm,
            precision=gate_precision,
        ),
        model_loss=float(model_loss),
        baseline_loss=float(baseline_loss),
        frames=frames,
        duration_s=duration,
        frame_rate=float(g1_batch.frame_rate),
    )
    stats = {
        "g1_events": event_total(g1_batch),
        "g1_lines": float(g1_batch.n_lines()),
        "g1_samples": float(g1_batch.batch_size()),
        "g3_events": g3_events,
        "g3_lines": g3_lines,
        "g3_samples": g3_samples_used,
        "g3_chunks": float(g3_chunks),
        "gate_min_batch_events": float(min_events),
        # global 层的自注意力长度 L = K * T：成本 ∝ L^2（每样本独立，但 L^2 缓冲区随
        # 分段内样本数线性叠加）。（2026-09-27 第四轮实测：这是真实门禁墙钟的量级来源。）
        "gate_optimizer_lr": float(gate_lr),
        "g1_initial_head_bias": float(gates.initial_head_bias),
        "contrast_initial_head_bias": float(gates.contrast_initial_head_bias),
        # RFC-0037 R4：门禁臂精度固定 fp32（bf16 重复噪声 2.8×，读数不可解释）。
        "gate_precision": 32.0,
        "field_t_bins": float(g1_batch.grid.t_bins),
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


def vram_reserved_gib() -> float:
    """分配器当前**保留**量（GiB）；CPU 上恒为 0。

    与 `_peak_vram_gib`（峰值已分配）一起读才看得出事故机制（plan 07 §9-57）：
    保留量 ≫ 已分配量 = 一堆**空闲但仍占着驱动显存**的块。
    """
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.memory_reserved()) / 2**30


def maintain_vram_hygiene(threshold_gib: float) -> bool:
    """保留量 − 已分配量超过阈值时把缓存还给驱动（plan 07 §9-57）。

    动机：一次 K≈128 的大批把分配器峰值保留量顶到 ~5.2 GiB 且**长期不释放**，驱动侧总量
    因此停在 ~7.86/8.15 GiB；下一次大分配无处可放 ⇒ Windows 静默回退共享显存，功耗从
    ~102 W 塌到 ~31 W、步时放大一个量级以上（2026-09-28 实测 step 951 卡死）。
    只回收「远超真实需求」的那部分（默认 1 GiB 阈值），正常步不付重分配代价。

    Returns:
        是否真的调用了 `torch.cuda.empty_cache()`。
    """
    if threshold_gib <= 0.0 or not torch.cuda.is_available():
        return False
    gap = float(torch.cuda.memory_reserved()) - float(torch.cuda.memory_allocated())
    if gap > threshold_gib * 2**30:
        torch.cuda.empty_cache()
        return True
    return False


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
        # val / 分层 / 条件对照（plan 07 §9-47 A–D、§9-46）：名字只在 SCALAR_TAGS 里写一次。
        # 非有限的读数（例如 λ ≡ 0 处的全事件 NLL = +∞，那是契约行为）只留在 jsonl 里，
        # 不进 TB——图上的 ±inf 会把整条曲线压平，而 jsonl 才是权威记录。
        for key, tag in SCALAR_TAGS.items():
            if key not in row:
                continue
            value = float(row[key])
            if math.isfinite(value):
                writer.add_scalar(tag, value, step)
        writer.add_scalar("sys/peak_vram_gib", float(row["peak_vram_gib"]), step)
        # 分配器保留量（§9-57）：峰值与保留量之间的差就是「空闲占位」——事故的直接机制。
        if "vram_reserved_gib" in row:
            writer.add_scalar("sys/vram_reserved_gib", float(row["vram_reserved_gib"]), step)
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


def _save_best(
    artifacts: RunArtifacts,
    cfg: TrainConfig,
    model: nn.Module,
    *,
    step: int,
    data_rev: str,
    gates_green: bool,
    reason: str,
) -> Path:
    """刷新最优快照 `best.pt`（**判据由调用方给出**，本函数只负责写盘与留痕）。

    判据的两种口径（plan 07 §9-47 A8，这是本轮的**核心修复**）：

    - **有 val**：按 `val/ratio` 选（`reason="val_ratio=..."`）。旧的「按训练损失选」
      在本项目里等于「按谁抽到最空的窗来选模型」——实测选中的是 step 6234 那个
      `events=0, K=1` 的空窗（§9-46）。
    - **无 val**（val_every=0 / 合成来源）：沿用训练损失口径（向后兼容的旧行为，
      并在日志里**明说**它不作为模型选择依据）。

    `best.pt` 的写盘时刻是**测出更优判据的那一步**（不随 `run.save_every`）：
    否则「最优」会落在两个存盘点的中间，而权重只存在于其中一个点上。
    """
    path = save_checkpoint(
        artifacts.checkpoints / BEST_CHECKPOINT_NAME,
        model=model,
        cfg=cfg,
        step=step,
        data_rev=data_rev,
        git_rev=git_rev(artifacts.root),
        gates_green=gates_green,
    )
    logger.info("最优快照 best.pt @step %d（%s）", step, reason)
    return path


def _save_step(
    artifacts: RunArtifacts,
    cfg: TrainConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    data_rev: str,
    gates_green: bool,
    report: TrainReport,
) -> Path:
    """存一个步级 checkpoint（**先写后删**），然后旋转旧文件。

    最优快照 `best.pt` 不在这里写：它的判据是 `val/ratio`（`_save_best`），
    而 val 的周期与 `run.save_every` 无关 ⇒ 把它挂在存盘步上会让「最优」落到
    两个 val 点之间。旋转仍然把它列为 `keep_paths`（**永不删**）。
    """
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
    rotate_checkpoints(artifacts.root, keep_last=cfg.run.keep_last, keep_paths=[best_path])
    logger.info("checkpoint @step %d -> %s", step, path.name)
    return path


def _resolve_val_source(
    cfg: TrainConfig,
    source: BatchSource,
    val_source: ValBatchSource | None,
) -> ValBatchSource | None:
    """决定这次训练用哪个验证集（None = 不跑 val）。

    三种情形，且**没有静默降级**：

    1. 显式给了 `val_source`（测试与将来的消融臂）：直接用；
    2. 没给、且训练来源是真实清单：按 `data.split_val` 与 `optim.seed` 构造
       :class:`ManifestValSource`——**留出集**必须是另一份 split（§9-47 A1）；
    3. 没给、且来源是合成（`SmokeBatchSource`）：**打警告并跳过**，因为合成来源没有
       「留出集」这回事（拿合成批当 val 只会把「同一批被反复评估」伪装成验证）。
       `val_every=0` 是显式关闭，不打警告。
    """
    if val_source is not None:
        return val_source
    if int(cfg.optim.val_every) <= 0:
        return None
    if isinstance(source, ManifestBatchSource):
        return ManifestValSource(cfg, seed=int(cfg.optim.seed))
    logger.warning(
        "optim.val_every=%d 但训练来源不是真实清单（%s）：跳过 val。"
        "合成批没有留出集，把它当 val 只是把同一批数据反复评估了一遍",
        cfg.optim.val_every,
        type(source).__name__,
    )
    return None


def train(  # noqa: PLR0912, PLR0915 - 循环的分支/语句数靠注释说明更清楚，拆函数会把状态切碎
    cfg: TrainConfig,
    *,
    source: BatchSource,
    artifacts: RunArtifacts,
    data_rev: str,
    gates_green: bool,
    device: str = "cpu",
    resume_from: Path | None = None,
    val_source: ValBatchSource | None = None,
) -> TrainReport:
    """参考训练循环（默认 CPU 可跑；极小配置用于 CI 与冒烟）。

    **验证集（plan 07 §9-47，2026-09-27 第十轮）**：`val_source` 给出时，每
    `optim.val_every` 步在**同一批固定窗口**上跑一次 `evaluate_val`，读数进 TB 与
    `logs/loss_history.jsonl`（`val_*` / `cond_*`），并且 **`best.pt` 改按 `val/ratio` 选**
    （A8）——旧的「按训练损失选」实测等价于「按谁抽到最空的窗选」。
    `val_source=None` 且训练来源是 :class:`ManifestBatchSource` 时自动按 `data.split_val`
    构造 :class:`ManifestValSource`（§9-47 A1：固定集必须是**留出集**，不能拿训练窗口凑）。
    val **不产生任何梯度**（`no_grad` + `model.eval()`），也不参与 LR / 早停（G-④）。

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
    val = _resolve_val_source(cfg, source, val_source)
    val_every = max(0, int(cfg.optim.val_every)) if val is not None else 0
    if val is None:
        logger.info("验证集：未启用（val_every=0，或来源没有留出集）")
    else:
        # 顺序有讲究：**先把「要发生什么」说清楚，再触发索引构建**。val split 的索引是
        # 另一份缓存，首次要解析该切分的每张谱面（分钟级，§9-47 G-①）——先把话说出来，
        # 否则那几分钟的静默就是一个看起来像卡死的窗口。
        logger.info(
            "验证集：split=%s 的前 %d 个窗口，每 %d 步原样重放一次（首次会构建 val 索引，见下）",
            cfg.data.split_val,
            int(cfg.optim.val_windows),
            val_every,
        )
        logger.info(
            "⚠️ val 的数字与 gates.txt **不可直接比**：initial_head_bias=%g 只在门禁路径生效",
            cfg.gates.initial_head_bias,
        )
        logger.info(
            "（训练路径 bias=0）；val 精度沿用训练口径 %s（bf16 舍入确定性，不影响跨步比较）",
            cfg.optim.precision,
        )
        windows = val.windows()  # ← 这一行触发 val 索引的构建（上面的日志已经把话说明白了）
        if windows <= 0:
            logger.warning(
                "本轮关闭 val：split=%s 里一个窗口都没有（清单没有留出集，或 max_samples 把它切空了）。"
                "**不要**在没有留出集的情况下宣称做过模型选择",
                cfg.data.split_val,
            )
            val = None
            val_every = 0
        else:
            logger.info("val 来源：%s（%d 个窗口）", val.describe(), windows)
    # 取批入口：同步或 DataLoader（见 `ManifestBatchSource.batches`）。探测批在 start_step==1
    # 时就是第一步的批（不白抽一个窗口）；续训时它只用来拿 grid，游标由 start_step 定位。
    stream = source.batches(start_step=start_step, first=first_batch if start_step == 1 else None)
    history: list[tuple[int, float]] = []
    #: 训练损失的最优（`metrics.json` 的 best_loss，**语义不变**）。
    best = float("inf")
    #: `val/ratio` 的最优（**模型选择判据**，A8）；None = 尚未测过或没有 val。
    best_ratio: float | None = None
    best_ratio_step = 0
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
            # 显存卫生（plan 07 §9-57）：backward 之后本步激活已释放回缓存，若保留量远超
            # 真实需求就还给驱动，避免「峰值保留量长期占位 ⇒ 下一次大分配滑进共享显存」。
            maintain_vram_hygiene(cfg.optim.vram_hygiene_gib)
            # loss.detach() 取标量会**强制同步**：到这里 GPU 上的搬运/前向/反向/裁剪/更新
            # 都已落地，因此下面的 compute 才是真计算时间，而不是 kernel 排队时间。
            value = float(loss.detach())
            # RFC-0037 R3：per_event 的除子 D（E=0 时 D=1）⇒ loss_sum_raw 还原旧 sum 口径。
            d_norm = event_normalizer(batch)
            computed = time.perf_counter()
            elapsed = computed - started
            data_time = drawn - started
            compute_time = computed - drawn
            history.append((step, value))
            best = min(best, value)
            if step == start_step:
                report.first_loss = value
            coverage = source.coverage()
            # 训练侧的**近乎免费**仪表（plan 07 §9-47 I + §9-46 / 任务 B）：
            # ∫λ 在损失里本来就算过（这里走同一个 integral_term，零额外前向）；
            # 分层则用损失**本身**（reduction="none"）把一步拆成两个总体——
            # 43.8% 空窗 / 56.2% 非空窗的中位差五个数量级，混在一起的单步数字没有趋势。
            integral_value = float(integral_term(output, batch).sum().item())
            batch_events = event_total(batch)
            row: dict[str, float] = {
                "step": float(step),
                "loss": value,
                # 旧口径（sum）对照：raw = 归一化 loss × D，与历史 run 的曲线可比（RFC-0037 R3）。
                "loss_sum_raw": value * d_norm,
                "step_time_s": elapsed,
                # 步时**拆开**（plan 07 §9-42）：GPU 利用率偏低时，必须能区分
                # 「数据侧供给不足（GPU 在等）」与「计算侧本身慢」。混成一个数只能
                # 靠功率/利用率反推，而这两者都看不出瓶颈在哪一侧。
                "data_time_s": data_time,
                "compute_time_s": compute_time,
                "grad_norm": grad_norm,
                # 该步是否触发梯度裁剪（grad_norm 是**裁剪前**总范数）：归一化损失落地后
                # 用它量化「clip=1.0 还在不在每一步生效」（RFC-0037 R3 / §6-1）。
                "clip_active": float(
                    cfg.optim.grad_clip_norm is not None
                    and math.isfinite(grad_norm)
                    and grad_norm > float(cfg.optim.grad_clip_norm)
                ),
                "peak_vram_gib": _peak_vram_gib(),
                # 分配器保留量（GiB）：与 peak 一起看才分得清「真需求」与「空闲占位」（§9-57）。
                "vram_reserved_gib": vram_reserved_gib(),
                "lr": float(cfg.optim.lr),
                # 批的 K 与事件数：显存墙与「空批」两个老问题都靠它在线可见
                # （plan 07 §9-23 / §9-35；步时 ∝ K²，K 必须和 loss 一起看）。
                "batch_n_lines": float(batch.n_lines()),
                "batch_events": batch_events,
                # 数据覆盖：epoch 内进度与累计「见过多少窗口/谱面」。采样器一旦饱和，
                # loss 曲线**看不出来**（它只是反复拟合同一小撮样本）——只有覆盖率能。
                "epoch": coverage["epoch"],
                "windows_seen": coverage["windows_seen"],
                "windows_total": coverage["windows_total"],
                "charts_seen": coverage["charts_seen"],
                "charts_total": coverage["charts_total"],
            }
            # Σ∫λdV / Σn（§9-47 I）：无事件时**不报**（缺失 != 0）。空窗本来就只有
            # 积分项，除一个 0 会造出一个看起来像尖峰的假读数。
            if batch_events > 0.0:
                row["integral_per_event"] = integral_value / batch_events
            row.update(stratified_step_loss(output, batch))
            val_due = val is not None and val_every > 0 and step % val_every == 0
            if val_due:
                assert val is not None  # val_due 蕴含 val 非 None（mypy 收窄）
                val_started = time.perf_counter()
                val_scalars, ratio = evaluate_val(
                    model,
                    val,
                    device=target_device,
                    precision=cfg.optim.precision,
                    seed=cfg.optim.seed,
                )
                # val 的墙钟**单独记账**：它不落在任何 step_time_s 里（计时终点在它之前），
                # 不报出来就等于「开销看不见」——那正是 §9-47 E 的算式要防的事。
                row["val_time_s"] = time.perf_counter() - val_started
                row.update(val_scalars)
                logger.info(
                    "val @step %d：nll=%.6g（常数基线 %.6g，ratio=%.4f）空窗占比 %.3f，耗时 %.2f s",
                    step,
                    float(val_scalars.get("val_nll", float("nan"))),
                    float(val_scalars.get("val_nll_constant", float("nan"))),
                    float(val_scalars.get("val_ratio", float("nan"))),
                    float(val_scalars.get("val_empty_share", float("nan"))),
                    row["val_time_s"],
                )
                if ratio is not None and (best_ratio is None or ratio < best_ratio):
                    previous = "首次" if best_ratio is None else f"上一次在 step {best_ratio_step}"
                    best_ratio = ratio
                    best_ratio_step = step
                    _save_best(
                        artifacts,
                        cfg,
                        model,
                        step=step,
                        data_rev=data_rev,
                        gates_green=gates_green,
                        reason=f"val/ratio={ratio:.6f}（{previous}）",
                    )
            pending.append(row)
            if step % log_every == 0 or step == cfg.optim.max_steps or val_due:
                # val 步强制刷盘：val 读数必须与它在同一步的训练标量落在同一行里，
                # 否则「这一步的 val」要跨 log_every 去找（§9-47 A9 的「走 _flush_scalars 那条路」）。
                _flush_scalars(writer, artifacts.logs / HISTORY_FILENAME, pending)
                pending.clear()
            if cfg.run.save_every > 0 and step % cfg.run.save_every == 0:
                _save_step(
                    artifacts,
                    cfg,
                    model,
                    optimizer,
                    step=step,
                    data_rev=data_rev,
                    gates_green=gates_green,
                    report=report,
                )
                if val is None and value <= best:
                    # 无 val（val_every=0 / 合成来源）：沿用训练损失口径的旧行为。
                    # **它不是模型选择依据**（§9-46）；训练开始的日志行已明说这一点。
                    _save_best(
                        artifacts,
                        cfg,
                        model,
                        step=step,
                        data_rev=data_rev,
                        gates_green=gates_green,
                        reason=f"训练损失={value:.6f}（无 val：仅兼容口径）",
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

    #: **A/B 诊断开关**（生产恒为 `False`）：`True` = 退回 `torch` 默认的按序交付。
    #:
    #: 只给 `scripts/local_inorder_ab.py` 做对照实测用（plan 07 §9-50 的三臂标定）；
    #: 不配置化、不进续训指纹——它不是语义旋钮：两种取值下**交付顺序逐位一致**
    #: （`True` 时队头阻塞，`False` 时靠主进程重排），差别只在吞吐。
    _ab_force_in_order: ClassVar[bool] = False

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
                    window_cache_dir=(
                        None if data.window_cache_dir is None else Path(data.window_cache_dir)
                    ),
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
        samples: int | None = None,
        min_events: int = 0,
    ) -> FieldBatch:
        """取若干**同网格身份**的窗口并 collate。

        `min_events > 0`：逐次重抽直到批内事件数达标（最多 :attr:`max_event_draws` 次）。
        **为什么必须**：真实切片上第一个桶的第一个窗口实测 0 事件，门禁因此空过
        （plan 07 §9-23；G1 报了 0.0014 的「过拟合」却没有任何事件可过拟合）。

        Raises:
            ValueError: 清单里没有可用样本，或连续重抽都取不到 `min_events` 个事件。
            GridMismatchError: 分桶逻辑失效（不应发生；由 collate 兜底）。
        """
        return draw_until_min_events(
            lambda: self._draw(masked=masked, samples=samples),
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

        **乱序交付 + 主进程按槽位重排**（plan 07 §9-50；这是 RFC-0034 §5 那条不变量的
        落地——旧代码只是**假设**按序交付，现在改成**证明**）。为什么必须动这里：`torch` 的
        `DataLoader` 默认 `in_order=True`，而它的 `_try_put_index` **只在按序交付时**
        被调用一次 ⇒ 队头一条慢窗口（实测每窗 p50 0.57 s / p99 5.9 s）会让全部 worker
        干完手上的活**集体等索引**：实测 8 worker 只跑出 4.8-5.2 核 / 6.85 窗口/s，
        worker 侧 `py-spy dump` 全部停在 `index_queue.get()`。
        `in_order=False` 下每交付一批就补发一个索引 ⇒ 8.05-8.15 核 / 9.85-10.08 窗口/s。

        但**不能**把乱序直接交给训练循环：`self._cursor` 是「计划前缀长度」一个标量，
        覆盖率在线标量 / 续训 O(1) 定位 / `data.workers` 语义中性（RFC-0034 §5）全由它
        保证。因此每批自带槽位终点标签（:class:`SlotTag`），主进程用字典缓冲**重排回
        计划顺序**再产出 —— 并发拿满，而样本序列与 `workers=0` 仍**逐位一致**。

        重排缓冲的上界 = `DataLoader` 的在飞批次上界（`prefetch_factor * workers`），
        与 `in_order=True` 时 `_task_info` 自己攒的乱序批次**同一量级** ⇒ 不新增内存风险。
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
        dispatched: deque[int] = deque()
        order = plan.order

        def _indices() -> Iterator[list[int]]:
            for start, stop in sampler.spans():
                dispatched.append(stop)
                # 首元素 = 本批的槽位终点标签（负编码 `-stop-1`，与合法的非负窗口下标
                # 不可能冲突）；`_SlotTaggedDataset` 在 worker 侧把它翻译成 `SlotTag`，
                # 于是「批 <-> 槽位」这个对应关系能穿过进程边界回来。
                yield [-(stop + 1), *(int(value) for value in order[start:stop])]

        loader = DataLoader(
            _SlotTaggedDataset(dataset),
            batch_sampler=_indices(),
            collate_fn=partial(_collate_with_slot, collate_field_batch),
            num_workers=workers,
            persistent_workers=True,
            prefetch_factor=2,
            pin_memory=torch.cuda.is_available(),
            in_order=self._ab_force_in_order,
        )
        # 生成器被提前关闭时由 GC 回收 loader（DataLoader.__del__ 会收掉 worker 进程）；
        # 长跑里 generator 与训练同寿命，不必手动收尾。**产出一批才更新一次 `_cursor`**：
        # 缓冲里多收的批次不改记账，因此 `coverage()` 与 `workers=0` 逐位一致。
        buffer: dict[int, FieldBatch] = {}
        iterator = iter(loader)
        while True:
            if not (dispatched and dispatched[0] in buffer):
                try:
                    stop, batch = next(iterator)
                except StopIteration:
                    if dispatched or buffer:
                        raise RuntimeError(
                            "取批重排缓冲未排空（待交付 "
                            f"{len(dispatched)} 批 / 已收 {len(buffer)} 批）——乱序交付丢了批"
                        ) from None
                    return
                buffer[stop] = batch
                continue
            stop = dispatched.popleft()
            self._cursor = stop
            yield buffer.pop(stop)

    def _draw(
        self,
        *,
        masked: bool,
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
        return batch

    def describe(self) -> str:
        """一行来源说明（含清单路径与切分）。"""
        buckets = "" if self._plan is None else f"，网格桶={int(self._plan.bucket_id.max()) + 1}"
        return (
            f"manifest({self.split})：{self.cfg.data.manifest_path}"
            f"（max_samples={self.cfg.data.max_samples}，t_window={self.cfg.data.t_window}{buckets}）"
        )


@dataclass(slots=True)
class ManifestValSource:
    """固定验证集：从 `split_val` 计划里**按网格桶成组**取 N 个窗口，每次重放出逐位一致的一批。

    **为什么要「计划前缀」而不是随机抽 N 个窗口**（plan 07 §9-47 A1）：

    1. 计划层是纯函数（`(seed, epoch)` 决定顺序，RFC-0034 §5）⇒ 续训后前缀**逐位一致**；
    2. 遮盖种子由 `_window_seed(seed, row_index, window_index)` 派生（`data/dataset.py`）
       ⇒ 连**遮盖模式**都一样，不只是窗口一样；
    3. 于是「第 3000 步的 val」与「第 15000 步的 val」只差权重——这才叫跨步可比。

    **不缓存张量**：每次 `batches()` 重放同一批窗口（重新物化）。缓存 128 个真实窗口
    的场大约 2 GB，而本机 `commit` 已经贴过上限（§9-49）；重放的数据成本由 worker 摊薄
    （§9-47 E：workers=0 时单窗口 0.85 s，workers>0 时约 0.038 s）。

    **不写回任何状态**：本类没有游标、没有覆盖率记账——val 不是训练信号（§9-47 G-④）。
    """

    cfg: TrainConfig
    #: 遮盖种子与计划种子（默认 `optim.seed`）。与训练 seed 相同是**有意**的：val 的窗口
    #: 与遮盖模式因此与「训练看到的世界」同口径，差异只在「留出集」与「权重」。
    seed: int = 0
    #: 覆盖 `optim.val_windows`（测试与小样本用）。
    n_windows: int | None = None
    _dataset: ChartPairDataset | None = None
    _plan: WindowPlan | None = None

    def _ensure(self) -> ChartPairDataset:
        if self._dataset is None:
            from beatmorph.data.dataset import ChartPairDataset, DatasetConfig

            data = self.cfg.data
            split = str(data.split_val)
            # 坑①（§9-47 G）：val split 的索引是**另一份缓存**，首次要解析该切分的每张谱面
            # （按 train 的 36.5 min / 6750 张外推约 4-5 min，一次性）。必须提前打日志，
            # 否则这段静默看起来就是卡死。
            logger.info(
                "val 索引：split=%s。首次构建要解析该切分的每张谱面（按 train 的 36.5 min / "
                "6750 张外推约 4-5 min，一次性；之后走 .dataset_index 落盘缓存）",
                split,
            )
            started = time.perf_counter()
            self._dataset = ChartPairDataset(
                DatasetConfig(
                    manifest_path=Path(data.manifest_path),
                    chart_dir=Path(data.chart_dir),
                    feature_dir=Path(data.feature_dir),
                    split=split,
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
                    window_cache_dir=(
                        None if data.window_cache_dir is None else Path(data.window_cache_dir)
                    ),
                ),
            )
            logger.info(
                "val 索引就绪：%d 个窗口，耗时 %.1f s\n%s",
                len(self._dataset),
                time.perf_counter() - started,
                self._dataset.describe(),
            )
        return self._dataset

    def plan(self) -> WindowPlan:
        """epoch 0 的取批计划（**纯函数**：同配置同 seed 逐位一致）。"""
        if self._plan is None:
            from beatmorph.data.plan import plan_epoch

            batch_size = max(1, int(self.cfg.optim.batch_size))
            self._plan = plan_epoch(
                self._ensure(),
                seed=self.seed,
                epoch=0,
                chunk=batch_size,
                batch_size=batch_size,
            )
        return self._plan

    def windows(self) -> int:
        """验证集窗口数 = min(optim.val_windows, 本 split 的窗口数)；**空 split 记 0**。

        为什么空 split 要记 0 而不是抛：`plan_epoch` 对空索引直接报错（它的契约是「必须有
        窗口才能排顺序」），但「这份清单没有留出集」是一个**环境事实**，不是缺陷——
        训练入口应当**明确关闭 val 并大声告警**，而不是让整轮训练崩在一个本可以降级的
        观测项上（`train()` 里就是这么处理的）。
        """
        requested = int(
            self.n_windows if self.n_windows is not None else self.cfg.optim.val_windows
        )
        if len(self._ensure()) <= 0:
            return 0
        return max(1, min(requested, int(self.plan().n_windows)))

    def _slots(self) -> list[list[int]]:
        """按**网格桶**成组取 `val_windows` 个窗口（不是取计划前缀）。

        为什么不是前缀（§9-53，实测）：计划是**跨桶轮转发牌**的 ⇒ 前 N 个槽位落在 **N 个不同桶**
        ⇒ 每个窗口各自成 batch（B=1）× N 次前向。而前向成本 ∝ K²（global 层对 `L=K·T` 做全自
        注意力）⇒ 成本是 `Σf(K_i)` 而不是 `f(平均 K)`。实测 val 前缀 K 中位 30.5 / 均值 46.4 /
        p90 106 / **ΣK² = 422 139** —— 按桶成组后**同样的窗口数**，前向成本约降 `val_batch` 倍。

        固定性不受影响：选择仍是 `plan` 的**纯函数**（同 `seed` ⇒ 逐位一致，续训亦然），
        只是「取哪些窗口」从「前缀」变成「按桶首次出现的顺序、每桶取前 `val_batch` 个」。
        同桶 ⇒ 同 `grid_key` ⇒ `collate_field_batch` 的网格身份约束满足（K 由 collate 补齐，
        补齐行的 `line_mask=False`，指标按有效线加权 ⇒ **语义不变**）。
        """
        total = self.windows()
        if total <= 0:  # 空 split：没有槽位可取（`batches()` 因此产出空流）
            return []
        plan = self.plan()
        batch = max(1, int(self.cfg.optim.val_batch))
        per_bucket: dict[int, list[int]] = {}
        for slot in range(plan.n_windows):
            per_bucket.setdefault(int(plan.bucket_id[slot]), []).append(int(plan.order[slot]))
        chunks: list[list[int]] = []
        taken = 0
        # `bucket_id` 的编号本身就是「在计划里首次出现的顺序」⇒ 排序即计划序（纯函数）。
        for bucket in sorted(per_bucket):
            if taken >= total:
                break
            window_ids = per_bucket[bucket]
            take = min(batch, total - taken, len(window_ids))
            chunks.append(window_ids[:take])
            taken += take
        return chunks

    def batches(self) -> Iterator[FieldBatch]:
        """按计划顺序重放验证批（data.workers>0 时走 DataLoader，与训练同一套槽位机制）。

        **为什么可以走 worker 路径**（§9-47 A7 / RFC-0034 §5）：worker 数与样本序列无关，
        而同步路径的单窗口数据成本是 worker 路径的 20 倍（0.85 s vs 0.038 s）——val 若强制
        workers=0，光数据侧就要 100 s 以上，5% 的开销预算当场失效。
        """
        chunks = self._slots()
        workers = max(0, int(self.cfg.data.workers))
        if workers == 0:
            from beatmorph.data.dataset import collate_field_batch

            dataset = self._ensure()
            for chunk in chunks:
                yield collate_field_batch([dataset[index] for index in chunk])
            return
        yield from self._batches_parallel(chunks, workers=workers)

    def _batches_parallel(
        self, chunks: Sequence[Sequence[int]], *, workers: int
    ) -> Iterator[FieldBatch]:
        """DataLoader 路径：**与训练取批同一套** SlotTag / 主进程重排机制（§9-50）。

        重排是必需的：in_order=False 让交付顺序变成任意排列，而 val 的读数必须与
        workers=0 **逐位一致**（T1 的验收条件）。聚合量本身对顺序不敏感，但
        「同一批两次调用逐位一致」这条断言只有在顺序也确定时才可测。
        """
        from torch.utils.data import DataLoader

        from beatmorph.data.dataset import collate_field_batch

        dataset = self._ensure()
        ordered = [list(chunk) for chunk in chunks]
        dispatched: deque[int] = deque()

        def _indices() -> Iterator[list[int]]:
            for position, chunk in enumerate(ordered):
                dispatched.append(position)
                yield [-(position + 1), *chunk]

        loader = DataLoader(
            _SlotTaggedDataset(dataset),
            batch_sampler=_indices(),
            collate_fn=partial(_collate_with_slot, collate_field_batch),
            num_workers=workers,
            prefetch_factor=2,
            pin_memory=torch.cuda.is_available(),
            in_order=ManifestBatchSource._ab_force_in_order,
        )
        buffer: dict[int, FieldBatch] = {}
        iterator = iter(loader)
        while True:
            if not (dispatched and dispatched[0] in buffer):
                try:
                    position, batch = next(iterator)
                except StopIteration:
                    if dispatched or buffer:
                        raise RuntimeError(
                            "val 取批重排缓冲未排空（待交付 "
                            f"{len(dispatched)} 批 / 已收 {len(buffer)} 批）：乱序交付丢了批",
                        ) from None
                    return
                buffer[position] = batch
                continue
            yield buffer.pop(dispatched.popleft())

    def describe(self) -> str:
        """一行来源说明（落进训练日志）。"""
        return (
            f"manifest({self.cfg.data.split_val})：{self.cfg.data.manifest_path}"
            f"（按网格桶取 {self.windows()} 个窗口 × ≤{self.cfg.optim.val_batch}/批，seed={self.seed}）"
        )


def _drop_occlusion(batch: FieldBatch) -> FieldBatch:
    """去掉遮盖（G3 的无遮盖口径）。"""
    from dataclasses import replace

    return replace(batch, occlusion=None)
