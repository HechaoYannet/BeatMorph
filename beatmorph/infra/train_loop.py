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

from collections.abc import Callable, Sequence
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
from beatmorph.infra.checkpoint import save_checkpoint
from beatmorph.infra.config.schema import TrainConfig
from beatmorph.infra.gates import GateInputs
from beatmorph.infra.sanity import StepFn
from beatmorph.infra.smoke import shuffle_hidden_counts, shuffled_counts

__all__ = [
    "BatchSource",
    "ManifestBatchSource",
    "TrainReport",
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

logger = get_logger("infra.train_loop")


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


def make_step_fn(
    model: nn.Module,
    batch: FieldBatch,
    optimizer: torch.optim.Optimizer,
    *,
    grad_clip_norm: float | None = None,
    chunks: int = 1,
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

    def step() -> float:
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for part in parts:
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
    g3_model = build_model(
        g3_batch.grid, model_seed=seed + 1, head_bias=gates.contrast_initial_head_bias
    )
    g3_step = make_step_fn(
        g3_model,
        g3_batch,
        optimizer_for(g3_model),
        grad_clip_norm=cfg.optim.grad_clip_norm,
        chunks=g2_chunks,
    )
    model_loss = g3_step()
    for _ in range(max(1, gates.shuffle_steps) - 1):
        model_loss = g3_step()
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
            g1_model, g1_batch, optimizer_for(g1_model), grad_clip_norm=cfg.optim.grad_clip_norm
        ),
        step_fn_g2_real=make_step_fn(
            g2_real_model,
            real_batch,
            optimizer_for(g2_real_model),
            grad_clip_norm=cfg.optim.grad_clip_norm,
            chunks=g2_chunks,
        ),
        step_fn_g2_shuffled=make_step_fn(
            shuffled_model,
            shuffled_batch,
            optimizer_for(shuffled_model),
            grad_clip_norm=cfg.optim.grad_clip_norm,
            chunks=g2_chunks,
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

    def to_metrics(self) -> dict[str, Any]:
        """metrics.json 的载荷（**不含**权重；plan 07 §3.2）。"""
        return {
            "steps": self.steps,
            "first_loss": self.first_loss,
            "last_loss": self.last_loss,
            "best_loss": self.best_loss,
            "checkpoints": list(self.checkpoints),
            "data_source": self.data_source,
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


def train(
    cfg: TrainConfig,
    *,
    source: BatchSource,
    artifacts: RunArtifacts,
    data_rev: str,
    gates_green: bool,
    device: str = "cpu",
) -> TrainReport:
    """参考训练循环（默认 CPU 可跑；极小配置用于 CI 与冒烟）。

    Args:
        cfg: 训练配置。
        source: 批次来源（冒烟或真实清单）。
        artifacts: 实验产物目录（checkpoint 与 TB 标量写到这里）。
        data_rev: 数据版本（写进 checkpoint 元数据，恢复时校验）。
        gates_green: 本次运行的门禁是否全绿（写进 checkpoint 元数据）。
        device: 设备字符串。

    Returns:
        TrainReport。
    """
    report = TrainReport(
        steps=0, first_loss=float("nan"), last_loss=float("nan"), best_loss=float("nan")
    )
    report.data_source = source.describe()
    first_batch = source.batch(masked=True)
    model = model_from_config(cfg, first_batch.grid, seed=cfg.optim.seed).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        betas=betas_of(cfg),
    )
    # ⚠️ --device cuda 必须**同时搬批次**（plan 07 §9-22）：此前只把模型 .to(device)，
    # 批次留在 CPU ⇒ 训练路径上的 --device cuda 会直接因设备不一致失败。门禁路径早已
    # 经 build_gate_inputs 的 to_device 搬批次，这里补齐同一行为（cpu 下是空操作）。
    target_device = torch.device(device)
    model.train()
    history: list[tuple[int, float]] = []
    best = float("inf")
    for step in range(1, cfg.optim.max_steps + 1):
        batch = (first_batch if step == 1 else source.batch(masked=True)).to(target_device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        loss = output.loss
        if loss is None:  # pragma: no cover - compute_loss 默认 True
            raise RuntimeError("训练前向没有返回 loss")
        loss.backward()
        if cfg.optim.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip_norm)
        optimizer.step()
        value = float(loss.detach())
        history.append((step, value))
        best = min(best, value)
        if step == 1:
            report.first_loss = value
        if cfg.run.save_every > 0 and step % cfg.run.save_every == 0:
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
            logger.info("checkpoint @step %d -> %s", step, path.name)
    report.steps = cfg.optim.max_steps
    report.last_loss = history[-1][1] if history else float("nan")
    report.best_loss = best
    report.loss_history = history
    write_scalars(artifacts.logs, history)
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
    _buckets: list[list[int]] | None = None
    _bucket_cursor: int = 0
    _cursor: int = 0

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

    def _grouped(self, dataset: ChartPairDataset) -> list[list[int]]:
        """把窗口下标按网格身份分桶（桶序稳定：按 key 排序；桶内保持 index 升序）。

        复杂度 O(窗口数)（`grid_key` 只读索引计划，不解析谱面），首次调用后缓存。
        桶序在构造时**不做随机化**：训练侧的顺序由 `_bucket_cursor` 轮转决定，
        而「同一份配置两次运行逐位一致」是 M7.8 的硬要求（随机化必须显式带 seed）。
        """
        if self._buckets is None:
            grouped: dict[tuple[int, int, float], list[int]] = {}
            for index in range(len(dataset)):
                grouped.setdefault(dataset.grid_key(index), []).append(index)
            self._buckets = [grouped[key] for key in sorted(grouped)]
        return self._buckets

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

    def _draw(
        self,
        *,
        masked: bool,
        shuffled: bool = False,
        samples: int | None = None,
    ) -> FieldBatch:
        """按桶轮转取一批（`batch` 的重抽循环用它）。"""
        from beatmorph.data.dataset import collate_field_batch

        dataset = self._ensure()
        size = max(1, int(self.cfg.optim.batch_size if samples is None else samples))
        buckets = self._grouped(dataset)
        if not buckets:
            raise ValueError("清单里没有可用样本（检查 split / max_samples / 特征缓存）")
        bucket = buckets[self._bucket_cursor % len(buckets)]
        take = min(size, len(bucket))
        start = self._cursor % len(bucket)
        indices = [bucket[(start + offset) % len(bucket)] for offset in range(take)]
        self._cursor = (start + take) % len(bucket)
        self._bucket_cursor = (self._bucket_cursor + 1) % len(buckets)
        windows = [dataset[index] for index in indices]
        batch = collate_field_batch(windows)
        if not masked:
            batch = _drop_occlusion(batch)
        if shuffled:
            batch = _shuffle_targets(batch, seed=self.seed + 991)
        return batch

    def describe(self) -> str:
        """一行来源说明（含清单路径与切分）。"""
        buckets = "" if self._buckets is None else f"，网格桶={len(self._buckets)}"
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
