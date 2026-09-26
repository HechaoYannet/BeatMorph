"""参考训练循环（torch 后端）与门禁输入装配（plan 07 §4.1 / §4.3 / M7.2）。

分工：

- `beatmorph/infra/sanity.py`：**判据**（范式中立，只吃 step_fn）；
- 本模块：把「模型 + 批次 + 优化器」接成 `step_fn`，装配 G1-G4 的输入，并提供训练循环；
- `beatmorph/infra/gates.py`：落盘与 fail-closed；
- `beatmorph/infra/lightning_module.py`：plan 07 §4.1 的目标训练栈（可选依赖）。

为什么默认后端是 torch 而不是 Lightning：`pytorch-lightning` 属 `train` extra，而
**门禁必须在默认 CI 里能跑**（CLAUDE.md §4：契约级与门禁级断言不得依赖权重/GPU/可选依赖）。
Lightning 后端仍然提供，并在缺失时给出可操作的报错（而不是静默降级）。
"""

from __future__ import annotations

from collections.abc import Sequence
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
from beatmorph.infra.smoke import shuffled_counts

__all__ = [
    "BatchSource",
    "ManifestBatchSource",
    "TrainReport",
    "betas_of",
    "build_gate_inputs",
    "constant_baseline_for",
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
    ) -> FieldBatch:
        """取一个 batch；`masked` 控制遮盖路径，`samples` 覆盖批大小（None = 配置值）。"""
        ...

    def describe(self) -> str:
        """一行来源说明（落进 gates.txt 上下文）。"""
        ...


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
) -> StepFn:
    """把「前向 + 反传 + 一步优化」包成 `sanity.StepFn`（返回标量 loss）。"""

    def step() -> float:
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        loss = output.loss
        if loss is None:  # pragma: no cover - forward(compute_loss=True) 保证非 None
            raise RuntimeError("前向没有返回 loss：门禁需要 compute_loss=True")
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        return float(loss.detach())

    return step


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
) -> tuple[GateInputs, dict[str, float]]:
    """装配 G1-G4 的输入（各臂**同起点**：同一 seed 建模型）。

    Returns:
        (inputs, stats)：stats 里是落盘用的诊断量（事件数 / 基线 / 帧数 / 时长）。
    """
    seed = cfg.optim.seed if seed is None else seed
    gates = cfg.gates
    g1_batch = source.batch(masked=True)
    # G2 的样本数显式给足：样本太少时打乱臂可以直接背下样本，对照会退化成空转
    g2_samples = max(1, int(gates.shuffle_samples))
    real_batch = source.batch(masked=False, samples=g2_samples)
    shuffled_batch = source.batch(masked=False, shuffled=True, samples=g2_samples)

    def optimizer_for(model: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.optim.lr,
            weight_decay=cfg.optim.weight_decay,
            betas=betas_of(cfg),
        )

    g1_model = model_from_config(
        cfg, g1_batch.grid, seed=seed, initial_head_bias=gates.initial_head_bias
    )
    shuffled_model = model_from_config(
        cfg, shuffled_batch.grid, seed=seed, initial_head_bias=gates.initial_head_bias
    )
    g3_model = model_from_config(
        cfg, real_batch.grid, seed=seed + 1, initial_head_bias=gates.initial_head_bias
    )

    g3_step = make_step_fn(
        g3_model, real_batch, optimizer_for(g3_model), grad_clip_norm=cfg.optim.grad_clip_norm
    )
    model_loss = g3_step()
    for _ in range(max(1, gates.shuffle_steps) - 1):
        model_loss = g3_step()

    frames = int(real_batch.audio_emb.shape[1])
    duration = float(real_batch.grid.total_seconds)
    inputs = GateInputs(
        step_fn_real=make_step_fn(
            g1_model, g1_batch, optimizer_for(g1_model), grad_clip_norm=cfg.optim.grad_clip_norm
        ),
        step_fn_shuffled=make_step_fn(
            shuffled_model,
            shuffled_batch,
            optimizer_for(shuffled_model),
            grad_clip_norm=cfg.optim.grad_clip_norm,
        ),
        model_loss=float(model_loss),
        baseline_loss=constant_baseline_for(real_batch),
        frames=frames,
        duration_s=duration,
        frame_rate=float(real_batch.frame_rate),
    )
    stats = {
        "g1_events": float(g1_batch.counts.sum()) if g1_batch.counts is not None else 0.0,
        "g2_events": float(real_batch.counts.sum()) if real_batch.counts is not None else 0.0,
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
    model.train()
    history: list[tuple[int, float]] = []
    best = float("inf")
    for step in range(1, cfg.optim.max_steps + 1):
        batch = first_batch if step == 1 else source.batch(masked=True)
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
                    x_bins=data.x_bins,
                    k_max=data.k_max,
                    occlusion_ratio=data.occlusion_ratio,
                    seed=self.seed,
                    limit=data.max_samples,
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

    def batch(
        self,
        *,
        masked: bool,
        shuffled: bool = False,
        samples: int | None = None,
    ) -> FieldBatch:
        """取若干**同网格身份**的窗口并 collate（`shuffled` 时置换目标）。

        Raises:
            ValueError: 清单里没有可用样本。
            GridMismatchError: 分桶逻辑失效（不应发生；由 collate 兜底）。
        """
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
