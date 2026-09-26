"""Lightning 训练栈（可选依赖 {BT}train{BT} extra；plan 07 §4.1）。

口径：**plan 07 §4.1 的目标栈是 PyTorch Lightning**，但 {BT}pytorch-lightning{BT} 属可选依赖，
而门禁与契约级断言必须在默认 CI 里能跑（CLAUDE.md §4）。因此：

- 本模块**只在函数内**导入 lightning（模块级导入会让整个 infra 包在不装 train extra 时不可用）；
- 缺失依赖时抛出 {BT}MissingTrainingDependency{BT}——**可操作的报错**，而不是静默回落到 torch 后端
  （静默回落会让「我以为在跑目标栈」变成一个无人察觉的假设）；
- Lightning 的基类用 {BT}type(...){BT} 动态构造：既满足 mypy（不依赖未安装包的类型），
  又保持「只有真的要用 Lightning 时才 import」。
"""

# ruff: noqa: ANN401
# 理由：本模块是**动态互操作层**——lightning 只在函数内导入，其基类/回调/Trainer 都是以
# 运行期对象形式出现的，因此这里的 Any 是「可选依赖的运行期句柄」，不是接口松动。
# 静态契约仍由 beatmorph/infra/train_loop.py 的 BatchSource / TrainReport 承担。

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from beatmorph.core.logging import get_logger
from beatmorph.field.grid import FieldGrid
from beatmorph.generation.batch import FieldBatch
from beatmorph.infra.artifacts import RunArtifacts, git_rev
from beatmorph.infra.checkpoint import save_checkpoint
from beatmorph.infra.config.schema import TrainConfig
from beatmorph.infra.train_loop import BatchSource, TrainReport, betas_of, model_from_config

__all__ = [
    "MissingTrainingDependency",
    "build_trainer",
    "lightning_available",
    "run_lightning",
]

logger = get_logger("infra.lightning")


class MissingTrainingDependency(RuntimeError):  # noqa: N818 - 语义是「缺依赖」而非通用错误
    """缺少可选训练依赖（{BT}pytorch-lightning{BT} / {BT}tensorboard{BT}）。"""


def _import_lightning() -> Any:
    try:
        import pytorch_lightning as pl
    except ImportError as exc:
        raise MissingTrainingDependency(
            "缺少 pytorch-lightning：plan 07 §4.1 的训练栈需要 train extra。"
            "安装：uv sync --extra train（或把 run.backend 改成 torch 用参考循环）",
        ) from exc
    return pl


def lightning_available() -> bool:
    """lightning 是否可用（env doctor 的 E4 会把缺失记成 UNKNOWN）。"""
    try:
        import pytorch_lightning  # noqa: F401
    except ImportError:
        return False
    return True


class _BatchListDataset(torch.utils.data.Dataset[FieldBatch]):
    """把一个批次列表包成 Dataset（{BT}DataLoader(batch_size=None){BT} 时逐项原样产出）。

    这里刻意**不做** collate：{BT}FieldBatch{BT} 已经是 batch，它的网格身份由 data 层保证。
    """

    def __init__(self, batches: Sequence[FieldBatch]) -> None:
        self.batches = list(batches)

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, index: int) -> FieldBatch:
        return self.batches[index]


def build_trainer(cfg: TrainConfig, *, max_steps: int | None = None) -> Any:
    """按配置构造 {BT}pl.Trainer{BT}（bf16-mixed 只在 CUDA 上启用，CPU 上回落 32）。"""
    pl = _import_lightning()
    precision = cfg.optim.precision
    if precision != "32" and not torch.cuda.is_available():
        logger.warning("CPU 训练：precision=%s 回落为 32", precision)
        precision = "32"
    return pl.Trainer(
        max_steps=cfg.optim.max_steps if max_steps is None else max_steps,
        accelerator="auto",
        devices=1,
        precision=precision,
        gradient_clip_val=cfg.optim.grad_clip_norm,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        log_every_n_steps=1,
    )


def _module_class(cfg: TrainConfig, grid: FieldGrid) -> Any:
    """动态构造 LightningModule 子类（基类来自运行期导入的 lightning）。"""
    pl = _import_lightning()

    def lightning_init(self: Any) -> None:
        pl.LightningModule.__init__(self)
        self.model = model_from_config(
            cfg, grid, seed=cfg.optim.seed, initial_head_bias=cfg.gates.initial_head_bias
        )

    def training_step(self: Any, batch: FieldBatch, batch_idx: int) -> torch.Tensor:
        output = self.model(batch)
        loss = output.loss
        if loss is None:
            raise RuntimeError("前向没有返回 loss")
        self.log("train/loss", loss, prog_bar=False, on_step=True)
        result: torch.Tensor = loss
        return result

    def configure_optimizers(self: Any) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.optim.lr,
            weight_decay=cfg.optim.weight_decay,
            betas=betas_of(cfg),
        )

    return type(
        "FieldLitModule",
        (pl.LightningModule,),
        {
            "__init__": lightning_init,
            "training_step": training_step,
            "configure_optimizers": configure_optimizers,
        },
    )


def _recorder_class(pl: Any, history: list[tuple[int, float]]) -> Any:
    """把每步 loss 记进 history 的回调（训练摘要与 metrics.json 用）。"""

    def on_train_batch_end(
        self: Any,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        value = outputs["loss"] if isinstance(outputs, dict) else outputs
        history.append((int(batch_idx) + 1, float(value.detach())))

    return type("_LossRecorder", (pl.Callback,), {"on_train_batch_end": on_train_batch_end})


def run_lightning(
    cfg: TrainConfig,
    *,
    source: BatchSource,
    artifacts: RunArtifacts,
    data_rev: str,
    gates_green: bool,
    max_steps: int | None = None,
) -> TrainReport:
    """用 Lightning 跑训练，并把最终 checkpoint 落进实验目录。"""
    pl = _import_lightning()
    steps = cfg.optim.max_steps if max_steps is None else max_steps
    batches = [source.batch(masked=True) for _ in range(steps)]
    loader = torch.utils.data.DataLoader(_BatchListDataset(batches), batch_size=None)
    module = _module_class(cfg, batches[0].grid)()
    trainer = build_trainer(cfg, max_steps=steps)
    history: list[tuple[int, float]] = []
    trainer.callbacks.append(_recorder_class(pl, history)())
    trainer.fit(module, train_dataloaders=loader)

    report = TrainReport(
        steps=steps, first_loss=float("nan"), last_loss=float("nan"), best_loss=float("nan")
    )
    report.data_source = source.describe()
    report.loss_history = history
    if history:
        report.first_loss = history[0][1]
        report.last_loss = history[-1][1]
        report.best_loss = min(value for _, value in history)
    path = save_checkpoint(
        artifacts.checkpoints / "final.pt",
        model=module.model,
        cfg=cfg,
        step=steps,
        data_rev=data_rev,
        git_rev=git_rev(artifacts.root),
        gates_green=gates_green,
    )
    report.checkpoints.append(path.name)
    logger.info("%s", report.format())
    return report
