"""PyTorch Lightning 训练封装（Plan 09 子集，Phase 1 最小可跑）。

提供 :class:`PlannerLitModule`（Stage1 密度规划）与 :func:`build_trainer`。
后续 Stage0/2 各加对应 LitModule；W&B/FSDP/ckpt top-K 留 Phase 2。

奠基 §5：PyTorch Lightning + bf16-mixed + 梯度累积 + 梯度裁剪。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from beatmorph.core.logging import get_logger
from beatmorph.planner.density import DensityPlanner

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = get_logger(__name__)


class PlannerLitModule(LightningModule):
    """Stage1 密度规划 LightningModule。"""

    def __init__(
        self,
        planner: DensityPlanner,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__()
        self.planner = planner
        self.lr = lr
        self.weight_decay = weight_decay
        self._loss_fn = planner.loss_fn

    def forward(self, *args: object, **kwargs: object) -> object:
        return self.planner(*args, **kwargs)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        pred = self.planner(
            batch["audio_emb"],
            batch["difficulty"],
            batch["section_bounds"],
            section_mask=batch.get("section_mask"),
        )
        loss = self._loss_fn(pred, batch["target"])
        self.log("train/loss", loss, prog_bar=True)
        # density 分项 loss（mask-aware，跳过 padding 段）
        mask = pred.get("section_mask")
        den = pred["density"].squeeze(-1)
        tgt_den = batch["target"]["density"]
        if mask is not None:
            self.log(
                "train/loss_density",
                torch.nn.functional.huber_loss(den[mask], tgt_den[mask]),
            )
        else:
            self.log(
                "train/loss_density",
                torch.nn.functional.huber_loss(den, tgt_den),
            )
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        pred = self.planner(
            batch["audio_emb"],
            batch["difficulty"],
            batch["section_bounds"],
            section_mask=batch.get("section_mask"),
        )
        loss = self._loss_fn(pred, batch["target"])
        self.log("val/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        opt = torch.optim.AdamW(
            self.planner.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return opt


def _build_logger(infra: dict[str, Any], exp: dict[str, Any]) -> TensorBoardLogger | Literal[False]:
    """构造实验 logger。

    Phase 1 用 TensorBoard（奠基 §5 的 W&B 留 Phase 2）。
    配置 ``infra.tensorboard.{dir,name,version}`` 控制输出位置，
    ``enabled: false`` 可关闭记录。
    """
    tb = infra.get("tensorboard", {})
    if not tb.get("enabled", True):
        return False
    return TensorBoardLogger(
        save_dir=str(tb.get("dir", "runs/tensorboard")),
        name=str(tb.get("name", exp.get("name", "beatmorph"))),
        version=tb.get("version"),  # None → Lightning 按 run 自动新版本目录
    )


def build_trainer(cfg: DictConfig | dict[str, Any] | None = None) -> Trainer:
    """根据配置构造 Lightning Trainer（Phase 1 最小）。

    cfg 可含 ``experiment.{max_steps,grad_accum,precision,gradient_clip,log_every_n_steps}``、
    ``trainer.{strategy,devices,accelerator}``、
    ``infra.checkpoint.{dir,save_every_n_steps}``、``infra.tensorboard.{dir,name,version,enabled}``。
    """
    cfg = dict(cfg) if cfg is not None else {}

    exp = cfg.get("experiment", {})
    tr = cfg.get("trainer", {})
    infra = cfg.get("infra", {})
    ckpt_cfg = infra.get("checkpoint", {})

    callbacks: list[Any] = []
    ckpt_dir = ckpt_cfg.get(
        "dir",
        "runs/checkpoints",
    )
    save_every = int(ckpt_cfg.get("save_every_n_steps", 2000))
    if save_every > 0:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(ckpt_dir),
                every_n_train_steps=save_every,
                save_top_k=1,
            )
        )

    trainer = Trainer(
        max_steps=int(exp.get("max_steps", 100000)),
        accumulate_grad_batches=int(exp.get("grad_accum", 1)),
        precision=exp.get("precision", "bf16-mixed"),
        gradient_clip_val=float(exp.get("gradient_clip", 1.0)),
        strategy=tr.get("strategy", "auto"),
        devices=tr.get("devices", 1),
        accelerator=tr.get("accelerator", "gpu"),
        callbacks=callbacks,
        logger=_build_logger(infra, exp),
        log_every_n_steps=int(exp.get("log_every_n_steps", 50)),
    )
    return trainer
