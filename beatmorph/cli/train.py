"""训练 CLI 入口（Plan 09）。

Hydra 驱动，按 ``--config-name`` 选 stage（如 ``stage1_planner``）::

    beatmorph-train --config-name stage1_planner
    python -m beatmorph.cli.train --config-name stage1_planner experiment.max_steps=10

Phase 1 实现 Stage1（planner）训练；Stage0（MERT，仅 Adapter）后续补。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from beatmorph.core.logging import get_logger, setup_logging
from beatmorph.data.datasets import PlannerDataset
from beatmorph.infra.trainer import PlannerLitModule, build_trainer
from beatmorph.planner.density import DensityPlanner

logger = get_logger(__name__)


def build_planner(cfg: DictConfig) -> DensityPlanner:
    """从 model/train config 构造 DensityPlanner。"""
    m = cfg.model
    loss = m.get("loss", {})
    return DensityPlanner(
        n_layers=int(m.get("n_layers", 6)),
        n_heads=int(m.get("n_heads", 8)),
        dim=int(m.get("dim", 768)),
        section_bars=int(m.get("section_bars", 4)),
        n_type_classes=int(m.get("n_type_classes", 5)),
        huber_weight=float(loss.get("huber", 1.0)) if isinstance(loss, dict) else 1.0,
        tv_weight=float(loss.get("tv", 0.1)) if isinstance(loss, dict) else 0.1,
        ce_type_weight=float(loss.get("ce_type", 0.5)) if isinstance(loss, dict) else 0.5,
    )


def _resolve_data_paths(cfg: DictConfig) -> tuple[Path, Path]:
    """解析 charts 文件 + embeddings 目录。"""
    data = cfg.get("data", {})
    processed_dir = Path(
        str(
            data.get(
                "parquet_dir",
                os.environ.get("BEATMORPH_DATA_DIR", "data/processed"),
            )
        )
    )
    charts = Path(str(data.get("charts_path", processed_dir / "charts.jsonl")))
    emb_dir = Path(
        str(
            data.get(
                "embeddings_dir",
                os.environ.get("BEATMORPH_DATA_DIR", "data/embeddings") + "/mert_v1_330m",
            )
        )
    )
    return charts, emb_dir


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """简易 collate：各样本 section 数可能不一，逐样本拼成 batch tensor（按最大 S 填充）。

    Phase 1 简化：要求 batch 内 S 一致（DataLoader 默认 batch_size 可设 1 适配变长）。
    """
    import torch

    audio_emb = torch.stack([b["audio_emb"] for b in batch])
    difficulty = torch.stack([b["difficulty"] for b in batch])
    section_bounds = torch.stack([b["section_bounds"] for b in batch])
    target = {
        k: torch.stack([b["target"][k] for b in batch])
        for k in ("density", "energy", "rest", "type")
    }
    return {
        "audio_emb": audio_emb,
        "difficulty": difficulty,
        "section_bounds": section_bounds,
        "target": target,
    }


@hydra.main(version_base=None, config_path="../../configs", config_name="stage1_planner")
def main(cfg: DictConfig) -> None:
    setup_logging(cfg.get("log_level", "INFO"))
    logger.info("Train config:\n%s", OmegaConf.to_yaml(cfg))

    charts, emb_dir = _resolve_data_paths(cfg)
    dataset = PlannerDataset(charts, emb_dir, section_bars=int(cfg.model.get("section_bars", 4)))
    if len(dataset) == 0:
        raise RuntimeError(
            f"PlannerDataset 空：charts={charts} embeddings={emb_dir}。"
            "先跑 PreprocessPipeline + extract_mert_embeddings。"
        )

    tcfg = cfg.get("train", {})
    loader = DataLoader(
        dataset,
        batch_size=int(tcfg.get("batch_size", 8)),
        shuffle=True,
        collate_fn=collate,
        num_workers=int(tcfg.get("num_workers", 0)),
    )

    planner = build_planner(cfg)
    lit = PlannerLitModule(
        planner,
        lr=float(tcfg.get("lr", 3e-4)),
        weight_decay=float(tcfg.get("weight_decay", 0.01)),
    )
    trainer = build_trainer(cfg)

    logger.info("开始训练：dataset=%d, batch=%d", len(dataset), int(tcfg.get("batch_size", 8)))
    trainer.fit(lit, loader)


if __name__ == "__main__":  # pragma: no cover
    main()
