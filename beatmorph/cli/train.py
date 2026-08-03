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
    """padding-aware collate：变长 T_seq 与 S 按批次最大值填充，支持 batch>1。

    产出 ``section_mask [B,S]``（真段=True，pad 段=False）供 ``loss_fn`` 过滤 padding 段；
    ``audio_mask [B,T]`` 供下游清晰使用。padding 段的 section_bounds pad 为 [0,0]，
    ``_pool_sections`` 天然将其 pool 成 0 向量（mask 0<=t<0 全 False），无需改 pool 逻辑。
    """
    import torch

    b0 = batch[0]
    feat = b0["audio_emb"].shape[-1]
    dtype = b0["audio_emb"].dtype
    b = len(batch)

    # ── T_seq 填充（audio_emb）──
    t_max = max(x["audio_emb"].shape[0] for x in batch)
    audio_emb = torch.zeros(b, t_max, feat, dtype=dtype)
    audio_mask = torch.zeros(b, t_max, dtype=torch.bool)
    for i, x in enumerate(batch):
        t = x["audio_emb"].shape[0]
        audio_emb[i, :t] = x["audio_emb"]
        audio_mask[i, :t] = True

    # ── S 填充（section_bounds / target）──
    s1_max = max(x["section_bounds"].shape[0] for x in batch)  # = S+1
    s_max = s1_max - 1
    section_bounds = torch.zeros(b, s1_max, dtype=torch.float32)
    section_mask = torch.zeros(b, s_max, dtype=torch.bool)  # 真段=True
    target = {
        k: torch.zeros(b, s_max, dtype=torch.long if k == "type" else torch.float32)
        for k in ("density", "energy", "rest", "type")
    }
    for i, x in enumerate(batch):
        s1 = x["section_bounds"].shape[0]
        s = s1 - 1
        section_bounds[i, :s1] = x["section_bounds"]
        section_mask[i, :s] = True
        for k in ("density", "energy", "rest", "type"):
            target[k][i, :s] = x["target"][k]

    difficulty = torch.stack([x["difficulty"] for x in batch])
    return {
        "audio_emb": audio_emb,
        "audio_mask": audio_mask,
        "difficulty": difficulty,
        "section_bounds": section_bounds,
        "section_mask": section_mask,
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
