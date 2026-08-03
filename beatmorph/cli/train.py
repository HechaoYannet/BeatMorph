"""训练 CLI 入口（Plan 09）。

Hydra 驱动，按 ``--config-name`` 选 stage（如 ``stage1_planner`` / ``stage_vqvae``）::

    beatmorph-train --config-name stage1_planner
    beatmorph-train --config-name stage_vqvae
    python -m beatmorph.cli.train --config-name stage_vqvae experiment.max_steps=10

按 ``cfg.experiment.name`` 分发到对应 ``_run_<stage>``：Stage1（planner）与 Stage2
前置 VQ-VAE。Stage0（MERT，仅 Adapter）后续补。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from beatmorph.core.logging import get_logger, setup_logging
from beatmorph.data.datasets import PlannerDataset, VQVAEDataset
from beatmorph.infra.trainer import PlannerLitModule, VQVAELitModule, build_trainer
from beatmorph.planner.density import DensityPlanner
from beatmorph.tokenizer.vqvae import VQVAETokenizer

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

    stage = cfg.experiment.name
    if stage == "stage1_planner":
        _run_planner(cfg)
    elif stage == "stage_vqvae":
        _run_vqvae(cfg)
    else:
        raise ValueError(
            f"未知 stage: experiment.name={stage!r}（应为 stage1_planner / stage_vqvae）"
        )


def _run_planner(cfg: DictConfig) -> None:
    """Stage1 密度规划训练（Plan 03）。"""
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


def build_vqvae(cfg: DictConfig) -> VQVAETokenizer:
    """从 model/train config 构造 VQVAETokenizer。"""
    m = cfg.model
    cnn_channels = m.get("cnn_channels", [64, 128])
    return VQVAETokenizer(
        codebook_size=int(m.get("codebook_size", 2048)),
        latent=int(m.get("latent", 256)),
        lane=int(m.get("lane", 4)),
        time_bins=int(m.get("time_bins", 64)),
        feat=int(m.get("feat", 6)),
        cnn_channels=tuple(cnn_channels) if cnn_channels else (64, 128),
        transformer_layers=int(m.get("transformer_layers", 2)),
        transformer_heads=int(m.get("transformer_heads", 4)),
        dropout=float(m.get("dropout", 0.1)),
        commit_weight=float(m.get("commit_weight", 0.25)),
        codebook_weight=float(m.get("codebook_weight", 1.0)),
        util_weight=float(m.get("util_weight", 0.1)),
        type_weight=float(m.get("type_weight", 1.0)),
        present_weight=float(m.get("present_weight", 1.0)),
        present_pos_weight=float(m.get("present_pos_weight", 15.0)),
        dur_weight=float(m.get("dur_weight", 0.5)),
        dead_code_steps=int(m.get("dead_code_steps", 500)),
    )


def collate_vqvae(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """padding-aware collate：变长小节数按批次最大值填充，支持 batch>1。

    产出 ``bar_mask [B,bars_max]``（真小节=True，pad 小节=False）供 ``forward``/``loss_fn``
    过滤 padding 小节。padding 小节的 ``bar_grid`` 全 0。
    """
    import torch

    b = len(batch)
    lane = batch[0]["bar_grid"].shape[1]
    time_bins = batch[0]["bar_grid"].shape[2]
    feat = batch[0]["bar_grid"].shape[3]
    bars_max = max(x["bar_grid"].shape[0] for x in batch)

    bar_grid = torch.zeros(b, bars_max, lane, time_bins, feat, dtype=torch.float32)
    bar_mask = torch.zeros(b, bars_max, dtype=torch.bool)
    for i, x in enumerate(batch):
        nb = x["bar_grid"].shape[0]
        bar_grid[i, :nb] = x["bar_grid"]
        bar_mask[i, :nb] = True
    return {"bar_grid": bar_grid, "bar_mask": bar_mask}


def _init_codebook_kmeans(tokenizer: VQVAETokenizer, dataset: VQVAEDataset) -> None:
    """用首批样本 z_e 做 K-means 初始化码本（R-2 防坍缩）。

    取前若干 chart 栅格化 → forward 仅过 encoder 采集 z_e → ``tokenizer.kmeans_init``。
    在 ``trainer.fit`` 前调用，比在第一个 ``training_step`` 内惰性 init 干净。
    """
    import torch

    device = tokenizer.codebook.device
    # 采集样本：尽量凑够 ~2x 码本大小的 z_e（每 chart 若干 bar）
    n_samples = min(len(dataset), max(8, tokenizer.codebook_size * 2 // 8))
    if n_samples == 0:
        logger.warning("K-means init 跳过：数据集空")
        return

    z_es: list[torch.Tensor] = []
    for i in range(n_samples):
        item = dataset[i]
        grid = item["bar_grid"].unsqueeze(0).to(device).to(tokenizer.codebook.dtype)
        # 仅过 encoder 段（forward 末段不跑 quantize/decode）
        b, bars, *_ = grid.shape
        x = grid.reshape(b * bars, tokenizer.lane * tokenizer.feat, tokenizer.time_bins)
        x = tokenizer.encoder_cnn(x)
        x = tokenizer.encoder_pool(x).squeeze(-1)  # (b*bars, latent)
        z_es.append(x)
    z_e_all = torch.cat(z_es, dim=0).detach()  # (M, latent)
    logger.info(
        "K-means init：用 %d 个 z_e 初始化码本 %d", z_e_all.shape[0], tokenizer.codebook_size
    )
    tokenizer.kmeans_init(z_e_all)


def _run_vqvae(cfg: DictConfig) -> None:
    """Stage2 前置 VQ-VAE tokenizer 训练（Plan 02）。"""
    charts, _emb_dir = _resolve_data_paths(cfg)
    dataset = VQVAEDataset(
        charts,
        time_bins=int(cfg.model.get("time_bins", 64)),
        lane=int(cfg.model.get("lane", 4)),
    )
    if len(dataset) == 0:
        raise RuntimeError(
            f"VQVAEDataset 空：charts={charts}。先跑 PreprocessPipeline 产 charts.jsonl。"
        )

    tcfg = cfg.get("train", {})
    loader = DataLoader(
        dataset,
        batch_size=int(tcfg.get("batch_size", 8)),
        shuffle=True,
        collate_fn=collate_vqvae,
        num_workers=int(tcfg.get("num_workers", 0)),
    )

    tokenizer = build_vqvae(cfg)
    # K-means 初始化码本（R-2）
    _init_codebook_kmeans(tokenizer, dataset)
    lit = VQVAELitModule(
        tokenizer,
        lr=float(tcfg.get("lr", 3e-4)),
        weight_decay=float(tcfg.get("weight_decay", 0.01)),
        restart_every_n_steps=int(cfg.model.get("restart_every_n_steps", 200)),
    )
    trainer = build_trainer(cfg)

    logger.info("开始训练：dataset=%d, batch=%d", len(dataset), int(tcfg.get("batch_size", 8)))
    trainer.fit(lit, loader)


if __name__ == "__main__":  # pragma: no cover
    main()
