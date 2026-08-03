"""训练数据集加载器（Plan 09 / Plan 03 / Plan 01）。

- :class:`PlannerDataset`：Stage1 密度规划训练数据。每样本 yield
  ``(audio_emb, target, difficulty, section_bounds)``：从 ``charts.jsonl`` 读
  ``Chart``（含 ``compute_section_stats`` 产出的 sections 伪标签），配对同
  ``beatmap_set_id`` 的离线 MERT embedding ``.pt``（同 set 多难度共享一份去冗余）。
- :class:`MERTExtractionDataset`：离线提取用，yield ``(.osu_path, audio_path)``。

数据来源：PreprocessPipeline 的 jsonl/parquet 产出（含 sections）+
``extract_mert_embeddings`` 产的 ``{beatmap_set_id}.pt``（同 set 共享一份）。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from torch.utils.data import Dataset

from beatmorph.core.contracts import Chart, GameMode
from beatmorph.core.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

_TYPE_TO_IDX = {"intro": 0, "verse": 1, "chorus": 2, "bridge": 3, "outro": 4}


class PlannerDataset(Dataset):
    """Stage1 密度规划训练数据集。

    Args:
        charts_path: ``charts.jsonl``（PreprocessPipeline 产出，每行一个 Chart JSON）。
        embeddings_dir: ``{beatmap_set_id}.pt`` 目录（同 set 共享一份）（extract_mert_embeddings 产出）。
        section_bars: Section 小节数（与 planner 一致，默认 4）。

    每样本：
        - ``audio_emb``: ``[T_seq, 768]`` float（从 .pt 读）
        - ``density/energy/rest``: ``[S]`` float [0,1]
        - ``type``: ``[S]`` long 0..4
        - ``difficulty``: int
        - ``section_bounds``: ``[S+1]`` float 秒
    跳过缺 embedding / 无 sections 的样本。
    """

    def __init__(
        self,
        charts_path: Path,
        embeddings_dir: Path,
        section_bars: int = 4,
    ) -> None:
        self.charts_path = Path(charts_path)
        self.embeddings_dir = Path(embeddings_dir)
        self.section_bars = section_bars
        self._samples: list[dict] = self._load_index()

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        import torch

        entry = self._samples[idx]
        emb = torch.load(entry["emb_path"], map_location="cpu", weights_only=True)
        if emb.dim() == 3 and emb.shape[0] == 1:
            emb = emb.squeeze(0)
        elif emb.dim() != 2:
            raise ValueError(f"unexpected emb shape {tuple(emb.shape)} for {entry['emb_path']}")

        secs = entry["sections"]
        density = torch.tensor([s["density_target"] for s in secs], dtype=torch.float32)
        energy = torch.tensor([s["energy_level"] for s in secs], dtype=torch.float32)
        rest = torch.tensor([s["rest_probability"] for s in secs], dtype=torch.float32)
        type_idx = torch.tensor(
            [_TYPE_TO_IDX.get(s["sections_type"], 1) for s in secs],
            dtype=torch.long,
        )
        bounds = torch.tensor(entry["bounds"], dtype=torch.float32)
        difficulty = torch.tensor(entry["difficulty"], dtype=torch.long)

        return {
            "audio_emb": emb,
            "target": {"density": density, "energy": energy, "rest": rest, "type": type_idx},
            "difficulty": difficulty,
            "section_bounds": bounds,
        }

    # ── 索引构建 ──────────────────────────────────────────────

    def _load_index(self) -> list[dict]:
        samples: list[dict] = []
        if not self.charts_path.exists():
            logger.warning("charts file not found: %s", self.charts_path)
            return samples

        n_missing = 0
        n_non_4k = 0
        with self.charts_path.open("r", encoding="utf-8") as f:
            for ln, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    chart = Chart.model_validate_json(line)
                except Exception as exc:
                    logger.debug("skip invalid chart line %d: %s", ln, exc)
                    continue

                # Stage1 训练只吃 4K（守 R-6，RFC-0025）。多 K 数据虽落盘但
                # 在此过滤，4K 主路径不污染；未来多 K 扩展由独立工程重写。
                if chart.mode != GameMode.MANIA_4K:
                    n_non_4k += 1
                    continue

                if not chart.sections:
                    continue

                # embedding 按 beatmap_set_id 存储（同 set 多难度共享一份，见 embed.py）
                sid = chart.meta.get("beatmap_set_id")
                key = (
                    str(sid)
                    if sid is not None
                    else chart.meta.get("beatmap_id", self.charts_path.stem)
                )
                emb_path = self.embeddings_dir / f"{key}.pt"
                if not emb_path.exists():
                    n_missing += 1
                    continue

                # Section 时间边界（秒）
                bounds = [chart.sections[0].start_time]
                bounds.extend(s.end_time for s in chart.sections)
                samples.append(
                    {
                        "emb_path": emb_path,
                        "sections": [s.model_dump() for s in chart.sections],
                        "difficulty": chart.difficulty,
                        "bounds": bounds,
                    }
                )

        logger.info(
            "PlannerDataset: %d samples (skipped: %d no-emb, %d non-4K)",
            len(samples),
            n_missing,
            n_non_4k,
        )
        return samples


class MERTExtractionDataset(Dataset):
    """离线 MERT 提取用索引：yield (.osu path, audio path)。"""

    def __init__(self, raw_dir: Path) -> None:
        from beatmorph.data.parsers.osu_path import parse_osu

        self.raw_dir = Path(raw_dir)
        self._items: list[tuple[Path, Path]] = []
        for osu in sorted(self.raw_dir.rglob("*.osu")):
            try:
                chart = parse_osu(osu)
            except Exception:
                continue
            if chart.meta.get("skip_reason") or not chart.audio_path:
                continue
            audio = osu.parent / chart.audio_path
            if audio.exists():
                self._items.append((osu, audio))
        logger.info("MERTExtractionDataset: %d osu+audio pairs", len(self._items))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> tuple[Path, Path]:
        return self._items[idx]
