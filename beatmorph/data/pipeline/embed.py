"""预处理流水线编排（奠基文档 §4.2）。

原始 (.osu + .mp3)
  → Step 1: parse_osu() → Chart IR + 清理
  → Step 2: compute_section_stats() → Stage 1 伪标签回填
  → Step 3: 音频切片 & MERT 离线预提取 Embedding（Phase 1 预留）
  → Step 4: 构建 VQ-VAE 训练数据集（Phase 1 预留）
  → 存入本地 Parquet
"""

from __future__ import annotations

import json
import typing
from pathlib import Path

from beatmorph.core.contracts import Chart
from beatmorph.core.logging import get_logger
from beatmorph.data.parsers.osu_path import compute_section_stats, parse_osu

if typing.TYPE_CHECKING:
    from beatmorph.data.manifest import ManifestInjector

logger = get_logger(__name__)


class PreprocessPipeline:
    """端到端预处理流水线编排。

    Phase 1 实现 Step1+2；Step3（MERT 提取）和 Step4（VQ-VAE 数据集）
    留 NotImplementedError 接口，待 Plan 01/02 就绪后补充。

    质量过滤（奠基文档 §4.3）：保留 ≥3 星且 play_count > 500 的谱面。
    Phase 1 降级处理：谱面文件不含 difficulty_rating 和 playcount 字段，
    缺失时宽松通过并记 info log。
    """

    MIN_STARS: float = 3.0
    MIN_PLAY_COUNT: int = 500

    def __init__(
        self,
        raw_dir: Path,
        out_dir: Path,
        cache_format: str = "parquet",
        manifest_path: Path | None = None,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.out_dir = Path(out_dir)
        self.cache_format = cache_format

        self.out_dir.mkdir(parents=True, exist_ok=True)

        # ── manifest 注入器（RFC-0024）：激活 §4.3 质量过滤 ──
        # 不传/文件不存在 → None，退回 Phase 1 宽松通过（向后兼容）。
        self._injector: ManifestInjector | None = None
        if manifest_path is not None and Path(manifest_path).exists():
            from beatmorph.data.manifest import build_injector, load_manifest

            self._injector = build_injector(load_manifest(Path(manifest_path)))
            logger.info(
                "Manifest loaded from %s (%d sets) — §4.3 过滤已激活",
                manifest_path,
                len(self._injector),
            )

        # 统计计数
        self.stats: dict[str, int] = {
            "total": 0,
            "parsed": 0,
            "passed_filter": 0,
            "skipped_mode": 0,
            "failed": 0,
        }

    # ── 主入口 ─────────────────────────────────────────────────

    def run(self, limit: int | None = None) -> None:
        """执行完整预处理流水线（Phase 1: Step1+2）。

        Args:
            limit: 最大处理文件数，None = 全部。
        """
        osu_files = sorted(self.raw_dir.rglob("*.osu"))
        if limit is not None:
            osu_files = osu_files[:limit]

        self.stats["total"] = len(osu_files)
        logger.info(
            "PreprocessPipeline.run: %d .osu files found in %s (limit=%s)",
            len(osu_files),
            self.raw_dir,
            limit,
        )

        charts: list[Chart] = []

        for i, path in enumerate(osu_files, start=1):
            try:
                chart = parse_osu(path)

                # 检测跳过的非 mania 谱面
                if chart.meta.get("skip_reason"):
                    self.stats["skipped_mode"] += 1
                    logger.debug(
                        "[%d/%d] Skipped %s (reason=%s)",
                        i,
                        len(osu_files),
                        path.name,
                        chart.meta["skip_reason"],
                    )
                    continue

                self.stats["parsed"] += 1

                # ── manifest 注入（RFC-0024）：把集级 difficulty_rating/playcount
                #    注入 Chart.meta，使 §4.3 过滤真正生效；不传 manifest 时 no-op。
                if self._injector is not None:
                    self._injector.apply(chart.meta)
                # §6 版权标记——setdefault 合并，绝不覆盖既有 meta 键。
                chart.meta.setdefault("license", "academic")

                # ── 质量过滤（§4.3）──
                if not self._quality_filter(chart, path):
                    continue

                self.stats["passed_filter"] += 1

                # ── 注入音频时长（统一 Section 边界口径，待决项 2）──
                # compute_section_stats 优先用音频全长切 Section，与推理时 plan()/
                # extract_mert 的边界同源；无音频则 fallback 到 Note 时长（向后兼容）。
                if chart.audio_path and "audio_duration" not in chart.meta:
                    dur = _audio_duration(path.parent / chart.audio_path)
                    if dur is not None:
                        chart.meta["audio_duration"] = dur

                # ── Step 2: 统计量 ──
                chart = compute_section_stats(chart)

                charts.append(chart)

            except Exception:
                self.stats["failed"] += 1
                logger.exception("[%d/%d] Failed to process %s", i, len(osu_files), path)

        logger.info(
            "Pipeline complete: total=%d parsed=%d passed=%d skipped_mode=%d failed=%d",
            self.stats["total"],
            self.stats["parsed"],
            self.stats["passed_filter"],
            self.stats["skipped_mode"],
            self.stats["failed"],
        )

        # ── 落盘 ──
        if charts:
            self._save(charts)
        else:
            logger.warning("No charts passed filtering, nothing to save.")

    # ── Step 3: MERT 离线提取（Phase 1 预留）─────────────────

    def extract_mert_embeddings(
        self,
        audio_dir: Path,
        *,
        encoder: object | None = None,
        out_dir: Path | None = None,
        source: str = "modelscope",
        device: str = "auto",
        limit: int | None = None,
    ) -> None:
        """Step 3：批量离线提取 MERT Embedding，节省训练时算力（§4.2 Step3）。

        遍历 ``audio_dir`` 下所有 ``.osu``（rglob）→ ``parse_osu`` 取 ``Chart`` →
        相对 ``.osu`` 父目录解析 ``chart.audio_path``（basename）→ 加载+重采样
        → ``MERTAdapter.encode`` → ``[T_seq, feat]`` 落盘（断点续抽，已存在跳过）。
        按 ``beatmap_set_id`` 去冗余：同 set 多难度共享同一音频，只提取一份 ``{sid}.pt``。

        Args:
            audio_dir: 含 ``{sid}/*.osu`` + 音频的根目录（= :attr:`raw_dir`）。
            encoder: 已构造的 :class:`MERTAdapter`（避免每曲重建）；None 则首次惰性构造。
            out_dir: embedding 落盘目录（默认 ``BEATMORPH_DATA_DIR/embeddings``）。
            source / device: 构造 encoder 时的权重来源 / 设备。
            limit: 最大处理文件数，None=全部。
        """
        out_dir = Path(out_dir) if out_dir else self._default_embed_dir(audio_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        osu_files = sorted(Path(audio_dir).rglob("*.osu"))
        if limit is not None:
            osu_files = osu_files[:limit]

        logger.info(
            "extract_mert_embeddings: %d .osu in %s → %s",
            len(osu_files),
            audio_dir,
            out_dir,
        )

        enc = encoder
        n_ok = n_skip = n_fail = 0
        # sid → 首次提取该 set 时用的音频 basename，用于检测「同 set 不同音频」边界
        sid_first_audio: dict[str, str] = {}
        for i, path in enumerate(osu_files, start=1):
            try:
                chart = parse_osu(path)
                if chart.meta.get("skip_reason") or not chart.audio_path:
                    n_skip += 1
                    continue

                # 按 beatmap_set_id 去冗余：同 set 多难度共享同一 audio.mp3，
                # 只为每个 set 提取一份 embedding ({sid}.pt)，避免 N× 冗余 (2.4× 经验值)。
                sid = chart.meta.get("beatmap_set_id")
                if sid is None:
                    # 无 set_id 兜底：回退 beatmap_id（不享去重，但保证可配对）
                    key = str(chart.meta.get("beatmap_id", path.stem))
                else:
                    key = str(sid)
                emb_path = out_dir / f"{key}.pt"
                if emb_path.exists():
                    # 同 set 已提过 → 跳过（去重核心）。检测同 set 不同音频边界并 warn。
                    if sid is not None and chart.audio_path:
                        first_audio = sid_first_audio.get(key)
                        if first_audio is not None and first_audio != chart.audio_path:
                            logger.warning(
                                "Set %s: 多个 .osu 引用不同音频（%s vs 首次 %s），"
                                "已按首次提取，后续忽略（RFC-0024 边界，Phase1 罕见）",
                                key,
                                chart.audio_path,
                                first_audio,
                            )
                    n_skip += 1
                    continue
                if sid is not None and chart.audio_path:
                    sid_first_audio[key] = chart.audio_path

                wav = _load_audio_resampled(path.parent / chart.audio_path)
                if wav is None:
                    n_fail += 1
                    logger.warning("[%d/%d] no audio for %s", i, len(osu_files), path.name)
                    continue

                if enc is None:
                    enc = _build_mert_encoder(source=source, device=device)

                import torch

                wav_t = torch.from_numpy(wav).unsqueeze(0)  # [1, samples]
                with torch.inference_mode():
                    emb = enc.encode(wav_t)  # [1, T_seq, 768]

                torch.save(emb.squeeze(0).cpu().float(), emb_path)  # [T_seq, 768]
                n_ok += 1
                logger.debug(
                    "[%d/%d] %s → %s %s",
                    i,
                    len(osu_files),
                    path.name,
                    tuple(emb.shape),
                    emb_path.name,
                )
            except Exception:
                n_fail += 1
                logger.exception("[%d/%d] failed %s", i, len(osu_files), path)

        logger.info(
            "MERT extraction done: ok=%d skip=%d fail=%d (out=%s)",
            n_ok,
            n_skip,
            n_fail,
            out_dir,
        )

    def _default_embed_dir(self, audio_dir: Path) -> Path:
        import os

        base = os.environ.get("BEATMORPH_DATA_DIR", "data/processed")
        return Path(base) / "embeddings" / "mert_v1_330m"

    # ── 内部方法 ──────────────────────────────────────────────

    def _quality_filter(self, chart: Chart, path: Path) -> bool:
        """质量过滤：≥3 星 且 play_count > 500。

        Phase 1 降级：
            - difficulty_rating 从 chart.meta 查（外部注入），缺失宽松通过
            - playcount 从 chart.meta 查，缺失宽松通过
        """
        rating = chart.meta.get("difficulty_rating")
        playcount = chart.meta.get("playcount")

        if rating is not None and isinstance(rating, (int, float)) and rating < self.MIN_STARS:
            logger.debug("Filtered out %s: stars=%.1f < %.1f", path.name, rating, self.MIN_STARS)
            return False

        if (
            playcount is not None
            and isinstance(playcount, (int, float))
            and playcount <= self.MIN_PLAY_COUNT
        ):
            logger.debug(
                "Filtered out %s: playcount=%d <= %d",
                path.name,
                int(playcount),
                self.MIN_PLAY_COUNT,
            )
            return False

        if rating is None:
            logger.info("No difficulty_rating in %s, passing (Phase 1 relaxed filter)", path.name)
        if playcount is None:
            logger.info("No playcount in %s, passing (Phase 1 relaxed filter)", path.name)

        return True

    def _save(self, charts: list[Chart]) -> None:
        """落盘：将 Chart 列表序列化为 Parquet。

        每行一个 Chart 的 JSON dump，附加 chart_id 列。
        """
        if self.cache_format == "parquet":
            self._save_parquet(charts)
        else:
            logger.warning("Unsupported cache_format=%s, falling back to jsonl", self.cache_format)
            self._save_jsonl(charts)

    def _save_parquet(self, charts: list[Chart]) -> None:
        """写入 Parquet（pyarrow 必需）。"""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            logger.warning("pyarrow not installed, falling back to jsonl")
            self._save_jsonl(charts)
            return

        rows: list[dict] = []
        for chart in charts:
            rows.append(
                {
                    "chart_json": chart.model_dump_json(),
                    "title": chart.title,
                    "artist": chart.artist,
                    "difficulty": chart.difficulty,
                    "mode": int(chart.mode),
                    "bpm": chart.primary_bpm(),
                    "num_notes": len(chart.notes),
                    "num_sections": len(chart.sections),
                    "num_bpm_points": len(chart.bpm_points),
                }
            )

        table = pa.Table.from_pylist(rows)
        out_path = self.out_dir / "charts.parquet"
        pq.write_table(table, out_path)
        logger.info("Saved %d charts to %s", len(charts), out_path)

    def _save_jsonl(self, charts: list[Chart]) -> None:
        """JSONL 兜底写入。"""
        out_path = self.out_dir / "charts.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for chart in charts:
                f.write(chart.model_dump_json() + "\n")
        logger.info("Saved %d charts to %s (jsonl)", len(charts), out_path)

    def write_stats(self) -> None:
        """将统计结果写入 out_dir/stats.json。"""
        stats_path = self.out_dir / "stats.json"
        stats_path.write_text(json.dumps(self.stats, indent=2), encoding="utf-8")
        logger.info("Pipeline stats written to %s", stats_path)


# ── 模块级辅助：音频加载 + MERT 编码器构造 ──────────────────────


def _audio_duration(audio_path: Path) -> float | None:
    """读音频时长（秒），只读头不解码（soundfile.info，轻量）。

    供 PreprocessPipeline 注入 ``chart.meta["audio_duration"]``，使
    ``compute_section_stats`` 的 Section 边界与推理时（音频全长）同源。
    soundfile 缺失或读失败 → None（调用方 fallback 到 Note 时长）。
    """
    if not audio_path.exists():
        logger.debug("audio file not found for duration: %s", audio_path)
        return None
    try:
        import soundfile as sf

        info = sf.info(str(audio_path))
        return float(info.frames / info.samplerate)
    except Exception as exc:
        logger.debug("soundfile.info failed for %s: %s", audio_path, exc)
        return None


def _load_audio_resampled(audio_path: Path, target_sr: int = 24000) -> object | None:
    """加载音频文件并重采样为目标采样率单声道，返回 numpy float32 ``[samples]``。

    优先 torchaudio（已装），fallback soundfile + 线性重采样。失败返回 None。
    """
    import numpy as np

    if not audio_path.exists():
        logger.warning("audio file not found: %s", audio_path)
        return None

    # ── 优先 torchaudio（自带重采样）──
    try:
        import torchaudio

        wav, sr = torchaudio.load(str(audio_path))  # [C, samples], float
        wav = wav.mean(dim=0)  # 单声道 [samples]
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        return wav.numpy().astype(np.float32)
    except Exception as exc:
        logger.debug("torchaudio load failed (%s), trying soundfile", exc)

    # ── fallback: soundfile + 线性重采样 ──
    try:
        import soundfile as sf

        data, sr = sf.read(str(audio_path), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != target_sr:
            import numpy as np

            n_out = round(len(data) * target_sr / sr)
            idx = np.linspace(0, len(data) - 1, n_out)
            data = np.interp(idx, np.arange(len(data)), data).astype(np.float32)
        return data
    except Exception as exc:
        logger.warning("soundfile load also failed for %s: %s", audio_path, exc)
        return None


def _build_mert_encoder(*, source: str = "modelscope", device: str = "auto") -> object:
    """惰性构造 MERTAdapter（adapter='none' 用于离线提取表征）。"""
    from beatmorph.audio.encoder.mert import MERTAdapter

    return MERTAdapter(
        adapter="none",  # 离线提取只取主干表征，不训 Adapter
        source=source,
        device=device,
        fp16=(device != "cpu"),
    )
