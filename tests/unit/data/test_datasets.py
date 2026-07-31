"""数据集 + 离线提取循环单元测试（离线，mock encoder，不需权重/GPU）。

覆盖 PlannerDataset 索引/getitem 形状、MERTExtractionDataset 配对、
extract_mert_embeddings 用 mock encoder 端到端产 .pt。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from beatmorph.core.contracts import BpmPoint, Chart, Section
from beatmorph.data.datasets import MERTExtractionDataset, PlannerDataset

# ── 夹具构造 ──────────────────────────────────────────────────


def _chart_json(beatmap_id: int, n_secs: int = 3) -> str:
    """构造一个带 sections 的 Chart JSON（含 beatmap_id + sections）。"""
    secs = [
        Section(
            index=i,
            start_time=float(i * 8.0),
            end_time=float((i + 1) * 8.0),
            bar_count=4,
            density_target=0.3 + 0.1 * i,
            energy_level=0.4,
            rest_probability=0.1,
            sections_type=("intro" if i == 0 else "verse"),
        )
        for i in range(n_secs)
    ]
    chart = Chart(
        mode=__import__("beatmorph.core.contracts", fromlist=["GameMode"]).GameMode.MANIA_4K,
        difficulty=8,
        bpm_points=[BpmPoint(time=0.0, bpm=120.0)],
        audio_path="audio.mp3",
        title="t",
        artist="a",
        notes=[],
        sections=secs,
        meta={"beatmap_id": beatmap_id, "beatmap_set_id": 1000, "creator": "x"},
    )
    return chart.model_dump_json()


def _write_charts(path: Path, bids: list[int]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for bid in bids:
            f.write(_chart_json(bid) + "\n")


class TestPlannerDataset:
    def test_index_and_getitem(self, tmp_path: Path) -> None:
        charts = tmp_path / "charts.jsonl"
        emb_dir = tmp_path / "emb"
        emb_dir.mkdir()
        bids = [101, 102]
        _write_charts(charts, bids)
        for bid in bids:
            torch.save(torch.randn(50, 768), emb_dir / f"{bid}.pt")

        ds = PlannerDataset(charts, emb_dir, section_bars=4)
        assert len(ds) == 2

        item = ds[0]
        assert item["audio_emb"].shape == (50, 768)
        assert item["target"]["density"].shape == (3,)
        assert item["target"]["type"].shape == (3,)
        assert item["target"]["type"].dtype == torch.long
        assert item["difficulty"].item() == 8
        assert item["section_bounds"].shape == (4,)  # S+1 = 3+1

    def test_skips_missing_embedding(self, tmp_path: Path) -> None:
        charts = tmp_path / "charts.jsonl"
        emb_dir = tmp_path / "emb"
        emb_dir.mkdir()
        _write_charts(charts, [101, 102])
        torch.save(torch.randn(50, 768), emb_dir / "101.pt")  # 102 缺

        ds = PlannerDataset(charts, emb_dir)
        assert len(ds) == 1  # 仅 101

    def test_skips_no_sections(self, tmp_path: Path) -> None:
        charts = tmp_path / "charts.jsonl"
        emb_dir = tmp_path / "emb"
        emb_dir.mkdir()
        # 写一个无 sections 的 chart（手动清空 sections）
        c = Chart(difficulty=1, bpm_points=[BpmPoint(time=0.0, bpm=120.0)])
        with charts.open("w", encoding="utf-8") as f:
            f.write(c.model_dump_json() + "\n")
        torch.save(torch.randn(50, 768), emb_dir / "0.pt")

        ds = PlannerDataset(charts, emb_dir)
        assert len(ds) == 0

    def test_missing_charts_file(self, tmp_path: Path) -> None:
        ds = PlannerDataset(tmp_path / "nope.jsonl", tmp_path / "emb")
        assert len(ds) == 0


class TestMERTExtractionDataset:
    def test_pairs_osu_and_audio(self, tmp_path: Path) -> None:
        # 构造一个最小 mania .osu + 一个假 audio
        osu_dir = tmp_path / "1000"
        osu_dir.mkdir()
        (osu_dir / "diff.osu").write_text(
            "osu file format v14\n"
            "[General]\nMode: 3\nAudioFilename: audio.mp3\n"
            "[Metadata]\nTitle:T\nArtist:A\nCreator:M\nVersion:V\n"
            "BeatmapID:1\nBeatmapSetID:1000\n"
            "[Difficulty]\nCircleSize:4\nOverallDifficulty:7\n"
            "[TimingPoints]\n0,500,4,0,0,100,1,0\n"
            "[HitObjects]\n64,192,1000,1,0,0:0:0:0:\n",
            encoding="utf-8",
        )
        (osu_dir / "audio.mp3").write_bytes(b"fake")  # 存在即可触发配对

        ds = MERTExtractionDataset(tmp_path)
        assert len(ds) == 1
        osu, audio = ds[0]
        assert osu.name == "diff.osu"
        assert audio.name == "audio.mp3"


class TestExtractMertEmbeddingsMock:
    """用 mock encoder 跑 extract_mert_embeddings 离线循环，不需真实 MERT。"""

    def test_extracts_and_skips_existing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:

        # 一个短 wav（用 soundfile 写一个真 wav 避免 torchaudio 解码问题）
        import numpy as np

        try:
            import soundfile as sf
        except ImportError:
            pytest.skip("soundfile not installed")
        osu_dir = tmp_path / "raw" / "1000"
        osu_dir.mkdir(parents=True)
        (osu_dir / "diff.osu").write_text(
            "osu file format v14\n"
            "[General]\nMode: 3\nAudioFilename: tone.wav\n"
            "[Metadata]\nTitle:T\nArtist:A\nCreator:M\nVersion:V\n"
            "BeatmapID:5\nBeatmapSetID:1000\n"
            "[Difficulty]\nCircleSize:4\nOverallDifficulty:7\n"
            "[TimingPoints]\n0,500,4,0,0,100,1,0\n"
            "[HitObjects]\n64,192,1000,1,0,0:0:0:0:\n",
            encoding="utf-8",
        )
        # 1s 16kHz 正弦 wav
        sr = 16000
        wav = (np.sin(2 * np.pi * 440 * np.arange(sr) / sr) * 0.3).astype(np.float32)
        sf.write(str(osu_dir / "tone.wav"), wav, sr)

        # mock encoder：返回 [1, T_seq, 768]
        class _MockEnc:
            def __init__(self) -> None:
                self.calls = 0

            def encode(self, w: torch.Tensor) -> torch.Tensor:
                self.calls += 1
                dur = w.shape[-1] / 16000.0
                return torch.randn(1, max(1, round(dur * 25)), 768)

        mock = _MockEnc()
        out_dir = tmp_path / "emb"
        from beatmorph.data.pipeline.embed import PreprocessPipeline

        pipe = PreprocessPipeline(tmp_path / "raw", tmp_path / "processed")
        pipe.extract_mert_embeddings(
            tmp_path / "raw",
            encoder=mock,
            out_dir=out_dir,
            device="cpu",
        )
        emb_path = out_dir / "5.pt"
        assert emb_path.exists()
        emb = torch.load(emb_path, weights_only=True)
        assert emb.dim() == 2
        assert emb.shape[-1] == 768
        assert mock.calls == 1

        # 二次跑：断点续抽，已存在跳过（mock 不应再被调）
        pipe.extract_mert_embeddings(
            tmp_path / "raw",
            encoder=mock,
            out_dir=out_dir,
            device="cpu",
        )
        assert mock.calls == 1  # 仍 1，未重抽
