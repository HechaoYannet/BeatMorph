"""PreprocessPipeline + manifest 注入激活 §4.3 过滤的集成（离线）。

构造 2 个 mania 4K .osu：一个 BeatmapSetID 高星(7.0) 高 play(2000)，
一个 BeatmapSetID 低星(2.0) 低 play(10)。写 2 行 manifest，传 manifest_path
跑 run()，断言仅高星 set 进入 parquet（低星被 §4.3 拒）。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from beatmorph.data.pipeline.embed import PreprocessPipeline


def _mania_osu(beatmap_id: int, beatmap_set_id: int, version: str) -> str:
    """构造一个合法 mania 4K .osu 文本。"""
    return (
        "osu file format v14\n"
        "\n"
        "[General]\nMode: 3\nAudioFilename: audio.mp3\n"
        "[Metadata]\n"
        f"Title:T{beatmap_set_id}\nArtist:A\nCreator:M\nVersion:{version}\n"
        f"BeatmapID:{beatmap_id}\nBeatmapSetID:{beatmap_set_id}\nTags:test\n"
        "[Difficulty]\nCircleSize:4\nOverallDifficulty:7\n"
        "[TimingPoints]\n0,500,4,0,0,100,1,0\n"
        "[HitObjects]\n"
        "64,192,1000,1,0,0:0:0:0:\n"
        "192,192,1500,1,0,0:0:0:0:\n"
        "320,192,2000,128,0,2500,0:0:0:0:\n"
        "448,192,2500,1,0,0:0:0:0:\n"
    )


def _manifest_line(sid: int, stars: float, play_count: int, approved: int = 1) -> str:
    return json.dumps(
        {
            "sid": sid,
            "title": f"T{sid}",
            "artist": "A",
            "creator": "M",
            "stars": stars,
            "play_count": play_count,
            "approved": approved,
            "modes": 8,
            "downloaded_at": "2026-07-31T00:00:00",
            "n_diffs": 1,
            "path": str(sid),
            "unranked": stars == 0.0 or approved == 3,
        },
        ensure_ascii=False,
    )


@pytest.fixture
def raw_dir() -> Path:
    tmpdir = Path(tempfile.mkdtemp())
    # 高星集（应通过 ≥3 & >500）
    (tmpdir / "high.osu").write_text(
        _mania_osu(beatmap_id=101, beatmap_set_id=1000, version="Insane"),
        encoding="utf-8",
    )
    # 低星集（应被拒：<3 或 ≤500）
    (tmpdir / "low.osu").write_text(
        _mania_osu(beatmap_id=201, beatmap_set_id=2000, version="Easy"),
        encoding="utf-8",
    )
    return tmpdir


@pytest.fixture
def manifest_path(raw_dir: Path) -> Path:
    path = raw_dir.parent / "manifest.jsonl"
    path.write_text(
        _manifest_line(sid=1000, stars=7.0, play_count=2000)
        + "\n"
        + _manifest_line(sid=2000, stars=2.0, play_count=10)
        + "\n",
        encoding="utf-8",
    )
    return path


class TestPipelineManifestFilter:
    def test_manifest_activates_filter(
        self,
        raw_dir: Path,
        manifest_path: Path,
    ) -> None:
        """传 manifest → §4.3 激活，仅高星 set 进 parquet。"""
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            pytest.skip("pyarrow not installed")

        out_dir = Path(tempfile.mkdtemp())
        pipeline = PreprocessPipeline(
            raw_dir=raw_dir,
            out_dir=out_dir,
            manifest_path=manifest_path,
        )
        assert pipeline._injector is not None  # 注入器已加载

        pipeline.run()

        # total=2（两个 mania），parsed=2，passed_filter=1（仅高星）
        assert pipeline.stats["total"] == 2
        assert pipeline.stats["parsed"] == 2
        assert pipeline.stats["passed_filter"] == 1
        assert pipeline.stats["failed"] == 0

        # 验证 parquet 仅含高星 set
        import pyarrow.parquet as pq

        table = pq.read_table(out_dir / "charts.parquet")
        assert table.num_rows == 1

        row = json.loads(table.column("chart_json").to_pylist()[0])
        assert row["meta"]["beatmap_set_id"] == 1000
        assert row["meta"]["difficulty_rating"] == 7.0
        assert row["meta"]["playcount"] == 2000
        assert row["meta"]["license"] == "academic"

    def test_no_manifest_relaxed_pass(self, raw_dir: Path) -> None:
        """不传 manifest → 宽松通过，两个 set 都进 parquet（向后兼容）。"""
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            pytest.skip("pyarrow not installed")

        out_dir = Path(tempfile.mkdtemp())
        pipeline = PreprocessPipeline(raw_dir=raw_dir, out_dir=out_dir)
        assert pipeline._injector is None  # 无 manifest → 不激活

        pipeline.run()

        assert pipeline.stats["passed_filter"] == 2  # 宽松通过
        import pyarrow.parquet as pq

        table = pq.read_table(out_dir / "charts.parquet")
        assert table.num_rows == 2

    def test_license_marked_for_all(self, raw_dir: Path, manifest_path: Path) -> None:
        """license=academic 标记应出现在通过的 chart meta。用 jsonl 兜底避免 pyarrow 依赖。"""
        out_dir = Path(tempfile.mkdtemp())
        pipeline = PreprocessPipeline(
            raw_dir=raw_dir,
            out_dir=out_dir,
            cache_format="jsonl",
            manifest_path=manifest_path,
        )
        pipeline.run()

        jsonl = (out_dir / "charts.jsonl").read_text(encoding="utf-8").strip().split("\n")
        assert len(jsonl) == 1  # 仅高星
        chart = json.loads(jsonl[0])
        assert chart["meta"]["license"] == "academic"
