"""PreprocessPipeline 集成测试。

小规模端到端流水线：raw_dir（3-5 个 .osu 文件）→ parse_osu → 质量过滤
→ compute_section_stats → 落盘。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from beatmorph.data.pipeline.embed import PreprocessPipeline


@pytest.fixture()
def raw_dir() -> Path:
    """创建包含 3 个 .osu 文件的临时目录。"""
    tmpdir = Path(tempfile.mkdtemp())

    # fixture 1: 正常 4K mania
    (tmpdir / "test_4k.osu").write_text(
        "osu file format v14\n"
        "\n"
        "[General]\nMode: 3\n"
        "[Metadata]\nTitle:Test 4K\nArtist:Artist\nCreator:Mapper\nVersion:Hard\n"
        "BeatmapID:1001\nBeatmapSetID:1000\nTags:test 4k\n"
        "[Difficulty]\nCircleSize:4\nOverallDifficulty:6\n"
        "[TimingPoints]\n0,500,4,0,0,100,1,0\n"
        "[HitObjects]\n"
        "64,192,1000,1,0,0:0:0:0:\n"
        "192,192,1500,1,0,0:0:0:0:\n"
        "320,192,2000,128,0,3500,0:0:0:0:\n"
        "448,192,2500,1,0,0:0:0:0:\n",
        encoding="utf-8",
    )

    # fixture 2: 非 mania (taiko)
    (tmpdir / "test_taiko.osu").write_text(
        "osu file format v14\n"
        "[General]\nMode: 1\n"
        "[Metadata]\nTitle:Test Taiko\nArtist:Artist\nCreator:Mapper\nVersion:Oni\n"
        "BeatmapID:1002\nBeatmapSetID:1000\nTags:taiko\n"
        "[Difficulty]\nCircleSize:4\nOverallDifficulty:5\n"
        "[TimingPoints]\n500,400,4,0,0,100,1,0\n"
        "[HitObjects]\n"
        "256,192,1000,1,0,0:0:0:0:\n"
        "256,192,1500,1,0,0:0:0:0:\n",
        encoding="utf-8",
    )

    # fixture 3: 损坏文件（不应中断流水线）
    (tmpdir / "corrupt.osu").write_text("not a valid osu file", encoding="utf-8")

    # fixture 4: 变速 4K mania
    (tmpdir / "test_tempo.osu").write_text(
        "osu file format v14\n"
        "[General]\nMode: 3\n"
        "[Metadata]\nTitle:Test Tempo\nArtist:Artist\nCreator:Mapper\nVersion:Insane\n"
        "BeatmapID:1003\nBeatmapSetID:1000\nTags:tempo change\n"
        "[Difficulty]\nCircleSize:4\nOverallDifficulty:7\n"
        "[TimingPoints]\n"
        "0,500,4,0,0,100,1,0\n"
        "15000,250,4,0,0,100,1,0\n"
        "[HitObjects]\n"
        "64,192,1000,1,0,0:0:0:0:\n"
        "192,192,3000,1,0,0:0:0:0:\n"
        "320,192,5000,128,0,7000,0:0:0:0:\n"
        "448,192,8000,1,0,0:0:0:0:\n"
        "64,192,16000,1,0,0:0:0:0:\n"
        "192,192,18000,1,0,0:0:0:0:\n"
        "320,192,20000,128,0,22000,0:0:0:0:\n"
        "448,192,22000,1,0,0:0:0:0:\n",
        encoding="utf-8",
    )

    return tmpdir


class TestPipelineRun:
    """小规模端到端流水线测试。"""

    def test_run_basic(self, raw_dir: Path) -> None:
        """端到端 run(limit=5) 跑通并产出 parquet。"""
        out_dir = Path(tempfile.mkdtemp())
        pipeline = PreprocessPipeline(raw_dir=raw_dir, out_dir=out_dir)

        pipeline.run(limit=5)

        # 统计检查
        assert pipeline.stats["total"] == 4  # 4 个 .osu 文件
        assert pipeline.stats["parsed"] == 2  # 2 个 mania 正常解析
        assert pipeline.stats["skipped_mode"] >= 2  # taiko(1) + corrupt(0-默认为非mania)
        # 损坏文件不会抛异常：latin-1 解码兜底 + 格式缺失则 Mode=0 空 Chart
        assert pipeline.stats["failed"] == 0

        # 输出验证
        output_files = list(out_dir.iterdir())
        assert len(output_files) >= 1  # 至少一个输出文件

    def test_run_parquet_output(self, raw_dir: Path) -> None:
        """验证 Parquet 产出可被 pyarrow 读回。"""
        try:
            import pyarrow.parquet as pq
        except ImportError:
            pytest.skip("pyarrow not installed")

        out_dir = Path(tempfile.mkdtemp())
        pipeline = PreprocessPipeline(raw_dir=raw_dir, out_dir=out_dir)
        pipeline.run(limit=5)

        parquet_path = out_dir / "charts.parquet"
        assert parquet_path.exists()

        table = pq.read_table(parquet_path)
        assert table.num_rows >= 1
        assert "chart_json" in table.column_names
        assert "num_notes" in table.column_names
        assert "num_sections" in table.column_names

    def test_run_jsonl_fallback(self, raw_dir: Path) -> None:
        """parquet 不可用时 jsonl 兜底。"""
        out_dir = Path(tempfile.mkdtemp())
        pipeline = PreprocessPipeline(raw_dir=raw_dir, out_dir=out_dir)

        # 绕过 pyarrow 检测
        pipeline.cache_format = "jsonl"
        pipeline.run(limit=5)

        jsonl_path = out_dir / "charts.jsonl"
        assert jsonl_path.exists()
        lines = jsonl_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) >= 1

    def test_stats_written(self, raw_dir: Path) -> None:
        """pipeline 运行后 stats 可写入文件。"""
        out_dir = Path(tempfile.mkdtemp())
        pipeline = PreprocessPipeline(raw_dir=raw_dir, out_dir=out_dir)
        pipeline.run(limit=5)
        pipeline.write_stats()

        stats_path = out_dir / "stats.json"
        assert stats_path.exists()
        import json

        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        assert stats["total"] == 4
