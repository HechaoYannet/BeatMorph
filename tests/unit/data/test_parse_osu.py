"""parse_osu() 清理规则与兜底单元测试。"""

from __future__ import annotations

import tempfile
from pathlib import Path

from beatmorph.data.parsers.osu_path import parse_osu


def _make_osu_content(
    mode: int = 3,
    circle_size: str = "4",
    timing_points: list[str] | None = None,
    hit_objects: list[str] | None = None,
) -> str:
    lines = [
        "osu file format v14",
        "",
        "[General]",
        f"Mode: {mode}",
        "",
        "[Metadata]",
        "Title:Test",
        "Artist:Test",
        "Creator:Test",
        "Version:Hard",
        "BeatmapID:1",
        "BeatmapSetID:1",
        "Tags:",
        "",
        "[Difficulty]",
        f"CircleSize:{circle_size}",
        "OverallDifficulty:5",
        "",
        "[Events]",
        "",
    ]
    if timing_points is not None:
        lines.append("[TimingPoints]")
        lines.extend(timing_points)
    if hit_objects is not None:
        lines.append("")
        lines.append("[HitObjects]")
        lines.extend(hit_objects)
    return "\n".join(lines) + "\n"


def _write_temp_osu(content: str) -> Path:
    import os

    fd, name = tempfile.mkstemp(suffix=".osu", text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return Path(name)


class TestParseOsuCleanup:
    """清理规则：负时间、越界、空文件。"""

    def test_drop_negative_time(self) -> None:
        content = _make_osu_content(
            timing_points=["0,500,4,0,0,100,1,0"],
            hit_objects=[
                "64,192,-500,1,0,0:0:0:0:",  # 负时间 → 丢弃
                "64,192,1000,1,0,0:0:0:0:",  # 正常
            ],
        )
        path = _write_temp_osu(content)
        try:
            chart = parse_osu(path)
            assert len(chart.notes) == 1
            assert chart.notes[0].time == 1.0
        finally:
            path.unlink(missing_ok=True)

    def test_drop_oob_lane(self) -> None:
        """越界 lane (>= keyCount) 应丢弃。"""
        content = _make_osu_content(
            circle_size="4",
            timing_points=["0,500,4,0,0,100,1,0"],
            hit_objects=[
                "600,192,1000,1,0,0:0:0:0:",  # lane=4，越界 → 丢弃
                "200,192,1500,1,0,0:0:0:0:",  # lane=1，正常
                "64,192,2000,1,0,0:0:0:0:",  # lane=0，正常
            ],
        )
        path = _write_temp_osu(content)
        try:
            chart = parse_osu(path)
            assert len(chart.notes) == 2
            for n in chart.notes:
                assert 0 <= n.lane < 4
        finally:
            path.unlink(missing_ok=True)

    def test_empty_file_returns_empty_chart(self) -> None:
        """空文件（无 HitObjects）返回 0 个 Note。"""
        content = _make_osu_content(
            timing_points=["0,500,4,0,0,100,1,0"],
            hit_objects=[],
        )
        path = _write_temp_osu(content)
        try:
            chart = parse_osu(path)
            assert len(chart.notes) == 0
        finally:
            path.unlink(missing_ok=True)

    def test_corrupt_file_returns_empty_chart(self) -> None:
        """损坏文件（无 .osu 结构）不抛异常，返回空 Chart。"""
        fd, name = tempfile.mkstemp(suffix=".osu", text=True)
        import os

        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("this is not a valid osu file\n")
        path = Path(name)
        try:
            chart = parse_osu(path)
            assert len(chart.notes) == 0
        finally:
            path.unlink(missing_ok=True)

    def test_fallback_bpm_when_no_red_line(self) -> None:
        """无红线 TimingPoint → fallback BPM=120。"""
        content = _make_osu_content(
            timing_points=["0,-50,4,0,0,100,0,0"],  # 仅绿线
            hit_objects=["64,192,1000,1,0,0:0:0:0:"],
        )
        path = _write_temp_osu(content)
        try:
            chart = parse_osu(path)
            assert len(chart.bpm_points) == 1
            assert abs(chart.bpm_points[0].bpm - 120.0) < 0.01
        finally:
            path.unlink(missing_ok=True)


class TestParseOsuEdgeCases:
    """边界情况。"""

    def test_all_notes_in_bounds(self) -> None:
        """所有 Note 都在合法范围内，全部保留。"""
        content = _make_osu_content(
            timing_points=["0,500,4,0,0,100,1,0"],
            hit_objects=[
                "64,192,1000,1,0,0:0:0:0:",
                "192,192,1100,1,0,0:0:0:0:",
                "320,192,1200,1,0,0:0:0:0:",
                "448,192,1300,1,0,0:0:0:0:",
            ],
        )
        path = _write_temp_osu(content)
        try:
            chart = parse_osu(path)
            assert len(chart.notes) == 4
        finally:
            path.unlink(missing_ok=True)

    def test_negative_time_at_boundary(self) -> None:
        """t=0 的 Note 保留，t<0 丢弃。"""
        content = _make_osu_content(
            timing_points=["0,500,4,0,0,100,1,0"],
            hit_objects=[
                "64,192,0,1,0,0:0:0:0:",  # 保留（=0）
                "64,192,-1,1,0,0:0:0:0:",  # 丢弃（<0）
            ],
        )
        path = _write_temp_osu(content)
        try:
            chart = parse_osu(path)
            assert len(chart.notes) == 1
        finally:
            path.unlink(missing_ok=True)
