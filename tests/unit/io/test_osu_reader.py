"""OsuManiaReader 单元测试。

覆盖：各 section 解析、HitObject type bitmask 组合、TimingPoints 过滤、编码兜底、非 mania 模式拒绝。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from beatmorph.core.contracts import Chart, GameMode, NoteType
from beatmorph.io.formats.osu import OsuManiaReader

# ── 辅助：构造最小 .osu 文件内容 ──────────────────────────────


def _make_osu_content(
    *,
    mode: int = 3,
    title: str = "Test Song",
    artist: str = "Test Artist",
    creator: str = "TestCreator",
    version: str = "Hard",
    beatmap_id: int = 123456,
    beatmap_set_id: int = 7890,
    circle_size: str = "4",
    overall_difficulty: str = "6",
    timing_points: list[str] | None = None,
    hit_objects: list[str] | None = None,
) -> str:
    """构造合法的 .osu 文件内容字符串。"""
    lines = [
        "osu file format v14",
        "",
        "[General]",
        "AudioFilename: audio.mp3",
        "AudioLeadIn: 0",
        "PreviewTime: -1",
        f"Mode: {mode}",
        "",
        "[Metadata]",
        f"Title:{title}",
        f"Artist:{artist}",
        f"Creator:{creator}",
        f"Version:{version}",
        f"BeatmapID:{beatmap_id}",
        f"BeatmapSetID:{beatmap_set_id}",
        "Tags:test tag1 tag2",
        "",
        "[Difficulty]",
        "HPDrainRate:5",
        f"CircleSize:{circle_size}",
        f"OverallDifficulty:{overall_difficulty}",
        "ApproachRate:9",
        "",
        "[Events]",
        "//Background and Video events",
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
    """将内容写入临时 .osu 文件，返回路径。"""
    fd, name = tempfile.mkstemp(suffix=".osu", text=True)
    import os

    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)
    return Path(name)


# ── 基础解析测试 ──────────────────────────────────────────────


class TestBasicParsing:
    """正常 mania 4K 谱面的基础解析。"""

    def test_read_mania_4k_basic(self) -> None:
        content = _make_osu_content(
            timing_points=["1182,500,4,0,0,70,1,0"],
            hit_objects=[
                "256,192,1000,1,0,0:0:0:0:",
                "320,192,1500,1,0,0:0:0:0:",
            ],
        )
        path = _write_temp_osu(content)
        try:
            reader = OsuManiaReader()
            chart = reader.read(path)
            assert isinstance(chart, Chart)
            assert chart.title == "Test Song"
            assert chart.artist == "Test Artist"
            assert chart.mode == GameMode.MANIA_4K
            assert len(chart.notes) == 2
        finally:
            path.unlink(missing_ok=True)

    def test_read_returns_artist_and_title(self) -> None:
        content = _make_osu_content(
            title="My Song",
            artist="My Artist",
            timing_points=["0,600,4,0,0,100,1,0"],
            hit_objects=["128,192,500,1,0,0:0:0:0:"],
        )
        path = _write_temp_osu(content)
        try:
            chart = OsuManiaReader().read(path)
            assert chart.title == "My Song"
            assert chart.artist == "My Artist"
        finally:
            path.unlink(missing_ok=True)


# ── Mode 检测测试 ──────────────────────────────────────────────


class TestModeDetection:
    """非 mania 模式拒绝测试。"""

    def test_taiko_mode_rejected(self) -> None:
        content = _make_osu_content(
            mode=1,  # taiko
            timing_points=["2000,500,4,0,0,100,1,0"],
            hit_objects=["256,192,1000,1,0,0:0:0:0:"],
        )
        path = _write_temp_osu(content)
        try:
            chart = OsuManiaReader().read(path)
            assert len(chart.notes) == 0
            assert chart.meta.get("skip_reason") == "mode=1"
        finally:
            path.unlink(missing_ok=True)

    def test_osu_standard_mode_rejected(self) -> None:
        content = _make_osu_content(
            mode=0,
            timing_points=["2000,500,4,0,0,100,1,0"],
            hit_objects=["256,192,1000,1,0,0:0:0:0:"],
        )
        path = _write_temp_osu(content)
        try:
            chart = OsuManiaReader().read(path)
            assert len(chart.notes) == 0
        finally:
            path.unlink(missing_ok=True)


# ── Metadata 解析测试 ──────────────────────────────────────────


class TestMetadataParsing:
    """[Metadata] 节字段提取。"""

    def test_metadata_basic(self) -> None:
        content = _make_osu_content(
            creator="TestMapper",
            version="Insane",
            beatmap_id=5746284,
            beatmap_set_id=2520906,
            timing_points=["0,500,4,0,0,100,1,0"],
            hit_objects=["64,192,1000,1,0,0:0:0:0:"],
        )
        path = _write_temp_osu(content)
        try:
            chart = OsuManiaReader().read(path)
            assert chart.meta["creator"] == "TestMapper"
            assert chart.meta["version"] == "Insane"
            assert chart.meta["beatmap_id"] == 5746284
            assert chart.meta["beatmap_set_id"] == 2520906
            assert chart.meta["tags"] == "test tag1 tag2"
        finally:
            path.unlink(missing_ok=True)


# ── Difficulty 解析测试 ────────────────────────────────────────


class TestDifficultyParsing:
    """[Difficulty] 节键位数 / OD 提取。"""

    def test_difficulty_4k(self) -> None:
        content = _make_osu_content(
            circle_size="4",
            overall_difficulty="6.5",
            timing_points=["1000,600,4,0,0,80,1,0"],
            hit_objects=["192,192,1500,1,0,0:0:0:0:"],
        )
        path = _write_temp_osu(content)
        try:
            chart = OsuManiaReader().read(path)
            assert chart.meta["key_count"] == 4
            assert chart.meta["od"] == 6.5
        finally:
            path.unlink(missing_ok=True)

    def test_difficulty_7k(self) -> None:
        content = _make_osu_content(
            circle_size="7",
            overall_difficulty="8.0",
            timing_points=["500,400,4,0,0,90,1,0"],
            hit_objects=["320,192,1000,1,0,0:0:0:0:"],
        )
        path = _write_temp_osu(content)
        try:
            chart = OsuManiaReader().read(path)
            assert chart.meta["key_count"] == 7
            # RFC-0025：非 4K mania 诚实标记真实 K 数，不降级为 4K（避免 lane 误删）
            assert chart.mode == GameMode.MANIA_7K
            assert chart.lane_count() == 7
        finally:
            path.unlink(missing_ok=True)

    def test_circle_size_float(self) -> None:
        content = _make_osu_content(
            circle_size="4.0",
            timing_points=["0,500,4,0,0,100,1,0"],
            hit_objects=["64,192,1000,1,0,0:0:0:0:"],
        )
        path = _write_temp_osu(content)
        try:
            chart = OsuManiaReader().read(path)
            assert chart.meta["key_count"] == 4.0
        finally:
            path.unlink(missing_ok=True)


# ── TimingPoints 解析测试 ──────────────────────────────────────


class TestTimingPointsParsing:
    """红线提取与绿线跳过。"""

    def test_single_red_line(self) -> None:
        content = _make_osu_content(
            timing_points=["1000,500,4,0,0,100,1,0"],
            hit_objects=["64,192,1500,1,0,0:0:0:0:"],
        )
        path = _write_temp_osu(content)
        try:
            chart = OsuManiaReader().read(path)
            assert len(chart.bpm_points) == 1
            bp = chart.bpm_points[0]
            assert bp.time == 1.0  # 1000ms -> 1.0s
            assert abs(bp.bpm - 120.0) < 0.01  # 60000/500=120
        finally:
            path.unlink(missing_ok=True)

    def test_bpm_calculation(self) -> None:
        content = _make_osu_content(
            timing_points=["0,333.33,4,0,0,100,1,0"],  # ~180 BPM
            hit_objects=["64,192,500,1,0,0:0:0:0:"],
        )
        path = _write_temp_osu(content)
        try:
            chart = OsuManiaReader().read(path)
            assert abs(chart.bpm_points[0].bpm - 180.0) < 0.5
        finally:
            path.unlink(missing_ok=True)

    def test_multiple_red_lines_tempo_changes(self) -> None:
        content = _make_osu_content(
            timing_points=[
                "0,500,4,0,0,100,1,0",  # BPM=120 @ 0s
                "30000,250,4,0,0,100,1,0",  # BPM=240 @ 30s
                "60000,1000,4,0,0,100,1,0",  # BPM=60 @ 60s
            ],
            hit_objects=["64,192,1500,1,0,0:0:0:0:"],
        )
        path = _write_temp_osu(content)
        try:
            chart = OsuManiaReader().read(path)
            assert len(chart.bpm_points) == 3
            assert chart.bpm_points[0].time == 0.0
            assert abs(chart.bpm_points[0].bpm - 120.0) < 0.01
            assert chart.bpm_points[1].time == 30.0
            assert abs(chart.bpm_points[1].bpm - 240.0) < 0.01
            assert chart.bpm_points[2].time == 60.0
            assert abs(chart.bpm_points[2].bpm - 60.0) < 0.01
        finally:
            path.unlink(missing_ok=True)

    def test_green_lines_skipped(self) -> None:
        content = _make_osu_content(
            timing_points=[
                "1000,500,4,0,0,100,1,0",  # 红线 BPM=120
                "2000,-50,4,3,0,80,0,1",  # 绿线（继承，跳过）
                "3000,-25,4,0,0,60,0,0",  # 绿线（继承，跳过）
            ],
            hit_objects=["64,192,1500,1,0,0:0:0:0:"],
        )
        path = _write_temp_osu(content)
        try:
            chart = OsuManiaReader().read(path)
            assert len(chart.bpm_points) == 1  # 仅红线
            assert abs(chart.bpm_points[0].bpm - 120.0) < 0.01
        finally:
            path.unlink(missing_ok=True)

    def test_empty_timing_points_fallback(self) -> None:
        content = _make_osu_content(
            timing_points=["0,-50,4,0,0,100,0,0"],  # 只有绿线
            hit_objects=["64,192,500,1,0,0:0:0:0:"],
        )
        path = _write_temp_osu(content)
        try:
            chart = OsuManiaReader().read(path)
            assert len(chart.bpm_points) == 1
            assert abs(chart.bpm_points[0].bpm - 120.0) < 0.01
        finally:
            path.unlink(missing_ok=True)


# ── HitObjects 解析测试 ────────────────────────────────────────


class TestHitObjectsParsing:
    """物件解析：TAP / HOLD / 列位 / 时间换算。"""

    def test_tap_note_basic(self) -> None:
        content = _make_osu_content(
            timing_points=["0,500,4,0,0,100,1,0"],
            hit_objects=["128,192,1000,1,0,0:0:0:0:"],
        )
        path = _write_temp_osu(content)
        try:
            chart = OsuManiaReader().read(path)
            assert len(chart.notes) == 1
            note = chart.notes[0]
            assert note.type == NoteType.TAP
            assert note.time == 1.0  # 1000ms -> 1.0s
            assert note.lane == 1  # floor(128 * 4 / 512) = 1
            assert note.duration == 0.0
        finally:
            path.unlink(missing_ok=True)

    def test_tap_with_new_combo(self) -> None:
        content = _make_osu_content(
            timing_points=["0,500,4,0,0,100,1,0"],
            hit_objects=["256,192,1000,5,0,0:0:0:0:"],  # type=5 (1+4)
        )
        path = _write_temp_osu(content)
        try:
            chart = OsuManiaReader().read(path)
            assert len(chart.notes) == 1
            assert chart.notes[0].type == NoteType.TAP
        finally:
            path.unlink(missing_ok=True)

    def test_hold_note_basic(self) -> None:
        content = _make_osu_content(
            timing_points=["0,500,4,0,0,100,1,0"],
            hit_objects=["256,192,1000,128,0,2500,0:0:0:0:"],
        )
        path = _write_temp_osu(content)
        try:
            chart = OsuManiaReader().read(path)
            assert len(chart.notes) == 1
            note = chart.notes[0]
            assert note.type == NoteType.HOLD
            assert note.time == 1.0
            assert note.duration == 1.5  # (2500-1000)/1000
        finally:
            path.unlink(missing_ok=True)

    def test_hold_with_both_bits(self) -> None:
        content = _make_osu_content(
            timing_points=["0,500,4,0,0,100,1,0"],
            hit_objects=["320,192,2000,129,0,4000,0:0:0:0:"],  # type=129 (128+1)
        )
        path = _write_temp_osu(content)
        try:
            chart = OsuManiaReader().read(path)
            assert len(chart.notes) == 1
            note = chart.notes[0]
            assert note.type == NoteType.HOLD
            assert note.duration == 2.0
        finally:
            path.unlink(missing_ok=True)

    def test_hold_with_new_combo(self) -> None:
        content = _make_osu_content(
            timing_points=["0,500,4,0,0,100,1,0"],
            hit_objects=["64,192,1000,132,0,3000,0:0:0:0:"],  # type=132 (128+4)
        )
        path = _write_temp_osu(content)
        try:
            chart = OsuManiaReader().read(path)
            assert len(chart.notes) == 1
            assert chart.notes[0].type == NoteType.HOLD
        finally:
            path.unlink(missing_ok=True)

    def test_hold_invalid_duration_degraded_to_tap(self) -> None:
        content = _make_osu_content(
            timing_points=["0,500,4,0,0,100,1,0"],
            hit_objects=["256,192,5000,128,0,3000,0:0:0:0:"],  # end < start
        )
        path = _write_temp_osu(content)
        try:
            chart = OsuManiaReader().read(path)
            assert len(chart.notes) == 1
            assert chart.notes[0].type == NoteType.TAP  # 降级
            assert chart.notes[0].duration == 0.0
        finally:
            path.unlink(missing_ok=True)

    def test_slider_skipped_in_mania(self) -> None:
        content = _make_osu_content(
            timing_points=["0,500,4,0,0,100,1,0"],
            hit_objects=[
                "256,192,1000,1,0,0:0:0:0:",  # TAP
                "100,100,2000,2,0,B|200:200|300:150,1,200,0:0:0:0:",  # Slider
                "256,192,3000,1,0,0:0:0:0:",  # TAP
            ],
        )
        path = _write_temp_osu(content)
        try:
            chart = OsuManiaReader().read(path)
            assert len(chart.notes) == 2  # 跳过 Slider
        finally:
            path.unlink(missing_ok=True)

    def test_spinner_skipped_in_mania(self) -> None:
        content = _make_osu_content(
            timing_points=["0,500,4,0,0,100,1,0"],
            hit_objects=[
                "256,192,1000,12,0,5000,0:0:0:0:",  # Spinner (bit 3)
                "256,192,3000,1,0,0:0:0:0:",  # TAP
            ],
        )
        path = _write_temp_osu(content)
        try:
            chart = OsuManiaReader().read(path)
            assert len(chart.notes) == 1
            assert chart.notes[0].type == NoteType.TAP
        finally:
            path.unlink(missing_ok=True)

    def test_lane_mapping(self) -> None:
        content = _make_osu_content(
            timing_points=["0,500,4,0,0,100,1,0"],
            hit_objects=[
                "64,192,1000,1,0,0:0:0:0:",  # lane 0
                "192,192,1100,1,0,0:0:0:0:",  # lane 1
                "256,192,1200,1,0,0:0:0:0:",  # lane 2
                "448,192,1300,1,0,0:0:0:0:",  # lane 3
            ],
        )
        path = _write_temp_osu(content)
        try:
            chart = OsuManiaReader().read(path)
            notes = chart.sorted_notes()
            assert [n.lane for n in notes] == [0, 1, 2, 3]
        finally:
            path.unlink(missing_ok=True)

    def test_lane_out_of_bounds(self) -> None:
        content = _make_osu_content(
            timing_points=["0,500,4,0,0,100,1,0"],
            hit_objects=["600,192,1000,1,0,0:0:0:0:"],  # lane = 4
        )
        path = _write_temp_osu(content)
        try:
            chart = OsuManiaReader().read(path)
            assert len(chart.notes) == 1
            assert chart.notes[0].lane == 4  # 越界，parse_osu 清理
        finally:
            path.unlink(missing_ok=True)

    def test_time_conversion_ms_to_seconds(self) -> None:
        content = _make_osu_content(
            timing_points=["0,500,4,0,0,100,1,0"],
            hit_objects=["64,192,0,1,0,0:0:0:0:"],
        )
        path = _write_temp_osu(content)
        try:
            chart = OsuManiaReader().read(path)
            assert chart.notes[0].time == 0.0
        finally:
            path.unlink(missing_ok=True)


# ── 编码测试 ────────────────────────────────────────────────────


class TestEncodingFallback:
    """多编码兜底验证。"""

    def test_utf8_file_reads_correctly(self) -> None:
        content = _make_osu_content(
            title="Test✓",
            timing_points=["0,500,4,0,0,100,1,0"],
            hit_objects=["64,192,1000,1,0,0:0:0:0:"],
        )
        path = _write_temp_osu(content)
        try:
            chart = OsuManiaReader().read(path)
            assert chart.title == "Test✓"
        finally:
            path.unlink(missing_ok=True)

    def test_invalid_binary_content_graceful(self) -> None:
        """二进制非文本内容不抛异常——latin-1 总能解码，但解析结果为空。"""
        path = _write_temp_osu("")
        path.write_bytes(b"\xff\xfe\xfd\xfc\xfb\xfa\xf9\xf8" * 10)
        try:
            chart = OsuManiaReader().read(path)
            # latin-1 降级使 decoding 总成功，但内容非 .osu 格式，应返回空 Chart
            assert len(chart.notes) == 0
        finally:
            path.unlink(missing_ok=True)


# ── 鲁棒性测试 ────────────────────────────────────────────────


class TestRobustness:
    """边界情况与异常输入。"""

    def test_empty_hit_objects(self) -> None:
        content = _make_osu_content(
            timing_points=["0,500,4,0,0,100,1,0"],
            hit_objects=[],
        )
        path = _write_temp_osu(content)
        try:
            chart = OsuManiaReader().read(path)
            assert len(chart.notes) == 0
        finally:
            path.unlink(missing_ok=True)

    def test_missing_timing_points(self) -> None:
        content = (
            "osu file format v14\n\n"
            "[General]\nMode: 3\n\n"
            "[Metadata]\nTitle:Test\nArtist:Test\nCreator:Test\nVersion:Hard\n"
            "BeatmapID:1\nBeatmapSetID:1\nTags:\n\n"
            "[Difficulty]\nCircleSize:4\nOverallDifficulty:5\n\n"
            "[HitObjects]\n64,192,1000,1,0,0:0:0:0:\n"
        )
        path = _write_temp_osu(content)
        try:
            chart = OsuManiaReader().read(path)
            assert len(chart.bpm_points) == 1
            assert abs(chart.bpm_points[0].bpm - 120.0) < 0.01
            assert len(chart.notes) == 1
        finally:
            path.unlink(missing_ok=True)

    def test_sorted_notes_order(self) -> None:
        content = _make_osu_content(
            timing_points=["0,500,4,0,0,100,1,0"],
            hit_objects=[
                "320,192,2000,1,0,0:0:0:0:",  # t=2.0s lane=2
                "64,192,1000,1,0,0:0:0:0:",  # t=1.0s lane=0
                "256,192,1000,1,0,0:0:0:0:",  # t=1.0s lane=2
            ],
        )
        path = _write_temp_osu(content)
        try:
            chart = OsuManiaReader().read(path)
            notes = chart.sorted_notes()
            assert [(n.time, n.lane) for n in notes] == [
                (1.0, 0),
                (1.0, 2),
                (2.0, 2),
            ]
        finally:
            path.unlink(missing_ok=True)

    def test_malformed_hit_object_skipped(self) -> None:
        content = _make_osu_content(
            timing_points=["0,500,4,0,0,100,1,0"],
            hit_objects=[
                "broken,line,here",  # 字段数不足
                "64,192,1000,1,0,0:0:0:0:",  # 正常
            ],
        )
        path = _write_temp_osu(content)
        try:
            chart = OsuManiaReader().read(path)
            assert len(chart.notes) == 1
        finally:
            path.unlink(missing_ok=True)
