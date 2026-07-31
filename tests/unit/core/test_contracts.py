"""核心契约的单元测试样例。

验证 :mod:`beatmorph.core.contracts` 的数据模型约束生效，
作为跨设备环境是否就绪的冒烟测试（不依赖 torch）。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from beatmorph.core.contracts import (
    AR_CONTEXT_TOKENS,
    CODEBOOK_BASE,
    MERT_FRAME_RATE_HZ,
    BpmPoint,
    Chart,
    GameMode,
    Note,
    NoteType,
    PatternToken,
    Section,
)


def test_note_defaults() -> None:
    n = Note(time=1.5, lane=2)
    assert n.type is NoteType.TAP
    assert n.duration == 0.0
    assert not n.is_hold()


def test_hold_note_is_hold() -> None:
    n = Note(time=0.0, lane=0, type=NoteType.HOLD, duration=0.5)
    assert n.is_hold()
    assert n.duration == 0.5


def test_note_rejects_negative_time() -> None:
    with pytest.raises(ValidationError):
        Note(time=-1.0, lane=0)


def test_section_density_range() -> None:
    s = Section(
        index=0,
        start_time=0.0,
        end_time=8.0,
        density_target=0.5,
        energy_level=0.5,
        rest_probability=0.1,
    )
    assert 0.0 <= s.density_target <= 1.0
    with pytest.raises(ValidationError):
        Section(
            index=0,
            start_time=0.0,
            end_time=8.0,
            density_target=1.5,
            energy_level=0.5,
            rest_probability=0.1,
        )


def test_chart_lane_count_4k() -> None:
    c = Chart(difficulty=8, bpm_points=[BpmPoint(time=0.0, bpm=180.0)], mode=GameMode.MANIA_4K)
    assert c.lane_count() == 4


def test_chart_difficulty_bounds() -> None:
    with pytest.raises(ValidationError):
        Chart(difficulty=20, bpm_points=[BpmPoint(time=0.0, bpm=180.0)])  # 超出 1-15
    with pytest.raises(ValidationError):
        Chart(difficulty=0, bpm_points=[BpmPoint(time=0.0, bpm=180.0)])


def test_chart_sorted_notes_deterministic() -> None:
    c = Chart(
        difficulty=5,
        bpm_points=[BpmPoint(time=0.0, bpm=120.0)],
        notes=[
            Note(time=2.0, lane=1),
            Note(time=0.5, lane=3),
            Note(time=0.5, lane=0),
        ],
    )
    ordered = c.sorted_notes()
    assert ordered[0].time <= ordered[-1].time
    # 按 (time, lane) 升序：(0.5,0) → (0.5,3) → (2.0,1)
    assert [(n.time, n.lane) for n in ordered] == [(0.5, 0), (0.5, 3), (2.0, 1)]


def test_constants_align_with_base_plan() -> None:
    """常量与奠基文档约定一致。"""
    assert MERT_FRAME_RATE_HZ == 25.0
    assert CODEBOOK_BASE == 2048
    assert AR_CONTEXT_TOKENS == 256


# ── bpm_points（RFC-0005）──


def test_chart_requires_at_least_one_bpm_point() -> None:
    """bpm_points 不可为空（min_length=1）。"""
    with pytest.raises(ValidationError):
        Chart(difficulty=5, bpm_points=[])


def test_bpm_point_rejects_nonpositive_bpm() -> None:
    with pytest.raises(ValidationError):
        BpmPoint(time=0.0, bpm=0.0)
    with pytest.raises(ValidationError):
        BpmPoint(time=0.0, bpm=-10.0)


def test_bpm_point_rejects_negative_time() -> None:
    with pytest.raises(ValidationError):
        BpmPoint(time=-1.0, bpm=180.0)


def test_chart_supports_tempo_changes() -> None:
    """变速曲：多个 BpmPoint 按时间升序合成一段。"""
    c = Chart(
        difficulty=9,
        bpm_points=[
            BpmPoint(time=0.0, bpm=120.0),
            BpmPoint(time=30.0, bpm=180.0),
            BpmPoint(time=60.0, bpm=240.0),
        ],
    )
    assert len(c.bpm_points) == 3
    assert c.primary_bpm() == 120.0  # 取首点


def test_primary_bpm_constant_tempo() -> None:
    """常速曲退化为单点：primary_bpm 即唯一 BPM。"""
    c = Chart(difficulty=5, bpm_points=[BpmPoint(time=0.0, bpm=150.0)])
    assert c.primary_bpm() == 150.0


# ── M3：IR 序列化往返（Plan 00 M3 验收）──


def test_chart_roundtrip_constant_tempo() -> None:
    """常速曲 IR 序列化往返不丢信息。

    对齐 Plan 00 M3：``model_validate_json(model_dump_json())`` 等价。
    """
    c = Chart(
        version="ir-1",
        difficulty=7,
        bpm_points=[BpmPoint(time=0.0, bpm=174.0)],
        title="Roundtrip Test",
        artist="AFK",
        notes=[
            Note(time=0.0, lane=0),
            Note(time=0.5, lane=1, type=NoteType.HOLD, duration=0.25),
            Note(time=1.0, lane=3, type=NoteType.MINE),
        ],
        sections=[
            Section(
                index=0,
                start_time=0.0,
                end_time=5.0,
                density_target=0.6,
                energy_level=0.4,
                rest_probability=0.1,
                sections_type="intro",
            )
        ],
        meta={"license": "academic", "star": 4.5},
    )
    recon = Chart.model_validate_json(c.model_dump_json())
    assert recon.model_dump() == c.model_dump()
    assert recon.notes == c.notes
    assert recon.bpm_points == c.bpm_points
    assert recon.sections == c.sections
    assert recon.meta == c.meta


def test_chart_roundtrip_tempo_changes() -> None:
    """变速曲 bpm_points 经 JSON 往返后完全保留。"""
    c = Chart(
        difficulty=9,
        bpm_points=[
            BpmPoint(time=0.0, bpm=120.0),
            BpmPoint(time=30.0, bpm=180.0),
        ],
    )
    recon = Chart.model_validate_json(c.model_dump_json())
    assert recon.bpm_points == c.bpm_points


def test_pattern_token_roundtrip() -> None:
    """PatternToken 同样满足序列化往返。"""
    t = PatternToken(code=1024, bar_index=3, start_time=5.5, duration_bars=1)
    recon = PatternToken.model_validate_json(t.model_dump_json())
    assert recon == t
