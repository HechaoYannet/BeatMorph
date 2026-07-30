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
    Chart,
    GameMode,
    MERT_FRAME_RATE_HZ,
    Note,
    NoteType,
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
    c = Chart(difficulty=8, bpm=180.0, mode=GameMode.MANIA_4K)
    assert c.lane_count() == 4


def test_chart_difficulty_bounds() -> None:
    with pytest.raises(ValidationError):
        Chart(difficulty=20, bpm=180.0)  # 超出 1-15
    with pytest.raises(ValidationError):
        Chart(difficulty=0, bpm=180.0)


def test_chart_sorted_notes_deterministic() -> None:
    c = Chart(
        difficulty=5,
        bpm=120.0,
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
