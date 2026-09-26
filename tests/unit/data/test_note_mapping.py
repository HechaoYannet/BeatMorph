"""M5：note type 两套数字 + above / isFake / isCover（plan 02 §3.4-2/10）。"""

from __future__ import annotations

import json

import pytest

from beatmorph.core.contracts import (
    ChartFormat,
    NoteType,
    Side,
    note_type_from_official,
    note_type_from_rpe,
)
from beatmorph.data.parsers import analyze_chart_bytes
from beatmorph.data.parsers.rpejson import is_cover_masks, parse_rpejson
from tests.unit.data._helpers import build_rpe_bytes, chart_source, note


def test_same_number_means_opposite_types() -> None:
    """陷阱 3 的核心：数字 2 在 RPE 是 Hold、在官谱是 Drag。"""
    assert note_type_from_rpe(2) is NoteType.HOLD
    assert note_type_from_official(2) is NoteType.DRAG
    assert note_type_from_rpe(3) is NoteType.FLICK
    assert note_type_from_official(3) is NoteType.HOLD


def test_dispatch_is_driven_by_content_not_suffix() -> None:
    """RPE 内容 + .json 后缀 + 数字 2 → 必须走 RPE 分派（Hold），而不是被官谱分派成 Drag。"""
    payload = build_rpe_bytes(
        lines=[{"Name": "l0", "notes": [note(2, [0, 0, 1], end=[1, 0, 1], position_x=10.0)]}],
    )
    analysis = analyze_chart_bytes(payload)
    assert analysis.fmt is ChartFormat.RPE
    assert analysis.accepted
    assert analysis.chart is not None
    assert analysis.chart.notes[0].type is NoteType.HOLD
    assert analysis.chart.notes[0].type is not NoteType.DRAG


@pytest.mark.parametrize(
    ("above", "expected"),
    [(1, Side.FRONT), (0, Side.BACK), (2, Side.BACK)],
)
def test_above_maps_to_side(above: int, expected: Side) -> None:
    """above == 1 才是正面；0 与 2 都是背面（实测取值域 {0,1,2}）。"""
    payload = build_rpe_bytes(
        lines=[{"Name": "l0", "notes": [note(1, [0, 0, 1], above=above)]}],
    )
    chart = parse_rpejson(payload, chart_source())
    parsed = chart.notes[0]
    assert parsed.side is expected
    assert parsed.above_raw == above


def test_is_fake_only_one_is_true() -> None:
    payload = build_rpe_bytes(
        lines=[
            {
                "Name": "l0",
                "notes": [
                    note(1, [0, 0, 1], is_fake=1),
                    note(1, [1, 0, 1], is_fake=0),
                    note(1, [2, 0, 1], is_fake=2),
                ],
            },
        ],
    )
    chart = parse_rpejson(payload, chart_source())
    assert [item.is_fake for item in chart.notes] == [True, False, False]
    assert [item.is_fake_raw for item in chart.notes] == [1, 0, 2]


def test_is_cover_only_one_masks() -> None:
    """isCover：1 = 遮罩，其余 = 不遮罩（`bool(2)` 会得到错误答案）。"""
    assert is_cover_masks(1) is True
    assert is_cover_masks(2) is False
    assert is_cover_masks(0) is False
    payload = build_rpe_bytes(lines=[{"Name": "l0", "isCover": 2, "notes": []}])
    chart = parse_rpejson(payload, chart_source())
    assert chart.lines[0].is_cover == 2
    assert is_cover_masks(chart.lines[0].is_cover) is False


def test_hold_time_is_seconds_difference() -> None:
    """hold_time = (endTime - startTime) 经 BPMList 换算到秒。"""
    payload = build_rpe_bytes(
        lines=[{"Name": "l0", "notes": [note(2, [2, 0, 1], end=[4, 0, 1])]}],
    )
    chart = parse_rpejson(payload, chart_source())
    parsed = chart.notes[0]
    assert parsed.is_hold()
    assert parsed.hold_time == pytest.approx(2.0 * 60.0 / 120.0)


def test_non_hold_has_zero_hold_time() -> None:
    payload = build_rpe_bytes(lines=[{"Name": "l0", "notes": [note(1, [1, 0, 1])]}])
    chart = parse_rpejson(payload, chart_source())
    assert chart.notes[0].hold_time == 0.0


def test_hold_with_end_before_start_is_rejected() -> None:
    payload = build_rpe_bytes(
        lines=[{"Name": "l0", "notes": [note(2, [4, 0, 1], end=[2, 0, 1])]}],
    )
    analysis = analyze_chart_bytes(payload)
    assert not analysis.accepted
    assert analysis.quarantine is not None
    assert "endTime" in analysis.quarantine.reasons[0]


@pytest.mark.parametrize("bad_type", [0, 5, 99])
def test_invalid_note_type_is_rejected(bad_type: int) -> None:
    payload = build_rpe_bytes(lines=[{"Name": "l0", "notes": [note(bad_type, [0, 0, 1])]}])
    analysis = analyze_chart_bytes(payload)
    assert not analysis.accepted
    assert analysis.quarantine is not None
    assert "type" in analysis.quarantine.reasons[0]


def test_raw_fields_survive_for_lossless_round_trip(rpe_min_bytes: bytes) -> None:
    """双写策略：语义字段供建模，*_raw 供导出往返。"""
    chart = parse_rpejson(rpe_min_bytes, chart_source())
    raw = json.loads(rpe_min_bytes)
    raw_notes = [item for line in raw["judgeLineList"] for item in line.get("notes", [])]
    assert len(raw_notes) == len(chart.notes)
    for parsed, original in zip(chart.notes, raw_notes, strict=True):
        assert parsed.type_raw == original["type"]
        assert parsed.above_raw == original["above"]
        assert parsed.is_fake_raw == original["isFake"]
