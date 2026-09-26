"""M7：三层质检 —— schema 违约进隔离区、越界只统计不钳位。"""

from __future__ import annotations

import json

import pytest

from beatmorph.core.contracts import (
    ChartFormat,
    NoteType,
    PhigrosNote,
    Side,
    note_type_from_rpe,
)
from beatmorph.data.parsers import analyze_chart_bytes, parse_rpejson
from beatmorph.data.qc import (
    KNOWN_ABOVE_VALUES,
    QuarantineStage,
    distribution_stats,
    max_simultaneous_onsets,
    min_same_line_same_time_gap_x,
    quality_check,
    unit_contract_violations,
)
from tests.unit.data._helpers import (
    build_rpe_bytes,
    chart_source,
    note,
    out_of_range_position_x,
)


def test_unit_contracts_are_consistent() -> None:
    """G4 数据侧落点：舞台几何与 x 网格必须派生一致。"""
    assert unit_contract_violations() == []


def test_fixture_passes_and_reports_one_out_of_range(rpe_min_bytes: bytes) -> None:
    chart = parse_rpejson(rpe_min_bytes, chart_source(chart_id=1000))
    report = quality_check(chart, audio_duration_s=None, chart_id=1000)
    assert report.passed, report.errors
    assert report.out_of_visible_range == 1
    assert report.n_lines == len(chart.lines)
    assert report.n_notes == len(chart.notes)


def test_out_of_range_note_is_not_clamped(rpe_min_bytes: bytes) -> None:
    """M7：越界样本 `out_of_visible_range > 0` 且原值未被修改（逐字段比对原 JSON）。"""
    chart = parse_rpejson(rpe_min_bytes, chart_source())
    report = quality_check(chart, None)
    assert report.out_of_visible_range > 0

    raw = json.loads(rpe_min_bytes)
    raw_notes = [item for line in raw["judgeLineList"] for item in line.get("notes", [])]
    assert len(raw_notes) == len(chart.notes)
    for parsed, original in zip(chart.notes, raw_notes, strict=True):
        assert parsed.position_x == original["positionX"]


def test_out_of_range_counter_matches_contract_stats() -> None:
    payload = build_rpe_bytes(
        lines=[
            {
                "Name": "l0",
                "notes": [
                    note(1, [0, 0, 1], position_x=out_of_range_position_x()),
                    note(1, [1, 0, 1], position_x=-out_of_range_position_x()),
                    note(1, [2, 0, 1], position_x=0.0),
                ],
            },
        ],
    )
    chart = parse_rpejson(payload, chart_source())
    report = quality_check(chart, None)
    assert report.out_of_visible_range == 2
    assert chart.out_of_visible_range().count == 2


def test_schema_violation_goes_to_quarantine() -> None:
    """schema 违约 100% 进隔离区（这里用非法 note type）。"""
    payload = build_rpe_bytes(lines=[{"Name": "l0", "notes": [note(9, [0, 0, 1])]}])
    analysis = analyze_chart_bytes(payload, chart_id=7)
    assert not analysis.accepted
    assert analysis.quarantine is not None
    assert analysis.quarantine.stage is QuarantineStage.PARSE
    assert analysis.quarantine.chart_id == 7
    assert analysis.quarantine.to_dict()["reasons"]


def test_empty_judge_line_list_is_rejected() -> None:
    payload = build_rpe_bytes(lines=[])
    analysis = analyze_chart_bytes(payload)
    assert not analysis.accepted
    assert analysis.quarantine is not None


def test_chart_without_notes_is_a_schema_error() -> None:
    payload = build_rpe_bytes(lines=[{"Name": "l0", "notes": []}])
    chart = parse_rpejson(payload, chart_source())
    report = quality_check(chart, None)
    assert not report.passed
    assert any("note" in message for message in report.errors)


def test_non_hold_with_hold_time_is_an_error() -> None:
    """直接构造违约 IR（解析器不会产出这种样本）→ 质检必须抓住。"""
    payload = build_rpe_bytes(lines=[{"Name": "l0", "notes": [note(1, [0, 0, 1])]}])
    chart = parse_rpejson(payload, chart_source())
    broken = chart.notes[0].model_copy(update={"hold_time": 1.0})
    patched = chart.model_copy(update={"notes": [broken]})
    report = quality_check(patched, None)
    assert not report.passed
    assert any("hold_time" in message for message in report.errors)


def test_bpm_factor_is_reported_not_ignored(rpe_min_bytes: bytes) -> None:
    """bpmfactor != 1.0 必须计入隔离报告（存储但不参与换算，存疑 D4）。"""
    chart = parse_rpejson(rpe_min_bytes, chart_source())
    assert chart.lines[3].bpm_factor == 2.0
    report = quality_check(chart, None)
    assert any("bpm_factor" in message for message in report.warnings)


def test_unknown_above_value_warns_but_keeps_raw() -> None:
    payload = build_rpe_bytes(lines=[{"Name": "l0", "notes": [note(1, [0, 0, 1], above=7)]}])
    chart = parse_rpejson(payload, chart_source())
    report = quality_check(chart, None)
    assert chart.notes[0].above_raw == 7
    assert any("above_raw" in message for message in report.warnings)
    assert 7 not in KNOWN_ABOVE_VALUES


def test_lines_without_notes_warn_only(rpe_min_bytes: bytes) -> None:
    """实测 ~42% 的判定线不带 note（调研 §7.2）→ 单线为空只告警，不是 schema 违约。"""
    chart = parse_rpejson(rpe_min_bytes, chart_source())
    report = quality_check(chart, None)
    assert report.passed
    assert any("line 3" in message for message in report.warnings)


def test_out_of_audio_window_counts_notes_past_duration(rpe_min_bytes: bytes) -> None:
    chart = parse_rpejson(rpe_min_bytes, chart_source())
    short = quality_check(chart, audio_duration_s=0.1)
    assert short.out_of_audio_window > 0
    long = quality_check(chart, audio_duration_s=chart.duration_s() + 1.0)
    assert long.out_of_audio_window == 0


def test_hold_end_past_audio_window_counts() -> None:
    payload = build_rpe_bytes(
        lines=[{"Name": "l0", "notes": [note(2, [0, 0, 1], end=[8, 0, 1])]}],
    )
    chart = parse_rpejson(payload, chart_source())
    report = quality_check(chart, audio_duration_s=1.0)
    assert report.out_of_audio_window == 1


def test_distribution_stats_shape(rpe_min_bytes: bytes) -> None:
    chart = parse_rpejson(rpe_min_bytes, chart_source())
    stats = distribution_stats(chart)
    assert stats.n_lines == len(chart.lines)
    assert stats.n_notes == len(chart.notes)
    assert set(stats.type_counts) <= {int(member) for member in NoteType}
    # 夹具共 6 个 note：Tap 2 个；above = {1: 4, 2: 1, 0: 1}
    assert stats.type_counts[int(NoteType.TAP)] == 2
    assert stats.above_counts[1] == 4
    assert stats.back_fraction == pytest.approx(2 / 6)
    assert stats.tap_fraction == pytest.approx(2 / 6)
    assert 0.0 < stats.notes_per_line_entropy_norm <= 1.0
    assert stats.bpm_min == 120.0
    assert stats.bpm_max == 180.0
    assert stats.max_simultaneous_onsets == 1
    assert stats.to_dict()["type_counts"]["1"] == 2


def test_same_time_same_line_gap_and_onsets() -> None:
    payload = build_rpe_bytes(
        lines=[
            {
                "Name": "l0",
                "notes": [
                    note(1, [0, 0, 1], position_x=0.0),
                    note(1, [0, 0, 1], position_x=15.0),
                ],
            },
            {"Name": "l1", "notes": [note(1, [0, 0, 1], position_x=0.0)]},
        ],
    )
    chart = parse_rpejson(payload, chart_source())
    assert min_same_line_same_time_gap_x(chart) == pytest.approx(15.0)
    assert max_simultaneous_onsets(chart) == 3


def test_gap_is_none_when_no_simultaneous_notes(rpe_min_bytes: bytes) -> None:
    chart = parse_rpejson(rpe_min_bytes, chart_source())
    assert min_same_line_same_time_gap_x(chart) is None


def test_gap_is_zero_for_duplicate_positions() -> None:
    payload = build_rpe_bytes(
        lines=[
            {
                "Name": "l0",
                "notes": [
                    note(1, [0, 0, 1], position_x=5.0),
                    note(1, [0, 0, 1], position_x=5.0),
                ],
            },
        ],
    )
    chart = parse_rpejson(payload, chart_source())
    assert min_same_line_same_time_gap_x(chart) == 0.0


def test_report_serialization_has_plan_fields(rpe_min_bytes: bytes) -> None:
    chart = parse_rpejson(rpe_min_bytes, chart_source(chart_id=42))
    report = quality_check(chart, None, chart_id=42, fmt=ChartFormat.RPE)
    payload = report.to_dict()
    for key in (
        "chart_id",
        "qc_passed",
        "n_lines",
        "n_notes",
        "format",
        "out_of_visible_range",
        "out_of_audio_window",
        "errors",
        "warnings",
    ):
        assert key in payload
    assert payload["chart_id"] == 42
    assert payload["distribution"] is not None


def test_parser_rejects_bad_hold_via_contract() -> None:
    """Hold 的 endTime < startTime 由解析器拒收（schema 级）。"""
    payload = build_rpe_bytes(lines=[{"Name": "l0", "notes": [note(2, [4, 0, 1], end=[1, 0, 1])]}])
    analysis = analyze_chart_bytes(payload)
    assert not analysis.accepted


def test_bpm_list_must_start_at_beat_zero() -> None:
    """BPMList 首段必须起于 0 拍，否则两条秒↔拍实现会分叉（必须进隔离区）。

    实测：首段起于第 4 拍时，格式层 beat↔秒与 field/ 的 τ↔秒相差常量 2 秒
    ——同一个谱面就有两条时间轴，正是 beat-aligned 版的 25 Hz 事件形态。
    """
    payload = build_rpe_bytes(
        bpm_list=[{"bpm": 120.0, "startTime": [4, 0, 1]}],
        lines=[{"Name": "l0", "notes": [note(1, [4, 0, 1])]}],
    )
    analysis = analyze_chart_bytes(payload)
    assert not analysis.accepted


def test_note_side_enum_is_not_boolean() -> None:
    """Side 的两个成员都是真值（设计约束）：任何 `if side:` 都必然失效。"""
    assert bool(Side.FRONT) is True
    assert bool(Side.BACK) is True
    parsed = PhigrosNote.model_validate(
        {"line_id": 0, "t": 0.0, "position_x": 0.0, "side": Side.BACK, "type": NoteType.TAP},
    )
    assert parsed.side is Side.BACK
    assert parsed.type is note_type_from_rpe(1)
