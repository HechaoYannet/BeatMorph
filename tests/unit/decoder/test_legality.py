"""M5.4：后处理红线正确性 —— Plan 05 §4.3 / §6 / §8。

验收（M5.4）：注入式合成非法谱（越界 / 反向 Hold / 重复事件）检出率 100%；
**`positionX` 钳位次数恒为 0**（断言）；`edits` 可完整重建"后处理前"状态。
"""

from __future__ import annotations

import pytest

from beatmorph.core.contracts.legality import (
    POSITION_X_CLAMPED_KEY,
    EditKind,
    ViolationKind,
    assert_no_position_clamp,
)
from beatmorph.core.contracts.phigros import (
    RPE_STAGE_HALF_WIDTH,
    Beat,
    EventKeyframe,
    EventLayer,
    JudgeLine,
    NoteType,
    PhigrosChart,
    Side,
    side_from_above,
)
from beatmorph.decoder.events import DecodedEvent, PairingStats
from beatmorph.decoder.postprocess.legality import (
    LegalityConfig,
    check_chart,
    check_events,
    fix_chart,
    merge_reports,
    postprocess_chart,
    postprocess_events,
    same_instant_groups,
)
from tests.unit.field._builders import make_bpm_points, make_chart, make_note

BPM_POINTS = make_bpm_points((0.0, 120.0))


def _out_of_range_x() -> float:
    """一个确定越界（> 半宽）的 positionX（**派生**，不写裸数字）。"""
    return RPE_STAGE_HALF_WIDTH * 1.2


def _legal_chart() -> PhigrosChart:
    return make_chart(
        notes=[
            make_note(t=0.0, line_id=0, position_x=0.0),
            make_note(t=0.25, line_id=0, position_x=100.0, note_type=NoteType.HOLD, hold_time=0.5),
            make_note(t=0.25, line_id=1, position_x=-100.0, above=0, note_type=NoteType.DRAG),
            make_note(t=1.0, line_id=1, position_x=300.0, note_type=NoteType.FLICK),
        ],
        bpm_points=BPM_POINTS,
        k=2,
        chart_time_s=2.0,
    )


def test_legal_chart_is_reported_clean_and_left_untouched() -> None:
    """合法谱面：零违规、零留痕，且后处理**原样返回**（可审计性的前提）。"""
    chart = _legal_chart()
    report = check_chart(chart)
    assert report.violations == []
    assert report.edits == []
    assert report.is_legal
    assert report.stat("chart_notes_total") == 4.0
    assert report.stat(POSITION_X_CLAMPED_KEY) == 0.0
    assert_no_position_clamp(report)
    result = postprocess_chart(chart)
    assert result.chart is chart
    assert result.report.violations == []


def test_injected_out_of_range_is_counted_but_never_clamped() -> None:
    """M5.4 / M5.6：越界只统计、不钳位、不丢弃，且与契约侧计数一致。"""
    x = _out_of_range_x()
    chart = make_chart(
        notes=[
            make_note(t=0.0, line_id=0, position_x=x),
            make_note(t=0.5, line_id=0, position_x=-x),
            make_note(t=1.0, line_id=0, position_x=0.0),
        ],
        bpm_points=BPM_POINTS,
        k=1,
    )
    report = check_chart(chart)
    assert report.violations == [], "越界不是违规项（红线 3：只统计）"
    assert report.stat("chart_out_of_range") == 2.0
    assert report.stat("chart_out_of_range") == float(chart.out_of_visible_range().count)
    assert report.stat(POSITION_X_CLAMPED_KEY) == 0.0
    result = postprocess_chart(chart)
    assert result.chart is chart
    assert [note.position_x for note in result.chart.notes] == [x, -x, 0.0]
    assert result.report.edits == []


def test_injected_reversed_hold_is_detected_and_dropped_with_a_trail() -> None:
    """反向 Hold：事件层检出 100%，丢弃并留痕（IR 层不可表示，故拦在事件层）。"""
    good = DecodedEvent(
        t_s=1.0,
        line_id=0,
        position_x=0.0,
        side=Side.FRONT,
        note_type=NoteType.TAP,
        hold_time_s=0.0,
        is_fake=False,
        confidence=1.0,
    )
    reversed_hold = DecodedEvent(
        t_s=2.0,
        line_id=0,
        position_x=10.0,
        side=Side.FRONT,
        note_type=NoteType.HOLD,
        hold_time_s=-0.25,
        is_fake=False,
        confidence=1.0,
    )
    report = check_events([good, reversed_hold])
    assert [item.kind for item in report.violations] == [ViolationKind.HOLD_REVERSED]
    fixed, fixed_report = postprocess_events([good, reversed_hold])
    assert fixed == [good]
    assert [edit.kind for edit in fixed_report.edits] == [EditKind.DROP_HOLD_REVERSED]
    edit = fixed_report.edits[0]
    assert edit.note_index == 1
    assert edit.before == "-0.25", "留痕必须记录**原值**，否则无法重建后处理前状态"
    assert fixed_report.violations == []
    # 留痕可按 index + 原值重建"后处理前"
    original = [good, reversed_hold]
    assert original[edit.note_index].hold_time_s == float(edit.before)


def test_injected_duplicate_is_detected_and_deduped_with_a_trail() -> None:
    """重复事件：谱面层与事件层都能检出 100%，去重后复检为空违规项。"""
    duplicate = make_note(t=1.0, line_id=0, position_x=42.0)
    chart = make_chart(
        notes=[duplicate, make_note(t=0.0, line_id=0), duplicate],
        bpm_points=BPM_POINTS,
        k=1,
    )
    report = check_chart(chart)
    assert [item.kind for item in report.violations] == [ViolationKind.DUPLICATE_EVENT]
    assert report.stat("chart_duplicates") == 1.0
    fixed, edits = fix_chart(chart)
    assert len(fixed.notes) == 2
    assert [edit.kind for edit in edits] == [EditKind.DROP_DUPLICATE]
    assert check_chart(fixed).violations == []
    result = postprocess_chart(chart)
    assert result.report.violations == []
    assert len(result.chart.notes) == 2
    assert result.report.edits[0].note_index == 2

    events = [
        DecodedEvent(
            t_s=0.5,
            line_id=0,
            position_x=1.0,
            side=Side.FRONT,
            note_type=NoteType.TAP,
            hold_time_s=0.0,
            is_fake=False,
            confidence=1.0,
        ),
    ] * 2
    fixed_events, event_report = postprocess_events(events)
    assert len(fixed_events) == 1
    assert event_report.edits[0].kind is EditKind.DROP_DUPLICATE


def test_line_index_out_of_range_is_detected_and_dropped() -> None:
    """line_id 越界：必须检出并丢弃（否则写进 RPEJSON 会落到不存在的判定线）。"""
    chart = make_chart(
        notes=[make_note(t=0.0, line_id=0), make_note(t=1.0, line_id=3)],
        bpm_points=BPM_POINTS,
        k=1,
    )
    report = check_chart(chart)
    assert [item.kind for item in report.violations] == [ViolationKind.LINE_INDEX_OUT_OF_RANGE]
    result = postprocess_chart(chart)
    assert len(result.chart.notes) == 1
    assert [edit.kind for edit in result.report.edits] == [EditKind.DROP_LINE_OUT_OF_RANGE]
    assert result.report.violations == []


def test_same_instant_limit_is_off_by_default_and_switchable() -> None:
    """同刻按键上限数值未查证（plan 05 §9-1）：默认只统计；显式给出上限才成红线。"""
    chart = make_chart(
        notes=[
            make_note(t=1.0, line_id=0, position_x=-200.0),
            make_note(t=1.0, line_id=0, position_x=0.0),
            make_note(t=1.0, line_id=0, position_x=200.0),
        ],
        bpm_points=BPM_POINTS,
        k=1,
    )
    default_report = check_chart(chart)
    assert default_report.violations == []
    assert default_report.stat("chart_same_instant_max") == 3.0
    assert "chart_same_instant_over_limit" not in default_report.stats
    strict = check_chart(chart, config=LegalityConfig(same_instant_limit=2))
    assert [item.kind for item in strict.violations] == [ViolationKind.SAME_INSTANT_OVER_LIMIT]
    assert strict.stat("chart_same_instant_over_limit") == 1.0
    assert same_instant_groups(list(chart.notes)) == [[0, 1, 2]]


def test_cross_line_conflict_is_counted_only() -> None:
    """跨线几何冲突：只报告计数，**不移动任何 note**（plan 05 §9-2 阈值未定）。"""
    chart = make_chart(
        notes=[
            make_note(t=1.0, line_id=0, position_x=0.0),
            make_note(t=1.0, line_id=1, position_x=1.0),
            make_note(t=1.0, line_id=1, position_x=RPE_STAGE_HALF_WIDTH),
        ],
        bpm_points=BPM_POINTS,
        k=2,
    )
    report = check_chart(chart)
    assert report.violations == []
    assert report.stat("chart_cross_line_conflicts") == 1.0
    assert report.stat("chart_cross_line_pairs") == 2.0
    assert report.criterion, "判据必须随报告落盘（数字要有出处）"
    off = check_chart(chart, config=LegalityConfig(check_cross_line=False))
    assert "chart_cross_line_conflicts" not in off.stats


def test_hold_during_line_speed_change_is_a_warning_count() -> None:
    """「Hold 期间判定线速度变化」在 RPE 下未确证（plan 05 §9-3）-> 只报 warning 计数。"""
    line = JudgeLine(
        line_id=0,
        event_layers=[
            EventLayer(
                speed=[
                    EventKeyframe(
                        start_time=Beat(i=0),
                        end_time=Beat(i=4),
                        start=1.0,
                        end=2.0,
                    ),
                ],
            ),
        ],
    )
    chart = PhigrosChart(
        lines=[line],
        notes=[
            make_note(t=0.0, line_id=0, note_type=NoteType.HOLD, hold_time=1.0),
            make_note(t=2.0, line_id=0),
        ],
        bpm_points=BPM_POINTS,
    )
    report = check_chart(chart)
    assert report.violations == []
    assert report.stat("chart_hold_line_speed_change") == 1.0
    assert report.stat("chart_holds") == 1.0
    skipped = check_chart(chart, config=LegalityConfig(check_hold_line_speed=False))
    assert "chart_hold_line_speed_change" not in skipped.stats


def test_position_x_clamp_is_structurally_impossible() -> None:
    """M5.4：钳位次数恒为 0 —— 在**类型层**就不存在"钳位"这个动作（红线 3）。"""
    assert all("clamp" not in member.value for member in EditKind)
    chart = make_chart(
        notes=[make_note(t=0.0, line_id=0, position_x=_out_of_range_x())],
        bpm_points=BPM_POINTS,
        k=1,
    )
    result = postprocess_chart(chart)
    assert result.report.stat(POSITION_X_CLAMPED_KEY) == 0.0
    assert_no_position_clamp(result.report)
    with pytest.raises(AssertionError):
        assert_no_position_clamp(
            result.report.with_stats({POSITION_X_CLAMPED_KEY: 1.0}),
        )


def test_hold_time_and_side_semantics_are_preserved_by_fixes() -> None:
    """后处理只丢弃、从不改写 note 的落点/键型/侧别（红线 3）。"""
    note = make_note(
        t=1.0, line_id=0, position_x=123.0, above=2, note_type=NoteType.HOLD, hold_time=0.25
    )
    chart = make_chart(notes=[note, note], bpm_points=BPM_POINTS, k=1)
    fixed, edits = fix_chart(chart)
    assert edits
    assert fixed.notes[0].position_x == note.position_x
    assert fixed.notes[0].side is note.side is side_from_above(2)
    assert fixed.notes[0].hold_time == note.hold_time
    assert fixed.notes[0].type is NoteType.HOLD


def test_merge_reports_keeps_kinds_and_orders() -> None:
    """报告合并：违规/留痕顺序拼接，统计后者覆盖前者（口径不得静默混合）。"""
    first = check_events([])
    second = check_chart(_legal_chart())
    merged = merge_reports(first, second)
    assert merged.violations == []
    assert merged.stat("events_total") == 0.0
    assert merged.stat("chart_notes_total") == 4.0
    assert merged.criterion
    pairing = PairingStats(
        n_starts=1,
        n_ends=1,
        n_paired=1,
        n_unpaired_starts=0,
        n_orphan_ends=0,
        n_zero_length=0,
    )
    _, report = postprocess_events([], pairing=pairing)
    assert report.stat("hold_paired") == 1.0
