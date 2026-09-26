"""M6.4 分解：时间组（含 12/24 三连）与难度分档（plan 06 §4.4）。

关键判据：

- 归组规则**与解码网格无关**（1/32 拍不在 1/48 网格上，仍须归入 1/32）；
- δ = 0（整拍事件）时全部落入 1/4 组；
- difficulty 先 round 到 0.1；level 自由文本**绝不解析**。
"""

from __future__ import annotations

import inspect

from beatmorph.core.contracts.phigros import SUBDIVISIONS_PER_BEAT
from beatmorph.eval.breakdown import (
    TIME_GROUPS,
    TimeGroup,
    assign_time_groups,
    classify_beat_position,
    classify_interval,
    difficulty_band,
)
from beatmorph.eval.metrics import evaluate_case
from beatmorph.eval.protocol import UNKNOWN_DIFFICULTY_BAND, EvalConfig, TimeGroupRule
from beatmorph.eval.report import difficulty_breakdown
from tests.unit.eval._builders import (
    beat_fraction,
    beats_to_seconds,
    make_bpm_points,
    make_case,
    make_event,
    perfect_case,
)

CONFIG = EvalConfig()
BPM_POINTS = make_bpm_points()


def test_position_rule_hits_each_declared_subdivision() -> None:
    """1/4、1/8、1/12（三连）、1/16、1/24（三连）、1/32 六档都要能命中。"""
    expected = {
        TimeGroup.QUARTER: 4,
        TimeGroup.EIGHTH: 8,
        TimeGroup.TWELFTH: 12,
        TimeGroup.SIXTEENTH: 16,
        TimeGroup.TWENTY_FOURTH: 24,
        TimeGroup.THIRTY_SECOND: 32,
    }
    assert set(TIME_GROUPS) == set(expected)
    for group, denominator in expected.items():
        assert classify_beat_position(beat_fraction(denominator)) is group
    assert classify_beat_position(0.0) is TimeGroup.QUARTER
    assert classify_beat_position(1.0) is TimeGroup.QUARTER


def test_grouping_is_independent_of_the_decoding_grid() -> None:
    """1/32 拍**不在** 1/48 解码网格上，仍必须正确归组（plan §4.4-1 的硬要求）。"""
    off_grid = beat_fraction(32)
    assert (off_grid * SUBDIVISIONS_PER_BEAT) % 1.0 != 0.0, "1/32 确实不在 1/48 网格上"
    assert classify_beat_position(off_grid) is TimeGroup.THIRTY_SECOND
    on_grid = beat_fraction(4)
    assert (on_grid * SUBDIVISIONS_PER_BEAT) % 1.0 == 0.0
    assert classify_beat_position(on_grid) is TimeGroup.QUARTER


def test_triplet_groups_cannot_be_omitted() -> None:
    """12/24 三连组必须存在且可命中（文献库 §7.2-5：三连组不可省）。"""
    assert classify_beat_position(beat_fraction(12)) is TimeGroup.TWELFTH
    assert classify_beat_position(beat_fraction(24)) is TimeGroup.TWENTY_FOURTH
    assert classify_beat_position(2.0 * beat_fraction(12)) is TimeGroup.TWELFTH


def test_zero_phase_events_all_land_in_the_quarter_group() -> None:
    """M6.4：δ = 0（事件都在整拍上）→ 全部落入 1/4 组。"""
    case = perfect_case(count=6)
    groups = assign_time_groups(case.gold, case.bpm_points)
    assert set(groups) == {TimeGroup.QUARTER}
    assert len(groups) == len(case.gold)


def test_assign_time_groups_is_order_invariant() -> None:
    """归组与输入顺序无关（corruption 的输入置换不变性依赖这一点）。"""
    case = perfect_case(count=6)
    forward = assign_time_groups(case.gold, case.bpm_points)
    backward = assign_time_groups(tuple(reversed(case.gold)), case.bpm_points)
    assert forward == tuple(reversed(backward))


def test_interval_rule_is_documented_and_differs_from_position_rule() -> None:
    """间隔口径 = 与**同线前一事件**的拍间隔；首事件退回位置口径；跨线不参与。"""
    step = beat_fraction(8)
    start = beat_fraction(16)
    events = tuple(
        make_event(beats_to_seconds(start + index * step), line_id=0) for index in range(4)
    )
    position = assign_time_groups(events, BPM_POINTS, rule=TimeGroupRule.POSITION)
    interval = assign_time_groups(events, BPM_POINTS, rule=TimeGroupRule.INTERVAL)
    assert position[1] is TimeGroup.SIXTEENTH
    assert interval[1] is TimeGroup.EIGHTH
    assert interval[0] is position[0], "首个事件没有前驱 → 退回位置口径"
    assert classify_interval(step) is TimeGroup.EIGHTH
    assert classify_interval(-step) is TimeGroup.EIGHTH
    other_line = (*events, make_event(beats_to_seconds(start + step / 2.0), line_id=1))
    assert assign_time_groups(other_line, BPM_POINTS, rule=TimeGroupRule.INTERVAL)[:4] == interval


def test_difficulty_band_rounds_to_one_decimal_and_never_parses_level_text() -> None:
    """难度分档：round 到 0.1；level 文本不参与（函数签名里没有 level 参数）。"""
    assert "level" not in inspect.signature(difficulty_band).parameters
    assert difficulty_band(14.900001) == "14.9"
    assert difficulty_band(18.000004) == "18.0"
    assert difficulty_band(15.0) == "15.0"
    assert difficulty_band(None) == UNKNOWN_DIFFICULTY_BAND
    assert difficulty_band(float("nan")) == UNKNOWN_DIFFICULTY_BAND
    weird = perfect_case(count=4, difficulty=None, level_text="AT  Lv.16")
    assert evaluate_case(weird, CONFIG).difficulty_band == UNKNOWN_DIFFICULTY_BAND
    normal = perfect_case(count=4, difficulty=15.6, level_text="sweet")
    assert evaluate_case(normal, CONFIG).difficulty_band == "15.6"


def test_time_group_breakdown_covers_every_event_and_keeps_groups_apart() -> None:
    """分解必须覆盖全部事件，且 1/4 与 1/12 分组各自成行。"""
    beats = (0.0, beat_fraction(8), 1.0, beat_fraction(12), 2.0 * beat_fraction(12))
    gold = tuple(make_event(beats_to_seconds(beat), line_id=0) for beat in beats)
    evaluation = evaluate_case(make_case("mix", pred=gold, gold=gold), CONFIG)
    names = set(evaluation.time_groups)
    assert names <= {group.value for group in TIME_GROUPS}
    assert TimeGroup.QUARTER.value in names
    assert TimeGroup.TWELFTH.value in names
    assert sum(row.n_gold for row in evaluation.time_groups.values()) == len(gold)
    assert sum(row.n_pred for row in evaluation.time_groups.values()) == len(gold)
    for row in evaluation.time_groups.values():
        assert row.timing.f1 == 1.0
        assert row.event.f1 == 1.0


def test_difficulty_breakdown_groups_by_rounded_band() -> None:
    """难度分解按 round(0.1) 后的档名分组（14.900001 与 14.9 同档）。"""
    first = perfect_case("low-a", count=4, difficulty=14.900001)
    second = perfect_case("low-b", count=6, difficulty=14.9)
    high = perfect_case("high", count=4, difficulty=18.000004)
    rows = difficulty_breakdown(
        [
            evaluate_case(first, CONFIG),
            evaluate_case(second, CONFIG),
            evaluate_case(high, CONFIG),
        ],
        CONFIG,
    )
    assert set(rows) == {"14.9", "18.0"}
    assert rows["14.9"].n_gold == len(first.gold) + len(second.gold)
    assert rows["18.0"].n_gold == len(high.gold)
    assert rows["14.9"].timing.f1 == 1.0
