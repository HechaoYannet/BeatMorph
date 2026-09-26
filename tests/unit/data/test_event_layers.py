"""M6：事件跨层求和 / 补洞 / 三态归一 / father 递归 / 缓动归一化。"""

from __future__ import annotations

import pytest

from beatmorph.core.contracts import (
    EASING_FUNCS,
    RPE_EVENT_TAIL_BEATS,
    RPE_MAX_EVENT_LAYERS,
    EventKeyframe,
    EventLayer,
    track_value,
)
from beatmorph.data.parsers import analyze_chart_bytes
from beatmorph.data.parsers.rpejson import parse_rpejson
from tests.unit.data._helpers import build_rpe_bytes, chart_source, keyframe, note


def _line_with_layers(
    layers: list[dict[str, object]],
    *,
    name: str = "l0",
    **extra: object,
) -> dict[str, object]:
    return {"Name": name, "notes": [], "eventLayers": layers, **extra}


def test_cross_layer_sum_is_sum_not_top_layer() -> None:
    """① 跨层求和：两层各给一半位移 → 最终位移等于两层之和。"""
    payload = build_rpe_bytes(
        lines=[
            _line_with_layers(
                [
                    {"moveXEvents": [keyframe([0, 0, 1], [4, 0, 1], 60.0, 60.0)]},
                    {"moveXEvents": [keyframe([0, 0, 1], [4, 0, 1], 40.0, 40.0)]},
                ],
            ),
        ],
    )
    chart = parse_rpejson(payload, chart_source())
    line = chart.lines[0]
    at = 2.0
    per_layer = [track_value(layer.move_x, at) for layer in line.event_layers]
    assert per_layer == [60.0, 40.0]
    assert line.sum_track("move_x", at) == pytest.approx(sum(per_layer))
    assert line.pose_at(at, chart).x == pytest.approx(sum(per_layer))


def test_gap_holds_previous_end_value(rpe_min_bytes: bytes) -> None:
    """② 补洞：空隙内取值 == 前一事件终值（**不是**默认值 0）。"""
    chart = parse_rpejson(rpe_min_bytes, chart_source())
    layer = chart.lines[0].event_layers[0]
    gap_time = 4.0  # 第一条 moveX 事件在 2 拍结束，第二条从 6 拍起
    value_in_gap = track_value(layer.move_x, gap_time)
    assert value_in_gap == pytest.approx(200.0)
    assert value_in_gap != 0.0
    filled = [item for item in layer.move_x if item.start == item.end]
    assert filled, "补洞应插入常量事件"
    assert filled[0].start == 200.0


def test_fill_gap_event_starts_at_previous_end() -> None:
    """补洞事件的区间恰好覆盖空隙（[前一事件终值, 后一事件起始]）。"""
    payload = build_rpe_bytes(
        lines=[
            _line_with_layers(
                [
                    {
                        "moveXEvents": [
                            keyframe([0, 0, 1], [2, 0, 1], 0.0, 10.0),
                            keyframe([5, 0, 1], [6, 0, 1], 30.0, 40.0),
                        ],
                    },
                ],
            ),
        ],
    )
    chart = parse_rpejson(payload, chart_source())
    track = chart.lines[0].event_layers[0].move_x
    assert len(track) == 4  # 2 原始 + 1 补洞 + 1 末尾
    assert track[1].start_time.to_beats() == 2.0
    assert track[1].end_time.to_beats() == 5.0
    assert track[1].start == track[1].end == 10.0
    assert track[-1].end_time.to_beats() == RPE_EVENT_TAIL_BEATS


@pytest.mark.parametrize(
    "lines",
    [
        [{"Name": "l0", "notes": [], "eventLayers": None}],
        [{"Name": "l0", "notes": [], "eventLayers": [None]}],
        [{"Name": "l0", "notes": []}],
        [{"Name": "l0", "notes": [], "eventLayers": []}],
    ],
)
def test_three_state_normalization_yields_same_ir(lines: list[dict[str, object]]) -> None:
    """③ 三态归一：null 层 / 缺字段 / 缺整段 eventLayers → 同一 IR。"""
    chart = parse_rpejson(build_rpe_bytes(lines=lines), chart_source())
    layers = chart.lines[0].event_layers
    assert len(layers) == 1
    assert layers[0] == EventLayer()
    assert layers[0].layer_index == 0


def test_missing_track_field_is_empty_list() -> None:
    payload = build_rpe_bytes(
        lines=[_line_with_layers([{"rotateEvents": [keyframe([0, 0, 1], [1, 0, 1], 1.0, 2.0)]}])],
    )
    chart = parse_rpejson(payload, chart_source())
    layer = chart.lines[0].event_layers[0]
    assert layer.move_x == []
    assert layer.move_y == []
    assert layer.speed == []
    assert layer.rotate


def test_too_many_event_layers_is_rejected() -> None:
    layers: list[dict[str, object]] = [{} for _ in range(RPE_MAX_EVENT_LAYERS + 1)]
    analysis = analyze_chart_bytes(build_rpe_bytes(lines=[_line_with_layers(layers)]))
    assert not analysis.accepted
    assert analysis.quarantine is not None
    assert "eventLayers" in analysis.quarantine.reasons[0]


def test_father_adds_parent_position() -> None:
    """④ father 递归：子线位置 == 自身 + 父线位置。"""
    payload = build_rpe_bytes(
        lines=[
            _line_with_layers(
                [{"moveXEvents": [keyframe([0, 0, 1], [4, 0, 1], 100.0, 100.0)]}],
                name="parent",
            ),
            _line_with_layers(
                [{"moveXEvents": [keyframe([0, 0, 1], [4, 0, 1], 5.0, 5.0)]}],
                name="child",
                father=0,
            ),
        ],
    )
    chart = parse_rpejson(payload, chart_source())
    at = 1.0
    parent_x = chart.lines[0].pose_at(at, chart).x
    child = chart.lines[1]
    assert parent_x == pytest.approx(100.0)
    assert child.sum_track("move_x", at) == pytest.approx(5.0)
    assert child.pose_at(at, chart).x == pytest.approx(parent_x + 5.0)


def test_father_cycle_is_rejected() -> None:
    payload = build_rpe_bytes(
        lines=[
            {"Name": "a", "notes": [], "father": 1},
            {"Name": "b", "notes": [], "father": 0},
        ],
    )
    analysis = analyze_chart_bytes(payload)
    assert not analysis.accepted
    assert analysis.quarantine is not None


def test_father_out_of_range_is_rejected() -> None:
    payload = build_rpe_bytes(lines=[{"Name": "a", "notes": [], "father": 3}])
    analysis = analyze_chart_bytes(payload)
    assert not analysis.accepted


def test_easing_type_is_normalized(rpe_min_bytes: bytes) -> None:
    """缓动归一化：>29 → 29（末端值），<1 → 1。"""
    chart = parse_rpejson(rpe_min_bytes, chart_source())
    move_x = chart.lines[0].event_layers[0].move_x
    # 轨道已被补洞：[原事件1, 补洞事件, 原事件2, 末尾事件]
    assert move_x[2].easing_type == len(EASING_FUNCS)
    rotate = chart.lines[0].event_layers[0].rotate
    assert rotate[0].easing_type == 1


@pytest.mark.parametrize("raw", [1.5, "3", None, True])
def test_non_integer_easing_type_falls_back_to_one(raw: object) -> None:
    payload = build_rpe_bytes(
        lines=[
            _line_with_layers(
                [{"moveXEvents": [keyframe([0, 0, 1], [1, 0, 1], 0.0, 1.0, easing_type=raw)]}],
            ),
        ],
    )
    chart = parse_rpejson(payload, chart_source())
    assert chart.lines[0].event_layers[0].move_x[0].easing_type == 1


def test_speed_events_without_bezier_fields_parse() -> None:
    """speedEvents 只有 5 个字段（无 bezier/bezierPoints/easing*）。"""
    payload = build_rpe_bytes(
        lines=[
            _line_with_layers(
                [
                    {
                        "speedEvents": [
                            {
                                "startTime": [0, 0, 1],
                                "endTime": [2, 0, 1],
                                "start": 10.0,
                                "end": 20.0,
                                "linkgroup": 0,
                            },
                        ],
                    },
                ],
            ),
        ],
    )
    chart = parse_rpejson(payload, chart_source())
    speed = chart.lines[0].event_layers[0].speed
    assert speed
    assert speed[0].bezier is False
    assert speed[0].start == 10.0


def test_extended_layer_is_parsed(rpe_min_bytes: bytes) -> None:
    """第 5 层（extended）独立于 eventLayers，不参与跨层求和。"""
    chart = parse_rpejson(rpe_min_bytes, chart_source())
    extended = chart.lines[3].extended
    assert extended.scale_x
    assert extended.color
    assert extended.text
    assert chart.lines[1].extended.incline


def test_event_layer_normalizes_on_construction() -> None:
    """契约 EventLayer 构造时自动补洞 → 解析层不得重复补洞（否则事件数翻倍）。"""
    track = [
        EventKeyframe(
            start_time={"i": 0, "n": 0, "d": 1},
            end_time={"i": 2, "n": 0, "d": 1},
            start=0.0,
            end=1.0,
        )
    ]
    layer = EventLayer(move_x=list(track))
    assert len(layer.move_x) == 2
    payload = build_rpe_bytes(
        lines=[_line_with_layers([{"moveXEvents": [keyframe([0, 0, 1], [2, 0, 1], 0.0, 1.0)]}])],
    )
    chart = parse_rpejson(payload, chart_source())
    assert len(chart.lines[0].event_layers[0].move_x) == len(layer.move_x)


def test_helper_factory_produces_parseable_notes() -> None:
    """自检：辅助工厂产出的 note 仍能被解析（防止工厂与解析器漂移）。"""
    payload = build_rpe_bytes(lines=[{"Name": "l0", "notes": [note(4, [0, 0, 1])]}])
    chart = parse_rpejson(payload, chart_source())
    assert chart.notes[0].type_raw == 4
