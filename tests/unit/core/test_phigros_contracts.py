"""Plan 00 契约测试 —— Phigros 领域对象（默认 CI，无权重 / 无 GPU / 无 torch）。

对应里程碑：M1（可导入）、M3（无损往返）、M5（side 语义）、M6（type 双格式分派）、
M8（最小闭环的手写 IR 夹具）。不变量编号沿用 docs/plans/00-core-contracts.md §3.8。

夹具是**契约形状**的手写 IR（snake_case），不是 RPEJSON 原文——RPEJSON 的
camelCase 到 IR 的字段映射由 plan 02 的解析器负责，本文件不重复实现它。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from beatmorph.core.contracts import (
    EASING_FUNCS,
    RPE_MAX_EVENT_LAYERS,
    RPE_STAGE_HALF_WIDTH,
    SUBDIVISIONS_PER_BEAT,
    TAU_GRID_DT,
    Beat,
    BpmPoint,
    EventKeyframe,
    EventLayer,
    JudgeLine,
    NoteType,
    PhigrosChart,
    PhigrosNote,
    Side,
    fill_gaps,
    note_type_from_official,
    note_type_from_rpe,
    side_from_above,
    side_index,
    track_value,
)


def _kf(x0: float, x1: float, st: float, et: float) -> dict[str, object]:
    """一个事件关键帧的契约形状字典（整拍起止，测试内不出现物理常量）。"""
    return {
        "start_time": [int(st), 0, 1],
        "end_time": [int(et), 0, 1],
        "start": x0,
        "end": x1,
    }


def minimal_ir() -> dict[str, object]:
    """手写最小 IR 夹具（<= 32 KB，多线 / 多事件层 / 覆盖 above 三种取值）。

    含一个越界 position_x（= 半宽的 1.5 倍）用于 I3 的「只统计不钳位」断言。
    """
    return {
        "version": "phigros-ir-1",
        "lines": [
            {
                "line_id": 0,
                "name": "line-a",
                "event_layers": [
                    {"layer_index": 0, "move_x": [_kf(0.0, 0.0, 0, 4)]},
                    {"layer_index": 1, "move_x": [_kf(0.0, 0.0, 0, 4)]},
                ],
            },
            {
                "line_id": 1,
                "name": "line-b",
                "event_layers": [{"layer_index": 0, "move_y": [_kf(0.0, 0.0, 0, 4)]}],
            },
        ],
        "notes": [
            {
                "line_id": 0,
                "t": 1.0,
                "position_x": 0.0,
                "side": 1,
                "type": 1,
                "above_raw": 1,
                "type_raw": 1,
            },
            {
                "line_id": 0,
                "t": 2.0,
                "position_x": RPE_STAGE_HALF_WIDTH * 1.5,
                "side": -1,
                "type": 2,
                "hold_time": 1.0,
                "above_raw": 2,
                "type_raw": 2,
            },
            {
                "line_id": 1,
                "t": 2.5,
                "position_x": RPE_STAGE_HALF_WIDTH,
                "side": -1,
                "type": 4,
                "above_raw": 0,
                "type_raw": 4,
            },
        ],
        "bpm_points": [{"time_beats": 0.0, "bpm": 180.0}],
        "meta": {"level_text": "AT Lv.15", "chart_time_s": 30.0},
    }


def _chart() -> PhigrosChart:
    return PhigrosChart.model_validate(minimal_ir())


# ── M1：可导入且不依赖 torch ───────────────────────────────────


def test_contract_layer_never_imports_torch() -> None:
    """I12：契约层不得依赖 torch（契约级测试不得依赖权重或 GPU）。"""
    contracts_dir = Path(__file__).resolve().parents[3] / "beatmorph" / "core" / "contracts"
    for path in sorted(contracts_dir.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        assert "import torch" not in source, path.name
        assert "from torch" not in source, path.name
    assert PhigrosChart is not None


# ── M5：side 语义（I4 / I5）────────────────────────────────────


def test_side_from_above_maps_all_observed_values() -> None:
    """I4：1 -> FRONT；0 与 2 都 -> BACK（实测取值域）。"""
    assert side_from_above(1) is Side.FRONT
    assert side_from_above(0) is Side.BACK
    assert side_from_above(2) is Side.BACK


def test_both_sides_are_truthy() -> None:
    """I5：两侧都是真值，任何 if side: 写法都必须失效。"""
    assert bool(Side.FRONT) is True
    assert bool(Side.BACK) is True
    assert Side.FRONT == 1
    assert Side.BACK == -1


def test_side_index_differs_from_enum_value() -> None:
    """场通道索引是 0/1，与枚举值 ±1 **不同**。"""
    assert side_index(Side.FRONT) == 0
    assert side_index(Side.BACK) == 1
    assert side_index(Side.FRONT) != int(Side.FRONT)


# ── M6：type 双格式分派（I6）───────────────────────────────────


def test_note_type_dispatch_two_formats() -> None:
    """I6：同一个数字 2 在 RPE 是 HOLD、在官谱是 DRAG。"""
    assert note_type_from_rpe(1) is NoteType.TAP
    assert note_type_from_rpe(2) is NoteType.HOLD
    assert note_type_from_rpe(3) is NoteType.FLICK
    assert note_type_from_rpe(4) is NoteType.DRAG
    assert note_type_from_official(1) is NoteType.TAP
    assert note_type_from_official(2) is NoteType.DRAG
    assert note_type_from_official(3) is NoteType.HOLD
    assert note_type_from_official(4) is NoteType.FLICK


def test_note_type_dispatch_signature_has_no_filename() -> None:
    """签名级约束：分派函数不得接受文件名 / 后缀参数（只能由内容嗅探驱动）。"""
    import inspect

    for func in (note_type_from_rpe, note_type_from_official):
        assert list(inspect.signature(func).parameters) == ["raw"]


def test_note_type_rejects_out_of_domain() -> None:
    with pytest.raises(ValueError, match="RPE note type"):
        note_type_from_rpe(0)
    with pytest.raises(ValueError, match="RPE note type"):
        note_type_from_rpe(5)


# ── I7：事件层长度不截断不补齐 ─────────────────────────────────


def test_event_layers_length_preserved() -> None:
    for n_layers in (1, 2, RPE_MAX_EVENT_LAYERS):
        layers = [EventLayer(layer_index=i) for i in range(n_layers)]
        assert len(JudgeLine(line_id=0, event_layers=layers).event_layers) == n_layers


def test_event_layers_rejects_out_of_domain() -> None:
    with pytest.raises(ValidationError):
        JudgeLine(line_id=0, event_layers=[])
    with pytest.raises(ValidationError):
        JudgeLine(
            line_id=0,
            event_layers=[EventLayer() for _ in range(RPE_MAX_EVENT_LAYERS + 1)],
        )


def test_empty_layer_is_the_normalization_result() -> None:
    """三态归一（null 层 / 缺字段 / 缺整段）在契约侧的落点：空轨列表。"""
    assert EventLayer.model_validate({}) == EventLayer()
    assert EventLayer().move_x == []
    assert JudgeLine(line_id=0).event_layers == [EventLayer()]


def test_beat_accepts_rpe_sequence_notation() -> None:
    assert Beat.model_validate([3, 1, 2]).to_beats() == pytest.approx(3.5)
    with pytest.raises(ValidationError):
        Beat.model_validate([1, 2])


# ── 事件求值语义 ───────────────────────────────────────────────


def _evaluate(move_x: list[dict[str, object]], t_beats: float) -> float:
    line = JudgeLine(
        line_id=0,
        event_layers=[
            EventLayer(layer_index=0, move_x=[EventKeyframe.model_validate(k) for k in move_x])
        ],
    )
    return line.sum_track("move_x", t_beats)


def test_cross_layer_sum_is_sum_not_top_layer() -> None:
    """跨层**求和**（不是取最上层）：两层各给一半位移。"""
    half = float(RPE_STAGE_HALF_WIDTH)
    layers = [
        EventLayer(layer_index=0, move_x=[EventKeyframe.model_validate(_kf(0.0, half, 0, 4))]),
        EventLayer(layer_index=1, move_x=[EventKeyframe.model_validate(_kf(0.0, half, 0, 4))]),
    ]
    chart = PhigrosChart(
        lines=[JudgeLine(line_id=0, event_layers=layers)],
        bpm_points=[BpmPoint(time_beats=0.0, bpm=180.0)],
    )
    assert chart.lines[0].pose_at(4.0, chart).x == pytest.approx(half + half)


def test_gap_filling_holds_previous_end_value() -> None:
    """补洞：空隙内取值 == 前一事件终值（而不是默认值 0）。"""
    events = [
        EventKeyframe.model_validate(_kf(5.0, 5.0, 0, 1)),
        EventKeyframe.model_validate(_kf(9.0, 9.0, 5, 6)),
    ]
    layer = EventLayer(layer_index=0, move_x=events)
    # 1 个原事件 + 1 个空隙常量事件 + 1 个原事件 + 1 个尾部常量事件
    assert len(fill_gaps(events)) == 4
    assert layer.move_x[1].start_time.to_beats() == 1.0
    assert layer.move_x[1].end_time.to_beats() == 5.0
    assert layer.move_x[1].start == 5.0
    assert layer.move_x[1].end == 5.0


def test_gap_filled_track_is_defined_everywhere() -> None:
    layer = EventLayer(
        layer_index=0,
        move_x=[
            EventKeyframe.model_validate(_kf(2.0, 2.0, 0, 1)),
            EventKeyframe.model_validate(_kf(7.0, 7.0, 5, 6)),
        ],
    )
    for t in (0.0, 0.5, 1.0, 3.0, 4.999, 5.0, 5.5, 6.0, 100.0):
        assert track_value(layer.move_x, t) in (2.0, 7.0)


def test_gap_filling_preserves_values_where_the_raw_track_is_defined() -> None:
    """补洞只插入常量事件：原事件覆盖区间内的取值一字不变。"""
    events = [_kf(1.0, 3.0, 0, 2), _kf(5.0, 5.0, 6, 8)]
    filled = [_kf(1.0, 3.0, 0, 2), _kf(3.0, 3.0, 2, 6), _kf(5.0, 5.0, 6, 8)]
    for t in (0.0, 0.5, 1.0, 1.5, 2.0, 6.0, 7.0, 8.0):
        assert _evaluate(filled, t) == pytest.approx(_evaluate(events, t)), t
    # 空隙内原轨无定义（取默认 0），补洞后取前一事件终值 3.0
    raw = [EventKeyframe.model_validate(k) for k in events]
    assert track_value(raw, 4.0) == 0.0
    assert track_value(fill_gaps(raw), 4.0) == pytest.approx(3.0)


def test_easing_type_normalization() -> None:
    base = _kf(0.0, 1.0, 0, 1)
    assert EventKeyframe.model_validate({**base, "easing_type": "nonsense"}).easing_type == 1
    assert EventKeyframe.model_validate({**base, "easing_type": 0}).easing_type == 1
    assert EventKeyframe.model_validate({**base, "easing_type": 999}).easing_type == len(
        EASING_FUNCS
    )


def test_unknown_keys_are_rejected_not_silently_dropped() -> None:
    """RPE 的 camelCase 键名必须**报错**，不得被静默忽略。

    静默丢弃字段名会让「解析器写错字段名」变成全谱取值错误却毫无提示——
    这正是 25 Hz 事件同一类的静默失效（POSTMORTEM-2026-08-05）。
    """
    with pytest.raises(ValidationError):
        EventKeyframe.model_validate({**_kf(0.0, 1.0, 0, 1), "easingType": 5})
    with pytest.raises(ValidationError):
        EventKeyframe.model_validate(
            {
                "startTime": [0, 0, 1],
                "endTime": [1, 0, 1],
                "start": 0.0,
                "end": 1.0,
            },
        )


def _eased_value(easing_type: int, left: float, right: float, t: float) -> float:
    kf = EventKeyframe.model_validate(
        {
            **_kf(0.0, 1.0, 0, 1),
            "easing_type": easing_type,
            "easing_left": left,
            "easing_right": right,
        },
    )
    value = kf.value_at(t)
    assert isinstance(value, float)
    return value


def test_easing_cut_matches_the_three_documented_workings() -> None:
    """切割公式由知识文档同时给出的三条工作实例唯一确定。

    ① Linear 切割不起作用；② Out Quad 改变左边界不起作用；③ In Quad 改变
    右边界不起作用。三条只有归一化形式 g(t) = (f(l+t(r-l)) - f(l)) / (f(r) - f(l))
    同时成立（原文公式丢失了归一化分母，照抄会使事件终点取不到 end 值）。
    """
    for t in (0.125, 0.5, 0.875):
        assert _eased_value(1, 0.25, 0.75, t) == pytest.approx(t)
        assert _eased_value(4, 0.3, 1.0, t) == pytest.approx(_eased_value(4, 0.0, 1.0, t))
        assert _eased_value(5, 0.0, 0.7, t) == pytest.approx(_eased_value(5, 0.0, 1.0, t))


def test_easing_reaches_endpoints_for_all_cuts_and_types() -> None:
    """1..29 全表 x 多种切割：曲线必须从 start 走到 end（端点精确可达）。"""
    for easing_type in range(1, len(EASING_FUNCS) + 1):
        for left, right in ((0.0, 1.0), (0.25, 0.75), (0.0, 0.5)):
            kf = EventKeyframe.model_validate(
                {
                    **_kf(1.0, 3.0, 0, 1),
                    "easing_type": easing_type,
                    "easing_left": left,
                    "easing_right": right,
                },
            )
            assert kf.value_at(0.0) == pytest.approx(1.0)
            assert kf.value_at(1.0) == pytest.approx(3.0)
            # 逼近终点：曲线连续（Out Expo 在 t=1 前 1e-6 处仍差约 1e-3，故容差取 1e-2）
            assert kf.value_at(0.999999) == pytest.approx(3.0, abs=1e-2)


def test_bezier_easing_is_monotone_and_hits_endpoints() -> None:
    kf = EventKeyframe.model_validate(
        {**_kf(0.0, 1.0, 0, 1), "bezier": 1, "bezier_points": [0.42, 0.0, 0.58, 1.0]},
    )
    values = [kf.value_at(i / 16) for i in range(17)]
    assert values[0] == pytest.approx(0.0)
    assert values[-1] == pytest.approx(1.0)
    assert values == sorted(values)


# ── I8：father 链 ──────────────────────────────────────────────


def test_father_index_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        PhigrosChart(
            lines=[JudgeLine(line_id=0, father=3)],
            bpm_points=[BpmPoint(time_beats=0.0, bpm=180.0)],
        )


def test_father_cycle_rejected() -> None:
    with pytest.raises(ValidationError):
        PhigrosChart(
            lines=[JudgeLine(line_id=0, father=1), JudgeLine(line_id=1, father=0)],
            bpm_points=[BpmPoint(time_beats=0.0, bpm=180.0)],
        )


def test_father_position_is_added_recursively() -> None:
    parent_offset = 10.0
    child_offset = 4.0
    chart = PhigrosChart(
        lines=[
            JudgeLine(
                line_id=0,
                event_layers=[
                    EventLayer(
                        move_x=[
                            EventKeyframe.model_validate(_kf(parent_offset, parent_offset, 0, 8))
                        ],
                    ),
                ],
            ),
            JudgeLine(
                line_id=1,
                father=0,
                event_layers=[
                    EventLayer(
                        move_x=[
                            EventKeyframe.model_validate(_kf(child_offset, child_offset, 0, 8))
                        ],
                    ),
                ],
            ),
        ],
        bpm_points=[BpmPoint(time_beats=0.0, bpm=180.0)],
    )
    assert chart.lines[1].pose_at(2.0, chart).x == pytest.approx(parent_offset + child_offset)


def test_local_to_stage_is_rigid_transform() -> None:
    chart = _chart()
    transform = chart.lines[0].local_to_stage(1.0, chart)
    # 长度为 0 的局部向量必须映射到锚点自身
    assert transform.apply(0.0, 0.0) == (
        pytest.approx(transform.translate_x),
        pytest.approx(transform.translate_y),
    )


# ── I3：越界只统计不钳位 ───────────────────────────────────────


def test_out_of_range_is_counted_not_clamped() -> None:
    """I3：越界只统计，positionX 原值一字不改地留在 IR 里。"""
    ir = minimal_ir()
    chart = _chart()
    stats = chart.out_of_visible_range()
    assert stats.count == 1
    assert stats.total == len(chart.notes)
    assert stats.fraction == pytest.approx(1.0 / len(chart.notes))
    assert stats.max_x > RPE_STAGE_HALF_WIDTH
    raw_notes = ir["notes"]
    assert isinstance(raw_notes, list)
    expected = max(float(n["position_x"]) for n in raw_notes if isinstance(n, dict))
    assert max(n.position_x for n in chart.notes) == expected


# ── M3 / I10：无损往返 ─────────────────────────────────────────


def test_json_roundtrip_is_lossless() -> None:
    chart = _chart()
    assert PhigrosChart.model_validate_json(chart.model_dump_json()) == chart


def test_raw_fields_survive_roundtrip() -> None:
    """above_raw 覆盖 {0,1,2}，type_raw / is_fake_raw 原样往返。"""
    chart = _chart()
    restored = PhigrosChart.model_validate_json(chart.model_dump_json())
    assert {n.above_raw for n in restored.notes} == {0, 1, 2}
    assert [n.type_raw for n in restored.notes] == [n.type_raw for n in chart.notes]
    assert [n.is_fake_raw for n in restored.notes] == [n.is_fake_raw for n in chart.notes]
    assert [n.side for n in restored.notes] == [
        side_from_above(n.above_raw) for n in restored.notes
    ]
    assert [n.type for n in restored.notes] == [
        note_type_from_rpe(n.type_raw) for n in restored.notes
    ]


def test_ir_json_is_compact_enough_for_fixture_budget() -> None:
    """M8：手写最小 IR 夹具 <= 32 KB。"""
    assert len(json.dumps(minimal_ir(), ensure_ascii=False).encode("utf-8")) <= 32 * 1024


# ── 确定性排序与 per-line 计数 ──────────────────────────────────


def test_sorted_notes_deterministic() -> None:
    keys = [(n.t, n.line_id, n.position_x) for n in _chart().sorted_notes()]
    assert keys == sorted(keys)


def test_notes_per_line_counts_empty_lines() -> None:
    chart = _chart()
    counts = chart.notes_per_line()
    assert len(counts) == len(chart.lines)
    assert sum(counts) == len(chart.notes)


def test_hold_semantics() -> None:
    chart = _chart()
    holds = [n for n in chart.notes if n.type is NoteType.HOLD]
    assert holds
    assert holds[0].hold_time > 0.0
    assert all(n.hold_time == 0.0 for n in chart.notes if n.type is not NoteType.HOLD)


def test_duration_uses_max_of_notes_and_chart_time() -> None:
    assert _chart().duration_s() == pytest.approx(30.0)


# ── 时间基本格（派生式，红线 7）────────────────────────────────


def test_tau_grid_is_derived() -> None:
    assert TAU_GRID_DT * SUBDIVISIONS_PER_BEAT == 1.0
    assert TAU_GRID_DT == 1.0 / SUBDIVISIONS_PER_BEAT


def test_bpm_points_must_be_sorted() -> None:
    with pytest.raises(ValidationError):
        BpmPoint(time_beats=-1.0, bpm=180.0)
    with pytest.raises(ValidationError):
        BpmPoint(time_beats=0.0, bpm=0.0)
    with pytest.raises(ValidationError):
        PhigrosChart(
            lines=[JudgeLine(line_id=0)],
            bpm_points=[
                BpmPoint(time_beats=8.0, bpm=180.0),
                BpmPoint(time_beats=0.0, bpm=200.0),
            ],
        )


def test_phigros_note_rejects_bad_alpha() -> None:
    with pytest.raises(ValidationError):
        PhigrosNote(
            line_id=0,
            t=0.0,
            position_x=0.0,
            side=Side.FRONT,
            type=NoteType.TAP,
            alpha=300,
        )
