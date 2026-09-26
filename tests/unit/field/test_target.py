"""M2：目标构建的计数守恒、hold-end 配对、三项计数与「不钳位」（默认 CI，无权重无 GPU）。"""

from __future__ import annotations

import numpy as np
import pytest

from beatmorph.core.contracts.field import TYPE_CHANNELS, x_bin_index
from beatmorph.core.contracts.phigros import (
    RPE_STAGE_HALF_WIDTH,
    NoteType,
    PhigrosChart,
    side_from_above,
    side_index,
)
from beatmorph.field.grid import BEAT_SUBDIVISION, FieldGrid
from beatmorph.field.target import CHANNEL_INDEX, HOLD_END_CHANNEL, build_target
from tests.unit.field._builders import (
    make_bpm_points,
    make_chart,
    make_note,
    seconds_for_beats,
)

#: 测试用 BPM 与谱面长度（任意取值，与物理常量无关）
TEST_BPM = 120.0
CHART_BEATS = 16.0
#: 合法 Hold 的起点与时长（拍）
HOLD_START_BEATS = 5.0
HOLD_LENGTH_BEATS = 1.0
#: 同格双击（n_j = 2）所在拍与位置
COCELL_BEATS = 2.0
COCELL_X = RPE_STAGE_HALF_WIDTH / 2.0
#: 越界 note 的原值（**不得钳位**）
OUT_OF_RANGE_X = RPE_STAGE_HALF_WIDTH * 2.0
#: Hold 起点位置
HOLD_X = -RPE_STAGE_HALF_WIDTH / 4.0
#: 正面侧索引（above == 1）
FRONT_INDEX = side_index(side_from_above(1))


def _seconds(beats: float) -> float:
    return seconds_for_beats(beats, TEST_BPM)


@pytest.fixture
def chart_and_grid() -> tuple[PhigrosChart, FieldGrid]:
    """一张覆盖 M2 全部边界的合成谱面 + 绑定好时间轴的网格。"""
    notes = [
        make_note(t=_seconds(0.0), position_x=0.0, line_id=0),
        make_note(t=_seconds(1.0), position_x=RPE_STAGE_HALF_WIDTH / 4.0, line_id=0),
        make_note(t=_seconds(COCELL_BEATS), position_x=COCELL_X, line_id=0),
        make_note(t=_seconds(COCELL_BEATS), position_x=COCELL_X, line_id=0),
        make_note(
            t=_seconds(3.0),
            position_x=-RPE_STAGE_HALF_WIDTH / 8.0,
            line_id=1,
            note_type=NoteType.DRAG,
            above=0,
        ),
        make_note(
            t=_seconds(4.0),
            position_x=RPE_STAGE_HALF_WIDTH / 8.0,
            line_id=1,
            note_type=NoteType.FLICK,
            above=2,
        ),
        make_note(
            t=_seconds(HOLD_START_BEATS),
            position_x=HOLD_X,
            line_id=2,
            note_type=NoteType.HOLD,
            hold_time=_seconds(HOLD_LENGTH_BEATS),
        ),
        make_note(
            t=_seconds(6.0),
            position_x=0.0,
            line_id=2,
            note_type=NoteType.HOLD,
            hold_time=0.0,
        ),
        make_note(t=_seconds(7.0), position_x=0.0, line_id=0, is_fake=True),
        make_note(t=_seconds(8.0), position_x=OUT_OF_RANGE_X, line_id=0),
        make_note(t=_seconds(9.0), position_x=0.0, line_id=2),
    ]
    bpm_points = make_bpm_points((0.0, TEST_BPM))
    chart = make_chart(
        notes=notes,
        bpm_points=bpm_points,
        k=3,
        chart_time_s=_seconds(CHART_BEATS),
    )
    return chart, FieldGrid().for_chart(chart)


def test_counts_shape_dtype_and_conservation(
    chart_and_grid: tuple[PhigrosChart, FieldGrid],
) -> None:
    """M2：sum_j n_j == meta.n_events（整数严格相等）；dtype int16；形状 (K,T,X,S,C)。"""
    chart, grid = chart_and_grid
    target = build_target(chart, grid)
    assert target.counts.dtype == np.int16
    assert target.counts.shape == (
        len(chart.lines),
        grid.t_bins,
        grid.x_bins,
        grid.sides,
        grid.channels,
    )
    assert int(target.counts.sum()) == target.meta.n_events
    target.assert_conservation()


def test_expected_event_count_from_construction(
    chart_and_grid: tuple[PhigrosChart, FieldGrid],
) -> None:
    """事件点数由构造独立推出：合法 note + 合法 Hold 的终点，越界 / fake 均排除。"""
    chart, grid = chart_and_grid
    target = build_target(chart, grid)
    accepted = [
        note
        for note in chart.notes
        if not note.is_fake and abs(note.position_x) <= RPE_STAGE_HALF_WIDTH
    ]
    expected = len(accepted) + sum(
        1 for note in accepted if note.type is NoteType.HOLD and note.hold_time > 0.0
    )
    assert target.meta.n_events == expected
    assert target.meta.n_notes == len(accepted)
    assert target.meta.n_fake == 1
    assert target.meta.n_out_of_range == 1
    assert target.meta.n_illegal_hold == 1


def test_hold_end_shares_fiber_with_its_start(
    chart_and_grid: tuple[PhigrosChart, FieldGrid],
) -> None:
    """M2：hold-end 与起点同 (k, i_x, s)，位于 hold 通道之后的 hold_end 通道。"""
    chart, grid = chart_and_grid
    target = build_target(chart, grid)
    x_index = x_bin_index(HOLD_X, target.spec)
    assert x_index is not None
    line_index = 2
    start_index = int(HOLD_START_BEATS * BEAT_SUBDIVISION)
    end_index = int((HOLD_START_BEATS + HOLD_LENGTH_BEATS) * BEAT_SUBDIVISION)
    hold_channel = CHANNEL_INDEX[NoteType.HOLD]
    assert target.counts[line_index, start_index, x_index, FRONT_INDEX, hold_channel] == 1
    assert target.counts[line_index, end_index, x_index, FRONT_INDEX, HOLD_END_CHANNEL] == 1
    assert end_index - start_index == BEAT_SUBDIVISION
    # 终点**不得**出现在 hold 通道，起点**不得**出现在 hold_end 通道
    assert target.counts[line_index, start_index, x_index, FRONT_INDEX, HOLD_END_CHANNEL] == 0
    assert target.counts[line_index, end_index, x_index, FRONT_INDEX, hold_channel] == 0


def test_illegal_hold_is_counted_and_gets_no_end_point(
    chart_and_grid: tuple[PhigrosChart, FieldGrid],
) -> None:
    """M2：非法 Hold（endTime <= startTime）只计入 n_illegal_hold，不补终点、不修复。"""
    chart, grid = chart_and_grid
    target = build_target(chart, grid)
    assert target.meta.n_illegal_hold == 1
    assert int(target.counts[:, :, :, :, HOLD_END_CHANNEL].sum()) == 1
    assert int(target.counts[:, :, :, :, CHANNEL_INDEX[NoteType.HOLD]].sum()) == 2


def test_out_of_range_is_counted_not_clamped(
    chart_and_grid: tuple[PhigrosChart, FieldGrid],
) -> None:
    """M2/红线 3：越界 note 只统计（保留原值），不进事件项，**不钳位**。"""
    chart, grid = chart_and_grid
    target = build_target(chart, grid)
    stats = chart.out_of_visible_range()
    assert target.meta.n_out_of_range == stats.count == 1
    assert target.meta.out_of_range_x == (OUT_OF_RANGE_X,)
    assert OUT_OF_RANGE_X > RPE_STAGE_HALF_WIDTH
    assert target.as_contract().out_of_window == target.meta.n_out_of_range


def test_fake_notes_are_excluded_by_default_and_counted() -> None:
    """fake 默认排除；include_fake=True 时计入事件项（计数无论如何都报）。"""
    bpm_points = make_bpm_points((0.0, TEST_BPM))
    chart = make_chart(
        notes=[
            make_note(t=_seconds(0.0), line_id=0),
            make_note(t=_seconds(1.0), line_id=0, is_fake=True),
        ],
        bpm_points=bpm_points,
        k=1,
        chart_time_s=_seconds(2.0),
    )
    grid = FieldGrid().for_chart(chart)
    excluded = build_target(chart, grid)
    included = build_target(chart, grid, include_fake=True)
    assert excluded.meta.n_fake == 1
    assert included.meta.n_fake == 1
    assert included.meta.n_events == excluded.meta.n_events + 1


def test_above_semantics_zero_and_two_are_both_back() -> None:
    """M2：above == 1 -> FRONT；其余（含 0 与 2）-> BACK；**禁止当布尔解析**。"""
    assert side_index(side_from_above(1)) == 0
    assert side_index(side_from_above(0)) == 1
    assert side_index(side_from_above(2)) == 1


def test_per_side_per_line_counts_match_construction(
    chart_and_grid: tuple[PhigrosChart, FieldGrid],
) -> None:
    """分层计数必须与构造一致（含空线计入 per_line）。"""
    chart, grid = chart_and_grid
    target = build_target(chart, grid)
    expected_line = [0] * len(chart.lines)
    expected_side = [0, 0]
    for note in chart.notes:
        if note.is_fake or abs(note.position_x) > RPE_STAGE_HALF_WIDTH:
            continue
        expected_line[note.line_id] += 1
        expected_side[side_index(note.side)] += 1
        if note.type is NoteType.HOLD and note.hold_time > 0.0:
            expected_line[note.line_id] += 1
            expected_side[side_index(note.side)] += 1
    assert list(target.meta.per_line) == expected_line
    assert list(target.meta.per_side) == expected_side
    assert len(target.meta.per_channel) == len(TYPE_CHANNELS)
    assert sum(target.meta.per_channel) == target.meta.n_events
    assert sum(target.meta.per_line) == target.meta.n_events


def test_co_cell_duplicate_keeps_count_two(chart_and_grid: tuple[PhigrosChart, FieldGrid]) -> None:
    """M2/M4：同格两个事件必须留下 n_j = 2（**禁止去重**）。"""
    chart, grid = chart_and_grid
    target = build_target(chart, grid)
    x_index = x_bin_index(COCELL_X, target.spec)
    assert x_index is not None
    index = int(COCELL_BEATS * BEAT_SUBDIVISION)
    assert target.counts[0, index, x_index, 0, CHANNEL_INDEX[NoteType.TAP]] == 2
    assert target.multi_event_cells() == 1
    assert target.occupied_cells() == int(np.count_nonzero(target.counts))


def test_notes_outside_tau_axis_are_counted_not_silently_dropped() -> None:
    """时间窗外的 note 必须计数（不得静默丢弃）。"""
    bpm_points = make_bpm_points((0.0, TEST_BPM))
    chart = make_chart(
        notes=[
            make_note(t=_seconds(0.0), line_id=0),
            make_note(t=_seconds(8.0), line_id=0),
        ],
        bpm_points=bpm_points,
        k=1,
        chart_time_s=_seconds(2.0),
    )
    short_grid = FieldGrid().for_chart(chart, tau_end_s=_seconds(2.0))
    target = build_target(chart, short_grid)
    assert target.meta.n_out_of_time == 1
    assert target.meta.n_events == 1


def test_to_tensor_matches_numpy_counts(chart_and_grid: tuple[PhigrosChart, FieldGrid]) -> None:
    """张量出口：形状 / dtype 与 numpy 计数一致（**不改变数值**）。"""
    torch = pytest.importorskip("torch")
    chart, grid = chart_and_grid
    target = build_target(chart, grid)
    tensor = target.to_tensor()
    assert isinstance(tensor, torch.Tensor)
    assert tuple(tensor.shape) == target.shape
    assert tensor.dtype == torch.int16
    assert int(tensor.sum().item()) == target.meta.n_events
