"""M6：共格碰撞统计工具；M7：N 消融的接口与曲线（默认 CI，无权重无 GPU）。"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import numpy as np
import pytest

from beatmorph.core.contracts.phigros import (
    RPE_STAGE_HALF_WIDTH,
    RPE_STAGE_WIDTH,
    PhigrosChart,
    PhigrosNote,
)
from beatmorph.field.collision import (
    ABLATION_METRIC_KEYS,
    bins_for_zero_collision,
    cocell_report,
    cocell_sweep,
    exact_collision_report,
    merge_ablation_metrics,
    min_same_instant_dx,
)
from beatmorph.field.grid import X_BIN_SWEEP, FieldGrid
from tests.unit.field._builders import make_bpm_points, make_chart, make_note, seconds_for_beats

TEST_BPM = 120.0
CHART_BEATS = 8.0
#: 同刻两个 note 的间隔（= 最细消融档位的桶宽，用于检验碰撞率随 N 下降）
FINE_DX = RPE_STAGE_WIDTH / max(X_BIN_SWEEP)


def _seconds(beats: float) -> float:
    return seconds_for_beats(beats, TEST_BPM)


def _chart(notes: Sequence[PhigrosNote], *, k: int = 2) -> PhigrosChart:
    return make_chart(
        notes=notes,
        bpm_points=make_bpm_points((0.0, TEST_BPM)),
        k=k,
        chart_time_s=_seconds(CHART_BEATS),
    )


def test_same_instant_same_line_same_side_is_detected() -> None:
    """同刻 + 同线 + 同侧：必须给出 |Delta positionX|。"""
    spacing = RPE_STAGE_WIDTH / 64.0
    chart = _chart(
        [
            make_note(t=_seconds(1.0), position_x=0.0, line_id=0),
            make_note(t=_seconds(1.0), position_x=spacing, line_id=0),
        ],
    )
    report = exact_collision_report(chart)
    assert len(report.groups) == 1
    assert report.min_abs_dx == (spacing,)
    assert report.p0 == pytest.approx(spacing)
    assert min_same_instant_dx(chart) == [spacing]
    assert report.n_notes == 2
    assert report.n_fake_excluded == 0


def test_three_notes_report_the_minimum_gap() -> None:
    """三个同刻 note：取最小相邻间隔。"""
    small = RPE_STAGE_WIDTH / 64.0
    chart = _chart(
        [
            make_note(t=_seconds(2.0), position_x=0.0, line_id=0),
            make_note(t=_seconds(2.0), position_x=small, line_id=0),
            make_note(t=_seconds(2.0), position_x=RPE_STAGE_HALF_WIDTH, line_id=0),
        ],
    )
    report = exact_collision_report(chart)
    assert report.min_abs_dx == (small,)
    assert report.groups[0].positions == (0.0, small, RPE_STAGE_HALF_WIDTH)


def test_different_instants_do_not_collide() -> None:
    """异刻即使位置相同也不算碰撞。"""
    chart = _chart(
        [
            make_note(t=_seconds(1.0), position_x=0.0, line_id=0),
            make_note(t=_seconds(1.5), position_x=0.0, line_id=0),
        ],
    )
    assert min_same_instant_dx(chart) == []
    assert exact_collision_report(chart).groups == ()


def test_different_sides_do_not_collide() -> None:
    """同刻同线但不同侧（above 1 vs 0）不算碰撞；同侧（above 0 vs 2）算。"""
    same_side = _chart(
        [
            make_note(t=_seconds(1.0), position_x=0.0, line_id=0, above=0),
            make_note(t=_seconds(1.0), position_x=RPE_STAGE_WIDTH / 64.0, line_id=0, above=2),
        ],
    )
    assert len(exact_collision_report(same_side).groups) == 1
    other_side = _chart(
        [
            make_note(t=_seconds(1.0), position_x=0.0, line_id=0, above=1),
            make_note(t=_seconds(1.0), position_x=RPE_STAGE_WIDTH / 64.0, line_id=0, above=0),
        ],
    )
    assert exact_collision_report(other_side).groups == ()


def test_different_lines_do_not_collide() -> None:
    """同刻同侧但不同判定线不算碰撞（多线共用同一 x 网格，但线身份不可互换）。"""
    chart = _chart(
        [
            make_note(t=_seconds(1.0), position_x=0.0, line_id=0),
            make_note(t=_seconds(1.0), position_x=RPE_STAGE_WIDTH / 64.0, line_id=1),
        ],
    )
    assert exact_collision_report(chart).groups == ()


def test_fake_notes_are_excluded_from_collision_stats() -> None:
    """假音符无判定，默认不参与碰撞统计；include_fake 可恢复。"""
    chart = _chart(
        [
            make_note(t=_seconds(1.0), position_x=0.0, line_id=0),
            make_note(t=_seconds(1.0), position_x=RPE_STAGE_WIDTH / 64.0, line_id=0, is_fake=True),
        ],
    )
    report = exact_collision_report(chart)
    assert report.n_fake_excluded == 1
    assert report.groups == ()
    assert len(exact_collision_report(chart, include_fake=True).groups) == 1


def test_bins_for_zero_collision_is_derived_from_stage_width() -> None:
    """不共格所需 N = ceil(RPE_STAGE_WIDTH / min|Delta x|)；0 间隔必须拒绝。"""
    spacing = RPE_STAGE_WIDTH / 64.0
    assert bins_for_zero_collision(spacing) == 64
    assert bins_for_zero_collision(RPE_STAGE_WIDTH / 3.0) == 3
    chart = _chart(
        [
            make_note(t=_seconds(1.0), position_x=0.0, line_id=0),
            make_note(t=_seconds(1.0), position_x=0.0, line_id=0),
        ],
    )
    with pytest.raises(ValueError, match="最小间隔必须为正"):
        _ = exact_collision_report(chart)
    with pytest.raises(ValueError, match="最小间隔必须为正"):
        bins_for_zero_collision(0.0)


def test_cocell_rate_curve_decreases_with_n() -> None:
    """M6/M7：碰撞率-N 曲线必须随 N 单调不增（并且**不是**单点结论）。"""
    chart = _chart(
        [
            make_note(t=_seconds(1.0), position_x=0.0, line_id=0),
            make_note(t=_seconds(1.0), position_x=FINE_DX, line_id=0),
        ],
    )
    grid = FieldGrid().for_chart(chart)
    sweep = cocell_sweep(chart, grid)
    assert tuple(row.x_bins for row in sweep) == X_BIN_SWEEP
    rates = [row.cocell_rate for row in sweep]
    assert rates[0] == pytest.approx(1.0)
    assert all(later <= earlier for earlier, later in pairwise(rates))
    assert rates[-1] == pytest.approx(0.0)
    assert sweep[-1].n_pairs_cocell == 0
    assert sweep[0].n_cells_ge2 == 1
    report = cocell_report(chart, grid)
    bins, curve = report.rate_curve()
    assert bins == X_BIN_SWEEP
    assert curve == tuple(rates)
    assert report.n_events == 2
    assert report.bpm_range == (TEST_BPM, TEST_BPM)
    assert "N=" in report.format()


def test_sweep_rows_carry_dx_and_tau_resolution() -> None:
    """每行必须带 dx 与 tau 分辨率（报告须同时给出 BPM 区间，plan §4.8-2）。"""
    chart = _chart([make_note(t=_seconds(1.0), position_x=0.0, line_id=0)])
    grid = FieldGrid().for_chart(chart)
    for row in cocell_sweep(chart, grid):
        assert row.dx == pytest.approx(RPE_STAGE_WIDTH / row.x_bins)
        assert row.t_bins == grid.t_bins
        assert row.n_events == 1
        assert row.cocell_rate == 0.0
        assert row.cell_fraction_ge2 == 0.0


def test_ablation_metrics_can_be_filled_in_by_downstream() -> None:
    """M7 接口：本模块只固定接口，NLL / F1 / 显存 / 耗时由 plan 04 / 06 回填。"""
    chart = _chart(
        [
            make_note(t=_seconds(1.0), position_x=0.0, line_id=0),
            make_note(t=_seconds(1.0), position_x=FINE_DX, line_id=0),
        ],
    )
    grid = FieldGrid().for_chart(chart)
    report = cocell_report(chart, grid)
    assert all(row.metrics == () for row in report.sweep)
    assert report.metric_curve("point_nll") == ((), ())
    filled_bins = X_BIN_SWEEP[:2]
    metrics = {
        value: {"point_nll": 1.5 * (index + 1), "binned_nll": 2.5, "f1_20ms": 0.75}
        for index, value in enumerate(filled_bins)
    }
    filled = merge_ablation_metrics(report, metrics)
    bins, values = filled.metric_curve("point_nll")
    assert bins == filled_bins
    assert values == (1.5, 3.0)
    assert filled.sweep[0].metric("f1_20ms") == 0.75
    assert filled.sweep[-1].metric("point_nll") is None
    with pytest.raises(ValueError, match="未知的消融指标键"):
        merge_ablation_metrics(report, {X_BIN_SWEEP[0]: {"unknown_metric": 1.0}})
    assert "point_nll" in ABLATION_METRIC_KEYS


def test_exact_report_quantiles_are_ordered() -> None:
    """多组同刻簇时 p0 <= p1 <= p5 <= 中位，且 p0 == 最小值。"""
    spacings = [RPE_STAGE_WIDTH / (64.0 * factor) for factor in (1, 2, 4, 8, 16)]
    notes = []
    for index, spacing in enumerate(spacings):
        notes.append(make_note(t=_seconds(1.0 + index), position_x=0.0, line_id=0))
        notes.append(make_note(t=_seconds(1.0 + index), position_x=spacing, line_id=0))
    report = exact_collision_report(_chart(notes))
    assert len(report.groups) == len(spacings)
    assert report.p0 == pytest.approx(min(spacings))
    assert report.p0 <= report.p1 <= report.p5 <= report.median
    assert report.n_required_bins == bins_for_zero_collision(min(spacings))
    assert np.isfinite(report.median)
