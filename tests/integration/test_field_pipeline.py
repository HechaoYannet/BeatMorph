"""场流水线集成测试：谱面 IR -> 网格 -> 目标 -> NLL -> 常数基线 -> 可视化。

全流程**无权重、无 GPU**（plan §8「集成」）。matplotlib 缺失时只跳过可视化那一步。
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from beatmorph.core.contracts.phigros import (
    RPE_STAGE_HALF_WIDTH,
    NoteType,
    PhigrosChart,
    side_from_above,
    side_index,
)
from beatmorph.field.collision import cocell_report
from beatmorph.field.grid import BEAT_SUBDIVISION, FieldGrid, seconds_to_tau, tau_to_seconds
from beatmorph.field.integrate import cell_volumes_tensor, integrate_grid, omega
from beatmorph.field.loss import (
    binned_point_gap,
    binned_poisson_nll,
    constant_baseline_nll,
    nll_decomposition,
    poisson_nll,
)
from beatmorph.field.target import build_target
from tests.unit.field._builders import make_bpm_points, make_chart, make_note

pytestmark = pytest.mark.integration

#: 多 BPM 段（含变速）：整条流水线必须在非均匀 tau 网格上成立
BPM_PAIRS: tuple[tuple[float, float], ...] = ((0.0, 120.0), (4.0, 180.0), (8.0, 150.0))
X_BINS = 32
K_LINES = 3
TOL = 1e-6


#: 谱面使用的 BPMList（秒 <-> 拍换算的唯一依据）
BPM_POINTS = make_bpm_points(*BPM_PAIRS)
#: 谱面总长（拍）
CHART_BEATS = 12.0


def _at(beats: float) -> float:
    """把拍坐标换成秒——**经被测模块的唯一换算入口**（不在测试里另写一套积分）。"""
    return float(tau_to_seconds(beats, BPM_POINTS))


def _chart() -> PhigrosChart:
    """合成谱面：覆盖正 / 背面、Hold、fake 与越界。"""
    notes = []
    for index, beats in enumerate((0.0, 1.0, 2.0, 3.0, 5.0, 6.5, 9.0, 11.0)):
        notes.append(
            make_note(
                t=_at(beats),
                position_x=RPE_STAGE_HALF_WIDTH * (index / 8.0 - 0.5),
                line_id=index % K_LINES,
                note_type=NoteType.TAP if index % 3 else NoteType.DRAG,
                above=1 if index % 4 else 0,
            ),
        )
    notes.append(
        make_note(
            t=_at(4.0),
            position_x=0.0,
            line_id=0,
            note_type=NoteType.HOLD,
            hold_time=_at(6.0) - _at(4.0),
        ),
    )
    notes.append(make_note(t=_at(7.0), position_x=0.0, line_id=0, is_fake=True))
    notes.append(
        make_note(t=_at(3.5), position_x=RPE_STAGE_HALF_WIDTH * 2.0, line_id=1),
    )
    return make_chart(
        notes=notes,
        bpm_points=BPM_POINTS,
        k=K_LINES,
        chart_time_s=_at(CHART_BEATS),
    )


def test_field_pipeline_end_to_end(tmp_path: Path) -> None:
    """M2 + M3 + M4 的串联：计数守恒 -> 两条路径一致 -> 优于常数基线 -> 碰撞报告。"""
    chart = _chart()
    grid = FieldGrid(x_bins=X_BINS).for_chart(chart)
    expected_bins = math.floor(
        seconds_to_tau(chart.duration_s(), chart.bpm_points) * BEAT_SUBDIVISION + 0.5,
    )
    assert grid.t_bins == expected_bins
    target = build_target(chart, grid)
    target.assert_conservation()
    assert target.counts.sum() == target.meta.n_events
    assert target.meta.n_fake == 1
    assert target.meta.n_out_of_range == 1

    # 模型场：真实事件的完美拟合（lambda = n / dV），非均匀测度下必须优于常数基线
    counts = target.to_tensor(dtype=torch.float64)
    volumes = cell_volumes_tensor(grid, dtype=torch.float64).reshape(1, -1, 1, 1, 1)
    perfect = torch.where(counts > 0, counts / volumes, torch.zeros_like(counts))
    per_line = integrate_grid(perfect, grid)
    n_events = float(counts.sum().item())
    total_volume = omega(grid, n_lines=K_LINES)
    assert float(per_line.sum().item()) == pytest.approx(n_events, rel=1e-9)
    model_nll = poisson_nll(counts, perfect, grid).item()
    baseline = constant_baseline_nll(n_events, total_volume)
    assert model_nll < baseline
    gap = (binned_poisson_nll(counts, perfect, grid) - poisson_nll(counts, perfect, grid)).item()
    assert gap == pytest.approx(binned_point_gap(counts, perfect, grid).item(), rel=TOL)
    decomposed = nll_decomposition(counts, perfect, grid)
    assert decomposed.total == pytest.approx(model_nll, rel=TOL)
    assert sum(decomposed.per_channel) == pytest.approx(model_nll, rel=TOL)


def test_field_pipeline_collision_and_seconds_roundtrip() -> None:
    """M6 碰撞统计与 M12 秒 <-> tau 往返在真实 BPMList 上同时成立。"""
    chart = _chart()
    grid = FieldGrid(x_bins=X_BINS).for_chart(chart)
    report = cocell_report(chart, grid)
    assert report.bpm_range == (min(bpm for _, bpm in BPM_PAIRS), max(bpm for _, bpm in BPM_PAIRS))
    assert report.n_events == build_target(chart, grid).meta.n_events
    assert len(report.rate_curve()[0]) >= 2
    for probe in (0.0, 1.0, 4.0, 8.0, 12.0):
        assert seconds_to_tau(tau_to_seconds(probe, chart.bpm_points), chart.bpm_points) == (
            pytest.approx(probe, rel=1e-12)
        )
    assert grid.total_beats == pytest.approx(grid.t_bins / BEAT_SUBDIVISION)
    assert side_index(side_from_above(0)) == side_index(side_from_above(2)) == 1


def test_field_layer_and_format_layer_agree_on_wellformed_bpm_list() -> None:
    """红线 7 的两个换算点（field/ 与 data/ 格式层）在规范 BPMList 上必须逐点一致。

    ⚠️ 只在**首段从 0 拍起**（RPE 规范与契约口径）时断言：首段起点 > 0 时两处的
    「首段之前」外推约定不同（field 从 0 拍起算，格式层从首段起点起算），
    该差异属 plan §9-13 尚未裁定的接缝问题（见回报的存疑清单）。
    """
    rpejson = pytest.importorskip("beatmorph.data.parsers.rpejson")
    for probe in (0.0, 1.0, 3.5, 4.0, 8.0, 11.9):
        field_value = tau_to_seconds(probe, BPM_POINTS)
        assert rpejson.beat_to_seconds(probe, BPM_POINTS) == pytest.approx(field_value, rel=1e-12)
        assert rpejson.seconds_to_beat(field_value, BPM_POINTS) == pytest.approx(probe, rel=1e-12)


def test_field_pipeline_visualization(tmp_path: Path) -> None:
    """M8：整条流水线的产物可以直接渲染成确定性 PNG。"""
    pytest.importorskip("matplotlib")
    from beatmorph.field.viz import field_panel_shape, png_pixel_size, render_field_png

    chart = _chart()
    grid = FieldGrid(x_bins=X_BINS).for_chart(chart)
    target = build_target(chart, grid)
    counts = target.to_tensor(dtype=torch.float64)
    volumes = cell_volumes_tensor(grid, dtype=torch.float64).reshape(1, -1, 1, 1, 1)
    lam = torch.where(counts > 0, counts / volumes, torch.zeros_like(counts))
    rows, cols = field_panel_shape(grid, len(chart.lines))
    out = render_field_png(lam, target.counts, tmp_path / "pipeline.png", grid=grid)
    assert out.exists()
    assert png_pixel_size(rows, cols)[0] > 0
