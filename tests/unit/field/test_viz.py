"""M8：可视化 / 调试工具（matplotlib Agg；无权重、无 GPU；输出确定性）。

matplotlib 未安装时整文件跳过（pytest.importorskip），并在回报中说明。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from beatmorph.core.contracts.phigros import RPE_STAGE_WIDTH
from beatmorph.field.collision import cocell_report
from beatmorph.field.grid import BEAT_SUBDIVISION, FieldGrid
from beatmorph.field.target import build_target
from tests.unit.field._builders import make_bpm_points, make_chart, make_note, seconds_for_beats

matplotlib = pytest.importorskip("matplotlib")
matplotlib_image = pytest.importorskip("matplotlib.image")
pytest.importorskip("matplotlib.pyplot")

from beatmorph.field.viz import (  # noqa: E402
    MARGIN_IN,
    MAX_LAYOUT_LINES,
    PANEL_HEIGHT_IN,
    PANEL_WIDTH_IN,
    build_collision_figure,
    build_field_figure,
    default_field_png_name,
    field_panel_shape,
    figure_digest,
    figure_size_inches,
    png_pixel_size,
    render_collision_report_png,
    render_field_png,
)

TEST_BPM = 120.0
T_BINS = BEAT_SUBDIVISION
X_BINS = 8
K_LINES = 2


def _grid() -> FieldGrid:
    grid = FieldGrid(x_bins=X_BINS).with_time(T_BINS, make_bpm_points((0.0, TEST_BPM)))
    grid.assert_grid()
    return grid


def _lam(grid: FieldGrid, *, scale: float = 1.0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(3)
    base = torch.rand(
        (K_LINES, grid.t_bins, grid.x_bins, grid.sides, grid.channels),
        generator=generator,
        dtype=torch.float64,
    )
    return base * scale


def _counts(grid: FieldGrid) -> torch.Tensor:
    counts = torch.zeros(
        (K_LINES, grid.t_bins, grid.x_bins, grid.sides, grid.channels),
        dtype=torch.int16,
    )
    counts[0, 0, 0, 0, 0] = 1
    counts[1, grid.t_bins // 2, grid.x_bins // 2, 1, 2] = 2
    return counts


def test_field_png_is_written_with_deterministic_size(tmp_path: Path) -> None:
    """PNG 必须可生成，且像素尺寸由模块常量确定性给出。"""
    grid = _grid()
    rows, cols = field_panel_shape(grid, K_LINES)
    out = render_field_png(_lam(grid), _counts(grid), tmp_path / "field.png", grid=grid)
    assert out.exists()
    assert out.suffix == ".png"
    image = matplotlib_image.imread(str(out))
    width, height = png_pixel_size(rows, cols)
    assert image.shape[:2] == (height, width)
    assert figure_size_inches(rows, cols) == (
        cols * PANEL_WIDTH_IN + MARGIN_IN,
        rows * PANEL_HEIGHT_IN + MARGIN_IN,
    )


def test_figure_digest_is_deterministic_and_content_sensitive() -> None:
    """同一输入 -> 同一摘要；不同 lambda -> 不同摘要（输出确定性契约）。"""
    grid = _grid()
    first = figure_digest(build_field_figure(_lam(grid), _counts(grid), grid=grid))
    second = figure_digest(build_field_figure(_lam(grid), _counts(grid), grid=grid))
    # 注意：整体等比缩放**不会**改变 imshow 自动归一化后的像素，因此用具名扰动
    perturbed = _lam(grid).clone()
    perturbed[0, 0, 0, 0, 0] += 10.0
    other = figure_digest(build_field_figure(perturbed, _counts(grid), grid=grid))
    assert first == second
    assert other != first


def test_default_file_name_is_deterministic_and_has_no_timestamp() -> None:
    """文件名确定性（不含时间戳），随网格变化。"""
    grid = _grid()
    name = default_field_png_name(grid, k=K_LINES)
    assert name == default_field_png_name(grid, k=K_LINES)
    assert name.endswith(".png")
    assert str(grid.t_bins) in name
    assert str(grid.x_bins) in name
    wider = FieldGrid(x_bins=X_BINS * 2).with_time(T_BINS, grid.bpm_points)
    assert default_field_png_name(wider, k=K_LINES) != name


def test_occlusion_mask_and_log_scale_are_supported(tmp_path: Path) -> None:
    """OcclusionMask 阴影与对数色标都必须可用（不得抛错）。"""
    grid = _grid()
    occlusion = np.zeros(
        (K_LINES, grid.t_bins, grid.x_bins, grid.sides, grid.channels),
        dtype=np.float64,
    )
    occlusion[:, : T_BINS // 2] = 1.0
    out = render_field_png(
        _lam(grid),
        _counts(grid),
        tmp_path / "occluded.png",
        grid=grid,
        occlusion_mask=occlusion,
        log_scale=True,
    )
    assert out.exists()


def test_panel_shape_respects_max_lines() -> None:
    """超出一张图的行数上限时只画前 max_lines 条（布局仍确定）。"""
    grid = _grid()
    rows, cols = field_panel_shape(grid, MAX_LAYOUT_LINES * 3, max_lines=MAX_LAYOUT_LINES)
    assert rows == MAX_LAYOUT_LINES
    assert cols == grid.sides * grid.channels
    single_rows, _ = field_panel_shape(grid, 0)
    assert single_rows == 1


def test_field_figure_rejects_wrong_shape() -> None:
    """形状不符必须抛（不得静默画出错位的图）。"""
    grid = _grid()
    with pytest.raises(ValueError, match=r"lam 必须是"):
        build_field_figure(np.zeros((K_LINES, T_BINS, X_BINS), dtype=np.float64), None, grid=grid)
    unbound = FieldGrid(x_bins=X_BINS)
    with pytest.raises(ValueError, match="已绑定时间轴"):
        build_field_figure(_lam(grid), None, grid=unbound)


def test_collision_report_png(tmp_path: Path) -> None:
    """碰撞报告（|Delta x| 直方图 + 碰撞率-N 曲线）必须可生成；空报告也不崩。"""
    spacing = RPE_STAGE_WIDTH / 64.0
    chart = make_chart(
        notes=[
            make_note(t=seconds_for_beats(1.0, TEST_BPM), position_x=0.0, line_id=0),
            make_note(t=seconds_for_beats(1.0, TEST_BPM), position_x=spacing, line_id=0),
            make_note(t=seconds_for_beats(2.0, TEST_BPM), position_x=0.0, line_id=1),
        ],
        bpm_points=make_bpm_points((0.0, TEST_BPM)),
        k=2,
        chart_time_s=seconds_for_beats(4.0, TEST_BPM),
    )
    grid = FieldGrid(x_bins=X_BINS).for_chart(chart)
    report = cocell_report(chart, grid)
    out = render_collision_report_png(report, tmp_path / "collision.png")
    assert out.exists()
    digest = figure_digest(build_collision_figure(report))
    assert digest == figure_digest(build_collision_figure(report))
    empty = make_chart(
        notes=[make_note(t=seconds_for_beats(1.0, TEST_BPM), position_x=0.0, line_id=0)],
        bpm_points=make_bpm_points((0.0, TEST_BPM)),
        k=1,
        chart_time_s=seconds_for_beats(2.0, TEST_BPM),
    )
    empty_report = cocell_report(empty, FieldGrid(x_bins=X_BINS).for_chart(empty))
    assert empty_report.exact.groups == ()
    assert render_collision_report_png(empty_report, tmp_path / "empty.png").exists()


def test_render_uses_agg_backend(tmp_path: Path) -> None:
    """Agg 后端（无显示环境 / 无 GPU 可用）必须被强制设置。"""
    grid = _grid()
    render_field_png(_lam(grid), None, tmp_path / "agg.png", grid=grid)
    assert matplotlib.get_backend().lower() == "agg"


def test_render_accepts_contract_target_counts(tmp_path: Path) -> None:
    """与目标构建串联：直接把 FieldTarget.counts 画出来（numpy int16）。"""
    chart = make_chart(
        notes=[make_note(t=seconds_for_beats(1.0, TEST_BPM), position_x=0.0, line_id=0)],
        bpm_points=make_bpm_points((0.0, TEST_BPM)),
        k=1,
        chart_time_s=seconds_for_beats(2.0, TEST_BPM),
    )
    grid = FieldGrid(x_bins=X_BINS).for_chart(chart)
    target = build_target(chart, grid)
    lam = np.zeros(target.shape, dtype=np.float64)
    out = render_field_png(lam, target.counts, tmp_path / "target.png", grid=grid)
    assert out.exists()
