"""可视化 / 调试工具（Plan 03 §4.9 / M8）。

- render_field_png：每行一条判定线，列 = {above, below} × 5 通道；横轴时间（**秒**，
  不是帧索引、也不是 tau 索引），纵轴 positionX（RPE-x，标注边界）；颜色 = lambda；
  GT 事件以散点叠加；OcclusionMask 区域以阴影标出；RangeMask 外区域置灰。
- render_collision_report_png：精确同刻 |Delta x| 直方图 + 碰撞率-N 曲线。

契约：**无权重、无 GPU、脱离训练即可运行**；matplotlib 用 Agg 后端；
输出尺寸与文件名**确定性**（便于 CI 与人工比对）；日志走 core.logging，禁止裸 print。
matplotlib 为可选依赖（仅本文件用到）：未安装时模块仍可导入，调用时才抛 ImportError，
测试用 pytest.importorskip("matplotlib") 跳过。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
from numpy.typing import NDArray

from beatmorph.core.contracts.field import TYPE_CHANNELS
from beatmorph.core.logging import get_logger
from beatmorph.field.collision import CocellReport
from beatmorph.field.grid import FieldGrid, tau_to_seconds

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

logger = get_logger("field.viz")

FloatArray = NDArray[np.float64]

#: 单格面板的物理尺寸（英寸）与 DPI：确定性输出尺寸的唯一来源
PANEL_WIDTH_IN: Final[float] = 1.7
PANEL_HEIGHT_IN: Final[float] = 1.1
MARGIN_IN: Final[float] = 0.45
DEFAULT_DPI: Final[int] = 100
#: 一张图最多画多少条判定线（超出部分只记录告警，不静默改变布局）
MAX_LAYOUT_LINES: Final[int] = 8
#: 碰撞报告图相对单格面板的宽度倍数
COLLISION_WIDTH_PANELS: Final[float] = 3.2
#: 碰撞报告图相对单格面板的高度倍数
COLLISION_HEIGHT_PANELS: Final[float] = 2.4


def _as_numpy(value: np.ndarray | torch.Tensor) -> FloatArray:
    """torch / numpy -> numpy float64（本模块只做读操作，不建图）。"""
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=np.float64)
    return np.asarray(value.detach().cpu().numpy(), dtype=np.float64)


def field_panel_shape(
    grid: FieldGrid,
    k: int,
    *,
    max_lines: int = MAX_LAYOUT_LINES,
) -> tuple[int, int]:
    """确定性布局 (rows, cols)：行 = 判定线，列 = 侧别 × 通道。"""
    return max(1, min(int(k), int(max_lines))), grid.sides * grid.channels


def figure_size_inches(rows: int, cols: int) -> tuple[float, float]:
    """确定性图尺寸（英寸）= 面板尺寸 × 行列数 + 边距。"""
    return (cols * PANEL_WIDTH_IN + MARGIN_IN, rows * PANEL_HEIGHT_IN + MARGIN_IN)


def png_pixel_size(rows: int, cols: int, *, dpi: int = DEFAULT_DPI) -> tuple[int, int]:
    """确定性像素尺寸（CI 与人工比对用）。"""
    width_in, height_in = figure_size_inches(rows, cols)
    return round(width_in * dpi), round(height_in * dpi)


def default_field_png_name(grid: FieldGrid, *, k: int, tag: str = "field") -> str:
    """确定性文件名（**不含时间戳**，便于人工比对与 CI 快照）。"""
    return f"{tag}_K{k}_T{grid.t_bins}_X{grid.x_bins}.png"


def figure_digest(fig: Figure) -> str:
    """图内容的确定性摘要（同一输入必须给出同一摘要）。

    buffer_rgba 只由 Agg 系后端提供，故用 getattr 取（FigureCanvasBase 上没有声明），
    取不到即说明后端不是 Agg —— 直接报错而不是给出一个假摘要。
    """
    fig.canvas.draw()
    read_buffer = getattr(fig.canvas, "buffer_rgba", None)
    if read_buffer is None:  # pragma: no cover - 只有非 Agg 后端会走到
        raise RuntimeError("当前 matplotlib 后端不提供 buffer_rgba（本模块要求 Agg）")
    buffer = np.asarray(read_buffer())
    return hashlib.sha256(buffer.tobytes()).hexdigest()


def _shade_out_of_range(ax: Axes, grid: FieldGrid) -> None:
    """RangeMask 外的 x 桶置灰（网格恰好铺满可见范围时不会有任何一桶）。"""
    mask = grid.range_mask()
    if bool(mask.all()):
        return
    for center, inside in zip(grid.x_centers(), mask, strict=True):
        if not inside:
            ax.axhspan(center - grid.dx / 2.0, center + grid.dx / 2.0, color="0.85", zorder=0)


def build_field_figure(
    lam: np.ndarray | torch.Tensor,
    gt_counts: np.ndarray | torch.Tensor | None,
    *,
    grid: FieldGrid,
    occlusion_mask: np.ndarray | torch.Tensor | None = None,
    line_ids: Sequence[int] | None = None,
    title: str | None = None,
    dpi: int = DEFAULT_DPI,
    max_lines: int = MAX_LAYOUT_LINES,
    log_scale: bool = False,
) -> Figure:
    """构造场可视化图（不落盘；测试用 figure_digest 做确定性断言）。"""
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    if grid.t_bins <= 0 or not grid.bpm_points:
        raise ValueError("可视化需要已绑定时间轴的网格（t_bins > 0 且 bpm_points 非空）")
    values = _as_numpy(lam)
    expected = (grid.t_bins, grid.x_bins, grid.sides, grid.channels)
    if values.ndim != 5 or tuple(values.shape[1:]) != expected:
        raise ValueError(
            f"lam 必须是 (K, T, X, S, C) 且后四维为 {expected}，得到 {tuple(values.shape)}",
        )
    counts = None if gt_counts is None else _as_numpy(gt_counts)
    occlusion = None if occlusion_mask is None else _as_numpy(occlusion_mask)
    k = int(values.shape[0])
    if k > max_lines:
        logger.warning("只画前 %d 条判定线（共 %d 条）；max_lines 可调", max_lines, k)
    rows, cols = field_panel_shape(grid, k, max_lines=max_lines)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=figure_size_inches(rows, cols),
        dpi=dpi,
        squeeze=False,
    )
    sec_edges = np.asarray(tau_to_seconds(grid.tau_edges(), grid.bpm_points), dtype=np.float64)
    sec_centers = np.asarray(
        tau_to_seconds(grid.tau_centers(), grid.bpm_points),
        dtype=np.float64,
    )
    x_centers = grid.x_centers()
    extent = (float(sec_edges[0]), float(sec_edges[-1]), grid.x_min, grid.x_max)
    ids = list(range(k)) if line_ids is None else list(line_ids)
    tiny = float(np.finfo(np.float64).tiny)

    for row in range(rows):
        for col in range(cols):
            side = col // grid.channels
            channel = col % grid.channels
            ax = axes[row][col]
            panel = values[row, :, :, side, channel].T
            shown = np.log10(np.maximum(panel, tiny)) if log_scale else panel
            ax.imshow(
                shown,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap="magma",
                interpolation="nearest",
            )
            _shade_out_of_range(ax, grid)
            if occlusion is not None:
                ax.imshow(
                    occlusion[row, :, :, side, channel].T,
                    origin="lower",
                    aspect="auto",
                    extent=extent,
                    cmap="Greys",
                    alpha=0.35,
                    interpolation="nearest",
                )
            if counts is not None:
                hits = counts[row, :, :, side, channel] > 0
                if bool(hits.any()):
                    tau_index, x_index = np.nonzero(hits)
                    ax.scatter(
                        sec_centers[tau_index],
                        x_centers[x_index],
                        s=4.0,
                        c="cyan",
                        marker="o",
                        linewidths=0.0,
                        zorder=3,
                    )
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(grid.x_min, grid.x_max)
            ax.set_yticks([grid.x_min, 0.0, grid.x_max])
            ax.tick_params(labelsize=4.0)
            if row == 0:
                ax.set_title(
                    f"{'above' if side == 0 else 'below'}·{TYPE_CHANNELS[channel]}",
                    fontsize=6.0,
                )
            if col == 0:
                ax.set_ylabel(f"L{ids[row]}", fontsize=5.0)
            if row == rows - 1:
                ax.set_xlabel("t (s)", fontsize=5.0)
    heading = (
        title
        if title is not None
        else (
            f"lambda field | K={k} T={grid.t_bins} X={grid.x_bins} "
            f"| t in [0, {float(sec_edges[-1]):.3f}] s"
        )
    )
    fig.suptitle(heading, fontsize=7.0)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig


def render_field_png(
    lam: np.ndarray | torch.Tensor,
    gt_counts: np.ndarray | torch.Tensor | None,
    out_path: str | Path,
    *,
    grid: FieldGrid,
    occlusion_mask: np.ndarray | torch.Tensor | None = None,
    line_ids: Sequence[int] | None = None,
    title: str | None = None,
    dpi: int = DEFAULT_DPI,
    max_lines: int = MAX_LAYOUT_LINES,
    log_scale: bool = False,
) -> Path:
    """把场渲染成 PNG 并返回路径（无权重、无 GPU、Agg 后端、尺寸确定性）。"""
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    fig = build_field_figure(
        lam,
        gt_counts,
        grid=grid,
        occlusion_mask=occlusion_mask,
        line_ids=line_ids,
        title=title,
        dpi=dpi,
        max_lines=max_lines,
        log_scale=log_scale,
    )
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=dpi, format="png")
    plt.close(fig)
    logger.info("场可视化已写出：%s", target)
    return target


def collision_histogram_bins(minima: FloatArray) -> int:
    """直方图桶数（确定性：只由样本量决定，不依赖数据分布）。"""
    return max(1, int(np.ceil(np.sqrt(float(minima.size)))))


def collision_figure_size_inches() -> tuple[float, float]:
    """碰撞报告图的确定性尺寸（英寸）。"""
    return (PANEL_WIDTH_IN * COLLISION_WIDTH_PANELS, PANEL_HEIGHT_IN * COLLISION_HEIGHT_PANELS)


def build_collision_figure(
    report: CocellReport,
    *,
    title: str | None = None,
    dpi: int = DEFAULT_DPI,
) -> Figure:
    """构造碰撞报告的图（左：精确同刻 |Delta x| 直方图；右：碰撞率-N 曲线）。"""
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=collision_figure_size_inches(), dpi=dpi)
    minima = np.asarray(report.exact.min_abs_dx, dtype=np.float64)
    left, right = axes[0], axes[1]
    if minima.size:
        left.hist(minima, bins=collision_histogram_bins(minima), color="steelblue")
    else:
        left.text(0.5, 0.5, "no same-instant collisions", ha="center", va="center")
    left.set_xlabel("min |Delta positionX| (RPE-x)", fontsize=6.0)
    left.set_ylabel("groups", fontsize=6.0)
    left.set_title("exact same-instant collisions", fontsize=7.0)
    left.tick_params(labelsize=5.0)
    bins, rates = report.rate_curve()
    if bins:
        right.plot(range(len(bins)), rates, marker="o", color="firebrick")
        right.set_xticks(list(range(len(bins))))
        right.set_xticklabels([str(value) for value in bins])
    right.set_xlabel("N (x bins)", fontsize=6.0)
    right.set_ylabel("cocell rate", fontsize=6.0)
    right.set_title("cocell rate vs N", fontsize=7.0)
    right.tick_params(labelsize=5.0)
    heading = (
        title
        if title is not None
        else (
            f"collision report | n_events={report.n_events} "
            f"| required N={report.exact.n_required_bins}"
        )
    )
    fig.suptitle(heading, fontsize=7.0)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    return fig


def render_collision_report_png(
    report: CocellReport,
    out_path: str | Path,
    *,
    title: str | None = None,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """把共格碰撞报告渲染成 PNG 并返回路径（M6/M7 的曲线工具）。"""
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    fig = build_collision_figure(report, title=title, dpi=dpi)
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=dpi, format="png")
    plt.close(fig)
    logger.info("碰撞报告已写出：%s", target)
    return target


__all__ = [
    "DEFAULT_DPI",
    "MARGIN_IN",
    "MAX_LAYOUT_LINES",
    "PANEL_HEIGHT_IN",
    "PANEL_WIDTH_IN",
    "build_collision_figure",
    "build_field_figure",
    "collision_figure_size_inches",
    "collision_histogram_bins",
    "default_field_png_name",
    "field_panel_shape",
    "figure_digest",
    "figure_size_inches",
    "png_pixel_size",
    "render_collision_report_png",
    "render_field_png",
]
