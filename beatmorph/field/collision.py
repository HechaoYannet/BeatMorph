"""共格碰撞统计（M6）与 N 消融的接口 / 曲线工具（M7）（Plan 03 §3.5/§4.8）。

两个统计量**都报**：

1. **精确同刻碰撞**（units §7.3 的定义）：同一判定线 + **同一精确判定时刻** + 同侧 的
   note 对的最小 |Delta positionX|；报告 p0/p1/p5/中位，以及「需要多大 N 才能不共格」
   （N >= RPE_STAGE_WIDTH / min|Delta x|）。
2. **网格化后的共格率**：按 **tau 格（1/48 拍）**分桶后落进同一
   (k, i_tau, i_x, s, c) 格的事件对数与 n_j >= 2 的格子占比，对 N ∈ {64,128,256,512}
   各报一次；同时给出该谱的 BPM 区间（tau 分桶与秒的对应随 BPM 变化）。

**已可预判的结论**（防止把 128 当新魔数）：手工谱常见量化网格为 RPE_STAGE_WIDTH/60 与
/120，而任意浮点谱实测最小间隔约 0.32（survey §7.4.1）→ N = 512 仍不足以零碰撞。
因此「零碰撞」不是 N 的可行性判据：**M7 必须给碰撞率-N 曲线并声明可接受阈值**。

本模块复用 target.build_target 做分桶（**不复制**任何网格化逻辑），只做统计。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import TYPE_CHECKING, Final

import numpy as np
from numpy.typing import NDArray

from beatmorph.core.contracts.phigros import RPE_STAGE_WIDTH
from beatmorph.field.grid import X_BIN_SWEEP, FieldGrid
from beatmorph.field.target import build_target

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from beatmorph.core.contracts.phigros import PhigrosChart

#: M7 要求每 N 回填的下游指标键（本模块只固定**接口**，数值由 plan 04/06 提供）
ABLATION_METRIC_KEYS: Final[tuple[str, ...]] = (
    "point_nll",
    "binned_nll",
    "peak_mem_mb",
    "step_ms",
    "f1_20ms",
    "f1_50ms",
)


@dataclass(frozen=True, slots=True)
class SameInstantGroup:
    """同一 (line, 精确时刻, side) 上的 note 位置簇。"""

    line_id: int
    t: float
    side: str
    positions: tuple[float, ...]
    min_abs_dx: float


@dataclass(frozen=True, slots=True)
class ExactCollisionReport:
    """精确同刻碰撞的分布（p0 / p1 / p5 / 中位）与「不共格所需 N」。"""

    groups: tuple[SameInstantGroup, ...]
    min_abs_dx: tuple[float, ...]
    p0: float
    p1: float
    p5: float
    median: float
    n_required_bins: int
    n_notes: int
    n_fake_excluded: int

    @property
    def n_colliding_groups(self) -> int:
        """至少两个 note 同刻同侧的簇数。"""
        return len(self.groups)


@dataclass(frozen=True, slots=True)
class NAblationRow:
    """N 消融的一行（碰撞统计 + 可回填的下游指标）。

    metrics 的键见 ABLATION_METRIC_KEYS；本模块只提供接口与曲线，
    NLL / 显存 / 耗时 / F1 的**真实数值**由 plan 04 / plan 06 回填。
    """

    x_bins: int
    dx: float
    t_bins: int
    n_events: int
    n_pairs_cocell: int
    cocell_rate: float
    n_cells_ge2: int
    cell_fraction_ge2: float
    event_fraction_ge2: float
    metrics: tuple[tuple[str, float], ...] = ()

    def metric(self, name: str) -> float | None:
        """取回填的指标（未回填返回 None）。"""
        for key, value in self.metrics:
            if key == name:
                return value
        return None

    def with_metrics(self, metrics: Mapping[str, float]) -> NAblationRow:
        """回填下游指标（返回新行；未知键会被拒绝，避免指标名漂移）。"""
        unknown = sorted(set(metrics) - set(ABLATION_METRIC_KEYS))
        if unknown:
            raise ValueError(f"未知的消融指标键：{unknown}（见 ABLATION_METRIC_KEYS）")
        merged = dict(self.metrics)
        merged.update(metrics)
        return replace(self, metrics=tuple(sorted(merged.items())))


@dataclass(frozen=True, slots=True)
class CocellReport:
    """共格碰撞报告：精确分布 + N 消融扫参 + 每谱 BPM 区间。"""

    exact: ExactCollisionReport
    sweep: tuple[NAblationRow, ...]
    bpm_range: tuple[float, float]
    n_events: int

    def rate_curve(self) -> tuple[tuple[int, ...], tuple[float, ...]]:
        """碰撞率-N 曲线（M7 的两条曲线之一）。"""
        return tuple(row.x_bins for row in self.sweep), tuple(row.cocell_rate for row in self.sweep)

    def metric_curve(self, name: str) -> tuple[tuple[int, ...], tuple[float, ...]]:
        """指标-N 曲线（未回填该指标的档位被跳过）。"""
        bins: list[int] = []
        values: list[float] = []
        for row in self.sweep:
            value = row.metric(name)
            if value is None:
                continue
            bins.append(row.x_bins)
            values.append(value)
        return tuple(bins), tuple(values)

    def format(self) -> str:
        """渲染成可直接进报告 / 训练日志的多行文本。"""
        lines = [
            f"共格碰撞报告：n_events={self.n_events}，BPM 区间 "
            f"[{self.bpm_range[0]:g}, {self.bpm_range[1]:g}]",
            f"  精确同刻：簇数 {self.exact.n_colliding_groups}，"
            f"p0={self.exact.p0:.6g} p1={self.exact.p1:.6g} "
            f"p5={self.exact.p5:.6g} 中位={self.exact.median:.6g}",
            f"  不共格所需 N：{self.exact.n_required_bins}（N >= RPE_STAGE_WIDTH / min|dx|）",
            "  N 消融：",
        ]
        lines += [
            f"    N={row.x_bins:<4d} dx={row.dx:.6g} 共格对={row.n_pairs_cocell} "
            f"共格率={row.cocell_rate:.6f} n_j>=2 格占比={row.cell_fraction_ge2:.6f}"
            for row in self.sweep
        ]
        return "\n".join(lines)


def exact_collision_report(
    chart: PhigrosChart, *, include_fake: bool = False
) -> ExactCollisionReport:
    """精确同刻（同一 line + 同一判定时刻 + 同侧）的最小 |Delta positionX| 分布。

    假音符默认排除（无判定，不构成碰撞）；**只统计，不修改任何落点**（红线 3）。
    """
    buckets: dict[tuple[int, float, str], list[float]] = {}
    n_fake = 0
    n_notes = 0
    for note in chart.sorted_notes():
        if note.is_fake and not include_fake:
            n_fake += 1
            continue
        side = "front" if note.side.value > 0 else "back"
        buckets.setdefault((note.line_id, note.t, side), []).append(note.position_x)
        n_notes += 1
    groups: list[SameInstantGroup] = []
    for key in sorted(buckets):
        positions = buckets[key]
        if len(positions) < 2:
            continue
        ordered = sorted(positions)
        gaps = [b - a for a, b in pairwise(ordered)]
        groups.append(
            SameInstantGroup(
                line_id=key[0],
                t=key[1],
                side=key[2],
                positions=tuple(ordered),
                min_abs_dx=float(min(gaps)),
            ),
        )
    minima = [group.min_abs_dx for group in groups]
    if minima:
        quantiles = np.percentile(np.asarray(minima, dtype=np.float64), [0.0, 1.0, 5.0, 50.0])
        p0, p1, p5, median = (float(value) for value in quantiles)
    else:
        p0 = p1 = p5 = median = math.nan
    return ExactCollisionReport(
        groups=tuple(groups),
        min_abs_dx=tuple(minima),
        p0=p0,
        p1=p1,
        p5=p5,
        median=median,
        n_required_bins=bins_for_zero_collision(min(minima)) if minima else 0,
        n_notes=n_notes,
        n_fake_excluded=n_fake,
    )


def min_same_instant_dx(chart: PhigrosChart) -> list[float]:
    """同 (line, 精确时刻, side) 的最小 |Delta positionX|，逐簇一个值（plan §3.5）。"""
    return list(exact_collision_report(chart).min_abs_dx)


def bins_for_zero_collision(min_abs_dx: float) -> int:
    """不共格所需的最小桶数 N = ceil(RPE_STAGE_WIDTH / min|Delta x|)（plan §4.8-1）。"""
    if not min_abs_dx > 0.0:
        raise ValueError(f"最小间隔必须为正，得到 {min_abs_dx!r}（0 表示已完全重合）")
    return math.ceil(RPE_STAGE_WIDTH / float(min_abs_dx))


def _row(counts: NDArray[np.int16], x_bins: int, dx: float) -> NAblationRow:
    """由桶内计数张量算一档 N 的碰撞统计。"""
    flat = counts.reshape(-1).astype(np.int64)
    positive = flat[flat > 0]
    n_events = int(positive.sum())
    n_pairs = int(np.sum(positive * (positive - 1) // 2))
    total_pairs = n_events * (n_events - 1) // 2
    cells_ge2 = int(np.count_nonzero(flat >= 2))
    occupied = int(positive.size)
    events_ge2 = int(np.sum(positive[positive >= 2]))
    return NAblationRow(
        x_bins=x_bins,
        dx=dx,
        t_bins=int(counts.shape[1]),
        n_events=n_events,
        n_pairs_cocell=n_pairs,
        cocell_rate=float(n_pairs / total_pairs) if total_pairs else 0.0,
        n_cells_ge2=cells_ge2,
        cell_fraction_ge2=float(cells_ge2 / occupied) if occupied else 0.0,
        event_fraction_ge2=float(events_ge2 / n_events) if n_events else 0.0,
    )


def cocell_sweep(
    chart: PhigrosChart,
    grid: FieldGrid,
    *,
    n_bins: Sequence[int] = X_BIN_SWEEP,
) -> tuple[NAblationRow, ...]:
    """N ∈ n_bins 的碰撞率扫描（分桶复用 build_target，**不复制**网格化逻辑）。"""
    rows: list[NAblationRow] = []
    for bins in n_bins:
        sub_grid = FieldGrid(
            x_bins=int(bins),
            t_bins=grid.t_bins,
            bpm_points=grid.bpm_points,
        )
        sub_grid.assert_grid()
        counts = build_target(chart, sub_grid).counts
        rows.append(_row(counts, int(bins), sub_grid.dx))
    return tuple(rows)


def cocell_report(
    chart: PhigrosChart,
    grid: FieldGrid,
    *,
    n_bins: Sequence[int] = X_BIN_SWEEP,
) -> CocellReport:
    """M6 的完整报告：精确同刻分布 + 每 N 的共格率 + BPM 区间。"""
    exact = exact_collision_report(chart)
    sweep = cocell_sweep(chart, grid, n_bins=n_bins)
    bpms = [point.bpm for point in chart.bpm_points]
    return CocellReport(
        exact=exact,
        sweep=sweep,
        bpm_range=(float(min(bpms)), float(max(bpms))) if bpms else (0.0, 0.0),
        n_events=sweep[0].n_events if sweep else 0,
    )


def merge_ablation_metrics(
    report: CocellReport,
    metrics_by_n: Mapping[int, Mapping[str, float]],
) -> CocellReport:
    """把下游（plan 04/06）的指标回填进消融表（M7 的接口，本模块不产出这些数值）。"""
    rows = tuple(
        row.with_metrics(metrics_by_n[row.x_bins]) if row.x_bins in metrics_by_n else row
        for row in report.sweep
    )
    return replace(report, sweep=rows)


__all__ = [
    "ABLATION_METRIC_KEYS",
    "CocellReport",
    "ExactCollisionReport",
    "NAblationRow",
    "SameInstantGroup",
    "bins_for_zero_collision",
    "cocell_report",
    "cocell_sweep",
    "exact_collision_report",
    "merge_ablation_metrics",
    "min_same_instant_dx",
]
