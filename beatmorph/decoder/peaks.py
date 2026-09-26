"""D1：峰值检测 + 阈值（v0 基线解码器）——Plan 05 §4.1 / RFC-0029 §3.4。

**这是消融臂 B6 的一侧，不是终点**（RFC-0029 §3.4 明确 D1 为基线）。设计约束逐条
对应 plan 05：

1. **同网格**：峰检测直接在 `(T, X)` 平面上做（`T` = τ 格数），不重采样——
   重采样等于引入第二个分辨率假设。
2. **阈值不是魔数**：`threshold = alpha * lambda_0`，`lambda_0 = N/|Omega|` 是 G3
   常数基线标度（`beatmorph.decoder.fieldops.intensity_scale`）；`alpha` 显式进配置
   并写进报告。文献实测阈值 0.5 与最优阈值之间 F1 差 0.23，故阈值必须声明。
3. **NMS 半径由网格派生**：默认 = **1 格**（τ 与 x 各 1 格），可配置为格数的整数倍；
   不使用像素、不使用秒为单位的裸半径（秒要经 `J(τ)` 折算，折算只在 `field/`）。
4. **短距双峰抑制**：Hamming 窗平滑，窗宽以**秒**表达，默认 = 1 个 MERT 帧
   （`1 / MERT_FRAME_RATE_HZ`，派生量）——即"音频证据能分辨的最小时间尺度"。
5. **置信度**（plan 05 §4.1-5，文献无对应做法）：`confidence = 峰高 / (alpha * lambda_0)`，
   被接受的峰恒 >= 1；它是**单调变换**，只用于 plan 04 迭代解码的优先级排序。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from beatmorph.core.contracts.field import TYPE_CHANNELS, ChartFieldSpec
from beatmorph.core.contracts.phigros import TAU_GRID_DT, Side, side_index
from beatmorph.core.contracts.tensors import MERT_FRAME_RATE_HZ
from beatmorph.decoder.events import FieldEvent
from beatmorph.decoder.fieldops import (
    FloatArray,
    assert_field_shape,
    intensity_scale,
    to_numpy,
)
from beatmorph.field.grid import FieldGrid

#: 默认平滑窗宽（秒）：1 个 MERT 帧 —— 派生量，不是魔数。
DEFAULT_SMOOTH_SECONDS: Final[float] = 1.0 / MERT_FRAME_RATE_HZ

#: Hamming 窗的标准系数（窗函数定义，非本项目物理量）。
_HAMMING_ALPHA: Final[float] = 0.54
_HAMMING_BETA: Final[float] = 0.46


@dataclass(frozen=True, slots=True)
class PeakConfig:
    """D1 的全部自由度（每一项都必须显式声明并写进报告）。

    Attributes:
        alpha: 阈值系数，`threshold = alpha * lambda_0`。
        nms_tau_bins: τ 轴 NMS 半径（格）。
        nms_x_bins: x 轴 NMS 半径（格）。
        smooth_seconds: Hamming 平滑窗宽（秒）；0 表示不平滑。
        n_events: 显式给定 N（G3 口径）；`None` 表示用场自身的 `int lambda` 自洽标度。
    """

    alpha: float = 1.0
    nms_tau_bins: int = 1
    nms_x_bins: int = 1
    smooth_seconds: float = DEFAULT_SMOOTH_SECONDS
    n_events: float | None = None

    def __post_init__(self) -> None:
        if self.alpha <= 0.0:
            raise ValueError(f"alpha 必须为正，得到 {self.alpha!r}")
        if self.nms_tau_bins < 1 or self.nms_x_bins < 1:
            raise ValueError("NMS 半径必须 >= 1 格（0 会让同一格内产生多个峰）")
        if self.smooth_seconds < 0.0:
            raise ValueError(f"smooth_seconds 必须 >= 0，得到 {self.smooth_seconds!r}")


def hamming_kernel(cells: int) -> FloatArray:
    """长度 `cells` 的归一化 Hamming 窗；`cells <= 1` 时退化为 `[1.0]`（不平滑）。"""
    if cells <= 1:
        return np.ones(1, dtype=np.float64)
    index = np.arange(cells, dtype=np.float64)
    window = _HAMMING_ALPHA - _HAMMING_BETA * np.cos(
        2.0 * np.pi * index / float(cells - 1),
    )
    return np.asarray(window / window.sum(), dtype=np.float64)


def smooth_cells(smooth_seconds: float, grid: FieldGrid) -> int:
    """平滑窗宽（秒）-> τ 格数（经 `J(τ)` 的**平均**值折算）。

    网格在变速曲下非均匀，故以全谱平均 `J` 折算并取最近整数格；
    折算用的 `J` 来自 `field/`（本模块不实现第二套换算）。
    """
    if smooth_seconds <= 0.0:
        return 1
    mean_jacobian = float(np.mean(grid.jacobian())) if grid.t_bins > 0 else 0.0
    if mean_jacobian <= 0.0:
        raise ValueError("jacobian 必须为正（网格未绑定时间轴？）")
    cells = max(1, round(smooth_seconds / (mean_jacobian * grid.d_tau)))
    # **强制奇数窗**：偶数长度的卷积核没有中心格，会把峰整体挪半格（系统性时间偏移）。
    return cells if cells % 2 == 1 else cells + 1


def _smooth_time_axis(plane: FloatArray, kernel: FloatArray) -> FloatArray:
    """沿 τ 轴（第 0 轴）做一维卷积；边界用**端点复制**（不引入零值假峰）。"""
    cells = kernel.size
    if cells <= 1:
        return plane
    left = cells // 2
    right = cells - 1 - left
    padded = np.pad(plane, ((left, right), (0, 0)), mode="edge")
    out = np.zeros_like(plane, dtype=np.float64)
    for index, weight in enumerate(kernel):
        out += weight * padded[index : index + plane.shape[0], :]
    return out


def _local_max_mask(plane: FloatArray, radius_tau: int, radius_x: int) -> NDArray[np.bool_]:
    """`(2*radius_tau+1, 2*radius_x+1)` 窗口内的**非严格**局部极大掩码。

    用 `>=` 而不是 `>`：严格极大在平坦高原上**无峰**，而常数场正是 G3 基线的
    形态；非严格版本把整个高原都标成候选，再由 `_greedy_peaks` 的 NMS 给出
    确定性的唯一代表点。

    "先取局部极大、再做 NMS"两步都不能省：只做 NMS（不加局部极大约束）会让
    每个高斯峰的**侧翼**每隔一个抑制半径就冒出一个假峰。
    """
    n_tau, n_x = plane.shape
    padded = np.pad(
        plane,
        ((radius_tau, radius_tau), (radius_x, radius_x)),
        mode="constant",
        constant_values=-np.inf,
    )
    best = np.full(plane.shape, -np.inf, dtype=np.float64)
    for offset_tau in range(2 * radius_tau + 1):
        for offset_x in range(2 * radius_x + 1):
            if offset_tau == radius_tau and offset_x == radius_x:
                continue
            window = padded[offset_tau : offset_tau + n_tau, offset_x : offset_x + n_x]
            np.maximum(best, window, out=best)
    return np.asarray(plane >= best, dtype=np.bool_)


def _greedy_peaks(
    plane: FloatArray,
    threshold: float,
    *,
    radius_tau: int,
    radius_x: int,
) -> tuple[NDArray[np.intp], NDArray[np.intp], FloatArray]:
    """**局部极大 + 阈值 + 贪心 NMS** 的峰值选择（确定性：值降序 -> τ 升序 -> x 升序）。"""
    mask = _local_max_mask(plane, radius_tau, radius_x) & (plane > threshold)
    candidates = np.argwhere(mask)
    if candidates.size == 0:
        empty_i = np.zeros(0, dtype=np.intp)
        return empty_i, empty_i, np.zeros(0, dtype=np.float64)
    values = plane[candidates[:, 0], candidates[:, 1]]
    order = np.lexsort((candidates[:, 1], candidates[:, 0], -values))  # 值降序 -> τ 升序 -> x 升序
    taken = np.zeros(plane.shape, dtype=bool)
    keep_tau: list[int] = []
    keep_x: list[int] = []
    keep_value: list[float] = []
    n_tau, n_x = plane.shape
    for position in order:
        tau_index = int(candidates[position, 0])
        x_index = int(candidates[position, 1])
        if taken[tau_index, x_index]:
            continue
        keep_tau.append(tau_index)
        keep_x.append(x_index)
        keep_value.append(float(plane[tau_index, x_index]))
        tau_lo = max(0, tau_index - radius_tau)
        tau_hi = min(n_tau, tau_index + radius_tau + 1)
        x_lo = max(0, x_index - radius_x)
        x_hi = min(n_x, x_index + radius_x + 1)
        taken[tau_lo:tau_hi, x_lo:x_hi] = True
    return (
        np.asarray(keep_tau, dtype=np.intp),
        np.asarray(keep_x, dtype=np.intp),
        np.asarray(keep_value, dtype=np.float64),
    )


def decode_peaks(
    lam: object,
    grid: FieldGrid,
    spec: ChartFieldSpec,
    *,
    config: PeakConfig | None = None,
) -> tuple[list[FieldEvent], dict[str, float]]:
    """D1：从强度场解出标记点（`FieldEvent` 列表）+ 诊断统计。

    Args:
        lam: 强度场 `(K, T, X, S, C)`（torch / numpy 均可）。
        grid: 已绑定时间轴的网格。
        spec: 网格规格（形状断言的唯一依据）。
        config: D1 自由度；省略时取默认（`alpha = 1`，1 MERT 帧平滑）。

    Returns:
        `(events, stats)`：events 按 `(line_id, tau, x_bin, side, channel)` 排序；
        stats 含 `scale` / `threshold` / `smooth_cells` / `n_peaks` 等，直接并入报告。
    """
    settings = PeakConfig() if config is None else config
    assert_field_shape(to_numpy(lam), spec)
    values = to_numpy(lam)
    if grid.t_bins <= 0 or not grid.bpm_points:
        raise ValueError("decode_peaks 需要已绑定时间轴的网格（t_bins > 0）")
    grid.assert_grid()
    spec.assert_grid()

    scale = intensity_scale(values, grid, n_events=settings.n_events)
    threshold = settings.alpha * scale
    cells = smooth_cells(settings.smooth_seconds, grid)
    kernel = hamming_kernel(cells)

    events: list[FieldEvent] = []
    for line_id in range(values.shape[0]):
        for side in Side:
            side_channel = side_index(side)
            for channel in range(values.shape[4]):
                plane = np.ascontiguousarray(values[line_id, :, :, side_channel, channel])
                if not np.any(plane > 0.0):
                    continue
                smoothed = _smooth_time_axis(plane, kernel)
                tau_index, x_index, peak_value = _greedy_peaks(
                    smoothed,
                    threshold,
                    radius_tau=settings.nms_tau_bins,
                    radius_x=settings.nms_x_bins,
                )
                for tau, x_bin, value in zip(tau_index, x_index, peak_value, strict=True):
                    events.append(
                        FieldEvent(
                            line_id=line_id,
                            tau=float(tau) * TAU_GRID_DT + TAU_GRID_DT / 2.0,
                            x_bin=int(x_bin),
                            side=side,
                            channel=channel,
                            confidence=float(value) / threshold,
                        ),
                    )
    events.sort(
        key=lambda event: (event.line_id, event.tau, event.x_bin, int(event.side), event.channel)
    )
    stats = {
        "d1_scale": float(scale),
        "d1_alpha": float(settings.alpha),
        "d1_threshold": float(threshold),
        "d1_smooth_cells": float(cells),
        "d1_nms_tau_bins": float(settings.nms_tau_bins),
        "d1_nms_x_bins": float(settings.nms_x_bins),
        "d1_n_events": float(len(events)),
    }
    return events, stats


def channel_name(channel: int) -> str:
    """通道索引 -> 通道名（诊断/报告用；`hold_end` 也在其中）。"""
    return TYPE_CHANNELS[int(channel)]


__all__ = [
    "DEFAULT_SMOOTH_SECONDS",
    "PeakConfig",
    "channel_name",
    "decode_peaks",
    "hamming_kernel",
    "smooth_cells",
]
