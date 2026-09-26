"""判定线普通事件轨在 tau 轴上的求值（plan 02 → plan 04 §3.2 的数据侧接缝）。

本模块只做一件事：把契约层的 **跨层求和**（:meth:`~beatmorph.core.contracts.phigros.JudgeLine.sum_track`）
在 **tau（拍）轴**上求值成 generation 侧需要的张量 ``(K, T, N_ORDINARY_TRACKS)``。

三条纪律（不满足就会静默错位）：

1. **通道顺序由契约决定**：通道 j 就是 :data:`RPE_TRACK_FIELDS` 的第 j 项，
   本模块**不写死** `("move_x", ...)` 的顺序，也不写死通道数 5；
2. **跨层求和交给契约**：逐层取值会得到 4 x 5 = 20 通道，正是
   :data:`beatmorph.generation.batch.FORBIDDEN_LAYERED_TRACKS` 拦下的静默失效，
   因此本模块只调用 `JudgeLine.sum_track`，**自己不实现任何求和**；
3. **时间轴是 tau（拍）**：求值点是 `grid.tau_centers()`，秒 <-> tau 换算一律不在这里做
   （只有 :mod:`beatmorph.field.grid` 一个实现，红线 7）。

复杂度与实测耗时（口径写在 docstring 里，便于更换实现时对比）：
本模块是纯 Python 逐点求值，复杂度 `O(K * T * F * L * KF)`（L = 事件层数 <= 5、
KF = 命中关键帧数，`track_value` 命中即返回）。基准（Windows / CPython 3.11，
K = 71、F = 5、**每条线 2 层 x 每层 2 个关键帧**）：T = 192 格 0.159 s、
T = 480 格 0.393 s、T = 960 格 0.812 s，即约 **0.43e6 次 `sum_track`/s**
（关键帧越多越慢；层数/关键帧数直接进线性因子）。因此：

- **窗口级求值**（T = t_window，通常几百格）是设计用法，
  :class:`~beatmorph.data.dataset.ChartPairDataset` 就是按窗口求值的；
- 全谱求值（T 上千）只在诊断脚本里用，不要放进训练内循环。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from beatmorph.core.contracts import RPE_TRACK_FIELDS, PhigrosChart
from beatmorph.field.grid import FieldGrid

__all__ = ["line_tracks_at", "line_tracks_tensor"]

#: 普通轨通道数 = len(RPE_TRACK_FIELDS)（派生量，**不得**写死 5）
N_TRACK_CHANNELS: int = len(RPE_TRACK_FIELDS)


def line_tracks_at(chart: PhigrosChart, taus: NDArray[np.float64] | Sequence[float]) -> Tensor:
    """在给定的 **绝对 tau**（拍）序列上求值：返回 `(K, T, len(RPE_TRACK_FIELDS))` float32。

    Args:
        chart: 谱面 IR（判定线事件轨的唯一来源）。
        taus: 求值点（**拍**，绝对 tau 轴；窗口样本必须传
            `窗口起点 + grid.tau_centers()`，否则事件轨会整体平移）。

    Returns:
        `(K, T, F)` float32 张量；通道顺序 == :data:`RPE_TRACK_FIELDS`，
        跨层求和已完成（由 `JudgeLine.sum_track` 负责）。

    Note:
        本函数是 :func:`line_tracks_tensor` 的求值原语：后者就是
        `line_tracks_at(chart, grid.tau_centers())`。窗口数据集需要**绝对** tau
        （窗口局部网格的 tau 轴从 0 起），因此分成两层 API——两者**不复制**任何逻辑。
    """
    values: NDArray[np.float32] = np.empty(
        (len(chart.lines), len(taus), N_TRACK_CHANNELS),
        dtype=np.float32,
    )
    for line_index, line in enumerate(chart.lines):
        for field_index, field_name in enumerate(RPE_TRACK_FIELDS):
            values[line_index, :, field_index] = np.asarray(
                [line.sum_track(field_name, float(tau)) for tau in taus],
                dtype=np.float32,
            )
    return torch.from_numpy(values)


def line_tracks_tensor(chart: PhigrosChart, grid: FieldGrid) -> Tensor:
    """在 `grid.tau_centers()` 上求值：返回 `(K, T, len(RPE_TRACK_FIELDS))` float32。

    这是 plan 04 §3.2 输入契约里 `line_tracks` 的**单样本**形式
    （:class:`~beatmorph.generation.batch.FieldBatch` 的对应字段多一个 batch 轴）。

    Args:
        chart: 谱面 IR。
        grid: 已绑定 tau 轴的网格（`t_bins > 0`）；求值点是 `grid.tau_centers()`。

    Raises:
        ValueError: `grid.t_bins <= 0`（未绑定时间轴）。

    Note:
        复杂度与实测耗时见模块 docstring；窗口用法请用 :func:`line_tracks_at`
        并把窗口起点加到 tau 上。
    """
    if grid.t_bins <= 0:
        raise ValueError(f"line_tracks_tensor 需要已绑定 tau 轴的网格，得到 t_bins={grid.t_bins}")
    tracks = line_tracks_at(chart, grid.tau_centers())
    expected = (len(chart.lines), grid.t_bins, N_TRACK_CHANNELS)
    if tuple(tracks.shape) != expected:
        raise AssertionError(f"line_tracks 形状必须是 {expected}，得到 {tuple(tracks.shape)}")
    return tracks
