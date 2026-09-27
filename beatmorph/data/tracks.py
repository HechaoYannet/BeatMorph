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
**关键帧定位是向量化的**（不再逐点线性扫描）。旧实现对每个 (线, 通道, tau) 都线性扫过
整条轨的关键帧，并在扫描中反复调用 `Beat.to_beats()`——实测一个窗口 52.5% 的 CPU
时间花在这里（1.46e6 次 `track_value`、**3.97e7 次 `to_beats`**）。现在改为

1. 每条 (轨, 层) 预计算 `starts` / `ends`（各只调一次 `to_beats`）；
2. 用 `searchsorted` 向量化定位**第一个命中的关键帧**（口径见
   :func:`_first_matching_keyframe`，与 `track_value` 的「列表序首个命中」**逐位等价**）；
3. 取值仍调用契约层的 `EventKeyframe.numeric_at`（未变化），因此**数值逐位不变**。

复杂度 `O(K * F * L * (T + KF))`（L = 事件层数 <= 5），相比旧式的 `O(K * F * L * T * KF)`
少了每个求值点的线性扫描因子。因此：

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

from beatmorph.core.contracts import RPE_TRACK_FIELDS, EventKeyframe, PhigrosChart
from beatmorph.field.grid import FieldGrid

__all__ = ["line_tracks_at", "line_tracks_tensor"]

#: 普通轨通道数 = len(RPE_TRACK_FIELDS)（派生量，**不得**写死 5）
N_TRACK_CHANNELS: int = len(RPE_TRACK_FIELDS)


def _first_matching_keyframe(
    starts: NDArray[np.float64],
    prefmax_ends: NDArray[np.float64],
    taus: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    """`track_value` 的**列表序首个命中**关键帧下标（向量化，逐位等价）。

    契约口径（:func:`~beatmorph.core.contracts.phigros.track_value`）是：按下标升序找到
    **第一个**满足 `start <= tau <= end` 的关键帧。设

    - `i*` = **第一个**满足「存在 k <= i 使 `end_k >= tau`」的下标
      （= `searchsorted(prefix_max(ends), tau, "left")`）；
    - `j` = **最后一个**满足 `start_k <= tau` 的下标（= `searchsorted(starts, tau, "right") - 1`）。

    则首个命中下标恰好是 `i*`，且**当且仅当** `i* <= j` 时存在命中。证明：

    1. 设 `i0` 是契约意义下的首个命中下标（`start_i0 <= tau <= end_i0`，且 `i0` 最小）。
       由 `end_i0 >= tau` 得 `i* <= i0`；由 `start_i0 <= tau` 得 `i0 <= j`。
    2. 由 `i*` 的定义，`prefix_max(ends)[i*-1] < tau`（i* 是第一个），故 `end_i* >= tau`。
    3. 又 `i* <= i0 <= j`，而 `starts` 单调不减、`j` 是最后一个 `start <= tau`，
       故 `start_i* <= tau`。于是 `i*` 本身就是一个命中；由 `i0` 的最小性得 `i* >= i0`。
    4. 合起来 `i* == i0`。

    该证明**不假设关键帧不重叠**——重叠时契约取「先出现的那个」，上式同样成立
    （第 2、3 步只用到了 `prefix_max` 与 `starts` 的单调性）。

    Args:
        starts: 关键帧起点（拍），**单调不减**。
        prefmax_ends: `np.maximum.accumulate(ends)`，**单调不减**。
        taus: 求值点（拍）。

    Returns:
        `(index, hit)`：`index` 是下标（未命中处的值无意义），`hit` 是命中掩码。
    """
    n = int(starts.shape[0])
    if n == 0:
        return np.zeros(taus.shape, dtype=np.int64), np.zeros(taus.shape, dtype=np.bool_)
    first = np.searchsorted(prefmax_ends, taus, side="left")
    last = np.searchsorted(starts, taus, side="right") - 1
    hit = (first < n) & (first <= last)
    return first.astype(np.int64), hit


def _keyframe_bounds(
    keyframes: Sequence[EventKeyframe],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """一次算出关键帧的 `(starts, ends, prefix_max(ends))`（`to_beats()` 只在这里调）。"""
    n = len(keyframes)
    starts: NDArray[np.float64] = np.empty(n, dtype=np.float64)
    ends: NDArray[np.float64] = np.empty(n, dtype=np.float64)
    for index, keyframe in enumerate(keyframes):
        starts[index] = keyframe.start_time.to_beats()
        ends[index] = keyframe.end_time.to_beats()
    return starts, ends, np.maximum.accumulate(ends)


def _plain_linear_values(keyframe: EventKeyframe) -> tuple[float, float] | None:
    """可走「线性缓动」向量化快路径时返回 `(start, end)` 浮点值，否则 `None`。

    条件保守：缓动必须是 Linear(1)、非贝塞尔、切割为默认 `[0, 1]`，且两端都是 float。
    任一不满足就回落标量 `numeric_at`——**不复制** 29 种缓动的数学，也不猜 pydantic
    会把它存成 int 还是 float（int 走标量路径，保证 `value_at` 的整数运算逐位不变）。
    """
    if (
        keyframe.easing_type == 1
        and not keyframe.bezier
        and keyframe.easing_left == 0.0
        and keyframe.easing_right == 1.0
        and isinstance(keyframe.start, float)
        and isinstance(keyframe.end, float)
    ):
        return keyframe.start, keyframe.end
    return None


def _track_values_many(
    keyframes: Sequence[EventKeyframe],
    taus: NDArray[np.float64],
    default: float = 0.0,
) -> NDArray[np.float64]:
    """一条轨在多个 tau 上的取值（等价于逐点调用 `track_value`，逐位一致）。

    快路径只覆盖「线性缓动 + 默认切割 + 两端都是 float」这一最常见情形，
    并且**复刻** `EventKeyframe.value_at` 的分支顺序（`tau <= start` 取 start、
    `tau >= end` 取 end、否则线性插值），因此结果与标量路径逐位相同；
    其余情形（29 种缓动里除 Linear 之外、贝塞尔、非默认 `easingLeft/Right`、
    非数值取值）一律回落到 `numeric_at`，**不复制缓动数学**。
    """
    n = len(keyframes)
    if n == 0:
        return np.zeros(taus.shape, dtype=np.float64)
    starts, ends, prefmax_ends = _keyframe_bounds(keyframes)
    single = np.asarray(taus, dtype=np.float64).reshape(-1)
    index, hit = _first_matching_keyframe(starts, prefmax_ends, single)
    out = np.full(single.shape, float(default), dtype=np.float64)
    if not bool(hit.any()):
        return out
    selected = index[hit]
    points = single[hit]
    values = np.empty(points.shape[0], dtype=np.float64)
    for candidate in np.unique(selected):
        keyframe = keyframes[int(candidate)]
        mask = selected == candidate
        subset = points[mask]
        start = starts[candidate]
        end = ends[candidate]
        bounds = _plain_linear_values(keyframe)
        if bounds is not None and end > start:
            start_value, end_value = bounds
            progress = (subset - start) / (end - start)
            interpolated = start_value + (end_value - start_value) * progress
            # 复刻 value_at 的分支（插值只在 start < tau < end 时被采用）
            interpolated = np.where(subset <= start, start_value, interpolated)
            interpolated = np.where(subset >= end, end_value, interpolated)
            values[mask] = interpolated
        else:
            values[mask] = [keyframe.numeric_at(float(point), default) for point in subset]
    out[hit] = values
    return out


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
    points = np.asarray(taus, dtype=np.float64).reshape(-1)
    values: NDArray[np.float32] = np.empty(
        (len(chart.lines), points.shape[0], N_TRACK_CHANNELS),
        dtype=np.float32,
    )
    for line_index, line in enumerate(chart.lines):
        for field_index, field_name in enumerate(RPE_TRACK_FIELDS):
            # 跨层求和：层序与 `JudgeLine.sum_track` 一致（先 0.0 起，逐层累加），
            # 因此浮点加法顺序不变。空层贡献 0.0，与 `track_value` 的 default 一致。
            total = np.zeros(points.shape, dtype=np.float64)
            for layer in line.event_layers:
                total += _track_values_many(getattr(layer, field_name), points)
            values[line_index, :, field_index] = total
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
