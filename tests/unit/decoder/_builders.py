"""decoder 单元测试的合成构造器。

**纪律**（AGENTS.md §3.3 / 红线 7）：夹具与 mock **不得固化物理常量**。时间一律由
BPM 与契约的 `TAU_GRID_DT` 派生，x 一律由 `RPE_STAGE_WIDTH / x_bins` 派生，
窗宽一律引用 `MERT_FRAME_RATE_HZ` —— 需要"某个数字"时先问它是哪个契约常量的函数。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from beatmorph.core.contracts.field import TYPE_CHANNELS, ChartFieldSpec
from beatmorph.core.contracts.phigros import (
    RPE_STAGE_HALF_WIDTH,
    RPE_STAGE_WIDTH,
    SUBDIVISIONS_PER_BEAT,
    TAU_GRID_DT,
    BpmPoint,
    ChartMeta,
    ChartSource,
    JudgeLine,
    NoteType,
    PhigrosChart,
    Side,
)
from beatmorph.decoder.events import DecodedEvent, FieldEvent
from beatmorph.field.grid import (
    DEFAULT_X_BINS,
    SECONDS_PER_MINUTE,
    FieldGrid,
    tau_to_seconds,
)
from beatmorph.field.target import CHANNEL_INDEX, HOLD_END_CHANNEL

FloatArray = NDArray[np.float64]

#: 合成场的默认逐格幅值（**无量纲的强度单位**；不含任何物理常量）。
DEFAULT_AMPLITUDE: float = 1.0


def seconds_for_beats(beats: float, bpm: float) -> float:
    """单一 BPM 段内 beat -> 秒（测试用解析式，独立于被测实现）。"""
    return beats * SECONDS_PER_MINUTE / bpm


def make_bpm_points(*segments: tuple[float, float]) -> list[BpmPoint]:
    """(拍, BPM) 序列 -> BpmPoint 列表。"""
    return [BpmPoint(time_beats=beats, bpm=bpm) for beats, bpm in segments]


def make_grid(
    *,
    bpm_points: Sequence[BpmPoint],
    t_bins: int,
    x_bins: int | None = None,
) -> FieldGrid:
    """已绑定时间轴的网格（`x_bins=None` 时用契约默认值）。"""
    grid = FieldGrid(x_bins=DEFAULT_X_BINS if x_bins is None else x_bins)
    bound = grid.with_time(t_bins, bpm_points)
    bound.assert_grid()
    return bound


def spec_for(grid: FieldGrid, k: int) -> ChartFieldSpec:
    """网格 + K -> 契约规格（含 `assert_grid`）。"""
    return grid.spec(k)


def empty_field(k: int, spec: ChartFieldSpec) -> FloatArray:
    """全 0 场 `(K, T, X, S, C)`（float64）。"""
    return np.zeros((k, *spec.shape()[1:]), dtype=np.float64)


def tau_of_bin(tau_bin: int) -> float:
    """τ 格索引 -> 格中心（拍）。"""
    return (int(tau_bin) + 0.5) * TAU_GRID_DT


def place_gaussian(
    field: FloatArray,
    *,
    line_id: int,
    tau_bin: int,
    x_bin: int,
    side: Side = Side.FRONT,
    note_type: NoteType = NoteType.TAP,
    sigma_tau: float = 1.0,
    sigma_x: float = 1.0,
    amplitude: float = DEFAULT_AMPLITUDE,
    offset_tau: float = 0.0,
    offset_x: float = 0.0,
    channel: int | None = None,
) -> None:
    """在场上叠加一个窄高斯（M5.2 的"已知事件的合成场"）。

    Args:
        sigma_tau / sigma_x: 标准差（**格**为单位）。
        offset_tau / offset_x: 峰心相对格中心的偏移（格），用于制造"峰不在格中心"
            的非对称情形（解码只能给到格中心，误差 <= 半格）。
        channel: 显式指定通道（用于 `hold_end` 这类**标记通道**——它不是音符类型）。
    """
    n_tau, n_x = field.shape[1], field.shape[2]
    tau_axis = np.arange(n_tau, dtype=np.float64)[:, None]
    x_axis = np.arange(n_x, dtype=np.float64)[None, :]
    center_tau = float(tau_bin) + offset_tau
    center_x = float(x_bin) + offset_x
    bump = amplitude * np.exp(
        -0.5 * ((tau_axis - center_tau) / sigma_tau) ** 2
        - 0.5 * ((x_axis - center_x) / sigma_x) ** 2,
    )
    side_channel = 0 if side is Side.FRONT else 1
    target_channel = CHANNEL_INDEX[note_type] if channel is None else int(channel)
    field[line_id, :, :, side_channel, target_channel] += bump


def truth_from_bins(
    *,
    grid: FieldGrid,
    line_id: int,
    tau_bin: int,
    x_bin: int,
    side: Side = Side.FRONT,
    note_type: NoteType = NoteType.TAP,
) -> DecodedEvent:
    """由格索引构造"真值事件"（τ = 格中心，x = 桶中心）。"""
    spec = grid.spec(line_id + 1)
    return DecodedEvent(
        t_s=float(tau_to_seconds(tau_of_bin(tau_bin), grid.bpm_points)),
        line_id=line_id,
        position_x=spec.x_min + (x_bin + 0.5) * spec.dx,
        side=side,
        note_type=note_type,
        hold_time_s=0.0,
        is_fake=False,
        confidence=1.0,
    )


def event_id(event: DecodedEvent) -> tuple[int, int, int]:
    """事件的类别标识 `(line_id, side_index, note_type)`（时间之外的全部匹配维度）。"""
    return (event.line_id, 0 if event.side is Side.FRONT else 1, int(event.note_type))


def field_event_id(event: FieldEvent) -> tuple[int, int, int, int]:
    """场事件的类别标识 `(line_id, side_index, channel)`。"""
    return (event.line_id, 0 if event.side is Side.FRONT else 1, event.channel, event.x_bin)


def timing_f1(
    truth: Sequence[DecodedEvent],
    predicted: Sequence[DecodedEvent],
    *,
    tol_s: float,
) -> tuple[float, float, float]:
    """按类别的贪心时间匹配 F1：返回 `(f1, precision, recall)`。

    匹配规则：同一 `(line_id, side, type)` 组内按时间升序双指针，`|dt| <= tol_s`
    即命中（每个真值/预测最多匹配一次）。这是 plan 05 M5.2 的 timing-F1 口径，
    **不涉及模型质量**（合成场的真值来自构造，不是模型输出）。
    """
    matched = 0
    for key in {event_id(item) for item in truth} | {event_id(item) for item in predicted}:
        left = sorted(item.t_s for item in truth if event_id(item) == key)
        right = sorted(item.t_s for item in predicted if event_id(item) == key)
        i = 0
        j = 0
        while i < len(left) and j < len(right):
            if abs(left[i] - right[j]) <= tol_s:
                matched += 1
                i += 1
                j += 1
            elif left[i] < right[j]:
                i += 1
            else:
                j += 1
    precision = matched / len(predicted) if predicted else 0.0
    recall = matched / len(truth) if truth else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return f1, precision, recall


def max_abs_dt(truth: Sequence[DecodedEvent], predicted: Sequence[DecodedEvent]) -> float:
    """同一类别、成对（按序）时的最大时间误差（诊断用）。"""
    worst = 0.0
    for key in {event_id(item) for item in truth}:
        left = sorted(item.t_s for item in truth if event_id(item) == key)
        right = sorted(item.t_s for item in predicted if event_id(item) == key)
        for first, second in zip(left, right, strict=False):
            worst = max(worst, abs(first - second))
    return worst


def make_template(
    *,
    k: int,
    bpm_points: Sequence[BpmPoint],
    chart_time_s: float = 0.0,
) -> PhigrosChart:
    """模板谱面：K 条空判定线 + BPMList（判定线由条件输入提供，RFC-0029 Q2）。"""
    return PhigrosChart(
        lines=[JudgeLine(line_id=index) for index in range(k)],
        notes=[],
        bpm_points=list(bpm_points),
        meta=ChartMeta(chart_time_s=chart_time_s),
        source=ChartSource(sniff_evidence="synthetic-template"),
    )


def max_abs_x(spec: ChartFieldSpec) -> float:
    """x 桶中心的最大绝对值（越界夹具用；派生自半宽与桶宽）。"""
    return abs(spec.x_min) + spec.dx


__all__ = [
    "CHANNEL_INDEX",
    "DEFAULT_AMPLITUDE",
    "HOLD_END_CHANNEL",
    "RPE_STAGE_HALF_WIDTH",
    "RPE_STAGE_WIDTH",
    "SUBDIVISIONS_PER_BEAT",
    "TAU_GRID_DT",
    "TYPE_CHANNELS",
    "empty_field",
    "event_id",
    "field_event_id",
    "make_bpm_points",
    "make_grid",
    "make_template",
    "max_abs_dt",
    "max_abs_x",
    "place_gaussian",
    "seconds_for_beats",
    "spec_for",
    "tau_of_bin",
    "timing_f1",
    "truth_from_bins",
]
