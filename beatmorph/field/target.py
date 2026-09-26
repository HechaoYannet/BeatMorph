"""目标构建：PhigrosChart（IR）-> 桶内计数张量 n_j + 诊断 meta（Plan 03 §3.5 / M2）。

口径（plan 03 §4.2-§4.4）：

- 时间轴是 **tau**（拍，1/48 拍格），i_tau = floor(tau / d_tau)；秒 -> tau 只经
  FieldGrid.seconds_to_tau（**本模块不自行实现任何 BPM 积分**）；
- x 轴是 RPE 舞台系，i_x 只经契约层 x_bin_index（**唯一映射路径**）；
- |positionX| > 675 的 note **从事件项排除并单独计数，不钳位**（红线 3 / §2 偏离 4）；
- isFake 默认排除（假音符无判定、不计物量），include_fake 开关保留，计数必报；
- hold 起点进 hold 通道、终点进 hold_end 通道，**同一 (k, i_x, s) 纤维**（§2 偏离 3）；
  end_time <= start_time 的 Hold 计入 n_illegal_hold（**不修复**——修复属 plan 05）。

本文件不依赖 torch（计数用 numpy int16），因此 M2 的计数守恒契约测试可在最小环境跑。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
from numpy.typing import NDArray

from beatmorph.core.contracts.field import (
    TYPE_CHANNELS,
    ChartFieldSpec,
    ChartTargetField,
    x_bin_index,
)
from beatmorph.core.contracts.phigros import (
    RPE_STAGE_HALF_WIDTH,
    NoteType,
    side_from_above,
    side_index,
)
from beatmorph.core.logging import get_logger
from beatmorph.field.grid import BEAT_SUBDIVISION, FieldGrid, seconds_to_tau, tau_bin_index

if TYPE_CHECKING:
    import torch

    from beatmorph.core.contracts.phigros import PhigrosChart

logger = get_logger("field.target")

#: RPE note type -> 类型通道名（成员必须出现在 contracts.TYPE_CHANNELS 中）
_TYPE_CHANNEL_NAME: Final[dict[NoteType, str]] = {
    NoteType.TAP: "tap",
    NoteType.DRAG: "drag",
    NoteType.HOLD: "hold",
    NoteType.FLICK: "flick",
}

#: 类型通道索引（全部经 TYPE_CHANNELS 求，**不得**写死数字）
CHANNEL_INDEX: Final[dict[NoteType, int]] = {
    note_type: TYPE_CHANNELS.index(name) for note_type, name in _TYPE_CHANNEL_NAME.items()
}
#: hold-end 标记所在的通道索引
HOLD_END_CHANNEL: Final[int] = TYPE_CHANNELS.index("hold_end")


@dataclass(frozen=True, slots=True)
class TargetMeta:
    """目标侧诊断 meta（M2 要求的三项计数 + 分层计数）。

    Attributes:
        n_notes: 计入场的 note 数（已排除 fake / 越界 / 越时间窗）。
        n_events: 事件点数 = sum_j n_j（Hold 起点与终点各算一点）。
        n_fake: 被排除的假音符数（isFake == 1）。
        n_out_of_range: |positionX| > 675 的 note 数（**从事件项排除，不钳位**）。
        n_illegal_hold: end_time <= start_time（hold_time <= 0）的 Hold 数。
        n_out_of_time: 判定时刻落在 tau 轴之外（或 Hold 终点越界）的 note 数。
        out_of_range_x: 越界 note 的 positionX **原值**（不钳位，供审计）。
        per_line / per_side / per_channel: 事件点数的分层计数。
    """

    n_notes: int
    n_events: int
    n_fake: int
    n_out_of_range: int
    n_illegal_hold: int
    n_out_of_time: int
    out_of_range_x: tuple[float, ...]
    per_line: tuple[int, ...]
    per_side: tuple[int, int]
    per_channel: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FieldTarget:
    """目标侧场：桶内计数张量 + 网格规格 + 诊断 meta。

    counts 形状 (K, T, X, S, C)，dtype int16（**桶内计数，不是 0/1 热图**）。
    """

    counts: NDArray[np.int16]
    spec: ChartFieldSpec
    meta: TargetMeta

    @property
    def shape(self) -> tuple[int, ...]:
        """(K, T, X, S, C)。"""
        return tuple(self.counts.shape)

    def total_events(self) -> int:
        """sum_j n_j（与 meta.n_events 必须整数严格相等）。"""
        return int(self.counts.sum())

    def occupied_cells(self) -> int:
        """n_j > 0 的格数。"""
        return int(np.count_nonzero(self.counts))

    def multi_event_cells(self) -> int:
        """n_j >= 2 的格数（共格碰撞的主指标，plan §4.5/§4.8）。"""
        return int(np.count_nonzero(self.counts >= 2))

    def assert_conservation(self) -> None:
        """计数守恒断言：sum_j n_j == meta.n_events（整数严格相等，M2）。"""
        total = self.total_events()
        if total != self.meta.n_events:
            raise AssertionError(f"sum_j n_j = {total} != meta.n_events = {self.meta.n_events}")

    def as_contract(self) -> ChartTargetField:
        """转成契约侧的 ChartTargetField（out_of_window 显式带出）。"""
        return ChartTargetField(counts=self.counts, out_of_window=self.meta.n_out_of_range)

    def to_tensor(self, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        """转 torch 张量（延迟导入 torch，保持本模块可在最小环境导入）。"""
        import torch

        kind = torch.int16 if dtype is None else dtype
        return torch.as_tensor(np.ascontiguousarray(self.counts)).to(kind)


def _in_time_window(index: int, t_bins: int) -> bool:
    return 0 <= index < t_bins


def build_target(
    chart: PhigrosChart,
    grid: FieldGrid,
    *,
    include_fake: bool = False,
) -> FieldTarget:
    """把谱面 IR 网格化为桶内计数张量（K, T, X, S, C）。

    Args:
        chart: 谱面 IR（时间单位是秒；BPMList 是 tau 换算的唯一依据）。
        grid: 已绑定 tau 轴的网格（t_bins > 0 且 bpm_points 非空）。
        include_fake: 是否把 isFake 音符计入事件项（默认 False；计数无论如何都报）。
    """
    if grid.t_bins <= 0 or not grid.bpm_points:
        raise ValueError("build_target 需要已绑定时间轴的网格（t_bins > 0 且 bpm_points 非空）")
    grid.assert_grid()
    spec = grid.spec(len(chart.lines))
    counts = np.zeros(spec.shape(), dtype=np.int16)
    bpm_points = grid.bpm_points

    n_fake = 0
    n_out_of_range = 0
    n_illegal_hold = 0
    n_out_of_time = 0
    n_notes = 0
    out_of_range_x: list[float] = []

    for note in chart.sorted_notes():
        if note.is_fake:
            # 假音符无判定、不计物量：默认排除，但计数**无论如何都报**（plan §4.3）
            n_fake += 1
            if not include_fake:
                continue
        if abs(note.position_x) > RPE_STAGE_HALF_WIDTH:
            # 不钳位：原值留在 meta 中供审计（红线 3 / §2 偏离 4）
            n_out_of_range += 1
            out_of_range_x.append(note.position_x)
            continue
        x_index = x_bin_index(note.position_x, spec)
        if x_index is None:  # pragma: no cover - 由上面的范围判断保证不可达
            raise AssertionError(f"x_bin_index 返回 None：positionX={note.position_x!r}")
        i_tau = tau_bin_index(float(seconds_to_tau(note.t, bpm_points)))
        if not _in_time_window(i_tau, grid.t_bins):
            n_out_of_time += 1
            continue
        line_index = note.line_id
        side = side_index(side_from_above(note.above_raw))
        channel = CHANNEL_INDEX[note.type]
        counts[line_index, i_tau, x_index, side, channel] += 1
        n_notes += 1

        if note.type is NoteType.HOLD:
            if note.hold_time <= 0.0:
                n_illegal_hold += 1
                continue
            end_tau = float(seconds_to_tau(note.t + note.hold_time, bpm_points))
            end_index = tau_bin_index(end_tau)
            if not _in_time_window(end_index, grid.t_bins):
                n_out_of_time += 1
                continue
            # 同一 (k, i_x, s) 纤维的第二个点（§2 偏离 3；RPE 的 Hold 只有单一 positionX）
            counts[line_index, end_index, x_index, side, HOLD_END_CHANNEL] += 1

    n_events = int(counts.sum())
    if n_out_of_time:
        logger.warning(
            "有 %d 个事件点落在 tau 轴之外（T=%d，%d 拍）：加大 tau_end_s 或核查终点口径",
            n_out_of_time,
            grid.t_bins,
            grid.t_bins / BEAT_SUBDIVISION,
        )
    meta = TargetMeta(
        n_notes=n_notes,
        n_events=n_events,
        n_fake=n_fake,
        n_out_of_range=n_out_of_range,
        n_illegal_hold=n_illegal_hold,
        n_out_of_time=n_out_of_time,
        out_of_range_x=tuple(out_of_range_x),
        per_line=tuple(int(value) for value in counts.sum(axis=(1, 2, 3, 4))),
        per_side=(
            int(counts[:, :, :, 0, :].sum()),
            int(counts[:, :, :, 1, :].sum()),
        ),
        per_channel=tuple(int(value) for value in counts.sum(axis=(0, 1, 2, 3))),
    )
    result = FieldTarget(counts=counts, spec=spec, meta=meta)
    result.assert_conservation()
    return result


__all__ = [
    "CHANNEL_INDEX",
    "HOLD_END_CHANNEL",
    "FieldTarget",
    "TargetMeta",
    "build_target",
]
