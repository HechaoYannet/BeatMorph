"""解码事件：场事件 -> （Hold 配对）-> `DecodedEvent` -> 契约 `PhigrosNote`（Plan 05 §3.2）。

时间口径（红线 7 / Plan 00 §3.8 I11）：`DecodedEvent` 的**唯一时间字段是秒**，
τ 只存在于解码器内部的 `FieldEvent`（配对需要它）；秒 -> τ 的全部换算经
`beatmorph/field/grid` 的权威接口，本模块不实现第二套 BPM 积分。

Hold 语义：目标场把 Hold 拆成 **hold 通道（起点）+ hold_end 通道（终点）**，
且两者落在**同一 `(k, i_x, s)` 纤维**（plan 03 §2 偏离 3）。因此解码侧的配对规则
就是该编码的逆：在同一线、同一 x 桶、同一侧的纤维内，把每个起点与**其后最近的**
终点配对（终点在起点之前 = 无法配对，退回 hold_time = 0 并计入 `PairingStats`）。
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from beatmorph.core.contracts.field import ChartFieldSpec
from beatmorph.core.contracts.phigros import (
    NoteType,
    PhigrosNote,
    Side,
    above_from_side,
)
from beatmorph.field.grid import FieldGrid, tau_to_seconds
from beatmorph.field.target import CHANNEL_INDEX, HOLD_END_CHANNEL

#: 通道索引 -> 音符类型（hold_end 通道**不在**其中：它是标记通道，不是音符）。
CHANNEL_NOTE_TYPE: Final[dict[int, NoteType]] = {
    channel: note_type for note_type, channel in CHANNEL_INDEX.items()
}


@dataclass(frozen=True, slots=True)
class FieldEvent:
    """场上的一个**标记点**（decoder 内部中间态，不跨模块）。

    与 `DecodedEvent` 的分工：本类型还带 τ（Hold 配对需要），且**不区分**起点/终点
    （由 `channel` 表达）。构造 `PhigrosNote` 之后即丢弃。
    """

    line_id: int
    tau: float
    x_bin: int
    side: Side
    channel: int
    confidence: float = 0.0


@dataclass(frozen=True, slots=True)
class DecodedEvent:
    """解码出的一个**事件**（Plan 05 §3.2；构造 `PhigrosNote` 后即丢弃）。

    Attributes:
        t_s: 判定时刻（**秒**；公共接口不出现帧索引，Plan 00 §3.8 I11）。
        line_id: 所属判定线（`judgeLineList` 索引）。
        position_x: RPE 舞台系 x 坐标单位。
        side: `side_from_above` 得到的侧别（禁 truthiness 分支）。
        note_type: RPE 音符类型（`type_raw = int(note_type)`）。
        hold_time_s: 仅 HOLD 有意义；**非 HOLD 恒为 0.0**（格式事实：非 Hold 的
            `endTime == startTime`）。HOLD 配不到终点时同样为 0.0（计入配对统计），
            只有被注入的非法事件才可能为负——后处理据此报 `HOLD_REVERSED`。
        is_fake: 生成侧默认 False（`isFake` 是否作为生成目标未统计，plan 05 §9-10）。
        confidence: 供 plan 04 迭代重掩码使用的置信度（Plan 05 §4.1-5）。
    """

    t_s: float
    line_id: int
    position_x: float
    side: Side
    note_type: NoteType
    hold_time_s: float
    is_fake: bool
    confidence: float


@dataclass(frozen=True, slots=True)
class PairingStats:
    """Hold 配对的诊断计数（不丢弃、不改动，只报告）。"""

    n_starts: int
    n_ends: int
    n_paired: int
    #: 配不到终点的 Hold 起点：`hold_time = 0`，**保留不丢弃**（含痕在报告里）
    n_unpaired_starts: int
    #: 配不到起点的 Hold 终点：不产生音符，**丢弃**并计数
    n_orphan_ends: int
    #: 配对成功但 `end <= start`（零时长 Hold；格式允许 `endTime == startTime`）
    n_zero_length: int

    @property
    def as_stats(self) -> dict[str, float]:
        """转成报告统计字典（键名以 `hold_` 前缀分组）。"""
        return {
            "hold_starts": float(self.n_starts),
            "hold_ends": float(self.n_ends),
            "hold_paired": float(self.n_paired),
            "hold_unpaired_starts": float(self.n_unpaired_starts),
            "hold_orphan_ends": float(self.n_orphan_ends),
            "hold_zero_length": float(self.n_zero_length),
        }


def note_type_for_channel(channel: int) -> NoteType:
    """通道 -> 音符类型（hold_end 通道不属于音符，调用前必须已排除）。"""
    try:
        return CHANNEL_NOTE_TYPE[int(channel)]
    except KeyError as exc:
        raise ValueError(
            f"通道 {channel!r} 不是音符通道（合法值：{sorted(CHANNEL_NOTE_TYPE)}；"
            f"hold_end 通道 {HOLD_END_CHANNEL} 是标记通道）",
        ) from exc


def x_center(x_bin: int, spec: ChartFieldSpec) -> float:
    """x 桶 -> 桶中心坐标（**唯一的**反查路径；禁止各模块自行 floor / +0.5）。"""
    if not 0 <= x_bin < spec.x_bins:
        raise IndexError(f"x_bin={x_bin} 越界（X={spec.x_bins}）")
    return spec.x_min + (x_bin + 0.5) * spec.dx


def pair_events(
    events: Sequence[FieldEvent],
    grid: FieldGrid,
    *,
    spec: ChartFieldSpec | None = None,
) -> tuple[list[DecodedEvent], PairingStats]:
    """场标记点 -> `DecodedEvent`（含 Hold 起点/终点配对），输出**确定性排序**。

    排序键 `(t_s, line_id, position_x, side, note_type)`：与
    `PhigrosChart.sorted_notes()` 同口径，保证同 seed 两次解码逐字段一致。
    """
    if grid.t_bins <= 0 or not grid.bpm_points:
        raise ValueError("pair_events 需要已绑定时间轴的网格（t_bins > 0 且 bpm_points 非空）")
    bpm_points = grid.bpm_points
    if spec is None:
        k = max((event.line_id for event in events), default=0) + 1
        spec = grid.spec(k)

    ordered = sorted(
        events,
        key=lambda event: (event.line_id, event.tau, event.x_bin, int(event.side), event.channel),
    )
    ends: dict[tuple[int, int, int], list[float]] = {}
    starts: list[FieldEvent] = []
    for event in ordered:
        if event.channel == HOLD_END_CHANNEL:
            ends.setdefault((event.line_id, event.x_bin, int(event.side)), []).append(event.tau)
        else:
            starts.append(event)
    used: dict[tuple[int, int, int], list[bool]] = {
        key: [False] * len(values) for key, values in ends.items()
    }

    decoded: list[DecodedEvent] = []
    n_paired = 0
    n_unpaired = 0
    n_zero_length = 0
    for event in starts:
        note_type = note_type_for_channel(event.channel)
        hold_time = 0.0
        if note_type is NoteType.HOLD:
            key = (event.line_id, event.x_bin, int(event.side))
            fiber = ends.get(key, [])
            index = bisect_left(fiber, event.tau)
            while index < len(fiber) and used[key][index]:
                index += 1
            if index < len(fiber):
                used[key][index] = True
                n_paired += 1
                hold_time = float(
                    tau_to_seconds(fiber[index], bpm_points)
                    - tau_to_seconds(event.tau, bpm_points),
                )
                if hold_time <= 0.0:
                    n_zero_length += 1
            else:
                n_unpaired += 1
        decoded.append(
            DecodedEvent(
                t_s=float(tau_to_seconds(event.tau, bpm_points)),
                line_id=event.line_id,
                position_x=x_center(event.x_bin, spec),
                side=event.side,
                note_type=note_type,
                hold_time_s=hold_time,
                is_fake=False,
                confidence=float(event.confidence),
            ),
        )

    n_orphan = sum(1 for key, flags in used.items() for flag in flags if not flag)
    decoded.sort(key=event_sort_key)
    stats = PairingStats(
        n_starts=len(starts),
        n_ends=sum(len(values) for values in ends.values()),
        n_paired=n_paired,
        n_unpaired_starts=n_unpaired,
        n_orphan_ends=n_orphan,
        n_zero_length=n_zero_length,
    )
    return decoded, stats


def event_sort_key(event: DecodedEvent) -> tuple[float, int, float, int, int]:
    """`DecodedEvent` 的确定性排序键（与 `sorted_notes` 同口径 + 侧别/类型消歧）。"""
    return (
        event.t_s,
        event.line_id,
        event.position_x,
        int(event.side),
        int(event.note_type),
    )


def events_to_notes(events: Sequence[DecodedEvent]) -> list[PhigrosNote]:
    """`DecodedEvent` -> 契约 `PhigrosNote`（`*_raw` 双写口径）。

    映射全部经契约函数：`type_raw = int(note_type)`（RPE 枚举即 RPE 数字）、
    `above_raw = above_from_side(side)`（背面规范代表值 0）。

    Raises:
        ValueError: `hold_time_s < 0`（反向 Hold；应先经后处理检出并留痕，见
            `beatmorph.decoder.postprocess.legality`）。
    """
    notes: list[PhigrosNote] = []
    for event in events:
        if event.hold_time_s < 0.0:
            raise ValueError(
                f"反向 Hold（hold_time_s={event.hold_time_s}）不得进入契约层："
                "IR 的 hold_time 有 ge=0 约束，必须先由后处理按红线检出并留痕",
            )
        notes.append(
            PhigrosNote(
                line_id=event.line_id,
                t=float(event.t_s),
                position_x=float(event.position_x),
                side=event.side,
                type=event.note_type,
                hold_time=float(event.hold_time_s),
                is_fake=event.is_fake,
                above_raw=above_from_side(event.side),
                type_raw=int(event.note_type),
                is_fake_raw=1 if event.is_fake else 0,
            ),
        )
    return notes


def confidence_array(events: Sequence[DecodedEvent]) -> np.ndarray:
    """置信度数组（与 `events` 同序；供 plan 04 迭代重掩码与评估对齐）。"""
    return np.asarray([event.confidence for event in events], dtype=np.float64)


__all__ = [
    "CHANNEL_NOTE_TYPE",
    "DecodedEvent",
    "FieldEvent",
    "PairingStats",
    "confidence_array",
    "event_sort_key",
    "events_to_notes",
    "note_type_for_channel",
    "pair_events",
    "x_center",
]
