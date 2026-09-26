"""分解报告：时间组（含 12/24 三连）与难度分档（plan 06 §4.4）。

**时间组**（plan §4.4-1）：1/4、1/8、1/12（三连）、1/16、1/24（三连）、1/32。
三连组不可省（GOCT 的分解法，文献库 §7.2-5）。

**归组规则（显式声明，且必须与解码网格无关）**：

- 位置口径（默认，plan §4.4-1 的「最近 beat 细分」）：事件在拍域的**位置** b 对每组
  细分 n 的「到最近格点距离」= min(frac(b*n), 1-frac(b*n)) / n（单位：拍），取距离
  最小的组；并列时取**最粗**的组——因此恰好落在整拍 / 各细分格点上的事件归入它能
  命中的最粗组（整拍事件 → 1/4，即 M6.4 的 δ = 0 用例）。
- 间隔口径（可选）：与**同线前一事件**的拍间隔 Δb 取最近细分。间隔定义 =
  「同线 + 按规范序的紧邻前驱」，**跨线间隔不参与**（不同判定线的事件序列是各自的
  时值流）；每条线的首个事件退化为位置口径（没有前驱）。
- 两个口径都只用 beatmorph.field.grid.seconds_to_tau 把秒换成拍（红线 7：秒 <-> 拍
  换算只在 field/ 内实现），**不引用 TAU_GRID_DT / SUBDIVISIONS_PER_BEAT**，因此
  「1/32 拍」这类**不在 1/48 解码网格上**的位置照样能被正确归组——plan §4.4-1 要求
  「归组规则与解码网格无关（否则分解会自我实现）」，测试对该点有显式断言。

**难度**（plan §4.4-2）：difficulty 是 f32，**必须先 round 到 0.1** 再比较（实测存在
14.900001 / 18.000004 这类浮点误差）；level 是自由文本，**绝不解析**（实测出现
"AT  Lv.16" / "sweet" / "酔い"）。低难度必须单独成档（文献一致报告低难度是难点）。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import StrEnum
from typing import Final

from beatmorph.core.contracts.phigros import BpmPoint
from beatmorph.decoder.events import DecodedEvent, event_sort_key
from beatmorph.eval.protocol import (
    DIFFICULTY_ROUND_DIGITS,
    UNKNOWN_DIFFICULTY_BAND,
    TimeGroupRule,
)
from beatmorph.field.grid import seconds_to_tau


class TimeGroup(StrEnum):
    """时间组枚举（值即报告里的键名，如 "1/12"）。"""

    QUARTER = "1/4"
    EIGHTH = "1/8"
    TWELFTH = "1/12"
    SIXTEENTH = "1/16"
    TWENTY_FOURTH = "1/24"
    THIRTY_SECOND = "1/32"


#: 每拍细分数（**协议常量**：GOCT 的分解类，plan §4.4-1 列举的六档）。
TIME_GROUP_SUBDIVISIONS: Final[dict[TimeGroup, int]] = {
    TimeGroup.QUARTER: 4,
    TimeGroup.EIGHTH: 8,
    TimeGroup.TWELFTH: 12,
    TimeGroup.SIXTEENTH: 16,
    TimeGroup.TWENTY_FOURTH: 24,
    TimeGroup.THIRTY_SECOND: 32,
}

#: 组的遍历顺序：**粗 → 细**（并列时的优先级即由此而来）。
TIME_GROUPS: Final[tuple[TimeGroup, ...]] = tuple(TIME_GROUP_SUBDIVISIONS)

#: 浮点护栏：仅用于抵消 b*n 的 1~2 ulp 误差（1e-9 个细分格，远小于任何真实量化）。
#: 它不是物理常量，也不是分辨率。
PHASE_TIE_EPS: Final[float] = 1e-9


def _phase_distance(beats: float, per_beat: int) -> float:
    """「到最近细分格点的距离」（拍）：min(frac, 1-frac)/n，frac = frac(b*n)。"""
    frac = (float(beats) * per_beat) % 1.0
    near_tie = frac < PHASE_TIE_EPS or frac > 1.0 - PHASE_TIE_EPS
    return (0.0 if near_tie else min(frac, 1.0 - frac)) / per_beat


def _nearest_group(value: float) -> TimeGroup:
    """取「到最近格点距离」最小的组；并列取最粗（TIME_GROUPS 的先后顺序）。"""
    best = TIME_GROUPS[0]
    best_distance = _phase_distance(value, TIME_GROUP_SUBDIVISIONS[best])
    for group in TIME_GROUPS[1:]:
        distance = _phase_distance(value, TIME_GROUP_SUBDIVISIONS[group])
        if distance < best_distance:
            best = group
            best_distance = distance
    return best


def classify_beat_position(beats: float) -> TimeGroup:
    """拍位置 -> 时间组（位置口径）。"""
    return _nearest_group(float(beats))


def classify_interval(interval_beats: float) -> TimeGroup:
    """拍间隔 -> 时间组（间隔口径；取绝对值，间隔为 0 时归入最粗组）。"""
    return _nearest_group(abs(float(interval_beats)))


def event_beats(
    events: Sequence[DecodedEvent], bpm_points: Sequence[BpmPoint]
) -> tuple[float, ...]:
    """事件拍坐标（秒 -> 拍**经 field/ 的权威换算**；红线 7，评估内不重写 BPM 积分）。"""
    return tuple(float(seconds_to_tau(float(event.t_s), bpm_points)) for event in events)


def assign_time_groups(
    events: Sequence[DecodedEvent],
    bpm_points: Sequence[BpmPoint],
    *,
    rule: TimeGroupRule = TimeGroupRule.POSITION,
) -> tuple[TimeGroup, ...]:
    """给每个事件分配时间组（与输入顺序无关；规则见模块 docstring）。

    返回序列与 events **同序同长**（下标一一对应）。
    """
    if not events:
        return ()
    beats = event_beats(events, bpm_points)
    if rule is TimeGroupRule.POSITION:
        return tuple(classify_beat_position(value) for value in beats)
    order = sorted(range(len(events)), key=lambda index: event_sort_key(events[index]))
    assigned: dict[int, TimeGroup] = {}
    previous: dict[int, float] = {}
    for index in order:
        line_id = int(events[index].line_id)
        if line_id in previous:
            assigned[index] = classify_interval(beats[index] - previous[line_id])
        else:
            assigned[index] = classify_beat_position(beats[index])
        previous[line_id] = beats[index]
    return tuple(assigned[index] for index in range(len(events)))


def difficulty_band(difficulty: float | None) -> str:
    """难度档名：round 到 0.1 后格式化；缺省 / 非有限值 -> "unknown"。

    **level 文本不参与本函数**（plan §4.4-2：自由文本绝不解析），因此调用方无法
    用 "AT  Lv.16" 之类的字符串影响分档——测试对此有断言。
    """
    if difficulty is None:
        return UNKNOWN_DIFFICULTY_BAND
    value = float(difficulty)
    if not math.isfinite(value):
        return UNKNOWN_DIFFICULTY_BAND
    return f"{round(value, DIFFICULTY_ROUND_DIGITS):.{DIFFICULTY_ROUND_DIGITS}f}"


__all__ = [
    "PHASE_TIE_EPS",
    "TIME_GROUPS",
    "TIME_GROUP_SUBDIVISIONS",
    "TimeGroup",
    "assign_time_groups",
    "classify_beat_position",
    "classify_interval",
    "difficulty_band",
    "event_beats",
]
