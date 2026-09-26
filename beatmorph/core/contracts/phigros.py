"""Phigros 谱面领域契约 —— 判定线 / 音符 / 谱面（Plan 00 M1）。

本模块是 **跨模块物理量的唯一定义处**（CLAUDE.md 红线 7）：RPE 舞台几何、
x 网格、速度单位、事件层数、时间基本格全部写成**派生式**，只有
RPE_STAGE_WIDTH / RPE_STAGE_HEIGHT / RPE_HEIGHT_RATIO / SUBDIVISIONS_PER_BEAT
四个**来源常量**允许是字面量。

时间量纲（CLAUDE.md 红线 7 / RFC-0029 §7-5 / §7-8）：

- 契约层 PhigrosNote.t 与 PhigrosChart.meta 一律用**秒**；
- 事件轨的求值发生在**拍（beat）域**（RPE 的 beat 三元组原生单位）；
- **秒 ↔ 拍（= τ 场网格）的换算只允许在 beatmorph/field/ 内实现**，
  J(τ) = dt/dτ 由谱面 bpm_points 派生。因此本模块的 JudgeLine.pose_at /
  JudgeLine.local_to_stage 接受**拍**而不是秒：契约层不实现任何 BPM 分段积分
  （否则 beat-aligned 的数学会外溢到所有下游）。

对应文档：docs/plans/00-core-contracts.md §3.1-§3.6、docs/BasePlan.md §3.2、
docs/knowledges/phigros-format.md（字段级事实，含来源分级）。
"""

from __future__ import annotations

import math
from collections.abc import Callable
from enum import IntEnum, StrEnum
from typing import TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: 缓动函数签名：单位区间 [0,1] 到实数。
EasingFn: TypeAlias = Callable[[float], float]

# ══════════════════════════════════════════════════════════════
# §3.1 常量（全部派生式；本表是这些物理量的唯一出处）
# ══════════════════════════════════════════════════════════════

# ── RPE 舞台几何 ──
# 来源常量（A 级：prpr parse/rpe.rs RPE_WIDTH / RPE_HEIGHT）。只允许在本文件出现一次。
RPE_STAGE_WIDTH: float = 1350.0
RPE_STAGE_HEIGHT: float = 900.0
RPE_STAGE_HALF_WIDTH: float = RPE_STAGE_WIDTH / 2.0
RPE_STAGE_HALF_HEIGHT: float = RPE_STAGE_HEIGHT / 2.0

# ── x 网格（RFC-0029 §3.1 裁定 dx = RPE_STAGE_WIDTH / N，默认 N = 128）──
RPE_X_GRID_BINS: int = 128
RPE_X_GRID_DX: float = RPE_STAGE_WIDTH / RPE_X_GRID_BINS
RPE_X_GRID_MIN: float = -RPE_STAGE_HALF_WIDTH
RPE_X_GRID_MAX: float = +RPE_STAGE_HALF_WIDTH
#: 必做消融（红线 7：128 只是默认值，不是结论；最终取值待 plan 02 的共格碰撞统计）
RPE_X_GRID_BIN_SWEEP: tuple[int, ...] = (64, 128, 256, 512)

# ── 速度单位（A 级：prpr core.rs SPEED_RATIO = 10/45/HEIGHT_RATIO；出处存疑 D3）──
RPE_HEIGHT_RATIO: float = 0.83175
RPE_SPEED_RATIO: float = 10.0 / 45.0 / RPE_HEIGHT_RATIO
RPE_SPEED_UNIT_RPE_Y_PER_SEC: float = RPE_STAGE_HALF_HEIGHT * RPE_SPEED_RATIO

# ── 事件层 ──
RPE_MAX_EVENT_LAYERS: int = 5  # 实测 eventLayers 长度 1..5
RPE_NORMAL_EVENT_LAYERS: int = 4  # RFC-0029 §8.1 文档口径（非归一化依据）
RPE_NORMAL_TRACKS: tuple[str, ...] = (
    "moveXEvents",
    "moveYEvents",
    "rotateEvents",
    "alphaEvents",
    "speedEvents",
)
#: 普通轨在 IR 中的字段名（RPE 原始名 moveXEvents → IR 名 move_x，映射属 plan 02）。
RPE_TRACK_FIELDS: tuple[str, ...] = ("move_x", "move_y", "rotate", "alpha", "speed")
RPE_EXTENDED_TRACKS: tuple[str, ...] = (
    "scaleXEvents",
    "scaleYEvents",
    "colorEvents",
    "textEvents",
    "gifEvents",
    "inclineEvents",
)

# ── 时间基本格（Q15 决议：beat-aligned，1/48 拍；来源常量，禁止写字面量小数）──
SUBDIVISIONS_PER_BEAT: int = 48
TAU_GRID_DT: float = 1.0 / SUBDIVISIONS_PER_BEAT

#: 事件轨补洞的「足够大的时间」（拍）。官方参考实现用 Beat(31250000, 0, 1)。
RPE_EVENT_TAIL_BEATS: float = 31250000.0


class GameMode(IntEnum):
    """目标游戏模式。**单一目标 Phigros**（CLAUDE.md 红线 4）。"""

    PHIGROS = 1


class ChartFormat(StrEnum):
    """谱面文件格式（**只按内容嗅探**判定，不看后缀、不看 info.yml.format）。

    ChartFormat 是跨模块类型（data 产出 → qc / decoder 消费），因此定义在契约层；
    beatmorph/data/parsers/sniff.py 只做**再导出**，不另立一份枚举——「同一常量
    两处定义」正是红线 7 要挡的那类漂移。
    """

    RPE = "rpe"
    PEC = "pec"
    OFFICIAL = "official"
    PBC = "pbc"
    UNKNOWN = "unknown"


class Side(IntEnum):
    """判定线哪一侧下落。**不是布尔**。

    above == 1 → 正面；**其余任何值**（实测出现 0 与 2）→ 背面。
    两侧几何上是关于判定线局部 X 轴的 Y 镜像（A 级：prpr line.rs）。
    两个成员都是**真值**，因此任何 if side: 写法都必然失效——这是用类型设计
    挡住静默错位的约束（plan 00 偏离 3）。
    """

    FRONT = +1
    BACK = -1


class NoteType(IntEnum):
    """RPEJSON note type（A 级源码确证：1/2/3/4 = Click/Hold/Flick/Drag）。"""

    TAP = 1
    HOLD = 2
    FLICK = 3
    DRAG = 4


def side_from_above(above: int) -> Side:
    """above == 1 → FRONT，其余值 → BACK。

    这是 above 语义的**唯一实现路径**；禁止任何模块自行写 bool(above) 或
    above != 0（0 与 2 都是背面，plan 00 I4）。
    """
    return Side.FRONT if above == 1 else Side.BACK


def side_index(side: Side) -> int:
    """场通道索引：FRONT → 0，BACK → 1。

    索引与枚举值**不同**（枚举值是 ±1），必须用本函数。
    """
    return 0 if side is Side.FRONT else 1


def note_type_from_rpe(raw: int) -> NoteType:
    """RPEJSON 分派：1/2/3/4 = Tap/Hold/Flick/Drag。非法值抛 ValueError。"""
    try:
        return NoteType(raw)
    except ValueError as exc:
        raise ValueError(f"RPE note type 必须属于 (1,2,3,4)，得到 {raw!r}") from exc


#: 官谱 JSON 分派（C 级来源；v1 主路径只做嗅探识别与记账，不解析）。
_OFFICIAL_TYPE_MAP: dict[int, NoteType] = {
    1: NoteType.TAP,
    2: NoteType.DRAG,
    3: NoteType.HOLD,
    4: NoteType.FLICK,
}


def note_type_from_official(raw: int) -> NoteType:
    """官谱 JSON 分派：同一数字语义与 RPE **不同**（2=Drag / 3=Hold / 4=Flick）。"""
    try:
        return _OFFICIAL_TYPE_MAP[raw]
    except KeyError as exc:
        raise ValueError(f"官谱 note type 必须属于 (1,2,3,4)，得到 {raw!r}") from exc


# ══════════════════════════════════════════════════════════════
# §3.3 事件与层
# ══════════════════════════════════════════════════════════════


class Beat(BaseModel):
    """RPE beat 三元组：beats = i + n / d（A 级：prpr Triple(i32,u32,u32)）。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    i: int = 0
    n: int = 0
    d: int = Field(default=1, gt=0)

    @model_validator(mode="before")
    @classmethod
    def _accept_sequence(cls, value: object) -> object:
        """接受 JSON 的 [i, n, d] 数组写法。"""
        if isinstance(value, list | tuple):
            if len(value) != 3:
                raise ValueError(f"beat 三元组长度必须为 3，得到 {value!r}")
            return {"i": value[0], "n": value[1], "d": value[2]}
        return value

    def to_beats(self) -> float:
        """转浮点拍数（i + n / d）。"""
        return self.i + self.n / self.d


#: 事件数值：数值 / 文本（textEvents）/ RGB（colorEvents）。
EventValue = float | str | tuple[int, int, int]


def _linear(t: float) -> float:
    return t


def _out_sine(t: float) -> float:
    return math.sin((t * math.pi) / 2.0)


def _in_sine(t: float) -> float:
    return 1.0 - math.cos((t * math.pi) / 2.0)


def _out_quad(t: float) -> float:
    return 1.0 - (1.0 - t) * (1.0 - t)


def _in_quad(t: float) -> float:
    return t * t


def _in_out_sine(t: float) -> float:
    return -(math.cos(math.pi * t) - 1.0) / 2.0


def _in_out_quad(t: float) -> float:
    if t < 0.5:
        return 2.0 * t * t
    return 1.0 - ((-2.0 * t + 2.0) ** 2) / 2.0


def _out_cubic(t: float) -> float:
    return 1.0 - (1.0 - t) ** 3


def _in_cubic(t: float) -> float:
    return t**3


def _out_quart(t: float) -> float:
    return 1.0 - (1.0 - t) ** 4


def _in_quart(t: float) -> float:
    return t**4


def _in_out_cubic(t: float) -> float:
    if t < 0.5:
        return 4.0 * t**3
    return 1.0 - ((-2.0 * t + 2.0) ** 3) / 2.0


def _in_out_quart(t: float) -> float:
    if t < 0.5:
        return 8.0 * t**4
    return 1.0 - ((-2.0 * t + 2.0) ** 4) / 2.0


def _out_quint(t: float) -> float:
    return 1.0 - (1.0 - t) ** 5


def _in_quint(t: float) -> float:
    return t**5


def _out_expo(t: float) -> float:
    if t >= 1.0:
        return 1.0
    return 1.0 - math.pow(2.0, -10.0 * t)


def _in_expo(t: float) -> float:
    if t <= 0.0:
        return 0.0
    return math.pow(2.0, 10.0 * t - 10.0)


def _out_circ(t: float) -> float:
    return math.sqrt(max(0.0, 1.0 - (t - 1.0) ** 2))


def _in_circ(t: float) -> float:
    return 1.0 - math.sqrt(max(0.0, 1.0 - t * t))


def _out_back(t: float) -> float:
    c1 = 1.70158
    c3 = c1 + 1.0
    return 1.0 + c3 * (t - 1.0) ** 3 + c1 * (t - 1.0) ** 2


def _in_back(t: float) -> float:
    c1 = 1.70158
    c3 = c1 + 1.0
    return c3 * t**3 - c1 * t * t


def _in_out_circ(t: float) -> float:
    if t < 0.5:
        return (1.0 - math.sqrt(max(0.0, 1.0 - (2.0 * t) ** 2))) / 2.0
    return (math.sqrt(max(0.0, 1.0 - (-2.0 * t + 2.0) ** 2)) + 1.0) / 2.0


def _in_out_back(t: float) -> float:
    c1 = 1.70158
    c2 = c1 * 1.525
    if t < 0.5:
        return ((2.0 * t) ** 2 * ((c2 + 1.0) * 2.0 * t - c2)) / 2.0
    return ((2.0 * t - 2.0) ** 2 * ((c2 + 1.0) * (t * 2.0 - 2.0) + c2) + 2.0) / 2.0


def _out_elastic(t: float) -> float:
    c4 = (2.0 * math.pi) / 3.0
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return math.pow(2.0, -10.0 * t) * math.sin((t * 10.0 - 0.75) * c4) + 1.0


def _in_elastic(t: float) -> float:
    c4 = (2.0 * math.pi) / 3.0
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return -math.pow(2.0, 10.0 * t - 10.0) * math.sin((t * 10.0 - 10.75) * c4)


def _out_bounce(t: float) -> float:
    n1 = 7.5625
    d1 = 2.75
    if t < 1.0 / d1:
        return n1 * t * t
    if t < 2.0 / d1:
        t -= 1.5 / d1
        return n1 * t * t + 0.75
    if t < 2.5 / d1:
        t -= 2.25 / d1
        return n1 * t * t + 0.9375
    t -= 2.625 / d1
    return n1 * t * t + 0.984375


def _in_bounce(t: float) -> float:
    return 1.0 - _out_bounce(1.0 - t)


def _in_out_bounce(t: float) -> float:
    if t < 0.5:
        return (1.0 - _out_bounce(1.0 - 2.0 * t)) / 2.0
    return (1.0 + _out_bounce(2.0 * t - 1.0)) / 2.0


def _in_out_elastic(t: float) -> float:
    c5 = (2.0 * math.pi) / 4.5
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    if t < 0.5:
        return -(math.pow(2.0, 20.0 * t - 10.0) * math.sin((20.0 * t - 11.125) * c5)) / 2.0
    return (math.pow(2.0, -20.0 * t + 10.0) * math.sin((20.0 * t - 11.125) * c5)) / 2.0 + 1.0


#: easingType 1..29 对照表（A 级：Phira Docs extend；名称与顺序逐条对齐）。
EASING_FUNCS: tuple[EasingFn, ...] = (
    _linear,  # 1  Linear
    _out_sine,  # 2  Out Sine
    _in_sine,  # 3  In Sine
    _out_quad,  # 4  Out Quad
    _in_quad,  # 5  In Quad
    _in_out_sine,  # 6  In Out Sine
    _in_out_quad,  # 7  In Out Quad
    _out_cubic,  # 8  Out Cubic
    _in_cubic,  # 9  In Cubic
    _out_quart,  # 10 Out Quart
    _in_quart,  # 11 In Quart
    _in_out_cubic,  # 12 In Out Cubic
    _in_out_quart,  # 13 In Out Quart
    _out_quint,  # 14 Out Quint
    _in_quint,  # 15 In Quint
    _out_expo,  # 16 Out Expo
    _in_expo,  # 17 In Expo
    _out_circ,  # 18 Out Circ
    _in_circ,  # 19 In Circ
    _out_back,  # 20 Out Back
    _in_back,  # 21 In Back
    _in_out_circ,  # 22 In Out Circ
    _in_out_back,  # 23 In Out Back
    _out_elastic,  # 24 Out Elastic
    _in_elastic,  # 25 In Elastic
    _out_bounce,  # 26 Out Bounce
    _in_bounce,  # 27 In Bounce
    _in_out_bounce,  # 28 In Out Bounce
    _in_out_elastic,  # 29 In Out Elastic
)


def normalize_easing_type(raw: object) -> int:
    """缓动类型归一化：非整数 → 1；< 1 → 1；> 29 → 29（末端值）。"""
    if isinstance(raw, bool) or not isinstance(raw, int):
        return 1
    return 1 if raw < 1 else (len(EASING_FUNCS) if raw > len(EASING_FUNCS) else raw)


def _bezier_ease(progress: float, points: tuple[float, float, float, float]) -> float:
    """三次贝塞尔缓动（起止点固定 (0,0) → (1,1)）。

    以二分法解 x(u) = progress 再取 y(u)：确定性、无外部依赖，64 次迭代
    远优于所需的 1e-9 精度。
    """
    x1, y1, x2, y2 = (min(1.0, max(0.0, p)) for p in points)
    if progress <= 0.0:
        return 0.0
    if progress >= 1.0:
        return 1.0

    def bezier(u: float, c1: float, c2: float) -> float:
        return 3.0 * c1 * (1.0 - u) ** 2 * u + 3.0 * c2 * (1.0 - u) * u * u + u**3

    lo, hi = 0.0, 1.0
    for _ in range(64):
        mid = (lo + hi) / 2.0
        if bezier(mid, x1, x2) < progress:
            lo = mid
        else:
            hi = mid
    return bezier((lo + hi) / 2.0, y1, y2)


def _easing_progress(
    t: float,
    easing_type: int,
    bezier: bool,
    bezier_points: tuple[float, float, float, float],
    left: float,
    right: float,
) -> float:
    """把事件内进度 t ∈ [0,1] 映射为缓动输出（0 取起点值，1 取终点值）。

    easingLeft/Right 切割的**归一化形式**（plan 00 §3.4 的「切割规则」）：

        g(t) = (f(l + t*(r-l)) - f(l)) / (f(r) - f(l))

    该形式由知识文档同时给出的三条工作实例唯一确定（原文公式在转录中丢失了
    归一化分母，直接照抄会使事件终点取不到 end 值）：① Linear 切割不起作用；
    ② Out Quad 改变左边界不起作用；③ In Quad 改变右边界不起作用——三条只有
    归一化形式同时成立。
    """
    lo = min(max(left, 0.0), 1.0)
    hi = min(max(right, 0.0), 1.0)
    if hi <= lo:
        lo, hi = 0.0, 1.0
    f: EasingFn
    if bezier:

        def _bezier(u: float) -> float:
            return _bezier_ease(u, bezier_points)

        f = _bezier
    else:
        f = EASING_FUNCS[easing_type - 1]
    if (lo, hi) == (0.0, 1.0):
        return float(f(t))
    f_lo, f_hi = float(f(lo)), float(f(hi))
    span = f_hi - f_lo
    if span == 0.0:
        return f_lo
    return (float(f(lo + t * (hi - lo))) - f_lo) / span


class EventKeyframe(BaseModel):
    """一个事件关键帧（普通轨与特殊轨共用；speedEvents 无贝塞尔字段）。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    start_time: Beat
    end_time: Beat
    start: EventValue
    end: EventValue
    easing_type: int = 1
    bezier: bool = False
    bezier_points: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    easing_left: float = 0.0
    easing_right: float = 1.0
    link_group: int = 0

    @field_validator("easing_type", mode="before")
    @classmethod
    def _norm_easing(cls, value: object) -> int:
        return normalize_easing_type(value)

    def value_at(self, t_beats: float) -> EventValue:
        """本关键帧在 t_beats（拍）处的取值（时间范围外取端点值）。"""
        st = self.start_time.to_beats()
        et = self.end_time.to_beats()
        if t_beats <= st or et <= st:
            return self.start
        if t_beats >= et:
            return self.end
        progress = (t_beats - st) / (et - st)
        eased = _easing_progress(
            progress,
            self.easing_type,
            self.bezier,
            self.bezier_points,
            self.easing_left,
            self.easing_right,
        )
        if isinstance(self.start, str) or isinstance(self.end, str):
            return self.start
        if isinstance(self.start, tuple) and isinstance(self.end, tuple):
            return (
                round(self.start[0] + (self.end[0] - self.start[0]) * eased),
                round(self.start[1] + (self.end[1] - self.start[1]) * eased),
                round(self.start[2] + (self.end[2] - self.start[2]) * eased),
            )
        if isinstance(self.start, tuple) or isinstance(self.end, tuple):
            raise TypeError("事件起止值类型不一致（数值 vs RGB）")
        return self.start + (self.end - self.start) * eased

    def numeric_at(self, t_beats: float, default: float = 0.0) -> float:
        """取数值型取值；非数值（文本 / RGB）返回 default。"""
        value = self.value_at(t_beats)
        return float(value) if isinstance(value, int | float) else default


def track_value(keyframes: list[EventKeyframe], t_beats: float, default: float = 0.0) -> float:
    """在一条轨上取值（命中即返回，未命中取默认值；补洞后必然命中）。"""
    for kf in keyframes:
        if kf.start_time.to_beats() <= t_beats <= kf.end_time.to_beats():
            return kf.numeric_at(t_beats, default)
    return default


def fill_gaps(keyframes: list[EventKeyframe]) -> list[EventKeyframe]:
    """按 startTime 排序并**补洞**（官方 _init_events 语义）。

    相邻事件之间存在时间空隙时，插入一个「保持前一事件终值」的常量事件；
    末尾追加一个足够长的常量事件。**不补洞则间隙内取值无定义。**
    """
    if not keyframes:
        return []
    ordered = sorted(keyframes, key=lambda kf: kf.start_time.to_beats())
    out: list[EventKeyframe] = [ordered[0]]
    for kf in ordered[1:]:
        prev = out[-1]
        if kf.start_time.to_beats() > prev.end_time.to_beats():
            out.append(
                EventKeyframe(
                    start_time=prev.end_time,
                    end_time=kf.start_time,
                    start=prev.end,
                    end=prev.end,
                ),
            )
        out.append(kf)
    tail = out[-1]
    if tail.end_time.to_beats() < RPE_EVENT_TAIL_BEATS:
        out.append(
            EventKeyframe(
                start_time=tail.end_time,
                end_time=Beat(i=int(RPE_EVENT_TAIL_BEATS)),
                start=tail.end,
                end=tail.end,
            ),
        )
    return out


class EventLayer(BaseModel):
    """一个普通事件层级：5 条轨，跨层**求和**。

    三态归一（null 层 / 字段缺失 / 缺整段 eventLayers）统一产出「空轨列表」
    —— 本类的默认值即该归一化结果。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    layer_index: int = 0
    move_x: list[EventKeyframe] = Field(default_factory=list)
    move_y: list[EventKeyframe] = Field(default_factory=list)
    rotate: list[EventKeyframe] = Field(default_factory=list)
    alpha: list[EventKeyframe] = Field(default_factory=list)
    speed: list[EventKeyframe] = Field(default_factory=list)

    @field_validator("move_x", "move_y", "rotate", "alpha", "speed", mode="after")
    @classmethod
    def _normalize_track(cls, value: list[EventKeyframe]) -> list[EventKeyframe]:
        """补洞 + 排序（保证 IR 内取值处处有定义）。

        field_validator(mode="after") 允许返回不同的值，因此这里可以直接用
        fill_gaps 的返回值；用 model_validator 返回新实例是 pydantic 不支持的写法。
        """
        return fill_gaps(value)


class ExtendedLayer(BaseModel):
    """第 5 层（extended 字段），独立于 event_layers。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scale_x: list[EventKeyframe] = Field(default_factory=list)
    scale_y: list[EventKeyframe] = Field(default_factory=list)
    color: list[EventKeyframe] = Field(default_factory=list)
    text: list[EventKeyframe] = Field(default_factory=list)
    gif: list[EventKeyframe] = Field(default_factory=list)
    incline: list[EventKeyframe] = Field(default_factory=list)


class LinePose(BaseModel):
    """判定线在拍 t 处的舞台系位姿（**跨层求和 + 父线递归**后的结果）。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    x: float
    y: float
    rotate_deg: float
    alpha: float
    scale_x: float = 1.0
    scale_y: float = 1.0


class Transform(BaseModel):
    """判定线局部系到舞台系的刚体变换（绕锚点旋转 + 平移，顺时针为正）。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    translate_x: float
    translate_y: float
    rotate_deg: float

    def apply(self, local_x: float, local_y: float) -> tuple[float, float]:
        """把判定线局部系坐标映射到舞台系坐标。"""
        rad = math.radians(self.rotate_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        return (
            self.translate_x + local_x * cos_a - local_y * sin_a,
            self.translate_y + local_x * sin_a + local_y * cos_a,
        )


# ══════════════════════════════════════════════════════════════
# §3.4 JudgeLine
# ══════════════════════════════════════════════════════════════


class JudgeLine(BaseModel):
    """一条判定线：4 层普通事件轨（跨层求和）+ extended 第 5 层。

    ⚠️ 判定线身份由「judgeLineList 索引 + 事件轨条件」共同确定，
    **排序不变性不成立**（RFC-0029 §2.4-3）：line_id 不是可互换的对称标签。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    line_id: int
    group: int = 0
    name: str = "Untitled"
    texture: str = "line.png"
    anchor: tuple[float, float] = (0.5, 0.5)
    event_layers: list[EventLayer] = Field(
        default_factory=lambda: [EventLayer()],
        min_length=1,
        max_length=RPE_MAX_EVENT_LAYERS,
    )
    extended: ExtendedLayer = Field(default_factory=ExtendedLayer)
    father: int = -1
    rotate_with_father: bool = True
    #: **1 = 遮罩，其余 = 不遮罩**（同 above 类陷阱，禁止 bool()）
    is_cover: int = 1
    z_order: int = 0
    attach_ui: str | None = None
    #: 存储但**不参与换算**：prpr 标为 TODO（存疑 D4）
    bpm_factor: float = 1.0
    #: 冗余字段，仅供解析期交叉校验，不参与语义
    num_of_notes_raw: int = 0

    def sum_track(self, name: str, t_beats: float, default: float = 0.0) -> float:
        """跨层求和取一条普通轨的值（**不是取最上层**）。"""
        total = 0.0
        for layer in self.event_layers:
            total += track_value(getattr(layer, name), t_beats, default)
        return total

    def ancestry(self, chart: PhigrosChart) -> list[JudgeLine]:
        """自身到根的父线链；成环或越界抛 ValueError（契约不变量 I8）。"""
        chain: list[JudgeLine] = [self]
        seen = {self.line_id}
        current = self
        while current.father != -1:
            if not 0 <= current.father < len(chart.lines):
                raise ValueError(f"line {current.line_id} 的 father={current.father} 越界")
            if current.father in seen:
                raise ValueError(f"father 链成环：line {current.line_id}")
            seen.add(current.father)
            current = chart.lines[current.father]
            chain.append(current)
        return chain

    def pose_at(self, t_beats: float, chart: PhigrosChart) -> LinePose:
        """判定线在**拍** t_beats 处的位姿（跨层求和 + 父线位置递归叠加）。

        ⚠️ 参数是**拍**（RPE 的原生时间单位），不是秒：秒到拍的换算只允许在
        beatmorph/field/ 内实现（CLAUDE.md 红线 7），契约层不持有 BPM 积分。
        """
        chain = self.ancestry(chart)
        default_alpha = -255.0 if t_beats < 0.0 else 0.0
        x = sum(line.sum_track("move_x", t_beats) for line in chain)
        y = sum(line.sum_track("move_y", t_beats) for line in chain)
        rotate = sum(line.sum_track("rotate", t_beats) for line in chain)
        alpha_raw = sum(line.sum_track("alpha", t_beats, default_alpha) for line in chain)
        return LinePose(
            x=x,
            y=y,
            rotate_deg=rotate,
            alpha=max(0.0, min(1.0, alpha_raw / 255.0)),
            scale_x=track_value(self.extended.scale_x, t_beats, 1.0),
            scale_y=track_value(self.extended.scale_y, t_beats, 1.0),
        )

    def local_to_stage(self, t_beats: float, chart: PhigrosChart) -> Transform:
        """局部系到舞台系的变换（供跨线几何检查；**契约层不做任何钳位**）。"""
        pose = self.pose_at(t_beats, chart)
        return Transform(translate_x=pose.x, translate_y=pose.y, rotate_deg=pose.rotate_deg)


# ══════════════════════════════════════════════════════════════
# §3.5 PhigrosNote
# ══════════════════════════════════════════════════════════════


class PhigrosNote(BaseModel):
    """一个 Phigros 标记（建模字段 + 无损往返所需的原始字段）。

    时间一律**秒**（CLAUDE.md 红线 7；禁止存 beat 或帧索引）。
    above / type / isFake 采用**双写**策略：语义字段供建模，*_raw 供导出往返，
    既不把格式陷阱带进模型，也不在 IR 上制造不可逆的信息损失。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    line_id: int
    #: 判定时刻（秒）= startTime beat 经 BPMList 换算。
    #: 契约层**不**对负值设限：RPE 未规定 startTime 非负，负值由质检统计而非拒收。
    t: float
    #: RPE 舞台系 x 坐标单位；可见范围 [-RPE_STAGE_HALF_WIDTH, +RPE_STAGE_HALF_WIDTH]
    position_x: float
    side: Side
    type: NoteType
    #: 秒；仅 HOLD 有意义（= endTime - startTime），其余恒为 0（格式事实，非默认值）
    hold_time: float = Field(default=0.0, ge=0.0)
    #: 音符流速倍率（实际速度 = 判定线速度 * 本值）
    speed: float = 1.0
    #: isFake == 1 → 假音符（1 为假、其余为真，只认 == 1）
    is_fake: bool = False
    # ── 无损往返所需的原始字段（不参与建模，导出时必须原样写回）──
    above_raw: int = 1
    type_raw: int = 1
    is_fake_raw: int = 0
    #: RPE-y 单位，实际偏移 = y_offset * speed（A 级源码）
    y_offset: float = 0.0
    #: 秒（默认值三方一致裁定：999999.0）
    visible_time: float = 999999.0
    alpha: int = Field(default=255, ge=0, le=255)
    size: float = 1.0

    def is_hold(self) -> bool:
        """是否 Hold（契约层唯一的 Hold 判定）。"""
        return self.type is NoteType.HOLD


class OutOfRangeStats(BaseModel):
    """positionX 越界统计（**只统计不钳位**，契约不变量 I3）。

    count 是越界数；min_x / max_x 恒为**全体** note 的 positionX 极值，
    语义唯一——越界只是「极值超过半宽」，原值一律保留在 notes 里。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    count: int = 0
    total: int = 0
    min_x: float = 0.0
    max_x: float = 0.0

    @property
    def fraction(self) -> float:
        """越界比例。"""
        return self.count / self.total if self.total else 0.0


# ══════════════════════════════════════════════════════════════
# §3.6 PhigrosChart
# ══════════════════════════════════════════════════════════════


class BpmPoint(BaseModel):
    """一个 BPM 段起点（拍坐标 + BPM）。

    时间单位是**拍**（不是秒）：RPE 的 BPMList 以 beat 三元组给出分段点。
    本类型是下游 field/ 派生 J(τ) = dt/dτ 与秒到拍换算的**唯一依据**
    （红线 7、RFC-0029 §7-8）——解析器不得丢弃或规范化掉非整拍 / 非常见分母。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    time_beats: float = Field(ge=0.0)
    bpm: float = Field(gt=0.0, description="每分钟节拍数")


class ChartMeta(BaseModel):
    """谱面元数据（RPE META.* 的契约化视图）。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: RPE 为**毫秒**（与官谱的秒制不同）；只记录不使用（Q11 / 存疑 D10）
    offset_ms: float = 0.0
    #: 100~160，但**不能**据此判定能力集
    rpe_version: int = 0
    #: 写谱时长（秒）
    chart_time_s: float = 0.0
    #: 自由文本等级（**不得** regex 解析成难度）
    level_text: str = ""
    #: info.yml 的 f32 定数；比较前 round 到 0.1。None = 未提供
    difficulty: float | None = None
    name: str = ""
    composer: str = ""
    charter: str = ""
    song: str = ""
    background: str = ""


class ChartSource(BaseModel):
    """谱面来源与嗅探留痕（合规硬约束③：可逐张追溯）。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chart_id: int | None = None
    format: ChartFormat = ChartFormat.UNKNOWN
    #: 内容嗅探依据（命中的关键字段名 / 行结构），供审计
    sniff_evidence: str = ""
    #: info.yml 定位到的谱面文件名（**只来自** info.yml.chart）
    chart_file: str = ""
    #: info.yml 定位到的音频文件名（**只来自** info.yml.music）
    music_file: str = ""
    #: 谱面条目 sha1（去重与追溯）
    chart_sha1: str = ""


class PhigrosChart(BaseModel):
    """一张 Phigros 谱面的规范中间表示（IR，格式无关）。

    K = len(lines) >= 1（v1 **不限制** K，也不限制 K = 1）；N = len(notes) >= 0。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: IR schema 版本，破坏性变更升号
    version: str = "phigros-ir-1"
    mode: GameMode = GameMode.PHIGROS
    lines: list[JudgeLine] = Field(min_length=1)
    notes: list[PhigrosNote] = Field(default_factory=list)
    bpm_points: list[BpmPoint] = Field(min_length=1)
    meta: ChartMeta = Field(default_factory=ChartMeta)
    source: ChartSource = Field(default_factory=ChartSource)

    @model_validator(mode="after")
    def _check_bpm_and_fathers(self) -> PhigrosChart:
        """I8 / I13：bpm_points 按拍升序、首段从 0 起；father 索引合法且无环。"""
        previous = -1.0
        for point in self.bpm_points:
            if point.time_beats < previous:
                raise ValueError("bpm_points 必须按 time_beats 升序")
            previous = point.time_beats
        for line in self.lines:
            if line.father != -1 and not 0 <= line.father < len(self.lines):
                raise ValueError(f"line {line.line_id} 的 father={line.father} 越界")
            line.ancestry(self)
        return self

    def line_by_id(self, line_id: int) -> JudgeLine:
        """按 line_id 取判定线（line_id 即 judgeLineList 索引）。"""
        if not 0 <= line_id < len(self.lines):
            raise IndexError(f"line_id={line_id} 越界（K={len(self.lines)}）")
        return self.lines[line_id]

    def duration_s(self) -> float:
        """谱面时长（秒）：以最后一次事件的判定时刻与 chart_time 的较大者为准。

        ⚠️ Q15 之后场的时间轴由 BPMList 与时长共同决定，但**终点口径**
        （chartTime / 最后一事件 / 音频时长 / 是否含 offset）尚未查证
        （plan 03 §9-14）——本方法只提供契约侧的保守上界，正式口径待裁定。
        """
        last = max((note.t + note.hold_time for note in self.notes), default=0.0)
        return max(last, self.meta.chart_time_s)

    def sorted_notes(self) -> list[PhigrosNote]:
        """按 (t, line_id, position_x) 的确定性排序。"""
        return sorted(self.notes, key=lambda n: (n.t, n.line_id, n.position_x))

    def notes_per_line(self) -> list[int]:
        """长度 K 的每线 note 数（**空线计入**）。"""
        counts = [0] * len(self.lines)
        for note in self.notes:
            counts[note.line_id] += 1
        return counts

    def out_of_visible_range(self) -> OutOfRangeStats:
        """统计 positionX 越界（**只统计，不抛错、不钳位**，I3）。"""
        if not self.notes:
            return OutOfRangeStats(count=0, total=0)
        xs = [note.position_x for note in self.notes]
        return OutOfRangeStats(
            count=sum(1 for x in xs if abs(x) > RPE_STAGE_HALF_WIDTH),
            total=len(xs),
            min_x=min(xs),
            max_x=max(xs),
        )
