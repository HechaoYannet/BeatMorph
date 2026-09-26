"""eval 单元测试的合成构造器（**不得固化物理常量**，AGENTS.md §3.3 / 红线 7）。

纪律：

- 时间一律由 BPM 与契约常量派生（SECONDS_PER_MINUTE），不写字面量秒数；
- 位置一律由 RPE_STAGE_WIDTH / x_bins 派生；
- **容差一律取自 EvalConfig**（plan 06 §8），测试内不得出现 0.020 / 0.050 这类字面量；
- 拍细分用 SUBDIVISIONS_PER_BEAT 派生（1/4、1/8、1/12 … 都是它的整数分之一）。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from beatmorph.core.contracts.phigros import (
    RPE_STAGE_WIDTH,
    BpmPoint,
    ChartMeta,
    ChartSource,
    JudgeLine,
    NoteType,
    PhigrosChart,
    PhigrosNote,
    Side,
    side_from_above,
)
from beatmorph.decoder.events import DecodedEvent
from beatmorph.eval.protocol import EvalCase, EvalConfig
from beatmorph.field.grid import SECONDS_PER_MINUTE, seconds_to_tau

#: 测试用 BPM（**配置量**，不是物理常量）。
BPM: float = 120.0

#: 标记轮换表（让 corruption 的「类型/侧别」注入真的能改变事件，而不是同义改写）。
TYPE_CYCLE: tuple[NoteType, ...] = (NoteType.TAP, NoteType.HOLD, NoteType.FLICK, NoteType.DRAG)
SIDE_CYCLE: tuple[Side, ...] = (Side.FRONT, Side.BACK)


def beats_to_seconds(beats: float, bpm: float = BPM) -> float:
    """拍 -> 秒（测试用解析式，独立于被测实现）。"""
    return beats * SECONDS_PER_MINUTE / bpm


def make_bpm_points(bpm: float = BPM) -> tuple[BpmPoint, ...]:
    """单一 BPM 段（秒 <-> 拍换算的唯一依据）。"""
    return (BpmPoint(time_beats=0.0, bpm=bpm),)


def x_for_bin(x_bin: int, *, x_bins: int) -> float:
    """x 桶 -> 桶中心（由 RPE_STAGE_WIDTH / x_bins 派生，禁止写裸坐标）。"""
    dx = RPE_STAGE_WIDTH / x_bins
    return -RPE_STAGE_WIDTH / 2.0 + (x_bin + 0.5) * dx


def make_event(
    t_s: float,
    *,
    line_id: int = 0,
    position_x: float = 0.0,
    side: Side = Side.FRONT,
    note_type: NoteType = NoteType.TAP,
    hold_time_s: float = 0.0,
    confidence: float = 1.0,
) -> DecodedEvent:
    """一个生成侧事件（秒域）。"""
    return DecodedEvent(
        t_s=float(t_s),
        line_id=int(line_id),
        position_x=float(position_x),
        side=side,
        note_type=note_type,
        hold_time_s=float(hold_time_s),
        is_fake=False,
        confidence=float(confidence),
    )


def varied_events(
    count: int,
    *,
    beat_step: float = 1.0,
    bpm: float = BPM,
    x_bins: int = 128,
    start_beat: float = 0.0,
) -> tuple[DecodedEvent, ...]:
    """标记各异、时间规则的事件列（类型轮换 + 侧别轮换 + x 逐桶变化）。

    `beat_step» 以**拍**为单位（1/4、1/8、1/12 都用 SUBDIVISIONS_PER_BEAT 的分数表达），
    因此构造器本身不引入任何物理常量。
    """
    events: list[DecodedEvent] = []
    for index in range(count):
        events.append(
            make_event(
                beats_to_seconds(start_beat + index * beat_step, bpm),
                line_id=index % 2,
                position_x=x_for_bin(index % x_bins, x_bins=x_bins),
                side=SIDE_CYCLE[index % len(SIDE_CYCLE)],
                note_type=TYPE_CYCLE[index % len(TYPE_CYCLE)],
            )
        )
    return tuple(events)


def perfect_case(
    key: str = "song-a",
    *,
    count: int = 8,
    beat_step: float = 1.0,
    difficulty: float | None = None,
    level_text: str = "",
    song: str = "",
) -> EvalCase:
    """pred == gold 的用例（M6.1 的解析解：F1 必须为 1）。"""
    events = varied_events(count, beat_step=beat_step)
    return make_case(
        key,
        pred=events,
        gold=events,
        difficulty=difficulty,
        level_text=level_text,
        song=song,
    )


def make_case(
    key: str,
    *,
    pred: Sequence[DecodedEvent],
    gold: Sequence[DecodedEvent],
    difficulty: float | None = None,
    level_text: str = "",
    song: str = "",
    bpm: float = BPM,
) -> EvalCase:
    """评估输入单元（gold 提供 BPMList：人类谱是时间轴权威）。"""
    return EvalCase(
        key=key,
        pred=tuple(pred),
        gold=tuple(gold),
        bpm_points=make_bpm_points(bpm),
        difficulty=difficulty,
        level_text=level_text,
        song=song,
    )


def make_note(
    *,
    t: float,
    line_id: int = 0,
    position_x: float = 0.0,
    note_type: NoteType = NoteType.TAP,
    above: int = 1,
    hold_time: float = 0.0,
) -> PhigrosNote:
    """一条契约 note（双写口径与语义字段一致）。"""
    return PhigrosNote(
        line_id=line_id,
        t=float(t),
        position_x=float(position_x),
        side=side_from_above(above),
        type=note_type,
        hold_time=float(hold_time),
        above_raw=above,
        type_raw=int(note_type),
    )


def make_chart(
    *,
    notes: Iterable[PhigrosNote],
    k: int = 2,
    chart_time_s: float = 0.0,
    difficulty: float | None = None,
    level_text: str = "",
    bpm: float = BPM,
) -> PhigrosChart:
    """合成谱面 IR（用于 evaluate_charts 的全链路集成测试）。"""
    return PhigrosChart(
        lines=[JudgeLine(line_id=index) for index in range(k)],
        notes=list(notes),
        bpm_points=list(make_bpm_points(bpm)),
        meta=ChartMeta(chart_time_s=chart_time_s, difficulty=difficulty, level_text=level_text),
        source=ChartSource(sniff_evidence="synthetic"),
    )


def events_to_notes(events: Sequence[DecodedEvent]) -> list[PhigrosNote]:
    """事件 -> 契约 note（集成测试的「生成谱」构造路径）。"""
    return [
        make_note(
            t=event.t_s,
            line_id=event.line_id,
            position_x=event.position_x,
            note_type=event.note_type,
            above=1 if event.side is Side.FRONT else 0,
            hold_time=event.hold_time_s,
        )
        for event in events
    ]


def beats_of(t_s: float, bpm: float = BPM) -> float:
    """秒 -> 拍（**经 field/ 的权威换算**；测试不重写 BPM 积分）。"""
    return float(seconds_to_tau(t_s, make_bpm_points(bpm)))


def beat_fraction(denominator: int) -> float:
    """1/denominator 拍（时间组用的细分；不依赖任何物理常量）。"""
    return 1.0 / denominator


def tolerance_of(config: EvalConfig, index: int = 0) -> float:
    """从 EvalConfig 取容差（plan §8：容差必须来自 EvalConfig，不得是测试内字面量）。"""
    return float(config.tolerances_s[index])
