"""field 单元测试的合成构造器。

**不得固化任何物理常量**（AGENTS.md §3.3）：时间一律由 BPM 派生、位置一律由
RPE_STAGE_HALF_WIDTH 派生、tau 格一律由 SUBDIVISIONS_PER_BEAT 派生。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from beatmorph.core.contracts.phigros import (
    RPE_STAGE_HALF_WIDTH,
    BpmPoint,
    ChartMeta,
    ChartSource,
    JudgeLine,
    NoteType,
    PhigrosChart,
    PhigrosNote,
    side_from_above,
)
from beatmorph.field.grid import DEFAULT_X_BINS, SECONDS_PER_MINUTE, FieldGrid


def seconds_for_beats(beats: float, bpm: float) -> float:
    """单一 BPM 段内 beat -> 秒（测试用解析式，独立于被测实现）。"""
    return beats * SECONDS_PER_MINUTE / bpm


def make_bpm_points(*segments: tuple[float, float]) -> list[BpmPoint]:
    """(拍, BPM) 序列 -> BpmPoint 列表。"""
    return [BpmPoint(time_beats=beats, bpm=bpm) for beats, bpm in segments]


def make_lines(k: int) -> list[JudgeLine]:
    """k 条无事件的判定线（空线必须计入积分，故测试也要覆盖）。"""
    return [JudgeLine(line_id=index) for index in range(k)]


def make_note(
    *,
    t: float,
    line_id: int = 0,
    position_x: float = 0.0,
    note_type: NoteType = NoteType.TAP,
    above: int = 1,
    hold_time: float = 0.0,
    is_fake: bool = False,
) -> PhigrosNote:
    """一条 note（side / *_raw 与语义字段保持一致的双写口径）。"""
    return PhigrosNote(
        line_id=line_id,
        t=t,
        position_x=position_x,
        side=side_from_above(above),
        type=note_type,
        hold_time=hold_time,
        is_fake=is_fake,
        above_raw=above,
        type_raw=int(note_type),
        is_fake_raw=1 if is_fake else 0,
    )


def make_chart(
    *,
    notes: Iterable[PhigrosNote],
    bpm_points: Sequence[BpmPoint],
    k: int = 1,
    chart_time_s: float = 0.0,
) -> PhigrosChart:
    """合成谱面 IR（K 条线 + BPMList + 元数据）。"""
    return PhigrosChart(
        lines=make_lines(k),
        notes=list(notes),
        bpm_points=list(bpm_points),
        meta=ChartMeta(chart_time_s=chart_time_s),
        source=ChartSource(sniff_evidence="synthetic"),
    )


def make_grid(
    *,
    bpm_points: Sequence[BpmPoint],
    t_bins: int,
    x_bins: int | None = None,
) -> FieldGrid:
    """已绑定时间轴的网格（x_bins=None 时用契约默认值）。"""
    grid = FieldGrid(x_bins=DEFAULT_X_BINS if x_bins is None else x_bins)
    bound = grid.with_time(t_bins, bpm_points)
    bound.assert_grid()
    return bound


def fractions_inside(positions: Sequence[float]) -> bool:
    """所有位置是否落在可见范围内（**只判断，不钳位**）。"""
    return all(abs(value) <= RPE_STAGE_HALF_WIDTH for value in positions)
