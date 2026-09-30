"""`data.scorable_target`：目标只入账**可计分** note（plan 07 §9-79）。

为什么值得一组断言：这条口径决定**模型学什么**，而它错起来是完全静默的——
实测（`runs/_probe_scorable_share.py`）train 的 **22.81%** / val 的 **18.68%** 事件
是「命中时该线不可见」的 note（其中一半落在**真线**上 ⇒ 线级剥离拿不掉）；
真实语料里「可计分 note 落在装饰线上」= **0**，也就是说这两类本不相交，是管线把它们混在了一起。
判据只有一份实现（`decoder.events.note_is_scorable`），生成闸门与训练目标共用。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from beatmorph.core.contracts.phigros import (
    SUBDIVISIONS_PER_BEAT,
    Beat,
    ChartMeta,
    ChartSource,
    EventKeyframe,
    EventLayer,
    JudgeLine,
    NoteType,
    PhigrosChart,
    PhigrosNote,
    Side,
)
from beatmorph.data.dataset import DatasetConfig, _plan_fingerprint
from beatmorph.decoder.events import gameplay_subchart, scorable_lines, scorable_note_mask

BPM = 120.0


def _alpha_track(value_at_zero: float, value_at_late: float, switch_beat: float) -> EventLayer:
    whole = int(switch_beat)
    switch = Beat(
        i=whole,
        n=round((switch_beat - whole) * SUBDIVISIONS_PER_BEAT),
        d=SUBDIVISIONS_PER_BEAT,
    )
    return EventLayer(
        layer_index=0,
        alpha=[
            EventKeyframe(
                start_time=Beat(i=0), end_time=switch, start=value_at_zero, end=value_at_zero
            ),
            EventKeyframe(
                start_time=switch,
                end_time=Beat(i=64),
                start=value_at_late,
                end=value_at_late,
            ),
        ],
    )


def _note(line_id: int, seconds: float, *, fake: bool = False) -> PhigrosNote:
    return PhigrosNote(
        line_id=line_id,
        t=seconds,
        position_x=0.0,
        side=Side.FRONT,
        type=NoteType.TAP,
        is_fake=fake,
    )


def _chart(lines: list[JudgeLine], notes: list[PhigrosNote]) -> PhigrosChart:
    from tests.unit.decoder._builders import make_bpm_points

    return PhigrosChart(
        lines=lines,
        notes=notes,
        bpm_points=make_bpm_points((0.0, BPM)),
        meta=ChartMeta(chart_time_s=10.0),
        source=ChartSource(sniff_evidence="test"),
    )


def _four_line_chart() -> PhigrosChart:
    """线 0 可见 / 线 1 命中时不可见 / 线 2 只有假音符 / 线 3 无 alpha 轨（可见性未知）。"""
    return _chart(
        [
            JudgeLine(line_id=0, event_layers=[_alpha_track(255.0, 255.0, 100.0)]),
            JudgeLine(line_id=1, event_layers=[_alpha_track(0.0, 255.0, 10.0)]),
            JudgeLine(line_id=2, event_layers=[_alpha_track(255.0, 255.0, 100.0)]),
            JudgeLine(line_id=3, event_layers=[EventLayer(layer_index=0)]),
        ],
        [
            _note(0, 0.25),
            _note(1, 0.25),
            _note(2, 0.25, fake=True),
            _note(3, 0.25),
        ],
    )


def test_mask_is_aligned_and_marks_only_scorable_notes() -> None:
    chart = _four_line_chart()
    assert scorable_note_mask(chart) == [True, False, False, True]
    assert scorable_lines(chart) == frozenset({0, 3})


def test_predicate_reads_the_line_alpha_at_the_hit_time() -> None:
    """判据看的是**命中时刻**的 alpha：同一条线，命中晚于切换点就变成可计分。"""
    chart = _chart(
        [JudgeLine(line_id=0, event_layers=[_alpha_track(0.0, 255.0, 10.0)])],
        [_note(0, 0.25), _note(0, 6.0)],
    )
    assert scorable_note_mask(chart) == [False, True]


def test_fake_note_is_never_scorable() -> None:
    chart = _chart(
        [JudgeLine(line_id=0, event_layers=[_alpha_track(255.0, 255.0, 100.0)])],
        [_note(0, 0.25, fake=True)],
    )
    assert scorable_note_mask(chart) == [False]


def test_gameplay_subchart_drops_decoration_lines_and_remaps_indices() -> None:
    """去表演 = **真线 ∪ 祖先线** + 只留可计分 note；father 与 line_id 一并重映射。"""
    chart = _chart(
        [
            # 0 装饰线（无 note）—— 但它是线 1 的父线，必须留着当几何载体
            JudgeLine(line_id=0, event_layers=[_alpha_track(255.0, 255.0, 100.0)]),
            JudgeLine(line_id=1, father=0, event_layers=[_alpha_track(255.0, 255.0, 100.0)]),
            # 2 装饰线：note 命中时不可见
            JudgeLine(line_id=2, event_layers=[_alpha_track(0.0, 255.0, 10.0)]),
            # 3 空装饰线
            JudgeLine(line_id=3, event_layers=[EventLayer(layer_index=0)]),
        ],
        [
            _note(1, 0.25),
            _note(2, 0.25),
            _note(3, 0.25),
        ],
    )
    playable = gameplay_subchart(chart)
    # 线 1 有可计分 note ⇒ 留下；线 0 是它的父线 ⇒ 留下（几何载体）；2 / 3 摘掉
    assert len(playable.lines) == 2
    assert playable.lines[0].line_id == 0
    assert playable.lines[1].line_id == 1
    assert playable.lines[1].father == 0
    assert [note.line_id for note in playable.notes] == [1]


def test_gameplay_subchart_raises_when_nothing_is_scorable() -> None:
    chart = _chart(
        [JudgeLine(line_id=0, event_layers=[_alpha_track(0.0, 255.0, 10.0)])],
        [_note(0, 0.25)],
    )
    with pytest.raises(ValueError, match="没有任何可计分 note"):
        gameplay_subchart(chart)


def test_scorable_target_and_window_cache_are_mutually_exclusive(tmp_path: Path) -> None:
    base = {
        "manifest_path": Path("m.json"),
        "chart_dir": Path("charts"),
        "feature_dir": Path("features"),
        "t_window": 192,
    }
    with pytest.raises(ValueError, match="不能同时开"):
        DatasetConfig(  # type: ignore[arg-type]
            **base, scorable_target=True, window_cache_dir=tmp_path
        )


def test_plan_fingerprint_covers_the_target_regime(tmp_path: Path) -> None:
    """换口径必须让索引缓存失效，否则会静默复用旧口径的窗口/台账。"""
    base = {
        "manifest_path": Path("m.json"),
        "chart_dir": Path("charts"),
        "feature_dir": Path("features"),
        "t_window": 192,
    }
    off = DatasetConfig(**base, scorable_target=False)  # type: ignore[arg-type]
    on = DatasetConfig(**base, scorable_target=True)  # type: ignore[arg-type]
    assert _plan_fingerprint(off, []) != _plan_fingerprint(on, [])
