"""判定线资格闸门：装饰线 / 不可见线上不该产生 note（决策者 2026-09-30 实测报告）。

为什么值得一组契约级断言：`beatmorph-eval` 之前对「note 落在一条 alpha=0 的判定线上」
**完全没有判据**（`legality.py` 只查 line_id 越界 / 重复 / 同刻上限 / 跨线几何），
于是 e2e 产物 665 个 note 里 238 个（35.8%）落在不可见线上、149 个（22.4%）落在
装饰线上，而 `is_legal=True` 照样成立。**去掉这个闸门不会报任何错**，只会让产物重新
变成不可判定的样子——这正是要钉住它的理由。
"""

from __future__ import annotations

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
from beatmorph.decoder.events import (
    LINE_FILTER_KEYS,
    FieldEvent,
    filter_field_events_by_line,
    scorable_lines,
)
from beatmorph.field.grid import TAU_GRID_DT
from tests.unit.decoder._builders import make_bpm_points, make_grid

BPM = 120.0
T_BINS = 48


def _alpha_track(value_at_zero: float, value_at_late: float, switch_beat: float):
    """两段式 alpha 轨：beat < switch 取第一个值，其后取第二个值（Beat = i + n/d）。"""
    whole = int(switch_beat)
    switch = Beat(
        i=whole, n=round((switch_beat - whole) * SUBDIVISIONS_PER_BEAT), d=SUBDIVISIONS_PER_BEAT
    )
    return EventLayer(
        layer_index=0,
        alpha=[
            EventKeyframe(
                start_time=Beat(i=0), end_time=switch, start=value_at_zero, end=value_at_zero
            ),
            EventKeyframe(
                start_time=switch,
                end_time=Beat(i=int(T_BINS * 4)),
                start=value_at_late,
                end=value_at_late,
            ),
        ],
    )


def _chart(lines: list[JudgeLine]) -> PhigrosChart:
    return PhigrosChart(
        lines=lines,
        notes=[],
        bpm_points=make_bpm_points((0.0, BPM)),
        meta=ChartMeta(chart_time_s=10.0),
        source=ChartSource(sniff_evidence="test"),
    )


def _grid():
    return make_grid(bpm_points=make_bpm_points((0.0, BPM)), t_bins=T_BINS)


def _event(line_id: int, tau_bin: int) -> FieldEvent:
    return FieldEvent(
        line_id=line_id,
        tau=(int(tau_bin) + 0.5) * TAU_GRID_DT,
        x_bin=0,
        side=Side.FRONT,
        channel=0,
    )


def _run(chart, events, grid=None, **kwargs):
    return filter_field_events_by_line(
        events,
        chart=chart,
        window_grid=grid if grid is not None else _grid(),
        origin_s=0.0,
        **kwargs,
    )


def test_invisible_line_events_are_dropped() -> None:
    chart = _chart(
        [
            JudgeLine(line_id=0, event_layers=[_alpha_track(255, 255, 100.0)]),
            JudgeLine(line_id=1, event_layers=[_alpha_track(0, 0, 100.0)]),
        ]
    )
    kept, stats = _run(chart, [_event(0, 4), _event(1, 4)])
    assert [e.line_id for e in kept] == [0]
    assert stats["line_filter_dropped_invisible"] == 1.0
    assert stats["line_filter_kept_events"] == 1.0
    assert set(stats) == set(LINE_FILTER_KEYS)


def test_visibility_is_evaluated_at_the_event_time() -> None:
    """同一条线：先可见、后不可见 —— 判据必须**逐事件**求值，不是逐线一次性。"""
    chart = _chart([JudgeLine(line_id=0, event_layers=[_alpha_track(255, 0, 0.5)])])
    early, late = _event(0, 4), _event(0, 40)
    kept, stats = _run(chart, [early, late])
    assert [e.tau for e in kept] == [early.tau]
    assert stats["line_filter_dropped_invisible"] == 1.0


def test_child_of_invisible_parent_is_dropped() -> None:
    """alpha 跨层求和 + 父线递归：父线不可见时子线也不可见（pose_at 的语义）。"""
    chart = _chart(
        [
            JudgeLine(line_id=0, event_layers=[_alpha_track(0, 0, 100.0)]),
            JudgeLine(line_id=1, father=0),
        ]
    )
    kept, stats = _run(chart, [_event(0, 4), _event(1, 4)])
    assert kept == []
    assert stats["line_filter_dropped_invisible"] == 2.0


def test_missing_alpha_track_means_unknown_not_invisible() -> None:
    """合成模板（解码器自建的空判定线）**没有** alpha 轨 ⇒ 不得据此判不可见。

    RPE 的默认值语义（没有 alphaEvents ⇒ 求和为 0）只适用于「谱面确实带了 alpha 轨」，
    而合成路径连这个信息都没有。把它当 0 会把合成路径的全部 note 一次清空。
    """
    chart = _chart([JudgeLine(line_id=0), JudgeLine(line_id=1)])
    kept, stats = _run(chart, [_event(0, 4), _event(1, 4)])
    assert len(kept) == 2
    assert stats["line_filter_dropped_invisible"] == 0.0


def test_allowed_lines_drops_decorative_lines() -> None:
    chart = _chart(
        [
            JudgeLine(line_id=0, event_layers=[_alpha_track(255, 255, 100.0)]),
            JudgeLine(line_id=1, event_layers=[_alpha_track(255, 255, 100.0)]),
            JudgeLine(line_id=2, event_layers=[_alpha_track(255, 255, 100.0)]),
        ]
    )
    kept, stats = _run(
        chart, [_event(0, 4), _event(1, 4), _event(2, 4)], allowed_lines=frozenset({0, 1})
    )
    assert [e.line_id for e in kept] == [0, 1]
    assert stats["line_filter_dropped_empty_line"] == 1.0
    assert stats["line_filter_allowed_lines"] == 2.0


def test_no_allowed_lines_means_no_decorative_filtering() -> None:
    chart = _chart([JudgeLine(line_id=0, event_layers=[_alpha_track(255, 255, 100.0)])])
    kept, stats = _run(chart, [_event(0, 4)])
    assert len(kept) == 1
    assert stats["line_filter_allowed_lines"] == -1.0
    assert stats["line_filter_dropped_empty_line"] == 0.0


# ── allowed_lines 的**来源**口径（决策者 2026-09-30 裁定）─────────────────────────
# 旧口径 `frozenset(note.line_id for note in template.notes)` 只看「有没有 note」，
# 于是「只有假音符」和「命中时线不可见」的表演线也被放进了 allowed_lines。
# 真实语料 45 张 / 53 007 个 note 里，落在装饰线上的**可计分** note = **0**，
# 而我们的 e2e 产物是唯一的例外（23 个，5.6%）。下面这条断言钉住新口径。


def _note(line_id: int, seconds: float, *, fake: bool = False) -> PhigrosNote:
    return PhigrosNote(
        line_id=line_id,
        t=seconds,
        position_x=0.0,
        side=Side.FRONT,
        type=NoteType.TAP,
        is_fake=fake,
    )


def test_scorable_lines_needs_a_scorable_note_not_just_a_note() -> None:
    """只有「非 fake 且命中时线可见」的 note 才让线进入 allowed_lines。"""
    chart = _chart(
        [
            JudgeLine(line_id=0, event_layers=[_alpha_track(255, 255, 100.0)]),
            # 命中时刻（beat 0.5 = 0.25 s @120BPM）alpha = 0
            JudgeLine(line_id=1, event_layers=[_alpha_track(0.0, 255.0, 10.0)]),
            JudgeLine(line_id=2, event_layers=[_alpha_track(255.0, 255.0, 100.0)]),
            # 有层但**没有 alpha 轨**：可见性**未知**，不因此剔除
            JudgeLine(line_id=3, event_layers=[EventLayer(layer_index=0)]),
        ],
    )
    chart = chart.model_copy(
        update={
            "notes": [
                _note(0, 0.25),
                _note(1, 0.25),
                _note(2, 0.25, fake=True),
                _note(3, 0.25),
            ],
        },
    )
    assert scorable_lines(chart) == frozenset({0, 3})
    # 旧口径（只看有没有 note）会给出全部四条线 —— 这正是被修掉的那个 bug
    assert frozenset(int(note.line_id) for note in chart.notes) == frozenset({0, 1, 2, 3})


def test_scorable_lines_empty_chart_has_no_allowed_lines() -> None:
    """空模板 ⇒ 空集合（调用方据此退回「不按装饰线过滤」，而不是把所有线都当成可承载）。"""
    assert (
        scorable_lines(_chart([JudgeLine(line_id=0, event_layers=[EventLayer(layer_index=0)])]))
        == frozenset()
    )
