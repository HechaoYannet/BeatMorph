"""秒 <-> 拍换算的**接缝一致性**测试（plan 02 §9-11 / plan 03 §9-13）。

背景：CLAUDE.md 红线 7 规定「beat-aligned 的时间换算只允许在 field/ 内实现」，
而 plan 02 的解析器又必须把 RPE 的原生 beat 三元组转成契约层的**秒**
（PhigrosNote.t 用秒）。两处目前各自实现同一个分段积分，plan 明确记为
**未裁定**（"裁定前只保留 plan 02 那一处格式层换算，不得出现第三处"）。

在本轮把接缝裁定为 RFC 之前，本测试是该接缝的**唯一防线**：
数据层与场层的两条换算路径必须在多 BPM 段谱面上逐点一致，
否则「同一个谱面有两个时间轴」——这正是 beat-aligned 版的 25 Hz 事件。

如果本测试失败，**不要**去改其中一个实现让它"对上"，而应：
① 确认哪一侧符合 docs/knowledges/phigros-format.md §7.1 的分段积分语义；
② 把两侧的差异写进 plan 03 §9-13 并提 RFC。
"""

from __future__ import annotations

import pytest

from beatmorph.core.contracts import (
    BpmPoint,
    ChartMeta,
    ChartSource,
    JudgeLine,
    NoteType,
    PhigrosChart,
    PhigrosNote,
    Side,
)
from beatmorph.data.parsers.rpejson import beat_to_seconds, seconds_to_beat
from beatmorph.data.qc import quality_check
from beatmorph.field.grid import seconds_to_tau, tau_to_seconds

#: 相对误差容差：纯 Python float64 分段积分，两条路径只有运算顺序差异。
REL_TOL = 1e-9

#: 多 BPM 段，含**非整拍**变更点（RPE 的 BPMList 起点是 i + n/d，d 不保证整除 48）。
BPM_SEGMENTS: tuple[tuple[float, float], ...] = (
    (0.0, 180.0),
    (7.25, 200.0),
    (16.5, 150.0),
    (31.75, 240.0),
)


def _bpm_points() -> tuple[BpmPoint, ...]:
    return tuple(BpmPoint(time_beats=beat, bpm=bpm) for beat, bpm in BPM_SEGMENTS)


def _analytic_seconds(beats: float, segments: tuple[tuple[float, float], ...]) -> float:
    """逐段解析积分：sum(段内拍数 * 60 / 段 BPM)。与任何实现无关的第三方口径。"""
    total = 0.0
    for index, (start, bpm) in enumerate(segments):
        if beats <= start:
            break
        end = segments[index + 1][0] if index + 1 < len(segments) else beats
        span = min(beats, end) - start
        if span <= 0.0:
            continue
        total += span * (60.0 / bpm)
        if beats <= end:
            break
    return total


def _probe_beats() -> list[float]:
    """覆盖段内、段界、段界两侧极近处的探针（含 0 与末段之外）。"""
    probes = [0.0]
    for beat, _ in BPM_SEGMENTS:
        probes += [beat, beat + 1e-9, beat - 1e-9, beat + 0.5]
    probes += [50.0, 100.0]
    return probes


@pytest.mark.integration
def test_two_entry_points_agree_on_multi_bpm_charts() -> None:
    """数据层的格式换算与场层的 tau 换算必须给出同一个秒值。"""
    points = _bpm_points()
    for beats in _probe_beats():
        assert beat_to_seconds(beats, points) == pytest.approx(
            tau_to_seconds(beats, points),
            rel=REL_TOL,
            abs=REL_TOL,
        ), beats


@pytest.mark.integration
def test_two_entry_points_agree_on_the_inverse() -> None:
    points = _bpm_points()
    for beats in _probe_beats():
        seconds = _analytic_seconds(beats, BPM_SEGMENTS)
        assert seconds_to_beat(seconds, points) == pytest.approx(
            seconds_to_tau(seconds, points),
            rel=REL_TOL,
            abs=REL_TOL,
        ), beats


@pytest.mark.integration
def test_both_entry_points_match_the_analytic_piecewise_integral() -> None:
    """两侧都必须等于「逐段 拍数 * 60 / BPM 求和」的解析值（多 BPM 段）。"""
    points = _bpm_points()
    for beats in _probe_beats():
        expected = _analytic_seconds(beats, BPM_SEGMENTS)
        assert beat_to_seconds(beats, points) == pytest.approx(expected, rel=REL_TOL, abs=REL_TOL)
        assert tau_to_seconds(beats, points) == pytest.approx(expected, rel=REL_TOL, abs=REL_TOL)


@pytest.mark.integration
def test_roundtrip_is_lossless_in_both_layers() -> None:
    """往返无损：拍 -> 秒 -> 拍 与 秒 -> 拍 -> 秒 在两侧都成立。"""
    points = _bpm_points()
    for beats in _probe_beats():
        seconds = beat_to_seconds(beats, points)
        assert seconds_to_tau(seconds, points) == pytest.approx(beats, rel=REL_TOL, abs=REL_TOL)
        assert seconds_to_beat(seconds, points) == pytest.approx(beats, rel=REL_TOL, abs=REL_TOL)
    for beats in _probe_beats():
        seconds = _analytic_seconds(beats, BPM_SEGMENTS)
        assert beat_to_seconds(seconds_to_beat(seconds, points), points) == pytest.approx(
            seconds,
            rel=REL_TOL,
            abs=REL_TOL,
        )


@pytest.mark.integration
def test_changing_bpm_changes_both_entry_points() -> None:
    """改写 BPMList 必须使两侧结果都变化（否则实现里藏了硬编码）。"""
    points = _bpm_points()
    slower = tuple(BpmPoint(time_beats=point.time_beats, bpm=point.bpm / 2.0) for point in points)
    beats = 20.0
    assert beat_to_seconds(beats, slower) == pytest.approx(
        2.0 * beat_to_seconds(beats, points),
        rel=REL_TOL,
    )
    assert tau_to_seconds(beats, slower) == pytest.approx(
        2.0 * tau_to_seconds(beats, points),
        rel=REL_TOL,
    )


@pytest.mark.integration
def test_non_zero_bpm_anchor_diverges_and_is_gated_by_qc() -> None:
    """首段不在 0 拍时两条换算会相差一个常量——该情形必须被**质检拒收**。

    这是接缝目前**唯一已知的分歧点**（field-agent 2026-09-27 实测）：
    格式层以 BPMList 首段为时间原点（与 Phira 官方参考实现 beat2sec 一致），
    而 field/ 把 tau=0 当作谱面时间原点并在前面外推一段。RPE 规范首段是
    [0,0,1]，因此该情形只可能来自畸形谱面。

    临时裁定（待 RFC）：**不静默二选一，而是按 schema 违约进隔离区**。
    本测试把「分歧真实存在」与「分歧被拦住」两件事同时钉住——若哪天两侧
    统一了，本测试会失败并提醒删除 qc.py 的对应规则与 plan 的临时裁定。
    """
    points = (BpmPoint(time_beats=4.0, bpm=120.0), BpmPoint(time_beats=8.0, bpm=240.0))
    assert beat_to_seconds(4.0, points) != pytest.approx(tau_to_seconds(4.0, points), rel=1e-9)

    chart = PhigrosChart(
        lines=[JudgeLine(line_id=0)],
        notes=[
            PhigrosNote(line_id=0, t=0.0, position_x=0.0, side=Side.FRONT, type=NoteType.TAP),
        ],
        bpm_points=list(points),
        meta=ChartMeta(chart_time_s=10.0),
        source=ChartSource(),
    )
    report = quality_check(chart, None)
    assert not report.passed
    assert any("BPMList 首段" in error for error in report.errors), report.errors
