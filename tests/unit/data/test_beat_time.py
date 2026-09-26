"""M6⑤：格式层 beat ↔ 秒换算（plan 02 §3.4-1，多 BPM 段往返 1e-9）。"""

from __future__ import annotations

import pytest

from beatmorph.core.contracts import BpmPoint
from beatmorph.data.parsers.rpejson import beat_to_seconds, seconds_to_beat

#: 测试内**独立复算**用的常量（与实现的私有常量无关，避免同源错误）。
_SECONDS_PER_MINUTE = 60.0

TOLERANCE = 1e-9


def _points() -> list[BpmPoint]:
    """两个 BPM 段：120 BPM 从 0 拍起，180 BPM 从第 8 拍起。"""
    return [BpmPoint(time_beats=0.0, bpm=120.0), BpmPoint(time_beats=8.0, bpm=180.0)]


def _reference_beat_to_seconds(beats: float, points: list[BpmPoint]) -> float:
    """独立复算（朴素逐段循环，不用实现里的任何私有函数）。"""
    if beats <= points[0].time_beats:
        return (beats - points[0].time_beats) * _SECONDS_PER_MINUTE / points[0].bpm
    seconds = 0.0
    for index, point in enumerate(points):
        span_end = points[index + 1].time_beats if index + 1 < len(points) else float("inf")
        if beats <= span_end:
            return seconds + (beats - point.time_beats) * _SECONDS_PER_MINUTE / point.bpm
        seconds += (span_end - point.time_beats) * _SECONDS_PER_MINUTE / point.bpm
    raise AssertionError("unreachable")


def test_single_segment_matches_bpm_definition() -> None:
    points = _points()
    assert beat_to_seconds(4.0, points) == pytest.approx(
        4.0 * _SECONDS_PER_MINUTE / 120.0,
        abs=TOLERANCE,
    )


def test_zero_beat_is_zero_seconds() -> None:
    assert beat_to_seconds(0.0, _points()) == 0.0


def test_multi_segment_is_piecewise() -> None:
    points = _points()
    expected = 8.0 * _SECONDS_PER_MINUTE / 120.0 + 2.0 * _SECONDS_PER_MINUTE / 180.0
    assert beat_to_seconds(10.0, points) == pytest.approx(expected, abs=TOLERANCE)


def test_segment_boundary_is_exact() -> None:
    points = _points()
    boundary = beat_to_seconds(8.0, points)
    assert boundary == pytest.approx(8.0 * _SECONDS_PER_MINUTE / 120.0, abs=TOLERANCE)
    assert beat_to_seconds(8.0 + 1e-12, points) >= boundary


@pytest.mark.parametrize("beats", [-4.0, -0.5, 0.0, 0.25, 3.0, 8.0, 8.0001, 12.5, 40.0])
def test_beat_round_trip(beats: float) -> None:
    """beat → 秒 → beat 往返（多 BPM 段，1e-9 容差）。"""
    points = _points()
    assert seconds_to_beat(beat_to_seconds(beats, points), points) == pytest.approx(
        beats,
        abs=TOLERANCE,
    )


@pytest.mark.parametrize("seconds", [-1.0, 0.0, 0.5, 4.0, 4.0001, 7.25, 30.0])
def test_seconds_round_trip(seconds: float) -> None:
    """秒 → beat → 秒 往返。"""
    points = _points()
    assert beat_to_seconds(seconds_to_beat(seconds, points), points) == pytest.approx(
        seconds,
        abs=TOLERANCE,
    )


def test_negative_time_extrapolates_with_first_bpm() -> None:
    """首段之前用首段 BPM 线性外推（与官方 beat2sec 的循环行为一致）。"""
    points = _points()
    assert beat_to_seconds(-2.0, points) == pytest.approx(
        -2.0 * _SECONDS_PER_MINUTE / 120.0,
        abs=TOLERANCE,
    )
    assert seconds_to_beat(-1.0, points) == pytest.approx(
        -1.0 / (_SECONDS_PER_MINUTE / 120.0),
        abs=TOLERANCE,
    )


def test_matches_independent_reference_over_a_sweep() -> None:
    """与独立复算逐点比对（含段边界前后）。"""
    points = _points()
    for step in range(-20, 400):
        beats = step * 0.25
        assert beat_to_seconds(beats, points) == pytest.approx(
            _reference_beat_to_seconds(beats, points),
            abs=1e-9,
        )


def test_non_integer_denominator_is_preserved() -> None:
    """非常见分母（如 1/3 拍）不得被规范化掉：0.5 BPM 段的换算仍然正确。"""
    points = [BpmPoint(time_beats=0.0, bpm=120.0), BpmPoint(time_beats=1.0 / 3.0, bpm=90.0)]
    beats = 1.0 / 3.0 + 2.0
    expected = (1.0 / 3.0) * _SECONDS_PER_MINUTE / 120.0 + 2.0 * _SECONDS_PER_MINUTE / 90.0
    assert beat_to_seconds(beats, points) == pytest.approx(expected, abs=TOLERANCE)


def test_empty_bpm_points_raises() -> None:
    with pytest.raises(ValueError, match="不得为空"):
        beat_to_seconds(1.0, [])
    with pytest.raises(ValueError, match="不得为空"):
        seconds_to_beat(1.0, [])


def test_unsorted_bpm_points_raises() -> None:
    points = [BpmPoint(time_beats=4.0, bpm=120.0), BpmPoint(time_beats=2.0, bpm=150.0)]
    with pytest.raises(ValueError, match="升序"):
        beat_to_seconds(1.0, points)
