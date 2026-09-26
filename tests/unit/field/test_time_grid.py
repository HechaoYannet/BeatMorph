"""M12：「场网格 <-> 秒」往返无损契约测试（Q15 硬要求，默认 CI，无权重无 GPU）。

覆盖（plan §6.2 M12 / RFC-0029 §3.1）：

- 多 BPM 段（>= 2 段、含变速）上 seconds_to_tau(tau_to_seconds(tau)) == tau 且反向亦然；
- tau_to_seconds 与「逐段 Delta 拍 x 60 / bpm 求和」的**独立解析实现**一致；
- J(tau) 段内常量、段界跳变；
- **改写 BPMList 必须使结果变化**（否则说明实现里藏了硬编码或用的是帧率）；
- tau 格数与总拍数一致；cell_volumes 非均匀且与 volume 同一口径。

来历：POSTMORTEM-2026-08-05（25 Hz 误值存活到万级数据规模）——「没有这条测试，
beat-aligned 会变成下一个 25 Hz」。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from beatmorph.core.contracts.phigros import RPE_STAGE_WIDTH
from beatmorph.field.grid import (
    BEAT_SUBDIVISION,
    SECONDS_PER_MINUTE,
    FieldGrid,
    bpm_segments,
    jacobian_at,
    seconds_to_tau,
    tau_to_seconds,
)
from tests.unit.field._builders import make_bpm_points

#: 三段变速（含升速与降速），段界落在非整拍数量级上以覆盖一般情形
MULTI_BPM_PAIRS: tuple[tuple[float, float], ...] = (
    (0.0, 120.0),
    (4.0, 180.0),
    (9.5, 90.0),
)


def _reference_tau_to_seconds(tau: float, pairs: Sequence[tuple[float, float]]) -> float:
    """独立解析实现：逐段 Delta 拍 x 60 / bpm 求和（官方 beat2sec 的循环形态）。

    只依赖 (拍, BPM) 序列与 SECONDS_PER_MINUTE 这一单位定义，不调用被测模块。
    """
    assert pairs[0][0] == 0.0, "参考实现要求首段从 0 拍起"
    total = 0.0
    remaining = tau
    for index, (start, bpm) in enumerate(pairs):
        seconds_per_beat = SECONDS_PER_MINUTE / bpm
        if index + 1 == len(pairs):
            return total + remaining * seconds_per_beat
        span = pairs[index + 1][0] - start
        if remaining >= span:
            total += span * seconds_per_beat
            remaining -= span
        else:
            return total + remaining * seconds_per_beat
    return total


def _close(a: float, b: float, *, rel: float) -> bool:
    return abs(a - b) <= rel * max(1.0, abs(a), abs(b))


def test_seconds_to_tau_inverts_tau_to_seconds_multi_bpm() -> None:
    """M12：多 BPM 段上 seconds_to_tau(tau_to_seconds(tau)) == tau（相对容差 1e-12）。"""
    bpm_points = make_bpm_points(*MULTI_BPM_PAIRS)
    total_beats = MULTI_BPM_PAIRS[-1][0] * 2.0
    taus = np.linspace(0.0, total_beats, 257)
    seconds = tau_to_seconds(taus, bpm_points)
    back = seconds_to_tau(seconds, bpm_points)
    assert np.allclose(back, taus, rtol=1e-12, atol=0.0)
    # 段界与段界邻域必须逐个精确（最容易漏测的地方）
    for beats, _ in MULTI_BPM_PAIRS:
        for probe in (beats, beats * (1.0 + 1e-9), beats * (1.0 - 1e-9)):
            if probe <= 0.0:
                continue
            assert _close(
                float(seconds_to_tau(tau_to_seconds(probe, bpm_points), bpm_points)),
                probe,
                rel=1e-12,
            )


def test_tau_to_seconds_inverts_seconds_to_tau_multi_bpm() -> None:
    """M12：反向往返（秒 -> tau -> 秒）同样无损。"""
    bpm_points = make_bpm_points(*MULTI_BPM_PAIRS)
    end_s = tau_to_seconds(MULTI_BPM_PAIRS[-1][0], bpm_points)
    seconds = np.linspace(0.0, end_s * 1.2, 257)
    taus = seconds_to_tau(seconds, bpm_points)
    back = tau_to_seconds(taus, bpm_points)
    assert np.allclose(back, seconds, rtol=1e-12, atol=0.0)


def test_tau_to_seconds_matches_analytic_segment_sum() -> None:
    """M12：与「逐段 Delta 拍 x 60 / bpm 求和」的解析值一致（含段界与段外延长）。"""
    bpm_points = make_bpm_points(*MULTI_BPM_PAIRS)
    for tau in (0.0, 1.0, 3.9, 4.0, 4.1, 7.25, 9.5, 12.0, 20.0):
        expected = _reference_tau_to_seconds(tau, MULTI_BPM_PAIRS)
        assert tau_to_seconds(tau, bpm_points) == pytest.approx(expected, rel=1e-12, abs=0.0)
    # 段外延长：最后一个 BPM 段线性外推（总函数化，不抛错）
    beyond = MULTI_BPM_PAIRS[-1][0] * 3.0
    assert tau_to_seconds(beyond, bpm_points) == pytest.approx(
        _reference_tau_to_seconds(beyond, MULTI_BPM_PAIRS),
        rel=1e-12,
    )


def test_jacobian_is_piecewise_constant_and_jumps_at_segment_borders() -> None:
    """M12：J(tau) 段内常量、段界跳变；且 J == 60 / bpm。"""
    pairs = MULTI_BPM_PAIRS
    bpm_points = make_bpm_points(*pairs)
    for index, (start, bpm) in enumerate(pairs):
        end = pairs[index + 1][0] if index + 1 < len(pairs) else start * 2.0
        inside = np.linspace(start, end, 5)[:-1] if end > start else np.asarray([start])
        for tau in inside:
            assert jacobian_at(float(tau), bpm_points) == pytest.approx(SECONDS_PER_MINUTE / bpm)
    # 段界处取**右段**（左闭右开）；紧邻左侧取左段
    for beats, bpm in pairs[1:]:
        right = jacobian_at(beats, bpm_points)
        left = jacobian_at(beats * (1.0 - 1e-9), bpm_points)
        assert right == pytest.approx(SECONDS_PER_MINUTE / bpm)
        assert right != pytest.approx(left)
    grid = FieldGrid().with_time(BEAT_SUBDIVISION * (int(pairs[-1][0]) * 2), bpm_points)
    per_cell = grid.jacobian()
    assert isinstance(per_cell, np.ndarray)
    assert len(set(np.round(per_cell, 12).tolist())) == len(pairs)


def test_rewriting_bpm_list_must_change_the_result() -> None:
    """M12：改写 BPMList 必须使结果变化（否则说明藏了硬编码 / 用的是帧率）。"""
    base = make_bpm_points(*MULTI_BPM_PAIRS)
    faster = make_bpm_points(*[(beats, bpm * 2.0) for beats, bpm in MULTI_BPM_PAIRS])
    other = make_bpm_points((0.0, 200.0), (4.0, 200.0), (9.5, 200.0))
    tau = 6.0
    base_s = tau_to_seconds(tau, base)
    assert tau_to_seconds(tau, faster) == pytest.approx(base_s / 2.0, rel=1e-12)
    assert tau_to_seconds(tau, other) != pytest.approx(base_s)
    assert seconds_to_tau(base_s, other) != pytest.approx(seconds_to_tau(base_s, base))
    # J 也必须随 BPMList 变化（不是常数、也不是帧率）
    assert jacobian_at(tau, base) != pytest.approx(jacobian_at(tau, other))
    assert jacobian_at(tau, base) == pytest.approx(SECONDS_PER_MINUTE / 180.0)


def test_tau_bins_match_total_beats() -> None:
    """M12：tau 格数与总拍数一致（T = round(tau_end * BEAT_SUBDIVISION)）。"""
    bpm_points = make_bpm_points(*MULTI_BPM_PAIRS)
    grid = FieldGrid()
    duration_s = tau_to_seconds(10.0, bpm_points)
    t_bins = grid.tau_bins_for(duration_s, bpm_points)
    assert t_bins == round(seconds_to_tau(duration_s, bpm_points) * BEAT_SUBDIVISION)
    bound = grid.with_time(t_bins, bpm_points)
    assert bound.total_beats == pytest.approx(t_bins / BEAT_SUBDIVISION, rel=0.0, abs=0.0)
    assert bound.total_seconds == pytest.approx(duration_s, rel=1e-12)
    assert len(bound.tau_edges()) == t_bins + 1
    assert len(bound.tau_centers()) == t_bins


def test_cell_volumes_are_non_uniform_and_consistent_with_volume() -> None:
    """M12/M3：dV_j = J_j * d_tau * dx 逐格不同；sum_j dV_j == volume()。"""
    bpm_points = make_bpm_points(*MULTI_BPM_PAIRS)
    t_bins = BEAT_SUBDIVISION * (int(MULTI_BPM_PAIRS[-1][0]) * 2)
    grid = FieldGrid(x_bins=FieldGrid().x_bins).with_time(t_bins, bpm_points)
    volumes = grid.cell_volumes()
    assert volumes.shape == (t_bins,)
    assert len(set(np.round(volumes, 15).tolist())) == len(MULTI_BPM_PAIRS)
    expected = float(np.sum(volumes)) * grid.x_bins * grid.sides * grid.channels
    assert grid.volume(t_bins, 1) == pytest.approx(expected, rel=1e-12)
    assert grid.volume(t_bins, 3) == pytest.approx(expected * 3.0, rel=1e-12)
    # 显式传入 jacobian 数组时与派生式同口径
    jacobian = grid.jacobian()
    assert isinstance(jacobian, np.ndarray)
    assert grid.volume(t_bins, 1, jacobian) == pytest.approx(grid.volume(t_bins, 1), rel=1e-12)


def test_left_cell_rule_is_exact_when_borders_fall_on_cell_edges() -> None:
    """变更点恰好落在格界上时，"left" 与 exact **逐格恒等**（左闭右开约定，§9-16）。

    "right" 只在段界**前一格**与 exact 不同（它把该格归给右段）——这正是
    plan §9-16「取左段 / 右段 / 加权」三种取法的差别所在，测试必须把它显式量化。
    """
    bpm_points = make_bpm_points((0.0, 120.0), (2.0, 240.0))
    t_bins = BEAT_SUBDIVISION * 4
    grid = FieldGrid().with_time(t_bins, bpm_points)
    left = grid.cell_seconds(rule="left")
    right = grid.cell_seconds(rule="right")
    exact = grid.cell_seconds(rule="exact")
    assert np.allclose(left, exact, rtol=1e-12, atol=0.0)
    assert left[0] == pytest.approx(SECONDS_PER_MINUTE / 120.0 * grid.d_tau)
    boundary_cell = int(2.0 * BEAT_SUBDIVISION) - 1
    assert right[boundary_cell] != pytest.approx(exact[boundary_cell])
    assert right[boundary_cell] == pytest.approx(SECONDS_PER_MINUTE / 240.0 * grid.d_tau)
    assert np.allclose(
        np.delete(right, boundary_cell),
        np.delete(exact, boundary_cell),
        rtol=1e-12,
        atol=0.0,
    )


def test_cell_seconds_rules_differ_only_inside_a_straddling_cell() -> None:
    """变更点落在格内时 exact 与 left 不同——量化 plan §9-16 未定的那一条。"""
    half_cell_bpm_switch = 2.0 + 0.5 / BEAT_SUBDIVISION
    bpm_points = make_bpm_points((0.0, 120.0), (half_cell_bpm_switch, 240.0))
    grid = FieldGrid().with_time(BEAT_SUBDIVISION * 4, bpm_points)
    left = grid.cell_seconds(rule="left")
    exact = grid.cell_seconds(rule="exact")
    straddling = int(2.0 * BEAT_SUBDIVISION)
    assert left[straddling] != pytest.approx(exact[straddling])
    assert np.allclose(np.delete(left, straddling), np.delete(exact, straddling), rtol=1e-12)


def test_first_bpm_point_after_zero_is_extrapolated_backwards() -> None:
    """首段起点 > 0 时向前外推（换算在全 tau 轴有定义，段列表也从 0 起）。"""
    bpm_points = make_bpm_points((4.0, 120.0))
    segments = bpm_segments(bpm_points)
    assert segments[0].tau_start == 0.0
    assert segments[0].bpm == pytest.approx(120.0)
    assert tau_to_seconds(4.0, bpm_points) == pytest.approx(4.0 * SECONDS_PER_MINUTE / 120.0)


def test_volume_uses_full_k_lines_by_definition() -> None:
    """|Omega| 按全 K 条线计（plan §2 偏离 2：空线照样有成本）。"""
    bpm_points = make_bpm_points((0.0, 120.0))
    grid = FieldGrid().with_time(BEAT_SUBDIVISION, bpm_points)
    one_line = grid.volume(grid.t_bins, 1)
    assert grid.volume(grid.t_bins, 5) == pytest.approx(one_line * 5.0, rel=1e-12)
    assert pytest.approx(grid.dx * grid.x_bins) == RPE_STAGE_WIDTH
