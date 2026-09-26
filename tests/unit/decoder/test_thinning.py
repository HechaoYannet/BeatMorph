"""M5.3：D2 Ogata thinning 的正确性 —— Plan 05 §4.2 / §6。

验收金标准：以**解析可算**的非齐次泊松过程 `lambda(t) = A + B t` 为参照，
用 KS 检验比对采样分布（`p > 0.05`，样本量 >= 1e4），并校验采样总数相对 `int lambda` 的偏差。

⚠️ **口径细化（M5.3 的"计数相对误差 < 1%"需要说清）**：单次抽样的计数是
`Poisson(Lambda)`，其相对偏差本身就是 `1/sqrt(Lambda)` 量级——在 `Lambda = 1e4` 时
恰好是 1%，即"单次抽样落在 1% 内"只有约 68% 的概率，作为断言会在 32% 的情况下误报。
因此本文件把它拆成两条**统计上成立**的断言：① 多个 seed 的均值与 `int lambda` 的
相对偏差 < 1%（检验**无系统性偏置**）；② 单次偏差 <= `4*sqrt(Lambda)`（约 4 sigma）。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from beatmorph.core.contracts.phigros import SUBDIVISIONS_PER_BEAT, NoteType, Side
from beatmorph.decoder.fieldops import total_intensity
from beatmorph.decoder.thinning import (
    ThinningBoundError,
    ThinningConfig,
    decode_thinning,
    ogata_thinning,
    sample_mark,
)
from tests.unit.decoder._builders import (
    empty_field,
    make_bpm_points,
    make_grid,
    place_gaussian,
    spec_for,
)

#: 解析强度 `lambda(t) = A + B*t`（t 单位 = 拍）。
A_RATE = 40.0
B_RATE = 4.0
T_SPAN = 64.0


def _linear(t: np.ndarray) -> np.ndarray:
    return A_RATE + B_RATE * np.asarray(t, dtype=np.float64)


def _analytic_total(t_end: float) -> float:
    """`int_0^{t_end} (A + B t) dt = A t + B t^2 / 2`。"""
    return A_RATE * t_end + 0.5 * B_RATE * t_end**2


def _analytic_partial(t: float) -> float:
    """部分积分 `int_0^t (A + B s) ds`（用于构造理论 CDF）。"""
    return A_RATE * t + 0.5 * B_RATE * t * t


def _cdf(t: float) -> float:
    """理论分布函数 `F(t) = Lambda(t) / Lambda(T)`。"""
    return _analytic_partial(t) / _analytic_total(T_SPAN)


def _ks_statistic(sorted_samples: np.ndarray, cdf) -> float:
    """单样本 KS 统计量 `D = sup|F_n - F|`（无需 scipy）。"""
    n = sorted_samples.size
    index = np.arange(1, n + 1, dtype=np.float64)
    cdf_values = np.asarray([cdf(float(value)) for value in sorted_samples], dtype=np.float64)
    upper = np.max(index / n - cdf_values)
    lower = np.max(cdf_values - (index - 1) / n)
    return float(max(upper, lower))


def _ks_p_value(d: float, n: int) -> float:
    """渐近 Kolmogorov 分布（Numerical Recipes 形式），无需 scipy。"""
    if n <= 0:
        return 1.0
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    total = 0.0
    for k in range(1, 101):
        total += 2.0 * (-1.0) ** (k - 1) * math.exp(-2.0 * k * k * lam * lam)
    return float(min(1.0, max(0.0, total)))


def _sample(seed: int, *, upper_bound: float | None = None, policy: str = "raise"):
    rng = np.random.default_rng(seed)
    return ogata_thinning(
        _linear,
        0.0,
        T_SPAN,
        upper_bound=_analytic_total(T_SPAN) / T_SPAN * 2.0 if upper_bound is None else upper_bound,
        rng=rng,
        bound_policy=policy,  # type: ignore[arg-type]
    )


def test_ks_test_against_analytic_linear_intensity() -> None:
    """M5.3 主验收：`lambda(t) = A + B t` 上 KS 检验 `p > 0.05`（n >= 1e4）。"""
    result = _sample(20260927)
    n = int(result.times.size)
    assert n >= 10_000, f"样本量必须 >= 1e4，得到 {n}"
    assert float(result.times.min()) >= 0.0
    assert float(result.times.max()) < T_SPAN
    statistic = _ks_statistic(result.times, _cdf)
    p_value = _ks_p_value(statistic, n)
    assert p_value > 0.05, f"KS 检验失败：D={statistic:.5f}, p={p_value:.4f}, n={n}"


def test_ks_helper_detects_a_wrong_distribution() -> None:
    """KS 辅助函数自检：错的分布必须被拒（避免"门禁恒绿"）。"""
    rng = np.random.default_rng(7)
    uniform = np.sort(rng.random(4000) * T_SPAN)
    assert _ks_p_value(_ks_statistic(uniform, _cdf), uniform.size) < 1e-6


def test_event_count_bias_and_dispersion_across_seeds() -> None:
    """M5.3（口径细化）：多 seed 均值相对 `int lambda` < 1%；单次 <= 4 sigma。"""
    expected = _analytic_total(T_SPAN)
    counts = np.asarray([_sample(seed).times.size for seed in range(12)], dtype=np.float64)
    relative_bias = abs(counts.mean() - expected) / expected
    assert relative_bias < 0.01, f"存在系统性偏置：均值 {counts.mean()} vs {expected}"
    sigma = math.sqrt(expected)
    assert np.all(np.abs(counts - expected) <= 4.0 * sigma), counts


def test_same_seed_reproduces_elementwise() -> None:
    """M5.3：固定 seed 后逐元素可复现。"""
    first = _sample(99).times
    second = _sample(99).times
    np.testing.assert_array_equal(first, second)
    third = _sample(100).times
    assert third.size != first.size or not np.array_equal(third, first)


def test_constant_intensity_gives_poisson_counts() -> None:
    """常数强度下事件数服从泊松（用固定 seed 的统计量断言）。"""
    rate = 1.0
    span = 25.0
    counts = []
    for seed in range(40):
        rng = np.random.default_rng(seed)
        result = ogata_thinning(
            lambda t: np.full_like(np.asarray(t, dtype=np.float64), rate),
            0.0,
            span,
            upper_bound=rate,
            rng=rng,
        )
        counts.append(result.times.size)
    counts_array = np.asarray(counts, dtype=np.float64)
    expected = rate * span
    assert abs(counts_array.mean() - expected) <= 4.0 * math.sqrt(expected / counts_array.size)
    assert 0.4 * expected <= counts_array.var() <= 2.2 * expected


def test_bound_violation_raises_instead_of_silently_wrong() -> None:
    """Plan §8：上界失效必须**报错**（不得静默给出另一个分布）。"""
    with pytest.raises(ThinningBoundError):
        _sample(3, upper_bound=A_RATE)
    result = _sample(3, upper_bound=A_RATE, policy="resample")
    assert result.times.size > 0
    assert float(result.times.max()) < T_SPAN
    assert result.n_restarts >= 1
    with pytest.raises(ValueError, match="upper_bound"):
        _sample(3, upper_bound=-1.0)


def test_zero_width_or_zero_bound_is_empty_not_an_error() -> None:
    rng = np.random.default_rng(0)
    empty = ogata_thinning(_linear, 5.0, 5.0, upper_bound=10.0, rng=rng)
    assert empty.times.size == 0
    assert empty.n_candidates == 0
    empty_bound = ogata_thinning(_linear, 0.0, 5.0, upper_bound=0.0, rng=rng)
    assert empty_bound.times.size == 0


def test_step_intensity_is_the_discrete_meanings_of_the_field() -> None:
    """场解码：采样事件数在多个 seed 上收敛到 `int lambda`（同网格同测度）。"""
    bpm = make_bpm_points((0.0, 150.0))
    grid = make_grid(bpm_points=bpm, t_bins=SUBDIVISIONS_PER_BEAT * 8)
    spec = spec_for(grid, 1)
    field = empty_field(1, spec)
    for tau_bin, x_bin, note_type, side in (
        (10, 20, NoteType.TAP, Side.FRONT),
        (40, 100, NoteType.DRAG, Side.BACK),
        (100, 64, NoteType.HOLD, Side.FRONT),
        (300, 10, NoteType.FLICK, Side.FRONT),
    ):
        place_gaussian(
            field,
            line_id=0,
            tau_bin=tau_bin,
            x_bin=x_bin,
            note_type=note_type,
            side=side,
            amplitude=100.0,
        )
    expected = float(total_intensity(field, grid).sum())
    assert expected > 50.0
    counts = []
    for seed in range(16):
        events, stats = decode_thinning(
            field,
            grid,
            spec,
            config=ThinningConfig(seed=seed),
        )
        counts.append(len(events))
        assert stats["d2_block_cells"] == float(SUBDIVISIONS_PER_BEAT)
    mean = float(np.mean(counts))
    assert abs(mean - expected) / expected < 0.05, (mean, expected)


def test_empty_lines_are_counted_and_produce_no_events() -> None:
    bpm = make_bpm_points((0.0, 150.0))
    grid = make_grid(bpm_points=bpm, t_bins=SUBDIVISIONS_PER_BEAT * 2)
    spec = spec_for(grid, 2)
    field = empty_field(2, spec)
    place_gaussian(field, line_id=1, tau_bin=10, x_bin=64, amplitude=10.0)
    events, stats = decode_thinning(field, grid, spec, config=ThinningConfig(seed=5))
    assert stats["d2_n_empty_lines"] == 1.0
    assert all(event.line_id == 1 for event in events)


def test_mark_sampling_joint_matches_the_cell_distribution() -> None:
    """joint 口径必须复现该格的归一化强度分布（条件分布的正确形式）。"""
    rng = np.random.default_rng(11)
    slab = np.zeros((4, 2, 5), dtype=np.float64)
    slab[1, 0, 2] = 3.0
    slab[3, 1, 4] = 1.0
    counts: dict[tuple[int, int, int], int] = {}
    for _ in range(20000):
        key = sample_mark(slab, rng, mode="joint")
        counts[key] = counts.get(key, 0) + 1
    assert counts[(1, 0, 2)] / 20000 == pytest.approx(0.75, abs=0.02)
    assert counts[(3, 1, 4)] / 20000 == pytest.approx(0.25, abs=0.02)
    assert len(counts) == 2


def test_factorized_mark_sampling_differs_when_marks_are_correlated() -> None:
    """对照臂：标记完全相关时，factorized 会采到 joint **永不**采到的组合。"""
    rng = np.random.default_rng(13)
    slab = np.zeros((4, 2, 5), dtype=np.float64)
    for index in range(4):
        slab[index, index % 2, index] = 1.0
    joint = {sample_mark(slab, rng, mode="joint") for _ in range(2000)}
    factorized = {sample_mark(slab, rng, mode="factorized") for _ in range(2000)}
    assert all(channel == x_bin for x_bin, _, channel in joint)
    assert any(channel != x_bin for x_bin, _, channel in factorized)
    with pytest.raises(ValueError, match="总强度为 0"):
        sample_mark(np.zeros((2, 2, 5)), rng)
    with pytest.raises(ValueError, match="mark_mode"):
        sample_mark(slab, rng, mode="nope")  # type: ignore[arg-type]


def test_config_guards() -> None:
    with pytest.raises(ValueError, match="safety"):
        ThinningConfig(safety=0.5)
    with pytest.raises(ValueError, match="block_beats"):
        ThinningConfig(block_beats=0.0)
    with pytest.raises(ValueError, match="max_restarts"):
        ThinningConfig(max_restarts=-1)
