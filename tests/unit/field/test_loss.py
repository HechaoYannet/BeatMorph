"""M4：泊松 NLL 四条恒等式；M5：强不平衡 / 线集中诊断且**不含任何重加权**。

默认 CI，无权重 / 无 GPU。容差：1e-6（float64，plan §9-5）。
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Sequence

import numpy as np
import pytest
import torch

from beatmorph.field.grid import BEAT_SUBDIVISION, SECONDS_PER_MINUTE, FieldGrid
from beatmorph.field.integrate import cell_volumes_tensor, omega, relative_error
from beatmorph.field.loss import (
    assert_lambda_valid,
    binned_point_gap,
    binned_poisson_nll,
    constant_baseline_lambda,
    constant_baseline_nll,
    constant_lambda_field,
    nll_decomposition,
    normalized_line_entropy,
    poisson_nll,
    softplus_lambda,
)
from tests.unit.field._builders import make_bpm_points

MULTI_BPM_PAIRS: tuple[tuple[float, float], ...] = ((0.0, 120.0), (2.0, 180.0))
SINGLE_BPM_PAIRS: tuple[tuple[float, float], ...] = ((0.0, 120.0),)
T_BINS = BEAT_SUBDIVISION * 4
X_BINS = 16
K_LINES = 3
TOL = 1e-6
#: λ 各格相同的测试常数
LAMBDA_VALUE = 0.75


def _grid(
    *,
    pairs: Sequence[tuple[float, float]] = MULTI_BPM_PAIRS,
    t_bins: int = T_BINS,
) -> FieldGrid:
    """默认网格跨两个 BPM 段 => dV_j **非均匀**（M4 必须在非均匀测度下成立）。"""
    grid = FieldGrid(x_bins=X_BINS).with_time(t_bins, make_bpm_points(*pairs))
    grid.assert_grid()
    return grid


def _shape(grid: FieldGrid, k: int = K_LINES) -> tuple[int, ...]:
    return (k, grid.t_bins, grid.x_bins, grid.sides, grid.channels)


def _place(counts: torch.Tensor, *, k: int, side: int, channel: int, n: int) -> None:
    """把 n 个事件确定性摊到 (t, x) 平面上（含 n_j >= 2 的格子）。"""
    flat = counts[k, :, :, side, channel].reshape(-1)
    quotient, remainder = divmod(n, int(flat.numel()))
    flat += float(quotient)
    flat[:remainder] += 1.0


def _counts(grid: FieldGrid, *, total: int = 120) -> torch.Tensor:
    """构造一张有事件、且覆盖多 (s, c) 的计数张量。"""
    counts = torch.zeros(_shape(grid), dtype=torch.float64)
    per_slab = total // (grid.sides * grid.channels)
    for side in range(grid.sides):
        for channel in range(grid.channels):
            _place(counts, k=0, side=side, channel=channel, n=per_slab)
    return counts


# ══════════════════════════════════════════════════════════════
# M4 (i)：常数基线的闭式
# ══════════════════════════════════════════════════════════════


def test_constant_baseline_matches_closed_form() -> None:
    """M4(i)：lambda == N/|Omega| 时 NLL == N * (1 + log(|Omega| / N))，相对误差 <= 1e-6。"""
    grid = _grid()
    counts = _counts(grid)
    n_events = float(counts.sum().item())
    total_volume = omega(grid, n_lines=K_LINES)
    lam = constant_lambda_field(
        grid,
        n_events,
        omega_value=total_volume,
        n_lines=K_LINES,
    )
    value = poisson_nll(counts, lam, grid).item()
    expected = constant_baseline_nll(n_events, total_volume)
    assert value == pytest.approx(expected, rel=TOL)
    assert constant_baseline_lambda(n_events, total_volume) == pytest.approx(
        n_events / total_volume,
        rel=TOL,
    )


def test_closed_form_baseline_is_the_numerical_optimum() -> None:
    """M4(i)：闭式基线必须与数值最小化结果一致（否则闭式写错了）。"""
    grid = _grid()
    counts = _counts(grid)
    n_events = float(counts.sum().item())
    total_volume = omega(grid, n_lines=K_LINES)
    shape = _shape(grid)
    volumes = cell_volumes_tensor(grid, dtype=torch.float64).reshape(1, -1, 1, 1, 1)

    def loss_at(value: float) -> float:
        lam = torch.full(shape, value, dtype=torch.float64)
        return float(poisson_nll(counts, lam, grid).item())

    optimum = constant_baseline_lambda(n_events, total_volume)
    coarse = np.geomspace(optimum / 4.0, optimum * 4.0, 1201)
    best = min(coarse, key=loss_at)
    fine = np.linspace(best * 0.99, best * 1.01, 401)
    best = min(fine, key=loss_at)
    assert best == pytest.approx(optimum, rel=1e-5)
    assert loss_at(best) == pytest.approx(constant_baseline_nll(n_events, total_volume), rel=TOL)
    assert float((torch.full(shape, best, dtype=torch.float64) * volumes).sum().item()) == (
        pytest.approx(n_events, rel=1e-4)
    )


# ══════════════════════════════════════════════════════════════
# M4 (ii)：lambda == 0 必须非有限（禁止 eps 平滑）
# ══════════════════════════════════════════════════════════════


def test_zero_lambda_gives_non_finite_nll() -> None:
    """M4(ii)：lambda == 0 且存在事件时 NLL 必须为 +inf——**禁止 eps 平滑**。"""
    grid = _grid()
    counts = _counts(grid)
    lam = torch.zeros(_shape(grid), dtype=torch.float64)
    value = poisson_nll(counts, lam, grid)
    assert not torch.isfinite(value)
    assert value.item() == math.inf
    assert not torch.isfinite(binned_poisson_nll(counts, lam, grid))


def test_zero_lambda_with_zero_events_is_finite_zero() -> None:
    """N = 0 且 lambda == 0 时 NLL = 0（数学极限 0 * log 0 = 0，不是 NaN）。"""
    grid = _grid()
    counts = torch.zeros(_shape(grid), dtype=torch.float64)
    lam = torch.zeros(_shape(grid), dtype=torch.float64)
    value = poisson_nll(counts, lam, grid)
    assert torch.isfinite(value)
    assert value.item() == 0.0


def test_no_epsilon_smoothing_parameter_exists() -> None:
    """M4(ii) 的代码审查项：NLL 入口不得有 eps / 平滑参数。"""
    for function in (poisson_nll, binned_poisson_nll):
        parameters = set(inspect.signature(function).parameters)
        assert parameters == {"counts", "lam", "grid", "line_mask", "reduction"}


# ══════════════════════════════════════════════════════════════
# M4 (iii)：桶内计数语义（n_j = 2 贡献 2 log lambda）
# ══════════════════════════════════════════════════════════════


def test_duplicate_events_contribute_two_log_terms() -> None:
    """M4(iii)：n_j = 2 的格子贡献 2 * log lambda_j（**禁止去重**）。"""
    grid = _grid()
    counts = torch.zeros(_shape(grid), dtype=torch.float64)
    counts[0, 0, 0, 0, 0] = 2.0
    counts[0, 1, 1, 0, 0] = 1.0
    lam = torch.full(_shape(grid), LAMBDA_VALUE, dtype=torch.float64)
    total_volume = omega(grid, n_lines=K_LINES)
    expected = -(2.0 * math.log(LAMBDA_VALUE) + math.log(LAMBDA_VALUE)) + (
        LAMBDA_VALUE * total_volume
    )
    assert poisson_nll(counts, lam, grid).item() == pytest.approx(expected, rel=TOL)
    # 去重版本（把两格都当成 1 个事件）必须给出不同的值——证明我们没有去重
    dedup = torch.zeros_like(counts)
    dedup[0, 0, 0, 0, 0] = 1.0
    dedup[0, 1, 1, 0, 0] = 1.0
    assert poisson_nll(dedup, lam, grid).item() != pytest.approx(expected, rel=TOL)
    assert poisson_nll(counts, lam, grid).item() == pytest.approx(
        poisson_nll(dedup, lam, grid).item() - math.log(LAMBDA_VALUE),
        rel=TOL,
    )


def test_events_only_enter_through_log_lambda() -> None:
    """事件项 = -sum n_j log lambda_j（逐格，系数恒为 1）。"""
    grid = _grid()
    counts = _counts(grid)
    lam = torch.full(_shape(grid), LAMBDA_VALUE, dtype=torch.float64)
    total_volume = omega(grid, n_lines=K_LINES)
    expected = -float(counts.sum().item()) * math.log(LAMBDA_VALUE) + LAMBDA_VALUE * total_volume
    assert poisson_nll(counts, lam, grid).item() == pytest.approx(expected, rel=TOL)


# ══════════════════════════════════════════════════════════════
# M4 (iv)：binned - point == sum log(n!) - sum n_j log dV_j
# ══════════════════════════════════════════════════════════════


def test_binned_minus_point_equals_parameter_free_constant() -> None:
    """M4(iv)：差值只含与参数无关的常数项（相对误差 <= 1e-6）。"""
    grid = _grid()
    counts = _counts(grid)
    lam = torch.full(_shape(grid), LAMBDA_VALUE, dtype=torch.float64)
    point = poisson_nll(counts, lam, grid)
    binned = binned_poisson_nll(counts, lam, grid)
    gap = (binned - point).item()
    closed_form = binned_point_gap(counts, lam, grid).item()
    assert gap == pytest.approx(closed_form, rel=TOL)
    volumes = cell_volumes_tensor(grid, dtype=torch.float64).reshape(1, -1, 1, 1, 1)
    manual = float(torch.lgamma(counts + 1.0).sum().item()) - float(
        (counts * torch.log(volumes)).sum().item(),
    )
    assert gap == pytest.approx(manual, rel=TOL)
    # 与参数无关：换一个 lambda，差值不变
    other = torch.full(_shape(grid), LAMBDA_VALUE * 3.0, dtype=torch.float64)
    assert (binned_poisson_nll(counts, other, grid) - poisson_nll(counts, other, grid)).item() == (
        pytest.approx(gap, rel=TOL)
    )


def test_uniform_grid_degenerates_to_single_delta_v() -> None:
    """均匀网格下退化式为 -N log dV（tau 网格非均匀时**不得**再用这个写法）。"""
    grid = _grid(pairs=SINGLE_BPM_PAIRS)
    volumes = grid.cell_volumes()
    assert len(set(np.round(volumes, 15).tolist())) == 1
    counts = _counts(grid)
    lam = torch.full(_shape(grid), LAMBDA_VALUE, dtype=torch.float64)
    gap = (binned_poisson_nll(counts, lam, grid) - poisson_nll(counts, lam, grid)).item()
    delta_v = float(volumes[0])
    n_events = float(counts.sum().item())
    expected = float(torch.lgamma(counts + 1.0).sum().item()) - n_events * math.log(delta_v)
    assert gap == pytest.approx(expected, rel=TOL)


# ══════════════════════════════════════════════════════════════
# line_mask 语义（§4.4）
# ══════════════════════════════════════════════════════════════


def test_line_mask_zeroes_contribution_and_gradient() -> None:
    """line_mask 置 False 的线对 NLL 的贡献与其梯度**恰为 0**。"""
    grid = _grid()
    counts = _counts(grid)
    lam = torch.full(_shape(grid), LAMBDA_VALUE, dtype=torch.float64).requires_grad_(True)
    mask = torch.tensor([True, True, False])
    masked = poisson_nll(counts, lam, grid, line_mask=mask)
    masked.backward()
    assert lam.grad is not None
    assert float(lam.grad[2].abs().sum().item()) == 0.0
    assert float(lam.grad[0].abs().sum().item()) > 0.0
    per_line = poisson_nll(counts, lam, grid, line_mask=mask, reduction="none")
    assert per_line[2].item() == 0.0
    assert (
        poisson_nll(
            counts,
            lam,
            grid,
            line_mask=torch.zeros(K_LINES, dtype=torch.bool),
        ).item()
        == 0.0
    )


def test_reduction_mean_uses_active_lines_only() -> None:
    """mean 归约按有效线数（padding 线不得进分母）。"""
    grid = _grid()
    counts = _counts(grid)
    lam = torch.full(_shape(grid), LAMBDA_VALUE, dtype=torch.float64)
    mask = torch.tensor([True, True, False])
    total = poisson_nll(counts, lam, grid, line_mask=mask)
    mean = poisson_nll(counts, lam, grid, line_mask=mask, reduction="mean")
    assert mean.item() == pytest.approx(total.item() / 2.0, rel=TOL)


def test_lambda_validity_check() -> None:
    """R-03-5：NaN / Inf / 负值必须在训练步被硬检查捕获。"""
    assert_lambda_valid(torch.zeros(3))
    with pytest.raises(AssertionError):
        assert_lambda_valid(torch.tensor([float("nan")]))
    with pytest.raises(AssertionError):
        assert_lambda_valid(torch.tensor([float("inf")]))
    with pytest.raises(AssertionError):
        assert_lambda_valid(torch.tensor([-1.0]))
    assert bool((softplus_lambda(torch.tensor([-3.0])) > 0).all())


# ══════════════════════════════════════════════════════════════
# M5：强不平衡 / 线集中诊断（**不做任何重加权**）
# ══════════════════════════════════════════════════════════════


def test_loss_has_no_channel_side_or_line_weights() -> None:
    """M5：损失签名里**没有任何**权重参数（5 通道 x 2 侧 x K 线的系数恒为 1）。"""
    for function in (poisson_nll, binned_poisson_nll):
        parameters = inspect.signature(function).parameters
        kinds = {parameter.kind for parameter in parameters.values()}
        assert inspect.Parameter.VAR_KEYWORD not in kinds
        assert inspect.Parameter.VAR_POSITIONAL not in kinds
        assert "weight" not in " ".join(parameters).lower()


def test_scaling_one_slice_changes_nll_exactly_by_its_analytic_delta() -> None:
    """M5：把某个 (s, c) 切片的 lambda 乘以 2，NLL 的变化必须**恰等于**该切片的解析变化。

    这证明每个格子的系数恒为 1（不存在隐藏的通道 / 侧别权重）。
    """
    grid = _grid()
    counts = _counts(grid)
    lam = torch.full(_shape(grid), LAMBDA_VALUE, dtype=torch.float64)
    side, channel = 1, 2
    scaled = lam.clone()
    scaled[:, :, :, side, channel] *= 2.0
    delta = (poisson_nll(counts, scaled, grid) - poisson_nll(counts, lam, grid)).item()
    # 切片是 3 维 (K, T, X)，dV 必须同步降到 (1, T, 1)，否则广播会错位
    volumes = cell_volumes_tensor(grid, dtype=torch.float64).reshape(1, -1, 1)
    events_in_slice = float(counts[:, :, :, side, channel].sum().item())
    integral_delta = float((lam[:, :, :, side, channel] * volumes).sum().item())
    expected = -events_in_slice * math.log(2.0) + integral_delta
    assert delta == pytest.approx(expected, rel=TOL)


def test_decomposition_parts_sum_to_total() -> None:
    """M5：per-line / per-side / per-channel 分解之和恒等于 total。"""
    grid = _grid()
    counts = _counts(grid)
    lam = torch.full(_shape(grid), LAMBDA_VALUE, dtype=torch.float64)
    mask = torch.tensor([True, True, True])
    decomposed = nll_decomposition(counts, lam, grid, line_mask=mask)
    total = poisson_nll(counts, lam, grid, line_mask=mask).item()
    assert decomposed.total == pytest.approx(total, rel=TOL)
    assert sum(decomposed.per_line) == pytest.approx(total, rel=TOL)
    assert sum(decomposed.per_side) == pytest.approx(total, rel=TOL)
    assert sum(decomposed.per_channel) == pytest.approx(total, rel=TOL)
    assert decomposed.event_term + decomposed.integral_term == pytest.approx(total, rel=TOL)
    assert decomposed.n_events == int(counts.sum().item())
    assert decomposed.omega == pytest.approx(omega(grid, line_mask=mask), rel=TOL)


def test_decomposition_reports_realistic_imbalance_shares() -> None:
    """M5：背面（survey §7.3 实测 2.4-3.0%）与 Flick（6-7%）的占比必须被如实报出。"""
    grid = _grid()
    counts = torch.zeros(_shape(grid), dtype=torch.float64)
    # 实测分布（survey §7.3）：背面 2.4-3.0%，Flick 6-7%
    total = 2000
    back_events = int(total * 0.027)
    flick_events = int(total * 0.065)
    front_events = total - back_events - flick_events
    _place(counts, k=0, side=1, channel=0, n=back_events)  # 背面全部落在 side = 1
    _place(counts, k=0, side=0, channel=4, n=flick_events)  # Flick 通道
    _place(counts, k=0, side=0, channel=0, n=front_events)  # 正面 Tap
    assert int(counts.sum().item()) == total
    lam = torch.full(_shape(grid), LAMBDA_VALUE, dtype=torch.float64)
    decomposed = nll_decomposition(counts, lam, grid)
    reported_back = decomposed.per_side_counts[1] / sum(decomposed.per_side_counts)
    reported_flick = decomposed.per_channel_counts[4] / sum(decomposed.per_channel_counts)
    assert 0.024 <= reported_back <= 0.030, reported_back
    assert 0.06 <= reported_flick <= 0.07, reported_flick
    assert 0.0 <= decomposed.line_entropy <= 1.0
    assert "per-side" in decomposed.format()


def test_normalized_line_entropy_bounds() -> None:
    """归一化线熵 ∈ [0, 1]：完全集中 -> 0，完全均匀 -> 1。"""
    concentrated = [10.0, 0.0, 0.0, 0.0]
    uniform = [5.0, 5.0, 5.0, 5.0]
    assert normalized_line_entropy(concentrated) == pytest.approx(0.0, abs=1e-12)
    assert normalized_line_entropy(uniform) == pytest.approx(1.0, rel=TOL)
    assert normalized_line_entropy([1.0]) == 0.0
    assert normalized_line_entropy([0.0, 0.0]) == 0.0
    assert normalized_line_entropy(uniform, line_mask=[True, True, False, False]) == (
        pytest.approx(1.0, rel=TOL)
    )


def test_relative_error_helper_is_not_scale_blind() -> None:
    """辅助断言：相对误差在量级差异下仍可判分歧（本文件多处依赖它）。"""
    a = torch.tensor([1.0, 1e-8], dtype=torch.float64)
    b = torch.tensor([1.0, 2e-8], dtype=torch.float64)
    assert float(relative_error(a, b).max().item()) == pytest.approx(0.5, rel=1e-6)


def test_seconds_per_minute_is_the_only_time_unit_definition() -> None:
    """测度口径与时间单位的一致性（防止有人偷偷用 1000 之类换算）。"""
    assert pytest.approx(1.0 / (1.0 / 60.0)) == SECONDS_PER_MINUTE
    grid = _grid()
    bpms: Sequence[float] = [point.bpm for point in grid.bpm_points]
    assert all(bpm > 0.0 for bpm in bpms)
