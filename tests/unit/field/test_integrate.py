"""M3：两条积分路径对拍 + 逐格 dV_j 加权门禁（默认 CI，无权重无 GPU）。

四类测试场（plan §3.4）：(i) 闭式可积（常数 / 线性 / 指数）；(ii) 随机分段常量；
(iii) 因子化（p 与 Lambda' 分离构造）；(iv) **多 BPM 段**（J(tau) 分段常量、dV_j 非均匀）。

容差是本计划设定的工程容差（plan §9-5）：1e-6（float64）/ 1e-4（float32）。
**失败模式即门禁价值**：漏乘 J_j、或掩码后忘记重新归一化 p，本文件必须当场捕获。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pytest
import torch

from beatmorph.core.contracts.phigros import RPE_STAGE_WIDTH
from beatmorph.field.grid import BEAT_SUBDIVISION, SECONDS_PER_MINUTE, FieldGrid
from beatmorph.field.integrate import (
    cell_volumes_tensor,
    cumulative_from_lam,
    delta_cumulative,
    factorized_lambda,
    integrate_cumulative,
    integrate_grid,
    lam_from_cumulative,
    normalize_cell_prob,
    omega,
    relative_error,
    uniform_cell_prob,
)
from tests.unit.field._builders import make_bpm_points

#: 多 BPM 段（含升速与降速）；段界刻意落在格界与非格界两种情形上
MULTI_BPM_PAIRS: tuple[tuple[float, float], ...] = ((0.0, 120.0), (2.0, 180.0), (3.0, 90.0))
#: 单 BPM（用于与解析式对拍）
SINGLE_BPM_PAIRS: tuple[tuple[float, float], ...] = ((0.0, 120.0),)
T_BINS = BEAT_SUBDIVISION * 4
X_BINS = 8
K_LINES = 2
#: 工程容差（plan §9-5）
TOL_F64 = 1e-6
TOL_F32 = 1e-4


def _reference_cell_seconds(pairs: Sequence[tuple[float, float]], t_bins: int) -> np.ndarray:
    """**独立**测度实现：逐格取格左边界所在段的 60 / bpm，乘 d_tau。

    只依赖 (拍, BPM) 序列与「分钟 = 60 秒」这一单位定义，不调用被测模块。
    """
    d_tau = 1.0 / BEAT_SUBDIVISION
    assert pairs[0][0] == 0.0, "参考实现要求首段从 0 拍起"
    out = np.empty(t_bins, dtype=np.float64)
    for index in range(t_bins):
        tau = index * d_tau
        out[index] = math.nan
        for position, (start, bpm) in enumerate(pairs):
            end = pairs[position + 1][0] if position + 1 < len(pairs) else math.inf
            if start <= tau < end:
                out[index] = SECONDS_PER_MINUTE / bpm * d_tau
                break
    assert not np.isnan(out).any()
    return out


def _grid(
    pairs: Sequence[tuple[float, float]] = MULTI_BPM_PAIRS, *, t_bins: int = T_BINS
) -> FieldGrid:
    grid = FieldGrid(x_bins=X_BINS).with_time(t_bins, make_bpm_points(*pairs))
    grid.assert_grid()
    return grid


def _random_lambda(grid: FieldGrid, *, seed: int, k: int = K_LINES) -> torch.Tensor:
    """随机正场（含空格，模拟稀疏目标）。"""
    generator = torch.Generator().manual_seed(seed)
    shape = (k, grid.t_bins, grid.x_bins, grid.sides, grid.channels)
    raw = torch.rand(shape, generator=generator, dtype=torch.float64)
    return torch.where(raw > 0.3, raw, torch.zeros_like(raw))


#: 闭式 Lambda 的系数（测试自定，与被测实现无关）
ANALYTIC_COEFFS: dict[str, tuple[float, float]] = {
    "constant": (3.0, 0.0),
    "linear": (0.5, 2.0),
    "exp": (4.0, 0.7),
}


def _analytic_value(kind: str, tau: float) -> float:
    """闭式 Lambda(tau) 的**标量**解析值（用于外部对拍）。"""
    scale, rate = ANALYTIC_COEFFS[kind]
    if kind == "constant":
        return scale * tau
    if kind == "linear":
        return scale * tau * tau + rate * tau
    return scale * (1.0 - math.exp(-rate * tau)) / rate


def _analytic_cum(grid: FieldGrid, kind: str, *, k: int = K_LINES) -> torch.Tensor:
    """闭式 Lambda(tau) 在每格**右边界**的取值；Lambda(0) = 0。"""
    edges = torch.as_tensor(grid.tau_edges()[1:], dtype=torch.float64)
    scale, rate = ANALYTIC_COEFFS[kind]
    if kind == "constant":
        profile = scale * edges
    elif kind == "linear":
        profile = scale * edges**2 + rate * edges
    else:
        profile = scale * (1.0 - torch.exp(-rate * edges)) / rate
    return profile.reshape(1, -1).expand(k, -1).contiguous()


# ══════════════════════════════════════════════════════════════
# 逐格 dV_j 加权（M3 的硬要求）
# ══════════════════════════════════════════════════════════════


def test_cell_volumes_equal_jacobian_times_d_tau_times_dx() -> None:
    """dV_j = J_j * d_tau * dx，且逐格与独立实现的 J 一致（非均匀网格）。"""
    grid = _grid()
    volumes = cell_volumes_tensor(grid, dtype=torch.float64).numpy()
    reference = _reference_cell_seconds(MULTI_BPM_PAIRS, T_BINS) * grid.dx
    assert np.allclose(volumes, reference, rtol=TOL_F64, atol=0.0)
    assert len(set(np.round(volumes, 15).tolist())) == len(MULTI_BPM_PAIRS)


def test_integrate_grid_matches_independent_weighted_sum() -> None:
    """路径 (a) 必须等于**独立**算出的 sum_j lambda_j * J_j * d_tau * dx。"""
    grid = _grid()
    lam = _random_lambda(grid, seed=7)
    volumes = torch.as_tensor(
        _reference_cell_seconds(MULTI_BPM_PAIRS, T_BINS) * grid.dx,
        dtype=torch.float64,
    ).reshape(1, -1, 1, 1, 1)
    expected = (lam * volumes).sum(dim=(1, 2, 3, 4))
    assert torch.allclose(integrate_grid(lam, grid), expected, rtol=TOL_F64, atol=0.0)


def test_missing_jacobian_factor_must_fail_the_gate() -> None:
    """漏乘 J_j（把「拍」当「秒」）= 单位错配，必须被抓到。"""
    grid = _grid()
    lam = _random_lambda(grid, seed=11)
    correct = integrate_grid(lam, grid)
    wrong = (lam * (grid.d_tau * grid.dx)).sum(dim=(1, 2, 3, 4))
    error = relative_error(correct, wrong).max().item()
    assert error > 1e-2, "漏乘 J_j 竟然通过了门禁——测度断言无效"
    assert not torch.allclose(correct, wrong, rtol=TOL_F64, atol=0.0)


def test_constant_lambda_integrates_to_lambda_times_omega() -> None:
    """lambda == 常数时两条路径都给出 lambda * sum_j dV_j（plan §3.4 明文要求）。"""
    grid = _grid()
    value = 0.75
    lam = torch.full(
        (K_LINES, grid.t_bins, grid.x_bins, grid.sides, grid.channels),
        value,
        dtype=torch.float64,
    )
    total_volume = omega(grid, n_lines=K_LINES)
    expected_volume = (
        float(np.sum(_reference_cell_seconds(MULTI_BPM_PAIRS, T_BINS)))
        * grid.dx
        * grid.x_bins
        * grid.sides
        * grid.channels
        * K_LINES
    )
    assert total_volume == pytest.approx(expected_volume, rel=TOL_F64)
    # integrate_grid 是**逐线**量：单线分得的体积是 |Omega| / K（Ω 覆盖全部 K 条线）
    per_line_volume = omega(grid, n_lines=1)
    assert total_volume == pytest.approx(per_line_volume * K_LINES, rel=TOL_F64)
    expected = torch.full((K_LINES,), value * per_line_volume, dtype=torch.float64)
    assert torch.allclose(integrate_grid(lam, grid), expected, rtol=TOL_F64, atol=0.0)
    cum = cumulative_from_lam(lam, grid)
    assert torch.allclose(integrate_cumulative(cum), expected, rtol=TOL_F64, atol=0.0)


def test_omega_counts_all_k_lines_including_empty_ones() -> None:
    """|Omega| 覆盖全部 K 条线（空线照罚，plan §2 偏离 2）。"""
    grid = _grid()
    single = omega(grid, n_lines=1)
    assert omega(grid, n_lines=5) == pytest.approx(single * 5.0, rel=TOL_F64)
    assert omega(grid, line_mask=[True, False, False, False, False]) == pytest.approx(single)


# ══════════════════════════════════════════════════════════════
# 两条路径对拍（四类测试场）
# ══════════════════════════════════════════════════════════════


@pytest.mark.parametrize("kind", ["constant", "linear", "exp"])
@pytest.mark.parametrize("mode", ["finite_diff", "autograd"])
def test_two_paths_agree_on_closed_form_fields(kind: str, mode: str) -> None:
    """(i) 闭式可积场：lambda 由闭式 Lambda 求导得到，两条路径互相一致。"""
    grid = _grid()
    cum = _analytic_cum(grid, kind)
    lam = lam_from_cumulative(cum, grid, mode=mode)
    path_a = integrate_grid(lam, grid)
    path_b = integrate_cumulative(cum)
    error = relative_error(path_a, path_b).max().item()
    assert error <= TOL_F64, f"{kind}/{mode} 两条路径分歧：{error}"
    assert torch.allclose(path_a, path_b, rtol=TOL_F64, atol=0.0)
    # 与闭式 Lambda(tau_end) 的**解析值**对拍（外部实现，不是 cum 自身）
    expected = _analytic_value(kind, grid.total_beats)
    assert path_b[0].item() == pytest.approx(expected, rel=TOL_F64)


def test_two_paths_agree_on_random_piecewise_constant_field() -> None:
    """(ii) 随机分段常量场。"""
    grid = _grid()
    generator = torch.Generator().manual_seed(23)
    cum = torch.rand((K_LINES, grid.t_bins), generator=generator, dtype=torch.float64).cumsum(dim=1)
    lam = lam_from_cumulative(cum, grid, mode="finite_diff")
    assert (
        relative_error(integrate_grid(lam, grid), integrate_cumulative(cum)).max().item() <= TOL_F64
    )


def test_two_paths_agree_on_factorized_field() -> None:
    """(iii) 因子化场：lambda = Delta Lambda * p / dV，sum p = 1。"""
    grid = _grid()
    generator = torch.Generator().manual_seed(31)
    cum = torch.rand((K_LINES, grid.t_bins), generator=generator, dtype=torch.float64).cumsum(dim=1)
    delta = delta_cumulative(cum, grid, mode="finite_diff")
    raw = torch.rand(
        (K_LINES, grid.t_bins, grid.x_bins, grid.sides, grid.channels),
        generator=generator,
        dtype=torch.float64,
    )
    prob = normalize_cell_prob(raw)
    assert torch.allclose(
        prob.sum(dim=(2, 3, 4)),
        torch.ones((K_LINES, grid.t_bins), dtype=torch.float64),
        rtol=TOL_F64,
        atol=0.0,
    )
    lam = factorized_lambda(delta, prob, grid)
    assert (
        relative_error(integrate_grid(lam, grid), integrate_cumulative(cum)).max().item() <= TOL_F64
    )


def test_two_paths_agree_on_multi_bpm_field() -> None:
    """(iv) 多 BPM 段场：dV_j 非均匀时两条路径仍一致（Q15 的核心风险点）。"""
    grid = _grid()
    cum = _analytic_cum(grid, "linear")
    lam = lam_from_cumulative(cum, grid, mode="autograd")
    assert (
        relative_error(integrate_grid(lam, grid), integrate_cumulative(cum)).max().item() <= TOL_F64
    )
    single_bpm = _grid(SINGLE_BPM_PAIRS)
    assert not np.allclose(
        cell_volumes_tensor(grid).numpy(),
        cell_volumes_tensor(single_bpm).numpy(),
    )


def test_factorization_identity_holds_cell_by_cell() -> None:
    """因子化恒等：逐格 sum_{x,s,c} lambda dV == Delta Lambda（不是只对总和对）。"""
    grid = _grid()
    cum = _analytic_cum(grid, "exp")
    delta = delta_cumulative(cum, grid, mode="finite_diff")
    prob = uniform_cell_prob(grid, ref=cum)
    lam = factorized_lambda(delta, prob, grid)
    volumes = cell_volumes_tensor(grid, dtype=torch.float64).reshape(1, -1, 1, 1, 1)
    per_cell = (lam * volumes).sum(dim=(2, 3, 4))
    assert torch.allclose(per_cell, delta, rtol=TOL_F64, atol=0.0)


# ══════════════════════════════════════════════════════════════
# 反例：p 未重新归一化必须失败（门禁判别力的来源）
# ══════════════════════════════════════════════════════════════


def test_unnormalized_prob_after_masking_must_be_caught() -> None:
    """反例：掩掉一整条 (s, c) 切片却忘记重新归一化 => 两条路径必然分歧。"""
    grid = _grid()
    cum = _analytic_cum(grid, "linear")
    delta = delta_cumulative(cum, grid, mode="finite_diff")
    prob = uniform_cell_prob(grid, ref=cum).expand(
        K_LINES,
        grid.t_bins,
        grid.x_bins,
        grid.sides,
        grid.channels,
    )
    valid = torch.ones_like(prob, dtype=torch.bool)
    valid[:, :, :, 0, 0] = False  # 掩掉正面 tap 通道（域外裁剪的等价物）
    masked_only = torch.where(valid, prob, torch.zeros_like(prob))
    bad = factorized_lambda(delta, masked_only, grid)
    good = factorized_lambda(delta, normalize_cell_prob(prob, valid_mask=valid), grid)
    bad_error = relative_error(integrate_grid(bad, grid), integrate_cumulative(cum)).max().item()
    good_error = relative_error(integrate_grid(good, grid), integrate_cumulative(cum)).max().item()
    assert bad_error > TOL_F64, "反例没有失败——门禁判别力不足"
    assert bad_error == pytest.approx(1.0 / (grid.sides * grid.channels), rel=1e-9)
    assert good_error <= TOL_F64


# ══════════════════════════════════════════════════════════════
# Lambda 单调性、两种求导实现互测、dtype 容差、line_mask
# ══════════════════════════════════════════════════════════════


def test_cumulative_lambda_is_monotone_non_decreasing() -> None:
    """Lambda 必须单调不减（lambda >= 0，dV > 0）。"""
    grid = _grid()
    lam = _random_lambda(grid, seed=41)
    cum = cumulative_from_lam(lam, grid)
    assert bool((cum.diff(dim=1) >= 0).all())
    assert bool((cum[:, 0] >= 0).all())
    zero = cumulative_from_lam(torch.zeros_like(lam), grid)
    assert bool((zero == 0).all())


def test_finite_diff_and_autograd_agree() -> None:
    """两种求导实现互测（M3 明文要求）。"""
    grid = _grid()
    cum = _analytic_cum(grid, "exp")
    finite = delta_cumulative(cum, grid, mode="finite_diff")
    auto = delta_cumulative(cum, grid, mode="autograd")
    assert relative_error(finite, auto).max().item() <= TOL_F64
    assert torch.allclose(finite, auto, rtol=TOL_F64, atol=0.0)


def test_autograd_path_is_differentiable_and_finite() -> None:
    """autograd 路径必须对 Lambda 可微（梯度有限）。"""
    grid = _grid()
    cum = _analytic_cum(grid, "constant").clone().requires_grad_(True)
    lam = lam_from_cumulative(cum, grid, mode="autograd")
    integrate_grid(lam, grid).sum().backward()
    assert cum.grad is not None
    assert bool(torch.isfinite(cum.grad).all())


def test_float32_tolerance_is_respected() -> None:
    """float32 下两条路径相对误差 <= 1e-4（plan §9-5 的 dtype 分档）。"""
    grid = _grid()
    cum = _analytic_cum(grid, "linear").to(torch.float32)
    lam = lam_from_cumulative(cum, grid, mode="finite_diff")
    error = relative_error(integrate_grid(lam, grid), integrate_cumulative(cum)).max().item()
    assert error <= TOL_F32, error


def test_line_mask_zeroes_contribution_and_gradient() -> None:
    """line_mask 置 False 的线贡献与梯度**恰为 0**（padding 必须排除）。"""
    grid = _grid()
    lam = _random_lambda(grid, seed=53).requires_grad_(True)
    mask = torch.tensor([True, False])
    per_line = integrate_grid(lam, grid, line_mask=mask)
    assert per_line[1].item() == 0.0
    per_line.sum().backward()
    assert lam.grad is not None
    assert float(lam.grad[1].abs().sum().item()) == 0.0
    assert float(lam.grad[0].abs().sum().item()) > 0.0
    assert integrate_grid(lam, grid, line_mask=torch.zeros(2, dtype=torch.bool)).sum().item() == 0.0
    cum = cumulative_from_lam(lam, grid, line_mask=mask)
    assert float(cum[1].abs().sum().item()) == 0.0


def test_relative_error_is_zero_for_identical_values() -> None:
    """相对误差辅助函数的边界行为（0 vs 0 定义为 0，不做 eps 平滑）。"""
    zeros = torch.zeros(3, dtype=torch.float64)
    assert float(relative_error(zeros, zeros).sum().item()) == 0.0
    one, zero = torch.ones(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64)
    assert float(relative_error(one, zero).item()) == 1.0
    assert float(relative_error(zero, one).item()) == 1.0


def test_dx_times_bins_still_matches_contract() -> None:
    """网格口径未漂移（与 M1 交叉核对）。"""
    grid = _grid()
    assert grid.dx * grid.x_bins == RPE_STAGE_WIDTH
