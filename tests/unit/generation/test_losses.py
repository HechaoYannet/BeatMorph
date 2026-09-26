"""M4：遮盖重标定契约、泊松 NLL 的遮盖口径、排列敏感性与「无 line 分类损失」。

默认 CI，无权重 / 无 GPU。核心推导（plan 04 §4.3，本模块发现的分歧）：

    只监督被遮盖事件（占比 r）时，未重标定损失的最优 lambda 被系统性缩放 r 倍；
    正确的 Horvitz-Thompson 系数是 1/r。RFC-0029 §3.3 的字面组合
    「被遮盖事件 + 1/(1-r)」在 r = 0.5 处与 1/r 数值相同，因此**只在 r != 0.5 时**
    才看得出差别——本文件就在 r = 0.25 / 0.5 / 0.75 三档上把它定量钉死。
"""

from __future__ import annotations

import ast
import inspect
import math
from dataclasses import replace
from pathlib import Path
from typing import get_args

import numpy as np
import pytest
import torch

from beatmorph.core.contracts.phigros import SUBDIVISIONS_PER_BEAT
from beatmorph.field.loss import constant_baseline_lambda
from beatmorph.generation.batch import FieldOutput
from beatmorph.generation.losses import (
    POISSON_LOSSES,
    ReweightMode,
    apply_line_mask_batched,
    event_term,
    full_poisson_loss,
    gaussian_heatmap_target,
    hamming_smooth,
    integral_term,
    masked_poisson_loss,
    occlusion_ratio,
    penalty_reduced_focal_loss,
    per_line_nll,
    poisson_measure_volume,
    timestep_weighted_masked_ce,
)
from beatmorph.generation.masks import build_occlusion_batch, occluded_event_share
from tests.unit.generation._builders import make_batch, make_counts, make_grid

TOL = 1e-6
#: M4 的工程容差（plan 04 §6.2 的 ±5%）
M4_TOL = 0.05
T_BINS = SUBDIVISIONS_PER_BEAT // 4
X_BINS = 8
K_LINES = 3


def _grid():
    return make_grid(t_bins=T_BINS, x_bins=X_BINS)


def _batch(*, ratio: float, reweight_granularity: str = "event", k: int = K_LINES):
    """带遮盖通道的 batch（事件级遮盖，默认路径）。"""
    grid = _grid()
    counts = make_counts(batch=1, k=k, grid=grid, events=10, holds=4, seed=3)
    occlusion, stats = build_occlusion_batch(
        counts,
        ratio=ratio,
        granularity=reweight_granularity,  # type: ignore[arg-type]
        seed=11,
    )
    batch = make_batch(k=k, grid=grid, counts=counts, occlusion=occlusion, audio_dim=16)
    return batch, stats


def _constant_field(batch, value: float) -> FieldOutput:
    """lambda == value 的常数场（G3 口径；|Omega| 取全 K 条线）。"""
    lam = torch.full(batch.batch_field_shape(), float(value), dtype=torch.float32)
    lam = lam * batch.line_mask_bool().reshape(1, -1, 1, 1, 1, 1).to(dtype=lam.dtype)
    return FieldOutput(lam=lam)


def _best_scale(
    batch,
    *,
    reweight: ReweightMode,
    truth: float,
    low: float = 0.01,
    high: float = 10.0,
    samples: int = 2001,
) -> tuple[float, float]:
    """在 lambda = c * truth 的一维族上数值最小化 masked loss，返回 (c*, 最优 loss)。"""
    candidates = np.geomspace(low, high, samples)
    best_c = float(candidates[0])
    best_loss = math.inf
    for value in candidates:
        loss = float(
            masked_poisson_loss(
                _constant_field(batch, float(value) * truth), batch, reweight=reweight
            ),
        )
        if loss < best_loss:
            best_loss = loss
            best_c = float(value)
    return best_c, best_loss


# ══════════════════════════════════════════════════════════════
# 契约 §3.3-1：r == 0 时 masked == full
# ══════════════════════════════════════════════════════════════════


def test_r_zero_masked_equals_full() -> None:
    """plan 04 §3.3-1：无遮盖时 masked_poisson_loss == full_poisson_loss（相对误差 <= 1e-6）。"""
    batch, stats = _batch(ratio=0.0)
    assert stats.ratio == 0.0
    out = _constant_field(batch, 0.7)
    masked = masked_poisson_loss(out, batch)
    full = full_poisson_loss(out, batch)
    assert float(masked) == pytest.approx(float(full), rel=TOL)
    assert occlusion_ratio(batch) == 0.0


def test_r_zero_is_reported_in_log_note() -> None:
    """重标定口径的分歧必须有一条可进训练日志的说明（RFC 待裁定）。"""
    from beatmorph.generation.losses import log_ratio_note

    note = log_ratio_note()
    assert "1/r" in note
    assert "RFC" in note


# ══════════════════════════════════════════════════════════════
# M4：欠计数与重标定（三档 r，暴露 (1-r) 与 r 的分歧）
# ══════════════════════════════════════════════════════════════


@pytest.mark.parametrize("ratio", [0.25, 0.5, 0.75])
def test_unreweighted_loss_undercounts_by_exactly_r(ratio: float) -> None:
    """M4：未重标定（reweight="none"）时，最优场积分 == r x 真值（±5%）。"""
    batch, stats = _batch(ratio=ratio)
    n_events = float(batch.counts.sum().item())
    omega = poisson_measure_volume(batch.grid, n_lines=K_LINES)
    truth = constant_baseline_lambda(n_events, omega)
    best_c, _ = _best_scale(batch, reweight="none", truth=truth)
    achieved = best_c * (truth * omega) / n_events
    assert achieved == pytest.approx(stats.ratio, rel=M4_TOL), (
        f"r={stats.ratio:.3f} 时未重标定的积分占比 {achieved:.3f}（应约等于 r）"
    )


@pytest.mark.parametrize("ratio", [0.25, 0.5, 0.75])
def test_hidden_reweighting_restores_the_integral(ratio: float) -> None:
    """M4：1/r 重标定后最优场积分回到真值（±5%），且与 r 无关。"""
    batch, _ = _batch(ratio=ratio)
    n_events = float(batch.counts.sum().item())
    omega = poisson_measure_volume(batch.grid, n_lines=K_LINES)
    truth = constant_baseline_lambda(n_events, omega)
    best_c, _ = _best_scale(batch, reweight="hidden", truth=truth)
    achieved = best_c * (truth * omega) / n_events
    assert achieved == pytest.approx(1.0, rel=M4_TOL), f"r={ratio} 时积分占比 {achieved:.3f}"


@pytest.mark.parametrize("ratio", [0.25, 0.5, 0.75])
def test_observed_reweighting_is_also_unbiased(ratio: float) -> None:
    """对照口径：监督**已观测**事件 + 1/(1-r) 同样无偏（但存在抄输入的退化解）。"""
    batch, _ = _batch(ratio=ratio)
    n_events = float(batch.counts.sum().item())
    omega = poisson_measure_volume(batch.grid, n_lines=K_LINES)
    truth = constant_baseline_lambda(n_events, omega)
    best_c, _ = _best_scale(batch, reweight="observed", truth=truth)
    achieved = best_c * (truth * omega) / n_events
    assert achieved == pytest.approx(1.0, rel=M4_TOL)


@pytest.mark.parametrize("requested", [0.25, 0.75])
def test_documented_combination_is_inconsistent_and_differs_from_hidden(requested: float) -> None:
    """分歧的定量证据：RFC 字面组合（被遮盖 + 1/(1-r)）的最优解是 r/(1-r) 倍，不是 1。

    r = 0.5 时 r/(1-r) == 1，两种口径**不可区分**；因此本测试刻意取 r != 0.5，
    并要求两者**必须给出不同的最优尺度**（否则说明实现把两种口径混为一谈）。
    注意用**实际** r（事件级遮盖的比例由单位粒度决定，与请求值有量化误差）。
    """
    batch, stats = _batch(ratio=requested)
    assert stats.ratio != pytest.approx(0.5, abs=1e-3)
    n_events = float(batch.counts.sum().item())
    omega = poisson_measure_volume(batch.grid, n_lines=K_LINES)
    truth = constant_baseline_lambda(n_events, omega)
    hidden_c, _ = _best_scale(batch, reweight="hidden", truth=truth)
    doc_c, _ = _best_scale(batch, reweight="hidden_doc", truth=truth)
    assert hidden_c == pytest.approx(1.0, rel=M4_TOL)
    expected_ratio = stats.ratio / (1.0 - stats.ratio)
    assert doc_c / hidden_c == pytest.approx(expected_ratio, rel=M4_TOL)
    assert abs(doc_c / hidden_c - 1.0) > M4_TOL


def test_share_of_r_and_one_are_the_same_at_half() -> None:
    """r = 0.5 是两种系数重合的唯一位置（文档口径偏差在此**不可见**）。"""
    half = 0.5
    assert 1.0 / half == pytest.approx(1.0 / (1.0 - half), rel=TOL)
    for ratio in (0.25, 0.75):
        assert 1.0 / ratio != pytest.approx(1.0 / (1.0 - ratio), rel=1e-3)


def test_reweight_modes_are_enumerated() -> None:
    """四种口径必须全部实现（隐藏口径是默认）。"""
    assert set(get_args(ReweightMode)) == {"hidden", "observed", "hidden_doc", "none"}
    signature = inspect.signature(masked_poisson_loss)
    assert signature.parameters["reweight"].default == "hidden"


# ══════════════════════════════════════════════════════════════
# 积分项：永远在完整域上计算（不随遮盖比例缩放）
# ══════════════════════════════════════════════════════════════


def test_integral_term_does_not_depend_on_occlusion() -> None:
    """plan 04 §4.3：积分项覆盖全 K 线、全域，**不得**随遮盖比例缩放。"""
    values = []
    for ratio in (0.0, 0.25, 0.5, 0.75, 1.0):
        batch, _ = _batch(ratio=ratio)
        out = _constant_field(batch, 0.4)
        values.append(float(integral_term(out, batch).sum()))
    assert all(value == pytest.approx(values[0], rel=TOL) for value in values)


def test_integral_term_excludes_padding_lines() -> None:
    """padding 线（line_mask=False）不得进 int lambda 的求和（plan 04 §3.3-3）。"""
    batch, _ = _batch(ratio=0.4)
    padded = make_batch(
        k=K_LINES,
        grid=batch.grid,
        counts=batch.counts,
        occlusion=batch.occlusion,
        active_lines=[0],
        audio_dim=16,
    )
    out = _constant_field(padded, 0.5)
    per_line = integral_term(out, padded)
    assert float(per_line[0, 1:].abs().sum()) == 0.0
    assert float(per_line[0, 0]) > 0.0
    assert float(per_line[0, 0]) == pytest.approx(
        float(integral_term(out, batch)[0, 0]),
        rel=TOL,
    )


# ══════════════════════════════════════════════════════════════
# 契约 §3.3-3：line_mask 的零贡献（含梯度）
# ══════════════════════════════════════════════════════════════


def test_batched_line_mask_zeros_exactly_the_padding_lines() -> None:
    """M1 回归：field 的 apply_line_mask 对 (B, K, ...) 会**静默错误广播**。

    (1,3) 的掩码被 reshape 成 (3,1,1,1,1,1) 后与 (1,3,T,X,S,C) 相乘，
    第 0 维从 1 被广播成 3——本模块的 apply_line_mask_batched 必须避免这一点。
    """
    batch, _ = _batch(ratio=0.4)
    values = torch.ones(batch.batch_field_shape(), dtype=torch.float64)
    mask = torch.tensor([[True, False, True]])
    masked = apply_line_mask_batched(values, mask)
    assert tuple(masked.shape) == batch.batch_field_shape()
    assert float(masked[0, 1].abs().sum()) == 0.0
    assert float(masked[0, 0].sum()) > 0.0
    # 逐样本口径（(K,) 掩码 + (K, T, X, S, C) 张量）同样正确
    single = apply_line_mask_batched(values[0], torch.tensor([True, False, True]))
    assert float(single[1].abs().sum()) == 0.0


def test_padding_line_gets_no_gradient() -> None:
    """plan 04 §3.3-3：line_mask=False 的线对 loss 与梯度的贡献**恰为 0**。

    注意场本身**不预先置零**（否则 0 * log(0) 会先产生 NaN，测的就不是掩码语义了）。
    """
    batch, _ = _batch(ratio=0.4)
    padded = make_batch(
        k=K_LINES,
        grid=batch.grid,
        counts=batch.counts,
        occlusion=batch.occlusion,
        active_lines=[0, 2],
        audio_dim=16,
    )
    lam = torch.full(padded.batch_field_shape(), 0.5, dtype=torch.float32).requires_grad_(True)
    out = FieldOutput(lam=lam)
    loss = masked_poisson_loss(out, padded)
    loss.backward()
    assert lam.grad is not None
    assert float(lam.grad[0, 1].abs().sum()) == 0.0
    assert float(lam.grad[0, 0].abs().sum()) > 0.0
    per_line = masked_poisson_loss(out, padded, reduction="none")
    assert float(per_line[0, 1].detach()) == 0.0


# ══════════════════════════════════════════════════════════════
# 契约 §3.3-2：排列敏感性（判定线不可互换）
# ══════════════════════════════════════════════════════════════


def test_permuting_events_between_lines_changes_the_loss() -> None:
    """两条线的事件互换后 loss 必须改变（RFC-0029 §2.4-3，判定线不可互换）。

    场的强度按线不同（line 0 弱、line 1 强），因此「事件落在哪条线」不是无关信息。
    """
    batch, _ = _batch(ratio=0.4)
    lam = torch.full(batch.batch_field_shape(), 0.01, dtype=torch.float32)
    lam[0, 1] = 5.0
    out = FieldOutput(lam=lam)
    base = float(masked_poisson_loss(out, batch))
    permuted = batch.counts.clone()
    permuted[:, [0, 1]] = permuted[:, [1, 0]]
    occlusion = batch.occlusion_bool().clone()
    occlusion[:, [0, 1]] = occlusion[:, [1, 0]]
    swapped = make_batch(
        k=K_LINES,
        grid=batch.grid,
        counts=permuted,
        occlusion=occlusion,
        audio_dim=16,
    )
    assert float(masked_poisson_loss(out, swapped)) != pytest.approx(base, rel=1e-9)


def test_permuting_the_field_alone_changes_the_loss() -> None:
    """只交换场（不动事件）也必须改变 loss——否则损失对线身份不敏感。"""
    batch, _ = _batch(ratio=0.5)
    counts = batch.counts.float()
    # 让两条线的强度差异极大，交换后事件项必然变化
    lam = torch.full(batch.batch_field_shape(), 0.01, dtype=torch.float32)
    lam[0, 1] = 5.0
    lam[0, 0] = 0.001
    base = float(masked_poisson_loss(FieldOutput(lam=lam), batch))
    swapped = lam.clone()
    swapped[0, 0], swapped[0, 1] = lam[0, 1].clone(), lam[0, 0].clone()
    assert float(masked_poisson_loss(FieldOutput(lam=swapped), batch)) != pytest.approx(
        base,
        rel=1e-9,
    )
    assert counts.sum() > 0


# ══════════════════════════════════════════════════════════════
# 契约 §3.3-4：不存在 line 分类损失（源码级）
# ══════════════════════════════════════════════════════════════

#: 泊松 NLL 的入口里**禁止**出现的调用（它们意味着引入了分类 / 分配项）
FORBIDDEN_CALLS = {
    "softmax",
    "log_softmax",
    "cross_entropy",
    "nll_loss",
    "binary_cross_entropy",
    "binary_cross_entropy_with_logits",
}


def _called_names(function: object) -> set[str]:
    """静态收集一个函数体里出现的所有被调用名字（含属性名）。"""
    source = inspect.getsource(function)
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Attribute):
                names.add(target.attr)
            elif isinstance(target, ast.Name):
                names.add(target.id)
    return names


@pytest.mark.parametrize("function", POISSON_LOSSES)
def test_poisson_losses_have_no_classification_terms(function: object) -> None:
    """plan 04 §3.3-4：泊松损失只由事件项 + 积分项构成，**没有任何**分类 / 分配项。"""
    names = _called_names(function)
    forbidden = names & FORBIDDEN_CALLS
    assert not forbidden, f"{getattr(function, '__name__', function)} 出现了分类项：{forbidden}"


def test_forbidden_scan_is_not_vacuous() -> None:
    """扫描器自检：对消融臂的 CE 目标必须**抓到** log_softmax（否则门禁是空转的）。"""
    names = _called_names(timestep_weighted_masked_ce)
    assert "log_softmax" in names


def test_poisson_loss_signatures_have_no_weights() -> None:
    """损失签名里没有任何权重参数（通道 / 侧别 / 线的系数恒为 1）。"""
    weight_like = {"weight", "weights", "class_weight", "pos_weight", "loss_weight"}
    for function in POISSON_LOSSES:
        parameters = inspect.signature(function).parameters
        assert inspect.Parameter.VAR_KEYWORD not in {p.kind for p in parameters.values()}
        assert not (set(parameters) & weight_like), parameters


# ══════════════════════════════════════════════════════════════
# 事件项语义（继承 plan 03 的桶内计数口径）
# ══════════════════════════════════════════════════════════════


def test_duplicate_events_contribute_two_log_terms() -> None:
    """n_j = 2 的格子贡献 2 * log lambda（**禁止去重**，plan 03 §3.3-3）。"""
    batch, _ = _batch(ratio=0.0)
    shape = batch.batch_field_shape()
    counts = torch.zeros(shape, dtype=torch.float32)
    counts[0, 0, 0, 0, 0, 0] = 2.0
    lam = torch.full(shape, 0.5, dtype=torch.float32)
    duplicated = float(event_term(counts, lam, None)[0, 0])
    counts[0, 0, 0, 0, 0, 0] = 1.0
    single = float(event_term(counts, lam, None)[0, 0])
    assert duplicated == pytest.approx(2.0 * single, rel=TOL)
    assert single == pytest.approx(-math.log(0.5), rel=TOL)


def test_zero_lambda_at_supervised_event_is_infinite() -> None:
    """被监督事件处 lambda == 0 必须给出非有限值（**禁止 eps 平滑**）。"""
    batch, _ = _batch(ratio=0.5)
    lam = torch.zeros(batch.batch_field_shape(), dtype=torch.float32)
    loss = masked_poisson_loss(FieldOutput(lam=lam), batch)
    assert not bool(torch.isfinite(loss))


def test_loss_requires_counts() -> None:
    """没有目标时损失入口必须显式报错，而不是静默返回 0。"""
    batch, _ = _batch(ratio=0.4)
    without = replace(batch, counts=None, occlusion=None)
    with pytest.raises(ValueError, match="counts"):
        full_poisson_loss(_constant_field(batch, 0.5), without)


def test_full_occlusion_is_rejected_for_hidden_supervision() -> None:
    """r == 1（全部被遮盖）时事件项没有可见上下文：必须拒绝而不是给出 0。"""
    batch, stats = _batch(ratio=1.0)
    assert stats.ratio == pytest.approx(1.0)
    with pytest.raises(ValueError, match="r == 1"):
        masked_poisson_loss(_constant_field(batch, 0.5), batch, reweight="hidden")


def test_reduction_none_returns_per_sample_per_line() -> None:
    """reduction="none" 返回逐 (B, K) 张量（供 per-line 诊断使用）。"""
    batch, _ = _batch(ratio=0.4)
    out = _constant_field(batch, 0.5)
    per_line = masked_poisson_loss(out, batch, reduction="none")
    assert tuple(per_line.shape) == (1, K_LINES)
    assert float(per_line.sum()) == pytest.approx(float(masked_poisson_loss(out, batch)), rel=TOL)
    assert tuple(per_line_nll(out, batch, masked=True).shape) == (1, K_LINES)


def test_occlusion_share_matches_mask_statistics() -> None:
    """r 的口径：按事件计（桶内计数加权），不是按格计。"""
    batch, stats = _batch(ratio=0.5)
    share = occluded_event_share(batch.counts, batch.occlusion_bool())
    assert share == pytest.approx(stats.ratio, rel=TOL)
    assert share == pytest.approx(occlusion_ratio(batch), rel=TOL)


# ══════════════════════════════════════════════════════════════
# 消融臂目标（B1 热图 + focal / B5 时间步加权掩码 CE）
# ══════════════════════════════════════════════════════════════


def test_heatmap_target_is_unnormalized_and_max_overlapped() -> None:
    """B1 的 y 是**未归一化**高斯且重叠取 max：int y 与事件数不同测度（RFC-0029 §3.4）。"""
    counts = torch.zeros(1, 5, 4, 1, 1, dtype=torch.float32)
    counts[0, 2, 1, 0, 0] = 1.0
    target = gaussian_heatmap_target(counts, sigma_t=1.0, sigma_x=1.0, radius_t=2, radius_x=1)
    assert float(target.max()) == pytest.approx(1.0, rel=TOL)
    assert float(target.sum()) != pytest.approx(1.0, rel=1e-3)
    peak = target[0, 2, 1, 0, 0]
    assert float(peak) == pytest.approx(1.0, rel=TOL)
    # 重叠取 max（不是相加）：两个相邻事件的高斯不得让中心值超过 1
    counts[0, 3, 1, 0, 0] = 1.0
    overlapped = gaussian_heatmap_target(counts, sigma_t=1.0, sigma_x=1.0, radius_t=2, radius_x=1)
    assert float(overlapped.max()) == pytest.approx(1.0, rel=TOL)


def test_focal_loss_requires_explicit_alpha_beta() -> None:
    """B1 的 alpha / beta 在文献中未核实：签名里**不得**有默认值。"""
    parameters = inspect.signature(penalty_reduced_focal_loss).parameters
    assert parameters["alpha"].default is inspect.Parameter.empty
    assert parameters["beta"].default is inspect.Parameter.empty
    with pytest.raises(TypeError):
        penalty_reduced_focal_loss(torch.zeros(2), torch.zeros(2))  # type: ignore[call-arg]


def test_focal_loss_is_minimised_by_matching_target() -> None:
    """focal loss 在预测与目标一致时应低于错配预测（不是空转的损失）。"""
    target = torch.tensor([1.0, 0.0])
    good = penalty_reduced_focal_loss(torch.tensor([6.0, -6.0]), target, alpha=2.0, beta=4.0)
    bad = penalty_reduced_focal_loss(torch.tensor([-6.0, 6.0]), target, alpha=2.0, beta=4.0)
    assert float(good) < float(bad)


def test_hamming_smooth_preserves_the_time_axis_mass() -> None:
    """DDC 式 Hamming 平滑：权重归一化 -> 沿 tau 轴的总量守恒（不改幅度语义）。"""
    target = torch.zeros(1, 4, 2, 1, 1, dtype=torch.float32)
    target[0, 2, 0, 0, 0] = 3.0
    smoothed = hamming_smooth(target, window=3)
    assert tuple(smoothed.shape) == tuple(target.shape)
    assert float(smoothed.sum()) == pytest.approx(float(target.sum()), rel=1e-5)


def test_timestep_weighted_ce_matches_manual_computation() -> None:
    """B5：时间步加权的掩码 CE（absorbing 扩散 NELBO 的等价形式）。"""
    logits = torch.zeros(1, 2, 3)
    targets = torch.tensor([[0, 2]])
    mask = torch.tensor([[True, False]])
    weights = torch.tensor([[0.5, 0.5]])
    value = timestep_weighted_masked_ce(
        logits,
        targets,
        mask=mask,
        timestep_weights=weights,
        reduction="sum",
    )
    expected = 0.5 * math.log(3.0)
    assert float(value) == pytest.approx(expected, rel=1e-5)
    zeroed = timestep_weighted_masked_ce(
        logits,
        targets,
        mask=torch.zeros(1, 2, dtype=torch.bool),
        timestep_weights=weights,
    )
    assert float(zeroed) == 0.0


def test_losses_module_lives_under_generation() -> None:
    """本测试文件的被测模块必须落在 beatmorph/generation/（防止误测到别处）。"""
    module = inspect.getmodule(full_poisson_loss)
    assert module is not None
    path = Path(module.__file__ or "")
    assert path.parent.name == "generation"
