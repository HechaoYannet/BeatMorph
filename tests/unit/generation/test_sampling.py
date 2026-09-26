"""M6：迭代并行解码（`steps >= 2` 契约、schedule 单调性、三种置信度定义）。

默认 CI，无权重 / 无 GPU。采样是**推理路径**：不得建图（no_grad）、不得有隐式随机数。
"""

from __future__ import annotations

import itertools

import pytest
import torch

from beatmorph.core.contracts.phigros import SUBDIVISIONS_PER_BEAT
from beatmorph.generation.masks import build_occlusion_batch
from beatmorph.generation.model import MaskedFieldModel, ModelConfig
from beatmorph.generation.sampling import (
    DEFAULT_STEPS,
    MIN_STEPS,
    SamplingConfig,
    assert_schedule_monotone,
    confidence_map,
    expected_counts,
    local_contrast,
    reveal_schedule,
    sample,
)
from tests.unit.generation._builders import TEST_AUDIO_DIM, make_batch, make_counts, make_grid

T_BINS = SUBDIVISIONS_PER_BEAT // 4
X_BINS = 8
K_LINES = 2
MODEL_CONFIG = ModelConfig(
    d_model=16,
    n_heads=2,
    n_layers=2,
    window=2,
    global_period=2,
    k_max=8,
    audio_dim=TEST_AUDIO_DIM,
    dropout=0.1,
)


def _setup(*, ratio: float = 0.5, events: int = 5, holds: int = 1):
    grid = make_grid(t_bins=T_BINS, x_bins=X_BINS)
    counts = make_counts(k=K_LINES, grid=grid, events=events, holds=holds, seed=6)
    occlusion, _ = build_occlusion_batch(counts, ratio=ratio, seed=6)
    batch = make_batch(
        k=K_LINES,
        grid=grid,
        counts=counts,
        occlusion=occlusion,
        audio_dim=TEST_AUDIO_DIM,
    )
    model = MaskedFieldModel(MODEL_CONFIG, grid).eval()
    return model, batch


# ══════════════════════════════════════════════════════════════
# 契约 1：steps >= 2（一步到位不可行）
# ══════════════════════════════════════════════════════════════


@pytest.mark.parametrize("steps", [1, 0, -3])
def test_steps_below_two_are_rejected(steps: int) -> None:
    """MaskGIT 原文：一次推断全部与训练分布不一致 —— 因此 steps < 2 必须被拒绝。"""
    with pytest.raises(ValueError, match="steps"):
        SamplingConfig(steps=steps)
    with pytest.raises(ValueError, match="steps"):
        reveal_schedule(steps)


def test_min_steps_and_default_steps_are_sane() -> None:
    """最小步数为 2；默认步数取 MaskGIT 的 8 步参照。"""
    assert MIN_STEPS == 2
    assert DEFAULT_STEPS >= MIN_STEPS
    assert SamplingConfig().steps == DEFAULT_STEPS


# ══════════════════════════════════════════════════════════════
# 契约 2：schedule 单调不减且末项为 1
# ══════════════════════════════════════════════════════════════


@pytest.mark.parametrize("schedule", ["linear", "cosine"])
@pytest.mark.parametrize("steps", [2, 3, 8])
def test_reveal_schedule_is_monotone_and_complete(schedule: str, steps: int) -> None:
    """保留比例 gamma(t/T) 必须单调不减且最后一步揭开全部（终止条件）。"""
    fractions = reveal_schedule(steps, schedule)  # type: ignore[arg-type]
    assert len(fractions) == steps
    assert fractions[-1] == 1.0
    assert all(later >= earlier for earlier, later in itertools.pairwise(fractions))
    assert all(0.0 < value <= 1.0 for value in fractions)
    assert_schedule_monotone(fractions)


def test_schedule_guard_catches_a_decreasing_schedule() -> None:
    """门禁自检：非单调 / 末尾不为 1 的 schedule 必须被抓到。"""
    with pytest.raises(AssertionError, match="单调"):
        assert_schedule_monotone((0.5, 0.2, 1.0))
    with pytest.raises(AssertionError, match="末项"):
        assert_schedule_monotone((0.5, 0.9))
    with pytest.raises(AssertionError, match="不得为空"):
        assert_schedule_monotone(())


# ══════════════════════════════════════════════════════════════
# 采样主流程
# ══════════════════════════════════════════════════════════════


def test_sample_returns_a_complete_field_with_diagnostics() -> None:
    """采样结束后全部格子都已「未遮盖」，diagnostics 必须带步数与逐步揭开比例。"""
    model, batch = _setup()
    result = sample(model, batch, config=SamplingConfig(steps=3, confidence="event_count"))
    assert tuple(result.lam.shape) == batch.batch_field_shape()
    assert bool(torch.isfinite(result.lam).all())
    assert float(result.lam.min()) >= 0.0
    steps = result.diagnostics["steps"]
    assert int(steps.item()) == 3
    revealed = result.diagnostics["revealed_fraction"].tolist()
    assert len(revealed) == 3
    assert revealed[-1] == pytest.approx(1.0)
    assert all(later >= earlier for earlier, later in itertools.pairwise(revealed))
    assert result.cell_prob is not None
    assert result.cum is not None


def test_sample_is_deterministic() -> None:
    """同一模型 / 同一输入两次采样必须逐元素相等（无隐式随机数）。"""
    model, batch = _setup()
    config = SamplingConfig(steps=4, confidence="peak_salience")
    first = sample(model, batch, config=config)
    second = sample(model, batch, config=config)
    assert torch.equal(first.lam, second.lam)
    assert (
        first.diagnostics["revealed_fraction"].tolist()
        == second.diagnostics["revealed_fraction"].tolist()
    )


def test_sample_does_not_build_a_graph() -> None:
    """采样是推理路径：输出不得带梯度（内部必须 no_grad）。"""
    model, batch = _setup()
    result = sample(model, batch, config=SamplingConfig(steps=2))
    assert not result.lam.requires_grad


@pytest.mark.parametrize("confidence", ["event_count", "peak_salience", "variance"])
def test_three_confidence_definitions_all_run(confidence: str) -> None:
    """plan 04 §4.4 的三种连续场置信度候选必须全部可跑（选择属消融，未定）。"""
    model, batch = _setup()
    config = SamplingConfig(steps=3, confidence=confidence)  # type: ignore[arg-type]
    result = sample(model, batch, config=config)
    assert tuple(result.lam.shape) == batch.batch_field_shape()
    code = int(result.diagnostics["confidence"].item())
    assert code == {"event_count": 0, "peak_salience": 1, "variance": 2}[confidence]


def test_variance_confidence_requires_at_least_two_forwards() -> None:
    """方差置信度至少两次前向，否则方差恒为 0。"""
    with pytest.raises(ValueError, match="variance"):
        SamplingConfig(steps=3, confidence="variance", variance_samples=1)


def test_state_fill_modes_both_work() -> None:
    """被保留位置的两种写入方式（binary / lambda）都必须可跑。"""
    model, batch = _setup()
    for fill in ("binary", "lambda"):
        result = sample(
            model,
            batch,
            config=SamplingConfig(steps=3, state_fill=fill),  # type: ignore[arg-type]
        )
        assert bool(torch.isfinite(result.lam).all())


def test_more_steps_are_allowed() -> None:
    """步数是消融维度：更多步必须仍然收敛到全部揭开。"""
    model, batch = _setup()
    result = sample(model, batch, config=SamplingConfig(steps=DEFAULT_STEPS))
    assert result.diagnostics["revealed_fraction"][-1].item() == pytest.approx(1.0)


# ══════════════════════════════════════════════════════════════
# 置信度与基础量
# ══════════════════════════════════════════════════════════════


def test_expected_counts_uses_the_grid_measure() -> None:
    """事件级置信度 = lambda * dV_j（dV 必须来自 plan 03 的网格，不得另算）。"""
    model, batch = _setup()
    out = model.forward_state(batch, batch.observed_counts(), batch.occlusion_bool())
    counts = expected_counts(out, batch)
    volumes = torch.as_tensor(batch.grid.cell_volumes(), dtype=out.lam.dtype).reshape(
        1,
        1,
        -1,
        1,
        1,
        1,
    )
    assert torch.allclose(counts, out.lam * volumes, atol=1e-6)


def test_local_contrast_highlights_peaks() -> None:
    """峰值显著性：孤立峰处为正，平坦处为 0；window = 0 时恒为 0。"""
    values = torch.zeros(1, 1, 5, 5, 1, 1)
    values[0, 0, 2, 2, 0, 0] = 3.0
    contrast = local_contrast(values, window=1)
    assert float(contrast[0, 0, 2, 2, 0, 0]) > 0.0
    assert float(contrast[0, 0, 0, 0, 0, 0]) <= 0.0
    assert float(local_contrast(values, window=0).abs().sum()) == 0.0


def test_confidence_map_variance_needs_the_variance_tensor() -> None:
    """方差口径缺少方差张量时必须显式报错（不得静默退化为其它口径）。"""
    model, batch = _setup()
    out = model.forward_state(batch, batch.observed_counts(), batch.occlusion_bool())
    with pytest.raises(ValueError, match="variance"):
        confidence_map(out, batch, SamplingConfig(steps=2, confidence="variance"))
    provided = confidence_map(
        out,
        batch,
        SamplingConfig(steps=2, confidence="variance"),
        variance=torch.zeros_like(out.lam),
    )
    assert torch.equal(provided, torch.zeros_like(out.lam))
