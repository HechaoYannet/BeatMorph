"""`reduction="per_event"` 的契约（RFC-0037 R2 / §2.2）——默认 CI，无 GPU / 无权重。

钉死四件事：

1. **尺度恒等式**：per_event == sum / max(E_total, 1)（masked 与 full 两条路径都对）；
2. **argmin 不变**：整式缩放 ⇒ 梯度恰差 1/D 倍 ⇒ 驻点（最优强度）逐位不变。
   这是「只除事件项会把最优 λ 缩小 D 倍」陷阱（RFC-0037 §2.2）的护栏；
3. **空批**：E_total == 0 时 D = 1 ⇒ per_event 与 sum 相同（纯积分项，行为不变）；
4. **r == 0 契约**：无遮盖时 masked(per_event) == full(per_event)（原契约在归一化下保持）。
"""

from __future__ import annotations

import pytest
import torch

from beatmorph.generation.batch import FieldBatch, FieldOutput
from beatmorph.generation.losses import (
    event_normalizer,
    full_poisson_loss,
    masked_poisson_loss,
)
from tests.unit.generation._builders import make_batch, make_grid

T_BINS = 12
X_BINS = 8


def _batch(*, events: int, seed: int = 3, occlusion: bool = True) -> FieldBatch:
    grid = make_grid(t_bins=T_BINS, x_bins=X_BINS)
    batch = make_batch(k=2, grid=grid, batch=2, events=events, holds=0, seed=seed)
    if not occlusion and batch.occlusion is not None:
        from dataclasses import replace

        batch = replace(batch, occlusion=None)
    batch.assert_shapes()
    return batch


def _lam(batch: FieldBatch, *, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    shape = batch.batch_field_shape()
    return torch.rand(shape, generator=generator) * 0.5 + 0.05


def test_normalizer_is_event_total_with_floor_one() -> None:
    batch = _batch(events=6)
    assert batch.counts is not None
    total = float(batch.counts.sum())
    assert total > 0
    assert event_normalizer(batch) == pytest.approx(total)
    empty = _batch(events=0)
    assert event_normalizer(empty) == 1.0


def test_per_event_equals_sum_over_divisor_masked_and_full() -> None:
    """契约 1：两条损失路径上 per_event == sum / max(E, 1)。"""
    for occlusion in (True, False):
        batch = _batch(events=6, occlusion=occlusion)
        out = FieldOutput(lam=_lam(batch))
        d = event_normalizer(batch)
        masked_sum = masked_poisson_loss(out, batch, reduction="sum")
        masked_pe = masked_poisson_loss(out, batch, reduction="per_event")
        assert float(masked_pe) == pytest.approx(float(masked_sum) / d, rel=1e-6)
        full_sum = full_poisson_loss(out, batch, reduction="sum")
        full_pe = full_poisson_loss(out, batch, reduction="per_event")
        assert float(full_pe) == pytest.approx(float(full_sum) / d, rel=1e-6)


def test_gradients_scale_exactly_by_inverse_divisor() -> None:
    """契约 2：∇L_per_event == ∇L_sum / D ⇒ 驻点（argmin）不变。

    若有人把归一化改成「只除事件项」，事件项与积分项的梯度比例会被改掉，
    本断言立刻失败——那正是 RFC-0037 §2.2 推导的静默失效形态。
    """
    batch = _batch(events=6)
    d = event_normalizer(batch)
    lam_sum = _lam(batch).clone().requires_grad_(True)
    lam_pe = lam_sum.detach().clone().requires_grad_(True)
    masked_poisson_loss(FieldOutput(lam=lam_sum), batch, reduction="sum").backward()
    masked_poisson_loss(FieldOutput(lam=lam_pe), batch, reduction="per_event").backward()
    assert lam_sum.grad is not None
    assert lam_pe.grad is not None
    assert torch.allclose(lam_pe.grad * d, lam_sum.grad, rtol=1e-5, atol=1e-8)


def test_empty_batch_falls_back_to_divisor_one() -> None:
    """契约 3：E == 0 时 D = 1，per_event 与 sum 逐位相同（纯积分项，旧行为不变）。"""
    batch = _batch(events=0)
    out = FieldOutput(lam=_lam(batch))
    assert event_normalizer(batch) == 1.0
    assert float(masked_poisson_loss(out, batch, reduction="per_event")) == pytest.approx(
        float(masked_poisson_loss(out, batch, reduction="sum")), rel=1e-6
    )


def test_r_zero_contract_survives_normalization() -> None:
    """契约 4：r == 0（无遮盖）时 masked(per_event) == full(per_event)。"""
    batch = _batch(events=6, occlusion=False)
    assert batch.occlusion is None
    out = FieldOutput(lam=_lam(batch))
    assert float(masked_poisson_loss(out, batch, reduction="per_event")) == pytest.approx(
        float(full_poisson_loss(out, batch, reduction="per_event")), rel=1e-6
    )
