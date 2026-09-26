"""Plan 04 M2：可变 K 前向（K = 1 / 30 / 82、批量内混合 K、line_mask 零贡献）。

契约来源（docs/plans/04-generation.md）：
- §3.3-2/3 契约断言：判定线不可互换（交换两条线 -> loss 必须改变）；
  line_mask 为 False 的线对 loss 与梯度的贡献**恰为 0**；
- §4.1 判定线的可变 K：K **不进入任何输出层形状**，同一组权重服务 K = 2 与 K = 82，
  线数只经 K 维批处理 + line_mask 处理；
- §6.2 M2 的验收：同一组权重跑通 K = 1 / 30 / 82 与批量内混合 K（padding + line_mask）。

默认 CI：**无权重、无 GPU**。网格规模（tau 格数、x 桶数、音频帧数）全部由
`SUBDIVISIONS_PER_BEAT` / `FieldGrid` / `audio_frames_for` 派生，
不出现任何硬编码物理常量；模型极小（d_model=16、2 层），单文件耗时 < 5 s。
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from torch import Tensor

from beatmorph.core.contracts.phigros import SUBDIVISIONS_PER_BEAT
from beatmorph.field.grid import FieldGrid
from beatmorph.generation.batch import FieldBatch, FieldOutput
from beatmorph.generation.losses import masked_poisson_loss
from beatmorph.generation.masks import build_occlusion_batch
from beatmorph.generation.model import MaskedFieldModel, ModelConfig
from tests.unit.generation._builders import TEST_AUDIO_DIM, make_batch, make_grid

# ── 测试规模（全部派生 / 超参）────────────────────────────────────────
#: tau 格数 = SUBDIVISIONS_PER_BEAT // 4（派生，**不写 12**）
T_BINS: int = SUBDIVISIONS_PER_BEAT // 4
#: x 桶数（测试用超参）
X_BINS: int = 8
#: line embedding 容量（M2 允许的上界）
K_MAX: int = 96
#: M2 要求跑通的 K 档位：1（退化单线）/ 30（实测中位）/ 82（实测极值）
K_SWEEP: tuple[int, ...] = (1, 30, 82)
#: 批量内混合 K 的线数与有效线（其余为 padding）
MIXED_K: int = 6
MIXED_ACTIVE: tuple[int, ...] = (0, 2, 3)
#: 排列敏感性用到的线数（交换线 0 与线 1）
PERM_K: int = 4
PERM_SWAP: tuple[int, ...] = (1, 0, 2, 3)
#: 模型超参（极小，CPU 上毫秒级）
D_MODEL: int = 16
N_HEADS: int = 2
N_LAYERS: int = 2
WINDOW: int = 2
GLOBAL_PERIOD: int = 2
MODEL_SEED: int = 0
#: 遮盖比例（M2 的 padding 零贡献断言在遮盖路径下也成立）
OCCLUSION_RATIO: float = 0.5
#: padding 线上注入的「垃圾」强度值：若 line_mask 真的生效，它不得影响 loss
PADDING_GARBAGE: float = 1e3

# ══════════════════════════════════════════════════════════════
# 回归记录（本测试曾以 strict xfail 标记一个真实缺陷，现已修复）
# ══════════════════════════════════════════════════════════════
#: event_term 曾无条件 torch.log(lam)，而 padding 线的 lam 恰为 0 ——
#: log(0) 的梯度 inf 乘以上游零梯度得 NaN，反传到**全部参数**，而 loss 有限、不报错。
#: 修法：先按 line_mask 清零监督集合、再取 log（plan 04 §9-19）。本测试现在必须通过。


# ══════════════════════════════════════════════════════════════
# 夹具与构造器
# ══════════════════════════════════════════════════════════════
@pytest.fixture(scope="module")
def grid() -> FieldGrid:
    """小网格：tau 格数由 SUBDIVISIONS_PER_BEAT 派生，x 轴 8 桶。"""
    return make_grid(t_bins=T_BINS, x_bins=X_BINS)


@pytest.fixture(scope="module")
def model(grid: FieldGrid) -> MaskedFieldModel:
    """同一组权重（整个模块共享），eval + dropout=0 => 前向确定性。"""
    torch.manual_seed(MODEL_SEED)
    config = ModelConfig(
        k_max=K_MAX,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        window=WINDOW,
        global_period=GLOBAL_PERIOD,
        audio_dim=TEST_AUDIO_DIM,
    )
    net = MaskedFieldModel(config, grid)
    net.eval()
    return net


def _batch(
    grid: FieldGrid, *, k: int, seed: int = 0, active: tuple[int, ...] | None = None
) -> FieldBatch:
    """带桶内计数目标的合成 batch（事件 / Hold 位置由 seed 决定）。"""
    batch = make_batch(k=k, grid=grid, events=2 * k, holds=2, seed=seed, active_lines=active)
    batch.assert_shapes()
    return batch


def _zero_padding_lines(batch: FieldBatch) -> Tensor:
    """把 padding 线（line_mask=False）上的计数清零——它们本就不该有事件。"""
    keep = batch.line_mask.reshape(*batch.line_mask.shape, 1, 1, 1, 1)
    assert batch.counts is not None
    return batch.counts.masked_fill(~keep, 0)


def _mixed_batch(grid: FieldGrid) -> FieldBatch:
    """混合 K batch：K=MIXED_K，只有 MIXED_ACTIVE 有效，其余为 padding。"""
    batch = _batch(grid, k=MIXED_K, seed=5, active=MIXED_ACTIVE)
    counts = _zero_padding_lines(batch)
    occlusion, _ = build_occlusion_batch(
        counts,
        ratio=OCCLUSION_RATIO,
        granularity="event",
        seed=0,
        line_mask=batch.line_mask,
    )
    return dataclasses.replace(batch, counts=counts, occlusion=occlusion)


# ══════════════════════════════════════════════════════════════
# M2：同一组权重的可变 K
# ══════════════════════════════════════════════════════════════
@pytest.mark.parametrize("k", K_SWEEP)
def test_forward_shapes_hold_for_variable_k(
    model: MaskedFieldModel,
    grid: FieldGrid,
    k: int,
) -> None:
    """M2：同一组权重跑通 K = 1 / 30 / 82，输出 (B, K, T, X, S, C) 且 lambda >= 0、有限。"""
    batch = _batch(grid, k=k, seed=k)
    with torch.no_grad():
        out = model(batch)
    out.assert_shapes(batch)
    assert (
        tuple(out.lam.shape)
        == batch.batch_field_shape()
        == (
            batch.batch_size(),
            k,
            grid.t_bins,
            grid.x_bins,
            grid.sides,
            grid.channels,
        )
    )
    assert float(out.lam.min().item()) >= 0.0
    assert bool(torch.isfinite(out.lam).all())
    assert float(out.lam.max().item()) > 0.0
    # K 不进入输出层形状：同一模型实例对三档 K 都给出 (B, K, ...)
    assert out.lam.shape[1] == k
    assert out.n_lines() == k


def test_forward_is_deterministic(model: MaskedFieldModel, grid: FieldGrid) -> None:
    """M2：eval + dropout=0 下同一 batch 连续两次前向必须逐元素相等（训练可复现）。"""
    batch = _mixed_batch(grid)
    with torch.no_grad():
        first = model(batch).lam
        second = model(batch).lam
    assert torch.equal(first, second)


# ══════════════════════════════════════════════════════════════
# M2：批量内混合 K，padding 线的零贡献
# ══════════════════════════════════════════════════════════════
def test_mixed_k_padding_lines_have_exactly_zero_lambda(
    model: MaskedFieldModel,
    grid: FieldGrid,
) -> None:
    """M2：批量内混合 K 时 padding 线（line_mask=False）的 lambda **恰为 0**。"""
    batch = _mixed_batch(grid)
    padding = ~batch.line_mask[0]
    assert bool(padding.any()), "混合 K 的构造必须真的产生 padding 线"
    assert not bool(padding.all()), "混合 K 的构造必须保留有效线"
    with torch.no_grad():
        out = model(batch)
    assert float(out.lam[0, padding].abs().max().item()) == 0.0
    # 有效线不得被一起清零（否则「恰为 0」是空转的）
    assert float(out.lam[0, ~padding].abs().max().item()) > 0.0


def test_padding_lines_leave_the_loss_untouched(
    model: MaskedFieldModel,
    grid: FieldGrid,
) -> None:
    """M2 / §3.3-3：padding 线上的 lambda 取值不影响 loss（损失层把其贡献恰置为 0）。"""
    batch = _mixed_batch(grid)
    padding = ~batch.line_mask[0]
    with torch.no_grad():
        reference = model(batch).lam.detach().clone()
    clean = reference.clone().requires_grad_(True)
    garbage = reference.clone()
    with torch.no_grad():
        garbage[0, padding] = PADDING_GARBAGE
    garbage.requires_grad_(True)
    loss_clean = masked_poisson_loss(FieldOutput(lam=clean), batch)
    loss_garbage = masked_poisson_loss(FieldOutput(lam=garbage), batch)
    assert float(loss_clean.item()) == pytest.approx(float(loss_garbage.item()), abs=1e-9)
    loss_garbage.backward()
    assert garbage.grad is not None
    assert float(garbage.grad[0, padding].abs().max().item()) == 0.0
    assert float(garbage.grad[0, ~padding].abs().max().item()) > 0.0


def test_padding_lines_contribute_no_gradient(
    model: MaskedFieldModel,
    grid: FieldGrid,
) -> None:
    """M2 / §3.3-3：line_mask=False 的线对 loss 的**梯度**贡献恰为 0（不是 NaN）。

    注意：不能用 `out.lam.sum().backward()` 来检查——那是「和对其自身操作数的梯度」，
    在**所有**位置恒为 1，padding 位置也不例外，因此证明不了任何事。
    这里直接取 loss 对 lam 的梯度（retain_grad），并检查参数梯度必须全部有限。
    """
    batch = _mixed_batch(grid)
    padding = ~batch.line_mask[0]
    model.zero_grad(set_to_none=True)
    out = model(batch)
    out.lam.retain_grad()
    loss = masked_poisson_loss(out, batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert out.lam.grad is not None
    assert float(out.lam.grad[0, padding].abs().max().item()) == 0.0
    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert grads
    assert all(bool(torch.isfinite(grad).all()) for grad in grads), "padding 线污染了参数梯度"


# ══════════════════════════════════════════════════════════════
# §3.3-2：判定线不可互换（RFC-0029 §2.4-3）
# ══════════════════════════════════════════════════════════════
def test_swapping_two_lines_changes_the_output(
    model: MaskedFieldModel,
    grid: FieldGrid,
) -> None:
    """RFC-0029 §2.4-3 / Plan 04 §3.3-2：交换两条线的场与事件后，输出与 loss 必须改变。"""
    batch = _batch(grid, k=PERM_K, seed=7)
    assert batch.counts is not None
    swapped = dataclasses.replace(
        batch,
        counts=batch.counts[:, PERM_SWAP, ...],
        line_tracks=batch.line_tracks[:, PERM_SWAP, ...],
    )
    swapped.assert_shapes()
    with torch.no_grad():
        reference = model(batch).lam
        permuted = model(swapped).lam
    assert float((reference - permuted).abs().max().item()) > 0.0
    assert float(masked_poisson_loss(FieldOutput(lam=reference), batch).item()) != pytest.approx(
        float(masked_poisson_loss(FieldOutput(lam=permuted), swapped).item()),
        abs=1e-6,
    )


# ══════════════════════════════════════════════════════════════
# §4.1 / §9-8：K 的上界与形状门禁
# ══════════════════════════════════════════════════════════════
def test_k_above_k_max_is_rejected(model: MaskedFieldModel, grid: FieldGrid) -> None:
    """M2 / Plan 04 §9-8：K > k_max 必须抛 ValueError（不得静默截断或越界索引）。"""
    batch = _batch(grid, k=K_MAX + 1, seed=1)
    with pytest.raises(ValueError, match="k_max"), torch.no_grad():
        model(batch)


def test_forward_state_rejects_mismatched_state_or_occlusion(
    model: MaskedFieldModel,
    grid: FieldGrid,
) -> None:
    """Plan 04 §3.1：forward_state 对 state / occlusion 形状不匹配必须抛 AssertionError。"""
    batch = _batch(grid, k=PERM_K, seed=3)
    good = batch.observed_counts()
    with pytest.raises(AssertionError, match="state / occlusion"):
        model.forward_state(batch, torch.zeros(1, dtype=torch.float32), good.bool())
    with pytest.raises(AssertionError, match="state / occlusion"):
        model.forward_state(batch, good, torch.zeros(1, dtype=torch.bool))
