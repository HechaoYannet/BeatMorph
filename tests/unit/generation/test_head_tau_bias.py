"""头的无条件 τ 偏置必须真的是一条 τ 通路（plan 07 §9-77）。

为什么要这条护栏：本轮的免拟合定位把根因钉在——τ 位置信息只加在**输入** token 上，
三个 local 自注意力层在训练中把它压成「沿 τ 不变」（未训练 rel 2.06 -> 训练后 0.00000），
而 `head_skip=False` 时头只剩 decode 的输出 ⇒ **头没有别的 τ 通路**。
`head_tau_bias` 就是补这条路：它必须让「完全沿 τ 不变的 token」也能产出沿 τ 变化的 ΔΛ(t)。
一旦形状或加和位置写错（例如加在 softplus 之后、或 reshape 广播到 batch/线轴），
这个实验臂就退化成基线，而且**不报错**。
"""

from __future__ import annotations

import torch
from torch import Tensor

from beatmorph.field.grid import FieldGrid
from beatmorph.generation.model import FieldHead, ModelConfig
from tests.unit.generation._builders import TEST_AUDIO_DIM, make_batch, make_grid

T_BINS = 12
X_BINS = 4
K_LINES = 3


def _config(**overrides) -> ModelConfig:
    base = {
        "d_model": 16,
        "n_heads": 2,
        "n_layers": 2,
        "global_period": 2,
        "k_max": 4,
        "audio_dim": TEST_AUDIO_DIM,
    }
    base.update(overrides)
    return ModelConfig(**base)


def _grid() -> FieldGrid:
    return make_grid(t_bins=T_BINS, x_bins=X_BINS)


def _flat_tokens(batch, grid: FieldGrid, *, constant: bool) -> Tensor:
    """(B,K,T,d) 的合成 token；constant=True 时沿 τ 逐位不变（= 训练后 decode 的形状）。"""
    generator = torch.Generator().manual_seed(0)
    tokens = torch.randn(
        batch.batch_size(),
        batch.n_lines(),
        grid.t_bins,
        16,
        generator=generator,
    )
    if constant:
        tokens = tokens.mean(dim=2, keepdim=True).expand(-1, -1, grid.t_bins, -1).contiguous()
    return tokens


def _profile(head: FieldHead, grid: FieldGrid, batch, *, constant: bool) -> Tensor:
    """沿 τ 的场质量剖面 (T,)。"""
    head.eval()
    with torch.no_grad():
        lam, _, _ = head(
            _flat_tokens(batch, grid, constant=constant),
            grid,
            line_mask=batch.line_mask_bool(),
            range_mask=batch.range_mask_bool(),
        )
    return lam.sum(dim=(0, 1, 3, 4, 5))


def test_flag_off_registers_no_parameter() -> None:
    head = FieldHead(_config(head_skip=False), _grid())
    assert head.tau_bias is None
    assert head.cell_tau_bias is None
    assert not [name for name, _ in head.named_parameters() if "tau_bias" in name]


def test_flag_on_registers_shaped_parameter() -> None:
    head = FieldHead(_config(head_skip=False, head_tau_bias=True, head_cell_tau_bias=True), _grid())
    assert head.tau_bias is not None
    assert tuple(head.tau_bias.shape) == (T_BINS,)
    assert head.cell_tau_bias is not None
    assert head.cell_tau_bias.shape[0] == T_BINS


def test_constant_tokens_cannot_vary_along_tau_without_the_bias() -> None:
    """对照组：token 沿 τ 不变 ⇒ 场质量沿 τ 也不变（就是 §9-68 量到的「精确均匀」）。"""
    grid = _grid()
    batch = make_batch(k=K_LINES, grid=grid)
    head = FieldHead(_config(head_skip=False), grid)
    profile = _profile(head, grid, batch, constant=True)
    assert float((profile - profile.mean()).abs().max()) < 1e-6


def test_tau_bias_restores_tau_variation_from_constant_tokens() -> None:
    """修法：有了无条件 τ 偏置，沿 τ 不变的 token 也能产出沿 τ 变化的场。"""
    grid = _grid()
    batch = make_batch(k=K_LINES, grid=grid)
    head = FieldHead(_config(head_skip=False, head_tau_bias=True), grid)
    assert head.tau_bias is not None
    with torch.no_grad():
        head.tau_bias.copy_(torch.randn_like(head.tau_bias))
    profile = _profile(head, grid, batch, constant=True)
    assert float((profile - profile.mean()).abs().max()) > 1e-3, "τ 通路没有接上"


def test_tau_bias_orders_tau_identically_across_lines() -> None:
    """偏置只沿 τ 变：不同 (b,k) 的 τ **排序**必须一致。

    注意不能断言「剖面逐位相同」——`delta = softplus(cum_head(token) + bias(t))`，
    每条的 `cum_head` 项不同，softplus 是非线性的 ⇒ 形状可差一个单调重标定。
    真正要锁的是**排序**：它决定 (窗,线) 组内的 τ AUC。reshape 若广播到线轴，排序就会散。
    """
    grid = _grid()
    batch = make_batch(k=K_LINES, grid=grid)
    head = FieldHead(_config(head_skip=False, head_tau_bias=True), grid)
    assert head.tau_bias is not None
    with torch.no_grad():
        head.tau_bias.copy_(torch.randn_like(head.tau_bias))
        lam, _, _ = head(
            _flat_tokens(batch, grid, constant=True),
            grid,
            line_mask=batch.line_mask_bool(),
            range_mask=batch.range_mask_bool(),
        )
    profile = lam.sum(dim=(3, 4, 5)).reshape(-1, T_BINS)  # (B*K, T)
    reference = torch.argsort(profile[0])
    for row in profile[1:]:
        assert bool(torch.equal(torch.argsort(row), reference)), "不同线的 τ 排序不一致"
