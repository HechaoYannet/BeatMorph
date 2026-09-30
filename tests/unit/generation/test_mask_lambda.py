"""`model.mask_lambda` 的护栏（plan 07 §9-74）。

这条硬约束的语义只有一句话：**未被遮盖的格子 λ 必须恰为 0**（最优解已知如此）。
写错一个方向（比如乘 `~occlusion`）就等于把整个被监督集合清零，loss 会直接爆成 inf ——
所以必须有一条测试把它钉住。
"""

from __future__ import annotations

from dataclasses import replace

import torch

from beatmorph.generation.model import MaskedFieldModel, ModelConfig
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


def _half_masked(batch):
    """把**前半段 τ** 的格子全部遮盖（默认夹具是全可见，测不出这条约束）。"""
    mask = torch.zeros_like(batch.occlusion_bool())
    mask[:, :, : T_BINS // 2] = True
    return replace(batch, occlusion=mask)


def _pair(flag: bool):
    grid = make_grid(t_bins=T_BINS, x_bins=X_BINS)
    batch = _half_masked(make_batch(grid=grid, k=K_LINES))
    on = MaskedFieldModel(_config(mask_lambda=True), grid).eval()
    off = MaskedFieldModel(_config(mask_lambda=False), grid).eval()
    on.load_state_dict(off.state_dict())
    return grid, batch, on, off


def test_the_flag_changes_the_output_at_all() -> None:
    _grid, batch, on, off = _pair(True)
    with torch.no_grad():
        a = on(batch, compute_loss=False).lam
        b = off(batch, compute_loss=False).lam
    assert not torch.equal(a, b), "mask_lambda=True 竟然没有改变输出"


def test_mask_lambda_zeroes_exactly_the_unmasked_cells() -> None:
    _grid, batch, on, _off = _pair(True)
    with torch.no_grad():
        lam = on(batch, compute_loss=False).lam
    occlusion = batch.occlusion_bool()
    assert torch.all(lam[~occlusion] == 0.0), "未被遮盖的格子必须恰为 0"
    assert torch.all(lam[occlusion] >= 0.0), "被遮盖的格子不得为负"
    assert bool(lam[occlusion].gt(0).any()), "被遮盖处也不该全被清零"


def test_mask_lambda_is_the_identity_when_nothing_is_masked() -> None:
    """**全可见**批（G3 门禁 / 全事件口径）下必须是恒等 —— 否则事件项会被打成 +inf。

    这是本轮实测抓到的真缺陷：不带这个守卫时门禁直接退出码 7（「lambda 出现 NaN」）。
    """
    _grid, batch, on, off = _pair(True)
    visible = replace(batch, occlusion=torch.zeros_like(batch.occlusion_bool()))
    with torch.no_grad():
        a = on(visible, compute_loss=False).lam
        b = off(visible, compute_loss=False).lam
    assert torch.equal(a, b), "全可见时 mask_lambda 必须退回恒等"


def test_mask_lambda_is_the_identity_when_everything_is_masked() -> None:
    """生成制度（遮盖通道全 1）下该约束必须是恒等 —— 否则推理第一步会被它改坏。"""
    _grid, batch, on, off = _pair(True)
    swapped = replace(batch, occlusion=torch.ones_like(batch.occlusion_bool()))
    with torch.no_grad():
        a = on(swapped, compute_loss=False).lam
        b = off(swapped, compute_loss=False).lam
    assert torch.equal(a, b), "全遮盖时 mask_lambda 必须是恒等"
