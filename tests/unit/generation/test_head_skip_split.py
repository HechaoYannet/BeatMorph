"""`head_skip` 的两条通路必须能**分开**（plan 07 §9-68）。

为什么要这条护栏：`head_skip=false` 修好了空间轴，却让模型对 τ 轴**完全无感**
（置换输出的 τ 轴：Δ = −7e-6，而 `head_skip=true` 的老基线是 +2.148）。
实验臂「要回 τ、不要回空间捷径」= `head_skip=false` + `head_cum_skip=true`；
一旦两条通路的解析写反（或 None 的语义漂移），这个实验就变成了别的东西，而且**不报错**。
"""

from __future__ import annotations

import torch

from beatmorph.generation.model import FieldHead, MaskedFieldModel, ModelConfig
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


def _head(config: ModelConfig) -> FieldHead:
    return FieldHead(config, make_grid(t_bins=T_BINS, x_bins=X_BINS))


def test_resolution_follows_head_skip_when_unspecified() -> None:
    head = _head(_config(head_skip=False))
    assert head.use_cell_skip is False
    assert head.use_cum_skip is False
    head = _head(_config(head_skip=True))
    assert head.use_cell_skip is True
    assert head.use_cum_skip is True


def test_split_switches_override_head_skip() -> None:
    head = _head(_config(head_skip=False, head_cum_skip=True, head_cell_skip=False))
    assert head.use_cum_skip is True
    assert head.use_cell_skip is False
    head = _head(_config(head_skip=True, head_cum_skip=False, head_cell_skip=True))
    assert head.use_cum_skip is False
    assert head.use_cell_skip is True


def _grad_norms(config: ModelConfig) -> tuple[float | None, float | None]:
    grid = make_grid(t_bins=T_BINS, x_bins=X_BINS)
    torch.manual_seed(0)
    model = MaskedFieldModel(config, grid)
    batch = make_batch(k=K_LINES, grid=grid, events=4, holds=0)
    output = model(batch, compute_loss=True)
    assert output.loss is not None
    output.loss.backward()
    cum = model.head.cum_skip.weight.grad
    cell = model.head.cell_skip.weight.grad
    return (
        None if cum is None else float(cum.norm()),
        None if cell is None else float(cell.norm()),
    )


def test_cum_only_arm_trains_only_the_cumulative_skip() -> None:
    cum, cell = _grad_norms(_config(head_skip=False, head_cum_skip=True, head_cell_skip=False))
    assert cum is not None, "τ 那条 skip 必须拿到梯度（否则实验臂是空的）"
    assert cum > 0.0
    assert cell is None or cell == 0.0, "空间 skip 必须完全不参与（§9-62 指控的就是它）"


def test_full_skip_arm_trains_both() -> None:
    cum, cell = _grad_norms(_config(head_skip=True))
    assert cum is not None
    assert cum > 0.0
    assert cell is not None
    assert cell > 0.0


def test_off_arm_trains_neither() -> None:
    cum, cell = _grad_norms(_config(head_skip=False))
    assert cum is None or cum == 0.0
    assert cell is None or cell == 0.0
