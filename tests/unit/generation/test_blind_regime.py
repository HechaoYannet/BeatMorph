"""生成制度（`visible_input=false`）：训练输入必须等于**推理第一步**的输入。

为什么值得一条契约级断言（plan 07 §9-66）：训练里的可见场永远是「随机 50% 的事件」，
而迭代并行解码的第一步是「什么都还没确证」（可见场全 0、遮盖通道全 1）。
关掉 `visible_input` 就等于把训练送进那个制度 —— 一旦它被改回 True（或忘记置零），
「模型只能靠音乐出谱」这件事就悄悄不成立了，而且**不会报任何错**。
"""

from __future__ import annotations

import torch

from beatmorph.field.grid import FieldGrid
from beatmorph.generation.model import MaskedFieldModel, ModelConfig
from tests.unit.generation._builders import TEST_AUDIO_DIM, make_batch, make_counts, make_grid

T_BINS = 16
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
        "head_skip": False,
    }
    base.update(overrides)
    return ModelConfig(**base)


def _model(config: ModelConfig, grid: FieldGrid) -> MaskedFieldModel:
    torch.manual_seed(0)
    return MaskedFieldModel(config, grid).eval()


def _forward(
    model: MaskedFieldModel,
    grid: FieldGrid,
    state: torch.Tensor,
    occlusion: torch.Tensor,
):
    batch = make_batch(k=K_LINES, grid=grid, events=3, holds=0)
    with torch.no_grad():
        return model.forward_state(batch, state, occlusion).lam


def test_blind_model_ignores_the_visible_field_and_the_occlusion_channel() -> None:
    grid = make_grid(t_bins=T_BINS, x_bins=X_BINS)
    shape = (1, K_LINES, T_BINS, X_BINS, grid.sides, grid.channels)
    counts = make_counts(batch=1, k=K_LINES, grid=grid, events=4, holds=0, seed=1)
    occl_a = torch.zeros(shape, dtype=torch.bool)
    occl_b = torch.ones(shape, dtype=torch.bool)
    model = _model(_config(visible_input=False), grid)
    lam_a = _forward(model, grid, counts.to(torch.float32), occl_a)
    lam_b = _forward(model, grid, torch.zeros(shape), occl_b)
    assert torch.allclose(
        lam_a, lam_b, rtol=0.0, atol=0.0
    ), "visible_input=false 时可见场与遮盖通道都必须被忽略（否则不是生成制度）"


def test_seeing_model_does_depend_on_the_visible_field() -> None:
    grid = make_grid(t_bins=T_BINS, x_bins=X_BINS)
    shape = (1, K_LINES, T_BINS, X_BINS, grid.sides, grid.channels)
    counts = make_counts(batch=1, k=K_LINES, grid=grid, events=4, holds=0, seed=1)
    occl = torch.zeros(shape, dtype=torch.bool)
    model = _model(_config(visible_input=True), grid)
    lam_a = _forward(model, grid, counts.to(torch.float32), occl)
    lam_b = _forward(model, grid, torch.zeros(shape), occl)
    assert not torch.allclose(lam_a, lam_b), "默认（补全制度）必须看得见可见场"


def test_blind_flag_matches_the_config_field() -> None:
    grid = make_grid(t_bins=T_BINS, x_bins=X_BINS)
    assert MaskedFieldModel(_config(visible_input=False), grid).embedding.blind is True
    assert MaskedFieldModel(_config(visible_input=True), grid).embedding.blind is False


def test_blind_model_is_deterministic() -> None:
    grid = make_grid(t_bins=T_BINS, x_bins=X_BINS)
    shape = (1, K_LINES, T_BINS, X_BINS, grid.sides, grid.channels)
    counts = make_counts(batch=1, k=K_LINES, grid=grid, events=4, holds=0, seed=1)
    occl = torch.ones(shape, dtype=torch.bool)
    model = _model(_config(visible_input=False), grid)
    first = _forward(model, grid, counts.to(torch.float32), occl)
    second = _forward(model, grid, counts.to(torch.float32), occl)
    assert torch.equal(first, second)
