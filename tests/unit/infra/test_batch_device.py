"""`FieldBatch.to` / 设备搬运（门禁在 GPU 上跑真实批次的前提，默认 CI）。"""

from __future__ import annotations

import pytest
import torch

from beatmorph.core.contracts.phigros import SUBDIVISIONS_PER_BEAT
from beatmorph.infra.train_loop import make_step_fn
from tests.unit.generation._builders import TEST_AUDIO_DIM, make_batch, make_grid

T_BINS = SUBDIVISIONS_PER_BEAT // 4
X_BINS = 8
K_LINES = 2


def test_to_moves_every_tensor_and_keeps_shared_quantities() -> None:
    """`to("cpu")` 是恒等搬运：张量内容不变，grid / 帧率 / 定义域语义不变。"""
    grid = make_grid(t_bins=T_BINS, x_bins=X_BINS)
    batch = make_batch(k=K_LINES, grid=grid, batch=2, events=6, holds=1, seed=4)
    moved = batch.to("cpu")
    moved.assert_shapes()
    assert moved.grid is batch.grid
    assert moved.frame_rate == batch.frame_rate
    assert torch.equal(moved.counts, batch.counts)
    assert torch.equal(moved.audio_emb, batch.audio_emb)
    assert torch.equal(moved.range_mask_bool(), batch.range_mask_bool())
    assert all(parameter.device.type == "cpu" for parameter in [moved.counts, moved.line_mask])


def test_to_preserves_none_optionals() -> None:
    """无遮盖批次的 `occlusion` / `counts` 在搬运后仍是 None（不得变成空张量）。"""
    grid = make_grid(t_bins=T_BINS, x_bins=X_BINS)
    batch = make_batch(k=K_LINES, grid=grid, batch=1, events=0, holds=0, seed=1)
    bare = batch.__class__(
        audio_emb=batch.audio_emb,
        frame_rate=batch.frame_rate,
        line_tracks=batch.line_tracks,
        line_mask=batch.line_mask,
        difficulty=batch.difficulty,
        grid=batch.grid,
    )
    moved = bare.to("cpu")
    assert moved.counts is None
    assert moved.occlusion is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
def test_to_cuda_keeps_the_step_fn_working() -> None:  # pragma: no cover - 无 GPU 时不跑
    """有 GPU 时：批次与模型同设备，一步优化仍能返回有限 loss（门禁的真实用途）。"""
    from beatmorph.generation.model import MaskedFieldModel, ModelConfig

    grid = make_grid(t_bins=T_BINS, x_bins=X_BINS)
    batch = make_batch(k=K_LINES, grid=grid, batch=2, events=6, holds=1, seed=6).to("cuda")
    config = ModelConfig(
        d_model=32,
        n_heads=2,
        n_layers=2,
        window=4,
        global_period=2,
        k_max=8,
        audio_dim=TEST_AUDIO_DIM,
    )
    torch.manual_seed(0)
    model = MaskedFieldModel(config, grid).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = make_step_fn(model, batch, optimizer, chunks=2)()
    assert torch.isfinite(torch.tensor(loss))
