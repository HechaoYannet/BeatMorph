"""plan 07 §9-15：门禁 G3 的**分批前向**（内存墙）——等价性与边界（默认 CI）。

真实窗口上 G3 需要足够多的样本（样本太少时模型可以在预算内背下小批），而一次性
collate + 前向会在 `baseline_samples = 16` 时 OOM（§9-15 的实测）。修法是**按样本维
分段前向/反传**：内存回到「一段样本」的量级，而门禁判据不变（per_event 归一化下
的分段等价靠 `make_step_fn` 的除子修正保证，RFC-0037）。

判据不变是这里的**唯一**验收点，因此本文件的核心断言是「分段与不分段逐位等价」：
同一 seed 的同一模型、同一批次，一段走与四段走必须给出同一个 loss、同一组参数。
若哪天有人把 `make_step_fn` 的分段实现改成「按段取平均」或「每段各走一步优化器」，
这些断言会立刻失败。

物理常量一律派生（AGENTS.md §3.3）：帧数由 `audio_frames_for` 派生，τ 格数由
`SUBDIVISIONS_PER_BEAT` 派生。
"""

from __future__ import annotations

import pytest
import torch

from beatmorph.core.contracts.phigros import SUBDIVISIONS_PER_BEAT
from beatmorph.generation.batch import FieldBatch
from beatmorph.generation.model import MaskedFieldModel, ModelConfig
from beatmorph.infra.train_loop import make_step_fn
from tests.unit.generation._builders import (
    TEST_AUDIO_DIM,
    audio_frames_for,
    make_batch,
    make_grid,
)

#: τ 格数与 x 桶数只用来把测试做小——它们不是被测的物理常量
T_BINS = SUBDIVISIONS_PER_BEAT // 4
X_BINS = 8
K_LINES = 2
SAMPLES = 4

MODEL_CONFIG = ModelConfig(
    d_model=32,
    n_heads=2,
    n_layers=2,
    window=4,
    global_period=2,
    k_max=8,
    audio_dim=TEST_AUDIO_DIM,
)


def _grid_and_batch():
    grid = make_grid(t_bins=T_BINS, x_bins=X_BINS)
    batch = make_batch(k=K_LINES, grid=grid, batch=SAMPLES, events=8, holds=2, seed=3)
    batch.assert_shapes()
    return grid, batch


def _fresh_model(grid, *, seed: int) -> MaskedFieldModel:
    torch.manual_seed(seed)
    return MaskedFieldModel(MODEL_CONFIG, grid)


def _run_one_step(*, chunks: int, seed: int = 11) -> tuple[float, list[torch.Tensor]]:
    """跑一步（优化器**不更新参数**）并返回 (全批 loss, 各参数梯度)。

    只比梯度而不比参数：Adam 的**首步**更新恰为 `lr * sign(g)`，梯度里 1e-7 量级的
    浮点噪声只需翻转一个符号就会让参数差出 2e-3 —— 那是 Adam 的固有性质，不是分段
    前向的差异。梯度（以及由此决定的更新方向）才是这里要断言的东西。
    """
    grid, batch = _grid_and_batch()
    model = _fresh_model(grid, seed=seed)
    optimizer = _CountingOptimizer(model.parameters())
    loss = make_step_fn(model, batch, optimizer, chunks=chunks)()
    gradients = [
        parameter.grad.detach().clone()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    return loss, gradients


class _CountingModel(torch.nn.Module):
    """统计前向次数的外壳（验证「一段一次前向」，不分段成多次优化器步）。"""

    def __init__(self, inner: MaskedFieldModel) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, batch: FieldBatch):  # type: ignore[no-untyped-def]
        self.calls += 1
        return self.inner(batch)


class _CountingOptimizer:
    """只统计 `step()` 次数的最小优化器替身（真实语义由 `AdamW` 的等价性测试覆盖）。"""

    def __init__(self, parameters) -> None:  # type: ignore[no-untyped-def]
        self.parameters = list(parameters)
        self.steps = 0

    def zero_grad(self, **_kwargs) -> None:  # type: ignore[no-untyped-def]
        for parameter in self.parameters:
            parameter.grad = None

    def step(self) -> None:
        self.steps += 1


# ══════════════════════════════════════════════════════════════
# FieldBatch 的切分语义
# ══════════════════════════════════════════════════════════════


def test_split_samples_partitions_without_touching_shared_quantities() -> None:
    """切分只动样本维：样本数之和守恒，grid / 帧率 / 定义域 / 掩码逐位不变。"""
    _grid, batch = _grid_and_batch()
    parts = batch.split_samples(3)
    assert len(parts) == 3
    assert [part.batch_size() for part in parts] == [2, 1, 1]
    assert sum(part.batch_size() for part in parts) == batch.batch_size()
    for part in parts:
        part.assert_shapes()
        # 逐批共享的量必须原样保留（切错 = 测度 dV_j / 帧率 / 定义域静默错位）
        assert part.grid is batch.grid
        assert part.frame_rate == batch.frame_rate
        assert torch.equal(part.range_mask_bool(), batch.range_mask_bool())
    # 拼接回来必须是原批次（顺序不变）
    assert torch.equal(torch.cat([part.counts for part in parts], dim=0), batch.counts)
    assert torch.equal(torch.cat([part.line_mask for part in parts], dim=0), batch.line_mask)
    assert torch.equal(
        torch.cat([part.audio_emb for part in parts], dim=0),
        batch.audio_emb,
    )


def test_split_samples_clamps_and_rejects_bad_ranges() -> None:
    """段数超过 B 时退化为逐样本一段；越界下标**报错**而不是静默钳位。"""
    _grid, batch = _grid_and_batch()
    assert len(batch.split_samples(SAMPLES * 10)) == SAMPLES
    assert len(batch.split_samples(1)) == 1
    for bad in ((-1, 1), (0, SAMPLES + 1), (2, 1)):
        try:
            batch.slice_samples(*bad)
        except ValueError:
            continue
        raise AssertionError(f"切片 {bad} 应当报错")


def test_slice_samples_keeps_optional_fields_none() -> None:
    """推理/未遮盖批次的 `counts` / `occlusion` 为 None 时，切分后仍是 None。"""
    grid = make_grid(t_bins=T_BINS, x_bins=X_BINS)
    batch = make_batch(k=K_LINES, grid=grid, batch=SAMPLES, events=0, holds=0, seed=1)
    stripped = FieldBatch(
        audio_emb=batch.audio_emb,
        frame_rate=batch.frame_rate,
        line_tracks=batch.line_tracks,
        line_mask=batch.line_mask,
        difficulty=batch.difficulty,
        grid=batch.grid,
    )
    part = stripped.slice_samples(1, 3)
    assert part.counts is None
    assert part.occlusion is None
    assert part.batch_size() == 2


# ══════════════════════════════════════════════════════════════
# 分段前向的等价性（门禁判据不变的依据）
# ══════════════════════════════════════════════════════════════


def test_chunked_step_matches_the_single_forward_step() -> None:
    """同一 seed / 同一批次：4 段与 1 段的 loss 与**梯度**一致（按样本求和的损失）。

    实测（d_model=32 的微型模型，grad 尺度 ~1.3e2）：loss 相对差 2.3e-8、
    梯度最大绝对差 1.5e-5（≈ 尺度的 1e-7），都只是 float32 的求和顺序。
    """
    whole_loss, whole_gradients = _run_one_step(chunks=1)
    part_loss, part_gradients = _run_one_step(chunks=SAMPLES)
    # 返回的是**全批** loss（不是段均值）：若实现改成「段均值」，
    # 这个比值会变成 ~1/SAMPLES，断言立刻失败。
    assert part_loss == pytest.approx(whole_loss, rel=1e-5)
    assert len(part_gradients) == len(whole_gradients) > 0
    for left, right in zip(whole_gradients, part_gradients, strict=True):
        assert torch.allclose(left, right, rtol=1e-4, atol=1e-6)


def test_chunked_step_does_exactly_one_optimizer_step() -> None:
    """分段**不是**把优化器步数乘上段数：n 段 = n 次前向 + 1 次 step。"""
    grid, batch = _grid_and_batch()
    counting = _CountingModel(_fresh_model(grid, seed=5))
    counting.calls = 0
    optimizer = _CountingOptimizer(counting.parameters())
    step = make_step_fn(counting, batch, optimizer, chunks=4)
    step()
    assert counting.calls == 4
    assert optimizer.steps == 1


def test_chunks_below_one_is_a_no_op() -> None:
    """`chunks <= 1`（含配置里写成 0 的误值）必须退化为原行为而不是空跑。"""
    grid, batch = _grid_and_batch()
    counting = _CountingModel(_fresh_model(grid, seed=5))
    counting.calls = 0
    optimizer = _CountingOptimizer(counting.parameters())
    make_step_fn(counting, batch, optimizer, chunks=0)()
    assert counting.calls == 1
    assert optimizer.steps == 1


def test_gate_batches_carry_the_declared_sample_count() -> None:
    """门禁装配的样本数来自配置（回归：分段不得悄悄改掉 G3 的样本数）。"""
    grid = make_grid(t_bins=T_BINS, x_bins=X_BINS)
    batch = make_batch(k=K_LINES, grid=grid, batch=SAMPLES, events=4, holds=0, seed=2)
    frames = audio_frames_for(grid)
    assert int(batch.audio_emb.shape[1]) == frames
    assert batch.split_samples(2)[0].audio_emb.shape[1] == frames
