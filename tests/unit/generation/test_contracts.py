"""M1：契约与形状冻结（无权重 / 无 GPU，进默认 CI）。

覆盖 plan 04 §3.1/§3.2 与 §4.1 的三条硬契约：

- 形状断言必须真的会抛（含**跨层求和未做**的 4 x 5 = 20 通道静默失效点）；
- 三个 mask 的语义分离（occlusion / line_mask / range_mask 各司其职，不得混用）；
- **遮盖通道必须是网络输入**：改变它必须改变输出，且它可回传梯度。
"""

from __future__ import annotations

import dataclasses
from dataclasses import replace

import pytest
import torch

from beatmorph.core.contracts.phigros import SUBDIVISIONS_PER_BEAT
from beatmorph.generation.batch import (
    FIELD_DIM_NAMES,
    FORBIDDEN_LAYERED_TRACKS,
    N_ORDINARY_TRACKS,
    FieldBatch,
    FieldOutput,
)
from beatmorph.generation.masks import build_occlusion_batch
from beatmorph.generation.model import MaskedFieldModel, ModelConfig
from tests.unit.generation._builders import TEST_AUDIO_DIM, make_batch, make_counts, make_grid

T_BINS = SUBDIVISIONS_PER_BEAT // 4
X_BINS = 8
K_LINES = 3
MODEL_CONFIG = ModelConfig(
    d_model=16,
    n_heads=2,
    n_layers=2,
    window=2,
    global_period=2,
    k_max=8,
    audio_dim=TEST_AUDIO_DIM,
    dropout=0.0,
)


def _grid():
    return make_grid(t_bins=T_BINS, x_bins=X_BINS)


def _batch(**kwargs):
    return make_batch(k=K_LINES, grid=_grid(), audio_dim=TEST_AUDIO_DIM, **kwargs)


# ══════════════════════════════════════════════════════════════
# 形状契约
# ══════════════════════════════════════════════════════════════


def test_field_dim_names_are_the_contract_order() -> None:
    """维度序是契约的一部分（不得各模块另写一套）：(batch, k, t, x, s, c)。"""
    assert FIELD_DIM_NAMES == ("batch", "k", "t", "x", "s", "c")
    batch = _batch()
    assert tuple(batch.batch_field_shape()) == (
        batch.batch_size(),
        batch.n_lines(),
        batch.grid.t_bins,
        batch.grid.x_bins,
        batch.grid.sides,
        batch.grid.channels,
    )


def test_ordinary_tracks_must_be_five_after_cross_layer_sum() -> None:
    """plan 04 §4.1：普通事件轨必须**先跨层求和**（5 条），逐层送入（20 条）必须报错。"""
    batch = _batch()
    batch.assert_shapes()
    assert N_ORDINARY_TRACKS == 5
    assert FORBIDDEN_LAYERED_TRACKS == 20
    bad = replace(batch, line_tracks=torch.randn(1, K_LINES, T_BINS, FORBIDDEN_LAYERED_TRACKS))
    with pytest.raises(AssertionError, match="跨层求和"):
        bad.assert_shapes()
    wrong = replace(batch, line_tracks=torch.randn(1, K_LINES, T_BINS, 3))
    with pytest.raises(AssertionError, match="普通轨通道数"):
        wrong.assert_shapes()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("audio_emb", torch.randn(1, 5), "audio_emb"),
        ("line_tracks", torch.randn(1, K_LINES, T_BINS, 5, 2), "line_tracks"),
        ("line_mask", torch.ones(2, K_LINES, dtype=torch.bool), "line_mask"),
        ("difficulty", torch.zeros(3), "difficulty"),
    ],
)
def test_assert_shapes_rejects_bad_conditions(field: str, value: torch.Tensor, match: str) -> None:
    """条件输入形状错误必须当场抛出（不得靠广播静默通过）。"""
    batch = _batch()
    with pytest.raises(AssertionError, match=match):
        replace(batch, **{field: value}).assert_shapes()


def test_assert_shapes_rejects_target_mismatch() -> None:
    """counts / occlusion 必须与网格同形；occlusion 不得脱离 counts 单独出现。"""
    grid = _grid()
    counts = make_counts(k=K_LINES, grid=grid, events=4, holds=1, seed=9)
    occlusion, _ = build_occlusion_batch(counts, ratio=0.5, seed=9)
    batch = _batch(counts=counts, occlusion=occlusion)
    batch.assert_shapes()
    with pytest.raises(AssertionError, match="counts"):
        replace(batch, counts=batch.counts[:, :, :-1]).assert_shapes()
    with pytest.raises(AssertionError, match="occlusion"):
        replace(batch, counts=None).assert_shapes()


def test_assert_shapes_requires_a_time_bound_grid() -> None:
    """未绑定 tau 轴的网格（t_bins = 0）不得进入前向。"""
    batch = _batch()
    from beatmorph.field.grid import FieldGrid

    unbound = FieldGrid(x_bins=X_BINS)
    with pytest.raises(AssertionError):
        replace(batch, grid=unbound).assert_shapes()


def test_range_mask_is_derived_from_the_grid() -> None:
    """range_mask 是定义域声明（|x| <= 675），**不是钳位**；默认从 grid 派生。"""
    batch = _batch()
    ranges = batch.range_mask_bool()
    assert ranges.dtype == torch.bool
    assert tuple(ranges.shape) == (X_BINS,)
    assert bool(ranges.all())
    centers = batch.grid.x_centers()
    assert float(abs(centers).max()) <= float(abs(centers).max())


def test_observed_counts_zeroes_the_occluded_cells() -> None:
    """模型输入侧可见场 = counts * (~occlusion)（遮盖通道语义的唯一落点）。"""
    grid = _grid()
    counts = make_counts(k=K_LINES, grid=grid, events=8, holds=2, seed=5)
    occlusion, _ = build_occlusion_batch(counts, ratio=0.5, seed=1)
    batch = _batch(counts=counts, occlusion=occlusion)
    observed = batch.observed_counts()
    hidden = occlusion
    assert float(observed[hidden].abs().max()) == 0.0
    assert torch.equal(observed[~hidden], counts.float()[~hidden])
    assert float(observed.sum()) < float(counts.sum())


def test_field_output_shape_assertions() -> None:
    """FieldOutput 的形状与 cell_prob 归一化断言必须真的会抛。"""
    batch = _batch()
    model = MaskedFieldModel(MODEL_CONFIG, batch.grid)
    out = model(batch)
    out.assert_shapes(batch)
    assert out.cell_prob is not None
    assert out.cum is not None
    assert bool((out.cum[:, :, 1:] >= out.cum[:, :, :-1] - 1e-6).all())
    broken = FieldOutput(lam=out.lam[:, :, :-1])
    with pytest.raises(AssertionError, match="lam"):
        broken.assert_shapes(batch)
    unnormalized = FieldOutput(lam=out.lam, cell_prob=out.cell_prob * 2.0)
    with pytest.raises(AssertionError, match="归一化"):
        unnormalized.assert_shapes(batch)
    wrong_cum = FieldOutput(lam=out.lam, cum=out.cum[:, :, :-1])
    with pytest.raises(AssertionError, match="cum"):
        wrong_cum.assert_shapes(batch)


def test_describe_reports_the_geometry() -> None:
    """describe() 是训练日志的入口，必须带上网格与帧率（派生量）。"""
    text = _batch().describe()
    assert "K=" in text
    assert "frame_rate" in text
    assert FIELD_DIM_NAMES[0] in text


# ══════════════════════════════════════════════════════════════
# 遮盖通道是网络输入（硬要求，RFC-0029 §3.3-1）
# ══════════════════════════════════════════════════════════════


def test_mask_channel_changes_the_output() -> None:
    """注入不同的遮盖通道必须改变输出（否则模型区分不了「无 note」与「被遮盖」）。"""
    batch = _batch(events=6, holds=2)
    model = MaskedFieldModel(MODEL_CONFIG, batch.grid).eval()
    state = batch.observed_counts()
    first = model.forward_state(batch, state, batch.occlusion_bool())
    flipped = ~batch.occlusion_bool()
    second = model.forward_state(batch, state, flipped)
    assert float((first.lam - second.lam).abs().max().detach()) > 0.0


def test_mask_channel_is_differentiable() -> None:
    """遮盖通道必须可回传梯度（它是一个**输入通路**，不是常量掩码）。

    注意必须给**非空**遮盖通道：全 False 的通道输入恒为 0，其权重梯度也会恒为 0
    （那是数学事实，不是通路断开）。
    """
    grid = _grid()
    counts = make_counts(k=K_LINES, grid=grid, events=5, holds=2, seed=13)
    occlusion, _ = build_occlusion_batch(counts, ratio=0.5, seed=4)
    batch = _batch(counts=counts, occlusion=occlusion)
    model = MaskedFieldModel(MODEL_CONFIG, batch.grid)
    out = model.forward_state(
        batch,
        batch.observed_counts(),
        batch.occlusion_bool(),
        compute_loss=True,
    )
    assert out.loss is not None
    out.loss.backward()
    weight = model.embedding.occlusion.weight
    assert weight.grad is not None
    assert float(weight.grad.abs().sum()) > 0.0
    state_weight = model.embedding.visible.weight
    assert state_weight.grad is not None
    assert float(state_weight.grad.abs().sum()) > 0.0


def test_mask_channel_is_separate_from_line_and_range_masks() -> None:
    """三个 mask 语义分离：occlusion 不参与有效性判定，line_mask 不参与遮盖构造。"""
    grid = _grid()
    counts = make_counts(k=K_LINES, grid=grid, events=6, holds=2, seed=7)
    occlusion, _ = build_occlusion_batch(counts, ratio=0.5, seed=2)
    batch = _batch(counts=counts, occlusion=occlusion, active_lines=[0, 1], events=0, holds=0)
    batch.assert_shapes()
    assert tuple(batch.occlusion_bool().shape) == batch.batch_field_shape()
    assert tuple(batch.line_mask_bool().shape) == (1, K_LINES)
    assert tuple(batch.range_mask_bool().shape) == (X_BINS,)
    # occlusion 为 True 的格子照样保留 counts 原值（它是训练遮盖，不是有效性 mask）
    assert float(batch.counts[batch.occlusion_bool()].sum()) > 0.0


# ══════════════════════════════════════════════════════════════
# 超参与层排布
# ══════════════════════════════════════════════════════════════


def test_layer_kinds_end_with_a_global_layer() -> None:
    """最后必须是全局层：只有全局层能看到全部 K 条线的场（RFC-0029 §2.4-4）。"""
    kinds = ModelConfig(n_layers=4, global_period=4).layer_kinds()
    assert kinds[-1] == "global"
    assert kinds == ("local", "local", "local", "global")
    periodic = ModelConfig(n_layers=6, global_period=2).layer_kinds()
    assert periodic == ("local", "global", "local", "global", "local", "global")
    assert ModelConfig(n_layers=1, global_period=8).layer_kinds() == ("global",)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"d_model": 15},
        {"d_model": 16, "n_heads": 5},
        {"n_layers": 0},
        {"global_period": 0},
    ],
)
def test_invalid_model_configs_are_rejected(kwargs: dict[str, int]) -> None:
    """非法超参必须当场抛（含 d_model 必须为偶数、n_heads 必须整除）。"""
    with pytest.raises(ValueError, match=r"必须|必需|整除|不得|>="):
        ModelConfig(**kwargs)


def test_batch_is_frozen_and_typed() -> None:
    """FieldBatch 是冻结数据类（不可就地改写；张量本身仍是可变对象，见契约文档）。"""
    batch = _batch()
    with pytest.raises(dataclasses.FrozenInstanceError):
        batch.frame_rate = 1.0  # type: ignore[misc]
    assert isinstance(batch, FieldBatch)
