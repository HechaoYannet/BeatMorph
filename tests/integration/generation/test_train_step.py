"""集成：field 目标构建 -> generation 前向/损失/采样 的一条完整通路（默认 CI，CPU）。

覆盖 plan 04 §8「集成」一节：用**合成微型谱**（plan 03 的 `build_target`）跑一个完整
训练步与一次 `sample()`，全程无 GPU、无权重、无网络。

这条测试的价值在于**接缝**：桶内计数由 plan 03 产出（tau 网格 + 测度），
判定线事件轨由 contract 的 `JudgeLine.sum_track` 在 tau 轴上求值（**跨层求和**），
generation 只消费、不重算。任何一方偷偷自建换算都会在这里对不上。
"""

from __future__ import annotations

import pytest
import torch

from beatmorph.core.contracts.field import TYPE_CHANNELS
from beatmorph.core.contracts.phigros import (
    RPE_TRACK_FIELDS,
    SUBDIVISIONS_PER_BEAT,
    NoteType,
)
from beatmorph.core.contracts.tensors import MERT_FRAME_RATE_HZ
from beatmorph.field.grid import FieldGrid
from beatmorph.field.target import build_target
from beatmorph.generation.batch import N_ORDINARY_TRACKS, FieldBatch
from beatmorph.generation.losses import full_poisson_loss, masked_poisson_loss, occlusion_ratio
from beatmorph.generation.masks import build_occlusion_batch
from beatmorph.generation.model import MaskedFieldModel, ModelConfig
from beatmorph.generation.sampling import SamplingConfig, reveal_schedule, sample
from tests.unit.field._builders import make_bpm_points, make_chart, make_note
from tests.unit.generation._builders import TEST_AUDIO_DIM

pytestmark = pytest.mark.integration

T_BINS = SUBDIVISIONS_PER_BEAT
X_BINS = 8
K_LINES = 2
LEARNING_RATE = 0.05
STEPS = 40
MODEL_CONFIG = ModelConfig(
    d_model=32,
    n_heads=2,
    n_layers=2,
    window=4,
    global_period=2,
    k_max=8,
    audio_dim=TEST_AUDIO_DIM,
)


def _chart_and_grid() -> tuple[object, FieldGrid]:
    """合成谱：两条判定线、每个 tau 格一个 Tap（另加两个 Hold）。"""
    bpm_points = make_bpm_points((0.0, 120.0))
    notes = []
    for tau_index in range(T_BINS):
        notes.append(
            make_note(
                t=tau_index * 60.0 / 120.0 / SUBDIVISIONS_PER_BEAT,
                line_id=tau_index % K_LINES,
                position_x=(-1.0) ** tau_index * 100.0,
            ),
        )
    notes.append(make_note(t=0.0, line_id=0, note_type=NoteType.HOLD, hold_time=1.0))
    chart = make_chart(notes=notes, bpm_points=bpm_points, k=K_LINES)
    grid = FieldGrid(x_bins=X_BINS).for_chart(chart)
    grid.assert_grid()
    return chart, grid


def _line_tracks(chart, grid: FieldGrid) -> torch.Tensor:
    """(1, K, T_line, 5)：在 **tau** 轴上对每条线逐轨求值（跨层求和由契约负责）。"""
    taus = grid.tau_centers()
    tracks = torch.zeros(1, K_LINES, grid.t_bins, N_ORDINARY_TRACKS, dtype=torch.float32)
    for line_index, line in enumerate(chart.lines):
        for field_index, name in enumerate(RPE_TRACK_FIELDS):
            values = [line.sum_track(name, float(tau)) for tau in taus]
            tracks[0, line_index, :, field_index] = torch.tensor(values, dtype=torch.float32)
    return tracks


def _batch(*, occlusion_ratio_value: float | None) -> FieldBatch:
    chart, grid = _chart_and_grid()
    target = build_target(chart, grid)
    target.assert_conservation()
    counts = target.to_tensor()
    assert counts.dtype == torch.int16
    counts = counts.unsqueeze(0)
    occlusion = None
    if occlusion_ratio_value is not None:
        occlusion, _ = build_occlusion_batch(counts, ratio=occlusion_ratio_value, seed=3)
    frames = round(grid.total_seconds * MERT_FRAME_RATE_HZ)
    return FieldBatch(
        audio_emb=torch.randn(
            1, frames, TEST_AUDIO_DIM, generator=torch.Generator().manual_seed(0)
        ),
        frame_rate=MERT_FRAME_RATE_HZ,
        line_tracks=_line_tracks(chart, grid),
        line_mask=torch.ones(1, K_LINES, dtype=torch.bool),
        difficulty=torch.tensor([15.5]),
        grid=grid,
        counts=counts,
        occlusion=occlusion,
    )


def test_full_loss_on_real_targets_is_finite_and_trains() -> None:
    """r = 0 路径：L_full 有限，且 40 步 Adam 后 loss 必须下降（通路是通的）。"""
    batch = _batch(occlusion_ratio_value=None)
    batch.assert_shapes()
    model = MaskedFieldModel(MODEL_CONFIG, batch.grid)
    output = model(batch)
    assert output.loss is not None
    assert bool(torch.isfinite(output.loss))
    assert float(full_poisson_loss(output, batch).detach()) == pytest.approx(
        float(output.loss.detach()),
        rel=1e-5,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    first = float(output.loss.detach())
    last = first
    for _ in range(STEPS):
        optimizer.zero_grad()
        step_output = model(batch)
        assert step_output.loss is not None
        step_output.loss.backward()
        optimizer.step()
        last = float(step_output.loss.detach())
    assert last < first
    assert torch.isfinite(torch.tensor(last))


def test_masked_loss_and_ratio_are_consistent_on_real_targets() -> None:
    """r > 0 路径：遮盖比例由事件加权给出，且 HT 重标定后损失仍有限、可反传。"""
    batch = _batch(occlusion_ratio_value=0.5)
    ratio = occlusion_ratio(batch)
    assert 0.0 < ratio < 1.0
    model = MaskedFieldModel(MODEL_CONFIG, batch.grid)
    output = model(batch)
    assert output.loss is not None
    manual = masked_poisson_loss(output, batch)
    assert float(manual.detach()) == pytest.approx(float(output.loss.detach()), rel=1e-5)
    output.loss.backward()
    finite = [p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()]
    assert all(finite)


def test_sample_produces_a_field_on_the_chart_grid() -> None:
    """一次迭代并行解码：形状与网格一致、lambda 非负有限、揭示比例单调到 1。"""
    batch = _batch(occlusion_ratio_value=0.5)
    model = MaskedFieldModel(MODEL_CONFIG, batch.grid).eval()
    result = sample(model, batch, config=SamplingConfig(steps=3))
    assert tuple(result.lam.shape) == batch.batch_field_shape()
    assert float(result.lam.min()) >= 0.0
    assert bool(torch.isfinite(result.lam).all())
    revealed = result.diagnostics["revealed_fraction"].tolist()
    schedule = reveal_schedule(3, "cosine")
    # 实际揭示比例按**格数**取整（ceil(目标比例 x 总格数)），因此只要求贴近 schedule
    assert revealed == pytest.approx(list(schedule), rel=1e-3, abs=1e-3)
    assert revealed[-1] == pytest.approx(1.0)


def test_channel_contract_is_five_channels() -> None:
    """接缝自检：plan 03 的通道契约（4 类音符 + hold-end）必须被本模块原样消费。"""
    assert len(TYPE_CHANNELS) == len(NoteType) + 1
    batch = _batch(occlusion_ratio_value=None)
    assert batch.grid.channels == len(TYPE_CHANNELS)
    assert batch.line_tracks.shape[-1] == N_ORDINARY_TRACKS
