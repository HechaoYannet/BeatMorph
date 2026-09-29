"""`track_input=false`：条件里不再给事件轨**内容**（plan 07 §9-67）。

为什么值得一条契约级断言：e2e 的判定线事件轨是从**该曲真谱**复制的，而人类谱的可读性做法
（线在音符时刻闪现/变速）让这些轨本身编码了音符时刻——实测「仅事件轨」的组内 AUC 0.643，
比音频（0.576）还高。关掉 `track_input` 就是「把答案从条件里拿掉」的实验臂；
一旦它被改回 True（或位置/线身份被一起清零），实验的含义就变了，而且**不会报任何错**。
"""

from __future__ import annotations

from dataclasses import replace

import torch

from beatmorph.generation.model import MaskedFieldModel, ModelConfig
from tests.unit.generation._builders import TEST_AUDIO_DIM, make_batch, make_grid

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


def _forward(config: ModelConfig, tracks: torch.Tensor) -> torch.Tensor:
    grid = make_grid(t_bins=T_BINS, x_bins=X_BINS)
    torch.manual_seed(0)
    model = MaskedFieldModel(config, grid).eval()
    batch = make_batch(k=K_LINES, grid=grid, events=3, holds=0)
    batch = replace(batch, line_tracks=tracks)
    with torch.no_grad():
        return model(batch, compute_loss=False).lam


def test_no_track_content_makes_the_output_invariant_to_the_tracks() -> None:
    grid = make_grid(t_bins=T_BINS, x_bins=X_BINS)
    torch.manual_seed(1)
    tracks_a = torch.randn(1, K_LINES, T_BINS, 5)
    tracks_b = torch.randn(1, K_LINES, T_BINS, 5)
    lam_a = _forward(_config(track_input=False), tracks_a)
    lam_b = _forward(_config(track_input=False), tracks_b)
    assert torch.equal(
        lam_a, lam_b
    ), "track_input=false 时输出必须与事件轨内容无关（否则不是「拿掉答案」的实验臂）"
    _ = grid


def test_track_content_is_used_by_default() -> None:
    torch.manual_seed(1)
    tracks_a = torch.randn(1, K_LINES, T_BINS, 5)
    tracks_b = torch.randn(1, K_LINES, T_BINS, 5)
    lam_a = _forward(_config(track_input=True), tracks_a)
    lam_b = _forward(_config(track_input=True), tracks_b)
    assert not torch.equal(lam_a, lam_b), "默认必须看得见事件轨内容"
