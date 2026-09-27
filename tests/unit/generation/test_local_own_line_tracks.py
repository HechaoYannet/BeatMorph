"""RFC-0032：局部层的事件轨条件必须是**本线轨**，不是全部 K 条线。

为什么值得一条契约级断言：把 tracks.repeat_interleave(n_lines) 换成本线 reshape 只是
一行改动，但它决定「局部层能不能看到别的判定线」。**改回去不会报任何错**，只会让显存
重新变成 O(K^2)（实测 K=20 时 2.47 -> 1.31 GiB、K=128 时根本跑不动）。因此这里直接
钉住 cross_tracks 的 K/V 长度：

- 局部层：K/V = 本线 T_line 个轨 token，batch = B*K（每条线一份 query）；
- 全局层：K/V = 全部 K*T_line 个轨 token，batch = B（跨线只发生在这里）。
"""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor, nn

from beatmorph.field.grid import FieldGrid
from beatmorph.generation.model import MaskedFieldModel, ModelConfig
from tests.unit.generation._builders import TEST_AUDIO_DIM, make_batch, make_grid

T_BINS = 32
X_BINS = 4
K_LINES = 4
T_LINE = 8
MODEL_CONFIG = ModelConfig(
    d_model=16,
    n_heads=2,
    n_layers=4,
    window=4,
    global_period=2,
    k_max=8,
    audio_dim=TEST_AUDIO_DIM,
)


class _TrackSpy(nn.Module):
    """包住 cross_tracks，记下每次调用的 (层类型, batch, K/V 长度)。"""

    def __init__(self, inner: nn.Module, kind: str, log: list[tuple[str, int, int]]) -> None:
        super().__init__()
        self.inner = inner
        self.kind = kind
        self.log = log

    def forward(self, query: Tensor, key: Tensor, value: Tensor, **kwargs: Any) -> Any:
        self.log.append((self.kind, int(key.shape[0]), int(key.shape[1])))
        return self.inner(query, key, value, **kwargs)


def _spied_model(grid: FieldGrid) -> tuple[MaskedFieldModel, list[tuple[str, int, int]]]:
    model = MaskedFieldModel(MODEL_CONFIG, grid)
    log: list[tuple[str, int, int]] = []
    for layer in model.layers:
        recorder = _TrackSpy(layer.cross_tracks, layer.kind, log)
        layer.cross_tracks = cast(nn.MultiheadAttention, recorder)
    return model, log


def _run(grid: FieldGrid, k: int) -> list[tuple[str, int, int]]:
    batch = make_batch(k=k, grid=grid, t_line=T_LINE, events=4, holds=0)
    model, log = _spied_model(grid)
    model.eval()
    with torch.no_grad():
        model(batch)
    return log


def test_local_layers_see_only_their_own_line_tracks() -> None:
    """局部层的 K/V 长度 == T_line 且 batch == K（每条线一份 query）。"""
    log = _run(make_grid(t_bins=T_BINS, x_bins=X_BINS), K_LINES)
    local = [entry for entry in log if entry[0] == "local"]
    assert local, "配置里必须至少有一个局部层，否则这条测试没测到东西"
    for _, batch_dim, kv_len in local:
        assert kv_len == T_LINE, f"局部层的 K/V 必须是本线轨（{T_LINE}），得到 {kv_len}"
        assert (
            batch_dim == K_LINES
        ), f"局部层每条线一份 query，期望 batch={K_LINES}，得到 {batch_dim}"


def test_global_layers_still_see_all_line_tracks() -> None:
    """跨线没有被删掉：全局层仍然拿到全部 K 条线的轨（RFC-0029 §2.4-4）。"""
    log = _run(make_grid(t_bins=T_BINS, x_bins=X_BINS), K_LINES)
    glob = [entry for entry in log if entry[0] == "global"]
    assert glob, "配置里必须至少有一个全局层（layer_kinds 保证最后一层是全局层）"
    for _, batch_dim, kv_len in glob:
        assert kv_len == K_LINES * T_LINE
        assert batch_dim == 1


def test_local_track_length_does_not_grow_with_k() -> None:
    """回归护栏：K 翻倍**不得**改变局部层 K/V 的长度（旧实现是 K*T_line）。"""
    grid = make_grid(t_bins=T_BINS, x_bins=X_BINS)
    lengths = [next(entry[2] for entry in _run(grid, k) if entry[0] == "local") for k in (2, 6)]
    assert lengths == [T_LINE, T_LINE]
