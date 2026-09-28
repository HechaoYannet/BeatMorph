"""窗口预切缓存：**逐位一致**是唯一的验收判据（默认 CI，无权重 / 无 GPU / 无真实语料）。

本缓存把「一个窗口的样本」在数据处理阶段物化，训练期只做 mmap 读取。它是一层
**纯派生加速器**：如果它读出来的样本与原路径差一个字节，那它就不是优化而是缺陷。
因此本文件的中心断言只有一条 —— 同一 index，`WindowCacheReader.sample` 与
`ChartPairDataset.__getitem__` 的**每一个字段**都相等（张量用 `torch.equal`）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from beatmorph.data.window_build import build_window_cache
from beatmorph.data.window_cache import (
    WindowCacheError,
    WindowCacheReader,
    read_index,
    window_cache_directory,
    windows_on_disk,
)
from tests.unit.data.test_dataset import _dataset, _spec

SPECS = [_spec(), _spec(), _spec()]


def _assert_same_sample(original: object, cached: object, index: int) -> None:
    """逐字段比对两个 `PairSample`（张量用 `torch.equal`）。"""
    for field in (
        "audio_emb",
        "line_tracks",
        "line_mask",
        "difficulty",
        "counts",
        "occlusion",
    ):
        left = getattr(original, field)
        right = getattr(cached, field)
        assert isinstance(left, torch.Tensor)
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype, f"index={index} 字段 {field} dtype 不同"
        assert left.shape == right.shape, f"index={index} 字段 {field} 形状不同"
        assert torch.equal(left, right), f"index={index} 字段 {field} **不是逐位一致**"
    left_grid, right_grid = original.grid, cached.grid
    assert left_grid.x_bins == right_grid.x_bins
    assert left_grid.t_bins == right_grid.t_bins
    assert left_grid.bpm_points == right_grid.bpm_points
    for field in (
        "frame_rate",
        "chart_id",
        "song_key",
        "split",
        "pair_index",
        "window_index",
        "tau_start",
        "chart_path",
        "feature_key",
        "audio_frame_start",
        "audio_padded_frames",
    ):
        assert getattr(original, field) == getattr(
            cached, field
        ), f"index={index} 字段 {field} 不同"


def _reader(tmp_path: Path, dataset: object) -> WindowCacheReader:
    report = build_window_cache(
        dataset.config,  # type: ignore[attr-defined]
        root=tmp_path / "wc",
        seed=dataset.config.seed,  # type: ignore[attr-defined]
        jobs=1,
    )
    index = read_index(report.directory, fingerprint=report.fingerprint)
    assert index is not None
    return WindowCacheReader(
        report.directory,
        index,
        dataset.rows(),
        split=dataset.config.split,  # type: ignore[attr-defined]
    )


def test_cached_samples_are_bit_identical_to_the_original_path(tmp_path: Path) -> None:
    """**核心验收**：每一个窗口的每一个字段都与原路径逐位一致。"""
    dataset = _dataset(tmp_path, SPECS)
    assert len(dataset) > 0
    reader = _reader(tmp_path, dataset)
    assert len(reader) == len(dataset)
    for index in range(len(dataset)):
        _assert_same_sample(dataset[index], reader.sample(index), index)


def test_cache_is_deterministic_across_reads(tmp_path: Path) -> None:
    """同一 index 读两次逐位一致（缓存自己不能引入随机性）。"""
    dataset = _dataset(tmp_path, SPECS)
    reader = _reader(tmp_path, dataset)
    for index in range(len(dataset)):
        _assert_same_sample(reader.sample(index), reader.sample(index), index)


def test_index_is_refused_on_fingerprint_mismatch(tmp_path: Path) -> None:
    """指纹不符必须**拒绝使用**（返回 None ⇒ 调用方回退原路径），不是静默降级。"""
    dataset = _dataset(tmp_path, SPECS)
    report = build_window_cache(
        dataset.config, root=tmp_path / "wc", seed=dataset.config.seed, jobs=1
    )
    assert read_index(report.directory, fingerprint=report.fingerprint) is not None
    assert read_index(report.directory, fingerprint="deadbeef") is None


def test_shards_on_disk_add_up_to_the_index(tmp_path: Path) -> None:
    """shard 落盘数必须等于索引声明的窗口数（半成品不允许带 `index.json`）。"""
    dataset = _dataset(tmp_path, SPECS)
    report = build_window_cache(
        dataset.config, root=tmp_path / "wc", seed=dataset.config.seed, jobs=1, shard_windows=2
    )
    index = read_index(report.directory, fingerprint=report.fingerprint)
    assert index is not None
    assert index.n_shards > 1  # 否则本测试没验到跨 shard
    assert windows_on_disk(report.directory, index) == len(dataset)


def test_build_refuses_a_non_token_block_occlusion() -> None:
    """遮盖若不是 token 区块结构，构建期必须**抛**（宁可在构建期炸，也不喂错掩码）。"""
    from beatmorph.data.window_cache import _token_view

    k, t, x, s, c = 2, 3, 2, 1, 3
    token = np.zeros((k, t), dtype=np.bool_)
    token[0, 1] = True
    block = np.broadcast_to(token.reshape(k, t, 1, 1, 1), (k, t, x, s, c)).copy()
    assert _token_view(block, k=k, t_bins=t).shape == (k, t)
    broken = block.copy()
    broken[0, 0, 0, 0, 0] = True  # 只翻一个格 ⇒ 同一 token 内不一致 ⇒ 不再是区块结构
    with pytest.raises(WindowCacheError, match="token 区块结构"):
        _token_view(broken, k=k, t_bins=t)


def test_window_cache_directory_is_fingerprinted(tmp_path: Path) -> None:
    """目录名带指纹 ⇒ 换配置自然分叉，不会误读旧产物。"""
    dataset = _dataset(tmp_path, SPECS)
    report = build_window_cache(
        dataset.config, root=tmp_path / "wc", seed=dataset.config.seed, jobs=1
    )
    expected = window_cache_directory(tmp_path / "wc", dataset.config.split, report.fingerprint)
    assert report.directory == expected
