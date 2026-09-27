"""取批路径接入计划层（RFC-0034 S1/S3；默认 CI，无权重 / 无真实语料 / 不占显存）。

修复前：顺序由 `self._positions` 决定、覆盖率在取批时**就地累加** ⇒ worker 各持副本既
不可复现、覆盖率还会静默失真，于是「必须主进程串行取批」看起来成了硬要求。本文件钉死搬进
计划层之后的三条结论：

1. `data.workers` **不改变样本序列**（0 与 2 个 worker 逐位一致）；
2. `data.workers` **不改变覆盖率**（覆盖率是计划前缀的纯函数，与谁去物化无关）；
3. 续训按 `start_step` **O(1) 定位**，且 `data.workers` 是**续训无关字段**
   （进 `RESUME_IGNORED_KEYS`；反向对照：改 `data.t_window` 必须让指纹变化）。
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from itertools import islice
from pathlib import Path

import pytest

import beatmorph.data.dataset as dataset_mod
from beatmorph.infra.checkpoint import config_fingerprint
from beatmorph.infra.config.loading import config_from_mapping
from beatmorph.infra.train_loop import ManifestBatchSource

# spawn 出来的 worker 要能 import 本目录下的 stub（spawn 会把父进程的 sys.path 传给子进程）。
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _index_dataset(buckets: Sequence[tuple[int, int]]) -> object:
    """构造 stub 实例（**惰性导入**：模块级导入会排在 sys.path 修改之前，必然失败）。"""
    from _plan_worker_stub import IndexDataset

    return IndexDataset(buckets)


def _collate_fn() -> Callable[[Sequence[int]], list[int]]:
    """返回 stub 里的 collate **函数对象本身**：子进程按「模块 + 限定名」重新导入它。"""
    from _plan_worker_stub import identity_collate

    return identity_collate


PROVENANCE = {
    "source": "fixtures",
    "query": "n/a",
    "fetched_at": "t",
    "purpose": "train",
    "script": "s",
    "script_version": "v1",
}

#: 桶大小不均，且把 **1** 放进去（那是旧全局游标被清零的触发条件）。
BUCKETS: tuple[tuple[int, int], ...] = ((1, 1), (2, 5), (5, 7), (3, 30))


def _config(*, workers: int = 0, t_window: int = 192):
    return config_from_mapping(
        {
            "data": {
                "source": "synthetic",
                "max_samples": 4,
                "occlusion_ratio": 0.5,
                "k_max": 8,
                "t_window": t_window,
                "workers": workers,
                "provenance": PROVENANCE,
            },
            "model": {
                "d_model": 32,
                "n_heads": 2,
                "n_layers": 2,
                "window": 4,
                "global_period": 2,
                "k_max": 8,
                "audio_dim": 16,
            },
            "optim": {"lr": 1e-3, "batch_size": 1, "max_steps": 4},
        }
    )


def _source(monkeypatch: pytest.MonkeyPatch, *, workers: int, seed: int = 3) -> ManifestBatchSource:
    monkeypatch.setattr(dataset_mod, "collate_field_batch", _collate_fn())
    source = ManifestBatchSource(_config(workers=workers), split="train", seed=seed)
    source._dataset = _index_dataset(BUCKETS)  # type: ignore[assignment]
    return source


def _take(source: ManifestBatchSource, steps: int, **kwargs: object) -> list[int]:
    stream = source.batches(**kwargs)  # type: ignore[arg-type]
    return [index for batch in islice(stream, steps) for index in batch]


def test_workers_do_not_change_the_sample_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    """`data.workers=0` 与 `=2` 必须给出**逐位一致**的样本序列，覆盖率也一致。

    这是「`data.workers` 是语义无关字段」的实证，也是它敢进 `RESUME_IGNORED_KEYS` 的依据。
    """
    sync = _source(monkeypatch, workers=0)
    parallel = _source(monkeypatch, workers=2)
    from_sync = _take(sync, 24)
    from_parallel = _take(parallel, 24)
    assert from_sync == from_parallel
    assert sync.coverage() == parallel.coverage()
    assert sync.coverage()["windows_seen"] == 24.0


def test_first_batch_is_reused_not_redrawn(monkeypatch: pytest.MonkeyPatch) -> None:
    """`first` 交回来的批就是第 1 步的批（不能白抽一个窗口）——同步与并行路径都是。"""
    for workers in (0, 2):
        source = _source(monkeypatch, workers=workers)
        probe = source.batch(masked=True)
        indices = _take(source, 3, start_step=1, first=probe)
        assert indices[0] == probe[0]
        assert source.coverage()["windows_seen"] == 3.0


def test_resume_seeks_the_plan_instead_of_replaying(monkeypatch: pytest.MonkeyPatch) -> None:
    """`start_step>1` 从计划里**定位**（O(1)），结果与从头迭代到该步逐位一致。"""
    full = _take(_source(monkeypatch, workers=0), 9)
    resumed_source = _source(monkeypatch, workers=0)
    resumed = _take(resumed_source, 5, start_step=5)
    assert resumed == full[4:9]
    assert resumed_source.coverage()["windows_seen"] == 9.0


def test_first_with_resume_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """`first` 与 `start_step>1` 同时给出必须报错：那会把游标定位与探测批对不上。"""
    source = _source(monkeypatch, workers=0)
    probe = source.batch(masked=True)
    with pytest.raises(ValueError, match="first 只在 start_step == 1"):
        _take(source, 1, start_step=4, first=probe)


def test_workers_is_resume_neutral_but_semantic_fields_are_not() -> None:
    """`data.workers` 不参与续训指纹；`data.t_window`（真语义字段）必须参与。"""
    assert config_fingerprint(_config(workers=0)) == config_fingerprint(_config(workers=3))
    assert config_fingerprint(_config(t_window=192)) != config_fingerprint(_config(t_window=96))


def test_batches_stream_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    """取批是**流式**的：没有它就没有「GPU 算当前批的同时构造下一批」这件事（RFC-0034 动机）。"""
    source = _source(monkeypatch, workers=0)
    stream = source.batches()
    assert source.coverage()["windows_seen"] == 0.0
    next(stream)
    assert source.coverage()["windows_seen"] == 1.0
