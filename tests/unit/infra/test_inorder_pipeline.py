"""乱序交付 + 主进程按槽位重排（RFC-0035；默认 CI，无权重 / 无真实语料 / 不占显存）。

背景（plan 07 §9-49，决定性实测）：`torch` 的 `DataLoader` 默认 `in_order=True`，而它的
`_try_put_index` **只在按序交付时**被调用一次 ⇒ 队头一条慢窗口（真实窗口 p50 0.57 s /
p99 5.9 s）会让全部 worker 干完手上的活**集体等索引**：8 worker 只跑出 4.8-5.2 核 /
6.85 窗口/s，`py-spy dump` 全停在 `index_queue.get()`。改成 `in_order=False` 后每交付
一批就补发一个索引 ⇒ 8.05-8.15 核 / 9.85-10.08 窗口/s。

代价是**交付顺序变成任意排列**，而 `self._cursor`（覆盖率在线标量 / 续训 O(1) 定位 /
`data.workers` 语义中性）只认计划前缀。本文件钉死修法：每批自带槽位终点标签，主进程
缓冲重排回计划顺序 ⇒ 吞吐修好，而**样本序列与 `workers=0` 仍逐位一致**。

为什么用假 `DataLoader` 而不是「造一个慢窗口」：真实调度下的乱序是**概率性**的（慢窗口
能不能排到队头取决于线程时序），那样的测试会时灵时不灵；假 loader 把**最坏情形**（整段
倒序交付）变成确定性输入，且能断言 `in_order` 真的传对了。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from itertools import islice
from typing import Any, ClassVar

import pytest
import torch.utils.data

import beatmorph.data.dataset as dataset_mod
import beatmorph.infra.train_loop as train_loop_mod
from beatmorph.infra.config.loading import config_from_mapping
from beatmorph.infra.train_loop import ManifestBatchSource, SlotTag, _SlotTaggedDataset

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


class _IndexDataset:
    """样本 = 窗口下标本身（被测的是**取批顺序与记账**，不是样本组装）。"""

    def __init__(self, buckets: Sequence[tuple[int, int]]) -> None:
        self.keys: list[tuple[int, int, float]] = []
        self.rows: list[int] = []
        for bucket, (n_rows, per_row) in enumerate(buckets):
            for row in range(n_rows):
                for _ in range(per_row):
                    self.keys.append((8, 4, 100.0 + bucket))
                    self.rows.append(1000 * bucket + row)

    def __len__(self) -> int:
        return len(self.keys)

    def grid_key(self, index: int) -> tuple[int, int, float]:
        return self.keys[index]

    def window_row_index(self, index: int) -> int:
        return self.rows[index]

    def __getitem__(self, index: int) -> int:
        return index


def _identity_collate(samples: Sequence[Any]) -> list[Any]:
    return list(samples)


def _config(*, workers: int = 0) -> Any:
    return config_from_mapping(
        {
            "data": {
                "source": "synthetic",
                "max_samples": 4,
                "occlusion_ratio": 0.5,
                "k_max": 8,
                "t_window": 192,
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


def _source(monkeypatch: pytest.MonkeyPatch, *, workers: int) -> ManifestBatchSource:
    monkeypatch.setattr(dataset_mod, "collate_field_batch", _identity_collate)
    source = ManifestBatchSource(_config(workers=workers), split="train", seed=3)
    source._dataset = _IndexDataset(BUCKETS)  # type: ignore[assignment]
    return source


def _take(source: ManifestBatchSource, steps: int) -> list[int]:
    stream = source.batches()
    return [index for batch in islice(stream, steps) for index in batch]


class _FakeLoader:
    """假 `DataLoader`：按 `mode` 决定**交付顺序**，并记录构造参数供断言。

    Args:
        mode: "reversed" = 整段倒序交付（最坏情形）；"drop_one" = 丢掉一批（模拟丢批）。
    """

    seen_kwargs: ClassVar[dict[str, Any]] = {}
    plan_order: ClassVar[list[int]] = []
    delivery_order: ClassVar[list[int]] = []

    def __init__(self, dataset: Any, *, mode: str = "reversed", **kwargs: Any) -> None:
        type(self).seen_kwargs = dict(kwargs)
        self._dataset = dataset
        self._mode = mode
        spans = kwargs["batch_sampler"]
        collate: Callable[[Sequence[Any]], tuple[int, Any]] = kwargs["collate_fn"]
        self._batches = [collate([self._dataset[i] for i in indices]) for indices in spans]
        type(self).plan_order = [stop for stop, _ in self._batches]

    def __iter__(self) -> Iterator[tuple[int, Any]]:
        batches = self._batches
        if self._mode == "drop_one" and len(batches) > 3:
            batches = batches[:3] + batches[4:]
        delivered = list(reversed(batches))
        type(self).delivery_order = [stop for stop, _ in delivered]
        return iter(delivered)


def test_slot_tagged_dataset_translates_and_forwards() -> None:
    """负编码 = 槽位标签；非负 = 真实窗口下标原样转发。"""
    dataset = _SlotTaggedDataset(_IndexDataset(BUCKETS))
    assert len(dataset) == 136
    assert dataset[0] == 0
    assert dataset[-1] == SlotTag(stop=0)
    assert dataset[-137] == SlotTag(stop=136)


def test_collate_requires_the_slot_tag() -> None:
    """缺标签必须**报错**而不是猜：记账一旦退化成「假设按序」就是本轮要拆掉的那个假设。"""
    with pytest.raises(TypeError, match="缺少 SlotTag"):
        train_loop_mod._collate_with_slot(_identity_collate, [1, 2, 3])


def test_collate_splits_tag_from_samples() -> None:
    stop, samples = train_loop_mod._collate_with_slot(_identity_collate, [SlotTag(stop=7), 5, 6])
    assert stop == 7
    assert samples == [5, 6]


def test_reversed_delivery_is_reordered_to_plan_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """**核心护栏**：最坏情形（整段倒序交付）下，样本序列与覆盖率仍与 `workers=0` 一致。"""
    monkeypatch.setattr(torch.utils.data, "DataLoader", _FakeLoader)
    sync = _take(_source(monkeypatch, workers=0), 24)
    parallel_source = _source(monkeypatch, workers=2)
    parallel = _take(parallel_source, 24)
    assert parallel == sync
    assert parallel_source.coverage()["windows_seen"] == 24.0
    # 交付确实被打乱了（否则本测试是恒真的：计划顺序 == 交付顺序就什么都没验证）
    assert len(_FakeLoader.plan_order) > 3
    assert sorted(_FakeLoader.delivery_order) == sorted(_FakeLoader.plan_order)
    assert _FakeLoader.delivery_order != _FakeLoader.plan_order
    # 且 in_order 真的关掉了——这是本轮吞吐修复的**唯一开关**
    assert _FakeLoader.seen_kwargs["in_order"] is False


def test_ab_flag_restores_torch_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """A/B 开关（诊断用）必须能把 `in_order` 退回 `True`，否则标定脚本没有对照臂。"""
    monkeypatch.setattr(torch.utils.data, "DataLoader", _FakeLoader)
    monkeypatch.setattr(ManifestBatchSource, "_ab_force_in_order", True)
    _take(_source(monkeypatch, workers=2), 4)
    assert _FakeLoader.seen_kwargs["in_order"] is True


def test_dropped_batch_is_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    """乱序交付**丢批**时必须抛，而不是静默少训：`_cursor` 会因此停在错误的槽位。"""

    class _DroppingLoader(_FakeLoader):
        def __init__(self, dataset: Any, **kwargs: Any) -> None:
            super().__init__(dataset, mode="drop_one", **kwargs)

    monkeypatch.setattr(torch.utils.data, "DataLoader", _DroppingLoader)
    with pytest.raises(RuntimeError, match="重排缓冲未排空"):
        _take(_source(monkeypatch, workers=2), 200)
