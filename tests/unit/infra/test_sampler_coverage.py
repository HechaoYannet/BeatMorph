"""采样器覆盖率与 epoch（RFC-0033；默认 CI，无权重 / 无 GPU / 无真实语料）。

历史缺陷（2026-09-27，用**仓库里那份 `_draw`** 实测）：`ManifestBatchSource` 只有一个
**全局共享游标** `(start + take) % len(bucket)`，而桶长最小为 **1** —— 碰到长度 1 的桶游标
就被清零，于是在「每个桶的前几个窗口」之间震荡，**1000 步后彻底饱和**：
此后无论 `max_steps` 加到多少，永远只抽同一批 **993 个窗口（全库 0.156%）/ 674 张谱面**。
20000 步停下来看，唯一窗口数还是 993（`scripts/local_draw_verify.py`）。loss 曲线看不出
任何异常——它只是反复拟合同一小撮样本。

本文件钉死修复后的四条不变量：

1. **一个 epoch 内每个窗口恰好被抽一次**（不再饱和）；
2. epoch 结束后重开一轮，覆盖率**只增不减**；
3. 同 seed 的抽签顺序**逐位可复现**（M7.8）；不同 seed 不同；
4. 选桶**按剩余窗口数加权**：否则轮转会让长度 1 的小桶被抽干、大桶几乎不动——
   真实数据实测那样 20,000 步只覆盖 47% 谱面，而按窗口均匀应当是 ≈91%。
"""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

import pytest

import beatmorph.data.dataset as dataset_mod
from beatmorph.infra.config.loading import config_from_mapping
from beatmorph.infra.train_loop import ManifestBatchSource

PROVENANCE = {
    "source": "fixtures",
    "query": "n/a",
    "fetched_at": "t",
    "purpose": "train",
    "script": "s",
    "script_version": "v1",
}

#: 桶大小刻意极度不均，并把 **1** 放进去——那正是旧游标被清零的触发条件。
BUCKET_SIZES = (1, 2, 5, 7, 30)


class _StubDataset:
    """只实现 `ManifestBatchSource` 真正调用的那几个方法（构造 63 万窗口对单测无必要）。"""

    def __init__(self, bucket_sizes: Sequence[int], n_rows: int) -> None:
        self.keys: list[tuple[int, int, float]] = []
        self.rows_of_window: list[int] = []
        self.bucket_of_window: list[int] = []
        row = 0
        for bucket, size in enumerate(bucket_sizes):
            for _ in range(size):
                self.keys.append((8, 4, 100.0 + bucket))
                self.bucket_of_window.append(bucket)
                self.rows_of_window.append(row % n_rows)
                row += 1
        self.n_rows = n_rows

    def __len__(self) -> int:
        return len(self.keys)

    def grid_key(self, index: int) -> tuple[int, int, float]:
        return self.keys[index]

    def window_row_index(self, index: int) -> int:
        return self.rows_of_window[index]

    def rows(self) -> list[int]:
        return list(range(self.n_rows))

    def stats(self) -> SimpleNamespace:
        return SimpleNamespace(n_rows_used=len(set(self.rows_of_window)))

    def __getitem__(self, index: int) -> int:
        return index


def _passthrough(samples: list[int]) -> list[int]:
    """`collate_field_batch` 的替身：取批逻辑是本测试的对象，样本组装不是。"""
    return list(samples)


def _config(*, batch_size: int = 1):
    return config_from_mapping(
        {
            "data": {
                "source": "synthetic",
                "max_samples": 4,
                "occlusion_ratio": 0.5,
                "k_max": 8,
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
            "optim": {"lr": 1e-3, "batch_size": batch_size, "max_steps": 4},
        }
    )


def _source(
    monkeypatch: pytest.MonkeyPatch,
    *,
    bucket_sizes: Sequence[int] = BUCKET_SIZES,
    n_rows: int = 6,
    seed: int = 3,
    batch_size: int = 1,
) -> tuple[ManifestBatchSource, _StubDataset]:
    # `_draw` 在函数内 `from beatmorph.data.dataset import collate_field_batch`，
    # 因此替换模块属性即可让真实取批逻辑跑在 stub 上（不解析任何谱面）。
    monkeypatch.setattr(dataset_mod, "collate_field_batch", _passthrough)
    stub = _StubDataset(bucket_sizes, n_rows)
    source = ManifestBatchSource(_config(batch_size=batch_size), split="train", seed=seed)
    source._dataset = stub  # type: ignore[assignment]
    return source, stub


def _indices(batch: object) -> list[int]:
    return list(batch)  # type: ignore[arg-type]


def test_full_epoch_draws_every_window_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """一个 epoch 内**每个窗口恰好一次**——旧实现的共享游标在这里恒为「前几个窗口反复」。"""
    source, stub = _source(monkeypatch)
    total = len(stub)
    drawn = [index for _ in range(total) for index in _indices(source._draw(masked=True))]
    assert len(drawn) == total
    assert sorted(drawn) == list(range(total)), "epoch 内出现重复窗口或漏抽"

    coverage = source.coverage()
    assert coverage["windows_seen"] == float(total)
    assert coverage["windows_total"] == float(total)
    assert coverage["charts_seen"] == coverage["charts_total"] > 0
    assert coverage["epoch"] == pytest.approx(1.0)


def test_coverage_starts_empty_and_never_decreases(monkeypatch: pytest.MonkeyPatch) -> None:
    """未取批时覆盖率全 0；取批后只增不减（这是「采样器有没有在走数据」的唯一在线证据）。"""
    source, stub = _source(monkeypatch)
    empty = source.coverage()
    assert empty["windows_seen"] == 0.0
    assert empty["windows_total"] == 0.0

    seen: list[float] = []
    for _ in range(20):
        source._draw(masked=True)
        coverage = source.coverage()
        seen.append(coverage["windows_seen"])
        assert coverage["windows_total"] == float(len(stub))
    assert seen == [float(index) for index in range(1, 21)]


def test_epoch_restarts_after_full_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """走完一个 epoch 后重开一轮：epoch 继续增长，且新一轮的窗口集合仍然是全集。"""
    source, stub = _source(monkeypatch)
    total = len(stub)
    for _ in range(total):
        source._draw(masked=True)
    assert source.coverage()["epoch"] == pytest.approx(1.0)

    second = [index for _ in range(total) for index in _indices(source._draw(masked=True))]
    assert sorted(second) == list(range(total)), "第二个 epoch 不再是全集的排列"
    assert source.coverage()["epoch"] == pytest.approx(2.0)


def test_draw_order_is_reproducible_and_seed_dependent(monkeypatch: pytest.MonkeyPatch) -> None:
    """同一 seed 两次运行逐位一致（M7.8）；换 seed 顺序不同。"""
    first, stub = _source(monkeypatch, seed=11)
    total = len(stub)
    second = _source(monkeypatch, seed=11)[0]
    other = _source(monkeypatch, seed=12)[0]

    def sequence(source: ManifestBatchSource) -> list[int]:
        return [index for _ in range(total) for index in _indices(source._draw(masked=True))]

    assert sequence(first) == sequence(second)
    assert sequence(other) != sequence(first)


def test_bucket_choice_is_weighted_by_remaining_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """大桶必须按**窗口数**被抽到，而不是每个桶平均分——否则大桶里的谱面永远见不到。"""
    source, stub = _source(monkeypatch, bucket_sizes=(1, 1_000), n_rows=6)
    bucket_of: dict[int, int] = {}
    for index in range(len(stub)):
        bucket_of[index] = stub.bucket_of_window[index]

    draws = [index for _ in range(200) for index in _indices(source._draw(masked=True))]
    from_large = sum(1 for index in draws if bucket_of[index] == 1)
    # 均匀口径：大桶占 1000/1001 的窗口 ⇒ 期望 ~199.8/200；轮转会给出 ~100/200。
    assert from_large >= 190, f"选桶没有按剩余窗口数加权：大桶只被抽到 {from_large}/200"


def test_batches_keep_grid_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """batch_size > 1 时同一批必须来自**同一个桶**（`FieldBatch` 只带一个网格）。"""
    source, stub = _source(monkeypatch, batch_size=2)
    for _ in range(10):
        batch = _indices(source._draw(masked=True))
        assert 1 <= len(batch) <= 2
        keys = {stub.grid_key(index) for index in batch}
        assert len(keys) == 1, f"同一批混了两个网格身份：{keys}"
