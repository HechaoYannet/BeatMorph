"""`shuffle_counts_within_line` 的五项契约（RFC-0037 §2.4 / R5）——默认 CI，无 GPU。

val 的置换对照（`val/nll_shuffled_delta`）用它构造「同输入、只换标签」的对照批。
旧的全局置换（被删的 `shuffle_hidden_counts`）会连**每线事件预算**一起改
（nuisance，RFC-0036 §2.5 实证：同一批 real 166.74 / 线内置换 257.61 / 全局 −26.49）；
线内置换把控制收回到「只破坏位置 ↔ 上下文对应」这一件事上。
"""

from __future__ import annotations

import pytest
import torch

from beatmorph.infra.config.loading import config_from_mapping
from beatmorph.infra.smoke import SmokeBatchSource, shuffle_counts_within_line

PROVENANCE = {
    "source": "fixtures",
    "query": "n/a",
    "fetched_at": "t",
    "purpose": "train",
    "script": "s",
    "script_version": "v1",
}
BASE = {
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
}


def _batch():  # type: ignore[no-untyped-def]
    cfg = config_from_mapping(BASE)
    source = SmokeBatchSource(cfg, seed=cfg.optim.seed)
    return cfg, source.batch(masked=True, samples=4)


def test_visible_field_is_bit_identical() -> None:
    """契约①：模型输入（可见场 + 遮盖位图）逐位不变。"""
    cfg, batch = _batch()
    shuffled = shuffle_counts_within_line(batch, seed=cfg.optim.seed + 991)
    assert torch.equal(shuffled.observed_counts(), batch.observed_counts())
    assert torch.equal(shuffled.occlusion_bool(), batch.occlusion_bool())


def test_per_line_event_counts_are_preserved() -> None:
    """契约②③：每线事件数不变（nuisance 被钉死），总事件数随之不变。"""
    cfg, batch = _batch()
    shuffled = shuffle_counts_within_line(batch, seed=cfg.optim.seed + 991)
    assert batch.counts is not None
    assert shuffled.counts is not None
    per_line = lambda t: t.sum(dim=(2, 3, 4, 5))  # noqa: E731 - (B, K)
    assert torch.equal(per_line(shuffled.counts), per_line(batch.counts))
    assert int(shuffled.counts.sum()) == int(batch.counts.sum())


def test_same_seed_is_reproducible_and_result_differs() -> None:
    """契约④：同 seed 逐位可复现；且置换确实改变了被遮盖事件的排布。"""
    _cfg, batch = _batch()
    a = shuffle_counts_within_line(batch, seed=7)
    b = shuffle_counts_within_line(batch, seed=7)
    assert torch.equal(a.counts, b.counts)
    assert not torch.equal(a.counts, batch.counts)


def test_global_rng_is_untouched() -> None:
    """契约⑤：用独立 Generator，不污染全局 RNG（否则同 seed 的训练臂会跟着漂）。"""
    _cfg, batch = _batch()
    torch.manual_seed(123)
    expected = torch.randn(8)
    torch.manual_seed(123)
    shuffle_counts_within_line(batch, seed=99)
    actual = torch.randn(8)
    assert torch.equal(actual, expected)


def test_requires_a_masked_batch() -> None:
    """无遮盖时「输入就是目标」，没有「待补全的标签」可置换——必须报错而不是静默退化。"""
    cfg = config_from_mapping(BASE)
    source = SmokeBatchSource(cfg, seed=cfg.optim.seed)
    unmasked = source.batch(masked=False, samples=2)
    with pytest.raises(ValueError, match="遮盖"):
        shuffle_counts_within_line(unmasked, seed=1)
