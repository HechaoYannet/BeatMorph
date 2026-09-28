"""门禁批**非空**的契约（plan 07 §9-23）——默认 CI，无权重 / 无 GPU。

真实 200 行切片上 G1 抽到的批实测 **K=2 / 0 事件**，`gates.txt` 因此报了
`[PASS] G1 loss 7819.5 -> 0.0014`，而那个批里**没有任何事件可过拟合**——门禁是**空过**。
根因是门禁装配对「批里有没有事件」没有任何要求，而真实窗口里存在大量空窗
（另见 τ 轴终点缺陷：`beatmorph/data/dataset.py` 模块 docstring）。

本文件钉死三件事：
1. `draw_until_min_events` 会重抽到达标为止，取不到就**抛**（fail-closed）；
2. 两个门禁批（G1 / G3；RFC-0037 起 G2 已删除）都带上 `gates.batch_min_events`；
3. 合成来源在事件数不足时**报错**而不是静默发出空批。
"""

from __future__ import annotations

import pytest

from beatmorph.infra.config.loading import config_from_mapping
from beatmorph.infra.smoke import SmokeBatchSource
from beatmorph.infra.train_loop import build_gate_inputs, draw_until_min_events

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
    "gates": {
        "baseline_samples": 4,
        "baseline_steps": 5,
        "overfit_steps": 3,
        "initial_head_bias": 0.0,
    },
}


class _FakeBatch:
    """只带 `counts` 的假批（`event_total` 只读这一个字段）。"""

    def __init__(self, events: float) -> None:
        self.counts = None if events < 0 else _FakeCounts(events)


class _FakeCounts:
    def __init__(self, total: float) -> None:
        self._total = total

    def sum(self) -> object:
        return self._total


def _drawer(values: list[float]) -> object:
    """按给定序列逐次返回假批的函数（序列用尽后重复最后一个）。"""
    state = {"index": 0}

    def draw() -> object:
        index = min(state["index"], len(values) - 1)
        state["index"] += 1
        return _FakeBatch(values[index])

    return draw


def test_retry_until_non_empty() -> None:
    """空批会被重抽掉：返回的是**第一个非空**批。"""
    draw = _drawer([0.0, 0.0, 7.0])
    batch = draw_until_min_events(draw, min_events=1, attempts=8)  # type: ignore[arg-type]
    assert batch.counts is not None
    assert batch.counts.sum() == 7.0


def test_fail_closed_when_never_non_empty() -> None:
    """一直取不到就抛——**不得**把空批当成有效门禁输入。"""
    draw = _drawer([0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="取不到"):
        draw_until_min_events(draw, min_events=1, attempts=3)  # type: ignore[arg-type]


def test_min_events_zero_keeps_old_behaviour() -> None:
    """`min_events=0` 时只抽一次（旧行为不变；训练路径用默认值）。"""
    draw = _drawer([0.0, 5.0])
    batch = draw_until_min_events(draw, min_events=0, attempts=8)  # type: ignore[arg-type]
    assert batch.counts is not None
    assert batch.counts.sum() == 0.0


class _SpySource:
    """记录每次 `batch()` 的实参（两个门禁批的观测点）。"""

    def __init__(self, inner: SmokeBatchSource) -> None:
        self.inner = inner
        self.calls: list[tuple[bool, int | None, int]] = []

    def batch(
        self,
        *,
        masked: bool,
        samples: int | None = None,
        min_events: int = 0,
    ) -> object:
        self.calls.append((bool(masked), samples, int(min_events)))
        return self.inner.batch(
            masked=masked,
            samples=samples,
            min_events=min_events,
        )

    def describe(self) -> str:
        return self.inner.describe()


def test_all_gate_batches_require_events() -> None:
    """G1 / G3 两个批都带上 `gates.batch_min_events`（空批不得进判据）。"""
    mapping = {key: dict(value) for key, value in BASE.items()}
    mapping["gates"]["batch_min_events"] = 3
    cfg = config_from_mapping(mapping)
    spy = _SpySource(SmokeBatchSource(cfg, seed=0, k_lines=2))
    _inputs, stats = build_gate_inputs(cfg, spy)  # type: ignore[arg-type]
    assert spy.calls, "装配必须取批"
    assert all(min_events == 3 for _masked, _samples, min_events in spy.calls)
    assert stats["gate_min_batch_events"] == 3.0
    # 生效值必须落进 gates.txt 的上下文（否则事后无法判断门禁是不是空过的）
    assert stats["g1_events"] > 0.0
    assert stats["g1_lines"] > 0.0


def test_gate_config_default_is_non_empty() -> None:
    """默认口径就是「非空」（`batch_min_events=1`），不依赖调用方记得打开。"""
    cfg = config_from_mapping({key: dict(value) for key, value in BASE.items()})
    assert cfg.gates.batch_min_events == 1


def test_synthetic_source_rejects_impossible_min_events() -> None:
    """合成来源：要求的事件数超过谱面实有事件时**报错**（不静默发空批）。"""
    mapping = {key: dict(value) for key, value in BASE.items()}
    cfg = config_from_mapping(mapping)
    source = SmokeBatchSource(cfg, seed=0, k_lines=1)
    with pytest.raises(ValueError, match="合成批只有"):
        source.batch(masked=True, samples=1, min_events=10**9)
