"""集成：真实数据通路 dataset -> collate -> MaskedFieldModel（CPU，无网络 / 无权重 / 无 GPU）。

与被造的 `tests/integration/generation/test_train_step.py` 的区别：那条用手写的
`_line_tracks` / 手写 counts 拼 FieldBatch；本条**从磁盘开始**——RPEJSON 谱面文件 +
迷你特征缓存 + 清单 JSON——经 :class:`~beatmorph.data.dataset.ChartPairDataset` 与
:func:`~beatmorph.data.dataset.collate_field_batch` 走到 generation 主干，
是「真实数据接通 generation」的证据（README「下一步」第 1 项）。

夹具纪律：帧率 / 采样率 / 网格宽 / 拍格宽一律引用契约常量，不写字面量。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from beatmorph.core.contracts import (
    MERT_DEFAULT_FEAT_DIM,
    MERT_FRAME_RATE_HZ,
    MERT_SAMPLE_RATE_HZ,
    RPE_STAGE_HALF_WIDTH,
    RPE_X_GRID_BINS,
    SUBDIVISIONS_PER_BEAT,
    TAU_GRID_DT,
)
from beatmorph.data.dataset import ChartPairDataset, DatasetConfig, collate_field_batch
from beatmorph.data.pipeline.embed import PairRow, PairSplits, feature_cache_paths
from beatmorph.field.grid import SECONDS_PER_MINUTE
from beatmorph.generation.losses import occlusion_ratio
from beatmorph.generation.model import MaskedFieldModel, ModelConfig

pytestmark = pytest.mark.integration

#: 窗口长度 = 1 拍；谱面 3.5 拍 -> 3 个整窗 + 半个窗的尾巴（派生式，不写字面量）
WINDOW: int = SUBDIVISIONS_PER_BEAT
BEATS: float = 3 * SUBDIVISIONS_PER_BEAT * TAU_GRID_DT + SUBDIVISIONS_PER_BEAT * TAU_GRID_DT / 2
BPM: float = 120.0
K_LINES: int = 2

MODEL_CONFIG = ModelConfig(
    d_model=16,
    n_heads=2,
    n_layers=2,
    window=2,
    global_period=2,
    k_max=4,
    audio_dim=MERT_DEFAULT_FEAT_DIM,
)


def _beat(value: float) -> list[int]:
    """拍值 -> RPE beat 三元组 [i, n, d]（分母取契约细分）。"""
    index = int(value // 1)
    return [index, round((value - index) * SUBDIVISIONS_PER_BEAT), SUBDIVISIONS_PER_BEAT]


def _note(beat: float, *, line_id: int) -> dict[str, Any]:
    """一个 Tap（位置按半宽派生，保证落在可见范围内）。"""
    return {
        "type": 1,
        "startTime": _beat(beat),
        "endTime": _beat(beat),
        "positionX": (-1.0) ** line_id * RPE_STAGE_HALF_WIDTH / 4,
        "above": 1,
        "isFake": 0,
        "size": 1.0,
        "speed": 1.0,
        "alpha": 255,
        "visibleTime": 999999.0,
        "yOffset": 0.0,
    }


def _chart_dict() -> dict[str, Any]:
    """2 条线；每个窗口各 2 个事件 token（>= 2 个遮盖单位，满足 r < 1 的可行性）。"""
    lines = [{"Name": f"line-{index}", "notes": []} for index in range(K_LINES)]
    for window in range(3):
        for offset, line_id in ((1, 0), (2, 1)):
            beat = window + offset / 4
            lines[line_id]["notes"].append(_note(beat, line_id=line_id))
    return {
        "BPMList": [{"bpm": BPM, "startTime": _beat(0)}],
        "META": {"RPEVersion": 150, "name": "integration", "composer": "integration"},
        "chartTime": BEATS * SECONDS_PER_MINUTE / BPM,
        "judgeLineList": lines,
    }


def _write_feature_cache(feature_dir: Path, key: str, duration_s: float) -> None:
    """迷你特征缓存（rate / sample_rate / feat_dim 全部引用契约常量，帧数与时长自洽）。"""
    npz_path, meta_path = feature_cache_paths(feature_dir, key)
    feature_dir.mkdir(parents=True, exist_ok=True)
    frames = round(duration_s * MERT_FRAME_RATE_HZ)
    emb = np.random.default_rng(0).standard_normal(
        (frames, MERT_DEFAULT_FEAT_DIM), dtype=np.float32
    )
    np.savez_compressed(npz_path, emb=emb)
    meta_path.write_text(
        json.dumps(
            {
                "rate": MERT_FRAME_RATE_HZ,
                "sample_rate": MERT_SAMPLE_RATE_HZ,
                "layer": 12,
                "model_rev": "integration-test",
                "duration_s": duration_s,
                "original_sample_rate": MERT_SAMPLE_RATE_HZ,
                "feat_dim": MERT_DEFAULT_FEAT_DIM,
                "dtype": str(emb.dtype),
                "adapter": "none",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _dataset(tmp_path: Path) -> ChartPairDataset:
    """物化 谱面 / 特征 / 清单 三件套并构造数据集（全程无网络、无权重）。"""
    chart_dir = tmp_path / "charts"
    feature_dir = tmp_path / "features"
    relative = "1/chart.json"
    target = chart_dir / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_chart_dict(), ensure_ascii=False), encoding="utf-8")
    _write_feature_cache(feature_dir, "feat-1", BEATS * SECONDS_PER_MINUTE / BPM)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            PairSplits(
                train=[
                    PairRow(
                        chart_id=1,
                        song_key="song|composer",
                        split="train",
                        chart_path=relative,
                        feature_key="feat-1",
                        difficulty=15.0,
                        fmt="rpe",
                        name="song",
                        composer="composer",
                    ),
                ],
            ).to_dict(),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return ChartPairDataset(
        DatasetConfig(
            manifest_path=manifest,
            chart_dir=chart_dir,
            feature_dir=feature_dir,
            t_window=WINDOW,
            x_bins=RPE_X_GRID_BINS,
        ),
    )


def test_dataset_pickle_round_trip_drops_derived_state(tmp_path: Path) -> None:
    """worker 序列化必须**剥掉**派生状态，且还原后逐位给出同样的样本（RFC-0034 S2）。

    没有 `__getstate__`，spawn 出来的每个 worker 都会收到一份完整索引计划与主进程的 LRU
    （内存按 worker 数翻倍），而诊断计数还会被复制成 N 份。
    """
    import pickle

    dataset = _dataset(tmp_path)
    before = dataset[0]
    assert dataset._plan is not None
    assert len(dataset._chart_cache) > 0

    restored = pickle.loads(pickle.dumps(dataset))
    assert restored._plan is None, "索引计划被序列化给了 worker（应当按需从落盘缓存重建）"
    assert not restored._chart_cache
    assert not restored._feature_cache
    assert restored.no_visible_context_fallbacks() == 0

    after = restored[0]
    assert after.chart_id == before.chart_id
    assert torch.equal(after.counts, before.counts)
    assert torch.equal(after.audio_emb, before.audio_emb)
    assert torch.equal(after.line_tracks, before.line_tracks)


def test_dataset_window_trains_generation_on_cpu(tmp_path: Path) -> None:
    """一个真实窗口 -> FieldBatch -> 前向 loss 有限 -> 反向梯度有限。"""
    dataset = _dataset(tmp_path)
    expected_windows = round(BEATS * SUBDIVISIONS_PER_BEAT) // WINDOW
    assert len(dataset) == expected_windows
    sample = dataset[0]
    assert int(sample.counts.sum()) > 0

    batch = collate_field_batch([sample])
    batch.assert_shapes()
    assert batch.grid.t_bins == WINDOW
    assert batch.n_lines() == K_LINES
    ratio = occlusion_ratio(batch)
    assert 0.0 < ratio < 1.0, "遮盖补全训练需要 0 < r < 1（r == 1 时 losses 直接拒绝）"
    observed = batch.observed_counts()
    assert float(observed[batch.occlusion_bool()].abs().sum()) == 0.0

    model = MaskedFieldModel(MODEL_CONFIG, batch.grid)
    output = model(batch)
    assert output.loss is not None
    assert bool(torch.isfinite(output.loss))
    output.loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients, "反向必须产生梯度（模型没有可训练参数则说明接线断了）"
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
