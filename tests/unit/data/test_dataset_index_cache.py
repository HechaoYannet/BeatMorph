"""索引落盘缓存（plan 02 §9 ④ 的修法①③）——默认 CI，无网络 / 无权重 / 无 GPU。

**为什么必须有**：索引构建要对**每一行**读文件 + 嗅探 + 解析 + 规划（全库 8551 行，
实测 0.19 s/行 ⇒ 约 27 min），而它对同一份（清单 + 配置 + 谱面文件）是纯函数。
不缓存 ⇒ 每次开训都要先等半小时才轮到第一个优化步。

本文件钉死四条**缓存纪律**：

1. 命中缓存与重建**逐位一致**（缓存是加速器，不是第二个事实源）；
2. 指纹覆盖配置与文件 stat ⇒ 改了配置/换了谱面**不可能**复用旧计划；
3. 缓存损坏 / 指纹不符一律**回退重建**（不抛错、不静默用坏数据）；
4. `index_cache=False` 时一个文件都不写（测试与对照用）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from beatmorph.core.contracts import (
    MERT_DEFAULT_FEAT_DIM,
    MERT_FRAME_RATE_HZ,
    MERT_SAMPLE_RATE_HZ,
    RPE_STAGE_HALF_WIDTH,
    SUBDIVISIONS_PER_BEAT,
    TAU_GRID_DT,
)
from beatmorph.data.dataset import ChartPairDataset, DatasetConfig, _plan_cache_path
from beatmorph.data.pipeline.embed import (
    FeatureCacheMeta,
    PairRow,
    PairSplits,
    feature_cache_paths,
    save_feature_cache,
)
from beatmorph.field.grid import SECONDS_PER_MINUTE
from tests.unit.data._helpers import build_rpe, note

#: 窗口长度 = 1 拍（派生）
WINDOW: int = SUBDIVISIONS_PER_BEAT
#: 谱面拍数（测试输入）
BEATS: float = 4.5
#: 测试输入（不是物理常量）
BPM: float = 120.0


def _seconds(beats: float) -> float:
    """拍 -> 秒（测试侧独立算式；正式换算只在 field/）。"""
    return beats * SECONDS_PER_MINUTE / BPM


def _beat(value: float) -> list[int]:
    """拍值 -> RPE beat 三元组。"""
    index = int(value // 1)
    return [index, round((value - index) * SUBDIVISIONS_PER_BEAT), SUBDIVISIONS_PER_BEAT]


def _payload() -> dict[str, Any]:
    """最小 RPEJSON（每窗 2 个遮盖单位 + 2 个 token；与 test_dataset 的布局同口径）。"""
    lines: list[dict[str, Any]] = [{"Name": "l0", "notes": []}, {"Name": "l1", "notes": []}]
    lines[0]["notes"] = [
        note(1, _beat(1 / 4), position_x=-RPE_STAGE_HALF_WIDTH / 2),
        note(2, _beat(1.0), end=_beat(1.5), position_x=0.0),
        note(1, _beat(5 / 2), position_x=RPE_STAGE_HALF_WIDTH / 4),
    ]
    lines[1]["notes"] = [
        note(1, _beat(1 / 2), position_x=RPE_STAGE_HALF_WIDTH / 2),
        note(1, _beat(3 / 2), position_x=0.0),
        note(1, _beat(2), position_x=0.0),
        note(4, _beat(7 / 2), position_x=0.0),
    ]
    return build_rpe(
        lines=lines,
        bpm_list=[{"bpm": BPM, "startTime": _beat(0)}],
        chart_time=_seconds(BEATS),
    )


def _materialize(tmp_path: Path) -> tuple[Path, Path, Path]:
    """物化 charts / features / manifest（特征缓存的时长与谱面一致）。"""
    chart_dir = tmp_path / "charts"
    feature_dir = tmp_path / "features"
    relative = "chart-0/chart.json"
    target = chart_dir / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_payload(), ensure_ascii=False), encoding="utf-8")
    key = "feat-0"
    feature_dir.mkdir(parents=True, exist_ok=True)
    npz_path, meta_path = feature_cache_paths(feature_dir, key)
    duration = _seconds(BEATS)
    emb = np.zeros((round(duration * MERT_FRAME_RATE_HZ), MERT_DEFAULT_FEAT_DIM), dtype=np.float32)
    save_feature_cache(
        emb,
        npz_path,
        meta_path,
        FeatureCacheMeta(
            rate=MERT_FRAME_RATE_HZ,
            sample_rate=MERT_SAMPLE_RATE_HZ,
            layer=12,
            model_rev="unit-test",
            duration_s=duration,
            original_sample_rate=MERT_SAMPLE_RATE_HZ,
            feat_dim=MERT_DEFAULT_FEAT_DIM,
            dtype="float32",
            adapter="none",
        ),
    )
    manifest = tmp_path / "manifest.json"
    row = PairRow(
        chart_id=1,
        song_key="song|composer",
        split="train",
        chart_path=relative,
        feature_key=key,
        difficulty=15.0,
        fmt="rpe",
    )
    manifest.write_text(
        json.dumps(PairSplits(train=[row]).to_dict(), ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest, chart_dir, feature_dir


def _open(
    manifest: Path,
    chart_dir: Path,
    feature_dir: Path,
    **overrides: Any,
) -> ChartPairDataset:
    """按同一份物化产物开数据集（`overrides` 逐项覆盖 DatasetConfig）。"""
    params: dict[str, Any] = {"t_window": WINDOW}
    params.update(overrides)
    return ChartPairDataset(
        DatasetConfig(
            manifest_path=manifest,
            chart_dir=chart_dir,
            feature_dir=feature_dir,
            **params,
        ),
    )


def _snapshot(dataset: ChartPairDataset) -> list[tuple[int, int, int, float]]:
    """索引的可比较快照（窗口定位 + 网格身份）。"""
    return [
        (entry.row_index, entry.window_index, entry.tau_start_bins, entry.bpm_eff)
        for entry in dataset._ensure_plan().entries
    ]


def test_cache_round_trip_is_bit_identical(tmp_path: Path) -> None:
    """命中缓存与重建的索引**逐位一致**（含记账）。"""
    manifest, chart_dir, feature_dir = _materialize(tmp_path)
    fresh = _open(manifest, chart_dir, feature_dir)
    assert len(fresh) > 0
    assert not fresh._plan_from_cache
    expected = _snapshot(fresh)

    warm = _open(manifest, chart_dir, feature_dir)
    # ⚠️ 索引是**惰性**构建的：必须先取一次，再问「是不是来自缓存」
    assert _snapshot(warm) == expected
    assert warm._plan_from_cache, "第二次开数据集必须命中落盘缓存"
    assert warm.stats() == fresh.stats(), "记账也必须一起命中（否则 gates.txt 会说谎）"


def test_cache_disabled_writes_nothing(tmp_path: Path) -> None:
    """`index_cache=False`：不写缓存文件（对照与测试用）。"""
    manifest, chart_dir, feature_dir = _materialize(tmp_path)
    dataset = _open(manifest, chart_dir, feature_dir, index_cache=False)
    assert len(dataset) > 0
    assert not (tmp_path / ".dataset_index").exists()


def test_fingerprint_follows_config(tmp_path: Path) -> None:
    """换配置（t_window）⇒ 换指纹 ⇒ 两条缓存并存，互不污染。"""
    manifest, chart_dir, feature_dir = _materialize(tmp_path)
    base = _open(manifest, chart_dir, feature_dir)
    assert len(base) > 0
    alternates = _open(manifest, chart_dir, feature_dir, t_window=WINDOW * 2)
    assert len(alternates) > 0
    cache_files = sorted((tmp_path / ".dataset_index").glob("*.npz"))
    assert len(cache_files) == 2, "不同配置必须落成不同的缓存文件"


def test_fingerprint_follows_chart_file(tmp_path: Path) -> None:
    """谱面文件被改写（stat 变）⇒ 旧缓存不得复用。"""
    manifest, chart_dir, feature_dir = _materialize(tmp_path)
    first = _open(manifest, chart_dir, feature_dir)
    expected = _snapshot(first)
    path = chart_dir / "chart-0/chart.json"
    payload = _payload()
    # 加在**窗口 3**（[3,4) 拍）里：让它从「只有 1 个遮盖单位 ⇒ 被跳过」变成可用窗口
    payload["judgeLineList"][0]["notes"].append(note(1, _beat(3.2), position_x=0.0))
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    second = _open(manifest, chart_dir, feature_dir)
    rebuilt = _snapshot(second)
    assert not second._plan_from_cache, "谱面变了就必须重建"
    assert rebuilt != expected or len(second) != len(first)


def test_corrupt_cache_falls_back(tmp_path: Path) -> None:
    """缓存损坏 ⇒ 回退重建（不抛错、也不用坏数据）。"""
    manifest, chart_dir, feature_dir = _materialize(tmp_path)
    first = _open(manifest, chart_dir, feature_dir)
    expected = _snapshot(first)
    cache_files = list((tmp_path / ".dataset_index").glob("*.npz"))
    assert len(cache_files) == 1
    cache_files[0].write_bytes(b"not an npz at all")
    second = _open(manifest, chart_dir, feature_dir)
    rebuilt = _snapshot(second)
    assert not second._plan_from_cache, "缓存坏了就必须重建（而不是报错或静默用坏数据）"
    assert rebuilt == expected


def test_manifest_mismatch_does_not_reuse_cache(tmp_path: Path) -> None:
    """`limit` 变（行集变）⇒ 指纹变 ⇒ 不复用（否则训练的「切片」会静默变成全量计划）。"""
    manifest, chart_dir, feature_dir = _materialize(tmp_path)
    full = _open(manifest, chart_dir, feature_dir)
    assert len(full) > 0
    limited = _open(manifest, chart_dir, feature_dir, limit=0)
    assert len(limited) == 0
    assert not limited._plan_from_cache


def test_plan_cache_path_is_none_when_disabled(tmp_path: Path) -> None:
    """缓存的开关是**配置**而不是环境：关掉时连路径都不产生。"""
    manifest, chart_dir, feature_dir = _materialize(tmp_path)
    config = DatasetConfig(
        manifest_path=manifest,
        chart_dir=chart_dir,
        feature_dir=feature_dir,
        t_window=WINDOW,
        index_cache=False,
    )
    assert _plan_cache_path(config, []) is None


def test_window_is_one_beat_by_derivation() -> None:
    """夹具自检：窗口 = 1 拍（派生量，不写 48）。"""
    assert WINDOW * TAU_GRID_DT == 1.0
