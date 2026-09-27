"""τ 轴终点口径（plan 02 §9 ④ / plan 03 §9-14）：真实语料的 `chartTime` 不可信。

**为什么这是缺陷而不是配置口味**：`PhigrosChart.duration_s()` 取 `max(最后一事件, META.chartTime)`，
而全库 8551 张里 52.2% 的 `time_span_s / audio_duration_s > 10`（中位 52.8×，最大 3.4e6×）。
旧口径因此把 200 行切片切出 674 万个窗口（≈3.4 万窗/行，正常谱面 50–110 窗），
其中绝大多数是空窗（无事件 + 音频整段越界补零），G1 抽到的批实测 K=2 / 0 事件。

夹具纪律（AGENTS.md §3.3 / 红线 7）：拍 <-> 秒一律用 `SECONDS_PER_MINUTE` / `SUBDIVISIONS_PER_BEAT`
现算；谱面拍数 / BPM / `chartTime` 是**测试输入**，允许写字面量。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from beatmorph.core.contracts import (
    MERT_DEFAULT_FEAT_DIM,
    MERT_FRAME_RATE_HZ,
    MERT_SAMPLE_RATE_HZ,
    RPE_NORMAL_TRACKS,
    RPE_STAGE_HALF_WIDTH,
    SUBDIVISIONS_PER_BEAT,
    TAU_GRID_DT,
    ChartSource,
)
from beatmorph.data.dataset import ChartPairDataset, DatasetConfig
from beatmorph.data.parsers.rpejson import parse_rpejson
from beatmorph.data.pipeline.embed import (
    FeatureCacheMeta,
    PairRow,
    PairSplits,
    feature_cache_paths,
    save_feature_cache,
)
from beatmorph.field.grid import SECONDS_PER_MINUTE
from tests.unit.data._helpers import build_rpe, note

#: 窗口长度 = 1 拍（派生，不写 48）
WINDOW: int = SUBDIVISIONS_PER_BEAT
#: 谱面真实内容长度：5.5 拍（5 个整窗 + 半窗尾巴）
CONTENT_BEATS: float = 5.5
#: 音频时长（拍）：与真实内容一致
AUDIO_BEATS: float = 5.5
#: 被写坏的 `META.chartTime`：100 拍（真实内容只有 5.5 拍）
BOGUS_CHART_BEATS: float = 100.0
#: 测试输入（不是物理常量）
BPM: float = 120.0
DIFFICULTY: float = 15.0


def _seconds(beats: float) -> float:
    """拍 -> 秒（测试侧独立算式；正式换算只在 field/）。"""
    return beats * SECONDS_PER_MINUTE / BPM


def _beat(value: float) -> list[int]:
    """拍值 -> RPE beat 三元组。"""
    index = int(value // 1)
    return [index, round((value - index) * SUBDIVISIONS_PER_BEAT), SUBDIVISIONS_PER_BEAT]


def _chart_payload(*, chart_time_s: float, extra_note_beat: float | None = None) -> dict[str, Any]:
    """最小 RPEJSON：布局与 `test_dataset._chart_dict` 一致（每个窗口都满足 r < 1 的规划条件）。"""
    lines: list[dict[str, Any]] = [{"Name": f"line-{index}", "notes": []} for index in range(2)]
    lines[0]["notes"].append(note(1, _beat(1 / 4), position_x=-RPE_STAGE_HALF_WIDTH / 2))
    lines[1]["notes"].append(note(1, _beat(1 / 2), position_x=+RPE_STAGE_HALF_WIDTH / 2))
    lines[1]["notes"].append(note(1, _beat(3 / 2), position_x=0.0))
    lines[1]["notes"].append(note(3, _beat(2), position_x=0.0, above=2))
    lines[0]["notes"].append(note(1, _beat(5 / 2), position_x=RPE_STAGE_HALF_WIDTH / 4))
    lines[1]["notes"].append(note(4, _beat(3), position_x=0.0))
    lines[0]["notes"].append(note(1, _beat(7 / 2), position_x=-RPE_STAGE_HALF_WIDTH / 4))
    # Hold(1..3/2)：窗口 [1,2) 拍否则只有 1 个遮盖单位 => 规划器会按「无法满足 r<1」跳过它
    lines[0]["notes"].append(note(2, _beat(1.0), end=_beat(1.5), position_x=0.0))
    if extra_note_beat is not None:
        lines[0]["notes"].append(note(1, _beat(extra_note_beat), position_x=0.0))
    return build_rpe(
        lines=lines,
        bpm_list=[{"bpm": BPM, "startTime": _beat(0)}],
        chart_time=chart_time_s,
    )


def _materialize(
    tmp_path: Path,
    *,
    chart_time_s: float,
    audio_s: float,
    extra_note_beat: float | None = None,
    write_meta: bool = True,
) -> tuple[Path, Path, Path]:
    """物化一份 charts / features / manifest（特征缓存的 `duration_s` = 音频时长）。"""
    chart_dir = tmp_path / "charts"
    feature_dir = tmp_path / "features"
    relative = "chart-0/chart.json"
    target = chart_dir / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(_chart_payload(chart_time_s=chart_time_s, extra_note_beat=extra_note_beat)),
        encoding="utf-8",
    )
    key = "feat-0"
    feature_dir.mkdir(parents=True, exist_ok=True)
    npz_path, meta_path = feature_cache_paths(feature_dir, key)
    emb = np.zeros((round(audio_s * MERT_FRAME_RATE_HZ), MERT_DEFAULT_FEAT_DIM), dtype=np.float32)
    meta = FeatureCacheMeta(
        rate=MERT_FRAME_RATE_HZ,
        sample_rate=MERT_SAMPLE_RATE_HZ,
        layer=12,
        model_rev="unit-test",
        duration_s=audio_s,
        original_sample_rate=MERT_SAMPLE_RATE_HZ,
        feat_dim=MERT_DEFAULT_FEAT_DIM,
        dtype="float32",
        adapter="none",
    )
    save_feature_cache(emb, npz_path, meta_path, meta)
    if not write_meta:
        meta_path.unlink()
    manifest = tmp_path / "manifest.json"
    row = PairRow(
        chart_id=1,
        song_key="song|composer",
        split="train",
        chart_path=relative,
        feature_key=key,
        difficulty=DIFFICULTY,
        fmt="rpe",
    )
    manifest.write_text(
        json.dumps(PairSplits(train=[row]).to_dict(), ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest, chart_dir, feature_dir


def _dataset(
    tmp_path: Path,
    *,
    config: dict[str, Any] | None = None,
    **materialize_kwargs: Any,
) -> ChartPairDataset:
    """默认物化「chartTime 虚高」的那张谱（真实内容 5.5 拍 / 音频 5.5 拍 / chartTime 100 拍）。"""
    options: dict[str, Any] = {
        "chart_time_s": _seconds(BOGUS_CHART_BEATS),
        "audio_s": _seconds(AUDIO_BEATS),
    }
    options.update(materialize_kwargs)
    manifest, chart_dir, feature_dir = _materialize(tmp_path, **options)
    return _reopen(manifest, chart_dir, feature_dir, config)


def _reopen(
    manifest: Path,
    chart_dir: Path,
    feature_dir: Path,
    config: dict[str, Any] | None = None,
) -> ChartPairDataset:
    """在同一份物化产物上按另一组配置开数据集（口径对照用）。"""
    return ChartPairDataset(
        DatasetConfig(
            manifest_path=manifest,
            chart_dir=chart_dir,
            feature_dir=feature_dir,
            t_window=WINDOW,
            **(config or {}),
        ),
    )


def test_audio_policy_caps_bogus_chart_time(tmp_path: Path) -> None:
    """默认口径把 τ 轴截到音频时长：100 拍 -> 5 个整窗（+ 半窗尾巴），并**显式记账**。"""
    dataset = _dataset(tmp_path)
    stats = dataset.stats()
    assert len(dataset) == 5, "τ 轴应被截到音频长度（5.5 拍 -> 5 个整窗）"
    assert stats.tau_end_truncated_rows == 1
    assert stats.tau_end_truncated_seconds == pytest.approx(
        _seconds(BOGUS_CHART_BEATS - AUDIO_BEATS),
    )
    # 真实事件全在音频范围内 ⇒ 一个都不该被口径丢掉
    assert stats.events_beyond_tau_end == 0
    assert "τ 轴截断 1 行" in stats.describe()


def test_chart_policy_keeps_legacy_behaviour(tmp_path: Path) -> None:
    """`tau_end_policy="chart"` 保留旧行为（plan 03 §9-14 未裁定前可对照）。"""
    dataset = _dataset(tmp_path)
    legacy = _reopen(
        dataset.config.manifest_path,
        dataset.config.chart_dir,
        dataset.config.feature_dir,
        {"tau_end_policy": "chart"},
    )
    assert len(legacy) == int(BOGUS_CHART_BEATS), "旧口径按 chartTime 切出 100 个窗"
    assert legacy.stats().tau_end_truncated_rows == 0


def test_explicit_tau_end_s_wins_over_policy(tmp_path: Path) -> None:
    """显式 `tau_end_s` 优先于 policy（显式覆盖不得被默认口径悄悄改写）。"""
    dataset = _dataset(tmp_path)
    explicit = _reopen(
        dataset.config.manifest_path,
        dataset.config.chart_dir,
        dataset.config.feature_dir,
        {"tau_end_s": _seconds(4.0)},
    )
    assert len(explicit) == 4
    assert explicit.stats().tau_end_truncated_rows == 0


def test_early_windows_match_between_policies(tmp_path: Path) -> None:
    """截断只砍尾巴：同一个窗口在两种口径下**逐个计数相同**（不偷偷改内容）。"""
    dataset = _dataset(tmp_path)
    legacy = _reopen(
        dataset.config.manifest_path,
        dataset.config.chart_dir,
        dataset.config.feature_dir,
        {"tau_end_policy": "chart"},
    )
    for index in range(len(dataset)):
        assert int(dataset[index].counts.sum()) == int(legacy[index].counts.sum())


def test_events_beyond_axis_are_counted(tmp_path: Path) -> None:
    """音频之外的 note 会被口径丢掉 —— 必须计数，不得静默消失。"""
    dataset = _dataset(tmp_path, extra_note_beat=8.0)
    stats = dataset.stats()
    assert stats.events_beyond_tau_end == 1
    assert stats.tau_end_truncated_rows == 1
    assert "轴外事件 1" in stats.describe()


def test_missing_feature_meta_falls_back_and_counts(tmp_path: Path) -> None:
    """特征元数据缺失时回退到谱面口径，并计入 `tau_end_fallback_rows`（不静默）。"""
    dataset = _dataset(tmp_path, write_meta=False)
    stats = dataset.stats()
    assert len(dataset) == int(BOGUS_CHART_BEATS), "回退 = 旧行为（按 chartTime）"
    assert stats.tau_end_fallback_rows == 1
    assert stats.tau_end_truncated_rows == 0


def test_row_cache_avoids_reparsing(tmp_path: Path) -> None:
    """行级 LRU：同一行的多个窗口只解析一次谱面（plan 02 §9 ④ 的取批成本）。"""
    dataset = _dataset(tmp_path, config={"chart_cache_size": 4})
    assert len(dataset) == 5, "先建好索引（索引期每行解析一次）"
    calls = 0
    original = dataset._read_chart_or_skip

    def counting(row: PairRow) -> Any:
        nonlocal calls
        calls += 1
        return original(row)

    dataset._read_chart_or_skip = counting  # type: ignore[method-assign]
    for index in range(len(dataset)):
        dataset[index]
    assert calls == 0, "索引已建好，取样本不该再解析（命中 LRU）"
    assert len(dataset) == 5


def test_row_cache_disabled_reparses(tmp_path: Path) -> None:
    """`chart_cache_size=0` 时回到旧行为（每个窗口重解析一次）——对照用。"""
    dataset = _dataset(tmp_path, config={"chart_cache_size": 0})
    assert len(dataset) == 5, "先建好索引（索引期每行解析一次）"
    calls = 0
    original = dataset._read_chart_or_skip

    def counting(row: PairRow) -> Any:
        nonlocal calls
        calls += 1
        return original(row)

    dataset._read_chart_or_skip = counting  # type: ignore[method-assign]
    for index in range(len(dataset)):
        dataset[index]
    assert calls == len(dataset)


def test_invalid_policy_rejected() -> None:
    """非法 policy 立即报错（不静默退回默认值）。"""
    with pytest.raises(ValueError, match="tau_end_policy"):
        DatasetConfig(
            manifest_path=Path("m.json"),
            chart_dir=Path("charts"),
            feature_dir=Path("features"),
            t_window=WINDOW,
            tau_end_policy="guess",
        )


def test_tau_dt_is_the_only_subdivision_source() -> None:
    """窗口的 τ 长度 = 窗口格数 x `TAU_GRID_DT`（派生量，不写 1/48）。"""
    assert pytest.approx(1.0) == WINDOW * TAU_GRID_DT, "1 拍 = 48 x (1/48)"


def test_parse_source_is_rpe(tmp_path: Path) -> None:
    """夹具自检：payload 能被主路径解析器读成 RPE（否则上面的断言测的是别的东西）。"""
    payload = _chart_payload(chart_time_s=_seconds(BOGUS_CHART_BEATS))
    chart = parse_rpejson(
        json.dumps(payload).encode("utf-8"),
        ChartSource(chart_id=1),
    )
    assert len(chart.notes) == 8
    assert chart.duration_s() == pytest.approx(_seconds(BOGUS_CHART_BEATS))
    assert len(RPE_NORMAL_TRACKS) > 0
