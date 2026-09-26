"""M10：按曲目切分（同曲多谱同 split）+ 同曲跨谱泛化评测集。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from beatmorph.data.dataset import load_pairs
from beatmorph.data.phira.client import (
    Manifest,
    ManifestError,
    ManifestPurpose,
    provenance_for_api,
    read_manifest,
    write_manifest,
)
from beatmorph.data.pipeline.embed import (
    SPLIT_NAMES,
    DatasetStats,
    PairSplits,
    SplitError,
    build_pairs,
    dataset_stats,
)


def _provenance() -> Any:
    return provenance_for_api(
        query="fixture",
        purpose=ManifestPurpose.TRAIN,
        script="tests/unit/data/test_pairs_split.py",
        script_version="1",
        fetched_at="2026-09-26T00:00:00+00:00",
    )


def _prepare(
    tmp_path: Path,
    rows: list[dict[str, Any]],
    *,
    write_features: bool = True,
) -> tuple[Path, Path, Path]:
    """物化 chart_dir / feature_dir 并写出带 provenance 的清单。"""
    chart_root = tmp_path / "charts"
    feature_root = tmp_path / "features"
    chart_root.mkdir(parents=True, exist_ok=True)
    feature_root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        relative = str(row["chart_path"])
        if relative.startswith("missing/"):
            continue  # 负样本：故意不落盘
        (chart_root / relative).parent.mkdir(parents=True, exist_ok=True)
        (chart_root / relative).write_text("{}", encoding="utf-8")
        if write_features and row.get("feature_key"):
            (feature_root / f"{row['feature_key']}.npz").write_bytes(b"npz")
    table = write_manifest(Manifest(provenance=_provenance(), rows=rows), tmp_path / "meta.jsonl")
    return table, chart_root, feature_root


def _row(
    chart_id: int,
    name: str,
    difficulty: float,
    *,
    composer: str = "composer",
    feature_key: str | None = None,
    n_lines: int = 10,
    n_notes: int = 100,
    tap_fraction: float = 0.6,
    back_fraction: float = 0.03,
    n_hold: int = 0,
) -> dict[str, Any]:
    key = feature_key if feature_key is not None else f"feat{chart_id}"
    return {
        "chart_id": chart_id,
        "song_key": f"{name}|{composer}",
        "name": name,
        "composer": composer,
        "difficulty": difficulty,
        "format": "rpe",
        "chart_path": f"{chart_id}/chart.json",
        "feature_key": key,
        "qc_passed": True,
        "n_lines": n_lines,
        "n_notes": n_notes,
        "out_of_visible_range": 0,
        "out_of_audio_window": 0,
        "tap_fraction": tap_fraction,
        "back_fraction": back_fraction,
        "time_span_s": 100.0,
        "bpm_min": 120.0,
        "bpm_max": 180.0,
        "max_simultaneous_onsets": 3,
        "min_same_line_same_time_gap_x": 22.5,
        "type_counts": {"1": n_notes - n_hold, "2": n_hold},
        "above_counts": {"1": n_notes - 1, "2": 1},
    }


def _four_songs() -> list[dict[str, Any]]:
    """4 首曲目，其中 song-a 有 2 张谱（不同难度 → 泛化集样本）。"""
    return [
        _row(1, "song-a", 10.0),
        _row(2, "song-a", 15.0),
        _row(3, "song-b", 12.0),
        _row(4, "song-c", 13.0),
        _row(5, "song-d", 14.0),
    ]


def test_same_song_charts_share_one_split(tmp_path: Path) -> None:
    """M10 核心断言：同曲 2 张谱必须落在同一个 split（与种子无关）。"""
    table, charts, features = _prepare(tmp_path, _four_songs())
    for seed in range(8):
        splits = build_pairs(table, charts, features, seed=seed)
        all_rows = [*splits.train, *splits.val, *splits.test]
        song_a = [row for row in all_rows if row.name == "song-a"]
        assert len(song_a) == 2
        assert len({row.split for row in song_a}) == 1, (
            "同曲多谱必须落在同一 split（否则 val 泄漏）"
        )


def test_song_sets_are_pairwise_disjoint(tmp_path: Path) -> None:
    table, charts, features = _prepare(tmp_path, _four_songs())
    splits = build_pairs(table, charts, features, seed=3)
    splits.assert_disjoint_songs()
    sets = splits.song_sets()
    assert set(sets) == set(SPLIT_NAMES)
    assert sets["train"] & sets["val"] == set()
    assert sets["train"] & sets["test"] == set()
    assert sets["val"] & sets["test"] == set()
    expected_songs = {f"{name}|composer" for name in ("song-a", "song-b", "song-c", "song-d")}
    assert set().union(*sets.values()) == expected_songs


def test_every_split_is_non_empty(tmp_path: Path) -> None:
    table, charts, features = _prepare(tmp_path, _four_songs())
    splits = build_pairs(table, charts, features, seed=1)
    for name in SPLIT_NAMES:
        assert splits.rows(name), f"{name} 不应为空"


def test_generalization_pairs_use_same_song_different_difficulty(tmp_path: Path) -> None:
    """同曲跨谱泛化集只在 train 内挑（避免与 val/test 的曲目集合相交）。"""
    table, charts, features = _prepare(tmp_path, _four_songs())
    for seed in range(20):
        splits = build_pairs(table, charts, features, seed=seed)
        if any(row.name == "song-a" for row in splits.train):
            break
    else:  # pragma: no cover - 20 个种子都放不进 train 的概率为 0
        pytest.fail("未能构造 song-a 落入 train 的切分")
    assert splits.generalization
    pair = splits.generalization[0]
    assert pair.song_key == "song-a|composer"
    assert pair.train_chart_id == 1
    assert pair.eval_chart_id == 2
    assert pair.train_difficulty == pytest.approx(10.0)
    assert pair.eval_difficulty == pytest.approx(15.0)


def test_row_order_is_deterministic(tmp_path: Path) -> None:
    table, charts, features = _prepare(tmp_path, _four_songs())
    first = build_pairs(table, charts, features, seed=11)
    second = build_pairs(table, charts, features, seed=11)
    assert first.to_dict() == second.to_dict()


def test_different_seeds_can_change_assignment(tmp_path: Path) -> None:
    table, charts, features = _prepare(tmp_path, _four_songs())
    assignments = {
        build_pairs(table, charts, features, seed=seed).to_dict()["train"][0]["song_key"]
        for seed in range(6)
    }
    assert len(assignments) > 1


def test_ratios_must_sum_to_one(tmp_path: Path) -> None:
    table, charts, features = _prepare(tmp_path, _four_songs())
    with pytest.raises(SplitError, match="ratios"):
        build_pairs(table, charts, features, ratios=(0.5, 0.1, 0.1))


def test_missing_feature_is_accounted(tmp_path: Path) -> None:
    rows = _four_songs()
    rows[1]["feature_key"] = None
    table, charts, features = _prepare(tmp_path, rows)
    splits = build_pairs(table, charts, features)
    assert splits.skipped_no_feature == 1
    assert splits.n_pairs == len(rows) - 1


def test_require_feature_false_keeps_rows(tmp_path: Path) -> None:
    rows = _four_songs()
    rows[1]["feature_key"] = None
    table, charts, features = _prepare(tmp_path, rows)
    splits = build_pairs(table, charts, features, require_feature=False)
    assert splits.n_pairs == len(rows)
    all_rows = [*splits.train, *splits.val, *splits.test]
    without_feature = [row for row in all_rows if row.chart_id == 2]
    assert len(without_feature) == 1
    assert without_feature[0].feature_key is None


def test_missing_chart_file_is_skipped(tmp_path: Path) -> None:
    rows = _four_songs()
    rows[0]["chart_path"] = "missing/chart.json"
    table, charts, features = _prepare(tmp_path, rows)
    splits = build_pairs(table, charts, features)
    assert splits.skipped_no_chart == 1
    assert splits.skipped_total == 1


def test_manifest_chart_path_stays_relative_to_chart_dir(tmp_path: Path) -> None:
    """回归（实测 2026-09-27）：清单里的 `chart_path` 必须**保持相对 chart_dir**。

    `build_pairs` 曾经把**已解析**的路径写进清单，而 `ChartPairDataset` 会再拼一次
    chart_dir ⇒ 拼出 `data/processed/charts/data/processed/charts/...`，
    20/20 行判为「谱面缺失」，真实数据通路的门禁整体装配失败（退出码 7）。
    这里锁定的是**跨模块往返**：清单 → `load_pairs` → 按 chart_dir 解析 → 文件必须存在。
    """
    rows = _four_songs()
    table, charts, features = _prepare(tmp_path, rows)
    splits = build_pairs(table, charts, features)
    manifest = tmp_path / "pairs.json"
    manifest.write_text(json.dumps(splits.to_dict()), encoding="utf-8")

    loaded = load_pairs(manifest, "train")
    assert loaded
    for pair in loaded:
        assert not Path(pair.chart_path).is_absolute()
        assert (charts / pair.chart_path).is_file(), pair.chart_path


def test_invalid_split_name_raises() -> None:
    with pytest.raises(SplitError, match="split"):
        PairSplits().rows("validation")


def test_dataset_stats_reports_repeat_rate(tmp_path: Path) -> None:
    rows = _four_songs()
    table, _charts, _features = _prepare(tmp_path, rows)
    stats = dataset_stats(table)
    assert stats.n_charts == len(rows)
    assert stats.n_unique_songs == 4
    assert stats.repeat_rate == pytest.approx(1 - 4 / len(rows))
    assert stats.same_song_repeat_rate == stats.repeat_rate
    # 唯一音频按 feature_key（音频 sha1）去重：本用例每张谱各有独立音频
    assert stats.n_unique_audio == len(rows)
    assert stats.audio_repeat_rate == pytest.approx(0.0)
    assert stats.n_notes_total == sum(row["n_notes"] for row in rows)
    assert stats.format_counts == {"rpe": len(rows)}


def test_dataset_stats_dedupes_audio_by_feature_key(tmp_path: Path) -> None:
    """同曲多谱共享同一音频（同一个 feature_key）→ 唯一音频数 < 谱面数。"""
    rows = _four_songs()
    for row in rows:
        row["feature_key"] = "shared-audio"
    table, _charts, _features = _prepare(tmp_path, rows)
    stats = dataset_stats(table)
    assert stats.n_unique_audio == 1
    assert stats.audio_repeat_rate == pytest.approx(1 - 1 / len(rows))


def test_dataset_stats_requires_provenance(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"chart_id": 1}\n', encoding="utf-8")
    with pytest.raises(ManifestError):
        dataset_stats(path)


def test_corpus_outliers_within_baseline() -> None:
    rows = [_row(index, f"s{index}", 12.0, n_lines=27) for index in range(6)]
    stats = DatasetStats.from_rows(rows)
    assert stats.corpus_outliers() == []


def test_corpus_outliers_flags_deviation() -> None:
    rows = [
        _row(index, f"s{index}", 12.0, n_lines=100, tap_fraction=0.95, back_fraction=0.5)
        for index in range(6)
    ]
    notes = DatasetStats.from_rows(rows).corpus_outliers()
    assert len(notes) == 3
    assert any("背面" in note for note in notes)
    assert any("Tap" in note for note in notes)
    assert any("线数中位" in note for note in notes)


def test_corpus_outliers_needs_enough_samples() -> None:
    stats = DatasetStats.from_rows([_row(1, "s1", 12.0, n_lines=100)])
    notes = stats.corpus_outliers()
    assert len(notes) == 1
    assert "样本数" in notes[0]


def test_manifest_round_trip_used_by_pairs(tmp_path: Path) -> None:
    table, _charts, _features = _prepare(tmp_path, _four_songs())
    manifest = read_manifest(table)
    assert manifest.provenance.purpose is ManifestPurpose.TRAIN
    assert len(manifest.rows) == 5
