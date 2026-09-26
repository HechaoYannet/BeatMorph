"""集成：微缩包 → 嗅探 → 解析 → 质检 → （假编码器）特征 → 配对（无网络 / 无权重 / 无 GPU）。"""

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
    ChartFormat,
    PhigrosChart,
)
from beatmorph.data.parsers import PackageAnalysis, analyze_chart_bytes, parse_chart_package
from beatmorph.data.phira.client import (
    Manifest,
    ManifestPurpose,
    provenance_for_api,
    read_manifest,
    write_manifest,
)
from beatmorph.data.phira.package import ChartPackage
from beatmorph.data.pipeline.embed import (
    SPLIT_NAMES,
    build_pairs,
    chart_row,
    dataset_stats,
    extract_features,
)
from beatmorph.data.qc import QuarantineStage
from tests.fixtures.phigros import PKG_CHART_NAME, build_pkg_zip, read_pec_masquerade

#: 标记为集成测试（默认 CI 仍会运行：只排除 slow / gpu / e2e）。
pytestmark = pytest.mark.integration

SONGS: tuple[tuple[int, str, float], ...] = (
    (1000, "song-a", 10.0),
    (1001, "song-a", 15.0),
    (1002, "song-b", 12.0),
    (1003, "song-c", 13.0),
)


class _FakeEncoder:
    """假 MERT 编码器（默认 CI 不下载权重）。"""

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, wav: Any, sample_rate: int = MERT_SAMPLE_RATE_HZ) -> Any:
        assert sample_rate == MERT_SAMPLE_RATE_HZ
        self.calls += 1
        samples = int(np.asarray(wav).shape[-1])
        frames = round(samples / MERT_SAMPLE_RATE_HZ * MERT_FRAME_RATE_HZ)
        return np.zeros((1, frames, MERT_DEFAULT_FEAT_DIM), dtype=np.float32)

    def output_frame_rate(self) -> float:
        return MERT_FRAME_RATE_HZ


def _fake_loader(path: Path) -> tuple[Any, int]:
    return np.zeros(Path(path).stat().st_size, dtype=np.float32), MERT_SAMPLE_RATE_HZ


def _fake_resampler(wav: Any, source_rate: int, target_rate: int) -> Any:
    assert target_rate == MERT_SAMPLE_RATE_HZ
    return np.asarray(wav)


def _package_analysis(tmp_path: Path) -> tuple[ChartPackage, PackageAnalysis]:
    """步骤 1-2：微缩包 → 定位（M4）→ 嗅探/解析/质检（M3/M5/M6/M7）。"""
    package = ChartPackage.open(build_pkg_zip(tmp_path / "pkg_min.zip"))
    assert package.chart_file == PKG_CHART_NAME
    largest_json = package.largest_json_entry()
    assert largest_json is not None
    assert largest_json != PKG_CHART_NAME, "必须按 info.yml.chart 定位，而不是取最大的 json"
    assert package.missing_music, "夹具按合规要求不含音频"

    analysis = parse_chart_package(package, chart_id=1000)
    assert analysis.accepted
    assert analysis.fmt is ChartFormat.RPE
    assert analysis.chart is not None
    assert analysis.qc is not None
    assert analysis.qc.passed
    assert analysis.chart.meta.difficulty == 1.0, "定数必须来自 info.yml"
    assert PhigrosChart.model_validate(analysis.chart.model_dump()) == analysis.chart
    return package, analysis


def _feature_keys(tmp_path: Path) -> tuple[list[str], Path]:
    """步骤 3：假编码器提取特征（M9），返回缓存键与特征目录。"""
    audio_path = tmp_path / "audio" / "fixture.bin"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"\x00" * MERT_SAMPLE_RATE_HZ)
    feature_dir = tmp_path / "features"
    meta = extract_features(
        audio_path,
        feature_dir,
        _FakeEncoder(),
        loader=_fake_loader,
        resampler=_fake_resampler,
    )
    assert meta.rate == MERT_FRAME_RATE_HZ
    assert meta.feat_dim == MERT_DEFAULT_FEAT_DIM
    keys = sorted(path.stem for path in feature_dir.glob("*.npz"))
    assert len(keys) == 1
    return keys, feature_dir


def test_mini_pipeline_parse_and_qc(tmp_path: Path) -> None:
    """M3/M4/M6/M7：微缩包 → 嗅探 → 解析 → 质检 → IR 契约往返。"""
    package, analysis = _package_analysis(tmp_path)
    assert analysis.chart is not None
    assert len(analysis.chart.lines) >= 1
    assert analysis.chart.source.format is ChartFormat.RPE
    assert analysis.chart.source.chart_sha1
    assert "eventLayers" in analysis.chart.source.sniff_evidence
    chart_bytes = package.chart_bytes()
    assert chart_bytes.startswith(b"{"), "谱面条目应当是 JSON 内容（与后缀无关）"
    assert analysis.chart.source.chart_file == package.chart_file


def test_mini_pipeline_features_and_splits(tmp_path: Path) -> None:
    """M9/M10：特征提取 → 带 provenance 的清单 → 按曲目切分 → 全库统计。"""
    _package, analysis = _package_analysis(tmp_path)
    feature_keys, feature_dir = _feature_keys(tmp_path)

    chart_root = tmp_path / "charts"
    for chart_id, _name, _difficulty in SONGS:
        destination = chart_root / str(chart_id) / "chart.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("{}", encoding="utf-8")

    rows = [
        chart_row(
            chart_id=chart_id,
            name=name,
            composer="fixture-composer",
            difficulty=difficulty,
            fmt=ChartFormat.RPE,
            chart_path=f"{chart_id}/chart.json",
            feature_key=feature_keys[0],
            qc=analysis.qc,
        )
        for chart_id, name, difficulty in SONGS
    ]
    table = write_manifest(
        Manifest(
            provenance=provenance_for_api(
                query="fixture-only（无网络请求）",
                purpose=ManifestPurpose.TRAIN,
                script="tests/integration/test_pipeline_min.py",
                script_version="1",
                fetched_at="2026-09-26T00:00:00+00:00",
            ),
            rows=rows,
        ),
        tmp_path / "manifests" / "charts.jsonl",
    )
    assert read_manifest(table).provenance.query.startswith("fixture-only")

    for seed in range(10):
        splits = build_pairs(table, chart_root, feature_dir, seed=seed)
        all_rows = [*splits.train, *splits.val, *splits.test]
        if any(row.name == "song-a" for row in splits.train):
            break
    else:  # pragma: no cover
        pytest.fail("未能把 song-a 放进 train")

    splits.assert_disjoint_songs()
    assert splits.n_pairs == len(SONGS)
    song_a = [row for row in all_rows if row.name == "song-a"]
    assert len(song_a) == 2
    assert len({row.split for row in song_a}) == 1
    assert splits.generalization, "同曲不同难度必须产出泛化对"
    assert splits.generalization[0].song_key == "song-a|fixture-composer"
    for name in SPLIT_NAMES:
        assert splits.rows(name)

    # ── 5) 全库统计（M9/M10：唯一曲目数与同曲重复率）──
    stats = dataset_stats(table)
    assert stats.n_charts == len(SONGS)
    assert stats.n_unique_songs == 3
    assert stats.repeat_rate == pytest.approx(1 - 3 / len(SONGS))
    assert stats.n_unique_audio == 1  # 4 张谱共用同一段（假）音频
    assert stats.audio_repeat_rate == pytest.approx(1 - 1 / len(SONGS))
    assert isinstance(json.loads(json.dumps(stats.to_dict())), dict)


def test_pec_masquerade_is_rejected_and_accounted() -> None:
    """M3：PEC 内容 + .json 后缀 → 拒收 + 记账（**绝不**进 RPE 解析路径）。"""
    analysis = analyze_chart_bytes(read_pec_masquerade(), chart_id=7039, chart_file="24432296.json")
    assert analysis.fmt is ChartFormat.PEC
    assert not analysis.accepted
    assert analysis.chart is None
    assert analysis.quarantine is not None
    assert analysis.quarantine.stage is QuarantineStage.SNIFF
    assert "pec" in analysis.quarantine.reasons[0].lower()
    assert analysis.quarantine.chart_id == 7039


def test_invalid_chart_reference_in_package_raises(tmp_path: Path) -> None:
    """M4 负样本：info.yml.chart 指向不存在的条目必须报错，**不得**回退猜测。"""
    from beatmorph.data.phira.package import ChartPackageError
    from tests.fixtures.phigros import pkg_info_text

    broken = pkg_info_text().replace(f"chart: {PKG_CHART_NAME}", "chart: missing_entry.json")
    path = build_pkg_zip(tmp_path / "broken.zip", info_text=broken)
    with pytest.raises(ChartPackageError, match="不存在"):
        ChartPackage.open(path)
