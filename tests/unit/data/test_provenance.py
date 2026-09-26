"""M8：数据合规硬约束的工程落点（provenance / 不入库 / 无权重分发路径）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from beatmorph.data import LocalStorageError, assert_local_only, is_ignored, repo_root
from beatmorph.data.phira.client import (
    Manifest,
    ManifestError,
    ManifestPurpose,
    Provenance,
    provenance_for_api,
    read_manifest,
    write_manifest,
)

#: 「权重发布 / 分发」代码路径的关键词（M8 验收③：源码级断言）。
DISTRIBUTION_TOKENS: tuple[str, ...] = (
    "push_to_hub",
    "create_repo",
    "upload_folder",
    "upload_file",
    "hf_hub",
    "HfApi",
    "publish_weights",
    "release_weights",
    "weights_release",
)

FIXED_TIME = "2026-09-26T00:00:00+00:00"


def _provenance() -> Provenance:
    return provenance_for_api(
        query="page=1&pageNum=30&type=3",
        purpose=ManifestPurpose.TRAIN,
        script="tests/unit/data/test_provenance.py",
        script_version="1",
        chart_id_min=1,
        chart_id_max=9649,
        fetched_at=FIXED_TIME,
    )


def test_provenance_requires_every_field() -> None:
    with pytest.raises(ValidationError):
        Provenance()  # type: ignore[call-arg]
    base = _provenance().model_dump(mode="json")
    for field in ("source", "query", "fetched_at", "purpose", "script", "script_version"):
        broken = {key: value for key, value in base.items() if key != field}
        with pytest.raises(ValidationError):
            Provenance.model_validate(broken)
    with pytest.raises(ValidationError):
        Provenance.model_validate({**base, "source": ""})


def test_manifest_round_trip_jsonl(tmp_path: Path) -> None:
    rows = [{"chart_id": 1, "song_key": "a|b"}, {"chart_id": 2, "song_key": "c|d"}]
    path = write_manifest(Manifest(provenance=_provenance(), rows=rows), tmp_path / "meta.jsonl")
    loaded = read_manifest(path)
    assert loaded.provenance == _provenance()
    assert loaded.rows == rows


def test_manifest_without_provenance_record_raises(tmp_path: Path) -> None:
    """M8 验收②：provenance 缺失即报错（读侧）。"""
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"chart_id": 1}) + "\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="provenance"):
        read_manifest(path)


def test_write_manifest_rejects_missing_provenance(tmp_path: Path) -> None:
    """M8 验收②：provenance 缺失即报错（写侧）。"""
    # 故意绕过类型：模拟「调用方忘了传 provenance」的运行时情形
    manifest = Manifest(provenance=None, rows=[])  # type: ignore[arg-type]
    with pytest.raises(ManifestError, match="provenance"):
        write_manifest(manifest, tmp_path / "bad.jsonl")


def test_manifest_rejects_unknown_suffix(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="后缀"):
        write_manifest(Manifest(provenance=_provenance(), rows=[]), tmp_path / "meta.csv")
    target = tmp_path / "meta.csv"
    target.write_text("x", encoding="utf-8")
    with pytest.raises(ManifestError, match="后缀"):
        read_manifest(target)


def test_manifest_parquet_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    rows = [{"chart_id": 1, "song_key": "a|b"}]
    path = write_manifest(Manifest(provenance=_provenance(), rows=rows), tmp_path / "meta.parquet")
    loaded = read_manifest(path)
    assert loaded.provenance == _provenance()
    assert loaded.rows[0]["chart_id"] == 1


def test_audio_and_chart_paths_are_ignored() -> None:
    """M8 验收①：`data/` 下的音频与谱面文件不得入库。"""
    for relative in (
        "data/raw/1000/AT15.json",
        "data/raw/1000/nested/chart.json",
        "data/audio/deadbeef.ogg",
        "data/audio/deadbeef.mp3",
        "data/audio/deadbeef.wav",
        "data/audio/deadbeef.flac",
        "data/audio/deadbeef.m4a",
    ):
        assert is_ignored(relative), relative


def test_fixtures_stay_committable() -> None:
    """反例守卫：夹具必须仍可入库（`!data/fixtures/**` 的取反规则要生效）。"""
    assert not is_ignored("tests/fixtures/phigros/rpe_min.json")
    assert not is_ignored("data/fixtures/sample.wav")


def test_derived_artifacts_are_ignored() -> None:
    """派生产物（特征 / 清单）同样不得入库（`.gitignore` 的 `data/**` 规则）。"""
    for relative in (
        "data/features/deadbeef.npz",
        "data/features/deadbeef.meta.json",
        "data/manifests/charts.jsonl",
        "data/audio/deadbeef.ogg",
    ):
        assert is_ignored(relative), relative


def test_assert_local_only_blocks_uncovered_targets() -> None:
    """守卫本身可验：`data/**` 全部放行，`data/` 之外的非忽略路径必须被拦下。"""
    assert_local_only("data/raw/1000/chart.json")
    assert_local_only("data/features/deadbeef.npz")
    with pytest.raises(LocalStorageError, match="未被"):
        assert_local_only("artifacts/features/deadbeef.npz")
    assert_local_only("artifacts/features/deadbeef.npz", allow_unignored=True)


def test_repo_root_contains_gitignore() -> None:
    root = repo_root()
    assert (root / ".gitignore").is_file()
    assert (root / "pyproject.toml").is_file()
    assert (root / "beatmorph").is_dir()


def _scan_distribution_tokens(root: Path) -> list[str]:
    hits: list[str] = []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for token in DISTRIBUTION_TOKENS:
            if token in text:
                hits.append(f"{path.relative_to(root)}: {token}")
    return hits


def test_no_weight_distribution_code_path() -> None:
    """M8 验收③：源码级断言——不存在任何「权重发布/分发」代码路径。"""
    root = repo_root()
    hits = _scan_distribution_tokens(root / "beatmorph")
    assert hits == [], "发现权重发布/分发代码路径（CLAUDE.md 红线 5：最终不发布模型权重）：" + str(
        hits,
    )


def test_distribution_scanner_is_not_vacuous(tmp_path: Path) -> None:
    """扫描器自检：必须真的扫到文件，且植入关键词能被抓到。"""
    root = repo_root()
    files = list((root / "beatmorph").rglob("*.py"))
    assert len(files) >= 5
    planted = tmp_path / "planted.py"
    planted.write_text("def f():\n    return 'push_to_hub'\n", encoding="utf-8")
    assert _scan_distribution_tokens(tmp_path) == ["planted.py: push_to_hub"]
