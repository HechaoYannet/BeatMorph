"""M7.7 / M7.8：实验目录六件套、provenance 落盘、不覆盖历史实验、产物不入库。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from beatmorph.infra.artifacts import (
    CHECKPOINTS_DIR,
    CONFIG_FILENAME,
    GATES_FILENAME,
    LOGS_DIR,
    METRICS_FILENAME,
    PROVENANCE_FILENAME,
    REQUIRED_ARTIFACTS,
    RunArtifacts,
    file_sha1,
    git_rev,
)
from beatmorph.infra.config.loading import config_from_mapping

PROVENANCE = {
    "source": "fixtures:unit-test",
    "query": "n/a",
    "fetched_at": "2026-09-27T00:00:00+00:00",
    "purpose": "train",
    "script": "tests/unit/infra/test_artifacts.py",
    "script_version": "v1",
}
BASE = {
    "data": {
        "source": "synthetic",
        "max_samples": 4,
        "provenance": PROVENANCE,
    },
}


def test_create_makes_layout_and_reports_missing(tmp_path: Path) -> None:
    """新建目录只含骨架：六件套其余四项由流程逐步写入，缺一即被点名。"""
    artifacts = RunArtifacts.create(
        runs_dir=tmp_path, experiment="exp", timestamp="20260101-000000"
    )
    assert artifacts.checkpoints.is_dir()
    assert artifacts.logs.is_dir()
    missing = artifacts.missing()
    assert missing == [CONFIG_FILENAME, GATES_FILENAME, PROVENANCE_FILENAME, METRICS_FILENAME]
    with pytest.raises(AssertionError, match="六件套"):
        artifacts.assert_complete()


def test_complete_run_passes_assertion(tmp_path: Path) -> None:
    """六件套齐全后 assert_complete 必须通过。"""
    cfg = config_from_mapping(BASE)
    artifacts = RunArtifacts.create(
        runs_dir=tmp_path, experiment="exp", timestamp="20260101-000000"
    )
    artifacts.write_config(cfg)
    artifacts.write_gates("健全性门禁（G1-G4）：\n  [PASS] G1 x: y")
    artifacts.write_provenance(cfg, extra={"script_rev": git_rev(Path.cwd())})
    artifacts.write_metrics({"steps": 1})
    artifacts.assert_complete()
    assert artifacts.missing() == []
    assert set(REQUIRED_ARTIFACTS) == {
        CONFIG_FILENAME,
        GATES_FILENAME,
        PROVENANCE_FILENAME,
        CHECKPOINTS_DIR,
        LOGS_DIR,
        METRICS_FILENAME,
    }


def test_config_and_provenance_are_reloadable(tmp_path: Path) -> None:
    """config.yaml 必须能原样读回；provenance 必须带来源/用途/脚本/时间四要素。"""
    cfg = config_from_mapping(BASE)
    artifacts = RunArtifacts.create(
        runs_dir=tmp_path, experiment="exp", timestamp="20260101-000000"
    )
    config_path = artifacts.write_config(cfg)
    reloaded = config_from_mapping(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    assert reloaded == cfg

    provenance_path = artifacts.write_provenance(cfg, extra={"script_rev": "deadbeef"})
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    for key in ("source", "query", "fetched_at", "purpose", "script", "script_version"):
        assert payload[key]
    assert payload["run_purpose"] == "train"
    assert payload["script_rev"] == "deadbeef"
    assert payload["acquired_at"] == PROVENANCE["fetched_at"]


def test_existing_directory_is_never_overwritten(tmp_path: Path) -> None:
    """M7.8：同时间戳的第二次运行必须顺延后缀，不覆盖历史实验。"""
    first = RunArtifacts.create(runs_dir=tmp_path, experiment="exp", timestamp="20260101-000000")
    second = RunArtifacts.create(runs_dir=tmp_path, experiment="exp", timestamp="20260101-000000")
    assert first.root != second.root
    assert second.root.name.endswith("-01")
    assert first.root.is_dir()


def test_repeated_suffix_exhaustion_raises(tmp_path: Path) -> None:
    """后缀用尽时**报错**而不是覆盖（宁可失败也不毁历史）。"""
    RunArtifacts.create(runs_dir=tmp_path, experiment="exp", timestamp="t", max_suffix=1)
    with pytest.raises(FileExistsError):
        RunArtifacts.create(runs_dir=tmp_path, experiment="exp", timestamp="t", max_suffix=1)


def test_non_ignored_repo_path_is_refused(tmp_path: Path) -> None:
    """红线 5：仓库内未被 .gitignore 覆盖的位置不得写产物（且不得留下空目录）。"""
    root = Path(__file__).resolve().parents[3]
    target = root / "tests" / "fixtures" / "should-not-be-created"
    assert not target.exists()
    with pytest.raises(AssertionError, match="gitignore"):
        RunArtifacts.create(runs_dir=target.parent, experiment="should-not-be-created")
    assert not target.exists()


def test_outside_repo_path_is_allowed(tmp_path: Path) -> None:
    """仓库外的产物目录不受 .gitignore 约束（它本来就不在版本控制里）。"""
    artifacts = RunArtifacts.create(runs_dir=tmp_path, experiment="exp", timestamp="t")
    assert artifacts.root.is_dir()


def test_text_log_and_file_sha1(tmp_path: Path) -> None:
    """logs/ 下的文本日志与数据版本留痕。"""
    artifacts = RunArtifacts.create(runs_dir=tmp_path, experiment="exp", timestamp="t")
    path = artifacts.write_text_log("gates.txt", "hello")
    assert path.read_text(encoding="utf-8") == "hello\n"
    payload = tmp_path / "data.json"
    payload.write_text("{}", encoding="utf-8")
    assert file_sha1(payload) == file_sha1(payload)
    assert file_sha1(tmp_path / "missing.json") == "missing"
