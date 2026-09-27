"""长跑运维：断点续训 / checkpoint 旋转 / 增量标量（plan 07 §4.5/§4.6；默认 CI，无 GPU）。

这三件事在**短跑里看不出问题**，只在几小时到几十小时的真实训练里才暴露（崩溃丢进度、
磁盘被 checkpoint 撑爆、人力监控全程看不到曲线）。因此这里逐条钉死：

1. `train(resume_from=...)` 恢复模型 + **优化器** + 步号，并从下一步继续；
2. 旋转**先写后删**：只保留最新 `keep_last` 个步级 checkpoint，`best.pt` 永不删；
3. `logs/loss_history.jsonl` **追加**（续训后仍是同一条曲线），且每个 log_every 段都有行；
4. 配置 / 门禁 / 数据版本不一致时**拒绝恢复**（fail-closed），不给「环境已变」的实验续训。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from beatmorph.infra.artifacts import RunArtifacts
from beatmorph.infra.checkpoint import (
    BEST_CHECKPOINT_NAME,
    ResumeMismatchError,
    list_checkpoints,
    rotate_checkpoints,
)
from beatmorph.infra.config.loading import config_from_mapping
from beatmorph.infra.smoke import SmokeBatchSource
from beatmorph.infra.train_loop import HISTORY_FILENAME, train

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
    "optim": {"lr": 1e-3, "batch_size": 1, "max_steps": 4},
    "run": {"experiment": "resume-fixture", "save_every": 2, "keep_last": 3, "log_every": 1},
}


def _cfg(**overrides: object):
    """极小配置（CPU 秒级；与真实验证的装配同源）。"""
    mapping = {key: dict(value) for key, value in BASE.items()}
    for section, values in overrides.items():
        mapping.setdefault(section, {})
        mapping[section].update(values)  # type: ignore[arg-type]
    return config_from_mapping(mapping)


def _artifacts(tmp_path: Path) -> RunArtifacts:
    return RunArtifacts.create(runs_dir=tmp_path, experiment="resume-fixture", timestamp="T")


def test_train_writes_history_every_log_every(tmp_path: Path) -> None:
    """每个 log_every 段都有标量行（**在线**可见，而不是训练结束才写一次）。"""
    cfg = _cfg()
    artifacts = _artifacts(tmp_path)
    report = train(
        cfg,
        source=SmokeBatchSource(cfg, seed=0, k_lines=2),
        artifacts=artifacts,
        data_rev="rev-1",
        gates_green=True,
    )
    assert report.steps == 4
    rows = [
        json.loads(line) for line in (artifacts.logs / HISTORY_FILENAME).read_text().splitlines()
    ]
    assert [int(row["step"]) for row in rows] == [1, 2, 3, 4]
    assert all(row["step_time_s"] > 0 for row in rows)
    # 步级 checkpoint 的间隔由 run.save_every 决定
    assert [step for step, _path in list_checkpoints(artifacts.root)] == [2, 4]


def test_rotation_keeps_last_and_best(tmp_path: Path) -> None:
    """旋转只保留最新 keep_last 个，且**永不删** best.pt（先写后删的语义由 train 保证）。"""
    cfg = _cfg(run={"save_every": 1, "keep_last": 2, "log_every": 1})
    artifacts = _artifacts(tmp_path)
    train(
        cfg,
        source=SmokeBatchSource(cfg, seed=0, k_lines=2),
        artifacts=artifacts,
        data_rev="rev-1",
        gates_green=True,
    )
    assert [step for step, _path in list_checkpoints(artifacts.root)] == [3, 4]
    assert (artifacts.checkpoints / BEST_CHECKPOINT_NAME).is_file()


def test_resume_continues_from_next_step(tmp_path: Path) -> None:
    """续训：从 checkpoint 的下一步继续，且 history 是**追加**的（同一条曲线）。"""
    cfg = _cfg()
    artifacts = _artifacts(tmp_path)
    train(
        cfg,
        source=SmokeBatchSource(cfg, seed=0, k_lines=2),
        artifacts=artifacts,
        data_rev="rev-1",
        gates_green=True,
    )
    before = (artifacts.logs / HISTORY_FILENAME).read_text().splitlines()
    resume_cfg = _cfg(optim={"max_steps": 6})
    start = list_checkpoints(artifacts.root)[-1][1]
    report = train(
        resume_cfg,
        source=SmokeBatchSource(resume_cfg, seed=0, k_lines=2),
        artifacts=artifacts,
        data_rev="rev-1",
        gates_green=True,
        resume_from=start,
    )
    assert report.resumed_step == 4, "起点必须是 checkpoint 里记的步号"
    assert report.steps == 2, "本轮只跑了 6 - 4 步"
    rows = (artifacts.logs / HISTORY_FILENAME).read_text().splitlines()
    assert len(rows) == len(before) + 2, "续训的标量必须**追加**在旧曲线之后"
    assert json.loads(rows[-1])["step"] == 6


def test_resume_refuses_on_config_change(tmp_path: Path) -> None:
    """配置变了就拒绝续训（fail-closed：不把两次实验混成一个）。"""
    cfg = _cfg()
    artifacts = _artifacts(tmp_path)
    train(
        cfg,
        source=SmokeBatchSource(cfg, seed=0, k_lines=2),
        artifacts=artifacts,
        data_rev="rev-1",
        gates_green=True,
    )
    start = list_checkpoints(artifacts.root)[-1][1]
    changed = _cfg(optim={"lr": 5e-3})
    with pytest.raises(ResumeMismatchError, match="配置"):
        train(
            changed,
            source=SmokeBatchSource(changed, seed=0, k_lines=2),
            artifacts=artifacts,
            data_rev="rev-1",
            gates_green=True,
            resume_from=start,
        )


def test_resume_refuses_on_data_rev_change(tmp_path: Path) -> None:
    """数据换了也拒绝续训（同一判据的另一半）。"""
    cfg = _cfg()
    artifacts = _artifacts(tmp_path)
    train(
        cfg,
        source=SmokeBatchSource(cfg, seed=0, k_lines=2),
        artifacts=artifacts,
        data_rev="rev-1",
        gates_green=True,
    )
    start = list_checkpoints(artifacts.root)[-1][1]
    with pytest.raises(ResumeMismatchError, match="data_rev"):
        train(
            cfg,
            source=SmokeBatchSource(cfg, seed=0, k_lines=2),
            artifacts=artifacts,
            data_rev="rev-2",
            gates_green=True,
            resume_from=start,
        )


def test_rotate_checkpoints_keeps_requested_paths(tmp_path: Path) -> None:
    """rotate_checkpoints 的纯函数语义：保留最新 N 个 + 显式 keep_paths。"""
    directory = tmp_path / "checkpoints"
    directory.mkdir(parents=True)
    for step in (1, 2, 3, 4, 5):
        (directory / f"step-{step:07d}.pt").write_bytes(b"x")
    keep = directory / "best.pt"
    keep.write_bytes(b"x")
    removed = rotate_checkpoints(tmp_path, keep_last=2, keep_paths=[keep])
    assert {path.name for path in removed} == {
        "step-0000001.pt",
        "step-0000002.pt",
        "step-0000003.pt",
    }
    assert keep.is_file()
    assert [step for step, _path in list_checkpoints(tmp_path)] == [4, 5]


def test_open_existing_rejects_missing_dir(tmp_path: Path) -> None:
    """续训目标不存在时报错（不悄悄新建一个空目录）。"""
    with pytest.raises(FileNotFoundError):
        RunArtifacts.open_existing(tmp_path / "nope")
