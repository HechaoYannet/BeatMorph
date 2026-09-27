"""步时拆分与瓶颈侧巡检（plan 07 §9-42；默认 CI，无 GPU）。

「GPU 利用率不高」有两种互斥成因：**数据侧供给不足**（GPU 在等，瓶颈在数据管道）与
**计算侧本身慢**（K 大、步时长，瓶颈在算力）。旧口径把两者混进一个 step_time_s，于是只能
靠功耗和利用率反推——上一轮正是这样把 38 分钟的正常门禁误判成卡死。这里钉死三件事：

1. 每个标量行都带 data_time_s / compute_time_s，且两者之和逐行等于 step_time_s；
2. 巡检脚本会打印拆分（数据 / 计算 / 数据占比）；
3. 数据侧占比过高时巡检脚本**告警**，而不是等吞吐掉下来才被发现。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from beatmorph.infra.artifacts import RunArtifacts
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
    "optim": {"lr": 1e-3, "batch_size": 1, "max_steps": 3},
    "run": {"experiment": "split-fixture", "save_every": 2, "keep_last": 2, "log_every": 1},
}

#: 仓库根（本文件位于 tests/unit/infra/）。scripts/ 不是包，只能按路径载入。
_ROOT = Path(__file__).resolve().parents[3]


def _cfg():
    """极小配置（CPU 秒级；与真实验证的装配同源）。"""
    return config_from_mapping({key: dict(value) for key, value in BASE.items()})


def _artifacts(tmp_path: Path) -> RunArtifacts:
    return RunArtifacts.create(runs_dir=tmp_path, experiment="split-fixture", timestamp="T")


def _load_health() -> ModuleType:
    """按文件路径载入 scripts/training_health.py（它自带 __main__ 守卫，导入无副作用）。"""
    path = _ROOT / "scripts" / "training_health.py"
    spec = importlib.util.spec_from_file_location("training_health_under_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rows(*, data: float, compute: float, start: int = 1) -> list[dict[str, float]]:
    """合成标量：步时恒定、显存远低于阈值 ⇒ 只有「数据侧占比」能触发告警。"""
    step_time = data + compute
    return [
        {
            "step": float(step),
            "loss": 100.0,
            "step_time_s": step_time,
            "data_time_s": data,
            "compute_time_s": compute,
            "grad_norm": 1.0,
            "peak_vram_gib": 1.0,
        }
        for step in range(start, start + 20)
    ]


def _write_run(tmp_path: Path, rows: list[dict[str, float]]) -> Path:
    logs = tmp_path / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / HISTORY_FILENAME).write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return tmp_path


def _write_run_with_config(
    tmp_path: Path, rows: list[dict[str, float]], *, save_every: int, ckpt_step: int
) -> Path:
    """给 run 目录补上 config.yaml（run.save_every）与一个步级 checkpoint。"""
    run_dir = _write_run(tmp_path, rows)
    (run_dir / "config.yaml").write_text(
        f"optim:\n  max_steps: {int(rows[-1]['step'])}\nrun:\n  save_every: {save_every}\n",
        encoding="utf-8",
    )
    ckpt = run_dir / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / f"step-{ckpt_step}.pt").write_bytes(b"")
    return run_dir


def test_health_quiet_between_saves(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """间隔内「当前步 > 最近 checkpoint」是**正常状态**，不得告警。

    旧口径每步都告警（save_every=2000 时 1999/2000 的步都命中）⇒ 巡检退出码恒为 1，
    等于把 go/no-go 信号作废。实测于本轮长跑（step 2250 / checkpoint 2000）。
    """
    module = _load_health()
    rows = _rows(data=0.15, compute=0.85, start=2231)
    run_dir = _write_run_with_config(tmp_path, rows, save_every=2000, ckpt_step=2000)
    assert module.report(run_dir, window=20, gpu=False) == 0
    assert "落后于当前步" not in capsys.readouterr().out


def test_health_warns_when_a_save_is_actually_missed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """真漏存（间隙 > save_every）必须告警。"""
    module = _load_health()
    rows = _rows(data=0.15, compute=0.85)
    run_dir = _write_run_with_config(tmp_path, rows, save_every=2, ckpt_step=2)
    assert module.report(run_dir, window=20, gpu=False) == 1
    assert "落后于当前步（20，间隔 2）" in capsys.readouterr().out


def test_step_scalars_carry_split_that_sums_to_step_time(tmp_path: Path) -> None:
    """拆分是**无损**的：data + compute 逐行等于 step_time_s（两者取自同一次取样）。"""
    cfg = _cfg()
    artifacts = _artifacts(tmp_path)
    train(
        cfg,
        source=SmokeBatchSource(cfg, seed=0, k_lines=2),
        artifacts=artifacts,
        data_rev="rev-1",
        gates_green=True,
    )
    rows = [
        json.loads(line) for line in (artifacts.logs / HISTORY_FILENAME).read_text().splitlines()
    ]
    assert rows
    for row in rows:
        assert row["data_time_s"] > 0.0
        assert row["compute_time_s"] > 0.0
        assert row["data_time_s"] + row["compute_time_s"] == pytest.approx(row["step_time_s"])


def test_health_reports_split_and_warns_when_data_bound(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """数据侧主导 ⇒ 打印拆分**并告警**（退出码 1）。"""
    module = _load_health()
    run_dir = _write_run(tmp_path, _rows(data=0.8, compute=0.2))
    assert module.report(run_dir, window=20, gpu=False) == 1
    out = capsys.readouterr().out
    assert "步时拆分：数据 0.80s / 计算 0.20s（数据占比 80.0%，中位）" in out
    assert "数据侧占比 80%" in out


def test_health_quiet_when_compute_bound(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """计算侧主导 ⇒ 只打印拆分，不告警（退出码 0）。"""
    module = _load_health()
    run_dir = _write_run(tmp_path, _rows(data=0.15, compute=0.85))
    assert module.report(run_dir, window=20, gpu=False) == 0
    out = capsys.readouterr().out
    assert "（数据占比 15.0%，中位）" in out
    assert "数据侧占比" not in out


def test_health_tolerates_scalars_without_split(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """老曲线（无拆分字段）不炸、也不误报——正在跑的那一轮就是这种。"""
    module = _load_health()
    rows = _rows(data=0.8, compute=0.2)
    for row in rows:
        del row["data_time_s"]
        del row["compute_time_s"]
    run_dir = _write_run(tmp_path, rows)
    assert module.report(run_dir, window=20, gpu=False) == 0
    out = capsys.readouterr().out
    assert "步时拆分" not in out
