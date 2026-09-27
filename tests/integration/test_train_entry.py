"""集成：训练入口 beatmorph-train 的端到端行为（plan 07 M7.2 / M7.3 / M7.7 / M7.8）。

覆盖：

1. **门禁执行器接线**：{BT}--gates-only{BT} 产出六件套齐全的实验目录，退出码与 gates.txt 的结论一致；
2. **负例回归**：门禁 FAIL 时训练**被中止**（退出码 5），且失败证据已落盘、checkpoint 目录为空；
3. **fail-closed**：扩大数据规模却没有全绿 gates.txt → 拒绝启动（退出码 5）；
4. **provenance 必填**：缺失 → 启动失败（退出码 3），且不产生实验目录；
5. **复现性**：固定 seed 下两次运行的前 2 步 loss 逐位一致，历史目录不被覆盖。

本文件里的合成配置刻意很小（毫秒级），因此**不**要求四道门禁全绿——「全绿」由
{BT}test_shipped_smoke_config_passes_gates{BT}（标 slow，用仓库自带的 configs/smoke.yaml）证明。
全程无网络、无权重、无 GPU；帧数一律由契约帧率派生。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from beatmorph.cli.train import EXIT_CONFIG, EXIT_GATES, EXIT_OK, main
from beatmorph.core.contracts import (
    MERT_DEFAULT_FEAT_DIM,
    MERT_FRAME_RATE_HZ,
    MERT_SAMPLE_RATE_HZ,
    SUBDIVISIONS_PER_BEAT,
    ChartSource,
)
from beatmorph.infra.artifacts import (
    GATES_FILENAME,
    METRICS_FILENAME,
    PROVENANCE_FILENAME,
    RunArtifacts,
)
from beatmorph.infra.gates import gates_all_passed, parse_gate_results

pytestmark = pytest.mark.integration

PROVENANCE = {
    "source": "fixtures:synthetic",
    "query": "n/a",
    "fetched_at": "2026-09-27T00:00:00+00:00",
    "purpose": "train",
    "script": "tests/integration/test_train_entry.py",
    "script_version": "v1",
}


def _config_payload(tmp_path: Path, *, steps: int = 2, **data_overrides: object) -> dict:
    data: dict = {
        "source": "synthetic",
        "max_samples": 4,
        "provenance": PROVENANCE,
        "t_window": 8,
        "x_bins": 4,
        "k_max": 4,
        "occlusion_ratio": 0.5,
    }
    data.update(data_overrides)
    return {
        "run": {
            "experiment": "entry",
            "purpose": "smoke",
            "runs_dir": str(tmp_path / "runs"),
            "save_every": 0,
        },
        "data": data,
        "model": {
            "d_model": 16,
            "n_heads": 2,
            "n_layers": 2,
            "window": 2,
            "global_period": 2,
            "k_max": 4,
            "audio_dim": 8,
        },
        "optim": {"lr": 0.05, "batch_size": 1, "max_steps": steps, "seed": 1},
        "gates": {
            "smoke_max_samples": 8,
            "overfit_steps": 5,
            "shuffle_steps": 3,
            "initial_head_bias": 5.0,
            "shuffle_samples": 2,
        },
    }


def _write_config(tmp_path: Path, payload: dict, name: str = "entry") -> Path:
    directory = tmp_path / "configs"
    directory.mkdir(exist_ok=True)
    (directory / f"{name}.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")
    return directory


def _run(tmp_path: Path, payload: dict, *extra: str) -> int:
    return main(
        [
            "--config-name",
            "entry",
            "--config-dir",
            str(_write_config(tmp_path, payload)),
            "--skip-env-doctor",
            *extra,
        ],
    )


def _run_dirs(tmp_path: Path) -> list[Path]:
    return sorted((tmp_path / "runs" / "entry").iterdir())


def test_gates_only_writes_the_six_piece_set(tmp_path: Path) -> None:
    """M7.7：门禁运行落齐六件套；退出码必须与 gates.txt 的结论一致（不得自相矛盾）。"""
    code = _run(tmp_path, _config_payload(tmp_path), "--gates-only")
    run_dir = _runs[0] if (_runs := _run_dirs(tmp_path)) else None
    assert run_dir is not None
    artifacts = RunArtifacts(root=run_dir)
    assert artifacts.missing() == []

    text = (run_dir / GATES_FILENAME).read_text(encoding="utf-8")
    parsed = parse_gate_results(text)
    assert len(parsed) == 4
    assert "生效阈值" in text
    assert "git_rev" in text
    all_green = all(result.passed for result in parsed)
    assert gates_all_passed(text) is all_green
    assert code == (EXIT_OK if all_green else EXIT_GATES)

    metrics = json.loads((run_dir / METRICS_FILENAME).read_text(encoding="utf-8"))
    assert metrics["gates_passed"] is all_green
    payload = json.loads((run_dir / PROVENANCE_FILENAME).read_text(encoding="utf-8"))
    assert payload["source"] == PROVENANCE["source"]
    assert payload["run_purpose"] == "smoke"


def test_gate_failure_aborts_training(tmp_path: Path) -> None:
    """M7.2 负例回归：门禁 FAIL → 训练被中止（退出码 5），失败证据落盘且无 checkpoint。"""
    payload = _config_payload(tmp_path)
    # 把 G1 的判据改成在任何优化预算下都不可能达到（必须中止，而不是「尽力而为」）
    payload["gates"] = {**payload["gates"], "overfit_steps": 1, "overfit_target_ratio": 1e-12}
    assert _run(tmp_path, payload, "--gates") == EXIT_GATES
    run_dir = _run_dirs(tmp_path)[0]
    assert "[FAIL] G1" in (run_dir / GATES_FILENAME).read_text(encoding="utf-8")
    assert (
        json.loads((run_dir / METRICS_FILENAME).read_text(encoding="utf-8"))["gates_passed"]
        is False
    )
    assert list((run_dir / "checkpoints").iterdir()) == []


def test_fail_closed_refuses_scaled_run_without_gates(tmp_path: Path) -> None:
    """扩大数据规模却没有 gates.txt → 拒绝启动（不能靠「我记得跑过门禁」）。"""
    manifest = tmp_path / "pairs.json"
    manifest.write_text("{}", encoding="utf-8")
    payload = _config_payload(
        tmp_path,
        source="manifest",
        manifest_path=str(manifest),
        chart_dir=str(tmp_path),
        feature_dir=str(tmp_path),
        max_samples=100,
    )
    # purpose=train 才能进入「扩大规模」这条分支（smoke + 全量会被配置校验先拦下）
    payload["run"] = {**payload["run"], "purpose": "train"}
    assert _run(tmp_path, payload, "--no-gates") == EXIT_GATES
    run_dir = _run_dirs(tmp_path)[0]
    assert not (run_dir / GATES_FILENAME).is_file()


def test_missing_provenance_fails_at_startup(tmp_path: Path) -> None:
    """M7.3：provenance 缺失 → 启动失败（退出码 3），且不产生实验目录。"""
    payload = _config_payload(tmp_path)
    payload["data"] = {key: value for key, value in payload["data"].items() if key != "provenance"}
    assert _run(tmp_path, payload, "--gates-only") == EXIT_CONFIG
    assert not (tmp_path / "runs").exists()


def test_runs_are_reproducible_with_fixed_seed(tmp_path: Path) -> None:
    """M7.8：固定 seed 下两次运行的前 2 步 loss **逐位一致**，且不覆盖历史目录。"""
    payload = _config_payload(tmp_path, steps=2)
    assert _run(tmp_path, payload, "--no-gates") == EXIT_OK
    assert _run(tmp_path, payload, "--no-gates") == EXIT_OK
    runs = _run_dirs(tmp_path)
    assert len(runs) == 2
    histories = [
        json.loads((run / METRICS_FILENAME).read_text(encoding="utf-8"))["loss_history"]
        for run in runs
    ]
    assert histories[0][:2] == histories[1][:2]
    assert all(isinstance(value, float) for _, value in histories[0])


def test_train_moves_every_batch_to_the_requested_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """设备搬运回归（plan 07 §9-22）：`--device` 必须**同时**作用于批次，且每一步都搬。

    修复前 `train()` 只对模型 `.to(device)`、把批次留在 CPU ⇒ 训练路径上的
    `--device cuda` 会因设备不一致直接失败。默认 CI 无 GPU，因此这里做**CPU 等价断言**：
    记录 `FieldBatch.to` 的调用次数与目标设备（修复前为 0 次）。
    """
    from beatmorph.generation.batch import FieldBatch

    calls: list[str] = []
    real_to = FieldBatch.to

    def spy(self: FieldBatch, device: object) -> FieldBatch:
        calls.append(str(device))
        return real_to(self, device)

    monkeypatch.setattr(FieldBatch, "to", spy)
    payload = _config_payload(tmp_path, steps=3)
    assert _run(tmp_path, payload, "--no-gates") == EXIT_OK
    assert calls, "train() 没有把任何批次搬到目标设备（plan 07 §9-22 回归）"
    assert len(calls) >= 3, f"每一步都必须搬批次，实际只搬了 {len(calls)} 次"
    assert set(calls) == {"cpu"}, f"目标设备应统一为请求的 cpu，实际 {sorted(set(calls))}"


def _real_manifest_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    """物化一份**真实格式**的最小数据集：rpe_min 夹具 + 与之自洽的特征缓存 + 清单。

    全部数值都从夹具与契约派生（时长来自谱面、帧数来自契约帧率、维度来自契约 feat_dim），
    因此这里没有任何被固化的物理常量。
    """
    from beatmorph.data.parsers.rpejson import parse_rpejson
    from beatmorph.data.pipeline.embed import FeatureCacheMeta
    from tests.fixtures.phigros import read_rpe_min

    raw = read_rpe_min()
    chart = parse_rpejson(raw, ChartSource(sniff_evidence="tests/fixtures/phigros/rpe_min.json"))
    duration = float(chart.duration_s())

    chart_dir = tmp_path / "charts"
    chart_dir.mkdir()
    (chart_dir / "1000.json").write_bytes(raw)

    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    frames = round(duration * MERT_FRAME_RATE_HZ)
    np.savez_compressed(
        feature_dir / "audio-1.npz",
        emb=np.zeros((frames, MERT_DEFAULT_FEAT_DIM), dtype=np.float32),
    )
    meta = FeatureCacheMeta(
        rate=MERT_FRAME_RATE_HZ,
        sample_rate=MERT_SAMPLE_RATE_HZ,
        layer=12,
        model_rev="fixture",
        duration_s=duration,
        original_sample_rate=MERT_SAMPLE_RATE_HZ,
        feat_dim=MERT_DEFAULT_FEAT_DIM,
        dtype="float32",
        adapter="none",
    )
    (feature_dir / "audio-1.meta.json").write_text(meta.model_dump_json(), encoding="utf-8")

    manifest = tmp_path / "pairs.json"
    manifest.write_text(
        json.dumps(
            {
                "train": [
                    {
                        "chart_id": 1000,
                        "song_key": "fixture-min|fixture-composer",
                        "name": "fixture-min",
                        "composer": "fixture-composer",
                        "split": "train",
                        "chart_path": "1000.json",
                        "feature_key": "audio-1",
                        "difficulty": 14.0,
                        "format": "rpe",
                    },
                ],
                "val": [],
                "test": [],
                "generalization": [],
                "skipped_no_chart": 0,
                "skipped_no_feature": 0,
            },
        ),
        encoding="utf-8",
    )
    return manifest, chart_dir, feature_dir


def test_manifest_source_trains_through_the_real_data_path(tmp_path: Path) -> None:
    """真实数据通路（plan 07 交接件第 1 项）：清单 + 特征缓存 → dataset → collate → 训练。

    与冒烟测试的区别：这批数据走的是 **RPEJSON 解析器 → FieldGrid → build_target →
    FieldBatch** 的完整链路，而不是 infra.smoke 的合成谱。
    """
    manifest, chart_dir, feature_dir = _real_manifest_fixture(tmp_path)
    payload = _config_payload(
        tmp_path,
        steps=2,
        source="manifest",
        manifest_path=str(manifest),
        chart_dir=str(chart_dir),
        feature_dir=str(feature_dir),
        max_samples=2,
        t_window=SUBDIVISIONS_PER_BEAT,
        x_bins=8,
        k_max=8,
    )
    payload["model"] = {**payload["model"], "k_max": 8, "audio_dim": MERT_DEFAULT_FEAT_DIM}
    assert _run(tmp_path, payload, "--no-gates") == EXIT_OK
    run_dir = _run_dirs(tmp_path)[0]
    metrics = json.loads((run_dir / METRICS_FILENAME).read_text(encoding="utf-8"))
    assert metrics["data_source"].startswith("manifest(")
    assert len(metrics["loss_history"]) == 2
    assert all(isinstance(value, float) for _, value in metrics["loss_history"])


def test_manifest_source_batches_by_grid_identity(tmp_path: Path) -> None:
    """采样器必须按**网格身份**组批（真实语料里连续窗口会跨越 BPM 变更点）。

    `FieldBatch` 只携带一个 `FieldGrid`（测度 J 由它派生），因此混批会静默错掉积分项——
    数据侧的 `collate_field_batch` 对此**直接抛错**。本夹具自带 120/180 两段 BPM，
    所以「连续取 index 组批」在 batch_size > 1 时必然踩雷；本测试锁住「按身份分桶 + 轮转」。
    """
    from beatmorph.infra.config.loading import config_from_mapping
    from beatmorph.infra.train_loop import ManifestBatchSource

    manifest, chart_dir, feature_dir = _real_manifest_fixture(tmp_path)
    payload = _config_payload(
        tmp_path,
        source="manifest",
        manifest_path=str(manifest),
        chart_dir=str(chart_dir),
        feature_dir=str(feature_dir),
        max_samples=8,
        t_window=SUBDIVISIONS_PER_BEAT,
        x_bins=8,
        k_max=8,
    )
    payload["model"] = {**payload["model"], "k_max": 8, "audio_dim": MERT_DEFAULT_FEAT_DIM}
    payload["optim"] = {**payload["optim"], "batch_size": 2}
    cfg = config_from_mapping(payload)

    source = ManifestBatchSource(cfg, split="train", seed=1)
    observed: set[float] = set()
    for _ in range(4):
        batch = source.batch(masked=True)
        assert batch.batch_size() == 2
        assert len(batch.grid.bpm_points) == 1  # 窗口局部网格是单段（J 在窗内恒定）
        observed.add(batch.grid.bpm_points[0].bpm)
    assert len(observed) >= 2, f"未跨桶轮转：只见到 {sorted(observed)}"


@pytest.mark.slow
def test_shipped_smoke_config_passes_gates(tmp_path: Path) -> None:
    """仓库自带 configs/smoke.yaml 必须真的能跑绿 G1-G4（否则快速开始是错的）。

    标 slow：完整门禁预算（G1 120 步 / G2 100 步 x 16 样本、**两臂配对**）在 CPU 上约 2 分钟
    （G2 改成「同输入、只打乱被遮盖标签」的严格口径后，两臂必须真的收敛才有差距；
    旧口径的十几秒来自一次恒真对照，见 plan 07 §9-19）。
    """
    assert (
        main(
            [
                "--config-name",
                "smoke",
                "--runs-dir",
                str(tmp_path / "runs"),
                "--skip-env-doctor",
                "--gates-only",
            ],
        )
        == EXIT_OK
    )
