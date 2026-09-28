"""M7.2：门禁执行器 —— 落盘内容、解析语义与 fail-closed（默认 CI，无 GPU/权重）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from beatmorph.core.contracts import MERT_FRAME_RATE_HZ
from beatmorph.infra.config.loading import config_from_mapping
from beatmorph.infra.gates import (
    GateFailure,
    GateInputs,
    enforce_gates,
    execute_gates,
    format_gates_text,
    gates_all_passed,
    parse_gate_results,
    run_gates,
    thresholds_of,
)
from beatmorph.infra.sanity import GateResult

PROVENANCE = {
    "source": "fixtures",
    "query": "n/a",
    "fetched_at": "t",
    "purpose": "train",
    "script": "s",
    "script_version": "v1",
}
SCALE_BASE = {
    "data": {
        "source": "manifest",
        "provenance": PROVENANCE,
        "manifest_path": "data/processed/pairs.json",
        "chart_dir": "data/processed/charts",
        "feature_dir": "data/processed/features",
    },
}
SMOKE_BASE = {
    "data": {"source": "synthetic", "max_samples": 4, "provenance": PROVENANCE},
}


def _decreasing(start: float, *, factor: float = 0.5) -> object:
    """一个「越跑越小」的假 step_fn（可过 G1；不是真实模型）。"""
    state = {"value": start}

    def step() -> float:
        state["value"] *= factor
        return state["value"]

    return step


def _inputs(**overrides: object) -> GateInputs:
    duration = 2.0
    base: dict[str, object] = {
        # G1 的真实臂走训练路径（本文件里只是假回调；RFC-0037 起没有 G2 臂）
        "step_fn_real": _decreasing(1.0),
        "model_loss": 1.0,
        "baseline_loss": 10.0,
        "frames": round(duration * MERT_FRAME_RATE_HZ),
        "duration_s": duration,
        "frame_rate": MERT_FRAME_RATE_HZ,
    }
    base.update(overrides)
    return GateInputs(**base)  # type: ignore[arg-type]


def test_run_gates_returns_three_results() -> None:
    """三道门禁各一条结果（名称与 sanity.py 保持一致；G2 已删除，RFC-0037）。"""
    cfg = config_from_mapping(SMOKE_BASE)
    results = run_gates(_inputs(), cfg.gates)
    assert [result.name.split()[0] for result in results] == ["G1", "G3", "G4"]
    assert all(result.passed for result in results)


def test_thresholds_are_explicit_and_complete() -> None:
    """生效阈值必须显式落盘（plan 07 §3.1：默认值可覆盖但不可忽略）。"""
    cfg = config_from_mapping(SMOKE_BASE)
    thresholds = thresholds_of(cfg.gates)
    for key in (
        "g1_steps",
        "g1_target_ratio",
        "g3_steps",
        "g3_samples",
        "g3_chunks",
        "g3_min_improvement",
    ):
        assert key in thresholds
    assert not any(key.startswith("g2") for key in thresholds)


def test_text_round_trip_and_parse(tmp_path: Path) -> None:
    """gates.txt 必须能反解回 GateResult（fail-closed 依赖这条解析）。"""
    results = [
        GateResult("G1 单batch过拟合", True, "详情 1"),
        GateResult("G3 常数基线", False, "详情 2"),
    ]
    text = format_gates_text(results, thresholds={"g1_steps": 10}, context={"git_rev": "abc"})
    parsed = parse_gate_results(text)
    assert [(item.name, item.passed) for item in parsed] == [
        ("G1 单batch过拟合", True),
        ("G3 常数基线", False),
    ]
    assert gates_all_passed(text) is False
    assert gates_all_passed(None) is None
    assert gates_all_passed("没有任何结果行") is None


def test_execute_gates_writes_file_and_raises_on_failure(tmp_path: Path) -> None:
    """任何一项 FAIL 都必须中止，并且失败证据已经落盘。"""
    cfg = config_from_mapping(SMOKE_BASE)
    path = tmp_path / "gates.txt"
    with pytest.raises(GateFailure, match="门禁未通过"):
        execute_gates(
            _inputs(step_fn_real=lambda: 1.0), cfg, gates_path=path, context={"git_rev": "x"}
        )
    text = path.read_text(encoding="utf-8")
    assert gates_all_passed(text) is False
    assert "生效阈值" in text


def test_execute_gates_broken_gradient_is_caught(tmp_path: Path) -> None:
    """负例回归（M7.2）：梯度被切断（loss 恒定）时 G1 必须 FAIL 并中止训练。"""
    cfg = config_from_mapping(SMOKE_BASE)
    path = tmp_path / "gates.txt"
    constant = 5.0
    with pytest.raises(GateFailure) as excinfo:
        execute_gates(
            _inputs(step_fn_real=lambda: constant),
            cfg,
            gates_path=path,
            context={"git_rev": "x"},
        )
    assert "G1" in str(excinfo.value)
    assert "[FAIL] G1" in path.read_text(encoding="utf-8")


def test_enforce_gates_requires_green_file_when_scaling(tmp_path: Path) -> None:
    """fail-closed：扩大数据规模却没有（或没有全绿的）gates.txt → 拒绝启动。"""
    scaled = config_from_mapping({**SCALE_BASE, "data": {**SCALE_BASE["data"], "max_samples": 100}})
    with pytest.raises(GateFailure, match="缺少"):
        enforce_gates(run_dir=tmp_path, cfg=scaled)

    path = tmp_path / "gates.txt"
    path.write_text("没有结果行\n", encoding="utf-8")
    with pytest.raises(GateFailure, match="无法判定"):
        enforce_gates(run_dir=tmp_path, cfg=scaled)

    path.write_text("[PASS] G1 x: ok\n[FAIL] G2 y: bad\n", encoding="utf-8")
    with pytest.raises(GateFailure, match="含 FAIL"):
        enforce_gates(run_dir=tmp_path, cfg=scaled)

    path.write_text("[PASS] G1 x: ok\n", encoding="utf-8")
    enforce_gates(run_dir=tmp_path, cfg=scaled)


def test_enforce_gates_is_inert_within_smoke_scale(tmp_path: Path) -> None:
    """冒烟规模不需要 gates.txt（否则第一次跑门禁就死锁了）。"""
    cfg = config_from_mapping(SMOKE_BASE)
    enforce_gates(run_dir=tmp_path, cfg=cfg)


def test_enforce_gates_can_be_disabled_explicitly(tmp_path: Path) -> None:
    """gates.required=False 是显式开关（只允许出现在冒烟/调试配置里）。"""
    cfg = config_from_mapping(
        {
            **SCALE_BASE,
            "data": {**SCALE_BASE["data"], "max_samples": 100},
            "gates": {"required": False},
        },
    )
    enforce_gates(run_dir=tmp_path, cfg=cfg)
