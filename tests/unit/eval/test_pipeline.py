"""评估报告落盘接线（plan 06 §9-7 的遗留项 + plan 07 §3.2 的 metrics.json）。

覆盖（plan 06 §8 的集成层要求）：

- 图表对 -> EvalReport 的便捷入口**复用** report.evaluate_charts（两条路径逐字段一致）；
- 合并载荷向后兼容：训练摘要字段一个不改名 / 不删除 / 不重排（与真实生产者 TrainReport 对表）；
- `eval` 分节的字段顺序与 schema 冻结（缺格即失败，含负例）；
- 写盘 -> 读回逐字段一致（走真实的 RunArtifacts.write_metrics，不绕过产物层）；
- `meta.git_rev` / `meta.data_rev` 由实验上下文填充，且指向真实来源
  （本仓库的 git rev；tmp_path 里清单文件的 sha1）；
- 分节隔离：校准投毒后主判据与质量分节逐位不变（plan §4.6 / §4.7 的硬约束仍成立）。

夹具纪律（AGENTS.md §3.3 / 红线 7）：合成谱面一律走 tests/unit/eval/_builders.py（时间、坐标、
容差全部派生），本文件不出现帧率、坐标与容差的字面量。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from beatmorph.core.contracts.phigros import PhigrosChart
from beatmorph.eval.calibration import (
    NLL_WARNING,
    PRIMARY_CRITERION_SECTIONS,
    CalibrationReadout,
)
from beatmorph.eval.pipeline import (
    DATA_REV_MISSING,
    DATA_REV_PREFIX,
    DATA_REV_SYNTHETIC,
    DEFAULT_DATA_MANIFEST_PATH,
    EVAL_SECTION_KEY,
    EVAL_SECTION_ORDER,
    TRAIN_SUMMARY_FIELDS,
    ExperimentContext,
    assert_metrics_schema,
    assert_train_summary_preserved,
    eval_section,
    evaluate_chart_pair,
    merge_eval_into_metrics,
    with_experiment_context,
    write_eval_metrics,
)
from beatmorph.eval.protocol import EvalConfig
from beatmorph.eval.report import (
    REQUIRED_META_FIELDS,
    REQUIRED_REPORT_FIELDS,
    EvalReport,
    assert_report_schema,
    evaluate_charts,
)
from beatmorph.infra.artifacts import METRICS_FILENAME, RunArtifacts, file_sha1, git_rev
from beatmorph.infra.derive import derived_values
from tests.unit.eval._builders import (
    beat_fraction,
    events_to_notes,
    make_chart,
    varied_events,
)

CONFIG = EvalConfig()
REPO_ROOT = Path(__file__).resolve().parents[3]


# ── 夹具（合成谱面 / 训练载荷 / 实验目录；一律派生，不固化物理常量）──────────


def chart_pair(count: int = 6) -> tuple[PhigrosChart, PhigrosChart]:
    """生成谱与人类谱（同一组事件；k = 2 条判定线）。"""
    events = varied_events(count, beat_step=beat_fraction(4))
    notes = events_to_notes(events)
    gold = make_chart(notes=notes, k=2, difficulty=15.6, level_text="AT  Lv.16")
    pred = make_chart(notes=notes, k=2, difficulty=15.6)
    return pred, gold


def train_payload() -> dict[str, Any]:
    """训练入口当前写进 metrics.json 的载荷：**真实生产者** + 它追加的两个键。"""
    from beatmorph.infra.train_loop import TrainReport  # 惰性引入：避免收集期就拉起 torch

    payload = TrainReport(
        steps=2,
        first_loss=1.5,
        last_loss=0.75,
        best_loss=0.75,
        loss_history=[(1, 1.5), (2, 0.75)],
        checkpoints=["step-0000002.pt"],
        data_source="synthetic",
    ).to_metrics()
    payload["gates_passed"] = True
    payload["derived"] = derived_values()
    return payload


def manifest_file(tmp_path: Path) -> Path:
    """tmp_path 里的数据清单（data_rev 的真实来源）。"""
    path = tmp_path / "charts.jsonl"
    path.write_text('{"chart": "a"}\n', encoding="utf-8")
    return path


def run_dir(tmp_path: Path) -> RunArtifacts:
    """一个真实的实验产物目录（tmp_path 在仓库外，不受 .gitignore 检查约束）。"""
    return RunArtifacts.create(
        runs_dir=tmp_path, experiment="eval-wiring", timestamp="20260101-000000"
    )


# ── 1. 便捷入口：必须复用 report.evaluate_charts ──────────────────────────


def test_chart_pair_entry_reuses_the_report_pipeline() -> None:
    """单谱对入口与 report.evaluate_charts 必须逐字段一致（复用，不是第二套实现）。"""
    pred, gold = chart_pair()
    convenience = evaluate_chart_pair(pred, gold, CONFIG, key="song-a", decode_arm="peaks", seed=3)
    direct = evaluate_charts({"song-a": pred}, {"song-a": gold}, CONFIG, decode_arm="peaks", seed=3)
    assert convenience.model_dump_json() == direct.model_dump_json()
    assert [chart.key for chart in convenience.per_chart] == ["song-a"]
    assert convenience.meta.n_cases == 1
    assert convenience.meta.decode_arm == "peaks"


# ── 2. 向后兼容：训练摘要一个字段都不许动 ────────────────────────────────


def test_train_summary_field_list_matches_the_producer() -> None:
    """消费侧冻结清单必须与生产者 TrainReport.to_metrics() 的键**逐字同序**。"""
    from beatmorph.infra.train_loop import TrainReport

    produced = TrainReport(
        steps=0, first_loss=float("nan"), last_loss=float("nan"), best_loss=float("nan")
    ).to_metrics()
    assert tuple(produced) == TRAIN_SUMMARY_FIELDS


def test_merged_payload_preserves_every_train_summary_field() -> None:
    """合并后：既有顶层键原样、顺序不变，eval 分节追加在末尾。"""
    payload = train_payload()
    merged = merge_eval_into_metrics(payload, evaluate_chart_pair(*chart_pair(), CONFIG))
    assert list(merged)[: len(payload)] == list(payload)
    assert list(merged)[len(payload) :] == [EVAL_SECTION_KEY]
    for name, value in payload.items():
        assert merged[name] == value
    assert_metrics_schema(merged)


def test_train_summary_guard_rejects_rename_delete_and_reorder() -> None:
    """负例：改名 / 删除 / 重排 / 改值都必须被向后兼容判据挡下。"""
    before = {"steps": 1, "loss_history": []}
    assert_train_summary_preserved(before, {"steps": 1, "loss_history": [], EVAL_SECTION_KEY: {}})
    with pytest.raises(AssertionError, match="steps"):
        assert_train_summary_preserved(before, {"epochs": 1, "loss_history": []})
    with pytest.raises(AssertionError):
        assert_train_summary_preserved(before, {"loss_history": [], "steps": 1})
    with pytest.raises(AssertionError, match="steps"):
        assert_train_summary_preserved(before, {"steps": 2, "loss_history": []})


def test_existing_eval_section_is_never_silently_overwritten() -> None:
    """载荷里已有 eval 分节 -> 拒绝再合并（宁可报错，也不覆盖既有评估结论）。"""
    payload = train_payload()
    report = evaluate_chart_pair(*chart_pair(), CONFIG)
    merged = merge_eval_into_metrics(payload, report)
    with pytest.raises(ValueError, match=EVAL_SECTION_KEY):
        merge_eval_into_metrics(merged, report)


# ── 3. schema 与键顺序冻结（缺格即失败）──────────────────────────────────


def test_eval_section_order_is_frozen_and_matches_the_report_model() -> None:
    """冻结顺序 == EvalReport 的声明顺序 == 落盘 JSON 的键顺序；成员与 report 清单一致。"""
    assert tuple(EvalReport.model_fields) == EVAL_SECTION_ORDER
    assert set(EVAL_SECTION_ORDER) == set(REQUIRED_REPORT_FIELDS)
    section = eval_section(evaluate_chart_pair(*chart_pair(), CONFIG))
    assert tuple(section) == EVAL_SECTION_ORDER


def test_metrics_schema_freeze_rejects_any_missing_field() -> None:
    """缺一格即失败：顶层 eval 分节、分节内的字段、meta 的冻结字段都要能挡住。"""
    merged = merge_eval_into_metrics(train_payload(), evaluate_chart_pair(*chart_pair(), CONFIG))
    without_section = json.loads(json.dumps(merged))
    del without_section[EVAL_SECTION_KEY]
    with pytest.raises(AssertionError, match=EVAL_SECTION_KEY):
        assert_metrics_schema(without_section)
    for name in EVAL_SECTION_ORDER:
        stripped = json.loads(json.dumps(merged))
        del stripped[EVAL_SECTION_KEY][name]
        with pytest.raises(AssertionError):
            assert_metrics_schema(stripped)
    for name in REQUIRED_META_FIELDS:
        stripped = json.loads(json.dumps(merged))
        del stripped[EVAL_SECTION_KEY]["meta"][name]
        with pytest.raises(AssertionError):
            assert_metrics_schema(stripped)
    assert_metrics_schema(merged)


def test_metrics_schema_freeze_rejects_a_reordered_eval_section() -> None:
    """顺序也是冻结契约的一部分（同集合不同顺序必须失败）。"""
    merged = merge_eval_into_metrics(train_payload(), evaluate_chart_pair(*chart_pair(), CONFIG))
    section = merged[EVAL_SECTION_KEY]
    reordered = {name: section[name] for name in reversed(EVAL_SECTION_ORDER)}
    with pytest.raises(AssertionError, match="顺序"):
        assert_metrics_schema({**merged, EVAL_SECTION_KEY: reordered})


# ── 4. 写盘 -> 读回逐字段一致 ────────────────────────────────────────────


def test_merged_payload_roundtrips_through_disk(tmp_path: Path) -> None:
    """EvalReport 落进 runs/<exp>/<ts>/metrics.json 后，读回的每个字段都与报告一致。"""
    report = evaluate_chart_pair(*chart_pair(), CONFIG, key="song-a", decode_arm="peaks", seed=3)
    artifacts = run_dir(tmp_path)
    merged = merge_eval_into_metrics(train_payload(), report)
    path = artifacts.write_metrics(merged)
    assert path == artifacts.path(METRICS_FILENAME)

    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(merged))
    assert on_disk[EVAL_SECTION_KEY] == json.loads(report.model_dump_json())
    assert_report_schema(on_disk[EVAL_SECTION_KEY])
    assert_metrics_schema(on_disk)
    assert on_disk[EVAL_SECTION_KEY]["meta"]["seed"] == 3
    assert on_disk[EVAL_SECTION_KEY]["meta"]["decode_arm"] == "peaks"
    assert on_disk[EVAL_SECTION_KEY]["aggregate"]["per_chart_mean"]["n_gold"] > 0
    assert artifacts.path(METRICS_FILENAME).is_file()


# ── 5. meta.git_rev / meta.data_rev 由实验上下文填充 ─────────────────────


def test_experiment_context_fills_meta_from_real_sources(tmp_path: Path) -> None:
    """git_rev 指向本仓库 HEAD；data_rev 是清单文件的 sha1（文件一变读数就变）。"""
    manifest = manifest_file(tmp_path)
    context = ExperimentContext.from_paths(repo_root=REPO_ROOT, manifest_path=manifest)
    report = evaluate_chart_pair(*chart_pair(), CONFIG, key="song-a", context=context)
    assert report.meta.git_rev == git_rev(REPO_ROOT)
    assert re.fullmatch(r"[0-9a-f]{7,40}", report.meta.git_rev)
    assert report.meta.data_rev == f"{DATA_REV_PREFIX}{file_sha1(manifest)}"
    assert context.data_manifest == str(manifest)

    manifest.write_text('{"chart": "b"}\n', encoding="utf-8")
    changed = ExperimentContext.from_paths(repo_root=REPO_ROOT, manifest_path=manifest)
    assert changed.data_rev != context.data_rev
    assert changed.data_rev == f"{DATA_REV_PREFIX}{file_sha1(manifest)}"


def test_merge_fills_meta_when_the_report_carries_no_rev(tmp_path: Path) -> None:
    """报告未带 rev 时，合并点用实验上下文补齐，并且写盘后仍能读回同一个值。"""
    manifest = manifest_file(tmp_path)
    context = ExperimentContext.from_paths(repo_root=REPO_ROOT, manifest_path=manifest)
    report = evaluate_chart_pair(*chart_pair(), CONFIG, key="song-a")
    assert report.meta.git_rev == ""
    assert report.meta.data_rev == ""

    merged = merge_eval_into_metrics(train_payload(), report, context=context)
    meta = merged[EVAL_SECTION_KEY]["meta"]
    assert meta["git_rev"] == git_rev(REPO_ROOT)
    assert meta["data_rev"] == f"{DATA_REV_PREFIX}{file_sha1(manifest)}"

    artifacts = run_dir(tmp_path)
    write_eval_metrics(artifacts, report, context=context)
    on_disk = json.loads(artifacts.path(METRICS_FILENAME).read_text(encoding="utf-8"))
    assert on_disk[EVAL_SECTION_KEY]["meta"]["git_rev"] == git_rev(REPO_ROOT)
    assert (
        on_disk[EVAL_SECTION_KEY]["meta"]["data_rev"] == f"{DATA_REV_PREFIX}{file_sha1(manifest)}"
    )
    assert_metrics_schema(on_disk)


def test_missing_manifest_uses_the_declared_convention(tmp_path: Path) -> None:
    """清单缺失时 data_rev 记约定值（非空、可审计），而不是空字符串。"""
    context = ExperimentContext.from_paths(
        repo_root=REPO_ROOT, manifest_path=tmp_path / "not-there.jsonl"
    )
    assert context.data_rev == DATA_REV_MISSING
    assert context.data_rev
    assert context.git_rev == git_rev(REPO_ROOT)


def test_default_manifest_path_is_the_declared_dataset_manifest(tmp_path: Path) -> None:
    """未给清单时按 DEFAULT_DATA_MANIFEST_PATH 解析（相对仓库根，不依赖文件是否存在）。"""
    context = ExperimentContext.from_paths(repo_root=tmp_path)
    assert context.data_manifest == str(tmp_path / DEFAULT_DATA_MANIFEST_PATH)


def test_synthetic_source_has_no_manifest_and_reports_the_convention(tmp_path: Path) -> None:
    """合成数据没有清单可留痕：data_rev 记 synthetic，且不接受拼错的 data_source。"""
    context = ExperimentContext.from_paths(repo_root=REPO_ROOT, data_source="synthetic")
    assert context.data_rev == DATA_REV_SYNTHETIC
    assert context.data_manifest is None
    with pytest.raises(ValueError, match="data_source"):
        ExperimentContext.from_paths(repo_root=REPO_ROOT, data_source="synth")


def test_conflicting_rev_is_refused(tmp_path: Path) -> None:
    """报告已有 rev 且与实验上下文不一致 -> 报错，不静默改写可追溯字段。"""
    manifest = manifest_file(tmp_path)
    context = ExperimentContext.from_paths(repo_root=REPO_ROOT, manifest_path=manifest)
    pred, gold = chart_pair()
    report = evaluate_charts({"song-a": pred}, {"song-a": gold}, CONFIG, git_rev="deadbeef")
    with pytest.raises(ValueError, match="git_rev"):
        with_experiment_context(report, context)


# ── 6. 分节隔离仍成立：主判据不读 NLL ────────────────────────────────────


def test_primary_criterion_does_not_read_the_calibration_section() -> None:
    """把校准投毒后，主判据与质量分节必须逐位不变（plan §4.6 / §4.7 的结构性隔离）。"""
    report = evaluate_chart_pair(*chart_pair(), CONFIG, key="song-a")
    poisoned_readout = CalibrationReadout(
        available=True,
        nll=-1.0e9,
        nll_constant_baseline=-1.0e9,
        nll_per_event=-1.0e9,
        nll_constant_baseline_per_event=-1.0e9,
    )
    poisoned = report.model_copy(update={"calibration": poisoned_readout})
    poisoned.assert_isolation()

    clean_section = eval_section(report)
    poisoned_section = eval_section(poisoned)
    assert poisoned_section["calibration"]["nll"] == -1.0e9  # 投毒确实生效
    assert poisoned_section["aggregate"] == clean_section["aggregate"]
    assert poisoned_section["per_chart"] == clean_section["per_chart"]
    assert report.primary_score() == poisoned.primary_score()

    assert poisoned_section["calibration"]["role"] == "calibration"
    assert poisoned_section["calibration"]["warning"] == NLL_WARNING
    assert poisoned_section["calibration"]["is_primary_criterion"] is False
    assert poisoned_section["exploratory"]["is_primary_criterion"] is False
    assert PRIMARY_CRITERION_SECTIONS == ("quality",)
    assert "calibration" not in json.dumps(poisoned_section["aggregate"])


def test_nan_is_refused_in_the_frozen_payload() -> None:
    """NaN 不得进冻结 JSON（plan 06 §9-5）：分母为 0 记 0.0，缺失记 None。"""
    report = evaluate_chart_pair(*chart_pair(), CONFIG, key="song-a")
    poisoned = report.model_copy(
        update={"legality": report.legality.model_copy(update={"violation_rate": float("nan")})}
    )
    with pytest.raises(AssertionError, match="NaN"):
        eval_section(poisoned)


# ── 7. 落盘入口：在训练摘要之上追加（训练入口那一侧不用改）───────────────


def test_write_eval_metrics_appends_to_the_existing_metrics_file(tmp_path: Path) -> None:
    """先写训练摘要、再写评估：同一个 metrics.json 里既有训练字段也有 eval 分节。"""
    artifacts = run_dir(tmp_path)
    payload = train_payload()
    artifacts.write_metrics(payload)  # 模拟 cli/train.py 现有的写法

    manifest = manifest_file(tmp_path)
    context = ExperimentContext.from_paths(repo_root=REPO_ROOT, manifest_path=manifest)
    report = evaluate_chart_pair(*chart_pair(), CONFIG, key="song-a", context=context)
    path = write_eval_metrics(artifacts, report, context=context)

    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert {name: on_disk[name] for name in payload} == payload
    assert list(on_disk)[-1] == EVAL_SECTION_KEY
    assert (
        on_disk[EVAL_SECTION_KEY]["meta"]["data_rev"] == f"{DATA_REV_PREFIX}{file_sha1(manifest)}"
    )
    assert_metrics_schema(on_disk)

    with pytest.raises(ValueError, match=EVAL_SECTION_KEY):
        write_eval_metrics(artifacts, report, context=context)
