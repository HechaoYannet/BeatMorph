"""报告组装、schema 冻结与渲染（plan 06 §3.2 / §8 的集成层）。

覆盖：chart + chart -> EvalReport 全链路、冻结字段、相位两组数、合法性只统计不钳位、
meta 可追溯、渲染文本必含项。
"""

from __future__ import annotations

import json

import pytest

from beatmorph.core.contracts.phigros import RPE_STAGE_HALF_WIDTH
from beatmorph.eval.breakdown import TimeGroup
from beatmorph.eval.calibration import NLL_WARNING, CalibrationReadout, ExploratoryReadout
from beatmorph.eval.matching import shift_events
from beatmorph.eval.protocol import EvalConfig, PhaseSearchConfig
from beatmorph.eval.report import (
    ALIGNMENT_CONCLUSION,
    REQUIRED_AGGREGATE_FIELDS,
    REQUIRED_BREAKDOWN_FIELDS,
    REQUIRED_CALIBRATION_FIELDS,
    REQUIRED_EXPLORATORY_FIELDS,
    REQUIRED_LEGALITY_FIELDS,
    REQUIRED_META_FIELDS,
    REQUIRED_REPORT_FIELDS,
    AggregateReport,
    BreakdownReport,
    EvalReport,
    LegalityView,
    ReportMeta,
    assert_report_schema,
    build_report,
    evaluate_charts,
    render_text,
)
from tests.unit.eval._builders import (
    beat_fraction,
    events_to_notes,
    make_chart,
    make_note,
    perfect_case,
    varied_events,
)

CONFIG = EvalConfig()


def _charts(events_count: int = 6) -> tuple[object, object]:
    """生成谱与人类谱（同一组事件；k = 2 条判定线）。"""
    events = varied_events(events_count, beat_step=beat_fraction(4))
    notes = events_to_notes(events)
    gold = make_chart(notes=notes, k=2, difficulty=15.6, level_text="AT  Lv.16")
    pred = make_chart(notes=notes, k=2, difficulty=15.6)
    return (pred, gold)


def test_full_chain_report_has_every_frozen_field() -> None:
    """plan §3.2 的冻结字段一格不少；JSON 往返后仍然齐全。"""
    pred, gold = _charts()
    report = evaluate_charts(
        {"song-a": pred},
        {"song-a": gold},
        CONFIG,
        decode_arm="peaks",
        seed=3,
        git_rev="rev-1",
        data_rev="data-1",
    )
    payload = report.model_dump()
    assert_report_schema(payload)
    assert set(REQUIRED_REPORT_FIELDS) <= set(payload)
    assert report.meta.decode_arm == "peaks"
    assert report.meta.seed == 3
    assert report.meta.git_rev == "rev-1"
    assert report.meta.data_rev == "data-1"
    assert report.meta.tolerance_s == CONFIG.primary_tolerance
    assert report.meta.tolerances_s == CONFIG.tolerances_s
    assert report.meta.phase_search == CONFIG.phase_search.enabled
    assert report.meta.n_cases == 1
    assert report.meta.n_unique_songs == 1
    assert report.meta.x_bins == CONFIG.x_bins
    assert report.aggregate.per_chart_mean.timing_f1(CONFIG.primary_tolerance) == 1.0
    assert report.legality.n_charts == 1
    report.assert_isolation()
    assert_report_schema(json.loads(report.model_dump_json()))


REQUIRED_BY_SECTION = {
    "aggregate": REQUIRED_AGGREGATE_FIELDS,
    "breakdown": REQUIRED_BREAKDOWN_FIELDS,
    "calibration": REQUIRED_CALIBRATION_FIELDS,
    "exploratory": REQUIRED_EXPLORATORY_FIELDS,
    "legality": REQUIRED_LEGALITY_FIELDS,
    "meta": REQUIRED_META_FIELDS,
}


def test_schema_freeze_rejects_any_missing_field() -> None:
    """缺格即失败：顶层字段与各分节声明的冻结字段都要能挡住。"""
    pred, gold = _charts()
    payload = evaluate_charts({"s": pred}, {"s": gold}, CONFIG).model_dump()
    for section, names in REQUIRED_BY_SECTION.items():
        assert set(names) <= set(payload[section]), f"{section} 的冻结字段没有出现在报告里"
    for name in REQUIRED_REPORT_FIELDS:
        stripped = dict(payload)
        stripped.pop(name)
        with pytest.raises(AssertionError):
            assert_report_schema(stripped)
    for section, names in REQUIRED_BY_SECTION.items():
        for field in names:
            stripped = json.loads(json.dumps(payload))
            del stripped[section][field]
            with pytest.raises(AssertionError):
                assert_report_schema(stripped)


def test_frozen_field_lists_do_not_drift_from_the_models() -> None:
    """冻结清单与 pydantic 模型不得漂移（防止「报告少一格却没测试发现」）。"""
    assert set(REQUIRED_REPORT_FIELDS) == set(EvalReport.model_fields)
    assert set(REQUIRED_AGGREGATE_FIELDS) == set(AggregateReport.model_fields)
    assert set(REQUIRED_BREAKDOWN_FIELDS) == set(BreakdownReport.model_fields)
    assert set(REQUIRED_LEGALITY_FIELDS) <= set(LegalityView.model_fields)
    assert set(REQUIRED_CALIBRATION_FIELDS) <= set(CalibrationReadout.model_fields)
    assert set(REQUIRED_EXPLORATORY_FIELDS) <= set(ExploratoryReadout.model_fields)
    assert set(REQUIRED_META_FIELDS) <= set(ReportMeta.model_fields)
    for section in REQUIRED_BY_SECTION:
        assert section in REQUIRED_REPORT_FIELDS


def test_report_is_reproducible_field_by_field() -> None:
    """固定 seed / 同输入 → 报告逐字段一致（plan §4.9 的「可复现」验收格）。"""
    pred, gold = _charts()
    first = evaluate_charts({"s": pred}, {"s": gold}, CONFIG, seed=11).model_dump_json()
    second = evaluate_charts({"s": pred}, {"s": gold}, CONFIG, seed=11).model_dump_json()
    assert first == second


def test_phase_search_off_column_equals_the_aggregate() -> None:
    """「不搜索」这一组数就是报告主读数（结构上不可能被漏报或走样）。"""
    pred, gold = _charts()
    report = evaluate_charts({"s": pred}, {"s": gold}, CONFIG)
    assert report.phase.search_off.per_chart_mean == report.aggregate.per_chart_mean
    assert report.phase.search_off.micro == report.aggregate.micro
    assert report.phase.searched is False
    assert report.phase.search_on is None
    assert report.phase.offsets_s == {}


def test_phase_search_on_reports_two_columns_and_the_conclusion() -> None:
    """plan §4.3：搜索与不搜索两组数都要报；对齐是主要误差源时结论段必须写出来。"""
    search = PhaseSearchConfig(enabled=True)
    config = EvalConfig(phase_search=search)
    delta = search.step_s * 60.0
    events = varied_events(8, beat_step=beat_fraction(4))
    gold = make_chart(notes=events_to_notes(events), k=2)
    pred = make_chart(notes=events_to_notes(shift_events(events, -delta)), k=2)
    report = evaluate_charts({"s": pred}, {"s": gold}, config)
    assert report.phase.searched is True
    assert report.phase.search_on is not None
    off = report.phase.search_off.per_chart_mean.timing_f1(config.primary_tolerance)
    on = report.phase.search_on.per_chart_mean.timing_f1(config.primary_tolerance)
    assert off == 0.0
    assert on >= 0.999
    assert report.phase.alignment_is_main_error is True
    assert set(report.phase.offsets_s) == {"s"}
    assert ALIGNMENT_CONCLUSION in render_text(report)


def test_render_text_contains_every_mandatory_reading() -> None:
    """渲染文本必须含：NLL 警示、背面 recall、相位两组数、MAE + 量化下界、侧别基线等。"""
    pred, gold = _charts()
    report = evaluate_charts({"s": pred}, {"s": gold}, CONFIG, decode_arm="thinning")
    text = render_text(report)
    assert NLL_WARNING in text
    assert "背面 recall" in text
    assert "positionX MAE" in text
    assert "量化下界" in text
    assert "全预测正面" in text
    assert "相位搜索" in text
    assert "不搜索" in text
    assert "探索性" in text
    assert "时间组" in text
    assert "难度" in text
    assert "positionX 钳位 = 0" in text
    assert "两栏报告" in text
    assert "thinning" in text


def test_legality_view_counts_out_of_range_without_clamping() -> None:
    """越界只统计不钳位（红线 3）；越界**不是**违规项 → violation_rate = 0。"""
    out_of_range = RPE_STAGE_HALF_WIDTH * 1.2
    notes = [
        make_note(t=0.0, position_x=out_of_range),
        make_note(t=0.5, position_x=0.0),
    ]
    chart = make_chart(notes=notes, k=1)
    report = evaluate_charts({"s": chart}, {"s": chart}, CONFIG)
    assert report.legality.n_charts == 1
    assert report.legality.n_illegal == 0
    assert report.legality.violation_rate == 0.0
    assert report.legality.out_of_range_ratio > 0.0
    assert report.legality.position_x_clamped == 0.0


def test_breakdown_and_difficulty_band_are_reported_per_chart() -> None:
    """分解表与单谱难度档都必须落盘（level 文本不参与分档）。"""
    low = perfect_case("low", count=4, difficulty=15.6, level_text="sweet")
    high = perfect_case("high", count=4, difficulty=18.000004, level_text="AT  Lv.16")
    unknown = perfect_case("unknown", count=4, difficulty=None, level_text="AT  Lv.16")
    report = build_report([low, high, unknown], CONFIG)
    assert set(report.breakdown.by_difficulty) == {"15.6", "18.0", "unknown"}
    assert TimeGroup.QUARTER.value in report.breakdown.by_time_group
    bands = {chart.key: chart.difficulty_band for chart in report.per_chart}
    assert bands == {"low": "15.6", "high": "18.0", "unknown": "unknown"}
    assert report.meta.n_unique_songs == 3


def test_default_two_column_is_degenerate_and_reports_best_not_worse() -> None:
    """未给候选集时退化为单候选两栏（最优 == 固定），best_not_worse 恒为 True。"""
    report = build_report([perfect_case("a")], CONFIG, decode_arm="peaks")
    assert report.two_column.fixed_label == "peaks"
    assert report.two_column.per_chart_selection == {"a": "peaks"}
    assert report.two_column.best_not_worse is True
    assert report.two_column.per_chart_best.per_chart_mean == report.two_column.fixed.per_chart_mean


def test_evaluate_charts_requires_a_matching_prediction_for_every_gold() -> None:
    """pred 缺谱必须显式报错（不得静默少评一张）。"""
    _, gold = _charts()
    with pytest.raises(KeyError):
        evaluate_charts({}, {"s": gold}, CONFIG)
