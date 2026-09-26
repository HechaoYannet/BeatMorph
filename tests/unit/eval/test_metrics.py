"""M6.1 / M6.3 / M6.4 事件级指标：解析解、容差边界、分母语义、macro/micro、两栏。

容差一律取自 EvalConfig（plan 06 §8）；不固化任何物理常量（AGENTS.md §3.3）。
"""

from __future__ import annotations

import math
from dataclasses import replace
from itertools import pairwise

import pytest

from beatmorph.core.contracts.phigros import RPE_STAGE_WIDTH, NoteType, Side
from beatmorph.decoder.events import DecodedEvent
from beatmorph.eval.matching import shift_events
from beatmorph.eval.metrics import (
    aggregate,
    evaluate_case,
    select_per_chart_best,
    two_column_report,
)
from beatmorph.eval.protocol import EvalCase, EvalConfig, PhaseSearchConfig
from tests.unit.eval._builders import (
    beats_to_seconds,
    make_case,
    make_event,
    perfect_case,
    tolerance_of,
    varied_events,
)

CONFIG = EvalConfig()


def _spacing(events: tuple[DecodedEvent, ...]) -> float:
    """相邻事件的最小时间间隔（测试内派生，不写字面量秒数）。"""
    times = [float(event.t_s) for event in events]
    return min(b - a for a, b in pairwise(times))


def _with_extras(count: int = 6, extras: int = 3) -> tuple[EvalCase, EvalCase]:
    """构造「只多不少」的预测：gold 全对 + 若干远离任何 gold 的多余事件。"""
    base = perfect_case(count=count)
    step = _spacing(base.gold)
    last = max(event.t_s for event in base.gold)
    extra_events = tuple(
        make_event(last + step * (index + 1), line_id=9, note_type=NoteType.TAP)
        for index in range(extras)
    )
    return (base, base.with_pred(base.pred + extra_events))


def test_perfect_prediction_scores_one_everywhere() -> None:
    """M6.1 的解析解：完全正确的预测 → 双容差双族 F1 = 1、MAE = 0、各类准确率 = 1。"""
    case = perfect_case(count=8)
    quality = evaluate_case(case, CONFIG).quality
    assert set(quality.timing) == set(CONFIG.tolerances_s)
    assert set(quality.event) == set(CONFIG.tolerances_s)
    for tolerance in CONFIG.tolerances_s:
        timing = quality.timing[tolerance]
        event = quality.event[tolerance]
        assert (timing.precision, timing.recall, timing.f1) == (1.0, 1.0, 1.0)
        assert (event.precision, event.recall, event.f1) == (1.0, 1.0, 1.0)
        assert timing.tp == len(case.gold)
    assert quality.position_x_mae == 0.0
    assert quality.type_accuracy == 1.0
    assert quality.side_accuracy == 1.0
    assert quality.back_recall == 1.0
    assert quality.back_precision == 1.0
    assert quality.phase_offset_median_s == 0.0
    assert quality.phase_offset_mean_s == 0.0
    assert quality.type_recall[int(NoteType.TAP)] == 1.0


def test_shift_beyond_every_tolerance_zeroes_f1() -> None:
    """M6.1：整体平移 δ > 容差 → F1 = 0.000（两族、两容差都为 0）。"""
    case = perfect_case(count=6)
    delta = tolerance_of(CONFIG, -1) * 2.0
    shifted = shift_events(case.pred, -delta)
    evaluation = evaluate_case(make_case("shifted", pred=shifted, gold=case.gold), CONFIG)
    for tolerance in CONFIG.tolerances_s:
        assert evaluation.quality.timing[tolerance].f1 == 0.0
        assert evaluation.quality.event[tolerance].f1 == 0.0
    assert evaluation.quality.position_x_mae is None, "没有匹配对 → MAE 缺失（不是 0）"
    assert evaluation.quality.phase_offset_median_s is None


def test_dual_tolerance_separates_a_mid_scale_offset() -> None:
    """双容差必须**分别**报：40 ms 的平移在 20 ms 口径下全错、在 50 ms 口径下全对。"""
    assert len(CONFIG.tolerances_s) >= 2
    strict, loose = tolerance_of(CONFIG, 0), tolerance_of(CONFIG, -1)
    assert strict < loose
    case = perfect_case(count=6)
    delta = strict * 2.0
    assert strict < delta < loose
    shifted = shift_events(case.pred, -delta)
    evaluation = evaluate_case(make_case("mid", pred=shifted, gold=case.gold), CONFIG)
    assert evaluation.quality.timing[strict].f1 == 0.0
    assert evaluation.quality.timing[loose].f1 == 1.0
    assert evaluation.quality.timing[strict].f1 != evaluation.quality.timing[loose].f1


def test_tolerance_boundary_is_inclusive_at_both_tolerances() -> None:
    """容差边界必须明确：恰好等于容差算匹配，1 ulp 之外不算（plan §4.1 的 <=）。"""
    gold = (make_event(0.0),)
    for tolerance in CONFIG.tolerances_s:
        exact = evaluate_case(make_case("exact", pred=(make_event(tolerance),), gold=gold), CONFIG)
        over = evaluate_case(
            make_case("over", pred=(make_event(math.nextafter(tolerance, math.inf)),), gold=gold),
            CONFIG,
        )
        assert exact.quality.timing[tolerance].tp == 1
        assert over.quality.timing[tolerance].tp == 0


def test_unmatched_generated_events_stay_in_the_precision_denominator() -> None:
    """M6.3：precision 的分母 = 匹配数 + 未匹配 pred（plan §4.2-2）。"""
    base, case = _with_extras(count=6, extras=3)
    evaluation = evaluate_case(case, CONFIG)
    family = evaluation.quality.timing[CONFIG.primary_tolerance]
    assert family.tp == len(base.gold)
    assert family.fp == 3
    assert family.n_pred == len(base.gold) + 3
    assert family.precision == pytest.approx(len(base.gold) / (len(base.gold) + 3))
    assert family.recall == 1.0
    assert evaluation.primary_match.precision_denominator == family.n_pred
    assert len(evaluation.primary_match.unmatched_pred) == 3


def test_deleting_generated_events_never_raises_precision() -> None:
    """反向测试（plan §8）：删事件不得提高 precision（否则「少生成」可刷分）。"""
    base, case = _with_extras(count=6, extras=3)
    with_extras = evaluate_case(case, CONFIG)
    dropped = with_extras.primary_match.pairs[0].pred_index
    fewer = base.with_pred(
        tuple(event for index, event in enumerate(case.pred) if index != dropped),
    )
    fewer_evaluation = evaluate_case(fewer, CONFIG)
    tolerance = CONFIG.primary_tolerance
    assert (
        fewer_evaluation.quality.timing[tolerance].precision
        < with_extras.quality.timing[tolerance].precision
    )
    assert with_extras.quality.timing[tolerance].recall == 1.0
    assert fewer_evaluation.quality.timing[tolerance].recall < 1.0


def test_removing_all_generated_events_gives_zero_recall_and_no_precision_denominator() -> None:
    """M6.3 的极端：「删除全部生成事件」→ recall = 0、分母为空（0.0 只能靠计数区分）。"""
    case = perfect_case(count=5).with_pred(())
    evaluation = evaluate_case(case, CONFIG)
    family = evaluation.quality.timing[CONFIG.primary_tolerance]
    assert family.n_pred == 0
    assert family.tp == 0
    assert family.fp == 0
    assert family.recall == 0.0
    assert family.f1 == 0.0


def test_metrics_are_bitwise_repeatable_for_the_same_input() -> None:
    """同一输入可重复：两次评估逐字段一致；打乱输入顺序也不改变结果。"""
    case = perfect_case(count=6)
    first = evaluate_case(case, CONFIG).quality.model_dump_json()
    second = evaluate_case(case, CONFIG).quality.model_dump_json()
    assert first == second
    shuffled = case.with_pred(tuple(reversed(case.pred)))
    assert evaluate_case(shuffled, CONFIG).quality.model_dump_json() == first


def test_macro_and_micro_must_differ_on_unequal_length_charts() -> None:
    """M6.3：不等长谱上 per_chart（macro）与 micro **必须不同**（防实现退化成同一个数）。"""
    short = perfect_case("short", count=2)
    long_case = perfect_case("long", count=10)
    uneven = make_case(
        "uneven",
        pred=long_case.gold + short.gold,
        gold=long_case.gold + short.gold[:1],
    )
    evaluations = [evaluate_case(short, CONFIG), evaluate_case(uneven, CONFIG)]
    report = aggregate(evaluations, CONFIG)
    tolerance = CONFIG.primary_tolerance
    macro = report.per_chart_mean.timing_f1(tolerance)
    micro = report.micro.timing_f1(tolerance)
    assert macro != micro
    tp = sum(evaluation.quality.timing[tolerance].tp for evaluation in evaluations)
    n_pred = sum(evaluation.quality.timing[tolerance].n_pred for evaluation in evaluations)
    n_gold = sum(evaluation.quality.timing[tolerance].n_gold for evaluation in evaluations)
    assert micro == pytest.approx(2.0 * tp / (n_pred + n_gold)), "micro = 合并计数后重算"
    assert macro == pytest.approx(
        sum(evaluation.quality.timing[tolerance].f1 for evaluation in evaluations)
        / len(evaluations),
    )


def test_two_column_best_is_never_worse_than_fixed() -> None:
    """两栏报告：每谱最优栏按 F1 择优，因此逐谱与汇总都不会差于固定栏。"""
    cases = [perfect_case("a", count=6), perfect_case("b", count=6)]
    delta = tolerance_of(CONFIG, -1) * 4.0
    candidates = {
        "a": {"fixed": shift_events(cases[0].pred, -delta), "tuned": cases[0].pred},
        "b": {"fixed": cases[1].pred, "tuned": shift_events(cases[1].pred, -delta)},
    }
    report = two_column_report(cases, candidates, CONFIG, fixed_label="fixed")
    tolerance = CONFIG.primary_tolerance
    assert report.best_not_worse
    assert report.per_chart_selection == {"a": "tuned", "b": "fixed"}
    assert report.per_chart_best.per_chart_mean.timing_f1(tolerance) == 1.0
    assert report.fixed.per_chart_mean.timing_f1(tolerance) == pytest.approx(0.5), (
        "macro 固定栏 = (0.0 + 1.0) / 2：一谱坏、一谱好"
    )
    assert report.per_chart_best.per_chart_mean.timing_f1(
        tolerance
    ) >= report.fixed.per_chart_mean.timing_f1(tolerance)


def test_per_chart_selection_is_reproducible_and_tie_breaks_lexicographically() -> None:
    """并列时取标签字典序最小者（确定性 → 同输入同选择，plan §4.9「可复现」）。"""
    case = perfect_case("tie", count=4)
    candidates = {"beta": case.pred, "alpha": tuple(case.pred)}
    first, _ = select_per_chart_best(case, candidates, CONFIG)
    second, _ = select_per_chart_best(case, candidates, CONFIG)
    assert first == second == "alpha"


def test_case_level_phase_search_reports_both_readings() -> None:
    """M6.2：报告主读数仍是「不搜索」；「搜索后」单独给（plan §4.3 要两组数）。"""
    search = PhaseSearchConfig(enabled=True)
    config = EvalConfig(phase_search=search)
    case = perfect_case(count=6)
    delta = search.step_s * 60.0
    evaluation = evaluate_case(case.with_pred(shift_events(case.pred, -delta)), config)
    assert evaluation.phase.searched is True
    assert evaluation.phase.f1_before == 0.0
    assert evaluation.phase.f1_after >= 0.999
    assert evaluation.quality.timing[config.primary_tolerance].f1 == 0.0, (
        "主读数不得被搜索悄悄改写（两组数必须都在）"
    )


def test_back_recall_has_discriminative_power_while_total_accuracy_does_not() -> None:
    """M6.4：背面全部漏检时，侧别总准确率仍 >= 0.97，而背面 recall = 0.000。"""
    front_count, back_count = 97, 3
    gold = tuple(
        make_event(beats_to_seconds(index), side=Side.FRONT, line_id=index % 2)
        for index in range(front_count)
    ) + tuple(
        make_event(beats_to_seconds(front_count + index), side=Side.BACK, line_id=index % 2)
        for index in range(back_count)
    )
    all_front = tuple(replace(event, side=Side.FRONT) for event in gold)
    quality = evaluate_case(make_case("back", pred=all_front, gold=gold), CONFIG).quality
    assert quality.n_back_gold == back_count
    assert quality.back_recall == 0.0
    assert quality.side_accuracy >= 0.97
    assert quality.side_all_front_baseline == pytest.approx(
        front_count / (front_count + back_count)
    )
    assert quality.side_accuracy == quality.side_all_front_baseline, (
        "全预测正面基线与总准确率相同 → 总准确率无判别力，必须并列报"
    )


def test_position_x_mae_and_quantization_lower_bound() -> None:
    """plan §4.5：MAE 只在匹配对上算，且必须并列报量化下界 dx/4（随 x_bins 变）。"""
    config = EvalConfig()
    gold = varied_events(6, x_bins=config.x_bins)
    dx = RPE_STAGE_WIDTH / config.x_bins
    shifted = tuple(replace(event, position_x=event.position_x + dx) for event in gold)
    quality = evaluate_case(make_case("x", pred=shifted, gold=gold), config).quality
    assert quality.position_x_mae == pytest.approx(dx)
    assert quality.position_x_quantization_lower_bound == pytest.approx(dx / 4.0)
    assert quality.event[config.primary_tolerance].f1 == 1.0, (
        "恰好在容差内（dx = position_x_tolerance）的 x 偏差仍算「一致」"
    )
    coarser = EvalConfig(x_bins=config.x_bins // 2)
    assert coarser.quantization_lower_bound > config.quantization_lower_bound


def test_type_accuracy_and_per_class_recall() -> None:
    """类型准确率在匹配对上算；每类 recall 的分母是**全部**该类 gold。"""
    gold = varied_events(8)
    wrong = tuple(
        replace(event, note_type=NoteType.HOLD) if event.note_type is NoteType.TAP else event
        for event in gold
    )
    quality = evaluate_case(make_case("t", pred=wrong, gold=gold), CONFIG).quality
    taps = sum(1 for event in gold if event.note_type is NoteType.TAP)
    tolerance = CONFIG.primary_tolerance
    assert quality.timing[tolerance].f1 == 1.0, "类型错不改时间匹配"
    assert quality.event[tolerance].f1 < 1.0
    assert quality.type_recall[int(NoteType.TAP)] == 0.0
    assert quality.type_accuracy == pytest.approx((len(gold) - taps) / len(gold))
    assert quality.type_recall[int(NoteType.HOLD)] == 1.0


def test_event_family_requires_every_marker() -> None:
    """event 族要求 line_id / positionX / side / type 四项一致（plan §4.1）。"""
    case = perfect_case(count=8)
    tolerance = CONFIG.primary_tolerance
    variants = {
        "side": tuple(
            replace(event, side=Side.BACK if event.side is Side.FRONT else Side.FRONT)
            for event in case.pred
        ),
        "type": tuple(
            replace(
                event,
                note_type=NoteType.DRAG if event.note_type is not NoteType.DRAG else NoteType.TAP,
            )
            for event in case.pred
        ),
        "line": tuple(replace(event, line_id=event.line_id + 1) for event in case.pred),
        "x_far": tuple(
            replace(event, position_x=event.position_x + CONFIG.position_tolerance * 4.0)
            for event in case.pred
        ),
    }
    for name, pred in variants.items():
        quality = evaluate_case(make_case(name, pred=pred, gold=case.gold), CONFIG).quality
        assert quality.timing[tolerance].f1 == 1.0, f"{name}: 时间没动，timing 族应为 1"
        assert quality.event[tolerance].f1 < 1.0, f"{name}: event 族必须暴露标记错"


def test_match_marking_restricts_candidates_to_marker_equal_pairs() -> None:
    """match_marking=True 时 timing 族也要求标记（plan §3.1 的开关，默认关闭）。"""
    case = perfect_case(count=6)
    flipped = tuple(
        replace(event, side=Side.BACK if event.side is Side.FRONT else Side.FRONT)
        for event in case.pred
    )
    strict = EvalConfig(match_marking=True)
    marked = evaluate_case(make_case("m", pred=flipped, gold=case.gold), strict)
    assert marked.quality.timing[strict.primary_tolerance].f1 == 0.0
    assert marked.quality.event[strict.primary_tolerance].f1 == 0.0
    loose = evaluate_case(make_case("m", pred=flipped, gold=case.gold), CONFIG)
    assert loose.quality.timing[CONFIG.primary_tolerance].f1 == 1.0
    assert loose.quality.event[CONFIG.primary_tolerance].f1 < 1.0
