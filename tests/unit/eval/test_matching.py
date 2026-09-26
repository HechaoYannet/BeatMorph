"""M6.1 / M6.2 匹配与相位搜索：贪心一对一、容差边界、顺序无关、跨线不参与。

容差一律取自 EvalConfig（plan 06 §8 的 mock 纪律），测试内不出现 0.020 / 0.050。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from beatmorph.decoder.events import DecodedEvent, event_sort_key
from beatmorph.eval.matching import (
    MatchResult,
    greedy_match,
    marker_equal,
    precision_recall_f1,
    search_phase_offset,
    shift_events,
)
from beatmorph.eval.protocol import EvalConfig, PhaseSearchConfig
from tests.unit.eval._builders import make_event, perfect_case, tolerance_of

CONFIG = EvalConfig()


def _matched_keys(
    result: MatchResult,
    pred: Sequence[DecodedEvent],
    gold: Sequence[DecodedEvent],
) -> list[tuple[tuple[float, int, float, int, int], tuple[float, int, float, int, int]]]:
    """匹配对的**规范键**集合（与输入下标无关，供顺序无关性断言）。"""
    return sorted(
        (event_sort_key(pred[pair.pred_index]), event_sort_key(gold[pair.gold_index]))
        for pair in result.pairs
    )


def test_perfect_match_pairs_everything_and_leaves_nothing_unmatched() -> None:
    """完全一致的预测：全部配对，无未匹配项（M6.1 的前提）。"""
    case = perfect_case(count=6)
    result = greedy_match(case.pred, case.gold, tolerance_s=tolerance_of(CONFIG))
    assert result.n_pred == len(case.pred)
    assert result.n_gold == len(case.gold)
    assert result.n_pairs == len(case.gold)
    assert result.unmatched_pred == ()
    assert result.unmatched_gold == ()
    assert result.fp == 0
    assert result.fn == 0
    assert result.rates() == (1.0, 1.0, 1.0)


def test_greedy_matching_is_one_to_one() -> None:
    """两个 pred 抢同一个 gold：只有一个被配走，另一个留在分母（plan §4.2-1）。"""
    gold = (make_event(0.0),)
    pred = (make_event(0.0), make_event(0.0, line_id=1))
    result = greedy_match(pred, gold, tolerance_s=tolerance_of(CONFIG))
    assert result.n_pairs == 1
    assert len(result.unmatched_pred) == 1
    assert result.fp == 1
    assert result.precision_denominator == len(pred)


def test_empty_sides_are_supported() -> None:
    """空 pred / 空 gold 都必须可评估（比率记 0.0，由计数区分「无样本」）。"""
    case = perfect_case(count=3)
    empty_pred = greedy_match((), case.gold, tolerance_s=tolerance_of(CONFIG))
    assert empty_pred.n_pairs == 0
    assert empty_pred.rates() == (0.0, 0.0, 0.0)
    assert empty_pred.n_pred == 0
    empty_gold = greedy_match(case.pred, (), tolerance_s=tolerance_of(CONFIG))
    assert empty_gold.n_pairs == 0
    assert empty_gold.fp == empty_gold.n_pred
    both = greedy_match((), (), tolerance_s=tolerance_of(CONFIG))
    assert both.n_pairs == 0
    assert precision_recall_f1(0, 0, 0) == (0.0, 0.0, 0.0)


def test_tolerance_is_inclusive_and_the_next_float_is_not() -> None:
    """容差边界：|Δt| == 容差算匹配；大一个 ulp 就不算（plan §4.1 的 <= 判据）。"""
    gold = (make_event(0.0),)
    for tolerance in CONFIG.tolerances_s:
        assert greedy_match((make_event(tolerance),), gold, tolerance_s=tolerance).n_pairs == 1
        assert greedy_match((make_event(-tolerance),), gold, tolerance_s=tolerance).n_pairs == 1
        above = math.nextafter(tolerance, math.inf)
        below = math.nextafter(-tolerance, -math.inf)
        assert greedy_match((make_event(above),), gold, tolerance_s=tolerance).n_pairs == 0
        assert greedy_match((make_event(below),), gold, tolerance_s=tolerance).n_pairs == 0


def test_timing_matching_ignores_line_id() -> None:
    """timing 族**不得**偷偷要求线号一致（plan §4.2-4）：时间对上就算 TP。"""
    gold = (make_event(0.0, line_id=0),)
    pred = (make_event(0.0, line_id=7),)
    result = greedy_match(pred, gold, tolerance_s=tolerance_of(CONFIG))
    assert result.n_pairs == 1
    assert not marker_equal(pred[0], gold[0], position_x_tolerance=CONFIG.position_tolerance), (
        "线号不同 → 标记不一致；timing 族仍应算匹配（否则误差来源无法定位）"
    )


def test_matching_is_order_invariant() -> None:
    """打乱两侧输入顺序不改变匹配结果（corruption 的「输入置换」不变性控制的根据）。"""
    case = perfect_case(count=8)
    tolerance = tolerance_of(CONFIG)
    forward = greedy_match(case.pred, case.gold, tolerance_s=tolerance)
    backward = greedy_match(
        tuple(reversed(case.pred)),
        tuple(reversed(case.gold)),
        tolerance_s=tolerance,
    )
    assert forward.n_pairs == backward.n_pairs
    assert _matched_keys(forward, case.pred, case.gold) == _matched_keys(
        backward,
        tuple(reversed(case.pred)),
        tuple(reversed(case.gold)),
    )


def test_same_instant_ties_are_deterministic() -> None:
    """同刻多音：贪心在并列时仍给出确定结果（同输入两次运行结果相同）。"""
    gold = (make_event(0.0, line_id=0), make_event(0.0, line_id=1))
    pred = (make_event(0.0, line_id=1), make_event(0.0, line_id=0))
    first = greedy_match(pred, gold, tolerance_s=tolerance_of(CONFIG))
    second = greedy_match(pred, gold, tolerance_s=tolerance_of(CONFIG))
    assert first == second
    assert first.n_pairs == 2
    shuffled = greedy_match(tuple(reversed(pred)), gold, tolerance_s=tolerance_of(CONFIG))
    assert _matched_keys(first, pred, gold) == _matched_keys(shuffled, tuple(reversed(pred)), gold)


def test_phase_search_recovers_an_injected_offset() -> None:
    """M6.2：不搜索时 F1 在 δ > 容差后归零；搜索开启且 δ 在范围内则恢复 >= 0.999。"""
    search = PhaseSearchConfig(enabled=True)
    config = EvalConfig(phase_search=search)
    case = perfect_case(count=8)
    delta = search.step_s * 60.0  # 60 ms：> 主容差、在 ±range 内、且落在搜索网格上
    assert delta > config.primary_tolerance
    assert delta <= search.range_s
    delayed = shift_events(case.pred, -delta)
    outcome = search_phase_offset(
        delayed,
        case.gold,
        tolerance_s=config.primary_tolerance,
        search=search,
    )
    assert outcome.f1_before == 0.0, "延迟超过容差 → 不搜索时没有任何匹配"
    assert outcome.f1_after >= 0.999, "搜索应把对齐找回来"
    assert outcome.offset_s is not None
    # 恢复到的偏移落在 [delta - 容差, delta + 容差] 内即可令 F1 = 1（搜索取
    # **首个**最大值，因此残差只要不超过容差就算恢复成功）；
    assert abs(outcome.offset_s - delta) <= config.primary_tolerance + search.step_s
    assert outcome.improved()


def test_phase_search_without_injection_stays_within_one_step() -> None:
    """M6.2：未注入偏移时，搜索返回的 offset 与 0 之差 <= phase_step_s。"""
    search = PhaseSearchConfig(enabled=True)
    config = EvalConfig(phase_search=search)
    case = perfect_case(count=8)
    outcome = search_phase_offset(
        case.pred,
        case.gold,
        tolerance_s=config.primary_tolerance,
        search=search,
    )
    assert outcome.offset_s is not None
    assert abs(outcome.offset_s) <= search.step_s
    assert outcome.f1_after == 1.0
    assert outcome.grid_size == len(search.offsets())


def test_phase_search_disabled_reports_no_offset() -> None:
    """搜索关闭时必须显式给 None（缺失 != 0：0 表示「已对齐」）。"""
    case = perfect_case(count=4)
    outcome = search_phase_offset(
        case.pred,
        case.gold,
        tolerance_s=CONFIG.primary_tolerance,
        search=PhaseSearchConfig(enabled=False),
    )
    assert outcome.searched is False
    assert outcome.offset_s is None
    assert outcome.grid_size == 0
    assert outcome.f1_before == outcome.f1_after == 1.0


def test_phase_grid_always_contains_zero() -> None:
    """搜索网格必须含 0（否则「未注入偏移 → offset ≈ 0」不是结构性结论）。"""
    search = PhaseSearchConfig(enabled=True, range_s=0.05, step_s=0.01)
    offsets = search.offsets()
    assert offsets[0] == 0.0
    assert 0.0 in offsets
    assert all(abs(value) <= search.range_s + 1e-12 for value in offsets)
