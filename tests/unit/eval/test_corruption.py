"""M6.6 指标 corruption 准入：6 类注入 + x 轴补充注入、不变性控制、自动排除。

判据（plan 06 §4.8 / 文献库 §7.6(c)）：负的 dose-rank 关联 + 最强剂量显著 + 不变性
控制通过；**未通过者不得进主报告**（main_report_metric_names 会剔除）。
"""

from __future__ import annotations

import math

import pytest

from beatmorph.eval.corruption import (
    DEFAULT_DOSES,
    STANDARD_INJECTIONS,
    X_AXIS_INJECTIONS,
    MetricSpec,
    admit_metric,
    admit_metrics,
    assert_admissible,
    corruption_sweep,
    default_injections,
    density_burst_injection,
    density_scale_injection,
    invariance_controls,
    loop_collapse_injection,
    main_report_metric_names,
    position_x_jitter_injection,
    primary_metric_specs,
    side_flip_injection,
    time_shift_injection,
    type_shuffle_injection,
)
from beatmorph.eval.protocol import EvalCase, EvalConfig
from beatmorph.eval.stats import bootstrap_interval
from tests.unit.eval._builders import beat_fraction, make_case, varied_events

CONFIG = EvalConfig()


def _corpus(count_per_case: int = 12) -> list[EvalCase]:
    """三张**完美**预测的合成谱（每张一个 cluster）：任何注入都只会让它变差。"""
    cases: list[EvalCase] = []
    for index in range(3):
        events = varied_events(count_per_case, beat_step=beat_fraction(4))
        cases.append(
            make_case(f"song-{index}", pred=events, gold=events, song=f"song-{index}"),
        )
    return cases


def _metric(prefix: str) -> MetricSpec:
    return next(spec for spec in primary_metric_specs(CONFIG) if spec.name.startswith(prefix))


def _exact(name: str) -> MetricSpec:
    return next(spec for spec in primary_metric_specs(CONFIG) if spec.name == name)


def test_injection_set_covers_the_plan_and_the_declared_x_axis_addition() -> None:
    """plan §4.8 的 ①-⑥ 必须齐全；⑦ 是本实现新增的 x 轴注入（见模块 docstring）。"""
    names = [injection.name for injection in default_injections(CONFIG)]
    assert len(names) == len(set(names))
    assert set(STANDARD_INJECTIONS) <= set(names)
    assert set(X_AXIS_INJECTIONS) <= set(names)
    assert X_AXIS_INJECTIONS == ("position_x_jitter",)


def test_injections_are_deterministic_and_dose_zero_is_identity() -> None:
    """注入的契约：dose = 0 逐位不变；同 dose 两次结果一致（可复现）。"""
    events = varied_events(8)
    for injection in default_injections(CONFIG):
        assert injection.apply(events, 0.0) == events, injection.name
        assert injection.apply(events, 0.5) == injection.apply(events, 0.5), injection.name
    flipped = side_flip_injection().apply(events, 1.0)
    assert all(event.side is not source.side for event, source in zip(flipped, events, strict=True))
    shuffled = type_shuffle_injection().apply(events, 1.0)
    assert all(
        event.note_type is not source.note_type
        for event, source in zip(shuffled, events, strict=True)
    )
    assert density_scale_injection().apply(events, 1.0) == ()
    burst = density_burst_injection().apply(events, 1.0)
    assert len({event.t_s for event in burst}) == 1
    collapsed = loop_collapse_injection().apply(events, 1.0)
    assert len({event.t_s for event in collapsed}) == 1
    jitter = position_x_jitter_injection(CONFIG).apply(events, 1.0)
    assert all(
        event.position_x != source.position_x for event, source in zip(jitter, events, strict=True)
    )
    shifted = time_shift_injection(CONFIG).apply(events, 1.0)
    assert all(event.t_s > source.t_s for event, source in zip(shifted, events, strict=True))


def test_two_declared_invariance_controls_pass() -> None:
    """M6.6 的不变性控制：不注入逐位不变 + 打乱输入顺序逐位不变。"""
    corpus = _corpus()
    injections = default_injections(CONFIG)
    baseline = _metric("timing_f1").score(corpus, CONFIG)
    results = invariance_controls(
        corpus,
        metric=_metric("timing_f1"),
        injections=injections,
        config=CONFIG,
    )
    assert [result.name for result in results] == ["identity_recompute", "input_permutation"]
    assert all(result.passed for result in results)
    for injection in injections:
        zero = [case.with_pred(injection.apply(case.pred, 0.0)) for case in corpus]
        assert [case.pred for case in zero] == [case.pred for case in corpus]
        assert _metric("timing_f1").score(zero, CONFIG) == baseline


def test_every_injection_has_a_negative_dose_rank_response() -> None:
    """每一类注入都必须让**至少一个**主报告指标显著退化，且不得让任何指标上升。"""
    corpus = _corpus()
    metrics = primary_metric_specs(CONFIG)
    for injection in default_injections(CONFIG):
        responses = [
            corruption_sweep(corpus, metric=metric, injection=injection, config=CONFIG)
            for metric in metrics
        ]
        decreasing = [response for response in responses if response.significant_decrease]
        assert decreasing, f"{injection.name}: 没有任何指标显著退化（注入无效或指标无响应）"
        for response in responses:
            assert not response.significant_increase, (
                f"{injection.name} 使 {response.metric} 显著上升（方向错误）"
            )
            if response.dose_rank_correlation is not None:
                assert response.dose_rank_correlation <= 0.0 or not response.significant_decrease


def test_time_and_density_injections_degrade_monotonically() -> None:
    """对完美基线，时间偏移与密度缩放必须**单调**退化（可解析论证的两类）。"""
    corpus = _corpus()
    injections = {injection.name: injection for injection in default_injections(CONFIG)}
    timing = _metric("timing_f1")
    for name in ("time_shift", "density_scale"):
        response = corruption_sweep(
            corpus,
            metric=timing,
            injection=injections[name],
            config=CONFIG,
            doses=DEFAULT_DOSES,
        )
        assert response.monotone_nonincreasing, f"{name}: {response.values}"
        assert response.values[0] == 1.0
        assert response.values[-1] < response.values[0]
        assert response.delta < 0.0
        assert response.significant_decrease


def test_type_and_side_injections_move_the_matching_metric_families() -> None:
    """类型注入打 event 族与类型准确率；侧别注入打侧别准确率与背面 recall。"""
    corpus = _corpus()
    injections = {injection.name: injection for injection in default_injections(CONFIG)}
    event_response = corruption_sweep(
        corpus,
        metric=_metric("event_f1"),
        injection=injections["type_shuffle"],
        config=CONFIG,
    )
    assert event_response.significant_decrease
    side_response = corruption_sweep(
        corpus,
        metric=_exact("side_accuracy"),
        injection=injections["side_flip"],
        config=CONFIG,
    )
    assert side_response.significant_decrease
    back_response = corruption_sweep(
        corpus,
        metric=_exact("back_recall"),
        injection=injections["side_flip"],
        config=CONFIG,
    )
    assert back_response.values[-1] < back_response.values[0]


def test_position_x_metric_declares_the_x_axis_response_set() -> None:
    """MAE 只声明 x 抖动为响应注入（①-⑥ 与 x 轴正交，正是声明存在的理由）。"""
    corpus = _corpus()
    injections = {injection.name: injection for injection in default_injections(CONFIG)}
    mae = _exact("position_x_mae_neg")
    assert mae.responsive_to == X_AXIS_INJECTIONS
    x_response = corruption_sweep(
        corpus,
        metric=mae,
        injection=injections["position_x_jitter"],
        config=CONFIG,
    )
    assert x_response.significant_decrease
    assert all(value < 0.0 for value in x_response.values[1:]), "MAE 变大 → 负分下降"
    time_response = corruption_sweep(
        corpus,
        metric=mae,
        injection=injections["time_shift"],
        config=CONFIG,
    )
    assert not time_response.significant_decrease


def test_admission_admits_responsive_metrics_and_excludes_a_constant_one() -> None:
    """M6.6：「未通过者被自动排除出主报告」（CI 可验）。"""
    corpus = _corpus()
    injections = default_injections(CONFIG)
    real = [_metric("timing_f1"), _metric("event_f1")]
    constant = MetricSpec(name="constant_score", score=lambda cases, config: 1.0)
    admissions = admit_metrics(
        corpus,
        metrics=[*real, constant],
        injections=injections,
        config=CONFIG,
    )
    by_name = {admission.metric: admission for admission in admissions}
    assert by_name["constant_score"].admissible is False
    assert any("没有显著退化" in reason for reason in by_name["constant_score"].reasons)
    admitted = main_report_metric_names(admissions)
    assert "constant_score" not in admitted
    assert set(admitted) == {metric.name for metric in real}
    for name in admitted:
        assert_admissible(by_name[name])
    with pytest.raises(AssertionError):
        assert_admissible(by_name["constant_score"])


def test_sign_flipped_metric_is_excluded_for_moving_the_wrong_way() -> None:
    """方向错误的指标（腐败让它变好）必须被排除——文献库 §7.6(c) 的两条反面教训。"""
    corpus = _corpus()
    timing = _metric("timing_f1")
    inverted = MetricSpec(
        name="negative_timing_f1",
        score=lambda cases, config: -timing.score(cases, config),
    )
    injection = time_shift_injection(CONFIG)
    admission = admit_metric(
        corpus,
        metric=inverted,
        injections=[injection],
        config=CONFIG,
    )
    assert admission.admissible is False
    assert admission.responses[0].significant_increase
    assert any("上升" in reason for reason in admission.reasons)


def test_bootstrap_interval_is_deterministic_in_the_seed() -> None:
    """cluster bootstrap 必须可复现：同 seed 同区间（报告可追溯的前提）。"""
    samples = [0.1, -0.5, -0.3, 0.2, -0.9]
    first = bootstrap_interval(samples, resamples=64, seed=7)
    second = bootstrap_interval(samples, resamples=64, seed=7)
    assert first == second
    assert first[0] <= first[1]
    assert bootstrap_interval([0.5], resamples=8, seed=1) == (0.5, 0.5)
    empty = bootstrap_interval([], resamples=8, seed=1)
    assert all(math.isnan(value) for value in empty), "空样本 -> 未定义区间（缺失 != 0）"


def test_dose_grid_must_start_at_zero_and_be_ascending() -> None:
    """剂量网格必须以 0 打头且升序（否则不变性控制没有对照点）。"""
    corpus = _corpus(count_per_case=4)
    injection = time_shift_injection(CONFIG)
    metric = _metric("timing_f1")
    for doses in ((), (0.5, 1.0), (0.0, 1.0, 0.5)):
        with pytest.raises(ValueError, match=r"剂量网格|doses"):
            corruption_sweep(
                corpus,
                metric=metric,
                injection=injection,
                config=CONFIG,
                doses=doses,
            )
