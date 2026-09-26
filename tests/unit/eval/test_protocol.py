"""配置契约、派生量与统计辅助（plan 06 §3.1 的 EvalConfig / §4.5 的量化下界）。

红线 7：所有派生量都必须能由契约常量复算（dx = RPE_STAGE_WIDTH / x_bins）。
"""

from __future__ import annotations

import math

import pytest

from beatmorph.core.contracts.phigros import RPE_STAGE_WIDTH, BpmPoint
from beatmorph.eval.metrics import aggregate, evaluate_case, mean_strata
from beatmorph.eval.protocol import (
    DEFAULT_TOLERANCES_S,
    AverageMode,
    DecodeRegime,
    EvalCase,
    EvalConfig,
    PhaseSearchConfig,
    TimeGroupRule,
)
from beatmorph.eval.stats import (
    average_ranks,
    mean_or_nan,
    pearson_correlation,
    percentile_interval,
    spearman_rank_correlation,
)
from tests.unit.eval._builders import make_case, make_event, perfect_case, tolerance_of

CONFIG = EvalConfig()


def test_default_config_declares_both_tolerances_and_derived_quantities() -> None:
    """双容差默认值与所有派生量（dx / 量化下界 / positionX 容差）都自洽。"""
    assert CONFIG.tolerances_s == DEFAULT_TOLERANCES_S
    assert len(CONFIG.tolerances_s) >= 2
    assert CONFIG.primary_tolerance == CONFIG.tolerances_s[0]
    assert CONFIG.dx == pytest.approx(RPE_STAGE_WIDTH / CONFIG.x_bins)
    assert CONFIG.position_tolerance == pytest.approx(CONFIG.dx)
    assert CONFIG.quantization_lower_bound == pytest.approx(CONFIG.dx / 4.0)
    assert CONFIG.average is AverageMode.BOTH
    assert CONFIG.decode_regime is DecodeRegime.FIXED
    assert CONFIG.time_group_rule is TimeGroupRule.POSITION
    labels = CONFIG.tolerance_keys()
    assert set(labels) == set(CONFIG.tolerances_s)
    assert labels[CONFIG.primary_tolerance].endswith("ms")


def test_explicit_overrides_are_respected() -> None:
    """显式口径必须覆盖默认（positionX 容差 / x_bins / 主容差 / 单容差配置）。"""
    custom = EvalConfig(
        tolerances_s=(tolerance_of(CONFIG),),
        x_bins=CONFIG.x_bins // 2,
        position_x_tolerance=0.0,
    )
    assert custom.primary_tolerance == tolerance_of(CONFIG)
    assert custom.position_tolerance == 0.0
    assert custom.quantization_lower_bound == pytest.approx(RPE_STAGE_WIDTH / custom.x_bins / 4.0)
    assert custom.quantization_lower_bound > CONFIG.quantization_lower_bound


def test_config_rejects_incoherent_parameters() -> None:
    """配置自洽性校验：容差 / 主容差 / x_bins / bootstrap 的非法值必须报错。"""
    good = tolerance_of(CONFIG)
    with pytest.raises(ValueError, match="不得为空"):
        EvalConfig(tolerances_s=())
    with pytest.raises(ValueError, match="必须为正"):
        EvalConfig(tolerances_s=(0.0,))
    with pytest.raises(ValueError, match="升序"):
        EvalConfig(tolerances_s=(good * 2.0, good))
    with pytest.raises(ValueError, match="不得重复"):
        EvalConfig(tolerances_s=(good, good))
    with pytest.raises(ValueError, match="主容差"):
        EvalConfig(tolerances_s=(good,), primary_tolerance_s=good * 3.0)
    with pytest.raises(ValueError, match="x_bins"):
        EvalConfig(x_bins=0)
    with pytest.raises(ValueError, match="position_x_tolerance"):
        EvalConfig(position_x_tolerance=-1.0)
    with pytest.raises(ValueError, match="bootstrap_resamples"):
        EvalConfig(bootstrap_resamples=0)
    with pytest.raises(ValueError, match="phase_range_s"):
        PhaseSearchConfig(range_s=-1.0)
    with pytest.raises(ValueError, match="phase_step_s"):
        PhaseSearchConfig(step_s=0.0)


def test_with_phase_search_returns_a_new_config() -> None:
    """frozen 配置的开关改写走 with_phase_search（不就地修改）。"""
    enabled = CONFIG.with_phase_search(enabled=True)
    assert enabled.phase_search.enabled is True
    assert CONFIG.phase_search.enabled is False
    assert enabled.tolerances_s == CONFIG.tolerances_s


def test_eval_case_requires_bpm_and_exposes_cluster() -> None:
    """输入单元必须带 BPMList；cluster 默认退化为 key（bootstrap 的聚类单元）。"""
    case = perfect_case("song-a", song="song-a")
    assert case.cluster == "song-a"
    solo = perfect_case("song-b")
    assert solo.cluster == "song-b"
    with pytest.raises(ValueError, match="bpm_points"):
        EvalCase(key="bad", pred=(), gold=(), bpm_points=())
    assert BpmPoint(time_beats=0.0, bpm=120.0) in case.bpm_points
    replaced = case.with_pred(())
    assert replaced.pred == ()
    assert replaced.gold == case.gold


def test_macro_stratum_table_averages_per_chart() -> None:
    """分层表的 macro 版本：层的比率逐谱平均、计数取合计。"""
    first = perfect_case("a", count=4)
    second = perfect_case("b", count=8)
    evaluations = [evaluate_case(first, CONFIG), evaluate_case(second, CONFIG)]
    pooled = mean_strata(evaluations, CONFIG)
    assert "1/4" in pooled
    assert pooled["1/4"].n_gold == len(first.gold) + len(second.gold)
    assert pooled["1/4"].timing.f1 == 1.0
    micro = aggregate(evaluations, CONFIG).micro
    assert micro.n_gold == len(first.gold) + len(second.gold)


def test_rank_correlation_and_bootstrap_helpers() -> None:
    """统计辅助的解析用例：秩、Spearman、缺失约定。"""
    assert average_ranks([10.0, 20.0, 20.0, 30.0]) == [1.0, 2.5, 2.5, 4.0]
    assert spearman_rank_correlation([0.0, 1.0, 2.0], [5.0, 4.0, 3.0]) == pytest.approx(-1.0)
    assert spearman_rank_correlation([0.0, 1.0, 2.0], [3.0, 4.0, 5.0]) == pytest.approx(1.0)
    assert math.isnan(spearman_rank_correlation([1.0, 1.0], [1.0, 2.0]))
    assert math.isnan(pearson_correlation([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))
    assert math.isnan(pearson_correlation([1.0], [1.0]))
    with pytest.raises(ValueError, match="长度必须一致"):
        pearson_correlation([1.0, 2.0], [1.0])
    assert math.isnan(mean_or_nan([]))
    assert mean_or_nan([1.0, 3.0]) == 2.0
    low, high = percentile_interval([1.0, 2.0, 3.0, 4.0])
    assert low <= high
    assert all(math.isnan(value) for value in percentile_interval([]))


def test_event_helpers_round_trip_through_the_builders() -> None:
    """构造器与评估输入单元的口径一致（side / type / 双写字段不漂移）。"""
    case = make_case("pair", pred=(make_event(1.0),), gold=(make_event(1.0),))
    quality = evaluate_case(case, CONFIG).quality
    assert quality.n_pred == 1
    assert quality.n_gold == 1
    assert quality.timing_f1(CONFIG.primary_tolerance) == 1.0
