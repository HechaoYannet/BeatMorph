"""评估（Plan 06）：事件级指标 / 匹配协议 / 分解 / 校准隔离 / corruption 准入。

| 文件 | 职责 |
|------|------|
| protocol.py | EvalConfig / EvalCase（输入单元）与口径常量（双容差、难度 round、相位网格） |
| stats.py | 秩相关 / bootstrap（自实现，不引入 scipy） |
| matching.py | 贪心一对一匹配 + 全谱相位偏移搜索（未匹配 pred 留在分母） |
| breakdown.py | 时间组（含 12/24 三连）与难度分档（level 文本绝不解析） |
| metrics.py | 事件级指标、macro/micro、分层与两栏报告 |
| calibration.py | NLL 校准与探索性指标的**物理隔离**与标注 |
| corruption.py | dose-controlled 注入 + 不变性控制 + 准入（未通过者不得进主报告） |
| report.py | EvalReport 汇总 / schema 冻结 / 文本渲染 |

三条硬约束（本包的设计红线）：

1. **秒域**（RFC-0029 §7-5 / 红线 7）：评估一律用秒；本包**不实现**任何秒 <-> τ 换算，
   需要拍坐标时经 beatmorph.field.grid 的权威接口；公共接口无帧索引、无 τ 格索引。
2. **分母语义**（plan §4.2-2）：precision 的分母恒为**全部**生成事件，删事件不可能
   提高 precision。
3. **主判据限于事件级指标 + 人评**：NLL 只作校准（plan §4.6），能量相关性只作探索
   （plan §4.7）；隔离由分节类型与 primary_criterion 的签名共同保证。

本包模块级只依赖 numpy / pydantic / 契约与 decoder（**不 import torch**；
需要 torch 的 NLL 契约断言在函数内惰性引入），因此默认 CI（无权重、无 GPU）即可全跑。
"""

from beatmorph.eval.breakdown import (
    TIME_GROUP_SUBDIVISIONS,
    TIME_GROUPS,
    TimeGroup,
    assign_time_groups,
    classify_beat_position,
    classify_interval,
    difficulty_band,
    event_beats,
)
from beatmorph.eval.calibration import (
    EXPLORATORY_WARNING,
    NLL_WARNING,
    ONSET_DISCLAIMER,
    PRIMARY_CRITERION_SECTIONS,
    CalibrationReadout,
    ExploratoryReadout,
    assert_calibration_is_labeled,
    assert_exploratory_is_labeled,
    binned_density,
    check_zero_intensity_diverges,
    energy_correlation,
    poisson_nll_float,
    primary_criterion,
)
from beatmorph.eval.corruption import (
    DEFAULT_DOSES,
    STANDARD_INJECTIONS,
    X_AXIS_INJECTIONS,
    Injection,
    InjectionResponse,
    InvarianceResult,
    MetricAdmission,
    MetricSpec,
    admit_metric,
    admit_metrics,
    assert_admissible,
    corruption_sweep,
    default_injections,
    invariance_controls,
    main_report_metric_names,
    primary_metric_specs,
)
from beatmorph.eval.matching import (
    MatchPair,
    MatchResult,
    PhaseOutcome,
    greedy_match,
    marker_equal,
    precision_recall_f1,
    search_phase_offset,
    shift_events,
)
from beatmorph.eval.metrics import (
    AggregateMetrics,
    CaseEvaluation,
    FamilyMetrics,
    MetricCounts,
    QualityMetrics,
    StratumMetrics,
    TwoColumnReport,
    aggregate,
    evaluate_case,
    mean_quality,
    mean_strata,
    merge_strata,
    metrics_from_counts,
    select_per_chart_best,
    two_column_report,
)
from beatmorph.eval.protocol import (
    DDC_TOLERANCE_S,
    DEFAULT_TOLERANCES_S,
    GENELIVE_TOLERANCE_S,
    UNKNOWN_DIFFICULTY_BAND,
    AverageMode,
    DecodeRegime,
    EvalCase,
    EvalConfig,
    PhaseSearchConfig,
    TimeGroupRule,
)
from beatmorph.eval.report import (
    ALIGNMENT_CONCLUSION,
    REQUIRED_REPORT_FIELDS,
    AggregateReport,
    BreakdownReport,
    ChartReport,
    EvalReport,
    LegalityView,
    PhaseAggregateReport,
    PhaseReport,
    ReportMeta,
    assert_report_schema,
    build_meta,
    build_report,
    difficulty_breakdown,
    evaluate_charts,
    legality_view,
    render_text,
)
from beatmorph.eval.stats import (
    bootstrap_interval,
    pearson_correlation,
    percentile_interval,
    spearman_rank_correlation,
)

__all__ = [
    "ALIGNMENT_CONCLUSION",
    "DDC_TOLERANCE_S",
    "DEFAULT_DOSES",
    "DEFAULT_TOLERANCES_S",
    "EXPLORATORY_WARNING",
    "GENELIVE_TOLERANCE_S",
    "NLL_WARNING",
    "ONSET_DISCLAIMER",
    "PRIMARY_CRITERION_SECTIONS",
    "REQUIRED_REPORT_FIELDS",
    "STANDARD_INJECTIONS",
    "TIME_GROUPS",
    "TIME_GROUP_SUBDIVISIONS",
    "UNKNOWN_DIFFICULTY_BAND",
    "X_AXIS_INJECTIONS",
    "AggregateMetrics",
    "AggregateReport",
    "AverageMode",
    "BreakdownReport",
    "CalibrationReadout",
    "CaseEvaluation",
    "ChartReport",
    "DecodeRegime",
    "EvalCase",
    "EvalConfig",
    "EvalReport",
    "ExploratoryReadout",
    "FamilyMetrics",
    "Injection",
    "InjectionResponse",
    "InvarianceResult",
    "LegalityView",
    "MatchPair",
    "MatchResult",
    "MetricAdmission",
    "MetricCounts",
    "MetricSpec",
    "PhaseAggregateReport",
    "PhaseOutcome",
    "PhaseReport",
    "PhaseSearchConfig",
    "QualityMetrics",
    "ReportMeta",
    "StratumMetrics",
    "TimeGroup",
    "TimeGroupRule",
    "TwoColumnReport",
    "admit_metric",
    "admit_metrics",
    "aggregate",
    "assert_admissible",
    "assert_calibration_is_labeled",
    "assert_exploratory_is_labeled",
    "assert_report_schema",
    "assign_time_groups",
    "binned_density",
    "bootstrap_interval",
    "build_meta",
    "build_report",
    "check_zero_intensity_diverges",
    "classify_beat_position",
    "classify_interval",
    "corruption_sweep",
    "default_injections",
    "difficulty_band",
    "difficulty_breakdown",
    "energy_correlation",
    "evaluate_case",
    "evaluate_charts",
    "event_beats",
    "greedy_match",
    "invariance_controls",
    "legality_view",
    "main_report_metric_names",
    "marker_equal",
    "mean_quality",
    "mean_strata",
    "merge_strata",
    "metrics_from_counts",
    "pearson_correlation",
    "percentile_interval",
    "poisson_nll_float",
    "precision_recall_f1",
    "primary_criterion",
    "primary_metric_specs",
    "render_text",
    "search_phase_offset",
    "select_per_chart_best",
    "shift_events",
    "spearman_rank_correlation",
    "two_column_report",
]
