"""EvalReport：汇总、分节隔离、分解与文本渲染（plan 06 §3.2 / §4.3 / §4.6）。

报告分节（plan §3.2 的冻结字段 + 本实现的分节；**分节就是隔离的物理形式**）：

| 分节 | 内容 | 能否作主判据 |
|------|------|--------------|
| aggregate / per_chart | 事件级质量指标（双容差 × 两族 × macro/micro） | 是（主判据来源） |
| two_column | 「固定解码规则」与「每谱最优」两栏 | 是 |
| phase | 相位偏移：搜索 / 不搜索**两组数** | 是（时间对齐诊断） |
| breakdown | 时间组 × 难度 | 是 |
| calibration | 泊松 NLL + G3 常数基线 | **否：只作校准**（plan §4.6） |
| exploratory | 能量相关性 | **否：探索性**（plan §4.7） |
| legality | 违规率 / 越界比 / 跨线冲突 | 是（合法率是硬指标） |
| meta | seed / 解码臂 / 容差 / 相位搜索 / git rev / data rev | 可追溯性（**强制**） |

主判据入口 EvalReport.primary_score 只把 aggregate 的 quality 分节交给
primary_criterion（签名只接受 QualityMetrics），因此 NLL 与探索性读数**结构上**
进不了模型选择。

秒域（红线 7）：报告里的时间一律秒；meta 冻结容差与相位搜索口径。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

from pydantic import BaseModel, ConfigDict

from beatmorph.core.contracts.legality import POSITION_X_CLAMPED_KEY, LegalityReport
from beatmorph.core.contracts.phigros import NoteType, PhigrosChart
from beatmorph.decoder.postprocess.legality import check_chart
from beatmorph.eval.calibration import (
    CalibrationReadout,
    ExploratoryReadout,
    assert_calibration_is_labeled,
    assert_exploratory_is_labeled,
    primary_criterion,
)
from beatmorph.eval.matching import shift_events
from beatmorph.eval.metrics import (
    CaseEvaluation,
    FamilyMetrics,
    QualityMetrics,
    StratumMetrics,
    TwoColumnReport,
    aggregate,
    evaluate_case,
    merge_strata,
    two_column_report,
)
from beatmorph.eval.protocol import (
    MS_PER_S,
    UNKNOWN_DIFFICULTY_BAND,
    EvalCase,
    EvalConfig,
)

#: 「搜索后显著优于不搜索」的判据余量（严格改善即认定，浮点噪声另计）。
PHASE_IMPROVEMENT_MARGIN: Final[float] = 1e-6

#: plan §4.3 要求写进结论段的句子（时间对齐本身是主要误差源时）。
ALIGNMENT_CONCLUSION: Final[str] = (
    "结论：搜索后的 F1 显著优于不搜索 -> 时间对齐本身是主要误差源，而不是单纯的漏检/误检。"
)


class PhaseReport(BaseModel):
    """单谱的相位读数（plan §4.3 必须**单独报**）。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    searched: bool
    offset_s: float | None
    f1_before: float
    f1_after: float
    grid_size: int
    improved: bool


class ChartReport(BaseModel):
    """单谱报告（quality + 分层 + 相位；不含 NLL）。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str
    cluster: str
    difficulty: float | None
    difficulty_band: str
    quality: QualityMetrics
    time_groups: dict[str, StratumMetrics]
    phase: PhaseReport


class AggregateReport(BaseModel):
    """macro / micro 两栏（plan §3.2 的 aggregate 字段）。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    per_chart_mean: QualityMetrics
    micro: QualityMetrics


class BreakdownReport(BaseModel):
    """分解报告（plan §3.2 的 breakdown 字段）：时间组 × 难度。

    两个分解都是**层内 micro**（合并层内计数后重算）；层内的按谱平均可由
    per_chart 表自行聚合，故不重复落盘（plan §4.4 只要求分解本身）。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    by_time_group: dict[str, StratumMetrics]
    by_difficulty: dict[str, StratumMetrics]


class LegalityView(BaseModel):
    """合法性读数（plan §3.2 的 legality 字段；数据来自 Plan 05 的 LegalityReport）。

    position_x_clamped **恒为 0**（红线 3 / RFC-0029 §3.1：越界只统计不钳位）；
    它随报告落盘是为了让「没有被钳位」成为可审计的事实，而不是口头承诺。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    violation_rate: float
    out_of_range_ratio: float
    cross_line_conflicts: float
    duplicates: float
    position_x_clamped: float
    n_charts: int
    n_illegal: int


class PhaseAggregateReport(BaseModel):
    """相位偏移的两组数（plan §4.3：「搜索」与「不搜索」都必须报）。

    search_off 与报告的 aggregate **是同一份读数**（代码里直接复用同一对象，
    测试断言二者一致），这样「不搜索」这一组数不可能被漏报或走样。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    searched: bool
    search_off: AggregateReport
    search_on: AggregateReport | None
    offsets_s: dict[str, float]
    improved_charts: int
    alignment_is_main_error: bool
    note: str


class ReportMeta(BaseModel):
    """**强制**的可追溯字段（plan §3.2：任何报告必须能追到解码臂/seed/容差/数据版本）。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int | None
    decode_arm: str
    tolerance_s: float
    tolerances_s: tuple[float, ...]
    average: str
    decode_regime: str
    phase_search: bool
    phase_range_s: float | None
    phase_step_s: float | None
    x_bins: int
    position_x_tolerance: float
    time_group_rule: str
    git_rev: str
    data_rev: str
    n_cases: int
    n_unique_songs: int


class EvalReport(BaseModel):
    """评估报告（**冻结字段**，供 Plan 07 的日志与 Plan 08 的 CLI 消费）。

    字段与 plan §3.2 的模板一一对应：per_chart / aggregate / breakdown /
    calibration / legality / exploratory / meta，另加 two_column（两栏报告）与
    phase（相位两组数）。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    per_chart: list[ChartReport]
    aggregate: AggregateReport
    breakdown: BreakdownReport
    two_column: TwoColumnReport
    phase: PhaseAggregateReport
    calibration: CalibrationReadout
    exploratory: ExploratoryReadout
    legality: LegalityView
    meta: ReportMeta

    def primary_score(self) -> float:
        """模型选择主判据：per_chart（macro）栏的 timing-F1 @ 主容差。

        ⚠️ 只消费 aggregate 的 quality 分节——NLL 与能量相关性**在类型上**读不到
        （plan §4.6/§4.7 的硬约束靠签名兑现，而不是靠评审记得）。
        """
        return primary_criterion(self.aggregate.per_chart_mean, tolerance_s=self.meta.tolerance_s)

    def primary_score_micro(self) -> float:
        """micro 栏的主读数（同一判据的另一平均口径，供规模敏感的分析用）。"""
        return primary_criterion(self.aggregate.micro, tolerance_s=self.meta.tolerance_s)

    def assert_isolation(self) -> None:
        """自检：校准 / 探索性分节必须带标注且不得进主判据。"""
        assert_calibration_is_labeled(self.calibration)
        assert_exploratory_is_labeled(self.exploratory)


def _band_stratum(
    name: str,
    evaluations: Sequence[CaseEvaluation],
    config: EvalConfig,
) -> StratumMetrics:
    """难度档的层内 micro（计数合并后重算；难度分档见 breakdown.difficulty_band）。"""
    primary = config.primary_tolerance
    n_pred = sum(evaluation.counts.n_pred for evaluation in evaluations)
    n_gold = sum(evaluation.counts.n_gold for evaluation in evaluations)
    n_pairs = sum(evaluation.counts.pair_counts.get(primary, 0) for evaluation in evaluations)
    event_tp = sum(evaluation.counts.event_tp.get(primary, 0) for evaluation in evaluations)
    return StratumMetrics(
        name=name,
        n_pred=n_pred,
        n_gold=n_gold,
        n_pairs=n_pairs,
        timing=FamilyMetrics.build("timing", primary, n_pairs, n_pred, n_gold),
        event=FamilyMetrics.build("event", primary, event_tp, n_pred, n_gold),
    )


def difficulty_breakdown(
    evaluations: Sequence[CaseEvaluation],
    config: EvalConfig,
) -> dict[str, StratumMetrics]:
    """按难度档分解（plan §4.4-2；低难度单独成档，level 文本绝不参与）。"""
    grouped: dict[str, list[CaseEvaluation]] = {}
    for evaluation in evaluations:
        grouped.setdefault(evaluation.difficulty_band, []).append(evaluation)
    known = sorted(name for name in grouped if name != UNKNOWN_DIFFICULTY_BAND)
    order = [*known, UNKNOWN_DIFFICULTY_BAND] if UNKNOWN_DIFFICULTY_BAND in grouped else known
    return {name: _band_stratum(name, grouped[name], config) for name in order}


def legality_view(reports: Sequence[LegalityReport]) -> LegalityView:
    """Plan 05 的 LegalityReport -> 报告的合法性分节（**只读取，不改判**）。"""
    total = len(reports)
    illegal = sum(1 for report in reports if not report.is_legal)
    out_of_range = [report.stat("chart_out_of_range_fraction") for report in reports]
    return LegalityView(
        violation_rate=(illegal / total) if total else 0.0,
        out_of_range_ratio=(sum(out_of_range) / total) if total else 0.0,
        cross_line_conflicts=sum(report.stat("chart_cross_line_conflicts") for report in reports),
        duplicates=sum(report.stat("chart_duplicates") for report in reports),
        position_x_clamped=sum(report.stat(POSITION_X_CLAMPED_KEY) for report in reports),
        n_charts=total,
        n_illegal=illegal,
    )


def build_meta(
    config: EvalConfig,
    *,
    seed: int | None = None,
    decode_arm: str = "",
    git_rev: str = "",
    data_rev: str = "",
    n_cases: int = 0,
    n_unique_songs: int = 0,
) -> ReportMeta:
    """构造 meta（**强制字段**；git_rev / data_rev 由调用方注入，评估库不起子进程）。"""
    return ReportMeta(
        seed=seed,
        decode_arm=decode_arm,
        tolerance_s=config.primary_tolerance,
        tolerances_s=tuple(config.tolerances_s),
        average=str(config.average),
        decode_regime=str(config.decode_regime),
        phase_search=bool(config.phase_search.enabled),
        phase_range_s=float(config.phase_search.range_s) if config.phase_search.enabled else None,
        phase_step_s=float(config.phase_search.step_s) if config.phase_search.enabled else None,
        x_bins=int(config.x_bins),
        position_x_tolerance=config.position_tolerance,
        time_group_rule=str(config.time_group_rule),
        git_rev=git_rev,
        data_rev=data_rev,
        n_cases=int(n_cases),
        n_unique_songs=int(n_unique_songs),
    )


def _phase_aggregate(
    cases: Sequence[EvalCase],
    evaluations: Sequence[CaseEvaluation],
    config: EvalConfig,
    *,
    search_off: AggregateReport,
) -> PhaseAggregateReport:
    """相位分节：不搜索（复用 aggregate）+ 搜索后（整体平移 pred 后重评）。"""
    offsets: dict[str, float] = {}
    improved = 0
    for case, evaluation in zip(cases, evaluations, strict=True):
        if evaluation.phase.offset_s is not None:
            offsets[case.key] = float(evaluation.phase.offset_s)
        if evaluation.phase.improved(margin=PHASE_IMPROVEMENT_MARGIN):
            improved += 1
    if not config.phase_search.enabled:
        return PhaseAggregateReport(
            searched=False,
            search_off=search_off,
            search_on=None,
            offsets_s={},
            improved_charts=0,
            alignment_is_main_error=False,
            note="相位搜索未开启：本报告只给出「不搜索」一组数（plan §4.3 要求两组）。",
        )
    shifted = [
        case.with_pred(shift_events(case.pred, offsets.get(case.key, 0.0))) for case in cases
    ]
    quiet = config.with_phase_search(enabled=False)
    searched_metrics = aggregate([evaluate_case(case, quiet) for case in shifted], quiet)
    search_on = AggregateReport(
        per_chart_mean=searched_metrics.per_chart_mean,
        micro=searched_metrics.micro,
    )
    before = search_off.per_chart_mean.timing_f1(config.primary_tolerance)
    after = search_on.per_chart_mean.timing_f1(config.primary_tolerance)
    is_main = after > before + PHASE_IMPROVEMENT_MARGIN
    return PhaseAggregateReport(
        searched=True,
        search_off=search_off,
        search_on=search_on,
        offsets_s=offsets,
        improved_charts=improved,
        alignment_is_main_error=is_main,
        note=ALIGNMENT_CONCLUSION
        if is_main
        else "搜索后的 F1 未显著改善：时间对齐不是主要误差源。",
    )


def build_report(
    cases: Sequence[EvalCase],
    config: EvalConfig,
    *,
    calibration: CalibrationReadout | None = None,
    exploratory: ExploratoryReadout | None = None,
    legality: LegalityView | None = None,
    two_column: TwoColumnReport | None = None,
    seed: int | None = None,
    decode_arm: str = "",
    git_rev: str = "",
    data_rev: str = "",
) -> EvalReport:
    """由「谱面事件对」构造完整 EvalReport（plan §3.2）。

    Args:
        cases: 评估输入单元（同一首曲的 pred / gold 事件 + BPMList + 元数据）。
        config: 评估配置（容差、主容差、相位搜索、归组规则、bootstrap 种子）。
        calibration: 校准读数；None = 显式缺失（**不得填 0**）。
        exploratory: 探索性读数；None = 显式缺失。
        legality: 合法性分节；None = 未提供（n_charts = 0，与「零违规」可区分）。
        two_column: 两栏报告；None = 以 decode_arm（或 "fixed"）为唯一候选的退化两栏
            （此时最优 == 固定，best_not_worse 恒为 True）。
        seed / decode_arm / git_rev / data_rev: meta 的强制可追溯字段。
    """
    label = decode_arm or "fixed"
    evaluations = [evaluate_case(case, config) for case in cases]
    report_aggregate = aggregate(evaluations, config)
    if two_column is None:
        candidates = {case.key: {label: case.pred} for case in cases}
        two_column = two_column_report(cases, candidates, config, fixed_label=label)
    legality_value = (
        legality
        if legality is not None
        else LegalityView(
            violation_rate=0.0,
            out_of_range_ratio=0.0,
            cross_line_conflicts=0.0,
            duplicates=0.0,
            position_x_clamped=0.0,
            n_charts=0,
            n_illegal=0,
        )
    )
    per_chart = [
        ChartReport(
            key=evaluation.key,
            cluster=evaluation.cluster,
            difficulty=evaluation.difficulty,
            difficulty_band=evaluation.difficulty_band,
            quality=evaluation.quality,
            time_groups=dict(evaluation.time_groups),
            phase=PhaseReport(
                searched=evaluation.phase.searched,
                offset_s=evaluation.phase.offset_s,
                f1_before=evaluation.phase.f1_before,
                f1_after=evaluation.phase.f1_after,
                grid_size=evaluation.phase.grid_size,
                improved=evaluation.phase.improved(margin=PHASE_IMPROVEMENT_MARGIN),
            ),
        )
        for evaluation in evaluations
    ]
    songs = {case.cluster for case in cases}
    return EvalReport(
        per_chart=per_chart,
        aggregate=AggregateReport(
            per_chart_mean=report_aggregate.per_chart_mean,
            micro=report_aggregate.micro,
        ),
        breakdown=BreakdownReport(
            by_time_group=merge_strata(evaluations, config),
            by_difficulty=difficulty_breakdown(evaluations, config),
        ),
        two_column=two_column,
        phase=_phase_aggregate(
            cases,
            evaluations,
            config,
            search_off=AggregateReport(
                per_chart_mean=report_aggregate.per_chart_mean,
                micro=report_aggregate.micro,
            ),
        ),
        calibration=CalibrationReadout.unavailable() if calibration is None else calibration,
        exploratory=ExploratoryReadout.unavailable() if exploratory is None else exploratory,
        legality=legality_value,
        meta=build_meta(
            config,
            seed=seed,
            decode_arm=decode_arm,
            git_rev=git_rev,
            data_rev=data_rev,
            n_cases=len(cases),
            n_unique_songs=len(songs),
        ),
    )


def evaluate_charts(
    pred_charts: Mapping[str, PhigrosChart],
    gold_charts: Mapping[str, PhigrosChart],
    config: EvalConfig,
    *,
    calibration: CalibrationReadout | None = None,
    exploratory: ExploratoryReadout | None = None,
    seed: int | None = None,
    decode_arm: str = "",
    git_rev: str = "",
    data_rev: str = "",
) -> EvalReport:
    """plan §3.1 的 EvalInput 全链路：人类谱 + 生成谱 -> EvalReport（含合法性）。

    pred 必须是**已通过合法性校验**的谱（plan §3.1：否则评估的是「非法谱的质量」）；
    本函数只读取 pred 的 LegalityReport（check_chart 不做任何改动），把违规率单独报出，
    不做拦截——拦截是 Plan 05 导出路径的职责（红线 6）。
    """
    keys = list(gold_charts)
    missing = [key for key in keys if key not in pred_charts]
    if missing:
        raise KeyError(f"pred_charts 缺少谱面：{missing}")
    cases = [EvalCase.from_charts(key, pred_charts[key], gold_charts[key]) for key in keys]
    legality = legality_view([check_chart(pred_charts[key]) for key in keys])
    return build_report(
        cases,
        config,
        calibration=calibration,
        exploratory=exploratory,
        legality=legality,
        seed=seed,
        decode_arm=decode_arm,
        git_rev=git_rev,
        data_rev=data_rev,
    )


# ── 报告 schema 冻结（plan §3.2 的字段清单；缺格即失败）────────────────────
REQUIRED_REPORT_FIELDS: Final[tuple[str, ...]] = (
    "per_chart",
    "aggregate",
    "breakdown",
    "two_column",
    "phase",
    "calibration",
    "legality",
    "exploratory",
    "meta",
)
REQUIRED_AGGREGATE_FIELDS: Final[tuple[str, ...]] = ("per_chart_mean", "micro")
REQUIRED_BREAKDOWN_FIELDS: Final[tuple[str, ...]] = ("by_time_group", "by_difficulty")
REQUIRED_CALIBRATION_FIELDS: Final[tuple[str, ...]] = (
    "nll",
    "nll_constant_baseline",
    "nll_zero_is_inf",
    "role",
    "warning",
    "is_primary_criterion",
)
REQUIRED_EXPLORATORY_FIELDS: Final[tuple[str, ...]] = (
    "energy_correlation",
    "role",
    "warning",
    "is_primary_criterion",
)
REQUIRED_LEGALITY_FIELDS: Final[tuple[str, ...]] = (
    "violation_rate",
    "out_of_range_ratio",
    "cross_line_conflicts",
)
REQUIRED_META_FIELDS: Final[tuple[str, ...]] = (
    "seed",
    "decode_arm",
    "tolerance_s",
    "phase_search",
    "git_rev",
    "data_rev",
)


def _require(payload: Mapping[str, object], section: str, names: Sequence[str]) -> None:
    for name in names:
        if name not in payload:
            raise AssertionError(f"报告分节 {section!r} 缺少冻结字段 {name!r}")


def assert_report_schema(payload: Mapping[str, object]) -> None:
    """报告 JSON schema 的冻结检查（字段缺失即抛 AssertionError）。

    供 Plan 07 的日志落盘与 Plan 08 的 CLI 在**消费之前**校验：报告是一份契约，
    少一格就等于数字没有出处。
    """
    _require(payload, "report", REQUIRED_REPORT_FIELDS)
    for section, names in (
        ("aggregate", REQUIRED_AGGREGATE_FIELDS),
        ("breakdown", REQUIRED_BREAKDOWN_FIELDS),
        ("calibration", REQUIRED_CALIBRATION_FIELDS),
        ("exploratory", REQUIRED_EXPLORATORY_FIELDS),
        ("legality", REQUIRED_LEGALITY_FIELDS),
        ("meta", REQUIRED_META_FIELDS),
    ):
        nested = payload[section]
        if not isinstance(nested, Mapping):
            raise AssertionError(f"报告分节 {section!r} 必须是对象，得到 {type(nested).__name__}")
        _require(nested, section, names)


def _tolerance_label(tolerance_s: float) -> str:
    """容差 -> 显示标签（与 EvalConfig.tolerance_key 同口径）。"""
    return f"{tolerance_s * MS_PER_S:g}ms"


def _format_quality(metrics: QualityMetrics, tolerances: Sequence[float]) -> list[str]:
    """质量分节的多行渲染（双容差 × 两族）。"""
    lines = [
        f"  n_pred={metrics.n_pred} n_gold={metrics.n_gold} n_pairs={metrics.n_pairs}",
    ]
    for tolerance in tolerances:
        label = _tolerance_label(tolerance)
        timing = metrics.timing[tolerance]
        event = metrics.event[tolerance]
        lines.append(
            f"  @±{label}: timing P/R/F1 = {timing.precision:.4f}/{timing.recall:.4f}/{timing.f1:.4f}"
            f" | event P/R/F1 = {event.precision:.4f}/{event.recall:.4f}/{event.f1:.4f}",
        )
    return lines


def _phase_offset_text(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.4f}s"


def _position_x_text(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def render_text(report: EvalReport) -> str:
    """把报告渲染成可直接进日志/终端的多行文本。

    硬要求（测试断言）：头部固定含 NLL 警示行（plan §4.6）；背面 recall、相位偏移、
    positionX MAE（含量化下界）、侧别准确率（含「全预测正面」基线）各自独立成行；
    探索性分节显式标注；且当相位搜索显示对齐是主要误差源时，结论段必须写出该结论。
    """
    meta = report.meta
    macro = report.aggregate.per_chart_mean
    micro = report.aggregate.micro
    tolerances = list(meta.tolerances_s)
    primary = meta.tolerance_s
    lines: list[str] = [
        f"BeatMorph 评估报告：{len(report.per_chart)} 张谱（唯一曲目 {meta.n_unique_songs}）",
        f"[警示] {report.calibration.warning}",
        f"[探索性] {report.exploratory.warning}",
        f"meta：seed={meta.seed} 解码臂={meta.decode_arm or '(未声明)'} "
        f"主容差=±{_tolerance_label(primary)} 相位搜索={meta.phase_search} "
        f"git={meta.git_rev or '(未声明)'} data={meta.data_rev or '(未声明)'}",
        f"主判据（macro timing-F1@±{_tolerance_label(primary)}）= {report.primary_score():.4f}",
        "",
        "【事件级质量：macro（按谱平均）】",
        *_format_quality(macro, tolerances),
        "【事件级质量：micro（按事件合并）】",
        *_format_quality(micro, tolerances),
        "",
        f"背面 recall = {macro.back_recall:.4f}（背面 precision = {macro.back_precision:.4f}，"
        f"gold 背面事件 n={macro.n_back_gold}，pred 背面事件 n={macro.n_back_pred}）",
        f"侧别准确率 = {macro.side_accuracy:.4f}（**全预测正面**基线 = "
        f"{macro.side_all_front_baseline:.4f}；两者接近即说明总准确率无判别力）",
        f"类型准确率 = {macro.type_accuracy:.4f}；每类 recall = "
        + "，".join(
            f"{int(NoteType(key))}:{value:.4f}" for key, value in sorted(macro.type_recall.items())
        ),
        f"positionX MAE = {_position_x_text(macro.position_x_mae)}"
        f"（量化下界 dx/4 = {macro.position_x_quantization_lower_bound:.3f}；"
        f"x_bins={meta.x_bins}，MAE 必须与该下界及 x_bins 一起读）",
        f"全谱相位偏移（匹配对有符号偏移，独立读数）：中位 = "
        f"{_phase_offset_text(macro.phase_offset_median_s)}，均值 = "
        f"{_phase_offset_text(macro.phase_offset_mean_s)}",
        "",
    ]
    phase = report.phase
    lines.append(f"【相位搜索：{'开启' if phase.searched else '关闭'}】")
    lines.append(
        f"  不搜索 timing-F1@±{_tolerance_label(primary)} = "
        f"{phase.search_off.per_chart_mean.timing_f1(primary):.4f}",
    )
    if phase.search_on is not None:
        lines.append(
            f"  搜索后 timing-F1@±{_tolerance_label(primary)} = "
            f"{phase.search_on.per_chart_mean.timing_f1(primary):.4f}"
            f"（改善的谱 {phase.improved_charts}/{len(report.per_chart)}）",
        )
        offsets = "，".join(
            f"{key}={value:+.4f}s" for key, value in sorted(phase.offsets_s.items())
        )
        lines.append(f"  每谱最佳偏移：{offsets or '(空)'}")
    lines.append(f"  {phase.note}")
    lines.append("")
    two = report.two_column
    lines.append(
        f"【两栏报告】固定规则 = {two.fixed_label}（macro timing-F1 = "
        f"{two.fixed.per_chart_mean.timing_f1(primary):.4f}）；每谱最优 = "
        f"{two.per_chart_best.per_chart_mean.timing_f1(primary):.4f}"
        f"（best>=fixed 成立：{two.best_not_worse}；选择 = "
        + "，".join(f"{key}:{value}" for key, value in sorted(two.per_chart_selection.items()))
        + "）",
    )
    lines.append("")
    lines.append("【分解：时间组】")
    for name, row in sorted(report.breakdown.by_time_group.items()):
        lines.append(
            f"  {name}: timing-F1 = {row.timing.f1:.4f}（n_gold={row.n_gold}，n_pred={row.n_pred}）"
        )
    lines.append("【分解：难度】")
    for name, row in report.breakdown.by_difficulty.items():
        lines.append(
            f"  {name}: timing-F1 = {row.timing.f1:.4f}（n_gold={row.n_gold}，n_pred={row.n_pred}）"
        )
    lines.append("")
    calibration = report.calibration
    lines.append(
        "【校准（不参与模型选择）】"
        f" NLL = {calibration.nll}；G3 常数基线 = {calibration.nll_constant_baseline}；"
        f"NLL/事件 = {calibration.nll_per_event}；λ≡0 时为 +∞ = {calibration.nll_zero_is_inf}",
    )
    exploratory = report.exploratory
    lines.append(
        f"【探索性（{'已计算' if exploratory.available else '未计算'}）】"
        f" 能量相关性 = {exploratory.energy_correlation}（bin={exploratory.energy_bin_s}s）；"
        f"{exploratory.onset_disclaimer}",
    )
    legality = report.legality
    lines.append(
        f"【合法性】违规率 = {legality.violation_rate:.4f}（{legality.n_illegal}/{legality.n_charts}）；"
        f"越界比 = {legality.out_of_range_ratio:.4f}；跨线冲突 = {legality.cross_line_conflicts:.0f}；"
        f"positionX 钳位 = {legality.position_x_clamped:.0f}（红线 3 恒为 0）",
    )
    if phase.alignment_is_main_error:
        lines.append(ALIGNMENT_CONCLUSION)
    return "\n".join(lines)


__all__ = [
    "ALIGNMENT_CONCLUSION",
    "PHASE_IMPROVEMENT_MARGIN",
    "REQUIRED_AGGREGATE_FIELDS",
    "REQUIRED_BREAKDOWN_FIELDS",
    "REQUIRED_CALIBRATION_FIELDS",
    "REQUIRED_EXPLORATORY_FIELDS",
    "REQUIRED_LEGALITY_FIELDS",
    "REQUIRED_META_FIELDS",
    "REQUIRED_REPORT_FIELDS",
    "AggregateReport",
    "BreakdownReport",
    "ChartReport",
    "EvalReport",
    "LegalityView",
    "PhaseAggregateReport",
    "PhaseReport",
    "ReportMeta",
    "assert_report_schema",
    "build_meta",
    "build_report",
    "difficulty_breakdown",
    "evaluate_charts",
    "legality_view",
    "render_text",
]
