"""事件级指标：双容差 F1 / 相位偏移 / 背面 recall / 类型与侧别准确率 / positionX MAE
（plan 06 §4.1、§4.5；RFC-0029 §5.1）。

**指标族（必须分开报，plan §4.2-3）**：

- timing 族：只比时间，TP = 一对一匹配上的对数（不要求线号，plan §4.2-4）；
- event 族：TP 还要求 line_id / positionX / side / type 一致（plan §4.1）。

**两族的 FP / FN 都按「未匹配的生成事件留在分母」计**（plan §4.2-2）：precision 的
分母恒为**全部** pred 事件，因此删事件不可能提高 precision。

**平均口径（plan §4.2-4 / 文献库 §8.4-3）**：per_chart（macro，逐谱 F1 的算术平均，
GenéLive! 的 F1-c）与 micro（合并 tp/fp/fn 后重算）**都必须报**。⚠️ 两者不是彼此的
调和平均：macro 行的 tp/fp/fn 是**合计**（便于核对 micro），而 precision/recall/F1 是
逐谱值的平均——这不是 bug，是 F1-c 的定义（M6.3 有用例断言两者必须不同）。

**秒域（红线 7）**：全部时间量以秒为单位；本模块不实现任何秒 <-> τ 换算，
也不出现帧索引或 τ 格索引（时间组分解的拍换算经 field/ 的权威接口）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial

import numpy as np
from pydantic import BaseModel, ConfigDict

from beatmorph.core.contracts.phigros import Side
from beatmorph.decoder.events import DecodedEvent
from beatmorph.eval.breakdown import (
    TIME_GROUPS,
    TimeGroup,
    assign_time_groups,
    difficulty_band,
)
from beatmorph.eval.matching import (
    MarkerFilter,
    MatchResult,
    PhaseOutcome,
    greedy_match,
    marker_equal,
    precision_recall_f1,
    search_phase_offset,
)
from beatmorph.eval.protocol import EvalCase, EvalConfig


class FamilyMetrics(BaseModel):
    """一族的指标（族 + 容差定位；计数与比率**同时**给出）。

    Attributes:
        family: "timing" 或 "event"。
        tolerance_s: 本行使用的容差（秒）。
        tp / fp / fn: tp = 族判据下的 TP；fp = n_pred - tp；fn = n_gold - tp。
        precision / recall / f1: 由上面三个计数导出；分母为 0 时为 0.0
            （**必须连同计数一起读**：0.0 与「无样本」只能靠计数区分）。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    family: str
    tolerance_s: float
    n_pred: int
    n_gold: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float

    @classmethod
    def build(
        cls, family: str, tolerance_s: float, tp: int, n_pred: int, n_gold: int
    ) -> FamilyMetrics:
        """由 (tp, n_pred, n_gold) 构造（precision 分母含未匹配 pred）。"""
        precision, recall, f1 = precision_recall_f1(tp, n_pred, n_gold)
        return cls(
            family=family,
            tolerance_s=float(tolerance_s),
            n_pred=int(n_pred),
            n_gold=int(n_gold),
            tp=int(tp),
            fp=int(n_pred - tp),
            fn=int(n_gold - tp),
            precision=precision,
            recall=recall,
            f1=f1,
        )


class StratumMetrics(BaseModel):
    """一个分层（时间组 / 难度档）在**主容差**下的指标（plan §4.4）。

    分层只报主容差：分解是为了定位误差来源，双容差已在总表给出（plan §4.1 的双容差
    要求针对总表；分解口径如要改由 plan 决定）。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    n_pred: int
    n_gold: int
    n_pairs: int
    timing: FamilyMetrics
    event: FamilyMetrics


class QualityMetrics(BaseModel):
    """事件级质量指标（**主判据所在的分节**；NLL / 探索性指标不在此）。

    ⚠️ 冻结字段（plan §3.2 的 EvalReport 由 Plan 07/08 消费）：字段一旦发布，
    增删须走 RFC/plan 更新。

    Attributes:
        n_pred / n_gold: 两侧事件总数。
        n_pairs: **主容差**下的匹配对数（timing TP）。
        timing / event: 容差（秒）-> 族指标；键即 EvalConfig.tolerances_s 的元素。
        phase_offset_median_s / phase_offset_mean_s: 匹配对的**有符号**偏移
            （pred - gold）的中位 / 均值；无匹配对时为 None（**缺失不等于 0**）。
        back_recall / back_precision: 背面（side == BACK）的事件级读数（plan §4.1：
            背面只占 2.4-3.0%，必须**单独报**）。
        n_back_gold: gold 侧背面事件数（可判定性的前提）。
        side_accuracy: 匹配对上的侧别准确率；side_all_front_baseline: **全预测正面**
            基线在同一批匹配对上的准确率（约 0.97 → 总准确率无判别力，必须并列报）。
        type_accuracy: 匹配对上的类型准确率；type_recall: 每类 recall（键为
            int(NoteType)），分母是**全部**该类 gold 事件（含未匹配）。
        position_x_mae: **只在匹配对上**计算（plan §4.1 / §4.5）；无对时为 None。
        position_x_quantization_lower_bound: 量化下界 dx / 4（plan §4.5 的推导），
            随 x_bins 变化，必须与 MAE 并列报。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_pred: int
    n_gold: int
    n_pairs: int
    timing: dict[float, FamilyMetrics]
    event: dict[float, FamilyMetrics]
    phase_offset_median_s: float | None
    phase_offset_mean_s: float | None
    back_recall: float
    back_precision: float
    n_back_gold: int
    n_back_pred: int
    side_accuracy: float
    side_all_front_baseline: float
    type_accuracy: float
    type_recall: dict[int, float]
    position_x_mae: float | None
    position_x_quantization_lower_bound: float

    def timing_f1(self, tolerance_s: float) -> float:
        """timing 族在给定容差下的 F1（容差必须已在配置里声明）。"""
        return float(self.timing[float(tolerance_s)].f1)

    def event_f1(self, tolerance_s: float) -> float:
        """event 族在给定容差下的 F1。"""
        return float(self.event[float(tolerance_s)].f1)


@dataclass(frozen=True, slots=True)
class MetricCounts:
    """指标的**可合并计数**（micro 汇总与 bootstrap 都靠它，不靠比率）。

    只有计数与逐对读数进这里：比率可由计数重算，因此 micro 平均与 per_chart 平均
    的差异**只**来自「先平均还是先合并」这一个自由度。
    """

    n_pred: int
    n_gold: int
    pair_counts: dict[float, int]
    event_tp: dict[float, int]
    phase_offsets_s: tuple[float, ...]
    position_abs_errors: tuple[float, ...]
    back_pred: int
    back_gold: int
    back_tp: int
    side_correct: int
    side_gold_front: int
    type_correct: int
    type_gold: dict[int, int]
    type_tp: dict[int, int]

    def merge(self, other: MetricCounts) -> MetricCounts:
        """合并两份计数（micro 汇总的唯一入口）。"""

        def merged(left: dict[float, int], right: dict[float, int]) -> dict[float, int]:
            keys = set(left) | set(right)
            return {key: int(left.get(key, 0)) + int(right.get(key, 0)) for key in sorted(keys)}

        def merged_int(left: dict[int, int], right: dict[int, int]) -> dict[int, int]:
            keys = set(left) | set(right)
            return {key: int(left.get(key, 0)) + int(right.get(key, 0)) for key in sorted(keys)}

        return MetricCounts(
            n_pred=self.n_pred + other.n_pred,
            n_gold=self.n_gold + other.n_gold,
            pair_counts=merged(self.pair_counts, other.pair_counts),
            event_tp=merged(self.event_tp, other.event_tp),
            phase_offsets_s=self.phase_offsets_s + other.phase_offsets_s,
            position_abs_errors=self.position_abs_errors + other.position_abs_errors,
            back_pred=self.back_pred + other.back_pred,
            back_gold=self.back_gold + other.back_gold,
            back_tp=self.back_tp + other.back_tp,
            side_correct=self.side_correct + other.side_correct,
            side_gold_front=self.side_gold_front + other.side_gold_front,
            type_correct=self.type_correct + other.type_correct,
            type_gold=merged_int(self.type_gold, other.type_gold),
            type_tp=merged_int(self.type_tp, other.type_tp),
        )


def _mean(values: Sequence[float]) -> float:
    """算术平均（空序列 0.0：比率字段不得为 NaN，见 precision_recall_f1 的说明）。"""
    return float(np.mean(values)) if values else 0.0


def _mean_optional(values: Sequence[float | None]) -> float | None:
    """仅对已定义值取平均（无已定义值 -> None：**缺失不等于 0**）。"""
    clean = [float(value) for value in values if value is not None]
    return float(np.mean(clean)) if clean else None


def metrics_from_counts(counts: MetricCounts, config: EvalConfig) -> QualityMetrics:
    """计数 -> 指标（**micro / 池化口径**：先合并计数再算比率）。"""
    timing = {
        tol: FamilyMetrics.build(
            "timing", tol, counts.pair_counts.get(tol, 0), counts.n_pred, counts.n_gold
        )
        for tol in config.tolerances_s
    }
    event = {
        tol: FamilyMetrics.build(
            "event", tol, counts.event_tp.get(tol, 0), counts.n_pred, counts.n_gold
        )
        for tol in config.tolerances_s
    }
    n_pairs = counts.pair_counts.get(config.primary_tolerance, 0)
    offsets = counts.phase_offsets_s
    return QualityMetrics(
        n_pred=counts.n_pred,
        n_gold=counts.n_gold,
        n_pairs=n_pairs,
        timing=timing,
        event=event,
        phase_offset_median_s=float(np.median(offsets)) if offsets else None,
        phase_offset_mean_s=float(np.mean(offsets)) if offsets else None,
        back_recall=(counts.back_tp / counts.back_gold) if counts.back_gold else 0.0,
        back_precision=(counts.back_tp / counts.back_pred) if counts.back_pred else 0.0,
        n_back_gold=counts.back_gold,
        n_back_pred=counts.back_pred,
        side_accuracy=(counts.side_correct / n_pairs) if n_pairs else 0.0,
        side_all_front_baseline=(counts.side_gold_front / n_pairs) if n_pairs else 0.0,
        type_accuracy=(counts.type_correct / n_pairs) if n_pairs else 0.0,
        type_recall={
            note_type: (counts.type_tp.get(note_type, 0) / total) if total else 0.0
            for note_type, total in sorted(counts.type_gold.items())
        },
        position_x_mae=_mean_optional([float(value) for value in counts.position_abs_errors]),
        position_x_quantization_lower_bound=config.quantization_lower_bound,
    )


def _macro_family(
    family: str,
    tolerance_s: float,
    qualities: Sequence[QualityMetrics],
    counts: MetricCounts,
) -> FamilyMetrics:
    """一族一容差的 macro 行：**计数是合计，比率是逐谱平均**（F1-c 口径）。"""
    rows = [
        (quality.timing if family == "timing" else quality.event)[tolerance_s]
        for quality in qualities
    ]
    tp = sum(row.tp for row in rows)
    return FamilyMetrics(
        family=family,
        tolerance_s=float(tolerance_s),
        n_pred=counts.n_pred,
        n_gold=counts.n_gold,
        tp=tp,
        fp=counts.n_pred - tp,
        fn=counts.n_gold - tp,
        precision=_mean([row.precision for row in rows]),
        recall=_mean([row.recall for row in rows]),
        f1=_mean([row.f1 for row in rows]),
    )


def mean_quality(
    qualities: Sequence[QualityMetrics],
    counts: MetricCounts,
    config: EvalConfig,
) -> QualityMetrics:
    """per_chart（macro）汇总：比率逐谱平均，计数取合计。

    ⚠️ macro 行的 precision/recall/F1 **不是**由本行的 tp/fp/fn 导出的（那是 micro
    的定义）。两者在等长谱上会重合、在不等长谱上必须不同（M6.3 的判据）。

    「条件平均」的约定（逐条声明，避免 0/0 与分母偏置）：

    - back_recall / back_precision：只在**有背面 gold / 有背面 pred**的谱上平均
      （不含背面的谱不该拉低背面 recall）；
    - side / type 准确率：只在**有匹配对**的谱上平均；
    - 每类 type_recall：只在**该类 gold 非空**的谱上平均；
    - 相位偏移与 MAE：只在已定义（有匹配对）的谱上平均。
    """
    timing = {tol: _macro_family("timing", tol, qualities, counts) for tol in config.tolerances_s}
    event = {tol: _macro_family("event", tol, qualities, counts) for tol in config.tolerances_s}
    note_types = sorted({key for quality in qualities for key in quality.type_recall})
    return QualityMetrics(
        n_pred=counts.n_pred,
        n_gold=counts.n_gold,
        n_pairs=sum(quality.n_pairs for quality in qualities),
        timing=timing,
        event=event,
        phase_offset_median_s=_mean_optional([q.phase_offset_median_s for q in qualities]),
        phase_offset_mean_s=_mean_optional([q.phase_offset_mean_s for q in qualities]),
        back_recall=_mean([q.back_recall for q in qualities if q.n_back_gold > 0]),
        back_precision=_mean([q.back_precision for q in qualities if q.n_back_pred > 0]),
        n_back_gold=counts.back_gold,
        n_back_pred=counts.back_pred,
        side_accuracy=_mean([q.side_accuracy for q in qualities if q.n_pairs > 0]),
        side_all_front_baseline=_mean(
            [q.side_all_front_baseline for q in qualities if q.n_pairs > 0]
        ),
        type_accuracy=_mean([q.type_accuracy for q in qualities if q.n_pairs > 0]),
        type_recall={
            note_type: _mean(
                [q.type_recall[note_type] for q in qualities if note_type in q.type_recall]
            )
            for note_type in note_types
        },
        position_x_mae=_mean_optional([q.position_x_mae for q in qualities]),
        position_x_quantization_lower_bound=config.quantization_lower_bound,
    )


class AggregateMetrics(BaseModel):
    """macro 与 micro **两栏都必须有**（plan §4.2-4 / RFC-0029 §5.1）。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    per_chart_mean: QualityMetrics
    micro: QualityMetrics


class TwoColumnReport(BaseModel):
    """「固定解码规则」与「每谱最优」两栏（plan §4.2-4、文献库 §8.4-4）。

    两栏用**同一套指标与同一套格式**，否则「能调阈值的一方占便宜」（ITGPT 的不对称
    讨论，文献库 §2.2-1）。`per_chart_selection` 记录每张谱选了哪个候选（可复现性
    留痕：同输入 + 同 config -> 同选择）。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    fixed_label: str
    per_chart_selection: dict[str, str]
    fixed: AggregateMetrics
    per_chart_best: AggregateMetrics
    best_not_worse: bool


@dataclass(frozen=True, slots=True)
class CaseEvaluation:
    """一张谱的评估结果（含分层与相位读数）。"""

    key: str
    cluster: str
    difficulty: float | None
    difficulty_band: str
    counts: MetricCounts
    quality: QualityMetrics
    primary_match: MatchResult
    phase: PhaseOutcome
    time_groups: dict[str, StratumMetrics]


def _pair_event_flags(
    pred_events: Sequence[DecodedEvent],
    gold_events: Sequence[DecodedEvent],
    matches: MatchResult,
    config: EvalConfig,
) -> tuple[bool, ...]:
    """每个匹配对是否满足 event 族判据（顺序与 matches.pairs 一致）。

    match_marking=True 时候选集已要求标记一致，于是每个匹配对都满足判据（两族重合）。
    """
    if config.match_marking:
        return tuple(True for _ in matches.pairs)
    tolerance = config.position_tolerance
    return tuple(
        marker_equal(
            pred_events[pair.pred_index],
            gold_events[pair.gold_index],
            position_x_tolerance=tolerance,
        )
        for pair in matches.pairs
    )


def _strata(
    *,
    gold_groups: Sequence[TimeGroup],
    pred_groups: Sequence[TimeGroup],
    matches: MatchResult,
    pair_event_correct: Sequence[bool],
    tolerance_s: float,
) -> dict[str, StratumMetrics]:
    """主容差下的分层指标（按 gold 事件的组归层；**空层不进报告**）。

    - gold 与 pred 各自按自己的时间组归层（pred 可能整体错位，不能借用 gold 的组）；
    - 一个匹配对归属于 **gold 所在的组**（人类意图决定「这一拍是 1/8 还是三连」）；
    - 层的 precision 分母 = 该层**全部** pred（未匹配的留在分母，plan §4.2-2）。
    """
    rows: dict[str, StratumMetrics] = {}
    for group in TIME_GROUPS:
        gold_index = [index for index, value in enumerate(gold_groups) if value is group]
        pred_index = [index for index, value in enumerate(pred_groups) if value is group]
        if not gold_index and not pred_index:
            continue
        gold_set = set(gold_index)
        pairs = [
            (pair, correct)
            for pair, correct in zip(matches.pairs, pair_event_correct, strict=True)
            if pair.gold_index in gold_set
        ]
        n_pred = len(pred_index)
        n_gold = len(gold_index)
        timing_tp = len(pairs)
        event_tp = sum(1 for _, correct in pairs if correct)
        rows[group.value] = StratumMetrics(
            name=group.value,
            n_pred=n_pred,
            n_gold=n_gold,
            n_pairs=timing_tp,
            timing=FamilyMetrics.build("timing", tolerance_s, timing_tp, n_pred, n_gold),
            event=FamilyMetrics.build("event", tolerance_s, event_tp, n_pred, n_gold),
        )
    return rows


def merge_strata(
    evaluations: Sequence[CaseEvaluation],
    config: EvalConfig,
) -> dict[str, StratumMetrics]:
    """跨谱按层名池化（层内 micro；plan §4.4 的分解表由此而来）。"""
    pooled: dict[str, StratumMetrics] = {}
    for evaluation in evaluations:
        for name, row in evaluation.time_groups.items():
            timing_tp = row.timing.tp
            event_tp = row.event.tp
            if name not in pooled:
                pooled[name] = row
                continue
            current = pooled[name]
            pooled[name] = StratumMetrics(
                name=name,
                n_pred=current.n_pred + row.n_pred,
                n_gold=current.n_gold + row.n_gold,
                n_pairs=current.n_pairs + row.n_pairs,
                timing=FamilyMetrics.build(
                    "timing",
                    config.primary_tolerance,
                    current.timing.tp + timing_tp,
                    current.n_pred + row.n_pred,
                    current.n_gold + row.n_gold,
                ),
                event=FamilyMetrics.build(
                    "event",
                    config.primary_tolerance,
                    current.event.tp + event_tp,
                    current.n_pred + row.n_pred,
                    current.n_gold + row.n_gold,
                ),
            )
    return pooled


def mean_strata(
    evaluations: Sequence[CaseEvaluation],
    config: EvalConfig,
) -> dict[str, StratumMetrics]:
    """按谱平均的分层表（macro）：层的比率逐谱平均，计数取合计。"""
    pooled = merge_strata(evaluations, config)
    grouped: dict[str, list[StratumMetrics]] = {}
    for evaluation in evaluations:
        for name, row in evaluation.time_groups.items():
            grouped.setdefault(name, []).append(row)
    out: dict[str, StratumMetrics] = {}
    for name, total in pooled.items():
        rows = grouped.get(name, [])
        timing_tp = sum(row.timing.tp for row in rows)
        event_tp = sum(row.event.tp for row in rows)
        out[name] = StratumMetrics(
            name=name,
            n_pred=total.n_pred,
            n_gold=total.n_gold,
            n_pairs=total.n_pairs,
            timing=FamilyMetrics(
                family="timing",
                tolerance_s=float(total.timing.tolerance_s),
                n_pred=total.n_pred,
                n_gold=total.n_gold,
                tp=timing_tp,
                fp=total.n_pred - timing_tp,
                fn=total.n_gold - timing_tp,
                precision=_mean([row.timing.precision for row in rows]),
                recall=_mean([row.timing.recall for row in rows]),
                f1=_mean([row.timing.f1 for row in rows]),
            ),
            event=FamilyMetrics(
                family="event",
                tolerance_s=float(total.event.tolerance_s),
                n_pred=total.n_pred,
                n_gold=total.n_gold,
                tp=event_tp,
                fp=total.n_pred - event_tp,
                fn=total.n_gold - event_tp,
                precision=_mean([row.event.precision for row in rows]),
                recall=_mean([row.event.recall for row in rows]),
                f1=_mean([row.event.f1 for row in rows]),
            ),
        )
    return out


def evaluate_case(
    case: EvalCase,
    config: EvalConfig,
    *,
    pred: Sequence[DecodedEvent] | None = None,
) -> CaseEvaluation:
    """评估一张谱（秒域；匹配 + 双容差双族 + 相位 + 分层 + 背面/类型/侧别/MAE）。

    Args:
        case: 输入单元（pred 可被 pred 参数覆盖，供两栏报告与 corruption 注入使用）。
        config: 评估配置（容差、主容差、相位搜索、归组规则等全在配置里）。
        pred: 覆盖 case.pred 的生成事件（None = 用 case.pred）。
    """
    pred_events = tuple(case.pred if pred is None else pred)
    gold_events = tuple(case.gold)
    marker_filter: MarkerFilter | None = None
    if config.match_marking:
        marker_filter = partial(marker_equal, position_x_tolerance=config.position_tolerance)

    matches = {
        tol: greedy_match(
            pred_events,
            gold_events,
            tolerance_s=tol,
            marker_filter=marker_filter,
        )
        for tol in config.tolerances_s
    }
    primary = matches[config.primary_tolerance]
    flags = _pair_event_flags(pred_events, gold_events, primary, config)

    pair_counts: dict[float, int] = {}
    event_tp: dict[float, int] = {}
    for tol, result in matches.items():
        pair_counts[tol] = result.n_pairs
        if tol == config.primary_tolerance and config.match_marking:
            event_tp[tol] = result.n_pairs
        else:
            event_tp[tol] = sum(
                1
                for pair in result.pairs
                if marker_equal(
                    pred_events[pair.pred_index],
                    gold_events[pair.gold_index],
                    position_x_tolerance=config.position_tolerance,
                )
            )

    type_gold: dict[int, int] = {}
    for event in gold_events:
        key = int(event.note_type)
        type_gold[key] = type_gold.get(key, 0) + 1
    type_tp: dict[int, int] = {}
    for pair, correct in zip(primary.pairs, flags, strict=True):
        gold_type = int(gold_events[pair.gold_index].note_type)
        if (
            correct
            and pred_events[pair.pred_index].note_type is gold_events[pair.gold_index].note_type
        ):
            type_tp[gold_type] = type_tp.get(gold_type, 0) + 1

    counts = MetricCounts(
        n_pred=len(pred_events),
        n_gold=len(gold_events),
        pair_counts=pair_counts,
        event_tp=event_tp,
        phase_offsets_s=tuple(pair.delta_s for pair in primary.pairs),
        position_abs_errors=tuple(
            abs(
                float(pred_events[pair.pred_index].position_x)
                - float(gold_events[pair.gold_index].position_x)
            )
            for pair in primary.pairs
        ),
        back_pred=sum(1 for event in pred_events if event.side is Side.BACK),
        back_gold=sum(1 for event in gold_events if event.side is Side.BACK),
        back_tp=sum(
            1
            for pair in primary.pairs
            if gold_events[pair.gold_index].side is Side.BACK
            and pred_events[pair.pred_index].side is Side.BACK
        ),
        side_correct=sum(
            1
            for pair in primary.pairs
            if pred_events[pair.pred_index].side is gold_events[pair.gold_index].side
        ),
        side_gold_front=sum(
            1 for pair in primary.pairs if gold_events[pair.gold_index].side is Side.FRONT
        ),
        type_correct=sum(
            1
            for pair in primary.pairs
            if pred_events[pair.pred_index].note_type is gold_events[pair.gold_index].note_type
        ),
        type_gold=type_gold,
        type_tp=type_tp,
    )
    quality = metrics_from_counts(counts, config)
    phase = search_phase_offset(
        pred_events,
        gold_events,
        tolerance_s=config.primary_tolerance,
        search=config.phase_search,
        marker_filter=marker_filter,
    )
    gold_groups = assign_time_groups(gold_events, case.bpm_points, rule=config.time_group_rule)
    pred_groups = assign_time_groups(pred_events, case.bpm_points, rule=config.time_group_rule)
    return CaseEvaluation(
        key=case.key,
        cluster=case.cluster,
        difficulty=case.difficulty,
        difficulty_band=difficulty_band(case.difficulty),
        counts=counts,
        quality=quality,
        primary_match=primary,
        phase=phase,
        time_groups=_strata(
            gold_groups=gold_groups,
            pred_groups=pred_groups,
            matches=primary,
            pair_event_correct=flags,
            tolerance_s=config.primary_tolerance,
        ),
    )


def aggregate(
    evaluations: Sequence[CaseEvaluation],
    config: EvalConfig,
) -> AggregateMetrics:
    """macro + micro 两栏（plan §4.2-4）。空输入 -> 两栏都是零计数的空指标。"""
    counts = MetricCounts(
        n_pred=0,
        n_gold=0,
        pair_counts={},
        event_tp={},
        phase_offsets_s=(),
        position_abs_errors=(),
        back_pred=0,
        back_gold=0,
        back_tp=0,
        side_correct=0,
        side_gold_front=0,
        type_correct=0,
        type_gold={},
        type_tp={},
    )
    for evaluation in evaluations:
        counts = counts.merge(evaluation.counts)
    return AggregateMetrics(
        per_chart_mean=mean_quality(
            [evaluation.quality for evaluation in evaluations], counts, config
        ),
        micro=metrics_from_counts(counts, config),
    )


def select_per_chart_best(
    case: EvalCase,
    candidates: Mapping[str, Sequence[DecodedEvent]],
    config: EvalConfig,
) -> tuple[str, CaseEvaluation]:
    """每谱最优候选（按 **timing-F1 @ 主容差** 选，plan §4.2-4）。

    并列时取**标签字典序最小**者（确定性；同输入 + 同 config -> 同选择，
    plan §4.9 验收表的「可复现」一行）。
    """
    if not candidates:
        raise ValueError(f"候选集为空：{case.key!r} 无法选优")
    scored: list[tuple[str, CaseEvaluation]] = []
    for label in sorted(candidates):
        evaluation = evaluate_case(case, config, pred=candidates[label])
        scored.append((label, evaluation))
    best = scored[0]
    best_f1 = best[1].quality.timing_f1(config.primary_tolerance)
    for label, evaluation in scored[1:]:
        value = evaluation.quality.timing_f1(config.primary_tolerance)
        if value > best_f1:
            best = (label, evaluation)
            best_f1 = value
    return best


def two_column_report(
    cases: Sequence[EvalCase],
    candidates: Mapping[str, Mapping[str, Sequence[DecodedEvent]]],
    config: EvalConfig,
    *,
    fixed_label: str,
) -> TwoColumnReport:
    """构造「固定解码规则」与「每谱最优」两栏（plan §4.2-4）。

    「最优 >= 固定」是**结构性的**：每张谱的最优栏在自己的候选集上取 argmax，
    而固定栏是其中一个候选，因此逐谱 F1 不会更差（best_not_worse 记录该断言）。
    """
    fixed_evaluations: list[CaseEvaluation] = []
    best_evaluations: list[CaseEvaluation] = []
    selection: dict[str, str] = {}
    for case in cases:
        table = candidates.get(case.key)
        if table is None:
            raise ValueError(f"候选集缺少谱面 {case.key!r}")
        if fixed_label not in table:
            raise ValueError(f"谱面 {case.key!r} 的候选集缺少固定规则 {fixed_label!r}")
        fixed_evaluations.append(evaluate_case(case, config, pred=table[fixed_label]))
        label, evaluation = select_per_chart_best(case, table, config)
        selection[case.key] = label
        best_evaluations.append(evaluation)
    not_worse = all(
        best.quality.timing_f1(config.primary_tolerance)
        >= fixed.quality.timing_f1(config.primary_tolerance)
        for best, fixed in zip(best_evaluations, fixed_evaluations, strict=True)
    )
    return TwoColumnReport(
        fixed_label=fixed_label,
        per_chart_selection=selection,
        fixed=aggregate(fixed_evaluations, config),
        per_chart_best=aggregate(best_evaluations, config),
        best_not_worse=not_worse,
    )


__all__ = [
    "AggregateMetrics",
    "CaseEvaluation",
    "FamilyMetrics",
    "MetricCounts",
    "QualityMetrics",
    "StratumMetrics",
    "TwoColumnReport",
    "aggregate",
    "evaluate_case",
    "mean_quality",
    "mean_strata",
    "merge_strata",
    "metrics_from_counts",
    "select_per_chart_best",
    "two_column_report",
]
