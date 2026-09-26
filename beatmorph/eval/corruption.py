"""指标 corruption 准入：dose-controlled 注入 + 不变性控制 + 自动排除（plan 06 §4.8 / M6.6）。

文献依据（文献库 §7.6(c)，ChartGenEval 方法论）——对**每一个**候选指标要求三条同时成立：

1. 目标读数随失败强度呈**负的 dose-rank 关联**；
2. 最强剂量与匹配对照**显著不同**（此处用 song-cluster bootstrap 区间是否排除 0）；
3. 所有**预先声明的不变性控制**通过。

**未通过者只能进附录，不得进主报告**——本模块把这条落成可执行判据
（main_report_metric_names 只列出准入通过的指标）。

标配注入集（plan §4.8 拟定）：① 全谱时间偏移；② 类型打乱；③ 侧别翻转；
④ 循环塌缩；⑤ 密度缩放；⑥ 局部爆发。**本实现新增第 ⑦ 类 positionX 抖动**，
理由与偏离声明：plan §4.1 要求把 positionX MAE 列为主报告指标，而 ①-⑥ 都不改动
positionX（时间/类型/侧别/结构腐败与 x 轴正交）→ 按 §4.8 的判据 MAE 将无法通过
准入而被排除；为让 x 轴指标有可响应的注入，新增「把 dose 比例的事件沿 x 平移一个桶宽
（dx，派生量）」这一类，并在 MetricSpec.responsive_to 里**声明**每个指标的响应集。
该偏离需主会话裁定（已在 plan 06 §6 的 M6.6 实施状态列登记；是否补入 §9 由主会话决定）。

**指标方向统一约定**：所有被准入的分数一律是「**越高越好**」。误差型指标（MAE）
取负号（position_x_mae_neg），否则方向判据会出现两种符号、无法机械判定。
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import Final

from pydantic import BaseModel, ConfigDict

from beatmorph.core.contracts.phigros import NoteType, Side
from beatmorph.decoder.events import DecodedEvent
from beatmorph.eval.metrics import QualityMetrics, aggregate, evaluate_case
from beatmorph.eval.protocol import EvalCase, EvalConfig
from beatmorph.eval.stats import (
    DEFAULT_ALPHA,
    bootstrap_interval,
    spearman_rank_correlation,
)

#: 全部音符类型（注入用；顺序即契约顺序 TAP/HOLD/FLICK/DRAG）。
NOTE_TYPES: Final[tuple[NoteType, ...]] = tuple(NoteType)

#: 侧别翻转映射（**不是** truthiness：两个成员都是真值，plan 00 偏离 3）。
SIDE_FLIP: Final[dict[Side, Side]] = {Side.FRONT: Side.BACK, Side.BACK: Side.FRONT}

#: 默认剂量网格（0 为对照；**必须含 0**，否则不变性控制没有对照点）。
DEFAULT_DOSES: Final[tuple[float, ...]] = (0.0, 0.25, 0.5, 0.75, 1.0)

#: 单调性判据的浮点容差（比率是浮点运算，允许 1 ulp 级抖动）。
MONOTONE_EPS: Final[float] = 1e-12


@dataclass(frozen=True, slots=True)
class Injection:
    """一类 dose-controlled 注入。

    `apply(events, dose)` 的契约：

    - `dose == 0.0` **必须逐位返回原事件**（不变性控制的对照点，M6.6）；
    - 同一 `events` 下，剂量越大改动的事件集**嵌套包含**（否则剂量-响应不单调，
      指标退化的方向会被注入本身的不确定性污染）；
    - 只改动 pred（gold 不动）：这正是「生成侧失败」的语义。
    """

    name: str
    description: str
    apply: Callable[[Sequence[DecodedEvent], float], tuple[DecodedEvent, ...]]


def _selected(count: int, dose: float, seed: int) -> tuple[int, list[int]]:
    """按 dose 选出要改动的事件下标（**嵌套**：dose 越大集合越大）。"""
    if count <= 0:
        return (0, [])
    rng = random.Random(seed)
    order = list(range(count))
    rng.shuffle(order)
    take = round(max(0.0, min(1.0, float(dose))) * count)
    return (take, order[:take])


def time_shift_injection(config: EvalConfig, *, tolerance_multiple: float = 4.0) -> Injection:
    """① 全谱时间偏移：0 → `tolerance_multiple` × 主容差（plan §4.8：「容差的数倍」）。"""
    span = float(tolerance_multiple) * config.primary_tolerance

    def apply(events: Sequence[DecodedEvent], dose: float) -> tuple[DecodedEvent, ...]:
        shift = float(dose) * span
        if shift == 0.0:
            return tuple(events)
        return tuple(replace(event, t_s=float(event.t_s) + shift) for event in events)

    return Injection(
        name="time_shift",
        description=f"全谱时间偏移 0 → {span:g}s（= {tolerance_multiple:g} × 主容差）",
        apply=apply,
    )


def type_shuffle_injection(*, seed: int = 0) -> Injection:
    """② 类型打乱：dose 比例的事件被换成**另一种**类型（嵌套选择，确定种子）。"""

    def apply(events: Sequence[DecodedEvent], dose: float) -> tuple[DecodedEvent, ...]:
        events = tuple(events)
        if not events:
            return ()
        rng = random.Random(seed)
        order = list(range(len(events)))
        rng.shuffle(order)
        replacement = [
            [value for value in NOTE_TYPES if value is not event.note_type][
                rng.randrange(len(NOTE_TYPES) - 1)
            ]
            for event in events
        ]
        take = round(max(0.0, min(1.0, float(dose))) * len(events))
        if take == 0:
            return events
        changed = set(order[:take])
        return tuple(
            replace(event, note_type=replacement[index]) if index in changed else event
            for index, event in enumerate(events)
        )

    return Injection(
        name="type_shuffle",
        description="dose 比例的事件类型被换成另一种类型（时间不动）",
        apply=apply,
    )


def side_flip_injection(*, seed: int = 0) -> Injection:
    """③ 侧别翻转：dose 比例的事件 FRONT <-> BACK（背面 recall 的判据注入）。"""

    def apply(events: Sequence[DecodedEvent], dose: float) -> tuple[DecodedEvent, ...]:
        events = tuple(events)
        take, chosen = _selected(len(events), dose, seed)
        if take == 0:
            return events
        changed = set(chosen)
        return tuple(
            replace(event, side=SIDE_FLIP[event.side]) if index in changed else event
            for index, event in enumerate(events)
        )

    return Injection(
        name="side_flip",
        description="dose 比例的事件侧别翻转（FRONT <-> BACK）",
        apply=apply,
    )


def loop_collapse_injection() -> Injection:
    """④ 循环塌缩：把前 `(1-dose)` 比例的片段在时间上平铺（自相似度上升的经典腐败）。

    dose = 0 时 motif = 全部事件 → **逐位返回原事件**；dose = 1 时 motif = 1 个事件 →
    所有事件塌到同一时刻。
    """

    def apply(events: Sequence[DecodedEvent], dose: float) -> tuple[DecodedEvent, ...]:
        events = tuple(events)
        count = len(events)
        if count <= 1:
            return events
        motif = max(1, round(count * (1.0 - max(0.0, min(1.0, float(dose))))))
        if motif >= count:
            return events
        base = float(events[0].t_s)
        period = float(events[motif - 1].t_s) - base
        out: list[DecodedEvent] = []
        for index, event in enumerate(events):
            source = events[index % motif]
            block = index // motif
            out.append(
                replace(
                    source,
                    t_s=base + block * period + (float(source.t_s) - base),
                    hold_time_s=float(event.hold_time_s),
                    confidence=float(event.confidence),
                )
            )
        return tuple(out)

    return Injection(
        name="loop_collapse",
        description="前 (1-dose) 比例的片段在时间上平铺（结构塌缩为循环）",
        apply=apply,
    )


def density_scale_injection(*, seed: int = 0) -> Injection:
    """⑤ 密度缩放：保持结构、只改数量（dose 比例的事件被**丢弃**）。

    ⚠️ 这正是「未匹配的生成事件留在分母」的反向用例：删事件**不得**提高 precision
    （plan §4.2-2）。对完美基线而言 F1/recall 单调退化。
    """

    def apply(events: Sequence[DecodedEvent], dose: float) -> tuple[DecodedEvent, ...]:
        events = tuple(events)
        take, chosen = _selected(len(events), dose, seed)
        if take == 0:
            return events
        dropped = set(chosen)
        return tuple(event for index, event in enumerate(events) if index not in dropped)

    return Injection(
        name="density_scale",
        description="dose 比例的事件被丢弃（数量变化、结构与时刻不变）",
        apply=apply,
    )


def density_burst_injection(*, seed: int = 0) -> Injection:
    """⑥ 局部爆发：dose 比例的事件被堆到同一时刻（单点密集堆叠）。"""

    def apply(events: Sequence[DecodedEvent], dose: float) -> tuple[DecodedEvent, ...]:
        events = tuple(events)
        take, chosen = _selected(len(events), dose, seed)
        if take == 0:
            return events
        moved = set(chosen)
        burst_time = float(events[chosen[0]].t_s)
        return tuple(
            replace(event, t_s=burst_time) if index in moved else event
            for index, event in enumerate(events)
        )

    return Injection(
        name="density_burst",
        description="dose 比例的事件被搬到同一时刻（局部密度爆发）",
        apply=apply,
    )


def position_x_jitter_injection(config: EvalConfig, *, seed: int = 0) -> Injection:
    """⑦ positionX 抖动（**本实现新增**，见模块 docstring 的偏离声明）。

    dose 比例的事件沿 x 平移**一个桶宽 dx**（派生量，红线 7）——这让 positionX MAE
    有可响应的注入（①-⑥ 都与 x 轴正交）。
    """

    step = config.dx

    def apply(events: Sequence[DecodedEvent], dose: float) -> tuple[DecodedEvent, ...]:
        events = tuple(events)
        take, chosen = _selected(len(events), dose, seed)
        if take == 0:
            return events
        moved = set(chosen)
        return tuple(
            replace(event, position_x=float(event.position_x) + step) if index in moved else event
            for index, event in enumerate(events)
        )

    return Injection(
        name="position_x_jitter",
        description="dose 比例的事件沿 x 平移一个桶宽 dx（positionX MAE 的响应注入）",
        apply=apply,
    )


def default_injections(config: EvalConfig, *, seed: int = 0) -> tuple[Injection, ...]:
    """标配注入集：plan §4.8 的 ①-⑥ + 本实现新增的 ⑦（positionX 抖动）。"""
    return (
        time_shift_injection(config),
        type_shuffle_injection(seed=seed),
        side_flip_injection(seed=seed),
        loop_collapse_injection(),
        density_scale_injection(seed=seed),
        density_burst_injection(seed=seed),
        position_x_jitter_injection(config, seed=seed),
    )


MetricScore = Callable[[Sequence[EvalCase], EvalConfig], float]

#: plan §4.8 拟定的六类标配注入（名字与 default_injections 一致）。
STANDARD_INJECTIONS: Final[tuple[str, ...]] = (
    "time_shift",
    "type_shuffle",
    "side_flip",
    "loop_collapse",
    "density_scale",
    "density_burst",
)
#: 本实现新增的 x 轴注入（见模块 docstring 的偏离声明）。
X_AXIS_INJECTIONS: Final[tuple[str, ...]] = ("position_x_jitter",)


def _macro_quality(cases: Sequence[EvalCase], config: EvalConfig) -> QualityMetrics:
    """按谱平均（macro）的质量分节（corruption 的读数一律取 macro 栏）。"""
    return aggregate([evaluate_case(case, config) for case in cases], config).per_chart_mean


def _timing_score(cases: Sequence[EvalCase], config: EvalConfig, *, tolerance_s: float) -> float:
    """timing-F1 @ 容差（macro）。"""
    return float(_macro_quality(cases, config).timing_f1(tolerance_s))


def _event_score(cases: Sequence[EvalCase], config: EvalConfig, *, tolerance_s: float) -> float:
    """event-F1 @ 容差（macro）。"""
    return float(_macro_quality(cases, config).event_f1(tolerance_s))


def _back_recall_score(cases: Sequence[EvalCase], config: EvalConfig) -> float:
    """背面 recall（macro）。"""
    return float(_macro_quality(cases, config).back_recall)


def _side_accuracy_score(cases: Sequence[EvalCase], config: EvalConfig) -> float:
    """侧别准确率（匹配对上；macro）。"""
    return float(_macro_quality(cases, config).side_accuracy)


def _type_accuracy_score(cases: Sequence[EvalCase], config: EvalConfig) -> float:
    """类型准确率（匹配对上；macro）。"""
    return float(_macro_quality(cases, config).type_accuracy)


def _position_x_mae_neg_score(cases: Sequence[EvalCase], config: EvalConfig) -> float:
    """positionX MAE 的**负值**（统一为「越高越好」；MAE 缺失时记 0.0 = 最差）。"""
    value = _macro_quality(cases, config).position_x_mae
    return -float(value) if value is not None else 0.0


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """一个候选指标（**分数方向一律「越高越好」**）。

    Attributes:
        name: 指标名（报告与准入表的键）。
        score: corpus -> 分数（macro 栏；corruption 的读数一律取 macro）。
        responsive_to: **声明的响应注入集**——准入只在这个集合里要求「至少一个显著退化」；
            None = 全部注入。声明是必需的：时间/类型/侧别腐败与 x 轴正交，若不声明，
            positionX MAE 这类指标会被误判为不合格。
        note: 口径说明（随报告落盘）。
    """

    name: str
    score: MetricScore
    responsive_to: tuple[str, ...] | None = None
    note: str = ""


def primary_metric_specs(config: EvalConfig) -> tuple[MetricSpec, ...]:
    """主报告的候选指标集（plan §4.1 的事件级指标 + 误差型的负号版本）。"""
    specs: list[MetricSpec] = []
    for tolerance in config.tolerances_s:
        label = config.tolerance_key(tolerance)
        specs.append(
            MetricSpec(
                name=f"timing_f1@{label}",
                score=partial(_timing_score, tolerance_s=tolerance),
                note="只比时间；未匹配 pred 留在分母",
            )
        )
        specs.append(
            MetricSpec(
                name=f"event_f1@{label}",
                score=partial(_event_score, tolerance_s=tolerance),
                note="标记（line/x/side/type）也须一致",
            )
        )
    specs.append(
        MetricSpec(
            name="back_recall", score=_back_recall_score, note="背面 gold 的 recall（单独报）"
        )
    )
    specs.append(
        MetricSpec(name="side_accuracy", score=_side_accuracy_score, note="匹配对上的侧别准确率")
    )
    specs.append(
        MetricSpec(name="type_accuracy", score=_type_accuracy_score, note="匹配对上的类型准确率")
    )
    specs.append(
        MetricSpec(
            name="position_x_mae_neg",
            score=_position_x_mae_neg_score,
            responsive_to=X_AXIS_INJECTIONS,
            note="MAE 取负；①-⑥ 与 x 轴正交，故响应集只声明 x 抖动",
        )
    )
    return tuple(specs)


class InjectionResponse(BaseModel):
    """一个 (指标 × 注入) 的剂量-响应读数（plan §4.8 的三条判据之 (1)(2)）。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: str
    injection: str
    declared: bool
    doses: list[float]
    values: list[float]
    dose_rank_correlation: float | None
    monotone_nonincreasing: bool
    delta: float
    delta_ci: tuple[float, float]
    significant_decrease: bool
    significant_increase: bool


class InvarianceResult(BaseModel):
    """一项预先声明的不变性控制的结果（plan §4.8 判据之 (3)）。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    passed: bool
    detail: str


class MetricAdmission(BaseModel):
    """一个指标的准入结论（admissible=False 者**不得进主报告**，plan §4.8 末段）。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: str
    admissible: bool
    reasons: list[str]
    invariants: list[InvarianceResult]
    responses: list[InjectionResponse]


def _finite_or_none(value: float) -> float | None:
    """NaN -> None（报告是冻结字段，NaN 不是合法 JSON 数字）。"""
    return None if math.isnan(value) else float(value)


def _with_dose(cases: Sequence[EvalCase], injection: Injection, dose: float) -> list[EvalCase]:
    """把注入施加到每张谱的 pred 上（gold / BPM / 元数据不变）。"""
    return [case.with_pred(injection.apply(case.pred, dose)) for case in cases]


def _cluster_deltas(
    cases: Sequence[EvalCase],
    *,
    injection: Injection,
    metric: MetricSpec,
    config: EvalConfig,
    low_dose: float,
    high_dose: float,
) -> list[float]:
    """每个 cluster（曲目）的「最强剂量 - 对照」差值（song-cluster bootstrap 的样本）。

    cluster 定义在 plan §4.8/§9-2 未定 → 本实现按 EvalCase.cluster（曲目标识）聚类。
    """
    grouped: dict[str, list[EvalCase]] = {}
    for case in cases:
        grouped.setdefault(case.cluster, []).append(case)
    samples: list[float] = []
    for group in grouped.values():
        low = metric.score(_with_dose(group, injection, low_dose), config)
        high = metric.score(_with_dose(group, injection, high_dose), config)
        samples.append(float(high) - float(low))
    return samples


def corruption_sweep(
    cases: Sequence[EvalCase],
    *,
    metric: MetricSpec,
    injection: Injection,
    config: EvalConfig,
    doses: Sequence[float] = DEFAULT_DOSES,
) -> InjectionResponse:
    """一个 (指标 × 注入) 的完整剂量-响应（含 cluster bootstrap 区间）。

    Raises:
        ValueError: doses 为空、首项不为 0、或未按升序给出。
    """
    if not doses:
        raise ValueError("doses 不得为空")
    if float(doses[0]) != 0.0:
        raise ValueError(f"剂量网格必须以 0 打头（不变性控制的对照点），得到 {doses!r}")
    if list(doses) != sorted(doses):
        raise ValueError(f"剂量网格必须按升序给出，得到 {doses!r}")
    values = [float(metric.score(_with_dose(cases, injection, dose), config)) for dose in doses]
    rho = spearman_rank_correlation([float(dose) for dose in doses], values)
    monotone = all(
        values[index + 1] <= values[index] + MONOTONE_EPS for index in range(len(values) - 1)
    )
    samples = _cluster_deltas(
        cases,
        injection=injection,
        metric=metric,
        config=config,
        low_dose=float(doses[0]),
        high_dose=float(doses[-1]),
    )
    low, high = bootstrap_interval(
        samples,
        resamples=config.bootstrap_resamples,
        seed=config.bootstrap_seed,
        alpha=DEFAULT_ALPHA,
    )
    declared = metric.responsive_to is None or injection.name in metric.responsive_to
    return InjectionResponse(
        metric=metric.name,
        injection=injection.name,
        declared=declared,
        doses=[float(dose) for dose in doses],
        values=values,
        dose_rank_correlation=_finite_or_none(rho),
        monotone_nonincreasing=monotone,
        delta=values[-1] - values[0],
        delta_ci=(low, high),
        significant_decrease=(not math.isnan(high))
        and high < 0.0
        and (not math.isnan(rho))
        and rho < 0.0,
        significant_increase=(not math.isnan(low)) and low > 0.0,
    )


def invariance_controls(
    cases: Sequence[EvalCase],
    *,
    metric: MetricSpec,
    injections: Sequence[Injection],
    config: EvalConfig,
) -> list[InvarianceResult]:
    """**两项预先声明的不变性控制**（M6.6 要求 2 项，逐条声明）：

    1. `identity_recompute`：不注入（每个注入的 dose = 0）时指标**逐位不变**
       （同时覆盖「同一输入重复计算可复现」）；
    2. `input_permutation`：打乱 pred 的输入顺序后指标**逐位不变**
       （匹配与聚合必须与输入顺序无关——这正是贪心匹配用规范键排序的理由）。
    """
    baseline = float(metric.score(cases, config))
    identity_ok = True
    identity_detail = "每个注入的 dose=0 都逐位返回原指标"
    for injection in injections:
        value = float(metric.score(_with_dose(cases, injection, 0.0), config))
        if value != baseline:
            identity_ok = False
            identity_detail = (
                f"注入 {injection.name} 在 dose=0 时改变了指标：{value!r} != {baseline!r}"
            )
            break
    generator = random.Random(config.bootstrap_seed)
    permuted: list[EvalCase] = []
    for case in cases:
        events = list(case.pred)
        generator.shuffle(events)
        permuted.append(case.with_pred(events))
    permuted_value = float(metric.score(permuted, config))
    permutation_ok = permuted_value == baseline
    return [
        InvarianceResult(name="identity_recompute", passed=identity_ok, detail=identity_detail),
        InvarianceResult(
            name="input_permutation",
            passed=permutation_ok,
            detail=(
                "打乱 pred 顺序后指标逐位不变"
                if permutation_ok
                else f"打乱 pred 顺序改变了指标：{permuted_value!r} != {baseline!r}"
            ),
        ),
    ]


def admit_metric(
    cases: Sequence[EvalCase],
    *,
    metric: MetricSpec,
    injections: Sequence[Injection],
    config: EvalConfig,
    doses: Sequence[float] = DEFAULT_DOSES,
) -> MetricAdmission:
    """对**一个**指标跑 corruption 准入（plan §4.8 的三条判据）。

    判据（逐条落成机器可判的形式）：

    1. 声明的响应注入集内**至少一个**注入使指标显著退化
       （cluster bootstrap 区间上界 < 0 **且** dose-rank 关联为负）；
    2. **没有任何**注入使指标显著上升（区间下界 > 0）——方向错误的指标一律不合格
       （文献库 §7.6(c) 的两条反面教训：perplexity 与 self-similarity 都会朝反方向动）；
    3. 两项不变性控制全部通过。

    三条同时成立才 `admissible=True`；否则只进附录（main_report_metric_names 会剔除）。
    """
    invariants = invariance_controls(cases, metric=metric, injections=injections, config=config)
    responses = [
        corruption_sweep(cases, metric=metric, injection=injection, config=config, doses=doses)
        for injection in injections
    ]
    declared = [response for response in responses if response.declared]
    decreases = [response for response in declared if response.significant_decrease]
    increases = [response for response in responses if response.significant_increase]
    reasons: list[str] = []
    if not all(result.passed for result in invariants):
        failed = ", ".join(result.name for result in invariants if not result.passed)
        reasons.append(f"不变性控制未通过：{failed}")
    if not decreases:
        names = ", ".join(response.injection for response in declared) or "(空)"
        reasons.append(f"声明的响应注入集内没有显著退化（{names}）")
    if increases:
        names = ", ".join(response.injection for response in increases)
        reasons.append(f"以下注入使指标显著**上升**（方向错误）：{names}")
    return MetricAdmission(
        metric=metric.name,
        admissible=not reasons,
        reasons=reasons,
        invariants=invariants,
        responses=responses,
    )


def admit_metrics(
    cases: Sequence[EvalCase],
    *,
    metrics: Sequence[MetricSpec],
    injections: Sequence[Injection],
    config: EvalConfig,
    doses: Sequence[float] = DEFAULT_DOSES,
) -> tuple[MetricAdmission, ...]:
    """对全部候选指标跑准入（顺序与 metrics 一致）。"""
    return tuple(
        admit_metric(cases, metric=metric, injections=injections, config=config, doses=doses)
        for metric in metrics
    )


def main_report_metric_names(admissions: Sequence[MetricAdmission]) -> tuple[str, ...]:
    """主报告允许出现的指标名（**未通过准入者被自动剔除**，plan §4.8 末段）。"""
    return tuple(admission.metric for admission in admissions if admission.admissible)


def assert_admissible(admission: MetricAdmission) -> None:
    """把「未通过者不得进主报告」写成可执行断言（CI 可验）。"""
    if not admission.admissible:
        raise AssertionError(
            f"指标 {admission.metric!r} 未通过 corruption 准入：{admission.reasons}"
        )


__all__ = [
    "DEFAULT_DOSES",
    "MONOTONE_EPS",
    "NOTE_TYPES",
    "SIDE_FLIP",
    "STANDARD_INJECTIONS",
    "X_AXIS_INJECTIONS",
    "Injection",
    "InjectionResponse",
    "InvarianceResult",
    "MetricAdmission",
    "MetricSpec",
    "admit_metric",
    "admit_metrics",
    "assert_admissible",
    "corruption_sweep",
    "default_injections",
    "density_burst_injection",
    "density_scale_injection",
    "invariance_controls",
    "loop_collapse_injection",
    "main_report_metric_names",
    "position_x_jitter_injection",
    "primary_metric_specs",
    "side_flip_injection",
    "time_shift_injection",
    "type_shuffle_injection",
]
