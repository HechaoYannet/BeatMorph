"""解码编排：强度场 -> 事件 -> 谱面 IR -> 合法性报告（Plan 05 §3.2 / §3.3）。

一次调用的完整链路（每一步都有独立可测的入口，本模块只负责把它们串起来）：

    lambda 场 --(D1 peaks / D2 thinning)--> FieldEvent
             --(pair_events，Hold 配对)----> DecodedEvent
             --(postprocess_events)-------> 合法事件 + 报告(事件层)
             --(events_to_notes + 判定线)---> PhigrosChart
             --(postprocess_chart)--------> 可导出谱面 + 报告(谱面层)

**判定线不生成**（RFC-0029 §2.2 Q2：线事件轨是条件输入），因此解码必须接收一个
"模板谱面"（提供 `lines` / `bpm_points` / `meta`）；省略模板时按 K 条空线合成，
此时导出的谱面在模拟器里只是一堆静止判定线上的音符——**可用，但没有观感**。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from beatmorph.core.contracts.field import ChartFieldSpec
from beatmorph.core.contracts.legality import LegalityReport
from beatmorph.core.contracts.phigros import (
    BpmPoint,
    ChartMeta,
    ChartSource,
    JudgeLine,
    PhigrosChart,
)
from beatmorph.decoder.events import DecodedEvent, PairingStats, events_to_notes, pair_events
from beatmorph.decoder.fieldops import to_numpy
from beatmorph.decoder.peaks import PeakConfig, decode_peaks
from beatmorph.decoder.postprocess.legality import (
    LegalityConfig,
    merge_reports,
    postprocess_chart,
    postprocess_events,
)
from beatmorph.decoder.thinning import ThinningConfig, decode_thinning
from beatmorph.field.grid import FieldGrid

#: 解码臂（RFC-0029 §3.4 的 B6 对照臂）。
DecodeMethod = Literal["peaks", "thinning"]


@dataclass(frozen=True, slots=True)
class DecodeConfig:
    """一次解码的全部配置（两臂的超参各自独立声明，不共享隐式默认）。"""

    method: DecodeMethod = "peaks"
    peak: PeakConfig = field(default_factory=PeakConfig)
    thinning: ThinningConfig = field(default_factory=ThinningConfig)
    legality: LegalityConfig = field(default_factory=LegalityConfig)


@dataclass(frozen=True, slots=True)
class DecodeResult:
    """解码产物：可导出谱面 + 合并报告 + 事件（供 plan 06 的秒域评估消费）。"""

    chart: PhigrosChart
    report: LegalityReport
    events: tuple[DecodedEvent, ...]
    pairing: PairingStats
    method: DecodeMethod
    stats: dict[str, float]

    @property
    def is_legal(self) -> bool:
        """`violations` 为空 == 允许导出（红线 6 的唯一判据）。"""
        return self.report.is_legal


def default_lines(k: int) -> list[JudgeLine]:
    """K 条**无事件**的判定线（模板缺省时的最小可用线集）。"""
    if k < 1:
        raise ValueError(f"K 必须 >= 1，得到 {k!r}")
    return [JudgeLine(line_id=index) for index in range(k)]


def events_from_chart(chart: PhigrosChart) -> list[DecodedEvent]:
    """`PhigrosChart` -> `DecodedEvent` 列表（**秒域**；供评估与往返测试）。

    置信度在 IR 里不存在（它不是谱面信息），因此统一置 0.0 —— 需要置信度时
    必须从解码路径取，不得从谱面反推。
    """
    return [
        DecodedEvent(
            t_s=float(note.t),
            line_id=note.line_id,
            position_x=float(note.position_x),
            side=note.side,
            note_type=note.type,
            hold_time_s=float(note.hold_time),
            is_fake=bool(note.is_fake),
            confidence=0.0,
        )
        for note in chart.sorted_notes()
    ]


def chart_from_events(
    events: list[DecodedEvent],
    *,
    template: PhigrosChart | None,
    k: int,
    bpm_points: tuple[BpmPoint, ...] | list[BpmPoint],
    chart_time_s: float,
    method: DecodeMethod,
) -> PhigrosChart:
    """`DecodedEvent` -> `PhigrosChart`（判定线/元数据来自模板；无模板时合成）。"""
    if template is None:
        lines = default_lines(k)
        bpm = list(bpm_points)
        meta = ChartMeta(chart_time_s=float(chart_time_s))
        source = ChartSource(sniff_evidence=f"decoded:{method}")
    else:
        if len(template.lines) != k:
            raise ValueError(
                f"模板谱面有 {len(template.lines)} 条判定线，场有 K={k}："
                "判定线身份不可互换（RFC-0029 §2.4-3），必须一一对应",
            )
        lines = list(template.lines)
        bpm = list(template.bpm_points)
        meta = template.meta.model_copy(update={"chart_time_s": float(chart_time_s)})
        source = template.source.model_copy(update={"sniff_evidence": f"decoded:{method}"})
    return PhigrosChart(
        lines=lines,
        notes=events_to_notes(events),
        bpm_points=bpm,
        meta=meta,
        source=source,
    )


def decode_field(
    lam: object,
    grid: FieldGrid,
    *,
    template: PhigrosChart | None = None,
    config: DecodeConfig | None = None,
    spec: ChartFieldSpec | None = None,
) -> DecodeResult:
    """强度场 -> 谱面（Plan 05 §3.2 的对外入口）。

    Args:
        lam: 强度场 `(K, T, X, S, C)`（torch / numpy 均可）。
        grid: 已绑定时间轴的网格（`t_bins > 0` 且 `bpm_points` 非空）。
        template: 提供判定线与元数据的模板谱面（`lines` 数必须等于 K）。
        config: 解码配置。
        spec: 网格规格；省略时由 `grid` 与 `lam` 的 K 现构并断言。
    """
    settings = DecodeConfig() if config is None else config
    values = to_numpy(lam)
    if values.ndim != 5:
        raise ValueError(f"场张量必须是 5 维 (K, T, X, S, C)，得到 {values.shape}")
    k = int(values.shape[0])
    resolved_spec = grid.spec(k) if spec is None else spec
    resolved_spec.assert_grid()
    if values.shape != resolved_spec.shape():
        raise ValueError(f"场张量形状 {values.shape} != 契约形状 {resolved_spec.shape()}")

    if settings.method == "peaks":
        field_events, stats = decode_peaks(values, grid, resolved_spec, config=settings.peak)
    elif settings.method == "thinning":
        field_events, stats = decode_thinning(
            values,
            grid,
            resolved_spec,
            config=settings.thinning,
        )
    else:  # pragma: no cover - 由 Literal 约束
        raise ValueError(f"未知的解码臂：{settings.method!r}")

    decoded, pairing = pair_events(field_events, grid, spec=resolved_spec)
    fixed_events, event_report = postprocess_events(decoded, pairing=pairing)

    axis_seconds = float(grid.total_seconds)
    last_seconds = max((event.t_s + event.hold_time_s for event in fixed_events), default=0.0)
    chart_time_s = max(axis_seconds, last_seconds)
    chart = chart_from_events(
        fixed_events,
        template=template,
        k=k,
        bpm_points=grid.bpm_points,
        chart_time_s=chart_time_s,
        method=settings.method,
    )
    chart_result = postprocess_chart(chart, grid=grid, config=settings.legality)
    report = merge_reports(event_report, chart_result.report)

    stats = dict(stats)
    stats["decode_method_peaks"] = 1.0 if settings.method == "peaks" else 0.0
    stats["decode_events_field"] = float(len(field_events))
    stats["decode_events_final"] = float(len(fixed_events))
    stats["decode_axis_seconds"] = axis_seconds
    stats["decode_chart_time_s"] = chart_time_s
    merged = report.with_stats(stats)
    return DecodeResult(
        chart=chart_result.chart,
        report=merged,
        events=tuple(fixed_events),
        pairing=pairing,
        method=settings.method,
        stats=dict(merged.stats),
    )


def decode_both_arms(
    lam: object,
    grid: FieldGrid,
    *,
    template: PhigrosChart | None = None,
    config: DecodeConfig | None = None,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> dict[str, list[DecodeResult]]:
    """B6 对照臂（M5.7 的**出数工具**）：两臂各跑多个 seed，返回逐次结果。

    D1（峰值）没有随机性，因此多个 seed 结果相同——这不是 bug 而是如实行为：
    把它一并列出，正是为了在报告里说清"差异来自解码而非模型"的对照层级
    （RFC-0029 §5.2 B6）。D2 的 seed 逐次写入 `ThinningConfig.seed`。
    """
    settings = DecodeConfig() if config is None else config
    base = settings.thinning
    results: dict[str, list[DecodeResult]] = {"peaks": [], "thinning": []}
    results["peaks"].append(
        decode_field(
            lam,
            grid,
            template=template,
            config=DecodeConfig(
                method="peaks",
                peak=settings.peak,
                thinning=base,
                legality=settings.legality,
            ),
        ),
    )
    for seed in seeds:
        results["thinning"].append(
            decode_field(
                lam,
                grid,
                template=template,
                config=DecodeConfig(
                    method="thinning",
                    peak=settings.peak,
                    thinning=ThinningConfig(
                        safety=base.safety,
                        block_beats=base.block_beats,
                        mark_mode=base.mark_mode,
                        seed=int(seed),
                        bound_policy=base.bound_policy,
                        max_restarts=base.max_restarts,
                    ),
                    legality=settings.legality,
                ),
            ),
        )
    return results


def count_events(chart: PhigrosChart) -> int:
    """谱面的事件总数（`hold` 起点与终点各计 1，与目标场口径一致）。"""
    return sum(2 if note.is_hold() and note.hold_time > 0.0 else 1 for note in chart.notes)


__all__ = [
    "DecodeConfig",
    "DecodeMethod",
    "DecodeResult",
    "chart_from_events",
    "count_events",
    "decode_both_arms",
    "decode_field",
    "default_lines",
    "events_from_chart",
]
