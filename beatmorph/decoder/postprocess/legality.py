"""合法性后处理：只校验与钳位，**不美化、不改落点**——Plan 05 §4.3 / 红线 3。

三层职责（严格分开，互不越权）：

| 层 | 能做什么 | 不能做什么 |
|----|---------|-----------|
| `check_events` | 检出反向 Hold（`hold_time_s < 0`） | 不改事件 |
| `check_chart` | 检出重复事件 / line_id 越界 / （可选）同刻上限；**统计**越界、同刻分布、跨线冲突、Hold 期间线速度变化 | 不钳位、不移动、不删除 |
| `fix_events` / `fix_chart` | 丢弃**不可表示**的事件（反向 Hold、line_id 越界、完全重复）并逐条留痕 | 绝不改 `position_x`（`EditKind` 无 clamp 成员） |

三类"未查证"项按 plan 05 §4.3 **只进统计**，不作为红线：

1. **同刻按键上限**的具体数值未查证（plan 05 §9-1）→ 只有调用方显式给出
   `same_instant_limit` 时才升级为违规；默认只报分布。
2. **跨线几何冲突**的判据阈值未定义（plan 05 §9-2）→ 本模块给出一个**显式声明**的
   判据（同一 τ 格内不同线、舞台系距离 <= `tolerance_bins * dx`），只报计数。
3. **RPE 下「Hold 期间判定线速度变化」是否硬约束未确证**（plan 05 §9-3 / §9-11，
   格式文档 §7.4 原文限定 PEC 与官谱）→ 只报 warning 计数。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from beatmorph.core.contracts.legality import (
    POSITION_X_CLAMPED_KEY,
    Edit,
    EditKind,
    LegalityReport,
    Violation,
    ViolationKind,
    assert_no_position_clamp,
)
from beatmorph.core.contracts.phigros import (
    RPE_STAGE_HALF_WIDTH,
    RPE_X_GRID_DX,
    BpmPoint,
    NoteType,
    PhigrosChart,
    PhigrosNote,
    Transform,
    side_index,
)
from beatmorph.decoder.events import DecodedEvent, PairingStats, event_sort_key
from beatmorph.field.grid import FieldGrid, seconds_to_tau, tau_bin_index

#: 判定线速度是否"发生变化"的相对容差（浮点护栏，非物理常量）。
SPEED_REL_TOL: Final[float] = 1e-9

#: 跨线几何判据的声明文本（随报告落盘：数字必须有出处）。
CROSS_LINE_CRITERION: Final[str] = (
    "跨线几何：同一 tau 格内、不同判定线的 note 在**舞台系**（经 local_to_stage 含 father 嵌套）"
    "的欧氏距离 <= tolerance_bins * dx 记为一次冲突；只报告计数，不移动任何 note（plan 05 §9-2 阈值未定）"
)


@dataclass(frozen=True, slots=True)
class LegalityConfig:
    """后处理的全部自由度（默认口径一律是"未查证项只统计"）。

    Attributes:
        same_instant_limit: 同一时刻按键上限；`None`（默认）= 数值未查证，只统计分布。
        cross_line_tolerance_bins: 跨线冲突的舞台系距离容差（以 `dx` 为单位，派生量）。
        check_cross_line: 是否做跨线几何统计（`local_to_stage` 逐 note 求值，可关）。
        check_hold_line_speed: 是否统计「Hold 期间判定线速度变化」（只报告）。
    """

    same_instant_limit: int | None = None
    cross_line_tolerance_bins: float = 1.0
    check_cross_line: bool = True
    check_hold_line_speed: bool = True

    def __post_init__(self) -> None:
        if self.same_instant_limit is not None and self.same_instant_limit < 1:
            raise ValueError(f"same_instant_limit 必须 >= 1，得到 {self.same_instant_limit!r}")
        if self.cross_line_tolerance_bins < 0.0:
            raise ValueError("cross_line_tolerance_bins 必须 >= 0")


def note_key(note: PhigrosNote) -> tuple[int, float, float, int, int]:
    """重复事件的判据：`(line_id, t, position_x, side, type)` **全同**即为重复。"""
    return (
        note.line_id,
        float(note.t),
        float(note.position_x),
        side_index(note.side),
        int(note.type),
    )


def event_key(event: DecodedEvent) -> tuple[int, float, float, int, int]:
    """`DecodedEvent` 侧的同一判据（与 `note_key` 同口径，不得两套）。"""
    return (
        event.line_id,
        float(event.t_s),
        float(event.position_x),
        side_index(event.side),
        int(event.note_type),
    )


# ══════════════════════════════════════════════════════════════
# 事件层
# ══════════════════════════════════════════════════════════════


def check_events(events: Sequence[DecodedEvent]) -> LegalityReport:
    """事件层检查：反向 Hold（`endTime < startTime`）是**唯一**的硬违规。"""
    violations: list[Violation] = []
    for index, event in enumerate(events):
        if event.hold_time_s < 0.0:
            violations.append(
                Violation(
                    kind=ViolationKind.HOLD_REVERSED,
                    note_index=index,
                    detail=f"Hold 终点早于起点 {event.hold_time_s:.6g} 秒（line={event.line_id}）",
                ),
            )
    return LegalityReport(
        violations=violations,
        stats={"events_total": float(len(events)), "events_hold_reversed": float(len(violations))},
    )


def fix_events(
    events: Sequence[DecodedEvent],
) -> tuple[list[DecodedEvent], list[Edit]]:
    """丢弃**不可表示**的事件：反向 Hold 与完全重复事件；逐条留痕。

    保留顺序不变（`DecodedEvent` 列表已按 `event_sort_key` 排序），
    因此"未触发任何修复时输出与输入逐字段一致"这一契约可被断言。
    """
    edits: list[Edit] = []
    seen: set[tuple[int, float, float, int, int]] = set()
    kept: list[DecodedEvent] = []
    for index, event in enumerate(events):
        if event.hold_time_s < 0.0:
            edits.append(
                Edit(
                    kind=EditKind.DROP_HOLD_REVERSED,
                    note_index=index,
                    field="hold_time_s",
                    before=f"{event.hold_time_s:.6g}",
                    after="(丢弃)",
                    reason="Hold 的 endTime < startTime，格式 A 级语义要求区间非负",
                    resolved=ViolationKind.HOLD_REVERSED,
                ),
            )
            continue
        key = event_key(event)
        if key in seen:
            edits.append(
                Edit(
                    kind=EditKind.DROP_DUPLICATE,
                    note_index=index,
                    field="(line_id,t,position_x,side,type)",
                    before=str(key),
                    after="(丢弃)",
                    reason="点过程一次实现不应产生完全重复的事件",
                    resolved=ViolationKind.DUPLICATE_EVENT,
                ),
            )
            continue
        seen.add(key)
        kept.append(event)
    kept.sort(key=event_sort_key)
    return kept, edits


# ══════════════════════════════════════════════════════════════
# 谱面层（统计 + 未查证项的显式开关）
# ══════════════════════════════════════════════════════════════


def same_instant_groups(notes: Sequence[PhigrosNote]) -> list[list[int]]:
    """按判定时刻（秒，精确相等）分组，返回 size >= 2 的组（**下标**，确定性排序）。

    返回下标而不是 note 对象：调用方需要的是"同刻按键数的分布"，而下标是
    定位到具体 note 的唯一无歧义方式（两个 note 的字段可能完全相同）。
    """
    buckets: dict[float, list[int]] = defaultdict(list)
    for index, note in enumerate(notes):
        buckets[float(note.t)].append(index)
    return [buckets[key] for key in sorted(buckets) if len(buckets[key]) >= 2]


def _line_speed_varies(line: object, start_beats: float, end_beats: float) -> bool:
    """判定线的 `speed` 轨在 `[start_beats, end_beats)` 内是否**真的变化**。

    判据基于事件关键帧本身（而不是两端采样）：任何与区间有交、且 `start != end` 的
    关键帧都算变化。这样"中途变化后又变回原值"不会被漏检。
    """
    for layer in line.event_layers:  # type: ignore[attr-defined]
        for keyframe in layer.speed:
            if keyframe.end_time.to_beats() <= start_beats:
                continue
            if keyframe.start_time.to_beats() >= end_beats:
                continue
            start, end = keyframe.start, keyframe.end
            if isinstance(start, int | float) and isinstance(end, int | float):
                scale = max(abs(float(start)), abs(float(end)), 1.0)
                if abs(float(start) - float(end)) > SPEED_REL_TOL * scale:
                    return True
    return False


def stage_points(
    chart: PhigrosChart,
    notes: Sequence[PhigrosNote],
    bpm_points: Sequence[BpmPoint],
) -> list[tuple[float, float]]:
    """把 note 的 `positionX` 变换到**舞台系**（走契约 `local_to_stage`，含 father）。

    秒 -> 拍经 `field/` 的权威换算（红线 7）；**不得**在像素系计算（像素系依赖
    `aspectRatio`，与谱面语义无关，单位文档 §5.2）。
    """
    points: list[tuple[float, float]] = []
    cache: dict[tuple[int, float], Transform] = {}
    for note in notes:
        t_beats = float(seconds_to_tau(note.t, bpm_points))
        key = (note.line_id, t_beats)
        transform = cache.get(key)
        if transform is None:
            transform = chart.line_by_id(note.line_id).local_to_stage(t_beats, chart)
            cache[key] = transform
        points.append(transform.apply(note.position_x, 0.0))
    return points


def cross_line_conflicts(
    chart: PhigrosChart,
    *,
    tolerance_bins: float,
    dx: float,
) -> tuple[int, int]:
    """跨线几何冲突计数：返回 `(冲突数, 受检对数)`（**只报告不处置**）。

    `line_id` 越界的 note 不参与几何（它本身已是违规项，且取不到判定线位姿）——
    检查器**不得**因为另一项违规而崩掉。
    """
    notes = [note for note in chart.notes if 0 <= note.line_id < len(chart.lines)]
    if len(notes) < 2:
        return 0, 0
    bpm_points = chart.bpm_points
    points = stage_points(chart, notes, bpm_points)
    buckets: dict[int, list[int]] = defaultdict(list)
    for index, note in enumerate(notes):
        buckets[tau_bin_index(float(seconds_to_tau(note.t, bpm_points)))].append(index)
    tolerance = tolerance_bins * dx
    conflicts = 0
    pairs = 0
    for key in sorted(buckets):
        members = buckets[key]
        if len(members) < 2:
            continue
        for first in range(len(members)):
            for second in range(first + 1, len(members)):
                i, j = members[first], members[second]
                if notes[i].line_id == notes[j].line_id:
                    continue
                pairs += 1
                (x1, y1), (x2, y2) = points[i], points[j]
                if (x1 - x2) ** 2 + (y1 - y2) ** 2 <= tolerance * tolerance:
                    conflicts += 1
    return conflicts, pairs


def _range_stats(chart: PhigrosChart, notes: list[PhigrosNote], stats: dict[str, float]) -> None:
    """③ positionX 越界：**只统计**，并与契约侧计数交叉核对（口径必须唯一）。"""
    contract_stats = chart.out_of_visible_range()
    mine = sum(1 for note in notes if abs(note.position_x) > RPE_STAGE_HALF_WIDTH)
    if mine != contract_stats.count:
        raise AssertionError(
            f"越界计数不一致：本模块 {mine} vs 契约 out_of_visible_range {contract_stats.count}"
            "（同一物理量出现两套口径 = 红线 7 类漂移）",
        )
    stats["chart_out_of_range"] = float(mine)
    stats["chart_out_of_range_fraction"] = float(mine / len(notes)) if notes else 0.0
    stats["chart_max_abs_position_x"] = (
        float(max(abs(note.position_x) for note in notes)) if notes else 0.0
    )


def _same_instant_stats(
    notes: list[PhigrosNote],
    settings: LegalityConfig,
    stats: dict[str, float],
) -> list[Violation]:
    """④ 同刻按键数分布：**默认只统计**（上限数值未查证，plan 05 §9-1）。"""
    groups = same_instant_groups(notes)
    sizes = [len(group) for group in groups]
    stats["chart_same_instant_groups"] = float(len(groups))
    stats["chart_same_instant_max"] = float(max(sizes)) if sizes else float(bool(notes))
    stats["chart_same_instant_mean"] = (
        float(sum(sizes) / len(sizes)) if sizes else float(bool(notes))
    )
    violations: list[Violation] = []
    if settings.same_instant_limit is None:
        return violations
    over = 0
    for group in groups:
        if len(group) > settings.same_instant_limit:
            over += 1
            violations.append(
                Violation(
                    kind=ViolationKind.SAME_INSTANT_OVER_LIMIT,
                    note_index=group[0],
                    detail=f"t={notes[group[0]].t:.6g}s 处有 {len(group)} 个按键，"
                    f"超过配置上限 {settings.same_instant_limit}",
                ),
            )
    stats["chart_same_instant_over_limit"] = float(over)
    return violations


def _hold_stats(
    chart: PhigrosChart,
    notes: list[PhigrosNote],
    settings: LegalityConfig,
    stats: dict[str, float],
) -> None:
    """⑤⑥ Hold 统计：区间合法性由契约保证 `hold_time >= 0`（反向 Hold 在事件层拦）；
    「Hold 期间判定线速度变化」在 RPE 下**未确证是硬约束**，故只报 warning 计数。"""
    holds = [note for note in notes if note.type is NoteType.HOLD]
    stats["chart_holds"] = float(len(holds))
    stats["chart_holds_nonpositive"] = float(sum(1 for note in holds if note.hold_time <= 0.0))
    stats["chart_hold_mean_s"] = (
        float(sum(note.hold_time for note in holds) / len(holds)) if holds else 0.0
    )
    stats["chart_hold_max_s"] = float(max((note.hold_time for note in holds), default=0.0))
    if not (settings.check_hold_line_speed and holds and chart.bpm_points):
        return
    changed = 0
    for note in holds:
        line = chart.line_by_id(note.line_id)
        start = float(seconds_to_tau(note.t, chart.bpm_points))
        end = float(seconds_to_tau(note.t + note.hold_time, chart.bpm_points))
        if _line_speed_varies(line, start, end):
            changed += 1
    stats["chart_hold_line_speed_change"] = float(changed)


def _cross_line_stats(
    chart: PhigrosChart,
    notes: list[PhigrosNote],
    grid: FieldGrid | None,
    settings: LegalityConfig,
    stats: dict[str, float],
) -> None:
    """⑦ 跨线几何冲突：**只报告计数**（判据阈值未定义，plan 05 §9-2）。"""
    if not (settings.check_cross_line and len(notes) >= 2):
        return
    dx = float(grid.dx) if grid is not None else float(RPE_X_GRID_DX)
    conflicts, pairs = cross_line_conflicts(
        chart,
        tolerance_bins=settings.cross_line_tolerance_bins,
        dx=dx,
    )
    stats["chart_cross_line_conflicts"] = float(conflicts)
    stats["chart_cross_line_pairs"] = float(pairs)


def check_chart(
    chart: PhigrosChart,
    *,
    grid: FieldGrid | None = None,
    config: LegalityConfig | None = None,
) -> LegalityReport:
    """谱面层检查 + 统计（**不修改谱面**）。

    Args:
        chart: 待检查的谱面 IR。
        grid: 场网格；仅用于取 `dx`（跨线判据容差）与断言；省略时用契约默认 `dx`。
        config: 检查口径；默认 = 未查证项只统计。
    """
    settings = LegalityConfig() if config is None else config
    notes = list(chart.notes)
    violations: list[Violation] = []
    stats: dict[str, float] = {
        "chart_notes_total": float(len(notes)),
        "chart_lines_total": float(len(chart.lines)),
        POSITION_X_CLAMPED_KEY: 0.0,
    }

    # ① line_id 必须落在 [0, K)：否则写进 RPEJSON 会落到不存在的判定线上
    for index, note in enumerate(notes):
        if not 0 <= note.line_id < len(chart.lines):
            violations.append(
                Violation(
                    kind=ViolationKind.LINE_INDEX_OUT_OF_RANGE,
                    note_index=index,
                    detail=f"line_id={note.line_id} 不在 [0, {len(chart.lines)})",
                ),
            )

    # ② 完全重复事件
    seen: set[tuple[int, float, float, int, int]] = set()
    duplicates = 0
    for index, note in enumerate(notes):
        key = note_key(note)
        if key in seen:
            duplicates += 1
            violations.append(
                Violation(
                    kind=ViolationKind.DUPLICATE_EVENT,
                    note_index=index,
                    detail=f"与前面的事件完全相同：{key}",
                ),
            )
        seen.add(key)
    stats["chart_duplicates"] = float(duplicates)

    # ③ positionX 越界：只统计（越界是可见边界而非合法值域，红线 3）
    _range_stats(chart, notes, stats)

    # ④ 同刻按键数分布（上限数值未查证 -> 默认只统计）
    violations.extend(_same_instant_stats(notes, settings, stats))

    # ⑤⑥ Hold 统计（含「Hold 期间线速度变化」的 warning 计数）
    _hold_stats(chart, notes, settings, stats)

    # ⑦ 跨线几何冲突（阈值未定 -> 只报告计数）
    _cross_line_stats(chart, notes, grid, settings, stats)

    report = LegalityReport(
        violations=violations,
        stats=stats,
        criterion=CROSS_LINE_CRITERION,
    )
    assert_no_position_clamp(report)
    return report


def fix_chart(chart: PhigrosChart) -> tuple[PhigrosChart, list[Edit]]:
    """丢弃 line_id 越界与完全重复的 note；逐条留痕（**绝不改 `position_x`**）。"""
    edits: list[Edit] = []
    seen: set[tuple[int, float, float, int, int]] = set()
    kept: list[PhigrosNote] = []
    for index, note in enumerate(chart.notes):
        if not 0 <= note.line_id < len(chart.lines):
            edits.append(
                Edit(
                    kind=EditKind.DROP_LINE_OUT_OF_RANGE,
                    note_index=index,
                    field="line_id",
                    before=str(note.line_id),
                    after="(丢弃)",
                    reason=f"line_id 不在 [0, {len(chart.lines)})：目标线上不存在，无法导出",
                    resolved=ViolationKind.LINE_INDEX_OUT_OF_RANGE,
                ),
            )
            continue
        key = note_key(note)
        if key in seen:
            edits.append(
                Edit(
                    kind=EditKind.DROP_DUPLICATE,
                    note_index=index,
                    field="(line_id,t,position_x,side,type)",
                    before=str(key),
                    after="(丢弃)",
                    reason="点过程一次实现不应产生完全重复的事件",
                    resolved=ViolationKind.DUPLICATE_EVENT,
                ),
            )
            continue
        seen.add(key)
        kept.append(note)
    if len(kept) == len(chart.notes):
        return chart, []
    return chart.model_copy(update={"notes": kept}), edits


@dataclass(frozen=True, slots=True)
class PostprocessResult:
    """后处理结果。

    Attributes:
        chart: **修复后**的谱面（`report.violations` 为空时即可导出）。
        report: 描述**返回的这张谱面**的报告——修复动作在 `edits` 里逐条留痕
            （每条带 `resolved` 指向它消解的违规种类），因此 `violations` 为空
            与"可导出"始终同义（红线 6 的唯一判据）。
        findings: **修复前**的发现（诊断/审计用）；有违规时它的 `violations` 非空。
    """

    chart: PhigrosChart
    report: LegalityReport
    findings: LegalityReport


def merge_reports(*reports: LegalityReport) -> LegalityReport:
    """合并报告：`violations`/`edits` 顺序拼接，`stats` 后者覆盖前者。"""
    violations: list[Violation] = []
    edits: list[Edit] = []
    stats: dict[str, float] = {}
    criterion = ""
    for report in reports:
        violations.extend(report.violations)
        edits.extend(report.edits)
        stats.update(report.stats)
        criterion = report.criterion or criterion
    merged = LegalityReport(
        violations=violations,
        edits=edits,
        stats=stats,
        criterion=criterion,
    )
    assert_no_position_clamp(merged)
    return merged


def _resolved_report(
    findings: LegalityReport,
    after: LegalityReport,
    edits: list[Edit],
) -> LegalityReport:
    """构造"针对修复后状态"的报告：违规取后置、统计取两者并集、留痕取修复动作。

    统计取并集而不是后置覆盖：计数类统计（越界数、重复数、冲突数）描述的是**输入**，
    它们在被修复后仍然是有价值的事实；而"违反红线"这件事由 `edits[].resolved` 承载。
    """
    report = LegalityReport(
        violations=list(after.violations),
        edits=edits,
        stats={**findings.stats, **after.stats},
        criterion=after.criterion or findings.criterion,
    )
    assert_no_position_clamp(report)
    return report


def postprocess_chart(
    chart: PhigrosChart,
    *,
    grid: FieldGrid | None = None,
    config: LegalityConfig | None = None,
) -> PostprocessResult:
    """谱面层的完整后处理：检查 -> 修复（留痕）-> **复检**。

    复检是硬要求：`fix_chart` 之后剩下的违规项必须为空，否则说明修复动作本身
    没有覆盖某一类违规（那会让"violations 为空 == 可导出"的判据失效）。
    """
    settings = LegalityConfig() if config is None else config
    findings = check_chart(chart, grid=grid, config=settings)
    if not findings.violations:
        return PostprocessResult(chart=chart, report=findings, findings=findings)
    fixed, edits = fix_chart(chart)
    after = check_chart(fixed, grid=grid, config=settings)
    if after.violations:
        raise AssertionError(
            "fix_chart 之后仍有违规项——修复动作与检查项不一致："
            + "；".join(f"[{v.kind}] {v.detail}" for v in after.violations),
        )
    return PostprocessResult(
        chart=fixed,
        report=_resolved_report(findings, after, edits),
        findings=findings,
    )


def postprocess_events(
    events: Sequence[DecodedEvent],
    *,
    pairing: PairingStats | None = None,
) -> tuple[list[DecodedEvent], LegalityReport]:
    """事件层后处理：检查 -> 修复（留痕）-> 复检，并把配对统计并入报告。

    返回的报告针对**修复后的事件列表**（`violations` 为空），修复动作在 `edits` 里；
    修复前的发现在 `check_events(events)`（需要审计时单独调用）。
    """
    findings = check_events(events)
    fixed, edits = fix_events(events)
    after = check_events(fixed)
    if after.violations:
        raise AssertionError("fix_events 之后仍有违规项（修复与检查不一致）")
    report = _resolved_report(findings, after, edits)
    if pairing is not None:
        report = report.with_stats(pairing.as_stats)
    return fixed, report


__all__ = [
    "CROSS_LINE_CRITERION",
    "SPEED_REL_TOL",
    "LegalityConfig",
    "PostprocessResult",
    "check_chart",
    "check_events",
    "cross_line_conflicts",
    "event_key",
    "fix_chart",
    "fix_events",
    "merge_reports",
    "note_key",
    "postprocess_chart",
    "postprocess_events",
    "same_instant_groups",
    "stage_points",
]
