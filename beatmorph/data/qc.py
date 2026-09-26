"""谱面质检（plan 02 §3.5 / M7）：schema / 单位 / 分布 三层。

| 层级 | 规则 | 处置 |
|------|------|------|
| **schema** | `judgeLineList` 非空；note `type_raw ∈ {1,2,3,4}`；Hold `endTime >= startTime`；
`father` 索引合法且无环 | 违约 **拒收 → 隔离区** |
| **单位** | 全部时间在**秒域**；`|position_x| <= RPE_STAGE_HALF_WIDTH`；
`RPE_STAGE_WIDTH` 派生一致 | 越界**只计入 `out_of_visible_range`**（只统计不钳位，红线 3） |
| **分布** | 线数、note 数、type/above 分布、每线 note 数（含熵）、时间跨度、BPM 区间 | 与调研 §7 基线比对，**离群记为离群而不删样本** |

**越界只统计不钳位**（CLAUDE.md 红线 3 / units 文档 §7.5）：`|positionX| > 675` 是格式允许的
（无任何来源钳位它），钳位等于改变落点分布 = 改 AI 逻辑。`out_of_visible_range` 同时是
「我方解析单位错」的**哨兵**：若它突然飙到接近 100%，先怀疑换算而不是数据。

**与 plan §3.5 的一处偏离（有证据）**：原文把「每线有 notes」列为 schema 硬规则，但实测
「只有 ~58% 的判定线真的带 note」（调研 §7.2：60360 号谱 25 条线全部 note 集中在 1 条上）。
故本实现把「单条线没有 note」降为 **warning**，仅当**全谱 0 个 note** 时才判 schema 违约。
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import pairwise
from typing import Any

from beatmorph.core.contracts import (
    RPE_STAGE_HALF_HEIGHT,
    RPE_STAGE_HALF_WIDTH,
    RPE_STAGE_HEIGHT,
    RPE_STAGE_WIDTH,
    RPE_X_GRID_BINS,
    RPE_X_GRID_DX,
    ChartFormat,
    NoteType,
    PhigrosChart,
)
from beatmorph.core.logging import get_logger

logger = get_logger(__name__)

#: 合法的 `above` 取值（实测 {0, 1, 2}；语义是「== 1 为正面，其余为背面」）。
KNOWN_ABOVE_VALUES: frozenset[int] = frozenset({0, 1, 2})

#: 合法的 `type_raw` 取值（RPE：1 Tap / 2 Hold / 3 Flick / 4 Drag）。
KNOWN_TYPE_RAW_VALUES: frozenset[int] = frozenset(int(member) for member in NoteType)


class QuarantineStage(StrEnum):
    """样本被拒收的阶段（**拒收必须显式记账**，plan 02 §偏离 2）。"""

    SNIFF = "sniff"
    PARSE = "parse"
    QC = "qc"
    PACKAGE = "package"


@dataclass(frozen=True)
class QuarantineRecord:
    """隔离区记录：一条被拒收样本的**可追溯**理由。"""

    chart_id: int | None
    stage: QuarantineStage
    fmt: ChartFormat
    reasons: list[str] = field(default_factory=list)
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        """清单行（`format` 用字符串，便于 JSONL 落盘）。"""
        return {
            "chart_id": self.chart_id,
            "quarantine_stage": str(self.stage),
            "format": str(self.fmt),
            "reasons": list(self.reasons),
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class DistributionStats:
    """分布级统计（plan §3.5 第三层 / M7）。

    `max_simultaneous_onsets` 与 `min_same_line_same_time_gap_x` 是 plan §9 的 Q-16
    与「共格碰撞率」的输入：**同刻**口径取「判定时刻浮点完全相等」（同一 beat 经同一
    `BPMList` 积分必然得到同一浮点值），不做时间量化——量化阈值本身尚未裁定。
    """

    n_lines: int
    n_notes: int
    type_counts: dict[int, int]
    above_counts: dict[int, int]
    notes_per_line: list[int]
    lines_with_notes: int
    busiest_share: float
    notes_per_line_entropy: float
    notes_per_line_entropy_norm: float
    time_span_s: float
    bpm_min: float
    bpm_max: float
    back_fraction: float
    tap_fraction: float
    max_simultaneous_onsets: int
    min_same_line_same_time_gap_x: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """摊平成 JSON 友好字典（`type_counts` 的键转字符串）。"""
        return {
            "n_lines": self.n_lines,
            "n_notes": self.n_notes,
            "type_counts": {str(key): value for key, value in self.type_counts.items()},
            "above_counts": {str(key): value for key, value in self.above_counts.items()},
            "notes_per_line": list(self.notes_per_line),
            "lines_with_notes": self.lines_with_notes,
            "busiest_share": self.busiest_share,
            "notes_per_line_entropy": self.notes_per_line_entropy,
            "notes_per_line_entropy_norm": self.notes_per_line_entropy_norm,
            "time_span_s": self.time_span_s,
            "bpm_min": self.bpm_min,
            "bpm_max": self.bpm_max,
            "back_fraction": self.back_fraction,
            "tap_fraction": self.tap_fraction,
            "max_simultaneous_onsets": self.max_simultaneous_onsets,
            "min_same_line_same_time_gap_x": self.min_same_line_same_time_gap_x,
        }


@dataclass(frozen=True)
class QcReport:
    """一份谱面的质检报告（plan 02 §3.5 的字段 + 分布子结构）。

    `passed` 只由 schema + 单位两层的**硬错误**决定；分布离群只记入 `warnings`
    （plan §7-R3：「不预设阈值，先做全库统计再定」）。
    """

    chart_id: int | None = None
    passed: bool = False
    n_lines: int = 0
    n_notes: int = 0
    fmt: ChartFormat = ChartFormat.UNKNOWN
    out_of_visible_range: int = 0
    out_of_audio_window: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    distribution: DistributionStats | None = None

    def to_dict(self) -> dict[str, Any]:
        """清单行（嵌套 `distribution`）。"""
        return {
            "chart_id": self.chart_id,
            "qc_passed": self.passed,
            "n_lines": self.n_lines,
            "n_notes": self.n_notes,
            "format": str(self.fmt),
            "out_of_visible_range": self.out_of_visible_range,
            "out_of_audio_window": self.out_of_audio_window,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "distribution": None if self.distribution is None else self.distribution.to_dict(),
        }


# ══════════════════════════════════════════════════════════════
# 单位层（G4 数据侧落点：派生一致性断言）
# ══════════════════════════════════════════════════════════════


def unit_contract_violations() -> list[str]:
    """断言舞台几何与 x 网格是**派生一致**的（红线 7 / plan §6 M7 单位层）。

    只读契约常量、不写字面量：任何一条不成立都说明契约层被改坏了 → 计入 `errors`。
    """
    problems: list[str] = []
    if not math.isclose(RPE_STAGE_HALF_WIDTH * 2.0, RPE_STAGE_WIDTH):
        problems.append(f"RPE_STAGE_WIDTH != 2 * RPE_STAGE_HALF_WIDTH（{RPE_STAGE_WIDTH}）")
    if not math.isclose(RPE_STAGE_HALF_HEIGHT * 2.0, RPE_STAGE_HEIGHT):
        problems.append(f"RPE_STAGE_HEIGHT != 2 * RPE_STAGE_HALF_HEIGHT（{RPE_STAGE_HEIGHT}）")
    if not math.isclose(RPE_X_GRID_DX * RPE_X_GRID_BINS, RPE_STAGE_WIDTH):
        problems.append(f"RPE_X_GRID_DX * RPE_X_GRID_BINS != RPE_STAGE_WIDTH（{RPE_X_GRID_DX}）")
    return problems


# ══════════════════════════════════════════════════════════════
# 分布层
# ══════════════════════════════════════════════════════════════


def _entropy(counts: list[int]) -> tuple[float, float]:
    """每线 note 数的香农熵与归一化熵（`H / ln K`；K <= 1 时归一化熵为 0）。"""
    total = sum(counts)
    if total <= 0:
        return 0.0, 0.0
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        p = count / total
        entropy -= p * math.log(p)
    if len(counts) <= 1:
        return entropy, 0.0
    return entropy, entropy / math.log(len(counts))


def min_same_line_same_time_gap_x(chart: PhigrosChart) -> float | None:
    """同线 + 同刻 + 同侧的最小 `|ΔpositionX|`（共格碰撞率统计的输入，plan §9-5）。

    没有「同刻多 note」的样本时返回 None（不是 0.0：0.0 是「两个 note 完全重合」，
    语义不同，不可混）。
    """
    groups: dict[tuple[int, float, int], list[float]] = {}
    for note in chart.notes:
        groups.setdefault((note.line_id, note.t, int(note.side)), []).append(note.position_x)
    best: float | None = None
    for xs in groups.values():
        if len(xs) < 2:
            continue
        ordered = sorted(xs)
        for left, right in pairwise(ordered):
            gap = right - left
            if gap == 0.0:
                # 两个 note 完全重合：共格碰撞的最坏情形，直接返回 0.0。
                return 0.0
            if best is None or gap < best:
                best = gap
    return best


def max_simultaneous_onsets(chart: PhigrosChart) -> int:
    """同一判定时刻（跨所有判定线）的最大 note 数（Q-16「同刻按键上限」的输入）。"""
    if not chart.notes:
        return 0
    return max(Counter(note.t for note in chart.notes).values())


def distribution_stats(chart: PhigrosChart) -> DistributionStats:
    """计算分布级统计（**只读**，不修改任何字段）。"""
    notes_per_line = chart.notes_per_line()
    counts = Counter(int(note.type_raw) for note in chart.notes)
    above_counts = Counter(note.above_raw for note in chart.notes)
    total = len(chart.notes)
    entropy, entropy_norm = _entropy(notes_per_line)
    busiest = max(notes_per_line, default=0)
    bpms = [point.bpm for point in chart.bpm_points]
    return DistributionStats(
        n_lines=len(chart.lines),
        n_notes=total,
        type_counts=dict(counts),
        above_counts=dict(above_counts),
        notes_per_line=notes_per_line,
        lines_with_notes=sum(1 for count in notes_per_line if count > 0),
        busiest_share=(busiest / total) if total else 0.0,
        notes_per_line_entropy=entropy,
        notes_per_line_entropy_norm=entropy_norm,
        time_span_s=chart.duration_s(),
        # bpm_points 由契约保证非空；此处仅防御「绕过契约直接构造」的调用。
        bpm_min=min(bpms) if bpms else 0.0,
        bpm_max=max(bpms) if bpms else 0.0,
        back_fraction=(total - above_counts.get(1, 0)) / total if total else 0.0,
        tap_fraction=counts.get(int(NoteType.TAP), 0) / total if total else 0.0,
        max_simultaneous_onsets=max_simultaneous_onsets(chart),
        min_same_line_same_time_gap_x=min_same_line_same_time_gap_x(chart),
    )


# ══════════════════════════════════════════════════════════════
# schema 层（拆成两个 helper：把分支数压到 pylint 阈值以下，也便于单测定位）
# ══════════════════════════════════════════════════════════════


def _check_notes(chart: PhigrosChart, errors: list[str], warnings: list[str]) -> None:
    """逐 note 的 schema / 已知取值检查（**只读**）。"""
    for note in chart.notes:
        where = f"line {note.line_id} t={note.t}"
        if note.type_raw not in KNOWN_TYPE_RAW_VALUES:
            errors.append(f"line {note.line_id} note type_raw={note.type_raw} 不在 (1,2,3,4)")
        if note.is_hold() and note.hold_time <= 0.0:
            warnings.append(f"{where} 的 Hold 时长为 0（退化 Hold）")
        if not note.is_hold() and note.hold_time != 0.0:
            errors.append(f"{where} 非 Hold 却有 hold_time={note.hold_time}")
        if note.above_raw not in KNOWN_ABOVE_VALUES:
            warnings.append(
                f"line {note.line_id} note above_raw={note.above_raw} 不在实测取值 "
                f"{sorted(KNOWN_ABOVE_VALUES)}（按背面处理）",
            )
        if note.t > chart.meta.chart_time_s > 0.0:
            # 只提示：chartTime 是「写谱时长」，与音频时长不是同一口径（Q11/D10 未裁定）。
            warnings.append(f"{where} 超出 chartTime")


def _check_lines(chart: PhigrosChart, errors: list[str], warnings: list[str]) -> None:
    """逐判定线的 father 链、`bpmfactor`、`numOfNotes` 冗余校验（**只读**）。"""
    notes_per_line = chart.notes_per_line()
    non_hold_per_line = [0] * len(chart.lines)
    for note in chart.notes:
        if not note.is_hold():
            non_hold_per_line[note.line_id] += 1
    for index, count in enumerate(notes_per_line):
        if count == 0:
            warnings.append(f"line {index} 没有 note（实测 ~42% 的判定线不带 note，属正常）")

    for line in chart.lines:
        try:
            line.ancestry(chart)
        except ValueError as exc:
            errors.append(f"line {line.line_id} father 链非法：{exc}")
        if line.bpm_factor != 1.0:
            warnings.append(
                f"line {line.line_id} bpm_factor={line.bpm_factor} != 1.0"
                "（prpr 未实现该字段，存疑 D4；我方同样不参与换算 → 与目标运行时一致）",
            )
        expected_notes = notes_per_line[line.line_id]
        non_hold = non_hold_per_line[line.line_id]
        if line.num_of_notes_raw not in (0, expected_notes, non_hold):
            warnings.append(
                f"line {line.line_id} numOfNotes={line.num_of_notes_raw} 与实解析"
                f"（全部 {expected_notes} / 非 Hold {non_hold}）都不符",
            )


def count_out_of_audio_window(chart: PhigrosChart, audio_duration_s: float | None) -> int:
    """判定时刻或 Hold 结束时刻超出音频时长的事件数（**只统计不裁剪**）。"""
    if audio_duration_s is None or audio_duration_s <= 0.0:
        return 0
    return sum(
        1
        for note in chart.notes
        if note.t > audio_duration_s or note.t + note.hold_time > audio_duration_s
    )


# ══════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════


def quality_check(
    chart: PhigrosChart,
    audio_duration_s: float | None = None,
    *,
    chart_id: int | None = None,
    fmt: ChartFormat | None = None,
) -> QcReport:
    """三层质检（plan 02 §3.5）。

    Args:
        chart: 已解析的 IR。
        audio_duration_s: 音频时长（秒）；None = 不做音频窗口统计。
        chart_id: 谱面 id（留痕）。
        fmt: 覆盖格式标签（默认取 `chart.source.format`）。

    Returns:
        :class:`QcReport`；`passed=False` 的样本应进入隔离区（**不删除原始数据**）。
    """
    errors: list[str] = []
    warnings: list[str] = []

    # ── schema 层 ──
    if not chart.lines:
        errors.append("judgeLineList 为空（schema 违约）")
    if not chart.notes:
        errors.append("谱面 0 个 note（schema 违约：无法构成样本）")
    if not chart.bpm_points:
        errors.append("bpm_points 为空（秒↔拍换算失去依据，红线 7）")
    elif chart.bpm_points[0].time_beats != 0.0:
        # 实测（2026-09-27）：BPMList 首段起点 > 0 时，格式层 beat↔秒（以首段为原点，
        # 与 Phira 官方参考实现 beat2sec 一致）与 field/ 的 τ↔秒（把 τ=0 作为谱面时间
        # 原点并在前面外推一段）会**相差一个常量**——同一个谱面就有了两条时间轴。
        # RPE 规范首段是 [0,0,1]，该情形只可能来自畸形谱面；按「拒收进隔离区而非静默
        # 二选一」处理（plan 02 §9-11 / plan 03 §9-13 的临时裁定，待 RFC 统一）。
        errors.append(
            f"BPMList 首段起于 {chart.bpm_points[0].time_beats} 拍而非 0 拍（schema 违约）："
            "秒↔拍的两条实现（格式层 / field 的 τ）在该情形下相差一个常量偏移",
        )
    _check_notes(chart, errors, warnings)
    _check_lines(chart, errors, warnings)

    # ── 单位层 ──
    errors.extend(unit_contract_violations())
    out_of_range = chart.out_of_visible_range()

    report = QcReport(
        chart_id=chart_id,
        passed=not errors,
        n_lines=len(chart.lines),
        n_notes=len(chart.notes),
        fmt=fmt if fmt is not None else chart.source.format,
        out_of_visible_range=out_of_range.count,
        out_of_audio_window=count_out_of_audio_window(chart, audio_duration_s),
        errors=errors,
        warnings=warnings,
        distribution=distribution_stats(chart),
    )
    if not report.passed:
        logger.warning("质检未通过 chart_id=%s：%s", chart_id, "; ".join(errors[:3]))
    return report


__all__ = [
    "KNOWN_ABOVE_VALUES",
    "KNOWN_TYPE_RAW_VALUES",
    "DistributionStats",
    "QcReport",
    "QuarantineRecord",
    "QuarantineStage",
    "count_out_of_audio_window",
    "distribution_stats",
    "max_simultaneous_onsets",
    "min_same_line_same_time_gap_x",
    "quality_check",
    "unit_contract_violations",
]
