"""RPEJSON **写路径**（Plan 05 §4.4）——`PhigrosChart` IR -> RPEJSON 文件。

**许可证纪律**：与读路径同口径（`beatmorph/data/parsers/rpejson.py` 的模块 docstring）：
prpr（GPL-3.0）/ phichain（LGPL-3.0）只作为**行为规范**阅读，本文件不移植任何源码；
字段名、默认值、单位全部取自 `docs/knowledges/phigros-format.md`（A 级：Phira 官方文档）。

三个硬口径：

1. **秒 -> beat 三元组经 `field/` 的权威换算**（红线 7 / RFC-0029 §7-8）：先
   `tau = field.seconds_to_tau(t, bpm_points)`，再量化到 `1 / SUBDIVISIONS_PER_BEAT` 拍。
   **writer 内不得再写一套 BPM 分段积分**。量化误差上界 = 半个格 = `TAU_GRID_DT / 2` 拍
   （见 `beat_from_tau`）——这个步长正是场的时间基本格，因此写入不会比模型的分辨率更粗。
2. **映射一律走契约函数**：`type_raw`（RPE 数字，`int(NoteType)`）、
   `above_raw`（`above_from_side` 的规范代表值）、`Beat` 三元组；**不得**在 writer 内
   重写映射（`note_type_from_rpe(2) is HOLD` 而官谱 2 是 DRAG，混用会静默错位）。
3. **导出前门禁**（红线 6）：`report.violations` 非空即拒绝写出（`IllegalChartError`）；
   `require_legal=False` 必须显式声明，供写侧单元测试与"把任意 IR 落盘"的工具使用。

**已知的信息损失（读路径侧，不在本文件修复）**：`META.id` / `META.illustration` /
根级 `judgeLineGroup` / `multiLineString` 等字段**不在 IR 中**（`ChartMeta` 未承载），
因此"读入 -> 写出"会丢掉它们；这是读路径的记录口径问题，已记入 plan 05 §9。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, TypeAlias

from beatmorph.core.contracts.legality import LegalityReport
from beatmorph.core.contracts.phigros import (
    RPE_EXTENDED_TRACKS,
    RPE_NORMAL_TRACKS,
    RPE_TRACK_FIELDS,
    SUBDIVISIONS_PER_BEAT,
    Beat,
    EventKeyframe,
    EventValue,
    JudgeLine,
    PhigrosChart,
    PhigrosNote,
)
from beatmorph.field.grid import seconds_to_tau

#: RPE 普通轨字段名 <-> IR 字段名（**由契约常量派生**，禁止在 writer 里另写一份）。
_NORMAL_TRACK_FIELDS: tuple[tuple[str, str], ...] = tuple(
    zip(RPE_NORMAL_TRACKS, RPE_TRACK_FIELDS, strict=True),
)
#: RPE 扩展轨字段名 <-> IR 字段名。
_EXTENDED_TRACK_FIELDS: tuple[tuple[str, str], ...] = tuple(
    zip(
        RPE_EXTENDED_TRACKS,
        ("scale_x", "scale_y", "color", "text", "gif", "incline"),
        strict=True,
    ),
)
#: `speedEvents` 无贝塞尔字段（格式事实）；其余普通轨写出贝塞尔三件套。
_TRACKS_WITHOUT_BEZIER: frozenset[str] = frozenset({"speed"})
#: RPE 合法的 note type 取值（写出解析器读不回来的数字是无意义的：结构性守卫）。
_RPE_NOTE_TYPES: frozenset[int] = frozenset(int(value) for value in (1, 2, 3, 4))

#: 事件取值写进 JSON 后的形态（数值 / 文本 / RGB 三元组）。
JsonValue: TypeAlias = float | str | list[int]


class IllegalChartError(ValueError):
    """谱面不合法（`violations` 非空）或含结构性不可写字段 —— **拒绝写出**（红线 6）。"""


def beat_from_tau(
    tau: float,
    *,
    denominator: int = SUBDIVISIONS_PER_BEAT,
) -> Beat:
    """τ（拍）-> RPE beat 三元组 `[i, n, d]`，量化到 `1 / denominator` 拍。

    量化是**唯一的**精度损失点，误差上界 = `1 / (2 * denominator)` 拍 = `TAU_GRID_DT / 2`
    （恰好半个时间基本格）。分母固定取 `SUBDIVISIONS_PER_BEAT`：τ 网格的基本格就是
    `1 / SUBDIVISIONS_PER_BEAT` 拍，取更细的分母只会产生**网格之外**的假精度。
    """
    if denominator < 1:
        raise ValueError(f"denominator 必须 >= 1，得到 {denominator!r}")
    total = round(float(tau) * denominator)
    whole, numerator = divmod(total, denominator)
    return Beat(i=whole, n=numerator, d=denominator)


def beat_to_list(beat: Beat) -> list[int]:
    """beat 三元组 -> JSON 数组 `[i, n, d]`。"""
    return [int(beat.i), int(beat.n), int(beat.d)]


def value_to_json(value: EventValue) -> JsonValue:
    """事件取值 -> JSON（数值 / 文本 / RGB 三元组）。"""
    if isinstance(value, tuple):
        return [int(component) for component in value]
    if isinstance(value, str):
        return value
    return float(value)


def keyframe_to_rpe(keyframe: EventKeyframe, *, track: str) -> dict[str, Any]:
    """一个事件关键帧 -> RPE 对象（`speedEvents` 不写贝塞尔字段）。"""
    payload: dict[str, Any] = {
        "startTime": beat_to_list(keyframe.start_time),
        "endTime": beat_to_list(keyframe.end_time),
        "start": value_to_json(keyframe.start),
        "end": value_to_json(keyframe.end),
        "easingType": int(keyframe.easing_type),
    }
    if track not in _TRACKS_WITHOUT_BEZIER:
        payload["bezier"] = int(bool(keyframe.bezier))
        payload["bezierPoints"] = [float(point) for point in keyframe.bezier_points]
    payload["easingLeft"] = float(keyframe.easing_left)
    payload["easingRight"] = float(keyframe.easing_right)
    payload["linkgroup"] = int(keyframe.link_group)
    return payload


def note_to_rpe(note: PhigrosNote, bpm_points: Sequence[Any]) -> dict[str, Any]:
    """一个 note -> RPE 对象（`*_raw` 原样写回；秒 -> 拍经 `field/`）。

    Raises:
        IllegalChartError: `type_raw` 不是 RPE 合法值（写出后解析器读不回来）。
    """
    if int(note.type_raw) not in _RPE_NOTE_TYPES:
        raise IllegalChartError(
            f"note.type_raw={note.type_raw!r} 不是 RPE 合法取值 "
            f"{sorted(_RPE_NOTE_TYPES)}（结构性不可写）",
        )
    start_tau = float(seconds_to_tau(note.t, bpm_points))
    end_tau = float(seconds_to_tau(note.t + note.hold_time, bpm_points))
    return {
        "type": int(note.type_raw),
        "startTime": beat_to_list(beat_from_tau(start_tau)),
        "endTime": beat_to_list(beat_from_tau(end_tau)),
        "positionX": float(note.position_x),
        "above": int(note.above_raw),
        "isFake": int(note.is_fake_raw),
        "size": float(note.size),
        "speed": float(note.speed),
        "alpha": int(note.alpha),
        "visibleTime": float(note.visible_time),
        "yOffset": float(note.y_offset),
    }


def line_to_rpe(
    line: JudgeLine, notes: Sequence[PhigrosNote], bpm_points: Sequence[Any]
) -> dict[str, Any]:
    """一条判定线 -> RPE 对象（`notes` 为该线名下的音符，**由调用方分组**）。

    note 写在**所属判定线的 `notes` 数组内**（RPE 用单数组 + `above`；`notesAbove` /
    `notesBelow` 是官谱的双数组表示，plan 05 §4.4-4）。
    """
    layers: list[dict[str, Any]] = []
    for layer in line.event_layers:
        payload: dict[str, Any] = {}
        for rpe_name, field_name in _NORMAL_TRACK_FIELDS:
            keyframes = getattr(layer, field_name)
            if keyframes:
                payload[rpe_name] = [
                    keyframe_to_rpe(keyframe, track=field_name) for keyframe in keyframes
                ]
        layers.append(payload)
    extended: dict[str, Any] = {}
    for rpe_name, field_name in _EXTENDED_TRACK_FIELDS:
        keyframes = getattr(line.extended, field_name)
        if keyframes:
            extended[rpe_name] = [
                keyframe_to_rpe(keyframe, track=field_name) for keyframe in keyframes
            ]
    payload_line: dict[str, Any] = {
        "Group": int(line.group),
        "Name": line.name,
        "Texture": line.texture,
        "anchor": [float(line.anchor[0]), float(line.anchor[1])],
        "father": int(line.father),
        "isCover": int(line.is_cover),
        "rotateWithFather": bool(line.rotate_with_father),
        "zOrder": int(line.z_order),
        "bpmfactor": float(line.bpm_factor),
        "numOfNotes": len(notes),
        "eventLayers": layers,
        "extended": extended,
        "notes": [note_to_rpe(note, bpm_points) for note in notes],
    }
    if line.attach_ui:
        payload_line["attachUI"] = str(line.attach_ui)
    return payload_line


def _offset_json(offset_ms: float) -> int | float:
    """`META.offset` 为**毫秒整数**（A 级字段语义）；非整值时如实写浮点。"""
    rounded = round(float(offset_ms))
    return int(rounded) if float(rounded) == float(offset_ms) else float(offset_ms)


def chart_to_rpe_root(chart: PhigrosChart) -> dict[str, Any]:
    """`PhigrosChart` -> RPEJSON 根对象（纯数据，不落盘；键顺序固定 => 字节稳定）。"""
    if not chart.bpm_points:
        raise IllegalChartError("BPMList 不得为空（IR 契约要求至少一段 BPM）")
    if not chart.lines:
        raise IllegalChartError("judgeLineList 不得为空")
    grouped: list[list[PhigrosNote]] = [[] for _ in chart.lines]
    for note in chart.sorted_notes():
        if not 0 <= note.line_id < len(chart.lines):
            raise IllegalChartError(
                f"note.line_id={note.line_id} 越界（K={len(chart.lines)}）：无法确定写入哪条判定线",
            )
        grouped[note.line_id].append(note)
    return {
        "BPMList": [
            {
                "bpm": float(point.bpm),
                "startTime": beat_to_list(beat_from_tau(point.time_beats)),
            }
            for point in chart.bpm_points
        ],
        "META": {
            "RPEVersion": int(chart.meta.rpe_version),
            "background": chart.meta.background,
            "charter": chart.meta.charter,
            "composer": chart.meta.composer,
            "level": chart.meta.level_text,
            "name": chart.meta.name,
            "offset": _offset_json(chart.meta.offset_ms),
            "song": chart.meta.song,
        },
        "chartTime": float(chart.meta.chart_time_s),
        "judgeLineList": [
            line_to_rpe(line, grouped[line.line_id], chart.bpm_points) for line in chart.lines
        ],
    }


def rpejson_text(chart: PhigrosChart) -> str:
    """`PhigrosChart` -> RPEJSON 文本（缩进 2、`ensure_ascii=False`、**LF 结尾**）。"""
    root = chart_to_rpe_root(chart)
    return json.dumps(root, ensure_ascii=False, indent=2, sort_keys=False) + "\n"


def write_rpejson(
    chart: PhigrosChart,
    *,
    report: LegalityReport | None = None,
    require_legal: bool = True,
) -> bytes:
    """`PhigrosChart` -> RPEJSON 字节（导出前门禁见模块 docstring 第 3 条）。

    Args:
        chart: 谱面 IR。
        report: 合法性报告（`require_legal=True` 时**必须**提供）。
        require_legal: 是否执行红线 6 门禁；`False` 仅供写侧测试与工具使用。

    Raises:
        IllegalChartError: `report.violations` 非空，或谱面含结构性不可写字段。
        ValueError: `require_legal=True` 但未提供报告（不允许"忘了校验"这种路径）。
    """
    if require_legal:
        if report is None:
            raise ValueError(
                "require_legal=True 时必须提供 LegalityReport（红线 6：合法性校验为空违规项"
                "才允许导出；若确实要跳过校验，请显式传 require_legal=False）",
            )
        if not report.is_legal:
            summary = "；".join(f"[{item.kind}] {item.detail}" for item in report.violations)
            raise IllegalChartError(f"谱面有 {len(report.violations)} 项违规，拒绝写出：{summary}")
    return rpejson_text(chart).encode("utf-8")


def dump_rpejson(
    chart: PhigrosChart,
    path: str | Path,
    *,
    report: LegalityReport | None = None,
    require_legal: bool = True,
) -> Path:
    """把 RPEJSON 写到 `path`（返回写入路径；门禁同 `write_rpejson`）。"""
    data = write_rpejson(chart, report=report, require_legal=require_legal)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # 显式 binary 写 + LF：避免 Windows 下 newline 转换把字节稳定性破坏掉
    target.write_bytes(data)
    return target


def iter_note_groups(chart: PhigrosChart) -> Iterable[list[PhigrosNote]]:
    """按 `judgeLineList` 顺序逐线给出 note 分组（诊断/报告用）。"""
    grouped: list[list[PhigrosNote]] = [[] for _ in chart.lines]
    for note in chart.sorted_notes():
        if 0 <= note.line_id < len(chart.lines):
            grouped[note.line_id].append(note)
    return grouped


__all__ = [
    "IllegalChartError",
    "beat_from_tau",
    "beat_to_list",
    "chart_to_rpe_root",
    "dump_rpejson",
    "iter_note_groups",
    "keyframe_to_rpe",
    "line_to_rpe",
    "note_to_rpe",
    "rpejson_text",
    "value_to_json",
    "write_rpejson",
]
