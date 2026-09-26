"""RPEJSON → `PhigrosChart` 契约（plan 02 §3.4 / M5、M6）——⭐ **独立实现**。

**许可证纪律**：prpr（GPL-3.0）与 phichain（LGPL-3.0）只作为**行为规范**阅读，本文件
不移植任何源码（CLAUDE.md 红线 5 附注、plan 02 §4「解析器独立实现」）。所有字段名、
默认值、单位均取自 `docs/knowledges/phigros-format.md` 与 `phigros-units-and-geometry.md`
（A 级：prpr 源码 / Phira 官方文档；冲突时以 units 文档为准）。

必须实现的语义（plan 02 §3.4，逐条有格式事实依据）：

1. **beat 三元组** `i + n/d`；多 BPM 段按 `BPMList` 分段积分（`Σ Δbeats × 60/bpm`）。
2. **note 映射**走契约的 `note_type_from_rpe` / `side_from_above`（`above == 1` 才是正面），
   `is_fake = (isFake == 1)`，三者都保留 `*_raw`（双写，无损往返）。
3. **Hold**：`hold_time = endTime - startTime`（秒）；非 Hold 恒为 0；`endTime < startTime` 拒收。
4. **eventLayers 三态归一**：`null` 层 / 字段缺失 / 整段缺失 → 同一「空轨列表」，
   长度原样保留（契约要求 1..`RPE_MAX_EVENT_LAYERS`）。
5. **跨层求和**（契约 `JudgeLine.sum_track` 已实现；本层只保证不做错误折叠）。
6. **补洞**由契约 `EventLayer` 的 validator 完成（构造时**不要**重复补洞）。
7. **缓动** 1..29 归一化 + `easingLeft/Right` 切割 + 贝塞尔，全部由契约 `EventKeyframe` 承担；
   `speedEvents` 无 `bezier` 字段（格式事实）。
8. **father 嵌套**保留原始索引，成环/越界由契约 `PhigrosChart` 校验并拒收。
9. **bpm_factor** 存储但不参与换算（prpr 标为 TODO，存疑 D4）；`!= 1.0` 进隔离报告（qc）。
10. **is_cover**：`1 = 遮罩，其余 = 不遮罩`，**原样存 int**（禁止 `bool()`）。

时间口径（plan 02 §4）：解析后**立刻转秒**，IR 内只有秒；`BPMList` 完整保留在
`bpm_points`（下游 `field/` 派生 `J(τ)` 的唯一依据）。**本文件是全仓唯一的格式层
beat↔秒换算处**（plan 02 §9-11：接缝须裁定，裁定前不得出现第三处）。
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import ValidationError

from beatmorph.core.contracts import (
    RPE_EXTENDED_TRACKS,
    RPE_MAX_EVENT_LAYERS,
    RPE_NORMAL_TRACKS,
    Beat,
    BpmPoint,
    ChartFormat,
    ChartMeta,
    ChartSource,
    EventKeyframe,
    JudgeLine,
    NoteType,
    PhigrosChart,
    PhigrosNote,
    note_type_from_rpe,
    side_from_above,
)
from beatmorph.data.phira.client import sha1_hex

__all__ = [
    "RpeParseError",
    "RpeSchemaError",
    "beat_to_seconds",
    "is_cover_masks",
    "parse_rpejson",
    "seconds_to_beat",
]

#: 一分钟的秒数（时间换算的**唯一**来源常量，其余量一律由它派生）。
_SECONDS_PER_MINUTE: float = 60.0

#: RPE 轨字段名 → 契约 EventLayer 的 pythonic 字段名（**由契约常量派生**，避免两处漂移）。
_NORMAL_TRACK_FIELDS: tuple[tuple[str, str], ...] = tuple(
    zip(RPE_NORMAL_TRACKS, ("move_x", "move_y", "rotate", "alpha", "speed"), strict=True),
)
_EXTENDED_TRACK_FIELDS: tuple[tuple[str, str], ...] = tuple(
    zip(
        RPE_EXTENDED_TRACKS,
        ("scale_x", "scale_y", "color", "text", "gif", "incline"),
        strict=True,
    ),
)


class RpeParseError(ValueError):
    """RPEJSON 无法解析（不是合法 JSON / 顶层不是对象）。"""


class RpeSchemaError(RpeParseError):
    """schema 级违约（plan 02 §3.5）：**拒收 → 隔离区**，不得静默通过。"""


# ══════════════════════════════════════════════════════════════
# §3.4-1 格式层 beat ↔ 秒（本仓唯一的格式层换算处）
# ══════════════════════════════════════════════════════════════


def _require_bpm_points(bpm_points: Sequence[BpmPoint]) -> None:
    if not bpm_points:
        raise ValueError("bpm_points 不得为空（秒↔拍换算的唯一依据，红线 7）")
    previous = -math.inf
    for point in bpm_points:
        if point.time_beats < previous:
            raise ValueError("bpm_points 必须按 time_beats 升序")
        previous = point.time_beats


def _seconds_per_beat(bpm: float) -> float:
    """单段内「拍 → 秒」的比例系数 `60 / bpm`。"""
    return _SECONDS_PER_MINUTE / bpm


def beat_to_seconds(beats: float, bpm_points: Sequence[BpmPoint]) -> float:
    """格式层换算：RPE 原生拍 → 秒（分段积分 `Σ Δbeats × 60/bpm`）。

    首段之前（含负拍）用**首段 BPM 线性外推**——与 Phira 官方参考实现
    `beat2sec` 的循环行为一致（`t < et_beat` 时走 `t * 60/bpm` 分支）。

    Raises:
        ValueError: `bpm_points` 为空或未按拍升序。
    """
    _require_bpm_points(bpm_points)
    first = bpm_points[0]
    if beats <= first.time_beats:
        return (beats - first.time_beats) * _seconds_per_beat(first.bpm)

    seconds = 0.0
    for index, point in enumerate(bpm_points):
        span_start = point.time_beats
        if index + 1 >= len(bpm_points):
            return seconds + (beats - span_start) * _seconds_per_beat(point.bpm)
        span_end = bpm_points[index + 1].time_beats
        if beats <= span_end:
            return seconds + (beats - span_start) * _seconds_per_beat(point.bpm)
        seconds += (span_end - span_start) * _seconds_per_beat(point.bpm)
    return seconds  # pragma: no cover - 上面的最后一个分支必然返回


def seconds_to_beat(t_s: float, bpm_points: Sequence[BpmPoint]) -> float:
    """格式层换算：秒 → 拍（`beat_to_seconds` 的严格逆）。

    断言 `beat_to_seconds(seconds_to_beat(x)) == x`（1e-9 容差，M6⑤）。

    Raises:
        ValueError: `bpm_points` 为空或未按拍升序。
    """
    _require_bpm_points(bpm_points)
    first = bpm_points[0]
    if t_s <= 0.0:
        return first.time_beats + t_s / _seconds_per_beat(first.bpm)

    elapsed = 0.0
    for index, point in enumerate(bpm_points):
        per_beat = _seconds_per_beat(point.bpm)
        if index + 1 >= len(bpm_points):
            return point.time_beats + (t_s - elapsed) / per_beat
        span_beats = bpm_points[index + 1].time_beats - point.time_beats
        span_seconds = span_beats * per_beat
        if t_s <= elapsed + span_seconds:
            return point.time_beats + (t_s - elapsed) / per_beat
        elapsed += span_seconds
    return bpm_points[-1].time_beats  # pragma: no cover - 最后一个分支必然返回


def is_cover_masks(raw: int) -> bool:
    """`isCover` 语义：**1 = 遮罩，其余 = 不遮罩**（同 above 类陷阱，禁止 `bool()`）。"""
    return raw == 1


# ══════════════════════════════════════════════════════════════
# 原语
# ══════════════════════════════════════════════════════════════


def _as_float(value: object, what: str) -> float:
    """取浮点；bool / 字符串 / None 一律视为 schema 违约。"""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RpeSchemaError(f"{what} 必须是数值，得到 {value!r}")
    return float(value)


def _as_int(value: object, what: str) -> int:
    """取整数；bool 视为违约（JSON `true` 不是数字）。"""
    if isinstance(value, bool):
        raise RpeSchemaError(f"{what} 必须是整数，得到布尔 {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise RpeSchemaError(f"{what} 必须是整数，得到 {value!r}")


def _as_flag(value: object, what: str, *, default: bool) -> bool:
    """取布尔标志：接受 JSON 布尔或 0/1 数值，`None` 取默认。

    只认「`== 1` 为真」（与 `above` / `isCover` 同类的取值陷阱：`bool(2)` 是 True，
    但格式语义里 2 属于「其余」）。
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return float(value) == 1.0
    raise RpeSchemaError(f"{what} 必须是布尔或 0/1 数值，得到 {value!r}")


def _parse_beat(value: object, what: str) -> Beat:
    """解析 beat 三元组（`[i, n, d]` 或 `{"i":…,"n":…,"d":…}`）。"""
    if value is None:
        raise RpeSchemaError(f"{what} 缺失（RPE 的 beat 三元组为必填）")
    try:
        return Beat.model_validate(value)
    except ValidationError as exc:
        raise RpeSchemaError(f"{what} 不是合法 beat 三元组 {value!r}：{exc}") from exc


def _parse_event_value(value: object, what: str) -> float | str | tuple[int, int, int]:
    """解析事件取值：数值 / 文本（textEvents）/ RGB 三元组（colorEvents）。"""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        raise RpeSchemaError(f"{what} 不得为布尔")
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, list | tuple):
        if len(value) != 3:
            raise RpeSchemaError(f"{what} 数组长度必须为 3（RGB），得到 {value!r}")
        return (
            _as_int(value[0], f"{what}[0]"),
            _as_int(value[1], f"{what}[1]"),
            _as_int(value[2], f"{what}[2]"),
        )
    raise RpeSchemaError(f"{what} 取值类型不支持：{value!r}")


def _parse_bezier_points(value: object, what: str) -> tuple[float, float, float, float]:
    """解析 `bezierPoints`（4 个控制点；缺失时用契约默认值）。"""
    if value is None:
        return (0.0, 0.0, 0.0, 0.0)
    if not isinstance(value, list | tuple) or len(value) != 4:
        raise RpeSchemaError(f"{what} 必须是 4 元素数组，得到 {value!r}")
    return (
        _as_float(value[0], f"{what}[0]"),
        _as_float(value[1], f"{what}[1]"),
        _as_float(value[2], f"{what}[2]"),
        _as_float(value[3], f"{what}[3]"),
    )


def _parse_keyframe(raw: object, what: str) -> EventKeyframe:
    """解析一个事件关键帧（普通轨与特殊轨共用；speedEvents 无贝塞尔字段）。"""
    if not isinstance(raw, Mapping):
        raise RpeSchemaError(f"{what} 必须是对象，得到 {type(raw).__name__}")
    start_time = _parse_beat(raw.get("startTime"), f"{what}.startTime")
    end_raw = raw.get("endTime")
    end_time = start_time if end_raw is None else _parse_beat(end_raw, f"{what}.endTime")
    if end_time.to_beats() < start_time.to_beats():
        raise RpeSchemaError(
            f"{what} 的 endTime {end_time.to_beats()} < startTime {start_time.to_beats()}",
        )
    payload: dict[str, Any] = {
        "start_time": start_time,
        "end_time": end_time,
        "start": _parse_event_value(raw.get("start"), f"{what}.start"),
        "end": _parse_event_value(raw.get("end"), f"{what}.end"),
        "easing_type": raw.get("easingType", 1),
        "bezier": _as_flag(raw.get("bezier"), f"{what}.bezier", default=False),
        "bezier_points": _parse_bezier_points(raw.get("bezierPoints"), f"{what}.bezierPoints"),
        "easing_left": _as_float(raw.get("easingLeft", 0.0), f"{what}.easingLeft"),
        "easing_right": _as_float(raw.get("easingRight", 1.0), f"{what}.easingRight"),
        "link_group": _as_int(raw.get("linkgroup", 0), f"{what}.linkgroup"),
    }
    try:
        return EventKeyframe.model_validate(payload)
    except ValidationError as exc:
        raise RpeSchemaError(f"{what} 违约：{exc}") from exc


def _parse_track(raw: object, what: str) -> list[EventKeyframe]:
    """解析一条事件轨（`None`/缺失 → 空轨，即三态归一的一部分）。"""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise RpeSchemaError(f"{what} 必须是数组，得到 {type(raw).__name__}")
    return [_parse_keyframe(item, f"{what}[{i}]") for i, item in enumerate(raw)]


def _parse_event_layers(raw: object, what: str) -> list[Any]:
    """§3.4-4 三态归一：`null` 层 / 缺字段 / 缺整段 `eventLayers` → 同一「空轨列表」。

    长度**原样保留**（契约上限 `RPE_MAX_EVENT_LAYERS`）；整段缺失时补一个空层，
    因为契约要求 `event_layers` 至少 1 层（`min_length=1`）。
    """
    if raw is None:
        return [{}]
    if not isinstance(raw, list):
        raise RpeSchemaError(f"{what} 必须是数组或 null，得到 {type(raw).__name__}")
    if len(raw) > RPE_MAX_EVENT_LAYERS:
        raise RpeSchemaError(
            f"{what} 有 {len(raw)} 层，超过契约上限 {RPE_MAX_EVENT_LAYERS}（实测 1..5）",
        )
    layers: list[Any] = []
    for index, item in enumerate(raw):
        if item is None:
            layers.append({"layer_index": index})
            continue
        if not isinstance(item, Mapping):
            raise RpeSchemaError(f"{what}[{index}] 必须是对象或 null")
        payload: dict[str, Any] = {"layer_index": index}
        for rpe_name, field_name in _NORMAL_TRACK_FIELDS:
            payload[field_name] = _parse_track(item.get(rpe_name), f"{what}[{index}].{rpe_name}")
        layers.append(payload)
    return layers or [{}]


def _parse_extended(raw: object, what: str) -> dict[str, Any]:
    """解析第 5 层 `extended`（独立字段，不参与 eventLayers 的跨层求和）。"""
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise RpeSchemaError(f"{what} 必须是对象或 null，得到 {type(raw).__name__}")
    payload: dict[str, Any] = {}
    for rpe_name, field_name in _EXTENDED_TRACK_FIELDS:
        payload[field_name] = _parse_track(raw.get(rpe_name), f"{what}.{rpe_name}")
    return payload


# ══════════════════════════════════════════════════════════════
# §3.4-2/3 note 与 §3.4-1 BPMList
# ══════════════════════════════════════════════════════════════


def _parse_bpm_list(raw: object) -> list[BpmPoint]:
    """解析 `BPMList`（完整保留，不得丢弃或规范化掉非整拍 / 非常见分母的段）。"""
    if raw is None or (isinstance(raw, list) and not raw):
        raise RpeSchemaError("RPEJSON 缺少 BPMList（IR 要求至少一段 BPM）")
    if not isinstance(raw, list):
        raise RpeSchemaError(f"BPMList 必须是数组，得到 {type(raw).__name__}")
    points: list[BpmPoint] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise RpeSchemaError(f"BPMList[{index}] 必须是对象")
        what = f"BPMList[{index}]"
        beats = _parse_beat(item.get("startTime"), f"{what}.startTime").to_beats()
        bpm = _as_float(item.get("bpm"), f"{what}.bpm")
        try:
            points.append(BpmPoint(time_beats=beats, bpm=bpm))
        except ValidationError as exc:
            raise RpeSchemaError(f"{what} 违约：{exc}") from exc
    return points


def _parse_note(
    raw: object,
    *,
    line_id: int,
    note_index: int,
    bpm_points: Sequence[BpmPoint],
) -> PhigrosNote:
    """解析一个 note（§3.4-2/3）。"""
    what = f"judgeLineList[{line_id}].notes[{note_index}]"
    if not isinstance(raw, Mapping):
        raise RpeSchemaError(f"{what} 必须是对象，得到 {type(raw).__name__}")

    type_raw = _as_int(raw.get("type", int(NoteType.TAP)), f"{what}.type")
    try:
        note_type = note_type_from_rpe(type_raw)
    except ValueError as exc:
        raise RpeSchemaError(
            f"{what}.type={type_raw} 不合法（RPE 只允许 1/2/3/4 = Tap/Hold/Flick/Drag）：{exc}",
        ) from exc

    above_raw = _as_int(raw.get("above", 1), f"{what}.above")
    side = side_from_above(above_raw)
    is_fake_raw = _as_int(raw.get("isFake", 0), f"{what}.isFake")

    start_beats = _parse_beat(raw.get("startTime"), f"{what}.startTime").to_beats()
    t_seconds = beat_to_seconds(start_beats, bpm_points)

    hold_time = 0.0
    if note_type is NoteType.HOLD:
        end_beats = _parse_beat(raw.get("endTime"), f"{what}.endTime").to_beats()
        if end_beats < start_beats:
            raise RpeSchemaError(
                f"{what} 是 Hold，但 endTime {end_beats} < startTime {start_beats}",
            )
        hold_time = beat_to_seconds(end_beats, bpm_points) - t_seconds

    payload: dict[str, Any] = {
        "line_id": line_id,
        "t": t_seconds,
        "position_x": _as_float(raw.get("positionX", 0.0), f"{what}.positionX"),
        "side": side,
        "type": note_type,
        "hold_time": hold_time,
        "speed": _as_float(raw.get("speed", 1.0), f"{what}.speed"),
        "is_fake": is_fake_raw == 1,
        "above_raw": above_raw,
        "type_raw": type_raw,
        "is_fake_raw": is_fake_raw,
        "y_offset": _as_float(raw.get("yOffset", 0.0), f"{what}.yOffset"),
        "size": _as_float(raw.get("size", 1.0), f"{what}.size"),
    }
    # 只在字段存在时显式传值：缺省值由契约默认给出（visible_time 的默认值三方裁定为
    # 999999.0，本模块不重复写这个字面量，避免两处默认值漂移）。
    if raw.get("visibleTime") is not None:
        payload["visible_time"] = _as_float(raw.get("visibleTime"), f"{what}.visibleTime")
    if raw.get("alpha") is not None:
        payload["alpha"] = _as_int(raw.get("alpha"), f"{what}.alpha")
    try:
        return PhigrosNote.model_validate(payload)
    except ValidationError as exc:
        raise RpeSchemaError(f"{what} 违约：{exc}") from exc


def _parse_lines_and_notes(
    raw_lines: object,
    bpm_points: Sequence[BpmPoint],
) -> tuple[list[JudgeLine], list[PhigrosNote]]:
    """逐线解析判定线与其 notes（`line_id` = `judgeLineList` 索引）。"""
    if raw_lines is None:
        raise RpeSchemaError("RPEJSON 缺少 judgeLineList")
    if not isinstance(raw_lines, list) or not raw_lines:
        raise RpeSchemaError("judgeLineList 不得为空（schema 级违约 → 隔离区）")

    lines: list[JudgeLine] = []
    notes: list[PhigrosNote] = []
    for line_id, raw_line in enumerate(raw_lines):
        what = f"judgeLineList[{line_id}]"
        if not isinstance(raw_line, Mapping):
            raise RpeSchemaError(f"{what} 必须是对象，得到 {type(raw_line).__name__}")

        raw_notes = raw_line.get("notes") or []
        if not isinstance(raw_notes, list):
            raise RpeSchemaError(f"{what}.notes 必须是数组")
        notes.extend(
            _parse_note(note, line_id=line_id, note_index=index, bpm_points=bpm_points)
            for index, note in enumerate(raw_notes)
        )

        anchor_raw = raw_line.get("anchor", (0.5, 0.5))
        if not isinstance(anchor_raw, list | tuple) or len(anchor_raw) != 2:
            raise RpeSchemaError(f"{what}.anchor 必须是 2 元素数组，得到 {anchor_raw!r}")

        payload: dict[str, Any] = {
            "line_id": line_id,
            "group": _as_int(raw_line.get("Group", 0), f"{what}.Group"),
            "name": str(raw_line.get("Name", "Untitled")),
            "texture": str(raw_line.get("Texture", "line.png")),
            "anchor": (
                _as_float(anchor_raw[0], f"{what}.anchor[0]"),
                _as_float(anchor_raw[1], f"{what}.anchor[1]"),
            ),
            "event_layers": _parse_event_layers(raw_line.get("eventLayers"), f"{what}.eventLayers"),
            "extended": _parse_extended(raw_line.get("extended"), f"{what}.extended"),
            "father": _as_int(raw_line.get("father", -1), f"{what}.father"),
            "rotate_with_father": _as_flag(
                raw_line.get("rotateWithFather"),
                f"{what}.rotateWithFather",
                default=True,
            ),
            "is_cover": _as_int(raw_line.get("isCover", 1), f"{what}.isCover"),
            "z_order": _as_int(raw_line.get("zOrder", 0), f"{what}.zOrder"),
            "bpm_factor": _as_float(raw_line.get("bpmfactor", 1.0), f"{what}.bpmfactor"),
            "num_of_notes_raw": _as_int(raw_line.get("numOfNotes", 0), f"{what}.numOfNotes"),
        }
        attach_ui = raw_line.get("attachUI")
        if attach_ui:
            payload["attach_ui"] = str(attach_ui)
        try:
            lines.append(JudgeLine.model_validate(payload))
        except ValidationError as exc:
            raise RpeSchemaError(f"{what} 违约：{exc}") from exc
    return lines, notes


def _parse_meta(root: Mapping[str, Any]) -> ChartMeta:
    """解析 `META.*` 与根级 `chartTime`（offset 为**毫秒**，只记录不使用，Q11/D10）。"""
    raw = root.get("META") or {}
    if not isinstance(raw, Mapping):
        raise RpeSchemaError(f"META 必须是对象，得到 {type(raw).__name__}")
    payload: dict[str, Any] = {
        "offset_ms": _as_float(raw.get("offset", 0.0), "META.offset"),
        "rpe_version": _as_int(raw.get("RPEVersion", 0), "META.RPEVersion"),
        "chart_time_s": _as_float(root.get("chartTime", 0.0), "chartTime"),
        "level_text": str(raw.get("level", "")),
        "name": str(raw.get("name", "")),
        "composer": str(raw.get("composer", "")),
        "charter": str(raw.get("charter", "")),
        "song": str(raw.get("song", "")),
        "background": str(raw.get("background", "")),
    }
    try:
        return ChartMeta.model_validate(payload)
    except ValidationError as exc:
        raise RpeSchemaError(f"META 违约：{exc}") from exc


# ══════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════


def parse_rpejson(data: bytes, source: ChartSource) -> PhigrosChart:
    """RPEJSON 字节 → :class:`PhigrosChart` 契约（plan 02 §3.4）。

    Args:
        data: 谱面文件字节（**内容**必须确实是 RPEJSON；本函数不做格式嗅探）。
        source: 来源留痕；返回值的 `source` 会补上 `format=RPE`、`chart_sha1`。

    Raises:
        RpeParseError: 不是合法 JSON / 顶层不是对象。
        RpeSchemaError: schema 级违约（拒收 → 隔离区）。
    """
    try:
        root = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RpeParseError(f"RPEJSON 解析失败：{exc}") from exc
    if not isinstance(root, Mapping):
        raise RpeParseError(f"RPEJSON 顶层必须是对象，得到 {type(root).__name__}")

    bpm_points = _parse_bpm_list(root.get("BPMList"))
    lines, notes = _parse_lines_and_notes(root.get("judgeLineList"), bpm_points)
    meta = _parse_meta(root)

    resolved_source = source.model_copy(
        update={
            "format": ChartFormat.RPE,
            "chart_sha1": source.chart_sha1 or sha1_hex(data),
        },
    )
    try:
        return PhigrosChart(
            lines=lines,
            notes=notes,
            bpm_points=bpm_points,
            meta=meta,
            source=resolved_source,
        )
    except ValidationError as exc:
        raise RpeSchemaError(f"IR 契约校验失败（schema 违约 → 隔离区）：{exc}") from exc
