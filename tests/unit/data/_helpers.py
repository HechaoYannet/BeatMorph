"""测试内构造 RPEJSON 的最小工厂。

**纪律**（AGENTS.md §3.3 / 红线 7）：夹具与 mock 不得固化物理常量。凡需要坐标/网格，
一律引用 `beatmorph.core.contracts` 的派生常量；需要「越界值」时用
`RPE_STAGE_HALF_WIDTH * 系数` 现算，而不是写一个裸数字。
"""

from __future__ import annotations

import json
from typing import Any

from beatmorph.core.contracts import RPE_STAGE_HALF_WIDTH, ChartSource


def out_of_range_position_x() -> float:
    """一个确定越界（> `RPE_STAGE_HALF_WIDTH`）的 positionX（派生，不写裸数字）。"""
    return RPE_STAGE_HALF_WIDTH * 1.05


def keyframe(
    start: list[int],
    end: list[int],
    start_value: Any,
    end_value: Any,
    *,
    easing_type: int = 1,
    **extra: Any,
) -> dict[str, Any]:
    """一个普通事件关键帧（默认线性缓动）。"""
    payload: dict[str, Any] = {
        "startTime": start,
        "endTime": end,
        "start": start_value,
        "end": end_value,
        "easingType": easing_type,
        "bezier": 0,
        "bezierPoints": [0.0, 0.0, 0.0, 0.0],
        "easingLeft": 0.0,
        "easingRight": 1.0,
        "linkgroup": 0,
    }
    payload.update(extra)
    return payload


def note(
    note_type: int,
    start: list[int],
    *,
    end: list[int] | None = None,
    position_x: float = 0.0,
    above: int = 1,
    is_fake: int = 0,
    **extra: Any,
) -> dict[str, Any]:
    """一个 RPE note（endTime 缺省等于 startTime）。"""
    payload: dict[str, Any] = {
        "type": note_type,
        "startTime": start,
        "endTime": start if end is None else end,
        "positionX": position_x,
        "above": above,
        "isFake": is_fake,
        "size": 1.0,
        "speed": 1.0,
        "alpha": 255,
        "visibleTime": 999999.0,
        "yOffset": 0.0,
    }
    payload.update(extra)
    return payload


def build_rpe(
    *,
    lines: list[dict[str, Any]] | None = None,
    bpm_list: list[dict[str, Any]] | None = None,
    chart_time: float = 60.0,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """最小可解析的 RPEJSON 根对象（字段可由测试覆盖）。"""
    root: dict[str, Any] = {
        "BPMList": bpm_list if bpm_list is not None else [{"bpm": 120.0, "startTime": [0, 0, 1]}],
        "META": {
            "RPEVersion": 150,
            "charter": "unit-test",
            "composer": "unit-test",
            "level": "TEST Lv.1",
            "name": "unit-test",
            "offset": 0,
            "song": "unit.ogg",
            **(meta or {}),
        },
        "chartTime": chart_time,
        "judgeLineList": lines if lines is not None else [{"Name": "line-0", "notes": []}],
    }
    return root


def build_rpe_bytes(**kwargs: Any) -> bytes:
    """build_rpe 的字节形式。"""
    return json.dumps(build_rpe(**kwargs), ensure_ascii=False).encode("utf-8")


def chart_source(chart_id: int | None = None) -> ChartSource:
    """构造一个空 ChartSource（解析器会补 format 与 chart_sha1）。"""
    return ChartSource(chart_id=chart_id)
