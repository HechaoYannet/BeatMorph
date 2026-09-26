"""本轮新增契约的直测（Plan 00 + RFC-0030）：`legality.py` 与两个映射函数的逆性质。

放在独立文件而不是改写 `test_phigros_contracts.py`：前者是 plan 05 引入的契约补充，
后者是 plan 00 的契约测试清单；分开便于回溯「哪条断言是谁加的」。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from beatmorph.core.contracts import (
    POSITION_X_CLAMPED_KEY,
    SIDE_ORDER,
    Edit,
    EditKind,
    LegalityReport,
    Side,
    Violation,
    ViolationKind,
    above_from_side,
    assert_no_position_clamp,
    side_from_above,
    side_from_index,
    side_index,
)


def test_above_from_side_is_the_canonical_inverse() -> None:
    """`above_from_side` 给出规范代表值（FRONT -> 1，BACK -> 0），往返恒等。"""
    assert above_from_side(Side.FRONT) == 1
    assert above_from_side(Side.BACK) == 0
    for side in Side:
        assert side_from_above(above_from_side(side)) is side


def test_side_index_and_side_from_index_are_strict_inverses() -> None:
    """索引 <-> Side 必须互为逆，且索引与**枚举值不同**（枚举是 ±1）。"""
    assert SIDE_ORDER == (Side.FRONT, Side.BACK)
    for index, side in enumerate(SIDE_ORDER):
        assert side_index(side) == index
        assert side_from_index(index) is side
        assert side_index(side) != int(side)
    for bad in (-1, len(SIDE_ORDER), True, 1.0, "0"):
        with pytest.raises(ValueError, match="侧别索引"):
            side_from_index(bad)  # type: ignore[arg-type]


def test_report_legality_is_exactly_the_empty_violations_condition() -> None:
    """红线 6 的唯一判据：`violations` 为空 == 可导出。"""
    clean = LegalityReport(stats={"chart_notes_total": 3.0})
    assert clean.is_legal
    dirty = LegalityReport(
        violations=[Violation(kind=ViolationKind.DUPLICATE_EVENT, note_index=1, detail="dup")],
    )
    assert not dirty.is_legal
    assert clean.stat("chart_notes_total") == 3.0
    assert clean.stat("missing", 7.0) == 7.0
    assert "违规 1 项" in dirty.format()
    assert POSITION_X_CLAMPED_KEY in clean.with_stats({POSITION_X_CLAMPED_KEY: 0.0}).stats


def test_edit_kinds_cannot_express_a_clamp() -> None:
    """红线 3 的类型级保证：`EditKind` 里没有、也不许有"钳位"这个动作。"""
    assert not any("clamp" in member.value for member in EditKind)
    edit = Edit(
        kind=EditKind.DROP_DUPLICATE,
        note_index=0,
        field="(line_id,t,position_x,side,type)",
        before="(0, 1.0, 0.0, 0, 1)",
        after="(丢弃)",
        reason="重复事件",
        resolved=ViolationKind.DUPLICATE_EVENT,
    )
    assert edit.resolved is ViolationKind.DUPLICATE_EVENT
    assert "消解" in LegalityReport(edits=[edit]).format()


def test_assert_no_position_clamp_rejects_any_non_zero() -> None:
    assert_no_position_clamp(LegalityReport())
    assert_no_position_clamp(LegalityReport(stats={POSITION_X_CLAMPED_KEY: 0.0}))
    for value in (1.0, -1.0, 0.5):
        with pytest.raises(AssertionError, match="钳位"):
            assert_no_position_clamp(LegalityReport(stats={POSITION_X_CLAMPED_KEY: value}))


def test_report_is_frozen_and_rejects_unknown_fields() -> None:
    """契约是 pydantic 模型：不可变 + 拒绝未知字段（防止 schema 悄悄长出第二套）。"""
    report = LegalityReport()
    with pytest.raises(ValidationError):
        LegalityReport(unknown_field=1)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        report.stats = {"x": 1.0}  # type: ignore[misc]
