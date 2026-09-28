"""合法性报告契约（Plan 05 §3.3；类型归属见 RFC-0030 §2）。

`Violation` / `Edit` / `LegalityReport` 被 plan 05（后处理）、plan 06（合法性指标）、
plan 08（`report/legality.json`）共同消费，属**跨模块数据**，因此按 CLAUDE.md 红线 2
落在契约层（plan 05 §9-12 要求的"实现前必须裁决"，RFC-0030 记录该裁决）。

三条硬语义（与 plan 05 §3.3 / §4.3 一一对应）：

1. **`violations` 为空 == 完全合法**——这正是红线 6「100% 合法方可导出」的判据
   （writer 的门禁只认这一条，不另立第二套判据）。
2. **每一次丢弃/改动都必须留痕**：`edits` 为空时，导出结果必须与解码输出逐字段一致。
3. **钳位在类型层不可表达**：`EditKind` 里**没有** clamp 成员，且 `position_x_clamped`
   恒为 0（红线 3 / RFC-0029 §3.1：`±675` 是可见边界而非合法值域）。
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ViolationKind(StrEnum):
    """**会阻断导出**的违规种类（红线 6：`violations` 非空即拒绝写出）。

    只收录「格式/物理红线」——即无论模型想表达什么，该谱面都不可能被正确游玩或
    正确解析的情形。**不含**三类"未查证"项（同刻按键上限 / 跨线几何冲突 /
    Hold 期间线速度变化）：它们按 plan 05 §4.3 只进 `stats`，不作为红线。
    """

    LINE_INDEX_OUT_OF_RANGE = "line_index_out_of_range"
    """note 的 line_id 不在 [0, K) —— 写进 RPEJSON 会落到不存在的判定线上。"""

    HOLD_REVERSED = "hold_reversed"
    """Hold 的 endTime < startTime（`hold_time_s < 0`）。格式 A 级语义：区间必须非负。"""

    DUPLICATE_EVENT = "duplicate_event"
    """同一线、同一刻、同一 `position_x/side/type` 的事件出现两次以上（点过程一次实现
    不应产生重复事件）；可由后处理去重消解。"""

    SAME_INSTANT_OVER_LIMIT = "same_instant_over_limit"
    """同一时刻按键数超过显式配置的上限。**默认关闭**：上限数值未查证（plan 05 §9-1），
    只有调用方显式给出 `same_instant_limit` 时才成立。"""


class EditKind(StrEnum):
    """后处理**留痕**的动作种类。

    ⚠️ 本枚举**刻意不含任何钳位（clamp）成员**：红线 3 / RFC-0029 §3.1 规定
    `|positionX| > RPE_STAGE_HALF_WIDTH` **只统计不钳位**，所以「把 x 改小」
    这个动作在类型层就不可表达。新增成员前必须先开 RFC。
    """

    DROP_DUPLICATE = "drop_duplicate"
    DROP_HOLD_REVERSED = "drop_hold_reversed"
    DROP_LINE_OUT_OF_RANGE = "drop_line_out_of_range"


class Violation(BaseModel):
    """一条违规（`note_index` 为 `PhigrosChart.notes` 中的下标；`None` 表示谱面级）。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: ViolationKind
    note_index: int | None = None
    detail: str = ""


class Edit(BaseModel):
    """一次后处理改动的留痕（改了什么、原值、新值、理由）。

    值一律用**字符串**记录：本类型的职责是审计，不是数值计算；字符串不会因为
    float 格式化而在往返中改变，也允许记录"整条 note 被丢弃"这类非数值改动。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: EditKind
    note_index: int | None = None
    field: str = ""
    before: str = ""
    after: str = ""
    reason: str = ""
    #: 本条留痕**消解**的是哪一类违规（空 = 预防性 / 结构性清理）。
    #: 有了它，"修复前的发现"不必再留在 `violations` 里——后者只描述**当前**谱面状态，
    #: 从而保住"violations 为空 == 可导出"这条唯一判据（红线 6）。
    resolved: ViolationKind | None = None


class LegalityReport(BaseModel):
    """合法性报告：`violations`（阻断导出）+ `edits`（留痕）+ `stats`（统计）。

    `stats` 是**只读诊断**：越界计数、同刻按键数分布、Hold 时长分布、跨线冲突计数、
    Hold-期间线速度变化计数、被检出但未处置的重复事件数等。它不参与导出判定。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    violations: list[Violation] = Field(default_factory=list)
    edits: list[Edit] = Field(default_factory=list)
    stats: dict[str, float] = Field(default_factory=dict)
    #: 判据声明（例如跨线几何冲突的容差口径），随报告一起落盘，避免"数字没有出处"。
    criterion: str = ""

    @property
    def is_legal(self) -> bool:
        """`violations` 为空 == 完全合法（红线 6 的**唯一**判据）。"""
        return not self.violations

    def stat(self, name: str, default: float = 0.0) -> float:
        """取一个统计量（缺失时返回 `default`）。"""
        return float(self.stats.get(name, default))

    def with_stats(self, extra: dict[str, float]) -> LegalityReport:
        """覆盖/追加统计量（返回新报告）。"""
        return self.model_copy(update={"stats": {**self.stats, **extra}})

    def format(self) -> str:
        """渲染成可直接进日志/报告的多行文本。"""
        head = "合法（无违规）" if self.is_legal else f"违规 {len(self.violations)} 项"
        lines = [f"合法性报告：{head}；留痕 {len(self.edits)} 条"]
        for violation in self.violations:
            where = "谱面级" if violation.note_index is None else f"note#{violation.note_index}"
            lines.append(f"  ✗ [{violation.kind}] {where}：{violation.detail}")
        for edit in self.edits:
            origin = "" if edit.resolved is None else f" ⟨消解 [{edit.resolved}]⟩"
            lines.append(
                f"  • [{edit.kind}] note#{edit.note_index} {edit.field}: "
                f"{edit.before} -> {edit.after}（{edit.reason}）{origin}",
            )
        if self.stats:
            rendered = "，".join(f"{name}={value:g}" for name, value in sorted(self.stats.items()))
            lines.append(f"  统计：{rendered}")
        if self.criterion:
            lines.append(f"  判据：{self.criterion}")
        return "\n".join(lines)


#: 恒为 0 的「钳位计数」键名：**只许被断言为 0，不许被写入其它值**（红线 3）。
POSITION_X_CLAMPED_KEY: str = "position_x_clamped"


def assert_no_position_clamp(report: LegalityReport) -> None:
    """红线 3 的机器可验证断言：`positionX` 钳位次数**恒为 0**。

    `positionX` 越界只统计不钳位（RFC-0029 §3.1 / 单位几何文档 §7.5），
    因此任何非 0 值都意味着后处理越权改了模型的落点分配。
    """
    value = report.stat(POSITION_X_CLAMPED_KEY)
    if value != 0.0:
        raise AssertionError(f"positionX 钳位次数必须恒为 0，得到 {value!r}（红线 3 违例）")


__all__ = [
    "POSITION_X_CLAMPED_KEY",
    "Edit",
    "EditKind",
    "LegalityReport",
    "Violation",
    "ViolationKind",
    "assert_no_position_clamp",
]
