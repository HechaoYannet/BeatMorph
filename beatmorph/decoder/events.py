"""解码事件：场事件 -> （Hold 配对）-> `DecodedEvent` -> 契约 `PhigrosNote`（Plan 05 §3.2）。

时间口径（红线 7 / Plan 00 §3.8 I11）：`DecodedEvent` 的**唯一时间字段是秒**，
τ 只存在于解码器内部的 `FieldEvent`（配对需要它）；秒 -> τ 的全部换算经
`beatmorph/field/grid` 的权威接口，本模块不实现第二套 BPM 积分。

Hold 语义：目标场把 Hold 拆成 **hold 通道（起点）+ hold_end 通道（终点）**，
且两者落在**同一 `(k, i_x, s)` 纤维**（plan 03 §2 偏离 3）。因此解码侧的配对规则
就是该编码的逆：在同一线、同一 x 桶、同一侧的纤维内，把每个起点与**其后最近的**
终点配对（终点在起点之前 = 无法配对，退回 hold_time = 0 并计入 `PairingStats`）。
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from beatmorph.core.contracts.field import ChartFieldSpec
from beatmorph.core.contracts.phigros import (
    NoteType,
    PhigrosChart,
    PhigrosNote,
    Side,
    above_from_side,
)
from beatmorph.field.grid import FieldGrid, seconds_to_tau, tau_bin_index, tau_to_seconds
from beatmorph.field.target import CHANNEL_INDEX, HOLD_END_CHANNEL

#: 通道索引 -> 音符类型（hold_end 通道**不在**其中：它是标记通道，不是音符）。
CHANNEL_NOTE_TYPE: Final[dict[int, NoteType]] = {
    channel: note_type for note_type, channel in CHANNEL_INDEX.items()
}


@dataclass(frozen=True, slots=True)
class FieldEvent:
    """场上的一个**标记点**（decoder 内部中间态，不跨模块）。

    与 `DecodedEvent` 的分工：本类型还带 τ（Hold 配对需要），且**不区分**起点/终点
    （由 `channel` 表达）。构造 `PhigrosNote` 之后即丢弃。
    """

    line_id: int
    tau: float
    x_bin: int
    side: Side
    channel: int
    confidence: float = 0.0


@dataclass(frozen=True, slots=True)
class DecodedEvent:
    """解码出的一个**事件**（Plan 05 §3.2；构造 `PhigrosNote` 后即丢弃）。

    Attributes:
        t_s: 判定时刻（**秒**；公共接口不出现帧索引，Plan 00 §3.8 I11）。
        line_id: 所属判定线（`judgeLineList` 索引）。
        position_x: RPE 舞台系 x 坐标单位。
        side: `side_from_above` 得到的侧别（禁 truthiness 分支）。
        note_type: RPE 音符类型（`type_raw = int(note_type)`）。
        hold_time_s: 仅 HOLD 有意义；**非 HOLD 恒为 0.0**（格式事实：非 Hold 的
            `endTime == startTime`）。HOLD 配不到终点时同样为 0.0（计入配对统计），
            只有被注入的非法事件才可能为负——后处理据此报 `HOLD_REVERSED`。
        is_fake: 生成侧默认 False（`isFake` 是否作为生成目标未统计，plan 05 §9-10）。
        confidence: 供 plan 04 迭代重掩码使用的置信度（Plan 05 §4.1-5）。
    """

    t_s: float
    line_id: int
    position_x: float
    side: Side
    note_type: NoteType
    hold_time_s: float
    is_fake: bool
    confidence: float


@dataclass(frozen=True, slots=True)
class PairingStats:
    """Hold 配对的诊断计数（不丢弃、不改动，只报告）。"""

    n_starts: int
    n_ends: int
    n_paired: int
    #: 配不到终点的 Hold 起点：`hold_time = 0`，**保留不丢弃**（含痕在报告里）
    n_unpaired_starts: int
    #: 配不到起点的 Hold 终点：不产生音符，**丢弃**并计数
    n_orphan_ends: int
    #: 配对成功但 `end <= start`（零时长 Hold；格式允许 `endTime == startTime`）
    n_zero_length: int

    @property
    def as_stats(self) -> dict[str, float]:
        """转成报告统计字典（键名以 `hold_` 前缀分组）。"""
        return {
            "hold_starts": float(self.n_starts),
            "hold_ends": float(self.n_ends),
            "hold_paired": float(self.n_paired),
            "hold_unpaired_starts": float(self.n_unpaired_starts),
            "hold_orphan_ends": float(self.n_orphan_ends),
            "hold_zero_length": float(self.n_zero_length),
        }


def note_type_for_channel(channel: int) -> NoteType:
    """通道 -> 音符类型（hold_end 通道不属于音符，调用前必须已排除）。"""
    try:
        return CHANNEL_NOTE_TYPE[int(channel)]
    except KeyError as exc:
        raise ValueError(
            f"通道 {channel!r} 不是音符通道（合法值：{sorted(CHANNEL_NOTE_TYPE)}；"
            f"hold_end 通道 {HOLD_END_CHANNEL} 是标记通道）",
        ) from exc


def x_center(x_bin: int, spec: ChartFieldSpec) -> float:
    """x 桶 -> 桶中心坐标（**唯一的**反查路径；禁止各模块自行 floor / +0.5）。"""
    if not 0 <= x_bin < spec.x_bins:
        raise IndexError(f"x_bin={x_bin} 越界（X={spec.x_bins}）")
    return spec.x_min + (x_bin + 0.5) * spec.dx


def pair_events(
    events: Sequence[FieldEvent],
    grid: FieldGrid,
    *,
    spec: ChartFieldSpec | None = None,
) -> tuple[list[DecodedEvent], PairingStats]:
    """场标记点 -> `DecodedEvent`（含 Hold 起点/终点配对），输出**确定性排序**。

    排序键 `(t_s, line_id, position_x, side, note_type)`：与
    `PhigrosChart.sorted_notes()` 同口径，保证同 seed 两次解码逐字段一致。
    """
    if grid.t_bins <= 0 or not grid.bpm_points:
        raise ValueError("pair_events 需要已绑定时间轴的网格（t_bins > 0 且 bpm_points 非空）")
    bpm_points = grid.bpm_points
    if spec is None:
        k = max((event.line_id for event in events), default=0) + 1
        spec = grid.spec(k)

    ordered = sorted(
        events,
        key=lambda event: (event.line_id, event.tau, event.x_bin, int(event.side), event.channel),
    )
    ends: dict[tuple[int, int, int], list[float]] = {}
    starts: list[FieldEvent] = []
    for event in ordered:
        if event.channel == HOLD_END_CHANNEL:
            ends.setdefault((event.line_id, event.x_bin, int(event.side)), []).append(event.tau)
        else:
            starts.append(event)
    used: dict[tuple[int, int, int], list[bool]] = {
        key: [False] * len(values) for key, values in ends.items()
    }

    decoded: list[DecodedEvent] = []
    n_paired = 0
    n_unpaired = 0
    n_zero_length = 0
    for event in starts:
        note_type = note_type_for_channel(event.channel)
        hold_time = 0.0
        if note_type is NoteType.HOLD:
            key = (event.line_id, event.x_bin, int(event.side))
            fiber = ends.get(key, [])
            index = bisect_left(fiber, event.tau)
            while index < len(fiber) and used[key][index]:
                index += 1
            if index < len(fiber):
                used[key][index] = True
                n_paired += 1
                hold_time = float(
                    tau_to_seconds(fiber[index], bpm_points)
                    - tau_to_seconds(event.tau, bpm_points),
                )
                if hold_time <= 0.0:
                    n_zero_length += 1
            else:
                n_unpaired += 1
        decoded.append(
            DecodedEvent(
                t_s=float(tau_to_seconds(event.tau, bpm_points)),
                line_id=event.line_id,
                position_x=x_center(event.x_bin, spec),
                side=event.side,
                note_type=note_type,
                hold_time_s=hold_time,
                is_fake=False,
                confidence=float(event.confidence),
            ),
        )

    n_orphan = sum(1 for key, flags in used.items() for flag in flags if not flag)
    decoded.sort(key=event_sort_key)
    stats = PairingStats(
        n_starts=len(starts),
        n_ends=sum(len(values) for values in ends.values()),
        n_paired=n_paired,
        n_unpaired_starts=n_unpaired,
        n_orphan_ends=n_orphan,
        n_zero_length=n_zero_length,
    )
    return decoded, stats


def event_sort_key(event: DecodedEvent) -> tuple[float, int, float, int, int]:
    """`DecodedEvent` 的确定性排序键（与 `sorted_notes` 同口径 + 侧别/类型消歧）。"""
    return (
        event.t_s,
        event.line_id,
        event.position_x,
        int(event.side),
        int(event.note_type),
    )


def events_to_notes(events: Sequence[DecodedEvent]) -> list[PhigrosNote]:
    """`DecodedEvent` -> 契约 `PhigrosNote`（`*_raw` 双写口径）。

    映射全部经契约函数：`type_raw = int(note_type)`（RPE 枚举即 RPE 数字）、
    `above_raw = above_from_side(side)`（背面规范代表值 0）。

    Raises:
        ValueError: `hold_time_s < 0`（反向 Hold；应先经后处理检出并留痕，见
            `beatmorph.decoder.postprocess.legality`）。
    """
    notes: list[PhigrosNote] = []
    for event in events:
        if event.hold_time_s < 0.0:
            raise ValueError(
                f"反向 Hold（hold_time_s={event.hold_time_s}）不得进入契约层："
                "IR 的 hold_time 有 ge=0 约束，必须先由后处理按红线检出并留痕",
            )
        notes.append(
            PhigrosNote(
                line_id=event.line_id,
                t=float(event.t_s),
                position_x=float(event.position_x),
                side=event.side,
                type=event.note_type,
                hold_time=float(event.hold_time_s),
                is_fake=event.is_fake,
                above_raw=above_from_side(event.side),
                type_raw=int(event.note_type),
                is_fake_raw=1 if event.is_fake else 0,
            ),
        )
    return notes


def confidence_array(events: Sequence[DecodedEvent]) -> np.ndarray:
    """置信度数组（与 `events` 同序；供 plan 04 迭代重掩码与评估对齐）。"""
    return np.asarray([event.confidence for event in events], dtype=np.float64)


# ══════════════════════════════════════════════════════════════
# 判定线资格闸门（红线 6 的输入侧：不该被判定的线一个 note 都不该有）
# ══════════════════════════════════════════════════════════════

#: 闸门记账的稳定键（进 decode_stats / 产物 meta.json，便于事后审计）。
LINE_FILTER_KEYS: Final[tuple[str, ...]] = (
    "line_filter_allowed_lines",
    "line_filter_kept_events",
    "line_filter_dropped_empty_line",
    "line_filter_dropped_invisible",
)


def _has_alpha_track(chart: PhigrosChart, line_id: int) -> bool:
    """该判定线（含父线链）是否**真的有** alpha 事件轨。

    为什么必须先问这一句（不是防御式编程）：RPE 的默认 alpha 语义是「没有 alphaEvents
    的层求和为 0 ⇒ 不可见」（prpr A 级证据，见 docs/knowledges/phigros-format.md）。
    但解码器自建的**合成模板**造出来的线本来就**没有** alpha 轨——那是「没给这个信息」，
    不是「这条线不可见」。把两者混为一谈会把合成路径的全部 note 一次清空，且不报任何错。

    口径必须看**整条父线链**：alpha 是跨层求和（JudgeLine.pose_at），子线自己没有 alpha 轨
    但父线有时，子线的不透明度就是父线给的那个值——只看本线会把这种情形误判为「未知」。
    """
    return any(
        bool(layer.alpha)
        for line in chart.lines[line_id].ancestry(chart)
        for layer in line.event_layers
    )


def note_is_scorable(
    chart: PhigrosChart,
    note: PhigrosNote,
    *,
    opacity_threshold: float = 0.0,
    unknown_alpha_is_visible: bool = True,
) -> bool:
    """**唯一**的「这个 note 算不算可计分」判据（生成闸门与训练目标共用同一个实现）。

    可计分 = 不是假音符，且**命中时刻该线可见**（alpha 经 `pose_at` 跨层求和 + 父线递归）。
    线**没有 alpha 轨**时的默认值由 `unknown_alpha_is_visible` 给：生成闸门默认「未知 ⇒ 不算不可见」，
    训练目标显式传 False（真实谱面里没有 alphaEvents 的层求和为 0 ⇒ 不可见，A 级语义）。

    为什么必须只有一个实现：同一条判据现在有两个消费方——
    ① **生成侧**（`scorable_lines` → e2e 的 allowed_lines）：不该有 note 的线上一个 note 都不许有；
    ② **训练侧**（`data.scorable_target`）：目标只入账可计分的 note。
    两边口径一旦分叉，模型学的和产物守的就不是同一件事，而且**都不报错**。

    数据事实（`runs/_probe_scorable_share.py`，与调研任务 research/kipphi-rpejson 同口径）：
    train 300 张 / 392 502 note 里**命中时线不可见 22.81%**（其中真线上 12.94%）、可计分 75.08%；
    val 300 张 / 395 943 note 里不可见 18.68%、可计分 78.50%；两个 split 的
    「**可计分 note 落在装饰线上**」**都是 0** ⇒ 这两类在真实语料里本不相交。
    """
    if bool(note.is_fake):
        return False
    line_id = int(note.line_id)
    if not 0 <= line_id < len(chart.lines):
        return False
    if not _has_alpha_track(chart, line_id):
        # 两条路径的默认值**故意相反**：
        # * 生成闸门（默认 True）：模板/合成谱可能压根没给 alpha 轨 ⇒ 那是「没这个信息」，
        #   判成不可见会把合成路径的 note 一次清空（`_has_alpha_track` 的原始理由）；
        # * 训练目标（`gameplay_subchart` 传 False）：语料是**真实谱面**，RPE 的 A 级语义是
        #   「没有 alphaEvents 的层求和为 0 ⇒ 不可见」（docs/knowledges/phigros-format.md），
        #   所以没有 alpha 轨的线上的 note **不可计分** —— 这也正是语料实测（22.81%）的口径。
        return bool(unknown_alpha_is_visible)
    beats = float(seconds_to_tau(note.t, chart.bpm_points))
    return float(chart.lines[line_id].pose_at(beats, chart).alpha) > float(opacity_threshold)


def scorable_note_mask(
    chart: PhigrosChart,
    *,
    opacity_threshold: float = 0.0,
    unknown_alpha_is_visible: bool = True,
) -> list[bool]:
    """与 `chart.notes` **等长同序**的可计分掩码（目标过滤与台账都用它）。"""
    return [
        note_is_scorable(
            chart,
            note,
            opacity_threshold=opacity_threshold,
            unknown_alpha_is_visible=unknown_alpha_is_visible,
        )
        for note in chart.notes
    ]


def scorable_lines(chart: PhigrosChart, *, opacity_threshold: float = 0.0) -> frozenset[int]:
    """允许承载 note 的判定线集合 = 「**有可计分 note**」的线（决策者 2026-09-30 裁定的口径）。

    为什么不是「有 note 的线」：旧口径是

    ```python
    frozenset(int(note.line_id) for note in template.notes)
    ```

    它只看**有没有 note**，于是模板里「只有假音符」或「命中时线不可见」的线也被当成能承载 note 的线。
    实测（调研任务 research/kipphi-rpejson 的报告 §1，模板 15831）：34 条线里旧口径放过 0/28/29
    三条**表演线**，产物在它们上面放了 23 个 note（5.6%）；而真实语料 45 张 / 53 007 个 note 里
    **落在装饰线上的可计分 note = 0** ⇒ 我们的产物是唯一的例外。

    本仓自己的 split 上同口径复核（`runs/_probe_scorable_share.py`，train 300 张 / 392 502 note）：
    命中时线不可见 **22.81%**（其中真线上 12.94%）、可计分 75.08%、**可计分落在装饰线上 = 0**。

    口径与 :func:`filter_field_events_by_line` 逐条一致：fake 不计分；线**没有 alpha 轨**时
    「可见性未知」⇒ **不因可见性剔除**（理由见 :func:`_has_alpha_track`）。
    """
    mask = scorable_note_mask(chart, opacity_threshold=opacity_threshold)
    return frozenset(int(note.line_id) for note, ok in zip(chart.notes, mask, strict=True) if ok)


def gameplay_subchart(
    chart: PhigrosChart,
    *,
    opacity_threshold: float = 0.0,
    unknown_alpha_is_visible: bool = False,
) -> PhigrosChart:
    """谱面 -> **只含可玩内容**的子谱面：真线（∪ 其祖先线）+ 只留可计分 note。

    决策者口径（2026-09-30）：「将谱面表演的所有成分去掉，包括不计分 note、装饰线、不可见线等，
    **train 和 val 口径必须相同**」。实测（`runs/_probe_line_strip.py`，各 300 张）：

    | | 全部线（中位） | 真线（中位） | 可计分 note |
    |---|---|---|---|
    | train | 26（p90 68，max 240） | **5**（p90 12） | 75.17% |
    | val | 25（p90 62，max 434） | **5**（p90 13） | 79.02% |

    ⇒ 线轴缩到约 **1/5**，k_max=128 的超限行从 2.0–2.7% 降到 **0**（覆盖变好），
    而模型不再需要把大部容量用在「这条线上 λ 恒为 0」上。

    **为什么必须连带祖先线**：`JudgeLine.pose_at` 会把父线的位移/旋转合成进来，
    真线的**几何**可能由一条自己没有可计分 note 的线驱动；摘掉它等于换了输入而不是去掉表演。
    实测代价极小：train 平均 **0.06** 条/谱（4.7% 的谱需要），val **0.17** 条/谱（5.4%）。

    线序与原索引**保持相对顺序**（`father` 与 `note.line_id` 一并重映射）；
    被摘掉的装饰线由调用方**原样封存**，导出时按原索引插回（本函数不负责回插）。
    `meta` 不动 ⇒ τ 轴与窗口集合不因去表演而改变。

    Raises:
        ValueError: 整张谱面没有任何可计分 note（调用方应据此跳过该行，而不是喂一张空谱）。
    """
    mask = scorable_note_mask(
        chart,
        opacity_threshold=opacity_threshold,
        unknown_alpha_is_visible=unknown_alpha_is_visible,
    )
    real = {int(note.line_id) for note, ok in zip(chart.notes, mask, strict=True) if ok}
    keep: set[int] = set(real)
    for line_id in real:
        keep |= {int(line.line_id) for line in chart.lines[line_id].ancestry(chart)}
    if not keep:
        raise ValueError("谱面没有任何可计分 note：没有可玩内容（调用方应跳过该行）")
    order = [index for index in range(len(chart.lines)) if index in keep]
    remap = {old: new for new, old in enumerate(order)}
    lines = [
        chart.lines[old].model_copy(
            update={"line_id": remap[old], "father": remap.get(int(chart.lines[old].father), -1)},
        )
        for old in order
    ]
    notes = [
        note.model_copy(update={"line_id": remap[int(note.line_id)]})
        for note, ok in zip(chart.notes, mask, strict=True)
        if ok
    ]
    return chart.model_copy(update={"lines": lines, "notes": notes})


def filter_field_events_by_line(
    events: Sequence[FieldEvent],
    *,
    chart: PhigrosChart,
    window_grid: FieldGrid,
    origin_s: float,
    allowed_lines: frozenset[int] | None = None,
    opacity_threshold: float = 0.0,
) -> tuple[list[FieldEvent], dict[str, float]]:
    """丢掉「**此刻不该承载 note**」的判定线上的场事件（在配对之前）。

    两条判据（决策者 2026-09-30 实测报告：e2e 产物里 **35.8%** 的 note 落在 alpha=0 的线上、
    **22.4%** 落在模板里零 note 的装饰线上）：

    1. **不可见**：该线在该事件的**绝对时刻**的不透明度 <= opacity_threshold。不透明度经
       JudgeLine.pose_at 求出（跨层求和 + 父线递归，契约里的唯一实现）；仅当该线**确实带
       alpha 轨**时才判定（见 `_has_alpha_track`）。
    2. **不该有 note 的线**（装饰 / 纯表演线）：allowed_lines 给出「允许承载 note 的线集合」
       时，不在其中的线一律丢弃。无条件生成时没有这个信息 ⇒ 传 None。

    为什么放在**配对之前**：Hold 的起点与终点落在同一纤维，先配对后过滤会制造孤儿端点；
    先过滤则两者一起消失，PairingStats 仍然自洽。

    Returns:
        (保留的事件, 记账)；记账键见 LINE_FILTER_KEYS。
    """
    kept: list[FieldEvent] = []
    dropped_empty = 0
    dropped_invisible = 0
    opacity: dict[tuple[int, int], float] = {}
    for event in events:
        line_id = int(event.line_id)
        if allowed_lines is not None and line_id not in allowed_lines:
            dropped_empty += 1
            continue
        if 0 <= line_id < len(chart.lines) and _has_alpha_track(chart, line_id):
            # 缓存键必须用**权威的 τ 格下标**（tau_bin_index），不能写 int(tau)：
            # 一个 4 拍窗里 tau < 1 的格占绝大多数，int(tau) 会把它们塌成同一个键，
            # 于是整窗只按第一个事件判一次可见性（实测复现，test_line_eligibility 钉住）。
            key = (line_id, int(tau_bin_index(event.tau)))
            alpha = opacity.get(key)
            if alpha is None:
                absolute_s = float(origin_s) + float(
                    tau_to_seconds(event.tau, window_grid.bpm_points),
                )
                beats = float(seconds_to_tau(absolute_s, chart.bpm_points))
                alpha = float(chart.lines[line_id].pose_at(beats, chart).alpha)
                opacity[key] = alpha
            if alpha <= float(opacity_threshold):
                dropped_invisible += 1
                continue
        kept.append(event)
    allowed = float(len(allowed_lines)) if allowed_lines is not None else -1.0
    stats = {
        "line_filter_allowed_lines": allowed,
        "line_filter_kept_events": float(len(kept)),
        "line_filter_dropped_empty_line": float(dropped_empty),
        "line_filter_dropped_invisible": float(dropped_invisible),
    }
    return kept, stats


__all__ = [
    "CHANNEL_NOTE_TYPE",
    "LINE_FILTER_KEYS",
    "DecodedEvent",
    "FieldEvent",
    "PairingStats",
    "confidence_array",
    "event_sort_key",
    "events_to_notes",
    "filter_field_events_by_line",
    "gameplay_subchart",
    "note_is_scorable",
    "note_type_for_channel",
    "pair_events",
    "scorable_lines",
    "scorable_note_mask",
    "x_center",
]
