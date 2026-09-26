"""贪心一对一事件匹配与全谱相位偏移搜索（plan 06 §4.2 / §4.3）。

匹配协议（plan §4.2，逐条实现）：

1. **贪心一对一**：候选对按 |Δt| 升序贪心接受，每个 gold / 每个 pred 至多配对一次
   （STRUM 做法，文献库 §7.2-1）。候选对只在容差窗内生成（bisect 窗口），
   因此匹配是 O(n log n)，与谱面长度同阶。
2. **未匹配的生成事件留在分母**：MatchResult 保留 unmatched_pred / unmatched_gold；
   precision 的分母恒为**全部** pred 事件（ChartGenEval 的 timing clean rate 口径，
   文献库 §7.2-2）。否则「少生成」可以刷高 precision。
3. **匹配不跨线贪心**：默认按时间匹配，**不看 line_id**；线号不一致必须由 event 族
   暴露（plan §4.2-4）。
4. **确定性且与输入顺序无关**：候选对的排序键全部是事件的**规范键**
   （decoder.events.event_sort_key），不使用输入下标。因此打乱输入顺序不改变匹配
   结果（corruption 准入的不变性控制之二）。

秒域（红线 7）：本模块的所有量都是秒，**不出现帧索引，也不出现 τ 格索引**。
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Final

from beatmorph.decoder.events import DecodedEvent, event_sort_key
from beatmorph.eval.protocol import PhaseSearchConfig

#: 标记一致性判据（pred vs gold）；容差由 EvalConfig 给出。
MarkerFilter = Callable[[DecodedEvent, DecodedEvent], bool]

#: F1 的分母为 0 时返回 0.0 的约定（缺失比率**不得**用 nan 冒充：报告是冻结字段）。
_EMPTY_RATE: Final[float] = 0.0


@dataclass(frozen=True, slots=True)
class MatchPair:
    """一对匹配（下标指向**输入序列**；delta_s = pred.t - gold.t，带符号）。"""

    pred_index: int
    gold_index: int
    delta_s: float


@dataclass(frozen=True, slots=True)
class MatchResult:
    """一次匹配的全部结果（含未匹配项，**未匹配 pred 不得被丢弃**）。

    Attributes:
        tolerance_s: 本次匹配用的时间容差（秒）。
        n_pred / n_gold: 两侧事件总数（分母的唯一来源）。
        pairs: 匹配对（按 gold 规范序排序，确定性）。
        unmatched_pred: 未匹配的 pred 下标（**留在 precision 分母里**）。
        unmatched_gold: 未匹配的 gold 下标（= recall 的漏检）。
    """

    tolerance_s: float
    n_pred: int
    n_gold: int
    pairs: tuple[MatchPair, ...]
    unmatched_pred: tuple[int, ...]
    unmatched_gold: tuple[int, ...]

    @property
    def n_pairs(self) -> int:
        """匹配对数（timing 族的 TP）。"""
        return len(self.pairs)

    @property
    def tp(self) -> int:
        """timing 族的 TP = 匹配对数。"""
        return len(self.pairs)

    @property
    def fp(self) -> int:
        """timing 族的 FP = **全部**未匹配 pred（分母口径，plan §4.2-2）。"""
        return self.n_pred - len(self.pairs)

    @property
    def fn(self) -> int:
        """timing 族的 FN = 未匹配 gold。"""
        return self.n_gold - len(self.pairs)

    @property
    def precision_denominator(self) -> int:
        """precision 的分母 = 匹配数 + 未匹配 pred 数 = **全部** pred 事件。"""
        return self.n_pred

    def rates(self) -> tuple[float, float, float]:
        """timing 族的 (precision, recall, F1)。"""
        return precision_recall_f1(self.tp, self.n_pred, self.n_gold)


def precision_recall_f1(tp: int, n_pred: int, n_gold: int) -> tuple[float, float, float]:
    """(precision, recall, F1)；分母为 0 时该比率为 0.0（**并须连同计数一起读**）。

    ⚠️ 返回 0.0 而不是 nan 是刻意的：报告是冻结字段（plan §3.2），NaN 无法进 JSON；
    但「0.0」与「无样本」必须由随行的 n_pred / n_gold 区分——测试对此有断言。
    """
    precision = tp / n_pred if n_pred else _EMPTY_RATE
    recall = tp / n_gold if n_gold else _EMPTY_RATE
    denominator = precision + recall
    f1 = (2.0 * precision * recall / denominator) if denominator > 0.0 else _EMPTY_RATE
    return (float(precision), float(recall), float(f1))


def marker_equal(
    pred: DecodedEvent,
    gold: DecodedEvent,
    *,
    position_x_tolerance: float,
) -> bool:
    """标记是否一致（event 族 TP 的判据）：line_id / positionX / side / type。

    plan §4.1 的 event-F1 要求四项一致；positionX 的「一致」口径 plan 未写死，
    本实现按 §4.5 的量化粒度声明为 |Δx| <= position_x_tolerance（默认一个桶宽）。
    """
    return (
        pred.line_id == gold.line_id
        and pred.side is gold.side
        and pred.note_type is gold.note_type
        and abs(float(pred.position_x) - float(gold.position_x)) <= position_x_tolerance
    )


def greedy_match(
    pred: Sequence[DecodedEvent],
    gold: Sequence[DecodedEvent],
    *,
    tolerance_s: float,
    marker_filter: MarkerFilter | None = None,
) -> MatchResult:
    """贪心一对一匹配（plan §4.2-1；候选集可选地要求标记一致）。

    Args:
        pred / gold: 两侧事件（秒域）；**顺序无关**（内部按规范键排序）。
        tolerance_s: 时间容差（秒，含端点：|Δt| <= tolerance_s 才成为候选）。
        marker_filter: 非 None 时只保留通过该判据的候选对（match_marking=True）；
            默认 None，即 timing 族**不看标记**（plan §4.2-4）。

    Returns:
        MatchResult（未匹配 pred 全部保留，供 precision 分母使用）。
    """
    pred_tuple = tuple(pred)
    gold_tuple = tuple(gold)
    pred_order = sorted(range(len(pred_tuple)), key=lambda index: event_sort_key(pred_tuple[index]))
    gold_order = sorted(range(len(gold_tuple)), key=lambda index: event_sort_key(gold_tuple[index]))
    gold_times = [gold_tuple[index].t_s for index in gold_order]

    # 候选对：(|Δt|, gold 规范秩, pred 规范秩, pred 下标, gold 下标)。
    # 三个排序键都是规范量 → 与输入顺序无关（corruption 的不变性控制之二）。
    candidates: list[tuple[float, int, int, int, int]] = []
    for pred_rank, pred_index in enumerate(pred_order):
        event = pred_tuple[pred_index]
        low = bisect_left(gold_times, event.t_s - tolerance_s)
        high = bisect_right(gold_times, event.t_s + tolerance_s)
        for gold_rank in range(low, high):
            gold_index = gold_order[gold_rank]
            if marker_filter is not None and not marker_filter(event, gold_tuple[gold_index]):
                continue
            delta = float(event.t_s) - float(gold_tuple[gold_index].t_s)
            # 容差是**含端点**的判据（plan §4.1：TP = |Δt| <= 容差），因此窗口命中后
            # 仍要按差值再判一次：bisect 窗口在浮点上比 <= 判据宽一个 ulp，
            # 恰好落在边界外 1 ulp 的事件不得被算作匹配（M6.1 的边界用例）。
            if abs(delta) > tolerance_s:
                continue
            candidates.append((abs(delta), gold_rank, pred_rank, pred_index, gold_index))
    candidates.sort()

    used_pred: set[int] = set()
    used_gold: set[int] = set()
    accepted: list[tuple[int, int, MatchPair]] = []
    for _, gold_rank, pred_rank, pred_index, gold_index in candidates:
        if pred_index in used_pred or gold_index in used_gold:
            continue
        used_pred.add(pred_index)
        used_gold.add(gold_index)
        delta = float(pred_tuple[pred_index].t_s) - float(gold_tuple[gold_index].t_s)
        accepted.append((gold_rank, pred_rank, MatchPair(pred_index, gold_index, delta)))
    accepted.sort(key=lambda item: (item[0], item[1]))

    return MatchResult(
        tolerance_s=float(tolerance_s),
        n_pred=len(pred_tuple),
        n_gold=len(gold_tuple),
        pairs=tuple(pair for _, _, pair in accepted),
        unmatched_pred=tuple(index for index in pred_order if index not in used_pred),
        unmatched_gold=tuple(index for index in gold_order if index not in used_gold),
    )


@dataclass(frozen=True, slots=True)
class PhaseOutcome:
    """全谱相位偏移的搜索结果（plan 06 §4.3 必须**单独报**的读数）。

    Attributes:
        searched: 是否真的做了搜索（False 时 offset_s 为 None，两个 F1 相同）。
        offset_s: 最佳全局偏移（秒，**应从 pred 中减去**）；搜索关闭时为 None。
        f1_before: 不搜索（offset = 0）时的 timing-F1（主容差）。
        f1_after: 搜索后的 timing-F1（主容差）。
        grid_size: 搜索网格点数（可复现性留痕）。
    """

    searched: bool
    offset_s: float | None
    f1_before: float
    f1_after: float
    grid_size: int

    def improved(self, *, margin: float = 0.0) -> bool:
        """搜索是否**显著**优于不搜索（plan §4.3 要求据此写结论段）。"""
        return self.f1_after > self.f1_before + margin


def shift_events(events: Sequence[DecodedEvent], offset_s: float) -> tuple[DecodedEvent, ...]:
    """把事件整体平移 -offset_s（秒）：offset 是**待扣掉的**系统偏移。

    与 search_phase_offset 的方向约定一致：注入 +δ 的延迟后，搜索得到
    offset_s = δ，把它从 pred 减去即恢复对齐（M6.2）。
    """
    return tuple(replace(event, t_s=float(event.t_s) - float(offset_s)) for event in events)


def search_phase_offset(
    pred: Sequence[DecodedEvent],
    gold: Sequence[DecodedEvent],
    *,
    tolerance_s: float,
    search: PhaseSearchConfig,
    marker_filter: MarkerFilter | None = None,
) -> PhaseOutcome:
    """在 search 网格内搜索使 timing-F1（主容差）最大的全谱偏移（plan §4.3）。

    取最大 F1 时**首个最大值获胜**，而网格以 0 打头、其后按 |offset| 递增 →
    无注入偏移时返回的 |offset| <= step（M6.2 的判据）。
    """
    baseline = greedy_match(pred, gold, tolerance_s=tolerance_s, marker_filter=marker_filter)
    f1_before = baseline.rates()[2]
    if not search.enabled:
        return PhaseOutcome(
            searched=False,
            offset_s=None,
            f1_before=f1_before,
            f1_after=f1_before,
            grid_size=0,
        )
    best_offset = 0.0
    best_f1 = f1_before
    grid = search.offsets()
    for offset in grid:
        shifted = shift_events(pred, offset)
        result = greedy_match(shifted, gold, tolerance_s=tolerance_s, marker_filter=marker_filter)
        value = result.rates()[2]
        if value > best_f1:
            best_f1 = value
            best_offset = float(offset)
    return PhaseOutcome(
        searched=True,
        offset_s=best_offset,
        f1_before=f1_before,
        f1_after=best_f1,
        grid_size=len(grid),
    )


def phase_offsets(pairs: Sequence[MatchPair]) -> tuple[float, ...]:
    """匹配对的有符号时间偏移（pred - gold，秒）；**独立读数**，不参与匹配。

    全谱相位偏移的「匹配后中位/均值偏移」即本序列的中位/均值（plan §4.3 要求
    把相位偏移作为**独立字段**报出，不得混进 F1）。
    """
    return tuple(float(pair.delta_s) for pair in pairs)


__all__ = [
    "MarkerFilter",
    "MatchPair",
    "MatchResult",
    "PhaseOutcome",
    "greedy_match",
    "marker_equal",
    "phase_offsets",
    "precision_recall_f1",
    "search_phase_offset",
    "shift_events",
]
