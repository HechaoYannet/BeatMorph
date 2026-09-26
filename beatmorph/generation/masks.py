"""训练遮盖通道的构造与诊断（Plan 04 §4.2、里程碑 M4/M5）。

三条硬要求（RFC-0029 §3.3-1/2）：

1. **遮盖通道显式存在**：True = 被遮盖（待补全），False = 已观测；模型据此区分
   「此处无 note」与「此处被遮盖」。
2. **按事件遮盖**（`granularity="event"`，默认）：遮盖单位是 **note 事件**，不是帧也不是格。
   一个 Hold 事件连同它的 **hold-end 配对**一起遮，且同一 `(k, x 桶, s)` 纤维上的
   配对点**不得只遮一半**。
3. **遮盖比例 r 的统计正确性**：`r` = 被遮盖事件数 / 总事件数（**按事件计**，
   桶内计数 n_j >= 2 的格子按 n_j 个事件计），并且是训练损失重标定的输入。

⚠️ **mask 泄漏（本模块在 M3 门禁上抓到的范式级缺陷；plan 04 §9-18）**：
若遮盖**只落在有事件的格子**上（`token_block=False`，即 plan 04 §4.2 的**字面**口径），
则 `mask_leak == 1.0`——`occlusion == 1` 恰好等价于「此处有事件」，
模型只要「在 mask=1 处放质量」就能把事件项打到接近最优，**完全不需要看音频与
判定线事件轨**；实测 G2 打乱标签对照因此失效（真实与打乱两臂 loss 逐位相同）。
因此默认 `token_block=True`：先按事件选单位，再把遮盖**扩张到整个 `(k, tau)` token
（含该 token 的空格）**，使 mask 只表示「这一段未被观测」而不再指出事件位置。
`token_block=False` 保留为**泄漏度量与消融臂**（它同时是「按格遮盖」的上界情形）。

本模块**不实现**任何时间换算、不做任何损失重加权——前者归 `beatmorph/field/`，
后者归 `beatmorph/generation/losses.py`。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from beatmorph.core.contracts.phigros import NoteType
from beatmorph.field.target import CHANNEL_INDEX, HOLD_END_CHANNEL

#: 遮盖粒度（plan 04 §4.5「遮盖粒度」消融；默认必须是 "event"）
Granularity = Literal["event", "frame", "cell"]

#: 诊断「抄邻居」的默认 tau 邻域半径（格）
DEFAULT_LEAK_WINDOW: int = 1


@dataclass(frozen=True, slots=True)
class OcclusionStats:
    """遮盖统计（训练日志与 M5 消融报告用；**不含任何权重**）。

    Attributes:
        granularity: 遮盖粒度。
        requested_ratio: 请求的遮盖比例。
        ratio: 实际遮盖比例 r = n_occluded / n_events（重标定用的就是它）。
        n_events: 桶内计数总和（Hold 起点与终点各算一个事件）。
        n_occluded: 被遮盖的事件数。
        n_units: 遮盖单位数（事件单位 / 帧 / 格）。
        n_units_occluded: 被选中的单位数。
        n_hold_pairs: 成功配对的 hold 起止对数。
        n_hold_unpaired: 未配对的 hold 事件数（起点或终点；不修复，如实报告）。
        neighbor_leak: 「抄邻居」诊断——被遮盖事件中，同一 `(k, x, s, c)` 纤维在
            tau 邻域内存在**未遮盖**事件的比例（M5 的消融指标）。
        n_masked_cells: 被遮盖的**格子数**（不是事件数）。
        mask_leak: **遮盖通道的信息泄漏率** = 被遮盖格子中含事件的比例。
            字面口径（只遮有事件的格子）恒为 1.0 —— 那正是补全任务退化
            （照抄 mask 即可）的机器可验证标志；token 口径远小于 1。
        n_masked_tokens: 被遮盖的 `(k, tau)` token 数。
        mask_leak_tokens: 被遮盖 **token** 中含事件的比例（补全任务是否退化的主判据）。
            **信息中性**的诊断目标 = 全谱的 `事件 token 数 / 全部 token 数`：
            高于它就意味着 mask 在替模型回答「这里有没有事件」。
    """

    granularity: Granularity
    requested_ratio: float
    ratio: float
    n_events: int
    n_occluded: int
    n_units: int
    n_units_occluded: int
    n_hold_pairs: int
    n_hold_unpaired: int
    neighbor_leak: float
    n_masked_cells: int = 0
    mask_leak: float = 0.0
    n_masked_tokens: int = 0
    mask_leak_tokens: float = 0.0

    def format(self) -> str:
        """渲染成可直接进训练日志的多行文本。"""
        return (
            f"遮盖：粒度={self.granularity}，请求 r={self.requested_ratio:.3f}，"
            f"实际 r={self.ratio:.4f}（{self.n_occluded}/{self.n_events} 事件，"
            f"{self.n_units_occluded}/{self.n_units} 单位） | "
            f"hold 配对 {self.n_hold_pairs} 对 / 未配对 {self.n_hold_unpaired} 事件 | "
            f"mask 泄漏率 格 {self.mask_leak:.4f} / token {self.mask_leak_tokens:.4f}"
            f"（{self.n_masked_cells} 格，{self.n_masked_tokens} token） | "
            f"抄邻居诊断 {self.neighbor_leak:.4f}"
        )


def _check_counts(counts: Tensor) -> None:
    if counts.dim() < 5:
        raise ValueError(
            f"counts 至少要是 (K, T, X, S, C) 或 (B, K, T, X, S, C)，得到 {tuple(counts.shape)}",
        )


def _hold_pairs(counts: Tensor) -> tuple[list[tuple[int, int]], int]:
    """返回 (配对列表, 未配对事件数)，索引是 counts.reshape(-1) 的平坦下标。

    配对规则（plan 04 §4.2）：同一 `(k, x 桶, s)` 纤维内，按时间顺序把 hold 起点与
    其后最近的未占用 hold 终点配对；配不上的（畸形 Hold / 终点越界）**不修复**，
    只计入 unpaired（如实报告，修复属 plan 05）。
    """
    shape = counts.shape
    t_dim, x_dim, s_dim = int(shape[1]), int(shape[2]), int(shape[3])
    channels = int(shape[4])
    hold = int(CHANNEL_INDEX[NoteType.HOLD])
    starts = torch.nonzero(counts[..., hold] > 0, as_tuple=False)
    ends = torch.nonzero(counts[..., HOLD_END_CHANNEL] > 0, as_tuple=False)
    ends_by_fiber: dict[tuple[int, int, int], list[int]] = {}
    for row in ends.tolist():
        ends_by_fiber.setdefault((int(row[0]), int(row[2]), int(row[3])), []).append(int(row[1]))
    for values in ends_by_fiber.values():
        values.sort()

    def flat(k: int, t: int, x: int, s: int, c: int) -> int:
        return ((((k * t_dim + t) * x_dim + x) * s_dim + s) * channels) + c

    used: dict[tuple[int, int, int], set[int]] = {}
    pairs: list[tuple[int, int]] = []
    starts_sorted = sorted(tuple(int(value) for value in row) for row in starts.tolist())
    for k, t, x, s in starts_sorted:
        key = (k, x, s)
        taken = used.setdefault(key, set())
        for index, end_time in enumerate(ends_by_fiber.get(key, [])):
            if index in taken or end_time <= t:
                continue
            taken.add(index)
            pairs.append((flat(k, t, x, s, hold), flat(k, end_time, x, s, HOLD_END_CHANNEL)))
            break
    n_ends = sum(len(values) for values in ends_by_fiber.values())
    unpaired = (len(starts_sorted) - len(pairs)) + (n_ends - len(pairs))
    return pairs, unpaired


def _units(
    counts: Tensor,
    granularity: Granularity,
    pairs: list[tuple[int, int]],
) -> tuple[list[list[int]], list[float]]:
    """把一张 `(K, T, X, S, C)` 计数张量切成遮盖单位（平坦下标 + 事件权重）。"""
    flat_counts = counts.reshape(-1)
    if granularity == "event":
        index_of_pair: dict[int, int] = {}
        units: list[list[int]] = []
        weights: list[float] = []
        for left, right in pairs:
            index_of_pair[left] = len(units)
            index_of_pair[right] = len(units)
            units.append([left, right])
            weights.append(float(flat_counts[left].item()) + float(flat_counts[right].item()))
        for cell in torch.nonzero(flat_counts > 0, as_tuple=False).reshape(-1).tolist():
            if int(cell) in index_of_pair:
                continue
            units.append([int(cell)])
            weights.append(float(flat_counts[cell].item()))
        return units, weights
    if granularity == "cell":
        cells = torch.nonzero(flat_counts > 0, as_tuple=False).reshape(-1).tolist()
        return [[int(cell)] for cell in cells], [float(flat_counts[cell].item()) for cell in cells]
    if granularity == "frame":
        k_dim, t_dim = int(counts.shape[0]), int(counts.shape[1])
        per_frame = counts.reshape(k_dim, t_dim, -1).sum(dim=-1)
        cells_per_frame = int(counts.shape[2]) * int(counts.shape[3]) * int(counts.shape[4])
        units = []
        weights = []
        for k, t in torch.nonzero(per_frame > 0, as_tuple=False).tolist():
            base = (int(k) * t_dim + int(t)) * cells_per_frame
            units.append([base + offset for offset in range(cells_per_frame)])
            weights.append(float(per_frame[k, t].item()))
        return units, weights
    raise ValueError(f"未知的遮盖粒度：{granularity!r}")


def flat_counts_sum(counts: Tensor, occlusion: Tensor) -> float:
    """被遮盖的事件数 = sum(counts * occlusion)（桶内计数按 n_j 计）。"""
    weights = counts.to(dtype=torch.float32) * occlusion.to(dtype=torch.float32)
    return float(weights.sum().item())


def occluded_event_share(counts: Tensor, occlusion: Tensor | None) -> float:
    """r = 被遮盖事件数 / 总事件数（**losses.py 重标定的唯一输入**）。

    总事件数为 0 时定义为 0.0（**不是** 0/0），此时遮盖没有意义。
    """
    total = float(counts.to(dtype=torch.float32).sum().item())
    if total <= 0.0 or occlusion is None:
        return 0.0
    return flat_counts_sum(counts, occlusion) / total


def neighbor_leak_rate(
    counts: Tensor,
    occlusion: Tensor,
    *,
    window: int = DEFAULT_LEAK_WINDOW,
) -> float:
    """「抄邻居」诊断（plan 04 §4.5 / M5）。

    定义：被遮盖事件中，**同一 `(k, x, s, c)` 纤维**在 tau 的 ±window 格内存在
    未遮盖事件的比例。按帧/按格遮盖时该指标高，意味着模型可以把邻格的观测直接抄过来，
    而不需要学任何补全能力。没有事件时定义为 0.0。
    """
    events = counts > 0
    hidden = events & occlusion
    total_hidden = int(hidden.sum().item())
    if total_hidden == 0:
        return 0.0
    visible = events & (~occlusion)
    neighbourhood = visible.clone()
    for shift in range(1, int(window) + 1):
        neighbourhood[..., shift:, :, :, :] |= visible[..., :-shift, :, :, :]
        neighbourhood[..., :-shift, :, :, :] |= visible[..., shift:, :, :, :]
    leaked = int((hidden & neighbourhood).sum().item())
    return leaked / total_hidden


def dilute_with_empty_tokens(counts: Tensor, occlusion: Tensor, *, seed: int) -> Tensor:
    """把**遮盖稀释到信息中性**：再遮一批空格 token，使「被遮 token 含事件的比例」≈ 全谱事件密度。

    为什么必须稀释：遮盖是**以事件为锚**选出来的（plan 04 §4.2 要求 hold 配对同遮），
    因此被遮 token 天然偏向「含事件」。若不稀释，`mask == 1` 几乎等价于「此处有事件」，
    模型只需「在 mask=1 处放质量」即可（事件项接近最优），**完全不需要看条件**——
    补全任务退化。稀释后，模型对「这个被遮 token 到底有没有事件」只能靠上下文与条件判断，
    补全才有内容可学。

    稀释目标是**信息中性**：令被遮 token 中含事件的比例回到全谱的
    `事件 token 数 / 全部 token 数`；这只增加**空格**的遮盖，
    因此**不改变**被监督的事件集合与遮盖比例 r。
    """
    k_dim, t_dim = int(counts.shape[0]), int(counts.shape[1])
    per_token_events = counts.reshape(k_dim, t_dim, -1).sum(dim=-1) > 0
    masked_token = occlusion.reshape(k_dim, t_dim, -1).any(dim=-1)
    total_tokens = k_dim * t_dim
    total_event_tokens = int(per_token_events.sum().item())
    n_event_masked = int((masked_token & per_token_events).sum().item())
    if total_event_tokens == 0 or n_event_masked == 0:
        return occlusion
    base_density = total_event_tokens / total_tokens
    target_masked = min(total_tokens, round(n_event_masked / base_density))
    need = target_masked - int(masked_token.sum().item())
    if need <= 0:
        return occlusion
    # 只能稀释**真正没有事件**的 token：否则会把未选中的事件也遮进去，
    # 实际遮盖比例 r 会超过请求值（遮盖比例契约会被破坏）。
    candidates = torch.nonzero(~masked_token & ~per_token_events, as_tuple=False)
    if candidates.numel() == 0:
        return occlusion
    order = np.random.default_rng(seed).permutation(int(candidates.shape[0]))[:need]
    picked = candidates[torch.as_tensor(order, dtype=torch.long)]
    new_tokens = torch.zeros_like(masked_token)
    new_tokens[picked[:, 0], picked[:, 1]] = True
    diluted = masked_token | new_tokens
    return diluted.reshape(k_dim, t_dim, 1, 1, 1).expand_as(counts).clone()


def close_hold_pairs(counts: Tensor, occlusion: Tensor) -> Tensor:
    """把遮盖在 **token 级**收口：Hold 配对两端的 token 必须同遮或同不遮。

    动机（实测 2026-09-27，真实谱面）：`expand_to_tokens` 按 `(k, tau)` token 扩张，
    而长 Hold 的起点与终点可以落在不同 token。只要起点所在 token 里**另有**一个事件被选中，
    扩张就把该 token 整体遮住，而终点所在的 token 未被选中 ⇒ 配对只被遮了一半。

    收口口径是 **token 级**而不是格子级：格子级补遮会在未被遮的 token 里留下
    「被遮的孤立格子」，那正是 `expand_to_tokens` 要消除的 mask 泄漏。

    Args:
        counts: `(K, T, X, S, C)` 计数张量。
        occlusion: 同形状的遮盖（**假定已是 token 级**）。

    Returns:
        同形状的遮盖；未发生拆分时**原样返回**（不复制）。
    """
    pairs, _unpaired = _hold_pairs(counts)
    if not pairs:
        return occlusion
    shape = counts.shape
    k_dim, t_dim = int(shape[0]), int(shape[1])
    cells_per_token = int(shape[2]) * int(shape[3]) * int(shape[4])
    per_token = occlusion.reshape(k_dim, t_dim, -1).any(dim=-1)
    flat_tokens = per_token.reshape(-1)
    changed = False
    for left, right in pairs:
        left_token = left // cells_per_token
        right_token = right // cells_per_token
        if bool(flat_tokens[left_token]) != bool(flat_tokens[right_token]):
            flat_tokens[left_token] = True
            flat_tokens[right_token] = True
            changed = True
    if not changed:
        return occlusion
    return per_token.reshape(k_dim, t_dim, 1, 1, 1).expand_as(counts).clone()


def expand_to_tokens(counts: Tensor, occlusion: Tensor) -> Tensor:
    """把格子级遮盖扩张到整个 `(k, tau)` token（含该 token 的全部空格）。

    这是**防泄漏**的关键一步：格子级遮盖会把「哪些格子有事件」直接暴露在 mask 通道里，
    补全任务于是退化成「照抄 mask」（`mask_leak == 1.0`）。扩张后 mask 只表示
    「这一段未被观测」，模型必须靠上下文与条件把事件位置补出来。
    """
    k_dim, t_dim = int(counts.shape[0]), int(counts.shape[1])
    per_token = occlusion.reshape(k_dim, t_dim, -1).any(dim=-1)
    return per_token.reshape(k_dim, t_dim, 1, 1, 1).expand_as(counts).clone()


def build_occlusion(
    counts: Tensor,
    *,
    ratio: float,
    granularity: Granularity = "event",
    token_block: bool = True,
    dilute: bool = True,
    seed: int = 0,
) -> tuple[Tensor, OcclusionStats]:
    """为一张 `(K, T, X, S, C)` 计数张量构造遮盖通道（True = 被遮盖）。

    Args:
        counts: 桶内计数（plan 03 产出），非负。
        ratio: 请求遮盖的事件比例 r in [0, 1]；实际比例由单位粒度决定，如实回报。
        granularity: "event"（默认，契约路径）/ "frame" / "cell"（消融臂）。
        token_block: 是否把遮盖扩张到整个 `(k, tau)` token（**默认 True，防 mask 泄漏**）。
            设为 False 得到 plan 04 §4.2 的**字面**口径（只遮有事件的格子）——
            它会令 `mask_leak == 1.0`、补全任务退化成照抄 mask，仅用于泄漏度量与消融。
        seed: 选择遮盖单位的随机种子（同一 seed 结果可复现）。
    """
    _check_counts(counts)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"ratio 必须在 [0, 1]，得到 {ratio!r}")
    total_events = float(counts.sum().item())
    pairs, unpaired = _hold_pairs(counts)
    occlusion = torch.zeros(counts.shape, dtype=torch.bool)
    units, weights = _units(counts, granularity, pairs)
    chosen: list[int] = []
    if total_events > 0.0 and units and ratio > 0.0:
        order = np.random.default_rng(seed).permutation(len(units))
        target = float(ratio) * total_events
        accumulated = 0.0
        for position in order:
            if accumulated >= target:
                break
            index = int(position)
            chosen.append(index)
            accumulated += weights[index]
        flat = occlusion.reshape(-1)
        for index in chosen:
            flat[torch.as_tensor(units[index], dtype=torch.long)] = True
        if token_block:
            occlusion = expand_to_tokens(counts, occlusion)
            if dilute:
                occlusion = dilute_with_empty_tokens(counts, occlusion, seed=seed + 977)
            # ⚠️ 扩张是**按 (k, tau) token** 做的，而一个 Hold 的起点与终点可以落在
            # **不同 token**（长 Hold / 慢段落）。同一 token 里的另一个事件被选中时，
            # 扩张会把这个 token 整体遮住 ⇒「配对只有一端被遮」。实测（2026-09-27，真实谱面）：
            # 数据集在 index=12 的窗口上直接抛「hold 配对点被拆散」，真实数据通路的门禁装配失败。
            # 因此扩张之后必须**收口**（token 级，不产生半遮 token）。
            # ⚠️ 只对**契约路径** `granularity="event"` 收口：`cell` / `frame` 是消融臂，
            # 它们的「实际 r ≈ 请求 r」契约优先于「Hold 配对不拆散」（那两条臂本就不以配对为单位）。
            if granularity == "event":
                occlusion = close_hold_pairs(counts, occlusion)
    n_occluded = int(flat_counts_sum(counts, occlusion))
    n_masked_cells = int(occlusion.sum().item())
    n_masked_with_event = int((occlusion & (counts > 0)).sum().item())
    masked_tokens = occlusion.reshape(int(counts.shape[0]), int(counts.shape[1]), -1).any(dim=-1)
    event_tokens = counts.reshape(int(counts.shape[0]), int(counts.shape[1]), -1).sum(dim=-1) > 0
    n_masked_tokens = int(masked_tokens.sum().item())
    n_masked_event_tokens = int((masked_tokens & event_tokens).sum().item())
    return occlusion, OcclusionStats(
        granularity=granularity,
        requested_ratio=float(ratio),
        ratio=(n_occluded / total_events) if total_events > 0.0 else 0.0,
        n_events=int(total_events),
        n_occluded=n_occluded,
        n_units=len(units),
        n_units_occluded=len(chosen),
        n_hold_pairs=len(pairs),
        n_hold_unpaired=unpaired,
        neighbor_leak=neighbor_leak_rate(counts, occlusion),
        n_masked_cells=n_masked_cells,
        mask_leak=(n_masked_with_event / n_masked_cells) if n_masked_cells else 0.0,
        n_masked_tokens=n_masked_tokens,
        mask_leak_tokens=(n_masked_event_tokens / n_masked_tokens) if n_masked_tokens else 0.0,
    )


def build_occlusion_batch(
    counts: Tensor,
    *,
    ratio: float,
    granularity: Granularity = "event",
    token_block: bool = True,
    dilute: bool = True,
    seed: int = 0,
    line_mask: Tensor | None = None,
) -> tuple[Tensor, OcclusionStats]:
    """对 `(B, K, T, X, S, C)` 逐样本构造遮盖通道，并汇总统计。

    padding 线（line_mask=False）**不参与**遮盖选择——它们的计数本就不该存在；
    若传入的计数在 padding 线上非零，直接抛错（静默忽略会掩盖上游的错位）。
    """
    _check_counts(counts)
    if counts.dim() != 6:
        raise ValueError(
            f"build_occlusion_batch 需要 (B, K, T, X, S, C)，得到 {tuple(counts.shape)}",
        )
    batch_size = int(counts.shape[0])
    if line_mask is not None:
        invalid = counts.to(dtype=torch.float32) * (~line_mask).reshape(
            batch_size,
            -1,
            1,
            1,
            1,
            1,
        )
        if float(invalid.sum().item()) > 0.0:
            raise ValueError("padding 线（line_mask=False）上出现非零计数：上游错位，拒绝静默忽略")
    masks: list[Tensor] = []
    stats: list[OcclusionStats] = []
    for index in range(batch_size):
        mask, one = build_occlusion(
            counts[index],
            ratio=ratio,
            granularity=granularity,
            token_block=token_block,
            dilute=dilute,
            seed=seed + index,
        )
        if line_mask is not None:
            # padding 线永远不参与遮盖（稀释是随机选 token 的，必须在此显式收回；
            # 否则「padding 线零贡献」的契约会被稀释悄悄破坏）。
            mask = mask & line_mask[index].reshape(-1, 1, 1, 1, 1)
        masks.append(mask)
        stats.append(one)
    n_events = sum(one.n_events for one in stats)
    n_occluded = sum(one.n_occluded for one in stats)
    n_masked_cells = sum(one.n_masked_cells for one in stats)
    aggregated = OcclusionStats(
        granularity=granularity,
        requested_ratio=float(ratio),
        ratio=(n_occluded / n_events) if n_events else 0.0,
        n_events=n_events,
        n_occluded=n_occluded,
        n_units=sum(one.n_units for one in stats),
        n_units_occluded=sum(one.n_units_occluded for one in stats),
        n_hold_pairs=sum(one.n_hold_pairs for one in stats),
        n_hold_unpaired=sum(one.n_hold_unpaired for one in stats),
        neighbor_leak=(
            sum(one.neighbor_leak * one.n_occluded for one in stats) / n_occluded
            if n_occluded
            else 0.0
        ),
        n_masked_cells=n_masked_cells,
        mask_leak=(
            sum(one.mask_leak * one.n_masked_cells for one in stats) / n_masked_cells
            if n_masked_cells
            else 0.0
        ),
        n_masked_tokens=sum(one.n_masked_tokens for one in stats),
        mask_leak_tokens=(
            sum(one.mask_leak_tokens * one.n_masked_tokens for one in stats) / n_masked_tokens
            if (n_masked_tokens := sum(one.n_masked_tokens for one in stats))
            else 0.0
        ),
    )
    return torch.stack(masks, dim=0), aggregated


def assert_hold_pairs_not_split(counts: Tensor, occlusion: Tensor) -> None:
    """契约断言：同一 `(k, x 桶, s)` 纤维上的 hold 配对点**不得只遮一半**。

    配对成功的 hold 起止对必须**同遮或同不遮**；未配对的（畸形 Hold）不受此约束，
    但会被 `OcclusionStats.n_hold_unpaired` 如实报出（不修复，修复属 plan 05）。
    """
    flat_occlusion = occlusion.reshape(-1)
    for left, right in _hold_pairs(counts)[0]:
        if bool(flat_occlusion[left].item()) != bool(flat_occlusion[right].item()):
            raise AssertionError(
                f"hold 配对点被拆散：起止平坦下标 {left} / {right} 的遮盖状态不一致",
            )


def mask_semantics(
    counts: Tensor,
    occlusion: Tensor,
    line_mask: Sequence[bool] | None = None,
) -> dict[str, int]:
    """三个 mask 的语义分离自检：返回各自的统计（测试与日志用，**不做任何运算**）。"""
    weights = counts.to(dtype=torch.float32) * occlusion.to(dtype=torch.float32)
    stats = {
        "n_occluded_cells": int((occlusion & (counts > 0)).sum().item()),
        "n_observed_cells": int(((~occlusion) & (counts > 0)).sum().item()),
        "n_occluded_events": int(weights.sum().item()),
        "n_masked_cells": int(occlusion.sum().item()),
    }
    if line_mask is not None:
        stats["n_padding_lines"] = int(sum(1 for flag in line_mask if not flag))
    return stats


__all__ = [
    "DEFAULT_LEAK_WINDOW",
    "Granularity",
    "OcclusionStats",
    "assert_hold_pairs_not_split",
    "build_occlusion",
    "build_occlusion_batch",
    "close_hold_pairs",
    "dilute_with_empty_tokens",
    "expand_to_tokens",
    "flat_counts_sum",
    "mask_semantics",
    "neighbor_leak_rate",
    "occluded_event_share",
]
