"""Plan 04 M4/M5：遮盖通道的粒度正确性、比例统计与三个 mask 的语义分离。

契约来源（docs/plans/04-generation.md）：
- §3.2 张量形状：`OcclusionMask` (B,K,T,X,S,C) 与 `line_mask` / `range_mask` **语义分离**；
- §4.2 Decoder：按事件遮盖，Hold 起止配对点不得只遮一半；按帧/按格只作消融臂；
- §4.3 训练目标：遮盖比例 r = 被遮盖事件数 / 总事件数是重标定的唯一输入；
- §4.5 消融：遮盖粒度（event / frame / cell）与「抄邻居」诊断指标；
- §6.2 的 M4（遮盖重标定契约）与 M5（按事件 vs 按帧遮盖消融）。

默认 CI：无权重、无 GPU、无网络。所有规模量（tau 格数、x 桶数）均由
`beatmorph.core.contracts` 常量与 `FieldGrid` 派生，**不出现任何硬编码物理常量**。
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np
import pytest
import torch
from torch import Tensor

from beatmorph.core.contracts.phigros import SUBDIVISIONS_PER_BEAT, NoteType
from beatmorph.field.grid import FieldGrid
from beatmorph.field.target import CHANNEL_INDEX, HOLD_END_CHANNEL
from beatmorph.generation.masks import (
    DEFAULT_LEAK_WINDOW,
    Granularity,
    assert_hold_pairs_not_split,
    build_occlusion,
    build_occlusion_batch,
    close_hold_pairs,
    flat_counts_sum,
    mask_semantics,
    neighbor_leak_rate,
    occluded_event_share,
)
from tests.unit.generation._builders import make_batch, make_grid

# ── 测试规模（全部派生 / 超参，禁止写死物理量）──────────────────────────
#: tau 格数：1 拍，由 SUBDIVISIONS_PER_BEAT 派生（**不写 48**）
T_BINS: int = SUBDIVISIONS_PER_BEAT
#: x 桶数（测试用超参，与物理常量无关）
X_BINS: int = 8
#: 测试用判定线条数
K_LINES: int = 3
#: batch 口径的规模
BATCH_SIZE: int = 2
BATCH_K: int = 4
#: batch 口径下的有效线；其余为 padding（line_mask=False）
ACTIVE_LINES: tuple[int, ...] = (0, 1)
#: 请求遮盖比例的档位（M5 要求「粒度 x 比例」各报一组）
RATIOS: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
#: 单点事件所在的通道（全部经 TYPE_CHANNELS 派生，不写死通道下标）
_TAP: int = int(CHANNEL_INDEX[NoteType.TAP])
_DRAG: int = int(CHANNEL_INDEX[NoteType.DRAG])
_HOLD: int = int(CHANNEL_INDEX[NoteType.HOLD])
#: 单点事件条数（与 K_LINES 个 Hold 起止对合计 30 个事件）
_TAPS: int = 4 * T_BINS // 8
#: 同格多事件的计数（验收「n_j >= 2 的格子按 n_j 个事件计」）
MULTI_EVENT_CELL: int = 3

# ══════════════════════════════════════════════════════════════
# 回归记录（本测试曾以 strict xfail 标记一个真实缺陷，现已修复）
# ══════════════════════════════════════════════════════════════
#: _units 的 frame 分支曾把 cells_per_frame 取成 per_frame.shape[-1]（= T 维），
#: 整帧单位的平坦下标全错位：r=1.0 时实际 ratio 只有 0.0。现已取 X*S*C。


# ══════════════════════════════════════════════════════════════
# 构造器（全部确定性，不依赖随机权重）
# ══════════════════════════════════════════════════════════════
def _grid(*, t_bins: int = T_BINS) -> FieldGrid:
    """测试网格：x 轴固定 X_BINS 桶（小张量），tau 轴长度由参数派生。"""
    return make_grid(t_bins=t_bins, x_bins=X_BINS)


def _empty(k: int, grid: FieldGrid) -> Tensor:
    """(K, T, X, S, C) 全零桶内计数（int16，与 plan 03 的 FieldTarget 同 dtype）。"""
    return torch.zeros(k, grid.t_bins, grid.x_bins, grid.sides, grid.channels, dtype=torch.int16)


def _put(counts: Tensor, *, k: int, t: int, x: int, s: int, c: int, n: int = 1) -> None:
    """在 (k, t, x, s, c) 处累加 n 个事件；n >= 2 即同格多事件（n_j >= 2）。"""
    counts[k, t, x, s, c] += n


def _hold(counts: Tensor, *, k: int, start: int, end: int, x: int, s: int) -> None:
    """放一个合法 Hold：起点进 hold 通道、终点进 hold_end 通道（同一 (k, x, s) 纤维）。"""
    _put(counts, k=k, t=start, x=x, s=s, c=_HOLD)
    _put(counts, k=k, t=end, x=x, s=s, c=HOLD_END_CHANNEL)


def _binary_counts() -> Tensor:
    """0/1 计数：每条线一个 Hold 起止对 + 若干单点事件。

    刻意保持全 0/1，使「遮盖单位权重」只由粒度决定，从而给出**紧的**比例容差；
    n_j >= 2 的加权口径由 _weighted_counts 与专门测试覆盖。
    """
    grid = _grid()
    counts = _empty(K_LINES, grid)
    span = grid.t_bins // (2 * K_LINES)
    for index in range(K_LINES):
        _hold(
            counts,
            k=index,
            start=index,
            end=index + (index + 1) * span,
            x=index,
            s=index % grid.sides,
        )
    stride = max(grid.t_bins // _TAPS, 1)
    for step in range(_TAPS):
        _put(
            counts,
            k=step % K_LINES,
            t=(step * stride) % grid.t_bins,
            x=(2 * step + 1) % grid.x_bins,
            s=(step + 1) % grid.sides,
            c=_TAP if step % 2 == 0 else _DRAG,
        )
    assert int(counts.max().item()) == 1, "本构造器必须产出 0/1 计数（容差推导依赖它）"
    return counts


def _weighted_counts() -> Tensor:
    """在 0/1 计数之上把一格抬到 MULTI_EVENT_CELL，用于验收按 n_j 加权的口径。"""
    counts = _binary_counts().clone()
    cells = torch.nonzero(counts[1] > 0, as_tuple=False)
    assert int(cells.shape[0]) > 0, "线 1 上必须有事件可供抬升"
    t, x, s, c = (int(value) for value in cells[0])
    counts[1, t, x, s, c] = MULTI_EVENT_CELL
    return counts


def _ratio_tolerance(counts: Tensor, granularity: Granularity) -> float:
    """逐单元贪心选择带来的比例超调上界 = 最重单位的事件数 / 总事件数。

    - cell：单位 = 一个格，最重 = 格内最大计数；
    - event：单位 = 一个格或一对 Hold 起止点，最重 <= 2 x 格内最大计数；
    - frame：单位 = 一整帧（同一 (k, tau) 的全部格），最重 = 单帧事件数。
    """
    total = float(counts.sum().item())
    heaviest_cell = float(counts.max().item())
    if granularity == "frame":
        per_frame = counts.reshape(counts.shape[0], counts.shape[1], -1).sum(dim=-1)
        heaviest = float(per_frame.max().item())
    elif granularity == "event":
        heaviest = 2.0 * heaviest_cell
    else:
        heaviest = heaviest_cell
    return heaviest / total


def _batch_counts(grid: FieldGrid) -> tuple[Tensor, Tensor]:
    """返回 (counts, line_mask)：counts 已在 padding 线上清零（合法的混合 K 输入）。"""
    batch = make_batch(
        batch=BATCH_SIZE,
        k=BATCH_K,
        grid=grid,
        events=4 * BATCH_SIZE,
        holds=BATCH_SIZE,
        seed=11,
        active_lines=ACTIVE_LINES,
    )
    keep = batch.line_mask.reshape(*batch.line_mask.shape, 1, 1, 1, 1)
    return batch.counts.masked_fill(~keep, 0), batch.line_mask


def _cell_shape(grid: FieldGrid) -> tuple[int, ...]:
    """(B, K, T, X, S, C)（本文件口径）。"""
    return (BATCH_SIZE, BATCH_K, grid.t_bins, grid.x_bins, grid.sides, grid.channels)


# ══════════════════════════════════════════════════════════════
# §4.2 按事件遮盖：Hold 起止配对点不得只遮一半
# ══════════════════════════════════════════════════════════════
@pytest.mark.parametrize("ratio", RATIOS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_event_granularity_never_splits_hold_pairs(ratio: float, seed: int) -> None:
    """Plan 04 §4.2：按事件遮盖时同一 (k, x 桶, s) 纤维上的 Hold 起止点必须同遮或同不遮。"""
    counts = _binary_counts()
    mask, stats = build_occlusion(counts, ratio=ratio, granularity="event", seed=seed)
    # 门禁不得空转：构造的 Hold 起止对必须全部配对成功
    assert stats.n_hold_pairs == K_LINES
    assert stats.n_hold_unpaired == 0
    assert_hold_pairs_not_split(counts, mask)


def test_long_hold_spanning_tokens_is_closed_after_expansion() -> None:
    """回归（实测 2026-09-27，真实谱面）：扩张到 token 之后，跨 token 的 Hold 配对必须收口。

    机制：Hold 的起点在 token t1、终点在 token t2（长 Hold）。t1 里**另有**一个独立事件
    被选中时，`expand_to_tokens` 会把 t1 整体遮住，而 t2 不动 ⇒ 配对只遮一半。
    这里先复现这个半遮形态，再断言 `close_hold_pairs` 在 **token 级**把它收口。
    """
    grid = _grid()
    counts = _empty(1, grid)
    start, end, x_bin, side = 1, grid.t_bins - 1, 0, 0
    other_t = 1  # 与起点同 token 的独立事件（终点在另一个 token）
    counts[0, start, x_bin, side, _TAP] = 1
    counts[0, other_t, x_bin + 1, side, _TAP] = 1
    _hold(counts, k=0, start=start, end=end, x=x_bin, s=side)

    # 半遮形态：只遮起点所在 token（模拟扩张把 t1 整体遮住、t2 未选中）
    half = torch.zeros_like(counts, dtype=torch.bool)
    half[0, start, :, :, :] = True
    with pytest.raises(AssertionError, match="hold 配对点被拆散"):
        assert_hold_pairs_not_split(counts, half)

    closed = close_hold_pairs(counts, half)
    assert_hold_pairs_not_split(counts, closed)
    # 收口是 **token 级**：终点所在 token 整体被遮（不留半遮 token 的泄漏）
    assert bool(closed[0, end, :, :, :].all().item())
    # 未发生拆分时原样返回（不复制、不改语义）
    both = torch.zeros_like(counts, dtype=torch.bool)
    both[0, start, :, :, :] = True
    both[0, end, :, :, :] = True
    assert close_hold_pairs(counts, both) is both


@pytest.mark.parametrize("ratio", RATIOS)
@pytest.mark.parametrize("seed", range(8))
def test_build_occlusion_never_splits_a_multi_token_hold(ratio: float, seed: int) -> None:
    """同一构造下扫多个种子：`build_occlusion` 的产物不得拆散配对（收口在管线内生效）。"""
    grid = _grid()
    counts = _empty(1, grid)
    start, end, x_bin, side = 1, grid.t_bins - 1, 0, 0
    counts[0, start, x_bin, side, _TAP] = 1
    counts[0, start, x_bin + 1, side, _TAP] = 1
    _hold(counts, k=0, start=start, end=end, x=x_bin, s=side)
    mask, stats = build_occlusion(counts, ratio=ratio, granularity="event", seed=seed)
    assert stats.n_hold_pairs == 1
    assert_hold_pairs_not_split(counts, mask)


def test_close_hold_pairs_is_a_fixed_point_on_a_token_chain() -> None:
    """回归（实测 2026-09-27 第四轮，真实批暴露）：收口必须取**传递闭包**，单趟会互相拆散。

    真实数据反例的形态是「两个 Hold 配对共享一个 token」：配对 A=(t10, t19)、
    B=(t19, t21)、C=(t36, t38)，初始只有 token 21 被遮（expand_to_tokens 选中了
    B 的终点所在 token）。单趟先看 A（10 / 19 均未遮 ⇒ 不动），再看 B 时把 19 / 21
    都遮上 —— **A 的一端 19 被 B 顺带遮住而 10 没遮，A 反而被拆散**。
    传递闭包把 A、B 所在的连通分量 {10, 19, 21} 整体遮住，C 的分量不受影响。
    """
    grid = _grid()
    counts = _empty(1, grid)
    # x 纤维互不相同，避免随机单点事件造成配对归属歧义
    _hold(counts, k=0, start=10, end=19, x=0, s=0)
    _hold(counts, k=0, start=19, end=21, x=1, s=0)
    _hold(counts, k=0, start=36, end=38, x=2, s=0)
    seeded = torch.zeros_like(counts, dtype=torch.bool)
    seeded[0, 21, :, :, :] = True  # 只遮 B 的终点所在 token

    closed = close_hold_pairs(counts, seeded)
    assert_hold_pairs_not_split(counts, closed)
    for token in (10, 19, 21):  # 种子所在连通分量整体被遮（token 级）
        assert bool(closed[0, token].all().item()), token
    for token in (36, 38):  # 无种子的分量原样不动
        assert not bool(closed[0, token].any().item()), token


def test_build_occlusion_never_splits_hold_pairs_on_random_chains() -> None:
    """回归（同上）：固定 seed 的随机链式 Hold 上，build_occlusion 的产物必须处处不拆散配对。

    单趟收口的缺陷只在「两个配对共享一个 token」时暴露，随机构造专门覆盖这种链。
    """
    grid = _grid()
    rng = np.random.default_rng(20260927)
    for _ in range(150):
        counts = _empty(2, grid)
        for _ in range(int(rng.integers(1, 4))):
            k = int(rng.integers(0, 2))
            x = int(rng.integers(0, grid.x_bins))
            s = int(rng.integers(0, grid.sides))
            start = int(rng.integers(0, grid.t_bins - 1))
            end = int(rng.integers(start + 1, grid.t_bins))
            _hold(counts, k=k, start=start, end=end, x=x, s=s)
        for _ in range(int(rng.integers(0, 6))):
            k = int(rng.integers(0, 2))
            x = int(rng.integers(0, grid.x_bins))
            s = int(rng.integers(0, grid.sides))
            t = int(rng.integers(0, grid.t_bins))
            c = int(rng.integers(0, grid.channels))
            counts[k, t, x, s, c] = 1
        for seed in range(3):
            mask, _stats = build_occlusion(counts, ratio=0.5, granularity="event", seed=seed)
            assert_hold_pairs_not_split(counts, mask)


def test_hold_pair_gate_rejects_a_split_mask() -> None:
    """门禁自检（反例）：把配对的起止点只遮一半，assert_hold_pairs_not_split 必须抛。"""
    grid = _grid()
    counts = _empty(1, grid)
    start, end, x_bin, side = 1, grid.t_bins - 1, 0, 0
    _hold(counts, k=0, start=start, end=end, x=x_bin, s=side)
    # (a) 同遮：不抛
    both = torch.zeros_like(counts, dtype=torch.bool)
    both[0, start, x_bin, side, _HOLD] = True
    both[0, end, x_bin, side, HOLD_END_CHANNEL] = True
    assert_hold_pairs_not_split(counts, both)
    # (b) 只遮起点：必须抛（否则门禁是空转的）
    half = torch.zeros_like(counts, dtype=torch.bool)
    half[0, start, x_bin, side, _HOLD] = True
    with pytest.raises(AssertionError, match="hold 配对点被拆散"):
        assert_hold_pairs_not_split(counts, half)
    # (c) 只遮终点：同样必须抛
    other = torch.zeros_like(counts, dtype=torch.bool)
    other[0, end, x_bin, side, HOLD_END_CHANNEL] = True
    with pytest.raises(AssertionError, match="hold 配对点被拆散"):
        assert_hold_pairs_not_split(counts, other)


# ══════════════════════════════════════════════════════════════
# §4.3 遮盖比例 r 的统计正确性
# ══════════════════════════════════════════════════════════════
@pytest.mark.parametrize("ratio", RATIOS)
def test_ratio_statistics_equal_event_weighted_share(ratio: float) -> None:
    """Plan 04 §4.3 / M4：r == n_occluded / n_events，且与 occluded_event_share 一致。"""
    counts = _weighted_counts()
    mask, stats = build_occlusion(counts, ratio=ratio, granularity="event", seed=7)
    total = float(counts.sum().item())
    n_occluded = flat_counts_sum(counts, mask)
    assert stats.n_events == int(total)
    assert stats.n_occluded == int(n_occluded)
    assert stats.ratio == pytest.approx(n_occluded / total)
    assert stats.ratio == pytest.approx(occluded_event_share(counts, mask))
    assert stats.requested_ratio == pytest.approx(ratio)
    assert 0.0 <= stats.ratio <= 1.0
    if ratio <= 0.0:
        # r = 0：mask 全 False，且不消耗任何遮盖单位
        assert not bool(mask.any())
        assert stats.n_units_occluded == 0
    if ratio >= 1.0:
        # r = 1：mask 必须覆盖全部非零格
        assert bool(mask[counts > 0].all())
        assert stats.n_units_occluded == stats.n_units
    assert "r=" in stats.format()


def test_multi_event_cells_are_weighted_by_their_count() -> None:
    """Plan 04 §4.3：桶内计数 n_j >= 2 的格必须按 n_j 个事件计入 r（禁止按格去重）。"""
    grid = _grid()
    counts = _empty(1, grid)
    _put(counts, k=0, t=0, x=0, s=0, c=_TAP, n=MULTI_EVENT_CELL)
    mask = torch.zeros_like(counts, dtype=torch.bool)
    mask[0, 0, 0, 0, _TAP] = True
    assert flat_counts_sum(counts, mask) == float(MULTI_EVENT_CELL)
    assert occluded_event_share(counts, mask) == pytest.approx(1.0)
    # 去重版本（把该格当成 1 个事件）给出更小的被遮事件数——证明没有去重
    deduped = torch.zeros_like(counts)
    deduped[0, 0, 0, 0, _TAP] = 1
    assert flat_counts_sum(deduped, mask) == pytest.approx(1.0)
    assert flat_counts_sum(counts, mask) != flat_counts_sum(deduped, mask)


# ══════════════════════════════════════════════════════════════
# §4.5 遮盖粒度消融：三种粒度都必须逼近请求比例
# ══════════════════════════════════════════════════════════════
@pytest.mark.parametrize("requested", [0.25, 0.5])
@pytest.mark.parametrize(
    "granularity",
    [
        "event",
        "cell",
        pytest.param(
            "frame",
        ),
    ],
)
def test_every_granularity_approximates_the_requested_ratio(
    granularity: Granularity,
    requested: float,
) -> None:
    """Plan 04 §4.5 / M5：三种粒度都能构造，且遮盖事件数都接近请求值（frame 以整帧为单位）。"""
    counts = _binary_counts()
    mask, stats = build_occlusion(counts, ratio=requested, granularity=granularity, seed=0)
    assert tuple(mask.shape) == tuple(counts.shape)
    assert mask.dtype == torch.bool
    assert stats.granularity == granularity
    assert stats.n_events == int(counts.sum().item())
    assert stats.ratio == pytest.approx(occluded_event_share(counts, mask))
    tolerance = _ratio_tolerance(counts, granularity)
    assert requested <= stats.ratio <= requested + tolerance, (
        f"{granularity} 粒度在 r={requested} 下实际遮盖 {stats.ratio:.4f}，"
        f"超出 [{requested}, {requested + tolerance:.4f}]"
    )


# ══════════════════════════════════════════════════════════════
# §4.5 / M5「抄邻居」诊断
# ══════════════════════════════════════════════════════════════
def test_neighbor_leak_flags_copyable_twins() -> None:
    """M5：每个被遮事件在相邻 tau 格都有同 (k, x, s, c) 双胞胎时，抄邻居率必须 > 0。"""
    grid = _grid()
    counts = _empty(1, grid)
    mask = torch.zeros_like(counts, dtype=torch.bool)
    for t in range(0, grid.t_bins - 1, 2):
        _put(counts, k=0, t=t, x=0, s=0, c=_TAP)
        _put(counts, k=0, t=t + 1, x=0, s=0, c=_TAP)
        mask[0, t, 0, 0, _TAP] = True  # 只遮双胞胎中的一个，另一个仍可见
    leak = neighbor_leak_rate(counts, mask)
    assert leak == pytest.approx(1.0)
    assert 0.0 <= leak <= 1.0
    # build_occlusion 内嵌的诊断字段必须与独立重算一致（不得各算一套）
    occlusion, stats = build_occlusion(counts, ratio=0.5, granularity="cell", seed=0)
    assert stats.neighbor_leak == pytest.approx(neighbor_leak_rate(counts, occlusion))


def test_neighbor_leak_is_zero_for_isolated_events() -> None:
    """M5 反例：事件彼此间隔超过邻域窗口时，被遮事件没有可抄的邻居 -> leak == 0。"""
    grid = _grid()
    spacing = 2 * DEFAULT_LEAK_WINDOW + 2  # 严格大于窗口，保证互不为邻
    counts = _empty(1, grid)
    mask = torch.zeros_like(counts, dtype=torch.bool)
    times = list(range(0, grid.t_bins, spacing))
    for t in times:
        _put(counts, k=0, t=t, x=0, s=0, c=_TAP)
    for index, t in enumerate(times):
        if index % 2 == 0:
            mask[0, t, 0, 0, _TAP] = True
    # 门禁不得空转：确有事件被遮，也确有事件保持可见
    assert 0 < int(mask.sum().item()) < len(times)
    assert neighbor_leak_rate(counts, mask) == 0.0
    occlusion, stats = build_occlusion(counts, ratio=0.5, granularity="event", seed=1)
    assert 0.0 <= stats.neighbor_leak <= 1.0
    assert stats.neighbor_leak == pytest.approx(neighbor_leak_rate(counts, occlusion))


# ══════════════════════════════════════════════════════════════
# §4.2 batch 口径：padding 线拒绝 + 汇总统计
# ══════════════════════════════════════════════════════════════
def test_occlusion_batch_rejects_counts_on_padding_lines() -> None:
    """Plan 04 §4.2：padding 线（line_mask=False）上出现非零计数必须抛 ValueError。"""
    grid = _grid()
    counts, line_mask = _batch_counts(grid)
    mask, _ = build_occlusion_batch(counts, ratio=0.5, granularity="event", line_mask=line_mask)
    assert tuple(mask.shape) == tuple(counts.shape)
    assert mask.dtype == torch.bool
    padding = (~line_mask).reshape(*line_mask.shape, 1, 1, 1, 1)
    # padding 线永远不参与遮盖选择
    assert int((mask & padding).sum().item()) == 0
    dirty = counts.clone()
    dirty[:, BATCH_K - 1, 0, 0, 0, 0] = 1
    with pytest.raises(ValueError, match="padding"):
        build_occlusion_batch(dirty, ratio=0.5, granularity="event", line_mask=line_mask)


@pytest.mark.parametrize("ratio", [0.25, 0.5, 1.0])
def test_occlusion_batch_aggregate_ratio_is_the_weighted_mean(ratio: float) -> None:
    """Plan 04 §4.2 / M4：批次汇总 ratio == 逐样本比率的加权汇总 == occluded_event_share。"""
    grid = _grid()
    counts, line_mask = _batch_counts(grid)
    mask, stats = build_occlusion_batch(
        counts,
        ratio=ratio,
        granularity="event",
        seed=0,
        line_mask=line_mask,
    )
    per_sample = [
        build_occlusion(counts[index], ratio=ratio, granularity="event", seed=index)
        for index in range(BATCH_SIZE)
    ]
    # 批次口径**额外收回 padding 线的遮盖**（稀释是随机选 token 的，会把遮盖洒到 padding 线上；
    # 而「padding 线零贡献」是硬契约），因此只在有效线上要求与逐样本口径逐位相同。
    expected = torch.stack([one for one, _ in per_sample], dim=0)
    assert torch.equal(mask[line_mask], expected[line_mask])
    assert int(mask[~line_mask].sum().item()) == 0
    n_events = sum(one.n_events for _, one in per_sample)
    n_occluded = sum(one.n_occluded for _, one in per_sample)
    assert stats.n_events == n_events == int(counts.sum().item())
    assert stats.n_occluded == n_occluded
    assert stats.ratio == pytest.approx(n_occluded / n_events)
    assert stats.ratio == pytest.approx(occluded_event_share(counts, mask))
    assert stats.n_units == sum(one.n_units for _, one in per_sample)
    assert stats.n_units_occluded == sum(one.n_units_occluded for _, one in per_sample)
    assert 0.0 <= stats.ratio <= 1.0


# ══════════════════════════════════════════════════════════════
# §3.2 / M1 三个 mask 的语义分离
# ══════════════════════════════════════════════════════════════
def test_mask_semantics_counts_are_self_consistent() -> None:
    """M1：mask_semantics 的三个计数自洽（被遮事件 + 观测事件 == 总事件数）。

    注：该函数返回的第二个键是 **n_observed_cells**（格数）；0/1 计数下格数即事件数，
    此时 n_occluded_events + n_observed_cells == 总事件数严格成立；一般计数
    （n_j >= 2）下必须按 n_j 加权，两种口径都在此断言。
    """
    counts = _weighted_counts()
    mask, _ = build_occlusion(counts, ratio=0.5, granularity="event", seed=0)
    stats = mask_semantics(counts, mask)
    total = int(counts.sum().item())
    observed_mass = int((counts.to(torch.float32) * (~mask).to(torch.float32)).sum().item())
    assert stats["n_occluded_events"] == int(flat_counts_sum(counts, mask))
    assert stats["n_occluded_events"] + observed_mass == total
    assert stats["n_occluded_cells"] + stats["n_observed_cells"] == int((counts > 0).sum().item())
    # 0/1 计数下格数 == 事件数：三个计数必须严格相加守恒
    binary = (counts > 0).to(counts.dtype)
    binary_mask, _ = build_occlusion(binary, ratio=0.5, granularity="event", seed=0)
    binary_stats = mask_semantics(binary, binary_mask)
    assert binary_stats["n_occluded_events"] + binary_stats["n_observed_cells"] == int(
        binary.sum().item()
    )
    # line_mask 只贡献 padding 计数，不改变上面任何一项
    line_mask: Sequence[bool] = [True, True, False]
    with_padding = mask_semantics(binary, binary_mask, line_mask=line_mask)
    assert with_padding["n_padding_lines"] == 1
    assert with_padding["n_occluded_events"] == binary_stats["n_occluded_events"]


def test_three_masks_have_distinct_shapes_and_dtypes() -> None:
    """Plan 04 §3.2 / M1：occlusion (B,K,T,X,S,C) bool、line_mask (B,K) bool、range_mask (X,)。"""
    grid = _grid()
    batch = make_batch(
        batch=BATCH_SIZE,
        k=BATCH_K,
        grid=grid,
        events=4 * BATCH_SIZE,
        holds=BATCH_SIZE,
        seed=2,
        active_lines=ACTIVE_LINES,
    )
    batch.assert_shapes()
    occlusion = batch.occlusion_bool()
    line_mask = batch.line_mask_bool()
    range_mask = batch.range_mask_bool()
    assert tuple(occlusion.shape) == batch.batch_field_shape()
    assert tuple(occlusion.shape) == _cell_shape(grid)
    assert occlusion.dtype == torch.bool
    assert tuple(line_mask.shape) == (BATCH_SIZE, BATCH_K)
    assert line_mask.dtype == torch.bool
    assert tuple(range_mask.shape) == (grid.x_bins,)
    assert range_mask.dtype == torch.bool
    # range_mask 由 grid 的 x 桶中心派生（不是钳位，也不带 batch / K 维）
    centers = torch.as_tensor(np.asarray(grid.x_centers(), dtype=np.float64))
    assert torch.equal(range_mask, centers.abs() <= grid.x_max)
    assert not bool(occlusion.any()), "未给 occlusion 时必须视为全 False（无遮盖）"


def test_occlusion_shape_and_dtype_are_enforced() -> None:
    """M1：occlusion 的形状 / dtype 不符必须抛 AssertionError（不得降级为日志）。"""
    grid = _grid()
    wrong_shape = make_batch(
        batch=BATCH_SIZE,
        k=BATCH_K,
        grid=grid,
        occlusion=torch.zeros(1, dtype=torch.bool),
    )
    with pytest.raises(AssertionError, match="occlusion"):
        wrong_shape.assert_shapes()
    wrong_dtype = make_batch(
        batch=BATCH_SIZE,
        k=BATCH_K,
        grid=grid,
        occlusion=torch.zeros(_cell_shape(grid), dtype=torch.float32),
    )
    with pytest.raises(AssertionError, match="occlusion 必须是 bool"):
        wrong_dtype.occlusion_bool()
    # occlusion 的语义依赖 counts：只给 occlusion 不给 counts 必须被拒绝
    with_occlusion = make_batch(
        batch=BATCH_SIZE,
        k=BATCH_K,
        grid=grid,
        occlusion=torch.zeros(_cell_shape(grid), dtype=torch.bool),
    )
    with pytest.raises(AssertionError, match="没有 counts"):
        dataclasses.replace(with_occlusion, counts=None).assert_shapes()


def test_occlusion_all_true_keeps_counts_and_line_mask_intact() -> None:
    """M1：occlusion 全 True 只表示「全部待补全」，**不得**被当成有效性 mask。"""
    grid = _grid()
    reference = make_batch(
        batch=BATCH_SIZE,
        k=BATCH_K,
        grid=grid,
        events=4 * BATCH_SIZE,
        holds=BATCH_SIZE,
        seed=5,
        active_lines=ACTIVE_LINES,
    )
    filled = make_batch(
        batch=BATCH_SIZE,
        k=BATCH_K,
        grid=grid,
        counts=reference.counts.clone(),
        occlusion=torch.ones(_cell_shape(grid), dtype=torch.bool),
        active_lines=ACTIVE_LINES,
    )
    filled.assert_shapes()
    assert bool(filled.occlusion_bool().all())
    total = int(reference.counts.sum().item())
    assert total > 0
    assert int(filled.counts.sum().item()) == total
    assert torch.equal(filled.line_mask_bool(), reference.line_mask_bool())
    assert torch.equal(filled.range_mask_bool(), reference.range_mask_bool())
    # 输入侧全遮蔽，但目标计数完好无损——两者是不同通路
    assert not bool(filled.observed_counts().any())
    assert flat_counts_sum(filled.counts, filled.occlusion_bool()) == float(total)
    # 形状不同 => 三个 mask 无法互相替代（混用会在 assert_shapes 处炸）
    assert filled.occlusion_bool().dim() != filled.line_mask_bool().dim()
    assert filled.line_mask_bool().dim() != filled.range_mask_bool().dim()
