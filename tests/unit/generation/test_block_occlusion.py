"""块状遮盖（`granularity="block"`）的护栏（plan 07 §9-76）。

为什么需要这一条：§9-75② 的结构性结论是「逐 token 全有全无的遮盖把**被预测 token 内部的空间分布**
从输入里删掉了」。块状遮盖的全部意义就是**打破「token 内全有全无」** —— 一旦有人把它改回
「先选事件再扩张到 token」，这个实验就悄悄退回原状，而且不会有任何报错。
"""

from __future__ import annotations

import torch

from beatmorph.core.contracts.phigros import NoteType
from beatmorph.field.target import CHANNEL_INDEX
from beatmorph.generation.masks import BLOCK_TAU, BLOCK_X, build_occlusion
from tests.unit.generation._builders import make_grid

_TAP = int(CHANNEL_INDEX[NoteType.TAP])

T_BINS = 64
X_BINS = 128
K_LINES = 3


def _counts() -> torch.Tensor:
    """确定性计数：每条线每 2 个 τ 格一个 tap，x 位置随 τ 变化（覆盖整条 x 轴）。"""
    grid = make_grid(t_bins=T_BINS, x_bins=X_BINS)
    counts = torch.zeros(
        K_LINES, grid.t_bins, grid.x_bins, grid.sides, grid.channels, dtype=torch.int16
    )
    for k in range(K_LINES):
        for t in range(0, grid.t_bins, 2):
            counts[k, t, (t * 7 + k * 3) % grid.x_bins, 0, _TAP] += 1
    return counts


def test_block_mask_leaves_visible_cells_inside_masked_tokens() -> None:
    """**核心不变量**：块状遮盖必须留下「部分遮盖」的 token。"""
    counts = _counts()
    occlusion, _stats = build_occlusion(counts, ratio=0.5, granularity="block", seed=1)
    k_dim, t_dim = counts.shape[0], counts.shape[1]
    per_token = occlusion.reshape(k_dim, t_dim, -1)
    masked = per_token.any(dim=-1)
    fully = per_token.all(dim=-1)
    partial = masked & (~fully)
    assert bool(masked.any()), "一个 token 都没被遮：ratio 参数没生效"
    assert bool(partial.any()), (
        "块状遮盖下没有任何「部分遮盖」的 token ⇒ 它退化成了 token 全有全无",
        "（那正是 §9-75② 要修的结构）",
    )


def test_block_mask_is_not_degenerate() -> None:
    """遮盖指示器不得退化成「这里有事件」（mask_leak 远小于 1）。"""
    counts = _counts()
    _occlusion, stats = build_occlusion(counts, ratio=0.5, granularity="block", seed=1)
    assert stats.mask_leak < 0.5, f"mask_leak={stats.mask_leak}：遮盖几乎只落在有事件的格子上"
    assert stats.ratio > 0.0, "一个事件都没遮住"
    assert stats.ratio <= 1.0


def test_block_mask_is_deterministic() -> None:
    counts = _counts()
    first, _ = build_occlusion(counts, ratio=0.5, granularity="block", seed=7)
    again, _ = build_occlusion(counts, ratio=0.5, granularity="block", seed=7)
    other, _ = build_occlusion(counts, ratio=0.5, granularity="block", seed=8)
    assert torch.equal(first, again), "同一 seed 必须逐位一致"
    assert not torch.equal(first, other), "不同 seed 必须给出不同的遮盖"


def test_block_size_is_smaller_than_a_token() -> None:
    """块必须**远小于**一个 token，否则又变成整 token 遮挡。"""
    assert BLOCK_TAU < T_BINS
    assert BLOCK_X < X_BINS


def test_token_granularity_is_still_all_or_nothing() -> None:
    """对照：默认路径（event + token_block）**必须**保持全有全无 —— 不要顺手把它改了。"""
    counts = _counts()
    occlusion, _stats = build_occlusion(counts, ratio=0.5, granularity="event", seed=1)
    k_dim, t_dim = counts.shape[0], counts.shape[1]
    per_token = occlusion.reshape(k_dim, t_dim, -1)
    masked = per_token.any(dim=-1)
    fully = per_token.all(dim=-1)
    assert torch.equal(masked, fully), "默认路径的遮盖不再是 token 区块结构"
