"""generation 单元测试的合成构造器。

**不得固化任何物理常量**（AGENTS.md §3.3）：音频帧数由 duration x frame_rate 派生、
tau 格数由 SUBDIVISIONS_PER_BEAT 派生、位置由 RPE 舞台半宽派生。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import torch
from torch import Tensor

from beatmorph.core.contracts.phigros import (
    SUBDIVISIONS_PER_BEAT,
    BpmPoint,
    NoteType,
)
from beatmorph.core.contracts.tensors import MERT_FRAME_RATE_HZ
from beatmorph.field.grid import DEFAULT_X_BINS, FieldGrid
from beatmorph.field.target import CHANNEL_INDEX, HOLD_END_CHANNEL
from beatmorph.generation.batch import N_ORDINARY_TRACKS, FieldBatch

#: 测试用 BPM（超参，不是被测物理常量）
TEST_BPM: float = 120.0
#: 测试用音频特征维（避免为单测加载 MERT 的 1024 维投影）
TEST_AUDIO_DIM: int = 16


def make_bpm_points(*segments: tuple[float, float]) -> list[BpmPoint]:
    """(拍, BPM) 序列 -> BpmPoint 列表。"""
    return [BpmPoint(time_beats=beats, bpm=bpm) for beats, bpm in segments]


def make_grid(
    *,
    t_bins: int,
    x_bins: int = DEFAULT_X_BINS,
    pairs: Sequence[tuple[float, float]] = ((0.0, TEST_BPM),),
) -> FieldGrid:
    """已绑定 tau 轴的网格。"""
    grid = FieldGrid(x_bins=x_bins).with_time(t_bins, make_bpm_points(*pairs))
    grid.assert_grid()
    return grid


def audio_frames_for(grid: FieldGrid) -> int:
    """音频帧数 = round(duration x frame_rate)（**派生**，G4；不得写死）。"""
    return round(grid.total_seconds * MERT_FRAME_RATE_HZ)


def make_counts(
    *,
    batch: int = 1,
    k: int,
    grid: FieldGrid,
    events: int = 0,
    holds: int = 0,
    seed: int = 0,
) -> Tensor:
    """构造 (B, K, T, X, S, C) 的桶内计数（int16），含成对的 hold 起止点。

    事件位置由种子决定，**覆盖所有 (k, x, s, c)** 的组合空间，便于测试排列敏感性与
    mask 语义分离。
    """
    shape = (batch, k, grid.t_bins, grid.x_bins, grid.sides, grid.channels)
    counts = torch.zeros(shape, dtype=torch.int16)
    generator = torch.Generator().manual_seed(seed)
    single_channels = [CHANNEL_INDEX[NoteType.TAP], CHANNEL_INDEX[NoteType.DRAG]]
    for index in range(int(events)):
        b = index % batch
        kk = int(torch.randint(0, k, (1,), generator=generator).item())
        tt = int(torch.randint(0, grid.t_bins, (1,), generator=generator).item())
        xx = int(torch.randint(0, grid.x_bins, (1,), generator=generator).item())
        ss = int(torch.randint(0, grid.sides, (1,), generator=generator).item())
        cc = single_channels[index % len(single_channels)]
        counts[b, kk, tt, xx, ss, cc] += 1
    for index in range(int(holds)):
        b = index % batch
        kk = int(torch.randint(0, k, (1,), generator=generator).item())
        xx = int(torch.randint(0, grid.x_bins, (1,), generator=generator).item())
        ss = int(torch.randint(0, grid.sides, (1,), generator=generator).item())
        start = int(torch.randint(0, max(grid.t_bins - 1, 1), (1,), generator=generator).item())
        end = (
            start
            + 1
            + int(
                torch.randint(0, max(grid.t_bins - start - 1, 1), (1,), generator=generator).item(),
            )
        )
        counts[b, kk, start, xx, ss, CHANNEL_INDEX[NoteType.HOLD]] += 1
        counts[b, kk, end, xx, ss, HOLD_END_CHANNEL] += 1
    return counts


def make_tracks(
    *,
    batch: int = 1,
    k: int,
    t_line: int,
    columns: int = N_ORDINARY_TRACKS,
    seed: int = 0,
) -> Tensor:
    """(B, K, T_line, F) 的事件轨（已是**跨层求和后**的形态）。"""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, k, t_line, columns, generator=generator)


def make_line_mask(*, batch: int = 1, k: int, active: Iterable[int] | None = None) -> Tensor:
    """(B, K) bool 线掩码；active 省略时全部有效。"""
    mask = torch.zeros(batch, k, dtype=torch.bool)
    chosen = list(range(k)) if active is None else list(active)
    for index in chosen:
        mask[:, index] = True
    return mask


def make_batch(
    *,
    k: int = 3,
    grid: FieldGrid,
    batch: int = 1,
    events: int = 6,
    holds: int = 2,
    counts: Tensor | None = None,
    occlusion: Tensor | None = None,
    active_lines: Iterable[int] | None = None,
    t_audio: int | None = None,
    audio_dim: int = TEST_AUDIO_DIM,
    t_line: int | None = None,
    extended_dim: int = 0,
    seed: int = 0,
) -> FieldBatch:
    """组装一个 FieldBatch（默认带桶内计数目标；推理时传 counts=zeros 或临时替换）。"""
    frames = audio_frames_for(grid) if t_audio is None else int(t_audio)
    generator = torch.Generator().manual_seed(seed + 1)
    audio = torch.randn(batch, frames, audio_dim, generator=generator)
    tracks = make_tracks(batch=batch, k=k, t_line=t_line or grid.t_bins, seed=seed + 2)
    difficulty = torch.full((batch,), 15.0) + torch.arange(batch, dtype=torch.float32) * 0.1
    target = (
        make_counts(batch=batch, k=k, grid=grid, events=events, holds=holds, seed=seed)
        if counts is None
        else counts
    )
    extended = (
        torch.randn(batch, k, t_line or grid.t_bins, extended_dim, generator=generator)
        if extended_dim > 0
        else None
    )
    return FieldBatch(
        audio_emb=audio,
        frame_rate=MERT_FRAME_RATE_HZ,
        line_tracks=tracks,
        line_mask=make_line_mask(batch=batch, k=k, active=active_lines),
        difficulty=difficulty,
        grid=grid,
        counts=target,
        occlusion=occlusion,
        extended_tracks=extended,
    )


def total_events(counts: Tensor) -> int:
    """桶内计数总和（事件的唯一定义：Hold 起点与终点各算一个）。"""
    return int(counts.sum().item())


def cell_count(grid: FieldGrid, k: int) -> int:
    """格元总数 K*T*X*S*C（测试断言用）。"""
    return k * grid.t_bins * grid.x_bins * grid.sides * grid.channels


def subdivisions(count: int) -> int:
    """tau 格数 = count / SUBDIVISIONS_PER_BEAT 的取值（测试用派生助手）。"""
    return math.ceil(SUBDIVISIONS_PER_BEAT / count)
