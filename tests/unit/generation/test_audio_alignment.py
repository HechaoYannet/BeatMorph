"""audio_align / seconds_position 的**时间基**契约（plan 07 §9-64 的回归护栏）。

为什么需要这条护栏：`_field_seconds` 的缓存 key 曾经只有 `(t_bins, device)`，而
`t_bins` 是全库常数、窗口网格的 `bpm_eff` 却逐窗口不同 ⇒ 该张量在**第一次前向时被
冻结**成第一个窗口的 BPM，之后每个窗口的 τ→秒 都被同一个逐窗口变化的因子缩放
（真实 train split 实测：冻结口径 301.5 BPM vs val 中位 162 ⇒ 中位 0.54×）。
后果是 `audio_align` 注入的音频**不在该 token 的时刻上**，该通路被训练成纯噪声
（实测对其任何干预都不改变损失）。**改回去不会报任何错**——所以必须由断言钉住。

本文件全部是契约级：不加载权重、不用 GPU。
"""

from __future__ import annotations

import torch

from beatmorph.core.contracts.tensors import MERT_FRAME_RATE_HZ
from beatmorph.field.grid import FieldGrid
from beatmorph.generation.model import MaskedFieldModel, ModelConfig
from tests.unit.generation._builders import (
    TEST_AUDIO_DIM,
    TEST_BPM,
    audio_frames_for,
    make_batch,
    make_grid,
)

T_BINS = 48
X_BINS = 4
K_LINES = 2
MODEL_CONFIG = ModelConfig(
    d_model=16,
    n_heads=2,
    n_layers=2,
    global_period=2,
    k_max=4,
    audio_dim=TEST_AUDIO_DIM,
    audio_align=True,
    seconds_position=True,
    head_skip=False,
)
CPU = torch.device("cpu")


def _grids() -> tuple[FieldGrid, FieldGrid]:
    slow = make_grid(t_bins=T_BINS, x_bins=X_BINS, pairs=((0.0, TEST_BPM),))
    fast = make_grid(t_bins=T_BINS, x_bins=X_BINS, pairs=((0.0, 2.0 * TEST_BPM),))
    return slow, fast


def test_seconds_table_follows_each_grid_not_the_first_call() -> None:
    slow, fast = _grids()
    model = MaskedFieldModel(MODEL_CONFIG, slow)
    a = model._field_seconds(slow, device=CPU)
    b = model._field_seconds(fast, device=CPU)
    assert not torch.allclose(a, b), "同一张表被复用到了 BPM 不同的网格上（缓存 key 丢掉了 bpm）"
    # 秒数 = tau * 60/bpm ⇒ BPM 翻倍则秒数减半（同一张表被复用就会两头都错）
    assert torch.allclose(b, a * (slow.bpm_points[0].bpm / fast.bpm_points[0].bpm), rtol=1e-4)


def test_aligned_frame_index_matches_each_batch_time_base() -> None:
    slow, fast = _grids()
    model = MaskedFieldModel(MODEL_CONFIG, slow)
    for grid in (fast, slow):
        batch = make_batch(k=K_LINES, grid=grid, t_audio=audio_frames_for(grid), events=2, holds=0)
        index = model._aligned_frame_index(batch, grid, CPU)
        seconds = torch.as_tensor(grid.tau_seconds(), dtype=torch.float32)
        assert index.shape == (grid.t_bins,)
        assert int(index.min()) >= 0
        assert int(index.max()) < batch.audio_emb.shape[1]
        # 契约：帧下标必须落在该 token 的**时刻**上（量化误差 <= 半帧）；
        # 唯一例外是窗口末端的 clamp（tau 到达/越过音频末尾），此时下标 == 末帧且时刻在尾部。
        n_frames = int(batch.audio_emb.shape[1])
        free = index < n_frames - 1
        assert bool(free.any()), "全部被 clamp 说明音频长度与网格时长不自洽"
        offset = (index[free].to(torch.float32) / MERT_FRAME_RATE_HZ - seconds[free]).abs()
        assert (
            float(offset.max()) <= 0.5 / MERT_FRAME_RATE_HZ + 1e-6
        ), f"bpm={grid.bpm_points[0].bpm} 的帧下标与 tau 时刻不符：最大偏差 {float(offset.max()):.6f} s"
        assert bool(
            (seconds[~free] >= (n_frames - 1) / MERT_FRAME_RATE_HZ - 0.5 / MERT_FRAME_RATE_HZ).all()
        ), "被 clamp 的帧下标只允许出现在窗口末端"
        assert bool((index[1:] >= index[:-1]).all()), "帧下标必须随 tau 单调不减"


def test_aligned_frame_index_is_not_leaked_across_bpm() -> None:
    slow, fast = _grids()
    # 先喂快网格（模拟「第一个窗口」），再喂慢网格：慢网格必须拿到自己的时间基
    model = MaskedFieldModel(MODEL_CONFIG, fast)
    batch_fast = make_batch(k=K_LINES, grid=fast, t_audio=audio_frames_for(fast), events=2, holds=0)
    batch_slow = make_batch(k=K_LINES, grid=slow, t_audio=audio_frames_for(slow), events=2, holds=0)
    idx_fast = model._aligned_frame_index(batch_fast, fast, CPU)
    idx_slow = model._aligned_frame_index(batch_slow, slow, CPU)
    assert not torch.equal(idx_fast, idx_slow), "第二个网格复用了第一个网格的秒表（回归）"
    assert int(idx_slow.max()) > int(idx_fast.max())


def test_aligned_audio_depends_on_the_batch_grid() -> None:
    slow, fast = _grids()
    model = MaskedFieldModel(MODEL_CONFIG, slow).eval()
    b_slow = make_batch(k=K_LINES, grid=slow, t_audio=audio_frames_for(slow), events=2, holds=0)
    b_fast = make_batch(k=K_LINES, grid=fast, t_audio=audio_frames_for(fast), events=2, holds=0)
    with torch.no_grad():
        a_slow = model._aligned_audio(b_slow, slow, CPU)
        a_fast = model._aligned_audio(b_fast, fast, CPU)
    assert a_slow.shape == (1, 1, T_BINS, MODEL_CONFIG.d_model)
    assert not torch.allclose(a_slow, a_fast)
