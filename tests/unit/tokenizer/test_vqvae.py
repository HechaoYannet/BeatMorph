"""VQVAETokenizer 单元测试（CPU 可跑，不需权重/GPU）。

覆盖：栅格化、forward 形状、encode code∈范围、空小节、确定性、loss 反传、
codebook_usage、K-means init、死码重启、decode 合法 Chart、latent%n_heads 校验。

仿 tests/unit/planner/test_density.py 的 CPU 小模型风格（_small_tok() 返回 latent=8
小模型，避免 2048 码本/256 latent 在 CPU 上慢）。
"""

from __future__ import annotations

import pytest
import torch

from beatmorph.core.contracts import BpmPoint, Chart, GameMode, Note, NoteType
from beatmorph.tokenizer.vqvae import (
    VQVAETokenizer,
    rasterize_bar,
    rasterize_chart,
)


def _small_tok(
    codebook_size: int = 16,
    latent: int = 8,
    time_bins: int = 16,
    transformer_heads: int = 2,
    transformer_layers: int = 1,
    dead_code_steps: int = 3,
    present_pos_weight: float = 15.0,
) -> VQVAETokenizer:
    """CPU 小模型：避免默认 2048/256 在 CI 上慢。"""
    return VQVAETokenizer(
        codebook_size=codebook_size,
        latent=latent,
        lane=4,
        time_bins=time_bins,
        feat=6,
        cnn_channels=(8, 8),
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
        dead_code_steps=dead_code_steps,
        present_pos_weight=present_pos_weight,
    )


def _make_chart(n_bars: int = 3, bpm: float = 120.0, include_hold: bool = True) -> Chart:
    """构造测试谱面：120bpm 4/4 → bar=2s；每 bar 放若干 Note。含空 bar 测试。"""
    bar_dur = 4 * 60.0 / bpm
    notes: list[Note] = []
    for b in range(n_bars):
        bar_start = b * bar_dur
        if b == 1:
            continue  # 第 2 小节留空，测空 bar
        notes.append(Note(time=bar_start + 0.0, lane=0, type=NoteType.TAP))
        notes.append(Note(time=bar_start + 0.5, lane=1, type=NoteType.TAP))
        if include_hold:
            notes.append(Note(time=bar_start + 1.0, lane=2, type=NoteType.HOLD, duration=0.3))
    return Chart(
        mode=GameMode.MANIA_4K,
        difficulty=5,
        bpm_points=[BpmPoint(time=0.0, bpm=bpm)],
        notes=notes,
        meta={"audio_duration": n_bars * bar_dur},
    )


# ── 栅格化 ──────────────────────────────────────────────────


class TestRasterize:
    def test_rasterize_bar_shape_and_empty(self) -> None:
        grid = rasterize_bar([], 0.0, 2.0, lane=4, time_bins=16)
        assert grid.shape == (4, 16, 6)
        assert grid.sum() == 0.0

    def test_rasterize_bar_places_note(self) -> None:
        # bar=2s，16 bins，中心对齐：t=0 → bin0，t=1.0 → bin8，t=2.0→bin15
        notes = [Note(time=1.0, lane=1, type=NoteType.TAP)]
        grid = rasterize_bar(notes, 0.0, 2.0, lane=4, time_bins=16)
        # lane1, bin8 应有 TAP one-hot（type_idx=0）
        assert grid[1, 8, 0] == 1.0
        assert grid[1, 8, 1:].sum() == 0.0  # 其余 type 与 duration 为 0

    def test_rasterize_bar_hold_duration(self) -> None:
        notes = [Note(time=0.0, lane=0, type=NoteType.HOLD, duration=0.4)]
        grid = rasterize_bar(notes, 0.0, 2.0, lane=4, time_bins=16)
        # HOLD type_idx=1
        assert grid[0, 0, 1] == 1.0
        assert grid[0, 0, 5] == pytest.approx(0.4)  # duration 标量

    def test_rasterize_bar_out_of_range_skipped(self) -> None:
        # 超出本小节的 Note 不应入格
        notes = [Note(time=3.0, lane=0)]  # bar=2s，t=3.0 超界
        grid = rasterize_bar(notes, 0.0, 2.0, lane=4, time_bins=16)
        assert grid.sum() == 0.0

    def test_rasterize_chart_returns_grids_and_bounds(self) -> None:
        chart = _make_chart(n_bars=3)
        grids, bounds = rasterize_chart(chart, lane=4, time_bins=16)
        assert len(grids) == len(bounds) - 1
        assert bounds[0] == 0.0
        # 120bpm bar=2s，3 bar → bounds=[0,2,4,6]
        assert bounds == [0.0, 2.0, 4.0, 6.0]
        # 第 2 小节空但仍栅格化（全 0）
        assert grids[1].sum() == 0.0
        for g in grids:
            assert g.shape == (4, 16, 6)


# ── forward ──────────────────────────────────────────────────


class TestForward:
    def test_forward_shapes(self) -> None:
        tok = _small_tok()
        B, bars = 2, 3
        grid = torch.randn(B, bars, 4, 16, 6)
        mask = torch.ones(B, bars, dtype=torch.bool)
        out = tok(grid, bar_mask=mask)

        assert out["z_e"].shape == (B, bars, 8)
        assert out["z_q"].shape == (B, bars, 8)
        assert out["z_q_st"].shape == (B, bars, 8)
        assert out["indices"].shape == (B, bars)
        assert out["indices"].max() < 16
        assert out["indices"].min() >= 0
        assert out["type_logits"].shape == (B, bars, 4, 16, 5)
        assert out["present"].shape == (B, bars, 4, 16)
        assert out["duration"].shape == (B, bars, 4, 16)

    def test_forward_codes_in_range(self) -> None:
        tok = _small_tok(codebook_size=16)
        grid = torch.randn(2, 3, 4, 16, 6)
        out = tok(grid)
        assert out["indices"].max().item() < 16
        assert out["indices"].min().item() >= 0

    def test_forward_invalid_dim(self) -> None:
        tok = _small_tok()
        with pytest.raises(ValueError, match="bar_grid"):
            tok(torch.randn(1, 3, 4, 16))  # 4D 而非 5D

    def test_dim_not_divisible_by_heads(self) -> None:
        with pytest.raises(ValueError, match="整除"):
            VQVAETokenizer(latent=10, transformer_heads=4, codebook_size=8, time_bins=8)


# ── encode / decode ──────────────────────────────────────────


class TestEncodeDecode:
    def test_encode_codes_in_range(self) -> None:
        tok = _small_tok(codebook_size=16)
        chart = _make_chart(n_bars=3)
        tokens = tok.encode(chart)
        assert len(tokens) == 3  # 含空 bar
        assert all(0 <= t.code < 16 for t in tokens)
        for i, t in enumerate(tokens):
            assert t.bar_index == i
            assert t.duration_bars == 1

    def test_encode_empty_bar_included(self) -> None:
        tok = _small_tok(codebook_size=16)
        chart = _make_chart(n_bars=3)  # 含空第 2 bar
        tokens = tok.encode(chart)
        assert len(tokens) == 3  # 空 bar 也产 token
        # 空 bar 的 code 仍合法
        assert 0 <= tokens[1].code < 16

    def test_encode_empty_chart(self) -> None:
        tok = _small_tok(codebook_size=16)
        chart = Chart(
            mode=GameMode.MANIA_4K,
            difficulty=1,
            bpm_points=[BpmPoint(time=0.0, bpm=120.0)],
            notes=[],
        )
        tokens = tok.encode(chart)
        assert tokens == []

    def test_determinism(self) -> None:
        tok = _small_tok(codebook_size=16)
        chart = _make_chart(n_bars=3)
        t1 = tok.encode(chart)
        t2 = tok.encode(chart)
        assert [t.code for t in t1] == [t.code for t in t2]

    def test_decode_returns_valid_chart(self) -> None:
        tok = _small_tok(codebook_size=16)
        chart = _make_chart(n_bars=3)
        tokens = tok.encode(chart)
        out = tok.decode(tokens)
        assert out.mode == GameMode.MANIA_4K
        assert len(out.bpm_points) == 1
        assert out.bpm_points[0].bpm > 0
        total = chart.meta["audio_duration"]
        for n in out.notes:
            assert 0.0 <= n.time <= total + 1.0
            assert 0 <= n.lane < 4
            assert n.type in NoteType
            assert n.duration >= 0.0

    def test_decode_empty_tokens(self) -> None:
        tok = _small_tok(codebook_size=16)
        out = tok.decode([])
        assert out.notes == []
        assert out.bpm_points[0].bpm == 120.0

    def test_roundtrip_path(self) -> None:
        """encode→decode 通路打通（未训练，不要求高匹配率）。"""
        tok = _small_tok(codebook_size=16)
        chart = _make_chart(n_bars=3)
        tokens = tok.encode(chart)
        out = tok.decode(tokens)
        # 至少能产出合法 Note 结构（数量级合理）
        assert isinstance(out, Chart)
        assert out.notes is not None


# ── loss / 防坍缩 / 监控 ─────────────────────────────────────


class TestLossAndCollapse:
    def test_loss_backprop(self) -> None:
        tok = _small_tok(codebook_size=16)
        B, bars = 2, 3
        bar_grid = torch.zeros(B, bars, 4, 16, 6)
        # 构造若干真 Note：lane0,bin0 为 TAP；lane1,bin4 为 HOLD dur=0.3
        bar_grid[:, :, 0, 0, 0] = 1.0
        bar_grid[:, :, 1, 4, 1] = 1.0
        bar_grid[:, :, 1, 4, 5] = 0.3
        mask = torch.ones(B, bars, dtype=torch.bool)
        pred = tok(bar_grid, bar_mask=mask)
        loss = tok.loss_fn(pred, {"bar_grid": bar_grid})
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in tok.parameters())
        # 分项 loss 挂到 pred
        assert "loss_recon" in pred
        assert "loss_commit" in pred

    def test_loss_ignores_padding_bar(self) -> None:
        tok = _small_tok(codebook_size=16)
        tok.eval()  # 关 dropout：此测的是 padding mask 逻辑，非 dropout 容差
        B, bars = 2, 3
        bar_grid = torch.zeros(B, bars, 4, 16, 6)
        bar_grid[:, :, 0, 0, 0] = 1.0
        # 样本0 第3 bar padding，样本1 全真
        mask = torch.tensor([[True, True, False], [True, True, True]])
        pred = tok(bar_grid, bar_mask=mask)
        loss1 = tok.loss_fn(pred, {"bar_grid": bar_grid})

        # 改 padding 小节的真值 grid，loss 应不变
        bar_grid2 = bar_grid.clone()
        bar_grid2[0, 2, :, :, :] = 1.0  # 样本0 padding bar 灌满
        pred2 = tok(bar_grid2, bar_mask=mask)
        loss2 = tok.loss_fn(pred2, {"bar_grid": bar_grid2})
        assert torch.allclose(loss1, loss2, atol=1e-5)

    def test_present_pos_weight_amplifies_positive_grad(self) -> None:
        """pos_weight 大 → 有 Note bin（正类）的 present 梯度更大（缓解稀疏类不平衡）。

        RFC-0027 §1：present 头空 bin:有 Note ≈ 15:1，pos_weight 给正类加权防被压低。
        验证：pos_weight=50 vs pos_weight=1，正类位置（present_tgt=1）的 present
        logit 梯度量级应显著更大。
        """
        torch.manual_seed(0)
        # 相同初始权重的两个模型，仅 pos_weight 不同
        tok_low = _small_tok(codebook_size=16, present_pos_weight=1.0)
        torch.manual_seed(0)
        tok_high = _small_tok(codebook_size=16, present_pos_weight=50.0)
        tok_low.eval()
        tok_high.eval()
        B, bars = 1, 2
        bar_grid = torch.zeros(B, bars, 4, 16, 6)
        bar_grid[:, :, 0, 0, 0] = 1.0  # lane0,bin0 有 TAP（正类位）
        mask = torch.ones(B, bars, dtype=torch.bool)

        pred_low = tok_low(bar_grid, bar_mask=mask)
        loss_low = tok_low.loss_fn(pred_low, {"bar_grid": bar_grid})
        loss_low.backward()
        # present 头参数梯度（dec_cnn 最后一层含 present 通道）
        grad_low = tok_low.dec_cnn[-1].weight.grad.abs().mean().item()

        pred_high = tok_high(bar_grid, bar_mask=mask)
        loss_high = tok_high.loss_fn(pred_high, {"bar_grid": bar_grid})
        loss_high.backward()
        grad_high = tok_high.dec_cnn[-1].weight.grad.abs().mean().item()

        # pos_weight 大 → 正类加权 → decoder present 通道梯度更大
        assert (
            grad_high > grad_low
        ), f"pos_weight=50 应放大 present 梯度：high={grad_high} <= low={grad_low}"

    def test_codebook_usage_range(self) -> None:
        tok = _small_tok(codebook_size=16)
        usage = tok.codebook_usage()
        assert isinstance(usage, float)
        assert 0.0 <= usage <= 1.0

    def test_codebook_usage_after_forward(self) -> None:
        tok = _small_tok(codebook_size=16)
        grid = torch.randn(1, 3, 4, 16, 6)
        _ = tok(grid)
        usage = tok.codebook_usage()
        assert usage > 0.0  # forward 后必有 code 被用

    def test_kmeans_init(self) -> None:
        tok = _small_tok(codebook_size=16)
        data = torch.randn(32, 8)
        # 记录 init 前码本
        before = tok.codebook.clone()
        tok.kmeans_init(data, k=16, iters=5)
        assert tok.codebook.shape == (16, 8)
        # init 后码本应变化
        assert not torch.allclose(before, tok.codebook)
        # 非 NaN
        assert torch.isfinite(tok.codebook).all()

    def test_kmeans_init_few_samples(self) -> None:
        """样本数 < codebook_size 时用随机正态补齐，仍保证码本完整。"""
        tok = _small_tok(codebook_size=16)
        data = torch.randn(4, 8)  # 仅 4 样本 < 16
        tok.kmeans_init(data, k=16, iters=3)
        assert tok.codebook.shape == (16, 8)
        assert torch.isfinite(tok.codebook).all()

    def test_restart_dead_codes(self) -> None:
        tok = _small_tok(codebook_size=16, dead_code_steps=3)
        # 手动置 usage_count：部分码设 0（死码），部分设 10（活码）
        tok.code_usage_count[:] = 10
        tok.code_usage_count[3:6] = 0  # 3 个死码
        before = tok.codebook[3:6].clone()
        z_e = torch.randn(1, 3, 8)
        n = tok.restart_dead_codes(z_e)
        assert n == 3
        # 死码行被替换
        assert not torch.allclose(before, tok.codebook[3:6])
        # 重启后计数清零
        assert int(tok.code_usage_count[3:6].sum()) == 0

    def test_restart_dead_codes_none_dead(self) -> None:
        tok = _small_tok(codebook_size=16, dead_code_steps=3)
        tok.code_usage_count[:] = 10  # 全活
        z_e = torch.randn(1, 3, 8)
        n = tok.restart_dead_codes(z_e)
        assert n == 0
