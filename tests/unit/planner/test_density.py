"""DensityPlanner 单元测试（CPU 可跑，不需权重/GPU）。

覆盖：forward 形状/值域、plan→Section、loss 反传、section_boundaries_from_bpm、
difficulty 极值、style_emb=None 可跑、dim%n_heads 校验。
"""

from __future__ import annotations

import pytest
import torch

from beatmorph.core.contracts import MERT_DEFAULT_FEAT_DIM, BpmPoint
from beatmorph.planner.density import (
    _FRAME_RATE,
    DensityPlanner,
    section_boundaries_from_bpm,
)


def _make_bounds(batch: int, n_sec: int, dur_s: float = 10.0) -> torch.Tensor:
    """构造均匀的 section_bounds [B, S+1]。"""
    edges = torch.linspace(0, dur_s, n_sec + 1)
    return edges.unsqueeze(0).expand(batch, -1).contiguous()


class TestDensityPlannerForward:
    def test_forward_shapes_and_ranges(self) -> None:
        m = DensityPlanner()
        B, T = 2, round(10.0 * _FRAME_RATE)  # 10s @ 派生帧率（75Hz → 750）
        audio_emb = torch.randn(B, T, MERT_DEFAULT_FEAT_DIM)
        difficulty = torch.tensor([5, 12])
        bounds = _make_bounds(B, 10, 10.0)
        out = m(audio_emb, difficulty, bounds)

        assert out["density"].shape == (B, 10, 1)
        assert out["energy"].shape == (B, 10, 1)
        assert out["rest"].shape == (B, 10, 1)
        assert out["type_logits"].shape == (B, 10, 5)
        # sigmoid 输出 ∈ [0,1]
        for k in ("density", "energy", "rest"):
            assert out[k].min() >= 0.0
            assert out[k].max() <= 1.0

    def test_difficulty_extremes(self) -> None:
        m = DensityPlanner()
        out = m(
            torch.randn(2, 100, MERT_DEFAULT_FEAT_DIM),
            torch.tensor([1, 15]),
            _make_bounds(2, 5, 10.0),
        )
        assert out["density"].shape == (2, 5, 1)

    def test_style_emb_none_runs(self) -> None:
        m = DensityPlanner()
        out = m(
            torch.randn(1, 100, MERT_DEFAULT_FEAT_DIM),
            torch.tensor([8]),
            _make_bounds(1, 5, 10.0),
            style_emb=None,
        )
        assert out["type_logits"].shape == (1, 5, 5)

    def test_style_emb_with_runs(self) -> None:
        m = DensityPlanner()
        sty = torch.randn(1, 3, MERT_DEFAULT_FEAT_DIM)  # top_k=3
        out = m(
            torch.randn(1, 100, MERT_DEFAULT_FEAT_DIM),
            torch.tensor([8]),
            _make_bounds(1, 5, 10.0),
            style_emb=sty,
        )
        assert out["density"].shape == (1, 5, 1)

    def test_invalid_audio_emb_dim(self) -> None:
        m = DensityPlanner()
        with pytest.raises(ValueError, match="audio_emb"):
            m(torch.randn(1, 100, 512), torch.tensor([8]), _make_bounds(1, 5, 10.0))

    def test_dim_not_divisible_by_heads(self) -> None:
        with pytest.raises(ValueError, match="整除"):
            DensityPlanner(dim=770, n_heads=8)


class TestDensityPlannerPlan:
    def test_plan_returns_sections_covering_duration(self) -> None:
        m = DensityPlanner()
        dur = 10.0
        bounds = _make_bounds(1, 10, dur)
        secs = m.plan(
            torch.randn(1, int(dur * _FRAME_RATE), MERT_DEFAULT_FEAT_DIM),
            difficulty=8,
            section_bounds=bounds,
        )
        assert len(secs) == 10
        # 无重叠：每段 start==上一段 end
        for i, s in enumerate(secs):
            assert s.index == i
            assert 0.0 <= s.density_target <= 1.0
            assert 0.0 <= s.energy_level <= 1.0
            assert 0.0 <= s.rest_probability <= 1.0
            assert s.sections_type in ("intro", "verse", "chorus", "bridge", "outro")
        # 覆盖总时长、端点对齐
        assert secs[0].start_time == 0.0
        assert abs(secs[-1].end_time - dur) < 1e-5
        for i in range(len(secs) - 1):
            assert abs(secs[i].end_time - secs[i + 1].start_time) < 1e-5

    def test_plan_from_bpm_points(self) -> None:
        m = DensityPlanner()
        dur = 8.0  # 120BPM 4/4 → bar=2s, section_bars=4 → 每 section=8s → 1 section
        secs = m.plan(
            torch.randn(1, int(dur * _FRAME_RATE), MERT_DEFAULT_FEAT_DIM),
            difficulty=10,
            bpm_points=[BpmPoint(time=0.0, bpm=120.0)],
            duration_s=dur,
        )
        assert len(secs) >= 1
        assert secs[0].start_time == 0.0


class TestDensityPlannerLoss:
    def test_loss_backprop(self) -> None:
        m = DensityPlanner()
        B, S = 2, 8
        out = m(
            torch.randn(B, 200, MERT_DEFAULT_FEAT_DIM),
            torch.tensor([5, 10]),
            _make_bounds(B, S, 10.0),
        )
        target = {
            "density": torch.rand(B, S),
            "energy": torch.rand(B, S),
            "rest": torch.rand(B, S),
            "type": torch.randint(0, 5, (B, S)),
        }
        loss = m.loss_fn(out, target)
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        loss.backward()
        # 至少有梯度流到参数
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())


class TestSectionBoundariesUtil:
    def test_constant_bpm(self) -> None:
        # 120BPM 4/4: bar=2s; section_bars=4 → 8s/section; dur=16s → boundaries [0,8,16]
        b = section_boundaries_from_bpm([BpmPoint(time=0.0, bpm=120.0)], 16.0, section_bars=4)
        assert b[0] == 0.0
        assert abs(b[-1] - 16.0) < 1e-4
        assert len(b) >= 2

    def test_empty_bpm_fallback(self) -> None:
        b = section_boundaries_from_bpm([], 10.0)
        assert b == [0.0, 10.0]

    def test_variable_bpm(self) -> None:
        # 前 8s 120BPM，之后 240BPM
        bps = [BpmPoint(time=0.0, bpm=120.0), BpmPoint(time=8.0, bpm=240.0)]
        b = section_boundaries_from_bpm(bps, 20.0, section_bars=4)
        assert b[0] == 0.0
        assert b[-1] == 20.0
        assert len(b) >= 2


# ── padding-aware collate + mask loss（优化 B）──


def test_collate_pads_variable_length() -> None:
    """不同 T_seq / S 的两样本经 collate 后 shape 正确，section_mask 标记真段。"""
    from beatmorph.cli.train import collate

    # 样本 A：T=20, S=3（bounds 长度 4）
    a = {
        "audio_emb": torch.randn(20, _dim_or_skip()),
        "section_bounds": torch.tensor([0.0, 2.0, 4.0, 6.0]),
        "target": {
            "density": torch.tensor([0.1, 0.5, 0.9]),
            "energy": torch.tensor([0.1, 0.5, 0.9]),
            "rest": torch.tensor([0.0, 0.1, 0.0]),
            "type": torch.tensor([0, 2, 4]),
        },
        "difficulty": torch.tensor(5),
    }
    # 样本 B：T=10, S=2（bounds 长度 3）
    b = {
        "audio_emb": torch.randn(10, _dim_or_skip()),
        "section_bounds": torch.tensor([0.0, 3.0, 6.0]),
        "target": {
            "density": torch.tensor([0.2, 0.8]),
            "energy": torch.tensor([0.2, 0.8]),
            "rest": torch.tensor([0.1, 0.0]),
            "type": torch.tensor([1, 3]),
        },
        "difficulty": torch.tensor(10),
    }
    out = collate([a, b])
    assert out["audio_emb"].shape == (2, 20, _dim_or_skip())  # T 填充到 20
    assert out["section_bounds"].shape == (2, 4)  # S+1 填充到 4
    assert out["section_mask"].shape == (2, 3)  # S 填充到 3
    # A 有 3 真段，B 有 2 真段（第 3 段是 padding）
    assert out["section_mask"][0].tolist() == [True, True, True]
    assert out["section_mask"][1].tolist() == [True, True, False]
    # padding 帧的 audio_mask
    assert out["audio_mask"][0].all()  # A 全真
    assert out["audio_mask"][1, :10].all()  # B 前 10 帧真
    assert not out["audio_mask"][1, 10:].any()  # B 后续 padding


def test_loss_ignores_padding() -> None:
    """loss_fn 用 section_mask 过滤 padding 段：padding 段 target 改值不影响 loss。"""
    planner = DensityPlanner(n_layers=2, n_heads=4, dim=_dim_or_skip())
    B, S = 2, 4
    bounds = _make_bounds(B, S, dur_s=8.0)
    audio_emb = torch.randn(B, 40, _dim_or_skip())  # T=40
    diff = torch.tensor([5, 10])
    mask = torch.tensor([[True, True, True, False], [True, True, True, True]])

    pred = planner(audio_emb, diff, bounds, section_mask=mask)

    target_a = {
        "density": torch.tensor([[0.1, 0.5, 0.9, 0.0], [0.2, 0.8, 0.4, 0.6]]),
        "energy": torch.tensor([[0.1, 0.5, 0.9, 0.0], [0.2, 0.8, 0.4, 0.6]]),
        "rest": torch.tensor([[0.0, 0.1, 0.0, 0.0], [0.1, 0.0, 0.2, 0.1]]),
        "type": torch.tensor([[0, 2, 4, 0], [1, 3, 2, 0]]),
    }
    loss1 = planner.loss_fn(pred, target_a)

    # 改 padding 段（样本 0 第 4 段）的 target，loss 应不变
    target_b = {
        "density": torch.tensor([[0.1, 0.5, 0.9, 0.99], [0.2, 0.8, 0.4, 0.6]]),
        "energy": torch.tensor([[0.1, 0.5, 0.9, 0.99], [0.2, 0.8, 0.4, 0.6]]),
        "rest": torch.tensor([[0.0, 0.1, 0.0, 0.99], [0.1, 0.0, 0.2, 0.1]]),
        "type": torch.tensor([[0, 2, 4, 3], [1, 3, 2, 0]]),  # padding 段 type 改成 3
    }
    loss2 = planner.loss_fn(pred, target_b)
    assert torch.allclose(
        loss1, loss2, atol=1e-6
    ), f"padding 段 target 改动不应影响 loss：{loss1.item()} vs {loss2.item()}"
    # loss 有限
    assert torch.isfinite(loss1)


def test_loss_no_mask_backward_compat() -> None:
    """无 section_mask（plan() / B=1 路径）loss_fn 走原逻辑，可反传。"""
    planner = DensityPlanner(n_layers=2, n_heads=4, dim=_dim_or_skip())
    B, S = 1, 3
    bounds = _make_bounds(B, S, dur_s=6.0)
    audio_emb = torch.randn(B, 30, _dim_or_skip())
    diff = torch.tensor([5])
    pred = planner(audio_emb, diff, bounds)  # 不传 mask → section_mask=None
    assert pred["section_mask"] is None
    target = {
        "density": torch.tensor([[0.1, 0.5, 0.9]]),
        "energy": torch.tensor([[0.1, 0.5, 0.9]]),
        "rest": torch.tensor([[0.0, 0.1, 0.0]]),
        "type": torch.tensor([[0, 2, 4]]),
    }
    loss = planner.loss_fn(pred, target)
    assert torch.isfinite(loss)
    loss.backward()  # 确认可反传


def _dim_or_skip() -> int:
    """取 planner dim（与 DensityPlanner 默认一致）。"""
    return 8  # 测试用小 dim（n_layers=2, n_heads=4, dim=8 可整除）