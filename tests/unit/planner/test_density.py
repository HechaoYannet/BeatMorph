"""DensityPlanner 单元测试（CPU 可跑，不需权重/GPU）。

覆盖：forward 形状/值域、plan→Section、loss 反传、section_boundaries_from_bpm、
difficulty 极值、style_emb=None 可跑、dim%n_heads 校验。
"""

from __future__ import annotations

import pytest
import torch

from beatmorph.core.contracts import BpmPoint
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
        B, T = 2, 250  # 10s @25Hz
        audio_emb = torch.randn(B, T, 768)
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
        out = m(torch.randn(2, 100, 768), torch.tensor([1, 15]), _make_bounds(2, 5, 10.0))
        assert out["density"].shape == (2, 5, 1)

    def test_style_emb_none_runs(self) -> None:
        m = DensityPlanner()
        out = m(
            torch.randn(1, 100, 768), torch.tensor([8]), _make_bounds(1, 5, 10.0), style_emb=None
        )
        assert out["type_logits"].shape == (1, 5, 5)

    def test_style_emb_with_runs(self) -> None:
        m = DensityPlanner()
        sty = torch.randn(1, 3, 768)  # top_k=3
        out = m(
            torch.randn(1, 100, 768), torch.tensor([8]), _make_bounds(1, 5, 10.0), style_emb=sty
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
            torch.randn(1, int(dur * _FRAME_RATE), 768),
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
            torch.randn(1, int(dur * _FRAME_RATE), 768),
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
        out = m(torch.randn(B, 200, 768), torch.tensor([5, 10]), _make_bounds(B, S, 10.0))
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
