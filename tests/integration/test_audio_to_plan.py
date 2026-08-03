"""MERT.encode → DensityPlanner.plan 端到端集成测试。

- 合成 audio_emb 路径（不需 MERT 权重）：直接喂随机 [B,T,feat] 给 planner。
- 真实 MERT 路径（需 GPU + 权重，标 ``@gpu``+``@slow``）：真实 wav → MERT.encode → plan。
"""

from __future__ import annotations

import pytest
import torch

from beatmorph.core.contracts import MERT_DEFAULT_FEAT_DIM
from beatmorph.planner.density import DensityPlanner


class TestPlannerOnSyntheticAudioEmb:
    """不依赖 MERT 权重：直接喂合成 audio_emb。CPU 可跑。"""

    def test_synthetic_to_sections(self) -> None:
        planner = DensityPlanner(n_layers=2)
        dur = 32.0  # 120BPM 4/4 → bar=2s; section_bars=4 → 8s/section → 4 sections
        t_seq = round(dur * 25)
        audio_emb = torch.randn(1, t_seq, MERT_DEFAULT_FEAT_DIM)

        sections = planner.plan(
            audio_emb,
            difficulty=8,
            bpm_points=[
                __import__("beatmorph.core.contracts", fromlist=["BpmPoint"]).BpmPoint(
                    time=0.0,
                    bpm=120.0,
                )
            ],
            duration_s=dur,
        )

        assert len(sections) >= 1
        # 覆盖总时长、无重叠
        assert sections[0].start_time == 0.0
        assert abs(sections[-1].end_time - dur) < 1e-3
        for i in range(len(sections) - 1):
            assert abs(sections[i].end_time - sections[i + 1].start_time) < 1e-3
        for s in sections:
            assert 0.0 <= s.density_target <= 1.0
            assert s.sections_type in ("intro", "verse", "chorus", "bridge", "outro")

    def test_training_loop_smoke(self) -> None:
        """合成数据跑 2 步训练，loss 可反传、不崩。"""
        planner = DensityPlanner(n_layers=2)
        opt = torch.optim.AdamW(planner.parameters(), lr=1e-3)

        for _ in range(2):
            B, S = 2, 4
            audio_emb = torch.randn(B, 200, MERT_DEFAULT_FEAT_DIM)
            difficulty = torch.tensor([5, 10])
            bounds = torch.linspace(0, 8, S + 1).unsqueeze(0).expand(B, -1).contiguous()
            pred = planner(audio_emb, difficulty, bounds)
            target = {
                "density": torch.rand(B, S),
                "energy": torch.rand(B, S),
                "rest": torch.rand(B, S),
                "type": torch.randint(0, 5, (B, S)),
            }
            loss = planner.loss_fn(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
        assert torch.isfinite(loss)


@pytest.mark.gpu()
@pytest.mark.slow()
class TestRealMertToPlanner:
    """真实 MERT 权重 + GPU。wav → MERT.encode → planner.plan。"""

    def test_real_audio_to_plan(self, require_gpu: None) -> None:
        from beatmorph.audio.encoder.mert import MERTAdapter

        # 尝试权重就位；否则 skip
        try:
            from transformers import AutoConfig

            AutoConfig.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
        except Exception:
            pytest.skip("MERT weights not cached")

        enc = MERTAdapter(adapter="none", device="cuda", fp16=True)
        sr = 16000
        dur = 2.0
        wav = torch.randn(1, int(dur * sr))
        emb = enc.encode(wav)  # [1, T_seq, feat]
        assert emb.shape[-1] == MERT_DEFAULT_FEAT_DIM

        planner = DensityPlanner(n_layers=2).cuda()
        secs = planner.plan(
            emb,
            difficulty=8,
            bpm_points=[
                __import__("beatmorph.core.contracts", fromlist=["BpmPoint"]).BpmPoint(
                    time=0.0,
                    bpm=120.0,
                )
            ],
            duration_s=dur,
        )
        assert len(secs) >= 1
        assert abs(secs[-1].end_time - dur) < 1.0
