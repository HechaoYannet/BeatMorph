"""MERTAdapter 单元测试。

- 离线结构测试（不需权重）：模块构建相关、Adapter 选型校验。
- 真实 MERT 测试（需 GPU + 权重，标 ``@pytest.mark.gpu``）：encode 形状/帧率/冻结/可训参数。
  权重未就位时 skip，不失败。
"""

from __future__ import annotations

import pytest
import torch

from beatmorph.core.contracts import MERT_DEFAULT_FEAT_DIM, MERT_FRAME_RATE_HZ

pytestmark = [pytest.mark.gpu]


def _weights_available() -> bool:
    """尝试能否解析到 MERT 权重（HF cache / ModelScope 已拉）。仅探测，不下载。"""
    try:
        from transformers import AutoConfig

        AutoConfig.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
        return True
    except Exception:
        return False


def _cuda_available() -> bool:
    return torch.cuda.is_available()


@pytest.fixture()
def require_mert() -> None:
    if not _cuda_available():
        pytest.skip("no CUDA")
    if not _weights_available():
        pytest.skip("MERT weights not cached; set HF_HOME/modelscope cache or pre-download")


class TestMERTAdapterStructure:
    """无需权重/无需 GPU 的离线结构校验。"""

    def test_is_nn_module(self) -> None:
        from beatmorph.audio.encoder.mert import MERTAdapter

        assert issubclass(MERTAdapter, torch.nn.Module)

    def test_invalid_adapter_rejected(self) -> None:
        from beatmorph.audio.encoder.mert import MERTAdapter

        with pytest.raises(ValueError, match="adapter"):
            # 会尝试加载权重前的参数校验先触发（构造内 adapter 校验在加载前）
            MERTAdapter(adapter="bogus", source="huggingface", device="cpu")

    def test_invalid_layer_rejected(self) -> None:
        from beatmorph.audio.encoder.mert import MERTAdapter

        with pytest.raises(ValueError, match="layer"):
            MERTAdapter(layer=-1, source="huggingface", device="cpu")

    def test_use_demucs_phase1_not_implemented(self) -> None:
        from beatmorph.audio.encoder.mert import MERTAdapter

        # use_demucs=True 在加载主干后才触发；但 Phase1 demucs stub 应报 NotImplementedError
        with pytest.raises(NotImplementedError, match="Demucs"):
            MERTAdapter(use_demucs=True, source="huggingface", device="cpu")

    def test_mlp_adapter_module_built(self) -> None:
        """_MLPAdapter 结构与前向（不依赖主干，避免 1.3GB 下载）。"""
        from beatmorph.audio.encoder.mert import _MLPAdapter

        mlp = _MLPAdapter(MERT_DEFAULT_FEAT_DIM)
        x = torch.randn(2, 10, MERT_DEFAULT_FEAT_DIM)
        out = mlp(x)
        assert out.shape == (2, 10, MERT_DEFAULT_FEAT_DIM)

    def test_merge_overlapping_shape(self) -> None:
        from beatmorph.audio.encoder.mert import _FRAME_RATE, _TARGET_SR, MERTAdapter

        # 两段，每段 [B=1, T=125, feat]，第二段从 4s 处开始（hop=4s）
        feat = MERT_DEFAULT_FEAT_DIM
        seg1 = torch.ones(1, 125, feat)
        seg2 = torch.ones(1, 125, feat) * 2.0
        # starts/hop 用样本数表达 4s，必须用 _TARGET_SR（GPU 修复后 24kHz，见 TRAINING_LOG Bug1）
        hop = int(4.0 * _TARGET_SR)
        starts = [0, hop]
        merged = MERTAdapter._merge_overlapping([seg1, seg2], starts, hop, _FRAME_RATE)
        # 总帧数 = 第二段尾 = round(4s × 派生帧率) + 125；帧率不再写死（75Hz → 425）
        expected = round(4.0 * _FRAME_RATE) + 125
        assert merged.shape[0] == 1
        assert merged.shape[1] == expected == 425
        assert merged.shape[-1] == feat


class TestMERTAdapterReal:
    """需真实 MERT 权重 + GPU。"""

    def test_encode_shape_and_frame_rate(self, require_mert: None) -> None:
        from beatmorph.audio.encoder.mert import _TARGET_SR, MERTAdapter

        adapter = MERTAdapter(
            model_name="m-a-p/MERT-v1-330M",
            layer=12,
            adapter="none",
            device="cuda",
            fp16=True,
        )
        dur_s = 2.0
        wav = torch.randn(1, int(dur_s * _TARGET_SR))
        emb = adapter.encode(wav)

        assert emb.dim() == 3
        assert emb.shape[0] == 1
        assert emb.shape[-1] == MERT_DEFAULT_FEAT_DIM
        # 帧率必须由主干 config 派生（MERT-v1-330M: prod(conv_stride)=320 → 24000/320=75Hz），
        # 且 T_seq ≈ dur × 派生帧率（±1 帧，卷积取整）。
        # 此前写死 25Hz，使这条唯一的真实断言与实现同错，漏检 3× 偏差。
        frame_rate = adapter.output_frame_rate()
        assert frame_rate == MERT_FRAME_RATE_HZ, (
            f"派生帧率 {frame_rate} != 契约 {MERT_FRAME_RATE_HZ}"
        )
        expected = round(dur_s * frame_rate)
        assert abs(emb.shape[1] - expected) <= 1, f"T_seq={emb.shape[1]}, expected~{expected}"

    def test_backbone_frozen_no_adapter(self, require_mert: None) -> None:
        from beatmorph.audio.encoder.mert import MERTAdapter

        adapter = MERTAdapter(adapter="none", device="cuda")
        # adapter="none" 时无任何可训参数
        trainable = [p for p in adapter.parameters() if p.requires_grad]
        assert len(trainable) == 0

    def test_lora_trainable_params_nonempty(self, require_mert: None) -> None:
        from beatmorph.audio.encoder.mert import MERTAdapter

        adapter = MERTAdapter(adapter="lora", device="cuda")
        trainable = [p for p in adapter.parameters() if p.requires_grad]
        assert len(trainable) > 0
        # 主干参数应冻结（LoRA 只让 lora 参数可训）
        backbone_trainable = [n for n, p in adapter.backbone.named_parameters() if p.requires_grad]
        # LoRA 后 backbone 内只剩 lora_* 可训
        assert all("lora" in n.lower() for n in backbone_trainable)