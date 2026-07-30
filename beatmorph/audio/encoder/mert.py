"""Stage 0：多模态音频编码器（MERT-v1-330M + Adapter）。

奠基文档 §3.1。冻结 MERT，仅训练轻量 Adapter（LoRA/MLP），输出 25Hz、768 维
序列 embedding。Demucs 分轨为可选增强（见 beatmorph.audio.separation）。

详细计划：docs/plans/01-audio-encoder.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

# TODO(stage0): 实现 MERTAdapter
# - load frozen MERT-v1-330M (HuggingFace transformers)
# - 可训练 Adapter (LoRA 或 2 层 MLP) 注入到指定 transformer 层
# - forward(wav) -> audio_emb [B, T_seq, 768], T_seq = duration * 25


class MERTAdapter:
    """Stage 0 编码器骨架。后续按 docs/plans/01-audio-encoder.md 实现。"""

    def __init__(self, model_name: str = "m-a-p/MERT-v1-330M", layer: int = 12) -> None:
        self.model_name = model_name
        self.layer = layer
        raise NotImplementedError("Stage 0 编码器尚未实现，见 docs/plans/01-audio-encoder.md")

    def encode(self, wav: "torch.Tensor") -> "torch.Tensor":  # noqa: ARG002
        """编码音频为序列 embedding。

        Args:
            wav: ``[B, T_samples]`` 音频波形，16kHz 单声道。
        Returns:
            ``[B, T_seq, 768]`` 帧级 embedding，帧率 25Hz。
        """
        ...
