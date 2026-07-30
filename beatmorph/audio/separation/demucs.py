"""Stage 0 辅助：声源分离（Demucs / HTDemucs）。

奠基文档 §3.1.2，标记为「可选」。开启时输出 {drums, bass, vocals, other} 四轨，
分别过轻量 encoder 后沿特征维拼接；快速推理可跳过。

详细计划：docs/plans/01-audio-encoder.md §可选增强
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

# 默认四轨名称（奠基文档 §3.1.2）
STEMS: tuple[str, ...] = ("drums", "bass", "vocals", "other")


class DemucsSeparator:
    """Demucs (HTDemucs) 声源分离骨架。"""

    def __init__(self, model_name: str = "htdemucs", device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        raise NotImplementedError("声源分离未实现，见 docs/plans/01-audio-encoder.md")

    def separate(self, wav: "torch.Tensor") -> dict[str, "torch.Tensor"]:  # noqa: ARG002
        """将混合音频分解为四轨。

        Args:
            wav: ``[B, T_samples]`` 混合音频。
        Returns:
            ``{stem: [B, T_samples]}`` 映射，键见 :data:`STEMS`。
        """
        ...
