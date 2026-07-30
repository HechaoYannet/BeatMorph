"""Stage 1：全局密度规划模块。

奠基文档 §3.3。6 层双向 Transformer，自监督回归每 4 小节的
density_target / energy_level / rest_probability。
损失：Huber Loss + 相邻段落平滑约束（TV Loss）。
伪标签从 osu! 谱面自动统计，零人工标注。

详细计划：docs/plans/03-planner-density.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from beatmorph.core.contracts import Section


class DensityPlanner:
    """全局密度规划骨架。"""

    def __init__(self, n_layers: int = 6, section_bars: int = 4) -> None:
        self.n_layers = n_layers
        self.section_bars = section_bars
        raise NotImplementedError("Stage 1 规划模块尚未实现，见 docs/plans/03-planner-density.md")

    def plan(
        self,
        audio_emb: "torch.Tensor",      # noqa: ARG002
        difficulty: int,                 # noqa: ARG002
        style_emb: "torch.Tensor | None" = None,  # noqa: ARG002
    ) -> list[Section]:
        """生成宏观布局蓝图。

        Args:
            audio_emb: ``[B, T_seq, 768]`` 音频 embedding。
            difficulty: 难度 1-15。
            style_emb: RAG 检索的风格向量（可选）。
        Returns:
            每个 Section 的目标密度/能量/段落类型。
        """
        ...
