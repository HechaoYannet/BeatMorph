"""Stage 2：Pattern 序列生成主干（AR Transformer Decoder）。

奠基文档 §3.4.1。12-16 层自回归 Transformer，上下文 256 tokens（约 256 小节）。
条件注入：Cross-Attention（audio_emb）+ AdaLN（difficulty/style）。
训练：Teacher Forcing + Cross-Entropy。
备选未来升级：Flow Matching（见 docs/plans/04-generation.md §备选）。

详细计划：docs/plans/04-generation.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from beatmorph.core.contracts import PatternToken, Section


class ARTransformer:
    """Stage 2 自回归生成骨架。"""

    def __init__(
        self,
        n_layers: int = 12,
        n_heads: int = 16,
        context_tokens: int = 256,
        codebook_size: int = 2048,
    ) -> None:
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.context_tokens = context_tokens
        self.codebook_size = codebook_size
        raise NotImplementedError("Stage 2 AR 生成器尚未实现，见 docs/plans/04-generation.md")

    def generate(
        self,
        audio_emb: torch.Tensor,
        sections: list[Section],
        difficulty: int,
        rag_prefix: torch.Tensor | None = None,
        max_tokens: int = 256,
    ) -> list[PatternToken]:
        """自回归生成 Pattern Token 序列。

        Args:
            audio_emb: ``[B, T_seq, 768]`` 音频条件。
            sections: Stage 1 规划输出。
            difficulty: 难度 1-15。
            rag_prefix: RAG 检索的参考 Token 前缀（可选）。
            max_tokens: 最大生成长度。
        Returns:
            生成的 Pattern Token 序列。
        """
        raise NotImplementedError
