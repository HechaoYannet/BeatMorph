"""Stage 2：event 序列生成主干（AR Transformer Decoder）。

奠基文档 §3.4.1（RFC-0028 后口径）。12-16 层自回归 Transformer，上下文 ~1024
event tokens（段落级，分段生成 + 衔接覆盖全曲，见 RFC-0008）。
条件注入：Cross-Attention（audio_emb）+ AdaLN（difficulty/style）。
训练：Teacher Forcing + Cross-Entropy。
备选未来升级：Flow Matching（见 docs/plans/04-generation.md §备选）。

详细计划：docs/plans/04-generation.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from beatmorph.core.contracts import (
    AR_CONTEXT_TOKENS,
    BPE_DEFAULT_VOCAB,
    BpmPoint,
    EventToken,
    Section,
)


class ARTransformer:
    """Stage 2 自回归生成骨架。"""

    def __init__(
        self,
        n_layers: int = 12,
        n_heads: int = 16,
        context_tokens: int = AR_CONTEXT_TOKENS,
        vocab_size: int = BPE_DEFAULT_VOCAB,
    ) -> None:
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.context_tokens = context_tokens
        self.vocab_size = vocab_size
        raise NotImplementedError("Stage 2 AR 生成器尚未实现，见 docs/plans/04-generation.md")

    def generate(
        self,
        audio_emb: torch.Tensor,
        sections: list[Section],
        difficulty: int,
        bpm_points: list[BpmPoint],
        rag_prefix: torch.Tensor | None = None,
        max_tokens: int = AR_CONTEXT_TOKENS,
    ) -> list[EventToken]:
        """自回归生成 event token 序列（RFC-0028，BPE/event）。

        Args:
            audio_emb: ``[B, T_seq, 768]`` 音频条件。
            sections: Stage 1 规划输出。
            difficulty: 难度 1-15。
            bpm_points: 目标谱面 BPM 变速点（决策 4：AR 不生成绝对 tempo，
                decode 期 POS→秒用此）。
            rag_prefix: RAG 检索的参考 event token 前缀（可选）。
            max_tokens: 最大生成长度（event 数，默认 :data:`AR_CONTEXT_TOKENS`）。
        Returns:
            生成的 event token 序列（:class:`EventToken`）。
        """
        raise NotImplementedError
