"""Stage 2 风格控制：RAG 检索增强。

奠基文档 §3.5。无需训练，百万级 osu! 谱面按 BPM/流派/谱师/难度索引，
Top-K=3，检索结果作为 Decoder 的 Prefix 或 Cross-Attention KV 注入。
检索特征：音频 MERT Embedding + 谱面密度曲线。

详细计划：docs/plans/05-rag-retrieval.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from beatmorph.core.contracts import EventToken

if TYPE_CHECKING:
    import torch


class RAGRetriever:
    """RAG 风格检索骨架。"""

    def __init__(self, top_k: int = 3, index_path: str | None = None) -> None:
        self.top_k = top_k
        self.index_path = index_path
        raise NotImplementedError("RAG 检索器尚未实现，见 docs/plans/05-rag-retrieval.md")

    def build_index(self, corpus_dir: str) -> None:
        """构建 FAISS 检索索引（音频 emb + 密度曲线联合向量）。"""
        raise NotImplementedError

    def retrieve(
        self,
        query_emb: torch.Tensor,
        difficulty: int,
        bpm: float,
    ) -> list[list[EventToken]]:
        """检索 Top-K 参考谱面的 event token 序列（RFC-0028，BPE/event）。

        Args:
            query_emb: 查询音频 embedding。
            difficulty: 目标难度，用于过滤。
            bpm: 目标 BPM，用于过滤/加权。
        Returns:
            Top-K 条参考 event token 序列（:class:`EventToken`）。
        """
        raise NotImplementedError
