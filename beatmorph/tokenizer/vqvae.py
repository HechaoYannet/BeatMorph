"""Stage 2 前置：谱面语义 Tokenizer（VQ-VAE）。

奠基文档 §3.2。将一小节内的 Note 集合量化为单个离散 Token（码本 2048/4096）。
码本语义涵盖密度等级、节奏型、手型倾向、键位空间分布。
Codebook Collapse 缓解：K-means 初始化 + 增大 commitment loss + 随机重启（R-2）。

详细计划：docs/plans/02-tokenizer-vqvae.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from beatmorph.core.contracts import Chart, PatternToken


class VQVAETokenizer:
    """VQ-VAE 谱面 Tokenizer 骨架。

    训练目标（奠基文档 §3.2.1）：重建损失 + commitment loss + 码本利用率约束。
    验收：重建准确率 > 95%（Phase 1 里程碑）。
    """

    def __init__(self, codebook_size: int = 2048, bars_per_token: int = 1) -> None:
        self.codebook_size = codebook_size
        self.bars_per_token = bars_per_token
        raise NotImplementedError("VQ-VAE Tokenizer 尚未实现，见 docs/plans/02-tokenizer-vqvae.md")

    def encode(self, chart: Chart) -> list[PatternToken]:  # noqa: ARG002
        """将完整谱面编码为 Pattern Token 序列。

        Args:
            chart: 规范中间表示谱面。
        Returns:
            按小节顺序排列的 PatternToken 列表。
        """
        ...

    def decode(self, tokens: list[PatternToken]) -> Chart:  # noqa: ARG002
        """将 Token 序列解码回 Note 序列（Stage3 解码器调用）。"""
        ...

    def codebook_usage(self) -> float:
        """返回当前码本利用率（0-1，用于监控 R-2 坍缩）。"""
        ...
