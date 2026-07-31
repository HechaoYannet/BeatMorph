"""偏好对齐：DPO（Direct Preference Optimization）。

奠基文档 §3.6。基于 Stage 2 AR 预训练模型，用 osu! 玩家评分/Pass Rate/Play Count
构造偏好对微调。无需独立奖励模型，训练成本约 RLHF 的 1/3。
优化目标：max log σ(β · (log π_chosen − log π_rejected))。

数据构造：Chosen = 高评分(≥4.5星) & 高 Pass 率；Rejected = 低评分(≤2星)。

详细计划：docs/plans/06-dpo-alignment.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class DPOTrainer:
    """DPO 偏好对齐骨架。"""

    def __init__(self, beta: float = 0.1, reference_free: bool = False) -> None:
        self.beta = beta
        self.reference_free = reference_free
        raise NotImplementedError("DPO 训练器尚未实现，见 docs/plans/06-dpo-alignment.md")

    def compute_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> torch.Tensor:
        """计算 DPO 损失。

        见奠基文档 §3.6.1：maximize log σ(β · ((π_c−π_r) − (π_ref_c−π_ref_r)))。
        """
        raise NotImplementedError

    def build_preference_pairs(self, ratings_db: str) -> None:
        """从 osu! 排行榜评分数据构造偏好对。"""
        raise NotImplementedError
