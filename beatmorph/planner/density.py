"""Stage 1：全局密度规划模块。

奠基文档 §3.3。6 层双向 Transformer，自监督回归每 4 小节（一个 ``Section``）的
``density_target / energy_level / rest_probability`` 与段落类型（5 类）。

损失：Huber Loss（3 回归）+ TV Loss（相邻段落平滑）+ CE（段落类型）。
伪标签从 osu! 谱面自动统计（``compute_section_stats``），零人工标注。

详细计划：docs/plans/03-planner-density.md

输入契约（不可破）：``audio_emb [B, T_seq, 768]`` 来自 Stage 0（MERT）。
按 Section 时间边界把帧级 ``audio_emb`` mean-pool 成段级序列，喂 6 层双向
Transformer，输出 3 回归头（sigmoid [0,1]）+ 5 类分类头。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from beatmorph.core.contracts import (
    MERT_DEFAULT_FEAT_DIM,
    Section,
)
from beatmorph.core.logging import get_logger

if TYPE_CHECKING:
    from beatmorph.core.contracts import BpmPoint

logger = get_logger(__name__)

# ── 常量 ──────────────────────────────────────────────────────
_DEFAULT_METER = 4  # 默认拍号 4/4
_DEFAULT_N_HEADS = 8
_TYPE_CLASSES = 5
_TYPE_NAMES = ("intro", "verse", "chorus", "bridge", "outro")
_FRAME_RATE = 25.0  # audio_emb 帧率 Hz（契约 MERT_FRAME_RATE_HZ）


class DensityPlanner(nn.Module):
    """6 层双向 Transformer 全局密度规划器。

    Args:
        n_layers: Transformer 层数（默认 6，奠基 §3.3）。
        n_heads: 注意力头数。
        dim: 特征维（默认 768，与 MERT 对齐免投影）。
        section_bars: 每 Section 小节数（默认 4，契约）。
        n_type_classes: 段落类型类别数（默认 5）。
        dropout: dropout 率。
    """

    def __init__(
        self,
        n_layers: int = 6,
        n_heads: int = _DEFAULT_N_HEADS,
        dim: int = MERT_DEFAULT_FEAT_DIM,
        section_bars: int = 4,
        n_type_classes: int = _TYPE_CLASSES,
        dropout: float = 0.1,
        # loss 权值
        huber_weight: float = 1.0,
        tv_weight: float = 0.1,
        ce_type_weight: float = 0.5,
    ) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim({dim}) 必须能被 n_heads({n_heads}) 整除")
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = dim
        self.section_bars = section_bars
        self.n_type_classes = n_type_classes
        self.huber_weight = huber_weight
        self.tv_weight = tv_weight
        self.ce_type_weight = ce_type_weight

        # difficulty 标量 1-15 → embedding
        self.difficulty_embed = nn.Embedding(16, dim)  # 0..15，1-15 实际用

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 多任务头
        self.head_density = nn.Linear(dim, 1)
        self.head_energy = nn.Linear(dim, 1)
        self.head_rest = nn.Linear(dim, 1)
        self.head_type = nn.Linear(dim, n_type_classes)

    # ── 前向 ──────────────────────────────────────────────────

    def forward(
        self,
        audio_emb: torch.Tensor,
        difficulty: torch.Tensor,
        section_bounds: torch.Tensor,
        style_emb: torch.Tensor | None = None,
        section_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """前向：段级回归 + 段落类型。

        Args:
            audio_emb: ``[B, T_seq, dim]`` Stage0 输出（padding-aware collate 后 T_seq 取批次最大）。
            difficulty: ``[B]`` long，难度 1-15。
            section_bounds: ``[B, S+1]`` 每 batch 的 Section 时间边界秒，
                ``section_bounds[:, :-1]`` 为各 Section 起始时间，
                ``section_bounds[:, 1:]`` 为各 Section 结束时间。S = Section 数（padding 段 bounds=[0,0]）。
            style_emb: ``[B, top_k, dim]`` RAG 风格向量（可选，Phase 1 不用）。
            section_mask: ``[B, S]`` bool，真段 True / padding 段 False。``loss_fn`` 据此过滤 padding。
                None 表示无 padding（B=1 推理 / plan() 路径），此时 loss_fn 走原逻辑。
        Returns:
            dict 含 ``density/energy/rest [B,S,1]``（sigmoid 到 [0,1]）、
            ``type_logits [B,S,5]``、及透传的 ``section_mask``。
        """
        if audio_emb.dim() != 3 or audio_emb.shape[-1] != self.dim:
            raise ValueError(f"audio_emb 须 [B,T_seq,{self.dim}]，得到 {tuple(audio_emb.shape)}")

        # ── 段级 mean-pool：按 section_bounds 把帧级 audio_emb 聚成 [B, S, dim] ──
        sec_emb = self._pool_sections(audio_emb, section_bounds)  # [B, S, dim]

        # ── difficulty embedding 前缀 ──
        diff = difficulty.to(sec_emb.device).long().clamp(1, 15)
        diff_vec = self.difficulty_embed(diff).unsqueeze(1)  # [B, 1, dim]
        seq = torch.cat([diff_vec, sec_emb], dim=1)  # [B, S+1, dim]

        if style_emb is not None:
            # Phase 1 不强约束，预留拼接（沿序列维前缀）
            seq = torch.cat([style_emb.mean(dim=1, keepdim=True), seq], dim=1)

        out = self.transformer(seq)
        # 去掉 difficulty 前缀（及可能的 style 前缀），取段级输出
        prefix = 1 + (1 if style_emb is not None else 0)
        sec_out = out[:, prefix:, :]  # [B, S, dim]

        density = torch.sigmoid(self.head_density(sec_out))  # [B, S, 1]
        energy = torch.sigmoid(self.head_energy(sec_out))
        rest = torch.sigmoid(self.head_rest(sec_out))
        type_logits = self.head_type(sec_out)  # [B, S, 5]

        return {
            "density": density,
            "energy": energy,
            "rest": rest,
            "type_logits": type_logits,
            "section_mask": section_mask,
        }

    @torch.inference_mode()
    def plan(
        self,
        audio_emb: torch.Tensor,
        difficulty: int,
        style_emb: torch.Tensor | None = None,
        section_bounds: torch.Tensor | None = None,
        bpm_points: list[BpmPoint] | None = None,
        duration_s: float | None = None,
    ) -> list[Section]:
        """生成宏观布局蓝图（plan 03 §3.1 契约）。

        Args:
            audio_emb: ``[B, T_seq, 768]``（B=1）。
            difficulty: 难度 1-15。
            style_emb: RAG 风格向量（可选）。
            section_bounds: ``[1, S+1]`` 时间边界秒；若 None 则需 ``bpm_points``
                + ``duration_s`` 自动推。
            bpm_points: ``list[BpmPoint]``，推边界用。
            duration_s: 总时长秒，推边界用。
        Returns:
            每个 Section 的目标密度/能量/休息/段落类型。
        """
        if audio_emb.dim() == 2:
            audio_emb = audio_emb.unsqueeze(0)
        if audio_emb.shape[0] != 1:
            raise ValueError("plan() 期望 B=1")

        device = audio_emb.device
        if section_bounds is None:
            if bpm_points is None or duration_s is None:
                raise ValueError("需提供 section_bounds 或 (bpm_points + duration_s)")
            section_bounds = torch.tensor(
                [section_boundaries_from_bpm(bpm_points, duration_s, self.section_bars)],
                dtype=torch.float32,
                device=device,
            )
        else:
            section_bounds = section_bounds.to(device)

        device_diff = torch.tensor([difficulty], device=device)
        pred = self.forward(audio_emb, device_diff, section_bounds, style_emb)
        bounds = section_bounds[0].tolist()
        type_idx = pred["type_logits"][0].argmax(dim=-1).tolist()
        sections: list[Section] = []
        for i in range(len(bounds) - 1):
            sections.append(
                Section(
                    index=i,
                    start_time=float(bounds[i]),
                    end_time=float(bounds[i + 1]),
                    bar_count=self.section_bars,
                    density_target=float(pred["density"][0, i, 0]),
                    energy_level=float(pred["energy"][0, i, 0]),
                    rest_probability=float(pred["rest"][0, i, 0]),
                    sections_type=_TYPE_NAMES[type_idx[i]],
                )
            )
        return sections

    # ── Loss ──────────────────────────────────────────────────

    def loss_fn(
        self,
        pred: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """多任务损失：Huber×3 + TV + CE(type)。

        Args:
            pred: ``forward`` 输出（含 density/energy/rest/type_logits ``[B,S,*]`` 及
                可选 ``section_mask [B,S]``）。
            target: dict 含 ``density/energy/rest [B,S]`` (float [0,1])、
                ``type [B,S]`` (long 0..4)。
        """
        mask = pred.get("section_mask")  # [B,S] bool or None
        den = pred["density"].squeeze(-1)  # [B, S]

        if mask is None:
            # 无 padding（B=1 推理 / plan() 路径）：原逻辑
            l_den = F.huber_loss(den, target["density"])
            l_eng = F.huber_loss(pred["energy"].squeeze(-1), target["energy"])
            l_rst = F.huber_loss(pred["rest"].squeeze(-1), target["rest"])
            l_reg = l_den + l_eng + l_rst
            tv = (
                (den[:, 1:] - den[:, :-1]).abs().mean()
                if den.shape[1] > 1
                else torch.zeros((), device=den.device)
            )
            l_type = F.cross_entropy(
                pred["type_logits"].reshape(-1, self.n_type_classes),
                target["type"].reshape(-1).long(),
            )
        else:
            # padding-aware：仅真段（mask=True）参与 loss
            m = mask
            l_den = F.huber_loss(den[m], target["density"][m])
            l_eng = F.huber_loss(pred["energy"].squeeze(-1)[m], target["energy"][m])
            l_rst = F.huber_loss(pred["rest"].squeeze(-1)[m], target["rest"][m])
            l_reg = l_den + l_eng + l_rst
            # TV：仅相邻两段都为真时计差，跳过 padding 边界
            adj = m[:, 1:] & m[:, :-1]  # [B, S-1]
            if adj.any():
                tv = (den[:, 1:][adj] - den[:, :-1][adj]).abs().mean()
            else:
                tv = torch.zeros((), device=den.device)
            l_type = F.cross_entropy(
                pred["type_logits"][m],
                target["type"][m].long(),
            )
        return self.huber_weight * l_reg + self.tv_weight * tv + self.ce_type_weight * l_type

    # ── 段级 pool ─────────────────────────────────────────────

    @staticmethod
    def _pool_sections(
        audio_emb: torch.Tensor,
        section_bounds: torch.Tensor,
    ) -> torch.Tensor:
        """按时间边界把帧级 audio_emb mean-pool 成段级 [B, S, dim]。

        Args:
            audio_emb: ``[B, T_seq, dim]``，帧率 25Hz。
            section_bounds: ``[B, S+1]`` 秒。
        """
        batch, t_seq, dim = audio_emb.shape
        n_sec = section_bounds.shape[1] - 1
        device = audio_emb.device

        # 帧 index → 时间（秒）：frame t / 25
        frame_times = torch.arange(t_seq, device=device, dtype=audio_emb.dtype) / _FRAME_RATE  # [T]
        out = audio_emb.new_zeros(batch, n_sec, dim)
        for s in range(n_sec):
            t_start = section_bounds[:, s].to(device)  # [B]
            t_end = section_bounds[:, s + 1].to(device)  # [B]
            # 对每个 batch 的 mask：t_start <= frame_time < t_end
            mask = (frame_times.unsqueeze(0) >= t_start.unsqueeze(1)) & (
                frame_times.unsqueeze(0) < t_end.unsqueeze(1)
            )  # [B, T]
            counts = mask.sum(dim=1).clamp(min=1)  # [B]
            pooled = (audio_emb * mask.unsqueeze(-1)).sum(dim=1) / counts.unsqueeze(-1)
            out[:, s, :] = pooled
        return out


# ── 辅助：从 bpm_points 推 Section 时间边界 ──────────────────


def section_boundaries_from_bpm(
    bpm_points: list[BpmPoint],
    duration_s: float,
    section_bars: int = 4,
    meter: int = _DEFAULT_METER,
) -> list[float]:
    """按 bpm_points + section_bars 推 Section 时间边界（秒）。

    返回 ``[t0, t1, ..., tN]``，长度 = Section 数 + 1，末点 >= duration_s。
    复用 plan 08 ``_compute_bar_boundaries`` 的小节边界思路，按 section_bars 步进取边界。
    """
    if not bpm_points:
        return [0.0, float(duration_s)]

    boundaries: list[float] = [0.0]
    current = 0.0
    bp_idx = 0
    bars_since_section = 0

    while current < duration_s:
        while bp_idx + 1 < len(bpm_points) and bpm_points[bp_idx + 1].time <= current:
            bp_idx += 1
        bpm = max(bpm_points[bp_idx].bpm, 1e-6)
        bar_dur = meter * 60.0 / bpm
        current += bar_dur
        bars_since_section += 1
        if bars_since_section >= section_bars:
            boundaries.append(min(current, duration_s))
            bars_since_section = 0

    if boundaries[-1] < duration_s:
        boundaries.append(float(duration_s))
    return boundaries
