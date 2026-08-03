"""Stage 2 前置：谱面语义 Tokenizer（VQ-VAE）。

奠基文档 §3.2。将一小节内的 Note 集合量化为单个离散 Token（码本 2048/4096）。
码本语义涵盖密度等级、节奏型、手型倾向、键位空间分布。

架构（奠基 §3.2.1 表 + plan 02 §4）：
    per-bar 1D-CNN 编码 → 2 层小节间 Transformer（上下文增强）→
    直通量化（straight-through argmin）→ 轻量 MLP + 1D-CNN 解码。

防坍缩三连（R-2）：K-means 初始化码本 + 增大 commitment loss + 长期死码随机重启。

栅格化口径与数据流水线同源：复用
``beatmorph.data.parsers.osu_path._compute_bar_boundaries`` 推小节边界（RFC-0005
变速点分段），与 ``compute_section_stats`` 切 Section 的边界完全一致。

详细计划：docs/plans/02-tokenizer-vqvae.md
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from beatmorph.core.contracts import (
    CODEBOOK_BASE,
    BpmPoint,
    Chart,
    GameMode,
    Note,
    NoteType,
    PatternToken,
)
from beatmorph.core.logging import get_logger
from beatmorph.data.parsers.osu_path import _compute_bar_boundaries

logger = get_logger(__name__)

# ── 常量 ──────────────────────────────────────────────────────
_TIME_BINS = 64  # 每 bar 时间栅格数（16 分音 ×4，半 bin 15.6ms@120bpm < ±20ms 容差）
_FEAT = 6  # 5 NoteType one-hot + 1 duration 标量
_N_NOTE_TYPES = 5  # NoteType 枚举数（TAP/HOLD/MINE/ROLL/FAKE），plan 07 §3.2 对齐
_DEFAULT_LATENT = 256
_DEFAULT_HEADS = 4
_DEFAULT_LAYERS = 2
_CNN_CHANNELS: tuple[int, ...] = (64, 128)
_MAX_BARS = 1024  # bar positional embedding 上界（超长由 bar_mask 截断）
_LOSS_DIV_EPS = 1e-6


class VQVAETokenizer(nn.Module):
    """VQ-VAE 谱面 Tokenizer（奠基 §3.2.1，标准 VQ-VAE）。

    per-bar 1D-CNN + 小节间 Transformer 编码 → straight-through 量化 →
    MLP + 1D-CNN 解码。每小节产出一个离散 code。

    验收（奠基 §7 Phase1）：重建准确率 > 95%。

    Args:
        codebook_size: 码本大小（默认 :data:`CODEBOOK_BASE`=2048，精细档 4096）。
        latent: encoder / 码本维（需能被 transformer_heads 整除）。
        lane: 键位数（默认 4，契约 :data:`DEFAULT_LANE_COUNT`）。
        time_bins: 每 bar 时间栅格数（默认 64）。
        feat: 栅格特征维（默认 6 = 5 NoteType + 1 duration）。
        cnn_channels: per-bar 1D-CNN 各层通道数。
        transformer_layers: 小节间 Transformer 层数（默认 2）。
        transformer_heads: 注意力头数（默认 4）。
        dropout: dropout 率。
        commit_weight: commitment loss 权重（R-2 增大，默认 0.25）。
        codebook_weight: 标准 codebook loss 权重（默认 1.0）。
        util_weight: 码本熵约束权重 γ（默认 0.1）。
        type_weight / present_weight / dur_weight: 重建三项权重。
        dead_code_steps: 连续未激活步数阈值，超过视为死码（默认 500）。
    """

    def __init__(
        self,
        codebook_size: int = CODEBOOK_BASE,
        latent: int = _DEFAULT_LATENT,
        lane: int = 4,
        time_bins: int = _TIME_BINS,
        feat: int = _FEAT,
        cnn_channels: tuple[int, ...] = _CNN_CHANNELS,
        transformer_layers: int = _DEFAULT_LAYERS,
        transformer_heads: int = _DEFAULT_HEADS,
        dropout: float = 0.1,
        # loss 权值
        commit_weight: float = 0.25,
        codebook_weight: float = 1.0,
        util_weight: float = 0.1,
        type_weight: float = 1.0,
        present_weight: float = 1.0,
        present_pos_weight: float = 15.0,
        dur_weight: float = 0.5,
        dead_code_steps: int = 500,
    ) -> None:
        super().__init__()
        if latent % transformer_heads != 0:
            raise ValueError(
                f"latent({latent}) 必须能被 transformer_heads({transformer_heads}) 整除"
            )
        if codebook_size < 1:
            raise ValueError("codebook_size 须 ≥ 1")

        self.codebook_size = codebook_size
        self.latent = latent
        self.lane = lane
        self.time_bins = time_bins
        self.feat = feat
        self.transformer_heads = transformer_heads
        self.commit_weight = commit_weight
        self.codebook_weight = codebook_weight
        self.util_weight = util_weight
        self.type_weight = type_weight
        self.present_weight = present_weight
        self.present_pos_weight = present_pos_weight
        self.dur_weight = dur_weight
        self.dead_code_steps = dead_code_steps
        # present 头正类权重（buffer 跟随 device，缓解稀疏 bin 类不平衡，见 RFC-0027 §1）
        self.register_buffer(
            "_present_pos_weight", torch.tensor(present_pos_weight, dtype=torch.float32)
        )

        in_ch = lane * feat
        channels = (in_ch, *cnn_channels, latent)
        # ── per-bar 1D-CNN 编码器 ──
        enc_layers: list[nn.Module] = []
        for i in range(len(channels) - 1):
            enc_layers.append(nn.Conv1d(channels[i], channels[i + 1], kernel_size=3, padding=1))
            enc_layers.append(nn.GELU())
        self.encoder_cnn = nn.Sequential(*enc_layers)
        self.encoder_pool = nn.AdaptiveAvgPool1d(1)

        # ── 小节间 Transformer（上下文增强）──
        self.bar_pos_embed = nn.Embedding(_MAX_BARS, latent)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=latent,
            nhead=transformer_heads,
            dim_feedforward=latent * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=transformer_layers)

        # ── 码本 ──
        self.codebook = nn.Parameter(torch.empty(codebook_size, latent))
        nn.init.normal_(self.codebook, mean=0.0, std=0.02)
        # 码本使用计数（死码重启判定），register_buffer 跟随 state_dict
        self.register_buffer("code_usage_count", torch.zeros(codebook_size, dtype=torch.long))

        # ── 解码器（MLP + 1D-CNN 反向，三路输出）──
        self.dec_hidden = max(cnn_channels[-1], latent)
        self.dec_proj = nn.Linear(latent, self.dec_hidden * time_bins)
        self.dec_cnn = nn.Sequential(
            nn.Conv1d(self.dec_hidden, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(128, lane * _N_NOTE_TYPES + lane + lane, kernel_size=3, padding=1),
        )

    # ── 前向 ──────────────────────────────────────────────────

    def forward(
        self,
        bar_grid: torch.Tensor,
        bar_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """前向：栅格 → 量化 → 解码。

        Args:
            bar_grid: ``(B, bars, lane, time_bins, feat)`` 栅格化 batch 张量。
            bar_mask: ``(B, bars)`` bool，True=真小节 / False=padding。``None`` 表示无 padding。
        Returns:
            dict 含 ``z_e/z_q/z_q_st`` ``(B,bars,latent)``、``indices`` ``(B,bars)`` long、
            ``type_logits`` ``(B,bars,lane,time_bins,5)``、
            ``present`` ``(B,bars,lane,time_bins)``、
            ``duration`` ``(B,bars,lane,time_bins)``、及透传的 ``bar_mask``。
        """
        if bar_grid.dim() != 5:
            raise ValueError(
                f"bar_grid 须 (B,bars,lane,time_bins,feat)，得到 {tuple(bar_grid.shape)}"
            )
        b, bars, lane, tb, feat = bar_grid.shape
        if lane != self.lane or tb != self.time_bins or feat != self.feat:
            raise ValueError(
                f"bar_grid lane/time_bins/feat 不符：期望 {self.lane}/{self.time_bins}/{self.feat}"
                f"，得到 {lane}/{tb}/{feat}"
            )

        device = bar_grid.device

        # ── per-bar 1D-CNN 编码 ──
        x = bar_grid.reshape(b * bars, lane * feat, tb)  # (B*bars, lane*feat, time_bins)
        x = self.encoder_cnn(x)  # (B*bars, latent, time_bins)
        x = self.encoder_pool(x).squeeze(-1)  # (B*bars, latent)
        z_e = x.reshape(b, bars, self.latent)

        # ── 小节间 Transformer ──
        pos = torch.arange(bars, device=device).clamp(max=_MAX_BARS - 1)
        z_e_ctx = z_e + self.bar_pos_embed(pos).unsqueeze(0)  # (B, bars, latent)
        if bar_mask is not None:
            # padding bar 不应被注意；用 src_key_padding_mask（True 表示忽略）
            z_e_ctx = self.transformer(z_e_ctx, src_key_padding_mask=~bar_mask)
        else:
            z_e_ctx = self.transformer(z_e_ctx)

        # ── 量化（straight-through，bf16 强制 fp32 距离防抖动）──
        z_q, indices = self._quantize(z_e_ctx)
        z_q_st = z_e_ctx + (z_q - z_e_ctx).detach()  # straight-through

        # ── 死码计数 ──
        assert isinstance(self.code_usage_count, torch.Tensor)
        with torch.no_grad():
            self.code_usage_count.index_add_(
                0, indices.reshape(-1), torch.ones(indices.numel(), dtype=torch.long, device=device)
            )

        # ── 解码 ──
        type_logits, present, duration = self._decode_latent(z_q_st)

        return {
            "z_e": z_e_ctx,
            "z_q": z_q,
            "z_q_st": z_q_st,
            "indices": indices,
            "type_logits": type_logits,
            "present": present,
            "duration": duration,
            "bar_mask": bar_mask,
        }

    def _quantize(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """最近邻查表量化。

        Args:
            z_e: ``(B, bars, latent)``。
        Returns:
            (z_q ``(B,bars,latent)`` 与 z_e 同 dtype，indices ``(B,bars)`` long)。
        """
        dtype = z_e.dtype
        b, bars, _ = z_e.shape
        flat = z_e.reshape(-1, self.latent).float()  # fp32 距离
        cb = (
            self.codebook.detach().float()
        )  # detach 保证距离计算不回流到码本（码本由 codebook loss 更新）
        # ‖z‖² - 2 z·eᵀ + ‖e‖²
        d = (
            flat.pow(2).sum(-1, keepdim=True) - 2.0 * flat @ cb.t() + cb.pow(2).sum(-1)
        )  # (N, codebook_size)
        indices = d.argmin(-1)  # (N,)
        z_q = self.codebook[indices].to(dtype).reshape(b, bars, self.latent)
        return z_q, indices.reshape(b, bars)

    def _decode_latent(self, z_q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """码本行（或量化后 latent）→ 三路解码输出。

        Args:
            z_q: ``(B, bars, latent)``。
        Returns:
            type_logits ``(B,bars,lane,time_bins,5)``、
            present ``(B,bars,lane,time_bins)``、
            duration ``(B,bars,lane,time_bins)``。
        """
        dtype = z_q.dtype
        b, bars, _ = z_q.shape
        h = self.dec_proj(z_q).reshape(b * bars, self.dec_hidden, self.time_bins)
        h = self.dec_cnn(h)  # (B*bars, lane*(5+1+1), time_bins)
        out = h.reshape(b, bars, self.lane, _N_NOTE_TYPES + 2, self.time_bins)
        type_logits = out[..., :_N_NOTE_TYPES, :]  # (B,bars,lane,5,time_bins)
        # 注意：解码器最后一轴是 time_bins，类型维在倒数第二
        type_logits = type_logits.permute(0, 1, 2, 4, 3).contiguous()  # (B,bars,lane,time_bins,5)
        present = out[..., _N_NOTE_TYPES, :]  # (B,bars,lane,time_bins)
        duration = out[..., _N_NOTE_TYPES + 1, :]
        return type_logits, present.to(dtype), duration.to(dtype)

    # ── Loss ──────────────────────────────────────────────────

    def loss_fn(  # noqa: PLR0915  # 多分项 loss，拆分损可读性
        self,
        pred: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """VQ-VAE 损失：recon + commit + codebook + util。

        Args:
            pred: ``forward`` 输出（含 z_e/z_q/indices/type_logits/present/duration/bar_mask）。
            target: dict 含 ``bar_grid`` ``(B,bars,lane,time_bins,feat)``（栅格真值，
                feat 末维 = [5 one-hot(type) | 1 duration]）。可由 :func:`rasterize` 产出。
        Returns:
            标量 loss。分项 loss 挂到 ``pred``（``loss_recon/loss_commit/loss_util``）便于 log。
        """
        bar_grid = target["bar_grid"]
        bar_mask = pred.get("bar_mask")  # (B,bars) or None
        b, bars, _lane, _tb, _feat = bar_grid.shape

        # ── 真值：present / type_idx / duration ──
        grid = bar_grid[..., :_N_NOTE_TYPES] >= 0.5  # one-hot 存在性 (B,bars,lane,tb,5)
        present_tgt = grid.any(-1).to(torch.float32)  # (B,bars,lane,tb)
        # 该 bin 的 type（取 one-hot 的 argmax；空 bin 默认 0=TAP，会被 present mask 掉）
        type_tgt = grid.long().argmax(-1)  # (B,bars,lane,tb)
        dur_tgt = bar_grid[..., _N_NOTE_TYPES]  # (B,bars,lane,tb)

        present = pred["present"].float()
        type_logits = pred["type_logits"].float()
        duration = pred["duration"].float()
        z_e = pred["z_e"].float()
        z_q = pred["z_q"].float()

        # ── 小节 mask：把 padding bar 的 present_tgt 抹 0 ──
        # present_tgt 驱动 present_bool，CE/MSE/dur 均只在 present_bool=True 位计算，
        # 故 padding 段（present_tgt=0）天然不参与 type/dur loss；type_tgt/type_logits
        # 无需单独 mask。
        if bar_mask is not None:
            m = bar_mask.view(b, bars, 1, 1)  # (B,bars,1,1) 广播到 (B,bars,lane,tb)
            present_tgt = present_tgt * m
            dur_tgt = dur_tgt * m
            # z_e / z_q padding bar 置 0（commit/codebook loss 不计入 padding）
            zm = bar_mask.view(b, bars, 1)
            z_e = z_e * zm
            z_q = z_q * zm

        # ── L_present：masked BCE + pos_weight（仅真小节 bin；缓解稀疏类不平衡，RFC-0027）──
        # pos_weight 给正类（有 Note bin）加权，防 present 头被空 bin 主导压低。
        assert isinstance(self._present_pos_weight, torch.Tensor)
        pos_w = self._present_pos_weight.to(present.dtype).to(present.device)
        if bar_mask is not None:
            valid_bin = bar_mask.view(b, bars, 1, 1).expand_as(present_tgt).to(present.dtype)
            n_valid = valid_bin.sum().clamp(min=1.0)
            l_present = (
                F.binary_cross_entropy_with_logits(
                    present, present_tgt, pos_weight=pos_w, reduction="none"
                )
                * valid_bin
            ).sum() / n_valid
            # 仅真小节 indices 计入码本熵（排除 padding bar）
            bar_valid = bar_mask.reshape(-1)
            valid_indices = pred["indices"].reshape(-1)[bar_valid]
        else:
            l_present = F.binary_cross_entropy_with_logits(
                present, present_tgt, pos_weight=pos_w, reduction="mean"
            )
            valid_indices = pred["indices"].reshape(-1)

        # ── L_type：masked CE（仅 present 位）──
        present_bool = present_tgt > 0.5
        l_type = present_bool.new_tensor(0.0)
        if present_bool.any():
            l_type = F.cross_entropy(
                type_logits[present_bool],  # (M,5)
                type_tgt[present_bool].long(),  # (M,)
            )

        # ── L_dur：masked MSE（仅 present 且 HOLD/ROLL）──
        hold_types = torch.tensor((int(NoteType.HOLD), int(NoteType.ROLL)), device=type_tgt.device)
        hold_mask = present_bool & torch.isin(type_tgt, hold_types)
        l_dur = hold_mask.new_tensor(0.0)
        if hold_mask.any():
            l_dur = F.mse_loss(duration[hold_mask], dur_tgt[hold_mask])

        l_recon = (
            self.type_weight * l_type + self.present_weight * l_present + self.dur_weight * l_dur
        )

        # ── commitment + codebook loss（R-2 增大 commit）──
        l_commit = F.mse_loss(z_e, z_q.detach())  # commitment：拉近 encoder 输出与码本
        l_codebook = F.mse_loss(z_e.detach(), z_q)  # 码本向 encoder 输出靠拢

        # ── L_util：码本熵约束（γ·Σ p_k log p_k，最小化负熵推均匀使用）──
        # 仅真小节 indices 计入（排除 padding bar）
        l_util = z_e.new_zeros(())
        if valid_indices.numel() > 0:
            counts = torch.bincount(valid_indices, minlength=self.codebook_size).float()
            p = counts / counts.sum().clamp(min=_LOSS_DIV_EPS)
            nonzero = p > 0
            l_util = -(p[nonzero] * p[nonzero].log()).sum()

        loss = (
            l_recon
            + self.commit_weight * l_commit
            + self.codebook_weight * l_codebook
            + self.util_weight * l_util
        )

        pred["loss_recon"] = l_recon.detach()
        pred["loss_present"] = l_present.detach()
        pred["loss_type"] = (
            l_type.detach() if torch.is_tensor(l_type) else torch.tensor(float(l_type))
        )
        pred["loss_dur"] = l_dur.detach() if torch.is_tensor(l_dur) else torch.tensor(float(l_dur))
        pred["loss_commit"] = l_commit.detach()
        pred["loss_util"] = l_util.detach()
        return loss

    # ── 防坍缩三连（R-2）────────────────────────────────────

    @torch.no_grad()
    def kmeans_init(self, data: torch.Tensor, k: int | None = None, iters: int = 10) -> None:
        """K-means 初始化码本（R-2）。

        用首批样本的 z_e 聚类中心初始化 ``codebook``，样本不足 k 时用随机正态补齐。

        Args:
            data: ``(M, latent)`` 样本 latent（建议 encoder 前向采集后 detach）。
            k: 码本大小（默认 self.codebook_size）。
            iters: K-means 迭代次数。
        """
        k = k or self.codebook_size
        if k != self.codebook_size:
            raise ValueError(f"k({k}) 须与 codebook_size({self.codebook_size}) 一致")
        flat = data.detach().float().reshape(-1, self.latent)
        n = flat.shape[0]
        device = self.codebook.device

        if n == 0:
            logger.warning("K-means init: 无样本，保持随机正态码本")
            return

        if n >= k:
            # 随机选 k 个不重复样本作初始中心
            perm = torch.randperm(n, device=device)[:k]
            centers = flat[perm].clone()
        else:
            centers = flat.clone()
            # 不足 k 用随机正态补齐
            extra = torch.randn(k - n, self.latent, device=device) * 0.02
            centers = torch.cat([centers, extra], dim=0)

        for _ in range(iters):
            d = (
                centers.pow(2).sum(-1)
                - 2.0 * flat @ centers.t()
                + flat.pow(2).sum(-1, keepdim=True)
            )  # (N, k)
            assign = d.argmin(-1)  # (N,)
            for c in range(k):
                mask = assign == c
                if mask.any():
                    centers[c] = flat[mask].mean(0)

        self.codebook.data = centers.to(self.codebook.dtype).to(device)
        assert isinstance(self.code_usage_count, torch.Tensor)
        self.code_usage_count.zero_()

    @torch.no_grad()
    def restart_dead_codes(self, z_e_batch: torch.Tensor) -> int:
        """随机重启长期未激活的死码（R-2）。

        将 ``code_usage_count < dead_code_steps`` 的码本行重置为当前 batch 的随机样本向量。

        Args:
            z_e_batch: ``(B, bars, latent)`` 当前 batch 的 z_e。
        Returns:
            重启的死码数。
        """
        assert isinstance(self.code_usage_count, torch.Tensor)
        dead = (self.code_usage_count < self.dead_code_steps).nonzero(as_tuple=False).squeeze(-1)
        if dead.numel() == 0:
            return 0
        samples = z_e_batch.detach().reshape(-1, self.latent).float()
        if samples.shape[0] == 0:
            return 0
        picks_idx = torch.randint(0, samples.shape[0], (dead.numel(),), device=samples.device)
        picks = samples[picks_idx].to(self.codebook.dtype).to(self.codebook.device)
        self.codebook.data[dead] = picks
        self.code_usage_count[dead] = 0
        return int(dead.numel())

    def codebook_usage(self) -> float:
        """当前码本利用率（0-1，M3 监控 ≥0.5 红线）。

        取累计使用计数 > 0 的码本比例。
        """
        assert isinstance(self.code_usage_count, torch.Tensor)
        return float((self.code_usage_count > 0).float().mean().item())

    # ── Chart ↔ Token 往返 ──────────────────────────────────

    @torch.inference_mode()
    def encode(self, chart: Chart) -> list[PatternToken]:
        """将完整谱面编码为 Pattern Token 序列（按小节顺序）。

        Args:
            chart: 规范中间表示谱面。
        Returns:
            按小节顺序排列的 PatternToken 列表（含空小节）。
        """
        bar_grids, boundaries = rasterize_chart(chart, self.lane, self.time_bins)
        if len(bar_grids) == 0:
            return []
        grid = torch.stack(bar_grids).unsqueeze(0)  # (1, bars, lane, tb, feat)
        grid = grid.to(self.codebook.device).to(self.codebook.dtype)
        # 切 eval 关 dropout 确保推理确定性，结束后恢复原模式
        was_training = self.training
        self.eval()
        try:
            pred = self.forward(grid, bar_mask=None)
        finally:
            if was_training:
                self.train()
        codes = pred["indices"][0].tolist()  # (bars,)
        return [
            PatternToken(
                code=int(c),
                bar_index=i,
                start_time=float(boundaries[i]),
                duration_bars=1,
            )
            for i, c in enumerate(codes)
        ]

    @torch.inference_mode()
    def decode(self, tokens: list[PatternToken]) -> Chart:
        """将 Token 序列解码回 Chart（Plan 07 §4 口径）。

        Args:
            tokens: PatternToken 列表（按小节顺序）。
        Returns:
            Chart（mode=MANIA_4K，bpm_points 单元素常速近似）。
        """
        if not tokens:
            return Chart(
                bpm_points=[BpmPoint(time=0.0, bpm=120.0)],
                difficulty=1,
                mode=GameMode.MANIA_4K,
            )

        device = self.codebook.device
        dtype = self.codebook.dtype
        codes = torch.tensor([t.code for t in tokens], device=device, dtype=torch.long)
        z_q = self.codebook[codes].unsqueeze(0).to(dtype)  # (1, bars, latent)
        # 切 eval 关 dropout 确保推理确定性，结束后恢复原模式
        was_training = self.training
        self.eval()
        try:
            type_logits, present, duration = self._decode_latent(z_q)  # 各 (1,bars,lane,tb,...)
        finally:
            if was_training:
                self.train()

        type_logits = type_logits[0]  # (bars,lane,tb,5)
        present = present[0]  # (bars,lane,tb)
        duration = duration[0]  # (bars,lane,tb)

        n_bars = len(tokens)
        # bar_dur：从相邻 token.start_time 差推；末 bar 用倒数第二 bar dur 外推
        bar_durs: list[float] = []
        for i in range(n_bars):
            if i + 1 < n_bars:
                bar_durs.append(max(float(tokens[i + 1].start_time - tokens[i].start_time), 1e-3))
            elif i >= 1:
                bar_durs.append(bar_durs[i - 1])
            else:
                # 单 bar：用首个 token.duration_bars 与默认 BPM 推标准 bar 长度
                bar_durs.append(4 * 60.0 / 120.0)
        bar_durs = [max(d, 1e-3) for d in bar_durs]

        notes: list[Note] = []
        present_mask = present > 0.5  # (bars,lane,tb)
        # 仅在有 Note 的 bin 上迭代，避免扫描全 0 张量
        bars_idx, lanes_idx, tbs_idx = torch.nonzero(present_mask, as_tuple=True)
        for bi, li, ti in zip(bars_idx.tolist(), lanes_idx.tolist(), tbs_idx.tolist(), strict=True):
            ty = int(type_logits[bi, li, ti].argmax().item())
            note_type = NoteType(ty)
            start = float(tokens[bi].start_time)
            bar_dur = bar_durs[bi]
            # 中心对齐反映射
            # 拍点对齐反映射（RFC-0026，与 rasterize_bar 一致）：rel = tb / bins
            time = start + ti / self.time_bins * bar_dur
            dur = 0.0
            if note_type in (NoteType.HOLD, NoteType.ROLL):
                dur = max(0.0, float(duration[bi, li, ti].item()))
            notes.append(Note(time=time, lane=li, type=note_type, duration=dur))

        # bpm：用中位 bar_dur 反推（4/4：bar = 4*60/bpm → bpm = 240/bar_dur）
        median_bar_dur = sorted(bar_durs)[len(bar_durs) // 2]
        bpm = 240.0 / max(median_bar_dur, 1e-3)

        notes.sort(key=lambda n: (n.time, n.lane))
        return Chart(
            version="ir-1",
            mode=GameMode.MANIA_4K,
            difficulty=1,
            bpm_points=[BpmPoint(time=0.0, bpm=bpm)],
            notes=notes,
        )


# ── 栅格化 ──────────────────────────────────────────────────────


def rasterize_bar(
    notes: list[Note],
    bar_start: float,
    bar_dur: float,
    lane: int = 4,
    time_bins: int = _TIME_BINS,
) -> torch.Tensor:
    """将一小节内的 Note 集合栅格化为 ``(lane, time_bins, feat=6)`` 张量。

    feat 维 = [5 NoteType one-hot | 1 duration 标量]。中心对齐（bin 中心时间一一反映射）。
    空 bin 全 0；同 (lane, bin) 多 Note 冲突保留先到者（64 bin/bar 下罕见）。

    Args:
        notes: 本小节 Note（已按时间排序为佳）。
        bar_start: 小节起点秒。
        bar_dur: 小节时长秒（>0）。
        lane: 键位数。
        time_bins: 每 bar 栅格数。
    Returns:
        ``(lane, time_bins, 6)`` float32 张量。
    """
    grid = torch.zeros(lane, time_bins, _FEAT, dtype=torch.float32)
    if bar_dur <= 0 or not notes:
        return grid

    # 拍点对齐映射（RFC-0026）：tb = round(rel * bins) % bins。
    # 用 bins（非 bins-1）使音乐拍点 rel=0.25/0.5/0.75 精确落 bin=bins/4 等，
    # rel=1.0（下一 bar 起点）wrap 回 bin 0。中心对齐 round(rel*(bins-1)) 会把
    # 拍点偏移到 %4==3（实证 50% note 错位），故改用左对齐 bin 区间 + wrap。
    for note in notes:
        rel = (note.time - bar_start) / bar_dur
        if rel < 0.0 or rel >= 1.0 + 1e-6:
            continue  # 不属于本小节
        tb = round(rel * time_bins) % time_bins
        li = note.lane
        if li < 0 or li >= lane:
            continue
        ty = int(note.type)
        if ty < 0 or ty >= _N_NOTE_TYPES:
            continue
        # one-hot 冲突：若该 bin 已有 Note，跳过（保留先到者）
        if grid[li, tb, :_N_NOTE_TYPES].sum() > 0:
            continue
        grid[li, tb, ty] = 1.0
        if note.is_hold():
            grid[li, tb, _N_NOTE_TYPES] = float(note.duration)
    return grid


def rasterize_chart(
    chart: Chart,
    lane: int = 4,
    time_bins: int = _TIME_BINS,
) -> tuple[list[torch.Tensor], list[float]]:
    """将整张谱面栅格化为每小节张量列表。

    小节边界复用 :func:`_compute_bar_boundaries`（与数据流水线 ``compute_section_stats``
    同源口径，RFC-0005 变速点分段推）。``total_duration`` 取值同 ``compute_section_stats``：
    优先 ``chart.meta["audio_duration"]``，否则末 Note 时长。

    Args:
        chart: 谱面 IR。
        lane: 键位数（默认 4）。
        time_bins: 每 bar 栅格数。
    Returns:
        (每小节 ``(lane, time_bins, 6)`` 张量列表, 小节边界秒列表)。
        空谱面返回 ([], [])。
    """
    from beatmorph.core.contracts import GameMode

    sorted_notes = chart.sorted_notes()
    if not sorted_notes:
        return [], []

    note_duration = sorted_notes[-1].time + max(n.duration for n in sorted_notes)
    audio_dur = chart.meta.get("audio_duration")
    audio_total = (
        float(audio_dur) if isinstance(audio_dur, int | float) and audio_dur > 0 else note_duration
    )
    total_duration = max(audio_total, note_duration)

    boundaries = _compute_bar_boundaries(chart.bpm_points, total_duration)
    if len(boundaries) < 2:
        return [], []

    grids: list[torch.Tensor] = []
    for i in range(len(boundaries) - 1):
        t_start = float(boundaries[i])
        t_end = float(boundaries[i + 1])
        bar_dur = t_end - t_start
        if bar_dur <= 0:
            continue
        bar_notes = [n for n in sorted_notes if t_start <= n.time < t_end]
        grids.append(rasterize_bar(bar_notes, t_start, bar_dur, lane, time_bins))

    # 过滤超曲尾 padding（避免空小节）；但保留小节内的真实空段
    # 这里直接返回所有栅格化小节，空小节产 code（plan 02 测试要求）
    _ = GameMode  # 保 GameMode 引用（未来多 K 扩展按 mode 取 lane_count）
    return grids, [float(x) for x in boundaries]
