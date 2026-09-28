"""掩码补全 Encoder-Decoder 主干（Plan 04 §4.1/§4.2，里程碑 M2/M3）。

结构（与 plan 04 逐条对应）：

- **Encoder**：音频分支（音高无关的线性投影 + **以秒为基准**的位置编码，秒由
  `frame_rate` 派生，禁止写死帧号）+ 判定线事件轨分支（输入必须是**跨层求和后**的
  5 条普通轨，见 `batch.N_ORDINARY_TRACKS`；可选 extended 第 5 层）。
- **条件注入**（BasePlan §3.5）：难度 -> 可学习嵌入 -> 每层 AdaLN；线事件轨与音频
  -> cross-attention 的 K/V。
- **Decoder**：掩码补全。输入 = 可见场 + **遮盖通道**（硬要求，RFC-0029 §3.3-1）+
  位置编码 + line embedding；**多线共享权重**，K 不进任何输出层形状；
  局部层是**滑动窗口自注意力**（窗口以 tau 格为单位），且事件轨条件**只注入本线**
  （RFC-0032：局部层只看自己那条线的运动）；每 `global_period` 层插一个**全局层**
  （层内同时看到全部 K 条线的场与轨，RFC-0029 §2.4-4）——**跨线只发生在全局层**。
- **输出头**：默认 `factorized`（lambda = Lambda' * p，p 在 (X, S, C) 上归一化），
  于是 `int lambda` 精确、两条积分路径的一致性成为可断言条件（plan 03 §4.6）；
  `direct`（softplus 直接出 lambda）保留给消融与调试。
- **不做排列匹配**：没有 Hungarian matching、没有 line 分类头——判定线不可互换
  （RFC-0029 §2.4-3），line 的身份只经 line embedding 进入。

限制（v1，如实声明）：全局层对 (K x T) 个 token 做稠密注意力，显存 O((K*T)^2)；
生产训练必须分段（plan 04 §4.6 / R-04-4）。本模块的测试与门禁全部在**小规模 CPU** 上跑。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from beatmorph.core.contracts.tensors import MERT_DEFAULT_FEAT_DIM
from beatmorph.field.grid import FieldGrid
from beatmorph.field.integrate import factorized_lambda
from beatmorph.field.loss import assert_lambda_valid
from beatmorph.generation.batch import N_ORDINARY_TRACKS, FieldBatch, FieldOutput
from beatmorph.generation.losses import (
    apply_line_mask_batched,
    cumulative_lambda_batched,
    full_poisson_loss,
    masked_poisson_loss,
    occlusion_ratio,
)

#: 输出头口径
HeadMode = Literal["factorized", "direct"]
#: 自注意力作用域：局部（滑动窗口，逐线） / 全局（同 batch 内全 K 线）
LayerKind = Literal["local", "global"]

#: 默认 K 容量：覆盖实测长尾（中位 30 / p75 52 / 极值 82），留出余量
DEFAULT_K_MAX: int = 128


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """模型超参（全部可从 configs/ 覆盖；本模块不读配置文件）。

    Attributes:
        d_model: 隐维。
        n_heads: 注意力头数（必须整除 d_model）。
        n_layers: 解码器层数。
        window: 滑动窗口半径（tau 格）；局部层可见 2*window+1 个 tau 位置。
        global_period: 每 n 层插一个全局层（plan 04 §9-2 的待扫描超参）。
        k_max: line embedding 容量（覆盖实测极值；K 不进入输出层形状）。
        dropout: dropout 概率（0 表示确定性前向，采样期的方差置信度需要 > 0）。
        head: "factorized"（默认）/ "direct"。
        audio_dim: 音频特征维（MERT-v1-330M 各层均为 MERT_DEFAULT_FEAT_DIM）。
        extended_dim: extended 轨通道数；0 表示不启用（plan 04 §9-12）。
        position_max_period: 正弦位置编码的底座（超参，不是物理常量）。
        check_lambda: 前向是否做 lambda 非负 / 有限性硬检查（plan R-03-5）。
    """

    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    window: int = 32
    global_period: int = 4
    k_max: int = DEFAULT_K_MAX
    dropout: float = 0.0
    head: HeadMode = "factorized"
    audio_dim: int = MERT_DEFAULT_FEAT_DIM
    extended_dim: int = 0
    position_max_period: float = 10000.0
    check_lambda: bool = True

    def __post_init__(self) -> None:
        if self.d_model % 2 != 0:
            raise ValueError(f"d_model 必须为偶数（正弦位置编码）：{self.d_model}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"n_heads={self.n_heads} 必须整除 d_model={self.d_model}")
        if self.n_layers < 1:
            raise ValueError("n_layers 必须 >= 1")
        if self.window < 0 or self.global_period < 1:
            raise ValueError("window >= 0 且 global_period >= 1")

    def layer_kinds(self) -> tuple[LayerKind, ...]:
        """层类型排布：每 `global_period` 层一个全局层；最后**必须**是全局层。

        「最后必须是全局层」不是审美：只有全局层能看到全部 K 条线的场
        （RFC-0029 §2.4-4），它必须在输出头之前。
        """
        kinds: list[LayerKind] = ["local"] * self.n_layers
        for index in range(self.global_period - 1, self.n_layers, self.global_period):
            kinds[index] = "global"
        kinds[-1] = "global"
        return tuple(kinds)


def sinusoidal_encoding(positions: Tensor, d_model: int, *, max_period: float) -> Tensor:
    """正弦位置编码：positions 为任意形状（单位由调用方决定），输出 (... , d_model)。

    音频轴传**秒**（由 frame_rate 派生），场轴传 tau 格下标（plan 04 §9-5 的残留问题，
    v1 取格下标并在文档中标注）。
    """
    if d_model % 2 != 0:
        raise ValueError("d_model 必须为偶数")
    half = d_model // 2
    device = positions.device
    frequencies = torch.exp(
        -math.log(float(max_period))
        * torch.arange(half, dtype=torch.float32, device=device)
        / half,
    )
    arguments = positions.to(dtype=torch.float32).unsqueeze(-1) * frequencies
    return torch.cat([torch.cos(arguments), torch.sin(arguments)], dim=-1)


class AdaNorm(nn.Module):
    """AdaLN：由条件（难度嵌入）产生 scale / shift，作用在 token 上。"""

    def __init__(self, d_model: int, cond_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: Tensor, cond: Tensor) -> Tensor:
        """tokens (B, L, d)，cond (B, cond_dim) -> (B, L, d)。"""
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        # 显式标注：torch 的 nn.Module.__call__ 在 stub 里返回 Any（mypy strict 下必须落型）
        adapted: Tensor = self.norm(tokens) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return adapted


class DecoderLayer(nn.Module):
    """一层解码器：AdaLN -> 自注意力 -> 音频 cross-attention -> 轨道 cross-attention -> FFN。"""

    def __init__(self, config: ModelConfig, kind: LayerKind) -> None:
        super().__init__()
        self.kind: LayerKind = kind
        d_model = config.d_model
        self.ada = AdaNorm(d_model, d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.cross_audio = nn.MultiheadAttention(
            d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.cross_tracks = nn.MultiheadAttention(
            d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm_self = nn.LayerNorm(d_model)
        self.norm_audio = nn.LayerNorm(d_model)
        self.norm_tracks = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        tokens: Tensor,
        cond: Tensor,
        audio: Tensor,
        tracks: Tensor,
        *,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """tokens (N, L, d)；cond (N, d)；audio (N, Ta, d)；tracks (N, Tl, d)。"""
        prepared = self.ada(tokens, cond)
        attended, _ = self.self_attn(
            prepared,
            prepared,
            prepared,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        tokens = self.norm_self(tokens + attended)
        from_audio, _ = self.cross_audio(tokens, audio, audio, need_weights=False)
        tokens = self.norm_audio(tokens + from_audio)
        from_tracks, _ = self.cross_tracks(tokens, tracks, tracks, need_weights=False)
        tokens = self.norm_tracks(tokens + from_tracks)
        projected: Tensor = self.norm_ffn(tokens + self.ffn(tokens))
        return projected


class FieldTokenEmbedding(nn.Module):
    """可见场 + 遮盖通道 + 位置 + line embedding -> (B, K, T, d)。"""

    def __init__(self, config: ModelConfig, grid: FieldGrid) -> None:
        super().__init__()
        self.cells = grid.x_bins * grid.sides * grid.channels
        self.n_lines_embeddings = config.k_max
        self.visible = nn.Linear(self.cells, config.d_model)
        # 遮盖通道是**独立输入通路**（硬要求）：没有它模型无法区分「无 note」与「被遮盖」
        self.occlusion = nn.Linear(self.cells, config.d_model)
        self.line_embedding = nn.Embedding(config.k_max, config.d_model)
        self.norm = nn.LayerNorm(config.d_model)
        self.max_period = config.position_max_period

    def forward(
        self,
        state: Tensor,
        occlusion: Tensor,
        *,
        n_lines: int,
        t_bins: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        """state / occlusion 形状 (B, K, T, X, S, C) -> ((B, K, T, d), (B, K, T, 2 * cells))。

        第二个返回值是**该 token 自身的输入特征**（可见场 + 遮盖通道，拼在通道轴上），
        供输出头做**直连 skip**——见 `FieldHead` 的说明（补全任务里「输入的结构」必须是
        一条短梯度路径，否则优化会停在只学边际分布的盆地里）。
        """
        if n_lines > self.n_lines_embeddings:
            raise ValueError(
                f"K={n_lines} 超过 line embedding 容量 k_max={self.n_lines_embeddings}"
                "（plan 04 §9-8 的 batching 策略未定；请提高 k_max 或按 K 分桶）",
            )
        batch_size = int(state.shape[0])
        flattened = state.reshape(batch_size, n_lines, t_bins, self.cells)
        occlusion_flat = occlusion.reshape(batch_size, n_lines, t_bins, self.cells).to(
            dtype=state.dtype,
        )
        tokens = self.visible(flattened) + self.occlusion(occlusion_flat)
        positions = torch.arange(t_bins, dtype=torch.float32, device=device)
        tokens = tokens + sinusoidal_encoding(
            positions,
            self.norm.normalized_shape[0],
            max_period=self.max_period,
        ).reshape(1, 1, t_bins, -1)
        line_ids = torch.arange(n_lines, device=device)
        tokens = tokens + self.line_embedding(line_ids).reshape(1, n_lines, 1, -1)
        features = torch.cat([flattened, occlusion_flat], dim=-1)
        return self.norm(tokens), features


class FieldHead(nn.Module):
    """强度场输出头：lambda >= 0（因子化 -> int lambda 精确；直接 -> softplus）。

    **直连 skip（本模块的实测结论，plan 04 §9-17）**：输出头除了读解码器输出，还直接读
    该 token 自身的输入特征（可见场 + 遮盖通道）。没有这条短路径时，联合训练会停在
    「只学边际分布」的盆地：实测在「把可见位置复制到输出」这一最小任务上，全参训练
    800 步仍停在边际解（348.4，下界 150.2），而**冻结解码器只训头部与输入投影**立刻到达
    150.2——说明表达力够、优化路径不够。补上 skip 后该任务可直接从零学会。
    """

    def __init__(self, config: ModelConfig, grid: FieldGrid) -> None:
        super().__init__()
        self.mode: HeadMode = config.head
        self.x_bins = grid.x_bins
        self.sides = grid.sides
        self.channels = grid.channels
        self.cells = grid.x_bins * grid.sides * grid.channels
        if config.head == "factorized":
            self.cum_head = nn.Linear(config.d_model, 1)
            self.cell_head = nn.Linear(config.d_model, self.cells)
        else:
            self.rate_head = nn.Linear(config.d_model, self.cells)
        self.cum_skip = nn.Linear(2 * self.cells, 1)
        self.cell_skip = nn.Linear(2 * self.cells, self.cells)

    def forward(
        self,
        tokens: Tensor,
        grid: FieldGrid,
        *,
        line_mask: Tensor,
        range_mask: Tensor,
        input_features: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        """tokens (B, K, T, d) -> (lam, cum, cell_prob)；input_features 为该 token 的输入特征。"""
        batch_size, n_lines, t_bins, _ = tokens.shape
        if self.mode == "factorized":
            delta_logits = self.cum_head(tokens).squeeze(-1)
            cell_logits = self.cell_head(tokens)
            if input_features is not None:
                delta_logits = delta_logits + self.cum_skip(input_features).squeeze(-1)
                cell_logits = cell_logits + self.cell_skip(input_features)
            delta = F.softplus(delta_logits)
            probability = torch.softmax(cell_logits, dim=-1).reshape(
                batch_size,
                n_lines,
                t_bins,
                self.x_bins,
                self.sides,
                self.channels,
            )
            lam = factorized_lambda(delta, probability, grid)
            cum: Tensor | None = cumulative_lambda_batched(lam, grid)
        else:
            raw = self.rate_head(tokens)
            if input_features is not None:
                raw = raw + self.cell_skip(input_features)
            rate = F.softplus(raw).reshape(
                batch_size,
                n_lines,
                t_bins,
                self.x_bins,
                self.sides,
                self.channels,
            )
            lam = rate
            probability = None
            cum = None
        lam = lam * range_mask.reshape(1, 1, 1, -1, 1, 1).to(dtype=lam.dtype)
        lam = apply_line_mask_batched(lam, line_mask)
        return lam, cum, probability


class MaskedFieldModel(nn.Module):
    """掩码补全强度场模型（Plan 04 主线 B2）。"""

    def __init__(self, config: ModelConfig, grid: FieldGrid) -> None:
        super().__init__()
        self.config = config
        self.grid = grid
        d_model = config.d_model
        self.audio_norm = nn.LayerNorm(config.audio_dim)
        self.audio_proj = nn.Linear(config.audio_dim, d_model)
        track_dim = N_ORDINARY_TRACKS + (config.extended_dim if config.extended_dim > 0 else 0)
        self.track_norm = nn.LayerNorm(track_dim)
        self.track_proj = nn.Linear(track_dim, d_model)
        self.difficulty_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.embedding = FieldTokenEmbedding(config, grid)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, kind) for kind in config.layer_kinds()],
        )
        self.head = FieldHead(config, grid)
        self._band_cache: dict[tuple[int, int, torch.device, torch.dtype], Tensor] = {}

    # ── 条件分支 ────────────────────────────────────────────────
    def difficulty_embedding(self, difficulty: Tensor) -> Tensor:
        """定数 difficulty -> (B, d)；v1 用单位尺度正弦编码（**不做统计归一化**）。"""
        encoded = sinusoidal_encoding(
            difficulty.to(dtype=torch.float32),
            self.config.d_model,
            max_period=self.config.position_max_period,
        )
        embedded: Tensor = self.difficulty_proj(encoded)
        return embedded

    def encode_audio(self, batch: FieldBatch) -> Tensor:
        """(B, T_audio, D) -> (B, T_audio, d)；位置编码以**秒**为基准（由 frame_rate 派生）。"""
        if batch.frame_rate <= 0.0:
            raise ValueError(f"frame_rate 必须为正（派生量），得到 {batch.frame_rate!r}")
        audio = self.audio_proj(self.audio_norm(batch.audio_emb))
        time_steps = int(batch.audio_emb.shape[1])
        seconds = torch.arange(time_steps, dtype=torch.float32, device=audio.device) / float(
            batch.frame_rate,
        )
        positioned: Tensor = audio + sinusoidal_encoding(
            seconds,
            self.config.d_model,
            max_period=self.config.position_max_period,
        ).reshape(1, time_steps, -1)
        return positioned

    def encode_tracks(self, batch: FieldBatch) -> Tensor:
        """(B, K, T_line, F) -> (B, K*T_line, d)（普通轨 5 条 + 可选 extended）。"""
        tracks = batch.line_tracks
        if self.config.extended_dim > 0:
            if batch.extended_tracks is None:
                raise ValueError("配置启用了 extended 轨但 batch 未提供 extended_tracks")
            if int(batch.extended_tracks.shape[-1]) != self.config.extended_dim:
                raise ValueError(
                    f"extended_tracks 通道数应为 {self.config.extended_dim}，"
                    f"得到 {batch.extended_tracks.shape[-1]}",
                )
            tracks = torch.cat([tracks, batch.extended_tracks.to(dtype=tracks.dtype)], dim=-1)
        elif batch.extended_tracks is not None:
            raise ValueError(
                "batch 提供了 extended_tracks 但配置 extended_dim == 0（拒绝静默丢弃）"
            )
        batch_size, n_lines, t_line, _ = tracks.shape
        projected = self.track_proj(self.track_norm(tracks))
        positions = torch.arange(t_line, dtype=torch.float32, device=tracks.device)
        projected = projected + sinusoidal_encoding(
            positions,
            self.config.d_model,
            max_period=self.config.position_max_period,
        ).reshape(1, 1, t_line, -1)
        line_ids = torch.arange(n_lines, device=tracks.device)
        projected = projected + self.embedding.line_embedding(line_ids).reshape(1, n_lines, 1, -1)
        flattened: Tensor = projected.reshape(batch_size, n_lines * t_line, self.config.d_model)
        return flattened

    # ── 自注意力的两种作用域 ─────────────────────────────────────
    def band_mask(self, t_bins: int, *, device: torch.device, dtype: torch.dtype) -> Tensor | None:
        """滑动窗口掩码 (T, T) bool（True = 可见）；window < 0 或覆盖全长时为 None（全可见）。"""
        window = self.config.window
        if window < 0 or 2 * window + 1 >= t_bins:
            return None
        key = (t_bins, window, device, dtype)
        cached = self._band_cache.get(key)
        if cached is None:
            index = torch.arange(t_bins, device=device)
            distance = (index.reshape(-1, 1) - index.reshape(1, -1)).abs()
            cached = (distance <= window).to(dtype=dtype)
            self._band_cache[key] = cached
        return cached

    def decode(self, batch: FieldBatch, tokens: Tensor) -> Tensor:
        """把 (B, K, T, d) 过完整解码器栈（局部 / 全局层交替，见 ModelConfig.layer_kinds）。"""
        batch_size, n_lines, t_bins, d_model = tokens.shape
        cond = self.difficulty_embedding(batch.difficulty)
        audio = self.encode_audio(batch)
        tracks = self.encode_tracks(batch)
        line_mask = batch.line_mask_bool()
        # 局部层的**本线轨**（RFC-0032）：encode_tracks 把 K 条线拍平成一条
        # `K * T_line` 的序列——它的**序列维里已经含判定线轴**，因此**不能**再按线
        # repeat_interleave：那会让每条线的 token 都 attend 全部 K 条线的事件轨，
        # 代价是 `K^2 * T * T_line` 的 K/V（实测占 K=20 时 2.4x 的峰值显存，且显存随
        # K 二次增长；换成 SDPA 的 mem_efficient 核只省 8%，因为重复发生在注意力之前）。
        # reshape 之后每条线只拿到自己的 `T_line` 个轨 token，局部层的显存回到
        # O(K * T * T_line)。**跨线信息仍由全局层提供**（下面的 else 分支用未拆分的
        # `tracks` 作 K/V，正是 RFC-0029 §2.4-4 说的「全局层同时看到全部 K 条线」）。
        t_line = int(tracks.shape[1]) // n_lines
        own_tracks = tracks.reshape(batch_size, n_lines, t_line, d_model).reshape(
            batch_size * n_lines, t_line, d_model
        )
        for module in self.layers:
            # ModuleList 的元素静态类型是 Module；narrow 之后才能直接调用（mypy strict）
            assert isinstance(module, DecoderLayer)
            if module.kind == "local":
                flat = tokens.reshape(batch_size * n_lines, t_bins, d_model)
                local = module(
                    flat,
                    cond.repeat_interleave(n_lines, dim=0),
                    audio.repeat_interleave(n_lines, dim=0),
                    own_tracks,
                    attn_mask=self.band_mask(t_bins, device=flat.device, dtype=flat.dtype),
                )
                tokens = local.reshape(batch_size, n_lines, t_bins, d_model)
            else:
                flat = tokens.reshape(batch_size, n_lines * t_bins, d_model)
                padding = (
                    (~line_mask)
                    .reshape(batch_size, n_lines, 1)
                    .expand(
                        batch_size,
                        n_lines,
                        t_bins,
                    )
                )
                global_out = module(
                    flat,
                    cond,
                    audio,
                    tracks,
                    key_padding_mask=padding.reshape(batch_size, n_lines * t_bins),
                )
                tokens = global_out.reshape(batch_size, n_lines, t_bins, d_model)
        decoded: Tensor = tokens
        return decoded

    # ── 前向 ───────────────────────────────────────────────────
    def forward_state(
        self,
        batch: FieldBatch,
        state: Tensor,
        occlusion: Tensor,
        *,
        compute_loss: bool = False,
    ) -> FieldOutput:
        """用**任意**可见场状态前向（迭代并行解码的中间状态走这里）。

        Args:
            batch: 条件与网格（counts 可有可无；有则用于损失）。
            state: (B, K, T, X, S, C) 当前可见场（桶内计数或已确证的强度）。
            occlusion: (B, K, T, X, S, C) bool，True = 仍被遮盖。
            compute_loss: 是否计算训练损失（需要 batch.counts）。
        """
        expected = batch.batch_field_shape()
        if tuple(state.shape) != expected or tuple(occlusion.shape) != expected:
            raise AssertionError(f"state / occlusion 必须是 {expected}")
        tokens, features = self.embedding(
            state,
            occlusion,
            n_lines=batch.n_lines(),
            t_bins=batch.grid.t_bins,
            device=state.device,
        )
        tokens = self.decode(batch, tokens)
        lam, cum, probability = self.head(
            tokens,
            batch.grid,
            line_mask=batch.line_mask_bool(),
            range_mask=batch.range_mask_bool(),
            input_features=features,
        )
        if self.config.check_lambda:
            assert_lambda_valid(lam)
        diagnostics: dict[str, Tensor] = {
            "occlusion_ratio": torch.as_tensor(occlusion_ratio(batch), dtype=torch.float32),
            "lam_max": lam.amax(),
        }
        output = FieldOutput(
            lam=lam,
            cum=cum,
            cell_prob=probability,
            loss=None,
            diagnostics=diagnostics,
        )
        if compute_loss and batch.counts is not None:
            # RFC-0037 R2：训练/门禁两臂统一走 per_event 归一化（整式除以 max(E_total,1)，
            # argmin 不变、步间量级可比）。旧的 sum 口径见 loss_sum_raw 标量（对照用）。
            loss = (
                masked_poisson_loss(output, batch, reduction="per_event")
                if batch.occlusion is not None
                else full_poisson_loss(output, batch, reduction="per_event")
            )
            output = replace(output, loss=loss)
        return output

    def forward(self, batch: FieldBatch, *, compute_loss: bool = True) -> FieldOutput:
        """标准前向：可见场 = counts * (~occlusion)（RFC-0029 §3.3-1 的输入约定）。"""
        batch.assert_shapes()
        return self.forward_state(
            batch,
            batch.observed_counts(),
            batch.occlusion_bool(),
            compute_loss=compute_loss,
        )


__all__ = [
    "DEFAULT_K_MAX",
    "AdaNorm",
    "DecoderLayer",
    "FieldHead",
    "FieldTokenEmbedding",
    "HeadMode",
    "LayerKind",
    "MaskedFieldModel",
    "ModelConfig",
    "sinusoidal_encoding",
]
