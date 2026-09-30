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
    #: 场 token 是否**额外**加一条「以秒为基准」的位置编码（plan 04 §5 的残留问题）。
    #:
    #: 为什么必须有它：窗口网格把 τ 轴重新标定为**窗口等效 BPM**（`bpm_eff = 60/J(τ_start)`，
    #: 见 `beatmorph/data/dataset.py` 的 `_window_grid`），即 τ 格 → 秒的换算因子
    #: **逐窗口不同**；而音频帧的位置编码以**秒**为基准（`encode_audio`）。模型此前只拿到
    #: τ **格下标**、拿不到 `J`，因此**无法把音频帧与场 token 对齐**（实测：音频帧轴置换只值
    #: 0.6% 的 NLL，plan 07 §9-61③）。打开它 = 把 τ 格映射到窗口内的秒数（唯一实现处
    #: `beatmorph/field/grid.py` 的 `tau_to_seconds`），使两条轴落在同一时间基上。
    seconds_position: bool = False
    #: 是否把**与 τ 对齐的音频帧**直接注入场 token（`audio_align`）。
    #:
    #: 为什么需要：探针实测（`runs/_diag_audio_probe.py`）表明，按 τ→秒对齐取出 MERT 帧、
    #: 用**线性**读出就能以 **val AUC 0.64** 预测「哪个 τ 切片有音符」（仅位置 0.54、随机 0.51）
    #: ——信息在音频里；但模型的 `cond_audio_perm_delta ≈ 0` 说明**它一点没用上**（cross-attention
    #: 这条路没学到对齐/没学出权重）。本开关绕开那条路：按 token 自己的秒数取音频帧与其差分，
    #: 投影后**直接加**到 token 嵌入上（与 `beamtmorph/field` 的 τ→秒换算同源，见 `FieldGrid.tau_seconds`）。
    audio_align: bool = False
    #: 条件里是否**真的给**判定线事件轨的内容（`track_input`）。
    #:
    #: 为什么需要这个开关（plan 07 §9-67）：e2e 的判定线事件轨是**从该曲真谱复制**来的，
    #: 而人类谱的可读性做法（线在音符时刻闪现/变速）让这些轨本身就编码了音符时刻——
    #: 实测「仅事件轨」的 (窗口,线) 组内 AUC 0.643，比音频（0.576）还高，且两者高度冗余
    #: （联合 0.629 < 事件轨单独）。于是「从音乐出谱」这件事在**条件里**就被解决了大半，
    #: 模型把依赖压在事件轨上（cond_track_zero 0.9-1.8）而音频边际掉到 ≈0。
    #: 关掉它 = 只保留**线身份与位置编码**（模型仍然知道「哪条线、第几拍」），
    #: 但拿不到任何运动/透明度/速度内容 ⇒ 内容只能来自音频。
    track_input: bool = True
    #: 输入侧是否**真的看到**可见场（`visible_input`）。
    #:
    #: 为什么需要这个开关（plan 07 §9-66）：训练送进去的可见场永远是「随机 50% 的事件」，
    #: 而推理（迭代并行解码）的**第一步**是「什么都还没确证」⇒ 可见场全 0、遮盖通道全 1。
    #: 模型因此从未在「没有任何可见证据」的制度下训练过，而那个制度正是「只能靠音乐出谱」
    #: 的制度。关掉它 = 把训练送进推理第一步的输入分布（可见场置 0、遮盖通道置 1），
    #: 迫使落点只能来自音频 + 事件轨 + τ/线先验；损失与遮盖测度**一个字不改**
    #: （事件项仍只监督被遮盖事件、仍按 1/r 重标定 ⇒ 仍是对全谱密度 MLE 的无偏估计）。
    visible_input: bool = True
    #: 输出场是否**按遮盖通道置零**（lam <- lam * occlusion）。
    #:
    #: 为什么需要这个开关（plan 07 §9-74）：遮盖测度下的最优 λ 在**未被遮盖的格子上恒为 0**——
    #: 事件项只监督被遮盖处，而积分项对任何非零强度都是纯成本。实测（`runs/_probe_mask_response.py`，
    #: armH 的 8k 快照）：模型的 λ 质量有 **55.6% 落在未被遮盖 token 上**（被遮盖 token 只占 44.2%），
    #: 即**与 token 数成正比、对遮盖毫无响应**；而在被遮盖 token 内部（= 真正被监督的子集），
    #: 组内 τ AUC 只有 **0.5130**。把 λ 按遮盖置零是**与被监督集合逐格对齐**的硬约束，
    #: 不引入任何新参数、也不改变损失语义（它只是把「最优解已知为 0 的那些格子」钉成 0）。
    #: ⚠️ 推理时（生成制度）遮盖通道全 1 ⇒ 该乘法是恒等，不影响迭代解码的第一步；
    #: 后续迭代步里仍未确证的格子仍为 1 ⇒ 只裁掉「已经确证」的地方，正是想要的语义。
    mask_lambda: bool = False
    #: 输出头是否保留**输入直连 skip**（`cell_skip` / `cum_skip`）。
    #:
    #: 原设计理由（plan 04 §9-17）：补全任务需要一条从「输入结构」到输出的**短梯度路径**，
    #: 否则优化会停在只学边际分布的盆地。但实测（`runs/_diag_anti.py`）在**真实任务**上它可能
    #: 是反向的：对**被遮盖** token，skip 的输入是常数 `[0…0, 1…1]` ⇒ 它只能给一个**与 token
    #: 无关**的空间先验，而实测该先验**与真实事件格反相关**（命中率 0.367 < 0.5），且 token 内
    #: 空间分布的熵只有 **1.88 / 7.16 nats**（近似 one-hot）。关掉它 = 逼 p(x,s,c) 只能来自 trunk。
    head_skip: bool = True
    #: 两条 skip 的**分开关**（plan 07 §9-68）：None = 跟随 `head_skip`（默认，向后兼容）。
    #:
    #: 为什么必须能分开：实测（`runs/_probe_tau_reward.py`）—— `head_skip=false` 之后模型对 τ 轴
    #: **完全无感**（把输出的 τ 轴整体置换：`val_nll` Δ = −7e-6…+6e-5，四个不同臂都一样），
    #: 而 `head_skip=true` 的老基线是 **+2.148**，两条轴的 x 置换都还在 +0.28…+0.44。
    #: 也就是说：**拆掉 skip 换来了空间轴（命中率 0.367→0.75、val_ratio 0.9987→0.762），
    #: 代价是把时间轴整个丢了**（skip 那条短路径曾经把「可见场在哪些 τ 上有证据」直接喂给 δ 分支，
    #: 而可见事件与隐藏事件同处一片区域 ⇒ δ 天然带 τ 形状）。
    #:
    #: `cum_skip` 只作用于**标量 δ(τ)**（每 token 一个数），它给不出「与 token 无关的空间先验」
    #: ——§9-62 指控的正是 `cell_skip` 那一侧（p(x,s,c) 的常数先验、命中率 0.367）。
    #: 因此实验臂 = `head_skip=false` + `head_cum_skip=true`：**要回 τ，不要回空间捷径**。
    head_cell_skip: bool | None = None
    head_cum_skip: bool | None = None
    #: 头的**无条件 τ 偏置**（plan 07 §9-77）。True ⇒ 在 `cum`（δ 分支）的 logits 上加一个
    #: 形状 (t_bins,) 的**可学习**偏置，softplus **之前**相加。
    #:
    #: 为什么必须有这条路：实测（`runs/_probe_tau_layer_trace.py`，1 批 CPU）
    #: 未训练模型 decode 输出沿 τ 的相对起伏是 **2.06**（架构能表达 τ），
    #: 而训练后同一个量是 **0.00000** —— 三个 local 自注意力层把它压成「沿 τ 不变」。
    #: 而 τ 位置编码只加在**输入** token 上、`head_skip=False` 时头只剩 decode 的输出
    #: ⇒ **头没有任何别的 τ 通路**，于是「学一张语料 τ 表」在当前装配下不可达。
    #: 那张表值多少：`runs/_probe_window_phase.py` 实测语料 τ 直方图在真实 val 上
    #: （窗,线）组内 AUC = **0.9292**（模型 0.5009），而 §9-69 量到 τ 轴占常数基线的 42.46%。
    #: 192 个参数、初始化为 0 ⇒ **开启时与关闭时逐位同构**，改动不改变既有语义。
    #: ⚠️ 与 `head_cum_skip` 的区别：skip 是「从**输入特征**读」，偏置是「无视输入直接给先验」
    #: ——前者在被遮盖 token 上输入恒为常数，后者不依赖输入。
    head_tau_bias: bool = False
    #: 空间分支的同一件事：在 `cell` 的 logits 上加形状 (t_bins, cells) 的可学习偏置。
    #: 语义 = 让头直接表达**无条件联合先验** p(τ, x, s, c)；条件调制仍全部来自解码器。
    head_cell_tau_bias: bool = False

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
        self.use_seconds_position: bool = bool(config.seconds_position)
        #: 见 ModelConfig.visible_input：True = 看得见可见场（补全制度），False = 生成制度。
        self.blind: bool = not bool(config.visible_input)

    def forward(
        self,
        state: Tensor,
        occlusion: Tensor,
        *,
        n_lines: int,
        t_bins: int,
        device: torch.device,
        seconds: Tensor | None = None,
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
        if self.blind:
            # 生成制度（plan 07 §9-66）：可见场全 0、遮盖通道全 1 —— 迭代解码第一步的输入。
            flattened = torch.zeros_like(flattened)
            occlusion_flat = torch.ones_like(occlusion_flat)
        tokens = self.visible(flattened) + self.occlusion(occlusion_flat)
        positions = torch.arange(t_bins, dtype=torch.float32, device=device)
        tokens = tokens + sinusoidal_encoding(
            positions,
            self.norm.normalized_shape[0],
            max_period=self.max_period,
        ).reshape(1, 1, t_bins, -1)
        if seconds is not None:
            # 与音频帧**同一时间基**（秒）的位置编码：音频轴走 encode_audio 的秒编码，
            # 场轴此前只有 τ 格下标 ⇒ 两条轴相差一个逐窗口不同的因子 J(τ_start)。
            tokens = tokens + sinusoidal_encoding(
                seconds.to(dtype=torch.float32).reshape(-1),
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
        self.use_skip: bool = bool(config.head_skip)
        #: 分开关（None = 跟随 head_skip）；见 ModelConfig.head_cell_skip 的说明。
        self.use_cell_skip: bool = (
            self.use_skip if config.head_cell_skip is None else bool(config.head_cell_skip)
        )
        self.use_cum_skip: bool = (
            self.use_skip if config.head_cum_skip is None else bool(config.head_cum_skip)
        )
        self.x_bins = grid.x_bins
        self.sides = grid.sides
        self.channels = grid.channels
        self.cells = grid.x_bins * grid.sides * grid.channels
        self.t_bins = grid.t_bins
        #: 两条无条件 τ 偏置（plan 07 §9-77）。None = 关闭（**不注册参数** ⇒ 旧 checkpoint 逐位兼容）。
        enabled = config.head == "factorized"
        self.tau_bias: nn.Parameter | None = (
            nn.Parameter(torch.zeros(self.t_bins)) if enabled and config.head_tau_bias else None
        )
        self.cell_tau_bias: nn.Parameter | None = (
            nn.Parameter(torch.zeros(self.t_bins, self.cells))
            if enabled and config.head_cell_tau_bias
            else None
        )
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
            if input_features is not None and self.use_cum_skip:
                delta_logits = delta_logits + self.cum_skip(input_features).squeeze(-1)
            if input_features is not None and self.use_cell_skip:
                cell_logits = cell_logits + self.cell_skip(input_features)
            # 无条件 τ 偏置：**绕开**「τ 位置只存在于输入 token、被 local 层平均掉」这条断链。
            # 加在 softplus/softmax **之前** ⇒ 它直接决定 ΔΛ(t) 与 p(τ,·) 的形状。
            if self.tau_bias is not None:
                delta_logits = delta_logits + self.tau_bias.reshape(1, 1, self.t_bins)
            if self.cell_tau_bias is not None:
                cell_logits = cell_logits + self.cell_tau_bias.reshape(
                    1, 1, self.t_bins, self.cells
                )
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
            if input_features is not None and self.use_cell_skip:
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
        #: 时间对齐音频的直连通道（`audio_align`）：输入 = [帧, 帧差分] ⇒ d_model。
        self.audio_align_proj: nn.Linear | None = (
            nn.Linear(2 * config.audio_dim, config.d_model) if config.audio_align else None
        )
        self._band_cache: dict[tuple[int, int, torch.device, torch.dtype], Tensor] = {}
        #: 空间 softmax 的**熵正则权重**（0 = 关闭）。由训练器从 `optim.cell_entropy_weight` 注入。
        #:
        #: 为什么需要：实测（`runs/_diag_flat.py`）把模型自己的输出**在 token 内摊平**后
        #: `val_ratio` 从 **0.8867 → 0.7351**（只摊平 x 轴 → **0.6697**）⇒ 它那套又尖又错的
        #: `(x,s,c)` 分布**比均匀还差**，代价约 0.19–0.28 nats/line，比其它任何效应大两个数量级。
        #: 惩罚项与事件项**同结构**：`w · (1/r) · Σ_被遮盖 n · H(p_token) / D`（H 为 token 内
        #: 空间分布的熵，D = max(E_total,1)），因此 `w` 与 `−log λ` 同量纲、可直接比较。
        self.cell_entropy_weight: float = 0.0

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

    def _aligned_frame_index(
        self,
        batch: FieldBatch,
        grid: FieldGrid,
        device: torch.device,
    ) -> Tensor:
        """(T,) 每个 τ 格对应的**本窗音频帧下标**（τ→秒只走 `FieldGrid.tau_seconds`）。

        这是「音频帧轴」与「τ 轴」之间**唯一**的换算处；它的正确性直接决定 `audio_align`
        注入的是不是「该时刻的音乐」。独立成方法是为了让它可被契约测试直接钉住
        （`tests/unit/generation/test_audio_alignment.py`）——此前它埋在 `_aligned_audio`
        里，错到「逐窗口差一个 BPM 因子」也没有任何断言拦得住。
        """
        seconds = self._field_seconds(grid, device=device).to(dtype=torch.float32)
        n_frames = max(1, int(batch.audio_emb.shape[1]))
        return torch.clamp(
            (seconds * float(batch.frame_rate)).round().to(dtype=torch.long),
            0,
            n_frames - 1,
        )

    def _aligned_audio(self, batch: FieldBatch, grid: FieldGrid, device: torch.device) -> Tensor:
        """(B, 1, T, d)：把**该 τ 切片对应时刻**的音频帧（及其差分）投影进 token 空间。

        帧下标由 `_aligned_frame_index` 给出（τ→秒的唯一实现处是 `FieldGrid.tau_seconds`），
        生成侧不重写换算。
        """
        audio = self.audio_norm(batch.audio_emb)  # (B, Ta, D)
        n_frames = int(audio.shape[1])
        frame = self._aligned_frame_index(batch, grid, device)
        previous = torch.clamp(frame - 1, 0, n_frames - 1)
        current = audio[:, frame]  # (B, T, D)
        delta = current - audio[:, previous]
        features = torch.cat([current, delta], dim=-1)
        assert self.audio_align_proj is not None  # 由调用方保证
        projected: Tensor = self.audio_align_proj(features)
        return projected.to(dtype=current.dtype).unsqueeze(1)

    def _field_seconds(self, grid: FieldGrid, *, device: torch.device) -> Tensor:
        """窗口局部 τ 格 -> **窗口内秒数**（τ→秒换算的唯一实现处是 `beatmorph/field/grid.py`）。

        窗口网格的 `bpm_points` 是单段等效 BPM（`bpm_eff = 60/J(τ_start)`），因此这里的
        秒数正好落在与 `encode_audio` 相同的窗口相对时间基上。

        ⚠️ **这里刻意不做缓存**（plan 07 §9-64 的实测缺陷，代价一条 note 都不能再犯）：
        `tau_seconds()` 由 `(t_bins, bpm_points)` 完全决定，而窗口网格的 **`bpm_eff` 逐窗口
        不同**、`t_bins` 却是全库常数（`data.t_window`）。历史实现按 `(t_bins, device)` 缓存，
        于是这张表在**第一次前向时被冻结**成「第一个窗口的 BPM」：此后每个窗口的 τ→秒 都被
        同一个**逐窗口变化**的因子缩放（真实 train split 实测：冻结口径 301.5 BPM vs val 的
        bpm_eff 中位 162 ⇒ 中位 0.54×、100% 的窗口错位，最快的窗口整段 clamp 到末帧），
        `audio_align` 注入的音频因此**不在该 token 的时刻上**（实测对该通路的任何干预都不
        改变损失：置零 Δ = +2e-6 nats/line）；同一张表也冻结了 `seconds_position` 的秒轴。

        直接把缓存删掉是**修根因**而不是修键：每次前向只调用 1-2 次（逐批，不是逐 token），
        `tau_seconds()` 是 192 元素级的 numpy 运算 ⇒ 开销在噪声里；而「时间基」这种东西
        一旦有缓存，键里就必须重新表达一次 `field/` 已经拥有的换算语义（红线 7 的边界）。
        """
        return torch.as_tensor(grid.tau_seconds(), dtype=torch.float32, device=device)

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
        if not self.config.track_input:
            # 只留「哪条线、第几拍」：位置编码与 line embedding 仍在下面加上，
            # 因此模型保有线身份与时间轴，但拿不到运动/透明度/速度的内容（plan 07 §9-67）。
            projected = torch.zeros_like(projected)
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
            seconds=(
                self._field_seconds(batch.grid, device=state.device)
                if self.config.seconds_position
                else None
            ),
        )
        if self.audio_align_proj is not None:
            tokens = tokens + self._aligned_audio(batch, batch.grid, state.device)
        tokens = self.decode(batch, tokens)
        lam, cum, probability = self.head(
            tokens,
            batch.grid,
            line_mask=batch.line_mask_bool(),
            range_mask=batch.range_mask_bool(),
            input_features=features,
        )
        if self.config.mask_lambda and bool(occlusion.any()):
            # lam <- lam * occlusion：把最优解已知为 0 的**未被遮盖格子**钉成 0（见 ModelConfig）。
            # ⚠️ 必须带 `occlusion.any()` 守卫：**全可见**批（G3 门禁、val 的全事件对照）在全事件口径下
            # 监督的是**全部**事件，此时置零会把事件项打成 +inf（下一版实测：门禁退出码 7「lambda 出现 NaN」）。
            # 生成制度（全遮盖）下 `any()` 为真而乘法是恒等，两条边界都对。
            lam = lam * occlusion.to(dtype=lam.dtype)
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
            if self.cell_entropy_weight > 0.0 and output.cell_prob is not None:
                loss = loss + self._cell_entropy_penalty(output, batch)
            output = replace(output, loss=loss)
        return output

    def _cell_entropy_penalty(self, output: FieldOutput, batch: FieldBatch) -> Tensor:
        """`w · (1/r) · Σ_被遮盖 n · H(p) / D`——把空间分布从「又尖又错」推向平坦。

        只统计**被遮盖事件所在的 token**（那正是事件项监督的位置），并按与事件项相同的
        `1/r` 重标定与 `D` 归一，使权重的量纲与 `-log λ` 一致。
        """
        from beatmorph.generation.losses import _counts_or_raise, event_normalizer

        prob = output.cell_prob
        assert prob is not None
        counts = _counts_or_raise(batch).to(dtype=prob.dtype)
        occl = batch.occlusion_bool()
        supervised = counts * occl.to(dtype=counts.dtype)
        n_token = supervised.sum(dim=(3, 4, 5))  # (B, K, T)
        safe = prob.clamp_min(1e-30)
        entropy = -(safe * safe.log()).sum(dim=(3, 4, 5))  # (B, K, T)
        per_line = apply_line_mask_batched((n_token * entropy).sum(dim=2), batch.line_mask_bool())
        ratio = occlusion_ratio(batch)
        if ratio <= 0.0:
            return torch.zeros((), dtype=prob.dtype, device=prob.device)
        return per_line.sum() * (self.cell_entropy_weight / ratio) / event_normalizer(batch)

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
