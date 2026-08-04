"""张量级数据契约 — 模块间的「中间表征」张量形状约定。

为避免各模块对张量维度理解不一致，这里集中定义所有跨模块张量的
契约形状与语义。所有形状以 einops 风格字符串描述。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AudioEmbedding:
    """Stage0 音频编码器输出。

    einops: ``(batch, time_seq, feat)``

    Attributes:
        feat: 特征维，MERT 默认 768（第12层）或 1024（第24层）。
        time_seq: 帧序列长度，帧率 25Hz（秒数 * 25）。
        hop_rate: 帧率 Hz（默认 25.0，奠基文档 §3.1）。
    """

    feat: int = 768
    time_seq: int = 0  # 运行期确定
    hop_rate: float = 25.0


@dataclass(frozen=True)
class PlanOutput:
    """Stage1 全局规划层输出。

    - density_target: ``(batch, num_sections)`` 0-1
    - energy_level: ``(batch, num_sections)`` 0-1
    - rest_probability: ``(batch, num_sections)`` 0-1
    基金会文档 §3.3：每 4 小节一个 Section。
    """

    num_sections: int = 0
    section_bars: int = 4


@dataclass(frozen=True)
class TokenSeq:
    """Stage2 Pattern Token 序列（**legacy / baseline 分支**，RFC-0028）。

    einops: ``(batch, seq_len)`` 的 long 张量，取值范围 ``[0, codebook_size)``。
    每个元素是一小节的离散 Pattern ID。原 VQ-VAE 范式产物；主路径改用
    :class:`EventSeq`，本类型仅由 ``archive/vqvae-baseline`` 分支沿用。
    """

    seq_len: int = 0
    codebook_size: int = 2048  # 2048 基础 / 4096 精细


@dataclass(frozen=True)
class EventSeq:
    """Stage2 event token 序列（RFC-0028，主路径）。

    einops: ``(batch, seq_len)`` 的 long 张量，取值范围 ``[0, vocab_size)``。
    每个元素是一个 BPE event id（原子或复合）。``seq_len`` 单位是 event 而非小节
    （~4700 原子 event/曲 → BPE 合并后 ~2000-2500）。

    Attributes:
        seq_len: 序列长度（event 数）。
        vocab_size: BPE 词表大小（默认 :data:`BPE_DEFAULT_VOCAB`=4096）。
    """

    seq_len: int = 0
    vocab_size: int = 4096


@dataclass(frozen=True)
class RAGContext:
    """RAG 检索上下文（奠基文档 §3.5）。

    - token_prefix: ``(batch, top_k, ref_seq_len)`` 检索到的参考谱面 Token
    - style_emb: ``(batch, top_k, feat)`` 检索谱面风格向量
    """

    top_k: int = 3
    ref_seq_len: int = 256
    feat: int = 768


# ── 模块间约定的关键常量 ──
MERT_FRAME_RATE_HZ: float = 25.0  # 奠基文档 §3.1
MERT_DEFAULT_FEAT_DIM: int = 1024  # MERT-v1-330M hidden_size
# ── 以下 VQ 码本常量为 RFC-0028 legacy：仅 archive/vqvae-baseline 分支沿用 ──
CODEBOOK_BASE: int = 2048  # §3.2 基础码本（VQ-VAE, 已退役 baseline）
CODEBOOK_FINE: int = 4096  # §3.2 精细码本（VQ-VAE, 已退役 baseline）
DEFAULT_LANE_COUNT: int = 4  # §1.1 先攻 4K
# ── RFC-0028 BPE/event tokenizer 常量（主路径）──
BPE_DEFAULT_VOCAB: int = 4096  # §3.2.1 BPE 词表默认大小
BPE_VOCAB_POC_SWEEP: tuple[int, ...] = (2048, 4096, 8192)  # PoC 词表扫参
POS_DIVISIONS_PER_BEAT: int = 48  # §3.2.2 Position 子拍网格 1/48 拍
NUDGE_BUCKETS: int = 12  # §3.2.2 残差毫秒桶数（PoC 不达标可升 16）
AR_CONTEXT_TOKENS: int = 1024  # §3.4.1 ~1024 event 分段（原 256 小节）
