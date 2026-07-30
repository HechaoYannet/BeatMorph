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
    time_seq: int = 0          # 运行期确定
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
    """Stage2 Pattern Token 序列。

    einops: ``(batch, seq_len)`` 的 long 张量，取值范围 [0, codebook_size)。
    每个元素是一小节的离散 Pattern ID（奠基文档 §3.2 / §3.4）。
    """

    seq_len: int = 0
    codebook_size: int = 2048   # 2048 基础 / 4096 精细


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
MERT_FRAME_RATE_HZ: float = 25.0          # 奠基文档 §3.1
MERT_DEFAULT_FEAT_DIM: int = 768
CODEBOOK_BASE: int = 2048                 # §3.2 基础码本
CODEBOOK_FINE: int = 4096                 # §3.2 精细码本
DEFAULT_LANE_COUNT: int = 4               # §1.1 先攻 4K
AR_CONTEXT_TOKENS: int = 256              # §3.4.1 约 256 小节
