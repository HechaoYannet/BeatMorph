"""核心契约包：跨模块共享的数据类型与张量形状约定。

导入入口，集中暴露所有规范类型。其它模块统一
``from beatmorph.core.contracts import Chart, Note``。
"""

from beatmorph.core.contracts.events import (
    Chart,
    GameMode,
    Note,
    NoteType,
    PatternToken,
    Section,
)
from beatmorph.core.contracts.tensors import (
    AR_CONTEXT_TOKENS,
    AudioEmbedding,
    CODEBOOK_BASE,
    CODEBOOK_FINE,
    DEFAULT_LANE_COUNT,
    MERT_DEFAULT_FEAT_DIM,
    MERT_FRAME_RATE_HZ,
    PlanOutput,
    RAGContext,
    TokenSeq,
)

__all__ = [
    # events
    "Note",
    "NoteType",
    "GameMode",
    "Section",
    "Chart",
    "PatternToken",
    # tensors
    "AudioEmbedding",
    "PlanOutput",
    "TokenSeq",
    "RAGContext",
    # 常量
    "MERT_FRAME_RATE_HZ",
    "MERT_DEFAULT_FEAT_DIM",
    "CODEBOOK_BASE",
    "CODEBOOK_FINE",
    "DEFAULT_LANE_COUNT",
    "AR_CONTEXT_TOKENS",
]
