"""生成主干模块（Plan 04）：掩码补全 Encoder-Decoder + 泊松 NLL + 迭代并行解码。

职责边界（**不得越界**）：

- 本模块**不实现任何秒 <-> tau 换算**：那是 `beatmorph/field/` 的独占职责
  （CLAUDE.md 红线 7）。`FieldBatch.grid` 是测度与 tau 网格的唯一来源。
- 本模块**不实现**场 -> 离散事件的解码与合法性后处理（plan 05）、
  也不实现评估指标与人评协议（plan 06）。

命名速查：`FieldBatch` / `FieldOutput`（契约）、`MaskedFieldModel`（主干 B2）、
`masked_poisson_loss` / `full_poisson_loss`（目标）、`sample`（迭代并行解码）。
"""

from __future__ import annotations

from beatmorph.generation.batch import (
    DEFAULT_EXTENDED_TRACKS,
    FIELD_DIM_NAMES,
    FORBIDDEN_LAYERED_TRACKS,
    N_ORDINARY_TRACKS,
    FieldBatch,
    FieldOutput,
)
from beatmorph.generation.losses import (
    ABLATION_OBJECTIVES,
    POISSON_LOSSES,
    Reduction,
    ReweightMode,
    event_term,
    full_poisson_loss,
    gaussian_heatmap_target,
    hamming_smooth,
    integral_term,
    line_active,
    log_ratio_note,
    masked_poisson_loss,
    occlusion_ratio,
    penalty_reduced_focal_loss,
    per_line_nll,
    poisson_measure_volume,
    range_masked_lambda,
    timestep_weighted_masked_ce,
)
from beatmorph.generation.masks import (
    DEFAULT_LEAK_WINDOW,
    Granularity,
    OcclusionStats,
    assert_hold_pairs_not_split,
    build_occlusion,
    build_occlusion_batch,
    mask_semantics,
    neighbor_leak_rate,
    occluded_event_share,
)
from beatmorph.generation.model import (
    DEFAULT_K_MAX,
    AdaNorm,
    DecoderLayer,
    FieldHead,
    FieldTokenEmbedding,
    HeadMode,
    LayerKind,
    MaskedFieldModel,
    ModelConfig,
    sinusoidal_encoding,
)
from beatmorph.generation.sampling import (
    DEFAULT_STEPS,
    MIN_STEPS,
    Confidence,
    SamplingConfig,
    Schedule,
    StateFill,
    assert_schedule_monotone,
    confidence_map,
    expected_counts,
    local_contrast,
    reveal_schedule,
    sample,
)

__all__ = [
    "ABLATION_OBJECTIVES",
    "DEFAULT_EXTENDED_TRACKS",
    "DEFAULT_K_MAX",
    "DEFAULT_LEAK_WINDOW",
    "DEFAULT_STEPS",
    "FIELD_DIM_NAMES",
    "FORBIDDEN_LAYERED_TRACKS",
    "MIN_STEPS",
    "N_ORDINARY_TRACKS",
    "POISSON_LOSSES",
    "AdaNorm",
    "Confidence",
    "DecoderLayer",
    "FieldBatch",
    "FieldHead",
    "FieldOutput",
    "FieldTokenEmbedding",
    "Granularity",
    "HeadMode",
    "LayerKind",
    "MaskedFieldModel",
    "ModelConfig",
    "OcclusionStats",
    "Reduction",
    "ReweightMode",
    "SamplingConfig",
    "Schedule",
    "StateFill",
    "assert_hold_pairs_not_split",
    "assert_schedule_monotone",
    "build_occlusion",
    "build_occlusion_batch",
    "confidence_map",
    "event_term",
    "expected_counts",
    "full_poisson_loss",
    "gaussian_heatmap_target",
    "hamming_smooth",
    "integral_term",
    "line_active",
    "local_contrast",
    "log_ratio_note",
    "mask_semantics",
    "masked_poisson_loss",
    "neighbor_leak_rate",
    "occluded_event_share",
    "occlusion_ratio",
    "penalty_reduced_focal_loss",
    "per_line_nll",
    "poisson_measure_volume",
    "range_masked_lambda",
    "reveal_schedule",
    "sample",
    "sinusoidal_encoding",
    "timestep_weighted_masked_ce",
]
