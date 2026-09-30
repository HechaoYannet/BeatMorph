"""解码与合法性后处理（Plan 05）：强度场 -> 离散事件 -> 合法谱面 + RPEJSON 写路径。

模块划分：

| 文件 | 职责 |
|------|------|
| `fieldops.py` | 场张量的 numpy 适配、`lambda_0 = N/|Omega|` 标度、τ 轴边缘强度 |
| `events.py` | `FieldEvent` / `DecodedEvent` / Hold 配对 / `PhigrosNote` 构造 |
| `peaks.py` | **D1** 峰值检测 + 阈值（B6 基线臂） |
| `thinning.py` | **D2** Ogata thinning（与训练目标同构的原则性解码器） |
| `postprocess/legality.py` | 合法性检查与**留痕**修复（只校验/钳位，不改落点） |
| `pipeline.py` | 把上面串成一次解码，产出 `PhigrosChart` + `LegalityReport` |
| `../io/formats/rpejson/writer.py` | RPEJSON 写路径（导出前门禁：violations 非空即拒绝写出） |

本包**不 import torch**（只有 numpy），因此全部单元测试可在最小环境跑
（契约级测试不得依赖权重或 GPU，CLAUDE.md §4）。
"""

from beatmorph.decoder.events import (
    CHANNEL_NOTE_TYPE,
    DecodedEvent,
    FieldEvent,
    PairingStats,
    confidence_array,
    events_to_notes,
    gameplay_subchart,
    note_is_scorable,
    note_type_for_channel,
    pair_events,
    scorable_lines,
    scorable_note_mask,
    x_center,
)
from beatmorph.decoder.fieldops import (
    assert_field_shape,
    cell_volumes,
    intensity_scale,
    jacobian_cells,
    omega_value,
    tau_rate_per_beat,
    to_numpy,
    total_intensity,
)
from beatmorph.decoder.peaks import (
    DEFAULT_SMOOTH_SECONDS,
    PeakConfig,
    decode_peaks,
    hamming_kernel,
    smooth_cells,
)
from beatmorph.decoder.pipeline import (
    DecodeConfig,
    DecodeMethod,
    DecodeResult,
    chart_from_events,
    count_events,
    decode_both_arms,
    decode_field,
    default_lines,
    events_from_chart,
)
from beatmorph.decoder.postprocess.legality import (
    CROSS_LINE_CRITERION,
    LegalityConfig,
    PostprocessResult,
    check_chart,
    check_events,
    cross_line_conflicts,
    fix_chart,
    fix_events,
    merge_reports,
    postprocess_chart,
    postprocess_events,
    same_instant_groups,
    stage_points,
)
from beatmorph.decoder.thinning import (
    BOUND_REL_TOL,
    ThinningBoundError,
    ThinningConfig,
    ThinningResult,
    block_cells,
    decode_thinning,
    ogata_thinning,
    sample_mark,
)

__all__ = [
    "BOUND_REL_TOL",
    "CHANNEL_NOTE_TYPE",
    "CROSS_LINE_CRITERION",
    "DEFAULT_SMOOTH_SECONDS",
    "DecodeConfig",
    "DecodeMethod",
    "DecodeResult",
    "DecodedEvent",
    "FieldEvent",
    "LegalityConfig",
    "PairingStats",
    "PeakConfig",
    "PostprocessResult",
    "ThinningBoundError",
    "ThinningConfig",
    "ThinningResult",
    "assert_field_shape",
    "block_cells",
    "cell_volumes",
    "chart_from_events",
    "check_chart",
    "check_events",
    "confidence_array",
    "count_events",
    "cross_line_conflicts",
    "decode_both_arms",
    "decode_field",
    "decode_peaks",
    "decode_thinning",
    "default_lines",
    "events_from_chart",
    "events_to_notes",
    "fix_chart",
    "fix_events",
    "gameplay_subchart",
    "hamming_kernel",
    "intensity_scale",
    "jacobian_cells",
    "merge_reports",
    "note_is_scorable",
    "note_type_for_channel",
    "ogata_thinning",
    "omega_value",
    "pair_events",
    "postprocess_chart",
    "postprocess_events",
    "same_instant_groups",
    "sample_mark",
    "scorable_lines",
    "scorable_note_mask",
    "smooth_cells",
    "stage_points",
    "tau_rate_per_beat",
    "to_numpy",
    "total_intensity",
    "x_center",
]
