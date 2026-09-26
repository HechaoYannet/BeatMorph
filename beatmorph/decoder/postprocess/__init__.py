"""后处理子包（Plan 05 §4.3）：`legality.py` 是唯一实现处。"""

from beatmorph.decoder.postprocess.legality import (
    CROSS_LINE_CRITERION,
    SPEED_REL_TOL,
    LegalityConfig,
    PostprocessResult,
    check_chart,
    check_events,
    cross_line_conflicts,
    event_key,
    fix_chart,
    fix_events,
    merge_reports,
    note_key,
    postprocess_chart,
    postprocess_events,
    same_instant_groups,
    stage_points,
)

__all__ = [
    "CROSS_LINE_CRITERION",
    "SPEED_REL_TOL",
    "LegalityConfig",
    "PostprocessResult",
    "check_chart",
    "check_events",
    "cross_line_conflicts",
    "event_key",
    "fix_chart",
    "fix_events",
    "merge_reports",
    "note_key",
    "postprocess_chart",
    "postprocess_events",
    "same_instant_groups",
    "stage_points",
]
