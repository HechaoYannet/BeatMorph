"""RPEJSON 读写包（主路径唯一权威格式，CLAUDE.md §2）。

- **读路径**在 `beatmorph/data/parsers/rpejson.py`（plan 02，data-agent 主责）；
- **写路径**在本包的 `writer.py`（plan 05，解码与后处理组主责）。

两者共用 `beatmorph.core.contracts` 的字段常量与映射函数，不各自维护一份 schema。
`read(write(chart))` 的往返契约由 `tests/integration/test_decode_to_rpejson.py` 钉住。
"""

from beatmorph.io.formats.rpejson.writer import (
    IllegalChartError,
    beat_from_tau,
    beat_to_list,
    chart_to_rpe_root,
    dump_rpejson,
    iter_note_groups,
    keyframe_to_rpe,
    line_to_rpe,
    note_to_rpe,
    rpejson_text,
    value_to_json,
    write_rpejson,
)

__all__ = [
    "IllegalChartError",
    "beat_from_tau",
    "beat_to_list",
    "chart_to_rpe_root",
    "dump_rpejson",
    "iter_note_groups",
    "keyframe_to_rpe",
    "line_to_rpe",
    "note_to_rpe",
    "rpejson_text",
    "value_to_json",
    "write_rpejson",
]
