"""RPEJSON 写路径的单元测试（Plan 05 §4.4 / M5.5 的写侧）。

覆盖：秒 -> beat 三元组的**量化上界**、`type`/`above` 的契约映射、
`speedEvents` 无贝塞尔、扩展轨、导出门禁（红线 6）、以及"结构性不可写"的守卫。
读回与往返在 `tests/integration/test_decode_to_rpejson.py`。
"""

from __future__ import annotations

import json

import pytest

from beatmorph.core.contracts.legality import LegalityReport, Violation, ViolationKind
from beatmorph.core.contracts.phigros import (
    SUBDIVISIONS_PER_BEAT,
    TAU_GRID_DT,
    Beat,
    EventKeyframe,
    EventLayer,
    JudgeLine,
    NoteType,
    PhigrosChart,
    Side,
    above_from_side,
    side_from_above,
)
from beatmorph.io.formats.rpejson import (
    IllegalChartError,
    beat_from_tau,
    chart_to_rpe_root,
    dump_rpejson,
    rpejson_text,
    write_rpejson,
)
from tests.unit.field._builders import make_bpm_points, make_chart, make_note

BPM_POINTS = make_bpm_points((0.0, 120.0), (8.0, 180.0))

LEGAL_REPORT = LegalityReport()
ILLEGAL_REPORT = LegalityReport(
    violations=[Violation(kind=ViolationKind.DUPLICATE_EVENT, detail="测试用")],
)


def _chart(**kwargs: object) -> PhigrosChart:
    default: dict[str, object] = {
        "notes": [make_note(t=0.0, line_id=0, position_x=100.0)],
        "bpm_points": BPM_POINTS,
        "k": 1,
        "chart_time_s": 4.0,
    }
    default.update(kwargs)
    return make_chart(**default)


def test_beat_from_tau_quantization_bound_and_integers() -> None:
    """量化误差 <= 半格；分子恒在 [0, d)；负数用 floor 除法（不产生负分子）。"""
    half = TAU_GRID_DT / 2.0
    probes = [0.0, TAU_GRID_DT, 1.0 / 3.0, 7.25, 31.75, 1234.567, -1.5 * TAU_GRID_DT, -3.0]
    for tau in probes:
        beat = beat_from_tau(tau)
        assert 0 <= beat.n < beat.d == SUBDIVISIONS_PER_BEAT
        assert abs(beat.to_beats() - tau) <= half
    on_grid = 17 * TAU_GRID_DT
    # 格点上的 τ 用 [0, 17, 48] 精确表示（float 层可能有 1 ulp 的表示误差）
    assert beat_from_tau(on_grid) == Beat(i=0, n=17, d=SUBDIVISIONS_PER_BEAT)
    assert abs(beat_from_tau(on_grid).to_beats() - on_grid) < 1e-15
    with pytest.raises(ValueError, match="denominator"):
        beat_from_tau(0.0, denominator=0)


def test_type_numbers_follow_the_rpe_table_not_the_official_one() -> None:
    """RPE 的 2 是 Hold、3 是 Flick、4 是 Drag（官谱表**不同**，不得混用）。"""
    notes = [
        make_note(t=index * 0.25, line_id=0, note_type=note_type)
        for index, note_type in enumerate(
            (NoteType.TAP, NoteType.HOLD, NoteType.FLICK, NoteType.DRAG)
        )
    ]
    root = chart_to_rpe_root(_chart(notes=notes))
    written = [note["type"] for note in root["judgeLineList"][0]["notes"]]
    assert written == [1, 2, 3, 4]
    assert written[int(NoteType.HOLD) - 1] == 2


def test_above_raw_is_written_verbatim_and_canonical_form_comes_from_construction() -> None:
    """`above` 的三值语义分两处落地（都不许在 writer 里重写映射）：

    - **解析来的**谱面：`above_raw` 原样写回（0/1/2 都保留，无损往返）；
    - **解码来的**谱面：构造 note 时经契约 `above_from_side` 取规范代表值（FRONT -> 1，BACK -> 0）。
    """
    notes = [
        make_note(t=0.0, line_id=0, above=1),
        make_note(t=0.25, line_id=0, above=0),
        make_note(t=0.5, line_id=0, above=2),
    ]
    root = chart_to_rpe_root(_chart(notes=notes))
    written = [note["above"] for note in root["judgeLineList"][0]["notes"]]
    assert written == [1, 0, 2]
    assert [note.side for note in notes] == [Side.FRONT, Side.BACK, Side.BACK]
    for side in Side:
        assert side_from_above(above_from_side(side)) is side


def test_writer_requires_a_report_and_refuses_violations() -> None:
    """红线 6：导出前门禁。默认路径不允许"忘了校验"。"""
    chart = _chart()
    with pytest.raises(ValueError, match="LegalityReport"):
        write_rpejson(chart)
    with pytest.raises(IllegalChartError):
        write_rpejson(chart, report=ILLEGAL_REPORT)
    data = write_rpejson(chart, report=LEGAL_REPORT)
    assert isinstance(data, bytes)
    assert data.startswith(b"{")
    bypass = write_rpejson(chart, require_legal=False)
    assert bypass == data


def test_speed_events_have_no_bezier_fields() -> None:
    """格式事实：`speedEvents` 无贝塞尔字段；其余普通轨写出贝塞尔三件套。"""
    line = JudgeLine(
        line_id=0,
        event_layers=[
            EventLayer(
                speed=[EventKeyframe(start_time=Beat(i=0), end_time=Beat(i=2), start=1.0, end=2.0)],
                move_x=[
                    EventKeyframe(start_time=Beat(i=0), end_time=Beat(i=2), start=0.0, end=10.0),
                ],
            ),
        ],
    )
    root = chart_to_rpe_root(
        PhigrosChart(lines=[line], notes=[], bpm_points=BPM_POINTS),
    )
    layer = root["judgeLineList"][0]["eventLayers"][0]
    assert "bezier" not in layer["speedEvents"][0]
    assert "bezierPoints" not in layer["speedEvents"][0]
    assert layer["moveXEvents"][0]["bezier"] == 0
    assert layer["moveXEvents"][0]["bezierPoints"] == [0.0, 0.0, 0.0, 0.0]
    assert layer["speedEvents"][0]["easingType"] == 1


def test_extended_tracks_and_line_fields_are_written() -> None:
    """`extended`（第 5 层）与判定线字段都按 RPE 原名写出；空轨不写。"""
    line = JudgeLine(
        line_id=0,
        name="main",
        group=2,
        father=-1,
        is_cover=2,
        z_order=-1,
        attach_ui="combo",
        bpm_factor=2.0,
    )
    payload = chart_to_rpe_root(
        PhigrosChart(lines=[line], notes=[], bpm_points=BPM_POINTS),
    )["judgeLineList"][0]
    assert payload["Group"] == 2
    assert payload["Name"] == "main"
    assert payload["isCover"] == 2
    assert payload["zOrder"] == -1
    assert payload["attachUI"] == "combo"
    assert payload["bpmfactor"] == 2.0
    assert payload["numOfNotes"] == 0
    assert payload["extended"] == {}
    assert payload["eventLayers"] == [{}]


def test_num_of_notes_is_the_actual_count() -> None:
    """`numOfNotes` 是冗余字段：写出**真实条数**才不会产出自相矛盾的文件。"""
    chart = _chart(
        notes=[
            make_note(t=0.0, line_id=0),
            make_note(t=0.5, line_id=0),
            make_note(t=1.0, line_id=1),
        ],
        k=2,
    )
    root = chart_to_rpe_root(chart)
    assert [line["numOfNotes"] for line in root["judgeLineList"]] == [2, 1]


def test_structural_guards_refuse_unwritable_charts() -> None:
    """结构性不可写：type_raw 不在 RPE 取值域、line_id 越界 —— 都拒绝写出。"""
    broken_type = _chart(notes=[make_note(t=0.0, line_id=0).model_copy(update={"type_raw": 9})])
    with pytest.raises(IllegalChartError, match="type_raw"):
        write_rpejson(broken_type, report=LEGAL_REPORT)
    broken_line = _chart(notes=[make_note(t=0.0, line_id=5)])
    with pytest.raises(IllegalChartError, match="line_id"):
        write_rpejson(broken_line, report=LEGAL_REPORT)


def test_output_is_utf8_lf_and_has_the_root_keys() -> None:
    """字节纪律：LF 结尾、UTF-8、根对象含四个必需的 RPE 字段。"""
    chart = _chart(notes=[make_note(t=0.0, line_id=0)])
    text = rpejson_text(chart)
    assert text.endswith("\n")
    assert "\r" not in text
    root = json.loads(text)
    assert set(root) == {"BPMList", "META", "chartTime", "judgeLineList"}
    # BPM 段起点按 τ 格宽量化（分母 = SUBDIVISIONS_PER_BEAT），整拍时分子为 0
    assert root["BPMList"][1]["startTime"] == [8, 0, SUBDIVISIONS_PER_BEAT]
    assert root["META"]["RPEVersion"] == 0
    assert root["META"]["offset"] == 0
    assert isinstance(root["META"]["offset"], int)
    assert root["chartTime"] == 4.0


def test_dump_writes_bytes_to_disk(tmp_path) -> None:
    target = dump_rpejson(_chart(), tmp_path / "out" / "chart.json", report=LEGAL_REPORT)
    assert target.read_bytes() == write_rpejson(_chart(), report=LEGAL_REPORT)
    with pytest.raises(IllegalChartError):
        dump_rpejson(_chart(), tmp_path / "bad.json", report=ILLEGAL_REPORT)
