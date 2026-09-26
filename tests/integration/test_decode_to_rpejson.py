"""M5.5 / M5.6：写路径往返 + 越界只统计的**端到端**证据（Plan 05 §6 / §8）。

这是 plan 05 与 plan 02 的接缝测试：`write` 在 `beatmorph/io/formats/rpejson/`，
`read` 在 `beatmorph/data/parsers/rpejson.py`，两者共用契约常量与映射函数。
"""

from __future__ import annotations

import json

import pytest

from beatmorph.core.contracts import (
    RPE_STAGE_HALF_WIDTH,
    SUBDIVISIONS_PER_BEAT,
    ChartSource,
    LegalityReport,
    NoteType,
)
from beatmorph.data.parsers.rpejson import parse_rpejson
from beatmorph.decoder import DecodeConfig, check_chart, decode_field, postprocess_chart
from beatmorph.decoder.thinning import ThinningConfig
from beatmorph.field.grid import FieldGrid
from beatmorph.io.formats.rpejson import (
    IllegalChartError,
    dump_rpejson,
    rpejson_text,
    write_rpejson,
)
from tests.fixtures.phigros import read_rpe_min
from tests.unit.decoder._builders import (
    empty_field,
    make_bpm_points,
    make_grid,
    make_template,
    place_gaussian,
    spec_for,
)
from tests.unit.field._builders import make_bpm_points as field_bpm
from tests.unit.field._builders import make_chart, make_note

pytestmark = pytest.mark.integration


def _fixture_chart():
    return parse_rpejson(read_rpe_min(), ChartSource(sniff_evidence="fixture"))


def _quantization_step_seconds(chart) -> float:
    """一个 beat 量化步长在**秒域**的上界：取全谱最大的格子时长 `J * d_tau`。"""
    grid = FieldGrid().for_chart(chart)
    return float(max(grid.cell_seconds()))


def _assert_markers_match(expected, actual, step: float) -> None:
    """标记全等 + 时间 <= 一个量化步长（M5.5 的判据）。

    用**贪心匹配**而不是排序后逐位比较：时间量化会改变"同一时刻邻近 note"的排序，
    逐位比较会把排序差异误报成标记差异（标记本身没变）。
    """
    pool = list(actual)
    for note in expected:
        for index, candidate in enumerate(pool):
            if (
                candidate.line_id == note.line_id
                and candidate.side is note.side
                and candidate.type is note.type
                and candidate.position_x == note.position_x
                and candidate.is_fake == note.is_fake
                and abs(candidate.t - note.t) <= step
            ):
                pool.pop(index)
                break
        else:
            raise AssertionError(f"往返后找不到匹配的 note：{note}")
    assert not pool, f"往返后多出 {len(pool)} 个 note"


def test_fixture_chart_is_legal_and_roundtrips_markers() -> None:
    """M5.4 + M5.5：夹具零违规；`read(write(chart))` 标记全等、时间 <= 一个量化步长。"""
    chart = _fixture_chart()
    report = check_chart(chart)
    assert report.violations == [], report.format()
    data = write_rpejson(chart, report=report)
    again = parse_rpejson(data, ChartSource())
    # 夹具共 6 个 note（line0 四个 + line1 + line2）+ 一条空线（K=4）
    assert len(again.notes) == len(chart.notes) == 6
    step = _quantization_step_seconds(chart)
    for original, copied in zip(chart.sorted_notes(), again.sorted_notes(), strict=True):
        assert original.line_id == copied.line_id
        assert original.position_x == copied.position_x
        assert original.side is copied.side
        assert original.type is copied.type
        assert original.is_fake == copied.is_fake
        assert abs(original.t - copied.t) <= step
        assert abs(original.hold_time - copied.hold_time) <= step
    assert again.bpm_points == chart.bpm_points
    assert again.lines[0].event_layers == chart.lines[0].event_layers
    assert again.meta.rpe_version == chart.meta.rpe_version
    assert again.meta.chart_time_s == chart.meta.chart_time_s


def test_write_read_write_is_byte_identical() -> None:
    """M5.5：写入 -> 读回 -> 再写入必须**字节级稳定**（幂等）。"""
    chart = _fixture_chart()
    first = write_rpejson(chart, report=check_chart(chart))
    again = parse_rpejson(first, ChartSource())
    second = write_rpejson(again, report=check_chart(again))
    assert first == second
    assert rpejson_text(again).encode("utf-8") == first


def test_fixture_roundtrip_keeps_the_hard_coded_traps_intact() -> None:
    """格式陷阱不得在写路径上被"顺手修正"：`above=2` 原样保留、PEC 后缀夹具仍是 PEC。"""
    chart = _fixture_chart()
    payload = json.loads(rpejson_text(chart))
    above_values = [note["above"] for line in payload["judgeLineList"] for note in line["notes"]]
    assert 2 in above_values, "above 的三值原样写回（不得在 writer 里归一化成 0/1）"
    types = {note["type"] for line in payload["judgeLineList"] for note in line["notes"]}
    assert types == {1, 2, 3, 4}


def test_out_of_range_notes_survive_the_export_and_the_stats_agree() -> None:
    """M5.6：越界 note **仍在输出文件里**，且 `stats.out_of_range` 与输入一致。"""
    x = RPE_STAGE_HALF_WIDTH * 1.1
    chart = make_chart(
        notes=[
            make_note(t=0.0, line_id=0, position_x=x),
            make_note(t=0.5, line_id=0, position_x=-x, above=0),
            make_note(t=1.0, line_id=0, position_x=0.0),
        ],
        bpm_points=field_bpm((0.0, 120.0)),
        k=1,
        chart_time_s=2.0,
    )
    result = postprocess_chart(chart)
    assert result.report.violations == []
    assert result.report.stat("chart_out_of_range") == 2.0
    exported = write_rpejson(result.chart, report=result.report)
    back = parse_rpejson(exported, ChartSource())
    xs = sorted(note.position_x for note in back.notes)
    assert xs == [-x, 0.0, x], "越界 note 必须原样存在（防回归到『偷偷钳位』）"
    assert back.out_of_visible_range().count == 2


def test_full_decode_to_file_pipeline_for_both_arms(tmp_path) -> None:
    """`field -> decode -> postprocess -> write` 全链路（两臂），含落盘与读回。"""
    bpm = make_bpm_points((0.0, 150.0))
    grid = make_grid(bpm_points=bpm, t_bins=SUBDIVISIONS_PER_BEAT * 8)
    spec = spec_for(grid, 1)
    field = empty_field(1, spec)
    placements = ((8, 20), (40, 100), (120, 64), (200, 40))
    for tau_bin, x_bin in placements:
        place_gaussian(field, line_id=0, tau_bin=tau_bin, x_bin=x_bin, amplitude=50.0)
    template = make_template(k=1, bpm_points=bpm, chart_time_s=4.0)

    for config in (
        DecodeConfig(method="peaks"),
        DecodeConfig(method="thinning", thinning=ThinningConfig(seed=3)),
    ):
        result = decode_field(field, grid, template=template, config=config)
        assert result.report.violations == [], result.report.format()
        assert result.chart.notes, f"{config.method} 臂没有解出任何音符"
        path = dump_rpejson(
            result.chart,
            tmp_path / f"{config.method}.json",
            report=result.report,
        )
        back = parse_rpejson(path.read_bytes(), ChartSource())
        step = _quantization_step_seconds(back)
        _assert_markers_match(result.chart.notes, back.notes, step)


def test_export_gate_blocks_an_illegal_chart_end_to_end(tmp_path) -> None:
    """红线 6：`violations` 非空时**拒绝写出**（CLI 退出码非 0 属 plan 08）。"""
    chart = make_chart(
        notes=[make_note(t=0.0, line_id=0), make_note(t=0.0, line_id=0)],
        bpm_points=field_bpm((0.0, 120.0)),
        k=1,
    )
    findings = check_chart(chart)
    assert not findings.is_legal
    with pytest.raises(IllegalChartError):
        dump_rpejson(chart, tmp_path / "illegal.json", report=findings)
    assert not (tmp_path / "illegal.json").exists()
    fixed = postprocess_chart(chart)
    assert isinstance(fixed.report, LegalityReport)
    assert fixed.report.is_legal
    dump_rpejson(fixed.chart, tmp_path / "illegal.json", report=fixed.report)
    assert (tmp_path / "illegal.json").exists()
    assert len(parse_rpejson((tmp_path / "illegal.json").read_bytes(), ChartSource()).notes) == 1
    assert fixed.report.edits[0].resolved is not None


def test_decoded_note_types_and_sides_match_the_contract_enums() -> None:
    """解码产出的 `type`/`above` 必须与契约枚举一致（RPE 表，不是官谱表）。"""
    bpm = make_bpm_points((0.0, 150.0))
    grid = make_grid(bpm_points=bpm, t_bins=SUBDIVISIONS_PER_BEAT * 4)
    spec = spec_for(grid, 1)
    field = empty_field(1, spec)
    from beatmorph.core.contracts import Side
    from beatmorph.field.target import HOLD_END_CHANNEL

    place_gaussian(field, line_id=0, tau_bin=10, x_bin=30, note_type=NoteType.HOLD)
    place_gaussian(field, line_id=0, tau_bin=40, x_bin=30, channel=HOLD_END_CHANNEL)
    place_gaussian(field, line_id=0, tau_bin=80, x_bin=90, note_type=NoteType.DRAG, side=Side.BACK)
    result = decode_field(field, grid, template=make_template(k=1, bpm_points=bpm))
    payload = json.loads(rpejson_text(result.chart))
    written = payload["judgeLineList"][0]["notes"]
    assert {note["type"] for note in written} == {int(NoteType.HOLD), int(NoteType.DRAG)}
    assert {note["above"] for note in written} == {1, 0}
    hold = next(note for note in written if note["type"] == int(NoteType.HOLD))
    assert hold["endTime"] != hold["startTime"]
