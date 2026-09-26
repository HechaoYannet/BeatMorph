"""M3：格式嗅探**只按内容**判定（plan 02 §3.3 / 陷阱 1：后缀不可信）。"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from beatmorph.core.contracts import ChartFormat
from beatmorph.data.parsers.sniff import sniff_format, sniff_format_with_evidence
from tests.fixtures.phigros import read_pec_masquerade, read_rpe_min


def test_signature_has_no_filename_parameter() -> None:
    """签名级约束：嗅探函数不得接受文件名/后缀/路径参数。"""
    parameters = list(inspect.signature(sniff_format).parameters)
    assert parameters == ["data"]
    for banned in ("filename", "name", "path", "suffix", "file", "chart_file", "info_format"):
        assert banned not in parameters


def test_signature_has_no_kwargs_escape_hatch() -> None:
    """不接受 **kwargs（否则可以用 chart_name= 绕过签名约束）。"""
    signature = inspect.signature(sniff_format)
    kinds = {parameter.kind for parameter in signature.parameters.values()}
    assert inspect.Parameter.VAR_KEYWORD not in kinds
    assert inspect.Parameter.VAR_POSITIONAL not in kinds


@pytest.mark.parametrize("content", ["rpe", "pec"])
@pytest.mark.parametrize("suffix", [".json", ".pec"])
def test_content_decides_regardless_of_suffix(tmp_path: Path, content: str, suffix: str) -> None:
    """后缀与内容冲突的四种组合：判定结果只看内容。"""
    payload = read_rpe_min() if content == "rpe" else read_pec_masquerade()
    path = tmp_path / f"chart{suffix}"
    path.write_bytes(payload)
    expected = ChartFormat.RPE if content == "rpe" else ChartFormat.PEC
    assert sniff_format(path.read_bytes()) is expected


def test_pec_content_in_json_file_is_pec(pec_masquerade_bytes: bytes) -> None:
    """M3 正样本：内容为 PEC 文本、后缀为 `.json`。"""
    assert sniff_format(pec_masquerade_bytes) is ChartFormat.PEC


def test_rpe_content_with_pec_suffix_is_rpe(rpe_min_bytes: bytes) -> None:
    """M3 反向用例：把 RPE 内容存成 `.pec` 后缀也必须嗅探为 RPE。"""
    assert sniff_format(rpe_min_bytes) is ChartFormat.RPE


def test_official_json_detected_by_content() -> None:
    """官谱 JSON 特征字段（notesAbove / notesBelow / formatVersion）。"""
    payload = json.dumps(
        {"formatVersion": 3, "judgeLineList": [{"notesAbove": [], "notesBelow": []}]},
    ).encode()
    assert sniff_format(payload) is ChartFormat.OFFICIAL


def test_truncated_prefix_still_classifies_rpe(rpe_min_bytes: bytes) -> None:
    """预筛只给 24 KB 压缩前缀：关键字段在前缀内即可判型。"""
    marker = b'"eventLayers"'
    index = rpe_min_bytes.index(marker)
    assert sniff_format(rpe_min_bytes[: index + len(marker)]) is ChartFormat.RPE


def test_pec_with_truncated_last_line_still_matches(pec_masquerade_bytes: bytes) -> None:
    """截断的末行必须被丢弃，否则前缀嗅探会把 PEC 判成 UNKNOWN。"""
    last_newline = pec_masquerade_bytes.rindex(b"\n", 0, len(pec_masquerade_bytes) - 1)
    truncated = pec_masquerade_bytes[: last_newline + 5]
    assert not truncated.endswith(b"\n")
    assert sniff_format(truncated) is ChartFormat.PEC


def test_rpe_root_key_combination_is_detected_without_event_layers() -> None:
    """兜底规则：前缀里没有 eventLayers（所有层级为空时该字段不出现）仍要判为 RPE。"""
    payload = json.dumps(
        {"BPMList": [{"bpm": 120.0, "startTime": [0, 0, 1]}], "judgeLineList": [{"notes": []}]},
    ).encode()
    assert b'"eventLayers"' not in payload
    fmt, evidence = sniff_format_with_evidence(payload)
    assert fmt is ChartFormat.RPE
    assert "BPMList" in evidence


def test_official_root_keys_do_not_look_like_rpe() -> None:
    """官谱根键是 formatVersion（无 BPMList）→ 不得被 RPE 兜底规则吃掉。"""
    payload = json.dumps({"formatVersion": 3, "judgeLineList": []}).encode()
    assert sniff_format(payload) is ChartFormat.OFFICIAL


def test_json_object_without_markers_is_unknown() -> None:
    assert sniff_format(b'{"hello": 1}') is ChartFormat.UNKNOWN


def test_empty_payload_is_unknown() -> None:
    assert sniff_format(b"") is ChartFormat.UNKNOWN


def test_pbc_is_not_guessed() -> None:
    """PBC 结构完全未查证（调研 §9-Q9）→ 不得凭猜测产出 PBC 判定，一律归 UNKNOWN 并记账。"""
    payload = bytes(range(256)) * 2
    fmt, evidence = sniff_format_with_evidence(payload)
    assert fmt is ChartFormat.UNKNOWN
    assert "PBC" in evidence


def test_unknown_payload_binary_is_unknown() -> None:
    fmt, evidence = sniff_format_with_evidence(b"\x00\x01\x02not-a-chart")
    assert fmt is ChartFormat.UNKNOWN
    assert evidence


def test_evidence_is_always_non_empty(rpe_min_bytes: bytes, pec_masquerade_bytes: bytes) -> None:
    """每次判定都留下可审计依据（写入 ChartSource.sniff_evidence）。"""
    for payload in (rpe_min_bytes, pec_masquerade_bytes, b"", b"{}"):
        _fmt, evidence = sniff_format_with_evidence(payload)
        assert evidence.strip()


def test_rpe_marker_wins_over_official_marker() -> None:
    """同时含两类标记时以 RPE 优先（判定顺序固定，不随 JSON 键序漂移）。"""
    payload = b'{"eventLayers": [], "notesAbove": []}'
    assert sniff_format(payload) is ChartFormat.RPE
