"""M2：微缩夹具的体积 / 合规 / README 契约（plan 02 §3.7）。"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from tests.fixtures.phigros import (
    FIXTURES_DIR,
    INFO_YML_NAME,
    PEC_MASQUERADE_NAME,
    PKG_CHART_NAME,
    PKG_DECOY_NAME,
    PKG_DISTRACTOR_NAME,
    PKG_ENTRY_NAMES,
    PKG_MUSIC_NAME,
    RPE_MIN_NAME,
    build_pkg_bytes,
    build_pkg_zip,
)

#: 单文件上限（plan §3.7）。
MAX_FILE_BYTES = 32 * 1024
#: 夹具目录上限（plan §3.7）。
MAX_DIR_BYTES = 128 * 1024

AUDIO_SUFFIXES = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".mid")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".svg")


def _fixture_files() -> list[Path]:
    """夹具目录下的全部文件（跳过 `__pycache__` 与 zip 产物）。"""
    return sorted(
        path
        for path in FIXTURES_DIR.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )


def test_fixture_files_within_size_limit() -> None:
    oversized = [
        (path.name, path.stat().st_size)
        for path in _fixture_files()
        if path.stat().st_size > MAX_FILE_BYTES
    ]
    assert oversized == [], f"单个夹具文件不得超过 {MAX_FILE_BYTES} 字节：{oversized}"


def test_fixture_directory_within_size_limit() -> None:
    total = sum(path.stat().st_size for path in _fixture_files())
    assert total <= MAX_DIR_BYTES, f"夹具目录合计 {total} 字节，超过 {MAX_DIR_BYTES}"


def test_fixtures_contain_no_audio_or_illustration() -> None:
    """合规硬约束：夹具不含音频与曲绘（plan §3.7）。"""
    offenders = [
        path.name
        for path in _fixture_files()
        if path.suffix.lower() in (*AUDIO_SUFFIXES, *IMAGE_SUFFIXES)
    ]
    assert offenders == [], f"夹具不得包含音频/曲绘：{offenders}"


def test_fixture_files_use_lf() -> None:
    """全仓 LF：夹具不得带 CRLF。"""
    offenders = [path.name for path in _fixture_files() if b"\r\n" in path.read_bytes()]
    assert offenders == [], f"夹具必须使用 LF 换行：{offenders}"


def test_readme_records_source_and_hashes() -> None:
    readme = (FIXTURES_DIR / "README.md").read_text(encoding="utf-8")
    for keyword in ("手工构造", "非真实谱面", "不含音频", "sha256", "构造方式"):
        assert keyword in readme, f"夹具 README 必须记录 {keyword}"


def test_rpe_min_is_json_and_covers_all_note_types() -> None:
    payload = json.loads((FIXTURES_DIR / RPE_MIN_NAME).read_text(encoding="utf-8"))
    assert len(payload["judgeLineList"]) >= 3
    types = {item["type"] for line in payload["judgeLineList"] for item in line.get("notes", [])}
    assert types == {1, 2, 3, 4}, "夹具必须覆盖四类 note type"
    aboves = {item["above"] for line in payload["judgeLineList"] for item in line.get("notes", [])}
    assert 2 in aboves, "夹具必须含 above == 2 的背面 note"
    assert 1 in aboves, "夹具必须含正面 note"
    layers = [
        line.get("eventLayers") for line in payload["judgeLineList"] if line.get("eventLayers")
    ]
    assert any(len(item) >= 2 for item in layers), "夹具必须含多层事件"


def test_pec_masquerade_is_text_not_json() -> None:
    raw = (FIXTURES_DIR / PEC_MASQUERADE_NAME).read_bytes()
    text = raw.decode("utf-8")
    assert text.splitlines()[1].startswith("bp ")
    with pytest.raises(json.JSONDecodeError):
        json.loads(text)


def test_pkg_min_info_points_to_non_default_name() -> None:
    info = (FIXTURES_DIR / "pkg_min" / INFO_YML_NAME).read_text(encoding="utf-8")
    assert f"chart: {PKG_CHART_NAME}" in info
    assert PKG_CHART_NAME != "chart.json", "夹具谱面文件不得叫默认名 chart.json"


def test_pkg_zip_is_deterministic() -> None:
    assert build_pkg_bytes() == build_pkg_bytes()


def test_pkg_zip_entries_and_distractor_bigger_than_chart(tmp_path: Path) -> None:
    path = build_pkg_zip(tmp_path / "pkg.zip")
    with zipfile.ZipFile(path) as archive:
        assert archive.namelist() == list(PKG_ENTRY_NAMES)
        sizes = {item.filename: item.file_size for item in archive.infolist()}
    assert PKG_DECOY_NAME in sizes
    assert sizes[PKG_DISTRACTOR_NAME] > sizes[PKG_CHART_NAME], "干扰 json 必须比谱面文件大"


def test_pkg_zip_declares_missing_audio(tmp_path: Path) -> None:
    """夹具声明了 music，但按合规要求**不入包**（构造时只写文本部件）。"""
    with zipfile.ZipFile(build_pkg_zip(tmp_path / "pkg.zip")) as archive:
        assert PKG_MUSIC_NAME not in archive.namelist()
