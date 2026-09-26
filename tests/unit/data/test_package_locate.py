r"""M4：谱面包定位 —— **只按 \`info.yml.chart\`**（陷阱 2 / R1 / R2 / R4）。

夹具包内同时埋了三个诱饵：decoy \`chart.json\`（R2）、更大的干扰 json（R1）、
\`info.yml.format: null\`（R3）。本文件断言「只有 info.yml.chart 这一条路径能选中谱面」。
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from beatmorph.data.phira.client import ZipEntry, parse_zip_index
from beatmorph.data.phira.package import (
    INFO_YML_NAME,
    ChartPackage,
    ChartPackageError,
    chart_dest_path,
    normalize_chart_filename,
    parse_info_yaml,
)
from tests.fixtures.phigros import (
    PKG_CHART_NAME,
    PKG_DECOY_NAME,
    PKG_DISTRACTOR_NAME,
    PKG_MUSIC_NAME,
    build_pkg_bytes,
    build_pkg_zip,
    pkg_info_text,
    read_bytes,
)


def _read_entry(pkg_bytes: bytes, name: str) -> bytes:
    with zipfile.ZipFile(io.BytesIO(pkg_bytes)) as archive:
        return archive.read(name)


def test_package_locates_entry_from_info_yml(pkg_zip_path: Path) -> None:
    package = ChartPackage.open(pkg_zip_path)
    assert package.chart_file == PKG_CHART_NAME
    assert package.chart_file != PKG_DECOY_NAME
    assert package.chart_bytes() == _read_entry(build_pkg_bytes(), PKG_CHART_NAME)


def test_decoy_and_distractor_are_not_chosen(pkg_zip_path: Path) -> None:
    """R1/R2 反面教材：包内最大的 json 与默认名 chart.json 都不是谱面文件。"""
    package = ChartPackage.open(pkg_zip_path)
    assert package.largest_json_entry() == PKG_DISTRACTOR_NAME
    assert package.largest_json_entry() != package.chart_file
    assert PKG_DECOY_NAME in package.entries
    assert (
        package.entries[PKG_DISTRACTOR_NAME].file_size > package.entries[PKG_CHART_NAME].file_size
    )
    assert package.entries[PKG_DECOY_NAME].file_size < package.entries[PKG_CHART_NAME].file_size


def test_info_yml_is_parsed_with_camel_case_aliases(pkg_zip_path: Path) -> None:
    package = ChartPackage.open(pkg_zip_path)
    info = package.info
    assert info.chart == PKG_CHART_NAME
    assert info.music == PKG_MUSIC_NAME
    assert info.format is None, "实测 info.yml.format 恒为 null（R3：不得用于判别格式）"
    assert info.aspect_ratio is not None
    assert info.line_length is not None
    assert info.difficulty_round == round(info.difficulty or 0.0, 1)
    assert info.tags


def test_missing_chart_entry_is_an_error(tmp_path: Path) -> None:
    """把 info.yml.chart 指向不存在条目 → 必须报错（**不得**回退猜测）。"""
    broken = pkg_info_text().replace(f"chart: {PKG_CHART_NAME}", "chart: missing_entry.json")
    path = build_pkg_zip(tmp_path / "broken.zip", info_text=broken)
    with pytest.raises(ChartPackageError, match="不存在"):
        ChartPackage.open(path)


def test_error_message_names_the_forbidden_fallbacks(tmp_path: Path) -> None:
    broken = pkg_info_text().replace(f"chart: {PKG_CHART_NAME}", "chart: nope.json")
    path = build_pkg_zip(tmp_path / "broken.zip", info_text=broken)
    with pytest.raises(ChartPackageError) as excinfo:
        ChartPackage.open(path)
    message = str(excinfo.value)
    assert "chart.json" in message
    assert "max" in message.lower() or "最大" in message


def test_empty_chart_field_is_an_error(tmp_path: Path) -> None:
    broken = pkg_info_text().replace(f"chart: {PKG_CHART_NAME}", 'chart: ""')
    path = build_pkg_zip(tmp_path / "broken.zip", info_text=broken)
    with pytest.raises(ChartPackageError, match="chart"):
        ChartPackage.open(path)


def test_missing_info_yml_is_an_error(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("chart.json", b"{}")
    path = tmp_path / "no_info.zip"
    path.write_bytes(buffer.getvalue())
    with pytest.raises(ChartPackageError, match=INFO_YML_NAME):
        ChartPackage.open(path)


def test_music_declared_but_absent_is_only_a_warning(pkg_zip_path: Path) -> None:
    r"""合规夹具不含音频：\`music\` 缺失只记账，不影响谱面定位。"""
    package = ChartPackage.open(pkg_zip_path)
    assert package.missing_music
    with pytest.raises(ChartPackageError, match="音频"):
        package.music_bytes()


def test_from_zip_bytes_matches_open(pkg_zip_path: Path) -> None:
    from_file = ChartPackage.open(pkg_zip_path)
    from_bytes = ChartPackage.from_zip_bytes(build_pkg_bytes())
    assert from_bytes.chart_file == from_file.chart_file
    assert from_bytes.chart_bytes() == from_file.chart_bytes()


def test_from_index_uses_injected_reader() -> None:
    """中央目录预筛路径：条目表来自 Range 预筛，字节由 reader 提供。"""
    pkg_bytes = build_pkg_bytes()
    index = parse_zip_index(
        pkg_bytes, start_offset=0, total_bytes=len(pkg_bytes), source_url="mem://"
    )
    package = ChartPackage.from_index(index, lambda name: _read_entry(pkg_bytes, name))
    assert package.chart_file == PKG_CHART_NAME
    assert package.entries[PKG_CHART_NAME].file_size == len(package.chart_bytes())
    assert package.source == "mem://"


def test_total_uncompressed_bytes_by_suffix(pkg_zip_path: Path) -> None:
    package = ChartPackage.open(pkg_zip_path)
    assert package.total_uncompressed_bytes(".json") > package.entries[PKG_CHART_NAME].file_size
    assert (
        package.total_uncompressed_bytes(INFO_YML_NAME) == package.entries[INFO_YML_NAME].file_size
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1817439042209534.json", "1817439042209534.json"),
        ("AT15.json", "AT15.json"),
        ("＃53682.json", "53682.json"),
        ("ドキドキ.pec", "chart.pec"),  # 全为非 ASCII → 回落 chart
        ("１２３.json", "123.json"),
        ("My Song!.json", "My_Song.json"),
        ("noext", "noext"),
    ],
)
def test_normalize_chart_filename(raw: str, expected: str) -> None:
    """R4：落盘不得沿用原始文件名（含全角字符）→ 规范化。"""
    result = normalize_chart_filename(raw)
    assert result == expected
    assert result.isascii()


def test_chart_dest_path_is_namespaced_by_chart_id() -> None:
    dest = chart_dest_path(Path("data/raw"), 1000, "＃53682.json")
    assert dest == Path("data/raw/1000/53682.json")
    assert chart_dest_path(Path("data/raw"), 7039, "24432296.json").name == "24432296.json"


def test_parse_info_yaml_rejects_non_mapping() -> None:
    with pytest.raises(ChartPackageError, match="映射"):
        parse_info_yaml(b"- just\n- a list\n")


def test_parse_info_yaml_tolerates_null_and_scalar_text_fields() -> None:
    """真实 `info.yml` 的文本字段可能是 null、数字或布尔（实测 2026-09-27，全库 9651 张）。

    - `tip: null` 极常见（首批 20 张里 10 张）；
    - 另有 `tip: 282033473720393` / `tip: False` / `charter: 55544462` / `composer: 416`
      这类把标量写进文本字段的包。

    原实现用裸 `str` 声明 ⇒ pydantic 拒绝整个 `info.yml` ⇒ 整张谱面在 package 阶段被误判为
    「结构错误」。这些字段只用于留痕与 `song_key` 分组，转成字符串不掩盖结构性错误。
    """
    info = parse_info_yaml(
        b"name: x\ntip: null\nlevel: null\ncharter: 55544462\ncomposer: false\n"
        b"chart: a.json\nmusic: b.mp3\ndifficulty: 15.5\n",
    )
    assert info.tip == ""
    assert info.level == ""
    assert info.charter == "55544462"
    assert info.composer == "False"
    assert info.chart == "a.json"
    assert info.song_key == "x|False"
    # format 恒为 null 且**不得**被归一化掉（R3：只记录不使用）
    assert parse_info_yaml(b"name: x\nformat: null\n").format is None


def test_parse_info_yaml_keeps_unknown_fields() -> None:
    info = parse_info_yaml(b"name: x\nchart: a.json\ncustomField: 42\n")
    assert info.name == "x"
    assert info.chart == "a.json"


def test_entry_lookup_is_exact() -> None:
    package = ChartPackage(
        entries={
            PKG_CHART_NAME: ZipEntry(
                name=PKG_CHART_NAME,
                compress_size=1,
                file_size=1,
                header_offset=0,
            ),
        },
        info=parse_info_yaml(read_bytes("pkg_min/info.yml")),
        chart_file=PKG_CHART_NAME,
        music_file="",
    )
    assert package.chart_file == PKG_CHART_NAME
    assert not package.missing_music
