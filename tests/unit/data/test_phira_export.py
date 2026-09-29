"""Phira 谱面包**导出**的护栏（默认 CI，无权重 / 无 GPU / 无真实语料）。

依据：Phira 官方文档 <https://teamflos.github.io/phira-docs/chart-standard/chartinfo.html>
与「谱面基本结构」。本文件钉死六件事：

1. **读侧能读回**：导出后用 `ChartPackage`（线上同款读法）重新打开，逐项核对；
2. **包结构**：zip 条目**全在根级**，且 `info.yml` 指名的三个文件都存在；
3. **不写客户端字段**：`format` / `id` / `uploader` / `created` / `updated` / `chartUpdated`
   一律不出现在 `info.yml` 里（官方明文 `format` 不应当手动填写）；
4. **红线 6**：不合法（`violations` 非空）时拒绝打包，且**不留半成品**；
5. **一份真相**：`offset` 由 RPE 的毫秒换算成 info.yml 的秒；`name`/`composer`/`difficulty`
   缺一即报错（官方默认值 `UK` / `10.0` 是占位符，不是事实）；
6. **可复现**：同样输入导出两次 ⇒ 逐字节相同。
"""

from __future__ import annotations

import hashlib
import json
import struct
import wave
import zipfile
import zlib
from pathlib import Path

import pytest
import yaml

from beatmorph.core.contracts import ChartMeta, ChartSource, PhigrosChart
from beatmorph.data.parsers.rpejson import parse_rpejson
from beatmorph.data.phira.export import (
    FORBIDDEN_INFO_FIELDS,
    PLACEHOLDER_ILLUSTRATION,
    PhiraExportError,
    PhiraExportSpec,
    build_info,
    duration_from_feature_meta,
    export_phira_package,
    is_generated_chart,
    load_chart_for_export,
    placeholder_png,
    read_wav_duration_s,
    safe_entry_name,
)
from beatmorph.data.phira.package import INFO_YML_NAME, ChartPackage
from beatmorph.decoder import check_chart
from beatmorph.io.formats.rpejson import dump_rpejson, write_rpejson
from tests.unit.field._builders import make_bpm_points, make_chart, make_note


def _chart(
    *,
    name: str = "测试曲",
    composer: str = "某作曲家",
    difficulty: float = 15.3,
    offset_ms: float = -250.0,
    k: int = 2,
    illegal: bool = False,
) -> PhigrosChart:
    """合成一张**合法**（或故意不合法）的谱面，元数据齐备以便测「从 IR 派生」。"""
    notes = [make_note(t=0.0, line_id=0), make_note(t=1.0, line_id=1)]
    if illegal:
        notes = [make_note(t=0.0, line_id=0), make_note(t=0.0, line_id=0)]
    chart = make_chart(
        notes=notes,
        bpm_points=make_bpm_points((0.0, 150.0)),
        k=k,
        chart_time_s=4.0,
    )
    return chart.model_copy(
        update={
            "meta": ChartMeta(
                chart_time_s=4.0,
                offset_ms=offset_ms,
                name=name,
                composer=composer,
                charter="某谱师",
                difficulty=difficulty,
                level_text="AT Lv.15",
            )
        }
    )


def _wav(path: Path, *, seconds: float = 2.0, rate: int = 8000) -> Path:
    """写一个最短的合法 WAV（1 声道 16bit 静音）——用于「时长可自动读」的路径。"""
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(seconds * rate))
    return path


def _fake_mp3(path: Path, *, size: int = 4096) -> Path:
    """假音频（读不出时长 ⇒ 走「时长未知」的分支）。"""
    path.write_bytes(b"ID3" + bytes(size))
    return path


def _spec(**overrides: object) -> PhiraExportSpec:
    values: dict[str, object] = {
        "chart": _chart(),
        "music": Path("song.mp3"),
    }
    values.update(overrides)
    return PhiraExportSpec(**values)  # type: ignore[arg-type]


def test_export_round_trips_through_the_reader(tmp_path: Path) -> None:
    """核心：导出后由**读侧**（ChartPackage）打开，字段 / 字节 / 谱面全部对得上。"""
    music = _fake_mp3(tmp_path / "song.mp3")
    chart = _chart()
    out = tmp_path / "pkg.pez"
    report = export_phira_package(
        PhiraExportSpec(
            chart=chart,
            music=music,
            name="测试曲",
            composer="某作曲家",
            level="AT Lv.15",
            difficulty=15.3,
            charter="某谱师",
            tags=("自动生成", "BeatMorph"),
            intro="一句话简介",
        ),
        out,
    )
    assert out.is_file()
    package = ChartPackage.open(out)
    assert package.info.name == "测试曲"
    assert package.info.composer == "某作曲家"
    assert package.info.charter == "某谱师"
    assert package.info.level == "AT Lv.15"
    assert package.info.difficulty == pytest.approx(15.3)
    assert package.info.chart == "chart.json"
    assert package.info.music == "song.mp3"
    assert package.info.tags == ["自动生成", "BeatMorph"]
    assert package.info.intro == "一句话简介"
    assert package.info.format is None, "官方：format 不应当手动填写"
    assert package.chart_bytes() == write_rpejson(chart, report=check_chart(chart))
    back = parse_rpejson(package.chart_bytes(), ChartSource())
    assert len(back.notes) == len(chart.notes)
    assert package.music_bytes() == music.read_bytes()
    assert report.chart_sha1 == hashlib.sha1(package.chart_bytes()).hexdigest()
    assert report.music_sha1 == hashlib.sha1(music.read_bytes()).hexdigest()
    assert report.illustration_placeholder is True
    assert report.chart_notes == 2


def test_zip_entries_are_root_level_and_match_info_yml(tmp_path: Path) -> None:
    """官方「解压后根级直接是文件」：条目不含目录，且与 info.yml 指名一致。"""
    music = _fake_mp3(tmp_path / "song.mp3")
    out = tmp_path / "pkg.pez"
    export_phira_package(_spec(music=music, name="曲", composer="作者", difficulty=1.0), out)
    with zipfile.ZipFile(out) as archive:
        names = [item.filename for item in archive.infolist() if not item.is_dir()]
        raw = yaml.safe_load(archive.read(INFO_YML_NAME).decode("utf-8"))
    assert names == [INFO_YML_NAME, "chart.json", "song.mp3", PLACEHOLDER_ILLUSTRATION]
    assert all("/" not in name for name in names)
    assert {raw["chart"], raw["music"], raw["illustration"]} <= set(names)
    assert set(FORBIDDEN_INFO_FIELDS).isdisjoint(raw)
    assert "tip" not in raw, "tip 留空则不写（官方：不写会塞一条自己的）"
    assert raw["previewStart"] == 0.0
    assert "previewEnd" not in raw, "时长未知 ⇒ 留给 Phira 自己截断"
    assert raw["holdPartialCover"] is False


def test_required_metadata_is_not_replaced_by_official_placeholders(tmp_path: Path) -> None:
    """`name` / `composer` / `difficulty` 缺一即报错——官方默认值是占位符，不是事实。"""
    music = _fake_mp3(tmp_path / "song.mp3")
    plain = make_chart(notes=[make_note(t=0.0)], bpm_points=make_bpm_points((0.0, 120.0)), k=1)
    with pytest.raises(PhiraExportError, match="name"):
        build_info(
            PhiraExportSpec(
                chart=plain,
                music=music,
                composer="作者",
                difficulty=1.0,
                music_filename="song.mp3",
                illustration_filename="a.png",
            )
        )
    with pytest.raises(PhiraExportError, match="composer"):
        build_info(
            PhiraExportSpec(
                chart=plain,
                music=music,
                name="曲",
                difficulty=1.0,
                music_filename="song.mp3",
                illustration_filename="a.png",
            )
        )
    with pytest.raises(PhiraExportError, match="difficulty"):
        build_info(
            PhiraExportSpec(
                chart=plain,
                music=music,
                name="曲",
                composer="作者",
                music_filename="song.mp3",
                illustration_filename="a.png",
            )
        )


def test_generated_chart_does_not_inherit_the_template_charter() -> None:
    """生成谱面**不得**继承模板的谱师 / 等级文本（实测踩过：包里写着模板作者的名字）。"""
    inherited = ChartSource(sniff_evidence="manifest:test")
    meta = ChartMeta(
        chart_time_s=4.0,
        name="赴大荒",
        composer="塞壬唱片-MSR",
        charter="shabu5553",
        difficulty=15.0,
        level_text="IN Lv.15",
    )
    template_like = make_chart(
        notes=[make_note(t=0.0)], bpm_points=make_bpm_points((0.0, 164.0)), k=1
    ).model_copy(update={"meta": meta, "source": inherited})
    info, warnings = build_info(
        PhiraExportSpec(
            chart=template_like,
            music=Path("song.mp3"),
            music_filename="song.mp3",
            illustration_filename="a.png",
        )
    )
    assert info["charter"] == "shabu5553", "真人谱面照常继承（这是往返导出）"
    assert info["level"] == "IN Lv.15"
    assert not any("不继承" in item for item in warnings)

    generated = template_like.model_copy(
        update={"source": ChartSource(sniff_evidence="decoded:peaks")}
    )
    assert is_generated_chart(generated) is True
    info, warnings = build_info(
        PhiraExportSpec(
            chart=generated,
            music=Path("song.mp3"),
            music_filename="song.mp3",
            illustration_filename="a.png",
        )
    )
    assert info["charter"] == "BeatMorph", "生成谱面只能署名生成方"
    assert info["level"] == "UK Lv.15"
    assert info["name"] == "赴大荒", "曲名描述的是这首歌 ⇒ 可以继承"
    assert info["composer"] == "塞壬唱片-MSR"
    assert sum("不继承" in item for item in warnings) == 2


def test_export_refuses_an_illegal_chart_and_leaves_no_file(tmp_path: Path) -> None:
    """红线 6：违规项非空 ⇒ 拒绝打包，且**不留半成品**。"""
    music = _fake_mp3(tmp_path / "song.mp3")
    out = tmp_path / "illegal.pez"
    with pytest.raises(PhiraExportError, match="红线 6"):
        export_phira_package(
            PhiraExportSpec(
                chart=_chart(illegal=True),
                music=music,
                name="曲",
                composer="作者",
                difficulty=15.3,
            ),
            out,
        )
    assert not out.exists(), "被拒绝的导出不得留下文件"


def test_offset_comes_from_the_rpe_meta_in_seconds(tmp_path: Path) -> None:
    """RPE 的 `META.offset` 是**毫秒**，`info.yml.offset` 是**秒**（写反了就是整首歌错位）。"""
    info, _warnings = build_info(
        _spec(
            chart=_chart(offset_ms=-250.0),
            name="曲",
            composer="作者",
            difficulty=15.3,
            music_filename="song.mp3",
            illustration_filename="a.png",
        )
    )
    assert info["offset"] == pytest.approx(-0.25)
    explicit, _w = build_info(
        _spec(
            name="曲",
            composer="作者",
            difficulty=15.3,
            offset_s=1.5,
            music_filename="song.mp3",
            illustration_filename="a.png",
        )
    )
    assert explicit["offset"] == pytest.approx(1.5), "显式值优先"


def test_preview_end_is_clamped_and_flagged(tmp_path: Path) -> None:
    """超出音频时长的 previewEnd 按官方行为截断，并如实告警（不静默改数）。"""
    info, warnings = build_info(
        _spec(
            name="曲",
            composer="作者",
            difficulty=1.0,
            preview_start=1.0,
            preview_end=99.0,
            music_duration_s=2.0,
            music_filename="song.mp3",
            illustration_filename="a.png",
        )
    )
    assert info["previewEnd"] == pytest.approx(2.0)
    assert any("截断" in item for item in warnings)
    with pytest.raises(PhiraExportError, match="previewEnd"):
        build_info(
            _spec(
                name="曲",
                composer="作者",
                difficulty=1.0,
                preview_start=5.0,
                preview_end=5.0,
                music_filename="song.mp3",
                illustration_filename="a.png",
            )
        )


def test_export_is_byte_reproducible(tmp_path: Path) -> None:
    """同输入两次导出**逐字节相同**（zip 条目时间戳固定）——归档与比对的前提。"""
    music = _fake_mp3(tmp_path / "song.mp3")
    spec = _spec(music=music, name="曲", composer="作者", difficulty=1.0)
    first = export_phira_package(spec, tmp_path / "a.pez")
    second = export_phira_package(spec, tmp_path / "b.pez")
    assert (tmp_path / "a.pez").read_bytes() == (tmp_path / "b.pez").read_bytes()
    assert first.size_bytes == second.size_bytes


def test_root_level_names_reject_directories_and_keep_cjk() -> None:
    """包内文件名必须落在根级；CJK 保留（真实谱面包里就有全角名）。"""
    assert safe_entry_name("a/b/song.mp3", field_name="music") == "song.mp3"
    assert safe_entry_name("..\\song.mp3", field_name="music") == "song.mp3"
    assert safe_entry_name("赴大荒.mp3", field_name="music") == "赴大荒.mp3"
    for dotted in ("", "..", "..."):
        with pytest.raises(PhiraExportError, match="非法"):
            safe_entry_name(dotted, field_name="music")


def test_music_named_entry_is_what_info_points_to(tmp_path: Path) -> None:
    """源文件名带目录时，写成根级名，且 `info.yml.music` 指的就是那个名字。"""
    nested = tmp_path / "audio"
    nested.mkdir()
    music = _fake_mp3(nested / "song.mp3")
    out = tmp_path / "pkg.pez"
    report = export_phira_package(
        _spec(
            music=music, chart_filename="谱面/foo.json", name="曲", composer="作者", difficulty=1.0
        ),
        out,
    )
    assert report.info["chart"] == "foo.json"
    assert report.info["music"] == "song.mp3"
    package = ChartPackage.open(out)
    assert package.chart_file == "foo.json"


def test_placeholder_png_is_a_valid_png() -> None:
    """占位曲绘必须是**真 PNG**（Phira 要渲染它），只用标准库生成。"""
    data = placeholder_png(width=64, height=32, rgb=(10, 20, 30))
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    offset = 8
    chunks: dict[bytes, bytes] = {}
    while offset < len(data):
        (length,) = struct.unpack(">I", data[offset : offset + 4])
        kind = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + length]
        (crc,) = struct.unpack(">I", data[offset + 8 + length : offset + 12 + length])
        assert crc == zlib.crc32(kind + payload), "PNG 每个块的 CRC 必须自洽"
        chunks[kind] = payload
        offset += 12 + length
    assert offset == len(data)
    width, height, depth, colour_type = struct.unpack(">IIBB", chunks[b"IHDR"][:10])
    assert (width, height, depth, colour_type) == (64, 32, 8, 2)
    raw = zlib.decompress(chunks[b"IDAT"])
    assert len(raw) == height * (1 + width * 3), "每行一个 filter 字节 + RGB"
    with pytest.raises(ValueError, match="尺寸"):
        placeholder_png(width=0, height=4)


def test_wav_duration_is_read_and_other_formats_are_not_guessed(tmp_path: Path) -> None:
    """时长只从 WAV 精确读；读不到返回 None（**不猜**：近似值会被当成事实写进 previewEnd）。"""
    wav = _wav(tmp_path / "song.wav", seconds=1.5)
    assert read_wav_duration_s(wav) == pytest.approx(1.5, abs=1e-3)
    assert read_wav_duration_s(_fake_mp3(tmp_path / "song.mp3")) is None


def test_duration_from_feature_meta_reads_our_own_cache(tmp_path: Path) -> None:
    """本项目特征缓存里的 `duration_s` 是精确值（mp3 也能用它做预览截断）。"""
    meta = tmp_path / "song.json"
    meta.write_text(json.dumps({"rate": 75.0, "duration_s": 214.31}), encoding="utf-8")
    assert duration_from_feature_meta(meta) == pytest.approx(214.31)
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert duration_from_feature_meta(bad) is None
    other = tmp_path / "other.json"
    other.write_text("{}", encoding="utf-8")
    assert duration_from_feature_meta(other) is None


def test_load_chart_for_export_reports_missing_file(tmp_path: Path) -> None:
    """读谱面：文件不存在要报错（不是静默返回空谱）。"""
    with pytest.raises(PhiraExportError, match="不存在"):
        load_chart_for_export(tmp_path / "nope.json")
    chart_path = dump_rpejson(_chart(), tmp_path / "chart.json", report=check_chart(_chart()))
    loaded = load_chart_for_export(chart_path)
    assert len(loaded.notes) == 2
    assert loaded.lines
    assert len(loaded.lines) == 2
