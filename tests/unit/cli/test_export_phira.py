"""`beatmorph-export-phira` 的命令行契约（默认 CI；无权重 / 无 GPU）。

钉住三件事：**产物目录能一键补齐**（曲名/作者/定数/音频/时长都来自 `meta.json`）、
**退出码语义**（0 成功 / 3 输入不可用）、以及**导出后读得回来**（`ChartPackage`）。
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

from beatmorph.cli.export_phira import EXIT_INPUT, EXIT_OK, main
from beatmorph.core.contracts import ChartMeta, PhigrosChart
from beatmorph.data.phira.package import ChartPackage
from beatmorph.decoder import check_chart
from beatmorph.io.formats.rpejson import dump_rpejson
from tests.unit.field._builders import make_bpm_points, make_chart, make_note


def _chart(*, illegal: bool = False) -> PhigrosChart:
    notes = [make_note(t=0.0, line_id=0), make_note(t=1.0, line_id=0)]
    if illegal:
        notes = [make_note(t=0.0, line_id=0), make_note(t=0.0, line_id=0)]
    chart = make_chart(notes=notes, bpm_points=make_bpm_points((0.0, 120.0)), k=1, chart_time_s=4.0)
    return chart.model_copy(
        update={
            "meta": ChartMeta(
                chart_time_s=4.0,
                name="",
                composer="",
                charter="模板作者",
                level_text="IN Lv.15",
                difficulty=None,
            )
        }
    )


def _wav(path: Path, *, seconds: float = 2.0) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(b"\x00\x00" * int(seconds * 8000))
    return path


def _artifact(tmp_path: Path, *, illegal: bool = False, chart_written: bool = True) -> Path:
    """造一个「端到端产物」目录：`chart.json` + `meta.json`（字段与真产物同形）。"""
    directory = tmp_path / "outputs" / "20260929-214209-step0"
    directory.mkdir(parents=True)
    chart = _chart(illegal=illegal)
    if chart_written:
        report = check_chart(chart) if not illegal else None
        dump_rpejson(chart, directory / "chart.json", report=report)
    audio = _wav(directory / "song.wav")
    meta = {
        "step": 0,
        "chart_written": chart_written,
        "aborted": None,
        "difficulty": 15.3,
        "decode": {"method": "thinning"},
        "audio": {"path": str(audio), "sha1": "x"},
        "feature": {"path": "f.npz", "meta_path": None},
        "template": {"source": "manifest:test", "song_key": "赴大荒|塞壬唱片-MSR"},
    }
    (directory / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return directory


def test_cli_exports_from_an_artifact_directory(tmp_path: Path, capsys) -> None:
    """一键导出：`--from-e2e-artifact` 补齐曲名/作者/定数/音频，产出可读回的 `.pez`。"""
    directory = _artifact(tmp_path)
    out = tmp_path / "pkg.pez"
    code = main(["--from-e2e-artifact", str(directory), "--out", str(out), "--json"])
    assert code == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["info"]["name"] == "赴大荒"
    assert payload["info"]["composer"] == "塞壬唱片-MSR"
    assert payload["info"]["difficulty"] == 15.3
    assert payload["chart_notes"] == 2
    assert payload["size_bytes"] > 0
    assert payload["illustration_placeholder"] is True
    package = ChartPackage.open(out)
    assert package.info.name == "赴大荒"
    assert package.music_bytes() == (directory / "song.wav").read_bytes()
    assert "由 BeatMorph 自动生成" in package.info.intro
    assert "manifest:test" in package.info.intro, "溯源要写清条件取自哪"
    assert package.info.charter == "BeatMorph", "生成的谱面不得继承模板的谱师署名"
    assert package.info.level == "UK Lv.15.3", "也不继承模板的等级文本"


def test_cli_defaults_output_next_to_the_chart(tmp_path: Path) -> None:
    """不给 `--out` 时落在**谱面同目录**、同名换后缀（产物目录本身是不入库的）。"""
    directory = _artifact(tmp_path)
    code = main(["--from-e2e-artifact", str(directory)])
    assert code == EXIT_OK
    assert (directory / "chart.pez").is_file()


def test_cli_explicit_arguments_win_over_artifact_defaults(tmp_path: Path) -> None:
    """显式参数优先于产物默认值（否则「一键」会变成「无法覆盖」）。"""
    directory = _artifact(tmp_path)
    out = tmp_path / "custom.pez"
    code = main(
        [
            "--from-e2e-artifact",
            str(directory),
            "--out",
            str(out),
            "--name",
            "改过的曲名",
            "--level",
            "AT Lv.16",
            "--charter",
            "某人",
            "--tags",
            "a, b",
        ]
    )
    assert code == EXIT_OK
    package = ChartPackage.open(out)
    assert package.info.name == "改过的曲名"
    assert package.info.level == "AT Lv.16"
    assert package.info.charter == "某人"
    assert package.info.tags == ["a", "b"]


def test_cli_refuses_when_the_artifact_has_no_chart(tmp_path: Path, caplog) -> None:
    """产物被中止 / 不合法（`chart_written=false`）⇒ 退出码 3，且错误信息要说明原因。"""
    directory = _artifact(tmp_path, chart_written=False)
    code = main(["--from-e2e-artifact", str(directory)])
    assert code == EXIT_INPUT
    assert not (directory / "chart.pez").exists()


def test_cli_missing_chart_and_music_is_an_input_error() -> None:
    """既没给 `--chart/--music` 也没给产物目录 ⇒ 退出码 3（不是崩栈）。"""
    assert main([]) == EXIT_INPUT


def test_cli_refuses_an_illegal_chart_file(tmp_path: Path) -> None:
    """红线 6 在 CLI 一侧也要挡住（谱面文件不合法 ⇒ 3，不产出包）。"""
    chart_path = tmp_path / "illegal.json"
    dump_rpejson(_chart(illegal=True), chart_path, report=None, require_legal=False)
    audio = _wav(tmp_path / "song.wav")
    out = tmp_path / "pkg.pez"
    code = main(
        [
            "--chart",
            str(chart_path),
            "--music",
            str(audio),
            "--name",
            "曲",
            "--composer",
            "作者",
            "--difficulty",
            "15.3",
            "--out",
            str(out),
        ]
    )
    assert code == EXIT_INPUT
    assert not out.exists()
