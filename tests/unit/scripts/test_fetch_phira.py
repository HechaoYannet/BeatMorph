"""scripts/fetch_phira.py 的驱动逻辑测试（**默认 CI：不发真实网络请求**）。

覆盖的是「驱动脚本自己写的那一层」：清单 provenance 落盘与续写、选择/抽样、
三态记账、以及 `ingest_chart` 用假 PhiraClient（本地 zip 提供 Range 语义）跑通
预筛 → 选择性下载 → 解析 → 质检 → 清单行 的完整链路。

真实全库拉取属 e2e（需要网络），不进默认 CI。
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from beatmorph.data.parsers.sniff import ChartFormat
from beatmorph.data.phira.client import (
    RANGE_PREFIX_BYTES,
    Manifest,
    PhiraApiError,
    Provenance,
    ZipEntry,
    ZipIndex,
    read_manifest,
    write_manifest,
)
from scripts.fetch_phira import (
    EXIT_ARGS,
    EXIT_DATA,
    EXIT_OK,
    AudioOutcome,
    FetchReport,
    IngestResult,
    Paths,
    _accumulate,
    _append_jsonl,
    _done_chart_ids,
    _ingest_guarded,
    _prepare_manifest,
    _safe_suffix,
    _select_rows,
    build_parser,
    cmd_fetch,
    cmd_pairs,
    ingest_chart,
    main,
)
from tests.fixtures.phigros import build_pkg_bytes, pkg_info_text, read_pec_masquerade
from tests.unit.data._helpers import build_rpe_bytes, note

# ══════════════════════════════════════════════════════════════
# 假客户端：用本地 zip 提供 Range 语义
# ══════════════════════════════════════════════════════════════


class FakePhiraClient:
    """离线假客户端：只实现 `ingest_chart` 用到的三个方法。

    `fetch_prefix` 返回**解压后**的前缀（真实现返回压缩前缀并部分解压），
    `n >= compress_size` 时返回整个条目——与真实现的语义一致（请求整个条目 ⇒ 拿到全文）。
    """

    def __init__(self, payload: bytes, *, fail: bool = False) -> None:
        self.payload = payload
        self.fail = fail
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            infos = [item for item in archive.infolist() if not item.is_dir()]
            self.contents = {item.filename: archive.read(item.filename) for item in infos}
            entries = tuple(
                ZipEntry(
                    name=item.filename,
                    compress_size=item.compress_size,
                    file_size=item.file_size,
                    header_offset=item.header_offset,
                    compress_type=item.compress_type,
                )
                for item in infos
            )
        self.index = ZipIndex(entries=entries, total_bytes=len(payload), source_url="mem://pkg")

    def fetch_zip_index(self, url: str, **_kwargs: object) -> ZipIndex:
        if self.fail:
            raise PhiraApiError("模拟网络失败")
        return self.index

    def fetch_prefix(self, url: str, entry: ZipEntry, n: int = RANGE_PREFIX_BYTES) -> bytes:
        if self.fail:
            raise PhiraApiError("模拟网络失败")
        content = self.contents[entry.name]
        return content if n >= entry.compress_size else content[:n]

    def download_entry(self, url: str, entry: ZipEntry, dest: Path) -> Path:
        if self.fail:
            raise PhiraApiError("模拟网络失败")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.contents[entry.name])
        return dest


def _zip_bytes(entries: dict[str, bytes]) -> bytes:
    """把 `{条目名: 字节}` 打成确定性 zip（固定时间戳）。"""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            info = zipfile.ZipInfo(filename=name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    return buffer.getvalue()


AUDIO_BYTES = b"never-a-real-audio-payload"


def _rpe_package_zip(*, chart_name: str = "1817439042209534.json") -> bytes:
    """一个可入库的最小谱面包（谱面文件不叫 chart.json；含音频条目）。"""
    info = (
        "name: fixture-song\n"
        "difficulty: 12.5\n"
        "level: FIX Lv.12\n"
        "charter: fixture-charter\n"
        "composer: fixture-composer\n"
        f"chart: {chart_name}\n"
        "music: song.mp3\n"
    )
    chart = build_rpe_bytes(
        lines=[{"Name": "line-0", "notes": [note(1, [0, 0, 1])]}],
    )
    return _zip_bytes(
        {"info.yml": info.encode("utf-8"), chart_name: chart, "song.mp3": AUDIO_BYTES}
    )


class _ExplodingClient:
    """一被请求就抛错的假客户端：用来证明「某些 chart_id 根本没被请求」。"""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def __enter__(self) -> _ExplodingClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def fetch_zip_index(self, *_args: object, **_kwargs: object) -> ZipIndex:
        raise PhiraApiError("不该被调用：该 chart_id 本应被跳过")


def _meta_row(url: str = "mem://pkg", chart_id: int = 1000) -> dict[str, object]:
    """一条 `/chart` 元数据行（形状同 `PhiraChartMeta.to_row()`）。"""
    return {
        "id": chart_id,
        "name": "fixture-song",
        "composer": "fixture-composer",
        "charter": "fixture-charter",
        "difficulty": 12.500001,
        "difficulty_round": 12.5,
        "level": "FIX Lv.12",
        "file": url,
        "song_key": "fixture-song|fixture-composer",
    }


def _provenance() -> Provenance:
    return Provenance(
        source="https://api.phira.cn/chart",
        query="unit-test",
        fetched_at="2026-09-27T00:00:00+00:00",
        purpose="train",
        script="scripts/fetch_phira.py",
        script_version="test",
    )


# ══════════════════════════════════════════════════════════════
# 小工具与记账
# ══════════════════════════════════════════════════════════════


def test_safe_suffix_sanitizes_and_falls_back() -> None:
    """扩展名只保留字母数字，缺失/异常一律回落 .bin。"""
    assert _safe_suffix("song.mp3") == ".mp3"
    assert _safe_suffix("song.MP3") == ".mp3"
    assert _safe_suffix("song") == ".bin"
    assert _safe_suffix("song.wav/../x") == ".bin"


def test_select_rows_window_offset_and_sampling() -> None:
    """顺序窗口与 offset；抽样必须可复现且与顺序无关。"""
    rows = [{"id": index} for index in range(50)]
    assert [row["id"] for row in _select_rows(rows, limit=3, offset=0, sample_seed=None)] == [
        0,
        1,
        2,
    ]
    assert [row["id"] for row in _select_rows(rows, limit=3, offset=10, sample_seed=None)] == [
        10,
        11,
        12,
    ]
    assert len(_select_rows(rows, limit=None, offset=0, sample_seed=None)) == 50

    first = _select_rows(rows, limit=8, offset=0, sample_seed=7)
    assert first == _select_rows(rows, limit=8, offset=0, sample_seed=7)
    assert [row["id"] for row in first] == sorted(row["id"] for row in first)
    assert first != _select_rows(rows, limit=8, offset=0, sample_seed=8)


def test_manifest_prepare_append_and_resume(tmp_path: Path) -> None:
    """provenance 先落盘、数据行逐条追加（崩溃可续跑），已入库 id 可读回。"""
    paths = Paths(tmp_path)
    _prepare_manifest(paths.charts, _provenance(), overwrite=True)
    _append_jsonl(paths.charts, [{"chart_id": 1}, {"chart_id": 2}])
    _append_jsonl(paths.charts, [{"chart_id": 3}])

    manifest = read_manifest(paths.charts)
    assert manifest.provenance == _provenance()
    assert [row["chart_id"] for row in manifest.rows] == [1, 2, 3]
    assert _done_chart_ids(paths.charts) == {1, 2, 3}
    assert _done_chart_ids(tmp_path / "missing.jsonl") == set()

    # overwrite 会重建清单（provenance 记录不会重复堆积）
    _prepare_manifest(paths.charts, _provenance(), overwrite=True)
    assert read_manifest(paths.charts).rows == []


def test_accumulate_counts_three_states_and_audio() -> None:
    """入库 / 拒收 / 网络失败三态与音频三态都必须各自计数。"""
    report = FetchReport()
    _accumulate(
        report,
        IngestResult(
            status="ok",
            chart_bytes=100,
            fmt=ChartFormat.RPE,
            audio=AudioOutcome(status="ok", n_bytes=7, key="k"),
        ),
    )
    _accumulate(report, IngestResult(status="rejected", fmt=ChartFormat.PEC))
    _accumulate(report, IngestResult(status="failed", fmt=ChartFormat.UNKNOWN))
    _accumulate(
        report,
        IngestResult(
            status="ok",
            chart_bytes=1,
            fmt=ChartFormat.RPE,
            audio=AudioOutcome(status="failed"),
        ),
    )
    _accumulate(
        report,
        IngestResult(
            status="ok",
            chart_bytes=1,
            fmt=ChartFormat.RPE,
            audio=AudioOutcome(status="absent"),
        ),
    )
    assert (report.ok, report.failed, sum(report.rejected.values())) == (3, 1, 1)
    assert (report.audio_ok, report.audio_failed, report.audio_absent) == (1, 1, 1)
    assert report.chart_bytes == 102
    assert report.audio_bytes == 7
    assert report.formats[str(ChartFormat.RPE)] == 3
    assert report.formats[str(ChartFormat.PEC)] == 1


# ══════════════════════════════════════════════════════════════
# ingest_chart：三段式链路
# ══════════════════════════════════════════════════════════════


def test_ingest_chart_ok_writes_chart_audio_and_row(tmp_path: Path) -> None:
    """正样本：落盘谱面 + 音频（sha1 命名）+ 清单行（含 provenance 留痕列）。"""
    paths = Paths(tmp_path)
    payload = _rpe_package_zip()
    client = FakePhiraClient(payload)

    result = ingest_chart(client, _meta_row(), paths, with_audio=True)

    assert result.status == "ok"
    assert result.row is not None
    row = result.row
    expected_key = hashlib.sha1(AUDIO_BYTES).hexdigest()
    assert row["chart_id"] == 1000
    assert row["difficulty"] == 12.5
    assert row["name"] == "fixture-song"
    assert (paths.charts_dir / str(row["chart_path"])).is_file()
    assert not Path(str(row["chart_path"])).is_absolute()
    assert row["feature_key"] == expected_key
    assert row["audio_sha1"] == expected_key
    assert row["audio_status"] == "ok"
    assert (paths.audio_dir / f"{expected_key}.mp3").is_file()
    assert row["qc_passed"] is True
    assert row["chart_file"] == "1817439042209534.json"
    # 音频条目**不得**留在临时文件里
    assert list(paths.audio_dir.glob("*.part")) == []


def test_ingest_chart_dedups_shared_audio(tmp_path: Path) -> None:
    """同曲多谱共享同一音频：第二次只复用缓存，不重复落盘。"""
    paths = Paths(tmp_path)
    client = FakePhiraClient(_rpe_package_zip())
    first = ingest_chart(client, _meta_row(chart_id=1), paths, with_audio=True)
    second = ingest_chart(client, _meta_row(chart_id=2), paths, with_audio=True)
    assert first.row is not None
    assert second.row is not None
    assert first.row["feature_key"] == second.row["feature_key"]
    assert len(list(paths.audio_dir.iterdir())) == 1


def test_ingest_chart_no_audio_flag(tmp_path: Path) -> None:
    """--no-audio：谱面照收，音频状态记账为 absent，feature_key 为空。"""
    paths = Paths(tmp_path)
    client = FakePhiraClient(_rpe_package_zip())
    result = ingest_chart(client, _meta_row(), paths, with_audio=False)
    assert result.status == "ok"
    assert result.row is not None
    assert result.row["feature_key"] is None
    assert result.row["audio_status"] == "absent"
    assert not paths.audio_dir.exists()


def test_ingest_chart_rejects_pec_masquerade(tmp_path: Path) -> None:
    """陷阱 1：内容是 PEC、后缀是 .json → 必须拒收进隔离区，且格式记为 pec。"""
    paths = Paths(tmp_path)
    info = "name: fixture\nchart: 24432296.json\nmusic: song.mp3\n"
    payload = _zip_bytes(
        {"info.yml": info.encode("utf-8"), "24432296.json": read_pec_masquerade()},
    )
    client = FakePhiraClient(payload)
    result = ingest_chart(client, _meta_row(), paths, with_audio=False)
    assert result.status == "rejected"
    assert result.quarantine is not None
    assert result.quarantine.fmt is ChartFormat.PEC
    assert str(result.quarantine.stage) == "sniff"
    # 拒收样本不得写进谱面目录（只有嗅探为 RPE 才落盘）
    assert not (paths.charts_dir / "1000").exists()


def test_ingest_chart_rejects_when_info_chart_entry_missing(tmp_path: Path) -> None:
    """陷阱 2：``info.yml.chart`` 指向不存在的条目 → 拒收，**不得**回退猜测。"""
    paths = Paths(tmp_path)
    info = pkg_info_text().replace("chart: min_chart.json", "chart: nope.json")
    client = FakePhiraClient(build_pkg_bytes(info_text=info))
    result = ingest_chart(client, _meta_row(), paths, with_audio=False)
    assert result.status == "rejected"
    assert result.quarantine is not None
    assert str(result.quarantine.stage) == "package"
    assert any("nope.json" in reason for reason in result.quarantine.reasons)


def test_ingest_guarded_maps_network_failure_to_failed_not_rejected(tmp_path: Path) -> None:
    """网络失败与内容拒收必须分开记账（前者可重跑，后者是数据问题）。"""
    paths = Paths(tmp_path)
    client = FakePhiraClient(_rpe_package_zip(), fail=True)
    result = _ingest_guarded(client, _meta_row(), paths, with_audio=False)
    assert result.status == "failed"
    assert result.quarantine is None
    assert "模拟网络失败" in result.reason


def test_ingest_chart_missing_file_url_is_rejected(tmp_path: Path) -> None:
    """元数据缺 file 直链 → 拒收（stage=package），不发任何请求。"""
    paths = Paths(tmp_path)
    client = FakePhiraClient(_rpe_package_zip())
    result = ingest_chart(client, _meta_row(url=""), paths, with_audio=False)
    assert result.status == "rejected"
    assert result.quarantine is not None
    assert str(result.quarantine.stage) == "package"


# ══════════════════════════════════════════════════════════════
# 子命令与 CLI
# ══════════════════════════════════════════════════════════════


def test_cmd_fetch_without_meta_returns_data_error(tmp_path: Path) -> None:
    """没有枚举清单就不能抓取（退出码 3），避免「凭空的抓取」。"""
    args = build_parser().parse_args(["fetch", "--root", str(tmp_path)])
    assert cmd_fetch(args) == EXIT_DATA


def test_quarantined_rows_are_not_refetched(tmp_path: Path) -> None:
    """已拒收的 chart_id 与已入库的一样算「处理过」——重跑不白烧带宽。"""
    paths = Paths(tmp_path)
    write_manifest(Manifest(provenance=_provenance(), rows=[]), paths.charts)
    write_manifest(
        Manifest(provenance=_provenance(), rows=[{"chart_id": 7, "quarantine_stage": "sniff"}]),
        paths.quarantine,
    )
    meta_rows = [{"id": 7, "file": "mem://pkg"}]
    write_manifest(Manifest(provenance=_provenance(), rows=meta_rows), paths.meta)

    # 默认：7 号已拒收 ⇒ 跳过（不发任何请求；这里用一个「一被调用就炸」的客户端证明这点）
    args = build_parser().parse_args(["fetch", "--root", str(tmp_path), "--no-audio"])
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("scripts.fetch_phira.PhiraClient", _ExplodingClient)
        code = cmd_fetch(args)
    assert code == EXIT_OK
    report = json.loads(paths.report.read_text(encoding="utf-8"))
    assert report["skipped_done"] == 1
    assert report["selected"] == 1

    # --retry-quarantined：必须真的再去试（客户端被调用 ⇒ 抛错 ⇒ 记为 failed）
    retry_args = build_parser().parse_args(
        ["fetch", "--root", str(tmp_path), "--no-audio", "--retry-quarantined"],
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("scripts.fetch_phira.PhiraClient", _ExplodingClient)
        assert cmd_fetch(retry_args) != EXIT_OK


def test_cmd_pairs_without_charts_manifest_returns_data_error(tmp_path: Path) -> None:
    """没有谱面清单就不能配对（退出码 3）。"""
    args = build_parser().parse_args(["pairs", "--root", str(tmp_path)])
    assert cmd_pairs(args) == EXIT_DATA


def test_cmd_pairs_splits_by_song_with_materialized_chart(tmp_path: Path) -> None:
    """配对链路：清单行 → pairs.json（带 provenance），谱面文件缺失的行被记账跳过。"""
    paths = Paths(tmp_path)
    relative = "1000/1817439042209534.json"
    target = paths.charts_dir / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(build_rpe_bytes())
    rows = [
        {"chart_id": 1000, "song_key": "a|x", "name": "a", "composer": "x", "chart_path": relative},
        {
            "chart_id": 1001,
            "song_key": "b|y",
            "name": "b",
            "composer": "y",
            "chart_path": "1001/none.json",
        },
    ]
    write_manifest(Manifest(provenance=_provenance(), rows=rows), paths.charts)
    args = build_parser().parse_args(
        ["pairs", "--root", str(tmp_path), "--allow-missing-features"],
    )
    assert cmd_pairs(args) == EXIT_OK
    payload = json.loads(paths.pairs.read_text(encoding="utf-8"))
    assert payload["provenance"]["script"] == "scripts/fetch_phira.py"
    assert payload["skipped_no_chart"] == 1
    assert len(payload["train"]) == 1


def test_main_rejects_negative_limit(tmp_path: Path) -> None:
    """`--limit` 为负 → 参数错误退出码，不做任何工作。"""
    assert main(["fetch", "--root", str(tmp_path), "--limit", "-1"]) == EXIT_ARGS


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
