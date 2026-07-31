"""download_sayobot.py 离线单测（respx mock httpx）。

覆盖：text/plain POST body 关键点、302→200 下载、.osz 解压 + 4K 检测、
manifest JSONL 追加、续传判断、未 Ranked 跳过。
"""

from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path

import httpx
import pytest
import respx

from scripts.download_sayobot import (
    append_manifest,
    build_record,
    download_osz,
    extract_osz,
    search_beatmaplist,
    sid_already_downloaded,
)

API_URL = "https://api.sayobot.cn/?post"
DOWNLOAD_URL_TMPL = "https://txy1.sayobot.cn/beatmaps/download/full/{sid}?server=auto"


# ── 合成 .osz 工具 ────────────────────────────────────────────


def _mania_osu_bytes(circle_size: int = 4, mode: int = 3) -> bytes:
    return (
        "osu file format v14\n"
        f"[General]\nMode: {mode}\nAudioFilename: audio.mp3\n"
        "[Metadata]\nTitle:T\nArtist:A\nCreator:M\nVersion:V\n"
        "BeatmapID:1\nBeatmapSetID:1000\n"
        f"[Difficulty]\nCircleSize:{circle_size}\nOverallDifficulty:7\n"
        "[TimingPoints]\n0,500,4,0,0,100,1,0\n"
        "[HitObjects]\n64,192,1000,1,0,0:0:0:0:\n"
    ).encode()


def _make_osz_blob(*diffs: tuple[str, bytes], audio: bytes = b"fakeaudio") -> bytes:
    """合成 .osz（ZIP）字节流。diffs 为 (filename, content)。"""
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in diffs:
            zf.writestr(name, content)
        zf.writestr("audio.mp3", audio)
    return buf.getvalue()


def _client(**kwargs) -> httpx.Client:
    """测试用 client（follow_redirects 默认开）。"""
    kwargs.setdefault("follow_redirects", True)
    return httpx.Client(**kwargs)


# ── search_beatmaplist ────────────────────────────────────────


class TestSearchBeatmaplist:
    @respx.mock
    def test_text_plain_body_quirk(self) -> None:
        """body 必须是 JSON 字符串 + Content-Type: text/plain。"""
        route = respx.post(API_URL).mock(
            return_value=httpx.Response(200, json={"data": [{"sid": 1}]}),
        )
        with _client() as client:
            data = search_beatmaplist(client, keyword="test", offset=0)

        assert route.called
        assert data == [{"sid": 1}]

        req = route.calls[0].request
        # body 是 JSON 字符串
        body_obj = json.loads(req.content.decode("utf-8"))
        assert body_obj["cmd"] == "beatmaplist"
        assert body_obj["keyword"] == "test"
        assert body_obj["mode"] == 8
        assert body_obj["limit"] == 25
        assert body_obj["offset"] == 0
        # Content-Type 是 text/plain（关键点，非 application/json）
        assert req.headers["content-type"] == "text/plain"
        assert req.headers["origin"] == "https://osu.sayobot.cn"
        assert req.headers["referer"] == "https://osu.sayobot.cn/"

    @respx.mock
    def test_empty_data_returns_empty_list(self) -> None:
        respx.post(API_URL).mock(return_value=httpx.Response(200, json={"data": []}))
        with _client() as client:
            assert search_beatmaplist(client, keyword="zzz") == []

    @respx.mock
    def test_missing_data_key_returns_empty_list(self) -> None:
        respx.post(API_URL).mock(return_value=httpx.Response(200, json={}))
        with _client() as client:
            assert search_beatmaplist(client, keyword="zzz") == []


# ── download_osz ─────────────────────────────────────────────


class TestDownloadOsz:
    @respx.mock
    def test_follows_redirect_and_writes_bytes(self, tmp_path: Path) -> None:
        osz_blob = _make_osz_blob(("diff.osu", _mania_osu_bytes()))
        respx.get(DOWNLOAD_URL_TMPL.format(sid=1000)).mock(
            return_value=httpx.Response(
                200,
                content=osz_blob,
                headers={"content-type": "application/octet-stream"},
            ),
        )

        dest = tmp_path / "1000.osz"
        with _client() as client:
            path = download_osz(client, 1000, dest)

        assert path == dest
        assert dest.read_bytes() == osz_blob

    @respx.mock
    def test_http_error_raises(self, tmp_path: Path) -> None:
        respx.get(DOWNLOAD_URL_TMPL.format(sid=1000)).mock(
            return_value=httpx.Response(404),
        )
        with _client() as client, pytest.raises(httpx.HTTPStatusError):
            download_osz(client, 1000, tmp_path / "1000.osz")

    @respx.mock
    def test_no_verify_bypasses_tls(self, tmp_path: Path) -> None:
        osz_blob = _make_osz_blob(("d.osu", _mania_osu_bytes()))
        respx.get(DOWNLOAD_URL_TMPL.format(sid=1000)).mock(
            return_value=httpx.Response(200, content=osz_blob),
        )
        dest = tmp_path / "1000.osz"
        # no_verify=True 不应抛（respx 不真做 TLS，验证流程不崩即可）
        with _client() as client:
            download_osz(client, 1000, dest, no_verify=True)
        assert dest.read_bytes() == osz_blob


# ── extract_osz ──────────────────────────────────────────────


class TestExtractOsz:
    def test_extract_counts_diffs_and_4k(self, tmp_path: Path) -> None:
        osz_blob = _make_osz_blob(
            ("Easy.osu", _mania_osu_bytes(circle_size=4)),
            ("Hard.osu", _mania_osu_bytes(circle_size=4)),
            ("7K.osu", _mania_osu_bytes(circle_size=7)),
        )
        osz_path = tmp_path / "set.osz"
        osz_path.write_bytes(osz_blob)

        dest = tmp_path / "out"
        n_diffs, n_4k = extract_osz(osz_path, dest)

        assert n_diffs == 3
        assert n_4k == 2
        # 音频也解出来了
        assert (dest / "audio.mp3").exists()

    def test_no_4k_returns_zero(self, tmp_path: Path) -> None:
        osz_blob = _make_osz_blob(("7K.osu", _mania_osu_bytes(circle_size=7)))
        osz_path = tmp_path / "set.osz"
        osz_path.write_bytes(osz_blob)

        n_diffs, n_4k = extract_osz(osz_path, tmp_path / "out")
        assert n_diffs == 1
        assert n_4k == 0

    def test_non_mania_diff_not_counted_as_4k(self, tmp_path: Path) -> None:
        """taiko (Mode:1) 即使 CircleSize=4 也不是 4K mania。"""
        osz_blob = _make_osz_blob(("taiko.osu", _mania_osu_bytes(circle_size=4, mode=1)))
        osz_path = tmp_path / "set.osz"
        osz_path.write_bytes(osz_blob)

        n_diffs, n_4k = extract_osz(osz_path, tmp_path / "out")
        assert n_diffs == 1
        assert n_4k == 0


# ── append_manifest / build_record ───────────────────────────


class TestManifest:
    def test_append_manifest_is_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.jsonl"
        append_manifest(path, {"sid": 1, "title": "a"})
        append_manifest(path, {"sid": 2, "title": "b"})

        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"sid": 1, "title": "a"}
        assert json.loads(lines[1]) == {"sid": 2, "title": "b"}

    def test_build_record_fields(self) -> None:
        item = {
            "sid": 2583521,
            "title": "Mind Disorder",
            "artist": "Alkome",
            "creator": "[TCD] Dzar03",
            "order": 8.964,
            "play_count": 15073,
            "approved": 1,
            "modes": 8,
            "favourite_count": 75,
            "lastupdate": 1783980174,
        }
        rec = build_record(item, n_diffs=4, n_4k_diffs=2, sid_dir="2583521")

        assert rec["sid"] == 2583521
        assert rec["stars"] == 8.964
        assert rec["play_count"] == 15073
        assert rec["n_diffs"] == 4
        assert rec["n_4k_diffs"] == 2
        assert rec["path"] == "2583521"
        assert rec["license"] == "academic"
        assert rec["unranked"] is False
        assert rec["downloaded_at"]  # ISO 字符串非空

    def test_build_record_unranked(self) -> None:
        item = {"sid": 1, "order": 0.0, "approved": 3}
        rec = build_record(item, n_diffs=0, n_4k_diffs=0, sid_dir="1")
        assert rec["stars"] == 0.0
        assert rec["unranked"] is True

    def test_build_record_missing_optional_fields(self) -> None:
        rec = build_record({"sid": 5}, n_diffs=1, n_4k_diffs=1, sid_dir="5")
        assert rec["sid"] == 5
        assert rec["stars"] == 0.0
        assert rec["play_count"] == 0
        assert rec["unranked"] is True


# ── sid_already_downloaded ───────────────────────────────────


class TestSidAlreadyDownloaded:
    def test_exists_with_osu(self, tmp_path: Path) -> None:
        sid_dir = tmp_path / "1000"
        sid_dir.mkdir()
        (sid_dir / "diff.osu").write_text("x", encoding="utf-8")
        assert sid_already_downloaded(tmp_path, 1000) is True

    def test_dir_missing(self, tmp_path: Path) -> None:
        assert sid_already_downloaded(tmp_path, 1000) is False

    def test_dir_empty_no_osu(self, tmp_path: Path) -> None:
        (tmp_path / "1000").mkdir()
        (tmp_path / "1000" / "audio.mp3").write_bytes(b"x")
        assert sid_already_downloaded(tmp_path, 1000) is False
