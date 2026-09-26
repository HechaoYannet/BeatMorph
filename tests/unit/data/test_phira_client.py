"""M1：PhiraClient 分页枚举 + Range 预筛（**全部 mock，默认 CI 不发真实请求**）。"""

from __future__ import annotations

import inspect
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest

from beatmorph.data.phira.client import (
    CHART_LIST_RESULTS_KEY,
    CHART_PAGE_SIZE_MAX,
    REQUEST_INTERVAL_S,
    PhiraApiError,
    PhiraChartMeta,
    PhiraClient,
    PhiraZipError,
    ZipEntry,
    decompress_prefix,
    iter_meta_rows,
    parse_local_header,
)
from tests.fixtures.phigros import PKG_CHART_NAME, PKG_ENTRY_NAMES, build_pkg_bytes

CDN_URL = "https://cdn.example.invalid/files/fixture"
CALLBACK = Callable[[httpx.Request], httpx.Response]

#: 断点续传用例的落盘名（含子目录，顺带覆盖 mkdir 分支）。
CHART_NAME_FOR_TEST = "resumed/min_chart.json"


# ── 通用 mock 工具 ────────────────────────────────────────────


def _meta_payload(identifier: int) -> dict[str, Any]:
    """一条 /chart 列表记录（字段名对齐实测响应）。"""
    return {
        "id": identifier,
        "name": f"song-{identifier}",
        "level": f"IN Lv.{identifier}",
        "difficulty": 14.900001 + identifier,
        "charter": "charter",
        "composer": "composer",
        "tags": ["regular"],
        "created": "2026-09-25T00:00:00Z",
        "updated": "2026-09-25T00:00:00Z",
        "file": f"{CDN_URL}-{identifier}.zip",
        "illustration": "x.jpg",
        "preview": "p.mp3",
        "uploader": identifier,
    }


def _client(handler: CALLBACK, **kwargs: Any) -> PhiraClient:
    """把 httpx.MockTransport 注入客户端（不发真实网络请求）。"""
    return PhiraClient(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=0,
        retry_backoff_s=0.0,
        **kwargs,
    )


def _paged_handler(pages: dict[int, Sequence[dict[str, Any]]], count: int) -> CALLBACK:
    def handler(request: httpx.Request) -> httpx.Response:
        page = int(dict(request.url.params)["page"])
        payload = {"count": count, CHART_LIST_RESULTS_KEY: list(pages.get(page, []))}
        return httpx.Response(200, json=payload)

    return handler


class _RangeServer:
    """最小 HTTP Range 服务器（`bytes=start-end` → 206 + Content-Range）。"""

    def __init__(self, data: bytes, *, ignore_range: bool = False, fail_first: int = 0) -> None:
        self.data = data
        self.ignore_range = ignore_range
        self.fail_first = fail_first
        self.ranges: list[str | None] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        header = request.headers.get("range")
        self.ranges.append(header)
        if self.fail_first > 0:
            self.fail_first -= 1
            return httpx.Response(503)
        if header is None or self.ignore_range:
            return httpx.Response(200, content=self.data)
        spec = header.split("=", 1)[1]
        start_text, _, end_text = spec.partition("-")
        start = int(start_text)
        end = int(end_text) if end_text else len(self.data) - 1
        body = self.data[start : end + 1]
        headers = {"Content-Range": f"bytes {start}-{end}/{len(self.data)}"}
        return httpx.Response(206, content=body, headers=headers)


# ── M1：分页枚举 ──────────────────────────────────────────────


def test_pagination_yields_every_record() -> None:
    pages = {1: [_meta_payload(1), _meta_payload(2)], 2: [_meta_payload(3)], 3: []}
    client = _client(_paged_handler(pages, count=3))
    metas = list(client.iter_chart_meta(sleep_s=0.0))
    assert [meta.id for meta in metas] == [1, 2, 3]


def test_pagination_stops_at_count_without_extra_page() -> None:
    pages = {1: [_meta_payload(1), _meta_payload(2)]}
    client = _client(_paged_handler(pages, count=2))
    assert len(list(client.iter_chart_meta(sleep_s=0.0))) == 2


def test_results_key_must_be_plural() -> None:
    """M1 负样本：把响应键改成 C 级文档写的 `result` 必须报错。"""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"count": 1, "result": [_meta_payload(1)]})

    client = _client(handler)
    with pytest.raises(PhiraApiError, match="results"):
        list(client.iter_chart_meta(sleep_s=0.0))


def test_page_size_above_max_fails_fast() -> None:
    """pageNum 上限是实测事实（31 → HTTP 400）：越界必须在本地就给出可读错误。"""
    client = _client(_paged_handler({1: []}, count=0))
    with pytest.raises(PhiraApiError, match="pageNum"):
        next(client.iter_chart_meta(page_size=CHART_PAGE_SIZE_MAX + 1, sleep_s=0.0))
    with pytest.raises(PhiraApiError, match="pageNum"):
        next(client.iter_chart_meta(page_size=0, sleep_s=0.0))


def test_default_page_size_is_the_measured_max() -> None:
    assert inspect.signature(PhiraClient.iter_chart_meta).parameters["page_size"].default == (
        CHART_PAGE_SIZE_MAX
    )


def test_http_400_is_reported_readably() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="bad request")

    client = _client(handler)
    with pytest.raises(PhiraApiError) as excinfo:
        list(client.iter_chart_meta(sleep_s=0.0))
    message = str(excinfo.value)
    assert "400" in message
    assert "pageNum" in message


def test_count_mismatch_between_pages_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        page = int(dict(request.url.params)["page"])
        count = 4 if page == 1 else 5
        payload = {"count": count, CHART_LIST_RESULTS_KEY: [_meta_payload(1)]}
        return httpx.Response(200, json=payload)

    client = _client(handler)
    with pytest.raises(PhiraApiError, match="count"):
        list(client.iter_chart_meta(sleep_s=0.0))


def test_incomplete_pagination_raises() -> None:
    pages = {1: [_meta_payload(1)], 2: [], 3: []}
    client = _client(_paged_handler(pages, count=5))
    with pytest.raises(PhiraApiError, match="累计条数"):
        list(client.iter_chart_meta(sleep_s=0.0))


def test_missing_count_raises() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={CHART_LIST_RESULTS_KEY: []})

    client = _client(handler)
    with pytest.raises(PhiraApiError, match="count"):
        list(client.iter_chart_meta(sleep_s=0.0))


def test_max_pages_truncates_enumeration() -> None:
    """干跑开关：`max_pages` 截断时**不做**完整性断言（用于小规模速率探测）。"""
    pages = {1: [_meta_payload(1)], 2: [_meta_payload(2)], 3: [_meta_payload(3)]}
    client = _client(_paged_handler(pages, count=99))
    metas = list(client.iter_chart_meta(sleep_s=0.0, max_pages=2))
    assert [meta.id for meta in metas] == [1, 2]


def test_iter_meta_rows_flattens_to_manifest_rows() -> None:
    pages = {1: [_meta_payload(1), _meta_payload(2)], 2: []}
    client = _client(_paged_handler(pages, count=2))
    rows = list(iter_meta_rows(client))
    assert len(rows) == 2
    assert rows[0]["song_key"] == "song-1|composer"


def test_sleep_is_applied_between_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[float] = []
    monkeypatch.setattr("beatmorph.data.phira.client.time.sleep", recorded.append)
    pages = {1: [_meta_payload(1)], 2: [_meta_payload(2)], 3: []}
    client = _client(_paged_handler(pages, count=2))
    list(client.iter_chart_meta(sleep_s=0.25))
    assert recorded == [0.25]


def test_default_interval_is_positive() -> None:
    assert REQUEST_INTERVAL_S > 0.0


def test_meta_row_has_manifest_columns() -> None:
    meta = PhiraChartMeta.model_validate(_meta_payload(9))
    row = meta.to_row()
    for column in (
        "id",
        "name",
        "level",
        "difficulty",
        "charter",
        "composer",
        "tags",
        "created",
        "updated",
        "file",
    ):
        assert column in row
    # 定数比较口径：round 到 0.1（实测 f32 有 14.900001 这类误差）
    assert row["difficulty_round"] == round(meta.difficulty, 1)
    assert row["difficulty_round"] != meta.difficulty
    assert meta.song_key == "song-9|composer"


# ── Range 预筛与单条目下载 ────────────────────────────────────


@pytest.fixture
def pkg_bytes() -> bytes:
    return build_pkg_bytes()


@pytest.fixture
def real_sizes(pkg_bytes: bytes) -> dict[str, int]:
    import io

    with zipfile.ZipFile(io.BytesIO(pkg_bytes)) as archive:
        return {item.filename: item.file_size for item in archive.infolist()}


def test_fetch_zip_index_reads_central_directory(
    pkg_bytes: bytes,
    real_sizes: dict[str, int],
) -> None:
    server = _RangeServer(pkg_bytes)
    client = _client(server)
    index = client.fetch_zip_index(CDN_URL)
    assert set(index.names) == set(PKG_ENTRY_NAMES)
    assert index.total_bytes == len(pkg_bytes)
    for entry in index.entries:
        assert entry.file_size == real_sizes[entry.name]


def test_fetch_zip_index_survives_tiny_tail_window(pkg_bytes: bytes) -> None:
    """中央目录不在尾部窗口内时必须追加一次精确 Range 抓取。"""
    server = _RangeServer(pkg_bytes)
    client = _client(server)
    index = client.fetch_zip_index(CDN_URL, tail_bytes=64)
    assert set(index.names) == set(PKG_ENTRY_NAMES)
    assert len(server.ranges) >= 3


def test_fetch_zip_index_when_server_ignores_range(pkg_bytes: bytes) -> None:
    server = _RangeServer(pkg_bytes, ignore_range=True)
    client = _client(server)
    index = client.fetch_zip_index(CDN_URL)
    assert set(index.names) == set(PKG_ENTRY_NAMES)


def test_non_zip_payload_raises() -> None:
    server = _RangeServer(b"not a zip at all" * 16)
    client = _client(server)
    with pytest.raises(PhiraZipError):
        client.fetch_zip_index(CDN_URL)


def test_fetch_prefix_yields_sniffable_content(pkg_bytes: bytes) -> None:
    """24 KB 压缩前缀部分解压后仍能看到 RPE 特征字段。"""
    client = _client(_RangeServer(pkg_bytes))
    entry = client.fetch_zip_index(CDN_URL).by_name(PKG_CHART_NAME)
    prefix = client.fetch_prefix(CDN_URL, entry)
    assert b'"eventLayers"' in prefix
    assert len(prefix) <= entry.file_size


def test_download_entry_writes_only_that_entry(pkg_bytes: bytes, tmp_path: Path) -> None:
    """只下载谱面条目：不整包下载，也不落其它条目。"""
    client = _client(_RangeServer(pkg_bytes))
    entry = client.fetch_zip_index(CDN_URL).by_name(PKG_CHART_NAME)
    dest = tmp_path / "1000" / "min_chart.json"
    client.download_entry(CDN_URL, entry, dest)
    assert dest.read_bytes() == build_entry_bytes(pkg_bytes, PKG_CHART_NAME)
    assert not dest.with_name(dest.name + ".part").exists()


def test_download_entry_resumes_from_part_file(pkg_bytes: bytes, tmp_path: Path) -> None:
    server = _RangeServer(pkg_bytes)
    client = _client(server)
    entry = client.fetch_zip_index(CDN_URL).by_name(PKG_CHART_NAME)
    dest = tmp_path / CHART_NAME_FOR_TEST
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    part.write_bytes(compressed_slice(pkg_bytes, entry, fraction=0.5))
    client.download_entry(CDN_URL, entry, dest)
    assert dest.read_bytes() == build_entry_bytes(pkg_bytes, PKG_CHART_NAME)
    assert not part.exists()


def test_download_entry_detects_size_mismatch(pkg_bytes: bytes, tmp_path: Path) -> None:
    client = _client(_RangeServer(pkg_bytes))
    entry = client.fetch_zip_index(CDN_URL).by_name(PKG_CHART_NAME)
    tampered = type(entry)(
        name=entry.name,
        compress_size=entry.compress_size,
        file_size=entry.file_size + 1,
        header_offset=entry.header_offset,
        compress_type=entry.compress_type,
    )
    with pytest.raises(PhiraZipError, match="解压大小"):
        client.download_entry(CDN_URL, tampered, tmp_path / "x.json")


def test_content_length_uses_content_range(pkg_bytes: bytes) -> None:
    client = _client(_RangeServer(pkg_bytes))
    assert client.content_length(CDN_URL) == len(pkg_bytes)


def test_decompress_prefix_tolerates_truncation(pkg_bytes: bytes) -> None:
    entry = _entry_of(pkg_bytes, PKG_CHART_NAME)
    full = compressed_slice(pkg_bytes, entry, fraction=1.0)
    assert len(decompress_prefix(full, partial=False)) == entry.file_size
    truncated = decompress_prefix(full[: len(full) // 2], partial=True)
    assert 0 < len(truncated) < entry.file_size


def test_decompress_prefix_rejects_unknown_method() -> None:
    with pytest.raises(PhiraZipError, match="压缩方法"):
        decompress_prefix(b"data", compress_type=99)


# ── 本地辅助（解压/偏移复算，用于断点续传与尺寸断言）────────────────


def _entry_of(pkg_bytes: bytes, name: str) -> ZipEntry:
    client = _client(_RangeServer(pkg_bytes))
    return client.fetch_zip_index(CDN_URL).by_name(name)


def _data_offset(pkg_bytes: bytes, entry: ZipEntry) -> int:
    header = pkg_bytes[entry.header_offset : entry.header_offset + 30]
    _name_length, _extra_length, relative = parse_local_header(header)
    return entry.header_offset + relative


def compressed_slice(pkg_bytes: bytes, entry: ZipEntry, *, fraction: float) -> bytes:
    start = _data_offset(pkg_bytes, entry)
    length = max(1, int(entry.compress_size * fraction))
    return pkg_bytes[start : start + length]


def build_entry_bytes(pkg_bytes: bytes, name: str) -> bytes:
    import io

    with zipfile.ZipFile(io.BytesIO(pkg_bytes)) as archive:
        return archive.read(name)
