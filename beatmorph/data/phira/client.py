"""Phira 官方 API 客户端 + 包结构预筛 + 清单 provenance（plan 02 §3.1/§3.2/§4 / M1、M8）。

三段式获取的**前两段**（plan 02 §4）：

1. **枚举**（`iter_chart_meta`）：`GET /chart` 分页（实测 9649 张 / 322 页 / `pageNum` 上限 30）。
   响应键是 **`results`（复数）**，不是 C 级文档写的 `result`——冲突时以实测为准。
2. **预筛**（`fetch_zip_index` / `fetch_prefix` / `download_entry`）：HTTP Range 取 zip 尾部
   解析 EOCD + 中央目录，只取谱面条目的 24 KB 压缩前缀做嗅探，最后只下载那一个条目。

**为什么不用 zipfile 直接开包**：预筛阶段手里只有若干 HTTP Range 片段，不是一个可 seek 的
本地文件；而「全量下载再解包」正是本 plan 要避免的 76 GB 带宽。故本地实现 EOCD/中央目录
解析（标准库 `struct`），只用规范定义的固定布局，不引用任何第三方解析器源码。

**本模块同时承载 M8 硬约束③**（获取与处理脚本必须记录来源与用途）：`Provenance` 随
`Manifest` 落盘，缺 provenance 的清单在写入与读取时**都**报错。

对应用户文档：docs/knowledges/phira-dataset-survey.md §2.1、§4.2、§9.2。
"""

from __future__ import annotations

import hashlib
import json
import struct
import time
import zlib
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ConfigDict, Field

from beatmorph.core.logging import get_logger

logger = get_logger(__name__)

# ══════════════════════════════════════════════════════════════
# §3.1 常量（**实测事实**，不是物理常量；物理量一律派生自 core.contracts）
# ══════════════════════════════════════════════════════════════

PHIRA_API_BASE: str = "https://api.phira.cn"
CHART_LIST_ENDPOINT: str = "/chart"
#: ⚠️ 实测键名；C 级非官方文档写作 `result`（单数），二者冲突时以实测为准。
CHART_LIST_RESULTS_KEY: str = "results"
#: C 级文档的写法，仅用于错误提示（**不得**用于取值）。
CHART_LIST_DOCUMENTED_KEY: str = "result"
#: 实测 `pageNum=31` → HTTP 400。
CHART_PAGE_SIZE_MAX: int = 30
#: 2026-09-26 实测 `count`；作为枚举完整性断言，非硬编码语义。
CHART_TOTAL_EXPECTED: int = 9649
#: 谱面条目嗅探用的压缩前缀（实测有效）。
RANGE_PREFIX_BYTES: int = 24 * 1024
#: 中央目录读取窗口（实测有效）。
ZIP_TAIL_BYTES: int = 200 * 1024
#: 自限速：未观测到限流 ≠ 无限流（调研 Q-2）。
REQUEST_INTERVAL_S: float = 0.5
#: info.yml 中定位谱面文件的字段（**不得按名猜**：196/196 张都不叫 chart.json）。
CHART_FILE_FIELD: str = "chart"
#: 音频文件名的唯一来源。
CHART_MUSIC_FIELD: str = "music"
#: /chart 的 type 参数：3 = any（= 全部 9649）。
CHART_TYPE_ANY: int = 3
#: 单条目下载的分块大小（I/O 粒度，非物理常量）。
DOWNLOAD_CHUNK_BYTES: int = 1024 * 1024

#: zip 结构签名与定长头（PKWARE APPNOTE 定义）。
_EOCD_SIGNATURE = b"PK\x05\x06"
_EOCD_STRUCT = struct.Struct("<4sHHHHIIH")
_CENTRAL_SIGNATURE = b"PK\x01\x02"
_CENTRAL_STRUCT = struct.Struct("<4sHHHHHHIIIHHHHHII")
_LOCAL_SIGNATURE = b"PK\x03\x04"
_LOCAL_STRUCT = struct.Struct("<4sHHHHHIIIHH")
_ZIP64_EOCD_LOCATOR = b"PK\x06\x07"
_UTF8_NAME_FLAG = 0x800

T = TypeVar("T")


# ══════════════════════════════════════════════════════════════
# 异常
# ══════════════════════════════════════════════════════════════


class PhiraApiError(RuntimeError):
    """Phira API / HTTP Range 层的可读错误（含 HTTP 状态与请求参数）。"""


class PhiraZipError(RuntimeError):
    """zip 结构解析失败（EOCD / 中央目录 / 本地头 / deflate 流）。"""


class ManifestError(ValueError):
    """清单 schema 违约（M8 硬约束③：缺失 provenance 即报错）。"""


# ══════════════════════════════════════════════════════════════
# §4 清单 provenance（M8 硬约束③）
# ══════════════════════════════════════════════════════════════


class ManifestPurpose(StrEnum):
    """清单用途（M8：来源与用途必须留痕）。"""

    TRAIN = "train"
    EVAL = "eval"
    STATS = "stats"


class Provenance(BaseModel):
    """清单的来源/用途/脚本留痕（合规硬约束③，逐张可追溯）。

    全部字段必填（`min_length=1`）：任一为空即 schema 违约。`fetched_at` 用 ISO8601
    UTC 字符串（不引入 tz-naive datetime 的解析歧义）。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str = Field(min_length=1, description="API 端点 / 夹具路径")
    query: str = Field(min_length=1, description="查询条件（page/type/... 的文本化）")
    fetched_at: str = Field(min_length=1, description="ISO8601 UTC 抓取时间")
    purpose: ManifestPurpose
    script: str = Field(min_length=1, description="脚本标识")
    script_version: str = Field(min_length=1, description="脚本版本")
    chart_id_min: int | None = None
    chart_id_max: int | None = None


def utc_now_iso() -> str:
    """当前 UTC 时间的 ISO8601 字符串（清单留痕用）。"""
    return datetime.now(UTC).isoformat()


def provenance_for_api(
    *,
    query: str,
    purpose: ManifestPurpose,
    script: str,
    script_version: str,
    base_url: str = PHIRA_API_BASE,
    chart_id_min: int | None = None,
    chart_id_max: int | None = None,
    fetched_at: str | None = None,
) -> Provenance:
    """构造 API 来源的 provenance（枚举脚本的标准写法）。"""
    return Provenance(
        source=f"{base_url}{CHART_LIST_ENDPOINT}",
        query=query,
        fetched_at=fetched_at or utc_now_iso(),
        purpose=purpose,
        script=script,
        script_version=script_version,
        chart_id_min=chart_id_min,
        chart_id_max=chart_id_max,
    )


@dataclass(frozen=True)
class Manifest:
    """一份清单 = provenance + 行记录（M8：provenance 缺失即报错）。"""

    provenance: Provenance
    rows: list[dict[str, Any]] = field(default_factory=list)


def write_manifest(manifest: Manifest, path: Path) -> Path:
    """把清单连同 provenance 落盘。

    `.jsonl`（**CI 友好的规范格式**）：首行是 provenance 记录，其后是数据行。
    `.parquet`：需要可选的 `pyarrow`；provenance 冗余写入每一行的 `provenance_json` 列
    并同时写进文件级 schema metadata（列式格式没有「首行」这个概念）。
    """
    if not isinstance(manifest.provenance, Provenance):
        raise ManifestError(
            "清单缺少 provenance（M8 硬约束③：来源与用途必须留痕，缺失即报错）",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    provenance_json = manifest.provenance.model_dump_json()
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        record = {"_record": "provenance", **manifest.provenance.model_dump(mode="json")}
        lines = [json.dumps(record, ensure_ascii=False)]
        lines.extend(json.dumps(row, ensure_ascii=False, default=str) for row in manifest.rows)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    if suffix == ".parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - 默认 CI 环境无 pyarrow
            raise ManifestError(
                "写 .parquet 清单需要可选依赖 pyarrow（uv sync --extra train）；"
                "默认 CI 请用 .jsonl",
            ) from exc
        rows = [{**row, "provenance_json": provenance_json} for row in manifest.rows]
        table = pa.Table.from_pylist(rows)
        table = table.replace_schema_metadata({b"beatmorph_provenance": provenance_json.encode()})
        pq.write_table(table, path)
        return path

    raise ManifestError(f"不支持的清单后缀 {path.suffix!r}（仅 .jsonl / .parquet）")


def read_manifest(path: Path) -> Manifest:
    """读取清单；**provenance 缺失即报错**（M8 硬约束③）。"""
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        provenance: Provenance | None = None
        for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw.strip():
                continue
            record = json.loads(raw)
            if not isinstance(record, dict):
                raise ManifestError(f"{path}:{lineno} 记录不是对象")
            if record.get("_record") == "provenance":
                provenance = Provenance.model_validate(
                    {k: v for k, v in record.items() if k != "_record"},
                )
                continue
            rows.append(record)
        if provenance is None:
            raise ManifestError(f"{path} 缺少 provenance 记录（M8 硬约束③：来源与用途必须留痕）")
        return Manifest(provenance=provenance, rows=rows)

    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover
            raise ManifestError("读 .parquet 清单需要可选依赖 pyarrow") from exc
        table = pq.read_table(path)
        metadata = table.schema.metadata or {}
        rows = table.to_pylist()
        raw_provenance = metadata.get(b"beatmorph_provenance")
        if raw_provenance is not None:
            provenance = Provenance.model_validate_json(raw_provenance)
        elif rows and rows[0].get("provenance_json"):
            provenance = Provenance.model_validate_json(rows[0]["provenance_json"])
        else:
            raise ManifestError(f"{path} 缺少 provenance（M8 硬约束③）")
        return Manifest(
            provenance=provenance,
            rows=[{k: v for k, v in row.items() if k != "provenance_json"} for row in rows],
        )

    raise ManifestError(f"不支持的清单后缀 {path.suffix!r}（仅 .jsonl / .parquet）")


def sha1_hex(data: bytes) -> str:
    """内容 sha1（音频/谱面去重与外键）。"""
    # 非安全用途（去重外键）：sha1 足够且与调研的音频去重口径一致。
    return hashlib.sha1(data).hexdigest()


def sha1_file(path: Path, chunk_bytes: int = DOWNLOAD_CHUNK_BYTES) -> str:
    """文件内容 sha1（流式，不整文件读入内存）。"""
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


# ══════════════════════════════════════════════════════════════
# §3.2 元数据记录与 zip 条目
# ══════════════════════════════════════════════════════════════


class PhiraChartMeta(BaseModel):
    """`GET /chart` 列表的一条记录（M1 落盘列）。

    ⚠️ **不是**契约层的 `ChartMeta`：后者是 RPE `META.*` 的视图
    （`offset_ms` / `rpe_version` / `chart_time_s` …），字段完全不同。plan 02 §3.2 的
    签名沿用了 `ChartMeta` 这个名字，本实现改用 `PhiraChartMeta` 以免与契约类型混淆
    （偏离记录见交付回报）。
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    id: int
    name: str = ""
    level: str = ""
    #: **f32 定数**；实测存在浮点误差（14.900001 / 18.000004），比较前应 round 到 0.1
    difficulty: float | None = None
    charter: str = ""
    composer: str = ""
    tags: list[str] = Field(default_factory=list)
    created: str = ""
    updated: str = ""
    #: 谱面包 zip 的 CDN 直链（第三段「选择性下载」的输入）
    file: str = ""
    illustration: str = ""
    preview: str = ""
    uploader: int | None = None
    stable: bool | None = None
    ranked: bool | None = None

    @property
    def difficulty_round(self) -> float | None:
        """按 plan 02 §偏离 3：数值分层前 round 到 0.1（**禁止** regex 解析 level）。"""
        return None if self.difficulty is None else round(self.difficulty, 1)

    @property
    def song_key(self) -> str:
        """同曲近似键 = `name | composer`（info.yml 无全局唯一歌曲 ID，调研 §5.3-7）。"""
        return f"{self.name}|{self.composer}"

    def to_row(self) -> dict[str, Any]:
        """清单行（M1 要求含 id/name/level/difficulty/charter/composer/tags/created/updated/file）。"""
        row = self.model_dump(mode="json")
        row["difficulty_round"] = self.difficulty_round
        row["song_key"] = self.song_key
        return row


@dataclass(frozen=True)
class ZipEntry:
    """zip 中央目录中的一个条目。"""

    name: str
    compress_size: int
    file_size: int
    header_offset: int
    compress_type: int = 8

    @property
    def suffix(self) -> str:
        """条目名后缀（**仅作记账展示**；格式判定与定位都不得依赖它）。"""
        return Path(self.name).suffix.lower()


@dataclass(frozen=True)
class ZipIndex:
    """一个谱面包 zip 的中央目录视图。"""

    entries: tuple[ZipEntry, ...]
    total_bytes: int
    source_url: str
    tail_start: int = 0

    @property
    def names(self) -> tuple[str, ...]:
        """全部条目名（保持中央目录顺序）。"""
        return tuple(entry.name for entry in self.entries)

    def by_name(self, name: str) -> ZipEntry:
        """按精确条目名取条目；不存在抛 KeyError。"""
        for entry in self.entries:
            if entry.name == name:
                return entry
        raise KeyError(name)

    def get(self, name: str) -> ZipEntry | None:
        """按精确条目名取条目；不存在返回 None。"""
        for entry in self.entries:
            if entry.name == name:
                return entry
        return None

    def as_dict(self) -> dict[str, ZipEntry]:
        """`{条目名: 条目}`（重名时后者覆盖，zip 规范下不应出现）。"""
        return {entry.name: entry for entry in self.entries}


# ══════════════════════════════════════════════════════════════
# zip 结构解析（仅标准库 struct；不引用第三方解析器源码）
# ══════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class _Eocd:
    entry_count: int
    directory_size: int
    directory_offset: int


def _parse_eocd(data: bytes, *, start_offset: int, total_bytes: int) -> _Eocd:
    """从尾部窗口定位并解析 EOCD（End Of Central Directory）。"""
    if _ZIP64_EOCD_LOCATOR in data:
        raise PhiraZipError("检测到 ZIP64 EOCD 定位器；本实现不支持 ZIP64（谱面包实测均 < 32 MB）")
    position = data.rfind(_EOCD_SIGNATURE)
    if position < 0:
        raise PhiraZipError(
            f"未在尾部窗口（{len(data)} B，起点 {start_offset}/{total_bytes}）找到 EOCD 签名；"
            "可能是 zip 注释过长或该 URL 不是 zip",
        )
    (
        _signature,
        disk_number,
        directory_disk,
        _entries_this_disk,
        entry_count,
        directory_size,
        directory_offset,
        _comment_length,
    ) = _EOCD_STRUCT.unpack_from(data, position)
    if disk_number != 0 or directory_disk != 0:
        raise PhiraZipError(f"多卷 zip 不受支持（disk={disk_number}, cd_disk={directory_disk}）")
    return _Eocd(
        entry_count=int(entry_count),
        directory_size=int(directory_size),
        directory_offset=int(directory_offset),
    )


def _decode_entry_name(raw: bytes, flags: int) -> str:
    """条目名解码：UTF-8 标志位优先，否则 UTF-8 → cp437 兜底（PKWARE 规范）。"""
    if flags & _UTF8_NAME_FLAG:
        return raw.decode("utf-8", "replace")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp437", "replace")


def _parse_central_directory(
    data: bytes,
    *,
    start_offset: int,
    expected: int,
) -> tuple[ZipEntry, ...]:
    """解析中央目录条目表（文件名/压缩大小/解压大小/本地头偏移/压缩方法）。

    循环次数**恰好**等于 EOCD 声明的条目数（含目录条目）：窗口可能只覆盖中央目录本身
    （精确 Range 抓取路径），此时窗口末尾之后没有 EOCD 可供 break。
    """
    entries: list[ZipEntry] = []
    cursor = 0
    for _ in range(expected):
        signature = data[cursor : cursor + 4]
        if signature != _CENTRAL_SIGNATURE:
            raise PhiraZipError(
                f"中央目录签名不符（偏移 {start_offset + cursor}）：{signature!r}",
            )
        (
            _sig,
            _version_made,
            _version_needed,
            flags,
            compress_type,
            _mod_time,
            _mod_date,
            _crc,
            compress_size,
            file_size,
            name_length,
            extra_length,
            comment_length,
            _disk_start,
            _internal_attrs,
            _external_attrs,
            header_offset,
        ) = _CENTRAL_STRUCT.unpack_from(data, cursor)
        name_start = cursor + _CENTRAL_STRUCT.size
        name = _decode_entry_name(data[name_start : name_start + name_length], flags)
        cursor = name_start + name_length + extra_length + comment_length
        if name.endswith("/"):
            continue
        entries.append(
            ZipEntry(
                name=name,
                compress_size=int(compress_size),
                file_size=int(file_size),
                header_offset=int(header_offset),
                compress_type=int(compress_type),
            ),
        )
    return tuple(entries)


def entries_in_window(data: bytes, *, start_offset: int, eocd: _Eocd) -> tuple[ZipEntry, ...]:
    """在字节窗口内解析中央目录（窗口须覆盖 `[cd_offset, cd_offset + cd_size)`）。"""
    relative = eocd.directory_offset - start_offset
    if relative < 0 or relative + eocd.directory_size > len(data):
        raise PhiraZipError(
            f"中央目录 [{eocd.directory_offset}, +{eocd.directory_size}) 不在窗口 "
            f"[{start_offset}, {start_offset + len(data)}) 内",
        )
    return _parse_central_directory(
        data[relative:],
        start_offset=eocd.directory_offset,
        expected=eocd.entry_count,
    )


def parse_zip_index(
    data: bytes,
    *,
    start_offset: int,
    total_bytes: int,
    source_url: str = "",
) -> ZipIndex:
    """从（可能只是尾部的）zip 字节窗口解析条目表。

    调用方须保证窗口**同时**覆盖 EOCD 与中央目录；否则先用 :func:`_parse_eocd` 拿偏移、
    再抓一次中央目录区间，然后走 :func:`entries_in_window`（`fetch_zip_index` 即这条路径）。
    """
    eocd = _parse_eocd(data, start_offset=start_offset, total_bytes=total_bytes)
    return ZipIndex(
        entries=entries_in_window(data, start_offset=start_offset, eocd=eocd),
        total_bytes=total_bytes,
        source_url=source_url,
        tail_start=start_offset,
    )


def parse_local_header(data: bytes) -> tuple[int, int, int]:
    """解析本地文件头，返回 `(name_length, extra_length, data_offset)`。"""
    if len(data) < _LOCAL_STRUCT.size:
        raise PhiraZipError(f"本地文件头不足 {_LOCAL_STRUCT.size} 字节（得到 {len(data)}）")
    if data[:4] != _LOCAL_SIGNATURE:
        raise PhiraZipError(f"本地文件头签名不符：{data[:4]!r}")
    values = _LOCAL_STRUCT.unpack_from(data, 0)
    name_length = int(values[9])
    extra_length = int(values[10])
    return name_length, extra_length, _LOCAL_STRUCT.size + name_length + extra_length


# ══════════════════════════════════════════════════════════════
# §3.2 客户端
# ══════════════════════════════════════════════════════════════


class PhiraClient:
    """Phira API 客户端：分页枚举 + Range 预筛 + 单条目下载。

    Args:
        base_url: API 根（默认 :data:`PHIRA_API_BASE`）。
        client: 注入的 `httpx.Client`（单测用 `httpx.MockTransport`；默认自建）。
        max_retries: 传输失败的重试次数（指数退避）。
        retry_backoff_s: 退避基数（秒）；0 表示不睡眠（单测）。
        timeout_s: 单请求超时。
        total_expected: 枚举完整性断言的期望总数（默认 :data:`CHART_TOTAL_EXPECTED`）。

    Note:
        `iter_chart_meta` 只在**最后一页**做累计条数断言；若消费者提前 break，
        断言不会执行（生成器语义）。
    """

    def __init__(
        self,
        *,
        base_url: str = PHIRA_API_BASE,
        client: httpx.Client | None = None,
        max_retries: int = 3,
        retry_backoff_s: float = 1.0,
        timeout_s: float = 30.0,
        total_expected: int = CHART_TOTAL_EXPECTED,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self.total_expected = total_expected
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=timeout_s, follow_redirects=True)

    # ── 生命周期 ──────────────────────────────────────────────

    def close(self) -> None:
        """关闭自建连接池（注入的客户端由调用方负责）。"""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> PhiraClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ── M1：元数据枚举 ────────────────────────────────────────

    def iter_chart_meta(
        self,
        type: int = CHART_TYPE_ANY,  # 签名与 plan 02 §3.2 一致（名字沿用 API 参数名）
        sleep_s: float = REQUEST_INTERVAL_S,
        page_size: int = CHART_PAGE_SIZE_MAX,
        max_pages: int | None = None,
    ) -> Iterator[PhiraChartMeta]:
        """分页枚举全部谱面元数据（`type=3` 为 any，= 全部 9649）。

        断言（M1）：`results` 键存在；每页 `count` 与首页一致；累计条数 == `count`。

        Args:
            type: `/chart` 的类别参数（0 ranked / 1 special / 2 unstable / 3 any）。
            sleep_s: 页间自限速（秒）。
            page_size: 每页条数，必须落在 `1..CHART_PAGE_SIZE_MAX`。
            max_pages: 只取前 N 页（干跑/单测用）；None = 取到收敛。

        Raises:
            PhiraApiError: 分页参数越界、HTTP 错误、响应缺少 `results`、`count` 不一致。
        """
        if not 1 <= page_size <= CHART_PAGE_SIZE_MAX:
            raise PhiraApiError(
                f"pageNum 必须落在 1..{CHART_PAGE_SIZE_MAX}，得到 {page_size}"
                f"（实测 pageNum={CHART_PAGE_SIZE_MAX + 1} → HTTP 400）",
            )

        page = 1
        seen = 0
        expected: int | None = None
        while True:
            if page > 1 and sleep_s > 0:
                time.sleep(sleep_s)
            payload = self._get_json(
                CHART_LIST_ENDPOINT,
                params={"page": page, "pageNum": page_size, "type": type},
            )
            results = self._extract_results(payload)
            count = self._extract_count(payload)
            if expected is None:
                expected = count
                logger.info("Phira /chart 命中 %d 张（type=%d，每页 %d）", count, type, page_size)
            elif count != expected:
                raise PhiraApiError(
                    f"第 {page} 页 count={count} 与首页 count={expected} 不一致（分页期间库在变？）",
                )
            for raw in results:
                yield PhiraChartMeta.model_validate(raw)
            seen += len(results)
            if seen >= expected or not results:
                if seen != expected:
                    raise PhiraApiError(f"累计条数 {seen} != count {expected}（分页不完整）")
                return
            if max_pages is not None and page >= max_pages:
                logger.warning("max_pages=%d 截断枚举（已收 %d/%d）", max_pages, seen, expected)
                return
            page += 1

    def _extract_results(self, payload: Mapping[str, Any]) -> list[Any]:
        """取 `results` 列表；键名不符时给出可读错误（M1 负样本）。"""
        if CHART_LIST_RESULTS_KEY not in payload:
            raise PhiraApiError(
                f"响应缺少 {CHART_LIST_RESULTS_KEY!r} 键（C 级文档写作 "
                f"{CHART_LIST_DOCUMENTED_KEY!r}，实测为复数；收到的键：{sorted(payload)}）",
            )
        results = payload[CHART_LIST_RESULTS_KEY]
        if not isinstance(results, list):
            raise PhiraApiError(
                f"{CHART_LIST_RESULTS_KEY!r} 必须是数组，得到 {type(results).__name__}"
            )
        return results

    @staticmethod
    def _extract_count(payload: Mapping[str, Any]) -> int:
        """取 `count` 整数。"""
        count = payload.get("count")
        if not isinstance(count, int) or isinstance(count, bool):
            raise PhiraApiError(f"响应缺少整数 count，得到 {count!r}")
        return count

    # ── HTTP 原语 ─────────────────────────────────────────────

    def _get_json(self, path: str, *, params: Mapping[str, Any]) -> dict[str, Any]:
        """GET 并解析 JSON 对象；任何非 200 都抛可读错误。"""
        url = f"{self.base_url}{path}" if path.startswith("/") else path
        try:
            response = self._client.get(url, params=dict(params))
        except httpx.HTTPError as exc:
            raise PhiraApiError(f"请求失败 {url} params={dict(params)}: {exc}") from exc
        if response.status_code != httpx.codes.OK:
            raise PhiraApiError(_http_error_message(response, params))
        try:
            payload = response.json()
        except ValueError as exc:
            raise PhiraApiError(f"{url} 响应不是合法 JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise PhiraApiError(f"{url} 响应顶层不是对象，得到 {type(payload).__name__}")
        return payload

    def _request_range(self, url: str, start: int, end: int) -> tuple[int, bytes]:
        """取字节区间，返回 `(实际起点, 字节)`。

        服务端忽略 Range 而返回 200 时，返回整份内容与起点 0（调用方据此纠偏）；
        206 时若带 `Content-Range` 则以其声明的起点为准。**不做重试**（由调用方包裹）。
        """
        try:
            response = self._client.get(url, headers={"Range": f"bytes={start}-{end}"})
        except httpx.HTTPError as exc:
            raise PhiraApiError(f"Range 请求失败 {url} [{start}, {end}]: {exc}") from exc
        if response.status_code == httpx.codes.OK:
            logger.debug("服务端忽略 Range，返回整份内容（%d B）", len(response.content))
            return 0, response.content
        if response.status_code != httpx.codes.PARTIAL_CONTENT:
            raise PhiraApiError(
                f"Range 请求 {url} [{start}, {end}] 得到 HTTP {response.status_code}",
            )
        content_range = response.headers.get("content-range")
        actual = start
        if content_range:
            try:
                actual = int(content_range.split()[1].split("-")[0])
            except (IndexError, ValueError):
                logger.warning("无法解析 Content-Range=%r，沿用请求起点 %d", content_range, start)
        return actual, response.content

    def _with_retries(self, label: str, operation: Callable[[], T]) -> T:
        """指数退避重试（实测 200 张扫描有 ~3.5% 网络失败）。"""
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return operation()
            except (httpx.HTTPError, PhiraApiError) as exc:
                last = exc
                if attempt >= self.max_retries:
                    break
                delay = self.retry_backoff_s * (2**attempt)
                logger.warning(
                    "%s 失败（第 %d 次）：%s；%.1fs 后重试", label, attempt + 1, exc, delay
                )
                if delay > 0:
                    time.sleep(delay)
        raise PhiraApiError(f"{label} 重试 {self.max_retries} 次后仍失败：{last}") from last

    # ── §3.2 预筛：中央目录 + 前缀 ────────────────────────────

    def content_length(self, file_url: str) -> int:
        """取远端对象总字节数（`Range: bytes=0-0` → `Content-Range`）。"""

        def load() -> int:
            return self._content_length(file_url)

        return self._with_retries(f"content_length {file_url}", load)

    def _content_length(self, file_url: str) -> int:
        try:
            response = self._client.get(file_url, headers={"Range": "bytes=0-0"})
        except httpx.HTTPError as exc:
            raise PhiraApiError(f"取长度失败 {file_url}: {exc}") from exc
        if response.status_code == httpx.codes.OK:
            return len(response.content)
        if response.status_code != httpx.codes.PARTIAL_CONTENT:
            raise PhiraApiError(f"取长度 {file_url} 得到 HTTP {response.status_code}")
        content_range = response.headers.get("content-range", "")
        try:
            return int(content_range.rsplit("/", 1)[1])
        except (IndexError, ValueError) as exc:
            raise PhiraApiError(
                f"{file_url} 缺少可解析的 Content-Range：{content_range!r}"
            ) from exc

    def fetch_zip_index(self, file_url: str, *, tail_bytes: int = ZIP_TAIL_BYTES) -> ZipIndex:
        """HTTP Range 取 zip 尾部 → 解析 EOCD/中央目录 → 条目表。

        Args:
            file_url: 谱面包 zip 的 CDN 直链（`/chart` 的 `file` 字段）。
            tail_bytes: 尾部窗口大小；中央目录不在窗口内时会追加一次精确抓取。

        Raises:
            PhiraApiError: 传输失败。
            PhiraZipError: EOCD/中央目录结构异常。
        """
        total = self.content_length(file_url)

        def load_tail() -> tuple[int, bytes]:
            start = max(0, total - tail_bytes)
            return self._request_range(file_url, start, total - 1)

        start, window = self._with_retries(f"tail {file_url}", load_tail)
        if start != 0 and len(window) < total - start:
            raise PhiraApiError(
                f"{file_url} 尾部窗口不完整：拿到 {len(window)} B，期望 {total - start} B"
            )

        eocd = _parse_eocd(window, start_offset=start, total_bytes=total)
        if eocd.directory_offset < start:
            # 中央目录不在尾部窗口内：按 EOCD 给出的偏移**精确再抓一次**，只取目录本身。
            directory_end = eocd.directory_offset + eocd.directory_size
            logger.debug(
                "中央目录在尾部窗口之外（offset=%d < %d），追加一次 Range 抓取",
                eocd.directory_offset,
                start,
            )
            start, window = self._with_retries(
                f"cd {file_url}",
                lambda: self._request_range(file_url, eocd.directory_offset, directory_end - 1),
            )
        return ZipIndex(
            entries=entries_in_window(window, start_offset=start, eocd=eocd),
            total_bytes=total,
            source_url=file_url,
            tail_start=start,
        )

    def fetch_prefix(
        self,
        file_url: str,
        entry: ZipEntry,
        n: int = RANGE_PREFIX_BYTES,
    ) -> bytes:
        """取条目压缩前缀并经 raw deflate 部分解压（截断流可解出前缀）。

        截断的 deflate 流在 `decompressobj.decompress` 下**不报错**，只解出可得部分——
        这正是「24 KB 判格式」可行的原因（调研 §9.2-5）。
        """
        data_offset = self._entry_data_offset(file_url, entry)
        take = min(entry.compress_size, max(0, n))
        if take <= 0:
            return b""

        def load() -> tuple[int, bytes]:
            return self._request_range(file_url, data_offset, data_offset + take - 1)

        start, payload = self._with_retries(f"prefix {file_url}:{entry.name}", load)
        if start != data_offset:
            raise PhiraApiError(
                f"{file_url}:{entry.name} 前缀窗口纠偏失败（实际起点 {start} != {data_offset}）",
            )
        return decompress_prefix(payload, compress_type=entry.compress_type)

    def _entry_data_offset(self, file_url: str, entry: ZipEntry) -> int:
        """读本地文件头，算出条目数据段起点。"""

        def load() -> tuple[int, bytes]:
            last = entry.header_offset + _LOCAL_STRUCT.size - 1
            return self._request_range(file_url, entry.header_offset, last)

        start, header = self._with_retries(f"local header {file_url}:{entry.name}", load)
        if start != entry.header_offset:
            raise PhiraApiError(f"{file_url}:{entry.name} 本地头窗口纠偏失败（起点 {start}）")
        _name_length, _extra_length, relative = parse_local_header(header)
        return entry.header_offset + relative

    # ── 选择性下载（第三段）────────────────────────────────────

    def download_entry(self, file_url: str, entry: ZipEntry, dest: Path) -> Path:
        """只下载**单个条目**（如谱面文件）并校验解压大小。

        带指数退避重试与 **Range 断点续传**：压缩字节写入 `<dest>.part`，
        续传时从其长度对应的偏移继续；完成后整体解压、校验 `file_size`，再原子改名。

        Raises:
            PhiraApiError: 传输失败 / 起点纠偏失败。
            PhiraZipError: 解压后大小与中央目录声明不符。
        """
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        data_offset = self._entry_data_offset(file_url, entry)
        offset = part.stat().st_size if part.exists() else 0
        if offset > entry.compress_size:
            logger.warning("%s 断点文件超出条目大小，重新下载", part)
            part.unlink()
            offset = 0

        mode = "ab" if offset else "wb"
        with part.open(mode) as handle:
            while offset < entry.compress_size:
                end = min(offset + DOWNLOAD_CHUNK_BYTES, entry.compress_size) - 1
                # 用 partial 绑定本次迭代的区间（避免闭包捕获循环变量，也便于类型推断）
                fetch_chunk = partial(
                    self._request_range,
                    file_url,
                    data_offset + offset,
                    data_offset + end,
                )
                start, chunk = self._with_retries(
                    f"download {file_url}:{entry.name}",
                    fetch_chunk,
                )
                if start != data_offset + offset:
                    raise PhiraApiError(
                        f"{file_url}:{entry.name} 分块起点 {start} != 期望 {data_offset + offset}",
                    )
                handle.write(chunk)
                offset += len(chunk)
                if not chunk:
                    raise PhiraApiError(f"{file_url}:{entry.name} 分块为空，中断以避免死循环")

        raw = part.read_bytes()
        content = decompress_prefix(raw, compress_type=entry.compress_type, partial=False)
        if len(content) != entry.file_size:
            raise PhiraZipError(
                f"{entry.name} 解压大小 {len(content)} != 中央目录声明 {entry.file_size}",
            )
        dest.write_bytes(content)
        part.unlink()
        logger.info("已下载 %s（%d B → %d B）", entry.name, entry.compress_size, entry.file_size)
        return dest


def decompress_prefix(payload: bytes, *, compress_type: int = 8, partial: bool = True) -> bytes:
    """raw deflate 解压（`-zlib.MAX_WBITS`）；`partial=True` 时容忍截断流。

    Args:
        payload: 压缩字节（可以是流的前缀）。
        compress_type: zip 压缩方法（0 = stored，8 = deflate）。
        partial: True = 截断流只解出可得部分；False = 期望完整流，截断即报错。

    Raises:
        PhiraZipError: 压缩方法不支持，或 `partial=False` 时流被截断/损坏。
    """
    if compress_type == 0:
        return payload
    if compress_type != 8:
        raise PhiraZipError(f"不支持的 zip 压缩方法 {compress_type}（仅 0=stored / 8=deflate）")
    decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
    try:
        content = decompressor.decompress(payload)
    except zlib.error as exc:
        raise PhiraZipError(f"deflate 解压失败：{exc}") from exc
    if not partial and not decompressor.eof:
        raise PhiraZipError(
            f"deflate 流不完整（解出 {len(content)} B 后仍有未消费输入或未到流尾）",
        )
    return content


def _http_error_message(response: httpx.Response, params: Mapping[str, Any]) -> str:
    """构造可读的 HTTP 错误信息（含 pageNum 上限提示）。"""
    message = f"HTTP {response.status_code} {response.request.url}"
    page_num = params.get("pageNum")
    over_limit = isinstance(page_num, int) and page_num > CHART_PAGE_SIZE_MAX
    if response.status_code == httpx.codes.BAD_REQUEST and over_limit:
        message += f"（pageNum={page_num} 超过实测上限 {CHART_PAGE_SIZE_MAX}，服务端返回 400）"
    return message


def iter_meta_rows(
    client: PhiraClient,
    *,
    page_size: int = CHART_PAGE_SIZE_MAX,
) -> Iterator[dict[str, Any]]:
    """枚举并把每条记录摊平成清单行（`PhiraChartMeta.to_row`）。"""
    yield from (meta.to_row() for meta in client.iter_chart_meta(page_size=page_size))


__all__ = [
    "CHART_FILE_FIELD",
    "CHART_LIST_DOCUMENTED_KEY",
    "CHART_LIST_ENDPOINT",
    "CHART_LIST_RESULTS_KEY",
    "CHART_MUSIC_FIELD",
    "CHART_PAGE_SIZE_MAX",
    "CHART_TOTAL_EXPECTED",
    "CHART_TYPE_ANY",
    "DOWNLOAD_CHUNK_BYTES",
    "PHIRA_API_BASE",
    "RANGE_PREFIX_BYTES",
    "REQUEST_INTERVAL_S",
    "ZIP_TAIL_BYTES",
    "Manifest",
    "ManifestError",
    "ManifestPurpose",
    "PhiraApiError",
    "PhiraChartMeta",
    "PhiraClient",
    "PhiraZipError",
    "Provenance",
    "ZipEntry",
    "ZipIndex",
    "decompress_prefix",
    "entries_in_window",
    "iter_meta_rows",
    "parse_local_header",
    "parse_zip_index",
    "provenance_for_api",
    "read_manifest",
    "sha1_file",
    "sha1_hex",
    "utc_now_iso",
    "write_manifest",
]
