"""谱面包访问：**只按 `info.yml.chart` 定位谱面文件**（plan 02 §3.2 R1/R2/R4 / M4）。

硬规则（每一条都有实测反面教材，见 plan 02 §3.2 与 phira-dataset-survey.md §4.2）：

- **R1**：包内 `.json` 解压总量可达数百 MB（特效资源），**不得「取最大的 json」**
  （实测 id 45756：json 总量 294 MB，而谱面文件仅 3.25 MB）。
- **R2**：**不得**依赖默认名 `chart.json`（实测 196/196 张的谱面文件都不叫这个名字，
  形如 `1817439042209534.json`、`AT15.json`）。
- **R3**：**不得**依赖 `info.yml.format`（实测恒为 `null`）→ 格式只能按内容嗅探。
- **R4**：落盘**不得**沿用原始文件名（含全角字符，如 `＃53682.json`）→ 规范化为
  `<chart_id>/<normalized>`（:func:`chart_dest_path`）。

本模块只做「定位与读取」，不做解析（解析在 :mod:`beatmorph.data.parsers.rpejson`），
也不看后缀判定格式（嗅探在 :mod:`beatmorph.data.parsers.sniff`）。
"""

from __future__ import annotations

import io
import re
import unicodedata
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from beatmorph.core.logging import get_logger
from beatmorph.data.phira.client import (
    CHART_FILE_FIELD,
    CHART_MUSIC_FIELD,
    ZipEntry,
    ZipIndex,
)

logger = get_logger(__name__)

#: 包元数据文件名（Phira ChartInfo，YAML）。
INFO_YML_NAME: str = "info.yml"

#: 旧式文本元信息（PhiEditer 遗留，实测 191/196 张存在）——**只记账不使用**。
INFO_TXT_NAME: str = "info.txt"

#: 条目读取器：给定包内条目名，返回其解压字节。
EntryReader = Callable[[str], bytes]


class ChartPackageError(ValueError):
    """谱面包结构错误（缺 info.yml / `info.yml.chart` 指向不存在的条目 …）。"""


class ChartInfo(BaseModel):
    """`info.yml`（Phira ChartInfo）的契约化视图。

    字段名与 YAML 的 camelCase 一致（`aspectRatio` / `lineLength` / `previewStart`）；
    未收录的字段一律保留（`extra="allow"`），避免丢信息。

    ⚠️ `format` 字段**实测恒为 null**，本类只记录不使用（R3）。
    """

    model_config = ConfigDict(frozen=True, extra="allow", populate_by_name=True)

    name: str = ""
    #: 定数（f32）；实测存在浮点误差，比较前 round 到 0.1（plan 02 §偏离 3）
    difficulty: float | None = None
    #: 自由文本等级（`"AT  Lv.16"` / `"sweet"` / `"酔い"` …）——**不得** regex 解析
    level: str = ""
    charter: str = ""
    composer: str = ""
    illustrator: str = ""
    #: 谱面文件名（**定位谱面文件的唯一来源**，R2）
    chart: str = CHART_FILE_FIELD
    #: 实测恒为 null；只记录
    format: Any = None
    #: 音频文件名的唯一来源
    music: str = CHART_MUSIC_FIELD
    illustration: str = ""
    offset: float = 0.0
    aspect_ratio: float | None = Field(default=None, alias="aspectRatio")
    line_length: float | None = Field(default=None, alias="lineLength")
    tags: list[str] = Field(default_factory=list)
    preview_start: float | None = Field(default=None, alias="previewStart")
    preview_end: float | None = Field(default=None, alias="previewEnd")
    intro: str = ""
    tip: str = ""

    @property
    def difficulty_round(self) -> float | None:
        """数值分层口径（plan 02 §偏离 3）：round 到 0.1。"""
        return None if self.difficulty is None else round(self.difficulty, 1)

    @property
    def song_key(self) -> str:
        """同曲近似键 = `name | composer`（info.yml 无全局唯一歌曲 ID，调研 §5.3-7）。"""
        return f"{self.name}|{self.composer}"


def parse_info_yaml(data: bytes) -> ChartInfo:
    """解析 `info.yml` 字节。

    Raises:
        ChartPackageError: YAML 非法、顶层不是映射、或字段类型不符。
    """
    try:
        raw = yaml.safe_load(data.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ChartPackageError(f"{INFO_YML_NAME} 不是 UTF-8 文本：{exc}") from exc
    except yaml.YAMLError as exc:
        raise ChartPackageError(f"{INFO_YML_NAME} YAML 解析失败：{exc}") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ChartPackageError(f"{INFO_YML_NAME} 顶层必须是映射，得到 {type(raw).__name__}")
    try:
        return ChartInfo.model_validate(raw)
    except ValidationError as exc:
        raise ChartPackageError(f"{INFO_YML_NAME} 字段校验失败：{exc}") from exc


def normalize_chart_filename(name: str) -> str:
    """把包内文件名规范化为安全落盘名（R4）。

    保留扩展名，其余非 `[0-9A-Za-z._-]` 的字符（含全角字符、空格、CJK）替换为 `_`；
    结果为空时回落 `chart`。
    """
    path = Path(name)
    suffix = path.suffix.lower()
    # NFKC 先把全角/兼容字符折叠成 ASCII 近似（`１２３` → `123`、`＃` → `#`），
    # 再把剩余的非安全字符整体折叠成一个 `_`，最后去掉首尾分隔符。
    folded = unicodedata.normalize("NFKC", path.stem)
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", folded).strip("._-")
    return f"{safe or 'chart'}{suffix}"


def chart_dest_path(root: Path, chart_id: int, chart_file: str) -> Path:
    """规范落盘路径 `<root>/<chart_id>/<normalized>`（R4，不沿用原始文件名）。"""
    return root / str(chart_id) / normalize_chart_filename(chart_file)


@dataclass
class ChartPackage:
    """一个谱面包 = zip：`info.yml` + 谱面文件 + 音频 + 曲绘（+ 可选 extra/贴图/着色器）。

    Attributes:
        entries: 包内条目表（名 → :class:`ZipEntry`）。
        info: `info.yml` 的契约化视图。
        chart_file: **只来自** `info.yml[CHART_FILE_FIELD]`；构造时校验其存在于 `entries`。
        music_file: **只来自** `info.yml[CHART_MUSIC_FIELD]`。
        missing_music: `music` 声明了但包内不存在（**只告警不报错**：音频下载是后续阶段，
            且合规夹具按 §3.7 要求不含音频）。
        source: 留痕（本地路径或 CDN URL）。
    """

    entries: dict[str, ZipEntry]
    info: ChartInfo
    chart_file: str
    music_file: str
    source: str = ""
    zip_path: Path | None = None
    #: 内存 zip 字节（`from_zip_bytes` 路径；与 `zip_path` 二选一即可读）
    raw_bytes: bytes | None = field(default=None, repr=False)
    entry_reader: EntryReader | None = field(default=None, repr=False)
    missing_music: bool = False

    def __post_init__(self) -> None:
        """R1/R2 的落点：`info.yml.chart` 必须存在，**不得**回退猜测。"""
        if not self.chart_file:
            raise ChartPackageError(
                f"{INFO_YML_NAME} 缺少 {CHART_FILE_FIELD!r} 字段，无法定位谱面文件"
                "（不得按后缀、大小或默认名 chart.json 猜测）",
            )
        if self.chart_file not in self.entries:
            raise ChartPackageError(
                f"{INFO_YML_NAME}.{CHART_FILE_FIELD}={self.chart_file!r} 在包内不存在；"
                f"包内条目：{sorted(self.entries)}。"
                "**禁止**回退到「取最大的 json」或「取名为 chart.json 的文件」（R1/R2）",
            )
        if self.music_file and self.music_file not in self.entries:
            self.missing_music = True
            logger.warning(
                "%s.music=%r 在包内不存在（音频缺失；合规夹具不含音频）",
                INFO_YML_NAME,
                self.music_file,
            )

    # ── 构造 ──────────────────────────────────────────────────

    @classmethod
    def open(cls, path: Path) -> ChartPackage:
        """从本地 zip 文件打开谱面包。"""
        with zipfile.ZipFile(path) as archive:
            entries: dict[str, ZipEntry] = {}
            for item in archive.infolist():
                if item.is_dir():
                    continue
                entries[item.filename] = ZipEntry(
                    name=item.filename,
                    compress_size=item.compress_size,
                    file_size=item.file_size,
                    header_offset=item.header_offset,
                    compress_type=item.compress_type,
                )
            info_name = _locate_info_entry(entries)
            info = parse_info_yaml(archive.read(info_name))
        return cls._build(entries, info, source=str(path), zip_path=path)

    @classmethod
    def from_zip_bytes(cls, data: bytes, *, source: str = "") -> ChartPackage:
        """从内存中的 zip 字节打开谱面包（单测 / 中央目录预筛路径）。"""
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = {
                item.filename: ZipEntry(
                    name=item.filename,
                    compress_size=item.compress_size,
                    file_size=item.file_size,
                    header_offset=item.header_offset,
                    compress_type=item.compress_type,
                )
                for item in archive.infolist()
                if not item.is_dir()
            }
            info_name = _locate_info_entry(entries)
            info = parse_info_yaml(archive.read(info_name))
        package = cls._build(entries, info, source=source)
        package.raw_bytes = data
        return package

    @classmethod
    def from_index(cls, index: ZipIndex, reader: EntryReader) -> ChartPackage:
        """从**中央目录预筛结果**打开（配合 `PhiraClient.fetch_prefix/download_entry`）。"""
        entries = index.as_dict()
        info_name = _locate_info_entry(entries)
        info = parse_info_yaml(reader(info_name))
        package = cls._build(entries, info, source=index.source_url)
        package.entry_reader = reader
        return package

    @classmethod
    def _build(
        cls,
        entries: dict[str, ZipEntry],
        info: ChartInfo,
        *,
        source: str = "",
        zip_path: Path | None = None,
    ) -> ChartPackage:
        return cls(
            entries=entries,
            info=info,
            chart_file=str(info.chart or "").strip(),
            music_file=str(info.music or "").strip(),
            source=source,
            zip_path=zip_path,
        )

    # ── 读取 ──────────────────────────────────────────────────

    def chart_bytes(self) -> bytes:
        """谱面文件字节（由 :attr:`chart_file` 唯一定位）。"""
        return self._read(self.chart_file)

    def music_bytes(self) -> bytes:
        """音频字节；包内没有该条目时抛 :class:`ChartPackageError`。"""
        if not self.music_file or self.music_file not in self.entries:
            raise ChartPackageError(f"包内没有音频条目 {self.music_file!r}")
        return self._read(self.music_file)

    def _read(self, name: str) -> bytes:
        if self.entry_reader is not None:
            return self.entry_reader(name)
        if self.zip_path is not None:
            with zipfile.ZipFile(self.zip_path) as archive:
                return archive.read(name)
        if self.raw_bytes is not None:
            with zipfile.ZipFile(io.BytesIO(self.raw_bytes)) as archive:
                return archive.read(name)
        raise ChartPackageError(
            f"谱面包没有可用的读取器（source={self.source!r}）；"
            "请用 open / from_zip_bytes / from_index 构造",
        )

    # ── 统计 ──────────────────────────────────────────────────

    def total_uncompressed_bytes(self, suffix: str | None = None) -> int:
        """包内条目解压总字节（可按后缀过滤，用于 R1 的反面教材复现）。"""
        return sum(
            entry.file_size
            for entry in self.entries.values()
            if suffix is None or entry.name.lower().endswith(suffix)
        )

    def largest_json_entry(self) -> str | None:
        """解压体积最大的 `.json` 条目名（**仅供审计对照**：它常常不是谱面文件）。"""
        candidates = [e for e in self.entries.values() if e.name.lower().endswith(".json")]
        if not candidates:
            return None
        return max(candidates, key=lambda e: e.file_size).name


def _locate_info_entry(entries: Iterable[str]) -> str:
    """定位 `info.yml`：优先根级精确名，其次**唯一**的同名条目；否则报错。"""
    names = list(entries)
    if INFO_YML_NAME in names:
        return INFO_YML_NAME
    matches = [name for name in names if Path(name).name == INFO_YML_NAME]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ChartPackageError(f"包内没有 {INFO_YML_NAME}（条目：{sorted(names)}）")
    raise ChartPackageError(f"包内有多个 {INFO_YML_NAME}：{sorted(matches)}（无法确定用哪一个）")


__all__ = [
    "INFO_TXT_NAME",
    "INFO_YML_NAME",
    "ChartInfo",
    "ChartPackage",
    "ChartPackageError",
    "EntryReader",
    "chart_dest_path",
    "normalize_chart_filename",
    "parse_info_yaml",
]
