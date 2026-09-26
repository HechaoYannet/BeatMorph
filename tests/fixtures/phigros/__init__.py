"""Phigros 微缩夹具包（plan 02 §3.7 / M2）。

三个夹具（全部**手工构造**，不从真实谱面裁剪）：

| 夹具 | 性质 | 用途 |
|------|------|------|
| `rpe_min.json` | 标准 RPE | 正样本：多线、四类 type、背面 note、事件多层 |
| `pec_masquerade.json` | **内容为 PEC 文本、后缀为 `.json`** | 负样本：陷阱 1（后缀不可信） |
| `pkg_min/` | 最小谱面包**部件** + 确定性打包函数 | 陷阱 2 / R1 / R2（必须读 `info.yml.chart`） |

**pkg_min 的构造方式**：`pkg_min/` 目录里放的是**文本部件**（`info.yml` + 谱面文件 +
干扰 json），zip 由本模块的 :func:`build_pkg_zip` 用标准库 `zipfile` **确定性**构造
（固定条目顺序 + 固定 `date_time`），测试把它物化到 `tmp_path`。这样夹具是纯文本、可 diff、
可 review，且不往仓库里塞二进制。详见同目录 `README.md`。

**合规**：全部夹具**不含音频与曲绘**（plan §3.7 硬性约束），且不含任何真实 Phira 用户内容。
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

FIXTURES_DIR: Path = Path(__file__).parent
PKG_MIN_DIR: Path = FIXTURES_DIR / "pkg_min"

#: 单文件夹具名。
RPE_MIN_NAME: str = "rpe_min.json"
PEC_MASQUERADE_NAME: str = "pec_masquerade.json"

#: pkg_min 部件名（`chart.json` 是**干扰项**：真实包 196/196 都不叫这个名字）。
INFO_YML_NAME: str = "info.yml"
PKG_CHART_NAME: str = "min_chart.json"
PKG_DECOY_NAME: str = "chart.json"
PKG_DISTRACTOR_NAME: str = "resources/effects.json"
#: 声明存在但**故意不入包**的音频/曲绘名（合规：夹具不含音频与曲绘）。
PKG_MUSIC_NAME: str = "min.ogg"
PKG_ILLUSTRATION_NAME: str = "min.png"

#: zip 条目顺序（固定 → 打包确定性）。
PKG_ENTRY_NAMES: tuple[str, ...] = (
    INFO_YML_NAME,
    PKG_CHART_NAME,
    PKG_DECOY_NAME,
    PKG_DISTRACTOR_NAME,
)

#: 固定时间戳（1980-01-01，zip 纪元起点）→ 打包确定性。
_PKG_FIXED_TIME: tuple[int, int, int, int, int, int] = (1980, 1, 1, 0, 0, 0)


def read_bytes(relative: str) -> bytes:
    """按相对路径读夹具字节。"""
    return (FIXTURES_DIR / relative).read_bytes()


def read_rpe_min() -> bytes:
    """标准 RPE 微缩夹具字节。"""
    return read_bytes(RPE_MIN_NAME)


def read_pec_masquerade() -> bytes:
    """伪装成 `.json` 的 PEC 文本夹具字节。"""
    return read_bytes(PEC_MASQUERADE_NAME)


def pkg_info_text() -> str:
    """pkg_min 的 `info.yml` 原文（负样本测试用它做文本替换）。"""
    return (PKG_MIN_DIR / INFO_YML_NAME).read_text(encoding="utf-8")


def build_pkg_bytes(*, info_text: str | None = None) -> bytes:
    """构造 pkg_min 的 zip 字节（确定性）。

    Args:
        info_text: 覆盖 `info.yml` 内容（负样本：把 `chart` 指向不存在的条目）。
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in PKG_ENTRY_NAMES:
            data = (
                (info_text if info_text is not None else pkg_info_text()).encode("utf-8")
                if name == INFO_YML_NAME
                else (PKG_MIN_DIR / name).read_bytes()
            )
            info = zipfile.ZipInfo(filename=name, date_time=_PKG_FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    return buffer.getvalue()


def build_pkg_zip(dest: Path, *, info_text: str | None = None) -> Path:
    """把 pkg_min 物化成 zip 文件，返回其路径。"""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(build_pkg_bytes(info_text=info_text))
    return dest


__all__ = [
    "FIXTURES_DIR",
    "INFO_YML_NAME",
    "PEC_MASQUERADE_NAME",
    "PKG_CHART_NAME",
    "PKG_DECOY_NAME",
    "PKG_DISTRACTOR_NAME",
    "PKG_ENTRY_NAMES",
    "PKG_ILLUSTRATION_NAME",
    "PKG_MIN_DIR",
    "PKG_MUSIC_NAME",
    "RPE_MIN_NAME",
    "build_pkg_bytes",
    "build_pkg_zip",
    "pkg_info_text",
    "read_bytes",
    "read_pec_masquerade",
    "read_rpe_min",
]
