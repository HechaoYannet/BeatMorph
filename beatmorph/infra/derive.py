"""派生常量的单一事实源、启动断言与字面量扫描（plan 07 §3.3 / M7.4）。

红线 7 的两个半边都在本模块落地：

1. **派生**：帧率 / x 桶宽 / 拍格宽等物理量只在 `beatmorph/core/contracts/` 定义一次，
   且写成派生式（如 `MERT_FRAME_RATE_HZ = MERT_SAMPLE_RATE_HZ / MERT_CONV_STRIDE_PRODUCT`）；
2. **断言**：启动时**打印取值**并断言派生式成立，另有一条**全仓扫描**把「第二处字面量」
   与「第二处定义」变成可失败的门禁。

来历见 docs/POSTMORTEM-2026-08-05-frame-rate-misalignment.md：25 Hz 误值在**三个互相独立
的位置**各写了一份，任何一处修好都不足以让系统正确。

扫描范围刻意只覆盖**生产路径**（`beatmorph/` 与 `configs/`）：测试与脚本里的数字是
「关于常量的断言」，而生产代码里的数字是「常量本身的第二份定义」——后者才是事故的来源。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from beatmorph.core.contracts import (
    MERT_CONV_STRIDE_PRODUCT,
    MERT_DEFAULT_FEAT_DIM,
    MERT_FRAME_RATE_HZ,
    MERT_SAMPLE_RATE_HZ,
    RPE_STAGE_WIDTH,
    RPE_X_GRID_BINS,
    RPE_X_GRID_DX,
    SUBDIVISIONS_PER_BEAT,
    TAU_GRID_DT,
)

__all__ = [
    "DERIVED",
    "DERIVED_NAMES",
    "SCAN_DIRS",
    "DerivedConstant",
    "LiteralHit",
    "assert_derived_identities",
    "derived_values",
    "format_derived_banner",
    "scan_derived_literals",
]


@dataclass(frozen=True, slots=True)
class DerivedConstant:
    """一个派生常量：名字、取值、派生式与**唯一**定义处的仓库相对路径。"""

    name: str
    value: float
    expression: str
    source: str


#: 派生量的**唯一事实源**清单（值在 import 期由 contracts 计算，本模块不重复推导）
DERIVED: tuple[DerivedConstant, ...] = (
    DerivedConstant(
        "MERT_SAMPLE_RATE_HZ",
        float(MERT_SAMPLE_RATE_HZ),
        "字面量 24000 —— 唯一允许写字面量的地方（来自 MERT 官方 config）",
        "beatmorph/core/contracts/tensors.py",
    ),
    DerivedConstant(
        "MERT_FRAME_RATE_HZ",
        float(MERT_FRAME_RATE_HZ),
        "MERT_SAMPLE_RATE_HZ / MERT_CONV_STRIDE_PRODUCT",
        "beatmorph/core/contracts/tensors.py",
    ),
    DerivedConstant(
        "RPE_X_GRID_DX",
        float(RPE_X_GRID_DX),
        "RPE_STAGE_WIDTH / RPE_X_GRID_BINS",
        "beatmorph/core/contracts/phigros.py",
    ),
    DerivedConstant(
        "TAU_GRID_DT",
        float(TAU_GRID_DT),
        "1 / SUBDIVISIONS_PER_BEAT",
        "beatmorph/core/contracts/phigros.py",
    ),
)

#: 派生常量名（扫描「第二处定义」时用）
DERIVED_NAMES: tuple[str, ...] = (
    "MERT_SAMPLE_RATE_HZ",
    "MERT_CONV_STRIDE_PRODUCT",
    "MERT_FRAME_RATE_HZ",
    "MERT_DEFAULT_FEAT_DIM",
    "RPE_X_GRID_DX",
    "TAU_GRID_DT",
)

#: 允许出现派生量定义/字面量的文件（**只允许这一处**）
DEFINITION_ALLOWLIST: tuple[str, ...] = (
    "beatmorph/core/contracts/tensors.py",
    "beatmorph/core/contracts/phigros.py",
)

#: 扫描目录（相对仓库根）
SCAN_DIRS: tuple[str, ...] = ("beatmorph", "configs")

#: 扫描后缀
SCAN_SUFFIXES: tuple[str, ...] = (".py", ".yaml", ".yml")

#: 帧率语境提示词：只有**带这些词**的行才检查字面量，避免把 r = 0.75 之类的超参误伤
_RATE_HINT = re.compile(r"(?i)\b(frame[ _]?rate|hop[ _]?rate|sample[ _]?rate|conv[ _]?stride|hz)\b")
#: 裸数字字面量（不匹配标识符/属性里的数字）
_NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])")
#: 派生量名 = 定义（含类型标注）
_DEFINITION = re.compile(r"^\s*(?:" + "|".join(DERIVED_NAMES) + r")\s*(?::[^=]+)?=(?!=)")


def derived_values() -> dict[str, float]:
    """派生量的取值快照（启动日志与产物落盘用）。"""
    return {item.name: float(item.value) for item in DERIVED}


def assert_derived_identities() -> None:
    """断言每个派生式**在运行期成立**（红线 7 的「断言」半边）。

    Raises:
        AssertionError: 任一等式不成立（例如有人把某个常量改成了字面量）。
    """
    checks: tuple[tuple[str, float, float], ...] = (
        (
            "MERT_FRAME_RATE_HZ == MERT_SAMPLE_RATE_HZ / MERT_CONV_STRIDE_PRODUCT",
            float(MERT_FRAME_RATE_HZ),
            float(MERT_SAMPLE_RATE_HZ) / float(MERT_CONV_STRIDE_PRODUCT),
        ),
        (
            "RPE_X_GRID_DX == RPE_STAGE_WIDTH / RPE_X_GRID_BINS",
            float(RPE_X_GRID_DX),
            float(RPE_STAGE_WIDTH) / float(RPE_X_GRID_BINS),
        ),
        (
            "TAU_GRID_DT == 1 / SUBDIVISIONS_PER_BEAT",
            float(TAU_GRID_DT),
            1.0 / float(SUBDIVISIONS_PER_BEAT),
        ),
    )
    failures = [text for text, actual, expected in checks if actual != expected]
    if failures:
        raise AssertionError("派生式不成立：" + "；".join(failures))
    if MERT_DEFAULT_FEAT_DIM <= 0:
        raise AssertionError(f"MERT_DEFAULT_FEAT_DIM 必须为正，得到 {MERT_DEFAULT_FEAT_DIM}")


def format_derived_banner() -> str:
    """启动时打印的派生量横幅（plan 07 §3.3 要求「打印取值」）。"""
    lines = ["派生常量（唯一事实源 = beatmorph/core/contracts）："]
    lines += [
        f"  {item.name} = {item.value:g}  <-  {item.expression}  [{item.source}]"
        for item in DERIVED
    ]
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class LiteralHit:
    """一处可疑的派生量字面量 / 重复定义。

    Attributes:
        path: 仓库相对路径（POSIX 分隔符）。
        line_number: 1 起的行号。
        line: 命中行原文（去掉行尾空白）。
        kind: `literal`（帧率语境里的魔法数字）或 `definition`（派生量的第二处定义）。
        detail: 人类可读的说明。
    """

    path: str
    line_number: int
    line: str
    kind: str
    detail: str


_DOCSTRING = '"""'


def _strip_comments(raw: str, suffix: str) -> str:
    """去掉行尾注释（YAML 的 {BT}#{BT}、Python 引号外的 {BT}#{BT}），避免把说明文字当代码。"""
    if suffix in (".yaml", ".yml"):
        position = raw.find(" #")
        return raw if position < 0 else raw[:position]
    result: list[str] = []
    quote: str | None = None
    for index, char in enumerate(raw):
        if quote is None and char in "\"'":
            quote = char
        elif quote is not None and char == quote:
            quote = None
        elif quote is None and char == "#":
            return raw[:index]
        result.append(char)
    return raw


def _code_lines(raw: str, suffix: str) -> list[tuple[int, str]]:
    """逐行给出「去掉注释与 docstring 之后」的代码。

    docstring 里的说明文字（如「24000/320 = 75 Hz」）**不是**第二份定义，
    若不剔除会让扫描器把文档写成违规项——一个只会制造噪声的门禁最终会被关掉。
    """
    lines: list[tuple[int, str]] = []
    in_doc = False
    for number, line in enumerate(raw.splitlines(), start=1):
        stripped = line.strip()
        if suffix == ".py":
            if in_doc:
                if _DOCSTRING in stripped:
                    in_doc = False
                continue
            if stripped.startswith(_DOCSTRING):
                if stripped.count(_DOCSTRING) == 1:
                    in_doc = True
                continue
        lines.append((number, _strip_comments(line, suffix).strip()))
    return lines


def _iter_scan_files(root: Path, include_dirs: Sequence[str]) -> list[Path]:
    files: list[Path] = []
    for name in include_dirs:
        base = root / name
        if not base.exists():
            continue
        files.extend(
            path
            for path in sorted(base.rglob("*"))
            if path.is_file() and path.suffix in SCAN_SUFFIXES
        )
    return files


def scan_derived_literals(
    root: Path,
    *,
    include_dirs: Sequence[str] = SCAN_DIRS,
) -> list[LiteralHit]:
    """扫描生产路径里的**派生量字面量**与**派生量重复定义**（M7.4 的 grep 断言）。

    Args:
        root: 仓库根目录。
        include_dirs: 相对 root 的扫描目录（默认 beatmorph/ + configs/）。

    Returns:
        命中列表（空列表 = 干净），顺序按 (path, line_number)。
    """
    root = Path(root)
    forbidden_values = {
        float(MERT_FRAME_RATE_HZ),
        float(MERT_SAMPLE_RATE_HZ),
        float(MERT_CONV_STRIDE_PRODUCT),
    }
    hits: list[LiteralHit] = []
    for path in _iter_scan_files(root, include_dirs):
        relative = path.relative_to(root).as_posix()
        if relative in DEFINITION_ALLOWLIST:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for number, raw in _code_lines(text, path.suffix):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if path.suffix == ".py" and _DEFINITION.match(raw):
                hits.append(
                    LiteralHit(
                        path=relative,
                        line_number=number,
                        line=line,
                        kind="definition",
                        detail="派生量的第二处定义；必须从 beatmorph.core.contracts 导入",
                    ),
                )
                continue
            if not _RATE_HINT.search(line):
                continue
            for match in _NUMBER.finditer(line):
                if float(match.group(1)) in forbidden_values:
                    hits.append(
                        LiteralHit(
                            path=relative,
                            line_number=number,
                            line=line,
                            kind="literal",
                            detail=f"帧率语境的字面量 {match.group(1)}；必须写成派生式",
                        ),
                    )
                    break
    return hits
