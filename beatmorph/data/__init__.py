"""数据流水线模块（plan 02）：Phira 获取 + RPEJSON 解析 + 质检 + 特征离线提取。

各子模块的职责与对应里程碑：

| 子模块 | 职责 | 里程碑 |
|--------|------|--------|
| :mod:`beatmorph.data.phira.client` | `GET /chart` 分页枚举、Range 预筛、单条目下载、清单 provenance | M1 / M8 |
| :mod:`beatmorph.data.phira.package` | 谱面包定位（**只认 `info.yml.chart`**） | M4 |
| :mod:`beatmorph.data.parsers.sniff` | 格式内容嗅探（后缀不可信） | M3 |
| :mod:`beatmorph.data.parsers.rpejson` | RPEJSON → `PhigrosChart`（含格式层 beat↔秒） | M5 / M6 |
| :mod:`beatmorph.data.qc` | 三层质检 + 隔离区记账 | M7 |
| :mod:`beatmorph.data.pipeline.embed` | 特征缓存 + 配对切分 + 全库统计 | M9 / M10 |

**落盘纪律（M8 硬约束②：数据不得入库）**：本模块提供 :func:`is_ignored` /
:func:`assert_local_only`，用**纯 Python** 的 `.gitignore` 匹配器（不依赖 git 可执行文件）
在写入前检查目标路径确实被忽略。规范落盘布局见 :data:`DATA_LAYOUT`。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from beatmorph.data.dataset import (
    ChartPairDataset,
    DatasetConfig,
    DatasetIndexStats,
    DatasetManifestError,
    GridMismatchError,
    PairSample,
    collate_field_batch,
    load_pairs,
)
from beatmorph.data.parsers import (
    PRIMARY_FORMAT,
    PackageAnalysis,
    RpeParseError,
    RpeSchemaError,
    analyze_chart_bytes,
    beat_to_seconds,
    is_cover_masks,
    parse_chart_package,
    parse_rpejson,
    seconds_to_beat,
    sniff_format,
    sniff_format_with_evidence,
)
from beatmorph.data.phira.client import (
    PHIRA_API_BASE,
    Manifest,
    ManifestError,
    ManifestPurpose,
    PhiraApiError,
    PhiraChartMeta,
    PhiraClient,
    PhiraZipError,
    Provenance,
    ZipEntry,
    ZipIndex,
    provenance_for_api,
    read_manifest,
    sha1_file,
    sha1_hex,
    write_manifest,
)
from beatmorph.data.phira.package import (
    ChartInfo,
    ChartPackage,
    ChartPackageError,
    chart_dest_path,
    normalize_chart_filename,
    parse_info_yaml,
)
from beatmorph.data.pipeline.embed import (
    CACHE_META_KEYS,
    FEATURE_CACHE_META_KEYS,
    DatasetStats,
    FeatureCacheMeta,
    FeatureCacheMismatchError,
    GeneralizationPair,
    PairRow,
    PairSplits,
    SplitError,
    audio_cache_key,
    build_pairs,
    chart_row,
    dataset_stats,
    extract_features,
    feature_cache_paths,
    load_feature_cache,
)
from beatmorph.data.qc import (
    DistributionStats,
    QcReport,
    QuarantineRecord,
    QuarantineStage,
    distribution_stats,
    max_simultaneous_onsets,
    min_same_line_same_time_gap_x,
    quality_check,
    unit_contract_violations,
)
from beatmorph.data.tracks import line_tracks_at, line_tracks_tensor

__all__ = [
    "CACHE_META_KEYS",
    "DATA_LAYOUT",
    "FEATURE_CACHE_META_KEYS",
    "GITIGNORE_NAME",
    "PHIRA_API_BASE",
    "PRIMARY_FORMAT",
    "ChartInfo",
    "ChartPackage",
    "ChartPackageError",
    "ChartPairDataset",
    "DatasetConfig",
    "DatasetIndexStats",
    "DatasetManifestError",
    "DatasetStats",
    "DistributionStats",
    "FeatureCacheMeta",
    "FeatureCacheMismatchError",
    "GeneralizationPair",
    "GridMismatchError",
    "LocalStorageError",
    "Manifest",
    "ManifestError",
    "ManifestPurpose",
    "PackageAnalysis",
    "PairRow",
    "PairSample",
    "PairSplits",
    "PhiraApiError",
    "PhiraChartMeta",
    "PhiraClient",
    "PhiraZipError",
    "Provenance",
    "QcReport",
    "QuarantineRecord",
    "QuarantineStage",
    "RpeParseError",
    "RpeSchemaError",
    "SplitError",
    "ZipEntry",
    "ZipIndex",
    "analyze_chart_bytes",
    "assert_local_only",
    "audio_cache_key",
    "beat_to_seconds",
    "build_pairs",
    "chart_dest_path",
    "chart_row",
    "collate_field_batch",
    "dataset_stats",
    "distribution_stats",
    "extract_features",
    "feature_cache_paths",
    "is_cover_masks",
    "is_ignored",
    "line_tracks_at",
    "line_tracks_tensor",
    "load_feature_cache",
    "load_pairs",
    "max_simultaneous_onsets",
    "min_same_line_same_time_gap_x",
    "normalize_chart_filename",
    "parse_chart_package",
    "parse_info_yaml",
    "parse_rpejson",
    "provenance_for_api",
    "quality_check",
    "read_manifest",
    "seconds_to_beat",
    "sha1_file",
    "sha1_hex",
    "sniff_format",
    "sniff_format_with_evidence",
    "unit_contract_violations",
    "write_manifest",
]

# ══════════════════════════════════════════════════════════════
# §4 落盘布局与「不入库」守卫（M8 硬约束②）
# ══════════════════════════════════════════════════════════════

GITIGNORE_NAME: str = ".gitignore"

#: 规范落盘布局（plan 02 §4）。`<...>` 为占位符，`*` 为通配。
DATA_LAYOUT: dict[str, str] = {
    "raw": "data/raw/<chart_id>/<normalized-chart-file>",
    "audio": "data/audio/<audio-sha1><ext>",
    "features": "data/features/<audio-sha1>.npz（+ .meta.json）",
    "manifests": "data/manifests/*.jsonl",
}


class LocalStorageError(RuntimeError):
    """落盘目标不在 `.gitignore` 覆盖范围内（M8 硬约束②：谱面/音频不得入库）。"""


@dataclass(frozen=True)
class _GitignoreRule:
    """一条 `.gitignore` 规则（覆盖本仓用到的语法子集）。"""

    pattern: str
    negated: bool
    directory_only: bool
    anchored: bool
    regex: re.Pattern[str]


#: 本包相对仓库根的层级（`beatmorph/data/__init__.py` → 仓库根）。
_PACKAGE_DEPTH_FROM_ROOT: int = 2


def repo_root(start: Path | None = None) -> Path:
    """向上查找仓库根（以 `.gitignore` + `pyproject.toml` 同时存在为准）。

    默认从**本文件位置**向上找（不依赖 cwd），这样在任何工作目录下运行都指向仓库根。
    """
    current = (start or Path(__file__).resolve().parents[_PACKAGE_DEPTH_FROM_ROOT]).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / GITIGNORE_NAME).is_file() and (candidate / "pyproject.toml").is_file():
            return candidate
    return current


def _translate_pattern(pattern: str) -> str:
    """把 gitignore 通配语法翻译成正则（`*` / `**` / `?`；字符类不在此语法子集内）。"""
    out: list[str] = []
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if pattern.startswith("**/", index):
            # gitignore 的 `a/**/b` 匹配零层或多层目录 → `(?:.*/)?`。
            out.append("(?:.*/)?")
            index += 3
            continue
        if pattern.startswith("**", index):
            out.append(".*")
            index += 2
            continue
        if char == "*":
            out.append("[^/]*")
        elif char == "?":
            out.append("[^/]")
        elif char in ".+^$()|{}[]\\":
            out.append("\\" + char)
        else:
            out.append(char)
        index += 1
    return "".join(out)


def load_gitignore_rules(path: Path) -> list[_GitignoreRule]:
    """解析 `.gitignore`（注释 / 空行 / 取反 / 目录模式 / 锚定模式）。"""
    rules: list[_GitignoreRule] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        negated = line.startswith("!")
        if negated:
            line = line[1:].strip()
        directory_only = line.endswith("/")
        body = line.rstrip("/")
        anchored = "/" in body
        if body.startswith("/"):
            body = body[1:]
        rules.append(
            _GitignoreRule(
                pattern=line,
                negated=negated,
                directory_only=directory_only,
                anchored=anchored,
                regex=re.compile(_translate_pattern(body)),
            ),
        )
    return rules


def _rule_matches(rule: _GitignoreRule, relative: str) -> bool:
    """判断一条规则是否命中相对路径（含其祖先目录链——忽略目录即忽略其中一切）。"""
    parts = relative.split("/")
    prefixes = ["/".join(parts[: index + 1]) for index in range(len(parts))]
    if rule.directory_only:
        prefixes = prefixes[:-1]
    for prefix in prefixes:
        if rule.anchored:
            if rule.regex.fullmatch(prefix):
                return True
        # 不带斜杠的模式匹配任意层级（等价于匹配路径任一层级的基名）。
        elif rule.regex.fullmatch(prefix.split("/")[-1]):
            return True
    return False


def is_ignored(path: Path, *, root: Path | None = None) -> bool:
    """路径是否被仓库 `.gitignore` 忽略（取反规则按「最后命中者胜」）。"""
    base = root or repo_root()
    target = Path(path)
    absolute = target if target.is_absolute() else (base / target)
    try:
        relative = absolute.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return False
    ignored = False
    for rule in load_gitignore_rules(base / GITIGNORE_NAME):
        if _rule_matches(rule, relative):
            ignored = not rule.negated
    return ignored


def assert_local_only(
    path: Path,
    *,
    root: Path | None = None,
    allow_unignored: bool = False,
) -> None:
    """断言落盘目标确实被 `.gitignore` 覆盖（M8 硬约束②）。

    Raises:
        LocalStorageError: 目标未被忽略且未显式放行。
    """
    if allow_unignored:
        return
    if not is_ignored(path, root=root):
        raise LocalStorageError(
            f"{path} 未被 {GITIGNORE_NAME} 覆盖：谱面/音频/特征不得入库（CLAUDE.md 红线 5 / "
            f"plan 02 M8）。若确需落盘在该位置，须先让 infra-agent 在 {GITIGNORE_NAME} 加规则，"
            "或显式传 allow_unignored=True 并说明理由",
        )
