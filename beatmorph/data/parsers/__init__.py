"""谱面解析层入口（plan 02 §3.3/§3.4）：**内容嗅探 → 格式分派 → IR → 质检**。

本包把三类静默陷阱收束成一个入口：

- **陷阱 1（后缀不可信）**：`sniff_format` 只看内容，签名不含文件名（:mod:`.sniff`）。
- **陷阱 2（文件名不可信）**：谱面条目由 :class:`~beatmorph.data.phira.package.ChartPackage`
  按 `info.yml.chart` 唯一定位。
- **陷阱 3（type 数字两套语义）**：分派**只由嗅探结果驱动**——RPE 走
  `note_type_from_rpe`（2 = Hold），官谱走 `note_type_from_official`（2 = Drag），
  且 v1 对官谱/PBC 只**记账**不解析（plan §偏离 2）。

`parse_chart_package` 的返回是 :class:`PackageAnalysis`：无论接受还是拒收，都留下
`format` / `sniff_evidence` / `quarantine` 记账，**不存在静默丢弃**。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from beatmorph.core.contracts import ChartFormat, ChartSource, PhigrosChart
from beatmorph.core.logging import get_logger
from beatmorph.data.parsers.rpejson import (
    RpeParseError,
    RpeSchemaError,
    beat_to_seconds,
    is_cover_masks,
    parse_rpejson,
    seconds_to_beat,
)
from beatmorph.data.parsers.sniff import sniff_format, sniff_format_with_evidence
from beatmorph.data.phira.client import sha1_hex
from beatmorph.data.phira.package import ChartInfo, ChartPackage
from beatmorph.data.qc import QcReport, QuarantineRecord, QuarantineStage, quality_check

logger = get_logger(__name__)

#: v1 主路径唯一可解析的格式（plan §偏离 2 / RFC-0029 §6）。
PRIMARY_FORMAT: ChartFormat = ChartFormat.RPE

__all__ = [
    "PRIMARY_FORMAT",
    "ChartFormat",
    "PackageAnalysis",
    "RpeParseError",
    "RpeSchemaError",
    "analyze_chart_bytes",
    "beat_to_seconds",
    "is_cover_masks",
    "parse_chart_package",
    "parse_rpejson",
    "seconds_to_beat",
    "sniff_format",
    "sniff_format_with_evidence",
]


@dataclass(frozen=True)
class PackageAnalysis:
    """一个谱面包的解析结论（接受 or 拒收，两者都带记账）。

    Attributes:
        chart_id: 谱面 id（可能未知 → None）。
        fmt: 内容嗅探出的格式。
        sniff_evidence: 嗅探依据（写入 `ChartSource.sniff_evidence`）。
        chart_sha1: 谱面条目内容 sha1（去重与追溯）。
        chart: 解析出的 IR；拒收时为 None。
        info: `info.yml` 的契约化视图（被拒收时仍保留，便于事后审计）。
        qc: 质检报告；未进入解析路径时为 None。
        quarantine: 隔离区记录；接受时为 None。
    """

    chart_id: int | None
    fmt: ChartFormat
    sniff_evidence: str = ""
    chart_sha1: str = ""
    chart: PhigrosChart | None = None
    info: ChartInfo | None = None
    qc: QcReport | None = None
    quarantine: QuarantineRecord | None = field(default=None)

    @property
    def accepted(self) -> bool:
        """是否可进入训练集（必须解析成功**且**质检通过）。"""
        return self.chart is not None and self.quarantine is None

    def to_dict(self) -> dict[str, Any]:
        """清单行摘要（不含 IR 本体）。"""
        return {
            "chart_id": self.chart_id,
            "format": str(self.fmt),
            "sniff_evidence": self.sniff_evidence,
            "chart_sha1": self.chart_sha1,
            "accepted": self.accepted,
            "qc": None if self.qc is None else self.qc.to_dict(),
            "quarantine": None if self.quarantine is None else self.quarantine.to_dict(),
        }


def _merge_info_meta(chart: PhigrosChart, info: ChartInfo | None) -> PhigrosChart:
    """用 `info.yml` 补齐 RPE `META` 里缺失/不可用的字段（**RPE 优先**）。

    `difficulty` 只能来自 `info.yml`（RPE META 没有定数字段），且必须 round 到 0.1
    （plan §偏离 3：**禁止** regex 解析 `level` 自由文本）。
    """
    if info is None:
        return chart
    updates: dict[str, Any] = {}
    if chart.meta.difficulty is None:
        updates["difficulty"] = info.difficulty
    if not chart.meta.name and info.name:
        updates["name"] = info.name
    if not chart.meta.composer and info.composer:
        updates["composer"] = info.composer
    if not chart.meta.charter and info.charter:
        updates["charter"] = info.charter
    if not chart.meta.level_text and info.level:
        updates["level_text"] = info.level
    if not updates:
        return chart
    return chart.model_copy(update={"meta": chart.meta.model_copy(update=updates)})


def analyze_chart_bytes(
    data: bytes,
    *,
    chart_id: int | None = None,
    info: ChartInfo | None = None,
    chart_file: str = "",
    music_file: str = "",
    audio_duration_s: float | None = None,
    source_format_hint: ChartFormat | None = None,
) -> PackageAnalysis:
    """嗅探 → 分派 → 解析 → 质检（**纯内容驱动**，不看任何文件名/后缀）。

    Args:
        data: 谱面条目字节。
        chart_id: 谱面 id。
        info: 所属包的 `info.yml`（用于补齐元数据；不影响格式判定）。
        chart_file / music_file: 留痕（**只来自 `info.yml`**）。
        audio_duration_s: 音频时长（秒），用于 `out_of_audio_window` 统计。
        source_format_hint: 仅用于**断言**嗅探结果（不参与判定）；单测用来验证
            「`info.yml.format` 不参与判型」。

    Returns:
        :class:`PackageAnalysis`；非 RPE 格式与 schema 违约都返回**带记账的拒收**。
    """
    fmt, evidence = sniff_format_with_evidence(data)
    if source_format_hint is not None and source_format_hint is not fmt:
        logger.debug(
            "嗅探结果 %s 与外部提示 %s 不一致（以内容为准，内容优先原则）",
            fmt,
            source_format_hint,
        )
    digest = sha1_hex(data)
    base_source = ChartSource(
        chart_id=chart_id,
        format=fmt,
        sniff_evidence=evidence,
        chart_file=chart_file,
        music_file=music_file,
        chart_sha1=digest,
    )

    if fmt is not PRIMARY_FORMAT:
        reason = (
            f"格式 {fmt} 不是 v1 主路径（RPEJSON）；只嗅探 + 记账，不解析"
            if fmt is not ChartFormat.UNKNOWN
            else "格式 UNKNOWN：未命中任何已知格式特征（PBC 结构未查证）"
        )
        logger.info("拒收 chart_id=%s：%s", chart_id, reason)
        return PackageAnalysis(
            chart_id=chart_id,
            fmt=fmt,
            sniff_evidence=evidence,
            chart_sha1=digest,
            info=info,
            quarantine=QuarantineRecord(
                chart_id=chart_id,
                stage=QuarantineStage.SNIFF,
                fmt=fmt,
                reasons=[reason],
                evidence=evidence,
            ),
        )

    try:
        chart = parse_rpejson(data, base_source)
    except RpeParseError as exc:
        return PackageAnalysis(
            chart_id=chart_id,
            fmt=fmt,
            sniff_evidence=evidence,
            chart_sha1=digest,
            info=info,
            quarantine=QuarantineRecord(
                chart_id=chart_id,
                stage=QuarantineStage.PARSE,
                fmt=fmt,
                reasons=[str(exc)],
                evidence=evidence,
            ),
        )

    chart = _merge_info_meta(chart, info)
    report = quality_check(chart, audio_duration_s, chart_id=chart_id, fmt=fmt)
    if not report.passed:
        return PackageAnalysis(
            chart_id=chart_id,
            fmt=fmt,
            sniff_evidence=evidence,
            chart_sha1=digest,
            chart=chart,
            info=info,
            qc=report,
            quarantine=QuarantineRecord(
                chart_id=chart_id,
                stage=QuarantineStage.QC,
                fmt=fmt,
                reasons=list(report.errors),
                evidence=evidence,
            ),
        )
    return PackageAnalysis(
        chart_id=chart_id,
        fmt=fmt,
        sniff_evidence=evidence,
        chart_sha1=digest,
        chart=chart,
        info=info,
        qc=report,
    )


def parse_chart_package(
    package: ChartPackage,
    *,
    chart_id: int | None = None,
    audio_duration_s: float | None = None,
) -> PackageAnalysis:
    """解析一个谱面包（**只读 `info.yml.chart` 指向的条目**）。

    Raises:
        ChartPackageError: 包结构错误（缺 `info.yml` / `chart` 指向不存在条目）。
    """
    data = package.chart_bytes()
    return analyze_chart_bytes(
        data,
        chart_id=chart_id,
        info=package.info,
        chart_file=package.chart_file,
        music_file=package.music_file,
        audio_duration_s=audio_duration_s,
    )
