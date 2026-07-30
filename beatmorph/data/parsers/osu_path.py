"""osu! 谱面解析器（.osu → NoteEvent[]）。

奠基文档 §4.2 Step 1-2：解析 .osu → 清理乱码/非标准 → 自动计算统计量
（密度、段落边界、BPM）用于 Stage 1 伪标签。
"""

from __future__ import annotations

from pathlib import Path

from beatmorph.core.contracts import Chart


def parse_osu(path: Path) -> Chart:
    """解析单个 .osu 文件为 Chart IR。

    奠基文档 §4.2 Step1：清理乱码/非标准 Note。
    """
    raise NotImplementedError(".osu 解析未实现，见 docs/plans/08-data-pipeline.md")


def compute_section_stats(chart: Chart, section_bars: int = 4) -> Chart:
    """自动计算段落统计量（密度/能量/段落边界），回填到 chart.sections。

    奠基文档 §4.2 Step2：作为 Stage 1 自监督回归的伪标签来源。
    """
    raise NotImplementedError("段落统计量计算未实现，见 docs/plans/08-data-pipeline.md")
