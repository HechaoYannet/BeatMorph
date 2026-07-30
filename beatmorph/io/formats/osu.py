"""osu!mania (.osu) 格式读写。

奠基文档 §1.1 优先支持 4K VSRG，输出 .osu。
本文件实现 ChartWriter/ChartReader 骨架。
"""

from __future__ import annotations

from pathlib import Path

from beatmorph.core.contracts import Chart
from beatmorph.io.formats.base import ChartReader, ChartWriter


class OsuManiaWriter(ChartWriter):
    """写出 osu!mania 谱面。"""

    def suffix(self) -> str:
        return ".osu"

    def write(self, chart: Chart, path: Path) -> Path:
        raise NotImplementedError("OsuManiaWriter 未实现，见 docs/plans/07-decoder-postprocess.md")


class OsuManiaReader(ChartReader):
    """解析 .osu 为 Chart IR（数据预处理流水线使用）。"""

    def read(self, path: Path) -> Chart:
        raise NotImplementedError("OsuManiaReader 未实现，见 docs/plans/08-data-pipeline.md")
