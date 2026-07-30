"""StepMania / Etterna (.sm) 格式读写。

奠基文档 §4.1，作为 VSRG 变体补充数据源。
"""

from __future__ import annotations

from pathlib import Path

from beatmorph.core.contracts import Chart
from beatmorph.io.formats.base import ChartReader, ChartWriter


class SmWriter(ChartWriter):
    def suffix(self) -> str:
        return ".sm"

    def write(self, chart: Chart, path: Path) -> Path:
        raise NotImplementedError("SmWriter 未实现")


class SmReader(ChartReader):
    def read(self, path: Path) -> Chart:
        raise NotImplementedError("SmReader 未实现")
