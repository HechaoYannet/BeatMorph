"""谱面格式读写接口（IR ↔ 各游戏格式）。

奠基文档 §3.7。IR(JSON) 作为内部规范表示，Writer 负责转 .osu / .sm / .ma2。
新增模式仅需实现 Writer，遵循 §1.1「模式扩展视为独立适配工程」。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from beatmorph.core.contracts import Chart


class ChartWriter(ABC):
    """谱面格式写入器抽象基类。"""

    @abstractmethod
    def write(self, chart: Chart, path: Path) -> Path:
        """将 Chart 写为目标格式文件。"""
        ...

    @abstractmethod
    def suffix(self) -> str:
        """该 Writer 产出的文件后缀（含点）。"""
        ...


class ChartReader(ABC):
    """谱面格式读取器抽象基类（用于数据预处理：解析现成谱面）。"""

    @abstractmethod
    def read(self, path: Path) -> Chart:
        """从目标格式文件解析为 Chart IR。"""
        ...
