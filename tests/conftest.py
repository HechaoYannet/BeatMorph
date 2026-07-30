"""pytest 共享夹具。"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def fixtures_dir() -> Path:
    """测试夹具目录（小体积合法样本）。"""
    return Path(__file__).parent / "fixtures"
