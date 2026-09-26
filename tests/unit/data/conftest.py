"""tests/unit/data 的共享夹具。"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.fixtures.phigros import build_pkg_zip, read_pec_masquerade, read_rpe_min


@pytest.fixture
def rpe_min_bytes() -> bytes:
    """标准 RPE 微缩夹具字节（tests/fixtures/phigros/rpe_min.json）。"""
    return read_rpe_min()


@pytest.fixture
def pec_masquerade_bytes() -> bytes:
    """伪装成 .json 的 PEC 文本夹具字节。"""
    return read_pec_masquerade()


@pytest.fixture
def pkg_zip_path(tmp_path: Path) -> Path:
    """物化后的 pkg_min zip（确定性构造）。"""
    return build_pkg_zip(tmp_path / "pkg_min.zip")
