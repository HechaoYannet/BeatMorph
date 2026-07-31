"""pytest 共享夹具。"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    """测试夹具目录（小体积合法样本）。"""
    return Path(__file__).parent / "fixtures"


def require_cuda() -> None:
    """无 CUDA 时 skip（用于 ``@pytest.mark.gpu`` 测试的守卫）。

    在测试体内调用：``require_cuda()``，或用 fixture 形式 ``require_gpu``。
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA")


@pytest.fixture
def require_gpu() -> None:
    """无 CUDA 的 CI 跳过带本 fixture 的测试。"""
    require_cuda()
