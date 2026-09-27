"""plan 07 §9-36：`optim.precision` 必须**真的被读**（此前只声明、无人使用）。

这条路径只有真机才走到，所以决策部分被抽成纯函数 `autocast_dtype` 以便在默认 CI 里断言。
"""

from __future__ import annotations

import pytest
import torch

from beatmorph.infra.train_loop import autocast_context, autocast_dtype

CPU = torch.device("cpu")
CUDA = torch.device("cuda")


def test_bf16_on_cuda_maps_to_bfloat16() -> None:
    assert autocast_dtype("bf16-mixed", CUDA) is torch.bfloat16
    assert autocast_dtype("bf16", CUDA) is torch.bfloat16


def test_fp32_and_cpu_never_enable_autocast() -> None:
    """CPU 保持 fp32：默认 CI（CPU）的数值必须逐位不变。"""
    assert autocast_dtype("fp32", CUDA) is None
    assert autocast_dtype("32-true", CUDA) is None
    assert autocast_dtype("32", CUDA) is None  # configs/smoke.yaml 用的就是这个写法
    assert autocast_dtype("bf16-mixed", CPU) is None


def test_null_context_paths_do_not_turn_autocast_on() -> None:
    with autocast_context("fp32", CUDA):
        assert not torch.is_autocast_enabled("cuda")
    with autocast_context("bf16-mixed", CPU):
        assert not torch.is_autocast_enabled("cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA 才能真的进入 autocast 上下文")
def test_bf16_mixed_on_cuda_actually_enables_bf16() -> None:
    with autocast_context("bf16-mixed", CUDA):
        assert torch.is_autocast_enabled("cuda")
        assert torch.get_autocast_dtype("cuda") is torch.bfloat16
    assert not torch.is_autocast_enabled("cuda")
