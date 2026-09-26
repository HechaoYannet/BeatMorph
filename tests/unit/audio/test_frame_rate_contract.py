"""帧率契约门禁 G4 —— **不依赖 MERT 权重/torch，默认 CI 必跑**。

本文件存在的原因：`tests/unit/audio/test_mert.py::test_encode_shape_and_frame_rate`
带 `@pytest.mark.gpu` + 权重探测，`make test-fast` 永远跳过它，于是
"25Hz" 这个错误假设一路存活到万级数据规模才暴露。

见 `docs/POSTMORTEM-2026-08-05-frame-rate-misalignment.md`。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from beatmorph.core.contracts import (
    MERT_CONV_STRIDE_PRODUCT,
    MERT_DEFAULT_FEAT_DIM,
    MERT_FRAME_RATE_HZ,
    MERT_SAMPLE_RATE_HZ,
)

# 官方 config.json（ModelScope / HF: m-a-p/MERT-v1-330M）
_OFFICIAL_CONV_STRIDE = (5, 2, 2, 2, 2, 2, 2)
_OFFICIAL_CONV_KERNEL = (10, 3, 3, 3, 3, 2, 2)


def _conv_out_len(length: int, kernel: int, stride: int) -> int:
    """复刻 HF `Wav2Vec2Model._conv_out_length`（无 padding）。"""
    return (length - kernel) // stride + 1


def _stack_length(duration_s: float) -> int:
    """把 duration_s 秒音频过完 7 层卷积特征提取器，返回帧数。"""
    length = int(duration_s * MERT_SAMPLE_RATE_HZ)
    for kernel, stride in zip(_OFFICIAL_CONV_KERNEL, _OFFICIAL_CONV_STRIDE, strict=True):
        length = _conv_out_len(length, kernel, stride)
    return length


def test_contract_frame_rate_is_derived_not_assumed() -> None:
    """帧率是派生量：24000 / prod(conv_stride) = 75.0，且不得再是 25.0。"""
    assert MERT_SAMPLE_RATE_HZ == 24000
    assert MERT_CONV_STRIDE_PRODUCT == 320
    assert MERT_FRAME_RATE_HZ == MERT_SAMPLE_RATE_HZ / MERT_CONV_STRIDE_PRODUCT
    assert MERT_FRAME_RATE_HZ == 75.0
    assert MERT_FRAME_RATE_HZ != 25.0, "25Hz 是历史误值（差 3×，见 POSTMORTEM-2026-08-05）"


@pytest.mark.parametrize("duration_s", [1.0, 2.0, 5.0, 10.0])
def test_conv_stack_reproduces_contract_frame_rate(duration_s: float) -> None:
    """逐层卷积几何必须复现契约帧率——帧率是算出来的，不是设出来的。

    若不成立，说明 conv_stride/采样率被改动而契约未同步（正是本 bug 的成因）。
    """
    frames = _stack_length(duration_s)
    derived = frames / duration_s
    assert abs(derived - MERT_FRAME_RATE_HZ) <= 1.0, (
        f"{duration_s}s → {frames} 帧 = {derived:.1f}Hz，偏离契约 {MERT_FRAME_RATE_HZ}Hz"
    )


def test_5s_window_frame_count_is_pinned() -> None:
    """5s 窗（训练滑窗）钉死 374 帧；改动 conv 栈/采样率必须同步改这里。"""
    assert _stack_length(5.0) == 374


def test_feat_dim_is_1024() -> None:
    """MERT-v1-330M 各层 hidden_size 均为 1024（TRAINING_LOG Bug5 实测）。"""
    assert MERT_DEFAULT_FEAT_DIM == 1024


def test_stride_product_helper_matches_official_config() -> None:
    """`mert._conv_stride_product` 必须能从 config 推出 320，读不到时返回 None。"""
    pytest.importorskip("torch")
    from beatmorph.audio.encoder.mert import _conv_stride_product

    assert _conv_stride_product(SimpleNamespace(conv_stride=list(_OFFICIAL_CONV_STRIDE))) == 320
    assert _conv_stride_product({"conv_stride": list(_OFFICIAL_CONV_STRIDE)}) == 320
    assert _conv_stride_product(SimpleNamespace(conv_stride=())) is None
    assert _conv_stride_product(SimpleNamespace()) is None
    assert _conv_stride_product(None) is None
