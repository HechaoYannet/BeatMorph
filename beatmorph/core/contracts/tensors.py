"""张量级数据契约 —— 音频侧的跨模块常量与形状约定。

Plan 00 之后，本轮范式的物理常量分两处，且**都只允许派生、不允许硬编码**
（CLAUDE.md 红线 7）：

- **音频帧轴**：本文件（MERT-v1-330M 的采样率与卷积步长累乘）；
- **谱面 / 场网格**：beatmorph.core.contracts.phigros 与
  beatmorph.core.contracts.field。

帧率是**派生量**，不是可自由设定的超参：改采样率或主干必须同步改这两个基数。
来历见 docs/POSTMORTEM-2026-08-05-frame-rate-misalignment.md。
"""

from __future__ import annotations

from dataclasses import dataclass

# ── MERT-v1-330M 音频侧常量 ──
# 来源：ModelScope/HF 官方仓 config.json + preprocessor_config.json
#   conv_kernel = [10, 3, 3, 3, 3, 2, 2]
#   conv_stride = [ 5, 2, 2, 2, 2, 2, 2]  -> 累乘 320
MERT_SAMPLE_RATE_HZ: int = 24000  # MERT 特征提取器要求 24kHz 单声道
MERT_CONV_STRIDE_PRODUCT: int = 320  # prod(conv_stride)
MERT_FRAME_RATE_HZ: float = MERT_SAMPLE_RATE_HZ / MERT_CONV_STRIDE_PRODUCT
MERT_DEFAULT_FEAT_DIM: int = 1024  # MERT-v1-330M hidden_size（所有层均为 1024）


@dataclass(frozen=True, slots=True)
class AudioEmbedding:
    """Stage 0 音频编码器输出。

    einops: (batch, time_seq, feat)

    Attributes:
        feat: 特征维，MERT-v1-330M 各层均为 MERT_DEFAULT_FEAT_DIM（1024）。
        time_seq: 帧序列长度 = round(duration_s * hop_rate)。
        hop_rate: 帧率 Hz，= MERT_SAMPLE_RATE_HZ / MERT_CONV_STRIDE_PRODUCT
            （**派生量**，勿硬编码）。
    """

    feat: int = MERT_DEFAULT_FEAT_DIM
    time_seq: int = 0  # 运行期确定
    hop_rate: float = MERT_FRAME_RATE_HZ
