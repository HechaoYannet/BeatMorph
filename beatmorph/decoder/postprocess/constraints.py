"""Stage 3 & 4：解码与后处理。

Stage 3：VQ-VAE Decoder 将 Token 映射回精确 (time, lane, type, duration)。
Stage 4：规则后处理引擎 + 格式 Writer。

奠基文档 §3.7。关键原则：AI 负责创意生成，规则引擎负责物理安全与合法性，
两者解耦，保证 100% 可玩性。规则仅做「红线」校验，不改变键型排列逻辑（R-5）。

详细计划：docs/plans/07-decoder-postprocess.md
"""

from __future__ import annotations

from beatmorph.core.contracts import Chart, GameMode


class PostProcessor:
    """物理约束与合法性校验引擎。

    红线规则（奠基文档 §3.7）：
    - 4K 标准：单帧最大同时按键 ≤ 2
    - 同手最小间隔 ≥ 70ms
    - 禁止越界（lane ∈ [0, lane_count)）
    - 长条 duration 合法性
    """

    # 4K 物理常量
    MAX_SIMULTANEOUS_4K: int = 2
    MIN_SAME_HAND_GAP_MS: int = 70

    def __init__(self, mode: GameMode = GameMode.MANIA_4K) -> None:
        self.mode = mode

    def apply(self, chart: Chart) -> Chart:
        """对解码后的谱面做红线校验与修正。

        只修正「物理不可达」的 Note，不改动 AI 的键型语义（R-5）。
        """
        raise NotImplementedError("后处理引擎尚未实现，见 docs/plans/07-decoder-postprocess.md")

    def validate(self, chart: Chart) -> list[str]:
        """返回违规项描述列表；空列表表示完全合法可玩。"""
        ...
