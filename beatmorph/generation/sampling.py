"""迭代并行解码（MaskGIT / Mask-Predict 式；Plan 04 §4.4，里程碑 M6）。

流程：全部遮盖 -> 前向预测全场 -> 按**置信度**保留最确定的一部分 -> 其余重新遮盖
-> 重复。契约（plan 04 §4.4）：

1. **`steps >= 2`**：MaskGIT 原文明确指出「一次推断全部」与训练分布不一致
   （*we find this challenging due to inconsistency with the training task*），
   因此一步到位**不是**允许的配置，本模块直接拒绝 `steps < 2`。
2. **保留比例是单调 schedule `gamma(t/T)`**，步数与 schedule 都写进 `diagnostics`。
3. **已保留的位置在后续步中以「未遮盖」的形式作为条件输入**（mask 通道同步翻转）。
4. **连续场的置信度需要重新定义**（文献无对应做法，plan 04 §9-4）：
   本模块实现三种候选并允许消融——(ii) 事件级期望计数（默认）、
   (i) 峰值显著性（局部对比度）、(iii) 多次前向的方差。

采样一律在 `torch.no_grad()` 下进行（推理路径），并且对同一模型/输入**确定性**：
选择用的是 stable 排序，没有隐式随机数。
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from beatmorph.generation.batch import FieldBatch, FieldOutput
from beatmorph.generation.model import MaskedFieldModel

#: 连续场置信度的三种候选定义（plan 04 §4.4 / §9-4）
Confidence = Literal["event_count", "peak_salience", "variance"]
#: 保留比例 schedule
Schedule = Literal["linear", "cosine"]
#: 被保留位置的写入方式：二值「已确证」或直接写入当前 lambda
StateFill = Literal["binary", "lambda"]

#: 参照步数（MaskGIT 原文 8 步完成 256 token；plan 04 §4.4）
DEFAULT_STEPS: int = 8
#: 最小步数（一步到位不可行，见模块 docstring 契约 1）
MIN_STEPS: int = 2


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """迭代并行解码的超参（全部显式声明，写进日志）。

    Attributes:
        steps: 迭代步数，**必须 >= 2**。
        schedule: 保留比例 schedule（"linear" / "cosine"，两者都单调不减、末步为 1）。
        confidence: 置信度口径（三候选之一）。
        state_fill: 保留位置写入 "binary"（1.0）或 "lambda"（当前强度值）。
        variance_samples: `confidence="variance"` 时的随机前向次数（>= 2）。
        salience_window: 峰值显著性的邻域半径（tau/x 格）。
    """

    steps: int = DEFAULT_STEPS
    schedule: Schedule = "cosine"
    confidence: Confidence = "event_count"
    state_fill: StateFill = "binary"
    variance_samples: int = 4
    salience_window: int = 1

    def __post_init__(self) -> None:
        if self.steps < MIN_STEPS:
            raise ValueError(
                f"steps 必须 >= {MIN_STEPS}：一步到位与训练分布不一致（MaskGIT 原文，"
                "plan 04 §4.4 契约），禁止配置为 1",
            )
        if self.confidence == "variance" and self.variance_samples < 2:
            raise ValueError("variance 置信度至少需要 2 次前向")
        if self.salience_window < 0:
            raise ValueError("salience_window 必须 >= 0")


def reveal_schedule(steps: int, schedule: Schedule = "cosine") -> tuple[float, ...]:
    """返回累计保留比例 gamma(1..steps)，**单调不减**且末项 = 1。"""
    if steps < MIN_STEPS:
        raise ValueError(f"steps 必须 >= {MIN_STEPS}，得到 {steps}")
    fractions: list[float] = []
    for step in range(1, steps + 1):
        progress = step / steps
        if schedule == "linear":
            fractions.append(progress)
        elif schedule == "cosine":
            fractions.append(1.0 - math.cos(0.5 * math.pi * progress))
        else:  # pragma: no cover - 由 Literal 约束
            raise ValueError(f"未知的 schedule：{schedule!r}")
    fractions[-1] = 1.0
    return tuple(fractions)


def assert_schedule_monotone(fractions: tuple[float, ...]) -> None:
    """契约：保留比例必须单调不减且末项为 1（plan 04 §4.4 终止条件）。"""
    if not fractions:
        raise AssertionError("schedule 不得为空")
    if any(later < earlier for earlier, later in itertools.pairwise(fractions)):
        raise AssertionError(f"schedule 必须单调不减：{fractions}")
    if fractions[-1] != 1.0:
        raise AssertionError(f"schedule 末项必须为 1：{fractions}")


def expected_counts(out: FieldOutput, batch: FieldBatch) -> Tensor:
    """每个格子的期望计数 lambda_j * dV_j（事件级置信度的基础量）。"""
    volumes = torch.as_tensor(
        batch.grid.cell_volumes(),
        dtype=out.lam.dtype,
        device=out.lam.device,
    ).reshape(1, 1, -1, 1, 1, 1)
    return out.lam * volumes


def local_contrast(values: Tensor, *, window: int) -> Tensor:
    """局部对比度：values - 邻域均值（沿 tau 与 x 两个轴，(S, C) 各自独立）。

    plan 04 §4.4 候选 (i)「峰值显著性」。window = 0 时退化为全 0（无对比度可言）。
    """
    if window <= 0:
        return torch.zeros_like(values)
    shape = values.shape
    batch_size, n_lines, t_bins, x_bins, sides, channels = shape
    flat = values.reshape(batch_size * n_lines * sides * channels, 1, t_bins, x_bins)
    kernel = 2 * int(window) + 1
    pooled = F.avg_pool2d(flat, kernel_size=kernel, stride=1, padding=int(window))
    return values - pooled.reshape(shape)


def _variance_confidence(
    model: MaskedFieldModel,
    batch: FieldBatch,
    state: Tensor,
    occlusion: Tensor,
    *,
    samples: int,
) -> Tensor:
    """候选 (iii)：dropout 多次前向的方差（取负号 -> 越稳定越优先）。

    `dropout == 0` 时方差恒为 0（确定性前向），此时所有格子置信度相同——
    这是**如实的行为**而不是补偿：该口径只在开启 dropout 时有意义。
    """
    was_training = model.training
    model.train()
    try:
        stacked = torch.stack(
            [model.forward_state(batch, state, occlusion).lam for _ in range(int(samples))],
        )
    finally:
        model.train(was_training)
    return -stacked.var(dim=0, unbiased=False)


def confidence_map(
    out: FieldOutput,
    batch: FieldBatch,
    config: SamplingConfig,
    *,
    variance: Tensor | None = None,
) -> Tensor:
    """三种置信度定义的统一入口（返回与 lambda 同形的张量，越大越优先保留）。"""
    if config.confidence == "event_count":
        return expected_counts(out, batch)
    if config.confidence == "peak_salience":
        return local_contrast(expected_counts(out, batch), window=config.salience_window)
    if config.confidence == "variance":
        if variance is None:
            raise ValueError("variance 置信度需要调用方提供方差张量")
        return variance
    raise ValueError(f"未知的置信度口径：{config.confidence!r}")


def sample(
    model: MaskedFieldModel,
    batch: FieldBatch,
    *,
    config: SamplingConfig | None = None,
) -> FieldOutput:
    """迭代并行解码，返回补全后的场（`steps >= 2` 契约，见模块 docstring）。

    Returns:
        FieldOutput：最后一轮的 lambda（此时全部格子都已「未遮盖」），
        `diagnostics` 里带 `steps`、`schedule` 与每步实际揭开比例（训练/评估日志用）。
    """
    settings = SamplingConfig() if config is None else config
    schedule = reveal_schedule(settings.steps, settings.schedule)
    assert_schedule_monotone(schedule)
    with torch.no_grad():
        shape = batch.batch_field_shape()
        state = torch.zeros(shape, dtype=torch.float32, device=batch.line_mask.device)
        occlusion = torch.ones(shape, dtype=torch.bool, device=batch.line_mask.device)
        total_cells = int(torch.tensor(shape).prod().item())
        revealed = 0
        revealed_per_step: list[float] = []
        for target in schedule:
            output = model.forward_state(batch, state, occlusion)
            variance = (
                _variance_confidence(
                    model,
                    batch,
                    state,
                    occlusion,
                    samples=settings.variance_samples,
                )
                if settings.confidence == "variance"
                else None
            )
            confidence = confidence_map(output, batch, settings, variance=variance)
            confidence = torch.where(
                occlusion,
                confidence,
                torch.full_like(confidence, float("-inf")),
            )
            want = min(math.ceil(target * total_cells), total_cells)
            take = max(want - revealed, 0)
            take = min(take, int(occlusion.sum().item()))
            if take > 0:
                flat_confidence = confidence.reshape(-1)
                order = torch.argsort(flat_confidence, descending=True, stable=True)[:take]
                flat_state = state.reshape(-1)
                flat_occlusion = occlusion.reshape(-1)
                fill = (
                    torch.ones_like(order, dtype=state.dtype)
                    if settings.state_fill == "binary"
                    else output.lam.reshape(-1)[order].to(dtype=state.dtype)
                )
                flat_state[order] = fill
                flat_occlusion[order] = False
                revealed += take
            revealed_per_step.append(revealed / max(total_cells, 1))
        final = model.forward_state(batch, state, occlusion)
    diagnostics = dict(final.diagnostics)
    diagnostics["steps"] = torch.as_tensor(settings.steps, dtype=torch.int64)
    diagnostics["revealed_fraction"] = torch.as_tensor(revealed_per_step, dtype=torch.float32)
    diagnostics["confidence"] = torch.as_tensor(
        {"event_count": 0, "peak_salience": 1, "variance": 2}[settings.confidence],
        dtype=torch.int64,
    )
    return FieldOutput(
        lam=final.lam,
        cum=final.cum,
        cell_prob=final.cell_prob,
        loss=None,
        diagnostics=diagnostics,
    )


__all__ = [
    "DEFAULT_STEPS",
    "MIN_STEPS",
    "Confidence",
    "SamplingConfig",
    "Schedule",
    "StateFill",
    "assert_schedule_monotone",
    "confidence_map",
    "expected_counts",
    "local_contrast",
    "reveal_schedule",
    "sample",
]
