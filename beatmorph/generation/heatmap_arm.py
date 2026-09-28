"""B1 消融臂：热图 + penalty-reduced focal（Plan 04 §4.5 / M8；RFC-0029 §3.4）。

**对照层级 = 表征层 + 训练目标层**（不是采样 / 解码层）。与主线 B2 的差异只有两处：

| 维度 | B2（主线） | B1（本模块） |
|---|---|---|
| 输出头 | 强度场 lambda >= 0（softplus / 因子化） | **未归一化热图 logits**（线性，不接 softplus / sigmoid） |
| 目标 y | 桶内计数 n_j（强度：计数 / 单位面积） | **未归一化高斯**（`gaussian_heatmap_target` + DDC 式 `hamming_smooth`） |
| 损失 | 非齐次泊松 NLL（事件项 + 积分项） | CornerNet 式 penalty-reduced focal |

条件注入、主干（音频 / 事件轨编码器 + 局部 / 全局解码器栈 + 线嵌入）、`FieldBatch`
契约与形状**完全复用** B2（`MaskedFieldModel`），因此两臂的差异可以被解释为
「训练目标 / 输出表征」的差异，而不是「两份实现各自漂移」。

⚠️ **严禁把 focal 与泊松 NLL 混用**（`losses.py` 第 286 行附近的明文约定、RFC-0029 §3.4）：
B1 的 y 是未归一化高斯（`int y != 事件数`），B2 的 lambda 是强度；两者**不在同一测度**。
本模块因此做了三件事，缺一不可：

1. **不 import** 任何泊松损失入口（`full_poisson_loss` / `masked_poisson_loss` /
   `per_line_nll` / `event_term` / `integral_term`）；
2. 把 `nn.Module.forward` **显式关闭**（它是最容易被顺手调用的入口，一旦走通就会
   在 logits 上算泊松 NLL：数值有限、不报错、静默毁掉对照实验）；
3. 由 `tests/unit/generation/test_heatmap_arm.py` 做**源码级 + 行为级**双重护栏。

**未核实、因此本模块一律不写具体数字（全部必填，由调用方给）**：

- `alpha` / `beta`：CornerNet / CenterNet 原文只写「它们是超参」，具体取值未出现在
  抓取片段中（literature §9-1）；本模块不提供默认值，也**不得**在源码里出现字面量
  （由源码级护栏断言）；
- `sigma_t` / `sigma_x` / `radius_t` / `radius_x`：CornerNet 的「sigma = radius / 3」
  属二手转述，未逐字核实；
- Hamming 平滑窗宽（DDC 用它抑制短距双峰，窗宽未核实）；
- **解码阈值**：DDC 的「每难度一个阈值」是从数据拟合出来的（literature §2.2-1），
  本模块只提供**表结构**与查找口，不给任何数值；
- 「每谱最优」的候选网格与打分函数：属 plan 06 的评估协议，以**注入**的方式传入。

**分层（不得反向依赖）**：本模块**不 import `beatmorph.infra`**。门禁回调以
`Callable[[], float]` 的形式导出（`focal_step_fn`，与 `infra.sanity.StepFn` 同形），
装配留给 infra / CLI——plan 07 §3.1 要求 `sanity.py` 保持范式中立，各臂同理。

**时间口径（红线 7）**：本模块不出现任何秒 <-> tau 换算、不含 BPM 分段积分；
tau 网格与测度只经 `FieldGrid`，秒 -> 格由 `beatmorph.decoder.peaks` 自己派生。
峰值提取**复用 `beatmorph.decoder.peaks.decode_peaks`**，不另写第二份算法。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, NoReturn

import torch
from torch import Tensor, nn

from beatmorph.core.contracts.field import ChartFieldSpec
from beatmorph.decoder.events import FieldEvent
from beatmorph.decoder.fieldops import intensity_scale, to_numpy
from beatmorph.decoder.peaks import PeakConfig, decode_peaks
from beatmorph.field.grid import FieldGrid
from beatmorph.generation.batch import FieldBatch
from beatmorph.generation.losses import (
    Reduction,
    gaussian_heatmap_target,
    hamming_smooth,
    penalty_reduced_focal_loss,
)
from beatmorph.generation.model import MaskedFieldModel, ModelConfig

# ══════════════════════════════════════════════════════════════
# 1. 目标热图（未归一化高斯 + DDC 式 Hamming 平滑）
# ══════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class HeatmapTargetConfig:
    """B1 的目标热图超参（**全部必填**：文献未核实，见模块 docstring）。

    Attributes:
        sigma_t: tau 方向高斯宽度（**格**，不是秒；秒 <-> 格只在 field/ 内换算）。
        sigma_x: x 方向高斯宽度（格）。
        radius_t: tau 方向撒点半径（格）；窗口宽 = 2 * radius_t + 1。
        radius_x: x 方向撒点半径（格）。
        smooth_window: Hamming 平滑窗宽（tau 格）；1 表示不平滑（DDC 用它抑制短距双峰）。
    """

    sigma_t: float
    sigma_x: float
    radius_t: int
    radius_x: int
    smooth_window: int

    def __post_init__(self) -> None:
        if self.sigma_t <= 0.0 or self.sigma_x <= 0.0:
            raise ValueError(f"sigma 必须为正，得到 ({self.sigma_t!r}, {self.sigma_x!r})")
        if self.radius_t < 0 or self.radius_x < 0:
            raise ValueError(f"radius 必须 >= 0，得到 ({self.radius_t!r}, {self.radius_x!r})")
        if self.smooth_window < 1:
            raise ValueError(f"smooth_window 必须 >= 1，得到 {self.smooth_window!r}")


def heatmap_target(counts: Tensor, config: HeatmapTargetConfig) -> Tensor:
    """桶内计数 (B, K, T, X, S, C) -> **未归一化**热图 y，同形。

    走 plan 04 §4.5 的既有实现：`gaussian_heatmap_target`（逐事件撒高斯、重叠取
    element-wise max、边界裁剪）+ `hamming_smooth`（沿 tau 轴、逐 (K, S, C) 切片）。

    两处与 B2 的口径差异（正是两者**不同测度**的具体体现，不得混用损失）：

    1. y 未归一化：`sum(y) != 事件数`，峰值取 1.0 而不是计数；
    2. 桶内计数被折叠为**存在性**（`n_j = 2` 与 `n_j = 1` 给出同一张热图），
       而 B2 的事件项是 `n_j * log lambda_j`（不去重）。

    Args:
        counts: `(B, K, T, X, S, C)` 桶内计数（int16 / float 均可）。
        config: 目标热图超参（全必填）。

    Raises:
        ValueError: counts 不是 6 维。
    """
    if counts.dim() != 6:
        raise ValueError(f"counts 必须是 (B, K, T, X, S, C)，得到 {tuple(counts.shape)}")
    batch_size, n_lines, t_bins, x_bins, sides, channels = counts.shape
    folded = counts.reshape(batch_size * n_lines, t_bins, x_bins, sides, channels)
    peaks = gaussian_heatmap_target(
        folded,
        sigma_t=config.sigma_t,
        sigma_x=config.sigma_x,
        radius_t=config.radius_t,
        radius_x=config.radius_x,
    )
    smoothed = hamming_smooth(peaks, window=config.smooth_window)
    return smoothed.reshape(batch_size, n_lines, t_bins, x_bins, sides, channels)


# ══════════════════════════════════════════════════════════════
# 2. 目标函数（focal；**与泊松 NLL 无任何入口相连**）
# ══════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class HeatmapObjective:
    """B1 的训练目标配置：目标热图 + focal 的 alpha / beta + 归约口径。

    `alpha` / `beta` / `reduction` **全部必填**：前两者在文献中未核实（模块 docstring），
    后者决定 loss 的量级口径（`sum` 随格数线性放大，`mean` 被占多数的空格摊薄）——
    plan 07 §3.1 明确要求「每次使用必须显式记录实际生效的口径与阈值」。

    Attributes:
        target: 目标热图超参。
        alpha: focal 的**正样本**指数（正样本项 `(1 - p)^alpha * log p`）。
        beta: focal 的**负样本**指数（负样本项 `(1 - y)^beta * p^alpha * log(1 - p)`）。
        reduction: "sum" / "mean" / "none"；"none" 返回**逐格**张量（focal 是逐格损失，
            与泊松 NLL 的逐 (B, K) 口径不同——这是两者不可混用的另一面）。
    """

    target: HeatmapTargetConfig
    alpha: float
    beta: float
    reduction: Reduction

    def __post_init__(self) -> None:
        if self.alpha < 0.0 or self.beta < 0.0:
            raise ValueError(f"alpha / beta 必须非负，得到 ({self.alpha!r}, {self.beta!r})")


def _reduce_supervised(per_cell: Tensor, batch: FieldBatch, reduction: Reduction) -> Tensor:
    """按 **line_mask** 归约逐格损失：padding 线的贡献与梯度**恰为 0**（与 B2 同契约）。

    `mean` 的分母是**被监督的格数**（有效线 x T*X*S*C），不是含 padding 的总格数——
    否则同一批数据里 padding 比例一变，loss 的标度就跟着变（门禁阈值随之失真）。
    """
    active = batch.line_mask_bool().reshape(batch.batch_size(), batch.n_lines(), 1, 1, 1, 1)
    supervised = per_cell * active.to(dtype=per_cell.dtype)
    if reduction == "none":
        return supervised
    total = supervised.sum()
    if reduction == "sum":
        return total
    if reduction == "mean":
        cells = int(
            batch.grid.t_bins * batch.grid.x_bins * batch.grid.sides * batch.grid.channels,
        )
        divisor = float(active.sum().item()) * float(cells)
        return total / max(divisor, 1.0)
    raise ValueError(f"未知的 reduction：{reduction!r}")


def focal_heatmap_loss(
    logits: Tensor,
    batch: FieldBatch,
    objective: HeatmapObjective,
    *,
    target: Tensor | None = None,
) -> Tensor:
    """B1 的训练损失：`penalty_reduced_focal_loss`(logits, y)（y 由 counts 构造）。

    Args:
        logits: `(B, K, T, X, S, C)` **未归一化**热图 logits。
        batch: 含 `counts` 的 batch（推理时没有目标，不应调用本函数）。
        objective: 目标配置（alpha / beta / 归约口径全显式）。
        target: 预先构造好的 y（供逐 step 复用同一条目标，避免重复撒点）。

    Returns:
        标量损失（`reduction="none"` 时是逐格张量）。

    Raises:
        ValueError: batch 没有 counts（推理路径不得调用训练损失）。
        AssertionError: logits 与 batch 的场形状不一致。
    """
    if batch.counts is None:
        raise ValueError(
            "B1 的 focal 损失需要 batch.counts（桶内计数 -> 热图目标）；推理时不应调用"
        )
    expected = batch.batch_field_shape()
    if tuple(logits.shape) != expected:
        raise AssertionError(f"logits 必须是 {expected}，得到 {tuple(logits.shape)}")
    y = heatmap_target(batch.counts, objective.target) if target is None else target
    per_cell = penalty_reduced_focal_loss(
        logits,
        y,
        alpha=objective.alpha,
        beta=objective.beta,
        reduction="none",
    )
    return _reduce_supervised(per_cell, batch, objective.reduction)


# ══════════════════════════════════════════════════════════════
# 3. 模型臂（复用 B2 主干；输出头换成热图 logits）
# ══════════════════════════════════════════════════════════════


class HeatmapHead(nn.Module):
    """B1 的输出头：**未归一化**热图 logits（不接 softplus / sigmoid）。

    与 `FieldHead` 的「直连 skip」同构（plan 04 §9-17）：输出头除了读解码器输出，
    还直接读该 token 的输入特征（可见场 + 遮盖通道）。这不是可选的工程糖——缺了这条
    短梯度路径，优化会停在只学边际分布的盆地里（该结论是在 B2 上实测得到的，
    两臂共享同一套主干与输入通路，因此同样适用）。
    """

    def __init__(self, config: ModelConfig, grid: FieldGrid) -> None:
        super().__init__()
        self.x_bins: int = grid.x_bins
        self.sides: int = grid.sides
        self.channels: int = grid.channels
        self.cells: int = grid.x_bins * grid.sides * grid.channels
        self.cell_head = nn.Linear(config.d_model, self.cells)
        self.cell_skip = nn.Linear(2 * self.cells, self.cells)

    def forward(self, tokens: Tensor, *, input_features: Tensor | None = None) -> Tensor:
        """tokens (B, K, T, d) -> logits (B, K, T, X, S, C)；input_features 为该 token 的输入特征。

        不做任何非负 / 归一化变换：focal 自带 sigmoid，任何前置变换都会改变目标的语义。
        """
        batch_size, n_lines, t_bins, _ = tokens.shape
        logits: Tensor = self.cell_head(tokens)
        if input_features is not None:
            logits = logits + self.cell_skip(input_features)
        return logits.reshape(
            batch_size,
            n_lines,
            t_bins,
            self.x_bins,
            self.sides,
            self.channels,
        )


@dataclass(frozen=True, slots=True)
class HeatmapOutput:
    """B1 臂的输出（**不是** `FieldOutput`：lambda >= 0 的语义在这里不成立）。

    Attributes:
        logits: `(B, K, T, X, S, C)` 未归一化热图 logits（可为负）。
        probability: `(B, K, T, X, S, C)` = sigmoid(logits)，供解码与诊断。
        loss: 标量 focal 损失（未提供目标时为 None）。
        diagnostics: 训练日志用的诊断张量（**不作为损失项**）。
    """

    logits: Tensor
    probability: Tensor
    loss: Tensor | None = None
    diagnostics: Mapping[str, Tensor] = field(default_factory=dict)

    def assert_shapes(self, batch: FieldBatch) -> None:
        """输出形状与 batch 网格严格一致，且 probability == sigmoid(logits)（失败即抛）。"""
        expected = batch.batch_field_shape()
        if tuple(self.logits.shape) != expected:
            raise AssertionError(f"logits 必须是 {expected}，得到 {tuple(self.logits.shape)}")
        if tuple(self.probability.shape) != expected:
            raise AssertionError(
                f"probability 必须是 {expected}，得到 {tuple(self.probability.shape)}"
            )
        if not bool(torch.isfinite(self.logits).all()):
            raise AssertionError("热图 logits 出现 NaN / Inf")
        if not bool(torch.allclose(self.probability, torch.sigmoid(self.logits), atol=1e-6)):
            raise AssertionError("probability 必须等于 sigmoid(logits)")


class HeatmapArm(nn.Module):
    """B1 臂（Plan 04 §4.5 / M8）：**复用 B2 主干**，输出未归一化热图 logits。

    复用方式（composition 而非复制）：内部持有一个 `MaskedFieldModel` 作主干，
    只借用它的 `embedding`（可见场 + 遮盖通道 + 位置 + 线嵌入）、`decode`
    （局部 / 全局层交替的解码器栈）与条件编码分支；主干自带的 `FieldHead`（lambda
    输出头）在本臂里被**显式删除**——留着它就是一组永不更新、却照样进 optimizer /
    checkpoint 的「死参数」。两臂因此共享同一份主干实现，不会各自漂移。

    `MaskedFieldModel.forward` 走泊松 NLL，与 focal 不同测度（RFC-0029 §3.4）；
    本类因此把 `nn.Module.forward` **关闭**（`__call__` 会直接落到它上面）并指向
    `forward_heatmap`——不提供任何「顺手调用」的入口。
    """

    def __init__(
        self,
        config: ModelConfig,
        grid: FieldGrid,
        objective: HeatmapObjective,
    ) -> None:
        super().__init__()
        self.config = config
        self.grid = grid
        self.objective = objective
        self.trunk = MaskedFieldModel(config, grid)
        del self.trunk.head
        self.head = HeatmapHead(config, grid)

    # ── 前向 ───────────────────────────────────────────────────
    def encode_state(
        self, batch: FieldBatch, state: Tensor, occlusion: Tensor
    ) -> tuple[Tensor, Tensor]:
        """(可见场, 遮盖通道) -> (解码器输出 (B, K, T, d), 该 token 的输入特征)。"""
        tokens, features = self.trunk.embedding(
            state,
            occlusion,
            n_lines=batch.n_lines(),
            t_bins=batch.grid.t_bins,
            device=state.device,
        )
        return self.trunk.decode(batch, tokens), features

    def forward_heatmap_state(
        self,
        batch: FieldBatch,
        state: Tensor,
        occlusion: Tensor,
        *,
        compute_loss: bool = True,
        objective: HeatmapObjective | None = None,
    ) -> HeatmapOutput:
        """用**任意**可见场状态前向（热图臂没有迭代解码，此入口供诊断 / 消融用）。"""
        expected = batch.batch_field_shape()
        if tuple(state.shape) != expected or tuple(occlusion.shape) != expected:
            raise AssertionError(f"state / occlusion 必须是 {expected}")
        tokens, features = self.encode_state(batch, state, occlusion)
        logits: Tensor = self.head(tokens, input_features=features)
        probability = torch.sigmoid(logits)
        loss: Tensor | None = None
        if compute_loss and batch.counts is not None:
            loss = focal_heatmap_loss(
                logits,
                batch,
                self.objective if objective is None else objective,
            )
        diagnostics: dict[str, Tensor] = {
            "logit_max": logits.detach().amax(),
            "probability_mean": probability.detach().mean(),
        }
        return HeatmapOutput(
            logits=logits,
            probability=probability,
            loss=loss,
            diagnostics=diagnostics,
        )

    def forward_heatmap(
        self,
        batch: FieldBatch,
        *,
        compute_loss: bool = True,
        objective: HeatmapObjective | None = None,
    ) -> HeatmapOutput:
        """标准前向：可见场 = counts * (~occlusion)（与 B2 同一输入约定）。"""
        batch.assert_shapes()
        return self.forward_heatmap_state(
            batch,
            batch.observed_counts(),
            batch.occlusion_bool(),
            compute_loss=compute_loss,
            objective=objective,
        )

    def forward(self, batch: FieldBatch, *, compute_loss: bool = True) -> NoReturn:
        """**显式关闭**：`nn.Module.forward` 在本仓的语义是「走泊松 NLL 的前向」。

        B1 与泊松 NLL 不同测度（RFC-0029 §3.4），因此这里不提供「顺手调用」的入口：
        一旦 `model(batch)` 能跑通，实验就会在 logits 上算泊松 NLL——loss 有限、
        不报错、静默毁掉整个对照臂。
        """
        raise NotImplementedError(
            "B1 臂不使用 nn.Module.forward（本仓该入口 = 泊松 NLL 前向，与 focal 不同测度，"
            "RFC-0029 §3.4）：请改用 forward_heatmap（或 forward_heatmap_state）。"
            f"（batch 的 compute_loss={compute_loss}）",
        )


# ══════════════════════════════════════════════════════════════
# 4. 门禁回调（与 beatmorph.infra.sanity.StepFn 同形；**不 import infra**）
# ══════════════════════════════════════════════════════════════

#: 「前向 + 反传 + 一步优化」并返回标量 loss 的回调，与 `beatmorph.infra.sanity.StepFn`
#: **同形**（`Callable[[], float]`）。本模块刻意不 import `beatmorph.infra`：
#: 装配（G1-G4 的执行、日志、退出码）归 infra / CLI（plan 07 §3.1）。
HeatmapStepFn = Callable[[], float]


def focal_step_fn(
    model: HeatmapArm,
    batch: FieldBatch,
    *,
    learning_rate: float,
    objective: HeatmapObjective | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    clip_grad_norm: float | None = None,
) -> HeatmapStepFn:
    """构造可直接交给 `beatmorph.infra.sanity` 的 `StepFn`（B1 臂的一步优化）。

    返回的闭包每次调用做一次「前向 -> 反传 -> 一步优化」，返回该步的**标量 loss**
    （`float(loss.detach())`），因此 `overfit_single_batch` 可以直接吃它（判据、阈值、
    日志全部留在 infra，本模块只提供这一步）。

    Args:
        model: B1 臂。
        batch: 含 counts 的 batch（门禁要求单 / 少样本过拟合，因此调用方通常只给 1-4 个样本）。
        learning_rate: 优化器学习率（**必填**：任何默认值都是未核实的超参）。
        objective: 目标配置；None 时用 `model.objective`（便于只换 alpha / beta 做扫描）。
        optimizer: 外部优化器；None 时按 `learning_rate` 现构 Adam。
        clip_grad_norm: 可选的梯度范数裁剪（None = 不裁剪）。

    Returns:
        `Callable[[], float]`：跑一步并返回该步 loss。

    Note:
        本函数会把模型切到 `train()` 模式（dropout 的有无由 `ModelConfig.dropout` 决定，
        默认 0 = 确定性前向）；调用方若需要 `eval()`，请在拿到回调后自行切换。
    """
    if learning_rate <= 0.0:
        raise ValueError(f"learning_rate 必须为正，得到 {learning_rate!r}")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("模型没有可训练参数：门禁回调无法优化任何东西")
    resolved = (
        optimizer if optimizer is not None else torch.optim.Adam(parameters, lr=learning_rate)
    )
    model.train()

    def step() -> float:
        resolved.zero_grad(set_to_none=True)
        output = model.forward_heatmap(batch, objective=objective)
        loss = output.loss
        if loss is None:
            raise ValueError("forward_heatmap 未给出 loss：batch 缺少 counts，门禁需要监督目标")
        # torch 的 stub 未标注 Tensor.backward（本仓 mert.py 同款处理）
        loss.backward()  # type: ignore[no-untyped-call]
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(parameters, clip_grad_norm)
        resolved.step()
        return float(loss.detach())

    return step


# ══════════════════════════════════════════════════════════════
# 5. 解码路径（热图 -> 事件；峰值提取**复用** decoder.peaks）
# ══════════════════════════════════════════════════════════════

#: 阈值口径（两栏报告）：与 `beatmorph.eval.protocol.DecodeRegime` **同名同义**，
#: 便于报告把两栏拼到一起（plan 06 §3.1：`decode_regime: {fixed, per_chart_best}`）。
#: 本模块不 import `beatmorph.eval`——评估在依赖方向上是生成 / 解码的**下游**。
ThresholdRegime = Literal["fixed", "per_chart_best"]

#: 固定阈值表的难度取整位数：与 `beatmorph/eval/protocol.py::DIFFICULTY_ROUND_DIGITS`
#: 同口径（plan 06 §4.4-2：实测定数出现 14.900001 / 18.000004 这类浮点噪声）。
#: 此处不 import eval（生成侧不依赖评估模块）；两处若分叉，由 plan 06 的口径单测负责抓。
DIFFICULTY_ROUND_DIGITS: int = 1


@dataclass(frozen=True, slots=True)
class HeatmapDecodeConfig:
    """热图解码的**非阈值**自由度（阈值属两栏口径的数据结构，见 `DifficultyThresholdTable`）。

    阈值本身**不在**本类里：固定阈值按难度给（`每难度一个阈值`，DDC 配置），
    每谱最优阈值由候选网格 + 打分函数搜出来——两者是**两栏**，不是同一格的默认值比拼。

    Attributes:
        nms_tau_bins: tau 轴 NMS 半径（格）。
        nms_x_bins: x 轴 NMS 半径（格）。
        smooth_seconds: 解码侧 Hamming 平滑窗宽（**秒**；0 表示不平滑）。
            秒 -> 格由 `beatmorph.decoder.peaks.smooth_cells` 经 `J(tau)` 折算，
            本模块不参与换算（红线 7）。
    """

    nms_tau_bins: int
    nms_x_bins: int
    smooth_seconds: float

    def __post_init__(self) -> None:
        if self.nms_tau_bins < 1 or self.nms_x_bins < 1:
            raise ValueError("NMS 半径必须 >= 1 格（0 会让同一格内产生多个峰）")
        if self.smooth_seconds < 0.0:
            raise ValueError(f"smooth_seconds 必须 >= 0，得到 {self.smooth_seconds!r}")


@dataclass(frozen=True, slots=True)
class DifficultyThresholdTable:
    """「**固定阈值**」栏的数据结构：难度 -> 阈值（**热图概率单位**，即 sigmoid 后的取值）。

    每难度一个阈值是 DDC 的配置（literature §2.2-1：DDC 实测阈值 0.5 -> 最优时
    F1 由 0.5006 升到 0.7317，故阈值必须显式声明）；**具体数值必须从数据拟合，未核实**，
    因此本类不提供任何默认阈值，也**不提供缺档回退**——静默回退会把「难度条件没接上」
    洗成绿灯，正是本项目最怕的失效类型。

    Attributes:
        thresholds: 难度（已 round 到 `round_digits`）-> 阈值；键必须已取整。
        round_digits: 难度比较前的取整位数（默认 1，与 plan 06 §4.4-2 同口径）。
    """

    thresholds: Mapping[float, float]
    round_digits: int = DIFFICULTY_ROUND_DIGITS

    def __post_init__(self) -> None:
        if self.round_digits < 0:
            raise ValueError(f"round_digits 必须 >= 0，得到 {self.round_digits!r}")
        for difficulty, threshold in self.thresholds.items():
            if threshold <= 0.0:
                raise ValueError(
                    f"阈值必须为正（热图概率单位），难度 {difficulty!r} 得到 {threshold!r}"
                )

    def threshold_for(self, difficulty: float) -> float:
        """查该难度的固定阈值；缺档**报错**（不得静默回退到别的难度）。"""
        key = round(float(difficulty), self.round_digits)
        if key not in self.thresholds:
            raise KeyError(
                f"难度 {difficulty!r}（round 后 {key!r}）不在固定阈值表里："
                f"{sorted(self.thresholds)}。缺档必须显式补，不得回退（否则难度条件失效"
                "会被静默洗成绿灯）",
            )
        return float(self.thresholds[key])


@dataclass(frozen=True, slots=True)
class ThresholdSearchResult:
    """「**每谱最优阈值**」栏的一次搜索结果（保留全部候选与得分，供报告复核）。"""

    threshold: float
    score: float
    n_events: int
    candidates: tuple[float, ...]
    scores: tuple[float, ...]

    def summary(self) -> dict[str, float]:
        """一行读数（阈值 / 得分 / 事件数 / 候选数）。"""
        return {
            "per_chart_best_threshold": float(self.threshold),
            "per_chart_best_score": float(self.score),
            "per_chart_best_n_events": float(self.n_events),
            "per_chart_best_n_candidates": float(len(self.candidates)),
        }


@dataclass(frozen=True, slots=True)
class HeatmapDecodeResult:
    """**一栏**的解码结果。两栏（固定阈值 / 每谱最优）共用**同一个结构**。

    这不是审美问题：plan 06 §3.1 / §4.2-4 要求两栏同格式，否则「能调阈值的一方占便宜」
    （ITGPT 实测同一模型 F1@0.5 = 0.7801 vs Max F1 = 0.8022；DDC 更极端 0.5006 -> 0.7317）。

    Attributes:
        regime: 本栏口径（"fixed" / "per_chart_best"）。
        threshold: 本栏实际生效的阈值（热图概率单位）。
        events: 解码出的标记点（`beatmorph.decoder.events.FieldEvent`）。
        stats: 解码器诊断（**含 decoder 自己的 d1_* 键**，证明走的是同一个峰值实现）。
        difficulty: 该谱的定数（固定阈值栏据此查表）；None = 未提供。
        search: 每谱最优栏的搜索记录（固定阈值栏为 None）。
    """

    regime: ThresholdRegime
    threshold: float
    events: tuple[FieldEvent, ...]
    stats: Mapping[str, float]
    difficulty: float | None = None
    search: ThresholdSearchResult | None = None

    def summary(self) -> dict[str, float]:
        """本栏的一行读数（栏名进键，便于两栏并排）。"""
        report: dict[str, float] = {
            f"{self.regime}_threshold": float(self.threshold),
            f"{self.regime}_n_events": float(len(self.events)),
        }
        return report


@dataclass(frozen=True, slots=True)
class ThresholdColumns:
    """两栏口径的**同一份**结构（plan 04 §4.5 / M8：固定解码规则 + 每谱最优阈值）。"""

    fixed: HeatmapDecodeResult
    per_chart_best: HeatmapDecodeResult

    def summary(self) -> dict[str, float]:
        """两栏并排的一行读数（**同一个字典、同一套键**）。"""
        report = dict(self.fixed.summary())
        report.update(self.per_chart_best.summary())
        difficulty = self.fixed.difficulty
        if difficulty is not None:
            report["difficulty"] = float(difficulty)
        if self.per_chart_best.search is not None:
            report.update(self.per_chart_best.search.summary())
        return report


def threshold_as_alpha(probability: object, grid: FieldGrid, *, threshold: float) -> float:
    """把「热图概率阈值」换算成 decoder 的 `alpha` 口径（`threshold = alpha * lambda_0`）。

    `lambda_0` 取 `beatmorph.decoder.fieldops.intensity_scale`——**decoder 自己的**
    标度函数（plan 05 §4.1-3），因此本模块不新造测度；换算后 `decode_peaks` 内部算出的
    `d1_threshold` 恰好等于传入的绝对阈值（由测试定量断言）。

    Raises:
        ValueError: 阈值非正，或热图全零（无法定义标度；全零热图通常意味着模型塌陷）。
    """
    if threshold <= 0.0:
        raise ValueError(f"阈值必须为正（热图概率单位），得到 {threshold!r}")
    values = to_numpy(probability)
    scale = float(intensity_scale(values, grid, n_events=None))
    if scale <= 0.0:
        raise ValueError(
            "热图全零：lambda_0 = 0，无法把绝对阈值换算成 alpha（模型塌陷的信号，"
            "不静默返回空事件）",
        )
    return float(threshold) / scale


def decode_heatmap(
    probability: object,
    grid: FieldGrid,
    *,
    threshold: float,
    config: HeatmapDecodeConfig,
    spec: ChartFieldSpec | None = None,
    regime: ThresholdRegime = "fixed",
    difficulty: float | None = None,
    search: ThresholdSearchResult | None = None,
) -> HeatmapDecodeResult:
    """热图（概率）-> 事件：**复用** `beatmorph.decoder.peaks.decode_peaks`（不另写峰值算法）。

    阈值的单位是**热图概率**（sigmoid 后的取值）；decoder 的阈值口径是 `alpha * lambda_0`，
    换算经 `threshold_as_alpha`（用 decoder 自己的标度函数）。

    ⚠️ 输入是**单样本**的场（`(K, T, X, S, C)`）——与 `decode_peaks` / `decode_field`
    同一口径（plan 05 的解码器是单谱的）。批输出 `HeatmapOutput.probability`
    `(B, K, T, X, S, C)` 必须由调用方按样本取出（本模块不替它决定"按谱"的语义：
    两栏报告本来就是**逐谱**的）。

    Args:
        probability: `(K, T, X, S, C)` 热图概率（torch / numpy 均可）。
        grid: 已绑定时间轴的网格（`t_bins > 0` 且带 BPMList）。
        threshold: 本栏的阈值（热图概率单位）。
        config: 解码自由度（NMS 半径、解码侧平滑窗宽）。
        spec: 网格规格；省略时由 grid 与 K 现构并断言。
        regime: 本栏口径（写进结果）。
        difficulty: 该谱定数（写进结果）。
        search: 每谱最优栏的搜索记录（写进结果）。

    Raises:
        ValueError: 热图不是 5 维 / 阈值非正 / 热图全零。

    Returns:
        `HeatmapDecodeResult`（`stats` 里带 decoder 自己的 `d1_*` 键）；`events` 是解码器
        内部的 `FieldEvent`（含 tau）。进入 plan 06 的评估前须经
        `beatmorph.decoder.events.pair_events` 转成**秒域** `DecodedEvent`（Hold 配对也
        在那里做）——本模块不重复实现该转换（plan 05 的职责）。
    """
    if threshold <= 0.0:
        raise ValueError(f"阈值必须为正（热图概率单位），得到 {threshold!r}")
    values = to_numpy(probability)
    if values.ndim != 5:
        raise ValueError(
            f"热图必须是**单样本**的 (K, T, X, S, C)（decoder 口径），得到 {values.shape}；"
            "若来自 HeatmapOutput.probability，请先按样本取出（B=1 时即 probability[0]）",
        )
    k = int(values.shape[0])
    resolved = grid.spec(k) if spec is None else spec
    resolved.assert_grid()
    alpha = threshold_as_alpha(values, grid, threshold=threshold)
    peak_config = PeakConfig(
        alpha=alpha,
        nms_tau_bins=config.nms_tau_bins,
        nms_x_bins=config.nms_x_bins,
        smooth_seconds=config.smooth_seconds,
        n_events=None,
    )
    events, stats = decode_peaks(values, grid, resolved, config=peak_config)
    stats = dict(stats)
    stats["b1_threshold"] = float(threshold)
    stats["b1_k"] = float(k)
    return HeatmapDecodeResult(
        regime=regime,
        threshold=float(threshold),
        events=tuple(events),
        stats=stats,
        difficulty=None if difficulty is None else float(difficulty),
        search=search,
    )


def search_threshold(
    probability: object,
    grid: FieldGrid,
    *,
    candidates: Sequence[float],
    score_fn: Callable[[Sequence[FieldEvent]], float],
    config: HeatmapDecodeConfig,
    spec: ChartFieldSpec | None = None,
    difficulty: float | None = None,
) -> tuple[HeatmapDecodeResult, ThresholdSearchResult]:
    """「**每谱最优阈值**」栏：在候选网格上按 `score_fn` 取最大，返回 (结果, 搜索记录)。

    打分函数由调用方注入（plan 06 的 F1@±20ms / ±50ms，或任何报告口径）——
    本模块**不 import `beatmorph.eval`**：评估是下游，且「能调阈值的臂占便宜」这件事
    只能靠**同格式两栏报告**来约束，不能靠生成侧偷偷选一个好看的指标。

    并列时取候选顺序中**更早**的一个（确定性：同一输入两次搜索必须逐位一致）。

    Args:
        probability: `(K, T, X, S, C)` 热图概率。
        grid: 已绑定时间轴的网格。
        candidates: 候选阈值（**由调用方给**；本模块不提供任何默认网格）。
        score_fn: 打分函数（入参是该阈值下的事件序列，返回标量，越大越好）。
        config: 解码自由度。
        spec: 网格规格。
        difficulty: 该谱定数（写进结果）。

    Raises:
        ValueError: 候选为空或含非正值。
    """
    grid_values = tuple(float(value) for value in candidates)
    if not grid_values:
        raise ValueError("候选阈值不得为空（本模块不提供默认阈值网格）")
    if any(value <= 0.0 for value in grid_values):
        raise ValueError(
            f"候选阈值必须为正（热图概率单位），得到 {sorted(set(grid_values))[:3]}..."
        )
    scores: list[float] = []
    columns: list[HeatmapDecodeResult] = []
    for value in grid_values:
        column = decode_heatmap(
            probability,
            grid,
            threshold=value,
            config=config,
            spec=spec,
            regime="per_chart_best",
            difficulty=difficulty,
        )
        columns.append(column)
        scores.append(float(score_fn(column.events)))
    best_index = max(range(len(grid_values)), key=lambda index: (scores[index], -index))
    best = ThresholdSearchResult(
        threshold=grid_values[best_index],
        score=scores[best_index],
        n_events=len(columns[best_index].events),
        candidates=grid_values,
        scores=tuple(scores),
    )
    winner = columns[best_index]
    return (
        HeatmapDecodeResult(
            regime="per_chart_best",
            threshold=winner.threshold,
            events=winner.events,
            stats=winner.stats,
            difficulty=winner.difficulty,
            search=best,
        ),
        best,
    )


def decode_threshold_columns(
    probability: object,
    grid: FieldGrid,
    *,
    fixed: DifficultyThresholdTable,
    difficulty: float,
    candidates: Sequence[float],
    score_fn: Callable[[Sequence[FieldEvent]], float],
    config: HeatmapDecodeConfig,
    spec: ChartFieldSpec | None = None,
) -> ThresholdColumns:
    """**两栏口径一次出数**：固定阈值栏 + 每谱最优阈值栏（结构与格式完全相同）。

    Args:
        probability: `(K, T, X, S, C)` 热图概率。
        grid: 已绑定时间轴的网格。
        fixed: 固定阈值表（每难度一个阈值；缺档报错）。
        difficulty: 该谱定数（查表键）。
        candidates: 「每谱最优」栏的候选阈值网格。
        score_fn: 「每谱最优」栏的打分函数（如 plan 06 的 F1）。
        config: 解码自由度。
        spec: 网格规格。
    """
    fixed_result = decode_heatmap(
        probability,
        grid,
        threshold=fixed.threshold_for(difficulty),
        config=config,
        spec=spec,
        regime="fixed",
        difficulty=difficulty,
    )
    best_result, _ = search_threshold(
        probability,
        grid,
        candidates=candidates,
        score_fn=score_fn,
        config=config,
        spec=spec,
        difficulty=difficulty,
    )
    return ThresholdColumns(fixed=fixed_result, per_chart_best=best_result)


__all__ = [
    "DIFFICULTY_ROUND_DIGITS",
    "DifficultyThresholdTable",
    "HeatmapArm",
    "HeatmapDecodeConfig",
    "HeatmapDecodeResult",
    "HeatmapHead",
    "HeatmapObjective",
    "HeatmapOutput",
    "HeatmapStepFn",
    "HeatmapTargetConfig",
    "ThresholdColumns",
    "ThresholdRegime",
    "ThresholdSearchResult",
    "decode_heatmap",
    "decode_threshold_columns",
    "focal_heatmap_loss",
    "focal_step_fn",
    "heatmap_target",
    "search_threshold",
    "threshold_as_alpha",
]
