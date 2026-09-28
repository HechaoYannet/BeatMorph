"""训练目标：非齐次泊松 NLL（全谱 / 遮盖补全两种口径）与消融臂的目标函数
（Plan 04 §3.3/§4.3，里程碑 M4）。

口径（与 plan 03 的 L_point 同测度，dV_j = J_j * d_tau * dx 由 field/ 给出）：

    L_full   = -sum_{n in all}      n log lambda + sum_k int lambda_k        （校准 / G3 / 报告）
    L_masked = -(1/s) * sum_{n in S} n log lambda + sum_k int lambda_k       （训练）

其中 S 是被监督的事件集合、s 是其期望比例。**积分项永远在完整域上计算**
（全 K 条线、全 tau 域、含 J(tau) 测度），不随遮盖比例缩放。

⚠️ 重标定口径的分歧（本模块发现，待 RFC 裁定；plan 04 §9-6）：
RFC-0029 §3.3 的文字是「只对被遮盖事件计算事件项」+ 系数 `1/(1-r)`，但这两者在
数学上不自洽——只监督被遮盖事件时，最优 lambda 被系统性缩放 **r** 倍
（r = 0.5 时积分 = 0.5 x 真值），因此正确的 Horvitz-Thompson 系数是 **1/r**；
`1/(1-r)` 对应的是「监督**已观测**事件」这一被 RFC 同段明确否决的退化设计
（把输入抄到输出即可）。两者在 **r = 0.5 处数值相同**（1/0.5 == 1/(1-0.5) == 2），
所以文档里的口径偏差只在 r != 0.5 时可见——这正是本项目最怕的静默失效类型。
本模块把三种口径全部实现（`ReweightMode`），默认 `hidden`（数学自洽的那一种），
并由测试在 r = 0.25 / 0.5 / 0.75 三档上分别定量。

契约（plan 04 §3.3）：

1. `r == 0`（无遮盖）时 `masked_poisson_loss == full_poisson_loss`（相对误差 <= 1e-6）；
2. **排列敏感性**：交换两条判定线的场与其事件后 loss 必须改变（线不可互换）；
3. `line_mask` 为 False 的线对 loss 与梯度的贡献**恰为 0**；
4. **不存在**任何 line 分类损失 / softmax 分配项——损失只由事件项与积分项构成。
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from beatmorph.field.grid import FieldGrid
from beatmorph.field.integrate import cumulative_from_lam, field_cell_volumes
from beatmorph.generation.batch import FieldBatch, FieldOutput
from beatmorph.generation.masks import occluded_event_share

# ══════════════════════════════════════════════════════════════
# batch 维安全的场算子（**不要**对 (B, K, ...) 直接用 field.integrate 的对应函数）
# ══════════════════════════════════════════════════════════════
#: field.integrate.apply_line_mask / cumulative_from_lam 面向**单样本** (K, T, X, S, C)；
#: 对 (B, K, ...) 的批次输入，前者会**静默错误广播**（(1,3) 掩码 x (3,3) 结果），
#: 后者会把 tau 轴当成 X 轴求和。generation 侧一律走下面两个右对齐实现。


def apply_line_mask_batched(values: Tensor, line_mask: Tensor | None) -> Tensor:
    """按 (K,) 或 (B, K) 掩码置零，掩码维度与 values 的**前导维度右对齐**。

    - values (B, K, T, X, S, C) + mask (B, K) -> mask 视为 (B, K, 1, 1, 1, 1)；
    - values (K, T, X, S, C) + mask (K,) -> mask 视为 (K, 1, 1, 1, 1)。

    置 False 的位置贡献与梯度**恰为 0**（plan 04 §3.3-3）。
    """
    if line_mask is None:
        return values
    if line_mask.dim() not in (1, 2):
        raise ValueError(f"line_mask 必须是 (K,) 或 (B, K)，得到 {tuple(line_mask.shape)}")
    if line_mask.dim() > values.dim():
        raise ValueError(f"line_mask 维度 {line_mask.dim()} 不得 > values 维度 {values.dim()}")
    if line_mask.dim() == values.dim():
        if tuple(line_mask.shape) != tuple(values.shape):
            raise ValueError(
                f"同为 {values.dim()} 维时必须同形：{tuple(line_mask.shape)} vs {tuple(values.shape)}",
            )
        mask = line_mask.to(dtype=values.dtype, device=values.device)
    else:
        shape = (*line_mask.shape, *([1] * (values.dim() - line_mask.dim())))
        mask = line_mask.reshape(shape).to(dtype=values.dtype, device=values.device)
    return values * mask


def cumulative_lambda_batched(lam: Tensor, grid: FieldGrid) -> Tensor:
    """(B, K, T, X, S, C) -> (B, K, T) 的累积强度 Lambda_k(tau)，逐样本、单调不减。

    走 plan 03 的 cumulative_from_lam（把 (B, K) 折成单样本的 K 轴），
    **不在本模块重算测度**（红线 7）。
    """
    if lam.dim() != 6:
        raise ValueError(f"lam 必须是 (B, K, T, X, S, C)，得到 {tuple(lam.shape)}")
    batch_size, n_lines, t_bins = int(lam.shape[0]), int(lam.shape[1]), int(lam.shape[2])
    folded = lam.reshape(batch_size * n_lines, t_bins, *lam.shape[3:])
    return cumulative_from_lam(folded, grid).reshape(batch_size, n_lines, t_bins)


#: 归约方式（"none" 返回逐 (B, K) 张量；"per_event" 见下，RFC-0037 R2）
#:
#: - "per_event"：**整式**除以 `D = max(批内有效线上的事件总数, 1)`。整式缩放不改变驻点
#:   （∇L' = ∇L/D）⇒ 泊松语义（λ* = 真强度）逐位保留；改变的只是**步间尺度与相对权重**。
#:   ⚠️ **只除事件项是静默失效**：那会把最优强度整体缩小 D 倍（RFC-0037 §2.2 的推导），
#:   而积分项读数看起来仍正常——禁止。
Reduction = Literal["sum", "mean", "none", "per_event"]

#: 重标定口径（见模块 docstring；默认 hidden 是数学自洽的那一种）
#:
#: - "hidden"：监督**被遮盖**事件 + 系数 1/r     -> 最优 lambda 无偏（RFC 的语义表述）
#: - "observed"：监督**已观测**事件 + 系数 1/(1-r) -> 最优 lambda 亦无偏，但存在
#:   「把输入抄到输出」的退化解（RFC-0029 §3.3 明确否决），保留供 RFC 裁定对比
#: - "hidden_doc"：RFC/plan **字面**组合（被遮盖事件 + 1/(1-r)）——数学上不自洽
#:   （最优解被缩放 (1-r)/r 倍），**仅用于定量展示分歧**，不得用于训练
#: - "none"：监督被遮盖事件但不重标定（M4 的欠计数参照臂）
ReweightMode = Literal["hidden", "observed", "hidden_doc", "none"]


def _check_pair(out: FieldOutput, batch: FieldBatch) -> None:
    if tuple(out.lam.shape) != batch.batch_field_shape():
        raise AssertionError(
            f"lam 形状 {tuple(out.lam.shape)} 与 batch {batch.batch_field_shape()} 不一致",
        )


def _reduced(per_item: Tensor, reduction: Reduction, active: Tensor | None = None) -> Tensor:
    if reduction == "none":
        return per_item
    total = per_item.sum()
    if reduction == "sum":
        return total
    if reduction == "mean":
        divisor = float(per_item.numel()) if active is None else float(active.sum().item())
        return total / max(divisor, 1.0)
    raise ValueError(f"未知的 reduction：{reduction!r}")


def range_masked_lambda(out: FieldOutput, batch: FieldBatch) -> Tensor:
    """把 lambda 限制在定义域内（range_mask 外的格恒为 0；**这不是钳位**）。"""
    ranges = batch.range_mask_bool().to(device=out.lam.device)
    return out.lam * ranges.reshape(1, 1, 1, -1, 1, 1).to(dtype=out.lam.dtype)


def line_active(batch: FieldBatch) -> Tensor:
    """(B, K) bool：有效线（padding 线不参与任何求和）。"""
    return batch.line_mask_bool()


def event_normalizer(batch: FieldBatch) -> float:
    """`"per_event"` 归约的除子 `D = max(E_total, 1)`（RFC-0037 §2.1）。

    E_total = 批内**有效线**上的事件总数（counts 按 line_mask 置零后求和）。
    空批（E_total == 0）时 D = 1 ⇒ 损失退化为积分项本身（与旧行为一致）。
    """
    if batch.counts is None:
        return 1.0
    counts = batch.counts.to(dtype=torch.float32)
    mask = batch.line_mask_bool().to(dtype=torch.float32)
    while mask.dim() < counts.dim():
        mask = mask.unsqueeze(-1)
    return max(float((counts * mask).sum()), 1.0)


def integral_term(out: FieldOutput, batch: FieldBatch) -> Tensor:
    """积分项 sum_j lambda_j dV_j，逐 (B, K)；**全 K 线、全域**（含 padding 线的置零）。"""
    _check_pair(out, batch)
    lam = range_masked_lambda(out, batch)
    volumes = field_cell_volumes(batch.grid, ref=lam)
    per_line = (lam * volumes).sum(dim=(2, 3, 4, 5))
    return apply_line_mask_batched(per_line, line_active(batch))


def event_term(
    counts: Tensor,
    lam: Tensor,
    mask: Tensor | None,
    *,
    line_mask: Tensor | None = None,
) -> Tensor:
    """-sum n_j log lambda_j（逐 (B, K)）；mask 指定参与监督的格子。

    空格的 0 * log(0) 按数学极限取 0（**不引入 eps 平滑**，与 plan 03 M4(ii) 同口径）；
    被监督且有事件的位置若 lambda == 0，则给出 +Inf（这是契约行为，不是 bug）。

    ⚠️ 实现要点（M2 门禁抓到的静默失效，见 plan 04 §9-19）：**先按 line_mask 把监督集合
    清零，再取 log**。若先 `log(lam)` 再用 `where` 选 0，反向传播会算出
    `0 * (1/0) = NaN`——padding 线的 lambda 恰为 0，于是**全部参数梯度变成 NaN**，
    而 loss 本身是有限的，不报错、只静默毁掉训练。同理，非监督格子也不得进入 log。
    """
    supervised = counts if mask is None else counts * mask.to(dtype=counts.dtype)
    if line_mask is not None:
        supervised = apply_line_mask_batched(supervised, line_mask)
    safe_lam = torch.where(supervised > 0, lam, torch.ones_like(lam))
    return -(supervised * torch.log(safe_lam)).sum(dim=(2, 3, 4, 5))


def _counts_or_raise(batch: FieldBatch) -> Tensor:
    if batch.counts is None:
        raise ValueError("该损失需要 batch.counts（桶内计数目标）；推理时不应调用")
    return batch.counts.to(dtype=torch.float32)


def full_poisson_loss(
    out: FieldOutput,
    batch: FieldBatch,
    *,
    reduction: Reduction = "sum",
) -> Tensor:
    """L_full：**全事件**事件项 + 全域积分项（校准 / G3 / 报告口径，plan 04 §4.3）。"""
    counts = _counts_or_raise(batch)
    lam = range_masked_lambda(out, batch)
    per_line = apply_line_mask_batched(
        event_term(counts, lam, None, line_mask=line_active(batch)),
        line_active(batch),
    )
    total = per_line + integral_term(out, batch)
    if reduction == "per_event":
        return total.sum() / event_normalizer(batch)
    return _reduced(total, reduction, active=line_active(batch))


def masked_poisson_loss(
    out: FieldOutput,
    batch: FieldBatch,
    *,
    reweight: ReweightMode = "hidden",
    reduction: Reduction = "sum",
) -> Tensor:
    """L_masked：遮盖补全训练损失（plan 04 §4.3；重标定口径见模块 docstring）。

    Args:
        out: 模型输出（含 lambda）。
        batch: 含 counts 与 occlusion（True = 被遮盖）的 batch。
        reweight: "hidden"（默认，监督被遮盖事件 + 1/r）/ "observed"（RFC 字面口径，
            监督已观测事件 + 1/(1-r)）/ "none"（监督被遮盖事件但不重标定；M4 的欠计数参照臂）。
        reduction: "sum" / "mean" / "none"。
    """
    counts = _counts_or_raise(batch)
    occlusion = batch.occlusion_bool()
    ratio = occluded_event_share(counts, occlusion)
    if reweight == "hidden":
        if ratio <= 0.0:
            # 无遮盖 => 无「待补全位置」=> 退化为纯密度拟合（plan 04 §3.3-1 的 r = 0 契约）
            return full_poisson_loss(out, batch, reduction=reduction)
        if ratio >= 1.0:
            raise ValueError("r == 1：全部事件都被遮盖，事件项没有可见上下文，拒绝训练")
        scale = 1.0 / ratio
        supervised = occlusion
    elif reweight == "observed":
        if ratio >= 1.0:
            raise ValueError("r == 1：没有已观测事件，observed 口径无法定义")
        scale = 1.0 / (1.0 - ratio)
        supervised = ~occlusion
    elif reweight == "hidden_doc":
        if ratio <= 0.0:
            return full_poisson_loss(out, batch, reduction=reduction)
        if ratio >= 1.0:
            raise ValueError("r == 1：全部事件都被遮盖，事件项没有可见上下文，拒绝训练")
        scale = 1.0 / (1.0 - ratio)
        supervised = occlusion
    elif reweight == "none":
        scale = 1.0
        supervised = occlusion
    else:  # pragma: no cover - 由 Literal 约束
        raise ValueError(f"未知的重标定口径：{reweight!r}")
    lam = range_masked_lambda(out, batch)
    per_line = apply_line_mask_batched(
        event_term(counts, lam, supervised, line_mask=line_active(batch)),
        line_active(batch),
    )
    total = per_line * scale + integral_term(out, batch)
    if reduction == "per_event":
        return total.sum() / event_normalizer(batch)
    return _reduced(total, reduction, active=line_active(batch))


def occlusion_ratio(batch: FieldBatch) -> float:
    """batch 的遮盖比例 r（重标定与日志的唯一来源）。"""
    if batch.counts is None:
        return 0.0
    return occluded_event_share(batch.counts.to(dtype=torch.float32), batch.occlusion_bool())


def per_line_nll(
    out: FieldOutput,
    batch: FieldBatch,
    *,
    masked: bool = False,
    reweight: ReweightMode = "hidden",
) -> Tensor:
    """逐 (B, K) 的 NLL（per-line 线熵与「最忙线」诊断用；**不含任何权重**）。"""
    counts = _counts_or_raise(batch)
    lam = range_masked_lambda(out, batch)
    if masked:
        ratio = occluded_event_share(counts, batch.occlusion_bool())
        if reweight == "hidden" and ratio <= 0.0:
            return full_poisson_loss(out, batch, reduction="none")
        if reweight == "hidden":
            scale = 1.0 / ratio
            supervised = batch.occlusion_bool()
        elif reweight == "observed":
            scale = 1.0 / (1.0 - ratio)
            supervised = ~batch.occlusion_bool()
        elif reweight == "hidden_doc":
            scale = 1.0 / (1.0 - ratio)
            supervised = batch.occlusion_bool()
        else:
            scale = 1.0
            supervised = batch.occlusion_bool()
    else:
        scale = 1.0
        supervised = None
    per_line = apply_line_mask_batched(
        event_term(counts, lam, supervised, line_mask=line_active(batch)),
        line_active(batch),
    )
    return per_line * scale + integral_term(out, batch)


# ══════════════════════════════════════════════════════════════
# 消融臂的目标函数（B1 热图 + focal；B5 absorbing 扩散 = 时间步加权掩码 CE）
# ══════════════════════════════════════════════════════════════
#: B1 的目标 y 是**未归一化高斯**（int y != 事件数），与点过程不在同一测度，
#: **禁止**与 B2 的泊松 NLL 混用（RFC-0029 §3.4）——本模块把它们分开导出，不提供混合入口。


def gaussian_heatmap_target(
    counts: Tensor,
    *,
    sigma_t: float,
    sigma_x: float,
    radius_t: int,
    radius_x: int,
) -> Tensor:
    """B1 的目标热图：每个事件在 (tau, x) 平面上撒**未归一化**高斯，重叠取 element-wise max。

    通道轴 (K, S, C) 各自独立（与 plan 03 的场同网格；不是 2D 检测器的那种单通道图）。
    """
    if counts.dim() != 5:
        raise ValueError(f"counts 必须是 (K, T, X, S, C)，得到 {tuple(counts.shape)}")
    if sigma_t <= 0.0 or sigma_x <= 0.0:
        raise ValueError("sigma 必须为正")
    shape = counts.shape
    t_dim, x_dim = int(shape[1]), int(shape[2])
    offsets_t = torch.arange(-int(radius_t), int(radius_t) + 1, dtype=torch.float32)
    offsets_x = torch.arange(-int(radius_x), int(radius_x) + 1, dtype=torch.float32)
    kernel_t = torch.exp(-(offsets_t**2) / (2.0 * sigma_t**2))
    kernel_x = torch.exp(-(offsets_x**2) / (2.0 * sigma_x**2))
    heatmap = torch.zeros_like(counts, dtype=torch.float32)
    events = torch.nonzero(counts > 0, as_tuple=False)
    for k, t, x, s, c in events.tolist():
        lo_t = max(0, t - int(radius_t))
        hi_t = min(t_dim, t + int(radius_t) + 1)
        lo_x = max(0, x - int(radius_x))
        hi_x = min(x_dim, x + int(radius_x) + 1)
        patch = torch.outer(
            kernel_t[lo_t - t + int(radius_t) : hi_t - t + int(radius_t)],
            kernel_x[lo_x - x + int(radius_x) : hi_x - x + int(radius_x)],
        )
        window = heatmap[k, lo_t:hi_t, lo_x:hi_x, s, c]
        heatmap[k, lo_t:hi_t, lo_x:hi_x, s, c] = torch.maximum(window, patch)
    return heatmap


def hamming_smooth(target: Tensor, *, window: int) -> Tensor:
    """DDC 式 Hamming 平滑（沿 tau 轴、逐 (K, S, C) 切片做）。

    DDC（Donahue et al. 2017）用它对标签做时间平滑；本函数按窗内权重归一化，
    **不改幅度语义**，因此 B1 的 y 仍是「未归一化高斯」而不是概率。
    """
    if window <= 1:
        return target
    t_dim = int(target.shape[-4])
    weights = torch.hamming_window(int(window), periodic=False, dtype=target.dtype)
    weights = weights / weights.sum()
    pad = int(window) // 2
    flat = target.movedim(-4, -1).reshape(-1, 1, t_dim)
    padded = torch.nn.functional.pad(flat, (pad, pad), mode="replicate")
    smoothed = torch.nn.functional.conv1d(padded, weights.reshape(1, 1, -1))
    restored = smoothed.reshape(*target.shape[:-4], *target.shape[-3:], t_dim)
    return restored.movedim(-1, -4).contiguous()


def penalty_reduced_focal_loss(
    logits: Tensor,
    target: Tensor,
    *,
    alpha: float,
    beta: float,
    reduction: Reduction = "sum",
) -> Tensor:
    """CornerNet 式 penalty-reduced focal loss（B1 臂的**独立**目标；RFC-0029 §3.4）。

    alpha / beta **必须显式给出**：文献中未核实其具体取值（plan 04 §4.5），
    本模块**不提供默认值**，避免把未核实数字写进代码。
    """
    if alpha < 0.0 or beta < 0.0:
        raise ValueError("alpha / beta 必须是超参且非负")
    probability = torch.sigmoid(logits)
    positive = -(1.0 - probability).pow(alpha) * torch.log(probability.clamp_min(1e-12))
    negative = (
        -(1.0 - target).pow(beta)
        * probability.pow(alpha)
        * torch.log(
            (1.0 - probability).clamp_min(1e-12),
        )
    )
    loss = torch.where(target > 0, positive, negative)
    return _reduced(loss, reduction)


def timestep_weighted_masked_ce(
    logits: Tensor,
    targets: Tensor,
    *,
    mask: Tensor,
    timestep_weights: Tensor,
    label_smoothing: float = 0.0,
    reduction: Reduction = "sum",
) -> Tensor:
    """B5 臂的目标：**时间步加权的掩码交叉熵**（absorbing-state 扩散 NELBO 的等价形式）。

    arXiv 2510.03289 式 (10)：absorbing-state 离散扩散的 NELBO 等价于按时间步加权的
    掩码 CE，故 B2 与 B5 在**训练目标层不是两件事**，差异在采样 / 调度层（RFC-0029 §3.3-4）。
    本函数只实现这一等价形式，不实现采样器（离散 tokenizer 与采样器属 plan 04 M10，未落地）。

    Args:
        logits: (B, L, V) 未归一化 logits。
        targets: (B, L) 目标 token 下标。
        mask: (B, L) bool，True = 被遮盖（参与损失）。
        timestep_weights: (B, L) 每个位置的时间步权重 w(t)（由调度给出）。
        label_smoothing: 平滑系数（GOCT 配置为 0.02；本函数不设默认值来源，由调用方给）。
        reduction: "sum" / "mean" / "none"（"none" 返回 (B, L)）。
    """
    if logits.dim() != 3:
        raise ValueError(f"logits 必须是 (B, L, V)，得到 {tuple(logits.shape)}")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing 必须在 [0, 1) 内")
    log_probability = torch.log_softmax(logits, dim=-1)
    nll = -log_probability.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    if label_smoothing > 0.0:
        smooth = -log_probability.mean(dim=-1)
        nll = (1.0 - label_smoothing) * nll + label_smoothing * smooth
    weighted = nll * timestep_weights.to(dtype=nll.dtype) * mask.to(dtype=nll.dtype)
    return _reduced(weighted, reduction)


def poisson_measure_volume(grid: FieldGrid, *, n_lines: int) -> float:
    """|Omega| = sum_j dV_j（全 K 条线）；G3 常数基线的分母，直接取 plan 03 的实现。"""
    return float(grid.volume(grid.t_bins, int(n_lines)))


def log_ratio_note() -> str:
    """把重标定口径的分歧写进训练日志（RFC 待裁定；plan 04 §9-6）。"""
    return (
        "遮盖重标定口径：默认 hidden（1/r，数学自洽）；RFC-0029 §3.3 的字面口径为 "
        "observed（1/(1-r)，监督已观测事件——与「只预测被遮盖区域」冲突）。"
        "两者在 r = 0.5 处相同，r != 0.5 时不同；本分歧待 RFC 裁定。"
    )


__all__ = [
    "ABLATION_OBJECTIVES",
    "POISSON_LOSSES",
    "Reduction",
    "ReweightMode",
    "apply_line_mask_batched",
    "cumulative_lambda_batched",
    "event_normalizer",
    "event_term",
    "full_poisson_loss",
    "gaussian_heatmap_target",
    "hamming_smooth",
    "integral_term",
    "line_active",
    "log_ratio_note",
    "masked_poisson_loss",
    "occlusion_ratio",
    "penalty_reduced_focal_loss",
    "per_line_nll",
    "poisson_measure_volume",
    "range_masked_lambda",
    "timestep_weighted_masked_ce",
]


# 供源码级契约测试引用：本模块导出的**损失**入口（必须全部只由事件项 + 积分项构成，
# 或明确标注为独立消融臂的目标）。
POISSON_LOSSES = (full_poisson_loss, masked_poisson_loss, per_line_nll)
ABLATION_OBJECTIVES = (penalty_reduced_focal_loss, timestep_weighted_masked_ce)
