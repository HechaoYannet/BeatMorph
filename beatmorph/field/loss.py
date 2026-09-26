"""非齐次泊松 NLL（桶内计数语义）+ G3 常数基线 + 强不平衡诊断（Plan 03 §3.3/§4.5/§4.7）。

口径（plan §4.5，逐项与 RFC-0029 §3.2 一致，测度按 Q15 改为 dV_j = J_j*d_tau*dx）：

    L_point  = -sum_j n_j log lambda_j + sum_j lambda_j dV_j
    L_binned = -sum_j n_j log(lambda_j dV_j) + sum_j lambda_j dV_j + sum_j log(n_j!)
             = L_point - sum_j n_j log dV_j + sum_j log(n_j!)

两者对参数**等价**（梯度相同）、数值不同；默认报告 L_point 并同报 L_binned。

三条硬契约（M4）：

1. lambda == N/|Omega| 时 L_point == N * (1 + log(|Omega|/N))（闭式，G3 基线）；
2. lambda == 0 时 NLL **必须非有限**——**禁止 eps 平滑**（那是 G3 立论的凭据）；
3. 桶内计数 n_j = 2 贡献 2*log(lambda_j)：**禁止对事件去重**（literature §4.4-4）。

本模块**不做任何通道 / 侧别 / 线的重加权**（plan §4.7，M5 的契约测试即证明这一点）：
损失是逐格项的直接求和，没有任何权重参数。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from beatmorph.field.grid import FieldGrid
from beatmorph.field.integrate import apply_line_mask, count_true, field_cell_volumes

#: NLL 归约方式（"none" 返回逐线 (K,) 张量）
Reduction = Literal["sum", "mean", "none"]


def softplus_lambda(raw: Tensor, *, beta: float = 1.0) -> Tensor:
    """lambda >= 0 的参数化（plan §4.6：软加 / exp 保证非负，禁止裸线性输出）。"""
    return torch.nn.functional.softplus(raw, beta=beta)


def assert_lambda_valid(lam: Tensor) -> None:
    """训练步的硬检查：lambda 必须非负且无 NaN/Inf（plan R-03-5，配合 plan 04）。"""
    if bool(torch.isnan(lam).any()):
        raise AssertionError("lambda 出现 NaN")
    if bool(torch.isinf(lam).any()):
        raise AssertionError("lambda 出现 Inf")
    if bool((lam < 0).any()):
        raise AssertionError("lambda 必须非负")


def _event_term(counts: Tensor, lam: Tensor, log_lam: Tensor) -> Tensor:
    """-sum n_j log lambda_j，逐线 (K,)。

    空格的 0 * log(0) 按数学极限取 0（不引入 eps）：n_j = 0 的格子对事件项无贡献；
    n_j > 0 且 lambda_j = 0 时给出 +Inf（正是 M4(ii) 要的契约行为）。
    """
    safe = torch.where(counts > 0, log_lam, torch.zeros_like(log_lam))
    return -(counts * safe).sum(dim=(1, 2, 3, 4))


def nll_terms(
    counts: Tensor,
    lam: Tensor,
    grid: FieldGrid,
    *,
    line_mask: Tensor | Sequence[bool] | None = None,
) -> tuple[Tensor, Tensor]:
    """返回逐线 (事件项, 积分项)，均已在 line_mask 之外置 0。"""
    n = counts.to(dtype=lam.dtype)
    volumes = field_cell_volumes(grid, ref=lam)
    event = _event_term(n, lam, torch.log(lam))
    integral = (lam * volumes).sum(dim=(1, 2, 3, 4))
    return apply_line_mask(event, line_mask), apply_line_mask(integral, line_mask)


def poisson_nll(
    counts: Tensor,
    lam: Tensor,
    grid: FieldGrid,
    *,
    line_mask: Tensor | Sequence[bool] | None = None,
    reduction: Reduction = "sum",
) -> Tensor:
    """非齐次泊松 NLL 的**点过程口径** L_point（默认报告口径）。

    Args:
        counts: 桶内计数 n_j（int16/float），形状 (K, T, X, S, C)；**不是 0/1 热图**。
        lam: 强度 lambda_j >= 0，形状同 counts。
        grid: 已绑定时间轴的网格（dV_j 的唯一来源）。
        line_mask: (K,) 有效线；False 的线贡献与梯度**恰为 0**（padding 必须排除）。
        reduction: "sum" / "mean"（按有效线数）/ "none"（逐线 (K,)）。
    """
    event, integral = nll_terms(counts, lam, grid, line_mask=line_mask)
    per_line = event + integral
    if reduction == "none":
        return per_line
    total = per_line.sum()
    if reduction == "sum":
        return total
    if reduction == "mean":
        active = per_line.shape[0] if line_mask is None else count_true(line_mask)
        return total / max(float(active), 1.0)
    raise ValueError(f"未知的 reduction：{reduction!r}")


def binned_poisson_nll(
    counts: Tensor,
    lam: Tensor,
    grid: FieldGrid,
    *,
    line_mask: Tensor | Sequence[bool] | None = None,
    reduction: Reduction = "sum",
) -> Tensor:
    """分块泊松计数口径 L_binned（与 L_point 只差一个与参数无关的常数）。

    差 = sum_j log(n_j!) - sum_j n_j log dV_j（M4(iv) 的机器可验证表述）。
    """
    n = counts.to(dtype=lam.dtype)
    volumes = field_cell_volumes(grid, ref=lam)
    log_rate = torch.log(lam * volumes)
    event = _event_term(n, lam, log_rate)
    integral = (lam * volumes).sum(dim=(1, 2, 3, 4))
    log_factorial = torch.lgamma(n + 1.0).sum(dim=(1, 2, 3, 4))
    per_line = apply_line_mask(event + integral + log_factorial, line_mask)
    if reduction == "none":
        return per_line
    total = per_line.sum()
    if reduction == "sum":
        return total
    if reduction == "mean":
        active = per_line.shape[0] if line_mask is None else count_true(line_mask)
        return total / max(float(active), 1.0)
    raise ValueError(f"未知的 reduction：{reduction!r}")


def binned_point_gap(counts: Tensor, lam: Tensor, grid: FieldGrid) -> Tensor:
    """L_binned - L_point 的**闭式**（与参数无关）：sum log(n!) - sum n_j log dV_j。"""
    n = counts.to(dtype=lam.dtype)
    volumes = field_cell_volumes(grid, ref=lam)
    log_factorial = torch.lgamma(n + 1.0).sum()
    return log_factorial - (n * torch.log(volumes)).sum()


# ══════════════════════════════════════════════════════════════
# G3 常数基线（闭式，M4(i)）
# ══════════════════════════════════════════════════════════════


def constant_baseline_lambda(n_events: float, omega: float) -> float:
    """常数场的极值点 c* = N / |Omega|（**不是 0**）。"""
    if omega <= 0.0:
        raise ValueError(f"|Omega| 必须为正，得到 {omega!r}")
    return float(n_events) / float(omega)


def constant_baseline_nll(n_events: float, omega: float) -> float:
    """常数基线的极小值 N * (1 + log(|Omega| / N))（N = 0 时为 0）。"""
    if omega <= 0.0:
        raise ValueError(f"|Omega| 必须为正，得到 {omega!r}")
    if n_events <= 0.0:
        return 0.0
    return float(n_events) * (1.0 + math.log(float(omega) / float(n_events)))


def constant_lambda_field(
    grid: FieldGrid,
    n_events: float,
    *,
    omega_value: float,
    n_lines: int,
    line_mask: Sequence[bool] | None = None,
    dtype: torch.dtype = torch.float64,
) -> Tensor:
    """构造常数场 lambda == N/|Omega|（有效线全部取该值；无效线为 0）。"""
    value = constant_baseline_lambda(n_events, omega_value)
    lam = torch.full(
        (n_lines, grid.t_bins, grid.x_bins, grid.sides, grid.channels),
        value,
        dtype=dtype,
    )
    return apply_line_mask(lam, line_mask)


# ══════════════════════════════════════════════════════════════
# 强不平衡 / 线集中：分层诊断（**替代**损失重加权，plan §4.7）
# ══════════════════════════════════════════════════════════════


def normalized_line_entropy(
    per_line_counts: Sequence[float] | np.ndarray | Tensor,
    *,
    line_mask: Sequence[bool] | np.ndarray | None = None,
) -> float:
    """归一化线熵 H / log(K_active) ∈ [0, 1]（survey §7.6 思路；0 表示完全集中）。

    K_active <= 1 或总计数为 0 时定义为 0.0（**不是** 0/0）。
    """
    counts = np.asarray(per_line_counts, dtype=np.float64)
    if line_mask is not None:
        counts = counts[np.asarray(line_mask, dtype=bool)]
    total = float(counts.sum())
    if counts.size <= 1 or total <= 0.0:
        return 0.0
    prob = counts / total
    positive = prob > 0.0
    entropy = -float(np.sum(prob[positive] * np.log(prob[positive])))
    return entropy / math.log(float(counts.size))


@dataclass(frozen=True, slots=True)
class NllDecomposition:
    """NLL 的分层分解（训练日志与可视化用；**不含任何权重**）。

    每层之和恒等于 total（M5 的契约测试即断言这一点）——
    这就是「逐格项权重恒为 1」的机器可验证表述。
    """

    total: float
    event_term: float
    integral_term: float
    per_line: tuple[float, ...]
    per_side: tuple[float, float]
    per_channel: tuple[float, ...]
    per_line_counts: tuple[int, ...]
    per_side_counts: tuple[int, int]
    per_channel_counts: tuple[int, ...]
    line_entropy: float
    n_events: int
    omega: float

    def format(self) -> str:
        """渲染成可直接进训练日志的多行文本。"""
        sides = " ".join(
            f"{name}={value:.6g}"
            for name, value in zip(("front", "back"), self.per_side, strict=True)
        )
        channels = " ".join(
            f"{name}={value:.6g}"
            for name, value in zip(
                ("tap", "drag", "hold", "hold_end", "flick"),
                self.per_channel,
                strict=True,
            )
        )
        counts_by_channel = " ".join(
            f"{name}={value}"
            for name, value in zip(
                ("tap", "drag", "hold", "hold_end", "flick"),
                self.per_channel_counts,
                strict=True,
            )
        )
        return "\n".join(
            [
                f"NLL 分解：total={self.total:.6f}（事件项 {self.event_term:.6f} + 积分项 "
                f"{self.integral_term:.6f}）",
                f"  计数：n_events={self.n_events}，|Omega|={self.omega:.6g}",
                f"  per-side NLL：{sides}",
                f"  per-side 计数：front={self.per_side_counts[0]} back={self.per_side_counts[1]}",
                f"  per-channel NLL：{channels}",
                f"  per-channel 计数：{counts_by_channel}",
                f"  per-line 计数：{list(self.per_line_counts)}",
                f"  per-line NLL：{[round(v, 6) for v in self.per_line]}",
                f"  归一化线熵 H/logK：{self.line_entropy:.4f}",
            ],
        )


def nll_decomposition(
    counts: Tensor,
    lam: Tensor,
    grid: FieldGrid,
    *,
    line_mask: Tensor | Sequence[bool] | None = None,
) -> NllDecomposition:
    """per-side / per-channel / per-line 的计数与 NLL 分解 + 归一化线熵（M5）。"""
    n = counts.to(dtype=lam.dtype)
    volumes = field_cell_volumes(grid, ref=lam)
    event_cells = torch.where(n > 0, torch.log(lam), torch.zeros_like(lam)) * (-n)
    integral_cells = lam * volumes
    cells = apply_line_mask(event_cells + integral_cells, line_mask)
    per_line = cells.sum(dim=(1, 2, 3, 4))
    per_side = cells.sum(dim=(0, 1, 2, 4))
    per_channel = cells.sum(dim=(0, 1, 2, 3))
    counts_per_line = n.sum(dim=(1, 2, 3, 4))
    counts_per_side = n.sum(dim=(0, 1, 2, 4))
    counts_per_channel = n.sum(dim=(0, 1, 2, 3))
    masked_counts = counts_per_line
    if line_mask is not None:
        mask = line_mask if isinstance(line_mask, Tensor) else torch.as_tensor(line_mask)
        masked_counts = counts_per_line * mask.to(counts_per_line.dtype)
    total = float(per_line.sum().item())
    return NllDecomposition(
        total=total,
        event_term=float(apply_line_mask(event_cells, line_mask).sum().item()),
        integral_term=float(apply_line_mask(integral_cells, line_mask).sum().item()),
        per_line=tuple(float(value) for value in per_line.tolist()),
        per_side=(float(per_side[0]), float(per_side[1])),
        per_channel=tuple(float(value) for value in per_channel.tolist()),
        per_line_counts=tuple(int(value) for value in counts_per_line.tolist()),
        per_side_counts=(int(counts_per_side[0]), int(counts_per_side[1])),
        per_channel_counts=tuple(int(value) for value in counts_per_channel.tolist()),
        line_entropy=normalized_line_entropy(
            masked_counts,
            line_mask=None if line_mask is None else np.asarray(line_mask, dtype=np.bool_),
        ),
        n_events=int(n.sum().item()),
        omega=float(omega_value_for(grid, lam, line_mask=line_mask)),
    )


def omega_value_for(
    grid: FieldGrid,
    lam: Tensor,
    *,
    line_mask: Tensor | Sequence[bool] | None = None,
) -> float:
    """有效线上的 |Omega|（口径与 poisson_nll 的积分项一致）。"""
    n_lines = lam.shape[0]
    if line_mask is None:
        return grid.volume(grid.t_bins, n_lines)
    return grid.volume(grid.t_bins, count_true(line_mask))


__all__ = [
    "NllDecomposition",
    "Reduction",
    "assert_lambda_valid",
    "binned_point_gap",
    "binned_poisson_nll",
    "constant_baseline_lambda",
    "constant_baseline_nll",
    "constant_lambda_field",
    "nll_decomposition",
    "nll_terms",
    "normalized_line_entropy",
    "omega_value_for",
    "poisson_nll",
    "softplus_lambda",
]
