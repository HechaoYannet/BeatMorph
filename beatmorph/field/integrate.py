"""两条积分路径（同网格数值积分 / 累积强度参数化）与一致性门禁（Plan 03 §3.4 / M3）。

- 路径 (a) 同网格数值积分：integrate_grid = sum_j lambda_j * dV_j，
  dV_j = J_j * d_tau * dx（**逐格不同**，tau 网格非均匀；漏乘 J_j 即把「拍」当「秒」）。
- 路径 (b) 累积强度参数化（Omi et al., NeurIPS 2019）：cumulative_from_lam 给出
  单调不减的 Lambda_k(tau)，integrate_cumulative = Lambda_k(T) - Lambda_k(0) 精确。
- lam_from_cumulative 由 Lambda 求导得到 lambda（finite_diff 或 autograd 两条实现，
  二者互测）；因子化 lambda = Lambda'(tau) * p（sum_{x,s,c} p = 1）时两条路径**恒等**——
  于是门禁检验的是「归一化是否被掩码/域外裁剪破坏」这一真实 bug 类。

本模块**不做**任何损失重加权（plan §4.7：重加权会让事件项与积分项不再同测度）。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from beatmorph.field.grid import FieldGrid

#: Lambda -> lambda 的两种求导实现（M3 要求二者互测）
CumulativeMode = Literal["finite_diff", "autograd"]


# ══════════════════════════════════════════════════════════════
# 测度（dV 的唯一张量出口）
# ══════════════════════════════════════════════════════════════


def cell_volumes_tensor(
    grid: FieldGrid,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> Tensor:
    """每格体积 dV_j = J_j * d_tau * dx，形状 (T,)。"""
    if grid.t_bins <= 0 or not grid.bpm_points:
        raise ValueError("cell_volumes_tensor 需要已绑定时间轴的网格（t_bins > 0）")
    return torch.as_tensor(grid.cell_volumes(), dtype=dtype, device=device)


def field_cell_volumes(grid: FieldGrid, *, ref: Tensor) -> Tensor:
    """可广播到 (K, T, X, S, C) 的 dV（形状 (1, T, 1, 1, 1)）。"""
    return cell_volumes_tensor(grid, dtype=ref.dtype, device=ref.device).reshape(1, -1, 1, 1, 1)


def omega(
    grid: FieldGrid,
    *,
    n_lines: int | None = None,
    line_mask: Sequence[bool] | np.ndarray | Tensor | None = None,
) -> float:
    """|Omega| = sum_j dV_j，j 遍历**全部 K 条线**的全部格元（plan §2 偏离 2）。

    全 K 条线计入（含空线）——不得只对「有 note 的线」积分，否则「线选择」无成本。
    """
    if line_mask is not None:
        active = count_true(line_mask)
    elif n_lines is not None:
        active = int(n_lines)
    else:
        raise ValueError("omega 需要 n_lines 或 line_mask 之一")
    return grid.volume(grid.t_bins, active)


def count_true(mask: Sequence[bool] | np.ndarray | Tensor) -> int:
    """统计真值个数（同时接受 numpy / torch / 序列）。"""
    if isinstance(mask, Tensor):
        return int(mask.sum().item())
    array = np.asarray(mask)
    return int(np.count_nonzero(array))


def relative_error(a: Tensor, b: Tensor) -> Tensor:
    """相对误差 |a-b| / max(|a|, |b|)；两者同为 0 时为 0（**不做 eps 平滑**）。"""
    diff = (a - b).abs()
    scale = torch.maximum(a.abs(), b.abs())
    return torch.where(scale > 0, diff / scale, torch.zeros_like(diff))


def apply_line_mask(values: Tensor, line_mask: Tensor | Sequence[bool] | None) -> Tensor:
    """按 LineMask 置零（padding 线必须排除；置 False 的线贡献与梯度恰为 0）。"""
    if line_mask is None:
        return values
    if not isinstance(line_mask, Tensor):
        line_mask = torch.as_tensor(np.asarray(line_mask), dtype=torch.bool, device=values.device)
    mask = line_mask.to(device=values.device, dtype=values.dtype)
    return values * mask.reshape(-1, *([1] * (values.dim() - 1)))


# ══════════════════════════════════════════════════════════════
# 路径 (a)：同网格数值积分
# ══════════════════════════════════════════════════════════════


def integrate_grid(
    lam: Tensor,
    grid: FieldGrid,
    *,
    line_mask: Tensor | Sequence[bool] | None = None,
) -> Tensor:
    """路径 (a)：sum_j lambda_j * dV_j -> (K,)（同网格数值积分）。"""
    volumes = field_cell_volumes(grid, ref=lam)
    per_line = (lam * volumes).sum(dim=(1, 2, 3, 4))
    return apply_line_mask(per_line, line_mask)


def cumulative_from_lam(
    lam: Tensor,
    grid: FieldGrid,
    *,
    line_mask: Tensor | Sequence[bool] | None = None,
) -> Tensor:
    """Lambda_k(tau)：对 tau 的累积强度（含 J(tau) 测度），(K, T) **单调不减**。

    约定：cum[k, t] = int_0^{tau_{t+1}} lambda = 到**第 t 格右边界**的累积，
    因此 cum[:, 0] 只含第一格，Lambda_k(0) = 0（空前缀）。
    """
    volumes = field_cell_volumes(grid, ref=lam)
    per_cell = (lam * volumes).sum(dim=(2, 3, 4))
    return apply_line_mask(per_cell.cumsum(dim=1), line_mask)


# ══════════════════════════════════════════════════════════════
# 路径 (b)：累积强度参数化
# ══════════════════════════════════════════════════════════════


def _piecewise_linear_slope(values: Tensor, d_tau: float) -> Tensor:
    """把 (K, T+1) 的结点值视为 tau 上的**分段线性** Lambda，取每格斜率 * d_tau。

    结点取 tau = 0, d_tau, ..., T*d_tau（Lambda(0) = 0）；格中心落在自己那一格内，
    因此本函数给出的 dLambda 与有限差分**完全一致**——这正是两条实现互测的前提。
    """
    n_knots = values.shape[-1]
    bins = n_knots - 1
    knots = torch.arange(n_knots, dtype=values.dtype, device=values.device) * d_tau
    # tau 必须与 values 同形（每行一个独立叶子），否则 autograd 会把 K 行的偏导求和
    tau = (torch.arange(bins, dtype=values.dtype, device=values.device) + 0.5) * d_tau
    tau = tau.reshape(1, -1).expand(values.shape[0], -1).clone().requires_grad_(True)
    index = torch.arange(bins, device=values.device)
    left = values[..., index]
    right = values[..., index + 1]
    span = knots[index + 1] - knots[index]
    weight = (tau - knots[index]) / span
    interpolated = left + (right - left) * weight
    # create_graph=True：Lambda 来自网络时必须可继续反传（否则训练通路在此断开）
    (slope,) = torch.autograd.grad(interpolated.sum(), tau, create_graph=True)
    return slope * d_tau


def delta_cumulative(
    cum: Tensor,
    grid: FieldGrid,
    *,
    mode: CumulativeMode = "finite_diff",
) -> Tensor:
    """每格的 dLambda_k,t = Lambda'(tau) * d_tau，形状 (K, T)。

    - "finite_diff"：与网格一致的有限差分（Lambda(0) = 0 作为左边界）；
    - "autograd"：对分段线性 Lambda(tau) 的结点值自动微分（Omi et al. 2019 求导路径）。
    """
    if cum.dim() != 2:
        raise ValueError(f"cum 必须是 (K, T) 二维张量，得到 {tuple(cum.shape)}")
    zero = torch.zeros_like(cum[:, :1])
    if mode == "finite_diff":
        return torch.cat([cum[:, :1], cum[:, 1:] - cum[:, :-1]], dim=1)
    if mode == "autograd":
        knots = torch.cat([zero, cum], dim=1)
        return _piecewise_linear_slope(knots, grid.d_tau)
    raise ValueError(f"未知的 mode：{mode!r}（只支持 finite_diff / autograd）")


def uniform_cell_prob(grid: FieldGrid, *, ref: Tensor) -> Tensor:
    """均匀先验 p = 1 / (X*S*C)，形状 (1, 1, X, S, C)（sum_{x,s,c} p = 1）。"""
    value = 1.0 / float(grid.x_bins * grid.sides * grid.channels)
    return torch.full(
        (1, 1, grid.x_bins, grid.sides, grid.channels),
        value,
        dtype=ref.dtype,
        device=ref.device,
    )


def normalize_cell_prob(p: Tensor, *, valid_mask: Tensor | None = None) -> Tensor:
    """把 p 在 (x, s, c) 轴上归一化为 sum = 1（**掩码后必须重新归一化**）。

    这是 M3 反例测试的靶点：掩掉域外格却忘记重新归一化 => 两条积分路径必然分歧。
    valid_mask 为 False 的格子先置 0；某条 (k, tau) 纤维全 0 时该纤维保持全 0
    （**不做 eps 平滑**：全 0 会让 lambda 恒 0，从而在事件项上正确地发散）。
    """
    if valid_mask is not None:
        p = torch.where(valid_mask, p, torch.zeros_like(p))
    total = p.sum(dim=(2, 3, 4), keepdim=True)
    positive = total > 0
    safe_total = torch.where(positive, total, torch.ones_like(total))
    normalized = p / safe_total
    return torch.where(positive.expand_as(p), normalized, torch.zeros_like(p))


def factorized_lambda(delta_cum: Tensor, cell_prob: Tensor, grid: FieldGrid) -> Tensor:
    """因子化场 lambda_j = dLambda_k,t * p_j / dV_j（**两条路径恒等**，plan §4.6）。

    sum_{x,s,c} p = 1 时，sum_{x,s,c} lambda_j * dV_j = dLambda_k,t 逐格成立。
    """
    volumes = field_cell_volumes(grid, ref=cell_prob)
    return delta_cum.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * cell_prob / volumes


def lam_from_cumulative(
    cum: Tensor,
    grid: FieldGrid,
    *,
    mode: CumulativeMode = "finite_diff",
    cell_prob: Tensor | None = None,
) -> Tensor:
    """由累积强度 Lambda 得到强度场 lambda：(K, T, X, S, C)。

    cell_prob 省略时用均匀先验（等价于把 tau 边缘强度均匀摊到 (x, s, c) 上）。
    """
    delta = delta_cumulative(cum, grid, mode=mode)
    prob = uniform_cell_prob(grid, ref=cum) if cell_prob is None else cell_prob
    return factorized_lambda(delta, prob, grid)


def integrate_cumulative(
    cum: Tensor,
    *,
    line_mask: Tensor | Sequence[bool] | None = None,
) -> Tensor:
    """路径 (b)：int lambda_k = Lambda_k(T) - Lambda_k(0) = cum[:, -1]（**精确**）。"""
    if cum.dim() != 2:
        raise ValueError(f"cum 必须是 (K, T) 二维张量，得到 {tuple(cum.shape)}")
    if cum.shape[-1] == 0:
        return torch.zeros(cum.shape[0], dtype=cum.dtype, device=cum.device)
    return apply_line_mask(cum[:, -1], line_mask)


__all__ = [
    "CumulativeMode",
    "apply_line_mask",
    "cell_volumes_tensor",
    "count_true",
    "cumulative_from_lam",
    "delta_cumulative",
    "factorized_lambda",
    "field_cell_volumes",
    "integrate_cumulative",
    "integrate_grid",
    "lam_from_cumulative",
    "normalize_cell_prob",
    "omega",
    "relative_error",
    "uniform_cell_prob",
]
