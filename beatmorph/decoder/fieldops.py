"""场张量的 numpy 适配与标度（`lambda_0 = N / |Omega|`）——Plan 05 §4.1-3。

本模块**不 import torch**：`FieldOutput.lam` 在推理时是 torch 张量，但解码器只需要
数值，因此用鸭子类型适配（`detach` / `cpu` / `numpy` / `__array__`）。这样
plan 05 的全部单元测试可以在最小环境（无权重、无 GPU）跑，符合 CLAUDE.md §4。

标度口径（plan 05 §4.1-3）：阈值必须是 `alpha * lambda_0`，其中 `lambda_0 = N/|Omega|`
是 **G3 常数基线**的强度标度（RFC-0029 §3.2）。理由：文献实测阈值 0.5 与最优阈值
之间 F1 相差 0.23（0.5006 -> 0.7317），阈值是**必须显式声明的自由度**，不允许以
裸魔数形式出现在解码器里。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from beatmorph.core.contracts.field import FIELD_DIMS, ChartFieldSpec
from beatmorph.field.grid import FieldGrid

FloatArray = NDArray[np.float64]

#: 场张量的维度数（与契约 FIELD_DIMS 一致，不得写死 6）。
FIELD_RANK: int = len(FIELD_DIMS)


def to_numpy(value: object) -> FloatArray:
    """把 torch 张量 / numpy 数组 / 嵌套序列统一成 float64 数组（**不修改输入**）。

    不 import torch：torch 张量经 `detach().cpu().numpy()` 走鸭子类型即可，
    因此本函数在只装了 numpy 的最小环境里也能用来解码 numpy 场。
    """
    if isinstance(value, np.ndarray):
        return value.astype(np.float64, copy=False)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy") and not isinstance(value, np.ndarray):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64)


def assert_field_shape(lam: FloatArray, spec: ChartFieldSpec) -> None:
    """场张量形状必须是契约的 `(K, T, X, S, C)`（唯一写法，不得另写维度序）。"""
    expected = spec.shape()
    if lam.shape != expected:
        raise AssertionError(
            f"场张量形状必须等于契约 `{FIELD_DIMS}` 去掉 batch 轴 = {expected}，得到 {lam.shape}",
        )
    if not np.all(np.isfinite(lam)):
        raise AssertionError("场张量出现 NaN / Inf（先跑 field.assert_lambda_valid 再解码）")
    if np.any(lam < 0.0):
        raise AssertionError("强度 lambda 必须非负（softplus / exp 参数化）")


def cell_volumes(grid: FieldGrid) -> FloatArray:
    """每格体积 `dV_j = J_j * d_tau * dx`，形状 `(T,)`（**唯一的 dV 出口**）。"""
    return np.asarray(grid.cell_volumes(), dtype=np.float64)


def jacobian_cells(grid: FieldGrid) -> FloatArray:
    """逐格 Jacobian `J(tau) = dt/dtau`（秒/拍），形状 `(T,)`。"""
    return np.asarray(grid.jacobian(), dtype=np.float64)


def total_intensity(lam: FloatArray, grid: FieldGrid) -> FloatArray:
    """逐线 `int lambda_k`（路径 (a) 同网格数值积分），形状 `(K,)`。

    与 `beatmorph/field/integrate.integrate_grid` 同式同测度（`sum_j lambda_j dV_j`）；
    本模块用 numpy 复算一次是为了让解码器保持 torch-free。
    """
    volumes = cell_volumes(grid).reshape(1, -1, 1, 1, 1)
    return np.asarray((lam * volumes).sum(axis=(1, 2, 3, 4)), dtype=np.float64)


def omega_value(
    grid: FieldGrid,
    n_lines: int,
    *,
    line_mask: Sequence[bool] | NDArray[np.bool_] | None = None,
) -> float:
    """`|Omega| = sum_j dV_j`，遍历**全部 K 条线**的全部格元（含空线）。"""
    lines = int(n_lines) if line_mask is None else int(np.count_nonzero(np.asarray(line_mask)))
    return float(grid.volume(grid.t_bins, lines))


def intensity_scale(
    lam: FloatArray,
    grid: FieldGrid,
    *,
    n_events: float | None = None,
    line_mask: Sequence[bool] | NDArray[np.bool_] | None = None,
) -> float:
    """`lambda_0 = N / |Omega|`：G3 常数基线标度（plan 05 §4.1-3 的阈值单位）。

    Args:
        lam: 场张量 `(K, T, X, S, C)`。
        grid: 已绑定时间轴的网格（`dV_j` 的唯一来源）。
        n_events: 事件数 N。`None`（默认）取**场自身的积分** `int lambda`——阈值随
            场的整体强弱自适应，是解码预测场时的自洽口径；显式给出时即 G3 口径
            （用于复现 plan / 报告的绝对数字）。
        line_mask: 有效线；False 的线不计入 N 与 |Omega|。

    Raises:
        ValueError: `|Omega| <= 0`（网格未绑定时间轴）。
    """
    lines = lam.shape[0] if line_mask is None else int(np.count_nonzero(np.asarray(line_mask)))
    omega = omega_value(grid, lam.shape[0], line_mask=line_mask)
    if omega <= 0.0:
        raise ValueError(f"|Omega| 必须为正，得到 {omega!r}（网格是否已绑定时间轴？）")
    if n_events is None:
        per_line = total_intensity(lam, grid)
        if line_mask is not None:
            per_line = per_line * np.asarray(line_mask, dtype=np.float64)
        total = float(per_line.sum())
    else:
        total = float(n_events)
    if not lines:
        raise ValueError("至少需要一条有效判定线")
    return total / omega


def tau_rate_per_beat(lam: FloatArray, grid: FieldGrid) -> FloatArray:
    """τ 轴边缘强度 `lambda_k(tau)`，单位 **计数/拍**，形状 `(K, T)`。

        lambda_k(tau_t) = J(tau_t) * dx * sum_{x,s,c} lambda[k, t, x, s, c]

    这一条公式是 thinning 的全部依据（plan 05 §4.2-1）：`J` 因子保证
    `int lambda_k(tau) dtau == sum_j lambda_j dV_j`（与 ∫λ 同测度），
    漏掉 J 就等于把「拍」当「秒」用（POSTMORTEM 的同构错误）。
    """
    jacobian = jacobian_cells(grid)
    margin = lam.sum(axis=(2, 3, 4))
    return np.asarray(margin * (jacobian * grid.dx), dtype=np.float64)


__all__ = [
    "FIELD_RANK",
    "FloatArray",
    "assert_field_shape",
    "cell_volumes",
    "intensity_scale",
    "jacobian_cells",
    "omega_value",
    "tau_rate_per_beat",
    "to_numpy",
    "total_intensity",
]
