"""D2：Ogata (1981) thinning（原则性解码器）——Plan 05 §4.2 / RFC-0029 §3.4。

与训练目标同构：训练用的是非齐次泊松 NLL，因此"从 λ 直接采样事件"的 thinning 是
唯一与其概率语义严格一致的解码方式（D1 峰值检测只是基线）。

实现要点（逐条对应 plan 05 §4.2）：

1. **同网格同测度**：把场压成 τ 轴强度

       lambda_k(tau_t) = J(tau_t) * dx * sum_{x,s,c} lambda[k, t, x, s, c]      [计数/拍]

   （`beatmorph.decoder.fieldops.tau_rate_per_beat`）。J 因子保证
   `int lambda_k dtau == sum_j lambda_j dV_j`，即与 ∫λ **同测度**；量纲是"计数/拍"。
2. **上界必须保守**：分块（默认 **1 拍**，由 `SUBDIVISIONS_PER_BEAT` 派生）取块内最大值
   ×`safety`（要求 >= 1）。thinning 的正确性依赖 `lambda_max >= sup lambda`，因此
   `ogata_thinning` 会**逐候选点校验**；一旦越界，默认**直接报错**
   （`ThinningBoundError`）而不是静默给出错误分布；`bound_policy="resample"` 时按
   plan §4.2-2 的"失败重采样"抬高端点上界并重跑该区间。
3. **K 条线各自独立采样**：训练时 NLL 是 K 个场的求和，各线场互为**竞争**而非归一化
   分布；因此同刻多线多事件是点过程天然允许的，不做"均匀分配"假设。
4. **标记的采样**：采到 τ 后按该格的 `(x, side, type)` 归一化强度采样——
   `mark_mode="joint"`（默认，数学上等于在乘积空间上直接 thinning 的条件分布）
   或 `mark_mode="factorized"`（plan §9-4 的对照臂：逐轴独立边缘，边缘分布不同）。
5. **随机性可控**：`seed` 固定后逐元素可复现（M5.3 的验收条件之一）。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np

from beatmorph.core.contracts.field import ChartFieldSpec
from beatmorph.core.contracts.phigros import (
    SUBDIVISIONS_PER_BEAT,
    TAU_GRID_DT,
    side_from_index,
)
from beatmorph.decoder.events import FieldEvent
from beatmorph.decoder.fieldops import (
    FloatArray,
    assert_field_shape,
    tau_rate_per_beat,
    to_numpy,
)
from beatmorph.field.grid import TAU_INDEX_EPS, FieldGrid

#: 区间强度函数：接受 τ 数组、返回同形强度数组（向量化）。
IntensityFn = Callable[[FloatArray], FloatArray]
#: 标记采样口径（plan 05 §9-4 的两个候选）。
MarkMode = Literal["joint", "factorized"]
#: 上界失效时的策略：报错（默认）或抬高端点重采样（plan §4.2-2）。
BoundPolicy = Literal["raise", "resample"]

#: 上界校验的**相对**容差：抵消浮点噪声，不是物理常量。
BOUND_REL_TOL: Final[float] = 1e-9


class ThinningBoundError(RuntimeError):
    """上界失效：候选点处强度超过 `upper_bound`（thinning 正确性的前提被破坏）。

    这个异常**不是**内部错误而是契约检查：静默用一个失效的上界会把采样分布
    悄悄变成另一个分布——正是本项目最警惕的"不报错但全错"形态。
    """

    def __init__(self, *, tau: float, value: float, upper_bound: float) -> None:
        super().__init__(
            f"thinning 上界失效：tau={tau!r} 处强度 {value!r} > 上界 {upper_bound!r}"
            f"（相对超出 {value / upper_bound - 1.0:.3e}）",
        )
        self.tau = tau
        self.value = value
        self.upper_bound = upper_bound


@dataclass(frozen=True, slots=True)
class ThinningConfig:
    """D2 的全部自由度（每一项都必须显式声明并写进报告）。

    Attributes:
        safety: 分块上界的安全系数，**必须 >= 1**（上界必须保守）。
        block_beats: 分块长度（拍）；`None` 表示 1 拍（= `SUBDIVISIONS_PER_BEAT` 格）。
        mark_mode: 标记采样口径（"joint" 默认 / "factorized" 对照臂）。
        seed: 随机种子（固定后可复现）。
        bound_policy: 上界失效时的策略（"raise" 默认 / "resample"）。
        max_restarts: `bound_policy="resample"` 时允许的重采样次数。
    """

    safety: float = 1.0
    block_beats: float | None = None
    mark_mode: MarkMode = "joint"
    seed: int = 0
    bound_policy: BoundPolicy = "raise"
    max_restarts: int = 4

    def __post_init__(self) -> None:
        if self.safety < 1.0:
            raise ValueError(
                f"safety 必须 >= 1（上界必须保守），得到 {self.safety!r}；"
                "若要测试上界失效路径，请直接给 ogata_thinning 传一个偏低的上界",
            )
        if self.block_beats is not None and self.block_beats <= 0.0:
            raise ValueError(f"block_beats 必须为正，得到 {self.block_beats!r}")
        if self.max_restarts < 0:
            raise ValueError("max_restarts 必须 >= 0")


@dataclass(frozen=True, slots=True)
class ThinningResult:
    """一次区间 thinning 的结果（含诊断计数）。"""

    times: FloatArray
    n_candidates: int
    n_accepted: int
    n_restarts: int


def ogata_thinning(
    intensity: IntensityFn,
    tau_start: float,
    tau_end: float,
    *,
    upper_bound: float,
    rng: np.random.Generator,
    bound_policy: BoundPolicy = "raise",
    max_restarts: int = 4,
) -> ThinningResult:
    """在 `[tau_start, tau_end)` 上按强度 `intensity` 采样事件时刻（Ogata 1981）。

    **向量化等价写法**：速率恒为 `L` 的泊松过程在长度 `W` 的区间内，事件数
    `N ~ Poisson(L*W)`，且给定 `N` 后事件位置 i.i.d. `Uniform(a, b)`。
    因此一次抽 `N` 个均匀候选再逐个以 `intensity(t)/L` 接受，与逐次抽指数间隔的
    经典 thinning **同分布**，且没有 Python 循环（万级候选也不慢）。

    Args:
        intensity: 向量化强度函数（计数/拍）。
        tau_start / tau_end: 区间（拍）。
        upper_bound: `L >= sup intensity`；**每个候选点都会被校验**。
        rng: numpy 随机数发生器（调用方持有，保证 seed 可复现）。
        bound_policy: "raise"（默认，越界即抛）或 "resample"（抬高端点重跑该区间）。
        max_restarts: "resample" 的最大重跑次数。

    Raises:
        ThinningBoundError: 上界失效且策略为 "raise"，或重采样次数用尽。
    """
    if upper_bound < 0.0:
        raise ValueError(f"upper_bound 必须 >= 0，得到 {upper_bound!r}")
    width = float(tau_end) - float(tau_start)
    if width <= 0.0 or upper_bound == 0.0:
        return ThinningResult(
            times=np.zeros(0, dtype=np.float64),
            n_candidates=0,
            n_accepted=0,
            n_restarts=0,
        )
    bound = float(upper_bound)
    n_candidates_total = 0
    for restart in range(int(max_restarts) + 1):
        n_candidates = int(rng.poisson(bound * width))
        n_candidates_total += n_candidates
        if n_candidates == 0:
            return ThinningResult(
                times=np.zeros(0, dtype=np.float64),
                n_candidates=n_candidates_total,
                n_accepted=0,
                n_restarts=restart,
            )
        times = float(tau_start) + rng.random(n_candidates) * width
        values = np.asarray(intensity(times), dtype=np.float64)
        if values.shape != times.shape:
            raise ValueError(
                f"intensity 必须返回与输入同形的数组：输入 {times.shape}，得到 {values.shape}",
            )
        worst_index = int(np.argmax(values))
        worst = float(values[worst_index])
        if worst > bound * (1.0 + BOUND_REL_TOL):
            error = ThinningBoundError(
                tau=float(times[worst_index]),
                value=worst,
                upper_bound=bound,
            )
            if bound_policy == "raise":
                raise error
            bound = worst * (1.0 + BOUND_REL_TOL)
            continue
        accept = rng.random(n_candidates) < values / bound
        accepted = np.sort(times[accept])
        return ThinningResult(
            times=np.asarray(accepted, dtype=np.float64),
            n_candidates=n_candidates_total,
            n_accepted=int(accepted.size),
            n_restarts=restart,
        )
    raise ThinningBoundError(tau=float(tau_end), value=bound, upper_bound=float(upper_bound))


def _step_intensity(rate_row: FloatArray, d_tau: float, t_bins: int) -> IntensityFn:
    """把逐格速率变成**格内常量**的阶梯强度函数（场的离散语义即如此）。

    格内常量是目标构建（桶内计数）的逆语义：桶内的事件时刻在给定计数后是均匀的，
    故采到的 τ 可以落在格内任意位置（亚格分辨），不需要额外假设。
    """

    def intensity(times: FloatArray) -> FloatArray:
        index = np.floor(times / d_tau + TAU_INDEX_EPS).astype(np.intp)
        np.clip(index, 0, t_bins - 1, out=index)
        return np.asarray(rate_row[index], dtype=np.float64)

    return intensity


def block_cells(config: ThinningConfig, grid: FieldGrid) -> int:
    """分块长度（格数）：`block_beats` 经 τ 格宽折算；默认 **1 拍**。

    默认值写成 `TAU_GRID_DT * SUBDIVISIONS_PER_BEAT`（= 1 拍，由契约不变量 I1 保证）
    而不是字面量 1.0：块长是**时间量**，它的取值必须能追溯到契约常量。
    块长决定上界的松紧：块越大上界越松、拒绝率越高（空场 + 尖峰时尤其明显），
    因此 1 拍是"上界够紧、块内速率又近似平稳"的折中。
    """
    beats = (
        TAU_GRID_DT * SUBDIVISIONS_PER_BEAT if config.block_beats is None else config.block_beats
    )
    return max(1, round(beats / grid.d_tau))


def sample_mark(
    slab: FloatArray,
    rng: np.random.Generator,
    *,
    mode: MarkMode = "joint",
) -> tuple[int, int, int]:
    """在一个 τ 格的 `(X, S, C)` 片上采样标记，返回 `(x_bin, side_index, channel)`。

    - "joint"（默认）：在整片上归一化后一次采样 —— 与"在乘积空间上直接 thinning"
      严格等价（条件分布的正确形式）；
    - "factorized"：逐轴取边缘后独立采样（plan §9-4 的对照臂，边缘分布不同）。
    """
    x_bins, sides, channels = slab.shape
    total = float(slab.sum())
    if total <= 0.0:
        raise ValueError("标记片总强度为 0：该格无事件可采样（调用方应先按速率跳过）")
    if mode == "joint":
        flat = np.asarray(slab.reshape(-1) / total, dtype=np.float64)
        index = int(rng.choice(flat.size, p=flat))
        x_bin, side, channel = np.unravel_index(index, slab.shape)
        return int(x_bin), int(side), int(channel)
    if mode == "factorized":
        px = np.asarray(slab.sum(axis=(1, 2)) / total, dtype=np.float64)
        ps = np.asarray(slab.sum(axis=(0, 2)) / total, dtype=np.float64)
        pc = np.asarray(slab.sum(axis=(0, 1)) / total, dtype=np.float64)
        return (
            int(rng.choice(x_bins, p=px)),
            int(rng.choice(sides, p=ps)),
            int(rng.choice(channels, p=pc)),
        )
    raise ValueError(f"未知的 mark_mode：{mode!r}")


def decode_thinning(
    lam: object,
    grid: FieldGrid,
    spec: ChartFieldSpec,
    *,
    config: ThinningConfig | None = None,
) -> tuple[list[FieldEvent], dict[str, float]]:
    """D2：从强度场用 thinning 解出标记点 + 诊断统计（`(events, stats)`）。"""
    settings = ThinningConfig() if config is None else config
    values = to_numpy(lam)
    assert_field_shape(values, spec)
    if grid.t_bins <= 0 or not grid.bpm_points:
        raise ValueError("decode_thinning 需要已绑定时间轴的网格（t_bins > 0）")
    grid.assert_grid()
    spec.assert_grid()

    rates = tau_rate_per_beat(values, grid)
    cells = block_cells(settings, grid)
    d_tau = grid.d_tau
    rng = np.random.default_rng(settings.seed)

    events: list[FieldEvent] = []
    n_candidates = 0
    n_restarts = 0
    n_empty_lines = 0
    for line_id in range(values.shape[0]):
        line_rate = np.asarray(rates[line_id], dtype=np.float64)
        if float(line_rate.sum()) <= 0.0:
            n_empty_lines += 1
            continue
        intensity = _step_intensity(line_rate, d_tau, grid.t_bins)
        for start in range(0, grid.t_bins, cells):
            stop = min(grid.t_bins, start + cells)
            bound = float(np.max(line_rate[start:stop])) * settings.safety
            if bound <= 0.0:
                continue
            result = ogata_thinning(
                intensity,
                float(start) * d_tau,
                float(stop) * d_tau,
                upper_bound=bound,
                rng=rng,
                bound_policy=settings.bound_policy,
                max_restarts=settings.max_restarts,
            )
            n_candidates += result.n_candidates
            n_restarts += result.n_restarts
            for tau in result.times:
                cell = int(np.floor(float(tau) / d_tau + TAU_INDEX_EPS))
                cell = min(max(cell, 0), grid.t_bins - 1)
                slab = np.asarray(values[line_id, cell], dtype=np.float64)
                x_bin, side_channel, channel = sample_mark(
                    slab,
                    rng,
                    mode=settings.mark_mode,
                )
                events.append(
                    FieldEvent(
                        line_id=line_id,
                        tau=float(tau),
                        x_bin=x_bin,
                        side=side_from_index(side_channel),
                        channel=channel,
                        confidence=float(line_rate[cell]),
                    ),
                )
    events.sort(
        key=lambda event: (event.line_id, event.tau, event.x_bin, int(event.side), event.channel),
    )
    stats = {
        "d2_seed": float(settings.seed),
        "d2_safety": float(settings.safety),
        "d2_block_cells": float(cells),
        "d2_mark_mode_joint": 1.0 if settings.mark_mode == "joint" else 0.0,
        "d2_n_candidates": float(n_candidates),
        "d2_n_events": float(len(events)),
        "d2_n_restarts": float(n_restarts),
        "d2_n_empty_lines": float(n_empty_lines),
    }
    return events, stats


__all__ = [
    "BOUND_REL_TOL",
    "BoundPolicy",
    "IntensityFn",
    "MarkMode",
    "ThinningBoundError",
    "ThinningConfig",
    "ThinningResult",
    "block_cells",
    "decode_thinning",
    "ogata_thinning",
    "sample_mark",
]
