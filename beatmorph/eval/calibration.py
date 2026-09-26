"""校准指标（NLL）与探索性指标（能量相关性）的**物理隔离与标注**（plan 06 §4.6 / §4.7）。

三条硬约束（RFC-0029 §5.1、BasePlan §9、文献库 §7.3/§7.4）：

1. **NLL 只作校准，不得作质量分数**：ChartGenEval 实测 perplexity 在「常见图案重写」
   腐败下**下降 37%**（9.44 → 5.98）——似然改善 ≠ 质量改善。报告头部必须固定一行
   警示（`NLL_WARNING`）。
2. **能量相关性只是探索性**：ChartGenEval 明确标为 Exploratory development analysis
   （无 held-out 证据）→ 可以报，但必须标注，且**不得作为模型选择主判据**。
3. **隔离是结构性的**，不是口号：主判据函数 primary_criterion 的签名**只接受 quality
   分节**（QualityMetrics），因此它**在类型上就看不到** NLL 与探索性读数。报告里两者
   分列不同的 pydantic 分节（CalibrationReadout / ExploratoryReadout），各自带 role
   与 warning 字段。

本模块**不复制任何损失公式**：常数基线 λ0 = N/|Ω| 的 NLL 走 field/loss.py 的权威实现
（plan 06 §4.6 的「同时报 G3 常数基线」要求）；该 import 会带 torch，因此**惰性引入**，
使 beatmorph.eval 的导入保持 numpy 级轻量（测试据此断言 eval 不在模块顶层 import torch）。

秒域（红线 7）：本模块不实现任何秒 <-> τ 换算。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Final, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from beatmorph.eval.metrics import QualityMetrics
from beatmorph.eval.stats import pearson_correlation

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查期需要 torch
    from torch import Tensor

    from beatmorph.field.grid import FieldGrid

#: 报告模板头部的**固定警示行**（plan §4.6：「必须固定一行警示」）。
NLL_WARNING: Final[str] = "NLL 是校准读数，不是质量分数"
#: 探索性指标的标注（plan §4.7 / 文献库 §7.3）。
EXPLORATORY_WARNING: Final[str] = "探索性指标（无 held-out 证据）：不得作为模型选择的主判据"
#: 与 onset 支持率并报时的必需注记（plan §4.7 末段）。
ONSET_DISCLAIMER: Final[str] = "「未匹配到人类谱」与「未匹配到音频」是两件事，不得合并解读"
#: 主判据只允许消费的分节（供测试与审计核对）。
PRIMARY_CRITERION_SECTIONS: Final[tuple[str, ...]] = ("quality",)


class CalibrationReadout(BaseModel):
    """概率校准读数（**只作校准与门禁，不得作质量分数**）。

    Attributes:
        role: 恒为 "calibration"（分节标注，报告消费方据此分列）。
        warning: 固定警示行（plan §4.6：报告头部必须出现）。
        available: 是否真的拿到了 NLL（False 时不得把 None 当 0 读）。
        nll: 验证集泊松 NLL（点过程口径 L_point）；缺失时 None。
        nll_constant_baseline: G3 常数基线 λ0 = N/|Ω| 的 NLL（**不是** λ ≡ 0）。
        nll_per_event / nll_constant_baseline_per_event: 除以事件数的归一化读数
            （plan §9-6：λ 的绝对标度是 per-chart 的，跨谱汇总需归一化）。
        nll_zero_is_inf: 契约断言「λ ≡ 0 且 N > 0 时 NLL = +∞」（禁止 eps 平滑）。
        is_primary_criterion: 恒为 False（本分节**不得**进主判据）。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["calibration"] = "calibration"
    warning: str = NLL_WARNING
    available: bool = False
    nll: float | None = None
    nll_constant_baseline: float | None = None
    nll_per_event: float | None = None
    nll_constant_baseline_per_event: float | None = None
    nll_zero_is_inf: bool = True
    is_primary_criterion: bool = False

    @classmethod
    def unavailable(cls) -> CalibrationReadout:
        """没有 NLL 时的**显式缺失**（不得填 0：0 是合法 NLL 值）。"""
        return cls()

    @classmethod
    def from_counts(cls, *, nll: float, n_events: float, omega: float) -> CalibrationReadout:
        """由 (NLL, N, |Ω|) 构造，并附带 G3 常数基线（经 field/loss.py 的权威实现）。

        Raises:
            ValueError: N 或 |Ω| 非正（基线无定义）。
        """
        from beatmorph.field.loss import constant_baseline_nll

        if n_events <= 0.0 or omega <= 0.0:
            raise ValueError(f"N 与 |Ω| 必须为正，得到 N={n_events!r}, |Ω|={omega!r}")
        baseline = float(constant_baseline_nll(n_events, omega))
        value = float(nll)
        return cls(
            available=True,
            nll=value,
            nll_constant_baseline=baseline,
            nll_per_event=value / float(n_events),
            nll_constant_baseline_per_event=baseline / float(n_events),
        )


class ExploratoryReadout(BaseModel):
    """探索性读数（plan §4.7：**可以报，但必须标注**）。

    Attributes:
        role: 恒为 "exploratory"。
        warning: 固定标注（不得作为模型选择的主判据）。
        onset_disclaimer: 与 onset 支持率并报时的必需注记。
        energy_correlation: 密度曲线与音乐能量曲线的相关性（Pearson；未定义时 None）。
        energy_bin_s: 相关性所用的分箱宽度（秒）——读数必须带口径。
        available: 是否真的算了（False 时 None 不是 0）。
        is_primary_criterion: 恒为 False。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["exploratory"] = "exploratory"
    warning: str = EXPLORATORY_WARNING
    onset_disclaimer: str = ONSET_DISCLAIMER
    available: bool = False
    energy_correlation: float | None = None
    energy_bin_s: float | None = None
    is_primary_criterion: bool = False

    @classmethod
    def unavailable(cls) -> ExploratoryReadout:
        """未计算时的显式缺失。"""
        return cls()

    @classmethod
    def from_curves(
        cls,
        density: Sequence[float],
        energy: Sequence[float],
        *,
        bin_s: float,
    ) -> ExploratoryReadout:
        """由两条**同网格**曲线构造（长度必须一致；相关性未定义时记 None）。"""
        if len(density) != len(energy):
            raise ValueError(f"密度与能量曲线长度必须一致：{len(density)} != {len(energy)}")
        correlation = pearson_correlation(list(density), list(energy))
        return cls(
            available=True,
            energy_correlation=None if math.isnan(correlation) else float(correlation),
            energy_bin_s=float(bin_s),
        )


def primary_criterion(quality: QualityMetrics, *, tolerance_s: float) -> float:
    """模型选择的**主判据**：事件级 timing-F1 @ 主容差（plan §4.6 / §4.7 的落点）。

    ⚠️ 本函数的签名**只接受 QualityMetrics**：它在类型上就读不到 CalibrationReadout
    （NLL）与 ExploratoryReadout（能量相关性）。「NLL 不得作主判据」这条硬约束因此
    是**结构性**的，而不是靠评审记得（测试同时做签名检查与投毒行为检查）。
    """
    return float(quality.timing_f1(tolerance_s))


def assert_calibration_is_labeled(readout: CalibrationReadout) -> None:
    """校准分节的标注自检（role / warning / 不得进主判据）。"""
    if readout.role != "calibration":
        raise AssertionError(f"校准分节的 role 必须是 calibration，得到 {readout.role!r}")
    if readout.warning != NLL_WARNING:
        raise AssertionError(f"校准分节必须带固定警示行：{NLL_WARNING!r}")
    if readout.is_primary_criterion:
        raise AssertionError("校准分节不得作为模型选择的主判据（plan §4.6 硬约束）")
    if readout.available and readout.nll is None:
        raise AssertionError("available=True 时必须给出 nll（缺失 != 0）")


def assert_exploratory_is_labeled(readout: ExploratoryReadout) -> None:
    """探索性分节的标注自检（role / warning / 不得进主判据）。"""
    if readout.role != "exploratory":
        raise AssertionError(f"探索性分节的 role 必须是 exploratory，得到 {readout.role!r}")
    if readout.warning != EXPLORATORY_WARNING:
        raise AssertionError(f"探索性分节必须带标注：{EXPLORATORY_WARNING!r}")
    if readout.is_primary_criterion:
        raise AssertionError("探索性指标不得作为模型选择的主判据（plan §4.7 硬约束）")


def binned_density(
    event_times_s: Sequence[float],
    *,
    bin_s: float,
    duration_s: float | None = None,
) -> tuple[float, ...]:
    """事件密度曲线（每个箱的事件数 / 箱宽，单位「事件每秒」）。

    Args:
        event_times_s: 事件时刻（秒）。
        bin_s: 箱宽（秒）；**必须显式给出**（plan §4.7 只要求相关性，未定箱宽口径，
            因此由调用方声明并随读数记录）。
        duration_s: 时间轴长度（秒）；None = 取最后一个事件所在箱的右边界。

    Returns:
        长度 = ceil(duration_s / bin_s) 的密度序列（末尾箱若不满也按实际宽度归一）。
    """
    if bin_s <= 0.0:
        raise ValueError(f"bin_s 必须 > 0，得到 {bin_s!r}")
    times = [float(value) for value in event_times_s]
    if not times and duration_s is None:
        return ()  # 无事件且未给时间轴 -> 空曲线（而不是「一个空箱」）
    span = max(times, default=0.0) + bin_s if duration_s is None else float(duration_s)
    if span <= 0.0:
        return ()
    count = max(1, math.ceil(span / bin_s))
    histogram, _ = np.histogram(
        np.asarray(times, dtype=np.float64),
        bins=count,
        range=(0.0, count * bin_s),
    )
    return tuple(float(value) / bin_s for value in histogram)


def energy_correlation(
    density: Sequence[float],
    energy: Sequence[float],
    *,
    bin_s: float,
) -> ExploratoryReadout:
    """密度-能量相关性（**探索性读数**，plan §4.7）。

    返回值是 ExploratoryReadout（自带 role / warning / 不得进主判据的标注），
    而不是裸 float——标注必须无法被顺手丢掉。
    """
    return ExploratoryReadout.from_curves(density, energy, bin_s=bin_s)


def poisson_nll_float(
    counts: Tensor,
    lam: Tensor,
    grid: FieldGrid,
    *,
    line_mask: Sequence[bool] | None = None,
) -> float:
    """调用 field/loss.py 的权威 NLL 并取标量（**不在 eval 内复制损失公式**）。

    torch 在此**惰性引入**：beatmorph.eval 的模块级导入保持 numpy 级轻量。
    """
    from beatmorph.field.loss import poisson_nll

    value = poisson_nll(counts, lam, grid, line_mask=line_mask, reduction="sum")
    return float(value.item())


def check_zero_intensity_diverges(
    counts: Tensor,
    grid: FieldGrid,
    *,
    line_mask: Sequence[bool] | None = None,
) -> float:
    """契约断言（M6.5）：λ ≡ 0 且 N > 0 时 NLL **必须为 +∞**（禁止 eps 平滑）。

    Raises:
        AssertionError: NLL 有限（说明有人加了 eps 平滑或改了 NLL 口径）。
    """
    import torch

    lam = torch.zeros_like(counts)
    value = poisson_nll_float(counts, lam, grid, line_mask=line_mask)
    if not (math.isinf(value) and value > 0.0):
        raise AssertionError(f"λ ≡ 0 的 NLL 必须为 +∞，得到 {value!r}（禁止 eps 平滑）")
    return value


__all__ = [
    "EXPLORATORY_WARNING",
    "NLL_WARNING",
    "ONSET_DISCLAIMER",
    "PRIMARY_CRITERION_SECTIONS",
    "CalibrationReadout",
    "ExploratoryReadout",
    "assert_calibration_is_labeled",
    "assert_exploratory_is_labeled",
    "binned_density",
    "check_zero_intensity_diverges",
    "energy_correlation",
    "poisson_nll_float",
    "primary_criterion",
]
