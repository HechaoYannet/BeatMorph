"""评估统计辅助：秩相关 / bootstrap / 相关性（自实现，**不引入 scipy**）。

plan 06 §5 明确 scipy **当前不在依赖中**，而 corruption 准入（§4.8 / M6.6）需要
dose-rank 关联与 song-cluster bootstrap 区间。本模块用 numpy 自实现这两件事，
并**固定随机种子**，保证报告可复现（plan §4.9 验收表「可复现」一行的前提）。

约定：

- 所有区间都是**百分位 bootstrap**（percentile bootstrap），不是 BCa；
- 样本不足（len < 2）或方差为 0 时返回 `nan` ——**缺失不等于 0**（0 是合法读数，
  用它冒充缺失会让报告失真）。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

#: 默认显著性水平（plan 06 §9-5 的多重比较校正方法**未定**，此处只固定单次比较的 α）。
DEFAULT_ALPHA: Final[float] = 0.05


def average_ranks(values: Sequence[float]) -> list[float]:
    """平均秩（并列取平均秩，Spearman 的并列标准做法）。"""
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="stable")
    ranks = np.empty(array.size, dtype=np.float64)
    index = 0
    while index < order.size:
        stop = index
        while stop + 1 < order.size and array[order[stop + 1]] == array[order[index]]:
            stop += 1
        shared = (index + stop) / 2.0 + 1.0
        ranks[order[index : stop + 1]] = shared
        index = stop + 1
    return [float(value) for value in ranks]


def pearson_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson 相关系数；长度不足或任一侧方差为 0 时返回 `nan`。"""
    if len(x) != len(y):
        raise ValueError(f"x 与 y 长度必须一致：{len(x)} != {len(y)}")
    if len(x) < 2:
        return float("nan")
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def spearman_rank_correlation(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman 秩相关（并列取平均秩）；样本不足或方差为 0 时返回 `nan`。

    corruption 准入（plan 06 §4.8「目标读数随失败强度呈**负的 dose-rank 关联**」）
    的判据量：dose 递增而指标单调退化时该值为 -1。
    """
    return pearson_correlation(average_ranks(x), average_ranks(y))


def mean_or_nan(values: Sequence[float]) -> float:
    """算术平均；空序列返回 `nan`（缺失 != 0）。"""
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def percentile_interval(
    samples: Sequence[float],
    *,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[float, float]:
    """百分位区间（默认 95%）；空样本返回 `(nan, nan)`。"""
    if not samples:
        return (float("nan"), float("nan"))
    array = np.asarray(samples, dtype=np.float64)
    low, high = np.percentile(array, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return (float(low), float(high))


def bootstrap_interval(
    samples: Sequence[float],
    *,
    resamples: int,
    seed: int,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[float, float]:
    """样本均值的百分位 bootstrap 区间（**确定性**：同 seed 同结果）。

    重采样单元由调用方决定：corruption 准入按 **cluster**（曲目）给出
    `samples`，即 song-cluster bootstrap（plan 06 §4.8；cluster 定义见 §9-2，
    本实现按曲目标识聚类）。
    """
    if resamples < 1:
        raise ValueError(f"resamples 必须 >= 1，得到 {resamples!r}")
    if not samples:
        return (float("nan"), float("nan"))
    array = np.asarray(samples, dtype=np.float64)
    if array.size == 1:
        value = float(array[0])
        return (value, value)
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, array.size, size=(resamples, array.size))
    means: FloatArray = np.asarray(array[draws].mean(axis=1), dtype=np.float64)
    return percentile_interval([float(value) for value in means], alpha=alpha)


__all__ = [
    "DEFAULT_ALPHA",
    "average_ranks",
    "bootstrap_interval",
    "mean_or_nan",
    "pearson_correlation",
    "percentile_interval",
    "spearman_rank_correlation",
]
