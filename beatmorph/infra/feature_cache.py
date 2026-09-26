"""特征缓存元数据的**配置级**校验（plan 07 §3.4 / M7.6）。

plan 02 的 `FeatureCacheMeta.verify` 已经把缓存自身与**契约常量**对齐（rate / sample_rate /
feat_dim / 帧数哨兵…）。本模块补的是另一半：**配置侧的期望值**。

为什么两半都要有：帧率是**派生量**，它唯一的事实源是 `core/contracts`；如果某天有人在配置里
写下 `frame_rate_hz = 25`，那么「缓存与配置一致」与「缓存与契约一致」会各自成立或各自失败，
而**两者都通过**才说明整条链没有第二份帧率。因此这里做两件事：

1. 配置期望帧率必须**等于**契约派生值（不一致 → 直接报错，不是 warning）；
2. 缓存元数据 rate 必须**等于**配置期望值（不一致 → 报错，不是 warning）。

理由见 POSTMORTEM-2026-08-05：帧率错配不会报错，只会让下游静默错位。
"""

from __future__ import annotations

from pathlib import Path

from beatmorph.core.contracts import MERT_FRAME_RATE_HZ
from beatmorph.data.pipeline.embed import (
    FeatureArray,
    FeatureCacheMeta,
    FeatureCacheMismatchError,
    load_feature_cache,
)

__all__ = [
    "FeatureRateMismatch",
    "expected_frame_rate",
    "load_feature_cache_checked",
    "validate_feature_cache",
]


class FeatureRateMismatch(FeatureCacheMismatchError):  # noqa: N818 - 语义是「不匹配」而非通用错误
    """配置期望帧率 / 缓存帧率与契约派生值不一致（**直接报错，不得降级为 warning**）。"""


def expected_frame_rate() -> float:
    """配置侧唯一允许的期望帧率 = 契约派生值。"""
    return float(MERT_FRAME_RATE_HZ)


def validate_feature_cache(
    meta: FeatureCacheMeta,
    *,
    expected_rate: float | None = None,
    expected_layer: int | None = None,
    expected_model_rev: str | None = None,
) -> None:
    """校验缓存元数据的帧率与可选口径字段。

    Args:
        meta: 缓存元数据。
        expected_rate: 配置期望帧率（None = 用契约派生值）。
        expected_layer / expected_model_rev: 可选的额外口径校验。

    Raises:
        FeatureRateMismatch: 期望帧率与契约不一致，或缓存 rate 与期望不一致。
    """
    rate = expected_frame_rate() if expected_rate is None else float(expected_rate)
    if rate != expected_frame_rate():
        raise FeatureRateMismatch(
            f"配置期望帧率 {rate} != 契约派生值 {expected_frame_rate()}（"
            f"MERT_SAMPLE_RATE_HZ / MERT_CONV_STRIDE_PRODUCT）：整条链只允许一个帧率事实源",
        )
    if float(meta.rate) != rate:
        raise FeatureRateMismatch(
            f"缓存 rate={meta.rate} != 期望 {rate}：拒绝加载（错配不会报错，只会静默错位）",
        )
    if expected_layer is not None and meta.layer != expected_layer:
        raise FeatureRateMismatch(f"缓存 layer={meta.layer} != 期望 {expected_layer}")
    if expected_model_rev is not None and meta.model_rev != expected_model_rev:
        raise FeatureRateMismatch(
            f"缓存 model_rev={meta.model_rev!r} != 期望 {expected_model_rev!r}"
        )


def load_feature_cache_checked(
    npz_path: Path,
    meta_path: Path | None = None,
    *,
    expected_rate: float | None = None,
    expected_layer: int | None = None,
    expected_model_rev: str | None = None,
    expected_dtype: str | None = None,
    expected_adapter: str | None = None,
) -> tuple[FeatureArray, FeatureCacheMeta]:
    """加载缓存：先走 plan 02 的完整校验，再做配置级帧率校验。

    Raises:
        FeatureRateMismatch: 帧率相关不一致。
        FeatureCacheMismatchError: plan 02 的六项校验失败。
    """
    array, meta = load_feature_cache(
        Path(npz_path),
        meta_path,
        expected_layer=expected_layer,
        expected_model_rev=expected_model_rev,
        expected_dtype=expected_dtype,
        expected_adapter=expected_adapter,
    )
    validate_feature_cache(
        meta,
        expected_rate=expected_rate,
        expected_layer=expected_layer,
        expected_model_rev=expected_model_rev,
    )
    return array, meta
