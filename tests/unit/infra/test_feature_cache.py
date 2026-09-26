"""M7.6：特征缓存元数据的配置级校验（不一致 → 报错，不是 warning）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from beatmorph.core.contracts import (
    MERT_DEFAULT_FEAT_DIM,
    MERT_FRAME_RATE_HZ,
    MERT_SAMPLE_RATE_HZ,
)
from beatmorph.data.pipeline.embed import FeatureCacheMeta, FeatureCacheMismatchError
from beatmorph.infra.feature_cache import (
    FeatureRateMismatch,
    expected_frame_rate,
    load_feature_cache_checked,
    validate_feature_cache,
)


def _write_cache(
    directory: Path,
    *,
    rate: float | None = None,
    duration_s: float = 1.0,
    layer: int = 6,
    key: str = "song",
) -> tuple[Path, Path]:
    """写一份**迷你**特征缓存（帧数由 duration x rate 派生，不写字面量帧率）。"""
    rate = expected_frame_rate() if rate is None else rate
    frames = round(duration_s * rate)
    emb = np.zeros((frames, MERT_DEFAULT_FEAT_DIM), dtype=np.float32)
    meta = FeatureCacheMeta(
        rate=rate,
        sample_rate=MERT_SAMPLE_RATE_HZ,
        layer=layer,
        model_rev="mert-v1-330M",
        duration_s=duration_s,
        original_sample_rate=MERT_SAMPLE_RATE_HZ,
        feat_dim=MERT_DEFAULT_FEAT_DIM,
        dtype="float32",
        adapter="none",
    )
    npz = directory / f"{key}.npz"
    meta_path = directory / f"{key}.meta.json"
    np.savez_compressed(npz, emb=emb)
    meta_path.write_text(meta.model_dump_json(), encoding="utf-8")
    return npz, meta_path


def test_expected_rate_comes_from_contract(tmp_path: Path) -> None:
    """期望帧率必须**等于**契约派生值（配置里没有第二个事实源）。"""
    assert expected_frame_rate() == MERT_FRAME_RATE_HZ


def test_matching_cache_loads(tmp_path: Path) -> None:
    """正例：元数据与契约一致 → 正常加载。"""
    npz, meta_path = _write_cache(tmp_path)
    array, meta = load_feature_cache_checked(npz, meta_path)
    assert array.shape[0] == round(meta.duration_s * meta.rate)
    assert meta.layer == 6


def test_config_expected_rate_mismatch_is_an_error(tmp_path: Path) -> None:
    """配置侧期望帧率偏离契约 → 报错（这正是 25 Hz 事故的形态）。"""
    npz, meta_path = _write_cache(tmp_path)
    with pytest.raises(FeatureRateMismatch, match="契约派生值"):
        load_feature_cache_checked(npz, meta_path, expected_rate=25.0)


def test_cache_rate_mismatch_is_an_error(tmp_path: Path) -> None:
    """缓存 rate 偏离契约 → 报错（plan 02 的六项校验 + 本模块的配置级校验）。"""
    npz, meta_path = _write_cache(tmp_path, rate=25.0)
    with pytest.raises(FeatureCacheMismatchError):
        load_feature_cache_checked(npz, meta_path)


def test_layer_and_model_rev_are_checked(tmp_path: Path) -> None:
    """层号 / 主干版本不一致同样拒绝加载。"""
    npz, meta_path = _write_cache(tmp_path)
    with pytest.raises(FeatureCacheMismatchError, match="layer"):
        load_feature_cache_checked(npz, meta_path, expected_layer=9)


def test_validate_feature_cache_directly(tmp_path: Path) -> None:
    """直接校验元数据对象：一致时静默通过，不一致时报错。"""
    _, meta_path = _write_cache(tmp_path)
    meta = FeatureCacheMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
    validate_feature_cache(meta)
    validate_feature_cache(meta, expected_layer=6, expected_model_rev="mert-v1-330M")
    with pytest.raises(FeatureRateMismatch):
        validate_feature_cache(meta, expected_model_rev="other-rev")
