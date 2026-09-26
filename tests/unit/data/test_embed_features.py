"""M9：特征离线提取与缓存六项校验（**默认 CI 用假编码器，不下载权重**）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from beatmorph.core.contracts import (
    MERT_CONV_STRIDE_PRODUCT,
    MERT_DEFAULT_FEAT_DIM,
    MERT_FRAME_RATE_HZ,
    MERT_SAMPLE_RATE_HZ,
)
from beatmorph.data.pipeline.embed import (
    CACHE_META_KEYS,
    FeatureCacheMeta,
    FeatureCacheMismatchError,
    audio_cache_key,
    extract_features,
    feature_cache_paths,
    load_feature_cache,
)

SECONDS = 1.0


class FakeEncoder:
    """假 MERT 编码器：帧数 = round(样本数 / 采样率 × 帧率)，不加载任何权重。"""

    def __init__(
        self,
        *,
        rate: float = MERT_FRAME_RATE_HZ,
        feat_dim: int = MERT_DEFAULT_FEAT_DIM,
        dtype: Any = np.float32,
    ) -> None:
        self.rate = rate
        self.feat_dim = feat_dim
        self.dtype = dtype
        self.calls = 0

    def encode(self, wav: Any, sample_rate: int = MERT_SAMPLE_RATE_HZ) -> Any:
        assert sample_rate == MERT_SAMPLE_RATE_HZ, "重采样必须发生在流水线入口"
        self.calls += 1
        samples = int(np.asarray(wav).shape[-1])
        frames = round(samples / MERT_SAMPLE_RATE_HZ * self.rate)
        return np.zeros((1, frames, self.feat_dim), dtype=self.dtype)

    def output_frame_rate(self) -> float:
        return self.rate


class FakeLoader:
    """假音频加载器：把文件字节数当作样本数（不解析容器格式）。"""

    def __init__(self, original_sample_rate: int = MERT_SAMPLE_RATE_HZ) -> None:
        self.original_sample_rate = original_sample_rate

    def __call__(self, path: Path) -> tuple[Any, int]:
        return np.zeros(Path(path).stat().st_size, dtype=np.float32), self.original_sample_rate


class FakeResampler:
    """假重采样器：只做采样率断言与记录（真正的重采样在 torchaudio）。"""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def __call__(self, wav: Any, source_rate: int, target_rate: int) -> Any:
        self.calls.append((source_rate, target_rate))
        assert target_rate == MERT_SAMPLE_RATE_HZ
        return np.asarray(wav)


def _write_audio(path: Path, seconds: float = SECONDS) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * round(seconds * MERT_SAMPLE_RATE_HZ))
    return path


def _extract(tmp_path: Path, **kwargs: Any) -> tuple[Path, Path, FeatureCacheMeta]:
    audio = _write_audio(tmp_path / "audio" / "fake.bin")
    encoder = kwargs.pop("encoder", None) or FakeEncoder()
    out_dir = tmp_path / "features"
    meta = extract_features(
        audio,
        out_dir,
        encoder,
        loader=kwargs.pop("loader", None) or FakeLoader(),
        resampler=kwargs.pop("resampler", None) or FakeResampler(),
        **kwargs,
    )
    npz_path, meta_path = feature_cache_paths(out_dir, audio_cache_key(audio))
    return npz_path, meta_path, meta


def test_cache_meta_keys_match_plan(tmp_path: Path) -> None:
    """六项键 + 四个留痕键（plan 01 §3.3 / plan 02 §3.6）。"""
    assert CACHE_META_KEYS == ("rate", "sample_rate", "layer", "model_rev", "duration_s")
    _npz, _meta_path, meta = _extract(tmp_path)
    for key in CACHE_META_KEYS:
        assert getattr(meta, key) is not None


def test_extract_writes_npz_and_sidecar(tmp_path: Path) -> None:
    npz_path, meta_path, meta = _extract(tmp_path)
    assert npz_path.is_file()
    assert meta_path.is_file()
    emb, loaded = load_feature_cache(npz_path, meta_path, expected_layer=meta.layer)
    assert emb.shape == (round(SECONDS * MERT_FRAME_RATE_HZ), MERT_DEFAULT_FEAT_DIM)
    assert loaded == meta


def test_meta_rate_is_derived_not_literal(tmp_path: Path) -> None:
    _npz, _meta_path, meta = _extract(tmp_path)
    assert meta.rate == MERT_FRAME_RATE_HZ
    assert meta.rate == MERT_SAMPLE_RATE_HZ / MERT_CONV_STRIDE_PRODUCT


def test_meta_records_both_sample_rates(tmp_path: Path) -> None:
    _npz, _meta_path, meta = _extract(tmp_path, loader=FakeLoader(original_sample_rate=44100))
    assert meta.original_sample_rate == 44100
    assert meta.sample_rate == MERT_SAMPLE_RATE_HZ


def test_cache_file_count_equals_audio_count(tmp_path: Path) -> None:
    out_dir = tmp_path / "features"
    for index in range(3):
        audio = _write_audio(tmp_path / "audio" / f"a{index}.bin", seconds=0.5 + index * 0.1)
        extract_features(
            audio,
            out_dir,
            FakeEncoder(),
            loader=FakeLoader(),
            resampler=FakeResampler(),
        )
    assert len(list(out_dir.glob("*.npz"))) == 3
    assert len(list(out_dir.glob("*.meta.json"))) == 3


def test_duplicate_audio_is_extracted_once(tmp_path: Path) -> None:
    """按音频内容 sha1 去重：同曲多谱只提取一次特征。"""
    audio = _write_audio(tmp_path / "audio" / "same.bin")
    copy = tmp_path / "audio" / "copy.bin"
    copy.write_bytes(audio.read_bytes())
    out_dir = tmp_path / "features"
    encoder = FakeEncoder()
    first = extract_features(
        audio,
        out_dir,
        encoder,
        loader=FakeLoader(),
        resampler=FakeResampler(),
    )
    second = extract_features(
        copy,
        out_dir,
        encoder,
        loader=FakeLoader(),
        resampler=FakeResampler(),
    )
    assert encoder.calls == 1
    assert first == second
    assert len(list(out_dir.glob("*.npz"))) == 1


def test_overwrite_forces_re_extraction(tmp_path: Path) -> None:
    audio = _write_audio(tmp_path / "audio" / "same.bin")
    out_dir = tmp_path / "features"
    encoder = FakeEncoder()
    extract_features(audio, out_dir, encoder, loader=FakeLoader(), resampler=FakeResampler())
    extract_features(
        audio,
        out_dir,
        encoder,
        loader=FakeLoader(),
        resampler=FakeResampler(),
        overwrite=True,
    )
    assert encoder.calls == 2


def test_frame_rate_drift_is_caught(tmp_path: Path) -> None:
    """帧率漂移哨兵：编码器报出的帧率与契约不符必须抛错（25 Hz 事故的同类防线）。"""
    with pytest.raises(FeatureCacheMismatchError, match="rate"):
        _extract(tmp_path, encoder=FakeEncoder(rate=MERT_FRAME_RATE_HZ / 3))


def test_feature_dim_mismatch_is_caught(tmp_path: Path) -> None:
    with pytest.raises(FeatureCacheMismatchError, match="feat_dim"):
        _extract(tmp_path, encoder=FakeEncoder(feat_dim=MERT_DEFAULT_FEAT_DIM // 2))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rate", MERT_FRAME_RATE_HZ / 3),
        ("sample_rate", MERT_SAMPLE_RATE_HZ // 2),
        ("layer", 6),
        ("model_rev", "other-revision"),
        ("duration_s", SECONDS * 3.0),
        ("feat_dim", MERT_DEFAULT_FEAT_DIM // 2),
        ("dtype", "float16"),
    ],
)
def test_tampered_meta_field_is_rejected(tmp_path: Path, field: str, value: Any) -> None:
    """M9：篡改任一元数据字段后加载必须抛错（六项校验各有负样本）。"""
    npz_path, meta_path, meta = _extract(tmp_path)
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    payload[field] = value
    meta_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FeatureCacheMismatchError):
        load_feature_cache(
            npz_path,
            meta_path,
            expected_layer=meta.layer,
            expected_model_rev=meta.model_rev,
            expected_dtype=meta.dtype,
            expected_adapter=meta.adapter,
        )


def test_truncated_embedding_is_rejected(tmp_path: Path) -> None:
    npz_path, meta_path, meta = _extract(tmp_path)
    payload = dict(np.load(npz_path, allow_pickle=False))
    truncated = payload["emb"][: max(1, payload["emb"].shape[0] // 2)]
    np.savez_compressed(npz_path, emb=truncated)
    with pytest.raises(FeatureCacheMismatchError, match="帧数"):
        load_feature_cache(npz_path, meta_path, expected_layer=meta.layer)


def test_meta_json_must_parse(tmp_path: Path) -> None:
    npz_path, meta_path, _meta = _extract(tmp_path)
    meta_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(FeatureCacheMismatchError):
        load_feature_cache(npz_path, meta_path)


def test_missing_sidecar_is_reported(tmp_path: Path) -> None:
    npz_path, meta_path, _meta = _extract(tmp_path)
    meta_path.unlink()
    with pytest.raises(FeatureCacheMismatchError):
        load_feature_cache(npz_path, meta_path)


def test_feature_cache_meta_rejects_extra_field() -> None:
    base = {
        "rate": MERT_FRAME_RATE_HZ,
        "sample_rate": MERT_SAMPLE_RATE_HZ,
        "layer": 12,
        "model_rev": "rev",
        "duration_s": SECONDS,
        "original_sample_rate": MERT_SAMPLE_RATE_HZ,
        "feat_dim": MERT_DEFAULT_FEAT_DIM,
        "dtype": "float32",
        "adapter": "none",
    }
    assert FeatureCacheMeta.model_validate(base).rate == MERT_FRAME_RATE_HZ
    with pytest.raises(ValidationError):
        FeatureCacheMeta.model_validate({**base, "unknown_field": 1})


def test_audio_cache_key_is_content_hash(tmp_path: Path) -> None:
    audio = _write_audio(tmp_path / "a.bin")
    assert audio_cache_key(audio) == audio_cache_key(audio)
    other = _write_audio(tmp_path / "b.bin", seconds=2.0)
    assert audio_cache_key(audio) != audio_cache_key(other)
