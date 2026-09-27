"""流水线提取（--pipeline / --shard）与落盘钩子的契约测试（**不加载权重、不上 GPU**）。

覆盖三件容易「改了但没人发现」的事：
1. extract_features 的 writer 钩子与默认落盘**逐字节一致**（否则新路径产物会悄悄漂移）；
2. --shard i/N 的分片是**两两不相交且并集完整**的（缓存契约要求同一 key 只由一个进程写）；
3. 流水线与串行路径在同一批输入上产出**同名同内容**的缓存。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from beatmorph.core.contracts import MERT_DEFAULT_FEAT_DIM, MERT_FRAME_RATE_HZ, MERT_SAMPLE_RATE_HZ
from beatmorph.data.pipeline.embed import (
    extract_features,
    feature_cache_paths,
    save_feature_cache,
)
from scripts.extract_features import (
    EXIT_ARGS,
    AudioTarget,
    _extract_pipelined,
    _extract_serial,
    _parse_shard,
    build_parser,
    select_subset,
    stable_shard,
)


class FakeEncoder:
    """假 MERT 编码器（与 tests/unit/data/test_embed_features.py 同款口径，不加载权重）。"""

    def __init__(self, *, rate: float = MERT_FRAME_RATE_HZ) -> None:
        self.rate = rate

    def encode(self, wav: Any, sample_rate: int = MERT_SAMPLE_RATE_HZ) -> Any:
        assert sample_rate == MERT_SAMPLE_RATE_HZ
        samples = int(np.asarray(wav).shape[-1])
        frames = round(samples / MERT_SAMPLE_RATE_HZ * self.rate)
        # 用可复现的确定值，便于比对两个目录的字节
        base = np.arange(frames * MERT_DEFAULT_FEAT_DIM, dtype=np.float32)
        return (base % 7.0).reshape(1, frames, MERT_DEFAULT_FEAT_DIM)

    def output_frame_rate(self) -> float:
        return self.rate


def _loader(path: Path) -> tuple[Any, int]:
    """假加载器：样本数 = 文件字节数，原始采样率取契约采样率。"""
    return np.zeros(Path(path).stat().st_size, dtype=np.float32), MERT_SAMPLE_RATE_HZ


def _resampler(wav: Any, source_rate: int, target_rate: int) -> Any:
    """假重采样器（源采样率已是目标采样率，恒等即可）。"""
    return np.asarray(wav)


def _make_target(root: Path, key: str, seconds: float = 1.0) -> AudioTarget:
    """造一个内容确定的假音频文件与对应目标。"""
    path = root / "audio" / (key + ".bin")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(round(seconds * MERT_SAMPLE_RATE_HZ)))
    return AudioTarget(key=key, path=path, chart_ids=(1,), duration_s=seconds)


def _payload() -> dict[str, Any]:
    """空记账表。"""
    return {"extracted": 0, "failures": []}


def _read_meta(out_dir: Path, key: str) -> dict[str, Any]:
    """读一份缓存 meta。"""
    _npz, meta_path = feature_cache_paths(out_dir, key)
    return json.loads(meta_path.read_text(encoding="utf-8"))


# ── ① writer 钩子 ─────────────────────────────────────────────


def test_default_path_equals_explicit_save_feature_cache(tmp_path: Path) -> None:
    """不传 writer 时，落盘结果与显式调用 save_feature_cache 逐字节一致。"""
    target = _make_target(tmp_path, "aaaaaaaa")
    encoder = FakeEncoder()
    plain = tmp_path / "plain"
    explicit = tmp_path / "explicit"

    extract_features(
        target.path, plain, encoder, key=target.key, loader=_loader, resampler=_resampler
    )
    extract_features(
        target.path,
        explicit,
        encoder,
        key=target.key,
        loader=_loader,
        resampler=_resampler,
        writer=save_feature_cache,
    )

    for suffix in (".npz", ".meta.json"):
        assert (plain / (target.key + suffix)).read_bytes() == (
            explicit / (target.key + suffix)
        ).read_bytes()


def test_writer_hook_receives_array_paths_and_meta(tmp_path: Path) -> None:
    """writer 钩子拿到 (数组, npz 路径, meta 路径, meta)，且默认落盘**不再发生**。"""
    target = _make_target(tmp_path, "bbbbbbbb")
    out_dir = tmp_path / "hooked"
    seen: list[tuple[Any, Path, Path, Any]] = []

    def hook(array: Any, npz_path: Path, meta_path: Path, meta: Any) -> None:
        seen.append((array, npz_path, meta_path, meta))

    meta = extract_features(
        target.path,
        out_dir,
        FakeEncoder(),
        key=target.key,
        loader=_loader,
        resampler=_resampler,
        writer=hook,
    )

    assert len(seen) == 1
    array, npz_path, meta_path, hook_meta = seen[0]
    assert array.shape == (round(meta.duration_s * meta.rate), MERT_DEFAULT_FEAT_DIM)
    assert npz_path == out_dir / (target.key + ".npz")
    assert meta_path == out_dir / (target.key + ".meta.json")
    assert hook_meta == meta
    # 钩子没落盘 → 文件不该出现（这正是「把落盘交给别人」的语义）
    assert not npz_path.exists()


# ── ② 分片 ────────────────────────────────────────────────────


def test_stable_shard_is_a_partition() -> None:
    """任意数量的分片必须「两两不相交、并集完整」——否则两个进程会写同一个 key。"""
    keys = [format(index, "040x") for index in range(2000)]
    for count in (1, 2, 3, 5, 8):
        buckets = [
            [key for key in keys if stable_shard(key, count) == index] for index in range(count)
        ]
        flattened = [key for bucket in buckets for key in bucket]
        assert sorted(flattened) == sorted(keys)
        assert len(flattened) == len(set(flattened))
    # 分摊均衡：2 片时各路不少于 40%（纯哈希，允许抖动）
    two = [sum(1 for key in keys if stable_shard(key, 2) == index) for index in (0, 1)]
    assert min(two) / len(keys) > 0.4


def test_select_subset_shard_covers_everything_exactly_once(tmp_path: Path) -> None:
    """select_subset 分片的并集 == 全集，且两两不交。"""
    targets = [_make_target(tmp_path, format(index, "040x")) for index in range(60)]
    shards = [select_subset(targets, shard=(index, 3), keys_from=None) for index in range(3)]
    flattened = [target.key for shard in shards for target in shard]
    assert sorted(flattened) == sorted(target.key for target in targets)
    assert all(not (set(a) & set(b)) for i, a in enumerate(shards) for b in shards[i + 1 :])


def test_select_subset_keys_from_reads_file_and_skips_comments(tmp_path: Path) -> None:
    """--keys-from 只认文件里列出的键，注释与空行忽略。"""
    targets = [_make_target(tmp_path, format(index, "040x")) for index in range(5)]
    keys_file = tmp_path / "keys.txt"
    keys_file.write_text(
        "# 注释行\n" + targets[1].key + "\n\n" + targets[3].key + "\n", encoding="utf-8"
    )
    picked = select_subset(targets, shard=None, keys_from=keys_file)
    assert [target.key for target in picked] == [targets[1].key, targets[3].key]


def test_select_subset_without_filters_returns_all(tmp_path: Path) -> None:
    """两个筛选都不给时必须原样返回（默认行为不变的守门测试）。"""
    targets = [_make_target(tmp_path, format(index, "040x")) for index in range(4)]
    assert select_subset(targets, shard=None, keys_from=None) == targets


def test_parse_shard_rejects_bad_values() -> None:
    """--shard 只接受 0 <= i < N。"""
    assert _parse_shard("1/3") == (1, 3)
    for bad in ("3/3", "-1/2", "0/0", "abc", "1"):
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_shard(bad)


def test_cli_rejects_shard_with_keys_from(tmp_path: Path) -> None:
    """--shard 与 --keys-from 互斥（同时给无法保证不相交）。"""
    from scripts.extract_features import main

    keys_file = tmp_path / "keys.txt"
    keys_file.write_text("x", encoding="utf-8")
    code = main(["--keys-from", str(keys_file), "--shard", "0/2"])
    assert code == EXIT_ARGS


# ── ③ 串行 vs 流水线 ─────────────────────────────────────────


@pytest.fixture()
def fake_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    """把脚本里的真实 torchaudio 通路换成假实现（本测试不加载 torch）。"""
    import scripts.extract_features as module

    monkeypatch.setattr(module, "_load_audio", _loader)
    monkeypatch.setattr(module, "_resample", _resampler)


def test_pipeline_matches_serial_byte_for_byte(tmp_path: Path, fake_audio: None) -> None:
    """同一批输入下，流水线与串行路径产出同名、同内容、同 meta 的缓存。"""
    targets = [_make_target(tmp_path, format(index, "040x")) for index in range(6)]
    serial_dir = tmp_path / "serial"
    pipeline_dir = tmp_path / "pipeline"
    serial_args = build_parser().parse_args(["--features-dir", str(serial_dir)])
    pipeline_args = build_parser().parse_args(["--features-dir", str(pipeline_dir), "--pipeline"])

    serial_payload = _payload()
    _extract_serial(serial_args, targets, serial_dir, FakeEncoder(), serial_payload)
    pipe_payload = _payload()
    _extract_pipelined(pipeline_args, targets, pipeline_dir, FakeEncoder(), pipe_payload)

    assert serial_payload["extracted"] == len(targets)
    assert pipe_payload["extracted"] == len(targets)
    assert serial_payload["failures"] == []
    assert pipe_payload["failures"] == []
    for target in targets:
        npz_a, meta_a = feature_cache_paths(serial_dir, target.key)
        npz_b, meta_b = feature_cache_paths(pipeline_dir, target.key)
        assert npz_a.read_bytes() == npz_b.read_bytes()
        assert json.loads(meta_a.read_text(encoding="utf-8")) == json.loads(
            meta_b.read_text(encoding="utf-8")
        )
        # meta 的九项校验键一个都不能少、不能变
        meta = _read_meta(serial_dir, target.key)
        assert set(meta) == {
            "rate",
            "sample_rate",
            "layer",
            "model_rev",
            "duration_s",
            "original_sample_rate",
            "feat_dim",
            "dtype",
            "adapter",
        }


def test_pipeline_records_decode_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """解码在预取线程里失败时，必须记账到 failures 而不是静默丢样本。"""
    import scripts.extract_features as module

    good = _make_target(tmp_path, "c" * 12)

    def boom(_path: Path) -> tuple[Any, int]:
        raise OSError("假解码失败")

    monkeypatch.setattr(module, "_load_audio", boom)
    monkeypatch.setattr(module, "_resample", _resampler)

    out_dir = tmp_path / "fail"
    args = build_parser().parse_args(["--features-dir", str(out_dir), "--pipeline"])
    payload = _payload()
    _extract_pipelined(args, [good], out_dir, FakeEncoder(), payload)

    assert payload["extracted"] == 0
    assert len(payload["failures"]) == 1
    assert payload["failures"][0]["key"] == good.key
