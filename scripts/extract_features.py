"""MERT 特征离线提取驱动脚本（plan 02 §3.6 / M9 的执行入口）。

输入是 `fetch_phira.py` 产出的谱面清单（`charts.jsonl`）：音频文件由清单的
`audio_path` 定位（那是**内容 sha1** 命名的文件），缓存键也取同一个 sha1 —— 于是
「同曲多谱只提取一次」是结构性的，而不是靠调用方记得去重。

    uv run python scripts/extract_features.py --dry-run          # 只报账，不动 GPU
    uv run python scripts/extract_features.py --limit 8          # 只抽 8 个音频（冒烟）
    uv run python scripts/extract_features.py                    # 全量抽取（原串行路径）

提速路径（**全部默认关闭，不改默认行为**）：

    uv run python scripts/extract_features.py --pipeline         # 单进程流水线
    uv run python scripts/extract_features.py --pipeline --shard 0/2   # 两进程分片
    uv run python scripts/extract_features.py --pipeline --shard 1/2
    uv run python scripts/extract_features.py --keys-from keys.txt     # 显式键集合

`--pipeline` 把「解码+重采样」预取到后台线程、把「落盘」移出关键路径——实测这两项
合计占原串行单首耗时的 ~63%。`--shard i/N` 的分片是**按缓存键的 sha1 取模**，不是按
pending 列表切片：后者会随别的进程落盘而整体左移，导致两个进程抢同一个 key（缓存契约
要求同一 key 只能由一个进程写）。`.shard` 与 `--keys-from` 互斥。

缓存契约（plan 01 §3.3 + `FeatureCacheMeta`）：`<sha1>.npz` + `<sha1>.meta.json`，
加载时逐项校验（帧率 / 采样率 / 层 / 模型版本 / 时长→帧数 / 精度 / adapter）。
**帧率是派生量**（`MERT_SAMPLE_RATE_HZ / prod(conv_stride) = 75 Hz`），元数据里的 `rate`
一旦与契约不符即报错——这是「25 Hz 事故」的防线。

退出码：0 成功｜2 参数错误｜3 清单/数据错误｜4 有音频提取失败（已记账，可重跑续传）。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from beatmorph.core.contracts import MERT_FRAME_RATE_HZ, MERT_SAMPLE_RATE_HZ
from beatmorph.core.logging import get_logger, setup_logging
from beatmorph.data.phira.client import (
    ManifestPurpose,
    Provenance,
    read_manifest,
    sha1_hex,
    utc_now_iso,
)
from beatmorph.data.pipeline.embed import (
    FeatureCacheMismatchError,
    audio_cache_key,
    extract_features,
    feature_cache_paths,
    save_feature_cache,
)

logger = get_logger("scripts.extract_features")

SCRIPT_ID = "scripts/extract_features.py"
SCRIPT_VERSION = "1.0.0"

DEFAULT_ROOT = Path("data/processed")
#: 本地权重目录（存在即优先用它，避免离线环境再走一次网络）。
LOCAL_MODEL_DIR = Path("models/pretrained/m-a-p/MERT-v1-330M")
HF_MODEL_ID = "m-a-p/MERT-v1-330M"

EXIT_OK = 0
EXIT_ARGS = 2
EXIT_DATA = 3
EXIT_EXTRACT = 4


# ══════════════════════════════════════════════════════════════
# 音频加载（torchaudio → soundfile 兜底）
# ══════════════════════════════════════════════════════════════


def _load_audio(path: Path) -> tuple[Any, int]:
    """读取音频为 `(单声道波形, 原始采样率)`；torchaudio 失败时回落 soundfile。

    两者都返回 **torch 张量**（编码器接口要求），soundfile 得到 numpy 后显式转换。
    """
    import torch

    try:
        import torchaudio

        waveform, sample_rate = torchaudio.load(str(path))
        return waveform[0], int(sample_rate)
    except Exception as exc:
        logger.debug("torchaudio 读取失败 %s：%s；改试 soundfile", path.name, exc)

    import numpy as np
    import soundfile

    data, sample_rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    mono = data[:, 0] if data.shape[1] > 0 else data.reshape(-1)
    return torch.from_numpy(np.ascontiguousarray(mono)), int(sample_rate)


def _resample(wav: Any, source_rate: int, target_rate: int) -> Any:
    """重采样（只在本入口发生，plan 01 §4）；torchaudio 不可用时回落 soxr。"""
    import torch

    if source_rate == target_rate:
        return wav
    try:
        import torchaudio

        tensor = wav if isinstance(wav, torch.Tensor) else torch.as_tensor(wav)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        return torchaudio.functional.resample(tensor, source_rate, target_rate)
    except Exception as exc:
        logger.debug("torchaudio 重采样失败：%s；改试 soxr", exc)

    import numpy as np
    import soxr

    array = wav.detach().cpu().numpy() if isinstance(wav, torch.Tensor) else np.asarray(wav)
    resampled = soxr.resample(np.asarray(array).reshape(-1), source_rate, target_rate)
    return torch.from_numpy(np.ascontiguousarray(resampled, dtype=np.float32))


# ══════════════════════════════════════════════════════════════
# 清单 → 唯一音频集合
# ══════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class AudioTarget:
    """一个待提取的音频（按内容 sha1 唯一）。"""

    key: str
    path: Path
    chart_ids: tuple[int, ...]
    duration_s: float | None


@dataclass
class CollectReport:
    """清单扫描的记账（缺音频 / 键不一致都必须显式记账）。"""

    rows: int = 0
    unique_keys: int = 0
    missing_audio: list[dict[str, Any]] = field(default_factory=list)
    key_mismatch: list[dict[str, Any]] = field(default_factory=list)
    no_audio: int = 0


def collect_targets(
    rows: Sequence[Mapping[str, Any]],
    root: Path,
    *,
    verify_hash: bool,
) -> tuple[list[AudioTarget], CollectReport]:
    """把清单行归并成「唯一音频」列表（同 sha1 只留一条，chart_ids 记录来源）。

    Args:
        rows: 清单行。
        root: `audio_path` 的解析根（`data/processed`）。
        verify_hash: True = 重算文件 sha1 并**断言与清单记录一致**（默认开；
            一致性一旦破坏，特征缓存就会挂到错音频上，且没有任何下游能发现）。
    """
    report = CollectReport(rows=len(rows))
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        audio_path = row.get("audio_path")
        if not audio_path:
            report.no_audio += 1
            continue
        path = root / str(audio_path)
        if not path.is_file():
            report.missing_audio.append(
                {"chart_id": row.get("chart_id"), "audio_path": str(audio_path)}
            )
            continue
        recorded = row.get("audio_sha1") or row.get("feature_key")
        key = str(recorded) if recorded else path.stem
        if verify_hash:
            actual = audio_cache_key(path)
            if recorded and actual != str(recorded):
                report.key_mismatch.append(
                    {
                        "chart_id": row.get("chart_id"),
                        "audio_path": str(audio_path),
                        "recorded": str(recorded),
                        "actual": actual,
                    },
                )
                continue
            key = actual
        entry = grouped.setdefault(
            key,
            {"path": path, "chart_ids": [], "duration_s": row.get("audio_duration_s")},
        )
        chart_id = row.get("chart_id")
        if isinstance(chart_id, int):
            entry["chart_ids"].append(chart_id)
    report.unique_keys = len(grouped)
    targets = [
        AudioTarget(
            key=key,
            path=Path(entry["path"]),
            chart_ids=tuple(sorted(entry["chart_ids"])),
            duration_s=entry["duration_s"],
        )
        for key, entry in sorted(grouped.items())
    ]
    return targets, report


def _pending(
    targets: Sequence[AudioTarget], features_dir: Path, *, overwrite: bool
) -> list[AudioTarget]:
    """还没有缓存的音频（`overwrite` 时全部）。"""
    if overwrite:
        return list(targets)
    pending: list[AudioTarget] = []
    for target in targets:
        npz_path, meta_path = feature_cache_paths(features_dir, target.key)
        if not (npz_path.is_file() and meta_path.is_file()):
            pending.append(target)
    return pending


# ══════════════════════════════════════════════════════════════
# 子集选取（多进程分片）
# ══════════════════════════════════════════════════════════════


def stable_shard(key: str, count: int) -> int:
    """把缓存键**稳定地**映射到分片编号（与遍历顺序无关）。

    为什么不用 `pending[index::count]`：每个进程的 `pending` 是各自扫描出的快照，别的
    进程一落盘快照就变短，切片下标整体左移 ⇒ 两个进程会抢到同一个 key，而缓存契约要求
    同一个 key 只能由一个进程写。用键的 sha1 取模则与快照无关：各分片天然两两不相交，
    且在大数定律下分摊均衡。
    """
    return int(sha1_hex(key.encode("utf-8")), 16) % count


def select_subset(
    pending: Sequence[AudioTarget],
    *,
    shard: tuple[int, int] | None,
    keys_from: Path | None,
) -> list[AudioTarget]:
    """按 `--shard i/N` 或 `--keys-from` 取子集；两者都不给则原样返回。"""
    if shard is not None:
        index, count = shard
        picked = [target for target in pending if stable_shard(target.key, count) == index]
        logger.info(
            "分片 %d/%d：%d 个待提取中本片占 %d 个", index, count, len(pending), len(picked)
        )
        return picked
    if keys_from is not None:
        wanted = {
            line.strip()
            for line in Path(keys_from).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        picked = [target for target in pending if target.key in wanted]
        logger.info("--keys-from %s：%d 个待提取中命中 %d 个", keys_from, len(pending), len(picked))
        return picked
    return list(pending)


# ══════════════════════════════════════════════════════════════
# 编码器
# ══════════════════════════════════════════════════════════════


def build_encoder(args: argparse.Namespace) -> Any:
    """构造 MERT 编码器（默认离线直出：`adapter=none`，主干冻结）。"""
    from beatmorph.audio.encoder.mert import MERTAdapter

    model_name = args.model_path or (
        str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.is_dir() else HF_MODEL_ID
    )
    logger.info(
        "编码器：model=%s layer=%d adapter=%s device=%s fp16=%s",
        model_name,
        args.layer,
        args.adapter,
        args.device,
        not args.no_fp16,
    )
    return MERTAdapter(
        model_name=model_name,
        layer=args.layer,
        adapter=args.adapter,
        source=args.source,
        device=args.device,
        fp16=not args.no_fp16,
    )


# ══════════════════════════════════════════════════════════════
# 提速路径：预取解码 + 后台落盘（`--pipeline`）
# ══════════════════════════════════════════════════════════════

#: 实测单首阶段分解（n=12，均值，RTX 5070 Laptop 8GB / MERT-v1-330M fp16）：
#: 落盘 `np.savez_compressed` 1.03s（52%）｜GPU 前向 0.72s（36%）｜
#: 解码 0.18s（9%）｜重采样 0.02s（1%）｜H2D+D2H 0.01s｜整文件读 0.01s。
#: 结论：串行路径里 GPU 有 ~2/3 的时间在等 CPU（zlib 单线程压缩 + 音频解码）。


def _save_feature_cache_store(array: Any, npz_path: Path, meta_path: Path, meta: Any) -> None:
    """`np.savez`（不压缩）落盘：与压缩版**同格式同键**，只是不做 deflate。

    MERT fp16 特征的 deflate 压缩比只有 ~1.09×（22.1MB → 20.4MB），却要花掉单首
    ~0.8s 的单线程 zlib 时间；换成 store 后落盘降到 ~0.02s，代价是 npz 体积 +9%。
    读取侧完全不受影响（`np.load` 对两种写法一视同仁）。
    """
    import numpy as np

    np.savez(npz_path, emb=array)
    meta_path.write_text(meta.model_dump_json(indent=2), encoding="utf-8")
    logger.info(
        "已提取特征 %s（%d 帧 × %d 维，%.2fs）", npz_path.name, *array.shape, meta.duration_s
    )


def _make_decode() -> Callable[[Path], tuple[Any, int]]:
    """构造「读文件 + 重采样到模型采样率」的解码函数（在预取线程里跑）。

    返回值刻意与 `_resample` 的输入语义对齐：`(已重采样的波形, 原始采样率)`。
    `extract_features` 拿到它之后会把「原始采样率」原样写进 meta，重采样则变成恒等
    （见 :func:`_identity_resampler`），因此 meta 与串行路径逐字段一致。
    """

    def decode(path: Path) -> tuple[Any, int]:
        wav, original_sample_rate = _load_audio(path)
        resampled = _resample(wav, int(original_sample_rate), MERT_SAMPLE_RATE_HZ)
        return resampled, int(original_sample_rate)

    return decode


def _identity_resampler(wav: Any, source_rate: int, target_rate: int) -> Any:
    """重采样已经在预取线程里做完，这里把结果原样交回（绝不重采样第二次）。"""
    return wav


class _AsyncDecoder:
    """后台线程预取「解码 + 重采样」，有界窗口，满则回压到提交方。

    torchaudio 的读文件走 C++、重采样是 torch 张量算子，两者都放开 GIL，所以线程
    就够用，不必再为每个进程复制一份 torch/torchaudio 运行时。
    """

    def __init__(
        self, decode: Callable[[Path], tuple[Any, int]], *, workers: int, depth: int
    ) -> None:
        self._decode = decode
        self._depth = max(1, depth)
        self._pool = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="prefetch")
        self._futures: dict[Path, Future[tuple[Any, int]]] = {}
        self._order: deque[Path] = deque()

    def prefetch(self, path: Path) -> None:
        """把 path 排进预取窗口；窗口已满则先阻塞到最老的一个落地。

        ⚠️ 回压必须**把最老的一项移出窗口**（`popleft`）：它是「已经等完了」的那一项，
        留在窗口里会让 `len(self._order) >= self._depth` 永远成立 ⇒ 死循环空转
        （实测症状：单核 100% CPU、GPU 0%、无任何输出）。`_futures` 里仍保留它，
        后续 `take()` 照常取得到。
        """
        while len(self._order) >= self._depth:
            oldest = self._order.popleft()
            future = self._futures.get(oldest)
            if future is not None:
                future.exception()  # 阻塞至完成；异常在此不抛（由 take() 抛出）
        self._order.append(path)
        self._futures[path] = self._pool.submit(self._decode, path)

    def take(self, path: Path) -> tuple[Any, int]:
        """取回预取结果（必要时阻塞）；解码异常在此按调用方的异常口径抛出。"""
        future = self._futures.pop(path)
        with contextlib.suppress(ValueError):  # 只在窗口被回压时可能发生
            self._order.remove(path)
        return future.result()

    def close(self) -> None:
        """等在途预取结束并关闭线程池。"""
        for future in list(self._futures.values()):
            future.exception()
        self._futures.clear()
        self._order.clear()
        self._pool.shutdown(wait=True)


class _AsyncWriter:
    """把落盘搬到后台线程池（有界队列 + 回压），并回传线程内的异常。

    落盘是**纯 CPU**（单线程 zlib），与 GPU 无关；留在关键路径上等于让 GPU 干等。
    """

    def __init__(self, write: Callable[..., None], *, workers: int, depth: int) -> None:
        self._write = write
        self._depth = max(1, depth)
        self._pool = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="npz")
        self._queue: deque[Future[None]] = deque()
        self._errors: list[tuple[str, BaseException]] = []

    def __call__(self, array: Any, npz_path: Path, meta_path: Path, meta: Any) -> None:
        """落盘钩子（传给 :func:`extract_features` 的 `writer`）。"""
        while len(self._queue) >= self._depth:
            self._queue.popleft().exception()  # 回压：等最老的一次落盘结束
        self._queue.append(self._pool.submit(self._write_guarded, array, npz_path, meta_path, meta))

    def _write_guarded(self, array: Any, npz_path: Path, meta_path: Path, meta: Any) -> None:
        """线程体：异常必须捕获后回传主线程，否则就是静默丢数据。"""
        try:
            self._write(array, npz_path, meta_path, meta)
        except BaseException as exc:  # 线程内异常只能这样回传主线程
            self._errors.append((npz_path.name, exc))

    def drain(self) -> list[tuple[str, BaseException]]:
        """等所有在途落盘结束并返回 `(文件名, 异常)` 列表。"""
        for future in list(self._queue):
            future.exception()
        self._queue.clear()
        self._pool.shutdown(wait=True)
        return list(self._errors)


def install_batched_forward(encoder: Any, max_batch: int) -> None:
    """把 5s 滑窗的**满窗段**堆成一个 batch 过一次主干（减少 kernel 发射次数）。

    动机（实测）：单首有 31–35 个 5s 段，每段一次 `backbone()`；单首约 1.8 万个 CUDA
    kernel、平均 kernel 仅 ~23µs ⇒ 前向是「发射受限」而不是算力受限。合并 batch 后
    单首前向 0.82s → 0.63s（batch=8）→ 0.59s（batch=24）。

    ⚠️ 与逐段前向**不是逐位一致**：不同 batch 尺寸走不同 cuDNN kernel，尾数会差
    （量级见基准报告）。故默认关闭，需显式 `--batch-chunks N` 才启用。
    尾部短段不参与堆批（避免 padding 改变注意力可见范围），仍走 `_encode_chunk`。
    """
    import torch

    from beatmorph.audio.encoder.mert import _OVERLAP_S, _TARGET_SR, _WINDOW_S, MERTAdapter

    win = int(_WINDOW_S * _TARGET_SR)
    hop = int((_WINDOW_S - _OVERLAP_S) * _TARGET_SR)

    @torch.inference_mode()
    def forward(self: Any, wav: Any) -> Any:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        device = next(self.parameters()).device
        wav = wav.to(device)
        if self.fp16:
            wav = wav.half()
        seq_len = wav.shape[1]
        if seq_len <= win:
            return self._encode_chunk(wav)
        chunks: list[Any] = []
        starts: list[int] = []
        pos = 0
        while pos < seq_len:
            end = min(pos + win, seq_len)
            chunks.append(wav[:, pos:end])
            starts.append(pos)
            if end >= seq_len:
                break
            pos += hop
        outs: list[Any] = []
        index = 0
        while index < len(chunks):
            end_index = index
            while (
                end_index < len(chunks)
                and chunks[end_index].shape[1] == win
                and end_index - index < max_batch
            ):
                end_index += 1
            group = chunks[index:end_index]
            if len(group) == 1:
                outs.append(self._encode_chunk(group[0]))
            else:
                dtype = torch.float16 if self.fp16 else torch.float32
                # 逐段过 _feat_processor（本模型 do_normalize=false，等价于恒等包装），
                # 再把已归一化的 input_values 堆成 batch —— 归一化语义与逐段调用一致。
                values = [
                    self._feat_processor(
                        chunk.squeeze(0), sampling_rate=_TARGET_SR, return_tensors="pt"
                    )["input_values"]
                    for chunk in group
                ]
                stacked = torch.cat([value.reshape(1, -1) for value in values], dim=0)
                stacked = stacked.to(dtype=dtype, device=wav.device)
                hidden = self.backbone(stacked, output_hidden_states=True).hidden_states
                emb = hidden[self.layer].to(dtype)
                outs.extend(emb[offset : offset + 1] for offset in range(len(group)))
            index = end_index
        return self._merge_overlapping(outs, starts, hop, self.output_frame_rate())

    if encoder.adapter == "mlp":
        logger.warning("--batch-chunks 对 mlp adapter 不生效（退回逐段前向）")
        return
    encoder.forward = forward.__get__(encoder, MERTAdapter)


# ══════════════════════════════════════════════════════════════
# 提取主循环（串行 / 流水线）
# ══════════════════════════════════════════════════════════════


def _extract_serial(
    args: argparse.Namespace,
    pending: Sequence[AudioTarget],
    features_dir: Path,
    encoder: Any,
    payload: dict[str, Any],
) -> None:
    """原串行路径：解码 → 重采样 → 前向 → 落盘，一步一步来（默认行为，未改动）。"""
    for position, target in enumerate(pending, start=1):
        try:
            meta = extract_features(
                target.path,
                features_dir,
                encoder,
                layer=args.layer,
                model_rev=args.model_rev,
                adapter=args.adapter,
                key=target.key,
                loader=_load_audio,
                resampler=_resample,
                overwrite=args.overwrite,
            )
            payload["extracted"] += 1
            logger.info(
                "[%d/%d] %s：%.1fs → %d 帧 @ %.1f Hz（同曲谱面 %d 张）",
                position,
                len(pending),
                target.key[:12],
                meta.duration_s,
                round(meta.duration_s * meta.rate),
                meta.rate,
                len(target.chart_ids),
            )
        except (FeatureCacheMismatchError, OSError, RuntimeError, ValueError) as exc:
            logger.error("[%d/%d] 提取失败 %s：%s", position, len(pending), target.key[:12], exc)
            payload["failures"].append(
                {"key": target.key, "audio": str(target.path), "reason": str(exc)}
            )


def _extract_pipelined(
    args: argparse.Namespace,
    pending: Sequence[AudioTarget],
    features_dir: Path,
    encoder: Any,
    payload: dict[str, Any],
) -> None:
    """三阶段流水线：预取解码（线程）→ GPU 前向（主线程）→ 落盘（线程）。

    串行时单首墙钟 = 三者**之和**；流水线把前两者移出关键路径后变成三者**取最大**，
    理论上限 ≈ 1 / 0.72s ≈ 2.7×。缓存契约、meta 字段、退出码语义全部不变。
    """
    write = _save_feature_cache_store if args.npz_compression == "store" else save_feature_cache
    decoder = _AsyncDecoder(_make_decode(), workers=args.prefetch, depth=args.prefetch_depth)
    writer = _AsyncWriter(write, workers=args.writers, depth=args.writer_depth)
    paths = [target.path for target in pending]
    total = len(pending)
    # 初始窗口 = 前 depth 项；之后每消费一项就补一项，窗口始终覆盖「接下来的 depth 项」。
    # 注意这里是 **position + depth - 1**（position 从 1 起）：写成 position + depth 会跳过
    # 索引恰为 depth 的那一项，使 take() 在 _futures 里找不到它（KeyError）。
    for path in paths[: args.prefetch_depth]:
        decoder.prefetch(path)

    started = time.monotonic()
    for position, target in enumerate(pending, start=1):
        ahead = position + args.prefetch_depth - 1
        if ahead < total:
            decoder.prefetch(paths[ahead])
        try:
            wav_model_rate, original_sample_rate = decoder.take(target.path)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.error("[%d/%d] 解码失败 %s：%s", position, total, target.key[:12], exc)
            payload["failures"].append(
                {"key": target.key, "audio": str(target.path), "reason": str(exc)}
            )
            continue
        try:
            meta = extract_features(
                target.path,
                features_dir,
                encoder,
                layer=args.layer,
                model_rev=args.model_rev,
                adapter=args.adapter,
                key=target.key,
                loader=lambda _path, _w=wav_model_rate, _r=original_sample_rate: (_w, _r),
                resampler=_identity_resampler,
                overwrite=args.overwrite,
                writer=writer,
            )
            payload["extracted"] += 1
            logger.info(
                "[%d/%d] %s：%.1fs → %d 帧 @ %.1f Hz（同曲谱面 %d 张）",
                position,
                total,
                target.key[:12],
                meta.duration_s,
                round(meta.duration_s * meta.rate),
                meta.rate,
                len(target.chart_ids),
            )
        except (FeatureCacheMismatchError, OSError, RuntimeError, ValueError) as exc:
            logger.error("[%d/%d] 提取失败 %s：%s", position, total, target.key[:12], exc)
            payload["failures"].append(
                {"key": target.key, "audio": str(target.path), "reason": str(exc)}
            )
        if args.log_every > 0 and (position % args.log_every == 0 or position == total):
            elapsed = time.monotonic() - started
            per_song = elapsed / position
            rate_per_min = 60.0 / per_song if per_song > 0 else 0.0
            logger.info(
                "进度 %d/%d：%.3f s/首（%.1f 首/分钟），已用 %.0fs，本片预计剩余 %.0fs",
                position,
                total,
                per_song,
                rate_per_min,
                elapsed,
                (total - position) * per_song,
            )

    decoder.close()
    for name, exc in writer.drain():
        logger.error("落盘失败 %s：%s", name, exc)
        payload["failures"].append(
            {"key": Path(name).stem, "audio": "", "reason": f"落盘失败：{exc}"}
        )


# ══════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════


def _provenance(manifest: Path, args: argparse.Namespace, source_note: str) -> Provenance:
    """特征提取报告的 provenance（M8 硬约束③）。"""
    return Provenance(
        source=source_note,
        query=(
            f"manifest={manifest}；layer={args.layer} adapter={args.adapter} "
            f"model={args.model_path or HF_MODEL_ID} device={args.device}"
        ),
        fetched_at=utc_now_iso(),
        purpose=ManifestPurpose.TRAIN,
        script=SCRIPT_ID,
        script_version=SCRIPT_VERSION,
    )


def run(args: argparse.Namespace) -> int:
    """执行一轮特征提取（可续跑：已有缓存默认跳过）。"""
    root: Path = args.root
    manifest_path: Path = args.manifest or (root / "charts.jsonl")
    features_dir: Path = args.features_dir or (root / "features")
    if not manifest_path.is_file():
        logger.error(
            "清单不存在：%s（先跑 scripts/fetch_phira.py fetch；退出码 %d）",
            manifest_path,
            EXIT_DATA,
        )
        return EXIT_DATA

    manifest = read_manifest(manifest_path)
    targets, report = collect_targets(manifest.rows, root, verify_hash=not args.skip_hash_check)
    pending = _pending(targets, features_dir, overwrite=args.overwrite)
    pending = select_subset(pending, shard=args.shard, keys_from=args.keys_from)
    if args.limit is not None:
        pending = pending[: args.limit]
    logger.info(
        "清单 %d 行 → 唯一音频 %d 个（无音频字段 %d、文件缺失 %d、sha1 不一致 %d）；待提取 %d 个",
        report.rows,
        report.unique_keys,
        report.no_audio,
        len(report.missing_audio),
        len(report.key_mismatch),
        len(pending),
    )
    for note in report.missing_audio[:10]:
        logger.warning("音频文件缺失：%s", note)
    for note in report.key_mismatch[:10]:
        logger.error("音频 sha1 与清单记录不一致（拒绝提取，避免把缓存挂到错音频上）：%s", note)

    payload: dict[str, Any] = {
        "provenance": _provenance(
            manifest_path,
            args,
            source_note=f"{manifest.provenance.source}（清单 provenance 见 {manifest_path}）",
        ).model_dump(mode="json"),
        "manifest": str(manifest_path),
        "features_dir": str(features_dir),
        "rows": report.rows,
        "unique_audio": report.unique_keys,
        "pending_before": len(pending),
        "extracted": 0,
        "cache_hits": len(targets) - len(_pending(targets, features_dir, overwrite=False)),
        "no_audio_rows": report.no_audio,
        "missing_audio": report.missing_audio,
        "key_mismatch": report.key_mismatch,
        "failures": [],
        "frame_rate_hz": MERT_FRAME_RATE_HZ,
        "sample_rate_hz": MERT_SAMPLE_RATE_HZ,
        "elapsed_s": 0.0,
    }
    report_path = args.report_path or (root / "features_report.json")
    if args.shard is not None and args.report_path is None:
        # 分片进程各自记账：绝不能互相覆盖同一份 report
        shard_index, shard_count = args.shard
        report_path = root / f"features_report.shard{shard_index}of{shard_count}.json"
    if args.dry_run:
        payload["dry_run"] = True
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("--dry-run：只报账，不加载权重、不写缓存 → %s", report_path)
        return EXIT_OK
    if not pending:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("没有待提取的音频（全部命中缓存）→ %s", report_path)
        return EXIT_OK

    started = time.monotonic()
    encoder = build_encoder(args)
    features_dir.mkdir(parents=True, exist_ok=True)
    if args.batch_chunks > 1:
        install_batched_forward(encoder, args.batch_chunks)
    if args.pipeline:
        _extract_pipelined(args, pending, features_dir, encoder, payload)
    else:
        _extract_serial(args, pending, features_dir, encoder, payload)
    payload["elapsed_s"] = round(time.monotonic() - started, 1)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "提取完成：新增 %d、失败 %d、用时 %.0fs → %s",
        payload["extracted"],
        len(payload["failures"]),
        payload["elapsed_s"],
        report_path,
    )
    return EXIT_EXTRACT if payload["failures"] else EXIT_OK


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════


def _parse_shard(text: str) -> tuple[int, int]:
    """解析 `--shard i/N`（i 从 0 起）。"""
    try:
        index_text, count_text = text.split("/", 1)
        index, count = int(index_text), int(count_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--shard 形如 i/N，得到 {text!r}") from exc
    if count < 1 or not 0 <= index < count:
        raise argparse.ArgumentTypeError(f"--shard 需要 0 <= i < N，得到 {text!r}")
    return index, count


def build_parser() -> argparse.ArgumentParser:
    """命令行解析器。"""
    parser = argparse.ArgumentParser(
        prog="extract_features.py", description="MERT 特征离线提取驱动脚本"
    )
    parser.add_argument(
        "--root", type=Path, default=DEFAULT_ROOT, help="产物根（默认 data/processed）"
    )
    parser.add_argument(
        "--manifest", type=Path, default=None, help="谱面清单（默认 <root>/charts.jsonl）"
    )
    parser.add_argument(
        "--features-dir", type=Path, default=None, help="缓存目录（默认 <root>/features）"
    )
    parser.add_argument("--limit", type=int, default=None, help="最多提取 N 个新音频（默认全部）")
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="记账 JSON 的落点（默认 <root>/features_report.json；分片时自动带 shard 后缀）",
    )
    parser.add_argument("--overwrite", action="store_true", help="忽略已有缓存，重抽")
    parser.add_argument("--dry-run", action="store_true", help="只报账：不加载权重、不写缓存")
    parser.add_argument("--layer", type=int, default=12, help="取第几层 hidden state（默认 12）")
    parser.add_argument(
        "--adapter", default="none", choices=["none", "lora", "mlp"], help="离线直出用 none"
    )
    parser.add_argument(
        "--model-path", default=None, help=f"本地权重目录（默认 {LOCAL_MODEL_DIR}）"
    )
    parser.add_argument(
        "--model-rev", default="m-a-p/MERT-v1-330M", help="写进缓存元数据的模型版本"
    )
    parser.add_argument("--source", default="huggingface", choices=["huggingface", "modelscope"])
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / cuda:0")
    parser.add_argument("--no-fp16", action="store_true", help="关闭 FP16 推理")
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="跳过音频 sha1 复核（**不建议**：缓存键与音频一旦错位，下游无法察觉）",
    )
    subset = parser.add_argument_group("子集（多进程分片）")
    subset.add_argument(
        "--shard",
        type=_parse_shard,
        default=None,
        metavar="I/N",
        help="只处理按缓存键 sha1 取模落到第 I 片（共 N 片）的音频；与 --keys-from 互斥",
    )
    subset.add_argument(
        "--keys-from",
        type=Path,
        default=None,
        help="只处理文件里列出的缓存键（每行一个，# 开头为注释）",
    )
    speed = parser.add_argument_group("提速路径（默认全部关闭，不改默认行为）")
    speed.add_argument(
        "--pipeline",
        action="store_true",
        help="启用流水线：解码预取到后台线程 + 落盘移出关键路径（实测约 2.2×）",
    )
    speed.add_argument("--prefetch", type=int, default=2, help="预取线程数（默认 2）")
    speed.add_argument("--prefetch-depth", type=int, default=4, help="预取窗口深度（默认 4）")
    speed.add_argument("--writers", type=int, default=2, help="落盘线程数（默认 2）")
    speed.add_argument("--writer-depth", type=int, default=4, help="落盘队列深度（默认 4）")
    speed.add_argument(
        "--npz-compression",
        choices=["deflate", "store"],
        default="deflate",
        help="npz 落盘方式：deflate=与现状逐位一致（默认）；store=不压缩，快 ~39×，体积 +9%%",
    )
    speed.add_argument(
        "--batch-chunks",
        type=int,
        default=1,
        help="把滑窗的满窗段堆成 batch（>1 启用）。注意：与逐段前向非逐位一致，默认 1",
    )
    speed.add_argument(
        "--log-every", type=int, default=25, help="流水线模式下每 N 首打一条吞吐进度"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """脚本入口：返回退出码。"""
    setup_logging()
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit < 0:
        logger.error("--limit 不能为负（退出码 %d）", EXIT_ARGS)
        return EXIT_ARGS
    if args.keys_from is not None and not args.keys_from.is_file():
        logger.error("--keys-from 文件不存在：%s（退出码 %d）", args.keys_from, EXIT_ARGS)
        return EXIT_ARGS
    if args.shard is not None and args.keys_from is not None:
        logger.error("--shard 与 --keys-from 互斥（退出码 %d）", EXIT_ARGS)
        return EXIT_ARGS
    code: int = run(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
