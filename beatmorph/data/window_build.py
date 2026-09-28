"""窗口预切缓存的构建器（一次性离线 pass；**不在训练路径上**）。

用法见 `beatmorph.cli.build_windows`（`beatmorph-build-windows`）。

## 为什么可以并行、而且必须并行

构建期要做的正是训练期那件贵事——逐窗口物化样本（实测每窗约 0.85 s）。但构建期有训练期
没有的两个条件：

1. **按行顺序遍历**（`window_start..stop` 落在连续的行上）⇒ `chart_cache` /
   `feature_cache` 的 LRU **必然命中**，整谱解析（49.1%）与整段解压（15.6%）
   每行只付一次；
2. 与 GPU 无关、与显存无关 ⇒ 可以按核数铺满。

全库 634 952 窗，去掉上面两项后每窗约 0.32 s ⇒ 单进程约 56 h，16 进程约 **3.5 h**。
这是**一次性**成本；索引缓存（36.5 min）每次开训都要付，本缓存的训练期收益是常驻的。

## 边界

- 不重抽特征、不改谱面文件、不动任何契约（与 RFC-0035 §4 的同一张表）；
- 产物**可删**：删掉目录即回到原路径（`ChartPairDataset` 会回落）；
- 产物**可校验**：`index.json` 带指纹，指纹不符一律拒绝使用（不是静默降级）；
- 产物**要么完整要么不存在**：先写 `<dir>.partial`，全部 shard 落盘并校验通过后
  才原子改名 ⇒ 中断只会留下一个 `.partial`（不写 `index.json`，读不出来）。
"""

from __future__ import annotations

import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from beatmorph.core.logging import get_logger
from beatmorph.data.window_cache import (
    DEFAULT_SHARD_WINDOWS,
    WindowCacheError,
    WindowCacheIndex,
    WindowCacheWriter,
    window_cache_directory,
    window_cache_fingerprint,
    windows_on_disk,
    write_index,
)
from beatmorph.field.grid import N_CHANNELS, N_SIDES

if TYPE_CHECKING:
    from collections.abc import Callable

    from beatmorph.data.dataset import DatasetConfig

logger = get_logger(__name__)

__all__ = ["WindowCacheBuildReport", "build_window_cache"]

#: 落盘音频 dtype。**恒为 float32**，不做 fp16 压缩：源特征缓存若是 fp16，先升到 float32
#: 再存是精确的；反过来（把 float32 压成 fp16）会**静默丢精度**，而省下的 122 GiB
#: 换一个「样本可能不再逐位一致」的风险不划算。
AUDIO_DTYPE: str = "float32"


@dataclass(frozen=True, slots=True)
class WindowCacheBuildReport:
    """构建结果（只有运维意义，不参与任何训练语义）。"""

    directory: Path
    fingerprint: str
    n_windows: int
    n_shards: int
    seconds: float
    degraded_windows: int

    def describe(self) -> str:
        """一行诊断文本。"""
        return (
            f"窗口缓存已构建：{self.directory} | 窗口 {self.n_windows} | shard {self.n_shards} | "
            f"耗时 {self.seconds / 60:.1f} min | 退化（r==0）窗口 {self.degraded_windows}"
        )


@dataclass(frozen=True, slots=True)
class _ShardJob:
    """一个 worker 负责的连续 shard 段（picklable：Windows 上走 spawn）。"""

    config: DatasetConfig
    directory: Path
    fingerprint: str
    first_shard: int
    window_start: int
    window_stop: int
    shard_windows: int
    t_bins: int
    x_bins: int
    sides: int
    channels: int
    feature_dim: int


def _run_shard_job(job: _ShardJob) -> tuple[int, int]:
    """物化一段窗口并落盘。**必须是模块级函数**（spawn 要按「模块 + 限定名」重新导入）。"""
    from beatmorph.data.dataset import ChartPairDataset

    dataset = ChartPairDataset(job.config)
    writer = WindowCacheWriter(
        job.directory,
        split=job.config.split,
        fingerprint=job.fingerprint,
        t_bins=job.t_bins,
        x_bins=job.x_bins,
        sides=job.sides,
        channels=job.channels,
        feature_dim=job.feature_dim,
        audio_dtype=AUDIO_DTYPE,
        shard_windows=job.shard_windows,
        start_shard=job.first_shard,
    )
    for index in range(job.window_start, job.window_stop):
        writer.add(dataset[index])
    return writer.close(), dataset.no_visible_context_fallbacks()


def build_window_cache(
    config: DatasetConfig,
    *,
    root: Path,
    seed: int,
    shard_windows: int = DEFAULT_SHARD_WINDOWS,
    jobs: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> WindowCacheBuildReport:
    """把一个 split 的窗口物化到 `root` 下（返回落盘报告）。

    Args:
        config: 数据集配置（与训练**同一份**；指纹覆盖它的一切语义字段）。
        root: 缓存根目录（训练侧用同一个 `data.window_cache_dir` 指过来）。
        seed: 运行种子——遮盖种子由 `(seed, row_index, window_index)` 派生，因此它进指纹。
        shard_windows: 每个 shard 的窗口数。
        jobs: 并行进程数（1 = 本进程直接跑）。
        progress: `(已完成窗口, 总窗口)` 回调（只用于打印进度）。

    Raises:
        ValueError: 索引里 0 个窗口。
        WindowCacheError: 落盘数与索引不符（宁可不写索引也不留半成品）。
    """
    from beatmorph.data.dataset import ChartPairDataset, load_pairs

    if shard_windows < 1:
        raise ValueError(f"shard_windows 必须 >= 1，得到 {shard_windows}")
    rows = load_pairs(config.manifest_path, config.split)
    if config.limit is not None:
        rows = rows[: config.limit]
    fingerprint = window_cache_fingerprint(config, rows, seed=seed)
    directory = window_cache_directory(root, config.split, fingerprint)

    probe = ChartPairDataset(config)
    n_windows = len(probe)
    if n_windows <= 0:
        raise ValueError("索引里有 0 个窗口：无法构建窗口缓存")
    feature_dim = int(probe[0].audio_emb.shape[1])
    n_shards = (n_windows + shard_windows - 1) // shard_windows

    staging = directory.with_name(directory.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    jobs = max(1, int(jobs))
    chunk = max(1, (n_shards + jobs - 1) // jobs)
    plan: list[_ShardJob] = []
    for first in range(0, n_shards, chunk):
        last = min(first + chunk, n_shards)
        plan.append(
            _ShardJob(
                config=config,
                directory=staging,
                fingerprint=fingerprint,
                first_shard=first,
                window_start=first * shard_windows,
                window_stop=min(last * shard_windows, n_windows),
                shard_windows=int(shard_windows),
                t_bins=int(config.t_window),
                x_bins=int(config.x_bins),
                sides=int(N_SIDES),
                channels=int(N_CHANNELS),
                feature_dim=feature_dim,
            )
        )
    logger.info(
        "构建窗口缓存：%d 个窗口 / %d 个 shard / %d 个进程 ⇒ %s",
        n_windows,
        n_shards,
        len(plan),
        directory,
    )

    started = time.perf_counter()
    written = 0
    degraded = 0
    if len(plan) == 1:
        count, fallbacks = _run_shard_job(plan[0])
        written, degraded = count, fallbacks
        if progress is not None:
            progress(written, n_windows)
    else:
        with ProcessPoolExecutor(max_workers=len(plan)) as pool:
            for count, fallbacks in pool.map(_run_shard_job, plan):
                written += count
                degraded += fallbacks
                if progress is not None:
                    progress(written, n_windows)

    index = WindowCacheIndex(
        split=config.split,
        fingerprint=fingerprint,
        n_windows=n_windows,
        shard_windows=int(shard_windows),
        t_bins=int(config.t_window),
        x_bins=int(config.x_bins),
        sides=int(N_SIDES),
        channels=int(N_CHANNELS),
        feature_dim=feature_dim,
        audio_dtype=AUDIO_DTYPE,
    )
    on_disk = windows_on_disk(staging, index)
    if on_disk != n_windows or written != n_windows:
        raise WindowCacheError(
            f"窗口缓存不完整：盘上 {on_disk} / 本次写入 {written} / 索引 {n_windows}"
            f"（未写 index.json，目录留在 {staging}）"
        )
    if directory.exists():
        shutil.rmtree(directory)
    staging.rename(directory)
    write_index(directory, index)
    return WindowCacheBuildReport(
        directory=directory,
        fingerprint=fingerprint,
        n_windows=n_windows,
        n_shards=n_shards,
        seconds=time.perf_counter() - started,
        degraded_windows=degraded,
    )
