"""窗口预切缓存的构建入口（`beatmorph-build-windows`；**离线一次性**，不在训练路径上）。

用法：

```powershell
.venv\\Scripts\\python.exe -m beatmorph.cli.build_windows --config-name phigros_masked --jobs 12
```

先跑一个有界切片确认（不碰全库）：`--limit 50`。

退出码与 `beatmorph-train` 同族（0 成功 / 2 参数 / 3 配置 / 7 构建失败）。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from beatmorph.core.logging import get_logger
from beatmorph.infra.config.loading import configs_dir, load_config
from beatmorph.infra.env_doctor import repo_root

logger = get_logger("cli.build_windows")

EXIT_OK = 0
EXIT_ARGS = 2
EXIT_CONFIG = 3
EXIT_BUILD = 7

#: 缓存根的默认位置：与索引缓存（`.dataset_index`）并列，都在清单所在目录下。
DEFAULT_CACHE_DIRNAME: str = "window_cache"


def build_parser() -> argparse.ArgumentParser:
    """构造参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="beatmorph-build-windows",
        description="把窗口样本预切落盘（训练期改为 mmap 读取）",
    )
    parser.add_argument("--config-name", default="phigros_masked", help="configs/ 下的配置名")
    parser.add_argument("--config-dir", default=None, help="配置目录（默认 <repo>/configs）")
    parser.add_argument("--split", default=None, help="split（默认取配置里的 data.split_train）")
    parser.add_argument(
        "--out", default=None, help=f"缓存根目录（默认 <清单目录>/{DEFAULT_CACHE_DIRNAME}）"
    )
    parser.add_argument("--jobs", type=int, default=1, help="并行进程数（纯 CPU，按核数给）")
    parser.add_argument(
        "--shard-windows", type=int, default=None, help="每个 shard 的窗口数（默认取模块常量 512）"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只物化前 N 行（**有界切片**，用于验收；默认用配置里的 data.max_samples）",
    )
    # 构建是**按行顺序**遍历的（每行约 96 个连续窗口）⇒ 两个 LRU 各留 1-2 条就够。
    # 实测（2026-09-28）：默认 chart_cache_size=8 让每个 worker 常驻多张解析后的谱面，
    # 20 进程时系统可提交内存只剩约 1.5 GiB ⇒ 逼近页文件。它们**不进指纹**（只影响命中率），
    # 因此单开构建用的旋钮，不动机器学习语义、不碰配置里的训练口径。
    parser.add_argument(
        "--chart-cache-size",
        type=int,
        default=None,
        help="行级谱面 LRU 容量（默认用 data.chart_cache_size）",
    )
    parser.add_argument(
        "--feature-cache-size",
        type=int,
        default=None,
        help="行级特征 LRU 容量（默认用 data.feature_cache_size）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """构建窗口缓存并打印报告（返回进程退出码）。"""
    from beatmorph.data.dataset import DatasetConfig
    from beatmorph.data.window_build import build_window_cache
    from beatmorph.data.window_cache import DEFAULT_SHARD_WINDOWS

    args = build_parser().parse_args(argv)
    # `--config-dir` 缺省时按仓库根解析（与 `beatmorph-train` 同一条口径）。
    directory = Path(args.config_dir) if args.config_dir else configs_dir(repo_root())
    try:
        cfg = load_config(config_name=args.config_name, directory=directory)
    except Exception as exc:
        logger.error("配置加载失败：%s", exc)
        return EXIT_CONFIG
    data = cfg.data
    split = args.split or data.split_train
    limit = args.limit if args.limit is not None else data.max_samples
    manifest = Path(data.manifest_path)
    root = Path(args.out) if args.out is not None else manifest.parent / DEFAULT_CACHE_DIRNAME
    # 用 `is None` 而不是 `or`：`or` 会把 `--shard-windows 0` 静默吞成默认值，
    # 于是下面的 `< 1` 校验对 0 **永远不可达**（本轮实测：测试里这条路会直接开一次全库构建）。
    shard_windows = DEFAULT_SHARD_WINDOWS if args.shard_windows is None else args.shard_windows
    if args.jobs < 1 or shard_windows < 1:
        logger.error("--jobs 与 --shard-windows 必须 >= 1")
        return EXIT_ARGS

    config = DatasetConfig(
        manifest_path=manifest,
        chart_dir=Path(data.chart_dir),
        feature_dir=Path(data.feature_dir),
        split=split,
        t_window=data.t_window,
        tau_end_s=data.tau_end_s,
        tau_end_policy=data.tau_end_policy,
        x_bins=data.x_bins,
        k_max=data.k_max,
        occlusion_ratio=data.occlusion_ratio,
        seed=cfg.optim.seed,
        limit=limit,
        chart_cache_size=(
            data.chart_cache_size if args.chart_cache_size is None else args.chart_cache_size
        ),
        feature_cache_size=(
            data.feature_cache_size if args.feature_cache_size is None else args.feature_cache_size
        ),
    )
    logger.info(
        "开始构建：split=%s 行上限=%s 并行=%d shard=%d ⇒ %s",
        split,
        limit,
        args.jobs,
        shard_windows,
        root,
    )
    # 首次会先建**索引缓存**（全库约 36.5 min，命中则秒级）——不提前说会看起来像卡死。
    logger.warning("提示：首次构建会先建索引缓存（全库约 36.5 min；已缓存则秒级），随后才是物化")
    started = time.perf_counter()
    try:
        report = build_window_cache(
            config,
            root=root,
            seed=cfg.optim.seed,
            shard_windows=shard_windows,
            jobs=args.jobs,
            progress=_progress,
        )
    except Exception as exc:
        logger.error("构建失败：%s", exc)
        return EXIT_BUILD
    logger.info("%s", report.describe())
    logger.info("总耗时 %.1f min", (time.perf_counter() - started) / 60)
    return EXIT_OK


def _progress(done: int, total: int, *, step: int = 2000) -> None:
    """每 `step` 个窗口打一行进度（构建要几小时，没有进度等于没有反馈）。"""
    if total and (done % step < 1 or done >= total):
        logger.info("进度 %d/%d（%.1f%%）", done, total, done / total * 100)


if __name__ == "__main__":
    sys.exit(main())
