"""MERT 特征离线提取驱动脚本（plan 02 §3.6 / M9 的执行入口）。

输入是 `fetch_phira.py` 产出的谱面清单（`charts.jsonl`）：音频文件由清单的
`audio_path` 定位（那是**内容 sha1** 命名的文件），缓存键也取同一个 sha1 —— 于是
「同曲多谱只提取一次」是结构性的，而不是靠调用方记得去重。

    uv run python scripts/extract_features.py --dry-run          # 只报账，不动 GPU
    uv run python scripts/extract_features.py --limit 8          # 只抽 8 个音频（冒烟）
    uv run python scripts/extract_features.py                    # 全量抽取

缓存契约（plan 01 §3.3 + `FeatureCacheMeta`）：`<sha1>.npz` + `<sha1>.meta.json`，
加载时逐项校验（帧率 / 采样率 / 层 / 模型版本 / 时长→帧数 / 精度 / adapter）。
**帧率是派生量**（`MERT_SAMPLE_RATE_HZ / prod(conv_stride) = 75 Hz`），元数据里的 `rate`
一旦与契约不符即报错——这是「25 Hz 事故」的防线。

退出码：0 成功｜2 参数错误｜3 清单/数据错误｜4 有音频提取失败（已记账，可重跑续传）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from beatmorph.core.contracts import MERT_FRAME_RATE_HZ, MERT_SAMPLE_RATE_HZ
from beatmorph.core.logging import get_logger, setup_logging
from beatmorph.data.phira.client import (
    ManifestPurpose,
    Provenance,
    read_manifest,
    utc_now_iso,
)
from beatmorph.data.pipeline.embed import (
    FeatureCacheMismatchError,
    audio_cache_key,
    extract_features,
    feature_cache_paths,
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
    report_path = root / "features_report.json"
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """脚本入口：返回退出码。"""
    setup_logging()
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit < 0:
        logger.error("--limit 不能为负（退出码 %d）", EXIT_ARGS)
        return EXIT_ARGS
    code: int = run(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
