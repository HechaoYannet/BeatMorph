"""scripts/extract_features.py 的驱动逻辑测试（**不加载权重、不联网**）。

只覆盖「清单 → 唯一音频集合 → 待提取集合 → 报告落盘」这一层：音频去重的外键纪律
（内容 sha1 一旦与清单记录不一致就必须拒抽，否则缓存会挂到错音频上）、缺文件/缺字段的
记账、以及 `--dry-run` 的报账路径。真实 MERT 前向属 slow/gpu，不进默认 CI。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from beatmorph.data.phira.client import Manifest, Provenance, read_manifest, write_manifest
from scripts.extract_features import (
    EXIT_ARGS,
    EXIT_DATA,
    EXIT_OK,
    AudioTarget,
    _pending,
    build_parser,
    collect_targets,
    main,
)


def _provenance() -> Provenance:
    return Provenance(
        source="https://api.phira.cn/chart",
        query="unit-test",
        fetched_at="2026-09-27T00:00:00+00:00",
        purpose="train",
        script="scripts/fetch_phira.py",
        script_version="test",
    )


def _write_audio(root: Path, relative: str, payload: bytes) -> tuple[Path, str]:
    """按内容 sha1 落一个「音频」文件，返回 `(路径, sha1)`。"""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path, hashlib.sha1(payload).hexdigest()


def test_collect_targets_groups_same_audio_and_keeps_chart_ids(tmp_path: Path) -> None:
    """同曲多谱共享同一音频 → 只产出一个目标，但 chart_ids 全部留痕。"""
    _, key = _write_audio(tmp_path, "audio/a.mp3", b"same-audio")
    _, other = _write_audio(tmp_path, "audio/b.mp3", b"other-audio")
    rows = [
        {"chart_id": 1, "audio_path": "audio/a.mp3", "audio_sha1": key},
        {"chart_id": 2, "audio_path": "audio/a.mp3", "audio_sha1": key},
        {"chart_id": 3, "audio_path": "audio/b.mp3", "audio_sha1": other},
    ]
    targets, report = collect_targets(rows, tmp_path, verify_hash=True)
    assert [target.key for target in targets] == sorted([key, other])
    grouped = next(target for target in targets if target.key == key)
    assert grouped.chart_ids == (1, 2)
    assert report.rows == 3
    assert report.unique_keys == 2
    assert report.missing_audio == []
    assert report.key_mismatch == []
    assert report.no_audio == 0


def test_collect_targets_accounts_missing_and_captionless_rows(tmp_path: Path) -> None:
    """缺音频字段与文件缺失都要显式记账（不得静默丢样本）。"""
    _, key = _write_audio(tmp_path, "audio/a.mp3", b"payload")
    rows = [
        {"chart_id": 1, "audio_path": "audio/a.mp3", "audio_sha1": key},
        {"chart_id": 2, "audio_path": None},
        {"chart_id": 3, "audio_path": "audio/gone.mp3", "audio_sha1": "0" * 40},
    ]
    targets, report = collect_targets(rows, tmp_path, verify_hash=True)
    assert len(targets) == 1
    assert report.no_audio == 1
    assert report.missing_audio == [{"chart_id": 3, "audio_path": "audio/gone.mp3"}]


def test_collect_targets_refuses_key_mismatch(tmp_path: Path) -> None:
    """清单记录的 sha1 与文件实际内容不符 → **拒抽并记账**（缓存错位是静默故障）。"""
    _, actual = _write_audio(tmp_path, "audio/a.mp3", b"payload")
    rows = [{"chart_id": 1, "audio_path": "audio/a.mp3", "audio_sha1": "f" * 40}]
    targets, report = collect_targets(rows, tmp_path, verify_hash=True)
    assert targets == []
    assert report.key_mismatch[0]["actual"] == actual
    assert report.key_mismatch[0]["recorded"] == "f" * 40

    # 关了复核就按文件实际内容取键（此时不再宣称「与清单一致」）
    relaxed, _ = collect_targets(rows, tmp_path, verify_hash=False)
    assert [target.key for target in relaxed] == ["f" * 40]


def test_pending_skips_cached_and_overwrite_forces(tmp_path: Path) -> None:
    """有缓存的音频默认跳过；`overwrite` 时全部重抽。"""
    features = tmp_path / "features"
    features.mkdir()
    target = AudioTarget(key="k1", path=tmp_path / "audio/a.mp3", chart_ids=(1,), duration_s=1.0)
    assert _pending([target], features, overwrite=False) == [target]
    (features / "k1.npz").write_bytes(b"x")
    (features / "k1.meta.json").write_text("{}", encoding="utf-8")
    assert _pending([target], features, overwrite=False) == []
    assert _pending([target], features, overwrite=True) == [target]


def test_dry_run_writes_report_with_provenance(tmp_path: Path) -> None:
    """`--dry-run`：只报账、不加载权重，报告必须带 provenance（M8 硬约束③）。"""
    _, key = _write_audio(tmp_path, "audio/a.mp3", b"payload")
    charts = tmp_path / "charts.jsonl"
    write_manifest(
        Manifest(
            provenance=_provenance(),
            rows=[{"chart_id": 1, "audio_path": "audio/a.mp3", "audio_sha1": key}],
        ),
        charts,
    )
    code = main(["--root", str(tmp_path), "--dry-run"])
    assert code == EXIT_OK
    report = json.loads((tmp_path / "features_report.json").read_text(encoding="utf-8"))
    assert report["dry_run"] is True
    assert report["unique_audio"] == 1
    assert report["pending_before"] == 1
    assert report["extracted"] == 0
    assert report["provenance"]["script"] == "scripts/extract_features.py"
    assert report["sample_rate_hz"] > 0
    assert report["frame_rate_hz"] > 0
    # 没有缓存被写出来（dry-run 的语义）
    assert not (tmp_path / "features").exists()
    # 清单本身没被改动
    assert len(read_manifest(charts).rows) == 1


def test_missing_manifest_returns_data_error(tmp_path: Path) -> None:
    """没有谱面清单 → 退出码 3（不去猜一个数据源）。"""
    assert main(["--root", str(tmp_path), "--dry-run"]) == EXIT_DATA


def test_main_rejects_negative_limit(tmp_path: Path) -> None:
    """`--limit` 为负 → 参数错误退出码。"""
    assert main(["--root", str(tmp_path), "--limit", "-1"]) == EXIT_ARGS


def test_parser_defaults_are_offline_friendly() -> None:
    """默认 adapter=none（离线直出）、device=auto、fp16 开；dry-run 可关掉全部计算。"""
    args = build_parser().parse_args([])
    assert args.adapter == "none"
    assert args.device == "auto"
    assert args.no_fp16 is False
    assert args.dry_run is False
    assert args.layer == 12


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
