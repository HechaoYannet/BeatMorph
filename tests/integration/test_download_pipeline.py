"""sayobot 下载 → 预处理管线 端到端集成测试（**联网，gated**）。

标记 ``@pytest.mark.integration`` + ``@pytest.mark.slow``（Makefile ``test-fast``
仅排除 slow/gpu/e2e，**不含 integration**，故须叠加 slow 防止 CI 触网）；
并要求 ``BEATMORPH_LIVE_NETWORK=1`` 显式开启，否则 skip。

真实抓取 1 个 mania set → 解压 → 验 manifest 行 + .osu 落盘 → 喂
PreprocessPipeline 验证 §4.3 过滤对真实数据生效。
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
]


def _live_network_enabled() -> bool:
    return os.environ.get("BEATMORPH_LIVE_NETWORK") == "1"


@pytest.fixture()
def _require_network() -> None:
    if not _live_network_enabled():
        pytest.skip("set BEATMORPH_LIVE_NETWORK=1 to run live sayobot tests")
    pytest.importorskip("httpx")


@pytest.mark.usefixtures("_require_network")
def test_download_one_set() -> None:
    """真实下载 1 个 mania set，验 .osu + manifest 落盘。"""
    import httpx

    from scripts.download_sayobot import (
        DEFAULT_USER_AGENT,
        append_manifest,
        build_record,
        download_osz,
        extract_osz,
        search_beatmaplist,
    )

    headers = {
        "User-Agent": DEFAULT_USER_AGENT,
        "Referer": "https://osu.sayobot.cn/",
    }
    with httpx.Client(follow_redirects=True, timeout=60.0, headers=headers) as client:
        batch = search_beatmaplist(client, keyword="", offset=0, limit=25)
        assert batch, "listing returned empty"

        # 取第一个有星级的 ranked set
        item = next(
            (
                x
                for x in batch
                if float(x.get("order", 0.0) or 0.0) > 0.0 and int(x.get("approved", 0)) == 1
            ),
            None,
        )
        assert item is not None, "no ranked set with stars in first page"

        sid = int(item["sid"])
        tmp = Path(tempfile.mkdtemp())
        osz_path = tmp / f"{sid}.osz"
        download_osz(client, sid, osz_path, no_verify=True)
        assert osz_path.exists()
        assert osz_path.stat().st_size > 0

        n_diffs, n_4k = extract_osz(osz_path, tmp / str(sid))
        assert n_diffs >= 1
        assert n_4k >= 1, "selected set should contain a 4K mania diff"

        manifest = tmp / "manifest.jsonl"
        record = build_record(item, n_diffs=n_diffs, n_4k_diffs=n_4k, sid_dir=str(sid))
        append_manifest(manifest, record)

        # manifest 行可解析且字段齐全
        line = manifest.read_text(encoding="utf-8").strip()
        rec = json.loads(line)
        assert rec["sid"] == sid
        assert rec["stars"] > 0.0
        assert rec["license"] == "academic"
        assert rec["n_4k_diffs"] >= 1

        # .osu 真实落盘
        osu_files = list((tmp / str(sid)).rglob("*.osu"))
        assert len(osu_files) == n_diffs


@pytest.mark.usefixtures("_require_network")
def test_pipeline_filters_real_data() -> None:
    """下载几个 set → 喂管线 → 验证带 manifest 后 parquet 只含过 §4.3 的。"""

    from beatmorph.data.pipeline.embed import PreprocessPipeline
    from scripts.download_sayobot import run_download

    tmp = Path(tempfile.mkdtemp())
    raw_dir = tmp / "raw"
    manifest = raw_dir / "manifest.jsonl"

    collected = run_download(
        keyword="",
        list_mode=True,
        max_sets=8,
        raw_dir=raw_dir,
        manifest_path=manifest,
        only_4k=True,
        skip_unranked=True,
        rate_delay=1.0,
        retries=2,
        keep_osz=False,
        no_verify=True,
        cache_dir=tmp / "cache",
    )
    if collected == 0:
        pytest.skip("no sets collected (network/mirror unavailable)")

    out_dir = tmp / "processed"
    pipeline = PreprocessPipeline(
        raw_dir=raw_dir,
        out_dir=out_dir,
        manifest_path=manifest,
    )
    assert pipeline._injector is not None
    pipeline.run()
    pipeline.write_stats()

    # stats 合理：parsed>=1, failed 尽量少
    assert pipeline.stats["total"] >= 1
    assert pipeline.stats["parsed"] >= 1

    # 若产出了 parquet，每行 stars>=3.0 且 playcount>500
    parquet = out_dir / "charts.parquet"
    if parquet.exists():
        try:
            import pyarrow.parquet as pq
        except ImportError:
            pytest.skip("pyarrow not installed")
        table = pq.read_table(parquet)
        for row_json in table.column("chart_json").to_pylist():
            meta = json.loads(row_json)["meta"]
            assert float(meta.get("difficulty_rating", 0.0)) >= 3.0
            assert int(meta.get("playcount", 0)) > 500
