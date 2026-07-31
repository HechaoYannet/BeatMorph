"""sayobot.cn 谱面集下载脚本（RFC-0024）。

从 osu.sayobot.cn 镜像抓取 osu!mania 谱面集，解压 .osz（ZIP）得到 .osu + 音频，
落盘到 ``data/raw/{sid}/``，并把集级元数据（star/order、play_count、approved…）
追加写入 ``manifest.jsonl``。该 manifest 被
:class:`beatmorph.data.pipeline.embed.PreprocessPipeline` 读取，注入
``Chart.meta`` 后真正激活奠基 §4.3 的 ``≥3星 且 play_count>500`` 质量过滤。

逆向出的 API（HAR 权威，无鉴权）：

- **列表**：``POST https://api.sayobot.cn/?post``
    - ``Content-Type: text/plain``（非常规！body 是 JSON 字符串，非 application/json）
    - body: ``{"cmd":"beatmaplist","limit":25,"offset":0,"type":"search",
      "keyword":"","mode":8,"class":31,"subtype":63,"genre":1535,"language":4095}``
    - ``mode:8`` = osu!mania。返回 ``{"data":[{sid,title,artist,creator,
      play_count,order,approved,modes,...}, ...]}``
    - ``order`` = 星级 float（未 Ranked/Pending 时 0.0；approved:1 Ranked / 3 Pending）
    - ``play_count`` = 集级总 play；``sid`` = 谱面集 id

- **下载**：``GET https://txy1.sayobot.cn/beatmaps/download/full/{sid}?server=auto``
    → 302 跳 ``https://cmcc.sayobot.cn:25225/beatmaps/{前3位}/{余}/full?filename=...``
    → 200 ``application/octet-stream``，body 即 ``.osz`` (ZIP)。
    - cmcc:25225 证书链可能无法验证 → 遇 SSL 错降级 verify=False 并 warn

用法示例（需 ``uv run --extra data`` 或装了 httpx 的环境）::

    uv run python scripts/download_sayobot.py --keyword "felys" --max-sets 50
    uv run python scripts/download_sayobot.py --list-mode --max-sets 5000

输出：``<raw_dir>/{sid}/*.osu`` + ``<raw_dir>/manifest.jsonl``。
"""

from __future__ import annotations

import argparse
import json
import shutil
import ssl
import time
import zipfile
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

import httpx

from beatmorph.core.logging import get_logger

logger = get_logger(__name__)

# ── 常量 ──────────────────────────────────────────────────────
API_URL = "https://api.sayobot.cn/?post"
DOWNLOAD_URL = "https://txy1.sayobot.cn/beatmaps/download/full/{sid}?server=auto"
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_MANIFEST = "manifest.jsonl"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36 Edg/150.0.0.0"
)
# sayobot 列表 API 默认查询参数（mania 全流派全语言）
DEFAULT_CLS = 31
DEFAULT_SUBTYPE = 63
DEFAULT_GENRE = 1535
DEFAULT_LANGUAGE = 4095
MODE_MANIA = 8  # sayobot 列表 API mania 位掩码（bit3）
OSU_MODE_MANIA = 3  # .osu 文件 [General] Mode 字段：3 = osu!mania
MANIA_4K_CIRCLESIZE = 4
PAGE_LIMIT = 25  # 列表 API 单页
APPROVED_RANKED = 1  # sayobot approved: 1=Ranked
APPROVED_PENDING = 3  # sayobot approved: 3=Pending/WIP（无星级，order=0.0）


# ── 列表 API ──────────────────────────────────────────────────


def search_beatmaplist(
    client: httpx.Client,
    *,
    keyword: str = "",
    mode: int = MODE_MANIA,
    cls: int = DEFAULT_CLS,
    subtype: int = DEFAULT_SUBTYPE,
    genre: int = DEFAULT_GENRE,
    language: int = DEFAULT_LANGUAGE,
    limit: int = PAGE_LIMIT,
    offset: int = 0,
) -> list[dict]:
    """POST 列表 API，返回 data 列表。

    关键点：body 以 ``Content-Type: text/plain`` 发送 JSON 字符串（非常规，
    ``httpx json=`` 会发 ``application/json`` 被拒）。
    """
    body_str = json.dumps(
        {
            "cmd": "beatmaplist",
            "limit": limit,
            "offset": offset,
            "type": "search",
            "keyword": keyword,
            "mode": mode,
            "class": cls,
            "subtype": subtype,
            "genre": genre,
            "language": language,
        }
    )
    headers = {
        "Content-Type": "text/plain",
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://osu.sayobot.cn",
        "Referer": "https://osu.sayobot.cn/",
    }
    resp = client.post(API_URL, content=body_str, headers=headers)
    resp.raise_for_status()
    payload = resp.json()
    data = payload.get("data") or []
    if not isinstance(data, list):
        logger.warning("Unexpected beatmaplist response shape: %s", type(data).__name__)
        return []
    return data


# ── 下载 .osz ────────────────────────────────────────────────


def download_osz(
    client: httpx.Client,
    sid: int,
    dest: Path,
    *,
    no_verify: bool = False,
) -> Path:
    """下载一个 set 的 .osz 到 dest，跟随 302 跳转，返回落盘路径。

    遇 ``SSL`` 证书验证错误（cmcc:25225 证书链）时降级 ``verify=False`` 重试一次
    并 warn；``no_verify=True`` 直接全程不校验。
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = DOWNLOAD_URL.format(sid=sid)

    if no_verify:
        return _fetch_to(client, url, dest, verify=False)

    try:
        return _fetch_to(client, url, dest, verify=True)
    except httpx.TransportError as exc:
        # 证书验证失败属于 TransportError 子类（ssl.SSLCertVerificationError）
        if isinstance(exc.__cause__, ssl.SSLCertVerificationError) or "SSL" in type(exc).__name__:
            logger.warning(
                "Set %d: TLS verify failed on cmcc:25225, retrying with verify=False: %s",
                sid,
                exc,
            )
            # 新建一个不校验的 client（同 client 改 verify 不便，重建更干净）
            verify_client = httpx.Client(
                follow_redirects=True,
                timeout=client.timeout,
                verify=False,
                headers=dict(client.headers),
            )
            try:
                return _fetch_to(verify_client, url, dest, verify=False)
            finally:
                verify_client.close()
        raise


def _fetch_to(client: httpx.Client, url: str, dest: Path, *, verify: bool) -> Path:
    resp = client.get(url)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


# ── 解压 .osz ────────────────────────────────────────────────


def extract_osz(osz_path: Path, dest_dir: Path) -> tuple[int, int]:
    """解压 .osz（ZIP）到 dest_dir，返回 (n_diffs, n_4k_diffs)。

    stdlib zipfile.extractall 自 3.6.2 起已防 zip-slip 路径穿越。
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(osz_path) as zf:
        zf.extractall(dest_dir)

    osu_files = sorted(dest_dir.rglob("*.osu"))
    n_diffs = len(osu_files)
    n_4k = sum(1 for f in osu_files if _is_4k_mania(f))
    return n_diffs, n_4k


def _is_4k_mania(osu_path: Path) -> bool:
    """快速扫 .osu 的 [General] Mode:3 且 [Difficulty] CircleSize:4。"""
    try:
        text = osu_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    mode: int | None = None
    circle_size: int | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("Mode:"):
            with suppress(ValueError):
                mode = int(line.split(":", 1)[1].strip())
        elif line.startswith("CircleSize:"):
            with suppress(ValueError):
                circle_size = int(float(line.split(":", 1)[1].strip()))
        if mode is not None and circle_size is not None:
            break
    return mode == OSU_MODE_MANIA and circle_size == MANIA_4K_CIRCLESIZE


# ── manifest ─────────────────────────────────────────────────


def append_manifest(manifest_path: Path, record: dict) -> None:
    """追加一行 JSON 到 manifest（JSONL = 断点续传友好）。"""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_record(
    api_item: dict,
    *,
    n_diffs: int,
    n_4k_diffs: int,
    sid_dir: str,
) -> dict:
    """从列表 API 项构造 manifest 记录 dict。"""
    stars = float(api_item.get("order", 0.0) or 0.0)
    approved = int(api_item.get("approved", 0) or 0)
    return {
        "sid": int(api_item["sid"]),
        "title": api_item.get("title", ""),
        "title_unicode": api_item.get("titleU", "") or "",
        "artist": api_item.get("artist", ""),
        "artist_unicode": api_item.get("artistU", "") or "",
        "creator": api_item.get("creator", ""),
        "stars": stars,
        "play_count": int(api_item.get("play_count", 0) or 0),
        "approved": approved,
        "modes": int(api_item.get("modes", 0) or 0),
        "favourite_count": int(api_item.get("favourite_count", 0) or 0),
        "lastupdate": int(api_item.get("lastupdate", 0) or 0),
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_diffs": n_diffs,
        "n_4k_diffs": n_4k_diffs,
        "path": sid_dir,
        "unranked": stars == 0.0 or approved == APPROVED_PENDING,
        "license": "academic",
    }


# ── 续传判断 ─────────────────────────────────────────────────


def sid_already_downloaded(raw_dir: Path, sid: int) -> bool:
    """若 data/raw/{sid}/ 已存在且含至少 1 个 .osu → 视为已下载。"""
    sid_dir = raw_dir / str(sid)
    if not sid_dir.is_dir():
        return False
    return any(sid_dir.rglob("*.osu"))


# ── 主流程 ───────────────────────────────────────────────────


def run_download(
    *,
    keyword: str | None,
    list_mode: bool,
    max_sets: int,
    raw_dir: Path,
    manifest_path: Path,
    only_4k: bool,
    skip_unranked: bool,
    rate_delay: float,
    retries: int,
    keep_osz: bool,
    no_verify: bool,
    cache_dir: Path,
) -> int:
    """执行抓取主循环，返回成功收集的 set 数。"""
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": DEFAULT_USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://osu.sayobot.cn/",
    }
    transport = httpx.HTTPTransport(retries=retries) if retries > 0 else None
    client_kwargs: dict = {
        "follow_redirects": True,
        "timeout": httpx.Timeout(60.0, connect=15.0),
        "headers": headers,
    }
    if transport is not None:
        client_kwargs["transport"] = transport

    if no_verify:
        client_kwargs["verify"] = False

    collected = 0
    offset = 0
    with httpx.Client(**client_kwargs) as client:
        while collected < max_sets:
            try:
                batch = search_beatmaplist(
                    client,
                    keyword=keyword or "",
                    offset=offset,
                )
            except httpx.HTTPError as exc:
                logger.error("Listing failed at offset=%d: %s", offset, exc)
                break

            if not batch:
                logger.info("Listing exhausted at offset=%d", offset)
                break

            for item in batch:
                if collected >= max_sets:
                    break
                collected += _process_set(
                    client=client,
                    item=item,
                    raw_dir=raw_dir,
                    manifest_path=manifest_path,
                    cache_dir=cache_dir,
                    only_4k=only_4k,
                    skip_unranked=skip_unranked,
                    rate_delay=rate_delay,
                    keep_osz=keep_osz,
                    no_verify=no_verify,
                )

            offset += len(batch)

    logger.info(
        "Done: collected %d sets into %s (manifest=%s)",
        collected,
        raw_dir,
        manifest_path,
    )
    return collected


def _process_set(
    *,
    client: httpx.Client,
    item: dict,
    raw_dir: Path,
    manifest_path: Path,
    cache_dir: Path,
    only_4k: bool,
    skip_unranked: bool,
    rate_delay: float,
    keep_osz: bool,
    no_verify: bool,
) -> int:
    """处理单个 set：跳过/下载/解压/写 manifest。返回本次收集数（0 或 1）。"""
    sid = int(item["sid"])
    stars = float(item.get("order", 0.0) or 0.0)
    approved = int(item.get("approved", 0) or 0)

    if skip_unranked and (stars == 0.0 or approved == APPROVED_PENDING):
        logger.debug("Skip set %d (unranked: order=%s approved=%s)", sid, stars, approved)
        return 0

    if sid_already_downloaded(raw_dir, sid):
        logger.info("Skip set %d (already downloaded)", sid)
        return 0

    osz_path = cache_dir / f"{sid}.osz"
    sid_dir = raw_dir / str(sid)
    try:
        download_osz(client, sid, osz_path, no_verify=no_verify)
        n_diffs, n_4k = extract_osz(osz_path, sid_dir)

        if only_4k and n_4k == 0:
            logger.info(
                "Set %d: no 4K mania diff (n_diffs=%d) — purging",
                sid,
                n_diffs,
            )
            shutil.rmtree(sid_dir, ignore_errors=True)
            return 0

        record = build_record(
            item,
            n_diffs=n_diffs,
            n_4k_diffs=n_4k,
            sid_dir=str(sid),
        )
        append_manifest(manifest_path, record)
        logger.info(
            "Set %d downloaded: '%s - %s' stars=%.2f play=%d diffs=%d(4K=%d)",
            sid,
            record["artist"],
            record["title"],
            record["stars"],
            record["play_count"],
            n_diffs,
            n_4k,
        )
        return 1
    except (httpx.HTTPError, zipfile.BadZipFile, OSError) as exc:
        logger.warning("Set %d failed: %s (continuing)", sid, exc)
        # 清理半成品
        if sid_dir.exists():
            shutil.rmtree(sid_dir, ignore_errors=True)
        return 0
    finally:
        if not keep_osz and osz_path.exists():
            with suppress(OSError):
                osz_path.unlink()
        time.sleep(rate_delay)


# ── CLI ───────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="download_sayobot",
        description="从 osu.sayobot.cn 下载 osu!mania 谱面集并写 manifest（RFC-0024）。",
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--keyword", help="按关键词搜索（留空则按热度列表）")
    mode.add_argument("--list-mode", action="store_true", help="不搜索，按 offset 翻页列表")

    p.add_argument("--max-sets", type=int, default=50, help="最多收集 set 数（默认 50）")
    p.add_argument(
        "--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="落地根目录（默认 data/raw）"
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=f"manifest 路径（默认 <raw_dir>/{DEFAULT_MANIFEST}）",
    )
    p.add_argument(
        "--cache-dir", type=Path, default=Path("data/.osz_cache"), help=".osz 临时缓存目录"
    )
    p.add_argument(
        "--only-4k",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="仅保留含 4K mania diff 的 set",
    )
    p.add_argument(
        "--skip-unranked",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="跳过未 Ranked/Pending（order=0 或 approved=3），默认开（省带宽）",
    )
    p.add_argument(
        "--rate-delay", type=float, default=1.0, help="每个 set 之间 sleep 秒（礼貌限流，默认 1.0）"
    )
    p.add_argument("--retries", type=int, default=3, help="连接级重试次数（默认 3）")
    p.add_argument("--keep-osz", action="store_true", help="保留下载的 .osz 缓存（默认解压后删除）")
    p.add_argument(
        "--no-verify", action="store_true", help="全程关闭 TLS 校验（cmcc:25225 证书兜底）"
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    manifest_path = args.manifest or (args.raw_dir / DEFAULT_MANIFEST)

    collected = run_download(
        keyword=args.keyword,
        list_mode=args.list_mode,
        max_sets=args.max_sets,
        raw_dir=args.raw_dir,
        manifest_path=manifest_path,
        only_4k=args.only_4k,
        skip_unranked=args.skip_unranked,
        rate_delay=args.rate_delay,
        retries=args.retries,
        keep_osz=args.keep_osz,
        no_verify=args.no_verify,
        cache_dir=args.cache_dir,
    )
    logger.info("Complete: %d sets collected", collected)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
