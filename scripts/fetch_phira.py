"""Phira 谱面获取驱动脚本（plan 02 §4「三段式获取」的执行入口 / M1–M10）。

三段式：**枚举**（API 分页）→ **预筛**（HTTP Range 取中央目录 + 24 KB 前缀嗅探）→
**选择性下载**（只取 `info.yml` 指名的谱面条目 + `info.yml` 指名的音频条目）。
产出一份带 provenance 的谱面清单，再经 `build_pairs` 按曲目切分 train/val/test。

用法::

    uv run python scripts/fetch_phira.py meta                    # 全量枚举 → meta.jsonl
    uv run python scripts/fetch_phira.py fetch --limit 200       # 预筛 + 下载 → charts.jsonl
    uv run python scripts/fetch_phira.py pairs                   # 切分 → pairs.json
    uv run python scripts/fetch_phira.py stats                   # 语料统计 → stats.json
    uv run python scripts/fetch_phira.py all --limit 200

落盘布局（全部在 `.gitignore` 的 `data/`** 之下，**不入库**）::

    <root>/meta.jsonl          枚举全量元数据（带 provenance）
    <root>/charts.jsonl        已入库谱面行（带 provenance；切分与统计的输入）
    <root>/quarantine.jsonl    拒收记录（带 provenance，**拒收必记账**）
    <root>/pairs.json          train/val/test + 同曲跨谱泛化集
    <root>/stats.json          语料统计（M7/M9/M10 的「先统计再定阈值」落点）
    <root>/fetch_report.json   本次抓取报告（计数 / 字节 / 耗时 / 失败清单）
    <root>/charts/<chart_id>/<normalized>.<ext>     谱面文件（R4 规范化名）
    <root>/audio/<sha1>.<ext>                       音频（内容 sha1 去重）
    <root>/features/<sha1>.npz + <sha1>.meta.json   特征缓存（extract_features.py 产出）

合规（M8 硬约束③）：每份清单都带 provenance（来源 / 查询 / 时间 / 用途 / 脚本与版本），
可逐张追溯到 chart id。

退出码：0 成功｜2 参数错误｜3 清单/数据错误｜4 抓取有失败项（已记账，可重跑续传）。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from beatmorph.core.contracts import ChartFormat, ChartSource, PhigrosChart
from beatmorph.core.logging import get_logger, setup_logging
from beatmorph.data.parsers.rpejson import RpeParseError, parse_rpejson
from beatmorph.data.parsers.sniff import sniff_format_with_evidence
from beatmorph.data.phira.client import (
    CHART_PAGE_SIZE_MAX,
    CHART_TOTAL_EXPECTED,
    CHART_TYPE_ANY,
    RANGE_PREFIX_BYTES,
    REQUEST_INTERVAL_S,
    Manifest,
    ManifestError,
    ManifestPurpose,
    PhiraApiError,
    PhiraClient,
    PhiraZipError,
    Provenance,
    ZipIndex,
    provenance_for_api,
    read_manifest,
    sha1_hex,
    utc_now_iso,
    write_manifest,
)
from beatmorph.data.phira.package import (
    ChartPackage,
    ChartPackageError,
    chart_dest_path,
)
from beatmorph.data.pipeline.embed import build_pairs, chart_row, dataset_stats
from beatmorph.data.qc import QcReport, QuarantineRecord, QuarantineStage, quality_check

logger = get_logger("scripts.fetch_phira")

#: 脚本标识与版本（写进 provenance；M8 硬约束③要求可追溯到脚本本身）。
SCRIPT_ID = "scripts/fetch_phira.py"
SCRIPT_VERSION = "1.0.0"

DEFAULT_ROOT = Path("data/processed")

EXIT_OK = 0
EXIT_ARGS = 2
EXIT_DATA = 3
EXIT_FETCH = 4


# ══════════════════════════════════════════════════════════════
# 路径布局与清单写入
# ══════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Paths:
    """`data/processed` 下的产物布局（与 configs/phigros_masked.yaml 对齐）。"""

    root: Path

    @property
    def meta(self) -> Path:
        return self.root / "meta.jsonl"

    @property
    def charts(self) -> Path:
        return self.root / "charts.jsonl"

    @property
    def quarantine(self) -> Path:
        return self.root / "quarantine.jsonl"

    @property
    def pairs(self) -> Path:
        return self.root / "pairs.json"

    @property
    def stats(self) -> Path:
        return self.root / "stats.json"

    @property
    def report(self) -> Path:
        return self.root / "fetch_report.json"

    @property
    def charts_dir(self) -> Path:
        return self.root / "charts"

    @property
    def audio_dir(self) -> Path:
        return self.root / "audio"

    @property
    def features_dir(self) -> Path:
        return self.root / "features"


def _append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """把数据行追加到已带 provenance 的 JSONL 清单（逐行 flush，崩溃后可续跑）。"""
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        handle.flush()


def _prepare_manifest(path: Path, provenance: Provenance, *, overwrite: bool) -> None:
    """首次（或 `overwrite`）落 provenance 记录；已存在则原样续写。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not path.is_file():
        write_manifest(Manifest(provenance=provenance, rows=[]), path)
        logger.info("新建清单 %s", path)


def _done_chart_ids(path: Path) -> set[int]:
    """已入库的 chart id（续跑去重；清单不存在则空集）。"""
    if not path.is_file():
        return set()
    done: set[int] = set()
    for row in read_manifest(path).rows:
        chart_id = row.get("chart_id")
        if isinstance(chart_id, int) and not isinstance(chart_id, bool):
            done.add(chart_id)
    return done


# ══════════════════════════════════════════════════════════════
# 预筛：Range 条目读取
# ══════════════════════════════════════════════════════════════


class RangeEntryReader:
    """用 HTTP Range 读**单个 zip 条目全文**（`info.yml` 这类小条目）。

    只做两次 Range（本地文件头 + 数据段），不下载整包——这正是「全量直抓 76 GB」
    被压到每张 ~10 KB 预筛成本的原因。
    """

    def __init__(self, client: PhiraClient, url: str, index: ZipIndex) -> None:
        self.client = client
        self.url = url
        self.index = index

    def __call__(self, name: str) -> bytes:
        entry = self.index.by_name(name)
        return self.client.fetch_prefix(self.url, entry, max(entry.compress_size, 1))


# ══════════════════════════════════════════════════════════════
# 抓取报告
# ══════════════════════════════════════════════════════════════


@dataclass
class FetchReport:
    """一次抓取运行的计数（**拒收与失败都必须显式记账**）。"""

    selected: int = 0
    skipped_done: int = 0
    ok: int = 0
    rejected: Counter = field(default_factory=Counter)
    failed: int = 0
    audio_ok: int = 0
    audio_absent: int = 0
    audio_failed: int = 0
    formats: Counter = field(default_factory=Counter)
    chart_bytes: int = 0
    audio_bytes: int = 0
    elapsed_s: float = 0.0
    failures: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """报告行（JSON 友好）。"""
        return {
            "selected": self.selected,
            "skipped_done": self.skipped_done,
            "ok": self.ok,
            "rejected": dict(self.rejected),
            "failed": self.failed,
            "audio_ok": self.audio_ok,
            "audio_absent": self.audio_absent,
            "audio_failed": self.audio_failed,
            "formats": dict(self.formats),
            "chart_bytes": self.chart_bytes,
            "audio_bytes": self.audio_bytes,
            "elapsed_s": round(self.elapsed_s, 1),
            "failures": self.failures,
        }


@dataclass
class AudioOutcome:
    """音频条目抓取结果。"""

    status: str  #: ok / absent / failed
    path: Path | None = None
    key: str | None = None
    n_bytes: int = 0
    duration_s: float | None = None
    reason: str = ""


def _safe_suffix(name: str) -> str:
    """音频落盘扩展名（只保留字母数字，缺失回落 `.bin`）。"""
    suffix = Path(name).suffix.lower().lstrip(".")
    return f".{suffix}" if suffix.isalnum() else ".bin"


def _audio_duration_s(path: Path) -> float | None:
    """音频时长（秒）；解码器不可用时返回 None（**不猜**，由 QC 记为未统计）。"""
    try:
        import soundfile

        info = soundfile.info(str(path))
        if info.frames > 0 and info.samplerate > 0:
            return float(info.frames) / float(info.samplerate)
    except Exception as exc:
        logger.debug("soundfile 读时长失败 %s：%s", path.name, exc)
    try:
        import torchaudio

        info = torchaudio.info(str(path))
        if info.num_frames > 0 and info.sample_rate > 0:
            return float(info.num_frames) / float(info.sample_rate)
    except Exception as exc:
        logger.debug("torchaudio 读时长失败 %s：%s", path.name, exc)
    return None


def _fetch_audio(
    client: PhiraClient,
    url: str,
    package: ChartPackage,
    audio_dir: Path,
) -> AudioOutcome:
    """按 `info.yml.music` 下载音频条目，按**内容 sha1** 落盘去重。"""
    entry = package.entries.get(package.music_file)
    if entry is None:
        return AudioOutcome(
            status="absent", reason=f"info.yml.music={package.music_file!r} 不在包内"
        )
    audio_dir.mkdir(parents=True, exist_ok=True)
    # 临时名带上条目名的短哈希：同目录内的并发/残留 .part 不会互相续传错数据
    temp = (
        audio_dir
        / f"download-{sha1_hex(entry.name.encode('utf-8'))[:8]}{_safe_suffix(package.music_file)}.part"
    )
    try:
        client.download_entry(url, entry, temp)
        payload = temp.read_bytes()
        key = sha1_hex(payload)
        final = audio_dir / f"{key}{_safe_suffix(package.music_file)}"
        if final.is_file():
            temp.unlink()  # 同曲多谱共享同一音频：已存在即复用
        else:
            temp.replace(final)
        return AudioOutcome(
            status="ok",
            path=final,
            key=key,
            n_bytes=len(payload),
            duration_s=_audio_duration_s(final),
        )
    except (PhiraApiError, PhiraZipError, OSError) as exc:
        # 残留的 .part 会被下一次 download_entry 当成断点续传的起点；失败即清掉，
        # 避免「一次坏下载污染后续所有重试」（大小校验能发现，但会一直失败下去）。
        temp.unlink(missing_ok=True)
        logger.warning("音频下载失败 %s:%s：%s", url, package.music_file, exc)
        return AudioOutcome(status="failed", reason=str(exc))


# ══════════════════════════════════════════════════════════════
# 单张谱面的抓取与质检
# ══════════════════════════════════════════════════════════════


@dataclass
class IngestResult:
    """单张谱面的抓取结果（三态：入库 / 拒收 / 网络失败）。"""

    status: str  #: ok / rejected / failed
    row: dict[str, Any] | None = None
    quarantine: QuarantineRecord | None = None
    audio: AudioOutcome | None = None
    fmt: ChartFormat = ChartFormat.UNKNOWN
    chart_bytes: int = 0
    reason: str = ""


def _reject(
    chart_id: int,
    stage: QuarantineStage,
    fmt: ChartFormat,
    reasons: Sequence[str],
    evidence: str = "",
) -> IngestResult:
    """构造一条拒收结果（**拒收必须显式记账**，plan 02 §偏离 2）。"""
    record = QuarantineRecord(
        chart_id=chart_id,
        stage=stage,
        fmt=fmt,
        reasons=[reason for reason in reasons if reason],
        evidence=evidence,
    )
    return IngestResult(status="rejected", quarantine=record, fmt=fmt)


def _sniff_chart(
    client: PhiraClient,
    url: str,
    package: ChartPackage,
    reader: RangeEntryReader,
) -> tuple[ChartFormat, str, bytes | None]:
    """24 KB 前缀嗅探；判定不是 RPE 时**用全文复核**再记账（截断容错）。

    Returns:
        `(格式, 证据, 已下载的全文或 None)`。
    """
    entry = package.entries[package.chart_file]
    prefix = client.fetch_prefix(url, entry, RANGE_PREFIX_BYTES)
    fmt, evidence = sniff_format_with_evidence(prefix)
    if fmt is ChartFormat.RPE:
        return fmt, evidence, None
    full = reader(package.chart_file)
    fmt_full, evidence_full = sniff_format_with_evidence(full)
    if fmt_full is not ChartFormat.RPE:
        return fmt_full, evidence_full, full
    logger.info("前缀嗅探为 %s、全文复核为 RPE（截断容错）：chart_file=%s", fmt, package.chart_file)
    return fmt_full, evidence_full, full


def _write_chart_file(
    client: PhiraClient,
    url: str,
    package: ChartPackage,
    paths: Paths,
    chart_id: int,
    full: bytes | None,
) -> tuple[Path, bytes]:
    """落盘谱面文件（R4 规范化名），返回 `(目标路径, 字节)`。"""
    dest = chart_dest_path(paths.charts_dir, chart_id, package.chart_file)
    if full is None:
        client.download_entry(url, package.entries[package.chart_file], dest)
        payload = dest.read_bytes()
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(full)
        payload = full
    return dest, payload


def _manifest_row(
    meta_row: Mapping[str, Any],
    package: ChartPackage,
    chart: PhigrosChart,
    qc: QcReport,
    dest: Path,
    paths: Paths,
    audio: AudioOutcome,
) -> dict[str, Any]:
    """构造清单行（`chart_row` + 音频/来源留痕列）。"""
    info = package.info
    difficulty = info.difficulty_round
    if difficulty is None:
        raw_difficulty = meta_row.get("difficulty_round", meta_row.get("difficulty"))
        difficulty = None if raw_difficulty is None else float(raw_difficulty)
    name = info.name or str(meta_row.get("name") or "")
    composer = info.composer or str(meta_row.get("composer") or "")
    row = chart_row(
        chart_id=int(meta_row["id"]),
        name=name,
        composer=composer,
        difficulty=difficulty,
        fmt=qc.fmt,
        chart_path=str(dest.relative_to(paths.charts_dir)).replace("\\", "/"),
        feature_key=audio.key,
        qc=qc,
    )
    row.update(
        {
            "level": info.level,
            "charter": info.charter or str(meta_row.get("charter") or ""),
            "chart_file": package.chart_file,
            "music_file": package.music_file,
            "chart_source_url": str(meta_row.get("file") or ""),
            "chart_bytes": dest.stat().st_size,
            "bpm_points": len(chart.bpm_points),
            "audio_path": (
                None
                if audio.path is None
                else str(audio.path.relative_to(paths.root)).replace("\\", "/")
            ),
            "audio_sha1": audio.key,
            "audio_status": audio.status,
            "audio_duration_s": audio.duration_s,
            "fetched_at": utc_now_iso(),
        },
    )
    return row


def ingest_chart(
    client: PhiraClient,
    meta_row: Mapping[str, Any],
    paths: Paths,
    *,
    with_audio: bool,
) -> IngestResult:
    """预筛 → 选择性下载 → 解析 → 质检 → 清单行（单张谱面的完整流水线）。"""
    chart_id = int(meta_row["id"])
    url = str(meta_row.get("file") or "")
    if not url:
        return _reject(
            chart_id, QuarantineStage.PACKAGE, ChartFormat.UNKNOWN, ["元数据缺 file 直链"]
        )

    index = client.fetch_zip_index(url)
    reader = RangeEntryReader(client, url, index)
    try:
        package = ChartPackage.from_index(index, reader)
    except ChartPackageError as exc:
        return _reject(chart_id, QuarantineStage.PACKAGE, ChartFormat.UNKNOWN, [str(exc)])

    fmt, evidence, full = _sniff_chart(client, url, package, reader)
    if fmt is not ChartFormat.RPE:
        return _reject(
            chart_id,
            QuarantineStage.SNIFF,
            fmt,
            [f"内容嗅探判定 {fmt}（v1 主路径只收 RPE，plan 02 §偏离 2）"],
            evidence=evidence,
        )

    dest, payload = _write_chart_file(client, url, package, paths, chart_id, full)
    source = ChartSource(
        chart_id=chart_id,
        format=fmt,
        sniff_evidence=evidence,
        chart_file=package.chart_file,
        music_file=package.music_file,
        chart_sha1=sha1_hex(payload),
    )
    try:
        chart = parse_rpejson(payload, source)
    except RpeParseError as exc:
        return _reject(
            chart_id, QuarantineStage.PARSE, fmt, [f"RPE 解析失败：{exc}"], evidence=evidence
        )

    audio = (
        _fetch_audio(client, url, package, paths.audio_dir)
        if with_audio
        else AudioOutcome(status="absent", reason="--no-audio")
    )
    qc = quality_check(chart, audio.duration_s, chart_id=chart_id, fmt=fmt)
    if not qc.passed:
        return IngestResult(
            status="rejected",
            quarantine=QuarantineRecord(
                chart_id=chart_id,
                stage=QuarantineStage.QC,
                fmt=fmt,
                reasons=list(qc.errors),
                evidence=evidence,
            ),
            audio=audio,
            fmt=fmt,
            chart_bytes=len(payload),
        )

    return IngestResult(
        status="ok",
        row=_manifest_row(meta_row, package, chart, qc, dest, paths, audio),
        audio=audio,
        fmt=fmt,
        chart_bytes=len(payload),
    )


# ══════════════════════════════════════════════════════════════
# 子命令
# ══════════════════════════════════════════════════════════════


def cmd_meta(args: argparse.Namespace) -> int:
    """M1：分页枚举全部谱面元数据 → `meta.jsonl`（带 provenance）。"""
    paths = Paths(args.root)
    purpose = ManifestPurpose(args.purpose)
    rows: list[dict[str, Any]] = []
    with PhiraClient() as client:
        for meta in client.iter_chart_meta(page_size=args.page_size, sleep_s=args.sleep_s):
            rows.append(meta.to_row())
    if not rows:
        logger.error("枚举到 0 条记录（退出码 %d）", EXIT_DATA)
        return EXIT_DATA
    ids = [int(row["id"]) for row in rows]
    provenance = provenance_for_api(
        query=(
            f"type={CHART_TYPE_ANY}（any）全量分页 pageNum={args.page_size}；"
            f"实测总数 {len(rows)}（脚本内参照 CHART_TOTAL_EXPECTED={CHART_TOTAL_EXPECTED}）"
        ),
        purpose=purpose,
        script=SCRIPT_ID,
        script_version=SCRIPT_VERSION,
        chart_id_min=min(ids),
        chart_id_max=max(ids),
    )
    write_manifest(Manifest(provenance=provenance, rows=rows), paths.meta)
    delta = len(rows) - CHART_TOTAL_EXPECTED
    logger.info(
        "枚举完成：%d 张（调研参照值 %d，偏差 %+d）→ %s",
        len(rows),
        CHART_TOTAL_EXPECTED,
        delta,
        paths.meta,
    )
    if delta:
        logger.warning(
            "库容量与调研实测值相差 %+d 张：Phira 是活的社区库（新谱持续上传），"
            "偏差须写进交接件而不是当作噪声",
            delta,
        )
    return EXIT_OK


def _select_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    limit: int | None,
    offset: int,
    sample_seed: int | None,
) -> list[Mapping[str, Any]]:
    """选取本轮抓取的谱面（默认按 API 顺序；给了种子则做**可复现的随机抽样**）。"""
    if sample_seed is not None and limit is not None:
        picked = random.Random(sample_seed).sample(list(rows), k=min(limit, len(rows)))
        return sorted(picked, key=lambda row: int(row["id"]))
    return list(rows[offset:] if limit is None else rows[offset : offset + limit])


def _accumulate(report: FetchReport, result: IngestResult) -> None:
    """把单张结果计入报告。"""
    report.formats[str(result.fmt)] += 1
    if result.status == "ok":
        report.ok += 1
        report.chart_bytes += result.chart_bytes
        _accumulate_audio(report, result.audio)
        return
    if result.status == "rejected":
        stage = "unknown" if result.quarantine is None else str(result.quarantine.stage)
        report.rejected[stage] += 1
        return
    report.failed += 1


def _accumulate_audio(report: FetchReport, audio: AudioOutcome | None) -> None:
    """音频三态计数。"""
    if audio is None:
        return
    if audio.status == "ok":
        report.audio_ok += 1
        report.audio_bytes += audio.n_bytes
    elif audio.status == "failed":
        report.audio_failed += 1
    else:
        report.audio_absent += 1


def _ingest_guarded(
    client: PhiraClient,
    meta_row: Mapping[str, Any],
    paths: Paths,
    *,
    with_audio: bool,
) -> IngestResult:
    """单张抓取 + 异常分级（网络失败 vs 结构拒收）。"""
    chart_id = int(meta_row["id"])
    try:
        return ingest_chart(client, meta_row, paths, with_audio=with_audio)
    except (PhiraApiError, PhiraZipError) as exc:
        return IngestResult(status="failed", reason=str(exc))
    except (ChartPackageError, RpeParseError) as exc:
        return _reject(chart_id, QuarantineStage.PACKAGE, ChartFormat.UNKNOWN, [str(exc)])
    except OSError as exc:
        return IngestResult(status="failed", reason=f"OSError: {exc}")


#: 并发上限（plan 02 §4 的纪律：实测 8 线程为上限，这里不放开）。
MAX_WORKERS = 8


def _iter_results(
    client: PhiraClient,
    todo: Sequence[Mapping[str, Any]],
    paths: Paths,
    args: argparse.Namespace,
    stop: threading.Event,
) -> Iterator[tuple[Mapping[str, Any], IngestResult | None]]:
    """按 `--workers` 串行或并发出结果（**结果只在主线程落盘**）。

    并发是这一节的性能要点：单张谱面要 ~10 次 Range 往返（取长 + 尾部 + 中央目录 +
    `info.yml` + 谱面前缀 + 谱面条目 + 音频分块），串行时 RTT 与退避重试完全串联
    ——实测串行 ~9.5 s/张（20 张样本，189 s / 20 张）；并发后网络等待互相重叠。

    `stop` 置位后，尚未开始的任务直接返回 None（不再发请求），用于「连续失败即中止」。
    """

    def task(row: Mapping[str, Any]) -> tuple[Mapping[str, Any], IngestResult | None]:
        if stop.is_set():
            return row, None
        if args.sleep_s > 0:
            time.sleep(args.sleep_s)
        return row, _ingest_guarded(client, row, paths, with_audio=not args.no_audio)

    if args.workers <= 1:
        for row in todo:
            yield task(row)
        return

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(task, row): row for row in todo}
        for future in as_completed(futures):
            row = futures[future]
            try:
                yield future.result()
            except Exception as exc:
                yield row, IngestResult(status="failed", reason=f"{type(exc).__name__}: {exc}")


def cmd_fetch(args: argparse.Namespace) -> int:
    """三段式的第二、三段：预筛 + 选择性下载（可续跑）。"""
    paths = Paths(args.root)
    if not paths.meta.is_file():
        logger.error("枚举清单不存在：%s（先跑 meta；退出码 %d）", paths.meta, EXIT_DATA)
        return EXIT_DATA
    meta_manifest = read_manifest(paths.meta)
    selection_note = f"从 {paths.meta.name} 选取 {args.limit if args.limit else 'ALL'} 条"
    if args.offset:
        selection_note += f"（offset={args.offset}）"
    if args.sample_seed is not None:
        selection_note += f"（sample_seed={args.sample_seed}）"
    provenance = meta_manifest.provenance.model_copy(
        update={
            "purpose": ManifestPurpose(args.purpose),
            "script": SCRIPT_ID,
            "script_version": SCRIPT_VERSION,
            "fetched_at": utc_now_iso(),
            "query": selection_note,
        },
    )
    _prepare_manifest(paths.charts, provenance, overwrite=args.overwrite)
    _prepare_manifest(paths.quarantine, provenance, overwrite=args.overwrite)
    # 已入库 + 已拒收的都算「处理过」：拒收是确定性的数据问题，重跑一遍只会白烧带宽
    # （--retry-quarantined 显式要求时才重试，例如换了质检口径之后）。
    done = set() if args.overwrite else _done_chart_ids(paths.charts)
    if not args.overwrite and not args.retry_quarantined:
        done |= _done_chart_ids(paths.quarantine)

    selected = _select_rows(
        meta_manifest.rows, limit=args.limit, offset=args.offset, sample_seed=args.sample_seed
    )
    todo = [row for row in selected if int(row["id"]) not in done]
    report = FetchReport(selected=len(selected), skipped_done=len(selected) - len(todo))
    started = time.monotonic()
    logger.info(
        "开始抓取：选中 %d 张（已处理 %d 张将跳过），并发 %d → %s",
        len(selected),
        report.skipped_done,
        args.workers,
        paths.root,
    )
    stop = threading.Event()
    consecutive_failures = 0
    with PhiraClient(max_retries=args.max_retries, retry_backoff_s=args.retry_backoff_s) as client:
        results = _iter_results(client, todo, paths, args, stop)
        for position, (meta_row, result) in enumerate(results, start=1):
            if result is None:
                continue
            chart_id = int(meta_row["id"])
            _accumulate(report, result)
            if result.row is not None:
                _append_jsonl(paths.charts, [result.row])
            if result.quarantine is not None:
                _append_jsonl(paths.quarantine, [result.quarantine.to_dict()])
            if result.status == "failed":
                consecutive_failures += 1
                report.failures.append({"chart_id": chart_id, "reason": result.reason})
                logger.warning("抓取失败 chart_id=%d：%s", chart_id, result.reason)
                if consecutive_failures >= args.max_failures:
                    stop.set()
                    logger.error(
                        "连续 %d 张失败，判定网络/服务异常，提前中止（退出码 %d）",
                        consecutive_failures,
                        EXIT_FETCH,
                    )
                    break
            else:
                consecutive_failures = 0
            if position % 25 == 0 or position == len(todo):
                logger.info(
                    "进度 %d/%d：入库 %d、拒收 %d、失败 %d",
                    position,
                    len(todo),
                    report.ok,
                    sum(report.rejected.values()),
                    report.failed,
                )

    report.elapsed_s = time.monotonic() - started
    paths.report.write_text(
        json.dumps(
            {"provenance": provenance.model_dump(mode="json"), **report.to_dict()},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(
        "抓取完成：入库 %d、拒收 %d %s、失败 %d；谱面 %.1f MB、音频 %.1f MB；%.0fs → %s",
        report.ok,
        sum(report.rejected.values()),
        dict(report.rejected),
        report.failed,
        report.chart_bytes / 1e6,
        report.audio_bytes / 1e6,
        report.elapsed_s,
        paths.report,
    )
    return EXIT_FETCH if report.failed else EXIT_OK


def cmd_pairs(args: argparse.Namespace) -> int:
    """M10：按曲目切分 train/val/test + 同曲跨谱泛化集。"""
    paths = Paths(args.root)
    if not paths.charts.is_file():
        logger.error("谱面清单不存在：%s（先跑 fetch；退出码 %d）", paths.charts, EXIT_DATA)
        return EXIT_DATA
    ratios = (float(args.ratios[0]), float(args.ratios[1]), float(args.ratios[2]))
    try:
        splits = build_pairs(
            paths.charts,
            paths.charts_dir,
            paths.features_dir,
            ratios=ratios,
            seed=args.seed,
            require_feature=not args.allow_missing_features,
        )
    except (ManifestError, ValueError) as exc:
        logger.error("配对失败（退出码 %d）：%s", EXIT_DATA, exc)
        return EXIT_DATA
    payload = splits.to_dict()
    payload["provenance"] = read_manifest(paths.charts).provenance.model_dump(mode="json")
    paths.pairs.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "配对完成：train=%d val=%d test=%d 泛化对=%d；跳过（谱面缺 %d / 特征缺 %d）→ %s",
        len(splits.train),
        len(splits.val),
        len(splits.test),
        len(splits.generalization),
        splits.skipped_no_chart,
        splits.skipped_no_feature,
        paths.pairs,
    )
    return EXIT_OK


def cmd_stats(args: argparse.Namespace) -> int:
    """语料统计（M7/M9/M10 的「先统计再定阈值」落点）。"""
    paths = Paths(args.root)
    if not paths.charts.is_file():
        logger.error("谱面清单不存在：%s（退出码 %d）", paths.charts, EXIT_DATA)
        return EXIT_DATA
    stats = dataset_stats(paths.charts)
    payload = stats.to_dict()
    payload["provenance"] = read_manifest(paths.charts).provenance.model_dump(mode="json")
    paths.stats.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "语料统计：%d 张 / %d 个唯一曲目（重复率 %.3f）/ %d 个唯一音频（重复率 %.3f）；"
        "判定线 中位 %.0f（P25 %.0f / P75 %.0f，范围 %d–%d）；格式 %s → %s",
        stats.n_charts,
        stats.n_unique_songs,
        stats.same_song_repeat_rate,
        stats.n_unique_audio,
        stats.audio_repeat_rate,
        stats.lines_median,
        stats.lines_p25,
        stats.lines_p75,
        stats.lines_min,
        stats.lines_max,
        stats.format_counts,
        paths.stats,
    )
    for note in stats.corpus_outliers():
        logger.warning("语料离群：%s", note)
    return EXIT_OK


def cmd_all(args: argparse.Namespace) -> int:
    """meta（缺失或 `--refresh` 时）→ fetch → pairs → stats。"""
    paths = Paths(args.root)
    if args.refresh or not paths.meta.is_file():
        code = cmd_meta(args)
        if code != EXIT_OK:
            return code
    fetch_code = cmd_fetch(args)
    if fetch_code == EXIT_FETCH and not args.continue_on_failure:
        logger.error("有抓取失败项，按约定不继续配对（退出码 %d）", EXIT_FETCH)
        return EXIT_FETCH
    pairs_code = cmd_pairs(args)
    stats_code = cmd_stats(args)
    for code in (pairs_code, stats_code):
        if code != EXIT_OK:
            return code
    return fetch_code


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════


def _add_common(parser: argparse.ArgumentParser) -> None:
    """公共参数（产物根 + 用途留痕）。"""
    parser.add_argument(
        "--root", type=Path, default=DEFAULT_ROOT, help="产物根（默认 data/processed）"
    )
    parser.add_argument(
        "--purpose",
        default=str(ManifestPurpose.TRAIN),
        choices=[str(member) for member in ManifestPurpose],
        help="清单用途留痕（M8 硬约束③）",
    )


def _add_fetch_options(parser: argparse.ArgumentParser) -> None:
    """`fetch` / `all` 共用的抓取参数。"""
    parser.add_argument("--limit", type=int, default=None, help="最多抓取 N 张（默认全部）")
    parser.add_argument("--offset", type=int, default=0, help="从第 N 条开始（默认 0）")
    parser.add_argument(
        "--sample-seed", type=int, default=None, help="可复现随机抽样种子（与 --limit 同用）"
    )
    parser.add_argument("--no-audio", action="store_true", help="不下载音频（只收谱面结构）")
    parser.add_argument("--overwrite", action="store_true", help="忽略已入库记录，重抓整批")
    parser.add_argument(
        "--retry-quarantined",
        action="store_true",
        help="连已拒收的谱面也重试（默认跳过：拒收是确定性的数据问题）",
    )
    parser.add_argument(
        "--sleep-s",
        type=float,
        default=0.0,
        help="每个任务开始前的自限速（秒）；并发时是「每任务一次」而不是全局间隔",
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-s", type=float, default=1.0, help="指数退避基数（秒）")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help=f"并发数（默认 4，**上限 {MAX_WORKERS}**：plan 02 §4 的限速纪律）",
    )
    parser.add_argument("--max-failures", type=int, default=20, help="连续失败上限（提前中止）")


def _add_pairs_options(parser: argparse.ArgumentParser) -> None:
    """`pairs` / `all` 共用的切分参数。"""
    parser.add_argument("--ratios", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--allow-missing-features",
        action="store_true",
        help="允许缺特征缓存的行进对（默认要求特征齐备）",
    )


def build_parser() -> argparse.ArgumentParser:
    """命令行解析器（`meta` / `fetch` / `pairs` / `stats` / `all`）。"""
    parser = argparse.ArgumentParser(prog="fetch_phira.py", description="Phira 谱面获取驱动脚本")
    sub = parser.add_subparsers(dest="command", required=True)

    meta = sub.add_parser("meta", help="M1：全量分页枚举元数据 → meta.jsonl")
    _add_common(meta)
    meta.add_argument("--page-size", type=int, default=CHART_PAGE_SIZE_MAX)
    meta.add_argument("--sleep-s", type=float, default=REQUEST_INTERVAL_S, help="页间自限速（秒）")
    meta.set_defaults(func=cmd_meta)

    fetch = sub.add_parser("fetch", help="预筛 + 选择性下载 → charts.jsonl")
    _add_common(fetch)
    _add_fetch_options(fetch)
    fetch.set_defaults(func=cmd_fetch)

    pairs = sub.add_parser("pairs", help="M10：按曲目切分 train/val/test → pairs.json")
    _add_common(pairs)
    _add_pairs_options(pairs)
    pairs.set_defaults(func=cmd_pairs)

    stats = sub.add_parser("stats", help="语料统计 → stats.json")
    _add_common(stats)
    stats.set_defaults(func=cmd_stats)

    everything = sub.add_parser("all", help="meta → fetch → pairs → stats")
    _add_common(everything)
    _add_fetch_options(everything)
    _add_pairs_options(everything)
    everything.add_argument("--refresh", action="store_true", help="强制重新枚举")
    everything.add_argument("--page-size", type=int, default=CHART_PAGE_SIZE_MAX)
    everything.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="即使有抓取失败也继续配对（默认失败即停）",
    )
    everything.set_defaults(func=cmd_all)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """脚本入口：返回退出码。"""
    setup_logging()
    args = build_parser().parse_args(argv)
    limit = getattr(args, "limit", None)
    if limit is not None and limit < 0:
        logger.error("--limit 不能为负（退出码 %d）", EXIT_ARGS)
        return EXIT_ARGS
    workers = getattr(args, "workers", None)
    if workers is not None and not 1 <= workers <= MAX_WORKERS:
        logger.error("--workers 必须落在 1..%d（退出码 %d）", MAX_WORKERS, EXIT_ARGS)
        return EXIT_ARGS
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
