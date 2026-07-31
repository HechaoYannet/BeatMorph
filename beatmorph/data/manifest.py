"""下载清单（manifest）加载与元数据注入（RFC-0024）。

奠基文档 §4.3 的质量过滤（`≥3 星 且 play_count > 500`）需要每个谱面的
`difficulty_rating` 与 `playcount`，但 ``.osu`` 文件本身不含这两个字段——它们
来自 osu! 服务端的集级统计数据。``scripts/download_sayobot.py`` 把抓取到的集级
star/order 与 play_count 落成 JSONL manifest，本模块负责：

1. :func:`load_manifest` —— 读取 JSONL → ``{sid: SetRecord}``；
2. :class:`ManifestInjector` —— 在 :class:`PreprocessPipeline` 解析每张谱面后，
   按 ``chart.meta["beatmap_set_id"]`` 查 manifest，把集级 ``difficulty_rating``
   /``playcount``/``license``/``approved`` **注入** ``Chart.meta``（只加不覆盖），
   从而真正激活 §4.3 过滤（此前因字段缺失而走 Phase 1 宽松通过）。

设计要点：
- 注入靠 ``Chart.meta: dict[str, str|int|float]``（契约既有类型），**不改契约**，
  因此本模块不涉及 contract RFC。
- 注入器「只加不覆盖」——若调用方已设 ``difficulty_rating``（如未来 per-diff
  精确星级来源），保留原值。
- ``sid`` 缺失或不在 manifest → no-op，退回宽松通过（向后兼容不传 manifest 的
  既有调用）。
- 集级 ``stars``（=API ``order``）注入到该集所有 diff，是 set 级近似（RFC-0024
  诚实记录）。未 Ranked（``order==0.0``）注入 ``difficulty_rating=0.0``，下游
  ``0.0 < 3.0`` 自动剔除，符合 §4.3「community-validated」语义。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from beatmorph.core.logging import get_logger

logger = get_logger(__name__)

# sayobot approved 状态码：1=Ranked，3=Pending/WIP（无星级，order=0.0）
APPROVED_PENDING = 3


@dataclass(frozen=True, slots=True)
class SetRecord:
    """manifest 中一条谱面集记录（set-level）。

    所有字段来源于 sayobot 列表 API（``sid/title/artist/creator/stars/
    play_count/approved/modes``）或下载后统计（``n_diffs/downloaded_at/path``）。
    """

    sid: int
    title: str
    artist: str
    creator: str
    stars: float  # API `order`，未 Ranked/Pending 时 0.0
    play_count: int  # 集级总 play
    approved: int  # 1 Ranked / 3 Pending 等
    modes: int  # 位掩码，bit3==8 即 mania
    downloaded_at: str  # ISO8601
    n_diffs: int  # 解压后该集 .osu 数
    path: str  # 相对 raw_dir 的目录名（=str(sid)）
    unranked: bool  # 派生：stars==0.0 or approved==3


def load_manifest(path: Path) -> dict[int, SetRecord]:
    """解析 JSONL manifest → ``{sid: SetRecord}``。

    坏行（JSON 解析失败 / 字段缺失 / 类型错）记 warning 并跳过，不中断。
    """
    records: dict[int, SetRecord] = {}
    if not path.exists():
        logger.warning("Manifest file does not exist: %s", path)
        return records

    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rec = _record_from_dict(obj)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                logger.warning("Skipping manifest line %d (malformed): %s", lineno, exc)
                continue
            records[rec.sid] = rec

    logger.info("Manifest loaded: %d sets from %s", len(records), path)
    return records


def build_injector(records: dict[int, SetRecord]) -> ManifestInjector:
    """构造 manifest 注入器（供 :class:`PreprocessPipeline` 使用）。"""
    return ManifestInjector(records)


class ManifestInjector:
    """按 ``beatmap_set_id`` 把集级元数据注入 ``Chart.meta``。

    仅在 :meth:`apply` 时合并字段，**只加不覆盖**。不持有可变状态，线程安全。
    """

    def __init__(self, records: dict[int, SetRecord]) -> None:
        self._records: dict[int, SetRecord] = dict(records)

    def __len__(self) -> int:
        return len(self._records)

    def apply(self, chart_meta: dict[str, str | int | float]) -> bool:
        """按 ``chart_meta["beatmap_set_id"]`` 注入集级元数据。

        注入键（仅当缺失时写入）：
            - ``difficulty_rating`` (float) ← SetRecord.stars
            - ``playcount`` (int)           ← SetRecord.play_count
            - ``license`` (str)             ← "academic"（§6，setdefault 语义）
            - ``approved`` (int)            ← SetRecord.approved（审计用）

        Returns:
            True 若命中 manifest 并注入了至少一个字段；False 若 sid 缺失 / 不在
            manifest（此时 meta 不变，调用方退回宽松通过）。
        """
        sid_raw = chart_meta.get("beatmap_set_id")
        if not isinstance(sid_raw, int):
            return False

        rec = self._records.get(sid_raw)
        if rec is None:
            return False

        chart_meta.setdefault("difficulty_rating", float(rec.stars))
        chart_meta.setdefault("playcount", int(rec.play_count))
        chart_meta.setdefault("license", "academic")
        chart_meta.setdefault("approved", int(rec.approved))
        return True


# ── 辅助 ──────────────────────────────────────────────────────


def _record_from_dict(obj: object) -> SetRecord:
    """从 manifest 一行 JSON dict 构造 :class:`SetRecord`（带类型校验）。"""
    if not isinstance(obj, dict):
        raise TypeError(f"manifest line is not a JSON object: {type(obj).__name__}")

    try:
        sid = int(obj["sid"])
        stars = float(obj.get("stars", 0.0))
        play_count = int(obj.get("play_count", 0))
        approved = int(obj.get("approved", 0))
    except (KeyError, TypeError, ValueError) as exc:
        raise TypeError(f"missing/invalid required field: {exc}") from exc

    modes = int(obj.get("modes", 0))
    n_diffs = int(obj.get("n_diffs", 0))
    unranked = bool(obj.get("unranked", stars == 0.0 or approved == APPROVED_PENDING))

    return SetRecord(
        sid=sid,
        title=str(obj.get("title", "")),
        artist=str(obj.get("artist", "")),
        creator=str(obj.get("creator", "")),
        stars=stars,
        play_count=play_count,
        approved=approved,
        modes=modes,
        downloaded_at=str(obj.get("downloaded_at", "")),
        n_diffs=n_diffs,
        path=str(obj.get("path", str(sid))),
        unranked=unranked,
    )
