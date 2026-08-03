"""osu!mania (.osu) 格式读写。

奠基文档 §1.1 优先支持 4K VSRG，输出 .osu。
本文件实现 ChartReader（解析 .osu → Chart IR），Writer 留 Phase 2。

.osu 文件格式权威参考：docs/knowledges/osu-file.md
"""

from __future__ import annotations

import contextlib
from pathlib import Path

from beatmorph.core.contracts import BpmPoint, Chart, GameMode, Note, NoteType
from beatmorph.core.logging import get_logger
from beatmorph.io.formats.base import ChartReader, ChartWriter

logger = get_logger(__name__)

# ── 编码兜底顺序 ──
_FALLBACK_ENCODINGS = ["utf-8", "cp936", "latin-1"]

# ── osu! Mode 常量 ──
_MODE_MANIA = 3  # osu!mania

# CircleSize(key_count) → GameMode 诚实映射（RFC-0025）。
# 非 4K 不降级，避免 lane_count() 与真实键位不一致导致 parse_osu 误删 lane。
_MANIA_MODES: dict[int, GameMode] = {
    4: GameMode.MANIA_4K,
    5: GameMode.MANIA_5K,
    6: GameMode.MANIA_6K,
    7: GameMode.MANIA_7K,
    8: GameMode.MANIA_8K,
}


class OsuManiaReader(ChartReader):
    """解析 .osu → Chart IR（数据预处理流水线使用）。

    仅支持 osu!mania (Mode=3)。非 mania 模式返回空 Chart 并记 warning。

    Implementation per:
    - docs/knowledges/osu-file.md（格式权威参考）
    - RFC-0001（Note.time 单位秒，内部换算 ms→秒）
    - RFC-0005（bpm_points 数组，仅提取非继承红线）
    """

    def __init__(self, encoding: str = "utf-8") -> None:
        self._preferred_encoding = encoding

    def suffix(self) -> str:
        return ".osu"

    # ── 公共入口 ──────────────────────────────────────────────

    def read(self, path: Path) -> Chart:
        """解析 .osu 文件为 Chart IR。"""
        content = self._decode_with_fallback(path)
        sections = self._split_sections(content)

        general = self._parse_general(sections.get("General", []))
        if general["mode"] != _MODE_MANIA:
            logger.warning(
                "File %s is not osu!mania (Mode=%d). Only Mode=3 supported in Phase 1. "
                "Returning empty Chart.",
                path,
                general["mode"],
            )
            return Chart(
                difficulty=1,
                bpm_points=[BpmPoint(time=0.0, bpm=120.0)],
                meta={"skip_reason": f"mode={general['mode']}"},
            )

        metadata = self._parse_metadata(sections.get("Metadata", []))
        difficulty = self._parse_difficulty(sections.get("Difficulty", []))
        bpm_points = self._parse_timing_points(sections.get("TimingPoints", []))
        notes = self._parse_hit_objects(sections.get("HitObjects", []), difficulty["key_count"])

        # ── 模式映射（CircleSize → GameMode）──
        # 诚实标记真实 K 数（RFC-0025）：非 4K mania 不降级为 4K，避免 lane_count()
        # 与真实键位不一致导致 parse_osu 误删 lane≥4 的 Note。4K 主路径靠训练层
        # mode==MANIA_4K 过滤守 R-6，数据层保留多 K 谱面备未来扩展。
        kc = int(difficulty["key_count"])
        mode = _MANIA_MODES.get(kc)
        if mode is None:
            logger.warning("Unusual key_count=%d, defaulting MANIA_4K", kc)
            mode = GameMode.MANIA_4K
        elif mode != GameMode.MANIA_4K:
            logger.info(
                "Non-4K mania key_count=%d → %s (kept for future, filtered at train)", kc, mode
            )

        # ── 难度代理（OD 0-10 → 1-15）──
        od = difficulty.get("overall_difficulty", 5.0)
        star_proxy = max(1, min(15, round(od * 1.5)))

        # ── 兜底 BPM ──
        if not bpm_points:
            logger.error("No uninherited timing points found in %s, fallback BPM=120", path)
            bpm_points = [BpmPoint(time=0.0, bpm=120.0)]

        extra_meta: dict[str, str | int | float] = {
            "creator": metadata.get("creator", "unknown"),
            "version": metadata.get("version", "unknown"),
            "key_count": kc,
            "od": od,
        }
        if metadata.get("beatmap_id"):
            extra_meta["beatmap_id"] = metadata["beatmap_id"]
        if metadata.get("beatmap_set_id"):
            extra_meta["beatmap_set_id"] = metadata["beatmap_set_id"]
        if metadata.get("tags"):
            extra_meta["tags"] = metadata["tags"]

        return Chart(
            mode=mode,
            difficulty=star_proxy,
            bpm_points=bpm_points,
            audio_path=general.get("audio_filename"),
            title=metadata.get("title", "untitled"),
            artist=metadata.get("artist", "unknown"),
            notes=notes,
            meta=extra_meta,
        )

    # ── 编码兜底 ──────────────────────────────────────────────

    def _decode_with_fallback(self, path: Path) -> str:
        """多编码兜底打开 .osu 文件。"""
        encodings = [self._preferred_encoding] + [
            e for e in _FALLBACK_ENCODINGS if e != self._preferred_encoding
        ]
        last_error: Exception | None = None
        for enc in encodings:
            try:
                return path.read_text(encoding=enc)
            except (UnicodeDecodeError, LookupError) as exc:
                last_error = exc
                logger.debug("Encoding %s failed for %s: %s", enc, path, exc)
                continue
        raise ValueError(f"Cannot decode {path} with encodings {encodings}") from last_error

    # ── Section 切分 ──────────────────────────────────────────

    @staticmethod
    def _split_sections(content: str) -> dict[str, list[str]]:
        """将 .osu 文件内容按 [Section] 标题切分为 dict。"""
        sections: dict[str, list[str]] = {}
        current_section: str | None = None

        for raw in content.splitlines():
            line = raw.strip()
            if not line:
                continue
            # 跳过格式版本行（如 "osu file format v14"）
            if line.startswith("osu file format"):
                continue
            if line.startswith("[") and line.endswith("]"):
                current_section = line[1:-1]
                sections[current_section] = []
            elif current_section is not None:
                sections[current_section].append(line)

        return sections

    # ── [General] 解析 ────────────────────────────────────────

    @staticmethod
    def _parse_general(lines: list[str]) -> dict[str, str | int]:
        """解析 [General] 节，返回 {mode, audio_filename, ...}。"""
        result: dict[str, str | int] = {"mode": 0}
        for line in lines:
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key, value = key.strip(), value.strip()
            if key == "Mode":
                try:
                    result["mode"] = int(value)
                except ValueError:
                    logger.warning("Invalid Mode value: %r", value)
            elif key == "AudioFilename":
                result["audio_filename"] = value.strip()
        return result

    # ── [Metadata] 解析 ───────────────────────────────────────

    @staticmethod
    def _parse_metadata(lines: list[str]) -> dict[str, str | int]:
        """解析 [Metadata] 节，返回 title/artist/creator/...。"""
        result: dict[str, str | int] = {}
        for line in lines:
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key, value = key.strip(), value.strip()
            if key == "Title":
                result["title"] = value
            elif key == "Artist":
                result["artist"] = value
            elif key == "Creator":
                result["creator"] = value
            elif key == "Version":
                result["version"] = value
            elif key == "BeatmapID":
                with contextlib.suppress(ValueError):
                    result["beatmap_id"] = int(value)
            elif key == "BeatmapSetID":
                with contextlib.suppress(ValueError):
                    result["beatmap_set_id"] = int(value)
            elif key == "Tags":
                result["tags"] = value
        return result

    # ── [Difficulty] 解析 ─────────────────────────────────────

    @staticmethod
    def _parse_difficulty(lines: list[str]) -> dict[str, float]:
        """解析 [Difficulty] 节。

        CircleSize 在 mania 下即键位数（osu-file.md 明确说明）。
        """
        result: dict[str, float] = {"key_count": 4.0, "overall_difficulty": 5.0}
        for line in lines:
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            key, value = key.strip(), value.strip()
            try:
                fval = float(value)
            except ValueError:
                continue
            if key == "CircleSize":
                result["key_count"] = fval
            elif key == "OverallDifficulty":
                result["overall_difficulty"] = fval
        return result

    # ── [TimingPoints] 解析 ───────────────────────────────────

    @staticmethod
    def _parse_timing_points(lines: list[str]) -> list[BpmPoint]:
        """解析 [TimingPoints] → 仅提取非继承红线（uninherited=1, beatLength>0）。

        格式（osu-file.md）：
            time,beatLength,meter,sampleSet,sampleIndex,volume,uninherited,effects

        - beatLength > 0 → 红线，BPM = 60000 / beatLength
        - beatLength < 0 → 绿线（继承），跳过
        - uninherited == 1 → 红线
        """
        bpm_points: list[BpmPoint] = []
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 7:
                continue
            try:
                time_ms = float(parts[0])
                beat_length = float(parts[1])
                uninherited = int(parts[6])
            except ValueError:
                logger.debug("Skipping malformed TimingPoint: %r", line)
                continue

            # 仅非继承红线（beatLength > 0 且 uninherited == 1）
            if uninherited != 1 or beat_length <= 0:
                continue

            bpm = 60000.0 / beat_length
            time_s = max(0.0, time_ms / 1000.0)  # ms→秒 (RFC-0001)，钳到 >= 0
            bpm_points.append(BpmPoint(time=time_s, bpm=bpm))

        # 按 time 升序（已由 osu! 规范保证，此处防御排序）
        bpm_points.sort(key=lambda bp: bp.time)
        return bpm_points

    # ── [HitObjects] 解析 ─────────────────────────────────────

    @staticmethod
    def _parse_hit_objects(lines: list[str], key_count_raw: float) -> list[Note]:
        """解析 [HitObjects] → list[Note]。

        osu!mania (Mode=3) 物件：
        - type & 1  → 普通音符 → NoteType.TAP
        - type & 128 → 长按音符 → NoteType.HOLD（endTime 在 objectParams）
        - Slider（bit 1）/ Spinner（bit 3）: mania 不使用，跳过

        列位算法：lane = int(x * key_count / 512)
        时间换算：ms → 秒 (RFC-0001)
        """
        key_count = int(key_count_raw)
        if key_count < 1:
            key_count = 4  # 防御

        notes: list[Note] = []
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue

            try:
                x = int(parts[0])
                # y = int(parts[1])  # mania 不使用 y
                time_ms = int(parts[2])
                hit_type = int(parts[3])
                # hit_sound = int(parts[4])  # 暂不解析音效
            except ValueError:
                logger.debug("Skipping malformed HitObject: %r", line)
                continue

            # ── 跳过负时间 Note（Note.time 受 pydantic NonNegativeFloat 约束）──
            if time_ms < 0:
                logger.debug("Skipping HitObject with negative time=%dms", time_ms)
                continue

            # ── 长按音符（bit 7 = 128）──
            if hit_type & 128:
                try:
                    end_time_ms = int(parts[5]) if len(parts) > 5 else time_ms
                except ValueError:
                    logger.debug("Invalid endTime in HOLD: %r", line)
                    end_time_ms = time_ms

                time_s = time_ms / 1000.0
                duration_s = (end_time_ms - time_ms) / 1000.0

                if duration_s <= 0:
                    logger.warning(
                        "HOLD note at t=%dms has invalid endTime=%d (duration≤0), "
                        "degrading to TAP",
                        time_ms,
                        end_time_ms,
                    )
                    notes.append(
                        Note(
                            time=time_s,
                            lane=int(x * key_count / 512),
                            type=NoteType.TAP,
                        )
                    )
                else:
                    notes.append(
                        Note(
                            time=time_s,
                            lane=int(x * key_count / 512),
                            type=NoteType.HOLD,
                            duration=duration_s,
                        )
                    )

            # ── 普通音符（bit 0 = 1）──
            elif hit_type & 1:
                time_s = time_ms / 1000.0
                notes.append(
                    Note(
                        time=time_s,
                        lane=int(x * key_count / 512),
                        type=NoteType.TAP,
                    )
                )

            else:
                # Slider (bit 1=2), Spinner (bit 3=8), 或未知类型
                logger.debug(
                    "Skipping unsupported HitObject type=%d (bits: %s) — "
                    "sliders/spinners not used in osu!mania.",
                    hit_type,
                    _type_bits_str(hit_type),
                )

        # 按时间升序排序（部分编辑器可能不保证顺序）
        notes.sort(key=lambda n: (n.time, n.lane))
        return notes


# ── 辅助 ──────────────────────────────────────────────────────


def _type_bits_str(hit_type: int) -> str:
    """将 HitObject type 值展开为位描述（调试用）。"""
    bits = []
    names = {
        0: "circle",
        1: "slider",
        2: "new_combo",
        3: "spinner",
        7: "mania_hold",
    }
    for b, name in names.items():
        if hit_type & (1 << b):
            bits.append(f"{b}:{name}")
    if hit_type & 0x70:  # bits 4-6 combo skip
        skip = (hit_type >> 4) & 7
        bits.append(f"4-6:skip_{skip}")
    return "|".join(bits) if bits else "none"


# ── Writer（Phase 2 占位）────────────────────────────────────


class OsuManiaWriter(ChartWriter):
    """写出 osu!mania 谱面。Phase 2 实现。"""

    def suffix(self) -> str:
        return ".osu"

    def write(self, chart: Chart, path: Path) -> Path:
        raise NotImplementedError("OsuManiaWriter 未实现，见 docs/plans/07-decoder-postprocess.md")
