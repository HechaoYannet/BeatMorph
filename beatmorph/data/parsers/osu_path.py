"""osu! 谱面解析器（.osu → Chart IR）+ 段落统计量。

奠基文档 §4.2 Step 1-2：解析 .osu → 清理 → 自动计算统计量（密度、段落边界、BPM）
用于 Stage 1 伪标签。

数据来源：docs/knowledges/osu-file.md（格式权威参考）
"""

from __future__ import annotations

from pathlib import Path

from beatmorph.core.contracts import BpmPoint, Chart, Note, Section
from beatmorph.core.logging import get_logger
from beatmorph.io.formats.osu import OsuManiaReader

logger = get_logger(__name__)

# ── 常数 ──────────────────────────────────────────────────────
_DEFAULT_METER = 4  # 默认拍号 4/4


def parse_osu(path: Path) -> Chart:
    """解析单个 .osu 文件为 Chart IR，叠加清理规则。

    奠基文档 §4.2 Step1。
    委托 :class:`OsuManiaReader` 完成格式解析，本函数只负责清理规则：
        - 移除 ``time < 0`` 的 Note（RFC-0001 约束）
        - 移除 ``lane >= lane_count()`` 的越界 Note
        - 空文件（无 Note）记 debug log
        - ``bpm_points`` 意外为空时 fallback BPM=120
    """
    try:
        chart = OsuManiaReader().read(path)
    except Exception:
        logger.exception("Failed to read .osu file: %s", path)
        return _empty_chart()

    # ── 清理 ──
    lane_max = chart.lane_count()
    valid_notes: list[Note] = []
    dropped_negative = 0
    dropped_oob = 0

    for note in chart.notes:
        if note.time < 0:
            dropped_negative += 1
            logger.debug("Dropping note with negative time %.3f at %s", note.time, path)
            continue
        if lane_max > 0 and note.lane >= lane_max:
            dropped_oob += 1
            logger.debug(
                "Dropping out-of-bounds note lane=%d (max=%d) at %s",
                note.lane,
                lane_max,
                path,
            )
            continue
        valid_notes.append(note)

    if dropped_negative:
        logger.info("Dropped %d notes with negative time from %s", dropped_negative, path)
    if dropped_oob:
        logger.info("Dropped %d out-of-bounds notes from %s", dropped_oob, path)

    if not valid_notes:
        logger.debug("No valid notes after cleanup in %s", path)

    chart.notes = valid_notes

    # ── 兜底 BPM ──
    if not chart.bpm_points:
        logger.error("No bpm_points extracted from %s, fallback BPM=120", path)
        chart.bpm_points = [BpmPoint(time=0.0, bpm=120.0)]

    return chart


def compute_section_stats(chart: Chart, section_bars: int = 4) -> Chart:
    """自动计算段落统计量，回填 ``chart.sections``。

    奠基文档 §4.2 Step2：作为 Stage 1 自监督回归的伪标签来源。

    算法：
        1. 按 ``bpm_points`` 推导小节边界
        2. 以 ``section_bars`` 小节为窗口切 Section，覆盖全曲
        3. 每 Section 计算 density / energy / rest / type
    """
    if not chart.notes:
        logger.debug("No notes in chart, returning empty sections")
        return chart

    sorted_notes = chart.sorted_notes()
    note_duration = sorted_notes[-1].time + max(n.duration for n in sorted_notes)
    # 优先用音频全长（PreprocessPipeline 注入），与推理时 plan()/extract_mert
    # 的 Section 边界同源；无音频时 fallback 到 Note 时长（向后兼容纯 .osu 测试）。
    # max(..., note_duration) 防御：音频时长异常短于末 Note 时回退 Note 时长，避免漏切尾部。
    audio_dur = chart.meta.get("audio_duration")
    audio_total = (
        float(audio_dur) if isinstance(audio_dur, (int, float)) and audio_dur > 0 else note_duration
    )
    total_duration = max(audio_total, note_duration)

    # ── 1. 推导小节边界 ──
    bar_boundaries = _compute_bar_boundaries(chart.bpm_points, total_duration)
    if len(bar_boundaries) < 2:
        logger.warning("Cannot compute bar boundaries, returning empty sections")
        return chart

    # ── 2. 切分 Section ──
    sections: list[Section] = []
    num_bars = len(bar_boundaries) - 1
    section_count = max(1, num_bars // section_bars)

    # 全曲峰值 NPS（用于归一化）
    peak_nps = _compute_peak_nps(sorted_notes, bar_boundaries, section_bars)

    for sec_idx in range(section_count):
        bar_start = sec_idx * section_bars
        bar_end = min(bar_start + section_bars, num_bars)
        t_start = bar_boundaries[bar_start]
        t_end = bar_boundaries[bar_end]
        duration = t_end - t_start

        if duration <= 0:
            continue

        # ── 段内 Note ──
        sec_notes = [n for n in sorted_notes if t_start <= n.time < t_end]
        nps = len(sec_notes) / duration if duration > 0 else 0.0
        density_target = nps / peak_nps if peak_nps > 0 else 0.0

        # HOLD 比例
        hold_count = sum(1 for n in sec_notes if n.is_hold())
        hold_ratio = hold_count / len(sec_notes) if sec_notes else 0.0

        # energy = density * 0.7 + hold_ratio * 0.3
        energy_level = density_target * 0.7 + hold_ratio * 0.3

        # rest_probability = 间隔 > 2 * mean_gap 的比例
        rest_prob = _compute_rest_probability(sec_notes, duration)

        # sections_type 启发式
        sec_type = _classify_section_type(sec_idx, section_count, density_target)

        sections.append(
            Section(
                index=sec_idx,
                start_time=t_start,
                end_time=t_end,
                bar_count=bar_end - bar_start,
                density_target=min(1.0, max(0.0, density_target)),
                energy_level=min(1.0, max(0.0, energy_level)),
                rest_probability=min(1.0, max(0.0, rest_prob)),
                sections_type=sec_type,
            )
        )

    chart.sections = sections
    return chart


# ── 辅助函数 ──────────────────────────────────────────────────


def _compute_bar_boundaries(bpm_points: list[BpmPoint], total_duration: float) -> list[float]:
    """按 bpm_points 分段推导所有小节边界时间点（秒）。

    返回 list[float]，长度 = 小节数 + 1（最后一个为 total_duration）。

    **相位对齐（RFC-0026）**：osu! 非继承红线（uninherited timing point）的 ``time``
    字段是节拍网格相位原点（downbeat 时刻）。小节网格 = ``phase + k × bar_dur``，
    ``phase = bpm_points[0].time``，而非从 ``0.0`` 起算。``phase > 0`` 时首段
    ``[0, phase]`` 为「网格前 intro 段」保留为独立边界（防丢该段 Note）；``phase == 0``
    时等价旧行为。
    """
    if not bpm_points:
        return [0.0, total_duration]

    phase = bpm_points[0].time
    # 首边界恒为 0.0；phase>0 时次边界补 [0, phase] intro 段（防丢该段 Note）
    boundaries: list[float] = [0.0]
    if phase > 0:
        boundaries.append(min(phase, total_duration))
    current_time = phase
    bp_idx = 0

    while current_time < total_duration:
        # 找到当前时间段的 BPM
        while bp_idx + 1 < len(bpm_points) and bpm_points[bp_idx + 1].time <= current_time:
            bp_idx += 1
        bpm = bpm_points[bp_idx].bpm
        bar_duration = _DEFAULT_METER * 60.0 / bpm
        current_time += bar_duration
        if current_time <= total_duration:
            boundaries.append(current_time)
        else:
            boundaries.append(total_duration)
            break

    # 确保终点 >= total_duration
    if boundaries[-1] < total_duration:
        boundaries.append(total_duration)

    return boundaries


def _compute_peak_nps(
    sorted_notes: list[Note],
    bar_boundaries: list[float],
    section_bars: int,
) -> float:
    """计算全曲滑动窗口的最大 NPS（用于 density 归一化）。"""
    max_nps = 0.0
    num_bars = len(bar_boundaries) - 1

    for i in range(0, num_bars, max(1, section_bars // 2)):
        t_start = bar_boundaries[i]
        t_end = bar_boundaries[min(i + section_bars, num_bars)]
        duration = t_end - t_start
        if duration <= 0:
            continue
        count = sum(1 for n in sorted_notes if t_start <= n.time < t_end)
        nps = count / duration
        max_nps = max(max_nps, nps)

    return max_nps if max_nps > 0 else 1.0


def _compute_rest_probability(notes: list[Note], duration: float) -> float:
    """计算段内 Note 间隔 > 2 * mean_gap 的比例。"""
    if len(notes) < 2:
        return 0.0 if len(notes) > 0 else 1.0

    gaps = [notes[i + 1].time - notes[i].time for i in range(len(notes) - 1)]
    if not gaps:
        return 0.0

    mean_gap = sum(gaps) / len(gaps)
    if mean_gap <= 0:
        return 0.0

    long_gaps = sum(1 for g in gaps if g > 2.0 * mean_gap)
    # 权重：长时间间隔的比例
    return long_gaps / len(gaps)


def _classify_section_type(
    sec_idx: int,
    total_sections: int,
    density_target: float,
) -> str:
    """启发式区分 Section 类型。"""
    if total_sections <= 1:
        return "verse"
    if sec_idx == 0:
        return "intro"
    if sec_idx == total_sections - 1:
        return "outro"
    if density_target < 0.15:
        return "bridge"
    if density_target > 0.7:
        return "chorus"
    return "verse"


def _empty_chart() -> Chart:
    """构造一个合法的空 Chart（解析失败兜底）。"""
    return Chart(
        difficulty=1,
        bpm_points=[BpmPoint(time=0.0, bpm=120.0)],
        meta={"skip_reason": "parse_error"},
    )
