"""BPE event tokenizer 原子层：Chart ↔ atomic event-name list（RFC-0028）。

范式中立（无 torch / HF 依赖），仅依赖 :mod:`beatmorph.core.contracts`。被
:mod:`beatmorph.tokenizer.bpe` 的 :class:`~beatmorph.tokenizer.bpe.BPETokenizer`
在内部调用，也被 eval/roundtrip 测试直接复用。

原子 event schema（mania 4K REMI 变体，奠基 §3.2.2）：

  - 哨兵: ``PAD`` / ``BOS`` / ``EOS`` / ``SEP``（仅在 BPE 词表登记，不出现在 atomic 流里）
  - ``BAR``               每 bar 一个，边界由 :func:`compute_bar_boundaries`
                          （RFC-0026 phase 对齐 + RFC-0005 变速分段）确定
  - ``POS_i``             i ∈ [0, beats_per_bar * ``POS_DIVISIONS_PER_BEAT``)，
                          1/48-beat sub-beat 位置（4/4 → 每 bar 192 格）
  - ``NUDGE_j``           j ∈ [0, ``NUDGE_BUCKETS``)，仅当 note 偏离 POS 格超过
                          ``NUDGE_TOL_S`` 时发射 → 使原子流**可证明无损**
  - ``NOTEV_<lane>_<type>`` lane ∈ [0,4) × NoteType ∈ {TAP,HOLD,MINE,ROLL,FAKE} = 20 种
  - ``DUR_k``             k ∈ [0, ``N_DUR_BUCKETS``)，仅 HOLD/ROLL 发；
                          TAP/MINE/FAKE duration=0 不发

v1 **不含** Tempo event（BPM 仅作 planner/RAG 条件，AR 不生成绝对 tempo，
遵守 planner-tokenizer 正交，RFC-0028 决策 4）。
v1 **不手编** Hand enum（让 BPE 从共现 NOTEV 自动发现手型 = RFC-0028 §3.2.2）。

**无损性保证**（NUDGE 数学，奠基 §3.2.2）：POS 网格 = 1/48 拍，
``pos_step(秒) = (60/bpm)/48``。``POS_i = round(note_rel / pos_step)``，残差
``r = note_rel - i*pos_step``，``|r| ≤ pos_step/2``。NUDGE 用 ``N`` 桶覆盖
``±pos_step/2``，桶宽 ``= pos_step/N``，NUDGE 量化误差 ``≤ pos_step/(2N)``。
总最坏解码误差 ``= pos_step*(1/2 + 1/(2N))``。``N=12`` → 系数 0.5417，120bpm 下
5.6ms ≪ ±20ms 容差；约束 ``pos_step < 36.9ms → bpm > 33.9``，对所有 mania BPM 安全。

核心不变式（测试断言 measure==1.0，非 <1.0）：
    ``decode_atomic(encode_atomic(chart), chart.bpm_points)`` 必须在 lane 精确 +
    ``|Δt| ≤ 20ms`` 内重建每个 Note——这即 RFC-0028 §2.3「不量化信息」的可证伪断言。
"""

from __future__ import annotations

import bisect
import math

from beatmorph.core.contracts import (
    NUDGE_BUCKETS,
    POS_DIVISIONS_PER_BEAT,
    BpmPoint,
    Chart,
    GameMode,
    Note,
    NoteType,
)
from beatmorph.core.logging import get_logger
from beatmorph.data.parsers.osu_path import compute_bar_boundaries

logger = get_logger(__name__)

# ── 常量 ──────────────────────────────────────────────────────
_N_NOTE_TYPES = 5  # TAP/HOLD/MINE/ROLL/FAKE（契约 NoteType 枚举数）
N_DUR_BUCKETS = 24  # HOLD/ROLL duration log-spaced 桶数（0.05s..4.0s）
_DUR_MIN_S = 0.05
_DUR_MAX_S = 4.0
# 残差量化噪声阈：|r| ≤ 此值视为 on-grid 不发 NUDGE（纯浮点噪声，~0.25ms@120bpm）
NUDGE_TOL_S = 0.00025
_DEFAULT_METER = 4  # 默认拍号 4/4
_DEFAULT_BPM_FALLBACK = 120.0

# DUR log-spaced 桶边界（module 级常量，确定性）
_DUR_BUCKETS_S: tuple[float, ...] = tuple(
    _DUR_MIN_S * (_DUR_MAX_S / _DUR_MIN_S) ** (k / (N_DUR_BUCKETS - 1))
    for k in range(N_DUR_BUCKETS)
)


# ── 原子 event 名 ──────────────────────────────────────────────


def _bar_name() -> str:
    return "BAR"


def _pos_name(i: int) -> str:
    return f"POS_{i}"


def _nudge_name(j: int) -> str:
    return f"NUDGE_{j}"


def _notev_name(lane: int, ty: NoteType) -> str:
    return f"NOTEV_{lane}_{int(ty)}"


def _dur_name(k: int) -> str:
    return f"DUR_{k}"


# ── POS / NUDGE / DUR 数学 ─────────────────────────────────────


def compute_pos_index(
    note_time_s: float,
    bar_start_s: float,
    bar_dur_s: float,
    beats_per_bar: int = _DEFAULT_METER,
) -> tuple[int, float]:
    """返回 ``(POS_i, residual_s)``。

    ``rel = (note_time - bar_start) / bar_dur`` ∈ [0,1]；``pos_step = 1 /
    (beats_per_bar * POS_DIVISIONS_PER_BEAT)``（相对单位）；``POS_i = round(rel /
    pos_step) % pos_count``。返回的 ``residual``（秒）= ``note_time - (bar_start +
    i * pos_step_s)``，供 :func:`nudge_bucket` 决定是否发 NUDGE。

    Args:
        note_time_s: note 击打时刻（秒）。
        bar_start_s: 所属小节起点（秒，phase 对齐后）。
        bar_dur_s: 小节时长（秒）。
        beats_per_bar: 拍号（默认 4，4/4）。
    Returns:
        ``(POS_i, residual_s)``，``POS_i ∈ [0, pos_count)``，``residual_s`` 带
        符号（正=note 晚于 POS 格，负=早于）。
    """
    if bar_dur_s <= 0:
        return 0, 0.0
    pos_count = beats_per_bar * POS_DIVISIONS_PER_BEAT
    pos_step_s = bar_dur_s / pos_count
    rel = (note_time_s - bar_start_s) / bar_dur_s
    # 左对齐 bin 区间 + wrap（与 RFC-0026 phase 同口径）：rel∈[0,1) → POS=round(rel*count)%count
    pos_i = round(rel * pos_count) % pos_count
    grid_time = bar_start_s + pos_i * pos_step_s
    residual = note_time_s - grid_time
    return pos_i, residual


def nudge_bucket(residual_s: float, pos_step_s: float) -> int | None:
    """残差映射到 NUDGE bucket index；``|residual| ≤ NUDGE_TOL_S`` 返回 ``None``。

    NUDGE 覆盖 ``±pos_step/2``，``N = NUDGE_BUCKETS`` 桶，桶宽 ``= pos_step/N``。
    bucket 0..N-1 线性映射 ``[-pos_step/2, +pos_step/2]``（中点偏移 N/2）。
    超出 ``±pos_step/2``（理论上不应发生，POS 已取最近格）钳到端点桶。

    Args:
        residual_s: :func:`compute_pos_index` 返回的残差（秒，带符号）。
        pos_step_s: 单 POS 步长（秒）。
    Returns:
        bucket index ∈ [0, NUDGE_BUCKETS)，或 ``None``（不发 NUDGE）。
    """
    if abs(residual_s) <= NUDGE_TOL_S:
        return None
    if pos_step_s <= 0:
        return None
    bucket_w = pos_step_s / NUDGE_BUCKETS
    half = pos_step_s / 2.0
    clamped = max(-half, min(half, residual_s))
    # residual∈[-half, +half] → bucket∈[0, N-1]，中点 N/2
    idx = round(clamped / bucket_w + NUDGE_BUCKETS / 2)
    return max(0, min(NUDGE_BUCKETS - 1, idx))


def dur_bucket(duration_s: float) -> int:
    """HOLD/ROLL duration 映射到 log-spaced DUR bucket（``[0, N_DUR_BUCKETS)``）。

    bucket 边界见 :data:`_DUR_BUCKETS_S`（0.05s..4.0s log-spaced）。
    ``duration ≤ _DUR_MIN_S`` → 0；``duration ≥ _DUR_MAX_S`` → ``N-1``。

    Args:
        duration_s: HOLD/ROLL 长度（秒，>0）。
    Returns:
        bucket index。
    """
    if duration_s <= _DUR_BUCKETS_S[0]:
        return 0
    if duration_s >= _DUR_BUCKETS_S[-1]:
        return N_DUR_BUCKETS - 1
    # 二分查找所属桶（duration < buckets[k+1] 的首个 k+1，减 1）
    return max(0, bisect.bisect_right(_DUR_BUCKETS_S, duration_s) - 1)


def bucket_center_time(
    bar_start_s: float,
    pos_i: int,
    nudge_idx: int | None,
    bar_dur_s: float,
    beats_per_bar: int = _DEFAULT_METER,
) -> float:
    """POS + NUGE 桶中心 → 解码时刻（秒）。

    解码时把 POS 格 + NUDGE 桶中心还原为绝对时间。NUDGE 桶中心相对 POS 格的偏移
    = ``(nudge_idx - N/2) * bucket_w``，``bucket_w = pos_step/N``。

    Args:
        bar_start_s: 小节起点（秒）。
        pos_i: Position 索引。
        nudge_idx: NUDGE bucket index（或 ``None`` = 无残差）。
        bar_dur_s: 小节时长（秒）。
        beats_per_bar: 拍号。
    Returns:
        note 击打时刻（秒）。
    """
    pos_count = beats_per_bar * POS_DIVISIONS_PER_BEAT
    pos_step_s = bar_dur_s / pos_count
    grid_time = bar_start_s + pos_i * pos_step_s
    if nudge_idx is None:
        return grid_time
    bucket_w = pos_step_s / NUDGE_BUCKETS
    offset = (nudge_idx - NUDGE_BUCKETS / 2.0) * bucket_w
    return grid_time + offset


def dur_bucket_center(k: int) -> float:
    """DUR 桶 index → 中心 duration（秒），解码 HOLD/ROLL 长度用。"""
    if k <= 0:
        return _DUR_BUCKETS_S[0]
    if k >= N_DUR_BUCKETS - 1:
        return _DUR_BUCKETS_S[-1]
    # 几何中心
    return math.sqrt(_DUR_BUCKETS_S[k] * _DUR_BUCKETS_S[k + 1])


# ── Chart → atomic event-name 序列 ─────────────────────────────


def encode_atomic(chart: Chart, beats_per_bar: int = _DEFAULT_METER) -> list[str]:
    """Chart → 原子 event-name 字符串列表（按时间顺序）。

    流程：
        1. ``chart.sorted_notes()`` + :func:`compute_bar_boundaries` 推 phase 对齐小节边界
        2. 逐 bar 发 ``BAR``；归属该 bar 的 note 依次发
           ``POS_i`` / ``[NUDGE_j]`` / ``NOTEV_<lane>_<type>`` / ``[DUR_k]``
        3. 空小节仍发 ``BAR``（保留段位结构，BPE 可学 ``BAR BAR BAR`` = rest）

    Args:
        chart: 谱面 IR（``notes`` 按 :meth:`Chart.sorted_notes` 升序处理）。
        beats_per_bar: 拍号（默认 4）。
    Returns:
        原子 event-name 字符串列表；空谱面返回 ``[]``。
    """
    sorted_notes = chart.sorted_notes()
    if not sorted_notes:
        return []

    note_duration = sorted_notes[-1].time + max(n.duration for n in sorted_notes)
    audio_dur = chart.meta.get("audio_duration")
    audio_total = (
        float(audio_dur) if isinstance(audio_dur, int | float) and audio_dur > 0 else note_duration
    )
    total_duration = max(audio_total, note_duration)

    boundaries = compute_bar_boundaries(chart.bpm_points, total_duration)
    pos_count = beats_per_bar * POS_DIVISIONS_PER_BEAT

    events: list[str] = []
    note_idx = 0
    n_notes = len(sorted_notes)
    for i in range(len(boundaries) - 1):
        t_start = float(boundaries[i])
        t_end = float(boundaries[i + 1])
        bar_dur = t_end - t_start
        if bar_dur <= 0:
            continue
        events.append(_bar_name())
        pos_step_s = bar_dur / pos_count
        # 归属本 bar 的 note（左闭右开，末 bar 用闭区间收尾）
        is_last_bar = i == len(boundaries) - 2
        while note_idx < n_notes:
            note = sorted_notes[note_idx]
            in_bar = (t_start <= note.time < t_end) or (is_last_bar and note.time == t_end)
            if not in_bar:
                break
            pos_i, residual = compute_pos_index(note.time, t_start, bar_dur, beats_per_bar)
            events.append(_pos_name(pos_i))
            n_idx = nudge_bucket(residual, pos_step_s)
            if n_idx is not None:
                events.append(_nudge_name(n_idx))
            events.append(_notev_name(note.lane, note.type))
            if note.is_hold() and note.duration > 0:
                events.append(_dur_name(dur_bucket(note.duration)))
            note_idx += 1
    return events


# ── atomic event-name 序列 → Chart ─────────────────────────────


def decode_atomic(
    events: list[str],
    chart_bpm_points: list[BpmPoint],
    beats_per_bar: int = _DEFAULT_METER,
) -> Chart:
    """原子 event-name 列表 → Chart（状态机扫描）。

    状态机：遇 ``BAR`` 累加 bar_index（bar 起点时间由
    :func:`compute_bar_boundaries` 反查，需先推断 total_duration）；遇 ``POS``
    更新 ``current_grid_time``；遇 ``NUDGE`` 更新 ``current_nudge``；遇 ``NOTEV``
    发 :class:`Note`（time = ``bucket_center_time``，type 由 event 名解析）；遇
    ``DUR`` 更新 ``pending_duration``（作用于上一个 HOLD/ROLL note）。

    ``chart_bpm_points`` 原样保留在输出 Chart 上（RFC-0028 决策 4：BPM 不由 AR
    生成，decode 用谱面 bpm_points 做 POS→秒，支持变速分段）。

    Args:
        events: 原子 event-name 字符串列表（:func:`encode_atomic` 产物或 BPE 解合并后）。
        chart_bpm_points: 目标谱面 BPM 变速点（至少 1 个，按 time 升序）。
        beats_per_bar: 拍号（默认 4）。
    Returns:
        Chart（``mode=MANIA_4K``，``bpm_points=chart_bpm_points``，note 按 time 升序）。
    """
    if not chart_bpm_points:
        chart_bpm_points = [BpmPoint(time=0.0, bpm=_DEFAULT_BPM_FALLBACK)]
    pos_count = beats_per_bar * POS_DIVISIONS_PER_BEAT

    # 第一遍：扫出 bar 数 + 末 note 相对时间，推 total_duration（用于 bar 边界反查）
    # 因 decode 时 bar 起点未知，需先估总时长再 compute_bar_boundaries。
    # 策略：边扫边按 bar 累进，用「当前 bar 起点时间 + bar_dur」增长；bar_dur 由
    # bpm_points 分段决定。为支持变速，必须先有 total_duration 才能算 boundaries ——
    # 但 total_duration 又依赖 note 时间。解法：两次扫，第一次只数 bar 数 + 估末 bar，
    # 用末 note 估出 total_duration 下界，再正式 boundaries。
    notes: list[Note] = []
    # 用 chart_bpm_points 推一个 total_duration 下界：末 note 的时间需在扫中确定,
    # 故一遍扫 + 边推边界（动态 bpm，与 compute_bar_boundaries 同逻辑但增量推进）。
    bar_times = _scan_bar_times(events, chart_bpm_points, beats_per_bar, pos_count)
    # bar_times[i] = 第 i 个 BAR 的起点秒（phase 对齐），长度 = bar 数

    # 第二遍：正式发 note
    bar_idx = -1
    current_pos = 0
    current_nudge: int | None = None
    pending_hold: Note | None = None  # 上一个 HOLD/ROLL，待 DUR 补 duration

    def _flush_hold(dur_s: float) -> None:
        nonlocal pending_hold
        if pending_hold is not None:
            pending_hold = pending_hold.model_copy(update={"duration": max(0.0, dur_s)})
            notes.append(pending_hold)
            pending_hold = None

    for ev in events:
        if ev == _bar_name():
            # 进入新 bar 前先把未补 DUR 的 hold 兜底为 0（理论上不会发生）
            _flush_hold(0.0)
            bar_idx += 1
            current_pos = 0
            current_nudge = None
        elif ev.startswith("POS_"):
            current_pos = int(ev.split("_", 1)[1])
            current_nudge = None
        elif ev.startswith("NUDGE_"):
            current_nudge = int(ev.split("_", 1)[1])
        elif ev.startswith("NOTEV_"):
            _flush_hold(0.0)  # 新 note 前若有未补 hold，兜底
            _lane_s, _ty_s = ev[len("NOTEV_") :].split("_", 1)
            lane = int(_lane_s)
            ty = NoteType(int(_ty_s))
            if bar_idx < 0 or bar_idx >= len(bar_times):
                # 无 BAR 前导的孤立 note（不应出现），跳过
                continue
            bar_start = bar_times[bar_idx]
            bar_dur = _bar_duration_at(chart_bpm_points, bar_start)
            if bar_dur <= 0:
                bar_dur = 4 * 60.0 / _DEFAULT_BPM_FALLBACK
            time_s = bucket_center_time(
                bar_start, current_pos, current_nudge, bar_dur, beats_per_bar
            )
            note = Note(time=time_s, lane=lane, type=ty, duration=0.0)
            if note.is_hold():
                pending_hold = note  # 等 DUR 补 duration
            else:
                notes.append(note)
        elif ev.startswith("DUR_"):
            k = int(ev.split("_", 1)[1])
            _flush_hold(dur_bucket_center(k))
        # 哨兵 / 未知 event 忽略（PAD/BOS/EOS/SEP 不在 atomic 流）

    _flush_hold(0.0)  # 末尾未补 hold 兜底

    notes.sort(key=lambda n: (n.time, n.lane))
    bpm = chart_bpm_points[0].bpm if chart_bpm_points else _DEFAULT_BPM_FALLBACK
    return Chart(
        version="ir-1",
        mode=GameMode.MANIA_4K,
        difficulty=1,
        bpm_points=[BpmPoint(time=bp.time, bpm=bp.bpm) for bp in chart_bpm_points],
        notes=notes,
        meta={"decoded_bpm_primary": bpm},
    )


# ── 辅助：动态推 bar 起点 + bar_dur ────────────────────────────


def _bar_duration_at(
    bpm_points: list[BpmPoint], time_s: float, meter: int = _DEFAULT_METER
) -> float:
    """返回 ``time_s`` 时刻生效的 BPM 对应的小节时长（秒）。"""
    if not bpm_points:
        return 4 * 60.0 / _DEFAULT_BPM_FALLBACK
    # 找 time_s 所属 bpm 段（最后一个 time <= time_s 的点）
    bpm = bpm_points[0].bpm
    for bp in bpm_points:
        if bp.time <= time_s + 1e-9:
            bpm = bp.bpm
        else:
            break
    return meter * 60.0 / max(bpm, 1e-6)


def _scan_bar_times(
    events: list[str],
    bpm_points: list[BpmPoint],
    beats_per_bar: int,
    pos_count: int,
) -> list[float]:
    """第一遍扫：逐 BAR 推进 bar 起点时间，返回每个 BAR 的起点秒列表。

    phase 对齐：首个音乐 bar 起点 = ``bpm_points[0].time``（RFC-0026）。
    若 ``phase > 0``，bar_times[0] = 0.0（intro 段），首个音乐 bar = bar_times[1]。
    与 :func:`compute_bar_boundaries` 同口径，但增量推进（decode 时 total_duration 未知）。
    """
    phase = bpm_points[0].time if bpm_points else 0.0
    bar_times: list[float] = []
    # intro 段：phase>0 时首个 BAR 视为 [0, phase] 的 intro（与 compute_bar_boundaries 一致）
    if phase > 0:
        bar_times.append(0.0)
        current = phase
    else:
        current = 0.0  # phase==0：首 bar 起 0
    first_music_bar = True
    for ev in events:
        if ev != _bar_name():
            continue
        if first_music_bar and phase > 0:
            # 第一个音乐 bar 起点 = phase（intro 已占 bar_times[0]）
            bar_times.append(phase)
            current = phase
            first_music_bar = False
            continue
        if first_music_bar:
            # phase==0：首 bar 起 0
            bar_times.append(0.0)
            first_music_bar = False
            continue
        bar_dur = _bar_duration_at(bpm_points, current, beats_per_bar)
        current += bar_dur
        bar_times.append(current)
    return bar_times
