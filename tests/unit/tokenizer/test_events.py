"""BPE atomic event 层单元测试（RFC-0028，CPU 可跑，无 HF/torch 依赖）。

覆盖（对齐计划 Step 3 验收策略）：
- :func:`encode_atomic` / :func:`decode_atomic` 无损往返（``measure == 1.0``，非 <1.0）
- 确定性（同输入同输出）
- off-beat note 的 NUDGE 发射 + 解码重建误差 ≤ 5.6ms
- 变速曲（多个 BpmPoint）POS→秒分段映射、往返成立
- 相位对齐（RFC-0026 复现：``bpm_points[0].time=0.339``）
- ``dur_bucket`` / ``nudge_bucket`` / ``compute_pos_index`` 数学正确性
"""

from __future__ import annotations

from beatmorph.core.contracts import BpmPoint, Chart, GameMode, Note, NoteType
from beatmorph.data.parsers.osu_path import compute_bar_boundaries
from beatmorph.tokenizer.events import (
    N_DUR_BUCKETS,
    NUDGE_BUCKETS,
    NUDGE_TOL_S,
    POS_DIVISIONS_PER_BEAT,
    bucket_center_time,
    compute_pos_index,
    decode_atomic,
    dur_bucket,
    dur_bucket_center,
    encode_atomic,
    nudge_bucket,
)

# ── 测试夹具 ──────────────────────────────────────────────────


def _make_chart(
    notes: list[Note],
    bpm: float = 120.0,
    phase: float = 0.0,
    n_bars: int = 3,
) -> Chart:
    """构造测试谱面：默认 120bpm 4/4 → bar=2s。"""
    bar_dur = 4 * 60.0 / bpm
    return Chart(
        mode=GameMode.MANIA_4K,
        difficulty=5,
        bpm_points=[BpmPoint(time=phase, bpm=bpm)],
        notes=notes,
        meta={"audio_duration": max(n_bars * bar_dur, phase + n_bars * bar_dur)},
    )


# ── 无损往返（measure == 1.0）────────────────────────────────


def _measure(orig: Chart, recon: Chart, tol_s: float = 0.02) -> float:
    """贪心 note 匹配率（lane 精确 + |Δt|≤tol），与 eval.util 口径一致。"""
    if not orig.notes:
        return 1.0
    o_sorted = orig.sorted_notes()
    r_sorted = recon.sorted_notes()
    used = [False] * len(r_sorted)
    matched = 0
    for o in o_sorted:
        for j, r in enumerate(r_sorted):
            if used[j]:
                continue
            if o.lane == r.lane and abs(o.time - r.time) <= tol_s:
                used[j] = True
                matched += 1
                break
    return matched / len(o_sorted)


class TestRoundtripLossless:
    def test_basic_taps_and_holds(self) -> None:
        """3 bar（含空 bar）的 TAP+HOLD 往返必须 measure == 1.0。"""
        bar_dur = 2.0  # 120bpm 4/4
        notes = []
        for b in (0, 2):  # 跳过 bar1 测空小节
            bs = b * bar_dur
            notes.append(Note(time=bs + 0.0, lane=0, type=NoteType.TAP))
            notes.append(Note(time=bs + 0.5, lane=1, type=NoteType.TAP))
            notes.append(Note(time=bs + 1.0, lane=2, type=NoteType.HOLD, duration=0.3))
        chart = _make_chart(notes, n_bars=3)

        events = encode_atomic(chart)
        assert len(events) > 0
        recon = decode_atomic(events, chart.bpm_points)
        assert _measure(chart, recon) == 1.0, "POS+NUDGE 设计无损，应 == 1.0"

    def test_empty_chart(self) -> None:
        chart = _make_chart([], n_bars=2)
        assert encode_atomic(chart) == []
        recon = decode_atomic([], chart.bpm_points)
        assert recon.notes == []

    def test_all_note_types(self) -> None:
        """5 种 NoteType 全覆盖往返。"""
        notes = [
            Note(time=0.0, lane=0, type=NoteType.TAP),
            Note(time=0.5, lane=1, type=NoteType.HOLD, duration=0.4),
            Note(time=1.0, lane=2, type=NoteType.MINE),
            Note(time=1.5, lane=3, type=NoteType.ROLL, duration=0.6),
            Note(time=0.25, lane=0, type=NoteType.FAKE),
        ]
        chart = _make_chart(notes, n_bars=1)
        recon = decode_atomic(encode_atomic(chart), chart.bpm_points)
        assert _measure(chart, recon) == 1.0


# ── 确定性 ────────────────────────────────────────────────────


class TestDeterminism:
    def test_same_input_same_output(self) -> None:
        notes = [Note(time=0.0, lane=0), Note(time=0.5, lane=1), Note(time=1.5, lane=2)]
        chart = _make_chart(notes, n_bars=2)
        e1 = encode_atomic(chart)
        e2 = encode_atomic(chart)
        assert e1 == e2
        r1 = decode_atomic(e1, chart.bpm_points)
        r2 = decode_atomic(e2, chart.bpm_points)
        assert r1.notes == r2.notes


# ── off-beat NUDGE 发射 ───────────────────────────────────────


class TestNudgeOffbeat:
    def test_offbeat_notes_emit_nudge_and_recover(self) -> None:
        """故意把 note 偏 POS 格 5ms，断言 NUDGE 发射且解码重建误差小。

        120bpm pos_step = (60/120)/48 ≈ 10.42ms；5ms 偏移 > NUDGE_TOL_S(0.25ms)
        → 应发 NUDGE。解码重建误差 ≤ pos_step/(2*NUDGE_BUCKETS) ≈ 0.43ms。
        """
        notes = [
            Note(time=0.005, lane=0, type=NoteType.TAP),  # +5ms off
            Note(time=0.505, lane=1, type=NoteType.TAP),  # +5ms off
            Note(time=1.005, lane=2, type=NoteType.TAP),  # +5ms off
        ]
        chart = _make_chart(notes, n_bars=1)
        events = encode_atomic(chart)
        # 应有 NUDGE event
        assert any(e.startswith("NUDGE_") for e in events), "off-beat note 应发 NUDGE"

        recon = decode_atomic(events, chart.bpm_points)
        # 每个 note 重建误差 < 1ms（远小于 5.6ms 上界）
        for o, r in zip(chart.sorted_notes(), recon.sorted_notes(), strict=True):
            assert o.lane == r.lane
            assert abs(o.time - r.time) < 0.001, f"重建误差 {abs(o.time - r.time)} > 1ms"

    def test_on_grid_no_nudge(self) -> None:
        """落 POS 格的 note 不发 NUDGE。"""
        notes = [Note(time=0.0, lane=0), Note(time=0.5, lane=1)]  # 0/0.5 落 4/4 1/4 拍点
        chart = _make_chart(notes, n_bars=1)
        events = encode_atomic(chart)
        assert not any(e.startswith("NUDGE_") for e in events)


# ── 变速曲 ────────────────────────────────────────────────────


class TestVariableBPM:
    def test_two_bpm_segments_roundtrip(self) -> None:
        """变速曲（120bpm→180bpm）往返成立。"""
        # bar0-1: 120bpm bar=2s; bar2: 180bpm bar=4*60/180≈1.333s
        notes = [
            Note(time=0.0, lane=0, type=NoteType.TAP),
            Note(time=2.0, lane=1, type=NoteType.TAP),  # bar1 起
            Note(time=4.0, lane=2, type=NoteType.TAP),  # bar2 起（180bpm 段）
            Note(time=4.667, lane=3, type=NoteType.TAP),  # bar2 内 1/3
        ]
        chart = Chart(
            mode=GameMode.MANIA_4K,
            difficulty=7,
            bpm_points=[
                BpmPoint(time=0.0, bpm=120.0),
                BpmPoint(time=4.0, bpm=180.0),
            ],
            notes=notes,
            meta={"audio_duration": 6.0},
        )
        events = encode_atomic(chart)
        recon = decode_atomic(events, chart.bpm_points)
        assert _measure(chart, recon) == 1.0, "变速曲 POS→秒分段映射应保证往返"


# ── 相位对齐（RFC-0026 复现）──────────────────────────────────


class TestPhaseAlignment:
    def test_phase_positive_bar_starts_at_phase(self) -> None:
        """bpm_points[0].time=0.339（RFC-0026 实证相位），首音乐 bar 起 0.339。"""
        bps = [BpmPoint(time=0.339, bpm=120.0)]
        b = compute_bar_boundaries(bps, 6.0)
        assert b[0] == 0.0
        assert abs(b[1] - 0.339) < 1e-6  # intro 段 [0, 0.339]
        assert abs(b[2] - 2.339) < 1e-6  # 首音乐 bar 起点 = phase + bar
        assert abs(b[3] - 4.339) < 1e-6

    def test_phase_positive_roundtrip(self) -> None:
        """phase>0 谱面往返仍无损。"""
        phase = 0.339
        bar_dur = 2.0  # 120bpm
        notes = [
            Note(time=phase + 0.0, lane=0, type=NoteType.TAP),
            Note(time=phase + 0.5, lane=1, type=NoteType.TAP),
            Note(time=phase + bar_dur + 0.25, lane=2, type=NoteType.TAP),
        ]
        chart = Chart(
            mode=GameMode.MANIA_4K,
            difficulty=9,
            bpm_points=[BpmPoint(time=phase, bpm=120.0)],
            notes=notes,
            meta={"audio_duration": 6.0},
        )
        events = encode_atomic(chart)
        recon = decode_atomic(events, chart.bpm_points)
        assert _measure(chart, recon) == 1.0, "phase>0 谱面 POS+NUDGE 应无损往返"


# ── 数学单元 ──────────────────────────────────────────────────


class TestMath:
    def test_pos_index_range(self) -> None:
        """POS_i ∈ [0, pos_count)，rel∈[0,1) 全周期覆盖。"""
        pos_count = 4 * POS_DIVISIONS_PER_BEAT
        # rel=0 → POS=0; rel≈1/4（1拍）→ POS=pos_count/4
        i0, _ = compute_pos_index(0.0, 0.0, 2.0)
        assert i0 == 0
        i25, _ = compute_pos_index(0.5, 0.0, 2.0)  # 0.5/2.0=0.25 → 1/4 拍
        assert i25 == pos_count // 4
        # rel 略 < 1 → wrap 回近 0
        i_wrap, _ = compute_pos_index(1.999, 0.0, 2.0)
        assert 0 <= i_wrap < pos_count

    def test_nudge_bucket_none_when_on_grid(self) -> None:
        assert nudge_bucket(0.0, 0.01) is None
        assert nudge_bucket(NUDGE_TOL_S, 0.01) is None
        assert nudge_bucket(NUDGE_TOL_S * 0.5, 0.01) is None

    def test_nudge_bucket_range(self) -> None:
        """NUDGE bucket ∈ [0, NUDGE_BUCKETS)。"""
        pos_step = 0.01  # 10ms
        # 残差接近 +pos_step/2 → 端点桶
        hi = nudge_bucket(pos_step * 0.49, pos_step)
        lo = nudge_bucket(-pos_step * 0.49, pos_step)
        assert hi is not None
        assert lo is not None
        assert 0 <= hi < NUDGE_BUCKETS
        assert 0 <= lo < NUDGE_BUCKETS
        assert hi > lo  # 正残差 bucket 更大

    def test_dur_bucket_monotonic(self) -> None:
        """duration 越大 dur_bucket 越大（单调）。"""
        assert dur_bucket(0.01) == 0
        assert dur_bucket(100.0) == N_DUR_BUCKETS - 1
        assert dur_bucket(0.1) < dur_bucket(0.5) < dur_bucket(2.0)

    def test_bucket_center_inverse(self) -> None:
        """POS+NUDGE 桶中心能近似还原原 time（误差 ≤ 一个 NUDGE 桶宽）。

        NUDGE 把残差量化到桶中心，故往返误差 ≤ ``bucket_w = pos_step /
        NUDGE_BUCKETS``（离散桶量化的确定性上界）。实证最坏 ~0.86ms（@120bpm），
        远小于 ±20ms 容差，亦优于保守上界 ``pos_step*(1/2+1/(2N))≈5.6ms``
        （后者仅在 NUGE 完全不发时才生效）。
        """
        bar_dur = 2.0
        pos_step = bar_dur / (4 * POS_DIVISIONS_PER_BEAT)
        # 一个 off-grid note
        note_time = 0.505
        pos_i, residual = compute_pos_index(note_time, 0.0, bar_dur)
        n_idx = nudge_bucket(residual, pos_step)
        assert n_idx is not None, "0.005 残差 >> NUDGE_TOL_S，应发 NUDGE"
        recon = bucket_center_time(0.0, pos_i, n_idx, bar_dur)
        bucket_w = pos_step / NUDGE_BUCKETS
        assert abs(recon - note_time) <= bucket_w + 1e-9

    def test_dur_bucket_center_positive(self) -> None:
        for k in range(N_DUR_BUCKETS):
            assert dur_bucket_center(k) > 0
