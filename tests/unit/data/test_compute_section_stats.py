"""compute_section_stats() 段落统计量单元测试。"""

from __future__ import annotations

from beatmorph.core.contracts import BpmPoint, Chart, Note, NoteType
from beatmorph.data.parsers.osu_path import _compute_bar_boundaries, compute_section_stats


def _make_chart(
    notes: list[Note] | None = None,
    bpm_points: list[BpmPoint] | None = None,
    mode: int = 4,
    difficulty: int = 5,
) -> Chart:
    return Chart(
        difficulty=difficulty,
        bpm_points=bpm_points or [BpmPoint(time=0.0, bpm=120.0)],
        notes=notes or [],
    )


class TestComputeSectionStats:
    """段落统计量正确性。"""

    def test_empty_notes_returns_empty_sections(self) -> None:
        chart = _make_chart(notes=[])
        chart = compute_section_stats(chart)
        assert chart.sections == []

    def test_single_bar_sections_have_no_overlap(self) -> None:
        """验证相邻 Section 无重叠、全覆盖。"""
        notes = [
            Note(time=i * 0.5, lane=0, type=NoteType.TAP)
            for i in range(20)  # 10s 内的 20 个 Note
        ]
        chart = _make_chart(notes=notes)
        chart = compute_section_stats(chart, section_bars=4)

        assert len(chart.sections) > 0

        for i in range(len(chart.sections) - 1):
            assert chart.sections[i].end_time <= chart.sections[i + 1].start_time + 0.01

    def test_density_target_in_range(self) -> None:
        """density_target 值域 ∈[0,1]。"""
        notes = [Note(time=i * 0.25, lane=i % 4, type=NoteType.TAP) for i in range(40)]
        chart = _make_chart(notes=notes)
        chart = compute_section_stats(chart, section_bars=4)

        for section in chart.sections:
            assert 0.0 <= section.density_target <= 1.0, (
                f"density_target={section.density_target} out of range"
            )

    def test_energy_level_in_range(self) -> None:
        """energy_level 值域 ∈[0,1]。"""
        notes = [
            Note(time=i * 0.5, lane=0, type=NoteType.TAP if i % 3 else NoteType.HOLD, duration=1.0)
            for i in range(30)
        ]
        chart = _make_chart(notes=notes)
        chart = compute_section_stats(chart, section_bars=4)

        for section in chart.sections:
            assert 0.0 <= section.energy_level <= 1.0, (
                f"energy_level={section.energy_level} out of range"
            )

    def test_rest_probability_in_range(self) -> None:
        """rest_probability 值域 ∈[0,1]。"""
        notes = [Note(time=i * 1.0, lane=0) for i in range(10)]
        chart = _make_chart(notes=notes)
        chart = compute_section_stats(chart, section_bars=4)

        for section in chart.sections:
            assert 0.0 <= section.rest_probability <= 1.0

    def test_no_nan_values(self) -> None:
        """所有统计量不含 NaN。"""
        notes = [Note(time=i * 0.25, lane=i % 4, type=NoteType.TAP) for i in range(40)]
        chart = _make_chart(notes=notes)
        chart = compute_section_stats(chart, section_bars=4)

        import math

        for section in chart.sections:
            assert not math.isnan(section.density_target)
            assert not math.isnan(section.energy_level)
            assert not math.isnan(section.rest_probability)

    def test_section_bars_size(self) -> None:
        """section_bars 参数改变 Section 数量。"""
        notes = [
            Note(time=i * 0.25, lane=i % 4)
            for i in range(80)  # 足够多 Note 覆盖 ~20s
        ]
        chart = _make_chart(notes=notes)
        chart4 = compute_section_stats(chart, section_bars=4)
        chart8 = compute_section_stats(chart, section_bars=8)

        # section_bars=8 应产生更少 Section（每 Section 更长）
        assert len(chart8.sections) <= len(chart4.sections)

    def test_tempo_change_sections(self) -> None:
        """变速曲：BPM 分段应产生正确的小节边界。"""
        notes = [
            Note(time=i * 0.3, lane=i % 4)
            for i in range(60)  # 覆盖 ~18s
        ]
        bpm_points = [
            BpmPoint(time=0.0, bpm=120.0),
            BpmPoint(time=10.0, bpm=240.0),
        ]
        chart = _make_chart(notes=notes, bpm_points=bpm_points)
        chart = compute_section_stats(chart, section_bars=4)

        assert len(chart.sections) > 0
        # 所有 Section 应无 NaN
        import math

        for s in chart.sections:
            assert not math.isnan(s.density_target)

    def test_sections_type_present(self) -> None:
        """每个 Section 的 sections_type 不为空。"""
        notes = [Note(time=i * 0.5, lane=i % 4) for i in range(50)]
        chart = _make_chart(notes=notes)
        chart = compute_section_stats(chart, section_bars=4)

        for section in chart.sections:
            assert section.sections_type in ("intro", "verse", "chorus", "bridge", "outro")


class TestBarBoundaryPhaseAlignment:
    """RFC-0026：小节边界相位对齐（bpm_points[0].time 作节拍网格原点）。"""

    def test_phase_zero_unchanged(self) -> None:
        """phase=0（bpm_points[0].time=0）时边界与旧逻辑一致（零破坏）。"""
        # 120bpm 4/4: bar=2s; dur=6s → [0, 2, 4, 6]
        b = _compute_bar_boundaries([BpmPoint(time=0.0, bpm=120.0)], 6.0)
        assert b[0] == 0.0
        assert abs(b[1] - 2.0) < 1e-6
        assert abs(b[-1] - 6.0) < 1e-6

    def test_phase_positive_bar_starts_at_phase(self) -> None:
        """phase>0：首段 [0, phase] 保留为 intro，首个音乐小节从 phase 起算。

        真实 osu! 谱面音乐常从 timing point 时刻起拍（如 0.339s），bar 网格应
        = phase + k*bar_dur，而非 0 + k*bar_dur（否则 Note 系统性错位，RFC-0026）。
        """
        # 120bpm bar=2s, phase=0.5 → [0, 0.5, 2.5, 4.5, ...]
        b = _compute_bar_boundaries([BpmPoint(time=0.5, bpm=120.0)], 6.0)
        assert b[0] == 0.0
        assert abs(b[1] - 0.5) < 1e-6  # intro 段 [0, 0.5]
        assert abs(b[2] - 2.5) < 1e-6  # 首个音乐小节起点 = phase + bar = 0.5 + 2.0
        assert abs(b[3] - 4.5) < 1e-6
        assert b[-1] >= 6.0 - 1e-6

    def test_phase_alignment_lands_notes_on_beat(self) -> None:
        """phase 对齐后，落在 bar 起点的 Note（=phase + k*bar）应属首个 bin。"""
        from beatmorph.tokenizer.vqvae import rasterize_bar

        # 120bpm bar=2s phase=0.5，Note 在 phase+2.0=2.5（首音乐 bar 起拍）
        bps = [BpmPoint(time=0.5, bpm=120.0)]
        bounds = _compute_bar_boundaries(bps, 10.0)
        # 找首个音乐 bar（bounds[1]=0.5 → bounds[2]=2.5 是首音乐 bar）
        bar_start = bounds[2]
        bar_dur = bounds[3] - bounds[2]
        # Note 精确在 bar 起点（downbeat），应落 bin 0
        grid = rasterize_bar(
            [Note(time=bar_start, lane=0, type=NoteType.TAP)],
            bar_start,
            bar_dur,
            lane=4,
            time_bins=64,
        )
        assert grid[0, 0, 0] == 1.0  # lane0, bin0, TAP

    def test_phase_no_intro_when_phase_covers_all(self) -> None:
        """total_duration < phase 时退化为 [0, phase]（不无限生成边界）。"""
        b = _compute_bar_boundaries([BpmPoint(time=5.0, bpm=120.0)], 3.0)
        assert b[0] == 0.0
        assert abs(b[-1] - 3.0) < 1e-6 or b[-1] >= 3.0 - 1e-6
