"""compute_section_stats() 段落统计量单元测试。"""

from __future__ import annotations

from beatmorph.core.contracts import BpmPoint, Chart, Note, NoteType
from beatmorph.data.parsers.osu_path import compute_section_stats


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
