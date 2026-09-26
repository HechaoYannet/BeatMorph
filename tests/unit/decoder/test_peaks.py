"""M5.2：D1 峰值解码（合成场上的解析验收 + 机制行为）——Plan 05 §4.1 / §6。

验收金标准（M5.2）：在**已知事件的窄高斯叠加强度场**上，`±20ms` 与 `±50ms` 的
timing-F1 **均 = 1.000**，`positionX` MAE <= `dx/2`（量化上界）。失败即通路坏。
"""

from __future__ import annotations

import numpy as np
import pytest

from beatmorph.core.contracts.phigros import SUBDIVISIONS_PER_BEAT, NoteType, Side
from beatmorph.core.contracts.tensors import MERT_FRAME_RATE_HZ
from beatmorph.decoder.events import event_sort_key, pair_events
from beatmorph.decoder.peaks import (
    DEFAULT_SMOOTH_SECONDS,
    PeakConfig,
    decode_peaks,
    hamming_kernel,
    smooth_cells,
)
from beatmorph.field.grid import tau_to_seconds
from beatmorph.field.target import HOLD_END_CHANNEL
from tests.unit.decoder._builders import (
    empty_field,
    make_bpm_points,
    make_grid,
    place_gaussian,
    spec_for,
    tau_of_bin,
    timing_f1,
    truth_from_bins,
)

BPM = 180.0
TOLERANCES_S = (0.02, 0.05)

#: 合成场里的已知事件：(line, tau 格, x 桶, 侧, 类型)
PLACEMENTS = (
    (0, 6, 10, Side.FRONT, NoteType.TAP),
    (0, 30, 60, Side.BACK, NoteType.DRAG),
    (0, 70, 120, Side.FRONT, NoteType.FLICK),
    (1, 100, 32, Side.FRONT, NoteType.TAP),
    (1, 200, 100, Side.BACK, NoteType.TAP),
    (1, 330, 70, Side.FRONT, NoteType.DRAG),
)


def _synthetic(**kwargs: object):
    bpm = make_bpm_points((0.0, BPM))
    grid = make_grid(bpm_points=bpm, t_bins=SUBDIVISIONS_PER_BEAT * 16)
    spec = spec_for(grid, 2)
    field = empty_field(2, spec)
    truth = []
    for line_id, tau_bin, x_bin, side, note_type in PLACEMENTS:
        place_gaussian(
            field,
            line_id=line_id,
            tau_bin=tau_bin,
            x_bin=x_bin,
            side=side,
            note_type=note_type,
            **kwargs,
        )
        truth.append(
            truth_from_bins(
                grid=grid,
                line_id=line_id,
                tau_bin=tau_bin,
                x_bin=x_bin,
                side=side,
                note_type=note_type,
            ),
        )
    return grid, spec, field, truth


def test_synthetic_field_decodes_with_perfect_timing_f1() -> None:
    """M5.2 主验收：两种容差下 timing-F1 均为 1.000。"""
    grid, spec, field, truth = _synthetic()
    events, stats = decode_peaks(field, grid, spec)
    decoded, _ = pair_events(events, grid, spec=spec)
    assert len(decoded) == len(truth)
    for tol in TOLERANCES_S:
        f1, precision, recall = timing_f1(truth, decoded, tol_s=tol)
        assert (f1, precision, recall) == (1.0, 1.0, 1.0), (tol, f1, precision, recall)
    assert stats["d1_n_events"] == float(len(truth))


def test_position_x_error_is_within_half_a_bin() -> None:
    """M5.2：`positionX` MAE <= `dx/2`（解码只能给到桶中心，这是量化上界）。"""
    grid, spec, field, truth = _synthetic()
    events, _ = decode_peaks(field, grid, spec)
    decoded, _ = pair_events(events, grid, spec=spec)
    errors = []
    for expected, actual in zip(
        sorted(truth, key=event_sort_key),
        sorted(decoded, key=event_sort_key),
        strict=True,
    ):
        assert expected.line_id == actual.line_id
        assert expected.side is actual.side
        assert expected.note_type is actual.note_type
        errors.append(abs(expected.position_x - actual.position_x))
    assert max(errors) <= spec.dx / 2.0


def test_peak_off_the_cell_center_is_recovered_within_half_a_cell() -> None:
    """峰不在格中心时，解码给格中心 —— 时间误差 <= 半格，仍落在 ±20ms 内。"""
    bpm = make_bpm_points((0.0, BPM))
    grid = make_grid(bpm_points=bpm, t_bins=SUBDIVISIONS_PER_BEAT * 4)
    spec = spec_for(grid, 1)
    field = empty_field(1, spec)
    place_gaussian(field, line_id=0, tau_bin=40, x_bin=64, offset_tau=0.4, offset_x=0.4)
    truth = truth_from_bins(grid=grid, line_id=0, tau_bin=40, x_bin=64)
    events, _ = decode_peaks(field, grid, spec)
    decoded, _ = pair_events(events, grid, spec=spec)
    assert len(decoded) == 1
    half_cell_seconds = float(grid.cell_seconds()[0]) / 2.0
    assert abs(decoded[0].t_s - truth.t_s) <= half_cell_seconds
    assert timing_f1([truth], decoded, tol_s=TOLERANCES_S[0])[0] == 1.0


def test_threshold_is_alpha_times_the_scale() -> None:
    """阈值必须是 `alpha * lambda_0` 而不是魔数：`alpha` 一变，检出数就变。"""
    grid, spec, field, truth = _synthetic()
    base, stats = decode_peaks(field, grid, spec)
    assert stats["d1_threshold"] == pytest.approx(stats["d1_scale"] * stats["d1_alpha"], rel=1e-12)
    huge, huge_stats = decode_peaks(field, grid, spec, config=PeakConfig(alpha=1e6))
    assert huge == []
    assert huge_stats["d1_n_events"] == 0.0
    scaled, scaled_stats = decode_peaks(field, grid, spec, config=PeakConfig(alpha=0.5))
    assert scaled_stats["d1_threshold"] == pytest.approx(stats["d1_threshold"] / 2.0, rel=1e-12)
    assert len(base) == len(scaled) == len(truth)


def test_nms_radius_suppresses_close_peaks_only() -> None:
    """NMS 半径按格派生：贴在一起的峰被抑制，隔开足够远则都保留。"""
    bpm = make_bpm_points((0.0, BPM))
    grid = make_grid(bpm_points=bpm, t_bins=SUBDIVISIONS_PER_BEAT * 2)
    spec = spec_for(grid, 1)
    close = empty_field(1, spec)
    place_gaussian(close, line_id=0, tau_bin=20, x_bin=64, sigma_tau=0.3, sigma_x=0.3)
    place_gaussian(close, line_id=0, tau_bin=21, x_bin=64, sigma_tau=0.3, sigma_x=0.3)
    events_close, _ = decode_peaks(close, grid, spec)
    assert len(events_close) == 1
    far = empty_field(1, spec)
    place_gaussian(far, line_id=0, tau_bin=20, x_bin=64, sigma_tau=0.3, sigma_x=0.3)
    place_gaussian(far, line_id=0, tau_bin=26, x_bin=64, sigma_tau=0.3, sigma_x=0.3)
    events_far, _ = decode_peaks(far, grid, spec)
    assert len(events_far) == 2


def test_flat_plateau_gives_a_deterministic_representative() -> None:
    """常数场（G3 基线形态）在"严格局部极大"下无峰 —— 必须给出确定性的代表点。"""
    bpm = make_bpm_points((0.0, BPM))
    grid = make_grid(bpm_points=bpm, t_bins=SUBDIVISIONS_PER_BEAT)
    spec = spec_for(grid, 1)
    plateau = np.full((1, *spec.shape()[1:]), 0.5, dtype=np.float64)
    first, _ = decode_peaks(plateau, grid, spec)
    second, _ = decode_peaks(plateau, grid, spec)
    assert first == second
    assert first, "常数为正值时必须能解出事件，否则解码器在 G3 基线上是空转的"


def test_empty_field_yields_no_events() -> None:
    bpm = make_bpm_points((0.0, BPM))
    grid = make_grid(bpm_points=bpm, t_bins=SUBDIVISIONS_PER_BEAT)
    spec = spec_for(grid, 2)
    events, stats = decode_peaks(empty_field(2, spec), grid, spec)
    assert events == []
    assert stats["d1_n_events"] == 0.0


def test_confidence_is_a_monotone_ratio_above_the_threshold() -> None:
    """置信度 = 峰高 / 阈值（被接受的峰恒 >= 1）；阈值随场自洽缩放，故它是**无量纲**量。

    推论：整体幅度缩放**不改变**置信度（阈值 `lambda_0 = int(lambda)/|Omega|` 同步缩放）——
    这是有意为之（阈值是相对于场自身的标度），因此这里用"总积分相同、峰更尖"的两个场
    来验证单调性，而不是用幅度。
    """
    grid, spec, field, _ = _synthetic(amplitude=3.0)
    events, _ = decode_peaks(field, grid, spec)
    assert events
    assert all(event.confidence >= 1.0 for event in events)

    single_spec = spec_for(grid, 1)
    broad = empty_field(1, single_spec)
    place_gaussian(broad, line_id=0, tau_bin=10, x_bin=64, sigma_tau=1.0, sigma_x=1.0)
    sharp = empty_field(1, single_spec)
    place_gaussian(
        sharp,
        line_id=0,
        tau_bin=10,
        x_bin=64,
        sigma_tau=0.5,
        sigma_x=0.5,
        amplitude=4.0,
    )
    broad_events, _ = decode_peaks(broad, grid, single_spec)
    sharp_events, _ = decode_peaks(sharp, grid, single_spec)
    assert sharp_events[0].confidence > broad_events[0].confidence

    doubled = empty_field(1, single_spec)
    place_gaussian(doubled, line_id=0, tau_bin=10, x_bin=64, amplitude=2.0)
    doubled_events, _ = decode_peaks(doubled, grid, single_spec)
    assert doubled_events[0].confidence == pytest.approx(
        broad_events[0].confidence,
        rel=1e-12,
    )


def test_hold_end_channel_pairs_into_a_hold_time() -> None:
    """hold 通道 + hold_end 通道的配对是目标构建（同一 `(k, i_x, s)` 纤维）的逆。"""
    bpm = make_bpm_points((0.0, BPM))
    grid = make_grid(bpm_points=bpm, t_bins=SUBDIVISIONS_PER_BEAT * 4)
    spec = spec_for(grid, 1)
    field = empty_field(1, spec)
    place_gaussian(field, line_id=0, tau_bin=24, x_bin=64, note_type=NoteType.HOLD)
    place_gaussian(
        field,
        line_id=0,
        tau_bin=72,
        x_bin=64,
        channel=HOLD_END_CHANNEL,
    )
    events, _ = decode_peaks(field, grid, spec)
    decoded, pairing = pair_events(events, grid, spec=spec)
    assert pairing.n_starts == 1
    assert pairing.n_ends == 1
    assert pairing.n_paired == 1
    hold = next(event for event in decoded if event.note_type is NoteType.HOLD)
    expected = float(tau_to_seconds(tau_of_bin(72), grid.bpm_points)) - float(
        tau_to_seconds(tau_of_bin(24), grid.bpm_points),
    )
    assert hold.hold_time_s > 0.0
    assert hold.hold_time_s == pytest.approx(expected, rel=1e-9)


def test_smoothing_window_is_derived_from_the_frame_rate() -> None:
    """平滑窗宽以秒表达并默认 = 1 个 MERT 帧（派生量）；折算是格数的最近整数。"""
    bpm = make_bpm_points((0.0, BPM))
    grid = make_grid(bpm_points=bpm, t_bins=SUBDIVISIONS_PER_BEAT)
    assert pytest.approx(1.0 / MERT_FRAME_RATE_HZ, rel=1e-12) == DEFAULT_SMOOTH_SECONDS
    one_frame = smooth_cells(DEFAULT_SMOOTH_SECONDS, grid)
    two_frames = smooth_cells(2.0 * DEFAULT_SMOOTH_SECONDS, grid)
    assert 1 <= one_frame <= two_frames
    assert one_frame % 2 == 1, "窗长必须为奇数（偶数窗会把峰挪半格）"
    assert two_frames % 2 == 1, "窗长必须为奇数（偶数窗会把峰挪半格）"
    assert smooth_cells(0.0, grid) == 1
    assert hamming_kernel(1).tolist() == [1.0]
    assert hamming_kernel(5).sum() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="alpha"):
        PeakConfig(alpha=0.0)
    with pytest.raises(ValueError, match="NMS"):
        PeakConfig(nms_tau_bins=0)
