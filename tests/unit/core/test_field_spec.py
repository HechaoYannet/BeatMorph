"""Plan 00 场契约测试（I1 / I2 / I9 与 M7 的全部断言）。

默认 CI，无权重 / 无 GPU / 无 torch。
"""

from __future__ import annotations

import pytest

from beatmorph.core.contracts import (
    FIELD_DIMS,
    FIELD_SHAPE,
    RPE_STAGE_HALF_WIDTH,
    RPE_STAGE_WIDTH,
    RPE_X_GRID_BIN_SWEEP,
    RPE_X_GRID_BINS,
    RPE_X_GRID_DX,
    RPE_X_GRID_MAX,
    RPE_X_GRID_MIN,
    SUBDIVISIONS_PER_BEAT,
    TAU_GRID_DT,
    TYPE_CHANNELS,
    BpmPoint,
    ChartField,
    ChartFieldSpec,
    ChartTargetField,
    NoteType,
    Side,
    x_bin_index,
)


def _spec(**overrides: object) -> ChartFieldSpec:
    kwargs: dict[str, object] = {
        "k": 3,
        "t_bins": 96,
        "bpm_points": (BpmPoint(time_beats=0.0, bpm=180.0),),
    }
    kwargs.update(overrides)
    return ChartFieldSpec(**kwargs)  # type: ignore[arg-type]


# ── I1 / I2：派生式常量 ────────────────────────────────────────


def test_dx_times_bins_equals_stage_width() -> None:
    """I1：dx * x_bins == RPE_STAGE_WIDTH（对全部消融取值）。"""
    for bins in (RPE_X_GRID_BINS, *RPE_X_GRID_BIN_SWEEP):
        spec = _spec(x_bins=bins, dx=RPE_STAGE_WIDTH / bins)
        assert spec.dx * spec.x_bins == RPE_STAGE_WIDTH
        spec.assert_grid()


def test_default_dx_is_derived_not_literal() -> None:
    assert RPE_X_GRID_DX == RPE_STAGE_WIDTH / RPE_X_GRID_BINS


def test_d_tau_times_subdivisions_equals_one() -> None:
    """I1：d_tau * SUBDIVISIONS_PER_BEAT == 1（单位：拍）。"""
    assert TAU_GRID_DT * SUBDIVISIONS_PER_BEAT == 1.0
    assert _spec().d_tau * SUBDIVISIONS_PER_BEAT == 1.0


def test_x_range_is_half_width() -> None:
    """I2：x_min / x_max 等于正负半宽。"""
    assert RPE_X_GRID_MIN == -RPE_STAGE_HALF_WIDTH
    assert RPE_X_GRID_MAX == +RPE_STAGE_HALF_WIDTH
    spec = _spec()
    assert spec.x_min == -spec.x_max


# ── I9：维度语义 ───────────────────────────────────────────────


def test_channel_and_side_cardinality() -> None:
    """I9：sides == len(Side)；channels == len(NoteType) + 1。"""
    spec = _spec()
    assert spec.sides == len(Side) == 2
    assert spec.channels == len(NoteType) + 1 == 5
    assert len(TYPE_CHANNELS) == spec.channels


def test_field_shape_spec_is_single_writing() -> None:
    """形状只有一处写法：FIELD_SHAPE / FIELD_DIMS 与 spec.shape() 一致。"""
    assert FIELD_SHAPE == "(batch, k, t, x, s, c)"
    assert FIELD_DIMS == ("batch", "k", "t", "x", "s", "c")
    assert ChartField.SHAPE == FIELD_SHAPE
    assert ChartTargetField.SHAPE == FIELD_SHAPE
    assert _spec().shape() == (3, 96, RPE_X_GRID_BINS, 2, 5)


def test_k_must_be_at_least_one() -> None:
    """I9：k >= 1；K 一旦为 0 则场无定义。"""
    with pytest.raises(AssertionError):
        _spec(k=0).assert_grid()


# ── assert_grid 的失败模式（不得降级为日志）────────────────────


def test_assert_grid_rejects_bad_dx() -> None:
    with pytest.raises(AssertionError):
        _spec(dx=RPE_X_GRID_DX * 1.0000001).assert_grid()


def test_assert_grid_rejects_bad_range() -> None:
    with pytest.raises(AssertionError):
        _spec(x_min=-RPE_STAGE_HALF_WIDTH * 0.5).assert_grid()


def test_assert_grid_rejects_unsorted_bpm() -> None:
    with pytest.raises(AssertionError):
        _spec(
            bpm_points=(
                BpmPoint(time_beats=8.0, bpm=180.0),
                BpmPoint(time_beats=0.0, bpm=200.0),
            ),
        ).assert_grid()


def test_assert_grid_rejects_empty_bpm_points() -> None:
    """bpm_points 是秒与 tau 换算的唯一依据，不得为空。"""
    with pytest.raises(AssertionError):
        _spec(bpm_points=()).assert_grid()


# ── M7：x_bin_index（唯一映射路径，域外不钳位）─────────────────


def test_x_bin_index_boundaries() -> None:
    spec = _spec()
    assert x_bin_index(spec.x_min, spec) == 0
    assert x_bin_index(spec.x_max, spec) == spec.x_bins - 1
    assert x_bin_index(0.0, spec) == spec.x_bins // 2


def test_x_bin_index_outside_returns_none_without_clamping() -> None:
    spec = _spec()
    assert x_bin_index(spec.x_min - 1e-9, spec) is None
    assert x_bin_index(spec.x_max + 1e-9, spec) is None


def test_x_bin_index_is_monotone_and_contiguous() -> None:
    spec = _spec()
    seen: list[int] = []
    for index in range(spec.x_bins * 4):
        x = spec.x_min + (index + 0.5) * spec.dx / 4
        if x > spec.x_max:
            break
        bin_index = x_bin_index(x, spec)
        assert bin_index is not None
        if not seen or seen[-1] != bin_index:
            seen.append(bin_index)
    assert seen == list(range(spec.x_bins))


# ── cell_count（plan 00 口径）─────────────────────────────────


def test_cell_count_matches_rfc_ra_definition() -> None:
    """RFC-0029 §8.4 R-a：|Omega| = k * t_bins * x_bins * sides * channels。"""
    spec = _spec()
    assert spec.cell_count() == spec.k * spec.t_bins * spec.x_bins * spec.sides * spec.channels
