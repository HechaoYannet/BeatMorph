"""M5.1：网格与契约断言（**不新增第二套网格断言**）——Plan 05 §6 / §8。

本模块的网格断言一律复用 `ChartFieldSpec.assert_grid()` / `FieldGrid.assert_grid()`；
这里额外钉住三件事：

1. `x_center` 与契约 `x_bin_index` **互为逆**（同一条映射，不得两套）；
2. τ 轴边缘强度 `lambda_k(tau)` 的积分 == 路径 (a) 的 `sum lambda dV`（**同测度**）；
3. 解码器源码里**没有**第二套 BPM 积分（AST 扫描，仿 plan 00 M4 的写法）。
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from beatmorph.core.contracts.field import TYPE_CHANNELS, x_bin_index
from beatmorph.core.contracts.phigros import (
    RPE_STAGE_WIDTH,
    SIDE_ORDER,
    SUBDIVISIONS_PER_BEAT,
    TAU_GRID_DT,
)
from beatmorph.decoder.events import x_center
from beatmorph.decoder.fieldops import (
    assert_field_shape,
    intensity_scale,
    omega_value,
    tau_rate_per_beat,
    to_numpy,
    total_intensity,
)
from tests.unit.decoder._builders import (
    empty_field,
    make_bpm_points,
    make_grid,
    place_gaussian,
    spec_for,
)


def _grid_and_spec(*, t_bins: int = SUBDIVISIONS_PER_BEAT * 4, k: int = 2):
    bpm = make_bpm_points((0.0, 120.0), (2.0, 180.0))
    grid = make_grid(bpm_points=bpm, t_bins=t_bins)
    return grid, spec_for(grid, k)


def test_grid_contract_is_reused_not_reinvented() -> None:
    """M5.1：网格自洽断言直接来自契约，解码器不新增第二套。"""
    grid, spec = _grid_and_spec()
    grid.assert_grid()
    spec.assert_grid()
    assert spec.d_tau * SUBDIVISIONS_PER_BEAT == 1.0
    assert spec.dx * spec.x_bins == RPE_STAGE_WIDTH
    assert spec.x_min == -spec.x_max
    # 场形状 (K, T, X, S, C)：第 2 轴是 τ（Q15），不是音频帧
    assert spec.shape() == (2, grid.t_bins, grid.x_bins, len(SIDE_ORDER), len(TYPE_CHANNELS))
    assert spec.sides == len(SIDE_ORDER)
    assert spec.channels == len(TYPE_CHANNELS)


def test_field_shape_is_asserted() -> None:
    """形状 / NaN / 负值三类违约都必须**抛错**（不得静默降级）。"""
    _, spec = _grid_and_spec()
    good = empty_field(2, spec)
    assert_field_shape(good, spec)
    with pytest.raises(AssertionError):
        assert_field_shape(good[:, :-1], spec)
    bad = good.copy()
    bad[0, 0, 0, 0, 0] = np.nan
    with pytest.raises(AssertionError):
        assert_field_shape(bad, spec)
    negative = good.copy()
    negative[0, 0, 0, 0, 0] = -1.0
    with pytest.raises(AssertionError):
        assert_field_shape(negative, spec)


def test_x_center_is_the_inverse_of_contract_x_bin_index() -> None:
    """x 桶中心 <-> 桶索引必须是**同一条映射**（禁止各模块自行 floor）。"""
    _, spec = _grid_and_spec()
    for x_bin in range(spec.x_bins):
        assert x_bin_index(x_center(x_bin, spec), spec) == x_bin
    with pytest.raises(IndexError):
        x_center(spec.x_bins, spec)


def test_tau_rate_integrates_to_the_same_measure_as_path_a() -> None:
    """M5.1 的核心：`int lambda_k(tau) dtau == sum_j lambda_j dV_j`（同测度）。

    漏乘 J(τ) 会让两者相差一个与 BPM 有关的因子——正是 POSTMORTEM 要挡的那类错误。
    """
    grid, spec = _grid_and_spec()
    field = empty_field(2, spec)
    place_gaussian(field, line_id=0, tau_bin=10, x_bin=30)
    place_gaussian(field, line_id=1, tau_bin=100, x_bin=90, amplitude=0.5)
    rates = tau_rate_per_beat(field, grid)
    integrated = rates.sum(axis=1) * grid.d_tau
    expected = total_intensity(field, grid)
    np.testing.assert_allclose(integrated, expected, rtol=1e-12)
    assert float(expected.sum()) > 0.0


def test_intensity_scale_equals_the_g3_constant_baseline() -> None:
    """`lambda_0 = N/|Omega|`；常数场下 `lambda_0 == c`（G3 基线的自洽性）。"""
    grid, spec = _grid_and_spec()
    constant = np.full((2, *spec.shape()[1:]), 0.25, dtype=np.float64)
    assert intensity_scale(constant, grid) == pytest.approx(0.25, rel=1e-12)
    assert intensity_scale(constant, grid, n_events=1000.0) == pytest.approx(
        1000.0 / omega_value(grid, 2),
        rel=1e-12,
    )
    field = empty_field(2, spec)
    place_gaussian(field, line_id=0, tau_bin=5, x_bin=5)
    assert intensity_scale(field, grid) == pytest.approx(
        float(total_intensity(field, grid).sum()) / omega_value(grid, 2),
        rel=1e-12,
    )


def test_to_numpy_accepts_any_array_like_without_copying_semantics() -> None:
    """解码器只依赖 numpy：列表 / float32 数组都能进入，且不修改输入。"""
    raw = [[[[[1.0]]]]]
    converted = to_numpy(raw)
    assert converted.dtype == np.float64
    assert converted.shape == (1, 1, 1, 1, 1)
    source = np.asarray([1.0, 2.0], dtype=np.float32)
    assert to_numpy(source).dtype == np.float64


def _numeric_literals(path: Path) -> list[float]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: list[float] = []
    for node in ast.walk(tree):
        is_number = isinstance(node, ast.Constant) and isinstance(node.value, int | float)
        if is_number and not isinstance(node.value, bool):
            out.append(float(node.value))
    return out


def test_decoder_has_no_second_bpm_integration() -> None:
    """红线 7：解码器**不得**实现第二套「秒 <-> 拍」换算。

    扫描 `beatmorph/decoder/**`：既不许出现 `SECONDS_PER_MINUTE`（BPM 积分的定义式），
    也不许出现拍格宽的字面量 `1/48`。换算只能经 `beatmorph.field.grid`。
    """
    decoder_dir = Path(__file__).resolve().parents[3] / "beatmorph" / "decoder"
    files = sorted(decoder_dir.rglob("*.py"))
    assert len(files) >= 6, "扫描器必须真的扫到解码器源码"
    offenders: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        if "SECONDS_PER_MINUTE" in text:
            offenders.append(f"{path.name}: 出现 SECONDS_PER_MINUTE（第二套 BPM 积分）")
        for value in _numeric_literals(path):
            if abs(value - TAU_GRID_DT) < 1e-12:
                offenders.append(f"{path.name}: 出现拍格宽字面量 {value!r}（应用 TAU_GRID_DT）")
    assert not offenders, "\n".join(offenders)
