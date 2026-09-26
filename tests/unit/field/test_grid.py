"""M1：网格与派生常量契约 + 本模块源码零物理常量字面量（默认 CI，无权重无 GPU）。"""

from __future__ import annotations

import ast
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from beatmorph.core.contracts.field import TYPE_CHANNELS, ChartFieldSpec
from beatmorph.core.contracts.phigros import (
    RPE_HEIGHT_RATIO,
    RPE_SPEED_UNIT_RPE_Y_PER_SEC,
    RPE_STAGE_HALF_HEIGHT,
    RPE_STAGE_HALF_WIDTH,
    RPE_STAGE_HEIGHT,
    RPE_STAGE_WIDTH,
    RPE_X_GRID_BIN_SWEEP,
    RPE_X_GRID_DX,
    SUBDIVISIONS_PER_BEAT,
    TAU_GRID_DT,
    Side,
)
from beatmorph.core.contracts.tensors import (
    MERT_CONV_STRIDE_PRODUCT,
    MERT_FRAME_RATE_HZ,
    MERT_SAMPLE_RATE_HZ,
)
from beatmorph.field.grid import (
    BEAT_SUBDIVISION,
    DEFAULT_X_BINS,
    N_CHANNELS,
    N_SIDES,
    SECONDS_PER_MINUTE,
    X_BIN_SWEEP,
    FieldGrid,
    bpm_segments,
    jacobian_at,
    seconds_to_tau,
    tau_bin_index,
    tau_to_seconds,
)
from beatmorph.field.grid import (
    MERT_FRAME_RATE_HZ as FIELD_FRAME_RATE_HZ,
)
from tests.unit.field._builders import make_bpm_points

#: 测试用 BPM（任意取值，与物理常量无关）
TEST_BPM = 120.0

FIELD_DIR = Path(__file__).resolve().parents[3] / "beatmorph" / "field"

#: 禁用字面量：**从契约常量派生**，避免与 tests/unit/core/test_derived_constants.py 漂移
FORBIDDEN_LITERALS: dict[float, str] = {
    RPE_STAGE_WIDTH: "RPE_STAGE_WIDTH 必须引用契约常量",
    RPE_STAGE_HEIGHT: "RPE_STAGE_HEIGHT 必须引用契约常量",
    RPE_STAGE_HALF_WIDTH: "RPE_STAGE_HALF_WIDTH 必须派生",
    RPE_STAGE_HALF_HEIGHT: "RPE_STAGE_HALF_HEIGHT 必须派生",
    RPE_X_GRID_DX: "桶宽必须派生为 RPE_STAGE_WIDTH / x_bins",
    RPE_HEIGHT_RATIO: "RPE_HEIGHT_RATIO 是来源常量，只在契约层",
    MERT_FRAME_RATE_HZ: "帧率必须派生为采样率 / 卷积步长累乘",
    float(MERT_CONV_STRIDE_PRODUCT): "MERT_CONV_STRIDE_PRODUCT 只在契约层",
    float(MERT_SAMPLE_RATE_HZ): "MERT_SAMPLE_RATE_HZ 只在契约层",
    TAU_GRID_DT: "拍格宽必须取自 TAU_GRID_DT",
    float(SUBDIVISIONS_PER_BEAT): "拍细分必须取自 SUBDIVISIONS_PER_BEAT",
}
#: 近似匹配（速度单位 120.228… 与其旧写法 120.23）
FORBIDDEN_NEAR: tuple[tuple[float, float, str], ...] = (
    (RPE_SPEED_UNIT_RPE_Y_PER_SEC, 0.05, "速度单位必须派生"),
)


def _numeric_literals(path: Path) -> list[tuple[int, float]]:
    """AST 扫描数值字面量（注释与文档字符串不算；bool 不算）。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[int, float]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            if isinstance(node.value, bool):
                continue
            found.append((node.lineno, float(node.value)))
    return found


def _field_sources() -> list[Path]:
    return sorted(FIELD_DIR.glob("*.py"))


def test_field_sources_contain_no_physical_constants() -> None:
    """M1：beatmorph/field/ 源码**零**物理常量字面量。"""
    violations: list[str] = []
    for path in _field_sources():
        for lineno, value in _numeric_literals(path):
            reason = FORBIDDEN_LITERALS.get(value)
            if reason is not None:
                violations.append(f"{path.name}:{lineno}: 字面量 {value!r} 被禁 —— {reason}")
                continue
            for target, tol, why in FORBIDDEN_NEAR:
                if abs(value - target) <= tol:
                    violations.append(f"{path.name}:{lineno}: 字面量 {value!r} 被禁 —— {why}")
                    break
    assert not violations, "发现硬编码物理常量：\n" + "\n".join(violations)


def test_scanner_actually_scans_the_field_module() -> None:
    """扫描器自检：必须真的扫到本模块源码，否则门禁是空转。"""
    sources = _field_sources()
    assert len(sources) >= 6
    assert {path.name for path in sources} >= {
        "__init__.py",
        "collision.py",
        "grid.py",
        "integrate.py",
        "loss.py",
        "target.py",
        "viz.py",
    }


def test_scanner_detects_a_planted_literal(tmp_path: Path) -> None:
    """扫描器自检：植入一个字面量必须被抓到。"""
    planted = tmp_path / "planted.py"
    planted.write_text(f"STAGE = {RPE_STAGE_WIDTH!r}\n", encoding="utf-8")
    literals = _numeric_literals(planted)
    assert literals == [(1, RPE_STAGE_WIDTH)]


# ── 派生常量 ──────────────────────────────────────────────────


def test_dx_times_bins_equals_stage_width_for_every_sweep_value() -> None:
    """M1：dx * x_bins == RPE_STAGE_WIDTH（含全部消融档位）。"""
    for bins in (DEFAULT_X_BINS, *X_BIN_SWEEP, *RPE_X_GRID_BIN_SWEEP):
        grid = FieldGrid(x_bins=bins)
        assert grid.dx * grid.x_bins == RPE_STAGE_WIDTH
        grid.assert_grid()
    assert FieldGrid().dx == RPE_X_GRID_DX


def test_x_range_is_symmetric_half_width() -> None:
    """M1：x_min == -x_max == 正负半宽。"""
    grid = FieldGrid()
    assert grid.x_min == -RPE_STAGE_HALF_WIDTH
    assert grid.x_max == +RPE_STAGE_HALF_WIDTH
    assert grid.x_min == -grid.x_max


def test_d_tau_times_subdivisions_equals_one() -> None:
    """M1：d_tau * BEAT_SUBDIVISION == 1（单位：拍）。"""
    grid = FieldGrid()
    assert grid.d_tau == TAU_GRID_DT
    assert grid.d_tau * BEAT_SUBDIVISION == 1.0
    assert BEAT_SUBDIVISION == SUBDIVISIONS_PER_BEAT


def test_frame_rate_is_derived_not_hardcoded() -> None:
    """M1：帧率 = MERT_SAMPLE_RATE_HZ / MERT_CONV_STRIDE_PRODUCT（派生量）。"""
    assert FIELD_FRAME_RATE_HZ == MERT_SAMPLE_RATE_HZ / MERT_CONV_STRIDE_PRODUCT
    assert FIELD_FRAME_RATE_HZ == MERT_FRAME_RATE_HZ


def test_side_and_channel_cardinality() -> None:
    """M1：N_SIDES == 2 == len(Side)；N_CHANNELS == 5 == len(TYPE_CHANNELS)。"""
    assert N_SIDES == len(Side) == 2
    assert N_CHANNELS == len(TYPE_CHANNELS) == 5
    grid = FieldGrid()
    assert (grid.sides, grid.channels) == (N_SIDES, N_CHANNELS)


def test_spec_matches_contract_and_asserts() -> None:
    """spec() 产出的 ChartFieldSpec 必须通过契约侧断言并与网格一致。"""
    bpm_points = make_bpm_points((0.0, 180.0))
    grid = FieldGrid().with_time(BEAT_SUBDIVISION, bpm_points)
    spec = grid.spec(k=3)
    assert isinstance(spec, ChartFieldSpec)
    assert spec.shape() == (3, BEAT_SUBDIVISION, DEFAULT_X_BINS, N_SIDES, N_CHANNELS)
    assert spec.dx * spec.x_bins == RPE_STAGE_WIDTH
    assert spec.bpm_points == tuple(bpm_points)


def test_assert_grid_rejects_inconsistent_grids() -> None:
    """assert_grid 失败即抛（不得降级为日志）。"""
    with pytest.raises(AssertionError):
        FieldGrid(x_bins=0).assert_grid()
    with pytest.raises(AssertionError):
        FieldGrid().with_time(BEAT_SUBDIVISION, ()).assert_grid()
    unsorted = make_bpm_points((4.0, 180.0), (0.0, 120.0))
    with pytest.raises(AssertionError):
        FieldGrid().with_time(BEAT_SUBDIVISION, unsorted).assert_grid()


def test_x_centers_stay_inside_the_visible_range() -> None:
    """x 桶中心必须落在可见范围内；RangeMask 由构造即全 True（**不是钳位**）。"""
    grid = FieldGrid()
    centers = grid.x_centers()
    assert centers.shape == (DEFAULT_X_BINS,)
    assert bool(np.all(np.abs(centers) <= RPE_STAGE_HALF_WIDTH))
    assert bool(grid.range_mask().all())


#: 120 BPM 的秒/拍（派生式，不写具体数字）
SECONDS_PER_BEAT_120 = SECONDS_PER_MINUTE / TEST_BPM


def test_bpm_segments_cover_the_whole_tau_axis() -> None:
    """BPM 分段必须覆盖全轴（左闭右开；首段起点 > 0 时向前外推）。"""
    segments = bpm_segments(make_bpm_points((4.0, 120.0), (8.0, 240.0)))
    assert segments[0].tau_start == 0.0
    assert segments[0].seconds_start == 0.0
    assert segments[0].seconds_per_beat == SECONDS_PER_BEAT_120
    assert math.isinf(segments[-1].tau_end)
    assert segments[1].seconds_start == 4.0 * SECONDS_PER_BEAT_120
    assert jacobian_at(4.0, make_bpm_points((4.0, 120.0), (8.0, 240.0))) == pytest.approx(
        segments[1].seconds_per_beat,
    )


def test_tau_bin_index_is_floor_with_float_guard() -> None:
    """tau 格索引 = floor(tau / d_tau)，恰好落界时不向左格漂移。"""
    assert tau_bin_index(0.0) == 0
    assert tau_bin_index(TAU_GRID_DT * 3) == 3
    assert tau_bin_index(TAU_GRID_DT * 3 - TAU_GRID_DT / 4) == 2
    assert tau_bin_index(TAU_GRID_DT * 3 * (1.0 - 1e-15)) == 3


def test_seconds_to_tau_and_tau_to_seconds_are_inverse_at_zero() -> None:
    """零点与单调性（M12 的细则在 test_time_grid.py）。"""
    bpm_points = make_bpm_points((0.0, 150.0))
    assert tau_to_seconds(0.0, bpm_points) == 0.0
    assert seconds_to_tau(0.0, bpm_points) == 0.0
    assert tau_to_seconds(1.0, bpm_points) == pytest.approx(60.0 / 150.0)


def test_field_package_import_does_not_require_torch() -> None:
    """最小环境契约：import beatmorph.field 不得牵入 torch（M12 可在无 torch 环境跑）。"""
    root = Path(__file__).resolve().parents[3]
    code = (
        "import sys, beatmorph.field; "
        "assert 'torch' not in sys.modules, 'import beatmorph.field 牵入了 torch'; "
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
