"""M6.5 校准隔离与探索性标注（plan 06 §4.6 / §4.7）。

三条断言：

1. 主判据函数**结构上**看不到 NLL / 能量相关性（签名只接受 QualityMetrics）；
2. 把校准分节投毒（NaN）不改变主判据读数；
3. λ ≡ 0 的 NLL 为 +∞（契约断言，禁止 eps 平滑）。
"""

from __future__ import annotations

import inspect
import math

import pytest
import torch

from beatmorph.eval.calibration import (
    EXPLORATORY_WARNING,
    NLL_WARNING,
    ONSET_DISCLAIMER,
    PRIMARY_CRITERION_SECTIONS,
    CalibrationReadout,
    ExploratoryReadout,
    assert_calibration_is_labeled,
    assert_exploratory_is_labeled,
    binned_density,
    check_zero_intensity_diverges,
    energy_correlation,
    poisson_nll_float,
    primary_criterion,
)
from beatmorph.eval.protocol import EvalConfig
from beatmorph.eval.report import build_report, render_text
from beatmorph.field.grid import FieldGrid
from tests.unit.eval._builders import make_bpm_points, perfect_case
from tests.unit.field._builders import make_grid

CONFIG = EvalConfig()


def test_primary_criterion_cannot_see_calibration_or_exploratory() -> None:
    """主判据的签名只接受 QualityMetrics —— 隔离是结构性的，不是约定。"""
    signature = inspect.signature(primary_criterion)
    assert set(signature.parameters) == {"quality", "tolerance_s"}
    annotations = " ".join(str(parameter.annotation) for parameter in signature.parameters.values())
    assert "Calibration" not in annotations
    assert "Exploratory" not in annotations
    assert "nll" not in annotations.lower()
    assert PRIMARY_CRITERION_SECTIONS == ("quality",)


def test_poisoned_calibration_does_not_move_the_primary_score() -> None:
    """把 NLL / 能量相关性改成 NaN，主判据逐位不变（读了就会变成 NaN）。"""
    case = perfect_case(count=6)
    clean = build_report([case], CONFIG)
    poisoned = build_report(
        [case],
        CONFIG,
        calibration=CalibrationReadout(available=True, nll=float("nan")),
        exploratory=ExploratoryReadout(
            available=True,
            energy_correlation=float("nan"),
            energy_bin_s=CONFIG.primary_tolerance,
        ),
    )
    assert math.isnan(poisoned.calibration.nll)
    assert math.isnan(poisoned.exploratory.energy_correlation or float("nan"))
    assert poisoned.primary_score() == clean.primary_score()
    assert clean.primary_score() == 1.0
    poisoned.assert_isolation()


def test_unavailable_readouts_are_explicit_not_zero() -> None:
    """缺失必须显式：不得用 0 冒充（0 是合法 NLL 值）。"""
    calibration = CalibrationReadout.unavailable()
    assert calibration.available is False
    assert calibration.nll is None
    assert calibration.nll_constant_baseline is None
    assert calibration.nll_zero_is_inf is True
    assert calibration.role == "calibration"
    assert calibration.warning == NLL_WARNING
    assert calibration.is_primary_criterion is False
    exploratory = ExploratoryReadout.unavailable()
    assert exploratory.available is False
    assert exploratory.energy_correlation is None
    assert exploratory.role == "exploratory"
    assert exploratory.warning == EXPLORATORY_WARNING
    assert exploratory.onset_disclaimer == ONSET_DISCLAIMER
    assert exploratory.is_primary_criterion is False


def test_constant_baseline_comes_from_the_field_authority() -> None:
    """G3 常数基线 λ0 = N/|Ω| 的 NLL 走 field/loss.py（eval 不复制公式）。"""
    from beatmorph.field.loss import constant_baseline_nll

    n_events, omega, nll = 40.0, 2.5, 17.5
    readout = CalibrationReadout.from_counts(nll=nll, n_events=n_events, omega=omega)
    baseline = float(constant_baseline_nll(n_events, omega))
    assert readout.available is True
    assert readout.nll == nll
    assert readout.nll_constant_baseline == pytest.approx(baseline)
    assert math.isfinite(baseline), "常数基线不是 λ ≡ 0（后者发散）"
    assert readout.nll_per_event == pytest.approx(nll / n_events)
    assert readout.nll_constant_baseline_per_event == pytest.approx(baseline / n_events)
    with pytest.raises(ValueError, match="必须为正"):
        CalibrationReadout.from_counts(nll=nll, n_events=0.0, omega=omega)


def _bound_grid(t_bins: int = 4, x_bins: int = 8) -> FieldGrid:
    """已绑定时间轴的小网格（NLL 契约断言用）。"""
    return make_grid(bpm_points=make_bpm_points(), t_bins=t_bins, x_bins=x_bins)


def test_zero_intensity_nll_diverges_contract() -> None:
    """M6.5：λ ≡ 0 且 N > 0 时 NLL = +∞（禁止 eps 平滑）。"""
    grid = _bound_grid()
    counts = torch.zeros(
        (1, grid.t_bins, grid.x_bins, grid.sides, grid.channels),
        dtype=torch.float64,
    )
    counts[0, 0, 0, 0, 0] = 1.0
    value = check_zero_intensity_diverges(counts, grid)
    assert math.isinf(value)
    assert value > 0.0
    lam = torch.full_like(counts, 0.01)
    finite = poisson_nll_float(counts, lam, grid)
    assert math.isfinite(finite)


def test_energy_correlation_is_labeled_exploratory() -> None:
    """能量相关性可以报，但必须带 exploratory 标注且不得进主判据。"""
    density = [1.0, 2.0, 3.0, 4.0]
    energy = [2.0, 4.0, 6.0, 8.0]
    readout = energy_correlation(density, energy, bin_s=0.5)
    assert readout.energy_correlation == pytest.approx(1.0)
    assert readout.available is True
    assert readout.energy_bin_s == 0.5
    assert readout.role == "exploratory"
    assert readout.warning == EXPLORATORY_WARNING
    assert readout.is_primary_criterion is False
    assert readout.onset_disclaimer == ONSET_DISCLAIMER
    plain = energy_correlation([1.0, 1.0, 1.0], [1.0, 2.0, 3.0], bin_s=0.5)
    assert plain.energy_correlation is None, "方差为 0 → 未定义（缺失 != 0）"
    with pytest.raises(ValueError, match="长度必须一致"):
        energy_correlation([1.0, 2.0], [1.0], bin_s=0.5)


def test_binned_density_uses_the_declared_bin_width() -> None:
    """密度曲线按声明的箱宽分箱（口径随读数一起给）。"""
    bin_s = 0.5
    density = binned_density((0.0, bin_s, 2.0 * bin_s), bin_s=bin_s)
    assert density == (1.0 / bin_s, 1.0 / bin_s, 1.0 / bin_s)
    with pytest.raises(ValueError, match="bin_s 必须"):
        binned_density((0.0,), bin_s=0.0)
    assert binned_density((), bin_s=bin_s) == ()


def test_label_assertions_reject_mislabelled_sections() -> None:
    """标注断言必须能挡住「把校准当主判据」这类误用。"""
    with pytest.raises(AssertionError):
        assert_calibration_is_labeled(CalibrationReadout(is_primary_criterion=True))
    with pytest.raises(AssertionError):
        assert_calibration_is_labeled(CalibrationReadout(available=True, nll=None))
    with pytest.raises(AssertionError):
        assert_exploratory_is_labeled(ExploratoryReadout(is_primary_criterion=True))
    assert_calibration_is_labeled(CalibrationReadout.unavailable())
    assert_exploratory_is_labeled(ExploratoryReadout.unavailable())


def test_report_header_carries_the_mandated_warning() -> None:
    """plan §4.6：报告模板头部必须固定一行警示。"""
    report = build_report([perfect_case()], CONFIG)
    text = render_text(report)
    assert NLL_WARNING in text
    assert EXPLORATORY_WARNING in text
    assert ONSET_DISCLAIMER in text
    assert report.calibration.role == "calibration"
    assert report.exploratory.role == "exploratory"
