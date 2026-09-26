"""强度场模块（Plan 03）：网格与测度、目标构建、两条积分路径、泊松 NLL、碰撞统计、可视化。

导入策略：grid / target / collision **无 torch 依赖**，直接导入；integrate / loss / viz
依赖 torch / matplotlib，用 PEP 562 惰性加载——因此 **import beatmorph.field 本身不需要
torch**，M12 的「秒 <-> tau」契约测试可在最小环境运行（红线 7 的换算唯一出口在 grid.py）。

本模块是「秒 <-> tau」换算的**唯一**实现处：下游一律经 FieldGrid.jacobian /
tau_to_seconds / seconds_to_tau / cell_volumes / volume，不得再写第二套 BPM 分段积分。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from beatmorph.field.collision import (
    ABLATION_METRIC_KEYS,
    CocellReport,
    ExactCollisionReport,
    NAblationRow,
    SameInstantGroup,
    bins_for_zero_collision,
    cocell_report,
    cocell_sweep,
    exact_collision_report,
    merge_ablation_metrics,
    min_same_instant_dx,
)
from beatmorph.field.grid import (
    BEAT_SUBDIVISION,
    DEFAULT_X_BINS,
    MERT_FRAME_RATE_HZ,
    N_CHANNELS,
    N_SIDES,
    X_BIN_SWEEP,
    BpmSegment,
    CellSecondsRule,
    FieldGrid,
    bpm_segments,
    jacobian_at,
    seconds_to_tau,
    tau_bin_index,
    tau_to_seconds,
)
from beatmorph.field.target import (
    CHANNEL_INDEX,
    HOLD_END_CHANNEL,
    FieldTarget,
    TargetMeta,
    build_target,
)

if TYPE_CHECKING:  # 仅为类型检查而导入（运行时经 __getattr__ 惰性加载）
    from beatmorph.field.integrate import (
        CumulativeMode,
        apply_line_mask,
        cell_volumes_tensor,
        count_true,
        cumulative_from_lam,
        delta_cumulative,
        factorized_lambda,
        field_cell_volumes,
        integrate_cumulative,
        integrate_grid,
        lam_from_cumulative,
        normalize_cell_prob,
        omega,
        relative_error,
        uniform_cell_prob,
    )
    from beatmorph.field.loss import (
        NllDecomposition,
        Reduction,
        assert_lambda_valid,
        binned_point_gap,
        binned_poisson_nll,
        constant_baseline_lambda,
        constant_baseline_nll,
        constant_lambda_field,
        nll_decomposition,
        nll_terms,
        normalized_line_entropy,
        omega_value_for,
        poisson_nll,
        softplus_lambda,
    )
    from beatmorph.field.viz import (
        build_collision_figure,
        build_field_figure,
        default_field_png_name,
        field_panel_shape,
        figure_digest,
        figure_size_inches,
        png_pixel_size,
        render_collision_report_png,
        render_field_png,
    )

#: 惰性加载的名字 -> 所在子模块（torch / matplotlib 只在这些名字被取用时才导入）
_LAZY: dict[str, str] = {
    name: module
    for module, names in (
        (
            "beatmorph.field.integrate",
            (
                "CumulativeMode",
                "apply_line_mask",
                "cell_volumes_tensor",
                "count_true",
                "cumulative_from_lam",
                "delta_cumulative",
                "factorized_lambda",
                "field_cell_volumes",
                "integrate_cumulative",
                "integrate_grid",
                "lam_from_cumulative",
                "normalize_cell_prob",
                "omega",
                "relative_error",
                "uniform_cell_prob",
            ),
        ),
        (
            "beatmorph.field.loss",
            (
                "NllDecomposition",
                "Reduction",
                "assert_lambda_valid",
                "binned_point_gap",
                "binned_poisson_nll",
                "constant_baseline_lambda",
                "constant_baseline_nll",
                "constant_lambda_field",
                "nll_decomposition",
                "nll_terms",
                "normalized_line_entropy",
                "omega_value_for",
                "poisson_nll",
                "softplus_lambda",
            ),
        ),
        (
            "beatmorph.field.viz",
            (
                "build_collision_figure",
                "build_field_figure",
                "default_field_png_name",
                "field_panel_shape",
                "figure_digest",
                "figure_size_inches",
                "png_pixel_size",
                "render_collision_report_png",
                "render_field_png",
            ),
        ),
    )
    for name in names
}

__all__ = [
    "ABLATION_METRIC_KEYS",
    "BEAT_SUBDIVISION",
    "CHANNEL_INDEX",
    "DEFAULT_X_BINS",
    "HOLD_END_CHANNEL",
    "MERT_FRAME_RATE_HZ",
    "N_CHANNELS",
    "N_SIDES",
    "X_BIN_SWEEP",
    "BpmSegment",
    "CellSecondsRule",
    "CocellReport",
    "ExactCollisionReport",
    "FieldGrid",
    "FieldTarget",
    "NAblationRow",
    "NllDecomposition",
    "Reduction",
    "SameInstantGroup",
    "TargetMeta",
    "apply_line_mask",
    "assert_lambda_valid",
    "binned_point_gap",
    "binned_poisson_nll",
    "bins_for_zero_collision",
    "bpm_segments",
    "build_collision_figure",
    "build_field_figure",
    "build_target",
    "cell_volumes_tensor",
    "cocell_report",
    "cocell_sweep",
    "constant_baseline_lambda",
    "constant_baseline_nll",
    "constant_lambda_field",
    "count_true",
    "cumulative_from_lam",
    "default_field_png_name",
    "delta_cumulative",
    "exact_collision_report",
    "factorized_lambda",
    "field_cell_volumes",
    "field_panel_shape",
    "figure_digest",
    "figure_size_inches",
    "integrate_cumulative",
    "integrate_grid",
    "jacobian_at",
    "lam_from_cumulative",
    "merge_ablation_metrics",
    "min_same_instant_dx",
    "nll_decomposition",
    "nll_terms",
    "normalize_cell_prob",
    "normalized_line_entropy",
    "omega",
    "omega_value_for",
    "png_pixel_size",
    "poisson_nll",
    "relative_error",
    "render_collision_report_png",
    "render_field_png",
    "seconds_to_tau",
    "softplus_lambda",
    "tau_bin_index",
    "tau_to_seconds",
    "uniform_cell_prob",
]


def __getattr__(name: str) -> object:
    """PEP 562 惰性加载（grid / target / collision 之外的公开名字）。"""
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted(__all__)
