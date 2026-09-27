"""M8：B1 消融臂（热图 + penalty-reduced focal）——默认 CI，无权重 / 无 GPU / 无网络。

本文件测的是**新臂**（`beatmorph/generation/heatmap_arm.py`），不重复
`test_losses.py` 已有的目标函数单测（`gaussian_heatmap_target` /
`hamming_smooth` / `penalty_reduced_focal_loss` 各自的性质在那里已钉过）。
这里钉的是**臂**：

1. 形状契约：输出与 `FieldBatch` 的 (B, K, T, X, S, C) 一致；主干与 B2 **逐参数同形**；
2. 目标热图的解析性质（新臂的构造口）：事件处取 1、重叠取 max、随 sigma 单调、边界裁剪、
   Hamming 只沿 tau 轴、batch 轴折叠正确；
3. focal 的**方向性**：完美预测的 loss 必须小于常数预测（p = 0.5 与 p = 均值两条基线）；
4. 门禁回调：单 batch 上 loss 下降（快测），以及标 `slow` 的 G1 实跑（判据取
   `beatmorph.infra.sanity` 默认值；训练预算显式声明，见文件末节）；
5. 解码路径：**复用** `beatmorph.decoder.peaks.decode_peaks`，绝对阈值被精确保留；
   两栏（固定阈值 / 每谱最优）同结构同格式；
6. 源码级护栏：本模块**不出现**任何秒 <-> tau 换算 / BPM 分段积分 / 泊松损失入口 /
   `beatmorph.infra` 依赖 / 自建峰值算法。

夹具纪律（AGENTS.md §3.3）：帧数与格数一律**派生**——tau 格数由
`SUBDIVISIONS_PER_BEAT` 派生、x 桶数由 `FieldGrid` 给出、音频帧数由
`tests/unit/generation/_builders.audio_frames_for` 按 duration x frame_rate 派生。
本文件**不写任何物理常量字面量**（75 / 1350 / 0.020833 一律不出现）。

超参纪律（plan 04 §4.5）：alpha / beta / sigma / 半径 / Hamming 窗 / 阈值在文献中
**未核实**，因此模块签名里没有默认值；测试里的取值是**测试局部预算**，刻意不取
CenterNet 的传闻值（2 / 4），以免被误读成对未核实数字的背书。
"""

from __future__ import annotations

import ast
import inspect
import tempfile
from dataclasses import MISSING, fields, replace
from pathlib import Path

import pytest
import torch

from beatmorph.core.contracts.phigros import SUBDIVISIONS_PER_BEAT, NoteType, Side, side_index
from beatmorph.decoder.fieldops import intensity_scale, to_numpy
from beatmorph.decoder.peaks import PeakConfig, decode_peaks
from beatmorph.field.grid import FieldGrid
from beatmorph.field.target import CHANNEL_INDEX
from beatmorph.generation.heatmap_arm import (
    DifficultyThresholdTable,
    HeatmapArm,
    HeatmapDecodeConfig,
    HeatmapDecodeResult,
    HeatmapObjective,
    HeatmapOutput,
    HeatmapTargetConfig,
    ThresholdColumns,
    decode_heatmap,
    decode_threshold_columns,
    focal_heatmap_loss,
    focal_step_fn,
    heatmap_target,
    search_threshold,
    threshold_as_alpha,
)
from beatmorph.generation.losses import gaussian_heatmap_target
from beatmorph.generation.model import MaskedFieldModel, ModelConfig
from tests.unit.generation._builders import make_batch, make_counts, make_grid
from tests.unit.generation.test_source_guards import _scan_source

# ── 测试局部超参（**不是文献口径**，见模块 docstring 的「超参纪律」）──────────
ALPHA = 1.5
BETA = 2.5
SIGMA_T = 1.0
SIGMA_X = 0.5
RADIUS_T = 2
RADIUS_X = 1
SMOOTH_WINDOW = 3
#: 无平滑（解析性质测试要看"事件处恰好 1.0"）
NO_SMOOTH = 1

K_LINES = 2
T_BINS = SUBDIVISIONS_PER_BEAT // 4
X_BINS = 8
AUDIO_DIM = 16
#: 测试用定数（难度是自由参数，不是物理常量；plan 06 §4.4-2 只规定比较前 round 到 0.1）
DIFFICULTY = 15.0

MODEL_CONFIG = ModelConfig(
    d_model=16,
    n_heads=2,
    n_layers=2,
    window=4,
    global_period=2,
    k_max=8,
    audio_dim=AUDIO_DIM,
)

#: 解码自由度的测试局部取值（NMS 半径 = 1 格；解码侧不再平滑——目标侧已平滑）
DECODE_CONFIG = HeatmapDecodeConfig(nms_tau_bins=1, nms_x_bins=1, smooth_seconds=0.0)

#: 模块**不得**出现的调用：泊松 NLL 的入口（focal 与它不同测度，RFC-0029 §3.4）
FORBIDDEN_POISSON_CALLS = {
    "full_poisson_loss",
    "masked_poisson_loss",
    "per_line_nll",
    "poisson_nll",
    "binned_poisson_nll",
    "event_term",
    "integral_term",
    "constant_baseline_nll",
    "constant_baseline_lambda",
}


def _target_config(*, smooth_window: int = NO_SMOOTH) -> HeatmapTargetConfig:
    return HeatmapTargetConfig(
        sigma_t=SIGMA_T,
        sigma_x=SIGMA_X,
        radius_t=RADIUS_T,
        radius_x=RADIUS_X,
        smooth_window=smooth_window,
    )


def _objective(*, reduction: str = "mean", smooth_window: int = SMOOTH_WINDOW) -> HeatmapObjective:
    return HeatmapObjective(
        target=_target_config(smooth_window=smooth_window),
        alpha=ALPHA,
        beta=BETA,
        reduction=reduction,  # type: ignore[arg-type]
    )


def _grid() -> FieldGrid:
    return make_grid(t_bins=T_BINS, x_bins=X_BINS)


def _one_event_counts(
    grid: FieldGrid,
    *,
    k: int = K_LINES,
    line: int = 0,
    tau_bin: int = 0,
    x_bin: int = 0,
    note_type: NoteType = NoteType.TAP,
    side: Side = Side.FRONT,
) -> torch.Tensor:
    """(1, K, T, X, S, C) 的桶内计数，只有一个事件（位置由调用方给）。"""
    counts = torch.zeros(
        1,
        k,
        grid.t_bins,
        grid.x_bins,
        grid.sides,
        grid.channels,
        dtype=torch.int16,
    )
    counts[0, line, tau_bin, x_bin, side_index(side), CHANNEL_INDEX[note_type]] = 1
    return counts


#: 解码测试的已知事件：(line, tau 格, x 桶, 侧, 类型)
PLACEMENTS = (
    (0, T_BINS // 4, X_BINS // 4, Side.FRONT, NoteType.TAP),
    (1, T_BINS // 2, X_BINS // 2, Side.BACK, NoteType.DRAG),
)


def _placed_counts(grid: FieldGrid) -> torch.Tensor:
    counts = torch.zeros(
        1,
        K_LINES,
        grid.t_bins,
        grid.x_bins,
        grid.sides,
        grid.channels,
        dtype=torch.int16,
    )
    for line, tau_bin, x_bin, side, note_type in PLACEMENTS:
        counts[0, line, tau_bin, x_bin, side_index(side), CHANNEL_INDEX[note_type]] = 1
    return counts


def _logit(values: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """概率 -> logits（钳到 [eps, 1 - eps]，避开 y = 1 / 0 处的无穷）。"""
    clamped = values.clamp(min=eps, max=1.0 - eps)
    return torch.log(clamped) - torch.log1p(-clamped)


def _module_path() -> Path:
    import beatmorph.generation.heatmap_arm as module

    return Path(module.__file__ or "")


def _literal_alpha_beta_keywords(tree: ast.AST) -> list[int]:
    """收集 `alpha=` / `beta=` 且取值为数值字面量的关键字实参行号。"""
    hits: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg in {"alpha", "beta"} and isinstance(keyword.value, ast.Constant):
                value = keyword.value.value
                if isinstance(value, int | float) and not isinstance(value, bool):
                    hits.append(keyword.value.lineno)
    return hits


# ══════════════════════════════════════════════════════════════
# 1. 形状 / 契约：与 FieldBatch 的 (B, K, T, X, S, C) 一致
# ══════════════════════════════════════════════════════════════


def test_heatmap_output_shapes_match_the_field_batch() -> None:
    """输出形状必须与 FieldBatch 的 K/T/X/S/C 完全一致，且 probability = sigmoid(logits)。"""
    grid = _grid()
    batch = make_batch(k=K_LINES, grid=grid, events=4, holds=1, audio_dim=AUDIO_DIM)
    torch.manual_seed(0)
    arm = HeatmapArm(MODEL_CONFIG, grid, _objective())
    output = arm.forward_heatmap(batch)
    expected = batch.batch_field_shape()
    assert isinstance(output, HeatmapOutput)
    assert tuple(output.logits.shape) == expected
    assert tuple(output.probability.shape) == expected
    assert output.loss is not None
    assert tuple(output.loss.shape) == ()
    assert torch.allclose(output.probability, torch.sigmoid(output.logits))
    output.assert_shapes(batch)
    # K 不进入任何输出层形状：形状完全由 FieldBatch 与 FieldGrid 给出
    assert expected == (
        batch.batch_size(),
        K_LINES,
        grid.t_bins,
        grid.x_bins,
        grid.sides,
        grid.channels,
    )


def test_probability_is_in_the_unit_interval() -> None:
    """sigmoid 之后必然落在 (0, 1)——热图概率正是阈值（两栏口径）的取值域。"""
    grid = _grid()
    batch = make_batch(k=K_LINES, grid=grid, events=3, holds=0, audio_dim=AUDIO_DIM)
    torch.manual_seed(1)
    arm = HeatmapArm(MODEL_CONFIG, grid, _objective())
    output = arm.forward_heatmap(batch, compute_loss=False)
    probability = output.probability.detach()
    assert float(probability.min()) > 0.0
    assert float(probability.max()) < 1.0
    assert output.loss is None


def test_trunk_is_parameter_for_parameter_the_b2_backbone() -> None:
    """B1 臂的主干必须**逐参数同形**于 B2（否则两臂的差异不可解释为"目标层差异"）。"""
    grid = _grid()
    torch.manual_seed(0)
    arm = HeatmapArm(MODEL_CONFIG, grid, _objective())
    torch.manual_seed(0)
    b2 = MaskedFieldModel(MODEL_CONFIG, grid)
    b1_trunk = {name: tuple(parameter.shape) for name, parameter in arm.trunk.named_parameters()}
    b2_backbone = {
        name: tuple(parameter.shape)
        for name, parameter in b2.named_parameters()
        if not name.startswith("head.")
    }
    assert b1_trunk == b2_backbone
    assert b1_trunk  # 非空：否则上面的相等断言是空转的
    # FieldHead 被显式摘掉：不留永不更新的「死参数」
    assert not hasattr(arm.trunk, "head")
    assert all("trunk.head" not in name for name, _ in arm.named_parameters())


def test_head_applies_no_normalisation() -> None:
    """输出头不得接 softplus / sigmoid / softmax：focal 要的是未归一化 logits。"""
    grid = _grid()
    torch.manual_seed(0)
    arm = HeatmapArm(MODEL_CONFIG, grid, _objective())
    head = arm.head
    tokens = torch.randn(
        2,
        3,
        grid.t_bins,
        MODEL_CONFIG.d_model,
        generator=torch.Generator().manual_seed(2),
    )
    features = torch.randn(
        2,
        3,
        grid.t_bins,
        2 * grid.x_bins * grid.sides * grid.channels,
        generator=torch.Generator().manual_seed(3),
    )
    logits = head(tokens, input_features=features)
    manual = head.cell_head(tokens) + head.cell_skip(features)
    assert torch.allclose(logits, manual.reshape(logits.shape))
    # 线性头必然出现负值（softplus 之后的强度场不会）
    assert float(logits.detach().min()) < 0.0


def test_forward_entry_is_closed_to_prevent_poisson_mixing() -> None:
    """nn.Module.forward 必须**关闭**：本仓该入口 = 泊松 NLL 前向（与 focal 不同测度）。

    RFC-0029 §3.4：B1 的 y 未归一化（int y != 事件数），泊松 NLL 的 lambda 是强度。
    若 model(batch) 能跑通，实验就会在 logits 上算泊松 NLL：数值有限、不报错、静默毁对照。
    """
    grid = _grid()
    batch = make_batch(k=K_LINES, grid=grid, events=2, holds=0, audio_dim=AUDIO_DIM)
    torch.manual_seed(0)
    arm = HeatmapArm(MODEL_CONFIG, grid, _objective())
    with pytest.raises(NotImplementedError, match="泊松 NLL"):
        arm(batch)
    with pytest.raises(NotImplementedError, match="forward_heatmap"):
        arm.forward(batch)


def test_focal_loss_requires_counts_and_the_right_shape() -> None:
    """推理 batch（无 counts）不得静默返回 0；形状不符必须报错。"""
    grid = _grid()
    batch = make_batch(k=K_LINES, grid=grid, events=2, holds=0, audio_dim=AUDIO_DIM)
    # 推理 batch：counts / occlusion 同时为 None（_builders 的 counts=None 表示"现造目标"）
    without = replace(batch, counts=None, occlusion=None)
    assert without.counts is None
    with pytest.raises(ValueError, match="counts"):
        focal_heatmap_loss(torch.zeros(batch.batch_field_shape()), without, _objective())
    with pytest.raises(AssertionError, match="logits"):
        focal_heatmap_loss(torch.zeros(1, 1, 1, 1, 1, 1), batch, _objective())


def test_padding_lines_contribute_no_loss_and_no_gradient() -> None:
    """line_mask=False 的线对 focal 的贡献与梯度**恰为 0**（与 B2 的主干契约一致）。"""
    grid = _grid()
    counts = make_counts(batch=1, k=K_LINES, grid=grid, events=6, holds=0, seed=4)
    padded = make_batch(
        k=K_LINES,
        grid=grid,
        counts=counts,
        active_lines=[0],
        audio_dim=AUDIO_DIM,
    )
    logits = torch.randn(
        padded.batch_field_shape(),
        generator=torch.Generator().manual_seed(7),
        requires_grad=True,
    )
    loss = focal_heatmap_loss(logits, padded, _objective(reduction="sum"))
    loss.backward()
    assert logits.grad is not None
    assert float(logits.grad[0, 1].abs().sum()) == 0.0
    assert float(logits.grad[0, 0].abs().sum()) > 0.0
    per_cell = focal_heatmap_loss(logits.detach(), padded, _objective(reduction="none"))
    assert float(per_cell[0, 1].abs().sum()) == 0.0


def test_target_and_objective_configs_validate_their_hyperparameters() -> None:
    """超参必须合法：非法取值当场抛（不得静默钳到别处——那会把配置错误洗成绿灯）。"""
    with pytest.raises(ValueError, match="sigma"):
        HeatmapTargetConfig(
            sigma_t=0.0,
            sigma_x=SIGMA_X,
            radius_t=RADIUS_T,
            radius_x=RADIUS_X,
            smooth_window=NO_SMOOTH,
        )
    with pytest.raises(ValueError, match="radius"):
        HeatmapTargetConfig(
            sigma_t=SIGMA_T,
            sigma_x=SIGMA_X,
            radius_t=-1,
            radius_x=RADIUS_X,
            smooth_window=NO_SMOOTH,
        )
    with pytest.raises(ValueError, match="smooth_window"):
        HeatmapTargetConfig(
            sigma_t=SIGMA_T,
            sigma_x=SIGMA_X,
            radius_t=RADIUS_T,
            radius_x=RADIUS_X,
            smooth_window=0,
        )
    with pytest.raises(ValueError, match="alpha"):
        HeatmapObjective(target=_target_config(), alpha=-1.0, beta=BETA, reduction="mean")
    with pytest.raises(ValueError, match="counts 必须是"):
        heatmap_target(torch.zeros(3, 4), _target_config())


def test_output_contract_rejects_inconsistent_payloads() -> None:
    """`HeatmapOutput.assert_shapes` 的每一条判据都必须真的会抛（不是装饰）。"""
    grid = _grid()
    batch = make_batch(k=K_LINES, grid=grid, events=2, holds=0, audio_dim=AUDIO_DIM)
    torch.manual_seed(0)
    arm = HeatmapArm(MODEL_CONFIG, grid, _objective())
    output = arm.forward_heatmap(batch)
    with pytest.raises(AssertionError, match="logits"):
        HeatmapOutput(
            logits=output.logits[:, :1],
            probability=output.probability,
        ).assert_shapes(batch)
    with pytest.raises(AssertionError, match="probability"):
        HeatmapOutput(
            logits=output.logits,
            probability=output.probability[:, :1],
        ).assert_shapes(batch)
    with pytest.raises(AssertionError, match="NaN"):
        HeatmapOutput(
            logits=torch.full_like(output.logits, float("nan")),
            probability=torch.full_like(output.probability, 0.5),
        ).assert_shapes(batch)
    with pytest.raises(AssertionError, match="sigmoid"):
        HeatmapOutput(
            logits=output.logits,
            probability=torch.full_like(output.probability, 0.5),
        ).assert_shapes(batch)
    with pytest.raises(AssertionError, match="state / occlusion"):
        arm.forward_heatmap_state(batch, output.logits[:, :1], batch.occlusion_bool())


def test_head_without_input_features_is_still_linear() -> None:
    """没有直连 skip 时输出头仍是同一套线性口径（诊断路径不得另起炉灶）。"""
    grid = _grid()
    torch.manual_seed(0)
    arm = HeatmapArm(MODEL_CONFIG, grid, _objective())
    tokens = torch.randn(
        1,
        1,
        grid.t_bins,
        MODEL_CONFIG.d_model,
        generator=torch.Generator().manual_seed(4),
    )
    logits = arm.head(tokens)
    assert torch.allclose(logits, arm.head.cell_head(tokens).reshape(logits.shape))


# ══════════════════════════════════════════════════════════════
# 2. 目标热图：新臂的构造口 + 解析性质
# ══════════════════════════════════════════════════════════════


def test_target_peaks_at_one_and_is_unnormalized() -> None:
    """事件处恰好取 1.0，且 sum(y) 既不等于 1 也不等于事件数（未归一化高斯）。"""
    grid = _grid()
    counts = _one_event_counts(grid, tau_bin=T_BINS // 2, x_bin=X_BINS // 2)
    target = heatmap_target(counts, _target_config())
    assert tuple(target.shape) == tuple(counts.shape)
    assert float(target[0, 0, T_BINS // 2, X_BINS // 2, 0, 0]) == pytest.approx(1.0, rel=1e-6)
    total = float(target.sum())
    assert total > 1.0
    assert total != pytest.approx(1.0, rel=1e-3)
    assert total != pytest.approx(float(counts.sum()), rel=1e-3)


def test_target_ignores_the_bucket_count_magnitude() -> None:
    """桶内计数被折叠为**存在性**（n_j = 2 与 n_j = 1 给出同一张热图）。

    这是与 B2 计数口径（n_j log lambda_j，不去重）的又一处差异，也是「不同测度」的
    具体体现（plan 03 §4.5、RFC-0029 §3.4）。
    """
    grid = _grid()
    single = _one_event_counts(grid, tau_bin=T_BINS // 3, x_bin=X_BINS // 3)
    doubled = single.clone()
    doubled[0, 0, T_BINS // 3, X_BINS // 3, 0, 0] = 2
    assert int(doubled.sum()) == 2 * int(single.sum())
    config = _target_config()
    assert torch.equal(heatmap_target(single, config), heatmap_target(doubled, config))


def test_target_grows_monotonically_with_sigma_at_a_fixed_distance() -> None:
    """固定距离处的取值随 sigma 单调增，而峰本身恒为 1（解析性质）。"""
    grid = _grid()
    counts = _one_event_counts(grid, tau_bin=T_BINS // 2, x_bin=X_BINS // 2)
    narrow = heatmap_target(counts, _target_config())
    wide = heatmap_target(
        counts,
        HeatmapTargetConfig(
            sigma_t=2.0 * SIGMA_T,
            sigma_x=2.0 * SIGMA_X,
            radius_t=RADIUS_T,
            radius_x=RADIUS_X,
            smooth_window=NO_SMOOTH,
        ),
    )
    peak = (0, 0, T_BINS // 2, X_BINS // 2, 0, 0)
    neighbour = (0, 0, T_BINS // 2 + 1, X_BINS // 2, 0, 0)
    assert float(narrow[peak]) == pytest.approx(1.0, rel=1e-6)
    assert float(wide[peak]) == pytest.approx(1.0, rel=1e-6)
    assert float(wide[neighbour]) > float(narrow[neighbour]) > 0.0


def test_overlapping_events_take_the_elementwise_max() -> None:
    """重叠取 element-wise max（不是相加）：相邻两峰不得让中点超过 1。"""
    grid = _grid()
    counts = _one_event_counts(grid, tau_bin=T_BINS // 2, x_bin=X_BINS // 2)
    config = _target_config()
    single = heatmap_target(counts, config)
    counts[0, 0, T_BINS // 2 + 1, X_BINS // 2, 0, 0] = 1
    overlapped = heatmap_target(counts, config)
    assert float(overlapped.max()) == pytest.approx(1.0, rel=1e-6)
    midpoint = (0, 0, T_BINS // 2 + 1, X_BINS // 2, 0, 0)
    assert float(overlapped[midpoint]) == pytest.approx(max(float(single[midpoint]), 1.0))
    # 若实现写成「相加」，中点会 > 1 —— 这条就是防它的
    assert float(overlapped[midpoint]) <= 1.0 + 1e-6
    # 两个峰位都仍是 1.0
    assert float(overlapped[0, 0, T_BINS // 2, X_BINS // 2, 0, 0]) == pytest.approx(1.0, rel=1e-6)
    assert float(overlapped[0, 0, T_BINS // 2 + 1, X_BINS // 2, 0, 0]) == pytest.approx(
        1.0, rel=1e-6
    )


@pytest.mark.parametrize(
    ("tau_bin", "x_bin"),
    [(0, 0), (T_BINS - 1, X_BINS - 1), (0, X_BINS - 1), (T_BINS - 1, 0)],
)
def test_target_is_clipped_at_the_grid_boundary(tau_bin: int, x_bin: int) -> None:
    """边界裁剪：撒点窗口被网格裁掉，不越界、不环绕、峰仍是 1.0。"""
    grid = _grid()
    counts = _one_event_counts(grid, tau_bin=tau_bin, x_bin=x_bin)
    target = heatmap_target(counts, _target_config())
    assert tuple(target.shape) == tuple(counts.shape)
    assert float(target[0, 0, tau_bin, x_bin, 0, 0]) == pytest.approx(1.0, rel=1e-6)
    # 不环绕：与事件相对的网格另一端必须仍是 0（无周期边界）
    opposite_tau = 0 if tau_bin == T_BINS - 1 else T_BINS - 1
    opposite_x = 0 if x_bin == X_BINS - 1 else X_BINS - 1
    assert float(target[0, 0, opposite_tau, opposite_x, 0, 0]) == 0.0
    assert bool(torch.isfinite(target).all())


def test_hamming_smoothing_only_touches_the_tau_axis() -> None:
    """Hamming 平滑沿 tau 轴、逐 (K, S, C) 切片：不改 x 轴支撑集、不改总量。"""
    grid = _grid()
    counts = _one_event_counts(grid, tau_bin=T_BINS // 2, x_bin=X_BINS // 2)
    raw = heatmap_target(counts, _target_config())
    smoothed = heatmap_target(counts, _target_config(smooth_window=SMOOTH_WINDOW))
    assert tuple(smoothed.shape) == tuple(raw.shape)
    assert float(smoothed.sum()) == pytest.approx(float(raw.sum()), rel=1e-5)
    assert float(smoothed[0, 0, T_BINS // 2, X_BINS // 2, 0, 0]) < 1.0
    # x 轴支撑集不变：平滑前为 0 的远处，平滑后仍是 0
    far_x = X_BINS // 2 + RADIUS_X + 1
    assert float(raw[0, 0, T_BINS // 2, far_x, 0, 0]) == 0.0
    assert float(smoothed[0, 0, T_BINS // 2, far_x, 0, 0]) == 0.0
    # tau 轴支撑集被摊宽：平滑前为 0 的相邻格，平滑后为正
    neighbour = T_BINS // 2 + RADIUS_T + 1
    assert float(raw[0, 0, neighbour, X_BINS // 2, 0, 0]) == 0.0
    assert float(smoothed[0, 0, neighbour, X_BINS // 2, 0, 0]) > 0.0


def test_target_folds_the_batch_axis_without_leaking_across_samples() -> None:
    """batch 轴折叠必须逐样本独立：样本 1 的目标不受样本 0 的事件影响。"""
    grid = _grid()
    counts = torch.zeros(
        2,
        K_LINES,
        grid.t_bins,
        grid.x_bins,
        grid.sides,
        grid.channels,
        dtype=torch.int16,
    )
    counts[0, 0, T_BINS // 4, X_BINS // 4, 0, 0] = 1
    counts[1, 1, 3 * T_BINS // 4, 3 * X_BINS // 4, 0, 0] = 1
    config = _target_config()
    batched = heatmap_target(counts, config)
    for sample in range(2):
        separate = heatmap_target(counts[sample : sample + 1], config)
        assert torch.equal(batched[sample : sample + 1], separate)
    assert not torch.equal(batched[0], batched[1])


def test_target_matches_the_losses_module_implementation() -> None:
    """构造口必须**逐位**等于 losses.py 的 gaussian_heatmap_target（无平滑时）。

    防的是「新臂偷偷换了一套目标构造」（把 max 改成 sum、把窗口改成全局）。
    """
    grid = _grid()
    counts = _one_event_counts(grid, tau_bin=T_BINS // 3, x_bin=X_BINS // 3)
    expected = gaussian_heatmap_target(
        counts[0],
        sigma_t=SIGMA_T,
        sigma_x=SIGMA_X,
        radius_t=RADIUS_T,
        radius_x=RADIUS_X,
    )
    assert torch.equal(heatmap_target(counts, _target_config())[0], expected)


# ══════════════════════════════════════════════════════════════
# 3. focal 的方向性：完美预测 < 常数预测
# ══════════════════════════════════════════════════════════════


@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_perfect_prediction_loses_less_than_a_constant_prediction(reduction: str) -> None:
    """方向性断言：**完美预测**（事件支撑集上 p -> 1、其余 p -> 0）必须优于常数预测。

    口径说明（重要，否则这条断言会被误读）：penalty-reduced focal 的两个分支
    （正样本 -(1-p)^a log p、负样本 -(1-y)^b p^a log(1-p)）分别在 p = 1 与 p = 0 处取极小，
    软目标 y 的角色是**权重**（按 (1-y)^b 降权）而**不是回归目标**——因此这里的「完美预测」
    = 在 y > 0 的支撑集上给出确定的正确判断，而不是回归到 p = y（后者不是 focal 的极小点，
    见本文件下一条梯度符号测试）。若实现把正负项写反 / 把 sigmoid 写丢 / 把 y 与 logits 互换，
    「确定的正确判断」就不会是 loss 更小的那一个。
    """
    grid = _grid()
    batch = make_batch(k=K_LINES, grid=grid, events=6, holds=2, audio_dim=AUDIO_DIM)
    assert batch.counts is not None
    objective = _objective(reduction=reduction)
    target = heatmap_target(batch.counts, objective.target)
    shape = batch.batch_field_shape()
    confident = 10.0
    perfect = torch.where(target > 0, torch.full(shape, confident), torch.full(shape, -confident))
    half = torch.zeros(shape)
    prior = torch.full(shape, float(target.mean()))
    loss_perfect = float(focal_heatmap_loss(perfect, batch, objective))
    loss_half = float(focal_heatmap_loss(half, batch, objective))
    loss_prior = float(focal_heatmap_loss(_logit(prior), batch, objective))
    assert loss_perfect < 1e-6
    assert loss_perfect < loss_half
    assert loss_perfect < loss_prior
    assert loss_perfect < 1e-3 * loss_half


def test_gradient_raises_the_logits_at_events_and_lowers_them_elsewhere() -> None:
    """方向性（梯度口径，与 alpha / beta 的具体取值无关）：

    在常数预测（logits = 0，p = 0.5）处，事件支撑集上的梯度必须为**负**（往上推），
    空格上的梯度必须为**正**（往下压）。这是"损失真的指向事件"的机器可验证表述。
    """
    grid = _grid()
    batch = make_batch(k=K_LINES, grid=grid, events=6, holds=2, audio_dim=AUDIO_DIM)
    assert batch.counts is not None
    objective = _objective(reduction="sum")
    target = heatmap_target(batch.counts, objective.target)
    logits = torch.zeros(batch.batch_field_shape(), requires_grad=True)
    focal_heatmap_loss(logits, batch, objective).backward()
    assert logits.grad is not None
    events = target > 0
    assert bool(events.any())
    assert not bool(events.all())
    assert float(logits.grad[events].max()) < 0.0
    assert float(logits.grad[~events].min()) > 0.0
    # 按格平均也必须是这个方向（不是被少数极端格带偏的假象）
    assert float(logits.grad[events].mean()) < 0.0
    assert float(logits.grad[~events].mean()) > 0.0


def test_focal_alpha_beta_have_no_defaults_anywhere() -> None:
    """alpha / beta 在文献中未核实：数据类字段与构造调用都**不得**有默认值。"""
    objective_fields = {field.name: field for field in fields(HeatmapObjective)}
    for name in ("alpha", "beta", "reduction", "target"):
        assert objective_fields[name].default is MISSING, name
        assert objective_fields[name].default_factory is MISSING, name
    for field in fields(HeatmapTargetConfig):
        assert field.default is MISSING, field.name
        assert field.default_factory is MISSING, field.name
    with pytest.raises(TypeError):
        HeatmapObjective(target=_target_config(), alpha=ALPHA, beta=BETA)  # type: ignore[call-arg]
    signature = inspect.signature(focal_heatmap_loss)
    assert "alpha" not in signature.parameters
    assert "beta" not in signature.parameters


def test_module_source_has_no_literal_alpha_or_beta() -> None:
    """源码级：本模块**不得**出现 alpha=<数字> / beta=<数字>（未核实数字不得进代码）。"""
    tree = ast.parse(_module_path().read_text(encoding="utf-8"))
    offenders = _literal_alpha_beta_keywords(tree)
    assert not offenders, offenders
    # 扫描器自检：植入的字面量必须被抓到（否则这条护栏是空转的）
    planted = ast.parse("f(x, alpha=2.0)\ng(y, beta=4)\n")
    assert len(_literal_alpha_beta_keywords(planted)) == 2


# ══════════════════════════════════════════════════════════════
# 4. 门禁回调（StepFn 同形；装配归 infra）
# ══════════════════════════════════════════════════════════════


def test_step_fn_has_the_sanity_step_fn_shape() -> None:
    """回调必须是 Callable[[], float]：零必填参数、返回 float（可直接喂给 sanity）。

    这里只做**结构**断言（不 import infra）；真正的 G1 实跑在文件末节的 slow 测试里。
    """
    grid = _grid()
    batch = make_batch(k=K_LINES, grid=grid, events=3, holds=0, audio_dim=AUDIO_DIM)
    torch.manual_seed(0)
    arm = HeatmapArm(MODEL_CONFIG, grid, _objective())
    step = focal_step_fn(arm, batch, learning_rate=0.02)
    assert not inspect.signature(step).parameters
    value = step()
    assert isinstance(value, float)
    assert value > 0.0


def test_step_fn_decreases_the_loss_within_a_small_budget() -> None:
    """单 batch 上 loss 必须下降（小步数即可）——通路（前向 / 反传 / 优化）坏掉就会平。"""
    grid = _grid()
    batch = make_batch(k=K_LINES, grid=grid, events=4, holds=1, audio_dim=AUDIO_DIM)
    torch.manual_seed(0)
    arm = HeatmapArm(MODEL_CONFIG, grid, _objective())
    step = focal_step_fn(arm, batch, learning_rate=0.02)
    history = [step() for _ in range(30)]
    assert history[-1] < history[0]
    assert history[-1] < 0.5 * history[0]
    assert all(value > 0.0 for value in history)
    assert all(parameter.grad is not None for parameter in arm.parameters())


def test_step_fn_rejects_invalid_budgets() -> None:
    """学习率必须为正、batch 必须有监督目标——不得静默返回 0。"""
    grid = _grid()
    batch = make_batch(k=K_LINES, grid=grid, events=2, holds=0, audio_dim=AUDIO_DIM)
    torch.manual_seed(0)
    arm = HeatmapArm(MODEL_CONFIG, grid, _objective())
    with pytest.raises(ValueError, match="learning_rate"):
        focal_step_fn(arm, batch, learning_rate=0.0)
    blank = replace(batch, counts=None, occlusion=None)
    step = focal_step_fn(arm, blank, learning_rate=0.02)
    with pytest.raises(ValueError, match="counts"):
        step()


def test_step_fn_can_clip_gradients_and_needs_trainable_parameters() -> None:
    """回调支持梯度裁剪；模型没有可训练参数时必须报错（不得静默空转）。"""
    grid = _grid()
    batch = make_batch(k=K_LINES, grid=grid, events=2, holds=0, audio_dim=AUDIO_DIM)
    torch.manual_seed(0)
    arm = HeatmapArm(MODEL_CONFIG, grid, _objective())
    step = focal_step_fn(arm, batch, learning_rate=0.02, clip_grad_norm=1.0)
    assert step() > 0.0
    for parameter in arm.parameters():
        parameter.requires_grad_(False)
    with pytest.raises(ValueError, match="可训练参数"):
        focal_step_fn(arm, batch, learning_rate=0.02)


# ══════════════════════════════════════════════════════════════
# 5. 解码路径：复用 decoder.peaks + 两栏口径
# ══════════════════════════════════════════════════════════════


def test_decode_reuses_the_decoder_peaks_and_keeps_the_absolute_threshold() -> None:
    """热图 -> 事件必须走 decode_peaks：d1_threshold 恰等于传入的绝对阈值，峰位正确。"""
    grid = _grid()
    counts = _placed_counts(grid)
    probability = heatmap_target(counts, _target_config())[0]
    threshold = 0.5
    result = decode_heatmap(probability, grid, threshold=threshold, config=DECODE_CONFIG)
    assert isinstance(result, HeatmapDecodeResult)
    assert result.regime == "fixed"
    assert result.stats["d1_threshold"] == pytest.approx(threshold, rel=1e-9)
    assert result.stats["d1_scale"] * result.stats["d1_alpha"] == pytest.approx(threshold, rel=1e-9)
    assert result.stats["d1_nms_tau_bins"] == float(DECODE_CONFIG.nms_tau_bins)
    assert result.stats["b1_threshold"] == pytest.approx(threshold, rel=1e-12)
    # 与直接调用 decoder 的结果逐位一致（同一套峰值算法、同一套平滑与 NMS）
    alpha = threshold_as_alpha(probability, grid, threshold=threshold)
    direct, _ = decode_peaks(
        probability,
        grid,
        grid.spec(K_LINES),
        config=PeakConfig(
            alpha=alpha,
            nms_tau_bins=DECODE_CONFIG.nms_tau_bins,
            nms_x_bins=DECODE_CONFIG.nms_x_bins,
            smooth_seconds=DECODE_CONFIG.smooth_seconds,
        ),
    )
    assert list(result.events) == direct
    assert len(result.events) == len(PLACEMENTS)
    found = {(event.line_id, event.x_bin, int(event.side)) for event in result.events}
    for line, _tau_bin, x_bin, side, _note_type in PLACEMENTS:
        assert (line, x_bin, int(side)) in found


def test_decoded_event_lands_on_the_event_cell_center() -> None:
    """解码只能给到格：事件落在自己的 tau 格上，误差 <= 半格；置信度 >= 1。"""
    grid = _grid()
    counts = _placed_counts(grid)
    probability = heatmap_target(counts, _target_config())[0]
    result = decode_heatmap(probability, grid, threshold=0.5, config=DECODE_CONFIG)
    ordered = sorted(result.events, key=lambda item: item.line_id)
    placements = sorted(PLACEMENTS, key=lambda item: item[0])
    assert len(ordered) == len(placements)
    for event, placement in zip(ordered, placements, strict=True):
        line, tau_bin, x_bin, side, note_type = placement
        assert event.line_id == line
        assert event.x_bin == x_bin
        assert event.side is side
        assert event.channel == CHANNEL_INDEX[note_type]
        assert abs(event.tau - (tau_bin + 0.5) * grid.d_tau) <= grid.d_tau / 2.0 + 1e-12
        assert event.confidence > 1.0


def test_decode_threshold_columns_share_one_report_format() -> None:
    """两栏（固定阈值 / 每谱最优）必须是**同一个结构、同一套键**（plan 06 §3.1/§4.2-4）。"""
    grid = _grid()
    counts = _placed_counts(grid)
    probability = heatmap_target(counts, _target_config())[0]
    table = DifficultyThresholdTable(thresholds={DIFFICULTY: 0.5})
    columns = decode_threshold_columns(
        probability,
        grid,
        fixed=table,
        difficulty=DIFFICULTY,
        candidates=(0.2, 0.4, 0.6),
        score_fn=lambda events: float(len(events)),
        config=DECODE_CONFIG,
    )
    assert isinstance(columns, ThresholdColumns)
    assert columns.fixed.regime == "fixed"
    assert columns.per_chart_best.regime == "per_chart_best"
    assert columns.fixed.threshold == pytest.approx(0.5)
    assert columns.fixed.search is None
    assert columns.per_chart_best.search is not None
    summary = columns.summary()
    for regime in ("fixed", "per_chart_best"):
        assert f"{regime}_threshold" in summary
        assert f"{regime}_n_events" in summary
    assert summary["difficulty"] == pytest.approx(DIFFICULTY)
    # 每谱最优栏必须真的取到候选里的最优（这里打分 = 事件数，越大越好）
    search = columns.per_chart_best.search
    assert search is not None
    assert search.candidates == (0.2, 0.4, 0.6)
    assert len(search.scores) == len(search.candidates)
    best_score = max(search.scores)
    assert search.score == pytest.approx(best_score)
    assert search.threshold == search.candidates[search.scores.index(best_score)]


def test_threshold_search_is_deterministic_and_prefers_the_earlier_candidate_on_ties() -> None:
    """并列时取候选顺序中更早的一个：同一输入两次搜索必须逐位一致。"""
    grid = _grid()
    probability = heatmap_target(_placed_counts(grid), _target_config())[0]
    first, record = search_threshold(
        probability,
        grid,
        candidates=(0.3, 0.5, 0.7),
        score_fn=lambda events: 1.0,
        config=DECODE_CONFIG,
    )
    second, _ = search_threshold(
        probability,
        grid,
        candidates=(0.3, 0.5, 0.7),
        score_fn=lambda events: 1.0,
        config=DECODE_CONFIG,
    )
    assert record.threshold == pytest.approx(0.3)
    assert first.events == second.events
    assert first.threshold == second.threshold


def test_threshold_search_rejects_empty_or_nonpositive_candidates() -> None:
    """候选网格由调用方给：空集 / 非正值必须报错（不得回退到魔数阈值）。"""
    grid = _grid()
    probability = heatmap_target(_placed_counts(grid), _target_config())[0]
    with pytest.raises(ValueError, match="候选阈值"):
        search_threshold(
            probability,
            grid,
            candidates=(),
            score_fn=lambda events: 0.0,
            config=DECODE_CONFIG,
        )
    with pytest.raises(ValueError, match="候选阈值"):
        search_threshold(
            probability,
            grid,
            candidates=(0.0, 0.5),
            score_fn=lambda events: 0.0,
            config=DECODE_CONFIG,
        )


def test_difficulty_table_never_falls_back_silently() -> None:
    """固定阈值表：缺档必须报错；定数的浮点噪声按 round_digits 归一。"""
    table = DifficultyThresholdTable(thresholds={15.0: 0.5, 16.0: 0.6})
    assert table.threshold_for(15.0) == pytest.approx(0.5)
    assert table.threshold_for(15.0000001) == pytest.approx(0.5)
    assert table.threshold_for(16.04) == pytest.approx(0.6)
    with pytest.raises(KeyError, match="难度"):
        table.threshold_for(17.0)
    with pytest.raises(ValueError, match="阈值必须为正"):
        DifficultyThresholdTable(thresholds={15.0: 0.0})


def test_an_all_zero_heatmap_is_reported_not_silently_decoded() -> None:
    """全零热图（概率下溢）必须报错：lambda_0 = 0 时阈值换算无定义，通常是模型塌陷。"""
    grid = _grid()
    blank = torch.zeros(K_LINES, grid.t_bins, grid.x_bins, grid.sides, grid.channels)
    with pytest.raises(ValueError, match="热图全零"):
        decode_heatmap(blank, grid, threshold=0.5, config=DECODE_CONFIG)
    with pytest.raises(ValueError, match="阈值必须为正"):
        decode_heatmap(blank + 0.5, grid, threshold=0.0, config=DECODE_CONFIG)
    with pytest.raises(ValueError, match="单样本"):
        decode_heatmap(blank.unsqueeze(0), grid, threshold=0.5, config=DECODE_CONFIG)


def test_decode_agrees_with_the_decoders_own_scale_function() -> None:
    """阈值 -> alpha 的换算必须经 decoder 自己的 intensity_scale（不新造测度）。"""
    grid = _grid()
    probability = heatmap_target(_placed_counts(grid), _target_config())[0]
    threshold = 0.4
    alpha = threshold_as_alpha(probability, grid, threshold=threshold)
    scale = float(intensity_scale(to_numpy(probability), grid, n_events=None))
    assert alpha * scale == pytest.approx(threshold, rel=1e-12)


def test_decode_config_validates_nms_and_smoothing() -> None:
    """解码自由度与阈值换算是同一条防线：非法取值当场抛。"""
    with pytest.raises(ValueError, match="NMS"):
        HeatmapDecodeConfig(nms_tau_bins=0, nms_x_bins=1, smooth_seconds=0.0)
    with pytest.raises(ValueError, match="smooth_seconds"):
        HeatmapDecodeConfig(nms_tau_bins=1, nms_x_bins=1, smooth_seconds=-0.1)
    with pytest.raises(ValueError, match="round_digits"):
        DifficultyThresholdTable(thresholds={DIFFICULTY: 0.5}, round_digits=-1)
    grid = _grid()
    probability = heatmap_target(_placed_counts(grid), _target_config())[0]
    with pytest.raises(ValueError, match="阈值必须为正"):
        threshold_as_alpha(probability, grid, threshold=0.0)


def test_threshold_columns_summary_tolerates_missing_metadata() -> None:
    """两栏 summary 必须容忍「未提供定数 / 无搜索记录」（固定阈值栏天然没有 search）。"""
    fixed = HeatmapDecodeResult(regime="fixed", threshold=0.5, events=(), stats={})
    assert fixed.summary() == {"fixed_threshold": 0.5, "fixed_n_events": 0.0}
    columns = ThresholdColumns(
        fixed=fixed,
        per_chart_best=replace(fixed, regime="per_chart_best", threshold=0.4),
    )
    summary = columns.summary()
    assert "difficulty" not in summary
    assert summary["per_chart_best_threshold"] == pytest.approx(0.4)


def test_module_defines_no_second_peak_algorithm() -> None:
    """源码级：峰值提取必须复用 decoder.peaks（不得在生成侧另写一份）。"""
    tree = ast.parse(_module_path().read_text(encoding="utf-8"))
    called = {
        node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert "decode_peaks" in called
    forbidden = {"argwhere", "pad", "conv1d", "hamming_window", "maximum_filter"}
    assert not (called & forbidden), sorted(called & forbidden)
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and any(key in node.name for key in ("peak", "nms"))
    }
    assert not defined, defined


# ══════════════════════════════════════════════════════════════
# 6. 源码级护栏（红线 7 + 范式隔离 + 分层）
# ══════════════════════════════════════════════════════════════


def test_heatmap_arm_has_no_self_built_time_conversion() -> None:
    """R-04-7：本模块不得出现任何秒 <-> tau 换算 / BPM 分段积分（复用既有扫描器）。"""
    problems = _scan_source(_module_path())
    assert not problems, "发现生成侧自建换算：\n" + "\n".join(problems)
    # 扫描器自检：确认它真的会抓（不是把空列表当成通过）
    with tempfile.TemporaryDirectory() as directory:
        planted = Path(directory) / "planted.py"
        planted.write_text("d_tau = 1 / 48\n", encoding="utf-8")
        assert _scan_source(planted)


def test_heatmap_arm_never_calls_a_poisson_objective() -> None:
    """范式隔离：本模块不得调用泊松 NLL 的任何入口（拒绝「两种目标混用」）。"""
    tree = ast.parse(_module_path().read_text(encoding="utf-8"))
    called: set[str] = set()
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Attribute):
                called.add(target.attr)
            elif isinstance(target, ast.Name):
                called.add(target.id)
        elif isinstance(node, ast.ImportFrom):
            imported.update(alias.name for alias in node.names)
    assert not (called & FORBIDDEN_POISSON_CALLS), sorted(called & FORBIDDEN_POISSON_CALLS)
    assert not (imported & FORBIDDEN_POISSON_CALLS), sorted(imported & FORBIDDEN_POISSON_CALLS)
    # 自检：植入探针必须被抓到
    probe = ast.parse("from x import masked_poisson_loss\nmasked_poisson_loss(a, b)\n")
    names = {
        node.func.id
        for node in ast.walk(probe)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert names & FORBIDDEN_POISSON_CALLS


def test_heatmap_arm_does_not_import_the_infra_layer() -> None:
    """分层：生成侧不得 import beatmorph.infra（门禁装配归 infra / CLI，plan 07 §3.1）。"""
    tree = ast.parse(_module_path().read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
    offenders = [name for name in modules if name.split(".")[0] == "beatmorph" and ".infra" in name]
    assert not offenders, offenders
    probe = ast.parse("from beatmorph.infra.sanity import overfit_single_batch\n")
    planted = [node.module or "" for node in ast.walk(probe) if isinstance(node, ast.ImportFrom)]
    assert any(".infra" in name for name in planted)


def test_heatmap_arm_is_scanned_by_the_package_gate() -> None:
    """新文件必须落在 R-04-7 扫描器的覆盖范围内（否则护栏会静默漏过它）。"""
    package = _module_path().parent
    files = sorted(package.rglob("*.py"))
    assert _module_path() in files
    assert len(files) >= 6


# ══════════════════════════════════════════════════════════════
# 7. G1 门禁实跑（slow：判据取 sanity 默认值，训练预算显式声明）
# ══════════════════════════════════════════════════════════════
#: G1 的样本数（plan 04 §6.2 M3：1-4 样本，与 B2 的 M3 门禁同量级）
G1_SAMPLES = 4
#: G1 的优化预算：**显式声明**（plan 07 §3.1 要求记录生效口径与阈值）。
#:
#: 实测（同 seed 可复现，reduction="mean"、300 步、遮盖补全路径）：
#:   lr = 0.02 -> 末步 loss 约 0.029（ceiling 0.05，余量 41%，seed 1/2/3 一致）
#:   lr = 0.05 -> 约 0.043（余量 14%）；lr = 0.1 或 400 步则**越过** 0.05
#: focal 在微型任务上的最优点很窄（与 plan 04 §9-15 记录的"微型合成任务固有性质"同型），
#: 因此这里的 lr 是**实测选定的预算**，不是普适推荐值；换模型/换数据必须重新标定。
G1_LEARNING_RATE = 0.02


@pytest.mark.slow
def test_g1_overfits_a_single_batch_on_the_real_masked_path(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """M8 门禁接入：G1 单 batch 过拟合（判据取 sanity 默认值；口径与预算显式记录）。

    用**遮盖补全**的真实训练路径（可见场 = counts * ~occlusion），即 B1 臂的正式训练形态；
    loss 口径 reduction="mean"——sum 口径随格数线性放大，与 sanity 的绝对阈值 0.05 不同量纲
    （实测同任务 sum 末步 ≈ 3e2），这正是 plan 07 §3.1 点名的"标度敏感"，故必须声明口径。
    """
    from beatmorph.generation.masks import build_occlusion_batch
    from beatmorph.infra.sanity import overfit_single_batch, summarize

    grid = _grid()
    counts = make_counts(
        batch=G1_SAMPLES,
        k=K_LINES,
        grid=grid,
        events=4 * G1_SAMPLES,
        holds=1,
        seed=5,
    )
    occlusion, stats = build_occlusion_batch(counts, ratio=0.5, seed=7)
    batch = make_batch(
        k=K_LINES,
        grid=grid,
        batch=G1_SAMPLES,
        counts=counts,
        occlusion=occlusion,
        audio_dim=AUDIO_DIM,
    )
    torch.manual_seed(1)
    arm = HeatmapArm(MODEL_CONFIG, grid, _objective(reduction="mean"))
    step = focal_step_fn(arm, batch, learning_rate=G1_LEARNING_RATE)
    result = overfit_single_batch(step, steps=300)
    with capsys.disabled():
        print(summarize([result]))
        print(f"  （B1 遮盖比例 r={stats.ratio:.3f}；reduction=mean；lr={G1_LEARNING_RATE}）")
    assert result.passed, result.detail
