"""val 指标口径（plan 07 §9-47 B/C/D 的落点；默认 CI，无权重 / 无 GPU）。

**为什么这个文件是 val 路径的地基**：val 的全部价值取决于「模型读数与基线在**同一测度**上」。
`train_loop.py:357-360` 明文禁止把遮盖路径的 loss 与全事件闭式基线配对（那是 G2 那个错的
同族）。本文件用**暴力扫描**把 `val/nll_masked_constant` 钉死成「常数场在同一遮盖测度下的
最优值」——扫描用的是**真的** `masked_poisson_loss`，因此 `val_metrics` 里那份镜像分支表
一旦漂移，这里当场失败。

覆盖：
1. `masked_readout` 的归约 == `masked_poisson_loss(reduction="mean")`（Σ损失 / Σ有效线）；
2. 常数场最优值 = 扫描极小值（hidden / observed / hidden_doc / none 四档）；
3. r == 0 的批退化为全事件口径（losses.py 的契约分支）；
4. 空窗的损失**恒等于积分项**（§9-46 的实测事实），分层据此分得开；
5. 跨批聚合**按有效线数加权**（不是逐批平均）；
6. 未定义的读数**不出现**（缺失 != 0）；
7. 全事件口径走离线评估的同一条入口（`calibration.poisson_nll_float`）。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from beatmorph.core.contracts.phigros import SUBDIVISIONS_PER_BEAT
from beatmorph.eval import calibration
from beatmorph.eval.val_metrics import (
    ValAccumulator,
    constant_field_nll,
    constant_field_solution,
    full_event_nll,
    masked_measure,
    masked_nll,
    masked_readout,
    measure_volume,
)
from beatmorph.generation.batch import FieldOutput
from beatmorph.generation.losses import integral_term, masked_poisson_loss
from beatmorph.generation.masks import build_occlusion_batch
from tests.unit.generation._builders import make_batch, make_counts, make_grid

#: τ 格数由契约派生（不得写死拍格宽）
T_BINS = SUBDIVISIONS_PER_BEAT // 4
X_BINS = 8
K_LINES = 3
#: 归约口径的数值容差：readout 先按窗口求和、损失按 (B, K) 求和 ⇒ 只差 float32 求和顺序
REL = 1e-6


def _grid():
    return make_grid(t_bins=T_BINS, x_bins=X_BINS)


def _masked_batch(*, ratio: float = 0.5, k: int = K_LINES, events: int = 10, seed: int = 3):
    """带遮盖的合成批（走 data/generation 的真实遮盖构造，不手搓 mask）。"""
    grid = _grid()
    counts = make_counts(batch=1, k=k, grid=grid, events=events, holds=0, seed=seed)
    occlusion, _stats = build_occlusion_batch(counts, ratio=ratio, seed=11)
    batch = make_batch(k=k, grid=grid, counts=counts, occlusion=occlusion, audio_dim=16)
    return batch


def _lam(batch, *, scale: float = 1.0, seed: int = 0):
    """一个确定的强度场（softplus 保证正；不依赖任何权重）。"""
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(batch.batch_field_shape(), generator=generator) * scale
    return torch.nn.functional.softplus(raw)


def test_masked_readout_matches_the_loss_mean_reduction() -> None:
    """`val/nll` 的归约口径必须与损失**逐位一致**（否则基线比值没有意义）。"""
    batch = _masked_batch()
    lam = _lam(batch)
    readout = masked_readout(batch, lam)
    direct = float(masked_poisson_loss(FieldOutput(lam=lam), batch, reduction="mean").item())
    assert readout.nll_masked == pytest.approx(direct, rel=REL)
    assert readout.lines == float(batch.line_mask_bool().sum().item())
    assert readout.windows == batch.batch_size()
    assert readout.events == float(batch.counts.sum().item())


def test_readout_matches_the_mean_reduction_with_padding_lines() -> None:
    """padding 线不进分母：`reduction="mean"` 的分母是**有效**线数（§9-47 C）。"""
    grid = _grid()
    counts = make_counts(batch=1, k=K_LINES, grid=grid, events=8, holds=0, seed=5)
    occlusion, _ = build_occlusion_batch(counts, ratio=0.5, seed=7)
    batch = make_batch(
        k=K_LINES, grid=grid, counts=counts, occlusion=occlusion, active_lines=[0, 2], audio_dim=16
    )
    lam = _lam(batch)
    readout = masked_readout(batch, lam)
    assert readout.lines == 2.0
    assert readout.nll_masked == pytest.approx(
        float(masked_poisson_loss(FieldOutput(lam=lam), batch, reduction="mean").item()), rel=REL
    )


@pytest.mark.parametrize("reweight", ["hidden", "observed", "hidden_doc", "none"])
def test_constant_field_is_the_brute_force_optimum(reweight: str) -> None:
    """**T2 的核心**：报告值必须是常数场在同一遮盖测度下的**最优值**（暴力扫描对照）。

    扫描用真的 `masked_poisson_loss`：`val_metrics` 的镜像分支表一旦与损失漂移，
    解析极小点就会与扫描极小点错开——这正是本测试要抓的失效。
    """
    batch = _masked_batch(ratio=0.5)
    solution = constant_field_solution(batch, reweight=reweight)  # type: ignore[arg-type]
    # 扫描与报告必须同归约（这里都用 sum）：mean 只是把同一个数除以有效线数。
    reported = constant_field_nll(batch, reweight=reweight, reduction="sum")  # type: ignore[arg-type]
    assert constant_field_nll(batch, reweight=reweight) == pytest.approx(  # type: ignore[arg-type]
        reported / float(batch.n_lines()), rel=1e-6
    )
    assert solution.volume == pytest.approx(measure_volume(batch), rel=1e-12)

    grid_scan = np.geomspace(max(solution.intensity, 1e-9) * 0.05, solution.intensity * 20.0, 4001)
    values = [
        float(
            masked_poisson_loss(
                FieldOutput(lam=torch.full(batch.batch_field_shape(), float(value))),
                batch,
                reweight=reweight,  # type: ignore[arg-type]
            ).item(),
        )
        for value in grid_scan
    ]
    best = int(np.argmin(values))
    assert float(grid_scan[best]) == pytest.approx(solution.intensity, rel=5e-3)
    assert values[best] == pytest.approx(reported, rel=1e-6)
    # 报告值还必须**不差于**扫描到的任何一点（方向正确：这是极小值而不是某个采样点）
    assert reported <= min(values) * (1.0 + 1e-6)


def test_constant_field_is_derived_from_the_measure_weights() -> None:
    """c* = s * N_sup / |Omega|（由测度权重**派生**，不是抄一个闭式常数）。

    ⚠️ 顺带钉死一条**推导出来的**恒等式（本轮发现，记在 plan 07 §9-51）：
    `hidden` 口径下 `s * N_h = (1/r) * N_h = N`（因为 r = N_h / N 本身）⇒ 常数场在
    遮盖测度下的最优值与**全事件闭式基线** `N(1 + log(|Omega|/N))` 数值相同。
    也就是说：§9-46 那份调研用全事件基线算出的 0.834 与这里的 `val/ratio` **是同一个量**
    （差别只在批与归约）。这不是「配对纪律被绕过了」——两个数相等是 Horvitz-Thompson
    重标定在常数场上的推论，而本实现仍然是在**同一测度里**求出来的（T2 的扫描即证据）。
    """
    batch = _masked_batch(ratio=0.25)
    measure = masked_measure(batch)
    solution = constant_field_solution(batch)
    assert measure.ratio == pytest.approx(0.25, abs=0.05)
    assert measure.scale == pytest.approx(1.0 / measure.ratio)
    assert solution.intensity == pytest.approx(
        measure.scale * measure.hidden_events / solution.volume, rel=1e-12
    )
    # HT 恒等式：重标定把「被遮盖事件」还原成「全部事件」
    assert measure.scale * measure.hidden_events == pytest.approx(measure.total_events, rel=1e-12)
    # 而 hidden_doc（RFC 字面口径）下两者**不相等** ⇒ 这条恒等式不是恒真的
    doc = masked_measure(batch, reweight="hidden_doc")
    assert doc.scale * doc.hidden_events != pytest.approx(doc.total_events, rel=1e-6)


def test_r_zero_batch_falls_back_to_the_full_event_measure() -> None:
    """r == 0（无遮盖）时损失退化为全事件口径 ⇒ 基线也必须跟着退化（镜像分支表的一支）。"""
    batch = _masked_batch(ratio=0.0)
    measure = masked_measure(batch)
    assert measure.ratio == 0.0
    assert measure.scale == 1.0
    assert measure.supervised_events == pytest.approx(measure.total_events)
    solution = constant_field_solution(batch)
    # 最优值就是闭式常数基线 N(1 + log(|Omega|/N))（与 field.loss 的权威实现同口径）
    from beatmorph.field.loss import constant_baseline_nll

    assert constant_field_nll(batch) == pytest.approx(
        constant_baseline_nll(measure.total_events, solution.volume) / float(batch.n_lines()),
        rel=1e-5,
    )


def test_empty_window_loss_equals_the_integral_term() -> None:
    """§9-46 的实测事实：**空窗的 loss 恒等于积分项** ∫λdV（分层据此才分得开）。"""
    grid = _grid()
    counts = torch.zeros(
        (1, 2, grid.t_bins, grid.x_bins, grid.sides, grid.channels), dtype=torch.int16
    )
    batch = make_batch(
        k=2, grid=grid, counts=counts, occlusion=torch.zeros_like(counts, dtype=torch.bool)
    )
    lam = _lam(batch)
    output = FieldOutput(lam=lam)
    loss = float(masked_poisson_loss(output, batch, reduction="sum").item())
    integral = float(integral_term(output, batch).sum().item())
    assert loss == pytest.approx(integral, rel=1e-5)
    readout = masked_readout(batch, lam)
    assert readout.events == 0.0
    assert readout.empty_windows == 1
    assert readout.nll_empty == pytest.approx(integral / readout.lines, rel=1e-5)
    assert readout.pred_over_true is None  # 无事件时校准比**没有定义**（不是 0）


def test_full_event_nll_goes_through_the_offline_entry_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A10：全事件口径必须与离线 `beatmorph-eval` **共用**同一份实现（不复制一份）。"""
    batch = _masked_batch()
    lam = _lam(batch)
    calls: list[int] = []
    original = calibration.poisson_nll_float

    def _spy(*args: object, **kwargs: object) -> float:
        calls.append(1)
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(calibration, "poisson_nll_float", _spy)
    value = full_event_nll(batch, lam)
    assert calls, "full_event_nll 没有走 calibration.poisson_nll_float"
    expected = original(
        batch.counts.to(dtype=torch.float32)[0],
        lam[0],
        batch.grid,
        line_mask=[bool(v) for v in batch.line_mask_bool()[0].tolist()],
    )
    assert value == pytest.approx(expected / batch.n_lines(), rel=1e-6)


def test_accumulator_weights_batches_by_active_lines() -> None:
    """跨批聚合是「Σ损失 / Σ有效线」，**不是**逐批平均（K 不同的批不等权）。"""
    small = _masked_batch(k=2, events=6, seed=2)
    large = _masked_batch(k=5, events=14, seed=4)
    accumulator = ValAccumulator()
    readouts = []
    for batch in (small, large):
        readout = masked_readout(batch, _lam(batch, seed=1))
        readouts.append(readout)
        accumulator.add(readout)
    assert accumulator.windows == 2
    expected = sum(r.nll_masked * r.lines for r in readouts) / sum(r.lines for r in readouts)
    assert accumulator.nll == pytest.approx(expected, rel=1e-9)
    naive = sum(r.nll_masked for r in readouts) / 2.0
    assert accumulator.nll != pytest.approx(naive, rel=1e-9)
    assert accumulator.ratio == pytest.approx(
        accumulator.nll / accumulator.nll_constant,
        rel=1e-9,  # type: ignore[operator]
    )


def test_scalars_omit_undefined_readings() -> None:
    """缺失 != 0：没有那一个总体时键**不出现**（填 0 会被读成「损失为 0」）。"""
    grid = _grid()
    counts = make_counts(batch=1, k=2, grid=grid, events=6, holds=0, seed=9)
    occlusion, _ = build_occlusion_batch(counts, ratio=0.5, seed=3)
    nonempty = make_batch(k=2, grid=grid, counts=counts, occlusion=occlusion, audio_dim=16)
    accumulator = ValAccumulator()
    accumulator.add(masked_readout(nonempty, _lam(nonempty)))
    scalars = accumulator.scalars()
    assert "val_nll_nonempty" in scalars
    assert "val_nll_empty" not in scalars
    assert scalars["val_empty_share"] == 0.0
    assert "val_nll_shuffled" not in scalars
    assert "cond_audio_zero_delta" not in scalars


def test_contrast_delta_is_measured_against_the_aggregated_model_nll() -> None:
    """条件干预三元组的 Δ = NLL(干预) - NLL(基线)，两臂**同一批**且按线数加权。"""
    batch = _masked_batch()
    lam = _lam(batch)
    readout = masked_readout(batch, lam)
    baseline = masked_nll(batch, lam)
    accumulator = ValAccumulator()
    accumulator.add(
        readout,
        contrasts={"audio_zero": baseline * 1.2, "track_zero": baseline * 0.9},
    )
    scalars = accumulator.scalars()
    assert scalars["cond_audio_zero_delta"] == pytest.approx(baseline * 0.2, rel=1e-6)
    assert scalars["cond_track_zero_delta"] == pytest.approx(-baseline * 0.1, rel=1e-6)


def test_shuffled_arm_reports_absolute_value_and_delta() -> None:
    """打乱臂（G2 口径：只在遮盖格内置换标签）要能看出「可见上下文有没有信息」。"""
    batch = _masked_batch()
    lam = _lam(batch)
    readout = masked_readout(batch, lam)
    accumulator = ValAccumulator()
    accumulator.add(readout, nll_shuffled=readout.nll_masked * 1.5)
    scalars = accumulator.scalars()
    assert scalars["val_nll_shuffled"] == pytest.approx(readout.nll_masked * 1.5, rel=1e-6)
    assert scalars["val_nll_shuffled_delta"] == pytest.approx(readout.nll_masked * 0.5, rel=1e-6)


def test_measure_volume_counts_only_active_lines() -> None:
    """|Omega| 只在**有效线**上求和（padding 线不得进测度）。"""
    grid = _grid()
    counts = make_counts(batch=1, k=4, grid=grid, events=4, holds=0, seed=1)
    full = make_batch(k=4, grid=grid, counts=counts, audio_dim=16)
    trimmed = make_batch(k=4, grid=grid, counts=counts, active_lines=[0, 1], audio_dim=16)
    assert measure_volume(trimmed) == pytest.approx(measure_volume(full) / 2.0, rel=1e-9)


def test_val_metrics_does_not_reimplement_the_measure() -> None:
    """红线 7 的同类约束：测度只能来自损失的积分项实现，不能自己乘 dV。"""
    source = (
        Path(__file__).resolve().parents[3] / "beatmorph" / "eval" / "val_metrics.py"
    ).read_text(encoding="utf-8")
    assert "field_cell_volumes" not in source
    assert "TAU_GRID_DT" not in source
    assert "MERT_FRAME_RATE_HZ" not in source
    # 测度与重标定只有一处实现 ⇒ 本模块必须 import 权威实现而不是重算
    assert "from beatmorph.generation.losses import" in source


def test_ratio_is_none_without_a_usable_baseline() -> None:
    """基线不可用时 ratio 记 None（不是 1.0、更不是 0）——聚合器不得编造相对判据。"""
    accumulator = ValAccumulator()
    assert accumulator.ratio is None
    assert accumulator.nll is None
    assert accumulator.nll_constant is None
    scalars = accumulator.scalars()
    assert "val_nll" not in scalars
    assert "val_ratio" not in scalars
    assert scalars["val_windows"] == 0.0
    readout = masked_readout(_masked_batch(), torch.zeros(_masked_batch().batch_field_shape()))
    assert readout.nll_constant > 0.0  # 但**有**一个可用的基线时它必须报出来
    assert math.isfinite(readout.nll_constant)
