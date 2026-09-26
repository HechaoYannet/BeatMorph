"""M9：G1-G4 门禁接入（标 slow，不进默认 CI；**必须实际跑过一次**）。

以本模块自带的**最小可微回归任务**（不依赖 plan 04）调用
beatmorph.infra.sanity 的四道门禁：

- 输入 = 一个二值条件场（哪些格子"被观测到"）；
- 模型 = 两个标量参数 (a, b) -> lambda = softplus(a * X + b)（**故意极小**，
  只检验通路与门禁，不检验容量）；
- 目标 = 由条件场生成的桶内计数（真实）或把这些计数随机置换（打乱对照）；
- 损失 = 本模块的 poisson_nll（测度含 J(tau)，多 BPM 段另有一档）。

G3 的基线值取自 M4 的闭式 constant_baseline_nll（**不是** 0）；
G4 同时校验 tau 轴速率与音频帧轴速率，两者都是**派生量**。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch

from beatmorph.core.contracts.tensors import (
    MERT_CONV_STRIDE_PRODUCT,
    MERT_SAMPLE_RATE_HZ,
)
from beatmorph.field.grid import (
    BEAT_SUBDIVISION,
    MERT_FRAME_RATE_HZ,
    SECONDS_PER_MINUTE,
    FieldGrid,
)
from beatmorph.field.integrate import omega
from beatmorph.field.loss import constant_baseline_nll, poisson_nll, softplus_lambda
from beatmorph.infra.sanity import (
    GateResult,
    constant_baseline_gate,
    frame_rate_gate,
    overfit_single_batch,
    shuffled_target_control,
    summarize,
)
from tests.unit.field._builders import make_bpm_points

pytestmark = pytest.mark.slow

TEST_BPM = 120.0
T_BINS = BEAT_SUBDIVISION // 6
X_BINS = 4
N_EVENTS = 8
#: 优化超参（与门禁判据无关；门禁判据取 sanity.py 的默认值）
LEARNING_RATE = 0.3
STEPS = 400


def _grid() -> FieldGrid:
    grid = FieldGrid(x_bins=X_BINS).with_time(T_BINS, make_bpm_points((0.0, TEST_BPM)))
    grid.assert_grid()
    return grid


def _shapes(grid: FieldGrid) -> tuple[int, ...]:
    return (1, grid.t_bins, grid.x_bins, grid.sides, grid.channels)


def _condition_and_counts(grid: FieldGrid) -> tuple[torch.Tensor, torch.Tensor]:
    """条件场（二值）与由它决定的桶内计数：模型可以精确拟合真实目标。"""
    shape = _shapes(grid)
    total_cells = int(np.prod(shape))
    generator = np.random.default_rng(20260805)
    chosen = generator.choice(total_cells, size=N_EVENTS, replace=False)
    condition = torch.zeros(shape, dtype=torch.float64).reshape(-1)
    counts = torch.zeros(shape, dtype=torch.float64).reshape(-1)
    condition[torch.as_tensor(np.sort(chosen))] = 1.0
    counts[torch.as_tensor(np.sort(chosen))] = 1.0
    return condition.reshape(shape), counts.reshape(shape)


def _shuffled(counts: torch.Tensor, *, seed: int) -> torch.Tensor:
    """把计数在全部格子上随机置换（输入对目标零信息）。"""
    generator = torch.Generator().manual_seed(seed)
    flat = counts.reshape(-1)
    return flat[torch.randperm(flat.numel(), generator=generator)].reshape(counts.shape)


def _make_runner(
    grid: FieldGrid,
    condition: torch.Tensor,
    counts: torch.Tensor,
    *,
    seed: int,
) -> Callable[[], float]:
    """构造一步优化回调（每次调用跑一步 Adam 并返回标量 loss）。"""
    torch.manual_seed(seed)
    weight = torch.zeros((), dtype=torch.float64, requires_grad=True)
    bias = torch.full((), 10.0, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=LEARNING_RATE)

    def step() -> float:
        optimizer.zero_grad()
        lam = softplus_lambda(weight * condition + bias)
        loss = poisson_nll(counts, lam, grid)
        loss.backward()
        optimizer.step()
        return float(loss.item())

    return step


def test_g1_g4_gates_all_pass() -> None:
    """M9：四道门禁全绿；summarize() 输出直接进训练日志。"""
    grid = _grid()
    condition, counts = _condition_and_counts(grid)
    shuffled = _shuffled(counts, seed=7)
    n_events = float(counts.sum().item())
    total_volume = omega(grid, n_lines=1)

    results: list[GateResult] = []
    results.append(overfit_single_batch(_make_runner(grid, condition, counts, seed=1), steps=STEPS))
    results.append(
        shuffled_target_control(
            _make_runner(grid, condition, counts, seed=2),
            _make_runner(grid, condition, shuffled, seed=3),
            steps=STEPS,
        ),
    )
    trained = _make_runner(grid, condition, counts, seed=4)
    model_loss = trained()
    for _ in range(STEPS - 1):
        model_loss = trained()
    results.append(
        constant_baseline_gate(model_loss, constant_baseline_nll(n_events, total_volume))
    )

    # G4：tau 轴速率（格/秒，派生自 BPM 与基本格）与音频帧轴速率（派生自 MERT config）
    tau_rate = BEAT_SUBDIVISION * TEST_BPM / SECONDS_PER_MINUTE
    results.append(frame_rate_gate(grid.t_bins, grid.total_seconds, tau_rate))
    duration_s = grid.total_seconds
    frames = round(duration_s * MERT_FRAME_RATE_HZ)
    results.append(frame_rate_gate(frames, duration_s, MERT_FRAME_RATE_HZ))

    summary = summarize(results)
    # 测试输出（非生产路径）：门禁结果须可直接贴进训练日志
    print(summary)
    assert all(result.passed for result in results), summary
    assert MERT_FRAME_RATE_HZ == MERT_SAMPLE_RATE_HZ / MERT_CONV_STRIDE_PRODUCT


def test_poisson_floor_is_above_the_g3_baseline_gap() -> None:
    """G3 立论的数值检查：真实目标的可达下界必须显著低于常数基线。"""
    grid = _grid()
    condition, counts = _condition_and_counts(grid)
    n_events = float(counts.sum().item())
    volumes = torch.as_tensor(grid.cell_volumes(), dtype=torch.float64).reshape(1, -1, 1, 1, 1)
    perfect = torch.where(counts > 0, counts / volumes, torch.zeros_like(counts))
    floor = poisson_nll(counts, perfect, grid).item()
    baseline = constant_baseline_nll(n_events, omega(grid, n_lines=1))
    assert floor < baseline
    assert condition.sum().item() == n_events
