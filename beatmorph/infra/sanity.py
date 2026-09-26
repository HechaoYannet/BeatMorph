"""训练健全性门禁（POSTMORTEM-2026-08-05 制度化）。

四道门禁，任何新模型/新范式在**扩大数据规模之前**必须依次通过：

- **G1 单 batch 过拟合**：在 1-4 个样本上必须能把 loss 打到接近 0。
  过不了说明通路是坏的（梯度断、loss 用错、输入没接上），与数据量无关。
- **G2 打乱标签对照**：把目标打乱后 loss **必须显著变差**。若不变，说明输入
  对目标零信息——帧率/对齐类 bug 会在这里当场现形。
- **G3 常数基线**：模型 loss 必须显著优于"直接预测数据集均值"。否则模型
  其实什么都没学到，只是停在了回归到均值的那个地板上。
- **G4 契约断言**：帧率/形状等物理量必须由模型 config **派生**并断言，
  不得硬编码（25Hz vs 75Hz 之痛）。

设计原则：**范式中立**。本模块只吃调用方给的 `step_fn`（跑一步优化并返回
标量 loss），不 import torch、不认识任何具体模型/数据集/tokenizer，因此
在最小环境下也能跑，也就不会被"依赖缺失"跳过——这正是 G4 当年失败的方式。

用法::

    from beatmorph.infra.sanity import overfit_single_batch, shuffled_target_control


    def step() -> float:
        loss = train_step(model, batch)  # 调用方自己的一步优化
        return float(loss)


    results = [overfit_single_batch(step), ...]
    assert all(r.passed for r in results), results
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

#: 跑一步优化并返回标量 loss 的回调（调用方负责 backward/step）。
StepFn = Callable[[], float]


@dataclass(frozen=True)
class GateResult:
    """单道门禁的结果。

    Attributes:
        name: 门禁名（G1..G4）。
        passed: 是否通过。
        detail: 人类可读的判据与实际值，便于直接贴进训练日志。
    """

    name: str
    passed: bool
    detail: str

    def __bool__(self) -> bool:
        return self.passed


def _run(step_fn: StepFn, steps: int) -> tuple[float, float]:
    """跑 `steps` 步，返回 (首步 loss, 末步 loss)。"""
    if steps < 1:
        raise ValueError(f"steps 必须 >= 1，得到 {steps}")
    first = float(step_fn())
    last = first
    for _ in range(steps - 1):
        last = float(step_fn())
    return first, last


def overfit_single_batch(
    step_fn: StepFn,
    *,
    steps: int = 300,
    target_loss: float = 0.05,
    target_ratio: float = 0.1,
) -> GateResult:
    """G1：单/少样本过拟合门禁。

    判据：末步 loss <= max(`target_loss`, `target_ratio` × 首步 loss)。
    在 1-4 个样本上**必须**能被打穿；打不穿说明通路本身是坏的。

    Args:
        step_fn: 一步优化回调。
        steps: 优化步数（默认 300）。
        target_loss: 绝对目标 loss。
        target_ratio: 相对首步的下降比例目标。
    """
    first, last = _run(step_fn, steps)
    ceiling = max(target_loss, target_ratio * first)
    passed = last <= ceiling
    return GateResult(
        "G1 单batch过拟合",
        passed,
        f"loss {first:.6f} -> {last:.6f}（{steps} 步），需 <= {ceiling:.6f}",
    )


def shuffled_target_control(
    step_fn_real: StepFn,
    step_fn_shuffled: StepFn,
    *,
    steps: int = 300,
    min_gap_ratio: float = 0.05,
) -> GateResult:
    """G2：打乱标签对照门禁。

    判据：打乱目标的末步 loss 必须 >= 真实目标的末步 loss × (1 + `min_gap_ratio`)。
    两者持平 => 输入对目标**零信息**（帧率错配、索引错位、特征与标签不同源）。

    Args:
        step_fn_real: 标签正确的优化回调。
        step_fn_shuffled: 标签被打乱的优化回调（同模型/同输入）。
        steps: 优化步数。
        min_gap_ratio: 要求的最小相对差距。
    """
    _, real_last = _run(step_fn_real, steps)
    _, shuf_last = _run(step_fn_shuffled, steps)
    need = real_last * (1.0 + min_gap_ratio)
    passed = shuf_last >= need
    return GateResult(
        "G2 打乱标签对照",
        passed,
        f"真实 loss {real_last:.6f} vs 打乱 {shuf_last:.6f}，需打乱 >= {need:.6f}",
    )


def constant_baseline_gate(
    model_loss: float,
    baseline_loss: float,
    *,
    min_improvement: float = 0.1,
) -> GateResult:
    """G3：常数（均值）基线门禁。

    判据：`model_loss` <= `baseline_loss` × (1 - `min_improvement`)。
    基线 = 直接预测训练集目标均值所达到的 loss。

    Args:
        model_loss: 模型在验证/训练集上的 loss。
        baseline_loss: 常数均值预测器的 loss。
        min_improvement: 要求的最小相对改进。
    """
    ceiling = baseline_loss * (1.0 - min_improvement)
    passed = model_loss <= ceiling
    return GateResult(
        "G3 常数基线",
        passed,
        f"模型 {model_loss:.6f} vs 均值基线 {baseline_loss:.6f}，需 <= {ceiling:.6f}",
    )


def frame_rate_gate(
    frames: int,
    duration_s: float,
    frame_rate: float,
    *,
    tol_frames: int = 2,
) -> GateResult:
    """G4：帧率契约门禁。

    判据：`frames` ≈ `duration_s` × `frame_rate`（±`tol_frames` 帧，卷积取整）。
    `frame_rate` **必须**由模型 config 派生传入，不得硬编码。

    Args:
        frames: 实际得到的帧数（如 `emb.shape[1]`）。
        duration_s: 输入音频时长（秒）。
        frame_rate: 由 config 派生的帧率（Hz）。
        tol_frames: 容差帧数。
    """
    expected = duration_s * frame_rate
    diff = abs(frames - expected)
    passed = diff <= tol_frames
    return GateResult(
        "G4 帧率契约",
        passed,
        f"{duration_s:.2f}s × {frame_rate:.1f}Hz = {expected:.1f} 帧，实际 {frames} 帧"
        f"（差 {diff:.1f}，容差 {tol_frames}）",
    )


def summarize(results: list[GateResult]) -> str:
    """把门禁结果渲染成可直接贴进训练日志的多行文本。"""
    lines = ["健全性门禁（G1-G4）："]
    lines += [f"  [{'PASS' if r.passed else 'FAIL'}] {r.name}: {r.detail}" for r in results]
    n_fail = sum(1 for r in results if not r.passed)
    lines.append(
        f"  => {'全部通过' if n_fail == 0 else f'{n_fail} 项未通过'}（共 {len(results)} 项）"
    )
    return "\n".join(lines)
