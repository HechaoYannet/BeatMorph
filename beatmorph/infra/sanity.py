"""训练健全性门禁（POSTMORTEM-2026-08-05 制度化；RFC-0037 起为**三道**）。

三道门禁，任何新模型/新范式在**扩大数据规模之前**必须依次通过：

- **G1 单 batch 过拟合**：在 1-4 个样本上必须能把 loss 打到接近 0。
  过不了说明通路是坏的（梯度断、loss 用错、输入没接上），与数据量无关。
- **G3 常数基线**：模型 loss 必须显著优于"直接预测数据集均值"。否则模型
  其实什么都没学到，只是停在了回归到均值的那个地板上。
- **G4 契约断言**：帧率/形状等物理量必须由模型 config **派生**并断言，
  不得硬编码（25Hz vs 75Hz 之痛）。

⚠️ **G2（打乱标签对照）已删除（RFC-0037，2026-09-28 决策者裁定）**：单批训练损失
「速度赛」在真实批上判别力未被证明（逐批 verdict 翻转、nuisance 沾染、跨进程漂移，
证据见 RFC-0036 §1-§2 与 plan 07 §9-54/§9-55）。「输入对目标有没有信息」改由
**held-out 在线对照**承担：`val/nll_shuffled_delta`（线内置换）+ `val/ratio` +
`cond_*_delta`（条件干预）。判读红线：任一 val 点 `val/ratio >= 1` 或
`val/nll_shuffled_delta <= 0` ⇒ 该 run 的扩规模结论作废（docs/TRAINING.md）。

设计原则：**范式中立**。本模块只吃调用方给的 `step_fn`（跑一步优化并返回
标量 loss），不 import torch、不认识任何具体模型/数据集/tokenizer，因此
在最小环境下也能跑，也就不会被"依赖缺失"跳过——这正是 G4 当年失败的方式。

用法::

    from beatmorph.infra.sanity import overfit_single_batch


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
        name: 门禁名（G1/G3/G4；RFC-0037 起 G2 已删除）。
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
    lines = ["健全性门禁（G1/G3/G4）："]
    lines += [f"  [{'PASS' if r.passed else 'FAIL'}] {r.name}: {r.detail}" for r in results]
    n_fail = sum(1 for r in results if not r.passed)
    lines.append(
        f"  => {'全部通过' if n_fail == 0 else f'{n_fail} 项未通过'}（共 {len(results)} 项）"
    )
    return "\n".join(lines)
