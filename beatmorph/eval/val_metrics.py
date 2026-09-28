"""验证集指标：遮盖测度下的 NLL 分解、常数场基线与条件对照读数（plan 07 §9-47 A–D / §9-51）。

**为什么这些函数住在这里，而不是训练循环里**

§9-47 F 要求 val 路径与离线 `beatmorph-eval` **共用同一份指标实现**。因此本模块是
「val 读数」的唯一实现处：训练循环只负责**取批 + 前向 + 决定什么时候算**，所有口径
（测度、归约、分母、基线）都在这里，且每个读数都指向同一批张量。

**三条口径（每一条都对应一个曾经错过的设计）**

1. **同一测度才可比**（§9-47 B）：训练损失走遮盖路径（只监督被遮盖事件 + 重标定 `1/r`），
   而 G3 的常数基线是全事件闭式解 —— 两者**不在同一测度上**，`infra/train_loop.py`
   明文禁止把它们配对。因此常数基线在这里**在与 val 完全相同的遮盖测度下重算**：
   常数场 `lambda == c` 的损失是 `L(c) = s * N_sup * (-log c) + c * |Omega|`
   （`s` = 重标定系数、`N_sup` = 参与监督的事件数），唯一极小点
   `c* = s * N_sup / |Omega|`；最小值再用 `masked_poisson_loss` **本身**求值 —— 于是基线与
   模型读数只差「谁给 lambda」，测度 / 归约 / 分母逐位一致。
   `tests/unit/eval/test_val_metrics.py` 用小合成批**暴力扫描** c 复核这个极小点。

2. **归约按有效线数**（§9-47 C）：`reduction="mean"` 的分母是批内**有效判定线数**
   （`_reduced(..., active=line_mask)`），因此 K 变化不会让读数跳动。跨批聚合沿用同一口径
   （Σ损失 / Σ有效线），`val_nll` 因此是「逐窗口读数按有效线加权」的均值。

3. **分层而不是混合**（§9-46 / 任务 B）：单步 loss 是**双峰**（43.8% 空窗中位 6.5e-4
   vs 56.2% 非空中位 96，差五个数量级），空窗的 loss **恒等于积分项**。因此
   `masked_readout` 同时给出 `nll_masked` / `nll_empty` / `nll_nonempty` 与两类窗口的
   有效线数；聚合器据此把它们分开上报，一步的数字不再混合两个总体。

**本模块不复制任何损失公式**（plan 06 §4.6 的同一条纪律）：事件项 / 积分项 / 重标定全部走
`beatmorph.generation.losses` 与 `beatmorph.generation.masks` 的权威实现；全事件口径走
`beatmorph.eval.calibration.poisson_nll_float`（离线评估的同一入口）。
torch 在**函数内**惰性引入，因此 `import beatmorph.eval` 仍不拉起 torch
（`tests/unit/eval/test_source_hygiene.py` 盯着这条）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查期需要 torch / batch
    from torch import Tensor

    from beatmorph.generation.batch import FieldBatch, FieldOutput
    from beatmorph.generation.losses import Reduction

__all__ = [
    "FULL_EVENT_WARNING",
    "ConstantField",
    "MaskedMeasure",
    "MaskedReadout",
    "ReweightMode",
    "ValAccumulator",
    "constant_field_nll",
    "constant_field_output",
    "constant_field_solution",
    "full_event_nll",
    "masked_measure",
    "masked_nll",
    "masked_readout",
    "measure_volume",
]

#: 重标定口径（与 `generation.losses.ReweightMode` **逐字一致**；本模块不重新定义语义）
ReweightMode = Literal["hidden", "observed", "hidden_doc", "none"]

#: 全事件口径读数的**固定标注**（§9-47 B：不得与遮盖测度的常数基线直接比）。
FULL_EVENT_WARNING: str = (
    "全事件口径：与遮盖测度的常数基线不可直接比（无遮盖时输入就是目标，plan 07 §9-47 B）"
)


@dataclass(frozen=True, slots=True)
class MaskedMeasure:
    """常数场在该遮盖测度下的两个权重（**只有这两个量决定最优常数 c**）。

    Attributes:
        reweight: 重标定口径（见 :data:`ReweightMode`）。
        ratio: 遮盖比例 r = 被遮盖事件数 / 总事件数（`occlusion_ratio` 的权威实现给出）。
        scale: 事件项的重标定系数 s（hidden -> 1/r；observed -> 1/(1-r)；none -> 1）。
        supervised_events: 参与监督的事件数 N_sup（hidden -> 被遮盖事件；observed -> 已观测事件）。
        total_events: 批内事件总数 N（桶内计数按 n_j 计）。
        hidden_events: 被遮盖格子里的事件数 N_h。
    """

    reweight: str
    ratio: float
    scale: float
    supervised_events: float
    total_events: float
    hidden_events: float

    def optimum_intensity(self, volume: float) -> float:
        """常数场的最优强度 c* = s * N_sup / |Omega|。

        `d/dc [s * N_sup * (-log c) + c * |Omega|] = 0` 的唯一正根。

        Raises:
            ValueError: |Omega| 非正（无有效域，基线无定义）。
        """
        if volume <= 0.0:
            raise ValueError(f"|Omega| 必须为正，得到 {volume!r}")
        return float(self.scale) * float(self.supervised_events) / float(volume)


@dataclass(frozen=True, slots=True)
class ConstantField:
    """常数场基线（**同一遮盖测度**下的最优常数场与其测度）。

    Attributes:
        intensity: c*（常数强度）。
        volume: |Omega| = sum_j dV_j（**有效线上的全域**，与积分项同一实现）。
        measure: 生成该最优值的测度权重。
    """

    intensity: float
    volume: float
    measure: MaskedMeasure


@dataclass(frozen=True, slots=True)
class MaskedReadout:
    """一个批在遮盖测度下的读数（**同一批、同一权重、同一测度**）。

    Attributes:
        windows: 本批窗口数 B。
        lines: 有效判定线数（`reduction="mean"` 的分母）。
        nll_masked: 模型输出在本测度下的 NLL（mean 口径 = Σ损失 / Σ有效线）。
        nll_constant: 常数场在同一测度下的最优值（`val_ratio` 的分母）。
        nll_empty / lines_empty: 空窗（events == 0）那一支的损失与有效线；无空窗时 None。
        nll_nonempty / lines_nonempty: 非空窗那一支；无非空窗时 None。
        integral: Σ int lambda dV（**校准比的分母**）。
        events: Σ n（真实事件数）。
        empty_windows: 空窗数量。
        r0_windows: **退化窗口**数量：有事件但一个都没被遮盖（r == 0 ⇒ 损失退化为全事件口径）。
            口径与数据集的 `no_visible_context_fallbacks` 一致；空窗不计入（它们的
            hidden == 0 是平凡真值，混进来会把这一项变成 empty_share 的副本）。
        ratio: 批级遮盖比 r。
    """

    windows: int
    lines: float
    nll_masked: float
    nll_constant: float
    nll_empty: float | None
    lines_empty: float
    nll_nonempty: float | None
    lines_nonempty: float
    integral: float
    events: float
    empty_windows: int
    r0_windows: int
    ratio: float

    @property
    def empty_share(self) -> float:
        """空窗占比（固定验证集里它是常数 ⇒ 影响指标**水平**不影响**趋势**，§9-47 G-⑤）。"""
        return float(self.empty_windows) / float(self.windows) if self.windows else 0.0

    @property
    def r0_share(self) -> float:
        """退化窗口（有事件但无遮盖）占比。"""
        return float(self.r0_windows) / float(self.windows) if self.windows else 0.0

    @property
    def pred_over_true(self) -> float | None:
        """校准比 Σ∫λdV / Σn（期望事件数 / 真实事件数）；无事件时为 None（**不是 0**）。"""
        return self.integral / self.events if self.events > 0.0 else None


def _float_counts(batch: FieldBatch) -> Tensor:
    """counts 的 float32 视图（缺失即抛：遮盖测度没有目标就无从定义）。"""
    import torch

    if batch.counts is None:
        raise ValueError("遮盖测度需要 batch.counts（推理批次没有目标，不得静默记 0）")
    return batch.counts.to(dtype=torch.float32)


def masked_measure(batch: FieldBatch, *, reweight: ReweightMode = "hidden") -> MaskedMeasure:
    """`masked_poisson_loss` 在**常数场**下的测度权重（分支表与损失**逐条对齐**）。

    ⚠️ 这里有意**镜像** `generation.losses.masked_poisson_loss` 的口径分支（本模块不得改动
    损失语义，见 CLAUDE.md §3.1）。镜像的正确性由 `test_val_metrics.py` 的暴力扫描护栏保证：
    扫描用的是**真的** `masked_poisson_loss`。

    Raises:
        ValueError: 缺 counts，或 r == 1（全部事件被遮盖，事件项没有可见上下文）。
    """
    from beatmorph.generation.losses import occlusion_ratio
    from beatmorph.generation.masks import flat_counts_sum

    counts = _float_counts(batch)
    occlusion = batch.occlusion_bool()
    total = float(counts.sum().item())
    hidden = float(flat_counts_sum(counts, occlusion)) if batch.occlusion is not None else 0.0
    ratio = float(occlusion_ratio(batch))
    if reweight == "hidden":
        if ratio <= 0.0:  # 无遮盖 => 损失退化为全事件口径（losses.py 的 r == 0 契约）
            return MaskedMeasure(reweight, ratio, 1.0, total, total, hidden)
        if ratio >= 1.0:
            raise ValueError("r == 1：全部事件都被遮盖，常数场基线在遮盖测度下无定义")
        return MaskedMeasure(reweight, ratio, 1.0 / ratio, hidden, total, hidden)
    if reweight == "observed":
        if ratio >= 1.0:
            raise ValueError("r == 1：没有已观测事件，observed 口径无法定义")
        return MaskedMeasure(reweight, ratio, 1.0 / (1.0 - ratio), total - hidden, total, hidden)
    if reweight == "hidden_doc":
        if ratio <= 0.0:
            return MaskedMeasure(reweight, ratio, 1.0, total, total, hidden)
        if ratio >= 1.0:
            raise ValueError("r == 1：全部事件都被遮盖，常数场基线在遮盖测度下无定义")
        return MaskedMeasure(reweight, ratio, 1.0 / (1.0 - ratio), hidden, total, hidden)
    if reweight == "none":
        return MaskedMeasure(reweight, ratio, 1.0, hidden, total, hidden)
    raise ValueError(f"未知的重标定口径：{reweight!r}")


def measure_volume(batch: FieldBatch) -> float:
    """|Omega|：**与积分项同一实现**的全域测度（常数 1 场走 `integral_term`）。

    为什么不自己乘 dV：积分项的实现里还有 range_mask 与 line_mask 两层口径，
    在这里重算一份就是第二份真相（红线 7 的同一类问题）。
    """
    import torch

    from beatmorph.generation.batch import FieldOutput
    from beatmorph.generation.losses import integral_term

    ones = torch.ones(
        batch.batch_field_shape(),
        dtype=torch.float32,
        device=batch.line_mask.device,
    )
    return float(integral_term(FieldOutput(lam=ones), batch).sum().item())


def constant_field_output(batch: FieldBatch, value: float) -> FieldOutput:
    """构造 `lambda == value` 的 `FieldOutput`（range / line 掩码由损失内部施加）。"""
    import torch

    from beatmorph.generation.batch import FieldOutput

    lam = torch.full(
        batch.batch_field_shape(),
        float(value),
        dtype=torch.float32,
        device=batch.line_mask.device,
    )
    return FieldOutput(lam=lam)


def constant_field_solution(
    batch: FieldBatch, *, reweight: ReweightMode = "hidden"
) -> ConstantField:
    """常数场在**该批的遮盖测度**下的最优解 (c*, |Omega|, 测度)。"""
    volume = measure_volume(batch)
    measure = masked_measure(batch, reweight=reweight)
    return ConstantField(
        intensity=measure.optimum_intensity(volume), volume=volume, measure=measure
    )


def constant_field_nll(
    batch: FieldBatch,
    *,
    reweight: ReweightMode = "hidden",
    reduction: Reduction = "mean",
) -> float:
    """常数场基线的 NLL（**在与 val 相同的测度与归约下**求值）。

    `reduction="mean"` 与 `masked_readout` 的 `nll_masked` 同口径 ⇒ 两者之比
    （`val_ratio`）才是有意义的相对判据（§9-47 B）。
    """
    from beatmorph.generation.losses import masked_poisson_loss

    solution = constant_field_solution(batch, reweight=reweight)
    output = constant_field_output(batch, solution.intensity)
    value = masked_poisson_loss(output, batch, reweight=reweight, reduction=reduction)
    return float(value.item())


def masked_nll(
    batch: FieldBatch,
    lam: Tensor,
    *,
    reweight: ReweightMode = "hidden",
    reduction: Reduction = "mean",
) -> float:
    """给定强度场在遮盖测度下的 NLL（对照臂只要这一个数时用它）。"""
    from beatmorph.generation.batch import FieldOutput
    from beatmorph.generation.losses import masked_poisson_loss

    value = masked_poisson_loss(FieldOutput(lam=lam), batch, reweight=reweight, reduction=reduction)
    return float(value.item())


def masked_readout(
    batch: FieldBatch,
    lam: Tensor,
    *,
    reweight: ReweightMode = "hidden",
) -> MaskedReadout:
    """一个 val 批的完整读数（模型臂 + 常数场基线 + 分层）。

    实现要点：`reduction="none"` 拿到逐 (B, K) 的损失后**先按 K 求和得到逐窗口损失**，
    再按「Σ损失 / Σ有效线」聚合 —— 这正是 `reduction="mean"` 的定义（`_reduced` 的
    `active=line_active(batch)`），因此 `nll_masked` 与直接调损失逐位一致
    （`test_val_metrics.py` 断言这一条）。
    """
    import torch

    from beatmorph.generation.batch import FieldOutput
    from beatmorph.generation.losses import integral_term, masked_poisson_loss

    counts = _float_counts(batch)
    output = FieldOutput(lam=lam)
    per_sample = masked_poisson_loss(output, batch, reweight=reweight, reduction="none").sum(dim=1)
    integral = integral_term(output, batch).sum(dim=1)
    lines = batch.line_mask_bool().sum(dim=1).to(dtype=torch.float32)
    events = counts.sum(dim=(1, 2, 3, 4, 5))
    hidden = (counts * batch.occlusion_bool().to(dtype=torch.float32)).sum(dim=(1, 2, 3, 4, 5))

    total_lines = float(lines.sum().item())
    divisor = max(total_lines, 1.0)
    empty = events <= 0.0
    degenerate = (hidden <= 0.0) & (events > 0.0)
    lines_empty = float(lines[empty].sum().item())
    lines_nonempty = float(lines[~empty].sum().item())
    return MaskedReadout(
        windows=batch.batch_size(),
        lines=total_lines,
        nll_masked=float(per_sample.sum().item()) / divisor,
        nll_constant=constant_field_nll(batch, reweight=reweight, reduction="mean"),
        nll_empty=(
            float(per_sample[empty].sum().item()) / lines_empty if lines_empty > 0.0 else None
        ),
        lines_empty=lines_empty,
        nll_nonempty=(
            float(per_sample[~empty].sum().item()) / lines_nonempty
            if lines_nonempty > 0.0
            else None
        ),
        lines_nonempty=lines_nonempty,
        integral=float(integral.sum().item()),
        events=float(events.sum().item()),
        empty_windows=int(empty.sum().item()),
        r0_windows=int(degenerate.sum().item()),
        ratio=masked_measure(batch, reweight=reweight).ratio,
    )


def full_event_nll(batch: FieldBatch, lam: Tensor) -> float:
    """全事件口径的 NLL（**附带读数，不可与常规常数基线直接比**）。

    走离线评估的同一条入口（`calibration.poisson_nll_float` -> `field.loss.poisson_nll`），
    因此 val 与 `beatmorph-eval` 的全事件数字是同一份实现。
    归约沿用有效线数均值（与 `masked_readout` 同分母口径），便于并列比较。
    """
    from beatmorph.eval.calibration import poisson_nll_float

    if batch.counts is None:
        raise ValueError("全事件口径需要 batch.counts")
    counts = _float_counts(batch)
    line_mask = batch.line_mask_bool()
    total = 0.0
    for index in range(batch.batch_size()):
        total += poisson_nll_float(
            counts[index],
            lam[index],
            batch.grid,
            line_mask=[bool(value) for value in line_mask[index].tolist()],
        )
    lines = float(line_mask.sum().item())
    return total / max(lines, 1.0)


@dataclass(slots=True)
class ValAccumulator:
    """跨 val 批的读数累加器（**按有效线数加权**，与 `reduction="mean"` 同口径）。

    为什么用累加器而不是逐批平均：`val_nll` 的定义是「Σ损失 / Σ有效线」，
    逐批平均会让 K 大的批与 K 小的批等权，而它们的线数并不相等。
    """

    windows: int = 0
    empty_windows: int = 0
    r0_windows: int = 0
    lines: float = 0.0
    loss_masked: float = 0.0
    loss_constant: float = 0.0
    lines_constant: float = 0.0
    loss_empty: float = 0.0
    lines_empty: float = 0.0
    loss_nonempty: float = 0.0
    lines_nonempty: float = 0.0
    integral: float = 0.0
    events: float = 0.0
    loss_shuffled: float = 0.0
    lines_shuffled: float = 0.0
    loss_full_event: float = 0.0
    lines_full_event: float = 0.0
    #: 条件对照臂：名字 -> (Σ(损失 x 有效线), Σ有效线)。
    contrasts: dict[str, tuple[float, float]] = field(default_factory=dict)

    def add(
        self,
        readout: MaskedReadout,
        *,
        nll_shuffled: float | None = None,
        nll_full_event: float | None = None,
        contrasts: dict[str, float] | None = None,
    ) -> None:
        """累加一个批的读数（`contrasts` 是**同一批**上对照臂的绝对 NLL）。"""
        self.windows += int(readout.windows)
        self.empty_windows += int(readout.empty_windows)
        self.r0_windows += int(readout.r0_windows)
        self.lines += readout.lines
        self.loss_masked += readout.nll_masked * readout.lines
        self.loss_constant += readout.nll_constant * readout.lines
        self.lines_constant += readout.lines
        self.loss_empty += (readout.nll_empty or 0.0) * readout.lines_empty
        self.lines_empty += readout.lines_empty
        self.loss_nonempty += (readout.nll_nonempty or 0.0) * readout.lines_nonempty
        self.lines_nonempty += readout.lines_nonempty
        self.integral += readout.integral
        self.events += readout.events
        if nll_shuffled is not None:
            self.loss_shuffled += float(nll_shuffled) * readout.lines
            self.lines_shuffled += readout.lines
        if nll_full_event is not None:
            self.loss_full_event += float(nll_full_event) * readout.lines
            self.lines_full_event += readout.lines
        for name, value in (contrasts or {}).items():
            previous, weight = self.contrasts.get(name, (0.0, 0.0))
            self.contrasts[name] = (previous + float(value) * readout.lines, weight + readout.lines)

    # ── 聚合读数 ────────────────────────────────────────────────
    @property
    def nll(self) -> float | None:
        """模型在遮盖测度下的 NLL（Σ损失 / Σ有效线）。"""
        return self.loss_masked / self.lines if self.lines > 0.0 else None

    @property
    def nll_constant(self) -> float | None:
        """常数场在同一测度下的最优值（按有效线加权的跨批均值）。"""
        return self.loss_constant / self.lines_constant if self.lines_constant > 0.0 else None

    @property
    def ratio(self) -> float | None:
        """**主判据** val/ratio = val/nll / val/nll_constant（< 1 = 优于常数场）。"""
        model, baseline = self.nll, self.nll_constant
        if model is None or baseline is None or baseline <= 0.0:
            return None
        return model / baseline

    def scalars(self) -> dict[str, float]:
        """落盘读数（jsonl 的 `val_*` / `cond_*` 字段；**无定义的项不出现**，不填 0）。"""
        out: dict[str, float] = {
            "val_windows": float(self.windows),
            "val_lines": float(self.lines),
            "val_integral": float(self.integral),
            "val_events": float(self.events),
            "val_empty_share": (float(self.empty_windows) / float(self.windows))
            if self.windows
            else 0.0,
            "val_r0_share": (float(self.r0_windows) / float(self.windows)) if self.windows else 0.0,
        }
        model, baseline = self.nll, self.nll_constant
        if model is not None:
            out["val_nll"] = float(model)
        if baseline is not None:
            out["val_nll_constant"] = float(baseline)
        if self.ratio is not None:
            out["val_ratio"] = float(self.ratio)
        if self.lines_empty > 0.0:
            out["val_nll_empty"] = float(self.loss_empty / self.lines_empty)
        if self.lines_nonempty > 0.0:
            out["val_nll_nonempty"] = float(self.loss_nonempty / self.lines_nonempty)
        if self.lines_shuffled > 0.0:
            shuffled = self.loss_shuffled / self.lines_shuffled
            out["val_nll_shuffled"] = float(shuffled)
            if model is not None:
                out["val_nll_shuffled_delta"] = float(shuffled - model)
        if self.lines_full_event > 0.0:
            out["val_nll_full_event"] = float(self.loss_full_event / self.lines_full_event)
        if self.events > 0.0:
            out["val_pred_over_true"] = float(self.integral / self.events)
        for name, (total, weight) in sorted(self.contrasts.items()):
            if weight > 0.0 and model is not None:
                out[f"cond_{name}_delta"] = float(total / weight - model)
        return out
