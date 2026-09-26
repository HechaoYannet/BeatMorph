"""强度场张量契约（Plan 00 §3.7）。

本模块只描述**网格元数据、张量形状与测度**，不含任何算法：

- 网格数值积分与累积强度参数化（两条路径）在 beatmorph/field/ 实现；
- 秒到拍（tau）的换算**只在 beatmorph/field/ 实现**（CLAUDE.md 红线 7），
  本模块只携带 bpm_points 作为**数据**（换算依据），不实现换算函数。

不引入 torch 依赖：张量类型标注放在 TYPE_CHECKING 下的前向引用，
保证契约层可在最小环境（无 torch、无权重、无 GPU）导入与测试（I12）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias

if TYPE_CHECKING:
    from beatmorph.core.contracts.phigros import BpmPoint

from beatmorph.core.contracts.phigros import (
    RPE_STAGE_HALF_WIDTH,
    RPE_STAGE_WIDTH,
    RPE_X_GRID_BINS,
    RPE_X_GRID_DX,
    RPE_X_GRID_MAX,
    RPE_X_GRID_MIN,
    SUBDIVISIONS_PER_BEAT,
    TAU_GRID_DT,
    NoteType,
    Side,
)

#: 张量占位类型：契约层**不导入 torch**（I12），因此张量字段用不透明别名标注。
#: 运行期由 beatmorph/field/ 与 generation/ 传入真实 torch.Tensor。
TensorLike: TypeAlias = Any

#: 强度场张量的 einops 形状（**唯一**写法；任何模块不得另写一套维度序）。
FIELD_SHAPE: str = "(batch, k, t, x, s, c)"
FIELD_DIMS: tuple[str, ...] = ("batch", "k", "t", "x", "s", "c")

#: 类型通道数 = 4 类音符 + 1 个 hold-end 标记通道。
TYPE_CHANNELS: tuple[str, ...] = ("tap", "drag", "hold", "hold_end", "flick")


@dataclass(frozen=True, slots=True)
class ChartFieldSpec:
    """强度场网格规格（**唯一**的网格元数据来源，缺一不可断言）。

    Attributes:
        k: 判定线条数（运行期可变，**不得**写进任何输出层维度）。
        t_bins: T = tau 轴格数（由 BPMList 与谱面时长派生，见 plan 00 §3.6）。
        d_tau: 拍格宽 = 1 / SUBDIVISIONS_PER_BEAT（派生，禁止写字面量）。
        bpm_points: tau 与秒换算的**唯一依据**；换算函数本身由 field/ 实现。
        x_bins: X 轴桶数（默认 128；消融见 RPE_X_GRID_BIN_SWEEP）。
        dx: 桶宽 = RPE_STAGE_WIDTH / x_bins。
        x_min / x_max: 可见范围边界（= 正负半宽）。
        sides: S = 2（FRONT 到 0 / BACK 到 1）。
        channels: C = 5（tap / drag / hold / flick / hold-end）。
    """

    k: int
    t_bins: int
    d_tau: float = TAU_GRID_DT
    bpm_points: tuple[BpmPoint, ...] = ()
    x_bins: int = RPE_X_GRID_BINS
    dx: float = RPE_X_GRID_DX
    x_min: float = RPE_X_GRID_MIN
    x_max: float = RPE_X_GRID_MAX
    sides: int = len(Side)
    channels: int = len(NoteType) + 1

    def assert_grid(self) -> None:
        """网格自洽断言。失败即抛，**不得降级为日志**（plan 00 I1/I2/I9）。"""
        if self.dx * self.x_bins != RPE_STAGE_WIDTH:
            raise AssertionError(
                f"dx * x_bins 必须等于 RPE_STAGE_WIDTH：{self.dx} * {self.x_bins}",
            )
        if self.d_tau * SUBDIVISIONS_PER_BEAT != 1.0:
            raise AssertionError(f"d_tau * SUBDIVISIONS_PER_BEAT 必须等于 1：{self.d_tau}")
        if self.x_min != -RPE_STAGE_HALF_WIDTH or self.x_max != +RPE_STAGE_HALF_WIDTH:
            raise AssertionError(f"x 范围必须等于正负半宽：[{self.x_min}, {self.x_max}]")
        if self.k < 1:
            raise AssertionError(f"k 必须 >= 1，得到 {self.k}")
        if self.t_bins < 0:
            raise AssertionError(f"t_bins 必须 >= 0，得到 {self.t_bins}")
        if self.sides != len(Side):
            raise AssertionError(f"sides 必须等于 len(Side)={len(Side)}，得到 {self.sides}")
        if self.channels != len(NoteType) + 1:
            raise AssertionError(
                f"channels 必须等于 len(NoteType)+1={len(NoteType) + 1}，得到 {self.channels}",
            )
        if not self.bpm_points:
            raise AssertionError("bpm_points 不得为空（tau 与秒换算的唯一依据）")
        previous = -1.0
        for point in self.bpm_points:
            if point.time_beats < previous:
                raise AssertionError("bpm_points 必须按 time_beats 升序")
            previous = point.time_beats

    def cell_count(self) -> int:
        """格元总数 k * T * X * S * C（plan 00 §3.7 / RFC-0029 §8.4 R-a 口径）。

        ⚠️ 这是**均匀网格**下的格元计数。Q15 之后 tau 网格随 BPM 变化，
        测度体积 |Omega| = sum_j dV_j 由 beatmorph/field/ 的 FieldGrid.volume()
        给出（plan 03 §2 偏离 2 / §9-2）；两者在常数 J 下一致。
        """
        return self.k * self.t_bins * self.x_bins * self.sides * self.channels

    def shape(self) -> tuple[int, ...]:
        """张量形状 (K, T, X, S, C)（不含 batch 轴）。"""
        return (self.k, self.t_bins, self.x_bins, self.sides, self.channels)


def x_bin_index(x: float, spec: ChartFieldSpec) -> int | None:
    """x 到桶索引的**唯一**映射路径（禁止各模块自行 floor）。

    闭区间右端 x == x_max 归入最后一桶；域外返回 None（**不钳位**）。
    """
    if x < spec.x_min or x > spec.x_max:
        return None
    if x == spec.x_max:
        return spec.x_bins - 1
    index = int((x - spec.x_min) // spec.dx)
    if index < 0 or index >= spec.x_bins:
        return None
    return index


@dataclass(frozen=True, slots=True)
class ChartField:
    """模型侧强度场 lambda（形状见 FIELD_SHAPE；float32；lambda >= 0）。

    Attributes:
        lam: 非齐次强度 lambda >= 0（softplus / exp 参数化，**禁止裸线性输出**）。
        mask: **1 = 被遮盖（待补全）**，0 = 已观测；必须显式存在。
        line_mask: 该 batch 内真实存在的线；k >= spec.k 的部分为 padding。
        time_mask: 有效时间帧（音频 padding 区为 0）。
    """

    SHAPE: ClassVar[str] = FIELD_SHAPE

    lam: TensorLike = None
    mask: TensorLike = None
    line_mask: TensorLike = None
    time_mask: TensorLike = None


@dataclass(frozen=True, slots=True)
class ChartTargetField:
    """目标侧场：**桶内事件计数**（把桶计数当泊松观测，不是 0/1 热图）。"""

    SHAPE: ClassVar[str] = FIELD_SHAPE

    counts: TensorLike = None
    #: 落在定义域外、已从事件项显式排除的事件数（plan 00 偏离 2）
    out_of_window: TensorLike = None


def measure_terms(counts: TensorLike, lam: TensorLike, cell_volumes: TensorLike) -> TensorLike:
    """测度声明（契约的**文字化**落点，不做数值计算）。

        integral(lambda) ~= sum_{k,t,x,s,c} lambda[k,t,x,s,c] * J(tau_t) * d_tau * dx

    侧别与类型通道是离散类别轴，各贡献 1；x 是连续轴的离散化，贡献 dx；
    时间是 beat-aligned 轴，贡献 J(tau_t) * d_tau。**J(tau) 只由 field/ 求值。**
    """
    raise NotImplementedError(
        "测度计算在 beatmorph/field/ 实现（plan 03）；契约层只固定测度与形状",
    )


__all__ = [
    "FIELD_DIMS",
    "FIELD_SHAPE",
    "TYPE_CHANNELS",
    "ChartField",
    "ChartFieldSpec",
    "ChartTargetField",
    "TensorLike",
    "measure_terms",
    "x_bin_index",
]
