"""强度场网格、派生常量与「秒 <-> tau」的唯一换算实现（Plan 03 §3.1 / M1 / M12）。

CLAUDE.md 红线 7：物理常量必须写成**派生式**，且「秒 <-> 拍（tau）」的换算
**只允许在 beatmorph/field/ 内实现**。本文件是那**唯一**的落点：

- FieldGrid.jacobian / tau_to_seconds / seconds_to_tau / volume / cell_volumes
  是下游（generation / decoder / io / eval）唯一允许调用的换算入口；
- 本文件**刻意不 import torch**（只用 numpy 与标量），因此最小环境即可跑
  M12 的往返无损契约测试；张量侧 API 在 integrate.py / loss.py。

tau 是**拍相对坐标**（beat-aligned，基本格 1/48 拍；RFC-0029 §3.1 Q15），既不是
秒，也不是音频帧（MERT 帧率只服务 Stage 0 的音频帧轴，见 contracts.tensors）。

测度：格体积 dV_j = J_j * d_tau * dx，J(tau) = dt/dtau = 60 / bpm(tau) 由谱面
BPMList **分段派生**；tau 网格因 BPM 变化而**非均匀**，因此 |Omega| = sum_j dV_j
（不再等于 K*T*X*S*C，见 plan 03 §2 偏离 2 / §9-2）。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal, overload

import numpy as np
from numpy.typing import NDArray

from beatmorph.core.contracts.field import TYPE_CHANNELS, ChartFieldSpec
from beatmorph.core.contracts.phigros import (
    RPE_STAGE_HALF_WIDTH,
    RPE_STAGE_WIDTH,
    RPE_X_GRID_BIN_SWEEP,
    RPE_X_GRID_BINS,
    SUBDIVISIONS_PER_BEAT,
    TAU_GRID_DT,
    BpmPoint,
    PhigrosChart,
    Side,
)
from beatmorph.core.contracts.tensors import (
    MERT_CONV_STRIDE_PRODUCT,
    MERT_SAMPLE_RATE_HZ,
)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

# ── 派生常量（全部引用 core/contracts，禁止写字面量；M1）────────────────────
#: x 轴默认桶数（RFC-0029 §3.1 裁定；消融见 RPE_X_GRID_BIN_SWEEP）
DEFAULT_X_BINS: Final[int] = RPE_X_GRID_BINS
#: N 消融档位（M7 只固定接口与曲线，最终取值待碰撞统计）
X_BIN_SWEEP: Final[tuple[int, ...]] = RPE_X_GRID_BIN_SWEEP
#: 侧别数 = len(Side)（above / below 两条真值）
N_SIDES: Final[int] = len(Side)
#: 类型通道数 = len(TYPE_CHANNELS) = 4 类音符 + hold-end
N_CHANNELS: Final[int] = len(TYPE_CHANNELS)
#: 时间基本格（1/48 拍）；网格随 BPM 变化，不再与音频帧率绑死
BEAT_SUBDIVISION: Final[int] = SUBDIVISIONS_PER_BEAT
#: 「分钟」这一时间单位的定义（**不是**被测物理量，故允许为字面量）；
#: 全部 bpm -> 秒 的换算都必须经它，不得散写 60。
SECONDS_PER_MINUTE: Final[float] = 60.0
#: 音频帧率：派生量，不是超参（BasePlan §3.1 / POSTMORTEM-2026-08-05）。
#: **它只服务 Stage 0 的音频帧轴**，不是场的时间轴；此处仅供 G4 门禁与对齐处引用。
MERT_FRAME_RATE_HZ: Final[float] = MERT_SAMPLE_RATE_HZ / MERT_CONV_STRIDE_PRODUCT
#: tau 格索引的浮点护栏：仅用于抵消「恰好落在格界」的 1~2 ulp 误差。
#: 它不是物理常量，量级（1e-9 格 = 1/48 拍的十亿分之一）远小于任何真实量化网格。
TAU_INDEX_EPS: Final[float] = 1e-9

#: 格宽取值规则（BPM 变更点不落在格界时的取法见 plan 03 §9-16，本实现默认「左段」）。
CellSecondsRule = Literal["left", "right", "exact"]


# ══════════════════════════════════════════════════════════════
# §3.1 BPM 分段与秒 <-> tau（本模块唯一入口）
# ══════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class BpmSegment:
    """一个 BPM 段的解析形式（tau 区间 + 段内常量 J）。

    Attributes:
        tau_start: 段起点（拍）。
        tau_end: 段终点（拍）；最后一段为 +inf。
        bpm: 段内 BPM（每分钟节拍数）。
        seconds_start: 段起点对应的秒（由前序段累加，**逐段精确**）。
    """

    tau_start: float
    tau_end: float
    bpm: float
    seconds_start: float

    @property
    def seconds_per_beat(self) -> float:
        """J = dt/dtau = SECONDS_PER_MINUTE / bpm（秒/拍），段内常量。"""
        return SECONDS_PER_MINUTE / self.bpm

    @property
    def beat_span(self) -> float:
        """段内拍数（最后一段为 +inf）。"""
        return self.tau_end - self.tau_start

    @property
    def seconds_span(self) -> float:
        """段内秒数（最后一段为 +inf）。"""
        return self.beat_span * self.seconds_per_beat


def bpm_segments(bpm_points: Sequence[BpmPoint]) -> tuple[BpmSegment, ...]:
    """把 BPMList 解析成**补全到整个 tau 轴**的分段常量 J。

    规则（plan 03 §3.1 + §9-16）：

    - 按 time_beats 升序；同一拍出现多个点时**后者覆盖前者**；
    - 首段起点若 > 0，则 [0, 首段起点) 沿用首段 BPM（外推），使换算在全轴有定义；
    - 段区间为**左闭右开** [tau_i, tau_{i+1})，最后一段延伸到 +inf；
      因此 tau 恰好落在变更点上时取**右段**（J 在段界跳变）。
    """
    if not bpm_points:
        raise ValueError("bpm_points 不得为空（秒与 tau 换算的唯一依据）")
    ordered = sorted(bpm_points, key=lambda point: point.time_beats)
    starts: list[tuple[float, float]] = []
    for point in ordered:
        if point.bpm <= 0.0:
            raise ValueError(f"bpm 必须为正，得到 {point.bpm!r}")
        if starts and starts[-1][0] == point.time_beats:
            starts[-1] = (point.time_beats, point.bpm)
        else:
            starts.append((point.time_beats, point.bpm))
    if starts[0][0] > 0.0:
        starts.insert(0, (0.0, starts[0][1]))
    segments: list[BpmSegment] = []
    seconds = 0.0
    for index, (start, bpm) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else math.inf
        segments.append(
            BpmSegment(tau_start=start, tau_end=end, bpm=bpm, seconds_start=seconds),
        )
        seconds += (end - start) * (SECONDS_PER_MINUTE / bpm)
    return tuple(segments)


def _segment_arrays(
    bpm_points: Sequence[BpmPoint],
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """分段数组：(段起点 tau, 段起点秒, 段内秒/拍)。"""
    segments = bpm_segments(bpm_points)
    return (
        np.asarray([seg.tau_start for seg in segments], dtype=np.float64),
        np.asarray([seg.seconds_start for seg in segments], dtype=np.float64),
        np.asarray([seg.seconds_per_beat for seg in segments], dtype=np.float64),
    )


def _locate(values: FloatArray, starts: FloatArray) -> NDArray[np.intp]:
    """定位每个值所属段（左闭右开；越界侧夹到首/末段）。"""
    index = np.searchsorted(starts, values, side="right") - 1
    return np.clip(index, 0, starts.size - 1).astype(np.intp)


def _is_scalar(value: float | FloatArray) -> bool:
    return np.isscalar(value) or (isinstance(value, np.ndarray) and value.ndim == 0)


@overload
def jacobian_at(tau: float, bpm_points: Sequence[BpmPoint]) -> float: ...


@overload
def jacobian_at(tau: FloatArray, bpm_points: Sequence[BpmPoint]) -> FloatArray: ...


def jacobian_at(
    tau: float | FloatArray,
    bpm_points: Sequence[BpmPoint],
) -> float | FloatArray:
    """J(tau) = dt/dtau = 60 / bpm(tau)（秒/拍）；段内常量、段界跳变。"""
    starts, _, seconds_per_beat = _segment_arrays(bpm_points)
    if _is_scalar(tau):
        index = int(_locate(np.asarray([float(tau)], dtype=np.float64), starts)[0])
        return float(seconds_per_beat[index])
    values = np.asarray(tau, dtype=np.float64)
    return np.asarray(seconds_per_beat[_locate(values, starts)], dtype=np.float64)


@overload
def tau_to_seconds(tau: float, bpm_points: Sequence[BpmPoint]) -> float: ...


@overload
def tau_to_seconds(tau: FloatArray, bpm_points: Sequence[BpmPoint]) -> FloatArray: ...


def tau_to_seconds(
    tau: float | FloatArray,
    bpm_points: Sequence[BpmPoint],
) -> float | FloatArray:
    """t = int_0^tau J = sum_段 (段内拍数 * 60 / bpm_段)（**逐段解析求和**）。

    与官方参考实现的 beat2sec 逐段累加同式（docs/knowledges/phigros-format.md §7.1）。
    """
    starts, seconds_starts, seconds_per_beat = _segment_arrays(bpm_points)
    if _is_scalar(tau):
        value = float(tau)
        position = int(_locate(np.asarray([value], dtype=np.float64), starts)[0])
        return float(
            seconds_starts[position] + (value - starts[position]) * seconds_per_beat[position]
        )
    values = np.asarray(tau, dtype=np.float64)
    index = _locate(values, starts)
    return np.asarray(
        seconds_starts[index] + (values - starts[index]) * seconds_per_beat[index],
        dtype=np.float64,
    )


@overload
def seconds_to_tau(t_s: float, bpm_points: Sequence[BpmPoint]) -> float: ...


@overload
def seconds_to_tau(t_s: FloatArray, bpm_points: Sequence[BpmPoint]) -> FloatArray: ...


def seconds_to_tau(
    t_s: float | FloatArray,
    bpm_points: Sequence[BpmPoint],
) -> float | FloatArray:
    """tau = 上式按段反解（官方参考实现的 sec2beat 同式；与 tau_to_seconds 互逆）。

    超出最后一个 BPM 段起点的时间用最后一段 BPM 线性延长（总函数化，不抛错）。
    """
    starts, seconds_starts, seconds_per_beat = _segment_arrays(bpm_points)
    if _is_scalar(t_s):
        value = float(t_s)
        position = int(_locate(np.asarray([value], dtype=np.float64), seconds_starts)[0])
        return float(
            starts[position] + (value - seconds_starts[position]) / seconds_per_beat[position]
        )
    values = np.asarray(t_s, dtype=np.float64)
    index = _locate(values, seconds_starts)
    return np.asarray(
        starts[index] + (values - seconds_starts[index]) / seconds_per_beat[index],
        dtype=np.float64,
    )


def tau_bin_index(tau: float) -> int:
    """tau -> tau 格索引：i = floor(tau / d_tau)（plan §4.2）。

    浮点护栏：tau 恰好落在格界时允许 1e-9 格的正向偏移，避免 floor 把事件推到左格。
    **不做区间钳位**（越界由调用方按「不钳位、只计数」处理）。
    """
    return math.floor(float(tau) / TAU_GRID_DT + TAU_INDEX_EPS)


# ══════════════════════════════════════════════════════════════
# §3.1 网格
# ══════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class FieldGrid:
    """强度场网格：x 轴（RPE 舞台系）+ tau 轴（拍）+ 该谱的 BPMList。

    t_bins = 0 表示**尚未绑定时间轴**（只做几何断言）；经 with_time / for_chart
    绑定后即可给出测度 dV_j、|Omega| 与秒 <-> tau 换算。

    Attributes:
        x_bins: x 轴桶数（默认 128；消融见 X_BIN_SWEEP）。
        t_bins: tau 轴格数 T（由 BPMList 与谱面时长派生，**不得**由帧率派生）。
        bpm_points: 秒 <-> tau 换算的唯一依据（来自 PhigrosChart.bpm_points）。
    """

    x_bins: int = DEFAULT_X_BINS
    t_bins: int = 0
    bpm_points: tuple[BpmPoint, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.bpm_points, tuple):
            object.__setattr__(self, "bpm_points", tuple(self.bpm_points))

    # ── 几何（全部派生式）──────────────────────────────────────
    @property
    def dx(self) -> float:
        """x 桶宽 = RPE_STAGE_WIDTH / x_bins（**禁止写 10.546875**）。"""
        return RPE_STAGE_WIDTH / self.x_bins

    @property
    def d_tau(self) -> float:
        """tau 格宽（拍）= 1 / SUBDIVISIONS_PER_BEAT。"""
        return TAU_GRID_DT

    @property
    def x_min(self) -> float:
        """x 下界 = -半宽。"""
        return -RPE_STAGE_HALF_WIDTH

    @property
    def x_max(self) -> float:
        """x 上界 = +半宽。"""
        return +RPE_STAGE_HALF_WIDTH

    @property
    def sides(self) -> int:
        """S = len(Side) = 2。"""
        return N_SIDES

    @property
    def channels(self) -> int:
        """C = len(TYPE_CHANNELS) = 5。"""
        return N_CHANNELS

    @property
    def total_beats(self) -> float:
        """tau 轴总拍数 = T * d_tau（网格随 BPM 变化，拍数恒定）。"""
        return self.t_bins * self.d_tau

    @property
    def total_seconds(self) -> float:
        """tau 轴总秒数（**唯一的**秒域口径；下游报告一律回到秒域）。"""
        if not self.bpm_points or self.t_bins <= 0:
            return 0.0
        return float(tau_to_seconds(self.total_beats, self.bpm_points))

    def cell_count(self) -> int:
        """格元总数 K*T*X*S*C 口径下的单线格数 = T*X*S*C。"""
        return self.t_bins * self.x_bins * self.sides * self.channels

    def x_centers(self) -> FloatArray:
        """每个 x 桶的中心（RPE-x 单位）。"""
        return np.asarray(
            self.x_min + (np.arange(self.x_bins) + 0.5) * self.dx,
            dtype=np.float64,
        )

    def tau_edges(self) -> FloatArray:
        """tau 格边界（T+1 个，拍）。"""
        return np.asarray(np.arange(self.t_bins + 1) * self.d_tau, dtype=np.float64)

    def tau_centers(self) -> FloatArray:
        """tau 格中心（T 个，拍）。"""
        return np.asarray((np.arange(self.t_bins) + 0.5) * self.d_tau, dtype=np.float64)

    def range_mask(self) -> BoolArray:
        """RangeMask：(X,) 的 |x_center| <= 675。

        由于 dx * x_bins == RPE_STAGE_WIDTH **精确成立**，网格恰好铺满可见范围，
        本掩码结构上恒为 True——它不是钳位，只是「定义域恰好是可见范围」的声明。
        """
        return np.asarray(np.abs(self.x_centers()) <= RPE_STAGE_HALF_WIDTH, dtype=np.bool_)

    # ── 绑定时间轴 ────────────────────────────────────────────
    def with_time(self, t_bins: int, bpm_points: Sequence[BpmPoint]) -> FieldGrid:
        """返回绑定了 tau 轴的新网格（本类 frozen，不就地修改）。"""
        return FieldGrid(x_bins=self.x_bins, t_bins=int(t_bins), bpm_points=tuple(bpm_points))

    def tau_bins_for(self, duration_s: float, bpm_points: Sequence[BpmPoint] | None = None) -> int:
        """T = round(tau_end * BEAT_SUBDIVISION)，tau_end = seconds_to_tau(duration)（plan §3.2）。"""
        points = self.bpm_points if bpm_points is None else bpm_points
        if not points:
            raise ValueError("tau_bins_for 需要 bpm_points（tau 与秒换算的唯一依据）")
        tau_end = float(seconds_to_tau(float(duration_s), points))
        return math.floor(tau_end * BEAT_SUBDIVISION + 0.5)

    def for_chart(self, chart: PhigrosChart, *, tau_end_s: float | None = None) -> FieldGrid:
        """从谱面派生 tau 轴长度（终点口径见 plan §9-14，默认取 chart.duration_s()）。"""
        if not chart.bpm_points:
            raise ValueError("谱面必须带 BPMList（tau 与秒换算的唯一依据）")
        end_s = chart.duration_s() if tau_end_s is None else float(tau_end_s)
        return self.with_time(self.tau_bins_for(end_s, chart.bpm_points), chart.bpm_points)

    # ── 测度（唯一出口）──────────────────────────────────────
    def cell_seconds(self, *, rule: CellSecondsRule = "left") -> FloatArray:
        """每个 tau 格的时长（秒）：J_j * d_tau。

        - "left"（默认）：取**格左边界所在 BPM 段**的 J（plan §3.1 的 dV_j = J_j*d_tau*dx）；
        - "right"：取格右边界（左闭右开区间的另一侧）；
        - "exact"：精确值 = tau_to_seconds(tau_{t+1}) - tau_to_seconds(tau_t)，
          与 BPM 变更点是否落在格内无关。三者**仅在变更点落在格内时**不同
          （plan §9-16 未定的那一条；本模块默认 "left"，并在测试中量化差异）。
        """
        if self.t_bins <= 0:
            return np.zeros(0, dtype=np.float64)
        if not self.bpm_points:
            raise ValueError("cell_seconds 需要 bpm_points")
        edges = self.tau_edges()
        if rule == "exact":
            seconds = np.asarray(tau_to_seconds(edges, self.bpm_points), dtype=np.float64)
            return np.asarray(np.diff(seconds), dtype=np.float64)
        sample = edges[:-1] if rule == "left" else edges[1:]
        return np.asarray(
            jacobian_at(sample, self.bpm_points) * self.d_tau,
            dtype=np.float64,
        )

    def jacobian(
        self,
        bpm_points: Sequence[BpmPoint] | None = None,
        *,
        rule: CellSecondsRule = "left",
    ) -> float | FloatArray:
        """J(tau) = dt/dtau（秒/拍）；返回每格值（等价 cell_seconds / d_tau）或标量。

        t_bins == 0（未绑定时间轴）时返回 tau = 0 处的标量 J。
        """
        points = self.bpm_points if bpm_points is None else bpm_points
        if not points:
            raise ValueError("jacobian 需要 bpm_points")
        if self.t_bins <= 0:
            return jacobian_at(0.0, points)
        return np.asarray(self.cell_seconds(rule=rule) / self.d_tau, dtype=np.float64)

    def cell_volumes(
        self,
        bpm_points: Sequence[BpmPoint] | None = None,
        *,
        rule: CellSecondsRule = "left",
    ) -> FloatArray:
        """每格体积 dV_j = J_j * d_tau * dx（(T,)；**唯一的 dV 出口**）。

        tau 网格非均匀 => 逐格不同（plan §4.5）；漏乘 J_j 等价于把「拍」当「秒」用。
        """
        source = self if bpm_points is None else self.with_time(self.t_bins, bpm_points)
        return np.asarray(source.cell_seconds(rule=rule) * self.dx, dtype=np.float64)

    def volume(
        self,
        tau_bins: int | None = None,
        n_lines: int = 1,
        jacobian: float | FloatArray | None = None,
        *,
        rule: CellSecondsRule = "left",
    ) -> float:
        """|Omega| = sum_j dV_j，j 遍历 (k, tau, x, s, c) 全部格元（plan §2 偏离 2）。

        = n_lines * X * S * C * sum_t (J_t * d_tau) * dx

        **全 K 条线、全部格元计入**（含空线）；不得只对「有 note 的线」积分。
        """
        bins = self.t_bins if tau_bins is None else int(tau_bins)
        if bins < 0:
            raise ValueError(f"tau_bins 必须 >= 0，得到 {bins}")
        if jacobian is None:
            seconds = self.cell_seconds(rule=rule)[:bins]
        else:
            raw = np.asarray(jacobian, dtype=np.float64)
            if raw.ndim == 0:
                seconds = np.full(bins, float(raw) * self.d_tau, dtype=np.float64)
            else:
                seconds = np.asarray(raw[:bins] * self.d_tau, dtype=np.float64)
        # 逐格体积之和 * x 轴格元数 * 侧别数 * 通道数（x 是连续轴的离散化，贡献 dx 与 X 个格元）
        total = float(np.sum(seconds)) * self.dx * float(self.x_bins)
        return total * float(n_lines) * float(self.sides) * float(self.channels)

    # ── 契约 ─────────────────────────────────────────────────
    def spec(self, k: int) -> ChartFieldSpec:
        """构造并断言契约侧的 ChartFieldSpec（网格元数据的唯一携带者）。"""
        spec = ChartFieldSpec(
            k=k,
            t_bins=self.t_bins,
            d_tau=self.d_tau,
            bpm_points=self.bpm_points,
            x_bins=self.x_bins,
            dx=self.dx,
            x_min=self.x_min,
            x_max=self.x_max,
            sides=self.sides,
            channels=self.channels,
        )
        spec.assert_grid()
        return spec

    def assert_grid(self) -> None:
        """网格自洽断言；失败即抛（**不得降级为日志**）。"""
        if self.x_bins < 1:
            raise AssertionError(f"x_bins 必须 >= 1，得到 {self.x_bins}")
        if self.dx * self.x_bins != RPE_STAGE_WIDTH:
            raise AssertionError(
                f"dx * x_bins 必须等于 RPE_STAGE_WIDTH：{self.dx} * {self.x_bins}",
            )
        if self.x_min != -self.x_max:
            raise AssertionError(f"x 范围必须对称：[{self.x_min}, {self.x_max}]")
        if self.d_tau * BEAT_SUBDIVISION != 1.0:
            raise AssertionError(f"d_tau * BEAT_SUBDIVISION 必须等于 1：{self.d_tau}")
        if self.t_bins < 0:
            raise AssertionError(f"t_bins 必须 >= 0，得到 {self.t_bins}")
        if self.t_bins > 0 and not self.bpm_points:
            raise AssertionError("t_bins > 0 时必须给出 bpm_points（换算的唯一依据）")
        previous = -1.0
        for point in self.bpm_points:
            if point.time_beats < previous:
                raise AssertionError("bpm_points 必须按 time_beats 升序")
            previous = point.time_beats
        if self.sides != N_SIDES or self.channels != N_CHANNELS:
            raise AssertionError(f"(S, C) 必须等于 ({N_SIDES}, {N_CHANNELS})")


__all__ = [
    "BEAT_SUBDIVISION",
    "DEFAULT_X_BINS",
    "MERT_FRAME_RATE_HZ",
    "N_CHANNELS",
    "N_SIDES",
    "SECONDS_PER_MINUTE",
    "TAU_INDEX_EPS",
    "X_BIN_SWEEP",
    "BpmSegment",
    "CellSecondsRule",
    "FieldGrid",
    "bpm_segments",
    "jacobian_at",
    "seconds_to_tau",
    "tau_bin_index",
    "tau_to_seconds",
]
