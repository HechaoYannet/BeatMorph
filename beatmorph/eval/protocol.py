"""评估协议：配置、输入单元与口径常量（Plan 06 §3.1 / §4）。

时间口径（红线 7 / RFC-0029 §7-5 / §7-8）：

- 评估**一律在秒域**进行，所有时间字段名以 `_s` 结尾；
- 公共接口**不出现帧索引，也不出现 τ 格索引**；
- 本模块**不实现任何秒 <-> τ（拍）换算**——需要拍坐标（时间组分解）时一律调用
  `beatmorph.field.grid` 的权威接口。红线 7 明确：beat-aligned 的换算只在
  `field/` 内实现。

外部依赖：本模块只用 numpy 与契约（`core.contracts`）+ 解码事件契约
（`decoder.events.DecodedEvent`）。scipy **当前不是本仓依赖**（plan 06 §5），
因此统计辅助（rank / bootstrap）在 `beatmorph/eval/stats.py` 内自实现。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Final

from beatmorph.core.contracts.phigros import (
    RPE_STAGE_WIDTH,
    RPE_X_GRID_BINS,
    BpmPoint,
    PhigrosChart,
)
from beatmorph.decoder.events import DecodedEvent

# ══════════════════════════════════════════════════════════════
# 容差常量（**协议常量**，不是物理常量）
# ══════════════════════════════════════════════════════════════

#: DDC（ICML 2017）口径的 TP 判据容差；文献库 §7.1、plan 06 §3.1 / §4.1。
DDC_TOLERANCE_S: Final[float] = 0.020
#: GenéLive!（AAAI 2023）口径的 TP 判据容差；文献库 §7.1、plan 06 §3.1 / §4.1。
GENELIVE_TOLERANCE_S: Final[float] = 0.050
#: 双容差**必须同时报**（RFC-0029 §5.1、文献库 §8.4-1）。
DEFAULT_TOLERANCES_S: Final[tuple[float, float]] = (DDC_TOLERANCE_S, GENELIVE_TOLERANCE_S)

#: 项目内「毫秒」只作为**显示单位**（容差标签），与秒 <-> τ 换算无关（红线 7）。
MS_PER_S: Final[float] = 1000.0

#: 未提供 difficulty 时的难度档名（**不是**把 None 当成 0.0）。
UNKNOWN_DIFFICULTY_BAND: Final[str] = "unknown"
#: difficulty 比较前必须 round 到的位数（plan 06 §4.4-2：实测 14.900001 / 18.000004）。
DIFFICULTY_ROUND_DIGITS: Final[int] = 1


class AverageMode(StrEnum):
    """平均口径（plan 06 §3.1：`{per_chart, micro, both}`）。

    `per_chart` = 按谱平均（GenéLive! 的 F1-c，macro）；`micro` = 按事件合并计数。
    RFC-0029 §5.1 要求**两者都报**，因此报告默认 `both`。
    """

    PER_CHART = "per_chart"
    MICRO = "micro"
    BOTH = "both"


class DecodeRegime(StrEnum):
    """两栏报告的解码规则（plan 06 §3.1 / §4.2-4、文献库 §8.4-4）。

    `fixed` = 全库同一套固定解码规则；`per_chart_best` = 每谱按 F1 选最优参数。
    两个范式必须用**同一套报告格式**（否则「能调阈值的一方占便宜」）。
    """

    FIXED = "fixed"
    PER_CHART_BEST = "per_chart_best"


class TimeGroupRule(StrEnum):
    """时间组归组规则（plan 06 §4.4-1）。

    `position`（默认，plan 口径）：按事件的**拍位置**归入「最近 beat 细分」；
    `interval`：按与**同线前一事件**的拍间隔归入最近细分（首个事件退化为
    `position`）。两者都由 `field.grid.seconds_to_tau` 换算，**都不依赖
    1/48 拍的解码网格**（`breakdown.py` 的 docstring 给出间隔口径）。
    """

    POSITION = "position"
    INTERVAL = "interval"


@dataclass(frozen=True, slots=True)
class PhaseSearchConfig:
    """全谱相位偏移搜索的网格（plan 06 §3.1 / §4.3）。

    ⚠️ **范围与步长在 plan 06 §9-1 仍是开放问题**（ChartGenEval 用 0.5 ms 步长 +
    三档网格加权；我们是否照搬未决）。这里的默认值是**临时口径**，随报告一并冻结
    进 `meta`，因此任何数字都可追溯：`range_s = ±100 ms`（取 STRUM 的 onset
    窗口量级）、`step_s = 1 ms`（0.5 ms 会使搜索成本翻倍，收益未验证）。

    偏移网格为 `0, +step, -step, +2*step, -2*step, ...`（|offset| <= range）：
    **0 必在网格内**，因此「未注入偏移时搜索返回的 offset ≈ 0」是结构性结论
    （M6.2），而不是靠步长凑巧对齐。
    """

    enabled: bool = False
    range_s: float = 0.100
    step_s: float = 0.001

    def __post_init__(self) -> None:
        if self.range_s < 0.0:
            raise ValueError(f"phase_range_s 必须 >= 0，得到 {self.range_s!r}")
        if self.step_s <= 0.0:
            raise ValueError(f"phase_step_s 必须 > 0，得到 {self.step_s!r}")

    def offsets(self) -> tuple[float, ...]:
        """搜索网格（**确定性有序**：0 在前，其后按 |offset| 递增交替正负）。

        顺序即优先级：取最大 F1 时**首个最大值获胜**，从而在无注入偏移时
        倾向最小的 |offset|（M6.2 的 |offset| <= step 判据由此成立）。
        """
        count = int(self.range_s / self.step_s + 1e-9) if self.range_s > 0.0 else 0
        grid: list[float] = [0.0]
        for index in range(1, count + 1):
            grid.append(index * self.step_s)
            grid.append(-index * self.step_s)
        return tuple(grid)


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """一次评估的全部配置（plan 06 §3.1 的 EvalConfig + 实现所需的派生口径）。

    Attributes:
        tolerances_s: 时间容差（秒）；默认双容差 20ms / 50ms（DDC / GenéLive!）。
        primary_tolerance_s: **主容差**（主判据、分解、相位搜索与 MAE 等单读数指标）；
            `None` 表示取 `tolerances_s[0]`，且必须 ∈ `tolerances_s`。
        match_marking: 时间匹配是否要求标记一致（plan 06 §3.1）。**默认 False**：
            plan §4.2-4 要求 timing 族**不得**偷偷要求线号一致。置 True 时匹配候选集
            被限制为「标记也一致」的对，两族随之退化为一族（报告仍分列）。
        phase_search: 相位搜索配置（默认关闭；开启后报告仍必须同时给两组数）。
        average: 平均口径；报告默认两栏都出（`both`）。
        decode_regime: 两栏报告的默认栏位（`fixed` 或 `per_chart_best`）。
        x_bins: 解码侧的 x 桶数。**只用于派生** dx 与量化下界（plan 06 §4.5：
            dx = RPE_STAGE_WIDTH / x_bins，下界 = dx / 4）；评估本身**不重算网格**，
            只为「MAE 的改善可能只是网格变细」提供参照。
        position_x_tolerance: 判定 positionX 一致（event 族）的容差（RPE 单位）；
            `None` = `dx`（同一个桶或相邻桶内视为一致）。plan §4.1 只写「一致」，
            未给容差口径 → 本实现显式声明并允许覆盖（见 plan 06 §9 存疑清单）。
        time_group_rule: 时间组归组规则，见 TimeGroupRule。
        bootstrap_resamples / bootstrap_seed: song-cluster bootstrap 的重采样数与
            种子（**必须固定**，否则报告不可复现）。cluster 定义在 plan 06 §9-2 未定，
            本实现按曲目标识聚类（`EvalCase.cluster`）。
    """

    tolerances_s: tuple[float, ...] = DEFAULT_TOLERANCES_S
    primary_tolerance_s: float | None = None
    match_marking: bool = False
    phase_search: PhaseSearchConfig = field(default_factory=PhaseSearchConfig)
    average: AverageMode = AverageMode.BOTH
    decode_regime: DecodeRegime = DecodeRegime.FIXED
    x_bins: int = RPE_X_GRID_BINS
    position_x_tolerance: float | None = None
    time_group_rule: TimeGroupRule = TimeGroupRule.POSITION
    bootstrap_resamples: int = 200
    bootstrap_seed: int = 0

    def __post_init__(self) -> None:
        if not self.tolerances_s:
            raise ValueError("tolerances_s 不得为空（至少一档容差）")
        if any(tol <= 0.0 for tol in self.tolerances_s):
            raise ValueError(f"容差必须为正（秒），得到 {self.tolerances_s!r}")
        if tuple(sorted(self.tolerances_s)) != tuple(self.tolerances_s):
            raise ValueError(f"tolerances_s 必须按升序给出，得到 {self.tolerances_s!r}")
        if len(set(self.tolerances_s)) != len(self.tolerances_s):
            raise ValueError(f"tolerances_s 不得重复，得到 {self.tolerances_s!r}")
        if (
            self.primary_tolerance_s is not None
            and self.primary_tolerance_s not in self.tolerances_s
        ):
            raise ValueError(
                f"主容差 {self.primary_tolerance_s!r} 必须属于 tolerances_s="
                f"{self.tolerances_s!r}（否则报告里没有对应的一格）",
            )
        if self.x_bins < 1:
            raise ValueError(f"x_bins 必须 >= 1，得到 {self.x_bins!r}")
        if self.position_x_tolerance is not None and self.position_x_tolerance < 0.0:
            raise ValueError(f"position_x_tolerance 必须 >= 0，得到 {self.position_x_tolerance!r}")
        if self.bootstrap_resamples < 1:
            raise ValueError(f"bootstrap_resamples 必须 >= 1，得到 {self.bootstrap_resamples!r}")

    @property
    def dx(self) -> float:
        """x 桶宽 = RPE_STAGE_WIDTH / x_bins（**不写 10.546875**）。"""
        return RPE_STAGE_WIDTH / self.x_bins

    @property
    def primary_tolerance(self) -> float:
        """主容差（秒）。"""
        return (
            self.tolerances_s[0] if self.primary_tolerance_s is None else self.primary_tolerance_s
        )

    @property
    def position_tolerance(self) -> float:
        """positionX 一致判据的容差（RPE 单位）：默认一个桶宽。"""
        return self.dx if self.position_x_tolerance is None else self.position_x_tolerance

    @property
    def quantization_lower_bound(self) -> float:
        """positionX MAE 的量化下界 = dx / 4（plan 06 §4.5 的推导）。

        解码输出被吸附到桶中心、真值在桶内近似均匀时，期望绝对误差即 dx / 4。
        该下界**随 x_bins 变化** → 必须与 MAE 并列报出，否则 MAE 的改善可能只是
        网格变细（plan §4.5 末段、单位几何文档 §7.3 的 N 消融）。
        """
        return self.dx / 4.0

    def tolerance_key(self, tolerance_s: float) -> str:
        """容差 -> 稳定标签（如 20ms）；报告与日志的键名唯一来源。"""
        return f"{tolerance_s * MS_PER_S:g}ms"

    def tolerance_keys(self) -> dict[float, str]:
        """全部容差 -> 标签（键为秒值，值为标签）。"""
        return {tol: self.tolerance_key(tol) for tol in self.tolerances_s}

    def with_phase_search(self, *, enabled: bool) -> EvalConfig:
        """返回相位搜索开关被改写的新配置（本类 frozen）。"""
        return replace(self, phase_search=replace(self.phase_search, enabled=enabled))


@dataclass(frozen=True, slots=True)
class EvalCase:
    """一次评估的最小输入单元：**同一首曲**的生成事件与人类事件（秒域）。

    Attributes:
        key: 谱面标识（报告主键）。
        pred: 生成侧事件（decoder.events.DecodedEvent，Plan 05 的产物）。
        gold: 人类谱事件（同一契约；由 PhigrosChart 经
            decoder.pipeline.events_from_chart 转换，**秒域**）。
        bpm_points: 秒 <-> 拍换算的唯一依据（时间组分解需要）。来自 **gold** 谱面
            ——人类谱的 BPMList 是时间轴权威；两谱 BPMList 不一致属数据问题，不在
            评估内静默对齐。
        difficulty: info.yml 的 f32 定数（比较前 round 到 0.1）；`None` = 未提供。
        level_text: 自由文本等级。**只记录，绝不解析**（plan 06 §4.4-2：实测出现
            "AT  Lv.16" / "sweet" / "酔い"）。
        song: 曲目标识 —— bootstrap 的聚类单元（plan 06 §9-2 未定，默认同 key）。
    """

    key: str
    pred: tuple[DecodedEvent, ...]
    gold: tuple[DecodedEvent, ...]
    bpm_points: tuple[BpmPoint, ...]
    difficulty: float | None = None
    level_text: str = ""
    song: str = ""

    def __post_init__(self) -> None:
        if not self.bpm_points:
            raise ValueError(f"EvalCase {self.key!r} 必须带 bpm_points（秒 <-> 拍换算的唯一依据）")

    @property
    def cluster(self) -> str:
        """bootstrap 的 cluster 键（同曲不同难度谱默认同一 cluster）。"""
        return self.song or self.key

    def with_pred(self, events: Sequence[DecodedEvent]) -> EvalCase:
        """替换生成侧事件（gold / BPM / 元数据不变）——corruption 注入的入口。"""
        return replace(self, pred=tuple(events))

    @classmethod
    def from_charts(
        cls,
        key: str,
        pred_chart: PhigrosChart,
        gold_chart: PhigrosChart,
        *,
        song: str = "",
    ) -> EvalCase:
        """PhigrosChart + PhigrosChart -> EvalCase（秒域事件视图）。

        difficulty / level_text 取自 **gold** 谱面元数据；bpm_points 同样取 gold
        （人类谱是时间轴权威）。
        """
        from beatmorph.decoder.pipeline import events_from_chart

        return cls(
            key=key,
            pred=tuple(events_from_chart(pred_chart)),
            gold=tuple(events_from_chart(gold_chart)),
            bpm_points=tuple(gold_chart.bpm_points),
            difficulty=gold_chart.meta.difficulty,
            level_text=gold_chart.meta.level_text,
            song=song,
        )


__all__ = [
    "DDC_TOLERANCE_S",
    "DEFAULT_TOLERANCES_S",
    "DIFFICULTY_ROUND_DIGITS",
    "GENELIVE_TOLERANCE_S",
    "MS_PER_S",
    "UNKNOWN_DIFFICULTY_BAND",
    "AverageMode",
    "DecodeRegime",
    "EvalCase",
    "EvalConfig",
    "PhaseSearchConfig",
    "TimeGroupRule",
]
