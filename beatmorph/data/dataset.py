"""窗口数据集：RPEJSON 谱面 + MERT 特征缓存 -> `PairSample` -> `FieldBatch`（plan 02 / plan 04 §3.2）。

这是**数据侧与 generation 主干之间的真实接缝**：本模块把「谱面文件 + 特征缓存」
变成 :class:`~beatmorph.generation.batch.FieldBatch` 可直接消费的张量。所有物理换算
都走既有实现，本模块**不新增任何换算**：

| 量 | 唯一来源 |
|---|---|
| 秒 <-> tau | :func:`beatmorph.field.grid.seconds_to_tau` / `tau_to_seconds` |
| tau 网格与 T | `FieldGrid.for_chart(chart, tau_end_s=...)` |
| 桶内计数 | :func:`beatmorph.field.target.build_target` |
| 普通事件轨 | :mod:`beatmorph.data.tracks`（跨层求和由契约 `JudgeLine.sum_track` 负责） |
| 音频帧率 | `MERT_FRAME_RATE_HZ`（契约派生量，禁止字面量） |
| 遮盖通道 | :func:`beatmorph.generation.masks.build_occlusion_batch` |

四条硬约束（都会被断言、抛错或**显式记账**，不静默降级）：

1. **窗口不得切断 Hold**：窗口边界按「配对起点格 s / 终点格 e」的阻塞区间选点
   （:func:`_hold_blocked_boundaries`），必要时把窗口起点前移；移不动就**跳过并记账**。
2. **同一 batch 必须同一网格身份**（`x_bins` / `t_bins` / `bpm_points` 完全一致）：
   `FieldBatch` 只携带**一个** :class:`~beatmorph.field.grid.FieldGrid`，而测度 `J(tau)`
   与积分项都由它派生——混批会让两者静默错掉。不一致时抛 :class:`GridMismatchError`。
   本模块进一步让该身份**足以决定 J 向量**（见下），因此“身份相同”是 J 相同的充分条件。
3. **确定性**：`(config, index)` 两次取值逐位一致；遮盖种子由 index 派生，
   不依赖任何全局随机状态（不用 `hash()`——它按进程随机化）。
4. **绝不发出 `r == 1` 的样本**：`masked_poisson_loss` 在 `r == 1` 时**拒绝训练**
   （「全部事件都被遮盖，事件项没有可见上下文」）——而稀疏窗口上「遮盖单位恰好覆盖全部事件」
   是真实数据必然遇到的情况（安静段落）。因此窗口规划先排除**与种子无关**的不可行窗口
   （事件 token 数 == 1，跳过并计数），`__getitem__` 再对残余情形重掷种子
   （:data:`MAX_OCCLUSION_SEED_ATTEMPTS` 次）；仍然全遮时退化为无遮盖（`r == 0`，
   losses 明文支持的分支）并记账告警，**不跳过、不静默**。

**窗口局部网格（tau 起点与 J 的口径）**
`PairSample.grid` 的 tau 轴是**窗口局部**的（`0 .. t_window * d_tau`）；它的
`bpm_points` 是单段列表 `[(0, bpm_eff)]`，其中 `bpm_eff = 60 / J(tau_start)` 由
:func:`beatmorph.field.grid.jacobian_at` 派生（等价于「把谱面 BPMList 平移到窗口起点
并剪裁到窗口内的段」）。于是 `grid.cell_volumes()` 恰好给出**该窗口**的逐格体积，
且同一 BPM 段内的任意窗口网格身份相同 => 可同批、且同批的逐格 J 完全相同。
**前提**：窗口必须落在单个 BPM 段内（否则 J 在窗内变化，单段网格会算错），
因此窗口规划把 BPM 变更点当作与 Hold 同级的阻塞条件（同样记账）。
绝对 tau 位置由 :attr:`PairSample.tau_start` 携带；`line_tracks` 在
`tau_start + grid.tau_centers()` 上求值（事件轨是**绝对**拍时刻的函数）。

**音频轴**：窗口的秒区间由**绝对** tau 经 `tau_to_seconds` 与 `MERT_FRAME_RATE_HZ`
派生后切片；批内音频长度不一致时由 :func:`collate_field_batch` 补零到批内最长
（`FieldBatch` 没有 `time_mask` 字段，这是当前契约的已知缺口）。

**存疑（需主会话裁定，未擅自改契约）**

1. 批内音频长度不一致时只能**补零**（`FieldBatch` 无 `time_mask`），补零帧是伪造的
   静音特征；若主会话要给音频 padding 一个显式掩码，需先改 `generation/batch.py` 契约。
2. 窗口局部网格把 `bpm_points` 收成**单段**（J 在窗内恒定、可同批）。代价是跨 BPM 变更点
   的窗口被跳过并计数；若希望「整谱 BPMList 进 grid、允许同谱跨段同批」，则同批样本的
   `J` 向量会**不再相同**，需要主会话先裁定测度口径。
3. 事件 token 数 == 1 的窗口（稀疏段落）**跳过并计数**——它无论如何都无法满足 `r < 1`。
   `losses.py` 的 `r == 0` 分支明文支持「无遮盖 => 纯密度拟合」，因此另一种口径是把这些
   窗口以 `occlusion` 全 False 发出（不丢数据）；本实现按主会话口径取**跳过 + 记账**。
"""

from __future__ import annotations

import json
import zlib
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from beatmorph.core.contracts import (
    RPE_STAGE_HALF_WIDTH,
    RPE_X_GRID_BINS,
    TAU_GRID_DT,
    BpmPoint,
    ChartFieldSpec,
    ChartFormat,
    ChartSource,
    NoteType,
    PhigrosChart,
    PhigrosNote,
    side_from_above,
    side_index,
    x_bin_index,
)
from beatmorph.core.contracts.tensors import MERT_FRAME_RATE_HZ
from beatmorph.core.logging import get_logger
from beatmorph.data.parsers.rpejson import RpeParseError, parse_rpejson
from beatmorph.data.parsers.sniff import sniff_format_with_evidence
from beatmorph.data.pipeline.embed import (
    SPLIT_NAMES,
    PairRow,
    SplitError,
    feature_cache_paths,
    load_feature_cache,
)
from beatmorph.data.tracks import N_TRACK_CHANNELS, line_tracks_at
from beatmorph.field.grid import (
    SECONDS_PER_MINUTE,
    FieldGrid,
    bpm_segments,
    jacobian_at,
    seconds_to_tau,
    tau_bin_index,
    tau_to_seconds,
)
from beatmorph.field.target import CHANNEL_INDEX, HOLD_END_CHANNEL, build_target
from beatmorph.generation.batch import FieldBatch
from beatmorph.generation.masks import (
    assert_hold_pairs_not_split,
    build_occlusion_batch,
    occluded_event_share,
)
from beatmorph.generation.model import DEFAULT_K_MAX

logger = get_logger(__name__)

#: 运行期**动态**输入的类型别名（JSON 值、numpy/torch 张量）。
#:
#: 为什么不是 `object`：这些位置会对任意 JSON 值做 `int()`/`float()` 或形状访问，
#: 写成 `object` 只会把类型错误推到运行期并且误述真实宽容度；而 flake8-annotations
#: 的 ANN401 不允许把 `Any` 直接写进签名。此处集中别名一次并说明理由（同 embed.py）。
Dynamic = Any

__all__ = [
    "ChartPairDataset",
    "DatasetConfig",
    "DatasetIndexStats",
    "DatasetManifestError",
    "GridMismatchError",
    "PairSample",
    "collate_field_batch",
    "load_pairs",
]


class GridMismatchError(RuntimeError):
    """同批样本的网格身份不一致（`x_bins` / `t_bins` / `bpm_points` 必须完全相同）。

    `FieldBatch` 只携带**一个**网格，而测度 `J(tau)` 与积分项 `sum lam * J * d_tau * dx`
    都由它派生：把不同网格的样本拼在一起，loss 会**静默算错**而不报任何错。
    因此本异常是硬边界，**不得**用「取第一个样本的网格」之类的降级方式绕过。
    """


class DatasetManifestError(ValueError):
    """清单 JSON 结构不合法（不是 `PairSplits.to_dict()` 的形状 / 行缺字段）。"""


#: 遮盖种子重掷上限：`build_occlusion_batch` 选中的遮盖单位集可能恰好覆盖**全部**事件，
#: 此时窗口的 r == 1，而 :func:`beatmorph.generation.losses.masked_poisson_loss` 会**拒绝**它
#: （「全部事件都被遮盖，事件项没有可见上下文」）。数据集是遮盖的产出方，因此必须保证
#: **绝不发出 r == 1 的样本**：先用派生种子试一次，不行就重掷；重掷上限见本常量。
MAX_OCCLUSION_SEED_ATTEMPTS: int = 8

#: 重掷时种子序列的步长（任意奇数常量，只用于让候选种子在 PCG 流上散开）。
_OCCLUSION_SEED_STEP: int = 0x9E3779B1
#: 种子取模（`np.random.default_rng` 接受任意非负整数，这里统一到 32 位便于留痕）。
_SEED_MODULUS: int = 1 << 32


def _window_seed(seed: int, row_index: int, window_index: int) -> int:
    """由 `(config.seed, 行号, 窗口序号)` 派生的确定性遮盖种子。

    不用内建 `hash()`：字符串哈希受 `PYTHONHASHSEED` 影响，会让「同一 index 两次取值
    逐位一致」的承诺在**跨进程**时失效（同一进程内也会随重启变化）。
    """
    key = f"{int(seed)}:{int(row_index)}:{int(window_index)}".encode()
    return int(zlib.crc32(key))


def _effective_bpm(bpm_points: Sequence[BpmPoint], tau_start: float) -> float:
    """窗口起点的等效 BPM = `60 / J(tau_start)`（J 由 field 派生，本模块不重写换算）。"""
    return SECONDS_PER_MINUTE / float(jacobian_at(tau_start, bpm_points))


def _note_bins(
    note: PhigrosNote,
    bpm_points: Sequence[BpmPoint],
    t_bins: int,
) -> tuple[int | None, int | None]:
    """一个 note 会进桶的 `(起点格, 终点格)`——**与 :func:`build_target` 的入账口径逐条对齐**。

    返回 `None` 表示该端点不进桶：fake 音符、`|positionX| > 半宽`（红线 3：越界只统计不钳位）、
    非 Hold 的终点、`hold_time <= 0` 的畸形 Hold、以及落在 tau 轴之外的端点
    （起点越界时 build_target 整条跳过，因此终点也不计）。
    """
    if note.is_fake or abs(note.position_x) > RPE_STAGE_HALF_WIDTH:
        return None, None
    start = tau_bin_index(float(seconds_to_tau(note.t, bpm_points)))
    if not 0 <= start < t_bins:
        return None, None
    if note.type is not NoteType.HOLD or note.hold_time <= 0.0:
        return start, None
    end = tau_bin_index(float(seconds_to_tau(note.t + note.hold_time, bpm_points)))
    return start, (end if 0 <= end < t_bins else None)


def _hold_blocked_boundaries(
    chart: PhigrosChart,
    bpm_points: Sequence[BpmPoint],
    t_bins: int,
) -> NDArray[np.bool_]:
    """边界阻塞表：`blocked[b]` = 存在 hold 配对 `(s, e)` 使 `s < b <= e`（b 是格边界，0..T）。

    为什么盯**边界**而不是窗口：窗口 `[a, a+W)` 恰好切断某个配对，当且仅当
    「起点在内、终点在外」（`a+W` 落在 `(s, e]`）或「终点在内、起点在外」
    （`a` 落在 `(s, e]`）。于是只要窗口**两端边界都不被阻塞**，窗口就装得下所有配对。

    入账口径见 :func:`_note_bins`（build_target 是计数的唯一来源，本表只是它的**边界视图**，
    用差分数组一次算出，复杂度 O(T + 配对数)）。只有**两端都进桶**的 Hold 才构成配对：
    终点越 tau 轴的 Hold 在 counts 里只有一个起点（unpaired），
    :func:`assert_hold_pairs_not_split` 也不约束它。
    """
    blocked = np.zeros(t_bins + 1, dtype=np.bool_)
    diff = np.zeros(t_bins + 2, dtype=np.int64)
    n_pairs = 0
    for note in chart.sorted_notes():
        start, end = _note_bins(note, bpm_points, t_bins)
        if start is None or end is None:
            continue
        diff[start + 1] += 1
        diff[end + 1] -= 1
        n_pairs += 1
    if n_pairs:
        running = np.cumsum(diff[: t_bins + 1])
        blocked = np.asarray(running > 0, dtype=np.bool_)
    return blocked


@dataclass(frozen=True, slots=True)
class _EventPoint:
    """一个会进桶的事件点（索引期的轻量视图，口径见 :func:`_note_bins`）。

    Attributes:
        bin: tau 格索引。
        cell: 桶内格元键 `(x_bin, side, channel)`（与该线索引一起唯一确定一个格）。
        pair_id: 所属 hold 配对的编号；`-1` = 独立事件点（Tap / Flick / Drag /
            终点越 tau 轴的 Hold 起点——后者在 counts 里也没有配对）。
        partner_bin: 配对另一端的 tau 格；`-1` = 无配对。
    """

    bin: int
    cell: tuple[int, int, int]
    pair_id: int
    partner_bin: int


def _event_points(
    chart: PhigrosChart,
    bpm_points: Sequence[BpmPoint],
    t_bins: int,
    spec: ChartFieldSpec,
) -> list[list[_EventPoint]]:
    """逐线的**事件点**（按 tau 格升序），供窗口规划判断可遮盖性。

    用途：`build_occlusion` 的遮盖单位是「一个 hold 配对」或「一个事件格」，
    并按 token 区块扩张。若一个窗口内的**遮盖单位数 <= 1**，选择过程（ratio > 0 时至少选一个）
    必然把全部事件遮掉 => `r == 1`——正是 `masked_poisson_loss` 拒绝训练的情形
    （与种子无关，重掷无用）。该判断只看事件位置，不需要构造 `(K, T, X, S, C)`
    计数张量，故可在索引期廉价完成（复杂度 O(事件数 log 事件数)）。
    """
    rows: list[list[_EventPoint]] = [[] for _ in range(len(chart.lines))]
    pair_id = 0
    for note in chart.sorted_notes():
        start, end = _note_bins(note, bpm_points, t_bins)
        if start is None:
            continue
        x_index = x_bin_index(note.position_x, spec)
        if x_index is None:  # pragma: no cover - _note_bins 已按半宽排除
            continue
        side = side_index(side_from_above(note.above_raw))
        channel = CHANNEL_INDEX[note.type]
        if end is None:
            rows[note.line_id].append(_EventPoint(start, (x_index, side, channel), -1, -1))
            continue
        rows[note.line_id].append(_EventPoint(start, (x_index, side, channel), pair_id, end))
        rows[note.line_id].append(
            _EventPoint(end, (x_index, side, HOLD_END_CHANNEL), pair_id, start),
        )
        pair_id += 1
    for row in rows:
        row.sort(key=lambda point: point.bin)
    return rows


def _window_event_shape(
    rows: Sequence[Sequence[_EventPoint]],
    start: int,
    width: int,
) -> tuple[int, int, int]:
    """窗口 `[start, start+width)` 的 `(遮盖单位数, 事件 token 数, 事件数)`。

    - **遮盖单位**（与 `build_occlusion` 的口径对齐）= 一个**两端都在窗内**的 hold 配对，
      或一个独立的事件格；跨窗的配对退化成两个独立事件格（保守估计——窗口规划本身禁止配对跨窗）。
    - **事件 token** = `(线, tau 格)` 对：遮盖经 `expand_to_tokens` 扩张到整个 token，
      因此「窗内只有 1 个事件 token」时**任何**遮盖选择都会把全部事件遮掉。

    复杂度 O(窗内事件数)。
    """
    end = start + width
    pairs: set[int] = set()
    cells: set[tuple[int, int, int, int]] = set()
    tokens: set[tuple[int, int]] = set()
    n_events = 0
    for line_index, row in enumerate(rows):
        low = bisect_left(row, start, key=lambda point: point.bin)
        high = bisect_left(row, end, key=lambda point: point.bin)
        for point in row[low:high]:
            n_events += 1
            tokens.add((line_index, point.bin))
            if point.pair_id >= 0 and start <= point.partner_bin < end:
                pairs.add(point.pair_id)
            else:
                cells.add((point.bin, *point.cell))
    return len(pairs) + len(cells), len(tokens), n_events


def _bpm_change_points(bpm_points: Sequence[BpmPoint]) -> NDArray[np.float64]:
    """`J(tau)` 真正跳变的 tau 列表（经 field 的 :func:`bpm_segments` 判定）。

    `bpm_segments` 会在首段前补一个外推段（相同 BPM），因此只把**相邻两段 BPM 不同**
    的段起点算作变更点：BPM 相同但写法上分了段的谱面不该被误判为跨段。
    """
    changes = [
        float(current.tau_start)
        for previous, current in pairwise(bpm_segments(bpm_points))
        if current.bpm != previous.bpm
    ]
    return np.asarray(changes, dtype=np.float64)


def _window_is_safe(
    start: int,
    width: int,
    blocked: NDArray[np.bool_],
    changes: NDArray[np.float64],
) -> bool:
    """窗口 `[start, start+width)` 是否既不切断 Hold、也不跨 BPM 变更点。"""
    if bool(blocked[start]) or bool(blocked[start + width]):
        return False
    low = start * TAU_GRID_DT
    high = (start + width) * TAU_GRID_DT
    index = int(np.searchsorted(changes, low, side="right"))
    return index >= int(changes.size) or float(changes[index]) >= high


def _window_grid(
    x_bins: int,
    width: int,
    bpm_points: Sequence[BpmPoint],
    tau_start: float,
) -> FieldGrid:
    """窗口局部网格：`t_bins = width`，`bpm_points` 是**单段**的窗口等效 BPM（见模块 docstring）。

    单段化是「同批样本的 J 向量逐格相同」的充分条件（`collate_field_batch` 靠网格身份
    保证测度正确），前提是窗口 BPM 纯（窗口规划强制）。
    """
    return FieldGrid(
        x_bins=x_bins,
        t_bins=width,
        bpm_points=(BpmPoint(time_beats=0.0, bpm=_effective_bpm(bpm_points, tau_start)),),
    )


def _window_subchart(
    chart: PhigrosChart,
    grid: FieldGrid,
    start: int,
    width: int,
    *,
    bpm_eff: float,
) -> PhigrosChart:
    """把谱面裁成**窗口子谱**：只保留起点格落在窗口内的 note，时间平移到窗口起点。

    为什么必须这么做：:func:`build_target` 的 tau 索引是**全谱绝对**的，对全谱调用它会分配
    `(K, T_full, X, S, C)` 的计数张量——真实谱面（5 分钟 / K=30 / X=128）约 1e9 格 ≈ 2 GB
    **每个样本**，训练不可用。窗口是 BPM 纯的（窗口规划强制），窗内「秒 -> tau」是线性的，
    因此「时间平移 + 窗口网格上的 build_target」与「全谱 build_target 后切窗」**逐格等价**；
    该等价性由 `tests/unit/data/test_dataset.py::test_window_counts_equal_full_chart_slice` 锁定。

    选点口径与 :func:`_note_bins` 一致（= build_target 的入账口径）。Hold 的两端不会跨窗
    （窗口规划已禁止），越出 tau 轴的端点由 build_target 自行处理，这里不额外改动。
    """
    t_bins = grid.t_bins
    origin_s = float(tau_to_seconds(start * TAU_GRID_DT, grid.bpm_points))
    notes: list[PhigrosNote] = []
    for note in chart.notes:
        note_start, note_end = _note_bins(note, grid.bpm_points, t_bins)
        if note_start is None or not start <= note_start < start + width:
            continue
        if note_end is not None and not start <= note_end < start + width:
            raise AssertionError(
                "Hold 配对跨窗：窗口规划与子谱裁剪的口径不一致"
                f"（line_id={note.line_id}，格 {note_start} -> {note_end}，窗口 [{start}, "
                f"{start + width})）",
            )
        notes.append(note.model_copy(update={"t": note.t - origin_s}))
    return chart.model_copy(
        update={
            "notes": notes,
            "bpm_points": [BpmPoint(time_beats=0.0, bpm=bpm_eff)],
        },
    )


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """数据集配置（**全部字段都是输入**，不含任何运行期状态）。

    Attributes:
        manifest_path: 清单 JSON（形状 == :meth:`~beatmorph.data.pipeline.embed.PairSplits.to_dict`）。
        chart_dir: 谱面落盘根；行内 `chart_path` 相对它解析。
        feature_dir: 特征缓存根；行内 `feature_key` 经
            :func:`~beatmorph.data.pipeline.embed.feature_cache_paths` 定位。
        t_window: tau 轴窗口长度（格）；一个样本 = 一个窗口。
        split: `"train"` / `"val"` / `"test"`。
        tau_end_s: tau 轴终点（秒）；None = 谱面自带口径
            （`FieldGrid.for_chart` 的默认，plan 03 §9-14 未裁定）。
        x_bins: x 轴桶数（默认契约值 `RPE_X_GRID_BINS`）。
        k_max: 判定线容量（默认 `generation.model.DEFAULT_K_MAX`）；超过它的谱面在索引期**跳过并记账**。
        occlusion_ratio: 每个窗口独立的遮盖比例 r（按**事件**计，见 generation.masks）。
        seed: 遮盖种子的根（实际种子由 `(seed, 行号, 窗口序号)` 派生）。
        limit: 只取 split 的前 N 行（调试用；None = 全部）。

    Note:
        `split` 是 **keyword-only**：规格把它写在 `t_window` 之前且带默认值，
        而 dataclass 不允许「有默认值的字段」后面跟「无默认值的字段」——把 `split`
        标成 kw_only 既保住了声明顺序，也让三个必填路径参数可以按位置传。
    """

    manifest_path: Path
    chart_dir: Path
    feature_dir: Path
    t_window: int
    split: str = field(default="train", kw_only=True)
    tau_end_s: float | None = None
    x_bins: int = RPE_X_GRID_BINS
    k_max: int = DEFAULT_K_MAX
    occlusion_ratio: float = 0.5
    seed: int = 0
    limit: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest_path", Path(self.manifest_path))
        object.__setattr__(self, "chart_dir", Path(self.chart_dir))
        object.__setattr__(self, "feature_dir", Path(self.feature_dir))
        if self.split not in SPLIT_NAMES:
            raise SplitError(f"split 必须是 {SPLIT_NAMES} 之一，得到 {self.split!r}")
        if self.t_window < 1:
            raise ValueError(f"t_window 必须 >= 1，得到 {self.t_window}")
        if self.x_bins < 1:
            raise ValueError(f"x_bins 必须 >= 1，得到 {self.x_bins}")
        if self.k_max < 1:
            raise ValueError(f"k_max 必须 >= 1，得到 {self.k_max}")
        if not 0.0 <= self.occlusion_ratio <= 1.0:
            raise ValueError(f"occlusion_ratio 必须在 [0, 1]，得到 {self.occlusion_ratio}")
        if self.limit is not None and self.limit < 0:
            raise ValueError(f"limit 必须 >= 0 或 None，得到 {self.limit}")
        if self.tau_end_s is not None and self.tau_end_s <= 0.0:
            raise ValueError(f"tau_end_s 必须为正或 None，得到 {self.tau_end_s}")


@dataclass(frozen=True, slots=True)
class PairSample:
    """一个**窗口**样本：条件（音频 + 事件轨 + 定数）+ 目标（计数 + 遮盖）。

    Attributes:
        audio_emb: `(T_audio_win, D)` float32；MERT 特征在窗口秒区间上的切片。
        frame_rate: 音频帧率 Hz（== `MERT_FRAME_RATE_HZ`，契约派生量）。
        line_tracks: `(K, T_window, N_ORDINARY_TRACKS)` float32；跨层求和后的 5 条普通事件轨，
            在 `tau_start + grid.tau_centers()` 上求值。
        line_mask: `(K,)` bool；本样本真实存在的线（**padding 在 collate 里做**）。
        difficulty: `()` float32 标量定数（不是 level 字符串）。
        grid: **窗口局部**网格（tau 轴 `0 .. T_window`；`bpm_points` 见模块 docstring）。
        counts: `(K, T_window, X, S, C)` int16 桶内计数（build_target 的窗口切片）。
        occlusion: `(K, T_window, X, S, C)` bool；**True = 被遮盖（待补全）**。
        chart_id / song_key / split / chart_path / feature_key: 可追溯字段（诊断与合规留痕）。
        pair_index: 该行在 split 内的下标（`Dataset.rows()[pair_index]`）。
        window_index: 该窗口在本谱窗口列表中的序号（0-based，按 tau 升序）。
        tau_start: 窗口起点的**绝对** tau（拍）= `tau_start_bins * d_tau`。
        audio_frame_start: 窗口起点在特征缓存里的帧下标（诊断用）。
        audio_padded_frames: 因特征缓存不足而补零的帧数（**不为 0 就是数据缺口**）。
    """

    audio_emb: Tensor
    frame_rate: float
    line_tracks: Tensor
    line_mask: Tensor
    difficulty: Tensor
    grid: FieldGrid
    counts: Tensor
    occlusion: Tensor
    chart_id: int | None
    song_key: str
    split: str
    pair_index: int
    window_index: int
    tau_start: float
    chart_path: str
    feature_key: str | None = None
    audio_frame_start: int = 0
    audio_padded_frames: int = 0


@dataclass(frozen=True, slots=True)
class DatasetIndexStats:
    """索引构建的记账（**没有静默丢弃**：每一类跳过都有计数器）。"""

    n_rows: int
    n_rows_used: int
    n_windows: int
    skipped_chart_missing: int
    skipped_no_feature: int
    skipped_no_difficulty: int
    skipped_parse_error: int
    skipped_too_many_lines: int
    skipped_no_windows: int
    skipped_windows_hold_split: int
    skipped_windows_bpm_crossing: int
    skipped_windows_no_visible_context: int
    dropped_events_no_visible_context: int
    shifted_windows: int
    dropped_tail_bins: int
    skipped_format: dict[str, int] = field(default_factory=dict)

    def describe(self) -> str:
        """一行诊断文本（训练日志用；不含任何权重）。"""
        formats = "、".join(
            f"{name}={count}" for name, count in sorted(self.skipped_format.items())
        )
        return (
            f"数据集索引：行 {self.n_rows_used}/{self.n_rows}，窗口 {self.n_windows} | "
            f"跳过行：谱面缺失 {self.skipped_chart_missing}、无特征 {self.skipped_no_feature}、"
            f"无定数 {self.skipped_no_difficulty}、非 RPE [{formats or '无'}]、"
            f"解析失败 {self.skipped_parse_error}、线数超 k_max {self.skipped_too_many_lines}、"
            f"无可用窗口 {self.skipped_no_windows} | "
            f"跳过窗口：Hold 切断 {self.skipped_windows_hold_split}、"
            f"跨 BPM 变更点 {self.skipped_windows_bpm_crossing}、"
            f"无法满足 r<1 {self.skipped_windows_no_visible_context}"
            f"（连带 {self.dropped_events_no_visible_context} 个事件点） | "
            f"前移 {self.shifted_windows} 窗口 | 丢弃尾巴 {self.dropped_tail_bins} 格"
        )


@dataclass(frozen=True, slots=True)
class _WindowEntry:
    """索引项：一个窗口的定位信息。

    `bpm_eff` 是该窗口局部网格的等效 BPM（= 60 / J(tau_start)），与 `(x_bins, t_window)`
    一起构成**网格身份**（:meth:`ChartPairDataset.grid_key`）——只有身份相同的窗口
    才能进同一个 batch（见 :func:`collate_field_batch`）。
    """

    row_index: int
    window_index: int
    tau_start_bins: int
    bpm_eff: float


@dataclass(frozen=True, slots=True)
class _SkipRow:
    """行级跳过（原因键 + 细节），给计数器用。"""

    reason: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class _RowPlan:
    """一行的窗口规划结果（含该行的窗口统计）。"""

    starts: tuple[int, ...]
    bpm_effs: tuple[float, ...]
    dropped_tail_bins: int
    skipped_hold: int
    skipped_bpm: int
    skipped_no_context: int
    dropped_events_no_context: int
    shifted: int


@dataclass(slots=True)
class _IndexCounters:
    """索引构建期间的可变计数器（只在 build 内使用）。"""

    n_rows: int = 0
    n_rows_used: int = 0
    skipped_chart_missing: int = 0
    skipped_no_feature: int = 0
    skipped_no_difficulty: int = 0
    skipped_parse_error: int = 0
    skipped_too_many_lines: int = 0
    skipped_no_windows: int = 0
    skipped_windows_hold_split: int = 0
    skipped_windows_bpm_crossing: int = 0
    skipped_windows_no_visible_context: int = 0
    dropped_events_no_visible_context: int = 0
    shifted_windows: int = 0
    dropped_tail_bins: int = 0
    skipped_format: dict[str, int] = field(default_factory=dict)

    def add_skip(self, skip: _SkipRow) -> None:
        """行级跳过记账（reason 键与字段名一一对应）。"""
        if skip.reason == "format":
            self.skipped_format[skip.detail] = self.skipped_format.get(skip.detail, 0) + 1
            return
        current = getattr(self, f"skipped_{skip.reason}")
        setattr(self, f"skipped_{skip.reason}", current + 1)

    def freeze(self, n_windows: int) -> DatasetIndexStats:
        """固化成不可变的 :class:`DatasetIndexStats`。"""
        return DatasetIndexStats(
            n_rows=self.n_rows,
            n_rows_used=self.n_rows_used,
            n_windows=n_windows,
            skipped_chart_missing=self.skipped_chart_missing,
            skipped_no_feature=self.skipped_no_feature,
            skipped_no_difficulty=self.skipped_no_difficulty,
            skipped_parse_error=self.skipped_parse_error,
            skipped_too_many_lines=self.skipped_too_many_lines,
            skipped_no_windows=self.skipped_no_windows,
            skipped_windows_hold_split=self.skipped_windows_hold_split,
            skipped_windows_bpm_crossing=self.skipped_windows_bpm_crossing,
            skipped_windows_no_visible_context=self.skipped_windows_no_visible_context,
            dropped_events_no_visible_context=self.dropped_events_no_visible_context,
            shifted_windows=self.shifted_windows,
            dropped_tail_bins=self.dropped_tail_bins,
            skipped_format=dict(self.skipped_format),
        )


@dataclass(frozen=True, slots=True)
class _DatasetPlan:
    """索引 + 记账（一次构建，之后只读）。"""

    entries: tuple[_WindowEntry, ...]
    stats: DatasetIndexStats


class ChartPairDataset(torch.utils.data.Dataset[PairSample]):
    """(谱面, 音频特征) 对的**窗口**数据集（plan 02 -> plan 04 的数据入口）。

    行为要点：

    - **取行**：读 `config.manifest_path`（`PairSplits.to_dict()` 形状），取 `config.split` 的行；
    - **判型**：谱面按**内容**嗅探（后缀不可信）；PEC / 官谱 / UNKNOWN / 解析失败
      **显式记账后跳过**（见 :meth:`describe`），不静默丢弃；
    - **切窗**：全谱网格由 `FieldGrid(...).for_chart(chart, tau_end_s=...)` 建立，
      计数由 `build_target` 出，随后按 `t_window` 切窗；
      **不足一窗的尾巴丢弃并计数**；会切断 Hold、跨 BPM 变更点、或**无法满足 `r < 1`**
      的窗口先尝试前移边界，移不动则跳过并分类计数；
    - **遮盖**：每个窗口独立 `build_occlusion_batch(counts_window, ratio=..., seed=派生自 index)`，
      随后 `assert_hold_pairs_not_split` 复核；`r == 1`（全部事件被遮）时按固定步长
      **重掷种子**，仍不行则退化为无遮盖（`r == 0`）并告警计数——**绝不**发出 `r == 1`
      （`masked_poisson_loss` 会拒绝训练）；
    - **音频**：窗口秒区间由绝对 tau 经 `tau_to_seconds` 与 `MERT_FRAME_RATE_HZ` 派生后切片。

    采样约束：同一批的窗口必须**网格同身份**（`x_bins` / `t_window` / `bpm_points` 全同），
    否则 :func:`collate_field_batch` 抛 :class:`GridMismatchError`（测度 J 会静默错掉）。
    采样器用 :meth:`grid_key` 分组即可（同一 BPM 段内的窗口同身份）。

    Note:
        索引（每个窗口的 tau 起点）在**首次** `__len__` / `__getitem__` 时惰性构建：
        构建要读一遍 split 内所有谱面（O(行数)），但只做解析 + 网格 + 边界规划
        （**不做** build_target、不载特征）。这样 `__len__` 与真实窗口数严格一致
        （含尾巴口径），又不会在构造时把全库解析一遍。
    """

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        rows = load_pairs(config.manifest_path, config.split)
        if config.limit is not None:
            rows = rows[: config.limit]
        self._rows: tuple[PairRow, ...] = tuple(rows)
        self._plan: _DatasetPlan | None = None
        #: 「重掷种子后仍 r == 1 => 退化为无遮盖」的窗口数（**诊断计数**，
        #: 逐进程累加：DataLoader 各 worker 各有一份副本）。
        self._no_context_fallbacks = 0

    # ── 只读诊断 ────────────────────────────────────────────────
    def rows(self) -> list[PairRow]:
        """本 split 的行（`limit` 已生效；只读副本）。"""
        return list(self._rows)

    def stats(self) -> DatasetIndexStats:
        """索引记账（会触发索引构建）。"""
        return self._ensure_plan().stats

    def no_visible_context_fallbacks(self) -> int:
        """重掷种子后仍全遮、因而**退化为无遮盖（`r == 0`）**的窗口数（只增的诊断计数）。

        与索引期的 `skipped_windows_no_visible_context`（**与种子无关、必然全遮**而跳过的
        窗口）相对。逐进程累计：DataLoader 各 worker 各持一份副本，全量只在主进程累计。
        """
        return self._no_context_fallbacks

    def grid_key(self, index: int) -> tuple[int, int, float]:
        """第 `index` 个窗口的**网格身份** `(x_bins, t_window, bpm_eff)`。

        `collate_field_batch` 只接受身份完全相同的样本（`FieldBatch` 只携带一个网格，
        测度 J 由它派生）。因此**采样器必须按本键分桶组批**：同一谱面中跨越 BPM 变更点
        的窗口 `bpm_eff` 不同，不能同批（本方法让这件事可判定，而不必去重算 J）。
        """
        plan = self._ensure_plan()
        total = len(plan.entries)
        if index < 0:
            index += total
        if not 0 <= index < total:
            raise IndexError(f"窗口下标 {index} 越界（共 {total} 个窗口）")
        entry = plan.entries[index]
        return (self.config.x_bins, self.config.t_window, entry.bpm_eff)

    def describe(self) -> str:
        """多行诊断文本：配置 + 索引记账（训练日志用）。"""
        config = self.config
        lines = [
            f"ChartPairDataset(split={config.split}, t_window={config.t_window}, "
            f"x_bins={config.x_bins}, k_max={config.k_max}, "
            f"occlusion_ratio={config.occlusion_ratio}, seed={config.seed})",
            self.stats().describe(),
            f"退化（重掷种子后仍 r==1 => 无遮盖）窗口数：{self._no_context_fallbacks}",
        ]
        return "\n".join(lines)

    # ── Dataset 协议 ────────────────────────────────────────────
    def __len__(self) -> int:
        """窗口总数（**不含**被丢弃的尾巴与被跳过的窗口）。"""
        return len(self._ensure_plan().entries)

    def __getitem__(self, index: int) -> PairSample:
        """取第 `index` 个窗口（**确定性**：同一 index 两次取值逐位一致）。"""
        plan = self._ensure_plan()
        total = len(plan.entries)
        if index < 0:
            index += total
        if not 0 <= index < total:
            raise IndexError(f"窗口下标 {index} 越界（共 {total} 个窗口）")
        entry = plan.entries[index]
        return self._build_sample(self._rows[entry.row_index], entry)

    # ── 索引构建 ────────────────────────────────────────────────
    def _ensure_plan(self) -> _DatasetPlan:
        """惰性构建索引（幂等；构建完成前失败则下次重试）。"""
        if self._plan is None:
            self._plan = self._build_plan()
        return self._plan

    def _build_plan(self) -> _DatasetPlan:
        counters = _IndexCounters()
        entries: list[_WindowEntry] = []
        for row_index, row in enumerate(self._rows):
            counters.n_rows += 1
            planned = self._plan_row(row)
            if isinstance(planned, _SkipRow):
                counters.add_skip(planned)
                continue
            counters.n_rows_used += 1
            counters.skipped_windows_hold_split += planned.skipped_hold
            counters.skipped_windows_bpm_crossing += planned.skipped_bpm
            counters.skipped_windows_no_visible_context += planned.skipped_no_context
            counters.dropped_events_no_visible_context += planned.dropped_events_no_context
            counters.shifted_windows += planned.shifted
            counters.dropped_tail_bins += planned.dropped_tail_bins
            for window_index, (start, bpm_eff) in enumerate(
                zip(planned.starts, planned.bpm_effs, strict=True),
            ):
                entries.append(
                    _WindowEntry(
                        row_index=row_index,
                        window_index=window_index,
                        tau_start_bins=int(start),
                        bpm_eff=float(bpm_eff),
                    ),
                )
        stats = counters.freeze(len(entries))
        logger.info("%s", stats.describe())
        if stats.n_windows == 0:
            logger.warning("split=%s 没有任何可用窗口（全部行被跳过）", self.config.split)
        return _DatasetPlan(entries=tuple(entries), stats=stats)

    def _read_chart_or_skip(self, row: PairRow) -> PhigrosChart | _SkipRow:
        """行的前置条件 -> 谱面 IR；任一不满足即返回跳过原因（每一类都有计数器）。

        前置条件按「便宜在前」的顺序检查：特征键 / 定数（行内字段）→ 文件存在
        → **内容判型**（后缀不可信）→ 解析。非 RPE 格式与解析失败**只记账，不静默丢弃**。
        """
        if row.feature_key is None:
            return _SkipRow("no_feature")
        if row.difficulty is None:
            return _SkipRow("no_difficulty")
        path = Path(self.config.chart_dir) / row.chart_path
        try:
            data = path.read_bytes()
        except OSError:
            return _SkipRow("chart_missing")
        fmt, evidence = sniff_format_with_evidence(data)
        if fmt is not ChartFormat.RPE:
            # 非主路径格式（PEC / 官谱 / UNKNOWN）只记账不解析（plan 02 §偏离 2）
            return _SkipRow("format", str(fmt))
        try:
            return parse_rpejson(data, self._chart_source(row, fmt, evidence, path))
        except RpeParseError:
            return _SkipRow("parse_error")

    def _plan_row(self, row: PairRow) -> _RowPlan | _SkipRow:
        """一行的窗口规划；跳过时返回原因（每一类都有计数器）。"""
        config = self.config
        chart = self._read_chart_or_skip(row)
        if isinstance(chart, _SkipRow):
            return chart
        if len(chart.lines) > config.k_max:
            return _SkipRow("too_many_lines")
        grid = FieldGrid(x_bins=config.x_bins).for_chart(chart, tau_end_s=config.tau_end_s)
        if grid.t_bins < config.t_window:
            return _SkipRow("no_windows")
        planned = _plan_windows(
            chart, grid, config.t_window, occlusion_ratio=config.occlusion_ratio
        )
        if not planned.starts:
            return _SkipRow("no_windows")
        return planned

    # ── 样本构建 ────────────────────────────────────────────────
    def _chart_source(
        self,
        row: PairRow,
        fmt: ChartFormat,
        evidence: str,
        path: Path,
    ) -> ChartSource:
        """来源留痕（合规硬约束③：可逐张追溯）。"""
        return ChartSource(
            chart_id=row.chart_id,
            format=fmt,
            sniff_evidence=evidence,
            chart_file=path.name,
        )

    def _load_feature(self, row: PairRow) -> tuple[NDArray[Dynamic], int]:
        """加载特征缓存（**必须**经 `load_feature_cache` 做六项元数据校验）。

        Returns:
            `(数组 (T_full, D), 帧数)`。

        Raises:
            FeatureCacheMismatchError: 元数据不符（**报错而非警告**——它正是 25 Hz 事故的哨兵）。
        """
        feature_key = row.feature_key
        if feature_key is None:  # 索引期已排除；运行时再挡一次（mypy 收窄 + 防御）
            raise ValueError(f"行 chart_id={row.chart_id} 没有 feature_key")
        npz_path, meta_path = feature_cache_paths(self.config.feature_dir, str(feature_key))
        embedding, _meta = load_feature_cache(npz_path, meta_path)
        return embedding, int(embedding.shape[0])

    def _build_sample(self, row: PairRow, entry: _WindowEntry) -> PairSample:
        """构建一个窗口样本（纯函数式：全部输入来自 config / 行 / 索引项）。"""
        config = self.config
        chart = self._read_chart_or_skip(row)
        if isinstance(chart, _SkipRow):  # 索引期已通过；文件在两次访问之间变化才会到这里
            raise RuntimeError(
                f"索引期可用的行在取样本时失败（{chart.reason}）：chart_id={row.chart_id}",
            )
        embedding, _n_frames = self._load_feature(row)
        grid = FieldGrid(x_bins=config.x_bins).for_chart(chart, tau_end_s=config.tau_end_s)
        start = entry.tau_start_bins
        window_grid = _window_grid(
            config.x_bins,
            config.t_window,
            grid.bpm_points,
            start * TAU_GRID_DT,
        )
        window_chart = _window_subchart(
            chart,
            grid,
            start,
            config.t_window,
            bpm_eff=window_grid.bpm_points[0].bpm,
        )
        # 计数只经 build_target 产出（唯一的装箱实现）；此处调用的是**窗口子谱 + 窗口网格**，
        # 与「全谱 build_target 后切窗」逐格等价（见 _window_subchart），
        # 但内存从 K*T_full*X*S*C（真实谱面约 1e9 格）降到 K*t_window*X*S*C。
        target = build_target(window_chart, window_grid)
        window_counts = torch.as_tensor(np.ascontiguousarray(target.counts))
        seed = _window_seed(config.seed, entry.row_index, entry.window_index)
        occlusion, attempts, degraded = _window_occlusion(
            window_counts,
            ratio=config.occlusion_ratio,
            seed=seed,
        )
        if degraded:
            self._no_context_fallbacks += 1
            logger.warning(
                "chart_id=%s 窗口 %d：%d 次重掷种子后仍无法让事件可见（r<1），"
                "退化为无遮盖样本（r=0；losses 的 r==0 契约分支）",
                row.chart_id,
                entry.window_index,
                attempts,
            )
        assert_hold_pairs_not_split(window_counts, occlusion)
        tau_start = start * TAU_GRID_DT
        tau_end = (start + config.t_window) * TAU_GRID_DT
        tracks = line_tracks_at(
            chart,
            np.asarray(window_grid.tau_centers() + tau_start, dtype=np.float64),
        )
        audio, frame_start, padded = _slice_audio(
            embedding,
            grid.bpm_points,
            tau_start,
            tau_end,
        )
        difficulty = row.difficulty
        if difficulty is None:  # 索引期已排除（同上）
            raise ValueError(f"行 chart_id={row.chart_id} 没有定数")
        if padded > 0:
            logger.warning(
                "chart_id=%s 窗口 %d 的音频越出特征缓存，补零 %d/%d 帧（检查音频时长与终点口径）",
                row.chart_id,
                entry.window_index,
                padded,
                int(audio.shape[0]),
            )
        return PairSample(
            audio_emb=torch.as_tensor(audio, dtype=torch.float32),
            frame_rate=MERT_FRAME_RATE_HZ,
            line_tracks=tracks,
            line_mask=torch.ones(int(window_counts.shape[0]), dtype=torch.bool),
            difficulty=torch.tensor(float(difficulty), dtype=torch.float32),
            grid=window_grid,
            counts=window_counts,
            occlusion=occlusion,
            chart_id=row.chart_id,
            song_key=row.song_key,
            split=config.split,
            pair_index=entry.row_index,
            window_index=entry.window_index,
            tau_start=float(tau_start),
            chart_path=row.chart_path,
            feature_key=row.feature_key,
            audio_frame_start=frame_start,
            audio_padded_frames=padded,
        )


def _window_occlusion(
    counts: Tensor,
    *,
    ratio: float,
    seed: int,
) -> tuple[Tensor, int, bool]:
    """构造窗口遮盖，**保证 r < 1**（否则 losses 会拒绝该 batch）。

    `masked_poisson_loss` 的契约是 `r == 1` 时抛
    「全部事件都被遮盖，事件项没有可见上下文，拒绝训练」
    （:func:`beatmorph.generation.losses.masked_poisson_loss`）；而遮盖单位是**按 token 区块**
    扩张的，稀疏窗口上「选中单位恰好覆盖全部事件」并不罕见（安静段落、少事件窗口）。
    数据集是遮盖的产出方，因此在这里负责：先用派生种子试一次，`r == 1` 就按固定步长重掷
    种子（最多 :data:`MAX_OCCLUSION_SEED_ATTEMPTS` 次，种子序列仍然完全由 index 派生 =>
    确定性不变）。仍然全遮时退化为 **occlusion 全 False（r == 0）**——这是 losses 明文支持的
    分支（"无遮盖 => 无待补全位置 => 退化为纯密度拟合"），**绝不**发出 `r == 1` 的样本。

    Returns:
        `(occlusion (K, T, X, S, C) bool, 尝试次数, 是否退化)`。
    """
    batch, _stats = build_occlusion_batch(
        counts.unsqueeze(0), ratio=ratio, seed=seed % _SEED_MODULUS
    )
    occlusion = batch.squeeze(0)
    if occluded_event_share(counts, occlusion) < 1.0:
        return occlusion, 1, False
    for attempt in range(1, MAX_OCCLUSION_SEED_ATTEMPTS):
        candidate_seed = (seed + attempt * _OCCLUSION_SEED_STEP) % _SEED_MODULUS
        candidate, _stats = build_occlusion_batch(
            counts.unsqueeze(0),
            ratio=ratio,
            seed=candidate_seed,
        )
        candidate = candidate.squeeze(0)
        if occluded_event_share(counts, candidate) < 1.0:
            return candidate, attempt + 1, False
    return torch.zeros_like(counts, dtype=torch.bool), MAX_OCCLUSION_SEED_ATTEMPTS, True


def _window_can_be_masked(unit_count: int, token_count: int, occlusion_ratio: float) -> bool:
    """该窗口**是否存在**使遮盖比例 `r < 1` 的种子（与种子无关的可行性判定）。

    选择过程在 ratio > 0 时**至少选一个**遮盖单位（`unit_count`），并按 token 区块扩张遮盖
    （`token_count`），因此下面两种情形**与种子无关地**必然全遮，必须在规划期跳过：

    - `unit_count == 1`：唯一单位必被选中（例如窗口里只有一对 Hold）=> 全部事件被遮；
    - `token_count == 1`：全部事件都在同一个 `(线, tau)` token 里 => 任何选择都会扩张到该 token。

    `unit_count == 0`（窗口没有事件）时 `r == 0`，可用（losses 的 r == 0 契约分支）。
    两个条件都是**必要条件**而非充分条件（例如「一对 Hold + 同一 token 内的独立点」仍可能
    全遮），残余情形由 :func:`_window_occlusion` 重掷种子、必要时退化为 `r == 0` 兜底。
    `occlusion_ratio <= 0` 时根本不选遮盖单位（`r == 0`），不受本约束。
    """
    if occlusion_ratio <= 0.0 or unit_count == 0:
        return True
    return unit_count >= 2 and token_count >= 2


def _plan_windows(
    chart: PhigrosChart,
    grid: FieldGrid,
    width: int,
    *,
    occlusion_ratio: float,
) -> _RowPlan:
    """把 t_bins 切成 `width` 格的窗口。

    每个名义窗 `[i*W, (i+1)*W)` 在本窗范围内前移，取**第一个**满足全部条件的起点：

    1. 不切断 Hold 配对（:func:`_hold_blocked_boundaries`）；
    2. 不跨 BPM 变更点（单段窗口网格才能给出正确的 J，见模块 docstring）；
    3. 可被遮盖成 `r < 1`（:func:`_window_can_be_masked`：有事件时必须同时有
       >= 2 个遮盖单位与 >= 2 个事件 token，否则遮盖必然全遮）。

    本窗范围内找不到就**跳过并分类计数**（Hold / BPM / 无法满足 r<1 三类）。

    - **尾巴**：`t_bins % width` 格不足一窗，丢弃并计数（`dropped_tail_bins`）；
    - **前移**可能让相邻窗口轻微重叠（重复数据）而不留空隙；两种情况都由
      `shifted_windows` 与索引记账可见。
    """
    t_bins = grid.t_bins
    n_nominal = t_bins // width
    blocked = _hold_blocked_boundaries(chart, grid.bpm_points, t_bins)
    changes = _bpm_change_points(grid.bpm_points)
    points = _event_points(chart, grid.bpm_points, t_bins, grid.spec(len(chart.lines)))
    starts: list[int] = []
    bpm_effs: list[float] = []
    skipped_hold = 0
    skipped_bpm = 0
    skipped_no_context = 0
    dropped_events_no_context = 0
    shifted = 0
    for nominal in range(n_nominal):
        first = nominal * width
        last = min(first + width - 1, t_bins - width)
        chosen = -1
        for candidate in range(first, last + 1):
            if _window_is_safe(candidate, width, blocked, changes):
                chosen = candidate
                break
        if chosen < 0:
            if bool(blocked[first]):
                skipped_hold += 1
            else:
                skipped_bpm += 1
            continue
        units, tokens, n_events = _window_event_shape(points, chosen, width)
        if not _window_can_be_masked(units, tokens, occlusion_ratio):
            # **只跳过、不前移**：为了「可遮盖」而滑动窗口会把事件挤出所有窗口
            # （静默丢事件）。事件损失由 dropped_events_no_context 如实计数。
            skipped_no_context += 1
            dropped_events_no_context += n_events
            continue
        if chosen != first:
            shifted += 1
        starts.append(chosen)
        bpm_effs.append(_effective_bpm(grid.bpm_points, chosen * TAU_GRID_DT))
    return _RowPlan(
        starts=tuple(starts),
        bpm_effs=tuple(bpm_effs),
        dropped_tail_bins=t_bins - n_nominal * width,
        skipped_hold=skipped_hold,
        skipped_bpm=skipped_bpm,
        skipped_no_context=skipped_no_context,
        dropped_events_no_context=dropped_events_no_context,
        shifted=shifted,
    )


def _slice_audio(
    embedding: NDArray[Dynamic],
    bpm_points: Sequence[BpmPoint],
    tau_start: float,
    tau_end: float,
) -> tuple[NDArray[np.float32], int, int]:
    """按窗口的**秒区间**切片 MERT 特征（帧率经 `MERT_FRAME_RATE_HZ` 派生，不写字面量）。

    Args:
        embedding: `(T_full, D)` 特征缓存。
        bpm_points: 秒 <-> tau 换算的唯一依据（谱面 BPMList）。
        tau_start / tau_end: 窗口的**绝对** tau 区间（拍）。

    Returns:
        `(窗口数组 (frames, D) float32, 起始帧下标, 补零帧数)`。

    Note:
        特征缓存覆盖不到窗口尾部时**补零**并如实报出补零帧数（不静默截断）；
        窗口在 MERT 帧率下不足一帧时至少取 1 帧（由帧率派生，不是字面量）。
    """
    frame_start = round(float(tau_to_seconds(tau_start, bpm_points)) * MERT_FRAME_RATE_HZ)
    frame_end = round(float(tau_to_seconds(tau_end, bpm_points)) * MERT_FRAME_RATE_HZ)
    frames = max(1, frame_end - frame_start)
    total_frames = int(embedding.shape[0])
    feature_dim = int(embedding.shape[1])
    source_start = min(frame_start, total_frames)
    source_end = min(frame_start + frames, total_frames)
    copied = max(0, source_end - source_start)
    window = np.zeros((frames, feature_dim), dtype=np.float32)
    if copied > 0:
        window[:copied] = np.asarray(embedding[source_start:source_end], dtype=np.float32)
    return window, frame_start, frames - copied


def _assert_same_grid(reference: FieldGrid, other: FieldGrid, position: int) -> None:
    """网格身份断言（`x_bins` / `t_bins` / `bpm_points` 三者完全相同）。"""
    identity = (reference.x_bins, reference.t_bins, reference.bpm_points)
    if identity == (other.x_bins, other.t_bins, other.bpm_points):
        return
    raise GridMismatchError(
        "同批样本的网格身份不同（x_bins / t_bins / bpm_points 必须完全一致）："
        f"样本 0 = (x_bins={reference.x_bins}, t_bins={reference.t_bins}, "
        f"bpm_points={_format_bpm(reference)}), 样本 {position} = "
        f"(x_bins={other.x_bins}, t_bins={other.t_bins}, bpm_points={_format_bpm(other)})。"
        "FieldBatch 只携带一个 FieldGrid，而测度 J(tau) 与积分项都由它派生——混批会让两者"
        "静默错掉。请按网格身份分桶组批（本数据集里 = 同一 BPM 段的窗口）。"
    )


def _format_bpm(grid: FieldGrid) -> str:
    """紧凑打印 bpm_points（错误信息用）。"""
    inner = ", ".join(f"({point.time_beats:g}, {point.bpm:g})" for point in grid.bpm_points)
    return f"[{inner}]"


def _assert_sample_shapes(sample: PairSample, grid: FieldGrid, position: int) -> None:
    """单样本形状 / dtype 与网格一致（在 padding 之前逐条检查）。"""
    expected_tracks = (int(sample.line_tracks.shape[0]), grid.t_bins, N_TRACK_CHANNELS)
    if tuple(sample.line_tracks.shape) != expected_tracks:
        raise GridMismatchError(
            f"样本 {position} 的 line_tracks 形状 {tuple(sample.line_tracks.shape)} != "
            f"(K, T_window, {N_TRACK_CHANNELS})={expected_tracks}",
        )
    expected_field = (
        int(sample.counts.shape[0]),
        grid.t_bins,
        grid.x_bins,
        grid.sides,
        grid.channels,
    )
    if tuple(sample.counts.shape) != expected_field:
        raise GridMismatchError(
            f"样本 {position} 的 counts 形状 {tuple(sample.counts.shape)} != "
            f"(K, T, X, S, C)={expected_field}",
        )
    if tuple(sample.occlusion.shape) != expected_field:
        raise GridMismatchError(
            f"样本 {position} 的 occlusion 形状 {tuple(sample.occlusion.shape)} != {expected_field}",
        )
    if sample.line_tracks.dtype != torch.float32:
        raise ValueError(
            f"样本 {position} 的 line_tracks 必须是 float32，得到 {sample.line_tracks.dtype}"
        )
    if sample.counts.dtype != torch.int16:
        raise ValueError(f"样本 {position} 的 counts 必须是 int16，得到 {sample.counts.dtype}")
    if sample.occlusion.dtype != torch.bool:
        raise ValueError(f"样本 {position} 的 occlusion 必须是 bool，得到 {sample.occlusion.dtype}")


def collate_field_batch(samples: Sequence[PairSample], *, k_max: int | None = None) -> FieldBatch:
    """把同批窗口样本拼成 :class:`~beatmorph.generation.batch.FieldBatch` 并断言形状。

    **契约约束（重要）**：`FieldBatch` 只携带**一个** `FieldGrid`，因此同一批的样本
    必须具备**同一网格身份**（`x_bins` / `t_window` / `bpm_points` 完全一致）。
    不一致时抛 :class:`GridMismatchError`——**不得**静默混批：测度 `J(tau)` 与积分项
    `sum lam * J * d_tau * dx` 都由那唯一的网格派生，混批会让它们错掉且**不报错**。

    K 的处理：批内 K 不同时 pad 到批内最大 K（**不超过** `k_max`），
    `line_tracks` / `counts` / `occlusion` 同步 pad，`line_mask` 的 pad 位为 False。
    音频长度不同时补零到批内最长（`FieldBatch` 无 `time_mask`，见模块 docstring）。

    Args:
        samples: 同批窗口样本（至少一个）。
        k_max: 判定线容量上限；None 用 `generation.model.DEFAULT_K_MAX`。

    Returns:
        已通过 `assert_shapes()` 的 :class:`FieldBatch`。
    """
    if not samples:
        raise ValueError("collate_field_batch 需要至少一个样本")
    grid = samples[0].grid
    for position, sample in enumerate(samples[1:], start=1):
        _assert_same_grid(grid, sample.grid, position)
    cap = DEFAULT_K_MAX if k_max is None else int(k_max)
    n_lines = max(int(sample.line_tracks.shape[0]) for sample in samples)
    if n_lines > cap:
        raise ValueError(
            f"批内最大线数 K={n_lines} 超过 k_max={cap}："
            "超容量的谱面应在数据集索引期跳过（DatasetConfig.k_max），不应进入 collate",
        )
    frame_rate = float(samples[0].frame_rate)
    for position, sample in enumerate(samples[1:], start=1):
        if float(sample.frame_rate) != frame_rate:
            raise ValueError(
                f"样本 {position} 的 frame_rate={sample.frame_rate} != {frame_rate}："
                "FieldBatch 只有一个 frame_rate（音频帧轴的唯一口径）",
            )
    batch_size = len(samples)
    audio_frames = max(int(sample.audio_emb.shape[0]) for sample in samples)
    audio_dims = {int(sample.audio_emb.shape[1]) for sample in samples}
    if len(audio_dims) != 1:
        raise ValueError(f"批内音频特征维不一致：{sorted(audio_dims)}")
    audio_dim = audio_dims.pop()
    t_bins = grid.t_bins
    audio = torch.zeros(batch_size, audio_frames, audio_dim, dtype=torch.float32)
    tracks = torch.zeros(batch_size, n_lines, t_bins, N_TRACK_CHANNELS, dtype=torch.float32)
    counts = torch.zeros(
        batch_size,
        n_lines,
        t_bins,
        grid.x_bins,
        grid.sides,
        grid.channels,
        dtype=torch.int16,
    )
    occlusion = torch.zeros(
        batch_size,
        n_lines,
        t_bins,
        grid.x_bins,
        grid.sides,
        grid.channels,
        dtype=torch.bool,
    )
    line_mask = torch.zeros(batch_size, n_lines, dtype=torch.bool)
    difficulty = torch.zeros(batch_size, dtype=torch.float32)
    for position, sample in enumerate(samples):
        _assert_sample_shapes(sample, grid, position)
        k = int(sample.line_tracks.shape[0])
        audio[position, : int(sample.audio_emb.shape[0])] = sample.audio_emb.to(torch.float32)
        tracks[position, :k] = sample.line_tracks
        counts[position, :k] = sample.counts
        occlusion[position, :k] = sample.occlusion
        line_mask[position, :k] = True
        difficulty[position] = float(sample.difficulty)
    batch = FieldBatch(
        audio_emb=audio,
        frame_rate=frame_rate,
        line_tracks=tracks,
        line_mask=line_mask,
        difficulty=difficulty,
        grid=grid,
        counts=counts,
        occlusion=occlusion,
    )
    batch.assert_shapes()
    return batch


def _optional_int(value: Dynamic) -> int | None:
    """宽松取整（None / 缺失 / bool -> None）。"""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Dynamic) -> float | None:
    """宽松取浮点（None / 缺失 / bool -> None）。"""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pair_row(raw: Dynamic, split: str, position: int) -> PairRow:
    """一条清单行 -> :class:`PairRow`（缺 `chart_path` 即结构违约）。"""
    if not isinstance(raw, dict):
        raise DatasetManifestError(
            f"split={split} 的第 {position} 行必须是对象，得到 {type(raw).__name__}"
        )
    chart_path = raw.get("chart_path")
    if not isinstance(chart_path, str) or not chart_path:
        raise DatasetManifestError(f"split={split} 的第 {position} 行缺少 chart_path")
    name = str(raw.get("name") or "")
    composer = str(raw.get("composer") or "")
    feature_key = raw.get("feature_key")
    return PairRow(
        chart_id=_optional_int(raw.get("chart_id")),
        song_key=str(raw.get("song_key") or f"{name}|{composer}"),
        split=str(raw.get("split") or split),
        chart_path=chart_path,
        feature_key=None if feature_key is None else str(feature_key),
        difficulty=_optional_float(raw.get("difficulty")),
        fmt=str(raw.get("format") or str(ChartFormat.RPE)),
        name=name,
        composer=composer,
    )


def load_pairs(manifest_path: Path, split: str) -> list[PairRow]:
    """读清单 JSON 并取出指定 split 的行（形状 == `PairSplits.to_dict()`）。

    Args:
        manifest_path: 清单路径（JSON；JSONL 清单请先经 `build_pairs` 导出）。
        split: `"train"` / `"val"` / `"test"`。

    Returns:
        `PairRow` 列表（保持清单内的顺序，不做排序——顺序即 `pair_index`）。

    Raises:
        SplitError: split 名非法。
        DatasetManifestError: 根不是对象 / 该 split 不是数组 / 行缺 `chart_path`。
        OSError: 清单文件不存在或不可读。
    """
    if split not in SPLIT_NAMES:
        raise SplitError(f"split 必须是 {SPLIT_NAMES} 之一，得到 {split!r}")
    path = Path(manifest_path)
    text = path.read_text(encoding="utf-8")
    try:
        payload: Dynamic = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DatasetManifestError(f"清单 {path} 不是合法 JSON：{exc}") from exc
    if not isinstance(payload, dict):
        raise DatasetManifestError(
            f"清单 {path} 的根必须是对象（PairSplits.to_dict() 形状），得到 {type(payload).__name__}",
        )
    rows = payload.get(split)
    if not isinstance(rows, list):
        raise DatasetManifestError(f"清单 {path} 缺少数组字段 {split!r}")
    return [_pair_row(raw, split, position) for position, raw in enumerate(rows)]
