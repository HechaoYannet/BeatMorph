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

**τ 轴终点（本轮实测的语料级缺陷 → `tau_end_policy`）**
`PhigrosChart.duration_s()` 取 `max(最后一事件, META.chartTime)`，而 **`chartTime` 在真实语料里大面积不可信**：
全库 8551 张里 `time_span_s / audio_duration_s > 10` 的有 **4452 张（52.2%）**，中位比值 **52.8×**、
p99 2575×、最大 3 407 201×（chart 26102：span 7.49e8 s vs 音频 220 s）；
中位 `time_span_s` 是 **7766 s（2.2 小时）**，而中位音频只有 151 s。
分布是**双峰**的：47% 的谱面（`chartTime` 缺省或为 0）落在音频时长 1.1× 以内，另 53% 远超音频长度。
后果（本轮实测）：200 行切片切出 **6 744 925 个窗口**（≈3.4 万窗/行，而正常谱面只有 50–110 窗），
其中绝大多数是**空窗**（无事件、音频整段越界补零）；全库索引因此约 2.3 h 才轮到第一个优化步，
且 G1 抽到的批实测 K=2 / 0 事件（门禁空过）。
因此默认 `tau_end_policy="audio"`：**τ 轴终点 = min(谱面口径, 特征缓存记录的音频时长)**，
并按 `tau_end_truncated_rows` / `events_beyond_tau_end` **显式记账**（不静默截断）；
`policy="chart"` 保留旧行为。端点口径本身仍是 plan 03 §9-14 的**未裁定**项
（`chartTime` / 最后一事件 / 音频时长），本默认值的作用是**先消除不可信输入**，
不替该裁定下结论。

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

import dataclasses
import hashlib
import json
import zlib
from bisect import bisect_left
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

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
    FeatureCacheMeta,
    PairRow,
    SplitError,
    feature_cache_paths,
    load_feature_cache,
)
from beatmorph.data.tracks import N_TRACK_CHANNELS, line_tracks_at
from beatmorph.decoder.events import gameplay_subchart
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
    Granularity,
    assert_hold_pairs_not_split,
    build_occlusion_batch,
    occluded_event_share,
)
from beatmorph.generation.model import DEFAULT_K_MAX

if TYPE_CHECKING:
    from beatmorph.data.window_cache import WindowCacheReader

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
    "SongWindows",
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


#: 索引缓存的格式版本：**改了窗口规划口径就要 +1**（否则旧缓存会被当成新口径复用）。
_PLAN_CACHE_VERSION: int = 1
#: 索引缓存默认目录名（放在清单旁边；`data/processed/` 不入库）。
_PLAN_CACHE_DIRNAME: str = ".dataset_index"

#: τ 轴终点口径的合法取值（见模块 docstring 的实测依据）。
TAU_END_POLICIES: frozenset[str] = frozenset({"audio", "chart"})

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


def _count_events_beyond_axis(
    chart: PhigrosChart,
    bpm_points: Sequence[BpmPoint],
    t_bins: int,
) -> int:
    """起点落在 τ 轴之外、因而**整条不会被计入目标**的事件数（与 build_target 的跳过口径一致）。

    只在 τ 轴被截断（`tau_end_policy="audio"` 且 `chartTime` 虚高）时才会非 0：
    这些事件是**被口径丢掉的真实数据**，因此必须显式记账而不是静默消失。
    """
    n_beyond = 0
    for note in chart.notes:
        if note.is_fake or abs(note.position_x) > RPE_STAGE_HALF_WIDTH:
            continue
        start = tau_bin_index(float(seconds_to_tau(note.t, bpm_points)))
        if start >= t_bins:
            n_beyond += 1
    return n_beyond


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
        tau_end_s: τ 轴终点（秒）的**显式**覆盖；None = 由 `tau_end_policy` 决定。
        tau_end_policy: τ 轴终点口径（模块 docstring 有实测依据）——
            `"audio"`（默认）= `min(chart.duration_s(), 音频时长)`，用于消除真实语料里
            大面积不可信的 `chartTime`；`"chart"` = 旧行为 `chart.duration_s()`（保留以便对照）。
            `tau_end_s` 非 None 时本字段不生效。
        chart_cache_size: 已解析谱面的 LRU 容量（`__getitem__` 不再重复解析同一行；0 = 关闭）。
        feature_cache_size: 已载入特征数组的 LRU 容量（0 = 关闭）。
        index_cache_dir: 索引落盘缓存目录；None = `<manifest 所在目录>/.dataset_index`，
            `False`（或 `index_cache=False`）= 关闭缓存（每次都重建，测试与对照用）。
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
    tau_end_policy: str = "audio"
    chart_cache_size: int = 8
    feature_cache_size: int = 2
    index_cache: bool = True
    index_cache_dir: Path | None = None
    #: 窗口预切缓存的**根目录**（`beatmorph.data.window_cache`）。None = 关闭（走原路径）。
    #:
    #: 开启后 `__getitem__` 只读 mmap 重建，不再解析谱面 / 解压特征（每窗 0.85 s → 毫秒级）。
    #: 它是**纯派生加速器**：指纹不符 / 目录缺失 / 结构非法都回退到原路径，绝不静默降级。
    window_cache_dir: Path | None = None
    x_bins: int = RPE_X_GRID_BINS
    k_max: int = DEFAULT_K_MAX
    occlusion_ratio: float = 0.5
    #: 遮盖粒度（`generation.masks.Granularity`）。默认 `"event"`（契约路径）。
    #: `"block"` = (τ, x) 小块遮盖（块内含空格）—— 它保留「同一 token 内的可见邻居」，
    #: 从而让「被预测 token 内部的空间分布」重新可从输入推断（plan 07 §9-75②/§9-76）。
    #: ⚠️ **只有 `"event"` 与窗口缓存兼容**：缓存把遮盖压成 token 位图并断言区块结构；
    #: 其它粒度必须把 `window_cache_dir` 设为 None（否则装配期直接抛 WindowCacheError）。
    occlusion_granularity: Granularity = "event"
    #: **目标口径**：只把「可计分」note 计入目标（决策者 2026-09-30 裁定）。
    #:
    #: 可计分 = 非 fake 且**命中时刻该线可见**（判据只有一份实现：`decoder.events.note_is_scorable`）。
    #: 为什么必须能切换：实测（`runs/_probe_scorable_share.py`）train 里**命中时线不可见**的 note 占
    #: **22.81%**（其中 12.94% 落在**真线**上 ⇒ 线级剥离拿不掉），val 18.68%；而真实语料里
    #: 「可计分 note 落在装饰线上」= **0** ⇒ 我们的目标把两个本不相交的集合混在了一起。
    #: 那部分事件由**演出设计**决定、不由音乐决定 ⇒ 占目标两成的梯度在鼓励模型忽略音乐。
    #: ⚠️ 打开它会使目标口径改变 ⇒ 损失数值**不可**与关闭时直接比较（须在同一口径下对照）。
    scorable_target: bool = False
    seed: int = 0
    limit: int | None = None
    #: 索引构建的**进程数**（1 = 串行，保持默认行为）。
    #:
    #: 为什么需要：索引构建是**逐行解析全库谱面**的，实测单核跑满、24 核里只用 1 个，
    #: 全量 train 一次 30-40 min；而它能按行完美并行（每行独立、结果按行序拼装）。
    #: **不改任何读数**：并行路径与串行路径的窗口、记账、顺序逐位一致（`test_dataset_index_parallel`）。
    index_jobs: int = 1

    def __post_init__(self) -> None:  # noqa: PLR0912 —— 扁平校验清单，分支数不反映复杂度
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
        if self.occlusion_granularity not in ("event", "frame", "cell", "block"):
            raise ValueError(
                f"未知的遮盖粒度 {self.occlusion_granularity!r}（合法值：event/frame/cell/block）"
            )
        if self.index_jobs < 1:
            raise ValueError(f"index_jobs 必须 >= 1，得到 {self.index_jobs}")
        if self.scorable_target and self.window_cache_dir is not None:
            raise ValueError(
                "scorable_target=True 与窗口预切缓存**不能同时开**：缓存里的计数是按旧口径"
                "（含命中时不可见的 note）物化的，静默复用会得到一套与配置不符的目标。"
                "请显式把 window_cache_dir 设为 None（或重建覆盖该口径的缓存）。"
            )
        if self.limit is not None and self.limit < 0:
            raise ValueError(f"limit 必须 >= 0 或 None，得到 {self.limit}")
        if self.tau_end_s is not None and self.tau_end_s <= 0.0:
            raise ValueError(f"tau_end_s 必须为正或 None，得到 {self.tau_end_s}")
        if self.tau_end_policy not in TAU_END_POLICIES:
            raise ValueError(
                f"tau_end_policy 必须是 {sorted(TAU_END_POLICIES)} 之一，得到 {self.tau_end_policy!r}",
            )
        if self.chart_cache_size < 0 or self.feature_cache_size < 0:
            raise ValueError("chart_cache_size / feature_cache_size 必须 >= 0")
        if self.index_cache_dir is not None:
            object.__setattr__(self, "index_cache_dir", Path(self.index_cache_dir))


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
    #: τ 轴被音频时长截断的行数（`chartTime` 不可信的规模；\u2260 0 就是语料缺陷的度量）。
    tau_end_truncated_rows: int = 0
    #: 被截断掉的 τ 轴长度合计（秒）——即「旧口径会多切出多少秒的空窗」。
    tau_end_truncated_seconds: float = 0.0
    #: 音频时长不可用（缺特征缓存元数据）而回退到谱面口径的行数。
    tau_end_fallback_rows: int = 0
    #: 落在 τ 轴之外、未被计入目标的事件数（`build_target` 的同口径统计）。
    events_beyond_tau_end: int = 0
    #: `scorable_target=True` 时被摘掉的**不可计分** note 数（命中时线不可见 / 假音符）。
    #: 关闭该开关时恒为 0——**没开就不许有数**，否则台账会假装口径生效过。
    notes_non_scorable_dropped: int = 0
    #: `scorable_target=True` 时整张谱面没有可计分 note（去表演后无内容）而被跳过的行数。
    skipped_no_playable_lines: int = 0
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
            f"无可用窗口 {self.skipped_no_windows}、去表演后无内容 {self.skipped_no_playable_lines} | "
            f"跳过窗口：Hold 切断 {self.skipped_windows_hold_split}、"
            f"跨 BPM 变更点 {self.skipped_windows_bpm_crossing}、"
            f"无法满足 r<1 {self.skipped_windows_no_visible_context}"
            f"（连带 {self.dropped_events_no_visible_context} 个事件点） | "
            f"前移 {self.shifted_windows} 窗口 | 丢弃尾巴 {self.dropped_tail_bins} 格 | "
            f"τ 轴截断 {self.tau_end_truncated_rows} 行"
            f"（{self.tau_end_truncated_seconds / 3600:.1f} 小时空窗）"
            f"、回退 {self.tau_end_fallback_rows} 行、轴外事件 {self.events_beyond_tau_end}"
            + (
                ""
                if self.notes_non_scorable_dropped == 0
                else f" | 目标口径：摘掉不可计分 note {self.notes_non_scorable_dropped}"
            )
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
    #: `scorable_target=True` 时整张谱面没有可计分 note（去表演后无内容）而被跳过的行数。
    skipped_no_playable_lines: int = 0
    skipped_windows_hold_split: int = 0
    skipped_windows_bpm_crossing: int = 0
    skipped_windows_no_visible_context: int = 0
    dropped_events_no_visible_context: int = 0
    shifted_windows: int = 0
    dropped_tail_bins: int = 0
    tau_end_truncated_rows: int = 0
    tau_end_truncated_seconds: float = 0.0
    tau_end_fallback_rows: int = 0
    events_beyond_tau_end: int = 0
    notes_non_scorable_dropped: int = 0
    skipped_format: dict[str, int] = field(default_factory=dict)

    def merge(self, other: _IndexCounters) -> None:
        """把另一个计数器的读数并入本计数器（并行索引构建用；**逐字段相加**）。

        为什么必须有它而不是在各 worker 里直接改共享计数器：`_build_plan` 的并行路径要保证
        「与串行**逐位一致**」，而唯一的办法是让每个 worker 产出**自己的完整读数**、
        再按行序合并 —— 就地累加会随进程数改变结果。
        """
        for field_name in self.__slots__:
            current = getattr(self, field_name)
            incoming = getattr(other, field_name)
            if isinstance(current, dict):
                for key, value in incoming.items():
                    current[key] = current.get(key, 0) + value
            else:
                setattr(self, field_name, current + incoming)

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
            skipped_no_playable_lines=self.skipped_no_playable_lines,
            skipped_windows_hold_split=self.skipped_windows_hold_split,
            skipped_windows_bpm_crossing=self.skipped_windows_bpm_crossing,
            skipped_windows_no_visible_context=self.skipped_windows_no_visible_context,
            dropped_events_no_visible_context=self.dropped_events_no_visible_context,
            shifted_windows=self.shifted_windows,
            dropped_tail_bins=self.dropped_tail_bins,
            tau_end_truncated_rows=self.tau_end_truncated_rows,
            tau_end_truncated_seconds=self.tau_end_truncated_seconds,
            tau_end_fallback_rows=self.tau_end_fallback_rows,
            events_beyond_tau_end=self.events_beyond_tau_end,
            notes_non_scorable_dropped=self.notes_non_scorable_dropped,
            skipped_format=dict(self.skipped_format),
        )


@dataclass(frozen=True, slots=True)
class _DatasetPlan:
    """索引 + 记账（一次构建，之后只读）。"""

    entries: tuple[_WindowEntry, ...]
    stats: DatasetIndexStats


# ══════════════════════════════════════════════════════════════
# 索引落盘缓存（plan 02 §9 ④ 的修法①③）
# ══════════════════════════════════════════════════════════════
#
# 为什么要它：索引构建要对**每一行**读文件 + 嗅探 + 解析 + 规划（全库 8551 行，实测
# 0.19 s/行 ⇒ 约 27 min），而它对同一份（清单 + 配置 + 谱面文件）是**纯函数**。
# 不缓存 ⇒ 每次开训都要先等半小时才轮到第一个优化步；训练因此被数据侧饿死。
#
# 纪律：
# - 指纹覆盖**一切影响规划的输入**（口径版本 / 配置 / 行身份 / 谱面与特征元数据的 stat），
#   因此「改了配置却复用旧计划」不可能发生；
# - 任何读失败都**回退到重建**（缓存是加速器，不是事实源）；写入是**原子**的。


def _plan_cache_path(config: DatasetConfig, rows: Sequence[PairRow]) -> Path | None:
    """索引缓存的落盘路径（None = 该配置下不缓存）。"""
    if not config.index_cache:
        return None
    directory = (
        config.index_cache_dir
        if config.index_cache_dir is not None
        else config.manifest_path.parent / _PLAN_CACHE_DIRNAME
    )
    return directory / f"{_plan_fingerprint(config, rows)}.npz"


def _file_stamp(path: Path) -> str:
    """文件的 `(size, mtime_ns)` 戳（读失败 => `missing`；不读内容，只 stat）。"""
    try:
        info = path.stat()
    except OSError:
        return "missing"
    return f"{info.st_size}:{info.st_mtime_ns}"


#: 并行索引构建的**进程内**工作副本（每个 worker 进程一份；一次构建 = 一份配置）。
#: 不放进 payload：每行都重建一次 ChartPairDataset 会重复解析清单（6750 次）。
_WORKER_DATASET: ChartPairDataset | None = None


def _plan_row_worker(
    payload: tuple[DatasetConfig, int, PairRow],
) -> tuple[_RowPlan | _SkipRow, _IndexCounters]:
    """并行索引构建的 worker：**一行一个任务**，结果由调用方按行序拼装。

    走的是与串行**完全同一份** _plan_row（谱面 LRU、特征元数据、去表演过滤都在里面），
    因此并行不引入第二套口径；worker 之间不共享任何可变状态（计数器各自返回、按行序合并）。
    """
    global _WORKER_DATASET
    config, row_index, row = payload
    if _WORKER_DATASET is None:
        _WORKER_DATASET = ChartPairDataset(config)
    counters = _IndexCounters()
    return _WORKER_DATASET._plan_row(row_index, row, counters), counters


def _plan_fingerprint(config: DatasetConfig, rows: Sequence[PairRow]) -> str:
    """索引指纹：配置 + 每一行的身份 + 谱面/特征元数据文件的 stat。

    **不哈希文件内容**（那正是要避免的 O(全库) 读盘）；stat 里的 `mtime_ns` 已足以
    发现文件被替换。清单本身由调用方传入的行序列代表（已按 split/limit 切好）。
    """
    digest = hashlib.sha1()
    digest.update(f"plan-v{_PLAN_CACHE_VERSION}".encode())
    payload = {
        "split": config.split,
        "limit": config.limit,
        "t_window": config.t_window,
        "x_bins": config.x_bins,
        "k_max": config.k_max,
        "occlusion_ratio": config.occlusion_ratio,
        "tau_end_s": config.tau_end_s,
        "tau_end_policy": config.tau_end_policy,
        # 目标口径**必须进指纹**：它决定哪些 note 进桶 ⇒ 换口径必须让索引缓存失效
        "scorable_target": config.scorable_target,
    }
    digest.update(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode())
    for row in rows:
        chart_path = config.chart_dir / row.chart_path
        feature_key = None if row.feature_key is None else str(row.feature_key)
        meta_path = (
            None if feature_key is None else feature_cache_paths(config.feature_dir, feature_key)[1]
        )
        digest.update(
            (
                f"|{row.chart_id}|{row.chart_path}|{row.feature_key}|{row.difficulty}"
                f"|{_file_stamp(chart_path)}|{'n/a' if meta_path is None else _file_stamp(meta_path)}"
            ).encode(),
        )
    return digest.hexdigest()


def save_plan_cache(path: Path, plan: _DatasetPlan, *, fingerprint: str) -> None:
    """原子落盘索引（先写临时文件再 `replace`，避免半个文件被当成有效缓存）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = plan.entries
    temporary = path.with_suffix(".tmp.npz")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            row_index=np.asarray([e.row_index for e in entries], dtype=np.int32),
            window_index=np.asarray([e.window_index for e in entries], dtype=np.int32),
            tau_start_bins=np.asarray([e.tau_start_bins for e in entries], dtype=np.int32),
            bpm_eff=np.asarray([e.bpm_eff for e in entries], dtype=np.float64),
            fingerprint=np.asarray(fingerprint),
            stats=np.asarray(json.dumps(dataclasses.asdict(plan.stats), ensure_ascii=False)),
        )
    temporary.replace(path)


def load_plan_cache(path: Path, *, fingerprint: str) -> _DatasetPlan | None:
    """读回索引缓存；**任何异常都返回 None**（缓存不是事实源，坏了就重建）。"""
    try:
        with np.load(path, allow_pickle=False) as data:
            stored = str(data["fingerprint"])
            if stored != fingerprint:
                logger.warning(
                    "索引缓存指纹不符（存 %s / 算 %s），改为重建：%s",
                    stored[:12],
                    fingerprint[:12],
                    path,
                )
                return None
            stats = DatasetIndexStats(**json.loads(str(data["stats"])))
            entries = tuple(
                _WindowEntry(
                    row_index=int(row),
                    window_index=int(window),
                    tau_start_bins=int(start),
                    bpm_eff=float(bpm),
                )
                for row, window, start, bpm in zip(
                    data["row_index"],
                    data["window_index"],
                    data["tau_start_bins"],
                    data["bpm_eff"],
                    strict=True,
                )
            )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("索引缓存不可用（%s），改为重建：%s", exc, path)
        return None
    if len(entries) != stats.n_windows:
        logger.warning(
            "索引缓存自相矛盾（条目 %d != 记账 %d），改为重建", len(entries), stats.n_windows
        )
        return None
    return _DatasetPlan(entries=entries, stats=stats)


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
        #: 窗口预切缓存的只读视图（None = 未启用 / 不可用 ⇒ 走原路径）。
        self._window_cache = self._open_window_cache()
        self._plan: _DatasetPlan | None = None
        #: 特征缓存**元数据**（含音频时长）：小的 JSON，索引期就要用（τ 轴终点口径），
        #: 且每行只读一次。不做 LRU —— 全库 8551 条也只占几 MB。
        self._meta_cache: dict[str, FeatureCacheMeta] = {}
        #: 已解析谱面的 LRU（plan 02 §9 ④：`__getitem__` 曾对同一行重复解析）。
        self._chart_cache: OrderedDict[int, PhigrosChart | _SkipRow] = OrderedDict()
        #: 已载入特征数组的 LRU（同上；单首 (T, 1024) fp16 约 40-50 MB）。
        self._feature_cache: OrderedDict[str, NDArray[Dynamic]] = OrderedDict()
        #: 本次索引是否来自落盘缓存（诊断用；**不影响**任何结果）。
        self._plan_from_cache = False
        #: 「重掷种子后仍 r == 1 => 退化为无遮盖」的窗口数（**诊断计数**，
        #: 逐进程累加：DataLoader 各 worker 各有一份副本）。
        self._no_context_fallbacks = 0
        #: 每行被摘掉的**不可计分** note 数（`scorable_target=True` 时的台账）。
        #: 幂等赋值：`_chart_for` 关掉 LRU 时会被重复调用，累加会翻倍。
        self._dropped_non_scorable: dict[int, int] = {}

    #: 交给 DataLoader worker 时**不**序列化的派生 / 进程内状态（各 worker 自己按需重建）。
    #:
    #: 为什么必须剥掉（RFC-0034 S2）：
    #: ① `_plan` 是 10 MB 量级的派生结构，每个 worker 一份纯属浪费，而它可以从**落盘索引
    #:    缓存**按需重建（秒级命中，不重解析任何谱面）；
    #: ② 两个 LRU 是**进程内**状态，带着主进程的命中记录过去毫无意义，而命中率只与访问
    #:    顺序有关（计划层决定）；
    #: ③ 诊断计数必须**各进程独立**累加——复制过去等于把主进程的数字抄成 N 份，越加越错。
    _WORKER_LOCAL_STATE: ClassVar[tuple[str, ...]] = (
        "_plan",
        "_chart_cache",
        "_feature_cache",
        "_plan_from_cache",
        "_no_context_fallbacks",
        "_dropped_non_scorable",
    )

    def __getstate__(self) -> dict[str, Any]:
        """序列化给 DataLoader worker 的视图（剥掉派生 / 进程内状态，见 `_WORKER_LOCAL_STATE`）。

        没有它，spawn 出来的每个 worker 都会收到一份完整索引计划与主进程的 LRU —— 内存按
        worker 数翻倍，而诊断计数还会被复制成 N 份。
        """
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in self._WORKER_LOCAL_STATE
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """worker 侧还原：派生状态留空，首次访问时按需从落盘缓存重建。"""
        self.__dict__.update(state)
        self._plan = None
        self._chart_cache = OrderedDict()
        self._feature_cache = OrderedDict()
        self._plan_from_cache = False
        self._no_context_fallbacks = 0
        # ⚠️ 剥掉的属性必须在 worker 侧**重新建出**：少一个就会在 _apply_scorable_target 里
        # 变成 AttributeError（本轮实测：DataLoader 一开就崩，而单进程路径完全正常）。
        self._dropped_non_scorable = {}

    # ── 只读诊断 ────────────────────────────────────────────────
    def rows(self) -> list[PairRow]:
        """本 split 的行（`limit` 已生效；只读副本）。"""
        return list(self._rows)

    def _open_window_cache(self) -> WindowCacheReader | None:
        """尝试打开窗口预切缓存；**任何不可用都返回 None**（回退原路径，不抛、不静默降级）。

        指纹覆盖一切语义字段（`window_cache_fingerprint` 直接复用索引缓存的指纹）⇒
        指纹相同就意味着「同一批窗口、同一批遮盖」，因此这里不需要再去比 `n_windows`
        ——那会强制构建索引，正是本缓存要绕开的那一步。
        """
        root = self.config.window_cache_dir
        if root is None:
            return None
        from beatmorph.data.window_cache import (
            WindowCacheReader,
            read_index,
            window_cache_directory,
            window_cache_fingerprint,
        )

        fingerprint = window_cache_fingerprint(self.config, self._rows, seed=self.config.seed)
        directory = window_cache_directory(Path(root), self.config.split, fingerprint)
        index = read_index(directory, fingerprint=fingerprint)
        if index is None:
            return None
        logger.info("窗口预切缓存已启用：%s（%d 个窗口）", directory, index.n_windows)
        return WindowCacheReader(directory, index, list(self._rows), split=self.config.split)

    def stats(self) -> DatasetIndexStats:
        """索引记账（会触发索引构建）。"""
        return self._ensure_plan().stats

    def no_visible_context_fallbacks(self) -> int:
        """重掷种子后仍全遮、因而**退化为无遮盖（`r == 0`）**的窗口数（只增的诊断计数）。

        与索引期的 `skipped_windows_no_visible_context`（**与种子无关、必然全遮**而跳过的
        窗口）相对。

        ⚠️ **口径**：逐**进程**累计。`data.workers=0` 时它就是全程精确值；workers>0 时每个
        worker 各持一份副本，**worker 侧的计数目前不上报**（主进程只看到自己那份）——这是
        RFC-0034 S3 明确记录的缺口（旁路汇总见该 RFC §7-S3），不是「已经精确」。
        """
        return self._no_context_fallbacks

    # ── 廉价密度读数（RFC-0039 R2：val 分层抽样用）──────────────
    def window_event_count(self, index: int) -> int | None:
        """第 `index` 个窗口的**事件数**；不可廉价取得时返回 None（**不静默取 0**）。

        为什么要有「None」这条：事件数只能从**窗口预切缓存**的稀疏计数里廉价读到；
        无缓存时唯一的办法是物化整个样本（含音频，0.85 s/窗）。把「拿不到」与
        「真的是 0 事件」混成一个数（0）会让分层抽样把全体窗口塞进「空窗层」——
        那正是本机制要修的偏差，所以这里必须显式区分。
        """
        reader = self._window_cache
        if reader is None:
            return None
        total = len(reader)
        if index < 0:
            index += total
        if not 0 <= index < total:
            raise IndexError(f"窗口下标 {index} 越界（共 {total} 个窗口）")
        return reader.event_count(index)

    def window_line_count(self, index: int) -> int | None:
        """第 `index` 个窗口的 **K**（判定线条数）；同 :meth:`window_event_count`：缓存才有。"""
        reader = self._window_cache
        if reader is None:
            return None
        total = len(reader)
        if index < 0:
            index += total
        if not 0 <= index < total:
            raise IndexError(f"窗口下标 {index} 越界（共 {total} 个窗口）")
        return reader.line_count(index)

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

    def window_row_index(self, index: int) -> int:
        """第 `index` 个窗口所属的**split 内行下标**（覆盖率记账用；**不解析谱面**）。

        `ManifestBatchSource` 需要在线回答「这个 run 见过多少张谱面」，而窗口 -> 行的映射
        只在索引计划里、且必须 O(1) 拿到（每一步都会调用）。此前调用方只能去读私有
        `_plan`；这里给一个显式出口，避免覆盖率记账被迫依赖私有字段。
        """
        plan = self._ensure_plan()
        total = len(plan.entries)
        if index < 0:
            index += total
        if not 0 <= index < total:
            raise IndexError(f"窗口下标 {index} 越界（共 {total} 个窗口）")
        return int(plan.entries[index].row_index)

    def describe(self) -> str:
        """多行诊断文本：配置 + 索引记账（训练日志用）。"""
        config = self.config
        lines = [
            f"ChartPairDataset(split={config.split}, t_window={config.t_window}, "
            f"x_bins={config.x_bins}, k_max={config.k_max}, "
            f"occlusion_ratio={config.occlusion_ratio}, seed={config.seed})",
            self.stats().describe(),
            "索引来源：" + ("落盘缓存" if self._plan_from_cache else "本次构建"),
            f"退化（重掷种子后仍 r==1 => 无遮盖）窗口数：{self._no_context_fallbacks}",
        ]
        return "\n".join(lines)

    def _remask(self, sample: PairSample, index: int) -> PairSample:
        """按 `occlusion_granularity` **从计数重算**遮盖（非 `event` 粒度下必须走这一步）。

        为什么需要：窗口预切缓存把遮盖压成 **(K, T) token 位图**，并在装配时断言稠密遮盖恰等于
        它的广播（`window_cache._token_view`）⇒ 那个紧凑表示只对「按 token 扩张」的粒度成立。
        块状粒度下**计数本身仍然有效**（缓存存的就是计数），只是遮盖必须重新生成；
        种子沿用与建缓存时相同的派生式 ⇒ 与「无缓存路径」逐位一致。
        """
        plan = self._ensure_plan()
        entry = plan.entries[index]
        seed = _window_seed(self.config.seed, entry.row_index, entry.window_index)
        occlusion, _attempts, _degraded = _window_occlusion(
            sample.counts,
            ratio=self.config.occlusion_ratio,
            seed=seed,
            granularity=self.config.occlusion_granularity,
        )
        return dataclasses.replace(sample, occlusion=occlusion)

    # ── Dataset 协议 ────────────────────────────────────────────
    def __len__(self) -> int:
        """窗口总数（**不含**被丢弃的尾巴与被跳过的窗口）。"""
        return len(self._ensure_plan().entries)

    def __getitem__(self, index: int) -> PairSample:
        """取第 `index` 个窗口（**确定性**：同一 index 两次取值逐位一致）。

        启用窗口预切缓存时直接由 mmap 重建
        （`tests/unit/data/test_window_cache.py` 锁定它与本方法逐位一致）。
        """
        if self._window_cache is not None:
            sample = self._window_cache.sample(index)
            if self.config.occlusion_granularity != "event":
                sample = self._remask(sample, index)
            return sample
        plan = self._ensure_plan()
        total = len(plan.entries)
        if index < 0:
            index += total
        if not 0 <= index < total:
            raise IndexError(f"窗口下标 {index} 越界（共 {total} 个窗口）")
        entry = plan.entries[index]
        return self._build_sample(self._rows[entry.row_index], entry, row_index=entry.row_index)

    # ── τ 轴终点口径（模块 docstring 的实测依据）────────────────────
    def _feature_meta(self, row: PairRow) -> FeatureCacheMeta | None:
        """特征缓存的**元数据**（小 JSON，含 `duration_s` = 音频时长）。

        索引期读它是必要的：τ 轴终点口径要音频时长（模块 docstring），而 meta 只有几 KB，
        与「索引期不载特征数组」的纪律并不冲突。每行只读一次并缓存。

        Returns:
            None = 缺 `feature_key` / meta 不可读 / 结构非法（调用方按回退记账）。
        """
        key = row.feature_key
        if key is None:
            return None
        cached = self._meta_cache.get(str(key))
        if cached is not None:
            return cached
        _npz_path, meta_path = feature_cache_paths(self.config.feature_dir, str(key))
        try:
            meta = FeatureCacheMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("特征缓存元数据不可读：%s（τ 轴终点回退到谱面口径）", meta_path)
            return None
        self._meta_cache[str(key)] = meta
        return meta

    def _axis_end_s(
        self,
        chart: PhigrosChart,
        row: PairRow,
        *,
        counters: _IndexCounters | None,
    ) -> float:
        """τ 轴终点（秒）：显式 `tau_end_s` > `tau_end_policy`（见模块 docstring）。

        `counters` 为 None 时只算不记账（`__getitem__` 路径复用同一口径，保证与索引逐位一致）。
        """
        config = self.config
        if config.tau_end_s is not None:
            return float(config.tau_end_s)
        chart_end = float(chart.duration_s())
        if config.tau_end_policy == "chart":
            return chart_end
        meta = self._feature_meta(row)
        if meta is None:
            if counters is not None:
                counters.tau_end_fallback_rows += 1
            return chart_end
        audio_end = float(meta.duration_s)
        if chart_end > audio_end:
            if counters is not None:
                counters.tau_end_truncated_rows += 1
                counters.tau_end_truncated_seconds += chart_end - audio_end
            return audio_end
        return chart_end

    # ── 行级缓存（plan 02 §9 ④：取批成本是训练吞吐的前置）────────────
    def _chart_for(self, row_index: int, row: PairRow) -> PhigrosChart | _SkipRow:
        """取该行的谱面（带 LRU；`__getitem__` 不再重复解析同一行）。"""
        size = self.config.chart_cache_size
        if size <= 0:
            return self._read_chart_or_skip(row)
        cached = self._chart_cache.get(row_index)
        if cached is not None:
            self._chart_cache.move_to_end(row_index)
            return cached
        chart = self._read_chart_or_skip(row)
        chart = self._apply_scorable_target(row_index, chart)
        self._chart_cache[row_index] = chart
        if len(self._chart_cache) > size:
            self._chart_cache.popitem(last=False)
        return chart

    def _apply_scorable_target(
        self,
        row_index: int,
        chart: PhigrosChart | _SkipRow,
    ) -> PhigrosChart | _SkipRow:
        """按 `config.scorable_target` 把**不可计分**的 note 从目标谱面里摘掉（plan 07 §9-79）。

        为什么在这里做：`_chart_for` 是**唯一的**谱面入口，因此 `_note_bins`（窗口规划与
        Hold 配对）、`_hold_blocked_boundaries`、`_count_events_beyond_axis`、`build_target`
        四条镜像口径**自动**一致——在别处各滤一次就一定会分叉。

        ⚠️ `chart_time_s` 被**补回到原来的 `duration_s()`**：否则摘掉末尾的 note 会把 τ 轴
        缩短，于是「目标口径变了」和「窗口集合变了」会同时发生，A/B 无法归因。
        配对统计记在 `self._dropped_non_scorable[row_index]`（幂等赋值，供索引台账汇总）。
        """
        if not self.config.scorable_target or isinstance(chart, _SkipRow):
            return chart
        dropped = len(chart.notes)
        try:
            playable = gameplay_subchart(chart)
        except ValueError:
            # 整张谱面没有任何可计分 note ⇒ 没有可玩内容。**必须记账跳过**，不许喂空谱。
            self._dropped_non_scorable[row_index] = dropped
            return _SkipRow("no_playable_lines")
        self._dropped_non_scorable[row_index] = dropped - len(playable.notes)
        return playable

    def _embedding(self, row: PairRow) -> NDArray[Dynamic]:
        """取该行的特征数组（带 LRU；同一谱面的相邻窗口取批时命中率高）。"""
        key = None if row.feature_key is None else str(row.feature_key)
        size = self.config.feature_cache_size
        if size <= 0 or key is None:
            return self._load_feature(row)[0]
        cached = self._feature_cache.get(key)
        if cached is not None:
            self._feature_cache.move_to_end(key)
            return cached
        embedding, _frames = self._load_feature(row)
        self._feature_cache[key] = embedding
        if len(self._feature_cache) > size:
            self._feature_cache.popitem(last=False)
        return embedding

    # ── 索引构建 ────────────────────────────────────────────────
    def _ensure_plan(self) -> _DatasetPlan:
        """惰性构建索引（幂等；构建完成前失败则下次重试）。

        优先读**落盘缓存**（指纹覆盖配置与文件 stat，见 :func:`_plan_fingerprint`）：
        索引是全库级的 O(行数) 解析，不缓存就等于每次开训先等半小时（plan 02 §9 ④）。
        缓存缺失 / 损坏 / 指纹不符都回退到重建，且**任何**回退都不改变结果。
        """
        if self._plan is not None:
            return self._plan
        path = _plan_cache_path(self.config, self._rows)
        if path is not None and path.is_file():
            cached = load_plan_cache(path, fingerprint=_plan_fingerprint(self.config, self._rows))
            if cached is not None:
                self._plan = cached
                self._plan_from_cache = True
                logger.info("索引缓存命中：%s（窗口 %d）", path, cached.stats.n_windows)
                return self._plan
        self._plan = self._build_plan()
        if path is not None:
            save_plan_cache(
                path, self._plan, fingerprint=_plan_fingerprint(self.config, self._rows)
            )
        return self._plan

    def _build_plan(self) -> _DatasetPlan:
        """索引构建：config.index_jobs > 1 时按行并行，否则串行（默认）。

        两条路径共用 _assemble，因此**窗口、记账、顺序逐位一致**——差别只在「谁算的」。
        并行是纯加速器：index_jobs 不进任何指纹、不改任何读数。
        """
        jobs = max(1, int(self.config.index_jobs))
        if jobs == 1 or len(self._rows) < 2 * jobs:

            def serial_rows() -> Iterator[tuple[_RowPlan | _SkipRow, _IndexCounters]]:
                # ⚠️ **每个计数器的读数必须原样交给 _assemble**：早先的写法在这里新建并丢弃
                # 了一个空计数器，于是 events_beyond_tau_end / tau_end_* / 去表演台账全部归零，
                # 而窗口数看不出来 —— 被 tests/unit/data/test_dataset_tau_end.py 抓住。
                for index, row in enumerate(self._rows):
                    row_counters = _IndexCounters()
                    yield self._plan_row(index, row, row_counters), row_counters

            return self._assemble(serial_rows(), serial=True)
        from concurrent.futures import ProcessPoolExecutor

        payloads = [(self.config, index, row) for index, row in enumerate(self._rows)]
        chunk = max(1, len(payloads) // (jobs * 8))
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            return self._assemble(
                pool.map(_plan_row_worker, payloads, chunksize=chunk), serial=False, jobs=jobs
            )

    def _assemble(
        self,
        stream: Iterable[tuple[_RowPlan | _SkipRow, _IndexCounters]],
        *,
        serial: bool,
        jobs: int = 1,
    ) -> _DatasetPlan:
        """把逐行结果按行序拼装成索引（串行 / 并行**共用**，保证两条路径一致）。"""
        counters = _IndexCounters()
        entries: list[_WindowEntry] = []
        for row_index, (planned, row_counters) in enumerate(stream):
            counters.merge(row_counters)
            counters.n_rows += 1
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
        if not serial:
            logger.info("索引构建：并行 %d 进程（读数与串行逐位一致）", jobs)
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

    def _plan_row(
        self,
        row_index: int,
        row: PairRow,
        counters: _IndexCounters,
    ) -> _RowPlan | _SkipRow:
        """一行的窗口规划；跳过时返回原因（每一类都有计数器）。

        走 :meth:`_chart_for`（带 LRU）：索引期解析出的谱面**留在缓存里**，
        于是训练的第一个批次不必把同一张谱再解析一遍。
        """
        config = self.config
        chart = self._chart_for(row_index, row)
        if isinstance(chart, _SkipRow):
            return chart
        if len(chart.lines) > config.k_max:
            return _SkipRow("too_many_lines")
        grid = FieldGrid(x_bins=config.x_bins).for_chart(
            chart,
            tau_end_s=self._axis_end_s(chart, row, counters=counters),
        )
        if grid.t_bins < config.t_window:
            return _SkipRow("no_windows")
        counters.notes_non_scorable_dropped += self._dropped_non_scorable.get(row_index, 0)
        counters.events_beyond_tau_end += _count_events_beyond_axis(
            chart,
            grid.bpm_points,
            grid.t_bins,
        )
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

    def _build_sample(self, row: PairRow, entry: _WindowEntry, *, row_index: int) -> PairSample:
        """构建一个窗口样本（纯函数式：全部输入来自 config / 行 / 索引项）。

        `row_index` 只用于**行级 LRU 的键**（不影响输出：同一行两次取值逐位一致）。
        """
        config = self.config
        chart = self._chart_for(row_index, row)
        if isinstance(chart, _SkipRow):  # 索引期已通过；文件在两次访问之间变化才会到这里
            raise RuntimeError(
                f"索引期可用的行在取样本时失败（{chart.reason}）：chart_id={row.chart_id}",
            )
        embedding = self._embedding(row)
        # 与索引期**同一个**终点口径（显式覆盖 > policy），否则窗口定位会对不上。
        grid = FieldGrid(x_bins=config.x_bins).for_chart(
            chart,
            tau_end_s=self._axis_end_s(chart, row, counters=None),
        )
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
            granularity=config.occlusion_granularity,
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
    granularity: Granularity = "event",
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
        counts.unsqueeze(0),
        ratio=ratio,
        granularity=granularity,
        seed=seed % _SEED_MODULUS,
    )
    occlusion = batch.squeeze(0)
    if occluded_event_share(counts, occlusion) < 1.0:
        return occlusion, 1, False
    for attempt in range(1, MAX_OCCLUSION_SEED_ATTEMPTS):
        candidate_seed = (seed + attempt * _OCCLUSION_SEED_STEP) % _SEED_MODULUS
        candidate, _stats = build_occlusion_batch(
            counts.unsqueeze(0),
            ratio=ratio,
            granularity=granularity,
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


@dataclass(frozen=True, slots=True)
class SongWindows:
    """一首歌按**训练同一口径**切成的推理窗口（无遮盖、无目标；RFC-0039 R3）。

    为什么要有这一层：端到端生成要「整首歌 → 一串窗口 → 逐窗前向 → 拼回整谱」。窗口的
    **秒区间 / 网格身份 / 事件轨采样点**三件事在训练路径上各自已有一处实现
    （`_window_grid` / `_slice_audio` / `line_tracks_at`），生成端重写一遍就等于造出第二份
    换算——那正是 POSTMORTEM 的形态（红线 7：换算只许在 `field/` 内实现一次）。

    与 :class:`PairSample` 的差别只有一条：**没有 counts / occlusion**（推理没有目标，
    `generation/sampling.sample` 自己从「全遮盖」状态开始迭代）。

    Attributes:
        chart: 模板谱面（判定线事件轨 + BPMList + 元数据；这就是 R3 的「条件」）。
        embedding: `(T_full, D)` 特征缓存数组。
        grid: **全谱**网格（τ 轴终点已按口径截断）；解码与秒换算都用它。
        difficulty: 定数条件。
        x_bins / t_window: 与训练配置一致（同网格身份才谈得上「同一口径」）。

    Note:
        末尾不足一个窗口的 τ 余量**不生成**（见 :meth:`dropped_tail_bins`），并如实记账：
        补零会造出一段没有条件的假 τ 区，解码器会在那里产出音符——比少生成 1-2 秒更糟。
    """

    chart: PhigrosChart
    embedding: NDArray[Dynamic]
    grid: FieldGrid
    difficulty: float
    x_bins: int
    t_window: int

    def n_lines(self) -> int:
        """K：判定线条数（模板谱面给出；R3 不生成线）。"""
        return len(self.chart.lines)

    def n_windows(self) -> int:
        """可生成的窗口数（末尾不足一窗的余量不计，见 `dropped_tail_bins`）。"""
        return int(self.grid.t_bins) // int(self.t_window)

    def dropped_tail_bins(self) -> int:
        """被丢弃的 τ 尾部格数（0 表示整条轴都被覆盖）。"""
        return int(self.grid.t_bins) - self.n_windows() * int(self.t_window)

    def tau_start(self, index: int) -> float:
        """第 `index` 个窗口的**绝对** τ 起点（拍）。"""
        return float(index * int(self.t_window)) * TAU_GRID_DT

    def window_grid(self, index: int) -> FieldGrid:
        """窗口局部网格（`bpm_points` 单段化；与训练窗口逐位同口径）。"""
        return _window_grid(
            int(self.x_bins), int(self.t_window), self.grid.bpm_points, self.tau_start(index)
        )

    def window_start_seconds(self, index: int) -> float:
        """窗口起点的**绝对**秒（解码出的窗内事件据此平移回整谱时间基）。"""
        return float(tau_to_seconds(self.tau_start(index), self.grid.bpm_points))

    def batch(self, index: int) -> FieldBatch:
        """第 `index` 个窗口的 :class:`FieldBatch`（B=1；counts / occlusion 均为 None）。"""
        if not 0 <= index < self.n_windows():
            raise IndexError(f"窗口下标 {index} 越界（共 {self.n_windows()} 个）")
        start = int(index) * int(self.t_window)
        grid = self.window_grid(index)
        tracks = line_tracks_at(
            self.chart,
            np.asarray(grid.tau_centers() + self.tau_start(index), dtype=np.float64),
        )
        audio, _frame_start, padded = _slice_audio(
            self.embedding,
            self.grid.bpm_points,
            self.tau_start(index),
            float(start + int(self.t_window)) * TAU_GRID_DT,
        )
        if padded > 0:
            logger.warning(
                "e2e 窗口 %d：音频越出特征缓存，补零 %d/%d 帧（检查 τ 轴终点口径）",
                index,
                padded,
                int(audio.shape[0]),
            )
        batch = FieldBatch(
            audio_emb=torch.as_tensor(audio, dtype=torch.float32).unsqueeze(0),
            frame_rate=MERT_FRAME_RATE_HZ,
            line_tracks=tracks.to(dtype=torch.float32).unsqueeze(0),
            line_mask=torch.ones(1, self.n_lines(), dtype=torch.bool),
            difficulty=torch.tensor([float(self.difficulty)], dtype=torch.float32),
            grid=grid,
        )
        batch.assert_shapes()
        return batch


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
