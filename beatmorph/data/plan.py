"""取批**计划层**：把「取什么」变成纯函数（RFC-0034 S1/S4）。

## 为什么需要这一层

修复前的采样器（`ManifestBatchSource`）把三件事焊在同一个**有状态**对象里：

1. **顺序**（同批必须同网格 ⇒ 分桶 + 桶内游标）；
2. **可复现性**（同 seed 逐位一致，M7.8）；
3. **覆盖率记账**（RFC-0033 的教训：数据走了多少必须在线可见）。

后果是：顺序由 `self._positions` 决定、覆盖率在取批过程中就地累加 ⇒ **worker 各持副本就
不可复现、覆盖率还会静默失真**，于是「必须进程内串行」看起来成了硬要求。它不是——这三件事
全是**顺序与记账**问题，搬到本模块后都是 **(seed, epoch, 数据集索引) 的纯函数**：

- `plan_epoch(...) -> WindowPlan`：一次性物化一个 epoch 的取批顺序，**不做任何 I/O、不解析谱面**；
- `WindowPlan.coverage_at(consumed)`：覆盖率 = 顺序前缀的纯函数（主进程算，worker 不参与）；
- `PlanBatchSampler`：把 plan 切成批，供 `torch.utils.data.DataLoader` 使用。

## 顺序怎么排（S4：谱面分层的块调度）

原来桶内是**纯随机交错**（每个窗口一个独立随机键）。那有两处浪费：

1. 每步都换一张谱 ⇒ 每步都要重解析一张 3.8 MB 的 RJSON（实测占每窗口 CPU 的 46%，
   其中 pydantic 校验 77.5%、json.loads 21.7%、读文件 0.8%）；
2. 20,000 步只碰到 **91.3%** 的谱面——随机撒点必然漏掉一批小谱。

现在桶内改为 **按谱面行聚簇 + 轮转发牌**：

- 每个桶内按 `window_row_index` 把窗口分到「谱面行」，行序按种子洗牌；
- 每行的窗口切成**块**（长度 `chunk`，默认 1），**按轮次逐行发牌**（每一轮每个行发一块）
  ⇒ 任何谱面在「其它谱面都轮过一遍」之前不会被轮到第二次；
- 块之间用「桶偏移 + 块序」当全局排序键 ⇒ 桶按窗口数成比例交错（RFC-0033 的公平性），
  且**桶内发牌顺序原样保留**（同桶的块按块序出现）。

于是 `chunk=1` 时**谱面覆盖率严格优于**原实现（全部谱面在约「谱面数」步内出现，而随机撒点
20,000 步只有 91.3%），且不改变任何样本内容；块长大于 1 时再把解析摊薄到 `chunk` 个窗口一次
（代价是「全部谱面轮一遍」所需步数乘以 `chunk`）。

## 边界（本模块**不**做的事）

- 不解析谱面、不读特征、不构造样本（那是物化层 `ChartPairDataset.__getitem__` 的事）；
- 不持有任何跨调用状态（同一份 plan 可以被任意多个 worker 消费，结果逐位一致）。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

#: 默认发牌块长（1 = 只做谱面分层调度，不摊薄解析）。
#:
#: 为什么默认 1：块长大于 1 会把「全部谱面轮一遍」所需步数放大 `chunk` 倍，而 S3 的 worker
#: 并行已经把数据侧藏到计算后面 ⇒ 在拿到「worker 并行后还差多少」的实测数字之前，不应付出
#: 覆盖率代价。改这个值是**取批顺序的语义变更**，须按 RFC-0034 第 9 节的口径实测后再定。
DEFAULT_PLAN_CHUNK: int = 1

__all__ = [
    "DEFAULT_PLAN_CHUNK",
    "PlanBatchSampler",
    "PlanSource",
    "WindowDensitySource",
    "WindowPlan",
    "plan_epoch",
]


@runtime_checkable
class PlanSource(Protocol):
    """计划层需要的数据集视图：**只读、廉价、不解析谱面**。

    `ChartPairDataset` 满足它；单测用 stub 满足它（无需真实语料）。
    """

    def __len__(self) -> int:
        """窗口总数。"""
        ...

    def grid_key(self, index: int) -> tuple[int, int, float]:
        """第 `index` 个窗口的网格身份 `(x_bins, t_window, bpm_eff)`（同批必须同身份）。"""
        ...

    def window_row_index(self, index: int) -> int:
        """第 `index` 个窗口所属的 split 内**行**下标（谱面分层与覆盖率记账用）。"""
        ...


@runtime_checkable
class WindowDensitySource(Protocol):
    """计划层的**可选**能力：廉价回答「这个窗口有多密」（RFC-0039 R2）。

    为什么单列一个协议而不并进 :class:`PlanSource`：分层抽样需要密度，取批顺序不需要它。
    把密度做成 `PlanSource` 的必需方法，会让所有只关心顺序的调用方（含测试 stub）被迫实现
    一个它们用不到、也无法真实回答的方法——那正是「接口比事实更大」的典型代价。
    """

    def window_event_count(self, index: int) -> int | None:
        """窗口内事件数；**不可廉价取得时返回 None**（不得用 0 冒充「拿不到」）。"""
        ...

    def window_line_count(self, index: int) -> int | None:
        """窗口的 K（判定线条数）；同上，拿不到就 None。"""
        ...


@dataclass(frozen=True, slots=True)
class WindowPlan:
    """一个 epoch 的取批顺序（**纯数据**，不含任何运行期状态）。

    Attributes:
        order: 按取批顺序排列的窗口下标（全集的排列；同一「桶连续段」内网格身份相同）。
        bucket_id: 与 `order` 平行：每个槽位属于哪个网格桶。
        run_end: 与 `order` 平行：该槽位所在**桶连续段**的结束下标（开区间）。
        rows: 与 `order` 平行：每个槽位所属的谱面行（覆盖率用）。
        charts_total: 本 split 里**有窗口的**行数（覆盖率分母）。
        seed: 生成该 plan 的种子（诊断用）。
        epoch: 该 plan 属于第几个 epoch（0 起）。
        chunk: 实际生效的块长（诊断用）。
        first_seen: 每个谱面行第一次出现的槽位下标（升序），供 O(log n) 覆盖率查询。
    """

    order: NDArray[np.int64]
    bucket_id: NDArray[np.int64]
    run_end: NDArray[np.int64]
    rows: NDArray[np.int64]
    charts_total: int
    seed: int
    epoch: int
    chunk: int
    first_seen: NDArray[np.int64]

    @property
    def n_windows(self) -> int:
        """本 epoch 的窗口总数（= `len(dataset)`）。"""
        return int(self.order.shape[0])

    def coverage_at(self, consumed: int) -> tuple[float, float]:
        """`consumed` 个槽位之后的 `(windows_seen, charts_seen)`——**顺序前缀的纯函数**。

        为什么必须是纯函数：覆盖率曾经在取批过程中**就地累加**，而 worker 各持一份副本 ⇒
        一旦并行就会静默失真（RFC-0033 修掉的那类失效）。改成前缀查询后 worker 完全不参与
        记账，`data.workers` 取任何值都给出同一串数字。
        """
        seen = int(np.clip(consumed, 0, self.n_windows))
        return float(seen), float(np.searchsorted(self.first_seen, seen, side="left"))

    def batch_starts(self, batch_size: int) -> NDArray[np.int64]:
        """每个批次的起始槽位（批次**不跨越桶连续段**：`FieldBatch` 只带一个网格）。"""
        if batch_size < 1:
            raise ValueError(f"batch_size 必须 >= 1，得到 {batch_size}")
        n = self.n_windows
        if n == 0:
            return np.zeros(0, dtype=np.int64)
        starts, ends = _run_bounds(self.bucket_id)
        offset = np.arange(n, dtype=np.int64) - np.repeat(starts, ends - starts)
        return np.nonzero(offset % np.int64(batch_size) == 0)[0]


def plan_epoch(
    source: PlanSource,
    *,
    seed: int,
    epoch: int,
    chunk: int = DEFAULT_PLAN_CHUNK,
    batch_size: int = 1,
    window_shuffle: bool = False,
) -> WindowPlan:
    """物化第 `epoch` 个 epoch 的取批顺序（**纯函数**：同参数逐位一致）。

    Args:
        source: 计划层视图（`__len__` / `grid_key` / `window_row_index`）。
        seed: 运行种子（与 `optim.seed` 一致）。
        epoch: epoch 序号（0 起）；不同 epoch 用不同随机流。
        chunk: 发牌块长；实际生效值为 `max(chunk, batch_size)`（块不短于一个批，
            否则批大小大于 1 时永远凑不满一批）。
        batch_size: 训练批大小（参与块长下界的理由见上）。
        window_shuffle: **【默认 False = 旧行为】** 每一轮里，每张谱面发哪一块由该谱自己的
            种子化置换决定，而不是永远从第 0 块开始。

            **为什么需要这个开关**（plan 07 §9-70 的实测）：旧顺序下第 k 轮发的全是各谱的
            第 k 块 ⇒ 前「谱面数」个槽位**只包含每张谱的第 0 窗**（前奏/空窗）。
            真实 train split 实测（`chunk=1`、6614 张谱）：前 8000 步里 **61.5% 的窗口零事件**、
            平均 **4.40 事件/窗**，而全库均匀抽样是 **10.5% / 13.46** ⇒ 训练分布在前一万步里
            被系统性换成「稀疏 + 空」的切片，模型因此先学会「输出 0」。
            打开本开关后**轮转结构与覆盖率不变**（仍是每轮每谱一块 ⇒ 约「谱面数」步覆盖全部谱面），
            只是每一轮每张谱贡献的是**随机**那一块 ⇒ 任何前缀都是全库的无偏切片。

    Returns:
        `WindowPlan`（每个窗口恰好出现一次）。

    Raises:
        ValueError: 索引里没有任何窗口。
    """
    n = len(source)
    if n <= 0:
        raise ValueError("索引里有 0 个窗口：无法规划取批顺序")
    block = max(int(chunk), 1, int(batch_size))

    grouped: dict[tuple[int, int, float], list[int]] = {}
    for index in range(n):
        grouped.setdefault(source.grid_key(index), []).append(index)
    keys = sorted(grouped)

    seq = np.random.SeedSequence([int(seed), int(epoch)])
    rng_rows, rng_offset, rng_pieces = (np.random.default_rng(child) for child in seq.spawn(3))
    # 每个桶一个随机偏移：同一轮内桶的顺序因此随机（否则永远是「桶 0 的行全在前」）。
    offsets = rng_offset.random(len(keys))

    order = np.empty(n, dtype=np.int64)
    rows = np.empty(n, dtype=np.int64)
    #: 每个**块**的 (轮次, 轮内排序键, 起始槽位, 块结束槽位)。
    blocks: list[tuple[int, float, int, int]] = []
    cursor = 0
    for bucket, key in enumerate(keys):
        by_row: dict[int, list[int]] = {}
        for index in grouped[key]:
            by_row.setdefault(source.window_row_index(index), []).append(index)
        row_list = sorted(by_row)
        # 每行的块列表；`window_shuffle` 时按该行自己的种子化置换重排（**轮转结构不变**：
        # 第 k 轮仍然是「每行一块」，只是第 k 轮每行发它自己的第 perm[k] 块）。
        # ⚠️ RNG 消费顺序必须与 row_list 的**排序**绑定，不能用洗牌后的顺序 —— 否则
        # 开关一开就换了另一个随机流，实验臂与控制臂的差异不再只来自「发哪一块」。
        per_rows_in_order: list[list[list[int]]] = []
        for row in row_list:
            pieces = [by_row[row][k : k + block] for k in range(0, len(by_row[row]), block)]
            if window_shuffle:
                pieces = [pieces[int(j)] for j in rng_pieces.permutation(len(pieces))]
            per_rows_in_order.append(pieces)
        per_row = [per_rows_in_order[ri] for ri in rng_rows.permutation(len(row_list))]
        for k in range(max(len(item) for item in per_row)):  # 轮次
            for position, piece in enumerate(per_row):  # 行（已按种子洗牌）
                if k >= len(piece):
                    continue
                start = cursor
                for index in piece[k]:
                    order[cursor] = index
                    rows[cursor] = source.window_row_index(index)
                    cursor += 1
                blocks.append((k, float(offsets[bucket] + position), start, cursor))
    assert cursor == n, f"计划层漏掉了窗口：{cursor} != {n}"

    # 2) 全局**按轮次**发牌：第 k 轮访问每个桶的每个行恰好一次。
    #    为什么不是「按窗口数分槽位」：那样「窗口多、谱面少」的桶会反复访问同一张谱，
    #    同时饿死窗口少的桶 —— 实测（真实 train split，chunk=1）20,000 步只覆盖
    #    **52.6%** 的谱面，而旧的均匀撒点是 91.3%。按轮次则「全部谱面各访问一次」只需
    #    Σ_桶(行数) ≈ 谱面数 个槽位。
    new_order = np.empty(n, dtype=np.int64)
    new_rows = np.empty(n, dtype=np.int64)
    write = 0
    for _round, _key, start, stop in sorted(blocks):
        length = stop - start
        new_order[write : write + length] = order[start:stop]
        new_rows[write : write + length] = rows[start:stop]
        write += length

    bucket_id = _bucket_ids(new_order, source)
    first_seen = _first_seen_positions(new_rows, n)
    return WindowPlan(
        order=new_order,
        bucket_id=bucket_id,
        run_end=_run_ends(bucket_id),
        rows=new_rows,
        charts_total=int((first_seen < n).sum()),
        seed=int(seed),
        epoch=int(epoch),
        chunk=block,
        first_seen=first_seen,
    )


def _bucket_ids(order: NDArray[np.int64], source: PlanSource) -> NDArray[np.int64]:
    """`order` 中每个槽位的桶编号（桶号只用于判断「同批同网格」）。"""
    mapping: dict[tuple[int, int, float], int] = {}
    out = np.empty(order.shape[0], dtype=np.int64)
    for position, index in enumerate(order):
        key = source.grid_key(int(index))
        bucket = mapping.get(key)
        if bucket is None:
            bucket = len(mapping)
            mapping[key] = bucket
        out[position] = bucket
    return out


def _run_bounds(bucket_id: NDArray[np.int64]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """桶连续段的起始下标与结束下标（开区间）。"""
    n = int(bucket_id.shape[0])
    change = np.empty(n, dtype=bool)
    change[0] = True
    np.not_equal(bucket_id[1:], bucket_id[:-1], out=change[1:])
    starts = np.nonzero(change)[0]
    return starts, np.append(starts[1:], np.int64(n))


def _run_ends(bucket_id: NDArray[np.int64]) -> NDArray[np.int64]:
    """每个槽位所在桶连续段的结束下标（开区间）——批次切分按它收口。"""
    starts, ends = _run_bounds(bucket_id)
    return np.repeat(ends, ends - starts)


def _first_seen_positions(rows: NDArray[np.int64], n: int) -> NDArray[np.int64]:
    """每个谱面行第一次出现的槽位下标（升序；未出现的行记为 `n`）。

    覆盖率查询因此是 `searchsorted`（O(log n)），而不是每次重算去重（O(n log n)）。
    """
    total = int(rows.max()) + 1
    first = np.full(total, np.int64(n), dtype=np.int64)
    np.minimum.at(first, rows, np.arange(n, dtype=np.int64))
    first.sort()
    return first


class PlanBatchSampler:
    """把 `WindowPlan` 切成批次（`DataLoader` 的 duck-typed `batch_sampler`）。

    为什么不继承 `torch.utils.data.Sampler`：本模块因此**不 import torch**，
    计划层可以在无 torch 的环境（含默认 CI）里独立验证。

    Args:
        plan: 取批计划。
        batch_size: 批大小（>=1）。
        start_step: 从第几步开始（1 起；续训用）——直接索引 `plan.batch_starts`，
            因此续训是 O(1) 定位，不重放前面的批次。
    """

    def __init__(self, plan: WindowPlan, *, batch_size: int, start_step: int = 1) -> None:
        if start_step < 1:
            raise ValueError(f"start_step 必须 >= 1，得到 {start_step}")
        self._plan = plan
        self._batch_size = int(batch_size)
        self._starts = plan.batch_starts(batch_size)
        self._first = int(start_step) - 1

    @property
    def plan(self) -> WindowPlan:
        """底层计划（诊断用）。"""
        return self._plan

    def spans(self) -> Iterator[tuple[int, int]]:
        """每个批次的 `(起始槽位, 结束槽位)`（开区间）。

        为什么要单独暴露它：取批路径要靠**槽位**记账（覆盖率 = 前缀长度），而不能靠
        collate 出来的批次对象反推——那样等于把「记账」依赖到「物化」上，正是本 RFC 要拆开的耦合。
        """
        starts = self._starts
        run_end = self._plan.run_end
        for position in range(self._first, int(starts.shape[0])):
            start = int(starts[position])
            yield start, min(start + self._batch_size, int(run_end[start]))

    def __iter__(self) -> Iterator[list[int]]:
        order = self._plan.order
        for start, stop in self.spans():
            yield [int(value) for value in order[start:stop]]

    def __len__(self) -> int:
        return max(0, int(self._starts.shape[0]) - self._first)
