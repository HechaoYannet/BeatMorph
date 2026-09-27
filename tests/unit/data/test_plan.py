"""计划层不变量（RFC-0034 S1/S4；默认 CI，无权重 / 无 GPU / 无真实语料）。

这一层存在的理由（RFC-0034 §1.2）：修复前「顺序 / 可复现性 / 覆盖率记账」是同一个**有状态**
对象的内部状态，于是 worker 各持副本就不可复现、覆盖率还会静默失真。搬进纯函数后这三件事
同时变成可测的：

1. `order` 是窗口全集的**排列**（每个窗口恰好一次 ⇒ 不再饱和）；
2. **同批同网格**（`FieldBatch` 只带一个网格，批次不得跨桶连续段）；
3. 覆盖率 = **顺序前缀的纯函数**（与 worker 数无关，且与暴力去重逐位一致）；
4. 桶交错**按窗口数成比例**（RFC-0033 的公平性：大桶不能被小桶挤掉）；
5. **谱面分层**（S4）：全部谱面在「轮一遍」之前不会被重复轮到——`chunk=1` 时因此**严格优于**
   修复前的随机撒点（实测 20,000 步只碰到 91.3% 的谱面）；
6. `chunk>1` 把解析摊薄到 `chunk` 个窗口一次（同谱连续出现），代价是「轮一遍」的步数 ×`chunk`。
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from beatmorph.data.plan import (
    DEFAULT_PLAN_CHUNK,
    PlanBatchSampler,
    WindowPlan,
    plan_epoch,
)


class _Stub:
    """满足 `PlanSource` 的最小视图（不解析任何谱面）。

    Args:
        buckets: 每个桶的 `(行数, 每行窗口数)`；桶身份取 `(8, 4, 100+b)`。
    """

    def __init__(self, buckets: Sequence[tuple[int, int]]) -> None:
        self.keys: list[tuple[int, int, float]] = []
        self.rows: list[int] = []
        for bucket, (n_rows, per_row) in enumerate(buckets):
            for row in range(n_rows):
                for _ in range(per_row):
                    self.keys.append((8, 4, 100.0 + bucket))
                    self.rows.append(1000 * bucket + row)

    def __len__(self) -> int:
        return len(self.keys)

    def grid_key(self, index: int) -> tuple[int, int, float]:
        return self.keys[index]

    def window_row_index(self, index: int) -> int:
        return self.rows[index]


def _plan(source: _Stub, *, seed: int = 3, epoch: int = 0, **kwargs: int) -> WindowPlan:
    return plan_epoch(source, seed=seed, epoch=epoch, **kwargs)


def test_order_is_a_permutation_of_all_windows() -> None:
    """每个窗口恰好出现一次——旧实现的共享游标在这里会饱和在「每个桶的前几个窗口」。"""
    source = _Stub([(1, 1), (2, 5), (5, 7), (3, 30)])
    plan = _plan(source)
    assert plan.n_windows == len(source)
    assert sorted(int(v) for v in plan.order) == list(range(len(source)))
    assert plan.charts_total == 1 + 2 + 5 + 3


def test_batches_never_cross_grid_identity() -> None:
    """任何批大小下，同一批必须来自同一个桶（否则 `collate_field_batch` 会抛网格不一致）。"""
    source = _Stub([(1, 1), (2, 5), (5, 7), (3, 30)])
    plan = _plan(source)
    for batch_size in (1, 2, 3, 5):
        batches = list(PlanBatchSampler(plan, batch_size=batch_size))
        assert batches, "采样器没有产出任何批次"
        for batch in batches:
            assert 1 <= len(batch) <= batch_size
            assert len({source.grid_key(index) for index in batch}) == 1


def test_plan_is_deterministic_and_seed_dependent() -> None:
    """同 seed 逐位一致（M7.8）；换 seed 或换 epoch 顺序不同。"""
    source = _Stub([(2, 9), (4, 9)])
    base = _plan(source, seed=11)
    again = _plan(source, seed=11)
    other_seed = _plan(source, seed=12)
    other_epoch = _plan(source, seed=11, epoch=1)
    assert [int(v) for v in base.order] == [int(v) for v in again.order]
    assert [int(v) for v in base.order] != [int(v) for v in other_seed.order]
    assert [int(v) for v in base.order] != [int(v) for v in other_epoch.order]


def test_bucket_mix_is_proportional_to_window_count() -> None:
    """桶交错按**窗口数**成比例：否则小桶被抽干、大桶几乎不动（RFC-0033 的公平性）。"""
    source = _Stub([(1, 1), (1, 1000)])  # 桶 0 只有 1 个窗口，桶 1 有 1000 个
    plan = _plan(source)
    bucket_of = {index: (1 if index > 0 else 0) for index in range(len(source))}
    first200 = [bucket_of[int(v)] for v in plan.order[:200]]
    assert first200.count(1) >= 190, f"大桶只被抽到 {first200.count(1)}/200"


def test_every_chart_is_touched_before_any_chart_repeats() -> None:
    """S4 的核心主张：`chunk=1` 时**全部谱面**在「轮一遍」内出现 ⇒ 覆盖率严格优于随机撒点。"""
    rows, per_row = 6, 96
    source = _Stub([(rows, per_row)])
    plan = _plan(source, chunk=DEFAULT_PLAN_CHUNK)
    seen, charts = plan.coverage_at(rows)
    assert charts == float(rows), f"{rows} 步内只碰到 {charts} 张谱面（应当全部轮到）"
    assert seen == float(rows)


def test_chunk_amortizes_a_parse_over_consecutive_windows() -> None:
    """`chunk>1`：同一谱面的窗口在**顺序上连续**（每 `chunk` 个窗口才需要解析一次），
    代价是「轮一遍」所需步数乘以 `chunk`。"""
    rows, per_row, chunk = 5, 40, 8
    source = _Stub([(rows, per_row)])
    plan = _plan(source, chunk=chunk)
    assert plan.chunk == chunk
    # 每个谱面第一次出现的位置彼此间隔 >= chunk（轮转发牌 ⇒ 不会连着发同一行两块）
    firsts = sorted(int(v) for v in plan.first_seen[:rows])
    assert firsts[1] - firsts[0] >= chunk, f"同一轮里两行挨得太近：{firsts[:3]}"
    # 「全部谱面轮一遍」需要 rows*chunk 步（每行每轮发一块）
    assert plan.coverage_at((rows - 1) * chunk)[1] == float(rows - 1)
    assert plan.coverage_at(rows * chunk)[1] == float(rows)


def test_coverage_at_matches_brute_force() -> None:
    """覆盖率是**前缀的纯函数**：与「前 k 个槽位的行去重计数」逐位一致（worker 数无关）。"""
    source = _Stub([(3, 11), (7, 5), (2, 40)])
    plan = _plan(source)
    rows = [int(v) for v in plan.rows]
    for k in (0, 1, 5, 37, 100, len(rows) - 1, len(rows)):
        seen, charts = plan.coverage_at(k)
        assert seen == float(k)
        assert charts == float(len(set(rows[:k])))


def test_sampler_resume_starts_at_the_requested_step() -> None:
    """`start_step>1` 直接定位（续训 O(1)），且与从头迭代到该步的结果一致。"""
    source = _Stub([(2, 6), (3, 6)])
    plan = _plan(source)
    full = list(PlanBatchSampler(plan, batch_size=2))
    resumed = list(PlanBatchSampler(plan, batch_size=2, start_step=4))
    assert resumed == full[3:]
    assert len(PlanBatchSampler(plan, batch_size=2)) == len(plan.batch_starts(2))


def test_plan_rejects_empty_index_and_bad_batch_size() -> None:
    """空索引必须抛（门禁空批那类静默失效的同类防线）；批大小 <1 必须抛。"""
    with pytest.raises(ValueError, match="0 个窗口"):
        plan_epoch(_Stub([]), seed=0, epoch=0)
    source = _Stub([(1, 2)])
    with pytest.raises(ValueError, match="batch_size"):
        _plan(source).batch_starts(0)
    with pytest.raises(ValueError, match="start_step"):
        PlanBatchSampler(_plan(source), batch_size=1, start_step=0)
