"""`window_shuffle` 的护栏：**轮转结构不变、前缀变成无偏切片**（plan 07 §9-70）。

为什么需要这条测试：旧顺序下第 k 轮发的全是各谱第 k 窗 ⇒ 前「谱面数」个槽位**只含每张谱的
第 0 窗**（真实语料里那正是前奏/空窗：实测前 8000 步 61.5% 空窗、4.40 事件/窗，而全库是
10.5% / 13.46）。修法是「每轮每谱发它自己的随机一块」——**必须同时保住 S4 的覆盖率性质**
（全部谱面在约「谱面数」步内各被访问一次），否则就是拿覆盖率换分布。本文件两条都钉住。
"""

from __future__ import annotations

from beatmorph.data.plan import WindowPlan, plan_epoch


class _Stub:
    """最小 PlanSource：单个桶、`n_rows` 张谱、每张 `per_row` 个窗口（窗口按行内顺序编号）。"""

    def __init__(self, n_rows: int, per_row: int) -> None:
        self.keys: list[tuple[int, int, float]] = []
        self.rows: list[int] = []
        self.window_index: list[int] = []
        for row in range(n_rows):
            for k in range(per_row):
                self.keys.append((8, 4, 100.0))
                self.rows.append(row)
                self.window_index.append(k)

    def __len__(self) -> int:
        return len(self.keys)

    def grid_key(self, index: int) -> tuple[int, int, float]:
        return self.keys[index]

    def window_row_index(self, index: int) -> int:
        return self.rows[index]


def _plan(source: _Stub, *, shuffle: bool, seed: int = 5) -> WindowPlan:
    return plan_epoch(source, seed=seed, epoch=0, chunk=1, batch_size=1, window_shuffle=shuffle)


def test_both_modes_are_permutations() -> None:
    for shuffle in (False, True):
        source = _Stub(6, 4)
        plan = _plan(source, shuffle=shuffle)
        assert sorted(int(i) for i in plan.order) == list(range(len(source)))


def test_chart_coverage_is_preserved_by_the_shuffle() -> None:
    """轮一遍（= 谱面数）个槽位必须覆盖**全部**谱面 —— 这是 S4 不许被换掉的保证。"""
    source = _Stub(6, 4)
    for shuffle in (False, True):
        plan = _plan(source, shuffle=shuffle)
        _seen, charts = plan.coverage_at(6)
        assert charts == 6.0, f"shuffle={shuffle} 时 6 个槽位只覆盖了 {charts} 张谱"


def test_without_shuffle_the_prefix_is_only_the_first_window() -> None:
    """旧行为（默认）：前「谱面数」个槽位全部是每张谱的第 0 窗 —— 这就是被实测抓到的偏置。"""
    source = _Stub(6, 4)
    plan = _plan(source, shuffle=False)
    prefix = {source.window_index[int(i)] for i in plan.order[:6]}
    assert prefix == {0}


def test_shuffle_spreads_the_prefix_over_window_ordinals() -> None:
    """新行为：前「谱面数」个槽位覆盖多个窗口序号（不再全是第 0 窗）。"""
    source = _Stub(6, 8)
    plan = _plan(source, shuffle=True)
    prefix = [source.window_index[int(i)] for i in plan.order[:6]]
    assert len(set(prefix)) > 1, f"前缀仍然只发同一个窗口序号：{sorted(set(prefix))}"


def test_shuffle_is_deterministic_and_seed_dependent() -> None:
    source = _Stub(6, 8)
    first = _plan(source, shuffle=True, seed=11)
    again = _plan(source, shuffle=True, seed=11)
    other = _plan(source, shuffle=True, seed=12)
    assert [int(i) for i in first.order] == [int(i) for i in again.order]
    assert [int(i) for i in first.order] != [int(i) for i in other.order]


def test_off_by_default_keeps_the_old_order_bit_for_bit() -> None:
    """默认关闭时逐位等于旧顺序 —— 生产行为不被这次改动影响。"""
    source = _Stub(5, 3)
    explicit = plan_epoch(source, seed=7, epoch=0, chunk=1, batch_size=1, window_shuffle=False)
    default = plan_epoch(source, seed=7, epoch=0, chunk=1, batch_size=1)
    assert [int(i) for i in explicit.order] == [int(i) for i in default.order]
