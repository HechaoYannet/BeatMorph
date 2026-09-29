"""DataLoader worker 测试用的 stub（RFC-0034 S3）。

为什么单独一个模块：Windows 上 DataLoader 走 spawn，**dataset 与 collate_fn 都要在子进程里
按「模块名 + 限定名」重新导入**。定义在测试模块里会让子进程 import 失败（pytest 把测试文件
当顶层模块加载，子进程的 sys.path 里没有那个目录）。父进程把本目录塞进 sys.path 后，
spawn 会把 sys.path 一并传给子进程，因此子进程能导入本模块。
"""

from __future__ import annotations

from collections.abc import Sequence


class IndexDataset:
    """样本 = 窗口下标本身（被测的是**取批顺序**，不是样本组装）。

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

    def __getitem__(self, index: int) -> int:
        return index


class DensityIndexDataset(IndexDataset):
    """带**廉价密度读数**的版本（RFC-0039 R2）：`window_event_count` / `window_line_count`。

    为什么单列一个子类而不是给 `IndexDataset` 加上：现有的 val 取批测试钉的是**回退口径**
    （没有密度来源时按桶首现序取，= R2 之前的行为）；给基类加上密度方法会让那些测试
    悄悄改测另一条分支，而回退分支将**再没有任何测试覆盖**。

    Args:
        buckets: 同 :class:`IndexDataset`。
        events: 逐窗口事件数（长度必须等于窗口总数）——测试**显式控制总体分布**。
        k_base: K 的基数（`k_base + index % 3`，与桶身份无关，便于测「跨桶」）。
    """

    def __init__(
        self,
        buckets: Sequence[tuple[int, int]],
        events: Sequence[int],
        *,
        k_base: int = 2,
    ) -> None:
        super().__init__(buckets)
        if len(events) != len(self.keys):
            raise ValueError(f"密度序列长度 {len(events)} != 窗口数 {len(self.keys)}")
        self.events = [int(value) for value in events]
        self.k_base = int(k_base)

    def window_event_count(self, index: int) -> int:
        return self.events[index]

    def window_line_count(self, index: int) -> int:
        return self.k_base + index % 3


def identity_collate(samples: Sequence[int]) -> list[int]:
    """`collate_field_batch` 的替身：样本组装不是本测试的对象。"""
    return list(samples)
