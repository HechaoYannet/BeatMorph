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


def identity_collate(samples: Sequence[int]) -> list[int]:
    """`collate_field_batch` 的替身：样本组装不是本测试的对象。"""
    return list(samples)
