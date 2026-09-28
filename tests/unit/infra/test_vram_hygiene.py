"""显存卫生回收的契约（plan 07 §9-57 事故后）——默认 CI，**不需要 CUDA**。

事故机制：一次 K≈128 的大批把 PyTorch 分配器的**保留量**顶到 ~5.2 GiB 且长期不释放，
驱动侧总量因此停在 7.86/8.15 GiB；下一次大分配无处可放 ⇒ Windows **静默回退共享显存**
（用户观测 13.2 GB），功耗 102→31 W、步时放大一个数量级以上，step 951 直接卡死。

本文件钉死：只在「保留 − 已分配」超过阈值时才回收；阈值 <= 0 或无 CUDA 时**不碰**分配器。
"""

from __future__ import annotations

import torch

from beatmorph.infra.train_loop import maintain_vram_hygiene, vram_reserved_gib

GIB = 2**30


class _FakeCuda:
    def __init__(self, *, allocated: float, reserved: float) -> None:
        self.allocated = allocated
        self.reserved = reserved
        self.calls = 0

    def is_available(self) -> bool:
        return True

    def memory_allocated(self) -> float:
        return self.allocated

    def memory_reserved(self) -> float:
        return self.reserved

    def empty_cache(self) -> None:
        self.calls += 1


def _patch(monkeypatch, fake: _FakeCuda) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", fake.is_available)
    monkeypatch.setattr(torch.cuda, "memory_allocated", fake.memory_allocated)
    monkeypatch.setattr(torch.cuda, "memory_reserved", fake.memory_reserved)
    monkeypatch.setattr(torch.cuda, "empty_cache", fake.empty_cache)


def test_reclaims_only_when_reserved_far_exceeds_allocated(monkeypatch) -> None:
    """保留 5.2 GiB / 已分配 3.0 GiB、阈值 1 GiB ⇒ 必须回收（事故的形状）。"""
    fake = _FakeCuda(allocated=3.0 * GIB, reserved=5.2 * GIB)
    _patch(monkeypatch, fake)
    assert maintain_vram_hygiene(1.0) is True
    assert fake.calls == 1


def test_does_not_touch_allocator_below_threshold(monkeypatch) -> None:
    """保留 3.3 GiB / 已分配 3.0 GiB、阈值 1 GiB ⇒ 不回收（正常步不付重分配代价）。"""
    fake = _FakeCuda(allocated=3.0 * GIB, reserved=3.3 * GIB)
    _patch(monkeypatch, fake)
    assert maintain_vram_hygiene(1.0) is False
    assert fake.calls == 0


def test_zero_threshold_disables_hygiene(monkeypatch) -> None:
    """阈值 <= 0 = 关闭（显式开关；0 不得被当成「永远回收」）。"""
    fake = _FakeCuda(allocated=0.1 * GIB, reserved=9.0 * GIB)
    _patch(monkeypatch, fake)
    assert maintain_vram_hygiene(0.0) is False
    assert fake.calls == 0


def test_no_cuda_is_a_noop(monkeypatch) -> None:
    """CPU 路径不得触碰分配器（契约：CPU 上这些读数恒为 0）。"""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert maintain_vram_hygiene(1.0) is False
    assert vram_reserved_gib() == 0.0


def test_reserved_readout(monkeypatch) -> None:
    """保留量读数 = reserved / 2**30。"""
    fake = _FakeCuda(allocated=1.0 * GIB, reserved=4.5 * GIB)
    _patch(monkeypatch, fake)
    assert vram_reserved_gib() == 4.5
