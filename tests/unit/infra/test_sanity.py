"""健全性门禁单元测试（纯 Python，无 torch 依赖）。

门禁本身必须是"最小环境也能跑"的——依赖缺失导致的 skip 正是 25Hz bug 存活的原因。
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from beatmorph.infra.sanity import (
    constant_baseline_gate,
    frame_rate_gate,
    overfit_single_batch,
    shuffled_target_control,
    summarize,
)


def _steps(start: float, factor: float, floor: float = 0.0) -> Iterator[float]:
    """几何衰减的 loss 序列生成器。"""
    value = start
    while True:
        yield value
        value = max(floor, value * factor)


def _step_fn(seq: Iterator[float]):
    def _step() -> float:
        return next(seq)

    return _step


class TestG1OverfitSingleBatch:
    def test_passes_when_loss_collapses(self) -> None:
        r = overfit_single_batch(_step_fn(_steps(2.0, 0.5)), steps=20)
        assert r.passed, r.detail

    def test_fails_when_loss_is_stuck(self) -> None:
        """loss 纹丝不动 = 通路坏了，必须红灯（复现历史症状）。"""
        r = overfit_single_batch(_step_fn(_steps(1.0, 1.0)), steps=20)
        assert not r.passed
        assert bool(r) is False

    def test_rejects_zero_steps(self) -> None:
        with pytest.raises(ValueError, match="steps"):
            overfit_single_batch(_step_fn(_steps(1.0, 0.5)), steps=0)


class TestG2ShuffledTargetControl:
    def test_passes_when_shuffled_is_worse(self) -> None:
        real = _step_fn(_steps(2.0, 0.5))
        shuffled = _step_fn(_steps(2.0, 0.98))
        r = shuffled_target_control(real, shuffled, steps=20)
        assert r.passed, r.detail

    def test_fails_when_shuffled_is_equally_good(self) -> None:
        """打乱标签也不变差 => 输入对目标零信息（帧率错配的签名）。"""
        real = _step_fn(_steps(2.0, 0.5))
        shuffled = _step_fn(_steps(2.0, 0.5))
        r = shuffled_target_control(real, shuffled, steps=20)
        assert not r.passed

    def test_negative_losses_do_not_false_green(self) -> None:
        """回归（实测 2026-09-27 第四轮）：真实臂 loss 为负时的「假绿」必须被关闭。

        旧式 `need = real_last * (1 + min_gap_ratio)` 在 real_last < 0 时乘以 1.05 会得到
        **更负**（= 更好）的阈值 ⇒ 只要打乱臂「稍微更好」也会 PASS，判据方向反了。
        真实数据实测：真实臂 −326.33 / 打乱臂 −327.25（打乱臂好 0.28%）——按语义必须 FAIL。
        现口径 `real_last + min_gap_ratio * |real_last|` 在 real_last < 0 时要求打乱臂**更大**
        （= 更差）才通过。
        """
        # ⚠️ `_steps` 默认 floor=0.0 会把负值夹成 0，负 loss 场景必须显式给负 floor。
        real = _step_fn(_steps(-326.33, 1.0, floor=-1e9))
        shuffled = _step_fn(_steps(-327.25, 1.0, floor=-1e9))
        r = shuffled_target_control(real, shuffled, steps=5)
        assert not r.passed, r.detail

    def test_positive_losses_keep_the_relative_formula(self) -> None:
        """real > 0 时新旧口径**逐位相同**：need == real * (1 + min_gap_ratio)。"""
        real = _step_fn(_steps(100.0, 1.0))
        just_worse = _step_fn(_steps(105.0, 1.0))  # == need => PASS（边界）
        not_enough = _step_fn(_steps(104.9, 1.0))  # < need => FAIL
        assert shuffled_target_control(real, just_worse, steps=5).passed
        assert not shuffled_target_control(real, not_enough, steps=5).passed


class TestG3ConstantBaseline:
    def test_passes_when_model_beats_mean(self) -> None:
        assert constant_baseline_gate(0.5, 1.0).passed

    def test_fails_when_model_equals_mean_predictor(self) -> None:
        assert not constant_baseline_gate(1.0, 1.0).passed

    def test_respects_min_improvement(self) -> None:
        assert not constant_baseline_gate(0.95, 1.0, min_improvement=0.1).passed
        assert constant_baseline_gate(0.89, 1.0, min_improvement=0.1).passed


class TestG4FrameRate:
    def test_passes_on_exact_75hz(self) -> None:
        assert frame_rate_gate(750, 10.0, 75.0).passed

    def test_fails_on_historical_25hz_assumption(self) -> None:
        """真实 75Hz 的 embedding 用 25Hz 去解释：必须红灯。"""
        real_frames = 7500  # 100s @75Hz
        assert not frame_rate_gate(real_frames, 100.0, 25.0).passed
        assert frame_rate_gate(real_frames, 100.0, 75.0).passed

    def test_tolerance(self) -> None:
        assert frame_rate_gate(374, 5.0, 75.0, tol_frames=2).passed
        assert not frame_rate_gate(374, 5.0, 75.0, tol_frames=0).passed


def test_summarize_reports_failures() -> None:
    results = [
        frame_rate_gate(374, 5.0, 75.0),
        constant_baseline_gate(1.0, 1.0),
    ]
    text = summarize(results)
    assert "PASS" in text
    assert "FAIL" in text
    assert "1 项未通过" in text
