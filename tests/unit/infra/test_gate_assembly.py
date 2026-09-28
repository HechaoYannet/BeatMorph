"""门禁装配纪律（plan 07 §4.3；RFC-0037 起为 G1/G3/G4 三道）——默认 CI，无权重 / 无 GPU。

本文件钉死三件事：

1. 门禁优化器的 lr 独立于训练 lr（预算内必须真的收敛，否则 G3 恒 FAIL）；
2. G1 的抬高初值**只**作用于 G1；G3 走 `contrast_initial_head_bias`（自然初始化）；
3. 装配纪律：G1 走**遮盖批**（训练路径），G3 走**无遮盖批**（它的基线是全事件口径的
   闭式常数基线，与遮盖损失不同测度）。

（前身 `test_g2_pairing.py` 里 G2 的配对测试随 RFC-0037 删除 G2 一并退役；
「打乱只在遮盖集合内、输入逐位不变」的纪律由 `test_shuffle_within_line.py`
在 val 置换对照的新口径上继续钉死。）
"""

from __future__ import annotations

import pytest

from beatmorph.infra.config.loading import config_from_mapping
from beatmorph.infra.smoke import SmokeBatchSource
from beatmorph.infra.train_loop import build_gate_inputs

PROVENANCE = {
    "source": "fixtures",
    "query": "n/a",
    "fetched_at": "t",
    "purpose": "train",
    "script": "s",
    "script_version": "v1",
}

#: 合成夹具的极小预算：只验证装配纪律，不追求统计功效
ASSEMBLY_BASE = {
    "data": {
        "source": "synthetic",
        "max_samples": 4,
        "occlusion_ratio": 0.5,
        "k_max": 8,
        "provenance": PROVENANCE,
    },
    "model": {
        "d_model": 32,
        "n_heads": 2,
        "n_layers": 2,
        "window": 4,
        "global_period": 2,
        "k_max": 8,
        "audio_dim": 16,
    },
    "gates": {
        "baseline_samples": 4,
        "baseline_steps": 20,
        "overfit_steps": 10,
        # 合成夹具下 `initial_head_bias=20` 会把前 20 步全花在把 λ 从极大值降下来。
        # 这条只影响本测试的**预算**，与门禁判据无关（判据是相对差距）。
        "initial_head_bias": 0.0,
    },
}


class _SpySource:
    """记录每次 `batch()` 的实参（装配纪律的观测点）。"""

    def __init__(self, inner: SmokeBatchSource) -> None:
        self.inner = inner
        self.calls: list[tuple[bool, int | None]] = []

    def batch(
        self,
        *,
        masked: bool,
        samples: int | None = None,
        min_events: int = 0,
    ):  # type: ignore[no-untyped-def]
        self.calls.append((bool(masked), samples))
        return self.inner.batch(
            masked=masked,
            samples=samples,
            min_events=min_events,
        )

    def describe(self) -> str:
        return self.inner.describe()


def test_gate_optimizer_lr_is_independent_from_training_lr() -> None:
    """门禁优化器的 lr 独立于训练 lr（plan 07 §9-14 的「真实数据预算」；默认 CI）。

    门禁只有 steps 步预算：若沿用**训练 lr**（真实配置 3e-4），模型几乎不动 ⇒
    G3（优于常数基线）停在初始点附近，判据退化成恒 FAIL（实测 2026-09-27 第四轮：
    G3 比基线差 68 倍）。这里钉死两条：
    ① 显式给 `gates.gate_optimizer_lr` 时，生效值就是它（并落进 stats → gates.txt 上下文）；
    ② 未给出时回落到 `optim.lr`（旧行为不变，向后兼容）。
    """
    base = dict(ASSEMBLY_BASE)
    base["optim"] = {"lr": 3e-4, "batch_size": 1, "max_steps": 1}
    default_cfg = config_from_mapping(base)
    _, stats_default = build_gate_inputs(default_cfg, SmokeBatchSource(default_cfg, seed=0))
    assert stats_default["gate_optimizer_lr"] == pytest.approx(3e-4)

    tuned = dict(ASSEMBLY_BASE)
    tuned["optim"] = {"lr": 3e-4, "batch_size": 1, "max_steps": 1}
    tuned["gates"] = {**ASSEMBLY_BASE["gates"], "gate_optimizer_lr": 0.05}
    tuned_cfg = config_from_mapping(tuned)
    _, stats_tuned = build_gate_inputs(tuned_cfg, SmokeBatchSource(tuned_cfg, seed=0))
    assert stats_tuned["gate_optimizer_lr"] == pytest.approx(0.05)


def test_contrast_arm_uses_contrast_initial_head_bias() -> None:
    """G1 的抬高初值**只**作用于 G1；G3 走 contrast_initial_head_bias（默认 0 = 自然初始化）。

    观测点：G3 的 `model_loss`（在 `baseline_steps` 步之后）。把基线臂初值抬高 ⇒ 积分项巨大 ⇒
    末步 loss 巨大；而只抬高 G1 的 `initial_head_bias` **不应**影响 G3。这钉死「去掉一个属于
    G1 的人为初值」确实生效（不是放宽判据，判据本身未动）。
    """
    base = dict(ASSEMBLY_BASE)
    base["gates"] = {**ASSEMBLY_BASE["gates"], "baseline_steps": 1, "baseline_samples": 4}
    only_g1 = {
        **base,
        "gates": {**base["gates"], "initial_head_bias": 100.0, "contrast_initial_head_bias": 0.0},
    }
    only_contrast = {
        **base,
        "gates": {**base["gates"], "initial_head_bias": 0.0, "contrast_initial_head_bias": 100.0},
    }
    cfg_g1 = config_from_mapping(only_g1)
    _, stats_g1 = build_gate_inputs(cfg_g1, SmokeBatchSource(cfg_g1, seed=0))
    cfg_contrast = config_from_mapping(only_contrast)
    _, stats_contrast = build_gate_inputs(cfg_contrast, SmokeBatchSource(cfg_contrast, seed=0))
    assert stats_g1["g3_model_loss"] < stats_contrast["g3_model_loss"] * 0.5
    assert stats_g1["g1_initial_head_bias"] == 100.0
    assert stats_contrast["contrast_initial_head_bias"] == 100.0


def test_gate_assembly_pairs_g1_masked_and_g3_unmasked() -> None:
    """装配纪律：G1 走遮盖批（训练路径）；G3 走无遮盖批（全事件口径才能与常数基线比）。"""
    cfg = config_from_mapping(ASSEMBLY_BASE)
    spy = _SpySource(SmokeBatchSource(cfg, seed=cfg.optim.seed))
    build_gate_inputs(cfg, spy, seed=cfg.optim.seed)
    assert spy.calls == [
        (True, None),  # G1：训练路径（遮盖），批大小取 optim.batch_size
        (False, cfg.gates.baseline_samples),  # G3：全事件口径，才能与常数基线比
    ]


def test_gate_arms_run_in_fp32() -> None:
    """RFC-0037 R4（原 P0）：门禁臂**固定 fp32**，即使训练精度是 bf16-mixed。

    依据：bf16 下同臂重复运行噪声 2.8×（RFC-0036 §2.3），阈值化读数不可解释。
    观测点：stats 里落盘的 `gate_precision`。
    """
    base = dict(ASSEMBLY_BASE)
    base["optim"] = {"lr": 3e-4, "batch_size": 1, "max_steps": 1, "precision": "bf16-mixed"}
    cfg = config_from_mapping(base)
    _, stats = build_gate_inputs(cfg, SmokeBatchSource(cfg, seed=0))
    assert stats["gate_precision"] == 32.0
