"""G2 的配对纪律（plan 07 §4.3）——默认 CI，无权重 / 无 GPU。

G2 的判据是「打乱标签后 loss 必须显著变差」。这句话只有在**两臂除了标签以外逐位相同**
时才有意义，因此本文件钉死三件事：

1. 打乱只在**被遮盖格子内**发生 ⇒ 模型的可见场（`counts * ~occlusion`）逐位不变；
2. 装配 G2 时用的是**遮盖批**（无遮盖时输入就是目标，输入会跟着目标一起变）；
3. G3 用的是**无遮盖**批（它的基线是全事件口径的闭式常数基线，与遮盖损失不同测度）。

历史缺陷（2026-09-27 修复）：G2 曾拿 G1 的遮盖臂当真实臂，而打乱臂是全事件目标 —— 两者
既不同批也不同损失，比值由批大小与重标定系数决定，门禁恒真（real 605.2 vs shuffled 644.6
那条记录就是这么来的）。
"""

from __future__ import annotations

import pytest
import torch

from beatmorph.infra.config.loading import config_from_mapping
from beatmorph.infra.sanity import shuffled_target_control
from beatmorph.infra.smoke import SmokeBatchSource, shuffle_hidden_counts
from beatmorph.infra.train_loop import build_gate_inputs

PROVENANCE = {
    "source": "fixtures",
    "query": "n/a",
    "fetched_at": "t",
    "purpose": "train",
    "script": "s",
    "script_version": "v1",
}

#: 合成夹具的极小预算：只验证「配对是否成立」，不追求统计功效
PAIRING_BASE = {
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
        "shuffle_samples": 4,
        "shuffle_steps": 20,
        "overfit_steps": 10,
        # 合成夹具下 `initial_head_bias=20` 会把前 20 步全花在把 λ 从极大值降下来，
        # 差距还没形成就结束（实测 bias=20 → 差距 0.4%，bias=0 → 差距 28%）。
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
        shuffled: bool = False,
        samples: int | None = None,
        min_events: int = 0,
    ):  # type: ignore[no-untyped-def]
        self.calls.append((bool(masked), samples))
        return self.inner.batch(
            masked=masked,
            shuffled=shuffled,
            samples=samples,
            min_events=min_events,
        )

    def describe(self) -> str:
        return self.inner.describe()


def test_gate_optimizer_lr_is_independent_from_training_lr() -> None:
    """门禁优化器的 lr 独立于训练 lr（plan 07 §9-14 的「真实数据预算」；默认 CI）。

    门禁只有 steps 步预算：若沿用**训练 lr**（真实配置 3e-4），模型几乎不动 ⇒
    G2（打乱对照）与 G3（优于常数基线）两臂都停在初始点附近，判据退化成恒 FAIL
    （实测 2026-09-27 第四轮：G2 差距 0.7%、G3 比基线差 68 倍）。这里钉死两条：
    ① 显式给 `gates.gate_optimizer_lr` 时，生效值就是它（并落进 stats → gates.txt 上下文）；
    ② 未给出时回落到 `optim.lr`（旧行为不变，向后兼容）。
    """
    base = dict(PAIRING_BASE)
    base["optim"] = {"lr": 3e-4, "batch_size": 1, "max_steps": 1}
    default_cfg = config_from_mapping(base)
    _, stats_default = build_gate_inputs(default_cfg, SmokeBatchSource(default_cfg, seed=0))
    assert stats_default["gate_optimizer_lr"] == pytest.approx(3e-4)

    tuned = dict(PAIRING_BASE)
    tuned["optim"] = {"lr": 3e-4, "batch_size": 1, "max_steps": 1}
    tuned["gates"] = {**PAIRING_BASE["gates"], "gate_optimizer_lr": 0.05}
    tuned_cfg = config_from_mapping(tuned)
    _, stats_tuned = build_gate_inputs(tuned_cfg, SmokeBatchSource(tuned_cfg, seed=0))
    assert stats_tuned["gate_optimizer_lr"] == pytest.approx(0.05)


def test_contrast_arms_use_contrast_initial_head_bias() -> None:
    """G1 的抬高初值**只**作用于 G1；G2/G3 走 contrast_initial_head_bias（默认 0 = 自然初始化）。

    观测点：G3 的 `model_loss`（在 `shuffle_steps` 步之后）。把对照臂初值抬高 ⇒ 积分项巨大 ⇒
    首步 loss 巨大；而只抬高 G1 的 `initial_head_bias` **不应**影响 G3。这钉死「去掉一个属于
    G1 的人为初值」确实生效（不是放宽判据，判据本身未动）。
    """
    base = dict(PAIRING_BASE)
    base["gates"] = {**PAIRING_BASE["gates"], "shuffle_steps": 1, "shuffle_samples": 4}
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


def _batch():  # type: ignore[no-untyped-def]
    cfg = config_from_mapping(PAIRING_BASE)
    source = SmokeBatchSource(cfg, seed=cfg.optim.seed)
    return cfg, source.batch(masked=True, samples=4)


def test_hidden_shuffle_is_invisible_to_the_model_input() -> None:
    """打乱后的可见场必须**逐位不变**，且事件总数守恒（只换位置，不换数量）。"""
    cfg, batch = _batch()
    shuffled = shuffle_hidden_counts(batch, seed=cfg.optim.seed + 991)
    assert torch.equal(shuffled.observed_counts(), batch.observed_counts())
    assert torch.equal(shuffled.occlusion_bool(), batch.occlusion_bool())
    assert int(shuffled.counts.sum()) == int(batch.counts.sum())
    assert not torch.equal(shuffled.counts, batch.counts)


def test_hidden_shuffle_requires_a_masked_batch() -> None:
    """无遮盖时「输入就是目标」，没有「待补全的标签」可打乱 —— 必须报错而不是静默退化。"""
    cfg = config_from_mapping(PAIRING_BASE)
    source = SmokeBatchSource(cfg, seed=cfg.optim.seed)
    unmasked = source.batch(masked=False, samples=2)
    with pytest.raises(ValueError, match="遮盖"):
        shuffle_hidden_counts(unmasked, seed=1)


def test_gate_assembly_pairs_g2_on_one_masked_batch_and_g3_unmasked() -> None:
    """装配纪律：G1 走遮盖批；G2 走**同一批**遮盖批；G3 走无遮盖批。"""
    cfg = config_from_mapping(PAIRING_BASE)
    spy = _SpySource(SmokeBatchSource(cfg, seed=cfg.optim.seed))
    build_gate_inputs(cfg, spy, seed=cfg.optim.seed)
    assert spy.calls == [
        (True, None),  # G1：训练路径（遮盖），批大小取 optim.batch_size
        (True, cfg.gates.shuffle_samples),  # G2：遮盖批（真实臂 + 由它派生的打乱臂）
        (False, cfg.gates.shuffle_samples),  # G3：全事件口径，才能与常数基线比
    ]


def test_g2_pair_discriminates_on_a_learnable_task() -> None:
    """端到端：合成夹具的**条件携带目标信息**（plan 04 §9-15），因此 G2 必须判绿。

    这条断言是「G2 不是恒真门禁」的活证据：把配对改回历史写法（真实臂 = G1 的 1 样本
    遮盖臂 / 打乱臂 = 全事件 16 样本）时它会以另一种方式失败，而把合成夹具的条件信息
    去掉（噪声上事件与条件独立）时它会**红**——两种都说明判据真的在起作用。
    """
    cfg = config_from_mapping(PAIRING_BASE)
    source = SmokeBatchSource(cfg, seed=cfg.optim.seed)
    inputs, _stats = build_gate_inputs(cfg, source, seed=cfg.optim.seed)
    result = shuffled_target_control(
        inputs.step_fn_g2_real,
        inputs.step_fn_g2_shuffled,
        steps=cfg.gates.shuffle_steps,
        min_gap_ratio=cfg.gates.shuffle_min_gap_ratio,
    )
    assert result.passed, result.detail
