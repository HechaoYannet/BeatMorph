"""`optim.train_head_only` 的护栏（plan 07 §9-72 的诊断开关）。

为什么需要：§9-72 把现场定位到「解码器把输入里完美可分的遮盖信号混成随机」，
而 plan 04 §9-17 记录过「冻结解码器只训头部立刻到达下界，全参训却停在边际解」。
要复现那个对照，就必须有一个**只训头部**的开关，且它必须真的只训头部
（否则实验臂与控制臂的差异不再只来自「谁拿梯度」）。
"""

from __future__ import annotations

import torch

from beatmorph.generation.model import MaskedFieldModel
from beatmorph.infra.config.loading import load_config
from beatmorph.infra.train_loop import model_from_config
from tests.unit.generation._builders import TEST_AUDIO_DIM, make_grid

T_BINS = 12
X_BINS = 4


def _model() -> MaskedFieldModel:
    """构造一个最小模型，并**复刻** `train()` 里那段冻结逻辑。"""
    import pathlib
    from dataclasses import replace

    cfg = load_config(directory=pathlib.Path("configs"), config_name="phigros_masked")
    cfg = replace(
        cfg,
        data=replace(cfg.data, workers=0),
        model=replace(cfg.model, d_model=16, n_heads=2, n_layers=2, audio_dim=TEST_AUDIO_DIM),
    )
    model = model_from_config(cfg, make_grid(t_bins=T_BINS, x_bins=X_BINS), seed=0)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("head."))
    return model


def test_head_only_freezes_everything_but_the_head() -> None:
    model = _model()
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable, "一个可训参数都没有：实验臂会是空的"
    outside = sorted(name for name in trainable if not name.startswith("head."))
    assert not outside, f"除 head 之外还有可训参数：{outside}"


def test_optimizer_sees_only_the_head_and_fewer_parameters() -> None:
    model = _model()
    params = [p for p in model.parameters() if p.requires_grad]
    assert params
    optimizer = torch.optim.AdamW(params, lr=3e-4)
    assert all(p.requires_grad for group in optimizer.param_groups for p in group["params"])
    trainable = sum(p.numel() for p in params)
    everything = sum(p.numel() for p in model.parameters())
    assert 0 < trainable < everything, f"冻结没有生效：可训 {trainable} / 全部 {everything}"


def test_the_head_is_still_a_real_fraction_of_the_model() -> None:
    """头不该小到「等于没训」——否则实验臂的读数无法解释。"""
    model = _model()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    everything = sum(p.numel() for p in model.parameters())
    assert (
        trainable / everything > 0.05
    ), f"head 只占 {trainable}/{everything}：实验臂几乎训不到任何东西"
