"""门禁的**数据规模**必须与训练规模解耦（RFC-0033 的后果之一；默认 CI，无 GPU）。

来历：采样器修好之后，门禁批不再恰好落在最小的几个 `bpm_eff` 桶上，而是按剩余窗口
**加权随机**抽桶（RFC-0033）——抽中的是最大那批桶（K 可到 `k_max=128`）。实测门禁
**5 min → 26 min**、显存 **7880/8151 MiB**，正好落在 docs/TRAINING.md §7.5 记录的
「滑进 Windows 共享内存」危险区。因此门禁改用**有界切片**（`gates.gate_samples`），
与第五轮权威记录（`data.max_samples=200`）同口径。

本文件钉死三件事：默认有界、副本不污染训练配置、非法值被拒。
"""

from __future__ import annotations

import pytest

from beatmorph.infra.config.loading import config_from_mapping
from beatmorph.infra.config.schema import ConfigError, GatesConfig, assert_valid_config
from beatmorph.infra.gates import bounded_gate_config

PROVENANCE = {
    "source": "fixtures",
    "query": "n/a",
    "fetched_at": "t",
    "purpose": "train",
    "script": "s",
    "script_version": "v1",
}


def _config():
    return config_from_mapping(
        {
            "data": {
                "source": "manifest",
                "manifest_path": "data/processed/pairs.json",
                "chart_dir": "data/processed/charts",
                "feature_dir": "data/processed/features",
                "max_samples": None,
                "occlusion_ratio": 0.5,
                "k_max": 128,
                "provenance": PROVENANCE,
            },
            "model": {
                "d_model": 32,
                "n_heads": 2,
                "n_layers": 2,
                "window": 4,
                "global_period": 2,
                "k_max": 128,
                "audio_dim": 16,
            },
            "optim": {"lr": 1e-3, "batch_size": 1, "max_steps": 4},
        }
    )


def test_default_gate_samples_is_bounded() -> None:
    """默认必须有界——不然门禁的成本会跟着全量采样器一起失控。"""
    assert GatesConfig().gate_samples == 200


def test_bounded_gate_config_only_changes_max_samples() -> None:
    """门禁副本只改 `data.max_samples`；**训练配置本身不被污染**（全量训练不变）。"""
    cfg = _config()
    assert cfg.data.max_samples is None

    bounded = bounded_gate_config(cfg)

    assert bounded is not cfg
    assert bounded.data.max_samples == cfg.gates.gate_samples == 200
    assert cfg.data.max_samples is None, "训练配置被就地改写了"
    assert bounded.optim.max_steps == cfg.optim.max_steps
    assert bounded.data.split_train == cfg.data.split_train


def test_bounded_gate_config_is_opt_out() -> None:
    """`gate_samples=None` 时不限界（诊断用），且原样返回同一个对象。"""
    cfg = _config()
    cfg.gates.gate_samples = None
    assert bounded_gate_config(cfg) is cfg


def test_gate_samples_must_be_positive() -> None:
    """0 / 负数必须被配置校验拒掉，而不是当成「无界」。"""
    cfg = _config()
    cfg.gates.gate_samples = 0
    with pytest.raises(ConfigError):
        assert_valid_config(cfg)
