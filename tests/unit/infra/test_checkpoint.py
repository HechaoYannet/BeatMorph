"""M7.5 / M7.8：checkpoint 保存与恢复校验（配置 / 门禁 / 数据版本任一不符即拒绝）。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from beatmorph.infra.checkpoint import (
    ResumeMismatchError,
    config_diff,
    config_fingerprint,
    load_checkpoint,
    save_checkpoint,
)
from beatmorph.infra.config.loading import config_from_mapping

PROVENANCE = {
    "source": "fixtures",
    "query": "n/a",
    "fetched_at": "t",
    "purpose": "train",
    "script": "s",
    "script_version": "v1",
}
BASE = {"data": {"source": "synthetic", "max_samples": 4, "provenance": PROVENANCE}}


def _cfg(**overrides: object):
    return config_from_mapping({**BASE, **overrides})


def _model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Linear(4, 2)


def _save(tmp_path: Path, cfg, **overrides: object) -> Path:
    kwargs: dict[str, object] = {
        "model": _model(),
        "cfg": cfg,
        "step": 3,
        "data_rev": "sha1:abc",
        "git_rev": "deadbee",
        "gates_green": True,
    }
    kwargs.update(overrides)
    return save_checkpoint(tmp_path / "step.pt", **kwargs)  # type: ignore[arg-type]


def test_round_trip_restores_state(tmp_path: Path) -> None:
    """正常恢复：权重逐位一致、元数据保留。"""
    cfg = _cfg()
    model = _model()
    path = save_checkpoint(
        tmp_path / "ckpt.pt",
        model=model,
        cfg=cfg,
        step=7,
        data_rev="sha1:abc",
        git_rev="deadbee",
        gates_green=True,
        optimizer=torch.optim.Adam(model.parameters()),
    )
    loaded = load_checkpoint(path, cfg=cfg, data_rev="sha1:abc")
    assert loaded.meta.step == 7
    assert loaded.optimizer_state is not None
    for key, value in model.state_dict().items():
        assert torch.equal(loaded.model_state[key], value)


def test_tampered_config_is_refused_with_field_diff(tmp_path: Path) -> None:
    """篡改配置后恢复必须被拒绝，并**报出差异字段**（不是只说哈希不同）。"""
    path = _save(tmp_path, _cfg())
    changed = _cfg(optim={"lr": 0.123})
    with pytest.raises(ResumeMismatchError) as excinfo:
        load_checkpoint(path, cfg=changed)
    message = str(excinfo.value)
    assert "optim.lr" in message
    assert "0.123" in message


def test_gates_not_green_is_refused(tmp_path: Path) -> None:
    """从「门禁没绿」的实验续训等于把不可信结论当基线。"""
    path = _save(tmp_path, _cfg(), gates_green=False)
    with pytest.raises(ResumeMismatchError, match="gates"):
        load_checkpoint(path, cfg=_cfg())
    load_checkpoint(path, cfg=_cfg(), require_gates_green=False)


def test_data_rev_change_is_refused_unless_explicit(tmp_path: Path) -> None:
    """数据换了还续训 = 把两次实验混成一个；放行必须由调用方显式写明。"""
    path = _save(tmp_path, _cfg())
    with pytest.raises(ResumeMismatchError, match="data_rev"):
        load_checkpoint(path, cfg=_cfg(), data_rev="sha1:other")
    loaded = load_checkpoint(path, cfg=_cfg(), data_rev="sha1:other", allow_data_rev_change=True)
    assert loaded.meta.data_rev == "sha1:abc"


def test_missing_or_broken_file_is_refused(tmp_path: Path) -> None:
    """文件不存在 / 不是 checkpoint：都以 ResumeMismatchError 收场。"""
    with pytest.raises(ResumeMismatchError, match="不存在"):
        load_checkpoint(tmp_path / "nope.pt", cfg=_cfg())
    broken = tmp_path / "broken.pt"
    broken.write_text("not a torch file", encoding="utf-8")
    with pytest.raises(ResumeMismatchError):
        load_checkpoint(broken, cfg=_cfg())


def test_fingerprint_is_order_independent_and_diff_is_fieldwise() -> None:
    """指纹按规范化 JSON 计算（键序无关）；diff 逐字段可比。"""
    cfg = _cfg()
    assert config_fingerprint(cfg) == config_fingerprint(_cfg())
    assert config_fingerprint(cfg) != config_fingerprint(_cfg(optim={"lr": 0.5}))
    differences = config_diff({"a": {"b": 1}}, {"a": {"b": 2, "c": 3}})
    assert "a.b: 1 -> 2" in differences
    assert any("a.c" in line for line in differences)
