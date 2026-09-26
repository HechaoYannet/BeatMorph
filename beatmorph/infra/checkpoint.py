"""checkpoint 保存与**恢复校验**（plan 07 §4.5 / M7.5）。

恢复语义的口径：**宁可拒绝恢复，也不用「环境已变」的实验续训出不可信结论**。
因此载入时必须逐项校验并在不一致时报出**差异字段**：

1. `config` 指纹（并给出逐字段 diff，而不是只报「哈希不同」）；
2. `gates.txt` 的存在性与全绿结论（保存时快照进 meta）；
3. `data_rev`（数据清单内容哈希）—— 数据换了还续训等于把两次实验混成一个。

大文件不入库（红线 5）：`runs/` 已在 `.gitignore`，本模块只负责读写与校验。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from beatmorph.core.logging import get_logger
from beatmorph.infra.config.schema import TrainConfig, config_to_dict

__all__ = [
    "CheckpointMeta",
    "LoadedCheckpoint",
    "ResumeMismatchError",
    "config_diff",
    "config_fingerprint",
    "flatten_config",
    "load_checkpoint",
    "save_checkpoint",
]

logger = get_logger("infra.checkpoint")


class ResumeMismatchError(RuntimeError):
    """恢复被拒绝（配置 / 门禁 / 数据版本不一致）。"""


def flatten_config(payload: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """把嵌套配置拍平成 `a.b.c -> value`（diff 与指纹都用它）。"""
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten_config(value, prefix=f"{path}."))
        else:
            flat[path] = value
    return flat


def config_fingerprint(cfg: TrainConfig) -> str:
    """配置指纹（规范化 JSON 的 sha256；键序无关）。"""
    canonical = json.dumps(config_to_dict(cfg), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def config_diff(stored: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    """两份配置的逐字段差异（人类可读；按字段名排序）。"""
    left = flatten_config(stored)
    right = flatten_config(current)
    lines: list[str] = []
    for key in sorted(set(left) | set(right)):
        before = left.get(key, "<缺失>")
        after = right.get(key, "<缺失>")
        if before != after:
            lines.append(f"{key}: {before!r} -> {after!r}")
    return lines


@dataclass(frozen=True, slots=True)
class CheckpointMeta:
    """checkpoint 的元数据（恢复校验的唯一依据）。"""

    step: int
    config_hash: str
    data_rev: str
    git_rev: str
    gates_green: bool
    config: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    """一次成功的恢复。"""

    meta: CheckpointMeta
    model_state: Mapping[str, Any]
    optimizer_state: Mapping[str, Any] | None


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    cfg: TrainConfig,
    step: int,
    data_rev: str,
    git_rev: str,
    gates_green: bool,
    optimizer: torch.optim.Optimizer | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """保存 checkpoint（权重 + 可选优化器状态 + 元数据）。"""
    payload: dict[str, Any] = {
        "meta": {
            "step": int(step),
            "config_hash": config_fingerprint(cfg),
            "data_rev": str(data_rev),
            "git_rev": str(git_rev),
            "gates_green": bool(gates_green),
            "config": config_to_dict(cfg),
        },
        "model_state": model.state_dict(),
        "optimizer_state": None if optimizer is None else optimizer.state_dict(),
    }
    if extra:
        payload["extra"] = dict(extra)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    logger.info("已保存 checkpoint %s（step=%d）", target, step)
    return target


def load_checkpoint(
    path: Path,
    *,
    cfg: TrainConfig,
    data_rev: str | None = None,
    require_gates_green: bool = True,
    allow_data_rev_change: bool = False,
) -> LoadedCheckpoint:
    """载入并**校验** checkpoint；任一不一致都拒绝恢复。

    Args:
        path: checkpoint 文件。
        cfg: 当前配置（与保存时的配置逐字段比对）。
        data_rev: 当前数据版本；None 表示不校验数据版本。
        require_gates_green: 是否要求保存时门禁为全绿。
        allow_data_rev_change: 显式放行数据版本变化（**必须**由调用方写明理由）。

    Raises:
        ResumeMismatchError: 配置 / 门禁 / 数据版本不一致，或文件不可读。
    """
    target = Path(path)
    if not target.is_file():
        raise ResumeMismatchError(f"checkpoint 不存在：{target}")
    try:
        payload = torch.load(target, map_location="cpu", weights_only=True)
    except Exception as exc:  # torch 的反序列化异常族
        raise ResumeMismatchError(f"{target} 无法载入：{type(exc).__name__}: {exc}") from exc
    raw_meta = payload.get("meta")
    if not isinstance(raw_meta, Mapping):
        raise ResumeMismatchError(f"{target} 缺少 meta：无法校验恢复条件")
    meta = CheckpointMeta(
        step=int(raw_meta.get("step", -1)),
        config_hash=str(raw_meta.get("config_hash", "")),
        data_rev=str(raw_meta.get("data_rev", "")),
        git_rev=str(raw_meta.get("git_rev", "")),
        gates_green=bool(raw_meta.get("gates_green", False)),
        config=dict(raw_meta.get("config", {})),
    )
    problems: list[str] = []
    current_hash = config_fingerprint(cfg)
    if current_hash != meta.config_hash:
        differences = config_diff(meta.config, config_to_dict(cfg))
        problems.append(
            "配置与 checkpoint 不一致（config.yaml 哈希不同）："
            + ("；".join(differences) if differences else "字段值全同但指纹不同（序列化异常）"),
        )
    if require_gates_green and not meta.gates_green:
        problems.append("保存该 checkpoint 时 gates.txt 不是全绿：拒绝从不可信实验续训")
    if data_rev is not None and not allow_data_rev_change and str(data_rev) != meta.data_rev:
        problems.append(f"data_rev 不一致：checkpoint={meta.data_rev} 当前={data_rev}")
    if problems:
        raise ResumeMismatchError("拒绝恢复 " + str(target) + "：\n  - " + "\n  - ".join(problems))
    logger.info("恢复 checkpoint %s（step=%d，git=%s）", target, meta.step, meta.git_rev)
    return LoadedCheckpoint(
        meta=meta,
        model_state=dict(payload.get("model_state", {})),
        optimizer_state=payload.get("optimizer_state"),
    )
