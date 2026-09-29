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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from beatmorph.core.logging import get_logger
from beatmorph.infra.config.schema import TrainConfig, config_to_dict

__all__ = [
    "CHECKPOINT_PREFIX",
    "CheckpointMeta",
    "LoadedCheckpoint",
    "ResumeMismatchError",
    "checkpoint_step",
    "config_diff",
    "config_fingerprint",
    "flatten_config",
    "list_checkpoints",
    "load_checkpoint",
    "rotate_checkpoints",
    "save_checkpoint",
]

#: 步级 checkpoint 的文件名前缀（step-0000123.pt；旋转与恢复都只认它）。
CHECKPOINT_PREFIX: str = "step-"
#: 旋转时**永不删除**的最优快照文件名（训练损失最低的那一次存盘）。
BEST_CHECKPOINT_NAME: str = "best.pt"

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


#: 续训**不**参与指纹的「运维字段」：它们是**预算与落盘策略**，不是实验语义。
#: 阻断它们的代价是「想延长步数 / 加密 checkpoint 就必须另起一个实验目录」——
#: 那恰好把同一次训练劈成两个目录（旧目录的曲线断在那里）。
#: 语义字段（模型 / 数据 / 优化器超参 / 门禁阈值 / seed）**仍然**参与指纹。
RESUME_IGNORED_KEYS: tuple[str, ...] = (
    "optim.max_steps",
    "run.save_every",
    "run.keep_last",
    "run.keep_best",
    "run.log_every",
    "run.log_level",
    "run.runs_dir",
    "run.experiment",
    # 取样本的**并行度**：顺序与覆盖率都是计划层的纯函数（RFC-0034 §5）⇒ 改变它既不改
    # 样本序列也不改任何语义，只改跑多快。把它算进指纹的代价是「为了让取批并行而重启实验
    # 目录」——那正是这条清单要避免的事。
    "data.workers",
    # val 的**观测节奏与规模**：val 走 `torch.no_grad()` + `model.eval()`，不产生梯度、
    # 不参与 LR / 早停（plan 07 §9-47 G-④ 明文禁止把它变成训练信号）⇒ 改它们**不可能**
    # 改变被训练的东西，只改变「多久看一次、看几个窗口」。把它们算进指纹的代价是
    # 「想加密验证就重启一个实验目录」——那正是本清单要避免的事（与 run.log_every 同类）。
    # 注意 `val_windows` 还是**新字段**：旧 checkpoint 的配置快照里没有它，若它参与指纹，
    # 所有既有实验目录会在升级后集体拒绝续训（这是一个真实的兼容性陷阱）。
    "optim.val_every",
    "optim.val_windows",
    "optim.val_batch",
    # 端到端产物的**节奏与落点**（RFC-0039 R3）：它读模型、写文件，**不回写任何训练状态**
    # （`generate_chart` 用 no_grad + eval，结束无条件恢复 train 状态）⇒ 与 val 同性：
    # 改它只改「多久看一眼」，不改被训练的东西。
    "run.e2e_every",
    "run.e2e_dir",
    # 窗口预切缓存：只改「窗口怎么被读出来」（0.85 s → 9.6 ms），**样本逐位不变**
    # （plan 07 §9-52）⇒ 与 `data.workers` 同性：改它不该让已有实验目录失效。
    "data.window_cache_dir",
)


def _resume_relevant(payload: Mapping[str, Any]) -> dict[str, Any]:
    """去掉运维字段后的配置视图（指纹与 diff 共用，保证两者口径一致）。"""
    flat = flatten_config(payload)
    return {key: value for key, value in flat.items() if key not in RESUME_IGNORED_KEYS}


def config_fingerprint(cfg: TrainConfig) -> str:
    """配置指纹（规范化 JSON 的 sha256；键序无关）。

    **口径**：指纹只覆盖「续训相关」的配置视图（见 :data:`RESUME_IGNORED_KEYS`）——
    预算（max_steps）与落盘策略（save_every / log_every / keep_*）变更**不会**让旧
    checkpoint 失效：它们不改变实验语义，只改变跑多久、多久存一次。
    """
    canonical = json.dumps(
        _resume_relevant(config_to_dict(cfg)), sort_keys=True, ensure_ascii=False, default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def config_diff(stored: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    """两份配置的逐字段差异（人类可读；按字段名排序；同样只看续训相关字段）。"""
    left = _resume_relevant(stored)
    right = _resume_relevant(current)
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


def checkpoint_step(path: Path) -> int | None:
    """从 step-NNNNNNN.pt 的文件名解出步号；不符合命名规范返回 None。"""
    stem = Path(path).stem
    if not stem.startswith(CHECKPOINT_PREFIX):
        return None
    suffix = stem[len(CHECKPOINT_PREFIX) :]
    return int(suffix) if suffix.isdigit() else None


def list_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    """列出产物目录里的步级 checkpoint（按步号升序）。

    只认 checkpoints/step-*.pt：best.pt 之类的快照不参与「最新一步」的判定
    （它的步号不在文件名里，拿它续训会把训练步数拉回去）。
    """
    directory = Path(run_dir) / "checkpoints"
    found: list[tuple[int, Path]] = []
    if not directory.is_dir():
        return found
    for child in sorted(directory.glob(f"{CHECKPOINT_PREFIX}*.pt")):
        step = checkpoint_step(child)
        if step is not None:
            found.append((step, child))
    return sorted(found)


def rotate_checkpoints(
    run_dir: Path,
    *,
    keep_last: int,
    keep_paths: Sequence[Path] = (),
) -> list[Path]:
    """只保留最新的 keep_last 个步级 checkpoint（外加 keep_paths），返回被删除的文件。

    **纪律**：调用方必须在**新 checkpoint 写盘成功之后**才调用本函数——先删后写会在
    「写失败」这个窗口里把唯一的恢复点也弄丢。keep_last <= 0 表示不旋转（保留全部）。
    """
    if keep_last <= 0:
        return []
    keep = {Path(item) for item in keep_paths}
    checkpoints = list_checkpoints(run_dir)
    keep.update(path for _step, path in checkpoints[-keep_last:])
    removed: list[Path] = []
    for _step, path in checkpoints:
        if path in keep:
            continue
        try:
            path.unlink()
        except OSError as exc:  # pragma: no cover - 磁盘故障才走到
            logger.warning("旧 checkpoint 删除失败（忽略）：%s（%s）", path, exc)
            continue
        removed.append(path)
    if removed:
        logger.info("checkpoint 旋转：保留最新 %d 个，删除 %d 个旧文件", keep_last, len(removed))
    return removed


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
