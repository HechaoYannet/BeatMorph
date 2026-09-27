"""实验产物目录与「六件套」契约（plan 07 §3.2 / M7.7 / M7.8）。

`runs/<experiment>/<timestamp>/` 的固定清单，**缺一即视为实验不可信**：

| 文件 | 内容 |
|------|------|
| `config.yaml` | Hydra/OmegaConf 解析后的**完整**配置（含 override 与 provenance） |
| `gates.txt` | `summarize()` 原文 + **生效阈值** + git rev + data rev |
| `data_provenance.json` | 数据来源与用途的解析后快照（合规硬约束③） |
| `checkpoints/` | 权重与优化器状态（不入库：红线 5） |
| `logs/` | TensorBoard event 文件 + 文本日志 |
| `metrics.json` | 评估产物（格式见 plan 06） |

两条纪律写进代码而不是写进口头约定：

1. **不覆盖历史实验**（M7.8）：目录已存在就顺延后缀，`exist_ok` 默认 False；
2. **产物不落进版本控制**（红线 5）：写入前用 plan 02 的 `assert_local_only` 校验
   `runs/` 确实被 `.gitignore` 覆盖（仓库外的路径不适用该检查）。
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from beatmorph.core.logging import get_logger
from beatmorph.infra.config.schema import TrainConfig, config_to_dict

__all__ = [
    "CHECKPOINTS_DIR",
    "CONFIG_FILENAME",
    "GATES_FILENAME",
    "LOGS_DIR",
    "METRICS_FILENAME",
    "PROVENANCE_FILENAME",
    "REQUIRED_ARTIFACTS",
    "RunArtifacts",
    "file_sha1",
    "git_rev",
]

logger = get_logger("infra.artifacts")

CONFIG_FILENAME: str = "config.yaml"
GATES_FILENAME: str = "gates.txt"
PROVENANCE_FILENAME: str = "data_provenance.json"
METRICS_FILENAME: str = "metrics.json"
CHECKPOINTS_DIR: str = "checkpoints"
LOGS_DIR: str = "logs"

#: 六件套（缺一即视为实验不可信；plan 07 §3.2）
REQUIRED_ARTIFACTS: tuple[str, ...] = (
    CONFIG_FILENAME,
    GATES_FILENAME,
    PROVENANCE_FILENAME,
    CHECKPOINTS_DIR,
    LOGS_DIR,
    METRICS_FILENAME,
)


def git_rev(root: Path) -> str:
    """当前 HEAD 的短 rev（拿不到就返回 `unknown`，**不**抛错）。"""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - 环境无 git 时
        return "unknown"
    if completed.returncode != 0:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def file_sha1(path: Path) -> str:
    """文件内容的 sha1（数据版本留痕；文件不存在返回 `missing`）。"""
    import hashlib

    path = Path(path)
    if not path.is_file():
        return "missing"
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(slots=True)
class RunArtifacts:
    """一次实验的产物目录句柄。"""

    root: Path

    # ── 构造 ────────────────────────────────────────────────────
    @classmethod
    def create(
        cls,
        *,
        runs_dir: Path,
        experiment: str,
        timestamp: str | None = None,
        max_suffix: int = 100,
    ) -> RunArtifacts:
        """建立（并**不覆盖**已有）实验目录 `<runs>/<experiment>/<timestamp>`。

        Args:
            runs_dir: 产物根（通常为 `runs`）。
            experiment: 实验名（一级子目录）。
            timestamp: 时间戳（默认 UTC `%Y%m%d-%H%M%S`；测试可注入）。
            max_suffix: 同名目录的后缀尝试上限。

        Raises:
            FileExistsError: 连续 `max_suffix` 个候选目录都已存在。
        """
        runs_dir = Path(runs_dir)
        if not experiment.strip():
            raise ValueError("experiment 不得为空")
        stamp = timestamp or datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        base = runs_dir / experiment
        for attempt in range(max_suffix):
            candidate = base / (stamp if attempt == 0 else f"{stamp}-{attempt:02d}")
            if candidate.exists():
                continue
            artifacts = cls(root=candidate)
            # 先判「该不该写这里」，再创建目录：否则一次被拒绝的调用会在仓库里留下空目录
            artifacts._assert_ignored()
            candidate.mkdir(parents=True)
            artifacts.init_layout()
            return artifacts
        raise FileExistsError(f"{base}/{stamp} 及其 {max_suffix} 个后缀均已存在，拒绝覆盖")

    @classmethod
    def open_existing(cls, root: Path) -> RunArtifacts:
        """打开一个**已存在**的实验目录（续训用：不新建、不覆盖、不换目录）。

        续训必须落在**原来那个目录**：六件套是一个实验的整体，另起一个目录会把同一次
        实验劈成两半（而 checkpoint 的 meta 里记的正是那一次实验的配置与数据版本）。

        Raises:
            FileNotFoundError: 目录不存在。
            AssertionError: 目录未被 .gitignore 覆盖（红线 5）。
        """
        artifacts = cls(root=Path(root))
        if not artifacts.root.is_dir():
            raise FileNotFoundError(f"实验目录不存在：{artifacts.root}")
        artifacts._assert_ignored()
        artifacts.init_layout()
        return artifacts

    def _assert_ignored(self) -> None:
        """红线 5：产物必须落在被 `.gitignore` 覆盖的位置（仓库外不做该检查）。"""
        from beatmorph.data import assert_local_only, is_ignored, repo_root

        root = repo_root()
        try:
            self.root.resolve().relative_to(root.resolve())
        except ValueError:
            logger.info("产物目录在仓库之外：%s（跳过 .gitignore 检查）", self.root)
            return
        if not is_ignored(self.root, root=root):
            raise AssertionError(
                f"{self.root} 未被 .gitignore 覆盖：权重/产物不得入库（CLAUDE.md 红线 5）",
            )
        assert_local_only(self.root, root=root)

    def init_layout(self) -> None:
        """建立 `checkpoints/` 与 `logs/`。"""
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)

    # ── 路径 ────────────────────────────────────────────────────
    @property
    def checkpoints(self) -> Path:
        """checkpoints 目录。"""
        return self.root / CHECKPOINTS_DIR

    @property
    def logs(self) -> Path:
        """logs 目录（TB event + 文本日志）。"""
        return self.root / LOGS_DIR

    def path(self, name: str) -> Path:
        """产物文件路径。"""
        return self.root / name

    # ── 写入 ────────────────────────────────────────────────────
    def write_config(self, cfg: TrainConfig) -> Path:
        """落盘解析后的完整配置（含 override 与 provenance）。"""
        import yaml

        target = self.path(CONFIG_FILENAME)
        text = yaml.safe_dump(
            config_to_dict(cfg), allow_unicode=True, sort_keys=False, default_flow_style=False
        )
        target.write_text(text, encoding="utf-8")
        return target

    def write_provenance(self, cfg: TrainConfig, *, extra: Mapping[str, Any] | None = None) -> Path:
        """落盘数据来源与用途快照（`data_provenance.json`）。

        以数据侧 `Provenance` 的 schema 为准（`to_manifest_provenance()`），
        另附运行级字段：`run_purpose`（实验用途）、`script_rev`（本项目 git rev）、
        `acquired_at`（数据获取时间，与 `fetched_at` 同值，便于按 plan 07 §3.2 的字段名核查）。
        """
        provenance = cfg.data.provenance.to_manifest_provenance()
        payload: dict[str, Any] = json.loads(provenance.model_dump_json())
        payload.update(
            {
                "run_purpose": str(cfg.run.purpose),
                "script_rev": extra.get("script_rev") if extra else None,
                "acquired_at": provenance.fetched_at,
                "experiment": cfg.run.experiment,
            }
        )
        if extra:
            payload.update({key: value for key, value in extra.items() if key != "script_rev"})
        target = self.path(PROVENANCE_FILENAME)
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        return target

    def write_gates(self, text: str) -> Path:
        """落盘 `gates.txt`（`summarize()` 原文 + 生效阈值 + 版本信息）。"""
        target = self.path(GATES_FILENAME)
        target.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
        return target

    def write_metrics(self, payload: Mapping[str, Any]) -> Path:
        """落盘 `metrics.json`（格式见 plan 06）。"""
        target = self.path(METRICS_FILENAME)
        target.write_text(
            json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        return target

    def write_text_log(self, name: str, text: str) -> Path:
        """在 `logs/` 下写一份文本日志（门禁原文 / 训练摘要等）。"""
        target = self.logs / name
        target.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
        return target

    # ── 自检 ────────────────────────────────────────────────────
    def missing(self) -> list[str]:
        """六件套里缺失的条目（按声明顺序）。"""
        return [name for name in REQUIRED_ARTIFACTS if not self.path(name).exists()]

    def assert_complete(self) -> None:
        """六件套齐全性断言（缺一即抛）。

        Raises:
            AssertionError: 缺少条目（消息列出具体名字）。
        """
        missing = self.missing()
        if missing:
            raise AssertionError(
                f"实验产物不完整，缺少 {missing}（六件套缺一即视为实验不可信，plan 07 §3.2）",
            )

    def describe(self) -> str:
        """一行诊断文本（不含权重内容）。"""
        missing = self.missing()
        return f"RunArtifacts({self.root}；缺失={missing or '无'})"
