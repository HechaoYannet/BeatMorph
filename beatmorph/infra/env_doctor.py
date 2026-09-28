"""环境自检（env doctor，plan 07 §4.4 / M7.1）。

动机（BasePlan §5 环境纪律 / POSTMORTEM-2026-08-05）：`.venv` 曾因指向已删除的 conda
环境而**整体不可用**，导致「唯一能证伪帧率」的测试长期无法运行——**环境不可用会让所有
其它门禁静默失效**。因此环境本身也要是一条可失败的门禁。

检查项（全部可离线、无 GPU、无权重）：

| # | 检查 | 失败判据 |
|---|------|---------|
| E1 | 当前解释器是否属于项目 `.venv` | 不一致 → FAIL（防止用系统 Python 跑出「另一个环境」的结论） |
| E2 | `.venv/pyvenv.cfg` 的基解释器路径是否存在 | 指向不存在的路径 → FAIL（**历史事故的根因**） |
| E3 | `uv.lock` 与 `pyproject.toml` 的运行时依赖是否一致 | 不一致 / 缺锁文件 → FAIL（可复现性） |
| E4 | 关键依赖 import 探针 | 必需项（numpy）缺失 → FAIL；可选项缺失 → **UNKNOWN** |
| E5 | 契约常量可导入且派生式成立 | 失败 → FAIL |

**三态结果**：PASS / FAIL / UNKNOWN。不确定一律 UNKNOWN，**不得**默认成 PASS。
退出码：0 = 全 PASS；1 = 存在 FAIL；2 = 无 FAIL 但有 UNKNOWN（「不知道」不等于「没问题」）。
"""

from __future__ import annotations

import argparse
import importlib
import re
import sys
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from beatmorph.core.logging import get_logger, setup_logging
from beatmorph.infra.derive import assert_derived_identities, derived_values

__all__ = [
    "OPTIONAL_MODULES",
    "REQUIRED_MODULES",
    "CheckState",
    "EnvCheck",
    "EnvReport",
    "check_contract_constants",
    "check_imports",
    "check_interpreter",
    "check_lock_consistency",
    "check_pyvenv_cfg",
    "main",
    "repo_root",
    "run_env_doctor",
]

logger = get_logger("infra.env_doctor")

#: 必需依赖（缺失即 FAIL）
REQUIRED_MODULES: tuple[str, ...] = ("numpy",)
#: 可选依赖（缺失 → UNKNOWN，**不得**报 PASS）
OPTIONAL_MODULES: tuple[str, ...] = ("torch", "pytorch_lightning", "tensorboard")

_PYVENV_KEYS: tuple[str, ...] = ("home", "executable", "base-executable")


class CheckState(StrEnum):
    """三态检查结果。"""

    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class EnvCheck:
    """单项检查结果。"""

    id: str
    name: str
    state: CheckState
    detail: str


@dataclass(frozen=True, slots=True)
class EnvReport:
    """E1-E5 的汇总。"""

    checks: tuple[EnvCheck, ...]

    @property
    def has_failure(self) -> bool:
        """是否存在 FAIL。"""
        return any(check.state is CheckState.FAIL for check in self.checks)

    @property
    def has_unknown(self) -> bool:
        """是否存在 UNKNOWN（无 FAIL 时才算「环境可疑但未证伪」）。"""
        return any(check.state is CheckState.UNKNOWN for check in self.checks)

    @property
    def exit_code(self) -> int:
        """0 = 全 PASS；1 = 有 FAIL；2 = 无 FAIL 但有 UNKNOWN。"""
        if self.has_failure:
            return 1
        return 2 if self.has_unknown else 0

    def format(self) -> str:
        """人类可读的多行报告（可直接贴进训练日志）。"""
        lines = ["环境自检（env doctor，plan 07 M7.1）："]
        lines += [
            f"  [{check.state}] {check.id} {check.name}：{check.detail}" for check in self.checks
        ]
        lines.append(
            f"  => 退出码 {self.exit_code}"
            f"（FAIL {sum(1 for c in self.checks if c.state is CheckState.FAIL)} 项，"
            f"UNKNOWN {sum(1 for c in self.checks if c.state is CheckState.UNKNOWN)} 项）"
        )
        return "\n".join(lines)


def repo_root() -> Path:
    """仓库根（本文件位于 `<root>/beatmorph/infra/env_doctor.py`）。"""
    return Path(__file__).resolve().parents[2]


# ══════════════════════════════════════════════════════════════
# E1 解释器归属
# ══════════════════════════════════════════════════════════════


def check_interpreter(*, venv_root: Path, executable: Path) -> EnvCheck:
    """E1：当前解释器必须来自项目 `.venv`。"""
    venv_root = Path(venv_root)
    executable = Path(executable)
    if not venv_root.exists():
        return EnvCheck(
            "E1",
            "解释器归属",
            CheckState.FAIL,
            f"项目虚拟环境不存在：{venv_root}（先跑 uv sync --group dev）",
        )
    try:
        executable.resolve().relative_to(venv_root.resolve())
    except ValueError:
        return EnvCheck(
            "E1",
            "解释器归属",
            CheckState.FAIL,
            f"当前解释器 {executable} 不在项目环境 {venv_root} 之内："
            "用系统 Python 跑出的结论属于**另一个环境**",
        )
    return EnvCheck("E1", "解释器归属", CheckState.PASS, f"解释器 {executable} 属于 {venv_root}")


# ══════════════════════════════════════════════════════════════
# E2 基解释器存在性（历史事故根因）
# ══════════════════════════════════════════════════════════════


def _read_pyvenv_cfg(path: Path) -> dict[str, str] | None:
    if not path.is_file():
        return None
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in raw:
            continue
        key, _, value = raw.partition("=")
        values[key.strip().lower()] = value.strip()
    return values


def check_pyvenv_cfg(*, venv_root: Path) -> EnvCheck:
    """E2：`pyvenv.cfg` 记录的基解释器必须真实存在。"""
    path = Path(venv_root) / "pyvenv.cfg"
    values = _read_pyvenv_cfg(path)
    if values is None:
        return EnvCheck("E2", "基解释器存在", CheckState.FAIL, f"缺少 {path}")
    present = [(key, values[key]) for key in _PYVENV_KEYS if key in values]
    if not present:
        return EnvCheck(
            "E2",
            "基解释器存在",
            CheckState.UNKNOWN,
            f"{path} 未记录 {' / '.join(_PYVENV_KEYS)}，无法判定基解释器是否可用",
        )
    missing = [f"{key}={value}" for key, value in present if not Path(value).exists()]
    if missing:
        return EnvCheck(
            "E2",
            "基解释器存在",
            CheckState.FAIL,
            f"{path} 指向不存在的基解释器：{'；'.join(missing)}（这正是历史事故的形态）",
        )
    return EnvCheck(
        "E2",
        "基解释器存在",
        CheckState.PASS,
        f"{path} 的基解释器存在：{'；'.join(key + '=' + value for key, value in present)}",
    )


# ══════════════════════════════════════════════════════════════
# E3 锁文件一致性
# ══════════════════════════════════════════════════════════════


def _normalize(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _pep508_name(spec: str) -> str:
    return _normalize(re.split(r"[<>=!~;\[ ]", spec.strip(), maxsplit=1)[0])


def _lock_dependency_name(entry: object) -> str | None:
    if isinstance(entry, str):
        return _normalize(entry)
    if isinstance(entry, dict) and "name" in entry:
        return _normalize(str(entry["name"]))
    return None


def check_lock_consistency(*, root: Path) -> EnvCheck:
    """E3：`uv.lock` 的运行时依赖集合必须与 `pyproject.toml` 一致。"""
    root = Path(root)
    pyproject = root / "pyproject.toml"
    lock = root / "uv.lock"
    if not pyproject.is_file():
        return EnvCheck("E3", "锁文件一致", CheckState.FAIL, f"缺少 {pyproject}")
    if not lock.is_file():
        return EnvCheck(
            "E3",
            "锁文件一致",
            CheckState.FAIL,
            f"缺少 {lock}：没有锁文件就无法复现环境（跑 uv lock 生成）",
        )
    try:
        declared_raw = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        lock_raw = tomllib.loads(lock.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        return EnvCheck("E3", "锁文件一致", CheckState.FAIL, f"TOML 解析失败：{exc}")
    project_name = _normalize(str(declared_raw.get("project", {}).get("name", "beatmorph")))
    declared = {
        _pep508_name(spec) for spec in declared_raw.get("project", {}).get("dependencies", [])
    }
    locked: set[str] = set()
    for package in lock_raw.get("package", []):
        if _normalize(str(package.get("name", ""))) != project_name:
            continue
        for entry in package.get("dependencies", []):
            name = _lock_dependency_name(entry)
            if name is not None:
                locked.add(name)
        break
    else:
        return EnvCheck("E3", "锁文件一致", CheckState.FAIL, f"{lock} 中没有 {project_name} 包条目")
    missing = sorted(declared - locked)
    extra = sorted(locked - declared)
    if missing or extra:
        return EnvCheck(
            "E3",
            "锁文件一致",
            CheckState.FAIL,
            f"pyproject 与 uv.lock 的运行时依赖不一致：锁中缺少 {missing}；锁中多出 {extra}",
        )
    return EnvCheck("E3", "锁文件一致", CheckState.PASS, f"运行时依赖 {len(declared)} 项两处一致")


# ══════════════════════════════════════════════════════════════
# E4 依赖 import 探针
# ══════════════════════════════════════════════════════════════


def check_imports(*, probe: Callable[[str], Any] | None = None) -> EnvCheck:
    """E4：必需依赖 present；可选依赖缺失 → UNKNOWN（**不得**默认成 PASS）。"""
    import_module = importlib.import_module if probe is None else probe

    def missing(modules: Sequence[str]) -> list[str]:
        absent: list[str] = []
        for name in modules:
            try:
                import_module(name)
            except ImportError:
                absent.append(name)
        return absent

    absent_required = missing(REQUIRED_MODULES)
    if absent_required:
        return EnvCheck(
            "E4",
            "依赖探针",
            CheckState.FAIL,
            f"必需依赖缺失：{absent_required}（安装：uv sync --group dev）",
        )
    absent_optional = missing(OPTIONAL_MODULES)
    if absent_optional:
        return EnvCheck(
            "E4",
            "依赖探针",
            CheckState.UNKNOWN,
            f"必需依赖齐备，但可选依赖缺失：{absent_optional} → 训练栈不可用"
            "（安装：uv sync --extra train）",
        )
    return EnvCheck(
        "E4",
        "依赖探针",
        CheckState.PASS,
        f"必需 {list(REQUIRED_MODULES)} 与可选 {list(OPTIONAL_MODULES)} 全部可导入",
    )


# ══════════════════════════════════════════════════════════════
# E5 契约常量与派生式
# ══════════════════════════════════════════════════════════════


def check_contract_constants() -> EnvCheck:
    """E5：契约常量可导入且派生式成立（红线 7 的断言半边）。"""
    try:
        assert_derived_identities()
    except AssertionError as exc:
        return EnvCheck("E5", "契约常量", CheckState.FAIL, str(exc))
    values = derived_values()
    rendered = "，".join(f"{name}={value:g}" for name, value in values.items())
    return EnvCheck("E5", "契约常量", CheckState.PASS, f"派生式成立：{rendered}")


# ══════════════════════════════════════════════════════════════
# 汇总与 CLI
# ══════════════════════════════════════════════════════════════


def run_env_doctor(
    *,
    root: Path | None = None,
    venv_root: Path | None = None,
    executable: Path | None = None,
    probe: Callable[[str], Any] | None = None,
) -> EnvReport:
    """跑完 E1-E5 并汇总（**不抛异常**：失败以 CheckState 表达）。"""
    repository = repo_root() if root is None else Path(root)
    venv = (repository / ".venv") if venv_root is None else Path(venv_root)
    python = Path(sys.executable) if executable is None else Path(executable)
    checks = (
        check_interpreter(venv_root=venv, executable=python),
        check_pyvenv_cfg(venv_root=venv),
        check_lock_consistency(root=repository),
        check_imports(probe=probe),
        check_contract_constants(),
    )
    return EnvReport(checks=checks)


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口：打印报告并返回退出码（0 PASS / 1 FAIL / 2 UNKNOWN）。"""
    parser = argparse.ArgumentParser(description="BeatMorph 环境自检（plan 07 M7.1）")
    parser.add_argument("--root", type=Path, default=None, help="仓库根（默认按本文件定位）")
    args = parser.parse_args(argv)
    setup_logging()
    report = run_env_doctor(root=args.root)
    logger.info("\n%s", report.format())
    return report.exit_code


if __name__ == "__main__":  # pragma: no cover - 手工运行入口
    raise SystemExit(main())
