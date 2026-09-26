"""红线 7 与工程约定的机器探针：eval 不实现秒 <-> τ 换算、不引 torch、不裸 print。

POSTMORTEM 的教训是「约束写在文档里不够，必须有默认 CI 内运行的探针」：25 Hz 误值
之所以活到万级数据规模，正是因为没有任何测试盯着它。本文件用 AST 检查三条：

1. 秒 <-> 拍换算的**实现**只能存在于 beatmorph/field/（eval 只许 import 使用）；
2. 物理常量名（帧率 / 采样率 / 拍格宽）不得在 eval 内出现；
3. eval 模块级不得 import torch（保证默认 CI 轻量、无权重依赖），且不得裸 print。
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel

import beatmorph.eval as eval_package

REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = REPO_ROOT / "beatmorph" / "eval"

#: 秒 <-> 拍换算的函数名（**只允许**在 beatmorph/field/ 内定义；红线 7 / RFC-0029 §7-8）。
TAU_FUNCTION_NAMES = frozenset(
    {"seconds_to_tau", "tau_to_seconds", "jacobian_at", "bpm_segments", "tau_bin_index"},
)
#: eval 允许从 field/ 借用的换算入口（其余一律不许出现）。
ALLOWED_TAU_IMPORTS = frozenset({"seconds_to_tau", "tau_to_seconds"})

#: 「秒 <-> 拍 / 帧率」类物理常量：eval 内**一律不得出现**（红线 7 的核心）。
PHYSICAL_CONSTANT_NAMES = frozenset(
    {
        "SUBDIVISIONS_PER_BEAT",
        "TAU_GRID_DT",
        "SECONDS_PER_MINUTE",
        "MERT_FRAME_RATE_HZ",
        "MERT_SAMPLE_RATE_HZ",
        "MERT_CONV_STRIDE_PRODUCT",
    },
)
#: 契约派生常量的前缀：允许引用，但**必须**来自 beatmorph.core.contracts（不得本地重写）。
CONTRACT_CONSTANT_PREFIX = "RPE_"


def _trees() -> list[tuple[Path, ast.Module]]:
    """eval 包的源码 AST（每个文件一份）。"""
    return [
        (path, ast.parse(path.read_text(encoding="utf-8")))
        for path in sorted(EVAL_DIR.glob("*.py"))
    ]


def _referenced_names(tree: ast.Module) -> set[str]:
    """源码里被引用的标识符（Name / Attribute 属性名 / import 别名）。"""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.alias):
            names.add(node.name)
    return names


def test_eval_package_has_the_expected_modules() -> None:
    """评估包必须有实现与测试（防止整包被误删/误排除）。"""
    modules = {path.name for path, _ in _trees()}
    assert {
        "__init__.py",
        "protocol.py",
        "matching.py",
        "metrics.py",
        "breakdown.py",
        "calibration.py",
        "corruption.py",
        "report.py",
        "stats.py",
    } <= modules
    assert (REPO_ROOT / "tests" / "unit" / "eval" / "__init__.py").is_file()


def test_no_seconds_to_beat_conversion_is_implemented_in_eval() -> None:
    """红线 7：eval 不得定义任何秒 <-> τ 换算（那是 field/ 的独占职责）。"""
    for path, tree in _trees():
        defined = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        }
        assert not (defined & TAU_FUNCTION_NAMES), (
            f"{path.name} 定义了秒 <-> 拍换算函数：{sorted(defined & TAU_FUNCTION_NAMES)}"
        )
        forbidden = _referenced_names(tree) & (TAU_FUNCTION_NAMES - ALLOWED_TAU_IMPORTS)
        assert not forbidden, f"{path.name} 引用了不该出现的换算接口：{sorted(forbidden)}"


def test_tau_conversion_is_borrowed_from_field_only() -> None:
    """若 eval 需要拍坐标，只能从 beatmorph.field.grid 借用（不得自建）。"""
    for path, tree in _trees():
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in ALLOWED_TAU_IMPORTS:
                        assert node.module == "beatmorph.field.grid", (
                            f"{path.name} 从 {node.module!r} 引入 {alias.name!r}："
                            "秒 <-> τ 换算的唯一出处是 beatmorph.field.grid"
                        )
                        imported.add(alias.asname or alias.name)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not any(name in alias.name for name in ALLOWED_TAU_IMPORTS), (
                        f"{path.name} 以 import 形式引入换算接口：{alias.name}"
                    )
        used = _referenced_names(tree) & ALLOWED_TAU_IMPORTS
        assert used <= imported, f"{path.name} 使用了未从 field.grid 引入的换算：{sorted(used)}"


def test_physical_constants_are_never_referenced_in_eval() -> None:
    """红线 7：物理常量（帧率 / 拍格宽 / x 网格）只在契约与 field 内派生。"""
    for path, tree in _trees():
        offenders = _referenced_names(tree) & PHYSICAL_CONSTANT_NAMES
        assert not offenders, f"{path.name} 直接引用了物理常量：{sorted(offenders)}"


def test_contract_constants_are_imported_from_the_contracts_package() -> None:
    """允许引用的派生常量（RPE_*）必须从 core/contracts 引入，而不是在 eval 内重写。"""
    for path, tree in _trees():
        imported: set[str] = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "beatmorph.core.contracts.phigros"
            ):
                imported |= {alias.name for alias in node.names}
        used = {
            name for name in _referenced_names(tree) if name.startswith(CONTRACT_CONSTANT_PREFIX)
        }
        assert used <= imported, (
            f"{path.name} 引用了未从契约引入的派生常量：{sorted(used - imported)}"
        )


def test_eval_modules_do_not_import_torch_at_module_level() -> None:
    """eval 只依赖 numpy / pydantic / 契约（需要 torch 的地方在函数内惰性引入）。"""
    for path, tree in _trees():
        for node in tree.body:
            if isinstance(node, ast.Import):
                assert all(alias.name != "torch" for alias in node.names), path.name
            elif isinstance(node, ast.ImportFrom):
                assert node.module != "torch", path.name


def test_no_bare_print_in_eval_sources() -> None:
    """生产路径禁止裸 print（CLAUDE.md §5.6：用 beatmorph.core.logging.get_logger）。"""
    for path, tree in _trees():
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
            ):
                raise AssertionError(f"{path.name} 出现裸 print（生产路径禁止）")


def test_public_fields_have_no_frame_or_tau_grid_indices() -> None:
    """公共接口不得出现帧索引 / τ 格索引（plan §3.1；一切时间量以秒为后缀 _s）。"""
    names: set[str] = set()
    for _, member in inspect.getmembers(eval_package, inspect.isclass):
        if issubclass(member, BaseModel):
            names |= set(member.model_fields)
        if dataclasses.is_dataclass(member):
            names |= set(member.__dataclass_fields__)
    banned = [name for name in sorted(names) if "frame" in name or "tau" in name]
    assert not banned, f"公共字段里出现帧/tau 索引：{banned}"


def test_importing_eval_does_not_pull_torch() -> None:
    """结构性保证：默认 CI 跑评估不需要 torch 的重量级导入。"""
    code = "import sys; import beatmorph.eval; print('torch' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False", (
        "import beatmorph.eval 意外拉起了 torch：请把重量级依赖改为函数内惰性引入"
    )
