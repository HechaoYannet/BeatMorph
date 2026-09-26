"""R-04-7：生成模块**不得**自建「秒 <-> tau」换算（源码级扫描，进默认 CI）。

红线 7（CLAUDE.md）：beat-aligned 的时间换算**只允许**在 `beatmorph/field/` 内实现。
25 Hz 事故的同类风险是「下游各自再写一套换算」——因此本文件用 AST 扫描把
`beatmorph/generation/**` 钉死：

- 只允许调用 `beatmorph.field` 的**测度 / 场算子**接口（白名单）；
- 禁止出现 `60 / bpm` 型分段积分、`d_tau` 型自建常量、直接读取 BPMList（`.bpm`）；
- 禁止导入换算函数本身（tau_to_seconds / seconds_to_tau / jacobian_at / tau_bin_index ...）。

来历见 CLAUDE.md 红线 7 与 docs/POSTMORTEM-2026-08-05-frame-rate-misalignment.md。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

#: 允许从 beatmorph.field 取用的名字（测度与场算子；换算函数一律不在其中）
ALLOWED_FIELD_API = {
    "FieldGrid",
    "assert_lambda_valid",
    "cumulative_from_lam",
    "factorized_lambda",
    "field_cell_volumes",
    "CHANNEL_INDEX",
    "HOLD_END_CHANNEL",
    "softplus_lambda",
    "constant_baseline_nll",
}

#: 出现即违规的名字（换算函数、换算常量）
FORBIDDEN_NAMES = {
    "tau_to_seconds",
    "seconds_to_tau",
    "jacobian_at",
    "tau_bin_index",
    "bpm_segments",
    "BpmSegment",
    "SECONDS_PER_MINUTE",
    "TAU_GRID_DT",
    "BEAT_SUBDIVISION",
    "SUBDIVISIONS_PER_BEAT",
    "MERT_FRAME_RATE_HZ",
}

#: 自建 tau 格宽常量的名字片段
FORBIDDEN_ASSIGNMENT_FRAGMENTS = (
    "d_tau",
    "dtau",
    "tau_dt",
    "_dt",
    "seconds_per_beat",
    "sec_per_beat",
)


def _is_forbidden_assignment(target: ast.expr) -> bool:
    """赋值目标是否是「自建 tau 格宽常量」。"""
    if not isinstance(target, ast.Name):
        return False
    lowered = target.id.lower()
    return any(fragment in lowered for fragment in FORBIDDEN_ASSIGNMENT_FRAGMENTS)


def _is_sixty_division(node: ast.BinOp) -> bool:
    """是否是「60 / ...」型换算（秒 <-> 拍的标志性写法）。"""
    if not isinstance(node.op, ast.Div) or not isinstance(node.left, ast.Constant):
        return False
    value = node.left.value
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return abs(float(value) - 60.0) < 1e-12


def _scan_node(path: Path, node: ast.AST) -> list[str]:
    """对单个 AST 节点判定违规（命中即返回 1 条说明，未命中返回空列表）。"""
    where = f"{path.name}:{getattr(node, 'lineno', 0)}"
    checks: list[tuple[bool, str]] = [
        (
            isinstance(node, ast.ImportFrom)
            and (node.module or "").startswith("beatmorph.field")
            and any(alias.name not in ALLOWED_FIELD_API for alias in node.names),
            f"{where}: 从 beatmorph.field 取用了非白名单算子（换算只经 FieldGrid / 测度算子）",
        ),
        (
            isinstance(node, ast.Attribute) and node.attr == "bpm",
            f"{where}: 直接读取 BPMList（.bpm）——换算归 field/",
        ),
        (
            isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES,
            f"{where}: 出现换算名字（红线 7）",
        ),
        (
            isinstance(node, ast.Name) and "bpm" in node.id.lower(),
            f"{where}: 生成侧出现 BPM 相关标识符（BPMList 只作为数据由 grid 携带）",
        ),
        (
            isinstance(node, ast.arg) and "bpm" in node.arg.lower(),
            f"{where}: 函数参数涉及 BPM —— 生成侧不得处理 BPM",
        ),
        (
            (
                isinstance(node, ast.Assign)
                and any(_is_forbidden_assignment(t) for t in node.targets)
            )
            or (isinstance(node, ast.AnnAssign) and _is_forbidden_assignment(node.target)),
            f"{where}: 自建 tau 格宽常量（应从网格派生）",
        ),
        (
            isinstance(node, ast.BinOp) and _is_sixty_division(node),
            f"{where}: 出现「60 / ...」型换算——秒 <-> 拍只允许在 field/",
        ),
    ]
    return [message for hit, message in checks if hit]


def _scan_source(path: Path) -> list[str]:
    """扫描单个文件，返回违规说明列表。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    problems: list[str] = []
    for node in ast.walk(tree):
        problems.extend(_scan_node(path, node))
    return problems


def _package_files() -> list[Path]:
    root = Path(__file__).resolve().parents[3]
    return sorted((root / "beatmorph" / "generation").rglob("*.py"))


def test_generation_has_no_self_built_time_conversion() -> None:
    """R-04-7：本模块源码不得出现任何秒 <-> tau 换算实现（含 60/bpm 型分段积分）。"""
    problems: list[str] = []
    for path in _package_files():
        problems.extend(_scan_source(path))
    assert not problems, "发现生成侧自建换算：\n" + "\n".join(problems)


def test_scanner_actually_scans_the_package() -> None:
    """扫描器自检：必须真的扫到本模块的全部源码文件（否则门禁是空转的）。"""
    files = _package_files()
    assert len(files) >= 5
    names = {path.name for path in files}
    assert {"model.py", "losses.py", "masks.py", "batch.py", "sampling.py"} <= names


@pytest.mark.parametrize(
    "source",
    [
        "from beatmorph.field.grid import seconds_to_tau\n",
        "from beatmorph.field.grid import tau_to_seconds\n",
        "TAU_LOCAL_DT = 0.02\n",
        "d_tau = 1 / 48\n",
        "def f(chart):\n    return 60 / chart.bpm\n",
        "def f(bpm_value):\n    return bpm_value\n",
        "MERT_FRAME_RATE_HZ = 75.0\n",
    ],
)
def test_scanner_catches_planted_violations(tmp_path: Path, source: str) -> None:
    """门禁自检：植入的每一类违规都必须被抓到（避免静默失效）。"""
    planted = tmp_path / "planted.py"
    planted.write_text(source, encoding="utf-8")
    problems = _scan_source(planted)
    assert problems, f"未抓到违规：{source!r}"


def test_scanner_accepts_the_allowed_api(tmp_path: Path) -> None:
    """白名单内的调用不得误报（否则门禁会被迫放宽）。"""
    planted = tmp_path / "ok.py"
    planted.write_text(
        "from beatmorph.field.grid import FieldGrid\n"
        "from beatmorph.field.integrate import field_cell_volumes\n"
        "def f(grid: FieldGrid):\n    return grid.t_bins\n",
        encoding="utf-8",
    )
    assert _scan_source(planted) == []
