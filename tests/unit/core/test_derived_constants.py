"""派生常量防线（Plan 00 M4）：源码扫描测试，进默认 CI。

红线 7 / RFC-0029 §7-1：任何帧率 / 采样率 / 单位换算**不得硬编码**。
本测试用 AST 扫描 beatmorph/ 下的**数值字面量**（不是正则匹配文本，
因此注释与文档字符串中的说明性数字不会误报），只允许 beatmorph/core/contracts/
内出现这些物理量的具体取值。

来历：25 Hz 误值（真值 75 Hz，差 3 倍）在 mock 的掩护下存活到万级数据规模
（docs/POSTMORTEM-2026-08-05-frame-rate-misalignment.md）。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

#: 允许出现来源常量的唯一目录（相对仓库根）。
CONSTANTS_DIR = Path("beatmorph") / "core" / "contracts"

#: 精确匹配的禁用字面量（物理量的具体取值必须写成派生式）。
FORBIDDEN_EXACT: dict[float, str] = {
    1350.0: "RPE_STAGE_WIDTH（应引用契约常量）",
    900.0: "RPE_STAGE_HEIGHT（应引用契约常量）",
    675.0: "RPE_STAGE_HALF_WIDTH（应派生）",
    450.0: "RPE_STAGE_HALF_HEIGHT（应派生）",
    10.546875: "RPE_X_GRID_DX（应派生为 RPE_STAGE_WIDTH / x_bins）",
    0.83175: "RPE_HEIGHT_RATIO（来源常量，只在契约层）",
    75.0: "MERT_FRAME_RATE_HZ（应派生为采样率 / 卷积步长累乘）",
    320.0: "MERT_CONV_STRIDE_PRODUCT（来源常量，只在契约层）",
    24000.0: "MERT_SAMPLE_RATE_HZ（来源常量，只在契约层）",
}

#: 容差匹配：速度单位（120.228… RPE-y/秒）与其旧写法 120.23。
FORBIDDEN_NEAR: tuple[tuple[float, float, str], ...] = (
    (120.23, 0.05, "RPE_SPEED_UNIT_RPE_Y_PER_SEC（应派生）"),
)

#: 1/48 拍格宽（禁止写字面量小数）。
FORBIDDEN_TAU_DT: float = 1.0 / 48.0


def _iter_numeric_literals(path: Path) -> list[tuple[int, float]]:
    """返回 (行号, 数值) 列表；bool 不算数值字面量。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: list[tuple[int, float]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            if isinstance(node.value, bool):
                continue
            out.append((node.lineno, float(node.value)))
    return out


def _scanned_files() -> list[Path]:
    root = Path(__file__).resolve().parents[3]
    allowed = (root / CONSTANTS_DIR).resolve()
    files: list[Path] = []
    for path in sorted((root / "beatmorph").rglob("*.py")):
        if allowed in path.resolve().parents:
            continue
        files.append(path)
    return files


def _violations() -> list[str]:
    root = Path(__file__).resolve().parents[3]
    found: list[str] = []
    for path in _scanned_files():
        rel = path.relative_to(root)
        for lineno, value in _iter_numeric_literals(path):
            reason = FORBIDDEN_EXACT.get(value)
            if reason is not None:
                found.append(f"{rel}:{lineno}: 字面量 {value!r} 被禁 —— {reason}")
                continue
            if abs(value - FORBIDDEN_TAU_DT) < 1e-12:
                found.append(
                    f"{rel}:{lineno}: 字面量 {value!r} 被禁 —— 拍格宽应取自 TAU_GRID_DT",
                )
                continue
            for target, tol, why in FORBIDDEN_NEAR:
                if abs(value - target) <= tol:
                    found.append(f"{rel}:{lineno}: 字面量 {value!r} 被禁 —— {why}")
                    break
    return found


def test_no_hardcoded_physical_constants_outside_contracts() -> None:
    """M4：beatmorph/ 下除 core/contracts/ 外不得出现物理常量数值字面量。"""
    violations = _violations()
    assert not violations, "发现硬编码物理常量：\n" + "\n".join(violations)


def test_scanner_actually_scans_something() -> None:
    """扫描器自检：必须真的扫到文件，否则本门禁是空转。"""
    files = _scanned_files()
    assert len(files) >= 5
    assert all(path.suffix == ".py" for path in files)


def test_scanner_detects_a_planted_literal(tmp_path: Path) -> None:
    """扫描器自检：植入一个字面量必须被抓到（避免门禁静默失效）。"""
    planted = tmp_path / "planted.py"
    planted.write_text("STAGE = 675.0\n", encoding="utf-8")
    literals = _iter_numeric_literals(planted)
    assert literals == [(1, 675.0)]
    assert FORBIDDEN_EXACT[675.0]


@pytest.mark.parametrize("value", sorted(FORBIDDEN_EXACT))
def test_blacklist_values_are_the_derived_names_sources(value: float) -> None:
    """黑名单非空且每条都带理由（防止有人把黑名单清空来"修绿"）。"""
    assert FORBIDDEN_EXACT[value]
