"""M7.4：派生量单一事实源 —— 全仓扫描 + 派生式断言（默认 CI）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from beatmorph.core.contracts import (
    MERT_CONV_STRIDE_PRODUCT,
    MERT_SAMPLE_RATE_HZ,
)
from beatmorph.infra.derive import (
    DEFINITION_ALLOWLIST,
    assert_derived_identities,
    derived_values,
    format_derived_banner,
    scan_derived_literals,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_derived_identities_hold() -> None:
    """派生式必须在运行期成立（红线 7 的「断言」半边）。"""
    assert_derived_identities()
    values = derived_values()
    assert values["MERT_FRAME_RATE_HZ"] == pytest.approx(
        MERT_SAMPLE_RATE_HZ / MERT_CONV_STRIDE_PRODUCT,
    )


def test_banner_prints_values_and_sources() -> None:
    """启动横幅必须**打印取值**（plan 07 §3.3 的两半：派生 + 断言 + 打印）。"""
    banner = format_derived_banner()
    assert "MERT_FRAME_RATE_HZ" in banner
    assert "beatmorph/core/contracts/tensors.py" in banner


def test_real_repo_has_no_second_definition_or_rate_literal() -> None:
    """全仓扫描必须干净：生产路径里不得出现第二处派生量定义/帧率字面量。"""
    hits = scan_derived_literals(_repo_root())
    assert hits == [], "发现派生量字面量/重复定义：" + "; ".join(
        f"{hit.path}:{hit.line_number} {hit.kind}" for hit in hits
    )


def test_allowlist_only_covers_the_contract_modules() -> None:
    """白名单必须窄：只允许契约模块定义派生量。"""
    assert all(path.startswith("beatmorph/core/contracts/") for path in DEFINITION_ALLOWLIST)


def test_scan_detects_planted_literal(tmp_path: Path) -> None:
    """故障注入：植入一行帧率字面量必须被抓到（否则扫描是个摆设）。"""
    module = tmp_path / "beatmorph" / "fake.py"
    module.parent.mkdir(parents=True)
    module.write_text("frame_rate = 75.0\n", encoding="utf-8")
    hits = scan_derived_literals(tmp_path)
    assert len(hits) == 1
    assert hits[0].kind == "literal"
    assert hits[0].path == "beatmorph/fake.py"


def test_scan_detects_second_definition(tmp_path: Path) -> None:
    """故障注入：第二处定义（即使表达式正确）同样必须被抓到。"""
    module = tmp_path / "beatmorph" / "fake.py"
    module.parent.mkdir(parents=True)
    module.write_text(
        "from beatmorph.core.contracts import MERT_SAMPLE_RATE_HZ, MERT_CONV_STRIDE_PRODUCT\n"
        "MERT_FRAME_RATE_HZ: float = MERT_SAMPLE_RATE_HZ / MERT_CONV_STRIDE_PRODUCT\n",
        encoding="utf-8",
    )
    hits = scan_derived_literals(tmp_path)
    assert [hit.kind for hit in hits] == ["definition"]


def test_scan_ignores_prose_and_comments(tmp_path: Path) -> None:
    """docstring / 注释里的说明文字不是违规项（噪声门禁最终会被关掉）。"""
    module = tmp_path / "beatmorph" / "doc.py"
    module.parent.mkdir(parents=True)
    module.write_text(
        '"""帧率 24000/320 = 75 Hz 的来历见 POSTMORTEM。"""\n'
        "# frame_rate = 75 的正确写法是从 contracts 派生\n"
        "VALUE = 1  # sample_rate: 24000\n",
        encoding="utf-8",
    )
    assert scan_derived_literals(tmp_path) == []


def test_scan_detects_yaml_literal(tmp_path: Path) -> None:
    """配置里写死帧率是最容易复发的一种（它看起来像「只是配置」）。"""
    config = tmp_path / "configs" / "bad.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("train:\n  frame_rate: 75.0\n", encoding="utf-8")
    hits = scan_derived_literals(tmp_path)
    assert len(hits) == 1
    assert hits[0].path == "configs/bad.yaml"
