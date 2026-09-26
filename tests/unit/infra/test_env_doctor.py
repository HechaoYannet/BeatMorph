"""M7.1：环境自检 E1-E5 的三态语义与两个验收场景（默认 CI，无 GPU/权重）。"""

from __future__ import annotations

from pathlib import Path

from beatmorph.infra.env_doctor import (
    CheckState,
    EnvCheck,
    EnvReport,
    check_contract_constants,
    check_imports,
    check_interpreter,
    check_lock_consistency,
    check_pyvenv_cfg,
    run_env_doctor,
)


def _venv(tmp_path: Path, *, home: str | None = None, executable: str | None = None) -> Path:
    """造一个最小虚拟环境目录（可注入基解释器路径）。"""
    venv = tmp_path / ".venv"
    (venv / "Scripts").mkdir(parents=True)
    (venv / "Scripts" / "python.exe").write_text("", encoding="utf-8")
    if home is not None or executable is not None:
        lines = []
        if home is not None:
            lines.append(f"home = {home}")
        if executable is not None:
            lines.append(f"executable = {executable}")
        (venv / "pyvenv.cfg").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return venv


def test_scenario_normal_environment_passes(tmp_path: Path) -> None:
    """场景 ①：正常环境 → E1/E2/E3/E5 全 PASS（E4 视依赖而定）。"""
    base = tmp_path / "python"
    base.mkdir()
    venv = _venv(tmp_path, home=str(base), executable=str(base))
    assert check_interpreter(venv_root=venv, executable=venv / "Scripts" / "python.exe").state is (
        CheckState.PASS
    )
    assert check_pyvenv_cfg(venv_root=venv).state is CheckState.PASS
    assert check_contract_constants().state is CheckState.PASS


def test_scenario_broken_base_interpreter_fails_with_path(tmp_path: Path) -> None:
    """场景 ②：把基解释器指向不存在的路径 → E2 FAIL，且诊断文本**指出具体路径**。"""
    missing = tmp_path / "deleted-conda-env" / "python.exe"
    venv = _venv(tmp_path, home=str(missing))
    check = check_pyvenv_cfg(venv_root=venv)
    assert check.state is CheckState.FAIL
    assert str(missing) in check.detail


def test_missing_pyvenv_cfg_is_a_failure(tmp_path: Path) -> None:
    """没有 pyvenv.cfg 就无法判定基解释器 → FAIL（不是 UNKNOWN：环境本身残缺）。"""
    venv = _venv(tmp_path)
    assert check_pyvenv_cfg(venv_root=venv).state is CheckState.FAIL


def test_interpreter_outside_venv_fails(tmp_path: Path) -> None:
    """用系统 Python 跑出的结论属于**另一个环境**。"""
    venv = _venv(tmp_path, home=str(tmp_path))
    outside = tmp_path / "other" / "python.exe"
    outside.parent.mkdir(parents=True)
    outside.write_text("", encoding="utf-8")
    check = check_interpreter(venv_root=venv, executable=outside)
    assert check.state is CheckState.FAIL
    assert str(venv) in check.detail


def test_lock_consistency_detects_missing_and_extra(tmp_path: Path) -> None:
    """E3：锁文件少一项 / 多一项都必须 FAIL 并点名。"""
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "demo"\ndependencies = ["numpy>=1.26", "torch>=2.5"]\n',
        encoding="utf-8",
    )
    lock = tmp_path / "uv.lock"
    lock.write_text(
        'version = 1\n\n[[package]]\nname = "demo"\nversion = "0.1.0"\n'
        'dependencies = [\n    { name = "numpy" },\n    { name = "scipy" },\n]\n',
        encoding="utf-8",
    )
    check = check_lock_consistency(root=tmp_path)
    assert check.state is CheckState.FAIL
    assert "torch" in check.detail
    assert "scipy" in check.detail


def test_lock_consistency_passes_when_aligned(tmp_path: Path) -> None:
    """E3 正例：运行时依赖集合一致（版本与 extra 标记不影响判定）。"""
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "demo"\ndependencies = ["numpy>=1.26"]\n',
        encoding="utf-8",
    )
    (tmp_path / "uv.lock").write_text(
        'version = 1\n\n[[package]]\nname = "demo"\nversion = "0.1.0"\n'
        'dependencies = [\n    { name = "numpy" },\n]\n\n'
        '[package.optional-dependencies]\ntrain = [\n    { name = "pytorch-lightning" },\n]\n',
        encoding="utf-8",
    )
    assert check_lock_consistency(root=tmp_path).state is CheckState.PASS


def test_missing_lock_file_is_a_failure(tmp_path: Path) -> None:
    """没有锁文件就谈不上可复现。"""
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "demo"\n', encoding="utf-8")
    assert check_lock_consistency(root=tmp_path).state is CheckState.FAIL


def test_import_probe_three_states() -> None:
    """E4：必需项缺失 → FAIL；可选项缺失 → UNKNOWN（**不得**报 PASS）。"""

    def ok(_name: str) -> object:
        return object()

    def fail_numpy(name: str) -> object:
        if name == "numpy":
            raise ImportError(name)
        return object()

    def fail_optional(name: str) -> object:
        if name == "torch":
            raise ImportError(name)
        return object()

    assert check_imports(probe=ok).state is CheckState.PASS
    assert check_imports(probe=fail_numpy).state is CheckState.FAIL
    assert check_imports(probe=fail_optional).state is CheckState.UNKNOWN


def _report(*states: CheckState) -> EnvReport:
    checks = tuple(
        EnvCheck(id=f"E{index}", name="x", state=state, detail="d")
        for index, state in enumerate(states, start=1)
    )
    return EnvReport(checks=checks)


def test_exit_code_semantics() -> None:
    """退出码：全 PASS=0；有 FAIL=1；无 FAIL 但有 UNKNOWN=2（不知道 ≠ 没问题）。"""
    assert _report(CheckState.PASS, CheckState.PASS).exit_code == 0
    assert _report(CheckState.PASS, CheckState.UNKNOWN).exit_code == 2
    assert _report(CheckState.UNKNOWN, CheckState.FAIL).exit_code == 1


def test_real_repository_env_is_usable() -> None:
    """本仓库当前环境：E1/E2/E3/E5 不得 FAIL（可选依赖缺失只允许是 UNKNOWN）。"""
    report = run_env_doctor()
    failing = [check.id for check in report.checks if check.state is CheckState.FAIL]
    assert failing == [], report.format()
    assert len(report.checks) == 5
