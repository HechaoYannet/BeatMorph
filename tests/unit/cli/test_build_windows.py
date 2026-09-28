"""`beatmorph-build-windows` 入口：参数解析与配置目录解析（plan 07 §9-52）。

回归重点：
1. `--config-dir` 缺省时**不得**把 `None` 传给 `load_config`（此前 `main([])` 会以 TypeError
   落进 `EXIT_CONFIG`，即默认用法开箱即坏）；
2. `--shard-windows 0` **不得**被 `or` 静默吞成默认值（那会让 `< 1` 校验变成死代码，
   本轮实测：测试里因此跑起了一次**全库构建**）。

"""

from __future__ import annotations

from pathlib import Path

import pytest

from beatmorph.cli import build_windows
from beatmorph.data import window_build
from beatmorph.infra.config.loading import configs_dir
from beatmorph.infra.env_doctor import repo_root


class _StubReport:
    """`main` 只用到 `report.describe()`。"""

    def describe(self) -> str:
        return "stub report"


def test_parser_defaults() -> None:
    """缺省值就是生产默认：配置文件默认名、config-dir 交给仓库根解析。"""
    args = build_windows.build_parser().parse_args([])
    assert args.config_name == "phigros_masked"
    assert args.config_dir is None
    assert args.split is None
    assert args.out is None
    assert args.jobs == 1
    assert args.shard_windows is None
    assert args.limit is None


def test_main_default_config_dir_loads_repo_configs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """★ 回归 1：缺省 `--config-dir` 必须解析到 `<repo>/configs`，而不是 `None`。"""
    captured: dict[str, object] = {}
    real_load = build_windows.load_config

    def spy_load(*, config_name: str, directory: Path, overrides: tuple[str, ...] = ()):
        captured["directory"] = Path(directory)
        return real_load(config_name=config_name, directory=directory, overrides=overrides)

    def fake_build_window_cache(config, **kwargs):
        captured["root"] = Path(kwargs["root"])
        captured["split"] = config.split
        captured["limit"] = config.limit
        captured["jobs"] = kwargs["jobs"]
        return _StubReport()

    monkeypatch.setattr(build_windows, "load_config", spy_load)
    monkeypatch.setattr(window_build, "build_window_cache", fake_build_window_cache)

    # `--limit 1` 是**安全网**：万一 patch 没生效，真构建也只做 1 行而不是全库。
    rc = build_windows.main(["--out", str(tmp_path), "--jobs", "3", "--limit", "1"])

    assert rc == build_windows.EXIT_OK
    assert captured["directory"] == configs_dir(repo_root())
    assert captured["root"] == tmp_path
    assert captured["jobs"] == 3
    assert isinstance(captured["split"], str)
    assert captured["limit"] == 1


def test_main_explicit_config_dir_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """显式 `--config-dir` 必须被采用；里面没有配置时**返回 EXIT_CONFIG 而不是抛**。"""
    captured: dict[str, object] = {}
    real_load = build_windows.load_config

    def spy_load(*, config_name: str, directory: Path, overrides: tuple[str, ...] = ()):
        captured["directory"] = Path(directory)
        return real_load(config_name=config_name, directory=directory, overrides=overrides)

    monkeypatch.setattr(build_windows, "load_config", spy_load)
    rc = build_windows.main(["--config-dir", str(tmp_path), "--out", str(tmp_path), "--limit", "1"])

    assert rc == build_windows.EXIT_CONFIG
    assert captured["directory"] == tmp_path


def test_main_rejects_bad_jobs(tmp_path: Path) -> None:
    """`--jobs < 1` 与 `--shard-windows < 1` 都是参数错误，**不得**退化成默认值。"""
    assert (
        build_windows.main(["--jobs", "0", "--limit", "1", "--out", str(tmp_path)])
        == build_windows.EXIT_ARGS
    )
    assert (
        build_windows.main(
            ["--jobs", "2", "--shard-windows", "0", "--limit", "1", "--out", str(tmp_path)]
        )
        == build_windows.EXIT_ARGS
    )
