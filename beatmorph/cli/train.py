"""训练入口 `beatmorph-train`（plan 07 §3.2 / M7.2 / M7.3 / M7.7）。

命令行契约：

    beatmorph-train --config-name <stage> [overrides...]
      --gates            # 先跑 G1-G4，结果落盘；FAIL 则以非 0 退出码中止
      --gates-only       # 只跑门禁，不进入正式训练

启动顺序（每一步都可能失败，且**失败即停**）：

1. 载入 structured config（缺字段 / 类型错 / provenance 为空 → 退出码 3）；
2. 打印派生常量并断言派生式（红线 7）；重新校验 override 之后的配置；
3. 环境自检 E1-E5（FAIL → 退出码 4；UNKNOWN 只告警，因为可选依赖缺失不等于环境坏了）；
4. 建实验目录与六件套骨架（不覆盖历史实验）；
5. 门禁：`--gates` 跑 G1-G4 并落盘；否则走 fail-closed（扩大数据规模却无全绿 gates.txt → 退出码 5）；
6. 训练（`run.backend` 选 torch 参考循环或 Lightning 目标栈）。

退出码：0 成功｜2 参数错误｜3 配置错误｜4 环境自检 FAIL｜5 门禁 FAIL / fail-closed 拒绝启动
｜6 缺少可选依赖｜7 训练异常。
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from beatmorph.core.logging import get_logger, setup_logging
from beatmorph.infra.artifacts import (
    GATES_FILENAME,
    METRICS_FILENAME,
    RunArtifacts,
    file_sha1,
)
from beatmorph.infra.checkpoint import list_checkpoints
from beatmorph.infra.config.loading import configs_dir, load_config
from beatmorph.infra.config.schema import (
    Backend,
    ConfigError,
    TrainConfig,
    assert_valid_config,
    data_scale_is_expanded,
)
from beatmorph.infra.derive import assert_derived_identities, derived_values, format_derived_banner
from beatmorph.infra.env_doctor import repo_root, run_env_doctor
from beatmorph.infra.gates import (
    GateFailure,
    enforce_gates,
    execute_gates,
    format_skipped_gates,
    gates_all_passed,
    gates_context,
)
from beatmorph.infra.smoke import SmokeBatchSource
from beatmorph.infra.train_loop import BatchSource, ManifestBatchSource, build_gate_inputs, train

__all__ = ["EXIT_CODES", "build_parser", "main"]

logger = get_logger("cli.train")

EXIT_OK = 0
EXIT_ARGS = 2
EXIT_CONFIG = 3
EXIT_ENV = 4
EXIT_GATES = 5
EXIT_DEPENDENCY = 6
EXIT_TRAIN = 7

#: 退出码语义（文档与测试共用一份）
EXIT_CODES: dict[int, str] = {
    EXIT_OK: "成功",
    EXIT_ARGS: "参数错误",
    EXIT_CONFIG: "配置错误",
    EXIT_ENV: "环境自检 FAIL",
    EXIT_GATES: "门禁 FAIL / fail-closed 拒绝启动",
    EXIT_DEPENDENCY: "缺少可选训练依赖",
    EXIT_TRAIN: "训练异常",
}


def build_parser() -> argparse.ArgumentParser:
    """构造参数解析器（`overrides` 为点号覆盖串，如 `optim.lr=1e-4`）。"""
    parser = argparse.ArgumentParser(prog="beatmorph-train", description="BeatMorph 训练入口")
    parser.add_argument("--config-name", default="smoke", help="configs/ 下的配置名（不含 .yaml）")
    parser.add_argument("--config-dir", default=None, help="配置目录（默认 <repo>/configs）")
    parser.add_argument("--runs-dir", default=None, help="产物根目录（默认取配置里的值）")
    parser.add_argument("--experiment", default=None, help="实验名（默认取配置里的值）")
    parser.add_argument("--max-steps", type=int, default=None, help="覆盖 optim.max_steps")
    parser.add_argument("--device", default="cpu", help="设备（默认 cpu；有 CUDA 时可用 cuda）")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        metavar="RUN_DIR",
        help=(
            "断点续训：从指定实验目录（或 latest = 本实验最近一个**有 checkpoint** 的目录）续跑。"
            "续训不重跑门禁，但要求该目录的 gates.txt 与 checkpoint 的 meta 都是全绿；"
            "配置/数据版本不一致会**拒绝恢复**（fail-closed）"
        ),
    )
    parser.add_argument("--gates", action="store_true", help="训练前强制跑 G1-G4")
    parser.add_argument("--gates-only", action="store_true", help="只跑门禁，不训练")
    parser.add_argument(
        "--no-gates", action="store_true", help="不跑门禁（扩大规模时会被 fail-closed 拒绝）"
    )
    parser.add_argument(
        "--skip-env-doctor",
        action="store_true",
        help="跳过 E1-E5 环境自检（**仅供测试/容器**：跳过它等于放弃环境这条门禁）",
    )
    parser.add_argument("overrides", nargs="*", help="点号覆盖串，如 optim.lr=1e-4")
    return parser


def _resolve(path_text: str, root: Path) -> Path:
    """相对路径按仓库根解析（避免「在哪个目录敲命令」改变产物位置）。"""
    candidate = Path(path_text)
    return candidate if candidate.is_absolute() else root / candidate


def _read_text(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.is_file() else None


def _open_resume_target(
    target: str,
    *,
    runs_dir: Path,
    experiment: str,
    root: Path,
) -> tuple[RunArtifacts, Path]:
    """解析续训目标，返回 (实验目录句柄, 起点 checkpoint)。

    目标可以是：`latest`（本实验最近一个**有 step checkpoint** 的目录）、运行目录名
    （`20260927-043828`）、或直接给路径。找目录与找 checkpoint 是两件事：最近的一个目录
    可能是只跑了 `--gates-only` 的（没有 checkpoint），因此要**逐个往前找**。

    Raises:
        FileNotFoundError: 目标目录不存在，或目录里没有可续训的起点。
    """
    experiment_dir = runs_dir / experiment
    if target == "latest":
        candidates = sorted(
            (path for path in experiment_dir.glob("*") if path.is_dir()),
            key=lambda path: path.name,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(f"{experiment_dir} 下没有任何实验目录")
        for candidate in candidates:
            if list_checkpoints(candidate):
                return RunArtifacts.open_existing(candidate), list_checkpoints(candidate)[-1][1]
        raise FileNotFoundError(f"{experiment_dir} 下的实验目录都没有 step-*.pt：没有可续训的起点")
    given = Path(target)
    for candidate in (given, experiment_dir / target, root / target):
        if candidate.is_dir():
            artifacts = RunArtifacts.open_existing(candidate)
            found = list_checkpoints(candidate)
            if not found:
                raise FileNotFoundError(f"{candidate}/checkpoints 下没有 step-*.pt")
            return artifacts, found[-1][1]
    raise FileNotFoundError(f"续训目标不存在：{target}")


def main(argv: Sequence[str] | None = None) -> int:  # noqa: PLR0911, PLR0912, PLR0915 - 启动编排
    """执行一次训练/门禁运行，返回退出码。"""
    args = build_parser().parse_args(argv)
    setup_logging()
    root = repo_root()

    # ── 1. 配置 ────────────────────────────────────────────────
    directory = Path(args.config_dir) if args.config_dir else configs_dir(root)
    try:
        cfg = load_config(
            config_name=args.config_name, directory=directory, overrides=args.overrides
        )
    except ConfigError as exc:
        logger.error("配置错误（退出码 %d）：%s", EXIT_CONFIG, exc)
        return EXIT_CONFIG

    # ── 2. 派生量：打印 + 断言（红线 7）─────────────────────────
    try:
        assert_derived_identities()
    except AssertionError as exc:
        logger.error("派生式断言失败：%s", exc)
        return EXIT_CONFIG
    logger.info("\n%s", format_derived_banner())

    if args.runs_dir:
        cfg.run.runs_dir = args.runs_dir
    if args.experiment:
        cfg.run.experiment = args.experiment
    if args.max_steps:
        cfg.optim.max_steps = args.max_steps
    try:
        assert_valid_config(cfg)
    except ConfigError as exc:
        logger.error("覆盖之后的配置不合法（退出码 %d）：%s", EXIT_CONFIG, exc)
        return EXIT_CONFIG

    # ── 3. 环境自检 ────────────────────────────────────────────
    if args.skip_env_doctor:
        logger.warning("已跳过环境自检（--skip-env-doctor）：环境这条门禁本轮不生效")
    else:
        env_report = run_env_doctor(root=root)
        logger.info("\n%s", env_report.format())
        if env_report.has_failure:
            logger.error("环境自检 FAIL（退出码 %d）：先修环境，再谈训练", EXIT_ENV)
            return EXIT_ENV

    # ── 4. 实验目录与六件套骨架 ─────────────────────────────────
    runs_dir = _resolve(cfg.run.runs_dir, root)
    resume_from: Path | None = None
    if args.resume is not None:
        try:
            artifacts, resume_from = _open_resume_target(
                args.resume, runs_dir=runs_dir, experiment=cfg.run.experiment, root=root
            )
        except (FileNotFoundError, AssertionError) as exc:
            logger.error("续训目标无法解析（退出码 %d）：%s", EXIT_ARGS, exc)
            return EXIT_ARGS
        logger.info("续训模式：实验目录 %s，起点 %s", artifacts.root, resume_from)
    else:
        artifacts = RunArtifacts.create(runs_dir=runs_dir, experiment=cfg.run.experiment)
    artifacts.write_config(cfg)
    manifest_path = _resolve(cfg.data.manifest_path, root) if cfg.data.manifest_path else None
    artifacts.write_provenance(cfg, extra={"script_rev": derived_values().__str__()})
    logger.info("实验目录：%s", artifacts.root)

    source = _build_source(cfg, manifest_path)
    if cfg.data.source == "synthetic":
        data_rev = "synthetic"
    else:
        data_rev = f"sha1:{file_sha1(manifest_path) if manifest_path is not None else 'missing'}"

    # ── 5. 门禁 ────────────────────────────────────────────────
    gates_green = False
    if resume_from is not None and not (args.gates or args.gates_only):
        # 续训**不重跑**门禁（那是几十到几十分钟的 GPU 预算），但要求原实验的 gates.txt 全绿；
        # checkpoint 的 meta 里也快照了一次 gates_green，load_checkpoint 会再校验一次。
        try:
            enforce_gates(run_dir=artifacts.root, cfg=cfg)
        except GateFailure as exc:
            logger.error("续训的 fail-closed 校验失败（退出码 %d）：%s", EXIT_GATES, exc)
            return EXIT_GATES
        gates_green = gates_all_passed(_read_text(artifacts.path(GATES_FILENAME))) is True
    elif args.gates or args.gates_only:
        context = gates_context(
            root=root,
            data_source=source.describe(),
            data_path=manifest_path if cfg.data.source == "manifest" else None,
        )
        try:
            inputs, stats = build_gate_inputs(cfg, source, device=args.device)
        except Exception as exc:  # 数据侧问题也要以明确退出码收场
            logger.error("门禁输入装配失败（退出码 %d）：%s", EXIT_TRAIN, exc)
            artifacts.write_metrics({"gates_passed": False, "reason": str(exc)})
            return EXIT_TRAIN
        context.update({name: f"{value:g}" for name, value in stats.items()})
        try:
            execute_gates(
                inputs,
                cfg,
                gates_path=artifacts.path(GATES_FILENAME),
                context=context,
                log_dir=artifacts.logs,
            )
        except GateFailure as exc:
            logger.error("门禁未通过（退出码 %d）：%s", EXIT_GATES, exc)
            artifacts.write_metrics({"gates_passed": False, "reason": str(exc)})
            return EXIT_GATES
        gates_green = True
    else:
        try:
            enforce_gates(run_dir=artifacts.root, cfg=cfg)
        except GateFailure as exc:
            logger.error("fail-closed 拒绝启动（退出码 %d）：%s", EXIT_GATES, exc)
            return EXIT_GATES
        # 六件套缺一即视为实验不可信：把「门禁未运行」这件事写进 gates.txt，
        # 而不是让文件缺失（缺失与漏写在事后无法区分）。
        artifacts.write_gates(
            format_skipped_gates(
                cfg,
                reason=(
                    "数据规模在冒烟范围内，且未传 --gates"
                    if not data_scale_is_expanded(cfg)
                    else "显式 --no-gates（仅冒烟规模允许）"
                ),
            ),
        )
        gates_green = gates_all_passed(_read_text(artifacts.path(GATES_FILENAME))) is True

    if args.gates_only:
        artifacts.write_metrics(
            {
                "gates_passed": gates_green,
                "steps": 0,
                "note": "gates-only 运行：不含训练产物",
                "derived": derived_values(),
            },
        )
        artifacts.assert_complete()
        logger.info("只跑了门禁（gates_only）：%s", artifacts.path(METRICS_FILENAME))
        return EXIT_OK

    # ── 6. 训练 ────────────────────────────────────────────────
    try:
        if cfg.run.backend_kind is Backend.LIGHTNING:
            from beatmorph.infra.lightning_module import run_lightning

            report = run_lightning(
                cfg,
                source=source,
                artifacts=artifacts,
                data_rev=data_rev,
                gates_green=gates_green,
            )
        else:
            report = train(
                cfg,
                source=source,
                artifacts=artifacts,
                data_rev=data_rev,
                gates_green=gates_green,
                device=args.device,
                resume_from=resume_from,
            )
    except ImportError as exc:
        logger.error("缺少可选训练依赖（退出码 %d）：%s", EXIT_DEPENDENCY, exc)
        artifacts.write_metrics({"gates_passed": gates_green, "reason": str(exc)})
        return EXIT_DEPENDENCY
    except Exception as exc:
        logger.exception("训练异常（退出码 %d）", EXIT_TRAIN)
        artifacts.write_metrics(
            {"gates_passed": gates_green, "reason": f"{type(exc).__name__}: {exc}"}
        )
        return EXIT_TRAIN

    metrics = report.to_metrics()
    metrics["gates_passed"] = gates_green
    metrics["derived"] = derived_values()
    artifacts.write_metrics(metrics)
    artifacts.assert_complete()
    logger.info("完成：%s", report.format())
    return EXIT_OK


def _build_source(cfg: TrainConfig, manifest_path: Path | None) -> BatchSource:
    """按 `data.source` 选择批次来源（显式声明，不做静默降级）。"""
    if cfg.data.source == "synthetic":
        logger.warning("data.source=synthetic：使用合成谱跑门禁/冒烟，**不代表**真实数据通路已验证")
        return SmokeBatchSource(cfg, seed=cfg.optim.seed)
    if manifest_path is None or not manifest_path.is_file():
        raise ConfigError(f"清单不存在：{manifest_path}（data.source=manifest 时必须存在）")
    return ManifestBatchSource(cfg, split=cfg.data.split_train, seed=cfg.optim.seed)


if __name__ == "__main__":  # pragma: no cover - 控制台入口
    raise SystemExit(main())
