"""评估报告落盘接线：图表对 -> EvalReport -> `metrics.json`（plan 06 §3.2 / §9-7；plan 07 §3.2 / §9-5）。

**现状（本模块补的那条线）**

- `runs/<experiment>/<timestamp>/metrics.json` 由 `infra/artifacts.RunArtifacts.write_metrics` 写盘；
  训练入口 `cli/train.py` 目前只写**训练摘要**（`TrainReport.to_metrics()` 的
  steps / *_loss / checkpoints / data_source / loss_history，再附 gates_passed 与 derived）；
- `EvalReport` 此前**没有落盘路径**（plan 06 §9-7 的「接线待做」、plan 07 §9-5 的六件套位置）。

**三条纪律**

1. **复用而非复制**：单张「生成谱 + 人类谱」-> EvalReport 直接走 `report.evaluate_charts`
   （plan §3.1 的全链路），本模块只做包装；分节（calibration / exploratory / legality）与文本
   渲染一律沿用 `report.py`，不另造一份。
2. **向后兼容**：训练摘要的顶层字段**不改名、不删除、不重排**；评估报告作为顶层 `eval`
   分节**追加在末尾**，其内部键顺序 == `EvalReport` 的声明顺序（即落盘 JSON 的键顺序，
   见 `EVAL_SECTION_ORDER`）；`assert_train_summary_preserved` 在合并点执行该校验。
3. **可追溯**：`meta.git_rev` / `meta.data_rev` 由 `ExperimentContext` 从实验上下文填充；
   git rev 与文件 sha1 复用 `beatmorph.infra.artifacts.git_rev` / `file_sha1`
   （**不写第二份实现**），但**惰性引入**——该 import 链会带 torch，而 `beatmorph.eval` 的
   模块级依赖要保持 numpy 级轻量（`tests/unit/eval/test_source_hygiene.py` 盯着这条）。

**接线用法（供 `cli/train.py` 采纳；本模块不越界改 `cli/`）**

    context = ExperimentContext.from_paths(
        repo_root=root, manifest_path=manifest_path, data_source=cfg.data.source
    )
    report = evaluate_chart_pair(pred, gold, config, decode_arm=arm, context=context)
    write_eval_metrics(artifacts, report, context=context)   # 并入已写的 metrics.json

**`data_rev` 的约定**（缺失时也必须是可读取值，不能是空字符串）

- 有清单文件：`sha1:<file_sha1(清单)>`（与 `cli/train.py` 的 data_rev 同口径）；
- 清单缺失：`sha1:missing`（`file_sha1` 的缺失值加同一前缀，恒非空）；
- 合成数据（`data_source="synthetic"`）：`synthetic`（无清单可留痕，与训练入口一致）。

⚠️ 训练入口的清单是 `data/processed/pairs.json`（`cfg.data.manifest_path`），而
**数据清单**是 `data/processed/charts.jsonl`（plan 06 §9-7 举的例子）；两者 sha1 不同。
**从训练入口接线时应显式传 `manifest_path=cfg.data.manifest_path`**，否则
`meta.data_rev` 与 checkpoint / `gates.txt` 里的 `data_rev` 将指向不同对象。
本模块的默认值取**数据清单**，只服务独立评估路径。

秒域（红线 7）：本模块的公共接口只有秒与元数据，不出现帧索引或 τ 格索引。
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from beatmorph.core.contracts.phigros import PhigrosChart
from beatmorph.core.logging import get_logger
from beatmorph.eval.calibration import CalibrationReadout, ExploratoryReadout
from beatmorph.eval.protocol import EvalConfig
from beatmorph.eval.report import (
    REQUIRED_REPORT_FIELDS,
    EvalReport,
    assert_report_schema,
    evaluate_charts,
)

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查期；运行期惰性引入（见模块 docstring 第 3 条）
    from beatmorph.infra.artifacts import RunArtifacts

__all__ = [
    "DATA_REV_MISSING",
    "DATA_REV_PREFIX",
    "DATA_REV_SYNTHETIC",
    "DATA_SOURCES",
    "DEFAULT_CHART_KEY",
    "DEFAULT_DATA_MANIFEST_PATH",
    "EVAL_SECTION_KEY",
    "EVAL_SECTION_ORDER",
    "TRAIN_SUMMARY_FIELDS",
    "ExperimentContext",
    "assert_metrics_schema",
    "assert_train_summary_preserved",
    "eval_section",
    "evaluate_chart_pair",
    "merge_eval_into_metrics",
    "with_experiment_context",
    "write_eval_metrics",
]

logger = get_logger("eval.pipeline")

# ══════════════════════════════════════════════════════════════
# 合并载荷的冻结名字与顺序（改动即改契约，测试会挡下）
# ══════════════════════════════════════════════════════════════

#: 评估报告在 `metrics.json` 里的分节名（顶层追加，恒为最后一个键）。
EVAL_SECTION_KEY: Final[str] = "eval"

#: `eval` 分节的**冻结字段顺序** == `EvalReport` 的声明顺序 == 落盘 JSON 的键顺序。
#:
#: 注意它与 `report.REQUIRED_REPORT_FIELDS` 是**同一份清单的两种用途**：那份负责
#: 「字段齐不齐」（成员检查），这里负责「键顺序是不是冻结的那一个」（顺序检查）。
EVAL_SECTION_ORDER: Final[tuple[str, ...]] = REQUIRED_REPORT_FIELDS

#: 训练摘要的顶层字段（`TrainReport.to_metrics()` 的键，顺序即其产出顺序）。
#: 合并时必须**逐字保留**：这些名字已被训练入口与集成测试消费（`cli/train.py` 在其上追加
#: `gates_passed` / `derived`，`tests/integration/test_train_entry.py` 读
#: `loss_history` / `data_source` / `gates_passed`），改名即破坏下游。
TRAIN_SUMMARY_FIELDS: Final[tuple[str, ...]] = (
    "steps",
    "first_loss",
    "last_loss",
    "best_loss",
    "checkpoints",
    "data_source",
    # 第六轮新增：续训留痕（None / 0 = 从头训练）。加字段必须同步本清单——
    # 生产侧的键顺序由 tests/unit/eval/test_pipeline.py::test_train_summary_field_list_matches_the_producer 锁死。
    "resumed_from",
    "resumed_step",
    "loss_history",
)

# ══════════════════════════════════════════════════════════════
# 数据版本（data_rev）的约定取值
# ══════════════════════════════════════════════════════════════

#: `data_rev` 的前缀：`sha1:<40 位十六进制>`（与 `cli/train.py` 同口径）。
DATA_REV_PREFIX: Final[str] = "sha1:"
#: 清单文件不存在时的**约定值**：同一前缀 + `file_sha1` 的缺失哨兵（恒非空，可审计）。
DATA_REV_MISSING: Final[str] = f"{DATA_REV_PREFIX}missing"
#: 合成数据没有清单可留痕（与训练入口的 data_rev 取值一致）。
DATA_REV_SYNTHETIC: Final[str] = "synthetic"
#: 允许的 `data_source` 取值（拼错即报错，不做静默降级）。
DATA_SOURCES: Final[tuple[str, ...]] = ("manifest", "synthetic")

#: 默认数据清单（`scripts/fetch_phira.py` 的产出；plan 06 §9-7 举的例子）。
#: 从训练入口接线时应传 `cfg.data.manifest_path`（见模块 docstring 的 ⚠️）。
DEFAULT_DATA_MANIFEST_PATH: Final[str] = "data/processed/charts.jsonl"

#: 单谱对便捷入口的默认谱面标识（报告主键，用于错误定位与 bootstrap 聚类）。
DEFAULT_CHART_KEY: Final[str] = "chart"


@dataclass(frozen=True, slots=True)
class ExperimentContext:
    """一次实验的可追溯上下文：`meta.git_rev` / `meta.data_rev` 的**唯一来源**。

    Attributes:
        git_rev: 代码版本（`beatmorph.infra.artifacts.git_rev`；无 git 时为 "unknown"）。
        data_rev: 数据版本（取值约定见模块 docstring；**恒非空**）。
        data_manifest: 参与 data_rev 的清单路径字符串；None = 合成数据（无清单）。
    """

    git_rev: str
    data_rev: str
    data_manifest: str | None = None

    def __post_init__(self) -> None:
        if not self.git_rev:
            raise ValueError("git_rev 不得为空（可追溯性是 plan 06 §3.2 的强制字段）")
        if not self.data_rev:
            raise ValueError("data_rev 不得为空（缺失请用约定值，不得留空字符串）")

    @classmethod
    def from_paths(
        cls,
        *,
        repo_root: Path,
        manifest_path: Path | None = None,
        data_source: str = "manifest",
    ) -> ExperimentContext:
        """由实验上下文（仓库根 + 数据清单）构造。

        Args:
            repo_root: 仓库根（git rev 的查询目录；也用于解析默认清单的相对路径）。
            manifest_path: 数据清单路径；None = `repo_root / DEFAULT_DATA_MANIFEST_PATH`。
            data_source: `manifest` 或 `synthetic`（与 `cfg.data.source` 同名）。

        Raises:
            ValueError: data_source 不是 DATA_SOURCES 之一（拼错不静默降级）。
        """
        # 惰性引入：beatmorph.infra.artifacts 的 import 链会带 torch（见模块 docstring 第 3 条）
        from beatmorph.infra.artifacts import file_sha1, git_rev

        if data_source not in DATA_SOURCES:
            raise ValueError(f"data_source 必须是 {DATA_SOURCES} 之一，得到 {data_source!r}")
        revision = git_rev(Path(repo_root))
        if data_source == "synthetic":
            return cls(git_rev=revision, data_rev=DATA_REV_SYNTHETIC, data_manifest=None)
        target = (
            Path(manifest_path)
            if manifest_path is not None
            else Path(repo_root) / DEFAULT_DATA_MANIFEST_PATH
        )
        data_rev = f"{DATA_REV_PREFIX}{file_sha1(target)}"
        if data_rev == DATA_REV_MISSING:
            logger.warning(
                "数据清单不存在：%s -> meta.data_rev 记约定值 %s", target, DATA_REV_MISSING
            )
        return cls(git_rev=revision, data_rev=data_rev, data_manifest=str(target))


def _resolved_rev(name: str, current: str, expected: str) -> str:
    """解决「报告里已有 rev」与「实验上下文 rev」的冲突（不一致即抛，不静默改写）。"""
    if current and current != expected:
        raise ValueError(
            f"报告 meta.{name}={current!r} 与实验上下文 {expected!r} 不一致："
            "不得静默改写可追溯字段（plan 06 §3.2）",
        )
    return expected or current


def with_experiment_context(report: EvalReport, context: ExperimentContext) -> EvalReport:
    """把实验上下文填进 `report.meta` 的 git_rev / data_rev（其它字段一个不动）。

    Raises:
        ValueError: 报告里已有非空 rev 且与上下文不一致（宁可失败，也不让同一份报告出现两个版本）。
    """
    meta = report.meta
    git_rev_value = _resolved_rev("git_rev", meta.git_rev, context.git_rev)
    data_rev_value = _resolved_rev("data_rev", meta.data_rev, context.data_rev)
    if (git_rev_value, data_rev_value) == (meta.git_rev, meta.data_rev):
        return report
    return report.model_copy(
        update={
            "meta": meta.model_copy(update={"git_rev": git_rev_value, "data_rev": data_rev_value})
        }
    )


def evaluate_chart_pair(
    pred: PhigrosChart,
    gold: PhigrosChart,
    config: EvalConfig,
    *,
    key: str = DEFAULT_CHART_KEY,
    calibration: CalibrationReadout | None = None,
    exploratory: ExploratoryReadout | None = None,
    seed: int | None = None,
    decode_arm: str = "",
    context: ExperimentContext | None = None,
) -> EvalReport:
    """单张「生成谱 + 人类谱」-> EvalReport（plan §3.1 的 EvalInput 全链路）。

    **本函数是 `report.evaluate_charts` 的薄包装**：谱面对先套上 key 再交给它，
    因此匹配 / 分解 / 合法性 / 分节全部只有一份实现（测试断言两条路径逐字段一致）。

    Args:
        pred: 生成谱（必须是已通过合法性校验的谱；本函数只统计违规率，不拦截）。
        gold: 人类谱（时间轴权威：BPMList 与难度元数据取自它）。
        config: 评估配置（双容差、主容差、相位搜索、归组规则）。
        key: 报告主键（也是 bootstrap 的 cluster 键；同曲多难度谱应传同一 `song`，
            单谱对入口暂不暴露该参数——多谱路径直接用 `evaluate_charts`）。
        calibration / exploratory: 校准与探索性读数；None = 显式缺失（**不得填 0**）。
        seed / decode_arm: meta 的可追溯字段。
        context: 实验上下文；给出时 git_rev / data_rev 由它填充。
    """
    return evaluate_charts(
        {key: pred},
        {key: gold},
        config,
        calibration=calibration,
        exploratory=exploratory,
        seed=seed,
        decode_arm=decode_arm,
        git_rev="" if context is None else context.git_rev,
        data_rev="" if context is None else context.data_rev,
    )


def _reject_nan(value: object, *, path: str) -> None:
    """冻结 JSON 里不得出现 NaN（plan 06 §9-5：分母为 0 记 0.0，缺失记 None）。

    `±inf` 不在此列：λ ≡ 0 的 NLL **就是** +∞（plan §4.6 的契约断言），
    是有含义的读数而不是缺失。
    """
    if isinstance(value, float):
        if math.isnan(value):
            raise AssertionError(f"{path} 出现 NaN：冻结 JSON 不允许 NaN（缺失请记 None）")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nan(item, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        for index, item in enumerate(value):
            _reject_nan(item, path=f"{path}[{index}]")


def eval_section(
    report: EvalReport,
    *,
    context: ExperimentContext | None = None,
) -> dict[str, Any]:
    """EvalReport -> `metrics.json` 的 `eval` 分节（JSON 原生类型，键顺序冻结）。

    Raises:
        AssertionError: 分节缺冻结字段、键顺序与冻结顺序不符、或出现 NaN。
    """
    effective = report if context is None else with_experiment_context(report, context)
    section: dict[str, Any] = effective.model_dump(mode="json")
    assert_report_schema(section)
    order = tuple(section)
    if order != EVAL_SECTION_ORDER:
        raise AssertionError(
            f"eval 分节的字段顺序与冻结顺序不一致：{order} != {EVAL_SECTION_ORDER}",
        )
    _reject_nan(section, path=EVAL_SECTION_KEY)
    return section


def assert_train_summary_preserved(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> None:
    """合并前后训练摘要必须逐字一致：不得改名、删除、改值或重排（向后兼容的机器判据）。

    Raises:
        AssertionError: 任一训练摘要字段缺失 / 取值改变 / 相对顺序改变。
    """
    keys_before = tuple(before)
    keys_after = tuple(after)
    if keys_after[: len(keys_before)] != keys_before:
        raise AssertionError(
            f"合并改动了既有顶层键的顺序：{keys_before} -> {keys_after[: len(keys_before)]}",
        )
    for name in TRAIN_SUMMARY_FIELDS:
        if name not in before:
            continue
        if name not in after:
            raise AssertionError(f"合并丢掉了训练摘要字段 {name!r}（向后兼容红线）")
        if after[name] != before[name]:
            raise AssertionError(f"合并改写了训练摘要字段 {name!r} 的取值（向后兼容红线）")


def merge_eval_into_metrics(
    payload: Mapping[str, Any] | None,
    report: EvalReport,
    *,
    context: ExperimentContext | None = None,
) -> dict[str, Any]:
    """把评估报告并入既有 metrics 载荷（训练摘要原样保留，`eval` 追加在末尾）。

    Args:
        payload: 既有载荷（通常是 `TrainReport.to_metrics()` 的返回值，可再带
            gates_passed / derived）；None 或空 = 只有评估分节。
        report: 评估报告。
        context: 实验上下文；给出时填充 `meta.git_rev` / `meta.data_rev`。

    Raises:
        ValueError: 载荷里**已有** `eval` 分节（不静默覆盖既有评估结果）。
        AssertionError: 合并破坏训练摘要，或报告分节不合冻结 schema。
    """
    merged: dict[str, Any] = dict(payload or {})
    if EVAL_SECTION_KEY in merged:
        raise ValueError(f"metrics 载荷里已有 {EVAL_SECTION_KEY!r} 分节：拒绝静默覆盖既有评估结果")
    merged[EVAL_SECTION_KEY] = eval_section(report, context=context)
    assert_train_summary_preserved(dict(payload or {}), merged)
    return merged


def assert_metrics_schema(payload: Mapping[str, Any]) -> None:
    """`metrics.json` 的冻结检查：`eval` 分节必须存在、齐全且键顺序正确。

    训练摘要字段**不**在这里强制（gates-only / 失败路径的载荷本来就没有 steps 等字段），
    它们的向后兼容由 `assert_train_summary_preserved` 在合并点保证。

    Raises:
        AssertionError: 缺 `eval` 分节、分节不是对象、字段缺失或键顺序不符。
    """
    if EVAL_SECTION_KEY not in payload:
        raise AssertionError(
            f"metrics 载荷缺少 {EVAL_SECTION_KEY!r} 分节：评估报告没有落盘路径（plan 06 §9-7）",
        )
    section = payload[EVAL_SECTION_KEY]
    if not isinstance(section, Mapping):
        raise AssertionError(
            f"{EVAL_SECTION_KEY!r} 分节必须是对象，得到 {type(section).__name__}",
        )
    assert_report_schema(section)
    order = tuple(section)
    if order != EVAL_SECTION_ORDER:
        raise AssertionError(
            f"eval 分节的字段顺序与冻结顺序不一致：{order} != {EVAL_SECTION_ORDER}",
        )


def _existing_metrics(artifacts: RunArtifacts) -> dict[str, Any]:
    """读回实验目录里已有的 metrics.json（不存在或为空 -> 空载荷）。"""
    from beatmorph.infra.artifacts import METRICS_FILENAME

    target = artifacts.path(METRICS_FILENAME)
    if not target.is_file():
        return {}
    text = target.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    loaded: Any = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"{target} 的顶层必须是对象，得到 {type(loaded).__name__}")
    return dict(loaded)


def write_eval_metrics(
    artifacts: RunArtifacts,
    report: EvalReport,
    *,
    payload: Mapping[str, Any] | None = None,
    context: ExperimentContext | None = None,
) -> Path:
    """把 EvalReport 并入 `runs/<experiment>/<timestamp>/metrics.json` 并写盘。

    `payload=None` 时**读回该实验目录已有的 metrics.json**（例如训练入口先写的训练
    摘要）再追加 `eval` 分节——「训练摘要 + 评估报告」因此落在同一个文件里，且训练入口
    那一侧的写法完全不用改（向后兼容的落点）。

    Returns:
        写盘路径。

    Raises:
        ValueError: 已有 metrics.json 里已含 `eval` 分节（不覆盖），或顶层不是对象。
        AssertionError: 合并破坏训练摘要，或报告分节不合冻结 schema。
    """
    base: Mapping[str, Any] = _existing_metrics(artifacts) if payload is None else payload
    merged = merge_eval_into_metrics(base, report, context=context)
    path = artifacts.write_metrics(merged)
    logger.info("评估报告已并入 %s（%d 张谱）", path, len(report.per_chart))
    return path
