"""门禁执行器：G1-G4 的组装、落盘与 fail-closed 语义（plan 07 §4.3 / M7.2）。

`beatmorph/infra/sanity.py` 提供**范式中立的四道门禁**（只吃 `step_fn`）；
本模块负责把它接进训练流程，并保证三件事：

1. **生效阈值必须落盘**：默认值可覆盖但不可忽略（§3.1 / §9-2），因此 `gates.txt` 里既有
   `summarize()` 原文，也有本次实际使用的阈值与上下文（git rev / data rev / 数据来源）；
2. **触发条件写死**：① 新增或修改任何训练目标/损失；② 把数据规模扩到超过冒烟规模。
   二者任一发生时，实验目录必须有**全绿**的 `gates.txt`，否则**拒绝启动**（退出码非 0）；
3. **门禁未绿不得扩数据**（CLAUDE.md §5.8 / BasePlan §9）：判定基准写在配置里
   （`gates.smoke_max_samples`），不靠记性。

TB 标量是**附加**记录：tensorboard 缺失时只告警，不改变门禁结论（权威记录始终是 gates.txt）。
"""

from __future__ import annotations

import copy
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from beatmorph.core.contracts import MERT_FRAME_RATE_HZ
from beatmorph.core.logging import get_logger
from beatmorph.infra.artifacts import GATES_FILENAME, file_sha1, git_rev
from beatmorph.infra.config.schema import GatesConfig, TrainConfig, data_scale_is_expanded
from beatmorph.infra.sanity import (
    GateResult,
    StepFn,
    constant_baseline_gate,
    frame_rate_gate,
    overfit_single_batch,
    shuffled_target_control,
    summarize,
)

__all__ = [
    "GateFailure",
    "GateInputs",
    "bounded_gate_config",
    "enforce_gates",
    "execute_gates",
    "format_gates_text",
    "format_skipped_gates",
    "gates_all_passed",
    "gates_context",
    "parse_gate_results",
    "run_gates",
    "thresholds_of",
    "write_tb_scalars",
]

logger = get_logger("infra.gates")

_RESULT_LINE = re.compile(r"^\s*\[(PASS|FAIL)\]\s+(?P<name>.+?)(?::\s(?P<detail>.*))?$")


class GateFailure(RuntimeError):  # noqa: N818 - 名字表达语义（门禁失败），非通用错误类型
    """门禁失败或 fail-closed 拒绝启动（训练必须中止，退出码非 0）。"""


@dataclass(frozen=True)
class GateInputs:
    """四道门禁所需的全部输入（**数据来源无关**：真实数据与冒烟数据都走这里）。

    Attributes:
        step_fn_real: **G1** 的真实臂 = 训练路径（遮盖补全）上跑一步优化。
        step_fn_g2_real: **G2** 的真实臂：必须与 `step_fn_g2_shuffled` **同批、同损失、
            同起点**（plan 07 §4.3「同模型同输入」）。它**不是** `step_fn_real`：
            G1 走的是遮盖路径（重标定 `1/r` + 只监督被遮盖事件），与打乱臂的
            全事件目标不在同一测度上，两者的绝对 loss 不可比。
        step_fn_g2_shuffled: **G2** 的打乱臂（目标被置换，输入随之失去信息）。
        model_loss: 模型在训练/验证集上的 loss（G3）。
        baseline_loss: 常数基线 loss（`λ = N/|Ω|`，**不是** λ ≡ 0）。
        frames: 实际音频特征帧数（G4）。
        duration_s: 对应音频时长（秒）。
        frame_rate: **由 config/契约派生**的帧率（G4；不得硬编码）。
    """

    step_fn_real: StepFn
    step_fn_g2_real: StepFn
    step_fn_g2_shuffled: StepFn
    model_loss: float
    baseline_loss: float
    frames: int
    duration_s: float
    frame_rate: float


def thresholds_of(cfg: GatesConfig) -> dict[str, float | int]:
    """本次**实际生效**的门禁阈值（落盘用；§3.1 要求显式记录）。"""
    return {
        "g1_steps": cfg.overfit_steps,
        "g1_target_loss": cfg.overfit_target_loss,
        "g1_target_ratio": cfg.overfit_target_ratio,
        "g2_steps": cfg.shuffle_steps,
        "g2_samples": cfg.shuffle_samples,
        "g2_chunks": cfg.shuffle_chunks,
        "g2_min_gap_ratio": cfg.shuffle_min_gap_ratio,
        "g3_min_improvement": cfg.baseline_min_improvement,
        "g3_normalized": int(cfg.constant_baseline_normalized),
        "g4_tol_frames": cfg.frame_rate_tol_frames,
    }


def run_gates(
    inputs: GateInputs,
    cfg: GatesConfig,
    *,
    progress: Callable[[str], None] | None = None,
) -> list[GateResult]:
    """按 `sanity.py` 的判据跑完 G1-G4（**只组装，不落盘、不抛错**）。

    `progress` 是逐门禁进度回调（默认 None = 静默）。**为什么必须有它**：
    正常 K 下门禁的预算是 **~38-45 min**（G2 = 200 步 x 16 段串行前向），而此前
    `execute_gates` 直到全部跑完才写第一行日志 ⇒ 外部**无法区分「慢」与「卡死」**，
    只能靠功耗 / util 猜。2026-09-27 实测踩坑：据此把一个**正常在跑**的门禁误判为卡死、
    掐掉两次（浪费约 40 min GPU）。逐门禁的耗时日志是那个误判的唯一解药。
    """
    stages: list[tuple[str, Callable[[], GateResult]]] = [
        (
            "G1 单batch过拟合",
            lambda: overfit_single_batch(
                inputs.step_fn_real,
                steps=cfg.overfit_steps,
                target_loss=cfg.overfit_target_loss,
                target_ratio=cfg.overfit_target_ratio,
            ),
        ),
        (
            "G2 打乱标签对照",
            lambda: shuffled_target_control(
                inputs.step_fn_g2_real,
                inputs.step_fn_g2_shuffled,
                steps=cfg.shuffle_steps,
                min_gap_ratio=cfg.shuffle_min_gap_ratio,
            ),
        ),
        (
            "G3 常数基线",
            lambda: constant_baseline_gate(
                inputs.model_loss,
                inputs.baseline_loss,
                min_improvement=cfg.baseline_min_improvement,
            ),
        ),
        (
            "G4 帧率契约",
            lambda: frame_rate_gate(
                inputs.frames,
                inputs.duration_s,
                inputs.frame_rate,
                tol_frames=cfg.frame_rate_tol_frames,
            ),
        ),
    ]
    results: list[GateResult] = []
    for name, run in stages:
        if progress is not None:
            progress(f"门禁 {name}：开始")
        started = time.perf_counter()
        result = run()
        results.append(result)
        if progress is not None:
            state = "PASS" if result.passed else "FAIL"
            progress(f"门禁 {name}：{state}（{time.perf_counter() - started:.1f}s）{result.detail}")
    return results


def bounded_gate_config(cfg: TrainConfig) -> TrainConfig:
    """门禁用的**有界**数据口径：返回 `data.max_samples = gates.gate_samples` 的配置副本。

    为什么必须单独给门禁限界（2026-09-27 第八轮实测）：门禁要跑 300 + 2x100 + 100 步，
    单步成本 ∝ (K·T)²。采样器修好之后（[RFC-0033](../../docs/decisions/RFC-0033-sampler-coverage-and-epoch.md)）
    门禁批从**全库**按剩余窗口加权抽桶，抽中的是最大的那批桶（K 可到 `k_max=128`）——
    实测门禁从 **5 min 涨到 26 min**，显存贴到 **7880/8151 MiB**，正好落在
    docs/TRAINING.md §7.5 记录的「滑进 Windows 共享内存」危险区。有界切片使它同时便宜、可复现，
    并与第五轮的权威记录（`data.max_samples=200`）同口径。

    Returns:
        `gates.gate_samples is None` 时原样返回 `cfg`；否则返回深拷贝并只改 `data.max_samples`。
        **训练侧仍用原始 cfg**——这个副本只喂 `build_gate_inputs` 与 `gates_context`。
    """
    if cfg.gates.gate_samples is None:
        return cfg
    bounded = copy.deepcopy(cfg)
    bounded.data.max_samples = int(cfg.gates.gate_samples)
    # 门禁**不走 worker**（RFC-0034 §5）：门禁批只有 1~16 个样本，worker 的启动成本
    # （Windows spawn 要重新 import torch）会盖过收益，而门禁要的是**确定的耗时与显存**。
    # 训练侧仍按 `data.workers` 取批——它才是有 20,000 步的地方。
    bounded.data.workers = 0
    return bounded


def gates_context(
    *,
    root: Path,
    data_source: str,
    data_path: Path | None = None,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """门禁上下文（git rev / data rev / 数据来源 / 派生帧率）。"""
    context = {
        "git_rev": git_rev(Path(root)),
        "data_source": data_source,
        "data_rev": "n/a" if data_path is None else f"sha1:{file_sha1(Path(data_path))}",
        "frame_rate_hz": f"{MERT_FRAME_RATE_HZ:g}（派生：MERT_SAMPLE_RATE_HZ / MERT_CONV_STRIDE_PRODUCT）",
    }
    if extra:
        context.update({key: str(value) for key, value in extra.items()})
    return context


def format_gates_text(
    results: Sequence[GateResult],
    *,
    thresholds: Mapping[str, float | int],
    context: Mapping[str, str],
) -> str:
    """`summarize()` 原文 + 生效阈值 + 上下文（可直接贴进训练日志）。"""
    lines = [summarize(list(results)), "", "生效阈值（默认值可覆盖但不可忽略，plan 07 §3.1）："]
    lines += [f"  {name} = {value}" for name, value in thresholds.items()]
    lines += ["", "上下文："]
    lines += [f"  {name} = {value}" for name, value in context.items()]
    n_fail = sum(1 for result in results if not result.passed)
    lines += ["", f"结论：{'全绿' if n_fail == 0 else f'{n_fail} 项 FAIL —— 不得扩大数据规模'}"]
    return "\n".join(lines)


def format_skipped_gates(cfg: TrainConfig, *, reason: str = "未传 --gates") -> str:
    """记录「门禁**没有运行**」这件事本身。

    为什么必须留痕：{BT}gates.txt{BT} 缺失与「跑了但没写」在事后无法区分，而六件套缺一即视为
    实验不可信（§3.2）。因此这里写一份**明确不可判定**的记录——它不含任何 {BT}[PASS]{BT}/{BT}[FAIL]{BT}
    行，因此 {BT}gates_all_passed(){BT} 返回 None（「不知道」），任何把文件存在当成「门禁绿了」的
    读法都会在解析这一步落空。
    """
    scale = "全量" if cfg.data.max_samples is None else str(cfg.data.max_samples)
    return "\n".join(
        [
            "健全性门禁（G1-G4）：**未运行**",
            f"  原因：{reason}",
            f"  数据规模：max_samples={scale}（冒烟上限 {cfg.gates.smoke_max_samples}）",
            "  触发条件提示：新增/修改训练目标或损失、或把数据规模扩到超过冒烟规模时，"
            "必须先跑 beatmorph-train --gates（plan 07 §4.3）",
            "  => 未运行：本文件**不含** [PASS]/[FAIL] 行，不得被解读为门禁通过",
        ],
    )


def parse_gate_results(text: str) -> list[GateResult]:
    """从 `gates.txt` 反解出 `GateResult`（缺失/格式不符时返回空列表）。"""
    results: list[GateResult] = []
    for line in text.splitlines():
        match = _RESULT_LINE.match(line)
        if match is None:
            continue
        results.append(
            GateResult(
                name=match.group("name").strip(),
                passed=match.group(1) == "PASS",
                detail=(match.group("detail") or "").strip(),
            ),
        )
    return results


def gates_all_passed(text: str | None) -> bool | None:
    """`gates.txt` 是否全绿。

    Returns:
        True / False；`text` 为 None 或解析不出任何一行结果时返回 **None**（= 无法判定，
        **不得**当成通过——「不知道」不等于「没问题」）。
    """
    if text is None:
        return None
    results = parse_gate_results(text)
    if not results:
        return None
    return all(result.passed for result in results)


def enforce_gates(*, run_dir: Path, cfg: TrainConfig) -> None:
    """fail-closed：扩大数据规模时，实验目录必须有**全绿**的 `gates.txt`。

    Raises:
        GateFailure: 缺少 gates.txt / 无法判定 / 存在 FAIL。
    """
    if not cfg.gates.required:
        logger.warning("gates.required=False：跳过门禁强制校验（只允许在冒烟/调试配置里出现）")
        return
    if not data_scale_is_expanded(cfg):
        logger.info(
            "数据规模在冒烟范围内（max_samples=%s <= %s）：不强制 gates.txt",
            cfg.data.max_samples,
            cfg.gates.smoke_max_samples,
        )
        return
    path = Path(run_dir) / GATES_FILENAME
    text = path.read_text(encoding="utf-8") if path.is_file() else None
    if text is None:
        raise GateFailure(
            f"缺少 {path}：data.max_samples={cfg.data.max_samples} 超过冒烟规模 "
            f"{cfg.gates.smoke_max_samples}，扩大数据规模前必须先跑通 G1-G4"
            "（beatmorph-train --gates）",
        )
    passed = gates_all_passed(text)
    if passed is None:
        raise GateFailure(f"{path} 无法判定（没有可解析的 [PASS]/[FAIL] 行）：拒绝启动")
    if not passed:
        failures = [result.name for result in parse_gate_results(text) if not result.passed]
        raise GateFailure(f"{path} 含 FAIL（{failures}）：门禁未绿不得扩大数据规模")


def write_tb_scalars(log_dir: Path, results: Sequence[GateResult]) -> bool:
    """把门禁布尔值写成 TB 标量 `gate/<Gn>_pass`（tensorboard 缺失时只告警）。

    Returns:
        True 表示已写入；False 表示 tensorboard 不可用（**不改变**门禁结论）。
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        logger.warning("tensorboard 不可用：gate/* 标量未写入（gates.txt 仍是权威记录）")
        return False
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    try:
        for result in results:
            tag = "gate/" + result.name.split()[0] + "_pass"
            writer.add_scalar(tag, float(result.passed), 0)
    finally:
        writer.close()
    return True


def execute_gates(
    inputs: GateInputs,
    cfg: TrainConfig,
    *,
    gates_path: Path,
    context: Mapping[str, str],
    log_dir: Path | None = None,
) -> list[GateResult]:
    """跑门禁 → 落盘 → 写 TB → **任一 FAIL 即抛** `GateFailure`。

    Raises:
        GateFailure: 存在 FAIL（训练必须中止，退出码非 0）。
    """
    results = run_gates(inputs, cfg.gates, progress=lambda message: logger.info("%s", message))
    text = format_gates_text(results, thresholds=thresholds_of(cfg.gates), context=context)
    target = Path(gates_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    logger.info("门禁结果已写入 %s\n%s", target, summarize(results))
    if log_dir is not None:
        write_tb_scalars(log_dir, results)
    failures = [result for result in results if not result.passed]
    if failures:
        raise GateFailure(
            "门禁未通过：" + "；".join(f"{result.name} -> {result.detail}" for result in failures),
        )
    return results
