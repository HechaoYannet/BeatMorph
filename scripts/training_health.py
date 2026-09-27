"""训练健康检查（长跑运维；plan 07 §4.6 / docs/TRAINING.md §7.5）。

为什么需要它：全量训练是**十小时级**的进程，而「有没有在跑、有没有降速、显存有没有贴顶、
ETA 还有多久」在只看 TB 曲线时要人眼盯。本脚本把这几件事压成一行结论 + 退出码，
于是 agent 可以每 2 小时调用一次（--watch 7200）而不必阅读日志。

读的是**落盘的事实**（不是进程内存）：

- logs/loss_history.jsonl：每 run.log_every 步一行（step / loss / step_time_s / grad_norm / peak_vram_gib）
- checkpoints/step-*.pt：最近一次存盘（崩溃恢复的起点）
- gates.txt / config.yaml：门禁结论与预算（max_steps）

退出码：0 = 健康｜1 = 有告警（降速 / 贴顶 / 停住）｜2 = 没有可用数据。
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml

#: 显存贴顶告警阈值（GiB）。8 GB 卡上超过它就会开始滑进 Windows 共享内存 ⇒ 步时放大。
VRAM_WARN_GIB: float = 7.5
#: 步时退化告警比例（后一半窗口 / 前一半窗口）。
STEP_TIME_DEGRADE_RATIO: float = 1.3
#: 标量停更告警阈值（秒）：超过它说明进程可能已停或卡住。
STALE_WARN_S: float = 1800.0


def _find_run_dir(root: Path, experiment: str, explicit: str | None) -> Path:
    """定位实验目录：显式路径 > 最近一个**有 loss_history.jsonl** 的目录。"""
    if explicit:
        path = Path(explicit)
        if not path.is_dir():
            raise SystemExit(f"实验目录不存在：{path}")
        return path
    base = root / "runs" / experiment
    candidates = sorted(
        (item for item in base.glob("*") if item.is_dir()), key=lambda item: item.name, reverse=True
    )
    for candidate in candidates:
        if (candidate / "logs" / "loss_history.jsonl").is_file():
            return candidate
    if candidates:
        return candidates[0]
    raise SystemExit(f"{base} 下没有任何实验目录")


def _read_rows(run_dir: Path) -> list[dict[str, float]]:
    """读标量 jsonl（坏行跳过；不因为一行坏了就整份读不出）。"""
    path = run_dir / "logs" / "loss_history.jsonl"
    if not path.is_file():
        return []
    rows: list[dict[str, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            rows.append(json.loads(text))
        except json.JSONDecodeError:
            continue
    return rows


def _max_steps(run_dir: Path) -> int | None:
    """从 config.yaml 读预算（读不到就是 None，不影响其它检查）。"""
    path = run_dir / "config.yaml"
    if not path.is_file():
        return None
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    try:
        return int(payload["optim"]["max_steps"])
    except (KeyError, TypeError, ValueError):
        return None


def _checkpoint_steps(run_dir: Path) -> list[int]:
    """已有的步级 checkpoint（按步号升序）。"""
    steps: list[int] = []
    for path in (run_dir / "checkpoints").glob("step-*.pt"):
        suffix = path.stem[len("step-") :]
        if suffix.isdigit():
            steps.append(int(suffix))
    return sorted(steps)


def _gpu_line() -> str:
    """一行 GPU 实况；调不到 nvidia-smi 就返回空串（健康检查**不依赖**驱动/GPU）。"""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu,power.draw",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.splitlines()[0] if out else ""


def _trend(rows: list[dict[str, float]], window: int) -> list[str]:
    """步时 / 显存 / 存盘的告警清单（空 = 健康）。"""
    times = [float(row["step_time_s"]) for row in rows if row.get("step_time_s")]
    vram = [float(row["peak_vram_gib"]) for row in rows if row.get("peak_vram_gib")]
    warnings: list[str] = []
    if len(times) >= 4:
        half = len(times) // 2
        early = statistics.mean(times[:half])
        late = statistics.mean(times[half:])
        if early > 0 and late > early * STEP_TIME_DEGRADE_RATIO:
            warnings.append(f"步时退化 {early:.2f}s -> {late:.2f}s（{late / early:.2f}x）")
    if vram and max(vram) >= VRAM_WARN_GIB:
        warnings.append(
            f"显存贴顶 峰值 {max(vram):.2f} GiB >= {VRAM_WARN_GIB} GiB（会滑进共享内存）"
        )
    del window
    return warnings


def report(  # noqa: PLR0912 - 线性报告；拆成多个函数反而更难照着看
    run_dir: Path, *, window: int, gpu: bool
) -> int:
    """打印一次健康检查；返回退出码（0/1/2）。"""
    history = run_dir / "logs" / "loss_history.jsonl"
    rows = _read_rows(run_dir)
    max_steps = _max_steps(run_dir)
    checkpoints = _checkpoint_steps(run_dir)
    age_s = time.time() - history.stat().st_mtime if history.is_file() else float("inf")
    print(f"实验目录：{run_dir}")
    if checkpoints:
        print(f"checkpoint：{checkpoints[-1]}（共 {len(checkpoints)} 个）")
    else:
        print("checkpoint：无")
    last_step = int(rows[-1]["step"]) if rows else (checkpoints[-1] if checkpoints else 0)
    if max_steps is not None:
        percent = 100.0 * last_step / max_steps
        print(f"预算：max_steps={max_steps}，已完成 {last_step}（{percent:.1f}%）")
    if not rows:
        print("标量：没有 loss_history.jsonl（还没跑到第一个 log_every，或从未启动）")
        print("结论：[无数据] 训练未产生任何标量")
        return 2
    recent = rows[-window:]
    losses = [float(row["loss"]) for row in recent]
    times = [float(row["step_time_s"]) for row in recent if row.get("step_time_s")]
    summary = (
        f"最近 {len(recent)} 步：loss {losses[0]:.1f} -> {losses[-1]:.1f}"
        f"（窗口内最优 {min(losses):.1f}）"
    )
    if times:
        summary += f"，步时 中位 {statistics.median(times):.2f}s 最大 {max(times):.2f}s"
    print(summary)
    warnings = _trend(recent, window)
    mean_time = statistics.mean(times) if times else 0.0
    if max_steps is not None and mean_time > 0:
        remaining = max(0, max_steps - last_step)
        eta_hours = remaining * mean_time / 3600.0
        print(f"ETA：还剩 {remaining} 步 x {mean_time:.2f}s ≈ {eta_hours:.2f} 小时")
    if age_s != float("inf"):
        print(f"标量陈旧度：{age_s / 60.0:.1f} 分钟")
        if age_s > STALE_WARN_S:
            warnings.append(f"标量 {age_s / 60.0:.1f} 分钟没有更新：进程可能已停/卡住")
    if checkpoints and last_step > checkpoints[-1]:
        warnings.append(f"最近 checkpoint（{checkpoints[-1]}）落后于当前步（{last_step}）")
    if gpu:
        line = _gpu_line()
        if line:
            print(f"GPU：{line}（memory.used / util / power.draw）")
    if warnings:
        for warning in warnings:
            print(f"告警：{warning}")
        print("结论：[告警] " + "；".join(warnings))
        return 1
    print("结论：[健康] 步时与显存相对上一窗稳定")
    return 0


def main(argv: list[str] | None = None) -> int:
    """入口（--watch N 用于「每 N 秒唤醒一次」的长跑巡检）。"""
    parser = argparse.ArgumentParser(prog="training_health", description="训练健康检查")
    parser.add_argument("--experiment", default="phigros_masked", help="实验名（runs/<名>/…）")
    parser.add_argument("--run-dir", default=None, help="直接指定实验目录")
    parser.add_argument("--window", type=int, default=200, help="统计窗口（最近多少行标量）")
    parser.add_argument("--gpu", action="store_true", help="顺带打印一行 nvidia-smi 实况")
    parser.add_argument(
        "--watch", type=int, default=0, help="每 N 秒重复检查（0 = 只查一次；推荐 7200）"
    )
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    run_dir = _find_run_dir(root, args.experiment, args.run_dir)
    if args.watch <= 0:
        return report(run_dir, window=args.window, gpu=args.gpu)
    while True:
        report(run_dir, window=args.window, gpu=args.gpu)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"（{stamp}；下一次检查在 {args.watch}s 后）", flush=True)
        time.sleep(args.watch)


if __name__ == "__main__":
    sys.exit(main())
