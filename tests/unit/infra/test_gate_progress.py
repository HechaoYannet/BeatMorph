"""门禁必须**能在线看见进度**（默认 CI，无 GPU / 无权重）。

为什么值得一个测试：门禁在正常 K 下的预算是 **~38-45 min**（G2 = 200 步 x 16 段串行前向），
而此前 `execute_gates` 直到全部跑完才写第一行日志 ⇒ 外部**只能靠功耗 / util 猜**它是在跑
还是卡死。2026-09-27 实测踩坑：据此把一个**正常在跑**的门禁误判为卡死、掐掉两次。
因此逐门禁的开始 / 结束（含耗时）回调是**必需的可观测性**，不是锦上添花。
"""

from __future__ import annotations

from beatmorph.infra.config.schema import GatesConfig
from beatmorph.infra.gates import GateInputs, run_gates


def _inputs() -> GateInputs:
    """四道门禁的极简输入（step_fn 是常数，所以整轮是毫秒级）。"""
    return GateInputs(
        step_fn_real=lambda: 1.0,
        step_fn_g2_real=lambda: 1.0,
        step_fn_g2_shuffled=lambda: 2.0,
        model_loss=1.0,
        baseline_loss=10.0,
        frames=75,
        duration_s=1.0,
        frame_rate=75.0,
    )


def test_run_gates_reports_each_gate_start_and_finish() -> None:
    """每道门禁都要上报「开始」与「PASS/FAIL（耗时）」——否则长跑里分不清慢与卡死。"""
    messages: list[str] = []
    results = run_gates(_inputs(), GatesConfig(), progress=messages.append)

    assert len(results) == 4
    assert sum("开始" in message for message in messages) == 4, messages
    assert sum(("PASS" in message) or ("FAIL" in message) for message in messages) == 4, messages
    assert all(
        "s）" in message for message in messages if "PASS" in message or "FAIL" in message
    ), f"完成行必须带耗时：{messages}"
    # 四道门禁的名字都要出现（顺序与 run_gates 一致）
    joined = "\n".join(messages)
    for name in ("G1", "G2", "G3", "G4"):
        assert name in joined


def test_run_gates_is_silent_by_default() -> None:
    """不传 progress 时不产生任何输出（默认路径不被日志污染）。"""
    assert len(run_gates(_inputs(), GatesConfig())) == 4
