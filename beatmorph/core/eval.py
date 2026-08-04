"""范式中立的评估工具（RFC-0028 从测试集成层提到此处复用）。

:func:`measure_reconstruction_accuracy` 是贪心 note 匹配度量（lane 精确 +
|Δt|≤容差），用于 tokenizer 往返验收（VQ-VAE 重建、BPE/event 往返）与未来
AR 生成评估。范式中立：只吃 :class:`Chart`，不依赖任何 tokenizer 实现。
"""

from __future__ import annotations

from beatmorph.core.contracts import Chart


def measure_reconstruction_accuracy(
    orig: Chart,
    recon: Chart,
    tol_s: float = 0.02,
) -> float:
    """重建准确率：对 ``orig`` 每 Note 找 ``recon`` 中（lane 同、|时间差|<tol_s）的贪心匹配比例。

    奠基 §7 Phase1 / Plan 02 M2 验收口径：Note time±20ms 且 lane 完全匹配。
    RFC-0028 后 BPE/event tokenizer 因 POS+NUDGE 设计无损，往返应 == 1.0
    （<1.0 即量化 bug）。

    Args:
        orig: 原始 Chart。
        recon: 重建 Chart。
        tol_s: 时间容差秒（默认 0.02=20ms）。
    Returns:
        匹配比例 ``[0,1]``；orig 无 Note 时返回 1.0。
    """
    if not orig.notes:
        return 1.0
    orig_sorted = orig.sorted_notes()
    recon_sorted = recon.sorted_notes()
    used: list[bool] = [False] * len(recon_sorted)
    matched = 0
    for o in orig_sorted:
        for j, r in enumerate(recon_sorted):
            if used[j]:
                continue
            if o.lane == r.lane and abs(o.time - r.time) <= tol_s:
                used[j] = True
                matched += 1
                break
    return matched / len(orig_sorted)
