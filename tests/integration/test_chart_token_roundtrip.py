"""Chart ↔ Token 往返集成测试（Plan 02 M1）。

真实 ``.osu`` → ``Chart``（parse_osu）→ ``encode`` → ``decode`` → 结构往返断言。
提供可复用 :func:`measure_reconstruction_accuracy` 工具（Note time±20ms 且 lane 完全
匹配比例），供 M2 训练后 ``@slow @gpu`` 测试用。

未训练小模型不达 95% 重建，集成测试只验路径打通（同
``tests/integration/test_audio_to_plan.py`` 的 CPU/gpu 分离模式）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from beatmorph.core.contracts import BpmPoint, Chart, GameMode, Note, NoteType
from beatmorph.data.parsers.osu_path import parse_osu
from beatmorph.tokenizer.vqvae import VQVAETokenizer

# CPU 小模型（同 unit/tokenizer/test_vqvae.py 风格）
_LATENT = 8
_CODEBOOK = 16
_TIME_BINS = 16


def _small_tok() -> VQVAETokenizer:
    return VQVAETokenizer(
        codebook_size=_CODEBOOK,
        latent=_LATENT,
        lane=4,
        time_bins=_TIME_BINS,
        feat=6,
        cnn_channels=(8, 8),
        transformer_layers=1,
        transformer_heads=2,
        dead_code_steps=3,
    )


def measure_reconstruction_accuracy(
    orig: Chart,
    recon: Chart,
    tol_s: float = 0.02,
) -> float:
    """重建准确率：对 orig 每 Note 找 recon 中（lane 同、|时间差|<tol_s）的，贪心匹配比例。

    奠基 §7 Phase1 / Plan 02 M2 验收口径：Note time±20ms 且 lane 完全匹配。

    Args:
        orig: 原始 Chart。
        recon: 重建 Chart。
        tol_s: 时间容差秒（默认 0.02=20ms）。
    Returns:
        匹配比例 [0,1]；orig 无 Note 时返回 1.0。
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


@pytest.mark.integration
def test_chart_token_roundtrip(fixtures_dir: Path) -> None:
    """真实 .osu → Chart → encode → decode 结构往返。"""
    osu_path = fixtures_dir / "sample_4k_mania.osu"
    if not osu_path.exists():
        pytest.skip(f"fixture 不存在: {osu_path}")

    chart = parse_osu(osu_path)
    assert chart.notes, "夹具谱面无 Note"
    assert chart.mode == GameMode.MANIA_4K

    tok = _small_tok()
    tokens = tok.encode(chart)
    # 至少能编码出非空 token 序列
    assert len(tokens) > 0
    assert all(0 <= t.code < _CODEBOOK for t in tokens)
    # 按 bar_index 升序
    assert [t.bar_index for t in tokens] == list(range(len(tokens)))

    out = tok.decode(tokens)
    assert isinstance(out, Chart)
    assert out.mode == GameMode.MANIA_4K
    assert out.bpm_points
    assert out.bpm_points[0].bpm > 0

    # 重建 Note 合法性
    total = chart.notes[-1].time + 5.0
    for n in out.notes:
        assert 0.0 <= n.time <= total
        assert 0 <= n.lane < 4
        assert n.type in NoteType
        assert n.duration >= 0.0

    # 未训练小模型：准确率可能低，但 measure 工具应可调用且返回合法 [0,1]
    acc = measure_reconstruction_accuracy(chart, out, tol_s=0.02)
    assert 0.0 <= acc <= 1.0


@pytest.mark.integration
def test_chart_token_roundtrip_minimal(fixtures_dir: Path) -> None:
    """最小夹具谱面（sample_minimal_mania.osu）往返。"""
    osu_path = fixtures_dir / "sample_minimal_mania.osu"
    if not osu_path.exists():
        pytest.skip(f"fixture 不存在: {osu_path}")

    chart = parse_osu(osu_path)
    tok = _small_tok()
    tokens = tok.encode(chart)
    out = tok.decode(tokens)
    assert out.mode == GameMode.MANIA_4K
    assert out.bpm_points[0].bpm > 0
    # 通路无异常即可（未训练不要求 Note 数匹配）


def test_measure_reconstruction_accuracy_perfect_match() -> None:
    """检证 measure 工具：完美匹配返回 1.0。"""
    notes = [Note(time=0.0, lane=0), Note(time=0.5, lane=1)]
    orig = Chart(
        mode=GameMode.MANIA_4K,
        difficulty=1,
        bpm_points=[BpmPoint(time=0.0, bpm=120.0)],
        notes=notes,
    )
    # 完全相同的重建
    recon = Chart(
        mode=GameMode.MANIA_4K,
        difficulty=1,
        bpm_points=[BpmPoint(time=0.0, bpm=120.0)],
        notes=[Note(time=0.0, lane=0), Note(time=0.5, lane=1)],
    )
    assert measure_reconstruction_accuracy(orig, recon, tol_s=0.02) == 1.0


def test_measure_reconstruction_accuracy_partial() -> None:
    """检证 measure 工具：部分匹配 + 容差边界。"""
    orig = Chart(
        mode=GameMode.MANIA_4K,
        difficulty=1,
        bpm_points=[BpmPoint(time=0.0, bpm=120.0)],
        notes=[Note(time=0.0, lane=0), Note(time=1.0, lane=1)],
    )
    # lane 错配 + 容差点上匹配
    recon = Chart(
        mode=GameMode.MANIA_4K,
        difficulty=1,
        bpm_points=[BpmPoint(time=0.0, bpm=120.0)],
        notes=[Note(time=0.01, lane=0), Note(time=2.0, lane=2)],  # 第1个 within 20ms，第2个错
    )
    acc = measure_reconstruction_accuracy(orig, recon, tol_s=0.02)
    assert acc == pytest.approx(0.5)
