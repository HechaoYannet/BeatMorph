"""Chart ↔ Token 往返集成测试（Plan 02 M1，RFC-0028 后含 VQ + BPE 双往返）。

真实 ``.osu`` → ``Chart``（parse_osu）→ ``encode`` → ``decode`` → 结构往返断言。
``measure_reconstruction_accuracy`` 工具已提到 :mod:`beatmorph.core.eval`
（RFC-0028），本文件保留 VQ 往返测（随 vqvae.py 留在 baseline 分支）+ 新增
BPE/event 往返测（主路径，应 == 1.0 无损）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from beatmorph.core.contracts import BpmPoint, Chart, GameMode, Note, NoteType
from beatmorph.core.eval import measure_reconstruction_accuracy
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


@pytest.mark.integration
def test_bpe_chart_roundtrip(fixtures_dir: Path, tmp_path: Path) -> None:
    """BPE/event tokenizer 真实 .osu → Chart → encode → decode 往返（RFC-0028）。

    POS+NUDGE 设计无损，往返 measure == 1.0（lane 精确 + |Δt|≤20ms）。
    """
    pytest.importorskip("tokenizers")  # 需 train extra
    osu_path = fixtures_dir / "sample_4k_mania.osu"
    if not osu_path.exists():
        pytest.skip(f"fixture 不存在: {osu_path}")

    chart = parse_osu(osu_path)
    assert chart.notes, "夹具谱面无 Note"

    # 单曲训小词表（够覆盖该曲原子 event 即可）
    from beatmorph.tokenizer import BPETokenizer

    vocab_path = tmp_path / "bpe.json"
    tok = BPETokenizer.train([chart], vocab_path, vocab_size=800, chart_id_key="beatmap_id")

    tokens = tok.encode(chart)
    assert tokens[0].id == 1  # BOS
    assert tokens[-1].id == 2  # EOS
    assert all(0 <= t.id < tok.vocab_size for t in tokens)

    out = tok.decode(tokens, chart.bpm_points)
    assert isinstance(out, Chart)
    assert out.mode == GameMode.MANIA_4K
    # POS+NUDGE 无损：往返 == 1.0
    acc = measure_reconstruction_accuracy(chart, out, tol_s=0.02)
    assert acc == 1.0, f"BPE 往返应无损，实际 {acc}"
