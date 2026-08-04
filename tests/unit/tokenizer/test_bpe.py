"""BPETokenizer 单元测试（RFC-0028，CPU 可跑，需 HF tokenizers 在 train extra）。

覆盖（对齐计划 Step 4 验收策略）：
- :meth:`BPETokenizer.train` 在小合成语料上训词表
- ``encode``/``decode`` 往返（``measure_reconstruction_accuracy == 1.0``）
- 确定性：同语料+排序 → 字节一致 vocab.json（sha256）
- vocab_size 合法 + BOS/EOS 哨兵 id 固定
- ``merged_events_per_note`` 监控指标可调用

PoC 词表扫参（真实 4K 语料）见 ``test_bpe_vocab_sweep.py``（标 ``@slow``）。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from beatmorph.core.contracts import BpmPoint, Chart, GameMode, Note, NoteType
from beatmorph.tokenizer import BPETokenizer
from beatmorph.tokenizer.bpe import BOS_ID, EOS_ID

tokenizers = pytest.importorskip("tokenizers")  # 需 train extra


def _make_chart(notes: list[Note], chart_id: int, bpm: float = 120.0, n_bars: int = 4) -> Chart:
    bar_dur = 4 * 60.0 / bpm
    return Chart(
        mode=GameMode.MANIA_4K,
        difficulty=7,
        bpm_points=[BpmPoint(time=0.0, bpm=bpm)],
        notes=notes,
        meta={"audio_duration": n_bars * bar_dur, "beatmap_id": chart_id},
    )


def _small_corpus(n_charts: int = 6) -> list[Chart]:
    """生成小合成语料：每曲 2-3 bar 的 TAP/HOLD pattern，重复以制造 BPE 合并机会。"""
    charts: list[Chart] = []
    for cid in range(n_charts):
        bar_dur = 2.0
        notes: list[Note] = []
        for b in range(3):
            bs = b * bar_dur
            # 稳定 pattern：lane0 downbeat + lane1 1/4 + lane2 1/2 hold
            notes.append(Note(time=bs + 0.0, lane=0, type=NoteType.TAP))
            notes.append(Note(time=bs + 0.5, lane=1, type=NoteType.TAP))
            notes.append(Note(time=bs + 1.0, lane=2, type=NoteType.HOLD, duration=0.3))
        charts.append(_make_chart(notes, chart_id=cid, n_bars=3))
    return charts


def _measure(orig: Chart, recon: Chart, tol_s: float = 0.02) -> float:
    if not orig.notes:
        return 1.0
    o_sorted = orig.sorted_notes()
    r_sorted = recon.sorted_notes()
    used = [False] * len(r_sorted)
    matched = 0
    for o in o_sorted:
        for j, r in enumerate(r_sorted):
            if used[j]:
                continue
            if o.lane == r.lane and abs(o.time - r.time) <= tol_s:
                used[j] = True
                matched += 1
                break
    return matched / len(o_sorted)


class TestTrainEncodeDecode:
    def test_train_and_roundtrip(self, tmp_path: Path) -> None:
        """训词表 → encode/decode 往返 measure == 1.0。"""
        charts = _small_corpus()
        vocab_path = tmp_path / "bpe.json"
        tok = BPETokenizer.train(charts, vocab_path, vocab_size=300)
        assert vocab_path.exists()
        assert tok.vocab_size > 0

        for chart in charts:
            tokens = tok.encode(chart)
            # 首 BOS、尾 EOS
            assert tokens[0].id == BOS_ID
            assert tokens[-1].id == EOS_ID
            # id 全在词表范围
            assert all(0 <= t.id < tok.vocab_size for t in tokens)
            recon = tok.decode(tokens, chart.bpm_points)
            assert _measure(chart, recon) == 1.0, "BPE 往返应无损（POS+NUDGE 保证）"

    def test_determinism(self, tmp_path: Path) -> None:
        """同语料+排序 → 字节一致 vocab.json（sha256）。"""
        charts = _small_corpus()
        p1 = tmp_path / "bpe1.json"
        p2 = tmp_path / "bpe2.json"
        BPETokenizer.train(charts, p1, vocab_size=300)
        BPETokenizer.train(charts, p2, vocab_size=300)
        h1 = hashlib.sha256(p1.read_bytes()).hexdigest()
        h2 = hashlib.sha256(p2.read_bytes()).hexdigest()
        assert h1 == h2, "BPE 训练对已排序语料应确定性（同 tokenizers 版本）"

    def test_empty_chart_encode(self, tmp_path: Path) -> None:
        """空谱面 encode 只产 BOS+EOS。"""
        charts = _small_corpus()
        tok = BPETokenizer.train(charts, tmp_path / "bpe.json", vocab_size=200)
        empty = _make_chart([], chart_id=999, n_bars=2)
        tokens = tok.encode(empty)
        assert [t.id for t in tokens] == [BOS_ID, EOS_ID]
        recon = tok.decode(tokens, empty.bpm_points)
        assert recon.notes == []

    def test_vocab_size_target(self, tmp_path: Path) -> None:
        """vocab_size 上限生效（实际词表 ≤ 目标）。"""
        charts = _small_corpus()
        tok = BPETokenizer.train(charts, tmp_path / "bpe.json", vocab_size=100)
        assert tok.vocab_size <= 100
        # 哨兵至少 4（PAD/BOS/EOS/SEP）+ UNK
        assert tok.vocab_size >= 4


class TestMergedMetric:
    def test_merged_events_per_note_callable(self, tmp_path: Path) -> None:
        """merged_events_per_note 返回 > 0（合成 pattern 有合并空间）。"""
        charts = _small_corpus()
        tok = BPETokenizer.train(charts, tmp_path / "bpe.json", vocab_size=300)
        ratio = tok.merged_events_per_note(charts[0])
        assert ratio > 0.0
        # 业务 token（去 BOS/EOS）数 = len(tokens) - 2
        tokens = tok.encode(charts[0])
        expected = (len(tokens) - 2) / len(charts[0].notes)
        assert ratio == pytest.approx(expected)
