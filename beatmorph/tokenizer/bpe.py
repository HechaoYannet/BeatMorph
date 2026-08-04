"""BPE event tokenizer（RFC-0028）。

封装 HF `tokenizers` BPE：``BPETokenizer.train`` 在谱面语料上训词表，
``encode`` / ``decode`` 做 Chart ↔ BPE event id 往返。

不继承 ``nn.Module``：BPE 是确定性统计，无可训参数；HF tokenizers（Rust）词表
训练 CPU 数小时完成。vocab JSON 落盘到
``${BEATMORPH_RUNS_DIR}/bpe/{vocab_size}/``，运行期 load 一次。

依赖 :mod:`beatmorph.tokenizer.events` 的原子层（``encode_atomic`` /
``decode_atomic``）——本模块只负责「原子 event-name 序列 ↔ BPE 复合 id」的
合并/拆分。原子 schema、POS+NUDGE 无损数学详见 :mod:`beatmorph.tokenizer.events`。

详细计划：docs/plans/02-tokenizer-vqvae.md（RFC-0028 后改写为 BPE）。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from beatmorph.core.contracts import (
    BPE_DEFAULT_VOCAB,
    BpmPoint,
    Chart,
    EventToken,
)
from beatmorph.core.logging import get_logger
from beatmorph.tokenizer.events import (
    _bar_name,
    decode_atomic,
    encode_atomic,
)

if TYPE_CHECKING:
    from tokenizers import Tokenizer

logger = get_logger(__name__)

# ── 哨兵 token 名（BPE 词表登记，不出现在 atomic 流里）──
PAD_TOKEN = "[PAD]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
SEP_TOKEN = "[SEP]"
SPECIAL_TOKENS: tuple[str, ...] = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, SEP_TOKEN)

_DEFAULT_BEATS_PER_BAR = 4


class BPETokenizer:
    """BPE event tokenizer（无状态确定性的统计 tokenizer，RFC-0028）。

    词表由 :meth:`train` 离线训练后落盘；:meth:`encode` / :meth:`decode`
    加载已训词表做 Chart↔BPE id 往返。

    Args:
        vocab_path: 词表 JSON 路径（``train()`` 产物；加载时必填，为 ``None``
            时仅可调 :meth:`train`）。
        vocab_size: 词表大小（仅 :meth:`train`` 用，默认 :data:`BPE_DEFAULT_VOCAB`）。
    """

    def __init__(
        self,
        vocab_path: Path | None = None,
        vocab_size: int = BPE_DEFAULT_VOCAB,
    ) -> None:
        self._vocab_path = vocab_path
        self._vocab_size = vocab_size
        self._hf: Tokenizer | None = None
        if vocab_path is not None:
            self._load(vocab_path)

    # ── 词表训练 ──────────────────────────────────────────────

    @classmethod
    def train(
        cls,
        charts: list[Chart],
        out_path: Path,
        vocab_size: int = BPE_DEFAULT_VOCAB,
        beats_per_bar: int = _DEFAULT_BEATS_PER_BAR,
        chart_id_key: str = "beatmap_id",
    ) -> BPETokenizer:
        """BPE 词表训练（确定性）。

        流程：
            1. 按 ``chart_id``（chart.meta 中的 key）升序排序 charts——决定性
               要求 + equal-frequency tie-break（同 key 时按 note 数稳定排序）。
            2. 每 chart → :func:`encode_atomic` → ``" ".join(events)`` → corpus 文本行。
            3. HF ``Tokenizer(models.BPE)`` + ``pre_tokenizer=Whitespace()`` +
               ``trainer=BpeTrainer(vocab_size, special_tokens)``。
            4. 落盘 ``out_path``（tokenizer.json）；返回 load 后的实例。

        确定性：HF BPE 对已排序语料 + 固定 special_tokens 顺序产出字节一致 vocab
        （同 tokenizers 版本下）；测试用 sha256 校验。

        Args:
            charts: 训练语料谱面列表。
            out_path: 词表输出路径（``.json`` 或 ``.json`` 目录名）。
            vocab_size: 目标词表大小（默认 :data:`BPE_DEFAULT_VOCAB`）。
            beats_per_bar: 拍号（传给 :func:`encode_atomic`）。
            chart_id_key: chart.meta 中用作排序 key 的字段名。
        Returns:
            训练后加载好的 :class:`BPETokenizer`。
        """
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.trainers import BpeTrainer

        if not charts:
            raise ValueError("训练语料为空")

        # 决定性排序：chart_id 升序 + note 数 tie-break
        def _sort_key(c: Chart) -> tuple[str, int]:
            cid = c.meta.get(chart_id_key)
            return (str(cid) if cid is not None else "", len(c.notes))

        ordered = sorted(charts, key=_sort_key)
        corpus = (" ".join(encode_atomic(c, beats_per_bar)) for c in ordered)

        hf_tok = Tokenizer(BPE(unk_token="[UNK]"))
        hf_tok.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=list(SPECIAL_TOKENS),
            unk_token="[UNK]",
        )
        hf_tok.train_from_iterator(corpus, trainer)

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        hf_tok.save(str(out_path))
        logger.info(
            "BPETokenizer.train: vocab_size=%d (target %d), %d charts, saved %s",
            hf_tok.get_vocab_size(),
            vocab_size,
            len(charts),
            out_path,
        )
        return cls(vocab_path=out_path, vocab_size=vocab_size)

    # ── 加载 ──────────────────────────────────────────────────

    def _load(self, vocab_path: Path) -> None:
        from tokenizers import Tokenizer

        self._hf = Tokenizer.from_file(str(vocab_path))
        assert self._hf is not None  # 刚赋值
        self._vocab_size = self._hf.get_vocab_size()

    @property
    def vocab_size(self) -> int:
        """实际词表大小（加载后由 HF tokenizer 报告）。"""
        return self._vocab_size

    # ── Chart ↔ BPE id 往返 ───────────────────────────────────

    def encode(
        self,
        chart: Chart,
        beats_per_bar: int = _DEFAULT_BEATS_PER_BAR,
    ) -> list[EventToken]:
        """Chart → EventToken 列表（含 BOS … EOS）。

        原子 event 流经 HF BPE 编码为合并 id；每个 EventToken 带 ``bar_index`` /
        ``start_time`` 锚点。BPE 合并跨多个原子词时（如 ``POS_0 NOTEV_0_0`` →
        单复合 token），HF ``word_ids`` 把每个输出 id 映射回其首个源原子词索引，
        锚点取该原子词所属 BAR 的坐标。

        Args:
            chart: 谱面 IR。
            beats_per_bar: 拍号。
        Returns:
            EventToken 列表，首 ``[BOS]`` 尾 ``[EOS]``。
        """
        if self._hf is None:
            raise RuntimeError("BPETokenizer 未加载词表（vocab_path=None）")
        atomic = encode_atomic(chart, beats_per_bar)
        text = " ".join(atomic)
        enc = self._hf.encode(text, add_special_tokens=False)
        # 每个 BPE id → 其源原子词索引（HF word_ids，None=special）
        word_ids = enc.word_ids
        anchors = _compute_token_anchors(chart, beats_per_bar)
        tokens = [EventToken(id=int(BOS_ID), bar_index=0, start_time=0.0)]
        for k, wid in enumerate(word_ids):
            if wid is None:
                continue
            idx = min(int(wid), len(anchors) - 1) if anchors else 0
            bar_idx, bar_start = anchors[idx] if anchors else (0, 0.0)
            tokens.append(EventToken(id=int(enc.ids[k]), bar_index=bar_idx, start_time=bar_start))
        tokens.append(EventToken(id=int(EOS_ID), bar_index=0, start_time=0.0))
        return tokens

    def decode(
        self,
        tokens: list[EventToken],
        chart_bpm_points: list[BpmPoint],
        beats_per_bar: int = _DEFAULT_BEATS_PER_BAR,
    ) -> Chart:
        """EventToken 列表 → Chart。

        BPE id → HF decode → 原子 event-name 字符串流 →
        :func:`decode_atomic`（``chart_bpm_points`` 做 POS→秒）。

        Args:
            tokens: EventToken 列表（:meth:`encode` 产物，含 BOS/EOS）。
            chart_bpm_points: 目标谱面 BPM 变速点（决策 4：AR 不生成 tempo，
                decode 用谱面 bpm_points）。
            beats_per_bar: 拍号。
        Returns:
            Chart（``bpm_points=chart_bpm_points``，note 按 time 升序）。
        """
        if self._hf is None:
            raise RuntimeError("BPETokenizer 未加载词表")
        ids = [t.id for t in tokens]
        text = self._hf.decode(ids, skip_special_tokens=True)
        events = text.split()
        return decode_atomic(events, chart_bpm_points, beats_per_bar)

    # ── PoC 监控 ──────────────────────────────────────────────

    def merged_events_per_note(
        self,
        chart: Chart,
        beats_per_bar: int = _DEFAULT_BEATS_PER_BAR,
    ) -> float:
        """PoC sweep 监控指标：BPE 合并后 token 数 / note 数。

        决策 3 验收门禁：vocab=4096 时均值 ≤ 1.8。值接近 3 说明 BPE 合并价值低
        （≈ 原子流，每 note ~3 event），需扫参或考虑复合 token。

        Args:
            chart: 谱面。
            beats_per_bar: 拍号。
        Returns:
            合并后 token 数 / note 数；空谱面返回 0.0。
        """
        if not chart.notes:
            return 0.0
        toks = self.encode(chart, beats_per_bar)
        # 去掉 BOS/EOS 计业务 token
        n = len(toks) - sum(1 for t in toks if t.id in (BOS_ID, EOS_ID))
        return n / len(chart.notes)


# ── 哨兵 id 常量（train 后由词表固定）────────────────────────
# special_tokens 顺序 = (PAD, BOS, EOS, SEP) + UNK → BOS=1, EOS=2 固定
BOS_ID = 1
EOS_ID = 2


# ── 辅助：推每个原子 event 词所属 BAR 的 (bar_index, start_time) ──


def _compute_token_anchors(
    chart: Chart,
    beats_per_bar: int,
) -> list[tuple[int, float]]:
    """逐原子 event 词返回 (bar_index, start_time) 锚点。

    遇 ``BAR`` 词：锚点 = 该 BAR 的 (index, bar_start)；进入下一 bar 后，
    后续 ``POS``/``NOTEV``/``DUR`` 继承当前 bar 锚点。锚点长度 = atomic 流词数，
    与 :meth:`encode` 的 ``word_ids`` 一一对齐。

    Args:
        chart: 谱面（用其 bpm_points + total_duration 推 bar 边界）。
        beats_per_bar: 拍号。
    Returns:
        每原子词的 (bar_index, start_time)；空谱面返回 ``[]``。
    """
    from beatmorph.data.parsers.osu_path import compute_bar_boundaries
    from beatmorph.tokenizer.events import encode_atomic

    atomic = encode_atomic(chart, beats_per_bar)
    if not atomic:
        return []

    sorted_notes = chart.sorted_notes()
    note_dur = (
        sorted_notes[-1].time + max(n.duration for n in sorted_notes) if sorted_notes else 0.0
    )
    audio_dur = chart.meta.get("audio_duration")
    audio_total = (
        float(audio_dur) if isinstance(audio_dur, int | float) and audio_dur > 0 else note_dur
    )
    boundaries = compute_bar_boundaries(chart.bpm_points, max(audio_total, note_dur))

    anchors: list[tuple[int, float]] = []
    bar_idx = 0
    bar_start = float(boundaries[0]) if boundaries else 0.0
    for ev in atomic:
        if ev == _bar_name():
            if bar_idx < len(boundaries):
                bar_start = float(boundaries[bar_idx])
            anchors.append((bar_idx, bar_start))
            bar_idx += 1
        else:
            # 非BAR词继承当前 bar 锚点（bar_idx 已 +1，指下一 bar；当前 bar = bar_idx-1）
            cur = max(0, bar_idx - 1)
            cur_start = float(boundaries[cur]) if cur < len(boundaries) else bar_start
            anchors.append((cur, cur_start))
    return anchors
