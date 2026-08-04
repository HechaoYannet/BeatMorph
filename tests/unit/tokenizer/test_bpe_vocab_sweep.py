"""BPE 词表扫参 PoC（RFC-0028，``@slow``——需真实 4K mania 语料）。

决策 3 验收门禁：
- 在 ≥1K 真实 diff9-12 谱面上扫 ``BPE_VOCAB_POC_SWEEP = {2048, 4096, 8192}``
- vocab=4096 时 ``avg merged_events_per_note ≤ 1.8``
- top-50 合并人工核为真实手型（jack/jump/hand/stream-step）

top-50 merges dump 到 ``runs/bpe/vocab{N}_top50.txt`` 供人工 review。

CI 默认跳过（``@slow``）：需 ``runs/`` 下有真实语料预处理产物。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from beatmorph.core.contracts import BPE_VOCAB_POC_SWEEP, Chart
from beatmorph.tokenizer import BPETokenizer

tokenizers = pytest.importorskip("tokenizers")

# 验收门禁（决策 3）
_MAX_MERGED_PER_NOTE = 1.8

# 默认真实语料路径（PreprocessPipeline 产物）
_DEFAULT_CORPUS = Path("data/processed/charts.jsonl")


def _load_charts(path: Path, limit: int = 1000) -> list[Chart]:
    """从 charts.jsonl 加载真实 4K 谱面（最多 limit 条）。"""
    charts: list[Chart] = []
    if not path.exists():
        pytest.skip(f"真实语料不存在: {path}（PoC 需先跑 PreprocessPipeline）")
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                charts.append(Chart.model_validate_json(line))
            except ValueError:
                # PoC 语料清洗容忍畸形行（pydantic 校验失败基类）
                continue
            if len(charts) >= limit:
                break
    if len(charts) < 100:
        pytest.skip(f"语料不足 100 条（仅 {len(charts)}），PoC 需 ≥1K")
    return charts


def _dump_top_merges(tok: BPETokenizer, out_path: Path, top_n: int = 50) -> None:
    """把词表按频率（粗略 = id 升序，special 在前）dump top-N 供人工核手型。"""
    # HF 无内置频率；这里 dump 词表前 top_n 个非 special token 名作快速核。
    # 严格频率需语料重扫统计——PoC 阶段人工核为主，频率留后续。
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(min(top_n * 3, tok.vocab_size)):
        # 复合 token 名（如 "POS_0 NOTEV_0_0"）即手型线索
        name = tok._hf.id_to_token(i)
        if name and not name.startswith("["):
            lines.append(f"{i}\t{name}")
        if len(lines) >= top_n:
            break
    out_path.write_text("\n".join(lines), encoding="utf-8")


@pytest.mark.slow
def test_vocab_sweep_merged_ratio_gate(tmp_path: Path) -> None:
    """词表扫参：vocab=4096 时 avg merged_events_per_note ≤ 1.8。"""
    charts = _load_charts(_DEFAULT_CORPUS, limit=1000)
    ratios: dict[int, float] = {}
    for vocab in BPE_VOCAB_POC_SWEEP:
        tok = BPETokenizer.train(charts, tmp_path / f"bpe_{vocab}.json", vocab_size=vocab)
        sample = charts[:200]  # 评估采样 200 曲
        total = sum(tok.merged_events_per_note(c) for c in sample)
        avg = total / len(sample)
        ratios[vocab] = avg
        _dump_top_merges(tok, Path(f"runs/bpe/vocab{vocab}_top50.txt"))
    assert ratios[BPE_VOCAB_POC_SWEEP[1]] <= _MAX_MERGED_PER_NOTE, (
        f"vocab=4096 时 merged_events_per_note={ratios[BPE_VOCAB_POC_SWEEP[1]]:.3f} "
        f"> {_MAX_MERGED_PER_NOTE}；扫参结果: {ratios}。"
        f"若全档超限，考虑升 NUDGE_BUCKETS 或复合 token（残留风险 1/2）。"
    )
