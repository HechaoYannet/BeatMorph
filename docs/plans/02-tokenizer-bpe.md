# Plan 02 — 谱面语义 Tokenizer（BPE/event）

> 状态：🟠 重写中（RFC-0028 采纳，VQ-VAE → BPE/event）｜ 阶段：Phase 1 ｜ 负责：Tokenizer 组
> 对应代码：`beatmorph/tokenizer/events.py`、`beatmorph/tokenizer/bpe.py` ｜ 对应奠基章节：§3.2
> 前身：`02-tokenizer-vqvae.md`（VQ-VAE 范式，随 RFC-0028 退役，VQ 实现移 `archive/vqvae-baseline` 分支）

## 1. 目标与范围

### 交付
- `encode_atomic(chart)->list[str]` / `decode_atomic(events, bpm_points)->Chart`：原子 event 层（`events.py`）。
- `BPETokenizer.train/encode/decode`：HF `tokenizers` BPE 词表训练 + Chart↔BPE id 往返（`bpe.py`）。
- POS+NUDGE **无损**保证：`decode_atomic(encode_atomic(chart), chart.bpm_points)` 在 lane 精确 + |Δt|≤20ms 内重建（测试断言 `measure_reconstruction_accuracy == 1.0`，非 <1.0）。
- PoC 词表扫参 `test_bpe_vocab_sweep.py`（`@slow`）：{2048,4096,8192} 上验收门禁 avg merged-events/note ≤ 1.8 + top-50 合并是真手型。

### 不交付
- AR Transformer 生成主干（plan 04）。
- 解码后的物理后处理与格式 Writer（plan 07）。
- VQ-VAE 路径（RFC-0028 退役，移 `archive/vqvae-baseline` 分支作对照基线）。

### 价值
把谱面原样离散化为 event 序列（Bar/Position/Lane/NoteType/Duration），BPE 合并高频共现组合为复合 token，**不量化掉任何信息**——未合并 note 仍以原子 event 出现。新曲可生成训练集没见过的 event 新组合（生成 vs VQ 的检索重组本质区别）。event LM 范式业界 SOTA（REMI/MIDI-Like/Octuple）。

## 2. 与奠基文档的对应

| 本计划项 | 奠基依据 | 偏离 | 理由 |
|----------|---------|------|------|
| BPE/event tokenizer（REMI 式） | §3.2.1（RFC-0028 重写） | — | RFC-0028 采纳 |
| 词表 ~4096 | §3.2.1 表 | — | `BPE_DEFAULT_VOCAB`，PoC 扫参 |
| 原子 event schema | §3.2.2 | — | mania 4K 裁剪 |
| Position 1/48 拍 + NUDGE 残差 | §3.2.2 | 新增 | 无损保证（VQ bin 量化缺陷不复现） |
| BPM 纯条件（不发 Tempo event） | §3.5/§5 | 决策4 | 守 planner 正交 |
| compute_bar_boundaries 复用 | RFC-0026 | — | phase 对齐脊柱，BPE 原样复用 |

## 3. 接口契约

### 3.1 BPETokenizer（对应 `beatmorph/tokenizer/bpe.py`）
```python
from beatmorph.core.contracts import BPE_DEFAULT_VOCAB, Chart, BpmPoint, EventToken

class BPETokenizer:
    def __init__(self, vocab_path: Path | None = None, vocab_size: int = BPE_DEFAULT_VOCAB) -> None: ...
    @classmethod
    def train(cls, charts: list[Chart], out_path: Path, vocab_size: int = BPE_DEFAULT_VOCAB,
              beats_per_bar: int = 4, chart_id_key: str = "beatmap_id") -> "BPETokenizer": ...
    def encode(self, chart: Chart, beats_per_bar: int = 4) -> list[EventToken]: ...   # 含 BOS…EOS
    def decode(self, tokens: list[EventToken], chart_bpm_points: list[BpmPoint],
               beats_per_bar: int = 4) -> Chart: ...  # ⚠️ 取 bpm_points 入参（决策4）
    @property
    def vocab_size(self) -> int: ...
    def merged_events_per_note(self, chart: Chart, beats_per_bar: int = 4) -> float: ...  # PoC 监控
```
`EventToken(id, bar_index, start_time)` 见 `core/contracts/events.py`；`id ∈ [0, vocab_size)`。

### 3.2 原子 event schema（`events.py`，mania 4K REMI 变体）
| Event | 取值 | 说明 |
|------|------|------|
| 哨兵 | PAD/BOS/EOS/SEP | 仅词表登记，不出现在 atomic 流 |
| BAR | 每 bar 一个 | 边界由 `compute_bar_boundaries`（RFC-0026 phase） |
| POS_i | i ∈ [0, beats_per_bar×48) | 1/48 拍子拍格（4/4→192 格） |
| NUDGE_j | j ∈ [0, 12) | 残差毫秒桶，仅 off-beat note 发 |
| NOTEV_<lane>_<type> | 4 lane × 5 NoteType | 20 种复合原子类型 |
| DUR_k | k ∈ [0, 24) | HOLD/ROLL 长度 log-spaced 桶 |

### 3.3 跨模块张量形状
| 名称 | 形状 | 含义 |
|------|------|------|
| `EventSeq` | `(batch, seq_len)` long | BPE event id ∈ [0, vocab_size)，契约 `EventSeq` |
| 词表 | `vocab.json` | HF tokenizers 产物 |
| 原子流 | `(n_events,)` str | `encode_atomic` 产物 |

## 4. 内部设计

- **原子编码** `encode_atomic`：`chart.sorted_notes()` + `compute_bar_boundaries` 推 phase 对齐小节边界 → 逐 bar 发 BAR → 归属 note 发 POS/[NUDGE]/NOTEV/[DUR]（HOLD/ROLL 才发 DUR，TAP/MINE/FAKE duration=0 跳）。空 bar 仍发 BAR。
- **POS+NUDGE 无损数学**：`POS_i = round(note_rel/pos_step) % pos_count`，残差 `r = note_rel - i·pos_step`。NUDGE 用 12 桶覆盖 `±pos_step/2`。实测最坏解码误差 ~0.86ms@120bpm（≤ bucket_w），≪ ±20ms 容差。
- **BPE 训练** `train`：按 chart_id 升序排序（决定性）→ 每 chart encode_atomic → `" ".join` → corpus → HF `Tokenizer(BPE)+Whitespace+BpeTrainer(vocab_size, special_tokens)`。
- **encode/decode 往返**：HF `word_ids` 把每个 BPE id 映射回源原子词索引 → bar 锚点。decode 取 `chart_bpm_points` 入参做 POS→秒（变速分段）。
- **不含**（v1）：Tempo event（BPM 纯条件，守 planner 正交）、手编 Hand enum（让 BPE 从共现 NOTEV 自动发现手型）。

## 5. 依赖关系

- **上游**：数据预处理流水线（plan 08）提供 `Chart`；`compute_bar_boundaries`（osu_path.py，RFC-0026 phase 脊柱）。
- **下游**：AR 生成（plan 04）消费 `EventSeq`；解码后处理（plan 07）调 `decode`。
- **外部库**：`tokenizers>=0.20`（`train` extra，HF Rust BPE）、`pydantic>2.5`（契约）。

## 6. 里程碑与验收标准

对齐奠基 §7 Phase 1（RFC-0028 后口径：M2 重建口径反思，重 M4 生成盲测）。

| 里程碑 | 验收（可量化） |
|--------|---------------|
| M1 原子往返 | `decode_atomic(encode_atomic(chart), bpm_points)` 无损，measure == 1.0 |
| M2 BPE 往返 | `BPETokenizer` 训词表后 encode/decode 往返 == 1.0；确定性 sha256 |
| M3 词表健康 | PoC 扫参 vocab=4096 时 avg merged-events/note ≤ 1.8 + top-50 真手型 |
| M4 50K 训练跑通 | AR 在 50K 谱面上 event 序列 CE 收敛（plan 04）；M4 生成盲测为主验收 |

## 7. 风险与缓解

| 风险 | 缓解 |
|------|------|
| NUDGE_BUCKETS=12 对真实 off-beat 负载不足 | PoC ≤1.8 门禁兜底；不达标升 16（误差仍<20ms） |
| BPE 合并价值低（mania 原子词表小） | top-50 真手型核；全档超限考虑复合 token（CPWord 式） |
| 变速曲 POS→秒分段映射错 | `compute_bar_boundaries` 已分段；测试覆盖 2-BpmPoint 往返 |

## 8. 测试策略

- **单元**：`tests/unit/tokenizer/test_events.py`——原子往返==1.0、确定性、off-beat NUDGE、变速、phase 0.339、POS/NUDGE/DUR 数学。
- **单元**：`tests/unit/tokenizer/test_bpe.py`——BPE train/encode/decode 往返==1.0、确定性 sha256、vocab_size、哨兵。
- **集成**：`tests/integration/test_chart_token_roundtrip.py::test_bpe_chart_roundtrip`——真实 .osu → BPE 往返 ==1.0。
- **e2e**：`tests/unit/tokenizer/test_bpe_vocab_sweep.py`（`@slow`）——真实 4K 语料 PoC 扫参门禁。

## 9. 开放问题

- [ ] RFC-0004：训练数据起步量 50K（§3.2.1）vs 100 万+（§4.1）——以 §7 Phase1 为先。
- [ ] NUDGE_BUCKETS 终值：默认 12，PoC 数据决定是否升 16。
- [ ] 词表终值：默认 4096，PoC 扫参结果决定。
