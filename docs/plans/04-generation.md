# Plan 04 — Stage 2 AR Pattern 序列生成主干

> 状态：🟡 草案 ｜ 阶段：Phase 2（核心生成突破）｜ 负责：生成组
> 对应代码：`beatmorph/generation/ar_transformer.py` ｜ 对应奠基章节：§3.4

## 1. 目标与范围

### 交付
- AR Transformer Decoder：以 audio_emb + Stage1 规划 + RAG 前缀 + 难度/风格为条件，自回归生成 Pattern Token 序列。
- Teacher Forcing 训练循环（Cross-Entropy）。
- 推理接口（KV-cache、top-k/top-p 采样、温度控制）。
- 备选 Flow Matching 的接口预留与蒸馏路径设计（本阶段仅预留，Phase 3 落地）。

### 不交付
- BPE/event tokenizer 本体（Plan 02）、Stage 1 规划器（Plan 03）、RAG 检索（Plan 05）、解码回 Note（Plan 07）。
- DPO 微调（Plan 06，基于本模块产出的预训练基座）。

### 价值
本模块是系统「最核心的生成引擎」（奠基 §3.4）。单阶段 event 序列自回归（BPE/event tokenizer，业界 SOTA），业界论证最充分、工具链最成熟（HF Transformers / Megatron-LM）。

## 2. 与奠基文档的对应

| 决策 | 奠基依据 | 本计划 |
|------|---------|--------|
| AR Transformer 主选 | §3.4.1 / §8 | 完全采纳 |
| 12-16 层、16 头 | §3.4.1 表 | 默认 12 层，可在 config 切 16 |
| ~1024 event tokens（段落级，分段生成 RFC-0008） | §3.4.1 / §3.4.2 | 采纳，常量 `AR_CONTEXT_TOKENS=1024` |
| Teacher Forcing + CE | §3.4.1 | 采纳 |
| Cross-Attention(audio) + AdaLN(diff/style) | §3.4.1 | 采纳 |
| Flow Matching 备选，未来蒸馏加速 | §3.4.2 / §8 | 本阶段仅预留接口与蒸馏设计 |

**偏离**：奠基 §3.4.1 未指定隐层维度。本计划规定 `dim=768` 与 MERT 输出对齐，免去投影层、稳定 cross-attention。记入 RFC-0007。

## 3. 接口契约

对应骨架 `ARTransformer`：

```python
class ARTransformer:
    def __init__(self, n_layers=12, n_heads=16, context_tokens=1024, vocab_size=4096): ...
    def generate(
        self,
        audio_emb: Tensor,          # (batch, time_seq, 768)  来自 Stage0
        sections: list[Section],    # 来自 Stage1
        difficulty: int,            # 1-15
        bpm_points: list[BpmPoint], # 解码期 POS→秒映射用（AR 不生成 tempo，见 §9）
        rag_prefix: Tensor | None,  # (batch, top_k, ref_seq_len) 来自 RAG，可选
        max_tokens: int = 1024,     # event 数，分段生成
    ) -> list[EventToken]: ...
```

- 输入张量引用 `AudioEmbedding`、`RAGContext` 形状契约（见 `core/contracts/tensors.py`）。
- 输出 `list[EventToken]`（见 `core/contracts/events.py`），`id ∈ [0, vocab_size)`。
- 条件嵌入：`difficulty` 经 Embedding(15) → AdaLN；`sections` 编码为局部条件向量。

## 4. 内部设计

- **主干**：标准 Pre-LN Decoder，因果掩码。Token embedding = 词表查表（BPE vocab）。
- **audio 条件**：Cross-Attention，audio_emb 作 K/V，每层注入。
- **AdaLN**：difficulty/style 经 SiLU+Linear 调制 LayerNorm 的 scale/shift。
- **RAG 注入**：参考 Token 序列拼接到序列前缀（prefix-tuning 风格），因果掩码允许 attend。
- **训练**：Teacher Forcing，shift 右移标签，CE Loss。配合 `torch.compile`。
- **推理**：KV-cache 逐 token；支持 greedy / top-k / top-p / temperature；`max_tokens≤1024`（event 数，分段）。
- **Flow Matching 预留**：抽象 `Backbone` 接口，AR 与 FM 共享 cross-attn/AdaLN 组件，蒸馏时复用。

## 5. 依赖关系

- **上游**：Stage0 `audio_emb`、Stage1 `sections`、BPE 词表（`vocab_size`，Plan 02 训练产物）、RAG `rag_prefix`（可选）。
- **下游**：Plan 07 解码器（消费 Token 序列）、Plan 06 DPO（以此为基座）。
- **外部库**：`torch>=2.5`、`einops`、`pytorch-lightning`（训练）、`wandb`。

## 6. 里程碑与验收标准

| 里程碑 | 验收（对应奠基 §7 Phase 2） |
|--------|------|
| M1 单步前向跑通 | 给定定长 audio_emb，输出合法 `EventToken` 序列，shape 正确 |
| M2 50K 谱面训练收敛 | CE Loss 收敛；event 序列 CE 收敛（BPE 无坍缩风险，R-2 已随 VQ 退役） |
| M3 生成可解码谱面 | 输出 Token 经 Plan 07 解码为合法 Note，首版可玩 `.osu` |
| M4 内部盲测 | 对比基线（随机/规则）人工盲测胜率 >70% |

## 7. 风险与缓解

| 风险 | 奠基编号 | 缓解 |
|------|---------|------|
| AR 推理慢（10-30s/首） | R-4 | Phase 3 用 Flow Matching 蒸馏至 4-8 步；缓存常用曲风 prefix |
| 谱面坏习惯 | R-3 | Plan 06 DPO 兜底；预处理过滤低评分（Plan 08） |
| 上下文不足覆盖长曲 | — | ~1024 event 覆盖一段落；超长曲滑窗分段生成 + 衔接（RFC-0008，升为分段生成主路径常态） |

## 8. 测试策略

- **单元**：因果掩码正确性（未来 token 不泄露）、AdaLN shape、cross-attn 对齐。
- **集成**：mock Stage0/1/RAG 输出，端到端 `generate` 返回合法 Token。
- **e2e**：真实 audio_emb → Token → Plan 07 → `.osu`，标记 `@pytest.mark.e2e`、`@pytest.mark.gpu`。

## 9. 开放问题

- [ ] RFC-0007：隐层 dim=768 是否需为容量上探到 1024。
- [ ] RFC-0008：超长曲目（>5 分钟）滑窗生成的段落衔接策略（现已升为分段生成主路径常态）。
- [ ] RFC-0009：Flow Matching 蒸馏的具体对齐目标（logit 蒸馏 vs 轨迹蒸馏）。
- [ ] RFC-0028 派生：`generate(..., bpm_points)` 契约占位——AR 不生成 tempo，由解码器据 `bpm_points` 把 `POS` event 反映射为秒。未来是否上抬到 Plan 03 planner 让其规划 BPM 轨迹、AR 仅消费，待 RFC 定稿。
