# Plan 05 — RAG 风格检索增强

> 状态：🟡 草案 ｜ 阶段：Phase 2（核心生成突破）｜ 负责：检索组
> 对应代码：`beatmorph/rag/retriever.py` ｜ 对应奠基章节：§3.5

## 1. 目标与范围

### 交付
- `RAGRetriever`：对百万级 osu! 谱面按 BPM/流派/谱师/难度建立 FAISS 索引，Top-K=3 检索。
- 联合检索特征：音频 MERT Embedding（pooling 向量）+ 谱面密度曲线（重采样定长），拼为单一查询/库向量。
- 检索结果组装为 `RAGContext`，以 Prefix（默认）或 Cross-Attention KV 两种方式交付 `ARTransformer`。
- 离线索引构建脚本与索引持久化（`build_index(corpus_dir)`）。

### 不交付
- 训练任何参数（奠基 §3.5.2：RAG 无需训练）。
- 检索特征提取本体（音频 emb 由 Plan 01 MERT 产出；密度曲线由 Plan 03 规划器/统计量产出）。
- AR 解码器本体（Plan 04 消费 `RAGContext`）。
- DPO 偏好对齐（Plan 06）。

### 价值
以零训练、强可解释方式实现「参考风格迁移」：用户可见模型引用了哪首谱面（奠基 §3.5.2）。在符号音乐/音乐生成领域，RAG 已被证明在客观指标与主观听感上优于 LoRA 微调，是「自监督理解 + 检索增强 + 偏好对齐」技术基座中最低成本的风格控制手段。

## 2. 与奠基文档的对应

| 决策 | 奠基依据 | 本计划 |
|------|---------|--------|
| RAG 检索增强，零训练 | §3.5.1 / §8 | 完全采纳 |
| 检索库按 BPM/流派/谱师/难度索引 | §3.5.1 表 | 采纳，FAISS 主索引 + 元数据二次过滤 |
| 检索特征 = 音频 MERT Emb + 谱面密度曲线 | §3.5.1 表 | 采纳，联合向量见 §4 |
| Top-K = 3 | §3.5.1 表 | 采纳，常量 `RAGContext.top_k=3` |
| 注入 = Prefix 或 Cross-Attention KV | §3.5.1 表 | 两式均实现，**默认 Prefix**（见偏离） |
| 不用对比学习（需大量同谱师标注对） | §3.5.2 | 完全采纳 |

**偏离 1**：奠基 §3.5.1 给出 Prefix / Cross-Attention KV 两式未指定默认。本计划规定**默认 Prefix**，理由：① 与 Plan 04 §4「RAG 参考序列拼接到前缀」一致；② 无需改 Decoder 结构即可生效。Cross-Attention KV 作为 `injection_mode="cross_kv"` 备选。记入 RFC-0010。

**偏离 2**：奠基文档未规定密度曲线如何与音频 emb 拼成检索向量。本计划规定：密度曲线按 Section 重采样到 `density_dim=64` 段（L2 归一化），拼接到 768d 音频 pooled 向量后，整体 L2 归一化作为库/查询向量。记入 RFC-0011。

## 3. 接口契约

对应骨架 `RAGRetriever`（`beatmorph/rag/retriever.py`）：

```python
class RAGRetriever:
    def __init__(self, top_k: int = 3, index_path: str | None = None,
                 injection_mode: str = "prefix") -> None: ...
    def build_index(self, corpus_dir: str) -> None: ...
    def retrieve(
        self,
        query_emb: Tensor,        # (time_seq, 768)  来自 Stage0，单条查询
        difficulty: int,          # 1-15，元数据过滤
        bpm: float,               # 元数据过滤/加权
    ) -> list[list[EventToken]]:  # 长度 = top_k，每条为参考序列
    def build_context(self, refs: list[list[EventToken]],
                      style_embs: Tensor) -> RAGContext: ...
```

张量形状（einops 风格，引用 `core/contracts/tensors.py`）：

| 名称 | 形状 | 含义 |
|------|------|------|
| `query_emb` | `(time_seq, 768)` | 查询音频 MERT 输出，帧率 `MERT_FRAME_RATE_HZ=25` |
| 检索库向量 | `(num_corpus, 768+density_dim)` | 音频 pooled 768d ⊕ 密度曲线 64d，L2 归一化 |
| 查询向量 | `(768+density_dim,)` | 同上构造 |
| `RAGContext.token_prefix` | `(batch, top_k, ref_seq_len)` | 检索到的参考 event 序列，`ref_seq_len≤AR_CONTEXT_TOKENS=1024` |
| `RAGContext.style_emb` | `(batch, top_k, feat)` | 各参考谱面风格向量，`feat=768` |

- 输出 `list[list[EventToken]]`（`EventToken.id ∈ [0, BPE_DEFAULT_VOCAB)`），由 Decoder（Plan 07）消费索引语义。
- `RAGContext` 通过 `tensors.RAGContext`（`top_k=3, ref_seq_len=256, feat=768`）描述，与契约常量一致。

## 4. 内部设计

- **索引构成**：`build_index` 遍历 `corpus_dir`，每条谱面读取①预提取的音频 MERT emb（Plan 01/08 产物）②密度曲线（Plan 03 统计量）；构造联合向量入库 FAISS `IndexFlatIP`（内积，配合 L2 归一化等价余弦）。并行维护结构化元数据表（BPM/流派/谱师/难度/`EventToken` 序列）。
- **密度曲线重采样**：将原始 Section 密度序列插值到 `density_dim=64`，与音频时间轴解耦，保证不同时长曲目向量同维。
- **检索流程**（`retrieve`）：查询 emb → pooling → 拼密度曲线 → L2 归一化 → FAISS Top-K（取 `top_k * 3` 候选）→ 元数据过滤（难度容差 ±2、BPM 容差 ±10%）→ 取前 `top_k`。
- **注入**：
  - `prefix`（默认）：参考 Token 序列拼接到 AR 序列前缀，因果掩码允许 attend（与 Plan 04 一致）。
  - `cross_kv`：参考序列经独立编码器映射为 Cross-Attention 的 K/V，每层注入（需 Plan 04 暴露接口）。
- **为何不用对比学习**（§3.5.2）：对比学习需大量「同谱师」正负样本对定义风格，标注获取困难；RAG 无需训练、可解释性强（用户可见参考来源），在符号音乐生成上客观与主观指标均优于 LoRA 微调。

## 5. 依赖关系

- **上游**：Plan 01 MERT 音频 emb（`AudioEmbedding`）、Plan 02 BPE 词表（`EventToken.id` 范围）、Plan 03 密度曲线、Plan 08 预处理产物（离线 emb + 统计量）。
- **下游**：Plan 04 `ARTransformer`（消费 `RAGContext`，可选注入）、Plan 09 CLI/API（暴露 `retrieve`）。
- **外部库**：`faiss-cpu`（主索引）、`numpy`、`torch`（仅张量搬运，无训练）。MERT/Demucs 不在本模块。

## 6. 里程碑与验收标准

对齐奠基 §7 Phase 2「集成 RAG 检索模块，实现参考风格迁移基础功能」。

| 里程碑 | 验收 |
|--------|------|
| M1 索引构建 | 10K 谱面建索引 < 1 小时（CPU 32 核），向量维度 `768+64` 无 NaN |
| M2 检索正确性 | 给定查询，Top-3 命中同 BPM 同难度同曲风的召回率 > 70%（人工抽样 200 查询） |
| M3 注入打通 | `retrieve` 输出经 Prefix 注入后 `ARTransformer.generate` 正常出码，shape 满足 `RAGContext` |
| M4 风格迁移盲测 | 启/关 RAG 的生成谱面在「风格相似度」人工盲测中胜率 > 65% |

## 7. 风险与缓解

| 风险 | 奠基编号 | 缓解 |
|------|---------|------|
| 检索到低质谱面污染风格 | R-3 | 仅索引质量过滤后语料（star≥3、play_count>500，Plan 08 过滤）；Plan 06 DPO 兜底 |
| Prefix 增加推理 token 数、拖慢 AR | R-4 | `ref_seq_len` 远小于 `AR_CONTEXT_TOKENS`；缓存常用曲风 prefix；必要时切 `cross_kv` |
| BPE 无坍缩风险（R-2 随 VQ 退役）；监控参考 event 多样性 | R-2 | 受 Plan 02 约束（BPE 词表）；监控检索结果 event token 熵，<40% 报警 |
| 元数据缺失无法按流派/谱师过滤 | — | 元数据缺失项仅按 BPM+难度+向量相似度兜底（RFC-0012） |

## 8. 测试策略

- **单元**：联合向量构造（维度、L2 范数=1）、密度曲线重采样确定性、元数据过滤逻辑、`retrieve` 返回长度恒为 `top_k`。
- **集成**：mock Plan 01/03 输出，`build_index` → `retrieve` → `build_context` 闭环，`RAGContext` shape 通过契约断言。
- **e2e**：真实 10K 索引 + 真实 MERT emb 查询 + Plan 04 AR 注入出码，标记 `@pytest.mark.e2e`；检索延迟 < 200ms/查询。

## 9. 开放问题

- [ ] RFC-0010：Prefix（默认）vs Cross-Attention KV 的最终生产默认值与切换成本。
- [ ] RFC-0011：密度曲线维度 `density_dim=64` 与归一化策略是否需根据召回率指标调参。
- [ ] RFC-0012：流派/谱师元数据缺失时的兜底过滤策略与权重。
