# 模块实施计划总索引

本目录收录 BeatMorph 各模块的详细实施计划（plan）。所有 plan 以
[`docs/BasePlan.md`](../BasePlan.md)（奠基文档 v2.0）为唯一技术权威，
本目录仅做落地拆解：接口契约、依赖关系、里程碑、验收标准、风险与回退。

## 如何阅读

1. 先读 [奠基文档](../BasePlan.md) 理解全局范式。
2. 读 [`docs/CODE_STRUCTURE.md`](../CODE_STRUCTURE.md) 理解代码骨架与模块拓扑。
3. 再按本索引进入对应模块 plan。

## 计划清单

| 编号 | 模块 | 对应奠基章节 | 状态 | 阶段 |
|------|------|-------------|------|------|
| 00 | [核心契约与张量形状](./00-core-contracts.md) | §2 / §3.7 | 🟡 草案 | Phase 1 |
| 01 | [Stage 0 音频编码器](./01-audio-encoder.md) | §3.1 | 🟡 草案 | Phase 1 |
| 02 | [BPE/event 谱面 Tokenizer](./02-tokenizer-bpe.md) | §3.2 | 🟡 草案 | Phase 1 |
| 03 | [Stage 1 全局密度规划](./03-planner-density.md) | §3.3 | 🟡 草案 | Phase 1 |
| 04 | [Stage 2 AR Pattern 生成](./04-generation.md) | §3.4 | 🟡 草案 | Phase 2 |
| 05 | [RAG 风格检索](./05-rag-retrieval.md) | §3.5 | 🟡 草案 | Phase 2 |
| 06 | [DPO 偏好对齐](./06-dpo-alignment.md) | §3.6 | 🟡 草案 | Phase 3 |
| 07 | [解码与后处理 / 格式导出](./07-decoder-postprocess.md) | §3.7 | 🟡 草案 | Phase 2 |
| 08 | [数据预处理流水线](./08-data-pipeline.md) | §4 | 🟡 草案 | Phase 1 |
| 09 | [训练/推理基础设施与 CLI/API](./09-infra-cli-api.md) | §5 / §7 | 🟡 草案 | Phase 1-3 |

**状态图例**：🟡草案 → 🟠评审中 → 🟢已批准 → 🔵实施中 → ✅已完成

## 计划统一格式

每份 plan 须包含以下章节（缺失项写「N/A」并说明原因）：

1. **目标与范围** — 本模块交付什么、不交付什么
2. **与奠基文档的对应** — 引用具体章节，标明任何偏离与理由
3. **接口契约** — 输入/输出类型（引用 `beatmorph.core.contracts`）、张量形状
4. **内部设计** — 子组件、算法、关键决策
5. **依赖关系** — 上游/下游/外部库
6. **里程碑与验收标准** — 可量化、可测试
7. **风险与缓解** — 对接奠基文档风险表 R-1..R-6
8. **测试策略** — 单元/集成/e2e
9. **开放问题** — 待决策项，指向 RFC（见 `docs/decisions/`）

## 变更管理

- plan 的实质性变更须提交 RFC 至 [`docs/decisions/`](../decisions/) 并更新本索引状态。
- plan 与代码同名编号便于追溯（如 `02-tokenizer-bpe.md` ↔ `beatmorph/tokenizer/bpe.py`）。
- **RFC-0028**：Tokenizer 范式由 VQ-VAE 改为 BPE/event，`02-tokenizer-vqvae.md` 已重命名为 `02-tokenizer-bpe.md`；旧 VQ-VAE 实现移至 `archive/vqvae-baseline` 分支，主路径不再维护。
