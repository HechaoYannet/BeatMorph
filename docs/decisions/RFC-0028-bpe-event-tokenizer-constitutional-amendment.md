# RFC-0028 — 修宪议案：谱面 tokenizer 范式从 VQ-VAE 小节粒度改为 BPE/event tokenizer

- 状态：**修宪议案（待决策，非采纳）** ｜ 提出日期：2026-08-03 ｜ 决定日期：未定
- 提出者：Tokenizer 组
- 影响模块：**BasePlan §3.2 全节**、§2 数据流、§3.4（AR）、§3.7（解码）、`core/contracts`（TokenSeq/PatternToken 契约）、plan 02/04/07、CLAUDE.md 红线 1。
- 性质：**范式级变更**，非局部 RFC。若采纳将修订 BasePlan §3.2，触及项目宪法（CLAUDE.md 红线 1「技术选型锁定」）。

> 本文档是修宪**议案**，记录动机、影响面、触发条件与风险，供项目决策者裁定。在决议前，VQ-VAE 小节粒度仍为唯一在用范式（[RFC-0026](RFC-0026-bar-boundary-phase-alignment.md) phase 修复 + [RFC-0027](RFC-0027-set-encoder-vqvae.md) set encoder 是其框架内的演进/兜底）。

## 1. 议案动议

**提议**：放弃 BasePlan §3.2 的「VQ-VAE + 小节粒度」范式，改采 **REMI 式 event/BPE tokenizer**——note 流直接离散化为 event token 序列（`Bar / Position / Pitch(lane) / Duration / Chord...`），经 BPE 合并高频 event 组合为复合 token，下游 AR 直接建模 event 序列，无 VQ-VAE、无小节栅格、无码本。

## 2. 动机（触发来源）

VQ-VAE GPU 冒烟 + 稀疏度诊断暴露「VQ-VAE + 小节栅格」范式的**结构性信息损失链**：

1. **栅格稀疏化**（[RFC-0027] 量化）：每 bar 16 Note 撒 256 格 → 6.4% occupancy → present 头类不平衡 15:1。
2. **bin 内 note 冲突**：同 (lane, bin) 多 Note 仅保留先到者。
3. **时间精度 vs 类平衡两难**：time_bins=64 满足 ±20ms 但放大稀疏；降 bins 缓解稀疏但破精度（plan §1 既定）。
4. **code 语义承载过重**：单 code（2048 选 1）需同时编码 density/节奏型/手型/键位分布 4 维（§3.2.2），信息瓶颈大，需大码本 + 长训练。

**BPE/event 范式的对比优势**（理论）：
- **无栅格**：note 作为离散 event 直接序列化，天然无稀疏类不平衡、无 bin 冲突、note 精确时间作为 Position event 保留。
- **数据驱动组合**：BPE 从数据学高频 event 组合（如「左手 16 分阶梯」合并为复合 token），等价于"自动发现节奏型词典"，比 VQ 码本更显式可解释。
- **业界验证**：REMI（音乐）、MIDI-Like、Octuple 等在符号音乐生成已是 SOTA 范式；VQ-VAE+GPT 在符号领域反非主流（VQ-VAE 强在连续信号压缩如图像/音频波形，§3.2.3 已论 FSQ 否决策同源理由，但 event 数据本就是离散的，无需 VQ 再离散化）。

## 3. 影响面（采纳则需改的全部）

| 受影响项 | 当前（VQ-VAE） | 改后（BPE/event） | 变更级别 |
|---------|--------------|-----------------|---------|
| BasePlan §3.2 全节 | VQ-VAE + 1 小节粒度 + 码本 2048/4096 + 防坍缩三连 | event tokenizer + BPE + 词表 | **重写** |
| BasePlan §3.4（AR） | 建模 PatternToken 序列（每小节一 token） | 建模 event 序列（note 级，序列长 ~10×） | 上下文长度/注意力需扩 |
| BasePlan §3.7（解码） | VQ-VAE Decoder 还原 note | 直接 event→note 映射（无解码网络） | 简化 |
| `core/contracts` | PatternToken(code,bar_index,...) / TokenSeq | EventToken 序列契约（重设） | **契约破坏性变更** |
| plan 02 | VQ-VAE 实现 | 重写为 event tokenizer + BPE 训练 | 重写 |
| plan 04（AR） | 消费 TokenSeq（每小节一 token） | 消费 event 序列 | 重设上下文/条件注入 |
| plan 07（解码） | 调 VQVAETokenizer.decode | 调 event→note 还原 | 简化 |
| CLAUDE.md 红线 1 | 锁定 VQ-VAE（不得换 BPE/规则/LLM） | 解锁 VQ-VAE、改锁 event tokenizer | **宪法修订** |
| 防坍缩（R-2） | VQ 码本坍缩风险 | BPE 无坍缩，风险消失 | 风险表删 R-2 |

## 4. 触发条件（何时真正升堂表决）

**当前不表决**。设门禁：仅当 [RFC-0026](RFC-0026-bar-boundary-phase-alignment.md) phase 修复 + [RFC-0027](RFC-0027-set-encoder-vqvae.md) set encoder（VQ-VAE 框架内最大努力）在 50K 谱面训练后**仍不达 M2 重建>95% 或 M3 codebook_usage≥0.5**，方启动本修宪表决。

表决前需补：
- event tokenizer 在 osu! mania 4K 上的 PoC（BPE 词表大小、序列长度、AR 训练可行性、生成质量）。
- VQ-VAE v1+v2 vs BPE 在同 50K 子集的重建质量/生成多样性/codebook 健康 对比。
- 契约破坏对下游 plan 04/07 的迁移成本评估。

## 5. 风险

1. **范式革命成本高**：契约（TokenSeq）破坏影响 AR/planner/RAG 全链路，迁移工程量大，Phase 1 已实现的 planner/VQ-VAE 脚手架需回炉。
2. **CLAUDE.md 红线 1 修订是宪法事件**：项目"自监督理解"哲学下 VQ-VAE 是范式符号，改 event tokenizer 需全员对齐新叙事。
3. **序列长度爆炸**：event 序列较"每小节一 token"长 ~10×，AR 上下文 256 tokens（§3.4.1）从覆盖 256 小节降到 ~25 小节，需扩上下文或分段生成，工程复杂度上升。
4. **BPE 词典对 mania 适配未验证**：REMI 为钢琴多轨设计，mania 4K 键位/手型语义是否受益 BPE 合并需 PoC 确认，可能词表冗余。
5. **过早放弃 VQ-VAE**：当前 VQ-VAE 问题（phase bug + 类不平衡）已各有定向修复（RFC-0026/0027），若修复见效则本议案不必要。**门禁设计正是为防过早革命**。

## 6. 建议决议（提案方立场）

**暂不采纳，登记为「待 PoC」**。坚持 VQ-VAE 范式（守 CLAUDE.md 红线 1），先做：
1. [RFC-0026] phase 修复（立即，确定性 bug）。
2. VQ-VAE present 头 pos_weight（立即，类平衡缓解）。
3. [RFC-0027] set encoder 作为 VQ-VAE v2 兜底（待 50K M2 数据触发）。
4. 本修宪议案（[RFC-0028]）**仅当 v1+v2 均不达 M2 时**启动表决，表决前补 PoC。

理由：范式革命应穷尽框架内修复后为之；当前问题有低成本定向解，尚不到革命门槛。

## 关联

- 奠基：§3.2（VQ-VAE 锁定，本议案提议修订）、§3.2.3（FSQ 否决，本议案另辟 event 路径）、§3.4（AR 依赖 token 化范式）。
- CLAUDE.md：红线 1（技术选型锁定 VQ-VAE，本议案提议解锁）。
- 相关 RFC：[RFC-0026](RFC-0026-bar-boundary-phase-alignment.md)（phase 修复，框架内）、[RFC-0027](RFC-0027-set-encoder-vqvae.md)（set encoder，框架内兜底）。
- 触发来源：VQ-VAE GPU 冒烟 + 稀疏度/phase 诊断（实现记忆层 1-2）。
