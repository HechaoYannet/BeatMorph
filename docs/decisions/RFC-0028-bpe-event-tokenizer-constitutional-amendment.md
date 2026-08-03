# RFC-0028 — 修宪议案：谱面 tokenizer 范式从 VQ-VAE 小节粒度改为 BPE/event tokenizer

- 状态：**进入审核流程（待决策者亲自审核）** ｜ 提出日期：2026-08-03 ｜ 更新：2026-08-04 ｜ 决定日期：未定
- 提出者：Tokenizer 组
- 影响模块：**BasePlan §3.2 全节**、§2 数据流、§3.4（AR）、§3.7（解码）、`core/contracts`（TokenSeq/PatternToken 契约）、plan 02/04/07、CLAUDE.md 红线 1。
- 性质：**范式级变更**，非局部 RFC。若采纳将修订 BasePlan §3.2，触及项目宪法（CLAUDE.md 红线 1「技术选型锁定」）。

> 本议案经两轮质疑与真实数据复核后，**定位从「VQ-VAE 失败后的 Plan B」升级为「与 VQ-VAE 并列且理论上更优的首选路线」**，并**战略跳过 VQ-VAE vs BPE 对比验收阶段，直接进入方案审核流程**，由决策者亲自审核。在决议前，VQ-VAE 小节粒度仍为已实现范式（[RFC-0026](RFC-0026-bar-boundary-phase-alignment.md) phase 修复已落地）；本议案若采纳，VQ-VAE 实现将作为对照保留或废弃。

## 1. 议案动议

**提议**：放弃 BasePlan §3.2 的「VQ-VAE + 小节粒度」范式，改采 **REMI 式 event/BPE tokenizer**——note 流直接离散化为 event token 序列（`Bar / Position / Pitch(lane) / Duration / Chord...`），经 BPE 合并高频 event 组合为复合 token，下游 AR 直接建模 event 序列，无 VQ-VAE、无小节栅格、无码本。

## 2. 动机

### 2.1 VQ-VAE 小节码本的结构性硬伤（范式级，非 bug）

VQ-VAE 把**每小节映射到一个 2048 码本里的一个 code**，训练后码本是 2048 个"典型小节 pattern"字典，AR 逐小节**选** code 再 decode 还原。**本质是"小节级 nearest-neighbor 检索重组"**——新曲第 k 小节只能落到训练集里"最像"的某一类 pattern。

**根本矛盾**：现实中不可能有两首歌某一小节节奏完全一致。一首新曲某小节的节奏型（16 分阶梯 + 切分 + 特定 lane 分布）几乎必然是训练集里没见过的精确组合，VQ-VAE 只能给"最近"的 code：
- 节奏微偏（切分点差 1/32 音）→ 量化到相邻 code → 还原后节奏失真，**信息损失不可逆**。
- 实证：phase 修复（RFC-0026）后仍有 24% note 不落规整拍点，VQ 的 bin 量化将其硬塞进最近 code。
- 2048 码本远覆盖不了小节 pattern 组合空间（4 lane × 64 bin × 5 type × duration），升 4096 只是缓兵。

**BPE 对比**：event 流原样保留每个 note 的精确 (time, lane, type, duration)，BPE 只合并高频共现 event 组合（如「左手 16 分阶梯 4 连」→ 复合 token），**不量化掉任何信息**——未合并 note 仍以原子 event 出现。生成新曲时 AR 可输出训练集没见过的 event **新组合**，而非从固定码本选最近的。这是**生成 vs 检索**的本质区别，BPE 原生契合，VQ-VAE 结构性受限。

> **对 VQ-VAE M2 验收的反思**：奠基 §7 M2「重建准确率>95%」测的是**训练集内部往返**（encode→decode 同一谱面），VQ 量化往返必然高分，**掩盖了泛化硬伤**。真正应测的是**新曲生成适配**（奠基 §3.4 M4 生成盲测），这才是范式可行性的试金石。前期"60 步 acc 93.5%"是 4 份样本小模型过拟合训练分布，不构成范式泛化可行的证据。

### 2.2 VQ-VAE 工程缺陷链（已诊断）

GPU 冒烟 + 稀疏度诊断暴露：
1. **栅格稀疏化**（[RFC-0027] 量化）：每 bar 16 Note 撒 256 格 → 6.4% occupancy → present 头类不平衡 15:1，需 pos_weight 缓解。
2. **bin 内 note 冲突**：同 (lane, bin) 多 Note 仅保留先到者，信息有损。
3. **精度 vs 类平衡两难**：time_bins=64 满足 ±20ms 但放大稀疏；降 bins 缓解稀疏但破精度。
4. **code 语义承载过重**：单 code 需同时编码 density/节奏型/手型/键位分布 4 维，信息瓶颈大。

### 2.3 BPE 范式优势
- **无栅格**：note 作为离散 event 直接序列化，无稀疏类不平衡、无 bin 冲突、note 精确时间作 Position event 保留。
- **数据驱动组合**：BPE 从数据学高频 event 组合，等价于"自动发现节奏型词典"，比 VQ 码本更显式可解释、可插拔。
- **业界验证**：REMI/MIDI-Like/Octuple 在符号音乐生成是 SOTA；BPE 在 LLM 已充分验证，扩展性强（质疑4）。

## 3. 训练开销复核（纠正前期过度悲观）

> 前期估算"序列 ×10、上下文 256→4096、算力 ×16-32"是脱离真实数据的教条。用本项目 4K mania 真实分布复核：

### 3.1 真实序列长度
```
14 份 4K mania（diff 9-12）：每 bar 平均 16.3 note，~94 bar/曲 → ~1533 note/曲
BPE 编码：每 note ≈ 3 event（Pitch/lane + Position + Duration）+ 每 bar 1 Bar event
→ 每曲 ~4700 event
```

### 3.2 上下文需求（非 4K+，是 ~1024 量级）
AR 建模音乐结构不需一次看全曲——pop/电子乐结构单位是 4-8 小节乐句、16-32 小节段落：

| 上下文窗口 | VQ-VAE（1 token/bar）覆盖 | BPE（~30 event/bar）覆盖 | 充分性 |
|-----------|----------------------|----------------------|--------|
| 256 | 256 bar（全曲） | ~8 bar（2 乐句） | BPE 不足全曲，够乐句级 |
| 512 | — | ~16 bar（1 段落） | 够段落级 |
| 1024 | — | ~32 bar（2 段落） | **够结构级** |

**结论**：BPE 上下文需求 ~1024（段落级），分段生成 + 衔接（RFC-0008）可覆盖全曲，每段 1024 上下文。**非前期估算的 4096，省 4×**。前期把"VQ 全曲 256 小节覆盖"硬套给 BPE 是误推——BPE 不需要单窗口全曲上下文。

### 3.3 算力开销（非 ×16-32，是 4-8×）
12 层 AR 算力主要在 attention（O(L²)）与 FFN（O(L)）：
- attention：1024²/256² = **16×**（FlashAttention 下占比本就小）
- FFN：1024/256 = **4×**（长序列下占大头）
- 实际 step 开销 ≈ **4-8× VQ 路线**，非 16-32×。
- 且 BPE **省掉 VQ-VAE 预训练整个阶段**（6.8M params + 栅格化 + 防坍缩三连调试），净抵消一部分。
- BPE 词表训练是确定性统计（sentencepiece 式），CPU 数小时完成，对比 VQ 神经网络训练成本可忽略。
- 音游 AR 非 LLM 级算力：1024 上下文 / 12 层 / vocab ~4K，单卡 RTX 4060 可训。

### 3.4 长上下文兜底（稀疏注意力）
若 1024 仍不足（需全曲结构 attention），**稀疏/滑窗注意力**把 O(L²) 降到 O(L·k)，4096 上下文也可承受——这是真实有效的大保底，前期未给足权重。

## 4. 决策者四条审核结论（2026-08-04）

经质疑与复核，决策者认定 BPE 应**直接进入方案审核流程，战略跳过 VQ vs BPE 对比验收**，理由：

1. **VQ-VAE 引入更多训练层，端到端误差累积，不确定因素增加**——栅格化 + CNN encoder + 量化 + decoder 四层串联，每层都引入误差与调参自由度（time_bins/码本大小/commit 权重/防坍缩三连…），不确定因素层层叠加；BPE 仅"离散化 + 合并"，确定性强。
2. **VQ-VAE 存在结构性工程偏差，对本项目问题的解决性弱**——见 §2.1/§2.2，VQ 为"连续信号压缩"设计，对"离散 note 事件序列生成"是错配，phase/稀疏/bin 冲突皆此偏差的表征。
3. **VQ-VAE 鲁棒性更弱、泛化较差，不符合适配更多类型音游的目标**——VQ 码本与"1 小节粒度 + 4K 量化"强绑定，扩到 6K/osu!std/maimai 需重训码本 + 重设栅格；BPE event 抽象与键位数/模式解耦，扩展性天然更强（质疑4 延伸：BPE 词表可跨模式共享或增量）。
4. **BPE 已在 LLM 充分验证，扩展性强**——event LM范式（GPT 式原生序列建模）toolchain 成熟（HF Transformers 直接套），未来蒸馏/对齐/DPO 复用 LLM 生态更顺；VQ+GPT 在符号领域反非主流。

## 5. 影响面（采纳则需改的全部）

| 受影响项 | 当前（VQ-VAE） | 改后（BPE/event） | 变更级别 |
|---------|--------------|-----------------|---------|
| BasePlan §3.2 全节 | VQ-VAE + 1 小节粒度 + 码本 2048/4096 + 防坍缩三连 | event tokenizer + BPE + 词表 | **重写** |
| BasePlan §3.4（AR） | 建模 PatternToken 序列（每小节一 token，上下文 256） | 建模 event 序列（~4700/曲，上下文 ~1024 分段） | 上下文重设、分段生成主路径化 |
| BasePlan §3.7（解码） | VQ-VAE Decoder 还原 note | 直接 event→note 映射（无解码网络） | 大幅简化 |
| `core/contracts` | PatternToken(code,bar_index,...) / TokenSeq | EventToken 序列契约（重设） | 契约替换（机械性，非架构阻力） |
| plan 02 | VQ-VAE 实现 | 重写为 event tokenizer + BPE 训练 | 重写 |
| plan 04（AR） | 消费 TokenSeq（256 上下文，每小节一 token） | 消费 event 序列（1024 上下文 + 分段） | 重设上下文/条件注入 |
| plan 07（解码） | 调 VQVAETokenizer.decode | 调 event→note 还原 | 简化 |
| CLAUDE.md 红线 1 | 锁定 VQ-VAE（不得换 BPE/规则/LLM） | 解锁 VQ-VAE、改锁 event tokenizer | **宪法修订** |
| 防坍缩（R-2） | VQ 码本坍缩风险 | BPE 无坍缩，风险消失 | 风险表删 R-2 |
| **planner（已实现已训练）** | DensityPlanner + compute_section_stats | **保留不动**（与 tokenizer 范式正交） | **零改动** |
| 已实现 VQ-VAE 脚手架 | vqvae.py/Dataset/LitModule/configs | 废弃或作对照保留 | ~30% 已实现代码回炉 |

**关键事实**：planner 与 tokenizer 范式正交（输入 audio_emb+section_bounds，输出 list[Section]，不碰 token），**切换 BPE 对 planner 零影响**，已训练 planner 权重可复用；phase 修复（RFC-0026）BPE 下仍有效且必需（Bar/Position event 仍需正确小节边界）。

## 6. 流程决定

- **战略跳过 VQ-VAE vs BPE 对比验收**：决策者认定 VQ 范式级缺陷（§2.1/§4）已使"对比谁更好"无必要，直接审核 BPE 方案本身。前期 RFC-0026 phase 修复 + RFC-0027 set encoder + pos_weight 等 VQ 框架内努力作为"框架内最大努力"已完成其历史使命（证伪了 VQ 可低成本救活），不作为阻塞 BPE 审核的门禁。
- **进入方案审核流程**：由决策者亲自审核本议案 + 后续 BPE 详细方案（词表设计、event schema、AR 上下文策略、迁移计划）。
- 1K 级 VQ-VAE 训练（已在 GPU 平台准备）可继续跑作为**对照基线**（证伪/留档），不作为决策阻塞。

## 7. 风险与缓解

1. **范式革命成本**：契约（TokenSeq→EventToken）影响 AR/RAG/decoder 链路；planner 零影响。**缓解**：planner 全家 + 数据解析 + phase 修复保留，~30% 脚手架回炉（VQ 部分），70% 资产保留。
2. **CLAUDE.md 红线 1 修订是宪法事件**：**缓解**：本议案即为修宪议案，审核通过即同步修订红线 1 与 BasePlan §3.2，全员对齐新叙事（event tokenizer 仍是"自监督理解"哲学——从百万谱面学 event 分布，无显式标注）。
3. **AR 上下文/序列长度**：**缓解**：§3 复核示上下文 ~1024 可行（非 4096），分段生成主路径化（RFC-0008 从边界情况升为常态），稀疏注意力作大保底。
4. **BPE 词典对 mania 适配**：REMI 为钢琴多轨设计，mania 4K 键位/手型语义是否受益 BPE 合并需 PoC 确认。**缓解**：词表大小可调，event schema 可纳入 lane/chord 语义；4K 数据下 note 类型简单（多 TAP），BPE 合并空间主要在节奏型+手型组合，预期适配。
5. ~~过早放弃 VQ-VAE~~：**已撤销**。经 §2.1 范式级分析 + §4 决策者四结论，认定非"过早"而是"范式错配应纠偏"，前期门禁设计基于"VQ 可低成本救活"假设，该假设已被质疑1/2 证伪。

## 8. 待审核的 BPE 详细方案（决策者审核通过后展开）

本议案仅定**范式方向**，具体方案待审核通过后由 Tokenizer 组出详细设计：
- event schema（Bar/Position/Pitch(lane)/Duration/Chord/NoteType…，mania 4K 裁剪）
- BPE 词表训练流程（sentencepiece/HF tokenizers，词表大小 ~2K-8K）
- AR 改造（上下文 1024 + 分段 + 条件注入复用 planner/RAG）
- 契约迁移（PatternToken→EventToken，下游 plan 04/05/07 同步）
- BasePlan §3.2 重写文本 + CLAUDE.md 红线 1 修订

## 关联

- 奠基：§3.2（VQ-VAE 锁定，本议案提议修订）、§3.2.3（FSQ 否决，本议案另辟 event 路径）、§3.4（AR 依赖 token 化范式）、§7 M2/M4（验收口径反思：M2 重建掩盖泛化，应重 M4 生成盲测）。
- CLAUDE.md：红线 1（技术选型锁定 VQ-VAE，本议案提议解锁 + 改锁 event tokenizer）。
- 相关 RFC：[RFC-0026](RFC-0026-bar-boundary-phase-alignment.md)（phase 修复，BPE 下仍有效，保留）、[RFC-0027](RFC-0027-set-encoder-vqvae.md)（set encoder，BPE 采纳后作废/搁置）。
- 触发来源：VQ-VAE GPU 冒烟 + 稀疏度/phase 诊断 + 范式级质疑（VQ 检索重组 vs BPE 生成）+ 训练开销真实数据复核 + 决策者四条审核结论。
