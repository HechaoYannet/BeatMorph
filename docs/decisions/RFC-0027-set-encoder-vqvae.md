# RFC-0027 — VQ-VAE encoder：栅格 1D-CNN → note 集合 set-transformer（评估）

- 状态：提案 ｜ 提出日期：2026-08-03 ｜ 决定日期：待定
- 提出者：Tokenizer 组
- 影响模块：plan 02（`beatmorph/tokenizer/vqvae.py` 编码器/解码器）。**契约 PatternToken/TokenSeq 不变**（每小节一 code）。非范式变更，属 plan 02 内编码器实现演进。

## 背景

VQ-VAE 冒烟 + 稀疏度诊断（见实现记忆层 2）量化了**栅格化 encoder 的结构性代价**：

```
真实 4K mania（difficulty 9-12）每 bar 平均 16.3 Note，撒到 4 lane × 64 bin = 256 格
→ (lane, bin) occupancy 仅 6.4%（空:有 = 15:1）
→ present 头 BCE 类不平衡严重，present 被空 bin 主导压低
→ decode 阈值 0.5 时产 0 个 Note（短训下）
```

**栅格化本身是信息损失源**：
1. **稀疏类不平衡**：64-bin 网格制造 256 格/bar 的稀疏空间，绝大部分 bin 为空 → present 头难收敛。
2. **bin 内多 note 冲突**：同 (lane, bin) 多 Note 只能保留先到者，信息有损（虽罕见）。
3. **CNN 在稀疏网格上学节奏型信号密度低**：1D-CNN over (lane, bins) 输入大量 0，注意力/卷积算在 0 上浪费。

phase 修复（[RFC-0026](RFC-0026-bar-boundary-phase-alignment.md)）让 Note 落规整拍点，但**不消除栅格稀疏性**——即便完美对齐，每 bar 16 Note 在 256 格仍 6.4% occupancy（plan §1 决策 time_bins=64 为满足 ±20ms 精度，不可降）。

## 提议

**评估 VQ-VAE v2：将 encoder 从"栅格 1D-CNN"换为"note 集合 set-transformer"**，保持 PatternToken/TokenSeq 契约（每小节一 code）与 VQ-VAE 量化/码本/防坍缩不变。

**设计要点**：
- **输入**：每小节 note 集合 `{(time_rel, lane, type_onehot, duration)}`，note 数不定（set，非序列）。
- **encoder**：note 级 token embedding（lane/type/duration 各自 embedding + 连续 time_rel 经 sinusoidal 编码）→ DeepSets 或 set-transformer（permutation-invariant pooling）→ 每 bar 一个 latent。
- **decoder**：latent → set 预测（输出 note 集合，用匈牙利匹配 loss 或 set prediction loss，如 DETR 式）。**无空 bin 概念，无 present 头，无类不平衡**。
- **契约不变**：仍每小节一个 code（`encode(chart) → list[PatternToken]`），下游 AR/planner 无感。

**何时触发评估**（非立即实施，决策门禁）：
1. 完成 [RFC-0026](RFC-0026-bar-boundary-phase-alignment.md) phase 修复后，在 50K 谱面训练 VQ-VAE v1（栅格版）至收敛。
2. 若 v1 达 M2 重建>95% & M3 codebook_usage≥0.5 → **不升级**，栅格版足够。
3. 若 v1 不达 M2 / present 头持续被抑制 → **触发本 RFC 升级 set encoder**，作为 VQ-VAE v2。

**与红线关系**：
- BasePlan §3.2.1 表写"编码器结构 = 1D-CNN + Transformer"。set-transformer 仍是 Transformer 家族（self-attention over note set），属"Transformer"实现细化，**非偏离 §3.2 VQ-VAE 锁定**。
- plan 02 §4 明确写"1D-CNN"，改它需更新 plan 02 §4 文本。**本 RFC 采纳后同步修订 plan 02 §4**（编码器结构项加 set-transformer 备选）。
- 不触碰 CLAUDE.md 红线 1（VQ-VAE 锁定）/红线 2（契约不变）。

## 备选方案

1. **保留栅格 + 加 pos_weight 缓解类不平衡**（短期采用，见 [RFC-0026] 配套的 VQ-VAE pos_weight）：present 头 BCE 加 `pos_weight≈15`。**作为 v1 短期增益采纳**，本 RFC 的 set encoder 是其不达标时的长期升级。

2. **软栅格化输入增强**（note 在 ±k bin 高斯激活）：降低 bin 边界抖动鲁棒性，但仍是栅格、仍稀疏。**作为 encoder 输入增强可选**，不替代 set encoder 的根治。

3. **彻底 BPE/event tokenizer**（放弃每小节一 code）：见 [RFC-0028](RFC-0028-bpe-event-tokenizer-constitutional-amendment.md) 修宪议案。本 RFC 是 VQ-VAE 友好版（契约不变），[RFC-0028] 是范式级 Plan B。本 RFC 优先；若本 RFC set encoder 仍不达 M2，再考虑 [RFC-0028]。

## 后果

- **若触发采纳**：`vqvae.py` encoder/decoder 重写为 set 版，栅格化函数 `rasterize_*` 退居统计/可视化用；契约不变，AR/planner 无感。
- **若不触发**（v1 达标）：本 RFC 标记「驳回/搁置」，栅格版为最终版。
- **决策门禁**：50K 训练 M2 数据为准，不凭冒烟判断。当前不实施，仅登记。
- plan 02 §4 待本 RFC 决议后同步编码器结构文本。

## 关联

- 奠基：§3.2.1（编码器结构）、§3.2.2（码本语义 4 维度）。
- plan 02 §4（1D-CNN + Transformer）→ 本 RFC 提 set-transformer 备选。
- 相关 RFC：[RFC-0026](RFC-0026-bar-boundary-phase-alignment.md)（phase 修复，正交前置）、[RFC-0028](RFC-0028-bpe-event-tokenizer-constitutional-amendment.md)（BPE 修宪，范式级 Plan B）。
- 触发来源：present 头稀疏度诊断（实现记忆层 2）。
