# 模块实施计划总索引（v3.0）

本目录收录 BeatMorph 各模块的实施计划（plan）。所有 plan 以 [`docs/BasePlan.md`](../BasePlan.md)（**v3.0，最高技术权威**）与其完整论证 [`RFC-0029`](../decisions/RFC-0029-phigros-continuous-chart-generation.md)（**已采纳**）为准，本目录只做落地拆解：接口契约、内部设计、依赖、里程碑、风险、测试策略、开放问题。

> **v3.0（2026-08-05，RFC-0029 修宪）**：目标 osu!mania 4K → **Phigros**；生成范式 → **判定线局部系多线标记点过程 + 掩码补全 + 非齐次泊松 NLL**；主路径格式锁定 **RPEJSON**；**v1 即支持多判定线**。

## 如何阅读

1. 先读 [奠基文档](../BasePlan.md)（技术权威）与 [CLAUDE.md](../../CLAUDE.md)（工程红线）。
2. 再读 [RFC-0029](../decisions/RFC-0029-phigros-continuous-chart-generation.md)（范式论证、六条决议、§5 评估与消融设计）。
3. 需要格式/单位/数据/文献事实时查 [`docs/knowledges/`](../knowledges/)：
   [phigros-format.md](../knowledges/phigros-format.md)（RPEJSON 逐字段）｜[phigros-units-and-geometry.md](../knowledges/phigros-units-and-geometry.md)（单位与几何，prpr 源码 A 级）｜[phira-dataset-survey.md](../knowledges/phira-dataset-survey.md)（数据源实测）｜[chart-generation-literature.md](../knowledges/chart-generation-literature.md)（文献与评估协议）。
4. 再按本索引进入对应模块 plan。**plan 与 BasePlan 冲突时以 BasePlan 为准，并开 RFC。**

## 计划清单（00-08，共九份）

| 编号 | 模块 | 对应代码 | 对应奠基章节 | 阶段 | 状态 |
| --- | --- | --- | --- | --- | --- |
| 00 | [核心契约：判定线 / 音符 / 谱面 / 强度场张量](./00-core-contracts.md) | `beatmorph/core/contracts/` | §2、§3.2、§3.4 | Phase 1 | ✅ 已完成 |
| 01 | [Stage 0 音频编码器：MERT-v1-330M + LoRA Adapter](./01-audio-encoder.md) | `beatmorph/audio/` | §2、§3.1 | Phase 1 | 🟡 草案 |
| 02 | [数据流水线：Phira 获取 + RPEJSON 解析 + 质检 + 特征离线提取](./02-data-pipeline.md) | `beatmorph/data/`、`beatmorph/io/formats/rpejson/`（读） | §4、§3.2.4 | Phase 1 | 🔵 实施中 |
| 03 | [强度场模块（判定线局部系多线标记点过程）](./03-field.md) | `beatmorph/field/` | §1.2、§2、§3.2、§3.4 | Phase 2 | 🔵 实施中 |
| 04 | [生成主干：掩码补全 Encoder-Decoder](./04-generation.md) | `beatmorph/generation/` | §1.2、§2、§3.3、§3.4、§3.5 | Phase 2 | 🔵 实施中（M1–M4/M6；G1–G4 全绿） |
| 05 | [解码与合法性后处理（强度场 → 离散事件 → RPEJSON）](./05-decoder-postprocess.md) | `beatmorph/decoder/`、`beatmorph/io/formats/rpejson/`（写） | §2、§3.6、§9 | Phase 2 | 🟡 草案 |
| 06 | [评估与实验设计（指标 / 协议 / B1-B6 / 人评）](./06-eval.md) | `beatmorph/eval/` | §1.3、§3.2.2、§9 | Phase 2（人评 Phase 3） | 🟡 草案 |
| 07 | [训练基础设施（Lightning / Hydra / G1-G4 门禁 / 环境自检）](./07-infra-training.md) | `beatmorph/infra/`、`configs/` | §5、§9 | Phase 1-2 | 🟡 草案 |
| 08 | [CLI / API / 端到端入口](./08-cli-api.md) | `beatmorph/cli/`、`beatmorph/api/` | §2、§3.6、§7 | Phase 2（API Phase 3） | 🟡 草案 |

> 九份 plan 均已落地（00-04 由对应模块组并行重写，05-08 由解码/评估/基础设施组重写）。若某个链接打不开，说明该文件被重命名而本表尚未同步——请提交方一并更新本表。

**状态图例**：🟡 草案 → 🟠 评审中 → 🟢 已批准 → 🔵 实施中 → ✅ 已完成

## 计划统一格式（每份 plan 必含 9 节）

1. **目标与范围** —— 本模块交付什么、**明确不交付什么**；
2. **与奠基文档对应** —— 逐条引用 BasePlan / RFC 的具体章节，并显式声明任何偏离与理由；
3. **接口契约** —— 输入/输出类型与张量形状（引用 `beatmorph/core/contracts`，红线 2）；
4. **内部设计** —— 子组件、算法、关键决策；
5. **依赖关系** —— 上游 / 下游 / 外部库；
6. **里程碑与验收** —— **可量化、可测试**；**任何新训练目标/损失必须先过 G1-G4**（见下）；
7. **风险与缓解** —— 对接 BasePlan §6 的 **R 编号**（R-1 … R-8）；
8. **测试策略** —— 单元 / 集成 / e2e，并声明哪些断言进默认 CI；
9. **开放问题** —— 待决策项与「未查证」事实，指向 `docs/decisions/` 的 RFC。

文首固定两行状态头：

```
> 状态：🟡 草案 ｜ 阶段：Phase N ｜ 负责：<组名>
> 对应代码：beatmorph/... ｜ 对应奠基章节：§X
```

### 四条跨 plan 的硬性约束（反复出现在各 plan 中）

- **门禁先行**：任何新训练目标/损失，在扩大数据规模前必须通过 **G1-G4**（`beatmorph/infra/sanity.py`：单 batch 过拟合 / 打乱标签对照 / 常数基线 `λ = N/abs(Ω)` / 契约断言），**结果写入训练日志**；门禁未绿不得扩数据（BasePlan §9、CLAUDE.md §5.8）。
- **物理常量一律派生**：帧率/单位/网格必须写成派生式并在启动时断言（如 `MERT_FRAME_RATE_HZ = MERT_SAMPLE_RATE_HZ / MERT_CONV_STRIDE_PRODUCT`、`dx = RPE_STAGE_WIDTH / x_bins`），**禁止硬编码**；mock/fixture 同样不得固化物理常量（CLAUDE.md 红线 7、RFC-0029 §7-6）。
- **评估在原始时间域（秒）**：不得在帧索引域评估（RFC-0029 §7-5）。**场网格是 τ（拍相对坐标），评估一律换算回秒**。
- **时间换算只在 `field/` 内**：秒 ↔ τ（beat-aligned，基本格 **1/48 拍**）的换算**只允许在 `field/` 实现**，`J(τ) = dt/dτ` 由谱面 `BPMList` 派生、**不得硬编码**；契约层 `PhigrosNote.t` 仍用秒；下游（generation / decoder / io / eval）**不得各自再实现一套时间换算**，并有「场网格 ↔ 秒」**往返无损契约测试（覆盖多 BPM 段）**进默认 CI（CLAUDE.md 红线 7、RFC-0029 §3.1/§7-8；落地见 Plan 03 M12）。

## 变更管理

1. **实质性变更须提 RFC**：任何偏离 BasePlan / RFC-0029 的技术选型、契约变更或范式调整，先提 RFC 至 [`docs/decisions/`](../decisions/README.md) 并更新本索引状态；plan 自身只做落地拆解，不得私自改选型。
2. **状态流转**：草案 →（评审）→ 已批准 →（实现）→ 实施中 → 已完成；里程碑勾选与状态更新由**该模块的负责组**在完成时同步。
3. **编号可追溯**：plan 与代码按同一编号对应（`05-decoder-postprocess.md` ↔ `beatmorph/decoder/`、`06-eval.md` ↔ `beatmorph/eval/`），使「哪份文档管哪段代码」无歧义。
4. **一个文件一个 owner**：并行协作时按 [AGENTS.md](../../AGENTS.md) §2 的模块归属划界，**不并行写同一文件**；`core/contracts` 与 `configs/` 分别由 contracts-agent 与 infra-agent 统一维护。

## 已退役的 v2.x plan（RFC-0029）

> **RFC-0029 采纳（2026-08-05）后，以下 v2.x 计划整体退役，不在主路径维护：**
> - **`02-tokenizer-bpe.md`**（BPE/event tokenizer，RFC-0028 产物）—— 谱面表示改为**判定线局部系多线强度场**，不再需要 token 化主路径（其「POS+NUDGE 无损离散化」思想仅作参考，见 RFC-0029 §4.3）；
> - **`03-planner-density.md`**（Stage 1 全局密度规划）—— **取消该层**：段级均值池化是结构性信息瓶颈，且 Phigros 不需要密度规划（RFC-0029 §8 选型矩阵、POSTMORTEM §2.6）；
> - **`05-rag-retrieval.md`**（RAG 风格检索）—— Phase 3 待重估，RFC-0029 未涉及（BasePlan §3.5/§3.7）；
> - **`06-dpo-alignment.md`**（DPO 偏好对齐）—— Phase 3 待重估，RFC-0029 未涉及（BasePlan §3.7）。
>
> 旧实现（`beatmorph/tokenizer/`、`beatmorph/planner/`、`beatmorph/rag/`、`beatmorph/alignment/`、`io/formats/osu.py` 等 osu!mania 资产）归档到 **`archive/osu-mania` 分支**；主路径不再引用。
>
> ⚠️ 上述 v2.x 计划文件（`02-tokenizer-bpe.md`、`03-planner-density.md`、`05-rag-retrieval.md`、`06-dpo-alignment.md`、`07-decoder-postprocess.md`、`08-data-pipeline.md`、`09-infra-cli-api.md`）已随本次重写从本目录移除；**本索引以 00-08 九个编号为唯一有效集合**。

## 当前阻塞项（跨 plan 提醒）

| 阻塞 | 影响 | 出处 |
| --- | --- | --- |
| ✅ ~~数据合规未裁定（R-2，阻塞级）~~ **已裁定（2026-08-05）：训练可启动**（风险由决策者承担）。**硬约束 =** ① **最终不发布模型权重**（裁决成立的前提，属项目级承诺）② 谱面/音乐**可本地落盘但不得入库** ③ 获取/处理脚本**记录来源与用途** ④ **发布权重前必须重新裁定** | 训练侧阻塞解除（Plan 02 M8 / Plan 07 §3.2 已同步）；人评的原曲音频播放属**再分发**、裁决未逐字覆盖 → Plan 06 §9-9 | BasePlan §4.4、CLAUDE.md 红线 5 附注、RFC-0029 §7-7/§8.3 Q11b |
| ✅ ~~Q15 时间网格未定~~ **已决（2026-08-05）：beat-aligned**，基本格 **1/48 拍**，网格随 BPM 变化；积分测度 `dt → J(τ)dτ`（`J(τ) = dt/dτ` 由 `BPMList` 精确给出） | **数学改写限定在 `field/` 内**；契约层 `PhigrosNote.t` 仍用秒；下游不得各自实现时间换算；须有「场网格 ↔ 秒」往返无损契约测试（多 BPM 段） | RFC-0029 §3.1/§7-8/§8.3 Q15、BasePlan §3.2.4、CLAUDE.md 红线 7；落地见 Plan 03 M12、Plan 04 §9-5 |
| **同刻按键上限数值未查证** | 后处理该项只能统计、不能作红线 | Plan 05 §9-1 |
| **网格桶数 N 未裁定** | 需先做 `N ∈ {64,128,256,512}` 消融与共格碰撞统计 | 单位文档 §7.3、Plan 05 §9-5 |
| ✅ ~~`|Ω|` 的口径~~ **已裁决**（RFC-0029 §8.4 R-a） | 统一为「全 K 线、格元总数」；跨谱汇总报 per-chart NLL 与 NLL/N 两栏 | Plan 00 §3.7、Plan 03 偏离 2 |
| ✅ ~~`--lines` schema~~ **已裁决**：定义为 RPEJSON `judgeLineList` 子集的薄序列化，**复用 Plan 00 已有的 `JudgeLine` 契约**，不新开 RFC（RFC-0029 §8.4 R-d） | Plan 08 可直接实现 | Plan 08 §9-1 |
| ✅ ~~`LegalityReport` 归属~~ **已裁决**：属跨模块类型 → 落 `core/contracts`（RFC-0029 §8.4 R-c） | 影响 Plan 06 合法性指标与 Plan 08 报告 schema | Plan 00 / Plan 05 §9-12 |
| ✅ ~~Hold 期间速度变化口径~~ **已裁决：降级为 warning**（RPE 未确证，不得当红线；RFC-0029 §8.4 R-b） | 不计入"阻止导出"的违规项，只统计报告 | Plan 05 §9-11 |
