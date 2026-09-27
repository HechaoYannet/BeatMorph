# 决策记录（RFC / ADR）

本目录记录 BeatMorph 的架构决策与技术选型变更。任何对
[`BasePlan.md`](../BasePlan.md) 的实质性偏离或新决策，都须在此以 RFC 形式登记。

## 何时写 RFC

- 偏离奠基文档已确定的技术选型（如换掉 MERT、改用对比学习等）。
- 引入新的跨模块约定（如数据契约的破坏性变更）。
- 解决某个 plan「9. 开放问题」中悬而未决的项。

## RFC 编号约定

- 格式 `RFC-XXXX`，四位编号，**全局递增、唯一、不回收**。
- 文件名 `RFC-XXXX-<短标题>.md`。
- 状态：`提案` → `讨论中` → `采纳` / `驳回` / `废弃`。
- 新增 RFC 前先查下表确认下一个可用编号，禁止复用既有编号。

## RFC 模板

```markdown
# RFC-XXXX — <标题>

- 状态：提案 ｜ 提出日期：YYYY-MM-DD ｜ 决定日期：YYYY-MM-DD
- 提出者：<name>
- 影响模块：<plan 编号 / 代码路径>

## 背景
（为何需要这个决策，触发来源）

## 提议
（具体方案）

## 备选方案
（考虑过的其它选择及放弃理由）

## 后果
（采纳后的影响、对既有 plan 的修订）

## 关联
（奠基文档章节、相关 RFC）
```

## RFC 总表

下表为当前已分配编号（均来自各 plan「9. 开放问题」/偏离声明，状态为提案占位）。
正式立案时为本目录新增 `RFC-XXXX-*.md` 文件并更新本表状态。

| RFC | 主题 | 来源 plan | 状态 |
|-----|------|-----------|------|
| RFC-0001 | Note 时间单位（秒 vs 毫秒） | 00 | [采纳](RFC-0001-note-time-unit.md)（秒） |
| RFC-0002 | 长音频 MERT 滑窗策略 | 01 | 提案 |
| RFC-0003 | Adapter 选型 LoRA vs MLP | 01 | [采纳](RFC-0003-adapter-lora-vs-mlp.md)（LoRA） |
| RFC-0004 | Tokenizer 起步数据量 50K vs 100万 | 02（08 依赖） | 提案 |
| RFC-0005 | 变速曲 `bpm` 字段扩展为时间点 | 00/02/03（08 依赖） | [采纳](RFC-0005-bpm-timepoints.md) |
| RFC-0006 | `sections_type` 是否作分类目标 | 03 | 提案 |
| RFC-0007 | AR 隐层 dim=768 | 04 | **搁置**（RFC-0029 后 AR 退出主线，仅存为消融臂 B4；且 dim 应为 1024 而非 768） |
| RFC-0008 | 超长曲滑窗段落衔接 | 04 | 提案 |
| RFC-0009 | Flow Matching 蒸馏对齐目标 | 04 | 提案 |
| RFC-0010 | RAG 注入默认 Prefix vs Cross-Attention KV | 05 | 提案（暂定 Prefix） |
| RFC-0011 | 密度曲线检索维度 `density_dim=64` 与归一化 | 05 | 提案 |
| RFC-0012 | 流派/谱师元数据缺失兜底过滤 | 05 | 提案 |
| RFC-0013 | DPO 完整目标（含参考项）vs reference-free | 06 | 提案（暂定完整目标） |
| RFC-0014 | DPO β 默认值 0.1 | 06 | 提案 |
| RFC-0015 | DPO 用 LoRA 微调、ref=基座本身 | 06 | 提案 |
| RFC-0016 | 高分段评分稀少时分层采样 + PlayCount 权重 | 06 | 提案 |
| RFC-0017 | 4K 左右手 lane 分配（左{0,1}/右{2,3}） | 07 | 提案 |
| RFC-0018 | `apply` 越界 Note 钳位 vs 删除取舍 | 07 | 提案 |
| RFC-0019 | `.ma2`(maimai) 导出预留时机（Phase2 壳 vs Phase4） | 07 | 提案 |
| RFC-0020 | 解析产出统一到 `Chart` IR（vs NoteEvent[]） | 08 | 提案（暂定 Chart） |
| RFC-0021 | `section_bars` 默认 4 是否随曲风/拍号动态 | 08 | 提案（暂定 4） |
| RFC-0022 | 日志 JSON 化强制范围 | 09 | 提案 |
| RFC-0023 | 检查点本地 vs 云（OSS/S3）存储 | 09 | 提案 |
| RFC-0024 | sayobot 数据源与 §4.3 质量过滤激活 | 08 | [采纳](RFC-0024-sayobot-data-source.md) |
| RFC-0025 | R-6 细化：多 K mania 数据诚实标记 + 训练层 4K 过滤 | 00/07/08 | [采纳](RFC-0025-multikey-data-r6.md) |
| RFC-0026 | 小节边界相位对齐（修复栅格化系统性错位） | 02/03/08 | [采纳](RFC-0026-bar-boundary-phase-alignment.md) |
| RFC-0027 | VQ-VAE encoder：栅格 1D-CNN → note 集合 set-transformer（评估） | 02 | 搁置（RFC-0028 采纳后作废，VQ 范式退出） |
| RFC-0028 | 修宪议案：tokenizer 范式 VQ-VAE 小节粒度 → BPE/event（范式级首选） | 02/04/07（BasePlan §3.2） | [采纳](RFC-0028-bpe-event-tokenizer-constitutional-amendment.md)（BPE/event tokenizer，2026-08-04） |
| RFC-0029 | 修宪议案：目标转向 Phigros + 生成范式改为「判定线局部系**多线**标记点过程 + 掩码补全」 | 全量（BasePlan §1/§2/§3/§4/§7） | [采纳](RFC-0029-phigros-continuous-chart-generation.md)（2026-08-05；Q1/Q2/Q4/Q6/Q7/Q8 已批复） |
| RFC-0030 | 解码/导出契约归属（`legality.py`）与 plan 05 的六项实现口径（分母策略 / 无 scipy / τ 边缘强度的 J 因子 / Hold 配对 / M5.3 口径细化 / D1 两个细节） | 00/05/06/08 | [**采纳**](RFC-0030-decoder-export-contract-ownership.md)（2026-09-27；实现已落地） |
| RFC-0032 | 局部层的事件轨条件注入改为「本线轨」（实测：旧写法让每条线 attend 全部 K 条线的轨，显存 ∝K²、K 上限仅 ≈31；改为本线后 ∝K，K=128 亦可训） | 04/07 | [**采纳**](RFC-0032-local-layer-own-line-tracks.md)（2026-09-27；实现已落地） |
| RFC-0031 | τ 轴终点口径（`T` 取自 `chartTime` / 最后一事件 / 音频时长）：实测 `chartTime` 在 52% 语料上虚高（中位 52.8×），裁定 **`min(谱面口径, 音频时长)`** | 00/02/03/04/07 | [**采纳**](RFC-0031-tau-axis-endpoint.md)（2026-09-27；实现已落地） |
| RFC-0033 | 采样器覆盖率缺陷（**1000 步后饱和在 993 个窗口 / 674 张谱面**，根因是全局共享游标遇长度 1 的桶被清零）与 epoch 定义（每桶独立游标 + 桶内种子化洗牌 + 按剩余窗口加权选桶；新增覆盖率标量） | 02/07 | [**采纳**](RFC-0033-sampler-coverage-and-epoch.md)（2026-09-27；实现已落地） |

> ⚠️ **RFC-0029 采纳后**：RFC-0006~0023 中与「4K VSRG / VQ-VAE / BPE-event tokenizer / planner 密度规划 / osu! 数据源」相关的开放项**随之失效**，待按 Phigros 范式重写；RFC-0001（Note 时间单位=秒）、0005（变速 bpm 时间点）、0026（小节相位对齐）的**思想仍适用**，但载体契约需重设。

> 下一个可用编号：**RFC-0034**。
