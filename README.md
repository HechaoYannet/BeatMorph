# BeatMorph

> **Phigros 谱面端到端自动生成系统** —— 从音频到可玩谱面，自监督学习取代显式标注。
>
> **核心范式**：`音频 → MERT 隐式理解 → 判定线局部系多线强度场 → 掩码补全 → 泊松 NLL → RPEJSON`

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Status: Pre-Alpha](https://img.shields.io/badge/status-pre--alpha-orange.svg)]()

BeatMorph 从原始音频（WAV/MP3）+ 难度（+ 可选判定线事件轨）出发，端到端生成高质量、可玩的 **Phigros 谱面（RPEJSON）**。模型从社区海量自制谱自主学习创作规律，无需人工标注（标注成本 ≈ 0）。

> 📌 **状态**：Pre-Alpha。范式（[RFC-0029](docs/decisions/RFC-0029-phigros-continuous-chart-generation.md)）、奠基文档（v3.0）、格式/单位/数据/文献事实库已就绪；**代码尚未迁移**（当前仓库仍为 osu!mania 版）。

---

## 当前状态与下一步（**跨 session 交接件**）

> 本节读者是**下一个 session 的 agent**，不是历史记录。
> **每次交接必须整节重写，不得追加**（规则见 [AGENTS.md](AGENTS.md) §6）。
> 上次交接：**2026-08-05** ｜ 交接人：主会话

### 当前状态

**已就绪**

- **范式**：[RFC-0029](docs/decisions/RFC-0029-phigros-continuous-chart-generation.md) 已采纳（Phigros + 判定线局部系多线标记点过程 + 掩码补全 + 非齐次泊松 NLL）；beat-aligned 时间网格与数据合规两项裁决已落 RFC/BasePlan/CLAUDE。
- **文档库**：[BasePlan v3.0](docs/BasePlan.md)（397 行）、[CLAUDE.md v3.0](CLAUDE.md)、[plans 00-08](docs/plans/README.md)（2538 行）、[知识库 4 份](docs/knowledges/)（格式 / 单位几何 / 数据集 / 文献）、[POSTMORTEM](docs/POSTMORTEM-2026-08-05-frame-rate-misalignment.md)。
- **门禁**：`beatmorph/infra/sanity.py`（G1-G4，纯 Python，无 torch 依赖）；帧率契约测试进默认 CI。
- **仓库**：`main` 已快进至 `67c7299`；osu!mania 实现在 **`archive/osu-mania`** 分支完整保留；主路径已删除 retired 模块。

**代码实况（重要：不要被文档的完备度误导）**

| 区域 | 实况 |
|------|------|
| `beatmorph/core/contracts/` | **仍是 v2.x**（`Note/Chart/Section/EventToken` + 常量）——**待重写为 Phigros 契约** |
| `beatmorph/audio/encoder/mert.py` | ✅ 可用（帧率已改为派生量 75Hz） |
| `beatmorph/infra/sanity.py` | ✅ 可用（G1-G4） |
| `beatmorph/field/`、`eval/`、`io/formats/rpejson/` | ⬜ **不存在，待建** |
| `beatmorph/generation/`、`decoder/` | 空包壳 |
| `configs/` | 仅剩 `model/mert.yaml`（已修正） |
| 测试 | 仅 3 份存活（契约 / 门禁 / 帧率契约）→ `36 passed, 1 skipped` |

### 下一步

1. **写 Phigros 基座契约**（[plan 00](docs/plans/00-core-contracts.md)，393→400 行，8 条里程碑）——这是后面 8 份 plan 的共同依赖，必须先做。
   - 首要：把 v2.x 的 `core/contracts/` 换成 `JudgeLine` / `PhigrosNote` / `PhigrosChart` / `ChartFieldSpec` / `ChartField` / `ChartTargetField`（形状 `(B,K,T,X,S,C)`，τ 时间轴，不变量 I1–I13）。
2. 按 [plan 02](docs/plans/02-data-pipeline.md) 建 RPEJSON 解析器（**独立实现**，只读 prpr/phichain 行为规范；夹具 `chart/1000` + `chart/7039`）。
3. 按 [plan 03](docs/plans/03-field.md) 建 `field/`（两条积分路径 + 一致性门禁 + τ↔秒往返测试）。
4. 数据获取按 plan 02 执行（合规已裁定；**脚本须记录来源与用途**）。

### 未决项（不阻塞第 1–3 步）

| 未决 | 出处 |
|------|------|
| 网格桶数 N 的最终值（需 N 消融 + 共格碰撞统计） | RFC §8.4 R-f、plan 03 M6/M7 |
| RPE 同刻按键上限数值（未查证 → 只统计不作红线） | plan 05 §9-1 |
| 旋转正方向的屏幕含义（存疑 D1） | plan 00 §9-8、单位文档 §9 |
| 官谱侧 `1X` 75 vs 75.94、v1 y 分母 520/530 冲突 | 单位文档 §9 |
| `bpmfactor` 参考实现未实现 | 单位文档 §9 |

---

## 为什么是 Phigros

| 维度 | 4K 下落式（osu!mania / DDR） | **Phigros** |
|------|------------------------------|------------|
| 空间 | 4 条静止离散轨道 | **判定线局部系 2 维**：`positionX` 连续 + `side`（哪一侧） |
| 屏幕方向 | 恒定自上而下 | **任意方向**——由判定线自身的 move/rotate 事件轨决定 |
| 舞台对象 | 无 | **判定线是一等公民**：多线（实测中位 **30** 条）、带完整事件轨的动画对象 |
| 音符类型 | TAP / HOLD / … | Tap / Drag / Hold / Flick |

**一条必须记住的事实**：STRUM 实测人类谱面事件**只有 89% 落在音频 onset ±100ms 内**。所以谱面生成**不是转录，而是创造性诠释**——那 11% 正是"创作"发生的地方。

## 系统架构

```
输入：① 音频(WAV/MP3)  ② 难度  ③ 判定线事件轨（可选，条件输入）
   │
   ▼
Stage 0  音频理解    MERT-v1-330M（冻结，24kHz）+ LoRA Adapter
         → audio_emb [B, T, 1024]，帧率 = 24000/320 = 75Hz（**派生量**）
   ▼
Stage 1  强度场生成   掩码补全 Encoder-Decoder（迭代并行解码）
         → λ_k(t, positionX, side, type)，k = 1..K 条判定线
   ▼
Stage 2  解码与导出   强度场 → 离散事件（Ogata thinning / find_peaks）
         → 合法性后处理（含**跨线几何冲突**）→ RPEJSON
```

**训练目标**：非齐次泊松 NLL —— 事件处强度必须高，全域积分必须低。note 分配到哪条判定线**不需要额外分类损失**，它是 K 个强度场竞争的自然结果（实测最忙的一条线独占中位 **73%** 的 note，任何"均匀 softmax 分配"假设都是错的）。

完整技术选型见 **[docs/BasePlan.md](docs/BasePlan.md)**；范式论证见 **[RFC-0029](docs/decisions/RFC-0029-phigros-continuous-chart-generation.md)**。

## 仓库结构

```
beatmorph/
├── audio/encoder/        Stage 0  MERT-v1-330M + LoRA Adapter
├── data/                 Phira 谱面获取 + RPEJSON 解析 + 质检 + 特征离线提取
├── core/contracts/       ★ 跨模块契约：JudgeLine / PhigrosNote / PhigrosChart / ChartField
├── field/                强度场：网格化、目标构建、双路径积分、可视化
├── generation/           掩码补全 Encoder-Decoder（主选）
├── decoder/              强度场 → 离散谱面 + 合法性与可玩性后处理
├── io/formats/rpejson/   RPEJSON 读写
├── eval/                 事件级 F1 / 校准 / 跨线合法性 / 人评协议
├── infra/                训练栈 + 健全性门禁 G1-G4
└── cli/  api/            命令行与服务接口
tests/   unit / integration / e2e / fixtures
configs/ Hydra 配置
docs/    BasePlan + plans + decisions + knowledges
```

详见 [docs/CODE_STRUCTURE.md](docs/CODE_STRUCTURE.md)（含「现状 vs 目标」对照）。

## 快速开始

需 Python 3.11 与 [uv](https://docs.astral.sh/uv/)。

```bash
git clone https://github.com/HechaoYann/BeatMorph.git BeatMorph && cd BeatMorph
uv sync --group dev
uv run pre-commit install
make test-fast
```

自检帧率契约（**应输出 5s → 374 帧 = 75Hz**）：

```bash
python scripts/verify_mert_frame_rate.py
```

## 数据与训练

⚠️ **数据合规是当前阻塞项**：唯一可用的万级数据源（Phira 官方 API，实测 9649 张）其 ToU **未授予机器学习训练权利**，且音频随谱 100% 捆绑分发。**裁定前不得开始训练**。详见 [BasePlan §4.4](docs/BasePlan.md) 与 [phira-dataset-survey.md](docs/knowledges/phira-dataset-survey.md)。

流程与操作见 **[docs/TRAINING.md](docs/TRAINING.md)**。

> **硬性纪律**：任何新模型/新范式在**扩大数据规模之前**必须通过 **G1-G4** 四道健全性门禁（单 batch 过拟合 / 打乱标签对照 / 常数基线 / 契约断言）。来历见 [POSTMORTEM-2026-08-05](docs/POSTMORTEM-2026-08-05-frame-rate-misalignment.md)。

## 技术选型一览

| 模块 | 决策 | 理由 |
|------|------|------|
| 音频编码 | **MERT-v1-330M + LoRA** | 音乐专用、轻量、层次表征好 |
| 谱面表示 | **判定线局部系多线强度场** | 局部系是原生存储系；无量化损失；多线 v1 即支持 |
| 生成主干 | **掩码补全 Encoder-Decoder** | 并行 + 双向上下文；事件间强共现 |
| 目标函数 | **非齐次泊松 NLL** | 稀疏目标下不塌陷；音符数量由积分隐式决定 |
| 解码 | **Ogata thinning** | 从强度场直接采样的原则性做法 |
| 判定线运动 | **条件化输入**（v1） | 难度与观感大量来自线运动；联合生成留待后续 |
| 全局规划层 | **取消** | 段级均值池化是结构性信息瓶颈，且不需要 |

## 实施路线

| Phase | 目标 |
|-------|------|
| **1 地基** | 数据合规裁定 → 全库预筛 → RPEJSON 解析器 → 契约断言 → MERT 特征 |
| **2 强度场** | field 模块（双积分路径）→ 掩码补全主干 → **G1-G4 全绿** → B1-B6 对照 → 首版可玩谱面 |
| **3 对齐与产品化** | DPO / 风格检索（待重估）→ 推理加速 → API / Demo |

详见 [docs/BasePlan.md §7](docs/BasePlan.md)。

## 文档

- [docs/BasePlan.md](docs/BasePlan.md) — **技术奠基（最高权威）**
- [CLAUDE.md](CLAUDE.md) — AI 协作宪法（红线、模块拓扑、开发流程）
- [docs/decisions/](docs/decisions/README.md) — RFC 决策记录（**RFC-0029 为当前范式权威**）
- [docs/knowledges/](docs/knowledges/) — 格式 / 单位几何 / 数据集 / 文献 事实库（**含 A 级源码证据与显式存疑清单**）
- [docs/plans/](docs/plans/README.md) — 各模块实施计划（00-08）
- [docs/TRAINING.md](docs/TRAINING.md) — 训练操作手册
- [docs/POSTMORTEM-2026-08-05-frame-rate-misalignment.md](docs/POSTMORTEM-2026-08-05-frame-rate-misalignment.md) — 帧率事件根因与门禁由来
- [AGENTS.md](AGENTS.md) — 子 agent 协作约定

## 协作

- 默认分支 `main`；特性分支 `feat/<module>-<topic>`。
- 提交遵循 Conventional Commits（pre-commit 强制）。
- **偏离 BasePlan 的技术变更须先开 RFC**；范式级变更须修宪（改 CLAUDE.md 红线）。
- 换行符统一 LF；模型权重 / 音频 / 数据集不入库（走 LFS 或外部存储）。

## 许可证

Apache-2.0 © 2026 BeatMorph Team。

> 训练数据仅用于学术/研究目的；生成系统输出不包含原音频拷贝。**数据来源的合规性仍在裁定中**（见上文）。