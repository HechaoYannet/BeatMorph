# BeatMorph

> **Phigros 谱面端到端自动生成系统** —— 从音频到可玩谱面，自监督学习取代显式标注。
>
> **核心范式**：`音频 → MERT 隐式理解 → 判定线局部系多线强度场 → 掩码补全 → 泊松 NLL → RPEJSON`

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Status: Pre-Alpha](https://img.shields.io/badge/status-pre--alpha-orange.svg)]()

BeatMorph 从原始音频（WAV/MP3）+ 难度（+ 可选判定线事件轨）出发，端到端生成高质量、可玩的 **Phigros 谱面（RPEJSON）**。模型从社区海量自制谱自主学习创作规律，无需人工标注（标注成本 ≈ 0）。

> 📌 **状态**：Pre-Alpha。范式（[RFC-0029](docs/decisions/RFC-0029-phigros-continuous-chart-generation.md)）、奠基文档（v3.0）、事实库已就绪；**核心契约 / 数据流水线 / 强度场 / 生成主干 / 解码与导出五层已落地并通过默认 CI**（608 项测试），**两套 G1–G4 门禁已实跑全绿**；评估、训练栈与消融臂待建（见下方交接件）。

---

## 当前状态与下一步（**跨 session 交接件**）

> 本节读者是**下一个 session 的 agent**，不是历史记录。
> **每次交接必须整节重写，不得追加**（规则见 [AGENTS.md](AGENTS.md) §6）。
> 上次交接：**2026-09-27** ｜ 交接人：主会话（plan 05 解码与导出 M5.1–M5.6）

### 当前状态

自检（默认 CI，无网络 / 无权重 / 无 GPU）：`make lint && make typecheck && make test-fast` → **608 passed**；
两套 G1-G4 门禁都**在真实主干上实跑过**：`uv run pytest tests/unit/field/test_sanity_gates.py tests/unit/generation/test_sanity_gates.py -m slow` → 全绿。

**端到端已通的一条链**：`λ 场 → decoder.decode_field（D1 峰值 / D2 thinning）→ 合法性后处理（逐条留痕）→ io/formats/rpejson 写出 → 读回`。全程无权重、无 GPU、无网络。

**已落地**（plan 00 / 02 / 03、plan 04 主干 M1–M4·M6、plan 05 M5.1–M5.6；逐条状态见各 plan 的状态列）：

- `core/contracts/` ✅ v3.0 契约：`phigros.py`（判定线 / 音符 / 谱面 / 事件轨求值）+ `field.py`（网格与张量形状）+ **`legality.py`**（`Violation`/`Edit`/`LegalityReport`：违约为空 == 可导出）
- `data/` ✅ RPEJSON 独立实现解析器、内容嗅探、`info.yml` 定位、三层质检、Phira 客户端（分页 / Range 预筛 / provenance）、特征缓存、按曲目切分
- `field/` ✅ 网格与**秒↔τ 唯一换算**（torch-free）、目标构建、两条 ∫λ 路径互校、泊松 NLL、G3 闭式基线、共格碰撞统计、可视化
- `generation/` ✅ 掩码补全 Enc-Dec（滑动窗口 + 周期全局层、难度 AdaLN、可变 K、因子化 λ 头）+ 迭代并行解码
- `decoder/` ✅ **本轮**：D1 峰值解码（阈值 = α·λ0、局部极大 + NMS、Hamming 窗长强制奇数）、D2 Ogata thinning（分块保守上界 + 越界时报错而非静默、seed 逐元素可复现）、Hold 枚举配对、合法性后处理（越界**只统计**、`EditKind` **无 clamp 成员**）、`decode_field` / `decode_both_arms`
- `io/formats/rpejson/` ✅ **本轮**：写路径（秒→beat 三元组、`type`/`above` 走契约映射、`speedEvents` 无贝塞尔、**导出门禁**、字节级幂等）
- 夹具 ✅ `tests/fixtures/phigros/`（手工构造，无音频无曲绘）

**接缝**（三条 integration 测试各钉一处跨模块一致性）：`test_time_conversion_seam.py`（格式层 beat↔秒 vs 场层秒↔τ）、`generation/test_train_step.py`（plan 03 的 `build_target` 直喂 generation）、`test_decode_to_rpejson.py`（**本轮**：plan 05 writer ↔ plan 02 reader——标记全等 + `Δt ≤ 一个量化步长` + 字节幂等）。

**本轮发现**（都已变成测试或 RFC 条目，不留在口头）：

1. **契约的父线合成与 A 级证据不一致**：`JudgeLine.pose_at` 只把父线链的平移**相加**，未把子线平移按父线旋转旋转（[units 文档 §2.6](docs/knowledges/phigros-units-and-geometry.md) 的 prpr 源码是 `parent_pos + R(parent_rot)·child_translation`）⇒ 含父线谱面（实测 **26%**）的**跨线几何计数是近似值**。该项只报计数、从不改谱面，故本轮**未擅自改契约**，修正建议见 RFC-0030 §后果-2。
2. **plan 05 §4.2-1 的 τ 边缘强度算式漏了 `J(τ)`**：漏掉它会让 `∫λ_k dτ` 与路径 (a) 相差一个与 BPM 有关的因子（与 POSTMORTEM 同类错误）。按**算式笔误**修正并在 RFC-0030 §6 登记（原文自己写的「同 J(τ)dτ 测度」与「量纲 = 计数/拍」只有加上 J 才成立）。
3. **M5.3 的「采样总数相对 ∫λ 的误差 < 1%」在统计上不成立**：单次计数偏差本身就是 `1/√Λ` 量级（Λ=1e4 时恰为 1%），照字面断言约有 32% 概率误报——改为「多 seed 均值无偏 < 1%」+「单次 ≤ 4σ」两条（RFC-0030 §8）。

**未落地**：plan 04 的消融臂与阶段出口（M5 的 F1 对照、M7–M12）、plan 05 的 M5.7（需已训练模型）；`eval/`（plan 06）、`infra/` + `configs/` 训练栈（plan 07）、`cli`/`api`（plan 08）。

### 下一步

1. **训练栈与真实数据通路（plan 07 + plan 02）**：`generation/` 目前只有模型与门禁，**没有训练循环、没有数据集加载**；解码侧已能消费 λ 场，但**还没有一个真实训练出来的 λ 场**。这两块合起来是「首个真实可玩谱面」的当前瓶颈。
2. **plan 06 评估**：秒域 `DecodedEvent` 与 `LegalityReport` 已就绪，评估层可直接消费；plan 05 的 M5.7（双解码臂 B6 对照）与 plan 04 的 M5/M12 都在等它。
3. **plan 04 的消融臂（M7–M11）**：B4 AR 上界臂、B1 热图 + focal 正式臂、B3 GOCT 配置臂、B5 absorbing 扩散臂。目标函数与 G1–G4 模板已就绪，缺的是模型臂、离散 tokenizer 与各自的门禁实跑；**扩数据规模前必须先跑通 G1–G4**。
4. **全库统计回填两个数字**：共格碰撞率 → `RPE_X_GRID_BINS` 的最终取值（**默认 128 已裁，取值待三条曲线**）；唯一曲目数 / 同曲重复率。

### 未决项（不阻塞第 1-4 步）

| 未决 | 出处 |
|------|------|
| **[RFC-0030](docs/decisions/RFC-0030-decoder-export-contract-ownership.md) 整体待裁定**：解码/导出契约归属 + 六项实现口径（分母策略 / 无 scipy / τ 强度含 J / Hold 配对 / M5.3 口径 / D1 细节）；**实现已按提案落地** | RFC-0030 |
| **契约父线合成**与 A 级证据不一致 ⇒ 含父线谱面的跨线几何计数为近似值（需 contracts-agent） | RFC-0030 §后果-2 |
| **遮盖重标定口径**：`hidden`(1/r，实现默认) vs 文档字面 `observed`(1/(1-r)) vs 不自洽的 `hidden_doc` | plan 04 §9-6 |
| **mask 泄漏口径**：默认「以事件为锚的 token 区块遮盖 + 空格稀释」偏离 §4.2 字面表述 | plan 04 §9-18 |
| plan 04 报告方法学两项：条件学习的「边际盆地」loss 差监控（R-04-5）、G2 合成任务与训练预算 | plan 04 §9-15 / §9-16 |
| 输出头**直连 skip** 是否写进 BasePlan §3.3 的架构描述 | plan 04 §9-17 |
| `N` 的最终取值（默认 128 已裁；仍需碰撞率–N / NLL–N / F1–N 三条曲线） | plan 03 §9-7、RFC-0029 §8.4 R-f |
| 秒↔τ 两个换算点是否合并为一份实现；τ 轴终点口径（`chartTime` / 最后一事件 / 音频时长 / 是否含 `META.offset`） | plan 02 §9-11、plan 03 §9-13 / §9-14 |
| BPM 变更点落在 1/48 拍格内时 `J_j` 的取值（已量化出 `rule` 三档，默认 `left`） | plan 03 §9-16 |
| `RPE_HEIGHT_RATIO` 的出处（D3）；旋转正方向的屏幕含义（D1）；`META.offset` 的符号解释（解码路径写 0） | plan 00 §9-6/§9-8、plan 05 §9-8 |

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

✅ **数据合规已裁定（2026-08-05）**：训练可启动，项目级硬约束为 ① **最终不发布模型权重** ② 谱面/音频可本地落盘但**不得入库** ③ 获取与处理脚本**记录来源与用途** ④ 发布权重前必须重新裁定。风险由决策者承担。详见 [BasePlan §4.4](docs/BasePlan.md)、[RFC-0029 §8.3 Q11b](docs/decisions/RFC-0029-phigros-continuous-chart-generation.md) 与 [phira-dataset-survey.md](docs/knowledges/phira-dataset-survey.md)。

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
| **2 强度场** | ✅ 场模块（双积分路径）→ ✅ 掩码补全主干（G1-G4 全绿）→ ✅ **解码与合法性后处理 + RPEJSON 写路径**（plan 05 M5.1–M5.6）→ ⬜ 训练栈 + 真实数据（plan 07/02）→ ⬜ B1-B6 对照与评估（plan 04 M7–M11 / plan 06）→ ⬜ 首版可玩谱面（M12，联合 plan 08） |
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
