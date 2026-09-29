# BeatMorph

> **Phigros 谱面端到端自动生成系统** —— 从音频到可玩谱面，自监督学习取代显式标注。
>
> **核心范式**：`音频 → MERT 隐式理解 → 判定线局部系多线强度场 → 掩码补全 → 泊松 NLL → RPEJSON`

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Status: Pre-Alpha](https://img.shields.io/badge/status-pre--alpha-orange.svg)]()

BeatMorph 从原始音频（WAV/MP3）+ 难度（+ 可选判定线事件轨）出发，端到端生成高质量、可玩的 **Phigros 谱面（RPEJSON）**。模型从社区海量自制谱自主学习创作规律，无需人工标注（标注成本 ≈ 0）。

> 📌 **状态**：Pre-Alpha。范式（[RFC-0029](docs/decisions/RFC-0029-phigros-continuous-chart-generation.md)）、奠基文档（v3.0）、事实库已就绪；**核心契约 / 数据流水线 / 强度场 / 生成主干 / 解码与导出 / 训练基础设施六层已落地并通过默认 CI**（**1089 项测试**），**真实 Phira 全库已拉到本地（8551 张 RPE 谱面 + 42.8 GB 音频，不入库）**，特征提取 **8551/8551**、训练窗口 **train 634 952 + val 74 889**，**全库窗口预切缓存已建成并终验通过**（train 353.6 GB / val 41.7 GB，逐位一致 500/500 抽查）；**显存墙已解除（[RFC-0032](docs/decisions/RFC-0032-local-layer-own-line-tracks.md)）；[RFC-0037](docs/decisions/RFC-0037-remove-g2-and-per-event-loss.md) 已裁定并落地**（门禁 G2 删除 → **G1/G3/G4** 三道 + val held-out 对照判读红线；训练损失**按事件归一**）；**首跑全量训练已跑完（20 000 步 / exit 0 / 零卡死），但这是一次「零进展」的 run** —— 训练 per-event 损失在 step ~1000 即饱和、`val/ratio` 最优停在 step 2000，且 `val@3000` 单点破线 ⇒ **依 RFC-0037 R1 该 run 的扩规模结论作废**；处置见 [RFC-0038](docs/decisions/RFC-0038-training-saturation-and-budget.md)（**提案，待裁定**）与 plan 07 §9-61；消融臂与评估接线待建**。

---

## 当前状态与下一步（**跨 session 交接件**）

> 本节读者是**下一个 session 的 agent**，不是历史记录。
> **每次交接必须整节重写，不得追加**（规则见 [AGENTS.md](AGENTS.md) §6）。
> 上次交接：**2026-09-30** ｜ 交接人：主会话（第十六轮：**判据侧诊断** —— 目标根本不需要音乐、
> 判据与目标脱节、输入侧封顶。§9-64 的对齐修复**是必要的但不是充分的**：它让音频从「被彻底抛弃」
> 变成载荷项（`cond_audio_zero` −0.075 → +1.033），却**没有**让 `val_ratio` 变好。全链见 plan 07 §9-66）

### 当前状态

自检：`ruff check beatmorph tests` ✅ ｜ `mypy beatmorph` ✅（84 files）｜ `pytest -m "not slow and not gpu and not e2e"` → **1154 passed**（16 deselected）。
⚠️ `ruff check .` 现在会**红**，但红的不是本仓代码：`research/kipphi-rpejson/`（未跟踪、由决策者另行加入）未进 `pyproject.toml` 的 `extend-exclude`。

**★★ 判据侧三条硬读数（本轮新增，直接回答「为什么学不会音乐」）**

1. **目标不需要音乐**：λ=exp(w·x) 的**9 维线性读出**（τ 的 5 个傅里叶项 + 可见场 3 个计数 + 偏置，
   token 内空间分布取训练集边缘）在**同一个 `masked_poisson_loss`** 下拿到 `val_ratio` **0.7669**，
   而 40M 参数模型是 **0.7620** ⇒ **模型的全部增益基本能被「抄可见场密度 + 空间边缘」解释完**。
   （r=0.5 时可见密度几乎是隐藏密度的充分统计量——这是判据本身的性质，不是泄漏。）
2. **判据与目标脱节**：同一个 armH@40000，`val_ratio 0.762` 与**真谱**的 `event F1 @50ms = 0.0022`
   （timing F1 0.192，只对上 12.5% 的音符时刻；x 的 MAE 299 / 半宽 617 = 基本随机）**同时为真**。
   要判「会不会写谱」必须看生成侧读数——本轮把这条读数建起来了（`runs/_eval_generation.py`，走 plan 06 权威实现）。
3. **输入侧封顶**：(窗口,线) **组内** AUC —— 条件事件轨 **0.643**（生成时**有**）、音频（对齐）**0.576**、
   音频（冻结）0.52、可见场 0.481（组内反信息，只告诉密度不告诉「哪一拍」）。模型真谱上的 timing F1@50ms
   0.192 与「只用事件轨的线性上界 ≈0.21」同量级 ⇒ **τ 定位的瓶颈在输入，不在容量或目标函数**。
   ⚠️ 上界那一列（top-N F1）只有 376 正例、对排序头部不稳 ⇒ **登记为未解决**，只引组内 AUC。

**§9-64 的对齐修复（上轮）仍然成立、且本轮给出信息层的定价**：音频是「哪一拍有音符」的**最强单特征**
（组内 0.5600 vs 事件轨 0.5369），而**冻结那张表把它砍到 0.5166**——被修掉的正是信息量最大的线索；
配对 8000 步里对照臂到 8000 步已把音频彻底抛弃（`cond_audio_zero = −0.075`），修复臂仍是 **+1.033**。

**生成制度臂（`model.visible_input=false`，新开关 + 4 项护栏）8k 步未分胜负**：盲臂 `val_ratio` 最好
（0.8326 vs 修复 0.9030 / 对照 0.8586），但依赖几乎全压在事件轨上（`cond_track_zero` 1.039、音频只 0.272），
生成侧更差（23 note / timing F1 0.009）。机制代价已知：盲模型对可见场与遮盖通道**完全无感** ⇒ 8 步迭代退化成 1 步。
**24 000 步盲臂正在跑**（`runs/probe_genregime24k`，本轮交付时未回来）。

**✅ 判定线资格闸门（上轮，已复验）**：e2e 产物 665 note 里 238 个（35.8%）落在 note 时刻透明度 = 0 的线上的问题已修
（在 `pair_events` 之前丢弃），复验 **665 → 411 note、两类违规 0 / 0**，主判定线 336 个未动。

**✅ 仍然成立的既有状态**（细节看对应文档）：数据侧全库就绪（8551 RPE / 窗口 train 634 952 + val 74 889 / 353.6 + 41.7 GB 缓存）；
生产配置 = `head_skip=false` + `audio_align=true`；门禁 **G1/G3/G4**；val = 512 窗分层代表集（指纹 `f9d9f7d1068c`）；训练损失按事件归一。
⚠️ 新增 `model.visible_input` 字段后，**旧 checkpoint 一律拒绝续训**（配置哈希 fail-closed，实测：`model.visible_input: '<缺失>' -> True`）——要续训只能重跑或改指纹口径。

### 下一步（按性价比排序，只留仍然有效的）

1. ★ **把判据接到生成侧，再谈「有效」**：`val_ratio` 与目标脱节（0.762 ↔ event F1 0.002）已经是实测事实。
   本轮已交付读数脚本（`runs/_eval_generation.py <chart.json>`，走 plan 06 的 `evaluate_case`）。
   下一步是把它接成**正式入口**（plan 06 的 `beatmorph-eval` + R3 产物自动评分），并把「哪一臂的产物更好」
   作为判据之一；否则后面所有训练都还在优化一个与目标脱节的数。
2. ★ **回收 24k 盲臂 + 跑一条配对非盲 24k**，判「推理制度训练」：判据用生成侧 timing/event F1 + `cond_audio_zero`，
   **不要**只看 `val_ratio`（盲臂的 val_ratio 更好而生成更差，已经出现一次）。
3. **质疑音频特征本身**：本轮所有探针都用「单帧 + 1 步差分」的 32 维随机投影。要抬 τ 定位（当前 0.576）
   得换更强的读出：多帧上下文 / onset 强度 / 对数梅尔 / 少量可学习前端。这是**输入天花板**层面的工作，
   与主干解耦，可以先用探针评估（不必训练）。
4. **plan 05 M5.7**：D1 阈值标定 + thinning 臂的 Hold 端点策略（R3 产物：64 个 Hold 里 63 个零时长）。
5. **评估入口接线**：形态已定＝独立 `beatmorph-eval`（仍未实现）。

### 未决项（不阻塞「下一步」1–5）

| 未决 | 出处 |
|------|------|
| **top-N F1 上界不稳**（376 正例 + 排序头部）⇒ 需要一个更大的生成侧评估集 | plan 07 §9-66 存疑 1 |
| **「9 维线性打平 40M 模型」只在这一条判据上验证**（换 val 集 / 换 r 未做） | plan 07 §9-66 存疑 2 |
| **音频特征未质疑**（多帧 / onset / 对数梅尔 未测） | plan 07 §9-66 存疑 3 |
| **盲训练的迭代退化**（8 步 → 1 步）未解：真做推理制度训练需要按 schedule 变的遮盖通道（数据侧改动） | plan 07 §9-66 存疑 4 |
| **`seconds_position` 修好后是否值得开**（cross-attn 的两条轴） | plan 07 §9-64 存疑 3 |
| **`cond_audio_perm ≈ 0` 的机制**（信息不足 vs 对齐学不出来） | plan 07 §9-64 存疑 4 |
| **跨 `data.workers` 的读数不可比**（4000 步就分叉）⇒ 配对实验必须固定这一项 | plan 07 §9-64 存疑 5 |
| **装饰线模型侧旁路尚未做**（只做了导出侧闸门） | plan 07 §9-65 / RFC-0032 |
| `val_windows=512` 抽样精度 / `val_every=10000` A/B | plan 07 §9-63 存疑 1–2 |
| RFC-0038 与 RFC-0039 重叠部分的处置 | RFC-0038 / RFC-0039 §5 |
| `val/nll_shuffled_delta` margin / `grad_clip_norm` / G1 下限 0.05 的 per-event 语义 | RFC-0037 §6 |
| 批 K 上限 / RFC-0035 §1 的 91.3% 未更正 / `r==0` 遮盖退化 / 预算按步还是按 epoch | plan 07 §9-59 / §9-52 / §9-43 / RFC-0033 |
| **评估 / 可玩性**：NLL 好 ≠ 谱面可玩（现在有数了：event F1 0.002） | plan 07 §9-47 H / §9-66 ③ |

**一次性探针（⚠️ `scripts/local_*` 与 `runs/_*` 均不入库，清理即丢）**：本轮读数脚本 = `runs/_probe_gain_ladder.py`（收益阶梯，走真损失）、`runs/_probe_localization.py`（定位组内 AUC）、`runs/_probe_ceiling.py`（输入上界，**口径有保留**）、`runs/_eval_generation.py <chart.json>`（生成侧 F1，走 plan 06）。上轮：`runs/_diag_align_audit.py`、`runs/_diag_audio_paths.py`（`--ckpt-path`）、`runs/_diag_align_path.py`、`runs/_probe_audio_info2.py` + `runs/_probe_audio_within.py`、`runs/_audit_e2e_lines.py`、`runs/_verify_r2r3.py`（`--mode val|e2e --ckpt|--ckpt-path [--blind]`）。装配件 `runs/_diag_common.py`。⚠️ `scripts/local_chart_mem.py` **坏了**（`deep_bytes` 漏 pydantic `__slots__`，偏低 1.85×）。⚠️ 长跑重定向探针的输出**只用 ASCII `=>`**（`⇒` 在 GBK 重定向下崩溃）。
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
| **2 强度场** | ✅ 场模块（双积分路径）→ ✅ 掩码补全主干（G1-G4 全绿）→ ✅ 解码与合法性后处理 + RPEJSON 写路径（plan 05 M5.1–M5.6）→ ✅ **训练栈 + 真实数据通路**（plan 07 M7.1–M7.8、plan 02 M11、`beatmorph-train --gates`）→ 🟡 评估指标库已落地（plan 06 M6.1–M6.6），**真实数据拉取**与 B1-B6 对照待建（plan 04 M7–M11）→ ⬜ 首版可玩谱面（M12，联合 plan 08） |
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
