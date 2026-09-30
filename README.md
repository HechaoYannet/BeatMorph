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
> 上次交接：**2026-09-30** ｜ 交接人：主会话（第二十一轮：**把「学不会音乐」定位到了具体一级**——
> 可见场里 τ 信息几乎是完备的（线性读出组内 AUC **0.9687**），而**解码器输出只剩 0.4983**（随机），
> 头输出 0.4970 ⇒ 崩塌发生在**「嵌入 + 解码器」这一段**，不是目标函数、不是训练分布、也不是输入没信息。
> 上一轮的取批修法已实测有效（val_ratio 首次随训练变好、8k 最好读数 0.7324），长跑在验证「是否持续改善」。
> 全链见 plan 07 §9-69/§9-70/§9-71）

### 当前状态

自检：`ruff check beatmorph tests` ✅ ｜ `mypy beatmorph` ✅（84 files）｜ `pytest -m "not slow and not gpu and not e2e"` → **1167 passed**。
⚠️ `ruff check .` 会因**未跟踪的** `research/kipphi-rpejson/`（决策者另行加入）报错，与本仓代码无关。

**★★★ 本轮头条：τ 信息在输入里，到解码器输出就没了**

| 一级（`runs/_probe_stage_auc.py`，同一目标 / 同一批窗口 / 同一岭回归口径） | 维度 | 组内 τ AUC |
|---|---|---|
| L0 输入：可见场多尺度 τ 剖面 | 11 | 0.5583 |
| **L1 token 输入特征（可见场 + 遮盖通道）** | 2560 | **0.9687** |
| **L2 解码器输出 token** | 256 | **0.4983** |
| **L3 模型头输出（每 token λ 质量）** | 1 | **0.4970** |

⇒ **输入里几乎完备（0.97）→ 解码器输出随机（0.498）→ 头输出随机（0.497）**。
至此三个候选已被逐个排除：目标函数（§9-69：τ 值 42.46% 的常数基线）、训练分布（§9-70：修好后 Δ 仍 −2e-5）、
「可见场没信息」（本轮 0.97）。**剩下的只有「可见场 → 表示」这一段。**
⚠️ 口径：L1 是**原始输入特征**、不是模型表示（读出它等于问「输入里有没有」）；L2 才是模型表示。

**上一轮的修法（已实测有效）**：`data.plan_window_shuffle` 把训练分布从 **61.5% 空窗 / 4.40 事件每窗**
拉回 **12.2% / 14.26**（全库 10.5% / 13.46），val_ratio **0.7857 → 0.7324（首次随训练变好）**、
质量 1.147 → 0.523（旧：0.955→0.509 / 0.845→0.055）、覆盖率 87.7% 不变。**8k 预算下全项目最好读数**
（旧最好 0.8137）。**24k 长跑（`probe_shuffle24k`）正在验证「是否持续改善」。**

**仍然成立**：数据侧全库就绪（8551 RPE / train 634 952 + val 74 889 窗 / 353.6+41.7 GB 缓存）；门禁 G1/G3/G4；
val 512 窗分层集（`f9d9f7d1068c`）；损失按事件归一；τ→秒表按窗口求值与判定线闸门（§9-64/§9-65）。
⚠️ 旧的四条无-skip 臂 + 三条 skip 臂（0.7636→1.0029 / 0.9734→1.1316 / →1.0830，三条破线）**都训练在偏置
切片上 ⇒ 作为「模型对比」全部作废**；`data.plan_window_shuffle` 是**语义字段**，旧 checkpoint 一律拒绝续训。

### 下一步（按性价比排序，只留仍然有效的）

1. ★ **补 L1.5 = 过完 `model.embedding`（+ 音频注入）之后的 token**（**只需要一次前向**）——
   它是唯一还没量的一级，直接给出「崩塌在**输入投影/嵌入**还是在**解码器栈**」：
   L1.5 ≈ 0.50 ⇒ 查 1280→256 的 `nn.Linear` 与 LayerNorm（可见场是稀疏计数，每 token 约 11/1280 个非零，
   投影后范数远小于位置编码/线嵌入）；L1.5 > 0.9 ⇒ 查局部层 band mask 与全局层 K*T 注意力是否把 τ 平均掉。
2. ★ **读 `probe_shuffle24k` 的 val 序列**：判据 = val_ratio **持续下降**（8k 时 0.7324；旧六条臂在 8k
   之后全部反弹到 0.83–1.13）。若持续下降 ⇒ 修法下「多训有用」，应重跑关键臂并重估预算。
3. ★ **修法转正**：`data.plan_window_shuffle` 默认改 `true` 并落 RFC-0034 §9 的决策记录（语义变更）。
4. **判据解耦**：先报 `val_pred_over_true` 与「固定质量后的 NLL」，再谈 arm 对比与 `best.pt`。
5. **验证 onset 前端**（+0.03 AUC，唯一抬动天花板的输入侧改动）：小切片训一臂再判值不值全库重提。

### 未决项（不阻塞「下一步」1–5）

| 未决 | 出处 |
|------|------|
| **崩塌在嵌入还是解码器**（L1.5 未量）；修法只把前缀变无偏，epoch 内池级相关性仍在 | §9-71①③、§9-70④ |
| **可见场抄写基线 0.744 其实是「均匀底」在起作用**（每 token 仅 11/1280 个非零格） | §9-69⑦ |
| **onset 前端值不值得全库重提**（+0.03 AUC）；**音频依赖为何在每个臂上都衰减**（4 臂复现） | §9-67 ⑤⑥ |
| 「9 维线性打平 40M 模型」只在这一条判据上验证；§9-66 的 top-N F1 上界列；盲训练的迭代退化 | §9-66 存疑 1/2/4 |
| `seconds_position` 是否值得开；装饰线**模型侧**旁路未做；`val_windows` 抽样精度 / `val_every` A/B | §9-64 ③、§9-65、§9-63 |
| RFC-0038/0039 重叠处置；`nll_shuffled_delta` margin；`grad_clip_norm`；G1 下限 0.05 的 per-event 语义 | RFC-0038/0039 §5 |
| 批 K 上限 / RFC-0035 §1 的 91.3% 未更正 / `r==0` 遮盖退化 / 预算按步还是按 epoch | §9-59 / §9-52 / §9-43 |

**一次性探针（⚠️ `scripts/local_*` 与 `runs/_*` 均不入库，清理即丢）**：本轮 = `_probe_stage_auc.py`（**逐级
τ AUC**）、`_probe_tau_variance.py`（逐维 τ 方差分解，**量具已修但尚未重跑**）；上轮 = `_probe_composition.py`、
`_probe_plan_prefix.py`、`_probe_plan_rounds.py`、`_probe_shuffle_check.py`、`_probe_tau_ceiling.py`、`_probe_mass.py`。
⚠️ **口径陷阱（第四次）**：`val.batches()` 前 40 批是稀疏层；跨 checkpoint 比较**必须固定同一批**。
**一次性探针（⚠️ `scripts/local_*` 与 `runs/_*` 均不入库，清理即丢）**：本轮 = `_probe_composition.py`（训练 vs
语料分布）、`_probe_plan_prefix.py`、`_probe_plan_rounds.py`（coverage_at 看轮次）、`_probe_shuffle_check.py`、
`_probe_tau_ceiling.py`（11 维岭回归天花板）、`_probe_mass.py`；
历史见 plan 07 §9-64/§9-66/§9-67/§9-69。
⚠️ **口径陷阱（第四次）**：`val.batches()` 前 40 批是稀疏层；跨 checkpoint 比较**必须固定同一批**，
绝不能把子集读数与全 val 混着说。

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
