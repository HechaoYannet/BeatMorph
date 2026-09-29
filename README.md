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
> 上次交接：**2026-09-30** ｜ 交接人：主会话（第十五轮：找到「模型学不会按音乐出谱」的根因并修掉——
> 修法让音频**真的被用上**（置零代价 −0.075 → **+1.033** nats/line），但 `val_ratio` **没有变好**，
> 故按决策者指令**未开 RFC**；同轮修掉 e2e 的「note 落在不可见 / 装饰判定线上」。
> 全链见 plan 07 §9-64 / §9-65）

### 当前状态

自检（默认 CI，无网络 / 无权重 / 无 GPU）：`uv run ruff check . && uv run mypy beatmorph && uv run pytest -m "not slow and not gpu and not e2e"` → **1150 passed**（16 deselected）。⚠️ 提交时 `pre-commit` 的 `ruff-format` 会重排若干**旧文件**并使首次 commit 失败——`git add` 后重提交即可，**不要为此做全仓格式化**。

**★ 根因（已实测）：τ→秒表被冻结 ⇒ 音频从来没在「对的时刻」进过模型**

| 事实 | 数值 |
|---|---|
| `_field_seconds` 的缓存键只有 `(t_bins, device)`，而 `t_bins` 是全库常数（`data.t_window`） | 表在**第一次前向**被冻结成**第一个窗口**的 BPM（**301.5**） |
| val 512 窗的 `bpm_eff`（69–480） | 中位 **162** ⇒ **100%** 的窗口错位 |
| 冻结下标只覆盖窗口**前段**的窗口 | **98.2%**（前 4 个 val 窗只取到前 50% 的音频） |
| 修复前该通路的因果效应（置零 / 置换 / 换**正确对齐** / ×4） | **全部 ≤ 1.1e-4 nats/line**（= 0） |
| 信息层「哪一拍有音符」（仅音频、**组内 AUC**） | 对齐 **0.534** → 冻结 **0.505**（随机 0.498） |

**解法**：`_field_seconds` 删掉缓存、逐次求值（= 每个窗口用自己的 BPM）。配对 8000 步（同 seed /
同数据 / 同 `data.workers=0`，唯一差别是这处代码；`runs/probe_alignctl2` vs `probe_alignfix2`）：

| 读数 | 对照 @4000 / @8000 | **修复 @4000 / @8000** |
|---|---|---|
| `cond_audio_zero_delta` | +0.548 / **−0.075** | **+1.547 / +1.033** |
| `cond_audio_perm_delta` | −0.003 / +0.001 | −0.003 / +0.004 |
| `val_ratio` | 0.774 / 0.859 | 0.814 / 0.903 |
| `val_pred_over_true` | 0.627 / 0.423 | 0.495 / 0.212 |

⇒ **「音频被用上」已验证**（对照臂到 8000 步已把音频**彻底抛弃**：置零**不增加**损失）；
**「谱面变好」未验证**（`val_ratio` 两处都略差，而单 run 步间摆幅本身有 0.08 ⇒ 未达可判读）。
`cond_audio_perm ≈ 0` ⇒ 用上的仍是**内容/存在**，不是**时间对齐**。

**⚠️ 两条量具更正（引用旧结论前必读）**

1. **「训练损失 10 → 0.015」是空窗步的读数**（`batch_events=0`，`loss_empty` = 纯积分项），不是拟合读数。
   同口径 `loss_nonempty_per_event` 分块中位：baseline **9.9–10.4**、armH(40k) **8.2–8.9** ⇒
   `head_skip=false` 的训练侧真实收益约 **20%**，不是 660×。`val_ratio` 那条（0.9987→0.7620，配对 t=+3.03）不受影响。
2. **探针的全局 AUC 会被「窗/线密度」同义反复主导**：`τ+可见场` 全局 **0.949**、**组内只有 0.597**
   ⇒ 凡问「哪一拍有音符」，一律报**组内 AUC**（按 (窗口, 线) 分组）。

**✅ 判定线资格闸门（决策者实测报告，已修并复验）**：e2e 产物 665 个 note 里 **238 个（35.8%）** 落在 note 时刻
不透明度 = 0 的线上、**149 个（22.4%）** 落在装饰 / 表演线上。修法 = 在 `pair_events` **之前**按「该线该时刻可见
+ 属于条件谱面承载 note 的线集合」丢弃场事件（`decoder.events.filter_field_events_by_line` + `infra/e2e.py::LineFilter`），
记账进 `meta.json` 的 `line_filter_*`。同一 checkpoint 复验：**665 → 411 note**，两类违规 **0 / 0**，主判定线 336 个未动。
⚠️ 合成模板**没有 alpha 轨** ⇒ 那是「没给信息」不是「不可见」，闸门必须放行（否则无条件生成路径会被一次清空）。

**✅ 仍然成立的既有状态**（需要细节时看对应文档，这里不重复）：数据侧全库就绪（8551 RPE / 8084 唯一音频 /
窗口 **train 634 952 + val 74 889** / 353.6 + 41.7 GB 预切缓存）；生产配置 = `head_skip=false` + `audio_align=true`
（[RFC-0039](docs/decisions/RFC-0039-training-baseline-val-and-e2e-artifacts.md) R1）；门禁 **G1/G3/G4**（RFC-0037 删 G2）；
val = 512 窗分层代表集（指纹 `f9d9f7d1068c`，空窗 10.8%）；训练损失按事件归一；窗口缓存使 `data_share` 中位 2.9%。

### 下一步（按性价比排序，只留仍然有效的）

1. ★ **音频问题的下一层：把「时间对齐」接上，或证明它接不上**
   修复只接上了**内容**（`cond_audio_perm ≈ 0`）。两条互斥假设**尚未区分**（plan 07 §9-64 存疑 3/4）：
   ① 音频的时间信息本来就不足（组内 AUC 0.534，只比随机高 0.036）；② cross-attention 的两条轴仍不在同一时间基
   （field token 用 τ 格下标、音频 key 用秒）。② 有现成开关 `model.seconds_position=true`——它**同样被这次的 bug
   冻结过**，现在才第一次可用。**建议**：跑 `seconds_position` 单臂（8000 步、与修复臂同 seed），判据用
   **组内 AUC 口径的探针** + `cond_audio_perm`，**不要**只看 `val_ratio`。
   ⚠️ 音频的增量信息在**当前目标**（遮盖补全、50% 事件可见）下接近 0（`τ+可见场` 组内 0.597 → 加音频 0.587）
   ⇒ **在改目标之前，不要指望 `val_ratio` 因音频而跃升**。
2. **解释「音频被用上」与 `val_ratio` 背道而驰**：修复臂 8k 步 `pred_over_true` 掉到 0.21（对照 0.42）。
   要么「载荷项挤占了校准」，要么「8k 步还在早期」——**需要 24k 步配对 + 多点**才能判（本轮只够 8k 单点）。
3. **plan 05 M5.7**：D1 阈值标定 + thinning 臂的 Hold 端点策略（R3 产物实测：665 个 hold 起点只有 2 个配上终点）。
4. **评估入口接线**：形态已定＝独立 `beatmorph-eval`（仍未实现）。

### 未决项（不阻塞「下一步」1–4）

| 未决 | 出处 |
|------|------|
| **`seconds_position` 修好后是否值得开**（cross-attn 的两条轴） | plan 07 §9-64 存疑 3 |
| **`cond_audio_perm ≈ 0` 的机制**（信息不足 vs 对齐学不出来） | plan 07 §9-64 存疑 4 |
| **跨 `data.workers` 的读数不可比**（实测 4000 步就分叉）⇒ 配对实验必须固定这一项 | plan 07 §9-64 存疑 5 |
| **装饰线模型侧旁路尚未做**（本轮只做了导出侧闸门） | plan 07 §9-65 / RFC-0032 |
| `val_windows=512` 的抽样精度未标定；`val_every=10000` 未做 A/B | plan 07 §9-63 存疑 1–2 |
| **RFC-0038 与 RFC-0039 重叠部分的处置** | RFC-0038 / RFC-0039 §5 |
| `val/nll_shuffled_delta` 的 margin 未标定；`grad_clip_norm=1.0`；G1 下限 0.05 的 per-event 语义 | RFC-0037 §6 |
| 批 K 上限 / RFC-0035 §1 的 91.3% 未更正 / `r==0` 遮盖退化 / 预算按步还是按 epoch | plan 07 §9-59 / §9-52 / §9-43 / RFC-0033 |
| **评估 / 可玩性**：NLL 好 ≠ 谱面可玩 | plan 07 §9-47 H |

**一次性探针（⚠️ `scripts/local_*` 与 `runs/_*` 均不入库，清理即丢）**：本轮读数脚本 = `runs/_diag_align_audit.py`（τ→秒错位的规模）、`runs/_diag_audio_paths.py`（按通路分解音频依赖，支持 `--ckpt-path`）、`runs/_diag_align_path.py`（λ 级直测）、`runs/_probe_audio_info2.py` + `runs/_probe_audio_within.py`（信息层，**含组内 AUC**）、`runs/_audit_e2e_lines.py <chart.json>`（note 落在哪条线上）、`runs/_trend_new.py`（损失趋势）。装配件 `runs/_diag_common.py`。历史：`local_cache_validate.py`、`local_loss_stability.py`、`local_build_profile.py`、`local_inorder_ab.py`、`local_worker_sweep.py`、`local_bucket_survey.py`、`local_cache_fingerprint_check.py`、`runs/_verify_r2r3.py`（`--mode val|e2e --ckpt baseline|armH`）。**G2 诊断套件已随 RFC-0037 退役**（勿引用其 verdict）。⚠️ `scripts/local_chart_mem.py` **坏了**（`deep_bytes` 漏 pydantic `__slots__`，偏低 1.85×）。⚠️ 长跑重定向探针的输出**只用 ASCII `=>`**（`⇒` 在 GBK 重定向下崩溃、吞掉结果）。

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
