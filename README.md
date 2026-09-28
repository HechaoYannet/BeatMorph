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
> 上次交接：**2026-09-29** ｜ 交接人：主会话（**首跑全量训练已跑完 20 000 步 / exit 0**：管线可用性证实，
> 但这一版**目标/优化器零进展** ⇒ 扩规模结论作废；处置待裁定 —— 提案见
> [RFC-0038](docs/decisions/RFC-0038-training-saturation-and-budget.md)，实测见 plan 07 §9-61）

### 当前状态

自检（默认 CI，无网络 / 无权重 / 无 GPU）：`uv run ruff check . && uv run mypy beatmorph && uv run pytest -m "not slow and not gpu and not e2e"` → **1089 passed**（16 deselected）。

**✅ 首跑全量训练已跑完：管线可用性已证实**（plan 07 §9-61）

- `runs/phigros_masked/20260928-144342`：**20 000 步、exit 0、零卡死**，墙钟 **97.3 min**。
- 门禁 **G1/G3/G4 全绿**（3.5 min）；谱面覆盖 **6614/6614 = 100%**（RFC-0034 的全局发牌在长跑里成立）。
- `data_share` 中位 **2.9%**（窗口缓存把数据侧移出关键路径；原路径 46%）；`vram_reserved_gib` 中位 1.23 GiB / max 1.53 GiB（显存卫生生效）。
- 权重：`best.pt` = **step 2000**（按 `val/ratio` 选），另有 step-16000/18000/20000。

**⚠️ 但它是一次「零进展」的 run ⇒ 依 RFC-0037 R1 该 run 的扩规模结论作废（处置待裁定）**

| 事实 | 实测 |
|------|------|
| 训练饱和 | 非空窗 per-event 损失按 2000 步分块的**中位 9.52 → 10.24（10 块无趋势）**；`val/ratio` 最优停在 **step 2000（0.7292）**，之后 18 000 步**无一超过** |
| 红线 | 20 个 val 点里 **`val/ratio >= 1` 只有 1 个**（step 3000 = 1.0033，其余 0.73–0.97）⇒ **作废**；`val/nll_shuffled_delta` **20/20 > 0**（+0.688）⇒「输入有信息」未破 |
| **量具限度** | `val_ratio` 残差 sd **0.067** ≈ 全程漂移 0.13（t = 1.73 不显著）；val 只有 **498 事件 / 57.8% 空窗**，空窗对 `val_nll` 恰好贡献 0 ⇒ **红线判据检验力不足** |
| **音频条件** | 帧轴置换 Δ **≤ 0.6%**、置零 Δ ≤ 6%，而事件轨置零 Δ 4.6–30%（占 `val_nll` ≈ 1.06）⇒ **当前目标没有奖励模型使用音频**（未消融，勿读作「音频没用」） |
| val 成本 | Σ val = **2234.9 s = 墙钟 38.3%**（单次 73–190 s，跨墙）；val 后第一步稳定慢 **6–70×** |

**✅ RFC-0037 已裁定并落地（G1/G3/G4 + per-event 损失）**：门禁 G2 整体删除（BasePlan §9 / CLAUDE 红线 7 / AGENTS §4 已修宪）；训练损失**整式除以 `max(E_total,1)`**（argmin 逐位不变）；新增 `loss_sum_raw` / `loss_nonempty_per_event` / `clip_active`；门禁臂固定 fp32；val 置换改**线内**。⚠️ **跨此变更不可比**：损失数值、旧 `gates.txt` 的 G2 行、旧 checkpoint 一律不混用，**不跨此变更 `--resume`**。

**✅ 数据侧就绪**：8551 张 RPE / 8084 唯一音频 / 1107 万 note；MERT 8551/8551 @ 75.0 Hz；窗口 **train 634 952 + val 74 889**；**窗口预切缓存已建成并终验通过**（353.6 + 41.7 GB、逐位一致 400/400 + 100/100、读取 95.2 / 76.2 窗/s）。重建照抄：`beatmorph-build-windows --config-name phigros_masked --jobs 12 --shard-windows 512 --chart-cache-size 2 --feature-cache-size 1`（`--split val` 再跑一次；12 进程是拐点）。

**⚠️ 长跑运维的三条实测结论（不要推翻）**

1. `optim.vram_hygiene_gib=1.0` 是**本机唯一真正生效的显存杠杆**；`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 在 **Windows 上是 no-op**（torch 启动即警告），不要指望它。
2. **跨墙 ≠ 卡死**：step 951 那次（共享 13.2 GB、功耗 31 W 不回来）是卡死；**val 每轮都跨墙**（共享 5.6 GB、功耗 30–65 W 摆动）但**会在 100–190 s 内自己恢复**。⇒ 停机判据（`nvidia-smi` 的 `memory.used > ~7.4 GiB` 或 `power.draw < 45 W` 且显存 > 5 GiB）**只适用于训练步**，动手前先看日志尾部确认不在 val 段。
3. **不要自建自动杀进程的看门狗**：v1 因跨 run 的步号基准误杀过健康运行（plan 07 §9-57 事故 2）；决策者要求**人工盯 + 主动通知**。

**若继续长跑（在 RFC-0038 裁定之前不要改任何训练语义）**：盯 `runs/_full_trainN.log` 与 `runs/phigros_masked/<时间戳>/logs/loss_history.jsonl` —— `loss` 是 **per-event 口径**（与历史 run 不可比，用 `loss_sum_raw` 对照），要看 `clip_active`（当前非空步 **100%** 触发）、`vram_reserved_gib`、`step_time_s` / `data_time_s`。⚠️ `metrics.json` 的 `best_loss` 与摘要里的 "best 0.000009" 是**误导性统计**（单步最小值落在空窗步上），**不要用它选权重**。

### 下一步（按性价比排序，只留仍然有效的）

1. **裁定 [RFC-0038](docs/decisions/RFC-0038-training-saturation-and-budget.md) 的 Q1–Q6**（预算收敛 / 对照臂降频 / val 检验力 / 探针是否批准 / 停机规则 / 权重确认）——**唯一阻塞项**。
2. 按裁定执行：**R1** `max_steps` 20000→4000、`val_every` 1000→500（两者都**不进续训指纹**）；**R2** 新增 `optim.val_contrast_every`（红线指标每次都跑，3 个条件臂降频）；**R3** val 检验力二选一（512 窗 ／ 红线改趋势判据＝**修宪**）。
3. **探针（若 Q4 批准）**：可学性上界（固定 16 个真实窗跑 2000 步）／`lr` 3 臂（3e-4 / 1e-4 / 3e-5，各 1500 步）／音频通路梯度占比。**不改生产代码、不改损失**；这是分清「学完了」与「学不动」的唯一路径。
4. **评估入口接线**：形态已定＝独立 `beatmorph-eval`（仍未实现），待后续 checkpoint。

### 未决项（不阻塞 1–4）

| 未决 | 出处 |
|------|------|
| **平台是「学完了」还是「学不动」**（不可约熵 vs 优化器/目标）——**现有数据分不开**，只能靠探针 | plan 07 §9-61 |
| **音频为何不被用**（LoRA 没更新 / 注入路径弱 / 目标不需要）——未分离 | plan 07 §9-61 ③ |
| **val 检验力与红线判据口径**（498 事件、57.8% 空窗、sd 0.067） | RFC-0038 Q3 |
| **`val/nll_shuffled_delta` 的 margin 未标定**（只有符号判据 > 0） | RFC-0037 §6-2 |
| **`grad_clip_norm=1.0` 在 per-event 损失下是否仍合适**（非空步 **100%** 触发，有效更新 ≈ `lr·g/‖g‖`） | RFC-0037 §6-1 |
| **G1 绝对下限 0.05 的 per-event 语义未标定** | RFC-0037 §6-3 |
| **val 预算模型偏 3–5×**（schema 写 0.09 s/窗/臂 ⇒ 37 s，实测 0.157–0.26 ⇒ 100–190 s） | plan 07 §9-59 / §9-61 |
| **批 K 上限**（val 已实际跨墙；val 侧建议按预算拆桶而非跳窗，**须裁定**） | plan 07 §9-59 |
| **RFC-0035 §1 的 91.3%/9.9% 未更正**（实测 46%/54%）⇒ M0/M1/M2 优先级需重排 | RFC-0035 / plan 07 §9-52 |
| **`r == 0` 遮盖退化要不要改遮盖策略**（≈1.0% 的全部步） | plan 07 §9-43 |
| **训练预算是否从「步数」改成「epoch 数」** | RFC-0033 |
| **评估 / 可玩性**：NLL 好 ≠ 谱面可玩 | plan 07 §9-47 H |
| **装饰线是否 / 如何从 λ 线轴去掉** —— 已定「用旁路、后续扩展」 | RFC-0032 |

**一次性探针（⚠️ `scripts/local_*` 与 `runs/_*` 均不入库，清理即丢）**：`local_cache_validate.py`（`--split/--reuse/--sample/--rows`）、`local_loss_stability.py`（量级标度律 + 尖峰检测）、`local_build_profile.py`、`local_inorder_ab.py`、`local_worker_sweep.py`、`local_bucket_survey.py`、`local_cache_fingerprint_check.py`；本轮读数脚本 `runs/_analyze_*.py`。**G2 诊断套件 `local_g2_*.py` 已随 RFC-0037 退役**（判据是探针简式 `real*1.05`，勿引用其 verdict）。⚠️ `scripts/local_chart_mem.py` **坏了**（`deep_bytes` 漏 pydantic `__slots__`，偏低 1.85×）。⚠️ 长跑重定向探针的输出**只用 ASCII `=>`**（`⇒` 在 GBK 重定向下崩溃、吞掉结果）。

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
