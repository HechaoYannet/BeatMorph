# BeatMorph

> **Phigros 谱面端到端自动生成系统** —— 从音频到可玩谱面，自监督学习取代显式标注。
>
> **核心范式**：`音频 → MERT 隐式理解 → 判定线局部系多线强度场 → 掩码补全 → 泊松 NLL → RPEJSON`

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Status: Pre-Alpha](https://img.shields.io/badge/status-pre--alpha-orange.svg)]()

BeatMorph 从原始音频（WAV/MP3）+ 难度（+ 可选判定线事件轨）出发，端到端生成高质量、可玩的 **Phigros 谱面（RPEJSON）**。模型从社区海量自制谱自主学习创作规律，无需人工标注（标注成本 ≈ 0）。

> 📌 **状态**：Pre-Alpha。范式（[RFC-0029](docs/decisions/RFC-0029-phigros-continuous-chart-generation.md)）、奠基文档（v3.0）、事实库已就绪；**核心契约 / 数据流水线 / 强度场 / 生成主干 / 解码与导出 / 训练基础设施六层已落地并通过默认 CI**（**1017 项测试**），**真实 Phira 全库已拉到本地（8551 张 RPE 谱面 + 42.8 GB 音频，不入库）**，特征提取 **8551/8551**、训练窗口 **train 634 952 + val 74 889**，**全库窗口预切缓存已建成并终验通过**（train 353.6 GB / val 41.7 GB，逐位一致 500/500 抽查）；**显存墙已解除（[RFC-0032](docs/decisions/RFC-0032-local-layer-own-line-tracks.md)）**；⛔ **全量大规模训练待 [RFC-0036](docs/decisions/RFC-0036-gate-batch-caliber-and-g2-power.md) 裁定**（真实门禁 G2 查出精度 / 控制沾染 / 判据符号三个独立缺陷），消融臂与评估接线待建**。

---

## 当前状态与下一步（**跨 session 交接件**）

> 本节读者是**下一个 session 的 agent**，不是历史记录。
> **每次交接必须整节重写，不得追加**（规则见 [AGENTS.md](AGENTS.md) §6）。
> 上次交接：**2026-09-28（第十三轮）** ｜ 交接人：主会话（**数据处理收官**：全库窗口预切缓存建成并终验通过；全量训练仍被 G2 门禁回归挡住，RFC-0036 待裁定）

### 当前状态

自检（默认 CI，无网络 / 无权重 / 无 GPU）：`uv run ruff check . && uv run mypy beatmorph && uv run pytest -m "not slow and not gpu and not e2e"` → **1086 passed**（17 deselected）。

**✅ 数据侧全部就绪**（真实语料 `data/processed/`，不入库）

| 项 | 实测值 |
|----|--------|
| 谱面 / 音频 / note | 8551 张 RPE；42.8 GB / 8084 唯一音频；note 1107 万 |
| MERT 特征 | 8551 行全量、`frame_rate = 75.0 Hz`（契约派生）；索引缓存 train / val 均已落盘命中 |
| **窗口预切缓存（本轮建成 + 终验通过）** | train **634 952 窗 / 353.6 GB**（`train/aa677031c17e3a3d`，11.4 h）+ val **74 889 窗 / 41.7 GB**（`val/fac0448693925c86`，1.2 h）；指纹与事前计算一致、`.partial` 残留 0；**逐位一致 400/400 + 100/100**；读取 **95.2 / 76.2 窗/s**（原路径 81× / 65×）⇒ plan 07 §9-54⑮ |
| 重建参数（如需重建照抄） | `beatmorph-build-windows --config-name phigros_masked --jobs 12 --shard-windows 512 --chart-cache-size 2 --feature-cache-size 1`（+`--split val` 再跑一次；**12 进程是拐点**，20 进程更慢且吃内存） |

**⛔ 全量训练的唯一阻塞：G2 门禁回归，RFC-0036 待裁定**（plan 07 §9-54①–⑬，权威记录 `runs/phigros_masked/20260927-180550/gates.txt`）

`beatmorph-train --config-name phigros_masked --gates --device cuda` ⇒ G1/G3/G4 绿、**G2 FAIL**（真实 182.856 vs 打乱 −31.199，需 `>= 192.0`），退出码 5。**与模型 / 损失 / 数据无关**（`model.py`/`losses.py`/`masks.py`/`sanity.py` 自 `b96b12f` 未动；唯一改到门禁输入的是 `cc0e166` 的取批计划层，门禁批退化成「一张谱的开场」）。已定位**两个独立缺陷**（fp32 下全部可复现；原报的第三个 P2 经复核**撤回**，见下）：

- **P0 精度**：bf16 下对照臂重复运行噪声 **2.8×** ≫ 判据余量 5% ⇒ **当前所有 G1-G4 读数（含 PASS）不可解释**；fp32 同进程逐位一致（跨进程仍翻转）。修复＝门禁对照臂跑 fp32（不改判据）。
- **P1 控制沾染**：`shuffle_hidden_counts` 在全部被遮盖格子上置换 ⇒ 连**每线事件预算**（nuisance）一起改。改**线内置换**后门禁批 real 166.74 / lp 257.61 → PASS；但 6 批 **5 PASS / 1 FAIL**，**非万能** ⇒ 多批口径（RFC-0036 §3.B）是必需。
- ~~**P2 判据符号**~~ **已复核撤回**：生产判据 `sanity.py` 自第四轮（`013f347`）起就是符号安全式 `real + 0.05·|real|`；「符号敏感」只存在于本轮 `local_g2_*` 探针脚本的简式 `real*1.05`。已记录的负 real 批按生产公式复核 **verdict 全部不变** ⇒ 无需裁定、无需改码（RFC-0036 §2.6 更正）。⚠️ 教训：**探针不得重新实现判据**。
- **投影**（非门禁记录）：P0+P1 修复后四道门实跑**全绿**（G1 27841→5.60、G2 91.21 vs 137.23、G3 −165.69 vs 1559.08、G4 98.9/99）——但 P1 覆盖面 5/6 + 跨进程漂移 ⇒ 单批判据不成立。⚠️ 此前所有 G2 记录均不可复现。改判据 / 抽取口径属方法论变更，**agent 不得自行决定**（AGENTS.md §1、红线 7）。

### 下一步（按性价比排序，只留仍然有效的）

1. **裁定 [RFC-0036](docs/decisions/RFC-0036-gate-batch-caliber-and-g2-power.md) §3**（推荐＝C：A 夹具确定性抽取 + B 八批口径，P0/P1 为必做前置；通过阈值需同时定死，证据基线＝线内置换 6 批 5 PASS）→ 实现 → `--gates` 全绿。**这是现在唯一挡住全量训练的东西。**
2. **跑全量大规模训练**：配置已就绪（`max_samples: null`、`data.window_cache_dir: data/processed/window_cache` 已写入 `configs/phigros_masked.yaml` 并过 CI）⇒ `beatmorph-train --config-name phigros_masked --gates --device cuda`（20 000 步，单步 ≈0.195 s ⇒ ≈65 min + 门禁；缓存把数据侧等待压到 ≈10 ms/窗）。
3. **val 的真实墙钟仍未实跑验证**（33 s 是推导值），需一次 GPU 前向。
4. **评估入口未接线**（形态已定＝独立 `beatmorph-eval`），待训练产出 checkpoint。

### 未决项（不阻塞 1–4）

| 未决 | 出处 |
|------|------|
| **门禁批抽取口径 + G2 统计效力**（最高优先，随 RFC-0036 裁定） | plan 07 §9-54 / RFC-0036 |
| **val 开销要不要再压**（现状 16.7%，`val_every=4000` ⇒ 4.1%） | plan 07 §9-53 |
| **`val_windows=128` 的构成未收敛**（空窗占比 51.6%→59.0%），指标差需 GPU 才能测 | plan 07 §9-51 ⑤ |
| **RFC-0035 §1 的 91.3%/9.9% 未更正**（实测 46%/54%）⇒ M0/M1/M2 优先级需重排 | RFC-0035 / plan 07 §9-52 |
| **`r == 0` 遮盖退化要不要改遮盖策略**（≈1.0% 的全部步） | plan 07 §9-43 |
| **训练预算是否从「步数」改成「epoch 数」** | RFC-0033 |
| **`grad_clip_norm=1.0` ⇒ 每一步都被裁剪**，`raw_grad_norm` 未落盘 | plan 07 §9-46 |
| **评估 / 可玩性**：NLL 好 ≠ 谱面可玩 | plan 07 §9-47 H |
| **装饰线是否 / 如何从 λ 线轴去掉** —— 已定「用旁路、后续扩展」 | RFC-0032 |

**一次性探针（⚠️ `scripts/local_*` 与 `runs/_*` 均不入库，清理即丢）**：缓存验收 `local_cache_validate.py`（**终验工具**，`--split/--reuse/--sample/--rows`）、`local_window_cache_check.py`、`local_build_profile.py`、`local_inorder_ab.py`、`local_worker_sweep.py`；**G2 诊断套件** `local_g2_*.py`（batchtest / position / inspect / components / full_control / precision_repro / input_control(_fp32) / fp32_sweep / line_preserving / lp_sweep / lp_only / lp_correctness）、`local_gate_projection_p0p1.py`、`local_bucket_survey.py`、`local_plan_runs.py`、`local_plan_positions.py`、`local_cache_fingerprint_check.py`。⚠️ `scripts/local_chart_mem.py` **坏了**（`deep_bytes` 漏 pydantic `__slots__`，读数偏低 1.85×），用前先修。⚠️ 长跑重定向探针的输出**只用 ASCII `=>`**（`⇒` 在 GBK 重定向下 `UnicodeEncodeError` 崩掉打印、吞掉结果——本轮实测）。


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
