# BeatMorph

> **Phigros 谱面端到端自动生成系统** —— 从音频到可玩谱面，自监督学习取代显式标注。
>
> **核心范式**：`音频 → MERT 隐式理解 → 判定线局部系多线强度场 → 掩码补全 → 泊松 NLL → RPEJSON`

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Status: Pre-Alpha](https://img.shields.io/badge/status-pre--alpha-orange.svg)]()

BeatMorph 从原始音频（WAV/MP3）+ 难度（+ 可选判定线事件轨）出发，端到端生成高质量、可玩的 **Phigros 谱面（RPEJSON）**。模型从社区海量自制谱自主学习创作规律，无需人工标注（标注成本 ≈ 0）。

> 📌 **状态**：Pre-Alpha。范式（[RFC-0029](docs/decisions/RFC-0029-phigros-continuous-chart-generation.md)）、奠基文档（v3.0）、事实库已就绪；**核心契约 / 数据流水线 / 强度场 / 生成主干 / 解码与导出 / 训练基础设施六层已落地并通过默认 CI**（**1089 项测试**），**真实 Phira 全库已拉到本地（8551 张 RPE 谱面 + 42.8 GB 音频，不入库）**，特征提取 **8551/8551**、训练窗口 **train 634 952 + val 74 889**，**全库窗口预切缓存已建成并终验通过**（train 353.6 GB / val 41.7 GB，逐位一致 500/500 抽查）；**显存墙已解除（[RFC-0032](docs/decisions/RFC-0032-local-layer-own-line-tracks.md)）；[RFC-0037](docs/decisions/RFC-0037-remove-g2-and-per-event-loss.md) 已裁定并落地**（门禁 G2 删除 → **G1/G3/G4** 三道 + val held-out 对照判读红线；训练损失**按事件归一**）；**全量大规模训练待启动**，消融臂与评估接线待建**。

---

## 当前状态与下一步（**跨 session 交接件**）

> 本节读者是**下一个 session 的 agent**，不是历史记录。
> **每次交接必须整节重写，不得追加**（规则见 [AGENTS.md](AGENTS.md) §6）。
> 上次交接：**2026-09-28（第十三轮收尾）** ｜ 交接人：主会话（**RFC-0037 已裁定并落地**：门禁 G2 删除 + 训练损失按事件归一；全库窗口预切缓存建成并终验；**全量训练即将启动**）

### 当前状态

自检（默认 CI，无网络 / 无权重 / 无 GPU）：`uv run ruff check . && uv run mypy beatmorph && uv run pytest -m "not slow and not gpu and not e2e"` → **1089 passed**（16 deselected）。

**✅ 门禁体系已按 RFC-0037 改造完毕（修宪级，2026-09-28 决策者裁定）**

- **G2（打乱标签对照）整体删除**：「单批 100 步速度赛」在真实批上判别力未被证明（逐批 verdict 翻转、置换 nuisance、跨进程漂移；证据 RFC-0036 §1-§2 + plan 07 §9-54/§9-55）。门禁现为 **G1/G3/G4 三道**。BasePlan §9、CLAUDE 红线 7、AGENTS §4 已同步修宪。
- **「输入对目标零信息」的命题改由 val 在线对照承担**：`val/nll_shuffled_delta`（**线内置换**、线数归一、128 held-out 窗）+ `val/ratio` + `cond_*_delta`。**判读红线**：任一 val 点 `val/ratio >= 1` 或 `val/nll_shuffled_delta <= 0` ⇒ 该 run 的扩规模结论**作废**。
- **训练损失改为 per-event 归一**：`L = [Σ_有效线((1/r)Σ_被遮盖 −n·log λ + ∫λdV)] / max(E_total, 1)`，**整式相除**（只除事件项会把最优强度缩小 D 倍——已写进 docstring 与单测）。argmin 逐位不变；改变的只是步间量级。实测旧口径 `loss ≈ 2.24·K^0.055·E^0.93`（R²=0.74）⇒ 锯齿是测度结构，新口径下才可读。
- **附带指标**：`loss_sum_raw`（旧 sum 口径，跨本次变更对照用）、`loss_nonempty_per_event`（主趋势）、`clip_active`（裁剪触发 0/1；旧 run 实测 56.4% 的步触发、裁剪前中位范数 316）。
- **门禁臂固定 fp32**（原 P0）；**val 置换改线内**（原 P1，吸收 RFC-0036 结论）。
- ⚠️ **跨本次变更不可比**：loss 数值、旧 `gates.txt` 的 G2 行、旧 checkpoint 一律不得混用；**不跨此变更 `--resume`**（GatesConfig 字段变化会让续训指纹 fail-closed，属预期）。
- 护栏：`tests/unit/generation/test_loss_per_event.py`（含**梯度恰差 1/D** 的 argmin 不变量）、`tests/unit/infra/test_shuffle_within_line.py`（五契约）、`tests/unit/infra/test_gate_assembly.py`（装配 + fp32）。

**✅ 数据侧全部就绪**（真实语料 `data/processed/`，不入库）

| 项 | 实测值 |
|----|--------|
| 谱面 / 音频 / note | 8551 张 RPE；42.8 GB / 8084 唯一音频；note 1107 万 |
| MERT 特征 | 8551 行全量、`frame_rate = 75.0 Hz`（契约派生）；索引缓存 train / val 均已落盘命中 |
| **窗口预切缓存（本轮建成 + 终验通过）** | train **634 952 窗 / 353.6 GB**（`train/aa677031c17e3a3d`，11.4 h）+ val **74 889 窗 / 41.7 GB**（`val/fac0448693925c86`，1.2 h）；指纹与事前计算一致、`.partial` 残留 0；**逐位一致 400/400 + 100/100**；读取 **95.2 / 76.2 窗/s**（原路径 81× / 65×）⇒ plan 07 §9-54⑮ |
| 重建参数（如需重建照抄） | `beatmorph-build-windows --config-name phigros_masked --jobs 12 --shard-windows 512 --chart-cache-size 2 --feature-cache-size 1`（+`--split val` 再跑一次；**12 进程是拐点**，20 进程更慢且吃内存） |

**🚀 下一步就是跑全量训练**（门禁已可跑：三道门 + 修宪后的判读红线；数据侧与损失口径都已就绪）

`uv run beatmorph-train --config-name phigros_masked --gates --device cuda --skip-env-doctor`
—— 门禁（G1/G3/G4，fp32 臂）预期 **10-15 min**，随后 20 000 步训练（单步 ≈0.195 s ⇒ ≈65 min + val）。
要看的三件事：① `clip_active` 率是否从旧的 **56.4%** 明显下降；② `val/ratio` 是否持续 < 1 且下行；
③ `val/nll_shuffled_delta` 是否 > 0（≤ 0 ⇒ 该 run 作废，RFC-0037 判读红线）。

### 下一步（按性价比排序，只留仍然有效的）

1. **跑全量大规模训练**（配置已就绪：`max_samples: null`、`data.window_cache_dir: data/processed/window_cache`、`data.workers: 8`）——命令与三个观察点见上。
2. **首 run 的必读三件事**：`clip_active` 率（旧 56.4%）、`val/ratio` 趋势、`val/nll_shuffled_delta > 0`。若红线破 ⇒ 停并开新 RFC，**不得**继续扩规模。
3. **val 的真实墙钟仍未实跑验证**（33 s 是推导值），首 run 的 `val_time_s` 即为实测。
4. **评估入口未接线**（形态已定＝独立 `beatmorph-eval`），待训练产出 checkpoint。

### 未决项（不阻塞 1–4）

| 未决 | 出处 |
|------|------|
| **`grad_clip_norm=1.0` 在 per-event 损失下是否仍合适**（旧 run 56.4% 的步触发、裁剪前中位 316 ⇒ 归一化梯度形态）——先看 `clip_active` 率再议 | RFC-0037 §6-1 / plan 07 §9-55 |
| **`val/nll_shuffled_delta` 的 margin 未标定**（现阶段只有符号判据 > 0） | RFC-0037 §6-2 |
| **G1 绝对下限 0.05 的 per-event 语义未标定**（相对判据主导，暂不阻塞） | RFC-0037 §6-3 |
| **val 开销要不要再压**（现状 ≈16.7%，`val_every=4000` ⇒ 4.1%） | plan 07 §9-53 |
| **`val_windows=128` 的构成未收敛**（空窗占比 51.6%→59.0%） | plan 07 §9-51 ⑤ |
| **RFC-0035 §1 的 91.3%/9.9% 未更正**（实测 46%/54%）⇒ M0/M1/M2 优先级需重排 | RFC-0035 / plan 07 §9-52 |
| **`r == 0` 遮盖退化要不要改遮盖策略**（≈1.0% 的全部步） | plan 07 §9-43 |
| **训练预算是否从「步数」改成「epoch 数」** | RFC-0033 |
| **评估 / 可玩性**：NLL 好 ≠ 谱面可玩 | plan 07 §9-47 H |
| **装饰线是否 / 如何从 λ 线轴去掉** —— 已定「用旁路、后续扩展」 | RFC-0032 |

**一次性探针（⚠️ `scripts/local_*` 与 `runs/_*` 均不入库，清理即丢）**：缓存终验 `local_cache_validate.py`（`--split/--reuse/--sample/--rows`）、损失波动归因 `local_loss_stability.py`（对 `logs/loss_history.jsonl` 做量级标度律 + 尖峰检测）、`local_build_profile.py`、`local_inorder_ab.py`、`local_worker_sweep.py`、`local_bucket_survey.py`、`local_plan_runs.py`、`local_cache_fingerprint_check.py`；**G2 诊断套件 `local_g2_*.py` 已随 RFC-0037 退役**（保留在磁盘仅供追溯，勿再引用其 verdict：其判据是探针简式 `real*1.05`，不是生产判据）。⚠️ `scripts/local_chart_mem.py` **坏了**（`deep_bytes` 漏 pydantic `__slots__`，读数偏低 1.85×）。⚠️ 长跑重定向探针的输出**只用 ASCII `=>`**（`⇒` 在 GBK 重定向下 `UnicodeEncodeError` 崩掉打印、吞掉结果）。


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
