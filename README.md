# BeatMorph

> **Phigros 谱面端到端自动生成系统** —— 从音频到可玩谱面，自监督学习取代显式标注。
>
> **核心范式**：`音频 → MERT 隐式理解 → 判定线局部系多线强度场 → 掩码补全 → 泊松 NLL → RPEJSON`

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Status: Pre-Alpha](https://img.shields.io/badge/status-pre--alpha-orange.svg)]()

BeatMorph 从原始音频（WAV/MP3）+ 难度（+ 可选判定线事件轨）出发，端到端生成高质量、可玩的 **Phigros 谱面（RPEJSON）**。模型从社区海量自制谱自主学习创作规律，无需人工标注（标注成本 ≈ 0）。

> 📌 **状态**：Pre-Alpha。范式（[RFC-0029](docs/decisions/RFC-0029-phigros-continuous-chart-generation.md)）、奠基文档（v3.0）、事实库已就绪；**核心契约 / 数据流水线 / 强度场 / 生成主干 / 解码与导出 / 训练基础设施六层已落地并通过默认 CI**（**980 项测试**），**真实 Phira 全库已拉到本地（8551 张 RPE 谱面 + 42.8 GB 音频，不入库）**，特征提取 **8005/8084**、训练窗口 **634 952**（索引有落盘缓存）；**合成门禁与真实 200 行切片门禁均全绿**（真实一轮 EXIT=0，见下方交接件）；**真实数据训练、消融臂与评估接线待建**。

---

## 当前状态与下一步（**跨 session 交接件**）

> 本节读者是**下一个 session 的 agent**，不是历史记录。
> **每次交接必须整节重写，不得追加**（规则见 [AGENTS.md](AGENTS.md) §6）。
> 上次交接：**2026-09-27（第五轮）** ｜ 交接人：主会话（定位并修掉 τ 轴终点缺陷 ⇒ 真实门禁首次全绿；门禁批非空；索引落盘缓存）

### 当前状态

自检（默认 CI，无网络 / 无权重 / 无 GPU）：`uv run ruff check . && uv run mypy beatmorph && uv run pytest -m "not slow and not gpu and not e2e"` → **980 passed**（约 57 s；另有 17 项 slow/gpu/e2e 默认不跑）。

**实跑证据**：

- ★ **真实数据门禁首次全绿**：`uv run beatmorph-train --config-name phigros_masked --gates-only --device cuda --skip-env-doctor data.max_samples=200` → **EXIT=0、2305.8 s ≈ 38 min**：
  G1 PASS（47455.4 → 65.1，`g1_events=20` **非空**）｜ **G2 PASS**（真实 **1672.99** vs 打乱 **3875.21** ⇒ 打乱臂 2.32× 更差，门限 1.05×）｜ G3 PASS（107.77 ≤ 2412.17，0.040× 基线）｜ G4 PASS（215 帧）。
  权威记录：`runs/phigros_masked/20260927-043828/gates.txt`。**⇒ 真实数据训练的前置已清**（只剩算力决定 + RFC-0031 的口径确认）。
- **合成门禁全绿**：`uv run beatmorph-train --config-name smoke --gates-only`（CPU 约 2 min；G2 差距 45%）。
- **训练通路验证（合成夹具）**：`--config-name smoke --gates --skip-env-doctor optim.max_steps=2000 run.save_every=500` → EXIT=0、218.8 s、六件套齐全。**这是通路验证，不是训练结果**。
- **全量特征提取**：已完成 **8005 / 8084（99.0%）**；余 79 首是解不开的 mp3（`pairs` 已重建）。

**真实数据（`data/processed/`，不入库）**

| 项 | 实测值 |
|----|--------|
| 谱面 | **8551** 张 RPE（枚举 9651；拒收 1099） |
| 音频 | **42.8 GB** / **8084** 个唯一音频（sha1 去重）；唯一曲目 **6807** |
| 判定线 / note | 中位 25 条；note 计 1107 万（Tap 55.7% / 背面 3.11%） |
| 特征缓存 | **8005 / 8084**（可续跑；生产配置 0.76 s/首、GPU 96%） |
| ★ **τ 轴截断** | **3580 / 6750 行（53%）** 的 `META.chartTime` 虚高（中位 **52.8×**）⇒ 改由音频时长截断，省下 **30.8 万小时**空窗（RFC-0031） |
| ★ 训练窗口 | train **634 952**（94.1 窗/行）—— 旧口径是 3361 万（98% 空窗） |
| 训练清单 | `pairs.json`：train 6750 / val 814 / test 890 / 泛化 658（按曲目切分） |
| 索引 | 全库首次构建 **36.5 min**（0.325 s/行），落盘缓存后**秒级**命中 |

驱动脚本：`scripts/fetch_phira.py`（`meta` / `fetch` / `pairs` / `stats` / `all`）与 `scripts/extract_features.py`（按音频 sha1 去重 + 缓存六项校验），用法见 [docs/TRAINING.md](docs/TRAINING.md) §3/§5。

**已通的四条链**

1. **解码导出**：λ 场 → `decoder.decode_field`（D1 峰值 / D2 thinning）→ 合法性后处理 → `io/formats/rpejson` 写出 → 读回；
2. **训练通路**：清单 + 特征缓存 → `data.ChartPairDataset`（窗口化 + 行级 LRU）→ `collate_field_batch` → `FieldBatch` → `MaskedFieldModel` → 泊松 NLL → checkpoint；
3. **真实数据通路**：Phira API → 预筛 + 选择性下载 → RPEJSON 解析 → 三级质检 → 带 provenance 的清单 → 真实 MERT 权重特征 → 窗口数据集 + **索引落盘缓存**；
4. **门禁通路**：`--gates-only` 在**合成夹具**与**真实 200 行切片**上都给出全绿结论（真实一轮 38 min，必须 `--device cuda`）。

**本轮修掉的三处真缺陷**（都有回归测试 + 实跑证据）

1. **τ 轴终点缺陷（本轮最大的一处，RFC-0031）**：`PhigrosChart.duration_s() = max(最后一事件, META.chartTime)` 无条件信任 `chartTime`，而全库 **52.2%** 的谱面 `chartTime` 虚高（中位 52.8×、最大 3.4e6×）。
   后果：train split 切出 **3361 万窗**（98% 是空窗 + 音频整段补零）、全库索引 2.3 h、**G1 抽到 0 事件窗口报假绿**、**G2 打乱对照被伪造成「结构性失效」**（打乱臂 0.835× 反而更优）。
   修法：`data.tau_end_policy="audio"`（默认）= `min(谱面口径, 特征缓存元数据的音频时长)`，截断行数 / 轴外事件**显式记账**；`"chart"` 一行回退。
   ⇒ train split **634 952 窗**、索引 36.5 min，**同一配置下 G2 从 FAIL 变 PASS**（plan 02 §9 第四轮 / plan 07 §9-25）。
2. **门禁批可以是空的**（`gates.batch_min_events=1`）：`BatchSource.batch(min_events=k)` 重抽到达标为止，取不到即抛（fail-closed）；`gates.txt` 现落盘 `g1_events`/`g1_lines`/`g2_events`/`g2_hidden_events`/`g2_lines`/`g3_events`/`g3_lines`，让「空过」当场可判。
3. **门禁装配的显存卫生**：四个批 + 四个模型同时在场时 8 GB 卡会滑进 **Windows 共享内存**（实测 7.88/8.15 GB、功耗 88 W），同一个 G3 步从 **0.57 s 变成 8.8 s**，整轮跑不完；现改为 **G3 预计算先跑、跑完立即释放**（峰值回到 6.0-6.6 GB、74-94 W、38 min 完成）。
   ⚠️ 运维判据：`nvidia-smi` 的 `power.draw` 应稳定在 ~90 W；长期偏低且显存贴顶 ⇒ 先怀疑共享内存回退，**不要**盲目加步数。

**随本轮一起落地的吞吐修复**（plan 02 §9 ④）：索引**落盘缓存**（指纹覆盖口径版本/配置/行身份/文件 stat，损坏或指纹不符一律回退重建；写入原子）+ **行级 LRU**（`chart_cache_size=8` / `feature_cache_size=2`，同一行的多个窗口不再重复解析 4 MB 谱面、不再重复读 40-50 MB 特征）。

**未落地 / 未验证**

- **真实数据训练尚未启动**：门禁已全绿、fail-closed 不再拦，唯一前置是**算力决定**（一条命令，见「下一步」1）；
- **RFC-0031 待裁定**（τ 轴终点口径：音频时长 / 最后一事件 / 两者取小；是否含 `META.offset`）；实现已按提案落地，裁定若改变口径 ⇒ 一行配置 + 重跑门禁；
- **评估入口只差一行**（plan 06 §9-11 / plan 08）：`eval/pipeline.py` 已就绪并测过，缺的是「评估入口长什么样」这个政策决定；接上之前 `metrics.json` 不会有 `eval` 分节；
- plan 04 的其余消融臂（M7 B4 / M9 B3 / M10 B5 / M11 矩阵）与 **B1 的 G3 常数基线口径**（focal 不适用泊松闭式，plan 04 §9-20，**未定之前不得给 B1 报 G3**）；plan 06 M6.7/M6.8、plan 08 待建；plan 05 M5.7（双解码臂 B6 出数）待已训练模型；
- **Lightning 后端真机实跑**（`--extra train` 已装齐、env doctor E4 PASS，但 `run.backend=lightning` 未真机跑过）；
- ⚠️ **global 层的 `O((K·T)²)`**（架构级，须 RFC）：门禁/训练步时 ∝ `K²`（`K` = 批内最大判定线数，中位 25、`k_max=128`）⇒ `K=29` 单步 ~7 s、`K=33` ~9 s。它同时是**显存墙**与**吞吐墙**，在扩大 `t_window`/数据规模之前必须先裁定（候选：SDPA/分块注意力/降 `t_window`）。

### 下一步

0. **裁定 [RFC-0031](docs/decisions/RFC-0031-tau-axis-endpoint.md)（τ 轴终点口径，当前唯一挡住真实训练的「口径」问题）**：实测证据、三条备选、代价与记账都已在 RFC 内；实现已按提案落地（`data.tau_end_policy=audio`），裁定若改变口径 ⇒ 改一行 + 重跑权威 `gates.txt`。
1. **真实数据训练（前置已清，属算力决定）**：
   `uv run beatmorph-train --config-name phigros_masked --gates --device cuda --skip-env-doctor data.max_samples=200 optim.max_steps=2000`
   —— `--gates` 会先重跑一遍门禁（约 38 min，索引命中缓存），随后训练；步时 ∝ `K²`（真实批 `K` 中位 25）⇒ **先跑 2000 步量真实吞吐**，再谈全量 `max_samples=null`。
2. **评估接线只差一行**（政策决定：入口形态）：`eval/pipeline.py` 已就绪并测过，缺的是「评估入口长什么样」这个政策决定（plan 06 §9-11 / plan 08）——裁定后接上，`metrics.json` 才会有 `eval` 分节。
3. **plan 04 消融臂**（M9 B3 / M7 B4 / M10 B5 / M11 矩阵）：每个新目标先过 G1–G4，结果写进训练日志；**B1 的 G3 基线口径先裁定**（plan 04 §9-20）。
4. **门禁预算曲线**优先度已下降（16 样本已给出 2.32× 差距，门限 1.05×；更大的样本数会被 `K²` 步时劝退）⇒ 真要压，先处理 global 层的 `O(K²T²)`（架构级，须 RFC）。
5. **若之后补抽特征**：`uv run python scripts/extract_features.py`（可续跑）→ 再 `uv run python scripts/fetch_phira.py pairs`；索引缓存会因清单/文件 stat 变化自动重建。

### 未决项（不阻塞 1–5）

| 未决 | 出处 |
|------|------|
| **[RFC-0031](docs/decisions/RFC-0031-tau-axis-endpoint.md)：τ 轴终点口径**（实现已按提案落地，待裁定） | RFC-0031 |
| **[RFC-0030](docs/decisions/RFC-0030-decoder-export-contract-ownership.md) 整体待裁定**：解码/导出契约归属 + 六项实现口径；**实现已按提案落地** | RFC-0030 |
| **契约父线合成**与 A 级证据不一致 ⇒ 含父线谱面（实测 26%）的跨线几何计数为近似值（需 contracts-agent） | RFC-0030 §后果-2 |
| **跨谱 batching 的契约缺口**：`FieldBatch` 只带一个 grid，且**没有 `time_mask`**（音频 padding 是伪造静音）⇒ 跨谱批训练前须裁定 | plan 07 §9-13 / plan 02 §9 |
| **global 层的 `O((K·T)²)`**：既是显存墙也是吞吐墙（架构级，须 RFC）；候选＝SDPA / 分块注意力 / 降 `t_window` | plan 07 §9-22 / §9-28 |
| **B1 的 G3 常数基线口径未定**（泊松闭式 `N(1+log|Ω|/N)` 不适用于 focal；未定之前不得给 B1 报 G3） | plan 04 §9-20 |
| **遮盖重标定口径**：`hidden`(1/r，实现默认) vs 文档字面 `observed`(1/(1-r)) vs 不自洽的 `hidden_doc` | plan 04 §9-6 |
| **mask 泄漏口径**：默认「以事件为锚的 token 区块遮盖 + 空格稀释」偏离 §4.2 字面表述 | plan 04 §9-18 |
| plan 04 报告方法学两项：条件学习的「边际盆地」loss 差监控（R-04-5）、G2 合成任务与训练预算 | plan 04 §9-15 / §9-16 |
| 输出头**直连 skip** 是否写进 BasePlan §3.3 的架构描述 | plan 04 §9-17 |
| eval 三项口径待复核：多重比较校正、相位搜索范围/步长、难度分档边界 | plan 06 §9-1 / §9-3 / §9-5 |
| `N` 的最终取值（默认 128 已裁；仍需碰撞率–N / NLL–N / F1–N 三条曲线；**最小同线同刻间距 = 0.0 已复核**：现口径漏 type 通道，见 plan 02 §9 第三轮 ⇒ 统计函数待改） | plan 03 §9-7、plan 02 §9-5 |
| 秒↔τ 两个换算点是否合并为一份实现；τ 轴终点口径的**实现落点**（已在 dataset） | plan 02 §9-11、plan 03 §9-13 / §9-14 |
| BPM 变更点落在 1/48 拍格内时 `J_j` 的取值（默认 `left`） | plan 03 §9-16 |
| `RPE_HEIGHT_RATIO` 的出处（D3）；旋转正方向的屏幕含义（D1）；`META.offset` 的符号解释 | plan 00 §9-6/§9-8、plan 05 §9-8 |

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
