# BeatMorph

> **Phigros 谱面端到端自动生成系统** —— 从音频到可玩谱面，自监督学习取代显式标注。
>
> **核心范式**：`音频 → MERT 隐式理解 → 判定线局部系多线强度场 → 掩码补全 → 泊松 NLL → RPEJSON`

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Status: Pre-Alpha](https://img.shields.io/badge/status-pre--alpha-orange.svg)]()

BeatMorph 从原始音频（WAV/MP3）+ 难度（+ 可选判定线事件轨）出发，端到端生成高质量、可玩的 **Phigros 谱面（RPEJSON）**。模型从社区海量自制谱自主学习创作规律，无需人工标注（标注成本 ≈ 0）。

> 📌 **状态**：Pre-Alpha。范式（[RFC-0029](docs/decisions/RFC-0029-phigros-continuous-chart-generation.md)）、奠基文档（v3.0）、事实库已就绪；**核心契约 / 数据流水线 / 强度场 / 生成主干 / 解码与导出 / 训练基础设施六层已落地并通过默认 CI**（**1017 项测试**），**真实 Phira 全库已拉到本地（8551 张 RPE 谱面 + 42.8 GB 音频，不入库）**，特征提取 **8005/8084**、训练窗口 **634 952**（索引有落盘缓存）；**合成门禁与真实切片门禁均全绿**（见下方交接件）；**显存墙已解除（[RFC-0032](docs/decisions/RFC-0032-local-layer-own-line-tracks.md)），全量大规模训练正在运行（前置修复：[RFC-0033](docs/decisions/RFC-0033-sampler-coverage-and-epoch.md) 采样器覆盖率 + [RFC-0034](docs/decisions/RFC-0034-data-supply-throughput.md) 数据供给吞吐诊断，两者都属「结果看起来正常却是错的」类缺陷），消融臂与评估接线待建**。

---

## 当前状态与下一步（**跨 session 交接件**）

> 本节读者是**下一个 session 的 agent**，不是历史记录。
> **每次交接必须整节重写，不得追加**（规则见 [AGENTS.md](AGENTS.md) §6）。
> 上次交接：**2026-09-27（第十轮）** ｜ 交接人：主会话（[RFC-0034](docs/decisions/RFC-0034-data-supply-throughput.md) §11：GPU 利用率量准 **44.3%**，瓶颈坐实为数据供给，`data.workers 3 → 8` 把它抬到 **75.5%**）

### 当前状态

自检（默认 CI，无网络 / 无权重 / 无 GPU）：`uv run ruff check . && uv run mypy beatmorph && uv run pytest -m "not slow and not gpu and not e2e"` → **1033 passed**（17 deselected）。

**★ 全量大规模训练正在运行**：`runs/phigros_masked/20260927-100604/`（`max_steps=20000`，交接时 step ≈17300，ETA ≈9 min）。门禁 **36.5 min 全绿**（有界切片 `gates.gate_samples=200`，索引秒级命中）：G1 93552.7 → -7.9 ｜ G2 真实 73.68 vs 打乱 228.75（**3.10×**）｜ G3 -24.66 ≤ 2179.76 ｜ G4 154 帧 ≈ 2.05 s × 75 Hz。训练中 loss 16684 → ~100，K 中位 25，峰值显存 4.87 GiB，`training_health` 退出码 0。

> ⚠️ **本轮（第十轮）改正了上一轮写在这里的一个错误结论**：上一轮把 `perf/data_share` 的**中位 5.0%** 当成「数据侧不是瓶颈」的证据，并据此说「不要再加 worker」。**方向反了。** 该分布极端重尾（中位 5.0% vs **均值 45.0%**），实测 GPU 利用率只有 **44.3%**、其中 **42.7% 的采样完全空转**。解药恰恰是加 worker：`data.workers 3 → 8` 后 GPU 利用率 **75.5%**、空转降到 **9.5%**、步时 mean **0.296 → 0.195 s**。量法与逐项排除见 [RFC-0034 §11](docs/decisions/RFC-0034-data-supply-throughput.md)。

**★ 缺陷一：采样器覆盖率（[RFC-0033](docs/decisions/RFC-0033-sampler-coverage-and-epoch.md)，已修）**——本轮修掉的两个缺陷都属「结果看起来正常、实则错误」类。

旧采样器 `ManifestBatchSource` 只有**一个全局共享游标**，且每步按**当前桶**长度取模；桶长最小为 **1**，撞到就被清零 ⇒ **1000 步后彻底饱和**（用仓库里那份 `_draw` 本身实测）：

| 步数 | 抽到的不同窗口 | 不同谱面 |
|---|---|---|
| 1,000 | **993** | **674** |
| 3,000 / 20,000 | **993（不变）** | **674（不变）** |

⇒ `max_steps` 加到 600,000 也不会见到新数据：「全量训练」实际只在 **0.156% 的窗口 / 10.2% 的谱面**上训练，而 loss 曲线完全正常。修法 = 每桶独立游标 + 桶内种子化洗牌 + **按剩余窗口数加权选桶**；新增 `epoch` / `coverage/windows_seen` / `coverage/charts_seen` 在线标量。**修复后 20,000 步覆盖 6,036/6,614 张谱面（91.3%）**。

- **epoch**（本轮立的口径）：1 epoch = 走遍全库、每窗口恰好一次 = **634,952 步**（本机 ≈**88 h**）⇒ **不作预算单位、只作可观测指标**；20,000 步 ≈ 3.15% window-epoch 但 ≈**91% 谱面覆盖**——**谱面覆盖率才是诚实的进度单位**。
- **配套**：门禁批改用**有界切片** `gates.gate_samples=200`（否则会加权抽到 K≈128 的大桶 ⇒ 5 min → **36.5 min**）；护栏 `test_sampler_coverage.py`（6）+ `test_gate_samples_bound.py`（4）。

**★ 缺陷二：事件轨求值占每窗口 CPU 52.5%（逐位等价，32.9×，已修）**

剖面显示每个窗口 **52.5%** 的 CPU 花在 `beatmorph/data/tracks.py::line_tracks_at`（1.46e6 次 `track_value`、**3.97e7 次 `Beat.to_beats()`**）。改为 `searchsorted(starts)` + `prefix_max(ends)` 的「列表序首个命中」定位（证明见 `_first_matching_keyframe`），取值仍调用契约层 `numeric_at` ⇒ **数值逐位不变**（判据 `torch.equal`）。每窗口 CPU **1.218 → 0.582 s**。护栏 `tests/unit/data/test_tracks_vectorized.py`（4）。

**★ GPU 空转的修法已落地：[RFC-0034](docs/decisions/RFC-0034-data-supply-throughput.md) S1–S4（计划 / 物化 / 搬运三层分离）**

决策者问「GPU 平均占用率依然不算高，是不是 workers 太小了」。**答案：这个旋钮不存在**——全仓 `num_workers` 只出现在**数据准备**脚本（`fetch_phira.py` / `extract_features.py`），训练侧唯一的 `DataLoader`（`infra/lightning_module.py`）默认 **0**；真实取批 `ManifestBatchSource.batch()` 在训练进程**主线程里同步**完成「选桶 → 取窗口 → 解析谱面 → 建场 → collate」，而 `step_time_s` 的计时起点就在它之前 ⇒ **数据构建时间一直混在步时里**。

实测（训练进行中）：GPU 利用率**均值 35.7% / 中位 5.0%**（**双峰**：要么满载要么空转）、功耗均值 **21.4 W = 115 W 上限的 18.6%**、训练进程占 **6.45/24** 核；步时中位 **0.539 s** 却 **25.8%（478/1850）的步 ≥1 s**。（上一轮把这条长尾归给「∝(K·T)² 的注意力，**不是**数据侧」——**第十轮实测推翻**：`data_time_s` 的**均值**占墙钟 **45.0%**，长尾正是数据等待。）真正的数据侧瓶颈是**每窗口重解析整张谱面**（一张谱平均 96 窗，≈**46%** 的每窗口 CPU；调大 `chart_cache_size` 实测无效，LRU 命中率封顶 18.6%）。**已落地（本轮）**：步时拆成 `data_time_s` + `compute_time_s`（逐行和恒等于步时）+ `perf/data_share`，巡检脚本打印拆分并在**数据占比 ≥50% 时告警**；护栏 `tests/unit/infra/test_step_time_split.py`（4）。

**实跑证据** —— ★ **门禁 + 全量训练**：`runs/phigros_masked/20260927-100604/`（`gates.txt` + 在线标量 `logs/loss_history.jsonl`：`batch_n_lines` / `batch_events` / `epoch` / `windows_seen` / `charts_seen` / `data_time_s` / `compute_time_s`）。
- `runs/phigros_masked/20260927-085105/`（RFC-0032 语义、采样器未修）：**K 中位 26 / p99 108 / max 123**、峰值 5.67 GiB ⇒ RFC-0032「本线轨 + bf16」在真实全量 K 上成立；其**数据覆盖结论已作废**。第五轮权威门禁 `runs/phigros_masked/20260927-043828/gates.txt`（2305.8 s，全绿）**不受覆盖率缺陷影响**（每次新建 source、只取 1–16 样本）。

**真实数据（`data/processed/`，不入库）**

| 项 | 实测值 |
|----|--------|
| 谱面 / 音频 | **8551** 张 RPE；42.8 GB / 8084 唯一音频；唯一曲目 6807 |
| 判定线 / note | 327 170 条判定线；note 1107 万（Tap 55.7% / 背面 3.11%） |
| **判定线利用率** | **57.8%** 的判定线不承载任何可判定事件（48.9% 零 note + 8.9% 全 `isFake`）⇒ **表演线**（[survey §7.7](docs/knowledges/phira-dataset-survey.md)） |
| 训练窗口 | train **634,952**；**每谱窗口均值 96 / 中位 90 / max 922** |
| 训练清单 | `pairs.json`：train 6750 / val 814 / test 890 / 泛化 658（按曲目切分） |

**四条已通的链**

1. **解码导出 / 训练通路**：λ 场 → `decoder.decode_field` → 合法性后处理 → `io/formats/rpejson` 写出 → 读回；清单 + 特征缓存 → `ChartPairDataset`（窗口化 + 索引缓存 + 行级 LRU）→ `collate_field_batch` → `MaskedFieldModel`（本线轨 + bf16）→ 泊松 NLL → checkpoint（可旋转、可续训）；
2. **真实数据 / 长跑运维通路**：Phira API → 预筛 + 选择性下载 → RPEJSON 解析 → 三级质检 → 带 provenance 的清单 → 真实 MERT 特征 → 窗口数据集；`--resume`（配置 / 门禁 / data_rev 三项校验）→ 在线标量（TB + `logs/loss_history.jsonl`，含覆盖率与**步时拆分**）→ `scripts/training_health.py --watch 7200`。

**未落地 / 未验证**

- **整谱工作被按窗口重复（当前最大的剩余浪费）**：**每个窗口都把同一张谱重解析、重解压一遍**（一张谱平均 96 个窗口）。第十轮量清了**为什么调缓存没用**：计划层跨 **991 个桶**轮转发牌 ⇒ 一张谱的 96 个窗口相隔约 991 个槽位，`chart_cache_size=8` / `feature_cache_size=2` 的**命中率必然为 0**，每个 epoch 同一张谱被完整重解析 **96 次**（解析 44% + 建场 33% + 解压 12%，单窗口 0.85 s）。**这是 RFC-0033/0034 换来覆盖率所付的代价**，调参数绕不开；削它**会动取批顺序语义**（局部性 vs 每步谱面多样性）⇒ **必须先开 RFC**；
- **GPU 与 CPU 的重叠已修并验收（RFC-0034 S1–S4 提交 `cc0e166`；§11 提交见 git log）**：步时中位 **0.539 → 0.099 s（5.4×）**；第十轮把利用率量准后 `data.workers 3 → 8`，GPU 利用率 **44.3% → 75.5%**、`data_share` **46.4% → 22.4%**、步时 mean **0.296 → 0.195 s**。**判据 3b（>50%）首次达标；判据 3a（<20%）22.4% 仍未达标**（尾部 p99 1.68 s：一个 worker 卡一次能拖穿 16 个预取槽）。⚠️ `commit charge` **38.1/40.1 GiB（95%）** ⇒ **N=8 同时是内存上限**；**遮盖退化 `r == 0` 约占 1.6% 的步**（30/1850）：退化为**全事件目标**（非空批、非零梯度——1900 步里 `loss == 0` 的行数为 **0**），即目标在少数步上**漂移**，而 loss 曲线看不出来（plan 07 §9-43）；
- **装饰线旁路**：决策者已定「用额外旁路、标记为后续扩展，等主路线训练完成后再做」；
- **评估入口未接线**（形态已裁定 = 独立 `beatmorph-eval` 子命令，`eval/pipeline.py` 已就绪），待训练产出 checkpoint；**val 路径未实现**：`optim.val_every` 空转 ⇒ **TB 里没有 val 曲线**；`best.pt` 只是「训练损失最优」，**不是模型选择依据**；
- **plan 04 消融臂**（M7 B4 / M9 B3 / M10 B5 / M11）按决策者口径等大规模训练完毕；**B1 的 G3 常数基线口径仍未定**（plan 04 §9-20）；**Lightning 后端真机实跑**未做（`--extra train` 已装齐、env doctor E4 PASS）。

### 下一步

1. ★ **盯本轮全量训练**

   ```bash
   uv run python scripts/training_health.py --experiment phigros_masked --gpu --watch 7200
   uv run tensorboard --logdir runs/phigros_masked
   # 崩溃/中断后原地续跑（不重跑门禁）
   uv run beatmorph-train --config-name phigros_masked --resume latest --device cuda --skip-env-doctor
   ```

   核对：**`coverage/charts_seen` 必须持续上升**（平掉 = 采样器又饱和，立刻查 RFC-0033 的护栏，**不要**靠加步数掩盖）；**`perf/data_share` 要看「按和」的占比，不是中位**——该分布极端重尾，中位 5% 与均值 45% 可以同时为真，**第十轮就是被中位骗过一次**（[RFC-0034 §11](docs/decisions/RFC-0034-data-supply-throughput.md)）；停止判据仍是**功耗塌陷 + 步时放大一个量级**（显存贴顶本身**不是**判据，见 [docs/TRAINING.md](docs/TRAINING.md) §7.5）。
2. ★ **下一刀：削掉「整谱工作被按窗口重复」**（当前最大的剩余浪费，见「未落地」第 1 条）。**必须先开 RFC**——它要动取批顺序的局部性语义（正是 S4 第一版翻车的那条线：局部性 vs 每步谱面多样性），**不得在配置里悄悄改**。动手前先量两件事：① 加深 `prefetch_factor` 能否把 22.4% 的 `data_share` 再压下去（代价是主机内存，而 `commit` 只剩约 2 GiB 余量）；② 谱面级紧凑表示的内存上界（完整 pydantic 谱面 9.79 MB/张 × 991 桶的常驻工作集 = 9.7 GB，**放不下**，所以不能靠「把缓存调大」）。
3. **装饰线旁路（等 1 完成后）**：λ 的线轴改为「承载有效音符的线集合」（train 有效 K 中位 12 / p90 28），装饰线不进模型、不进损失，**导出时原样写回**。
4. **评估接线**：独立 `beatmorph-eval` 子命令（读 `run_dir` + split → `metrics.json` 的 `eval` 分节）——必须在训练产出 checkpoint 之后。
5. **val 路径**：实现验证循环与 `val/loss` 标量，并裁定 `best.pt` 的模型选择口径（plan 07 §4.5 / §9-31）；接通后顺带按「掩码目标实际占比」评估 `r == 0` 退化要不要改遮盖策略（plan 07 §9-43）。
6. **plan 04 消融臂 + B1 的 G3 基线口径**：等大规模训练完毕；口径先裁定。
7. 若之后补抽特征：`uv run python scripts/extract_features.py`（可续跑）→ `uv run python scripts/fetch_phira.py pairs`（索引缓存会因清单/stat 变化自动重建）。

### 未决项（不阻塞 1–7）

| 未决 | 出处 |
|------|------|
| **RFC-0034 §11：削整谱重复要怎么动取批顺序**（局部性 vs 每步谱面多样性）；**N 与 BLAS 线程最优点已实测**（供给 N=3/6/8/10/12 = 2.51/4.42/**6.33**/5.75/6.28 窗口/s，OMP=1 与 2 无差别 ⇒ **N≥8 封顶**）；`np.memmap` 旁路计数的异常退出语义未定 | [RFC-0034](docs/decisions/RFC-0034-data-supply-throughput.md) §11.3/§11.5 |
| **`r == 0` 遮盖退化（≈1.6% 的步）要不要改遮盖策略**（换遮盖单位 / 放宽重掷上限 / 退化窗口降权） | plan 07 §9-43 |
| **训练预算是否从「步数」改成「epoch 数」**；**是否采用谱面分层采样** | RFC-0033 未决定的两件事 / 备选 3 |
| **「谱内连续取批」vs 每步多样性**（同谱相邻窗口高度相关，可能伤 SGD）；全谱级派生缓存的内存上界未算；46% 里 pydantic 校验占多少未拆开测 | plan 02 §9 第八/九轮存疑 |
| **装饰线是否/如何从 λ 线轴去掉**（含无条件生成缺口）——已定「用旁路、后续扩展」，尚未设计 | RFC-0032 §不决定 / [survey §7.7](docs/knowledges/phira-dataset-survey.md) |
| **局部层若被证实需要别线的轨**：回归路径 = 共享 K/V（语义不变、显存 ∝K、计算仍 `K²`） | RFC-0032 备选 1 |
| **val 路径与模型选择口径**（`optim.val_every` 空转；`best.pt` 暂按训练损失） | plan 07 §4.5 / §9-31 |
| **续训的随机数状态**：dropout 流不恢复（数据顺序与遮盖种子仍由 index+seed 派生） | plan 07 §9-29 |
| **契约父线合成**与 A 级证据不一致 ⇒ 含父线谱面的跨线几何计数为近似值（需 contracts-agent） | RFC-0030 §后果-2 |
| **跨谱 batching 的契约缺口**：`FieldBatch` 只带一个 grid，且**没有 `time_mask`**（音频 padding 是伪造静音） | plan 07 §9-13 / plan 02 §9 |
| **B1 的 G3 常数基线口径未定**（泊松闭式不适用 focal；未定之前不得给 B1 报 G3） | plan 04 §9-20 |
| **遮盖重标定口径**：`hidden`(1/r，实现默认) vs 文档字面 `observed`(1/(1-r)) vs 不自洽的 `hidden_doc` | plan 04 §9-6 |
| **mask 泄漏口径**：默认「以事件为锚的 token 区块遮盖 + 空格稀释」偏离 §4.2 字面表述 | plan 04 §9-18 |
| plan 04 报告方法学两项：条件学习的「边际盆地」loss 差监控（R-04-5）、G2 合成任务与训练预算 | plan 04 §9-15 / §9-16 |
| 输出头**直连 skip** 是否写进 BasePlan §3.3 的架构描述 | plan 04 §9-17 |
| eval 三项口径待复核：多重比较校正、相位搜索范围/步长、难度分档边界 | plan 06 §9-1 / §9-3 / §9-5 |
| `N` 的最终取值（默认 128 已裁；仍需碰撞率–N / NLL–N / F1–N 三条曲线；统计函数漏 type 通道待改） | plan 03 §9-7、plan 02 §9-5 |
| 秒↔τ 两个换算点是否合并为一份实现 | plan 02 §9-11、plan 03 §9-13 |
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
