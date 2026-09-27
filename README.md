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
> 上次交接：**2026-09-27（第十轮）** ｜ 交接人：主会话（GPU 利用率量准 **44.3%** ⇒ 数据供给有**两级**瓶颈：第一级 worker 数已修，第二级 `DataLoader(in_order=True)` 已定位**未修**）

### 当前状态

自检（默认 CI，无网络 / 无权重 / 无 GPU）：`uv run ruff check . && uv run mypy beatmorph && uv run pytest -m "not slow and not gpu and not e2e"` → **1033 passed**（17 deselected）。

**★ 训练已跑满一轮，GPU 当前空闲**：`runs/phigros_masked/20260927-100604/`（`max_steps=20000` 跑满，退出码 0，checkpoint 16000/18000/20000 各 153 MiB）。门禁 **36.5 min 全绿**（有界切片 `gates.gate_samples=200`，索引秒级命中）：G1 93552.7 → -7.9 ｜ G2 真实 73.68 vs 打乱 228.75（**3.10×**）｜ G3 -24.66 ≤ 2179.76 ｜ G4 154 帧 ≈ 2.05 s × 75 Hz。K 中位 25，峰值显存 4.87 GiB。

**★ 本轮唯一主线：数据供给的两级瓶颈（第一级已修，第二级已定位未修）**

| 级 | 是什么 | 实测 | 状态 |
|---|---|---|---|
| ① | **训练路径原本根本没有 worker 这个旋钮**——`ManifestBatchSource.batch()` 在训练**主线程同步**取批，`step_time_s` 的起点在它之前 | GPU 利用率 **44.3% → 75.5%**、完全空转 42.7% → **9.5%**、步时 mean 0.278 → **0.199 s**、`data_share` 45.0% → **19.0%** | ✅ 已修：`data.workers: 3 → 8` |
| ② | **torch 的 `DataLoader` 默认 `in_order=True`**（`_batches_parallel` 没传）：乱序先到的结果只进 `_task_info`，`_try_put_index` **只在按序交付时调一次** ⇒ **队头一条慢窗口让 8 个 worker 集体等索引** | 只改这一个 kwarg：**6.85 → 9.85-10.08 窗口/s**、worker 占用 60-65% → **100%**、交付 gap 415-523 ms → **9.4 ms**；重尾 p50/p90/p99 = 568/2303/**5891** ms | ⚠️ **已定位，未修**（见「下一步 1」）|

- ⚠️ **上一轮写在这里的结论被本轮推翻两次，两次都是「量具/读法」的错**：① `perf/data_share` 的**中位 5.0%** 被当成「数据侧不是瓶颈」（真实**均值 45.0%**，极端重尾，中位无代表性）；② 本轮先猜「共享内存搬运」是上限，被实验 A 否掉（`ForkingPickler.dumps` 只有 **1.1 KB / 3 ms**，句柄共享零拷贝）。
- **「改成谱面序」已被实测否掉，不要再为它开 RFC**：`same_set` 对照（同一批 480 窗、payload 差 0.15%、只改顺序）**12.64 vs 11.77 = 1.07×**；其收益几乎全部来自消灭「跨 991 桶的整谱重复解析」（解析 296→81、特征 116→18），**同一桶内部再排序没有额外收益**，而且它把**遮盖构造成本抬高 47%**（274→404）。
- **逐项排除**（都不是瓶颈）：盘（并发 1587-1759 MiB/s，需求 ≈490）、解压（npz 压缩比仅 1.09×）、`pin_memory`（开关 6.33/6.34）、主进程 CPU（**0.05 核**）、OMP 线程（1 与 2 无差别）、Defender（本就关闭）。
- **配置里的「N≥8 封顶 / 8 是拐点」是 `in_order` 的产物**：关掉后 W=10/12 到 **11.55 / 12.21** 窗口/s，仍在上行。`configs/phigros_masked.yaml` 已加修正横幅。

**★ 训练期仪表体检：现有 2 万步参数答不了「模型在学吗」**（全文见 plan 07 §9-46）

- **loss 是双峰，不是一条曲线**：**43.8% 空窗（中位 6.5e-4）/ 56.2% 非空（中位 96）**，差 5 个数量级；`loss<1e-3` 的行**全部**是空窗；空窗的 loss **恒等于积分项 ∫λdV**。
- ⇒ **`best.pt` 按训练损失选 = 按「谁抽到最空的窗」选**（实测最小 loss 在 step 6234，`events=0, K=1`）。
- ⇒ 唯一可比的量（非空批 vs **同批**最优常数场基线）在**已验证可还原的** 16,000 步里**完全平坦**：比值中位 **0.834**、四段改善 16.0/20.7/17.2/**13.1%**、`Spearman(step, 比值) = −0.120` ⇒ 只能说「**明显优于平凡常数场约 16%，但优势没有扩大**」。
- `grad_norm` 与 loss 的 log-log Pearson **0.996**（只是 loss 的影子）；`optim.grad_clip_norm=1.0` ⇒ **每一步都被裁剪**，AdamW 下近乎安慰剂。
- **在线材料里没有任何「音频条件是否被用上」的证据**（G2 用同线未遮盖的事件轨就能过，不必用音频）。
- ⚠️ **分析 `logs/loss_history.jsonl` 前必读**：21,100 行 / 20,000 step，**1100 个 step 有两条副本**（4001-4650 数据不同；**16001-16450** 同数据、权重已分叉）。正确口径 = 按 step 去重 + **整段丢弃 1–4000**（旧采样器，`charts_seen` 一致率仅 0.9%）。

**真实数据（`data/processed/`，不入库）**

| 项 | 实测值 |
|----|--------|
| 谱面 / 音频 | **8551** 张 RPE；42.8 GB / 8084 唯一音频；唯一曲目 6807 |
| 判定线 / note | 327 170 条判定线；note 1107 万（Tap 55.7% / 背面 3.11%） |
| **判定线利用率** | **57.8%** 的判定线不承载任何可判定事件（48.9% 零 note + 8.9% 全 `isFake`）⇒ **表演线**（[survey §7.7](docs/knowledges/phira-dataset-survey.md)） |
| 训练窗口 | train **634,952**；每谱窗口均值 96 / 中位 90 / max 922 |
| 训练清单 | `pairs.json`：train 6750 / val 814 / test 890 / 泛化 658（**按曲目切分**） |
| **1 epoch** | **634,952 步**；按 0.199 s/步 ≈ **35 h** ⇒ 20,000 步只覆盖 **3.15%** 的窗口（谱面覆盖已 **100%**） |

**四条已通的链**

1. **解码导出 / 训练通路**：λ 场 → `decoder.decode_field` → 合法性后处理 → `io/formats/rpejson` 写出 → 读回；清单 + 特征缓存 → `ChartPairDataset`（窗口化 + 索引缓存 + 行级 LRU）→ `collate_field_batch` → `MaskedFieldModel`（本线轨 + bf16）→ 泊松 NLL → checkpoint（可旋转、可续训）；
2. **真实数据 / 长跑运维通路**：Phira API → 预筛 + 选择性下载 → RPEJSON 解析 → 三级质检 → 带 provenance 的清单 → 真实 MERT 特征 → 窗口数据集；`--resume`（配置 / 门禁 / data_rev 三项校验）→ 在线标量（TB + `logs/loss_history.jsonl`）→ `scripts/training_health.py --watch 7200`。

**★ 调研报告与一次性探针（⚠️ 都在 `runs/_research/` 与 `scripts/local_*`，两者**均不入库**，清理 `runs/` 就会丢）**

| 路径 | 内容 |
|---|---|
| `runs/_research/q3_supply_ceiling.md` | **共享上限定位**（`in_order=True`）全文 + 实验 A 原始表 + 11 条存疑 |
| `runs/_research/q3_*.md` 之外：`runs/_research/q1_cache_ram.md` | 「扩大缓存换 RAM」的否证：18.12 MB/张、50% 命中率需 **72.5 GiB/worker**、epoch 内命中率上限 **91.67%**、「共享」只对字节成立 |
| `runs/_research/q2_training_diagnostics.md` | 训练期仪表体检全文（上文那一段）+ 9 条存疑 |
| `runs/_research/probes/q3_*` / `q2_*` | 探针脚本与原始 JSON |
| `scripts/local_worker_scaling.py` | worker 数 → 供给标定（**注意：其「N≥8 封顶」是 `in_order` 的产物**）|
| `scripts/local_order_throughput.py` / `local_order_worker.py` | 取批顺序的吞吐对照（含 monkeypatch 先例）|
| `scripts/local_build_split.py` | 物化成本的函数级拆分（全谱 91.3% / 逐窗口 9.9%，cProfile 口径偏高）|
| `scripts/local_gpu_duty.py` / `local_spy_dump.py` | 占空比采样（**必须用 Popen 逐行读**）与主线程快照（**只能用 dump，record 会丢样本**）|
| ⚠️ `scripts/local_chart_mem.py` | **坏了**：`deep_bytes` 漏了 pydantic `__slots__` ⇒ 读数偏低 **1.85×**（真值 18.12 MB/张）。用前先修 |

**未落地 / 未验证**

- **第 1 层未做**（见「下一步 1」）：`in_order=False` 会把供给从 6.85 抬到 ~10 窗口/s，但**会让「槽位 ↔ 批」错位**，必须先让批自带槽位号；
- **第 2 层未剖析**：物化侧**遮盖构造 274-404 ms**（占单窗口 ~34%）是本轮唯一看不懂的大头；
- **val 路径未实现**：`optim.val_every` 空转 ⇒ **TB 里没有 val 曲线**；`best.pt` 只是「训练损失最优」，**不是模型选择依据**（且如上所述，那个损失是噪声）；
- **评估入口未接线**（形态已裁定 = 独立 `beatmorph-eval` 子命令，`eval/pipeline.py` 已就绪），待训练产出 checkpoint；
- **装饰线旁路**：决策者已定「用额外旁路、标记为后续扩展」；**plan 04 消融臂**（M7/M9/M10/M11）按决策者口径等大规模训练完毕；**Lightning 后端真机实跑**未做（`--extra train` 已装齐、env doctor E4 PASS）；
- **内存是硬上限**：`commit charge` 已达 **35.8/37.0 GiB**（页文件上限自己在长到 45 GiB），W=12 时可用物理最低 7.4 GiB ⇒ **`in_order` 修好之前加 worker 只吃内存不涨吞吐**。

### 未决项（不阻塞 1–7）

| 未决 | 出处 |
|------|------|
| ~~取批顺序改成「谱面序」~~ **已实测否掉**（`same_set` 对照 **1.07×**）；~~N 最优点~~ **已实测，但「N≥8 封顶」是 `in_order` 的产物**（关掉后 W=10/12 = 11.55/12.21 仍上行）；**`np.memmap` 旁路计数的异常退出语义**未定 | plan 07 §9-49 / RFC-0034 §11.5 |
| **`r == 0` 遮盖退化要不要改遮盖策略**（换遮盖单位 / 放宽重掷上限 / 退化窗口降权）；口径已复核为「**≈1.0% 的全部步 / ≈1.8% 的非空批**」，且 **43.8% 的步天然就是 `r==0`**（空窗合法） | plan 07 §9-43 |
| **训练预算是否从「步数」改成「epoch 数」**；**是否采用谱面分层采样** | RFC-0033 未决定的两件事 / 备选 3 |
| **「谱面序」对 SGD 的实际影响**：已量出监督信号 **lag-1 自相关 +0.024 → +0.426**、相邻步同谱 0.2% → **94.6%**、空窗连续段最长 17 → 31，但**没有 val 可以把它换算成性能损失**；全谱级派生缓存的内存上界已实测（不可行） | plan 07 §9-47/§9-48/§9-49 |
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
