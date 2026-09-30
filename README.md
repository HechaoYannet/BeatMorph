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
> 上次交接：**2026-09-30** ｜ 交接人：主会话（第二十八轮：**τ 通道的真因被量死并修好了**——
> 语料 τ 表（192 格常数表）在真实 val 上组内 AUC **0.9292**，而模型是 **0.5009（精确并列）**；免拟合定位
> 把根因钉死：τ 位置只在**输入** token 上，训练把三个 local 层推到「沿 τ 不变」（rel 2.064 → 0.00000），
> 头没有第二条 τ 通路 ⇒ 「学一张 τ 表」在装配上不可达。修法 = `model.head_tau_bias`（**192 个参数**）⇒
> τ AUC **0.9522**、`val_ratio` **0.6115**（历史最好 0.6838，且首次 4k→8k 继续下降）。**同轮作废** §9-75 的
> 「天花板 0.5753 / 先验 0.5555」（探针 `pos` 特征周期写错）。全链见 plan 07 §9-77 / §9-78）

### 当前状态

自检：`ruff check beatmorph tests` ✅ ｜ `mypy beatmorph` ✅（84 files）｜ `pytest -m "not slow and not gpu and not e2e"` → **1185 passed**。
⚠️ `ruff check .` 会因**未跟踪的** `research/kipphi-rpejson/`（决策者另行加入）报错，与本仓代码无关。

**★★★ 本轮头条：τ 通道的真因被量死并修好了 —— 组内 τ AUC 0.5009 → 0.9513，`val_ratio` 0.6890 → 0.5571**

| 口径（**被遮盖 token 内部**组内 τ AUC） | AUC |
|---|---|
| **语料 τ 表**（train 直方图 = 192 格常数表，零参数） | **0.9292** |
| 1/4 拍指示器（12 格步长） | 0.8312 |
| **模型 `probe_taubias`（+ 头 τ 偏置）** | **0.9531 / 0.9522 / 0.9513**（4k / 8k / 16k） |
| 模型 `probe_masklam` / `probe_block` / armH | 0.4992 / 0.5198 / 0.5130（**精确并列**） |

⇒ **真因**（plan 07 §9-77）：τ 位置信息只加在**输入** token 上，训练把三个 local 自注意力层推到「沿 τ 不变」
（免拟合 rel：未训练 **2.064** → 训练后 **0.00000**），而 `head_skip=false` 时头只剩解码器输出
⇒ **「学一张 τ 表」在当前装配下不可达**。**修法** = `model.head_tau_bias`（**192 个参数**，加在头的 logits 上、
softplus **之前**）：`val_ratio` **0.6397 @4k → 0.6115 @8k → 0.5927 @12k → 0.5571 @16k**
（历史最好 0.6838；**首次**出现单调下降的 val 轨迹，两条预登记判据全过），τ AUC 四点稳定 0.9513–0.9531
（**高于语料表本身** ⇒ 学到的是**条件**排序，不是背表）。提交 `d7f429b`，护栏 5 项。
⚠️ τ 那半已饱和（AUC 不随步数变），12k→16k 的继续下降来自**空间/质量轴**。
⚠️ **同时作废一条旧结论**：§9-75 的「输入天花板 0.5753 / 纯 τ 先验 0.5555」是探针 `pos` 特征**周期写错**（`sin(2π·tau_f·4)`
的周期是 **1 拍**而不是 1/4 拍）造的假天花板 —— 0.5538/0.5583/0.5753/0.5792/0.5818 只说明那条 5 维特征族不行，**不说明输入里没有 τ 信息**。

**基线阶梯**（自检 `line_only` 复现 0.7196）：`mask_pure`（形状与档位都只用输入）= **0.7594**
｜ `line_only`（oracle 每线总量）= 0.7196 ｜ `mask_event`（形状+档位都 oracle）= 0.6067 ｜
`space_flat`（τ 完美、空间均匀）= 0.4171 ｜ `tau_flat`（空间完美、τ 常数）= **0.2559** ｜ oracle −0.1687。
⇒ **空间轴的净值（58.58%）仍大于 τ 轴（42.46%）**，本轮只修了 τ（§9-78⑤）。

**★ 数据侧「去表演」口径：已落地，但实测是负结果**（§9-79/§9-80）。目标里 **22.81%/18.68%**（train/val）的事件是
「命中时线不可见」的表演 note（一半在**真线**上 ⇒ 线级剥离拿不掉），线轴 83% 是装饰线（K 中位 26 → 真线 **5**）；
按决策者裁定 train/val 同口径去掉全部表演成分（`gameplay_subchart`：真线∪祖先 + 只留可计分 note）。
**但等步数对照的收益在噪声之内**（12k：0.5872 vs 0.5808，Δ=0.0064 « 本底 0.051；16k 去表演反弹到 0.6815）
⇒ **表演 note 不是「学不会音乐」的主因**。同轮附带 **索引并行化**（`data.index_jobs`，**6.1×**、逐位一致）。

**★ 零参数检验（`runs/_probe_embedding_sep.py`）**：遮盖信号在**解码器输入**处跨半边 AUC **1.0000**、在**解码器输出**处
**0.5057**（头输出还剩 0.6546）；输入表示的尺度失衡 **134 倍**（`visible` 0.2809 vs `occlusion` 37.6518）；遮盖逐 token 全有全无（部分遮盖 0.0000%）。

**★ 方法学**：**同配置重跑差 0.051** ⇒ 单跑的 0.0x 级差不可作结论；val 序列常非单调 ⇒ 以 `best.pt` 判读。
τ 那条结论（−0.077 / +0.43 AUC）远在噪声之外；去表演那条（−0.0064）**在里面**。

**仍然成立**：数据侧全库就绪（8551 RPE / train 634 952 + val 74 889 窗 / 353.6+41.7 GB 缓存）；门禁 G1/G3/G4；
val 512 窗分层集（`f9d9f7d1068c`）；损失按事件归一；τ→秒表按窗口求值与判定线闸门（§9-64/§9-65）；
`data.plan_window_shuffle`（训练分布回到语料）；`model.mask_lambda`（0.6890 → 0.6115 的一半功劳）；
**`model.head_tau_bias`（本轮修法，192 参数）**；`optim.train_head_only`（诊断开关）。
⚠️ 已知：`plan_window_shuffle` 只改取窗顺序却进了窗口缓存指纹 ⇒ train 缓存对不上、退回原路径（步时 ~0.5 s）；旧的四条
无-skip 臂 + 三条 skip 臂**都训练在偏置切片上 ⇒ 作为「模型对比」全部作废**；新增字段使旧 checkpoint 拒绝续训（fail-closed）。

### 下一步（按性价比排序，只留仍然有效的）

1. ★ **去表演口径的下一步只有一条**（§9-80④）：补一条「去表演但**保留全部线轴**（λ 在装饰线上恒为 0）」的臂——
   现有实验把「去掉噪声」与「换了输入分布（K 26→5）」混在一起，不可分。
2. ★ **空间轴照 τ 的办法修一遍**（`model.head_cell_tau_bias` 已实现、**未验证**）：`gap_space` = 58.58%
   > `gap_tau` = 42.46%，而 `tau_flat`（完美空间 + τ 常数）= 0.2559 ⇒ **空间的净值比 τ 还大**，
   本轮只修了 τ。**预登记判据（跑之前写死）**：`val_ratio` **< 0.55**（在 0.6115 上再要一成）
   + 空间轴命中率上升。臂 = `probe_taubias` + `model.head_cell_tau_bias=true`。
2. ★ **把「被遮盖内部组内 τ AUC」做成常驻读数**（现在只有手工探针）：本轮证明它比 `val_ratio` 灵敏得多
   （τ AUC 0.5009 → 0.9522 时 `val_ratio` 只动了 0.077），而 `val_ratio` 被总质量与未遮盖处的
   浪费稀释（§9-74②）。接进 `val/` 指标。
3. **臂对比的新纪律**：任何「A 比 B 好 0.0x」都必须给**至少 2 次同配置重复**；跨口径比较必须用
   `runs/_probe_val_ratio.py` 在同一套 val 窗口、同一步数上复算（本轮的对照就是这么做的）。
4. **判据解耦**：先报 `val_pred_over_true` 与「固定质量后的 NLL」，再谈 arm 对比与 `best.pt`。

### 未决项（不阻塞「下一步」1–5）

| 未决 | 出处 |
|------|------|
| **local 层把 τ 抹平的机制**（未定）：attention 低通平均 vs 条件注入主导 —— 已量到现象（rel 2.06 → 0.00000） | §9-77④ |
| **`visible`/`occlusion` 的 134 倍尺度失衡**要不要修（归一化 / 换输入表示）；**遮盖逐 token 全有全无 ⇒ 被遮盖 slot 的空间信息在输入里不存在** | §9-72②、§9-71⑥ |
| **同配置重跑差 0.051** ⇒ 历史单跑臂对比的哪些结论需要重估 | §9-72④ |
| **onset 前端值不值得全库重提**（旧 +0.03 量的口径已作废，要重测）；**音频依赖为何在每个臂上都衰减**（4 臂复现）；§9-66 的「9 维打平 40M / top-N F1 上界」随 §9-77③ 一并作废、要重算 | §9-67 ⑤⑥、§9-77③ |
| `seconds_position`；装饰线**模型侧**旁路；`val_windows` 抽样精度 / `val_every` A/B；RFC-0038/0039 重叠处置、`nll_shuffled_delta` margin、`grad_clip_norm` | §9-64③、§9-65、§9-63、RFC-0038/0039 §5 |

**一次性探针（⚠️ `scripts/local_*` 与 `runs/_*` 均不入库，清理即丢）**：本轮 = `_probe_grid_snap.py`/`_probe_window_phase.py`（语料 τ 表 0.9292）、
`_probe_tau_layer_trace.py`（逐层定位）、`_probe_line_strip.py`/`_probe_scorable_share.py`（去表演规模）、`_probe_index_equiv.py`（并行索引逐位一致）、`_probe_val_ratio.py`（跨口径复算）；历史见 §9-70～§9-76。
⚠️ **口径陷阱**：① `val.batches()` 前 40 批是稀疏层（跨 checkpoint 比较必须固定同一批）；
② 罕见标签上拟合高维读出会欠拟合（§9-72①，`pos` 特征族那次就是被它掩盖的）；
③ **探针不要一律丢 CPU**：真实批的全局层是 `O((K·T)²)`（K=104 ⇒ 25.5 GB）⇒ **fp32/CPU 会 OOM、fp32/GPU 会溢出 8 GB
掉进 Windows 共享显存（GPU 空转、任务"还在跑"）**；正解 = 与训练同口径 `autocast_context(...)` + GPU + **同一时刻只跑一个作业**（§9-77⑧）。
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
