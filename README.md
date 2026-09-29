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
> 上次交接：**2026-09-30** ｜ 交接人：主会话（第二十轮：**找到训练侧的主因——取批计划的「每轮每谱第 k 窗」
> 把训练分布换成了前奏切片：前 8000 步 61.5% 空窗 / 4.40 事件每窗，而全库是 10.5% / 13.46**；
> 本项目跑过的最长 run = 40 000 步 = **6.3% 个 epoch** ⇒ 至今没有一次训练跑在语料的无偏切片上。
> **修法已实测有效**：训练分布回到语料、`val_ratio` **首次随训练变好**（0.7857→**0.7324**，旧六条臂
> 全部变差、最好 0.8137）。同轮把问题拆成两半：**质量/校准问题已解决**，**τ 定位问题仍未解决且已排除
> 「训练分布」这一项**。全链见 plan 07 §9-69/§9-70）

### 当前状态

自检：`ruff check beatmorph tests` ✅ ｜ `mypy beatmorph` ✅（84 files）｜ `tests/unit/data`+`tests/unit/infra` **508 passed**。
⚠️ 本轮只跑了定向测试，未跑完整 `pytest -m "not slow and not gpu and not e2e"`。
⚠️ `ruff check .` 会因**未跟踪的** `research/kipphi-rpejson/`（决策者另行加入）报错，与本仓代码无关。

**★★★ 本轮头条：训练分布 ≠ 语料分布（差 6 倍空窗 / 3 倍密度），根因是计划的发牌顺序**

| 口径 | 空窗占比 | 事件/窗均值 | 中位 |
|---|---|---|---|
| **步级（训练日志，12 个 run 逐位相同）** | **0.615** | **4.40** | 0.0 |
| 窗口级（计划均匀抽样） | 0.105 | 13.46 | 12.0 |

**机制**：`plan_epoch`（RFC-0034 的 S4）**按轮次发牌**——第 k 轮每张谱发它的第 k 块 ⇒ 前 6 614 个槽位
只含每张谱的**第 0 窗**（真实谱面的前奏空窗）。前缀画像逐窗实测：500 步 47.6% 空窗 / 5.20 事件、
8000 步 **61.5% / 4.40**、20000 步 52.6% / 5.38、60000 步 35.1% / 7.68，而全库是 10.5% / 13.46。
整个 epoch = 634 952 窗 = **96 轮**，而项目里最长的一次训练是 40 000 步 = **6.3% 个 epoch**。
空窗在遮盖测度下（r=0 退化支）的最优 λ **恰为 0** ⇒ 每步都在把输出推小。这**同时解释**了
§9-69 的「每条臂的质量都下漂」、§9-68 的「τ 完全平坦」、以及「越训练越不听音乐」。
**旁证**：门禁批的事件数从 63（旧采样）变成 **304**（同一配置、只开修法）。
**修法（`data.plan_window_shuffle`，默认 false）**：每轮每谱发它自己的**随机**一块，轮转结构与覆盖率
**逐条不变**；实测前 2000 窗从 0.496 空窗 / 5.85 事件变为 **0.114 / 17.39**（全库 0.105 / 13.46）。
**臂已跑完（8000 步）**：步级 0.122 空窗 / 14.26 事件，`val_ratio` **0.7857 → 0.7324（变好）**，
质量 1.147 → 0.523（旧：0.955 → 0.509 / 0.845 → 0.055），覆盖率 87.7% 不变，**8k 预算下全项目最好读数**。

**判据侧独立证据**：11 维线性回归（只用可见场的多尺度 τ 剖面，**不训练网络**）在 (窗口,线) 组内
τ AUC = **0.5723**，**打赢 40M 参数的模型（0.5009）**；但它的 `val_ratio` 只到 **0.7294**
（`line_only` 0.7196、`space_flat` 族最优 0.4171）⇒ τ 定位信息**很便宜但买不到多少损失**，
大头在空间轴 ⇒ 矛头指向**训练分布**而不是表达能力。

**剩余问题 = τ 定位**：修法后 τ 置换 Δ 仍是 **−2e-5**（判据 ③ 不成立）⇒ 训练分布**不是**它的原因。
同样的输入，11 维线性回归能做到组内 AUC **0.5723** / `val_ratio` 0.7294，模型是 **0.5009 / 0.7324**
⇒ 只剩「输入表示」与「优化路径」两个候选。

**skip 三档矩阵已补齐**（旧采样、8000 步、const = 3.94133）：无 skip 四条臂 0.8137→0.9030 / 0.7740→0.8586 /
0.8787→0.8326 / 0.8754→0.9116；**cum-skip 0.7636→1.0029、skip-all 0.9734→1.1316、cell-skip →1.0830（三条都破线）**
⇒ 六种配置没有一条打到 per-line oracle = 0.7196。⚠️ **这六条臂训练在偏置切片上 ⇒ 作为「模型对比」全部作废**。

**仍然成立**：数据侧全库就绪（8551 RPE / train 634 952 + val 74 889 窗 / 353.6+41.7 GB 缓存）；门禁 G1/G3/G4；
val 512 窗分层集（`f9d9f7d1068c`）；损失按事件归一；τ→秒表按窗口求值与判定线闸门（§9-64/§9-65）。
⚠️ 本轮新增字段 `data.plan_window_shuffle` ⇒ **旧 checkpoint 再次 fail-closed 拒绝续训**（预期行为）。

### 下一步（按性价比排序，只留仍然有效的）

1. ★ **修法已实测有效（§9-70⑥）：默认打开 `data.plan_window_shuffle`**——把配置里的 `false` 改成 `true`
   并记录 RFC-0034 §9 决策（这是取批顺序的语义变更）。依据：训练分布 0.615→0.122 空窗 / 4.40→14.26
   事件、`val_ratio` **首次随训练变好** 0.7857→0.7324（旧六条臂全部变差、最好的只有 0.8137）。
2. ★ **τ 定位仍未解决，且训练分布这一项已被排除**（§9-70⑦：shuffle 后 τ 置换 Δ 仍 ≈ −2e-5）。
   下一刀是 §9-69⑥ 的**冻结解码器 + 只训头部与输入投影**对照：同样的输入，11 维线性回归能做到
   组内 AUC 0.5723，模型却是 0.5009 ⇒ 要把「表示」与「优化路径」分开。
3. ★ **所有旧臂的「模型对比」作废**（它们训练在偏置切片上），关键臂（无 skip / cum-skip / 音乐-only）
   必须在修法下重跑；`best.pt` 口径也要重新标定。
4. **把判据从总质量里解耦**：先报 `val_pred_over_true` 与「固定质量后的 NLL」，再谈 arm 对比。
5. **验证 onset 前端**（上轮遗留，唯一抬动天花板的改动，+0.03 AUC）：小切片训一臂再判值不值全库重提。
6. **plan 05 M5.7**：D1 阈值标定 + thinning 臂的 Hold 端点策略（R3 产物：64 个 Hold 里 63 个零时长）。

### 未决项（不阻塞「下一步」1–6）

| 未决 | 出处 |
|------|------|
| **修法只把前缀变无偏，epoch 内「每轮各谱一块」的池级相关性仍在**；可见场抄写基线 0.744 其实是「均匀底」在起作用 | §9-70 ④、§9-69⑦ |
| **onset 前端值不值得全库重提**（+0.03 AUC）；**音频依赖为何在每个臂上都衰减**（4 臂复现） | §9-67 ⑤⑥ |
| 「9 维线性打平 40M 模型」只在这一条判据上验证；§9-66 的 top-N F1 上界列；盲训练的迭代退化 | §9-66 存疑 1/2/4 |
| `seconds_position` 是否值得开；装饰线**模型侧**旁路未做；`val_windows` 抽样精度 / `val_every` A/B | §9-64 ③、§9-65、§9-63 |
| RFC-0038/0039 重叠处置；`nll_shuffled_delta` margin；`grad_clip_norm`；G1 下限 0.05 的 per-event 语义 | RFC-0038/0039 §5 |
| 批 K 上限 / RFC-0035 §1 的 91.3% 未更正 / `r==0` 遮盖退化 / 预算按步还是按 epoch | §9-59 / §9-52 / §9-43 |

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
