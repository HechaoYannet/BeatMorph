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
> 上次交接：**2026-09-30** ｜ 交接人：主会话（第十九轮：**目标函数到底奖不奖励 τ —— 奖励，占常数基线 42.5%**；
> 而模型连「每条线几个音符」都没打赢 ⇒ 问题不在目标、在**优化落进「只学边际分布」的盆地**；
> 同轮另发现判据被一个不稳的标量（λ 总质量）主导。全链见 plan 07 §9-68/§9-69）

### 当前状态

自检：`ruff check beatmorph tests` ✅ ｜ `mypy beatmorph` ✅（84 files）｜ 新增护栏 5 项全过（`tests/unit/generation/test_head_skip_split.py`）。
⚠️ 本轮只跑了**局部测试**，未跑完整 `pytest -m "not slow and not gpu and not e2e"`。
⚠️ `ruff check .` 会因**未跟踪的** `research/kipphi-rpejson/`（决策者另行加入）报错，与本仓代码无关。

**★★★ 本轮头条：目标函数奖励 τ，模型没在用它 —— 它连「每条线几个音符」都没打赢**

把真值沿各轴做**族最优**边缘化（`runs/_probe_objective_tau.py`：遮盖测度下逐格可分 ⇒ 每族最优成员有闭式解，
全部用**权威损失实现**求值，**不需要模型**）：

| 族（oracle 场） | 除以常数基线 | 含义 |
|---|---|---|
| `line_only` | **0.7196** | 只知道每条线**总共**几个音符 |
| `space_flat` | **0.4171** | τ 全对、空间无结构 |
| `tau_flat` | **0.2559** | 空间全对、τ 无结构 |
| `oracle` | **−0.1687** | 无约束最优 |

⇒ **τ 值 42.46%、空间 58.58%、线 88.83%** ⇒ **「目标函数不奖励 τ」被否定**。而模型 `val_ratio`：
修复臂 **0.8137/0.9030**（4k/8k）、对照臂 0.7740/0.8586、生成制度 0.8787/0.8326、音乐-only 0.8754/0.9116、
cum-skip **0.7636/1.0029** ⇒ **最好的模型也没打赢 `line_only`**：λ 在 (τ,x,s,c) 上**没有任何净有益的偏离**。

**机制**：因子化头写死 `λ = ΔΛ(t)·p_t(x,s,c)/dV`（`field/integrate.factorized_lambda`）⇒ 模型能表达的 τ 结构
只有「每 τ 格的质量 `ΔΛ(t)`」与「`p_t` 随 τ 怎么变」，而 §9-68 量到前者**精确均匀**。`FieldHead` 的实测记录
（plan 04 §9-17）早已写过：**没有直连 skip 时联合训练会停在「只学边际分布」的盆地**；§9-62 把两条 skip
**一起关掉**（理由是 `cell_skip` 的无条件常数空间先验），但那个结论**早于 τ→秒对齐修复（§9-64）**、
从未在修复后的代码上复测。本轮的平坦 τ 剖面、τ 置换代价 ≈ 0、组内 τ AUC 0.50、打不赢 `line_only`
**全是这个盆地的签名**（上一轮「模型的 τ 定位 = 随机」由此解释）。

**⚠️ 判据侧新缺陷：`val_ratio` 主要在抽「总质量」这签**。`val_pred_over_true = Σ∫λdV/Σn`（1.0 = 遮盖测度下
最优校准）在全量 20k run 的 20 个 val 点摆动于 0.08–1.25，`val_ratio` 与之同步 ⇒ **读数方差主要由这一个
标量贡献**。同批复测 11 个 checkpoint（`runs/_probe_mass.py`，**同一批 40 个 val 批**）：cum-skip
0.0015→0.0013→**0.0003**、修复臂 0.9551→0.5085 ⇒ **cum-skip 把 `val_ratio` 顶到 1.0029，破了
`val/ratio < 1` 红线**（依 RFC-0037 R1 该臂扩规模结论作废）。

**⚠️ 预登记判据 1 已否定：三种 skip 开关没有一条救回来**（修复后的代码，4k→8k ratio / 质量）：
无 skip 的四条臂 0.8137→0.9030 / 0.7740→0.8586 / 0.8787→0.8326 / 0.8754→0.9116（质量 0.96→0.51 等）；
**cum-skip 0.7636→1.0029（破线，质量 0.845→0.055）**；**skip-all 0.9734→1.1316（破线，质量 0.314→0.060）**。
⇒ **六种配置没有一条打到 per-line oracle = 0.7196**，而**每条臂的质量都随训练下漂**（8k 时 0.055–0.51）——
「越训练越不听音乐」在判据上的形状不是音频权重掉了，而是**整个输出场在缩**。同轮**可见场基线**（`runs/_probe_visible_baseline.py`，与模型**同一批 40 个 val 批**、const=1.16460）：
「抄可见密度 + 5% 均匀底 + 族最优档位」= **0.7440**，而模型同批 = 0.7767（修复臂@4k）/ 0.6724（长跑 best）/ 1.1765
（cum-skip@4k）⇒ **一条不需要音乐理解的基线就在模型同一区间**。
⚠️ 该探针的 4 个 `sigma_x` 行与 `no_smooth` **逐位相同**（平滑那一轴未生效，原因未定）⇒ 只用 `no_smooth` 行。

**仍然成立**：数据侧全库就绪（8551 RPE / train 634 952 + val 74 889 窗 / 353.6+41.7 GB 缓存）；门禁 G1/G3/G4；
val 512 窗分层集（`f9d9f7d1068c`）；损失按事件归一；τ→秒表按窗口求值与判定线闸门（§9-64/§9-65）。
⚠️ `visible_input`/`track_input`/`head_cell_skip`/`head_cum_skip` 是**新字段** ⇒ 旧 checkpoint 一律拒绝续训（fail-closed）。

### 下一步（按性价比排序，只留仍然有效的）

1. ★ **换方向：开关已排除，下一刀查「输入表示 / 优化」**（skip 三档全试过，没有一条打赢 `line_only` 0.7196，
   而可见场抄写基线就在同一区间 0.744 ⇒ 瓶颈是**模型用不上自己输入里的结构**）。建议：
   ① 补上只开空间 skip 的那一格（`probe_cellskip`，本轮主动停掉，避免无人值守 2 h GPU 作业）；
   ② 用**冻结解码器 + 只训头部与输入投影**（plan 04 §9-17 的手法，已知能立刻到达下界）做对照，
   看能不能至少打赢可见场抄写基线；③ 若仍不能，查输入投影（1280→d 的 `nn.Linear`）与 LR 尺度。
2. ★ **把判据从总质量里解耦 + 查出质量为何下漂**：先报告 `val_pred_over_true` 与「固定质量后的 NLL」，
   再谈 arm 对比与 `best.pt`；质量在**每条臂**上都下漂（机制未定）是当前最大的单点缺陷。
3. **验证 onset 前端**（上轮遗留，唯一抬动天花板的改动，+0.03 AUC）：小切片（`data.max_samples=200`，
   缓存指纹会变、需重建，约 20 min）训一臂再判值不值得全库重提。
4. **回收音乐-only 臂的生成侧**（e2e + 真谱 F1）：不可见线上的 note 会被闸门丢掉，要如实报，别当成模型失败。
5. **plan 05 M5.7**：D1 阈值标定 + thinning 臂的 Hold 端点策略（R3 产物：64 个 Hold 里 63 个零时长）。

### 未决项（不阻塞「下一步」1–5）

| 未决 | 出处 |
|------|------|
| **天花板在哪**：输入表示、优化、还是数据本身（可见场抄写基线 0.744 已在模型同区间）；**质量为何会崩**（机制未定） | §9-69 ④⑦ |
| **onset 前端值不值得全库重提**（+0.03 AUC）；**音频依赖为何在每个臂上都衰减**（4 臂复现，机制未定） | §9-67 ⑤⑥ |
| 「9 维线性打平 40M 模型」只在这一条判据上验证；§9-66 的 top-N F1 上界列；盲训练的迭代退化 | §9-66 存疑 1/2/4 |
| `seconds_position` 是否值得开；跨 `data.workers` 读数不可比（配对实验必须固定）；装饰线**模型侧**旁路未做 | §9-64 ③⑤、§9-65 |
| `val_windows=512` 抽样精度 / `val_every` A/B；**`val.batches()` 前缀不是代表集**（本轮又栽一次） | §9-63、§9-69 |
| RFC-0038/0039 重叠处置；`nll_shuffled_delta` margin；`grad_clip_norm`；G1 下限 0.05 的 per-event 语义 | RFC-0038/0039 §5 |
| 批 K 上限 / RFC-0035 §1 的 91.3% 未更正 / `r==0` 遮盖退化 / 预算按步还是按 epoch | §9-59 / §9-52 / §9-43 |

**一次性探针（⚠️ `scripts/local_*` 与 `runs/_*` 均不入库，清理即丢）**：本轮 = `_probe_objective_tau.py`（**目标函数轴分解，族最优**）、`_probe_mass.py <ckpt...>`（**质量校准 + τ 置换，同一批 val**）、`_probe_visible_baseline.py`；历史见 plan 07 §9-64/§9-66/§9-67。
⚠️ **口径陷阱（本轮第三次踩到）**：`val.batches()` 的前 40 批是**稀疏层**（同一 checkpoint 在全 val 上
`pred/true=0.845`、前 40 批上 0.0015）⇒ 跨 checkpoint 比较**只要固定同一批**就没问题，
**绝不能**把子集读数与全 val 读数混着说。
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
