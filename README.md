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
> 上次交接：**2026-09-30** ｜ 交接人：主会话（第十七轮：**模型的 τ 定位 = 随机**——它学会了
> 「谱面长什么样」（密度 / x / 类型）却**没学会「哪一拍」**；同时更正 §9-66 一条读数、
> 找到唯一抬动天花板的改动（显式 onset 前端）。全链见 plan 07 §9-67）

### 当前状态

自检：`ruff check beatmorph tests` ✅ ｜ `mypy beatmorph` ✅（84 files）｜ `pytest -m "not slow and not gpu and not e2e"` → **1156 passed**（16 deselected）。
⚠️ `ruff check .` 会因**未跟踪的** `research/kipphi-rpejson/`（决策者另行加入）报错，与本仓代码无关。

**★★★ 本轮头条：模型在同一条判定线内对 τ 的排序 = 随机（0.5009）**

把 λ 折成每个 token 的总强度、在 **(窗口,线) 组内**排序（512 窗全 val、422 万 token、4903 正例）：

| 分数 | 组内 AUC |
|---|---|
| **模型 λ（armH@40000）** | **0.5009**（音乐-only@8000 的模型 **0.5021**） |
| 模型 λ 的 τ 边缘（线上求和后广播） | 0.4985 / 0.5004 |
| 真值计数 / 随机 / 常数（度量自检，合成夹具） | 1.0000 / 0.5289 / 0.5000 ✓ |
| **输入里可达的水平**（线性探针，同在 val） | τ 先验 0.5583 ｜ MERT 现状 0.5506 ｜ **onset 0.5792** ｜ τ+onset **0.5818** |

⇒ **模型学到的是「谱面长什么样」（密度、x/类型的边缘），不是「这首歌的音符落在哪一拍」**。
这与 §9-66 完全自洽：`val_ratio` 的增益来自「可见场密度 + 空间边缘」，与「时间落点」是两件事；
R3 产物的 `event F1 0.002` / `timing F1 0.19` 就是它的下游后果。
⚠️ **不要据此说「模型有 bug」**：可达信号本身很弱（0.58 AUC ⇒ 192 个 token 里挑中那一个的概率只是
0.5% → ~2%），对损失贡献极小 ⇒ 「不学它」可能是**理性的**。判「优化失败 vs 信号太弱」需要一个
**把 τ 定位单独做判据**的读数（下一轮第一件事）。

**⚠️ 更正 §9-66 的一条读数**：§9-66 ④ 的「事件轨是最强 τ 定位线索（0.643）」取自 **val 前 128 窗**（= 分层集最稀疏的一端，**非代表集**）⇒ **作废**。代表集（512 窗）：事件轨全 5 通道 **0.5085**、τ 先验 **0.5555**、音频（对齐）**0.5551**、**音频（冻结）0.4983 = 随机**。⇒ 事件轨的价值在**窗/线密度**（组内 AUC 刻意扣掉的那一维），不在 τ 定位；而**冻结那张表把音频的 τ 信息抹到与随机不可区分**（§9-64 修复的最干净证据）。

**★ 唯一抬动了天花板的改动：显式 onset 前端**（`runs/_probe_onset.py`：原始音频 → STFT hop=24000/75=320，与 MERT 帧**同栅格** → 谱通量）：单帧 **0.5792**、τ+起始包络 **0.5818**，比模型现在吃的「MERT 单帧+1 步差分」（0.5506 / 0.5724）高约 **+0.03 AUC**；**换更强的 MERT 读出抬不动**（多尺度包络 0.4997、能量 0.5531、并联 0.5518）⇒ 瓶颈在**表示**，方向 = **给模型加一条 onset 通道**。⚠️ 幅度只 +0.03，且 onset 须从原始音频重算 = 全库新特征提取（与 MERT 提取同量级工程）。

**音乐-only 臂（`visible_input=false` + `track_input=false`，2 项护栏）**：消融**精确**成立（`cond_track_zero = 9.4e-9`），模型**确实转向音乐**（`cond_audio_zero` +0.5426 @4k → 8000 步衰减到 +0.159，而 `val_ratio` 同 时从 0.875 变好到 0.912）⇒ **「越训练越不听音乐」在每一个臂上都复现**（对照 −0.075 / 修复 +1.033 / 盲 +0.30→+0.07 / 音乐-only +0.54→+0.16）。

**仍然成立**：数据侧全库就绪（8551 RPE / train 634 952 + val 74 889 窗 / 353.6+41.7 GB 缓存）；生产配置 `head_skip=false`+`audio_align=true`；门禁 G1/G3/G4；val 512 窗分层集（`f9d9f7d1068c`）；损失按事件归一；τ→秒表按窗口求值与判定线闸门（§9-64/§9-65）。⚠️ `visible_input`/`track_input` 是**新字段** ⇒ 之前存的 checkpoint 一律拒绝续训（fail-closed）。

### 下一步（按性价比排序，只留仍然有效的）

1. ★ **把「τ 定位」单独做成判据**：现在只有「λ 的组内 AUC」这一条手工读数，且它把「线内排序」与
   「密度」混在一起。建议：给定 (窗口,线) 只在该组内取样预测/真值（或直接报组内 AUC + 组内 top-N 命中）
   并写进 val 读数（`beatmorph/eval/val_metrics.py` 已有分层机制可挂）。判据不独立，下一轮仍会看不清。
2. ★ **验证 onset 前端**：先在小切片（`data.max_samples=200`）上把 onset 包络物化（缓存指纹会变，
   需重建该切片的窗口缓存，约 20 min），训练一臂并与同切片基线比**组内 AUC + 生成侧 F1**。
   这是目前唯一有证据的方向（+0.03 AUC），代价是全库重提特征——所以先用小切片判值不值得。
3. **回收音乐-only 臂的生成侧**（e2e + 真谱 F1）：不知道可见性时会有一批 note 落在不可见线上、
   被闸门丢掉，这条要如实报，不要当成模型失败。
4. **plan 05 M5.7**：D1 阈值标定 + thinning 臂的 Hold 端点策略（R3 产物：64 个 Hold 里 63 个零时长）。
5. **评估入口接线**：形态已定＝独立 `beatmorph-eval`（仍未实现）；`runs/_eval_generation.py` 是它的原型。

### 未决项（不阻塞「下一步」1–5）

| 未决 | 出处 |
|------|------|
| **「模型 τ 随机」是优化失败还是信号太弱**（可达 0.58 ⇒ 逐 token 命中率 0.5%→2%） | plan 07 §9-67 ⑦ |
| **onset 前端值不值得全库重提**（+0.03 AUC，工程成本 = 一次全库特征提取） | plan 07 §9-67 ⑤ |
| **音频依赖为何在每个臂上都衰减**（4 个臂都复现，机制未定） | plan 07 §9-67 ⑥ |
| **§9-66 的 top-N F1 上界列**（376 正例、排序头部不稳）仍然只作登记 | plan 07 §9-66 存疑 1 |
| **「9 维线性打平 40M 模型」只在这一条判据上验证** | plan 07 §9-66 存疑 2 |
| 盲训练的迭代退化（8 步 → 1 步）；真做推理制度训练需要按 schedule 变的遮盖通道（数据侧改动） | plan 07 §9-66 存疑 4 |
| `seconds_position` 修好后是否值得开（cross-attn 的两条轴） | plan 07 §9-64 存疑 3 |
| 跨 `data.workers` 的读数不可比（4000 步就分叉）⇒ 配对实验必须固定这一项 | plan 07 §9-64 存疑 5 |
| 装饰线**模型侧**旁路尚未做（只做了导出侧闸门） | plan 07 §9-65 / RFC-0032 |
| `val_windows=512` 抽样精度 / `val_every=10000` A/B | plan 07 §9-63 存疑 1–2 |
| RFC-0038/0039 重叠部分处置；`val/nll_shuffled_delta` margin；`grad_clip_norm`；G1 下限 0.05 的 per-event 语义 | RFC-0038/0039 §5、RFC-0037 §6 |
| 批 K 上限 / RFC-0035 §1 的 91.3% 未更正 / `r==0` 遮盖退化 / 预算按步还是按 epoch | plan 07 §9-59 / §9-52 / §9-43 / RFC-0033 |

**一次性探针（⚠️ `scripts/local_*` 与 `runs/_*` 均不入库，清理即丢）**：本轮 = `runs/_probe_track_channels.py`（逐通道定位力）、`runs/_probe_audio_frontend.py`（更强 MERT 读出）、`runs/_probe_onset.py`（原始音频 onset；**首次运行会把包络落盘 `runs/_onset_cache.npz`**）、`runs/_probe_model_within.py <ckpt> [--blind --no-track]`（**模型自己的组内 AUC 分解**）、`runs/_probe_ceiling.py`、`runs/_eval_generation.py`、`runs/_check_group_auc.py`（度量自检）。历史与本轮其余见 plan 07 §9-64/§9-66 的列表。⚠️ **组内 AUC 的样本口径必须写清**：`val_windows()[:N]` 取到的是**空窗/稀疏层**，§9-66 ④ 正是栽在这里。
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
