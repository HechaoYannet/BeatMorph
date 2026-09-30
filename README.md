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
> 上次交接：**2026-09-30** ｜ 交接人：主会话（第二十四轮：**模型的 λ 对遮盖通道毫无响应**——质量 44.4% 落在
> 被遮盖 token（占比 44.2%），**与 token 数成正比**，而最优解要求未被遮盖处 λ 恒为 0；
> 把 AUC 收窄到**真正被监督的那批 token**，组内 τ AUC 只有 **0.5130**。同轮补齐**完全无 oracle** 的基线：
> 一条两行的输入启发式拿到 **0.7594**，而模型的**最好**读数是 0.7287 ⇒ 只比它好一点点。
> 据此新增 `model.mask_lambda`（λ←λ⊙occlusion）并在门禁处抓到一处语义错误。全链见 plan 07 §9-73/§9-74）

### 当前状态

自检：`ruff check beatmorph tests` ✅ ｜ `mypy beatmorph` ✅（84 files）｜ `pytest -m "not slow and not gpu and not e2e"` → **1167 passed**。
⚠️ `ruff check .` 会因**未跟踪的** `research/kipphi-rpejson/`（决策者另行加入）报错，与本仓代码无关。

**★★★ 本轮头条：模型的 λ 对遮盖通道零响应，且在被监督集合内 τ 仍是随机**

| 读数（`runs/_probe_mask_response.py`，armH 的 8k 快照） | 值 |
|---|---|
| λ 质量落在**被遮盖** token | **44.4%**（token 占比 44.2% ⇒ 与 token 数成正比、对遮盖零响应） |
| 最优解要求 | 未被遮盖处 λ **恒为 0** ⇒ 模型把 55.6% 的质量花在零回报处 |
| **被遮盖 token 内部**（= 真正被监督的子集）组内 τ AUC | **0.5130**（未被遮盖内部 0.4853） |

**基线阶梯（档位用族最优闭式；自检 `line_only` 复现 0.7196）**：`mask_pure`（**形状与档位都只用输入**，
两行启发式）= **0.7594** ｜ `line_only`（oracle 每线总量）= 0.7196 ｜ `mask_plus_visible` = 0.6793 ｜
`mask_ratio` = 0.6867 ｜ `mask_event`（形状+档位都 oracle）= 0.6067 ｜ oracle 族 0.4171 / 0.2559。
⇒ **模型最好 0.7287 只比「两行输入启发式」好一点点**，离 oracle 族极远。

**★ 新开关 `model.mask_lambda`（λ ← λ ⊙ occlusion）**：既然最优解在未被遮盖处恒为 0，这是一条
**不引入新参数、不改损失语义**的硬约束。首版被**门禁**拦下（退出码 7「lambda 出现 NaN」）：G3 的
**全可见**批走全事件口径、置零会把事件项打成 +inf ⇒ 加 `occlusion.any()` 守卫（全可见退回恒等，
全遮盖下乘法恒等，两条边界对称）。护栏 4 项。**臂 `probe_masklam` 本轮在跑，判据已预登记（§9-74③）**。

零参数检验（`runs/_probe_embedding_sep.py`：方向 = 遮盖/全可见两组 token 的均值差，**只用前一半 val 估**，
在**后一半**算 AUC；标签 = 「本 token 是否全遮盖」，30% 正例）：

| 一级 | 跨半边 AUC |
|---|---|
| L1 输入特征 / `occlusion` 分量 / LN 之前 / **LN+音频之后（= 解码器输入）** | **1.0000**（四者全同） |
| **L2 解码器输出** | **0.5057** |
| L3 头输出（每 token λ 质量） | 0.6546 |

⇒ **输入里完美可分 → 解码器输出随机**。§9-71⑤ 的「崩塌在 embedding（0.4790）」是**岭回归拟合稀有标签
（正例 0.03%）欠拟合的产物，已作废**。头输出仍有 0.65 的遮盖意识 ⇒ 解码器是**非线性地混掉**了它。

**★ 决定性对照：只读遮盖通道的场拿到 `val_ratio = 0.6067`**（`mask_event`：1[全遮盖]×(可见计数+1)+可见计数，
档位用族最优闭式；自检 `line_only` 复现 0.7196 逐位一致）⇒ **零音乐理解的场打赢每一个模型**（0.73–0.90）。
**模型连「照着遮盖放质量」都没做到，卡点在解码器。**

**★ 仍然成立（上轮）**：遮盖信号在**解码器输入**处零参数跨半边 AUC **1.0000**、在**解码器输出**处 **0.5057**；
输入表示两个数量级尺度失衡（`visible` 0.2809 vs `occlusion` 37.6518）；同配置重跑差值 **0.051**；
24k 的 val 序列是 **0.8005 / 0.7287 / 0.7890（非单调）**；head-only 最好的 0.7457（质量 1.84→0.30）。

**★ 方法学：同配置重跑的差值达 0.051 val_ratio**。`probe_shuffle24k` 因卡死重跑，两次 `config.yaml`
逐字段相同、步级分布相同，step 4000 的 `val_ratio` 分别是 **0.7492 / 0.8005** ⇒ **单次跑的 0.0x 级差
不可作为结论**（今后臂对比至少要 2 次重复）。⚠️ 这不推翻 §9-70⑥：修法的证据是**量级性**的
（空窗 0.615→0.122、质量从崩溃变稳定），但其中 `0.7324 vs 0.8586` 那个具体差值被噪声污染。

**★ 24k 长跑（第二次，⚠️ 已更正）**：4k = 0.8005 → 8k = 0.7287 → **12k = 0.7890** ⇒ **不是单调改善**，
结合同配置重跑差值 0.051，正确说法是「**在 0.73–0.80 的带里波动**」；而旧采样的八条臂在 8k 处集体进入
**0.83–1.13** 并伴随质量崩溃 ⇒ **修法的证据是量级性的（分布 / 质量），不是趋势性的**。

**仍然成立**：数据侧全库就绪（8551 RPE / train 634 952 + val 74 889 窗 / 353.6+41.7 GB 缓存）；门禁 G1/G3/G4；
val 512 窗分层集（`f9d9f7d1068c`）；损失按事件归一；τ→秒表按窗口求值与判定线闸门（§9-64/§9-65）；
`data.plan_window_shuffle` 修法（训练分布回到语料）。⚠️ 遮盖是**逐 τ token 全有全无**（部分遮盖 0.0000%）。
⚠️ 旧的四条无-skip 臂 + 三条 skip 臂**都训练在偏置切片上 ⇒ 作为「模型对比」全部作废**。

### 下一步（按性价比排序，只留仍然有效的）

1. ★ **读 `probe_masklam` 的判据（本轮在跑）**：① `val_ratio ≤ 0.72`；② 质量 100% 落在被遮盖处
   （构造保证，用 `runs/_probe_mask_response.py` 复核）；③ **被遮盖内部 τ AUC > 0.55**。
   ③ 是关键分叉：**若仍 ≈0.51，说明「浪费质量」只是并发症**，主因在别处（那时才轮到改损失/参数化）。
2. ★ **修表示尺度**：`visible` 支路比 `occlusion` 小 134 倍。最便宜的候选：给两支各自的**归一化**
   （per-channel / per-token L1），或把稀疏计数换成多尺度密度。**改一个就跑一条 8000 步臂 ×2 次重复**。
3. ★ **把「被遮盖内部 τ AUC」做成常驻读数**（现在只有手工探针）：它是唯一直接对应「损失真正监督的
   那一批 token」的指标；`val_ratio` 被总质量与未遮盖处的浪费稀释，单独看它会误判（§9-74②）。
4. **臂对比的新纪律**：任何「A 比 B 好 0.0x」都必须给**至少 2 次同配置重复**，否则只登记不作结论。
5. **判据解耦**：先报 `val_pred_over_true` 与「固定质量后的 NLL」，再谈 arm 对比与 `best.pt`。

### 未决项（不阻塞「下一步」1–5）

| 未决 | 出处 |
|------|------|
| **解码器抹掉遮盖信号的机制**（未定）；**遮盖感知场值多少**（探针在跑，前几版栽在档位校准上） | §9-72①⑤ |
| **同配置重跑差 0.051** ⇒ 历史单跑臂对比的哪些结论需要重估 | §9-72④ |
| **`visible`/`occlusion` 的 134 倍尺度失衡**要不要修、怎么修（归一化 or 换输入表示） | §9-72② |
| **遮盖逐 token 全有全无 ⇒ 被遮盖 slot 的空间信息在输入里不存在**（任务更像生成而非补全） | §9-71⑥ |
| **onset 前端值不值得全库重提**（+0.03 AUC）；**音频依赖为何在每个臂上都衰减**（4 臂复现） | §9-67 ⑤⑥ |
| 「9 维线性打平 40M 模型」只在这一条判据上验证；§9-66 的 top-N F1 上界列；盲训练的迭代退化 | §9-66 存疑 1/2/4 |
| `seconds_position` 是否值得开；装饰线**模型侧**旁路未做；`val_windows` 抽样精度 / `val_every` A/B | §9-64 ③、§9-65、§9-63 |
| RFC-0038/0039 重叠处置；`nll_shuffled_delta` margin；`grad_clip_norm`；G1 下限 0.05 的 per-event 语义 | RFC-0038/0039 §5 |

**一次性探针（⚠️ `scripts/local_*` 与 `runs/_*` 均不入库，清理即丢）**：本轮 = `_probe_embedding_sep.py`（**零参数
可分性**，本轮主证据）、`_probe_mask_protocol.py`（遮盖协议）、`_probe_visible_indicator.py`、`_probe_mask_only.py`
（遮盖感知场基线）；上轮 = `_probe_stage_auc.py`（**其 L1.5/L2 读数已作废，见 §9-72①**）、`_probe_composition.py`、
`_probe_plan_prefix.py`、`_probe_plan_rounds.py`、`_probe_shuffle_check.py`、`_probe_tau_ceiling.py`、`_probe_mass.py`。
⚠️ **口径陷阱**：① `val.batches()` 前 40 批是稀疏层（跨 checkpoint 比较必须固定同一批）；
② 罕见标签上拟合 256 维读出会欠拟合（§9-72①）；③ 与训练并行的探针要**限内存**（§9-71⑦）。
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
