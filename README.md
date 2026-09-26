# BeatMorph

> **Phigros 谱面端到端自动生成系统** —— 从音频到可玩谱面，自监督学习取代显式标注。
>
> **核心范式**：`音频 → MERT 隐式理解 → 判定线局部系多线强度场 → 掩码补全 → 泊松 NLL → RPEJSON`

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Status: Pre-Alpha](https://img.shields.io/badge/status-pre--alpha-orange.svg)]()

BeatMorph 从原始音频（WAV/MP3）+ 难度（+ 可选判定线事件轨）出发，端到端生成高质量、可玩的 **Phigros 谱面（RPEJSON）**。模型从社区海量自制谱自主学习创作规律，无需人工标注（标注成本 ≈ 0）。

> 📌 **状态**：Pre-Alpha。范式（[RFC-0029](docs/decisions/RFC-0029-phigros-continuous-chart-generation.md)）、奠基文档（v3.0）、事实库已就绪；**核心契约 / 数据流水线 / 强度场 / 生成主干 / 解码与导出 / 训练基础设施六层已落地并通过默认 CI**（859 项测试），**真实 Phira 全库已拉取到本地（8551 张 RPE 谱面 + 42.8 GB 音频，不入库）**，**门禁有两套 slow 实跑 + `beatmorph-train --gates` 三种证据**；真实特征的**全量提取**、消融臂与评估接线待建（见下方交接件）。

---

## 当前状态与下一步（**跨 session 交接件**）

> 本节读者是**下一个 session 的 agent**，不是历史记录。
> **每次交接必须整节重写，不得追加**（规则见 [AGENTS.md](AGENTS.md) §6）。
> 上次交接：**2026-09-27** ｜ 交接人：主会话（真实数据落地：plan 02 M12 驱动脚本 + 全库拉取 + 真实数据缺陷修复）

### 当前状态

自检（默认 CI，无网络 / 无权重 / 无 GPU）：`make lint && make typecheck && make test-fast` → **859 passed**（约 90 s）；
另有实跑证据：`uv run pytest tests/unit/field/test_sanity_gates.py tests/unit/generation/test_sanity_gates.py -m slow`（6 项全绿）
与 `uv run beatmorph-train --config-name smoke --gates-only`（退出码 0、六件套落盘、gates.txt 四道 [PASS]）。

**真实数据已落地（本轮，全部在 `data/processed/`，不入库）**

| 项 | 实测值 |
|----|--------|
| 元数据枚举 | **9651** 张（调研基线 9649，+2） |
| 已入库谱面 | **8551** 张 RPE（拒收 **1099**：PEC 872 / 官谱 14 / 解析 191 / 质检 22；未解网络失败 1） |
| 音频 | **42.8 GB**，**8084** 个唯一音频（内容 sha1 去重，同曲重复率 5.5%） |
| 谱面文件 | 34.5 GB；共 77 GB（与调研「全量直抓 ≈76 GB」吻合） |
| 唯一曲目 | **6807**（同曲重复率 **20.4%**）→ 真实 (audio, chart) 对的数量级 |
| 特征缓存 | **24 首**（冒烟；全量待抽，实测 ≈2 s/首） |
| 判定线 | 中位 **25**（P25 24 / P75 42，范围 1–770）；note 计 1107 万 |
| 分布 | Tap 55.7% / 背面 **3.11%**（调研基线 2.4–3.0%，被 `corpus_outliers` 标为轻微离群）；越界 0.13% |

两条驱动脚本：`scripts/fetch_phira.py`（`meta` / `fetch` / `pairs` / `stats` / `all`，可续跑、记账齐全）
与 `scripts/extract_features.py`（按音频 sha1 去重 + 缓存六项校验）。用法见 [docs/TRAINING.md](docs/TRAINING.md) §3/§5。

**已通的三条链**

1. **解码导出**：λ 场 → `decoder.decode_field`（D1 峰值 / D2 thinning）→ 合法性后处理 → `io/formats/rpejson` 写出 → 读回；
2. **训练通路**：清单 + 特征缓存 → `data.ChartPairDataset`（窗口化）→ `collate_field_batch` → `FieldBatch` → `MaskedFieldModel` → 泊松 NLL → checkpoint；
3. **真实数据通路（本轮新增）**：Phira API → 预筛 + 选择性下载 → RPEJSON 解析 → 三级质检 → 带 provenance 的清单 → **真实 MERT 权重**特征 → 窗口数据集（冒烟 19 行 → **354620 个窗口**）。

**本轮修掉的真实数据缺陷**（每条都有回归测试；详细现象见 plan 02 §9 与 TRAINING.md §3.3）

1. `info.yml` 文本字段实测常为 `null` / 数字 / 布尔（`tip` / `level` / `charter` / `composer`），裸 `str` 声明让 pydantic 拒收整包——首批 20 张里 **10 张**、全库 5 张被误判为「结构错误」；
2. **PEC 语法族比调研样例大**（`&` / `cf` / `cr` 三种符号命令），旧谱面 PEC 实测占 **9.0%**（调研抽样给 2.5%，抽样偏差比原估计更大）；
3. `build_pairs` 把**已解析**路径写进 `chart_path` ⇒ `ChartPairDataset` 再拼一次 chart_dir ⇒ 20/20 行判「谱面缺失」，门禁装配直接失败（跨模块往返 bug）；
4. token 扩张会拆散**跨 token 的长 Hold 配对**（数据集抛「hold 配对点被拆散」）→ 新增 `close_hold_pairs`（token 级收口）；
5. 分页路径**没有重试**：实跑第 54 / 87 / 286 页各命中一次读超时，没有重试时整轮 322 页枚举直接报废；
6. **必须强制 IPv4**：httpx 串行尝试 getaddrinfo 的地址，本机 IPv6 黑洞让每次连接等满 connect 超时（整体 ~0.2 MB/s）；`local_address="0.0.0.0"` 后同一 URL **≈20 MB/s**（100 倍）；
7. 装齐 `--extra train / audio` 后 `mypy --strict` 才暴露 `mert.py` 的 3 处第三方无类型调用（此前依赖缺失 ⇒ Any ⇒ 静默通过）——**类型门禁是环境相关的**。

**未落地 / 未验证**

- **全量 MERT 特征提取**（8551 首 ≈ 5 h GPU；本轮只做了 24 首冒烟）：这是训练前的唯一大块前置；
- **真实批次上的 G1–G4 全绿**：可装配（19 行 → 35 万窗口 → 批次 → 前向），但受 **G2 内存标度**限制——`shuffle_samples=16` 时单次分配 5.4 GB 直接 OOM，`4` 时进程私有内存涨到 32 GB（本机 31 GB）⇒ 已降到 `2` 并记入 plan 07 §9-15；
- plan 04 消融臂 M7–M11、plan 06 M6.7（B1–B6 矩阵）/ M6.8（人评）、plan 05 M5.7（需已训练模型）、plan 02 的驱动脚本之外的自动化、`api/`（plan 08）；
- Lightning 后端仍**未真机实跑**（`--extra train` 已装齐、env doctor E4 已 PASS）。

### 下一步

1. **全量特征提取**（唯一的训练前置）：`uv run python scripts/extract_features.py`（可续跑）→ 完成后**必须重跑** `uv run python scripts/fetch_phira.py pairs` 重建清单（当前 `pairs.json` 只含 24 首冒烟子集）。
2. **真实批次的门禁预算**（plan 07 §9-14 / §9-15）：G2 的内存标度要么换更大内存的机器，要么把 G2 改成分批前向（代码改动）；改之前先量「真实数据上多少样本才够」。
3. **评估接线**（plan 06 M6.7 + plan 07 §3.2）：`eval/` 指标库已就绪，等解码/生成臂出数接成完整报告（六件套已留好位置）。
4. **plan 04 消融臂**（M7–M11）：每个新目标先过 G1–G4，结果写进训练日志。
5. **两个数字已可直接引用**（`data/processed/stats.json`）：唯一曲目 6807 / 同曲重复率 20.4% ⇒ 真实对数规模；判定线中位 25 ⇒ `RPE_X_GRID_BINS` 的消融口径。⚠️ `max_simultaneous_onsets = 12012` 是明显离群值，**先查它是不是数据异常**，再拿去定「同刻跨线并发上限」。

### 未决项（不阻塞第 1–5 步）

| 未决 | 出处 |
|------|------|
| **[RFC-0030](docs/decisions/RFC-0030-decoder-export-contract-ownership.md) 整体待裁定**：解码/导出契约归属 + 六项实现口径；**实现已按提案落地** | RFC-0030 |
| **契约父线合成**与 A 级证据不一致 ⇒ 含父线谱面（实测 26%）的跨线几何计数为近似值（需 contracts-agent） | RFC-0030 §后果-2 |
| **跨谱 batching 的契约缺口**：`FieldBatch` 只带一个 grid，且**没有 `time_mask`**（音频 padding 是伪造静音）⇒ 跨谱批训练前须裁定 | 本轮（plan 07 §9-13） |
| **门禁在真实数据上的内存标度**：G2 一次性 collate 全部样本 ⇒ 需分批或换机器 | plan 07 §9-15 |
| **G2 的真实样本数预算**：`shuffle_samples` 从 16 降到 2 是为内存让路，对照是否仍有效需实测曲线 | plan 07 §9-12 / §9-14 |
| **遮盖重标定口径**：`hidden`(1/r，实现默认) vs 文档字面 `observed`(1/(1-r)) vs 不自洽的 `hidden_doc` | plan 04 §9-6 |
| **mask 泄漏口径**：默认「以事件为锚的 token 区块遮盖 + 空格稀释」偏离 §4.2 字面表述 | plan 04 §9-18 |
| plan 04 报告方法学两项：条件学习的「边际盆地」loss 差监控（R-04-5）、G2 合成任务与训练预算 | plan 04 §9-15 / §9-16 |
| 输出头**直连 skip** 是否写进 BasePlan §3.3 的架构描述 | plan 04 §9-17 |
| eval 三项口径待复核：多重比较校正、相位搜索范围/步长、难度分档边界 | plan 06 §9-1 / §9-3 / §9-5 |
| `N` 的最终取值（默认 128 已裁；仍需碰撞率–N / NLL–N / F1–N 三条曲线；全库最小同线同刻间距实测 **0.0** ⇒ 该统计口径本身要复核） | plan 03 §9-7、plan 02 §9-5 |
| 秒↔τ 两个换算点是否合并为一份实现；τ 轴终点口径 | plan 02 §9-11、plan 03 §9-13 / §9-14 |
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
