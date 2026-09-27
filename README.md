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
> 上次交接：**2026-09-27（第六轮）** ｜ 交接人：主会话（长跑运维落地：断点续训 / checkpoint 旋转 / 在线标量 / 巡检脚本；RFC-0030、RFC-0031 均采纳）

### 当前状态

自检（默认 CI，无网络 / 无权重 / 无 GPU）：`uv run ruff check . && uv run mypy beatmorph && uv run pytest -m "not slow and not gpu and not e2e"` → **987 passed**（另有 17 项 slow/gpu/e2e 默认不跑）。

**实跑证据**：

- ★ **真实数据门禁全绿**：`uv run beatmorph-train --config-name phigros_masked --gates-only --device cuda --skip-env-doctor data.max_samples=200` → **EXIT=0、2305.8 s ≈ 38 min**：
  G1 PASS（47455.4 → 65.1，`g1_events=20`）｜ **G2 PASS**（真实 1672.99 vs 打乱 3875.21，**2.32×**）｜ G3 PASS（107.77 ≤ 2412.17）｜ G4 PASS（215 帧）。权威记录 `runs/phigros_masked/20260927-043828/gates.txt`。
- **合成门禁全绿**：`--config-name smoke --gates-only`（CPU 约 2 min）。
- **训练通路验证（合成夹具）**：`--config-name smoke --gates --skip-env-doctor optim.max_steps=2000 run.save_every=500` → EXIT=0、218.8 s、六件套齐全。**这是通路验证，不是训练结果**。
- **长跑运维已验证**（默认 CI，无 GPU）：`tests/unit/infra/test_resume_and_rotation.py`（7 项）钉死「续训从下一步继续 / history 追加 / 配置与 data_rev 变了拒绝恢复 / 旋转先写后删 / best.pt 永不删」。
- **全量特征提取**：8005 / 8084（99.0%）；**训练窗口**：train 634 952（94.1/行）。

**真实数据（`data/processed/`，不入库）**

| 项 | 实测值 |
|----|--------|
| 谱面 / 音频 | **8551** 张 RPE（拒收 1099）；**42.8 GB** / 8084 唯一音频；唯一曲目 6807 |
| 判定线 / note | 中位 25 条；note 1107 万（Tap 55.7% / 背面 3.11%） |
| τ 轴截断（RFC-0031） | **3580 / 6750 行（53%）** 的 `META.chartTime` 虚高（中位 52.8×）⇒ 改由音频时长截断，省下 30.8 万小时空窗 |
| 训练窗口 | train **634 952**（旧口径 3361 万，98% 是空窗） |
| 训练清单 | `pairs.json`：train 6750 / val 814 / test 890 / 泛化 658（按曲目切分） |
| 索引 | 全库首次 **36.5 min**，落盘缓存后**秒级**命中 |

**四条已通的链**

1. **解码导出**：λ 场 → `decoder.decode_field`（D1 峰值 / D2 thinning）→ 合法性后处理 → `io/formats/rpejson` 写出 → 读回；
2. **训练通路**：清单 + 特征缓存 → `ChartPairDataset`（窗口化 + 索引缓存 + 行级 LRU）→ `collate_field_batch` → `MaskedFieldModel` → 泊松 NLL → checkpoint（**可旋转、可续训**）；
3. **真实数据通路**：Phira API → 预筛 + 选择性下载 → RPEJSON 解析 → 三级质检 → 带 provenance 的清单 → 真实 MERT 权重特征 → 窗口数据集；
4. **长跑运维通路**：`--resume`（配置/门禁/data_rev 三项校验）→ 在线标量（TB + `logs/loss_history.jsonl`）→ `scripts/training_health.py --watch 7200` 巡检。

**本轮（第六轮）落地的长跑运维**（事故与口径都写进了 [docs/TRAINING.md](docs/TRAINING.md) §7.5、plan 07 §9-29~§9-34）

- **断点续训**：`--resume latest`（逐个往前找有 checkpoint 的目录）或 `--resume <目录>`；恢复模型 + 优化器 + 步号，从 `step+1` 继续；**配置指纹 / 门禁全绿 / data_rev** 任一不符即拒绝（退出码 7）。
  指纹只覆盖**语义字段**：`optim.max_steps` 与 `run.save_every`/`log_every`/`keep_*` 变更**不会**让旧 checkpoint 失效（否则「想多跑几步」就得另起目录，把同一次训练劈成两半）。
- **checkpoint 旋转**：`run.save_every=2000`（每 4 轮，1 轮 = `optim.val_every`）、`keep_last=3`、`keep_best=1`，**先写后删**。
- **在线标量**：每 `run.log_every=50` 步刷 TB（`train/loss` / `step_time_s` / `grad_norm` / `lr` / `sys/peak_vram_gib`）并追加 `logs/loss_history.jsonl`——此前**只在训练结束时写一次**，人力监控整轮看不到曲线。
- **巡检脚本**：`scripts/training_health.py [--watch 7200] [--gpu]`，退出码 0 健康 / 1 告警 / 2 无数据；预警：步时退化 1.3×、显存 ≥ 7.5 GiB、标量 30 min 未更新、checkpoint 落后。
- ⚠️ **事故**：一次「扫 K vs 显存」的 GPU 探测（`sdpa_kernel` 强制后端 + 无界前向/反传循环）触发驱动层 TDR，**Windows 被重启**。禁令已写进手册：不强制注意力后端、不跑无界 GPU 扫批、同一时刻只跑一个 GPU 作业、全程盯功耗与显存。

**未落地 / 未验证**

- **全量大规模训练尚未启动**（本轮唯一核心任务，见「下一步」1）；
- **评估入口未接线**（形态已裁定 = 独立 `beatmorph-eval` 子命令；`eval/pipeline.py` 已就绪并测过），**待大规模训练完毕后**再做；
- **plan 04 消融臂（M7 B4 / M9 B3 / M10 B5 / M11 矩阵）按决策者口径「等大规模训练完毕再进行」**；B1 的 G3 常数基线口径仍未定（focal 不适用泊松闭式，plan 04 §9-20）；plan 06 M6.7/M6.8、plan 08 待建；plan 05 M5.7 待已训练模型；
- **Lightning 后端真机实跑**未做（`--extra train` 已装齐、env doctor E4 PASS）；
- **val 路径未实现**（`optim.val_every` 目前不影响训练；`best.pt` 因此只是「训练损失最优」，不是模型选择依据）；
- ⚠️ **global 层的 `O((K·T)²)`**（架构级，须 RFC）：步时 ∝ `K²`（实测 `K=12 → 0.10 s`、`K=29 → ~7 s`、`K=33 → ~9 s`），同时是 8 GB 卡的显存墙——扩大 `t_window`/数据规模前必须先裁定。

### 下一步

1. ★ **全量大规模训练（本轮唯一核心任务）**：

   ```bash
   # 启动（先跑门禁 38 min，索引命中缓存；随后训练）
   uv run beatmorph-train --config-name phigros_masked --gates --device cuda --skip-env-doctor
   # 崩溃/中断后原地续跑（不重跑门禁）
   uv run beatmorph-train --config-name phigros_masked --resume latest --device cuda --skip-env-doctor
   # 每 2 小时巡检（或用 --watch 7200 常驻）
   uv run python scripts/training_health.py --experiment phigros_masked --gpu
   # 人力监控
   uv run tensorboard --logdir runs/phigros_masked
   ```

   交接前必须复核的五个易错点：**① 每 4 轮存盘**（`save_every=2000`，崩溃丢 ≤ 一个间隔）｜**② 断点续训**（`--resume`，三项校验 fail-closed）｜**③ 每 2 小时巡检**（`training_health.py`，看降速/贴顶/停住/ETA）｜**④ TB 在线曲线**（`log_every=50` 增量刷盘 + `loss_history.jsonl`）｜**⑤ 显存墙不降速**（判据：`power.draw` 长期偏低 + `peak_vram_gib` 贴顶 ⇒ 立即停；步时 ∝ `K²`）。

   ⚠️ 训练是 `batch_size=1` ⇒ **单样本的 K 就是显存与步时的上限**；`K` 大的谱面会显著拖慢，必要时先开 RFC 处理 global 层。
2. **评估接线**（形态已定）：实现独立 `beatmorph-eval` 子命令（读 `run_dir` + split → 写 `metrics.json` 的 `eval` 分节；plan 06 §9-11 / plan 08），**在大规模训练产出 checkpoint 之后**做。
3. **plan 04 消融臂 + B1 的 G3 基线口径**：按决策者口径**等大规模训练完毕再进行**；B1 的 G3 口径先裁定（候选：最优常数 logit 的 focal 损失 / 「只输出先验」对照臂）。
4. **门禁预算曲线**优先度低（16 样本已给出 2.32× 差距）；真要压，先处理 global 层的 `O(K²T²)`（架构级，须 RFC）。
5. 若之后补抽特征：`uv run python scripts/extract_features.py`（可续跑）→ `uv run python scripts/fetch_phira.py pairs`；索引缓存会因清单/文件 stat 变化自动重建。

### 未决项（不阻塞 1–5）

| 未决 | 出处 |
|------|------|
| **global 层的 `O((K·T)²)`**：既是显存墙也是吞吐墙（架构级，须 RFC）；候选＝SDPA / 分块注意力 / 降 `t_window` | plan 07 §9-22 / §9-28 |
| **val 路径与模型选择口径**（`optim.val_every` 目前空转；`best.pt` 暂按训练损失） | plan 07 §4.5 / §9-31 |
| **续训的随机数状态**：dropout 流不恢复（数据顺序与遮盖种子仍由 index+seed 派生）⇒ 「续训 ≈ 继续」不是「逐位等价」 | plan 07 §9-29 |
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
