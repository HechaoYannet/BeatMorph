# BeatMorph

> **Phigros 谱面端到端自动生成系统** —— 从音频到可玩谱面，自监督学习取代显式标注。
>
> **核心范式**：`音频 → MERT 隐式理解 → 判定线局部系多线强度场 → 掩码补全 → 泊松 NLL → RPEJSON`

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Status: Pre-Alpha](https://img.shields.io/badge/status-pre--alpha-orange.svg)]()

BeatMorph 从原始音频（WAV/MP3）+ 难度（+ 可选判定线事件轨）出发，端到端生成高质量、可玩的 **Phigros 谱面（RPEJSON）**。模型从社区海量自制谱自主学习创作规律，无需人工标注（标注成本 ≈ 0）。

> 📌 **状态**：Pre-Alpha。范式（[RFC-0029](docs/decisions/RFC-0029-phigros-continuous-chart-generation.md)）、奠基文档（v3.0）、事实库已就绪；**核心契约 / 数据流水线 / 强度场 / 生成主干 / 解码与导出 / 训练基础设施六层已落地并通过默认 CI**（795 项测试），**门禁有两套 slow 实跑 + `beatmorph-train --gates` 三种证据**；真实数据拉取、消融臂与评估接线待建（见下方交接件）。

---

## 当前状态与下一步（**跨 session 交接件**）

> 本节读者是**下一个 session 的 agent**，不是历史记录。
> **每次交接必须整节重写，不得追加**（规则见 [AGENTS.md](AGENTS.md) §6）。
> 上次交接：**2026-09-27** ｜ 交接人：主会话（plan 06 评估 M6.1–M6.6 + plan 07 训练基础设施 M7.1–M7.8 + 真实数据通路 M11）

### 当前状态

自检（默认 CI，无网络 / 无权重 / 无 GPU）：`make lint && make typecheck && make test-fast` → **795 passed**（约 22 s）；
另有两条实跑证据：`uv run pytest tests/unit/field/test_sanity_gates.py tests/unit/generation/test_sanity_gates.py -m slow`（6 项全绿）
与 `uv run beatmorph-train --config-name smoke --gates-only`（退出码 0、六件套落盘、gates.txt 四道 [PASS]）。

**已通的两条端到端链**（都不需要权重 / GPU / 网络）：

1. **解码导出**：`λ 场 → decoder.decode_field（D1 峰值 / D2 thinning）→ 合法性后处理 → io/formats/rpejson 写出 → 读回`；
2. **训练通路**（本轮新增）：`清单 + 特征缓存 → data.ChartPairDataset（窗口化）→ collate_field_batch → FieldBatch → MaskedFieldModel → 泊松 NLL → checkpoint`；
   入口两条都实测过：`--config-name smoke`（合成谱，秒级）与 `data.source=manifest`（**真实 RPEJSON 夹具 + 真实特征缓存**）。

**已落地**（plan 00 / 02 / 03、plan 04 主干 M1–M4·M6、plan 05 M5.1–M5.6、plan 06 M6.1–M6.6、plan 07 M7.1–M7.8；逐条状态见各 plan 的「实施状态」列）：

- `core/contracts/` ✅ v3.0 契约：判定线 / 音符 / 谱面 / 事件轨求值 + 场网格规格 + `Violation`/`Edit`/`LegalityReport`
- `data/` ✅ RPEJSON 独立实现解析器、内容嗅探、`info.yml` 定位、三层质检、Phira 客户端、特征缓存、配对切分；**本轮**：`tracks.py`（事件轨在 τ 轴上求值）与 `dataset.py`（窗口数据集 + collate：网格不一致即抛、**绝不发出 r == 1 的样本**）
- `field/` ✅ 网格与**秒↔τ 唯一换算**、目标构建、两条 ∫λ 路径互校、泊松 NLL、G3 闭式基线、共格碰撞统计、可视化
- `generation/` ✅ 掩码补全 Enc-Dec（滑动窗口 + 周期全局层、难度 AdaLN、可变 K、因子化 λ 头）+ 迭代并行解码
- `decoder/` ✅ D1 峰值解码、D2 Ogata thinning、Hold 枚举配对、合法性后处理（越界**只统计**、`EditKind` **无 clamp 成员**）
- `io/formats/rpejson/` ✅ 写路径（秒→beat 三元组、导出门禁、字节级幂等）
- `eval/` ✅ **本轮**：双容差 F1（±20 / ±50 ms）、全谱相位偏移、背面 recall、`positionX` MAE、类型/侧别准确率；贪心一对一匹配（**未匹配的生成事件留在分母**）、macro/micro 双报、固定规则/每谱最优两栏；时间组（含 1/12、1/24 三连）与难度分解；NLL 校准与探索性指标**结构隔离**；7 类 dose-controlled corruption 准入 + 2 项不变性控制
- `infra/` + `cli/` ✅ **本轮**：环境自检 E1–E5（PASS/FAIL/UNKNOWN 三态）、structured config（**`data.provenance` 必填**；缺字段/类型错/多余字段**启动期**失败）、门禁执行器（`--gates` / `--gates-only` + **fail-closed**）、实验六件套、checkpoint 恢复校验（配置逐字段 diff / 门禁 / `data_rev`）、派生量全仓扫描（M7.4）、合成批次来源、torch 参考训练循环 + Lightning 适配
- 夹具 ✅ `tests/fixtures/phigros/`（手工构造，无音频无曲绘）

**接缝**（integration 测试各钉一处跨模块一致性）：`test_time_conversion_seam.py`（格式层 beat↔秒 vs 场层秒↔τ）、`generation/test_train_step.py`、`test_decode_to_rpejson.py`，**本轮新增** `test_dataset_to_generation.py`（真实窗口 → 前向 → 反传）与 `test_train_entry.py`（CLI ↔ 门禁 ↔ 六件套 ↔ 退出码）。

**本轮发现**（都已变成测试或 plan 条目，不留在口头）：

1. **G2 的样本数是判据的一部分**：单样本时打乱臂能直接背样本——实测打乱 loss（80.8）**低于**真实 loss（185.1），对照完全失效。`gates.shuffle_samples`（默认 16）已进配置并落进 gates.txt 的生效阈值。
2. **数据集绝不能发出 r == 1 的样本**：`masked_poisson_loss` 在「全部事件被遮盖」时**拒绝训练**，而稀疏窗口（安静段落）必然触发。数据集现在先排除与种子无关的不可行窗口，再重掷种子（上限 8），仍不行则以无遮盖发出并**记账告警**。
3. **泊松 NLL 的下界不是 0**：G1 实际生效的是**相对**判据（`target_ratio × 首步 loss`）——plan 07 §9-2 的预判被实测证实。
4. **跨谱批次受「一个 `FieldBatch` 只带一个 `FieldGrid`」约束**：已落地的缓释是「窗口局部网格（窗口落在单一 BPM 段内 ⇒ 同批逐格 J 相同）+ 采样器按 `grid_key` **分桶轮转**」，跨 BPM 变更点的窗口跳过并记账。仍**未解决**的是「同一批里混合不同 BPM 段」——那要做 per-sample 网格或按样本取 J，**契约级问题，须开 RFC**。
5. **全谱 `build_target` 的内存标度**：全谱计数张量 `K × T_full × X × S × C` 在 5 分钟谱、K=30、X=128 下约 **2 GB / 样本**——训练不可用。数据集改为「窗口子谱 + 窗口网格」调用，等价性有测试锁定（plan 03 §9-17）；这也意味着 `RPE_X_GRID_BINS` 的消融必须报告**窗口口径**的内存。
6. **上一轮提交就让 `make lint` 变红**（`scripts/_check_text.py`：E401/I001/F401/PLW1510/W292）：本轮修好并补上非 0 退出码。教训：交接件里的「lint 全绿」必须当场跑过再写。

**未落地**：plan 04 的消融臂与阶段出口（M5 的 F1 对照、M7–M12）、plan 05 的 M5.7（需已训练模型）、plan 06 的 M6.7（B1–B6 矩阵）与 M6.8（人评）、plan 02 的**驱动脚本与真实拉取**、`api/`（plan 08）。**Lightning 后端已实现但未实跑**（本机未装 `train` extra）。

### 下一步

1. **真实数据落地（plan 02）**：`phira/client.py`（分页枚举 / Range 预筛 / 选择性下载 / 清单 provenance）与 `pipeline/embed.py`（MERT 特征 + 缓存元数据校验）**都已就绪**，缺的是**把它们串起来的驱动脚本**（建议 `scripts/fetch_phira.py`：枚举 → 预筛 → 下载 → `build_pairs`；`scripts/extract_features.py`：按 `info.yml.music` 定位 + sha1 去重 → 缓存）。有了清单与特征缓存，`beatmorph-train --config-name phigros_masked --gates` 才能拿到**真实批次**。
2. **评估接线（plan 06 M6.7 + plan 07 §3.2）**：`eval/` 的指标库已就绪，但 `runs/<exp>/<ts>/metrics.json` 目前只写训练摘要；等解码/生成臂出数后接成完整报告（六件套已为它留好位置）。
3. **plan 04 的消融臂（M7–M11）**：B1 热图 + focal、B3 GOCT 配置臂、B4 AR 上界臂、B5 absorbing 扩散臂；**每个新目标先过 G1–G4**（`beatmorph-train --gates`），结果写进训练日志。
4. **全库统计回填两个数字**：共格碰撞率 → `RPE_X_GRID_BINS` 的最终取值；唯一曲目数 / 同曲重复率（plan 06 的难度分档与 cluster 定义也在等它）。

### 未决项（不阻塞第 1–4 步）

| 未决 | 出处 |
|------|------|
| **[RFC-0030](docs/decisions/RFC-0030-decoder-export-contract-ownership.md) 整体待裁定**：解码/导出契约归属 + 六项实现口径；**实现已按提案落地** | RFC-0030 |
| **契约父线合成**与 A 级证据不一致 ⇒ 含父线谱面（实测 26%）的跨线几何计数为近似值（需 contracts-agent） | RFC-0030 §后果-2 |
| **跨谱 batching 的契约缺口**：`FieldBatch` 只带一个 grid，且**没有 `time_mask`**（音频 padding 是伪造静音）⇒ 跨谱批训练前须裁定 | 本轮（plan 07 §9-13、plan 02 §9 实施期裁定） |
| **遮盖重标定口径**：`hidden`(1/r，实现默认) vs 文档字面 `observed`(1/(1-r)) vs 不自洽的 `hidden_doc` | plan 04 §9-6 |
| **mask 泄漏口径**：默认「以事件为锚的 token 区块遮盖 + 空格稀释」偏离 §4.2 字面表述 | plan 04 §9-18 |
| plan 04 报告方法学两项：条件学习的「边际盆地」loss 差监控（R-04-5）、G2 合成任务与训练预算 | plan 04 §9-15 / §9-16 |
| 输出头**直连 skip** 是否写进 BasePlan §3.3 的架构描述 | plan 04 §9-17 |
| **门禁预算在真实数据上是否够用**：G1 的步数、G2 的样本数都需要实测曲线 | plan 07 §9-12 / §9-14 |
| **Lightning 后端未实跑**；env doctor 的 E4 因缺 `train` extra 恒为 UNKNOWN，TB 标量也没写过（gates.txt 仍是权威记录） | plan 07 §9-10 / §9-11 |
| eval 三项口径待复核：多重比较校正、相位搜索范围/步长、难度分档边界 | plan 06 §9-1 / §9-3 / §9-5 |
| `N` 的最终取值（默认 128 已裁；仍需碰撞率–N / NLL–N / F1–N 三条曲线） | plan 03 §9-7、RFC-0029 §8.4 R-f |
| 秒↔τ 两个换算点是否合并为一份实现；τ 轴终点口径（`chartTime` / 最后一事件 / 音频时长 / 是否含 `META.offset`） | plan 02 §9-11、plan 03 §9-13 / §9-14 |
| BPM 变更点落在 1/48 拍格内时 `J_j` 的取值（已量化出 `rule` 三档，默认 `left`） | plan 03 §9-16 |
| `RPE_HEIGHT_RATIO` 的出处（D3）；旋转正方向的屏幕含义（D1）；`META.offset` 的符号解释（解码路径写 0） | plan 00 §9-6/§9-8、plan 05 §9-8 |

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
