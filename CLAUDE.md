# CLAUDE.md — BeatMorph 项目宪法

> 本文件是 Claude Code（及任何 AI 协作 agent）在本仓库工作时的最高行为准则。
> 一切开发以 [docs/BasePlan.md](docs/BasePlan.md) 奠基文档为唯一技术权威。
> **文档版本：v3.0**（RFC-0029 修宪，2026-08-05 采纳：目标 osu!mania 4K → **Phigros**；生成范式 → **判定线局部系多线标记点过程 + 掩码补全**）

## 1. 项目是什么

BeatMorph 是 **Phigros 谱面端到端自动生成系统**：从原始音频（WAV/MP3）+ 难度 + 判定线事件轨（可选）出发，自动生成可玩的 Phigros 谱面（RPEJSON 格式）。

**核心范式（不可动摇，见 [BasePlan §1.2](docs/BasePlan.md) 与 [RFC-0029](docs/decisions/RFC-0029-phigros-continuous-chart-generation.md)）**：

```
音频 → MERT 隐式理解 → 判定线局部系多线强度场 λ_k(τ, positionX, side, type)
     → 掩码补全迭代并行解码 → 非齐次泊松 NLL 训练 → 强度场解码为离散谱面 → RPEJSON
（τ = 拍相对时间坐标，beat-aligned，1/48 拍；见 BasePlan §3.2.4）
```

设计哲学：**自监督理解取代显式标注**。模型从海量社区自制谱自主学习创作规律，人只提供极简控制（难度 + 判定线运动条件 + 可选参考风格），标注成本 ≈ 0。

> **历史与当前的关系**：v2.x 曾锁定 osu!mania 4K + VQ-VAE（后经 RFC-0028 改为 BPE/event tokenizer）。RFC-0029（2026-08-05 采纳）将**目标游戏**改为 Phigros 并**替换生成范式**。osu!mania 资产归档在 `archive/osu-mania` 分支，**不在主路径维护**。

## 2. 模块拓扑

```
beatmorph/
├── audio/encoder/        Stage 0  MERT-v1-330M + LoRA Adapter（帧率由模型 config 派生，=75Hz）
├── data/                 数据流水线：Phira 谱面获取 + RPEJSON 解析 + 质检 + MERT 特征离线提取
├── core/contracts/       ★ 跨模块契约：JudgeLine / PhigrosNote / PhigrosChart / ChartField
├── field/                强度场：λ_k(τ,x,s,c) 的网格化、beat↔秒换算、目标构建、∫λ 双路径积分、可视化
├── generation/           Stage 1  掩码补全 Encoder-Decoder（主选）；离散扩散对照臂
├── decoder/              强度场 → 离散谱面（迭代/峰值解码）+ 合法性与可玩性后处理
├── io/formats/rpejson/   RPEJSON 读写（主路径唯一权威格式）
├── eval/                 事件级 F1 / 类型准确率 / 跨线合法性 / NLL 校准 / 人评协议
├── infra/                训练栈 + 健全性门禁 G1/G3/G4（beatmorph/infra/sanity.py；G2 已随 RFC-0037 删除）
├── cli/  api/            命令行与服务接口
└── （alignment/ rag/）   Phase 3 待重估：DPO 偏好对齐 / 风格检索，RFC-0029 未涉及
```

各模块详细计划见 [docs/plans/](docs/plans/README.md)（00-08，每份含接口契约/里程碑/风险/测试策略）。

## 3. 不可违背的约束（红线）

1. **技术选型锁定 BasePlan v3.0（= RFC-0029 范式）**：不得擅自把 MERT 换成其它音频编码器、把「掩码补全 + 泊松 NLL」换成「逐 token 自回归 + CE」或「朴素 BCE 热图」、把**判定线局部系**换成屏幕系、把**多判定线**降级为单线。任何变更须先开 RFC（[docs/decisions/](docs/decisions/README.md)）。
   > **RFC-0029 采纳附注**：`.pec` 是已淘汰的 PhiEditer 格式，**不是**官方格式；主路径锁定 **RPEJSON**。RPE 与官谱 JSON 的 note `type` 数字含义不同，混用会静默错位。
2. **模块间只通过 `core/contracts` 通信**：跨模块数据用已定义契约类型。**新增跨模块类型前先开 RFC**。
3. **AI 与规则解耦**：后处理规则引擎只做「物理/格式红线」校验与钳位（同刻按键上限、Hold 区间合法性、跨线几何冲突、越界），**不得改变模型的落点分配与键型逻辑**。
4. **单一目标 Phigros**：不得为其它游戏模式污染主路径；其它游戏（osu!mania / osu!std / maimai 等）一律视为**独立适配工程**，须新开 RFC 立项。
5. **数据/权重不入库**（见 `.gitignore`）：模型权重、原始音频、数据集走外部存储或 Git LFS，仅保留 `data/fixtures/**`、`tests/fixtures/**` 微型夹具。
   > ✅ **数据合规已裁定（2026-08-05）**：风险由**决策者自行承担**（定位为对 Phira 社区的贡献与共创，最终善后由决策者以合规方式完成）。**由此产生的项目级硬约束**：
   > 1. **最终不发布模型权重** —— 本裁决成立的前提条件，**属项目级承诺**；
   > 2. 谱面与音乐数据**允许本地落盘**，但**不得入库**；
   > 3. 获取与处理脚本须**记录来源与用途**，便于追溯；
   > 4. **发布权重前必须重新裁定** —— 本裁决不覆盖任何分发场景。
   > 风险事实与完整裁决见 [RFC-0029 §8.3 Q11b](docs/decisions/RFC-0029-phigros-continuous-chart-generation.md) 与 [phira-dataset-survey.md](docs/knowledges/phira-dataset-survey.md)。
   > ⚠️ **解析器许可证**：事实标准 prpr（Rust）为 **GPL-3.0**、phichain 为 **LGPL-3.0**，**逐行移植有衍生作品风险** → 只读其行为规范、独立实现。
6. **100% 可玩/合法**：生成谱面必须经合法性校验为空违规项才允许导出。
7. **【v3.0 新增】物理常量必须派生 + 断言；新范式必须过健全性门禁**：
   - 任何帧率/采样率/单位换算**不得硬编码**，必须由权威来源（模型 config、格式规范）派生，并有**默认 CI 内运行**的契约测试。
   - 任何新模型/新范式在**扩大数据规模之前**必须通过 **G1/G3/G4** 三道门禁（单 batch 过拟合 / 常数基线 / 契约断言），结果写入训练日志。
     > **RFC-0037 修订（2026-09-28 决策者裁定）**：原 G2（打乱标签对照）**已删除**——单批训练损失「速度赛」在真实批上判别力未被证明（RFC-0036 §1-§2 证据）。「输入对目标零信息」改由 **held-out 在线对照**承担：`val/nll_shuffled_delta`（线内置换）+ `val/ratio` + 条件干预 `cond_*_delta`。**判读红线**：任一 val 点 `val/ratio >= 1` 或 `val/nll_shuffled_delta <= 0` ⇒ 该 run 的扩大数据规模结论**作废**。同一裁定把训练损失改为**按事件归一**（整式除以 `max(E_total,1)`，argmin 不变，见 RFC-0037 §2.2 的「只除事件项」陷阱推导）。
   - **beat-aligned 的时间换算只允许在 `field/` 内实现**（秒 ↔ 拍相对坐标 τ，Jacobian 由 `BPMList` 派生）。契约层 `PhigrosNote.t` 仍用秒；**下游不得各自再实现一套时间换算**。
   - 来历与代价见 [docs/POSTMORTEM-2026-08-05-frame-rate-misalignment.md](docs/POSTMORTEM-2026-08-05-frame-rate-misalignment.md)：25 Hz 误值（真值 75 Hz，差 3×）在 mock 的掩护下存活到万级数据规模。

## 4. 工程约定

- **语言/版本**：Python 3.11。
- **包管理**：`uv`。环境复现：`uv sync`（见 Makefile）。
- **格式/lint**：`ruff`（配置见 `pyproject.toml`）。提交前 `make lint`。
- **类型**：`mypy --strict`；ML 库缺 stub 处已在 overrides 忽略。
- **测试**：`pytest`，标记 slow / gpu / integration / e2e。CI 默认跑「非 slow、非 gpu、非 e2e」。
  - ⚠️ **契约级测试不得依赖权重或 GPU**。凡「只有拿到权重才能跑」的断言，等于没有断言。
- **提交**：Conventional Commits（pre-commit commitizen 强制）；每个 commit 引用相关 plan 或 RFC。
- **换行符**：全仓 LF（`.gitattributes` + `.editorconfig` 已强制）。

## 5. 开发流程（对 AI agent）

1. **改任何模块前，先读对应 `docs/plans/0X-*.md`**，确认接口契约与里程碑；冲突时以 plan 为准，plan 与 BasePlan 冲突时以 BasePlan 为准并开 RFC。
2. **跨模块改动**先动 `core/contracts`，契约变更须同步 plan 与 RFC。
3. **新功能**遵循 plan 里程碑顺序，不得跳阶段（例如未跑通数据流水线就写 API）。
4. **不确定的技术细节**：查 `docs/knowledges/` 的格式/单位文档 → 查 BasePlan 相应章节 → 仍不确定则写进 plan「开放问题」并提 RFC，**不要凭猜测改代码**。
5. **测试先行**：每个模块至少有契约/形状级单元测试。
6. **日志**：`from beatmorph.core.logging import get_logger`，禁止裸 `print` 进生产路径。
7. **大文件**：写代码前确认产物不在 `.gitignore` 黑名单（权重/音频/parquet）。
8. **【v3.0 新增】门禁先行**：新增训练目标或损失函数时，**先把门禁（G1/G3/G4；RFC-0037 起）跑通再谈扩数据**，并盯住 val 对照红线（`val/ratio < 1`、`val/nll_shuffled_delta > 0`）。

## 6. 当前状态（2026-09-29）

- ✅ **范式已定**：RFC-0029 采纳（Phigros + 多线标记点过程 + 掩码补全 + 泊松 NLL），Q1/Q2/Q4/Q6/Q7/Q8 已由决策者批复。
- ✅ **根因已清**：MERT 帧率 25Hz→75Hz 修复落地（三条独立证据链），G1-G4 门禁与 G4 契约测试入库并通过。
- ✅ **格式事实已就绪**：`docs/knowledges/phigros-format.md`（RPEJSON 逐字段、坐标几何、来源分级 A/B/C）。
- ✅ **信息准备已完成（四项全齐）**：[phigros-format.md](docs/knowledges/phigros-format.md)（格式，含勘误横幅）、[phigros-units-and-geometry.md](docs/knowledges/phigros-units-and-geometry.md)（单位/几何，prpr 源码 A 级）、[phira-dataset-survey.md](docs/knowledges/phira-dataset-survey.md)（数据源实测：9649 张、K 中位 30、音频 100% 捆绑）、[chart-generation-literature.md](docs/knowledges/chart-generation-literature.md)（文献与评估协议）。
- ✅ **两项阻塞已清（2026-08-05）**：① **数据合规已裁定**（风险由决策者承担，硬约束 = 最终不发布权重，见红线 5 附注）；② **Q15 时间网格定为 beat-aligned**（数学改写限定在 `field/` 内，见 BasePlan §3.2.4）。
- 🔵 **施工进行中**：plan 00 / 02 / 03、plan 04 主干（M1–M4/M6，**B1 臂的模型与 G1 门禁**）、plan 05（M5.1–M5.6）、plan 07（M7.1–M7.8）、plan 06（M6.1–M6.6 + 报告落盘接线）已落地并通过默认 CI（**1086 项**）；**合成门禁与真实切片门禁均实跑全绿**，且**真实数据 + CUDA 的训练实跑已通过**（24 步内命中 K=100，峰值 4.00 GiB）；plan 04 的其余消融臂（M7/M9/M10/M11）与 plan 06 M6.7/M6.8、plan 08 待建；plan 05 的 M5.7（双解码臂 B6 出数）待已训练模型；**全量大规模训练待启动**（前置已清）。
- ⚠️ **2026-09-27 门禁口径修复（重要）**：G2 曾经是**恒真对照**（真实臂 = G1 的遮盖补全臂、打乱臂 = 全事件目标，两者不同批不同损失）⇒ **此前所有 `gates.txt` 里的 G2 数字作废**；现口径 = 同一遮盖批、只置换被遮盖格子内的标签（可见场逐位不变），合成夹具上差距 45%。
- ⚠️ **2026-09-27 第五轮：τ 轴终点缺陷（RFC-0031）——本轮最大的一处数据侧缺陷**：`duration_s()` 无条件信任 `META.chartTime`，而全库 **52.2%** 的谱面该值虚高（中位 **52.8×**、最大 3.4e6×）⇒ train split 曾切出 **3361 万窗**（98% 空窗）、索引 2.3 h，并**伪造**出「G1 空过（0 事件）/ G2 结构性 FAIL（打乱臂反而更优）」两个假结论。默认口径改为 `data.tau_end_policy="audio"` = `min(谱面口径, 特征缓存的音频时长)`，截断与轴外事件**显式记账**；修复后 train split **634 952 窗**、索引 **36.5 min**，真实切片门禁 **G1-G4 全绿**（EXIT=0）。**同一轮**：门禁批必须非空（`gates.batch_min_events=1`，取不到即抛）、索引**落盘缓存** + 行级 LRU、门禁装配的显存卫生（G3 预计算先跑后释放，避免 8 GB 卡滑进 Windows 共享内存把单步放大 15×）。见 plan 02 §9 第四轮、plan 07 §9-25/§9-26/§9-27、[RFC-0031](docs/decisions/RFC-0031-tau-axis-endpoint.md)。
- ✅ **RFC-0030 / RFC-0031 / RFC-0032 均已采纳（2026-09-27 决策者裁定）**：RFC-0030＝解码/导出契约归属与 plan 05 六项口径；RFC-0031＝**τ 轴终点 = `min(谱面口径, 音频时长)`**（`META.offset` 仍留待后续）；RFC-0032＝**局部层的事件轨条件只注入本线轨，跨线只经全局层**。三案实现均已落地
- ⚠️ **第八轮：采样器覆盖率缺陷（RFC-0033，最严重的一处训练侧缺陷，已修）**：`ManifestBatchSource` 用**一个全局共享游标**并按当前桶长度取模，桶长最小为 1 ⇒ **1000 步后饱和在 993 个窗口（0.156%）/ 674 张谱面（10%）**，`max_steps` 加到多少都不见新数据，而 loss 曲线完全正常。已改为「每桶独立游标 + 桶内种子化洗牌 + 按剩余窗口加权选桶」并新增 `epoch` / `coverage/charts_seen` 在线标量；**此前所有真实数据训练的"数据覆盖"结论作废**（第五轮门禁记录不受影响）。**门禁批同时改用有界切片 `gates.gate_samples=200`**（否则门禁会抽到 K≈128 的大桶）。护栏：`tests/unit/infra/test_sampler_coverage.py` + `test_gate_samples_bound.py`。
- ⚠️ **第九轮：数据供给吞吐 / 「GPU 利用率不高是不是 workers 太小」（RFC-0034，已裁定 S1–S4）**：实测（训练进行中）GPU 利用率**均值 35.7% / 中位 5.0% / 29/60 采样 ≤1%**（双峰：要么满载要么空转），功耗均值 **21.4 W = 115 W 上限的 18.6%**，训练进程占 **6.45/24** 核。**关键事实：训练路径里根本没有 worker 这个旋钮**——全仓 `num_workers` 只在数据准备脚本里，训练侧唯一 `DataLoader` 的 `num_workers` 默认 0；真实取批 `ManifestBatchSource.batch()` 在**训练主线程同步**完成，`step_time_s` 的计时起点在它之前 ⇒ 数据构建时间**一直混在步时里**。瓶颈是**每窗口重解析整张谱面**（一张谱平均 96 窗，≈**46%** 的每窗口 CPU；调大 `chart_cache_size` 实测无效，命中率封顶 18.6%）。**已落地**：步时拆成 `data_time_s` + `compute_time_s`（和恒等于步时）+ `perf/data_share`，巡检脚本打印拆分并在**数据占比 ≥50% 时告警**（护栏 `tests/unit/infra/test_step_time_split.py`，6 项）。同轮修掉巡检脚本的**恒真告警**：旧口径 `last_step > checkpoints[-1]` 在 `save_every=2000` 时 1999/2000 的步都命中 ⇒ 退出码恒为 1；现只在**真漏存（间隙 > save_every）**时告警。**已落地（RFC-0034 S1–S4，提交 `cc0e166`）**：新增 `beatmorph/data/plan.py` 把「取什么」变成纯函数（顺序 = `(seed, epoch, 数据集索引)`，覆盖率 = 顺序前缀的 O(log n) 查询）⇒ 采样器不再有状态，`data.workers` 才能开（进 `RESUME_IGNORED_KEYS`，门禁强制 0）；`train()` 改为流式 `next(stream)`。**S4 第一版设计被实测抓住并改掉**：按窗口数分槽位让 20,000 步只覆盖 **52.6%** 的谱面（旧均匀撒点 91.3%）⇒ 改「全局按轮次发牌」后 **13,200 步 100.0%**（真实 train split）。护栏 15 项（含 workers=0/2 **逐位一致**的真实 spawn 等价性），默认 CI **1033 passed**。**验收已实测**（`--resume latest` 从 step 4000 起，`data.workers=3`，索引缓存命中、**未跑门禁**）：
步时中位 **0.539 → 0.099 s（5.4×）**、峰值显存 4.87 GiB 无回退、K 中位 25（旧 26，无偏斜）。未做：
S5 巡检、worker 侧 `r==0` 计数旁路汇总。同轮另登记：**遮盖退化 `r==0`** 占约 **1.6%** 的步
（**第十轮复核纠正：数完整日志是「约 1.0% 的全部步 / ≈1.8% 的非空批」，且 43.8% 的步本来就天然
是 `r==0`（空窗合法），该问法必须带口径——见 plan 07 §9-43 与 §9-46**）。
- ⚠️ **第十轮：把 GPU 利用率量准 + 改正第九轮的读法（2026-09-27，已修）**：第九轮报「`perf/data_share`
  中位 **5.0%** ⇒ 数据侧已不是瓶颈，下一步是 H2D 重叠、**不要再加 worker**」——**方向完全反了**。
  该分布**极端重尾**：中位 5.0% 与**均值 45.0%** 同时为真（前 10% 的步吃掉 77.4% 的数据时间），
  而决定吞吐的是均值。**三量具读数一致**：`data_time_s` 均值 45.0%；`nvidia-smi` 200 ms×750 =
  **均值 44.3% / 中位 47.0%**，且 **42.7% 的采样 ≤5%（完全空转）**；`py-spy dump` 主线程 70 次里
  **45.7% 停在 `_try_get_data`**（worker 队列）。**逐项排除**：盘（原始读 2006 MiB/s）、解压
  （npz 压缩比 1.09×）、`pin_memory`、主进程 CPU（0.05 核）、OMP 线程、Defender——都不是瓶颈。
  **标定曲线**（`scripts/local_worker_scaling.py`）：供给 **2.51 / 4.42 / 6.33 / 5.75 / 6.28 窗口/s
  @ N=3/6/8/10/12**，**N≥8 封顶**，而需求约 11.5 窗口/s。**根因**：计划层跨 991 桶轮转发牌 ⇒
  一张谱的 ~96 个窗口相隔 ~991 个槽位，`chart_cache_size=8` **命中率必然为 0**，同一张谱每个
  epoch 被完整重解析 **96 次**（解析 44% + 建场 33% + 解压 12%）——**这就是换覆盖率付的代价**，
  调参数绕不开。**已修**：`data.workers: 3 → 8`（纯配置 + `OMP_NUM_THREADS=1`），同分布段间实测
  **GPU 利用率 44.3% → 75.5%**（空转 42.7% → **9.5%**）、`data_share` **46.4% → 22.4%**、
  步时 mean **0.296 → 0.195 s**、data 等待 p90 **0.451 → 0.0118 s（38×）** ⇒ **判据 3b 首次达标**；
  跑完全段（16201–20000，n=4050）后 `data_share` **19.0% < 20%，判据 3a 也达标**。
  ⚠️ `commit charge` 已达 **38.1/40.1 GiB（95%）** ⇒ **N=8 同时
  是内存上限**，再加 worker 会踩页文件。⚠️ **量具陷阱**：PowerShell 管道会缓冲 `nvidia-smi --loop-ms`
  输出（第一版脚本 0 行样本）；`py-spy record` 在 Windows **静默丢样本**（把 45.7% 的数据等待报成
  0.2%），长跑诊断只用 `dump`。
- ✅ **供给管线队头阻塞已修（第十一轮，§9-50，已落地）**：`torch` 的 `DataLoader` 默认
  `in_order=True`，而它的 `_try_put_index` **只在按序交付时**被调用 ⇒ 队头一条慢窗口（p50 0.57 s /
  p99 5.9 s）让 8 个 worker 集体等索引。改为 `in_order=False` + **每批自带槽位标签、主进程
  重排回计划顺序**（`train_loop.py` 的 `SlotTag` / `_SlotTaggedDataset` / `_collate_with_slot`）。
  实测（`scripts/local_inorder_ab.py`，真实 train split、ABBA 交错、60 s×2 臂）：供给
  **6.39 → 8.67 窗口/s（1.36×）**，交付 p90 **561-575 → 15-139 ms**；不重排的裸
  `in_order=False` = 8.71 ⇒ **重排代价 0.5%**。**样本序列与覆盖率逐位不变**（`windows_seen`
  轨迹对齐 389 步），护栏 `tests/unit/infra/test_inorder_pipeline.py`（6 项，假 DataLoader 造
  「整段倒序交付」的最坏情形）。⚠️ 代价：p99 2.1-2.3 → 2.8 s（重排把尾部等到了队头）。
  ⚠️ **仍差需求（11.5 窗口/s）约 25%，但不要靠加 worker 补**：内存是硬上限，且**端到端剩余
  空间只有约 1.29×**（compute mean 0.151 s vs step mean 0.195 s）。下一层是**物化侧的整谱重复**
  ——每窗口 **91.3%** 的 CPU 花在「一张谱只该做一次」的事上（RFC-0035 的 M0/M1/M2，**待裁定**）。
- ✅ **窗口预切缓存建成并验收（第十二轮，§9-52，已落地）**：按**窗口**物化「重建稠密所需的最小记录」
  （稀疏计数 + token 遮盖位图 + 轨道 + 音频），训练期只做 mmap 读取。**实测真实切片 50 行 / 4581 窗：
  逐位一致 4581/4581 ✅、读取 9.6 ms/窗（单进程 104.7 窗口/s，原路径 0.85 s/窗 ⇒ 89×）、
  559.8 KiB/窗 ⇒ 全库 339 GiB**。开关 `data.window_cache_dir`（进 `RESUME_IGNORED_KEYS`）。
  ⚠️ **它推翻了 RFC-0035 §1 的关键数字**：全谱工作不是 91.3%，实测**首窗 530 ms / 后续窗 287 ms
  ⇒ 46% / 54%** ⇒ M0/M1/M2 只能打那 46%，**逐窗口的 54% 一个也打不掉，而窗口预切两半都打掉**。
  ✅ **全量已建成并终验通过（第十三轮，§9-54⑮）**：train **634 952 窗 / 353.6 GB / 11.4 h**（15.5 窗/s）
  + val **74 889 窗 / 41.7 GB / 1.2 h**（17.3 窗/s），指纹与事前计算一致、**逐位一致 400/400 + 100/100**、
  读取 95.2 / 76.2 窗/s（原路径 81× / 65×）。旧「上限 ≈13 窗/s」系短窗有界探针低估；`--jobs 12` 全程
  实测 **15.5–17.3 窗/s**（12 进程仍是拐点，20 进程更慢且吃内存）。
- ✅ **第十三轮 G2 回归已裁定收束（RFC-0037 采纳，2026-09-28）**：决策者裁定 **G2 的设计本身是错误，直接删除**（BasePlan §9 / 红线 7 / AGENTS §4 已修宪为 G1/G3/G4 三道 + val held-out 置换对照判读红线）；同案裁定**训练损失按事件归一**（整式除以 max(E_total,1)，argmin 不变）+ 附带指标（`loss_sum_raw`/`loss_nonempty_per_event`/`clip_active`）+ 门禁臂 fp32（原 P0）+ val 置换改线内（原 P1）。RFC-0036 → **废弃**（诊断记录保留为证据）。实现与护栏见 plan 07 §9-56。原始回归记录如下（历史）：
  `--gates --device cuda` ⇒ G1/G3/G4 全绿、**G2 FAIL**（真实 182.856 vs 打乱 −31.199，需 ≥192.0），退出码 5，
  **训练未启动**（记录 `runs/phigros_masked/20260927-180550/gates.txt`）。**逐文件核对**：`model.py` /
  `losses.py` / `masks.py` / `sanity.py` 自 `b96b12f` 起**未被触碰**，`shuffle_hidden_counts` **逐字未变**，
  `tracks.py` 的向量化在上一次全绿（`45a46cd`）之前就在场 ⇒ 唯一改到门禁输入的是 `cc0e166` 的**取批计划层**：
  `_consume(16)` 取计划**头部**，而 16 个连续窗口＝**一张谱的开场**（实测第 0 批 K=24 / 63 事件 /
  **只用 1 条判定线**；全库均值约 17.4 事件/窗）。**G2 随批剧烈变化**：本轮 6 个批 **5 FAIL / 1 PASS**
  （线数 5 FAIL / 7 PASS、事件数 **248 仍 FAIL**、事件/线 71 PASS 而 26 FAIL，**没有单一量能分开**）
  ⇒ **不能靠调 `gates.batch_min_events` 追绿**（那是拿结果反推判据）。**要裁定的两件事（须 RFC）**：
  ① 门禁批的**抽取口径**（不得只取计划头部、必须跨桶跨谱取、批构成落盘可复查）；
  ② G2 判据的**统计效力**（单批抽样 vs 多批平均 + 方差）——⚠️ 这意味着**此前所有 G2 记录都不可复现**。
  见 plan 07 §9-54。**2026-09-28 决策者质疑 G2 有效性本身，复核成立**：现行 G2 = 单批 100 步
  「速度赛」，判据量是未归一化 sum 损失（实测 loss ≈ 2.24·K^0.055·E^0.93，R²=0.74），真实批上
  判别力未被证明；RFC-0036 增补 §2.8 与选项 **D（held-out 置换对照）/ E（退役 G2，改用已落地的
  `val/nll_shuffled_delta` + `val/ratio` + G3/G4，触及红线 7 文字）**，见 plan 07 §9-55。
  **同轮实测更正**：短窗探针的「构建速率 ≈13 窗/s 与进程数无关」被全量构建**证伪**——
  `--jobs 12` 实跑 **train 15.5 窗/s / 11.4 h、val 17.3 窗/s / 1.2 h**（合计 395.3 GB），加进程仍只吃内存
  （`--jobs 20` 更慢）；缓存终验**逐位一致 400/400 + 100/100**（§9-54⑮）。
- ✅ **val 落地且开销已收敛（§9-51 / §9-53）**：固定验证集 + **同一遮盖测度下重算的常数基线** ⇒
  主判据 `val/ratio`，**`best.pt` 改按它选**；单步 loss 按双峰实测**分层**（`loss_empty`/`loss_nonempty`）
  + `integral_per_event`。**§9-47 E 的 5.8% 算式三项全偏**（实测数据侧 0.230 s/窗、对照臂 5 次前向、
  val 前缀 **ΣK² = 422 139**）⇒ 落成后一次 val ≈ **预算的 0.7–1.5×**。§9-53 两处结构修复：
  **按网格桶成组**（128 窗 128 批 → **17 批**，`Σ(max K)²` **422 139 → 71 165 = 5.93×**）+ val 吃窗口缓存
  ⇒ 一次 val ≈ **33 s = 16.7%**；配 `val_every=4000` ⇒ **4.1%**。⚠️ 组批改变归约顺序 ⇒ 数值
  **不再与组批前逐位相同**（指标语义不变）。
- 🔵 **显存墙已解除（第七轮，RFC-0032 + bf16）**：旧写法让**每条线的 token attend 全部 K 条线的事件轨**（`tracks.repeat_interleave(n_lines)`），显存 ∝`K²`、8 GB 卡上安全上限仅 **K≈31**，而 train 有 36.4% 的谱面 K>31 ⇒ **全量训练根本开不起来**。改成本线轨后 ∝`K`，再接入配置里早已声明却从未被读的 `optim.precision: bf16-mixed`（峰值 0.58×、步时 0.49×），**K=128 亦可训**。实测：K=20 2.46→1.31 GiB、K=32 5.05→1.86、K=128 → 5.10 GiB（配 bf16）。见 [RFC-0032](docs/decisions/RFC-0032-local-layer-own-line-tracks.md)、plan 07 §9-35~§9-37。
  ⚠️ **但那是 torch 分配器口径、单样本**：全量训练实测**驱动侧照样贴顶并卡死**（§9-57，共享显存 13.2 GB），
  且 **`val` 每轮都跨墙**（共享显存 5.6 GB，**只变慢不崩**，§9-59）⇒ **跨墙 ≠ 必然卡死**，
  判据必须用 `nvidia-smi memory.used` 且要排除 val 段。
- 🔵 **长跑运维已落地（第六轮）**：断点续训 `--resume [latest|目录]`（配置指纹 / 门禁全绿 / data_rev 三项校验，fail-closed；指纹只覆盖语义字段，故可安全延长 `max_steps`）、checkpoint 旋转（`save_every=2000` 即每 4 轮、`keep_last=3`、`keep_best=1`，先写后删）、**在线**标量（TB + `logs/loss_history.jsonl`，每 `log_every=50` 步刷新）、巡检脚本 `scripts/training_health.py`（`--watch 7200`，退出码 0/1/2）。见 plan 07 §9-29~§9-33 与 [docs/TRAINING.md](docs/TRAINING.md) §7.5。
- ⚠️ **候选决策（不阻塞训练，但影响扩张）**：
  - **遮盖通道的信息泄漏**（plan 04 §9-18）与**遮盖重标定系数** `1/r` vs 文档 `1/(1-r)`（plan 04 §9-6）——两者都已给出默认口径并如实记录；
  - RFC-0030 登记的两项**新发现**：① `JudgeLine.pose_at` 的父线位置合成与 prpr A 级证据不一致（父线旋转未作用在子线平移上 ⇒ 含父线谱面的跨线几何计数是**近似值**）；② plan 05 §4.2-1 的 τ 边缘强度算式漏了 `J(τ)`（已按算式笔误修正并登记，见 RFC-0030 §6）；
  - ~~**global 层的 `O((K·T)²)`**~~ **已被实测推翻并解决**：墙在**局部层**，根因是条件注入的 K/V 作用域，修法见 RFC-0032（plan 04 §9-22 已就地更正）；
  - **装饰线旁路**（决策者已定：用额外旁路，**标记为后续扩展**，等主路线训练完成后再做）：λ 的线轴改为「承载有效音符的线集合」，装饰线不进模型/损失、导出时原样写回；无条件生成时由谁产出随之裁定（[survey §7.7](docs/knowledges/phira-dataset-survey.md)、plan 04 §9-24）；
  - ~~**val 路径与模型选择口径**~~ **已落地**（`best.pt` 按 `val/ratio` 选，见上）。**新的未决项**：val 开销要不要再压
    （单次实测 **100-168 s、跨墙**，`val_every=4000` 只压频率，§9-59）、以及 `val_windows=128` 的**构成未收敛**（§9-51 ⑤）。
- ⚠️ **首跑全量 20000 步完成：管线跑通，但这一版目标/优化器零进展（2026-09-29，第十三轮⑧）**：
  `runs/phigros_masked/20260928-144342` **exit 0、无卡死**，门禁 G1/G3/G4 全绿，谱面覆盖 **6614/6614（100%）**，
  `data_share` 中位 **2.9%**（窗口缓存 85×）；**但**非空窗 per-event 训练损失按 2000 步分块中位
  **9.52 → 10.24 无趋势**、`val/ratio` 最优停在 **step 2000（0.7292）**，之后 18 000 步无一超过
  ⇒ **20 000 步里约 19 000 步买到 0**。**判读红线单点破**（`val/ratio = 1.0033` @step 3000，其余 19 点为
  0.73–0.97；`shuffled_delta` **20/20 > 0**，均值 +0.688）⇒ 依 **RFC-0037 R1 该 run 扩规模结论作废**；
  `best.pt` = **step 2000**。**新证据**：音频条件的边际价值 **≤0.6%（帧轴置换）/ ≤6%（置零）**，
  事件轨 4.6–30% ⇒ **当前训练目标没有奖励模型使用音频**（未消融，勿下结论）。**量具缺陷**：val 占墙钟
  **38.3%**（2234.9 s）而只有 **498 事件 / 57.8% 空窗**、残差 sd **0.067** ≈ 漂移 0.13 ⇒ 红线判据
  **检验力不足**。处置与下一步见 [RFC-0038](docs/decisions/RFC-0038-training-saturation-and-budget.md)
  （**提案，待裁定**）与 plan 07 **§9-61**。
- ⚠️ **硬件安全（2026-09-27 事故）**：一次 GPU 探测（`sdpa_kernel` 强制后端 + 无界前向/反传扫批）触发驱动层 TDR，**Windows 被重启**。三条禁令已写进 [docs/TRAINING.md](docs/TRAINING.md) §7.5：不强制注意力后端、不跑无界 GPU 扫批、同一时刻只跑一个 GPU 作业并全程盯功耗与显存。

## 7. 文档导航

| 文档 | 用途 |
|------|------|
| [README.md](README.md) | 快速开始与总览 |
| [docs/BasePlan.md](docs/BasePlan.md) | **技术奠基（最高权威）** |
| [docs/CODE_STRUCTURE.md](docs/CODE_STRUCTURE.md) | 代码结构详解 |
| [docs/plans/](docs/plans/README.md) | 各模块实施计划（00-08） |
| [docs/decisions/](docs/decisions/README.md) | RFC 决策记录（**RFC-0029 为当前范式权威**） |
| [docs/knowledges/](docs/knowledges/) | 格式/单位/数据/文献事实库（Phigros RPEJSON 等） |
| [docs/POSTMORTEM-2026-08-05-frame-rate-misalignment.md](docs/POSTMORTEM-2026-08-05-frame-rate-misalignment.md) | 帧率事件根因与门禁由来 |
| [docs/TRAINING.md](docs/TRAINING.md) | 训练操作手册 |
| [docs/glossary.md](docs/glossary.md) | 术语表 |
| [AGENTS.md](AGENTS.md) | 子 agent 协作约定 |
