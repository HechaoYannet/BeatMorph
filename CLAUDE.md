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
├── infra/                训练栈 + 健全性门禁 G1-G4（beatmorph/infra/sanity.py）
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
   - 任何新模型/新范式在**扩大数据规模之前**必须通过 **G1-G4** 四道门禁（单 batch 过拟合 / 打乱标签对照 / 常数基线 / 契约断言），结果写入训练日志。
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
8. **【v3.0 新增】门禁先行**：新增训练目标或损失函数时，**先把 G1-G4 跑通再谈扩数据**。

## 6. 当前状态（2026-09-26）

- ✅ **范式已定**：RFC-0029 采纳（Phigros + 多线标记点过程 + 掩码补全 + 泊松 NLL），Q1/Q2/Q4/Q6/Q7/Q8 已由决策者批复。
- ✅ **根因已清**：MERT 帧率 25Hz→75Hz 修复落地（三条独立证据链），G1-G4 门禁与 G4 契约测试入库并通过。
- ✅ **格式事实已就绪**：`docs/knowledges/phigros-format.md`（RPEJSON 逐字段、坐标几何、来源分级 A/B/C）。
- ✅ **信息准备已完成（四项全齐）**：[phigros-format.md](docs/knowledges/phigros-format.md)（格式，含勘误横幅）、[phigros-units-and-geometry.md](docs/knowledges/phigros-units-and-geometry.md)（单位/几何，prpr 源码 A 级）、[phira-dataset-survey.md](docs/knowledges/phira-dataset-survey.md)（数据源实测：9649 张、K 中位 30、音频 100% 捆绑）、[chart-generation-literature.md](docs/knowledges/chart-generation-literature.md)（文献与评估协议）。
- ✅ **两项阻塞已清（2026-08-05）**：① **数据合规已裁定**（风险由决策者承担，硬约束 = 最终不发布权重，见红线 5 附注）；② **Q15 时间网格定为 beat-aligned**（数学改写限定在 `field/` 内，见 BasePlan §3.2.4）。
- 🔵 **施工进行中**：plan 00 / 02 / 03 与 plan 04 主干（M1–M4/M6）已落地并通过默认 CI（545 项）；**field 与 generation 两套 G1-G4 门禁均在真实主干上实跑全绿**；plan 04 的消融臂（M7–M11）与 plan 05/06/07/08 待建。
- ⚠️ **待 RFC 裁定（2026-09-26 落地时发现）**：遮盖通道的信息泄漏（plan 04 §9-18）与遮盖重标定系数 `1/r` vs 文档 `1/(1-r)`（plan 04 §9-6）——两者都已在实现里给出默认口径并如实记录，详见交接件。


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
