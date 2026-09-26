# 术语表

> BeatMorph v3.0（Phigros 范式）术语速查。**每条一句话定义**，附权威出处；与 [`BasePlan.md`](BasePlan.md)（最高技术权威）冲突时以 BasePlan 为准。
> 编排：先按主题分节，节内先拉丁字母条目（字母序）再中文条目；**已淘汰术语集中在 §5 并标注「（v2.x 已退役）」**。
> 事实分级沿用 [knowledges/](knowledges/) 的约定：**A** = 参考实现源码 / 官方文档，**B** = 社区 wiki，**C** = 二手。

---

## 1. 格式与几何（RPEJSON / 坐标系）

| 术语 | 定义（一句） | 权威出处 |
|------|-------------|---------|
| **above** | RPEJSON note 的侧别字段：`above == 1` 表示音符从判定线**正面**下落、**其余值**从背面下落 —— 实现必须写 `above == 1 ? FRONT : BACK`，**不得当布尔解析**（实测取值域 `{0,1,2}`，0 与 2 都出现）。 | [phigros-format.md](knowledges/phigros-format.md) §5.1/§5.3；RFC-0029 §8.1 Q3 |
| **beat 三元组** | RPEJSON 的时间表示 `int[3]`，语义为 `beat = [1]/[2] + [0]`，`秒 = 60/BPM × beat`；多 BPM 段由根级 `BPMList` 分段积分换算（**A** 级 prpr `Triple(i32,u32,u32)`）。 | [phigros-format.md](knowledges/phigros-format.md) §5.1/§7.1；[units](knowledges/phigros-units-and-geometry.md) §6.5 |
| **eventLayers（事件轨）** | 判定线的事件层级：**4 层普通事件**（每层 5 条轨：`moveXEvents` / `moveYEvents` / `rotateEvents` / `alphaEvents` / `speedEvents`）+ `extended` **第 5 层**；**事件必须跨层求和，不是取最上层**，且事件间隙必须**补洞**。 | [phigros-format.md](knowledges/phigros-format.md) §4.2/§4.3；RFC-0029 §8.1 |
| **isFake** | RPEJSON note 字段：值为 **1 表示假音符**（无判定、无打击特效与音效、不计分、不计物量），其余值为真音符。 | [phigros-format.md](knowledges/phigros-format.md) §5.4 |
| **judgeLineList（判定线列表）** | RPEJSON 根级的判定线数组：K 在**单张谱内恒定、跨谱可变**（实测中位 **30** 条，p25=24 / p75=52 / min=2）。 | BasePlan §3.2.3；[survey](knowledges/phira-dataset-survey.md) §7.1 |
| **lineLength** | Phira `info.yml` 字段：判定线在**局部系**中的**半长**，单位 = 半屏宽（675 RPE-x 单位），默认 **6.0** ⟹ 半长 4050 = 3 倍屏宽；**与 `positionX` 完全无关**（**A** 级 prpr `draw_line(-len,0,len,0)`）。 | [units](knowledges/phigros-units-and-geometry.md) §4 |
| **note type（tap / drag / hold / flick）** | RPEJSON 的 `type` 数字：**1 Tap / 2 Hold / 3 Flick / 4 Drag**（**A** 级源码确证）；⚠️ **官谱 JSON 的同一数字含义不同**（2=Drag / 3=Hold / 4=Flick，C 级）—— 混用会**静默全员错位**。 | [phigros-format.md](knowledges/phigros-format.md) §5.2；[units](knowledges/phigros-units-and-geometry.md) §6.7 |
| **PEC** | PhiEditer 的**旧格式（已淘汰）**，**不是**官方格式；Phira 仍可解析但主路径不采用。 | RFC-0029 §8.1 Q3；[phigros-format.md](knowledges/phigros-format.md) §2.1 |
| **positionX** | note 沿判定线局部 x 轴的落点，单位 = **RPE 舞台系 x 坐标单位**（1 单位 = 舞台宽 1/1350），语义可见范围 **[−675, 675]**，与 `moveX` **同刻度**、**与 `lineLength` 无关**（**A** 级 prpr `position_x / (RPE_WIDTH/2)`）。 | [units](knowledges/phigros-units-and-geometry.md) §3；RFC-0029 §3.1 |
| **RPEJSON** | Re:PhiEdit 的标准谱面格式，Phira 原生支持（**忽略文件后缀**、须按内容判型），本项目**主路径唯一权威格式**。 | [phigros-format.md](knowledges/phigros-format.md) §2/§3；RFC-0029 §8.1 Q3 |
| **side** | note 从判定线**哪一侧**落向判定线（局部 +Y 侧 / −Y 侧），格式字段为 `above`；两侧几何上是**关于判定线 X 轴的 Y 镜像**，是**独立于 positionX 的硬自由度**（实测背面仅 **2.4–3.0%** → 评估须单独报背面 recall）。 | [units](knowledges/phigros-units-and-geometry.md) §2.1；BasePlan §3.2.2 |
| **visibleTime** | note 在被判定前多少**秒**开始显示（淡入提前量）；默认值裁定 **999999.0**（`99999` 判为笔误）。 | [units](knowledges/phigros-units-and-geometry.md) §6.4 |
| **yOffset** | note 的局部 Y 轴偏移，单位 = RPE-y 坐标单位；**实际偏移 = `yOffset × note.speed`**（speed = 0 时恒为 0），方向随 side 镜像（**A** 级 prpr）。 | [units](knowledges/phigros-units-and-geometry.md) §6.3 |
| **官谱 JSON（phi / Official）** | Phigros 官方本体的内部格式（**未公开文档化**）：note 分 `notesAbove`/`notesBelow` 两个数组、时间单位为 1/32 拍（`1T = 1.875/bpm` 秒）、每条判定线自带 `bpm` —— **不在本项目主路径**。 | [phigros-format.md](knowledges/phigros-format.md) §1.1/§5.3/§7.1 |
| **判定线（judge line）** | Phigros 的**一等舞台对象**：拥有独立局部坐标系（锚点为原点、判定线本身为 X 轴、锚点朝向为 +Y 正半轴）与完整 animate 事件轨，且可经 `father` 嵌套（实测 26% 的谱面含嵌套线）。 | [phigros-format.md](knowledges/phigros-format.md) §4/§6.1；RFC-0029 §2.2 |
| **局部系 / 舞台系 / 屏幕系** | 三套坐标系：**局部系 L**（每线一个，谱面的**原生存储系**）→ **舞台系 S**（RPE，`[−675,675]×[−450,450]`，即 1350×900）→ Phira **世界系 W**（`x∈[−1,1]`、`y∈[−1/ar,1/ar]`）→ **屏幕像素系 P**（`u = 0.5 + x/1350`、`v = 0.5 − y/900`）；**本项目在局部系建模**（红线 1），舞台永远铺满视口，`aspectRatio` 只改像素长宽比、不改谱面语义。 | [units](knowledges/phigros-units-and-geometry.md) §2/§5；CLAUDE.md 红线 1 |
| **RPE 舞台尺寸** | `RPE_STAGE_WIDTH = 1350`、`RPE_STAGE_HEIGHT = 900`（**A** 级源码常量 `RPE_WIDTH`/`RPE_HEIGHT`），是 `positionX` 网格与速度换算的定义域。 | [units](knowledges/phigros-units-and-geometry.md) §2.2/§7.2 |

## 2. 建模范式（点过程 / 掩码 / 损失）

| 术语 | 定义（一句） | 权威出处 |
|------|-------------|---------|
| **Adapter / LoRA** | 冻结主干（MERT）之外附加的**可训练轻量模块**；本项目按 RFC-0003 采用 LoRA，注入 attention 的 q/v。 | BasePlan §3.1；[RFC-0003](decisions/RFC-0003-adapter-lora-vs-mlp.md) |
| **AR Transformer（对照臂）** | 逐 token 自回归 Transformer；v3.0 **不再是主线**，保留为 **B3（GOCT 配置）** 与 **B4（自回归上界臂）**。 | BasePlan §3.3/§3.2.1；RFC-0029 §5.2 |
| **B1-B6 对照臂** | 必须全部实现的六个实验臂：**B1** 热图+focal（文献主流，正式消融臂而非稻草人）、**B2** 主线（泊松 NLL + 掩码补全）、**B3** 离散 event token + AR（GOCT 配置）、**B4** 自回归上界臂、**B5** 掩码离散扩散（absorbing）、**B6** 解码策略消融（`find_peaks` vs Ogata thinning）。 | RFC-0029 §5.2 |
| **MERT-v1-330M** | 音乐自监督预训练音频编码器（**24 层、hidden = 1024、采样率 24000 Hz、全部冻结**），Stage 0 主选，输出帧级表征 `audio_emb [B,T,1024]`。 | BasePlan §3.1 |
| **Ogata thinning** | Ogata (1981) 的**舍选采样**算法，从强度场 λ **直接采样**事件；是 `find_peaks`+阈值的原则性对照解码器（B6）。 | RFC-0029 §3.4；[literature](knowledges/chart-generation-literature.md) §4.2 |
| **ChartField（强度场 λ）** | 谱面的连续表征 `λ_k(t, x, s, c) ≥ 0`（k = 1..K 条判定线；x = positionX 网格；s = side；c = 类型通道 tap/drag/hold/flick + hold-end 标记），网格桶宽 **`Δx = RPE_STAGE_WIDTH / N`，默认 N = 128 → 10.546875**（**必须派生，不得硬编码**）。 | RFC-0029 §3.1；BasePlan §3.4 |
| **累积强度 Λ(t)** | Omi et al. (NeurIPS 2019, arXiv 1905.09690) 的参数化：网络输出**累积强度** Λ(t) 再求导得到 λ，使 `∫λ` 可**精确**计算、免数值积分；`field/` 须与网格积分路径**互校一致**。 | BasePlan §3.4；[literature](knowledges/chart-generation-literature.md) §4.2 |
| **非齐次泊松 NLL** | 训练目标 `L = −Σ_n log λ_{k(n)}(e_n) + Σ_k ∫∫∫ λ_k dt dx ds`：事件处强度必须高、**全域积分必须低**（惩罚"到处乱亮"以及塌到全 0 的解）。 | RFC-0029 §3.2；BasePlan §3.4 |
| **标记点过程（marked point process）** | 把一张谱面看成判定线局部系上一次点过程的实现，**标记 = (line_id, positionX, side, type, hold_time, speed, is_fake, ...)**；note 落到哪条线是 K 个强度场**竞争**的自然结果，不需要额外分类损失。 | RFC-0029 §2.3/§2.4-6 |
| **掩码补全（masked completion）** | 训练时遮盖目标场片段、**只在被遮盖位置**计算损失，推理时**迭代并行**填充；**必须显式提供 mask 通道**（否则分不清"无 note"与"被遮盖"），且**按事件遮盖而非按帧遮盖**。 | BasePlan §3.3；RFC-0029 §3.3 |
| **吸收态离散扩散（absorbing-state diffusion）** | 以 [MASK] 为**吸收态**的离散扩散；其 NELBO 等价于**按时间步加权的掩码交叉熵**（arXiv 2510.03289 式(10)）→ 与掩码补全的差异在**采样/加权层**，消融必须指明对照层级。 | RFC-0029 §3.3；[literature](knowledges/chart-generation-literature.md) §5.1 |
| **帧率派生量** | 帧率**不是超参而是派生量**：`MERT_FRAME_RATE_HZ = MERT_SAMPLE_RATE_HZ / MERT_CONV_STRIDE_PRODUCT = 24000 / 320 = 75 Hz`（经验对照：5 s 输入 → **374 帧** = 74.8 Hz）。 | BasePlan §3.1；[POSTMORTEM](POSTMORTEM-2026-08-05-frame-rate-misalignment.md) §2.1 |
| **多判定线（v1 即引入）** | v1 **不限制单条线**：输出层不得硬编码 K，须用**共享权重 + 每条线一个 line embedding（作 query）**；判定线**不可互换**（各有独立事件轨，排序不变性不成立）。 | RFC-0029 §2.4；BasePlan §3.2.3 |
| **条件化（判定线事件轨）** | Q2 决议：线事件轨作为**条件输入**（编码为舞台上下文，走 cross-attention），**v1 不做联合生成**。 | RFC-0029 §8.2 Q2；BasePlan §3.5 |

## 3. 评估、实验与工程流程

| 术语 | 定义（一句） | 权威出处 |
|------|-------------|---------|
| **G1-G4 门禁** | 四道**范式中立**的健全性门禁（实现在 [`beatmorph/infra/sanity.py`](../beatmorph/infra/sanity.py)）：**G1** 单 batch 过拟合、**G2** 打乱标签对照（loss 必须显著变差）、**G3** 常数基线（`λ = N/\|Ω\|`，**不是 λ=0**，后者 NLL = +∞）、**G4** 契约断言（帧率/形状由 config 派生）；**任何新模型/新范式在扩大数据规模之前必须全绿**，结果写入训练日志。 | BasePlan §9；CLAUDE.md 红线 7；[POSTMORTEM](POSTMORTEM-2026-08-05-frame-rate-misalignment.md) §6 |
| **Phira API** | 唯一可得的万级 Phira 谱面入口 `GET https://api.phira.cn/chart`（实测 `count = 9649`）；三个坑：响应键是 **`results`**、`pageNum` **上限 30**、**无按定数过滤参数**。 | [survey](knowledges/phira-dataset-survey.md) §2/§3 |
| **Range 预筛** | 谱面包 CDN **支持 HTTP Range**（206），可只抓 zip **中央目录 + 单条目前缀**来判型/定位，**省约 90% 带宽**（全量直抓约 **76 GB**，推断）。 | [survey](knowledges/phira-dataset-survey.md) §2.1.4/§9.2 |
| **plan** | 各模块实施计划（接口契约 / 里程碑 / 风险 / 测试策略），见 [`docs/plans/`](plans/README.md)；**当前索引仍是 v2.x 旧编号 00-09，待随 v3.0 重写**，目标编号见 [AGENTS.md](../AGENTS.md) §2。 | CLAUDE.md §7；AGENTS.md §2 |
| **RFC** | Request for Comments：记录技术决策或**偏离 BasePlan** 的议案，见 [`docs/decisions/`](decisions/README.md)；**当前范式权威是 RFC-0029**。 | CLAUDE.md §3 红线 1、§7 |
| **事件级 F1** | 主判据：**贪心一对一匹配**、**未匹配的生成事件留在分母**、**±20ms（DDC 口径）与 ±50ms（GenéLive! 口径）双报**，并按谱平均与 micro 平均都报。 | RFC-0029 §5.1 |
| **数据合规（已裁定）** | Phira ToU **未授予机器学习训练权利**且禁止未获授权创建衍生作品，**代码许可证（GPL-3.0 / MIT）不覆盖用户上传内容**。**2026-08-05 裁决：风险由决策者承担**，训练可启动；**硬约束 = 最终不发布模型权重**（裁决成立的前提）、数据可本地落盘但不得入库、脚本留存来源与用途、发布权重前须重新裁定。 | CLAUDE.md 红线 5 附注；RFC-0029 §8.3 Q11b |
| **红线** | CLAUDE.md §3 的 **7 条不可违背约束**：① 技术选型锁定 v3.0 范式 ② 模块间只经 `core/contracts` 通信 ③ AI 与规则解耦（后处理只做物理/格式钳位）④ 单一目标 Phigros ⑤ 数据/权重不入库 ⑥ 100% 可玩/合法 ⑦ 物理常量必须派生 + 断言、新范式必须过 G1-G4。 | CLAUDE.md §3 |
| **人评协议（MIREX 2026 三段式）** | 最终金标准：① 算法产出 → ② 约 5 名 mapper 专家评 *musical representation / creativity / gameplay* → ③ **社区混入人类谱盲测**（令玩家指认哪张是人写的）。 | RFC-0029 §5.3 |
| **R 编号** | BasePlan §6 的风险表条目 **R-1..R-8**（R-1 MERT 表征不足 / **R-2 数据合规** / R-3 数据质量 / R-4 稀疏塌陷 / R-5 多线长尾 / R-6 解码精度 / R-7 物理常量漂移 / R-8 跨线冲突）；**R-2 是唯一可能终止项目的一条**（许可风险，非技术风险）。 | BasePlan §6 |

## 4. 度量与数值速查（供写作引用）

| 量 | 值 | 出处 |
|----|----|------|
| MERT 帧率 | `24000 / 320 = 75 Hz`（5 s → 374 帧） | BasePlan §3.1；[POSTMORTEM](POSTMORTEM-2026-08-05-frame-rate-misalignment.md) |
| 舞台 / 网格 | 舞台 1350×900；`Δx = 1350 / 128 = 10.546875` | RFC-0029 §3.1；[units](knowledges/phigros-units-and-geometry.md) §7.1 |
| 速度单位 | 1 速度单位 = **120.23 RPE-y 单位/秒** | [units](knowledges/phigros-units-and-geometry.md) §6.1 |
| 判定线数量 | 中位 **30** 条 | BasePlan §3.2.3；[survey](knowledges/phira-dataset-survey.md) §7.1 |
| note 类型分布 | Tap 52–63% / Drag 20–31% / Hold 10–11% / Flick 6–7% | BasePlan §3.2.2；[survey](knowledges/phira-dataset-survey.md) §7.3 |
| 背面 note 占比 | 2.4–3.0%（`above ≠ 1`） | BasePlan §3.2.2；[survey](knowledges/phira-dataset-survey.md) §7.3 |
| 最忙判定线占比 | 独占中位 **73%** 的 note（仅中位 58% 的线带 note） | BasePlan §3.2.3；[survey](knowledges/phira-dataset-survey.md) §7.2 |
| 数据规模 | Phira 全站 **9649** 张；单包中位 6.56 MB；全量约 76 GB（推断） | [survey](knowledges/phira-dataset-survey.md) §3 |
| 格式分布 | RPE **97.2%** / PEC 2.5% / 官谱 JSON 0%（n=283 抽样） | BasePlan §4.1；[survey](knowledges/phira-dataset-survey.md) §5.2 |
| 人类落点落在 onset ±100ms 内 | **89.0%**（n=39,136）→ 谱面生成是**诠释**而非转录 | BasePlan §1.3；RFC-0029 §5.1 |

## 5. 已退役 / 非主路径术语

| 术语 | 状态与一句说明 | 出处 |
|------|---------------|------|
| **VQ-VAE（v2.x 已退役）** | 把小节 note 集合量化成离散码本的范式；本质是"小节级最近邻检索重组"，信息损失不可逆。 | BasePlan §3.2.1；RFC-0029 §4.3 |
| **码本坍缩（Codebook Collapse）（v2.x 已退役）** | VQ 训练中大部分码本索引不被使用的退化现象，随 VQ-VAE 一并退役。 | BasePlan §3.2（v2.x 章节）；RFC-0029 §4.3 |
| **Stage 1 密度规划 / Section / DensityPlanner（v2.x 已退役）** | 段级（每 4 小节）密度回归层：均值池化是**结构性信息瓶颈**（模型看不到它要预测的 onset），且 Phigros 不需要该层 —— v3.0 **取消**。 | RFC-0029 §4.3；[POSTMORTEM](POSTMORTEM-2026-08-05-frame-rate-misalignment.md) §2.6 |
| **BPE / event tokenizer（v2.x 已退役，存档）** | RFC-0028 的 "POS+NUDGE 无损离散化" 分词器；**保留存档**，其无损离散化思想对 Phigros 的 `positionX` 量化仍适用。 | RFC-0029 §4.3 |
| **osu!mania 4K / lane（v2.x 已退役）** | 4 条静止离散轨道的旧目标形态（`GameMode.MANIA_4K`、`Note(time,lane,type,duration)`）；旧资产归档 `archive/osu-mania` 分支。 | CLAUDE.md §1/§6；RFC-0029 §1 |
| **25 Hz 帧率假设（错误值，已修复）** | v2.x 三处硬编码的 MERT 帧率误值（真值 75 Hz，差 3×），在 mock 掩护下存活到万级数据规模 —— **已修复并制度化为 G4 门禁与红线 7**，保留为警示案例。 | [POSTMORTEM](POSTMORTEM-2026-08-05-frame-rate-misalignment.md)；[TRAINING.md](TRAINING.md) §7 |
| **Pass Rate（v2.x 已退役）** | 玩家通关率，曾作为 DPO 偏好对的数据来源；随 DPO 退出主路径一并退役。 | BasePlan §3.6（v2.x）；CLAUDE.md §2 |
| **DPO / RAG 风格检索 / Flow Matching（非主路径）** | **未退役但不在 v1 主路径**：RFC-0029 未涉及，列入 Phase 3 待重估（须立 RFC）。 | CLAUDE.md §2/§3.7；BasePlan §3.7 |
| **Demucs / HTDemucs（非主路径）** | 可选四轨声源分离增强；v3.0 BasePlan 未列入主路径，去留待定。 | [CODE_STRUCTURE.md](CODE_STRUCTURE.md) §3.1 |
| **`.osu` / `.sm` / `.ma2`（v2.x 已退役）** | 旧格式目标；`.osu` 解析器**保留可复用**（`.osu` 仍是通用格式），但不属主路径。 | RFC-0029 §4.3 |

## 相关文档

- [BasePlan.md](BasePlan.md) — 技术奠基（最高权威，v3.0）
- [CLAUDE.md](../CLAUDE.md) — 宪法、模块拓扑与七条红线
- [decisions/RFC-0029](decisions/RFC-0029-phigros-continuous-chart-generation.md) — 当前范式权威（Phigros + 标记点过程 + 掩码补全）
- [knowledges/](knowledges/) — 格式 / 单位几何 / 数据集 / 文献四条事实库
- [CODE_STRUCTURE.md](CODE_STRUCTURE.md) ｜ [TRAINING.md](TRAINING.md)
