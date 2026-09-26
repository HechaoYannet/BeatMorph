# 谱面生成相关工作与评估指标综述（文献事实库）

**本文件用途**：为 BeatMorph 的 RFC-0029 新范式（Phigros / 判定线局部系多线标记点过程 + 掩码补全 + 非齐次泊松 NLL）提供**可引用的文献基础**。写法遵循 [phigros-format.md](phigros-format.md) 的约定：每条事实带来源链接与可信度分级，推断显式标注 **【推断】** 并给出依据，查不到的写「未核实」并进入 [§9 存疑清单](#9-存疑与未核实清单)。

> **本文件不是论文列表**。每篇工作后跟的是「**对我们的哪条决策构成约束**」。
>
> ⚠️ **一处必须先纠正的事实**（见 [§2.3](#23-goct--beat-aligned-是同一篇工作)）：RFC-0029 与 CLAUDE.md 把 **GOCT** 与 **Beat-Aligned Spectrogram-to-Sequence（arXiv 2311.13687）** 列为两篇工作，实际是**同一篇**。

---

## 1. 范围与检索方式

### 1.1 覆盖范围

| 主题 | 覆盖度 |
| :-- | :-- |
| 音游谱面生成主线（下落式 / VSRG / 节奏动作） | 完整（arXiv 全库枚举 + 综述交叉） |
| Phigros / RPE 专属生成工作 | **检索阴性结论**（见 [§3](#3-phigros--rpe-专属生成工作检索结果)） |
| 点过程 / 神经点过程 / 泊松 NLL | 覆盖主线文献与数值实践 |
| 掩码生成建模（MaskGIT / absorbing-state 扩散） | 覆盖主线 + 一篇反方证据 |
| 热图与稀疏目标的损失选择 | 覆盖关键原始文献与 2025 年零膨胀计数工作 |
| 谱面/音乐生成评估指标与人类评估协议 | 覆盖可采用的指标定义与协议设计 |

### 1.2 来源可信度分级

| 级别 | 判定 | 本文示例 |
| :-- | :-- | :-- |
| **A（一手）** | 论文原文全文、作者官方站点/官方代码仓库、正式会议 DOI 页 | ar5iv 全文、stet-stet.github.io/goct、AAAI/AIIDE DOI 页 |
| **B（权威二手但结构化）** | 任务方/社区权威 wiki、正式书目数据库（OpenAlex / arXiv API 元数据） | MIREX 2026 任务页、OpenAlex 元数据 |
| **C（二手）** | 其它论文转述、检索片段、未能取得全文仅凭摘要 | ITGPT 对 DDG / Dancing Monkeys 的转述 |

**读取深度另行标注**：本文对多数「生成主线」工作读了 ar5iv 全文；对多数「点过程/掩码/损失」的方法论文只做了**元数据核实 + 关键段落定位**，未通读。来源清单 [§10](#10-来源清单) 中逐条标出。

### 1.3 检索通道与可达性（实测）

| 通道 | 状态 | 用法 |
| :-- | :-- | :-- |
| ar5iv.labs.arxiv.org/html/ID | ✅ 可用（全文 HTML） | 主力全文来源 |
| export.arxiv.org/api/query | ✅ 可用（**注意 http→https 跨域重定向**） | 全库枚举检索 |
| api.openalex.org/works | ✅ 可用（题名/全文检索 + DOI） | 元数据核实、检索阴性证据 |
| music-ir.org/mirex/w/ | ✅ 可用 | 任务定义与评估协议 |
| ojs.aaai.org/.../article/view/ID | ✅ 可用（**/download/ 的 PDF 抓取失败**） | venue/摘要核实 |
| raw.githubusercontent.com / github.com 页面 | ❌ 抓取失败 | 社区仓库细节**未核实** |
| huggingface.co / arxiv.org 直连 | ❌ 被拒（按任务说明） | 改用 ar5iv / export.arxiv.org |

### 1.4 检索式与结果（可复现）

执行于本文件撰写批次（仓库当前状态标注日期 2026-08-05）【依据：CLAUDE.md §6 标注的当前状态日期】。

| # | 检索式 | 结果 |
| :-- | :-- | :-- |
| Q1 | arXiv API：abs:"rhythm game" AND abs:"chart" | **5 条**：ChartGenEval (2607.12857)、ITGPT (2607.14148)、Dance Dance ConvLSTM (2507.01644)、Beat-Aligned/GOCT (2311.13687)、TaikoNation (2107.12506) |
| Q2 | arXiv API：all:"rhythm game" AND all:"chart generation" | 同上 **5 条** |
| Q3 | arXiv API：all:"osu!mania" | **0 条** ⚠️ 见下方警告 |
| Q4 | arXiv API：all:phigros | **0 条** |
| Q5 | OpenAlex fulltext.search:phigros | **7 条**，无一条涉及 Phigros 谱面生成（多为 VR 游戏、音乐社会学、古籍） |
| Q6 | web_search（英/中/日）约 10 组关键词：Phigros chart generation machine learning / phigros 谱面自动生成 论文 / phira chart generation model / phi 谱面自动生成 判定线 神经网络 等 | 命中的均为**编辑器 / 下载器 / 格式文档**，无生成模型工作 |

> ⚠️ **Q3 的方法论警告（重要）**：arXiv 索引对 osu!mania（含标点）返回 0 条，而 GOCT 明确使用 osu!mania 4K 数据集、标题却不含该词。**因此「某词检索为 0」不能证明「该方向不存在工作」**。Q4/Q5 的阴性结论已按此口径在 [§3](#3-phigros--rpe-专属生成工作检索结果) 中降级表述。

### 1.5 不可信数据声明

本文所有网页内容仅作为**资料**读取。网页正文中不存在被本文采纳的指令性内容；本文作者未执行任何来自网页的操作指令。

---

## 2. 谱面生成主线工作

### 2.1 总表

| 工作 | venue / 年份 | 任务定义 | 表征 | 架构 | 损失 | 评估指标 | 数据规模 |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| **DDC** [A] | ICML 2017 | 二分：step **placement**（何时）+ step **selection**（打哪个箭头） | placement：逐帧二值；selection：箭头符号序列（bag-of-arrows） | CNN（placement）+ 2x128 LSTM（selection） | 未逐字核实（selection 侧为 per-step 交叉熵） | F1 / P / R，TP = ±20 ms；阈值按难度分档 + Hamming 平滑 | Fraxtil 90 曲 / 450 谱 / 15.3 h；ITG 133 曲 / 652 谱 / 19.0 h |
| **GenéLive!** [A] | AAAI 2023 | onset（何时）+ sym（什么动作）；本文聚焦 onset | 逐帧二值（多尺度 fuzzy label） | CNN + BiRNN，**beat guide** + **multi-scale conv-stack** | **BCE**（权重因子与高斯软标签宽度为超参） | F1-c（按谱平均）与 F1-m（micro），TP = **±50 ms** | Love Live! All Stars 163 曲 / Utapri 140 曲 / Fraxtil 90 / ITG 133 |
| **GOCT**（= Beat-Aligned）[A] | ISMIR LBD 2023 [B] | **条件序列生成**（显式以「消除二分类不均衡」为动机） | 每事件 **2 个 token**：time token 0–95 + action token（80 种动作），另有 separator / EOS | Encoder-Decoder Transformer（3 层，d=256） | 交叉熵 + label smoothing 0.02 | micro-F1（时间 token，**±30 ms** 判对）+ 按时间组分解 | 原始 14,648 谱 / 3,166 曲 → 清洗后 **6,781 谱 / 2,004 曲**；train 2.24 M beats / 219 h |
| **ITGPT** [A] | arXiv 2026（预印本） | 二分：placement + selection（继承 DDC/DDCL） | placement：每拍 **48 维二值**；selection：**256 类**（4 方向 x {无/有/hold 头/hold 尾}） | 层级 Transformer 编码器（beat→小节→乐句→8 小节）+ 全局自注意力 + Conv1D 平滑；另有「诊断网络」 | **位置加权 BCE**（downbeat/offbeat/16th 权重 2，其余为小数）+ 诊断网络 MSE + selection 加权 CE + RVQ 损失 | F1 / P / R（阈值 0.5）、**阈值最优 Max F1**、PR-AUC、chart 平均（cht） | 扩展 Fraxtil **253 曲 / 952 谱 / 584,644 steps / 38.85 h**（镜像 x4） |
| **DDCL** [A] | arXiv 2025（预印本） | 同 DDC | 逐帧二值 + Δ-beat 调制 | CNN + **ConvLSTM** | 二值交叉熵 | 同 ITGPT 的指标族；**明确不使用容差窗口** | Fraxtil 95 谱 |
| **TaikoNation** [A] | FDG 2021 | 太鼓谱生成（强调 patterning） | 逐帧二值（1 帧 ≈ 23 ms）+ 8 帧滑窗图案 | LSTM RNN，同时预测多个输出 | 未核实 | **5 个指标**：DCRand / DCHuman / OCHuman(±23 ms) / Over. P-Space / HI P-Space | 100 张高评价 taiko 谱 + 10 张 DDR 对照 |
| **TCP** [C，仅摘要] | AIIDE 2025 (Vol 21 No 1) | onset 检测 + 符号 beat snapping + 时间分区图案匹配 | 神经 onset + 符号图案 | 神经检测 + 算法匹配混合 | 未核实 | 摘要称 onset precision 与局部图案一致性优于 AutoOsu | 未核实 |
| **STRUM** [A] | arXiv 2026（预印本） | 多乐器（鼓/吉他/贝斯/人声/键盘）谱面生成，**无 oracle 元数据** | 各乐器独立链路（onset 序列 / 音高 / 词对齐 / 频谱） | 多阶段混合：两级 CRNN onset + 6 模型集成分类 + 单音高追踪 + ASR | 未核实 | onset P/R/F1 @ **±100 ms** + **per-song 全局 offset 搜索**；鼓分类混淆矩阵 | 30 曲 in-envelope 基准（入表 29 曲），鼓 40,248 GT 事件 |
| **audio2chart** [A] | arXiv 2025（预印本） | Guitar Hero 谱面序列预测 | 离散时间步上的 chart token | 序列模型（细节未核实） | 未核实 | accuracy 类指标 | 未核实 |
| **AutoOsu** [B] | ISMIR LBD 2023 | osu! 动作生成 | 未核实 | 未核实 | 未核实 | 被 TCP 作为 SOTA 基线 | 未核实 |
| **GenerationMania** [B] | AIIDE 2019 | BeatMania 语义编舞 | 未核实 | 未核实 | 未核实 | 未核实 | 未核实 |
| **Mapperatorinator / osu!dreamer / BeatLearning / osumapper** [B] | 无正式论文 | osu! 谱面生成 | 未核实 | 未核实（社区仓库抓取被拒） | 未核实 | 未核实 | 未核实 |
| **Dancing Monkeys（规则法）/ DDG（难度渐变）** [C] | 未定位 | 见 [§2.4](#24-仅由二手转述的工作) | — | — | — | — | — |
| **ChartGenEval** [A] | arXiv 2026（预印本） | **评估框架**（非生成器） | 六问 / 七输出轴 | — | — | 见 [§7](#7-评估指标与人类评估协议) | 校准集 3,880 谱 / 813 曲组；开发 40 曲组 / 170 谱；held-out 80 曲组 / 333 谱 |
| **MIREX 2026 任务** [B] | MIREX 2026 | osu!taiko 谱面生成（hit objects 必需，inherited timing points 可选） | .osu 文件 | — | — | 三段式：算法 → 专家 → 社区 | 训练集限定 727 beatmapsets（featured artist 子集 174） |

**单位换算提醒**：TaikoNation 的 ±23 ms 宽容窗 = 1 帧（其 1 帧 ≈ 23 ms）；DDC 的音频帧步长 10 ms、上下文最多 15 帧（150 ms）。

### 2.2 逐篇可迁移点

#### DDC（ICML 2017，arXiv 1703.06891）[A]

1. **二分结构（placement / selection）是历史最长的分解**，DDCL、ITGPT 都继承它。→ 对我们：RFC-0029 把 note 作为带标记的点 (t, positionX, side, type) **不再分解**；这与 DDC 系相反。**这是一处需要明确取舍的地方**：文献侧证据是「分解可行且有效」，不是「分解必要」。
2. **热图 + 固定阈值 + 平滑的朴素解码**：DDC 对预测概率做 Hamming 窗卷积以抑制短距离双峰，再按难度取不同常数阈值。→ 这正是 RFC-0029 §3.4 要作为 **v0 消融项**的做法；文献显示它**必须配「每难度一个阈值」**，否则数量不可控。
3. **±20 ms 容差**的最早约定之一。
4. **数据规模极小（15–19 小时音频）**：GOCT 明确称自己的数据集「约为 Fraxtil 与 ITG 的 10 倍」。→ 我们的数据规模目标应参照 GOCT 量级，而非 DDC。

#### GenéLive!（AAAI 2023，arXiv 2202.12823）[A]

1. **低难度谱面是公认难点**：原文明确「DDC 在高难度上已达人类竞争水平，但低难度生成有改进空间」，且其改进「对较易难度尤其有效」。GOCT 独立报告「低难度谱面灾难性失败」。→ **对我们的硬约束**：难度条件化必须**按难度分档报告指标**，低难度不能作为「顺带能做」的假设。
2. **BCE + 高斯软标签（fuzzy label）宽度作为超参**：与我们「事件处必须高」的软目标构造对应，是除热图 MSE / focal 之外的第三条软标签路线。
3. **评估同时给按谱平均（F1-c）与 micro（F1-m）**，TP 判据 **±50 ms**。→ 直接采用双口径。
4. **工业部署证据**（成本减半）属作者自述，未提供指标化的用户研究设计 → 只能作为「可行性存在」的证据 [C 级用法]。

#### GOCT / Beat-Aligned（ISMIR LBD 2023，arXiv 2311.13687）[A]

1. **把「谱面生成」从逐帧二分类改成条件序列生成，理由是消除二分类不均衡**——原文：such approaches were observed to exhibit binary class imbalance ... We newly formulate chart generation as a conditional sequence generation task, thus removing the binary class imbalance. → **这是 RFC-0029 用泊松 NLL 解决同一问题的平行路线**，也是我们最重要的对照臂。
2. **beat-aligned 预处理被原文称为 integral for successful training**：hop = 1/48 beat；消融显示 unaligned 模型在 8 分音符上 70.0% vs aligned 87.9%，且未对齐模型的输出会**随时间漂移**。→ 对我们的约束：**时间网格的构造方式（是否按 beat 归一化）会实质影响训练成败**，且漂移失败模式与 postmortem 中的帧率错配属于同类「坐标系错误」。
3. **按时间组分解报告**（8th / 16th / 12th / 32nd / 24th）：→ Phigros 同样存在三连音 / 32 分，应照搬这一分解（12th/24th 即 triplet 组）。
4. **三次独立工作都选 48 细分/拍**：GOCT 的 hop = 1/48 beat 与 ITGPT 的 placement 输出 48 维/拍**独立地选了同一细分**。→ 【推断】这是时间量化分辨率的一个经验下界，可作为桶宽选择的对照锚点（**不是**可直接搬用的常数，见 RFC-0029 §7 硬约束 1）。
5. **规模阈值效应**：小数据集（Fraxtil/ITG）上 GOCT 输给 CNN 基线；在 10 倍数据上胜出。→ 【对我们的约束】任何「新范式胜过旧范式」的结论必须在**足够规模**上给出；小规模对比不具判别力。与 G1–G4 门禁的分工一致（门禁判「是否退化」，规模判「是否更好」）。
6. 官方站点与代码：stet-stet.github.io/goct、github.com/stet-stet/goct_ismir2023（B：MIREX wiki 列出）。

#### ITGPT（arXiv 2607.14148）[A]

1. **评估必须固定解码规则**：同一模型 F1@0.5 = 0.7801 vs 阈值最优 Max F1 = 0.8022；DDC 更极端（0.5006 → 0.7317）。→ **对我们的直接约束**：强度场与离散 token 两种范式的对比中，**不能一方调阈值另一方不调**（原文专门讨论了这个不对称：GOCT 是 token 分类器，no threshold to optimize）。报告必须并列「固定解码规则」与「各自最优解码」两栏。
2. **诊断网络正则化**：训练一个能从生成谱反推 BPM 与难度的辅助网络，其损失加权加进主损失。→ 与「难度条件必须可恢复」的诉求同构，是实现 G4 类契约断言之外的一种**训练期**保障手段。
3. **位置加权 BCE 的动机原文**：most positions within a beat will be uniformly empty, thus a minor reweighting scheme is appropriate to address class imbalance——权重：downbeat/offbeat/16th = 2，triplet/24th 与其余为小数（具体数值在 MathML 中被剥离，见 [§9](#9-存疑与未核实清单)）。→ 稀疏目标下「轻量重加权」的实证先例，直接支持 RFC-0029 §3.2 对朴素 BCE 的批评，同时说明**加权 BCE 不是无用的稻草人，而是文献主流做法**——我们的消融必须包含「加权 BCE」而不只是「朴素 BCE」。
4. **chart 平均 vs 预测平均**两套口径（cht 下标）。

#### DDCL（arXiv 2507.01644）[A]

1. **「不使用容差窗口」的评估主张**：原文称其架构使其能精确落在 downbeat/offbeat/16th，therefore we do not require the usage of a hamming or tolerance window。→ ⚠️ 这与 STRUM 对人类谱面时间戳分布的实测**直接冲突**。→ 【对我们的约束】必须**同时报告**「原始时间域 F1@容差」与「网格吸附后 F1」，否则两种口径下的结论不可比。这也正是 RFC-0029 §7 硬约束 5（评估在秒域）的文献依据。
2. 沿用 Fraxtil 数据集（95 谱）→ 该领域**数据集长期未更新**，跨论文数字可比的窗口很窄。

#### TaikoNation（FDG 2021，arXiv 2107.12506）[A]

1. **5 个指标的完整定义**（可直接采用）：DCRand（与随机噪声逐时刻比较）、DCHuman（逐时刻相似度，等价于把人类谱当 gold 的 accuracy）、OCHuman（±23 ms 宽容窗的 onset 比较）、Over. P-Space（8 帧滑窗的唯一有序组合数 / 可能空间）、HI P-Space（模型与人类图案空间的交集）。→ **图案空间覆盖率是「新颖性」指标**，可用于跨判定线图案；**DCRand 用随机噪声做对照**，与 G3「常数基线」门禁同构（原文明确：noise serves as a control for our chosen evaluation metrics）。
2. 数据只有 100+ 谱 → 该领域早期工作的规模下限。

#### TCP（AIIDE 2025）[C，仅摘要]

1. **「神经 onset 检测 + 符号化 beat snapping + 图案匹配」的混合路线**在 2025 年仍被作为正面贡献发表。→ 对我们的直接意义：RFC-0029 §3.4 的 v0 解码（峰值检测 + 量化）**在文献中有同类先例且未被淘汰**；但 TCP 把 snapping 当成**框架的一部分**而非权宜之计，提示我们「检测 + 吸附」若要长期保留，就应作为**一等组件**设计（且必须与 NLL 同网格），而非外挂。
2. 全文未取到 → 其指标定义与数据规模**未核实**（[§9](#9-存疑与未核实清单)）。

#### STRUM（arXiv 2605.12135）[A]

1. **最重要的一条事实**：在 29 首基准曲、39,136 个鼓 GT 事件上，**只有 89.0% 的人类谱面事件落在任何音频 onset 峰值的 ±100 ms 内**；其余 11% 是「无可听 onset 的编排事件（如视觉填充的长音）或时间戳偏离实际敲击超过容差」。原文结论：This sets a hard upper bound: a perfect transcriber that matches the audio cannot exceed 0.89 recall against community ground truth.
   → **对 RFC-0029 的硬约束**：
   - 我们的强度场目标**不能**被默认为「音频 onset 的狄拉克和」。人类谱面包含大量**非 onset 驱动**的编排决策（Phigros 只会更强：判定线运动、视觉演出、filler 音符）。
   - 任何以 MERT / onset 为输入、以人类谱为目标的模型，其 recall **天然存在与音频无关的上界**；报告时必须区分「未匹配到人类谱」与「未匹配到音频」。
   - 与 audio2chart 的「音频条件增益有限」相互印证。
2. **per-song 全局 offset 搜索**是标准做法（对每首歌取使鼓 F1 最大的偏移，同曲所有乐器共用）。→ 与 ChartGenEval 的全局相位估计是同一件事的两个实现；**我们必须报告「不做 / 做」全局 offset 搜索的两组数**。
3. **音频质量门槛（in-envelope 协议）**：先公布筛选判据（鼓干声 1 s 中位 RMS），再报数字。→ 对我们的数据流水线（Phira 社区谱质量差异大）直接可借用：**先声明数据入选判据，再报指标**。

#### audio2chart（arXiv 2511.03337）[A]

1. **无音频条件基线不可省**：原文明确 An unconditional baseline demonstrates strong predictive performance, while the addition of audio conditioning yields consistent improvements。→ 即**音频条件的增益是「一致但有限」的**。→ 必须报告「无音频 / 无判定线事件」的对照臂，否则无法证明 MERT 特征与新条件是有效贡献（对应 G2/G3 的精神）。

#### ChartGenEval（arXiv 2607.12857）[A]

见 [§7](#7-评估指标与人类评估协议)——它是本文件评估章节的主要骨架。

### 2.3 GOCT = Beat-Aligned 是同一篇工作

**结论：二者是同一篇工作的模型名与论文名。** 证据：

1. 作者官方演示站 stet-stet.github.io/goct 下挂的 PDF 标题即 BEAT-ALIGNED SPECTROGRAM-TO-SEQUENCE GENERATION OF RHYTHM-GAME CHARTS（A：作者官方站点）。[A]
2. ITGPT 在 Related Work 中把「节奏游戏 transformer [20]」称为 **GOCT**，并明确 we will ... compare ITGPT against ... GOCT, using their osu! four panel fine-tuned version，且 we test using the checkpoints the authors provide——这与 2311.13687 的 osu!mania 数据集与公开权重一致。[A]
3. MIREX 2026 任务页把该工作列为 Yi, Jayeon, Sungho Lee, and Kyogu Lee. Beat-Aligned Spectrogram-to-Sequence Generation of Rhythm-Game Charts. ISMIR LBD 2023，并给出仓库 github.com/stet-stet/goct_ismir2023。[B]

**影响**：RFC-0029「关联」小节与 CLAUDE.md 把 GOCT 与 Beat-Aligned 并列为两条参考，应合并为一条；**GOCT 是模型名（缩写展开未核实）**，论文名是 Beat-Aligned。

### 2.4 仅由二手转述的工作

| 工作 | 转述内容 | 来源级别 |
| :-- | :-- | :-- |
| **DDG** | 「扩展 DDC 以生成更低难度：通过把训练集限制到特定粗难度来制造渐变」；ITGPT 认为「从该论文无法判断哪个模型应作为基线」 | C（ITGPT 相关工作，原文未定位） |
| **Dancing Monkeys** | 「纯规则方法」 | C（ITGPT 相关工作，原文未定位） |
| **GenerationMania** | BeatMania 方向的语义编舞（venue: AIIDE 2019；arXiv 1806.11170，元数据已核实） | B（元数据）/ 正文未取 |
| **AutoOsu** | ISMIR LBD 2023，被 TCP 作为 SOTA 对比基线 | B（MIREX 列表）/ 正文未取 |

---

## 3. Phigros / RPE 专属生成工作检索结果

### 3.1 结论

**未检索到任何 Phigros（或 RPE / Phira 格式）专属的谱面生成模型工作。** 本文的检索为**阴性结果**，受 [§1.4](#14-检索式与结果可复现) 的通道限制，按 Q3 的警告口径**不能表述为「该方向不存在研究」**，只能表述为「在本文件使用的通道与检索式下未命中」。

### 3.2 检索方式与命中情况

| 检索 | 结果 |
| :-- | :-- |
| arXiv API all:phigros | 0 条 |
| OpenAlex fulltext.search:phigros | 7 条，**全部与 Phigros 谱面生成无关**（Beat Saber 用户普查、游戏音乐交互、在线游戏隐私、古籍植物志、电子音乐评论、视频 MLLM、锥虫染色质） |
| web_search 中文：phigros 谱面自动生成 论文 / phi 谱面自动生成 判定线 神经网络 / Phigros 谱面 生成 模型 RPE json | 命中的是**编辑器**（Ex-Phiedit、KipPhiApparatusLegacy）、**格式文档**（Phira Docs、pgrfm wiki）、**资源下载器**（phigros-chart-downloader，逐曲导出可玩谱面） |
| web_search 英文：Phigros chart generation machine learning / "Phigros" chart generation dataset | 同上，无生成模型 |
| web_search 日文：phira chart generation model | 命中 Phira 相关仓库，无生成模型 |
| 交叉核对：本领域相关工作章节（ITGPT §2、GOCT、ChartGenEval §2、TaikoNation §2、MIREX 2026 任务页论文清单） | **均未提及 Phigros / RPE / Phira** |

### 3.3 可用的邻近资产（非生成模型）

- github.com/swordalt/phigros-chart-downloader：导出/下载 Phigros 曲目资源与可玩谱面。→ 对**数据获取**可能有价值（是否覆盖 Phira 自制谱**未核实**）。
- Phira 官方文档与社区 wiki 已由 [phigros-format.md](phigros-format.md) 完整整理。→ 格式侧我们**不缺**文献，缺的是生成侧的先例。

### 3.4 对决策的含义

【推断，依据 §3.2】Phigros 谱面生成在公开学术文献中**没有可直接对标的先例**。因此：

1. 我们的**指标与协议必须自建**，但可整体移植太鼓 / osu! 谱面生成的协议骨架（[§7](#7-评估指标与人类评估协议)）。
2. 我们的**基线不能引用「Phigros 上的前人数字」**，只能引用**同任务结构**（下落式、多难度、含舞台动画对象）下的数字。
3. 「多判定线 + 判定线事件轨」这一维**没有文献对照** → 见 [§8 消融 7/8](#8-建议的基线与消融设计)。

---

## 4. 点过程与泊松 NLL 基础

### 4.1 非齐次泊松过程的对数似然（我们的目标函数）

在观测窗 [0, T)、观测到事件时刻 {t_1..t_N} 时，非齐次泊松过程对数似然的标准形式为：

    log L = Σ_k log λ(t_k)  −  ∫_0^T λ(t) dt

带标记（marked）的情形：标记的条件分布与 ground intensity 分开，即把上式写成 Σ_k log λ(t_k) + Σ_k log f(mark_k | t_k)；我们的标记 = (line_id, positionX, side, type, ...)。

**依据**：点过程似然的标准形式；本文以 Omi et al. 2019 摘要原文（the log-likelihood function, which contains the integral of the intensity function）[A] 与 EBC-ZIP 的泊松/零膨胀泊松计数似然实际形式 [A] 交叉印证。**未逐字核实**：某一本教材/综述对该式的原始陈述（[§9](#9-存疑与未核实清单) 第 14 条）。

**对 RFC-0029 §3.2 的对应**：RFC 写的 L = -Σlog λ(e_k) + ∫∫∫ λ dt dx ds 与该式一致；空间/侧别/类型维度进入标记或进入强度场的定义域都可，但**必须显式声明是哪种**（二者的归一化方式不同）。

### 4.2 基础与谱系文献

| 文献 | 贡献 | 对我们 |
| :-- | :-- | :-- |
| **Hawkes 1971**，Biometrika 58(1):83，DOI 10.1093/biomet/58.1.83 [B 元数据] | 自激/互激点过程：λ(t) = μ + Σ_{t_i<t} φ(t − t_i) | 谱面事件**成簇**（连打、楼梯），泊松的「独立增量」假设不成立；Hawkes 是「事件引发更多事件」的标准建模。→ 【对我们的约束】若 v1 用纯泊松，必须在文档中承认「自激未被建模」，把它作为已知容量限制而非事实（见 §4.4 陷阱 3） |
| **RMTPP**（Du et al.），KDD 2016，DOI 10.1145/2939672.2939875 [B 元数据] | 用 RNN 把事件历史嵌入向量，直接参数化条件强度 | 神经点过程的开端；「历史 → 强度」的接口与我们的条件化同构 |
| **Neural Hawkes**（Mei & Eisner），NeurIPS 2017，arXiv 1612.09328 [B 元数据] | 连续时间 LSTM；隐藏状态连续演化，强度由状态决定 | 若要在**任意时间分辨率**上取 λ，连续时间状态演化是标准解 |
| **Omi et al.**，NeurIPS 2019，arXiv 1905.09690 [A] | **参数化累积强度 Λ(t)（前馈网络）再求导得到 λ** | ⭐ **对我们硬约束 #4 的直接解法**：若网络输出 Λ 而非 λ，则 ∫λ 由 Λ(t_end) − Λ(t_start) **精确**得到，无需数值积分，也就天然满足「同网格」要求。**建议在 field 模块设计评审中把这条作为备选参数化** |
| **Intensity-free**（Omi et al.），ICLR 2020，arXiv 1909.12127 [B 元数据] | 不建模强度，直接建模事件间隔分布 | 备选路线；但难以表达 positionX 的联合结构 |
| **NSTPP**（Chen et al.），arXiv 2011.04583 [B 元数据] | 时空点过程 λ(t, x, y) | ⭐ 与我们的 λ(t, positionX, side) **同构**（时间 x 连续空间）；其空间维度处理方式值得对照阅读（正文未取，见 §9） |
| **Shchur et al.**，IJCAI 2021 Review，arXiv 2104.03528 [B 元数据] | 神经点过程综述 | 综述级引用；正文未取（§9） |
| **Integration-free Training**，NeurIPS 2023，arXiv 2310.05485 [B 元数据] | 免积分的点过程训练 | 与 Omi 2019 同向的第二条证据：**「积分项难算」是该领域公认的工程问题** |
| **Ogata 1981**，IEEE T-IT，DOI 10.1109/tit.1981.1056305 [B 元数据] | thinning（舍选）采样算法 | ⭐ **解码端的正确工具**：从 λ 采样得到事件（而非阈值 + 峰值）。RFC-0029 §3.4 的 v0 用 find_peaks；thinning 是「与训练目标同构」的对照解码器，应进消融 |
| **Peeling & Li**，JASA 2007，DOI 10.1121/1.2716156 [B 元数据；作者名据 Semantic Scholar，未逐字核实] | Poisson point process modeling for polyphonic music transcription | ⭐ **音乐/音频领域的直接先例**：把音符起始当作泊松点过程建模。→ 我们的做法在音乐领域**不是首创**，引用它能显著降低「新范式无先例」的论证成本 |
| **Inhomogeneous Weibull-Hawkes**，JABES 2024（St Andrews 仓库）[C：仅检索片段，正文未取] | 用非齐次 Weibull-Hawkes 建模欠离散声学线索 | 说明声学事件的点过程建模仍在活跃发展；**仅作旁证，不引用具体结论** |

### 4.3 空间计数 / 密度场的先例（与 §6 强耦合）

| 文献 | 贡献 | 对我们 |
| :-- | :-- | :-- |
| **Chan & Vasconcelos**，ICCV 2009，DOI 10.1109/iccv.2009.5459191 [B 元数据] | 用 **Bayesian Poisson regression** 在图像分块上做人群计数 | 与我们的场 + 积分项**同构的早期先例**（把计数当泊松观测） |
| **Lempitsky & Zisserman**，NIPS 2010，Learning To Count Objects in Images [B 元数据] | 密度图回归 + 计数约束（MESA 距离） | 密度图范式的奠基；**「目标里显式包含总量约束」**这一思想与我们的 ∫λ 同源 |
| **EBC-ZIP**，arXiv 2506.19955 [A] | **零膨胀泊松（ZIP）回归**替代 MSE，明确针对「极端稀疏 + 零膨胀」 | ⭐ 见 [§6](#6-稀疏目标的损失选择为什么我们可能重蹈塌到全-0) —— 它是我们「不要用 MSE」的最强引文 |

### 4.4 已知陷阱（逐条给出处或标注）

1. **积分项必须显式选方法**：Omi et al. 2019 的立论即「既有方法要数值近似，我们用 Λ 的导数精确计算」[A]；Integration-free 一文把「积分难算」当成独立问题 [B]。→ 我们的硬约束 #4 有直接文献支撑。
2. **Monte-Carlo 估计 ∫λ 会引入偏差**：E[log X] ≤ log E[X]（Jensen），用 log(MC 估计) 替换 log(∫λ) 不是无偏的。**【推断 / 数学事实】**（本文自行推导，未找到直接文献陈述，见 §9 第 15 条）。→ 结论：优先**确定性数值积分**（固定网格 / 求积）或 **Λ 参数化**（陷阱 1），不要用 MC。
3. **泊松假设 = 事件相互独立、无自激、无排斥**：Hawkes 1971 存在的理由就是打破「无自激」[B]。而谱面同时存在**自激（连打成簇）**与**排斥（不应期 / 最小间隔）**。→ 【对我们的约束】纯泊松 NLL 是**近似**，必须在文档中承认；解码端的硬约束后处理（同刻按键上限、hold 区间合法性）在文献里对应「把物理约束放在解码/后处理」的常规做法。
4. **离散化后「每桶事件数」的语义必须选定**：把连续场离散成网格后，∫λ ≈ Σ λ_b Δ 与「事件处 log λ」之间的**计数语义**要一致——若一个桶内出现 2 个事件而模型只产生 1 个 log λ 项，似然就不是泊松似然。正确形式是**桶内计数作为泊松观测**（这正是 EBC-ZIP 的 blockwise 计数形式 [A]）。**【推断，依据：泊松计数似然与逐点强度的等价性 + EBC-ZIP 的分块计数形式】**。→ 在我们的网格分辨率与 Phigros 密度下（同刻多键 / 多判定线同刻），**这不是理论问题而是必然发生**。
5. **数值稳定**：强度用 softplus / 指数参数化保证 λ ≥ 0；事件项与积分项**必须同量纲**（都是「计数」）。**【推断 / 数学】**。
6. **NLL 不能当唯一质量分**：ChartGenEval 实测 common-pattern rewriting 使 LM perplexity **下降 37%**（9.44 → 5.98），即**似然类指标可以朝错误方向移动** [A]。→ 与 RFC-0029 §5 要求报告 NLL 一致，但**必须同时有非似然指标**。

## 5. 掩码生成建模

### 5.1 三条路线与它们的真实关系

| 路线 | 训练目标 | 解码 | 代表 |
| :-- | :-- | :-- | :-- |
| **掩码补全 / 并行解码** | 在**被 mask 的位置**上算交叉熵 | 迭代：预测全部 → 保留高置信 → 重新 mask 低置信 → 重复 | Mask-Predict [A]、MaskGIT [A] |
| **absorbing-state 离散扩散** | 同上，但按时间步**加权** | 反向扩散采样（一般化为多步） | D3PM [B]、MDLM（Shi et al. 2024）[B] |
| **逐 token 自回归（AR）** | 全位置交叉熵 | 顺序 | GOCT / ITGPT [A] |

**关键等价关系（可直接引用）**：absorbing-state（吸收态 = [MASK]）离散扩散的 NELBO，在时间步加权下**就是掩码位置上的加权交叉熵**：

    L = Σ_i  (α_{s(i)} − α_{t(i)}) / (1 − α_{t(i)}) · 1{x_{t(i)} = m} · ( − log f_θ(x_{t(i)})_{x0} )

即「只在被 mask 的位置计算损失、并按时间步加权」（来源：arXiv 2510.03289 式(10)；该文同时给出 ELBO/NELBO 推导，并声明连续时间极限来自 Shi et al. 2024）[A]。

> ⭐ **对 RFC-0029 §3.3 的直接含义**：「MaskGIT 式迭代并行解码」与「absorbing-state 离散扩散」在**训练目标上不是两个不同的东西**，差异在于 (a) 时间步加权的具体形式、(b) 采样/解码器、(c) 是否使用连续噪声调度。因此 RFC-0029 把它们并列成两个对照臂时，**必须指明对照的是哪一层的差异**，否则消融不可解释。**【推断，依据上式】**。

### 5.2 MaskGIT（CVPR 2022，arXiv 2202.04200）[A]

- 训练：掩码视觉 token 建模（MVTM），a similar proxy task to the mask prediction in BERT；**条件依赖有两个方向**（双向注意力），这是相对 AR 的本质差别。
- 解码：迭代（predict → sample → mask schedule → mask），每步**只保留最置信的部分**，其余重新掩码；置信度 = 被采样 token 的预测概率（未掩码位置置信度设为 1.0）；掩码数由调度函数 γ(t/T) 决定。
- **关键经验事实**：理论上可以一次推断全部 token，但作者明确 we find this challenging due to inconsistency with the training task；实际用 **8 步**完成 256 token（相对 AR 的一个数量级加速）。
- → **对我们的约束**：掩码补全**不能一步到位**（会与训练分布不一致）；必须实现「迭代 + 置信度保留 + 重新掩码」，并**声明步数**（步数直接影响质量-速度权衡，是必须报告的消融维度）。对连续强度场，「置信度」需要重新定义（如 λ 的峰值显著性）——**文献无对应做法，属我们的新增设计**。

### 5.3 Mask-Predict（EMNLP 2019，arXiv 1904.09324）[A]

- 场景：**条件生成**（机器翻译），用条件掩码语言模型做非自回归解码。
- 结果：迭代「预测全部 → 掩掉最不确定的一部分 → 重预测」使其比当时的非自回归 / 并行解码模型平均 **+4 BLEU**，且 within about 1 BLEU point of a typical left-to-right transformer。
- → **对我们的约束**：我们的任务**是条件生成**（音频 + 难度 + 判定线事件轨为条件），比 MaskGIT（图像、类别条件）**更接近** Mask-Predict 的设定。因此「并行解码接近但略逊于 AR」应作为**预期**而非失败；同理，**AR 上界臂不可省**。

### 5.4 反方证据：Why Mask Diffusion does not Work（arXiv 2510.03289）[A，预印本]

该文（2025-10 预印本）论证掩码扩散在**并行生成与双向注意力**上有内在困难，三条理由（原文）：

1. 模型输出的是每个 [MASK] 位置的**条件边缘分布**，**不是**所有 [MASK] 位置的联合分布 → 并行采样**无理论保证**。
2. 远离未掩码位置的 [MASK] 位置分布**平滑、众数同质** → many of the probabilities produced by the model are correct, they provide little useful information for sampling。
3. **最可靠稳定的生成策略很可能仍是自回归**，因此难以有效利用双向注意力。

→ **对我们的约束**：

- RFC-0029 §3.3 把「掩码补全」列为**主选**架构。该文不否定掩码补全，但指出**「并行」与「一步到位」的收益有限**，且**必须有 AR 对照**。
- 该文同时给出一般化结论：掩码扩散的采样器设计不是免费的（与 5.2 的「一次全解不可行」互相印证）。
- **注意分级**：这是**新近预印本**，且结论针对语言模型；本文把它作为**风险提示**而非定论。

### 5.5 与「掩码补全」三条补充要求的对照（RFC-0029 §3.3）

| RFC 要求 | 文献支持情况 |
| :-- | :-- |
| ① 必须显式提供 mask 通道 | ✅ **有直接支持**：掩码扩散 / 掩码补全的损失只在被 mask 的位置计算（2510.03289 式(10) 的指示函数 1{x_t = m}）[A]。对**连续场**而言没有天然的 [MASK] token，mask 通道是我们必须自建的对应物（无直接文献，属新增设计） |
| ② 按事件遮盖而非按帧遮盖 | ⚠️ **未找到直接文献**。邻近证据：GOCT 使用**块级**条件（前两拍的 token → 后两拍的 token）而非逐 token [A]；Mask-Predict 的「重新掩码低置信位置」是**内容自适应**的掩码 [A]。→ 标注为**【推断】**（RFC 的论证「邻帧几乎必然含相同事件 → 可抄邻居」在数学上成立，但本文未找到实证对比） |
| ③ 保留离散扩散对照臂 | ✅ 有支持：见 5.1 的等价关系与 5.4 的并行性风险 |

---

## 6. 稀疏目标的损失选择（为什么我们可能重蹈「塌到全 0」）

### 6.1 最强的一条外部证据：EBC-ZIP（arXiv 2506.19955）[A]

原文（引言）：

> In real-world crowd scenes, the vast majority of spatial regions contain no people, resulting most values of the ground-truth density maps being zero. **If this severe imbalance is not properly addressed, it leads to two significant issues. First, models trained with standard regression losses (e.g., MSE or MAE) become biased toward predicting low or zero values, reducing their sensitivity to sparse yet informative regions. Second, the dominance of zero-valued regions weakens the supervisory signal during training, impeding the model's ability to learn fine-grained density variations.**

以及摘要：

> the vast majority of spatial regions (**often over 95%**) contain no people ... most loss functions used in density estimation are majorly based on MSE and **implicitly assume Gaussian distributions, which are ill-suited for modeling discrete, non-negative count data**.

→ ⭐ **这是「塌到全 0」在空间稀疏目标上的权威陈述**，与 RFC-0029 §3.2 表格中「朴素 BCE → 塌到全 0 解」的判断**同向**，且给出了 MSE 类回归损失的**两个具体失效机制**（偏向低/零、监督信号被零区主导）。我们的目标稀疏度比其「95% 为零」**更极端**（音符占全部 (t, x, side) 单元的比例远低于 5%）——**【推断，依据 Phigros 密度量级；具体比例必须在数据集统计后填入，不得凭猜测写数字】**。

EBC-ZIP 的解法：用**零膨胀泊松（ZIP）的负对数似然**替代回归损失，enabling better handling of zero-heavy distributions while preserving count accuracy。→ 我们的「泊松 NLL + 积分项」是同一族做法（泊松 vs 零膨胀泊松；ZIP 多一个零膨胀参数 π 来表达「结构性的零」）。
**未核实**：ZIP 相对 MSE 的具体数值增益（未取到结果表，见 §9）。

### 6.2 候选损失的对照

| 方案 | 文献中的定义 / 做法 | 极端稀疏下的已知行为 | 与「λ 场 + 积分项」的兼容性 |
| :-- | :-- | :-- | :-- |
| **MSE（热图回归）** | 高斯热图目标 + L2 | ⚠️ EBC-ZIP 明文：**偏向预测低值/零**、零区主导监督信号 [A] | ❌ 与「给出一致计数」冲突；无概率语义 |
| **加权 BCE** | ITGPT：位置加权（downbeat/offbeat/16th = 2）[A]；GenéLive!：BCE + 权重因子 + 高斯软标签 [A]；DDC：阈值按难度分档 [A] | 文献主流可用方案，但**必须配阈值/权重超参**；DDC 的 F1 从 0.5006（0.5 阈值）到 0.7317（最优阈值）显示**超参敏感度极高** [A] | ⚠️ 逐 cell 独立 Bernoulli，多个高斯叠加后全图积分远大于 1（RFC-0029 §3.2 已指出）；**不能与泊松积分项混用** |
| **Focal / penalty-reduced focal（热图版）** | CornerNet：正样本 (1−p)^α log p，负样本 (1−y)^β p^α log(1−p)，其中 y 为**未归一化** 2D 高斯、σ = 半径的 1/3、重叠取 element-wise max [A]；CenterNet 同型（α, β 为超参；**具体取值未核实**）[A]；Focal Loss 原始动机 = 前景/背景极端不均衡（1:1000）[B 元数据] | ✅ 为「海量 easy negative」而生，是该问题最成熟的工程解 | ⚠️ **不兼容**：其目标 y 是**未归一化高斯**（不是概率、不满足 ∫y = 事件数），而泊松 NLL 的积分项要求 λ 是**强度**（计数 / 单位面积）。**【推断，依据 CornerNet 对 y 的定义】** |
| **泊松 NLL（我们的主线）** | −Σ log λ(e_k) + ∫λ；EBC-ZIP 用 ZIP-NLL 处理零膨胀 [A] | 积分项惩罚「到处乱亮」；事件项在 λ→0 时**发散** | ✅ 自洽；需处理 §4.4 陷阱 4（桶内计数语义） |

### 6.3 一个可自证的、与 G3 门禁直接对应的判据

对常数强度场 λ ≡ c，泊松 NLL 为 −N·log c + c·|Ω|，其最小值在 c* = N / |Ω|（人类平均密度）处取得。

**关键推论（【推断 / 数学事实】，本文自证）**：

- 常数解**不是** λ ≡ 0：当 N > 0 时 −N·log c → +∞，所以「全 0」在该目标下**不是有限损失解**。
- 相对地，等价的**逐 cell BCE** 在 p ≡ 0 处损失是**有限且很小**的（约为正样本比例量级），因此「全 0」是一个**低损失强吸引子**。
- → **可操作结论**：G3「常数基线」应实现为 λ ≡ N/|Ω|（而非 λ ≡ 0），并同时记录 λ ≡ 0 时的损失（应为 +∞ 或数值溢出）作为**契约断言**。这条把 RFC-0029 的定性论断变成了可跑的测试。

> ⚠️ 诚实标注：**本文未检索到「泊松 NLL vs BCE 在塌到全 0 上的直接 A/B 对照」的文献**。上述论证是数学推导 + 邻近领域间接证据（6.1 的 MSE 失效、focal loss 的动机）。见 §9。

### 6.4 其它可参考的失衡对策

- **Gradient Harmonized Detector**（AAAI 2019，arXiv 1811.05181）[B 元数据]：从**梯度分布**而非样本数出发解释不均衡，可作为 focal 之外的第二备选。
- **DARK**（CVPR 2020，arXiv 1910.06278）[B 元数据]：热图**解码**阶段的系统性偏差（把 argmax 当坐标会有偏），提示「解码器」是独立误差来源 → 与 RFC-0029 §3.4「把难题从生成挪到检测」的批评同向。

---

## 7. 评估指标与人类评估协议

### 7.1 时间容差：跨论文差 5 倍，必须显式声明

| 工作 | 容差 | 说明 |
| :-- | :-- | :-- |
| DDC [A] | **±20 ms** | TP 判据 |
| TaikoNation [A] | **±23 ms** | = 1 帧；OCHuman 向前后各查 1 帧 |
| GOCT [A] | **±30 ms** | 时间 token 判对 |
| GenéLive! [A] | **±50 ms** | TP 判据 |
| STRUM [A] | **±100 ms** | 贪心匹配 |
| ChartGenEval [A] | **6 ms**（局部对齐阈值）+ 12/18 ms 诊断档 | 依据「短间隔的时间分辨力」文献；另有**全谱相位偏移**估计（0.5 ms 步长搜索；bar/beat/eighth 网格权重 0.2/0.3/0.5） |
| DDCL [A] | **不使用容差窗** | 主张能精确落在网格上 |

→ **对我们的直接约束**：RFC-0029 §5 写「F1 @ ±20ms」。文献侧 20 ms 是**最严的一档**（DDC 2017 的标准）。建议：**主指标 @ ±20 ms（与 DDC 可比）+ 副指标 @ ±50 ms（与 GenéLive! 可比）双报告**，并**单独报告全谱相位偏移**（STRUM / ChartGenEval 的做法）。**【推断：双报告是本次调研的直接推论】**

### 7.2 事件级指标（可直接采用的定义）

1. **贪心一对一匹配**：把预测事件按时间贪心匹配到 GT，每个 GT 至多匹配一个预测（STRUM 原文）[A]。
2. **F1 / Precision / Recall @ 容差**；**未匹配的生成事件保留在分母**（ChartGenEval 的 timing clean rate 明确 Unmatched generated notes remain in the denominator）[A]。→ 否则「删掉难判的音符」能刷高 precision。
3. **双口径平均**：按谱 / 按事件（micro）都报。
   - GenéLive!：F1-c（按谱平均，同曲不同难度视为不同谱）与 F1-m（micro）[A]。
   - ITGPT：cht 下标 = 按谱平均 [A]。
4. **阈值 / 采样自由度必须固定并声明**：ITGPT 同时给「阈值 0.5」与「每谱最优阈值」两栏 [A]。→ 我们应给「固定解码规则」与「每谱最优」两栏，且**两个范式用同一套报告格式**。
5. **按时间分组分解**（GOCT：8th / 16th / 12th / 32nd / 24th 各组 micro-F1；原文给出 8th 87.9% vs 未对齐 70.0%）[A]。→ Phigros 应至少分 1/4、1/8、1/12（三连）、1/16、1/24、1/32。
6. **按难度分解**（GenéLive! 与 ITGPT 都发现低难度差异最大）[A]。
7. **类型 / 动作准确率**：GOCT 的 action token micro-F1 [A]；ITGPT 的 256 类 selection 准确率 [A]。→ 我们的 type（tap/drag/hold/flick）与 side 准确率是同类量，可直接采用。

### 7.3 结构级 / 分布级指标

**来自 ChartGenEval 的七条核心输出（有 held-out corruption 证据的）**[A]：

| 输出轴 | 定义要点 |
| :-- | :-- |
| timing clean rate | 落在作者网格 6 ms 内的音符比例（未匹配者留在分母） |
| timing-error p99 | 绝对误差的 99 分位（暴露稀有的大偏差，均值会掩盖） |
| grid-phase offset | 全谱最佳对齐偏移（三档网格加权） |
| transition familiarity | 1 − (同课程训练谱中未出现过的 interval-type trigram 比率)（单侧分数） |
| 4-gram typicality | 重复 4-gram 与唯一 4-gram 的同课程典型性（**三个不同 corruption 共用这一个轴**） |
| note-rate typicality | 与同难度人类谱的**中段 80% 带**比较 |
| density-spike limit | −max(0, 谱的局部密度跳变 95 分位 − 同课程上限)（单侧约束） |

**带分数（band score）定义**（ChartGenEval 式(1)）[A]：取同课程训练谱的 10/90 分位为 b_lo 与 b_hi，带内为 1，带外**对称衰减**。→ 这是把「无方向性属性」（太稀疏 / 太密都不好）变成分数的标准做法，**可直接用于我们的密度与重复度**。

**明确被 ChartGenEval 标为「探索性」的读数**[A]（**不得**当已验证指标使用）：

- **response to music**（小节尺度能量响应与 onset 支持）——正是 RFC-0029 §5 的「密度曲线与音乐能量曲线相关性」！原文标注：Exploratory development analysis。
- human-chart distance、long-form readings。

→ ⭐ **对 RFC-0029 的直接约束**：§5 把「结构级：密度曲线与音乐能量曲线相关性」与其它指标并列。文献侧这条**尚无 held-out 证据**，我们若要把它作为报告指标，必须在文档中标注其为**探索性**，且**不能**用它作为模型选择的主判据。

**其它可采用的指标**：

- **TaikoNation 的图案空间指标** [A]：Over. P-Space（8 帧滑窗唯一有序组合数 / 全部可能组合数）、HI P-Space（模型与人类图案空间的交集）。→ 可用于「跨判定线图案覆盖率 / 新颖性」。
- **DCRand** [A]：与随机噪声逐时刻比较。→ **与 G3 门禁同构的外部先例**（原文明确 noise 的作用是 a control for our chosen evaluation metrics）。
- **per-song 全局 offset 搜索** [A]（STRUM）与 **grid-phase offset** [A]（ChartGenEval）→ 强制我们区分「局部时间准」与「整谱对齐」。

### 7.4 概率校准

- RFC-0029 §5 要求报告验证集 NLL —— 与 §4 的目标函数自洽；EBC-ZIP 也以 NLL（ZIP）作为训练目标而非附加指标 [A]。
- ⚠️ **必须与 ChartGenEval 的警告同读**：LM perplexity 在 common-pattern rewriting 腐败下**下降 37%**（9.44 → 5.98），即「似然改善 ≠ 质量改善」[A]。→ NLL 是**校准**指标，不是**质量**指标。

### 7.5 可玩性代理指标

- **MIREX 2026 的算法评估阶段** [B]：① 用 **osu! star rating 计算器**计算生成谱的星级，以判断**难度条件是否生效**；② 用 **AiMod / MapsetVerifier** 自动检测不可玩摆放。通过后才进入人评。
- → **对我们的对应物**：② 对应我们的 validate() 合法性校验（RFC-0029 §5 的「物理约束合法率」）；① 对应「难度条件是否生效」。**注意**：Phigros 是否存在公开的等效难度计算器——**未查证**（§9）。可替代的已知做法是 **ITGPT 的诊断网络**（用可预测性检验难度条件是否被注入）[A]。
- ChartGenEval 的 density-spike limit 也是一种可玩性代理（局部过载）[A]。

### 7.6 人类评估协议（可直接采用）

**（a）MIREX 2026 三段式** [B]——本文件推荐直接移植的骨架：

1. **算法评估**：人工挑 3 首歌；先用 star rating 计算器检查难度条件；再用 AiMod / MapsetVerifier 检测不可玩摆放；通过者进入下一阶段。
2. **专家评估**：约 **5 名 osu!taiko mapper** 组成 jury，参考 **Monthly Beatmapping Contest** 的维度打分：**musical representation / creativity / gameplay**，外加一个 **humanity score**（生成谱有多像人写的）；据此选出进入玩家评估的决赛作品。
3. **社区评估**：选一首 showcase 曲；**玩家按自身水平选择难度**，把各模型生成的谱与**人类谱混在一起试玩**，投票维度同样是 musical representation / creativity / gameplay，并被要求**指出哪一张是人类写的**（盲测识别）。

**（b）生成音乐通用协议**（Survey on the Evaluation of Generative Models in Music, ACM Computing Surveys 2025 / arXiv 2506.05104）[A]：

- 主观测试最常用 **Likert 型量表**；常评维度：overall quality / preference or enjoyment / stylistic appropriateness / complexity / coherence / aesthetic response (interestingness) / musicality。
- **MUSHRA**：带 anchor 信号，可同时呈现多个受损刺激，**比 MOS 需要更少被试即可达到统计显著**。
- **MOS**：源自语音（ITU-T），在生成音频中广泛使用。
- ⚠️ 原文强调：个体差异（尤其**音乐专业训练**）影响审美判断，研究者必须**显式操作化定义**要测的构念。
- ⚠️ 原文亦指出客观指标与主观判断的关系存在争论（既有「主观不可信」的主张，也有「客观指标不无意义」的论证）。

**（c）⚠️ 关键方法论：corruption 测试（ChartGenEval）**[A]

对**每一个**候选指标：**注入已知失败（dose-controlled corruption）**，要求 (1) 目标输出随失败强度呈**负的 dose-rank 关联**；(2) 最强剂量与匹配对照有显著差；(3) 所有**预先声明的不变性控制**通过；用 song-cluster bootstrap 区间并做多重比较校正。
该框架还给出两条**反面教训**（对我们是直接的警告）：

- **perplexity 会被「常见图案重写」改善**（下降 37%），方向错误；
- **self-similarity 会被「循环塌缩」改善**（上升 62%），方向错误。

→ **对我们的直接约束（强烈建议采纳）**：确定指标集后先做一轮 corruption 测试（如：注入全谱偏移 / 类型打乱 / 循环塌缩 / 密度缩放 / 局部爆发），只有通过「响应 + 不变性」双条件的指标才进入**主报告**。这与 G1-G4 门禁是同一思想在**评估侧**的对应物。

**（d）⚠️ 人评的已知陷阱**：ChartGenEval 明确指出其 human bands 只来自**单一太鼓语料**，且 corruption 测试确立的是对**定义好的失败**的响应，不是对谱面质量的普遍排序 [A]。→ 我们的 Phigros 人评不能引用其 band 数值，只能引用其**方法论**。

---

## 8. 建议的基线与消融设计

### 8.1 基线选择：GOCT 的配置最合理（理由 + 不可照搬之处）

**结论：以 GOCT（arXiv 2311.13687）的配置作为最主要的可复现对照基线**，理由：

1. **它正面对着我们同样的问题**：原文把二分类不均衡作为重构任务的**第一动机**，与我们用泊松 NLL 的动机**完全同源**，因此它的数字是**同一问题下的竞争解**，而不是无关基线的数字。[A]
2. **规模论证充分**：10 倍于 DDC 的数据、明确的小/大数据集对照，且给出「小数据上输给 CNN、大数据上胜出」的诚实结论 → 我们的对比可以复用其规模阈值意识。[A]
3. **有公开代码/权重**（github.com/stet-stet/goct_ismir2023，B：MIREX wiki 列表），且 **ITGPT 用它做过第三方复现并给出对照数字**（Max F1：GOCT 0.7754 vs ITGPT 0.8022）→ 若我们复现它，有第三方复核记录可参照。[A]
4. **预处理与我们的网格同构**：hop = 1/48 beat，与 ITGPT 的 48 位置/拍一致；beat-aligned 消融给出了**量化收益**（8th：87.9% vs 70.0%）。[A]

**不可照搬之处（必须重写）**：

- GOCT 是 **4 lane 离散**；我们是 **(positionX 连续, side, type) x 多判定线**。其 80 个 action token 的构造无法直接覆盖 positionX。
- GOCT **不含舞台对象**；我们的判定线事件轨是一等条件（RFC-0029 §2.2）。
- GOCT 的输出 token 是**事件序列**；我们的主选是**强度场**。→ 因此 GOCT 只能作为**离散 token 对照臂（B3）**的骨架，不能作为主线基线。

### 8.2 建议的基线 / 对照臂清单

| 臂 | 内容 | 文献依据 | 作用 |
| :-- | :-- | :-- | :-- |
| **B0 常数 / 先验基线** | λ ≡ N/|Ω|；以及「按人类平均密度采样 + 位置均匀」 | 本文 §6.3【推断】；DCRand [A] | G3 门禁；给出「什么都不学」的下界。**注意**：常数应取 N/|Ω| 而非 0 |
| **B1 热图回归 + 加权 BCE / focal + 峰值解码** | 未归一化高斯目标 + penalty-reduced focal（CornerNet/CenterNet 配置）+ Hamming 平滑 + 每难度阈值（DDC 配置） | CornerNet / CenterNet [A]、DDC [A]、ITGPT [A] | **RFC-0029 §3.2 要否定的方案必须作为消融臂**，否则无法证明泊松 NLL 的收益；同时给出「文献主流做法」的真实强度（不是稻草人） |
| **B2 主线：泊松 NLL 强度场 + 掩码补全** | λ_θ(t, x, s, c 以 audio、line_tracks、difficulty 为条件)；Λ 参数化或同网格数值积分；mask 通道 + 迭代解码 | Omi 2019 [A]、EBC-ZIP [A]、MaskGIT [A] | 主路径 |
| **B3 离散 token + 掩码离散扩散 / MaskGIT 式迭代解码** | GOCT 式 tokenization（time token 按拍细分 + 类别 token 扩到 line_id x side x type x positionX 桶）+ 掩码训练 | GOCT [A]、MaskGIT [A]、D3PM / MDLM [B] | RFC-0029 §3.3 要求的对照臂；也是唯一能与文献数字对话的臂 |
| **B4 AR 上界** | GOCT / ITGPT 式自回归解码 | GOCT / ITGPT [A]、Mask-Predict [A]、2510.03289 [A] | 质量上界；文献共识是 AR 质量更高但更慢 |

### 8.3 消融清单（每条绑定一个文献约束）

| # | 消融 | 对应文献约束 | 预期要看到什么 |
| :-- | :-- | :-- | :-- |
| 1 | **损失**：MSE 热图 / 加权 BCE / focal / 泊松 NLL | §6：EBC-ZIP 的「MSE → 偏向零」、ITGPT 的位置加权「轻微重加权」 | MSE 臂出现「预测趋零 / 监督信号弱」；泊松臂的 λ ≡ 0 损失发散（可作为契约断言） |
| 2 | 表征：连续强度场 vs 离散 token | GOCT 的重构动机、MaskGIT vs AR | 两条路线的**质量-速度**折中 |
| 3 | 解码：find_peaks + 阈值 / Ogata thinning / 迭代掩码解码 | Ogata 1981、MaskGIT 的迭代必要性 | thinning 应给出与训练目标同构的样本；阈值解码对超参敏感 |
| 4 | 遮盖粒度：按帧 vs 按事件 / 跨度 | §5.5【推断】（无直接文献） | 按帧遮盖疑似「抄邻居」（需以 loss 与生成多样性指标判定） |
| 5 | 时间网格：beat-aligned vs 固定 hop | GOCT 的 aligned/unaligned 消融（8th：87.9% vs 70.0%） | 未对齐应出现**随时间漂移**的输出（与我们的帧率事故同类） |
| 6 | 条件：无音频 / 有音频；无判定线事件 / 有判定线事件 | audio2chart 的「unconditional baseline 很强」 | 音频条件的增益应「一致但有限」；判定线条件的增益是我们的新变量 |
| 7 | **多判定线：单线 vs 多线** | **无文献对照**（太鼓 / osu! 无等价舞台对象） | RFC-0029 v1 的硬性新增；需自建指标（跨线合法率、跨线分配熵） |
| 8 | **侧别：含 side vs 合并 side** | 无文献对照 | 直接检验 RFC-0029 §2.1 的论断（丢掉 side 会损失什么） |
| 9 | 全局 offset 搜索：做 vs 不做 | STRUM / ChartGenEval | 报两栏；若差别很大说明时间对齐本身是主要误差源 |

### 8.4 必须固定的报告协议（源自文献的强制项）

1. 容差：**±20 ms（主）与 ±50 ms（副）双报告**；另报全谱相位偏移。
2. 匹配：贪心一对一；未匹配的生成事件**留在分母**。
3. 平均：按谱平均与 micro 平均**都报**。
4. 解码：**固定规则**与**每谱最优**两栏；两个范式**用同一报告格式**（避免 ITGPT 指出的「一方能调阈值另一方不能」的不对称）。
5. 分解：按时间组（含 12/24 三连组）与按难度**各自分解**。
6. 数据：**先声明入选判据**（STRUM 的 in-envelope 协议），再报数字。
7. 指标验收：新指标先过 **corruption 测试**（响应 + 不变性）才进主报告。
8. 人评：MIREX 三段式的 Phigros 版——算法（合法性 + 难度条件生效）→ 专家（不少于 5 名有经验玩家；维度：音乐契合度 / 创造性 / 手感，外加 humanity score）→ 社区（人类谱混入盲测 + 「指出哪张是人写」）。

---

## 9. 存疑与未核实清单

| # | 条目 | 状态 |
| :-- | :-- | :-- |
| 1 | CenterNet 的 α = 2, β = 4 | **未核实**：原文只写 α and β are hyper-parameters，具体取值未出现在抓取片段中。引用时不得写具体数字 |
| 2 | ITGPT 的位置权重（triplet/24th 及其余位置的具体数值）、诊断损失权重、学习率、dropout 上界、生成耗时 | **未核实**：ar5iv 把 MathML 剥离为占位符；仅确认 downbeat/offbeat/16th = 2 |
| 3 | STRUM 的同行评审状态 | **未核实**：arXiv 预印本（独立研究者），是否已发表未知 |
| 4 | TCP 的指标定义、损失、数据规模 | **未取到全文**（仅摘要 + 检索片段）；其论文中的数字一律不得引用 |
| 5 | DDG（难度渐变）、Dancing Monkeys（规则法）原文 | **未定位**；仅由 ITGPT 相关工作转述（C 级） |
| 6 | Mapperatorinator / osu!dreamer / BeatLearning / osumapper 的架构与规模 | **未核实**：GitHub 页面抓取被拒；仅由 MIREX wiki 列为「有代码无论文」 |
| 7 | EBC-ZIP 中 ZIP-NLL 相对 MSE 的具体增益数值 | **未取到**：只确认动机陈述与做法 |
| 8 | GenéLive! 的业务反馈细节（成本减半） | **作者自述**，未提供指标化的用户研究设计 |
| 9 | 「Phigros 专属生成工作不存在」 | **检索阴性结论**，非穷尽性证明。受 §1.4 通道限制；且 Q3 已证明 arXiv 索引对含标点关键词会漏检 |
| 10 | 「泊松 NLL 比 BCE 更不容易塌到全 0」的直接 A/B 文献 | **未检索到**。§6.3 的论证是数学推导 + 邻近领域间接证据（【推断】） |
| 11 | Phigros 是否存在公开的星级 / 难度计算器（MIREX 用的是 osu! 的） | **未查证** |
| 12 | DDC 的人评部分细节 | **未核实**（未取该章节） |
| 13 | GOCT 缩写 GOCT 的展开 | **未核实**（作者站点只给出模型名） |
| 14 | 非齐次泊松对数似然标准式的原始教材 / 综述出处 | **未逐字核实**：本文以 Omi 2019 摘要与 EBC-ZIP 损失形式交叉印证 |
| 15 | 「用 MC 估计 ∫λ 会因 Jensen 不等式产生偏差」的文献出处 | **未找到**；本文自证（【推断 / 数学事实】） |
| 16 | Shchur 综述、NSTPP、D3PM、MDLM、Focal Loss、Hawkes 1971、Chan & Vasconcelos、Lempitsky & Zisserman 等 | **仅元数据核实**（题名 / 年份 / venue / DOI 经 OpenAlex 或会议页确认），**正文未通读**。引用其具体做法或结论前需补读 |
| 17 | 「按事件遮盖优于按帧遮盖」 | **无直接文献**（§5.5 表②），属 RFC 的【推断】 |
| 18 | Phigros 侧 positionX 的桶宽 | **仍阻塞**（RFC-0029 §8.2 Q8）；本文件不能提供任何单位依据 |

---

## 10. 来源清单

**分级**：A = 一手（论文全文 / 官方页面 / 官方仓库）；B = 权威二手或结构化书目元数据；C = 二手转述或摘要级。
**深度**：全文 = 本文抓取并阅读了正文相关章节；摘要 = 只读了摘要或引言；元数据 = 只确认题名 / 年份 / venue / DOI。

### 10.1 谱面生成主线

| 文献 | 分级 | 深度 | URL |
| :-- | :-- | :-- | :-- |
| Donahue, Lipton, McAuley. Dance Dance Convolution. ICML 2017 | A | 全文（相关章节） | https://ar5iv.labs.arxiv.org/html/1703.06891 |
| Yi, Lee, Lee. Beat-Aligned Spectrogram-to-Sequence Generation of Rhythm-Game Charts（= **GOCT**）. ISMIR LBD 2023 | A | 全文 | https://ar5iv.labs.arxiv.org/html/2311.13687 ；官方站 https://stet-stet.github.io/goct ；代码 https://github.com/stet-stet/goct_ismir2023 |
| O'Malley. ITGPT: A Transformer Based Architecture for the Generation of DDR and ITG Charts. arXiv 2607.14148 | A | 全文 | https://ar5iv.labs.arxiv.org/html/2607.14148 ；代码 https://github.com/miguelomalley/ITGPT |
| O'Malley. Dance Dance ConvLSTM. arXiv 2507.01644 | A | 全文（相关章节） | https://ar5iv.labs.arxiv.org/html/2507.01644 |
| Takada et al. GenéLive! Generating Rhythm Actions in Love Live! AAAI 2023 | A | 全文（相关章节） | https://ar5iv.labs.arxiv.org/html/2202.12823 ；DOI https://doi.org/10.1609/aaai.v37i4.25657 ；代码 https://github.com/KLab/AAAI-23.6040 |
| Halina, Guzdial. TaikoNation. FDG 2021 | A | 全文（相关章节） | https://ar5iv.labs.arxiv.org/html/2107.12506 ；DOI https://doi.org/10.1145/3472538.3472589 |
| Hanzen, Halina, Guzdial. Time-based Chart Partitioning. AIIDE 2025 | A（摘要页） | 摘要 | https://ojs.aaai.org/index.php/AIIDE/article/view/36808 ；DOI https://doi.org/10.1609/aiide.v21i1.36808 |
| Opria. STRUM. arXiv 2605.12135 | A | 全文（相关章节） | https://ar5iv.labs.arxiv.org/html/2605.12135 |
| Tripodi. audio2chart. arXiv 2511.03337 | A | 摘要 + 引言 | https://ar5iv.labs.arxiv.org/html/2511.03337 |
| Lin. ChartGenEval. arXiv 2607.12857 | A | 全文 | https://ar5iv.labs.arxiv.org/html/2607.12857 |
| Lee, Jeong. AutoOsu. ISMIR LBD 2023 | B | 元数据（经 MIREX 列表） | https://github.com/issyun/AutoOsu |
| GenerationMania: Learning to Semantically Choreograph. AIIDE 2019 | B | 元数据 | DOI https://doi.org/10.1609/aiide.v15i1.5224 ；arXiv https://arxiv.org/abs/1806.11170 |
| Mapperatorinator / osu!dreamer / BeatLearning / osumapper | B | 元数据（仓库页面不可达） | https://github.com/OliBomby/Mapperatorinator 等，见 MIREX 任务页 |
| MIREX 2026: Rhythm Game Chart Generation 任务页 | B | 全文 | https://music-ir.org/mirex/w/index.php?title=2026:Rhythm_Game_Chart_Generation |

### 10.2 点过程 / 泊松 NLL

| 文献 | 分级 | 深度 | URL |
| :-- | :-- | :-- | :-- |
| Hawkes. Spectra of some self-exciting and mutually exciting point processes. Biometrika 1971 | B | 元数据 | https://doi.org/10.1093/biomet/58.1.83 |
| Du et al. Recurrent Marked Temporal Point Processes. KDD 2016 | B | 元数据 | https://doi.org/10.1145/2939672.2939875 |
| Mei, Eisner. The Neural Hawkes Process. NeurIPS 2017 | B | 元数据 | https://arxiv.org/abs/1612.09328 |
| Omi, Ueda, Aihara. Fully Neural Network based Model for General Temporal Point Processes. NeurIPS 2019 | A | 摘要 + 引言 | https://ar5iv.labs.arxiv.org/html/1905.09690 |
| Omi, Aihara, Asoh. Intensity-Free Learning of Temporal Point Processes. ICLR 2020 | B | 元数据 | https://arxiv.org/abs/1909.12127 |
| Chen et al. Neural Spatio-Temporal Point Processes. arXiv 2011.04583 | B | 元数据 | https://arxiv.org/abs/2011.04583 |
| Shchur et al. Neural Temporal Point Processes: A Review. IJCAI 2021 | B | 元数据 | https://arxiv.org/abs/2104.03528 ；DOI https://doi.org/10.24963/ijcai.2021/623 |
| Integration-free Training for Spatio-temporal Multimodal Covariate Deep Kernel Point Processes. NeurIPS 2023 | B | 元数据 | https://arxiv.org/abs/2310.05485 |
| Ogata. On Lewis' simulation method for point processes. IEEE T-IT 1981 | B | 元数据 | https://doi.org/10.1109/tit.1981.1056305 |
| Peeling, Li. Poisson point process modeling for polyphonic music transcription. JASA 2007 | B | 元数据 | https://doi.org/10.1121/1.2716156 |
| An Inhomogeneous Weibull-Hawkes Process to Model Underdispersed Acoustic Cues. JABES 2024 | C | 检索片段 | St Andrews 仓库 PDF（见 §4.2） |
| Chan, Vasconcelos. Bayesian Poisson regression for crowd counting. ICCV 2009 | B | 元数据 | https://doi.org/10.1109/iccv.2009.5459191 |
| Lempitsky, Zisserman. Learning To Count Objects in Images. NIPS 2010 | B | 元数据 | https://robots.ox.ac.uk/~vgg/research/counting/ |
| Ma, Sanchez, Guha. EBC-ZIP: Improving Blockwise Crowd Counting with Zero-Inflated Poisson Regression. arXiv 2506.19955 | A | 摘要 + 引言 | https://ar5iv.labs.arxiv.org/html/2506.19955 |

### 10.3 掩码生成建模 / 离散扩散

| 文献 | 分级 | 深度 | URL |
| :-- | :-- | :-- | :-- |
| Chang et al. MaskGIT: Masked Generative Image Transformer. CVPR 2022 | A | 全文（相关章节） | https://ar5iv.labs.arxiv.org/html/2202.04200 |
| Ghazvininejad et al. Mask-Predict. EMNLP 2019 | A | 摘要 + 引言 | https://ar5iv.labs.arxiv.org/html/1904.09324 |
| Austin et al. Structured Denoising Diffusion Models in Discrete State-Spaces (D3PM). NeurIPS 2021 | B | 元数据 | https://arxiv.org/abs/2107.03006 |
| Shi et al. Simplified and Generalized Masked Diffusion for Discrete Data. NeurIPS 2024 | B | 元数据（经 2510.03289 引用） | https://arxiv.org/abs/2406.04329 |
| Why Mask Diffusion does not Work. arXiv 2510.03289 | A | 全文（相关章节） | https://ar5iv.labs.arxiv.org/html/2510.03289 |

### 10.4 损失 / 关键点热图 / 评估

| 文献 | 分级 | 深度 | URL |
| :-- | :-- | :-- | :-- |
| Lin et al. Focal Loss for Dense Object Detection. ICCV 2017 | B | 元数据（正文未读；动机经 CornerNet / CenterNet 转述） | https://arxiv.org/abs/1708.02002 |
| Law, Deng. CornerNet. ECCV 2018 | A | 全文（相关段落） | https://ar5iv.labs.arxiv.org/html/1808.01244 |
| Zhou, Wang, Krähenbühl. Objects as Points (CenterNet). arXiv 1904.07850 | A | 全文（相关段落） | https://ar5iv.labs.arxiv.org/html/1904.07850 |
| Li et al. Gradient Harmonized Single-stage Detector. AAAI 2019 | B | 元数据 | https://arxiv.org/abs/1811.05181 |
| Zhang et al. DARK: Distribution-Aware Coordinate Representation. CVPR 2020 | B | 元数据 | https://arxiv.org/abs/1910.06278 |
| Hawthorne et al. Onsets and Frames. ISMIR 2018 | B | 元数据 | https://arxiv.org/abs/1710.11153 |
| Raffel et al. mir_eval. ISMIR 2014 | B | 元数据 | https://doi.org/10.5281/zenodo.1416527 |
| Yang, Lerch. On the evaluation of generative models in music. Neural Computing and Applications | B | 元数据 | https://doi.org/10.1007/s00521-018-3849-7 |
| Survey on the Evaluation of Generative Models in Music. ACM Computing Surveys 2025 | A | 全文（相关章节） | https://ar5iv.labs.arxiv.org/html/2506.05104 |

### 10.5 工具与其它

| 资源 | 分级 | 说明 |
| :-- | :-- | :-- |
| OpenAlex API | B | 所有 DOI / 年份 / venue 的核实通道 |
| arXiv export API | B | 全库枚举检索通道（§1.4） |
| github.com/swordalt/phigros-chart-downloader | C | Phigros 谱面 / 资源导出工具（非生成模型） |


