# Plan 04 — 生成主干：掩码补全 Encoder-Decoder

> 状态：🟡 草案 ｜ 阶段：Phase 2 ｜ 负责：gen-agent（生成组）
> 对应代码：`beatmorph/generation/` ｜ 对应奠基章节：§1.2 / §2 / §3.3 / §3.4 / §3.5

## 1. 目标与范围

### 1.1 交付

1. **主线 B2**：掩码补全（完形填空）Encoder-Decoder，对 K 条判定线输出强度场 `λ_k(τ, x, s, c)`（**`τ` = 拍相对坐标，beat-aligned，基本格 1/48 拍，随 BPM 变化**；形状、网格与秒↔τ 换算均由 plan 03 定义，本模块**不得**自行实现时间换算）。
2. **Encoder**：输入 `audio_emb` + **判定线事件轨**——普通事件 **4 层跨层求和后**的 5 条轨（moveX / moveY / rotate / alpha / speed）+ `extended` **第 5 层**；**滑动窗口自注意力 + 周期性全局层**。
3. **Decoder**：完形填空；**必须显式提供 mask 通道**；**按事件遮盖而不是按帧遮盖**；多线**共享权重 + 每条线一个 line embedding（作 query）**，支持**可变 K**；**判定线不可互换**（排序不变性不成立，禁止集合预测式任意匹配）。
4. **训练目标 = 非齐次泊松 NLL，对 K 个场求和**；**不需要**为「分配到哪条线」另设分类损失（RFC-0029 §2.4-6）。
5. **迭代并行解码**（Mask-Predict / MaskGIT 式：预测 → 保留高置信 → 重新遮盖 → 重复），步数显式声明；连续场的「置信度」需重新定义。
6. **消融臂 B1 / B3 / B4 / B5**（+ B6 的接口对齐）与条件、表征、时间网格消融——**全部必须实现**，且**每个对照必须指明层级**（训练目标层 / 采样解码层 / 表征层）。**时间网格已由 Q15 定为 beat-aligned**；固定帧网格对照若保留，只作**证据复现臂**，不再作为主路径候选。
7. **G1–G4 门禁**：任何新训练目标/损失先过门禁，门禁未绿不得扩大数据规模。

### 1.2 不交付

- 场的网格、目标构建、NLL 本体、常数基线、碰撞统计、可视化（plan 03，本模块只消费）；**秒 ↔ τ（拍）的任何换算**——那是 `field/` 的独占职责（红线 7、RFC-0029 §3.1/§7-8，见 §5）。
- 场 → 离散事件的解码器（`find_peaks` / Ogata thinning）与合法性后处理（plan 05）。
- 评估指标、双容差报告、人评协议（plan 06）；RPEJSON 读写（plan 02）；RAG / DPO（RFC-0029 未涉及，Phase 3 待重估）。

### 1.3 价值

本模块是范式的心脏：**并行 + 双向上下文**，直接用上谱面事件间的强共现（同时音、连续 Hold、跨判定线齐奏），而这是逐 token 自回归的固有弱点（BasePlan §3.3）。同时它承担整个项目最危险的一步——**把「并行解码是否够用」这一经验问题变成可消融的实验**（B4 上界臂不可省）。

---

## 2. 与奠基文档对应

| 本计划项 | 文档依据 | 落地 |
|---|---|---|
| 掩码补全 Encoder-Decoder 为主选 | BasePlan §3.3、§8 决策矩阵 | §4.1/§4.2 |
| `audio_emb` + 判定线事件轨进 Encoder | BasePlan §2、§3.3、§3.5 | §4.1 |
| 滑动窗口自注意力 + 周期性全局层 | BasePlan §3.3 | §4.1（窗口/周期为 config；见 §9-2） |
| **必须显式提供 mask 通道** | BasePlan §3.3、RFC-0029 §3.3-1 | §4.2 + 契约测试 |
| **按事件遮盖而非按帧遮盖** | BasePlan §3.3、RFC-0029 §3.3-2 | §4.2（`literature §5.5②` 标注该判据无直接文献，属【推断】） |
| 多线共享权重 + line embedding 作 query，可变 K | RFC-0029 §2.4-2 | §4.2 |
| 判定线不可互换，`line_id` 非对称标签 | RFC-0029 §2.4-3/§2.4-8 | §4.2/§4.3 + 排列敏感性契约测试 |
| Decoder 必须看到全部 K 条线的场 | RFC-0029 §2.4-4 | §4.2（禁止分线独立预测） |
| 泊松 NLL 对 K 个场求和，**无** line 分类损失 | BasePlan §2、RFC-0029 §2.4-6/§3.2 | §4.3 |
| 场的时间轴是 **τ（beat-aligned，1/48 拍）** | BasePlan §1.2/§3.2.4、RFC-0029 §3.1（Q15 决议） | §3.1/§3.2（形状中的 `T` 即 τ 格数） |
| 秒 ↔ τ 换算只在 `field/` 内；本模块不实现第二套 | RFC-0029 §3.1/§7-8、CLAUDE.md 红线 7 | §1.2/§5 + 源码级断言（§8） |
| 条件注入：难度必做 / 线事件轨必做 / 音频必做 | BasePlan §3.5 | §4.1 |
| B1 / B3 / B4 / B5 消融臂 | BasePlan §3.3、RFC-0029 §5.2、`literature §8.2` | §4.5 |
| absorbing 扩散 NELBO ≡ 时间步加权掩码 CE | BasePlan §3.3 注（arXiv 2510.03289 式 (10)） | §4.5「对照层级」 |
| 新损失先过 G1–G4 | BasePlan §9、CLAUDE.md 红线 7、AGENTS.md §4 | §6.1 |

**偏离 1（窗口与全局层周期）**：奠基只写「滑动窗口自注意力 + 周期性全局层」，**未给数字**。`docs/references/newplan.md`（非权威参考）建议「窗口 256–512 帧 + 每 4 层一个全局层」，但该文档的「MERT ≈ 50 Hz」是**已知错误值**（权威派生值为 `24000 / 320 = 75 Hz`，BasePlan §3.1）。本计划把窗口长度与全局层周期设为 **config 派生的超参**，默认值待 RFC 定稿。见 §9-2。

**偏离 2（遮盖比例与调度）**：文档未给。本计划把遮盖比例 `r` 与 MaskGIT 式调度 `γ(t/T)` 作为超参并列入消融（初始区间取 newplan 的 30%–50%，**非权威来源**）。见 §9-3。

**偏离 3（B3 与 B4 的边界）**：BasePlan §3.3 与 RFC-0029 §5.2 同时列了「离散 token + 自回归（GOCT 配置）」与「自回归上界臂」，字面上重叠。本计划的划分：**B3 = GOCT 配置复现**（与文献公开数字对话，含 4-lane 结构映射）；**B4 = 自研离散化上的 AR**（与 B2 严格同表征、只差「顺序 vs 并行」），用作 B2 的质量上界。此为解释性划分，**需 RFC 确认**。见 §9-1。

**偏离 4（连续场置信度）**：迭代并行解码需要「置信度」，而文献中的 MaskGIT 置信度是 token 概率；**连续强度场无对应做法**（literature §5.2 明确「属我们的新增设计」）。本计划给出三种候选定义并做消融。见 §9-4。

**偏离 5（训练目标的重标定）**：遮盖补全若只对被遮盖事件算事件项、却对**全域**算积分项，最优解会被系统性压低 `(1-r)` 倍（欠计数）。本计划要求 `1/(1-r)` 的 Horvitz–Thompson 式重标定作为**契约**，并保留 `L_full` 作为校准口径。推导见 §4.3，属本计划新增设计。见 §9-6。

---

## 3. 接口契约

### 3.1 模型 API

```python
class MaskedFieldModel(nn.Module):
    def forward(self, batch: FieldBatch) -> FieldOutput: ...
    def sample(self, batch: FieldBatch, *, steps: int, schedule: str, confidence: str) -> FieldOutput: ...

@dataclass
class FieldBatch:
    audio_emb: Tensor          # (B, T_audio, 1024)  Stage 0 输出；**音频帧轴**，帧率由 config 派生（G4）
    frame_rate: float          # 派生量，禁止硬编码（BasePlan §3.1）；只服务音频帧轴，**不是场的时间轴**
    line_tracks: Tensor        # (B, K, T_line, F_line)  见 §3.2；4 层普通事件轨已跨层求和；
                               # 其时间轴经 field/ 的换算对齐到 τ（本模块不实现该换算）
    line_mask: Tensor          # (B, K) bool            padding / 无效线
    difficulty: Tensor         # (B,) float32           定数 difficulty（非 level 字符串）
    counts: Tensor | None      # (B, K, T, N, 2, C) int16   训练目标（plan 03 产出）；T = **τ 格数**（1/48 拍）
    occlusion: Tensor | None   # (B, K, T, N, 2, C) bool    训练遮盖 mask 通道

@dataclass
class FieldOutput:
    lam: Tensor                # (B, K, T, N, 2, C) float32   λ ≥ 0（plan 03 的 FieldLambda）
    cum: Tensor | None         # (B, K, T) float32           Λ_k(τ)（走 Λ 参数化时非 None；测度含 J(τ)）
    cell_prob: Tensor | None   # (B, K, T, N, 2, C) float32  p_k(x,s,c | t)，Σ = 1
    loss: Tensor | None
    diagnostics: dict[str, Tensor]
```

### 3.2 张量形状（einops 风格）

| 名称 | 形状 | 含义 |
|---|---|---|
| `AudioEmb` | `(B, T_audio, 1024)` | Stage 0 输出；帧率 = `MERT_SAMPLE_RATE_HZ / MERT_CONV_STRIDE_PRODUCT` |
| `LineTracksOrdinary` | `(B, K, T_line, 5)` | moveX / moveY / rotate / alpha / speed，**跨 4 层求和后**（禁止逐层送入） |
| `LineTracksExtended` | `(B, K, T_line, F_ext)` | `extended` 第 5 层选定通道（默认 scaleX / scaleY / color RGB / gif / incline，F_ext = 7；`text` 只作存在性标记，见 §9-12） |
| `LineEmbedding` | `(K_max, d_model)` | 每条判定线一个可学习 query；`K_max` 覆盖实测长尾（中位 30 / p75 52 / 极值 82） |
| `FieldLambda` | `(B, K, T, N, 2, C)` | 与 plan 03 `FieldLambda` 同形同义（`T` = **τ 格数**，由 `BPMList` 与 `BEAT_SUBDIVISION` 派生；`N` 由 `FieldGrid` 决定） |
| `CumulativeLambda` | `(B, K, T)` | `Λ_k(τ)`（对 τ 的累积强度，测度含 `J(τ)`），单调不减；`∫λ_k = Λ_k(T) − Λ_k(0)` |
| `OcclusionMask` | `(B, K, T, N, 2, C)` | 遮盖通道；**与 `line_mask`、`range_mask` 语义分离**（plan 03 §4.4） |

### 3.3 损失 API

```python
# 复用 plan 03：poisson_nll / binned_poisson_nll / constant_baseline_nll
# 积分测度为 J(τ)dτ（J 由 BPMList 派生、只在 field/ 内实现）——本模块只调用，**不重算**
def masked_poisson_loss(out: FieldOutput, batch: FieldBatch, *, reduction="sum") -> Tensor:
    # = -(1 / (1 - r)) * Σ_{n ∈ occluded} log λ(e_n)  +  Σ_k ∫ λ_k        （见 §4.3 推导）
def full_poisson_loss(out: FieldOutput, batch: FieldBatch) -> Tensor:
    # = -Σ_{n ∈ all} log λ(e_n) + Σ_k ∫ λ_k                              （校准 / G3 / 报告口径）
```

**契约断言**：
1. `r == 0` 时 `masked_poisson_loss == full_poisson_loss`（相对误差 ≤ 1e-6）。
2. **排列敏感性**：交换两条判定线的场与其事件（同一条 `line_mask` 置换）后，loss **必须改变**——判定线不可互换（RFC-0029 §2.4-3）。这与 G2 同构，但对象是线的身份。
3. `line_mask` 为 False 的线对 loss 与梯度的贡献**恰为 0**；`∫λ` 不跨 padding 线求和。
4. **不存在**任何 line 分类损失/softmax 分配项（源码级断言：损失函数只由事件项与积分项构成）。
5. 泊松 NLL **不得**与 focal / 热图 / BCE 损失混用（RFC-0029 §3.4、BasePlan §3.4 注：两者不在同一测度）。

---

## 4. 内部设计

### 4.1 Encoder

- **音频分支**：`Linear / Conv1d(1024 → d_model)` + 以**秒**为基准的位置编码（由帧率换算，禁止写死 75 或帧号）——这是**音频帧轴**的编码，与场的时间轴（τ）是两回事。
- **判定线事件轨分支**：**跨层求和必须在编码之前完成**。语义依据（A 级）：「判定线的最终普通事件值等于每个普通事件层级的事件值**相加**」，`extended`（第 5 层）只有一层故直接取值（[phigros-format §4.3](knowledges/phigros-format.md)、RFC-0029 §8.1）。契约测试：编码器输入的普通轨维度必须是 **5**（求和后），而非 5×4=20。事件轨到 τ 的对齐**只经 `field/` 的换算接口**（本模块不自建）。
- **稀疏注意力**：滑动窗口自注意力（窗口以**帧**为单位，帧率派生；若该层作用在**场张量**上则窗口改以 **τ 格**为单位——两者不得混用，场的时间轴见 plan 03 §3.1）+ 每 `global_period` 层插一个全局层；复杂度约 `O(T·w)` + `O(T²/global_period)`。窗口/周期为 config 超参（§9-2）。
- **条件注入顺序**（BasePlan §3.5）：难度 → 可学习嵌入 → AdaLN（或前缀）；线事件轨 → cross-attention 的 K/V（舞台上下文）；音频 → cross-attention 的 K/V。
- **判定线的可变 K**：`K` 不进入任何输出层形状；线数通过 `K` 维批处理 + `line_mask` 处理。实测线数分布（中位 30 / p25 24 / p75 52 / 极值 2–82）要求同一组权重同时服务 K=2 与 K=82（M2 验收项）。

### 4.2 Decoder（掩码补全）

- **输入**：被遮盖的场表示 + 三个语义分离的 mask（`occlusion` / `line_mask` / `range_mask`，plan 03 §4.4）+ 位置编码 + line embedding。
- **mask 通道是硬要求**：没有它，模型无法区分「此处无 note」与「此处被遮盖」（RFC-0029 §3.3-1）。契约测试：向前向中注入不同的 mask 通道必须改变输出（随机权重下可判定），且 mask 通道必须是网络输入（梯度可回传）。
- **按事件遮盖**：以 **note 事件**为单位遮盖——一个事件连同其 `hold-end` 配对一起遮，且同一 `(k, x 桶, s)` 纤维上的配对点不得只遮一半。**禁止**按帧/按格随机遮盖（note 极稀疏，邻帧几乎必然含相同事件 → 模型可以「抄邻居」，RFC-0029 §3.3-2；该判据在 literature §5.5② 被标注为无直接文献支持的【推断】，故必须由本模块的消融给出实证）。
- **多线**：共享权重 + line embedding 作 query；解码器必须**同时看到全部 K 条线的场**（全局注意力天然满足；RFC-0029 §2.4-4）。**禁止**分线独立预测（会丢掉「哪条线该接这个音取决于所有线当前位姿」的交互）。
- **不做排列匹配**：不使用 Hungarian matching、不做集合预测式对齐——判定线各有独立事件轨，`line_id` 不是对称标签（RFC-0029 §2.4-3）。
- **输出头**：`λ`（softplus/exp 保证 ≥ 0）；或采用 plan 03 §4.6 的因子化 `λ = Λ'·p`，使 `∫λ` 精确且两条积分路径的一致性成为可断言条件。

### 4.3 训练目标与**遮盖重标定**（本计划的关键推导）

```
L_masked(r) = -Σ_{n ∈ occluded} log λ(e_n) + Σ_k ∫ λ_k
E_mask[L_masked(r)] = -(1-r)·Σ_{n ∈ all} log λ(e_n) + Σ_k ∫ λ_k
```
即事件项被系统性地缩小 `(1-r)` 倍而积分项不变。由于泊松 NLL 对 `λ` 的尺度敏感（`λ → c·λ` 时最优 `c` 由两项平衡决定），**未重标定的最优解会把整场强度乘上 `(1-r)`，即系统性欠计数**。r=0.5 时模型会少生成约一半的 note——这正是本项目最怕的静默失效类型。

因此契约：**训练损失必须做 `1/(1-r)` 的 Horvitz–Thompson 式重标定**，使期望等于全谱 NLL：

```
L_train = -(1/(1-r)) · Σ_{n ∈ occluded} log λ(e_n) + Σ_k ∫ λ_k        （全 K 线、全域积分）
```
- `L_full`（全事件 + 全域积分）用于**验证 / 校准 / G3 对照**，并作为 `r = 0` 时的等价性断言（契约 §3.3-1）。
- 积分项**永远在完整域上计算**（不得随遮盖比例缩放），且覆盖**全部 K 条线**（含无 note 的线，理由见 plan 03 §4.7）。
- 事件项使用 plan 03 的**桶内计数形式** `n_j log λ_j`（同格多事件全额计入）。

> **与 Q15（时间网格）无关（须显式声明，避免误读）**：`1/(1-r)` 的 Horvitz–Thompson 重标定修正的是**事件项的期望值**（概率层面），与时间参数化无关——测度改写为 `J(τ)dτ` 后本节推导逐字成立，`r` 是遮盖比例、不是网格量。积分项仍**永远在完整域上计算**（全 K 线、全 τ 域、含 `J(τ)` 测度）。

### 4.4 迭代并行解码

- **流程**（MaskGIT / Mask-Predict 式）：全部遮盖 → 一次前向预测全场 → 按置信度保留最确定的一部分 → 其余重新遮盖 → 重复。**一步到位不可行**（MaskGIT 原文：理论上可以一次推断全部 token，但作者明确 *we find this challenging due to inconsistency with the training task*，实际用 8 步完成 256 token），因此契约：`steps ≥ 2`，且步数是**必须报告**的消融维度。
- **连续场的置信度重定义**（文献无对应做法）候选：
  (i) **峰值显著性**：局部对比度 / top-1 与 top-2 峰之比；
  (ii) **事件级置信度**：场在事件位置的 `λ` 与该邻域期望计数的比（v1 默认）；
  (iii) 多次前向或多采样的方差。
  三种都实现并消融（§9-4）。
- **已确定位置的处理**：被保留的位置在后续步中作为条件输入（与训练时的「部分可见」分布一致），其 mask 通道相应置为「未遮盖」。
- **终止**：步数上限、保留比例为单调 schedule `γ(t/T)`，均显式声明并写入日志。

### 4.5 消融臂与**对照层级**

| 臂 | 内容 | 对照层级 | 不可省的理由 |
|---|---|---|---|
| **B0** | 常数 / 先验基线 `λ ≡ N/\|Ω\|`（plan 03 闭式） | 训练目标层 | G3 门禁的下界（DCRand 同构先例） |
| **B1** | 热图 + focal（CornerNet / CenterNet 配置）+ 峰值解码 | 表征 + 目标层（**独立臂**） | 文献主流做法，**必须是正式消融臂而非稻草人**；其目标 y 是**未归一化**高斯（`∫y ≠ 事件数`），与点过程不同测度，**不得与 B2 混用损失** |
| **B2** | **主线**：泊松 NLL 强度场 + 掩码补全 | — | 本计划主体 |
| **B3** | 离散 event token + 自回归（**GOCT 配置**） | 表征 + 解码层 | 唯一能与文献数字对话的臂（GOCT / ITGPT 公开数字） |
| **B4** | **自回归上界臂**（自研离散化上的 AR） | 解码层（顺序 vs 并行） | arXiv 2510.03289 对并行采样无理论保证的质疑 → **不可省** |
| **B5** | 掩码**离散**扩散（absorbing-state） | **采样 / 调度层** | 与 B2 在训练目标层**不是两件事**（见下），必须在文档与报告中写明比的是哪一层 |
| **B6** | 解码策略：`find_peaks` vs **Ogata thinning** | 解码层 | 解码是独立变量，不能与损失混谈；实现归 plan 05，编排与冻结接口归本模块 |

**B1 的具体配置（防稻草人）**：CornerNet 式 penalty-reduced focal（正样本 `(1-p)^α log p`、负样本 `(1-y)^β p^α log(1-p)`，y 为未归一化 2D 高斯、重叠取 element-wise max）+ DDC 的 Hamming 平滑 + **每难度一个阈值**（DDC 实测阈值从 0.5 换到最优时 F1 由 0.5006 → 0.7317，阈值敏感度极高，因此必须与 B2 使用**同一套报告格式**：固定解码规则 + 每谱最优两栏）。α、β 的具体取值在文献中未核实（literature §9-1），**不得写具体数字**，作为超参搜索。

**B3 的 GOCT 配置**：3 层 Encoder-Decoder Transformer（d = 256）、每事件 2 个 token（time token 0–95 + action token）、`hop = 1/48` beat、CE + label smoothing 0.02、`±30 ms` micro-F1。
**不可照搬之处**：GOCT 是 4 lane 离散、无舞台对象；我们的标记是 `(line_id, positionX, side, type)` × 多判定线 → 其 80 个 action token 无法直接覆盖 `positionX`，必须重写 tokenization（literature §8.1）。

**B5 的层级声明（必须写进报告）**：absorbing-state 离散扩散的 NELBO **等价于按时间步加权的掩码交叉熵**（arXiv 2510.03289 式 (10)，BasePlan §3.3 注）——因此 B2 与 B5 在**训练目标层是同一件事**，真正的差异在 (a) 时间步加权的具体形式、(b) 采样器、(c) 是否用连续噪声调度。若要比「连续场 vs 精确离散」，正确对照是 **B2 vs B3**（表征层）；把 B2 vs B5 说成「训练目标消融」是**不可解释**的。

**其它消融（每条绑定一个文献约束）**：
- 遮盖粒度：按事件 vs 按帧（literature §8.3-4）。
- 时间网格：**beat-aligned（主路径，Q15 已决）** vs 固定帧网格（**仅证据复现臂**；复现 GOCT 的 8 分音符 87.9% vs 未对齐 70.0%，未对齐会随时间漂移 → 与帧率事故同类，literature §2.2/§8.3-5）。该臂**不得**被用来重新选型；网格定义归 plan 03。
- 条件：无音频 / 有音频（audio2chart：unconditional baseline 很强，音频增益「一致但有限」）；无判定线事件轨 / 有（**这是本项目的新变量，无文献对照**）。
- 多线：单线 / 多线（literature §8.3-7）；侧别：含 `side` / 合并 `side`（§8.3-8，直接检验 RFC-0029 §2.1 的论断）。
- 迭代解码步数（MaskGIT 的 8 步为参照）。

### 4.6 训练与工程

- bf16 混合精度；AdamW；PyTorch Lightning；Hydra 组合式配置（`configs/` 由 infra-agent 维护，本模块提需求）；TensorBoard（`wandb.entity=null`）。
- 上下文是**段落级**而非全曲（BasePlan §5，单卡 ≥16 GB 起步）；超长曲的分段与衔接策略需 RFC（原 RFC-0008 议题，编号由主会话统一重排）。
- 训练日志必须包含：G1–G4 门禁结果（`beatmorph/infra/sanity.py` 的 `summarize()` 文本）、per-line 线熵、per-side / per-channel 的 NLL 分解、遮盖比例 `r`、迭代解码步数。
- 日志统一走 `beatmorph.core.logging.get_logger`，禁止裸 `print` 进生产路径。

---

## 5. 依赖关系

- **上游**：plan 01（`audio_emb` + 由 config 派生的帧率常量）；plan 02（RPEJSON 解析、`judgeLineList`、**跨层求和后的线事件轨**、`difficulty` 定数）；plan 03（`FieldGrid` / `FieldTarget` / `poisson_nll` / 常数基线 / 碰撞统计 / 三个 mask 的语义）；`core/contracts`（contracts-agent）；`configs/`（infra-agent）。
- **下游**：plan 05（消费 `FieldLambda` 做双解码臂与合法性后处理）；plan 06（评估矩阵与人评）；plan 07/08（训练栈、CLI、API）。
- **外部库**：`torch>=2.5`、`einops`、`pytorch-lightning`、`hydra-core`、`tensorboard`；可选 `flash-attn` / `xformers` 仅作加速（**不作为正确性依赖**，缺失时必须能退化为标准注意力）。

---

## 6. 里程碑与验收

### 6.1 门禁先行（红线 7 / AGENTS.md §4）

**任何新训练目标/损失（泊松 NLL、遮盖重标定、focal、CE、NELBO）在扩大数据规模之前必须先过 G1–G4**，统一调用 `beatmorph/infra/sanity.py`（判据取该模块默认值），结果写入训练日志；**门禁未绿不得扩大数据规模**。G3 的常数基线值取自 plan 03 的闭式 `N * (1 + log(\|Ω\|/N))`。

### 6.2 里程碑表（12 条）

| # | 里程碑 | 验收（可量化） |
|---|---|---|
| M1 | 契约与形状冻结 | `FieldBatch`/`FieldOutput` 形状断言全绿；三个 mask 语义分离测试通过；**无权重、无 GPU**；`uv run mypy beatmorph/generation` strict 零错误 |
| M2 | 可变 K 前向 | 同一组权重跑通 `K = 1 / 30 / 82`（中位与实测极值）与批量内混合 K（padding + `line_mask`）；`line_mask=False` 的线对输出与 loss 贡献恰为 0 |
| M3 | **G1–G4 门禁全绿** | G1 单 batch 过拟合：1–4 样本上末步 loss ≤ `max(0.05, 0.1×首步)`；G2 打乱标签：shuffled ≥ real × 1.05；G3：模型 ≤ 0.9 × 常数基线；G4：`frames ≈ duration × frame_rate`（±2 帧）。四项结果文本入训练日志 |
| M4 | 遮盖重标定契约 | 合成场上：未重标定损失（`r = 0.5`）优化得到的全场积分 `Σ_k ∫λ_k` ≈ `0.5 ×` 真值（±5%），重标定后回到 `真值 ±5%`；`r = 0` 时 `L_train == L_full`（相对误差 ≤ 1e-6） |
| M5 | **按事件 vs 按帧遮盖**消融 | 同一模型/数据/步数下，事件遮盖臂在 held-out 的 F1@±20ms 更高，且「抄邻居」诊断指标（被遮盖事件邻域的事件命中率）更低；两维（粒度 × 比例 `r`）各报一组 |
| M6 | 迭代并行解码跑通 + 步数消融 | `steps ≥ 2` 契约；给出 `{steps} × {F1@±20ms, F1@±50ms, 平均耗时}` 的质量–速度曲线；`steps = 1` 的失败被记录（对齐 MaskGIT 的「一次到位不可行」） |
| M7 | **B4 AR 上界臂** | 同表征下的 AR 臂跑通并给出与 B2 的质量差（上界）与推理耗时比；若 B2 已接近 B4，则「并行足够」有据；若差距显著，则 B2 的迭代策略必须改进——两种结论都必须写进报告 |
| M8 | **B1 热图 + focal 正式臂** | 采用 CornerNet 配置（α/β 作超参搜索，**不写未核实的具体值**）+ DDC 式 Hamming 平滑 + **每难度阈值**；报告「固定解码规则」与「每谱最优阈值」两栏，格式与 B2 **完全一致** |
| M9 | **B3 GOCT 配置臂** | 3 层 d=256、time token 0–95 + action token、`hop = 1/48` beat、CE + label smoothing 0.02；报 `±30 ms` micro-F1 与公开数字并肩呈现；tokenization 的改写点（`positionX` 桶 + `line_id` + `side`）单独说明 |
| M10 | **B5 absorbing 扩散臂 + 层级声明** | 跑通并**在报告中显式写明对照层级**（采样/调度层）；文档中同时给出「B2 与 B5 训练目标同构（NELBO ≡ 时间步加权掩码 CE）」的说明，避免把采样层差异读成目标层差异 |
| M11 | 条件 / 表征消融矩阵 | 每格报 F1@±20ms 与 ±50ms **双栏**、按难度分档、按时间组（含 12/24 三连）分解：无音频/有音频、无事件轨/有、单线/多线、含 side/合并 side、beat-aligned（主路径）/固定帧网格（证据复现臂）；单线臂需额外报跨线合法率 |
| M12 | **阶段出口：首版可玩 RPEJSON**（与 plan 05/06 联合） | 生成的谱面经合法性校验**空违规**（同刻按键上限、Hold 区间、越界、跨线几何冲突）；难度条件生效（可区分不同 `difficulty` 的密度/类型分布）；产出双容差 F1 报告与人评素材；**NLL 只作校准，不作质量分** |

---

## 7. 风险与缓解

| # | 风险 | BasePlan 编号 | 缓解 |
|---|---|---|---|
| R-1 | MERT 对特定曲风表征不足 | R-1 | Adapter 保持可训练；「无音频 / 有音频」消融**量化**音频条件增益（audio2chart 提示该增益「一致但有限」）；解冻后 6 层须先开 RFC |
| R-3 | 社区谱质量参差，学到坏习惯 | R-3 | 数据质量过滤（plan 02/06）+ 人评兜底；「谱面质量」不作为损失项（无可靠标签） |
| R-4 | **稀疏目标塌陷** | R-4 | 泊松 NLL + **G1–G4 门禁** + B1/B2 对照；**遮盖重标定**（§4.3 的欠计数机制是本模块自证的新风险点，M4 专门验收） |
| R-5 | **多线 K 长尾 / note 极度集中** | R-5 | 全 K 线参与 `∫λ`；不做 line softmax；per-line 线熵诊断；单线/多线消融量化「多线是否真的被用上」 |
| R-6 | 连续场 → 离散事件的解码精度 | R-6 | B6 双解码臂（plan 05 实现）；评估一律在**秒域**，不在帧索引域 |
| R-7 | 物理常量 / 单位漂移（含 beat-aligned 换算外溢） | R-7 | 帧率与桶宽**派生 + 断言**（G4）；**秒↔τ 只在 `field/` 内换算**；mock / fixture 不得固化物理常量；源码字面量扫描 |
| R-8 | 跨线几何冲突不可玩 | R-8 | plan 05 的跨线几何检查；本模块只负责在解码时**可见全部 K 条线**（RFC-0029 §2.4-4） |
| R-04-1 | **并行采样缺乏理论保证**（arXiv 2510.03289） | — | **B4 上界臂不可省**；迭代步数与调度显式声明；「并行略逊于 AR」作为**预期**而非失败（Mask-Predict：+4 BLEU、within about 1 BLEU of AR） |
| R-04-2 | 一步到位解码与训练分布不一致（MaskGIT） | — | `steps ≥ 2` 契约 + 步数消融；置信度保留 + 重新遮盖的 schedule 必须实现 |
| R-04-3 | 连续场「置信度」无文献定义 | — | §4.4 三种候选定义全部实现 + 消融；把该选择写进报告的方法学部分 |
| R-04-4 | 显存：稠密场 `(B, K, T, N, 2, C)` | — | plan 03 的派生估算：每线每秒 96,000 格 × K=30（中位）≈ 2.88e6 格/音频秒（bf16 ≈ 5.8 MB/音频秒）；对策：分段窗口、bf16、梯度检查点、按需线子集（`∫λ` 仍覆盖全部 K）、必要时降 batch |
| R-04-5 | line embedding 退化成「最忙线偏置」，模型只学固定先验 | — | per-line NLL / 线熵监控；**打乱 line embedding 的对照**（交换线身份后 loss 必须变差，与 G2 同构）；单线臂作为参照 |
| R-04-6 | 用 NLL 选模型导致错误方向 | BasePlan §9 | NLL 只作**校准**与门禁；主判据是事件级 F1 双容差 + 合法性 + 人评（ChartGenEval：perplexity 在「常见图案重写」下**下降 37%**，方向错误） |
| R-04-7 | **生成侧自建秒↔τ 换算** → 与 `field/` 分叉（25 Hz 同型静默失效） | R-7 | 换算只在 `field/`（§1.2/§5）；源码级断言：本模块不出现 BPM 分段积分 / `60 / bpm` 型换算 / 自建 `d_tau` 常量；形状中的 `T` 只从 `FieldGrid` 取 |

---

## 8. 测试策略

- **单元（默认 CI，无权重 / 无 GPU）**
  - `tests/unit/generation/test_masks.py`：三个 mask 语义分离；遮盖通道改变输出；mask 通道可回传梯度。
  - `tests/unit/generation/test_variable_k.py`：`K = 1/30/82`、批量内混合 K、`line_mask` 排除 padding 的零贡献断言。
  - `tests/unit/generation/test_losses.py`：`r = 0` 等价性；`(1-r)` 欠计数反例（M4）；**排列敏感性**（交换两条线 → loss 改变）；无 line 分类项的源码级断言。
  - `tests/unit/generation/test_masking.py`：按事件遮盖的粒度正确性（hold-end 配对不被拆散）；比例 `r` 的统计正确性。
  - `tests/unit/generation/test_sampling.py`：`steps ≥ 2` 契约、schedule 单调性、终止条件；置信度三种定义的确定性。
- **契约级**：全部形状/帧率/网格断言进默认 CI；mock 的 plan 01/03 输出**必须引用契约常量**（音频帧数由 `duration × frame_rate` 派生、τ 格数由总拍数 × `BEAT_SUBDIVISION` 派生），不得写死 75 / 1350 / 10.546875；**并断言本模块源码不含任何秒↔τ 换算实现**（无 `60 / bpm` 型分段积分、无独立 `d_tau` 常量）——换算只经 plan 03 的接口。
- **集成**：`tests/integration/generation/test_train_step.py`——用 `data/fixtures/` 微型谱 + 合成 audio_emb 跑一个完整训练步与一次 `sample()`，全程无 GPU。
- **e2e（`@pytest.mark.gpu` / `slow`）**：G1–G4 门禁、B 臂矩阵、端到端 RPEJSON 生成（M12）。
- **门禁脚本化**：G1–G4 作为可重复运行的脚本（同一 seed 下结果可复现），其输出直接进训练日志——**门禁结果不是一次性的检查，而是每次扩大规模前的例行关卡**。

---

## 9. 开放问题

1. **B3 与 B4 的边界**（§2 偏离 3）：本计划按「文献对标 vs 严格上界」划分，BasePlan §3.3 与 RFC-0029 §5.2 的字面重叠需 RFC 裁定，否则消融不可解释。
2. **窗口长度与全局层周期**：文档未给；newplan 的「256–512 帧 + 每 4 层全局」为非权威建议，且其「50 Hz」是错误值。需以 `frame_rate` 为单位重述并做参数扫描。
3. **遮盖比例 `r` 与调度 `γ(t/T)`**：默认值与扫描范围未定（初步 30%–50% 来自非权威参考）。
4. **连续场置信度定义**（§4.4 三候选）的最终选型；文献无先例，属本项目新增设计。
5. ~~**时间网格**~~ **已决（Q15，2026-08-05）**：主路径 **beat-aligned**（1/48 拍），数学改写（测度 `J(τ)dτ`）限定在 `field/` 内；本模块只消费 τ 网格、不实现任何换算（§1.2/§5/§8）。残留问题：**位置编码与条件注入在 τ 轴上的具体形式**（绝对拍 / 小节相对 / 与音频帧轴对齐的插值方式），本模块未定 → 与 plan 03 §9-15 联动。
6. **遮盖重标定形式**（§4.3）：`(1-r)` 的 Horvitz–Thompson 重标定是本计划新增设计；替代方案是「纯 `L_full` 训练 + 遮盖仅作条件增强」（此时不存在欠计数，但对被遮盖位置的梯度信号被稀释）。需评审 + 可能开 RFC。
7. **超长曲的分段与衔接**（原 RFC-0008 议题）：段落级上下文的分段边界、重叠区处理、跨段判定线状态传递。
8. **可变 K 的 batching 策略**：`K_max` 上限、按 K 分桶、以及「K=82 与 K=2 混批」的效率/正确性权衡。
9. **line embedding 是否携带 `father` 关系**：实测 26% 的谱含嵌套线，父线旋转影响子线锚点位置（A 级：`parent_pos + R(parent_rot)·child_translation`）——line embedding 是否需要父线索引/掩码通道，未定。
10. **难度条件的表示**：定数 `difficulty` 实测范围 9.9–18.5、中位 15.6，且存在浮点误差（`14.900001`）→ 需 round 到 0.1；用连续回归头还是分档嵌入，未定（`level` 字段是自由文本，**绝不能 regex 解析**）。
11. **是否引入「主判定线 + 稀疏次级线集合」的结构先验**：数据调研据此提出（survey §7.2 明确标注为**推断、未做实验验证**），与 RFC-0029 §2.4-6「不要另设分配策略」存在张力，须先做实验再定。
12. **`extended` 通道子集与 `text` 事件**：默认取 scaleX / scaleY / color RGB / gif / incline（F_ext = 7），`textEvents` 需文本编码，v1 只记存在性；子集最终由消融决定。
13. **损失的归一化口径**：`sum` vs `per-event`（跨谱可比性 vs 与 RFC 写法一致）；门禁与报告必须用同一口径，未定。
14. **是否对 `fill`/`isFake` 音符单独建通道**：与 plan 03 §9-11 联动，v1 默认排除。
