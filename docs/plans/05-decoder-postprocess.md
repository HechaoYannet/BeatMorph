> 状态：🟡 草案 ｜ 阶段：Phase 2 ｜ 负责：解码与后处理组
> 对应代码：`beatmorph/decoder/`、`beatmorph/io/formats/rpejson/`（写路径）｜ 对应奠基章节：§2、§3.2、§3.6、§9

# Plan 05 — 解码与合法性后处理（强度场 → 离散事件 → RPEJSON）

## 1. 目标与范围

### 1.1 交付什么

把 Stage 1 产出的**判定线局部系多线强度场** `λ_k(τ, positionX, side, type)`（`τ` = 拍相对坐标，beat-aligned，Q15）转成**离散谱面事件**，经合法性后处理后导出 **RPEJSON**。三个交付物：

1. **两条解码路径（互为对照臂 B6）**
   - **D1 峰值检测 + 阈值**（`find_peaks`，v0 基线；RFC-0029 §3.4 明确它是**消融项而非终点**）；
   - **D2 Ogata (1981) thinning**（从 λ 直接采样事件，与训练目标同构的原则性解码器）。
   - 两者**必须同网格**（RFC-0029 §3.4 / §7 硬约束 4），且**都在原始时间域（秒）**输出（RFC-0029 §7 硬约束 5、BasePlan §3.6）。
   - ⚠️ **该网格的时间轴是 τ**（拍相对坐标，beat-aligned，基本格 1/48 拍，随 BPM 变化；RFC-0029 §3.1 Q15）：本模块**只消费** `field/` 给出的 τ 网格与换算，**不得**自行实现秒↔τ 换算（红线 7、RFC-0029 §7-8）。
2. **合法性后处理引擎**：同刻按键上限、Hold 区间合法性、越界**统计**、**跨线几何冲突**检测。
   - ⚠️ **红线 3**：后处理**只做红线校验与钳位，不得改变模型的落点分配与键型逻辑**；
   - ⚠️ **`positionX` 越界只统计不钳位**——`±675` 是**可见边界而非合法值域**（BasePlan §3.2.4、RFC-0029 §3.1、`phigros-units-and-geometry.md` §3.1/§7.5：格式与参考实现均无校验）。
3. **RPEJSON 写路径**（`beatmorph/io/formats/rpejson/`）：IR → RPEJSON 文件，且**只在合法性校验为空违规项时允许导出**（CLAUDE.md 红线 6）。

### 1.2 不交付什么

- 不实现强度场本身（网格/目标/积分归 Plan 03 `field/`）；
- 不实现掩码补全的迭代解码主循环（归 Plan 04 `generation/`）；本 plan 只提供「场 → 事件」的**单次解码算子**与置信度定义接口；
- **不生成判定线事件轨**（RFC-0029 §2.2 决议 Q2：线事件轨是**条件输入**，v1 不做联合生成）；
- 不做评估指标实现（归 Plan 06）与人评；
- **不做任何改变落点分布的「美化」**：不改线、不改 `side`、不把 note 挪到「更合理」的 `positionX`。

## 2. 与奠基文档对应

| 本 plan 内容 | 奠基出处 | 关系 |
| --- | --- | --- |
| 强度场 → 离散事件两条路径 | BasePlan §2（Stage 2 ①）、§3.6-1、RFC-0029 §3.4 | 直接落实 |
| `find_peaks` 是 v0、thinning 是原则性做法 | BasePlan §8 选型矩阵、RFC-0029 §3.4 | 直接落实（二者为 B6 对照臂） |
| 与训练目标同网格 | RFC-0029 §3.2 硬约束、§7-4 | 约束 |
| 秒域评估、秒域解码 | RFC-0029 §7-5、BasePlan §3.6 附注 | 约束（帧索引域评估一律禁止） |
| 合法性后处理四类红线 | BasePlan §3.6-2、RFC-0029 §5.1（合法性）、§2.4-5（跨线几何检查） | 直接落实 |
| 只校验/钳位、不改落点 | CLAUDE.md 红线 3、RFC-0029 §3.1 越界不钳位 | 约束 |
| 100% 合法方可导出 | CLAUDE.md 红线 6 | 约束 |
| 一切单位派生 | CLAUDE.md 红线 7、POSTMORTEM §7-1 | 约束 |
| 事件空间与标记 | BasePlan §3.2.2、RFC-0029 §2.3 | 输入/输出定义 |
| 多线解码需同时看到全部 K 条线 | RFC-0029 §2.4-4 | 设计约束（禁止分线独立解码成谱） |
| 跨线几何冲突 | RFC-0029 §2.4-5、BasePlan R-8 | 直接落实 |

**偏离声明**：无。本 plan 未引入任何偏离 BasePlan/RFC-0029 的决策。

## 3. 接口契约

> 跨模块类型**以 `beatmorph/core/contracts` 为准**（红线 2，由 contracts-agent 落定）。本节只声明本模块**需要**的字段与形状；若与 Plan 00 冲突，以 Plan 00 为准并回改本节。

### 3.1 输入：强度场（**按名引用 Plan 00 契约，不本地复制**）

> 契约权威是 [Plan 00](./00-core-contracts.md) §3.7（`ChartFieldSpec` / `ChartField`）与 §3.8 的不变量表。本节只列出本模块**消费**的字段，字段名与形状一律以 Plan 00 为准（红线 2；Plan 00 §5 明确 `ChartFieldSpec` / `ChartField` / `x_bin_index` 等符号「必须按名引用，不得本地复制」）。

```
ChartFieldSpec            # 唯一的网格元数据来源（不得由解码器自行推断）
  k                       # 判定线条数（运行期可变，不得写进任何输出层维度）
  tau_bins / d_tau        # **τ 轴（beat-aligned，Q15）**：T = τ 格数；d_tau = 1 / BEAT_SUBDIVISION 拍（派生）
  jacobian(τ)             # J(τ) = dt/dτ（秒/拍），由 BPMList 分段给出；∫λ 的测度是 J(τ)dτ（换算只在 field/ 内）
  x_bins / dx             # X = RPE_X_GRID_BINS；dx = RPE_STAGE_WIDTH / x_bins（派生）
  x_min / x_max           # = ∓RPE_STAGE_HALF_WIDTH（= RPE_STAGE_WIDTH / 2）
  sides / channels        # S = 2；C = len(NoteType) + 1（含 hold-end 通道）
  assert_grid()           # dx * x_bins == RPE_STAGE_WIDTH 且 d_tau * BEAT_SUBDIVISION == 1（拍）；失败即抛

ChartField                # 模型侧（推理时 batch = 1）
  lam    (batch, k, tau, x, s, c)  float32  # λ ≥ 0（softplus / exp 参数化）；第 3 轴是 τ（Q15）
  mask   (batch, k, tau, x, s, c)  bool     # 1 = 被遮盖 —— 迭代解码的输入（Plan 04）
  line_mask (batch, k) / time_mask (batch, tau)  bool  # padding 掩码，解码时必须跳过
```

- **✅ 字段名已与 Plan 00 同步（2026-08-05）**：Plan 00 §3.7 已改为 τ 口径。**契约字段名 = `t_bins`（τ 格数）/ `d_tau` / `bpm_points`（换算依据的数据）**；**换算函数 `jacobian(τ)` / `tau_to_seconds(τ)` 由 `beatmorph/field/` 提供**（契约只带数据、不实现换算，RFC-0029 §8.4 R-g）。本模块**不得**自行发明第二套字段名或第二套换算。
- **索引映射只有一条路径**：`x_bin_index(x, spec)`（Plan 00 §3.7）。解码器**禁止**自行 `floor` 或重采样——重采样等于引入第二个分辨率假设。
- **域外 λ 归零属生成侧**：Plan 00 §3.7 规定「生成阶段 `abs(x) > RPE_STAGE_HALF_WIDTH` 区域的 λ 由参数化保证为 0；解析阶段不作任何钳位」。本 plan 只依赖该约定，并在解码后**仍然统计**越界（防御性，不钳位）。
- **时间换算不得在解码器内重算**：任何「秒 ↔ τ」换算只允许经由 **`field/` 的权威接口**（`field.jacobian(τ)` / `field.tau_to_seconds(τ)`）；契约本身只带 `spec.t_bins` / `spec.d_tau` / `spec.bpm_points`（**数据**，不含函数）。**秒↔τ 的实现只在 `field/` 内**（红线 7、RFC-0029 §7-8）。τ 格索引与外部接口一律以**秒**表达（POSTMORTEM §7-1）。

### 3.2 解码产物：直接构造契约对象（**不新增跨模块类型**）

- **对外输出即 `PhigrosChart`**（Plan 00 §3.6：`lines` / `notes` / `bpm_points` / `meta` / `source`）。Plan 06 的评估与 Plan 08 的写出都消费它，因此本模块**不需要**引入新的跨模块契约类型（红线 2：新增跨模块类型前先开 RFC）。
- **`DecodedEvent` 是本模块内部中间态**（`beatmorph/decoder/` 内，不跨模块）：

```
DecodedEvent:            # decoder 内部；构造 PhigrosNote / PhigrosChart 后即丢弃
  t_s          : float   # 判定时刻，**秒**（写回 RPE 时才转 beat 三元组）
  line_id      : int     # 0 <= line_id < k
  position_x   : float   # RPE 舞台系 x 坐标单位（字段名对齐 Plan 00）
  side         : Side    # 由 side_from_above(above_raw) 得到，禁止本地复制映射
  note_type    : NoteType
  hold_time_s  : float | None   # 非 HOLD 的 endTime == startTime ⟹ hold_time == 0（格式事实）
  is_fake      : bool
  confidence   : float   # 供 Plan 04 的迭代重掩码使用（定义见 §4.1.5）
```

- **字段命名与语义严格对齐 Plan 00**：`position_x`、`side`（`Side` 枚举，禁 truthiness 分支）、`note_type`（`NoteType` 枚举）；原始值经 `*_raw` 无损往返（Plan 00 §4）。
- **公共接口不出现帧索引**：`t_s` 是唯一时间字段（Plan 00 §3.8 I11、RFC-0029 §7-5）。

### 3.3 输出：合法性报告 + RPEJSON

```
LegalityReport:
  violations   : Violation[]   # 空列表 == 完全合法（红线 6 的判据）
  stats        : dict          # 越界计数、同刻按键数分布、Hold 时长分布、跨线冲突计数
  edits        : Edit[]        # **每一次钳位/丢弃都必须留痕**（改了什么、原值、新值、理由）
```

- **可审计性**是本契约的硬要求：后处理不得静默改动任何事件；`edits` 为空时导出结果必须与解码输出**逐字段一致**（可写成测试断言）。
- **类型归属待裁**：`LegalityReport` / `Violation` / `Edit` 会被 Plan 06（合法性指标）与 Plan 08（`report/legality.json`）消费，属跨模块数据 → 按红线 2 应落 `core/contracts`；也可能被界定为「模块公开 API 的返回类型」。**实现前必须有结论**（见 §9-12）。
- writer 产出 RPEJSON 根对象（`BPMList` / `META` / `judgeLineList` / `chartTime` 等），字段级规范见 `phigros-format.md` §3-§5。

## 4. 内部设计

### 4.1 D1：峰值检测 + 阈值（v0 基线）

1. **同网格**：峰检测直接在 `λ` 的 `(T, X)` 平面上进行（`T` = **τ 格数**，Q15），不再重采样（重采样即引入第二个分辨率假设）。
2. **峰显著性**：`(line, side, type)` 通道上做非极大抑制（NMS 半径由 `dx` 与 `d_tau` 派生、不写魔数；若需以秒表达则由 `J(τ)` 折算，折算经 `field/`）。
3. **阈值不是魔数**：阈值以 **G3 常数基线**的强度标度 `λ0 = N / |Ω|`（RFC-0029 §3.2）为单位表达，即 `threshold = α · λ0`；`α` 作为配置项并在报告中声明。理由：DDC 的 F1 在阈值 0.5 与最优阈值间差 0.23（文献库 §2.2-1，0.5006 → 0.7317），阈值是必须显式声明的自由度。
4. **短距双峰抑制**：采用 DDC 的 Hamming 窗平滑（文献库 §2.2-1），窗宽以秒表达。
5. **置信度定义（本 plan 新增设计）**：对连续场没有天然的 `[MASK]` token，掩码补全迭代解码所需的「置信度」需自建。建议 `confidence = 峰值高度 / (λ0 · α)` 的单调变换，并在 Plan 04 中固定语义。文献无对应做法（文献库 §5.2 明确 MaskGIT 的置信度定义不可直接搬），故此项进 §9。

### 4.2 D2：Ogata (1981) thinning（原则性解码器）

1. **算法**：对每条线 k 的每一 `(side, type)` 通道，把场压成 **τ 轴**强度 `λ_k(τ) = Σ_{x,s,c} lam[k,τ,x,s,c] · spec.dx`（**与 `∫λ` 同网格、同 Δx、同 `J(τ)dτ` 测度**，量纲 = 计数/拍），再对上界 `λ_max` 做 thinning 采样；采样得到 τ 后**由 `field/` 的权威换算**回到秒（`t = ∫_0^τ J`）——本模块**不实现**该 BPM 分段积分（红线 7）。
2. **上界必须保守**：`λ_max` 取分块上界（分块内取最大值 × 安全系数），并实现**失败重采样**以保证算法正确性（thinning 的正确性依赖 `λ_max ≥ λ(t)` 对所有 t 成立）；不得用「全局最大值」以外的近似而不做校验。
3. **K 条线各自独立采样**：训练时 NLL 是 K 个场的求和（RFC-0029 §2.4-6），故各线场互为**竞争**而非归一化分布；解码同样对各线独立 thinning，同刻多线多事件是点过程天然允许的。
4. **标记的采样**：在采到时刻 `t` 后，按 `(x, side, type)` 的**归一化强度**采样标记（条件分布），需声明是「联合采样」还是「逐轴分解采样」——两者语义不同，见 §9。
5. **随机性**：thinning 是随机算法 → 必须固定 seed、报告多 seed 方差；评估（Plan 06）必须区分「解码随机性」与「模型质量」。
6. **验收金标准**：以解析可算的非齐次泊松过程（如 `λ(t) = a + b·t`）为参照，用 KS 检验比对采样分布（§6 M5.3）。

### 4.3 合法性后处理（只校验与钳位）

| 检查项 | 处置 | 依据 |
| --- | --- | --- |
| `position_x` 越界（`abs(x) > RPE_STAGE_HALF_WIDTH`） | **只统计**，写 `stats`，**不钳位、不丢弃**；与契约侧 `PhigrosChart.out_of_visible_range()` 的计数必须一致 | 红线 3；Plan 00 §3.8 I3；RFC-0029 §3.1 |
| 同刻按键上限 | **上限数值未查证** → 先只统计分布并报告；数值裁定后才可作红线（§9） | BasePlan §3.6-2 要求该项存在，但未给数值 |
| Hold 区间合法性 | `endTime >= startTime`（RPE 中非 Hold 的 `endTime == startTime`，`phigros-format.md` §5.1）；违反者**丢弃并留痕** | 格式 A 级字段语义 |
| Hold 期间判定线速度变化 | **RPE 下未确证是硬约束**（原文只写「PEC 和官谱 JSON 中」）→ 默认**只报告为 warning** | `phigros-format.md` §7.4（原文限定 PEC/官谱） |
| 跨线几何冲突 | 把 `(positionX, side)` 经线事件轨变换到**舞台系**，检测两条线的线段相交/夹角过小 → **报告计数**，不移动 note | RFC-0029 §2.4-5、R-8；判据阈值见 §9 |
| 同一线同刻重复事件 | 去重（完全同 `(t, x, side, type)`）并留痕 | 点过程一次实现不应产生重复事件 |

**跨线几何的坐标系**：必须先按 `phigros-units-and-geometry.md` §2.6 的 `p_stage = Σ_layers(R(rotate)·p_local + move) + p_stage(father)` 把线变换到舞台系（含 `father` 嵌套，实测 26% 谱面有父线），再算几何关系。**不得**在屏幕像素系计算（像素系依赖 `aspectRatio`，与谱面语义无关，单位文档 §5.2）。

### 4.4 RPEJSON 写路径

1. **秒 → beat 三元组**：按 `BPMList` 分段反解（`phigros-format.md` §7.1 的 `sec2beat` 逆函数）；三元组 `[i, n, d]` 表示 `i + n/d` 拍。**该分段反解必须复用 `field/` 的权威换算实现**（红线 7、RFC-0029 §7-8：不得在 writer 内再写一套 BPM 积分；接缝归属见 §9-14）。分母 `d` 的选择是**量化决策**，必须集中在一处并记录量化误差（§9）。
2. **`above` 语义**：一律经契约函数 `side_from_above()` 往返（Plan 00 §3.8 I4：`side_from_above(1) is FRONT`、`(0)` 与 `(2) is BACK`），**禁止**在后处理/writer 内重写映射或把 `above` 当布尔解析（RFC-0029 §8.1、格式文档 §5.3）。
3. **`type` 数字**：经契约函数 `note_type_from_rpe()` / `note_type_from_official()` 分派（Plan 00 §3.8 I6：`note_type_from_rpe(2) is HOLD`，而 `note_type_from_official(2) is DRAG`）。**两套映射不得共用**，也不得在 writer 内本地复制（格式文档 §5.2，A 级源码确证）。
4. **`notes` 归属**：note 写在所属判定线的 `notes` 数组内（`judgeLineList[k].notes`）；`notesAbove`/`notesBelow` 是**官谱**的双数组表示，RPE 用单数组 + `above`。
5. **`META`**：`song`/`background` 引用文件名而非拷贝媒体（红线 5）；`offset` 单位为**毫秒**且符号语义在文档中自相缠绕（格式文档 §7.2），实现须固定一种解释并记录（§9）。
6. **导出前门禁**：`LegalityReport.violations` 非空则**拒绝写出**（红线 6），CLI 退出码非 0（Plan 08）。

## 5. 依赖关系

| 方向 | 模块 | 内容 |
| --- | --- | --- |
| 上游 | Plan 00 `core/contracts` | `ChartFieldSpec` / `ChartField` / `PhigrosChart` / `side_from_above` / `note_type_from_rpe` / `x_bin_index` / 派生常量（按名引用，不本地复制） |
| 上游 | Plan 02 `data/`、`io/formats/rpejson/`（读路径） | RPEJSON **读**与 schema、`chart/1000` 与 `chart/7039` 夹具 |
| 上游 | Plan 03 `field/` | 网格与测度（契约见 Plan 00 §3.7）、**秒↔τ 的唯一换算（`d_tau` / `J(τ)`，Q15）**、`λ0 = N/abs(Ω)` 常数基线、两条积分路径的一致性门禁 |
| 上游 | Plan 04 `generation/` | 强度场输出、掩码迭代解码主循环（消费本模块的 `confidence`） |
| 下游 | Plan 06 `eval/` | 秒域 `DecodedEvent` 与合法性指标（合法率、越界占比、跨线冲突数） |
| 下游 | Plan 08 `cli/` | 端到端入口的最后一跳与退出码 |
| 外部库 | `numpy`（已在依赖） | 向量化实现 |
| 外部库 | `scipy.signal.find_peaks` | **当前不在 `pyproject.toml` 依赖中**（实读确认）→ 引入需 infra-agent 加依赖，或**自实现峰值检测**以避免新依赖（§9） |

> **文件归属协调项**：AGENTS.md v3.0 把 `beatmorph/io/formats/rpejson/` 划给 data-agent（Plan 02），本 plan 需要其**写路径**。建议同一 schema 模块内「读归 02、写归 05」，共用字段常量；实现前须与 data-agent 对齐文件边界（§9）。

## 6. 里程碑与验收

> **门禁硬性要求**：本 plan 不新增训练目标/损失。**若**在任一里程碑引入可学习解码器或任何新损失（含「置信度头」监督），必须先通过 **G1-G4**（`beatmorph/infra/sanity.py`）并把 `summarize()` 输出写入训练日志，**门禁未绿不得扩大数据规模**（BasePlan §9、CLAUDE.md §5.8）。

| # | 里程碑 | 可量化验收 |
| --- | --- | --- |
| **M5.1** | 网格与契约断言 | 直接复用 `ChartFieldSpec.assert_grid()` 与 Plan 00 §3.8 的 I1/I2（`dx * x_bins == RPE_STAGE_WIDTH`；`d_tau * BEAT_SUBDIVISION == 1`（拍，Q15）；`x_min/x_max == ∓RPE_STAGE_HALF_WIDTH`）——**本 plan 不新增第二套网格断言**；「场网格 ↔ 秒」往返无损契约测试归 Plan 03 M12，本模块只消费。测试**不依赖权重/GPU**，进默认 CI |
| **M5.2** | D1 峰值解码 | 在**合成场**（已知事件的窄高斯叠加强度场）上：`±20ms` 与 `±50ms` 的 timing-F1 **均 = 1.000**，`positionX` MAE ≤ `dx/2`（量化上界）。失败即通路坏 |
| **M5.3** | D2 thinning 正确性 | 对解析可算的 `λ(t) = a + b·t`：KS 检验 `p > 0.05`（n ≥ 10^4 次采样），且采样总数相对 `∫λ` 的相对误差 < 1%；seed 固定后可复现（同 seed 两次运行逐元素一致） |
| **M5.4** | 后处理红线正确性 | 夹具（`chart/1000` 标准 RPE + `chart/7039` 伪装后缀 PEC）零违规；注入式合成非法谱（越界 / 反向 Hold / 重复事件）检出率 **100%**；**`positionX` 钳位次数恒为 0**（断言）；`edits` 可完整重建「后处理前」状态 |
| **M5.5** | RPEJSON 写路径往返 | 与 Plan 02 reader 联合：`read(write(chart))` 在**秒域**上 `max abs(Δt) ≤ 一个 beat 量化步长`、标记（`line_id/positionX/side/type/is_fake`）**全等**；写入→读回→再写入字节级稳定（幂等） |
| **M5.6** | 越界只统计的端到端证据 | 在含越界 note 的夹具上跑完整管线：输出文件中越界 note **仍然存在**，且 `stats.out_of_range` 计数与输入一致（防回归到「偷偷钳位」） |
| **M5.7** | 双解码臂对照（B6）出数 | 同一模型同一谱面下 D1/D2 各跑 ≥ 5 个 seed，报告 timing-F1 双容差 + 事件数分布；结论必须写明「差异来自解码而非模型」（RFC-0029 §5.2 B6 的对照层级要求） |

## 7. 风险与缓解

| 风险 | 来源 | 本模块的缓解 |
| --- | --- | --- |
| **R-6** 连续场 → 离散事件的解码精度 | BasePlan §6 | 双解码臂（D1/D2）+ 秒域评估；合成场解析验收（M5.2/M5.3）把「解码误差」与「模型误差」分离 |
| **R-8** 跨线几何冲突不可玩 | BasePlan §6 | §4.3 跨线几何检查（只报告不改创意），输出冲突计数进 `stats`，由 Plan 06 报合法率 |
| **R-7** 物理常量/单位漂移 | BasePlan §6 | 一切网格量派生 + 契约断言（M5.1）；**秒↔τ 只在 `field/` 内换算**（本模块不实现第二套）；公共接口无帧索引、也无 τ 格索引字段（对外只出秒）；`find_peaks` 的窗口参数以秒表达 |
| **R-5** 多线 K 长尾 / note 极度集中 | BasePlan §6 | 解码**逐线独立**、不做「均匀分配」假设；同刻多事件按点过程语义保留；按线分解的统计进 `stats` |
| **R-3** 社区谱面质量参差 | BasePlan §6 | 后处理不美化、不改落点；把「谱面怪异」暴露给评估与人评，而不是在解码端抹平 |
| 后处理越权（红线 3 违例） | CLAUDE.md 红线 3 | `edits` 强制留痕 + 「`positionX` 钳位次数恒为 0」断言（M5.4）+ 越界端到端回归（M5.6） |
| 阈值解码超参敏感 | 文献库 §2.2-1（DDC 0.5006 → 0.7317） | 阈值以 `λ0 = N/abs(Ω)` 为单位表达并显式声明；Plan 06 强制「固定规则」与「每谱最优」两栏 |
| thinning 采样随机性被误读为质量差异 | 本 plan §4.2-5 | 固定 seed + 多 seed 方差报告；评估协议区分解码随机性与模型质量 |

## 8. 测试策略

**单元（默认 CI，无权重、无 GPU）**

- 网格契约：派生式与断言（M5.1）；`dx * x_bins == RPE_STAGE_WIDTH`；越界边界值 `±RPE_STAGE_WIDTH/2` 的**半开/闭区间**语义显式测试；
- D1：合成场上的峰位、NMS 行为、阈值缩放（`α·λ0`）；
- D2：thinning 的**性质测试**——常数场下事件数服从泊松（用固定 seed 的统计量断言）、`λ_max` 上界失效时必须报错而不是静默出错；
- 后处理：表驱动用例（每一行 = 一个检查项 × 合法/非法）；`edits` 留痕完整性；
- writer：`type` 数字映射（RPE 表）、`above` 三值（0/1/2）语义、beat 三元组反解、幂等性；
- **mock 纪律**：夹具/mock **不得固化物理常量**（AGENTS.md §3.3、RFC-0029 §7-6）。凡需要帧数/坐标，必须引用契约常量或 `FieldGrid` 派生量。

**集成**：`field → decode → postprocess → write` 全链路（小规模，可在 CI 跑）；含越界输入的回归用例（M5.6）。

**e2e（标记 `slow` / `gpu`）**：真实模型 + 真实音频 → 可玩 RPEJSON（与 Plan 08 M8.2 联合验收）。

**不做的事**：不在本模块单测里断言 F1 的绝对水平（那属于 Plan 06 与模型质量，会随模型变化而失效）。

## 9. 开放问题

1. **同刻按键上限的具体数值未查证**。BasePlan §3.6-2 与 RFC-0029 §5.1 都要求该项检查存在，但两份文献都**没给数值**。裁定前该项**只统计不阻断**。
2. **跨线几何冲突的判据阈值未定义**（两线最近距离 / 夹角 / 是否计入 alpha=0 的隐藏线）。需先做数据统计再定；在此之前只报告计数，不做任何自动处置。
3. **RPE 下「Hold 期间判定线速度变化」是否硬约束未确证**：格式文档原文限定「PEC 和官谱 JSON 中」（§7.4）。当前按 warning 处理，需 RFC 或数据统计裁定。
4. **标记采样的分解方式**：`(x, side, type)` 是联合归一化后一次采样，还是逐轴条件采样后组合？两者产生不同的边缘分布；需在 Plan 04 的迭代解码定稿时一并决定。
5. **网格 X（桶数 N）的最终值**：`N ∈ {64,128,256,512}` 消融与「同线同刻同侧最小 `|ΔpositionX|`」共格碰撞统计**尚未做**（单位文档 §7.3 明确要求先统计再定 N）。本 plan 的解码器必须对 N 参数化，不得假定 128。
6. **`scipy` 依赖**：`scipy` 当前不在 `pyproject.toml`；是否引入（或用 `numpy` 自实现峰值检测）待定。
7. **beat 三元组的分母策略**：秒 → beat 的量化误差上界尚未推导。若分母取固定值（如 480 分音符细分）会产生与 BPM 无关的误差；若按 BPM 自适应则体积膨胀。须给出误差上界公式并写入契约断言。
8. **`META.offset` 的符号解释**在格式文档中自相缠绕（§7.2），且 prpr 未在该处使用它。生成侧应写 0 还是解析输入音频的偏移，未定。
9. **`beatmorph/io/formats/rpejson/` 的读写归属**：AGENTS.md 划给 data-agent，本 plan 需要写路径。属于协作边界而非技术未决，需在实现前排期确认。
10. **`is_fake` 是否作为生成目标**：RFC-0029 §2.3 把 `is_fake` 列入标记，但 BasePlan §3.2.2 未说明生成侧是否预测它。假音符不计物量（格式文档 §5.4），其占比与语义价值**未统计**。
11. **与 Plan 00 的一处实质分歧（需裁定）**：Plan 00 §3.5 的注记把「Hold 期间判定线不得发生速度变化」声明为**格式硬合法性约束**；但依据 `phigros-format.md` §7.4，该约束的原文限定为「**PEC 和官谱 JSON 中**」，**RPE 侧未确证**。本 plan 暂按 **warning** 处理（§4.3）。若裁定为硬约束，则该项在 §4.3 的处置须由 warning 升级为 `violations`（会改变退出码 3 的触发面）。建议由 contracts-agent 与决策者统一口径。
12. **`LegalityReport` / `Violation` / `Edit` 的类型归属**：这三个类型被 Plan 06 与 Plan 08 消费，属跨模块数据，按红线 2 应落 `core/contracts`；也可界定为「模块公开 API 返回类型」而非跨模块数据契约。**实现前必须裁决**，否则 Plan 08 的 `report/legality.json` schema 没有权威来源。
13. **编号一致性（跨文档）**：Plan 00 §3.5/§5 两处以「plan 07」指代后处理与导出，与本索引（后处理 = plan 05、infra = plan 07）不一致。本 plan 按 AGENTS.md v3.0 与 `docs/plans/README.md` 的编号（05）为准，建议 Plan 00 同步修正。属文档一致性问题，非技术未决。
14. ~~**Plan 00 §3.7 的 `ChartFieldSpec` 时间轴口径待同步（Q15 遗留）**~~ ✅ **已解决（2026-08-05）**：Plan 00 §3.7 已改为 τ 口径（`t_bins` / `d_tau` / `bpm_points`），并新增不变量 I13（τ→秒→τ 往返 ≤1e-9，须覆盖多 BPM 段）。字段名以 Plan 00 为准，换算函数由 `field/` 提供（RFC-0029 §8.4 R-g）。
