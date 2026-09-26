# RFC-0030 — 解码/导出契约归属与 plan 05 的六项实现口径

- 状态：提案 ｜ 提出日期：2026-09-27 ｜ 决定日期：（待裁定）
- 提出者：解码与后处理组（plan 05）
- 影响模块：plan 00（`beatmorph/core/contracts/`）、plan 05（`beatmorph/decoder/`、`beatmorph/io/formats/rpejson/` 写侧）、
  plan 06（`eval/` 消费合法性指标）、plan 08（`cli/` 的 `report/legality.json` 与退出码）

## 背景

plan 05 在动工前有 **§9-12 明确要求"实现前必须有结论"** 的一项：
`LegalityReport` / `Violation` / `Edit` 属于跨模块数据（plan 06 报合法率、plan 08 落盘
`report/legality.json`）。**归属方向已有裁决**——[RFC-0029 §8.4 R-c](RFC-0029-phigros-continuous-chart-generation.md)
已判"属跨模块类型 → 落 `core/contracts`"，但**具体 schema 与报告语义仍未定**，
plan 08 的 `report/legality.json` 因此没有权威来源。本 RFC 只补这一步（不重新决定归属）。

同时，plan 05 §9 里另有若干条只能由实现来定口径的问题（分母策略、依赖取舍、
配对语义），以及一处**算式笔误**（§4.2-1 漏了 `J(τ)`）。本 RFC 一并登记，避免它们
以"实现细节"的形式悄悄固化。

## 提议

### 1. 新增 `beatmorph/core/contracts/legality.py`（类型归属）

- `ViolationKind`：**会阻断导出**的四类——`LINE_INDEX_OUT_OF_RANGE` / `HOLD_REVERSED` /
  `DUPLICATE_EVENT` / `SAME_INSTANT_OVER_LIMIT`（最后一项默认关闭，见第 4 条）；
- `EditKind` + `Edit`：后处理留痕（`kind/note_index/field/before/after/reason/resolved`）；
- `LegalityReport`：`violations` + `edits` + `stats` + `criterion`，**`violations` 为空 == 完全合法**，
  这是红线 6「100% 合法方可导出」的**唯一**判据（writer 的门禁只认它）；
- `assert_no_position_clamp(report)`：红线 3 的机器可验证断言。

**关键语义**：报告**只描述它伴随的那张谱面**。因此 `postprocess_chart` 返回的
`report.violations` 恒为空（修复后），修复前的发现在 `findings` 里、
修复动作在 `edits`（每条带 `resolved` 指回它消解的违规种类）。
否则"修好了却仍然不可导出"，红线 6 会变成死锁。

### 2. 两处契约补充（同一理由：禁止各模块自行重写映射）

- `above_from_side(Side) -> int`：`above` 的**规范代表值**（FRONT → 1，BACK → 0）。
  `side_from_above` 是多对一的（0 与 2 都是背面），导出侧必须选一个代表值；
- `side_from_index(int) -> Side` + `SIDE_ORDER`：`side_index` 的严格逆（解码把通道还原成侧别）。

**已避免的做法**：在解码器里写 `Side.FRONT if s == 0 else Side.BACK`、在 writer 里写
`1 if side is Side.FRONT else 0` —— 那正是红线 7 要挡的"同一映射两处定义"。

### 3. 秒 → beat 三元组的分母策略（plan 05 §9-7）

**分母固定为 `SUBDIVISIONS_PER_BEAT`（= τ 网格的基本格，1/48 拍）**，量化只发生在 writer
的 `beat_from_tau` 一处，**误差上界 = `TAU_GRID_DT / 2` 拍**（半格）。

- 不取更大分母（如 480 分音符）：**网格之外的精度是假精度**——模型的时间分辨率就是 1/48 拍，
  写更细的分母只是把浮点噪声编码进文件；
- 不按 BPM 自适应分母：会话内体积膨胀，且误差上界在**拍**域已经是常数（这正是 beat-aligned 的好处）；
- 秒 ↔ τ 的换算经 `field/` 的权威接口（红线 7）：writer 内**不得**再写一套 BPM 分段积分。

### 4. 三条"未查证项"的地位不变，但都变成**显式开关**

三项的地位**已由决策者裁定过**（分别是 RFC-0029 §8.4 的 R-e / 无 / R-b），本 RFC **不改变**它们，
只是把"只统计"落成代码里**一处可见的开关**，而不是散在各处的隐式约定：

| 项 | 出处 | 默认 | 开关 |
|----|------|------|------|
| 同刻按键上限数值未查证 | plan 05 §9-1 / BasePlan §3.6-2 | **只统计分布**，不作红线 | `LegalityConfig.same_instant_limit`（`None` = 关闭） |
| 跨线几何冲突判据阈值未定义 | plan 05 §9-2 | **只报告计数**，不移动任何 note | `LegalityConfig.cross_line_tolerance_bins`（默认 `1.0 * dx`，以网格分辨率为单位） |
| RPE 下 Hold 期间线速度变化是否硬约束未确证 | plan 05 §9-3 / §9-11 / 格式文档 §7.4 | **只报 warning 计数** | `LegalityConfig.check_hold_line_speed` |

判据文本随报告落盘（`LegalityReport.criterion`）：数字必须有出处，而不是散在代码里。

⚠️ **`λ_0 = N / |Omega|` 里的 `|Omega|` 采用 plan 03 的 Q15 测度**（`|Omega| = Σ_j dV_j`，
`dV_j = J_j · d_tau · dx`，见 plan 03 §2 偏离 2），而不是 RFC-0029 §8.4 R-a 里那个
**beat-aligned 之前的无量纲形式** `K·T·X·S·C`。Q15 之后测度随 BPM 变化，两者只在常速下相等；
阈值标度必须与训练用的 NLL 用**同一个** `|Omega|`，否则阈值与损失不在同一把尺子上。

### 5. 不引入 `scipy`（plan 05 §9-6）

峰值检测（**局部极大 + 阈值 + 贪心 NMS**）与 KS 检验（渐近 Kolmogorov 分布）都用
numpy / 标准库自实现。理由：两处实现都短（合计 < 60 行），而 `scipy` 是本项目目前
唯一不需要的重依赖；更重要的是，**门禁依赖的检验函数越少、越可审计**。

### 6. D2 的 τ 边缘强度**必须含 Jacobian**（§4.2-1 的算式笔误）

plan 05 §4.2-1 写作 `lambda_k(tau) = sum_{x,s,c} lam * dx`（"量纲 = 计数/拍"），
但正确式是

    lambda_k(tau_t) = J(tau_t) * dx * sum_{x,s,c} lambda[k, t, x, s, c]        [计数/拍]

漏掉 `J` 会让 `int lambda_k dtau` 与路径 (a) 的 `sum lambda dV` **相差一个与 BPM 有关的因子**
（= 把"拍"当"秒"用，与 POSTMORTEM 同类）。plan 原文自己写的"**同 J(τ)dτ 测度**"与
"量纲 = 计数/拍"两点只有加上 J 才同时成立，故本 RFC 判定为**算式笔误的修正**，
不构成偏离 BasePlan。

### 7. Hold 配对语义（plan 05 §3.2 未写，实现必须定）

目标场把 Hold 拆成 hold / hold_end 两个通道并放在**同一 `(k, i_x, s)` 纤维**
（plan 03 §2 偏离 3），因此解码侧的配对就是该编码的逆：

- 同一 `(k, i_x, s)` 纤维内，起点配**其后最近的**未占用终点；
- 配不到终点的起点：`hold_time = 0` **保留**（格式允许 `endTime == startTime`）并计数；
- 配不到起点的终点：丢弃并计数（它不产生音符）；
- 配对结果与计数全部进报告（`hold_unpaired_starts` / `hold_orphan_ends` / `hold_zero_length`）。

### 8. M5.3 的"采样总数相对 ∫λ 的相对误差 < 1%"口径细化

单次抽样的计数是 `Poisson(Lambda)`，相对偏差本身就是 `1/sqrt(Lambda)` 量级——
在 `Lambda = 1e4` 时恰好 1%，即"单次落在 1% 内"只有约 68% 概率，作为断言会**必然误报**。
实现改为三条统计上成立的断言：① KS 检验 `p > 0.05`（n >= 1e4）；
② 多个 seed 的**均值**与 `int lambda` 的相对偏差 < 1%（无系统性偏置）；
③ 单次偏差 <= `4 * sqrt(Lambda)`（约 4σ）。

### 9. D1 的两个必须做对的细节

- **先局部极大再 NMS**：只做贪心 NMS 会让每个高斯峰的**侧翼**每隔一个抑制半径冒出一个假峰
  （实测 4 个真事件解出 12 个峰）；局部极大用 `>=` 而非 `>`，否则常数场（G3 基线形态）无峰；
- **Hamming 窗长强制奇数**：偶数长度的卷积核没有中心格，会把峰整体挪半格（系统性时间偏移，
  实测 1 格误差 = 在 180 BPM 下 6.9 ms）。

## 备选方案

1. **`LegalityReport` 留在 `beatmorph/decoder/`（模块 API 返回类型）**：驳回。plan 06/08
   跨模块消费它，红线 2 的判据是"是否跨模块"，不是"是否在 contracts 目录里舒服"。
2. **分母取 480 / 960 分音符**：驳回。见第 3 条——把网格之外的浮点噪声写进文件，
   还会让"量化步长"与"模型分辨率"这两个概念再也对不上。
3. **引入 `scipy`（`find_peaks` + `kstest`）**：驳回。见第 5 条；且 `scipy.signal.find_peaks`
   的默认语义（min distance / prominence）并不比自实现更贴合本项目的 NMS 半径=格数的口径。
4. **把"修复前的违规"留在 `report.violations` 里**：驳回。那会让后处理永远无法产出可导出的谱面，
   红线 6 变成死锁；改为 `findings` + `edits[].resolved` 双留痕。
5. **修 `JudgeLine.pose_at` 的父线合成**：见"后果"第 2 条，本 RFC **不擅自改契约语义**，留待裁定。

## 后果

1. **plan 05 的 §9-12 / §9-6 / §9-7 有结论**；§9-1 / §9-2 / §9-3 的地位不变（仍为"未查证 → 只统计"），
   但都成了**一处可配置的开关**而不是散落的硬编码。
2. ⚠️ **发现（需裁定，本次未改）**：`core/contracts/phigros.py::JudgeLine.pose_at` 的父线位置合成
   是"**跨链求和 move_x/move_y**"，而 A 级证据（[phigros-units-and-geometry.md §2.6](knowledges/phigros-units-and-geometry.md)，
   prpr `core/line.rs`）给出的式子是 `parent_pos + R(parent_rot) · child_translation` ——
   **父线的旋转应当作用在子线的平移上**。实测 26% 的谱面含嵌套线，因此"跨线几何冲突"的计数
   在含父线的谱面上是**近似值**。由于该项只报告计数、从不改动谱面（红线 3），本轮如实记录、
   不改契约；修正应走 contracts-agent（会改变 `pose_at` / `local_to_stage` 的语义）。
3. 新增两个契约函数（`above_from_side` / `side_from_index`）与一个契约模块（`legality.py`），
   契约测试相应扩展；plan 00 §5 的"跨 plan 共享符号"清单需同步。
4. `io/formats/rpejson/` 的**写入侧落点**确认为该包（读侧仍在 `beatmorph/data/parsers/rpejson.py`，
   plan 02 主责）；M5.5 的往返测试钉住两者一致（plan 05 §9-9 的协作边界由此闭环）。
5. 已知信息损失（读路径侧，非本 RFC 范围）：`META.id` / `META.illustration` / 根级 `judgeLineGroup`
   等字段不在 IR 中，故"读入 → 写出"会丢；如需无损归档应扩 `ChartMeta`（另开 RFC）。

## 关联

- [RFC-0029](RFC-0029-phigros-continuous-chart-generation.md) §8.4 **R-b**（Hold 变速 → warning）/ **R-c**（`LegalityReport` 归属 → `core/contracts`）/ **R-e**（同刻上限 → 只统计）/ **R-f**（N=128 默认 + 必做消融）——本 RFC 是这四条的执行，不是重新裁定
- [plan 05](../plans/05-decoder-postprocess.md) §4.2/§4.3/§4.4、§9-1/§9-2/§9-3/§9-6/§9-7/§9-9/§9-12
- [plan 00](../plans/00-core-contracts.md) §3.7/§3.8（不变量 I3/I11）
- [RFC-0029](RFC-0029-phigros-continuous-chart-generation.md) §3.1/§3.4/§7-5/§7-8（秒域、同网格、换算唯一出口）
- [phigros-format.md](../knowledges/phigros-format.md) §5.1-§5.3、§7.1/§7.2/§7.4
- [phigros-units-and-geometry.md](../knowledges/phigros-units-and-geometry.md) §2.6、§7.3/§7.5
- [POSTMORTEM-2026-08-05](POSTMORTEM-2026-08-05-frame-rate-misalignment.md)（拍/秒混淆的代价）
