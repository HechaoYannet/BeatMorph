# RFC-0037 — 删除门禁 G2 + 训练损失按事件归一化（per-event）与附带指标

- 状态：**已裁定（采纳，2026-09-28 决策者）**
- 提出者：主会话（训练基础设施 / 生成主干）
- 取代：[RFC-0036](RFC-0036-gate-batch-caliber-and-g2-power.md) → **废弃**（其 §1–§2 诊断记录
  保留为本案的证据基础；P0 由本案 §3.3 吸收、P1 由 §3.4 吸收；A/B/C/D 选项随 G2 删除失去意义）
- 影响模块：`infra/sanity.py`、`infra/gates.py`、`infra/train_loop.py`、`infra/smoke.py`、
  `generation/losses.py`、`generation/model.py`、`infra/config/schema.py`、`configs/*.yaml`；
  文档：BasePlan §7/§9、CLAUDE.md 红线 7、AGENTS.md §4、docs/TRAINING.md、plan 04 §3.3/§4.3、plan 07 §4.3。

## 0. 裁定记录（决策者原话要点，2026-09-28）

> 「直接去掉G2，BasePlan中对G2的设计是一个错误，这一条直接贯彻。」
> 「loss如果不对mask平均，从自然理解上就不对路，训练多半会不稳，20000step训练中loss曲线中平滑值的浮动就是证明。
> 你需要做的下一步不是分析loss，而是设计一组更合理的loss和附带指标。」
> 「你的最终交付是一个RFC（包括G2删除、loss更改），直接标记为已裁定。然后写好交接件，准备训练。」

## 1. 裁定内容

- **R1 删除 G2（打乱标签对照）**：门禁从 G1-G4 变为 **G1 / G3 / G4 三道**，全链路删除
  （判据函数、装配、配置字段、合成夹具的打乱臂、阈值落盘）。**修宪级**：BasePlan §9 门禁表、
  CLAUDE.md 红线 7、AGENTS.md §4 的「G1-G4」文字同步改为三道门禁。
  - 依据（详证见 RFC-0036 §1–§2、plan 07 §9-54/§9-55）：现行 G2 = 单批、100 步、**未归一化 sum
    损失**上的「拟合速度赛」，判据量级 ∝ E^0.93 而余量只有 5%；真实批上判别力未被证明且反证充分
    （fp32 逐批 4 FAIL / 1 PASS；修掉 P1 nuisance 后仍 5/6；同批输入干预自相矛盾；跨进程翻转）；
    唯一正面证据是合成解耦夹具（45% 差距）——只证明「能报警」，不证明真实批 verdict 携带信息。
  - 「输入对目标有没有信息」这一命题改由**已落地的在线仪器**承担：
    `val/nll_shuffled_delta`（held-out 128 窗、遮盖内置换、线数归一、`no_grad`）+
    `val/ratio`（对同测度常数场的技能分，主判据）+ `cond_*_delta`（条件干预三元组，§9-46③）+
    G3 + G4。它们是**运行中监测**而非运行前门禁。
  - **判读红线（写入 docs/TRAINING.md）**：任一 val 点出现 `val/ratio >= 1` 或
    `val/nll_shuffled_delta <= 0` ⇒ 该 run 的「扩大数据规模」结论**作废**（fail-closed 的精神保留，
    只是从「训前单批赛跑」移到「训中 held-out 对照」）。
- **R2 训练损失按事件归一**：
  `L_train = [ Σ_{有效线} ( (1/r)·Σ_{被遮盖格} −n·log λ + Σ_{全域} λ·dV ) ] / D`，
  `D = max(E_total, 1)`，`E_total` = 该步批内**有效线**上的事件总数（`counts` 按 `line_mask` 求和）。
  - **硬约束（防静默失效，推导见 §2.2）**：必须**整式**除以 D。**只除事件项会改变最优强度**
    （λ* 整体缩小 D 倍）——这与 `1/r` vs `1/(1-r)` 之争是同类陷阱。整式缩放 ⇒ **逐窗口 argmin
    逐位不变**（泊松语义不动），改变的只有步间相对权重与量级可读性。
  - 空窗（E=0）：D=1 ⇒ loss = 积分项，与现行行为一致（继续压 λ→0）。
  - `reduction` 枚举新增 `"per_event"`；`masked_poisson_loss` / `full_poisson_loss` 都支持；
    `r==0` 时两者相等的契约在 per_event 下同样成立（同一个 D）。
  - **model.forward 的 `compute_loss` 改用 `per_event`** ⇒ 训练、G1、G3 门禁臂自动继承。
- **R3 附带指标**（`loss_history.jsonl` + TB，训练侧新增/调整）：
  | 标量 | 定义 | 用途 |
  |---|---|---|
  | `loss` | 归一化后（per-event） | **主曲线**（跨步可比） |
  | `loss_sum_raw` | 旧口径 sum | 与历史曲线对照 |
  | `loss_nonempty_per_event` | 非空窗「逐窗 sum / 逐窗 E」的均值 | 主趋势（分层） |
  | `loss_empty` | 空窗逐窗 sum（=积分项）均值，口径不变 | 空窗行为 |
  | `integral_per_event` | Σ∫λdV / E（已有） | 强度校准：→1 = 校准正确 |
  | `clip_active` | 该步 `grad_norm > grad_clip_norm` 的 0/1 | 量化归一化对裁剪触发率的影响 |
  | `grad_norm` | 裁剪前总范数（已有） | 稳定性 |
  val 全套（`val/ratio`、`val/nll*`、`cond_*_delta`）不变——它们本就线数归一。
- **R4 P0 保留**：门禁 G1/G3 臂**固定 fp32**（训练仍 `bf16-mixed`）。依据：bf16 下同臂重复
  运行噪声 2.8×（RFC-0036 §2.3），任何阈值化读数都不可解释；fp32 同进程逐位一致。
- **R5 P1 迁移**：val 的置换对照从**全局置换**改为**线内置换**（`shuffle_counts_within_line`：
  每条 (样本, 线) 的被遮盖集合内各自置换）——全局置换连每线事件预算一起改（nuisance，
  RFC-0036 §2.5 实证），该缺陷对 val 对照同样成立。全局版 `shuffle_hidden_counts` /
  `shuffled_counts` 及 `SmokeBatchSource.batch(shuffled=True)` 随 G2 一并删除。

## 2. 设计细节与推导

### 2.1 归一化除子为什么是「事件总数」而不是「mask 格数」

- 泊松语义下损失的天然单位是**每事件**：最优点处事件项 ≈ Σ n·(1−log λ*)、积分项 ≈ E ⇒
  整式除以 E 后量级 O(1–10)，可读作「每事件编码成本 + 1」，且 `integral_per_event → 1`
  直接就是校准读数。
- mask 格数含大量 n=0 的格，用它做除子会把读数稀释进遮盖几何（r、格数），跨窗不可比；
- K / 格数归一消不掉主导项：实测标度律 `loss ≈ 2.24·K^0.055·E^0.93`（20k 步真实 run，R²=0.744）
  ⇒ 量级由 **E** 支配，按 E 归一是对症的。
- 决策者口径「对 mask 平均」按上取**对被监督事件平均**实现（整式除以 E）；若坚持按格数，
  改一个除子即可，但上表的可解释性会丢失。

### 2.2 argmin 不变性推导（记录在案，防止后人只改一半）

整式缩放：`L' = L/D`，D 与 λ 无关 ⇒ `∇L' = ∇L/D` ⇒ 驻点集合相同 ⇒ **λ* 不变**。
只除事件项：`L'' = (1/D)·Σ −n log λ + Σ λ dV`，对 λ_j 求导置零 ⇒
`λ*_j = n_j / (D·dV_j)` = 真强度 / D ⇒ **系统性低估 D 倍**，且积分项读数看起来仍「正常」
——典型的静默失效形态（本项目最怕的一类，见 CLAUDE.md 红线 7 的由来）。

### 2.3 门禁在归一化下的判据解读

- **G1**：`last <= max(target_loss=0.05, target_ratio=0.1 × first)`——相对项尺度无关，继续主导；
  绝对下限 0.05 的单位变为「每事件 NLL」（旧 sum 口径下它几乎永不起作用，新口径下同样只是下限）。
- **G3**：模型臂走同一 forward（per_event），基线**同除一个 D**：
  `baseline' = constant_baseline_nll(N, |Ω|) / max(E,1)`（同一批 ⇒ 同一个 E）。
  不等式 `model <= (1−0.1)·baseline` 两边同乘正数 ⇒ **判据逐位等价**，语义不变。
- **G4**：与损失无关，不动。

### 2.4 线内置换的正确性契约（进默认 CI）

① 可见场 `counts·~occlusion` 逐位不变；② **每线**事件数不变；③ 总事件数不变；
④ 同 seed 同进程逐位可复现；⑤ 用独立 `torch.Generator`，不污染全局 RNG。

## 3. 实现清单（本 RFC 随裁定即实施）

1. `generation/losses.py`：`Reduction` 加 `"per_event"`；`event_normalizer(batch)`（counts 按
   line_mask 求和，下限 1）；`masked/full_poisson_loss` 支持；docstring 记录 §2.2 陷阱。
2. `generation/model.py`：`compute_loss` 改 `reduction="per_event"`。
3. `infra/train_loop.py`：`build_gate_inputs` 删 G2 两臂；门禁臂 `precision="fp32"`（R4）；
   `constant_baseline_for` 同除 D；`shuffle_*` 消费点改 `baseline_*`；训练行新增
   `loss_sum_raw` / `loss_nonempty_per_event` / `clip_active`；`evaluate_val` 换线内置换；
   `_hidden_event_total` 随 G2 诊断量处理（gates.txt 上下文删 g2_* 键）。
4. `infra/smoke.py`：删 `shuffle_hidden_counts` / `shuffled_counts` / `batch(shuffled=)`；
   新增 `shuffle_counts_within_line(batch, *, seed)`（§2.4 契约）。
5. `infra/sanity.py`：删 `shuffled_target_control`；文档字符串同步（三门禁）。
6. `infra/gates.py`：`GateInputs` 删 `step_fn_g2_*`；`run_gates` 删 G2 stage；`thresholds_of`
   删 `g2_*`、增 `g3_steps/g3_samples/g3_chunks`；fail-closed 文案「G1-G4」→「G1/G3/G4」。
7. `infra/config/schema.py`：`GatesConfig` 删 `shuffle_steps/shuffle_samples/shuffle_chunks/
   shuffle_min_gap_ratio`，增 `baseline_steps=100 / baseline_samples=16 / baseline_chunks=4`
   （G3 沿用原 G2/G3 共享的预算值）；校验分支同步；`configs/smoke.yaml`、
   `configs/phigros_masked.yaml` 的 `gates.*` 同步。
8. 测试：删 G2 专属用例（`test_g2_pairing.py`、`test_sanity.py` / 两个 `test_sanity_gates.py`
   的 shuffled 用例、`test_gates.py` 的 g2 用例、`test_gate_progress.py` 等）；改
   `test_gate_min_events.py` / `test_gate_chunking.py` / `test_train_entry.py`（字段更名）；
   新增 `test_loss_per_event.py`（argmin 不变 + 尺度恒等式 + 空窗 D=1 + r==0 契约 + G3 基线
   同除子）、`test_shuffle_within_line.py`（§2.4 五契约）、门禁 fp32 装配断言。
9. 文档修宪：BasePlan §9 门禁表删 G2 行（加注 RFC-0037）、§7 line「G1-G4 门禁全绿」与
   风险表 R-4 改「G1/G3/G4」；CLAUDE.md 红线 7 + §5-8 + §6；AGENTS.md §4；docs/TRAINING.md
   （门禁章节 + R1 判读红线）；plan 04 §3.3/§4.3（损失口径 + per_event）；plan 07 §4.3 +
   新增 §9-56（实施记录）；decisions/README（0036 → 废弃、0037 → 已裁定）。
10. **续训边界**：损失语义变更 ⇒ 与旧 checkpoint / 旧 gates.txt **不可比**；不从旧 run
    `--resume` 跨此变更（GatesConfig 字段变化本身就会让续训指纹 fail-closed，属预期）。

## 4. 备选方案与放弃理由

| 方案 | 放弃理由 |
|---|---|
| 保留 G2、修夹具/判据（RFC-0036 A/B/C/D） | 决策者裁定 G2 的**设计**即错误；「速度赛」家族整体否决 |
| 归一化只做日志、不改优化目标 | 决策者明确要求重设计 loss；且 sum→per-event 的步间权重均等化正是「合理性」所在 |
| 只除事件项（字面「对 mask 平均」） | 改变最优强度（§2.2 推导），静默失效，否决 |
| 逐线归一（每线除自己的事件数） | 空事件线除子未定义、少事件线放大噪声；批内权重维持现状，只归一步间尺度 |
| 按 mask 格数归一 | 读数被遮盖几何稀释，跨窗不可比（§2.1） |

## 5. 后果

- `gates.txt` 变三行（G1/G3/G4）；门禁预算 ≈38 min → **≈10–15 min**（G2 两臂 ×16 段消失，fp32 略增）；
- 此前所有 gates.txt 的 **G2 行作废**（无论 PASS/FAIL）；G1/G3 历史读数因 bf16（P0 前）同样
  只能作历史参考；
- loss 量级变化 ⇒ 曲线跨此提交**不可比**（`loss_sum_raw` 保留映射）；
- 首次全量 run 要顺带验证：`clip_active` 率（预期从 56.4% 显著下降）、`val/ratio` 趋势、
  `val/nll_shuffled_delta > 0`；
- 红线 7 的「四道门禁」文字全仓同步为「三道门禁 + val 对照判读红线」。

## 6. 存疑清单

1. `grad_clip_norm=1.0` 在归一化损失下是否仍合适——先观察 `clip_active` 率再议（本次不动）；
2. `val/nll_shuffled_delta` 的判读红线目前只有符号（>0）——margin 待首个全量 run 的数据标定；
3. G1 绝对下限 0.05 的 per-event 语义未标定（相对判据主导，暂不阻塞）；
4. per-event 归一改变了步间隐式权重（事件重窗不再主导梯度方向）——对收敛速度的净影响
   未测，由 `val/ratio` 终裁。

## 7. 关联

[RFC-0036](RFC-0036-gate-batch-caliber-and-g2-power.md)（废弃；证据基础）、
[RFC-0029](RFC-0029-phigros-continuous-chart-generation.md)（§3.3 掩码补全——本案不动其语义，只归一量级）、
plan 04 §3.3/§4.3、plan 07 §9-54/§9-55/§9-56。
