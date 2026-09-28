# TensorBoard 标量参考（训练 + 门禁）

> 本文是**读图手册**：TensorBoard 里每一条曲线的**定义、单位、来源、怎么读、什么时候该警惕**。
> 权威记录是 `logs/loss_history.jsonl`（TB 只是它的人力监控视图），其次是 `gates.txt`。
> 键名 → TB 标签的映射**只有一处**：`beatmorph/infra/train_loop.py` 的 `SCALAR_TAGS`。改曲线名只改那一处。
>
> 覆盖版本：RFC-0037 落地后（per-event 归一损失、门禁 G1/G3/G4、val held-out 对照）。
> 相关：`docs/TRAINING.md` §7.5、`docs/plans/07-infra-training.md` §9-42/§9-46/§9-47/§9-51/§9-57/§9-59。

---

## 0. 从哪来、怎么看

| 项 | 值 |
|---|---|
| 启动 | `uv run tensorboard --logdir runs/phigros_masked` |
| event 文件 | `runs/<experiment>/<时间戳>/logs/events.out.tfevents.*` |
| 权威记录 | `runs/<experiment>/<时间戳>/logs/loss_history.jsonl`（每 `run.log_every` 步一行） |
| 门禁记录 | `runs/<experiment>/<时间戳>/gates.txt`（TB 的 `gate/*` 只是它的视图） |
| 写入代码 | 训练：`train_loop._flush_scalars()`；门禁：`gates.write_tb_scalars()` |
| 刷新频率 | 每 `run.log_every`（默认 **50**）步 flush 一次；`val/*` 与 `cond/*` **只在 val 步**出现 |

### 读图前必须知道的四条规则

1. **TB 会丢非有限值。** `±inf` / `NaN` **只留在 jsonl 里，不进 TB**（图上一条 ±inf 会把整条曲线压平）。
   所以「TB 没有这条线」≠「这一步没这个量」——缺数去 jsonl 找。
2. **缺失 ≠ 0。** 分层损失、`integral_per_event`、`val_*` 等在**无定义的步上直接不出现**，
   曲线是**断的**，不是掉到 0。把断点当 0 会读出假趋势。
3. **TB 不是权威。** 曲线给人看，判据与归档一律以 `loss_history.jsonl` / `gates.txt` 为准。
4. **`train/loss` 已换口径（RFC-0037）。** 它与本次变更**之前**的 run **不可比**；
   要跨变更对照，用 `train/loss_sum_raw`。

---

## 1. 命名空间总表

| 前缀 | 条数 | 什么时候有值 | 一句话 |
|---|---|---|---|
| `train/` | 14 | 每个记录步 | 训练损失及其分解、步时拆分、lr、梯度 |
| `perf/` | 1 | 每个记录步 | 数据侧占步时的比例（供给健康度） |
| `sys/` | 2 | 每个记录步 | 显存 **torch 分配器口径** 与「保留但空闲」量 |
| `coverage/` | 3 | 每个记录步 | 数据覆盖（防采样器饱和那类**静默失效**） |
| `val/` | 16 | **仅 val 步** | 固定验证集读数（主判据 `val/ratio`） |
| `cond/` | 3 | **仅 val 步** | 条件干预三元组（音频 / 线轨有没有被用上） |
| `gate/` | 3 | **step 0 一次** | G1 / G3 / G4 的 PASS(1) / FAIL(0) |

---

## 2. `train/*` —— 训练损失与它的分解

训练损失的权威定义（RFC-0037 §2.1）是**按事件归一**：

```
L = [ Σ_有效线 ( (1/r) · Σ_被遮盖格 ( −n·log λ + ∫λ dV ) ) ] / max(E_total, 1)
```

其中 `r` = 被遮盖格占比（重标定系数）、`E_total` = 批内**有效线**上的事件总数。
`max(E_total, 1)` 让空批退化为**积分项本身**（与旧行为一致）。

### 2.1 主曲线

| TB 标签 | 定义 | 单位 | 怎么读 |
|---|---|---|---|
| `train/loss` | 上式，当前口径的训练损失 | nats/事件量级 | **不是**主趋势曲线（见下） |
| `train/loss_nonempty_per_event` | 非空窗的**每事件**归一损失 | 同左 | ✅ **主趋势曲线**（跨步可比） |
| `train/loss_sum_raw` | `train/loss × max(E_total,1)` = 旧 sum 口径 | 原始和 | 只用于**跨 RFC-0037 对照** |

⚠️ `train/loss` 的分布是**双峰**的：空窗中位约 `6.5e-4`、非空窗中位约 `96`，相差五个数量级
（plan 07 §9-46 实测）。一个把两个总体混在一起的数字**不携带趋势信息**——本项目曾因此让
`best.pt` 选中一个 `events=0, K=1` 的空窗步。**看趋势请用分层曲线。**

### 2.2 分层（把双峰拆开）

| TB 标签 | 定义 | 出现条件 |
|---|---|---|
| `train/loss_empty` | 空窗的**逐窗求和**损失 | 本步存在空窗 |
| `train/loss_nonempty` | 非空窗的**逐窗求和**损失（`batch_size=1` 时等于 `loss_sum_raw`） | 本步存在非空窗 |
| `train/loss_nonempty_per_event` | `loss_nonempty / E`（主趋势曲线） | 同上 |
| `train/empty_share` | 本步批内空窗占比 | 总是 |
| `train/nonempty_share` | `1 − empty_share` | 总是 |

**关键恒等式**：空窗的损失**恒等于积分项** `∫λdV`（事件项为空）——所以 `loss_empty` 下降
只说明模型把 `λ` 压小，**不**说明它学会了放事件。这是本项目最容易误读的一条曲线。

### 2.3 其他训练读数

| TB 标签 | 定义 | 怎么读 |
|---|---|---|
| `train/integral_per_event` | `Σ∫λdV / Σn`（有事件时才出现） | 预期事件数 / 真实事件数，见 `val/pred_over_true` |
| `train/clip_active` | 该步**是否**触发梯度裁剪（0/1） | 旧 run 56.4% 的步触发；RFC-0037 后实测非空步仍 **100%** |
| `train/grad_norm` | **裁剪前**的梯度 L2 范数（仅有限值写入） | 与 `clip_active` 一起看：`grad_norm` 长期远大于 `grad_clip_norm` 说明实际更新近似归一化梯度 |
| `train/lr` | 学习率 | 当前是常数 `3e-4`；曲线用于确认它**没有**被意外改动 |

### 2.4 步时拆分（plan 07 §9-42）

| TB 标签 | 定义 |
|---|---|
| `train/step_time_s` | 一个优化步的**全部墙钟**（**含**取批） |
| `train/data_time_s` | 取批耗时（builder / DataLoader 等待） |
| `train/compute_time_s` | 前向 + 反传 + 优化器 |

恒等式：`step_time_s = data_time_s + compute_time_s`（差值是记账之外的固定开销）。
计时起点在**取批之前**，所以数据构建时间一直混在步时里，这正是要拆开它的原因。

---

## 3. `perf/*` —— 供给健康度

| TB 标签 | 定义 | 怎么读 |
|---|---|---|
| `perf/data_share` | `data_time_s / (data_time_s + compute_time_s)` | **看均值，不看中位** |

⚠️ 这条分布**极端重尾**：实测中位 **5.0%** 与均值 **45.0%** 可以**同时为真**（前 10% 的步吃掉
77.4% 的数据时间），而决定吞吐的是**均值**。第九轮曾只看中位、得出「数据侧已不是瓶颈」
的相反结论，第十轮用三把量具（jsonl / `nvidia-smi` 采样 / `py-spy dump`）才纠正。
**判据（RFC-0034）**：`data_share` 均值 **< 20%** 为达标。

---

## 4. `sys/*` —— 显存（**torch 分配器口径，不是驱动口径**）

| TB 标签 | 定义 | 陷阱 |
|---|---|---|
| `sys/peak_vram_gib` | `torch.cuda.max_memory_allocated() / 2^30`，**全局高水位**（单调不减） | ⚠️ **不含** CUDA 上下文 / cuBLAS·cuDNN workspace / 分配器碎片 / 驱动预留 |
| `sys/vram_reserved_gib` | `memory_reserved() − memory_allocated()` = **保留但空闲** | 与 `peak_vram_gib` 之差即「空闲占位」，事故的直接机制 |

**这条是全篇最重要的警告**：判显存墙**必须**用驱动侧读数
（`nvidia-smi --query-gpu=memory.used`），不是 `sys/peak_vram_gib`。
实测（plan 07 §9-57）：K=128 的批 torch 峰值 5.23 GiB，而驱动侧到 **7874 / 8151 MiB**，
下一步就滑进共享显存并卡死。

**回收是否生效**：看 `vram_reserved_gib` 是否**不随历史峰值累积**（`optim.vram_hygiene_gib`
每步在「保留量远超实际分配」时 `empty_cache()`）。

⚠️ **`val` 段没有任何独立显存遥测**（见 §10）——val 跨墙在 TB 里**完全看不见**。

---

## 5. `coverage/*` —— 数据覆盖（防静默失效）

没有这三条曲线，**采样器饱和**这类失效在 loss 上完全看不出来（它只是反复拟合同一小撮样本）。

| TB 标签 | 定义 | 形态 |
|---|---|---|
| `coverage/epoch` | 当前 epoch 序号 + 本 epoch 已消费窗口占比 | **单调递增**的分数 epoch |
| `coverage/windows_seen` | 本 epoch 内已消费窗口 / 总窗口（0–1） | **锯齿**（每个 epoch 归零） |
| `coverage/charts_seen` | 本 epoch 前缀里的**去重谱面**数 / 谱面总数 | **锯齿**（每个 epoch 归零） |

- **一个 epoch = 走遍全库、每个窗口恰好一次** = train split **634 952 步**（本机约 88 h @0.5 s/步 ⇒ **跑不完一个 epoch**）。
- 判据：`max_steps=20000` 时约 **91% 谱面覆盖**（约 3.15% window-epoch）。
- ⚠️ `windows_seen` / `charts_seen` 是**逐 epoch 前缀的纯函数**（RFC-0034 起），**不是累计量**；
  「覆盖率只增不减」是 RFC-0033 时代的旧描述，**不再成立**。
- 若 `charts_seen` **平掉**（长时间不涨）⇒ 采样器又饱和了，立刻停并查
  `tests/unit/infra/test_sampler_coverage.py`，**不要**靠加步数掩盖。

---

## 6. `val/*` —— 固定验证集读数（**仅 val 步**）

**同一测度才可比**：常数基线在与 val **完全相同的遮盖测度下重算**，因此模型读数与基线
只差「谁给 λ」（测度 / 归约 / 分母逐位一致）。归约**按有效线数**（`Σ损失 / Σ有效线`），
所以 K 变化不会让读数跳动。

### 6.1 主判据

| TB 标签 | 定义 | 怎么读 |
|---|---|---|
| `val/nll` | 模型在遮盖测度下的 NLL = `Σ损失 / Σ有效线` | 越低越好 |
| `val/nll_masked_constant` | 常数场**在同一测度下**的最优值（闭式 `c* = s·N_sup / 域体积`） | 基线 |
| `val/ratio` | `val_nll / val_nll_masked_constant` | ✅ **主判据；小于 1 表示优于常数场** |

> ⚠️ **`val/nll` 与 `gates.txt` 里的 G3 数字不可比**：门禁路径 `initial_head_bias=20`、
> 训练路径 `bias=0`，两者不是同一初值下的读数。

### 6.2 判读红线（RFC-0037 R1）

> **任一 val 点满足 `val/ratio >= 1` 或 `val/nll_shuffled_delta <= 0`**
> ⇒ 该 run 的**扩规模结论作废**。`best.pt` 按 `val/ratio` 选。

### 6.3 对照与分解

| TB 标签 | 定义 | 怎么读 |
|---|---|---|
| `val/nll_shuffled` | **线内置换**遮盖标签后的 NLL（每线事件数不变） | 模型应**显著优于**它 |
| `val/nll_shuffled_delta` | `val_nll_shuffled − val_nll` | ✅ **红线：必须大于 0** |
| `val/nll_empty` | 空窗的分层 NLL（按空窗有效线加权） | 只看积分项 |
| `val/nll_nonempty` | 非空窗的分层 NLL | 事件项的好坏主要看它 |
| `val/nll_full_event` | **全事件口径** NLL（走离线 `beatmorph-eval` 同一入口） | ⚠️ **不可与 `val/nll` / `val/nll_masked_constant` 直接比**（不同测度） |
| `val/pred_over_true` | `val_integral / val_events` = 预期事件数 / 真实事件数 | 1 约等于计数校准；**远小于 1 = 强度场塌缩、严重欠计数** |

### 6.4 构成与成本（诊断用）

| TB 标签 | 定义 | 备注 |
|---|---|---|
| `val/windows` | 本次 val 的窗口数（默认 128） | 构成**未收敛**（空窗占比随窗口数漂，plan 07 §9-51 ⑤） |
| `val/lines` | Σ有效判定线数 | 归约的分母 |
| `val/events` | Σ真实事件数 | |
| `val/integral` | Σ∫λdV（全 val 批） | 与 `val/events` 一起看校准 |
| `val/empty_share` | val 批内空窗占比 | 指标水平受它影响 |
| `val/r0_share` | **遮盖退化**窗口占比（有事件但无可遮盖格，`r==0`） | 这些窗口不携带对照信息 |
| `val/time_s` | 一次 val 的**真实墙钟**（秒） | ⚠️ 单次约 **100–170 s**（推导值 33 s，**miss 3×**，且跨墙，见 plan 07 §9-59） |

---

## 7. `cond/*` —— 条件干预三元组（**仅 val 步**）

**同批、同权重**的前向差分：`Δ = NLL(干预) − NLL(基线)`。

> `Δ > 0` ⇒ 该条件**降低了**损失（**被用上了**）；`Δ ≈ 0` ⇒ 模型没在看它；
> `Δ < 0` ⇒ 该条件在**帮倒忙**。

| TB 标签 | 干预 | 含义 |
|---|---|---|
| `cond/audio_zero_delta` | 音频特征**置零** | 模型只能靠事件轨与遮盖几何 |
| `cond/audio_perm_delta` | 音频**时间轴**置换（分布内，破坏时间对齐） | 比置零更狠；约等于 0 说明音频**时间对齐**没被用到 |
| `cond/track_zero_delta` | 判定线事件轨置零 | 模型只能靠音频 |

⚠️ 这是「音频条件有没有被用上」在本项目里的**唯一**证据通道（此前所有比较都是
「条件在场 vs 条件在场」）。但它目前**没有基线**：Δ 多大才算「用上了」**未标定**。

---

## 8. `gate/*` —— 门禁（step 0 写入一次）

| TB 标签 | 门禁 | =1 表示 |
|---|---|---|
| `gate/G1_pass` | G1 单 batch 过拟合 | 300 步内 loss 降到初值的 10% 且不超过 0.05 |
| `gate/G3_pass` | G3 常数基线 | 模型优于常数场（按 `max(E,1)` 归一后比较） |
| `gate/G4_pass` | G4 帧率契约 | 音频帧数 × 派生帧率 = 时长（容差 2 帧） |

- 原 **G2（打乱标签对照）已随 RFC-0037 整体删除**（判别力未被证明）；其命题改由
  `val/nll_shuffled_delta` + `val/ratio` 承担。
- **`gates.txt` 是权威**（含阈值与上下文）；TB 上的 `gate/*` 只是 0/1 视图。
- 门禁未全绿 ⇒ 训练 **fail-closed 不启动**，所以正常 run 的这三条都应为 1。

---

## 9. 常见误读清单

| 误读 | 事实 |
|---|---|
| 「`train/loss` 在降 = 训练在变好」 | 双峰混合，空窗步把曲线拉到底；看 `loss_nonempty_per_event` |
| 「`loss_empty` 降 = 学会了放事件」 | 空窗损失**恒等于积分项**，它降只说明 λ 被压小 |
| 「TB 上这条线断了 = 0」 | 缺失 ≠ 0；无定义的量不写 |
| 「`sys/peak_vram_gib` 没爆 = 显存安全」 | 它是**分配器口径**；必须看 `nvidia-smi memory.used` |
| 「`perf/data_share` 中位很低 = 数据不是瓶颈」 | 分布重尾；**只看均值** |
| 「`val/nll` 和 G3 的数字能比」 | 初值不同（bias 20 vs 0），只能各自内部比较 |
| 「`val/nll_full_event` 比 `val/nll` 差 = 模型变差」 | 两者**测度不同**，不可直接比 |
| 「加步数能提高覆盖率」 | 采样器饱和时曲线会平掉，加步数无效 |

---

## 10. 缺口与存疑（已知的「看不见」）

| 缺口 | 影响 | 出处 |
|---|---|---|
| **val 段无独立显存遥测** | val 跨墙（共享显存峰值 5.6 GB）在 TB / jsonl 里**不可见**，只能靠外部 `nvidia-smi` 读数 | plan 07 §9-59 |
| **val 批级计时未落盘** | 单次 val 墙钟的波动（100.52 / 167.58 / 120.81 s）无法归因到「哪一批、哪个桶」 | plan 07 §9-59 |
| **条件干预的 Δ 无基线** | `cond/*_delta` 是确定性数字，但「多大算用上了」未标定 | plan 07 §9-51 ④ |
| **`val_windows=128` 的构成未收敛** | 空窗占比随窗口数漂（51.6% → 59.0%），指标**水平**会跟着偏 | plan 07 §9-51 ⑤ |
| **`val/ratio` 回答不了「架构对不对」** | NLL 好 ≠ 谱面可玩；难度条件是否被用上需要**难度置换**对照（未实现） | plan 07 §9-47 H |
| **门禁 `gate/*` 不落 jsonl** | 门禁读数只在 `gates.txt` 与 TB，不在 `loss_history.jsonl` | 代码事实 |
| **`sys/peak_vram_gib` 是全局高水位** | 无法回答「哪一步把显存顶上去的」（除非逐段 reset，当前没有） | `train_loop._peak_vram_gib` |
