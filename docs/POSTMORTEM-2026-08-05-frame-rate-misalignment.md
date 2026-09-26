# Postmortem — Stage1 训练 loss 无法下降的根因（帧率假设错 3×）

> 日期：2026-08-05 ｜ 严重级别：**P0（静默数据损坏，全链路下游不可信）**
> 结论一句话：**MERT-v1-330M 的真实输出帧率是 75 Hz，而代码在三个地方硬编码 25 Hz，且提取路径不做任何降采样，导致"秒→帧"映射整体错 3×，段特征与段标签系统性错位。**
> 关联：RFC-0028（tokenizer 范式）、RFC-0026（phase 对齐）、`docs/TRAINING_LOG_2026-08-03.md`（首次训练记录）、待立 **RFC-0029**（Phigros + 点过程范式）

---

## 1. 症状

- 119 样本规模：DensityPlanner 训练 loss 1.8 → 0.369，被判定为"收敛正常"。
- 扩到万级样本后：**训练集 loss 完全不可下降**（不是不收敛、不是过拟合，是纹丝不动）。
- 该症状被初步归因为"tokenizer/规划范式结构性缺陷"。

## 2. 根因

### 2.1 MERT-v1-330M 的真实帧率 = 75 Hz

证据：ModelScope 官方仓库 `m-a-p/MERT-v1-330M` 的 `config.json`：

```json
"conv_dim":    [512, 512, 512, 512, 512, 512, 512],
"conv_kernel": [ 10,   3,   3,   3,   3,   2,   2],
"conv_stride": [  5,   2,   2,   2,   2,   2,   2],
"hidden_size": 1024
```

- stride 累乘 = `5·2·2·2·2·2·2 = 320`。
- 输入采样率 24000 Hz（`preprocessor_config.json`；RFC-0026/TRAINING_LOG Bug1 已确认）。
- **帧率 = 24000 / 320 = 75 Hz。**

逐层输出长度核对（5 s = 120000 样本，`L_out = floor((L_in - k)/s) + 1`）：

| 层 | k | s | L_out |
|----|---|---|-------|
| 1 | 10 | 5 | 23999 |
| 2 | 3 | 2 | 11999 |
| 3 | 3 | 2 | 5999 |
| 4 | 3 | 2 | 2999 |
| 5 | 3 | 2 | 1499 |
| 6 | 2 | 2 | 749 |
| 7 | 2 | 2 | **374** |

374 帧 / 5 s = **74.8 Hz**，与 320 的推导一致（±1 帧来自卷积取整）。

**经验验证（2026-08-05 本地实测，三条独立路径一致）**，脚本 `scripts/verify_mert_frame_rate.py`：

| 路径 | 结果 |
|------|------|
| config 推导 | `prod(conv_stride) = 320` → 24000/320 = **75.0 Hz** |
| torch 按 config 实搭卷积栈 | 1s→74 帧、2s→149 帧、5s→**374 帧 = 74.8 Hz** |
| 真实 `pytorch_model.bin` 权重 | `conv_layers.0=(512,1,10)`、`1-4=(512,512,3)`、`5-6=(512,512,2)`；`encoder.layers.0.attention.k_proj=(1024,1024)` → hidden=1024 |

即：帧率与特征维**都能从权重本身读出来**，而代码当年把它们分别写成了 25 Hz 与 768。

> 注：`hidden_size = 1024` 与 TRAINING_LOG Bug5 的实测一致——同一份 BasePlan §3.1 表格当年同时写错了 `768` 和 `25Hz` 两个数，768 后来被实测纠正，**25 Hz 从未被纠正**。

### 2.2 代码里三处硬编码 25 Hz，且互相独立

| 位置 | 用途 |
|------|------|
| `beatmorph/core/contracts/tensors.py` `MERT_FRAME_RATE_HZ = 25.0` | 契约常量 |
| `beatmorph/audio/encoder/mert.py` `_FRAME_RATE = 25.0` | 5 s 滑窗重叠拼接的样本偏移 → 帧偏移换算 |
| `beatmorph/planner/density.py` `_FRAME_RATE = 25.0` | `_pool_sections` 的"秒→帧"换算（**独立复制的第三份**） |

### 2.3 提取路径不做任何降采样

`beatmorph/data/pipeline/embed.py`：

```python
emb = enc.encode(wav_t)                      # [1, T_seq, 1024]，T_seq = 75 * duration
torch.save(emb.squeeze(0).cpu().float(), emb_path)
```

落盘的就是 75 Hz 的原生帧率，**没有任何 pool / interpolate / resample**。因此"25 Hz"从头到尾只是一个假设。

### 2.4 误差传导

以一首 120 s 的曲子为例：

| 量 | 值 |
|----|----|
| 实际落盘 T_seq | ≈ 120 × 75 = **9000** 帧 |
| 代码认为的时长 | 9000 / 25 = **360 s**（真值 120 s） |
| 段落 [8 s, 16 s] 的池化 mask | 取帧 [200, 400] |
| 这些帧的**真实**时间 | **[2.67 s, 5.33 s]** |
| 最后一段 [112 s, 120 s] | 帧 [2800, 3000] → 真实 **[37.3 s, 40.0 s]** |

即：**模型在每个段落上看到的，是歌曲前 1/3 的音频**，且错位量随段落序号漂移。输入对标签不含信息 → 模型只能收敛到"预测数据集均值"这一地板。

滑窗拼接同理：`_merge_overlapping` 用 25 Hz 把 4 s 的 hop 换成 100 帧，实际应为 300 帧，**每 4 秒的接缝区都在平均错位的帧**。

### 2.5 为什么症状是"小数据能降、扩规模不降"

`TRAINING_LOG` 记录：119 样本 / 10000 step / **84 epochs**。84 个 epoch 跑 119 个样本 = 纯记忆，loss 下降可以完全来自背下噪声。一旦样本数上万，记忆不可能，而输入对目标零信息，loss 只能停在均值预测器的水平——**怎么训都不动**。

这是"输入与标签不对齐"最典型的签名，与模型容量、学习率、数据量都无关。

### 2.6 叠加的第二个缺陷（真·结构性问题）

即使帧率修复，`DensityPlanner._pool_sections` 仍然把**一个 4 小节段落的全部帧均值池化成一个向量**，再让 6 层 Transformer 在约 30 个段向量上预测每段密度。密度的本质是 onset 速率，而段均值恰好把 onset 信息平均掉——**模型被设计成看不到它要预测的东西**。

两个缺陷相互独立：帧率是 bug（可修），段级均值池化是设计问题（应改成帧级/事件级建模）。修前者不会自动修后者，反之亦然。

## 3. 为什么 84 个 epoch 都没暴露

1. **唯一能证伪的测试被跳过**：`tests/unit/audio/test_mert.py::test_encode_shape_and_frame_rate` 断言 `T_seq ≈ dur*25`，但带 `@pytest.mark.gpu` + `require_mert`（权重未缓存也 skip）→ `make test-fast` 永远不会跑它。
2. **mock 把错误常量固化成绿灯**：

| 位置 | 内容 |
|------|------|
| `tests/unit/core/test_contracts.py:97` | `assert MERT_FRAME_RATE_HZ == 25.0` |
| `tests/unit/planner/test_density.py:29` | `B, T = 2, 250  # 10s @25Hz` |
| `tests/integration/test_audio_to_plan.py:22` | `t_seq = round(dur * 25)` |
| `tests/unit/data/test_datasets.py:170` | mock encoder 返回 `round(dur*25)` 帧 |

测试套件本身在**断言这个 bug 是正确的**——mock 层把假设洗成了不变量。
3. **缺少健全性门禁**：没有单 batch 过拟合测试、没有打乱标签对照、没有常数基线对照。

## 4. 影响面

**不可信（需重跑）**：
- 任何已提取的 `{sid}.pt` embedding（帧轴错 3×）。
- 基于这些 embedding 训练的 DensityPlanner 权重（`runs/checkpoints/*.ckpt`）。
- `TRAINING_LOG_2026-08-03` 中"loss 0.369 = 收敛正常"的结论。

**不受影响**：
- `.osu` 解析、`compute_bar_boundaries`（RFC-0026 phase 脊柱，纯时间域）。
- BPE/event 原子层（`tokenizer/events.py` 不依赖音频帧率）。
- 契约结构、训练栈、数据流水线骨架。

## 5. 修复

1. **契约**：`MERT_FRAME_RATE_HZ` 25.0 → **75.0**，并新增可推导的量
   （`MERT_SAMPLE_RATE_HZ = 24000`、`MERT_CONV_STRIDE_PRODUCT = 320`），使 `24000/320 = 75` 成为**可断言的关系**而不是魔法数。
2. **`MERTAdapter`**：新增 `output_frame_rate()`，**从主干 config 的 `conv_stride` 推导**帧率，不再硬编码。
3. **`DensityPlanner`**：删除自带的第三份 `_FRAME_RATE = 25.0`，统一引用契约。
4. **测试**：修正固化错误常量的断言，新增**默认 CI 内运行**的帧率契约测试（不依赖权重）。
5. **产物重跑**：embedding 与 Stage1 权重必须在修复后重跑（`TRAINING.md` §9 cheatsheet）。

## 6. 四道健全性门禁（制度化，`beatmorph/infra/sanity.py`）

任何新模型/新范式在扩大数据规模之前，必须依次通过：

| 门禁 | 判据 | 抓什么 |
|------|------|--------|
| G1 单 batch 过拟合 | 在 1–4 个样本上能把 loss 打到接近 0 | 通路断、梯度断、loss 用错 |
| G2 打乱标签对照 | shuffle 目标后 loss **必须显著变差** | 输入对目标无信息（本 bug 会在此当场现形） |
| G3 常数基线 | 模型 loss 必须显著优于"直接预测数据集均值" | 模型其实什么都没学到 |
| G4 契约断言 | `T_seq == round(duration × rate)`，且 rate 由模型 config 推导 | 单位/帧率/采样率漂移 |

> G2 与 G4 是关键：G4 会把本 bug 变成红灯，G2 会把"通路上其实没有信息"变成红灯。

## 7. 对 RFC-0029（Phigros + 点过程范式）的硬约束

1. **任何"秒→帧"映射不得硬编码**：帧率必须从特征提取器 config 推导，并在启动时打印 + 断言。
2. **特征缓存必须记录元数据**：`{rate, sample_rate, layer, model_rev, duration_s}` 随 `.pt` 一起落盘，加载时校验。
3. **新范式上线前必须过 G1–G4**，且门禁结果写入训练日志。
4. **点过程 NLL 的 `∫λ` 项**必须在与强度场同一离散网格上数值积分，且网格分辨率显式声明——否则会以另一种形式重现"单位错配"这一类 bug。
5. **评估必须在原始时间域**（秒）进行，而不是在帧索引域。

## 8. 教训

> 一个没有测试覆盖的物理常量，等价于一个未经声明的假设；而一个 **mock 覆盖了** 的物理常量，等价于一个被伪装成事实的假设。

BasePlan §3.1 的 `768` 被实测纠正了，`25Hz` 没有——差别只在于：768 有形状断言（`shape[-1]`）顺手撞上，25 Hz 只有一条被 skip 的 gpu 测试。
