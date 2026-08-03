# BeatMorph 训练小记 — Phase 1 首次跑通

> 日期：2026-08-03  
> 设备：Windows 11 + NVIDIA RTX 5070 Ti Laptop GPU  
> 数据规模：50 set（sayobot 下载 / 119 个 4K mania 谱面通过质量过滤）  
> 目标：验证全链路可行性（环境→数据→提取→训练）

---

## 一、执行流程

严格按照 `docs/TRAINING.md` §9 cheatsheet 执行：

```
uv sync → MERT 权重(ModelScope) → sayobot 下载(50 set) → PreprocessPipeline → MERT 离线提取 → Stage1 训练
```

**实际耗时**：约 55 分钟（含 MERT 权重首次下载 ~4GB）

---

## 二、训练结果

| 指标 | 值 |
|------|------|
| 模型 | DensityPlanner (6层双向 Transformer) |
| 参数量 | 75.6 M |
| 数据集 | 119 样本 |
| 训练步数 | 10,000 |
| Epochs | 84 |
| Batch size | 1（变长 embedding） |
| 最终 loss | **0.369** |
| 精度 | bf16-mixed |
| GPU | NVIDIA RTX 5070 Ti Laptop |
| Checkpoint | `runs/checkpoints/epoch=84-step=10000.ckpt` (~907 MB) |

loss 从初始约 1.8 降至 0.369，收敛正常。阶段 1 密度规划模块已可学习音频→段落密度/能量/类型的映射关系。

---

## 三、修复的 bug（共 5 个）

### Bug 1：MERT 采样率错误

**文件**：`beatmorph/audio/encoder/mert.py:36`  
**现象**：`ValueError: was trained using a sampling rate of 24000`  
**原因**：`_TARGET_SR = 16000`，但 MERT-v1-330M 实际训练采样率为 24kHz  
**修复**：改为 `_TARGET_SR = 24000`；同步修改 `embed.py` 的 `_load_audio_resampled` 默认值

### Bug 2：张量维度错配

**文件**：`beatmorph/audio/encoder/mert.py:157`  
**现象**：`RuntimeError: Expected 2D/3D input to conv1d, got [1, 1, 1, 120000]`  
**原因**：传入 2D `[1, samples]` 给 Wav2Vec2 特征提取器，新版 transformers 增加了 channel 维导致 4D  
**修复**：squeeze 到 1D 后传入 featurizer，输出后再归一化到 2D

### Bug 3：FP16 类型不匹配

**文件**：`beatmorph/audio/encoder/mert.py:171`  
**现象**：`Input type (FloatTensor) and weight type (HalfTensor) should be the same`  
**原因**：特征提取器输出 float32，但 MERT 主干是 fp16  
**修复**：`input_values.to(dtype=dtype, device=wav.device)` 先转 dtype 再送入 backbone

### Bug 4：BPM 时间点负值崩溃

**文件**：`beatmorph/io/formats/osu.py:272`  
**现象**：`ValidationError: BpmPoint.time should be >= 0 (input_value=-0.016)`  
**原因**：部分 .osu 文件 timing points 有微小负时间  
**修复**：`time_s = max(0.0, time_ms / 1000.0)`

### Bug 5：MERT 输出维度不匹配

**文件**：`beatmorph/core/contracts/tensors.py:70` + `configs/model/planner.yaml:8`  
**现象**：embedding 实际 1024 维，但模型配置 768 维  
**原因**：`MERT_DEFAULT_FEAT_DIM = 768`，但 v1-330M 所有层 hidden_size=1024  
**修复**：常量改为 1024，模型 dim 同步改为 1024

---

## 四、还存在的问题

1. **batch_size=1**：当前 collate 不支持变长序列 padding，batch_size 只能设为 1。后续需要实现 padding-aware collate。
2. **HOLD note 退化警告**：osu! 社区谱面中大量 HOLD note 的 `endTime == startTime`（duration=0），parser 会将其降级为 TAP。这是数据质量问题，不影响训练，但日志噪音大。
3. **SSL 证书错误**：HuggingFace SSL 验证失败，不影响运行（自动 fallback 到 ModelScope），但每次实例化 MERT 都重试，拖慢启动。
4. **仅验证 loss 下降**：尚未评估段落类型 F1、密度 MAE 等业务指标（对应 Plan 03 M3）。

---

## 五、下一步

- [ ] 扩大数据量：`--max-sets 10000`，跑完整 Phase 1 训练
- [ ] 实现变长 padding collate，支持 batch_size > 1
- [ ] 接入 W&B 实验追踪（当前 `wandb.entity=null`）
- [ ] 训练 VQ-VAE Tokenizer（Plan 02，Phase 1 第二个里程碑）
- [ ] 在 10K 数据上评估 Stage 1 段落类型 F1 和密度 MAE

---

## 六、教训

1. **MERT-v1-330M 的关键参数**：采样率 24kHz、hidden_size=1024。代码中的注释写"16kHz"和常量 768 都是错的——可能来自 MERT-v1-95M 版本的参数混淆。
2. **`uv sync` 后不要 `uv add`**：`uv add` 会重建 venv，需再跑一次 `uv sync` 恢复 editable install。
3. **sayobot 503**：约 10% 请求返回 503，脚本自带重试机制可兜底，但 10K 级下载需要更长的等待时间。
4. **Windows + Rich 终端**：`PYTHONUTF8=1` 解决 Rich 库 Unicode 渲染在 GBK 终端下的崩溃。
