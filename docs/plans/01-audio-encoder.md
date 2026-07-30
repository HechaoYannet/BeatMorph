# Plan 01 — Stage 0 多模态音频编码器（MERT + Adapter + 可选 Demucs）

> 状态：🟡 草案 ｜ 阶段：Phase 1 ｜ 负责：音频组
> 对应代码：`beatmorph/audio/encoder/mert.py`、`beatmorph/audio/separation/demucs.py` ｜ 对应奠基章节：§3.1

## 1. 目标与范围

### 交付
- `MERTAdapter`：冻结 MERT-v1-330M，注入可训练 Adapter（LoRA 或 2 层 MLP），输出 25Hz、768 维帧级序列 embedding。
- `DemucsSeparator`：HTDemucs 四轨分离骨架，作为可选增强（开关式）。
- 分轨特征融合策略：`{drums, bass, vocals, other}` 四轨分别过轻量 encoder 后沿特征维拼接。
- 离线 embedding 预提取脚本（对应奠基 §4.2 Step 3，喂下游 Stage1/Stage2）。

### 不交付
- MERT 主干预训练权重自身的训练（仅冻结加载）。
- Stage 1 规划、Stage 2 生成、解码与后处理（属 plan 03/04/07）。
- Demucs 模型自身的训练（仅加载预训权重，奠基标记为「可选」）。

### 价值
提供整条管线的「听觉感知层」：把音频压缩为节奏/旋律分层表征序列，供下游做密度回归与 Pattern 生成。MERT 音乐专用 + 层次表征 + 轻量，使其在仅取 embedding 的场景下显著优于 Qwen2-Audio（奠基 §3.1.3：7B 冗余、非为音乐生成微调最优）。

## 2. 与奠基文档的对应

| 本计划项 | 奠基依据 | 偏离 | 理由 |
|----------|---------|------|------|
| MERT-v1-330M 冻结 + Adapter | §3.1.1 表 | — | 直接采用 |
| 取第 12 层、768 维 | §3.1.1「768（第12层）」 | — | 与契约 `MERT_DEFAULT_FEAT_DIM=768`、`AudioEmbedding.feat=768` 一致 |
| 25Hz 帧率 | §3.1.1 表 / §2 架构图 | — | 与 `MERT_FRAME_RATE_HZ=25.0` 一致 |
| Demucs 四轨 + 特征维拼接 | §3.1.2 | — | 融合方式照搬 |
| Demucs 设为可选标志 | §3.1.2「设为可选标志」 | — | 开关位于 `MERTAdapter` 构造参数 |
| 输入 16kHz 单声道 | §3.1 未显式 | 微偏离 | MERT 官方要求 16kHz；契约记录为 `(batch, time_seq, 768)`，本计划补齐波形侧 `(batch, samples)` 与采样率约束 |
| 5 秒上下文窗口 | §3.1.1 表 | 微偏离 | 仅离线全曲推理时按长音频滑窗 + 重叠处理，不强制 5s 切片；记入 §9 RFC |

## 3. 接口契约

### 3.1 MERTAdapter（对应 `beatmorph/audio/encoder/mert.py`）
```python
class MERTAdapter:
    def __init__(
        self,
        model_name: str = "m-a-p/MERT-v1-330M",
        layer: int = 12,
        adapter: str = "lora",        # "lora" | "mlp" | "none"
        use_demucs: bool = False,     # §3.1.2 可选增强开关
    ) -> None: ...

    def encode(self, wav: "torch.Tensor") -> "torch.Tensor":
        """wav: [B, samples] 16kHz 单声道 → [B, time_seq, 768] 25Hz"""
```
张量契约引用 `beatmorph.core.contracts.AudioEmbedding`（`feat=768`、`hop_rate=25.0`）。`time_seq ≈ duration_s * MERT_FRAME_RATE_HZ`。

### 3.2 DemucsSeparator（对应 `beatmorph/audio/separation/demucs.py`）
```python
STEMS: tuple[str, ...] = ("drums", "bass", "vocals", "other")

class DemucsSeparator:
    def __init__(self, model_name: str = "htdemucs", device: str = "cpu") -> None: ...
    def separate(self, wav: "torch.Tensor") -> dict[str, "torch.Tensor"]:
        """wav: [B, samples] 混合 → {stem: [B, samples]}"""
```

### 3.3 跨模块张量形状（einops 风格）
| 名称 | 形状 | 含义 |
|------|------|------|
| 输入波形 | `(batch, samples)` | 16kHz 单声道 |
| 单轨 embedding | `(batch, time_seq, 768)` | 每轨过轻量 encoder |
| 融合输出（Demucs 开） | `(batch, time_seq, 768 * (1+4))` | 主轨 + 四轨沿 feat 拼接，再投影回 768 |
| 最终 `audio_emb` | `(batch, time_seq, 768)` | 下游 `DensityPlanner.plan` 输入 |

## 4. 内部设计

- **MERT 加载**：HuggingFace `transformers`，`requires_grad_(False)` 冻结全部参数；FP16 推理（T4 可实时，奠基 §3.1.1）。
- **Adapter 注入**：`adapter="lora"` 时对 MERT 后若干层 attention 投影注入 LoRA（rank=8）；`adapter="mlp"` 时在指定层输出后接 2 层 MLP 残差 Adapter。只训 Adapter，参数量 ≪ 330M。缓解 R-1：曲风适配不足时回退方案 = 微调 MERT 后 6 层。
- **特征抽取**：取第 `layer=12` 层隐藏态，按 MERT 25Hz 帧率输出。验证浅层节奏 / 深层旋律的层次假设（奠基 §3.1.1 优势 2）。
- **Demucs 可选分支**：`use_demucs=True` 时先 `DemucsSeparator.separate` 得四轨 → 每轨过同一 MERT（权重共享轻量 encoder）→ 主轨与四轨 embedding 沿 feat 维拼接 → 线性投影回 768。`use_demucs=False` 时直接返回主轨 embedding（奠基：模型可借注意力隐式分离）。
- **长音频处理**：以固定 hop 滑窗（窗口 ≈ 5s，重叠 1s，奠基 §3.1.1），对重叠区取平均，避免边界跳变。
- **日志**：统一 `from beatmorph.core.logging import get_logger`。

## 5. 依赖关系

- **上游**：数据预处理流水线（plan 08）提供 16kHz WAV。
- **下游**：Stage 1 密度规划（plan 03）消费 `audio_emb`；Stage 2 AR（plan 04）以 Cross-Attention 消费同一 embedding；RAG（plan 05）用其作检索特征。
- **外部库**：`transformers`（MERT）、`torchaudio`（重采样到 16kHz）、`demucs`（可选）、`torch>=2.5`。

## 6. 里程碑与验收标准

对齐奠基 §7 Phase 1（0-3 月）。

| 里程碑 | 验收（可量化） |
|--------|---------------|
| M1 MERT 离线推理通路 | 1 首完整曲 → `[B, T_seq, 768]`，`T_seq ≈ dur*25`（±1 帧），FP16 在 T4 单卡 < 实时倍率 0.5× |
| M2 Adapter 训练就绪 | Adapter 参数可训、MERT 冻结 grad 为 0；小样本回归 loss 下降 |
| M3 10K 首 embedding 离线提取 | 对应奠基 §7 Phase1 里程碑「10K 首 osu! 谱面 MERT Embedding 离线提取」，落盘 Parquet，校验和无损 |
| M4 Demucs 可选通路 | `use_demucs=True/False` 两路径输出形状一致；分离耗时记录入 W& |

## 7. 风险与缓解

| 风险 | 奠基编号 | 缓解 |
|------|---------|------|
| MERT 冻结对重型电子等曲风表征不佳 | R-1 | Adapter 可训；严重时微调后 6 层（奠基原文） |
| Demucs 引入显著延迟/显存 | §3.1.2 | 设为可选标志；快速推理跳过；仅训练时开启 |
| 长音频边界跳变 | （派生） | 滑窗 + 重叠区平均；验收 M1 校验帧数与拼接连续性 |
| 16kHz 重采样不一致 | （派生） | 统一在流水线入口重采样并断言采样率 |

## 8. 测试策略

- **单元**：`tests/unit/audio/test_mert.py`——形状断言、帧率断言（`T_seq == round(dur*25)`）、冻结参数断言、Adapter 可训参数集断言；`test_demucs.py`——四轨键名与形状、`use_demucs` 开关两路径形状一致。
- **集成**：`tests/integration/test_audio_to_plan.py`——`MERTAdapter.encode` 输出直接喂 `DensityPlanner.plan`，端到端不报错且 Section 时间落在音频时长内。
- **e2e**：离线脚本对 10K 曲批量提取，产出 Parquet，抽样人工核对首尾帧时间戳。

## 9. 开放问题

- [ ] RFC-0002：长音频滑窗策略（固定 5s 窗 vs 动态按段落）——奠基 §3.1.1 仅给「5 秒上下文窗口」，本计划暂定滑窗，待 RFC 定稿。
- [ ] RFC-0003：Adapter 选型 LoRA vs MLP 的最终默认值——奠基 §3.1 / §2 同时提及两者，需实验对比后定型。
- [ ] Demucs 分轨特征融合：沿 feat 拼接后投影回 768 是否优于加权求和，需消融（§3.1.2 仅给拼接）。
