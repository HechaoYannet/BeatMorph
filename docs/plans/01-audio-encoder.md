# Plan 01 — Stage 0 音频编码器：MERT-v1-330M + LoRA Adapter

> 状态：🟡 草案 ｜ 阶段：Phase 1 ｜ 负责：音频组（audio-agent）
> 对应代码：`beatmorph/audio/encoder/mert.py`（改造）、`beatmorph/audio/encoder/cache.py`（新增） ｜ 对应奠基章节：§2、§3.1、§9

## 1. 目标与范围

### 交付

1. `MertAudioEncoder`：冻结 `m-a-p/MERT-v1-330M`（24 层，hidden = `MERT_DEFAULT_FEAT_DIM`），注入可训练 **LoRA Adapter**（RFC-0003：attention q/v），输出帧级序列 embedding `(batch, time_seq, feat)`。
2. **帧率是派生量**：`MERT_FRAME_RATE_HZ = MERT_SAMPLE_RATE_HZ / MERT_CONV_STRIDE_PRODUCT` = 75 Hz，禁止硬编码（红线 7 / [POSTMORTEM](POSTMORTEM-2026-08-05-frame-rate-misalignment.md)）。
3. **长音频滑窗**：5 s 窗 / 1 s 重叠，重叠区**按帧率对齐**平均（BasePlan §3.1）。
4. **特征缓存与元数据校验**（`beatmorph/audio/encoder/cache.py`）：落盘 `{rate, sample_rate, layer, model_rev, duration_s}`，**加载时逐项校验**（RFC-0029 §7-2）。
5. 离线批量提取脚本（供 plan 02 数据流水的 [7] 步复用），输出可被训练侧直接 `mmap` 读取。
6. Adapter 训练通路的 **G1–G4 门禁接入**（`beatmorph/infra/sanity.py`）。

### 不交付

- MERT 主干自身的训练 / 全量微调（仅冻结加载；R-1 恶化时的「解冻后 6 层」须另开 RFC）；
- 强度场、生成主干、解码与后处理（plan 03 / plan 04 / plan 05）；
- 音频**来源**与版权合规（plan 02 §6-M8 的合规门禁；本模块只消费已获准的音频）；
- Demucs 分轨（v3.0 未采用；若日后启用须另开 RFC——BasePlan §3.1 已不再包含该分支）。

### 价值

音频侧是整个范式的**唯一感知入口**：`audio_emb` 是强度场 λ_k(t, x, s, c | audio, 线事件轨, 难度) 的三个条件之一（BasePlan §3.5）。它的帧率是**全项目时间轴的基准刻度**——秒 → 帧的唯一换算发生在这里，因此它是红线 7 的第一道也是最重要的一道闸门。

## 2. 与奠基文档的对应

| 本计划项 | 奠基依据 | 偏离 | 理由 |
|----------|---------|------|------|
| MERT-v1-330M 冻结 + LoRA Adapter | [BasePlan §3.1](BasePlan.md) 表 | — | 直接采用；Adapter 细节依 [RFC-0003](decisions/RFC-0003-adapter-lora-vs-mlp.md)（采纳） |
| 采样率 `MERT_SAMPLE_RATE_HZ` = 24000 Hz | §3.1「24000 Hz（官方 preprocessor 要求）」 | 见偏离 2 | v2.x 曾按 16 kHz 描述，与 v3.0 冲突 |
| 帧率 = `MERT_SAMPLE_RATE_HZ / MERT_CONV_STRIDE_PRODUCT` | §3.1「7 层卷积，累乘 320 → 24000/320 = 75 Hz，**派生量，禁止硬编码**」 | — | 红线 7 的核心条款 |
| 特征维 = `MERT_DEFAULT_FEAT_DIM` = 1024 | §3.1「hidden = 1024」+ `core/contracts/tensors.py` | 见偏离 1 | RFC-0003 与旧 plan 写的 768 是 v2.x 误值 |
| LoRA 注入 attention q/v（rank=8 / alpha=16 / dropout=0.05） | [RFC-0003](decisions/RFC-0003-adapter-lora-vs-mlp.md)「提议」 | — | 已采纳；`peft` 目标模块名须按实际 introspect 确认 |
| 5 s 窗 / 1 s 重叠 + 重叠区平均 | §3.1「长音频」行 | — | 直接采用 |
| 缓存带 `{rate, sample_rate, layer, model_rev, duration_s}` 元数据并校验 | [RFC-0029 §6-2/§7-2](decisions/RFC-0029-phigros-continuous-chart-generation.md)、[phira-dataset-survey.md §9.2](knowledges/phira-dataset-survey.md) [7] | — | 缓存元数据是「25 Hz 事故」的防线 |
| 统一重采样到模型采样率、记录原始采样率与时长 | 调研 §9.2 [7] | — | 实测音频格式多样（.mp3/.ogg/.wav/.m4a/.flac/.opus） |
| 无 Demucs 分支 | §3.1 已无该行 | — | v3.0 音频侧只剩「MERT + Adapter」 |

**偏离 1（特征维 768 → 1024）**：[RFC-0003](decisions/RFC-0003-adapter-lora-vs-mlp.md) §32/§35 仍写 `feat=768` 与「`[B,T_seq,768]` 不变」，与 v3.0 BasePlan §3.1「hidden = 1024」及已入库的 `MERT_DEFAULT_FEAT_DIM = 1024` 冲突（后者来自 TRAINING_LOG Bug5 实测：MERT-v1-330M 各层均为 1024，768 系与 MERT-v1-95M 混淆）。→ 本 plan 取 **1024**；RFC-0003 需按 v3.0 口径修订（记入 §9-1）。LoRA 参数量随 feat 线性变化，不影响「轻量 Adapter」结论。

**偏离 2（输入采样率 16 kHz → 24 kHz）**：v2.x 的 plan 01 写「16 kHz 单声道」，v3.0 §3.1 明确 MERT preprocessor 要求 **24000 Hz**。→ 本 plan 以 `MERT_SAMPLE_RATE_HZ` 为唯一口径；**重采样只允许发生在流水线入口**，且必须在缓存元数据里同时记录**原始采样率**与**模型采样率**。

**偏离 3（取哪一层的 hidden state）**：BasePlan §3.1 只给「24 层，hidden = 1024，全部冻结」，**未指定取第几层**；已入库实现默认 `layer=12`。→ 本 plan 把 `layer` 保留为配置项，**不擅自改变默认值**，但要求在 §6-M6 的门禁里带一次层选择消融；层选择的最终值记入 §9-2。

## 3. 接口契约

### 3.1 常量（派生式；数值定义在 `core/contracts/tensors.py`，本模块只引用）

```python
from beatmorph.core.contracts import (
    MERT_SAMPLE_RATE_HZ,        # 24000（特征提取器要求，唯一来源）
    MERT_CONV_STRIDE_PRODUCT,   # 320 = prod(conv_stride) = prod([5,2,2,2,2,2,2])
    MERT_FRAME_RATE_HZ,         # = MERT_SAMPLE_RATE_HZ / MERT_CONV_STRIDE_PRODUCT = 75.0（派生量）
    MERT_DEFAULT_FEAT_DIM,      # 1024（MERT-v1-330M hidden_size，各层相同）
)

# ── 本模块的结构参数（不是物理常量；物理量一律派生）──
MERT_MODEL_ID: str = "m-a-p/MERT-v1-330M"
MERT_WINDOW_SECONDS: float = 5.0             # BasePlan §3.1「5s 窗」
MERT_WINDOW_OVERLAP_SECONDS: float = 1.0     # BasePlan §3.1「1s 重叠」
MERT_WINDOW_FRAMES: int = round(MERT_WINDOW_SECONDS * MERT_FRAME_RATE_HZ)                          # 派生
MERT_WINDOW_HOP_FRAMES: int = round((MERT_WINDOW_SECONDS - MERT_WINDOW_OVERLAP_SECONDS) * MERT_FRAME_RATE_HZ)  # 派生
MERT_WINDOW_SAMPLES: int = round(MERT_WINDOW_SECONDS * MERT_SAMPLE_RATE_HZ)                        # 派生

LORA_RANK: int = 8            # RFC-0003（采纳）：rank=8 / alpha=16 / dropout=0.05
LORA_ALPHA: int = 16
LORA_DROPOUT: float = 0.05
LORA_TARGET_MODULES: tuple[str, ...] = ("query", "value")   # peft 默认名；须按实际模块名 introspect 确认
```

> **禁止**在本模块出现 `75` / `320` / `24000` 的数值字面量（除 import 与注释）；帧数、窗口长度、hop 长度**全部由上面派生式给出**。§6-M1 用源码扫描测试守住这条。

### 3.2 编码器

```python
class MertAudioEncoder(torch.nn.Module):
    def __init__(
        self,
        model_id: str = MERT_MODEL_ID,
        layer: int = 12,                 # 见 §2 偏离 3（配置项，非派生量）
        adapter: str = "lora",           # "lora" | "mlp" | "none"（RFC-0003）
        lora_rank: int = LORA_RANK,
        device: str | None = None,
        dtype: str = "float16",
    ) -> None: ...

    def encode(self, wav: "torch.Tensor", sample_rate: int) -> "torch.Tensor":
        """wav: (batch, samples) → (batch, time_seq, MERT_DEFAULT_FEAT_DIM)。

        前置断言：sample_rate == MERT_SAMPLE_RATE_HZ（**不静默重采样**，重采样由流水线入口负责）。
        后置断言：time_seq == round(samples / MERT_SAMPLE_RATE_HZ * MERT_FRAME_RATE_HZ)（±1 帧）。
        """

    def trainable_parameters(self) -> int: ...   # 仅 LoRA 参数；主干恒为 0
```

**契约要点**

- 主干全部 `requires_grad_(False)`；`adapter="lora"` 时 `peft.get_peft_model` 包装，仅 LoRA 矩阵可训；`adapter="none"` 用于离线提取；`adapter="mlp"` 保留为备选（RFC-0003）。
- 输出张量契约**引用** `beatmorph.core.contracts.AudioEmbedding`（`feat = MERT_DEFAULT_FEAT_DIM`、`hop_rate = MERT_FRAME_RATE_HZ`），不得在本模块重新定义形状。
- 输入张量契约：`(batch, samples)`，**单声道**、`MERT_SAMPLE_RATE_HZ`；多声道在入口处下混。
- 长音频：按 `MERT_WINDOW_SAMPLES` / `MERT_WINDOW_HOP_FRAMES` 切窗 → 逐窗编码 → 重叠区**按帧平均**（不是按样本平均）→ 拼接；尾窗不足一窗时右侧补零并在 `time_mask` 上标记（避免把补零区当成真实音频）。

### 3.3 特征缓存契约（`beatmorph/audio/encoder/cache.py`）

```python
CACHE_META_KEYS: tuple[str, ...] = ("rate", "sample_rate", "layer", "model_rev", "duration_s")

@dataclass(frozen=True)
class FeatureCacheMeta:
    rate: float          # 提取时的帧率，必须 == MERT_FRAME_RATE_HZ
    sample_rate: int     # 提取时的**模型**采样率，必须 == MERT_SAMPLE_RATE_HZ
    layer: int           # 取用的 hidden state 层号
    model_rev: str       # 主干 revision（HF commit / tag），非空
    duration_s: float    # **原始**音频时长（重采样前），秒
    # 以下为溯源扩展字段（调研 §9.2 [7]：记录原始采样率与时长）
    original_sample_rate: int
    feat_dim: int
    dtype: str
    adapter: str

def write_feature_cache(path: Path, emb: "np.ndarray", meta: FeatureCacheMeta) -> None: ...
def load_feature_cache(path: Path, expect: FeatureCacheMeta) -> "np.ndarray":
    """逐项校验 meta 与 expect；任一项不符 → 抛 CacheMetaMismatch（**不得降级为 warning**）。"""
```

**校验规则（加载时全部执行）**

| 检查 | 判据 | 失败后果 |
|------|------|---------|
| 帧率 | `meta.rate == MERT_FRAME_RATE_HZ` | 抛错（25 Hz 事故的同类防线） |
| 采样率 | `meta.sample_rate == MERT_SAMPLE_RATE_HZ` | 抛错 |
| 层号 | `meta.layer == 请求的 layer` | 抛错（换层即换特征空间） |
| 主干版本 | `meta.model_rev == 当前主干 revision` | 抛错（主干换版 = 特征分布漂移） |
| 时长一致 | `emb.shape[0] == round(meta.duration_s * meta.rate)`（±1 帧） | 抛错 |
| 特征维 | `emb.shape[1] == MERT_DEFAULT_FEAT_DIM` | 抛错 |

### 3.4 张量形状（einops 风格）

| 名称 | 形状 | 含义 |
|------|------|------|
| 输入波形 | `(batch, samples)` | 单声道，`MERT_SAMPLE_RATE_HZ` |
| 滑窗批 | `(num_windows, MERT_WINDOW_SAMPLES)` | 5 s 窗、1 s 重叠 |
| 单窗隐态 | `(num_windows, MERT_WINDOW_FRAMES, feat)` | `feat = MERT_DEFAULT_FEAT_DIM` |
| **`audio_emb`（契约）** | `(batch, time_seq, feat)` | `time_seq = round(duration_s * MERT_FRAME_RATE_HZ)`（±1 帧） |
| 时间有效掩码 | `(batch, time_seq)` | 尾窗补零区为 False（下游 `ChartField.time_mask` 与它对齐） |
| 缓存文件 | `.npz`（`emb`）+ `.meta.json`（`FeatureCacheMeta`） | 一个音频一份，按音频 hash 命名 |

### 3.5 不变量

| # | 不变量 |
|---|--------|
| A1 | `MERT_FRAME_RATE_HZ == MERT_SAMPLE_RATE_HZ / MERT_CONV_STRIDE_PRODUCT`（派生，非字面量） |
| A2 | `MERT_WINDOW_FRAMES == round(MERT_WINDOW_SECONDS * MERT_FRAME_RATE_HZ)`；hop 同理 |
| A3 | `time_seq == round(samples / MERT_SAMPLE_RATE_HZ * MERT_FRAME_RATE_HZ)`（±1 帧） |
| A4 | 主干参数 `requires_grad` 全为 False；可训参数集 == LoRA 参数集 |
| A5 | 缓存加载时六项校验全执行，任一项不符抛错 |
| A6 | 滑窗拼接后的帧数 == 全曲一次推理的帧数（±1 帧），且重叠区为两窗的逐帧均值 |

## 4. 内部设计

- **加载与冻结**：HuggingFace `transformers` 加载 `m-a-p/MERT-v1-330M` 与官方 preprocessor；`requires_grad_(False)` 冻结全部参数；以配置的 `revision` 固定主干版本（`model_rev` 落入缓存元数据）。
- **Adapter 注入**：`peft.LoraConfig(target_modules=LORA_TARGET_MODULES, r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT)`。**注入前必须 introspect 实际模块名**（RFC-0003 已提示 MERT 的 attention 子模块命名可能与 `query/value` 不一致）；introspect 结果写入日志与训练配置，不得静默 fallback 到「注入 0 个模块」。
- **特征抽取**：`output_hidden_states=True`，取第 `layer` 个 hidden state；断言 `layer` 落在 `[0, num_hidden_layers]`（含 embedding 层，共 n+1 项），越界即抛。
- **滑窗与拼接**：窗口长度与 hop **全部由帧率派生**（§3.1）；先按样本切、再按帧对齐；重叠区逐帧平均；尾窗右侧补零，补零区在 `time_mask` 上置 False。**禁止**用「样本数 // 320」这类手算替换派生式。
- **重采样边界**：本模块**只接受** `MERT_SAMPLE_RATE_HZ` 的输入并在签名处断言；重采样（`torchaudio` / `librosa`）发生在 plan 02 流水线入口，并把**原始**采样率与时长时间写入缓存元数据。
- **精度**：提取默认 FP16/BF16；缓存记录 `dtype`，加载时若 dtype 不同则显式转换并记日志（不静默混用）。
- **缓存布局**：按音频内容 hash 命名（不与谱面 id 耦合），使「同曲多谱」只提取一次；缓存目录在 `.gitignore` 黑名单内（红线 5：特征与权重不入库）。
- **日志**：`from beatmorph.core.logging import get_logger`，禁止裸 `print`。

## 5. 依赖关系

- **上游**：plan 02 数据流水线提供已解包的音频与元数据（并受其**合规门禁**约束）；`core/contracts`（plan 00）提供帧率/采样率/特征维常量与 `AudioEmbedding` 契约。
- **下游**：plan 03（强度场）与 plan 04（生成主干）消费 `audio_emb` 作为 cross-attention 条件（BasePlan §3.5）；plan 06（评估）消费同一缓存做可复现评估。
- **外部库**：`torch>=2.5`、`transformers`（MERT）、`peft>=0.11`（RFC-0003 新增依赖）、`torchaudio`/`librosa`（仅流水线侧重采样）、`numpy`。
- **契约符号复用**：`MERT_SAMPLE_RATE_HZ` / `MERT_CONV_STRIDE_PRODUCT` / `MERT_FRAME_RATE_HZ` / `MERT_DEFAULT_FEAT_DIM` / `AudioEmbedding` —— 只引用，不复制。

## 6. 里程碑与验收标准

| 里程碑 | 验收（可量化） |
|--------|---------------|
| **M1 帧率契约（默认 CI）** | `tests/unit/audio/test_mert_frame_rate.py` 全绿且**运行在默认 CI**（`-m "not slow and not gpu"`，**不加载权重、不需要 GPU**）：断言 A1/A2/A3（用 stub 主干或纯函数验证帧数公式）；同测试内含**源码扫描**——`beatmorph/audio/` 内不得出现帧率/采样率/累乘的字面量（`75` / `24000` / `320`）；破坏派生式（例如把 `MERT_CONV_STRIDE_PRODUCT` 改成 960）必须让测试**变红** |
| **M2 Adapter 契约（无权重）** | 用 toy `torch.nn.Module`（含 `query/value` 同名线性层）验证：注入后仅 LoRA 参数 `requires_grad`；主干参数 grad 全 0；`adapter="none"` 时零可训参数；模块名不匹配时**抛错**而非静默跳过（A4） |
| **M3 滑窗与拼接一致性** | A6：`duration_s ∈ {1, 4.9, 5.0, 5.1, 12.3, 300}` 的合成音频，滑窗拼接帧数与一次性推理帧数差 ≤ 1 帧；重叠区等于两窗逐帧均值（逐元素 assert）；不依赖 GPU 的纯数值路径 |
| **M4 缓存元数据与加载校验** | 写入-读取往返一致；**六项校验各有负样本**：篡改 `rate` / `sample_rate` / `layer` / `model_rev` / `duration_s` / `feat_dim` 任一字段后加载必须抛 `CacheMetaMismatch`（A5）；帧数不符（截断 emb）同样抛错 |
| **M5 真实权重通路（慢测）** | `@pytest.mark.slow`：单首短音频 → `[1, T, MERT_DEFAULT_FEAT_DIM]`，`T` 满足 A3；FP16 在 T4 上单曲提取 < 0.5× 实时；输出与 `adapter="none"` 路径形状一致 |
| **M6 离线批量提取 + 门禁** | 在**已获准**的音频子集上批量提取 → 缓存文件数 == 音频数，抽样加载校验全过；**Adapter 训练目标上线前必须 G1–G4 全绿**（`beatmorph/infra/sanity.py`）：G1 单 batch 过拟合 / G2 打乱标签对照 / G3 常数基线 / G4 契约断言（帧率与形状由 config 派生），结果写入训练日志；**门禁未绿不得扩大数据规模**，含一次 `layer` 选择消融（§2 偏离 3） |

> **合规前置（阻塞项）**：[RFC-0029 §8.3 Q11b](decisions/RFC-0029-phigros-continuous-chart-generation.md) 裁定前，本模块**只用公开可再分发的音频**（合成信号 + 自采）跑通 M1–M4、M6；真实语料的批量提取与 Adapter 训练**不得启动**（[BasePlan §4.4](BasePlan.md)）。

## 7. 风险与缓解

| 风险 | 编号 | 缓解 |
|------|------|------|
| MERT 对特定曲风（重型电子等）表征不足 | **R-1** | LoRA Adapter 可训（RFC-0003）；严重时「解冻后 6 层」须另开 RFC；本 plan 先把 M6 的层选择消融做掉 |
| 物理常量/帧率漂移（25 Hz 类） | **R-7** | 帧率派生 + A1–A3 + **M1 契约测试进默认 CI** + 源码字面量扫描；缓存元数据六项校验（A5） |
| 数据合规（音频随谱 100% 捆绑分发） | **R-2** | 本 plan 的真实语料提取与 Adapter 训练以合规裁定为前置门槛；缓存目录不入库（红线 5） |
| 主干版本漂移导致缓存失效 | 派生 | `model_rev` 入元数据并在加载时校验；换版本必须重抽，不得混用 |
| 静默重采样 / 声道数错配 | 派生 | `encode` 入口断言采样率与单声道；重采样只在流水线入口发生，原始采样率入缓存 |
| LoRA 目标模块名不匹配导致「注入 0 层」但训练照跑 | 派生 | 注入后断言可训参数 > 0 且层数 == introspect 命中数；命中 0 即抛错（M2） |
| 缓存与训练侧帧率口径不一致（下游自行换算） | **R-7** | 下游只允许通过 `AudioEmbedding.hop_rate` 换算；`ChartFieldSpec.dt` 由同一常量派生（plan 00 §3.7） |

## 8. 测试策略

- **单元（默认 CI，无权重、无 GPU）**
  - `tests/unit/audio/test_mert_frame_rate.py` — A1/A2/A3 与源码字面量扫描；**这是本 plan 的核心契约测试**，必须与其他非慢测试一起跑，不得打 `slow` 标记。
  - `tests/unit/audio/test_adapter_injection.py` — 用 toy 模块验证冻结/可训参数集/模块名不匹配抛错（A4）。
  - `tests/unit/audio/test_window_stitch.py` — 纯数值滑窗拼接（A6），不实例化 MERT。
  - `tests/unit/audio/test_feature_cache.py` — 元数据往返 + 六项负样本（A5）。
- **禁止事项**：mock **不得固化物理常量**——凡 mock 需要帧数或窗长，必须引用 `MERT_FRAME_RATE_HZ` / `MERT_WINDOW_FRAMES`（RFC-0029 §7-6；25 Hz 正是被 mock 洗成绿灯的）。
- **集成（无权重）**：`tests/integration/test_audio_to_field.py` — 假 `audio_emb`（形状由契约派生）经 `ChartFieldSpec` 对齐到时间轴，断言 `T_audio == T_field`；不依赖 MERT 权重。
- **慢测 / GPU**：`tests/integration/test_mert_real.py`（`@pytest.mark.slow` + `@pytest.mark.gpu`）加载真实权重跑 M5；不进默认 CI。
- **e2e（离线脚本）**：在合规允许的音频子集上跑批量提取，产出缓存 + 元数据清单；抽样将缓存与在线推理结果逐元素比对（容差按 dtype 设定），结果入训练日志。

## 9. 开放问题

1. **RFC-0003 的 `feat=768` 需修订**：该 RFC 文本与 v3.0 的 hidden = 1024 冲突（§2 偏离 1）。本 plan 取 1024；请在 RFC-0003 下次修订中同步，避免后续 agent 再按 768 写代码。
2. **取第几层 hidden state**：BasePlan §3.1 未指定；已入库实现默认 12。层选择对强度场条件质量的影响需由 §6-M6 的消融给出依据，在此之前不改默认值。
3. **LoRA 目标模块名**：RFC-0003 提示 MERT attention 子模块命名可能与 `query/value` 不一致，须在实现时 introspect 确认并把结果写回本 plan（当前 `LORA_TARGET_MODULES` 是**待验证**值）。
4. **滑窗是否必要**：若 `transformers` 路径能在显存内直接处理整曲（分段 attention），5 s/1 s 窗可能只是历史遗留。需在 M5 上做一次「滑窗 vs 整曲」的帧级一致性对照，再决定是否保留滑窗为唯一路径。
5. **缓存粒度**：按音频 hash 还是按 (音频 hash, layer, dtype) 建键？换层/换精度时是否重抽，取决于 §6-M6 消融的结论。
6. **`time_mask` 的消费方**：尾窗补零区掩码如何与 `ChartField.time_mask`（plan 00 §3.7）合并，需与 plan 03（强度场）对齐后再冻结字段名。
7. **AUX（辅助任务）**：是否给 Adapter 加 beats/onset 等辅助自监督头以提升 R-1 适配度，v3.0 未涉及；如要引入须先开 RFC（RFC-0029 未授权新增训练目标）。
