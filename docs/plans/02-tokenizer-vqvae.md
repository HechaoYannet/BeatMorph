# Plan 02 — 谱面语义 Tokenizer（VQ-VAE）

> 状态：🟡 草案 ｜ 阶段：Phase 1 ｜ 负责：Tokenizer 组
> 对应代码：`beatmorph/tokenizer/vqvae.py` ｜ 对应奠基章节：§3.2

## 1. 目标与范围

### 交付
- `VQVAETokenizer`：将一小节内的 Note 集合量化为单个离散 Token（码本 2048 基础 / 4096 精细）。
- 完整 `encode(Chart)->list[PatternToken]` / `decode(list[PatternToken])->Chart` 双向通路。
- 码本利用率监控 `codebook_usage()`，对应验收「重建准确率 > 95%」（奠基 §7 Phase1）。
- 训练损失：重建损失 + commitment loss + 码本利用率约束；K-means 初始化防坍缩（R-2）。

### 不交付
- AR Transformer 生成主干（plan 04）。
- 解码后的物理后处理与格式 Writer（plan 07）。
- FSQ / 连续 VAE 路径（奠基 §3.2.3 已否决）。

### 价值
把谱面压成「语义类型」离散序列，使下游 AR 自回归成为标准 VQ-VAE+GPT 范式（奠基 §3.2.3 业界标准）。每个 Code Index 表征密度等级/节奏型/手型倾向/键位空间分布（§3.2.2），是 Stage2 的语义原子。

## 2. 与奠基文档的对应

| 本计划项 | 奠基依据 | 偏离 | 理由 |
|----------|---------|------|------|
| 标准 VQ-VAE | §3.2.1 | — | 直接采用 |
| 时间粒度 1 小节（BPM 动态对齐） | §3.2.1 表 | — | 对应 `PatternToken.bar_index`、`duration_bars=1` |
| 码本 2048/4096 | §3.2.1 表 / 契约 `CODEBOOK_BASE/FINE` | — | 默认 2048，4096 为精细档 |
| 编码器 1D-CNN + Transformer | §3.2.1 表 | — | — |
| 码本语义 4 维度 | §3.2.2 | — | 作为可解释性监控，不强制分类头 |
| 损失含 commitment loss | §3.2.1 / R-2 | — | — |
| K-means 初始化防坍缩 | R-2 | — | 直引风险表缓解策略 |
| 随机重启死码 | R-2 | — | 训练中对长期未激活 code 重置为当前 batch 样本 |
| 不用 FSQ | §3.2.3 | — | FSQ 适合波形压缩，非语义抽象 |

**唯一偏离**：奠基 §3.2.1 表写「训练数据 50K+」，§4.1 写「100 万+」；本计划以 §7 Phase1「跑通」为准，先 50K 起步，扩展至 §4.1 量级。记入 §9。

## 3. 接口契约

### 3.1 VQVAETokenizer（对应 `beatmorph/tokenizer/vqvae.py`）
```python
from beatmorph.core.contracts import Chart, PatternToken, CODEBOOK_BASE

class VQVAETokenizer:
    def __init__(self, codebook_size: int = CODEBOOK_BASE, bars_per_token: int = 1) -> None: ...
    def encode(self, chart: Chart) -> list[PatternToken]: ...        # 按小节顺序
    def decode(self, tokens: list[PatternToken]) -> Chart: ...        # Stage3 调用
    def codebook_usage(self) -> float: ...                            # 0-1，监控 R-2
```
`PatternToken(code, bar_index, start_time, duration_bars)` 字段与骨架完全一致；`code ∈ [0, codebook_size)`。

### 3.2 跨模块张量形状（einops 风格）
| 名称 | 形状 | 含义 |
|------|------|------|
| 小节 Note 表示（编码前） | `(batch, bars, lane, time_bins, feat)` | 一小节 Note 栅格化输入 |
| 量化前向量 | `(batch, bars, latent)` | encoder 输出 |
| 码本 | `(codebook_size, latent)` | 可学习离散表 |
| `TokenSeq` | `(batch, seq_len)` long | Pattern ID，引用契约 `TokenSeq`，`codebook_size=2048` |
| 解码 Note 概率 | `(batch, bars, lane, time_bins, note_type)` | decoder 输出，argmax 得 Note |

## 4. 内部设计

- **栅格化**：按小节（BPM 动态对齐）将 `Note` 投到 `(lane, time_bins)` 网格，`time_bins` 按 16 分音或更细；保留 HOLD/ROLL 的 duration 维。时间用秒（契约 §3.1 `Note.time` 秒）对齐音频轴。
- **编码器**：1D-CNN（沿 time_bins）+ 轻量 Transformer，输出每小节一个 `latent` 向量。
- **量化**：最近邻查表 `argmin ‖z - e_k‖²`，直通估计器（straight-through）传梯度；记录未激活 code。
- **解码器**：轻量 MLP + 1D-CNN（奠基 §3.7「轻量 MLP + 1D-CNN」反向映射回 `(time, lane, type, duration)`）。
- **损失**：`L = L_recon + β·L_commit + γ·L_util`。`L_recon` 为 Note 栅格 CE/BCE；`L_commit` 拉近 encoder 输出与码本；`L_util` 软性促进码本熵（防坍缩）。
- **防坍缩三连（R-2）**：① K-means 初始化码本（用首批样本聚类中心）；② 增大 commitment loss 权重；③ 长期死码随机重启为当前 batch 向量。
- **可解释性监控**：按 §3.2.2 四维度（密度/节奏/手型/键位分布）离线统计每 code 的平均分布，写入 W&，非训练约束。

## 5. 依赖关系

- **上游**：数据预处理流水线（plan 08）提供 `Chart`（含 `.osu`→Note 解析）；奠基 §4.2 Step 1-2。
- **下游**：AR 生成（plan 04）消费 `TokenSeq`；解码后处理（plan 07）调用 `decode`。
- **外部库**：`torch>=2.5`、`einops`、`pydantic>=2.5`（契约）、`torchaudio`（仅栅格化时间对齐用）。

## 6. 里程碑与验收标准

对齐奠基 §7 Phase 1。

| 里程碑 | 验收（可量化） |
|--------|---------------|
| M1 编/解码往返 | `decode(encode(chart))` 形状合法、Note 数量误差 < 2% |
| M2 重建准确率 | 奠基 §7 Phase1 验收：**重建准确率 > 95%**（Note time±20ms、lane 完全匹配的比例） |
| M3 码本健康 | `codebook_usage() ≥ 0.5`（2048 码本至少半数活跃，R-2 量化红线） |
| M4 50K 训练跑通 | 在 50K osu! 谱面（§3.2.1）上训练收敛，损失曲线写入 W& |

## 7. 风险与缓解

| 风险 | 奠基编号 | 缓解 |
|------|---------|------|
| 码本坍缩（Index Collapse） | R-2 | K-means 初始化 + 增大 commitment loss + 死码随机重启（奠基原文三连） |
| 多 BPM / 变速小节边界不准 | （派生） | 严格按 BPM 计算小节边界；变速曲记入 §9 |
| 重建 > 95% 难达 | §7 Phase1 | 先 2048 码本，若不达标升 4096 精细档（契约支持） |
| HOLD/ROLL 还原丢 duration | （派生） | 栅格化显式保留 duration 维，decode 后由后处理校验合法性 |

## 8. 测试策略

- **单元**：`tests/unit/tokenizer/test_vqvae.py`——`encode/decode` 往返形状、`code ∈ [0, codebook_size)`、空小节（无 Note）能编码、确定性（同输入同输出）。
- **集成**：`tests/integration/test_chart_token_roundtrip.py`——真实 `.osu` 解析为 `Chart` → 编码 → 解码 → 与原 Chart 比较 Note 匹配率 > 95%。
- **e2e**：50K 谱面批量训练后，抽样 100 首：重建匹配率均值 > 95%、码本利用率 ≥ 0.5；结果入 W&。

## 9. 开放问题

- [ ] RFC-0004：训练数据起步量 50K（§3.2.1）vs 100 万+（§4.1）——本计划以 §7 Phase1为先，待与数据组（plan 08）对齐。
- [x] RFC-0005：变速曲目的 `bpm` 字段——已采纳 `bpm_points`（见 [RFC-0005](../decisions/RFC-0005-bpm-timepoints.md)），小节栅格化按分段 BPM 推边界。
- [ ] 码本 2048 vs 4096 的最终生效档位：先 2048 达标则不升，待 M2 数据决定。
