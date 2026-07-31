# Plan 03 — Stage 1 全局密度规划模块

> 状态：🟡 草案 ｜ 阶段：Phase 1 ｜ 负责：规划组
> 对应代码：`beatmorph/planner/density.py` ｜ 对应奠基章节：§3.3

## 1. 目标与范围

### 交付
- `DensityPlanner`：6 层双向 Transformer，自监督回归每 4 小节（一个 `Section`）的 `density_target / energy_level / rest_probability` 与段落类型。
- `plan(audio_emb, difficulty, style_emb=None) -> list[Section]` 通路。
- 训练损失：Huber Loss（鲁棒）+ TV Loss（相邻段落平滑，奠基 §3.3 原文）。
- 伪标签生成器：从 osu! `Chart` 自动统计每段密度/能量/休息概率，零人工标注。

### 不交付
- AR Pattern 生成（plan 04）——本模块仅出「宏观施工蓝图」。
- RAG 风格检索（plan 05）——本模块消费 `style_emb` 但不实现检索。
- 规则硬编码规划（奠基 §8 决策矩阵已否决，选自监督回归）。

### 价值
解决 AR 模型「目光短浅」问题（奠基 §3.3）：在生成 Pattern Token 前提供宏观蓝图，保证副歌高潮、前奏铺陈、尾奏收束的结构合理性。自监督回归零标注，数据来自 §4.2 自动统计量。

## 2. 与奠基文档的对应

| 本计划项 | 奠基依据 | 偏离 | 理由 |
|----------|---------|------|------|
| 6 层双向 Transformer | §3.3 表 | — | 直接采用 |
| 输入 audio_emb + difficulty + style_emb | §3.3 表 / §2 架构图 | — | — |
| 输出每 4 小节 density/energy/rest + 段落类型 | §3.3 表 | 微扩展 | 契约 `Section` 含 `sections_type`，本模块一并回归（奠基「段落类型」未列回归目标，但 Section 已承载该字段） |
| 自监督回归 + 零标注 | §3.3 / §8 决策矩阵 | — | 伪标签来自 §4.2 统计量 |
| Huber Loss + TV Loss | §3.3 表 | — | 直接采用 |
| 每 4 小节一个 Section | §3.3 / 契约 `PlanOutput.section_bars=4` | — | 与契约一致 |

**唯一偏离**：奠基 §3.3 列三个回归量但未把「段落类型」列为回归目标；本计划新增 `sections_type` 分类头（契约 `Section.sections_type` 已存在该字段）。理由：蓝图需含 intro/verse/chorus/bridge/outro 才能指导结构合理性。记入 §9。

## 3. 接口契约

### 3.1 DensityPlanner（对应 `beatmorph/planner/density.py`）
```python
from beatmorph.core.contracts import Section

class DensityPlanner:
    def __init__(self, n_layers: int = 6, section_bars: int = 4) -> None: ...

    def plan(
        self,
        audio_emb: "torch.Tensor",          # [B, T_seq, 768]
        difficulty: int,                     # 1-15
        style_emb: "torch.Tensor | None" = None,  # RAG 风格向量，可选
    ) -> list[Section]: ...
```
返回每个 `Section` 的 `density_target/energy_level/rest_probability`（均 ∈[0,1]，由契约 pydantic `Field(ge=0,le=1)` 强制）与 `sections_type`。

### 3.2 跨模块张量形状（einops 风格）
| 名称 | 形状 | 含义 |
|------|------|------|
| `audio_emb` 输入 | `(batch, time_seq, 768)` | Stage0 输出，契约 `AudioEmbedding` |
| difficulty embedding | `(batch, 64)` | 标量 1-15 → 嵌入 |
| `style_emb` 输入 | `(batch, top_k, 768)` | RAG，契约 `RAGContext`，可空 |
| 规划输出 | `(batch, num_sections, 3)` | density/energy/rest，0-1 |
| 段落类型 logit | `(batch, num_sections, 5)` | intro/verse/chorus/bridge/outro |
| 契约映射 | `PlanOutput` | `num_sections` ×3，`section_bars=4` |

## 4. 内部设计

- **Section 划分**：按 `section_bars=4` 小节切分音频时长（需 BPM，来自 `Chart.bpm_points`，见 RFC-0005；变速曲分段推小节），每段映射到对应 `audio_emb` 子区间并做 mean-pool 得段级向量。
- **编码器**：6 层双向 Transformer，输入 = 段级 audio 向量序列 + difficulty embedding（前缀或相加）+ style_emb（Cross-Attention 或前缀拼接）。
- **多任务头**：
  - 回归头：3 个独立 MLP 出 density/energy/rest，sigmoid 限 [0,1]。
  - 分类头：5 类段落类型 softmax。
- **损失**：`L = L_Huber(density) + L_Huber(energy) + L_Huber(rest) + λ_tv·TV + μ·CE(type)`。TV Loss = `‖ŷ_i − ŷ_{i-1}‖₁` 促平滑（奠基 §3.3「相邻段落平滑约束」）。
- **伪标签生成**：从 osu! `Chart` 统计——density = 段内 NPS / 全曲峰值 NPS；energy = 加权 Note 密度×HOLD 比例；rest = 间隔 > 阈值比例；sections_type = 启发式（密度低谷/outro 规则），零标注（奠基 §4.2 Step 2）。
- **日志**：统一 `from beatmorph.core.logging import get_logger`。

## 5. 依赖关系

- **上游**：Stage 0 音频编码（plan 01）提供 `audio_emb`；数据预处理（plan 08）提供 `Chart` 统计量伪标签。
- **下游**：Stage 2 AR（plan 04）以 Section 蓝图作条件；RAG（plan 05）提供 `style_emb`。
- **外部库**：`torch>=2.5`、`einops`、`pydantic>=2.5`（契约）。

## 6. 里程碑与验收标准

对齐奠基 §7 Phase 1。

| 里程碑 | 验收（可量化） |
|--------|---------------|
| M1 伪标签生成 | 10K Chart → Section 伪标签，密度/能量分布直方图入 W&，无 NaN |
| M2 训练收敛 | Huber Loss 收敛，TV 项使相邻段密度差方差 < 阈值 |
| M3 段落边界准确率 | 奠基 §7 Phase1 验收「段落边界预测准确率」——以伪标签为真值，段落类型 F1 > 0.6，密度 MAE < 0.15 |
| M4 端到端推理 | `plan(audio_emb, difficulty)` 返回 `list[Section]`，Section 时间覆盖音频全长且无重叠 |

## 7. 风险与缓解

| 风险 | 奠基编号 | 缓解 |
|------|---------|------|
| AR 目光短浅（本模块存在前提） | §3.3 原文 | 本模块即缓解手段：提供宏观蓝图 |
| 伪标签噪声（启发式 sections_type） | R-3 派生 | 仅作弱监督，Huber Loss 抗 outlier；DPO 阶段兜底（plan 06） |
| 模型学到坏习惯（低质谱面） | R-3 | 预处理严格过滤低评分谱面（§4.3），DPO 兜底 |
| 规划过度约束破坏创意 | R-5 派生 | 仅输出密度/能量概率，不干预具体键型（键型留给 AR + 后处理红线） |
| 多 BPM 变速段落划分错 | （派生） | 依赖 RFC-0005（变速 bpm 字段）；暂以主 BPM 切分，变速曲标注为例外集 |

## 8. 测试策略

- **单元**：`tests/unit/planner/test_density.py`——输出形状、值域 ∈[0,1]、Section 无重叠覆盖时长、difficulty 极端值（1/15）不崩、`style_emb=None` 可运行。
- **集成**：`tests/integration/test_audio_to_plan.py`——`MERTAdapter.encode` → `DensityPlanner.plan` 串联，Section 起止落在音频时长内（与 plan 01 共建）。
- **e2e**：50K 谱面训练后抽样 100 首：段落类型 F1 与密度 MAE 达 M3 阈值；副歌段密度显著高于前奏（结构合理性人工抽检）。

## 9. 开放问题

- [ ] RFC-0006：是否将 `sections_type` 纳入回归/分类目标——奠基 §3.3 仅列三个连续量，本计划暂新增分类头，待 RFC 定稿。
- [x] RFC-0005（依赖）：变速曲目的 Section 划分——核心契约已采纳 `bpm_points`（见 [RFC-0005](../decisions/RFC-0005-bpm-timepoints.md)），Section 划分按分段 BPM 推小节。
- [ ] 伪标签 `sections_type` 启发式规则集待与数据组（plan 08）联合定义，避免主观偏差。
