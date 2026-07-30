# Plan 08 — 训练数据策略与预处理流水线

> 状态：🟡 草案 ｜ 阶段：Phase 1 ｜ 负责：数据组
> 对应代码：`beatmorph/data/parsers/osu_path.py`、`beatmorph/data/pipeline/embed.py` ｜ 对应奠基章节：§4

## 1. 目标与范围

### 交付
- 数据源策略与质量过滤：osu! 100 万+ / StepMania 5 万+ / BMS / 合成数据 / DPO 偏好百万级（奠基 §4.1），筛选 `≥3 星且 play_count > 500`（§4.3）。
- 4 步自动化预处理流水线 `PreprocessPipeline(raw_dir, out_dir, cache_format='parquet').run()`（§4.2）：解析 `.osu`→NoteEvent[] 清理乱码 → 自动统计量做 Stage1 伪标签 → 音频切片 & MERT 离线预提取 Embedding → 构建 VQ-VAE 数据集，落 HF Datasets / Parquet。
- 解析器 `parse_osu(path) -> Chart` 与统计量 `compute_section_stats(chart, section_bars=4) -> Chart`。
- MERT 离线提取 `extract_mert_embeddings(audio_dir)`，对接 Plan 01。
- **Phase 1 里程碑**：10K 首 MERT Embedding 离线提取（奠基 §7）。

### 不交付
- MERT 模型本体与 Adapter 训练（Plan 01；本模块仅离线批量调用其 `encode`）。
- VQ-VAE 训练与 Stage1 伪标签回归训练（Plan 02 / Plan 03；本模块只产伪标签数据）。
- `.osu` 写入/导出（Plan 07 `Writer`；`Reader` 抽象归 io 层，本模块复用）。
- DPO 偏好对齐算法（Plan 06）；本模块仅产偏好对原始数据。
- 抓取与版权合规流程本身（§4.3 仅"学术研究"原则，工程外流程）。

### 价值
奠基 §1.2 范式转移核心是「自监督理解取代显式标注」——而自监督的燃料来自百万级现成谱面。本模块把原始 `.osu + .mp3` 加工成 (Chart + MERT Emb + 段落统计量伪标签 + 偏好对) 的训练就绪数据集，是 Phase 1 地基建设的基石，直接决定 Plan 01/02/03 能否起步。

## 2. 与奠基文档的对应

| 本计划项 | 奠基依据 | 偏离 | 理由 |
|----------|---------|------|------|
| 数据集 5 源构成（osu!/StepMania/BMS/合成/DPO） | §4.1 表 | — | 直接采用 |
| 4 步预处理流水线 | §4.2 流程图 | — | 直接采用，分别对应 4 步 |
| Step1 解析 .osu → NoteEvent[] 清理乱码 | §4.2 | 微语义 | 骨架用 `parse_osu()` 返回 `Chart`（契约类型），非裸 list；NoteEvent 即 `Chart.notes` |
| Step2 统计量 → Stage1 伪标签 | §4.2 | — | `compute_section_stats()` 回填 `Chart.sections` |
| Step3 音频切片 & MERT 离线预提取 | §4.2 | — | 节省训练算力，与 Plan 01 MERT 接口对齐 |
| Step4 构建 VQ-VAE 数据集存 HF/Parquet | §4.2 | — | `cache_format='parquet'` 默认 |
| 质量过滤 ≥3 星且 play_count>500 | §4.3 | — | 常量 `MIN_STARS=3.0`/`MIN_PLAY_COUNT=500` |
| 去重（同曲不同谱面为独立样本） | §4.3 | — | 按谱面级去重，非曲目级 |
| 版权仅学术研究 | §4.3 | — | 流程约束，非代码逻辑 |

**偏离 1**：奠基 §4.2 Step1 用「NoteEvent[]」与契约 `Chart` 术语不一致。本计划规定解析产出 `Chart`，NoteEvent 即 `Chart.notes`（契约 `Note` 列表），统一到核心契约 IR。记入 RFC-0020。
**偏离 2**：奠基 §4.2 未指定 `section_bars`。本计划固定 `section_bars=4`（与契约 `PlanOutput.section_bars=4`、Plan 03 一致），保证伪标签与 Stage1 输出同构。记入 RFC-0021。

## 3. 接口契约

### 3.1 复用骨架（`beatmorph/data/parsers/osu_path.py`、`beatmorph/data/pipeline/embed.py`）
```python
from pathlib import Path
from beatmorph.core.contracts import Chart, GameMode

def parse_osu(path: Path) -> Chart: ...                                 # §4.2 Step1
def compute_section_stats(chart: Chart, section_bars: int = 4) -> Chart: ...  # §4.2 Step2

class PreprocessPipeline:
    MIN_STARS: float = 3.0
    MIN_PLAY_COUNT: int = 500
    def __init__(self, raw_dir: Path, out_dir: Path, cache_format: str = "parquet") -> None: ...
    def run(self, limit: int | None = None) -> None: ...                # 端到端 4 步
    def extract_mert_embeddings(self, audio_dir: Path) -> None: ...      # §4.2 Step3
```
- `parse_osu` 内部委托 `beatmorph.io.formats.OsuManiaReader.read`（Plan 07 抽象，本模块复用），并叠加乱码/非标准 Note 清理。
- `compute_section_stats` 回填 `Chart.sections`（`density_target/energy_level/rest_probability/sections_type`），作为 Plan 03 自监督回归的伪标签。
- 质量过滤常量 `MIN_STARS=3.0`、`MIN_PLAY_COUNT=500`（§4.3）。
- StepMania 经 `SmReader` 同路径接入，统一为 `Chart`。

### 3.2 数据集样本张量形状（einops 风格）
| 名称 | 形状 | 含义 |
|------|------|------|
| `Chart.notes`（IR） | `(N_notes,)` per `(time,lane,type,duration)` | §4.2 Step1 产物 |
| MERT Emb（预提取） | `(sample, time_seq, 768)` | §4.2 Step3，`time_seq=秒数×25`，引用契约 `AudioEmbedding` |
| Sections 伪标签 | `(num_sections, 3+1)` | density/energy/rest + type，§4.2 Step2 |
| VQ-VAE 训练样本 | `(batch, audio_emb, chart_ir)` | §4.2 Step4 组装 |
| `TokenSeq`（产物后） | — | 经 Plan 02 encode 得到，引契约 `TokenSeq` |

## 4. 内部设计

- **Step1 解析**：`parse_osu(path)` → `OsuManiaReader.read` 得 `Chart`；清理规则——丢弃负时间 Note、`lane` 越界丢弃、同名文件 0 字节跳过、编码乱码以 cp936/utf-8 兜底重试。StepMania 走 `SmReader` 同构。
- **Step2 统计量（伪标签）**：`compute_section_stats(chart, section_bars=4)` 按 BPM 切每 4 小节一个 `Section`；density = 段内 NPS / 全曲峰值 NPS；energy = Note 密度加权×HOLD 比例；rest = 间隔 > 阈值比例；sections_type = 启发式（intro/outro/低谷规则）。回填 `Chart.sections`，供 Plan 03 直接读作监督信号。与 Plan 03 伪标签生成器共口径，避免语义漂移。
- **Step3 MERT 离线提取**：音频按切片策略切段 → Plan 01 MERT `encode`（FP16）→ `(time_seq, 768)` 写盘；切片对齐到 `Section` 边界便于下游。`extract_mert_embeddings(audio_dir)` 批量产出，单次产出多次训练复用（§4.2「节省训练时算力」）。
- **Step4 构建 VQ-VAE 数据集**：`(audio_emb, Chart, sections, 偏好对?)` 序列化为 HF `Dataset`，`cache_format='parquet'` 落地，支持流式 / 随机访问。
- **质量过滤（§4.3）**：`difficulty_rating` ≥ `MIN_STARS=3.0` 且 `playcount > 500`；同曲目不同谱面为独立样本保留；DPO 偏好对取 `≥4.5星高Pass` vs `≤2星` 构造（§3.6 数据来源）。
- **版权**：训练数据仅学术研究用途；元数据记录来源与许可，生成产物不含原音频拷贝（§4.3）。
- **去重口径**：谱面级（chart_id）去重，曲目级保留多谱面以保多样性（§4.3）。

## 5. 依赖关系

- **上游**：外部数据源（osu! / StepMania / BMS / 合成）；Plan 01 MERT `encode`（Step3 复用提取）；Plan 07 `OsuManiaReader`/`SmReader` 抽象（解析复用）。
- **下游**：Plan 01（MERT Emb 验证）、Plan 02（VQ-VAE 训练数据，50K+ @ §3.2.1）、Plan 03（Stage1 伪标签，§4.2 Step2 直接对接）、Plan 04（AR 训练 50K→扩展至 §4.1 量级）、Plan 06（DPO 偏好对）。
- **外部库**：`datasets`(HuggingFace)、`pyarrow`、`librosa`（音频切片/重采样）、`transformers`（MERT 加载）、`pydantic>=2.5`（契约）。

## 6. 里程碑与验收标准

对齐奠基 §7 Phase 1。

| 里程碑 | 验收（可量化） |
|--------|---------------|
| M1 `.osu` 解析 | 1K 样本 `parse_osu` 成功率 > 98%（失败仅因文件损坏），`Chart.notes` 无负时间/越界 |
| M2 段落统计量 | `compute_section_stats` 输出 Section 无重叠覆盖时长，密度/能量分布直方图入 W&B，无 NaN |
| M3 **10K MERT 离线提取** | 基奠 §7 Phase1 里程碑：10K 首 Embedding 落盘，`(sample, T_seq, 768)` shape 一致，耗时与显存记录入档 |
| M4 质量过滤与数据集 | `≥3星且play_count>500` 过滤后样本量与分布入档；VQ-VAE 训练集（audio_emb+Chart）以 Parquet 落地，可被 Plan 02 直接 `load_dataset` |

### 数据源分工（奠基 §4.1 表落细）
| 数据源 | 量级 | 接入 | 用途 |
|--------|------|------|------|
| osu! 官方/社区 | 100 万+ | `OsuManiaReader`（主） | VQ-VAE + AR 主训练；DPO 评分隐式标签 |
| StepMania/Etterna | 5 万+ | `SmReader` | VSRG 变体补充 |
| BMS | 大量 | 待封装 Reader（Phase 2+） | 复杂谱面逻辑验证 |
| 合成数据(DAW) | 无限 | 内部生成器 | 长尾拍号覆盖 |
| DPO 偏好对 | 百万级 | osu! 评分埋点 | Plan 06 偏好微调 |
Phase 1 聚焦 osu! 主链路（M1-M4 全在 osu!），其余源按 §7 R-6 节奏在后续 Phase 接入。

### 质量过滤执行（§4.3）
`run()` 内对每样本读 `difficulty_rating`/`playcount` 元数据：双条件 `≥MIN_STARS=3.0 and >MIN_PLAY_COUNT=500` 任一不满足即剔；阈值边界值纳入单元测试。去重按 `chart_id`（谱面级），同曲目多谱面保留以提高多样性。版权标记写入 `Chart.meta={"license":"academic"}`，生成产物不含原音频拷贝。

## 7. 风险与缓解

| 风险 | 奠基编号 | 缓解 |
|------|---------|------|
| 谱面质量参差，模型学坏习惯 | R-3 | §4.3 严格过滤 ≥3星且 play_count>500；Plan 06 DPO 兜底 |
| 模式扩展差异大（6K/osu!std） | R-6 | 4K 优先，StepMania/BMS 仅做 VSRG 变体补充，扩展需重写 Tokenizer |
| 标注缺失（自监督前提） | §1.2 范式 | 自动统计量作伪标签，零人工标注（Step2） |
| 乱码/非标准 .osu 解析失败 | 派生 | 多编码兜底重试 + 失败样本记日志不中断；失败率纳入 M1 验收 |
| MERT 离线提取算力/显存 | R-4 派生 | FP16 + 切段 + 断点续抽；离线一次产出多次复用（§4.2 原文「节省训练时算力」） |
| 版权合规 | §4.3 | 仅学术研究；元数据记来源许可；产物不含原音频拷贝 |

## 8. 测试策略

- **单元**：`tests/unit/data/test_parse_osu.py`——正常/乱码/损坏/越界 lane/负时间各构造样本，`parse_osu` 成功率与清理正确性；`MIN_STARS`/`MIN_PLAY_COUNT` 过滤边界值。`test_compute_section_stats.py`——Section 无重叠覆盖时长、值域 ∈[0,1]、无 NaN、`section_bars=4` 一致性。
- **集成**：`tests/integration/test_pipeline_run.py`——小规模 `raw_dir` 端到端 `run(limit=N)`，产出 Parquet 可被 `datasets.load_dataset` 读回，(audio_emb, Chart, sections) 字段齐全；`extract_mert_embeddings` mock Plan 01 `encode` 输出 shape 校验。
- **e2e**：10K 规模真实跑通（M3），产出 Embedding 抽样与在线 MERT 调用逐样本比对（误差 < 容差）；标注 `@pytest.mark.e2e`、`@pytest.mark.gpu`、`@pytest.mark.slow`。

## 9. 开放问题

- [ ] RFC-0020：解析产出用 `Chart` IR 统一 vs 沿用奠基「NoteEvent[]」术语——本计划已统一到契约，待 RFC 定稿确认。
- [ ] RFC-0021：`section_bars` 默认 4 是否随曲风/拍号动态调整（奠基未指定）。
- [ ] RFC-0004（依赖）：训练数据起步量 50K（§3.2.1）vs 100 万+（§4.1）的路线——与 Plan 02 对齐，本计划先 10K 提取为 Phase1 里程碑。
- [ ] RFC-0005（依赖）：变速曲目 `bpm` 字段扩展（Plan 02/03 共提）直接影响 `compute_section_stats` 的小节切分，本模块伪标签同受影响。
- [ ] DPO 偏好对构造的"高/低星"阈值（§3.6 的 4.5/2 星）与 §4.3 过滤阈值的协同口径待与 Plan 06 联合定义。
