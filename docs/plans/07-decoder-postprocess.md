# Plan 07 — Stage 3 & 4 解码、后处理与格式导出

> 状态：🟡 草案 ｜ 阶段：Phase 2 ｜ 负责：解码后处理组
> 对应代码：`beatmorph/decoder/postprocess/constraints.py`、`beatmorph/io/formats/` ｜ 对应奠基章节：§3.7

## 1. 目标与范围

### 交付
- Stage 3 解码编排：调用 `VQVAETokenizer.decode(tokens) -> Chart`，将离散 Pattern Token 映射回精确 `(time, lane, type, duration)` Note（奠基 §3.7「轻量 MLP + 1D-CNN」反向映射）。
- Stage 4 物理约束引擎 `PostProcessor(mode)`：硬编码红线规则——4K 单帧最大按键 ≤ 2、同手最小间隔 ≥ 70ms、禁止越界、长条 duration 合法性。`apply(chart) -> Chart` 修正物理不可达项；`validate(chart) -> list[str]` 只报告不修正。
- 格式导出层：`ChartWriter`/`ChartReader` 抽象基类（`io/formats/base.py`）+ `OsuManiaWriter`/`OsuManiaReader`（`.osu`）与 `SmWriter`/`SmReader`（`.sm`）。
- 端到端通路：`Token -> decode -> PostProcessor.apply -> Writer.write -> .osu/.sm`。

### 不交付
- VQ-VAE 训练与码本本体（Plan 02；`decode` 权重由 Plan 02 训练，本模块仅编排调用）。
- AR Pattern 生成（Plan 04，产出上游 `TokenSeq`）。
- 数据预处理侧的大量 `.osu` 批量解析逻辑（Plan 08；`Reader` 抽象归 io 层供其复用，但解析增强与统计量在 `data/parsers`）。
- 6K / osu!std / `.ma2`(maimai) 扩展格式（Phase 4，见 R-6、§9）。
- DPO 偏好对齐（Plan 06）。

### 价值
奠基 §3.7 关键原则：**AI 负责创意生成，规则引擎负责物理安全与合法性，两者解耦，保证 100% 可玩**。本模块是系统的「最后一公里」——把 AR 产出的语义 Token 落地为人类可击打、客户端可载入的谱面文件，是 Phase 2 里程碑「输出首版可玩 `.osu`，内部盲测」（奠基 §7）的唯一出口。

## 2. 与奠基文档的对应

| 本计划项 | 奠基依据 | 偏离 | 理由 |
|----------|---------|------|------|
| VQ-VAE Decoder（轻量 MLP+1D-CNN）映射回 `(time,lane,type,duration)` | §3.7 表 | — | 直接采用，重训在 Plan 02 |
| 物理约束引擎=规则系统硬编码 | §3.7 表 | — | 直接采用 |
| 4K 单帧最大按键 ≤ 2 | §3.7 表 | — | 常量 `PostProcessor.MAX_SIMULTANEOUS_4K=2` |
| 同手最小间隔 ≥ 70ms | §3.7 表 | — | 常量 `MIN_SAME_HAND_GAP_MS=70` |
| 禁止越界 | §3.7 表 | — | 用 `Chart.lane_count()` 与 `DEFAULT_LANE_COUNT=4` |
| 格式转换：`.osu`/`.sm`/未来 `.ma2` | §3.7 表 | — | `.osu`/`.sm` Phase 2，`.ma2` 预留 |
| AI 负责创意 / 规则负责物理，解耦 | §3.7 关键原则 | — | apply 严格红线，不触键型语义（R-5） |

**偏离 1**：奠基 §3.7「同手」未定义 4K 左右手 lane 分配。本计划规定左手 = `lane{0,1}`、右手 = `lane{2,3}`（约定俗成），同手间隔仅校验同侧两 lane。记入 RFC-0017。
**偏离 2**：奠基未拆 `apply` 与 `validate`。骨架已分离：`validate` 只报告违规（含 Note 索引/规则名），`apply` 修正后须使 `validate` 返回空列表；两者解耦便于盲测中"只提示不自改"。记入 RFC-0018。

## 3. 接口契约

### 3.1 复用骨架（`beatmorph/tokenizer/vqvae.py`、`beatmorph/decoder/postprocess/constraints.py`、`beatmorph/io/formats/`）
```python
from beatmorph.core.contracts import Chart, Note, NoteType, GameMode

class VQVAETokenizer:                       # decode 属 Stage3 入口（Plan 02 训练）
    def decode(self, tokens: list[PatternToken]) -> Chart: ...

class PostProcessor:
    MAX_SIMULTANEOUS_4K: int = 2
    MIN_SAME_HAND_GAP_MS: int = 70
    def __init__(self, mode: GameMode = GameMode.MANIA_4K) -> None: ...
    def apply(self, chart: Chart) -> Chart: ...        # 红线修正，不改键型语义（R-5）
    def validate(self, chart: Chart) -> list[str]: ... # 违规描述；空=合法可玩

class ChartWriter(ABC):
    def write(self, chart: Chart, path: Path) -> Path: ...
    def suffix(self) -> str: ...
class ChartReader(ABC):
    def read(self, path: Path) -> Chart: ...
class OsuManiaWriter(ChartWriter):  # suffix ".osu"
class OsuManiaReader(ChartReader):
class SmWriter(ChartWriter):        # suffix ".sm"
class SmReader(ChartReader):
```
`Note.time` 单位为秒（契约 00 §3.1），写入 `.osu`(毫秒整型)/`.sm`(beat 浮点) 时由 Writer 转换。

### 3.2 跨模块张量形状（einops 风格）
| 名称 | 形状 | 含义 |
|------|------|------|
| Token 输入 | `(batch, seq_len)` long | Pattern ID，引用契约 `TokenSeq`，`codebook_size=2048` |
| 解码前 latent | `(batch, bars, latent)` | 码本查表展开（decoder 内部） |
| 解码 Note 概率 | `(batch, bars, lane, time_bins, note_type)` | decoder 输出，argmax→Note |
| 还原 Note 流 | `(N_notes,)` per `(time,lane,type,duration)` | 组装为 `Chart.notes` |
| `Chart` IR | — | 与格式解耦，契约 `Chart` |

## 4. 内部设计

- **Stage 3 解码**：`decode(tokens)` 逐小节——码本查表 → 轻量 MLP+1D-CNN → 在 `(lane, time_bins)` 网格 argmax 还原 Note；时间 = `PatternToken.start_time` + 栅格偏移（秒）。`NoteType` 维 argmax 决定 TAP/HOLD/MINE/ROLL/FAKE；HOLD/ROLL 的 duration 由独立回归头给出。组装 `Chart(notes, mode=MANIA_4K, bpm_points=[BpmPoint(0, bpm)], sections=<可选>)`（RFC-0005）。解码权重与训练在 Plan 02，本模块只编排/调用。
- **红线规则集（apply）**——仅修物理不可达，不动键型排列语义（R-5）：
  - 越界：`lane ≥ lane_count()` 者——钳到合法区间内最近 lane（保密度优先于删除），记警示。
  - 单帧同按：同一 `time±10ms` 窗口按键数 > `MAX_SIMULTANEOUS_4K` → 按 lane 升序保留前 2，其余整体推迟到下一合法窗口（仅时间平移）。
  - 同手间隔：同侧 lane（见 RFC-0017）两 Note 间隔 < 70ms → 将后一者时间平移至 ≥ 70ms，不改 lane/type。
  - 长条：`HOLD/ROLL` 且 `duration ≤ 0` → 降级为 `TAP`；终点越过下一同时刻 Note 视为重叠，截断 duration。
- **validate（只报告）**：遍历同规则集，返回 `["rule:X note[idx=#i] ..."]`，空列表 = 完全合法可玩。盲测中可仅调 validate 不自改，交人工 review。
- **Writer 分发**：按目标模式/后缀选 `OsuManiaWriter`/`SmWriter`；时间制转换 `.osu`→毫秒整型（四舍五入，秒→ms，RFC-0001）、`.sm`→beat 浮点（按 `Chart.bpm_points` 分段，RFC-0005）。Writer 先 `chart.sorted_notes()` 保证确定性输出，再按各格式头 + HitObjects 段落写盘。
  - `.osu`：`[Difficulty] CircleSize=4` 固定 4K；Note 写为 `x,y,time,type,hitsound,...`，time = `round(note.time*1000)`；HOLD 末尾追加 `endTime`。
  - `.sm`：以 `#NOTES:` 段 + beat 网格（`beat = (time−section_start)*bpm/60`）描述；lane 映射 0/1 列。
- **Reader 对称性**：`OsuManiaReader`/`SmReader` 反向解析回 `Chart`，时间统一归一为秒，供数据流水线 Plan 08 复用（解析现成谱面）；读写同 `mode` 往返须保 Note 不丢、时间误差可量化（M3）。
- **模式扩展（R-6）**：新增模式仅需实现一对 `ChartWriter/ChartReader`（奠基 §1.1）；物理常量按模式覆写（如 7K 的 `MAX_SIMULTANEOUS` 待 Phase 4定义）。
- **解码编排**：`AROut` → `VQVAETokenizer.decode(tokens)` → `Chart` → `PostProcessor(mode).apply(chart)` → `Validate==[]` 断言 → `Writer.write(chart, path)`。编排逻辑放在 `beatmorph/decoder` 顶层薄函数，避免后处理子包反向依赖 io 层。

## 5. 依赖关系

- **上游**：Plan 02 `VQVAETokenizer.decode`、Plan 04 AR 产出 `TokenSeq`、Plan 03 `Section`/`bpm` 时间对齐。
- **下游**：无（终端输出文件）；`Reader` 抽象被数据流水线 Plan 08 复用解析现成谱面。
- **外部库**：`pydantic>=2.5`（契约）；`.osu`/`.sm` 文本写入纯标准库，无重型依赖。

## 6. 里程碑与验收标准

对齐奠基 §7 Phase 2（输出首版可玩 `.osu`，内部盲测）。

| 里程碑 | 验收（可量化） |
|--------|---------------|
| M1 decode 往返 | `decode(tokens)` 还原 `Chart`，Note 形状合法（依赖 Plan 02 重建 > 95%） |
| M2 红线引擎 | `validate` 对人工构造的越界/同按>2/同手<70ms/坏长条样本 100% 检出；`apply` 修正后 `validate` 必返回空 |
| M3 格式导出 | `OsuManiaWriter`/`SmWriter` 产出文件经对应 `Reader` 读回与原 `Chart` 等价（IR 不丢信息）；`.osu` 可被 osu! 客户端载入 |
| M4 首版可玩 `.osu` | 真实 `audio_emb → AR → decode → apply → .osu`，内部盲测可玩率 100%（无可玩性违规） |

## 7. 风险与缓解

| 风险 | 奠基编号 | 缓解 |
|------|---------|------|
| 规则过度约束破坏 AI 创意 | R-5 | `apply` 仅做红线校验、不改键型逻辑；`validate` 产报告供人工 review；创意被规则阻断时记日志不自改 |
| 模式扩展差异大（6K/osu!std/maimai） | R-6 | 4K 优先；扩展仅重写 Tokenizer + 新增 Writer/Reader，物理常量按模式覆写 |
| decode 重建误差累积 | R-2 派生 | 依赖 Plan 02 重建 > 95%；本模块只兜底物理不可达，不兜底创意质量 |
| 时间制转换精度（秒↔毫秒/beat） | 派生 | `.osu` 毫秒四舍五入、`.sm` beat 浮点；读写往返误差计入 Reader 兼容 |
| HOLD 还原丢 duration | Plan 02 §9 派生 | `apply` 校验 `duration≤0` 降级 TAP，截断越界长条 |

## 8. 测试策略

- **单元**：`tests/unit/decoder/test_postprocess.py`——`validate` 逐类违规检出；`apply` 修正后 `validate` 空；越界/同按>2/同手<70ms/坏长条各构造样本（含 lane=`lane_count()` 边界）。`tests/unit/io/test_osu_writer.py`、`test_sm_writer.py`——`read(write(chart))` 往返等价（Note 不丢）。
- **集成**：`tests/integration/test_decode_to_file.py`——`Token → decode → PostProcessor.apply → OsuManiaWriter.write`，产出文件能被 `OsuManiaReader` 读回等价；红线被触发时产物经 `validate` 为空。
- **e2e**：真实 `audio_emb → Plan 04 → Token → 本模块 → .osu`，osu! 客户端/开源解析器载入无报错；标注 `@pytest.mark.e2e`、`@pytest.mark.slow`。

### 可玩性保证闭环（奠基 §3.7「100% 可玩」落地）
契约不变式：经 `apply` 修正后的 `Chart` 必满足 `PostProcessor(mode).validate(chart) == []`。该断言同时作为 M2、集成测试与 e2e 的硬门禁——任一环节 `validate` 非空即视为该样本未通过物理安全校验，不进入盲测集、不写盘。此不变式是 AI 创意与规则物理安全解耦（R-5）的工程兑现：规则只守红线、不自改键型，但仍保证输出谱面对玩家「可达」。

## 9. 开放问题

- [ ] RFC-0017：4K 左右手 lane 分配（`{0,1}`/`{2,3}`）是否纳入契约常量，供 7K/maimai 复用约定。
- [ ] RFC-0018：`apply` 对越界 Note "钳到最近 lane" vs "删除" 的取舍（影响密度保真度）。
- [ ] RFC-0019：`.ma2`(maimai) 导出是否在 Phase 2 预留接口壳，还是延至 Phase 4。
- [ ] 同手间隔 70ms 在 7K / osu!std 模式下的换算（依赖 R-6 扩展时定义）。
