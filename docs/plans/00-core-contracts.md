# Plan 00 — 核心契约与张量形状

> 状态：🟡 草案 ｜ 阶段：Phase 1 ｜ 负责：架构组
> 对应代码：`beatmorph/core/contracts/` ｜ 对应奠基章节：§2、§3.7

## 1. 目标与范围

### 交付
- 跨模块共享的数据类型（`Note`/`BpmPoint`/`Chart`/`Section`/`PatternToken` 等）。
- 跨模块张量形状约定（`AudioEmbedding`/`PlanOutput`/`TokenSeq`/`RAGContext`）。
- 关键常量（MERT 帧率 25Hz、码本 2048/4096、4K 键位、AR 上下文 256）。
- 统一日志入口。

### 不交付
- 任何模型实现、训练逻辑、数据 IO。

### 价值
所有模块通过本契约通信，保证：接口稳定可演进（pydantic 版本字段）、可静态校验（mypy strict）、可序列化（IR ↔ JSON）、跨设备一致。

## 2. 与奠基文档的对应

| 契约 | 奠基依据 |
|------|---------|
| `Note(time,lane,type,duration)` | §3.7「`(time, lane, type, duration)`」 |
| `Section`（每 4 小节，density/energy/rest） | §3.3 Stage1 输出 |
| `PatternToken`（单小节离散码本） | §3.2 / §3.4 |
| `Chart` IR 与格式解耦 | §3.7「IR JSON → .osu/.sm/.ma2」 |
| `difficulty 1-15` | §1.2 / §2 用户输入层 |
| MERT 25Hz、768d | §3.1 |
| 码本 2048/4096、AR 256 tokens | §3.2 / §3.4.1 |

**偏离 1**：奠基文档未明确 Note 时间单位。本计划规定 `Note.time` 为**秒（float）**，理由：① 与音频时间轴天然对齐；② 内部统一浮点，写入各格式时由 Writer 转毫秒/Beat。已由 [RFC-0001](../decisions/RFC-0001-note-time-unit.md) 采纳定稿。

**偏离 2**：奠基文档全程以 scalar `bpm` 描述，未覆盖变速曲。`Chart` 改用 `bpm_points: list[BpmPoint]`（至少 1 个，按 time 升序）表示「时间→BPM」分段常数序列；常速曲退化为单元素列表。已由 [RFC-0005](../decisions/RFC-0005-bpm-timepoints.md) 采纳定稿，影响 plan 02/03/08。

## 3. 接口契约

### 3.1 Note
```python
class Note(BaseModel):
    time: NonNegativeFloat       # 秒
    lane: NonNegativeInt         # 0-based；4K 即 0..3
    type: NoteType = TAP         # TAP/HOLD/MINE/ROLL/FAKE
    duration: NonNegativeFloat = 0.0   # 仅 HOLD/ROLL
```
约束：`time≥0`、`lane≥0`、`duration≥0`。

### 3.2 Chart（IR）
```python
class Chart(BaseModel):
    version: str = "ir-1"        # IR schema 版本，破坏性变更时升号
    mode: GameMode = MANIA_4K
    difficulty: int              # 1-15
    bpm_points: list[BpmPoint]   # >=1，按 time 升序（RFC-0005 变速时间点）
    notes: list[Note]
    sections: list[Section]      # 可选（Stage1 产物）
    meta: dict[str, str|int|float]
```
方法：`primary_bpm()`（取首点 BPM，供标量场景）、`lane_count()`、`sorted_notes()`（按 (time,lane) 确定性排序）。`BpmPoint(time: 秒, bpm: float>0)`。

### 3.3 Section / PatternToken
见 `beatmorph/core/contracts/events.py`。`density_target`/`energy_level`/`rest_probability` 均 ∈[0,1]，由 pydantic `Field(ge=0,le=1)` 强制。

### 3.4 张量形状约定（einops 风格）
| 名称 | 形状 | 含义 |
|------|------|------|
| `AudioEmbedding` | `(batch, time_seq, 768)` | Stage0 输出，25Hz |
| `PlanOutput` | `(batch, num_sections)` ×3 | density/energy/rest |
| `TokenSeq` | `(batch, seq_len)` long | Pattern ID ∈[0,2048) |
| `RAGContext` | `(batch, top_k, seq/feat)` | 检索参考 |

## 4. 内部设计

- 契约分两文件：`events.py`（领域对象）、`tensors.py`（张量形状 dataclass + 常量）。`__init__.py` 集中导出。
- 所有领域对象继承 `pydantic.BaseModel`，享受校验与 JSON 序列化。
- 张量形状用 `@dataclass(frozen=True)`，仅描述不持有数据。
- 日志：`core/logging.py` 提供 `get_logger()`，统一 `beatmorph.` 前缀，可选 JSON 输出。

## 5. 依赖关系

- **上游**：无（最底层）。
- **下游**：所有模块。
- **外部库**：`pydantic>=2.5`（标准库以外唯一硬依赖，刻意保持最小）。

## 6. 里程碑与验收标准

| 里程碑 | 验收 |
|--------|------|
| M1 契约冻结 v1 | `from beatmorph.core.contracts import *` 可导入；`tests/unit/core/test_contracts.py` 全绿 |
| M2 类型检查通过 | `uv run mypy beatmorph/core` strict 零错误 |
| M3 IR 序列化往返 | `Chart.model_validate_json(chart.model_dump_json())` 等价（往返不丢信息）|

## 7. 风险与缓解

- **契约早冻结导致返工**：`Chart.version` 字段 + RFC 流程做向前兼容；新需求先开 RFC。
- **与奠基文档语义漂移**：本 plan §2 表逐项可追溯到奠基章节；奠基更新时触发本表复核。

## 8. 测试策略

- 单元：`tests/unit/core/test_contracts.py` 覆盖边界值、非法值、确定性排序、常量一致性。
- 契约即测试：pydantic 约束本身即运行期断言。

## 9. 开放问题

- [x] RFC-0001：Note 时间单位秒 vs 毫秒 —— [采纳，秒](../decisions/RFC-0001-note-time-unit.md)。
- [x] 多 BPM 变速曲目的 `bpm` 字段是否需扩为列表 + 时间点 —— [采纳 RFC-0005](../decisions/RFC-0005-bpm-timepoints.md)，`Chart.bpm_points: list[BpmPoint]`。
- [ ] `GameMode.OSU_STD` 非键位模型，`lane_count()` 返回 0 的语义待 Phase 4 明确。
