# RFC-0005 — 变速曲 `bpm` 字段扩展为时间点序列

- 状态：采纳 ｜ 提出日期：2026-07-30 ｜ 决定日期：2026-07-30
- 提出者：架构组
- 影响模块：plan 00（`beatmorph/core/contracts/events.py`）、plan 02（栅格化）、plan 03（Section 划分）、plan 08（`compute_section_stats`）

## 背景

奠基文档 §3.2.1 规定 VQ-VAE 的时间粒度为「1 小节（以 BPM 动态对齐）」，Plan 02「严格按 BPM 计算小节边界」、Plan 03「按 `section_bars=4` 小节切分音频时长」。这些逻辑都依赖 BPM 把时间轴切成小节。

但奠基文档「核心公式」「用户输入层」「§4.2」全程只提 `bpm` 单一标量字段，**未覆盖变速曲**（一首曲子中途换 BPM，在 osu!/BMS 中相当常见）。原契约 `Chart.bpm: float` 只能表示一个主 BPM：

- 常速曲：足够。
- 变速曲：小节边界会随变速点偏移，单一标量 BPM 无法还原真实小节结构，导致 Plan 02 栅格化、Plan 03 Section 划分、Plan 08 伪标签在变速曲上全部失真。

Plan 00 §9 开放问题与 Plan 02/03/08 的「§9 开放问题」均把变速 BPM 字段列为共同前置阻塞项。本 RFC 给出定型决策。

## 提议

将 `Chart.bpm: float` 替换为 `Chart.bpm_points: list[BpmPoint]`，每点携带「时间锚 + BPM」：

```python
class BpmPoint(BaseModel):
    time: NonNegativeFloat   # 该 BPM 生效起始时刻（秒）
    bpm: float = Field(gt=0.0)

class Chart(BaseModel):
    bpm_points: list[BpmPoint] = Field(min_length=1)  # 至少 1 个，按 time 升序
    ...
    def primary_bpm(self) -> float:
        return self.bpm_points[0].bpm
```

语义：BPM 是「时间→BPM」的**分段常数**序列。某点生效后，至下一变速点前的整段均使用该点 BPM。时间单位为秒，与 `Note.time`（RFC-0001 暂定秒）一致，便于与音频轴对齐。

- **常速曲**：退化为单元素列表 `[BpmPoint(time=0.0, bpm=...)]`。
- **变速曲**：按 time 升序追加，例如 `[Bpt(0,120), Bpt(30,180), Bpt(60,240)]`。
- **`min_length=1`**：禁止空序列，保证下游恒可取 `primary_bpm()`。
- **小节边界不预先存储**：由 `bpm_points` 与时间即时推得（奠基 §3.2.1「以 BPM 动态对齐」），Plan 02 栅格化逻辑据此实现。
- **`primary_bpm()`**：为只需标量 BPM 的场景（RAG 元数据过滤 Plan 05、`difficulty_rating`/流派展示）提供便捷取值，取首点；解析器在构造时应将「主段」置首。

## 备选方案

1. **保留 scalar `bpm` + 新增可选 `bpm_points`**：兼容性好，但「双字段」隐含「两者不一致」的取值歧义，且 Plan 02/03 在变速曲上仍需读 `bpm_points`，scalar 形同摆设，徒增维护面。放弃。
2. **统一改为 `bpm_points`，移除 scalar**（本提议采纳）：语义单一干净，常速曲仅多一层单元素列表包装，下游全用同一口径，无歧义。代价是破坏性变更，需同步测试与下游 plan 表述。鉴于项目尚处 Phase 1 骨架阶段、`Chart.bpm` 仅有 4 处测试引用、无生产代码依赖，破坏性代价可承受，故采纳。
3. **Phase 1 不定稿，维持 scalar**：搁置变速曲。风险：Phase 1 以 osu! 主链路为主（奠基 §7），osu! 谱面含变速（TimingPoints 的 BPM 变化段）比例不小，搁置会让 Plan 08 解析从一开始就失真，后补代价更大。放弃。

## 后果

- **契约破坏性变更**：`Chart.bpm` → `Chart.bpm_points`。IR 版本仍为 `ir-1`（Phase 1 骨架阶段无存量数据需迁移，故不升号；正式发布前若已产数据须升 `ir-2` 并提供迁移器）。
- **Plan 00**：§3.2 Chart 定义、§2 对应表更新；M1 测试已补 `bpm_points` 约束与变速构造用例；开放问题 RFC-0005 标记「采纳」。
- **Plan 02**：栅格化按 `bpm_points` 分段计算小节边界，§9 派生风险「变速曲小节边界不准」有据可依。
- **Plan 03**：Section 划分（`section_bars=4`）按分段 BPM 推小节，§4「需 BPM，来自 `Chart.bpm`」改为「来自 `Chart.bpm_points`」。
- **Plan 08**：`compute_section_stats(chart, section_bars=4)` 与 `parse_osu` 需从 `.osu` 的 TimingPoints 解析出 `bpm_points`，`.sm` 的 `#BPMS` 段同理；解析器负责按 time 升序并置主段于首。
- **Plan 05 RAG**：检索入参 `retrieve(..., bpm: float)` 取 `chart.primary_bpm()`，无需改接口签名（已是 scalar 入参）。
- **Plan 07 Writer**：`.sm`→beat 浮点换算用 `bpm_points` 分段，非 scalar。

## 关联

- 奠基：§3.2.1（BPM 动态对齐）、§4.2（解析流水线）。
- RFC-0001（Note.time 单位秒）——BpmPoint.time 与之一致用秒。
- 相关 plan 开放问题：00 §9、02 §9、03 §9、08 §9。
