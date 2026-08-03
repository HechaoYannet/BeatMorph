# RFC-0026 — 小节边界相位对齐（修复栅格化系统性错位）

- 状态：采纳 ｜ 提出日期：2026-08-03 ｜ 决定日期：2026-08-03
- 提出者：Tokenizer 组
- 影响模块：plan 02（`beatmorph/tokenizer/vqvae.py::rasterize_chart`）、plan 03（`section_boundaries_from_bpm`）、plan 08（`beatmorph/data/parsers/osu_path.py::_compute_bar_boundaries` / `compute_section_stats`）。**Stage1 planner 已训练伪标签受影响**（见后果）。

## 背景

VQ-VAE GPU 冒烟时发现真实 4K mania 谱面（difficulty 9-12，`data/raw`）经栅格化后 **Note 全不落在规整拍点**：

```
chart0（MIMI - Floremie, BPM=108）：
  [TimingPoints] 首个非继承红线: time=339ms, beatLength=555.55ms, meter=4
  首个 Note: time=339ms  ← 精确落在首红线（downbeat）时刻
  当前 _compute_bar_boundaries 从 t=0 推 bar 边界 → 错位 339ms（≈ 半拍）
  结果：Note 落 1/16 拍点比例 = 0%
  用 bpm_points[0].time 做相位对齐重算 → 落 1/16 拍点比例 = 88%
```

**根因**：`_compute_bar_boundaries` 与 `section_boundaries_from_bpm` 都从 `current_time = 0.0` 起算小节边界，**忽略 `bpm_points[0].time` 作为节拍网格相位原点**。

**osu! 语义确证**：uninherited timing point（红线）的 `time` 字段是**节拍网格相位原点 / downbeat 时刻**，osu! 小节网格 = `timing.time + k × bar_dur`，不是 `0 + k × bar_dur`。实证：谱面首 Note 精确等同首红线 `time`，证明音乐从该时刻起拍。

**影响范围**（跨模块既存 bug，非 plan 02 引入）：
- `compute_section_stats` 切 Section 边界（Stage1 伪标签）→ planner 训练用错位边界。
- `section_boundaries_from_bpm`（planner 推断边界）。
- `rasterize_chart`（VQ-VAE 栅格化，note 系统性错位到伪随机 bin，code 语义失真）。

**对 VQ-VAE 的影响**：encoder 学不到节奏型（Note 散落无规律），是本 plan 02 冒烟中"present 头类不平衡"表象下的真因之一（详见 [`vqvae-tokenizer-implementation`](../../../C:/Users/yhcne/.claude/projects/E--otherProject-BeatMorph/memory/) 记忆的层 1）。

## 提议

**修复 `_compute_bar_boundaries` 与 `section_boundaries_from_bpm` 的相位对齐**：

1. 小节相位原点 = `bpm_points[0].time`（首个非继承红线时刻）。
2. 首个小节起点 = `bpm_points[0].time`（而非 `0.0`）；bar 边界 = `phase + k × bar_dur`（常速）或按 bpm_points 分段累加（变速，每段内相位由该段红线 time 决定）。
3. 边界序列仍以 `0.0` 为首元素（保持"覆盖 [0, total_duration]"语义），首段 `[0, phase]` 视为「网格前 intro 段」，若该段无 Note 则不计入小节数；若有 Note（极少，intro 有击打）则作为独立短小节保留以保证不丢 Note。

**实现**（`osu_path.py::_compute_bar_boundaries`）：
```python
phase = bpm_points[0].time if bpm_points else 0.0
boundaries: list[float] = [0.0] if phase > 0 else [phase]
# phase > 0 时保留 [0, phase] 作 intro 段；phase == 0 时首边界即 phase=0
current_time = phase
...
```

**对 planner 伪标签的处理**：Stage1 `compute_section_stats` 的 Section 边界会因此移动（真实谱面 phase>0 时每段偏移 up to ~1 bar）。planner 已用旧（错位）伪标签训练。**本 RFC 不要求重训 planner**——决策为：后续 Phase 1 数据扩到 10K/50K 重跑 `PreprocessPipeline`（产新伪标签）时自然用上修复，planner 下次训练即对齐。当前已训练 planner 模型保留，不回滚（Phase 1 仍在迭代，非生产部署）。

**测试**：
- 新增 `_compute_bar_boundaries` phase 对齐单测（`bpm_points[0].time=0.339`，验证首 bar 起点对齐 phase、Note 落规整拍点）。
- 现有 `test_compute_section_stats.py` 全部用 `BpmPoint(time=0.0,...)`（phase=0），**修复后边界不变，零破坏**（已核对）。

## 备选方案

1. **软栅格化用区间模糊化缓冲错位**（放弃）：用 ±k bin 高斯激活钝化错位。问题：±1 bin @120bpm = ±31ms 已超 ±20ms 验收容差，且不修复根因（Note 仍错位、节奏型仍失真）。**驳回**：用精度掩盖确定性 bug 是错误工程。

2. **可学习时间扭曲**（放弃）：引入可学习因子吸收相位偏移。问题：① 确定性信息（timing point `time`）不该可学习化；② 音游谱面 quantized，无需时间扭曲；③ 给量化数据上可学习对齐引入不可解释性与训练不稳定。**驳回**：过度设计。

3. **彻底放弃小节粒度改 event tokenizer**（推迟）：见 [`RFC-0028`](RFC-0028-bpe-event-tokenizer-constitutional-amendment.md) 修宪议案。该议案属范式级决策，本 RFC 不替代。即便采纳 event tokenizer，按小节聚合的统计需求仍存在，phase 对齐仍是正交必做项。

## 后果

- **代码**：`osu_path.py::_compute_bar_boundaries` + `planner/density.py::section_boundaries_from_bpm` 加 phase 对齐；`rasterize_chart` 自动受益（复用同函数）。
- **测试**：phase 对齐新单测 + 现有测试零破坏（phase=0 样本边界不变）。
- **planner 伪标签**：旧（错位）伪标签影响的已训练 planner 模型保留不回滚；下次数据扩跑重产伪标签时自动用修复，planner 重训即对齐。记入 plan 03 §9。
- **VQ-VAE**：栅格化后 Note 落规整拍点，encoder 可学节奏型，codebook 语义恢复（density/节奏型/手型/键位分布 §3.2.2）。与 [`RFC-0027`](RFC-0027-set-encoder-vqvae.md)（set encoder）正交——两者都修，phase 先于架构。
- **风险**：phase>0 的 intro 段（[0, phase]）内若有 Note，需保留为短小节防丢；边界值变化可能让依赖确切边界秒数的下游（如未来切片对齐 Section）产生 1 bar 内偏移——属可接受，因伪标签本就需重产。

## 关联

- 奠基：§3.2.1（时间粒度 1 小节，以 BPM 动态对齐）、RFC-0005（bpm_points 变速点）。
- plan 02 §4（栅格化按 BPM 动态对齐）→ 本 RFC 补全"相位原点"语义。
- plan 08 §4 Step2（compute_section_stats 切 Section）→ 边界同源受益。
- 相关 RFC：[RFC-0027](RFC-0027-set-encoder-vqvae.md)（set encoder，正交架构演进）、[RFC-0028](RFC-0028-bpe-event-tokenizer-constitutional-amendment.md)（BPE 修宪，范式级 Plan B）。
- 触发来源：VQ-VAE GPU 冒烟 + present 头稀疏度诊断（见 vqvae 实现记忆层 1）。
