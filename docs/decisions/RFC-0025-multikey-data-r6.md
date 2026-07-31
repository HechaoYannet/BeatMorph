# RFC-0025 — R-6 细化：多 K mania 数据诚实标记 + 训练层 4K 过滤

- 状态：采纳 ｜ 提出日期：2026-07-31 ｜ 决定日期：2026-07-31
- 提出者：架构组
- 影响模块：plan 00（`GameMode` 枚举）、plan 07（`OsuManiaReader` 模式映射）、plan 08/09（`PlannerDataset` 训练过滤）

## 背景

奠基 R-6：「先攻 4K VSRG，其它模式（6K/osu!std/maimai）视为独立适配工程，**不得为它污染 4K 主路径**」。CLAUDE.md §3 红线 4 同义。但「不污染 4K 主路径」有两种实现解读：

1. **数据层剔除**：非 4K 谱面在解析/下载期即丢弃，4K 主路径天然纯净。
2. **训练层过滤**：非 4K 谱面诚实标记后落盘，训练时按 `mode==MANIA_4K` 过滤，4K 主路径同样纯净，且多 K 数据备未来扩展。

原 `OsuManiaReader` 采用了**第三种（错误）做法**：把 5/6/7/8K 谱面的 `mode` 强行设为 `MANIA_4K`，但 `lane = int(x * key_count / 512)` 仍用真实键位数计算。后果链：

- `chart.mode = MANIA_4K` → `lane_count() = 4`
- 7K 谱面 Note 的 lane ∈ {0..6}
- `parse_osu` 清理规则（`note.lane >= lane_max` 即 ≥4）**误删 lane 4/5/6 的 Note**
- 产出「残缺的 4K 谱面」（只剩 lane 0-3）混入训练集

这既违反 R-6（残缺谱面污染 4K），又丢失了多 K 数据（未来扩展要重新下载）。本 RFC 定型正确做法。

## 提议

**采纳方案：数据层诚实标记 + 训练层 4K 过滤（上表解读 2）**。

### 1. 契约：补全 `GameMode` 枚举（plan 00）

`GameMode` 原有 `MANIA_4K=4 / MANIA_7K=7`，缺 5/6/8K。补齐：

```python
MANIA_4K = 4   # 主攻
MANIA_5K = 5   # 扩展（数据侧诚实标记）
MANIA_6K = 6
MANIA_7K = 7
MANIA_8K = 8
```

`Chart.lane_count() = int(mode)` 对所有 mania K 数正确返回键位数。

### 2. Reader：诚实标记真实 K 数（plan 07）

`OsuManiaReader` 按 `CircleSize`（= mania 键位数）查表映射真实 `GameMode`，**不再降级为 4K**：

```python
_MANIA_MODES = {4: MANIA_4K, 5: MANIA_5K, 6: MANIA_6K, 7: MANIA_7K, 8: MANIA_8K}
mode = _MANIA_MODES.get(int(key_count), MANIA_4K)  # 异常 kc 防御性 fallback
```

- `lane = int(x * key_count / 512)` 与 `lane_count() = key_count` 一致 → **lane 不再误删**。
- 非 4K mania 谱面完整落盘（含全部 lane 的 Note），`mode` 诚实标记。

### 3. 训练层：`mode==MANIA_4K` 过滤（plan 08/09）

`PlannerDataset._load_index` 跳过 `chart.mode != MANIA_4K` 的样本（计数 `n_non_4k` 入日志）。**4K 主路径 100% 纯 4K**，守 R-6。

`MERTExtractionDataset` **不加** mode 过滤——MERT embedding 是音频表征，与键位数无关，多 K 谱面的 embedding 同样可复用（未来多 K planner 共享 Stage0）。

### 4. 下载层：`only_4k` 语义不变

`download_sayobot.py --only-4k` 仍只保留**含 4K diff 的 set**（set 级过滤，省带宽）。一个 set 若同时含 4K+7K diff，7K diff 会被诚实标记落盘，训练层过滤剔除。无需改下载脚本。

## 备选方案

1. **数据层剔除非 4K（Reader 返回 skip_reason）**：4K 主路径最纯净、改动最小。但多 K 数据丢弃，未来扩展（奠基 R-6「独立适配工程」）需重新下载 10K+ 谱面，代价大。**放弃**——与项目「最终不止 4K」的演进意图相悖。
2. **多 K 混训（所有 mania 当 4K 训）**：违反 R-6 + 数据残缺（即原 bug）。**排除**。
3. **本提议（诚实标记 + 训练过滤）**：数据层保留多 K 备扩展，训练层守 4K 纯净。代价是多 K 数据占盘（已被 gitignore），可接受。**采纳**。

## 后果

- **R-6 边界细化**：「不得污染 4K 主路径」= **训练层** mode 过滤，非数据层丢弃。数据层可收多 K mania。
- **契约**：`GameMode` 增 5/6/8K，IR 版本仍 `ir-1`（纯枚举扩展，向后兼容）。
- **plan 00**：`GameMode` 枚举与 docstring 更新（「Phase 1 训练仅 4K，数据侧可收多 K」）。
- **plan 07**：Reader 模式映射改为查表诚实标记；`test_difficulty_7k` 增 `mode==MANIA_7K` 断言。
- **plan 08/09**：`PlannerDataset` 增 4K 过滤；`MERTExtractionDataset` 不过滤。
- **未来多 K 扩展**：重写 Tokenizer（plan 02）与导出层（plan 07 Writer）即可，数据已就位，无需重下。

## 关联

- 奠基：§6 R-6（先攻 4K）、§1.1（4K 优先）。
- CLAUDE.md §3 红线 4（先攻 4K）。
- 相关 plan：00（GameMode）、07（Reader）、08/09（数据集过滤）。
