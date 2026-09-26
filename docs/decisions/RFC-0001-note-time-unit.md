# RFC-0001 — Note 时间单位（秒 vs 毫秒）

- 状态：采纳 ｜ 提出日期：2026-07-30 ｜ 决定日期：2026-07-30
- 提出者：架构组
- 影响模块：plan 00（`beatmorph/core/contracts/`）、plan 02（解析/栅格化）、plan 05（Writer 时间换算）

> ## ⚠️ 勘误（2026-08-05，RFC-0029 修宪后）
>
> 本 RFC 的**结论（`Note.time` 用秒）仍然有效**，但以下内容已过时：
> - 全文引用的 MERT 帧率 **25Hz 是错值**，真值为 **75 Hz**（`24000 / prod(conv_stride=[5,2,2,2,2,2,2]) = 24000/320`）。换算应为 `frame = round(time_s * 75)`。
> - 原文"音频侧是固定 25Hz 帧序列"这一**论据**需按 75Hz 重估；**结论不变**。
> - ****不与 [RFC-0029](RFC-0029-phigros-continuous-chart-generation.md) Q15 冲突**：RFC-0029 讨论的是**强度场的时间网格**是否改为 beat-aligned，本 RFC 管的是 **IR 契约 `Note.time` 的单位**——两者层次不同，IR 继续用秒、场网格可用拍。
>
> 见 [POSTMORTEM-2026-08-05](../POSTMORTEM-2026-08-05-frame-rate-misalignment.md)。

## 背景

奠基文档 §3.7 给出 Note 最小表示为 `(time, lane, type, duration)`，但未明确 `time`/`duration` 的单位。各游戏格式原生单位不一：

- `.osu`：HitObject time/endTime 为**毫秒整型**。
- `.sm`：`#NOTES` 段以 **beat（拍）浮点** 定位。
- 音频/MERT 帧轴：25Hz 帧率，自然以**秒**计。

若 IR 内部不加规定，各模块各取所需易产生单位漂移（如 Audio 层用秒、解析层用毫秒而忘记换算）。Plan 00 §2「唯一偏离」声明：暂定 `Note.time` 为**秒（float）**，理由是与音频时间轴天然对齐；本 RFC 将该暂定正式定稿。

## 提议

- `Note.time` 与 `Note.duration`：**秒，`NonNegativeFloat`**。这是系统内部一切 Note 时间的统一单位。
- `Section.start_time`/`end_time`、`PatternToken.start_time`、`BpmPoint.time`（RFC-0005）一律**秒**，内部时间轴全场对齐音频轴。
- **格式边界换算**只在 IO Writer/Reader 内进行，不外泄到 IR：
  - `.osu` Reader：ms int → 秒（`/ 1000.0`）；Writer：秒 → ms int（四舍五入）。
  - `.sm` Reader：beat float → 秒（`beat * 60 / bpm`，需对接 `bpm_points` 分段）；Writer 反向。
- MERT 帧位置与秒的换算固定用 `MERT_FRAME_RATE_HZ=25.0`：`frame = round(time_s * 25)`。

## 备选方案

1. **毫秒**：与 `.osu` 原生单位一致，省去 osu Reader 换算。但 `.sm` 仍需 beat↔ms 换算，且音频轴/MERT 帧以秒为自然单位，统一用秒的换算总量更少、浮点精度更直观（100bpm 下 1 拍 = 0.6s，毫秒则 600ms）。放弃。
2. **beat（拍）**：与音乐结构天然对齐，且独立于 BPM。但 MERT 输出是固定 25Hz 帧序列、音频解码以采样点/秒为单位，用 beat 会让音频侧频繁做 beat↔秒换算，反而把复杂度甩给最高频的音频路径。且变速曲下 beat→秒需分段积分（依赖 RFC-0005 的 `bpm_points`），不如直接用秒。放弃。
3. **秒**（采纳）：与音频轴、MERT 帧轴、`BpmPoint.time` 一致，内部无歧义；格式换算收敛到 IO 边界。

## 后果

- Plan 00：`events.py` 注释已写 `time: 秒`；本 RFC 正式定稿，移除开放问题。模块顶部 docstring 中「谱面 Note 以毫秒或秒计」的弱表述以本 RFC 为准——内部定秒。
- Plan 02：栅格化 `Note` 到 `(lane, time_bins)` 时，time_bins 由秒经 BPM 推得，单位一致。
- Plan 07：Writer 必须显式做 秒↔ms / 秒↔beat 换算，集中在其模块内，IR 不感知。
- Plan 08：`.osu`/`.sm` 解析入口做 ms→秒、beat→秒换算（对接 `bpm_points` 分段积分）。
- 测试：契约层已断言 `time` 为非负浮点；IO 层单测须覆盖单位换算边界（如 osu `time=0` → 0.0s）。`time±20ms` 的 VQ-VAE 重建验收（Plan 02 M2）以秒精度比较：容差 `0.02s`。

## 关联

- 奠基：§3.7（Note 表示）、§3.1（MERT 25Hz）。
- RFC-0005（`bpm_points` 时间锚同样以秒计）。
- 相关 plan 开放问题：00 §9、02 §7 R-2、07（Writer 时间制）、08（解析）。
