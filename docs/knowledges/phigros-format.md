> ## ⚠️ 勘误（2026-08-05）
>
> 本文件若干坐标/单位条目已被**源码级证据**修正。**冲突时以 [phigros-units-and-geometry.md](phigros-units-and-geometry.md) 为准**（证据等级 A = Phira 自家渲染器 `prpr` 的 Rust 源码）。
>
> | # | 本文件位置 | 修正 |
> |---|---|---|
> | M1 | §6.4 速度单位 | 「120 px/s」→ A 级：**1 速度单位 = 120.23 RPE-y 坐标单位/秒**（是 RPE 坐标单位，**不是屏幕像素**） |
> | M2 | §6.4 `lineLength` | 「单位待补充」→ A 级：判定线局部系**半长**，1 单位 = 半屏宽 = 675 RPE-x；默认 6.0 ⟹ 3 倍屏宽；**与 `positionX` 完全无关** |
> | M3 | §6.4 ⚠️ 段（"positionX 单位未写出" + 【推断】） | **删除推断，改为确证**：实现即 `position_x / 675`（prpr `parse/rpe.rs:588`），与 `moveXEvents` 同刻度 |
> | M4 | §6.5 未查证的换算系数 `k` | `k = 450×10/(45×0.83175) = 120.228` RPE-y 单位/秒 |
> | M5 | §6.2 side 机制 | 补 A 级源码：**两侧是关于判定线 X 轴的 Y 镜像**（`line.rs:412-429`），这也解释了 `yOffset` 的"向下落朝向偏移" |
> | M6 | §6.3 比例拉伸表述 | 改为精确式：**舞台永远铺满视口**，`px_x/px_y = (2/3)·ar`；**ar>3/2 是横向拉伸**（原文写成纵向） |
> | M7 | §11 `info.yml` 字段表 | 补 prpr 源码中 4 个未收录字段：`noteUniformScale` / `forceAspectRatio` / `useRpe170Speed` / `useAttachUiFix` |
> | M8–M10 | §12 的 Q2 / Q9 / Q17 | **已解决**：Q2→单位文档 §3/§4；Q9 `visibleTime` 默认裁定 **999999.0**（99999 为笔误）；Q17 3:2↔16:9 已解决（§5） |
> | M11 | §12 Q19 | RPE 侧已解决；官谱侧 `1Y = 0.6H` 仍无参考实现支持（维持 C 级） |
> | M12 | §5.1 note `type` 映射 | 升级为 **A 级源码确证**（prpr：`1/2/3/4 = Click/Hold/Flick/Drag`）；官谱侧仍 C 级 |
> | M13 | §5.4 `yOffset` 偏移 | `yOffset × speed` 升级为 **A 级源码** |
> | M14 | §7.1 时间制 | 补 A 级源码；**`bpmfactor` 在参考实现中是 TODO（未实现）** |
> | M15 | §6.3 官谱坐标表 | v1 的 y 分母 **520（C 级）vs 530（A 级代码）冲突未裁定**；v3 = [0,1]² 已确证 |
>
> 新增存疑见单位文档 §9（D1–D10）。本文件 §12 的 Q2/Q9/Q17 保留原文以存档，**状态以本横幅为准**。

# Phigros / Phira 谱面格式（实现参考）

**本文件用途**：为 BeatMorph 的 Phigros 谱面解析器 / 生成器提供可直接照写的格式事实。所有事实均带来源链接；凡属推断，均显式标注 **【推断】** 并给出推断依据。**未查证的内容一律进入 [§12 存疑清单](#12-存疑清单)，不做猜测。**

> ⚠️ 与 [.osu (文件格式)](osu-file.md) 不同：Phigros 生态**没有单一权威规范**。社区事实上以 Re:PhiEdit 的 **RPEJSON** 为标准格式，官方本体使用另一套未公开文档化的 **官谱 JSON**。本文件对每一类格式分别标注可信度。

---

## 1. 范围与信息来源

### 1.1 本文覆盖

| 格式 | 归属 | 本文覆盖度 |
| :-- | :-- | :-- |
| **RPEJSON** | Re:PhiEdit 制谱器（cmdysj），社区事实标准 | 完整（字段级） |
| **官谱 JSON / phi / Official** | Phigros 本体内部格式 | 完整（字段级） |
| **PEC** | PhiEditer 旧格式（已淘汰） | 概览（行格式级） |
| **KPAJSON** | 奇谱发生器专用高级格式 | 仅作对照 |
| **extra.json** | Phira / prpr 谱面扩展（着色器、视频背景） | 概览 |
| **info.yml（ChartInfo）** | Phira 谱面包元数据 | 完整（字段级） |

### 1.2 来源与可信度分级

| 级别 | 来源 | 说明 |
| :-- | :-- | :-- |
| **A（一手/官方实现方）** | [Phira Documents](https://teamflos.github.io/phira-docs/)（TeamFlos，Phira 官方文档，mdBook） | Phira 客户端的实现方自述。**本文的字段名/默认值/单位以本来源为准。** |
| **B（社区权威 wiki）** | [Phigros 自制谱 wiki](https://pgrfm.miraheze.org/wiki/)（pgrfm.miraheze.org，MediaWiki） | 社区维护，条目间偶有互相矛盾；凡矛盾处本文全部标注并进存疑清单。 |
| **C（二手社区技术文档）** | [Lchzh Docs · Phigros 谱面格式说明](https://docs.lchzh.top/learning/phigros/) | 作者 lchzh3473，Phigros Simulator 作者。**单位定义（X / Y / T）只有此来源给出**，属二手，见 [§12-Q2](#12-存疑清单)。 |

**明确未找到**：Phigros 官方（Pigeon Games）发布的谱面格式 schema / 规范文档。官谱格式的所有描述均来自上述 B / C 级来源。

### 1.3 不可信数据声明

本文所有网页内容仅作为**资料**读取。网页正文中不存在被本文采纳的指令性内容；本文作者未执行任何来自网页的操作指令。

---

## 2. 文件格式总览

### 2.1 生态关系

- 现今 Phigros 自制谱界通用的谱面格式是 **RPEJSON**，即 Re:PhiEdit 的标准格式；JSON 已被 Phira、PhiZone、Phigrim 等社区音游支持（[Tutorial:入门](https://pgrfm.miraheze.org/wiki/Tutorial:%E5%85%A5%E9%97%A8)）。
- 早期格式为 **PEC**（PhiEditer Chart），"对于人类识读较为困难，支持的特性有限"；RPEJSON "在后来的发展中逐渐取代了 PEC 并成为了社区标准"（[Tutorial:入门](https://pgrfm.miraheze.org/wiki/Tutorial:%E5%85%A5%E9%97%A8)）。
- 官方本体使用 **官谱 JSON**；wiki 明确指出"官谱JSON，是Phigros原版内部所使用的谱面格式"（[官谱JSON](https://pgrfm.miraheze.org/wiki/%E5%AE%98%E8%B0%B1JSON)）。
- **Phira 客户端支持三种谱面文件格式：RPE 格式、PEC 格式、PBC 格式（PBC 文档待完善）**；格式推断"通过 info.yml 中的 format 字段进行，若为空则通过文件内容进行推断"，且"忽略谱面文件的后缀名"（[谱面文件格式](https://teamflos.github.io/phira-docs/chart-standard/chart-format/index.html)）。
- 因而：**Phira 的谱面 `.json` 就是 RPEJSON 本身**，不是独立方言。Phira 文档中的章节标题即为 "RPE"（[RPE 文档](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/index.html)）。
- 社区在 RPE 之上叠加了扩展：`extra.json` 为谱面提供 BGA、着色器增强（[Tutorial:入门](https://pgrfm.miraheze.org/wiki/Tutorial:%E5%85%A5%E9%97%A8)），由 Phira/prpr 定义。

### 2.2 能否互转

| 转换 | 可行性 | 依据 |
| :-- | :-- | :-- |
| RPEJSON → PEC | **有条件**："只有X移动和Y移动事件能够完全配对时谱面才能转换为PEC" | [事件](https://pgrfm.miraheze.org/wiki/%E4%BA%8B%E4%BB%B6) |
| RPEJSON → PEC | 速度语义会漂移："由于RPE和PE的窗口长宽不同，把RPEJSON转换为PEC谱面可能出现不符合预期的表演" | [速度](https://pgrfm.miraheze.org/wiki/%E9%80%9F%E5%BA%A6) |
| 官谱 JSON → RPEJSON | 可行但非线性缓动需拟合：官谱"不存在'缓动'这一概念，使用线性事件叠出缓动" | [官谱JSON](https://pgrfm.miraheze.org/wiki/%E5%AE%98%E8%B0%B1JSON)、[缓动](https://pgrfm.miraheze.org/wiki/%E7%BC%93%E5%8A%A8) |
| 其它 | Phichain 使用自有格式，支持导出为 RPEJSON 和 官谱JSON（[Phichain](https://pgrfm.miraheze.org/wiki/Phichain)） | — |

**结论（对 BeatMorph）**：目标格式取 **RPEJSON**。它是 Phira 的原生输入，字段最完整，且有 A 级来源逐字段可查。

---

## 3. RPEJSON 顶层结构

### 3.1 根对象

| 字段 | 类型 | 说明 | 默认值 | 来源 |
| :-- | :-- | :-- | :-- | :-- |
| `BPMList` | JsonArray(BPMEvent) | BPM 列表 | — | [root](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/root.html) |
| `META` | JsonObject | 谱面元数据 | — | 同上 |
| `judgeLineList` | JsonArray(JudgeLine) | 判定线列表 | — | 同上 |
| `judgeLineGroup` | string[] | 判定线组名称列表 | — | 同上 |
| `chartTime` | double | 写谱时长，单位**秒**（141 版加入） | — | 同上 |
| `multiLineString` | string | RPE 多线编辑用；模拟器不需要 | — | 同上 |
| `multiScale` | float | RPE 多线编辑页面缩放；模拟器不需要 | — | 同上 |
| `timeTags` | JsonArray | RPE 时间标记；模拟器不需要 | — | 同上 |
| `xybind` | bool | 是否启用 XY 绑定；true 时每个 XEvent 必有一个等长 YEvent | — | 同上 |

> ⚠️ **字段名冲突**：根字段在 Phira 文档中写作 `judgeLineGroup`（单数，[root](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/root.html)），而 wiki 判定线条目又写作 `judgeLineGroups`（复数，"判定线组名称在RPEJSON中存储在 judgeLineGroups 数组属性下"，[判定线](https://pgrfm.miraheze.org/wiki/%E5%88%A4%E5%AE%9A%E7%BA%BF)）。wiki 的 RPEJSON 条目使用单数 `judgeLineGroup`（[RPEJSON](https://pgrfm.miraheze.org/wiki/RPEJSON)）。**解析时应同时接受两种拼写**。见 [§12-Q16](#12-存疑清单)。

### 3.2 `BPMList` 元素

| 字段 | 类型 | 说明 | 来源 |
| :-- | :-- | :-- | :-- |
| `bpm` | float | BPM 值 | [root](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/root.html) |
| `startTime` | beat（`int[3]`） | BPM 开始时间 | 同上 |

`@json
"BPMList": [
  { "bpm": 200.0, "startTime": [0, 0, 1] },
  { "bpm": 250.0, "startTime": [10, 1, 2] }
]
`@

（示例取自 [extra.json 文档](https://teamflos.github.io/phira-docs/en/chart-standard/extra/index.html)，其 BPM 配置使用同一 beat 表示法。）

### 3.3 `META`

| 字段 | 类型 | 说明 | 默认值 | 来源 |
| :-- | :-- | :-- | :-- | :-- |
| `RPEVersion` | int | RPE 版本，**100~160** | — | [root](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/root.html) |
| `background` | string | 背景图片相对谱面根目录路径 | — | 同上 |
| `charter` | string | 谱师名义 | — | 同上 |
| `composer` | string | 曲师 | — | 同上 |
| `id` | string | 谱面 ID；RPE 自动生成时为 long，实际存储为 string | — | 同上 |
| `illustration` | string | 曲绘画师（141 版加入） | — | 同上 |
| `level` | string | 谱面等级 | — | 同上 |
| `name` | string | 谱面名称 | — | 同上 |
| `offset` | int | **音乐偏移，单位毫秒** | — | 同上 |
| `song` | string | 音乐文件相对谱面根目录路径 | — | 同上 |

原文对 `offset` 符号的描述（[root](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/root.html)）：

> `offset` 字段为负数时，音乐应该在谱面开始前 `-offset` 毫秒时播放；为正数时，音乐应该在谱面开始后 `offset` 毫秒时播放。

⚠️ 该表述本身语义缠绕（"谱面开始前播放"与"谱面开始后播放"的参照物未定义），且与官谱 offset 的秒制/符号约定不同。见 [§12-Q11](#12-存疑清单)。

### 3.4 版本兼容陷阱

- RPE 1.5.0 ~ RPE 1.6.0（不含 1.6.0，含 Alpha 版）期间，`META.RPEVersion` 字段**保持为 150** 未改（[root](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/root.html)）。
- RPE 1.6.1 期间，`META.RPEVersion` 字段值**保持为 160** 未改（同上）。
- → **不能仅凭 `RPEVersion` 判定实际能力集**。

---

## 4. 判定线对象（JudgeLine）

### 4.1 字段表

| 字段 | 类型 | 说明 | 默认值 | 来源 |
| :-- | :-- | :-- | :-- | :-- |
| `Group` | int | 判定线所属组（对应根 `judgeLineGroup` 数组下标）；**不影响显示效果** | 0 | [judgeLine](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/judgeLine.html) |
| `Name` | string | 判定线名称；**不同判定线可以重名，不影响显示效果** | `Untitled` | 同上 |
| `Texture` | string | 判定线纹理；非默认时为相对谱面根目录的路径 | `line.png` | 同上 |
| `anchor` | float[2] | 纹理锚点（142 版加入） | `[0.5, 0.5]` | 同上 |
| `eventLayers` | EventLayer[]? | 事件层级，**默认包含至少一个层级，最大有五个** | — | 同上 |
| `extended` | JsonObject | 特殊事件层（第 5 层级） | — | 同上 |
| `father` | int | 父线索引；**-1 表示无父线** | -1 | 同上 |
| `isCover` | int | 是否遮罩；**1 表示遮罩，其他值为不遮罩** | 1 | 同上 |
| `notes` | Note[] | 线上所有音符 | — | 同上 |
| `numOfNotes` | int | 文档原文："音符总数量(包含 FakeNote，**不包含 Hold**)" | 0 | 同上 |
| `zOrder` | int | 线 z 轴（图层），文档称范围 ±100 并自注"范围需要验证" | 0 | 同上 |
| `attachUI` | string? | UI 绑定；无绑定时**不存在本字段** | — | 同上 |
| `isGif` | bool | 纹理是否为 GIF（150 版加入） | false | 同上 |
| `posControl` / `sizeControl` / `skewControl` / `yControl` / `alphaControl` | JsonArray | 距离到属性的映射关键帧，见 [§4.5](#45-controls控制序列) | — | 同上 |
| `bpmfactor` | float | BPM 因子；**无法在 RPE 中编辑** | 1.0 | 同上 |
| `rotateWithFather` | bool | 子线是否继承父线的旋转角度 | 表头写 `true`（163 版加入） | 同上 |
| `isCover` 语义补充 | — | wiki："若为 true 则不会显示[线另一侧的音符]……如果要制作音符从判定线冒出来的效果，应当设此项为真" | — | [判定线](https://pgrfm.miraheze.org/wiki/%E5%88%A4%E5%AE%9A%E7%BA%BF) |

**`eventLayers` 的空值规则**（[judgeLine](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/judgeLine.html)）：

- 若层级为空，在某版本之前字段为 `null`，某版本及以后空层级**无字段**（已知至少 143 版时无字段）。
- 若某层级中某事件不存在，则该事件字段**不会出现**。
- 若所有层级都为空，`eventLayers` 字段**不会出现**。

> → **解析器必须对 `null` 层级 / 缺失字段 / 缺失 `eventLayers` 三者做同一化处理。**

### 4.2 事件轨清单

RPE 有 **4 个普通事件层级 + 1 个特殊事件层级**（[事件](https://pgrfm.miraheze.org/wiki/%E4%BA%8B%E4%BB%B6)、[judgeLine](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/judgeLine.html)）。

**普通事件（每层 5 条轨）**

| 轨字段名 | 名称 | 作用 | 数值单位 |
| :-- | :-- | :-- | :-- |
| `moveXEvents` | X 坐标移动事件 | 移动判定线锚点 X | 像素 |
| `moveYEvents` | Y 坐标移动事件 | 移动判定线锚点 Y | 像素 |
| `rotateEvents` | 旋转事件 | 旋转判定线（**角度，以度计**） | 度 |
| `alphaEvents` | 不透明度事件 | 设置判定线不透明度（**0~255 整数**） | 0~255 |
| `speedEvents` | 流速事件 | 设置音符流速（**RPE 速度单位：每单位 = 每秒下降 120 px**） | RPE 速度单位 |

> 五种事件的 start/end 含义与单位来自 [RPEJSON](https://pgrfm.miraheze.org/wiki/RPEJSON) 的注释块；速度单位另有 [Phira event 文档](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/event.html) 佐证 alpha 范围与坐标范围。

**特殊事件（第 5 层级，位于 `extended`）**

| 字段名 | 名称 | 说明 | 来源 |
| :-- | :-- | :-- | :-- |
| `scaleXEvents` | X 轴缩放 | 缩放判定线/纹理/文字的**宽度**；1 表示正常大小 | [extendEvent](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/extendEvent.html) |
| `scaleYEvents` | Y 轴缩放 | 缩放判定线/纹理/文字的**高度**；1 表示正常大小 | 同上 |
| `colorEvents` | 颜色事件 | `start`/`end` 为 `int[3]` RGB（0~255）；对贴图乘算染色 | 同上 |
| `textEvents` | 文字事件 | `start`/`end` 为 string；含 `font` 字段（152 版起仅自定义字体时存在） | 同上 |
| `gifEvents` | GIF 播放进度 | `start`/`end` 为 0.0~1.0 的 GIF 播放进度（150 版加入） | 同上 |
| `paintEvents` | 画笔事件 | **143 版被移除**，无法编辑 | 同上 |
| `inclineEvents` | 倾斜事件 | "开始结束数值为判定线 Z 轴倾斜角度"；疑似已弃用；默认在 `extended` 下留一个垫底事件 | 同上 |

**着色器事件不在判定线字段内**：wiki 明确着色器事件"在 `extra.json` 中"（[事件](https://pgrfm.miraheze.org/wiki/%E4%BA%8B%E4%BB%B6)）。

### 4.3 普通事件的字段与插值语义

除 `speedEvents` 外，每个普通事件元素包含（[event](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/event.html)）：

| 字段 | 类型 | 说明 | 默认值 |
| :-- | :-- | :-- | :-- |
| `bezier` | int | 缓动是否为贝塞尔曲线，0 否 / 1 是 | 0 |
| `bezierPoints` | float[4] | 贝塞尔控制点，`bezier` 为 1 时生效 | `[0.0, 0.0, 0.0, 0.0]` |
| `easingLeft` | float | 缓动左边界，最小 0.0，最大 1.0 | 0.0 |
| `easingRight` | float | 缓动右边界，最小 0.0，最大 1.0 | 1.0 |
| `easingType` | int | 缓动类型（1~29，见 [§4.4](#44-缓动-easingtype-对照表)） | 1 |
| `linkgroup` | int | —（文档仅列字段，无描述） | — |
| `start` | float | 事件开始时数值 | — |
| `startTime` | beat | 事件开始时间 | — |
| `end` | float | 事件结束时数值 | — |
| `endTime` | beat | 事件结束时间 | — |

**`speedEvents` 的特殊性**：只有 `startTime`、`endTime`、`start`、`end`、`linkgroup` 五个字段；**没有** `bezier` / `bezierPoints`（[event](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/event.html)）。

**插值公式（可直接照写）**（[judgeLine](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/judgeLine.html)）：

`@python
def easing_interpolation(t, st, et, sv, ev, f):
    if t == st:
        return sv
    return f((t - st) / (et - st)) * (ev - sv) + sv
`@

**取值规则（逐事件遍历，命中即返回，未命中取默认值）**（[event](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/event.html)）：

`@python
def GetEventValue(t, es, default):
    for e in es:
        if e.startTime.value <= t <= e.endTime.value:
            if isinstance(e.start, float | int):
                return easing_interpolation(t, e.startTime.value, e.endTime.value, e.start, e.end, e.easingFunc)
            elif isinstance(e.start, str):
                return e.start
            elif isinstance(e.start, list):          # colorEvents
                r = easing_interpolation(t, ..., e.start[0], e.end[0], e.easingFunc)
                g = easing_interpolation(t, ..., e.start[1], e.end[1], e.easingFunc)
                b = easing_interpolation(t, ..., e.start[2], e.end[2], e.easingFunc)
                return (r, g, b)
    return default
`@

**官方给出的补洞预处理**（`_init_events`，[judgeLine](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/judgeLine.html)）：把事件按 `startTime` 排序，在相邻事件之间若存在时间空隙，则插入一个"保持前一事件终值"的常量事件；并在末尾追加一个从最后事件结束时间到 `Beat(31250000, 0, 1)` 的常量事件。

> → **解析器必须实现这个补洞步骤**，否则事件间隙期间的值无定义。

**事件列表的规范化条件（独立来源，C 级）**（[Lchzh Docs·相关计算](https://docs.lchzh.top/learning/phigros/calc)）：

> 规范的判定线事件列表应该满足以下条件：
> - 按照事件的 `startTime` 从小到大排序。
> - 对于速度事件列表，第一个事件的 `startTime` 为 0。
> - 对于其它事件列表，第一个事件的 `startTime` 为一个足够小的数，如 -999999。
> - 紧接着，每个事件的 `startTime` 应该与上一个事件的 `endTime` 相等。
> - 最后一个事件的 `endTime` 为一个足够大的数，如 1000000000。

该页还给出"速度事件的 `floorPosition` 表示事件开始时刻的垂直位置，单位 Y"以及判定线实时速度 / 垂直位置的计算公式——**但该页公式以 MathJax 渲染，纯文本抓取时数学内容丢失，故本文未能转录**（见 [§12-Q2](#12-存疑清单)）。

**跨层级求和规则（关键）**（[事件](https://pgrfm.miraheze.org/wiki/%E4%BA%8B%E4%BB%B6)）：

> 判定线的最终**普通事件**值将会等于每个**普通事件**层级的事件值**相加**，而特殊事件由于只有一层，所以其值直接等于最终特殊事件值。

官方 Python 参考实现（[judgeLine](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/judgeLine.html)）：

`@python
def GetPos(self, t, master):
    linePos = [0.0, 0.0]
    for layer in self.eventLayers:
        linePos[0] += self.GetEventValue(t, layer.moveXEvents, 0.0)
        linePos[1] += self.GetEventValue(t, layer.moveYEvents, 0.0)
    if self.father != -1:
        fatherPos = master.JudgeLineList[self.father].GetPos(t, master)
        linePos = list(map(lambda x, y: x + y, linePos, fatherPos))
    return linePos

# rotate 与 alpha 同样跨层求和：
for layer in self.eventLayers:
    lineAlpha  += GetEventValue(t, layer.alphaEvents, 0.0 if (t >= 0.0 or self.attachUI is not None) else -255.0)
    lineRotate += GetEventValue(t, layer.rotateEvents, 0.0)
`@

**默认值语义（可直接照写）**：

- `lineAlpha` 的默认值为 `0.0`（当 `t >= 0` 或有 `attachUI`），否则 `-255.0`；最终 `lineAlpha / 255` 归一化到 0~1（[judgeLine](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/judgeLine.html)）。
- `lineScaleX` / `lineScaleY` 默认 `1.0`（同上）。
- `lineColor` 默认 `defaultColor`，但**若存在 `textEvents` 则默认 (255, 255, 255)**（同上）。

### 4.4 缓动 `easingType` 对照表

1~29 的完整对照（[extend](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/extend.html)）：

| 值 | 缓动 | 值 | 缓动 | 值 | 缓动 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| 1 | Linear | 11 | In Quart | 21 | In Back |
| 2 | Out Sine | 12 | In Out Cubic | 22 | In Out Circ |
| 3 | In Sine | 13 | In Out Quart | 23 | In Out Back |
| 4 | Out Quad | 14 | Out Quint | 24 | Out Elastic |
| 5 | In Quad | 15 | In Quint | 25 | In Elastic |
| 6 | In Out Sine | 16 | Out Expo | 26 | Out Bounce |
| 7 | In Out Quad | 17 | In Expo | 27 | In Bounce |
| 8 | Out Cubic | 18 | Out Circ | 28 | In Out Bounce |
| 9 | In Cubic | 19 | In Circ | 29 | In Out Elastic |
| 10 | Out Quart | 20 | Out Back | — | （29 无法在速度事件使用；RPE 1.7.0 恢复其使用） |

同表另注：wiki 指出 "fixed 型（0 号）"仅奇谱发生器有，"保存时事件自动转换为钩定线性事件"；quint 与 expo 的 **IO 型不存在**（[缓动](https://pgrfm.miraheze.org/wiki/%E7%BC%93%E5%8A%A8)）。

**逐函数定义（Python，可直接照抄）** —— [extend](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/extend.html) 给出完整 29 个 lambda：

`@python
ease_funcs = [
  lambda t: t,                                   # 1  linear
  lambda t: math.sin((t * math.pi) / 2),         # 2  out sine
  lambda t: 1 - math.cos((t * math.pi) / 2),     # 3  in sine
  lambda t: 1 - (1 - t) * (1 - t),               # 4  out quad
  lambda t: t ** 2,                              # 5  in quad
  # ... 6~29 见来源页
]
`@

**缓动类型归一化（必须实现）**（[judgeLine](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/judgeLine.html)、[event](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/event.html)）：

`@python
if not isinstance(easingType, int): easingType = 1
easingType = 1 if easingType < 1 else (len(ease_funcs) if easingType > len(ease_funcs) else easingType)
`@

**缓动切割（`easingLeft` / `easingRight`）**（[缓动](https://pgrfm.miraheze.org/wiki/%E7%BC%93%E5%8A%A8)）：

> 设原缓动函数为 `f(t)`，新缓动函数为 `g(t)`，当截取左边界为 `l`，截取右边界为 `r` 时（`0 ≤ l < r ≤ 1`），`g(t) = [f(r) - f(l)] · f((t - l) / (r - l)) + f(l)`

同源推论：1 号（Linear）的切割不起作用；4 号（Out Quad）改变左边界不起作用；5 号（In Quad）改变右边界不起作用。

**贝塞尔缓动**（RPE 1.3.0 引入，[缓动](https://pgrfm.miraheze.org/wiki/%E7%BC%93%E5%8A%A8)）：使用起止点分别为 `(0,0)`、`(1,1)` 的**三次贝塞尔曲线**；控制点须满足 `x ∈ [0, 1]`。参数方程：

`@
x = 3·x_C1·(1-t)²·t + 3·x_C2·(1-t)·t² + t³
y = 3·y_C1·(1-t)²·t + 3·y_C2·(1-t)·t² + t³
`@

同一来源给出 TypeScript 参考实现（`BezierEasing`，256 段折线近似 + 跳跃数组），可按需取用。

**速度事件缓动的历史语义（三重语义，必须按版本分支）**：

| 版本 | 语义 | 来源 |
| :-- | :-- | :-- |
| < 1.6 | 速度事件只能线性变化 | [速度](https://pgrfm.miraheze.org/wiki/%E9%80%9F%E5%BA%A6) |
| 1.6.2 | 支持普通缓动，但"非线性缓动实际上是描述 floorPosition 而非速度的变化趋势" | [速度](https://pgrfm.miraheze.org/wiki/%E9%80%9F%E5%BA%A6) |
| 1.7 | 重构，回归原始逻辑，"所有缓动描述速度变化趋势" | [速度](https://pgrfm.miraheze.org/wiki/%E9%80%9F%E5%BA%A6) |

RPE 作者原文（[event](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/event.html) 引用）：

> 速度事件缓动不为 1 时，实际的速度变化与缓动的**导函数**形状相同，从而 floorposition 的变化遵循缓动曲线。为了兼容性，缓动为 1 时我们保持原含义不变，也即缓动为 1 和缓动为 5 都代表二次型的 floorposition 变化。

⚠️ **同一页面自相矛盾**：该页末尾又写"音符流速事件不支持缓动，即只有线性变化"。见 [§12-Q6](#12-存疑清单)。

### 4.5 Controls（控制序列）

以 `Control` 结尾的属性，"指定音符与判定线的距离到其五个属性数值的映射"（[判定线](https://pgrfm.miraheze.org/wiki/%E5%88%A4%E5%AE%9A%E7%BA%BF)）。全部字段（[controls](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/controls.html)）：

| Control | 字段 | 说明 | 默认值 |
| :-- | :-- | :-- | :-- |
| 通用 | `easing` | int，到下一个关键帧数值的缓动类型 | 1 |
| 通用 | `x` | float，**音符与判定线的纵向距离** | — |
| `alphaControl` | `alpha` | float，note 不透明度 | 1.0 |
| `sizeControl` | `size` | float，note 大小倍率（**真正的大小，不只是宽度**；**无法影响 Hold**） | 1.0 |
| `posControl` | `pos` | float，note 的 `positionX` **参数倍率**（**不能控制 Hold**） | — |
| `yControl` | `y` | float，（文档标注"待补充"） | — |
| `skewControl` | `skew` | float，（文档标注"待补充"；**对 Hold 无效**） | — |

混合公式：`noteAlpha = noteAlpha * nowAlpha`（先把 0~255 转 0~1 再算）（[controls](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/controls.html)）。

### 4.6 锚点、贴图与 UI 绑定

- `anchor` 为 `float[2]`，默认 `[0.5, 0.5]`（以锚点作为贴图中心绘制贴图）；`[0, 0]` 表示以锚点为**左下角**绘制（[故事版](https://pgrfm.miraheze.org/wiki/%E6%95%85%E4%BA%8B%E7%89%88)）。x 默认为 0.5 即中心，1 时纹理向左移，0 时向右移；y 类推（[extend](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/extend.html)）。**该属性不改变判定线位置，只规定图片位置。**
- 默认贴图是 `line.png`，"一个白色细长矩形"（[故事版](https://pgrfm.miraheze.org/wiki/%E6%95%85%E4%BA%8B%E7%89%88)）。
- 贴图缩放方式：**"每个像素对应一个 RPE 坐标系单位，忽略宽高比"**（[extend](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/extend.html)）。
- `attachUI` 取值：`pause` / `combonumber` / `combo` / `score` / `bar` / `name` / `level`（[故事版](https://pgrfm.miraheze.org/wiki/%E6%95%85%E4%BA%8B%E7%89%88)）。绑定了 UI 的判定线**不会再显示自身贴图**，但文字事件仍生效。UI 相对未绑定状态所作的变换，与判定线相对 `(0, 0)` 0 角度判定线所作的变换相同（即"UI 在未绑定状态下绑在一条初始判定线上"）。
- 有文字事件的判定线**始终隐藏**，只显示文字（即使该处没有文字事件），并清除自定义纹理；无颜色事件时文字颜色始终为白色（[extendEvent](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/extendEvent.html)）。

---

## 5. Note 对象

### 5.1 字段表（RPEJSON，A 级来源）

| 字段 | 类型 | 说明 | 默认值 | 加入版本 | 来源 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| `above` | int | **1 为从线的正面下落，其他数字为从线的背面下落** | 1 | — | [note](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/note.html) |
| `alpha` | int | 音符不透明度，0 完全透明，255 完全不透明 | 255 | — | 同上 |
| `startTime` | beat | 音符开始时间；`type` 为 2 时为 Hold 开始时间 | — | — | 同上 |
| `endTime` | beat | Hold 结束时间；非 Hold 与 `startTime` 一致 | — | — | 同上 |
| `isFake` | int | **1 为假，其他数为真** | 0 | — | 同上 |
| `positionX` | float | **音符相对于判定线中心点的 X 坐标** | — | — | 同上 |
| `size` | float | 音符大小倍率（**实际只控制宽度**） | 1.0 | — | 同上 |
| `speed` | float | 流速倍率 | 1.0 | — | 同上 |
| `type` | int | 1 Tap / 2 Hold / 3 Flick / 4 Drag | Tap | — | 同上 |
| `visibleTime` | float | 音符可见时间，**单位秒** | 999999.0000 | — | 同上 |
| `yOffset` | float | 音符 Y 轴偏移，正数向上、负数向下，同时偏移打击特效的位置 | 0 | — | 同上 |
| `hitsound` | string? | 自定义打击音文件相对谱面根目录路径；无自定义音效时**不存在** | — | 142 | 同上 |
| `judgeArea` | float | 判定区域宽度倍率 | 1.0 | 170 | 同上 |
| `tint` 或 `color` | int[3] | 音符颜色 `[R,G,B]`，0~255 | `[255,255,255]` | 170 | 同上 |
| `tintHitEffects` | int[3]? | 打击特效颜色 | `[255,255,255]` | 170 | 同上 |

**Beat 三元组的解析（全格式统一）**（[beat](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/beat.html)、[KPAJSON](https://pgrfm.miraheze.org/wiki/KPAJSON)）：

`@
beat 是 int[3]，在 RPE 中显示为 [0]:[1]/[2]
beat    = [1] / [2] + [0]
seconds = 60 / BPM * beat
`@

（KPAJSON 条目把它写作 `[拍数, 分子, 分母]`，与上式一致。）

### 5.2 `type` 枚举——RPE 与官谱**不同**（重大陷阱）

| 值 | **RPEJSON / PEC** | **官谱 JSON（phi/Official）** |
| :-- | :-- | :-- |
| 1 | Tap | Tap |
| 2 | **Hold** | **Drag** |
| 3 | **Flick** | **Hold** |
| 4 | **Drag** | **Flick** |
| 其它 | — | 表现为不可见也不可判定 |

- RPE 侧来源：[note](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/note.html)（"Note类型对照"表）、[RPEJSON](https://pgrfm.miraheze.org/wiki/RPEJSON)（`enum NoteType { tap=1, drag=4, flick=3, hold=2 }`）、[PEC](https://pgrfm.miraheze.org/wiki/PEC)（`n1`=Tap、`n4`=Drag、`n3`=Flick、`n2`=Hold）。
- 官谱侧来源：[Lchzh Docs](https://docs.lchzh.top/learning/phigros/)（二手来源，见 [§12-Q12](#12-存疑清单)）。

> 🚨 **这是最容易写出静默错误的点**：同一个数字 2 在两种格式里分别是 Hold 和 Drag。

### 5.3 `above` / 侧别语义

**A 级来源的表述**（[note](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/note.html)）：

> `above` 字段在为 1 时，音符从判定线的**正面**下落，其他数值时从判定线的**背面**下落。

作为对照，官谱格式**不使用 `above` 字段**，而是把音符分成两个数组（[官谱JSON](https://pgrfm.miraheze.org/wiki/%E5%AE%98%E8%B0%B1JSON)、[phi/judgeLine](https://teamflos.github.io/phira-docs/en/chart-standard/chart-format/phi/judgeLine.html)）：

- `notesAbove`：正面下落的音符 / "判定线上面的音符列表"
- `notesBelow`：反面下落的音符 / "判定线下面的音符列表"

**取值域**：wiki 的两个条目给出不同写法——`"above": 0 | 1 | 2`（[音符](https://pgrfm.miraheze.org/wiki/%E9%9F%B3%E7%AC%A6)，并注"1为上方，其余为下方"）与 `above: Bool | 2`（[RPEJSON](https://pgrfm.miraheze.org/wiki/RPEJSON)，注释"音符是否在判定线上方（2为下方）"）。

> **可确认**：语义是二值的"1 vs 非 1"，默认 1。
> **未确认**：是否只允许 `{0,1,2}` 三个取值。见 [§12-Q3](#12-存疑清单)。

**官方对"哪一侧"的定义**（[判定线](https://pgrfm.miraheze.org/wiki/%E5%88%A4%E5%AE%9A%E7%BA%BF)）：

> 在开启了遮罩的情况下，第一、二象限及 Y 轴正半轴上只能出现**上侧**音符，第三、四象限及 Y 轴负半轴上只能出现**下侧**音符，X 轴上可以出现任意方向的音符。

结合"锚点面向的方向就是坐标系 Y 轴正半轴"——**【推断】** `above == 1`（"正面"/"上方"）对应判定线局部坐标系的 **+Y 侧**。推断依据：官方把 `notesAbove` 描述为"判定线**上面**的音符"，且遮罩规则把"上侧"绑定到 +Y 半轴。**该等价关系未在任何来源中被直接写成一句话。**

### 5.4 `speed`、`yOffset`、`visibleTime` 的精确语义

| 事实 | 来源 |
| :-- | :-- |
| 音符的实际速度为**当前判定线的速度乘以该值** | [音符](https://pgrfm.miraheze.org/wiki/%E9%9F%B3%E7%AC%A6)、[速度](https://pgrfm.miraheze.org/wiki/%E9%80%9F%E5%BA%A6) |
| `yOffset` 的偏移量为 `yOffset * speed`；**若 speed 为 0，则 yOffset 设为任何数偏移量都为 0** | [note](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/note.html) |
| `yOffset` **并非相对于判定线位置的绝对偏移** | 同上 |
| 实际偏移**受音符下落朝向影响**——"若该值为负数，对不同下落朝向的音符的影响都是向**下落朝向**偏移，而不是向同一方向"（即偏移在**局部系**中施加） | [音符](https://pgrfm.miraheze.org/wiki/%E9%9F%B3%E7%AC%A6) |
| "0 速 Note 必须在判定线上判定" | 同上 |
| 若音符**实际速度**为负数**且判定线启用遮罩**，则音符自始至终不会显示，但仍会出现判定效果 | [音符](https://pgrfm.miraheze.org/wiki/%E9%9F%B3%E7%AC%A6) |
| 若**速度事件为正数、音符本身的速度为负数**，则音符仍会显示，只是下落方向与正常音符相反 | [速度](https://pgrfm.miraheze.org/wiki/%E9%80%9F%E5%BA%A6) |
| 若在音符被判定之前速度事件的值始终为负，则音符始终不显示——"利用这点可以做出只有打击特效的隐形音符" | 同上 |
| `visibleTime` 语义：设 1 表示音符在判定前 1 秒才会显示 | [音符](https://pgrfm.miraheze.org/wiki/%E9%9F%B3%E7%AC%A6) |
| 假音符"没有判定，没有打击特效与音效，不计分，不计物量，若为 Hold 则始终显示为未打击样式" | [note](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/note.html) |
| 染色方式为**顶点颜色乘法**：`noteColor = noteColor * color` | 同上 |

**【推断】负 speed 的几何含义**：`speed < 0` 时音符沿局部 −Y 方向运动，即**远离**判定线；"速度为正 → 靠近判定线"这一约定由 [速度](https://pgrfm.miraheze.org/wiki/%E9%80%9F%E5%BA%A6) 直接给出（"Note 往靠近判定线的方向移动，速度为正；往远离判定线的方向移动，速度为负"），故该推断依据充分。

### 5.5 KPAJSON / PhiZone Player 的扩展字段（非 RPE 标准）

| 字段 | 来源 | 说明 |
| :-- | :-- | :-- |
| `visibleBeats` | [音符](https://pgrfm.miraheze.org/wiki/%E9%9F%B3%E7%AC%A6)（KPAJSON 新增） | 替代 `visibleTime`；原属性仍存在但不发挥作用 |
| `absoluteYOffset` | 同上 | 替代 `yOffset`；有意义时无视 `yOffset` |
| `zIndex` | 同上（PhiZone Player 新增，部分被 RPE 1.7 收入标准） | Note 的 Z 轴层级 |
| `zIndexHitEffects` | 同上 | 判定效果 Z 轴层级，默认 7 |
| `tint` / `color` | 同上 | RGB 三元组；最初名为 `color`，后亦支持 `tint` |
| `judgeSize` / `judgeArea` | 同上 | `judgeArea` 为最终确认名，`judgeSize` 为 PhiZone Player 方言 |

---

## 6. 坐标系统与几何（**本节是 RFC-0029 Q5 的核心答案**）

### 6.1 判定线局部坐标系（官方定义）

[判定线](https://pgrfm.miraheze.org/wiki/%E5%88%A4%E5%AE%9A%E7%BA%BF) 条目原文：

> 每条判定线拥有一个**平面直角坐标系**。判定线锚点就是坐标系原点，判定线本身就是 X 轴，锚点面向的方向就是坐标系 Y 轴正半轴的方向。根据这三点，可以唯一确定判定线的坐标系。

> 判定线上的音符**永远平行于判定线的 X 轴**，其下落方向**永远平行于判定线的 Y 轴**。
> 音符的 X 坐标控制音符落点的 X 坐标，Y 值偏移控制音符落点的 Y 坐标。

> 每条判定线的位置和方向由一个**锚点**和一个**角度（顺时针）**描述。除此之外还有透明度以及速度属性。

### 6.2 任意方向下落的机制（RFC-0029 §2.1 的"决定性事实"）

**机制链条（全部由已查证事实构成）**：

1. 音符在判定线**局部系**中，永远沿**局部 ±Y 轴**方向接近判定线（[判定线](https://pgrfm.miraheze.org/wiki/%E5%88%A4%E5%AE%9A%E7%BA%BF)）。
2. 判定线本身由 `rotateEvents` 事件轨旋转，角度单位为**度**，"角度越大，判定线就会越往顺时针方向旋转"；角度 0 时判定线水平（[事件](https://pgrfm.miraheze.org/wiki/%E4%BA%8B%E4%BB%B6)）。
3. 判定线的锚点由 `moveXEvents` / `moveYEvents` 平移（单位：像素）（[事件](https://pgrfm.miraheze.org/wiki/%E4%BA%8B%E4%BB%B6)）。
4. 因此**同一条物理音符在屏幕系中的接近方向 = 局部 −Y（或 +Y）方向经该时刻旋转角旋转后的方向**，随 `rotateEvents` 的数值连续变化 → **屏幕方向上可为任意角度**。

> ✅ **结论：RFC-0029 §2.1 的建模前提成立。** 判定线局部系（`positionX`, `side`）是谱面的**原生存储系**；屏幕系是它经 move/rotate 事件变换后的像。把 `W` 简化为单一轴会丢掉 `above` 自由度这一判断，与格式事实一致。

**"哪一侧"与局部系符号**：见 [§5.3](#53-above--侧别语义)。遮罩规则把"上侧"绑定到局部 +Y 半轴与一、二象限（[判定线](https://pgrfm.miraheze.org/wiki/%E5%88%A4%E5%AE%9A%E7%BA%BF)）。

### 6.3 屏幕坐标系与转换公式

**RPE 屏幕系**（[事件](https://pgrfm.miraheze.org/wiki/%E4%BA%8B%E4%BB%B6)、[event](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/event.html)）：

> RPE 中，屏幕可见的 x 坐标范围为 **-675 ~ 675**，y 坐标范围为 **-450 ~ 450**。
> 坐标系锚点位于**屏幕中心**。

→ 舞台为 **1350 × 900**，宽高比 **3 : 2**。

**RPE 坐标 → 归一化屏幕坐标（官方 Python 参考实现，可直接照抄）**（[judgeLine](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/judgeLine.html)）：

`@python
def conrpepos(x: float, y: float):
    return (x + 675) / 1350, 1.0 - (y + 450) / 900
`@

即：

`@
sx = (x + 675) / 1350          # 0 = 左边缘, 1 = 右边缘
sy = 1 - (y + 450) / 900       # 0 = 上边缘, 1 = 下边缘（注意 y 轴翻转）
`@

**判定线状态的完整求值链**（[judgeLine](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/judgeLine.html)）：

`@
linePos    = Σ_layers (moveXEvents, moveYEvents)  + linePos(father, t)
lineAlpha  = Σ_layers (alphaEvents) / 255
lineRotate = Σ_layers (rotateEvents)              # 度，顺时针为正
lineColor  = extended.colorEvents  (默认 defaultColor；有 textEvents 时默认 (255,255,255))
lineScaleX = extended.scaleXEvents (默认 1.0)
lineScaleY = extended.scaleYEvents (默认 1.0)
lineText   = extended.textEvents   (默认 null)
→ 返回 (conrpepos(*linePos), lineAlpha/255, lineRotate, lineColor, lineScaleX, lineScaleY, lineText)
`@

**父线复合**：子线位置 = 自身事件位置 **+ 父线位置**（逐层求和后叠加），父线可嵌套（[judgeLine](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/judgeLine.html)）。父线角度**默认不影响**子线角度，但会影响父线坐标轴朝向，从而导致子线锚点位置改变（[判定线](https://pgrfm.miraheze.org/wiki/%E5%88%A4%E5%AE%9A%E7%BA%BF)）。此外"当谱面展示比例不为 3:2 时，在**纵向**上对判定线锚点坐标进行拉伸。不仅孤立判定线所在的根坐标系会被纵向拉伸，父线的坐标系也会被纵向拉伸"（同上）。

**官谱格式的屏幕系**（[官谱JSON](https://pgrfm.miraheze.org/wiki/%E5%AE%98%E8%B0%B1JSON)）：

> 官谱规定屏幕**左下角**为原点 (0,0)，右上角为 (1,1)，显然位于屏幕中心的位置为 (0.5, 0.5)

官谱移动事件的坐标读取方式随 `formatVersion` 变化（[Lchzh Docs](https://docs.lchzh.top/learning/phigros/)）：

| formatVersion | 原点 | 范围/单位 | 字段 |
| :-- | :-- | :-- | :-- |
| 1 | 左下角 | 右上角 (880, 520)；`start`/`end` = `1000x + y` | `start`, `end` |
| 3 | 左下角 | 右上角 (1, 1) | `start`/`end` = x，`start2`/`end2` = y |
| 其它 | **屏幕中心** | 两方向单位长度均为 `0.1 H` | `start`/`end` = x，`start2`/`end2` = y |

### 6.4 线长单位（**关键未决项**）

**已查证的三个不同刻度**：

| 语境 | 量 | 定义 | 来源 |
| :-- | :-- | :-- | :-- |
| 官谱 | `positionX` 的单位 **X** | `1 X = 0.05625 W = 108 px`（1920×1080） | [Lchzh Docs](https://docs.lchzh.top/learning/phigros/)（C 级） |
| 官谱 | `floorPosition` 与速度事件的单位 **Y** | `1 Y = 0.6 H = 648 px`（1920×1080） | 同上 |
| RPE | 屏幕可见 x 范围 | `-675 ~ 675`（舞台宽 1350） | [事件](https://pgrfm.miraheze.org/wiki/%E4%BA%8B%E4%BB%B6)（B 级）/ [event](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/event.html)（A 级） |
| RPE | 速度事件数值单位 | `1` = 每秒下降 **120 px** | [速度](https://pgrfm.miraheze.org/wiki/%E9%80%9F%E5%BA%A6)、[RPEJSON](https://pgrfm.miraheze.org/wiki/RPEJSON) |
| Phira | `info.yml.lineLength` | 默认 `6.0`；文档原文："谱面中线条的长度，**单位待补充**（涉及到渲染细节，文档待补充）" | [谱面信息](https://teamflos.github.io/phira-docs/chart-standard/chartinfo.html)（A 级，但作者自认未写完） |

> ⚠️ **RPEJSON 中 `positionX` 的物理单位在任何来源中都未被显式写出。** Phira 文档只说"相对于判定线中心点的 X 坐标"，wiki 只说"音符在判定线上的 X 坐标"。
>
> **【推断】** RPE 的 `positionX` 与 RPE 屏幕坐标系同刻度（有效范围约 `-675 ~ 675`，即舞台宽度 1350 为全长）。推断依据：(a) RPE 屏幕 x 范围确为 -675~675（[事件](https://pgrfm.miraheze.org/wiki/%E4%BA%8B%E4%BB%B6)）；(b) 官方 `conrpepos(x, y)` 直接把判定线位置 `x` 当作该刻度换算到屏幕（[judgeLine](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/judgeLine.html)）；(c) 判定线贴图"每个像素对应一个 RPE 坐标系单位"（[extend](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/extend.html)），说明该坐标系就是绘制单位。**但 `positionX` 是否与 `moveX` 同刻度、是否可超出 ±675，均未查证**。见 [§12-Q2](#12-存疑清单)。

### 6.5 音符屏幕位置——已确认片段与推断公式

**已确认的片段**：

1. 音符落点 = 判定线上 `positionX` 处的点，再叠加 `yOffset` 造成的局部 Y 偏移（[判定线](https://pgrfm.miraheze.org/wiki/%E5%88%A4%E5%AE%9A%E7%BA%BF)）。
2. `floorPosition` 是"**0 时刻音符到判定线的逻辑距离**，它通过对判定线速度曲线定积分解算出"（[Floor Position](https://pgrfm.miraheze.org/wiki/Floor_Position)）。
3. 速度是 `floorPosition` 的**导函数**；`floorPosition` 是速度"从 0 时刻到 Note 被判定时的**积分**"（[速度](https://pgrfm.miraheze.org/wiki/%E9%80%9F%E5%BA%A6)）。
4. 音符在局部系中永远平行于判定线 X 轴（即音符的长边沿局部 X），下落方向永远平行于局部 Y 轴（[判定线](https://pgrfm.miraheze.org/wiki/%E5%88%A4%E5%AE%9A%E7%BA%BF)）。

**【推断】音符在时刻 t 的局部坐标**：

`@
局部 X = positionX                        (可能再乘 posControl 的 pos 倍率)
局部 Y = sign(above==1 ? +1 : -1) · (floorPosition(t_judge) - floorPosition(t)) · k
         └ floorPosition 差的符号由实际速度决定；k 为单位换算常数（未查证）
`@

推断依据：`floorPosition` 被定义为"0 时刻音符到判定线的**逻辑距离**"的积分形式，且速度是它的导函数，说明**音符当前距判定线的距离 = floorPosition 在"当前时刻"与"判定时刻"之间的差**。**`k`（RPE 速度单位 → RPE 坐标系单位 的换算）未查证**；唯一相关的确证数字是"1 个 RPE 速度单位 = 每秒下降 120 px"（[速度](https://pgrfm.miraheze.org/wiki/%E9%80%9F%E5%BA%A6)），但该 px 是否就是 RPE 坐标系单位，未查证。见 [§12-Q2](#12-存疑清单)。

**【推断】局部 → 屏幕的变换**（依据 §6.1/§6.3 的已确认事实组合）：

`@
屏幕坐标 = R_line(t) · 局部坐标 + pos(line, t)
其中 R_line(t) 为绕锚点、顺时针旋转 rotate(line, t) 度的旋转矩阵
     pos(line, t) 为 moveX/moveY 跨层求和（+ 父线位置）
`@

该式是 §6.1"锚点 + 顺时针角度"与 §6.3 `GetPos` 的直接组合，**RPEJSON 中没有以公式形式写出**。

---

## 7. 时间与节拍

### 7.1 两套时间制

| 格式 | 时间表示 | 单位 | BPM 来源 |
| :-- | :-- | :-- | :-- |
| **RPEJSON** | `int[3]` beat 三元组 `[整拍, 分子, 分母]` | **拍** | 根级 `BPMList`（全局），判定线另有 `bpmfactor` 修正 |
| **官谱 JSON** | **整数** | **1/32 拍**（128 分音符） | **每条判定线自带 `bpm` 字段** |

来源：RPE 时间制 [beat](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/beat.html)；官谱时间制 [Lchzh Docs](https://docs.lchzh.top/learning/phigros/)（"T：1 T = 1.875/BPM s，相当于 128 分音符，是 Note.time / Note.holdTime 和各种事件的时间单位"）与 [phi/root](https://teamflos.github.io/phira-docs/en/chart-standard/chart-format/phi/root.html)（"所有的时间单位均为 128 分音符 即 60 / 32 / bpm，下文简写为 1.875 / bpm"）；官谱每线 bpm 见 [phi/judgeLine](https://teamflos.github.io/phira-docs/en/chart-standard/chart-format/phi/judgeLine.html)。

**beat → 秒（单一 BPM 段内）**（[beat](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/beat.html)）：

`@
beat    = RPEBeat[1] / RPEBeat[2] + RPEBeat[0]
seconds = 60 / BPM * beat
`@

**多 BPM 段的换算（官方 Python 参考实现）**（[beat](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/beat.html)）：

`@python
def sec2beat(self, t: float, bpmfactor: float):
    beat = 0.0
    for i, e in enumerate(self.BPMList):
        bpmv = e.bpm / bpmfactor
        if i != len(self.BPMList) - 1:
            et_beat = self.BPMList[i + 1].startTime.value - e.startTime.value
            et_sec  = et_beat * (60 / bpmv)
            if t >= et_sec:
                beat += et_beat
                t -= et_sec
            else:
                beat += t / (60 / bpmv)
                break
        else:
            beat += t / (60 / bpmv)
    return beat

def beat2sec(self, t: float, bpmfactor: float):
    sec = 0.0
    for i, e in enumerate(self.BPMList):
        bpmv = e.bpm / bpmfactor
        if i != len(self.BPMList) - 1:
            et_beat = self.BPMList[i + 1].startTime.value - e.startTime.value
            if t >= et_beat:
                sec += et_beat * (60 / bpmv)
                t -= et_beat
            else:
                sec += t * (60 / bpmv)
                break
        else:
            sec += t * (60 / bpmv)
    return sec
`@

⚠️ 该页要求 `beat2sec(sec2beat(x)) == x` 与 `sec2beat(beat2sec(x)) == x` 均为 True，并声明 `bpmv = e.bpm / bpmfactor`；但**同站点 [judgeLine](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/judgeLine.html) 页的示例代码写 `bpmv = e.bpm * bpmfactor`**。两者矛盾。见 [§12-Q7](#12-存疑清单)。

**`bpmfactor` 的语义（文字描述层面一致）**：

- wiki："指定此判定线应当应用当前 BPM **几分之一**的 BPM（即该判定线 BPM 为谱面 BPM 的 `1 / bpmfactor` 倍）"（[判定线](https://pgrfm.miraheze.org/wiki/%E5%88%A4%E5%AE%9A%E7%BA%BF)）。
- Phira 文档："判定线的当前 BPM 为 `nowBpm / bpmfactor`，而非 `nowBpm * bpmfactor`"（[judgeLine](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/judgeLine.html)）。

→ 文字描述一致（除法）；仅有示例代码冲突。

### 7.2 `offset` 的多处不同表述

| 格式 | 单位 | 表述 | 来源 |
| :-- | :-- | :-- | :-- |
| RPEJSON `META.offset` | **毫秒** | "为负数时，音乐应该在谱面开始前 `-offset` 毫秒时播放；为正数时，音乐应该在谱面开始后 `offset` 毫秒时播放" | [root](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/root.html) |
| 官谱 JSON `offset` | **秒** | "为正数时，谱面比音乐快，为负数时，谱面比音乐慢" | [phi/root](https://teamflos.github.io/phira-docs/en/chart-standard/chart-format/phi/root.html) |
| 官谱 JSON `offset` | **秒** | "值为非负数时，音乐立即开始，谱面延迟 offset 的绝对值秒开始。**目前为止，游戏本体从未出现 offset 为负数的谱面**" | [Lchzh Docs](https://docs.lchzh.top/learning/phigros/) |
| Phira `info.yml.offset` | **秒** | "谱面相对于音乐之间的时间延迟（秒），即该值为正时……若两种情况中音乐同时开始，则 offset 为正时谱面开始更晚；若两种情况中谱面同时开始，则 offset 为正时音乐开始更早" | [谱面信息](https://teamflos.github.io/phira-docs/chart-standard/chartinfo.html) |

⚠️ 官谱 offset 的两条表述（"正数谱面比音乐快" vs "非负时谱面延迟 |offset| 秒开始"）**方向相反**。见 [§12-Q10](#12-存疑清单)。

### 7.3 一张谱可以有几条判定线

- RPEJSON 的 `judgeLineList` 是 **JsonArray**，"包含若干个 JudgeLine"；**任何来源都未给出数量上限**（[root](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/root.html)）。
- 官谱 JSON 的 `judgeLineList` 同样是 JsonArray（[官谱JSON](https://pgrfm.miraheze.org/wiki/%E5%AE%98%E8%B0%B1JSON)、[phi/root](https://teamflos.github.io/phira-docs/en/chart-standard/chart-format/phi/root.html)）。
- **【推断】** 引擎层面无硬性上限（数组语义），但实际渲染/性能上限未查证。见 [§12-Q13](#12-存疑清单)。
- 工程上可参考的约束：RPE 引入"判定线组"用于编辑；**每条判定线属于且仅属于一个判定线组**（[判定线](https://pgrfm.miraheze.org/wiki/%E5%88%A4%E5%AE%9A%E7%BA%BF)）。

### 7.4 `speed` 与时间的相互作用（Hold 的硬约束）

- **"PEC 和官谱 JSON 中，Hold 被判定期间所在判定线不能发生速度变化。"**（[速度](https://pgrfm.miraheze.org/wiki/%E9%80%9F%E5%BA%A6)）——这是对 BeatMorph 生成器的**硬合法性约束**。
- 官谱中 Hold 的 `speed` 表示"打击时 Hold **尾**的速度倍率"，**Hold 头速度倍率恒为 1**（[Lchzh Docs](https://docs.lchzh.top/learning/phigros/)、[phi/note](https://teamflos.github.io/phira-docs/en/chart-standard/chart-format/phi/note.html)）。
- RPE 中流速为负数时"音符会向上飞"，若音符为 Hold，"在 Hold 尾出现时，整个音符都会出现（即使 Hold 还没完全回到判定线正面）"，Phira 文档注明"此行为与本家行为不符，请酌情选择"（[event](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/event.html)）。

---

## 8. 对 BeatMorph 建模的直接含义

> 本节把格式事实翻译成 RFC-0029 §3.1 强度场/条件输入的设计结论。每条结论后标出其依据的格式事实来源。

### 8.1 预测目标（标记 `(line_id, t, positionX, side, type, hold_time, ...)`）

| 标记分量 | 是否可作预测目标 | 依据 |
| :-- | :-- | :-- |
| `t`（时间） | ✅ | RPE 用 beat 三元组；可按 §7.1 无损转秒 |
| `positionX` | ✅ 连续 | [note](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/note.html)：float，相对于判定线中心点 |
| `side` / `above` | ✅ **二值**（1 vs 非 1） | 同上。⚠️ 这是**独立于 positionX 的硬自由度**，RFC-0029 §2.1 的判断成立 |
| `type` | ✅ 4 类 + 默认 Tap | 1 Tap / 2 Hold / 3 Flick / 4 Drag（[note](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/note.html)） |
| `hold_time`（=`endTime`） | ✅ 仅 Type=2 有意义 | "音符结束时间，若 type 为 2，此值为 Hold 的结束时间，否则与 startTime 一致"（同上） |
| `isFake` | ✅ 二值（1 为假，其余为真） | 同上 |