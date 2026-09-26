# Plan 00 — 核心契约：判定线 / 音符 / 谱面 / 强度场张量

> 状态：🟡 草案 ｜ 阶段：Phase 1 ｜ 负责：契约组（contracts-agent）
> 对应代码：`beatmorph/core/contracts/`（新增 `phigros.py`、`field.py`；音频常量沿用 `tensors.py`） ｜ 对应奠基章节：§2、§3.2、§3.4、§9

## 1. 目标与范围

### 交付

1. **谱面对象契约**（`beatmorph/core/contracts/phigros.py`）
   - `JudgeLine`：判定线 = **4 层普通事件轨**（`moveXEvents` / `moveYEvents` / `rotateEvents` / `alphaEvents` / `speedEvents`，**跨层求和**）+ `extended` **第 5 层**特殊事件轨；
   - `PhigrosNote`：标记 `(line_id, t, position_x, side, type, hold_time, speed, is_fake)`；
   - `PhigrosChart`：**K 条判定线 + N 个 note** + `BPMList` + META；v1 **不限制** K = 1。
2. **强度场张量契约**（`beatmorph/core/contracts/field.py`）：`ChartFieldSpec`（网格规格）、`ChartField`（模型侧 λ）、`ChartTargetField`（目标侧桶计数 + 三重掩码）。
3. **单位与几何常量（全部派生式）** + 不变量断言表（§3.8）。
4. `side` 语义的**唯一实现路径** `side_from_above()` 与 note `type` 的**双格式分派函数**。
5. 统一日志入口沿用 `beatmorph/core/logging.py`（不新增）。

### 不交付

- RPEJSON 解析器实现与数据获取（plan 02）；
- 强度场目标构建、∫λ 的两条数值路径实现（plan 03）；
- 生成主干、损失实现、解码与后处理（plan 04 / plan 05）；
- RPEJSON 写出（plan 05，`io/formats/rpejson/` 写侧）。本 plan **只冻结类型、张量形状、常量与不变量**，不含算法。

### 价值

所有模块（data / field / generation / decoder / eval）只经本契约通信：契约稳定则可并行开发；契约可静态校验（`mypy --strict`）；契约可序列化（IR ↔ JSON 无损往返）。**跨模块的物理量只有一处定义**，这是红线 7「一切派生 + 断言」的唯一落点。

## 2. 与奠基文档的对应

| 契约项 | 奠基依据 | 偏离 | 理由 |
|--------|---------|------|------|
| `JudgeLine`：4 层普通轨 + `extended` 第 5 层 | [BasePlan §3.2.2](BasePlan.md) 事件空间、[RFC-0029 §8.1](decisions/RFC-0029-phigros-continuous-chart-generation.md) 裁定 | 见偏离 1 | 实测 `eventLayers` 长度 1–5 且 `extended` 独立存在，契约须兼容两种读法 |
| `PhigrosNote` 八个标记字段 | [BasePlan §3.2.2](BasePlan.md) / [RFC-0029 §2.3](decisions/RFC-0029-phigros-continuous-chart-generation.md) | — | 直接采用 |
| `side` = ±1 枚举，不得当布尔 | [RFC-0029 §8.1](decisions/RFC-0029-phigros-continuous-chart-generation.md) Q3、[phigros-format.md §5.3](knowledges/phigros-format.md) | 见偏离 3 | 语义是「`above == 1` → 正面，**其余值** → 背面」，实测取值 `{0,1,2}` |
| `position_x` 单位 = RPE 舞台系 x 坐标（全域 `RPE_STAGE_WIDTH`） | [BasePlan §3.2.4](BasePlan.md)（A 级 prpr `parse/rpe.rs:588` 即 `position_x / (RPE_WIDTH/2)`） | — | 全部判定线共用同一 x 网格，多线**无需任何 per-line 坐标校正** |
| 可见范围 `[-RPE_STAGE_HALF_WIDTH, +RPE_STAGE_HALF_WIDTH]`，**越界只统计不钳位** | [BasePlan §3.2.4](BasePlan.md) ⚠️ 段、[phigros-units-and-geometry.md §7.5](knowledges/phigros-units-and-geometry.md) | 见偏离 2 | 钳位会改变落点分布 → 触犯红线 3 |
| 网格 `Δx = RPE_STAGE_WIDTH / N`，默认 `N = 128` | [BasePlan §3.2.4](BasePlan.md) / [RFC-0029 §3.1](decisions/RFC-0029-phigros-continuous-chart-generation.md) | — | 必须派生；须补 N ∈ {64,128,256,512} 消融与共格碰撞统计 |
| λ_k(t, x, s, c)，k = 1..K，共享 x 网格 | [BasePlan §1.2/§3.2.3](BasePlan.md)、[RFC-0029 §2.4-7](decisions/RFC-0029-phigros-continuous-chart-generation.md) | 见偏离 4 | 输出层**不得**把 K 硬编码 |
| 泊松 NLL 的 ∫λ 必须与场同网格、分辨率显式声明 | [BasePlan §3.4](BasePlan.md)、[RFC-0029 §3.2/§7](decisions/RFC-0029-phigros-continuous-chart-generation.md) | — | 契约固定**测度**，算法留 `field/` |
| G3 常数基线 `λ ≡ N/|Ω|`（非 `λ ≡ 0`） | [BasePlan §3.4](BasePlan.md)、[RFC-0029 §3.2](decisions/RFC-0029-phigros-continuous-chart-generation.md) | — | `λ ≡ 0` 时 NLL = +∞，写成契约断言 |
| 帧率由模型 config 派生 | [BasePlan §3.1](BasePlan.md)、红线 7、[POSTMORTEM](POSTMORTEM-2026-08-05-frame-rate-misalignment.md) | — | `MERT_FRAME_RATE_HZ = MERT_SAMPLE_RATE_HZ / MERT_CONV_STRIDE_PRODUCT` |
| 显式 mask 通道 | [RFC-0029 §3.3-1](decisions/RFC-0029-phigros-continuous-chart-generation.md) | — | 区分「此处无 note」「此处被遮盖」「padding」三态 |

**偏离 1（`eventLayers` 层数口径）**：RFC-0029 §8.1 写「4 层普通 + `extended` 第 5 层」，但实测 `eventLayers` 长度直方图为 `{1:506, 2:190, 3:9, 4:92, 5:60}`（23 张样本），且 `extended` 是**独立字段**、23/23 非空（[phira-dataset-survey.md §7.3](knowledges/phira-dataset-survey.md)）。→ 契约把 `event_layers` 定义为**长度 1..`RPE_MAX_EVENT_LAYERS` 的列表，原样保留长度，既不截断也不补齐**，`extended` 独立成字段；「4 层普通」只是文档口径，**不得用作解析期归一化依据**。记入 §9 开放问题（对应 phigros-format.md Q18 / 调研 Q-8）。

**偏离 2（越界事件在训练目标中的处理）**：解析阶段不钳位已由 RFC 裁定；但**目标场**的定义域是可见范围，域外 `λ ≡ 0` 会使该事件的 `log λ` 项 = −∞。→ 契约规定：域外事件**计入 `out_of_window` 计数并从事件项中显式排除**（既不钳位也不静默丢弃），该计数随数据集统计落盘。此为本 plan 新增的契约级裁定，记入 §9 待 RFC 追认。

**偏离 3（`side` 用 ±1 枚举）**：若把 `side` 建模为 `bool`，则实测的 `above = 0` 与 `above = 2` 都会静默落到同一个假值分支，而**正反两面只能靠 `== 1` 判定**。契约取 `Side.FRONT = +1 / Side.BACK = -1`（依据：两侧是关于判定线 X 轴的 Y 镜像，[phigros-units-and-geometry.md §2.1](knowledges/phigros-units-and-geometry.md) A 级），**两侧都是真值**，从而任何 `if side:` 写法都必然失效——这是一条**用类型设计挡住静默错位**的约束。

**偏离 4（场的定义域含 c 轴，积分记号以本 plan 为准）**：BasePlan §2 与 §3.4 的积分式写作 `∫∫∫ λ_k(t,x,s) dt dx ds`，而 RFC-0029 §3.1 与 BasePlan §3.2.2 的场写作 `λ_k(t, x, s, c)`。契约取**五个轴全积分**（含 c）：`∫λ ≈ Σ_{k,t,x,s,c} λ · Δt · Δx · 1 · 1`。→ 记号不一致记入 §9。

## 3. 接口契约

### 3.1 常量（**全部派生式**；本表是这些物理量的**唯一**定义处）

```python
# ── RPE 舞台几何 ──（外部权威源常量，A 级：prpr parse/rpe.rs:22-23 RPE_WIDTH / RPE_HEIGHT）
# 这两个数只允许在本文件出现一次；其余一切（675 / Δx / 场边界）必须由它们派生。
RPE_STAGE_WIDTH: float = 1350.0                       # prpr RPE_WIDTH，全域 = 舞台宽
RPE_STAGE_HEIGHT: float = 900.0                       # prpr RPE_HEIGHT
RPE_STAGE_HALF_WIDTH: float = RPE_STAGE_WIDTH / 2.0   # 可见范围 = [-此值, +此值]（可见边界，非合法值域）
RPE_STAGE_HALF_HEIGHT: float = RPE_STAGE_HEIGHT / 2.0

# ── x 网格（RFC-0029 §3.1 裁定 Δx = RPE_STAGE_WIDTH / N，默认 N = 128；须补 N 消融）──
RPE_X_GRID_BINS: int = 128
RPE_X_GRID_DX: float = RPE_STAGE_WIDTH / RPE_X_GRID_BINS        # 禁止在别处写 10.546875
RPE_X_GRID_MIN: float = -RPE_STAGE_HALF_WIDTH
RPE_X_GRID_MAX: float = +RPE_STAGE_HALF_WIDTH
RPE_X_GRID_BIN_SWEEP: tuple[int, ...] = (64, 128, 256, 512)     # 必做消融（红线 7）

# ── 速度单位 ──（A 级：prpr core.rs SPEED_RATIO = 10/45/HEIGHT_RATIO；出处存疑 D3）
RPE_HEIGHT_RATIO: float = 0.83175                               # prpr HEIGHT_RATIO（D3 未查证）
RPE_SPEED_RATIO: float = 10.0 / 45.0 / RPE_HEIGHT_RATIO         # 世界系 y 单位 / 秒 / 速度单位
RPE_SPEED_UNIT_RPE_Y_PER_SEC: float = RPE_STAGE_HALF_HEIGHT * RPE_SPEED_RATIO   # 禁写 120.23

# ── 事件层 ──
RPE_MAX_EVENT_LAYERS: int = 5          # 实测长度 1..5
RPE_NORMAL_EVENT_LAYERS: int = 4       # RFC-0029 §8.1 文档口径（非归一化依据，见偏离 1）
RPE_NORMAL_TRACKS: tuple[str, ...] = (
    "moveXEvents", "moveYEvents", "rotateEvents", "alphaEvents", "speedEvents",
)
RPE_EXTENDED_TRACKS: tuple[str, ...] = (
    "scaleXEvents", "scaleYEvents", "colorEvents", "textEvents", "gifEvents", "inclineEvents",
)

# ── 时间 ──
# RPEJSON 时间 = beat 三元组（i + n/d 拍），由根级 BPMList 分段积分 → 秒（A 级 prpr Triple）。
# 秒是本项目所有内部时间量纲；禁止在帧索引域评估（RFC-0029 §7-5）。

# ── 音频侧（**引用** beatmorph/core/contracts/tensors.py，不重复定义）──
# MERT_SAMPLE_RATE_HZ = 24000；MERT_CONV_STRIDE_PRODUCT = 320
# MERT_FRAME_RATE_HZ = MERT_SAMPLE_RATE_HZ / MERT_CONV_STRIDE_PRODUCT   # = 75 Hz（派生量）
# MERT_DEFAULT_FEAT_DIM = 1024
```

> **红线 7 的落地方式**：本文件内允许出现的外部数字只有 `RPE_STAGE_WIDTH` / `RPE_STAGE_HEIGHT` / `RPE_HEIGHT_RATIO` 三个**来源常量**与纯结构参数（层数、桶数 N、时间步长分母）。半宽、桶宽、速度换算、场边界、时间轴长度**一律写成派生式**，并由 §6-M4 的源码扫描测试守住。

### 3.2 枚举

```python
class Side(IntEnum):
    """判定线哪一侧下落。**不是布尔**（见 §2 偏离 3）。

    FRONT = +1：`above == 1`（A 级语义：1 为正面，其余值均为背面）
    BACK  = -1：**其余任何值**（实测 0 与 2 都出现）
    两侧几何上是关于判定线局部 X 轴的 Y 镜像。
    """
    FRONT = +1
    BACK = -1

class NoteType(IntEnum):
    """RPEJSON note type（A 级源码确证：prpr parse/rpe.rs 1/2/3/4 = Click/Hold/Flick/Drag）。"""
    TAP = 1
    HOLD = 2
    FLICK = 3
    DRAG = 4
```

**唯一转换路径（禁止任何模块自行写 `above` 判断）**：

```python
def side_from_above(above: int) -> Side:
    """above == 1 → FRONT，其余值 → BACK。不得写成 bool(above) 或 above != 0。"""
    return Side.FRONT if above == 1 else Side.BACK

def side_index(side: Side) -> int:
    """场通道索引：FRONT → 0，BACK → 1（索引与枚举值**不同**，必须用本函数）。"""
    return 0 if side is Side.FRONT else 1

def note_type_from_rpe(raw: int) -> NoteType: ...       # 主路径：1/2/3/4 = Tap/Hold/Flick/Drag
def note_type_from_official(raw: int) -> NoteType: ...  # 官谱 JSON（C 级）：2/3/4 = Drag/Hold/Flick
```

> ⚠️ **两套映射必须由「内容嗅探出的格式」分派，不得由文件后缀分派**（[RFC-0029 §6](decisions/RFC-0029-phigros-continuous-chart-generation.md)、[phira-dataset-survey.md §5.3](knowledges/phira-dataset-survey.md) 陷阱 3）：同一个数字 2 在 RPE 是 Hold、在官谱是 Drag，混用会**静默全员错位**。

### 3.3 事件与层

```python
class Beat(BaseModel):
    """RPE beat 三元组：beats = i + n / d（A 级 prpr Triple(i32,u32,u32)）。"""
    i: int; n: int = 0; d: int = 1
    def to_beats(self) -> float: ...

class EventKeyframe(BaseModel):
    start_time: Beat
    end_time: Beat
    start: float | str | tuple[int, int, int]     # 数值 / 文本(textEvents) / RGB(colorEvents)
    end:   float | str | tuple[int, int, int]
    easing_type: int = 1                          # 1..29（非 int → 1；越界 → 末端值）
    bezier: bool = False
    bezier_points: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    easing_left: float = 0.0                      # ∈ [0,1]
    easing_right: float = 1.0                     # ∈ [0,1]
    link_group: int = 0

class EventLayer(BaseModel):
    """一个普通事件层级：5 条轨。缺失字段 / null 层 / 缺 eventLayers 三者归一为空列表。"""
    layer_index: int
    move_x: list[EventKeyframe] = []      # 已按 startTime 排序并**补洞**（见 §4）
    move_y: list[EventKeyframe] = []
    rotate: list[EventKeyframe] = []
    alpha:  list[EventKeyframe] = []
    speed:  list[EventKeyframe] = []      # 仅 startTime/endTime/start/end/linkgroup 五字段

class ExtendedLayer(BaseModel):
    """第 5 层（`extended` 字段），独立于 `event_layers`。"""
    scale_x: list[EventKeyframe] = []
    scale_y: list[EventKeyframe] = []
    color:   list[EventKeyframe] = []
    text:    list[EventKeyframe] = []
    gif:     list[EventKeyframe] = []
    incline: list[EventKeyframe] = []
```

### 3.4 JudgeLine

```python
class JudgeLine(BaseModel):
    line_id: int                                  # = judgeLineList 索引；**不是对称标签**（不可互换）
    group: int = 0
    name: str = "Untitled"
    texture: str = "line.png"
    anchor: tuple[float, float] = (0.5, 0.5)      # 只影响贴图绘制，不改变判定线位置
    event_layers: list[EventLayer]                # 1 <= len <= RPE_MAX_EVENT_LAYERS（原样保留，见偏离 1）
    extended: ExtendedLayer = ExtendedLayer()
    father: int = -1                              # -1 = 无父线；实测 26% 的谱面含嵌套线
    rotate_with_father: bool = True
    is_cover: int = 1                             # **1 = 遮罩，其余 = 不遮罩**（同 above 类陷阱，禁 bool()）
    z_order: int = 0
    attach_ui: str | None = None
    bpm_factor: float = 1.0                       # 存储但**不参与换算**：prpr 标为 TODO（存疑 D4）
    num_of_notes_raw: int = 0                     # 冗余字段，仅供解析期交叉校验，不参与语义

    def pose_at(self, t_s: float, chart: "PhigrosChart") -> "LinePose": ...
    def local_to_stage(self, t_s: float, chart: "PhigrosChart") -> "Transform": ...

class LinePose(BaseModel):
    """判定线在 t（秒）时刻的舞台系位姿（**跨层求和**后的结果）。"""
    x: float; y: float                            # RPE 舞台系坐标
    rotate_deg: float                             # 度（正方向见存疑 D1）
    alpha: float                                  # Σ_layers(alphaEvents) 归一化后的不透明度
    scale_x: float = 1.0; scale_y: float = 1.0
```

**求值语义（契约级，所有模块必须一致）**：

- `moveX` / `moveY` / `rotate` / `alpha` 的最终值 = **各层事件值之和**（不是取最上层）；
- 坐标 = 自身跨层求和 + **父线坐标的递归叠加**；父线旋转影响子线**锚点位置**，默认不叠加进子线自身角度；
- 事件间隙必须**补洞**（相邻事件之间插入保持前一事件终值的常量事件，末尾追加到足够大的时间），否则间隙内取值无定义；
- 缓动求值与切割规则按格式文档逐条实现（1..29 全表 + 贝塞尔 + `easingLeft/Right`），**不得自造公式**；
- 判定线身份由「`judgeLineList` 索引 + 事件轨条件」共同确定，**排序不变性不成立**（RFC-0029 §2.4-3）。

### 3.5 PhigrosNote

```python
class PhigrosNote(BaseModel):
    line_id: int                 # 所属判定线索引
    t: float                     # **秒**（判定时刻 = startTime beat → 秒）；禁止存 beat 或帧索引
    position_x: float            # RPE 舞台系 x 坐标单位；可见范围 [-RPE_STAGE_HALF_WIDTH, +RPE_STAGE_HALF_WIDTH]
    side: Side                   # FRONT / BACK（由 above 唯一映射，见 §3.2）
    type: NoteType               # tap / drag / hold / flick
    hold_time: float = 0.0       # 秒；仅 HOLD 有意义（= endTime - startTime），其余恒为 0
    speed: float = 1.0           # 音符流速倍率（实际速度 = 判定线速度 × 本值）
    is_fake: bool = False        # isFake == 1 → 假音符（1 为假、其余为真 → 只认 == 1）
    # ── 以下为**无损往返**所需的原始字段（不参与建模，导出时必须原样写回）──
    above_raw: int = 1           # 实测 ∈ {0,1,2}；0 与 2 都必须保留原值
    type_raw: int = 1
    is_fake_raw: int = 0
    y_offset: float = 0.0        # RPE-y 单位，实际偏移 = y_offset × speed（A 级源码）
    visible_time: float = 999999.0   # 秒（默认值三方一致裁定，见知识文档 §6.4）
    alpha: int = 255             # 0..255
    size: float = 1.0
```

> **`hold_time` 的契约含义**：非 HOLD 的 `endTime == startTime`，因此 `hold_time == 0` 是**格式事实而非默认值**；Hold 期间判定线不得发生速度变化（格式硬合法性约束），该约束属后处理（plan 05），但契约在此声明它是**合法性红线**。
> **`position_x` 的越界处理**：解析与 IR **只统计不钳位**（I3）；越界样本进隔离报告，正是发现「我方解析单位错」的哨兵（[phira-dataset-survey.md §9.2](knowledges/phira-dataset-survey.md) [5]）。

### 3.6 PhigrosChart

```python
class PhigrosChart(BaseModel):
    version: str = "phigros-ir-1"                 # IR schema 版本，破坏性变更升号
    mode: GameMode = GameMode.PHIGROS             # RFC-0029 §4.3：GameMode 增 PHIGROS
    lines: list[JudgeLine]                        # K >= 1（v1 不限制 K）
    notes: list[PhigrosNote]                      # N >= 0
    bpm_points: list[BpmPoint]                    # >= 1 个，按 time_beats 升序
    meta: ChartMeta                               # META.*（offset 毫秒 / RPEVersion / chartTime 秒）
    source: ChartSource                           # chart_id / 包哈希 / 格式 / 内容嗅探依据

    def duration_s(self) -> float: ...                      # 以**音频时长**为准（见下）
    def sorted_notes(self) -> list[PhigrosNote]: ...        # (t, line_id, position_x) 确定性排序
    def notes_per_line(self) -> list[int]: ...              # 长度 K 的每线 note 数
    def out_of_visible_range(self) -> OutOfRangeStats: ...  # 只统计，无钳位
```

**时间轴口径（beat-aligned，Q15 决议 2026-08-05）**：场的时间轴是**拍相对坐标 τ**，基本格 `d_tau = 1 / SUBDIVISIONS_PER_BEAT`（= 1/48 拍，**派生式**）；`t_bins` 由**谱面的 `BPMList` 与时长共同决定**——即把 `[0, duration_s]` 在 BPM 映射下换算为 τ 区间后按 `d_tau` 分格。

> **⚠️ 接缝裁定（秒 ↔ τ 的唯一入口）**：换算**由 `beatmorph/field/` 唯一实现**，Jacobian `J(τ) = dt/dτ` 由 `BPMList` 精确给出。**本契约只声明网格元数据与不变量，不实现换算**——否则 beat-aligned 的数学就会外溢到所有下游（CLAUDE.md 红线 7）。契约因此携带 `bpm_points` 作为**数据**（换算依据），而换算**函数**留在 `field/`。

谱面事件超出定义域的部分记入 `out_of_window`（与 `positionX` 越界同一处理），不截断、不拉伸。

**`ChartMeta` 必含字段**：`offset_ms`（RPE 为**毫秒**，与官谱的秒制不同，见 Q11）、`rpe_version`（100~160，但**不能**据此判定能力集）、`chart_time_s`、`level_text`（自由文本，**不得** regex 解析成难度）、`difficulty`（`info.yml` 的 f32 定数，比较前 round 到 0.1）。

### 3.7 强度场张量契约

```python
@dataclass(frozen=True)
class ChartFieldSpec:
    """强度场网格规格（**唯一**的网格元数据来源，缺一不可断言）。"""
    k: int                 # 判定线条数（运行期可变，**不得**写进任何输出层维度）
    t_bins: int                      # T = τ 轴格数（由 BPMList + 时长派生，见上「时间轴口径」）
    d_tau: float = TAU_GRID_DT       # = 1 / SUBDIVISIONS_PER_BEAT（派生，禁止写 1/48）
    bpm_points: tuple[BpmPoint, ...] = ()   # τ ↔ 秒 换算的**唯一依据**；换算函数本身由 field/ 实现
    x_bins: int = RPE_X_GRID_BINS              # X = 128（默认；消融见 RPE_X_GRID_BIN_SWEEP）
    dx: float = RPE_X_GRID_DX                  # = RPE_STAGE_WIDTH / x_bins
    x_min: float = RPE_X_GRID_MIN              # = -RPE_STAGE_HALF_WIDTH
    x_max: float = RPE_X_GRID_MAX              # = +RPE_STAGE_HALF_WIDTH
    sides: int = len(Side)                     # S = 2（FRONT → 0 / BACK → 1）
    channels: int = len(NoteType) + 1          # C = 5：tap / drag / hold / flick / hold-end

    def assert_grid(self) -> None:
        """dx * x_bins == RPE_STAGE_WIDTH；d_tau * SUBDIVISIONS_PER_BEAT == 1；bpm_points 非空且按 time 升序。失败即抛，不得降级为日志。"""
```

| 张量 | einops 形状 | dtype | 语义 |
|------|------------|-------|------|
| `ChartField.lam`（模型侧） | `(batch, k, t, x, s, c)` | float32 | 非齐次强度 λ ≥ 0（softplus / exp 参数化，禁止裸线性输出） |
| `ChartField.mask` | `(batch, k, t, x, s, c)` | bool | **1 = 被遮盖（待补全）**，0 = 已观测；必须显式存在 |
| `ChartField.line_mask` | `(batch, k)` | bool | 该 batch 内真实存在的线；`k >= spec.k` 为 padding（K 随谱变化） |
| `ChartField.time_mask` | `(batch, t)` | bool | 有效时间帧（音频 padding 区为 0） |
| `ChartTargetField.counts` | `(batch, k, t, x, s, c)` | float32 | **桶内事件计数**（把桶计数当泊松观测，不是 0/1 热图） |
| `ChartTargetField.out_of_window` | `(batch,)` | int64 | 落在定义域外、已从事件项显式排除的事件数（见偏离 2） |

**测度声明（∫λ 的契约部分）**：

```
∫λ ≈ Σ_{k,t,x,s,c} λ[k,t,x,s,c] · J(τ_t) · d_tau · dx · 1 · 1
```

- 侧别与类型通道是**离散类别轴**，各自贡献 1（无 Δ 因子）；x 是连续轴的离散化，贡献 Δx；**时间是 beat-aligned 轴，贡献 Jacobian `J(τ_t)` × `d_tau`**；
- **`J(τ_t) = dt/dτ` 由谱面 `BPMList` 在 `field/` 内求值（契约内不实现）**；常速曲下 `J ≡ 60 / (bpm · SUBDIVISIONS_PER_BEAT)` 为常数，变速曲下逐段取值；
- 生成阶段：`|x| > RPE_STAGE_HALF_WIDTH` 区域的 λ 由参数化保证为 0；解析阶段**不作任何钳位**；
- 一个桶内出现多个事件时，**似然必须用桶计数形式**（Poisson count），不得对同一桶产生多个 `log λ` 项（[chart-generation-literature.md §4.4-4](knowledges/chart-generation-literature.md)）；
- **两条积分路径**（网格数值积分 / 累积强度 Λ(t) 参数化，Omi et al. 2019）必须给出同一 NLL（给定额差），这是 `field/` 的一致性门禁；契约只固定**测度与形状**。

**G3 常数基线（可立即实现的契约级检查）**：`λ ≡ N / |Ω|`，其中 `|Ω| = k * t_bins * x_bins * sides * channels`；契约断言 `λ ≡ 0` 时泊松 NLL = +∞（不得写成 `λ ≡ 0` 之外还留一条「全 0 是合法解」的路径）。

**索引映射（唯一路径，禁止各模块自行 floor）**：

```python
def x_bin_index(x: float, spec: ChartFieldSpec) -> int | None:
    """x → 桶索引。闭区间右端 x == x_max 归入最后一桶；域外返回 None（**不钳位**）。"""
```

### 3.8 不变量断言表（= 契约测试清单）

| # | 不变量 | 依据 |
|---|--------|------|
| I1 | `dx * x_bins == RPE_STAGE_WIDTH`；`d_tau * SUBDIVISIONS_PER_BEAT == 1` | 红线 7 / RFC §7-1 |
| I2 | `x_min == -RPE_STAGE_HALF_WIDTH`；`x_max == +RPE_STAGE_HALF_WIDTH` | BasePlan §3.2.4 |
| I3 | 越界 `position_x` **只统计不抛错、不钳位**，写入 `out_of_visible_range()` | BasePlan §3.2.4 ⚠️ |
| I4 | `side_from_above(1) is FRONT`；`side_from_above(0) is BACK`；`side_from_above(2) is BACK` | RFC §8.1 Q3 |
| I5 | `bool(Side.FRONT) and bool(Side.BACK)` 均为真（禁 truthiness 分支） | 本 plan 偏离 3 |
| I6 | RPE 分派下 `type_raw ∈ {1,2,3,4}`；`note_type_from_rpe(2) is HOLD` 且 `note_type_from_official(2) is DRAG` | RFC §6 / 陷阱 3 |
| I7 | `1 <= len(event_layers) <= RPE_MAX_EVENT_LAYERS`（不截断不补齐） | 实测 + 偏离 1 |
| I8 | `father` 链无环且索引 ∈ `[0, K)`；成环 → 质检拒绝 | 实测 26% 嵌套 |
| I9 | `k >= 1`；`sides == len(Side)`；`channels == len(NoteType) + 1` | RFC §3.1 |
| I10 | 往返：`PhigrosChart.model_validate_json(c.model_dump_json())` 等价，且 `above_raw / type_raw / is_fake_raw` 不变 | 契约稳定性 |
| I11 | 全流程时间量纲 = 秒；**τ 索引只允许**出现在 `ChartFieldSpec` 内，**秒 ↔ τ 换算只在 `field/` 内实现**（红线 7 / RFC §7-8） | RFC §7-5 / §7-8 |
| I13 | `bpm_points` 非空且按 `time` 升序；**τ → 秒 → τ 往返在 float64 下相对误差 ≤ 1e-9**（须覆盖**多 BPM 段**谱面） | RFC §3.1 ⚠️（beat-aligned 的"下一个 25 Hz"防线） |
| I12 | `phigros.py` / `field.py` 在无 `torch`、无权重、无 GPU 的环境下可导入 | CLAUDE.md §4（契约级测试不得依赖权重或 GPU）|

## 4. 内部设计

- **文件划分**：`phigros.py`（领域对象 + 枚举 + 唯一转换函数）、`field.py`（`ChartFieldSpec` / `ChartField` / `ChartTargetField` + 索引映射）。张量类用 `@dataclass(frozen=True)`（只描述形状，不持有数据）；领域对象用 `pydantic.BaseModel`（校验 + 序列化）。`__init__.py` 集中导出。
- **无损保留策略**：所有「原始值 → 语义值」的映射（`above` / `type` / `isFake`）**双写**：语义字段供建模，`*_raw` 供导出往返。既不把格式陷阱带进模型，也不在 IR 上制造不可逆的信息损失。
- **K 可变**：`lines` 是长度 K 的列表，`ChartFieldSpec.k` 是运行期值。**任何输出层的张量维度不得出现 K**（RFC-0029 §2.4-2：共享权重 + 每条线一个 line embedding 作 query）。
- **三态掩码**：`mask`（遮盖）/ `line_mask`（线 padding）/ `time_mask`（时间 padding）语义互不重叠，避免把「无 note」「被遮盖」「不存在」混为一谈（RFC-0029 §3.3-1）。
- **越界与超时长的处理路径统一**：解析 → 只统计；目标构建 → 计数 + 从事件项显式排除；生成 → 域外 λ 归零。三段职责写在同一张表（§3.8 I3 + §3.7），避免各模块各自发明。
- **不引入 torch 依赖**：`field.py` 只描述形状与测度；张量类型标注放在 `TYPE_CHECKING` 下的前向引用，保证契约层可在最小环境导入与测试。
- **日志**：`from beatmorph.core.logging import get_logger`，禁止裸 `print`。

## 5. 依赖关系

- **上游**：无（最底层；只依赖格式与单位知识文档的 A 级事实）。
- **下游**：plan 02（解析产出本契约）、plan 03（`beatmorph/field/`：消费 `ChartFieldSpec` 与测度）、plan 04（`beatmorph/generation/`：消费 λ 契约与三态掩码）、plan 05（`beatmorph/decoder/`：消费离散化规则与 `*_raw` 做导出）、plan 06（`beatmorph/eval/`）。
- **外部库**：`pydantic>=2.5`（唯一硬依赖，刻意保持最小）；`torch` **不是**本层的运行期依赖。
- **跨 plan 共享符号**：`side_from_above` / `side_index` / `x_bin_index` / `note_type_from_rpe` / `note_type_from_official` / `ChartFieldSpec` / `ChartField` / `ChartTargetField` 是契约边界，plan 01/02 与 field / generation 的 plan 必须**按名引用**，不得本地复制。

## 6. 里程碑与验收标准

| 里程碑 | 验收（可量化、可在默认 CI 跑） |
|--------|------------------------------|
| **M1 契约冻结 v1** | `from beatmorph.core.contracts import PhigrosChart, JudgeLine, PhigrosNote, ChartField, ChartTargetField, ChartFieldSpec` 可导入；`tests/unit/core/test_phigros_contracts.py` 全绿 |
| **M2 类型检查** | `uv run mypy beatmorph/core` strict 零错误 |
| **M3 无损往返** | `model_validate_json(model_dump_json())` 等价；`above_raw ∈ {0,1,2}`、`type_raw`、`is_fake_raw` 往返不变（I10） |
| **M4 单位派生断言（G4 落点）** | I1/I2 全部断言通过；**源码扫描测试**：`beatmorph/` 下除 `core/contracts/` 外不得出现 675 / 10.546875 / 120.23 / 帧率 75 作为**数值字面量**；该测试在**默认 CI**（`-m "not slow and not gpu"`）内运行 |
| **M5 side 语义** | I4/I5 全绿：0/1/2 → BACK/FRONT/BACK；两侧均为真值；`side_index` 映射为 0/1 且与枚举值不同 |
| **M6 type 双格式分派** | I6 全绿：同一数字 2 在 RPE / 官谱分派下分别为 HOLD / DRAG；分派函数签名**不接受**文件后缀/文件名参数（签名级约束） |
| **M7 场契约** | 形状 `(B,K,T,X,S,C)` 断言；`channels == len(NoteType) + 1`；`x_bin_index(+x_max) == x_bins - 1`；域外返回 None 且 `out_of_window` 递增（**无钳位**）；G3 基线 `λ ≡ N/|Ω|` 有限、`λ ≡ 0` → NLL = +∞（断言） |
| **M8 最小闭环（不依赖解析器）** | 用**手写的最小 IR 夹具**（Python dict，≤ 32 KB，不含音频/曲绘）构造 `PhigrosChart` → `ChartFieldSpec` + `ChartTargetField`；`Σ counts == 窗口内事件数`；`Σ counts · dt · dx == 同一数` |

> **G1–G4 门禁义务说明**：本 plan **不引入训练目标或损失**，故无 G1–G4 全绿义务；但它是 **G4（契约断言）的落点**——帧率、`Δx`、形状、量纲四项派生断言在本 plan 的 M4/M7 内以**默认 CI 契约测试**形式落地（[RFC-0029 §7](decisions/RFC-0029-phigros-continuous-chart-generation.md) 硬约束 1/4/6，红线 7）。泊松 NLL 目标实现与掩码补全主干的 G1–G4 全绿义务由 `field/` 与 `generation/` 的 plan 承担，且**必须在扩大数据规模之前完成**（[BasePlan §9](BasePlan.md)）。

## 7. 风险与缓解

| 风险 | 编号 | 缓解 |
|------|------|------|
| 多线 K 长尾 / note 极度集中被契约固化成均匀分类 | **R-5** | 契约只给「K 条线共享同一 x 网格 + 每线一个索引」的形状，**不含任何 line_id softmax / 均匀先验**；分配由 K 个强度场竞争决定 |
| 物理常量 / 单位漂移（25 Hz 类） | **R-7** | 本 plan 是红线 7 的唯一落点：全部派生式 + I1/I2/I11 + M4 源码扫描测试进默认 CI |
| 稀疏目标塌陷（「loss 不降」的同构形态） | **R-4** | 契约固定 **桶计数**目标语义与 `λ ≡ N/|Ω|` 的 G3 基线；禁止 0/1 热图语义进入 `counts` |
| 数据合规阻塞（夹具来源同属 Phira 内容） | **R-2** | 本 plan 仅入库**微缩**、无音频、无曲绘的夹具；夹具能否随仓库分发随 [RFC-0029 §8.3 Q11b](decisions/RFC-0029-phigros-continuous-chart-generation.md) 裁定（§9-2） |
| 跨线几何冲突不可玩 | **R-8** | 契约暴露 `pose_at` / `local_to_stage` 供后处理做跨线几何检查；**契约层不做任何钳位**（红线 3） |
| 契约早冻结导致返工 | 派生 | `version` 字段 + RFC 流程；新增跨模块类型先开 RFC（CLAUDE.md 红线 2） |

## 8. 测试策略

- **单元（默认 CI，无权重、无 GPU）**：
  - `tests/unit/core/test_phigros_contracts.py` — 边界值（`position_x = ±RPE_STAGE_HALF_WIDTH` 与略越界）、非法值（`type_raw = 0/5`、`father` 成环、`event_layers` 为空）、确定性排序、`*_raw` 往返、`Side` 真值性、`side_index` 映射；
  - `tests/unit/core/test_field_spec.py` — I1/I2/I9 与 M7 的全部断言、`x_bin_index` 边界（`x_min` / `x_max` / 域外）、`Σ counts` 守恒、G3 基线的有限性与 `λ ≡ 0` 的 +∞。
- **契约即测试**：pydantic 约束（枚举域、`ge/le`）本身是运行期断言；`assert_grid()` 是场的运行期门禁。
- **源码扫描测试**（M4）：扫 `beatmorph/**/*.py` 的数值字面量黑名单，只允许 `core/contracts/` 内出现来源常量。
- **禁止事项（RFC-0029 §7-6 / AGENTS.md §3.3）**：测试中的 mock / fixture **不得固化物理常量**——凡 mock 需要帧数或桶数，必须引用契约常量（`MERT_FRAME_RATE_HZ` / `RPE_X_GRID_DX`），不得写死数字；否则会复现「mock 把错误常量洗成绿灯」的历史事故。
- **集成（不依赖权重）**：`tests/integration/test_contract_to_field.py` — 手写最小 IR → 场规格 → 目标场，仅做形状 / 量纲 / 守恒断言。
- **e2e**：本 plan 无 e2e（不产生产物）；与 plan 02 的联合 e2e 见 plan 02 §8。

## 9. 开放问题

1. **`eventLayers` 的 5 层语义**（4 普通 + `extended` vs 5 普通层）：RFC-0029 §8.1 与 phigros-format.md Q18 / 调研 Q-8 互相矛盾。本 plan 取「原样保留长度 + `extended` 独立」，**不裁定**其语义；若最终需要按层语义分支（例如只有前 4 层参与求和），须开 RFC 并同步 plan 02。
2. **夹具的合规性**：微缩夹具改编自 Phira 用户上传内容，其随仓库分发是否在裁定范围内，须由 [RFC-0029 §8.3 Q11b](decisions/RFC-0029-phigros-continuous-chart-generation.md) 的裁定覆盖。
3. **越界事件的训练处理（偏离 2）**：本 plan 定为「统计 + 从事件项排除」，属建模选择，需 RFC 追认（备选：把场定义域扩到 `|x| > RPE_STAGE_HALF_WIDTH`，但那与「域外不可见、不可玩」冲突）。
4. **积分记号不一致**：BasePlan §2/§3.4 写 `∫∫∫ λ_k(t,x,s)`，RFC-0029 §3.1 写 `λ_k(t,x,s,c)`。契约取五轴全积分（偏离 4），请 RFC-0029 下次修订时消除此记号冲突。
5. **`c` 轴是否足以表达 Hold 时长**：`hold_time` 是连续量，当前只进「hold + hold-end」两个通道（RFC-0029 §3.1）。是否需要额外的时长通道取决于 generation 侧能否只在标记层回归，须与 field / generation 的 plan 联合裁定。
6. **`RPE_HEIGHT_RATIO` 的出处（存疑 D3）**：速度单位换算依赖 prpr 常量 `0.83175`（若取 5/6 则恰好 120.0）。契约按 A 级源码取 `0.83175`，但该常量来源未查证 → 若将来换成 5/6，速度单位变 0.19%，须消融确认其对 `yOffset` 相关指标无实质影响。
7. **`bpm_factor` 是否参与换算（存疑 D4）**：prpr 标为 TODO（未实现），Phira 文档两页示例代码矛盾（乘 vs 除）。契约**存储但不使用**；plan 02 遇到 `bpm_factor != 1.0` 的谱面必须计入隔离报告。
8. **旋转正方向的屏幕含义（存疑 D1）**：影响 `local_to_stage` 的符号；目前只影响可视化与跨线几何检查，不影响 note 建模（事件轨是条件输入）。若后续做跨线几何合法性判定，须先裁定。
9. **`RPE_X_GRID_BINS` 的最终取值**：默认 128 由 RFC 裁定，但「共格碰撞率」（同线 + 同刻 + 同侧的最小 |ΔpositionX|）尚无数据。契约提供 `RPE_X_GRID_BIN_SWEEP` 与 `assert_grid()`；**N 的最终值待 plan 02 的统计出来后再定**（红线 7：128 不能成为新魔数）。
10. ~~**`|Ω|`（场体积）定义需与其它 plan 互认**~~ ✅ **已裁决（RFC-0029 §8.4 R-a，2026-08-05）**：统一为**全 K 条线、格元总数** `|Ω| = k * t_bins * x_bins * sides * channels`（无量纲，λ 单位 = 每格元事件数），**空线计入**。契约层定义以本 plan §3.7 为准，plan 03/06/07 **只引用同一符号、不得各自定义**；跨谱汇总报 per-chart NLL 与 NLL/N 两栏。改口径须同步本 plan 并走 RFC（红线 2）。