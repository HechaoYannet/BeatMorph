# Phigros / Phira 单位与几何（实现参考）

**本文件用途**：钉死 [RFC-0029](../decisions/RFC-0029-phigros-continuous-chart-generation.md) §8.2 **Q8**（`positionX` 的物理单位与桶宽）与 **Q9**（3:2 舞台 vs `aspectRatio`），并给出 `positionX` / `lineLength` / `speed` / `yOffset` / `visibleTime` 的**可复算**单位定义。所有事实均带来源链接与可信度分级；凡属推断，均显式标注 **【推断】** 并给出依据；查不到的一律写「**未查证**」，不做猜测。

**配套文件**：[phigros-format.md](phigros-format.md)（RPEJSON 逐字段 / 坐标几何 / 19 条存疑清单）。**本文件只做单位与几何的深挖与裁定，字段级 schema 不重复。** 本文件第 8 节给出对 phigros-format.md 的逐条修正建议（**未直接改动该文件**）。

---

## 1. 范围、方法与来源分级

### 1.1 方法：以**参考实现源码**为最高证据

格式文档（wiki）互相矛盾，**代码不会**。本次调研的判定优先级为：

1. **A 级 · 参考实现源码 / 官方文档**——能直接引用到「把 `positionX` 变成屏幕坐标的那几行」；
2. **B 级 · 社区 wiki**——用于交叉验证与补默认值；
3. **C 级 · 二手技术文档**——仅在没有更高级别证据时保留，且必须标注。

本次读到的两个参考实现：

| 实现 | 语言 | 性质 | 与目标格式的关系 |
| :-- | :-- | :-- | :-- |
| **prpr**（[TeamFlos/phira · prpr/](https://github.com/TeamFlos/phira/tree/main/prpr)） | Rust | **Phira 客户端本体使用的谱面解析 + 渲染库**（`phira/Cargo.toml` 中 `prpr = { path = "prpr" }`） | **直接解析 RPEJSON**，是「Phira 到底怎么解释这些数字」的**唯一权威** |
| **phichain**（[Ivan-1F/phichain](https://github.com/Ivan-1F/phichain)） | Rust | 第三方制谱工具链（编辑器 / 转谱器 / 渲染器） | 独立实现 RPEJSON 与官谱 JSON 的互转，用于**交叉验证** |

> ⚠️ **证据范围的诚实声明**：本环境无法把整个 phira 仓库拉到本地做全仓 grep（`codeload.github.com` 不可达），因此下文凡称「`lineLength` 的唯一用途」时，**限定为「本文实际读过的源文件中」**。已读文件清单见 §10.2。

### 1.2 来源分级

| 级别 | 来源 | 说明 |
| :-- | :-- | :-- |
| **A** | [prpr 源码](https://github.com/TeamFlos/phira/tree/main/prpr)、[Phira 文档](https://teamflos.github.io/phira-docs/)（TeamFlos 官方 mdBook）、[phichain 源码](https://github.com/Ivan-1F/phichain)、[Phichain 文档](https://phichain.rs/docs/) | 官方实现方自述或直接可读的实现代码 |
| **B** | [KillKPA / Phigros 自制谱 wiki·术语](https://killkpa.miraheze.org/wiki/%E6%9C%AF%E8%AF%AD)、[pgrfm.miraheze.org](https://pgrfm.miraheze.org/wiki/) | 社区维护，条目间偶有矛盾 |
| **C** | [Lchzh Docs · Phigros 谱面格式说明](https://docs.lchzh.top/learning/phigros/) | 二手；**官谱侧单位只有此来源给出** |

### 1.3 不可信数据声明

本文件所有网页内容仅作为**资料**读取。网页正文中不存在被本文件采纳的指令性内容；本文件作者未执行任何来自网页的操作指令。源码仅作为**事实证据**引用，未被运行。

---

## 2. 三套坐标系总览与转换公式

### 2.0 一张图

`@
[判定线局部系 L]  ──(该线在 t 时刻的 平移+旋转，跨层求和)──▶  [RPE 舞台系 S]
   原点=锚点, +X=判定线本身,                                 [-675,675]×[-450,450]
   +Y=锚点朝向                                        │
                                                      │ (x/675, y/(450·ar))
                                                      ▼
                                        [Phira 世界系 W]  x∈[-1,1], y∈[-1/ar, 1/ar]
                                                      │  (viewport 按 ar 加黑边)
                                                      ▼
                                        [屏幕像素系 P]  px_x=W_px·(0.5+x/1350)
                                                        px_y=H_px·(0.5−y/900)
`@

### 2.1 判定线局部系 L（RFC-0029 的建模系）

- 每条判定线有自己的平面直角坐标系：**锚点为原点、判定线本身为 X 轴、锚点面向的方向为 Y 轴正半轴**（[判定线](https://pgrfm.miraheze.org/wiki/%E5%88%A4%E5%AE%9A%E7%BA%BF)，**B**）。
- 判定线上的音符**恒平行于局部 X 轴**，下落方向**恒平行于局部 Y 轴**（同上，**B**）。
- **A 级实现证据（新）**：prpr 渲染「下侧」音符时，把**整条局部坐标系做 Y 轴镜像**——`Matrix::identity().append_nonuniform_scaling(&Vector::new(1.0, -1.0))` 包裹了全部 `!note.above` 的音符（[prpr/src/core/line.rs:412-429](https://github.com/TeamFlos/phira/blob/main/prpr/src/core/line.rs#L412-L429)，**A**）。
  → **结论（A 级确证）**：`above == 1`（正面）的音符从局部 **+Y** 侧接近判定线，其余值的音符从局部 **−Y** 侧接近，两侧是**关于判定线 X 轴的镜像**；`side` 是一个真正的 ±1 符号自由度，**不改变 `positionX`**。
  这也**直接解释了 B 级的 `yOffset` 语义**：「若该值为负数，对不同下落朝向的音符的影响都是向**下落朝向**偏移」（[音符](https://pgrfm.miraheze.org/wiki/%E9%9F%B3%E7%AC%A6)，**B**）——因为局部 Y 轴被镜像了。

### 2.2 RPE 舞台系 S

- 屏幕可见 x 范围 **−675 ~ 675**，y 范围 **−450 ~ 450**，坐标系锚点在**屏幕中心**（[事件](https://pgrfm.miraheze.org/wiki/%E4%BA%8B%E4%BB%B6)（**B**）、[Phira event](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/event.html)（**A**））。
- **A 级源码常量**：`RPE_WIDTH = 1350.0`、`RPE_HEIGHT = 900.0`（[prpr/src/parse/rpe.rs:22-23](https://github.com/TeamFlos/phira/blob/main/prpr/src/parse/rpe.rs#L22-L23)，**A**）；`CANVAS_WIDTH = 1350.0`、`CANVAS_HEIGHT = 900.0`（[phichain-chart/src/constants.rs](https://github.com/Ivan-1F/phichain/blob/master/phichain-chart/src/constants.rs)，**A**）。两个独立实现给出**同一个舞台**。
- 归一化映射（Phira 官方 Python 参考实现，**A**）：`sx = (x + 675)/1350`、`sy = 1 − (y + 450)/900`（[judgeLine](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/judgeLine.html)）。

### 2.3 Phira 世界系 W（渲染系，本次新钉死）

prpr 的绘制空间**不是**像素，也不是 `[-675,675]`，而是一个**各向同性（像素意义上）的归一化系**：

- 全屏四边形 = `Rect::new(-1., -1/aspect_ratio, 2., 2/aspect_ratio)`（视频背景，[prpr/src/core/video.rs:153-154](https://github.com/TeamFlos/phira/blob/main/prpr/src/core/video.rs#L153-L154)，**A**）；
- 屏幕范围被硬编码为 `(vw, vh) = (1.1, 1.)`（[prpr/src/core/line.rs:386-391](https://github.com/TeamFlos/phira/blob/main/prpr/src/core/line.rs#L386-L391)，**A**）；
- 相机 `zoom = vec2(1., -aspect_ratio)` + 按 `aspect_ratio` 计算的黑边 viewport（[prpr/src/core/resource.rs:505-509](https://github.com/TeamFlos/phira/blob/main/prpr/src/core/resource.rs#L505-L509)、[597-619](https://github.com/TeamFlos/phira/blob/main/prpr/src/core/resource.rs#L597-L619)，**A**）。

> **A 级结论**：世界系 `x ∈ [−1, 1]` ↔ 视口宽度，`y ∈ [−1/ar, 1/ar]` ↔ 视口高度，其中 `ar = 生效的 aspectRatio`。每世界单位在两个轴上的**像素长度相同**（各向同性）。

### 2.4 舞台系 → 世界系（**Q8 的核心公式**）

`@
// prpr/src/parse/rpe.rs:588 —— 音符
translation.x = note.position_x / (RPE_WIDTH / 2.)   // = positionX / 675
// prpr/src/core/object.rs:51-56 —— 所有 Object（音符与判定线共用）
tr.y /= res.aspect_ratio
`@
（[prpr/src/parse/rpe.rs:588](https://github.com/TeamFlos/phira/blob/main/prpr/src/parse/rpe.rs#L588)、[prpr/src/core/object.rs:51-56](https://github.com/TeamFlos/phira/blob/main/prpr/src/core/object.rs#L51-L56)，**A**）

判定线自身的移动事件用**同一个刻度**：

`@
// prpr/src/parse/rpe.rs:682-684
move_x_events  × (2. / RPE_WIDTH )   // = moveX / 675
move_y_events  × (2. / RPE_HEIGHT)   // = moveY / 450（随后同样再 / ar）
`@
（[prpr/src/parse/rpe.rs:682-684](https://github.com/TeamFlos/phira/blob/main/prpr/src/parse/rpe.rs#L682-L684)，**A**）

**于是得到换算链（全部由 A 级代码直接给出）**：

`@
x_world = positionX / 675                     ∈ [−1, 1]    (屏幕宽 = 2 世界单位)
y_world = RPE_y / (450 · ar)                  ∈ [−1/ar, 1/ar]  (屏幕高 = 2/ar 世界单位)
u = x_world/2 + 0.5 = 0.5 + positionX/1350     （0=左边缘, 1=右边缘）
v = 0.5 − RPE_y/900                            （0=上边缘, 1=下边缘）
`@

**`u`/`v` 与官方 `conrpepos` 逐字一致**（`(x+675)/1350 = 0.5 + x/1350`；`1 − (y+450)/900 = 0.5 − y/900`）——两条独立证据链（官方 Python 片段 **A** + prpr Rust 实现 **A**）在此**完全吻合**。

### 2.5 【推断】世界系 → 屏幕像素系

`@
px_x = W_px · (0.5 + positionX/1350)      W_px = 视口宽度(px)
px_y = H_px · (0.5 − RPE_y/900)           H_px = 视口高度(px)，px_y 自顶向下
`@
推断依据：§2.3 的三个 A 级常量（全屏四边形、`(vw,vh)`、viewport 黑边）+ 官方 `conrpepos`。**prpr 没有把这两个式子写成一行**，故标注为推断；但 `u/v` 的归一化式本身是 A 级确证的。

**像素尺度比（重要的物理含义）**：

`@
px / RPE-x 单位 = W_px / 1350
px / RPE-y 单位 = H_px / 900
比值 = (W_px/H_px) · (900/1350) = (2/3)·ar
`@
- `ar = 3/2`（RPE 原生比例）→ 比值 = 1，**各向同性**；
- `ar = 16/9`（Phira 默认）→ 比值 = **32/27 ≈ 1.185**：**RPE 的 x 单位比 y 单位大 18.5%**（等价说法：3:2 的舞台被横向拉伸 18.5% 后填满 16:9 视口）。

### 2.6 局部系 → 舞台系

`@
p_stage = Σ_layers( R(rotate(t)) · p_local + move(t) ) + p_stage(father, t)
`@
- 锚点位置 = `moveXEvents` / `moveYEvents` **跨层求和**（+ 父线位置，可嵌套）（[judgeLine](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/judgeLine.html)，**A**；phichain 导出实现同样把每层事件独立写出后由渲染端求和，[from_phichain.rs](https://github.com/Ivan-1F/phichain/blob/master/phichain-format/src/rpe/from_phichain.rs)，**A**）。
- **实测（A）**：prpr 的父线位置合成是 `parent_pos + Rotation2::new(parent_rot)·child_translation`（[prpr/src/core/line.rs:221-234](https://github.com/TeamFlos/phira/blob/main/prpr/src/core/line.rs#L221-L234)，**A**）——即**父线的旋转会影响子线锚点的位置（但不默认叠加进子线自身角度）**，与 wiki 表述一致（[判定线](https://pgrfm.miraheze.org/wiki/%E5%88%A4%E5%AE%9A%E7%BA%BF)，**B**）。
- 旋转单位是**度**（[事件](https://pgrfm.miraheze.org/wiki/%E4%BA%8B%E4%BB%B6)，**B**）。
- **【存疑】旋转正方向的屏幕含义**：KillKPA 写「正数表示顺时针」（[术语](https://killkpa.miraheze.org/wiki/%E6%9C%AF%E8%AF%AD)，**B**）；而 phichain 在导出 RPE 时**对旋转值取负**并注明 `// RPE rotation uses opposite sign convention`（[from_phichain.rs:96-99](https://github.com/Ivan-1F/phichain/blob/master/phichain-format/src/rpe/from_phichain.rs#L96-L99)，**A**）。**推断**：phichain 内部用数学惯例（逆时针为正），RPE 用屏幕惯例（顺时针为正），两者一致；但**没有任何来源把「RPE 正角 = 屏幕顺时针」写成形式化定义**，故降级为 B 级事实 + 存疑 **D1**。

---

## 3. `positionX` 的单位与值域 —— **Q8 的答案**

### 3.1 结论（可直接照写进契约）

| 问题 | 结论 | 级别 | 证据 |
| :-- | :-- | :-- | :-- |
| 量纲 | **RPE 舞台系的 x 坐标单位**（非像素、非归一化）；1 单位 = 舞台宽度的 1/1350 | **A** | `RPE_WIDTH=1350` + `position_x / (RPE_WIDTH/2)`（[rpe.rs:22](https://github.com/TeamFlos/phira/blob/main/prpr/src/parse/rpe.rs#L22)、[rpe.rs:588](https://github.com/TeamFlos/phira/blob/main/prpr/src/parse/rpe.rs#L588)） |
| 有效范围 | **语义可见范围 [−675, 675]**（全域 1350 = 舞台宽度 = 屏幕宽度） | **A** | [事件](https://pgrfm.miraheze.org/wiki/%E4%BA%8B%E4%BB%B6)（**B**）+ [术语](https://killkpa.miraheze.org/wiki/%E6%9C%AF%E8%AF%AD)（**B**）+ 源码除 675（**A**） |
| 与 `moveX` 同刻度？ | **是**，两者都除以 675 进世界系 | **A** | [rpe.rs:588](https://github.com/TeamFlos/phira/blob/main/prpr/src/parse/rpe.rs#L588) vs [rpe.rs:683](https://github.com/TeamFlos/phira/blob/main/prpr/src/parse/rpe.rs#L683) |
| 是否受 `lineLength` 缩放？ | **否。完全无关** | **A** | §4 |
| 是否可超出 ±675？ | **格式与参考实现均无钳位**：解析为裸 `f32`，越界值只会在屏幕外 | **A（实现侧）** | [rpe.rs:151](https://github.com/TeamFlos/phira/blob/main/prpr/src/parse/rpe.rs#L151)（`position_x: f32`，无校验） |
| RPE 编辑器 UI 是否限制输入范围 | **未查证** | — | 未找到 RPE 官方 schema 或输入校验代码 |
| 是否只取整数值 | **未查证** | — | 类型是 `number`/`f32`；真实谱面分布未统计 |

### 3.2 逐条回答任务问题 1

1. **它就是 RPE 舞台像素坐标吗？** ——「像素」这个词要限定：它**不是屏幕像素**，而是**舞台坐标**（定义域恒为 1350×900 的一个虚拟画布）。在 `aspectRatio = 3:2` 且视口铺满屏幕时，1 单位 = 1 屏幕像素；在 `16:9` 下 1 单位 = W_px/1350 像素（x 向）。**A 级**（§2.4/§2.5）。
2. **是否可超出 ±675？** —— 无任何钳位或校验；±675 是「可见边界」而非「合法值域」。**A 级**（实现侧）。→ 数据流水线应**统计**越界音符占比，**不要钳位**（钳位会改变落点分布，触碰 CLAUDE.md 红线 3）。
3. **是否受 `lineLength` 缩放？** —— **否**。见 §4。

---

## 4. `info.yml.lineLength` 的单位与作用

### 4.1 A 级实现证据（那几行代码）

`@rust
// prpr/src/core/line.rs:247-253
res.apply_model(|res| match &self.kind {
    JudgeLineKind::Normal => {
        let mut color = color.unwrap_or(res.judge_line_color);
        color.a *= alpha.max(0.0);
        let len = res.info.line_length;
        draw_line(-len, 0., len, 0., if line_scaled { 0.0076 } else { 0.01 }, color);
    }
`@
（[prpr/src/core/line.rs:247-253](https://github.com/TeamFlos/phira/blob/main/prpr/src/core/line.rs#L247-L253)，**A**）

这一段位于 `res.with_model(self.now_transform(...))` 之内，即**判定线自己的局部系**（§2.3 的世界系刻度：屏幕宽 = 2 世界单位）。

### 4.2 结论

| 结论 | 级别 | 证据 |
| :-- | :-- | :-- |
| `lineLength` 是判定线**从锚点向两侧延伸的半长**，单位 = **半屏宽（= 675 RPE-x 单位）**，在**判定线局部系**内度量 | **A** | 上面那段代码：`draw_line(-len, 0, +len, 0)`；世界系半屏宽 = 1 = 675 RPE 单位 |
| 默认值 **6.0** | **A** | [Phira 文档·谱面信息](https://teamflos.github.io/phira-docs/chart-standard/chartinfo.html)（**A**）+ `ChartInfo::default() { line_length: 6. }`（[prpr/src/info.rs](https://github.com/TeamFlos/phira/blob/main/prpr/src/info.rs)，**A**） |
| 默认 6.0 的几何含义：半长 = 6 × 675 = **4050 RPE-x 单位 = 3 倍屏幕宽度**（两侧各 3 倍屏幕宽；全长 6 倍屏幕宽 = 8100 单位） | **A（推导自 A 级代码）** | 同上 |
| **交叉验证**：Phichain 文档直言「**判定线的长度为 3 倍屏幕宽度**」 | **A/B** | [Phichain·判定线](https://phichain.rs/docs/line) |
| `lineLength` **不改变 `positionX` 的映射**：同一 `positionX` 在 `lineLength=6` 与 `=12` 下屏幕位置**完全相同** | **A** | 音符 x 只除以常量 675（[rpe.rs:588](https://github.com/TeamFlos/phira/blob/main/prpr/src/parse/rpe.rs#L588)）；`lineLength` 只出现在 `draw_line` 一处 |
| `lineLength` 是否影响判定（判定区宽度）？ | **未查证**；本文读过的文件中未见其用于判定 | — |

> **⚠️ 残留歧义（很小，但必须标出）**：Phichain 文档的「长度为 3 倍屏幕宽度」**未说明「长度」是半长还是全长**。按 prpr 代码算，**半长**恰为 3 倍屏幕宽（6 × 675 = 4050 = 3 × 1350），全长则为 6 倍。两实现数值自洽，仅措辞歧义。→ 存疑 **D2**。

### 4.3 逐条回答任务问题 2

- **6.0 是什么单位？** —— 半屏宽（675 RPE-x 单位）的倍数；即**在判定线局部系内、以「屏幕宽度的一半」为 1 的长度**。
- **是否改变 `positionX` 的映射？** —— **否**。`positionX` 与 `lineLength` 在实现里**没有任何耦合**。
- **设计动机【推断】**：默认线长远大于屏幕（±3 屏宽）是为了**判定线旋转后仍能横跨屏幕**；这与「任意方向下落」的观感需求一致。

---

## 5. 宽高比与舞台尺寸换算（Q9 / 原 Q17）

### 5.1 事实链（全部 A 级）

1. `info.yml.aspectRatio` 默认 **16/9**（[Phira 文档·谱面信息](https://teamflos.github.io/phira-docs/chart-standard/chartinfo.html)，**A**；`aspect_ratio: 16. / 9.`，[prpr/src/info.rs](https://github.com/TeamFlos/phira/blob/main/prpr/src/info.rs)，**A**）。
2. prpr 把世界渲染到**按 `ar` 计算的黑边 viewport**（[resource.rs:597-619](https://github.com/TeamFlos/phira/blob/main/prpr/src/core/resource.rs#L597-L619)，**A**）：
   - `forceAspectRatio: true` → `ar = info.aspectRatio`（强制黑边）；
   - 否则 → `ar = min(info.aspectRatio, 窗口宽/窗口高)`（**永不比谱面声明比例更宽**）。
3. 世界系的 x 半宽恒为 1、y 半宽恒为 `1/ar`（§2.3）。
4. `positionX/675` 与 `RPE_y/(450·ar)`（§2.4）。

### 5.2 换算公式（可直接实现）

`@
ar = forceAspectRatio ? info.aspectRatio : min(info.aspectRatio, win_w / win_h)
u  = 0.5 + positionX / 1350        # 归一化视口 x，0=左 1=右
v  = 0.5 - RPE_y      / 900        # 归一化视口 y，0=上 1=下
px_x = u * W_px ;  px_y = v * H_px ;  W_px / H_px = ar
`@

> **关键结论**：**RPE 舞台的 [−675,675]×[−450,450] 永远被完整映射到整个视口矩形上，与 `aspectRatio` 无关。** `aspectRatio` 只决定**像素长宽比**（黑边 + 拉伸），不改变任何**谱面数据**的解读。
> 因此：**同一个谱面在任何 `aspectRatio` 下 `positionX` 的语义完全一致**；变的只是观感（图案被横向/纵向拉伸）。

### 5.3 对 wiki 表述的精确化（修正项）

wiki 说「当谱面展示比例不为 3:2 时，在**纵向**上对判定线锚点坐标进行拉伸」（[判定线](https://pgrfm.miraheze.org/wiki/%E5%88%A4%E5%AE%9A%E7%BA%BF)，**B**）。**精确表述应为**：

`@
px_x / px_y = (2/3)·ar
  ar > 3/2（如 16:9）→ 相对 3:2 是「横向拉伸、纵向压缩」（16:9 时 x 单位比 y 单位大 32/27 ≈ 1.185）
  ar < 3/2（如 4:3，或窗口比 16:9 更窄时）→ 相对 3:2 才是「纵向拉伸」
`@
（依据：§5.1 的 A 级事实链。wiki 只说了 `ar < 3/2` 的那一半。）

### 5.4 逐条回答任务问题 3

- **3:2 舞台（1350×900）与 `aspectRatio`（默认 16:9）的换算公式** = §5.2 的两行。**没有**「先把 3:2 舞台等比缩放到 16:9」这一步——**是直接铺满**（非等比）。这一点此前所有来源都没写清。

---

## 6. 速度 / 时间 / 偏移单位

### 6.1 `speed`（判定线速度事件 + 音符速度）

**A 级定义**（[prpr/src/parse/rpe.rs:24](https://github.com/TeamFlos/phira/blob/main/prpr/src/parse/rpe.rs#L24) + [core.rs:15-16](https://github.com/TeamFlos/phira/blob/main/prpr/src/core.rs#L15-L16)）：

`@rust
pub const NOTE_WIDTH_RATIO_BASE: f64 = 0.13175016;
pub const HEIGHT_RATIO: f64 = 0.83175;
const SPEED_RATIO: f64 = 10. / 45. / HEIGHT_RATIO;     // ≈ 0.2671743  世界系 y 单位 / 秒 / 速度单位
`@
用法：`start_speed = event.start * SPEED_RATIO`（[rpe.rs:371](https://github.com/TeamFlos/phira/blob/main/prpr/src/parse/rpe.rs#L371)），累积成音符的「高度」（[rpe.rs:363](https://github.com/TeamFlos/phira/blob/main/prpr/src/parse/rpe.rs#L363)）。

**换算到 RPE 舞台系**（prpr 用 `× RPE_HEIGHT/2` 把高度显示为舞台单位，[note.rs:172](https://github.com/TeamFlos/phira/blob/main/prpr/src/core/note.rs#L172)，**A**）：

`@
1 个 RPE 速度单位  =  450 × SPEED_RATIO  RPE-y 单位/秒
                  =  450 × 10 / (45 × 0.83175)
                  =  120.228 RPE-y 单位/秒
即：速度 v 的音符，纵向速度 = 120.23 · v  （RPE-y 单位/秒；舞台高 = 900 单位）
    穿过整个舞台高度所需时间 = 900 / (120.23·v) = 7.486 / v 秒
`@

### 6.2 两个说法如何统一（任务问题 5）

| 说法 | 级别 | 统一后的理解 |
| :-- | :-- | :-- |
| 「速度：改变线上 Note 下落的速度，**默认 10**」（[术语](https://killkpa.miraheze.org/wiki/%E6%9C%AF%E8%AF%AD)，**B**） | B | 说的是**事件值的默认数值**（量纲无关） |
| 「1 = 每秒下降 **120 px**」（[速度](https://pgrfm.miraheze.org/wiki/%E9%80%9F%E5%BA%A6)、[RPEJSON](https://pgrfm.miraheze.org/wiki/RPEJSON)，**B**） | B | 说的是**单位大小**。「px」= **RPE-y 坐标单位**（舞台高 = 900 个）。A 级算出 **120.23**，与 120 相差 **0.19%** |

**两说法不冲突**：前者是默认值，后者是单位。合并表述：**「RPE 速度事件的默认值为 10；1 个速度单位 ≈ 120.2 RPE 纵向坐标单位/秒」**。
**{推断} 120 vs 120.23 的来源**：若把 `HEIGHT_RATIO` 取为精确的 `5/6`，则 `450 × 10/(45 × 5/6) = 120.0` **恰好整数**。prpr 用的是 `0.83175`（比 5/6 小 0.2%）。→ **prpr `HEIGHT_RATIO = 0.83175` 的出处未查证**（存疑 **D3**）；wiki 的「120」很可能是对精确值 120.0 的表述，而 prpr 的常量另有来源（可能拟合自官方客户端观感）。
**默认 10 的 A 级旁证**：prpr 无默认值（RPE 文件必含该字段），phichain 的 `RpeNote` 默认 `speed: 1.0` 是**音符 speed**（乘数），**不是速度事件值**——两者是不同的量，勿混（[schema.rs:225](https://github.com/Ivan-1F/phichain/blob/master/phichain-format/src/rpe/schema.rs#L225)，**A**）。

### 6.3 `yOffset`

**A 级公式**（[prpr/src/parse/rpe.rs:538](https://github.com/TeamFlos/phira/blob/main/prpr/src/parse/rpe.rs#L538)）：

`@rust
let y_offset = note.y_offset * 2. / RPE_HEIGHT * note.speed;   // 世界系 y（后续还会 / ar）
`@
→ 换算回 RPE 舞台系：**实际偏移 (RPE-y 单位) = yOffset × note.speed**，与 KillKPA「实际的偏移量等于速度乘上 Y 值偏移」**逐字一致**（[术语](https://killkpa.miraheze.org/wiki/%E6%9C%AF%E8%AF%AD)，**B**），且 **speed = 0 时偏移恒为 0**（源码直接相乘，**A**）。
偏移方向随 `above` 侧别镜像（§2.1，**A**）。

### 6.4 `visibleTime` 默认值裁定（任务问题 4）

| 来源 | 值 | 级别 |
| :-- | :-- | :-- |
| [Phira 文档·note](https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/note.html) | `999999.0000` | **A** |
| [KillKPA·术语](https://killkpa.miraheze.org/wiki/%E6%9C%AF%E8%AF%AD) | `999999.0` | **B** |
| phichain 源码注释 **+** `Default` 实现 | `// ignored, default 999999.0000` / `visible_time: 999999.0` | **A**（[schema.rs:210](https://github.com/Ivan-1F/phichain/blob/master/phichain-format/src/rpe/schema.rs#L210)、[schema.rs:227](https://github.com/Ivan-1F/phichain/blob/master/phichain-format/src/rpe/schema.rs#L227)） |
| [pgrfm·RPEJSON](https://pgrfm.miraheze.org/wiki/RPEJSON) | `99999.0` | **B（少数派，判为笔误）** |

> **裁定：默认值 = `999999.0`。** 三条证据（Phira 官方文档 A、KillKPA wiki B、phichain 源码 A，其中后两者的数字**连小数位写法都一致**）对一条孤立 B 级值。建议在解析器中**不依赖默认值**（A 级 prpr 把该字段设为必填：`visible_time: f64` 无 `#[serde(default)]`，[rpe.rs:158](https://github.com/TeamFlos/phira/blob/main/prpr/src/parse/rpe.rs#L158)）。

**语义（A 级）**：`visibleTime` 是**音符在被判定前多少秒开始显示**，prpr 用绝对时间比较：`if note.visible_time >= time { 始终可见 } else { 在 (time − visible_time) 淡入 }`（[rpe.rs:578-587](https://github.com/TeamFlos/phira/blob/main/prpr/src/parse/rpe.rs#L578-L587)，**A**）。默认 999999 秒 ≈ 始终可见。

### 6.5 Beat 三元组

- A 级类型定义：`Triple(i32, u32, u32)`，文档注释 `/// (i, n, d): i + n / d`（[prpr/src/core.rs](https://github.com/TeamFlos/phira/blob/main/prpr/src/core.rs)，**A**）→ `[0,0,1] = 0 拍`。
- A 级换算：prpr 用 `BpmList::time_beats(beats) = Σ (Δbeats × 60/bpm)`（同上，**A**）——即 **beat → 秒**由 BPM 分段线性给出。
- **`bpmfactor`**：prpr **未实现**（`// TODO bpmfactor`，[rpe.rs:171](https://github.com/TeamFlos/phira/blob/main/prpr/src/parse/rpe.rs#L171)，**A**）。→ 与 phigros-format.md 的 Q7 相关：**Phira 客户端自身不处理 `bpmfactor`**；解析器是否需要支持它，建议以「Phira 是目标运行时」为准，**存疑 D4**。
- `META.offset`：prpr 未在本文件读到的位置使用（**未查证**其符号方向，沿用 phigros-format.md Q11）。

### 6.6 时间单位总表

| 量 | 单位 | 级别 |
| :-- | :-- | :-- |
| RPE `startTime`/`endTime` | Beat 三元组（i + n/d 拍） | **A**（prpr `Triple`） |
| Beat → 秒 | 由 `BPMList` 分段积分 | **A** |
| `visibleTime` | **秒**（判定前的显示提前量） | **A**（prpr 直接与秒比较）+ **B**（KillKPA「单位秒」） |
| `yOffset` | **RPE-y 坐标单位**（须乘 `speed`） | **A** |
| `speed`（事件值） | 无量纲数值；1 单位 = 120.23 RPE-y 单位/秒 | **A**（常量）+ 推导 |
| `alpha` | 0~255（prpr 除以 255；`alpha >= 255` 视为不透明） | **A**（[rpe.rs:582-586](https://github.com/TeamFlos/phira/blob/main/prpr/src/parse/rpe.rs#L582-L586)） |
| `size` | 音符宽度相对标准宽度的倍数 | **B**（KillKPA）；prpr 直接作 x 缩放（**A**：[rpe.rs:589](https://github.com/TeamFlos/phira/blob/main/prpr/src/parse/rpe.rs#L589)） |
| `info.yml.offset` | **秒** | **A**（Phira 文档）；phichain 官方转谱把官谱 offset `× 1000.0` 转毫秒（[into_phichain.rs:30](https://github.com/Ivan-1F/phichain/blob/master/phichain-format/src/official/into_phichain.rs#L30)，**A**） |

### 6.7 顺带确证：音符 `type` 数字（A 级源码）

`@rust
// prpr/src/parse/rpe.rs:539-552
1 => NoteKind::Click, 2 => Hold, 3 => Flick, 4 => Drag, _ => bail
`@
（[rpe.rs:539-552](https://github.com/TeamFlos/phira/blob/main/prpr/src/parse/rpe.rs#L539-L552)，**A**）→ **RPE 侧 `1=Tap / 2=Hold / 3=Flick / 4=Drag` 由 phira 官方实现源码确证**（此前只有 A 级文档 + B 级 wiki）。

---

## 7. 对 RFC-0029 §3.1 的直接裁定：`positionX` 网格桶宽

### 7.1 裁定

> **`positionX` 的单位已确证（§3）。因此「桶宽未知」这一阻塞条件解除。**
> **建议：`Δx = RPE_STAGE_WIDTH / N`，默认 `N = 128` → `Δx = 1350/128 = 10.546875` RPE-x 单位**（= 舞台宽度的 0.781%）。**该常量必须是派生量（`1350 / N`），不得写成裸的 `10.55`**——满足 CLAUDE.md v3.0 红线 7「物理常量必须派生 + 断言」。

### 7.2 依据

| # | 依据 | 来源 |
| :-- | :-- | :-- |
| 1 | 网格定义域就是舞台宽度 1350 单位（不是 128、不是 ±1） | **A**（§3） |
| 2 | 音符**自身贴图宽度** ≈ `NOTE_WIDTH_RATIO_BASE/2 × 1350 = 88.93` RPE-x 单位（`NOTE_WIDTH_RATIO_BASE = 0.13175016`，世界系全宽 = 2） | **A 常量 + 【推断】**（[core.rs:15](https://github.com/TeamFlos/phira/blob/main/prpr/src/core.rs#L15)、[note.rs:206-210](https://github.com/TeamFlos/phira/blob/main/prpr/src/core/note.rs#L206-L210)） |
| 3 | `N=128` 时一个音符宽 ≈ **8.4 个桶** → 网格比音符足迹细一个数量级，**不会把肉眼可分辨的落点压进同一格** | 由 1+2 推导 |
| 4 | RFC §3.2 硬约束：`∫λ` 必须与强度场**同网格**数值积分、分辨率**显式声明** → 桶宽必须是可断言的常量 | RFC-0029 §3.2/§7 |
| 5 | 128 正是 RFC 原文的候选值，现在它**由单位推导而来**，不再是魔数 | RFC-0029 §8.2 Q8 |
| 6 | `positionX` **与判定线无关**（无任何按线缩放，§3/§4）→ **所有判定线可共用同一条 x 轴网格**，多线 v1 不需要 per-line 的坐标校正 | **A**（[rpe.rs:588](https://github.com/TeamFlos/phira/blob/main/prpr/src/parse/rpe.rs#L588)） |

### 7.3 必须补做的一次消融（否则等于把 128 当成新魔数）

1. **桶宽敏感性**：`N ∈ {64, 128, 256, 512}`（Δx = 21.09 / 10.55 / 5.27 / 2.64），报告事件级 F1 与 Poission NLL 校准，按 RFC §6.5 作为消融项。
2. **共格碰撞率**：统计训练集中**同一判定线、同一判定时刻、同侧**的音符对的最小 |ΔpositionX|；要求 `Δx_grid ≤ 该最小值`（否则两个事件落进同一格，`Σ log λ(e_k)` 会重复计同一格而失真）。**该统计量目前无数据，先算再定 N**。
3. **契约断言**：`field.x_bins = 128`、`field.dx = RPE_STAGE_WIDTH / field.x_bins`、`field.x_min = -675`、`field.x_max = +675`，并断言 `dx * x_bins == 1350`。

### 7.4 是否应改为连续回归？

**v1 建议：不改，仍用离散桶**；理由：

1. 泊松 NLL 的`∫λ` 要求数值积分与网格一致（RFC §3.2 硬约束）。连续 x 需在 x 轴上做数值求积（例如每轴 32 点 Gauss–Legendre → 每 (t,side,type) 单元多 32 倍计算），在 1e4 级数据上**难以在门禁里验证**；
2. RFC §1 的失败教训是「单位/分辨率没钉死就扩规模」——离散 + 显式声明的 Δx 更符合这一门禁精神；
3. **但保留连续回归为对照臂**：等 §7.3 的消融跑完，若 `N=512` 仍显著优于 `N=128`，说明分辨率是瓶颈，此时再上连续回归（或 `N≥1024` + 亚格峰回归）才有依据。

### 7.5 越界处理（与红线 3 的关系）

- **解析阶段**：`|positionX| > 675` 的值**不得钳位**（改变落点 = 改 AI 逻辑，违 CLAUDE.md 红线 3），只**统计并报告**占比；
- **生成阶段**：`|x| > 675` 区域的 λ 应置 0（或加越界惩罚），使「舞台外落点」在泊松 NLL 下概率为 0。

---

## 8. 对 `docs/knowledges/phigros-format.md` 的修正清单

> **本文件未直接改动 phigros-format.md**，仅列出建议改法。行号对应该文件的当前版本。

| # | 位置 | 原表述 | 新证据 | 建议改法 |
| :-- | :-- | :-- | :-- | :-- |
| M1 | §6.4 表格第 4 行 | 「RPE 速度事件数值单位 `1` = 每秒下降 **120 px**（B 级）」 | **A**：`SPEED_RATIO = 10/45/HEIGHT_RATIO`，`HEIGHT_RATIO = 0.83175` → `450 × SPEED_RATIO = 120.228` | 改为「**A 级（源码常量）**：1 速度单位 = 120.23 **RPE-y 坐标单位**/秒；wiki 的 120 px 与之相符（差 0.19%）；『px』须明确为 RPE-y 单位而非屏幕像素」。并升级级别 B→A |
| M2 | §6.4 表格第 5 行 | 「`lineLength` 默认 6.0；单位待补充」 | **A**：`draw_line(-len,0,len,0)`（局部系半长），世界半屏宽 = 1 = 675 单位 | 改为「**半长**，单位 = 半屏宽 = 675 RPE-x 单位；默认 6.0 ⟹ 半长 4050 单位 = 3 倍屏幕宽（两侧）；**不影响 `positionX` 映射**」 |
| M3 | §6.4 的 ⚠️ 段 | 「**RPEJSON 中 `positionX` 的物理单位在任何来源中都未被显式写出**」+【推断】 | **A**：`position_x / (RPE_WIDTH/2)` 就是那段代码 | 删除「未写出」与【推断】，改为确证结论 + 源码链接；并补「与 `moveXEvents` 同刻度」 |
| M4 | §6.5 推断式里的 `k`（「RPE 速度单位 → RPE 坐标系单位 的换算」= 未查证） | 未查证 | **A**：`k = 450 × 10/(45 × 0.83175) = 120.228` RPE-y 单位/秒 | 把 `k` 替换为该常量并升级为 A 级；同时把「局部 X = positionX」的「可能再乘 posControl 的 pos 倍率」补上源码依据（prpr `tr.x *= incline_val * ctrl_obj.pos`，[note.rs:177-182](https://github.com/TeamFlos/phira/blob/main/prpr/src/core/note.rs#L177-L182)），并注明 Hold 音符**不**乘 pos 倍率 |
| M5 | §6.1/§6.2 | side 语义（`above==1` = 正面/上方） | **A**：下侧音符被整体 Y 镜像渲染（[line.rs:412-429](https://github.com/TeamFlos/phira/blob/main/prpr/src/core/line.rs#L412-L429)） | 在 §6.2 的机制链条里补一条 A 级源码证据：**两侧是关于判定线 X 轴的镜像**，这也解释了 `yOffset` 的「向下落朝向偏移」 |
| M6 | §6.3 末段 | 「当谱面展示比例不为 3:2 时，在**纵向**上对判定线锚点坐标进行拉伸」（B） | **A**：`tr.y /= aspect_ratio`；全屏四边形 = `x∈[-1,1], y∈[-1/ar,1/ar]`；viewport 按 `ar` 黑边 | 改为精确式：**舞台永远铺满视口**；`px_x/px_y = (2/3)·ar`；`ar>3/2` 是横向拉伸、`ar<3/2` 才是纵向拉伸；补 `ar = forceAspectRatio ? info.aspectRatio : min(info.aspectRatio, win_w/win_h)` |
| M7 | §11 表格 | 只列了 Phira 文档给出的 `info.yml` 字段 | **A**：prpr `ChartInfo` 还有 4 个字段（camelCase 序列化）：`noteUniformScale`、`forceAspectRatio`、`useRpe170Speed`、`useAttachUiFix`（[info.rs](https://github.com/TeamFlos/phira/blob/main/prpr/src/info.rs)） | 补这 4 行，并注明「Phira 文档未收录，来源为 prpr 源码」。`useRpe170Speed` 与 `SpeedEasingMode::Modern/Legacy` 的选择直接相关（[rpe.rs:671-674](https://github.com/TeamFlos/phira/blob/main/prpr/src/parse/rpe.rs#L671-L674)），对 phigros-format.md **Q6（速度缓动）**是 A 级证据 |
| M8 | §12 Q2 | 「`positionX` 单位/范围/与线长关系未确证」 | 见 §3/§4 | **标记为已解决**，指向本文件 §3/§4 |
| M9 | §12 Q9（`visibleTime` 默认值） | 99999 / 999999 / 999999.0000 三说 | phichain 源码 `999999.0` + Phira 文档 + KillKPA | **裁定 `999999.0`**；把 pgrfm 的 `99999` 标注为笔误 |
| M10 | §12 Q17（3:2 vs 16:9 换算） | 「未查到换算公式」 | §5 | **标记为已解决**，指向本文件 §5 |
| M11 | §12 Q19（RPE「120 px/s」与官谱 Y=0.6H 的换算） | 无法换算 | RPE 侧已 A 级（120.23 RPE-y 单位/秒）；官谱 `1Y = 0.6H` **仍未在参考实现中查证** | 拆成两半：RPE 侧**已解决**；官谱侧维持 C 级 + 存疑 |
| M12 | §5.1/§5.2（note `type` 映射） | RPE 侧 A 级文档 + B 级 wiki | prpr 源码 `1/2/3/4 = Click/Hold/Flick/Drag` | 补源码链接，升级为「A 级源码确证」；官谱侧（C 级 Lchzh）维持 |
| M13 | §5.4 表格「`yOffset` 偏移量 = `yOffset * speed`」 | 已引 Phira 文档 | prpr 源码逐字一致 | 升级为 A 级源码；补「回舞台单位后正好是 `yOffset × speed`」 |
| M14 | §7.1 两套时间制 | Beat 三元组语义 | prpr `Triple(i32,u32,u32)` 注释 `i + n/d`；`bpmfactor` 在 prpr 中是 **TODO（未实现）** | 补 A 级源码；并在 Q7 处补一条：「Phira 参考实现未实现 `bpmfactor`」 |
| M15 | §6.3 官谱坐标表 | v1 范围「右上角 (880, 520)」 | phichain 官谱转换用 **/880 与 /530**（y 用 530）；Lchzh（C）写 520 | 标注 **520 vs 530 冲突**，进存疑清单；同时确认 v3 = [0,1]²（**A**：`x/y = (v−0.5)×CANVAS_*`） |

---

## 9. 存疑清单（本文件新增）

| # | 问题 | 目前证据状态 |
| :-- | :-- | :-- |
| **D1** | RPE `rotateEvents` 正角在**屏幕**上到底是顺时针还是逆时针？ | B 级 wiki 明写「正数表示顺时针」；phichain 导出时对旋转取负（A 级代码 + 注释「RPE rotation uses opposite sign convention」），**与该说法相容但不能独立证明**。prpr 的旋转经 y 翻转后进入世界系，未做符号断言。**未查证**（对建模影响小：事件轨是条件输入） |
| **D2** | Phichain 文档「判定线的长度为 3 倍屏幕宽度」是半长还是全长？ | prpr 代码算出**半长** = 3 倍屏宽（6 × 675 = 4050 = 3 × 1350），两实现数值自洽，仅措辞歧义。**未查证** |
| **D3** | prpr `HEIGHT_RATIO = 0.83175` 的出处？ | **未查证**。若取 5/6 则「1 速度单位 = 恰好 120 RPE-y 单位/秒」，与 wiki 完全吻合；prpr 的 0.83175 比 5/6 小 0.2% |
| **D4** | RPE `bpmfactor` 是否需要在解析器中支持？ | prpr 标为 TODO（**未实现**）；Phira 文档两页示例代码互相矛盾（见 phigros-format.md Q7）。**未查证** |
| **D5** | 官谱 `positionX` 单位是 `W/18` 还是 `0.05625W`？ | phichain（A 级源码）：`note.x / 18.0 * CANVAS_WIDTH` ⟹ 1 单位 = **75** RPE 单位；Lchzh（C 级）：`1X = 0.05625W` ⟹ **75.94** RPE 单位（0.05625 × 1350）。**差 1.25%，未裁定**。注：`0.05625W = 0.1H`（16:9），即「屏幕高 = 10X」；而 `W/18` 意为「屏幕宽 = 18 单位」 |
| **D6** | 官谱 v1 移动事件的 y 归一化分母是 520 还是 530？ | phichain 用 **530**（[into_phichain.rs:73](https://github.com/Ivan-1F/phichain/blob/master/phichain-format/src/official/into_phichain.rs#L73)）；Lchzh（C）写 520。**冲突未裁定** |
| **D7** | 官谱 `floorPosition`/`speed` 的「`1Y = 0.6H`」是否有参考实现支持？ | phichain 的官谱 schema 定义了 `floorPosition` 字段，但**其官谱→phichain 转换实现中未见使用**（只用 `note.speed` 直传）。**未查证** → 该单位仍只有 C 级来源 |
| **D8** | `lineLength` 是否还参与判定/命中区计算？ | 本文读过的 prpr 源文件中，`line_length` **只出现在 `core/line.rs` 的绘制处**；**未做全仓 grep**（`codeload.github.com` 在本环境不可达），故不作「唯一用途」的强断言 |
| **D9** | 音符贴图宽度 88.93 RPE-x 单位的推导是否与官方观感一致？ | 【推断】由 A 级常量 `NOTE_WIDTH_RATIO_BASE` 与「世界系全宽 = 2」推出；`config.note_scale` 默认值**未查证**（渲染器配置项，非谱面字段） |
| **D10** | 官谱 `offset` 的符号方向？ | phichain 只做 `× 1000`（秒→毫秒）单位转换，**不涉及符号**。沿用 phigros-format.md Q10。**未查证** |

---

## 10. 来源清单

### 10.1 网页来源

| 标题 | URL | 级别 |
| :-- | :-- | :-- |
| Phira 文档·谱面信息（ChartInfo / info.yml） | <https://teamflos.github.io/phira-docs/chart-standard/chartinfo.html> | A |
| Phira 文档·RPE 判定线（含 `conrpepos`） | <https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/judgeLine.html> | A |
| Phira 文档·RPE 音符 | <https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/note.html> | A |
| Phira 文档·RPE 普通事件 | <https://teamflos.github.io/phira-docs/chart-standard/chart-format/rpe/event.html> | A |
| Phichain 文档·判定线（画布范围 / 线长） | <https://phichain.rs/docs/line> | A（第三方实现方） |
| Phichain 文档·Phira 格式兼容 | <https://phichain.rs/docs/respack/phira-compatibility> | A |
| KillKPA·术语（`positionX`/`yOffset`/`visibleTime`/速度默认值） | <https://killkpa.miraheze.org/wiki/%E6%9C%AF%E8%AF%AD> | B |
| Phigros 自制谱 wiki·判定线 | <https://pgrfm.miraheze.org/wiki/%E5%88%A4%E5%AE%9A%E7%BA%BF> | B |
| Phigros 自制谱 wiki·事件 | <https://pgrfm.miraheze.org/wiki/%E4%BA%8B%E4%BB%B6> | B |
| Phigros 自制谱 wiki·速度 | <https://pgrfm.miraheze.org/wiki/%E9%80%9F%E5%BA%A6> | B |
| Phigros 自制谱 wiki·RPEJSON | <https://pgrfm.miraheze.org/wiki/RPEJSON> | B |
| Lchzh Docs·Phigros 谱面格式说明（官谱 X/Y/T 单位） | <https://docs.lchzh.top/learning/phigros/> | **C** |

### 10.2 源码来源（A）

全部经 `raw.githubusercontent.com` / `cdn.jsdelivr.net` 逐文件读取，**行号对应当前 `main`/`master` 分支**。

| 文件 | 本文件用到的内容 |
| :-- | :-- |
| [phira/prpr/src/parse/rpe.rs](https://github.com/TeamFlos/phira/blob/main/prpr/src/parse/rpe.rs) | `RPE_WIDTH/HEIGHT`(22-23)、`SPEED_RATIO`(24)、`RPENote` schema(142-165)、音符 `type` 映射(539-552)、`y_offset`(538)、音符 `translation`(588)、move 事件因子(682-684)、速度事件解析(332-410)、`visibleTime` 淡入(578-587) |
| [phira/prpr/src/core.rs](https://github.com/TeamFlos/phira/blob/main/prpr/src/core.rs) | `NOTE_WIDTH_RATIO_BASE`/`HEIGHT_RATIO`(15-16)、`Triple` 与 `BpmList` |
| [phira/prpr/src/core/line.rs](https://github.com/TeamFlos/phira/blob/main/prpr/src/core/line.rs) | 判定线绘制 `draw_line(-len,0,len,0)`(251-252)、`(vw,vh)`(386-391)、下侧音符 Y 镜像(412-429)、父线合成(221-234) |
| [phira/prpr/src/core/note.rs](https://github.com/TeamFlos/phira/blob/main/prpr/src/core/note.rs) | 音符缩放/`note_width`(206-210)、高度与 `/aspect_ratio`(218-223)、`init_ctrl_obj` 的 `× RPE_HEIGHT/2`(172)、`pos` 倍率与 incline(175-192) |
| [phira/prpr/src/core/object.rs](https://github.com/TeamFlos/phira/blob/main/prpr/src/core/object.rs) | `Object` 定义、`now_translation` 的 `tr.y /= aspect_ratio`(51-56) |
| [phira/prpr/src/core/resource.rs](https://github.com/TeamFlos/phira/blob/main/prpr/src/core/resource.rs) | 相机 `zoom`(505-509)、`note_width`(520)、`viewport()` 与 `force_aspect_ratio`/`min(...)`(597-619) |
| [phira/prpr/src/core/video.rs](https://github.com/TeamFlos/phira/blob/main/prpr/src/core/video.rs) | 全屏四边形 `Rect(-1,-1/ar,2,2/ar)`(153-154) |
| [phira/prpr/src/core/chart.rs](https://github.com/TeamFlos/phira/blob/main/prpr/src/core/chart.rs) | 谱面渲染时的 Y 翻转(159) |
| [phira/prpr/src/info.rs](https://github.com/TeamFlos/phira/blob/main/prpr/src/info.rs) | `ChartInfo` 全字段与默认值（`line_length: 6.`、`aspect_ratio: 16./9.`、4 个未入文档字段） |
| [phichain/phichain-chart/src/constants.rs](https://github.com/Ivan-1F/phichain/blob/master/phichain-chart/src/constants.rs) | `CANVAS_WIDTH/HEIGHT = 1350/900` |
| [phichain/phichain-format/src/rpe/schema.rs](https://github.com/Ivan-1F/phichain/blob/master/phichain-format/src/rpe/schema.rs) | `visible_time` 默认 `999999.0`、`speed` 默认 `1.0`、`above` 默认 `1` |
| [phichain/phichain-format/src/rpe/from_phichain.rs](https://github.com/Ivan-1F/phichain/blob/master/phichain-format/src/rpe/from_phichain.rs) | `position_x` 直传、旋转取负注释、`numOfNotes` 不含 Hold |
| [phichain/phichain-format/src/official/into_phichain.rs](https://github.com/Ivan-1F/phichain/blob/master/phichain-format/src/official/into_phichain.rs) | 官谱 `T = ×1.875/60 拍`(36)、`x/y=(v−0.5)×CANVAS_*`(37-38)、note `x/18.0×CANVAS_WIDTH`(54)、v1 `/880`、`/530`(68,73)、v3 `[0,1]`(76-87)、offset 秒→毫秒(30) |

### 10.3 未能访问的来源

| 目标 | 状态 |
| :-- | :-- |
| `codeload.github.com`（整仓 tarball → 全仓 grep） | 本环境不可达（curl 下载 0 字节）→ 影响 D8 |
| `grep.app` 代码搜索 API | 被 Vercel 安全检查拦截 |
| GitHub Code Search API（`/search/code`） | 需认证，未使用 |
| RPE（Re:PhiEdit）官方 schema / 源码 | 未找到公开仓库（Re:PhiEdit 为闭源制谱器） |
