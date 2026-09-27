# Plan 03 — 强度场模块（判定线局部系多线标记点过程）

> 状态：🟢 已实施（M1–M12 全部落地并自检通过，2026-09-26；M6/M7 的全库数值与 N 的最终取值待 plan 02 统计回填）｜ 阶段：Phase 2 ｜ 负责：field-agent（强度场组）
> 实施留痕：`beatmorph/field/*.py`（源码零物理常量字面量，AST 扫描契约测试守住）｜ `tests/unit/field/` + `tests/integration/test_field_pipeline.py` 默认 CI **103 passed / 0 skipped**，全仓默认 CI 403 passed ｜ ruff 0 error、mypy strict 0 error ｜ G1–G4 门禁已实跑全绿（`tests/unit/field/test_sanity_gates.py`，标 slow）
> 对应代码：`beatmorph/field/` ｜ 对应奠基章节：§1.2 / §2 / §3.2 / §3.4 / §7

## 1. 目标与范围

### 1.1 交付

1. **场定义与网格契约**：`λ_k(τ, x, s, c)`，`k = 1..K`；**`τ` 为拍相对坐标（beat-aligned，基本格 `Δτ = 1/48` 拍，网格随 BPM 变化——RFC-0029 §3.1 Q15 决议）**；`x` 为 **RPE 舞台系 x 坐标**网格，桶宽 **`Δx = RPE_STAGE_WIDTH / N`（默认 N = 128）**；`s ∈ {above, below}`；`c` 为类型通道（tap / drag / hold / flick **+ hold-end 标记**）。全部量写成**派生式**，禁止字面量（CLAUDE.md 红线 7、AGENTS.md §4）。
2. **目标构建**：`PhigrosChart`（IR）→ **桶内计数张量** `n_j` + mask 通道 + meta。必须能处理**强不平衡**（背面仅 2.4–3.0% 的 note）与 **`line_id` 极度集中**（最忙线独占中位 73% 的 note）这两种真实分布。
3. **两条积分路径 + 一致性契约门禁**：(a) 与强度场**同网格**的数值积分；(b) **累积强度参数化** Λ(t) 再求导（Omi et al., NeurIPS 2019，arXiv 1905.09690）。同一场上二者给出的 NLL 必须一致到声明容差——**这本身是一道契约门禁**（奠基 §3.4、RFC-0029 §3.2）。
4. **非齐次泊松 NLL**（含**桶内计数语义**）+ **G3 常数基线 `λ = N/|Ω|`** + **`λ ≡ 0 ⇒ NLL = +∞` 的契约断言**。
5. **共格碰撞统计** + **N ∈ {64, 128, 256, 512} 消融**（否则 128 只是换了个魔数）。
6. **可视化 / 调试工具**：把场渲染成人眼可查的图。
7. 供 plan 04（生成主干）与 plan 05（解码）消费的稳定张量形状与 API。
8. **秒 ↔ τ 的唯一换算实现 + 「场网格 ↔ 秒」往返无损契约测试**：`J(τ) = dt/dτ` 由谱面 `BPMList` **派生**（不得硬编码），测度改为 `J(τ)dτ`；契约层 `PhigrosNote.t` **仍用秒**。本模块是**唯一**允许实现该换算的地方（RFC-0029 §3.1/§7-8、CLAUDE.md 红线 7）——**没有这条测试，beat-aligned 会变成下一个 25 Hz**。

### 1.2 不交付

- 网络结构、训练循环、消融臂编排（plan 04）；场 → 离散事件解码器与合法性后处理（plan 05）；评估指标与人评协议（plan 06）；RPEJSON 解析器与 Chart IR 构造（plan 02）；`core/contracts` 本体（contracts-agent，plan 00）。

### 1.3 价值

本模块是范式的**测度定义层**：它把「谱面」变成 Ω 上的可积强度场，NLL、门禁、解码、校准全部由它出口。它同时是**红线 7 的主战场**——网格桶宽、**拍格 `Δτ` 与 Jacobian `J(τ)`**、帧率（音频帧轴）、舞台宽度、`|Ω|` 任意一个漂移，下游全部静默失效（POSTMORTEM-2026-08-05 的同构形态）。**Q15 之后，秒 ↔ τ 的换算也落在这里，且只落在这里。**

---

## 2. 与奠基文档对应

| 本计划项 | 文档依据 | 落地 |
|---|---|---|
| `λ_k(τ,x,s,c)`，k = 1..K，K 随谱变化；**τ = 拍相对坐标（beat-aligned）** | BasePlan §1.2/§3.2.4、RFC-0029 §3.1/§2.4（Q15 决议）、§7-8 | §3.1 契约 |
| 积分测度 `dt → J(τ)dτ`；`J(τ) = dt/dτ` 由 `BPMList` **精确给出**，不得硬编码 | RFC-0029 §3.1「数学代价」、§7-8、CLAUDE.md 红线 7 | §3.1 / §4.2 / §4.5 / §4.6 |
| 秒 ↔ τ 换算**只在本模块实现**；下游（generation / decoder / io / eval）不得各自再写一套 | RFC-0029 §3.1/§7-8、CLAUDE.md 红线 7 | §3.1 / §5「时间换算的单向出口」 |
| 「场网格 ↔ 秒」**往返无损**契约测试（覆盖多 BPM 段） | RFC-0029 §3.1（原文：「否则 beat-aligned 会变成下一个 25 Hz」） | M12 + §8 |
| `x` = RPE 舞台系 x（非像素、非归一化） | RFC-0029 §3.1、[units §2.4/§3.1](knowledges/phigros-units-and-geometry.md)（A 级：prpr `position_x / (RPE_WIDTH/2)`） | `RPE_STAGE_WIDTH` 派生 |
| 桶宽 `Δx = RPE_STAGE_WIDTH / N`，默认 N=128 | BasePlan §3.2.4、RFC-0029 §8.2 Q8 | §3.1/§4.2 |
| 多线**共用同一 x 网格**（无 per-line 缩放） | RFC-0029 §2.4-7、units §3.1 | §4.2（不做逐线坐标校正） |
| `s` = above/below 是**硬自由度**；`above == 1` → 正面，其余 → 背面 | BasePlan §3.2.2、RFC-0029 §8.1 Q3、units §2.1（A 级：下侧整体 Y 镜像） | §4.3 |
| `c` = tap/drag/hold/flick + hold-end | RFC-0029 §3.1 | §4.3 |
| 非齐次泊松 NLL + `∫λ` 同网格 | BasePlan §2/§3.4、RFC-0029 §3.2、RFC-0029 §7.4 | §3.3/§4.5 |
| 两条积分路径必须互相校验 | BasePlan §3.4、RFC-0029 §3.2 | §4.6 + 契约门禁 |
| G3 常数基线 `λ = N/\|Ω\|`（**不是 0**） | BasePlan §3.4、[literature §6.3](knowledges/chart-generation-literature.md) | §4.5 + M4 |
| 帧率**派生**（24000/320 = 75 Hz） | BasePlan §3.1、RFC-0029 §7.1 | `MERT_SAMPLE_RATE_HZ / MERT_CONV_STRIDE_PRODUCT` |
| 越界**只统计不钳位**；生成阶段域外 λ 置 0 | BasePlan §3.2.4、units §7.5、红线 3 | §4.4 |
| N 消融 + 共格碰撞统计 | BasePlan §3.2.4、units §7.3 | M6/M7 |
| 多线不可互换、无 line 分类损失 | RFC-0029 §2.4-3/§2.4-6 | §4.7 |

**偏离 1（本计划新增设计，需评审）**：把既有的 Λ 参数化**因子化**为
`λ_k(τ,x,s,c) = Λ'_k(τ) · p_k(x,s,c | τ)`，其中 `Σ_{x,s,c} p_k = 1`（时间轴为 τ，Q15）。
奠基只要求「两条积分路径存在且一致」，未规定实现方式；因子化让「一致」**等价于一条可断言的归一化条件**（而不是把两条路径写成互为同义反复）。见 §9-1。

**偏离 2（测度口径）**：`|Ω| ≡ Σ_{k=1..K} |Ω_k|`（**全 K 条线、全部格元计入**；与 [plan 00 §3.7](00-core-contracts.md) 的 `k * t_bins * x_bins * sides * channels` **同一口径**），即 G3 常数基线在**全部 K 条线**的测度上取 `N/|Ω|`，λ 的单位为「每格元的事件数」。**不得只对「有 note 的线」积分**——那会取消对空线的惩罚，使「线选择」无成本，直接摧毁 RFC-0029 §2.4-6 的竞争解释。见 §9-2。
> **⚠️ Q15 带来的口径变化（随裁决同步，见 §9-2）**：τ 网格**不再均匀**（格宽随 BPM 变化）⟹ 格体积 **`ΔV_j = J_j · d_tau · dx` 逐格不同**，`|Ω| = Σ_j ΔV_j`；RFC-0029 §8.4 R-a 的 `K · T · X · S · C` 是**均匀网格**下的等价写法。λ 的「每格元」单位口径须随之复核。

**偏离 3（`hold-end` 语义）**：`hold-end` 作为**同一 `(k, s, positionX 桶)` 纤维上的第二个点**（hold 起点记入 `hold` 通道，终点记入 `hold_end` 通道），使 Hold 时长 = 同纤维两点的时差。RFC-0029 §3.1 只写「+ hold-end 标记」，未规定它是独立点还是条件标记。见 §9-3。

**偏离 4（越界处置）**：`|positionX| > 675` 的 note **从事件项排除并单独计数**（`meta.n_out_of_range`），**不钳位**（红线 3）；场在域外恒 0，使「舞台外落点」在泊松 NLL 下概率为 0（units §7.5）。见 §9-4。

---

## 3. 接口契约

### 3.1 场、网格与常量（`beatmorph/field/grid.py`）

```python
RPE_STAGE_WIDTH: Final[float] = (
    1350.0  # 源自 core/contracts（A 级：prpr RPE_WIDTH / phichain CANVAS_WIDTH）
)
RPE_STAGE_HEIGHT: Final[float] = 900.0
DEFAULT_X_BINS: Final[int] = 128  # RFC-0029 §3.1 裁定
N_SIDES: Final[int] = 2  # above / below
TYPE_CHANNELS: Final[tuple[str, ...]] = ("tap", "drag", "hold", "hold_end", "flick")
# 时间基本格：1/48 拍（RFC-0029 §3.1 Q15 决议；网格随 BPM 变化）——派生式，禁止写字面量
BEAT_SUBDIVISION: Final[int] = 48
# 帧率是派生量，不是超参（BasePlan §3.1；POSTMORTEM-2026-08-05）——**它只服务音频帧轴（Stage 0），
# 不再是场的时间轴**；场的时间轴是 τ（拍）。两者的换算只在本模块进行（红线 7）。
MERT_FRAME_RATE_HZ: Final[float] = (
    MERT_SAMPLE_RATE_HZ / MERT_CONV_STRIDE_PRODUCT
)  # = 24000 / 320 = 75


@dataclass(frozen=True)
class FieldGrid:
    x_bins: int = DEFAULT_X_BINS

    @property
    def dx(self) -> float:
        return RPE_STAGE_WIDTH / self.x_bins  # 禁止写 10.546875

    @property
    def d_tau(self) -> float:
        return 1.0 / BEAT_SUBDIVISION  # 单位：拍；禁止写 0.020833...

    @property
    def x_min(self) -> float:
        return -RPE_STAGE_WIDTH / 2

    @property
    def x_max(self) -> float:
        return +RPE_STAGE_WIDTH / 2

    # ── 秒 ↔ τ 的**唯一**换算点（RFC-0029 §3.1/§7-8；下游一律经此，不得自行实现）──
    def jacobian(
        self, bpm_points
    ) -> (
        FloatTensor
    ): ...  # J(τ) = dt/dτ = 60 / bpm(τ)（秒/拍），由 BPMList 分段给出；段内常量、段界跳变
    def tau_to_seconds(
        self, tau, bpm_points
    ) -> ...: ...  # t = ∫_0^τ J = Σ_段 (Δ拍数 × 60 / bpm_段)
    def seconds_to_tau(
        self, t_s, bpm_points
    ) -> ...: ...  # 上式按段反解；与 tau_to_seconds 互为逆（M12 断言往返无损）
    def volume(
        self, tau_bins: int, n_lines: int, jacobian
    ) -> (
        float
    ): ...  # |Ω| = Σ_j J_j · d_tau · dx，j 遍历 (k, τ, x, s, c) 全部格元；**非均匀**（§2 偏离 2）
```

契约断言（默认 CI，无权重无 GPU）：
`dx * x_bins == RPE_STAGE_WIDTH`；`x_min == -x_max`；`d_tau * BEAT_SUBDIVISION == 1`（拍）；`frame_rate == MERT_SAMPLE_RATE_HZ / MERT_CONV_STRIDE_PRODUCT`；
**`seconds_to_tau(tau_to_seconds(τ)) == τ` 且反向亦然（容差内），且 `tau_to_seconds` 与「逐段 `Δ拍 × 60 / bpm` 求和」的解析值一致（多 BPM 段）——M12**；
`N_SIDES == 2`；`len(TYPE_CHANNELS) == 5`；源码中不得出现 `"10.546875"` / `"1350"` / `"75.0"` / `"0.020833"` 字面量（用正则契约测试扫描本模块）。

### 3.2 张量形状（einops 风格）

| 名称 | 形状 | dtype | 含义 |
|---|---|---|---|
| `FieldCounts` | `(K, T, N, 2, C)` | int16 | 桶内计数 `n_j`（**不是**二值；见 §4.5） |
| `FieldLambda` | `(K, T, N, 2, C)` | float32 | 强度 `λ_j ≥ 0` |
| `OcclusionMask` | `(K, T, N, 2, C)` | bool | **训练遮盖 mask 通道**（RFC-0029 §3.3-1） |
| `LineMask` | `(K,)` | bool | 有效线（batching padding 为 False） |
| `RangeMask` | `(N,)` | bool | `\|x\| ≤ 675`（域外 False，λ 恒 0） |
| `CumulativeLambda` | `(K, T)` | float32 | `Λ_k(τ)`（对 τ 的累积强度，含 `J(τ)` 测度），单调不减 |
| `CellProb` | `(K, T, N, 2, C)` | float32 | `p_k(x,s,c \| τ)`，`Σ_{x,s,c} p = 1` |

`T` = **τ 格数**（`T = round(τ_end × BEAT_SUBDIVISION)`，`τ_end` = 谱面终点换算到拍），由 `BPMList` 与谱面时长派生；**不得由 `frame_rate` 或音频时长直接派生**（那是音频帧轴，不是场的时间轴）。`K` 为 `judgeLineList` 长度（**不得硬编码**：中位 30、p25 24 / p75 52 / 极值 2–82，BasePlan §3.2.3、survey §7.1）。

### 3.3 NLL API（`beatmorph/field/loss.py`）

```python
def poisson_nll(counts, lam, grid: FieldGrid, *, line_mask=None, reduction="sum") -> Tensor
    # = -Σ_j n_j log λ_j + Σ_j λ_j ΔV_j ，j 遍历 (k, τ, x, s, c)；ΔV_j = J_j · d_tau · dx（J 由 BPMList 派生）
def binned_poisson_nll(counts, lam, grid, *, line_mask=None) -> Tensor
    # 等价的「分块泊松计数」口径：额外含 Σ_j log(n_j!) − N·log ΔV 两项
    # （与参数无关的常数；用于与 EBC-ZIP 式分块计数口径对齐与报告，默认不进梯度）
def constant_baseline_lambda(n_events: float, omega: float) -> float   # N / |Ω|
def constant_baseline_nll(n_events: float, omega: float) -> float      # N * (1 + log(|Ω| / N))
```

**契约断言**：
1. `poisson_nll(counts, λ ≡ N/|Ω|)` 必须等于闭式 `N * (1 + log(|Ω|/N))`（相对误差 ≤ 1e-6，float64）。
2. `λ ≡ 0` 时 NLL **必须**为 `+inf` / 非有限——**禁止**用 `eps` 平滑把它变成有限值（那正是 G3 立论的凭据）。
3. 桶内计数语义：`n_j = 2` 的格子贡献 `2 · log λ_j`；**禁止对事件去重**（[literature §4.4-4](knowledges/chart-generation-literature.md)）。
4. `binned_poisson_nll - poisson_nll == Σ_j log(n_j!) - Σ_j n_j·log ΔV_j`（相对误差 ≤ 1e-6），即两者只差一个与参数无关的常数——这是「同一测度」的机器可验证表述。**均匀网格下退化为 `Σ_j log(n_j!) - N·log ΔV`；τ 网格非均匀，不得再用后者的写法**（§4.5）。

### 3.4 积分 API（`beatmorph/field/integrate.py`）

```python
def integrate_grid(lam, grid: FieldGrid, *, line_mask=None) -> Tensor      # (K,) = Σ_j λ_j ΔV_j（ΔV_j = J_j · d_tau · dx）
def cumulative_from_lam(lam, grid) -> Tensor                              # (K, T) 单调不减
def lam_from_cumulative(cum, grid, *, mode: Literal["finite_diff", "autograd"]) -> Tensor
def integrate_cumulative(cum, *, line_mask=None) -> Tensor                # (K,) = Λ_k(T) - Λ_k(0)
```

**一致性契约门禁（本模块的核心门禁）**：对同一场，`integrate_grid` 与 `integrate_cumulative` 必须一致——
相对误差 ≤ **1e-6（float64）/ 1e-4（float32）**（容差为本计划设定的工程容差，非文档数字，见 §9-5）。
测试场集合：(i) 闭式可积的解析场（常数、线性 `λ(τ)=a·τ+b`、指数 `λ(τ)=a·e^(-bτ)`）；(ii) 随机分段常量场；(iii) 因子化场（先验 `p` 与 `Λ'` 分离构造）；(iv) **多 BPM 段场**（`BPMList` ≥ 2 段 ⟹ `J(τ)` 分段常量、`ΔV_j` 非均匀）——检验两条路径在**非均匀测度**下仍一致，且 `λ ≡ 常数` 时都给出 `λ · Σ_j ΔV_j`。
**失败模式即门禁价值**：若 `p` 在掩掉域外格子后忘记重新归一化，路径 (a) 与 (b) 必然分歧——门禁当场捕获。

### 3.5 目标构建与诊断 API

```python
def build_target(chart: PhigrosChart, grid: FieldGrid, *, include_fake: bool = False) -> FieldTarget
def min_same_instant_dx(chart: PhigrosChart) -> list[float]        # 同 (line, 精确时刻, side) 的最小 |ΔpositionX|
def cocell_report(chart: PhigrosChart, grid: FieldGrid) -> CocellReport
def render_field_png(lam, gt_counts, out_path, *, grid, **sel) -> Path
```

---

## 4. 内部设计

### 4.1 数值来源（禁止编造）

| 量 | 值 | 来源级别 |
|---|---|---|
| `RPE_STAGE_WIDTH` = 1350 | A：prpr `RPE_WIDTH` / phichain `CANVAS_WIDTH` | units §2.2 |
| `positionX` 语义可见范围 [−675, 675]，**无钳位** | A（实现侧）+ 实测 12674 note 极值恰为 ±675.000 | units §3.1、survey §7.4 |
| 桶宽 `Δx = 1350/N`；N=128 → 10.546875 | RFC-0029 §8.2 Q8 裁定 | RFC-0029 §3.1 |
| 帧率 = 24000/320 = 75 Hz | MERT config 派生 | BasePlan §3.1 |
| 背面上限 2.4–3.0%；`above ∈ {0,1,2}`，`==1` 为正面 | 实测（两轮都出现 0 与 2） | survey §7.3、RFC-0029 §8.1 Q3 |
| 类型占比 Tap 52–63% / Drag 20–31% / Hold 10–11% / Flick 6–7% | 实测两轮 | BasePlan §3.2.2、survey §7.3 |
| K 中位 30（p25 24 / p75 52）；58% 的线带 note；最忙线独占中位 73% | 实测 n=196 / n=23 | BasePlan §3.2.3、survey §7.2 |
| 手工谱位置网格 22.5 = 1350/60、11.25 = 1350/120；任意浮点最小间隔 0.324639 | 实测 n=3 张 | survey §7.4.1 |
| 音符贴图宽 ≈ 88.93 RPE-x 单位（N=128 时 ≈ 8.4 桶） | A 级常量 + 推断 | units §7.2 |

### 4.2 网格化

- **单一 x 网格服务全部 K 条线**：`positionX` 与判定线无关（无 per-line 缩放，RFC-0029 §2.4-7）→ **不做任何逐线坐标校正**，这是 v1 多线可行的关键简化。
- **时间轴网格 = τ（拍相对坐标，beat-aligned）**：`i_tau = floor(τ × BEAT_SUBDIVISION)`，`τ` 与秒之间的换算由 `BPMList` **分段派生**（`FieldGrid.jacobian` / `tau_to_seconds` / `seconds_to_tau`）。评估/报告一律回到**秒域**（RFC-0029 §7.5）；τ 格索引只活在张量内部，**不得**泄漏到下游 API。
- **音频帧轴（`frame_rate`）不构成场的时间轴**：它只用于与 `audio_emb` 的条件输入对齐，两者之间的重采样在同一处实现（§9-15）。
- 位置桶索引 `i_x = floor((x - x_min) / dx)`，`i_x ∈ [0, N)`；`|x| > 675` 的 note 记入 `meta.n_out_of_range` 并**排除**出事件项（§2 偏离 4）。
- 越界**不做整除校验**（实测存在分母 1802/3889 的任意浮点取值，survey §7.4.1）；只做区间统计。

### 4.3 通道与侧别

- `c` 五通道：`tap / drag / hold / hold_end / flick`。RPE 的 `type` 数字映射 **1 Tap / 2 Hold / 3 Flick / 4 Drag**（A 级 prpr 源码），由 plan 02 的解析器保证，本模块只接受已映射的枚举。
- `s` 二值：`above == 1 → FRONT`，**其余值（含 0 与 2）→ BACK**（RFC-0029 §8.1 Q3 原文）。**禁止当布尔解析**。
- **hold-end 配对**：Hold 起点的 `(k, i_x, s)` 与终点相同（RPE 的 Hold 只有单一 `positionX`），终点记在 `hold_end` 通道；目标构建时断言配对存在且 `end_time > start_time`，违反者计入 `meta.n_illegal_hold`（不修复——修复属 plan 05 的合法性后处理）。
- `isFake` 默认**排除**（假音符无判定、不计物量，格式文档 A 级字段说明），开关 `include_fake` 保留；`meta.n_fake` 必报。其余标记（`speed`/`size`/`yOffset`/`visibleTime`）不进 v1 场的定义域，作为**边际化掉的标记**处理，其默认值由 plan 02/05 负责。

### 4.4 三个 mask 的语义分离（易错点）

| mask | 作用 | 不得混用为 |
|---|---|---|
| `OcclusionMask` | 训练遮盖通道（RFC-0029 §3.3-1 要求显式提供） | 不能当有效性 mask |
| `LineMask` | batching padding / 无效线 | 不能参与 `∫λ` 求和（padding 线必须排除） |
| `RangeMask` | `\|x\| ≤ 675`；域外 λ 恒 0 | **不是钳位**，只是定义域 |

契约测试：`∫λ` 在 `line_mask` 全 False 时必须为 0；`line_mask` 置 False 的线对 NLL 的贡献必须恰为 0（梯度亦然）。

### 4.5 泊松 NLL、桶内计数与 G3

离散化口径（推导，非引用数字）：设场在格 `j` 上分段常量，τ 网格下格体积 **`ΔV_j = J_j · d_tau · dx`**（`J_j` 为该格所属 BPM 段的 `60 / bpm`；**逐格不同**），则

    L_point  = -Σ_j n_j log λ_j + Σ_j λ_j ΔV_j                      （与 RFC-0029 §3.2 写法逐项一致，测度已按 Q15 改写）
    L_binned = -Σ_j n_j log(λ_j ΔV_j) + Σ_j λ_j ΔV_j + Σ_j log(n_j!)
             = L_point - Σ_j n_j log ΔV_j + Σ_j log(n_j!)           （常数项，与参数无关）

> **均匀网格下**（`ΔV_j ≡ ΔV`）第二式退化为 `L_point - N log ΔV + Σ log(n_j!)`（旧写法）。τ 网格**非均匀**，因此实现与测试都必须用逐格 `ΔV_j`——这正是「单位错配」类 bug 的新入口。

两者对参数**等价**（梯度相同），数值不同；实现必须**显式声明报告的是哪一个**（默认 `L_point`，并同报 `L_binned`）。
「桶内 2 个事件只产生 1 个 log λ 项」是文献 §4.4-4 点名的失真源——本模块用**计数形式 `n_j log λ_j`** 从结构上排除它，并把「`n_j ≥ 2` 的格子数」作为**共格碰撞率**主指标报出（§4.8）。

**G3 常数基线**：`λ ≡ c` 时 `L_point = -N log c + c|Ω|`（`|Ω| = Σ_j ΔV_j`，均匀/非均匀网格同式），极值点 `c* = N/|Ω|`，最小值 `N(1 + log(|Ω|/N))`。基线以闭式实现并与数值优化结果对拍（M4）。
`λ ≡ 0` 时 `L = +∞`（`N > 0`），写成契约断言。

### 4.6 两条积分路径与 Λ 参数化

- **路径 (a) 同网格数值积分**：`Σ_j λ_j ΔV_j`（`ΔV_j = J_j · d_tau · dx`），格值与场**同一网格**（RFC-0029 §7.4 硬约束）。对「τ 上分段常量 + BPM 段内 `J` 常量」的场它是精确的。
- **测度一致性（Q15 的落点）**：事件项与积分项都在 τ 网格上，积分项**必须**乘 `J_j`——漏乘等价于把「拍」当「秒」用，是新的单位错配 bug 类；M12 守换算、M3 守加权，各守一半。
- **路径 (b) 累积强度参数化**（Omi et al. 2019）：网络输出单调不减的 `Λ_k(τ)`，`∫λ_k = Λ_k(T) − Λ_k(0)` 精确（`T` = τ 格数；测度含 `J(τ)`）；`λ` 由 `Λ` 求导得到（`autograd`，或与网格一致的有限差分，二者都实现并互测）。
- **因子化**（§2 偏离 1）：`μ_j = ΔΛ_k · p_j`，`Σ_{x,s,c} p_j = 1` ⟹ 两条路径**恒等**；门禁因此检验的是「归一化是否被掩码/域外裁剪破坏」这一真实 bug 类，而非两条同义代码的自我印证。
- **禁止 Monte-Carlo 估计 `∫λ`**（Jensen 偏差，literature §4.4-2）。
- 数值稳定：`λ` 用 softplus/exp 保证 ≥ 0；事件项与积分项**同量纲**（都是计数）。

### 4.7 强不平衡与 `line_id` 极度集中的**处理方式**

本模块**不做**损失重加权（理由：λ 是强度，逐格重加权会让事件项与积分项不再同测度，正是 focal 与泊松 NLL 不能混用的原因，RFC-0029 §3.4）。处理方式是结构性的：

1. **积分覆盖全部 K 条线与两侧**：空线的 `∫λ` 照样惩罚，模型必须学会把无 note 的线压到 λ≈0（对「一条线吃掉 73%」的天然表达）。**不得**按 note 数裁剪线集合。
2. **不做 `line_id` softmax / 先验重加权**（RFC-0029 §2.4-6/§2.4-8：分配是场竞争的自然结果）。
3. **分层诊断替代重加权**：per-side（正面/背面）、per-channel、per-line 的计数与 NLL 分解 + 归一化线熵 `H/log K`（survey §7.6 思路）作为训练日志与可视化输出。
4. **桶内计数保真**：同格多事件以 `n_j ≥ 2` 全额进入事件项，不因去重而丢失强信号。
5. 背面（2.4–3.0%）与稀有类型（Flick 6–7%）的**召回**由 plan 06 单独报告；本模块只保证它们不被结构性抹掉（例如：禁止把 side 合并、禁止把 5 通道压成 1 通道——这两条各有专门的消融臂，plan 04 §4.9）。

### 4.8 共格碰撞统计（决定 N 的数据）

两个统计量，**都报**：

1. **精确同刻碰撞**（units §7.3 的定义）：同一判定线 + **同一精确判定时刻** + 同侧 的 note 对的最小 `|ΔpositionX|`。报告分布（p0 / p1 / p5 / 中位）与「需要多大 N 才能不共格」（`N ≥ RPE_STAGE_WIDTH / min|Δx|`）。
2. **网格化后的共格率**：按 **τ 格（1/48 拍）**分桶后，落在同一 `(k, i_tau, i_x, s, c)` 格的事件对数与「`n_j ≥ 2` 的格子占比」，对 N ∈ {64, 128, 256, 512} 各报一次（τ 分桶与秒的对应随 BPM 变化，报告须同时给出每谱 BPM 区间）。

**已可预判的结论（用于防止把 128 当新魔数）**：手工谱常见量化网格为 22.5（=1350/60）与 11.25（=1350/120），而任意浮点谱实测最小间隔 0.324639（survey §7.4.1）→ **N = 512（Δx = 2.63671875）仍不足以在这类谱上做到零碰撞**。因此「零碰撞」不是 N 的可行性判据；M7 必须给**碰撞率–N 曲线**并声明可接受阈值，而不是报单点。

### 4.9 可视化 / 调试工具（`beatmorph/field/viz.py`）

- `render_field_png(lam, gt_counts, ...)`：每行一条判定线，列 = {above, below} × 5 通道；横轴时间（秒，不是帧索引）、纵轴 `positionX`（单位 RPE-x，标注 −675/0/+675）；颜色 = λ；GT 事件以散点叠加；`OcclusionMask` 区域以阴影标出；`RangeMask` 外区域置灰。
- `render_collision_report_png(report, ...)`：`|Δx|` 直方图 + 碰撞率–N 曲线。
- 契约：**无权重、无 GPU、脱离训练即可运行**（合成场 + `data/fixtures/` 微型谱面）；matplotlib 用 Agg 后端；输出尺寸与文件名确定性（便于 CI 与人工比对）。日志走 `beatmorph.core.logging.get_logger`，禁止裸 `print`。

---

## 5. 依赖关系

- **上游**：`core/contracts`（`PhigrosChart` / `PhigrosNote` / `JudgeLine` / `ChartField` 与常量；由 contracts-agent 维护，**新增/变更契约须先通报**，AGENTS.md §3.1）；plan 01 的帧率派生常量（`MERT_SAMPLE_RATE_HZ` / `MERT_CONV_STRIDE_PRODUCT`，**音频帧轴用**）；plan 02 的 RPEJSON 解析结果（含 `judgeLineList`、**完整保留的 `BPMList`** 与跨层求和后的事件轨）。
- **下游**：plan 04（消费 `FieldGrid` / `FieldTarget` / `poisson_nll` / 常数基线）；plan 05（消费 `FieldLambda` 网格做 `find_peaks` 与 Ogata thinning）；plan 06（消费 NLL 校准、碰撞统计、分层诊断）；plan 07/08（configs/ 与训练日志编排）。
- **时间换算的单向出口（红线 7 / RFC-0029 §7-8）**：秒 ↔ τ 的换算**只在本模块实现**；plan 02 的格式层 beat↔秒、以及 plan 04/05/06/08 的任何时间轴处理**都不得再写第二套 BPM 分段积分**，一律经本模块暴露的换算接口（接缝归属见 §9-13）。
- **外部库**：`torch>=2.5`（仅张量/自动微分）、`numpy`、`matplotlib`（仅 viz）、`pydantic>=2.5`（契约）。**不引入** scipy（峰值检测属 plan 05；避免在本模块固化解码超参）。

---

## 6. 里程碑与验收

### 6.1 门禁先行（红线 7 / AGENTS.md §4）

本模块引入的**任何新训练目标/损失**（泊松 NLL、分块泊松口径、以及将来任何 auxiliary loss）在**扩大数据规模之前**必须先过 **G1–G4**，统一调用 `beatmorph/infra/sanity.py`（判据取该模块默认值：`overfit_single_batch` / `shuffled_target_control` / `constant_baseline_gate` / `frame_rate_gate`），结果写入训练日志。**门禁未绿不得进入 plan 04 的规模训练**（对应 M9）。

本模块另承担一条**非损失**的契约门禁：**「场网格 ↔ 秒」往返无损（M12）**——它不依赖权重、不依赖 GPU，进默认 CI。

### 6.2 里程碑表（12 条）

| # | 里程碑 | 验收（可量化） |
|---|---|---|
| M1 | ✅ 网格与派生常量契约（含字面量 AST 扫描，进默认 CI） | `dx * x_bins == RPE_STAGE_WIDTH`、`d_tau * BEAT_SUBDIVISION == 1`（拍）、`frame_rate == MERT_SAMPLE_RATE_HZ / MERT_CONV_STRIDE_PRODUCT` 全部断言通过；本模块源码**零**字面量（正则扫描，含 `0.020833…`）；测试进默认 CI（无权重/无 GPU） |
| M2 | ✅ 目标构建（越界/fake/非法 Hold 三项计数齐备，不钳位） | `Σ_j n_j == meta.n_events`（相对误差 0，整数严格相等）；hold-end 与起点同 `(k, i_x, s)` 断言通过；越界/fake/非法 Hold 三项计数与解析器统计**逐张一致**；不钳位（越界 note 的 `positionX` 原值保留在 meta 中） |
| M3 | ✅ 两条积分路径对拍（四类测试场 + 逐格 dV 加权 + 未归一化 p 的反例） | 四类测试场（解析闭式 / 随机分段常量 / 因子化 / **多 BPM 段**）上相对误差 ≤ 1e-6（float64）、≤ 1e-4（float32）；**逐格 `ΔV_j = J_j·d_tau·dx` 的加权断言**（非均匀网格下若漏乘 `J_j` 必须失败）；`Λ` 单调性断言；`p` 归一化断言（含掩码/域外裁剪的**反例测试**必须失败） |
| M4 | ✅ 泊松 NLL 恒等式（四条全绿；λ≡0 → +∞，无 eps 平滑） | (i) `λ ≡ N/\|Ω\|` 的 NLL == `N * (1 + log(\|Ω\|/N))`（相对误差 ≤ 1e-6，`\|Ω\| = Σ_j ΔV_j`）；(ii) `λ ≡ 0` 的 NLL 非有限（断言 + 禁止 eps 平滑的代码审查项）；(iii) `n_j = 2` 贡献 `2·log λ_j`；(iv) `binned - point` == `Σ_j log(n_j!) - Σ_j n_j log ΔV_j`（相对误差 ≤ 1e-6；均匀网格下退化为 `- N log ΔV`） |
| M5 | ✅ 强不平衡 / 线集中诊断（损失无任何重加权，契约测试证明系数恒为 1；实测占比待全库回填） | 输出 per-side、per-channel、per-line 计数与 NLL 分解 + 归一化线熵；契约测试证明损失**不含**任何通道/side/line 权重（对 5 通道 × 2 侧 × K 线的重加权系数恒为 1）；背面与 Flick 的计数占比与实测区间（2.4–3.0% / 6–7%）在同一量级 |
| M6 | 🟡 共格碰撞统计工具（已实现并自测；**全库统计待 plan 02 数据**） | 对全库（或声明的抽样集）产出：精确同刻最小 `\|Δx\|` 分布（p0/p1/p5/中位）、每 N 的碰撞率与 `n_j ≥ 2` 格占比、每谱「不共格所需 N」；结论**先于** N 的选型发布 |
| M7 | 🟡 N 消融接口与碰撞率–N 曲线工具（已实现；NLL/F1/显存**数值待 plan 04/06 回填**，N 取值未裁定） | 每 N 报：碰撞率、NLL（point 与 binned 两栏）、显存与单步耗时、下游事件级 F1@±20ms 与 ±50ms（数值由 plan 06 提供，本模块只固定接口与网格）；产出**碰撞率–N 与指标–N 曲线**；N 的最终取值由本里程碑数据决定并在 plan 中回填（不得默认 128 了事） |
| M8 | ✅ 可视化/调试工具（Agg、无权重无 GPU、尺寸与文件名确定性） | 合成场 + `data/fixtures/` 微型谱面可产出确定的 PNG（尺寸/命名确定性）；无权重、无 GPU、非 slow 即可运行；人工目视可用于判读「到处乱亮」「整条线塌到 0」「背面缺失」三类失败 |
| M9 | ✅ **G1–G4 门禁接入**（最小可微回归任务实跑全绿，标 slow） | 以本模块自带的最小可微回归任务（不依赖 plan 04）跑通 `beatmorph/infra/sanity.py` 四道门禁并全绿；`summarize()` 输出写入训练日志；G3 的基线值取自 M4 的闭式 |
| M10 | ✅ 契约测试进默认 CI（无权重无 GPU 非 slow 全绿；ruff/mypy 见回报） | `tests/unit/field/` 在无权重、无 GPU、非 slow 条件下全绿；`uv run mypy beatmorph/field` strict 零错误；`make lint && make test-fast` 通过 |
| M11 | 🟡 文档化与留痕（模块 docstring 写明 dV/\|Ω\|/L_point 与 L_binned 差、三项计数处置；N 的裁定依据待 M6/M7 数值） | 写明 `\|Ω\|` 与 `ΔV_j` 的定义、`L_point` 与 `L_binned` 的常数差推导、越界/fake/非法 Hold 的处置记录、以及 N 的裁定依据（含被否决的取值与理由） |
| M12 | ✅ **「场网格 ↔ 秒」往返无损契约测试（Q15 硬要求）**（多 BPM 段、段界跳变、改写 BPMList 必变、进默认 CI） | 在**多 BPM 段**谱面（≥ 2 段、含变速）上：`seconds_to_tau(tau_to_seconds(τ)) == τ` 与反向均在**声明容差**内（容差由本计划标定后回填，§9-5）；`tau_to_seconds` 与「逐段 `Δ拍 × 60 / bpm` 求和」的解析值一致；`J(τ)` 段内常量、段界跳变；**改写 `BPMList` 必须使结果变化**（否则实现里藏了硬编码或用的是帧率）；τ 格数与总拍数一致；进**默认 CI**、无权重无 GPU（RFC-0029 §3.1：「否则 beat-aligned 会变成下一个 25 Hz」） |

---

## 7. 风险与缓解

| # | 风险 | BasePlan 编号 | 缓解 |
|---|---|---|---|
| R-4 | **稀疏目标塌陷**（「loss 不降」的同构形态） | R-4 | 泊松 NLL 的积分项惩罚全 0；M4 的 `λ≡0 = +∞` 与 G3 常数基线闭式写成断言；不上朴素 BCE/MSE |
| R-5 | **多线 K 长尾 / note 极度集中** | R-5 | 全 K 条线参与 `∫λ`（空线有成本）；不做 line softmax；分层诊断暴露长尾；N 的消融不涉及线数裁剪 |
| R-6 | 连续场 → 离散事件的解码精度 | R-6 | 本模块只保证**同网格**与秒域口径（plan 05 的双解码臂在此网格上做） |
| R-7 | **物理常量/单位漂移**（25 Hz 类） | R-7 | M1 的派生式 + 字面量扫描 + 默认 CI 契约测试；**M12 的秒↔τ 往返无损断言**；mock/fixture 不得固化物理常量（AGENTS.md §3.3） |
| R-8 | 跨线几何冲突不可玩 | R-8 | 本模块提供 `line_mask` 与位置桶，供 plan 05 做跨线几何检查；**本模块不做任何落点改动**（红线 3） |
| R-03-1 | **N 成为新魔数** | — | M6 先出统计、M7 出曲线；只有数据能决定 N（units §7.3 明确要求） |
| R-03-2 | 共格碰撞使事件项失真（文献 §4.4-4） | — | `n_j log λ_j` 计数形式 + 共格率主指标 + M4(iii) 断言 |
| R-03-3 | 显存/算力：稠密场 `(K,T,N,2,C)` | — | 派生估算：每线每秒 `N·2·C = 1280` 格 × 75 帧 = 96,000 格/秒/线；K=30（中位）→ 2.88e6 格/秒/音频秒（bf16 ≈ 5.8 MB/音频秒）。对策：目标侧稀疏存储（计数几乎全零）、分段窗口训练、必要时按需计算线子集（**但 `∫λ` 仍覆盖全部 K**） |
| R-03-4 | `\|Ω\|` 定义在各处不一致（分母漂移） | — | 单一入口 `FieldGrid.volume`；M4(i) 闭式对拍；契约测试禁止在别处重复实现 |
| R-03-5 | `λ → 0` 数值爆炸 / NaN 传播 | — | softplus 参数化 + 事件项上限断言；NaN 检测作为训练步的硬检查（配合 plan 04） |
| R-03-6 | 越界处置越权（钳位 = 改落点） | 红线 3 | §2 偏离 4：只统计不钳位；域外置 0 属定义域声明而非修改落点 |
| R-03-7 | **beat-aligned 的换算外溢**：下游各自再实现一套秒↔τ ⟹ 下一个 25 Hz 类静默失效 | R-7 | 换算**只在本模块**实现（§5「时间换算的单向出口」）；M12 往返无损契约测试进默认 CI；下游 plan 侧以源码级断言禁止出现第二处 BPM 分段积分 |
| R-03-8 | **测度漏乘 `J(τ)`**：把「拍」当「秒」用，`∫λ` 系统性偏移 | R-7 | M3 增加多 BPM 段测试场与逐格 `ΔV_j` 加权断言；`ΔV_j` 只有 `FieldGrid` 一个出口 |

---

## 8. 测试策略

- **单元（默认 CI，无权重/无 GPU）**
  - `tests/unit/field/test_grid.py`：派生常量、边界、字面量扫描。
  - `tests/unit/field/test_target.py`：计数守恒、hold-end 配对、五通道与两侧、越界/fake/非法 Hold 计数、`above ∈ {0,1,2}` 语义（0 与 2 都进背面）。
  - `tests/unit/field/test_integrate.py`：两条路径一致性门禁（含反例）、`Λ` 单调性、`autograd` 与有限差分互测。
  - `tests/unit/field/test_loss.py`：M4 的四条恒等式；`line_mask` 语义（置 False 的线贡献恰为 0）；`n_j ≥ 2` 语义。
  - `tests/unit/field/test_collision.py`：碰撞统计在同刻/异刻、同侧/异侧、同线/异线三种构造下的判别正确性。
  - `tests/unit/field/test_viz.py`：PNG 可生成、尺寸确定、无 GPU 依赖。
  - `tests/unit/field/test_time_grid.py`：**M12**——秒↔τ 往返无损（多 BPM 段）、`J(τ)` 段内常量与段界跳变、换算只由 `BPMList` 派生（改写 BPMList 结果必变）、与逐段解析积分一致、τ 格数与总拍数一致；无权重无 GPU。
- **契约级**：全部物理量与形状断言进默认 CI；**夹具与 mock 必须引用契约常量**（例如 τ 格数由 `总拍数 × BEAT_SUBDIVISION` 派生、音频帧数由 `duration × MERT_FRAME_RATE_HZ` 派生），不得写死 75/1350/10.546875；**源码级断言：本模块之外不得出现第二份 `60 / bpm` 型分段积分**（由下游 plan 各自落地）。
- **集成**：`tests/integration/field/test_field_pipeline.py`——`data/fixtures/` 微型 RPEJSON → 解析 → 目标 → NLL → 常数基线对比 → PNG，全流程无权重。
- **性质测试**（可选 hypothesis）：随机合法谱面下 `Σ n_j == n_events`、`∫λ ≥ 0`、NLL ≥ `N(1 + log(|Ω|/N))` 的可行性（模型场不满足时该式不成立，故仅对常数场断言）。
- **e2e（标记 `gpu`/`slow`）**：与 plan 04 联合的 G1–G4 门禁运行（M9）。

---

## 9. 开放问题

1. **Λ 因子化的形式**（§2 偏离 1）：`λ = Λ'·p` 是否作为**唯一**参数化，还是保留「自由 λ head + 独立 Λ head」两套并只做一致性校验？后者更灵活但两条路径不再恒等，门禁判别力下降。→ 需 plan 评审 + RFC。
2. **`|Ω|` 的口径**（§2 偏离 2）：全 K 线测度之和 vs 仅有效线；BasePlan/RFC 均未写。本计划取前者（理由：保留对空线的惩罚），但需确认它不与「按谱归一化」的评估口径冲突。**Q15 后的残留子问题**：τ 网格非均匀 ⟹ `|Ω| = Σ_j ΔV_j`（不再等于 `K · T · X · S · C`），λ 的「每格元」单位口径与 RFC-0029 §8.4 R-a 的写法需同步复核；本计划仍按「全 K 线、全部格元」口径实现。
3. **hold-end 的语义**（§2 偏离 3）：独立通道的点 vs `hold` 点的条件标记。本计划取前者；若取后者，`hold_time` 需要额外的回归头，与「不要为分配另设损失」的精神是否一致需裁定。
4. **越界 note 的处置**（§2 偏离 4）：排除（本计划） vs 扩域（`x` 网格覆盖越界值） vs 域外惩罚项。实测 12674 个 note 全部落在 ±675 内，但格式无钳位——**先做全库统计再定**。
5. **一致性容差**：1e-6（float64）/ 1e-4（float32）是本计划设定的工程容差，文档未给；需在实现中按 dtype 与自动微分路径标定后回填。
6. ~~**时间网格是否 beat-aligned**~~ **已决（2026-08-05，RFC-0029 §8.3 Q15）**：采用 **beat-aligned**，基本格 **1/48 拍**，网格随 BPM 变化；数学改写限定在本模块（测度 `dt → J(τ)dτ`，`J` 由 `BPMList` 派生），契约层 `PhigrosNote.t` 仍用秒，下游不得各自换算。本计划的落地见 §3.1 / §4.2 / §4.5 / M12。
7. **N 的最终取值**：须等 M6/M7 数据；`{collision rate, NLL, F1} × N` 三条曲线共同决定，并需回答「N=512 仍碰撞时是否上连续回归/亚格回归」（units §7.4）。
8. ~~**时间桶宽是否等于 MERT 帧率**~~ **已被 Q15 取代**：场的时间轴是 τ（1/48 拍），不再与 75 Hz 音频帧轴绑死；两者在对齐处的重采样方式见 §9-15。残留问题：1/48 拍是否需要「对齐深度」消融（如 1/16 / 1/24 / 1/32 / 1/48）——GOCT 的 8 分音符消融只支撑「对齐优于不对齐」，**不支撑 48 这个具体深度**，未定。
9. **嵌套判定线（`father ≠ -1`，实测 26%）**：局部系下 `positionX` 不受影响，但**线身份/排序**与父线掩码语义需明确；本计划不动，交由 plan 04 的 line embedding 设计 + 数据统计。
10. **是否引入零膨胀（EBC-ZIP 式 ZIP-NLL）**：v1 不引入（保持与 RFC 写法一致），列为 B2 的备选扩展；引入即属新训练目标，须先过 G1–G4 并开 RFC。
11. **`isFake` 是否计入场**：本计划默认排除并统计；若「假音符的视觉编排」被判定为 Phigros 谱面的重要组成，则需重开此条。
12. **`speed`/`size`/`yOffset`/`visibleTime` 等标记**在 v1 被边际化：是否需要独立的「标记头」（v2 议题）——RFC-0029 未涉及。
13. **秒↔τ 换算与格式层的接缝（须裁定）**：本模块是「秒 ↔ τ」的**唯一**实现点（RFC-0029 §7-8），但 plan 02 的解析（beat → 秒）与 plan 05 的写出（秒 → beat 三元组反解）在数学上是**同一分段积分**。三者是否必须共用同一实现（依赖方向 `io → field`？）、还是格式层的 beat↔秒 被判定为独立于场网格的第二换算点，**未裁定** → 须与 contracts-agent / 决策者对齐。在此之前只保留 plan 02 那一处格式层换算，**不得**出现第三处。
   > **临时裁定（2026-09-27，实施期）**：接缝**实测确有分歧**，但只在一个畸形情形下：BPMList 首段起点 > 0 拍时，
   > 本模块把 τ=0 当作谱面时间原点（与契约 `PhigrosNote.t` 自洽）并在前面外推一段，而格式层以首段为原点（与 Phira 官方 `beat2sec` 一致），
   > 两者相差一个常量偏移。首段起于 0 拍时两处在 `tests/integration/test_time_conversion_seam.py` 中逐点一致（多 BPM 段 / 段界 / 往返 / 改写 BPMList 必变）。
   > **处置**：畸形情形由 plan 02 的 `qc.py` 以 schema 违约拒收（隔离区），不静默二选一。
   > 另两个残留子问题已在本轮**量化**而非裁定：① BPM 变更点落在格内时的 `J_j` 取值由 `FieldGrid.cell_seconds(rule="left"|"right"|"exact")` 显式暴露（默认 `left` = 本 plan 口径，§9-16）；
   > ② τ 终点口径仍取 `chart.duration_s()` 且可覆盖，`FieldGrid.for_chart` 对越窗事件计数而不静默丢弃（§9-14）。
14. **`T`（τ 轴长度）的终点口径未查证**：τ 格数需要「谱面终点换算到拍」，但终点取自哪个字段（`chartTime` / 最后一事件 / 音频时长）以及是否含 `META.offset`，本轮**未查证** → 不得凭猜实现；须先查 `phigros-format.md` 并由 plan 02 的 IR 明确给出。

    **⚠️ 2026-09-27 第五轮：从「未查证」升级为「已实测，且是本项目最大的数据侧缺陷」**（RFC-0031 待裁定）。
    全库 8551 张里 **4452 张（52.2%）** 的 `time_span_s / audio_duration_s > 10`（中位 **52.8×**、最大 3.4e6×），
    因为 `PhigrosChart.duration_s() = max(最后一事件, META.chartTime)` 而 `chartTime` 大面积不可信
    （双峰：47% 落在音频 1.1× 内，53% 远超音频）。后果：train split 切出 **3361 万窗**（98% 是空窗、音频整段补零），
    全库索引 2.3 h；G1 抽到 0 事件窗口报假绿，G2 打乱对照被**伪造**成「结构性失效」。
    **已落地默认口径**（`beatmorph/data/dataset.py` 的 `tau_end_policy`，可一行回退）：
    `T = min(谱面口径, 特征缓存元数据的音频时长)`，截断行数与轴外事件**显式记账**；
    修复后 train split = **634 952 窗**、索引 36.5 min，真实切片门禁 **G1-G4 全绿**（plan 07 §9-25）。
    **仍待裁定的是口径本身**（音频时长 / 最后一事件 / 两者取小；是否含 `META.offset`）——见 RFC-0031。
15. **τ 轴与音频帧轴的对齐方式未定**：`audio_emb` 在 75 Hz 帧轴上、场在 τ 格上，条件注入处的重采样（在 τ 格上取音频帧 / 在音频帧上取 τ）尚无结论；本计划只在**一处**实现该重采样（§4.2），具体形式待与 plan 04 联合定稿。
16. **BPM 变更点不落在 1/48 拍格上时的 `J_j` 取值**：RPE 的 `BPMList` 起点是 beat 三元组 `i + n/d`，`d` 不保证整除 48 ⟹ 存在跨格变速。该格取左段、右段还是按格内时长加权，**未定**（直接影响 M3 多 BPM 段测试场的构造与 M12 的容差标定）。

17. **【实施期实测，2026-09-27】全谱 build_target 的显存/内存标度不可用于真实语料**：全谱计数张量 = K x T_full x X x S x C；
    实测 5 分钟谱、K=30、X=128 约 1.1e9 格 ≈ **2 GB / 样本** —— 训练不可用。
    beatmorph/data/dataset.py（plan 02 M11）因此改为「**窗口子谱 + 窗口局部网格**」调用 build_target
    （窗口落在单一 BPM 段内 ⟹ 窗内秒↔τ 线性 ⟹ 与「全谱建表后切窗」**逐格等价**，等价性由
    tests/unit/data/test_dataset.py::test_window_counts_equal_full_chart_slice 锁定）。
    这条**不改变** field/ 的任何语义（装箱仍只经 build_target），但它把「N 的取值」与「单样本内存」绑在了一起：
    RPE_X_GRID_BINS 从 128 提到 256/512 时，全谱口径的内存会再翻 2-4 倍 → N 的消融（§9-7）
    必须同时报告**窗口口径**的内存，否则曲线不可用。
