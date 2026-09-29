"""训练配置的 structured schema（plan 07 §4.2 / M7.3）。

设计要点（每条都对应一个**会在启动期失败**的判据）：

1. **structured config**：本模块用 frozen dataclass 描述全部字段；缺字段 / 类型错 / 多字段
   在**启动的 1 秒内**失败，而不是跑到第 300 步才 KeyError（加载路径见 `loading.py`）。
2. **`data.provenance` 必填**（合规硬约束③，2026-08-05 裁定）：来源 / 用途 / 脚本版本 /
   获取时间缺一或为空 → 启动即失败。字段名与 plan 02 的 `Provenance` **逐字对应**，
   由测试断言两者不漂移（§9-7 的口径统一）。
3. **架构超参直接复用** `generation.ModelConfig`：不复制第二份，避免两处默认值漂移。
4. **配置里不出现任何派生物理量**：帧率 / 桶宽 / 拍格宽只来自 `beatmorph.core.contracts`
   （红线 7）。因此 §4.2 的「配置中的派生量必须写成表达式并断言」在这里是更强的形态——
   **配置根本没有机会写错它**（写了会被 M7.4 的全仓扫描抓到）。
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, get_args

from beatmorph.core.contracts import RPE_X_GRID_BINS
from beatmorph.generation.model import DEFAULT_K_MAX, HeadMode, ModelConfig

if TYPE_CHECKING:
    from beatmorph.data.phira.client import Provenance

__all__ = [
    "Backend",
    "ConfigError",
    "DataConfig",
    "DataProvenanceConfig",
    "GatesConfig",
    "ModelSchema",
    "OptimConfig",
    "RunConfig",
    "RunPurpose",
    "TrainConfig",
    "assert_valid_config",
    "config_from_dict",
    "config_to_dict",
    "data_scale_is_expanded",
    "validate_config",
]


class ConfigError(ValueError):
    """配置不合法（缺字段 / 类型错 / 取值越界 / provenance 为空）。"""


class RunPurpose(StrEnum):
    """**运行**用途（实验粒度）。

    与 plan 02 的 ManifestPurpose（**数据获取**粒度）是两层：数据清单记「这份数据是为
    什么抓的」，运行配置记「这次实验是什么性质」。两者都落盘，互不替代。
    """

    TRAIN = "train"
    ABLATION = "ablation"
    SMOKE = "smoke"


class Backend(StrEnum):
    """训练循环后端。

    - torch：仓库自带的参考循环（无额外依赖，默认 CI 可跑，用于门禁与冒烟）；
    - lightning：plan 07 §4.1 的目标训练栈（需要 uv sync --extra train）。
    """

    TORCH = "torch"
    LIGHTNING = "lightning"


#: ModelConfig 的默认实例：镜像字段的默认值**从这里取**，不复制字面量
_DEFAULT_MODEL = ModelConfig()


@dataclass(slots=True)
class ModelSchema:
    """generation.ModelConfig 的**配置镜像**（字段名与默认值逐字对齐，由测试断言）。

    为什么不直接用 ModelConfig 当 structured schema：{BT}ModelConfig.head{BT} 的类型是
    {BT}Literal["factorized", "direct"]{BT}，而 OmegaConf 不接受 Literal 注解
    （实测 {BT}ValidationError: Unexpected type annotation{BT}）。镜像 + 一致性测试是
    「既能被 OmegaConf 校验、又不产生第二份默认值」的折中：

    - 默认值全部写成 {BT}_DEFAULT_MODEL.<field>{BT}（改 ModelConfig 会自动传导）；
    - 测试断言两侧字段名集合一致、逐字段默认值一致；
    - {BT}to_model_config(){BT} 在启动期实例化真 ModelConfig，{BT}__post_init__{BT} 的校验照常生效。
    """

    d_model: int = _DEFAULT_MODEL.d_model
    n_heads: int = _DEFAULT_MODEL.n_heads
    n_layers: int = _DEFAULT_MODEL.n_layers
    window: int = _DEFAULT_MODEL.window
    global_period: int = _DEFAULT_MODEL.global_period
    k_max: int = _DEFAULT_MODEL.k_max
    dropout: float = _DEFAULT_MODEL.dropout
    head: str = str(_DEFAULT_MODEL.head)
    audio_dim: int = _DEFAULT_MODEL.audio_dim
    extended_dim: int = _DEFAULT_MODEL.extended_dim
    position_max_period: float = _DEFAULT_MODEL.position_max_period
    check_lambda: bool = _DEFAULT_MODEL.check_lambda
    seconds_position: bool = _DEFAULT_MODEL.seconds_position
    audio_align: bool = _DEFAULT_MODEL.audio_align
    head_skip: bool = _DEFAULT_MODEL.head_skip
    visible_input: bool = _DEFAULT_MODEL.visible_input
    track_input: bool = _DEFAULT_MODEL.track_input

    def to_model_config(self) -> ModelConfig:
        """构造真正的 ModelConfig（其 __post_init__ 做架构级校验）。

        Raises:
            ConfigError: 取值不合法（如 head 不在 Literal 内、d_model 不被 n_heads 整除）。
        """
        allowed_heads = set(get_args(HeadMode))
        if self.head not in allowed_heads:
            raise ConfigError(f"model.head={self.head!r} 不在 {sorted(allowed_heads)} 中")
        try:
            return ModelConfig(**dataclasses.asdict(self))
        except ValueError as exc:
            raise ConfigError(f"model 超参不合法：{exc}") from exc


@dataclass(slots=True)
class DataProvenanceConfig:
    """数据来源与用途（**必填**，合规硬约束③）。

    字段与 beatmorph.data.phira.client.Provenance 一一对应（由测试断言），
    因此训练配置的 provenance 可以**原样**转成数据清单的 provenance（§9-7 的统一口径）。

    Attributes:
        source: 数据源（API 端点 / 清单快照 / 夹具路径）。
        query: 查询条件（分页范围 / 过滤条件的文本化）。
        fetched_at: ISO8601 UTC 获取时间。
        purpose: 数据获取用途（train / eval / stats）。
        script: 获取脚本标识。
        script_version: 获取脚本版本。
        chart_id_min / chart_id_max: 枚举范围（可选，便于追溯）。
    """

    source: str
    query: str
    fetched_at: str
    purpose: str
    script: str
    script_version: str
    chart_id_min: int | None = None
    chart_id_max: int | None = None

    def to_manifest_provenance(self) -> Provenance:
        """转成数据侧的 Provenance（唯一的 schema 权威）。

        Raises:
            ConfigError: 字段不合法（由 pydantic 校验给出具体原因）。
        """
        from beatmorph.data.phira.client import Provenance

        try:
            return Provenance(**dataclasses.asdict(self))
        except ValueError as exc:  # pydantic ValidationError 是 ValueError 子类
            raise ConfigError(f"data.provenance 不合法：{exc}") from exc


@dataclass(slots=True)
class DataConfig:
    """数据集与规模。

    max_samples 是 **fail-closed 的判据**（plan 07 §3.2）：超过 gates.smoke_max_samples
    即视为「扩大数据规模」，此时必须先有全绿的 gates.txt。把判据写进配置而不是靠记性——
    口头约定不是门禁。

    Attributes:
        manifest_path: 配对清单（PairSplits.to_dict() 的 JSON）。
        chart_dir: 谱面根目录。
        feature_dir: 特征缓存根目录（<key>.npz + <key>.meta.json）。
        provenance: 数据来源与用途（**必填**）。
        split_train / split_val: 训练 / 验证切分名。
        t_window: τ 轴窗口长度（格）。
        tau_end_s: τ 轴终点的**显式**覆盖（秒）；None = 由 `tau_end_policy` 决定。
        tau_end_policy: τ 轴终点口径（plan 03 §9-14 未裁定；`beatmorph/data/dataset.py`
            的模块 docstring 有实测依据）：`"audio"`（默认）= min(谱面口径, 音频时长)，
            用于消除真实语料里大面积不可信的 `META.chartTime`；`"chart"` = 旧行为。
        chart_cache_size / feature_cache_size: 取批时的行级 LRU 容量（0 = 关闭）。
        x_bins: x 轴桶数（默认取契约值；消融见 RPE_X_GRID_BIN_SWEEP）。
        k_max: 判定线容量（与模型 k_max 一致）。
        occlusion_ratio: 训练遮盖比例 r。
        max_samples: 参与训练的最大样本数（None = 全量 ⇒ 视为扩大规模）。
    """

    provenance: DataProvenanceConfig
    #: 三个路径在 source=synthetic 时用不到，因此给空串默认值、由 validate_config
    #: 在 source=manifest 时强制非空（「缺字段」的失败信息比 MissingMandatoryValue 更具体）。
    manifest_path: str = ""
    chart_dir: str = ""
    feature_dir: str = ""
    #: 数据来源：manifest（真实清单+特征缓存）/ synthetic（infra.smoke 合成，只允许冒烟规模）。
    #: 显式声明而不是「文件不存在就自动降级」——静默降级会让门禁在全量数据上跑合成任务。
    source: str = "manifest"
    split_train: str = "train"
    split_val: str = "val"
    t_window: int = 192
    tau_end_s: float | None = None
    tau_end_policy: str = "audio"
    chart_cache_size: int = 8
    feature_cache_size: int = 2
    #: 窗口预切缓存的根目录（`beatmorph-build-windows` 的产物）。None = 关闭（走原解析路径）。
    #:
    #: 开启后每窗口从 **0.85 s → 9.6 ms**（实测，见 plan 07 §9-52），且**样本逐位不变**；
    #: 指纹不符 / 目录缺失一律回退到原路径，绝不静默降级。
    #: **语义无关字段**：它只改「窗口怎么被读出来」，不改顺序、不改覆盖率、不改任何数值。
    window_cache_dir: str | None = None
    #: 取样本的 DataLoader worker 进程数（0 = 主进程同步取批）。
    #: **语义无关字段**（RFC-0034 §5）：顺序与覆盖率都是「计划层」的纯函数，与 workers
    #: 取值无关 ⇒ 它不参与续训指纹（见 `checkpoint.RESUME_IGNORED_KEYS`），
    #: 改它不会让已有 checkpoint 失效，也不会改变样本序列。
    workers: int = 0
    x_bins: int = RPE_X_GRID_BINS
    k_max: int = DEFAULT_K_MAX
    occlusion_ratio: float = 0.5
    max_samples: int | None = None


@dataclass(slots=True)
class OptimConfig:
    """优化器与训练步数（含 val 的节奏与规模）。

    **val 的成本算式（plan 07 §9-47 E；实测算式见 §9-51）**

    单窗口 ≈ 数据 0.038 s（worker 路径；workers=0 时 0.85 s，贵 20 倍）
    + 纯前向 ~0.05 s ≈ **0.09 s**。约束：

        val_windows × 0.09 × 前向臂数 ≤ 5% × val_every × 0.199

    其中 0.199 s 是当前真实训练的步时（§9-45）。**前向臂数**是关键：§9-47 D 要求
    对照臂（遮盖内置换 1 次 + 条件干预三元组 3 次）在**同一批**上各跑一次前向
    ⇒ 生产默认是 5 个臂，不是 1 个。

        val_every=1000、val_windows=128：单臂 11.5 s；5 臂 ≈ 37 s / 199 s ≈ **18.6%**

    ⚠️ **上式仍然偏乐观，落地后实测/推导校正见 plan 07 §9-51 ③**：

    - 数据侧单窗口**实测** 0.12-0.28 s（不是 0.038 s）：val 前缀的 128 个槽位落在
      93 张谱、**128 个不同网格桶**上 ⇒ 每个窗口都是一次冷解析，且各自成桶 ⇒ B=1；
    - val 前缀的 K 分布比训练流重（p90 **106** vs 53-67），而前向 ∝ K²；
    - ⇒ 按 K² 律**推导**一次 val ≈ **143-293 s**（5 臂）≈ 训练预算（199 s/1000 步）的
      **0.7-1.5×**，不是 5.8%。

    ⇒ 本字段只声明**窗口数**，节奏由 val_every 给；要压回 5% 量级需要决策者裁定
    （`val_windows≈32` / `val_every≈4000` / 对照臂降频，见 §9-51 ③ 与存疑清单）。
    """

    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip_norm: float = 1.0
    #: **显存卫生阈值（GiB，plan 07 §9-57 事故后新增）**：当 PyTorch 缓存分配器的
    #: `reserved − allocated` 超过该值时调用 `torch.cuda.empty_cache()`，把**保留但未使用**
    #: 的块还给驱动。0 = 关闭。
    #: 为什么需要：一次 K≈128 的大批会把分配器的峰值保留量顶到 ~5.2 GiB 并**长期不释放**，
    #: 驱动侧总量因此停在 ~7.86 / 8.15 GiB，下一次大分配无处可放 ⇒ Windows **静默回退共享
    #: 显存**（系统内存），功耗从 ~102 W 塌到 ~31 W、步时放大一个数量级以上（2026-09-28 实测：
    #: step 951 直接卡死；用户观测到共享显存 13.2 GB）。默认 1.0 GiB 只在「保留量远超真实
    #: 需求」时触发，正常步不付出 `empty_cache` 的重分配代价。
    vram_hygiene_gib: float = 1.0
    batch_size: int = 1
    max_steps: int = 1000
    #: val 的周期（步）；0 = 不跑 val（此前的默认状态：`val_every` 存在但**空转**）。
    #: ⚠️ 它**不参与续训指纹**（见 `checkpoint.RESUME_IGNORED_KEYS`）：val 不产生梯度、
    #: 不参与 LR 与早停（§9-47 G-④）⇒ 改它不改变任何被训练的东西，只改变「多久看一次」。
    val_every: int = 100
    #: 验证集窗口数；每 `val_every` 步**原样重放同一批**（同窗口 + 同遮盖种子 ⇒ 逐位一致，
    #: §9-47 A1；跨步可比只差权重）。
    #:
    #: **抽取口径（RFC-0039 R2，2026-09-29 采纳）**：不再是「每桶前 N 个」——那个口径只覆盖
    #: 计划里最早出现的十几个桶，实测 **57.8% 空窗、事件/窗中位 0**，比总体与训练流系统性
    #: 容易（§9-62 ⑪㉒）。现在由 `ManifestValSource.selection()` 做**跨桶 + 按事件密度分层**的
    #: 确定性抽样（层 0 = 空窗；层 1..4 = 非空窗口的等量分位层；配额按层总体比例分配）；
    #: 构成落盘到 `logs/val_composition.json` 并打进日志（空窗占比 / 事件数 / K 分位 / 集合指纹）。
    #: ⚠️ **分层需要窗口预切缓存**（事件数只能从缓存的稀疏计数廉价读出）：没有缓存时
    #: **回退**到旧口径并如实记 `stratified=false`，不假装做过分层。
    #: ⚠️ **换 val 集合 = 换总体**：`val/ratio` 的历史数值与新集合**不可比**（RFC-0039 §4）。
    #: 本字段不进续训指纹（它只决定「看哪些窗口」，不决定训练语义）。
    #:
    #: 代码默认值**故意保持 R2 之前的 128**（同 R1 的理由：转正落在生产配置，
    #: 免得 smoke 等配置的总成本被静默改变）；`configs/phigros_masked.yaml` 取 512。
    val_windows: int = 128
    #: 每个验证批最多放几个窗口（**同一网格桶**内的窗口才可同批）。
    #:
    #: RFC-0039 R2 之后它同时是「分层抽样不牺牲组批」的旋钮：分层只改**取哪些桶/哪些窗口**，
    #: 批内仍强制同网格身份（`_stratified_chunks`），否则前向成本会按 K² 律炸开。
    #:
    #: 为什么需要它（§9-53，实测）：计划是跨桶轮转发牌的，取「前 N 个槽位」会让 N 个窗口落在
    #: **N 个不同桶** ⇒ 每个窗口各自成 batch（B=1）× N 次前向。而前向成本 ∝ K²
    #: （global 层对 L=K·T 做全自注意力）⇒ B=1 时是 Σf(K_i)，B=8 时是 Σf(max K_i)。
    #: 实测 val 前缀 **ΣK² = 422 139**（K 中位 30.5 / p90 106）⇒ 同桶成组后**同样的窗口数**、
    #: 前向成本约降 `val_batch` 倍，而**指标语义不变**（`reduction="mean"` 按有效线加权，
    #: 填充线的 `line_mask=False` 被排除；`ValAccumulator` 亦按有效线聚合）。
    #: 不改梯度、不进续训指纹（同 `val_windows`）。
    val_batch: int = 8
    #: 空间 softmax 的**熵正则权重**（0 = 关闭）。惩罚项与事件项同结构：
    #: `w · (1/r) · Σ_被遮盖 n · H(p_token) / max(E_total,1)`（H 为 token 内 (x,s,c) 分布的熵）
    #: ⇒ `w` 与 `−log λ` 同量纲。
    #: 实测依据（`runs/_diag_flat.py`，step-20000）：把模型自己的输出在 token 内摊平后
    #: `val_ratio` **0.8867 → 0.7351**（只摊平 x 轴 → **0.6697**）⇒ 它的空间分布比均匀还差，
    #: 代价 0.19-0.28 nats/line，比音频/LR/对齐等任何已测效应大两个数量级。
    cell_entropy_weight: float = 0.0
    precision: str = "bf16-mixed"
    seed: int = 0


@dataclass(slots=True)
class GatesConfig:
    """门禁阈值（**生效阈值必须落盘**：plan 07 §3.1 / §9-2）。

    Attributes:
        required: 是否强制 fail-closed（扩大规模时必须为 True）。
        smoke_max_samples: 冒烟规模上限；data.max_samples 超过它即视为扩大数据规模。
        overfit_steps / overfit_target_loss / overfit_target_ratio: G1。
        baseline_steps / baseline_samples / baseline_chunks: G3（RFC-0037 起 G2 已删除，
            原 shuffle_* 字段更名归 G3 独用，shuffle_min_gap_ratio 随 G2 一并删除）。
        batch_min_events: 门禁批的最小事件数（G1/G3 的空批会让判据空过）。
        baseline_min_improvement: G3（相对改进比例，**不是**绝对阈值——泊松 NLL 随场体积缩放）。
        frame_rate_tol_frames: G4 容差帧数。
        constant_baseline_normalized: G3 基线是否按事件数归一（§9-3 的归一化口径）。
    """

    required: bool = True
    smoke_max_samples: int = 8
    #: 门禁批的**数据规模上限（行数）**；None = 不限界（沿用 data.max_samples）。
    #: 为什么必须有界（2026-09-27 第八轮实测）：门禁要跑 300 + 2x100 + 100 步，而单步成本
    #: ∝ (K·T)²。采样器修好之后（RFC-0033）门禁批从**全库**按剩余窗口加权抽桶，抽中的是
    #: 最大的那批桶（K 可到 k_max=128）——门禁从 **5 min 涨到 26 min，且显存贴到 7880/8151 MiB**，
    #: 正好落在 docs/TRAINING.md §7.5 记录的「滑进 Windows 共享内存」危险区。
    #: 固定成有界切片后门禁既便宜又可复现，且与第五轮权威记录（data.max_samples=200）同口径。
    gate_samples: int | None = 200
    overfit_steps: int = 300
    overfit_target_loss: float = 0.05
    overfit_target_ratio: float = 0.1
    #: G3 常数基线臂的优化步数（原与 G2 共享 `shuffle_steps`；RFC-0037 删 G2 后独立命名）。
    baseline_steps: int = 300
    #: G3 批的样本数：**必须足够大**，否则预算内可以把小批背下来、让基线对照失效
    #: （plan 04 §9-17 的实测结论）。
    baseline_samples: int = 16
    #: G3 前向的**分段数**（plan 07 §9-15 的内存墙）：样本数足够大与内存够用
    #: 是一对矛盾，分段让二者同时成立——内存回到「一段样本」的量级，而经分段修正的
    #: per_event 损失下梯度与标量 loss 与不分段**一致**（见 `infra.train_loop.make_step_fn`）。
    #: 1 = 不分段（旧行为，真实数据上会 OOM）。
    baseline_chunks: int = 4
    #: 门禁批的**最小事件数**（默认 1 = 非空）。判据在空批上会空过：真实数据实测 G1 批
    #: K=2 / 0 事件 ⇒ 报了「loss 7819.5 → 0.0014」却没有任何事件可过拟合（plan 07 §9-23）。
    #: 取不到就**重抽**，重抽上限内仍取不到则抛（fail-closed，不静默降级）。
    batch_min_events: int = 1
    baseline_min_improvement: float = 0.1
    frame_rate_tol_frames: int = 2
    constant_baseline_normalized: bool = True
    #: 建门禁模型时把累积强度头的 bias 初始化为该值（G1 用**相对**判据，
    #: 首步必须远离最优才不会退化成「永远通过」；plan 04 §9-17 的实测口径）。
    initial_head_bias: float = 0.0
    #: G3（基线对照臂）的初始累积强度偏置。**默认 0 = 模型的自然初始化**。
    #: `initial_head_bias` 是**为 G1 的相对判据服务的**（首步必须远离最优，否则
    #: 「末步 ≤ 0.1 x 首步」会退化成永远通过）。把同一个人为初值强加给 G3 会让它
    #: 先花一半预算把与任务无关的积分项压回去：真实数据实测（2026-09-27 第四轮）
    #: bias=20 时 G3 在 100 步内比常数基线差 68 倍；对照臂用自然初始化后
    #: G3 在 300 步内收敛到 0.37 x 基线。**这不是放宽判据**，而是去掉一个属于 G1 的人为初值。
    contrast_initial_head_bias: float = 0.0
    #: 门禁优化器的学习率；None = 沿用 `optim.lr`（旧行为，向后兼容）。
    #: ⚠️ 门禁的优化预算（steps x lr）必须大到让模型**真的收敛**，否则 G3 的
    #: 「优于常数基线」会退化成恒 FAIL——因为臂还停在初始点附近（初始化 bias 由
    #: `initial_head_bias` 抬高，需要足够的优化量才能落回最优尺度）。合成夹具在
    #: `configs/smoke.yaml` 里用 0.05 校准（其注释记录 bias=20 / 16 样本 / 100 步的取值）；
    #: 真实配置的训练 lr 是 3e-4，直接沿用会让门禁模型几乎不动。
    gate_optimizer_lr: float | None = None


@dataclass(slots=True)
class RunConfig:
    """实验产物与运行形态。"""

    experiment: str = "masked_field"
    #: 用途与后端写成**字符串**：OmegaConf 的 EnumNode 只认枚举**名**（SMOKE）而不认取值
    #: （smoke），yaml 里写 smoke 会直接报错——那是个会让人写错配置的接口。
    #: 取值合法性在 validate_config 里校验，解析入口是 purpose_kind / backend_kind。
    purpose: str = RunPurpose.TRAIN.value
    runs_dir: str = "runs"
    backend: str = Backend.TORCH.value
    log_level: str = "INFO"
    save_every: int = 500
    #: 过程标量的刷新间隔（步）：每这么多步把 train/* 与 sys/peak_vram_gib 刷进 TB 与
    #: `logs/loss_history.jsonl`。**长跑必须能在线看到进度**——此前只在训练结束时写一次，
    #: 人力监控在整轮训练期间看不到任何曲线（plan 07 §4.6 / docs/TRAINING.md §7.5）。
    log_every: int = 50
    keep_last: int = 3
    keep_best: int = 1
    #: **端到端产物的周期（步）**；0 = 关闭。RFC-0039 R3：每 N 步生成一张谱面到
    #: `<e2e_dir>/outputs/<时间>-step<N>/` 供人工审阅。与 `save_every` / `optim.val_every`
    #: **解耦**（三个周期各自独立），且**不进续训指纹**（它不改变任何被训练的东西）。
    #:
    #: **它不是门禁**：不产生 PASS/FAIL、不阻塞训练、不参与选权重；失败只告警并继续。
    e2e_every: int = 20000
    #: e2e 资产目录（内含 `meta.json` 指针清单 + `audio/` + `feature/` + `outputs/`）。
    #: 相对路径按进程 CWD 解析（从仓库根启动）；资产缺失 ⇒ 启动时告警并把 e2e 关掉，
    #: 而不是等到 20 000 步才第一次报错。
    e2e_dir: str = "tests/e2e-val"

    @property
    def purpose_kind(self) -> RunPurpose:
        """用途枚举（非法值在 validate_config 里已被拦下）。"""
        return RunPurpose(self.purpose)

    @property
    def backend_kind(self) -> Backend:
        """训练后端枚举（同上）。"""
        return Backend(self.backend)


@dataclass(slots=True)
class TrainConfig:
    """完整训练配置（--config-name 的解析见 loading.py）。"""

    data: DataConfig
    #: 架构超参：ModelSchema 是 generation.ModelConfig 的镜像（默认值自动传导）
    model: ModelSchema = field(default_factory=ModelSchema)
    optim: OptimConfig = field(default_factory=OptimConfig)
    gates: GatesConfig = field(default_factory=GatesConfig)
    run: RunConfig = field(default_factory=RunConfig)


def data_scale_is_expanded(cfg: TrainConfig) -> bool:
    """是否「扩大数据规模」（fail-closed 的触发条件）。

    max_samples is None（= 全量）或超过 gates.smoke_max_samples → True。
    """
    if cfg.data.max_samples is None:
        return True
    return cfg.data.max_samples > cfg.gates.smoke_max_samples


def validate_config(cfg: TrainConfig) -> list[str]:  # noqa: PLR0912, PLR0915 - 一次报出全部问题
    """逐项校验并**一次报出全部问题**（不是首个失败就返回）。"""
    problems: list[str] = []
    data = cfg.data
    provenance = data.provenance
    required_text = {
        "source": provenance.source,
        "query": provenance.query,
        "fetched_at": provenance.fetched_at,
        "purpose": provenance.purpose,
        "script": provenance.script,
        "script_version": provenance.script_version,
    }
    empty = sorted(name for name, value in required_text.items() if not str(value).strip())
    if empty:
        problems.append(
            f"data.provenance 的 {empty} 为空：来源与用途必填（合规硬约束③ / plan 07 §3.2）",
        )
    from beatmorph.data.phira.client import ManifestPurpose

    allowed_purpose = {member.value for member in ManifestPurpose}
    if provenance.purpose and provenance.purpose not in allowed_purpose:
        problems.append(
            f"data.provenance.purpose={provenance.purpose!r} 不在 {sorted(allowed_purpose)} 中",
        )
    if data.source == "manifest":
        for name in ("manifest_path", "chart_dir", "feature_dir"):
            if not str(getattr(data, name)).strip():
                problems.append(f"data.{name} 不得为空（data.source=manifest 时必须给出路径）")
    if data.tau_end_policy not in {"audio", "chart"}:
        problems.append(
            f"data.tau_end_policy={data.tau_end_policy!r} 不在 ['audio', 'chart'] 中"
            "（见 beatmorph/data/dataset.py 的 τ 轴终点口径说明）",
        )
    if data.chart_cache_size < 0 or data.feature_cache_size < 0:
        problems.append("data.chart_cache_size / feature_cache_size 必须 >= 0")
    if data.workers < 0:
        problems.append(f"data.workers 必须 >= 0，得到 {data.workers}")
    if data.t_window < 1:
        problems.append(f"data.t_window 必须 >= 1，得到 {data.t_window}")
    if data.x_bins < 1:
        problems.append(f"data.x_bins 必须 >= 1，得到 {data.x_bins}")
    if data.k_max < 1:
        problems.append(f"data.k_max 必须 >= 1，得到 {data.k_max}")
    if not 0.0 <= data.occlusion_ratio < 1.0:
        problems.append(f"data.occlusion_ratio 必须落在 [0, 1)，得到 {data.occlusion_ratio}")
    if data.max_samples is not None and data.max_samples < 1:
        problems.append(f"data.max_samples 必须 >= 1 或为 null，得到 {data.max_samples}")
    allowed_sources = {"manifest", "synthetic"}
    if data.source not in allowed_sources:
        problems.append(f"data.source={data.source!r} 不在 {sorted(allowed_sources)} 中")
    if data.source == "synthetic" and data_scale_is_expanded(cfg):
        problems.append(
            "data.source=synthetic 却处于「扩大数据规模」区间：合成数据只能用于冒烟/门禁，"
            "不得用来冒充全量训练（把 max_samples 压到冒烟规模以内）",
        )

    optim = cfg.optim
    #: 支持的精度取值（Lightning 的命名习惯）：fp32 家族不启用 autocast，bf16 家族在 CUDA 上启用。
    allowed_precision = {"32", "32-true", "fp32", "bf16", "bf16-mixed"}
    if str(optim.precision) not in allowed_precision:
        problems.append(
            f"optim.precision={optim.precision!r} 不在 {sorted(allowed_precision)} 中"
            "（fp16 需要 GradScaler，本训练循环不支持：静默按 fp32 跑比拒掉更危险）",
        )
    if optim.lr <= 0.0:
        problems.append(f"optim.lr 必须为正，得到 {optim.lr}")
    if optim.batch_size < 1:
        problems.append(f"optim.batch_size 必须 >= 1，得到 {optim.batch_size}")
    if optim.max_steps < 1:
        problems.append(f"optim.max_steps 必须 >= 1，得到 {optim.max_steps}")
    if len(optim.betas) != 2 or not all(0.0 <= beta < 1.0 for beta in optim.betas):
        problems.append(f"optim.betas 必须是两个 [0,1) 内的数，得到 {optim.betas}")
    if optim.val_every < 0:
        problems.append(f"optim.val_every 必须 >= 0，得到 {optim.val_every}")
    if optim.val_windows < 1:
        problems.append(f"optim.val_windows 必须 >= 1，得到 {optim.val_windows}")
    if optim.val_batch < 1:
        problems.append(f"optim.val_batch 必须 >= 1，得到 {optim.val_batch}")
    if optim.cell_entropy_weight < 0.0:
        problems.append(
            f"optim.cell_entropy_weight 必须 >= 0（0 = 关闭），得到 {optim.cell_entropy_weight}"
        )

    run = cfg.run
    if run.log_every < 1:
        problems.append(f"run.log_every 必须 >= 1，得到 {run.log_every}")
    if run.save_every < 0:
        problems.append(f"run.save_every 必须 >= 0（0 = 不存盘），得到 {run.save_every}")
    if run.keep_last < 0 or run.keep_best < 0:
        problems.append(
            f"run.keep_last / keep_best 必须 >= 0，得到 {run.keep_last} / {run.keep_best}"
        )
    if run.e2e_every < 0:
        problems.append(f"run.e2e_every 必须 >= 0（0 = 关闭端到端产物），得到 {run.e2e_every}")

    gates = cfg.gates
    if gates.smoke_max_samples < 1:
        problems.append(f"gates.smoke_max_samples 必须 >= 1，得到 {gates.smoke_max_samples}")
    if gates.gate_samples is not None and gates.gate_samples < 1:
        problems.append(f"gates.gate_samples 必须 >= 1 或为 null，得到 {gates.gate_samples}")
    if gates.overfit_steps < 1 or gates.baseline_steps < 1:
        problems.append("gates 的过拟合/基线步数必须 >= 1")
    if gates.frame_rate_tol_frames < 0:
        problems.append(
            f"gates.frame_rate_tol_frames 必须 >= 0，得到 {gates.frame_rate_tol_frames}"
        )
    allowed_runs = {member.value for member in RunPurpose}
    if cfg.run.purpose not in allowed_runs:
        problems.append(f"run.purpose={cfg.run.purpose!r} 不在 {sorted(allowed_runs)} 中")
    allowed_backends = {member.value for member in Backend}
    if cfg.run.backend not in allowed_backends:
        problems.append(f"run.backend={cfg.run.backend!r} 不在 {sorted(allowed_backends)} 中")
    if cfg.run.purpose == RunPurpose.SMOKE.value and data_scale_is_expanded(cfg):
        problems.append(
            "run.purpose=smoke 但 data.max_samples 超过冒烟规模：冒烟运行不得在全量数据上启动"
            "（要么调小 max_samples，要么把 purpose 改成 train/ablation）",
        )
    try:
        cfg.model.to_model_config()
    except ConfigError as exc:
        problems.append(str(exc))
    if data.k_max != cfg.model.k_max:
        problems.append(
            f"data.k_max={data.k_max} 与 model.k_max={cfg.model.k_max} 必须一致："
            "判定线容量是同一个量，两处不一致会让 padding 与 embedding 表格错位",
        )
    return problems


def assert_valid_config(cfg: TrainConfig) -> None:
    """校验失败即抛 ConfigError（消息含**全部**问题）。"""
    problems = validate_config(cfg)
    if problems:
        raise ConfigError("配置不合法：\n  - " + "\n  - ".join(problems))


def config_to_dict(cfg: TrainConfig) -> dict[str, Any]:
    """转成可 YAML/JSON 落盘的纯字典（枚举转字符串、元组转列表）。"""

    def convert(value: Any) -> Any:  # noqa: ANN401 - 递归转换任意嵌套值
        if isinstance(value, StrEnum):
            return str(value)
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, list | tuple):
            return [convert(item) for item in value]
        return value

    raw = dataclasses.asdict(cfg)
    converted: dict[str, Any] = convert(raw)
    return converted


def config_from_dict(payload: dict[str, Any]) -> TrainConfig:
    """从纯字典重建配置（**严格**：未知键与缺字段都会报错）。

    Raises:
        ConfigError: 出现未知键，或必需字段缺失，或类型不匹配。
    """
    from beatmorph.infra.config.loading import config_from_mapping

    return config_from_mapping(payload)
