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
        tau_end_s: τ 轴终点口径（None = 由谱面自身决定，见 plan 03 §9-14）。
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
    x_bins: int = RPE_X_GRID_BINS
    k_max: int = DEFAULT_K_MAX
    occlusion_ratio: float = 0.5
    max_samples: int | None = None


@dataclass(slots=True)
class OptimConfig:
    """优化器与训练步数。"""

    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip_norm: float = 1.0
    batch_size: int = 1
    max_steps: int = 1000
    val_every: int = 100
    precision: str = "bf16-mixed"
    seed: int = 0


@dataclass(slots=True)
class GatesConfig:
    """门禁阈值（**生效阈值必须落盘**：plan 07 §3.1 / §9-2）。

    Attributes:
        required: 是否强制 fail-closed（扩大规模时必须为 True）。
        smoke_max_samples: 冒烟规模上限；data.max_samples 超过它即视为扩大数据规模。
        overfit_steps / overfit_target_loss / overfit_target_ratio: G1。
        shuffle_steps / shuffle_min_gap_ratio: G2。
        baseline_min_improvement: G3（相对改进比例，**不是**绝对阈值——泊松 NLL 随场体积缩放）。
        frame_rate_tol_frames: G4 容差帧数。
        constant_baseline_normalized: G3 基线是否按事件数归一（§9-3 的归一化口径）。
    """

    required: bool = True
    smoke_max_samples: int = 8
    overfit_steps: int = 300
    overfit_target_loss: float = 0.05
    overfit_target_ratio: float = 0.1
    shuffle_steps: int = 300
    #: G2 的样本数：**必须足够大**，否则打乱臂可以直接背下少量样本、让对照失效
    #: （plan 04 §9-17 的实测结论：样本少时两臂差距会缩到门限以下）。
    shuffle_samples: int = 16
    shuffle_min_gap_ratio: float = 0.05
    baseline_min_improvement: float = 0.1
    frame_rate_tol_frames: int = 2
    constant_baseline_normalized: bool = True
    #: 建门禁模型时把累积强度头的 bias 初始化为该值（G1/G2 用**相对**判据，
    #: 首步必须远离最优才不会退化成「永远通过」；plan 04 §9-17 的实测口径）。
    initial_head_bias: float = 0.0


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
    keep_last: int = 3
    keep_best: int = 1

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

    gates = cfg.gates
    if gates.smoke_max_samples < 1:
        problems.append(f"gates.smoke_max_samples 必须 >= 1，得到 {gates.smoke_max_samples}")
    if gates.overfit_steps < 1 or gates.shuffle_steps < 1:
        problems.append("gates 的过拟合/打乱步数必须 >= 1")
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
        if isinstance(value, (list, tuple)):
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
