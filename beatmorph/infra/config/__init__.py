"""训练配置：structured schema（`schema.py`）与严格加载（`loading.py`）。

对外入口：

    from beatmorph.infra.config import TrainConfig, load_config

    cfg = load_config(config_name="smoke", directory=Path("configs"))

缺字段 / 类型错 / 多余字段在**启动期**失败（OmegaConf struct 模式 + validate_config），
`data.provenance` 为空同样是启动期失败（合规硬约束③）。
"""

from beatmorph.infra.config.loading import (
    available_configs,
    config_from_mapping,
    configs_dir,
    dump_config,
    load_config,
)
from beatmorph.infra.config.schema import (
    Backend,
    ConfigError,
    DataConfig,
    DataProvenanceConfig,
    GatesConfig,
    ModelSchema,
    OptimConfig,
    RunConfig,
    RunPurpose,
    TrainConfig,
    assert_valid_config,
    config_to_dict,
    data_scale_is_expanded,
    validate_config,
)

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
    "available_configs",
    "config_from_mapping",
    "config_to_dict",
    "configs_dir",
    "data_scale_is_expanded",
    "dump_config",
    "load_config",
    "validate_config",
]
