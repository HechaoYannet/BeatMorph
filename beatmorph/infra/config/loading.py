"""配置加载：YAML + OmegaConf structured merge + 点号 override（plan 07 §4.2 / M7.3）。

为什么用 structured merge 而不是「yaml -> dict -> dataclass(**dict)」：
前者在**启动期**就把缺字段 / 类型错 / 多余字段变成异常（OmegaConf 的 struct 模式），
后者会把错误推迟到训练中途甚至静默吞掉（多写的字段被 `**kwargs` 丢掉这类事故）。

口径：**先 merge 再校验**——OmegaConf 管「形状」（缺/多/类型），`validate_config` 管「取值」
（范围、provenance 非空、k_max 一致性等），两层都有明确报错信息。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf

from beatmorph.infra.config.schema import (
    ConfigError,
    TrainConfig,
    assert_valid_config,
    config_to_dict,
)

__all__ = [
    "available_configs",
    "config_from_mapping",
    "configs_dir",
    "dump_config",
    "load_config",
    "schema_node",
]


def configs_dir(root: Path) -> Path:
    """仓库的 `configs/` 目录。"""
    return Path(root) / "configs"


def schema_node() -> Any:  # noqa: ANN401 - OmegaConf 节点是动态类型
    """structured schema 的 OmegaConf 节点（缺字段在这里就是 MISSING）。"""
    return OmegaConf.structured(TrainConfig)


def available_configs(directory: Path) -> list[str]:
    """`configs/` 下可用的 `--config-name`（按文件名排序，不含后缀）。"""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(path.stem for path in directory.glob("*.yaml"))


def _merge(payload: Mapping[str, Any], overrides: Sequence[str]) -> Any:  # noqa: ANN401
    try:
        merged = OmegaConf.merge(schema_node(), OmegaConf.create(dict(payload)))
        if overrides:
            merged = OmegaConf.merge(merged, OmegaConf.from_dotlist(list(overrides)))
    except Exception as exc:  # omegaconf 的异常族（MissingMandatoryValue / ValidationError / ...）
        raise ConfigError(f"配置合并失败：{type(exc).__name__}: {exc}") from exc
    return merged


def config_from_mapping(
    payload: Mapping[str, Any],
    *,
    overrides: Sequence[str] = (),
) -> TrainConfig:
    """从纯字典（+ 可选的 `key=value` override）构造并校验配置。

    Raises:
        ConfigError: 形状错误（缺/多/类型）或取值错误（含 provenance 为空）。
    """
    merged = _merge(payload, overrides)
    try:
        cfg = OmegaConf.to_object(merged)
    except Exception as exc:  # dataclass __post_init__ 的校验（如 ModelConfig）
        raise ConfigError(f"配置实例化失败：{type(exc).__name__}: {exc}") from exc
    if not isinstance(cfg, TrainConfig):
        raise ConfigError(f"配置根类型必须是 TrainConfig，得到 {type(cfg).__name__}")
    # OmegaConf 把 Tuple 注解还原成 ListConfig（to_object 后是 list）：归一化回 tuple，
    # 否则「类型标的是 tuple、拿到的是 list」会一路带到下游（不可哈希、比较行为不一致）。
    betas = tuple(float(value) for value in cfg.optim.betas)
    if len(betas) != 2:
        raise ConfigError(f"optim.betas 必须是两个数，得到 {betas}")
    cfg.optim.betas = (betas[0], betas[1])
    assert_valid_config(cfg)
    return cfg


def load_config(
    *,
    config_name: str,
    directory: Path,
    overrides: Sequence[str] = (),
) -> TrainConfig:
    """按 `--config-name` 载入 `<directory>/<name>.yaml` 并套用 override。

    Args:
        config_name: 配置名（不含 `.yaml`）。
        directory: 配置目录（通常是 `<repo>/configs`）。
        overrides: 点号覆盖串，例如 `optim.lr=1e-4` / `data.max_samples=4`。

    Raises:
        ConfigError: 文件不存在 / YAML 解析失败 / 形状或取值不合法。
    """
    path = Path(directory) / f"{config_name}.yaml"
    if not path.is_file():
        names = available_configs(Path(directory))
        raise ConfigError(f"找不到配置 {path}；可用配置：{names}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} 不是合法 YAML：{exc}") from exc
    if not isinstance(payload, Mapping):
        raise ConfigError(f"{path} 的顶层必须是映射，得到 {type(payload).__name__}")
    return config_from_mapping(payload, overrides=overrides)


def dump_config(cfg: TrainConfig, path: Path) -> None:
    """把**解析后**的完整配置落盘（plan 07 §3.2 的六件套之一）。"""
    text = yaml.safe_dump(
        config_to_dict(cfg),
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )
    Path(path).write_text(text, encoding="utf-8")
