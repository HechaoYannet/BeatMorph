"""M7.3：structured config 的启动期失败语义（缺字段 / 类型错 / provenance 必填）。"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml

from beatmorph.data.phira.client import ManifestPurpose, Provenance
from beatmorph.generation.model import ModelConfig
from beatmorph.infra.config.loading import config_from_mapping, dump_config, load_config
from beatmorph.infra.config.schema import (
    ConfigError,
    DataConfig,
    DataProvenanceConfig,
    ModelSchema,
    RunPurpose,
    TrainConfig,
    data_scale_is_expanded,
)

PROVENANCE = {
    "source": "https://api.phira.cn/chart",
    "query": "type=3 pageNum<=30",
    "fetched_at": "2026-09-27T00:00:00+00:00",
    "purpose": "train",
    "script": "scripts/fetch_phira.py",
    "script_version": "v1",
}
BASE = {
    "data": {
        "manifest_path": "data/processed/pairs.json",
        "chart_dir": "data/processed/charts",
        "feature_dir": "data/processed/features",
        "provenance": PROVENANCE,
    },
}


def test_defaults_fill_in_and_config_round_trips(tmp_path: Path) -> None:
    """默认值来自结构化 schema；落盘（config.yaml）后必须能原样读回。"""
    cfg = config_from_mapping(BASE)
    assert isinstance(cfg, TrainConfig)
    assert cfg.optim.betas == (0.9, 0.95)
    assert cfg.run.purpose_kind is RunPurpose.TRAIN
    path = tmp_path / "config.yaml"
    dump_config(cfg, path)
    reloaded = config_from_mapping(yaml.safe_load(path.read_text(encoding="utf-8")))
    assert reloaded == cfg


def test_missing_mandatory_sections_fail_at_startup() -> None:
    """缺 data / 缺 provenance 都是**启动期**失败（不是跑到第 300 步才 KeyError）。"""
    with pytest.raises(ConfigError, match="data"):
        config_from_mapping({})
    without_provenance = {
        "data": {key: value for key, value in BASE["data"].items() if key != "provenance"}
    }
    with pytest.raises(ConfigError, match="provenance"):
        config_from_mapping(without_provenance)


@pytest.mark.parametrize("field", sorted(PROVENANCE))
def test_empty_provenance_field_fails(field: str) -> None:
    """合规硬约束③：来源/用途/脚本/时间的任一项为空 → 启动失败。"""
    payload = {
        "data": dict(BASE["data"], provenance=dict(PROVENANCE, **{field: ""})),
    }
    with pytest.raises(ConfigError) as excinfo:
        config_from_mapping(payload)
    assert field in str(excinfo.value)


def test_invalid_purpose_and_backend_are_rejected() -> None:
    """枚举口径写成字符串以便 yaml 友好，但取值必须受校验。"""
    with pytest.raises(ConfigError, match="purpose"):
        config_from_mapping({**BASE, "run": {"purpose": "production"}})
    with pytest.raises(ConfigError, match="backend"):
        config_from_mapping({**BASE, "run": {"backend": "jax"}})


def test_type_error_and_unknown_key_fail() -> None:
    """类型错与多余字段都在启动期失败（后者最容易被静默吞掉）。"""
    with pytest.raises(ConfigError, match="t_window"):
        config_from_mapping({"data": dict(BASE["data"], t_window="abc")})
    with pytest.raises(ConfigError, match="nope"):
        config_from_mapping({"data": dict(BASE["data"], nope=1)})


def test_synthetic_source_needs_smoke_scale_only() -> None:
    """合成数据不得用于「扩大数据规模」（否则门禁会在合成任务上变绿）。"""
    payload = {"data": dict(BASE["data"], source="synthetic", max_samples=64)}
    with pytest.raises(ConfigError, match="synthetic"):
        config_from_mapping(payload)
    ok = config_from_mapping({"data": dict(BASE["data"], source="synthetic", max_samples=4)})
    assert not data_scale_is_expanded(ok)


def test_scale_expansion_semantics() -> None:
    """扩大规模的判据写在配置里：max_samples=None（全量）或超过冒烟上限。"""
    assert data_scale_is_expanded(config_from_mapping(BASE))
    scaled = config_from_mapping({"data": dict(BASE["data"], max_samples=4)})
    assert not data_scale_is_expanded(scaled)
    big = config_from_mapping({"data": dict(BASE["data"], max_samples=100)})
    assert data_scale_is_expanded(big)


def test_smoke_purpose_cannot_run_at_scale() -> None:
    """purpose=smoke 与「扩大规模」互斥（防止用冒烟配置跑全量再声称是冒烟）。"""
    with pytest.raises(ConfigError, match="smoke"):
        config_from_mapping({**BASE, "run": {"purpose": "smoke"}})


def test_k_max_must_match_between_data_and_model() -> None:
    """判定线容量是同一个量：两处不一致会让 padding 与 embedding 表格错位。"""
    with pytest.raises(ConfigError, match="k_max"):
        config_from_mapping({"data": dict(BASE["data"], k_max=4)})


def test_provenance_schema_matches_data_side() -> None:
    """§9-7 的口径统一：配置侧 provenance 的字段集必须与 plan 02 的 Provenance 一致。"""
    config_fields = {field.name for field in dataclasses.fields(DataProvenanceConfig)}
    data_fields = set(Provenance.model_fields)
    assert config_fields == data_fields
    converted = DataProvenanceConfig(**PROVENANCE).to_manifest_provenance()
    assert isinstance(converted, Provenance)
    assert converted.purpose is ManifestPurpose.TRAIN


def test_model_schema_mirrors_model_config() -> None:
    """镜像必须与 generation.ModelConfig 逐字段同默认值（否则就是第二份事实源）。"""
    schema_fields = {field.name for field in dataclasses.fields(ModelSchema)}
    model_fields = {field.name for field in dataclasses.fields(ModelConfig)}
    assert schema_fields == model_fields
    default = ModelConfig()
    mirror = ModelSchema().to_model_config()
    assert mirror == default


def test_model_schema_rejects_bad_head() -> None:
    """head 是 Literal：dataclass 不做运行期校验，因此由镜像显式拦住。"""
    with pytest.raises(ConfigError, match="head"):
        config_from_mapping({**BASE, "model": {"head": "magic"}})


def test_overrides_apply_after_merge() -> None:
    """点号 override 是 Hydra 风格：合并后生效，且仍然过校验。"""
    cfg = config_from_mapping(BASE, overrides=["optim.lr=0.001", "model.n_layers=3"])
    assert cfg.optim.lr == pytest.approx(0.001)
    assert cfg.model.n_layers == 3
    with pytest.raises(ConfigError):
        config_from_mapping(BASE, overrides=["optim.lr=-1"])


def test_load_config_reports_available_names(tmp_path: Path) -> None:
    """配置名写错时要给出可用清单（而不是一句 FileNotFoundError）。"""
    (tmp_path / "smoke.yaml").write_text(yaml.safe_dump(BASE), encoding="utf-8")
    cfg = load_config(config_name="smoke", directory=tmp_path)
    assert isinstance(cfg.data, DataConfig)
    with pytest.raises(ConfigError, match="smoke"):
        load_config(config_name="nope", directory=tmp_path)


def test_repository_configs_are_valid(tmp_path: Path) -> None:
    """仓库自带的配置必须永远是合法的（否则 README 的快速开始就是错的）。"""
    root = Path(__file__).resolve().parents[3]
    for name in ("smoke", "phigros_masked"):
        cfg = load_config(config_name=name, directory=root / "configs")
        assert cfg.run.experiment


def test_unsupported_precision_is_rejected() -> None:
    """优化精度必须是本训练循环**真的支持**的取值（fp16 需要 GradScaler，故 fail-closed）。"""
    with pytest.raises(ConfigError, match=r"optim\.precision"):
        config_from_mapping(BASE, overrides=["optim.precision=16-mixed"])
    cfg = config_from_mapping(BASE, overrides=["optim.precision=fp32"])
    assert cfg.optim.precision == "fp32"
