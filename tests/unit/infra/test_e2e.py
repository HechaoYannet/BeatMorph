"""RFC-0039 R3：端到端生成产物的**接线与纪律**（默认 CI，无权重 / 无 GPU / 无真实语料）。

本文件钉死五件事：

1. **清单解析**：`meta.json` 的必需/可选字段、`feature_meta` 的显式指路与默认名、缺文件报错；
2. **条件来源**：显式 `chart` > 按音频 sha1 在训练清单里找回该曲 > **合成模板**（无条件生成，
   如实记为 `synthetic`）；清单不可读只降级、不崩；
3. **τ 轴终点与训练同口径**（显式覆盖 > policy；`audio` 取谱面与音频的较小者）；
4. **端到端真跑一遍**（tiny 模型 + CPU + 合成特征）：逐窗解码 → 整谱后处理 → 产物落盘；
5. **红线 6**：不合法（`violations` 非空）时**拒绝写出 `chart.json`**，但 `meta.json` 必须写
   ——否则失败现场没有任何证据。
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from beatmorph.core.contracts import ChartSource, JudgeLine, PhigrosNote, side_from_above
from beatmorph.core.contracts.phigros import NoteType
from beatmorph.data.pipeline.embed import FeatureCacheMeta, save_feature_cache
from beatmorph.decoder import check_chart
from beatmorph.field.grid import FieldGrid
from beatmorph.generation.model import MaskedFieldModel, ModelConfig
from beatmorph.infra.config.loading import config_from_mapping
from beatmorph.infra.e2e import (
    DEFAULT_E2E_DIR,
    OUTPUT_DIRNAME,
    E2EInputError,
    E2EInputs,
    E2EResult,
    axis_end_seconds,
    find_corpus_row,
    load_e2e_inputs,
    resolve_template,
    run_e2e,
    sha1_file,
    write_artifact,
)
from tests.unit.field._builders import make_bpm_points as field_bpm
from tests.unit.field._builders import make_chart, make_note
from tests.unit.generation._builders import make_batch, make_counts, make_grid

PROVENANCE = {
    "source": "fixtures",
    "query": "n/a",
    "fetched_at": "t",
    "purpose": "train",
    "script": "s",
    "script_version": "v1",
}

#: 合成特征：8 秒 @ 75 Hz。帧率与特征维都是**契约派生值**，夹具不得自己写另一份。
FRAMES: int = 600
DURATION_S: float = 8.0
#: MERT 的特征维（FeatureCacheMeta.verify 会拿契约值校验，夹具必须照办）。
AUDIO_DIM: int = 1024


def _config(tmp_path: Path, **overrides: Any):
    mapping: dict[str, Any] = {
        "data": {
            "source": "manifest",
            "manifest_path": str(tmp_path / "pairs.json"),
            "chart_dir": str(tmp_path / "charts"),
            "feature_dir": str(tmp_path / "features"),
            "max_samples": 4,
            "occlusion_ratio": 0.5,
            "k_max": 8,
            "t_window": 32,
            "x_bins": 32,
            "provenance": PROVENANCE,
        },
        "model": {
            "d_model": 16,
            "n_heads": 2,
            "n_layers": 2,
            "window": 4,
            "global_period": 2,
            "k_max": 8,
            "audio_dim": AUDIO_DIM,
            "dropout": 0.0,
        },
        "optim": {"lr": 1e-3, "batch_size": 1, "max_steps": 2},
        "run": {"experiment": "e2e-fixture", "save_every": 0, "log_every": 1},
    }
    for section, values in overrides.items():
        mapping.setdefault(section, {})
        mapping[section].update(values)
    return config_from_mapping(mapping)


def _write_assets(root: Path, *, audio_name: str = "fudahuang.mp3") -> Path:
    """写一份**最小可用**的 e2e 资产：假音频（只为 sha1）+ 真特征 npz + 元数据 + 清单。"""
    (root / "audio").mkdir(parents=True, exist_ok=True)
    (root / "feature").mkdir(parents=True, exist_ok=True)
    audio = root / "audio" / audio_name
    audio.write_bytes(b"not-a-real-mp3-but-hashable")
    array = np.zeros((FRAMES, AUDIO_DIM), dtype=np.float16)
    array[:, 0] = np.linspace(0.0, 1.0, FRAMES, dtype=np.float16)
    meta = FeatureCacheMeta(
        rate=75.0,
        sample_rate=24000,
        layer=12,
        model_rev="m-a-p/MERT-v1-330M",
        duration_s=DURATION_S,
        original_sample_rate=48000,
        feat_dim=AUDIO_DIM,
        dtype="float16",
        adapter="none",
    )
    meta.verify(array)  # 夹具自己必须先过六项校验，否则测的是「夹具坏了」
    save_feature_cache(
        array, root / "feature" / "song.npz", root / "feature" / "song.meta.json", meta
    )
    (root / "meta.json").write_text(
        json.dumps({"audio": f"audio/{audio_name}", "feature": "feature/song.npz"}),
        encoding="utf-8",
    )
    return audio


def _manifest(path: Path, *, feature_key: str) -> None:
    """写一份最小清单（`load_pairs` 的形状 == `PairSplits.to_dict()`）。"""
    row = {
        "chart_id": 7,
        "song_key": "song|composer",
        "name": "song",
        "composer": "composer",
        "split": "train",
        "chart_path": "7/1.json",
        "feature_key": feature_key,
        "difficulty": 15.3,
        "format": "rpe",
    }
    path.write_text(json.dumps({"train": [row], "val": [], "test": []}), encoding="utf-8")


# ══════════════════════════════════════════════════════════════
# 1. 清单解析
# ══════════════════════════════════════════════════════════════


def test_load_e2e_inputs_reads_required_and_optional_fields(tmp_path: Path) -> None:
    """必需两项 + 可选字段（含 `feature_meta` 的默认名）。"""
    root = tmp_path / "e2e"
    _write_assets(root)
    inputs = load_e2e_inputs(root)
    assert inputs.audio.name == "fudahuang.mp3"
    assert inputs.feature.name == "song.npz"
    assert (
        inputs.feature_meta == (root / "feature" / "song.meta.json").resolve()
    ), "没有显式指路时取同名的 <feature>.meta.json"
    assert inputs.bpm == 120.0
    assert inputs.lines == 16
    assert inputs.difficulty == 15.0
    assert inputs.steps == 8
    assert inputs.method == "thinning", "默认解码臂是 D2（D1 的阈值未标定，见 E2EInputs）"
    assert inputs.alpha == 1.0
    assert inputs.seed == 0


def test_load_e2e_inputs_prefers_an_explicit_meta_path(tmp_path: Path) -> None:
    """显式写的那一项**优先于**默认名（例子：《赴大荒》的 json 名有拼写差异，只能显式指路）。"""
    root = tmp_path / "e2e"
    _write_assets(root)
    other = root / "feature" / "other.json"
    other.write_text(
        (root / "feature" / "song.meta.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    payload = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    payload["feature_meta"] = "feature/other.json"
    payload["difficulty"] = 14.0
    (root / "meta.json").write_text(json.dumps(payload), encoding="utf-8")
    inputs = load_e2e_inputs(root)
    assert inputs.feature_meta == other.resolve()
    assert inputs.difficulty == 14.0


def test_load_e2e_inputs_rejects_missing_or_malformed(tmp_path: Path) -> None:
    """缺清单 / 缺必需项 / 指到不存在的文件 ⇒ **报错**（跳过等于「这条通道从没生效过」）。"""
    with pytest.raises(E2EInputError, match="清单不可用"):
        load_e2e_inputs(tmp_path / "nowhere")
    root = tmp_path / "e2e"
    root.mkdir()
    (root / "meta.json").write_text("{}", encoding="utf-8")
    with pytest.raises(E2EInputError, match="audio"):
        load_e2e_inputs(root)
    (root / "meta.json").write_text(
        json.dumps({"audio": "missing.mp3", "feature": "missing.npz"}), encoding="utf-8"
    )
    with pytest.raises(E2EInputError, match="不存在"):
        load_e2e_inputs(root)


def test_default_e2e_dir_points_at_the_repo_asset() -> None:
    """默认目录就是本仓库的资产目录（拼错一个字母等于这条通道永远不生效）。"""
    assert Path("tests/e2e-val") == DEFAULT_E2E_DIR
    assert OUTPUT_DIRNAME == "outputs"


# ══════════════════════════════════════════════════════════════
# 2. 条件来源
# ══════════════════════════════════════════════════════════════


def test_find_corpus_row_matches_by_audio_content_hash(tmp_path: Path) -> None:
    """按**内容 sha1** 找回该曲的行（文件名相同但内容不同不会误配）。"""
    root = tmp_path / "e2e"
    audio = _write_assets(root)
    manifest = tmp_path / "pairs.json"
    _manifest(manifest, feature_key=sha1_file(audio))
    found = find_corpus_row(sha1_file(audio), manifest_path=manifest)
    assert found is not None
    split, row = found
    assert split == "train"
    assert row["chart_path"] == "7/1.json"
    assert find_corpus_row("0" * 40, manifest_path=manifest) is None


def test_resolve_template_falls_back_to_synthetic_and_says_so(tmp_path: Path) -> None:
    """清单不可读 ⇒ 合成模板（无条件生成），并如实记 `synthetic` + 参数。"""
    root = tmp_path / "e2e"
    _write_assets(root)
    inputs = load_e2e_inputs(root)
    chart, info = resolve_template(
        inputs,
        chart_dir=tmp_path / "charts",
        manifest_path=tmp_path / "nope.json",
        audio_seconds=DURATION_S,
    )
    assert info["source"] == "synthetic"
    assert len(chart.lines) == inputs.lines
    assert chart.notes == []
    assert chart.duration_s() == pytest.approx(DURATION_S), "合成模板的 τ 轴就是音频长度"


def test_resolve_template_prefers_an_explicit_chart(tmp_path: Path) -> None:
    """显式 `chart` 优先（并写下 sha1，便于追溯）。"""
    root = tmp_path / "e2e"
    _write_assets(root)
    from beatmorph.io.formats.rpejson import dump_rpejson

    template = make_chart(
        notes=[make_note(t=0.0, line_id=0)], bpm_points=field_bpm((0.0, 150.0)), k=2
    )
    chart_path = tmp_path / "template.json"
    dump_rpejson(template, chart_path, report=check_chart(template))
    payload = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    payload["chart"] = str(chart_path)
    (root / "meta.json").write_text(json.dumps(payload), encoding="utf-8")
    inputs = load_e2e_inputs(root)
    chart, info = resolve_template(
        inputs,
        chart_dir=tmp_path / "charts",
        manifest_path=tmp_path / "nope.json",
        audio_seconds=DURATION_S,
    )
    assert info["source"] == "meta.json"
    assert info["sha1"] == sha1_file(chart_path)
    assert len(chart.lines) == 2
    assert len(chart.notes) == 1


def test_axis_end_seconds_follows_the_training_policy(tmp_path: Path) -> None:
    """τ 轴终点：显式覆盖 > policy；`audio` = min(谱面, 音频)。"""
    cfg = _config(tmp_path)
    chart = make_chart(notes=[make_note(t=3.0, line_id=0)], bpm_points=field_bpm((0.0, 120.0)), k=1)
    assert axis_end_seconds(cfg, chart, 8.0) == pytest.approx(min(chart.duration_s(), 8.0))
    from dataclasses import replace

    loud = replace(cfg, data=replace(cfg.data, tau_end_s=5.0))
    assert axis_end_seconds(loud, chart, 8.0) == pytest.approx(5.0)
    by_chart = replace(cfg, data=replace(cfg.data, tau_end_policy="chart"))
    assert axis_end_seconds(by_chart, chart, 8.0) == pytest.approx(chart.duration_s())


# ══════════════════════════════════════════════════════════════
# 3. 端到端真跑一遍（tiny 模型 / CPU / 合成特征）
# ══════════════════════════════════════════════════════════════


def _tiny_model(cfg: Any) -> MaskedFieldModel:
    grid = make_grid(t_bins=int(cfg.data.t_window), x_bins=int(cfg.data.x_bins))
    return MaskedFieldModel(
        ModelConfig(
            d_model=int(cfg.model.d_model),
            n_heads=int(cfg.model.n_heads),
            n_layers=int(cfg.model.n_layers),
            window=int(cfg.model.window),
            global_period=int(cfg.model.global_period),
            k_max=int(cfg.model.k_max),
            audio_dim=int(cfg.model.audio_dim),
            dropout=0.0,
        ),
        grid,
    ).eval()


def test_run_e2e_writes_a_self_describing_artifact(tmp_path: Path) -> None:
    """端到端跑一遍：产物目录 + `meta.json`（来源/口径/合法性）+ `notes.txt`。"""
    cfg = _config(tmp_path, run={"e2e_dir": str(tmp_path / "e2e")})
    root = tmp_path / "e2e"
    _write_assets(root)
    # 合成模板的 K 必须不超过 model.k_max（夹具的 k_max=8，默认 lines=16 装不下）
    payload = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    payload["lines"] = 2
    (root / "meta.json").write_text(json.dumps(payload), encoding="utf-8")
    result = run_e2e(_tiny_model(cfg), cfg, step=20, device=torch.device("cpu"), directory=root)
    assert isinstance(result, E2EResult)
    assert result.n_windows >= 1
    assert result.template["source"] == "synthetic"
    assert result.pairing.n_starts >= 0
    directories = sorted((root / OUTPUT_DIRNAME).iterdir())
    assert len(directories) == 1
    assert directories[0].name.endswith("-step20")
    payload = json.loads((directories[0] / "meta.json").read_text(encoding="utf-8"))
    assert payload["step"] == 20
    assert payload["template"]["source"] == "synthetic"
    assert payload["sampling"]["steps"] == 8
    assert payload["windowed_decode"]["dropped_tail_bins"] >= 0
    assert payload["audio"]["sha1"] == sha1_file(root / "audio" / "fudahuang.mp3")
    assert payload["legal"] is True, "postprocess 之后必须合法（否则红线 6 拒绝导出）"
    assert (directories[0] / "chart.json").exists()
    assert (directories[0] / "notes.txt").read_text(encoding="utf-8").strip()


def test_artifact_omits_the_chart_when_illegal(tmp_path: Path) -> None:
    """红线 6：不合法的结果**不写 `chart.json`**，但 `meta.json` 必须写（失败现场的证据）。"""
    cfg = _config(tmp_path)
    root = tmp_path / "e2e"
    _write_assets(root)
    inputs = load_e2e_inputs(root)
    chart = make_chart(
        notes=[
            make_note(t=0.0, line_id=0),
            make_note(t=0.0, line_id=0),
        ],
        bpm_points=field_bpm((0.0, 120.0)),
        k=1,
        chart_time_s=1.0,
    )
    illegal = check_chart(chart)
    assert not illegal.is_legal, "夹具必须真的不合法，否则本测试是恒真的"
    grid = FieldGrid(x_bins=int(cfg.data.x_bins)).for_chart(chart)
    result = E2EResult(
        chart=chart,
        report=illegal,
        inputs=inputs,
        template={"source": "synthetic"},
        grid=grid,
        difficulty=15.0,
        steps=8,
        events=(),
        pairing=_empty_pairing(),
        n_windows=1,
        windows_done=1,
        dropped_tail_bins=0,
        elapsed_s=0.5,
        decode_stats={},
    )
    directory = write_artifact(result, step=20000, root=root / OUTPUT_DIRNAME)
    assert not (directory / "chart.json").exists()
    payload = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    assert payload["legal"] is False
    assert payload["chart_written"] is False
    assert payload["violations"], "不合法时必须把违规项写进元数据"
    assert payload["summary"].endswith("不合法")


def _empty_pairing():
    from beatmorph.decoder import PairingStats

    return PairingStats(
        n_starts=0, n_ends=0, n_paired=0, n_unpaired_starts=0, n_orphan_ends=0, n_zero_length=0
    )


def test_event_budget_stops_a_runaway_decoder(tmp_path: Path) -> None:
    """事件预算闸（plan 07 §9-63 的教训）：解码器狂出事件时**必须中止**，不得拖垮长跑。

    来历：D1（peaks）在 α=1 下每个窗口解出 8.7-15.7 万个场事件（模型自己的期望是 0.02-2.96），
    而第一版实现把整首歌的事件全留在内存里 ⇒ 实测 21 GB 常驻、单核跑满、GPU 空转。
    """
    cfg = _config(tmp_path)
    root = tmp_path / "e2e"
    _write_assets(root)
    payload = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    payload["lines"] = 2
    payload["method"] = "peaks"
    (root / "meta.json").write_text(json.dumps(payload), encoding="utf-8")
    from beatmorph.infra.e2e import generate_chart, load_e2e_inputs

    result = generate_chart(
        _tiny_model(cfg),
        cfg,
        load_e2e_inputs(root),
        device=torch.device("cpu"),
        max_events=1,
    )
    assert result.aborted is not None, "预算用尽必须中止"
    assert "事件预算" in result.aborted
    assert result.is_complete is False
    assert result.windows_done == 0, "第一个窗口就超预算 ⇒ 一个窗口都没跑完"
    assert "model_expected_events" in result.decode_stats, "模型期望事件数必须记账"
    directory = write_artifact(result, step=7, root=root / OUTPUT_DIRNAME)
    assert not (directory / "chart.json").exists(), "中止 ⇒ 不写谱面"
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    assert meta["chart_written"] is False
    assert meta["aborted"]
    assert "thinning" in meta["aborted"] or "peaks" in meta["aborted"]
    assert meta["decode"]["aggregated"]["model_expected_events"] >= 0.0
    assert meta["windowed_decode"]["windows_done"] == 0


def test_generate_chart_does_not_touch_training_state(tmp_path: Path) -> None:
    """端到端生成必须**恢复**调用前的 train/eval 状态（否则训练路径被观测手段改掉）。"""
    cfg = _config(tmp_path)
    root = tmp_path / "e2e"
    _write_assets(root)
    payload = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    payload["lines"] = 2
    (root / "meta.json").write_text(json.dumps(payload), encoding="utf-8")
    model = _tiny_model(cfg)
    model.train()
    from beatmorph.infra.e2e import generate_chart

    generate_chart(model, cfg, load_e2e_inputs(root), device=torch.device("cpu"))
    assert model.training is True, "生成结束后必须回到 train 模式"


def test_e2e_inputs_type_is_hashable_and_immutable() -> None:
    """`E2EInputs` 是 frozen dataclass（产物元数据会带着它，不该被就地改）。"""
    inputs = E2EInputs(
        directory=Path("d"),
        audio=Path("a.mp3"),
        feature=Path("f.npz"),
        feature_meta=None,
        chart=None,
        bpm=120.0,
        lines=16,
        difficulty=15.0,
        steps=8,
        method="thinning",
        alpha=1.0,
        seed=0,
    )
    with pytest.raises(FrozenInstanceError):
        inputs.bpm = 90.0  # type: ignore[misc]


def test_note_helpers_are_wired_to_the_contract() -> None:
    """解码侧的类型映射与契约一致（R3 的产物必须是 RPE 的 note 类型，不是官谱表）。"""
    note = PhigrosNote(
        t=0.0,
        line_id=0,
        position_x=0.0,
        side=side_from_above(1),
        type=NoteType.TAP,
    )
    assert int(note.type) == 1
    assert len(JudgeLine(line_id=0).event_layers) >= 1
    assert ChartSource().sniff_evidence == "", "契约的默认来源说明是空串（不是 None）"
    assert isinstance(make_batch(k=1, grid=make_grid(t_bins=4, x_bins=8)), object)
    assert int(make_counts(k=1, grid=make_grid(t_bins=4, x_bins=8), events=2, seed=0).sum()) == 2
