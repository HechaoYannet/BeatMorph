"""窗口数据集：RPEJSON + 特征缓存 -> `PairSample` -> `FieldBatch`（plan 02，默认 CI 无网络/无权重/无 GPU）。

夹具纪律（AGENTS.md §3.3 / 红线 7）：帧率、采样率、网格宽、拍格宽**全部引用契约常量**；
拍 -> 秒、tau 起点一律用 `SECONDS_PER_MINUTE` / `SUBDIVISIONS_PER_BEAT` / `TAU_GRID_DT` 现算，
不出现 75 / 24000 / 128 / 48 之类的字面量。BPM 与定数是**测试输入**，允许写字面量。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from beatmorph.core.contracts import (
    MERT_DEFAULT_FEAT_DIM,
    MERT_FRAME_RATE_HZ,
    MERT_SAMPLE_RATE_HZ,
    RPE_NORMAL_TRACKS,
    RPE_STAGE_HALF_WIDTH,
    RPE_TRACK_FIELDS,
    RPE_X_GRID_BINS,
    SUBDIVISIONS_PER_BEAT,
    TAU_GRID_DT,
    BpmPoint,
    ChartSource,
    NoteType,
)
from beatmorph.data.dataset import (
    ChartPairDataset,
    DatasetConfig,
    DatasetManifestError,
    GridMismatchError,
    collate_field_batch,
    load_pairs,
)
from beatmorph.data.parsers.rpejson import parse_rpejson
from beatmorph.data.pipeline.embed import (
    FeatureCacheMismatchError,
    PairRow,
    PairSplits,
    SplitError,
    feature_cache_paths,
)
from beatmorph.data.tracks import line_tracks_tensor
from beatmorph.field.grid import SECONDS_PER_MINUTE, FieldGrid, tau_to_seconds
from beatmorph.field.target import CHANNEL_INDEX, HOLD_END_CHANNEL, build_target
from beatmorph.generation.batch import N_ORDINARY_TRACKS, FieldBatch
from beatmorph.generation.masks import (
    assert_hold_pairs_not_split,
    mask_semantics,
    occluded_event_share,
)
from tests.unit.data._helpers import build_rpe, keyframe, note

#: 窗口长度 = 1 拍（派生，不写字面量 48）
WINDOW: int = SUBDIVISIONS_PER_BEAT
#: 谱面总长 = 5.5 拍 -> 5 个整窗 + 半个窗的尾巴（尾巴必须被丢弃并计数）
FULL_BEATS: float = (5 * SUBDIVISIONS_PER_BEAT + SUBDIVISIONS_PER_BEAT // 2) * TAU_GRID_DT
#: 测试输入（不是物理常量）
BPM: float = 120.0
BPM_ALT: float = 180.0
DIFFICULTY: float = 15.0


def _seconds(beats: float, bpm: float = BPM) -> float:
    """拍 -> 秒（测试侧独立算式；正式换算只在 field/）。"""
    return beats * SECONDS_PER_MINUTE / bpm


def _beat(value: float) -> list[int]:
    """拍值 -> RPE beat 三元组 [i, n, d]（分母取契约细分，保证精确）。"""
    index = int(value // 1)
    return [index, round((value - index) * SUBDIVISIONS_PER_BEAT), SUBDIVISIONS_PER_BEAT]


def _line(
    notes: list[dict[str, Any]],
    *,
    name: str = "line",
    tracks: dict[str, float] | None = None,
    beats: float = FULL_BEATS,
) -> dict[str, Any]:
    """一条判定线；`tracks` 给出各**常量**普通轨取值（键为 RPE 轨名）。"""
    line: dict[str, Any] = {"Name": name, "notes": notes}
    if tracks:
        line["eventLayers"] = [
            {
                rpe_name: [keyframe(_beat(0), _beat(beats), value, value)]
                for rpe_name, value in tracks.items()
            },
        ]
    return line


def _chart_dict(
    *,
    k: int = 2,
    beats: float = FULL_BEATS,
    bpm: float = BPM,
    hold: tuple[float, float] | None = (1.0, 1.5),
    constant_tracks: bool = False,
    sparse: tuple[tuple[int, float], ...] | None = None,
) -> dict[str, Any]:
    """最小 RPEJSON 谱面。

    默认布局让**每个窗口至少有 2 个遮盖单位与 2 个事件 token**（否则窗口无法满足 `r < 1`，
    会被规划器跳过——那正是数据集的契约，见 dataset 模块 docstring 第 4 条）：
    窗口 0：Tap@1/4 + Tap@1/2；窗口 1：Hold(1..3/2) + Tap@3/2；窗口 2：Flick@2 + Tap@5/2；
    窗口 3：Drag@3 + Tap@7/2；窗口 4：无事件（`r == 0`，仍是合法样本）。
    `sparse` 给出 `(线号, 拍)` 的极稀疏点集（用来构造少事件窗口）。
    """
    tracks = (
        {rpe_name: float(index + 1) for index, rpe_name in enumerate(RPE_NORMAL_TRACKS)}
        if constant_tracks
        else None
    )
    lines = [_line([], name=f"line-{index}", tracks=tracks, beats=beats) for index in range(k)]
    if sparse is not None:
        for line_index, sparse_beat in sparse:
            lines[line_index]["notes"].append(note(1, _beat(sparse_beat), position_x=0.0))
    else:
        lines[0]["notes"].append(note(1, _beat(1 / 4), position_x=-RPE_STAGE_HALF_WIDTH / 2))
        lines[k - 1]["notes"].append(note(1, _beat(1 / 2), position_x=+RPE_STAGE_HALF_WIDTH / 2))
        lines[k - 1]["notes"].append(note(1, _beat(3 / 2), position_x=0.0))
        lines[k - 1]["notes"].append(note(3, _beat(2), position_x=0.0, above=2))
        lines[0]["notes"].append(note(1, _beat(5 / 2), position_x=RPE_STAGE_HALF_WIDTH / 4))
        lines[k - 1]["notes"].append(note(4, _beat(3), position_x=0.0))
        lines[0]["notes"].append(note(1, _beat(7 / 2), position_x=-RPE_STAGE_HALF_WIDTH / 4))
    if hold is not None:
        lines[0]["notes"].append(note(2, _beat(hold[0]), end=_beat(hold[1]), position_x=0.0))
    return build_rpe(
        lines=lines,
        bpm_list=[{"bpm": bpm, "startTime": _beat(0)}],
        chart_time=_seconds(beats, bpm),
    )


def _write_feature(
    feature_dir: Path,
    key: str,
    *,
    duration_s: float,
    rate: float = MERT_FRAME_RATE_HZ,
    sample_rate: int = MERT_SAMPLE_RATE_HZ,
) -> None:
    """写一份迷你特征缓存（.npz + .meta.json），字段全部引用契约常量。"""
    npz_path, meta_path = feature_cache_paths(feature_dir, key)
    feature_dir.mkdir(parents=True, exist_ok=True)
    frames = round(duration_s * rate)
    emb = np.random.default_rng(0).standard_normal(
        (frames, MERT_DEFAULT_FEAT_DIM), dtype=np.float32
    )
    np.savez_compressed(npz_path, emb=emb)
    meta_path.write_text(
        json.dumps(
            {
                "rate": rate,
                "sample_rate": sample_rate,
                "layer": 12,
                "model_rev": "unit-test",
                "duration_s": duration_s,
                "original_sample_rate": MERT_SAMPLE_RATE_HZ,
                "feat_dim": MERT_DEFAULT_FEAT_DIM,
                "dtype": str(emb.dtype),
                "adapter": "none",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _materialize(tmp_path: Path, specs: list[dict[str, Any]]) -> tuple[Path, Path, Path]:
    """按规格物化 charts / features / manifest，返回三者的路径。"""
    chart_dir = tmp_path / "charts"
    feature_dir = tmp_path / "features"
    rows: list[PairRow] = []
    for index, spec in enumerate(specs):
        relative = f"chart-{index}/chart.json"
        payload = _chart_dict(**spec.get("chart", {}))
        target = chart_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        key = f"feat-{index}"
        chart_kwargs = spec.get("chart", {})
        _write_feature(
            feature_dir,
            key,
            duration_s=_seconds(
                chart_kwargs.get("beats", FULL_BEATS), chart_kwargs.get("bpm", BPM)
            ),
            rate=spec.get("rate", MERT_FRAME_RATE_HZ),
        )
        rows.append(
            PairRow(
                chart_id=index + 1,
                song_key=f"song-{index}|composer",
                split="train",
                chart_path=relative,
                feature_key=key,
                difficulty=spec.get("difficulty", DIFFICULTY),
                fmt="rpe",
                name=f"song-{index}",
                composer="composer",
            ),
        )
    manifest = tmp_path / "manifest.json"
    # 用**真实生产者**的序列化（PairSplits.to_dict）落盘，避免测试自造一套格式
    manifest.write_text(
        json.dumps(PairSplits(train=rows).to_dict(), ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest, chart_dir, feature_dir


def _dataset(tmp_path: Path, specs: list[dict[str, Any]], **overrides: Any) -> ChartPairDataset:
    """物化并构造数据集（`t_window` 默认 1 拍）。"""
    manifest, chart_dir, feature_dir = _materialize(tmp_path, specs)
    config = DatasetConfig(
        manifest_path=manifest,
        chart_dir=chart_dir,
        feature_dir=feature_dir,
        t_window=WINDOW,
        **overrides,
    )
    return ChartPairDataset(config)


def _spec(**chart: Any) -> dict[str, Any]:
    """一条行规格（`chart` 进 :func:`_chart_dict`）。"""
    return {"chart": chart}


def _window_count(beats: float) -> int:
    """谱面拍数 -> 整窗数（尾巴丢弃；与数据集口径一致）。"""
    return round(beats * SUBDIVISIONS_PER_BEAT) // WINDOW


def _hold_balance(counts: torch.Tensor) -> tuple[int, int]:
    """(窗口内 hold 起点事件数, hold 终点事件数)。"""
    return (
        int(counts[..., CHANNEL_INDEX[NoteType.HOLD]].sum()),
        int(counts[..., HOLD_END_CHANNEL].sum()),
    )


# ══════════════════════════════════════════════════════════════
# tracks：通道顺序与绝对 tau 求值
# ══════════════════════════════════════════════════════════════


def test_track_channel_order_follows_contract(tmp_path: Path) -> None:
    """通道 j 必须等于 RPE_TRACK_FIELDS[j]（**不得写死顺序**）：给出可区分的常量即可验证。"""
    payload = _chart_dict(k=1, constant_tracks=True)
    chart = parse_rpejson(
        json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        ChartSource(chart_id=1),
    )
    grid = FieldGrid(x_bins=RPE_X_GRID_BINS).for_chart(chart)
    tracks = line_tracks_tensor(chart, grid)
    assert tuple(tracks.shape) == (1, grid.t_bins, N_ORDINARY_TRACKS)
    assert tracks.dtype == torch.float32
    for field_index, name in enumerate(RPE_TRACK_FIELDS):
        rpe_name = RPE_NORMAL_TRACKS[field_index]
        expected = float(RPE_NORMAL_TRACKS.index(rpe_name) + 1)
        assert torch.allclose(tracks[0, :, field_index], torch.full((grid.t_bins,), expected)), (
            f"通道 {field_index}（{name}）的顺序与 RPE_TRACK_FIELDS 不一致"
        )


def test_line_tracks_tensor_requires_bound_grid() -> None:
    """未绑定 tau 轴的网格必须报错，而不是给出空张量。"""
    payload = _chart_dict(k=1)
    chart = parse_rpejson(
        json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        ChartSource(chart_id=1),
    )
    with pytest.raises(ValueError, match="t_bins"):
        line_tracks_tensor(chart, FieldGrid(x_bins=RPE_X_GRID_BINS))


# ══════════════════════════════════════════════════════════════
# len / 窗口口径
# ══════════════════════════════════════════════════════════════


def test_len_counts_full_windows_and_accounts_tail(tmp_path: Path) -> None:
    """`__len__` == 整窗数；不足一窗的尾巴被丢弃并**计数**。"""
    dataset = _dataset(tmp_path, [_spec()])
    expected_windows = _window_count(FULL_BEATS)
    expected_tail = round(FULL_BEATS * SUBDIVISIONS_PER_BEAT) - expected_windows * WINDOW
    assert len(dataset) == expected_windows
    stats = dataset.stats()
    assert stats.n_windows == expected_windows
    assert stats.dropped_tail_bins == expected_tail
    assert expected_tail > 0, "夹具必须覆盖「不足一窗的尾巴」这一口径"
    assert stats.n_rows == 1
    assert stats.n_rows_used == 1
    assert stats.skipped_windows_hold_split == 0
    assert stats.skipped_windows_bpm_crossing == 0


def test_grid_key_is_the_batching_identity(tmp_path: Path) -> None:
    """`grid_key` 必须与 `collate_field_batch` 的网格身份判定一致（采样器按它分桶组批）。"""
    dataset = _dataset(tmp_path, [_spec(k=2), _spec(k=2, bpm=BPM_ALT)])
    keys = {dataset.grid_key(index) for index in range(len(dataset))}
    assert len(keys) == 2, "两首不同 BPM 的谱必须给出两个不同的网格身份"
    for index in range(len(dataset)):
        sample = dataset[index]
        assert dataset.grid_key(index) == (
            sample.grid.x_bins,
            sample.grid.t_bins,
            sample.grid.bpm_points[0].bpm,
        )


def test_limit_selects_prefix_rows(tmp_path: Path) -> None:
    """`limit` 只取前 N 行（其余行不进索引）。"""
    dataset = _dataset(tmp_path, [_spec(k=2), _spec(k=4)], limit=1)
    assert {sample.pair_index for sample in (dataset[i] for i in range(len(dataset)))} == {0}


def test_describe_reports_index_accounting(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, [_spec()])
    text = dataset.describe()
    assert "ChartPairDataset" in text
    assert "数据集索引" in text
    assert len(dataset.rows()) == 1


# ══════════════════════════════════════════════════════════════
# __getitem__：形状 / dtype / 音频切片
# ══════════════════════════════════════════════════════════════


def test_sample_shapes_and_dtypes_match_grid(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, [_spec(k=2)])
    sample = dataset[0]
    k_lines = 2
    assert tuple(sample.grid.bpm_points) == (BpmPoint(time_beats=0.0, bpm=BPM),)
    assert sample.grid.t_bins == WINDOW
    assert sample.grid.x_bins == RPE_X_GRID_BINS
    assert tuple(sample.line_tracks.shape) == (k_lines, WINDOW, N_ORDINARY_TRACKS)
    assert sample.line_tracks.dtype == torch.float32
    assert tuple(sample.counts.shape) == (
        k_lines,
        WINDOW,
        sample.grid.x_bins,
        sample.grid.sides,
        sample.grid.channels,
    )
    assert sample.counts.dtype == torch.int16
    assert sample.occlusion.shape == sample.counts.shape
    assert sample.occlusion.dtype == torch.bool
    assert tuple(sample.line_mask.shape) == (k_lines,)
    assert sample.line_mask.dtype == torch.bool
    assert bool(sample.line_mask.all())
    assert tuple(sample.difficulty.shape) == ()
    assert sample.difficulty.dtype == torch.float32
    assert float(sample.difficulty) == pytest.approx(DIFFICULTY)
    assert sample.frame_rate == MERT_FRAME_RATE_HZ
    assert sample.audio_emb.dtype == torch.float32
    assert sample.audio_emb.dim() == 2
    assert int(sample.audio_emb.shape[1]) == MERT_DEFAULT_FEAT_DIM
    # 计数守恒：窗口内事件数 == build_target 在**全谱**上的事件数分配（此处只查非负与总量）
    assert int(sample.counts.sum()) > 0


def test_audio_slice_is_derived_from_frame_rate(tmp_path: Path) -> None:
    """音频按窗口**秒区间**切片：帧下标与帧数都由 MERT_FRAME_RATE_HZ 派生。"""
    dataset = _dataset(tmp_path, [_spec(k=2)])
    bpm_points = (BpmPoint(time_beats=0.0, bpm=BPM),)
    for index in (0, len(dataset) - 1):
        sample = dataset[index]
        start_s = float(tau_to_seconds(sample.tau_start, bpm_points))
        end_s = float(tau_to_seconds(sample.tau_start + WINDOW * TAU_GRID_DT, bpm_points))
        assert sample.audio_frame_start == round(start_s * MERT_FRAME_RATE_HZ)
        assert sample.audio_padded_frames == 0
        expected_frames = end_s * MERT_FRAME_RATE_HZ - start_s * MERT_FRAME_RATE_HZ
        assert abs(int(sample.audio_emb.shape[0]) - expected_frames) <= 1
        assert sample.window_index == index
        assert sample.tau_start == pytest.approx(index * WINDOW * TAU_GRID_DT)


def test_window_tracks_equal_full_chart_slice(tmp_path: Path) -> None:
    """窗口事件轨 == 全谱 `line_tracks_tensor` 的对应切片（求值点是**绝对** tau）。"""
    specs = [_spec(k=2)]
    dataset = _dataset(tmp_path, specs)
    chart_bytes = (tmp_path / "charts" / "chart-0" / "chart.json").read_bytes()
    chart = parse_rpejson(chart_bytes, ChartSource(chart_id=1))
    full_grid = FieldGrid(x_bins=RPE_X_GRID_BINS).for_chart(chart)
    full_tracks = line_tracks_tensor(chart, full_grid)
    for index in range(len(dataset)):
        sample = dataset[index]
        start = round(sample.tau_start / TAU_GRID_DT)
        expected = full_tracks[:, start : start + WINDOW, :]
        assert torch.equal(sample.line_tracks, expected)


def test_window_counts_equal_full_chart_slice(tmp_path: Path) -> None:
    """窗口计数 == **全谱** `build_target` 的对应切片。

    这条断言锁定「窗口子谱 + 窗口网格」与「全谱建表后切窗」的**逐格等价**：
    前者是内存可行的实现（全谱在真实谱面上要 ~1e9 格），后者是口径定义。
    等价性的前提是窗口 BPM 纯（由窗口规划强制）。
    """
    dataset = _dataset(tmp_path, [_spec(k=2)])
    chart_bytes = (tmp_path / "charts" / "chart-0" / "chart.json").read_bytes()
    chart = parse_rpejson(chart_bytes, ChartSource(chart_id=1))
    full_grid = FieldGrid(x_bins=RPE_X_GRID_BINS).for_chart(chart)
    full_counts = build_target(chart, full_grid).counts
    for index in range(len(dataset)):
        sample = dataset[index]
        start = round(sample.tau_start / TAU_GRID_DT)
        expected = torch.as_tensor(full_counts[:, start : start + WINDOW])
        assert torch.equal(sample.counts, expected), f"窗口 {index} 的计数与全谱切片不一致"


# ══════════════════════════════════════════════════════════════
# 遮盖语义
# ══════════════════════════════════════════════════════════════


def test_occlusion_hides_masked_cells_and_keeps_hold_pairs(tmp_path: Path) -> None:
    """遮盖：r > 0 时确有事件被遮；被遮格子在模型输入侧为 0；Hold 配对不被拆散。"""
    dataset = _dataset(tmp_path, [_spec(k=2)])
    sample = dataset[1]  # 含 Hold 的窗口（第 2 窗口 = 拍 1..2）
    counts = sample.counts
    total_events = int(counts.sum())
    assert total_events > 0
    assert_hold_pairs_not_split(counts, sample.occlusion)  # 契约断言：不抛即通过
    stats = mask_semantics(counts, sample.occlusion)
    assert stats["n_occluded_cells"] > 0, "r=0.5 时应确有事件被遮盖"
    assert stats["n_occluded_events"] + stats["n_observed_cells"] <= total_events
    batch = collate_field_batch([sample])
    observed = batch.observed_counts()
    hidden = sample.occlusion.unsqueeze(0)
    assert float(observed[hidden].abs().sum()) == 0.0
    hidden_events = float((counts.to(torch.float32) * sample.occlusion.to(torch.float32)).sum())
    assert float(observed.sum()) == pytest.approx(total_events - hidden_events)


def test_dataset_never_emits_fully_occluded_window(tmp_path: Path) -> None:
    """**不变量**：数据集绝不产出 `r == 1` 的窗口。

    `masked_poisson_loss` 在 `r == 1` 时直接抛
    （「全部事件都被遮盖，事件项没有可见上下文，拒绝训练」），
    而稀疏窗口上「遮盖单位恰好覆盖全部事件」是真实数据必然遇到的情况——
    因此这是数据集一方的责任，不是训练侧的容错。
    """
    specs = [
        _spec(k=2),
        _spec(k=2, sparse=((0, 1 / 4), (1, 13 / 4)), hold=None),
        _spec(k=2, sparse=((0, 1 / 4), (0, 1 / 4)), hold=None),
    ]
    dataset = _dataset(tmp_path, specs)
    assert len(dataset) > 0
    for index in range(len(dataset)):
        sample = dataset[index]
        assert occluded_event_share(sample.counts, sample.occlusion) < 1.0, (
            f"窗口 {index}（pair_index={sample.pair_index}）的 r == 1：losses 会拒绝该 batch"
        )


def test_single_unit_windows_are_skipped_and_accounted(tmp_path: Path) -> None:
    """少于 2 个遮盖单位 / 2 个事件 token 的窗口与种子无关地必然全遮 => 规划期跳过并记账。

    两种构造：① 两个窗口各只有 1 个 Tap（1 单位 1 token）；
    ② 同一线同一拍的 2 个 Tap（2 单位但**只有 1 个 token**，扩张遮盖后仍全遮）。
    """
    dataset = _dataset(tmp_path, [_spec(k=2, sparse=((0, 1 / 4), (1, 13 / 4)), hold=None)])
    stats = dataset.stats()
    assert stats.skipped_windows_no_visible_context == 2
    assert stats.dropped_events_no_visible_context == 2
    assert stats.n_windows == _window_count(FULL_BEATS) - 2

    same_token = _dataset(
        tmp_path / "same_token",
        [_spec(k=2, sparse=((0, 1 / 4), (0, 1 / 4)), hold=None)],
    )
    assert same_token.stats().skipped_windows_no_visible_context == 1
    assert all(
        occluded_event_share(same_token[i].counts, same_token[i].occlusion) < 1.0
        for i in range(len(same_token))
    )


def test_unsatisfiable_window_degrades_to_no_occlusion(tmp_path: Path) -> None:
    """残余情形（必要条件挡不住）：一对 Hold + **同一 token 内**的独立点。

    该窗口的 2 个遮盖单位（配对 + 独立点）无论以什么顺序被选中，token 扩张都会覆盖
    全部事件 => 任何种子都是 `r == 1`。数据集此时**退化为无遮盖**（`r == 0`，
    losses 的 r == 0 契约分支）并计数告警，而**不是**发出 `r == 1`、
    也不是在 `__getitem__` 里偷偷跳过（那会破坏 `__len__` 的语义）。
    """
    dataset = _dataset(tmp_path, [_spec(k=2, sparse=((0, 1.0),), hold=(1.0, 1.5))])
    with_events = [dataset[index] for index in range(len(dataset))]
    assert len(with_events) == _window_count(FULL_BEATS)
    assert dataset.no_visible_context_fallbacks() >= 1
    assert "退化" in dataset.describe()
    for sample in with_events:
        assert occluded_event_share(sample.counts, sample.occlusion) < 1.0
    non_empty = [sample for sample in with_events if int(sample.counts.sum()) > 0]
    assert len(non_empty) == 1
    assert int(non_empty[0].counts.sum()) == 3
    assert occluded_event_share(non_empty[0].counts, non_empty[0].occlusion) == 0.0
    assert not bool(non_empty[0].occlusion.any())


def test_occlusion_is_deterministic_per_index(tmp_path: Path) -> None:
    """同一 index 两次取值逐位一致（种子由 index 派生，不依赖全局随机状态）。"""
    dataset = _dataset(tmp_path, [_spec(k=2)])
    first = dataset[3]
    second = dataset[3]
    assert torch.equal(first.counts, second.counts)
    assert torch.equal(first.occlusion, second.occlusion)
    assert torch.equal(first.line_tracks, second.line_tracks)
    assert torch.equal(first.audio_emb, second.audio_emb)
    assert torch.equal(first.difficulty, second.difficulty)


# ══════════════════════════════════════════════════════════════
# collate：网格身份 / K padding
# ══════════════════════════════════════════════════════════════


def test_collate_single_sample_passes_assert_shapes(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, [_spec(k=2)])
    batch = collate_field_batch([dataset[0]])
    assert isinstance(batch, FieldBatch)
    batch.assert_shapes()
    assert batch.batch_size() == 1
    assert batch.n_lines() == 2
    assert batch.grid.t_bins == WINDOW
    assert batch.counts is not None
    assert batch.counts.dtype == torch.int16
    assert batch.line_tracks.shape == (1, 2, WINDOW, N_ORDINARY_TRACKS)
    assert "FieldBatch" in batch.describe()


def test_collate_pads_k_and_sets_line_mask_false(tmp_path: Path) -> None:
    """K 不同 -> pad 到批内最大 K，`line_mask` 的 pad 位为 False，三个张量同步 pad。"""
    dataset = _dataset(tmp_path, [_spec(k=2), _spec(k=4)])
    two_line = dataset[0]
    four_line = next(dataset[i] for i in range(len(dataset)) if dataset[i].pair_index == 1)
    assert int(two_line.line_tracks.shape[0]) == 2
    assert int(four_line.line_tracks.shape[0]) == 4
    batch = collate_field_batch([two_line, four_line])
    batch.assert_shapes()
    assert batch.n_lines() == 4
    assert tuple(batch.line_mask.shape) == (2, 4)
    assert torch.equal(
        batch.line_mask[0],
        torch.tensor([True, True, False, False]),
    )
    assert bool(batch.line_mask[1].all())
    assert float(batch.line_tracks[0, 2:].abs().sum()) == 0.0
    assert int(batch.counts[0, 2:].sum()) == 0
    assert not bool(batch.occlusion[0, 2:].any())


def test_collate_rejects_mixed_grids(tmp_path: Path) -> None:
    """负例：混批不同 `bpm_points`（= 不同 J 向量）必须抛错，不得静默混批。"""
    dataset = _dataset(tmp_path, [_spec(k=2), _spec(k=2, bpm=BPM_ALT)])
    same_bpm = dataset[0]
    other_bpm = next(dataset[i] for i in range(len(dataset)) if dataset[i].pair_index == 1)
    assert same_bpm.grid.bpm_points != other_bpm.grid.bpm_points
    with pytest.raises(GridMismatchError, match="网格身份"):
        collate_field_batch([same_bpm, other_bpm])


def test_collate_rejects_k_over_capacity(tmp_path: Path) -> None:
    """负例：批内 K 超过 `k_max` 必须抛错（pad 不得越界）。"""
    dataset = _dataset(tmp_path, [_spec(k=2)])
    with pytest.raises(ValueError, match="k_max"):
        collate_field_batch([dataset[0]], k_max=1)


# ══════════════════════════════════════════════════════════════
# 窗口边界：不切断 Hold
# ══════════════════════════════════════════════════════════════


def test_window_boundary_never_splits_a_hold_pair(tmp_path: Path) -> None:
    """Hold 跨名义窗界（拍 0.75..1.25 跨拍 1 的窗界）时，边界必须前移，且每个窗口
    内 hold 起点与终点事件数相等（**不得产生半截 Hold**）。"""
    dataset = _dataset(tmp_path, [_spec(k=2, hold=(0.75, 1.25))])
    assert len(dataset) == _window_count(FULL_BEATS)
    assert dataset.stats().shifted_windows >= 1
    assert dataset.stats().skipped_windows_hold_split == 0
    for index in range(len(dataset)):
        sample = dataset[index]
        starts, ends = _hold_balance(sample.counts)
        assert starts == ends, f"窗口 {index} 出现半截 Hold：起点 {starts} / 终点 {ends}"
        assert_hold_pairs_not_split(sample.counts, sample.occlusion)
    assert sum(_hold_balance(dataset[i].counts)[0] for i in range(len(dataset))) == 1


def test_hold_inside_one_window_keeps_nominal_boundaries(tmp_path: Path) -> None:
    """Hold 完全落在窗口内时**不**前移边界（前移只发生在真的会被切断时）。"""
    dataset = _dataset(tmp_path, [_spec(k=2)])
    assert dataset.stats().shifted_windows == 0
    for index in range(len(dataset)):
        starts, ends = _hold_balance(dataset[index].counts)
        assert starts == ends


# ══════════════════════════════════════════════════════════════
# 负例：特征缓存元数据 / 清单
# ══════════════════════════════════════════════════════════════


def test_feature_metadata_mismatch_raises_not_warns(tmp_path: Path) -> None:
    """负例：`rate` 与契约不符必须**报错**（25 Hz 事故的哨兵），且不是静默跳过。"""
    wrong_rate = MERT_FRAME_RATE_HZ / 3.0
    dataset = _dataset(tmp_path, [{"chart": {}, "rate": wrong_rate}])
    assert len(dataset) == _window_count(FULL_BEATS)
    with pytest.raises(FeatureCacheMismatchError, match="rate"):
        dataset[0]


def test_sample_rate_mismatch_raises(tmp_path: Path) -> None:
    """负例：`sample_rate` 不符同样报错（六项校验逐项生效）。"""
    manifest, chart_dir, feature_dir = _materialize(tmp_path, [_spec()])
    _write_feature(
        feature_dir,
        "feat-0",
        duration_s=_seconds(FULL_BEATS),
        sample_rate=MERT_SAMPLE_RATE_HZ * 2,
    )
    dataset = ChartPairDataset(
        DatasetConfig(
            manifest_path=manifest,
            chart_dir=chart_dir,
            feature_dir=feature_dir,
            t_window=WINDOW,
        ),
    )
    with pytest.raises(FeatureCacheMismatchError, match="sample_rate"):
        dataset[0]


def test_missing_difficulty_row_is_skipped_and_accounted(tmp_path: Path) -> None:
    """没有定数的行不得静默当成 0.0：跳过并计数。"""
    dataset = _dataset(tmp_path, [{"chart": {}, "difficulty": None}])
    assert len(dataset) == 0
    assert dataset.stats().skipped_no_difficulty == 1


def test_load_pairs_rejects_invalid_split(tmp_path: Path) -> None:
    """负例：split 名非法即抛 `SplitError`。"""
    manifest, _charts, _features = _materialize(tmp_path, [_spec()])
    with pytest.raises(SplitError, match="split"):
        load_pairs(manifest, "validation")


def test_load_pairs_reads_rows_from_producer_format(tmp_path: Path) -> None:
    """`load_pairs` 能读回生产者的 `PairSplits.to_dict()`（字段一致）。"""
    manifest, _charts, _features = _materialize(tmp_path, [_spec()])
    rows = load_pairs(manifest, "train")
    assert len(rows) == 1
    assert rows[0].chart_id == 1
    assert rows[0].difficulty == pytest.approx(DIFFICULTY)
    assert rows[0].feature_key == "feat-0"
    assert load_pairs(manifest, "val") == []


def test_load_pairs_rejects_malformed_manifest(tmp_path: Path) -> None:
    """负例：清单根不是对象 / 缺 split 数组 -> `DatasetManifestError`。"""
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"train": "nope"}), encoding="utf-8")
    with pytest.raises(DatasetManifestError, match="train"):
        load_pairs(path, "train")
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(DatasetManifestError, match="根"):
        load_pairs(path, "train")


def test_config_rejects_invalid_values(tmp_path: Path) -> None:
    """配置级负例：非法 split / 窗口 / k_max / 遮盖比例立即抛。"""
    manifest, chart_dir, feature_dir = _materialize(tmp_path, [_spec()])
    base: dict[str, Any] = {
        "manifest_path": manifest,
        "chart_dir": chart_dir,
        "feature_dir": feature_dir,
        "t_window": WINDOW,
    }
    with pytest.raises(SplitError):
        DatasetConfig(**base, split="validation")
    with pytest.raises(ValueError, match="t_window"):
        DatasetConfig(**{**base, "t_window": 0})
    with pytest.raises(ValueError, match="k_max"):
        DatasetConfig(**{**base, "k_max": 0})
    with pytest.raises(ValueError, match="occlusion_ratio"):
        DatasetConfig(**base, split="train", occlusion_ratio=1.5)
