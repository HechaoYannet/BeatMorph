"""RFC-0039 R2：val 集合的**分层重选**（默认 CI，无权重 / 无 GPU / 无真实语料）。

R2 的裁定是「val 窗口数 128 → ≥512，抽取口径从『按桶前 N 个』改成**跨桶、按事件密度分层**的
确定性抽样，构成必须落盘」。本文件钉死四件事：

1. **分层口径按总体比例，旧口径不按**：同一个 fixture 上直接对比 `_stratified_chunks` 与
   `_prefix_chunks` —— 前者选出的空窗占比等于总体，后者是 100%（这就是「128 窗系统性偏乐观」）;
2. **构成落盘**：`logs/val_composition.json` 在 val 步写出，含空窗/事件/K 分位/集合指纹；
3. **仍然可组批**：每一批同网格身份（前向成本 ∝ K²，§9-53 不能为了抽样放弃）；
4. **确定性**：同 seed 同指纹、不同 seed 不同指纹；没有密度来源时**回退并如实记账**
   （`stratified=False`），不假装做过分层。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from beatmorph.data.plan import plan_epoch
from beatmorph.generation.batch import FieldBatch
from beatmorph.generation.masks import build_occlusion_batch
from beatmorph.infra.artifacts import RunArtifacts
from beatmorph.infra.config.loading import config_from_mapping
from beatmorph.infra.smoke import SmokeBatchSource
from beatmorph.infra.train_loop import (
    VAL_COMPOSITION_FILENAME,
    VAL_DENSITY_STRATA,
    ManifestValSource,
    _density_strata,
    _flat,
    _largest_remainder,
    _prefix_chunks,
    _stratified_chunks,
    _top_up,
    train,
)
from tests.unit.generation._builders import make_batch, make_counts, make_grid
from tests.unit.infra._plan_worker_stub import DensityIndexDataset, IndexDataset

PROVENANCE = {
    "source": "fixtures",
    "query": "n/a",
    "fetched_at": "t",
    "purpose": "train",
    "script": "s",
    "script_version": "v1",
}

#: 40 个桶 × (1 行, 5 窗) = 200 个窗口（与真实语料一样是**多桶**结构，否则回退口径会因为
#: 桶被取空而取不满 requested，测出来的差异就不再来自分层）。
BUCKETS: tuple[tuple[int, int], ...] = tuple((1, 5) for _ in range(40))


def _config(**overrides: Any):
    mapping: dict[str, Any] = {
        "data": {
            "source": "synthetic",
            "max_samples": 4,
            "occlusion_ratio": 0.5,
            "k_max": 8,
            "t_window": 32,
            "provenance": PROVENANCE,
        },
        "model": {
            "d_model": 32,
            "n_heads": 2,
            "n_layers": 2,
            "window": 4,
            "global_period": 2,
            "k_max": 8,
            "audio_dim": 16,
            "dropout": 0.1,
        },
        "optim": {"lr": 1e-3, "batch_size": 1, "max_steps": 2},
        "run": {"experiment": "val-selection", "save_every": 2, "keep_last": 1, "log_every": 1},
    }
    for section, values in overrides.items():
        mapping.setdefault(section, {})
        mapping[section].update(values)
    return config_from_mapping(mapping)


def _events(*, n_buckets: int = 40, per_bucket: int = 5, empty_buckets: int = 8) -> list[int]:
    """逐窗口事件数：前 `empty_buckets` 个桶全空，其余 3..7 个事件（总体空窗占比 20%）。"""
    values: list[int] = []
    for bucket in range(n_buckets):
        for window in range(per_bucket):
            values.append(0 if bucket < empty_buckets else 3 + (bucket + window) % 5)
    return values


def _val_source(*, stratified: bool, n_windows: int, seed: int = 3, **optim: Any):
    """注入 stub 数据集的 val 来源（选择路径走真实实现，样本组装不是本文件的对象）。"""
    dataset: Any = DensityIndexDataset(BUCKETS, _events()) if stratified else IndexDataset(BUCKETS)
    cfg = _config(optim={"val_batch": 8, "val_windows": n_windows, **optim})
    source = ManifestValSource(cfg, seed=seed, n_windows=n_windows)
    source._dataset = dataset  # type: ignore[assignment]
    source._plan = plan_epoch(dataset, seed=seed, epoch=0, chunk=1, batch_size=1)
    return source


# ══════════════════════════════════════════════════════════════
# 1. 分层口径 vs 旧口径（同一个 fixture 上直接对比）
# ══════════════════════════════════════════════════════════════


def test_stratified_chunks_follow_the_population_where_prefix_does_not() -> None:
    """R2 的核心断言：分层口径的空窗占比 = 总体；旧口径 = 100%（系统性偏乐观）。

    构造：4 个桶各 4 个窗口；桶 0/1 全空、桶 2/3 全密集 ⇒ **总体空窗占比 50%**。
    取 8 个窗口：分层取 4 空 + 4 密；旧口径（每桶前 4 个、按桶序）取到的是**前两个桶**
    ——也就是全部 8 个空窗。
    """
    per_bucket = {0: [0, 1, 2, 3], 1: [4, 5, 6, 7], 2: [8, 9, 10, 11], 3: [12, 13, 14, 15]}
    counts = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5], dtype=np.int64)
    labels, bounds = _density_strata(counts, VAL_DENSITY_STRATA)
    # 层定义是**事件数的闭区间**：层 0 = 恰好 0 个事件，层 1 = 「>= 1」（这里实际取到 5）。
    assert bounds == [(0, 0), (1, -1)], "空窗单列一层，非空是另一层"
    population = [int(np.count_nonzero(labels == index)) for index in range(len(bounds))]
    quotas = _top_up(_largest_remainder(population, 8), population, 8)
    assert quotas == [4, 4], "配额按总体比例（8 个窗口里 4 空 4 密）"
    chunks, taken = _stratified_chunks(per_bucket, sorted(per_bucket), labels, quotas, batch=4)
    selected = _flat(chunks)
    assert taken == [4, 4]
    assert sum(1 for index in selected if counts[index] == 0) == 4
    prefix = _flat(_prefix_chunks(per_bucket, sorted(per_bucket), total=8, batch=4))
    assert all(counts[index] == 0 for index in prefix), "旧口径在这里 100% 是空窗"


def test_density_strata_merge_ties_and_handle_all_empty() -> None:
    """整数事件数大量并列 ⇒ 分位切点会重复，重复的层必须**合并**（不留空层）。"""
    counts = np.zeros(7, dtype=np.int64)
    labels, bounds = _density_strata(counts, VAL_DENSITY_STRATA)
    assert bounds == [(0, 0)]
    assert set(labels.tolist()) == {0}, "全空时只有一层"
    ties = np.asarray([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int64)
    _, tie_bounds = _density_strata(ties, VAL_DENSITY_STRATA)
    assert tie_bounds == [(0, 0), (1, -1)], "全并列 ⇒ 只有一层非空"


def test_largest_remainder_is_exact_proportional_and_deterministic() -> None:
    """配额分配：和恒等于 total、按比例、并列按下标（纯函数）。"""
    assert _largest_remainder([1, 1, 1], 10) == [
        1,
        1,
        1,
    ], "名额不得超过层总体（不足的部分由 _top_up 处理，见下一条）"
    assert sum(_largest_remainder([1, 1, 1], 3)) == 3
    assert _largest_remainder([3, 1], 4) == [3, 1]
    assert _largest_remainder([1, 1], 0) == [0, 0]
    first = _largest_remainder([5, 5, 5, 5], 10)
    assert first == _largest_remainder([5, 5, 5, 5], 10)
    assert sum(first) == 10


def test_top_up_never_exceeds_a_stratum_population() -> None:
    """补额只能补到「该层总体」为止（不得凭空多放窗口）。"""
    assert _top_up([0, 0], [1, 100], 10) == [1, 9]
    assert _top_up([2, 2], [2, 2], 4) == [2, 2]


# ══════════════════════════════════════════════════════════════
# 2. 通过 ManifestValSource 的结构性断言
# ══════════════════════════════════════════════════════════════


def test_selection_is_stratified_covers_more_buckets_and_keeps_grid_identity() -> None:
    """分层口径：跨桶、每批同网格身份、配额与实取一致、总数为 requested。"""
    source = _val_source(stratified=True, n_windows=128)
    selection = source.selection()
    assert selection.stratified is True
    assert selection.density_source == "window_cache"
    assert selection.selected == 128
    assert sum(stratum.taken for stratum in selection.strata) == 128
    for stratum in selection.strata:
        assert stratum.taken == stratum.quota
        assert stratum.quota <= stratum.population
    dataset = source._dataset
    for chunk in selection.chunks:
        keys = {dataset.grid_key(int(index)) for index in chunk}
        assert len(keys) == 1, f"同一 val 批必须同网格身份，得到 {keys}"
    assert len(selection.event_quantiles or ()) == 5
    assert len(selection.k_quantiles or ()) == 5
    buckets_used = {dataset.grid_key(int(index)) for index in selection.window_indices()}
    assert len(buckets_used) > 16, "跨桶：覆盖的桶数必须多于旧口径的 16 个"


def test_selection_is_deterministic_and_seed_sensitive() -> None:
    """同 (cfg, seed, 数据集) ⇒ 同指纹；换 seed ⇒ 换集合（否则「确定性」是恒真的）。"""
    first = _val_source(stratified=True, n_windows=64, seed=3).selection()
    again = _val_source(stratified=True, n_windows=64, seed=3).selection()
    other = _val_source(stratified=True, n_windows=64, seed=17).selection()
    assert first.fingerprint == again.fingerprint
    assert first.chunks == again.chunks
    assert first.fingerprint != other.fingerprint


def test_selection_cache_is_reused() -> None:
    """选择只算一次（密度扫描是秒级，每次 val 重算纯属浪费）。"""
    source = _val_source(stratified=True, n_windows=32)
    assert source.selection() is source.selection()


def test_fallback_without_density_is_recorded_not_pretended() -> None:
    """没有密度来源（无窗口缓存）⇒ 回退旧口径，并**如实记 `stratified=False`**。"""
    source = _val_source(stratified=False, n_windows=64)
    selection = source.selection()
    assert selection.stratified is False
    assert selection.density_source == "unavailable"
    assert selection.event_quantiles is None
    assert selection.k_quantiles is None
    assert selection.strata == ()
    assert selection.selected == 64
    assert selection.fingerprint, "回退口径也要有集合指纹（只是不含事件数）"


def test_empty_split_selects_nothing() -> None:
    """空 split：选择为空、不抛（降级到「关闭 val + 告警」由 `train()` 做）。"""
    cfg = _config(optim={"val_windows": 8})
    source = ManifestValSource(cfg, seed=3, n_windows=8)
    source._dataset = IndexDataset(())  # type: ignore[assignment]
    assert source.windows() == 0
    selection = source.selection()
    assert selection.selected == 0
    assert selection.chunks == ()
    assert selection.density_source == "empty-split"


# ══════════════════════════════════════════════════════════════
# 3. 构成落盘（RFC-0039 R2 的硬要求）
# ══════════════════════════════════════════════════════════════


def _field_batch(cfg: Any, *, batch: int, k: int = 2, seed: int = 0) -> FieldBatch:
    """一个真实的 val 批（evaluate_val 要真的跑前向，构造 stub 张量没有意义）。

    `batch` 必须等于 val 窗口数：`val_windows` 这个读数由**批内样本数**累加而来，
    用一个 B=1 的批冒充 8 个窗口会让构成文件里的实测读数是假的。
    """
    grid = make_grid(t_bins=int(cfg.data.t_window), x_bins=int(cfg.data.x_bins))
    counts = make_counts(batch=batch, k=k, grid=grid, events=4, holds=1, seed=seed)
    occlusion, _ = build_occlusion_batch(counts, ratio=0.5, seed=seed)
    return make_batch(
        batch=batch,
        k=k,
        grid=grid,
        counts=counts,
        occlusion=occlusion,
        audio_dim=int(cfg.model.audio_dim),
        seed=seed,
    )


class _InjectedValSource(ManifestValSource):
    """选择走真实实现、取批换成**一个固定真实批**（本文件不测样本组装）。"""

    def __init__(self, *args: Any, batch: FieldBatch, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._batch = batch
        self.calls = 0

    def batches(self) -> Iterator[FieldBatch]:
        self.calls += 1
        yield self._batch


def test_train_writes_val_composition_with_selection_and_measured(tmp_path: Path) -> None:
    """val 步写出 `logs/val_composition.json`：**选择**与**实测**都在里面。"""
    cfg = _config(optim={"max_steps": 2, "val_every": 1, "val_windows": 8})
    artifacts = RunArtifacts.create(runs_dir=tmp_path, experiment="val-comp", timestamp="T")
    dataset = DensityIndexDataset(BUCKETS, _events())
    val = _InjectedValSource(cfg, seed=3, n_windows=8, batch=_field_batch(cfg, batch=8))
    val._dataset = dataset  # type: ignore[assignment]
    val._plan = plan_epoch(dataset, seed=3, epoch=0, chunk=1, batch_size=1)
    train(
        cfg,
        source=SmokeBatchSource(cfg, seed=0, k_lines=2),
        artifacts=artifacts,
        data_rev="rev-1",
        gates_green=True,
        val_source=val,
    )
    path = artifacts.logs / VAL_COMPOSITION_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["step"] == 2
    selection = payload["selection"]
    assert selection["stratified"] is True
    assert selection["selected_windows"] == 8
    assert selection["fingerprint"]
    assert len(selection["event_quantiles"]) == 5
    assert len(selection["k_quantiles"]) == 5
    assert sum(stratum["taken"] for stratum in selection["strata"]) == 8
    measured = payload["measured"]
    assert measured["val_windows"] == 8.0
    assert 0.0 <= measured["val_empty_share"] <= 1.0
    assert "val_ratio" in measured


def test_composition_file_is_absent_without_a_manifest_source(tmp_path: Path) -> None:
    """反向对照：合成 val 来源（没有「选择」可言）**不写**构成文件——不得凭空造一份。"""
    cfg = _config(optim={"max_steps": 2, "val_every": 1})
    artifacts = RunArtifacts.create(runs_dir=tmp_path, experiment="val-nocomp", timestamp="T")
    train(
        cfg,
        source=SmokeBatchSource(cfg, seed=0, k_lines=2),
        artifacts=artifacts,
        data_rev="rev-1",
        gates_green=True,
    )
    assert not (artifacts.logs / VAL_COMPOSITION_FILENAME).exists()


def test_summary_names_the_strata_and_the_fingerprint() -> None:
    """日志摘要必须能一眼看出「分层与否 / 覆盖多少桶 / 指纹」——否则落盘也无人看。"""
    summary = _val_source(stratified=True, n_windows=64).selection().summary()
    assert "分层" in summary
    assert "window_cache" in summary
    assert "指纹" in summary
    assert "[0,0]" in summary, "空窗层的定义必须出现在摘要里"
