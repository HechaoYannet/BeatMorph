"""val 路径与「单步 loss ↔ 窗内事件数」解耦（plan 07 §9-47 A–I；默认 CI，无权重 / 无 GPU / 无真实语料）。

本文件钉死四件事（对应验收 T1/T3/T4 与「val 不得变成训练信号」）：

1. **固定验证集逐位一致**（T1）：`ManifestValSource` 的前 N 个槽位来自
   `plan_epoch(split_val, seed, epoch=0)` 的前缀，两次调用逐位一致，`data.workers=0` 与
   `=2` 也逐位一致（真实 spawn，不是 mock）；换一个实例同 seed 仍逐位一致（续训后可复现）。
2. **分层把两个总体分开了**（T3）：构造「一个空窗 + 一个非空窗」的批，断言
   `loss_empty` 落在**空窗**上（且恒等于积分项 ∫λdV），`loss_nonempty` 落在非空窗上，
   而混合的单步 `loss` 是两者的混合——单看它无法知道这一步来自哪个总体。
3. **`best.pt` 按 `val/ratio` 选**（T4）：构造「训练损失最优步 ≠ val/ratio 最优步」的
   剧本，断言 best.pt 落在后者。
4. **val 不改变训练**：同一 seed 下带 val 与不带 val 的两轮训练，训练损失序列**逐位一致**；
   并且 val 标量的名字全部在 `SCALAR_TAGS` 里（jsonl 与 TB 不许漂成两套）。
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
import torch

import beatmorph.data.dataset as dataset_mod
import beatmorph.infra.train_loop as train_loop_mod
from beatmorph.core.contracts.phigros import SUBDIVISIONS_PER_BEAT, NoteType
from beatmorph.eval.val_metrics import ValAccumulator, masked_readout
from beatmorph.field.target import CHANNEL_INDEX
from beatmorph.generation.batch import FieldOutput
from beatmorph.generation.losses import integral_term, masked_poisson_loss
from beatmorph.infra.artifacts import RunArtifacts
from beatmorph.infra.checkpoint import BEST_CHECKPOINT_NAME, config_fingerprint, load_checkpoint
from beatmorph.infra.config.loading import config_from_mapping
from beatmorph.infra.config.schema import ConfigError
from beatmorph.infra.smoke import SmokeBatchSource
from beatmorph.infra.train_loop import (
    HISTORY_FILENAME,
    SCALAR_TAGS,
    ManifestBatchSource,
    ManifestValSource,
    stratified_step_loss,
    train,
)
from tests.unit.generation._builders import make_batch, make_grid

# spawn 出来的 worker 要能 import 本目录下的 stub（spawn 会把父进程的 sys.path 传给子进程）。
sys.path.insert(0, str(Path(__file__).resolve().parent))

PROVENANCE = {
    "source": "fixtures",
    "query": "n/a",
    "fetched_at": "t",
    "purpose": "train",
    "script": "s",
    "script_version": "v1",
}

#: 桶大小不均（与 test_plan_batches / test_sampler_coverage 同夹具口径）。
BUCKETS: tuple[tuple[int, int], ...] = ((1, 1), (2, 5), (5, 7), (3, 30))


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
        "optim": {"lr": 1e-3, "batch_size": 1, "max_steps": 4},
        "run": {"experiment": "val-fixture", "save_every": 2, "keep_last": 3, "log_every": 1},
    }
    for section, values in overrides.items():
        mapping.setdefault(section, {})
        mapping[section].update(values)
    return config_from_mapping(mapping)


def _stub_collate(samples: Sequence[int]) -> list[int]:
    """`collate_field_batch` 的替身：val 取批路径是本文件的对象，样本组装不是。"""
    return list(samples)


def _val_source(
    monkeypatch: pytest.MonkeyPatch,
    *,
    workers: int = 0,
    n_windows: int = 8,
    seed: int = 3,
) -> ManifestValSource:
    from _plan_worker_stub import IndexDataset

    monkeypatch.setattr(dataset_mod, "collate_field_batch", _stub_collate)
    source = ManifestValSource(_config(data={"workers": workers}), seed=seed, n_windows=n_windows)
    source._dataset = IndexDataset(BUCKETS)  # type: ignore[assignment]
    return source


def _take(source: ManifestValSource) -> list[int]:
    """把 val 批摊平成一串窗口下标（顺序敏感：T1 要比的就是顺序）。"""
    return [index for batch in source.batches() for index in batch]


# ══════════════════════════════════════════════════════════════
# T1：固定验证集逐位一致
# ══════════════════════════════════════════════════════════════


def test_val_windows_come_from_the_plan_grouped_by_grid_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A1 的**修订**口径（§9-53）：val 取的是计划的**按桶成组**前缀，不是槽位前缀。

    为什么改了：计划跨桶轮转发牌 ⇒ 槽位前缀会让 N 个窗口落在 N 个不同桶 ⇒ B=1 × N 次前向，
    而前向成本 ∝ K²（实测 val 前缀 ΣK² = 422 139）⇒ 按桶成组把同样的窗口数变成 N/val_batch 批。
    这里钉两件事：**来源仍是计划**（纯函数、可复现），且**每一批同网格身份**（可同批的前提）。
    """
    source = _val_source(monkeypatch, n_windows=8)
    plan = source.plan()
    assert plan.epoch == 0
    assert source.windows() == 8
    taken = _take(source)
    assert len(taken) == 8
    assert len(set(taken)) == 8
    allowed = {int(value) for value in plan.order}
    assert set(taken) <= allowed, "val 窗口必须来自本 epoch 的计划"
    dataset = source._dataset
    for batch in source.batches():
        keys = {dataset.grid_key(int(index)) for index in batch}
        assert len(keys) == 1, f"同一 val 批必须同网格身份，得到 {keys}"


def test_val_batch_is_bitwise_identical_across_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """T1 上半：两次调用给出**逐位一致**的一批（同一批被重放，而不是每次重抽）。"""
    source = _val_source(monkeypatch)
    first = _take(source)
    second = _take(source)
    assert first == second
    assert len(first) == 8
    assert len(set(first)) == 8, "前 N 个槽位互不相同（计划层是排列）"


def test_val_prefix_is_reproducible_across_instances(monkeypatch: pytest.MonkeyPatch) -> None:
    """另一个实例（同 cfg / 同 seed）给出同一批 ⇒ 续训后 val 前缀可复现（A1①）。"""
    assert _take(_val_source(monkeypatch, seed=3)) == _take(_val_source(monkeypatch, seed=3))


def test_val_seed_changes_the_prefix_but_not_the_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """反向对照（防「恒真」）：换 seed 必须换一批（否则上面那条测的是「总是同一批」）。"""
    other = _take(_val_source(monkeypatch, seed=17))
    assert len(other) == 8
    assert other != _take(_val_source(monkeypatch, seed=3))


def test_val_workers_do_not_change_the_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """T1 下半：`data.workers=0` 与 `=2` **逐位一致**（真实 spawn；A7 允许 val 走 worker）。"""
    sync = _take(_val_source(monkeypatch, workers=0))
    parallel = _take(_val_source(monkeypatch, workers=2))
    assert sync == parallel


def test_val_batch_size_follows_optim_val_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """批大小由 `optim.val_batch` 决定（不再是训练 `batch_size`=1 的逐窗前向）。"""
    source = _val_source(monkeypatch, n_windows=8)
    sizes = [len(batch) for batch in source.batches()]
    assert sum(sizes) == 8
    assert max(sizes) <= 8
    assert len(sizes) < 8, "按桶成组后批数必须少于窗口数（否则等于逐窗前向，分组没生效）"


def test_smaller_val_batch_makes_more_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    """反向对照（防「恒真」）：调小 `val_batch` 必须**真的**变成更多批。"""
    wide = _val_source(monkeypatch, n_windows=64)
    narrow = _val_source(monkeypatch, n_windows=64)
    narrow.cfg = replace(narrow.cfg, optim=replace(narrow.cfg.optim, val_batch=1))
    wide_sizes = [len(batch) for batch in wide.batches()]
    narrow_sizes = [len(batch) for batch in narrow.batches()]
    assert max(wide_sizes) > max(narrow_sizes), "val_batch 必须真的改变批大小"
    assert set(narrow_sizes) == {1}


def test_empty_val_split_is_disabled_not_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    """空 split（清单里没有留出集）⇒ `windows() == 0`，训练入口据此**关闭 val 并告警**。

    来历：真实数据通路的集成夹具里 `"val": []`（只有 train 行），而 `plan_epoch` 对空索引
    直接报错 ⇒ 一个「没有留出集」的环境事实会让整轮训练崩掉。降级必须是**显式**的：
    `train()` 打 WARNING 并把 val 关掉，而不是静默跳过。
    """
    from _plan_worker_stub import IndexDataset

    monkeypatch.setattr(dataset_mod, "collate_field_batch", _stub_collate)
    source = ManifestValSource(_config(), seed=3, n_windows=8)
    source._dataset = IndexDataset(())  # type: ignore[assignment]
    assert source.windows() == 0
    assert list(source.batches()) == []


def test_val_windows_are_clipped_to_what_the_buckets_can_give(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """请求数超过「每桶最多 val_batch 个 × 桶数」时取满即可（不报错，也不重复取窗口）。"""
    source = _val_source(monkeypatch, n_windows=10**6)
    plan = source.plan()
    taken = _take(source)
    per_bucket = Counter(int(value) for value in plan.bucket_id)
    expected = sum(min(8, count) for count in per_bucket.values())
    assert len(taken) == expected
    assert len(set(taken)) == len(taken)
    assert len(taken) <= len(plan.order)


def test_val_source_is_resume_neutral_in_the_fingerprint() -> None:
    """val 的节奏与规模**不进续训指纹**（它们不改变被训练的东西，见 RESUME_IGNORED_KEYS）。"""
    assert config_fingerprint(_config(optim={"val_windows": 128})) == config_fingerprint(
        _config(optim={"val_windows": 8})
    )
    assert config_fingerprint(_config(optim={"val_every": 1})) == config_fingerprint(
        _config(optim={"val_every": 1000})
    )
    # 反向对照：真正的语义字段仍然参与指纹
    assert config_fingerprint(_config(optim={"lr": 1e-4})) != config_fingerprint(_config())


def test_val_windows_must_be_positive() -> None:
    with pytest.raises(ConfigError, match="val_windows"):
        _config(optim={"val_windows": 0})
    assert _config().optim.val_windows == 128, "A6 的默认值是 128"


def test_resolve_val_source_skips_synthetic_and_disabled() -> None:
    """来源选择是显式的：合成来源**不冒充**留出集；val_every=0 是显式关闭。"""
    cfg = _config(optim={"val_every": 0})
    assert train_loop_mod._resolve_val_source(cfg, SmokeBatchSource(cfg, seed=0), None) is None
    enabled = _config(optim={"val_every": 10})
    assert (
        train_loop_mod._resolve_val_source(enabled, SmokeBatchSource(enabled, seed=0), None) is None
    )
    manifest = ManifestBatchSource(enabled, split="train", seed=0)
    resolved = train_loop_mod._resolve_val_source(enabled, manifest, None)
    assert isinstance(resolved, ManifestValSource)
    # 显式给出的 val 来源优先（测试与将来的消融臂用）
    explicit = ManifestValSource(enabled, seed=1, n_windows=4)
    assert train_loop_mod._resolve_val_source(enabled, manifest, explicit) is explicit


# ══════════════════════════════════════════════════════════════
# T3：分层把「空窗 / 非空窗」两个总体分开
# ══════════════════════════════════════════════════════════════

#: τ 格数与线数（τ 由契约派生，不得写死拍格宽）
_T_BINS = SUBDIVISIONS_PER_BEAT // 4
_K = 2
#: 常数强度：小到让「事件项」主导非空窗的损失（空窗只有积分项 ∫λdV ∝ λ）。
#: 真实语料里两个总体的中位差是**五个数量级**（§9-46），这里用小幅合成批重现这个结构。
_LAMBDA = 1e-6


def _two_population_batch(*, events: int = 2):
    """一个批里放**一个空窗 + 一个非空窗**（T3 的构造）。

    遮盖**手工**指定（只遮掉第 0 个窗口的一个事件）而不是走
    `build_occlusion_batch`：事件级遮盖在只有 2 个事件时很容易把两个都遮掉，
    那样批的 r == 1，`masked_poisson_loss` 会（正确地）拒绝它。手工构造让
    「一个空窗 + 一个非空窗且 r = 0.5」这个局面**确定**成立。
    """
    grid = make_grid(t_bins=_T_BINS, x_bins=8)
    tap = CHANNEL_INDEX[NoteType.TAP]
    counts = torch.zeros((2, _K, _T_BINS, 8, 2, 5), dtype=torch.int16)
    # 第 0 个窗口：两个格子、每格 5 个事件（桶内计数按 n_j 计，不去重）
    counts[0, 0, 0, 0, 0, tap] = 5
    counts[0, 0, 1, 0, 0, tap] = 5
    # 第 1 个窗口：空窗（一个事件都没有）
    occlusion = torch.zeros_like(counts, dtype=torch.bool)
    occlusion[0, 0, 0, 0, 0, tap] = True  # 只遮一个格子 ⇒ 批级 r == 0.5（5 / 10 个事件）
    batch = make_batch(k=_K, grid=grid, batch=2, counts=counts, occlusion=occlusion, audio_dim=16)
    lam = torch.full(batch.batch_field_shape(), _LAMBDA)
    return batch, FieldOutput(lam=lam), lam


def test_stratified_step_loss_separates_the_two_populations() -> None:
    """**T3**：`loss_empty` 落在空窗上（恒等于积分项），`loss_nonempty` 落在非空窗上。"""
    batch, output, _lam = _two_population_batch()
    out = stratified_step_loss(output, batch)
    assert out["empty_share"] == pytest.approx(0.5)
    assert out["nonempty_share"] == pytest.approx(0.5)
    assert "loss_empty" in out
    assert "loss_nonempty" in out

    per_window = masked_poisson_loss(output, batch, reduction="none").sum(dim=1)
    integral = integral_term(output, batch).sum(dim=1)
    # 空窗的损失恒等于积分项（§9-46 的实测事实）
    assert float(per_window[1]) == pytest.approx(float(integral[1]), rel=1e-6)
    assert float(integral[1]) > 0.0
    assert out["loss_empty"] == pytest.approx(float(per_window[1]), rel=1e-6)
    assert out["loss_nonempty"] == pytest.approx(float(per_window[0]), rel=1e-6)
    # 两个总体相差三个数量级以上（真实语料是五个数量级）
    assert out["loss_nonempty"] > 1e3 * out["loss_empty"]


def test_the_mixed_step_loss_cannot_tell_the_two_apart() -> None:
    """单看 `loss` **做不到**：它是两个总体的混合值，落在两者之间、且不等于任何一支。"""
    batch, output, _lam = _two_population_batch()
    out = stratified_step_loss(output, batch)
    # 「一步的 loss」= 批内**逐窗口损失的平均**（reduction="mean" 的同一口径）：
    # 它是两个桶的混合值，落在两者之间 ⇒ 单看它无法分辨这一步抽到了哪一类窗口。
    mixed = float(masked_poisson_loss(output, batch, reduction="sum").item() / batch.batch_size())
    assert out["loss_empty"] < mixed < out["loss_nonempty"]
    assert mixed != pytest.approx(out["loss_empty"], rel=1e-3)
    assert mixed != pytest.approx(out["loss_nonempty"], rel=1e-3)
    # 而分层之后，「这一步来自哪个总体」是可判定的：空批只有 loss_empty
    empty_only = make_batch(
        k=_K,
        grid=make_grid(t_bins=_T_BINS, x_bins=8),
        counts=torch.zeros((1, _K, _T_BINS, 8, 2, 5), dtype=torch.int16),
        occlusion=torch.zeros((1, _K, _T_BINS, 8, 2, 5), dtype=torch.bool),
        audio_dim=16,
    )
    empty_out = stratified_step_loss(
        FieldOutput(lam=torch.full(empty_only.batch_field_shape(), _LAMBDA)), empty_only
    )
    assert empty_out["empty_share"] == 1.0
    assert "loss_nonempty" not in empty_out
    assert "loss_empty" in empty_out


def test_stratification_uses_the_training_loss_itself() -> None:
    """B3 的机器判据：分层取的是**训练损失本身**（逐窗口求和 == 全批 loss）。"""
    batch, output, _lam = _two_population_batch()
    per_window = masked_poisson_loss(output, batch, reduction="none").sum(dim=1)
    total = float(masked_poisson_loss(output, batch, reduction="sum").item())
    assert float(per_window.sum()) == pytest.approx(total, rel=1e-6)


def test_stratified_keys_have_tensorboard_tags() -> None:
    """jsonl 键与 TB 标签不许漂成两套：产出方的每个键都必须在 SCALAR_TAGS 里。"""
    batch, output, _lam = _two_population_batch()
    for key in stratified_step_loss(output, batch):
        assert key in SCALAR_TAGS, f"{key} 没有 TB 标签"
    accumulator = ValAccumulator()
    accumulator.add(
        masked_readout(batch, output.lam),
        nll_shuffled=1.0,
        nll_full_event=1.0,
        contrasts={"audio_zero": 1.0, "audio_perm": 1.0, "track_zero": 1.0},
    )
    for key in accumulator.scalars():
        assert key in SCALAR_TAGS, f"{key} 没有 TB 标签"


# ══════════════════════════════════════════════════════════════
# 训练循环接线：val 标量落盘 + val 不改变训练 + T4 的 best.pt 判据
# ══════════════════════════════════════════════════════════════


@dataclass(slots=True)
class _SyntheticValSource:
    """合成 val 来源（默认 CI 用）：每次重放**同一批**合成窗口。

    它**不是**真实留出集（合成来源没有 split 可言），只用于验证接线与逐位一致性；
    真实配置走 `ManifestValSource`。
    """

    cfg: Any
    seed: int = 7
    k_lines: int = 2
    calls: int = 0

    def batches(self) -> Iterator[Any]:
        self.calls += 1
        yield SmokeBatchSource(self.cfg, seed=self.seed, k_lines=self.k_lines).batch(masked=True)

    def describe(self) -> str:
        return f"synthetic-val(stub, seed={self.seed})"

    def windows(self) -> int:
        return 1


def _artifacts(tmp_path: Path) -> RunArtifacts:
    return RunArtifacts.create(runs_dir=tmp_path, experiment="val-fixture", timestamp="T")


def _rows(artifacts: RunArtifacts) -> list[dict[str, float]]:
    return [
        json.loads(line) for line in (artifacts.logs / HISTORY_FILENAME).read_text().splitlines()
    ]


def test_train_writes_val_scalars_on_val_steps_only(tmp_path: Path) -> None:
    """A9：val 指标走 `_flush_scalars` 那条路，落在**该步那一行**里；其它步没有 val 字段。"""
    cfg = _config(optim={"max_steps": 4, "val_every": 2})
    artifacts = _artifacts(tmp_path)
    source = SmokeBatchSource(cfg, seed=0, k_lines=2)
    val = _SyntheticValSource(cfg)
    train(
        cfg,
        source=source,
        artifacts=artifacts,
        data_rev="rev-1",
        gates_green=True,
        val_source=val,
    )
    rows = _rows(artifacts)
    assert [int(row["step"]) for row in rows] == [1, 2, 3, 4]
    val_steps = [int(row["step"]) for row in rows if "val_nll" in row]
    assert val_steps == [2, 4]
    assert val.calls == 2
    for row in rows:
        if int(row["step"]) in val_steps:
            assert row["val_ratio"] > 0.0
            assert row["val_time_s"] > 0.0
            assert row["val_windows"] == 1.0
            assert "val_nll_constant" in row
            assert "cond_audio_zero_delta" in row
            assert "cond_audio_perm_delta" in row
            assert "cond_track_zero_delta" in row
            assert "val_nll_shuffled" in row
            assert "val_nll_full_event" in row, "§9-47 B 的附带读数（不可直接比）也要落盘"
        else:
            assert "val_nll" not in row
            assert "val_time_s" not in row
    # 分层与积分比在**每一步**都有（B1/B2：量具修复，不进 val）
    for row in rows:
        assert "integral_per_event" in row
        assert row["empty_share"] + row["nonempty_share"] == pytest.approx(1.0)


def test_val_does_not_change_the_training_trajectory(tmp_path: Path) -> None:
    """§9-47 G-④ 的机器判据：带 val 与不带 val 的两轮训练，损失序列**逐位一致**。

    为什么必须钉死：val 一旦污染 RNG（dropout 流）或优化器状态，它就从量具变成了训练信号，
    而那种污染在曲线上完全看不出来（只是"曲线略有不同"）。
    """
    cfg = _config(optim={"max_steps": 4, "val_every": 1})
    without = _artifacts(tmp_path / "without")
    with_val = _artifacts(tmp_path / "with")
    train(
        cfg,
        source=SmokeBatchSource(cfg, seed=0, k_lines=2),
        artifacts=without,
        data_rev="rev-1",
        gates_green=True,
    )
    train(
        cfg,
        source=SmokeBatchSource(cfg, seed=0, k_lines=2),
        artifacts=with_val,
        data_rev="rev-1",
        gates_green=True,
        val_source=_SyntheticValSource(cfg),
    )
    losses_a = [row["loss"] for row in _rows(without)]
    losses_b = [row["loss"] for row in _rows(with_val)]
    assert losses_a == losses_b, "val 改变了训练轨迹：它已经不是量具了"
    assert all("val_nll" in row for row in _rows(with_val))


def test_keep_best_follows_val_ratio_not_training_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**T4**：`best.pt` 按 `val/ratio` 选——训练损失最优的那一步**不是**它。"""
    cfg = _config(
        optim={"max_steps": 6, "val_every": 2},
        run={"save_every": 2, "keep_last": 5, "log_every": 1},
    )
    artifacts = _artifacts(tmp_path)
    # 剧本：第 2 / 4 / 6 步的 val/ratio = 0.9 / 0.5 / 0.7（最优在第 4 步）
    scripted = [0.9, 0.5, 0.7]
    calls: list[int] = []

    def _fake_evaluate(
        model: Any, val_source: Any, **kwargs: Any
    ) -> tuple[dict[str, float], float]:
        index = len(calls)
        calls.append(index)
        ratio = scripted[index] if index < len(scripted) else scripted[-1]
        return {"val_nll": ratio * 10.0, "val_nll_constant": 10.0, "val_ratio": ratio}, ratio

    monkeypatch.setattr(train_loop_mod, "evaluate_val", _fake_evaluate)
    train(
        cfg,
        source=SmokeBatchSource(cfg, seed=0, k_lines=2),
        artifacts=artifacts,
        data_rev="rev-1",
        gates_green=True,
        val_source=_SyntheticValSource(cfg),
    )
    assert calls == [0, 1, 2], "val 应当在第 2/4/6 步各跑一次"
    rows = _rows(artifacts)
    assert [row["val_ratio"] for row in rows if "val_ratio" in row] == scripted
    best = artifacts.checkpoints / BEST_CHECKPOINT_NAME
    assert best.is_file(), "有 val 时 best.pt 必须由 val/ratio 产生"
    loaded = load_checkpoint(best, cfg=cfg, data_rev="rev-1")
    assert loaded.meta.step == 4, "best.pt 必须落在 val/ratio 最优的那一步"
    # 反向对照：训练损失最优步**不是**第 4 步（否则本测试是恒真的）
    loss_argmin = min(rows, key=lambda row: row["loss"])
    assert int(loss_argmin["step"]) != 4


def test_val_improvements_write_best_only_when_they_improve(tmp_path: Path) -> None:
    """单调变差的 val/ratio 只写一次 best.pt（第一次），不反复覆盖。"""
    cfg = _config(optim={"max_steps": 4, "val_every": 1}, run={"save_every": 0, "log_every": 1})
    artifacts = _artifacts(tmp_path)
    ratios = [1.0, 1.1, 1.2, 1.3]
    calls: list[int] = []

    def _fake_evaluate(
        model: Any, val_source: Any, **kwargs: Any
    ) -> tuple[dict[str, float], float]:
        index = len(calls)
        calls.append(index)
        return {"val_ratio": ratios[index]}, ratios[index]

    original_save_best = train_loop_mod._save_best
    original_evaluate = train_loop_mod.evaluate_val
    steps: list[int] = []

    def _spy(*args: Any, **kwargs: Any) -> Any:
        steps.append(int(kwargs["step"]))
        return original_save_best(*args, **kwargs)

    train_loop_mod._save_best = _spy  # type: ignore[assignment]
    train_loop_mod.evaluate_val = _fake_evaluate  # type: ignore[assignment]
    try:
        train(
            cfg,
            source=SmokeBatchSource(cfg, seed=0, k_lines=2),
            artifacts=artifacts,
            data_rev="rev-1",
            gates_green=True,
            val_source=_SyntheticValSource(cfg),
        )
    finally:
        train_loop_mod._save_best = original_save_best  # type: ignore[assignment]
        train_loop_mod.evaluate_val = original_evaluate  # type: ignore[assignment]
    assert steps == [1], f"只有第一次 val 是最优，却写了 {steps}"
