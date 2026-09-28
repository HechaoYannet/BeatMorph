"""冒烟批次来源：无权重 / 无网络 / 无 GPU 也能把门禁跑起来（plan 07 §4.3 / M7.2）。

门禁要证明的是「**通路是通的**」，不是「模型有多好」。因此这里造一个**可学**的合成任务：

- 谱面用真实的 `PhigrosChart` IR 表达（经 `build_target` 出桶内计数），
  走的是与真实数据**同一条**目标构建路径；
- 音频帧数由 `MERT_FRAME_RATE_HZ` **派生**（`round(时长 x 帧率)`），所以 G4 在这条路径上
  同样有内容；
- **条件必须携带目标的信息**（plan 04 §9-15）：随机噪声上的事件与条件独立时，模型无从
  学起，门禁变成空转。因此这里把合成谱的**事件密度**写进 `audio_emb` 的第 0 维
  （每个音频帧对应的 τ 区间内的 event 数）——模型能据此定位事件。

（2026-09-28，RFC-0037：门禁 G2 已删除，本模块不再提供打乱臂；`shuffle_counts_within_line`
是给 **val 的置换对照**（`val/nll_shuffled_delta`）用的线内置换，与被删的全局置换相比
保持每条线的事件数不变——见 RFC-0036 §2.5 的 nuisance 实证。）

合成参数（BPM / 拍数 / 每拍 note 数）都是**超参**，不是物理常量：它们只影响合成谱本身。
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace

import torch

from beatmorph.core.contracts.field import ChartFieldSpec
from beatmorph.core.contracts.phigros import (
    SUBDIVISIONS_PER_BEAT,
    BpmPoint,
    ChartMeta,
    ChartSource,
    JudgeLine,
    PhigrosChart,
    PhigrosNote,
    side_from_above,
)
from beatmorph.core.contracts.tensors import MERT_FRAME_RATE_HZ
from beatmorph.core.logging import get_logger
from beatmorph.data.tracks import line_tracks_tensor
from beatmorph.field.grid import FieldGrid, tau_to_seconds
from beatmorph.field.target import build_target
from beatmorph.generation.batch import FieldBatch
from beatmorph.generation.masks import build_occlusion_batch
from beatmorph.infra.config.schema import TrainConfig

__all__ = [
    "DEFAULT_SMOKE_BEATS",
    "DEFAULT_SMOKE_BPM",
    "SmokeBatchSource",
    "event_density_signal",
    "shuffle_counts_within_line",
    "spec_of",
    "synthetic_chart",
]

logger = get_logger("infra.smoke")

#: 合成谱的默认 BPM（**超参**，不是物理常量）
DEFAULT_SMOKE_BPM: float = 120.0
#: 合成谱的默认拍数
DEFAULT_SMOKE_BEATS: float = 8.0
#: 每拍放几个 note（合成参数；越小批次越轻）
DEFAULT_NOTES_PER_BEAT: int = 4


def synthetic_chart(
    *,
    k_lines: int,
    beats: float = DEFAULT_SMOKE_BEATS,
    bpm: float = DEFAULT_SMOKE_BPM,
    notes_per_beat: int = DEFAULT_NOTES_PER_BEAT,
    difficulty: float = 15.0,
    seed: int = 0,
) -> PhigrosChart:
    """造一张**可学**的合成谱（每条线在 τ 轴上按固定间隔落点，x 随机）。

    Args:
        k_lines: 判定线条数（>= 1）。
        beats: 谱面长度（拍）。
        bpm: 单一 BPM 段（合成参数）。
        notes_per_beat: 每拍 note 数（决定事件密度）。
        difficulty: 定数（条件输入）。
        seed: x 落点的随机种子（**确定性**：同 seed 同谱）。

    Returns:
        PhigrosChart（秒域时间；BPMList 是 τ 换算的唯一依据）。
    """
    if k_lines < 1:
        raise ValueError(f"k_lines 必须 >= 1，得到 {k_lines}")
    bpm_points = [BpmPoint(time_beats=0.0, bpm=bpm)]
    step = max(1, SUBDIVISIONS_PER_BEAT // max(1, notes_per_beat))
    generator = torch.Generator().manual_seed(seed)
    notes: list[PhigrosNote] = []
    total_steps = int(beats * SUBDIVISIONS_PER_BEAT)
    for tau_index in range(0, total_steps, step):
        tau = tau_index / SUBDIVISIONS_PER_BEAT
        line_id = (tau_index // step) % k_lines
        x_index = int(torch.randint(0, 17, (1,), generator=generator).item()) - 8
        # x 只由契约的舞台半宽与一个小整数派生（不写字面量坐标）
        position_x = x_index * 40.0
        notes.append(
            PhigrosNote(
                line_id=line_id,
                t=float(tau_to_seconds(tau, bpm_points)),
                position_x=position_x,
                side=side_from_above(1),
                type=int(_tap()),
                hold_time=0.0,
                is_fake=False,
                above_raw=1,
                type_raw=int(_tap()),
                is_fake_raw=0,
            ),
        )
    return PhigrosChart(
        lines=[JudgeLine(line_id=index) for index in range(k_lines)],
        notes=notes,
        bpm_points=bpm_points,
        meta=ChartMeta(chart_time_s=float(tau_to_seconds(beats, bpm_points))),
        source=ChartSource(sniff_evidence="synthetic:infra.smoke"),
    )


def _tap() -> int:
    from beatmorph.core.contracts.phigros import NoteType

    return int(NoteType.TAP)


@dataclass(slots=True)
class SmokeBatchSource:
    """合成批次来源（`BatchSource` 协议的一个实现）。

    `batch(masked=True)`：遮盖补全训练路径（G1 用）；`batch(masked=False)`：全事件口径（G3 用）。
    """

    cfg: TrainConfig
    seed: int = 0
    #: 合成谱的判定线条数（受模型 k_max 限制）
    k_lines: int = 2
    #: 合成谱长度（拍）
    beats: float = DEFAULT_SMOKE_BEATS
    #: 合成谱 BPM
    bpm: float = DEFAULT_SMOKE_BPM
    #: 每拍 note 数
    notes_per_beat: int = DEFAULT_NOTES_PER_BEAT

    def _grid_and_chart(self, *, seed_offset: int = 0) -> tuple[PhigrosChart, FieldGrid]:
        chart = synthetic_chart(
            k_lines=min(self.k_lines, self.cfg.model.k_max),
            beats=self.beats,
            bpm=self.bpm,
            notes_per_beat=self.notes_per_beat,
            seed=self.seed + seed_offset,
        )
        grid = FieldGrid(x_bins=self.cfg.data.x_bins).for_chart(chart)
        grid.assert_grid()
        return chart, grid

    def batches(
        self, *, start_step: int = 1, first: FieldBatch | None = None
    ) -> Iterator[FieldBatch]:
        """合成来源没有计划层：逐批调用 `batch`（同 seed 逐位一致）。

        `start_step` / `first` 只为满足 `BatchSource` 协议：合成批次由 `seed + 批内下标`
        决定，与步号无关，因此不需要定位。
        """
        del start_step
        if first is not None:
            yield first
        while True:
            yield self.batch(masked=True)

    def batch(
        self,
        *,
        masked: bool,
        samples: int | None = None,
        min_events: int = 0,
    ) -> FieldBatch:
        """组装一个合成 batch（同 seed 逐位可复现）。

        Args:
            masked: 是否生成遮盖通道（G1 的遮盖补全路径）。
            samples: 样本数；None 取 {BT}optim.batch_size{BT}。
            min_events: 门禁要求的最小事件数（合成谱本就密集，这里只做**断言**：
                满足不了说明合成参数被改坏了，必须报错而不是静默空过）。
        """
        count = max(1, int(self.cfg.optim.batch_size if samples is None else samples))
        counts: list[torch.Tensor] = []
        tracks: list[torch.Tensor] = []
        audio: list[torch.Tensor] = []
        grid: FieldGrid | None = None
        for index in range(count):
            chart, chart_grid = self._grid_and_chart(seed_offset=index)
            if grid is None:
                grid = chart_grid
            target = build_target(chart, chart_grid)
            target.assert_conservation()
            counts.append(target.to_tensor())
            tracks.append(line_tracks_tensor(chart, chart_grid))
            frames = round(chart_grid.total_seconds * MERT_FRAME_RATE_HZ)
            audio.append(
                torch.randn(
                    frames,
                    self.cfg.model.audio_dim,
                    generator=torch.Generator().manual_seed(self.seed + index),
                ),
            )
        assert grid is not None  # count >= 1
        original = torch.stack(counts, dim=0)
        # 条件必须携带目标的信息（plan 04 §9-15）：把事件密度写进 audio 的第 0 维。
        density = event_density_signal(original, int(audio[0].shape[0]))
        stacked = original
        n_lines = int(stacked.shape[1])
        line_mask = torch.ones(count, n_lines, dtype=torch.bool)
        occlusion = None
        if masked and self.cfg.data.occlusion_ratio > 0.0:
            occlusion, _ = build_occlusion_batch(
                stacked,
                ratio=self.cfg.data.occlusion_ratio,
                seed=self.seed,
                line_mask=line_mask,
            )
        audio_stack = torch.stack(audio, dim=0).clone()
        audio_stack[:, :, 0] = audio_stack[:, :, 0] + density
        batch = FieldBatch(
            audio_emb=audio_stack,
            frame_rate=float(MERT_FRAME_RATE_HZ),
            line_tracks=torch.stack(tracks, dim=0),
            line_mask=line_mask,
            difficulty=torch.full((count,), 15.0, dtype=torch.float32),
            grid=grid,
            counts=stacked,
            occlusion=occlusion,
        )
        batch.assert_shapes()
        if min_events > 0 and float(stacked.sum()) < float(min_events):
            raise ValueError(
                f"合成批只有 {float(stacked.sum())} 个事件 < 要求的 {min_events}："
                "合成参数（notes_per_beat / beats / k_lines）被改坏了",
            )
        return batch

    def describe(self) -> str:
        """一行来源说明（落进 gates.txt 的上下文）。"""
        chart, grid = self._grid_and_chart()
        return (
            f"synthetic(infra.smoke)：K={len(chart.lines)}，notes={len(chart.notes)}，"
            f"T={grid.t_bins}，X={grid.x_bins}，时长={grid.total_seconds:.3f}s，seed={self.seed}"
        )

    def coverage(self) -> dict[str, float]:
        """合成来源**没有有限数据集**，因此没有 epoch/覆盖率可言。

        显式返回 total=0（而不是伪造一个 1.0 的覆盖率）：训练循环据此跳过 coverage 标量，
        让「不知道」保持为不知道——合成冒烟本来也不能证明真实采样器走过数据（plan 07 §9-38）。
        """
        return {
            "epoch": 0.0,
            "windows_seen": 0.0,
            "windows_total": 0.0,
            "charts_seen": 0.0,
            "charts_total": 0.0,
        }


def shuffle_counts_within_line(batch: FieldBatch, *, seed: int) -> FieldBatch:
    """把**被遮盖格子**里的计数在「每条 (样本, 判定线) 各自的遮盖集合内」置换。

    RFC-0037 R5（吸收 RFC-0036 的 P1）：val 的置换对照（`val/nll_shuffled_delta`）用它。
    与被删除的全局置换（`shuffle_hidden_counts`）相比，**线内置换保持每条线的事件数不变**——
    全局置换会把事件从 1 条线摊到 17 条线，连「每线事件预算」这个 nuisance 一起改掉
    （RFC-0036 §2.5 实证：同一批 real 166.74 / 线内置换 257.61 / 全局置换 −26.49）。

    契约（进默认 CI，RFC-0037 §2.4）：

    1. 可见场 `counts·~occlusion` 逐位不变（模型输入不动）；
    2. **每线**事件数不变；
    3. 总事件数不变；
    4. 同 seed 同进程逐位可复现；
    5. 用独立 `torch.Generator`，不污染全局 RNG。

    Args:
        batch: 必须带 `counts` 与 `occlusion`（遮盖批次）。
        seed: 置换种子。

    Raises:
        ValueError: 缺 counts / occlusion（无遮盖就没有「待补全的标签」可置换）。
    """
    if batch.counts is None or batch.occlusion is None:
        raise ValueError("线内置换需要遮盖批次（counts + occlusion）：无遮盖时输入就是目标")
    occlusion = batch.occlusion_bool()
    counts = batch.counts.clone()
    generator = torch.Generator().manual_seed(int(seed))
    for row in range(int(counts.shape[0])):
        for line in range(int(counts.shape[1])):
            selector = occlusion[row, line]
            values = counts[row, line][selector]
            if int(values.numel()) <= 1:
                continue
            order = torch.randperm(int(values.numel()), generator=generator)
            counts[row, line][selector] = values[order]
    return replace(batch, counts=counts)


def event_density_signal(counts: torch.Tensor, frames: int) -> torch.Tensor:
    """每个音频帧对应的 τ 区间内的事件数 → `(B, frames)` 的密度信号。

    这是合成夹具的**条件**（写进 `audio_emb` 的第 0 维）：没有它，噪声上的事件与条件
    独立，模型无从学起、门禁空转（plan 04 §9-15）。
    """
    batch_size, t_bins = int(counts.shape[0]), int(counts.shape[1])
    per_tau = counts.to(dtype=torch.float32).sum(dim=(1, 3, 4, 5))
    index = (torch.arange(t_bins) * int(frames)) // max(t_bins, 1)
    density = torch.zeros(batch_size, int(frames), dtype=torch.float32)
    density.scatter_add_(1, index.unsqueeze(0).expand(batch_size, -1), per_tau)
    scale = per_tau.sum(dim=1, keepdim=True).clamp_min(1.0)
    return density / scale


def spec_of(batch: FieldBatch) -> ChartFieldSpec:
    """批次对应的场网格规格（诊断用）。"""
    return batch.grid.spec(batch.n_lines())
