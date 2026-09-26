"""冒烟批次来源：无权重 / 无网络 / 无 GPU 也能把门禁跑起来（plan 07 §4.3 / M7.2）。

门禁要证明的是「**通路是通的**」，不是「模型有多好」。因此这里造一个**可学**的合成任务：

- 谱面用真实的 `PhigrosChart` IR 表达（经 `build_target` 出桶内计数），
  走的是与真实数据**同一条**目标构建路径；
- 音频帧数由 `MERT_FRAME_RATE_HZ` **派生**（`round(时长 x 帧率)`），所以 G4 在这条路径上
  同样有内容；
- 打乱臂只置换目标（输入随之失去信息），与 plan 04 §9-17 的实测口径一致。

合成参数（BPM / 拍数 / 每拍 note 数）都是**超参**，不是物理常量：它们只影响合成谱本身。
"""

from __future__ import annotations

from dataclasses import dataclass

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
    "shuffled_counts",
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

    `batch(masked=True)`：遮盖补全训练路径（G1 用）；
    `batch(shuffled=True)`：目标被整体置换（G2 的对照臂用，输入随之失去信息）。
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

    def batch(
        self,
        *,
        masked: bool,
        shuffled: bool = False,
        samples: int | None = None,
    ) -> FieldBatch:
        """组装一个合成 batch（同 seed 逐位可复现）。

        Args:
            masked: 是否生成遮盖通道（G1 的遮盖补全路径）。
            shuffled: 是否置换目标（G2 的对照臂）。
            samples: 样本数；None 取 {BT}optim.batch_size{BT}。G2 需要较大的样本数
                （见 {BT}GatesConfig.shuffle_samples{BT} 的说明），否则对照会退化成背样本。
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
        stacked = torch.stack(counts, dim=0)
        if shuffled:
            stacked = shuffled_counts(stacked, seed=self.seed + 991)
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
        batch = FieldBatch(
            audio_emb=torch.stack(audio, dim=0),
            frame_rate=float(MERT_FRAME_RATE_HZ),
            line_tracks=torch.stack(tracks, dim=0),
            line_mask=line_mask,
            difficulty=torch.full((count,), 15.0, dtype=torch.float32),
            grid=grid,
            counts=stacked,
            occlusion=occlusion,
        )
        batch.assert_shapes()
        return batch

    def describe(self) -> str:
        """一行来源说明（落进 gates.txt 的上下文）。"""
        chart, grid = self._grid_and_chart()
        return (
            f"synthetic(infra.smoke)：K={len(chart.lines)}，notes={len(chart.notes)}，"
            f"T={grid.t_bins}，X={grid.x_bins}，时长={grid.total_seconds:.3f}s，seed={self.seed}"
        )


def shuffled_counts(counts: torch.Tensor, *, seed: int) -> torch.Tensor:
    """把目标按格整体置换（事件数不变；输入与目标的对应关系被破坏）。"""
    generator = torch.Generator().manual_seed(seed)
    flat = counts.reshape(int(counts.shape[0]), -1).clone()
    for row in range(int(flat.shape[0])):
        order = torch.randperm(int(flat.shape[1]), generator=generator)
        flat[row] = flat[row][order]
    return flat.reshape(tuple(counts.shape))


def spec_of(batch: FieldBatch) -> ChartFieldSpec:
    """批次对应的场网格规格（诊断用）。"""
    return batch.grid.spec(batch.n_lines())
