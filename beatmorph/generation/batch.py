"""FieldBatch / FieldOutput 契约与张量形状冻结（Plan 04 §3.1/§3.2，里程碑 M1）。

本模块只描述**形状、语义与断言**，不含任何算法。三个 mask 的语义分离在
`beatmorph/field/`（plan 03 §4.4）已定义，本模块只**携带**并在 `assert_shapes` 中
强制其形状与不可混用：

| mask | 形状 | 含义 | **不得**当作 |
|---|---|---|---|
| `occlusion` | `(B, K, T, X, S, C)` bool | 训练遮盖通道：**True = 被遮盖（待补全）** | 有效性 mask |
| `line_mask` | `(B, K)` bool | 该 batch 内真实存在的判定线（padding 为 False） | 参与 `int lambda` 的求和 |
| `range_mask` | `(X,)` bool | `|x| <= 675` 的定义域 | 钳位（它不是钳位，只是定义域） |

时间轴 `T` 是 **tau 格数**（beat-aligned，1/48 拍；Q15 / RFC-0029 §3.1），
**不是音频帧轴**（`T_audio` 才是；两者各自由自己的派生量给出，禁止互相推导）。
本模块**不实现任何秒 <-> tau 换算**——那是 `beatmorph/field/` 的独占职责（CLAUDE.md
红线 7）；`grid` 字段是**测度与换算的唯一来源**，字段本身只是把 plan 03 的对象
带进 batch。

不引入 torch 之外的依赖；形状断言全部可在**无权重、无 GPU**的默认 CI 中运行。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from beatmorph.core.contracts.field import FIELD_SHAPE
from beatmorph.core.contracts.phigros import (
    RPE_EXTENDED_TRACKS,
    RPE_NORMAL_EVENT_LAYERS,
    RPE_TRACK_FIELDS,
)
from beatmorph.field.grid import FieldGrid

if TYPE_CHECKING:
    from collections.abc import Mapping

#: 普通事件轨的通道数 = len(RPE_TRACK_FIELDS) = 5；**跨 4 层求和之后**才是这个维度
#: （plan 04 §4.1；A 级依据：判定线的普通事件值 = 各层事件值之和）。
N_ORDINARY_TRACKS: int = len(RPE_TRACK_FIELDS)
#: 禁忌维度：逐层送入会让普通轨变成 4 x 5 = 20，是「跨层求和没做」的静默失效点。
FORBIDDEN_LAYERED_TRACKS: int = RPE_NORMAL_EVENT_LAYERS * len(RPE_TRACK_FIELDS)
#: extended（第 5 层）默认通道数（plan 04 §9-12）；text 只作存在性标记，不进 v1 张量。
DEFAULT_EXTENDED_TRACKS: int = len(RPE_EXTENDED_TRACKS) - 1

#: 场张量的维度序（**唯一**写法，来自 core/contracts/field.py）
FIELD_DIM_NAMES: tuple[str, ...] = ("batch", "k", "t", "x", "s", "c")


def _is_bool_tensor(tensor: Tensor) -> bool:
    return tensor.dtype == torch.bool


@dataclass(frozen=True, slots=True)
class FieldBatch:
    """一次前向所需的全部输入（Plan 04 §3.1）。

    Attributes:
        audio_emb: `(B, T_audio, D_audio)` Stage 0 输出；T_audio 的帧率由 config 派生。
        frame_rate: 音频帧率（Hz，**派生量**，G4 门禁校验）；只服务音频帧轴。
        line_tracks: `(B, K, T_line, N_ORDINARY_TRACKS)` **跨层求和后**的普通事件轨；
            时间轴是 tau（经 plan 03 的换算对齐，本模块不自行换算）。
        line_mask: `(B, K)` bool，padding / 无效线为 False。
        difficulty: `(B,)` float32 定数 difficulty（**不是** level 字符串）。
        grid: 已绑定 tau 轴的 FieldGrid；测度 dV_j、|Omega| 与 tau 网格的唯一来源。
        counts: `(B, K, T, X, S, C)` 桶内计数**目标**（plan 03 产出）；推理时为 None。
        occlusion: `(B, K, T, X, S, C)` bool，**True = 被遮盖（待补全）**；None 视为全 False。
        range_mask: `(X,)` bool；None 时由 grid.range_mask() 派生（不是钳位）。
        extended_tracks: `(B, K, T_line, F_ext)` 可选的第 5 层通道（plan 04 §9-12）。
    """

    audio_emb: Tensor
    frame_rate: float
    line_tracks: Tensor
    line_mask: Tensor
    difficulty: Tensor
    grid: FieldGrid
    counts: Tensor | None = None
    occlusion: Tensor | None = None
    range_mask: Tensor | None = None
    extended_tracks: Tensor | None = None

    # ── 便捷访问（不改变数据）───────────────────────────────────
    def n_lines(self) -> int:
        """K（判定线条数，运行期可变；**不进入任何输出层维度**）。"""
        return int(self.line_mask.shape[1])

    def batch_size(self) -> int:
        """B。"""
        return int(self.line_mask.shape[0])

    def cell_shape(self) -> tuple[int, ...]:
        """场张量的 (T, X, S, C)。"""
        return (
            self.grid.t_bins,
            self.grid.x_bins,
            self.grid.sides,
            self.grid.channels,
        )

    def field_shape(self) -> tuple[int, ...]:
        """(K, T, X, S, C)（单样本口径，与 plan 03 一致）。"""
        return (self.n_lines(), *self.cell_shape())

    def batch_field_shape(self) -> tuple[int, ...]:
        """(B, K, T, X, S, C)（本模块口径）。"""
        return (self.batch_size(), *self.field_shape())

    def line_mask_bool(self) -> Tensor:
        """line_mask 的 bool 视图（就地校验 dtype）。"""
        if not _is_bool_tensor(self.line_mask):
            raise AssertionError(f"line_mask 必须是 bool，得到 {self.line_mask.dtype}")
        return self.line_mask

    def range_mask_bool(self) -> Tensor:
        """range_mask（None 时从 FieldGrid 派生；**不是钳位**）。"""
        if self.range_mask is None:
            values = np.asarray(self.grid.range_mask(), dtype=np.bool_)
            return torch.as_tensor(values, dtype=torch.bool, device=self.line_mask.device)
        if not _is_bool_tensor(self.range_mask):
            raise AssertionError(f"range_mask 必须是 bool，得到 {self.range_mask.dtype}")
        return self.range_mask

    def occlusion_bool(self) -> Tensor:
        """遮盖通道（None -> 全 False = 无遮盖）。"""
        if self.occlusion is None:
            return torch.zeros(
                self.batch_field_shape(), dtype=torch.bool, device=self.line_mask.device
            )
        if not _is_bool_tensor(self.occlusion):
            raise AssertionError(f"occlusion 必须是 bool，得到 {self.occlusion.dtype}")
        return self.occlusion

    def observed_counts(self) -> Tensor:
        """模型**输入侧**的可见场 = counts * (~occlusion)（float32）。

        counts 为 None（推理）时返回全 0；此时可见场由调用方经
        `MaskedFieldModel.forward_state` 提供（迭代解码的中间状态）。
        """
        if self.counts is None:
            return torch.zeros(
                self.batch_field_shape(), dtype=torch.float32, device=self.line_mask.device
            )
        visible = self.counts.to(dtype=torch.float32)
        return visible * (~self.occlusion_bool()).to(dtype=visible.dtype)

    # ── 形状契约（M1：失败即抛，不得降级为日志）──────────────────
    def _assert_conditions(self, batch_size: int) -> None:
        """条件输入（音频 / 事件轨 / 线掩码 / 定数）的形状与 batch 一致性。"""
        if self.audio_emb.dim() != 3:
            raise AssertionError(
                f"audio_emb 必须是 (B, T_audio, D)，得到 {tuple(self.audio_emb.shape)}"
            )
        if self.line_tracks.dim() != 4:
            raise AssertionError(
                f"line_tracks 必须是 (B, K, T_line, F)，得到 {tuple(self.line_tracks.shape)}",
            )
        tracks = int(self.line_tracks.shape[-1])
        if tracks == FORBIDDEN_LAYERED_TRACKS:
            raise AssertionError(
                "line_tracks 的通道数是 "
                f"{FORBIDDEN_LAYERED_TRACKS} = {RPE_NORMAL_EVENT_LAYERS} x {N_ORDINARY_TRACKS}："
                "普通事件轨必须**先跨层求和**再送入编码器（plan 04 §4.1），"
                "逐层送入会静默丢掉求和语义",
            )
        if tracks != N_ORDINARY_TRACKS:
            raise AssertionError(
                f"line_tracks 的普通轨通道数必须是 {N_ORDINARY_TRACKS}，得到 {tracks}",
            )
        batch_size = int(self.audio_emb.shape[0])
        if self.line_mask.dim() != 2:
            raise AssertionError(f"line_mask 必须是 (B, K)，得到 {tuple(self.line_mask.shape)}")
        if int(self.line_mask.shape[0]) != batch_size:
            raise AssertionError(
                f"line_mask 的 B={self.line_mask.shape[0]} 与 audio_emb 的 B={batch_size} 不一致",
            )
        if int(self.line_tracks.shape[0]) != batch_size:
            raise AssertionError(
                f"line_tracks 的 B={self.line_tracks.shape[0]} 与 audio_emb 的 B={batch_size} 不一致",
            )
        if int(self.line_tracks.shape[1]) != self.n_lines():
            raise AssertionError(
                f"line_tracks 的 K={self.line_tracks.shape[1]} 与 line_mask 的 K={self.n_lines()} 不一致",
            )
        if self.difficulty.shape != (batch_size,):
            raise AssertionError(f"difficulty 必须是 (B,)，得到 {tuple(self.difficulty.shape)}")
        if self.extended_tracks is not None and (
            self.extended_tracks.shape[:3] != self.line_tracks.shape[:3]
        ):
            raise AssertionError(
                "extended_tracks 必须与 line_tracks 同 (B, K, T_line)，得到 "
                f"{tuple(self.extended_tracks.shape)} vs {tuple(self.line_tracks.shape)}",
            )

    def _assert_targets(self) -> None:
        """目标侧场（counts / occlusion）与网格的一致性。"""
        for name, tensor in (("counts", self.counts), ("occlusion", self.occlusion)):
            if tensor is None:
                continue
            if tuple(tensor.shape) != self.batch_field_shape():
                raise AssertionError(
                    f"{name} 必须是 (B, K, T, X, S, C)={self.batch_field_shape()}，"
                    f"得到 {tuple(tensor.shape)}",
                )
        if self.occlusion is not None and self.counts is None:
            raise AssertionError("给出了 occlusion 却没有 counts：遮盖通道的语义依赖桶内计数目标")

    def assert_shapes(self) -> None:
        """形状与 dtype 断言；失败即抛。

        覆盖：音频 3 维、判定线事件轨 4 维且**普通轨恰为 5**（跨层求和已完成）、
        line_mask/difficulty 的 batch 一致性、counts/occlusion 与 grid 的网格一致、
        extended 轨的通道一致性。
        """
        self.grid.assert_grid()
        if self.grid.t_bins <= 0 or not self.grid.bpm_points:
            raise AssertionError("FieldBatch 需要已绑定 tau 轴的 FieldGrid（t_bins > 0）")
        ranges = self.range_mask_bool()
        if ranges.shape != (self.grid.x_bins,):
            raise AssertionError(
                f"range_mask 必须是 (X,)={(self.grid.x_bins,)}，得到 {tuple(ranges.shape)}",
            )
        self._assert_conditions(self.batch_size())
        self._assert_targets()

    # ── 分批前向（plan 07 §9-15 的内存墙）─────────────────────────
    def slice_samples(self, start: int, stop: int) -> FieldBatch:
        """按**样本维**切一段子批次（门禁的 G2/G3 分批前向用）。

        只切 batch 维（dim 0）。`grid` / `frame_rate` / `range_mask` 是**逐批共享**的量，
        必须原样保留——它们若被切错，测度 `dV_j`、帧率契约或定义域会静默错位。

        Args:
            start: 起始样本下标（含）。
            stop: 结束样本下标（不含）。

        Raises:
            ValueError: 下标越界或 `start > stop`（宁可报错，不静默钳位）。
        """
        total = self.batch_size()
        lo, hi = int(start), int(stop)
        if not 0 <= lo <= hi <= total:
            raise ValueError(f"切片 [{lo}, {hi}) 越界（B={total}）")

        def cut_required(tensor: Tensor) -> Tensor:
            return tensor[lo:hi]

        def cut_optional(tensor: Tensor | None) -> Tensor | None:
            return None if tensor is None else tensor[lo:hi]

        sliced = FieldBatch(
            audio_emb=cut_required(self.audio_emb),
            frame_rate=self.frame_rate,
            line_tracks=cut_required(self.line_tracks),
            line_mask=cut_required(self.line_mask),
            difficulty=cut_required(self.difficulty),
            grid=self.grid,
            counts=cut_optional(self.counts),
            occlusion=cut_optional(self.occlusion),
            range_mask=self.range_mask,
            extended_tracks=cut_optional(self.extended_tracks),
        )
        sliced.assert_shapes()
        return sliced

    def to(self, device: torch.device | str) -> FieldBatch:
        """把批次的**张量**搬到设备上（门禁在 GPU 上跑真实批次的入口）。

        `grid` 是宿主侧对象（不搬）；`frame_rate` / `range_mask` 的语义与取值不变，
        需要时由 `range_mask_bool()` 按当前设备的 dtype 派生。
        """
        target = torch.device(device)

        def move_required(tensor: Tensor) -> Tensor:
            return tensor.to(target)

        def move_optional(tensor: Tensor | None) -> Tensor | None:
            return None if tensor is None else tensor.to(target)

        moved = FieldBatch(
            audio_emb=move_required(self.audio_emb),
            frame_rate=self.frame_rate,
            line_tracks=move_required(self.line_tracks),
            line_mask=move_required(self.line_mask),
            difficulty=move_required(self.difficulty),
            grid=self.grid,
            counts=move_optional(self.counts),
            occlusion=move_optional(self.occlusion),
            range_mask=move_optional(self.range_mask),
            extended_tracks=move_optional(self.extended_tracks),
        )
        moved.assert_shapes()
        return moved

    def split_samples(self, chunks: int) -> list[FieldBatch]:
        """把批次切成至多 `chunks` 段**非空**子批次（段数不超过 B）。

        段长尽量均分（余数分摊到前几段）；`chunks <= 1` 时返回单段 `[self]`。
        段内样本数之和恒等于 B，且拼接顺序与原批次一致（G2 的置换在切分**之前**完成）。
        """
        total = self.batch_size()
        count = max(1, min(int(chunks), total))
        if count == 1:
            return [self]
        base, remainder = divmod(total, count)
        parts: list[FieldBatch] = []
        start = 0
        for index in range(count):
            size = base + (1 if index < remainder else 0)
            parts.append(self.slice_samples(start, start + size))
            start += size
        return parts

    def describe(self) -> str:
        """一行诊断文本（训练日志用；不含任何权重）。"""
        grid = self.grid
        return (
            f"FieldBatch(B={self.batch_size()}, K={self.n_lines()}, T={grid.t_bins}, "
            f"X={grid.x_bins}, S={grid.sides}, C={grid.channels}, "
            f"T_audio={self.audio_emb.shape[1]}, frame_rate={self.frame_rate:.3f}Hz, "
            f"shape={FIELD_SHAPE})"
        )


@dataclass(frozen=True, slots=True)
class FieldOutput:
    """模型输出（Plan 04 §3.1）。

    Attributes:
        lam: `(B, K, T, X, S, C)` 强度 lambda >= 0（softplus / exp 参数化）。
        cum: `(B, K, T)` 累积强度 Lambda_k(tau)（因子化口径下非 None；单调不减）。
        cell_prob: `(B, K, T, X, S, C)` p_k(x, s, c | tau)，sum = 1（因子化口径）。
        loss: 标量训练损失（未提供目标时为 None）。
        diagnostics: 训练日志用的诊断张量（**不作为损失项**）。
    """

    lam: Tensor
    cum: Tensor | None = None
    cell_prob: Tensor | None = None
    loss: Tensor | None = None
    diagnostics: Mapping[str, Tensor] = field(default_factory=dict)

    def n_lines(self) -> int:
        """K。"""
        return int(self.lam.shape[1])

    def assert_shapes(self, batch: FieldBatch) -> None:
        """输出形状必须与 batch 的网格严格一致。"""
        expected = batch.batch_field_shape()
        if tuple(self.lam.shape) != expected:
            raise AssertionError(f"lam 必须是 {expected}，得到 {tuple(self.lam.shape)}")
        if self.cum is not None and tuple(self.cum.shape) != (
            batch.batch_size(),
            batch.n_lines(),
            batch.grid.t_bins,
        ):
            raise AssertionError(f"cum 形状错误：{tuple(self.cum.shape)}")
        if self.cell_prob is not None:
            if tuple(self.cell_prob.shape) != expected:
                raise AssertionError(
                    f"cell_prob 必须是 {expected}，得到 {tuple(self.cell_prob.shape)}"
                )
            totals = self.cell_prob.sum(dim=(3, 4, 5))
            if not bool(torch.allclose(totals, torch.ones_like(totals), atol=1e-4)):
                raise AssertionError("cell_prob 必须在 (X, S, C) 上归一化（sum = 1）")


#: 场维度序的断言用（对外暴露，便于测试引用）
__all__ = [
    "DEFAULT_EXTENDED_TRACKS",
    "FIELD_DIM_NAMES",
    "FORBIDDEN_LAYERED_TRACKS",
    "N_ORDINARY_TRACKS",
    "FieldBatch",
    "FieldOutput",
]
