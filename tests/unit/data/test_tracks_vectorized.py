"""`line_tracks_at` 的向量化实现必须与逐点 `sum_track` **逐位相等**（默认 CI，无权重 / 无 GPU）。

来历（2026-09-27）：真实窗口的 CPU 剖面里 `line_tracks_at` 占 **52.5%**（1.46e6 次
`track_value` / **3.97e7 次 `Beat.to_beats()`**，见 plan 07 §9-38）。修法是把
「列表序首个命中关键帧」改成 `searchsorted(starts) + prefix_max(ends)` 的向量化定位，
取值仍调用契约层的 `EventKeyframe.numeric_at`。实测该段 **32.9x**、每窗口 1.218 -> 0.582 s。

本文件的作用是**不许它静默变值**：判据是 `torch.equal`（逐位），不是 `allclose`。
覆盖到的边界情形（每一种都曾让朴素实现出错）：
  - 关键帧之间的**空隙**（`fill_gaps` 插入保持型常量事件）；
  - **重叠**的关键帧（契约取「列表序先出现的那个」，而非「start 最大的那个」）；
  - 非 Linear 缓动（29 种里的几个代表）与**贝塞尔**；
  - 非默认 `easingLeft/easingRight`（归一化切割分支）；
  - **多层事件轨**（跨层求和）与空轨（默认值 0.0）；
  - 求值点正好落在关键帧端点上。
"""

from __future__ import annotations

import numpy as np
import torch

from beatmorph.core.contracts import RPE_NORMAL_TRACKS, RPE_TRACK_FIELDS, ChartSource
from beatmorph.data.parsers.rpejson import parse_rpejson
from beatmorph.data.tracks import line_tracks_at
from tests.unit.data._helpers import build_rpe_bytes, keyframe

#: 拍 -> RPE beat 三元组（细分用契约常量，避免裸数字）
SUBDIVISION = 48


def _beat(value: float) -> list[int]:
    index = int(value // 1)
    return [index, round((value - index) * SUBDIVISION), SUBDIVISION]


def _legacy(chart, taus: np.ndarray) -> torch.Tensor:
    """旧实现（每个求值点线性扫过整条轨）——等价性的参照物。"""
    values = np.empty((len(chart.lines), len(taus), len(RPE_TRACK_FIELDS)), dtype=np.float32)
    for line_index, line in enumerate(chart.lines):
        for field_index, name in enumerate(RPE_TRACK_FIELDS):
            values[line_index, :, field_index] = np.asarray(
                [line.sum_track(name, float(tau)) for tau in taus], dtype=np.float32
            )
    return torch.from_numpy(values)


def _layered_track_frames() -> tuple[dict, list[float]]:
    """两层的 `moveXEvents`：覆盖空隙 / 重叠 / 非默认切割 / 端点，外加几个缓动代表。"""
    # 第 1 层：线性 + 空隙 + 重叠（0->4 与 3->6 重叠在 [3,4]）
    layer0 = [
        keyframe(_beat(0.0), _beat(4.0), 0.0, 8.0),
        keyframe(_beat(3.0), _beat(6.0), 20.0, 26.0),
        keyframe(_beat(9.0), _beat(10.0), -5.0, 5.0, easing_type=4),  # Out Quad
        keyframe(
            _beat(11.0),
            _beat(12.5),
            1.0,
            3.0,
            easing_type=2,  # Out Sine + 非默认切割（归一化分支）
            easingLeft=0.25,
            easingRight=0.75,
        ),
    ]
    # 第 2 层：弹性缓动 + 贝塞尔（都必须走标量回落路径）
    layer1 = [
        keyframe(_beat(0.5), _beat(2.5), 100.0, 140.0, easing_type=24),  # Out Elastic
        keyframe(
            _beat(13.0),
            _beat(14.0),
            0.0,
            1.0,
            bezier=1,
            bezierPoints=[0.42, 0.0, 0.58, 1.0],
        ),
    ]
    frames = [*layer0, *layer1]
    bounds = [0.0, 0.5, 2.5, 3.0, 4.0, 6.0, 9.0, 10.0, 11.0, 12.5, 13.0, 14.0]
    return frames, bounds


def _chart_with_tracks():
    frames, _ = _layered_track_frames()
    line = {
        "Name": "line-tracks",
        "notes": [],
        "eventLayers": [
            {"moveXEvents": [k for k in frames if k["startTime"][0] < 9]},
            {"moveXEvents": [k for k in frames if k["startTime"][0] >= 9]},
        ],
    }
    payload = build_rpe_bytes(lines=[line], chart_time=30.0)
    return parse_rpejson(payload, ChartSource(chart_id=1))


def test_vectorized_tracks_equal_scalar_reference_bitwise() -> None:
    """向量化取值 == 逐点 `sum_track`（逐位），含端点、空隙、重叠与缓动代表。"""
    chart = _chart_with_tracks()
    _, bounds = _layered_track_frames()
    taus = np.concatenate(
        [
            np.linspace(-1.0, 16.0, 4001),
            np.asarray(bounds, dtype=np.float64),  # 关键帧端点（边界归属口径）
            np.asarray([b + 1e-12 for b in bounds], dtype=np.float64),
            np.asarray([b - 1e-12 for b in bounds], dtype=np.float64),
        ]
    )
    fast = line_tracks_at(chart, taus)
    reference = _legacy(chart, taus)
    assert fast.shape == reference.shape
    assert torch.equal(fast, reference), (
        "向量化事件轨求值出现逐位差异——首个不等位置："
        f"{torch.nonzero(fast != reference)[:5].tolist()}"
    )


def test_vectorized_tracks_fall_back_for_non_linear_easing() -> None:
    """非 Linear（弹性 / 贝塞尔 / 非默认切割）必须走标量回落，且仍然逐位相等。"""
    chart = _chart_with_tracks()
    # 取只命中第 2 层（弹性 + 贝塞尔）的 tau
    taus = np.linspace(12.6, 14.2, 700)
    assert torch.equal(line_tracks_at(chart, taus), _legacy(chart, taus))


def test_empty_track_yields_default_zero() -> None:
    """空轨（字段缺失）取默认值 0.0，且与标量路径一致。"""
    line = {"Name": "empty", "notes": []}
    chart = parse_rpejson(build_rpe_bytes(lines=[line], chart_time=4.0), ChartSource(chart_id=2))
    taus = np.linspace(0.0, 4.0, 64)
    tracks = line_tracks_at(chart, taus)
    assert tuple(tracks.shape) == (1, 64, len(RPE_TRACK_FIELDS))
    assert torch.equal(tracks, _legacy(chart, taus))
    assert float(tracks.abs().max()) == 0.0


def test_track_channel_order_is_contract_order() -> None:
    """通道 j 必须等于 `RPE_TRACK_FIELDS[j]`（不写死顺序）：用可区分的常量值验证。"""
    layer = {
        rpe_name: [keyframe(_beat(0.0), _beat(2.0), float(index + 1), float(index + 1))]
        for index, rpe_name in enumerate(RPE_NORMAL_TRACKS)
    }
    chart = parse_rpejson(
        build_rpe_bytes(
            lines=[{"Name": "ordered", "notes": [], "eventLayers": [layer]}], chart_time=4.0
        ),
        ChartSource(chart_id=3),
    )
    tracks = line_tracks_at(chart, np.linspace(0.05, 1.95, 33))
    for field_index in range(len(RPE_TRACK_FIELDS)):
        assert torch.allclose(
            tracks[0, :, field_index], torch.full((33,), float(field_index + 1))
        ), f"通道 {field_index} 的顺序与 RPE_TRACK_FIELDS 不一致"
