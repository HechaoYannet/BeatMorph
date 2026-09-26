"""特征离线提取 + 配对切分（plan 02 §3.6 / M9、M10）。

**M9 特征离线提取**：复用 plan 01 的 MERT 编码器，把音频重采样到
`MERT_SAMPLE_RATE_HZ` → `encode` → 写 `<key>.npz` + `<key>.meta.json`，
**按音频内容 sha1 去重**（同曲多谱只提取一次）。缓存元数据是「25 Hz 事故」的防线
（docs/POSTMORTEM-2026-08-05-frame-rate-misalignment.md）：加载时逐项校验，任一项不符
即抛 :class:`FeatureCacheMismatchError`。

**M10 配对与切分**：`build_pairs` **按曲目**（`name | composer` 近似）切分
train/val/test，**同曲多谱必须落在同一 split**（否则 val 泄漏）；三份清单的曲目集合
两两不相交；另产出「同曲跨谱泛化」评测集（同曲不同难度）。

**关于 `datasets.Dataset`**：plan §3.6 的签名写 `-> datasets.Dataset`，但 `datasets`/`pyarrow`
属 `[project.optional-dependencies].train`（**默认 dev 环境不含**），而默认 CI 必须能跑
配对与切分。故本模块的规范返回是纯 Python 的 :class:`PairSplits`，清单以 **JSONL** 为规范
落盘格式（`.parquet` 为可选导出，需要 pyarrow）。此为记录在案的偏离。

**关于 `FeatureCacheMeta` 的归属**：plan 01 §3.3 把缓存契约放在
`beatmorph/audio/encoder/cache.py`（plan 01 的文件，本 plan 不得越权创建），而红线 2
禁止跨模块直连内部类型。故本模块自带一个**结构等价**的 meta 模型：键集合
（`rate/sample_rate/layer/model_rev/duration_s` + `original_sample_rate/feat_dim/dtype/adapter`）
与 plan 01 §3.3 的表逐条一致；待两侧合并时应上提到 `core/contracts`（需 RFC）。
"""

from __future__ import annotations

import random
import statistics
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from beatmorph.core.contracts import (
    MERT_DEFAULT_FEAT_DIM,
    MERT_FRAME_RATE_HZ,
    MERT_SAMPLE_RATE_HZ,
    ChartFormat,
)
from beatmorph.core.logging import get_logger
from beatmorph.data.phira.client import Manifest, read_manifest, sha1_file, sha1_hex
from beatmorph.data.phira.package import chart_dest_path
from beatmorph.data.qc import QcReport

logger = get_logger(__name__)

#: 特征数组（`[T_seq, feat]`）。
FeatureArray = NDArray[Any]

#: 运行期**动态**输入的类型别名（JSON 值、numpy/torch 张量、采样器回调）。
#:
#: 为什么不是 `object`：这些位置会直接调用 `int()`/`.shape` 或在两个第三方库之间转发，
#: 写成 `object` 只会把类型错误推到运行期，还会误述「编码器协议」的真实宽容度。
#: flake8-annotations 的 ANN401 不允许把 `Any` 直接写进签名，故此处集中别名一次并说明理由。
Dynamic = Any

#: plan 01 §3.3 的**六项校验**键（`CACHE_META_KEYS`）。
CACHE_META_KEYS: tuple[str, ...] = ("rate", "sample_rate", "layer", "model_rev", "duration_s")

#: 本模块落盘的完整元数据键（六项 + 重采样/形状/精度/适配器留痕）。
FEATURE_CACHE_META_KEYS: tuple[str, ...] = (
    *CACHE_META_KEYS,
    "original_sample_rate",
    "feat_dim",
    "dtype",
    "adapter",
)

#: 帧数一致性容差（MERT 卷积栈的边界效应；plan 01 §3.3 的 ±1 帧）。
FRAME_TOLERANCE: int = 1

#: 清单行/切分的规范后缀。
MANIFEST_SUFFIX: str = ".jsonl"

#: 调研 §7.3 实测基线：背面（`above != 1`）占比 2.4%–3.0%。
SURVEY_BACK_FRACTION_RANGE: tuple[float, float] = (0.024, 0.030)
#: 调研 §7.3 实测基线：Tap 占比 52%–63%。
SURVEY_TAP_FRACTION_RANGE: tuple[float, float] = (0.52, 0.63)
#: 调研 §7.1 实测基线：判定线数中位 25–30（n=23 时中位 30，另一批 10 张中位 ~27）。
SURVEY_LINES_MEDIAN_RANGE: tuple[float, float] = (25.0, 30.0)
#: 语料级基线比对的最小样本数（低于此数不做区间断言，避免拿 2 张谱下结论）。
MIN_CHARTS_FOR_CORPUS_CHECK: int = 5

#: 音频加载器：(路径) → (波形, 原始采样率)。
AudioLoader = Callable[[Path], tuple[Dynamic, int]]
#: 重采样器：(波形, 原始采样率, 目标采样率) → 波形。
Resampler = Callable[[Dynamic, int, int], Dynamic]


class FeatureCacheMismatchError(RuntimeError):
    """特征缓存元数据校验失败（六项之一不符，或形状/帧数与元数据不一致）。"""


class SplitError(ValueError):
    """切分不合法（比例非法 / 曲目集合相交 → val 泄漏）。"""


class FeatureEncoder(Protocol):
    """plan 01 `MertAudioEncoder` 与本模块测试用假编码器共同满足的最小协议。"""

    def encode(self, wav: Dynamic, sample_rate: int = ...) -> Dynamic:
        """编码单声道波形 → `[batch, time_seq, feat]` 或 `[time_seq, feat]`。"""
        ...

    def output_frame_rate(self) -> float:
        """输出帧率 Hz（**派生量**，必须来自主干 config）。"""
        ...


class FeatureCacheMeta(BaseModel):
    """特征缓存元数据（plan 01 §3.3 + 本模块补充的留痕字段）。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: 提取时的帧率，必须 == `MERT_FRAME_RATE_HZ`
    rate: float = Field(gt=0.0)
    #: 提取时的**模型**采样率，必须 == `MERT_SAMPLE_RATE_HZ`
    sample_rate: int = Field(gt=0)
    #: 取第几层 hidden state
    layer: int = Field(ge=0)
    #: 主干版本（换版本必须重抽，不得混用）
    model_rev: str = Field(min_length=1)
    #: 音频时长（秒），用于帧数一致性校验
    duration_s: float = Field(gt=0.0)
    #: 重采样**之前**的原始采样率（plan 01 §偏离 2：两个采样率都要记录）
    original_sample_rate: int = Field(gt=0)
    #: 特征维（MERT-v1-330M 各层均为 1024）
    feat_dim: int = Field(gt=0)
    #: 落盘精度（如 `float16` / `float32`）
    dtype: str = Field(min_length=1)
    #: Adapter 类型（`none` = 离线纯冻结直出）
    adapter: str = Field(min_length=1)

    def verify(
        self,
        emb: FeatureArray,
        *,
        expected_layer: int | None = None,
        expected_model_rev: str | None = None,
        expected_dtype: str | None = None,
        expected_adapter: str | None = None,
        frame_tolerance: int = FRAME_TOLERANCE,
    ) -> None:
        """逐项校验（**一次报出所有问题**，而不是首个失败就返回）。

        Raises:
            FeatureCacheMismatchError: 任一项不符。
        """
        problems: list[str] = []
        if emb.ndim != 2:
            raise FeatureCacheMismatchError(
                f"特征数组必须是 2 维 [T, feat]，得到 shape={emb.shape}"
            )
        frames, feat_dim = int(emb.shape[0]), int(emb.shape[1])

        if abs(self.rate - MERT_FRAME_RATE_HZ) > 0.0:
            problems.append(f"rate={self.rate} != MERT_FRAME_RATE_HZ={MERT_FRAME_RATE_HZ}")
        if self.sample_rate != MERT_SAMPLE_RATE_HZ:
            problems.append(f"sample_rate={self.sample_rate} != {MERT_SAMPLE_RATE_HZ}")
        if expected_layer is not None and self.layer != expected_layer:
            problems.append(f"layer={self.layer} != 期望 {expected_layer}")
        if expected_model_rev is not None and self.model_rev != expected_model_rev:
            problems.append(f"model_rev={self.model_rev!r} != 期望 {expected_model_rev!r}")
        if self.feat_dim != feat_dim:
            problems.append(f"feat_dim={self.feat_dim} != 数组 feat={feat_dim}")
        if self.feat_dim != MERT_DEFAULT_FEAT_DIM:
            problems.append(
                f"feat_dim={self.feat_dim} != MERT_DEFAULT_FEAT_DIM={MERT_DEFAULT_FEAT_DIM}",
            )
        expected_frames = round(self.duration_s * self.rate)
        if abs(frames - expected_frames) > frame_tolerance:
            problems.append(
                f"帧数 {frames} 与 duration_s * rate = {expected_frames} 相差超过 "
                f"{frame_tolerance}（帧率漂移哨兵）",
            )
        if str(emb.dtype) != self.dtype:
            problems.append(f"数组 dtype={emb.dtype} != meta.dtype={self.dtype}")
        if expected_dtype is not None and self.dtype != expected_dtype:
            problems.append(f"dtype={self.dtype} != 期望 {expected_dtype}")
        if expected_adapter is not None and self.adapter != expected_adapter:
            problems.append(f"adapter={self.adapter!r} != 期望 {expected_adapter!r}")

        if problems:
            raise FeatureCacheMismatchError("特征缓存校验失败：" + "；".join(problems))


# ══════════════════════════════════════════════════════════════
# M9：特征提取与缓存
# ══════════════════════════════════════════════════════════════


def feature_cache_paths(out_dir: Path, key: str) -> tuple[Path, Path]:
    """缓存路径 `(<out>/<key>.npz, <out>/<key>.meta.json)`。"""
    return out_dir / f"{key}.npz", out_dir / f"{key}.meta.json"


def _sample_count(wav: Dynamic) -> int:
    """波形样本数（numpy / torch / 序列皆可）。"""
    shape = getattr(wav, "shape", None)
    if shape is not None:
        return int(shape[-1])
    return len(wav)


def _as_numpy(value: Dynamic) -> FeatureArray:
    """把编码器输出统一成 `[T, feat]` numpy 数组（`[B, T, F]` 取 batch 0）。"""
    array: FeatureArray
    if isinstance(value, np.ndarray):
        array = value
    elif hasattr(value, "detach"):  # torch.Tensor（鸭子类型，避免为此导入 torch）
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim == 3:
        array = array[0]
    if array.ndim != 2:
        raise FeatureCacheMismatchError(f"编码器输出必须是 2/3 维，得到 shape={array.shape}")
    return array


def _encoder_frame_rate(encoder: FeatureEncoder) -> float:
    """取编码器帧率；缺失时回落契约常量（**绝不**假设一个数）。"""
    method = getattr(encoder, "output_frame_rate", None)
    if callable(method):
        rate = float(method())
        if rate > 0.0:
            return rate
    logger.warning("编码器未提供 output_frame_rate，回落契约帧率 %.1f Hz", MERT_FRAME_RATE_HZ)
    return MERT_FRAME_RATE_HZ


def _encode(encoder: FeatureEncoder, wav: Dynamic) -> Dynamic:
    """调用编码器 `encode`（兼容 plan 01 当前单参实现）。"""
    try:
        return encoder.encode(wav, sample_rate=MERT_SAMPLE_RATE_HZ)
    except TypeError:
        logger.warning(
            "编码器 encode 不接受 sample_rate 参数，退回单参调用"
            "（plan 01 §3.2 声明的是两参签名；采样率断言由调用方在入口完成）",
        )
        return encoder.encode(wav)


def _default_loader(path: Path) -> tuple[Any, int]:
    """默认音频加载：`torchaudio.load` 取**第一声道**（单声道假设见 plan 01 §4）。"""
    import torchaudio

    waveform, sample_rate = torchaudio.load(str(path))
    return waveform[0], int(sample_rate)


def _default_resampler(wav: Dynamic, source_rate: int, target_rate: int) -> Dynamic:
    """默认重采样：`torchaudio.functional.resample`（重采样只发生在本入口，plan 01 §4）。"""
    import torch
    import torchaudio

    tensor = wav if isinstance(wav, torch.Tensor) else torch.as_tensor(np.asarray(wav))
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    return torchaudio.functional.resample(tensor, source_rate, target_rate)


def extract_features(
    audio_path: Path,
    out_dir: Path,
    encoder: FeatureEncoder,
    *,
    layer: int = 12,
    model_rev: str = "unset",
    adapter: str = "none",
    key: str | None = None,
    loader: AudioLoader | None = None,
    resampler: Resampler | None = None,
    overwrite: bool = False,
) -> FeatureCacheMeta:
    """提取一段音频的 MERT 特征并落盘（`<key>.npz` + `<key>.meta.json`）。

    Args:
        audio_path: 音频文件（内容 sha1 即默认缓存键 → 同曲多谱只提取一次）。
        out_dir: 缓存目录。
        encoder: 编码器（满足 :class:`FeatureEncoder`；测试用假编码器，不下载权重）。
        layer / model_rev / adapter: 提取配置，写入元数据并在加载时校验。
        key: 覆盖缓存键（默认音频内容 sha1）。
        loader / resampler: 覆盖默认的 `torchaudio` 通路（单测注入，避免依赖 torch）。
        overwrite: False 时命中缓存直接返回既有元数据（**去重的落点**）。

    Returns:
        落盘的 :class:`FeatureCacheMeta`。

    Raises:
        FeatureCacheMismatchError: 帧数与 `duration_s * rate` 不符（帧率漂移哨兵）。
    """
    audio_path = Path(audio_path)
    content = audio_path.read_bytes()
    cache_key = key or sha1_hex(content)
    npz_path, meta_path = feature_cache_paths(Path(out_dir), cache_key)
    if not overwrite and npz_path.is_file() and meta_path.is_file():
        _emb, meta = load_feature_cache(
            npz_path,
            meta_path,
            expected_layer=layer,
            expected_model_rev=model_rev,
            expected_adapter=adapter,
        )
        logger.info("特征缓存命中 %s（音频 sha1=%s）", npz_path.name, cache_key)
        return meta

    load = loader or _default_loader
    resample = resampler or _default_resampler
    wav, original_sample_rate = load(audio_path)
    wav_model_rate = resample(wav, int(original_sample_rate), MERT_SAMPLE_RATE_HZ)

    duration_s = _sample_count(wav_model_rate) / MERT_SAMPLE_RATE_HZ
    array = _as_numpy(_encode(encoder, wav_model_rate))
    rate = _encoder_frame_rate(encoder)

    meta = FeatureCacheMeta(
        rate=rate,
        sample_rate=MERT_SAMPLE_RATE_HZ,
        layer=layer,
        model_rev=model_rev,
        duration_s=duration_s,
        original_sample_rate=int(original_sample_rate),
        feat_dim=int(array.shape[1]),
        dtype=str(array.dtype),
        adapter=adapter,
    )
    meta.verify(array)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, emb=array)
    meta_path.write_text(meta.model_dump_json(indent=2), encoding="utf-8")
    logger.info("已提取特征 %s（%d 帧 × %d 维，%.2fs）", npz_path.name, *array.shape, duration_s)
    return meta


def load_feature_cache(
    npz_path: Path,
    meta_path: Path | None = None,
    *,
    expected_layer: int | None = None,
    expected_model_rev: str | None = None,
    expected_dtype: str | None = None,
    expected_adapter: str | None = None,
    frame_tolerance: int = FRAME_TOLERANCE,
) -> tuple[FeatureArray, FeatureCacheMeta]:
    """加载特征缓存并**逐项校验**（六项 + 形状/精度/适配器）。

    Raises:
        FeatureCacheMismatchError: 任一校验失败（含元数据 JSON 无法解析）。
    """
    npz_path = Path(npz_path)
    if meta_path is None:
        npz_path, meta_path = feature_cache_paths(npz_path.parent, npz_path.stem)
    try:
        meta = FeatureCacheMeta.model_validate_json(Path(meta_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FeatureCacheMismatchError(f"元数据 {meta_path} 不可用：{exc}") from exc
    try:
        with np.load(npz_path, allow_pickle=False) as payload:
            emb: FeatureArray = payload["emb"]
    except (OSError, KeyError, ValueError) as exc:
        raise FeatureCacheMismatchError(f"特征文件 {npz_path} 不可用：{exc}") from exc

    meta.verify(
        emb,
        expected_layer=expected_layer,
        expected_model_rev=expected_model_rev,
        expected_dtype=expected_dtype,
        expected_adapter=expected_adapter,
        frame_tolerance=frame_tolerance,
    )
    return emb, meta


def audio_cache_key(audio_path: Path) -> str:
    """音频内容 sha1（缓存键 / 唯一曲目统计的外键）。"""
    return sha1_file(Path(audio_path))


# ══════════════════════════════════════════════════════════════
# M10：清单行、配对与切分
# ══════════════════════════════════════════════════════════════

#: 三份 split 的名字（固定顺序，便于断言与落盘）。
SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")


def chart_row(
    *,
    chart_id: int | None,
    name: str = "",
    composer: str = "",
    difficulty: float | None = None,
    fmt: ChartFormat | str = ChartFormat.RPE,
    chart_path: str = "",
    feature_key: str | None = None,
    qc: QcReport | None = None,
) -> dict[str, Any]:
    """构造一行清单记录（切分与统计的输入；`song_key` 是切分单位）。"""
    distribution = None if qc is None else qc.distribution
    stats = None if distribution is None else distribution.to_dict()
    row: dict[str, Any] = {
        "chart_id": chart_id,
        "song_key": f"{name}|{composer}",
        "name": name,
        "composer": composer,
        "difficulty": difficulty,
        "format": str(fmt),
        "chart_path": chart_path,
        "feature_key": feature_key,
        "qc_passed": None if qc is None else qc.passed,
        "n_lines": None if qc is None else qc.n_lines,
        "n_notes": None if qc is None else qc.n_notes,
        "out_of_visible_range": None if qc is None else qc.out_of_visible_range,
        "out_of_audio_window": None if qc is None else qc.out_of_audio_window,
        "type_counts": None if stats is None else stats["type_counts"],
        "above_counts": None if stats is None else stats["above_counts"],
        "back_fraction": None if stats is None else stats["back_fraction"],
        "tap_fraction": None if stats is None else stats["tap_fraction"],
        "time_span_s": None if stats is None else stats["time_span_s"],
        "bpm_min": None if stats is None else stats["bpm_min"],
        "bpm_max": None if stats is None else stats["bpm_max"],
        "max_simultaneous_onsets": None if stats is None else stats["max_simultaneous_onsets"],
        "min_same_line_same_time_gap_x": (
            None if stats is None else stats["min_same_line_same_time_gap_x"]
        ),
    }
    return row


@dataclass(frozen=True)
class PairRow:
    """一条 (audio, chart) 对（已在某个 split 内）。"""

    chart_id: int | None
    song_key: str
    split: str
    chart_path: str
    feature_key: str | None = None
    difficulty: float | None = None
    fmt: str = str(ChartFormat.RPE)
    #: 曲名 / 曲师（`song_key` 的两个组成分量，便于人工核对与分组）
    name: str = ""
    composer: str = ""

    def to_dict(self) -> dict[str, Any]:
        """清单行。"""
        return {
            "chart_id": self.chart_id,
            "song_key": self.song_key,
            "name": self.name,
            "composer": self.composer,
            "split": self.split,
            "chart_path": self.chart_path,
            "feature_key": self.feature_key,
            "difficulty": self.difficulty,
            "format": self.fmt,
        }


@dataclass(frozen=True)
class GeneralizationPair:
    """「同曲跨谱泛化」评测对：同曲的低难度谱进 train、高难度谱作评估（plan §偏离 1）。"""

    song_key: str
    train_chart_id: int | None
    eval_chart_id: int | None
    train_difficulty: float | None
    eval_difficulty: float | None

    def to_dict(self) -> dict[str, Any]:
        """清单行。"""
        return {
            "song_key": self.song_key,
            "train_chart_id": self.train_chart_id,
            "eval_chart_id": self.eval_chart_id,
            "train_difficulty": self.train_difficulty,
            "eval_difficulty": self.eval_difficulty,
        }


@dataclass(frozen=True)
class PairSplits:
    """三份 split + 泛化评测集（M10 的产出）。"""

    train: list[PairRow] = field(default_factory=list)
    val: list[PairRow] = field(default_factory=list)
    test: list[PairRow] = field(default_factory=list)
    generalization: list[GeneralizationPair] = field(default_factory=list)
    #: 谱面文件不存在而未能成对的样本数（**必须显式记账**，不得静默丢弃）
    skipped_no_chart: int = 0
    #: 特征缓存缺失（音频缺失/未提取）而未能成对的样本数（同上）
    skipped_no_feature: int = 0

    @property
    def skipped_total(self) -> int:
        """被跳过的清单行总数（谱面缺失 + 特征缺失）。"""
        return self.skipped_no_chart + self.skipped_no_feature

    @property
    def n_pairs(self) -> int:
        """真实 (audio, chart) 对数（< 谱面总数：同曲多谱共享同一音频）。"""
        return len(self.train) + len(self.val) + len(self.test)

    def rows(self, split: str) -> list[PairRow]:
        """按名字取一份 split。"""
        if split not in SPLIT_NAMES:
            raise SplitError(f"split 必须是 {SPLIT_NAMES} 之一，得到 {split!r}")
        return {"train": self.train, "val": self.val, "test": self.test}[split]

    def song_sets(self) -> dict[str, set[str]]:
        """每份 split 的曲目集合（M10 断言两两不相交）。"""
        return {name: {row.song_key for row in self.rows(name)} for name in SPLIT_NAMES}

    def assert_disjoint_songs(self) -> None:
        """断言三份 split 的曲目集合两两不相交，否则抛 :class:`SplitError`。"""
        sets = self.song_sets()
        for left_index, left in enumerate(SPLIT_NAMES):
            for right in SPLIT_NAMES[left_index + 1 :]:
                overlap = sets[left] & sets[right]
                if overlap:
                    raise SplitError(
                        f"{left} 与 {right} 的曲目集合相交（val 泄漏）：{sorted(overlap)}"
                    )

    def to_dict(self) -> dict[str, Any]:
        """清单导出（每份 split 一个数组 + 泛化集）。"""
        return {
            "train": [row.to_dict() for row in self.train],
            "val": [row.to_dict() for row in self.val],
            "test": [row.to_dict() for row in self.test],
            "generalization": [pair.to_dict() for pair in self.generalization],
            "skipped_no_chart": self.skipped_no_chart,
            "skipped_no_feature": self.skipped_no_feature,
        }


def build_pairs(
    meta_table: Path,
    chart_dir: Path,
    feature_dir: Path,
    *,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
    require_feature: bool = True,
) -> PairSplits:
    """按曲目切分 train/val/test 并产出「同曲跨谱泛化」评测集。

    Args:
        meta_table: 清单路径（`.jsonl`/`.parquet`，**必须带 provenance**）。
        chart_dir: 谱面落盘根（用于校验 `chart_path` 存在）。
        feature_dir: 特征缓存根（用于校验 `<feature_key>.npz` 存在）。
        ratios: (train, val, test) 比例，和为 1。
        seed: 曲目打乱种子（确定性切分）。
        require_feature: True = 缺特征的样本不进对（计入 `skipped_no_feature`）。

    Raises:
        SplitError: 比例非法或曲目集合相交。
        ManifestError: 清单缺 provenance（M8 硬约束③）。
    """
    if len(ratios) != len(SPLIT_NAMES) or abs(sum(ratios) - 1.0) > 1e-9:
        raise SplitError(f"ratios 必须是三元组且和为 1，得到 {ratios}")

    manifest = read_manifest(Path(meta_table))
    chart_root = Path(chart_dir)
    feature_root = Path(feature_dir)

    resolved: list[tuple[str, PairRow]] = []
    skipped_no_chart = 0
    skipped_no_feature = 0
    for row in manifest.rows:
        song_key = str(row.get("song_key") or f"{row.get('name', '')}|{row.get('composer', '')}")
        chart_path = _resolve_chart_path(row, chart_root)
        if chart_path is None:
            skipped_no_chart += 1
            logger.warning("清单行 chart_id=%s 的谱面文件不存在，跳过", row.get("chart_id"))
            continue
        feature_key = row.get("feature_key")
        if feature_key:
            npz_path, _meta = feature_cache_paths(feature_root, str(feature_key))
            if not npz_path.is_file():
                if require_feature:
                    skipped_no_feature += 1
                    continue
                feature_key = None
        elif require_feature:
            skipped_no_feature += 1
            continue
        resolved.append(
            (
                song_key,
                PairRow(
                    chart_id=_optional_int(row.get("chart_id")),
                    song_key=song_key,
                    split="",
                    chart_path=str(chart_path),
                    feature_key=None if feature_key is None else str(feature_key),
                    difficulty=_optional_float(row.get("difficulty")),
                    fmt=str(row.get("format") or ChartFormat.RPE),
                    name=str(row.get("name") or ""),
                    composer=str(row.get("composer") or ""),
                ),
            ),
        )

    songs = sorted({song_key for song_key, _ in resolved})
    assignment = _assign_songs(songs, ratios, seed)
    splits: dict[str, list[PairRow]] = {name: [] for name in SPLIT_NAMES}
    for song_key, pair in resolved:
        split = assignment[song_key]
        splits[split].append(
            PairRow(
                chart_id=pair.chart_id,
                song_key=song_key,
                split=split,
                chart_path=pair.chart_path,
                feature_key=pair.feature_key,
                difficulty=pair.difficulty,
                fmt=pair.fmt,
                name=pair.name,
                composer=pair.composer,
            ),
        )
    for name in SPLIT_NAMES:
        splits[name].sort(key=_pair_sort_key)

    result = PairSplits(
        train=splits["train"],
        val=splits["val"],
        test=splits["test"],
        generalization=_generalization_pairs(splits["train"]),
        skipped_no_chart=skipped_no_chart,
        skipped_no_feature=skipped_no_feature,
    )
    result.assert_disjoint_songs()
    logger.info(
        "配对完成：train=%d val=%d test=%d 泛化对=%d 跳过（谱面缺 %d / 特征缺 %d）",
        len(result.train),
        len(result.val),
        len(result.test),
        len(result.generalization),
        skipped_no_chart,
        skipped_no_feature,
    )
    return result


def _pair_sort_key(row: PairRow) -> tuple[str, int]:
    """确定性排序键（`chart_id` 缺失时用 -1 排在前面）。"""
    return (row.song_key, row.chart_id if row.chart_id is not None else -1)


def _resolve_chart_path(row: Mapping[str, Any], chart_root: Path) -> Path | None:
    """定位清单行对应的谱面文件（`chart_path` 优先，其次按 R4 规范化名推导）。"""
    chart_id = row.get("chart_id")
    relative = row.get("chart_path")
    if relative:
        candidate = chart_root / str(relative)
        if candidate.is_file():
            return candidate
        return None
    chart_file = row.get("chart_file")
    if chart_file and chart_id is not None:
        candidate = chart_dest_path(chart_root, int(chart_id), str(chart_file))
        return candidate if candidate.is_file() else None
    return None


def _assign_songs(
    songs: Sequence[str],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, str]:
    """把曲目确定性地分到三份 split（**同曲多谱必然同 split**）。"""
    order = list(songs)
    random.Random(seed).shuffle(order)
    total = len(order)
    if total == 0:
        return {}
    if total == 1:
        # 单曲目无法同时撑起三份 split：放进 train，并在日志中可被 skipped/统计发现。
        return {order[0]: "train"}
    train_end = int(total * ratios[0])
    val_end = train_end + int(total * ratios[1])
    if total >= len(SPLIT_NAMES):
        # 保证三份都非空（样本很少时按比例取整会把某一 split 清零）。
        train_end = max(1, min(train_end, total - 2))
        val_end = max(train_end + 1, min(val_end, total - 1))
    assignment: dict[str, str] = {}
    for index, song_key in enumerate(order):
        if index < train_end:
            assignment[song_key] = "train"
        elif index < val_end:
            assignment[song_key] = "val"
        else:
            assignment[song_key] = "test"
    return assignment


def _generalization_pairs(train_rows: Sequence[PairRow]) -> list[GeneralizationPair]:
    """在 train 内挑「同曲 ≥2 张且定数不同」的曲目，产出同曲跨谱泛化对。"""
    grouped: dict[str, list[PairRow]] = {}
    for row in train_rows:
        grouped.setdefault(row.song_key, []).append(row)
    pairs: list[GeneralizationPair] = []
    for song_key in sorted(grouped):
        rows = [row for row in grouped[song_key] if row.difficulty is not None]
        if len(rows) < 2:
            continue
        ordered = sorted(rows, key=lambda row: (row.difficulty or 0.0, row.chart_id or -1))
        low, high = ordered[0], ordered[-1]
        if low.difficulty == high.difficulty:
            continue
        pairs.append(
            GeneralizationPair(
                song_key=song_key,
                train_chart_id=low.chart_id,
                eval_chart_id=high.chart_id,
                train_difficulty=low.difficulty,
                eval_difficulty=high.difficulty,
            ),
        )
    return pairs


def _optional_int(value: Dynamic) -> int | None:
    """宽松取整（None / 缺失 → None）。"""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Dynamic) -> float | None:
    """宽松取浮点（None / 缺失 → None）。"""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ══════════════════════════════════════════════════════════════
# 语料级统计（M9/M10 的「先统计再定阈值」落点）
# ══════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class DatasetStats:
    """全库分布统计（plan §3.6 `dataset_stats`）。

    阈值纪律（plan §7-R3）：本结构只**报告**分布与离群，不据此过滤样本。
    """

    n_charts: int
    #: 唯一曲目数（**按 `name | composer` 近似**，info.yml 无全局唯一歌曲 ID）
    n_unique_songs: int
    #: 同曲重复率 = 1 - 唯一曲目数 / 谱面数
    repeat_rate: float
    #: 唯一音频数（**按音频内容 sha1 去重**；决定真实 (audio, chart) 对数，plan §9-Q3）
    n_unique_audio: int
    #: 音频层面的重复率 = 1 - 唯一音频数 / 谱面数
    audio_repeat_rate: float
    format_counts: dict[str, int]
    lines_median: float
    lines_p25: float
    lines_p75: float
    lines_min: int
    lines_max: int
    n_notes_total: int
    tap_fraction: float | None
    back_fraction: float | None
    out_of_range_rate: float | None
    min_same_line_same_time_gap_x: float | None
    max_simultaneous_onsets: int | None

    @property
    def same_song_repeat_rate(self) -> float:
        """同曲重复率 = `1 - 唯一曲目数 / 谱面数`。"""
        return self.repeat_rate

    def to_dict(self) -> dict[str, Any]:
        """清单行（JSON 友好）。"""
        return {
            "n_charts": self.n_charts,
            "n_unique_songs": self.n_unique_songs,
            "repeat_rate": self.repeat_rate,
            "n_unique_audio": self.n_unique_audio,
            "audio_repeat_rate": self.audio_repeat_rate,
            "format_counts": dict(self.format_counts),
            "lines_median": self.lines_median,
            "lines_p25": self.lines_p25,
            "lines_p75": self.lines_p75,
            "lines_min": self.lines_min,
            "lines_max": self.lines_max,
            "n_notes_total": self.n_notes_total,
            "tap_fraction": self.tap_fraction,
            "back_fraction": self.back_fraction,
            "out_of_range_rate": self.out_of_range_rate,
            "min_same_line_same_time_gap_x": self.min_same_line_same_time_gap_x,
            "max_simultaneous_onsets": self.max_simultaneous_onsets,
        }

    @classmethod
    def from_rows(cls, rows: Sequence[Mapping[str, Any]]) -> DatasetStats:
        """从清单行聚合（缺失字段跳过，不猜默认值）。"""
        song_keys = {str(row.get("song_key") or "") for row in rows}
        audio_keys = {str(row["feature_key"]) for row in rows if row.get("feature_key")}
        lines = [int(row["n_lines"]) for row in rows if row.get("n_lines") is not None]
        notes = [int(row["n_notes"]) for row in rows if row.get("n_notes") is not None]
        formats = Counter(str(row.get("format")) for row in rows)
        tap = _weighted_fraction(rows, "tap_fraction")
        back = _weighted_fraction(rows, "back_fraction")
        out_of_range = sum(int(row.get("out_of_visible_range") or 0) for row in rows)
        gaps = [
            float(row["min_same_line_same_time_gap_x"])
            for row in rows
            if row.get("min_same_line_same_time_gap_x") is not None
        ]
        onsets = [
            int(row["max_simultaneous_onsets"])
            for row in rows
            if row.get("max_simultaneous_onsets") is not None
        ]
        quarters = statistics.quantiles(lines, n=4, method="inclusive") if len(lines) >= 2 else []
        total_notes = sum(notes)
        return cls(
            n_charts=len(rows),
            n_unique_songs=len(song_keys),
            repeat_rate=1.0 - (len(song_keys) / len(rows)) if rows else 0.0,
            n_unique_audio=len(audio_keys),
            audio_repeat_rate=1.0 - (len(audio_keys) / len(rows)) if rows else 0.0,
            format_counts=dict(formats),
            lines_median=float(statistics.median(lines)) if lines else 0.0,
            lines_p25=float(quarters[0]) if quarters else 0.0,
            lines_p75=float(quarters[2]) if quarters else 0.0,
            lines_min=min(lines) if lines else 0,
            lines_max=max(lines) if lines else 0,
            n_notes_total=total_notes,
            tap_fraction=tap,
            back_fraction=back,
            out_of_range_rate=(out_of_range / total_notes) if total_notes else None,
            min_same_line_same_time_gap_x=min(gaps) if gaps else None,
            max_simultaneous_onsets=max(onsets) if onsets else None,
        )

    def corpus_outliers(self) -> list[str]:
        """与调研 §7 实测基线比对，返回离群说明（**不删除样本**，plan §7-R3）。

        样本数 < :data:`MIN_CHARTS_FOR_CORPUS_CHECK` 时只报「样本不足」，不做区间断言。
        """
        notes: list[str] = []
        if self.n_charts < MIN_CHARTS_FOR_CORPUS_CHECK:
            notes.append(
                f"样本数 {self.n_charts} < {MIN_CHARTS_FOR_CORPUS_CHECK}，不做基线区间断言",
            )
            return notes
        if self.back_fraction is not None and not _in_range(
            self.back_fraction,
            SURVEY_BACK_FRACTION_RANGE,
        ):
            notes.append(
                f"背面占比 {self.back_fraction:.4f} 偏离调研 §7.3 基线 {SURVEY_BACK_FRACTION_RANGE}",
            )
        if self.tap_fraction is not None and not _in_range(
            self.tap_fraction,
            SURVEY_TAP_FRACTION_RANGE,
        ):
            notes.append(
                f"Tap 占比 {self.tap_fraction:.4f} 偏离调研 §7.3 基线 {SURVEY_TAP_FRACTION_RANGE}",
            )
        if not _in_range(self.lines_median, SURVEY_LINES_MEDIAN_RANGE):
            notes.append(
                f"线数中位 {self.lines_median} 偏离调研 §7.1 基线 {SURVEY_LINES_MEDIAN_RANGE}",
            )
        return notes


def _in_range(value: float, bounds: tuple[float, float]) -> bool:
    """闭区间判定。"""
    low, high = bounds
    return low <= value <= high


def _weighted_fraction(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    """按 note 数加权的占比（避免给小程序谱与长谱同权）。"""
    total = 0.0
    weight = 0
    for row in rows:
        if row.get(key) is None or row.get("n_notes") is None:
            continue
        total += float(row[key]) * int(row["n_notes"])
        weight += int(row["n_notes"])
    if weight == 0:
        return None
    return total / weight


def dataset_stats(table: Path) -> DatasetStats:
    """读清单并聚合全库统计（`unique_songs` / 同曲重复率 / 格式与分布）。"""
    manifest: Manifest = read_manifest(Path(table))
    return DatasetStats.from_rows(manifest.rows)


__all__ = [
    "CACHE_META_KEYS",
    "FEATURE_CACHE_META_KEYS",
    "FRAME_TOLERANCE",
    "MANIFEST_SUFFIX",
    "MIN_CHARTS_FOR_CORPUS_CHECK",
    "SPLIT_NAMES",
    "SURVEY_BACK_FRACTION_RANGE",
    "SURVEY_LINES_MEDIAN_RANGE",
    "SURVEY_TAP_FRACTION_RANGE",
    "AudioLoader",
    "DatasetStats",
    "FeatureArray",
    "FeatureCacheMeta",
    "FeatureCacheMismatchError",
    "FeatureEncoder",
    "GeneralizationPair",
    "PairRow",
    "PairSplits",
    "Resampler",
    "SplitError",
    "audio_cache_key",
    "build_pairs",
    "chart_row",
    "dataset_stats",
    "extract_features",
    "feature_cache_paths",
    "load_feature_cache",
]
