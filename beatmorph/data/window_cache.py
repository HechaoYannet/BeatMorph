"""窗口预切缓存：把「一个窗口的样本」在**数据处理阶段**物化，训练期只做读取。

## 为什么需要这一层（实测依据）

训练期每取一个窗口都要把**整张谱**重做一遍，而一张谱平均有 96 个窗口。函数级实测
（RFC-0035 §1，真实 train split 100 窗）里每窗口 0.85 s 的 CPU 有 **91.3%** 花在
「一张谱只该做一次」的事上：

| 顶层调用 | 占每窗口 CPU |
|---|---|
| `_read_chart_or_skip`（重解析 3.8 MB RJSON） | 49.1% |
| `_window_subchart`（线性扫过全部 1294 个 note 只挑 ~13 个） | 23.6% |
| `load_feature_cache`（整段解压 11.3 MB npz） | 15.6% |
| `line_tracks_at` | 3.0% |

而计划层跨 991 桶轮转发牌 ⇒ 同一张谱的两个窗口相隔 ~991 个槽位 ⇒ 行级 LRU 命中率必然为 0。
**顺序与缓存局部性不可兼得**（RFC-0035 §2 的三条否决全都建立在这个前提上）。
把窗口**按窗口**物化之后这个前提消失：每个窗口自包含，顺序可以随便排。

## 为什么**不能**直接存张量（这条决定了本模块的整个形状）

实测（`scripts/local_window_footprint.py`，真实 train split 24 窗）：

| 字段 | 均值 MiB/窗 |
|---|---|
| `counts` (K,T,X,S,C) int16 | **15.215** |
| `occlusion` 同形状 bool | **7.607** |
| `audio_emb` | 0.394 |
| `line_tracks` | 0.119 |

稠密 22.82 MiB/窗承载的是 **10.58 个非零格**（占用率 1.3e-6）。外推 634 952 窗 =
**13.8 TiB**，而 D: 只有 858 GiB ⇒ 差 16 倍。即使盘管够，目标吞吐下的带宽也超过实测盘能力。

因此本模块存的是**重建稠密张量所需的最小记录**（稀疏计数 + token 遮盖位图 + 轨道 + 音频），
训练期用 `np.zeros` + scatter 重建。重建实测 **0.02 ms/窗**（Windows 惰性零页；
真正 touch 页的成本在今天的 H2D 里本来就在付）。

## 语义纪律

- **逐位一致**：同一 index 经本缓存读出的 `PairSample` 必须与
  `ChartPairDataset.__getitem__` 逐字段 `torch.equal`（由
  `tests/unit/data/test_window_cache.py` 锁定）。
- **指纹不符即拒绝**：缓存是**派生加速器**，不是事实源；读失败/指纹不符一律回退到原路径，
  绝不静默降级（RFC-0035 判据 5）。
- **worker 侧零私有状态**：本模块只用 `mmap` 只读视图，页缓存跨 worker / 跨 epoch
  共享 ⇒ 这是「加 worker 不再吃内存」的依据。
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from beatmorph.core.contracts.tensors import MERT_FRAME_RATE_HZ
from beatmorph.core.logging import get_logger
from beatmorph.field.grid import FieldGrid
from beatmorph.generation.batch import N_ORDINARY_TRACKS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from beatmorph.data.dataset import DatasetConfig, PairSample
    from beatmorph.data.pipeline.embed import PairRow

logger = get_logger(__name__)

#: 缓存格式版本。**任何**布局/语义变化都必须 +1（旧目录因此自然失效，不会误读）。
WINDOW_CACHE_VERSION: int = 1

#: 一个 shard 里装多少个窗口。取 512：单个 shard 约 170 MB（内存里攒得下），
#: 全库约 1240 个 shard，读取期一次 shard 切换 = 10 次 `np.load`（mmap，微秒级）。
DEFAULT_SHARD_WINDOWS: int = 512

#: 索引文件名（一个 split 一个）。
INDEX_FILENAME: str = "index.json"

#: `meta.npy` 的列序（**只放逐窗口的量**；逐行的量从清单读，不重复存）。
META_COLUMNS: tuple[str, ...] = (
    "k",
    "bpm_eff",
    "tau_start",
    "difficulty",
    "frame_start",
    "padded_frames",
    "row_index",
    "window_index",
)

__all__ = [
    "DEFAULT_SHARD_WINDOWS",
    "META_COLUMNS",
    "WINDOW_CACHE_VERSION",
    "WindowCacheError",
    "WindowCacheIndex",
    "WindowCacheReader",
    "WindowCacheWriter",
    "read_index",
    "window_cache_directory",
    "window_cache_fingerprint",
    "windows_on_disk",
    "write_index",
]


class WindowCacheError(RuntimeError):
    """缓存不可用（结构非法 / 指纹不符 / 自相矛盾）。调用方应回退到原路径，不静默降级。"""


@dataclass(frozen=True, slots=True)
class WindowCacheIndex:
    """一个 split 的窗口缓存索引（`index.json` 的形状）。"""

    split: str
    fingerprint: str
    n_windows: int
    shard_windows: int
    t_bins: int
    x_bins: int
    sides: int
    channels: int
    feature_dim: int
    audio_dtype: str
    version: int = WINDOW_CACHE_VERSION

    @property
    def n_shards(self) -> int:
        """shard 数（最后一个可能不满）。"""
        if self.n_windows <= 0:
            return 0
        return (self.n_windows + self.shard_windows - 1) // self.shard_windows

    def shard_of(self, index: int) -> tuple[int, int]:
        """`(shard 序号, shard 内下标)`。"""
        if not 0 <= index < self.n_windows:
            raise IndexError(f"窗口下标 {index} 越界（共 {self.n_windows} 个窗口）")
        return divmod(int(index), self.shard_windows)


def window_cache_fingerprint(
    config: DatasetConfig,
    rows: Sequence[PairRow],
    *,
    seed: int,
) -> str:
    """缓存指纹：**一切影响逐窗口产物的输入**。

    与索引缓存（`dataset._plan_fingerprint`）同口径再加三项：格式版本、
    遮盖种子（`occlusion` 的种子由 `(seed, row_index, window_index)` 派生）、
    音频帧率（派生量，缓存里存的是帧索引）。
    """
    from beatmorph.data.dataset import _plan_fingerprint

    digest = hashlib.sha1()
    digest.update(f"window-cache-v{WINDOW_CACHE_VERSION}".encode())
    digest.update(f"|seed={int(seed)}|frame_rate={MERT_FRAME_RATE_HZ}|".encode())
    # 复用**索引缓存**的指纹而不是另写一套：它已经覆盖 split / limit / t_window / x_bins /
    # k_max / occlusion_ratio / tau_end_s / tau_end_policy + 逐行身份 + 谱面与特征元数据的 stat。
    # 另写一套的风险是「漏掉一个语义字段」——那正是本缓存最不该犯的错（静默喂错样本）。
    digest.update(_plan_fingerprint(config, rows).encode())
    return digest.hexdigest()


def window_cache_directory(root: Path, split: str, fingerprint: str) -> Path:
    """`<root>/<split>/<fingerprint 前 16 位>/`（目录名带指纹 ⇒ 换配置自然分叉）。"""
    return root / split / fingerprint[:16]


def _shard_dir(directory: Path, shard: int) -> Path:
    return directory / f"part-{shard:06d}"


# ══════════════════════════════════════════════════════════════
# 写
# ══════════════════════════════════════════════════════════════


class WindowCacheWriter:
    """把一个 split 的窗口样本按 shard 落盘（构造器侧；**不在训练路径上**）。

    Args:
        directory: 本 split 的缓存目录。
        index: 索引元数据（`n_windows` 由 `close` 回填校验）。
        audio_dtype: 音频落盘 dtype——取**源特征缓存的 dtype**，read 时再升到 float32。
            fp16 -> fp32 是**精确**的，因此逐位一致；若源是 fp32 就直接存 fp32。
    """

    def __init__(
        self,
        directory: Path,
        *,
        split: str,
        fingerprint: str,
        t_bins: int,
        x_bins: int,
        sides: int,
        channels: int,
        feature_dim: int,
        audio_dtype: str,
        shard_windows: int = DEFAULT_SHARD_WINDOWS,
        start_shard: int = 0,
    ) -> None:
        if shard_windows < 1:
            raise ValueError(f"shard_windows 必须 >= 1，得到 {shard_windows}")
        self._directory = directory
        self._split = split
        self._fingerprint = fingerprint
        self._t_bins = int(t_bins)
        self._x_bins = int(x_bins)
        self._sides = int(sides)
        self._channels = int(channels)
        self._feature_dim = int(feature_dim)
        self._audio_dtype = np.dtype(audio_dtype)
        self._shard_windows = int(shard_windows)
        #: 并行构建时每个 worker 只写自己那一段 shard ⇒ 起始编号由调用方给。
        self._shard = int(start_shard)
        self._written = 0
        self._meta: list[tuple[float, ...]] = []
        self._counts_idx: list[int] = []
        self._counts_val: list[int] = []
        self._counts_ptr: list[int] = [0]
        self._occl_tok: list[int] = []
        self._occl_ptr: list[int] = [0]
        self._tracks: list[NDArray[np.float32]] = []
        self._tracks_ptr: list[int] = [0]
        self._audio: list[NDArray[Any]] = []
        self._audio_ptr: list[int] = [0]
        self._non_token_block = 0

    # ── 写入 ────────────────────────────────────────────────────
    def add(self, sample: PairSample) -> None:
        """登记一个窗口样本（顺序**必须**等于窗口下标 0,1,2,...）。

        Raises:
            WindowCacheError: 形状与索引声明不符，或遮盖不是 token 区块结构。
        """
        counts = np.asarray(sample.counts.numpy(), dtype=np.int16)
        if counts.shape[1:] != (self._t_bins, self._x_bins, self._sides, self._channels):
            raise WindowCacheError(
                f"counts 尾维与索引声明不符：{counts.shape[1:]} != "
                f"{(self._t_bins, self._x_bins, self._sides, self._channels)}"
            )
        occlusion = np.asarray(sample.occlusion.numpy(), dtype=np.bool_)
        if occlusion.shape != counts.shape:
            raise WindowCacheError(f"occlusion 形状 {occlusion.shape} != counts {counts.shape}")
        k = int(counts.shape[0])
        token = _token_view(occlusion, k=k, t_bins=self._t_bins)
        self._meta.append(
            (
                float(k),
                float(sample.grid.bpm_points[0].bpm),
                float(sample.tau_start),
                float(sample.difficulty.item()),
                float(sample.audio_frame_start),
                float(sample.audio_padded_frames),
                float(sample.pair_index),
                float(sample.window_index),
            )
        )
        flat = counts.reshape(-1)
        nz = np.nonzero(flat)[0]
        self._counts_idx.extend(int(v) for v in nz)
        self._counts_val.extend(int(v) for v in flat[nz])
        self._counts_ptr.append(len(self._counts_idx))
        self._occl_tok.extend(int(v) for v in np.nonzero(token.reshape(-1))[0])
        self._occl_ptr.append(len(self._occl_tok))
        self._tracks.append(np.asarray(sample.line_tracks.numpy(), dtype=np.float32))
        self._tracks_ptr.append(self._tracks_ptr[-1] + k)
        audio = np.asarray(sample.audio_emb.numpy(), dtype=self._audio_dtype)
        self._audio.append(audio)
        self._audio_ptr.append(self._audio_ptr[-1] + int(audio.shape[0]))
        self._written += 1
        if len(self._meta) >= self._shard_windows:
            self._flush()

    def close(self) -> int:
        """刷最后一个 shard，返回本 writer 实际写入的窗口数。

        为什么不在这里写 `index.json`：并行构建时每个 worker 各写一段 shard，索引必须由
        **汇总方**在全部 worker 结束后写（否则会留下「索引说 N 个窗口、实际只落了一半」的半成品）。
        """
        self._flush()
        return self._written

    def _flush(self) -> None:
        if not self._meta:
            return
        target = _shard_dir(self._directory, self._shard)
        target.mkdir(parents=True, exist_ok=True)
        np.save(target / "meta.npy", np.asarray(self._meta, dtype=np.float64))
        np.save(target / "counts_ptr.npy", np.asarray(self._counts_ptr, dtype=np.int64))
        np.save(target / "counts_idx.npy", np.asarray(self._counts_idx, dtype=np.int32))
        np.save(target / "counts_val.npy", np.asarray(self._counts_val, dtype=np.int16))
        np.save(target / "occl_ptr.npy", np.asarray(self._occl_ptr, dtype=np.int64))
        np.save(target / "occl_tok.npy", np.asarray(self._occl_tok, dtype=np.int32))
        np.save(target / "tracks_ptr.npy", np.asarray(self._tracks_ptr, dtype=np.int64))
        tracks = (
            np.concatenate(self._tracks, axis=0)
            if self._tracks
            else np.zeros((0, self._t_bins, N_ORDINARY_TRACKS), dtype=np.float32)
        )
        np.save(target / "tracks.npy", tracks)
        np.save(target / "audio_ptr.npy", np.asarray(self._audio_ptr, dtype=np.int64))
        audio = (
            np.concatenate(self._audio, axis=0)
            if self._audio
            else np.zeros((0, self._feature_dim), dtype=self._audio_dtype)
        )
        np.save(target / "audio.npy", audio)
        self._shard += 1
        self._meta = []
        self._counts_idx = []
        self._counts_val = []
        self._counts_ptr = [0]
        self._occl_tok = []
        self._occl_ptr = [0]
        self._tracks = []
        self._tracks_ptr = [0]
        self._audio = []
        self._audio_ptr = [0]


def _token_view(occlusion: NDArray[np.bool_], *, k: int, t_bins: int) -> NDArray[np.bool_]:
    """把稠密遮盖压成 `(K, T)` token 位图，并**断言它确实是 token 区块结构**。

    为什么可以压：`build_occlusion_batch` 的遮盖是按 token（线, tau）成块扩张的
    （`generation/masks.py` 的三条返回路径都做 `reshape(k, t, 1, 1, 1).expand_as(...)`）。
    但这是**实现细节**，所以这里不假设：压完立刻展开比对，不等就**抛**
    （宁可在构建期炸，也不要在训练期给模型喂错掩码）。
    """
    cells = occlusion.shape[2] * occlusion.shape[3] * occlusion.shape[4]
    token = np.asarray(occlusion.reshape(k, t_bins, cells).any(axis=2), dtype=np.bool_)
    expanded = np.broadcast_to(token.reshape(k, t_bins, 1, 1, 1), occlusion.shape)
    if not np.array_equal(expanded, occlusion):
        raise WindowCacheError(
            "遮盖不是 token 区块结构：本缓存的紧凑表示不适用（不要静默退化，请检查 masks.py）"
        )
    return token


def write_index(directory: Path, index: WindowCacheIndex) -> None:
    """原子写 `index.json`（**汇总方**在全部 shard 落盘之后调用）。"""
    directory.mkdir(parents=True, exist_ok=True)
    _write_index(directory, index)


def windows_on_disk(directory: Path, index: WindowCacheIndex) -> int:
    """数一遍盘上真实的窗口数（写索引前的最后一道校验）。"""
    total = 0
    for shard in range(index.n_shards):
        path = _shard_dir(directory, shard) / "meta.npy"
        try:
            total += int(np.load(path, mmap_mode="r").shape[0])
        except OSError as exc:
            raise WindowCacheError(f"shard {shard} 缺失或不可读（{exc}）：{path}") from exc
    return total


def _write_index(directory: Path, index: WindowCacheIndex) -> None:
    payload = {
        "split": index.split,
        "fingerprint": index.fingerprint,
        "n_windows": index.n_windows,
        "shard_windows": index.shard_windows,
        "t_bins": index.t_bins,
        "x_bins": index.x_bins,
        "sides": index.sides,
        "channels": index.channels,
        "feature_dim": index.feature_dim,
        "audio_dtype": index.audio_dtype,
        "version": index.version,
    }
    temporary = directory / (INDEX_FILENAME + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(directory / INDEX_FILENAME)


def read_index(directory: Path, *, fingerprint: str) -> WindowCacheIndex | None:
    """读索引；**任何异常 / 指纹不符 / 版本不符都返回 None**（调用方回退到原路径）。"""
    path = directory / INDEX_FILENAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        index = WindowCacheIndex(**payload)
    except (OSError, ValueError, TypeError) as exc:
        logger.warning("窗口缓存索引不可用（%s），回退到原路径：%s", exc, path)
        return None
    if index.fingerprint != fingerprint:
        logger.warning(
            "窗口缓存指纹不符（存 %s / 算 %s），回退到原路径：%s",
            index.fingerprint[:12],
            fingerprint[:12],
            path,
        )
        return None
    if index.version != WINDOW_CACHE_VERSION:
        logger.warning("窗口缓存版本 %d != %d，回退到原路径", index.version, WINDOW_CACHE_VERSION)
        return None
    return index


# ══════════════════════════════════════════════════════════════
# 读（训练路径）
# ══════════════════════════════════════════════════════════════


class _Shard:
    """一个 shard 的 `mmap` 只读视图（10 个数组，全部惰性分页）。

    为什么是 `mmap` 而不是 `np.load`：页缓存**跨 worker、跨 epoch 共享**
    —— 这是本方案「加 worker 不再吃内存」的全部依据；每个 worker 的私有内存近似为 0。
    """

    __slots__ = (
        "audio",
        "audio_ptr",
        "counts_idx",
        "counts_ptr",
        "counts_val",
        "meta",
        "occl_ptr",
        "occl_tok",
        "tracks",
        "tracks_ptr",
    )

    def __init__(self, directory: Path) -> None:
        # 逐项 np.load（不抽助手）：mmap 视图的类型各不相同，抽成 Any 会被 ANN401 拦下，
        # 而「显式列出 10 个数组」本身就是格式的一部分，写开更好读。
        def _load(name: str) -> Any:  # noqa: ANN401 - np.load 的重载对 mmap_mode 返回类型不收敛
            return np.load(directory / f"{name}.npy", mmap_mode="r")

        self.meta = _load("meta")
        self.counts_ptr = _load("counts_ptr")
        self.counts_idx = _load("counts_idx")
        self.counts_val = _load("counts_val")
        self.occl_ptr = _load("occl_ptr")
        self.occl_tok = _load("occl_tok")
        self.tracks_ptr = _load("tracks_ptr")
        self.tracks = _load("tracks")
        self.audio_ptr = _load("audio_ptr")
        self.audio = _load("audio")

    def __len__(self) -> int:
        return int(self.meta.shape[0])


class WindowCacheReader:
    """窗口缓存的只读视图：`index -> PairSample`，**不读谱面、不读特征缓存**。

    Args:
        directory: 本 split 的缓存目录。
        index: `index.json` 解出的索引。
        rows: 与构建期**同一份、同一顺序**的行序列（`song_key` / `chart_path`
            等逐行字段从它取，缓存里不重复存）。
        split: split 名（原样带进 `PairSample`）。
        shard_cache: 同时驻留的 shard 数（LRU）。取批顺序跨桶轮转 ⇒ 命中率不高，
            因此只用来吸收同 shard 内的连续访问。
    """

    def __init__(
        self,
        directory: Path,
        index: WindowCacheIndex,
        rows: Sequence[PairRow],
        *,
        split: str,
        shard_cache: int = 4,
    ) -> None:
        self._directory = directory
        self._index = index
        self._rows = rows
        self._split = split
        self._shard_cache = max(1, int(shard_cache))
        self._shards: OrderedDict[int, _Shard] = OrderedDict()

    def __len__(self) -> int:
        return self._index.n_windows

    @property
    def index(self) -> WindowCacheIndex:
        """底层索引（诊断用）。"""
        return self._index

    def sample(self, index: int) -> PairSample:
        """重建第 `index` 个窗口的 `PairSample`（逐位等于原路径）。"""
        from beatmorph.core.contracts import BpmPoint
        from beatmorph.data.dataset import PairSample
        from beatmorph.generation.masks import assert_hold_pairs_not_split

        shard_id, local = self._index.shard_of(index)
        shard = self._shard(shard_id)
        meta = shard.meta[local]
        k = int(meta[0])
        shape = (k, self._index.t_bins, self._index.x_bins, self._index.sides, self._index.channels)
        counts = np.zeros(shape, dtype=np.int16)
        start, stop = int(shard.counts_ptr[local]), int(shard.counts_ptr[local + 1])
        if stop > start:
            counts.reshape(-1)[shard.counts_idx[start:stop]] = shard.counts_val[start:stop]
        token = np.zeros((k, self._index.t_bins), dtype=np.bool_)
        start, stop = int(shard.occl_ptr[local]), int(shard.occl_ptr[local + 1])
        if stop > start:
            token.reshape(-1)[shard.occl_tok[start:stop]] = True
        occlusion = np.broadcast_to(token.reshape(k, self._index.t_bins, 1, 1, 1), shape).copy()
        counts_t = torch.as_tensor(counts)
        occlusion_t = torch.as_tensor(occlusion)
        # 与 `dataset._build_sample` 同一条不变量（构建期已过一次，这里**不省**：
        # 「为了快一点就少一道校验」正是本项目反复付过代价的那条路）。
        assert_hold_pairs_not_split(counts_t, occlusion_t)
        start, stop = int(shard.tracks_ptr[local]), int(shard.tracks_ptr[local + 1])
        tracks = torch.as_tensor(np.array(shard.tracks[start:stop], dtype=np.float32, copy=True))
        start, stop = int(shard.audio_ptr[local]), int(shard.audio_ptr[local + 1])
        audio = np.array(shard.audio[start:stop], dtype=np.float32, copy=True)
        row_index = int(meta[6])
        row = self._rows[row_index]
        return PairSample(
            audio_emb=torch.as_tensor(audio, dtype=torch.float32),
            frame_rate=MERT_FRAME_RATE_HZ,
            line_tracks=tracks,
            line_mask=torch.ones(k, dtype=torch.bool),
            difficulty=torch.tensor(float(meta[3]), dtype=torch.float32),
            grid=FieldGrid(
                x_bins=self._index.x_bins,
                t_bins=self._index.t_bins,
                bpm_points=(BpmPoint(time_beats=0.0, bpm=float(meta[1])),),
            ),
            counts=counts_t,
            occlusion=occlusion_t,
            chart_id=row.chart_id,
            song_key=row.song_key,
            split=self._split,
            pair_index=row_index,
            window_index=int(meta[7]),
            tau_start=float(meta[2]),
            chart_path=row.chart_path,
            feature_key=row.feature_key,
            audio_frame_start=int(meta[4]),
            audio_padded_frames=int(meta[5]),
        )

    def _shard(self, shard_id: int) -> _Shard:
        cached = self._shards.get(shard_id)
        if cached is not None:
            self._shards.move_to_end(shard_id)
            return cached
        loaded = _Shard(_shard_dir(self._directory, shard_id))
        self._shards[shard_id] = loaded
        while len(self._shards) > self._shard_cache:
            self._shards.popitem(last=False)
        return loaded
