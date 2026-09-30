"""RFC-0039 R3：每 N 步端到端生成一张谱面，供**人工审阅**（plan 05 解码链的第一条真实用例）。

为什么需要它：`val/nll` 与 `val/ratio` 回答的是「模型在遮盖测度下拟合得多好」，**回答不了
「它会不会写谱」**。本模块给出唯一的端到端产物：

    整首歌 -> 逐窗迭代并行解码（`generation/sampling`）-> λ 场
          -> 解码事件（`decoder/peaks` + `pair_events`，**逐窗**）
          -> 事件层后处理 + 谱面层后处理（`decoder/postprocess`，**整谱一次**）
          -> RPEJSON 落盘（**违规项非空则拒绝导出**；红线 6）

三条纪律（RFC-0039 R3 的原文）：

1. **不是门禁**：不产生 PASS/FAIL、不阻塞训练、不参与选权重。失败只告警（异常由调用方
   `train_loop` 收住），与 val 的失败降级同口径；
2. **固定不换曲**：`tests/e2e-val/meta.json` 指到哪首就永远那首，跨 step 才可比；
3. **数据不入库**：`audio/` `feature/` `outputs/` 全部在 `.gitignore` 里（红线 5 附注②），
   只有指针清单（`meta.json`）入库。

**为什么逐窗解码，而不是「拼一条整谱 λ 再解一次」**：整谱 λ 的内存是
`K × T_full × X × S × C`——214 秒的歌在 120 BPM 下 T_full 约 2.1 万格，K=30 时约 3 GB
（float32），而本机训练期 commit 已经贴过上限（plan 07 §9-49）。逐窗把峰值压回一个窗口
（K=30 时约 30 MB）。代价是**跨窗的 Hold 会被窗边界截断**——训练窗口本身也禁止 Hold 跨窗
（`skipped_windows_hold_split`），因此这是与训练同口径的近似，不是新引入的失真；
它如实记在产物的 `meta.json` 里（`windowed_decode`）。

**条件从哪来**（R3 要求「实现时定，并写进产物元数据」）：

1. `meta.json` 里显式给的 `chart`；
2. 否则**按音频内容 sha1 在训练清单里找回该曲的谱面**——取它的判定线事件轨 + BPMList +
   定数（这是「条件生成」的原义。实测 `tests/e2e-val` 的《赴大荒》就在清单里：
   chart_id=15831 / split=test / 定数 15.3）；
3. 都没有 ⇒ **合成模板**（`bpm` / `lines` 来自 meta.json，事件轨全空）：一律如实记为
   `template.source="synthetic"`，因为它等价于**无条件生成**，与条件生成的产物不可比。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from beatmorph.core.contracts.phigros import (
    BpmPoint,
    ChartMeta,
    ChartSource,
    JudgeLine,
    PhigrosChart,
)
from beatmorph.core.logging import get_logger
from beatmorph.decoder import (
    LegalityConfig,
    PairingStats,
    PeakConfig,
    chart_from_events,
    decode_peaks,
    merge_reports,
    pair_events,
    postprocess_chart,
    postprocess_events,
    scorable_lines,
    to_numpy,
)
from beatmorph.field.grid import FieldGrid
from beatmorph.generation.model import MaskedFieldModel

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from beatmorph.core.contracts import LegalityReport, PhigrosChart
    from beatmorph.data.dataset import SongWindows
    from beatmorph.decoder import DecodedEvent
    from beatmorph.generation.sampling import SamplingConfig
    from beatmorph.infra.config.schema import TrainConfig

logger = get_logger("infra.e2e")

#: e2e 资产的默认目录（**相对仓库根**；`meta.json` 入库，其余按目录忽略）。
DEFAULT_E2E_DIR: Path = Path("tests/e2e-val")

#: 产物落点（相对 e2e 目录）：`<...>/outputs/<YYYYMMDD-HHMMSS>-step<N>/`。
OUTPUT_DIRNAME: str = "outputs"

#: 合成模板的兜底值（只在找不到谱面行、且 meta.json 没给时使用；都会写进产物元数据）。
DEFAULT_SYNTHETIC_BPM: float = 120.0
DEFAULT_SYNTHETIC_LINES: int = 16
DEFAULT_DIFFICULTY: float = 15.0

#: 迭代并行解码的步数（`SamplingConfig` 的下界是 2，见 `generation/sampling`）。
DEFAULT_SAMPLING_STEPS: int = 8

#: D1 峰值解码的阈值系数默认值（= `PeakConfig.alpha` 的默认值，即「不额外声明自由度」）。
DEFAULT_PEAK_ALPHA: float = 1.0

#: 默认解码臂（见 `E2EInputs.method` 的说明：D1 的阈值未标定前，产物默认走 D2）。
DEFAULT_DECODE_METHOD: str = "thinning"

#: 一条谱面**允许解出的最大事件数**（内存与墙钟的硬闸）。
#:
#: 为什么必须有闸：D1 的阈值是 `alpha * lambda_0`，而 `lambda_0` 是**场的平均强度**
#: （`intensity_scale`）。在真实训练场上实测（plan 07 §9-63）：`alpha=1` 时单个 1.6 秒窗口
#: 就解出 **15.7 万个**场事件（而该窗口的期望事件数是 12）——逐窗累加整首歌会造出**千万级**
#: 的 `DecodedEvent` 对象（实测 21 GB 常驻、单核跑满、GPU 空转）。D1 的阈值标定是 plan 05
#: M5.7 的事；在那之前，本闸保证「人工审阅件」**永远不会把长跑拖垮或 OOM**。
#: 60 000 个事件 ≈ 3 万音符，已经远超任何可审阅的谱面。
MAX_DECODED_EVENTS: int = 60_000

#: 逐窗进度日志的间隔（窗口数）：e2e 是分钟级的一次性动作，静默 = 看起来像卡死。
PROGRESS_EVERY: int = 20


class E2EInputError(RuntimeError):
    """e2e 资产缺失或不可读（**报错而非静默跳过**：跳过等于「这条通道从来没生效过」）。"""


@dataclass(frozen=True, slots=True)
class E2EInputs:
    """`meta.json` 解析结果（路径全部已解析成绝对路径）。"""

    directory: Path
    audio: Path
    feature: Path
    feature_meta: Path | None
    chart: Path | None
    bpm: float
    lines: int
    difficulty: float
    steps: int
    #: 解码臂（`decoder` 的 B6 两选一）："thinning"（默认）或 "peaks"。
    #:
    #: **首版默认 thinning**，理由是一条实测（plan 07 §9-63）：D1（peaks）的阈值是
    #: `alpha * lambda_0`，而 `lambda_0` 是场的**平均**强度；在 arm H 的真实场上 α=1
    #: 单个 1.6 秒窗口就解出 **8.7–15.7 万个**场事件（该窗口模型自己的期望事件数是
    #: 0.02–2.96）。D1 的阈值标定是 plan 05 M5.7 的事，未标定前它的产物**不可审阅**。
    #: D2（thinning）按场强度抽样 ⇒ 事件数自动等于模型的期望值 ∫λdV，是「模型预测的
    #: 忠实抽样」。两个臂的读数都会进产物元数据（`decode` 段）。
    method: str
    #: D1 的阈值系数（只用 `method="peaks"` 时）；见 `decoder/peaks.py`。
    alpha: float
    #: 解码随机种子的基准（每个窗口用 `seed + 窗口下标` ⇒ 整首歌可复现）。
    seed: int


def load_e2e_inputs(directory: Path | str) -> E2EInputs:
    """读 `<directory>/meta.json` 并解析出端到端输入（缺文件即报错）。

    字段（只有 `audio` / `feature` 是必需）：

    - `audio`：原始音频（只用来算 sha1 与记录来源；特征才是模型输入）；
    - `feature`：MERT 特征 npz；`feature_meta` 是它的元数据 json（默认同名 `.meta.json`，
      本仓库的资产用的是 `fudahaung.json` 这个手写名 ⇒ 必须能在 meta.json 里显式指路）；
    - `chart`（可选）：模板谱面（RPEJSON）。省略时按音频 sha1 去训练清单里找；
    - `bpm` / `lines` / `difficulty` / `steps`（可选）：合成模板与采样器的参数。
    """
    root = Path(directory)
    meta_path = root / "meta.json"
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise E2EInputError(f"e2e 清单不可用（{meta_path}）：{exc}") from exc
    if not isinstance(payload, dict):
        raise E2EInputError(f"e2e 清单必须是 JSON 对象：{meta_path}")

    def _required(key: str) -> Path:
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            raise E2EInputError(f"e2e 清单缺少 {key!r}（{meta_path}）")
        path = (root / value).resolve()
        if not path.exists():
            raise E2EInputError(f"e2e 清单里的 {key} 不存在：{path}")
        return path

    def _optional(key: str) -> Path | None:
        value = payload.get(key)
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            raise E2EInputError(f"e2e 清单的 {key!r} 必须是非空字符串或 null（{meta_path}）")
        return (root / value).resolve()

    audio = _required("audio")
    feature = _required("feature")
    feature_meta = _optional("feature_meta")
    if feature_meta is None:
        default_meta = feature.with_suffix(".meta.json")
        if default_meta.exists():
            feature_meta = default_meta
    chart = _optional("chart")
    return E2EInputs(
        directory=root.resolve(),
        audio=audio,
        feature=feature,
        feature_meta=feature_meta,
        chart=chart,
        bpm=float(payload.get("bpm") or DEFAULT_SYNTHETIC_BPM),
        lines=int(payload.get("lines") or DEFAULT_SYNTHETIC_LINES),
        difficulty=float(payload.get("difficulty") or DEFAULT_DIFFICULTY),
        steps=int(payload.get("steps") or DEFAULT_SAMPLING_STEPS),
        method=str(payload.get("method") or DEFAULT_DECODE_METHOD),
        alpha=float(payload.get("alpha") or DEFAULT_PEAK_ALPHA),
        seed=int(payload.get("seed") or 0),
    )


def sha1_file(path: Path) -> str:
    """文件内容 sha1（来源留痕；与 `data/pipeline/embed.audio_cache_key` 同口径）。"""
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_corpus_row(
    audio_key: str,
    *,
    manifest_path: Path,
) -> tuple[str, dict[str, Any]] | None:
    """在训练清单里按**音频内容 sha1** 找回该曲的行（`(split, row)`；找不到返回 None）。

    为什么按 `feature_key`：清单里的 `feature_key` 就是音频 sha1（`audio_cache_key`），
    它是**内容**身份而不是文件名身份——同名的不同曲子不会误配。
    """
    from beatmorph.data.dataset import DatasetManifestError, load_pairs
    from beatmorph.data.pipeline.embed import SPLIT_NAMES

    rows_seen = 0
    for split in SPLIT_NAMES:
        try:
            rows = load_pairs(Path(manifest_path), split)
        except (DatasetManifestError, OSError) as exc:
            # 清单里缺这个 split / 文件不可读：对「找回这首曲子的谱面」而言都只是「找不到」，
            # 不该让整条 e2e 通道崩掉（它本来就不是门禁）。
            logger.debug("e2e 清单查找跳过 split=%s：%s", split, exc)
            continue
        rows_seen += len(rows)
        for row in rows:
            if row.feature_key == audio_key:
                return split, {
                    "chart_id": row.chart_id,
                    "chart_path": row.chart_path,
                    "difficulty": row.difficulty,
                    "song_key": row.song_key,
                }
    logger.info("e2e 清单查找：%d 行里没有 feature_key=%s", rows_seen, audio_key[:12])
    return None


def resolve_template(
    inputs: E2EInputs,
    *,
    chart_dir: Path,
    manifest_path: Path,
    audio_seconds: float,
) -> tuple[PhigrosChart, dict[str, Any]]:
    """定下「条件」：显式 chart > 训练清单里该曲的谱面 > 合成模板（见模块 docstring）。"""
    from beatmorph.data.parsers.rpejson import parse_rpejson

    if inputs.chart is not None:
        chart = parse_rpejson(inputs.chart.read_bytes(), ChartSource(sniff_evidence="e2e"))
        return chart, {
            "source": "meta.json",
            "path": str(inputs.chart),
            "sha1": sha1_file(inputs.chart),
        }
    row = find_corpus_row(sha1_file(inputs.audio), manifest_path=Path(manifest_path))
    if row is not None:
        split, payload = row
        path = Path(chart_dir) / str(payload["chart_path"])
        if not path.exists():
            raise E2EInputError(f"清单里的谱面不存在：{path}")
        chart = parse_rpejson(path.read_bytes(), ChartSource(sniff_evidence="e2e"))
        return chart, {
            "source": f"manifest:{split}",
            "path": str(path),
            "sha1": sha1_file(path),
            "chart_id": payload["chart_id"],
            "song_key": payload["song_key"],
        }
    logger.warning(
        "e2e：清单里找不到这首曲子的谱面 ⇒ 退化为**无条件生成**（合成模板 bpm=%g / K=%d）",
        inputs.bpm,
        inputs.lines,
    )
    template = PhigrosChart(
        lines=[JudgeLine(line_id=index) for index in range(int(inputs.lines))],
        notes=[],
        bpm_points=[BpmPoint(time_beats=0.0, bpm=float(inputs.bpm))],
        meta=ChartMeta(chart_time_s=float(audio_seconds)),
        source=ChartSource(sniff_evidence="e2e:synthetic"),
    )
    return template, {
        "source": "synthetic",
        "path": None,
        "sha1": None,
        "bpm": float(inputs.bpm),
        "lines": int(inputs.lines),
    }


def axis_end_seconds(cfg: TrainConfig, chart: PhigrosChart, audio_seconds: float) -> float:
    """τ 轴终点（与训练**同一口径**：显式覆盖 > policy；policy=audio 取两者较小）。

    这里只做「两个秒数取 min」，不涉及任何秒 <-> 拍换算（红线 7 由 `FieldGrid` 承担）。
    """
    override = cfg.data.tau_end_s
    if override is not None:
        return float(override)
    if cfg.data.tau_end_policy == "chart":
        return float(chart.duration_s())
    return min(float(chart.duration_s()), float(audio_seconds))


#: 逐窗**求和**的解码统计键（计数类：跨窗累加才有意义）。
#:
#: ⚠️ 漏掉一个计数键的代价是**读数看起来是总数、其实是第一个窗口的值**——实测第一次跑
#: 产物时 `d2_n_events` 就因为这个写成了 0.0（第一个窗口是安静的前奏），而实际解出 665 个事件。
_STAT_SUM_KEYS: tuple[str, ...] = (
    "d1_n_events",
    "d2_n_events",
    "d2_n_candidates",
    "d2_n_restarts",
    "d2_n_empty_lines",
)

#: 逐窗取**范围**的解码统计键（随窗口的场强度变化，取 min/max 才有信息量）。
_STAT_RANGE_KEYS: tuple[str, ...] = ("d1_scale", "d1_threshold")


def _accumulate_decode_stats(acc: dict[str, float], stats: Mapping[str, float]) -> None:
    """把逐窗解码统计累加进账单（**绝不丢弃**）。

    为什么单列一个函数并强调「不丢」：R3 的第一版实现把 `decode_peaks` 的 stats 丢掉了，
    于是「每个窗口解出 15.7 万个场事件」这件事**在产物里完全看不见**——直到内存爆掉才
    暴露（plan 07 §9-63 的实测）。解码器的诊断量必须一路带到产物元数据里。
    """
    for key, value in stats.items():
        number = float(value)
        if key in _STAT_SUM_KEYS:
            acc[key] = acc.get(key, 0.0) + number
        elif key in _STAT_RANGE_KEYS:
            acc[f"{key}_min"] = min(acc.get(f"{key}_min", number), number)
            acc[f"{key}_max"] = max(acc.get(f"{key}_max", number), number)
        else:
            acc.setdefault(key, number)


def _merge_pairing(stats: Sequence[PairingStats]) -> PairingStats:
    """跨窗累加 Hold 配对统计（逐窗解码后要合成一份报告）。"""
    return PairingStats(
        n_starts=sum(item.n_starts for item in stats),
        n_ends=sum(item.n_ends for item in stats),
        n_paired=sum(item.n_paired for item in stats),
        n_unpaired_starts=sum(item.n_unpaired_starts for item in stats),
        n_orphan_ends=sum(item.n_orphan_ends for item in stats),
        n_zero_length=sum(item.n_zero_length for item in stats),
    )


@dataclass(frozen=True, slots=True)
class E2EResult:
    """一次端到端生成的结果（谱面 + 报告 + 记账）。"""

    chart: PhigrosChart
    report: LegalityReport
    inputs: E2EInputs
    template: dict[str, Any]
    grid: FieldGrid
    difficulty: float
    steps: int
    events: tuple[DecodedEvent, ...]
    pairing: PairingStats
    n_windows: int
    #: 实际跑完的窗口数（中止时 < `n_windows`）。
    windows_done: int
    dropped_tail_bins: int
    elapsed_s: float
    #: 逐窗解码统计的累加账单（`d1_n_events` 求和、`d1_threshold` 取范围……）。
    decode_stats: Mapping[str, float]
    #: 中止原因（None = 正常跑完）；中止时**不写** `chart.json`，但元数据照写。
    aborted: str | None = None

    @property
    def is_legal(self) -> bool:
        """红线 6 的唯一判据：`violations` 为空才允许导出。"""
        return self.report.is_legal

    @property
    def is_complete(self) -> bool:
        """是否**完整**产出了一张谱面（合法 且 未被预算闸中止）。"""
        return self.is_legal and self.aborted is None

    def summary(self) -> str:
        """一行人类可读摘要（日志与 `notes.txt` 用同一份，不许两套口径）。"""
        verdict = "中止" if self.aborted is not None else ("合法" if self.is_legal else "不合法")
        return (
            f"{self.chart.meta.chart_time_s:.1f}s / K={len(self.chart.lines)} / "
            f"{len(self.chart.notes)} note / 事件 {len(self.events)} / "
            f"窗 {self.windows_done}/{self.n_windows} x {self.steps} 步 / "
            f"模板 {self.template['source']} / {self.elapsed_s:.1f}s / {verdict}"
        )


@dataclass(frozen=True)
class LineFilter:
    """「哪些线、在哪些时刻可以承载 note」这一条条件（决策者实测 bug 的修法）。

    为什么要有它：模型在 K 条线上出强度场，而 K 条线里**未必**都该有 note —— 实测 e2e 产物
    665 个 note 里 238 个（35.8%）落在 note 时刻不透明度 = 0 的线上（游戏里根本看不见，
    等于不可判定；alpha 经 judgeLineList 的 alphaEvents 跨层求和，prpr A 级语义），
    149 个（22.4%）落在条件谱面里零 note 的装饰 / 表演线上。

    allowed_lines = None 表示「没有这个信息」（无条件生成 / 合成模板）⇒ 不按装饰线过滤、
    只按 alpha 过滤；两条判据都在 decoder.events.filter_field_events_by_line 里实现。
    """

    chart: PhigrosChart
    allowed_lines: frozenset[int] | None = None


@dataclass(frozen=True, slots=True)
class _WindowRun:
    """逐窗生成的中间产物（把长循环从 `generate_chart` 里拆出来；理由见 plan 07 §9-63）。"""

    events: list[DecodedEvent]
    pairings: list[PairingStats]
    decode_stats: dict[str, float]
    windows_done: int
    aborted: str | None


def _run_windows(
    model: MaskedFieldModel,
    windows: SongWindows,
    *,
    inputs: E2EInputs,
    sampling: SamplingConfig,
    peak: PeakConfig,
    method: str,
    max_events: int,
    device: torch.device,
    precision: str,
    line_filter: LineFilter | None = None,
) -> _WindowRun:
    """逐窗：迭代并行解码 -> 解码事件 -> 平移回整谱时间基（**流式，不累计原始场**）。

    **内存纪律**（plan 07 §9-63 的实测教训）：只保留 `DecodedEvent`；一旦累计事件数将超过
    `max_events` 就**立即中止**并如实记账——第一版实现把整首歌的场事件全留着，实测
    21 GB 常驻、单核跑满、GPU 空转（145 窗 × 12 万个事件）。
    """
    from beatmorph.decoder import decode_thinning
    from beatmorph.decoder.thinning import ThinningConfig
    from beatmorph.generation.sampling import sample
    from beatmorph.infra.train_loop import autocast_context

    events: list[DecodedEvent] = []
    pairings: list[PairingStats] = []
    decode_stats: dict[str, float] = {}
    aborted: str | None = None
    windows_done = 0
    total_windows = windows.n_windows()
    k = windows.n_lines()
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        for index in range(total_windows):
            batch = windows.batch(index).to(device)
            with autocast_context(precision, device):
                output = sample(model, batch, config=sampling)
            window_grid = windows.window_grid(index)
            spec = window_grid.spec(k)
            lam_np = to_numpy(output.lam[0])
            _accumulate_expected_events(decode_stats, lam_np, window_grid)
            field_events, stats = (
                decode_peaks(lam_np, window_grid, spec, config=peak)
                if method == "peaks"
                else decode_thinning(
                    lam_np, window_grid, spec, config=ThinningConfig(seed=int(inputs.seed) + index)
                )
            )
            _accumulate_decode_stats(decode_stats, stats)
            origin = windows.window_start_seconds(index)
            if line_filter is not None:
                from beatmorph.decoder.events import filter_field_events_by_line

                field_events, filter_stats = filter_field_events_by_line(
                    field_events,
                    chart=line_filter.chart,
                    window_grid=window_grid,
                    origin_s=origin,
                    allowed_lines=line_filter.allowed_lines,
                )
                _accumulate_decode_stats(decode_stats, filter_stats)
            if len(field_events) > max_events - len(events):
                aborted = _budget_message(
                    index=index,
                    total=total_windows,
                    found=len(field_events),
                    kept=len(events),
                    stats=stats,
                    decode_stats=decode_stats,
                    method=method,
                    peak=peak,
                    max_events=max_events,
                )
                logger.warning("e2e 中止：%s", aborted)
                break
            decoded, pairing = pair_events(field_events, window_grid, spec=spec)
            events.extend(replace(event, t_s=event.t_s + origin) for event in decoded)
            pairings.append(pairing)
            windows_done = index + 1
            if windows_done % PROGRESS_EVERY == 0 or windows_done == total_windows:
                logger.info(
                    "e2e：窗口 %d/%d（累计解码事件 %d，累计场事件 %.0f，模型期望 %.1f）",
                    windows_done,
                    total_windows,
                    len(events),
                    decode_stats.get("n_field_events", 0.0),
                    decode_stats.get("model_expected_events", 0.0),
                )
    finally:
        model.train(was_training)
    return _WindowRun(
        events=events,
        pairings=pairings,
        decode_stats=decode_stats,
        windows_done=windows_done,
        aborted=aborted,
    )


def _accumulate_expected_events(
    acc: dict[str, float],
    lam: object,
    grid: FieldGrid,
) -> None:
    """累加模型的**期望事件数** ∫λdV（判断「解出多少事件才算合理」的唯一基准）。

    第一版实现没有这个量，于是「每窗解出 15.7 万个事件」无从判别——它必须与解码器自己的
    计数一起进产物元数据。
    """
    import numpy as np

    values = to_numpy(lam)
    volumes = np.asarray(grid.cell_volumes(), dtype=np.float64).reshape(1, -1, 1, 1, 1)
    key = "model_expected_events"
    acc[key] = acc.get(key, 0.0) + float((values * volumes).sum())


def _budget_message(
    *,
    index: int,
    total: int,
    found: int,
    kept: int,
    stats: Mapping[str, float],
    decode_stats: Mapping[str, float],
    method: str,
    peak: PeakConfig,
    max_events: int,
) -> str:
    """预算用尽的可操作说明（把「模型期望多少 / 解出多少 / 阈值多少」一次说清）。"""
    detail = (
        f"事件预算 {max_events} 用尽：在第 {index + 1}/{total} 个窗口上本窗就解出 "
        f"{found} 个事件（累计 {kept} 个；模型自己的期望事件数是 "
        f"{decode_stats.get('model_expected_events', float('nan')):.1f}）。"
    )
    if method == "peaks":
        detail += (
            f"解码臂 peaks 的阈值 alpha={peak.alpha:g}"
            f"（该窗阈值 {stats.get('d1_threshold', float('nan')):.4g}）"
            "远低于真实场的噪声底：D1 的阈值标定是 plan 05 M5.7 的事，先改用 method=thinning"
        )
    return detail


def generate_chart(
    model: MaskedFieldModel,
    cfg: TrainConfig,
    inputs: E2EInputs,
    *,
    device: torch.device,
    precision: str = "fp32",
    max_events: int = MAX_DECODED_EVENTS,
) -> E2EResult:
    """整首歌端到端生成（逐窗迭代并行解码 + 逐窗解码事件 + 整谱后处理）。

    **不做合法性判定之外的任何筛选**：模型给出什么就解什么，违规项交给
    `postprocess_*` 修（修不掉就是 `is_legal=False`，调用方据此拒绝导出）。
    """
    from beatmorph.data.dataset import SongWindows
    from beatmorph.field.grid import FieldGrid
    from beatmorph.generation.sampling import SamplingConfig
    from beatmorph.infra.feature_cache import load_feature_cache_checked

    started = time.perf_counter()
    embedding, feature_meta = load_feature_cache_checked(inputs.feature, inputs.feature_meta)
    template, template_info = resolve_template(
        inputs,
        chart_dir=Path(cfg.data.chart_dir),
        manifest_path=Path(cfg.data.manifest_path),
        audio_seconds=float(feature_meta.duration_s),
    )
    # 判定线资格闸门（决策者 2026-09-30 实测报告）：装饰 / 表演线不该有 note，
    # 且 note 时刻不可见的线判不到——两条判据都在解码事件层（配对之前）拦掉。
    # ⚠️ 判据必须是「有**可计分** note」（非 fake 且命中时线可见），**不是**「有 note」：
    # 旧口径只看有没有 note，于是模板里只有假音符 / 命中时不可见的那几条表演线也被当成了
    # 「可以承载 note 的线」，实测让 23 个 note（5.6%）落到表演线上；而真实语料里
    # 「落在装饰线上的可计分 note」是 **0**（调研报告 §1/§2，本仓复核见 scorable_lines 的 docstring）。
    playable_lines = scorable_lines(template)
    line_filter = LineFilter(
        chart=template,
        allowed_lines=playable_lines if playable_lines else None,
    )
    tau_end = axis_end_seconds(cfg, template, float(feature_meta.duration_s))
    if tau_end <= 0.0:
        raise E2EInputError(f"τ 轴终点为 {tau_end}s（谱面与音频都没有时长？）")
    grid = FieldGrid(x_bins=int(cfg.data.x_bins)).for_chart(template, tau_end_s=tau_end)
    windows = SongWindows(
        chart=template,
        embedding=embedding,
        grid=grid,
        difficulty=float(inputs.difficulty),
        x_bins=int(cfg.data.x_bins),
        t_window=int(cfg.data.t_window),
    )
    if windows.n_lines() > int(cfg.model.k_max):
        raise E2EInputError(
            f"模板谱面有 {windows.n_lines()} 条判定线，超过 model.k_max={cfg.model.k_max}："
            "模型结构上就装不下（`k_max` 是 line embedding 的容量）——要么换模板，要么改 k_max",
        )
    if windows.n_windows() <= 0:
        raise E2EInputError(
            f"τ 轴只切出 {windows.n_windows()} 个窗口（t_bins={grid.t_bins} / t_window={cfg.data.t_window}）",
        )
    sampling = SamplingConfig(steps=int(inputs.steps))
    method = str(inputs.method).lower()
    if method not in ("peaks", "thinning"):
        raise E2EInputError(f"未知解码臂 {inputs.method!r}（只支持 peaks / thinning）")
    peak = PeakConfig(alpha=float(inputs.alpha))
    k = windows.n_lines()
    total_windows = windows.n_windows()
    run = _run_windows(
        model,
        windows,
        inputs=inputs,
        sampling=sampling,
        peak=peak,
        method=method,
        max_events=int(max_events),
        device=device,
        precision=precision,
        line_filter=line_filter,
    )
    events = run.events
    pairings = run.pairings
    decode_stats = run.decode_stats
    aborted = run.aborted
    windows_done = run.windows_done
    fixed, event_report = postprocess_events(events, pairing=_merge_pairing(pairings))
    last_seconds = max((event.t_s + event.hold_time_s for event in fixed), default=0.0)
    chart = chart_from_events(
        fixed,
        template=template,
        k=k,
        bpm_points=grid.bpm_points,
        chart_time_s=max(float(grid.total_seconds), last_seconds),
        method="peaks",
    )
    chart_result = postprocess_chart(chart, grid=grid, config=LegalityConfig())
    report = merge_reports(event_report, chart_result.report)
    return E2EResult(
        chart=chart_result.chart,
        report=report,
        inputs=inputs,
        template=template_info,
        grid=grid,
        difficulty=float(inputs.difficulty),
        steps=int(inputs.steps),
        events=tuple(fixed),
        pairing=_merge_pairing(pairings),
        n_windows=total_windows,
        windows_done=windows_done,
        dropped_tail_bins=windows.dropped_tail_bins(),
        elapsed_s=time.perf_counter() - started,
        decode_stats=decode_stats,
        aborted=aborted,
    )


def write_artifact(result: E2EResult, *, step: int, root: Path) -> Path:
    """把一次生成写成 `<root>/<YYYYMMDD-HHMMSS>-step<N>/`（时间命名，不覆盖历史）。

    目录内容：

    - `chart.json`：RPEJSON。**只在 `report.is_legal` 时写出**（红线 6）；
    - `meta.json`：step / 模板来源与哈希 / 音频与特征指纹 / 网格与 τ 轴口径 /
      采样器参数 / 合法性报告。**任何一次生成都写**（不合法时它是唯一证据）；
    - `notes.txt`：一行摘要（快速扫），内容与日志同源（`E2EResult.summary()`）。
    """
    from beatmorph.io.formats.rpejson import dump_rpejson

    directory = Path(root) / f"{time.strftime('%Y%m%d-%H%M%S')}-step{int(step)}"
    directory.mkdir(parents=True, exist_ok=True)
    chart_path = directory / "chart.json"
    if result.is_complete:
        dump_rpejson(result.chart, chart_path, report=result.report)
    elif result.aborted is not None:
        logger.warning("e2e 产物**不含 chart.json**：生成被中止（%s）", result.aborted)
    else:
        logger.warning(
            "e2e 产物**不含 chart.json**：合法性后处理后仍有 %d 条违规（红线 6 拒绝导出）",
            len(result.report.violations),
        )
    payload: dict[str, Any] = {
        "step": int(step),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "summary": result.summary(),
        "legal": result.is_legal,
        "aborted": result.aborted,
        "violations": [
            {"kind": item.kind, "detail": item.detail} for item in result.report.violations
        ],
        "chart_written": result.is_complete,
        "notes": len(result.chart.notes),
        "events": len(result.events),
        "k_lines": len(result.chart.lines),
        "difficulty": result.difficulty,
        "sampling": {
            "steps": result.steps,
            "schedule": "cosine",
            "confidence": "event_count",
            "state_fill": "binary",
        },
        "decode": {
            "method": result.inputs.method,
            "alpha": result.inputs.alpha,
            "seed": result.inputs.seed,
            # 逐窗解码统计的累加账单（**绝不丢**）：它才是「解出多少事件算合理」的判据。
            "aggregated": {key: float(value) for key, value in sorted(result.decode_stats.items())},
        },
        "windowed_decode": {
            "n_windows": result.n_windows,
            "windows_done": result.windows_done,
            "t_window": int(result.grid.t_bins) // max(result.n_windows, 1),
            "dropped_tail_bins": result.dropped_tail_bins,
            "tau_end_s": float(result.grid.total_seconds),
            "note": "逐窗解码：跨窗 Hold 会被窗边界截断（与训练窗口口径一致）",
        },
        "grid": {
            "x_bins": int(result.grid.x_bins),
            "t_bins": int(result.grid.t_bins),
            "bpm_points": [
                {"time_beats": float(point.time_beats), "bpm": float(point.bpm)}
                for point in result.grid.bpm_points
            ],
        },
        "audio": {
            "path": str(result.inputs.audio),
            "sha1": sha1_file(result.inputs.audio),
        },
        "feature": {
            "path": str(result.inputs.feature),
            "meta_path": None
            if result.inputs.feature_meta is None
            else str(result.inputs.feature_meta),
        },
        "template": result.template,
        "hold_pairing": result.pairing.as_stats,
        "stats": {key: float(value) for key, value in sorted(result.report.stats.items())},
        "elapsed_s": result.elapsed_s,
    }
    (directory / "meta.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (directory / "notes.txt").write_text(result.summary() + "\n", encoding="utf-8")
    return directory


def run_e2e(
    model: MaskedFieldModel,
    cfg: TrainConfig,
    *,
    step: int,
    device: torch.device,
    directory: Path | str | None = None,
) -> E2EResult:
    """跑一次端到端生成并落盘（**调用方负责容错**：R3 不是门禁，失败不得打断训练）。"""
    root = DEFAULT_E2E_DIR if directory is None else Path(directory)
    inputs = load_e2e_inputs(root)
    logger.info("e2e 生成开始：step=%d，输入 %s", step, inputs.feature.name)
    result = generate_chart(model, cfg, inputs, device=device, precision=cfg.optim.precision)
    path = write_artifact(result, step=step, root=root / OUTPUT_DIRNAME)
    logger.info("e2e 产物：%s | %s", path, result.summary())
    return result


__all__ = [
    "DEFAULT_E2E_DIR",
    "DEFAULT_SAMPLING_STEPS",
    "OUTPUT_DIRNAME",
    "E2EInputError",
    "E2EInputs",
    "E2EResult",
    "axis_end_seconds",
    "find_corpus_row",
    "generate_chart",
    "load_e2e_inputs",
    "resolve_template",
    "run_e2e",
    "sha1_file",
    "write_artifact",
]
