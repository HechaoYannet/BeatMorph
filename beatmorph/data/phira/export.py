"""谱面包**导出**：谱面 IR + 音频 + 曲绘 -> Phira 可导入的 `.pez`（plan 02 §3.2 的写侧）。

包结构与 `info.yml` 字段全部依据 **Phira 官方文档**（A 级来源）：
<https://teamflos.github.io/phira-docs/chart-standard/chartinfo.html> 与同目录的「谱面基本结构」。

1. **谱面包是一个压缩包**，解压后**根级直接是文件**（不是一个文件夹）；根级必须有 `info.yml`，
   以及它指名的其他文件（谱面 / 音乐 / 曲绘）⇒ 本模块写出的 zip **只允许根级条目**。
2. `info.yml` 是 `ChartInfo`（YAML），也是**权威元数据源**：官方明文「RPE 的 JSON 自带元数据
   **不被推荐**，Phira 以 info.yml 为准」⇒ 两边都要对，且只允许一处真相（见下）。
3. `format` 官方明文「不应当手动填写，由客户端自动识别」⇒ **一律不写**；
   `id` / `uploader` / `created` / `updated` / `chartUpdated` 由客户端或服务器管理 ⇒ **一律不写**。

三条纪律：

- **红线 6**：导出前必须 `report.is_legal`（`write_rpejson` 的门禁再挡一次）；
- **一份真相**：`name` / `composer` / `charter` / `difficulty` / `level` / `offset` 从谱面 IR 的
  `ChartMeta` 派生，调用方只能**覆盖**、不能凭空造 ⇒ 不会出现「RPE 与 info.yml 各说一套」；
- **可复现**：zip 条目时间戳固定 ⇒ 同样的输入产出**逐字节相同**的包（`deterministic=False` 可关）。

**导出后一定自检**：包会被 `ChartPackage` 重新打开（读侧是同一套 `ChartInfo`），逐项核对
「info.yml 能反解 / `chart` 与 `music` 条目存在 / 谱面字节能解析回同样多的 note」。
"""

from __future__ import annotations

import hashlib
import struct
import wave
import zipfile
import zlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from beatmorph.core.contracts.phigros import ChartSource, PhigrosChart
from beatmorph.core.logging import get_logger
from beatmorph.data.phira.package import INFO_YML_NAME, ChartPackage, parse_info_yaml
from beatmorph.decoder.postprocess.legality import check_chart
from beatmorph.io.formats.rpejson import write_rpejson

logger = get_logger(__name__)

#: zip 条目时间戳（1980-01-01 是 ZIP 格式的下限）——固定它才可能有可复现的产物。
ZIP_DATE_TIME: tuple[int, int, int, int, int, int] = (1980, 1, 1, 0, 0, 0)

#: 谱面文件在包内的默认名（`ChartInfo.chart` 的官方默认值）。
CHART_FILENAME: str = "chart.json"

#: 曲绘缺失时写出的占位图名（**自解释**：包里一眼能看出它不是真曲绘）。
PLACEHOLDER_ILLUSTRATION: str = "beatmorph-placeholder.png"

#: 占位图尺寸（16:9；纯色图压缩后只有几百字节）。
PLACEHOLDER_SIZE: tuple[int, int] = (1280, 720)

#: 未提供谱师 / 画师时写的署名（**会告警**：它是个占位符，不是事实）。
DEFAULT_CREDIT: str = "BeatMorph"

#: 「这张谱面是**生成的**」在 `ChartSource.sniff_evidence` 里的前缀。
#:
#: 来历（实测，2026-09-29）：从端到端产物导出时，解码链会**继承模板谱面的 META**
#: （`chart_from_events(template=...)` 复制 lines/bpm/meta），于是导出包里的
#: `charter` 写着模板作者的名字（实测 `shabu5553`）、`level` 写着模板的等级（`IN Lv.15`）
#: ——而那 665 个音符一个都不是他写的。**名与曲是同一首歌（可以继承），谱师不是**。
GENERATED_EVIDENCE_PREFIX: str = "decoded:"

#: 官方默认纵横比（`aspectRatio`）。
DEFAULT_ASPECT_RATIO: float = 16.0 / 9.0
#: 官方默认背景暗化（`backgroundDim`）。
DEFAULT_BACKGROUND_DIM: float = 0.6
#: 官方默认判定线长度（`lineLength`，单位待官方补充）。
DEFAULT_LINE_LENGTH: float = 6.0

#: 官方明文不写、或由客户端管理的字段（写出去就是错的）。
FORBIDDEN_INFO_FIELDS: tuple[str, ...] = (
    "format",
    "id",
    "uploader",
    "created",
    "updated",
    "chartUpdated",
)


class PhiraExportError(ValueError):
    """导出被拒绝（信息不全 / 文件缺失 / 谱面不合法 / 名字非法）。"""


@dataclass(frozen=True, slots=True)
class PhiraExportSpec:
    """一次导出的全部输入（**没有隐式默认**：每个字段都由调用方显式给出或显式留空）。

    Attributes:
        chart: 谱面 IR；写出的 `chart.json` 由 `write_rpejson` 生成（红线 6 由它把关）。
        music: 音频文件路径（**必须存在**）。
        illustration: 曲绘路径；None = 写一张自解释的占位图（见 `PLACEHOLDER_ILLUSTRATION`）。
        name / composer: 曲名与音乐作者（**必需**：官方默认值 `UK` 是占位符，不能替代事实）。
        charter / illustrator: 谱师与画师；空则用 `DEFAULT_CREDIT` 并**告警**。
        difficulty: 定数（f32）；None = 从 `chart.meta.difficulty` 取，仍为空则报错。
        level: 屏幕右下角的等级文本（自由文本，**不得**解析）；空则 `UK Lv.<定数>`。
        offset_s: `info.yml.offset`（秒）。None = 由 `chart.meta.offset_ms` 换算（RPE 是毫秒）。
            符号语义（官方）：正值 ⇒ 谱面更晚开始（等价于音乐更早）。
    """

    chart: PhigrosChart
    music: Path
    illustration: Path | None = None
    name: str = ""
    composer: str = ""
    charter: str = ""
    illustrator: str = ""
    difficulty: float | None = None
    level: str = ""
    tags: tuple[str, ...] = ()
    intro: str = ""
    tip: str = ""
    preview_start: float = 0.0
    preview_end: float | None = None
    offset_s: float | None = None
    aspect_ratio: float = DEFAULT_ASPECT_RATIO
    background_dim: float = DEFAULT_BACKGROUND_DIM
    line_length: float = DEFAULT_LINE_LENGTH
    hold_partial_cover: bool = False
    #: 包内谱面文件名（官方默认 `chart.json`）。
    chart_filename: str = CHART_FILENAME
    #: 包内音频 / 曲绘文件名；None = 沿用源文件名（规范化后）。
    music_filename: str | None = None
    illustration_filename: str | None = None
    #: 音频时长（秒）；None 时若源是 WAV 会自动读（其余格式读不到，需调用方给）。
    music_duration_s: float | None = None
    #: 「这张谱面是**生成**的」——决定**不**继承模板的谱师与等级文本（见 `GENERATED_EVIDENCE_PREFIX`）。
    #: None = 由 `ChartSource.sniff_evidence` 自动判定。
    #: ⚠️ **经 RPEJSON 往返后该标记会丢失**（`sniff_evidence` 不是 RPE 的字段）⇒ 从文件导出时
    #: 调用方**应当显式给值**（CLI 从端到端产物的 `decode` 段判定）。
    generated: bool | None = None
    #: zip 条目时间戳是否固定（True ⇒ 同输入逐字节同包）。
    deterministic: bool = True


@dataclass(frozen=True, slots=True)
class PhiraExportReport:
    """导出结果（**自检已跑过**：包被重新打开并逐项核对）。"""

    path: Path
    entries: tuple[str, ...]
    size_bytes: int
    info: dict[str, Any]
    chart_notes: int
    chart_sha1: str
    music_sha1: str
    music_bytes: int
    illustration_placeholder: bool
    warnings: tuple[str, ...] = ()

    def summary(self) -> str:
        """一行摘要（CLI 与日志共用，不许两套口径）。"""
        placeholder = "（占位曲绘）" if self.illustration_placeholder else ""
        return (
            f"{self.path.name}：{self.size_bytes / 1024:.0f} KiB / {self.chart_notes} note / "
            f"难度 {float(self.info['difficulty']):.1f} / {self.info['level']}{placeholder}"
        )


def _require(value: object, field_name: str, hint: str) -> None:
    """必需字段的**显式**校验（宁可报错，也不让官方占位默认值静默生效）。"""
    if value is None or (isinstance(value, str) and not value.strip()):
        raise PhiraExportError(f"{field_name} 缺失：{hint}")


def safe_entry_name(name: str, *, field_name: str) -> str:
    """把包内文件名规范成**根级安全名**（包内不得有目录层级，见官方「谱面基本结构」）。

    只做三件事：取 basename、把路径分隔符与控制字符换成 `_`、去首尾空白；
    **CJK 与其他 Unicode 原样保留**（真实谱面包里就有 `＃53682.json` 这种名字，
    Phira 与我们的读取器都按 `info.yml` 定位，不靠后缀猜）。

    **全点名的拒绝**（`.` / `..` / `...`）：Windows 解压时会把结尾的点丢掉，于是包里的
    名字与实际落盘的名字不一致——那正是「`info.yml` 定位失败」这类最难查的故障。
    """
    base = Path(name.strip()).name
    cleaned = "".join("_" if (ch in "/\\" or ord(ch) < 32) else ch for ch in base).strip()
    if not cleaned or set(cleaned) == {"."}:
        raise PhiraExportError(f"{field_name} 的文件名非法：{name!r}")
    return cleaned


def placeholder_png(
    width: int = PLACEHOLDER_SIZE[0],
    height: int = PLACEHOLDER_SIZE[1],
    rgb: tuple[int, int, int] = (26, 28, 36),
) -> bytes:
    """生成一张**纯色 PNG**（只用标准库：`struct` + `zlib`）。

    为什么需要它：官方要求包里必须有曲绘（`illustration` 是必需字段，真实谱面包 100% 带图），
    而自动生成的谱面通常没有曲绘 ⇒ 与其因为缺一张图而导不出，不如写一张**名字自解释**的占位图
    （`beatmorph-placeholder.png`），并如实告警。
    """
    if width < 1 or height < 1:
        raise ValueError(f"占位图尺寸必须为正，得到 {width}x{height}")
    if len(rgb) != 3 or any(not 0 <= int(value) <= 255 for value in rgb):
        raise ValueError(f"颜色必须是三个 0-255 整数，得到 {rgb!r}")
    colour = bytes(int(value) for value in rgb)
    raw = b"".join(b"\x00" + colour * width for _ in range(height))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8bit / truecolor
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def read_wav_duration_s(path: Path) -> float | None:
    """读 WAV 时长（秒）；**不是 WAV / 读不了就返回 None**（不猜、不用近似值）。

    为什么只支持 WAV：标准库只有 `wave`。mp3/ogg 的时长要么引依赖、要么自己扫帧（VBR 下是近似值）
    ——近似值会被写进 `previewEnd` 并当成事实，所以这里宁可不给：调用方可以显式传
    `music_duration_s`（本项目的 MERT 特征元数据里就有**精确**的 `duration_s`）。
    """
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
    except (wave.Error, OSError, EOFError):
        return None
    return None if rate <= 0 else float(frames) / float(rate)


def duration_from_feature_meta(path: Path) -> float | None:
    """从 MERT 特征元数据 JSON 读 `duration_s`（本项目的特征缓存格式；读不到返回 None）。"""
    import json

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    value = payload.get("duration_s")
    return float(value) if isinstance(value, int | float) else None


def load_chart_for_export(path: Path) -> PhigrosChart:
    """读一张 RPEJSON 成谱面 IR（**只读**；合法性由导出路径统一把关）。"""
    from beatmorph.data.parsers.rpejson import parse_rpejson

    target = Path(path)
    if not target.is_file():
        raise PhiraExportError(f"谱面文件不存在：{target}")
    return parse_rpejson(target.read_bytes(), ChartSource(sniff_evidence="export"))


def is_generated_chart(chart: PhigrosChart) -> bool:
    """这张谱面是不是**生成**出来的（解码链在 `ChartSource.sniff_evidence` 里留下的证据）。

    ⚠️ 只在**内存里的 IR** 上可靠：写成 RPEJSON 再读回来时该标记会丢（它不是 RPE 的字段）⇒
    从文件导出请用 `PhiraExportSpec.generated` 显式声明。
    """
    return chart.source.sniff_evidence.startswith(GENERATED_EVIDENCE_PREFIX)


def _generated(spec: PhiraExportSpec) -> bool:
    """`generated` 显式优先；None 才回落到 IR 里的证据。"""
    return is_generated_chart(spec.chart) if spec.generated is None else bool(spec.generated)


def _resolve_credits(
    spec: PhiraExportSpec,
    warnings: list[str],
) -> tuple[str, str, str, str]:
    """曲名 / 音乐作者 / 谱师 / 画师（前两个必需；后两个可缺省但要告警）。

    ⚠️ **生成谱面不得继承模板的谱师署名**（见 `GENERATED_EVIDENCE_PREFIX`）：曲名与音乐作者
    描述的是「这首歌」，谱师描述的是「这堆音符是谁写的」——后者对生成谱面只能是生成方。
    """
    meta = spec.chart.meta
    name = spec.name or meta.name
    _require(name, "info.yml.name", "用 --name 给曲名（官方默认值 'UK' 是占位符，不是曲名）")
    composer = spec.composer or meta.composer
    _require(composer, "info.yml.composer", "用 --composer 给音乐作者")
    generated = _generated(spec)
    charter = spec.charter or ("" if generated else meta.charter)
    if not charter:
        charter = DEFAULT_CREDIT
        if generated and meta.charter:
            warnings.append(
                f"该谱面是自动生成的 ⇒ **不继承**模板里的谱师署名 {meta.charter!r}；"
                f"写 {DEFAULT_CREDIT!r}（请用 --charter 覆盖成真实署名）"
            )
        else:
            warnings.append(f"未提供谱师（charter）⇒ 写 {DEFAULT_CREDIT!r}；请用 --charter 覆盖")
    illustrator = spec.illustrator
    if not illustrator:
        illustrator = DEFAULT_CREDIT
        warnings.append(
            f"未提供画师（illustrator）⇒ 写 {DEFAULT_CREDIT!r}；请用 --illustrator 覆盖"
        )
    return name, composer, charter, illustrator


def _resolve_difficulty(spec: PhiraExportSpec, warnings: list[str]) -> tuple[float, str]:
    """定数与等级文本：定数**不替你编**（官方原话「别乱来哈」），等级文本可缺省。"""
    difficulty = spec.difficulty if spec.difficulty is not None else spec.chart.meta.difficulty
    _require(
        difficulty,
        "info.yml.difficulty",
        "用 --difficulty 给定数（官方：'别乱来哈'，本模块不替你编一个）",
    )
    assert difficulty is not None  # _require 已保证非空
    generated = _generated(spec)
    level = spec.level or ("" if generated else spec.chart.meta.level_text)
    if not level:
        level = f"UK Lv.{float(difficulty):g}"
        if generated and spec.chart.meta.level_text:
            warnings.append(
                f"该谱面是自动生成的 ⇒ **不继承**模板的等级文本 "
                f"{spec.chart.meta.level_text!r}；写 {level!r}（请用 --level 覆盖）"
            )
        else:
            warnings.append(
                f"未提供等级文本（level）⇒ 写 {level!r}（官方默认格式）；请用 --level 覆盖"
            )
    return float(difficulty), level


def _resolve_offset(spec: PhiraExportSpec) -> float:
    """`info.yml.offset`（**秒**）：显式优先，否则由 RPE 的毫秒换算。"""
    if spec.offset_s is not None:
        return float(spec.offset_s)
    return float(spec.chart.meta.offset_ms) / 1000.0  # RPE 是毫秒，info.yml 是秒


def _resolve_preview(
    spec: PhiraExportSpec,
    warnings: list[str],
) -> tuple[float, float | None]:
    """预览区间；超出音频时长时按**官方行为**截断（并告警）。"""
    preview_start = float(spec.preview_start)
    if preview_start < 0.0:
        raise PhiraExportError(f"previewStart 必须 >= 0，得到 {preview_start}")
    if spec.preview_end is None:
        if spec.music_duration_s is None:
            warnings.append(
                "音频时长未知 ⇒ previewEnd 留空（Phira 取 previewStart + 15s 并自行截断）"
            )
        return preview_start, None
    preview_end = float(spec.preview_end)
    if preview_end <= preview_start:
        raise PhiraExportError(f"previewEnd({preview_end}) 必须大于 previewStart({preview_start})")
    duration = spec.music_duration_s
    if duration is not None and preview_end > float(duration):
        warnings.append(
            f"previewEnd {preview_end:.2f}s 超出音频时长 {float(duration):.2f}s "
            "⇒ 截断到音频结尾（官方行为：超出结尾会被截断）"
        )
        preview_end = float(duration)
    return preview_start, preview_end


def build_info(spec: PhiraExportSpec) -> tuple[dict[str, Any], list[str]]:
    """按官方 ChartInfo 构造写出的字段（**只写该写的**），返回 `(info, warnings)`。

    字段名与官方 YAML 的 camelCase 一致（`aspectRatio` / `lineLength` / `previewStart` /
    `previewEnd` / `holdPartialCover`）；键序即官方文档顺序，便于人工比对。

    Raises:
        PhiraExportError: 必需信息缺失，或 preview 区间非法。
    """
    warnings: list[str] = []
    name, composer, charter, illustrator = _resolve_credits(spec, warnings)
    difficulty, level = _resolve_difficulty(spec, warnings)
    offset = _resolve_offset(spec)
    preview_start, preview_end = _resolve_preview(spec, warnings)

    info: dict[str, Any] = {
        "name": name,
        "difficulty": round(float(difficulty), 1),
        "level": level,
        "charter": charter,
        "composer": composer,
        "illustrator": illustrator,
        "chart": spec.chart_filename,
        "music": spec.music_filename,
        "illustration": spec.illustration_filename,
        "previewStart": preview_start,
    }
    if preview_end is not None:
        info["previewEnd"] = preview_end
    info["aspectRatio"] = float(spec.aspect_ratio)
    info["backgroundDim"] = float(spec.background_dim)
    info["lineLength"] = float(spec.line_length)
    info["offset"] = offset
    if spec.tip.strip():
        info["tip"] = spec.tip.strip()  # 官方：不写的话会塞一条别的 ⇒ 空则省略
    info["tags"] = list(spec.tags)
    info["intro"] = spec.intro
    info["holdPartialCover"] = bool(spec.hold_partial_cover)

    for forbidden in FORBIDDEN_INFO_FIELDS:
        if forbidden in info:  # pragma: no cover - 构造里本就不该出现
            raise PhiraExportError(f"{forbidden!r} 不得写进 {INFO_YML_NAME}（见模块 docstring）")
    if not spec.music_filename:
        raise PhiraExportError("music_filename 未解析（导出路径必须先定包内文件名）")
    if not spec.illustration_filename:
        raise PhiraExportError("illustration_filename 未解析（同上）")

    # 自检①：写出的 YAML 必须能被**读侧同一套模型**反解（ChartInfo 在 package.py 里）。
    parse_info_yaml(yaml.safe_dump(info, allow_unicode=True, sort_keys=False).encode("utf-8"))
    return info, warnings


def _resolve_entry_names(spec: PhiraExportSpec) -> PhiraExportSpec:
    """定下包内三个文件名（**根级、无目录、互不重名**）。"""
    chart_name = safe_entry_name(spec.chart_filename, field_name="chart")
    music_name = safe_entry_name(spec.music_filename or Path(spec.music).name, field_name="music")
    if spec.illustration is not None:
        illustration_name = safe_entry_name(
            spec.illustration_filename or Path(spec.illustration).name, field_name="illustration"
        )
    else:
        illustration_name = safe_entry_name(
            spec.illustration_filename or PLACEHOLDER_ILLUSTRATION, field_name="illustration"
        )
    names = (chart_name, music_name, illustration_name)
    if len(set(names)) != len(names):
        raise PhiraExportError(f"包内文件名冲突（必须互不相同，且都在根级）：{names}")
    return replace(
        spec,
        chart_filename=chart_name,
        music_filename=music_name,
        illustration_filename=illustration_name,
    )


@dataclass(frozen=True, slots=True)
class _Payloads:
    """写进 zip 的四份字节 + 自检要用的摘要（**写侧唯一的字节来源**）。"""

    entries: tuple[tuple[str, bytes], ...]
    info: dict[str, Any]
    chart_bytes: bytes
    music_bytes: bytes
    chart_sha1: str
    music_sha1: str
    placeholder: bool


def _prepare_payloads(
    spec: PhiraExportSpec,
    *,
    music: Path,
    info: dict[str, Any],
) -> _Payloads:
    """准备四份字节：`info.yml` / 谱面（红线 6 门禁）/ 音频 / 曲绘（缺省则占位图）。"""
    report = check_chart(spec.chart)
    if not report.is_legal:
        detail = "；".join(f"[{item.kind}] {item.detail}" for item in report.violations)
        raise PhiraExportError(
            f"谱面有 {len(report.violations)} 项违规，拒绝打包（红线 6）：{detail}"
        )
    chart_bytes = write_rpejson(spec.chart, report=report)
    music_bytes = music.read_bytes()
    placeholder = spec.illustration is None
    if placeholder:
        illustration_bytes = placeholder_png()
        logger.warning(
            "未提供曲绘 ⇒ 写入占位图 %s（官方要求 illustration 必需；请用 --illustration 换真图）",
            spec.illustration_filename,
        )
    else:
        assert spec.illustration is not None  # 由 placeholder 的定义保证
        illustration_bytes = Path(spec.illustration).read_bytes()
    info_text = yaml.safe_dump(info, allow_unicode=True, sort_keys=False).encode("utf-8")
    return _Payloads(
        info=info,
        entries=(
            (INFO_YML_NAME, info_text),
            (spec.chart_filename, chart_bytes),
            (str(spec.music_filename), music_bytes),
            (str(spec.illustration_filename), illustration_bytes),
        ),
        chart_bytes=chart_bytes,
        music_bytes=music_bytes,
        chart_sha1=hashlib.sha1(chart_bytes).hexdigest(),
        music_sha1=hashlib.sha1(music_bytes).hexdigest(),
        placeholder=placeholder,
    )


def _write_zip(
    target: Path,
    entries: tuple[tuple[str, bytes], ...],
    *,
    deterministic: bool,
) -> None:
    """写 zip：**只允许根级条目**（官方「解压后根级直接是文件」）。"""
    for name, _ in entries:
        if "/" in name or "\\" in name:  # pragma: no cover - safe_entry_name 已挡
            raise PhiraExportError(f"包内条目 {name!r} 含目录层级（官方要求根级直挂）")
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in entries:
            if deterministic:
                archive.writestr(zipfile.ZipInfo(name, date_time=ZIP_DATE_TIME), payload)
            else:
                archive.writestr(name, payload)


def _self_check(target: Path, spec: PhiraExportSpec, payloads: _Payloads) -> None:
    """用**读侧**（`ChartPackage`）重新打开并逐项核对：写法与读法必须一致。

    自检是硬要求而不是礼貌：导出与导入读的是同一份 `ChartInfo`，一旦两边的口径漂开
    （字段名、根级约束、编码），只有「自己读自己写的包」能立刻发现。
    """
    package = ChartPackage.open(target)
    if str(package.info.chart) != spec.chart_filename:
        raise PhiraExportError("自检失败：info.yml.chart 与写入的谱面条目名不一致")
    if package.chart_bytes() != payloads.chart_bytes:
        raise PhiraExportError("自检失败：包内谱面字节与写出的不一致")
    if package.music_bytes() != payloads.music_bytes:
        raise PhiraExportError("自检失败：包内音频字节与源文件不一致")
    parsed = parse_info_yaml(dict(payloads.entries)[INFO_YML_NAME])
    if parsed.name != payloads.info["name"] or parsed.difficulty != payloads.info["difficulty"]:
        raise PhiraExportError("自检失败：info.yml 反解后与写出值不一致")


def export_phira_package(
    spec: PhiraExportSpec,
    out_path: Path | str,
) -> PhiraExportReport:
    """写出 `.pez` 并**自检**（重新打开核对），返回落盘报告。

    Raises:
        PhiraExportError: 信息不全 / 文件缺失 / 谱面不合法（红线 6）/ 包内命名冲突。
    """
    music = Path(spec.music)
    if not music.is_file():
        raise PhiraExportError(f"音频文件不存在：{music}")
    if spec.illustration is not None and not Path(spec.illustration).is_file():
        raise PhiraExportError(f"曲绘文件不存在：{spec.illustration}")
    duration = spec.music_duration_s
    if duration is None:
        duration = read_wav_duration_s(music)
    resolved = _resolve_entry_names(replace(spec, music_duration_s=duration))
    info, warnings = build_info(resolved)
    payloads = _prepare_payloads(resolved, music=music, info=info)

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _write_zip(target, payloads.entries, deterministic=resolved.deterministic)
    _self_check(target, resolved, payloads)

    size = target.stat().st_size
    logger.info("已导出 %s（%d 字节，%d 个条目）", target, size, len(payloads.entries))
    for warning in warnings:
        logger.warning("导出告警：%s", warning)
    return PhiraExportReport(
        path=target,
        entries=tuple(name for name, _ in payloads.entries),
        size_bytes=size,
        info=info,
        chart_notes=len(resolved.chart.notes),
        chart_sha1=payloads.chart_sha1,
        music_sha1=payloads.music_sha1,
        music_bytes=len(payloads.music_bytes),
        illustration_placeholder=payloads.placeholder,
        warnings=tuple(warnings),
    )


__all__ = [
    "CHART_FILENAME",
    "DEFAULT_ASPECT_RATIO",
    "DEFAULT_BACKGROUND_DIM",
    "DEFAULT_CREDIT",
    "DEFAULT_LINE_LENGTH",
    "FORBIDDEN_INFO_FIELDS",
    "GENERATED_EVIDENCE_PREFIX",
    "PLACEHOLDER_ILLUSTRATION",
    "PLACEHOLDER_SIZE",
    "ZIP_DATE_TIME",
    "PhiraExportError",
    "PhiraExportReport",
    "PhiraExportSpec",
    "build_info",
    "duration_from_feature_meta",
    "export_phira_package",
    "is_generated_chart",
    "load_chart_for_export",
    "placeholder_png",
    "read_wav_duration_s",
    "safe_entry_name",
]
