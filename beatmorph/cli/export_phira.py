"""导出入口 `beatmorph-export-phira`：谱面 + 音频 (+ 曲绘) -> Phira 可导入的 `.pez`。

依据 **Phira 官方文档**（A 级来源）：<https://teamflos.github.io/phira-docs/chart-standard/chartinfo.html>
与同目录的「谱面基本结构」。包结构与字段口径全部在 `beatmorph/data/phira/export.py` 里，
本模块只负责命令行契约与端点解析。

命令行契约：

    beatmorph-export-phira --chart chart.json --music song.mp3 \\
        --name 曲名 --composer 音乐作者 [--difficulty 15.3] [--level 'AT Lv.15'] \\
        [--illustration cover.png] [--out 曲名.pez] [--charter ...] [--illustrator ...]

**一键从端到端产物导出**（推荐；曲名/作者/定数/音频/时长都从产物元数据里取）：

    beatmorph-export-phira --from-e2e-artifact tests/e2e-val/outputs/20260929-214209-step0

退出码：0 成功｜2 参数错误｜3 输入不可用（缺文件 / 缺必需元数据 / 谱面不合法 ⇒ 红线 6 拒绝导出）。
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from beatmorph.core.logging import get_logger, setup_logging
from beatmorph.data.phira.export import (
    PhiraExportError,
    PhiraExportSpec,
    duration_from_feature_meta,
    export_phira_package,
    load_chart_for_export,
)

__all__ = ["EXIT_CODES", "build_parser", "main"]

logger = get_logger("cli.export_phira")

EXIT_OK = 0
EXIT_ARGS = 2
EXIT_INPUT = 3

#: 退出码语义（文档与测试共用一份）
EXIT_CODES: dict[int, str] = {
    EXIT_OK: "成功",
    EXIT_ARGS: "参数错误",
    EXIT_INPUT: "输入不可用（缺文件 / 缺必需元数据 / 谱面不合法）",
}


def build_parser() -> argparse.ArgumentParser:
    """构造参数解析器（`--from-e2e-artifact` 只补默认值，显式参数一律优先）。"""
    parser = argparse.ArgumentParser(prog="beatmorph-export-phira", description="导出 Phira 谱面包")
    parser.add_argument(
        "--from-e2e-artifact",
        default=None,
        metavar="DIR",
        help="从端到端产物目录（含 chart.json + meta.json）补齐默认值",
    )
    parser.add_argument(
        "--chart", default=None, help="RPEJSON 谱面文件（默认取产物目录里的 chart.json）"
    )
    parser.add_argument(
        "--music", default=None, help="音频文件（默认取产物 meta.json 的 audio.path）"
    )
    parser.add_argument(
        "--illustration", default=None, help="曲绘（缺省写占位图，官方要求该字段必需）"
    )
    parser.add_argument(
        "--out", default=None, help="输出 .pez 路径（默认与谱面同目录、同名换后缀）"
    )
    parser.add_argument("--name", default=None, help="曲名（**必需**；产物里可从 song_key 得到）")
    parser.add_argument("--composer", default=None, help="音乐作者（**必需**）")
    parser.add_argument("--charter", default=None, help="谱师（缺省写 BeatMorph 并告警）")
    parser.add_argument("--illustrator", default=None, help="曲绘画师（缺省写 BeatMorph 并告警）")
    parser.add_argument(
        "--difficulty", type=float, default=None, help="定数（必填或由谱面/产物给）"
    )
    parser.add_argument(
        "--level", default=None, help='等级文本（如 "AT Lv.15"；缺省 UK Lv.<定数>）'
    )
    parser.add_argument("--tags", default=None, help="标签，逗号分隔")
    parser.add_argument("--intro", default=None, help="简介（默认由产物目录补一行溯源事实）")
    parser.add_argument("--tip", default="", help="提示（留空则不写该字段：官方会塞一条自己的）")
    parser.add_argument("--preview-start", type=float, default=0.0, help="预览开始秒数（默认 0）")
    parser.add_argument(
        "--preview-end", type=float, default=None, help="预览结束秒数（超音频则截断）"
    )
    parser.add_argument(
        "--offset", type=float, default=None, help="谱面延迟（秒；默认取谱面 META 的毫秒值）"
    )
    parser.add_argument(
        "--aspect-ratio", type=float, default=16.0 / 9.0, help="显示纵横比（官方默认 16:9）"
    )
    parser.add_argument(
        "--background-dim", type=float, default=0.6, help="背景暗化（官方默认 0.6）"
    )
    parser.add_argument("--line-length", type=float, default=6.0, help="判定线长度（官方默认 6.0）")
    parser.add_argument(
        "--hold-partial-cover", action="store_true", help="Hold 遮罩放尾部（官方默认头部）"
    )
    parser.add_argument(
        "--music-duration", type=float, default=None, help="音频时长（秒；只有 WAV 能自动读）"
    )
    parser.add_argument(
        "--feature-meta",
        default=None,
        metavar="JSON",
        help="MERT 特征元数据 JSON（用于取精确的音频时长；产物里自动带上）",
    )
    parser.add_argument(
        "--generated",
        dest="generated",
        action="store_true",
        help="这张谱面是自动生成的（不继承模板的谱师/等级文本；产物目录会自动判定）",
    )
    parser.add_argument(
        "--not-generated",
        dest="generated",
        action="store_false",
        help="这张谱面是真人/既有谱面（照常继承谱面 META 里的谱师与等级）",
    )
    parser.set_defaults(generated=None)
    parser.add_argument(
        "--nondeterministic",
        action="store_true",
        help="zip 条目写入当前时间戳（默认固定时间戳 ⇒ 同输入逐字节同包）",
    )
    parser.add_argument("--json", action="store_true", help="把结果以 JSON 打到 stdout")
    return parser


def _artifact_defaults(directory: Path) -> dict[str, Any]:
    """从端到端产物目录读出可用的默认值（读不到就**不说**，由必需项校验去报错）。"""
    meta_path = directory / "meta.json"
    if not meta_path.is_file():
        raise PhiraExportError(f"产物目录里没有 meta.json：{meta_path}")
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PhiraExportError(f"产物 meta.json 必须是对象：{meta_path}")
    if payload.get("chart_written") is False or payload.get("aborted"):
        raise PhiraExportError(
            "该产物**不含可用的 chart.json**（生成被中止或不合法）："
            f"aborted={payload.get('aborted')!r}，chart_written={payload.get('chart_written')!r}"
        )
    defaults: dict[str, Any] = {"chart": directory / "chart.json"}
    audio = payload.get("audio")
    if isinstance(audio, dict) and isinstance(audio.get("path"), str):
        defaults["music"] = Path(str(audio["path"]))
    feature = payload.get("feature")
    if isinstance(feature, dict) and isinstance(feature.get("meta_path"), str):
        defaults["feature_meta"] = Path(str(feature["meta_path"]))
    template = payload.get("template")
    if isinstance(template, dict) and isinstance(template.get("song_key"), str):
        song_key = str(template["song_key"])
        name, _, composer = song_key.partition("|")
        defaults["name"] = name.strip()
        defaults["composer"] = composer.strip()
    if isinstance(template, dict) and isinstance(template.get("source"), str):
        # 溯源一行（**是事实**：谁生成的、条件取自哪），可用 --intro 覆盖。
        chart_id = template.get("chart_id")
        detail = f" / chart_id={chart_id}" if chart_id is not None else ""
        defaults["intro"] = f"由 BeatMorph 自动生成（条件来源：{template['source']}{detail}）"
    difficulty = payload.get("difficulty")
    if isinstance(difficulty, int | float):
        defaults["difficulty"] = float(difficulty)
    # 有 `decode` 段 ⇒ 这张谱面是解码出来的（**不能**继承模板的谱师署名，见 export.py）。
    if isinstance(payload.get("decode"), dict):
        defaults["generated"] = True
    return defaults


def _apply_defaults(args: argparse.Namespace, defaults: dict[str, Any]) -> None:
    """把产物默认值填进**用户没显式给**的项（显式参数永远优先）。"""
    for key, value in defaults.items():
        attribute = key.replace("-", "_")
        current = getattr(args, attribute, None)
        if current is None:
            setattr(args, attribute, value)


def _resolve_duration(args: argparse.Namespace) -> float | None:
    """音频时长：显式值 > 特征元数据（精确）> WAV 自动读（由导出器兜底）。"""
    if args.music_duration is not None:
        return float(args.music_duration)
    if args.feature_meta is not None:
        duration = duration_from_feature_meta(Path(args.feature_meta))
        if duration is not None:
            return duration
        logger.warning("特征元数据里没有可用的 duration_s：%s", args.feature_meta)
    return None


def main(argv: Sequence[str] | None = None) -> int:
    """命令入口；返回退出码（见 `EXIT_CODES`）。"""
    setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.from_e2e_artifact is not None:
            _apply_defaults(args, _artifact_defaults(Path(args.from_e2e_artifact)))
        if args.chart is None or args.music is None:
            raise PhiraExportError("必须给出 --chart 与 --music（或用 --from-e2e-artifact 补齐）")
        chart_path = Path(args.chart)
        music_path = Path(args.music)
        out = Path(args.out) if args.out is not None else chart_path.with_suffix(".pez")
        spec = PhiraExportSpec(
            chart=load_chart_for_export(chart_path),
            music=music_path,
            illustration=None if args.illustration is None else Path(args.illustration),
            name=args.name or "",
            composer=args.composer or "",
            charter=args.charter or "",
            illustrator=args.illustrator or "",
            difficulty=args.difficulty,
            level=args.level or "",
            tags=tuple(part.strip() for part in (args.tags or "").split(",") if part.strip()),
            intro=args.intro or "",
            tip=args.tip or "",
            preview_start=float(args.preview_start),
            preview_end=args.preview_end,
            offset_s=args.offset,
            aspect_ratio=float(args.aspect_ratio),
            background_dim=float(args.background_dim),
            line_length=float(args.line_length),
            hold_partial_cover=bool(args.hold_partial_cover),
            music_duration_s=_resolve_duration(args),
            generated=args.generated,
            deterministic=not args.nondeterministic,
        )
        report = export_phira_package(spec, out)
    except PhiraExportError as error:
        logger.error("导出被拒绝：%s", error)
        return EXIT_INPUT
    except OSError as error:
        logger.error("导出失败（文件系统）：%s", error)
        return EXIT_INPUT
    if args.json:
        print(
            json.dumps(
                {
                    "path": str(report.path),
                    "entries": list(report.entries),
                    "size_bytes": report.size_bytes,
                    "info": report.info,
                    "chart_notes": report.chart_notes,
                    "chart_sha1": report.chart_sha1,
                    "music_sha1": report.music_sha1,
                    "illustration_placeholder": report.illustration_placeholder,
                    "warnings": list(report.warnings),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(f"已导出 {report.path.resolve()}")
        print(f"  {report.summary()}")
        for warning in report.warnings:
            print(f"  ⚠️ {warning}")
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
