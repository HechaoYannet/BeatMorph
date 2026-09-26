"""格式内容嗅探（plan 02 §3.3 / M3）——⚠️ 陷阱 1：后缀不可信。

**只按内容判定** `RPE | PEC | OFFICIAL | PBC | UNKNOWN`。签名**不接受**文件名/后缀参数
（plan 02 §3.3 的签名级约束）：实测 `chart/7039` 的 `24432296.json` 内容是 PEC 文本，
而 `.pec` 里也可能是 JSON——Phira 官方文档原文即「忽略谱面文件的后缀名」。

判定顺序与依据（docs/TRAINING.md §格式嗅探启发式 + phira-dataset-survey.md §5.2）：

1. 含 `eventLayers` → `RPE`；若前缀里恰好没有任何 `eventLayers`（格式事实：所有层级
   都为空时该字段不出现），再退一步看**根键组合** `BPMList` + `judgeLineList`
   （RPE 独有；官谱根键是 `formatVersion`，不含 `BPMList`）；
2. 含 `notesAbove` / `notesBelow` / `formatVersion` → `OFFICIAL`；
3. 文本且匹配 PEC 行结构（`bp` / `cp` / `cm` / `ca` / `cv` / `cd` / `n1..n4`）→ `PEC`；
4. 其余 → `UNKNOWN`（**记账，不静默通过**）。

**PBC 不产出判定**：Phira 文档只写「文档待完善」，本仓调研明确「PBC 结构完全未查证，
建议先直接拒收并记账」（phira-dataset-survey.md §9-Q9/§9.2-583）。按其结构猜一个签名
等于凭空造事实，因此本实现把 PBC 归入 `UNKNOWN` 并由上层记账。

输入可能是**压缩流的部分解压前缀**（`RANGE_PREFIX_BYTES`），因此末行可能被截断：
本模块在判定行结构时丢弃未以换行结尾的末行（该行不完整，不可作为证据）。
"""

from __future__ import annotations

# 契约层是 ChartFormat 的唯一出处（红线 7：同一常量不得两处定义），此处只再导出。
from beatmorph.core.contracts import ChartFormat

__all__ = [
    "OFFICIAL_MARKERS",
    "PEC_COMMANDS",
    "RPE_MARKERS",
    "RPE_ROOT_KEY_MARKERS",
    "ChartFormat",
    "sniff_format",
    "sniff_format_with_evidence",
]

#: RPE 特征字段（根级 judgeLineList 元素内的键名）。带引号形式优先，裸词作为截断容错。
RPE_MARKERS: tuple[bytes, ...] = (b'"eventLayers"',)

#: RPE 根键组合：两者同时出现即 RPE（官谱根键是 `formatVersion`，不含 `BPMList`）。
#: 用于「前缀里没有 eventLayers」的兜底——实测「所有层级都为空时该字段不出现」。
RPE_ROOT_KEY_MARKERS: tuple[bytes, ...] = (b'"BPMList"', b'"judgeLineList"')

#: 官谱 JSON 特征字段（phi/root 与 phi/judgeLine 的字段名）。
OFFICIAL_MARKERS: tuple[bytes, ...] = (b'"notesAbove"', b'"notesBelow"', b'"formatVersion"')

#: PEC 命令行首 token（PhiEditer 旧格式；pgrfm wiki PEC 条目 + 调研 §4.4 实测样例）。
#:
#: ⚠️ **2026-09-27 扩表（实测证据）**：对全库抓取中 74 张 sniff 拒收样本逐个复核后，
#: 发现旧谱面（低 chart id）里的 PEC 远不止调研样例里的那 10 个命令——未识别行首的
#: 频次统计是 `&` 80198 行 / `cf` 49280 行 / `cr` 41644 行，三者都属于
#: 「符号命令 + 数值参数」的同一族语法。原表把它们判成 UNKNOWN，
#: 后果不是丢数据（v1 主路径本来就拒收 PEC），而是**格式占比统计被污染**
#: （Q-1「全库精确格式占比」会把这些 PEC 记成 unknown）。
#: `#` 行按注释丢弃（实测 `# 1.00` 形态），保持原行为。
PEC_COMMANDS: frozenset[str] = frozenset(
    {"bp", "cp", "cm", "cd", "ca", "cv", "cf", "cr", "&", "n1", "n2", "n3", "n4"},
)

#: PEC 至少必须出现的命令：一个 BPM 行 + 至少一个 note 行（否则不是谱面）。
_PEC_REQUIRED: frozenset[str] = frozenset({"bp"})
_PEC_NOTE_COMMANDS: frozenset[str] = frozenset({"n1", "n2", "n3", "n4"})


def _find_marker(data: bytes, markers: tuple[bytes, ...]) -> bytes | None:
    """在字节流中查找特征字段；先试带引号写法，再试裸词（容忍前缀截断）。"""
    for marker in markers:
        if marker in data:
            return marker
    for marker in markers:
        bare = marker.replace(b'"', b"")
        if bare in data:
            return bare
    return None


def _is_number(token: str) -> bool:
    """token 是否为十进制数值（PEC 行参数全为数值）。"""
    try:
        float(token)
    except ValueError:
        return False
    return True


def _pec_lines(text: str) -> list[str]:
    """切出 PEC 的有效行（丢弃空行/注释行，以及可能被截断的末行）。"""
    lines = [line.strip() for line in text.lstrip("\ufeff").splitlines()]
    if text and not text.endswith(("\n", "\r")):
        # 末行可能被截断（部分解压前缀），不作为证据。
        lines = lines[:-1]
    return [line for line in lines if line and not line.startswith(("//", "#"))]


def _pec_commands(lines: list[str]) -> tuple[set[str], bool] | None:
    """逐行校验 PEC 行语法，返回 `(命中的命令集, 是否含首行版本号)`；不合法返回 None。"""
    found: set[str] = set()
    version_line = False
    for index, line in enumerate(lines):
        parts = line.split()
        head = parts[0]
        if index == 0 and len(parts) == 1 and _is_number(head):
            # 实测 PEC 首行为版本号（id 7039 首行 "175"），非命令。
            version_line = True
            continue
        lowered = head.lower()
        if lowered not in PEC_COMMANDS:
            return None
        if len(parts) < 2 or not all(_is_number(token) for token in parts[1:]):
            return None
        found.add(lowered)
    return found, version_line


def _pec_evidence(data: bytes) -> str | None:
    """PEC 行结构判定：全部有效行都必须是 `命令 + 数值参数`，且含 bp 与 n1..n4 之一。

    返回证据字符串；不匹配返回 None。**保守判定**：任何一行不符合语法即判否
    （宁可归 UNKNOWN 记账，也不把未知文本当 PEC 送进解析器）。
    """
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None

    lines = _pec_lines(text)
    matched = _pec_commands(lines) if lines else None
    if matched is None:
        return None
    found, version_line = matched
    if not found >= _PEC_REQUIRED or not found & _PEC_NOTE_COMMANDS:
        return None
    head_note = "首行版本号 + " if version_line else ""
    commands = ",".join(sorted(found))
    return f"文本行结构匹配 PEC（{head_note}命令 {commands}）"


def sniff_format_with_evidence(data: bytes) -> tuple[ChartFormat, str]:
    """按内容判定格式，并返回可审计的判定依据（写入 `ChartSource.sniff_evidence`）。"""
    if not data:
        return ChartFormat.UNKNOWN, "空内容"

    marker = _find_marker(data, RPE_MARKERS)
    if marker is not None:
        return ChartFormat.RPE, f'命中关键字段 "{marker.decode("ascii", "replace")}"'

    if all(marker in data for marker in RPE_ROOT_KEY_MARKERS):
        keys = " + ".join(marker.decode("ascii") for marker in RPE_ROOT_KEY_MARKERS)
        return ChartFormat.RPE, f"命中 RPE 根键组合 {keys}（前缀内无 eventLayers）"

    marker = _find_marker(data, OFFICIAL_MARKERS)
    if marker is not None:
        return ChartFormat.OFFICIAL, f'命中关键字段 "{marker.decode("ascii", "replace")}"'

    evidence = _pec_evidence(data)
    if evidence is not None:
        return ChartFormat.PEC, evidence

    return (
        ChartFormat.UNKNOWN,
        "未命中任何已知格式特征（PBC 结构未查证 → 一律归 UNKNOWN 并记账，见调研 §9-Q9）",
    )


def sniff_format(data: bytes) -> ChartFormat:
    """**只按内容**判定谱面格式。

    Args:
        data: 谱面文件字节（可为压缩条目的部分解压前缀）。

    Returns:
        内容对应的 :class:`ChartFormat`；无法判定时为 `ChartFormat.UNKNOWN`。

    Note:
        本函数**没有**文件名/后缀/路径参数，这是 plan 02 §3.3 的签名级约束：
        后缀是三类静默陷阱之一（`.json` 里可以是 PEC 文本）。
    """
    return sniff_format_with_evidence(data)[0]
