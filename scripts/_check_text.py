"""提交前文本卫生检查：CRLF / 行尾空白 / 文件末尾换行。

用法（手工或 CI）::

    python scripts/_check_text.py

检查范围 = 「已暂存的文件 + 未跟踪的文件」；发现问题时以非 0 退出（fail-closed）。
本脚本**不被 pre-commit 引用**（那些检查由 pre-commit-hooks 负责），它是仓库自检的兜底：
make lint 会扫到它，所以它自己也必须过 lint。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

#: 取「待检查文件」的命令（暂存 + 未跟踪）。
_FILE_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("git", "diff", "--cached", "--name-only"),
    ("git", "ls-files", "--others", "--exclude-standard"),
)


def changed_files() -> list[str]:
    """当前需要检查的文件名列表（暂存 + 未跟踪）。"""
    names: list[str] = []
    for command in _FILE_COMMANDS:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        names.extend(completed.stdout.split())
    return names


def check_file(name: str) -> list[str]:
    """单个文件的文本卫生问题（空列表 = 干净）。"""
    path = Path(name)
    if not path.is_file():
        return []
    problems: list[str] = []
    data = path.read_bytes()
    if b"\r\n" in data:
        problems.append(f"{name}: CRLF 行尾")
    text = data.decode("utf-8", errors="replace")
    if text and not text.endswith("\n"):
        problems.append(f"{name}: 缺文件末尾换行")
    for index, line in enumerate(text.splitlines(), 1):
        if line != line.rstrip():
            problems.append(f"{name}:{index}: 行尾空白")
    return problems


def main() -> int:
    """返回 0 = 干净，1 = 有问题。"""
    files = changed_files()
    problems: list[str] = []
    for name in files:
        problems.extend(check_file(name))
    print(f"checked {len(files)} files")
    print("\n".join(problems) if problems else "OK: 无 CRLF / 无行尾空白 / 末尾换行齐全")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
