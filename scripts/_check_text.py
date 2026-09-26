import subprocess, sys
from pathlib import Path
files = subprocess.run(["git", "diff", "--cached", "--name-only"], capture_output=True, text=True).stdout.split()
files += subprocess.run(["git", "ls-files", "--others", "--exclude-standard"], capture_output=True, text=True).stdout.split()
bad = []
for name in files:
    p = Path(name)
    if not p.is_file():
        continue
    data = p.read_bytes()
    if b"\r\n" in data:
        bad.append(f"{name}: CRLF 行尾")
    text = data.decode("utf-8", errors="replace")
    if text and not text.endswith("\n"):
        bad.append(f"{name}: 缺文件末尾换行")
    for i, line in enumerate(text.splitlines(), 1):
        if line != line.rstrip():
            bad.append(f"{name}:{i}: 行尾空白")
print("checked", len(files), "files")
print("\n".join(bad) if bad else "OK: 无 CRLF / 无行尾空白 / 末尾换行齐全")