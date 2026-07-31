"""命令行入口：端到端生成谱面。

使用示例（待实现后启用）::

    beatmorph-generate --audio song.mp3 --difficulty 8 --ref reference.osu -o out.osu

奠基文档 §2 用户输入层：① 音频文件 ② 难度等级(1-15) ③ 参考谱面(可选)。
"""

from __future__ import annotations


def main() -> None:
    """生成谱面 CLI 主入口（待实现）。"""
    raise NotImplementedError("CLI 尚未实现，见 docs/plans/09-infra-cli-api.md")


if __name__ == "__main__":  # pragma: no cover
    main()
