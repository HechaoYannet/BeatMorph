"""统一日志配置。

全项目通过 ``from beatmorph.core.logging import get_logger`` 获取 logger，
保证日志格式、级别、W&B 集成一致。禁止直接使用 ``logging.getLogger`` 或 ``print``
做诊断输出（生产路径）。
"""

from __future__ import annotations

import logging
from typing import Any

_CONFIGURED = False


def setup_logging(level: str = "INFO", json_output: bool = False) -> None:
    """配置根日志。幂等；可在 Hydra 主函数或 CLI 入口调用一次。

    Args:
        level: 日志级别字符串。
        json_output: 若 True，输出结构化 JSON 日志（便于容器化/ELK 采集）。
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler()
    if json_output:
        handler.setFormatter(_JsonFormatter())
    else:
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
    root = logging.getLogger("beatmorph")
    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


class _JsonFormatter(logging.Formatter):
    """极简 JSON 行格式器（生产环境）。"""

    def format(self, record: logging.LogRecord) -> str:
        import json

        payload: dict[str, Any] = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str) -> logging.Logger:
    """获取以 ``beatmorph.`` 为前缀的 logger。"""
    if not name.startswith("beatmorph"):
        name = f"beatmorph.{name}"
    return logging.getLogger(name)
