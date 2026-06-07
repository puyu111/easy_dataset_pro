"""统一日志系统 —— 集中配置 root logger。

用法:
    from graphrag.src.logging_setup import setup_logging
    setup_logging(settings.logging)
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(config) -> None:
    """配置 root logger：控制台 + 轮转文件输出。

    Args:
        config: LoggingConfig 实例，包含 level / directory / filename
               / format_console / format_file / max_bytes / backup_count
               / console_enabled / file_enabled 字段。
    """
    root = logging.getLogger()
    # 清理已有 handlers，防止重复添加
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(getattr(logging, config.level.upper(), logging.INFO))

    formatter_console = logging.Formatter(config.format_console)
    formatter_file = logging.Formatter(config.format_file)

    # 控制台 handler
    if config.console_enabled:
        console = logging.StreamHandler()
        console.setLevel(getattr(logging, config.level.upper(), logging.INFO))
        console.setFormatter(formatter_console)
        root.addHandler(console)

    # 文件 handler (轮转)
    log_path = None
    if config.file_enabled:
        log_dir = Path(config.directory)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / config.filename
        fh = RotatingFileHandler(
            str(log_path),
            maxBytes=config.max_bytes,
            backupCount=config.backup_count,
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)  # 文件始终记录 DEBUG 级别
        fh.setFormatter(formatter_file)
        root.addHandler(fh)

    # 抑制第三方库的 verbose 日志
    for noisy in ("httpx", "urllib3", "PIL", "rapidocr", "docling"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root.info("日志系统初始化完成 (level=%s, file=%s)", config.level, str(log_path) if log_path else "disabled")
