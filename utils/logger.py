"""
日志工具模块

作用：
1. 提供简单控制台日志输出
2. 提供同时写入文件的 logger
3. 便于 main.py / experiments 中统一打印实验过程
"""

import logging
import os
from typing import Optional


def get_logger(
    name: str = "stock_graph_factor",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    创建并返回 logger

    参数
    ----
    name : str
        logger 名称
    log_file : str or None
        若提供，则同时输出到文件
    level : int
        日志级别，如 logging.INFO

    返回
    ----
    logger : logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台输出
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # 文件输出
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True) if os.path.dirname(log_file) else None
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_section(logger: logging.Logger, title: str) -> None:
    """
    打印分段标题

    参数
    ----
    logger : logging.Logger
    title : str
        标题内容
    """
    logger.info("=" * 60)
    logger.info(title)
    logger.info("=" * 60)