#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Централизованное логирование в <date_timestamp>.log
"""

import logging
from datetime import datetime
from pathlib import Path

_LOG_FILE: Path | None = None


def setup_logging(log_dir: Path | None = None) -> Path:
    """
    Настраивает логирование в файл <YYYYMMDD_HHMMSS>.log.
    Возвращает путь к созданному лог-файлу.
    """
    global _LOG_FILE
    if _LOG_FILE is not None:
        return _LOG_FILE

    if log_dir is None:
        try:
            from config_loader import get_paths
            log_dir = get_paths().get("logs_dir", Path(__file__).resolve().parent / "logs")
        except Exception:
            log_dir = Path(__file__).resolve().parent / "logs"
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _LOG_FILE = log_dir / f"{ts}.log"

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger("pipeline")
    root.setLevel(logging.INFO)
    root.handlers.clear()

    fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt, datefmt))
    root.addHandler(ch)

    root.info("Логирование в %s", _LOG_FILE)
    return _LOG_FILE


def get_logger(name: str = "") -> logging.Logger:
    """Возвращает логгер. Вызовите setup_logging() при старте приложения."""
    return logging.getLogger("pipeline" if not name else f"pipeline.{name}")
