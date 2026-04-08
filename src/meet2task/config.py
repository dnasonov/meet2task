#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Конфигурация: config.yaml в корне клона репозитория, секреты в .env.
Переопределение корня: переменная окружения MEET2TASK_ROOT.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None


# Корень клона по расположению пакета (src/meet2task/config.py -> parents[2])
_CANDIDATE_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    """Сначала .env рядом с пакетом; затем при необходимости — из MEET2TASK_ROOT."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(_CANDIDATE_ROOT / ".env", override=False)
    env_root = os.environ.get("MEET2TASK_ROOT")
    if env_root:
        load_dotenv(Path(env_root).resolve() / ".env", override=False)


_load_dotenv()


def get_project_root() -> Path:
    """
    Корень проекта (каталог с config.yaml, prompt/, .env).

    Задаётся MEET2TASK_ROOT (в т.ч. из .env) или каталог клона по пути к пакету.
    """
    env = os.environ.get("MEET2TASK_ROOT")
    if env:
        return Path(env).resolve()
    return _CANDIDATE_ROOT


_PROJECT_ROOT = get_project_root()
CONFIG_PATH = _PROJECT_ROOT / "config.yaml"


def load_config() -> dict[str, Any]:
    """
    Загружает config.yaml.
    Секреты: .env или окружение — GROQ_API_KEY, TELEGRAM_BOT_TOKEN.
    """
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Файл конфигурации не найден: {CONFIG_PATH}\n"
            "Создайте config.yaml на основе config.example.yaml"
        )

    if yaml is None:
        raise ImportError("Установите PyYAML: pip install pyyaml")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}

    if "groq" not in cfg:
        cfg["groq"] = {}
    if os.environ.get("GROQ_API_KEY"):
        cfg["groq"]["api_key"] = os.environ["GROQ_API_KEY"]
    if "telegram" not in cfg:
        cfg["telegram"] = {}
    if os.environ.get("TELEGRAM_BOT_TOKEN"):
        cfg["telegram"]["bot_token"] = os.environ["TELEGRAM_BOT_TOKEN"]

    return cfg


def get_groq_max_file_mb() -> int:
    cfg = load_config()
    return int((cfg.get("groq") or {}).get("max_file_size_mb", 25))


def get_groq_http_timeout():
    import httpx

    cfg = load_config()
    g = cfg.get("groq") or {}
    total = float(g.get("http_timeout_seconds", 600))
    connect = float(g.get("http_connect_timeout_seconds", 60))
    return httpx.Timeout(total, connect=connect)


def get_groq_api_key() -> str:
    cfg = load_config()
    groq = cfg.get("groq") or {}
    key = (groq.get("api_key") or groq.get("GROQ_API_KEY") or "").strip()
    if not key:
        raise ValueError(
            "GROQ_API_KEY не задан. Добавьте его в .env, в groq.api_key в config.yaml "
            "или в переменную окружения GROQ_API_KEY"
        )
    return key


def get_telegram_yandex_url_enabled() -> bool:
    cfg = load_config()
    return bool((cfg.get("telegram") or {}).get("yandex_disk_from_url", True))


def get_telegram_google_drive_url_enabled() -> bool:
    cfg = load_config()
    return bool((cfg.get("telegram") or {}).get("google_drive_from_url", True))


def get_telegram_bot_token() -> str:
    cfg = load_config()
    tg = cfg.get("telegram") or {}
    token = (tg.get("bot_token") or tg.get("TELEGRAM_BOT_TOKEN") or "").strip()
    if not token:
        raise ValueError(
            "TELEGRAM_BOT_TOKEN не задан. Добавьте его в .env, в telegram.bot_token в config.yaml "
            "или в переменную окружения TELEGRAM_BOT_TOKEN"
        )
    return token


def get_ollama_config() -> dict[str, Any]:
    cfg = load_config()
    return cfg.get(
        "ollama",
        {
            "url": "http://localhost:11434/api/generate",
            "model": "gpt-oss:20b",
            "timeout": 120,
        },
    )


def get_paths() -> dict[str, Path]:
    cfg = load_config()
    base = get_project_root()
    paths = cfg.get("paths", {})
    return {
        "prompts_dir": base / paths.get("prompts_dir", "prompt"),
        "output_dir": base / paths.get("output_dir", "output"),
        "temp_dir": base / paths.get("temp_dir", "temp"),
        "drop_dir": base / paths.get("drop_dir", "drop"),
        "logs_dir": base / paths.get("logs_dir", "logs"),
    }
