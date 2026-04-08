#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Загрузчик конфигурации из config.yaml.
Все ключи (Groq, Telegram, Ollama) вынесены в конфигурацию.
"""

import os
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    yaml = None

# Путь к конфигу относительно корня проекта
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
_PROJECT_ROOT = CONFIG_PATH.parent


def _load_dotenv() -> None:
    """Подхватывает переменные из .env в корне проекта (если установлен python-dotenv)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(_PROJECT_ROOT / ".env", override=False)


_load_dotenv()


def load_config() -> dict:
    """
    Загружает конфигурацию из config.yaml.
    Секреты задаются в файле .env (см. .env.example) или в окружении:
    - GROQ_API_KEY
    - TELEGRAM_BOT_TOKEN
    """
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Файл конфигурации не найден: {CONFIG_PATH}\n"
            "Создайте config.yaml на основе config.example.yaml"
        )

    if yaml is None:
        raise ImportError("Установите PyYAML: pip install pyyaml")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Переопределение из переменных окружения
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
    """Макс. размер файла для Groq Whisper (MB)."""
    cfg = load_config()
    return int((cfg.get("groq") or {}).get("max_file_size_mb", 25))


def get_groq_http_timeout():
    """Таймауты HTTP для Groq (длинное аудио, медленный upload)."""
    import httpx

    cfg = load_config()
    g = cfg.get("groq") or {}
    total = float(g.get("http_timeout_seconds", 600))
    connect = float(g.get("http_connect_timeout_seconds", 60))
    return httpx.Timeout(total, connect=connect)


def get_groq_api_key() -> str:
    """API ключ Groq для Whisper."""
    cfg = load_config()
    groq = cfg.get("groq") or {}
    key = groq.get("api_key") or groq.get("GROQ_API_KEY") or ""
    key = (key or "").strip()
    if not key:
        raise ValueError(
            "GROQ_API_KEY не задан. Добавьте его в .env, в groq.api_key в config.yaml "
            "или в переменную окружения GROQ_API_KEY"
        )
    return key


def get_telegram_yandex_url_enabled() -> bool:
    """Транскрипция по публичной ссылке Яндекс.Диска в тексте сообщения."""
    cfg = load_config()
    return bool((cfg.get("telegram") or {}).get("yandex_disk_from_url", True))


def get_telegram_google_drive_url_enabled() -> bool:
    """Транскрипция по публичной ссылке Google Drive в тексте сообщения."""
    cfg = load_config()
    return bool((cfg.get("telegram") or {}).get("google_drive_from_url", True))


def get_telegram_bot_token() -> str:
    """Токен Telegram бота."""
    cfg = load_config()
    tg = cfg.get("telegram") or {}
    token = tg.get("bot_token") or tg.get("TELEGRAM_BOT_TOKEN") or ""
    token = (token or "").strip()
    if not token:
        raise ValueError(
            "TELEGRAM_BOT_TOKEN не задан. Добавьте его в .env, в telegram.bot_token в config.yaml "
            "или в переменную окружения TELEGRAM_BOT_TOKEN"
        )
    return token


def get_ollama_config() -> dict:
    """Настройки Ollama для локальной LLM."""
    cfg = load_config()
    return cfg.get("ollama", {
        "url": "http://localhost:11434/api/generate",
        "model": "gpt-oss:20b",
        "timeout": 120,
    })


def get_paths() -> dict:
    """Пути к директориям."""
    cfg = load_config()
    base = Path(__file__).resolve().parent
    paths = cfg.get("paths", {})
    return {
        "prompts_dir": base / paths.get("prompts_dir", "prompt"),
        "output_dir": base / paths.get("output_dir", "output"),
        "temp_dir": base / paths.get("temp_dir", "temp"),
        "drop_dir": base / paths.get("drop_dir", "drop"),
        "logs_dir": base / paths.get("logs_dir", "logs"),
    }
