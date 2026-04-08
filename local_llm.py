#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Клиент для работы с локальными LLM (Ollama).
Конфигурация загружается из config.yaml через config_loader.
"""

from dataclasses import dataclass
from typing import Optional

import requests

try:
    from config_loader import load_config, get_ollama_config
except ImportError:
    load_config = get_ollama_config = None


@dataclass
class LocalLLMConfig:
    """Конфигурация локальной LLM."""
    backend: str = "ollama"
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_model: str = "gpt-oss:20b"
    timeout_s: int = 120
    max_retries: int = 2
    retry_backoff_s: float = 1.0
    temperature: float = 0.2
    num_ctx: int = 16384


def load_local_llm_config() -> LocalLLMConfig:
    """
    Загружает конфигурацию Ollama из config.yaml.
    """
    if get_ollama_config is None:
        return LocalLLMConfig()

    cfg = get_ollama_config()
    return LocalLLMConfig(
        backend="ollama",
        ollama_url=cfg.get("url", "http://localhost:11434/api/generate"),
        ollama_model=cfg.get("model", "gpt-oss:20b"),
        timeout_s=int(cfg.get("timeout", 120)),
    )


class LocalLLMClient:
    """Клиент для генерации через Ollama API."""

    def __init__(self, config: LocalLLMConfig, session: Optional[requests.Session] = None):
        self.config = config
        self.session = session or requests.Session()

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        num_ctx: Optional[int] = None,
    ) -> Optional[str]:
        """
        Генерирует ответ от Ollama.

        Args:
            prompt: Текст промпта
            system_prompt: Системный промпт (опционально)
            temperature: Температура (переопределяет конфиг)
            num_ctx: Размер контекста (переопределяет конфиг)

        Returns:
            Сгенерированный текст или None при ошибке
        """
        temp = temperature if temperature is not None else self.config.temperature
        ctx = num_ctx if num_ctx is not None else self.config.num_ctx

        payload = {
            "model": self.config.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temp,
                "num_ctx": ctx,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            r = self.session.post(
                self.config.ollama_url,
                json=payload,
                timeout=self.config.timeout_s,
            )
            r.raise_for_status()
            data = r.json()
            return data.get("response", "").strip()
        except Exception:
            raise
