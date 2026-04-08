#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Менеджер для локальных LLM через Ollama: промпты из каталога и контекст из файлов.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import requests

from .local_llm import LocalLLMClient, load_local_llm_config


class LocalLLMManager:
    """Менеджер для работы с Ollama API."""

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434/api/generate",
        model: str = "gpt-oss:20b",
        prompts_dir: str = "prompt",
        timeout: int = 120,
    ):
        self.ollama_url = ollama_url
        self.model = model
        self.prompts_dir = Path(prompts_dir)
        self.timeout = timeout
        self.prompts_dir.mkdir(exist_ok=True)

    def check_ollama_connection(self) -> bool:
        try:
            tags_url = self.ollama_url.replace("/api/generate", "/api/tags")
            r = requests.get(tags_url, timeout=5)
            return r.status_code == 200
        except Exception as e:
            print(f"Ошибка подключения к Ollama: {e}", file=sys.stderr)
            return False

    def check_model_available(self) -> bool:
        try:
            tags_url = self.ollama_url.replace("/api/generate", "/api/tags")
            r = requests.get(tags_url, timeout=5)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                return self.model in models
            return False
        except Exception as e:
            print(f"Ошибка при проверке модели: {e}", file=sys.stderr)
            return False

    def load_prompt(self, prompt_name: str) -> Optional[str]:
        prompt_file = self.prompts_dir / prompt_name
        if not prompt_file.exists():
            prompt_file = self.prompts_dir / f"{prompt_name}.txt"

        if not prompt_file.exists():
            print(f"Промпт '{prompt_name}' не найден в {self.prompts_dir}", file=sys.stderr)
            return None

        try:
            return prompt_file.read_text(encoding="utf-8").strip()
        except Exception as e:
            print(f"Ошибка при чтении промпта {prompt_name}: {e}", file=sys.stderr)
            return None

    def load_context(self, context_path: str) -> Optional[str]:
        context_file = Path(context_path)

        if not context_file.exists():
            print(f"Файл контекста '{context_path}' не найден", file=sys.stderr)
            return None

        try:
            return context_file.read_text(encoding="utf-8").strip()
        except Exception as e:
            print(f"Ошибка при чтении контекста {context_path}: {e}", file=sys.stderr)
            return None

    def format_prompt_with_context(self, prompt_template: str, context: str) -> str:
        formatted = prompt_template.replace("{context}", context)
        try:
            formatted = formatted.format(context=context)
        except (KeyError, ValueError):
            pass
        return formatted

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        num_ctx: int = 16384,
    ) -> Optional[str]:
        try:
            cfg = load_local_llm_config()
            cfg = cfg.__class__(
                backend=cfg.backend,
                ollama_url=self.ollama_url or cfg.ollama_url,
                ollama_model=self.model or cfg.ollama_model,
                timeout_s=int(self.timeout),
                max_retries=cfg.max_retries,
                retry_backoff_s=cfg.retry_backoff_s,
                temperature=float(temperature),
                num_ctx=int(num_ctx),
            )
            client = LocalLLMClient(cfg, session=requests.Session())
            return client.generate(
                prompt, system_prompt=system_prompt, temperature=temperature, num_ctx=num_ctx
            )
        except Exception as e:
            print(f"Ошибка при генерации: {e}", file=sys.stderr)
            return None

    def execute_prompt(
        self,
        prompt_name: str,
        context_path: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        num_ctx: int = 16384,
    ) -> Optional[str]:
        prompt_template = self.load_prompt(prompt_name)
        if not prompt_template:
            return None

        context = self.load_context(context_path)
        if context is None:
            print("Context is None")
            return None

        formatted_prompt = self.format_prompt_with_context(prompt_template, context)

        return self.generate(
            prompt=formatted_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            num_ctx=num_ctx,
        )

    def list_prompts(self) -> List[str]:
        prompts = []
        for file in self.prompts_dir.glob("*.txt"):
            prompts.append(file.stem)
        return sorted(prompts)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Менеджер для локальных LLM через Ollama")
    parser.add_argument("--prompt", "-p", required=True, help="Имя промпта из директории prompt")
    parser.add_argument("--context", "-c", required=True, help="Путь к файлу с контекстом")
    parser.add_argument("--model", "-m", default="gpt-oss:20b", help="Модель Ollama")
    parser.add_argument("--url", default="http://localhost:11434/api/generate", help="URL Ollama API")
    parser.add_argument("--system", "-s", help="Системный промпт (опционально)")
    parser.add_argument("--temperature", "-t", type=float, default=0.2, help="Температура генерации")
    parser.add_argument("--list-prompts", action="store_true", help="Показать список доступных промптов")
    parser.add_argument("--prompts-dir", default="prompt", help="Директория с промптами")

    args = parser.parse_args()

    manager = LocalLLMManager(
        ollama_url=args.url,
        model=args.model,
        prompts_dir=args.prompts_dir,
    )

    if args.list_prompts:
        prompts = manager.list_prompts()
        if prompts:
            print("Доступные промпты:")
            for p in prompts:
                print(f"  - {p}")
        else:
            print(f"Промпты не найдены в директории {args.prompts_dir}")
        return

    if not manager.check_ollama_connection():
        print("Ошибка: Ollama недоступен", file=sys.stderr)
        sys.exit(1)

    result = manager.execute_prompt(
        prompt_name=args.prompt,
        context_path=args.context,
        system_prompt=args.system,
        temperature=args.temperature,
    )

    if result:
        print(result)
    else:
        print("Ошибка при выполнении промпта", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
