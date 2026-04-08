#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Пайплайн обработки медиа: транскрипция (Groq) + обработка (локальная LLM).
Используется ботом и watcher'ом drop-директории.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from config_loader import get_paths, get_ollama_config
from pipeline_logger import get_logger
from webm_to_txt import transcribe_audio, SUPPORTED_EXTENSIONS
from localLLMmanager import LocalLLMManager

log = get_logger("media")


def process_media_file(
    media_path: str | Path,
    language: str = "ru",
    output_path: Optional[str | Path] = None,
) -> Optional[Path]:
    """
    Обрабатывает медиафайл: транскрипция → LLM → сохранение в output.

    Args:
        media_path: Путь к аудио/видео (.webm, .mp4, .ogg и др.)
        language: Язык аудио
        output_path: Путь для .txt (если None — в output_dir с timestamp)

    Returns:
        Путь к сохранённому документу или None при ошибке
    """
    path = Path(media_path)
    if not path.exists():
        print(f"Файл не найден: {path}", file=sys.stderr)
        return None

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print(f"Неподдерживаемый формат: {path.suffix}", file=sys.stderr)
        return None

    paths = get_paths()
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    paths["temp_dir"].mkdir(parents=True, exist_ok=True)

    log.info("Обработка медиа: %s", path)
    try:
        # 1. Транскрипция
        log.info("Шаг 1: транскрипция Groq Whisper")
        transcription = transcribe_audio(path, language=language)
        if not transcription.strip():
            log.warning("Пустая транскрипция")
            return None

        # 2. Обработка через локальную LLM
        log.info("Шаг 2: обработка локальной LLM (prompt: process_transcription)")
        ollama_cfg = get_ollama_config()
        manager = LocalLLMManager(
            ollama_url=ollama_cfg.get("url", "http://localhost:11434/api/generate"),
            model=ollama_cfg.get("model", "gpt-oss:20b"),
            prompts_dir=str(paths["prompts_dir"]),
            timeout=ollama_cfg.get("timeout", 120),
        )

        raw_txt = paths["temp_dir"] / f"transcription_{path.stem}.txt"
        raw_txt.write_text(transcription, encoding="utf-8")

        processed = manager.execute_prompt(
            prompt_name="process_transcription",
            context_path=str(raw_txt),
        )
        if not processed:
            processed = transcription

        raw_txt.unlink(missing_ok=True)

        # 3. Сохранение
        log.info("Шаг 3: сохранение документа")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_path is None:
            output_path = paths["output_dir"] / f"document_{path.stem}_{ts}.txt"
        else:
            output_path = Path(output_path)

        output_path.write_text(processed, encoding="utf-8")
        log.info("Документ сохранён: %s", output_path)
        return output_path

    except Exception as e:
        log.exception("Ошибка обработки %s", path)
        return None
