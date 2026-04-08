#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Наблюдатель за drop-директорией: автоматически обрабатывает видео/аудио.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from .config import get_paths
from .logging_config import get_logger, setup_logging
from .media_pipeline import process_media_file
from .transcription import SUPPORTED_EXTENSIONS

log = get_logger("watch")

STABILITY_SEC = 3
POLL_INTERVAL = 2


def watch_drop_dir(drop_dir: Path, language: str = "ru", move_after: bool = True) -> None:
    drop_dir = Path(drop_dir)
    drop_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = drop_dir / "processed"
    if move_after:
        processed_dir.mkdir(exist_ok=True)

    seen = set()
    log.info("Наблюдаю за %s. Форматы: %s", drop_dir, ", ".join(SUPPORTED_EXTENSIONS))
    print(f"Наблюдаю за {drop_dir}. Поддерживаемые форматы: {', '.join(SUPPORTED_EXTENSIONS)}")
    print("Кладите видео/аудио в папку для автоматической обработки. Ctrl+C для выхода.\n")

    while True:
        try:
            for f in drop_dir.iterdir():
                if not f.is_file():
                    continue
                if f.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                if f.name.startswith("."):
                    continue

                key = (f.name, f.stat().st_mtime)
                if key in seen:
                    continue

                mtime = f.stat().st_mtime
                time.sleep(STABILITY_SEC)
                if f.exists() and abs(f.stat().st_mtime - mtime) < 0.1:
                    seen.add(key)
                    log.info("Обнаружен файл, обрабатываю: %s", f.name)
                    print(f"Обрабатываю: {f.name}")
                    try:
                        out = process_media_file(f, language=language)
                        if out:
                            log.info("Обработан → %s", out)
                            print(f"  → {out}")
                            if move_after:
                                dest = processed_dir / f.name
                                f.rename(dest)
                                log.info("Перемещено в %s", dest)
                                print(f"  Перемещено в {dest}")
                    except Exception as e:
                        log.exception("Ошибка обработки %s", f.name)
                        print(f"  Ошибка: {e}", file=sys.stderr)
                        seen.discard(key)

        except KeyboardInterrupt:
            log.info("Выход по Ctrl+C")
            print("\nВыход.")
            break
        except Exception as e:
            log.exception("Ошибка цикла")
            print(f"Ошибка: {e}", file=sys.stderr)

        time.sleep(POLL_INTERVAL)


def main() -> None:
    import argparse

    setup_logging()

    parser = argparse.ArgumentParser(description="Автообработка видео/аудио из drop-папки")
    parser.add_argument("-d", "--dir", help="Drop-директория (по умолчанию из config)")
    parser.add_argument("-l", "--language", default="ru", help="Язык аудио")
    parser.add_argument("--no-move", action="store_true", help="Не перемещать обработанные файлы")

    args = parser.parse_args()

    paths = get_paths()
    drop_dir = Path(args.dir) if args.dir else paths["drop_dir"]

    watch_drop_dir(drop_dir, language=args.language, move_after=not args.no_move)


if __name__ == "__main__":
    main()
