#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Транскрипция аудио через Groq Whisper Large v3 Turbo.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from groq import BadRequestError, Groq

from .config import get_groq_api_key, get_groq_http_timeout, get_groq_max_file_mb, get_paths
from .logging_config import get_logger

log = get_logger("transcription")

WHISPER_MODEL = "whisper-large-v3-turbo"
SUPPORTED_EXTENSIONS = {".webm", ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mpga", ".mpeg", ".mp4"}
VIDEO_EXTENSIONS = {".webm", ".mp4", ".mpeg"}
MAX_FILE_SIZE_MB = 25


def _get_ffmpeg_path() -> str:
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        return get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def _get_ffprobe_path() -> str | None:
    ffmpeg = _get_ffmpeg_path()
    if ffmpeg == "ffmpeg":
        return None
    parent = Path(ffmpeg).parent
    suffix = Path(ffmpeg).suffix
    p = parent / (f"ffprobe{suffix}")
    return str(p) if p.exists() else None


def _extract_audio_only(path: Path) -> Path:
    try:
        from pydub import AudioSegment

        ffmpeg = _get_ffmpeg_path()
        ffprobe = _get_ffprobe_path()
        if ffmpeg != "ffmpeg":
            AudioSegment.converter = ffmpeg
            if ffprobe:
                AudioSegment.ffprobe = ffprobe
    except ImportError:
        log.warning("pydub не установлен, пропуск извлечения аудио")
        return path

    suffix = path.suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        return path

    out = Path(tempfile.gettempdir()) / f"groq_audio_only_{path.stem}.ogg"
    try:
        if suffix == ".webm":
            sound = AudioSegment.from_file(str(path), format="webm")
        elif suffix in (".mp4", ".mpeg"):
            sound = AudioSegment.from_file(str(path), format=suffix[1:])
        else:
            return path
        sound.export(str(out), format="ogg")
        log.info("Извлечён только звук: %s -> %s (%.1f MB)", path.name, out.name, out.stat().st_size / 1024 / 1024)
        return out
    except Exception as e:
        log.warning("pydub извлечение аудио: %s", e)
        return path


def _compress_with_ffmpeg(path: Path, max_mb: int = MAX_FILE_SIZE_MB) -> Path:
    size_mb = path.stat().st_size / 1024 / 1024
    if size_mb <= max_mb:
        return path

    max_bytes = max_mb * 1024 * 1024
    attempts = [
        (["-ar", "16000", "-ac", "1", "-map", "0:a", "-c:a", "flac"], ".flac"),
        (["-ar", "16000", "-ac", "1", "-map", "0:a", "-c:a", "libmp3lame", "-b:a", "64k"], ".mp3"),
    ]

    ffmpeg = _get_ffmpeg_path()
    for extra_args, ext in attempts:
        try:
            out = Path(tempfile.gettempdir()) / f"groq_whisper_{path.stem}{ext}"
            cmd = [ffmpeg, "-y", "-i", str(path)] + extra_args + [str(out)]
            subprocess.run(cmd, capture_output=True, check=True, timeout=600)
            if out.exists():
                sz = out.stat().st_size
                if sz <= max_bytes:
                    log.info("Файл сжат через FFmpeg: %s -> %s (%.1f MB)", path.name, out.name, sz / 1024 / 1024)
                    return out
                out.unlink(missing_ok=True)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            log.warning("FFmpeg: %s", e)
            break

    raise ValueError(
        f"Файл слишком большой: {size_mb:.1f} MB. Лимит Groq: {max_mb} MB. "
        "Установите FFmpeg для автосжатия или разбейте аудио на части."
    )


def _transcode_to_mp3_for_groq(path: Path) -> Path:
    import hashlib

    ffmpeg = _get_ffmpeg_path()
    h = hashlib.sha256(str(path.resolve()).encode("utf-8", errors="replace")).hexdigest()[:16]
    out = Path(tempfile.gettempdir()) / f"groq_normalized_{h}.mp3"
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "libmp3lame",
        "-b:a",
        "96k",
        str(out),
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=600)
    except FileNotFoundError:
        raise ValueError(
            "Нужен FFmpeg для перекодирования M4A под Groq. Установите FFmpeg или imageio-ffmpeg."
        ) from None
    except subprocess.CalledProcessError as e:
        log.warning("FFmpeg stderr: %s", (e.stderr or b"")[:800])
        raise ValueError("Не удалось перекодировать аудио в MP3 для Groq.") from e
    if not out.exists() or out.stat().st_size == 0:
        raise ValueError("FFmpeg не создал MP3 для Groq.")
    return out


def transcribe_audio(
    audio_path: str | Path,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    response_format: str = "text",
) -> str:
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    log.info("Транскрипция: %s", path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Неподдерживаемый формат: {suffix}. "
            f"Поддерживаются: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    api_key = get_groq_api_key()
    client = Groq(api_key=api_key, timeout=get_groq_http_timeout())

    kwargs = {
        "model": WHISPER_MODEL,
        "response_format": response_format,
        "temperature": 0.0,
    }
    if language:
        kwargs["language"] = language
    if prompt:
        kwargs["prompt"] = prompt[:224]

    temp_audio = None
    if path.suffix.lower() in VIDEO_EXTENSIONS:
        extracted = _extract_audio_only(path)
        if extracted != path:
            temp_audio = extracted
            path = extracted

    file_size = path.stat().st_size
    max_mb = get_groq_max_file_mb()
    max_bytes = max_mb * 1024 * 1024
    temp_file = temp_audio
    if file_size > max_bytes:
        path = _compress_with_ffmpeg(path, max_mb)
        temp_file = path
        file_size = path.stat().st_size
        if file_size > max_bytes:
            raise ValueError(
                f"Файл слишком большой: {file_size / 1024 / 1024:.1f} MB. Лимит: {max_mb} MB."
            )

    norm_for_groq: Optional[Path] = None
    if path.suffix.lower() == ".m4a":
        log.info("M4A: перекодирование в MP3 для Groq Whisper (обход 400 по типу файла)")
        norm_for_groq = _transcode_to_mp3_for_groq(path)
        path = norm_for_groq

    upload_name = f"audio{path.suffix.lower()}"

    try:
        data = path.read_bytes()
        try:
            transcription = client.audio.transcriptions.create(
                file=(upload_name, data),
                **kwargs,
            )
        except BadRequestError as e:
            err = str(e).lower()
            if "file must be one of" not in err:
                raise
            if norm_for_groq is not None:
                raise
            log.warning("Groq отклонил формат (%s), перекодирование в MP3", upload_name)
            norm_for_groq = _transcode_to_mp3_for_groq(path)
            path = norm_for_groq
            transcription = client.audio.transcriptions.create(
                file=("audio.mp3", path.read_bytes()),
                **kwargs,
            )

        text = (
            transcription
            if isinstance(transcription, str)
            else (transcription.text if hasattr(transcription, "text") else str(transcription))
        )
        log.info("Транскрипция завершена: %s (%d символов)", path, len(text))
        return text
    finally:
        if norm_for_groq is not None and norm_for_groq.exists():
            try:
                norm_for_groq.unlink()
            except OSError:
                pass
        for f in (temp_file, temp_audio):
            if f and f.exists() and str(f).startswith(tempfile.gettempdir()) and "groq_" in str(f):
                try:
                    f.unlink()
                except OSError:
                    pass


def transcribe_audio_url(
    audio_url: str,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    response_format: str = "text",
) -> str:
    api_key = get_groq_api_key()
    client = Groq(api_key=api_key, timeout=get_groq_http_timeout())

    kwargs = {
        "model": WHISPER_MODEL,
        "response_format": response_format,
        "temperature": 0.0,
        "url": audio_url,
    }
    if language:
        kwargs["language"] = language
    if prompt:
        kwargs["prompt"] = prompt[:224]

    log.info("Транскрипция по URL: %s", audio_url[:120] + ("..." if len(audio_url) > 120 else ""))
    transcription = client.audio.transcriptions.create(**kwargs)
    text = (
        transcription
        if isinstance(transcription, str)
        else (transcription.text if hasattr(transcription, "text") else str(transcription))
    )
    log.info("Транскрипция по URL завершена (%d символов)", len(text))
    return text


def webm_to_txt(
    webm_path: str | Path,
    output_path: Optional[str | Path] = None,
    language: Optional[str] = None,
) -> str:
    text = transcribe_audio(webm_path, language=language)
    path = Path(webm_path)

    if output_path is None:
        output_path = path.with_suffix(".txt")
    else:
        output_path = Path(output_path)

    output_path.write_text(text, encoding="utf-8")
    return text


def _default_audio_from_temp() -> Optional[Path]:
    temp_dir = get_paths()["temp_dir"]
    if not temp_dir.exists():
        return None
    candidates = [
        f for f in temp_dir.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    import argparse

    from .logging_config import setup_logging

    setup_logging()

    parser = argparse.ArgumentParser(description="Транскрипция аудио в txt через Groq Whisper")
    parser.add_argument(
        "audio",
        nargs="?",
        default=None,
        help="Путь к аудиофайлу (по умолчанию — последний файл из temp/)",
    )
    parser.add_argument("-o", "--output", help="Путь для сохранения .txt")
    parser.add_argument("-l", "--language", help="Язык аудио (ru, en, ...)")
    parser.add_argument("--no-save", action="store_true", help="Только вывести текст, не сохранять")

    args = parser.parse_args()

    audio_path = args.audio
    if audio_path is None:
        audio_path = _default_audio_from_temp()
        if audio_path is None:
            print("Ошибка: укажите файл или положите аудио в temp/", file=sys.stderr)
            sys.exit(1)
        print(f"Использую: {audio_path}")

    try:
        text = transcribe_audio(audio_path, language=args.language)
        if args.no_save:
            print(text)
        else:
            out = args.output or Path(audio_path).with_suffix(".txt")
            Path(out).write_text(text, encoding="utf-8")
            log.info("Сохранено: %s", out)
            print(f"Сохранено: {out}")
            print(text)
    except Exception as e:
        log.exception("Ошибка транскрипции")
        print(f"Ошибка: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
