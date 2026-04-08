#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Telegram-бот: приём аудио/видео, транскрипция Groq Whisper, постобработка через Ollama.
"""

import asyncio
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

from .config import (
    get_ollama_config,
    get_paths,
    get_telegram_bot_token,
    get_telegram_google_drive_url_enabled,
    get_telegram_yandex_url_enabled,
)
from .dialog_registry import (
    document_path_for_ts,
    dialogue_full_path_for_ts,
    filter_dialogue_entries,
    get_metadata,
    list_documents,
    parse_dialog_date,
    save_metadata,
)
from .google_drive import (
    extract_google_drive_file_id,
    extract_google_drive_folder_id,
    google_drive_download_file_to_temp,
    google_drive_folder_first_file_id,
)
from .local_llm_manager import LocalLLMManager
from .logging_config import get_logger, setup_logging
from .transcription import transcribe_audio
from .yandex_disk import extract_yandex_public_url, yandex_public_download_to_temp

try:
    from telegram.ext import ConversationHandler
except ImportError:
    ConversationHandler = None  # type: ignore[misc, assignment]

log = get_logger("telegram")

# Мастер /dialog_register: выбор файла → дата → участники → ключевые поинты
DIALOG_CHOOSE, DIALOG_DATE, DIALOG_PARTS, DIALOG_KP = range(4)

# Лимит скачивания файла ботом (Telegram Bot API)
TELEGRAM_MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024

# По умолчанию у PTB короткий connect — на нестабильной сети падает start_tls / get_file.
TELEGRAM_HTTP_CONNECT_TIMEOUT = 120.0
TELEGRAM_HTTP_READ_TIMEOUT = 180.0
TELEGRAM_HTTP_WRITE_TIMEOUT = 180.0
TELEGRAM_HTTP_POOL_TIMEOUT = 60.0
TELEGRAM_GET_UPDATES_CONNECT_TIMEOUT = 120.0
TELEGRAM_GET_UPDATES_READ_TIMEOUT = 120.0
TELEGRAM_API_RETRIES = 3

MSG_FILE_TOO_BIG = (
    "Файл больше 20 МБ — Telegram не даёт боту скачать такие файлы.\n\n"
    "Варианты:\n"
    "• Публичная ссылка Яндекс.Диска или Google Drive — вставьте в сообщение.\n"
    "• Папка drop/ на ПК и watch_drop.py.\n"
    "• Сжать или разрезать файл до < 20 МБ."
)


def _audio_ext_from_mime(mime: str | None) -> str | None:
    """
    Расширение файла по MIME из Telegram (должно совпадать с реальным содержимым для Groq).
    Иначе имя .m4a при OGG Opus даёт 400 от Whisper.
    """
    if not mime:
        return None
    m = mime.split(";")[0].strip().lower()
    if m in ("audio/ogg", "audio/opus", "application/ogg"):
        return ".ogg"
    if m in ("audio/mpeg", "audio/mp3"):
        return ".mp3"
    if m in ("audio/mp4", "audio/x-m4a", "audio/aac"):
        return ".m4a"
    if m in ("audio/wav", "audio/x-wav"):
        return ".wav"
    if m == "audio/webm":
        return ".webm"
    if m in ("audio/flac", "audio/x-flac"):
        return ".flac"
    if m.startswith("video/"):
        if "mp4" in m:
            return ".mp4"
        if "webm" in m:
            return ".webm"
    return None


def _ext_from_filename(fname: str) -> str | None:
    """Расширение по имени файла (нижний регистр)."""
    if not fname:
        return None
    low = fname.lower()
    for e in (".webm", ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4", ".mpeg", ".mpga"):
        if low.endswith(e):
            return e
    return None


# Инициализация путей
PATHS = None


def _ensure_paths():
    global PATHS
    if PATHS is None:
        PATHS = get_paths()
        PATHS["temp_dir"].mkdir(parents=True, exist_ok=True)
        PATHS["output_dir"].mkdir(parents=True, exist_ok=True)


async def _pipeline_llm_save_reply(msg, transcription: str, ts: str) -> None:
    """Обработка транскрипции локальной LLM, сохранение, ответ пользователю."""
    _ensure_paths()
    output_dir = PATHS["output_dir"]
    loop = asyncio.get_running_loop()

    if not transcription.strip():
        await msg.reply_text("Не удалось распознать речь в аудио.")
        return

    # Промежуточный результат: полный текст без суммаризации
    dialogue_full_path = output_dir / f"dialogue_full_{ts}.txt"
    dialogue_full_path.write_text(transcription, encoding="utf-8")
    log.info("Шаг 2: полный диалог сохранён (без суммаризации) %s", dialogue_full_path)

    log.info("Шаг 3: обработка локальной LLM")
    ollama_cfg = _get_ollama_config()
    manager = LocalLLMManager(
        ollama_url=ollama_cfg.get("url", "http://localhost:11434/api/generate"),
        model=ollama_cfg.get("model", "gpt-oss:20b"),
        prompts_dir=str(PATHS["prompts_dir"]),
        timeout=ollama_cfg.get("timeout", 120),
    )

    processed = await loop.run_in_executor(
        None,
        lambda: manager.execute_prompt(
            prompt_name="process_transcription",
            context_path=str(dialogue_full_path),
        ),
    )

    if not processed:
        processed = transcription

    doc_path = output_dir / f"document_{ts}.txt"
    doc_path.write_text(processed, encoding="utf-8")
    log.info("Шаг 4: документ после LLM сохранён %s", doc_path)

    if len(processed) <= 4096:
        await msg.reply_text(processed)
    else:
        with open(doc_path, "rb") as f:
            await msg.reply_document(document=f, filename=doc_path.name)

    await msg.reply_text(
        f"Полный текст (без суммаризации): {dialogue_full_path}\n"
        f"Документ после обработки: {doc_path}"
    )


async def _get_telegram_file(bot, file_id: str, file_size: int | None):
    """get_file; при известном размере > 20 МБ не вызывает API. Повторы при сетевых таймаутах."""
    from telegram.error import BadRequest, NetworkError

    if file_size is not None and file_size > TELEGRAM_MAX_DOWNLOAD_BYTES:
        raise BadRequest("File is too big")

    last_net: Exception | None = None
    for attempt in range(TELEGRAM_API_RETRIES):
        try:
            return await bot.get_file(file_id)
        except BadRequest as e:
            err = (getattr(e, "message", None) or str(e)).lower()
            if "too big" in err:
                raise BadRequest("File is too big") from e
            raise
        except NetworkError as e:
            last_net = e
            log.warning(
                "Telegram get_file: сеть/таймаут (попытка %d/%d): %s",
                attempt + 1,
                TELEGRAM_API_RETRIES,
                e,
            )
            if attempt + 1 < TELEGRAM_API_RETRIES:
                await asyncio.sleep(2.0 * (attempt + 1))
    assert last_net is not None
    raise last_net


async def _download_telegram_to_drive(file_obj, dest: Path) -> None:
    """Скачивание файла с серверов Telegram с повторами при NetworkError."""
    from telegram.error import NetworkError

    last_net: Exception | None = None
    for attempt in range(TELEGRAM_API_RETRIES):
        try:
            await file_obj.download_to_drive(dest)
            return
        except NetworkError as e:
            last_net = e
            log.warning(
                "Telegram download_to_drive: сеть/таймаут (попытка %d/%d): %s",
                attempt + 1,
                TELEGRAM_API_RETRIES,
                e,
            )
            if attempt + 1 < TELEGRAM_API_RETRIES:
                await asyncio.sleep(2.0 * (attempt + 1))
    assert last_net is not None
    raise last_net


async def handle_voice_or_video(update, context):
    """Обработка голосовых сообщений и видео-заметок."""
    from telegram.error import BadRequest, NetworkError

    _ensure_paths()
    temp_dir = PATHS["temp_dir"]
    output_dir = PATHS["output_dir"]

    msg = update.message
    if not msg:
        return

    file_obj = None
    ext = ".webm"
    try:
        if msg.voice:
            v = msg.voice
            file_obj = await _get_telegram_file(context.bot, v.file_id, v.file_size)
            ext = _audio_ext_from_mime(v.mime_type) or ".ogg"
        elif msg.video_note:
            vn = msg.video_note
            file_obj = await _get_telegram_file(context.bot, vn.file_id, vn.file_size)
            ext = _audio_ext_from_mime(vn.mime_type) or ".mp4"
        elif msg.video:
            vid = msg.video
            file_obj = await _get_telegram_file(context.bot, vid.file_id, vid.file_size)
            ext = _audio_ext_from_mime(vid.mime_type) or ".mp4"
        elif msg.audio:
            a = msg.audio
            file_obj = await _get_telegram_file(context.bot, a.file_id, a.file_size)
            ext = _audio_ext_from_mime(a.mime_type)
            if ext is None:
                ext = _ext_from_filename(a.file_name or "")
            if ext is None:
                ext = ".mp3"
        elif msg.document:
            doc = msg.document
            fname = doc.file_name or ""
            if any(fname.lower().endswith(e) for e in [".webm", ".ogg", ".mp3", ".wav", ".m4a", ".mp4"]):
                file_obj = await _get_telegram_file(context.bot, doc.file_id, doc.file_size)
                suf = Path(fname).suffix
                ext = (
                    _audio_ext_from_mime(doc.mime_type)
                    or _ext_from_filename(fname)
                    or (suf.lower() if suf else None)
                    or ".webm"
                )
    except BadRequest as e:
        err = (getattr(e, "message", None) or str(e)).lower()
        if "too big" in err:
            log.warning("Файл слишком большой для Telegram Bot API")
            await msg.reply_text(MSG_FILE_TOO_BIG)
            return
        log.exception("get_file: %s", e)
        await msg.reply_text(f"Ошибка Telegram: {e}")
        return
    except NetworkError as e:
        log.exception("Telegram API (сеть), get_file после повторов: %s", e)
        try:
            await msg.reply_text(
                "Не удалось связаться с серверами Telegram (таймаут TLS/сеть). "
                "Повторите отправку позже или проверьте доступ к api.telegram.org (VPN, прокси, файрвол)."
            )
        except Exception:
            pass
        return

    if not file_obj:
        await msg.reply_text(
            "Отправьте голосовое, аудио-файл (.mp3 и др.), видео-заметку или документ.\n"
            "Команда: /start (со слэшем /, не \\start)"
        )
        return

    log.info("Получено медиа от user_id=%s", msg.from_user.id if msg.from_user else "?")
    await msg.reply_text("Обрабатываю...")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        audio_path = temp_dir / f"audio_{ts}{ext}"
        await _download_telegram_to_drive(file_obj, audio_path)
        log.info("Скачано: %s", audio_path)

        # 1. Транскрипция через Groq Whisper (sync — выполняем в executor)
        log.info("Шаг 1: транскрипция Groq Whisper")
        loop = asyncio.get_running_loop()
        transcription = await loop.run_in_executor(
            None, lambda: transcribe_audio(audio_path, language="ru")
        )
        await _pipeline_llm_save_reply(msg, transcription, ts)

    except NetworkError as e:
        log.exception("Telegram при скачивании/обработке: %s", e)
        try:
            await msg.reply_text(
                "Таймаут сети Telegram при скачивании файла. Повторите отправку."
            )
        except Exception:
            pass
    except Exception as e:
        log.exception("Ошибка обработки медиа")
        err_text = f"Ошибка: {e}"
        if "timed out" in str(e).lower() or "timeout" in str(e).lower():
            err_text += (
                "\n\nТаймаут Groq или сети. Повторите позже; для длинного аудио увеличьте "
                "groq.http_timeout_seconds в config.yaml."
            )
        try:
            await msg.reply_text(err_text)
        except Exception as send_err:
            log.warning("Не удалось отправить сообщение об ошибке: %s", send_err)
    finally:
        # Очистка временных файлов
        try:
            for p in temp_dir.glob(f"*_{ts}*"):
                try:
                    p.unlink()
                except OSError:
                    pass
        except Exception:
            pass


def _get_ollama_config():
    return get_ollama_config()


async def process_yandex_disk_url(msg, public_url: str):
    """
    Транскрипция по публичной ссылке Яндекс.Диска.
    Файл скачивается локально: href на downloader.disk.yandex.ru даёт 302, Groq по url= это не обрабатывает.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    await msg.reply_text(
        "Яндекс.Диск — скачиваю файл, затем транскрибирую (обход редиректа 302 для Groq)..."
    )
    tmp_path = None
    try:
        loop = asyncio.get_running_loop()
        tmp_path = await loop.run_in_executor(
            None,
            lambda u=public_url: yandex_public_download_to_temp(u),
        )
        log.info("Яндекс.Диск: скачан во временный файл %s", tmp_path)
        transcription = await loop.run_in_executor(
            None,
            lambda p=tmp_path: transcribe_audio(p, language="ru"),
        )
        log.info("Шаг 1: транскрипция Groq с локального файла (Яндекс.Диск)")
        await _pipeline_llm_save_reply(msg, transcription, ts)
    except Exception as e:
        log.exception("Яндекс.Диск / Groq")
        await msg.reply_text(
            f"Ошибка: {e}\n\n"
            "Проверьте: ссылка публичная («Всем по ссылке»), в папке есть аудио/видео."
        )
    finally:
        if tmp_path is not None:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass


async def process_google_drive_url(msg, file_id: str):
    """
    Google Drive: скачивание локально (Groq по url= не обрабатывает редиректы 303 Google).
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    await msg.reply_text(
        "Google Drive — скачиваю файл, затем транскрибирую (это обходит ошибку 303 у Groq)..."
    )
    tmp_path = None
    try:
        loop = asyncio.get_running_loop()
        tmp_path = await loop.run_in_executor(
            None,
            lambda fid=file_id: google_drive_download_file_to_temp(fid),
        )
        log.info("Google Drive: скачан во временный файл %s", tmp_path)
        transcription = await loop.run_in_executor(
            None,
            lambda p=tmp_path: transcribe_audio(p, language="ru"),
        )
        log.info("Шаг 1: транскрипция Groq с локального файла (Google Drive)")
        await _pipeline_llm_save_reply(msg, transcription, ts)
    except Exception as e:
        log.exception("Google Drive / Groq")
        await msg.reply_text(
            f"Ошибка: {e}\n\n"
            "Проверьте: доступ «Любой, у кого есть ссылка», размер в лимите Groq, "
            "формат аудио/видео поддерживается."
        )
    finally:
        if tmp_path is not None:
            try:
                p = Path(tmp_path)
                if p.exists():
                    p.unlink(missing_ok=True)
                # gdown кладёт файл в mkdtemp(..., prefix="gdrive_dl_")
                parent = p.parent
                if parent.name.startswith("gdrive_dl_"):
                    shutil.rmtree(parent, ignore_errors=True)
            except OSError:
                pass


async def cmd_start(update, context):
    """Команда /start."""
    await update.message.reply_text(
        "Отправьте голосовое, аудио, документ до 20 МБ.\n\n"
        "Файлы больше 20 МБ: вставьте в чат публичную ссылку:\n"
        "• Яндекс.Диск — «Поделиться» → скопировать ссылку\n"
        "• Google Drive — ссылка на файл или на папку (доступ по ссылке)\n\n"
        "Или: папка drop/ на ПК и watch_drop.py.\n\n"
        "Диалоги:\n"
        "/dialogs — список обработанных документов\n"
        "/dialog_download — скачать полный текст / документ по номеру, ts или фильтру\n"
        "/dialog_show N — показать текст и карточку из базы\n"
        "/dialog_register — дата, участники, ключевые поинты в базу\n"
        "/cancel — выйти из мастера\n\n"
        "Команда /start"
    )


async def cmd_help_text(update, context):
    """Текст без команды: облачные ссылки или подсказка."""
    msg = update.message
    text = msg.text or ""
    if get_telegram_yandex_url_enabled():
        url = extract_yandex_public_url(text)
        if url:
            await process_yandex_disk_url(msg, url)
            return
    if get_telegram_google_drive_url_enabled():
        gid = extract_google_drive_file_id(text)
        if gid:
            await process_google_drive_url(msg, gid)
            return
        folder_id = extract_google_drive_folder_id(text)
        if folder_id:
            await msg.reply_text("Папка Google Drive — получаю список файлов...")
            loop = asyncio.get_running_loop()
            try:
                gid = await loop.run_in_executor(
                    None,
                    lambda fid=folder_id: google_drive_folder_first_file_id(fid),
                )
            except Exception as e:
                log.exception("Google Drive folder")
                await msg.reply_text(
                    f"Не удалось прочитать папку: {e}\n\n"
                    "Откройте доступ «Любой, у кого есть ссылка» или пришлите ссылку на один файл."
                )
                return
            await process_google_drive_url(msg, gid)
            return
    await msg.reply_text(
        "Напишите /start или отправьте аудио/голосовое.\n"
        "Большой файл — публичная ссылка Яндекс.Диска или Google Drive в сообщении.\n"
        "Команда: /start"
    )


def _format_dialog_list_lines(docs: list) -> list[str]:
    lines = []
    for i, d in enumerate(docs, start=1):
        flag = "в базе" if d["has_meta"] else "нет карточки"
        prev = d["preview"]
        if len(prev) > 100:
            prev = prev[:100] + "…"
        lines.append(f"{i}. {d['ts']} ({flag})\n   {prev}")
    return lines


async def cmd_dialogs(update, context):
    """Список уже сохранённых document_*.txt и отметка, есть ли карточка в БД."""
    _ensure_paths()
    docs = list_documents(PATHS["output_dir"])
    if not docs:
        await update.message.reply_text(
            "В каталоге выходных файлов пока нет document_*.txt. "
            "Сначала обработайте аудио."
        )
        return
    lines = _format_dialog_list_lines(docs)
    tail = (
        "\n\n/dialog_show N или /dialog_show YYYYMMDD_HHMMSS — открыть текст\n"
        "/dialog_register — заполнить дату, участников и ключевые поинты"
    )
    text = "Сохранённые документы (новые сверху):\n\n" + "\n\n".join(lines) + tail
    if len(text) > 4000:
        text = text[:3900] + "\n… (список обрезан; уточните вывод по ts)" + tail
    await update.message.reply_text(text)


def _parse_dialog_download_search_args(args: list[str]) -> tuple[str, str | None, str | None]:
    """
    После слова search/find: текст запроса и опционально --from YYYY-MM-DD --to YYYY-MM-DD.
    """
    date_from: str | None = None
    date_to: str | None = None
    chunks: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--from" and i + 1 < len(args):
            date_from = args[i + 1].strip()
            i += 2
            continue
        if a == "--to" and i + 1 < len(args):
            date_to = args[i + 1].strip()
            i += 2
            continue
        chunks.append(a)
        i += 1
    return " ".join(chunks).strip(), date_from, date_to


def _parse_dialog_download_kind_and_target(args: list[str]) -> tuple[str, str]:
    """Возвращает (kind, target) где kind: full | doc | both."""
    if not args:
        return "full", ""
    first = args[0].lower()
    if first in ("full", "doc", "document", "both"):
        kind = "doc" if first in ("doc", "document") else first
        return kind, " ".join(args[1:]).strip()
    return "full", " ".join(args).strip()


async def _reply_document_under_limit(msg, path: Path, caption: str | None = None) -> None:
    """Отправка файла документом; при превышении лимита Telegram — текст с путём."""
    try:
        sz = path.stat().st_size
    except OSError as e:
        await msg.reply_text(f"Не удалось прочитать файл: {e}")
        return
    if sz > TELEGRAM_MAX_DOWNLOAD_BYTES:
        await msg.reply_text(
            f"{path.name}: размер {sz // (1024 * 1024)} МБ — больше лимита Telegram для бота "
            f"({TELEGRAM_MAX_DOWNLOAD_BYTES // (1024 * 1024)} МБ). Файл на диске:\n{path}"
        )
        return
    with open(path, "rb") as f:
        await msg.reply_document(document=f, filename=path.name, caption=caption)


async def cmd_dialog_download(update, context):
    """
    Скачивание полного диалога (dialogue_full), обработанного документа или обоих.
    По номеру из /dialogs, по ts, либо поиск по фильтру (search).
    """
    msg = update.message
    if not msg:
        return
    _ensure_paths()
    outp = PATHS["output_dir"]
    args = context.args or []

    if not args:
        await msg.reply_text(
            "Скачать данные диалога:\n\n"
            "• По номеру из списка /dialogs или по ts:\n"
            "  /dialog_download full 3 — полный текст (без суммаризации)\n"
            "  /dialog_download doc 3 — документ после LLM\n"
            "  /dialog_download both 3 — оба файла\n"
            "  /dialog_download full 20250408_143022\n\n"
            "• Поиск по подстроке (превью, текст файлов, метаданные):\n"
            "  /dialog_download search ключевые слова\n"
            "  /dialog_download search слова --from 2025-01-01 --to 2025-12-31\n"
            "  (границы даты — по полю «дата» в карточке /dialog_register)\n\n"
            "По умолчанию kind=full: /dialog_download 3 то же, что /dialog_download full 3"
        )
        return

    if args[0].lower() in ("search", "find", "filter"):
        q, df, dt = _parse_dialog_download_search_args(args[1:])
        if not q and not df and not dt:
            await msg.reply_text(
                "Укажите текст поиска или границы дат, например:\n"
                "/dialog_download search планирование\n"
                "/dialog_download search --from 2025-04-01 --to 2025-04-30"
            )
            return
        entries = filter_dialogue_entries(
            outp,
            query=q or None,
            date_from=df,
            date_to=dt,
            limit=40,
        )
        if not entries:
            await msg.reply_text("Ничего не найдено по фильтру. Уточните запрос или /dialogs")
            return
        lines = []
        for i, e in enumerate(entries, start=1):
            flag = "в базе" if e["has_meta"] else "нет карточки"
            dd = e.get("dialog_date") or "—"
            hf = "есть полный" if e.get("has_full") else "нет полного"
            pv = e["preview"]
            if len(pv) > 90:
                pv = pv[:90] + "…"
            lines.append(
                f"{i}. {e['ts']}  дата: {dd}  ({flag}, {hf})\n   {pv}"
            )
        tail = (
            "\n\nСкачать: /dialog_download full <ts> или doc / both.\n"
            "Пример: /dialog_download full 20250408_143022"
        )
        text = "Найдено по фильтру:\n\n" + "\n\n".join(lines) + tail
        if len(text) > 4000:
            text = text[:3900] + "\n… (список обрезан)" + tail
        await msg.reply_text(text)
        return

    kind, target = _parse_dialog_download_kind_and_target(args)
    if not target:
        await msg.reply_text(
            "Укажите номер из /dialogs или ts, например:\n/dialog_download full 2"
        )
        return

    ts: str | None = None
    if re.match(r"^\d{8}_\d{6}$", target.strip()):
        ts = target.strip()
    elif target.strip().isdigit():
        docs = list_documents(outp, limit=200)
        n = int(target.strip())
        if 1 <= n <= len(docs):
            ts = docs[n - 1]["ts"]

    if not ts:
        await msg.reply_text(
            "Не найден диалог: укажите номер из /dialogs или ts вида 20250408_143022"
        )
        return

    full_p = dialogue_full_path_for_ts(outp, ts)
    doc_p = document_path_for_ts(outp, ts)

    if kind == "full":
        if not full_p:
            await msg.reply_text(
                f"Нет файла полного диалога для {ts} (dialogue_full_*.txt)."
            )
            return
        await _reply_document_under_limit(msg, full_p, caption=f"Полный текст · {ts}")
        return

    if kind == "doc":
        if not doc_p:
            await msg.reply_text(
                f"Нет обработанного документа для {ts} (document_*.txt)."
            )
            return
        await _reply_document_under_limit(msg, doc_p, caption=f"Документ после LLM · {ts}")
        return

    # both
    if not full_p and not doc_p:
        await msg.reply_text(f"Нет файлов для {ts}.")
        return
    if full_p:
        await _reply_document_under_limit(msg, full_p, caption=f"Полный текст · {ts}")
    else:
        await msg.reply_text(f"Нет dialogue_full_{ts}.txt — отправляю только документ.")
    if doc_p:
        await _reply_document_under_limit(msg, doc_p, caption=f"Документ после LLM · {ts}")
    elif full_p:
        await msg.reply_text(f"Нет document_{ts}.txt.")


async def cmd_dialog_show(update, context):
    """Демонстрация текста диалога и карточки из базы (если есть)."""
    _ensure_paths()
    if not context.args:
        await update.message.reply_text(
            "Укажите номер из списка или ts:\n"
            "/dialog_show 3\n"
            "/dialog_show 20250408_143022\n\n"
            "Список: /dialogs"
        )
        return
    arg = " ".join(context.args).strip()
    docs = list_documents(PATHS["output_dir"], limit=200)
    ts = None
    if arg.isdigit():
        n = int(arg)
        if 1 <= n <= len(docs):
            ts = docs[n - 1]["ts"]
    elif re.match(r"^\d{8}_\d{6}$", arg):
        ts = arg
    outp = PATHS["output_dir"]
    path = document_path_for_ts(outp, ts) if ts else None
    if not path:
        await update.message.reply_text(
            "Документ не найден. Проверьте номер или ts, команда /dialogs"
        )
        return
    meta = get_metadata(ts)
    try:
        body = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        await update.message.reply_text(f"Не удалось прочитать файл: {e}")
        return
    parts = [f"Диалог: {ts}"]
    if meta:
        parts.append(f"Дата (карточка): {meta['dialog_date']}")
        parts.append(f"Участники: {meta['participants']}")
        parts.append(f"Ключевые поинты: {meta['key_points']}")
    else:
        parts.append("Карточка в базе не заполнена — команда /dialog_register")
    header = "\n".join(parts) + "\n\n---\n\n"
    if len(header) + len(body) <= 4000:
        await update.message.reply_text(header + body)
    else:
        await update.message.reply_text(header[:3500] + "…\n\n(текст ниже файлом)")
        with open(path, "rb") as f:
            await update.message.reply_document(document=f, filename=path.name)


async def dialog_register_start(update, context):
    """Начало мастера: список файлов → выбор номера/ts."""
    _ensure_paths()
    docs = list_documents(PATHS["output_dir"])
    if not docs:
        await update.message.reply_text(
            "Нет document_*.txt для привязки. Сначала обработайте аудио."
        )
        return ConversationHandler.END
    context.user_data["dialog_register_list"] = [d["ts"] for d in docs]
    lines = _format_dialog_list_lines(docs)
    await update.message.reply_text(
        "Выберите диалог номером (1, 2, …) или укажите ts вида 20250408_143022:\n\n"
        + "\n\n".join(lines)
        + "\n\nОтмена: /cancel"
    )
    return DIALOG_CHOOSE


async def dialog_register_choose(update, context):
    text = (update.message.text or "").strip()
    outp = PATHS["output_dir"]
    lst = context.user_data.get("dialog_register_list") or []
    ts = None
    if text.isdigit():
        n = int(text)
        if 1 <= n <= len(lst):
            ts = lst[n - 1]
    elif re.match(r"^\d{8}_\d{6}$", text):
        if document_path_for_ts(outp, text):
            ts = text
    if not ts:
        await update.message.reply_text(
            "Не понял номер или ts. Введите число из списка или ts, например 20250408_143022"
        )
        return DIALOG_CHOOSE
    context.user_data["register_ts"] = ts
    await update.message.reply_text(
        "Дата диалога (обязательно). Формат: ГГГГ-ММ-ДД или ДД.ММ.ГГГГ\n"
        "Пример: 2025-04-08"
    )
    return DIALOG_DATE


async def dialog_register_date(update, context):
    raw = (update.message.text or "").strip()
    parsed = parse_dialog_date(raw)
    if not parsed:
        await update.message.reply_text(
            "Нужна корректная дата, например 2025-04-08 или 08.04.2025"
        )
        return DIALOG_DATE
    context.user_data["register_dialog_date"] = parsed
    await update.message.reply_text(
        "Кто участвовал (обязательно). Через запятую, например: Иван, Мария, отдел продаж"
    )
    return DIALOG_PARTS


async def dialog_register_parts(update, context):
    raw = (update.message.text or "").strip()
    if not raw:
        await update.message.reply_text("Поле не может быть пустым. Перечислите участников через запятую.")
        return DIALOG_PARTS
    context.user_data["register_participants"] = raw
    await update.message.reply_text(
        "Ключевые поинты обсуждения (обязательно). Кратко, можно через точку с запятой."
    )
    return DIALOG_KP


async def dialog_register_keypoints(update, context):
    raw = (update.message.text or "").strip()
    if not raw:
        await update.message.reply_text("Поле не может быть пустым. Опишите ключевые поинты.")
        return DIALOG_KP
    ts = context.user_data.get("register_ts")
    dialog_date = context.user_data.get("register_dialog_date")
    participants = context.user_data.get("register_participants")
    if not ts or not dialog_date or not participants:
        await update.message.reply_text("Сессия сброшена. Начните снова: /dialog_register")
        return ConversationHandler.END
    save_metadata(ts, dialog_date, participants, raw)
    for k in (
        "register_ts",
        "register_dialog_date",
        "register_participants",
        "dialog_register_list",
    ):
        context.user_data.pop(k, None)
    await update.message.reply_text(
        f"Сохранено для {ts}:\n"
        f"• дата: {dialog_date}\n"
        f"• участники: {participants}\n"
        f"• поинты: {raw}"
    )
    return ConversationHandler.END


async def dialog_register_cancel(update, context):
    for k in (
        "register_ts",
        "register_dialog_date",
        "register_participants",
        "dialog_register_list",
    ):
        context.user_data.pop(k, None)
    await update.message.reply_text("Отменено.")
    return ConversationHandler.END


async def error_handler(update, context):
    """Глобальный обработчик ошибок (лог + при возможности — ответ в чат)."""
    log.exception("Необработанная ошибка: %s", context.error)
    if update and getattr(update, "effective_message", None):
        try:
            await update.effective_message.reply_text(
                "Внутренняя ошибка. Проверьте логи на сервере бота."
            )
        except Exception:
            pass


def main():
    try:
        from telegram import Update
        from telegram.ext import (
            Application,
            MessageHandler,
            CommandHandler,
            filters,
            ConversationHandler,
        )
        from telegram.error import InvalidToken
    except ImportError:
        print("Установите python-telegram-bot: pip install python-telegram-bot", file=sys.stderr)
        sys.exit(1)

    setup_logging()
    token = get_telegram_bot_token()
    app = (
        Application.builder()
        .token(token)
        .connect_timeout(TELEGRAM_HTTP_CONNECT_TIMEOUT)
        .read_timeout(TELEGRAM_HTTP_READ_TIMEOUT)
        .write_timeout(TELEGRAM_HTTP_WRITE_TIMEOUT)
        .pool_timeout(TELEGRAM_HTTP_POOL_TIMEOUT)
        .get_updates_connect_timeout(TELEGRAM_GET_UPDATES_CONNECT_TIMEOUT)
        .get_updates_read_timeout(TELEGRAM_GET_UPDATES_READ_TIMEOUT)
        .build()
    )

    app.add_handler(CommandHandler("start", cmd_start))
    dialog_conv = ConversationHandler(
        entry_points=[CommandHandler("dialog_register", dialog_register_start)],
        states={
            DIALOG_CHOOSE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, dialog_register_choose),
            ],
            DIALOG_DATE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, dialog_register_date),
            ],
            DIALOG_PARTS: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, dialog_register_parts),
            ],
            DIALOG_KP: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, dialog_register_keypoints),
            ],
        },
        fallbacks=[CommandHandler("cancel", dialog_register_cancel)],
    )
    app.add_handler(dialog_conv)
    app.add_handler(CommandHandler("dialogs", cmd_dialogs))
    app.add_handler(CommandHandler("dialog_download", cmd_dialog_download))
    app.add_handler(CommandHandler("dialog_show", cmd_dialog_show))
    app.add_error_handler(error_handler)
    log.info("Бот запущен")

    # Голос, аудио-файлы (mp3 как музыка), видео, документы
    app.add_handler(
        MessageHandler(
            filters.VOICE
            | filters.AUDIO
            | filters.VIDEO_NOTE
            | filters.VIDEO
            | filters.Document.ALL,
            handle_voice_or_video,
        )
    )
    # Текст без слэша — короткая подсказка
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, cmd_help_text)
    )

    print("Бот запущен. Отправьте голосовое сообщение или .webm документ.")
    try:
        app.run_polling(allowed_updates=Update.ALL_TYPES)
    except InvalidToken:
        log.error("Токен отклонён сервером Telegram")
        print("Ошибка: Токен отклонён. Получите новый у @BotFather и обновите .env / config.yaml", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
