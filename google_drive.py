#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Публичные ссылки Google Drive.
Groq при url= не следует за редиректами Google (303) — для транскрипции файл скачивается локально.
"""

import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import gdown
import requests
from gdown.exceptions import FileURLRetrievalError

SESSION = requests.Session()
SESSION.headers.setdefault("User-Agent", "PipelineOpt/1.0 (Groq transcription)")


def extract_google_drive_file_id(text: str) -> Optional[str]:
    """Извлекает ID файла из текста сообщения."""
    if not text:
        return None
    patterns = [
        r"https?://(?:drive|docs)\.google\.com/file/d/([a-zA-Z0-9_-]{10,})",
        r"https?://(?:drive|docs)\.google\.com/open\?[^#]*\bid=([a-zA-Z0-9_-]{10,})",
        r"https?://(?:drive|docs)\.google\.com/uc\?[^#]*\bid=([a-zA-Z0-9_-]{10,})",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def extract_google_drive_folder_id(text: str) -> Optional[str]:
    """ID публичной папки: .../drive/folders/ID или .../drive/u/N/folders/ID"""
    if not text:
        return None
    patterns = [
        r"https?://drive\.google\.com/drive/(?:u/\d+/)?folders/([a-zA-Z0-9_-]{10,})",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


AUDIO_EXT_PREFERENCE = (
    ".mp3", ".m4a", ".ogg", ".opus", ".wav", ".webm", ".mp4", ".mpeg", ".flac",
)


def google_drive_folder_first_file_id(folder_id: str) -> str:
    """
    Публичная папка Google Drive → ID первого подходящего файла (gdown парсит HTML).
    """
    import gdown

    files = gdown.download_folder(
        id=folder_id.strip(),
        skip_download=True,
        quiet=True,
    )
    if not files:
        raise ValueError(
            "Не удалось прочитать папку или она пуста. "
            "Нужен доступ «Любой, у кого есть ссылка»."
        )
    for f in files:
        p = (getattr(f, "path", "") or "").lower()
        if any(p.endswith(ext) for ext in AUDIO_EXT_PREFERENCE):
            return f.id
    return files[0].id


def resolve_google_drive_file_id_from_text(text: str) -> Optional[str]:
    """
    Ссылка на файл или на папку → ID файла для скачивания.
    Сначала прямой файл, иначе первая запись в папке.
    """
    fid = extract_google_drive_file_id(text)
    if fid:
        return fid
    fold = extract_google_drive_folder_id(text)
    if fold:
        return google_drive_folder_first_file_id(fold)
    return None


def google_drive_to_direct_download_url(file_id: str) -> str:
    """
    URL для сценариев, где клиент сам следует редиректам (не Groq url=).
    """
    file_id = (file_id or "").strip()
    if not re.match(r"^[a-zA-Z0-9_-]{10,}$", file_id):
        raise ValueError("Некорректный ID файла Google Drive")

    base = f"https://drive.google.com/uc?export=download&id={file_id}"
    r = SESSION.get(base, allow_redirects=True, timeout=120)
    r.raise_for_status()
    ct = (r.headers.get("content-type") or "").lower()

    if "text/html" not in ct:
        return r.url

    m = re.search(r"confirm=([0-9A-Za-z_-]+)", r.text)
    if m:
        u = f"{base}&confirm={m.group(1)}"
    else:
        u = f"{base}&confirm=t"

    r2 = SESSION.get(u, allow_redirects=True, timeout=120)
    r2.raise_for_status()
    ct2 = (r2.headers.get("content-type") or "").lower()
    if "text/html" not in ct2:
        return r2.url
    return u


def google_drive_download_file_to_temp(file_id: str) -> Path:
    """
    Скачивает публичный файл через gdown (как браузер: формы confirm, большие файлы).

    Простой requests+confirm не хватает — Google отдаёт HTML с формой, не только ?confirm=.
    """
    file_id = (file_id or "").strip()
    if not re.match(r"^[a-zA-Z0-9_-]{10,}$", file_id):
        raise ValueError("Некорректный ID файла Google Drive")

    tmpdir = tempfile.mkdtemp(prefix="gdrive_dl_")
    try:
        # output=.../ — имя файла берётся из ответа (Content-Disposition)
        out = gdown.download(
            id=file_id,
            output=tmpdir + os.sep,
            quiet=True,
            verify=True,
        )
        if not out:
            raise ValueError(
                "Не удалось скачать файл. Проверьте доступ «Любой, у кого есть ссылка»."
            )
        path = Path(out)
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError("Скачано 0 байт — проверьте ссылку и доступ.")
        return path
    except FileURLRetrievalError as e:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise ValueError(
            "Google Drive не отдал файл для скачивания. "
            "Проверьте: доступ «Любой, у кого есть ссылка», это файл (не папка), не превышен лимит скачиваний."
        ) from e
    except Exception:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise
