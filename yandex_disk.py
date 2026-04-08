#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Публичные ссылки Яндекс.Диска → прямая ссылка на скачивание (для Groq по URL).
Документация API: https://yandex.ru/dev/disk/api/reference/public.html
"""

import os
import re
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse

import requests

# Аудио/видео для Whisper
AUDIO_EXT = (".mp3", ".m4a", ".ogg", ".opus", ".wav", ".flac", ".webm", ".mp4", ".mpeg")

YANDEX_URL_RE = re.compile(
    r"https?://(?:disk\.yandex\.(?:ru|com)|yadi\.sk)/[^\s<>\"']+",
    re.IGNORECASE,
)

API_BASE = "https://cloud-api.yandex.net/v1/disk/public/resources"
SESSION = requests.Session()
SESSION.headers.setdefault("User-Agent", "PipelineOpt/1.0")


def extract_yandex_public_url(text: str) -> Optional[str]:
    """Первая ссылка на публичный ресурс Яндекс.Диска в тексте."""
    m = YANDEX_URL_RE.search(text or "")
    if not m:
        return None
    url = m.group(0).rstrip(".,;:)\"'")
    return url


def _download_href(public_key: str, path: str = "") -> str:
    """GET .../download → поле href."""
    params = {"public_key": public_key}
    if path:
        params["path"] = path
    r = SESSION.get(f"{API_BASE}/download", params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    href = data.get("href")
    if not href:
        raise ValueError("Яндекс.Диск: в ответе нет ссылки на скачивание")
    return href


def _list_public_folder(public_key: str, path: str = "/") -> list[dict]:
    r = SESSION.get(
        API_BASE,
        params={"public_key": public_key, "path": path, "limit": 200},
        timeout=60,
    )
    r.raise_for_status()
    return r.json().get("_embedded", {}).get("items", []) or []


def yandex_public_to_direct_download_url(public_url: str) -> str:
    """
    Превращает публичную ссылку Яндекс.Диска во временную прямую ссылку (href),
    которую можно передать в Groq Whisper как url=.

    Поддерживается публикация одного файла или папки (берётся первый подходящий аудио/видео файл).
    """
    public_key = (public_url or "").strip()
    if not public_key.startswith("http"):
        raise ValueError("Нужна полная ссылка https://disk.yandex.ru/...")

    # 1) Прямая попытка (один опубликованный файл)
    try:
        return _download_href(public_key)
    except requests.HTTPError as e:
        code = e.response.status_code if e.response is not None else 0
        if code not in (400, 404):
            raise
    # 2) Папка: ищем первый аудио/видео файл
    items = _list_public_folder(public_key, "/")
    candidates: list[tuple[str, str]] = []
    for it in items:
        name = (it.get("name") or "").lower()
        p = it.get("path") or ""
        if it.get("type") == "file" and any(name.endswith(ext) for ext in AUDIO_EXT):
            candidates.append((name, p))
    if not candidates:
        raise ValueError(
            "В публичной папке не найден подходящий аудио/видео файл "
            f"({', '.join(AUDIO_EXT)}). Загрузите один файл или укажите папку с одним треком."
        )
    candidates.sort(key=lambda x: x[0])
    _, path = candidates[0]
    return _download_href(public_key, path)


def _suffix_from_download(r: requests.Response, href: str) -> str:
    """Расширение временного файла по Content-Disposition / URL / Content-Type."""
    disp = r.headers.get("content-disposition") or ""
    for pat in (
        r"filename\*=UTF-8''([^;\s]+)",
        r'filename="([^"]+)"',
        r"filename=([^;\s]+)",
    ):
        m = re.search(pat, disp, re.I)
        if m:
            name = unquote(m.group(1).strip().strip("'"))
            if "." in name:
                return "." + name.rsplit(".", 1)[-1].lower()[:12]
    path_part = urlparse(href).path
    if "." in path_part:
        return "." + path_part.rsplit(".", 1)[-1].lower()[:12]
    ct = (r.headers.get("content-type") or "").lower()
    if "webm" in ct:
        return ".webm"
    if "mpeg" in ct or "mp4" in ct:
        return ".mp4"
    if "audio" in ct or "mp3" in ct:
        return ".mp3"
    return ".bin"


def yandex_public_download_to_temp(public_url: str) -> Path:
    """
    Скачивает публичный файл во временный путь.

    Ссылку href от API нельзя отдавать в Groq url= — downloader.disk.yandex.ru часто отвечает 302,
    а сервер Groq редиректы не обрабатывает. Локальное скачивание с allow_redirects=True решает это.
    """
    href = yandex_public_to_direct_download_url(public_url)
    path: Optional[Path] = None
    r = SESSION.get(href, timeout=600, stream=True, allow_redirects=True)
    try:
        r.raise_for_status()
        ct = (r.headers.get("content-type") or "").lower()
        if "text/html" in ct:
            raise ValueError(
                "Яндекс.Диск отдал страницу вместо файла. Проверьте доступ «Всем по ссылке»."
            )
        suffix = _suffix_from_download(r, href)
        fd, raw = tempfile.mkstemp(prefix="yandex_", suffix=suffix)
        os.close(fd)
        path = Path(raw)
        with open(path, "wb") as out:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    out.write(chunk)
    finally:
        r.close()

    if path is None or path.stat().st_size == 0:
        if path is not None:
            path.unlink(missing_ok=True)
        raise ValueError("Скачано 0 байт — проверьте ссылку и доступ.")
    return path
