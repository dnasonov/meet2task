#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SQLite-реестр метаданных диалогов (дата, участники, ключевые поинты), привязка к ts из имён document_*.txt.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime
from pathlib import Path

from .config import get_project_root

_TS_RE = re.compile(r"^document_(\d{8}_\d{6})\.txt$")


def get_db_path() -> Path:
    d = get_project_root() / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d / "dialogues.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dialog_metadata (
                ts TEXT PRIMARY KEY,
                dialog_date TEXT NOT NULL,
                participants TEXT NOT NULL,
                key_points TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def parse_dialog_date(text: str) -> str | None:
    t = (text or "").strip()
    if not t:
        return None
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(t, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def list_documents(output_dir: Path, limit: int = 25) -> list[dict]:
    init_db()
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return []
    rows: list[tuple[Path, float]] = []
    for p in output_dir.glob("document_*.txt"):
        m = _TS_RE.match(p.name)
        if not m:
            continue
        try:
            rows.append((p, p.stat().st_mtime))
        except OSError:
            continue
    rows.sort(key=lambda x: x[1], reverse=True)
    out: list[dict] = []
    with _connect() as conn:
        for p, _mt in rows[:limit]:
            ts = _TS_RE.match(p.name).group(1)
            cur = conn.execute(
                "SELECT 1 FROM dialog_metadata WHERE ts = ? LIMIT 1", (ts,)
            )
            has_meta = cur.fetchone() is not None
            preview = ""
            try:
                raw = p.read_text(encoding="utf-8", errors="replace")
                preview = raw.strip().replace("\n", " ")[:120]
            except OSError:
                preview = ""
            out.append(
                {
                    "ts": ts,
                    "path": p,
                    "has_meta": has_meta,
                    "preview": preview,
                }
            )
    return out


def get_metadata(ts: str) -> dict | None:
    init_db()
    with _connect() as conn:
        cur = conn.execute(
            "SELECT ts, dialog_date, participants, key_points, updated_at "
            "FROM dialog_metadata WHERE ts = ?",
            (ts.strip(),),
        )
        row = cur.fetchone()
        if not row:
            return None
        return dict(row)


def save_metadata(ts: str, dialog_date: str, participants: str, key_points: str) -> None:
    init_db()
    ts = ts.strip()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO dialog_metadata (ts, dialog_date, participants, key_points, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(ts) DO UPDATE SET
                dialog_date = excluded.dialog_date,
                participants = excluded.participants,
                key_points = excluded.key_points,
                updated_at = excluded.updated_at
            """,
            (ts, dialog_date.strip(), participants.strip(), key_points.strip(), now),
        )
        conn.commit()


def document_path_for_ts(output_dir: Path, ts: str) -> Path | None:
    p = Path(output_dir) / f"document_{ts}.txt"
    return p if p.is_file() else None


def dialogue_full_path_for_ts(output_dir: Path, ts: str) -> Path | None:
    p = Path(output_dir) / f"dialogue_full_{ts}.txt"
    return p if p.is_file() else None


def _read_head(path: Path, max_chars: int = 12000) -> str:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if len(raw) <= max_chars:
        return raw
    return raw[:max_chars]


def get_dialogue_paths_for_ts(output_dir: Path, ts: str) -> dict[str, Path | None]:
    outp = Path(output_dir)
    return {
        "ts": ts,
        "full": dialogue_full_path_for_ts(outp, ts),
        "document": document_path_for_ts(outp, ts),
    }


def filter_dialogue_entries(
    output_dir: Path,
    *,
    query: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    has_meta_only: bool = False,
    limit: int = 50,
) -> list[dict]:
    init_db()
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return []

    rows: list[tuple[Path, float]] = []
    for p in output_dir.glob("document_*.txt"):
        m = _TS_RE.match(p.name)
        if not m:
            continue
        try:
            rows.append((p, p.stat().st_mtime))
        except OSError:
            continue
    rows.sort(key=lambda x: x[1], reverse=True)

    q = (query or "").strip().lower()
    df = (date_from or "").strip() or None
    dt = (date_to or "").strip() or None
    date_filter = bool(df or dt)

    out: list[dict] = []
    with _connect() as conn:
        for p, _mt in rows:
            ts = _TS_RE.match(p.name).group(1)
            cur = conn.execute(
                "SELECT dialog_date, participants, key_points FROM dialog_metadata WHERE ts = ?",
                (ts,),
            )
            mrow = cur.fetchone()
            has_meta = mrow is not None
            if has_meta_only and not has_meta:
                continue
            dialog_date = (mrow["dialog_date"] if mrow else None) or ""
            participants = (mrow["participants"] if mrow else "") or ""
            key_points = (mrow["key_points"] if mrow else "") or ""

            if date_filter:
                if not dialog_date:
                    continue
                if df and dialog_date < df:
                    continue
                if dt and dialog_date > dt:
                    continue

            if q:
                blob = f"{ts} {dialog_date} {participants} {key_points}".lower()
                blob += " " + _read_head(p).lower()
                fp = dialogue_full_path_for_ts(output_dir, ts)
                if fp:
                    blob += " " + _read_head(fp).lower()
                if q not in blob:
                    continue

            preview = ""
            try:
                raw = p.read_text(encoding="utf-8", errors="replace")
                preview = raw.strip().replace("\n", " ")[:120]
            except OSError:
                preview = ""

            out.append(
                {
                    "ts": ts,
                    "path": p,
                    "has_meta": has_meta,
                    "dialog_date": dialog_date or None,
                    "preview": preview,
                    "has_full": dialogue_full_path_for_ts(output_dir, ts) is not None,
                }
            )
            if len(out) >= limit:
                break

    return out
