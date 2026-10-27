"""SQLite checkpointing.

Two properties matter more than the schema itself:

**ASR and translations live in separate tables, keyed by separate fingerprints.**
Transcribing a 52-minute video costs ~3 GPU-minutes; translating it costs ~10.
Keeping them apart means swapping the LLM, editing the glossary or rewording the
prompt re-runs only the cheap half. Every A/B measurement in the acceptance plan
depends on this split.

**Translations are written per batch, not at the end.** A 52-minute job runs for
a quarter of an hour, and closing the tab pauses it (decision Q24). Whatever was
finished before the pause survives, and the next run picks up from there.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .config import DB_PATH
from .models import Sentence, Stage, Translation, TranslationStatus, VideoContext

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    video_id     TEXT PRIMARY KEY,
    title        TEXT NOT NULL DEFAULT '',
    uploader     TEXT NOT NULL DEFAULT '',
    duration_s   REAL NOT NULL DEFAULT 0,
    stage        TEXT NOT NULL DEFAULT 'queued',
    fraction     REAL,
    message      TEXT NOT NULL DEFAULT '',
    error        TEXT,
    summary      TEXT NOT NULL DEFAULT '',
    auto_terms   TEXT NOT NULL DEFAULT '{}',
    context_key  TEXT NOT NULL DEFAULT '',
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS segments (
    video_id    TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    idx         INTEGER NOT NULL,
    start       REAL NOT NULL,
    end         REAL NOT NULL,
    text        TEXT NOT NULL,
    speaker     TEXT,
    PRIMARY KEY (video_id, fingerprint, idx)
);

CREATE TABLE IF NOT EXISTS translations (
    video_id        TEXT NOT NULL,
    asr_fingerprint TEXT NOT NULL,
    fingerprint     TEXT NOT NULL,
    idx             INTEGER NOT NULL,
    text            TEXT NOT NULL,
    status          TEXT NOT NULL,
    PRIMARY KEY (video_id, asr_fingerprint, fingerprint, idx)
);

CREATE TABLE IF NOT EXISTS glossary_cache (
    video_id TEXT NOT NULL,
    source   TEXT NOT NULL,
    target   TEXT NOT NULL,
    PRIMARY KEY (video_id, source)
);

-- Wiki terminology lookups, shared across videos. A NULL result is a cached
-- miss, so a term no wiki knows is not asked about again on every video.
CREATE TABLE IF NOT EXISTS term_lookups (
    api        TEXT NOT NULL,
    title      TEXT NOT NULL,
    result     TEXT,
    fetched_at REAL NOT NULL,
    PRIMARY KEY (api, title)
);
"""

#: Columns added after the first release. SQLite cannot ADD COLUMN IF NOT
#: EXISTS, so each is attempted and "duplicate column" is ignored.
_MIGRATIONS = (
    "ALTER TABLE jobs ADD COLUMN language TEXT",
    "ALTER TABLE jobs ADD COLUMN description TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE jobs ADD COLUMN tags TEXT NOT NULL DEFAULT '[]'",
    "ALTER TABLE jobs ADD COLUMN domain TEXT NOT NULL DEFAULT ''",
)

#: A cached miss is retried after this long; wikis gain pages.
_MISS_TTL_S = 30 * 24 * 3600


class Store:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DB_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            for statement in _MIGRATIONS:
                try:
                    conn.execute(statement)
                except sqlite3.OperationalError as exc:
                    if "duplicate column" not in str(exc):
                        raise

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = sqlite3.connect(self.path, timeout=30)
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                yield conn
                conn.commit()
            finally:
                conn.close()

    # -- jobs ------------------------------------------------------------
    def ensure_job(self, video_id: str, **fields: object) -> None:
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO jobs (video_id, created_at, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(video_id) DO NOTHING",
                (video_id, now, now),
            )
        if fields:
            self.update_job(video_id, **fields)

    def update_job(self, video_id: str, **fields: object) -> None:
        if not fields:
            return
        allowed = {
            "title", "uploader", "duration_s", "stage", "fraction",
            "message", "error", "summary", "auto_terms", "context_key",
            "language", "description", "tags", "domain",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"unknown job field(s): {sorted(unknown)}")
        assignments = ", ".join(f"{k} = ?" for k in fields)
        values = [v.value if isinstance(v, Stage) else v for v in fields.values()]
        with self._connect() as conn:
            conn.execute(
                f"UPDATE jobs SET {assignments}, updated_at = ? WHERE video_id = ?",
                [*values, time.time(), video_id],
            )

    def get_job(self, video_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE video_id = ?", (video_id,)).fetchone()
        return dict(row) if row else None

    def list_jobs(self, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # -- sentences -------------------------------------------------------
    def save_sentences(self, video_id: str, fingerprint: str, sentences: list[Sentence]) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM segments WHERE video_id = ? AND fingerprint = ?",
                (video_id, fingerprint),
            )
            conn.executemany(
                "INSERT INTO segments (video_id, fingerprint, idx, start, end, text, speaker) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (video_id, fingerprint, s.index, s.start, s.end, s.text, s.speaker)
                    for s in sentences
                ],
            )

    def load_sentences(self, video_id: str, fingerprint: str) -> list[Sentence]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT idx, start, end, text, speaker FROM segments "
                "WHERE video_id = ? AND fingerprint = ? ORDER BY idx",
                (video_id, fingerprint),
            ).fetchall()
        return [
            Sentence(index=r["idx"], start=r["start"], end=r["end"], text=r["text"],
                     speaker=r["speaker"])
            for r in rows
        ]

    # -- translations ----------------------------------------------------
    def save_translations(
        self,
        video_id: str,
        asr_fingerprint: str,
        fingerprint: str,
        translations: list[Translation],
    ) -> None:
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO translations "
                "(video_id, asr_fingerprint, fingerprint, idx, text, status) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(video_id, asr_fingerprint, fingerprint, idx) "
                "DO UPDATE SET text = excluded.text, status = excluded.status",
                [
                    (video_id, asr_fingerprint, fingerprint, t.index, t.text, t.status.value)
                    for t in translations
                ],
            )

    def load_translations(
        self, video_id: str, asr_fingerprint: str, fingerprint: str
    ) -> list[Translation]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT idx, text, status FROM translations "
                "WHERE video_id = ? AND asr_fingerprint = ? AND fingerprint = ? ORDER BY idx",
                (video_id, asr_fingerprint, fingerprint),
            ).fetchall()
        return [
            Translation(index=r["idx"], text=r["text"], status=TranslationStatus(r["status"]))
            for r in rows
        ]

    def clear_translations(self, video_id: str) -> int:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM translations WHERE video_id = ?", (video_id,))
            return cur.rowcount

    def clear_video(self, video_id: str) -> None:
        with self._connect() as conn:
            for table in ("segments", "translations", "glossary_cache", "jobs"):
                conn.execute(f"DELETE FROM {table} WHERE video_id = ?", (video_id,))

    # -- context ---------------------------------------------------------
    def save_context(self, video_id: str, context_key: str, context: VideoContext) -> None:
        self.update_job(
            video_id,
            summary=context.summary,
            auto_terms=json.dumps(context.terms, ensure_ascii=False),
            context_key=context_key,
            domain=context.domain,
        )
        with self._connect() as conn:
            conn.execute("DELETE FROM glossary_cache WHERE video_id = ?", (video_id,))
            conn.executemany(
                "INSERT INTO glossary_cache (video_id, source, target) VALUES (?, ?, ?)",
                [(video_id, k, v) for k, v in context.terms.items()],
            )

    def load_context(self, video_id: str, context_key: str) -> VideoContext | None:
        job = self.get_job(video_id)
        if not job or job.get("context_key") != context_key:
            return None
        if not job.get("summary") and job.get("auto_terms") in (None, "", "{}"):
            return None
        try:
            terms = json.loads(job.get("auto_terms") or "{}")
        except json.JSONDecodeError:
            terms = {}
        return VideoContext(
            summary=job.get("summary") or "", terms=terms, domain=job.get("domain") or ""
        )

    # -- terminology lookups ------------------------------------------------
    def get_lookups(self, api: str, titles) -> dict[str, str | None]:
        titles = list(titles)
        if not titles:
            return {}
        now = time.time()
        out: dict[str, str | None] = {}
        with self._connect() as conn:
            for i in range(0, len(titles), 500):
                chunk = titles[i : i + 500]
                marks = ",".join("?" * len(chunk))
                rows = conn.execute(
                    f"SELECT title, result, fetched_at FROM term_lookups "
                    f"WHERE api = ? AND title IN ({marks})",
                    [api, *chunk],
                ).fetchall()
                for r in rows:
                    if r["result"] is None and now - r["fetched_at"] > _MISS_TTL_S:
                        continue
                    out[r["title"]] = r["result"]
        return out

    def save_lookups(self, api: str, results: dict[str, str | None]) -> None:
        now = time.time()
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO term_lookups (api, title, result, fetched_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(api, title) DO UPDATE SET result = excluded.result, "
                "fetched_at = excluded.fetched_at",
                [(api, t, r, now) for t, r in results.items()],
            )

