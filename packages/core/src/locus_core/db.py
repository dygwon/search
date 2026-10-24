"""SQLite system of record. Qdrant is derived from it and can be rebuilt."""

import sqlite3
from pathlib import Path

DB_FILENAME = "locus.db"
BUSY_TIMEOUT_S = 5.0

# JSON-valued columns hold TEXT; timestamps are ISO 8601 UTC strings.
SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_hash     TEXT PRIMARY KEY,
    source_path  TEXT NOT NULL,
    title        TEXT,
    title_source TEXT,
    mime_type    TEXT NOT NULL,
    size         INTEGER NOT NULL,
    added_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS parse_cache (
    doc_hash       TEXT NOT NULL REFERENCES documents (doc_hash),
    parser_version TEXT NOT NULL,
    json_path      TEXT NOT NULL,
    PRIMARY KEY (doc_hash, parser_version)
);

CREATE TABLE IF NOT EXISTS indexes (
    index_id        TEXT PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    model_bundle    TEXT NOT NULL,
    chunk_target    INTEGER NOT NULL,
    chunk_ceiling   INTEGER NOT NULL,
    chunker_version TEXT NOT NULL,
    collection      TEXT NOT NULL,
    alias           TEXT NOT NULL,
    status          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS index_members (
    index_id    TEXT NOT NULL REFERENCES indexes (index_id),
    source_path TEXT NOT NULL,
    doc_hash    TEXT NOT NULL REFERENCES documents (doc_hash),
    status      TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (index_id, source_path)
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,
    index_id     TEXT NOT NULL REFERENCES indexes (index_id),
    doc_hash     TEXT NOT NULL REFERENCES documents (doc_hash),
    ordinal      INTEGER NOT NULL,
    char_start   INTEGER NOT NULL,
    char_end     INTEGER NOT NULL,
    page         INTEGER,
    bbox         TEXT,
    modality     TEXT NOT NULL,
    text         TEXT NOT NULL,
    token_counts TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    job_id     TEXT PRIMARY KEY,
    index_id   TEXT NOT NULL REFERENCES indexes (index_id),
    doc_hash   TEXT NOT NULL,
    stage      TEXT NOT NULL,
    status     TEXT NOT NULL,
    attempts   INTEGER NOT NULL DEFAULT 0,
    error      TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_servers (
    server_id TEXT PRIMARY KEY,
    model_id  TEXT NOT NULL,
    revision  TEXT NOT NULL,
    kind      TEXT NOT NULL,
    port      INTEGER,
    pid       INTEGER,
    status    TEXT NOT NULL,
    last_used TEXT
);

CREATE TABLE IF NOT EXISTS query_log (
    query_id    TEXT PRIMARY KEY,
    index_id    TEXT NOT NULL REFERENCES indexes (index_id),
    query_text  TEXT NOT NULL,
    stage_lists TEXT,
    created_at  TEXT NOT NULL
);
"""


def connect(data_dir: Path) -> sqlite3.Connection:
    """Open the Locus database in data_dir, creating the schema if needed."""
    data_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(data_dir / DB_FILENAME, timeout=BUSY_TIMEOUT_S)
    # WAL lets the API read while the ingest worker writes.
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(SCHEMA)
    return conn
