"""The ingest pipeline for one file; each stage is recorded in jobs."""

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from locus_core.indexes import Index
from locus_core.jobs import run_stage

from locus_ingest.parse import DocumentParser, parse
from locus_ingest.register import check_pdf, hash_file, register
from locus_ingest.title import resolve_title


@dataclass(frozen=True)
class Ingested:
    path: Path
    doc_hash: str
    title: str
    cached: bool


def ingest_file(
    conn: sqlite3.Connection, parser: DocumentParser, data_dir: Path, index: Index, path: Path
) -> Ingested:
    path = path.resolve()
    check_pdf(path)
    doc_hash = hash_file(path)

    def stage(name, fn):
        return run_stage(conn, index.index_id, doc_hash, name, fn)

    stage("register", lambda: register(conn, index, path, doc_hash))
    doc, cached = stage("parse", lambda: parse(conn, parser, data_dir, doc_hash, path))

    [title] = conn.execute("SELECT title FROM documents WHERE doc_hash = ?", (doc_hash,)).fetchone()
    with conn:
        # Titles are user-editable, so only an unset title is resolved.
        if title is None:
            title, source = resolve_title(path, doc)
            conn.execute(
                "UPDATE documents SET title = ?, title_source = ? WHERE doc_hash = ?",
                (title, source, doc_hash),
            )
        conn.execute(
            "UPDATE index_members SET status = 'parsed', updated_at = ?"
            " WHERE index_id = ? AND source_path = ?",
            (datetime.now(UTC).isoformat(), index.index_id, str(path)),
        )
    return Ingested(path, doc_hash, title, cached)
