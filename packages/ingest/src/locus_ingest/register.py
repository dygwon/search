"""Register: identify a file by its bytes and add it to an index. Files are read in place."""

import hashlib
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from locus_core.indexes import Index

PDF_MIME = "application/pdf"
PDF_MAGIC = b"%PDF-"


class UnsupportedFileError(Exception):
    def __init__(self, path: Path):
        super().__init__(f"{path} is not a PDF; only PDFs are supported for now")
        self.path = path


def check_pdf(path: Path) -> None:
    with path.open("rb") as f:
        if f.read(len(PDF_MAGIC)) != PDF_MAGIC:
            raise UnsupportedFileError(path)


def hash_file(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def register(
    conn: sqlite3.Connection, index: Index, path: Path, doc_hash: str, replaces: str | None
) -> None:
    """Upsert the documents and index_members rows; the title is resolved after parsing.

    `replaces` is the path's last searchable version, kept until doc_hash is indexed.
    """
    now = datetime.now(UTC).isoformat()
    with conn:
        # The same bytes at another path are the same document; the first path is kept.
        conn.execute(
            "INSERT OR IGNORE INTO documents (doc_hash, source_path, mime_type, size, added_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (doc_hash, str(path), PDF_MIME, path.stat().st_size, now),
        )
        conn.execute(
            "INSERT INTO index_members"
            " (index_id, source_path, doc_hash, status, updated_at, replaces)"
            " VALUES (?, ?, ?, 'registered', ?, ?)"
            " ON CONFLICT (index_id, source_path) DO UPDATE"
            " SET doc_hash = excluded.doc_hash, status = excluded.status,"
            " updated_at = excluded.updated_at, replaces = excluded.replaces",
            (index.index_id, str(path), doc_hash, now, replaces),
        )
        # These bytes are in the index again, so an earlier rm no longer describes them.
        conn.execute(
            "DELETE FROM jobs WHERE index_id = ? AND doc_hash = ? AND stage = 'delete'",
            (index.index_id, doc_hash),
        )
