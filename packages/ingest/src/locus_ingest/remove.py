"""Remove a document from an index: its Qdrant points, chunks, jobs and vectors."""

import sqlite3
from pathlib import Path

from locus_core.indexes import Index
from locus_core.jobs import run_stage_async
from locus_core.qdrant_store import QdrantStore

from locus_ingest.embed import vectors_path

DELETE_STAGE = "delete"


class NotInIndexError(Exception):
    def __init__(self, path: Path, index_name: str):
        super().__init__(f"not in index {index_name!r}")
        self.path = path


async def remove_file(
    conn: sqlite3.Connection, store: QdrantStore, data_dir: Path, index: Index, path: Path
) -> str:
    """Take a path out of the index; returns the doc_hash it pointed at.

    Its documents are purged before its index_members row goes, so rerunning a killed rm
    finds the row and finishes the job.
    """
    # The file may already be gone from disk; only its recorded path matters.
    path = path.resolve()
    row = conn.execute(
        "SELECT doc_hash, replaces FROM index_members WHERE index_id = ? AND source_path = ?",
        (index.index_id, str(path)),
    ).fetchone()
    if row is None:
        raise NotInIndexError(path, index.name)
    doc_hash, replaces = row

    async def delete() -> None:
        # A replace that never finished leaves its last searchable document here too.
        for version in (doc_hash, replaces):
            if version is not None:
                await purge(conn, store, data_dir, index, version, path)
        with conn:
            conn.execute(
                "DELETE FROM index_members WHERE index_id = ? AND source_path = ?",
                (index.index_id, str(path)),
            )

    if used_elsewhere(conn, index, doc_hash, path):
        # The document stays in the index under another path, so it gets no delete job.
        await delete()
    else:
        await run_stage_async(conn, index.index_id, doc_hash, DELETE_STAGE, delete)
    return doc_hash


async def purge(
    conn: sqlite3.Connection,
    store: QdrantStore,
    data_dir: Path,
    index: Index,
    doc_hash: str,
    path: Path,
) -> None:
    """Delete one document from the index as `path` lets go of it, unless another path uses it.

    The caller drops its own reference afterwards. Each step is safe to repeat, so a rerun
    finishes a purge that was killed.
    """
    if used_elsewhere(conn, index, doc_hash, path):
        return
    # Jobs go first: once points or vectors are gone, a "done" stage would make an ingest
    # of these bytes skip work and leave them unsearchable.
    with conn:
        conn.execute(
            "DELETE FROM jobs WHERE index_id = ? AND doc_hash = ? AND stage != ?",
            (index.index_id, doc_hash, DELETE_STAGE),
        )
    await store.delete_doc(index.collection, doc_hash)
    vectors_path(data_dir, index, doc_hash).unlink(missing_ok=True)
    with conn:
        conn.execute(
            "DELETE FROM chunks WHERE index_id = ? AND doc_hash = ?", (index.index_id, doc_hash)
        )


def used_elsewhere(conn: sqlite3.Connection, index: Index, doc_hash: str, path: Path) -> bool:
    """Whether a member other than `path` holds this document, as its current or replaced one."""
    return (
        conn.execute(
            "SELECT 1 FROM index_members WHERE index_id = ? AND source_path != ?"
            " AND (doc_hash = ? OR replaces = ?)",
            (index.index_id, str(path), doc_hash, doc_hash),
        ).fetchone()
        is not None
    )
