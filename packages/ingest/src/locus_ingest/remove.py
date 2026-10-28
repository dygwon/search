"""Remove a document version from an index: its Qdrant points, chunks, jobs and vectors."""

import sqlite3
from pathlib import Path

from locus_core.indexes import Index
from locus_core.jobs import run_stage_async
from locus_core.qdrant_store import QdrantStore

from locus_ingest.embed import vectors_path

DELETE_STAGE = "delete"


class NotInIndexError(Exception):
    def __init__(self, path: Path, index_name: str):
        super().__init__(f"{path} is not in index {index_name!r}")
        self.path = path


async def remove_file(
    conn: sqlite3.Connection, store: QdrantStore, data_dir: Path, index: Index, path: Path
) -> str:
    """Take a path out of the index, in one job; returns the doc_hash it pointed at."""
    # The file may already be gone from disk; only its recorded path matters.
    path = path.resolve()
    row = conn.execute(
        "SELECT doc_hash FROM index_members WHERE index_id = ? AND source_path = ?",
        (index.index_id, str(path)),
    ).fetchone()
    if row is None:
        raise NotInIndexError(path, index.name)
    [doc_hash] = row

    await run_stage_async(
        conn,
        index.index_id,
        doc_hash,
        DELETE_STAGE,
        lambda: purge(conn, store, data_dir, index, doc_hash, removing=path),
    )
    return doc_hash


async def purge_orphans(
    conn: sqlite3.Connection, store: QdrantStore, data_dir: Path, index: Index
) -> None:
    """Purge every version that still has chunks but no member, such as a replaced one."""
    orphans = conn.execute(
        "SELECT DISTINCT doc_hash FROM chunks WHERE index_id = ?"
        " EXCEPT SELECT doc_hash FROM index_members WHERE index_id = ?",
        (index.index_id, index.index_id),
    ).fetchall()
    for (doc_hash,) in orphans:
        await purge(conn, store, data_dir, index, doc_hash)


async def purge(
    conn: sqlite3.Connection,
    store: QdrantStore,
    data_dir: Path,
    index: Index,
    doc_hash: str,
    removing: Path | None = None,
) -> None:
    """Delete one version from the index unless another path still has the same bytes.

    `removing` is the member being taken out. Its row goes in the last transaction, so a
    killed run leaves it in place and rerunning rm finishes the job.
    """
    removing_path = str(removing) if removing is not None else None
    shared = conn.execute(
        "SELECT 1 FROM index_members WHERE index_id = ? AND doc_hash = ? AND source_path IS NOT ?",
        (index.index_id, doc_hash, removing_path),
    ).fetchone()
    # SQLite goes last: while its rows remain, a rerun can still find this version.
    if not shared:
        await store.delete_doc(index.collection, doc_hash)
        vectors_path(data_dir, index, doc_hash).unlink(missing_ok=True)
    with conn:
        if not shared:
            conn.execute(
                "DELETE FROM chunks WHERE index_id = ? AND doc_hash = ?",
                (index.index_id, doc_hash),
            )
            # Stale "done" stages would skip the upsert if these bytes are added again.
            conn.execute(
                "DELETE FROM jobs WHERE index_id = ? AND doc_hash = ? AND stage != ?",
                (index.index_id, doc_hash, DELETE_STAGE),
            )
        conn.execute(
            "DELETE FROM index_members WHERE index_id = ? AND source_path = ?",
            (index.index_id, removing_path),
        )
