"""The ingest pipeline for one file; each stage is recorded in jobs."""

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from locus_core.indexes import Index
from locus_core.jobs import run_stage, run_stage_async, stage_done
from locus_core.protocols import DenseEmbedder, SparseEmbedder
from locus_core.qdrant_store import QdrantStore

from locus_ingest.chunk import write_chunks
from locus_ingest.embed import embed, upsert
from locus_ingest.parse import DocumentParser, parse
from locus_ingest.register import check_pdf, hash_file, register
from locus_ingest.remove import purge_orphans
from locus_ingest.title import resolve_title


@dataclass(frozen=True)
class Backends:
    """What ingesting into one index needs; the tokenizer and embedders follow its bundle."""

    parser: DocumentParser
    tokenizer: BaseTokenizer
    dense: DenseEmbedder
    sparse: SparseEmbedder
    # Tokens the dense model adds around every input, outside the tokenizer's count.
    special_tokens: int


@dataclass(frozen=True)
class Ingested:
    path: Path
    doc_hash: str
    title: str
    # "replaced" when the path held other bytes before, "ingested" when any stage after
    # register ran, "unchanged" when all were already done.
    status: str
    chunks: int


async def ingest_file(
    conn: sqlite3.Connection,
    store: QdrantStore,
    backends: Backends,
    data_dir: Path,
    index: Index,
    path: Path,
) -> Ingested:
    path = path.resolve()
    check_pdf(path)
    doc_hash = hash_file(path)

    def stage(name, fn):
        return run_stage(conn, index.index_id, doc_hash, name, fn)

    async def astage(name, fn):
        return await run_stage_async(conn, index.index_id, doc_hash, name, fn)

    def done(name):
        return stage_done(conn, index.index_id, doc_hash, name)

    previous = conn.execute(
        "SELECT doc_hash FROM index_members WHERE index_id = ? AND source_path = ?",
        (index.index_id, str(path)),
    ).fetchone()

    # Register always runs: it's what adds these bytes at a new path to the index.
    stage("register", lambda: register(conn, index, path, doc_hash))

    [title] = conn.execute("SELECT title FROM documents WHERE doc_hash = ?", (doc_hash,)).fetchone()
    pending = [name for name in ("parse", "chunk", "embed", "upsert") if not done(name)]
    doc = None
    if title is None or "parse" in pending or "chunk" in pending:
        if "parse" in pending:
            doc, _ = stage("parse", lambda: parse(conn, backends.parser, data_dir, doc_hash, path))
        else:
            # Parsed before, so this reads the cache without recording another attempt.
            doc, _ = parse(conn, backends.parser, data_dir, doc_hash, path)

    # Titles are user-editable, so only an unset title is resolved.
    if title is None:
        title, source = resolve_title(path, doc)
        with conn:
            conn.execute(
                "UPDATE documents SET title = ?, title_source = ? WHERE doc_hash = ?",
                (title, source, doc_hash),
            )

    if "chunk" in pending:
        stage(
            "chunk",
            lambda: write_chunks(
                conn, index, backends.tokenizer, doc_hash, doc, backends.special_tokens
            ),
        )
    if "embed" in pending:
        stage(
            "embed",
            lambda: embed(conn, data_dir, index, backends.dense, backends.sparse, doc_hash),
        )
    if "upsert" in pending:
        await astage("upsert", lambda: upsert(conn, store, data_dir, index, doc_hash))
    [chunks] = conn.execute(
        "SELECT count(*) FROM chunks WHERE index_id = ? AND doc_hash = ?",
        (index.index_id, doc_hash),
    ).fetchone()

    with conn:
        conn.execute(
            "UPDATE index_members SET status = 'indexed', updated_at = ?"
            " WHERE index_id = ? AND source_path = ?",
            (datetime.now(UTC).isoformat(), index.index_id, str(path)),
        )
    # The new version is searchable, so the old one can go. Sweeping every orphan, not just
    # `previous`, also finishes a replace that was killed before this point.
    await purge_orphans(conn, store, data_dir, index)

    if previous is not None and previous[0] != doc_hash:
        status = "replaced"
    else:
        status = "ingested" if pending else "unchanged"
    return Ingested(path, doc_hash, title, status, chunks)
