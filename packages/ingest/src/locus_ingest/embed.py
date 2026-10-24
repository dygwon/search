"""Embed and upsert: encode an index's chunks with its bundle and write them to Qdrant."""

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from locus_core.indexes import Index
from locus_core.protocols import DenseEmbedder, SparseEmbedder, SparseVector
from locus_core.qdrant_store import QdrantStore
from qdrant_client import models

EMBED_BATCH = 32
VECTORS_DIR = "vectors"


@dataclass(frozen=True)
class ChunkVectors:
    chunk_id: str
    dense: list[float]
    bm25: SparseVector


def vectors_path(data_dir: Path, index: Index, doc_hash: str) -> Path:
    return data_dir / VECTORS_DIR / index.index_id / f"{doc_hash}.npz"


def embed(
    conn: sqlite3.Connection,
    data_dir: Path,
    index: Index,
    dense: DenseEmbedder,
    sparse: SparseEmbedder,
    doc_hash: str,
) -> None:
    """Encode the document's chunks and save the vectors, so upsert can resume without them."""
    rows = conn.execute(
        "SELECT chunk_id, text, token_counts FROM chunks WHERE index_id = ? AND doc_hash = ?",
        (index.index_id, doc_hash),
    ).fetchall()
    # Similar lengths in a batch waste less padding.
    rows.sort(key=lambda row: json.loads(row[2])[index.bundle.dense.repo_id])
    vectors = []
    for start in range(0, len(rows), EMBED_BATCH):
        batch = rows[start : start + EMBED_BATCH]
        texts = [text for _, text, _ in batch]
        for (chunk_id, _, _), d, s in zip(
            batch, dense.embed_documents(texts), sparse.embed_documents(texts), strict=True
        ):
            vectors.append(ChunkVectors(chunk_id, d, s))
    _save(vectors_path(data_dir, index, doc_hash), vectors)


async def upsert(
    conn: sqlite3.Connection, store: QdrantStore, data_dir: Path, index: Index, doc_hash: str
) -> None:
    vectors = _load(vectors_path(data_dir, index, doc_hash))
    payloads = {row[0]: row for row in _payload_rows(conn, index, [v.chunk_id for v in vectors])}
    points = [
        models.PointStruct(
            id=v.chunk_id,
            vector={
                "dense": v.dense,
                "bm25": models.SparseVector(indices=v.bm25.indices, values=v.bm25.values),
            },
            payload=_payload(payloads[v.chunk_id]),
        )
        for v in vectors
    ]
    await store.upsert(index.collection, points)


def _payload_rows(conn: sqlite3.Connection, index: Index, chunk_ids: list[str]) -> list[tuple]:
    placeholders = ", ".join("?" * len(chunk_ids))
    return conn.execute(
        "SELECT c.chunk_id, c.doc_hash, d.title, c.page, c.bbox, c.modality, c.char_start,"
        " c.char_end, c.text, d.added_at FROM chunks c JOIN documents d USING (doc_hash)"
        f" WHERE c.index_id = ? AND c.chunk_id IN ({placeholders})",
        (index.index_id, *chunk_ids),
    ).fetchall()


def _payload(row: tuple) -> dict:
    chunk_id, doc_hash, title, page, bbox, modality, char_start, char_end, text, added_at = row
    return {
        "chunk_id": chunk_id,
        "doc_hash": doc_hash,
        "title": title,
        "page": page,
        "bbox": json.loads(bbox) if bbox else None,
        "modality": modality,
        "char_start": char_start,
        "char_end": char_end,
        "text": text,
        "added_at": added_at,
    }


def _save(target: Path, vectors: list[ChunkVectors]) -> None:
    # Sparse vectors are ragged, so they're stored flat with an offset per chunk.
    offsets = np.cumsum([0] + [len(v.bm25.indices) for v in vectors])
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".partial.npz")
    np.savez(
        partial,
        chunk_ids=np.array([v.chunk_id for v in vectors]),
        dense=np.array([v.dense for v in vectors], dtype=np.float32),
        bm25_indices=np.array([i for v in vectors for i in v.bm25.indices], dtype=np.uint32),
        bm25_values=np.array([x for v in vectors for x in v.bm25.values], dtype=np.float32),
        bm25_offsets=offsets,
    )
    partial.replace(target)


def _load(path: Path) -> list[ChunkVectors]:
    with np.load(path) as data:
        offsets = data["bm25_offsets"]
        return [
            ChunkVectors(
                chunk_id=str(chunk_id),
                dense=data["dense"][i].tolist(),
                bm25=SparseVector(
                    indices=data["bm25_indices"][offsets[i] : offsets[i + 1]].tolist(),
                    values=data["bm25_values"][offsets[i] : offsets[i + 1]].tolist(),
                ),
            )
            for i, chunk_id in enumerate(data["chunk_ids"])
        ]
