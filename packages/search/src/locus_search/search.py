"""The search pipeline for one index: encode, retrieve, fuse, group."""

from dataclasses import dataclass

from locus_core.indexes import Index
from locus_core.protocols import DenseEmbedder, Fuser, ScoredChunk, SparseEmbedder
from locus_core.qdrant_store import QdrantStore
from qdrant_client import models

from locus_search.fuse import RRFFuser

# Each list's depth bounds what fusion can recover.
CANDIDATES = 100
DEFAULT_LIMIT = 10


@dataclass(frozen=True)
class Encoders:
    """What searching one index needs; the embedders follow its bundle."""

    dense: DenseEmbedder
    sparse: SparseEmbedder


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    doc_hash: str
    title: str
    page: int | None
    text: str
    # The fused score, then each list's raw score; a list that missed the chunk has no entry.
    score: float
    scores: dict[str, float]


async def search(
    store: QdrantStore,
    encoders: Encoders,
    index: Index,
    query: str,
    limit: int = DEFAULT_LIMIT,
    candidates: int = CANDIDATES,
    fuser: Fuser | None = None,
) -> list[Hit]:
    """The best chunk from each of the top `limit` documents, best first."""
    bm25 = encoders.sparse.embed_query(query)
    lists = await store.query(
        index.collection,
        {
            "dense": encoders.dense.embed_query(query),
            "bm25": models.SparseVector(indices=bm25.indices, values=bm25.values),
        },
        candidates,
    )
    payloads = {
        point.payload["chunk_id"]: point.payload for points in lists.values() for point in points
    }
    ranked = {
        name: [ScoredChunk(point.payload["chunk_id"], point.score) for point in points]
        for name, points in lists.items()
    }
    raw = {name: {hit.chunk_id: hit.score for hit in hits} for name, hits in ranked.items()}

    hits: list[Hit] = []
    seen: set[str] = set()
    for fused in (fuser or RRFFuser()).fuse(ranked):
        payload = payloads[fused.chunk_id]
        # One document can't fill the page: only its best chunk is shown.
        if payload["doc_hash"] in seen:
            continue
        seen.add(payload["doc_hash"])
        hits.append(
            Hit(
                chunk_id=fused.chunk_id,
                doc_hash=payload["doc_hash"],
                title=payload["title"],
                page=payload["page"],
                text=payload["text"],
                score=fused.score,
                scores={
                    name: by_id[fused.chunk_id]
                    for name, by_id in raw.items()
                    if fused.chunk_id in by_id
                },
            )
        )
        if len(hits) == limit:
            break
    return hits
