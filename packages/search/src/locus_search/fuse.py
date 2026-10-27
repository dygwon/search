"""Fusers: merge per-representation ranked lists into one."""

from collections.abc import Mapping, Sequence

from locus_core.protocols import ScoredChunk

RRF_K = 60


class RRFFuser:
    """Reciprocal rank fusion: each list adds 1 / (k + rank), ranks counted from 1."""

    def __init__(self, k: int = RRF_K, missing: float = 0.0):
        self.k = k
        # What a list contributes for a chunk it didn't return.
        self.missing = missing

    def fuse(self, ranked_lists: Mapping[str, Sequence[ScoredChunk]]) -> list[ScoredChunk]:
        ranks = [
            {hit.chunk_id: rank for rank, hit in enumerate(hits, 1)}
            for hits in ranked_lists.values()
        ]
        chunk_ids = dict.fromkeys(chunk_id for by_id in ranks for chunk_id in by_id)
        scores = {
            chunk_id: sum(
                1 / (self.k + by_id[chunk_id]) if chunk_id in by_id else self.missing
                for by_id in ranks
            )
            for chunk_id in chunk_ids
        }
        # Ties keep first-seen order, so the earlier list breaks them.
        ordered = sorted(chunk_ids, key=lambda chunk_id: -scores[chunk_id])
        return [ScoredChunk(chunk_id, scores[chunk_id]) for chunk_id in ordered]
