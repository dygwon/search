"""Seams between the pipelines and their backends.

Pipeline code depends only on these, so swapping a backend (in-process
sentence-transformers, TEI, vLLM) never touches ingest or search.
Signatures firm up as the slices that use them land.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SparseVector:
    indices: list[int]
    values: list[float]


@dataclass(frozen=True)
class ScoredChunk:
    chunk_id: str
    score: float


class DenseEmbedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SparseEmbedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]: ...

    def embed_query(self, text: str) -> SparseVector: ...


class Fuser(Protocol):
    def fuse(self, ranked_lists: Mapping[str, Sequence[ScoredChunk]]) -> list[ScoredChunk]:
        """Merge per-representation ranked lists (keyed by name, e.g. "dense") into one."""
        ...


class VectorStore(Protocol):
    """Qdrant behind an interface; methods arrive with index creation (#5)."""
