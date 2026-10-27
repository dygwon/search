"""In-process embedders. TEI arrives later as another adapter behind the same protocols."""

from collections.abc import Sequence

from locus_core.models import ResolvedModel
from locus_core.protocols import SparseVector

# bge models expect this prefix on queries, not on passages.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class SentenceTransformerEmbedder:
    """A dense model loaded at its pinned revision, on Metal when available."""

    def __init__(self, model: ResolvedModel):
        self.model = model
        self._st = None

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(list(texts))

    def embed_query(self, text: str) -> list[float]:
        prefix = BGE_QUERY_INSTRUCTION if self.model.repo_id.startswith("BAAI/bge-") else ""
        [vector] = self._encode([prefix + text])
        return vector

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if self._st is None:
            # torch is slow to import, so wait until something needs encoding.
            import torch
            from sentence_transformers import SentenceTransformer

            device = "mps" if torch.backends.mps.is_available() else "cpu"
            self._st = SentenceTransformer(
                self.model.repo_id, revision=self.model.revision, device=device
            )
        return self._st.encode(texts, normalize_embeddings=True).tolist()


class FastEmbedBm25:
    """BM25 term frequencies; Qdrant applies IDF at query time."""

    MODEL = "Qdrant/bm25"

    def __init__(self) -> None:
        self._model = None

    def embed_documents(self, texts: Sequence[str]) -> list[SparseVector]:
        return [_sparse(e) for e in self._load().embed(list(texts))]

    def embed_query(self, text: str) -> SparseVector:
        [embedding] = self._load().query_embed(text)
        return _sparse(embedding)

    def _load(self):
        if self._model is None:
            from fastembed import SparseTextEmbedding

            self._model = SparseTextEmbedding(self.MODEL)
        return self._model


def _sparse(embedding) -> SparseVector:
    return SparseVector(
        indices=embedding.indices.tolist(), values=embedding.values.astype(float).tolist()
    )
