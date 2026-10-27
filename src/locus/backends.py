"""The real backends for an index, built from its pinned bundle."""

from locus_core.embedders import FastEmbedBm25, SentenceTransformerEmbedder
from locus_core.indexes import Index
from locus_ingest.parse import DoclingParser
from locus_ingest.pipeline import Backends
from locus_ingest.tokenizer import hf_tokenizer
from locus_search.search import Encoders


def real_backends(index: Index) -> Backends:
    tokenizer, special_tokens = hf_tokenizer(index.bundle.dense, budget=index.chunk_target)
    return Backends(
        parser=DoclingParser(),
        tokenizer=tokenizer,
        dense=SentenceTransformerEmbedder(index.bundle.dense),
        sparse=FastEmbedBm25(),
        special_tokens=special_tokens,
    )


def real_encoders(index: Index) -> Encoders:
    return Encoders(dense=SentenceTransformerEmbedder(index.bundle.dense), sparse=FastEmbedBm25())
