import asyncio

import pytest
from locus_core.db import connect
from locus_core.indexes import create_index
from locus_core.qdrant_store import QdrantStore
from locus_ingest.pipeline import Backends, ingest_file
from locus_search.search import Encoders, search
from qdrant_client import AsyncQdrantClient

from tests.fakes import (
    BGE_SMALL,
    FakeDenseEmbedder,
    FakeParser,
    FakeResolver,
    FakeSparseEmbedder,
    WhitespaceTokenizer,
    docling_doc,
    write_pdf,
)

pytestmark = pytest.mark.anyio

DOCS = {
    "budget": "The finance team approved the budget for the new warehouse.",
    "garden": "Tomatoes and basil grow well together in a sunny garden bed.",
    "travel": "The train to Lisbon leaves at nine and arrives before lunch.",
}


@pytest.fixture
def data_dir(tmp_path):
    return tmp_path / "data"


@pytest.fixture
def conn(data_dir):
    return connect(data_dir)


@pytest.fixture
def store():
    return QdrantStore(AsyncQdrantClient(":memory:"))


@pytest.fixture
def index(conn, store):
    return asyncio.run(create_index(conn, store, FakeResolver(BGE_SMALL), "notes"))


@pytest.fixture
def encoders():
    return Encoders(dense=FakeDenseEmbedder(), sparse=FakeSparseEmbedder())


async def add(conn, store, data_dir, index, path, text, max_tokens=512):
    backends = Backends(
        parser=FakeParser(docling_doc(text=text)),
        tokenizer=WhitespaceTokenizer(max_tokens=max_tokens),
        dense=FakeDenseEmbedder(),
        sparse=FakeSparseEmbedder(),
        special_tokens=0,
    )
    write_pdf(path, text=path.stem)
    return await ingest_file(conn, store, backends, data_dir, index, path)


@pytest.fixture
async def corpus(conn, store, data_dir, index, tmp_path):
    return {
        name: await add(conn, store, data_dir, index, tmp_path / f"{name}.pdf", text)
        for name, text in DOCS.items()
    }


async def test_a_keyword_query_ranks_the_matching_document_first(store, encoders, index, corpus):
    hits = await search(store, encoders, index, "Lisbon")

    assert hits[0].doc_hash == corpus["travel"].doc_hash
    assert hits[0].title == "travel"
    assert "Lisbon" in hits[0].text


async def test_each_hit_carries_fused_and_per_list_scores(store, encoders, index, corpus):
    [top, *_] = await search(store, encoders, index, "warehouse budget")

    assert set(top.scores) == {"dense", "bm25"}
    assert top.score > 0


async def test_a_list_that_missed_the_chunk_has_no_score_for_it(store, encoders, index, corpus):
    hits = await search(store, encoders, index, "basil")

    # Only the garden chunk shares a term with the query.
    missed = [hit for hit in hits if hit.doc_hash != corpus["garden"].doc_hash]
    assert missed and all("bm25" not in hit.scores for hit in missed)


async def test_a_document_appears_once_however_many_chunks_match(
    conn, store, data_dir, index, encoders, tmp_path
):
    long_text = " ".join(f"warehouse{i % 3} costs rose" for i in range(200))
    await add(conn, store, data_dir, index, tmp_path / "long.pdf", long_text, max_tokens=50)
    await add(conn, store, data_dir, index, tmp_path / "short.pdf", "warehouse0 inventory")

    hits = await search(store, encoders, index, "warehouse0 costs")

    doc_hashes = [hit.doc_hash for hit in hits]
    assert len(doc_hashes) == len(set(doc_hashes)) == 2


async def test_limit_caps_the_number_of_documents(store, encoders, index, corpus):
    hits = await search(store, encoders, index, "the", limit=2)

    assert len(hits) == 2


async def test_an_empty_index_returns_no_hits(store, encoders, index):
    assert await search(store, encoders, index, "anything") == []
