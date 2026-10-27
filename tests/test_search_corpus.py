"""Search the sample corpus in tests/fixtures with the real models end to end."""

from pathlib import Path

import pytest
from locus_core.db import connect
from locus_core.indexes import create_index
from locus_core.qdrant_store import QdrantStore
from locus_ingest.pipeline import ingest_file
from locus_search.search import search
from qdrant_client import AsyncQdrantClient

from locus.backends import real_backends, real_encoders
from tests.fakes import BGE_SMALL, FakeResolver

# Docling, bge and BM25 download from the Hugging Face Hub on first use.
pytestmark = [pytest.mark.network, pytest.mark.anyio]

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="module")
async def corpus(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("data")
    conn = connect(data_dir)
    store = QdrantStore(AsyncQdrantClient(":memory:"))
    index = await create_index(conn, store, FakeResolver(BGE_SMALL), "notes")
    backends = real_backends(index)
    titles = {}
    for pdf in sorted(FIXTURES.glob("*.pdf")):
        result = await ingest_file(conn, store, backends, data_dir, index, pdf)
        titles[pdf.name] = result.title
    return store, index, titles


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        # Exact keywords, where BM25 carries the match.
        ("banneton", "sourdough-baking.pdf"),
        ("W-2 1099", "tax-filing.pdf"),
        # Paraphrases sharing no content words with the document, where dense carries it.
        ("looking after my road bike", "bicycle-maintenance.pdf"),
        ("getting ready to send my yearly paperwork to the IRS", "tax-filing.pdf"),
    ],
)
async def test_the_right_document_ranks_first(corpus, query, expected):
    store, index, titles = corpus

    hits = await search(store, real_encoders(index), index, query)

    assert hits[0].title == titles[expected]
    assert len({hit.doc_hash for hit in hits}) == len(hits) == len(titles)


async def test_hits_carry_the_page_from_docling(corpus):
    store, index, _ = corpus

    [top, *_] = await search(store, real_encoders(index), index, "monstera")

    assert top.page == 1
