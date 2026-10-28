"""Replace and remove on a real Qdrant. Needs Docker: run with `pytest -m docker`."""

import pytest
from locus_core.db import connect
from locus_core.indexes import create_index
from locus_core.qdrant_store import QdrantStore
from locus_ingest.pipeline import ingest_file
from locus_ingest.remove import remove_file
from qdrant_client import AsyncQdrantClient

from tests.fakes import BGE_SMALL, FakeResolver, write_pdf
from tests.test_ingest import backends, point_count

pytestmark = [pytest.mark.docker, pytest.mark.anyio]


async def test_replace_then_rm_leaves_no_points(qdrant, tmp_path):
    conn = connect(tmp_path / "data")
    store = QdrantStore(AsyncQdrantClient(url="http://127.0.0.1:6333", prefer_grpc=True))
    index = await create_index(conn, store, FakeResolver(BGE_SMALL), "lifecycle")
    pdf = write_pdf(tmp_path / "sample.pdf", text="First draft")
    old = await ingest_file(conn, store, backends(), tmp_path / "data", index, pdf)
    write_pdf(pdf, text="Second draft")

    new = await ingest_file(conn, store, backends(), tmp_path / "data", index, pdf)

    assert await point_count(store, index, old.doc_hash) == 0
    assert await point_count(store, index, new.doc_hash) == new.chunks

    await remove_file(conn, store, tmp_path / "data", index, pdf)

    assert (await store.client.count(index.collection)).count == 0
