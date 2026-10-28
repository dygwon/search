import asyncio

import pytest
from locus_core.db import connect
from locus_core.indexes import create_index
from locus_core.qdrant_store import QdrantStore
from locus_ingest.embed import vectors_path
from locus_ingest.pipeline import ingest_file
from locus_ingest.remove import NotInIndexError, remove_file
from locus_search.search import Encoders, search
from qdrant_client import AsyncQdrantClient

from tests.fakes import (
    BGE_SMALL,
    FakeDenseEmbedder,
    FakeResolver,
    FakeSparseEmbedder,
    write_pdf,
)
from tests.test_ingest import backends, chunk_rows, point_count

pytestmark = pytest.mark.anyio


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
def pdf(tmp_path):
    return write_pdf(tmp_path / "sample.pdf", text="First draft")


def job_rows(conn, doc_hash):
    return conn.execute(
        "SELECT stage, status FROM jobs WHERE doc_hash = ? ORDER BY stage", (doc_hash,)
    ).fetchall()


async def test_reingesting_an_edited_file_replaces_the_old_document(
    conn, store, data_dir, index, pdf
):
    old = await ingest_file(conn, store, backends(), data_dir, index, pdf)
    write_pdf(pdf, text="Second draft")

    new = await ingest_file(conn, store, backends(), data_dir, index, pdf)

    assert new.doc_hash != old.doc_hash
    assert new.status == "replaced"
    assert await point_count(store, index, old.doc_hash) == 0
    assert chunk_rows(conn, old.doc_hash) == []
    assert job_rows(conn, old.doc_hash) == []
    assert not vectors_path(data_dir, index, old.doc_hash).exists()
    assert await point_count(store, index, new.doc_hash) == new.chunks


class SearchingStore(QdrantStore):
    """Searches just before the old document's points are deleted."""

    def __init__(self, client, index):
        super().__init__(client)
        self.index = index
        self.hits = None

    async def delete_doc(self, collection, doc_hash):
        encoders = Encoders(dense=FakeDenseEmbedder(), sparse=FakeSparseEmbedder())
        self.hits = await search(self, encoders, self.index, "word1")
        await super().delete_doc(collection, doc_hash)


async def test_search_finds_the_document_while_it_is_being_replaced(
    conn, store, data_dir, index, pdf
):
    await ingest_file(conn, store, backends(), data_dir, index, pdf)
    write_pdf(pdf, text="Second draft")
    searching = SearchingStore(store.client, index)

    new = await ingest_file(conn, searching, backends(), data_dir, index, pdf)

    assert searching.hits
    assert new.doc_hash in {hit.doc_hash for hit in searching.hits}


class KilledOnDelete(QdrantStore):
    """A store whose first delete dies the way a killed process would."""

    def __init__(self, client):
        super().__init__(client)
        self.killed = False

    async def delete_doc(self, collection, doc_hash):
        if not self.killed:
            self.killed = True
            raise KeyboardInterrupt
        await super().delete_doc(collection, doc_hash)


async def test_a_kill_before_the_old_document_is_deleted_is_cleaned_up_on_rerun(
    conn, store, data_dir, index, pdf
):
    old = await ingest_file(conn, store, backends(), data_dir, index, pdf)
    write_pdf(pdf, text="Second draft")
    killed = KilledOnDelete(store.client)
    with pytest.raises(KeyboardInterrupt):
        await ingest_file(conn, killed, backends(), data_dir, index, pdf)

    rerun = await ingest_file(conn, killed, backends(), data_dir, index, pdf)

    assert rerun.status == "replaced"
    assert await point_count(store, index, old.doc_hash) == 0
    assert chunk_rows(conn, old.doc_hash) == []


async def test_rm_deletes_the_documents_chunks_and_points(conn, store, data_dir, index, pdf):
    ingested = await ingest_file(conn, store, backends(), data_dir, index, pdf)

    removed = await remove_file(conn, store, data_dir, index, pdf)

    assert removed == ingested.doc_hash
    assert chunk_rows(conn, ingested.doc_hash) == []
    assert await point_count(store, index, ingested.doc_hash) == 0
    assert conn.execute("SELECT count(*) FROM index_members").fetchone() == (0,)
    assert job_rows(conn, ingested.doc_hash) == [("delete", "done")]
    assert not vectors_path(data_dir, index, ingested.doc_hash).exists()


async def test_rm_works_after_the_file_is_gone_from_disk(conn, store, data_dir, index, pdf):
    ingested = await ingest_file(conn, store, backends(), data_dir, index, pdf)
    pdf.unlink()

    await remove_file(conn, store, data_dir, index, pdf)

    assert await point_count(store, index, ingested.doc_hash) == 0


async def test_rm_of_a_path_not_in_the_index_fails_clearly(conn, store, data_dir, index, tmp_path):
    with pytest.raises(NotInIndexError, match="not in index 'notes'"):
        await remove_file(conn, store, data_dir, index, tmp_path / "missing.pdf")


async def test_rm_keeps_chunks_another_path_with_the_same_bytes_still_uses(
    conn, store, data_dir, index, pdf, tmp_path
):
    copy = tmp_path / "copy.pdf"
    copy.write_bytes(pdf.read_bytes())
    ingested = await ingest_file(conn, store, backends(), data_dir, index, pdf)
    await ingest_file(conn, store, backends(), data_dir, index, copy)

    await remove_file(conn, store, data_dir, index, pdf)

    assert await point_count(store, index, ingested.doc_hash) == ingested.chunks
    assert len(chunk_rows(conn, ingested.doc_hash)) == ingested.chunks

    await remove_file(conn, store, data_dir, index, copy)

    assert await point_count(store, index, ingested.doc_hash) == 0
    assert chunk_rows(conn, ingested.doc_hash) == []


async def test_a_removed_file_can_be_ingested_again(conn, store, data_dir, index, pdf):
    await ingest_file(conn, store, backends(), data_dir, index, pdf)
    await remove_file(conn, store, data_dir, index, pdf)

    again = await ingest_file(conn, store, backends(), data_dir, index, pdf)

    assert again.status == "ingested"
    assert await point_count(store, index, again.doc_hash) == again.chunks


class DiesAfterDelete(QdrantStore):
    """A store whose first delete goes through, then dies before Locus hears back."""

    def __init__(self, client):
        super().__init__(client)
        self.killed = False

    async def delete_doc(self, collection, doc_hash):
        await super().delete_doc(collection, doc_hash)
        if not self.killed:
            self.killed = True
            raise KeyboardInterrupt


async def test_rerunning_a_killed_rm_finishes_the_cleanup(conn, store, data_dir, index, pdf):
    ingested = await ingest_file(conn, store, backends(), data_dir, index, pdf)
    dies = DiesAfterDelete(store.client)
    with pytest.raises(KeyboardInterrupt):
        await remove_file(conn, dies, data_dir, index, pdf)

    await remove_file(conn, dies, data_dir, index, pdf)

    assert chunk_rows(conn, ingested.doc_hash) == []
    assert await point_count(store, index, ingested.doc_hash) == 0
    assert conn.execute("SELECT count(*) FROM index_members").fetchone() == (0,)
    assert job_rows(conn, ingested.doc_hash) == [("delete", "done")]
    assert not vectors_path(data_dir, index, ingested.doc_hash).exists()


async def test_ingesting_after_a_killed_rm_indexes_the_document_again(
    conn, store, data_dir, index, pdf
):
    await ingest_file(conn, store, backends(), data_dir, index, pdf)
    with pytest.raises(KeyboardInterrupt):
        await remove_file(conn, DiesAfterDelete(store.client), data_dir, index, pdf)

    again = await ingest_file(conn, store, backends(), data_dir, index, pdf)

    assert again.status == "ingested"
    assert again.chunks > 0
    assert await point_count(store, index, again.doc_hash) == again.chunks


def failing_backends():
    return backends(dense=FakeDenseEmbedder(fail_with=RuntimeError("embedder down")))


def member(conn, path):
    return conn.execute(
        "SELECT doc_hash, status, replaces FROM index_members WHERE source_path = ?", (str(path),)
    ).fetchone()


async def test_a_failed_replace_keeps_the_old_document_through_other_ingests(
    conn, store, data_dir, index, pdf, tmp_path
):
    old = await ingest_file(conn, store, backends(), data_dir, index, pdf)
    write_pdf(pdf, text="Second draft")
    with pytest.raises(RuntimeError):
        await ingest_file(conn, store, failing_backends(), data_dir, index, pdf)

    other = write_pdf(tmp_path / "other.pdf", text="Unrelated")
    await ingest_file(conn, store, backends(), data_dir, index, other)

    assert await point_count(store, index, old.doc_hash) == old.chunks
    assert member(conn, pdf)[2] == old.doc_hash

    new = await ingest_file(conn, store, backends(), data_dir, index, pdf)

    assert new.status == "replaced"
    assert await point_count(store, index, old.doc_hash) == 0
    assert chunk_rows(conn, old.doc_hash) == []
    assert member(conn, pdf) == (new.doc_hash, "indexed", None)


async def test_editing_again_after_a_failed_replace_purges_both_older_documents(
    conn, store, data_dir, index, pdf
):
    first = await ingest_file(conn, store, backends(), data_dir, index, pdf)
    write_pdf(pdf, text="Second draft")
    with pytest.raises(RuntimeError):
        await ingest_file(conn, store, failing_backends(), data_dir, index, pdf)
    [failed_hash, _, _] = member(conn, pdf)
    write_pdf(pdf, text="Third draft")

    third = await ingest_file(conn, store, backends(), data_dir, index, pdf)

    for gone in (first.doc_hash, failed_hash):
        assert chunk_rows(conn, gone) == []
        assert await point_count(store, index, gone) == 0
    assert await point_count(store, index, third.doc_hash) == third.chunks


async def test_reverting_a_failed_edit_keeps_the_original(conn, store, data_dir, index, pdf):
    original_bytes = pdf.read_bytes()
    first = await ingest_file(conn, store, backends(), data_dir, index, pdf)
    write_pdf(pdf, text="Second draft")
    with pytest.raises(RuntimeError):
        await ingest_file(conn, store, failing_backends(), data_dir, index, pdf)
    [failed_hash, _, _] = member(conn, pdf)
    pdf.write_bytes(original_bytes)

    again = await ingest_file(conn, store, backends(), data_dir, index, pdf)

    assert again.doc_hash == first.doc_hash
    assert await point_count(store, index, first.doc_hash) == first.chunks
    assert chunk_rows(conn, failed_hash) == []
    assert member(conn, pdf) == (first.doc_hash, "indexed", None)


async def test_rm_after_a_failed_replace_removes_both_documents(conn, store, data_dir, index, pdf):
    old = await ingest_file(conn, store, backends(), data_dir, index, pdf)
    write_pdf(pdf, text="Second draft")
    with pytest.raises(RuntimeError):
        await ingest_file(conn, store, failing_backends(), data_dir, index, pdf)

    await remove_file(conn, store, data_dir, index, pdf)

    assert (await store.client.count(index.collection)).count == 0
    assert conn.execute("SELECT count(*) FROM chunks").fetchone() == (0,)
    assert await point_count(store, index, old.doc_hash) == 0


class FailingDelete(QdrantStore):
    async def delete_doc(self, collection, doc_hash):
        raise RuntimeError("qdrant timed out")


async def test_a_failed_cleanup_does_not_fail_the_ingest_and_is_retried(
    conn, store, data_dir, index, pdf
):
    old = await ingest_file(conn, store, backends(), data_dir, index, pdf)
    write_pdf(pdf, text="Second draft")

    new = await ingest_file(conn, FailingDelete(store.client), backends(), data_dir, index, pdf)

    assert new.status == "replaced"
    assert await point_count(store, index, new.doc_hash) == new.chunks
    assert member(conn, pdf) == (new.doc_hash, "indexed", old.doc_hash)

    await ingest_file(conn, store, backends(), data_dir, index, pdf)

    assert await point_count(store, index, old.doc_hash) == 0
    assert member(conn, pdf)[2] is None


async def test_rm_of_one_of_two_paths_records_no_delete_job(
    conn, store, data_dir, index, pdf, tmp_path
):
    copy = tmp_path / "copy.pdf"
    copy.write_bytes(pdf.read_bytes())
    ingested = await ingest_file(conn, store, backends(), data_dir, index, pdf)
    await ingest_file(conn, store, backends(), data_dir, index, copy)

    await remove_file(conn, store, data_dir, index, pdf)

    assert ("delete", "done") not in job_rows(conn, ingested.doc_hash)


async def test_a_readded_file_has_no_delete_job(conn, store, data_dir, index, pdf):
    await ingest_file(conn, store, backends(), data_dir, index, pdf)
    await remove_file(conn, store, data_dir, index, pdf)

    again = await ingest_file(conn, store, backends(), data_dir, index, pdf)

    assert ("delete", "done") not in job_rows(conn, again.doc_hash)
