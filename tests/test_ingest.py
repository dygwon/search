import asyncio
import hashlib

import pytest
from locus_core.db import connect
from locus_core.indexes import create_index
from locus_core.qdrant_store import QdrantStore
from locus_ingest.pipeline import Backends, ingest_file
from locus_ingest.register import UnsupportedFileError
from qdrant_client import AsyncQdrantClient, models

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

LONG_TEXT = " ".join(f"word{i}" for i in range(300))


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
    return write_pdf(tmp_path / "sample.pdf", title="Quarterly Notes")


def backends(parser=None, dense=None):
    return Backends(
        parser=parser or FakeParser(docling_doc(heading="Intro", text=LONG_TEXT)),
        tokenizer=WhitespaceTokenizer(max_tokens=100),
        dense=dense or FakeDenseEmbedder(),
        sparse=FakeSparseEmbedder(),
    )


def jobs(conn):
    return conn.execute("SELECT stage, status, attempts, error FROM jobs ORDER BY stage").fetchall()


async def point_count(store, index, doc_hash):
    result = await store.client.count(
        index.collection,
        count_filter=models.Filter(
            must=[models.FieldCondition(key="doc_hash", match=models.MatchValue(value=doc_hash))]
        ),
    )
    return result.count


def chunk_rows(conn, doc_hash):
    return conn.execute(
        "SELECT chunk_id FROM chunks WHERE doc_hash = ? ORDER BY ordinal", (doc_hash,)
    ).fetchall()


async def test_ingest_upserts_one_point_per_chunk(conn, store, data_dir, index, pdf):
    result = await ingest_file(conn, store, backends(), data_dir, index, pdf)

    chunks = chunk_rows(conn, result.doc_hash)
    assert len(chunks) > 1
    assert result.chunks == len(chunks)
    assert await point_count(store, index, result.doc_hash) == len(chunks)


async def test_a_fresh_index_with_the_same_config_gets_identical_chunk_ids(
    conn, store, data_dir, index, pdf
):
    other = await create_index(conn, store, FakeResolver(BGE_SMALL), "notes-copy")

    first = await ingest_file(conn, store, backends(), data_dir, index, pdf)
    await ingest_file(conn, store, backends(), data_dir, other, pdf)

    rows = conn.execute(
        "SELECT index_id, chunk_id FROM chunks WHERE doc_hash = ? ORDER BY ordinal",
        (first.doc_hash,),
    ).fetchall()
    by_index = {i: [c for idx, c in rows if idx == i] for i in (index.index_id, other.index_id)}
    assert by_index[index.index_id] == by_index[other.index_id]
    assert len(by_index[index.index_id]) == first.chunks


async def test_a_different_chunk_target_gets_different_chunk_ids(conn, store, data_dir, index, pdf):
    first = await ingest_file(conn, store, backends(), data_dir, index, pdf)
    smaller = Backends(**{**vars(backends()), "tokenizer": WhitespaceTokenizer(max_tokens=50)})
    other = await create_index(conn, store, FakeResolver(BGE_SMALL), "small")

    await ingest_file(conn, store, smaller, data_dir, other, pdf)

    ids = {
        index_id: {
            c
            for (c,) in conn.execute("SELECT chunk_id FROM chunks WHERE index_id = ?", (index_id,))
        }
        for index_id in (index.index_id, other.index_id)
    }
    assert first.chunks and not ids[index.index_id] & ids[other.index_id]


async def test_a_kill_during_embed_resumes_without_reparsing(conn, store, data_dir, index, pdf):
    parser = FakeParser(docling_doc(heading="Intro", text=LONG_TEXT))
    with pytest.raises(KeyboardInterrupt):
        await ingest_file(
            conn,
            store,
            backends(parser, FakeDenseEmbedder(fail_with=KeyboardInterrupt())),
            data_dir,
            index,
            pdf,
        )
    assert ("embed", "running", 1, None) in jobs(conn)

    result = await ingest_file(conn, store, backends(parser), data_dir, index, pdf)

    assert parser.calls == 1
    assert jobs(conn) == [
        ("chunk", "done", 1, None),
        ("embed", "done", 2, None),
        ("parse", "done", 1, None),
        ("register", "done", 2, None),
        ("upsert", "done", 1, None),
    ]
    assert await point_count(store, index, result.doc_hash) == result.chunks


class KilledStore(QdrantStore):
    """A store whose first upsert dies the way a killed process would."""

    def __init__(self, client):
        super().__init__(client)
        self.killed = False

    async def upsert(self, collection, points):
        if not self.killed:
            self.killed = True
            raise KeyboardInterrupt
        await super().upsert(collection, points)


async def test_a_kill_during_upsert_resumes_without_reembedding(conn, store, data_dir, index, pdf):
    killed = KilledStore(store.client)
    dense = FakeDenseEmbedder()
    with pytest.raises(KeyboardInterrupt):
        await ingest_file(conn, killed, backends(dense=dense), data_dir, index, pdf)

    result = await ingest_file(conn, killed, backends(dense=dense), data_dir, index, pdf)

    assert dense.calls == 1
    assert ("upsert", "done", 2, None) in jobs(conn)
    assert await point_count(store, index, result.doc_hash) == result.chunks


async def test_ingest_registers_parses_and_indexes_the_document(conn, store, data_dir, index, pdf):
    result = await ingest_file(conn, store, backends(), data_dir, index, pdf)

    assert result.status == "ingested"
    assert conn.execute("SELECT doc_hash, source_path, mime_type FROM documents").fetchall() == [
        (result.doc_hash, str(pdf), "application/pdf")
    ]
    assert conn.execute("SELECT index_id, doc_hash, status FROM index_members").fetchall() == [
        (index.index_id, result.doc_hash, "indexed")
    ]
    [(json_path,)] = conn.execute(
        "SELECT json_path FROM parse_cache WHERE doc_hash = ? AND parser_version = 'fake/1'",
        (result.doc_hash,),
    ).fetchall()
    assert (data_dir / json_path).is_file()
    assert jobs(conn) == [
        ("chunk", "done", 1, None),
        ("embed", "done", 1, None),
        ("parse", "done", 1, None),
        ("register", "done", 1, None),
        ("upsert", "done", 1, None),
    ]


async def test_doc_hash_is_the_sha256_of_the_file_bytes(conn, store, data_dir, index, pdf):
    result = await ingest_file(conn, store, backends(), data_dir, index, pdf)

    assert result.doc_hash == hashlib.sha256(pdf.read_bytes()).hexdigest()


async def test_reingesting_an_indexed_file_does_nothing_new(conn, store, data_dir, index, pdf):
    parser = FakeParser(docling_doc(text="Parsed once."))
    dense = FakeDenseEmbedder()

    await ingest_file(conn, store, backends(parser, dense), data_dir, index, pdf)
    second = await ingest_file(conn, store, backends(parser, dense), data_dir, index, pdf)

    assert second.status == "unchanged"
    assert (parser.calls, dense.calls) == (1, 1)
    assert second.chunks == 1


async def test_the_same_bytes_at_a_second_path_join_the_index_without_reembedding(
    conn, store, data_dir, index, pdf, tmp_path
):
    copy = tmp_path / "copy.pdf"
    copy.write_bytes(pdf.read_bytes())
    dense = FakeDenseEmbedder()

    await ingest_file(conn, store, backends(dense=dense), data_dir, index, pdf)
    await ingest_file(conn, store, backends(dense=dense), data_dir, index, copy)

    assert dense.calls == 1
    assert {path for (path,) in conn.execute("SELECT source_path FROM index_members")} == {
        str(pdf),
        str(copy),
    }


async def test_the_data_dir_holds_no_copy_of_the_source_pdf(conn, store, data_dir, index, pdf):
    await ingest_file(conn, store, backends(), data_dir, index, pdf)

    stored = [path for path in data_dir.rglob("*") if path.is_file()]
    assert stored
    assert all(p.name.startswith("locus.db") or p.suffix in (".json", ".npz") for p in stored)
    assert all(p.read_bytes() != pdf.read_bytes() for p in stored)


async def test_title_comes_from_pdf_metadata_first(conn, store, data_dir, index, tmp_path):
    pdf = write_pdf(tmp_path / "a.pdf", title="Quarterly Notes")
    parser = FakeParser(docling_doc(heading="Intro"))

    result = await ingest_file(conn, store, backends(parser), data_dir, index, pdf)

    assert result.title == "Quarterly Notes"
    assert conn.execute("SELECT title, title_source FROM documents").fetchone() == (
        "Quarterly Notes",
        "metadata",
    )


async def test_title_falls_back_to_the_first_heading(conn, store, data_dir, index, tmp_path):
    pdf = write_pdf(tmp_path / "a.pdf")
    parser = FakeParser(docling_doc(heading="Intro"))

    await ingest_file(conn, store, backends(parser), data_dir, index, pdf)

    assert conn.execute("SELECT title, title_source FROM documents").fetchone() == (
        "Intro",
        "heading",
    )


async def test_title_falls_back_to_the_filename(conn, store, data_dir, index, tmp_path):
    pdf = write_pdf(tmp_path / "meeting-minutes.pdf")

    await ingest_file(conn, store, backends(FakeParser(docling_doc())), data_dir, index, pdf)

    assert conn.execute("SELECT title, title_source FROM documents").fetchone() == (
        "meeting-minutes",
        "filename",
    )


async def test_an_edited_title_is_kept_on_reingest(conn, store, data_dir, index, pdf):
    await ingest_file(conn, store, backends(), data_dir, index, pdf)
    with conn:
        conn.execute("UPDATE documents SET title = 'My Notes', title_source = 'user'")

    result = await ingest_file(conn, store, backends(), data_dir, index, pdf)

    assert result.title == "My Notes"


async def test_a_parse_failure_is_recorded_and_retried(conn, store, data_dir, index, pdf):
    with pytest.raises(RuntimeError):
        failing = backends(FakeParser(error=RuntimeError("bad xref")))
        await ingest_file(conn, store, failing, data_dir, index, pdf)

    assert jobs(conn) == [
        ("parse", "failed", 1, "RuntimeError: bad xref"),
        ("register", "done", 1, None),
    ]

    await ingest_file(conn, store, backends(), data_dir, index, pdf)

    assert ("parse", "done", 2, None) in jobs(conn)


async def test_a_non_pdf_is_refused_before_anything_is_written(
    conn, store, data_dir, index, tmp_path
):
    notes = tmp_path / "notes.txt"
    notes.write_text("not a pdf")

    with pytest.raises(UnsupportedFileError):
        await ingest_file(conn, store, backends(), data_dir, index, notes)

    assert conn.execute("SELECT count(*) FROM documents").fetchone() == (0,)
    assert jobs(conn) == []
