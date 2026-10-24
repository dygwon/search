import asyncio
import hashlib

import pytest
from locus_core.db import connect
from locus_core.indexes import create_index
from locus_core.qdrant_store import QdrantStore
from locus_ingest.pipeline import ingest_file
from locus_ingest.register import UnsupportedFileError
from qdrant_client import AsyncQdrantClient

from tests.fakes import BGE_SMALL, FakeParser, FakeResolver, docling_doc, write_pdf


@pytest.fixture
def data_dir(tmp_path):
    return tmp_path / "data"


@pytest.fixture
def conn(data_dir):
    return connect(data_dir)


@pytest.fixture
def index(conn):
    store = QdrantStore(AsyncQdrantClient(":memory:"))
    return asyncio.run(create_index(conn, store, FakeResolver(BGE_SMALL), "notes"))


@pytest.fixture
def pdf(tmp_path):
    return write_pdf(tmp_path / "sample.pdf", title="Quarterly Notes")


def jobs(conn):
    return conn.execute("SELECT stage, status, attempts, error FROM jobs ORDER BY stage").fetchall()


def test_ingest_registers_and_parses_the_document(conn, data_dir, index, pdf):
    result = ingest_file(conn, FakeParser(), data_dir, index, pdf)

    assert conn.execute("SELECT doc_hash, source_path, mime_type FROM documents").fetchall() == [
        (result.doc_hash, str(pdf), "application/pdf")
    ]
    assert conn.execute("SELECT index_id, doc_hash, status FROM index_members").fetchall() == [
        (index.index_id, result.doc_hash, "parsed")
    ]
    [(json_path,)] = conn.execute(
        "SELECT json_path FROM parse_cache WHERE doc_hash = ? AND parser_version = 'fake/1'",
        (result.doc_hash,),
    ).fetchall()
    assert (data_dir / json_path).is_file()
    assert jobs(conn) == [("parse", "done", 1, None), ("register", "done", 1, None)]


def test_doc_hash_is_the_sha256_of_the_file_bytes(conn, data_dir, index, pdf):
    result = ingest_file(conn, FakeParser(), data_dir, index, pdf)

    assert result.doc_hash == hashlib.sha256(pdf.read_bytes()).hexdigest()


def test_reingesting_the_same_file_is_a_cache_hit(conn, data_dir, index, pdf):
    parser = FakeParser(docling_doc(text="Parsed once."))

    first = ingest_file(conn, parser, data_dir, index, pdf)
    second = ingest_file(conn, parser, data_dir, index, pdf)

    assert parser.calls == 1
    assert (first.cached, second.cached) == (False, True)
    assert conn.execute("SELECT count(*) FROM parse_cache").fetchone() == (1,)


def test_the_data_dir_holds_no_copy_of_the_source_pdf(conn, data_dir, index, pdf):
    ingest_file(conn, FakeParser(), data_dir, index, pdf)

    stored = [path for path in data_dir.rglob("*") if path.is_file()]
    assert stored
    assert all(p.name.startswith("locus.db") or p.suffix == ".json" for p in stored)
    assert all(p.read_bytes() != pdf.read_bytes() for p in stored)


def test_title_comes_from_pdf_metadata_first(conn, data_dir, index, tmp_path):
    pdf = write_pdf(tmp_path / "a.pdf", title="Quarterly Notes")

    result = ingest_file(conn, FakeParser(docling_doc(heading="Intro")), data_dir, index, pdf)

    assert result.title == "Quarterly Notes"
    assert conn.execute("SELECT title, title_source FROM documents").fetchone() == (
        "Quarterly Notes",
        "metadata",
    )


def test_title_falls_back_to_the_first_heading(conn, data_dir, index, tmp_path):
    pdf = write_pdf(tmp_path / "a.pdf")

    ingest_file(conn, FakeParser(docling_doc(heading="Intro")), data_dir, index, pdf)

    assert conn.execute("SELECT title, title_source FROM documents").fetchone() == (
        "Intro",
        "heading",
    )


def test_title_falls_back_to_the_filename(conn, data_dir, index, tmp_path):
    pdf = write_pdf(tmp_path / "meeting-minutes.pdf")

    ingest_file(conn, FakeParser(docling_doc()), data_dir, index, pdf)

    assert conn.execute("SELECT title, title_source FROM documents").fetchone() == (
        "meeting-minutes",
        "filename",
    )


def test_an_edited_title_is_kept_on_reingest(conn, data_dir, index, pdf):
    ingest_file(conn, FakeParser(), data_dir, index, pdf)
    with conn:
        conn.execute("UPDATE documents SET title = 'My Notes', title_source = 'user'")

    result = ingest_file(conn, FakeParser(), data_dir, index, pdf)

    assert result.title == "My Notes"


def test_a_parse_failure_is_recorded_and_retried(conn, data_dir, index, pdf):
    with pytest.raises(RuntimeError):
        ingest_file(conn, FakeParser(error=RuntimeError("bad xref")), data_dir, index, pdf)

    assert jobs(conn) == [
        ("parse", "failed", 1, "RuntimeError: bad xref"),
        ("register", "done", 1, None),
    ]

    ingest_file(conn, FakeParser(), data_dir, index, pdf)

    assert jobs(conn) == [("parse", "done", 2, None), ("register", "done", 2, None)]


def test_a_non_pdf_is_refused_before_anything_is_written(conn, data_dir, index, tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_text("not a pdf")

    with pytest.raises(UnsupportedFileError):
        ingest_file(conn, FakeParser(), data_dir, index, notes)

    assert conn.execute("SELECT count(*) FROM documents").fetchone() == (0,)
    assert jobs(conn) == []
