import sqlite3

from locus_core.db import DB_FILENAME, connect


def table_names(conn):
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {name for (name,) in rows}


def test_fresh_data_dir_gets_every_table(tmp_path):
    conn = connect(tmp_path)

    assert table_names(conn) == {
        "documents",
        "parse_cache",
        "indexes",
        "index_members",
        "chunks",
        "jobs",
        "model_servers",
        "query_log",
    }


def test_connection_uses_wal_and_waits_on_locks(tmp_path):
    conn = connect(tmp_path)

    assert conn.execute("PRAGMA journal_mode").fetchone() == ("wal",)
    assert conn.execute("PRAGMA busy_timeout").fetchone() == (5000,)


def test_reopening_keeps_existing_rows(tmp_path):
    with connect(tmp_path) as conn:
        conn.execute(
            "INSERT INTO documents (doc_hash, source_path, mime_type, size, added_at)"
            " VALUES ('abc', '/docs/a.pdf', 'application/pdf', 10, '2026-10-24T00:00:00Z')"
        )
    conn.close()

    reopened = connect(tmp_path)

    assert reopened.execute("SELECT source_path FROM documents").fetchall() == [("/docs/a.pdf",)]


def test_an_older_data_dir_gets_the_replaces_column(tmp_path):
    old = sqlite3.connect(tmp_path / DB_FILENAME)
    old.execute(
        "CREATE TABLE index_members (index_id TEXT NOT NULL, source_path TEXT NOT NULL,"
        " doc_hash TEXT NOT NULL, status TEXT NOT NULL, updated_at TEXT NOT NULL,"
        " PRIMARY KEY (index_id, source_path))"
    )
    old.execute("INSERT INTO index_members VALUES ('i', '/a.pdf', 'abc', 'indexed', 't')")
    old.commit()
    old.close()

    conn = connect(tmp_path)

    assert conn.execute("SELECT doc_hash, replaces FROM index_members").fetchall() == [
        ("abc", None)
    ]


def test_chunks_are_indexed_by_document(tmp_path):
    conn = connect(tmp_path)

    plan = conn.execute(
        "EXPLAIN QUERY PLAN SELECT 1 FROM chunks WHERE index_id = 'i' AND doc_hash = 'd'"
    ).fetchall()

    assert any("chunks_by_doc" in row[-1] for row in plan)
