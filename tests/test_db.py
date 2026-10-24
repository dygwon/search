from locus_core.db import connect


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
