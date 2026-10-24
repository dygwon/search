"""Per-document, per-stage job records, so a crash or rerun resumes at the last finished stage."""

import sqlite3
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime


def run_stage[T](
    conn: sqlite3.Connection, index_id: str, doc_hash: str, stage: str, fn: Callable[[], T]
) -> T:
    """Run one stage of one document's job, recording each attempt and how it ended."""
    job_id = _job_id(conn, index_id, doc_hash, stage)
    _update(conn, job_id, "running", new_attempt=True)
    try:
        result = fn()
    except Exception as exc:
        _update(conn, job_id, "failed", error=f"{type(exc).__name__}: {exc}")
        raise
    _update(conn, job_id, "done")
    return result


async def run_stage_async[T](
    conn: sqlite3.Connection,
    index_id: str,
    doc_hash: str,
    stage: str,
    fn: Callable[[], Awaitable[T]],
) -> T:
    """run_stage for a stage that awaits, such as a Qdrant write."""
    job_id = _job_id(conn, index_id, doc_hash, stage)
    _update(conn, job_id, "running", new_attempt=True)
    try:
        result = await fn()
    except Exception as exc:
        _update(conn, job_id, "failed", error=f"{type(exc).__name__}: {exc}")
        raise
    _update(conn, job_id, "done")
    return result


def stage_done(conn: sqlite3.Connection, index_id: str, doc_hash: str, stage: str) -> bool:
    row = conn.execute(
        "SELECT status FROM jobs WHERE index_id = ? AND doc_hash = ? AND stage = ?",
        (index_id, doc_hash, stage),
    ).fetchone()
    return row is not None and row[0] == "done"


def _job_id(conn: sqlite3.Connection, index_id: str, doc_hash: str, stage: str) -> str:
    row = conn.execute(
        "SELECT job_id FROM jobs WHERE index_id = ? AND doc_hash = ? AND stage = ?",
        (index_id, doc_hash, stage),
    ).fetchone()
    if row:
        return row[0]
    job_id = uuid.uuid4().hex
    with conn:
        conn.execute(
            "INSERT INTO jobs (job_id, index_id, doc_hash, stage, status, updated_at)"
            " VALUES (?, ?, ?, ?, 'pending', ?)",
            (job_id, index_id, doc_hash, stage, _now()),
        )
    return job_id


def _update(
    conn: sqlite3.Connection,
    job_id: str,
    status: str,
    error: str | None = None,
    new_attempt: bool = False,
) -> None:
    with conn:
        conn.execute(
            "UPDATE jobs SET status = ?, error = ?, attempts = attempts + ?, updated_at = ?"
            " WHERE job_id = ?",
            (status, error, int(new_attempt), _now(), job_id),
        )


def _now() -> str:
    return datetime.now(UTC).isoformat()
