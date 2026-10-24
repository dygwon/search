import re
import subprocess
from pathlib import Path

import pytest
from locus_core.db import connect
from locus_core.qdrant_store import QdrantStore
from qdrant_client import AsyncQdrantClient
from typer.testing import CliRunner

from locus import cli
from locus.cli import Services, app
from tests.fakes import BGE_SMALL, FakeParser, FakeResolver, write_pdf

runner = CliRunner()


def listed_commands(help_output):
    return set(re.findall(r"^[│\s]*([a-z][a-z-]*)\s{2,}", help_output, re.MULTILINE))


def test_help_lists_up_command():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "up" in listed_commands(result.output)


class FakeRun:
    """Stands in for subprocess.run, the boundary between Locus and Docker."""

    def __init__(self, returncode=0):
        self.returncode = returncode
        self.calls = []

    def __call__(self, args, **kwargs):
        self.calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, self.returncode)


def test_up_starts_qdrant_with_compose_detached(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCUS_DATA_DIR", str(tmp_path))
    fake_run = FakeRun()
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = runner.invoke(app, ["up"])

    assert result.exit_code == 0
    [(args, kwargs)] = fake_run.calls
    assert args[:3] == ["docker", "compose", "-f"]
    assert Path(args[3]).name == "compose.yaml" and Path(args[3]).is_file()
    assert args[4:] == ["up", "-d"]
    assert kwargs["env"]["LOCUS_DATA_DIR"] == str(tmp_path)


def test_up_fails_when_compose_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCUS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(subprocess, "run", FakeRun(returncode=1))

    result = runner.invoke(app, ["up"])

    assert result.exit_code == 1


@pytest.fixture
def services(monkeypatch, tmp_path):
    services = Services(
        conn=connect(tmp_path),
        store=QdrantStore(AsyncQdrantClient(":memory:")),
        resolver=FakeResolver(BGE_SMALL),
        parser=FakeParser(),
    )
    monkeypatch.setattr(cli, "build_services", lambda settings: services)
    return services


def test_index_create_confirms_name_and_chunk_budget(services):
    result = runner.invoke(app, ["index", "create", "notes"])

    assert result.exit_code == 0
    assert "Created index 'notes'" in result.output
    assert "chunk target 512 tokens (ceiling 512)" in result.output


def test_index_create_says_when_the_target_is_capped(services):
    result = runner.invoke(app, ["index", "create", "notes", "--target", "2000"])

    assert result.exit_code == 0
    assert "Target 2000 exceeds the ceiling; capped to 512" in result.output


def test_index_create_refuses_a_duplicate_name(services):
    runner.invoke(app, ["index", "create", "notes"])

    result = runner.invoke(app, ["index", "create", "notes"])

    assert result.exit_code == 1
    assert "an index named 'notes' already exists" in result.output


def test_index_list_shows_each_index_with_its_dense_model(services):
    runner.invoke(app, ["index", "create", "notes"])
    runner.invoke(app, ["index", "create", "papers", "--target", "256"])

    result = runner.invoke(app, ["index", "list"])

    assert result.exit_code == 0
    assert result.output.splitlines() == [
        "notes   BAAI/bge-small-en-v1.5@5c38ec7  target 512",
        "papers  BAAI/bge-small-en-v1.5@5c38ec7  target 256",
    ]


def test_ingest_reports_each_file_then_its_cache_hit(services, tmp_path, monkeypatch):
    monkeypatch.setenv("LOCUS_DATA_DIR", str(tmp_path / "data"))
    pdf = write_pdf(tmp_path / "sample.pdf", title="Quarterly Notes")
    runner.invoke(app, ["index", "create", "notes"])

    first = runner.invoke(app, ["ingest", str(pdf), "--index", "notes"])
    second = runner.invoke(app, ["ingest", str(pdf), "--index", "notes"])

    assert first.exit_code == 0, first.output
    assert re.fullmatch(r"parsed  [0-9a-f]{12}  Quarterly Notes  \(sample.pdf\)\n", first.output)
    assert second.output.startswith("cached  ")


def test_ingest_into_an_unknown_index_fails_clearly(services, tmp_path):
    pdf = write_pdf(tmp_path / "sample.pdf")

    result = runner.invoke(app, ["ingest", str(pdf), "--index", "nope"])

    assert result.exit_code == 1
    assert "no index named 'nope'" in result.output


def test_ingest_keeps_going_past_a_bad_file_and_fails(services, tmp_path, monkeypatch):
    monkeypatch.setenv("LOCUS_DATA_DIR", str(tmp_path / "data"))
    notes = tmp_path / "notes.txt"
    notes.write_text("not a pdf")
    pdf = write_pdf(tmp_path / "sample.pdf")
    runner.invoke(app, ["index", "create", "notes"])

    result = runner.invoke(app, ["ingest", str(notes), str(pdf), "--index", "notes"])

    assert result.exit_code == 1
    assert "notes.txt is not a PDF" in result.output
    assert "parsed  " in result.output
