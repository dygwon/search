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
from tests.fakes import BGE_SMALL, FakeResolver

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
