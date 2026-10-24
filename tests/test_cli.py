import re
import subprocess
from pathlib import Path

from typer.testing import CliRunner

from locus.cli import app

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
