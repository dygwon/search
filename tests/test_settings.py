from pathlib import Path

from locus_core.settings import Settings


def test_data_dir_defaults_to_dot_locus_in_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("LOCUS_DATA_DIR", raising=False)

    assert Settings().data_dir == tmp_path / ".locus"


def test_data_dir_can_be_overridden_by_env(monkeypatch):
    monkeypatch.setenv("LOCUS_DATA_DIR", "/srv/locus-data")

    assert Settings().data_dir == Path("/srv/locus-data")
