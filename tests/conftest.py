import os
import socket
import subprocess
import time
import urllib.request

import pytest
from typer.testing import CliRunner

from locus.cli import COMPOSE_FILE, app

PORTS = (6333, 6334)


@pytest.fixture
def anyio_backend():
    return "asyncio"


def port_open(host, port):
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) == 0


def wait_for_health(timeout_s=60):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen("http://127.0.0.1:6333/healthz", timeout=2) as resp:
                if resp.status == 200:
                    return
        except OSError:
            pass
        time.sleep(1)
    pytest.fail("Qdrant did not become healthy")


@pytest.fixture(scope="session")
def qdrant(tmp_path_factory):
    if any(port_open("127.0.0.1", port) for port in PORTS):
        pytest.skip("ports 6333/6334 are already in use")

    data_dir = tmp_path_factory.mktemp("locus")
    env = {"LOCUS_DATA_DIR": str(data_dir), "COMPOSE_PROJECT_NAME": "locus-test"}
    saved = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    try:
        result = CliRunner().invoke(app, ["up"])
        assert result.exit_code == 0, result.output
        wait_for_health()
        yield
    finally:
        subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE_FILE), "down", "-v"],
            env={**os.environ},
            check=False,
        )
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
