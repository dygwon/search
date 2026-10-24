"""Qdrant must listen on loopback only. Needs Docker: run with `pytest -m docker`."""

import os
import socket
import subprocess
import time
import urllib.request

import pytest
from typer.testing import CliRunner

from locus.cli import COMPOSE_FILE, app

pytestmark = pytest.mark.docker

PORTS = (6333, 6334)


def port_open(host, port):
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) == 0


def non_loopback_ip():
    # A UDP connect picks the outbound interface without sending anything.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            sock.connect(("192.0.2.1", 80))
        except OSError:
            return None
        ip = sock.getsockname()[0]
    return None if ip.startswith("127.") else ip


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


@pytest.fixture(scope="module")
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


@pytest.mark.parametrize("port", PORTS)
def test_qdrant_answers_on_loopback(qdrant, port):
    assert port_open("127.0.0.1", port)


@pytest.mark.parametrize("port", PORTS)
def test_qdrant_is_unreachable_from_the_network_interface(qdrant, port):
    ip = non_loopback_ip()
    if ip is None:
        pytest.skip("no non-loopback interface to test from")

    assert not port_open(ip, port)
