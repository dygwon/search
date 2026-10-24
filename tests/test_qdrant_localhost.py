"""Qdrant must listen on loopback only. Needs Docker: run with `pytest -m docker`."""

import socket

import pytest

from tests.conftest import PORTS, port_open

pytestmark = pytest.mark.docker


def non_loopback_ip():
    # A UDP connect picks the outbound interface without sending anything.
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        try:
            sock.connect(("192.0.2.1", 80))
        except OSError:
            return None
        ip = sock.getsockname()[0]
    return None if ip.startswith("127.") else ip


@pytest.mark.parametrize("port", PORTS)
def test_qdrant_answers_on_loopback(qdrant, port):
    assert port_open("127.0.0.1", port)


@pytest.mark.parametrize("port", PORTS)
def test_qdrant_is_unreachable_from_the_network_interface(qdrant, port):
    ip = non_loopback_ip()
    if ip is None:
        pytest.skip("no non-loopback interface to test from")

    assert not port_open(ip, port)
