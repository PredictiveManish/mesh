"""Security and robustness checks that do not need torch.

The training ring and the telemetry API both accept input from the network,
so the limits on that input are part of the contract.
"""

import json
import socket
import struct
import sys
import os
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from mesh import protocol as P
from mesh.registry import Registry
from mesh.server import MAX_BODY_BYTES, MAX_LIMIT, make_handler


# ------------------------------------------------------------------ protocol
def test_oversized_header_is_rejected():
    with pytest.raises(ValueError):
        P._check_lengths(P.MAX_HEADER + 1, 0)


def test_oversized_payload_is_rejected():
    with pytest.raises(ValueError):
        P._check_lengths(10, P.MAX_PAYLOAD + 1)


def test_recv_msg_refuses_a_huge_payload_before_reading_it():
    """A peer used to be able to announce a 4 GB payload and get it allocated."""

    class FakeSocket:
        def __init__(self, frame):
            self.frame = frame
            self.read = 0

        def recv(self, n):
            chunk = self.frame[self.read:self.read + n]
            self.read += len(chunk)
            return chunk

    frame = struct.pack("!II", 4, 3 * 1024 ** 3) + b"null"
    with pytest.raises(ValueError):
        P.recv_msg(FakeSocket(frame))


def test_a_normal_frame_still_round_trips():
    header = {"type": "register", "node_id": "w0"}
    packed = P.pack_msg(header, b"payload")
    parsed, hl, pl = P.unpack_header(packed)
    assert parsed == header
    assert pl == len(b"payload")


# -------------------------------------------------------------------- server
@pytest.fixture
def server(tmp_path):
    registry = Registry(str(tmp_path / "fleet.db"))
    handler = make_handler(registry, "secret", Path("dashboard/index.html"))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.05)
    yield f"http://127.0.0.1:{httpd.server_address[1]}", registry
    httpd.shutdown()


def _post(url, body, token="secret", content_length=None):
    data = json.dumps(body).encode() if isinstance(body, dict) else body
    request = urllib.request.Request(url, data=data, method="POST")
    request.add_header("Content-Type", "application/json")
    if token:
        request.add_header("X-Mesh-Token", token)
    if content_length is not None:
        request.add_header("Content-Length", str(content_length))
    return urllib.request.urlopen(request, timeout=5)


def test_ingest_requires_the_token(server):
    base, _ = server
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(f"{base}/api/v1/nodes", {"node_id": "w0"}, token="")
    assert exc.value.code == 401


def test_ingest_rejects_a_wrong_token(server):
    base, _ = server
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(f"{base}/api/v1/nodes", {"node_id": "w0"}, token="nope")
    assert exc.value.code == 401


def test_ingest_accepts_a_valid_report(server):
    base, registry = server
    resp = _post(f"{base}/api/v1/nodes", {"node_id": "w0", "gpu_name": "RTX 3060"})
    assert resp.status == 200
    assert registry.count() == 1


def test_oversized_body_is_refused(server):
    base, _ = server
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(f"{base}/api/v1/nodes", b"{}", content_length=MAX_BODY_BYTES + 1)
    assert exc.value.code == 413


def test_malformed_json_is_a_400(server):
    base, _ = server
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(f"{base}/api/v1/nodes", b"{not json")
    assert exc.value.code == 400


def test_bad_limit_does_not_crash_the_endpoint(server):
    base, _ = server
    resp = urllib.request.urlopen(f"{base}/api/v1/nodes?limit=abc", timeout=5)
    assert resp.status == 200
    assert json.loads(resp.read())["nodes"] == []


def test_limit_is_clamped(server):
    base, registry = server
    for i in range(3):
        registry.upsert({"node_id": f"w{i}"})
    resp = urllib.request.urlopen(f"{base}/api/v1/nodes?limit=999999", timeout=5)
    assert resp.status == 200
    assert len(json.loads(resp.read())["nodes"]) <= MAX_LIMIT


def test_health_is_open(server):
    base, _ = server
    resp = urllib.request.urlopen(f"{base}/api/v1/health", timeout=5)
    assert json.loads(resp.read())["ok"] is True


# ------------------------------------------------------------------ registry
def test_concurrent_heartbeats_do_not_lock_the_database(tmp_path):
    registry = Registry(str(tmp_path / "fleet.db"))
    registry.upsert({"node_id": "w0"})
    errors = []

    def beat(i):
        try:
            registry.heartbeat("w0", {"ram_available_gb": 8.0 + i, "gpu_util": i})
        except Exception as exc:  # noqa: BLE001 - the test is about exceptions
            errors.append(exc)

    threads = [threading.Thread(target=beat, args=(i,)) for i in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert registry.count() == 1
