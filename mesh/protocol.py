"""Wire format, compression and the DiLoCo outer optimiser."""
from __future__ import annotations

import io
import json
import struct

import torch

HEADER = struct.Struct("!II")   # (header length, payload length)


# ---------------------------------------------------------------- framing
def pack_msg(header: dict, payload: bytes = b"") -> bytes:
    h = json.dumps(header).encode()
    return HEADER.pack(len(h), len(payload)) + h + payload


def unpack_header(buf: bytes) -> tuple[dict, int, int]:
    hl, pl = HEADER.unpack(buf[:8])
    return json.loads(buf[8:8 + hl]), hl, pl


def send_msg(sock, header: dict, payload: bytes = b"") -> None:
    sock.sendall(pack_msg(header, payload))


def _recv_exact(sock, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed")
        buf += chunk
    return buf


def recv_msg(sock) -> tuple[dict, bytes]:
    hl, pl = HEADER.unpack(_recv_exact(sock, 8))
    header = json.loads(_recv_exact(sock, hl))
    payload = _recv_exact(sock, pl) if pl else b""
    return header, payload


async def asend(writer, header: dict, payload: bytes = b"") -> None:
    writer.write(pack_msg(header, payload))
    await writer.drain()


async def arecv(reader) -> tuple[dict, bytes]:
    head = await reader.readexactly(8)
    hl, pl = HEADER.unpack(head)
    header = json.loads(await reader.readexactly(hl))
    payload = await reader.readexactly(pl) if pl else b""
    return header, payload


# ------------------------------------------------------------ tensor codec
def pack_state(state: dict) -> bytes:
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in state.items()}, buf)
    return buf.getvalue()


def unpack_state(blob: bytes) -> dict:
    return torch.load(io.BytesIO(blob), weights_only=True)


def compress(delta: dict, mode: str = "int8") -> bytes:
    """Quantise a pseudo-gradient for the wire.

    int8 gives 4x with per-tensor scaling and, in our runs, no measurable
    loss penalty -- the same finding Prime Intellect reports for their
    int8 all-reduce.
    """
    buf = io.BytesIO()
    if mode == "none":
        torch.save({"mode": "none", "t": delta}, buf)
    elif mode == "int8":
        q = {}
        for k, v in delta.items():
            scale = float(v.abs().max()) / 127.0 + 1e-12
            q[k] = (v / scale).round().clamp(-127, 127).to(torch.int8), scale
        torch.save({"mode": "int8", "t": q}, buf)
    else:
        raise ValueError(f"unknown compression mode {mode!r}")
    return buf.getvalue()


def decompress(blob: bytes) -> dict:
    # weights_only=True is mandatory: this blob arrives from an untrusted peer.
    # Unpickling arbitrary objects from a stranger is remote code execution.
    obj = torch.load(io.BytesIO(blob), weights_only=True)
    if obj["mode"] == "none":
        return obj["t"]
    return {k: t.float() * s for k, (t, s) in obj["t"].items()}


# ------------------------------------------------------------ outer optimiser
class OuterOptimizer:
    """Nesterov momentum over pseudo-gradients (DiLoCo's outer loop).

    delta = theta_start - theta_local, so a plain SGD step of lr=1 with no
    momentum reproduces federated averaging. Momentum is what makes it
    behave like an optimiser rather than a blender.
    """

    def __init__(self, state: dict, lr: float = 1.0, beta: float = 0.9):
        self.lr = lr
        self.beta = beta
        self.momentum = {k: torch.zeros_like(v) for k, v in state.items()}

    def step(self, state: dict, delta_avg: dict) -> dict:
        out = {}
        for k, v in state.items():
            m = self.beta * self.momentum[k] + delta_avg[k]
            self.momentum[k] = m
            out[k] = v - self.lr * (delta_avg[k] + self.beta * m)
        return out
