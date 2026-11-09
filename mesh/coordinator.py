"""Control plane: registry, round barrier, aggregation, checkpointing.

In this prototype the coordinator holds the global model, which makes it the
'server' of DiLoCo. In production that role is sharded across peers and the
coordinator only sequences rounds -- see docs/PRODUCTION-READINESS.md.
"""
from __future__ import annotations

import asyncio
import hmac
import time

import torch

from . import model as M
from . import protocol as P
from .types import NodeProfile
from .verify import RobustAggregator


class Conn:
    def __init__(self, node_id: str, profile: NodeProfile, reader, writer):
        self.node_id = node_id
        self.profile = profile
        self.reader = reader
        self.writer = writer
        self.alive = True
        self.round = -1
        self.update: dict | None = None
        self.task: asyncio.Task | None = None

    async def close(self) -> None:
        self.alive = False
        try:
            self.writer.close()
        except Exception:
            pass


class Coordinator:
    def __init__(self, host: str = "127.0.0.1", port: int = 0, rounds: int = 20,
                 H: int = 12, compression: str = "int8", lr: float = 1.0,
                 beta: float = 0.9, timeout: float = 30.0, microbatch: int = 4,
                 verbose: bool = True, token: str = ""):
        self.host, self.port = host, port
        self.rounds, self.H = rounds, H
        self.compression = compression
        self.lr, self.beta = lr, beta
        self.timeout = timeout
        self.microbatch = microbatch
        self.verbose = verbose
        # Shared secret for the training socket. Empty means the ring is open,
        # which is only acceptable on loopback.
        self.token = token

        self.model = M.build()
        self.state = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.corpus = M.make_corpus()
        self.probe = M.probe_batch(self.corpus)
        self.outer = P.OuterOptimizer(self.state, lr=lr, beta=beta)
        self.agg = RobustAggregator()

        self.conns: dict[str, Conn] = {}
        self.server: asyncio.AbstractServer | None = None
        self.history: list[dict] = []
        self._registered = asyncio.Event()

    # ---------------------------------------------------------------- server
    async def start(self) -> int:
        self.server = await asyncio.start_server(
            self._handle, self.host, self.port, limit=2 ** 26)
        self.port = self.server.sockets[0].getsockname()[1]
        return self.port

    async def _handle(self, reader, writer) -> None:
        try:
            header, _ = await P.arecv(reader)
            if header.get("type") != "register":
                await P.asend(writer, {"type": "error", "msg": "expected register"})
                return
            if self.token and not hmac.compare_digest(
                    str(header.get("token", "")), self.token):
                await P.asend(writer, {"type": "error", "msg": "unauthorised"})
                writer.close()
                return
            node_id = header["node_id"]
            profile = NodeProfile.from_dict(header.get("profile", {"node_id": node_id}))
            conn = Conn(node_id, profile, reader, writer)
            self.conns[node_id] = conn
            idx = len(self.conns) - 1
            await P.asend(writer, {
                "type": "welcome",
                "shard": idx,
                "workers": -1,            # filled at run time
                "H": self.H,
                "microbatch": self.microbatch,
                "seq_len": M.SEQ_LEN,
                "compression": self.compression,
                "rounds": self.rounds,
            })
            self._registered.set()

            while conn.alive:
                header, payload = await P.arecv(reader)
                if header.get("type") == "update":
                    conn.update = P.decompress(payload)
                    conn.round = int(header["round"])
                elif header.get("type") == "bye":
                    break
        except (asyncio.IncompleteReadError, ConnectionError, OSError):
            pass
        except Exception as exc:                       # pragma: no cover
            if self.verbose:
                print(f"  [coord] connection error: {exc!r}")
        finally:
            for c in self.conns.values():
                if c.reader is reader:
                    await c.close()

    async def wait_for_workers(self, n: int, timeout: float = 60.0) -> None:
        t0 = time.time()
        while len(self.conns) < n:
            if time.time() - t0 > timeout:
                raise TimeoutError(f"only {len(self.conns)}/{n} workers registered")
            await asyncio.sleep(0.05)

    # ------------------------------------------------------------------ run
    async def run(self) -> dict:
        assert self.server is not None
        live = list(self.conns.values())
        for c in live:
            await P.asend(c.writer, {"type": "config", "workers": len(live)})

        loss0 = M.evaluate(self.model, *self.probe)
        t_start = time.time()
        dropped: list[str] = []
        rejected_total = 0

        if self.verbose:
            print(f"  round   0  loss {loss0:6.4f}  (start)")

        for r in range(self.rounds):
            live = [c for c in self.conns.values() if c.alive]
            if not live:
                if self.verbose:
                    print("  no live workers left; stopping")
                break

            for c in live:
                c.update, c.round = None, -1
            payload = P.pack_state(self.state)
            for c in live:
                try:
                    await P.asend(c.writer, {"type": "round", "round": r}, payload)
                except Exception:
                    await c.close()

            pending = {c.node_id for c in live}
            deadline = time.time() + self.timeout
            while pending and time.time() < deadline:
                for c in live:
                    if not c.alive or (c.round == r and c.update is not None):
                        pending.discard(c.node_id)
                if not pending:
                    break
                await asyncio.sleep(0.004)

            got = {c.node_id: c.update for c in live
                   if c.round == r and c.update is not None and c.alive}
            for c in live:
                if c.node_id in pending or not c.alive:
                    if c.alive and self.verbose:
                        print(f"  [churn] {c.node_id} timed out; dropping from round {r}")
                    await c.close()
                    if c.node_id not in dropped:
                        dropped.append(c.node_id)

            if not got:
                if self.verbose:
                    print("  no updates received; stopping")
                break

            delta_avg, verdicts = self.agg.aggregate(got)
            rejected = [v for v in verdicts if not v.accepted]
            rejected_total += len(rejected)
            if rejected and self.verbose:
                for v in rejected:
                    print(f"  [verify] REJECT {v.node_id}: {v.reason}")

            self.state = self.outer.step(self.state, delta_avg)
            self.model.load_state_dict(self.state)
            loss = M.evaluate(self.model, *self.probe)

            elapsed = time.time() - t_start
            n = len(got)
            tok = n * self.microbatch * M.SEQ_LEN * self.H
            self.history.append({"round": r, "loss": loss, "n": n,
                                 "rejected": len(rejected), "t": elapsed})
            if self.verbose:
                print(f"  round {r+1:3d}  loss {loss:6.4f}  "
                      f"updates {n}/{len(live)}  rej {len(rejected)}  "
                      f"{elapsed:5.1f}s  {tok/ max(elapsed,1e-9):8,.0f} tok/s")

        wall = time.time() - t_start
        for c in list(self.conns.values()):
            if c.alive:
                try:
                    await P.asend(c.writer, {"type": "stop"})
                except Exception:
                    pass
                await c.close()
        if self.server:
            self.server.close()

        return {
            "loss_start": loss0,
            "loss_end": self.history[-1]["loss"] if self.history else loss0,
            "rounds": len(self.history),
            "wall_s": wall,
            "dropped": dropped,
            "rejected": rejected_total,
            "history": self.history,
        }
