"""One participant. Trains locally, talks rarely."""
from __future__ import annotations

import argparse
import socket
import sys
import time

import torch

from . import model as M
from . import protocol as P
from .census import probe_local


def attack(delta: dict, kind: str, scale: float = 5.0) -> dict:
    """What a dishonest worker might send."""
    if kind == "noise":
        # Large random vector: crude, and the norm gate catches it.
        real = torch.cat([v.reshape(-1) for v in delta.values()]).norm()
        return {k: torch.randn_like(v) * (real * scale / v.numel() ** 0.5)
                for k, v in delta.items()}
    if kind == "subtle":
        # Same magnitude as an honest update, but pointing nowhere.
        real = torch.cat([v.reshape(-1) for v in delta.values()]).norm()
        rand = torch.cat([torch.randn_like(v).reshape(-1) for v in delta.values()])
        f = real / (rand.norm() + 1e-12)
        out, i = {}, 0
        for k, v in delta.items():
            n = v.numel()
            out[k] = (rand[i:i + n] * f).reshape(v.shape)
            i += n
        return out
    raise ValueError(kind)


def run(args) -> int:
    torch.set_num_threads(1)

    # ---- do every expensive thing BEFORE registering --------------------
    # The coordinator starts its clock when the last worker registers, so
    # interpreter start-up, corpus construction and model build must all
    # happen on this side of the handshake or they show up as a phantom
    # 10-second first round.
    corpus = M.make_corpus()
    train = corpus[: int(len(corpus) * 0.95)]   # tail is the probe set
    model = M.build()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    with torch.no_grad():
        model(torch.zeros(1, M.SEQ_LEN, dtype=torch.long))   # warm the kernels

    profile = probe_local(args.worker_id)
    sock = socket.create_connection((args.host, args.port), timeout=60)
    sock.settimeout(300)

    P.send_msg(sock, {"type": "register", "node_id": args.worker_id,
                      "token": getattr(args, "token", ""),
                      "profile": profile.to_dict()})
    header, _ = P.recv_msg(sock)
    if header.get("type") != "welcome":
        print(f"[{args.worker_id}] handshake failed: {header}", file=sys.stderr)
        return 1

    H = int(header["H"])
    mb = int(header["microbatch"])
    seq = int(header["seq_len"])
    compression = header["compression"]
    shard_idx = int(header["shard"])

    gen = torch.Generator().manual_seed(1000 + shard_idx)
    data = M.shard(train, shard_idx, max(args.workers, 1))

    print(f"[{args.worker_id}] up  shard {shard_idx}  {len(data):,} tokens  "
          f"byzantine={args.byzantine or '-'}", flush=True)

    while True:
        header, payload = P.recv_msg(sock)
        kind = header.get("type")
        if kind == "stop":
            break
        if kind == "config":
            continue
        if kind != "round":
            continue

        r = int(header["round"])
        state = P.unpack_state(payload)
        model.load_state_dict(state)
        theta0 = {k: v.detach().clone() for k, v in model.state_dict().items()}

        model.train()
        for _ in range(H):
            x, y = M.batch(data, mb, seq, gen)
            loss = torch.nn.functional.cross_entropy(
                model(x).reshape(-1, M.VOCAB), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        cur = model.state_dict()
        delta = {k: (theta0[k] - cur[k]).detach().float() for k in theta0}
        if args.byzantine:
            delta = attack(delta, args.byzantine, args.attack_scale)

        P.send_msg(sock, {"type": "update", "round": r},
                   P.compress(delta, compression))

    P.send_msg(sock, {"type": "bye"})
    sock.close()
    print(f"[{args.worker_id}] done", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mesh.worker")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--worker-id", default="worker-0")
    ap.add_argument("--token", default=os.getenv("MESH_TOKEN", ""),
                    help="shared secret for the coordinator socket")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--byzantine", default="", choices=["", "noise", "subtle"])
    ap.add_argument("--attack-scale", type=float, default=5.0)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
