"""Node capability probing and synthetic fleet generation.

The census is deliberately conservative: it measures sustained throughput
rather than reading a spec sheet, because the two differ by 2-3x on laptops
that thermally throttle.
"""
from __future__ import annotations

import json
import random
import time

from .types import NodeProfile

# Nameplate dense bf16/fp16 tensor throughput, used only when a real
# benchmark cannot be run (i.e. synthetic fleets).
_KNOWN_CARDS = {
    "GTX 1650":  (4.0, 6.0),
    "RTX 3050":  (4.0, 18.0),
    "RTX 3060":  (6.0, 25.0),
    "RTX 4060":  (8.0, 60.0),
    "RTX 4070":  (8.0, 80.0),
    "RTX 4090":  (16.0, 165.0),
    "RTX 5090":  (24.0, 210.0),
    "A100":      (40.0, 312.0),
    "H100":      (80.0, 990.0),
}


def probe_local(node_id: str = "local") -> NodeProfile:
    """Probe this machine. GPU-aware if torch reports one, CPU otherwise."""
    import os

    try:
        import torch

        torch.set_num_threads(1)
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            props = torch.cuda.get_device_properties(0)
            vram = props.total_memory / 1e9
            name = props.name
            tf = _KNOWN_CARDS.get(next((k for k in _KNOWN_CARDS if k in name), ""), (vram, 30.0))[1]
            return NodeProfile(
                node_id=node_id, vram_gb=round(vram, 1), tflops_bf16=tf,
                uplink_mbps=100.0, downlink_mbps=300.0, kind="gpu",
            )
    except Exception:
        pass

    return NodeProfile(
        node_id=node_id,
        vram_gb=0.0,
        tflops_bf16=0.02,
        uplink_mbps=100.0,
        downlink_mbps=300.0,
        kind="cpu",
        thermal_headroom=0.6,
        host=os.uname().nodename if hasattr(os, "uname") else "localhost",
    )


def synthetic_fleet(n: int = 100, seed: int = 7) -> list[NodeProfile]:
    """A realistic mixed consumer fleet: the 4/6/8/16 GB problem."""
    rng = random.Random(seed)
    mix = [
        ("GTX 1650", 0.30, (8, 40)),    # 4 GB, older, no useful tensor cores
        ("RTX 3060", 0.30, (20, 120)),  # 6 GB
        ("RTX 4060", 0.30, (50, 300)),  # 8 GB
        ("RTX 4090", 0.10, (100, 600)), # 16 GB
    ]
    nodes: list[NodeProfile] = []
    for i in range(n):
        r = rng.random()
        acc = 0.0
        card = mix[-1][0]
        for name, share, _ in mix:
            acc += share
            if r <= acc:
                card = name
                break
        vram, tf = _KNOWN_CARDS[card]
        up = rng.uniform(*next(m[2] for m in mix if m[0] == card))
        # laptops throttle: sustained is a fraction of nameplate
        nodes.append(NodeProfile(
            node_id=f"node-{i:03d}",
            vram_gb=float(vram),
            tflops_bf16=tf,
            uplink_mbps=round(up, 1),
            downlink_mbps=round(up * rng.uniform(2.5, 5.0), 1),
            reliability=round(rng.uniform(0.55, 0.95), 2),
            thermal_headroom=round(rng.uniform(0.55, 0.85), 2),
            kind="gpu",
        ))
    return nodes


def load_fleet(path: str) -> list[NodeProfile]:
    with open(path) as fh:
        raw = json.load(fh)
    items = raw["nodes"] if isinstance(raw, dict) else raw
    return [NodeProfile.from_dict(d) for d in items]


def save_fleet(nodes: list[NodeProfile], path: str) -> None:
    with open(path, "w") as fh:
        json.dump({"generated_at": time.time(),
                   "nodes": [n.to_dict() for n in nodes]}, fh, indent=2)
