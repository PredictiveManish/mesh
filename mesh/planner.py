"""The capacity planner.

Given a job and a census of machines, decide which machines run what, how
often they synchronise, and how fast the whole thing will go.

This is the piece nobody else ships. Every existing system assumes a
homogeneous pool; a real volunteer pool or an enterprise's idle fleet is
4 GB cards next to 16 GB cards next to a machine on 20 Mbps of ADSL.
"""
from __future__ import annotations

import math
import statistics

from .types import JobSpec, NodeProfile, Placement

# Sustained model-FLOPs utilisation we are willing to promise.
# 0.30 is a realistic ceiling for small models on consumer cards; a
# datacentre run with large batches gets 0.45-0.55.
MFU = 0.30

# Bytes per parameter held on the device during training.
#   full: bf16 weights 2 + bf16 grads 2 + fp32 adam m,v 8 + fp32 master 4
#   lora: 4-bit frozen base 0.5 + adapters + activations, amortised
DEVICE_BYTES_PER_PARAM = {"full": 16.0, "lora": 1.2}

# Bytes per parameter actually put on the wire, after compression.
WIRE_BYTES_PER_PARAM = {"none": 4.0, "int8": 1.0, "topk": 0.30}

# Fraction of VRAM we are allowed to fill before activations.
VRAM_FILL = 0.80

# Target: communication should be no more than this share of a round.
COMM_SHARE = 0.20


def bytes_per_param(job: JobSpec) -> float:
    return DEVICE_BYTES_PER_PARAM.get(job.train_mode, 16.0)


def max_params_on(node: NodeProfile, job: JobSpec | None = None) -> float:
    """Largest model that fits on this node as a full replica."""
    if node.vram_gb <= 0:
        return 0.0
    bpp = bytes_per_param(job) if job else 16.0
    return node.vram_gb * 1e9 * VRAM_FILL / bpp


def effective_tflops(node: NodeProfile) -> float:
    return max(node.tflops_bf16 * node.thermal_headroom * MFU, 1e-9)


def _auto_microbatch(node: NodeProfile, job: JobSpec) -> int:
    """Heuristic microbatch.

    A placeholder: the production planner uses measured tokens/s from the
    census agent rather than a memory rule of thumb.
    """
    free = max(node.vram_gb * 1e9 * VRAM_FILL - job.params * bytes_per_param(job), 0.0)
    if node.vram_gb <= 0:                      # CPU node
        return 2
    per_sample = max(job.params * job.seq_len * 0.02, 1.0)
    return int(max(1, min(32, free // per_sample)))


def _step_time(node: NodeProfile, job: JobSpec, microbatch: int) -> float:
    """Seconds of compute for one inner step on this node."""
    tokens = microbatch * job.seq_len
    flops = 6.0 * job.params * tokens
    return flops / (effective_tflops(node) * 1e12)


def _comm_time(node: NodeProfile, job: JobSpec) -> float:
    """Seconds to push one compressed pseudo-gradient out of this node."""
    wire_params = job.params * (job.adapter_fraction if job.train_mode == "lora" else 1.0)
    payload = wire_params * WIRE_BYTES_PER_PARAM.get(job.compression, 1.0)
    if node.uplink_mbps <= 0:
        return float("inf")
    return payload * 8.0 / (node.uplink_mbps * 1e6)


def plan(job: JobSpec, nodes: list[NodeProfile], max_replicas: int = 0) -> Placement:
    """Produce a placement for this job on this fleet."""
    notes: list[str] = []
    if not nodes:
        raise ValueError("empty fleet")

    # ---- 1. Who can hold the model at all? -------------------------------
    holders = [n for n in nodes if max_params_on(n, job) >= job.params]
    excluded = [n.node_id for n in nodes if max_params_on(n, job) < job.params]

    if not holders:
        smallest = max(nodes, key=lambda n: max_params_on(n, job))
        notes.append(
            f"No node can hold {job.params/1e9:.2f}B params as a full replica in "
            f"{job.train_mode} mode (best is {smallest.node_id} at "
            f"{max_params_on(smallest, job)/1e9:.2f}B). "
            f"Needs pipeline parallelism (M3) or a smaller model."
        )
        return Placement(excluded=[n.node_id for n in nodes], notes=notes)

    if excluded:
        notes.append(
            f"{len(excluded)} node(s) excluded: they cannot hold the model and "
            f"are better used for data prep or inference."
        )

    # ---- 2. Rank by useful throughput -----------------------------------
    def score(n: NodeProfile) -> float:
        return effective_tflops(n) * n.reliability

    holders.sort(key=score, reverse=True)
    if max_replicas:
        holders = holders[:max_replicas]

    # ---- 3. Microbatch per node -----------------------------------------
    mb = {n.node_id: (job.microbatch or _auto_microbatch(n, job)) for n in holders}

    # ---- 4. Choose the sync interval H ----------------------------------
    # Pick H so that one communication round costs at most COMM_SHARE of the
    # compute it follows. Slow uplinks therefore get a larger H, which is
    # exactly the trade-off the papers describe.
    if job.sync_every:
        H = job.sync_every
        notes.append(f"sync interval fixed by caller at H={H}")
    else:
        per_step = statistics.median(
            _step_time(n, job, mb[n.node_id]) for n in holders
        )
        worst_comm = max(_comm_time(n, job) for n in holders)
        raw_H = math.ceil(worst_comm / (COMM_SHARE * max(per_step, 1e-9)))
        H = int(max(1, min(500, raw_H)))
        if raw_H > 500:
            notes.append(
                f"sync interval clamped at the 500-step ceiling (unclamped value "
                f"was {raw_H}). Uplinks are fast enough that compute, not "
                f"communication, is the binding constraint."
            )
        else:
            notes.append(
                f"sync interval auto-tuned to H={H} so that one round's upload is "
                f"<= {int(COMM_SHARE*100)}% of its compute."
            )

    # ---- 5. Round time and throughput -----------------------------------
    per_node_round: dict[str, float] = {}
    for n in holders:
        per_node_round[n.node_id] = (
            _step_time(n, job, mb[n.node_id]) * H + _comm_time(n, job)
        )

    round_s = max(per_node_round.values())
    bottleneck = max(per_node_round, key=per_node_round.get)

    tokens_per_round = sum(mb[n.node_id] * job.seq_len * H for n in holders)
    tps = tokens_per_round / round_s if round_s > 0 else 0.0

    rounds = int(math.ceil(job.target_tokens / tokens_per_round)) if job.target_tokens else 0
    eta_hours = (rounds * round_s / 3600.0) if rounds else 0.0

    # ---- 6. Heterogeneity check -----------------------------------------
    spread = max(per_node_round.values()) / max(min(per_node_round.values()), 1e-9)
    if spread > 3.0:
        notes.append(
            f"WARNING: {spread:.1f}x spread in per-round time. The slowest node "
            f"({bottleneck}) sets the pace. Consider dropping the bottom tier or "
            f"giving it a smaller microbatch."
        )

    return Placement(
        replicas=[[n.node_id] for n in holders],
        excluded=excluded,
        model_parallel=1,
        per_node_microbatch=mb,
        sync_every=H,
        expected_tps=tps,
        round_seconds=round_s,
        bottleneck=bottleneck,
        rounds=rounds,
        eta_hours=eta_hours,
        notes=notes,
    )


def render(placement: Placement, job: JobSpec, nodes: list[NodeProfile]) -> str:
    """Human-readable plan."""
    by_id = {n.node_id: n for n in nodes}
    out: list[str] = []
    out.append("")
    out.append(f"  job          {job.model}  ({job.params/1e6:.2f}M params, {job.method})")
    out.append(f"  compression  {job.compression}")
    out.append(f"  replicas     {placement.replica_count}")
    out.append(f"  sync every   H = {placement.sync_every} inner steps")
    out.append(f"  round time   {placement.round_seconds:.2f} s")
    out.append(f"  throughput   {placement.expected_tps:,.0f} tokens/s")
    if placement.rounds:
        out.append(f"  {job.target_tokens/1e9:.1f}B tokens  ->  {placement.rounds:,} rounds, "
                   f"{placement.eta_hours:,.1f} h ({placement.eta_hours/24:,.1f} days)")
    out.append(f"  bottleneck   {placement.bottleneck}")
    out.append("")

    if placement.replica_count:
        out.append(f"  {'node':<16} {'vram':>7} {'mb':>4} {'step ms':>8} {'comm s':>7} {'round s':>8}")
        out.append("  " + "-" * 58)
        for rid in placement.replicas:
            nid = rid[0]
            n = by_id[nid]
            mb = placement.per_node_microbatch[nid]
            out.append(
                f"  {nid:<16} {n.vram_gb:6.1f}G {mb:4d} "
                f"{_step_time(n, job, mb)*1000:8.1f} {_comm_time(n, job):7.3f} "
                f"{_step_time(n, job, mb)*placement.sync_every + _comm_time(n, job):8.2f}"
            )
    if placement.excluded:
        out.append("")
        out.append(f"  excluded: {len(placement.excluded)} node(s) below the memory floor")
    for note in placement.notes:
        out.append(f"  note: {note}")
    out.append("")
    return "\n".join(out)
