"""Command line: census, plan, run."""
from __future__ import annotations

import argparse
from pathlib import Path
import asyncio
import json
import os
import subprocess
import sys
import time

from .census import load_fleet, probe_local, save_fleet, synthetic_fleet
from . import seed as seedmod
from .planner import plan, render
from .types import JobSpec

DASHBOARD_PATH = Path(__file__).resolve().parent.parent / 'dashboard' / 'index.html'

BAR = "=" * 72


# ------------------------------------------------------------------- census
def cmd_census(args) -> int:
    if args.fleet:
        nodes = synthetic_fleet(args.fleet, seed=args.seed)
        if args.out:
            save_fleet(nodes, args.out)
    elif args.file:
        nodes = load_fleet(args.file)
    else:
        nodes = [probe_local()]

    print(f"\n  {len(nodes)} node(s)\n")
    print(f"  {'node':<16} {'kind':<4} {'vram':>8} {'tflops':>8} {'up/down Mbps':>18}")
    print("  " + "-" * 60)
    for n in nodes[: args.show]:
        print("  " + n.summary())
    if len(nodes) > args.show:
        print(f"  ... {len(nodes) - args.show} more")

    tiers: dict[float, int] = {}
    for n in nodes:
        tiers[n.vram_gb] = tiers.get(n.vram_gb, 0) + 1
    print("\n  memory tiers: " + "  ".join(
        f"{int(k)}GB x{v}" for k, v in sorted(tiers.items())))
    total = sum(n.vram_gb for n in nodes)
    print(f"  pooled VRAM: {total:,.0f} GB")
    print(f"  pooled peak: {sum(n.tflops_bf16 for n in nodes):,.0f} TFLOPS (nameplate)\n")
    if args.out:
        print(f"  written to {args.out}\n")
    return 0


# --------------------------------------------------------------------- plan
def cmd_plan(args) -> int:
    nodes = load_fleet(args.file) if args.file else synthetic_fleet(args.fleet, seed=args.seed)
    job = JobSpec(
        model=args.model, params=args.params * 1e9,
        target_tokens=args.tokens * 1e9, compression=args.compression,
        train_mode=args.mode, sync_every=args.sync_every,
        seq_len=args.seq_len, microbatch=args.microbatch,
    )
    p = plan(job, nodes, max_replicas=args.max_replicas)
    print(BAR)
    print(f"  FLEET: {len(nodes)} nodes, {sum(n.vram_gb for n in nodes):,.0f} GB pooled")
    print(render(p, job, nodes))
    if args.json:
        print(json.dumps({
            "replicas": p.replica_count, "sync_every": p.sync_every,
            "expected_tps": p.expected_tps, "round_seconds": p.round_seconds,
            "rounds": p.rounds, "eta_hours": p.eta_hours,
            "bottleneck": p.bottleneck, "excluded": len(p.excluded),
        }, indent=2))
    return 0


# ---------------------------------------------------------------------- run
async def _churn(coord: Coordinator, procs, at_round: int) -> None:
    while len(coord.history) < at_round:
        await asyncio.sleep(0.02)
    victim = procs[0]
    if victim.poll() is None:
        victim.terminate()
        print(f"\n  [churn] killed w0 after round {len(coord.history)}\n", flush=True)


async def _run_async(args) -> dict:
    from .coordinator import Coordinator     # torch is only needed for training
    coord = Coordinator(
        port=args.port, rounds=args.rounds, H=args.H,
        compression=args.compression, timeout=args.timeout,
        microbatch=args.microbatch, verbose=not args.quiet,
        token=getattr(args, "token", "") or os.getenv("MESH_TOKEN", ""),
    )
    port = await coord.start()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=root, OMP_NUM_THREADS="1")

    if not args.quiet:
        print(BAR)
        print(f"  {args.workers} workers   H={args.H}   rounds={args.rounds}   "
              f"mb={args.microbatch}   compression={args.compression}")
        if args.byzantine:
            print(f"  attack: {args.byzantine} worker(s) sending '{args.byzantine_kind}'")
        if args.churn_after:
            print(f"  churn: kill w0 after round {args.churn_after}")
        print(BAR)

    procs = []
    for i in range(args.workers):
        byz = args.byzantine_kind if i < args.byzantine else ""
        cmd = [sys.executable, "-m", "mesh.worker", "--port", str(port),
               "--worker-id", f"w{i}", "--workers", str(args.workers),
               "--byzantine", byz, "--lr", str(args.lr)]
        if getattr(args, "token", "") or os.getenv("MESH_TOKEN", ""):
            cmd += ["--token", getattr(args, "token", "") or os.getenv("MESH_TOKEN", "")]
        procs.append(subprocess.Popen(
            cmd, env=env, cwd=root,
            stdout=None if not args.quiet else subprocess.DEVNULL,
            stderr=None if not args.quiet else subprocess.DEVNULL))

    try:
        await coord.wait_for_workers(args.workers)
        task = None
        if args.churn_after:
            task = asyncio.create_task(_churn(coord, procs, args.churn_after))
        result = await coord.run()
        if task:
            task.cancel()
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except Exception:
                p.kill()
    return result


def cmd_run(args) -> int:
    result = asyncio.run(_run_async(args))
    h = result["history"]
    print(BAR)
    print(f"  loss   {result['loss_start']:.4f}  ->  {result['loss_end']:.4f}")
    print(f"  rounds {result['rounds']}   wall {result['wall_s']:.1f}s")
    if result["dropped"]:
        print(f"  dropped mid-run: {', '.join(result['dropped'])}")
    print(f"  updates rejected by verifier: {result['rejected']}")
    if len(h) >= 2:
        tok = args.workers * args.microbatch * 48 * args.H * len(h)
        print(f"  effective {tok / max(result['wall_s'], 1e-9):,.0f} tokens/s "
              f"(this machine, {args.workers} CPU processes)")
    print(BAR)
    return 0


# ------------------------------------------------------------ serve / agent
def cmd_serve(args) -> int:
    from .server import serve
    serve(db=args.db, host=args.host, port=args.port, token=args.token,
          dashboard=args.dashboard or None, verbose=args.verbose)
    return 0


def cmd_agent(args) -> int:
    from . import agent
    return agent.run(args)


def cmd_seed(args) -> int:
    n = seedmod.seed(args.server, args.nodes, args.token, args.seed)
    print(f"  seeded {n} nodes into {args.server}")
    return 0



# ------------------------------------------------------------------- scan
MODEL_CATALOGUE = [
    ("Qwen2.5-0.5B", 0.5), ("Llama-3.2-1B", 1.2), ("TinyLlama-1.1B", 1.1),
    ("Qwen2.5-3B", 3.0), ("Llama-3.2-3B", 3.2), ("Phi-3-mini", 3.8),
    ("Mistral-7B", 7.2), ("Llama-3.1-8B", 8.0), ("Qwen2.5-14B", 14.7),
    ("Qwen2.5-32B", 32.5), ("Llama-3.3-70B", 70.6),
]


def _bar(ok, width=0):
    return "yes" if ok else "—"


def cmd_scan(args) -> int:
    """What can this machine (or this fleet) actually train?"""
    from .planner import max_params_on

    if args.fleet:
        nodes = synthetic_fleet(args.fleet, seed=args.seed)
        title = f"Simulated fleet · {len(nodes)} nodes"
    elif args.file:
        nodes = load_fleet(args.file)
        title = f"Fleet from {args.file} · {len(nodes)} nodes"
    else:
        nodes = [probe_local("this-machine")]
        title = "This machine"

    print()
    print("  " + title)
    print("  " + "-" * 62)
    for n in nodes[:1] if len(nodes) == 1 else []:
        print(f"  {n.kind.upper()}  {n.vram_gb:.1f} GB VRAM  ·  "
              f"{n.tflops_bf16:.0f} TFLOPS  ·  {n.uplink_mbps:.0f} Mbps up")
        print(f"  host: {n.host}")
    if len(nodes) > 1:
        vram = sorted({n.vram_gb for n in nodes if n.vram_gb})
        print(f"  tiers: " + "  ".join(f"{int(v)}GB" for v in vram))
        print(f"  pooled VRAM: {sum(n.vram_gb for n in nodes):,.0f} GB")

    full_max = max(max_params_on(n, JobSpec(train_mode="full")) for n in nodes)
    lora_max = max(max_params_on(n, JobSpec(train_mode="lora")) for n in nodes)

    if full_max <= 0:
        print()
        print("  No GPU detected on this machine, so there is no VRAM budget to")
        print("  plan against. mesh sizes models by device memory, not system RAM.")
        print("  Try a simulated fleet instead:   mesh scan --fleet 100")
        print()
        return 0

    print()
    print(f"  Full fine-tune       up to {full_max/1e9:5.2f}B params per node")
    print(f"  QLoRA fine-tune      up to {lora_max/1e9:5.2f}B params per node")
    print()
    print(f"  {'model':<20}{'params':>8}{'full':>8}{'QLoRA':>8}"
          f"{'nodes that fit':>16}")
    print("  " + "-" * 60)
    for name, params in MODEL_CATALOGUE:
        p = params * 1e9
        n_full = sum(1 for n in nodes if max_params_on(n, JobSpec(train_mode="full")) >= p)
        n_lora = sum(1 for n in nodes if max_params_on(n, JobSpec(train_mode="lora")) >= p)
        if n_lora == 0:
            continue
        print(f"  {name:<20}{params:>7.1f}B{_bar(n_full):>8}{_bar(n_lora):>8}"
              f"{n_lora:>10} / {len(nodes):<4}")

    best_full = max((m for m in MODEL_CATALOGUE if m[1] * 1e9 <= full_max),
                    default=None, key=lambda m: m[1])
    best_lora = max((m for m in MODEL_CATALOGUE if m[1] * 1e9 <= lora_max),
                    default=None, key=lambda m: m[1])
    print()
    if best_full:
        print(f"  Largest full fine-tune that fits:  {best_full[0]}")
    else:
        print("  Largest full fine-tune that fits:  none — use QLoRA or a smaller model")
    if best_lora:
        print(f"  Largest QLoRA fine-tune that fits: {best_lora[0]}")
    print()
    print("  Next:  mesh plan --params <B> --mode full|lora    (throughput + ETA)")
    print("         mesh up                                  (dashboard)")
    print()
    return 0


# --------------------------------------------------------------------- up
def cmd_up(args) -> int:
    """One command: coordinator + dashboard, optionally seeded with a demo fleet."""
    import threading, webbrowser
    from http.server import ThreadingHTTPServer
    from .registry import Registry
    from .server import make_handler

    if not args.token and args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"  WARNING: binding {args.host} with no --token. Anyone who can\n"
              f"           reach this port can register a node and post telemetry.\n"
              f"           Set --token or MESH_TOKEN.\n", file=sys.stderr)

    reg = Registry(args.db)
    handler = make_handler(reg, args.token,
                           Path(args.dashboard) if args.dashboard else DASHBOARD_PATH)
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    if args.demo:
        from . import seed as seedmod
        n = seedmod.seed(f"http://127.0.0.1:{port}", args.demo_nodes, args.token)
        print(f"  seeded {n} synthetic nodes (demo data — not real installs)")

    url = f"http://localhost:{port}/"
    print()
    print("  mesh is running")
    print(f"  dashboard   {url}")
    print(f"  api         {url}api/v1/stats")
    print(f"  registry    {args.db}  ({reg.count()} nodes)")
    print()
    print("  install a real machine with:  mesh agent --server "
          f"http://<this-host>:{port}")
    print("  Ctrl-C to stop")
    print()
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("  stopped")
        httpd.shutdown()
    return 0


# --------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="mesh", description="mesh prototype")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("census", help="probe a machine or generate a fleet")
    c.add_argument("--fleet", type=int, default=0)
    c.add_argument("--file", default="")
    c.add_argument("--out", default="")
    c.add_argument("--seed", type=int, default=7)
    c.add_argument("--show", type=int, default=12)
    c.set_defaults(func=cmd_census)

    p = sub.add_parser("plan", help="capacity plan for a job on a fleet")
    p.add_argument("--file", default="")
    p.add_argument("--fleet", type=int, default=100)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--model", default="demo")
    p.add_argument("--params", type=float, default=0.5, help="billions")
    p.add_argument("--tokens", type=float, default=20.0, help="billions")
    p.add_argument("--mode", default="full", choices=["full", "lora"])
    p.add_argument("--compression", default="int8", choices=["none", "int8", "topk"])
    p.add_argument("--sync-every", type=int, default=0)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--microbatch", type=int, default=0)
    p.add_argument("--max-replicas", type=int, default=0)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_plan)

    r = sub.add_parser("run", help="run a real distributed training round-trip")
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--token", default=os.getenv("MESH_TOKEN", ""),
                   help="shared secret for the coordinator socket")
    r.add_argument("--rounds", type=int, default=20)
    r.add_argument("--H", type=int, default=12, help="inner steps per round")
    r.add_argument("--microbatch", type=int, default=4)
    r.add_argument("--compression", default="int8", choices=["none", "int8"])
    r.add_argument("--lr", type=float, default=1e-3, help="inner AdamW lr")
    r.add_argument("--port", type=int, default=0)
    r.add_argument("--timeout", type=float, default=30.0)
    r.add_argument("--byzantine", type=int, default=0)
    r.add_argument("--byzantine-kind", default="noise", choices=["noise", "subtle"])
    r.add_argument("--churn-after", type=int, default=0)
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("serve", help="run the coordinator API + dashboard")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--db", default="mesh_fleet.db")
    s.add_argument("--token", default="")
    s.add_argument("--dashboard", default="")
    s.add_argument("--verbose", action="store_true")
    s.set_defaults(func=cmd_serve)

    a = sub.add_parser("agent", help="install this machine into the fleet")
    a.add_argument("--server", default="http://localhost:8080")
    a.add_argument("--token", default="")
    a.add_argument("--interval", type=float, default=30.0)
    a.add_argument("--location", default="off", help="off | auto | City")
    a.add_argument("--consent-location", action="store_true")
    a.set_defaults(func=cmd_agent)

    sd = sub.add_parser("seed", help="populate a registry with a synthetic fleet")
    sd.add_argument("--server", default="http://localhost:8080")
    sd.add_argument("--token", default="")
    sd.add_argument("--nodes", type=int, default=240)
    sd.add_argument("--seed", type=int, default=42)
    sd.set_defaults(func=cmd_seed)

    sc = sub.add_parser("scan", help="what can this machine or fleet actually train?")
    sc.add_argument("--fleet", type=int, default=0)
    sc.add_argument("--file", default="")
    sc.add_argument("--seed", type=int, default=7)
    sc.set_defaults(func=cmd_scan)

    u = sub.add_parser("up", help="start the coordinator and dashboard in one command")
    u.add_argument("--host", default="127.0.0.1")
    u.add_argument("--port", type=int, default=8080)
    u.add_argument("--db", default="mesh_fleet.db")
    u.add_argument("--token", default="")
    u.add_argument("--dashboard", default="")
    u.add_argument("--demo", action="store_true", help="seed a synthetic fleet")
    u.add_argument("--demo-nodes", type=int, default=240)
    u.add_argument("--no-browser", action="store_true")
    u.set_defaults(func=cmd_up)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
