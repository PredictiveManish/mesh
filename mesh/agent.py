"""The installable client: probe this machine, register, keep heartbeating."""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

from .telemetry import probe


def post(url: str, payload: dict, token: str = "", timeout: float = 10.0) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 **({"X-Mesh-Token": token} if token else {})},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def run(args) -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)   # CLI output must not buffer
    except Exception:
        pass
    rep = probe(location=args.location,
                consent_location=args.consent_location).to_dict()

    print("\n  This machine, as the fleet will see it\n")
    print(f"    node       {rep['node_id'][:8]}...   (random, not a hardware ID)")
    print(f"    os         {rep['os']} {rep['os_version']} ({rep['arch']})")
    print(f"    cpu        {rep['cpu_model'][:48]}  x{rep['cpu_cores']}")
    print(f"    memory     {rep['ram_total_gb']:.1f} GB total, "
          f"{rep['ram_available_gb']:.1f} GB available")
    print(f"    storage    {rep['disk_free_gb']:.0f} GB free of {rep['disk_total_gb']:.0f} GB")
    if rep["gpu_count"]:
        print(f"    gpu        {rep['gpu_name']}  {rep['gpu_vram_gb']:.1f} GB VRAM"
              f"  driver {rep['gpu_driver']}  cuda {rep['cuda'] or '-'}")
    else:
        print("    gpu        none detected")
    if rep["city"]:
        print(f"    location   {rep['city']}, {rep['region']}, {rep['country']}"
              f"  (~{rep['lat']}, {rep['lon']}, city-level only)")
    else:
        print("    location   not shared")
    print()

    try:
        res = post(f"{args.server}/api/v1/nodes", rep, args.token)
    except Exception as exc:
        print(f"  could not reach {args.server}: {exc}", file=sys.stderr)
        return 1
    print(f"  registered. fleet size now {res.get('fleet_size')}\n")

    print("  heartbeating every "
          f"{args.interval:g}s. Ctrl-C to stop.\n")
    try:
        while True:
            time.sleep(args.interval)
            fresh = probe(location="off", consent_location=False)
            post(f"{args.server}/api/v1/heartbeat", {
                "node_id": rep["node_id"],
                "ram_available_gb": fresh.ram_available_gb,
                "disk_free_gb": fresh.disk_free_gb,
            }, args.token)
    except KeyboardInterrupt:
        print("  stopped")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mesh.agent")
    ap.add_argument("--server", default="http://localhost:8080")
    ap.add_argument("--token", default="")
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--location", default="off",
                    help="off | auto | City")
    ap.add_argument("--consent-location", action="store_true",
                    help="explicitly allow coarse (city-level) location sharing")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
