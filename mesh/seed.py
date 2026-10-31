"""Populate a registry with a plausible fleet, for demos and load tests.

Not part of the product. Clearly synthetic: real installs come from
mesh.agent.
"""
from __future__ import annotations

import json
import random
import time
import urllib.request
import uuid

CARDS = [
    # (name, vram_gb, tflops, weight)
    ("NVIDIA GeForce GTX 1650", 4.0, 6.0, 0.12),
    ("NVIDIA GeForce RTX 3050 Laptop", 4.0, 18.0, 0.09),
    ("NVIDIA GeForce RTX 2060", 6.0, 25.0, 0.08),
    ("NVIDIA GeForce RTX 3060 Laptop", 6.0, 25.0, 0.15),
    ("NVIDIA GeForce RTX 4060 Laptop", 8.0, 60.0, 0.21),
    ("NVIDIA GeForce RTX 3070", 8.0, 80.0, 0.08),
    ("NVIDIA GeForce RTX 4070", 8.0, 80.0, 0.06),
    ("NVIDIA GeForce RTX 4090 Laptop", 16.0, 165.0, 0.05),
    ("NVIDIA GeForce RTX 5090", 24.0, 210.0, 0.03),
    ("Apple M2 Pro", 16.0, 55.0, 0.07),
    ("Apple M4", 16.0, 65.0, 0.04),
    ("", 0.0, 0.0, 0.02),
]

CITIES = [
    ("Bengaluru", "Karnataka", "IN", 12.97, 77.59, 0.22),
    ("Mumbai", "Maharashtra", "IN", 19.08, 72.88, 0.16),
    ("Delhi", "Delhi", "IN", 28.61, 77.21, 0.14),
    ("Hyderabad", "Telangana", "IN", 17.39, 78.49, 0.12),
    ("Chennai", "Tamil Nadu", "IN", 13.08, 80.27, 0.09),
    ("Pune", "Maharashtra", "IN", 18.52, 73.86, 0.08),
    ("Kolkata", "West Bengal", "IN", 22.57, 88.36, 0.05),
    ("Ahmedabad", "Gujarat", "IN", 23.02, 72.57, 0.04),
    ("Jaipur", "Rajasthan", "IN", 26.91, 75.79, 0.03),
    ("Lucknow", "Uttar Pradesh", "IN", 26.85, 80.95, 0.03),
    ("Kochi", "Kerala", "IN", 9.93, 76.27, 0.02),
    ("Guwahati", "Assam", "IN", 26.14, 91.74, 0.02),
]

OSES = [("Windows", "11", 0.62), ("Windows", "10", 0.14),
        ("Linux", "6.8", 0.14), ("macOS", "15.2", 0.10)]


def _pick(rng, table, key=0):
    r = rng.random()
    acc = 0.0
    for row in table:
        acc += row[-1]
        if r <= acc:
            return row
    return table[-1]


def make_node(rng: random.Random, now: float) -> dict:
    card = _pick(rng, CARDS)
    city = _pick(rng, CITIES)
    os_name, os_ver, _ = _pick(rng, OSES)

    age_days = rng.expovariate(1 / 21.0)          # installs skew recent
    install_ts = now - min(age_days, 60) * 86400

    roll = rng.random()
    if roll < 0.62:
        last_seen = now - rng.uniform(0, 170)      # online
    elif roll < 0.87:
        last_seen = now - rng.uniform(200, 86400)  # seen today
    else:
        last_seen = now - rng.uniform(86400, 5 * 86400)
    last_seen = max(last_seen, install_ts)

    ram_total = rng.choice([8.0, 16.0, 16.0, 32.0, 32.0])
    consent = rng.random() < 0.72
    return {
        "node_id": str(uuid.uuid4()),
        "install_ts": install_ts,
        "last_seen": last_seen,
        "os": os_name, "os_version": os_ver,
        "arch": "arm64" if "Apple" in card[0] else "x86_64",
        "cpu_model": ("Apple M-series" if "Apple" in card[0]
                      else rng.choice(["Intel Core i5-12450H", "AMD Ryzen 5 5600H",
                                       "Intel Core i7-12700H", "AMD Ryzen 7 5800H"])),
        "cpu_cores": rng.choice([8, 12, 16]),
        "cpu_mhz": rng.choice([2400.0, 2900.0, 3200.0]),
        "ram_total_gb": ram_total,
        "ram_available_gb": round(ram_total * rng.uniform(0.18, 0.55), 1),
        "disk_total_gb": rng.choice([512.0, 512.0, 1024.0]),
        "disk_free_gb": round(rng.uniform(40, 400), 0),
        "gpu_count": 0 if not card[0] else 1,
        "gpu_name": card[0],
        "gpu_vram_gb": card[1],
        "gpu_driver": rng.choice(["560.94", "566.03", "550.54"]) if card[0].startswith("NVIDIA") else "1.0",
        "cuda": "12.8" if card[0].startswith("NVIDIA") else "",
        "compute_cap": "8.9" if card[0].startswith("NVIDIA") else "metal",
        "uplink_mbps": round(rng.choice([rng.uniform(15, 60), rng.uniform(60, 300)]), 1),
        "downlink_mbps": round(rng.uniform(150, 900), 1),
        "tflops_measured": round(card[2] * rng.uniform(0.45, 0.75), 1),
        "country": city[2] if consent else "",
        "region": city[1] if consent else "",
        "city": city[0] if consent else "",
        "lat": round(city[3] + rng.uniform(-0.12, 0.12), 2) if consent else None,
        "lon": round(city[4] + rng.uniform(-0.12, 0.12), 2) if consent else None,
        "consent_location": consent,
        "app_version": "0.1.0",
        "status": "training" if (last_seen > now - 170 and rng.random() < 0.35) else "idle",
        "tokens_done": round(rng.uniform(0, 2e8), 0) if rng.random() < 0.4 else 0.0,
        "extra": {"gpus": [{"name": card[0], "vram_gb": card[1],
                            "driver": "560.94", "compute_cap": "8.9"}] if card[0] else []},
    }


def seed(server: str, n: int, token: str = "", seed_val: int = 42) -> int:
    rng = random.Random(seed_val)
    now = time.time()
    ok = 0
    for _ in range(n):
        payload = make_node(rng, now)
        req = urllib.request.Request(
            f"{server}/api/v1/nodes", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     **({"X-Mesh-Token": token} if token else {})},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                r.read()
                ok += 1
        except Exception as exc:
            print(f"  seed failed: {exc}")
            break
    return ok
