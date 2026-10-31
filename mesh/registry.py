"""Fleet registry: SQLite storage and aggregation for the dashboard."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from .census import _KNOWN_CARDS
from .types import NodeProfile

DB_PATH = Path("mesh_fleet.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
  node_id TEXT PRIMARY KEY,
  install_ts REAL, last_seen REAL,
  os TEXT, os_version TEXT, arch TEXT,
  cpu_model TEXT, cpu_cores INTEGER, cpu_mhz REAL,
  ram_total_gb REAL, ram_available_gb REAL,
  disk_total_gb REAL, disk_free_gb REAL,
  gpu_count INTEGER, gpu_name TEXT, gpu_vram_gb REAL,
  gpu_driver TEXT, cuda TEXT, compute_cap TEXT,
  uplink_mbps REAL, downlink_mbps REAL, tflops_measured REAL,
  country TEXT, region TEXT, city TEXT, lat REAL, lon REAL,
  consent_location INTEGER DEFAULT 0,
  app_version TEXT, status TEXT, tokens_done REAL DEFAULT 0,
  gpus_json TEXT
);
CREATE TABLE IF NOT EXISTS samples (
  node_id TEXT, ts REAL, ram_available_gb REAL,
  gpu_util REAL, gpu_temp_c REAL, power_w REAL
);
CREATE INDEX IF NOT EXISTS idx_last_seen ON nodes(last_seen);
"""

FIELDS = [
    "node_id", "install_ts", "last_seen", "os", "os_version", "arch",
    "cpu_model", "cpu_cores", "cpu_mhz", "ram_total_gb", "ram_available_gb",
    "disk_total_gb", "disk_free_gb", "gpu_count", "gpu_name", "gpu_vram_gb",
    "gpu_driver", "cuda", "compute_cap", "uplink_mbps", "downlink_mbps",
    "tflops_measured", "country", "region", "city", "lat", "lon",
    "consent_location", "app_version", "status", "tokens_done",
]

ONLINE_WINDOW = 180.0     # seconds since last heartbeat


class Registry:
    def __init__(self, path: str | Path = DB_PATH):
        self.path = str(path)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.commit()

    # ------------------------------------------------------------- writes
    def upsert(self, report: dict) -> None:
        row = {k: report.get(k) for k in FIELDS}
        row["consent_location"] = int(bool(report.get("consent_location")))
        row["gpus_json"] = json.dumps(report.get("extra", {}).get("gpus", []))
        cols = list(row)
        vals = list(row.values())
        placeholders = ",".join("?" * len(cols))
        self.db.execute(
            f"INSERT INTO nodes ({','.join(cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT(node_id) DO UPDATE SET "
            + ",".join(f"{c}=excluded.{c}" for c in cols if c != "node_id"),
            vals)
        self.db.commit()

    def heartbeat(self, node_id: str, patch: dict) -> bool:
        patch = dict(patch)
        patch["last_seen"] = time.time()
        allowed = [c for c in patch if c in FIELDS]
        if not allowed:
            return False
        self.db.execute(
            f"UPDATE nodes SET {','.join(f'{c}=?' for c in allowed)} WHERE node_id=?",
            [patch[c] for c in allowed] + [node_id])
        self.db.commit()
        if "ram_available_gb" in patch:
            self.db.execute(
                "INSERT INTO samples VALUES (?,?,?,?,?,?)",
                (node_id, time.time(), patch.get("ram_available_gb"),
                 patch.get("gpu_util"), patch.get("gpu_temp_c"), patch.get("power_w")))
            self.db.commit()
        return True

    # -------------------------------------------------------------- reads
    def rows(self) -> list[sqlite3.Row]:
        return list(self.db.execute("SELECT * FROM nodes"))

    def recent(self, limit: int = 40) -> list[dict]:
        cur = self.db.execute(
            "SELECT * FROM nodes ORDER BY last_seen DESC LIMIT ?", (limit,))
        return [dict(r) for r in cur]

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]

    # --------------------------------------------------------- aggregation
    def stats(self) -> dict:
        now = time.time()
        rows = self.rows()
        online = [r for r in rows if now - (r["last_seen"] or 0) <= ONLINE_WINDOW]
        active_24h = [r for r in rows if now - (r["last_seen"] or 0) <= 86400]
        new_7d = [r for r in rows if now - (r["install_ts"] or 0) <= 7 * 86400]

        tiers: dict[str, int] = {}
        models: dict[str, dict] = {}
        cities: dict[str, dict] = {}
        oses: dict[str, int] = {}
        for r in rows:
            vram = r["gpu_vram_gb"] or 0
            key = f"{int(vram)}" if vram >= 1 else "cpu"
            tiers[key] = tiers.get(key, 0) + 1
            name = r["gpu_name"] or "no GPU"
            m = models.setdefault(name, {"name": name, "count": 0, "vram": vram,
                                         "online": 0})
            m["count"] += 1
            if now - (r["last_seen"] or 0) <= ONLINE_WINDOW:
                m["online"] += 1
            if r["city"]:
                c = cities.setdefault(r["city"], {
                    "city": r["city"], "region": r["region"],
                    "country": r["country"], "lat": r["lat"], "lon": r["lon"],
                    "count": 0, "online": 0})
                c["count"] += 1
                if now - (r["last_seen"] or 0) <= ONLINE_WINDOW:
                    c["online"] += 1
            oses[r["os"] or "?"] = oses.get(r["os"] or "?", 0) + 1

        # Installs per day, last 14 days. Real signal from install_ts --
        # unlike an "online over 24h" curve, which is flat for a fleet whose
        # nodes were up all day and therefore tells you nothing.
        daily = []
        for i in range(13, -1, -1):
            lo = now - (i + 1) * 86400
            hi = now - i * 86400
            daily.append({
                "day": time.strftime("%d %b", time.localtime(hi)),
                "installs": sum(1 for r in rows if lo < (r["install_ts"] or 0) <= hi),
            })

        series = []
        for i in range(23, -1, -1):
            t = now - i * 3600
            n = sum(1 for r in rows
                    if (r["install_ts"] or 0) <= t <= (r["last_seen"] or 0) + 3600)
            series.append({"t": t, "online": n})

        return {
            "generated_at": now,
            "totals": {
                "installs": len(rows),
                "online": len(online),
                "active_24h": len(active_24h),
                "new_7d": len(new_7d),
                "consented_location": sum(1 for r in rows if r["consent_location"]),
            },
            "pooled": {
                "vram_gb": round(sum(r["gpu_vram_gb"] or 0 for r in rows), 0),
                "ram_gb": round(sum(r["ram_total_gb"] or 0 for r in rows), 0),
                "disk_free_tb": round(sum(r["disk_free_gb"] or 0 for r in rows) / 1000, 1),
                "ram_available_gb": round(sum(r["ram_available_gb"] or 0 for r in rows), 0),
            },
            "tiers": dict(sorted(tiers.items(), key=lambda kv: (kv[0] == "cpu", kv[0]))),
            "models": sorted(models.values(), key=lambda m: -m["count"])[:8],
            "cities": sorted(cities.values(), key=lambda c: -c["count"])[:12],
            "os": oses,
            "series": series,
            "daily_installs": daily,
        }

    # ------------------------------------------------------------ planner
    def as_profiles(self) -> list[NodeProfile]:
        out = []
        for r in self.rows():
            vram = r["gpu_vram_gb"] or 0
            name = r["gpu_name"] or ""
            tf = 0.02
            for card, (v, t) in _KNOWN_CARDS.items():
                if card in name:
                    tf = t
                    break
            else:
                tf = max(2.0, vram * 6) if vram else 0.02
            out.append(NodeProfile(
                node_id=r["node_id"][:8],
                vram_gb=vram,
                tflops_bf16=tf,
                uplink_mbps=r["uplink_mbps"] or 50.0,
                downlink_mbps=r["downlink_mbps"] or 100.0,
                reliability=0.85,
                thermal_headroom=0.7,
                kind="gpu" if vram else "cpu",
            ))
        return out
