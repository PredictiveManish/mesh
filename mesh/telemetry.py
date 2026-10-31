"""Device telemetry: what a machine actually is, and where it is.

Everything here is collected on the user's machine and sent to the
coordinator. That makes it personal data in most jurisdictions, so three
rules apply and are enforced in code, not just documented:

  1. Location is opt-in. Default is "off" and the field is simply absent.
  2. Location is coarse: city level, coordinates rounded to ~1 km.
  3. Nothing identifying is sent. node_id is a random UUID minted on first
     run; there is no email, no hostname, no MAC, no IP stored.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

STATE_DIR = Path.home() / ".mesh"
IDENTITY_FILE = STATE_DIR / "identity.json"

# Coarse reference points. A production build resolves this from the
# connecting IP with a licensed GeoIP database; the prototype ships a static
# table so the whole system runs offline and reproducibly.
CITY_TABLE = {
    "Bengaluru": (12.97, 77.59, "Karnataka", "IN"),
    "Mumbai": (19.08, 72.88, "Maharashtra", "IN"),
    "Delhi": (28.61, 77.21, "Delhi", "IN"),
    "Hyderabad": (17.39, 78.49, "Telangana", "IN"),
    "Chennai": (13.08, 80.27, "Tamil Nadu", "IN"),
    "Pune": (18.52, 73.86, "Maharashtra", "IN"),
    "Kolkata": (22.57, 88.36, "West Bengal", "IN"),
    "Ahmedabad": (23.02, 72.57, "Gujarat", "IN"),
    "Jaipur": (26.91, 75.79, "Rajasthan", "IN"),
    "Lucknow": (26.85, 80.95, "Uttar Pradesh", "IN"),
    "Kochi": (9.93, 76.27, "Kerala", "IN"),
    "Chandigarh": (30.73, 76.78, "Punjab", "IN"),
    "Indore": (22.72, 75.86, "Madhya Pradesh", "IN"),
    "Guwahati": (26.14, 91.74, "Assam", "IN"),
    "Bhubaneswar": (20.30, 85.82, "Odisha", "IN"),
    "Coimbatore": (11.02, 76.96, "Tamil Nadu", "IN"),
    "Visakhapatnam": (17.69, 83.22, "Andhra Pradesh", "IN"),
    "Nagpur": (21.15, 79.09, "Maharashtra", "IN"),
    "Singapore": (1.35, 103.82, "Singapore", "SG"),
    "Dubai": (25.20, 55.27, "Dubai", "AE"),
    "London": (51.51, -0.13, "England", "GB"),
    "Berlin": (52.52, 13.40, "Berlin", "DE"),
    "San Francisco": (37.77, -122.42, "California", "US"),
    "Toronto": (43.65, -79.38, "Ontario", "CA"),
}


@dataclass
class DeviceReport:
    """One snapshot of a machine. Sent on install and every heartbeat."""

    node_id: str
    install_ts: float
    last_seen: float
    os: str = ""
    os_version: str = ""
    arch: str = ""
    cpu_model: str = ""
    cpu_cores: int = 0
    cpu_mhz: float = 0.0
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    disk_total_gb: float = 0.0
    disk_free_gb: float = 0.0
    gpu_count: int = 0
    gpu_name: str = ""
    gpu_vram_gb: float = 0.0
    gpu_driver: str = ""
    cuda: str = ""
    compute_cap: str = ""
    uplink_mbps: float = 0.0
    downlink_mbps: float = 0.0
    tflops_measured: float = 0.0
    country: str = ""
    region: str = ""
    city: str = ""
    lat: float | None = None
    lon: float | None = None
    consent_location: bool = False
    app_version: str = "0.1.0"
    status: str = "idle"
    tokens_done: float = 0.0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ identity
def node_identity() -> str:
    """A random UUID minted once per install. No hardware fingerprinting."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if IDENTITY_FILE.exists():
        try:
            return json.loads(IDENTITY_FILE.read_text())["node_id"]
        except Exception:
            pass
    nid = str(uuid.uuid4())
    IDENTITY_FILE.write_text(json.dumps({"node_id": nid, "created": time.time()}))
    return nid


# ------------------------------------------------------------------ probing
def _cpu_model() -> str:
    try:
        text = Path("/proc/cpuinfo").read_text()
        for line in text.splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _ram() -> tuple[float, float]:
    try:
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            info[k.strip()] = float(v.strip().split()[0]) / 1e6  # kB -> GB
        return round(info.get("MemTotal", 0), 1), round(info.get("MemAvailable", 0), 1)
    except Exception:
        return 0.0, 0.0


def _gpus() -> list[dict]:
    """Ask the driver, not torch: torch may not be installed on a fresh box."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip()
        gpus = []
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append({
                    "name": parts[0],
                    "vram_gb": round(float(parts[1]) / 1024, 1),
                    "driver": parts[2],
                    "compute_cap": parts[3] if len(parts) > 3 else "",
                })
        if gpus:
            return gpus
    except Exception:
        pass

    # Apple Silicon exposes unified memory through the OS, not a driver.
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            brand = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                   capture_output=True, text=True, timeout=3).stdout.strip()
            total, _ = _ram()
            return [{"name": brand or "Apple Silicon", "vram_gb": round(total * 0.7, 1),
                     "driver": platform.mac_ver()[0], "compute_cap": "metal"}]
        except Exception:
            pass
    return []


def _cuda_version() -> str:
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5).stdout
        for line in out.splitlines():
            if "CUDA Version" in line:
                return line.split("CUDA Version:")[1].split()[0]
    except Exception:
        pass
    return ""


def resolve_location(mode: str, consent: bool) -> dict:
    """mode: off | auto | 'City' | 'City,Region,Country'."""
    if not consent or mode == "off":
        return {}
    city = mode
    if mode == "auto":
        # Prototype: the operator maps the connecting IP to a city at the
        # edge, or the agent is launched with an explicit city. We never
        # store the IP itself.
        city = os.environ.get("MESH_CITY", "")
        if not city:
            return {}
    city = city.split(",")[0].strip()
    row = CITY_TABLE.get(city)
    if not row:
        return {"city": city, "region": "", "country": "", "lat": None, "lon": None}
    lat, lon, region, country = row
    return {"city": city, "region": region, "country": country,
            "lat": round(lat, 2), "lon": round(lon, 2)}


def probe(location: str = "off", consent_location: bool = False,
          measure_net: bool = False) -> DeviceReport:
    gpus = _gpus()
    ram_total, ram_avail = _ram()
    try:
        du = shutil.disk_usage(str(Path.home()))
        disk_total, disk_free = du.total / 1e9, du.free / 1e9
    except Exception:
        disk_total, disk_free = 0.0, 0.0

    rep = DeviceReport(
        node_id=node_identity(),
        install_ts=time.time(),
        last_seen=time.time(),
        os=platform.system(),
        os_version=platform.release(),
        arch=platform.machine(),
        cpu_model=_cpu_model()[:80],
        cpu_cores=os.cpu_count() or 0,
        ram_total_gb=ram_total,
        ram_available_gb=ram_avail,
        disk_total_gb=round(disk_total, 1),
        disk_free_gb=round(disk_free, 1),
        gpu_count=len(gpus),
        gpu_name=gpus[0]["name"] if gpus else "",
        gpu_vram_gb=gpus[0]["vram_gb"] if gpus else 0.0,
        gpu_driver=gpus[0]["driver"] if gpus else "",
        cuda=_cuda_version(),
        compute_cap=gpus[0]["compute_cap"] if gpus else "",
        consent_location=consent_location,
        extra={"gpus": gpus},
    )
    rep.__dict__.update(resolve_location(location, consent_location))
    return rep
