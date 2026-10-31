"""Telemetry, registry and API tests -- including the privacy guarantees."""
import json, os, sys, threading, time, urllib.request, urllib.error
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mesh.registry import Registry
from mesh.server import make_handler
from mesh.telemetry import probe, resolve_location, node_identity


def test_probe_reports_the_fields_a_planner_needs():
    r = probe(location="off").to_dict()
    for k in ("node_id", "os", "cpu_cores", "ram_total_gb", "ram_available_gb",
              "disk_total_gb", "disk_free_gb", "gpu_count", "gpu_name",
              "gpu_vram_gb"):
        assert k in r, k
    assert r["ram_total_gb"] > 0
    assert r["disk_free_gb"] > 0


def test_location_is_absent_without_consent():
    assert resolve_location("auto", consent=False) == {}
    assert resolve_location("Bengaluru", consent=False) == {}
    r = probe(location="Bengaluru", consent_location=False).to_dict()
    assert r["city"] == "" and r["lat"] is None


def test_location_is_coarse_when_consented():
    loc = resolve_location("Bengaluru", consent=True)
    assert loc["city"] == "Bengaluru" and loc["country"] == "IN"
    # rounded to 2 decimals: ~1 km, not a street address
    assert loc["lat"] == round(loc["lat"], 2)
    assert loc["lon"] == round(loc["lon"], 2)


def test_node_id_is_random_not_hardware():
    nid = node_identity()
    assert len(nid) == 36 and nid.count("-") == 4
    assert nid == node_identity()          # stable per install


def test_registry_aggregates_correctly(tmp_path):
    reg = Registry(tmp_path / "t.db")
    now = time.time()
    for i in range(10):
        reg.upsert({
            "node_id": f"n{i}", "install_ts": now - i * 3600, "last_seen": now - i * 60,
            "os": "Linux", "gpu_name": "RTX 4060", "gpu_vram_gb": 8.0,
            "ram_total_gb": 16.0, "ram_available_gb": 8.0, "disk_free_gb": 100.0,
            "city": "Pune" if i % 2 else "", "consent_location": bool(i % 2),
            "lat": 18.52 if i % 2 else None, "lon": 73.86 if i % 2 else None,
        })
    s = reg.stats()
    assert s["totals"]["installs"] == 10
    # nodes were staggered 60s apart; the online window is 180s, so 3 qualify
    assert s["totals"]["online"] == 3
    assert s["tiers"] == {"8": 10}
    assert s["pooled"]["vram_gb"] == 80.0
    assert len(s["cities"]) == 1 and s["cities"][0]["count"] == 5
    assert len(s["daily_installs"]) == 14


def test_api_requires_token_for_writes(tmp_path):
    reg = Registry(tmp_path / "t.db")
    handler = make_handler(reg, "secret", Path("/nonexistent"))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    time.sleep(0.2)

    body = json.dumps({"node_id": "x"}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/api/v1/nodes",
                                 data=body, headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=5)
        assert False, "should have been rejected"
    except urllib.error.HTTPError as e:
        assert e.code == 401

    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/v1/nodes", data=body,
        headers={"Content-Type": "application/json", "X-Mesh-Token": "secret"})
    with urllib.request.urlopen(req, timeout=5) as r:
        assert json.loads(r.read())["ok"] is True
    httpd.shutdown()
