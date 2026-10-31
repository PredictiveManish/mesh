"""Start the API, seed a fleet, capture a snapshot for the standalone dashboard."""
import json, sys, threading, time, urllib.request
from pathlib import Path
sys.path.insert(0, ".")

from mesh.registry import Registry
from mesh.server import make_handler
from http.server import ThreadingHTTPServer
from mesh import seed as seedmod

DB = "/tmp/mesh_demo.db"
PORT = 8099
Path(DB).unlink(missing_ok=True)

reg = Registry(DB)
handler = make_handler(reg, "", Path("dashboard/index.html"))
httpd = ThreadingHTTPServer(("127.0.0.1", PORT), handler)
threading.Thread(target=httpd.serve_forever, daemon=True).start()
time.sleep(0.3)

base = f"http://127.0.0.1:{PORT}"
n = seedmod.seed(base, 240, "", 42)
print("seeded", n)

def get(path):
    with urllib.request.urlopen(base + path, timeout=10) as r:
        return json.loads(r.read())

stats = get("/api/v1/stats")
cap_full = get("/api/v1/capacity?params=0.5&mode=full&model=ref-0.5B")
cap_lora = get("/api/v1/capacity?params=8&mode=lora&model=ref-8B")
cap_big = get("/api/v1/capacity?params=1.2&mode=full&model=ref-1.2B")
nodes = get("/api/v1/nodes?limit=30")["nodes"]

ladder = []
for size in (0.3, 0.5, 1.2, 3.0, 8.0, 70.0):
    c = get(f"/api/v1/capacity?params={size}&mode=full")
    ladder.append({"size": size, "label": f"{size:g}B", "mode": "full",
                   "replicas": c["replicas"], "excluded": c["excluded"],
                   "tps": c["tokens_per_s"], "eta_days": c["eta_days"]})

snap = {"stats": stats, "capacity": {"full_0_5B": cap_full, "lora_8B": cap_lora,
                                     "full_1_2B": cap_big},
        "nodes": nodes, "ladder": ladder}
Path("dashboard").mkdir(exist_ok=True)
Path("dashboard/snapshot.json").write_text(json.dumps(snap, indent=1))

t = stats["totals"]; p = stats["pooled"]
print(f"installs {t['installs']}  online {t['online']}  active24h {t['active_24h']}")
print(f"pooled vram {p['vram_gb']} GB  ram {p['ram_gb']} GB  disk {p['disk_free_tb']} TB")
print("tiers", stats["tiers"])
print("models", [(m['name'][:28], m['count']) for m in stats['models'][:5]])
print("cities", [(c['city'], c['count']) for c in stats['cities'][:5]])
print("capacity 0.5B full:", cap_full)
print("capacity 8B lora:", cap_lora)
print("capacity 1.2B full:", cap_big["replicas"], cap_big["notes"][:1])
httpd.shutdown()
