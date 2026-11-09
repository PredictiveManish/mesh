"""Coordinator HTTP surface: telemetry ingest and the dashboard."""
from __future__ import annotations

import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from .planner import plan
from .registry import Registry
from .types import JobSpec

DASHBOARD = Path(__file__).resolve().parent.parent / "dashboard" / "index.html"

# Node reports are a few hundred bytes; a megabyte is already generous.
MAX_BODY_BYTES = 1024 * 1024
MAX_LIMIT = 500
# The dashboard is served from this origin, so cross-origin access is off
# unless an operator asks for it explicitly.
CORS_ORIGIN = os.environ.get("MESH_CORS_ORIGIN", "")


def _clamp(value, default, maximum):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(parsed, maximum))


def make_handler(reg: Registry, token: str, dashboard: Path):
    class Handler(BaseHTTPRequestHandler):
        server_version = "mesh/0.1"

        def log_message(self, fmt, *args):      # quiet by default
            if getattr(self.server, "verbose", False):
                print(f"  [http] {self.address_string()} {fmt % args}")

        # ---------------------------------------------------------- helpers
        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            if CORS_ORIGIN:
                self.send_header("Access-Control-Allow-Origin", CORS_ORIGIN)
            self.end_headers()
            self.wfile.write(body)

        def _html(self, path: Path):
            if not path.exists():
                self.send_error(404, "dashboard not built")
                return
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorised(self) -> bool:
            if not token:
                return True
            return self.headers.get("X-Mesh-Token", "") == token

        def _body(self):
            """Read a JSON body, or respond and return None.

            Content-Length is attacker-controlled: a large value used to be
            read into memory in one go, and a non-numeric one raised an
            uncaught ValueError.
            """
            raw_length = self.headers.get("Content-Length", "0")
            try:
                n = int(raw_length or 0)
            except (TypeError, ValueError):
                self._json({"error": "invalid Content-Length"}, 400)
                return None
            if n <= 0:
                return {}
            if n > MAX_BODY_BYTES:
                self._json({"error": f"body exceeds {MAX_BODY_BYTES} bytes"}, 413)
                return None
            try:
                return json.loads(self.rfile.read(n))
            except Exception:
                self._json({"error": "body is not valid JSON"}, 400)
                return None

        # ------------------------------------------------------------- GET
        def do_GET(self):
            u = urlparse(self.path)
            if u.path in ("/", "/index.html", "/dashboard"):
                return self._html(dashboard)
            if u.path == "/api/v1/health":
                return self._json({"ok": True, "ts": time.time(),
                                   "nodes": reg.count()})
            if u.path == "/api/v1/stats":
                return self._json(reg.stats())
            if u.path == "/api/v1/nodes":
                q = parse_qs(u.query)
                limit = _clamp(q.get("limit", ["40"])[0], 40, MAX_LIMIT)
                return self._json({"nodes": reg.recent(limit)})
            if u.path == "/api/v1/capacity":
                q = parse_qs(u.query)
                params = float(q.get("params", ["0.5"])[0]) * 1e9
                mode = q.get("mode", ["full"])[0]
                job = JobSpec(model=q.get("model", ["reference"])[0],
                              params=params, target_tokens=20e9,
                              train_mode=mode, compression="int8")
                profiles = reg.as_profiles()
                if not profiles:
                    return self._json({"replicas": 0, "excluded": 0,
                                       "notes": ["no nodes registered yet"]})
                p = plan(job, profiles)
                return self._json({
                    "replicas": p.replica_count,
                    "excluded": len(p.excluded),
                    "sync_every": p.sync_every,
                    "tokens_per_s": round(p.expected_tps),
                    "round_seconds": round(p.round_seconds, 1),
                    "eta_days": round(p.eta_hours / 24, 1),
                    "bottleneck": p.bottleneck,
                    "notes": p.notes,
                })
            self.send_error(404)

        # ------------------------------------------------------------ POST
        def do_POST(self):
            u = urlparse(self.path)
            if not self._authorised():
                return self._json({"error": "unauthorised"}, 401)
            body = self._body()
            if body is None:
                return
            if u.path == "/api/v1/nodes":
                if not body.get("node_id"):
                    return self._json({"error": "node_id required"}, 400)
                body.setdefault("last_seen", time.time())
                body.setdefault("install_ts", body["last_seen"])
                reg.upsert(body)
                return self._json({"ok": True, "registered": body["node_id"],
                                   "fleet_size": reg.count()})
            if u.path == "/api/v1/heartbeat":
                nid = body.get("node_id")
                if not nid:
                    return self._json({"error": "node_id required"}, 400)
                ok = reg.heartbeat(nid, body)
                return self._json({"ok": ok})
            self.send_error(404)

    return Handler


def serve(db: str = "mesh_fleet.db", host: str = "0.0.0.0", port: int = 8080,
          token: str = "", dashboard: str | Path | None = None,
          verbose: bool = False) -> None:
    reg = Registry(db)
    handler = make_handler(reg, token, Path(dashboard) if dashboard else DASHBOARD)
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.verbose = verbose
    print(f"\n  mesh coordinator listening on http://{host}:{port}")
    print(f"  dashboard  http://localhost:{port}/")
    print(f"  api        /api/v1/stats  /api/v1/nodes  /api/v1/capacity")
    print(f"  registry   {db}  ({reg.count()} nodes)\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")
