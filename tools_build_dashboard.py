"""Bake the latest snapshot into the dashboard so it works offline."""
import json, sys
from pathlib import Path

root = Path(__file__).resolve().parent
snap = json.loads((root / "dashboard" / "snapshot.json").read_text())
tpl = (root / "dashboard" / "template.html").read_text()
out = tpl.replace("__SNAPSHOT__", json.dumps(snap, separators=(",", ":")))
(root / "dashboard" / "index.html").write_text(out)
print(f"built dashboard/index.html  ({len(out)//1024} KB, "
      f"{snap['stats']['totals']['installs']} nodes)")
