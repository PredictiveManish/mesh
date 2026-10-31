"""Bake the dashboard screenshot into the landing page."""
import base64, sys
from pathlib import Path

root = Path(__file__).resolve().parent
img = base64.b64encode((root / "site" / "shot.jpg").read_bytes()).decode()
tpl = (root / "site" / "template.html").read_text()
out = tpl.replace("__SHOT__", "data:image/jpeg;base64," + img)
(root / "site" / "index.html").write_text(out)
print(f"built site/index.html ({len(out)//1024} KB)")
