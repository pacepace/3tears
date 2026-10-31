"""Inline the kit and a data file into a page template.

Usage: python3 scripts/build.py <page.src.html> <data.json> <out.html>

The template holds three markers: /*KIT_CSS*/ inside <style>, /*KIT_JS*/ at the top of <script>,
and /*DATA*/ where the page reads its data (const D=/*DATA*/;). The output is one self-contained
HTML file with no external requests except Google Fonts.
"""

import json, pathlib, sys

root = pathlib.Path(__file__).resolve().parent.parent
src, data, out = (pathlib.Path(a) for a in sys.argv[1:4])
page = src.read_text()
for marker in ("/*KIT_CSS*/", "/*KIT_JS*/", "/*DATA*/"):
    if marker not in page:
        sys.exit(f"{src}: missing marker {marker}")
page = (
    page.replace("/*KIT_CSS*/", (root / "kit/kit.css").read_text())
    .replace("/*KIT_JS*/", (root / "kit/kit.js").read_text())
    .replace("/*DATA*/", json.dumps(json.loads(data.read_text()), separators=(",", ":")))
)
out.write_text(page)
print(f"{out} {len(page):,} bytes")
