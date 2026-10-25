"""Render a built page at desktop and phone widths, light and dark, and report what breaks.

Usage: python3 scripts/check.py <page.html> [--out DIR]
Needs: pip install playwright && playwright install chromium

Checks (GENERATOR.md §8, items 1-3): console and page errors; horizontal overflow
(scrollWidth > viewport); elements that extend past the right edge outside a scrolling table;
SVG text boxes that overlap each other inside one chart. Writes a full-page PNG per run.
Exit code 1 when any check fails.
"""

import asyncio, pathlib, sys
from playwright.async_api import async_playwright

PROBE = """() => {
  const vw = innerWidth, out = [];
  for (const e of document.querySelectorAll('body *')) {
    if (e.closest('.tbl') || e.closest('#tip')) continue;
    const r = e.getBoundingClientRect();
    if (r.width && r.right > vw + 1) out.push(`${e.tagName.toLowerCase()} "${(e.textContent||'').trim().slice(0,40)}" right=${Math.round(r.right)}`);
  }
  const overlaps = [];
  document.querySelectorAll('.chart svg').forEach((svg, si) => {
    const boxes = [...svg.querySelectorAll('text')].map(t => ({t: t.textContent, r: t.getBoundingClientRect()})).filter(b => b.r.width > 0 && b.t.trim());
    for (let i = 0; i < boxes.length; i++) for (let j = i + 1; j < boxes.length; j++) {
      const a = boxes[i].r, b = boxes[j].r, pad = -1;
      if (a.left < b.right - pad && a.right > b.left + pad && a.top < b.bottom - pad && a.bottom > b.top + pad)
        overlaps.push(`chart ${si}: "${boxes[i].t.slice(0,24)}" × "${boxes[j].t.slice(0,24)}"`);
    }
  });
  return {scrollWidth: document.documentElement.scrollWidth, past: out.slice(0, 8), overlaps: overlaps.slice(0, 12)};
}"""


async def run(page_path, out_dir):
    failed = False
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        for width in (1200, 390):
            for scheme in ("light", "dark"):
                pg = await browser.new_page(viewport={"width": width, "height": 900}, color_scheme=scheme)
                errors = []
                pg.on("pageerror", lambda e: errors.append(str(e)))
                pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
                await pg.goto(page_path.resolve().as_uri())
                await pg.wait_for_timeout(1200)
                res = await pg.evaluate(PROBE)
                shot = out_dir / f"{page_path.stem}-{width}-{scheme}.png"
                await pg.screenshot(path=str(shot), full_page=True)
                bad = errors or res["scrollWidth"] > width or res["past"] or res["overlaps"]
                failed |= bool(bad)
                print(
                    f"{'FAIL' if bad else 'ok  '} {width}px {scheme:5}  scrollWidth={res['scrollWidth']}  screenshot={shot}"
                )
                for k in ("past", "overlaps"):
                    for line in res[k]:
                        print(f"       {k}: {line}")
                for e in errors:
                    print(f"       error: {e}")
                await pg.close()
        await browser.close()
    return failed


if __name__ == "__main__":
    args = sys.argv[1:]
    page = pathlib.Path(args[0])
    out = pathlib.Path(args[args.index("--out") + 1]) if "--out" in args else page.parent / "screenshots"
    out.mkdir(parents=True, exist_ok=True)
    sys.exit(1 if asyncio.run(run(page, out)) else 0)
