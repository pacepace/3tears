"""Render a built page at desktop, host-frame and phone widths, light and dark, and report what breaks.

Usage: python3 scripts/check.py <page.html> [--out DIR]
Needs: pip install playwright && playwright install chromium

Checks (GENERATOR.md §8, items 1-3): console and page errors; horizontal overflow
(scrollWidth > viewport); elements that extend past the right edge outside a scrolling table;
SVG text boxes that overlap each other inside one chart; a table wider than its frame whose first
column does not stay in view (sticky); a pinned (sticky) column drawn narrower than its floor (its
min-width), or a word in it broken across lines (it wraps at spaces and after a hyphen or a slash; only
a cell classed unbroken, an id-like token, may break anywhere); a pinned cell painted over the page's
sticky nav. A table that fits its frame by crushing its key column does not fit. 688 px is the narrowest
frame a host gives the page (Discodon's iframe at a 1024 px window). Writes a full-page PNG per run.
Exit code 1 when any check fails.
"""

import asyncio, pathlib, sys
from playwright.async_api import async_playwright

PROBE = r"""() => {
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
  const unpinned = [];
  document.querySelectorAll('.tbl').forEach((frame, ti) => {
    const first = frame.querySelector('tr > :first-child');
    if (frame.scrollWidth > frame.clientWidth + 1 && first && getComputedStyle(first).position !== 'sticky')
      unpinned.push(`table ${ti}: "${(first.textContent||'').trim().slice(0,24)}" scrolls with its first column`);
  });
  // a pinned cell holds its floor, and breaks a word across lines only where it is id-like (classed unbroken)
  const crushed = [];
  document.querySelectorAll('.tbl tr > :first-child, .tbl.keys-2 tr > :nth-child(2)').forEach(cell => {
    const style = getComputedStyle(cell);
    if (style.position !== 'sticky' || cell.closest('details:not([open])')) return;
    const width = cell.getBoundingClientRect().width, floor = parseFloat(style.minWidth) || 0;
    const name = `"${(cell.textContent||'').trim().slice(0,24)}"`;
    if (width && width < floor - 0.5) crushed.push(`${name} is ${Math.round(width)} px, under its ${Math.round(floor)} px floor`);
    if (cell.classList.contains('unbroken')) return;
    const walk = document.createTreeWalker(cell, NodeFilter.SHOW_TEXT);
    for (let node = walk.nextNode(); node; node = walk.nextNode()) {
      for (const word of node.data.matchAll(/[^\s\-\/]+[\-\/]?/g)) {
        const range = document.createRange();
        range.setStart(node, word.index); range.setEnd(node, word.index + word[0].length);
        const tops = new Set([...range.getClientRects()].filter(r => r.width).map(r => Math.round(r.top)));
        if (tops.size > 1) { crushed.push(`${name} breaks "${word[0].slice(0,24)}" across ${tops.size} lines`); return; }
      }
    }
  });
  // the page's sticky nav paints over every table that scrolls beneath it
  const under = [], nav = document.querySelector('nav');
  if (nav && getComputedStyle(nav).position === 'sticky') {
    const frames = [...document.querySelectorAll('.tbl')].filter(f => f.querySelector('tr > :first-child') && f.offsetParent);
    for (const frame of frames.slice(0, 12)) {
      const cell = frame.querySelector('tr > :first-child');
      window.scrollTo(0, scrollY + cell.getBoundingClientRect().top - nav.getBoundingClientRect().height / 2);
      const box = nav.getBoundingClientRect(), r = cell.getBoundingClientRect();
      if (r.top > box.bottom || r.bottom < box.top) continue;
      const hit = document.elementFromPoint(r.left + Math.min(8, r.width / 2), (Math.max(box.top, r.top) + Math.min(box.bottom, r.bottom)) / 2);
      if (hit && !nav.contains(hit) && frame.contains(hit))
        under.push(`"${(cell.textContent||'').trim().slice(0,24)}" paints over the nav`);
    }
    window.scrollTo(0, 0);
  }
  return {scrollWidth: document.documentElement.scrollWidth, past: out.slice(0, 8), overlaps: overlaps.slice(0, 12),
          unpinned: unpinned.slice(0, 8), crushed: crushed.slice(0, 8), under: under.slice(0, 8)};
}"""


async def run(page_path, out_dir):
    failed = False
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        for width in (1200, 688, 390):
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
                bad = (
                    errors
                    or res["scrollWidth"] > width
                    or res["past"]
                    or res["overlaps"]
                    or res["unpinned"]
                    or res["crushed"]
                    or res["under"]
                )
                failed |= bool(bad)
                print(
                    f"{'FAIL' if bad else 'ok  '} {width}px {scheme:5}  scrollWidth={res['scrollWidth']}  screenshot={shot}"
                )
                for k in ("past", "overlaps", "unpinned", "crushed", "under"):
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
