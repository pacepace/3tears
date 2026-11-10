"""Render a built page at desktop, host-frame and phone widths, light and dark, and report what breaks.

Usage: python3 scripts/check.py <page.html> [--out DIR]
Needs: pip install playwright && playwright install chromium

Checks (GENERATOR.md §8, items 1-3): console and page errors; horizontal overflow (scrollWidth > viewport);
elements that extend past the right edge outside a scrolling table; SVG text boxes that overlap each other inside
one chart; a chart's label crossed by a drawn axis, zero line or reference line; a table wider than its frame
whose first column does not stay in view (sticky); a pinned (sticky) column drawn narrower than its floor (its
min-width), or a word in it broken across lines (it wraps at spaces and after a hyphen or a slash; only a cell
classed unbroken, an id-like token, may break anywhere); a pinned cell painted over the page's sticky nav. A table
that fits its frame by crushing its key column does not fit. A table wider than its frame with no visible cue: its
frame's right edge must fade while more of it lies past the edge, and not once it is scrolled to its end. A first
screen read at more than one measure: the masthead, the callouts under it, the headline, the lede and the
confidence share one maximum width. Text under 4.5:1 against the ground it is drawn on, and an axis or a chart's
mark (an interval's line) under 3:1 (WCAG 1.4.3 and 1.4.11), computed from the colours as drawn in each theme: a
chart's text against the mark under it where one is, else the nearest painted background. A font size off the
page's five-step scale (12, 14, 16, 20 and 28 px). A table drawn in a box or under a filled header row, or with no
rule under its header: a table is ruled with hairlines, never boxed or tinted. An interval in a table's cell ("0
to 43%", set in its .iv span) or a cell of figures (.figs) broken across lines, named with its table's caption,
heading or name. 1104 px and 688 px are the frames a host gives the page (Discodon's iframe at a 1440
and a 1024 px window). Writes a full-page PNG per run. Exit code 1 when any check fails.
"""

import asyncio, pathlib, sys
from playwright.async_api import async_playwright

PROBE = r"""async () => {
  const vw = innerWidth, out = [];
  for (const e of document.querySelectorAll('body *')) {
    if (e.closest('.tbl') || e.closest('#tip')) continue;
    const r = e.getBoundingClientRect();
    if (r.width && r.right > vw + 1) out.push(`${e.tagName.toLowerCase()} "${(e.textContent||'').trim().slice(0,40)}" right=${Math.round(r.right)}`);
  }
  // a chart named by its kind and the title or caption of the figure holding it, so a failure says which chart
  const chartName = (e, si) => {
    const fig = e.closest('figure'), kind = e.closest('[data-kind]')?.dataset.kind || 'chart';
    const title = (fig?.querySelector('figcaption, .cap, h3, h4')?.textContent || fig?.id || '').trim().slice(0, 40);
    return `chart${si === undefined ? '' : ` ${si}`} (${kind}${title ? ` "${title}"` : ''})`;
  };
  const overlaps = [];
  document.querySelectorAll('.chart svg').forEach((svg, si) => {
    const boxes = [...svg.querySelectorAll('text')].map(t => ({t: t.textContent, r: t.getBoundingClientRect()})).filter(b => b.r.width > 0 && b.t.trim());
    for (let i = 0; i < boxes.length; i++) for (let j = i + 1; j < boxes.length; j++) {
      const a = boxes[i].r, b = boxes[j].r, pad = -1;
      if (a.left < b.right - pad && a.right > b.left + pad && a.top < b.bottom - pad && a.bottom > b.top + pad)
        overlaps.push(`${chartName(svg, si)}: "${boxes[i].t.slice(0,24)}" × "${boxes[j].t.slice(0,24)}"`);
    }
  });
  // a chart's label sits clear of every axis, zero line and reference line drawn across the chart (a kit's chart: one
  // a page's own code draws, with no data-kind, is the page's)
  const crossed = [];
  document.querySelectorAll('.chart[data-kind] svg, .chart-drawn svg').forEach((svg, si) => {
    const boxes = [...svg.querySelectorAll('text')].map(t => ({t: t.textContent, r: t.getBoundingClientRect()})).filter(b => b.r.width > 0 && b.t.trim());
    for (const line of svg.querySelectorAll('line.axis, line.floor')) {
      const r = line.getBoundingClientRect(), pad = 1;
      if (!r.width && !r.height) continue;
      for (const b of boxes) {
        const a = b.r;
        if (r.left < a.right - pad && r.right > a.left + pad && r.top < a.bottom - pad && r.bottom > a.top + pad)
          crossed.push(`${chartName(svg, si)}: a ${line.getAttribute('class')} line crosses "${b.t.slice(0,32)}"`);
      }
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
  // a table wider than its frame fades the frame's right edge (a mask, over every cell) until it is scrolled to its
  // end, and a frame that fits shows none
  const frame = () => new Promise(done => requestAnimationFrame(() => requestAnimationFrame(done)));
  const uncued = [];
  for (const f of document.querySelectorAll('.tbl')) {
    if (!f.offsetParent || f.clientWidth <= 2 || f.closest('details:not([open])')) continue;
    const name = `"${(f.querySelector('tr > :first-child')?.textContent||'').trim().slice(0,24)}"`;
    const faded = () => getComputedStyle(f).maskImage !== 'none';
    // overflowing is what the browser scrolls, which a rounded scrollWidth misses by a pixel either way
    f.scrollLeft = f.scrollWidth; const scrolls = f.scrollLeft > 0; f.scrollLeft = 0;
    if (scrolls) {
      f.scrollLeft = 0; await frame(); const start = faded();
      f.scrollLeft = f.scrollWidth; await frame(); const end = faded();
      f.scrollLeft = 0; await frame();
      if (!start) uncued.push(`${name} overflows its frame with no cue`);
      if (end) uncued.push(`${name} still fades scrolled to its end`);
    } else if (faded()) uncued.push(`${name} fits its frame but fades`);
  }
  // the first screen's text reads at one measure: every part of it there is, one maximum width
  const measures = [], widths = new Set();
  const top = '.mast, .mast ~ .callout, .bluf > h1, .bluf > .lede, .bluf > [data-part="confidence"]';
  for (const e of document.querySelectorAll(top)) {
    const said = getComputedStyle(e).maxWidth;
    widths.add(said); measures.push(`${e.className || e.tagName.toLowerCase()} ${said}`);
  }
  // Colours as drawn: any computed CSS colour (rgb, a color-mix, oklab) painted on a 1 px canvas and read back as
  // sRGB, so every colour space the page's tokens mix in reads alike.
  const pad = document.createElement('canvas'); pad.width = pad.height = 1;
  const pen = pad.getContext('2d', {willReadFrequently: true}), seen = new Map();
  const rgba = c => {
    if (!c || c === 'none' || !CSS.supports('color', c)) return null;
    if (!seen.has(c)) {
      pen.clearRect(0, 0, 1, 1); pen.fillStyle = c; pen.fillRect(0, 0, 1, 1);
      const d = pen.getImageData(0, 0, 1, 1).data;
      seen.set(c, [d[0] / 255, d[1] / 255, d[2] / 255, d[3] / 255]);
    }
    return seen.get(c);
  };
  const over = (top, under) => [0, 1, 2].map(i => top[i] * top[3] + under[i] * (1 - top[3])).concat(1);
  const lum = c => c.slice(0, 3).map(v => v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4)
    .reduce((s, v, i) => s + v * [0.2126, 0.7152, 0.0722][i], 0);
  const ratio = (a, b) => { const [hi, lo] = [lum(a), lum(b)].sort((x, y) => y - x); return (hi + 0.05) / (lo + 0.05); };
  const opacity = e => { let o = 1; for (; e && e.nodeType === 1; e = e.parentElement) o *= parseFloat(getComputedStyle(e).opacity); return o; };
  // the ground under an element: its own and its ancestors' backgrounds, composited up to the first opaque one
  const ground = e => {
    const layers = [];
    for (; e && e.nodeType === 1; e = e.parentElement) {
      const c = rgba(getComputedStyle(e).backgroundColor);
      if (c && c[3] > 0) { layers.push(c); if (c[3] >= 1) break; }
    }
    return layers.reverse().reduce((g, c) => over(c, g), [1, 1, 1, 1]);
  };
  // drawn for a reader: a box of its own, outside every clipped (screen-reader-only) wrapper and closed fold
  const drawn = e => {
    const r = e.getBoundingClientRect();
    if (r.width <= 1 || r.height <= 1 || e.closest('#tip, details:not([open]) > :not(summary)')) return false;
    for (let a = e; a && a.nodeType === 1; a = a.parentElement) {
      const s = getComputedStyle(a);
      if (s.clipPath !== 'none' || s.visibility === 'hidden' || s.display === 'none') return false;
    }
    return opacity(e) > 0;
  };
  // a chart's text reads against the mark under its middle where one is filled there, else the figure's ground
  const shape = new Set(['rect', 'circle', 'ellipse', 'path', 'polygon']);
  const svgGround = t => {
    let r = t.getBoundingClientRect();
    if (r.top < 0 || r.bottom > innerHeight) { t.scrollIntoView({block: 'center'}); r = t.getBoundingClientRect(); }
    for (const e of document.elementsFromPoint(r.left + r.width / 2, r.top + r.height / 2)) {
      if (e === t || e.closest('text')) continue;
      if (e instanceof SVGElement) {
        if (!shape.has(e.tagName.toLowerCase())) continue;
        const s = getComputedStyle(e), f = rgba(s.fill);
        const a = f ? f[3] * parseFloat(s.fillOpacity) * opacity(e) : 0;
        if (a > 0.05) return over([f[0], f[1], f[2], a], ground(e.ownerSVGElement));
        continue;
      }
      return ground(e);
    }
    return ground(t.ownerSVGElement || t.parentElement);
  };
  const said = e => `${e.tagName.toLowerCase()}${e.classList.length ? '.' + [...e.classList].join('.') : ''} "${(e.textContent||'').trim().slice(0,24)}"`;
  // faint text and faint marks are listed apart, each capped on its own, so a faint mark is named however much text
  // is faint
  const SCALE = [12, 14, 16, 20, 28], faint = [], marks = [], sizes = [], texts = new Set();
  const walk = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  for (let node = walk.nextNode(); node; node = walk.nextNode()) {
    const e = node.parentElement;
    if (!node.data.trim() || !e || texts.has(e) || e.closest('script, style, title') || !drawn(e)) continue;
    texts.add(e);
    const s = getComputedStyle(e), size = parseFloat(s.fontSize);
    if (!SCALE.some(step => Math.abs(step - size) < 0.05)) sizes.push(`${said(e)} is ${+size.toFixed(2)} px`);
    const svg = e.closest('svg');
    const ink = rgba(svg ? s.fill : s.color);
    if (!ink) continue;
    const g = svg ? svgGround(e) : ground(e);
    const shown = over([ink[0], ink[1], ink[2], ink[3] * opacity(e) * (svg ? parseFloat(s.fillOpacity) : 1)], g);
    const r = ratio(shown, g);
    if (r < 4.5) faint.push(`${said(e)} ${r.toFixed(2)}:1`);
  }
  // a kit chart's mark drawn as a line (an interval) reads at 3:1 against its ground, as an axis does; a hairline grid
  // and a whisker under 2 px are rules, not marks, and a legend's swatch is drawn as the mark it keys
  for (const m of document.querySelectorAll('.chart[data-kind] svg:not(.legend svg) line, .chart-drawn svg:not(.legend svg) line')) {
    // a line's own box has no height or no width, so its drawing is drawn where its chart is
    if (!drawn(m.ownerSVGElement) || m.matches('.grid, .axis, .floor')) continue;
    const s = getComputedStyle(m), ink = rgba(s.stroke);
    if (!ink || parseFloat(s.strokeWidth) < 2) continue;
    const g = ground(m.ownerSVGElement), r = ratio(over([ink[0], ink[1], ink[2], ink[3] * parseFloat(s.strokeOpacity) * opacity(m)], g), g);
    if (r < 3) marks.push(`a mark in ${chartName(m.ownerSVGElement)} ${r.toFixed(2)}:1`);
  }
  for (const a of document.querySelectorAll('svg .axis')) {
    if (!drawn(a.ownerSVGElement)) continue;
    const s = getComputedStyle(a), ink = rgba(s.stroke);
    if (!ink) continue;
    const g = ground(a.ownerSVGElement), r = ratio(over([ink[0], ink[1], ink[2], ink[3] * parseFloat(s.strokeOpacity) * opacity(a)], g), g);
    if (r < 3) marks.push(`an axis in ${chartName(a.ownerSVGElement)} ${r.toFixed(2)}:1`);
  }
  window.scrollTo(0, 0);
  // an interval in a cell ("0 to 43%", "$25 to $34") holds its ends and the word between them on one line: the
  // package's page sets them in a span (.iv), found from the table's columns rather than the cell's text, a unit word
  // after it free to take the next line; a cell of a column of figures (.figs) holds its whole line. Each cell is
  // measured once (a .figs cell whole, its .iv span with it) and named with its table (tableName).
  // a table named by what the page heads it with: its caption, its block's heading or summary, its block's name
  // (data-table), else its place among the page's tables
  const frames = [...document.querySelectorAll('.tbl')];
  const tableName = f => {
    const block = f.closest('[data-table], figure, section'), said = e => (e?.textContent || '').trim().slice(0, 40);
    const head = said(f.querySelector('caption')) || said(block?.querySelector(':scope > h3, :scope > h4, :scope > summary, :scope > figcaption'));
    return head ? `table "${head}"` : block?.dataset?.table ? `table ${block.dataset.table}` : `table ${frames.indexOf(f) + 1}`;
  };
  const wrapped = [];
  for (const held of document.querySelectorAll('.tbl td:not(.figs) .iv, .tbl td.figs')) {
    const cell = held.closest('td');
    if (!drawn(cell)) continue;
    const range = document.createRange(); range.selectNodeContents(held);
    const tops = new Set([...range.getClientRects()].filter(r => r.width).map(r => Math.round(r.top)));
    if (tops.size > 1) wrapped.push(`"${(held.textContent || '').trim().slice(0, 32)}" in ${tableName(held.closest('.tbl'))} breaks across ${tops.size} lines`);
  }
  // a table is ruled, never boxed: no border on its frame, no fill behind its header row, a rule under it
  const tables = [];
  for (const f of document.querySelectorAll('.tbl')) {
    if (!f.querySelector('table') || !drawn(f)) continue;
    const name = `"${(f.querySelector('tr > :first-child')?.textContent||'').trim().slice(0,24)}"`, s = getComputedStyle(f);
    if (['Top', 'Right', 'Bottom', 'Left'].some(k => parseFloat(s[`border${k}Width`]) > 0 && s[`border${k}Style`] !== 'none'))
      tables.push(`${name} is drawn in a box`);
    const g = ground(f), heads = [...f.querySelectorAll('thead th')];
    if (heads.some(th => { const c = rgba(getComputedStyle(th).backgroundColor); return c && c[3] > 0 && c.some((v, i) => Math.abs(v - g[i]) > 1 / 255); }))
      tables.push(`${name} fills its header row`);
    const last = f.querySelector('thead tr:last-child > th');
    if (last && !(parseFloat(getComputedStyle(last).borderBottomWidth) >= 1)) tables.push(`${name} has no rule under its header`);
  }
  return {scrollWidth: document.documentElement.scrollWidth, past: out.slice(0, 8), overlaps: overlaps.slice(0, 12),
          crossed: crossed.slice(0, 8), wrapped: wrapped.slice(0, 8),
          unpinned: unpinned.slice(0, 8), crushed: crushed.slice(0, 8), under: under.slice(0, 8),
          uncued: uncued.slice(0, 8), measures: widths.size > 1 ? measures : [],
          contrast: [...faint.slice(0, 8), ...marks.slice(0, 8)], sizes: sizes.slice(0, 8), tables: tables.slice(0, 8)};
}"""


async def run(page_path, out_dir):
    failed = False
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        for width in (1200, 1104, 688, 390):
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
                    or res["crossed"]
                    or res["wrapped"]
                    or res["unpinned"]
                    or res["crushed"]
                    or res["under"]
                    or res["uncued"]
                    or res["measures"]
                    or res["contrast"]
                    or res["sizes"]
                    or res["tables"]
                )
                failed |= bool(bad)
                print(
                    f"{'FAIL' if bad else 'ok  '} {width}px {scheme:5}  scrollWidth={res['scrollWidth']}  screenshot={shot}"
                )
                for k in (
                    "past",
                    "overlaps",
                    "crossed",
                    "wrapped",
                    "unpinned",
                    "crushed",
                    "under",
                    "uncued",
                    "measures",
                    "contrast",
                    "sizes",
                    "tables",
                ):
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
