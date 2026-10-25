# Kit API

`kit.css` is the visual system: tokens, type, layout, tables, pills, controls. `kit.js` holds the chart and naming helpers. Both are inlined into each page by `scripts/build.py`. There are no dependencies beyond Google Fonts (IBM Plex Sans, Sans Condensed, Mono).

## Tokens (kit.css)

| Token | Use |
|---|---|
| `--host-accent` | **The only token a host overrides** (give light and dark values). Drives `--accent`, `--accent-wash`, focus rings and the recommended-row highlight. |
| `--bg --surface --sunk --ink --ink-2 --muted --line --base` | Neutrals, with a slight cool bias. |
| `--on-fill` | Text set on a filled mark (a pressed segment, an arm's tag square, a count inside a bar). |
| `--s1 … --s5` | Series colors in fixed order: blue, orange, aqua, gold, violet. The control uses `--ink-2`. |
| `--good --warn --bad` and `*-wash` | Status. Always shown with ✓ ! ✕ and a word (`pill()`). |
| `--seq-0 … --seq-5`, `--seq-t0 … --seq-t5` | Sequential ramp and the matching text color for each step. The dark ramp runs light-on-dark. |
| `--div-neg --div-mid --div-pos` | Diverging ramp for difference heatmaps. |
| `--f-display --f-body --f-mono` | Plex Sans Condensed, Plex Sans, Plex Mono. |

Dark values are defined twice, under `prefers-color-scheme: dark` (guarded by `:not([data-theme="light"])`) and under `[data-theme="dark"]`, so a host toggle wins both ways.

## Layout classes

`.page` (68rem) · `.prose` (44rem) · `.mast` · `.bluf` · `.decision` (4 slots; `.rec` highlights one) · `section > header` (`.eyebrow`, `h2`, `.takeaway`) · `.part` (divider before Explore and Full results) · `.fig` (card for a chart) · `.grid2` · `.mult` (auto-fill small multiples; `.m5` for 5 across) · `.tbl` (scrolling table wrapper) · `tr.hl` (recommended row) · `.controls` `.ctl` `.seg` `.toggle` · `.callout` · `.stats .stat .big .lab` · `.method` (definition list) · `.flag` (amber confidence line) · `.armchip .tag` · `.excard` (failure example card: `.top`, `.reason`, `pre`).

## Functions (kit.js)

All charts draw SVG at the container's real pixel width and redraw on resize. Colors are passed as token names (`"--s1"`), applied through `style`, so theme changes need no redraw.

| Function | Signature | Notes |
|---|---|---|
| `chart(el, fn)` | `fn(el, width)` | Responsive mount. Returns a `redraw()` to call after state changes (explore controls). |
| `svgIn(el, w, h)` | → svg | |
| `S(tag, attrs, parent)` / `H(tag, attrs, html)` | | SVG and HTML element helpers; `attrs.text` sets the text. |
| `lin(d0,d1,r0,r1)`, `logs(...)`, `ticks(a,b,n)` | | Scales and nice ticks. |
| `fmt.pct .usd .s .ms .x` | | Formatters. |
| `hit(g, x, y, w, h, html)` | | Focusable hit target with a tooltip, larger than the mark. |
| `placeLabels(items, bounds, obstacles)` | `items:[{px,py,text}]` → adds `x,y,anchor` | Tries right, left, above, below, then nudges; avoids other labels and obstacle boxes. Falls back to the least-overlapping spot. |
| `barsH(el, w, rows, opt)` | `rows:[{label, sl?, value, lo?, hi?, color, muted?, strong?, note?, tip?}]`, `opt:{domain, fmt, tickFmt?, ref?:{value,label}, labelW?, rowH?}` | Magnitude or diverging bars with optional whiskers and a reference line. Long labels use `sl` or are truncated; notes fall back to the value when they don't fit. |
| `forest(el, w, rows, opt)` | `rows:[{group?, label, value, lo, hi, color, strong?, note?, tip?}]`, `opt:{domain, fmt, margin?, marginLabel?, level?, labelW?, rowH?}` | Change against a control: zero line, interval, optional margin band at −margin, interval level printed. The shape D spine. |
| `lineChart(el, w, cats, series, opt)` | `series:[{name, color, values, lo?, hi?, strong?, dash?}]`, `opt:{domain, fmt, xTitle?, h?, endLabel?}` | Ordered categories only. End labels, de-collided; bands where `lo`/`hi` are given. |
| `armLabels(arms, attrs, controlName)` | `arms:[{id, role, alias?, spec}]`, `attrs:[{k, fmt?, sfmt?}]` | Sets `tag` (P, A, B…), `short` and `long`. Rules in GENERATOR.md §6.1. |
| `armChip(arm, color)` | → HTML | Tag square plus short name. |
| `pill(kind, text)` | `kind: good, warn, bad, neutral` | Status with icon. |
| `sortable(table, rows, render, key, dir)` | | Sortable headers (`th.sort[data-k]`). |

Charts that exist only as page-local code in the references, and that should move into the kit when the renderer is ported: the gate matrix (r1, an HTML table), dot + interval strips (r1 `dims`), latency range bars (r1), the gate-space scatter with factor paths (r2), lever-effect panels (r2), the extra-call stacked bars (r2), the config lattice (r3, HTML grid), the heatmap, difference heatmap and confusion (r3 `.mx` grids), the frontier scatter (r1, r3), and the savings-against-loss map with the price line (r4).
