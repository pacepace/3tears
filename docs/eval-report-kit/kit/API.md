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

`.page` (68rem) · `.prose` (44rem) · `.mast` · `.bluf` · `.decision` (4 slots; `.rec` highlights one) · `section > header` (`.eyebrow`, `h2`, `.takeaway`) · `.part` (divider before Explore and Full results) · `.fig` (card for a chart) · `.grid2` · `.mult` (auto-fill small multiples; a chart's `panel_min_rem` sets the narrowest column) · `.tbl` (scrolling table wrapper) · `tr.hl` (recommended row) · `.controls` `.ctl` `.seg` `.toggle` · `.callout` · `.stats .stat .big .lab` · `.method` (definition list) · `.flag` (amber confidence line) · `.armchip .tag` · `.excard` (failure example card: `.top`, `.q`, `dl`, `.reason`, `pre`, `.note`) · `.gate` (gate matrix table) · `.lattice` (factor tiles) · `.mx` (cell grids: heatmap, difference, confusion). The kit's HTML kinds write these; their column counts are set inline from the data.

## Chart data: the `data-chart` contract

A chart is one element. The kit finds it, parses its JSON and draws it; the page carries no chart code.

```html
<div class="chart" data-kind="forest" data-chart='{"domain":[-16,4],"fmt":"+num:0","level":"95%","rows":[…]}'></div>
```

- **Drawing.** `drawCharts()` runs on `DOMContentLoaded` and draws every `div.chart[data-kind][data-chart]`. SVG kinds draw at the element's real pixel width and redraw on resize and once web fonts load; HTML kinds (`gate_matrix`, `lattice`, `heatmap`, `diff_heatmap`, `confusion`, `examples`) draw once and size with CSS. Changing `data-chart` later (`setChart(el, data)`) redraws that chart, which is how an explore control updates one.
- **Refusals.** An unknown kind, JSON that does not parse, a missing required field, a color that is not a token, an unknown `fmt`, or intervals without a `level` draw nothing for that element and log `kit: chart "<kind>" #<id>: <reason>` with `console.error`, so `scripts/check.py` fails the page.
- **Pure data.** The JSON holds no functions and no markup. Every string is text: the kit escapes it, so a label, tooltip or excerpt can never add an element. The producer HTML-escapes the attribute value itself (`&`, and the quote it wraps the attribute in).
- **Units.** Values are in display units: percentages as 0–100 (85.3, not 0.853), points as points, dollars as dollars, latency in ms or s as the `fmt` says. The kit formats; it never rescales.

### Fields every kind shares

Each kind uses the subset its table lists. The same idea has the same name in every kind.

| Field | Type | Meaning |
|---|---|---|
| `fmt` | string | Named format for values and axis labels: `name` or `name:digits`, with a leading `+` to sign positive values. Names: `num` (plain), `int` (rounded, thousands separators), `pct` (`85.3%`), `pts` (`2.1 pts`), `usd` (no digits: $0.302, $11.5, $1,208; with digits: fixed), `usdk` ($7.6k), `ms` (`513 ms`), `s` (`3.0 s`), `dur` (ms in, `600ms` / `1s` out), `x` (`1.5×`), `k` (`3.1k`). Without digits a value prints as many decimals as it needs, up to two (`usd` and `ms` keep their own rounding). |
| `tick_fmt` | string | Format for axis labels when it differs from `fmt`. |
| `domain` | [number, number] | Axis range in value units. A value outside it is drawn at the edge. |
| `ticks` | number[] | Explicit tick positions. The kit drops any whose label would touch its neighbour at the drawn width. Without it, nice ticks are chosen so no two labels repeat. |
| `level` | string | Interval level, printed on the chart as "`<level>` intervals" ("95%", "99.69%"). Required by `forest` and `dot_interval`, and by `bars`, `diverging_bars` and `lines` whenever a row or series carries `lo`/`hi` (GENERATOR.md §5.2). |
| `reference` | {value, label?, kind} | A floor, budget, margin or no-preference point, in value units. `kind`: `floor` (pass at or above), `budget` (pass at or below), `margin` (forest: the band from `value` up to the domain's top is shaded), `no_preference` (diverging bars grow from `value` instead of zero; no line is drawn). `label` is printed beside the line when given. |
| `emphasis` | string | The `id` of the one row, point or cell to emphasize: bold label, ring (scatter) or outline (lattice). |
| `id` | string | On a row, point or cell: what `emphasis` and `paths` name. |
| `color` | string | A token name (`"--s1"`, `"--ink-2"`, `"--seq-2"`), checked against `^--[a-z][a-z0-9-]*$`. Arm identity: the same color for an arm on every chart. |
| `strong` / `muted` | boolean | On a row or point: bold label and note / faded mark. |
| `tip` | string[] | Tooltip lines; the first is set in bold. Each kind has a default when omitted. |
| `note` | string | Text printed at a bar's end or a forest row's right edge; falls back to the formatted value when it would not fit. |
| `short_label` | string | Used when `label` does not fit the label column. |
| `label_width`, `label_width_narrow` | number | Label column in px, above and below 480 px chart width. Default: fitted to the labels. |
| `row_height` | number | Row pitch in px. |
| `height` | number | Plot height in px (scatter, lines). |
| `legend` | [{label, color?, mark?, hollow?, opacity?}] | Drawn above the chart (above the grid for small multiples). `mark`: `dot` (default) or `square` (a swatch), `line` (a stroke sample, `opacity` applies), or a marker shape `circle`, `triangle`, `diamond`. |
| `panels` | object[] | Small multiples (`small_multiples_by`). Each panel is `{title, subtitle?, tag?, color?, …kind fields}`, drawn as a figure in a `.mult` grid; the chart's own fields are shared by every panel and a panel's fields override them. `tag` and `color` print the title as an arm chip. A panel carrying only text fields is drawn as a text card. |
| `panel_min_rem` | number | Narrowest panel before the grid drops a column (default 15.5). |

### Kinds

Required fields in **bold**.

| Kind | Fields | Draws |
|---|---|---|
| `bars` | **`rows`**, **`domain`**, **`fmt`**, `tick_fmt`, `ticks`, `reference`, `level`, `emphasis`, `label_width(_narrow)`, `row_height`, `legend`, `panels`. Row: **`label`**, **`value`** or **`segments`** `[{value, color}]`, `id`, `short_label`, `lo`, `hi`, `color`, `strong`, `muted`, `note`, `tip`, `group`. | Horizontal bars from zero, whiskers where `lo`/`hi` are given, a dashed reference line. `segments` stack from zero (extra calls, recovered failures); `note` defaults to their total. Rows sharing a `group` sit under its heading. |
| `diverging_bars` | As `bars`. | Bars about zero, or about `reference.value` when its kind is `no_preference`. Mute rows not separated from zero and bold those that are. |
| `forest` | **`rows`**, **`domain`**, **`fmt`**, **`level`**, `reference` (kind `margin`), `emphasis`, `label_width(_narrow)`, `row_height`. Row: **`label`**, **`value`**, **`lo`**, **`hi`**, `id`, `short_label`, `color`, `strong`, `note`, `tip`, `group`. | Change against a control: zero line, interval, square at the estimate, shaded margin band. The shape D spine. |
| `lines` | **`categories`** string[], **`series`**, **`domain`**, **`fmt`**, `x_title`, `height`, `level`. Series: **`name`**, **`values`** number[], `end_label`, `color`, `lo` number[], `hi` number[], `strong`, `dash`. | Ordered x only. End labels de-collided; bands where `lo`/`hi` are given (at most two series). |
| `scatter_frontier` | **`points`**, **`x_domain`**, **`y_domain`**, **`x_fmt`**, **`y_fmt`**, `x_log`, `x_ticks`, `y_ticks`, `x_title`, `y_title`, `reference` (y floor), `x_reference`, `frontier`, `paths`, `price_line`, `emphasis`, `height`, `legend`, `panels`. Point: **`x`**, **`y`**, `id`, `label`, `narrow_label`, `color`, `shape` (`circle`, `square`, `triangle`, `diamond`), `size` (radius px, default 5.5), `hollow`, `muted`, `tip`. | Two measures that trade off. `frontier: true` draws the Pareto line (lower x, higher y is better). `paths: [{ids, color, width?, opacity?}]` joins named points in order (a factor's levels, a hybrid's handoffs). `price_line: {value, label?, region_label?}` draws y = −x / value from the origin and shades the side above it (worth it at that price). Labels go through `placeLabels()`: `label` at 560 px and wider, `narrow_label` below; a point without one is unlabelled at that width. `emphasis` rings its point and labels it first, in bold. |
| `gate_scatter` | As `scatter_frontier`; **`reference`** and **`x_reference`** required, plus `region_label`. | Shades the region where both references pass (each by its `kind`) and names it with `region_label`. |
| `dot_interval` | **`rows`**, **`domain`**, **`fmt`**, **`level`**, `ticks`, `reference`, `emphasis`, `label_width(_narrow)`, `row_height`, `panels`. Row: **`label`**, **`value`**, `lo`, `hi`, `id`, `color`, `shape`, `hollow`, `strong`, `tip`. | One rate per candidate against a floor: mark at the value, faint interval, floor dashed. |
| `range_bars` | **`groups`**, **`domain`**, **`fmt`**, `ticks`, `reference` (budget), `legend`. Group: **`label`**, **`rows`**, `short_label` (below 480 px). Row: **`label`**, **`from`**, **`to`**, `color`, `shape`, `hollow`, `tip`. | Latency distributions: one bar per row from `from` (p50) to `to` (p95), a mark at `to`, the budget line. |
| `gate_matrix` | **`columns`** [{label}], **`rows`**, `row_header`, `cost_fmt`, `cost_unit`, `pick_label`. Row: **`label`**, **`cells`**, `detail`, `flag`. Cell: **`status`** (`good`, `warn`, `bad`, `neutral`), **`label`**, `reasons` string[], `cost`, `pick`. | An HTML table: per cell a pill (icon plus the producer's word), the binding reasons, the cost; `pick` cells washed and marked with `pick_label`. `flag` marks a row that applies to all traffic. |
| `lattice` | **`columns`** [{label}], **`rows`**, **`fmt`**, `bar_domain`, `cost_fmt`, `cost_unit`, `latency_fmt`, `emphasis`. Row: **`label`**, **`cells`**, `color`. Cell: **`value`**, `id`, `cost`, `latency`, `status`, `status_label`, `tip`. | Two crossed factors: a tile per cell with the metric, a bar over `bar_domain`, cost and latency, a status pill. |
| `heatmap` | **`columns`** [{label}], **`rows`**, **`domain`**, **`fmt`**, `n_fmt`, `reference` (floor), `panels`. Row: **`label`**, **`cells`**. Cell: **`value`**, `n`, `tip`. | Sequential shading in six steps over `domain`, `n` under each value, a red outline on cells below the floor. |
| `diff_heatmap` | **`columns`**, **`rows`**, **`fmt`**, `detail_fmt`, `neutral` (default 0.5), `saturate` (default 8). Cell: **`value`** (the difference), `detail` (printed small, e.g. the compared rate), `tip`. | Diverging shading: gray within ±`neutral`, full color at ±`saturate`. |
| `confusion` | **`labels`** string[], **`matrix`** number[][], `fmt` (default `pct:1`), `normalized` (default true: rows already sum to 100; false: the kit normalizes counts), `warn_at` (default 1), `bad_at` (default 3), `panels`. | Row-normalized confusion, one per label axis: diagonal tinted, off-diagonal cells amber and red by share. |
| `examples` | **`cards`**, `card_min_rem`. Card: `meta` [{text, mono?} or {status, label} or {tag, label, color}], `question`, `fields` [{term, text, tone?}] (`tone: "bad"` sets the text in red), `reason`, `excerpt`, `note`. | Failure example cards (`.excard`): the meta line (ids, stratum, a status pill, an arm chip), then each part present, in that order. |
| `table` | none | Not drawn by the kit: a table is written as an HTML table (`.tbl` wrapper, `sortable()` for sort headers). The kind exists in the plan schema so a plan can name it; the dispatcher refuses it. |

## Functions (kit.js)

All charts draw SVG at the container's real pixel width and redraw on resize. Colors are passed as token names (`"--s1"`), applied through `style`, so theme changes need no redraw. The declarative renderers above are built on these; a page calls them directly only for a chart no kind covers.

| Function | Signature | Notes |
|---|---|---|
| `drawCharts(root?)` | | Draws every `div.chart[data-kind][data-chart]` under `root` (default `document`) once and watches its `data-chart`. Runs on `DOMContentLoaded`. |
| `setChart(el, data)` | | Sets `el`'s `data-chart` to `JSON.stringify(data)`, which draws or redraws it. |
| `namedFmt(spec)` | → `fmt(v)` | The formatter for a `fmt` name; throws on an unknown one. |
| `chart(el, fn)` | `fn(el, width)` | Responsive mount. Returns a `redraw()` to call after state changes (explore controls). |
| `svgIn(el, w, h)` | → svg | |
| `S(tag, attrs, parent)` / `H(tag, attrs, html)` | | SVG and HTML element helpers; `attrs.text` sets the text. |
| `lin(d0,d1,r0,r1)`, `logs(...)`, `ticks(a,b,n)` | | Scales and nice ticks. |
| `fmt.pct .usd .s .ms .x` | | Formatters. |
| `esc(s)`, `tipHTML(lines)` | | Escape text for markup; build tooltip markup from text lines (first bold). |
| `hit(g, x, y, w, h, html)` | | Focusable hit target with a tooltip, larger than the mark; `html` is already-escaped markup (`tipHTML`). |
| `mark(g, shape, x, y, r, color, hollow)` | `shape: circle, square, triangle, diamond` | An arm's marker, filled or hollow. |
| `textW(text, size?, weight?)` | → px | Text width in the chart font, measured. |
| `placeLabels(items, bounds, obstacles)` | `items:[{px,py,text,strong?}]` → adds `x,y,anchor` | Tries right, left, above, below, then nudges up and down, preferring nearer spots; avoids other labels and obstacle boxes `[x0,y0,x1,y1,cost?]` (cost 10 by default; reference lines cost 1, so crossing one beats colliding with a label). Falls back to the cheapest spot. |
| `barsH(el, w, rows, opt)` | `rows:[{label, sl?, value \| segments, lo?, hi?, color, muted?, strong?, note?, tip?, group?}]`, `opt:{domain, fmt, tickFmt?, ticks?, ref?:{value,label}, origin?, originLabel?, level?, labelW?, rowH?}` | Magnitude, diverging, stacked or grouped bars with optional whiskers and a reference line. Long labels use `sl` or are truncated; notes fall back to the value when they don't fit. |
| `forest(el, w, rows, opt)` | `rows:[{group?, label, value, lo, hi, color, strong?, note?, tip?}]`, `opt:{domain, fmt, ref?:{value,label}, level?, labelW?, rowH?}` | Change against a control: zero line, interval, margin band from `ref.value` up, interval level printed. |
| `lineChart(el, w, cats, series, opt)` | `series:[{name, color, values, lo?, hi?, strong?, dash?, endLabel?}]`, `opt:{domain, fmt, xTitle?, h?, level?}` | Ordered categories only. End labels, de-collided; bands where `lo`/`hi` are given. |
| `armLabels(arms, attrs, controlName)` | `arms:[{id, role, alias?, spec}]`, `attrs:[{k, fmt?, sfmt?}]` | Sets `tag` (P, A, B…), `short` and `long`. Rules in GENERATOR.md §6.1. |
| `armChip(arm, color)` | → HTML | Tag square plus short name. |
| `pill(kind, text)` | `kind: good, warn, bad, neutral` | Status with icon. `text` is markup; escape model text first. |
| `sortable(table, rows, render, key, dir)` | | Sortable headers (`th.sort[data-k]`). |

Tooltips (`tip`) in the functions above are text lines, escaped by the kit, never markup.

Every chart in the five references is drawn from `data-chart`. What stays page code is explore logic (P6): the controls, the recompute and the prose they write, which set chart data with `setChart()`. One explore chart has no kind and stays page-local: r2's per-task comparison of the selected configuration against the recommended one (a dot for each, joined by a line).
