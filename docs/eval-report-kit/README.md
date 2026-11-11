# Eval report kit

What a campaign report is, how it is written, and the parts it is drawn with: the design documents, the visual system, five reference pages and the checks that hold a rendered page to them.

## Start here

1. **[`ANALYST.md`](ANALYST.md)** is how a report is written: an analyst model investigates a frozen snapshot of the campaign with tools and a skill, writes a report document, and code verifies it before an editor reviews it. Code computes and verifies; it never chooses structure.
2. **`references/`** holds five finished pages. Open the `.html` files in a browser.
3. **[`GENERATOR.md`](GENERATOR.md)** holds the page rules: gate logic, the call's modes, the page's parts, chart choice, words and the visual system. Code cites its sections by number, so it keeps its old name.
4. **[The upstream-asks issue](ISSUE_URL)** lists what the campaign bundle should still carry so the report input stops working around gaps.

## Files

| Path | What it is |
|---|---|
| `ANALYST.md` | The analyst: roles, workflow, the report document, tools, invariants, the editor's rubric |
| `analyst/README.md` | Where the analyst skill lives: in the evals package, as data the wheel ships |
| `GENERATOR.md` | The page rules: gates, modes, the page's parts, charts, words and visuals |
| `BLIND-TEST.md` | A dated record (2026-10-25): what a fresh agent found with only this kit, and how each gap was closed |
| `schema/report-input.schema.json` | The normalized input the analyst's tools read, frozen in each snapshot |
| `schema/report-doc.schema.json` | The report document the analyst writes and the store keeps (`report-doc/1`) |
| `adapters/bundle_to_report.py` | Campaign writer-input package → report input |
| `kit/kit.css`, `kit/kit.js`, `kit/API.md` | Visual system and chart helpers |
| `references/rN-*.src.html` + `.json` | Page source and its data; `.html` is the built page |
| `scripts/build.py` | Inline the kit and data into a page |
| `scripts/check.py` | Render at 1200, 1104, 688 and 390 px, light and dark; report overflow, label overlaps, a table that overflows with no cue, a first screen at more than one measure, text under 4.5:1 or an axis under 3:1 against what it is drawn on, a font size off the five-step scale, a boxed or header-filled table, and errors |

## The references

| Page | Shape | Data | What it demonstrates |
|---|---|---|---|
| `r1-model-selection` | A: candidates under gates | synthetic | Gate matrix, cross-cutting gate blocking a routing plan, behavior strips, latency ranges, gate explorer |
| `r2-config-sweep` | B: factorial sweep | synthetic | Gate-space scatter with factor paths, lever effects, failure anatomy, efficiency, failing examples |
| `r3-classifier` | C: classifier over a target matrix | synthetic | Model × prompt lattice, frontier, per-cell heatmaps, per-axis confusion, difference explorer |
| `r4-tradeoff` | D, tradeoff mode | synthetic | No rule set in advance: price per point, flip bands, modeled hybrid, human anchor, long arm names |
| `r5-kairo-round2` | D, decide mode | **real** (Kairo round 2 bundle) | Forest against the margin, combined confidence flag, stratum leads, failure examples, margin explorer |

The shapes name what each reference demonstrates. The analyst reads them as worked examples, never as a lookup: its reasoning note on each is in the skill's exemplars.

## Commands

```bash
python3 -I adapters/bundle_to_report.py <writer-input.json> report-input.json
python3 scripts/build.py references/r5-kairo-round2.src.html report-input.json out.html
pip install playwright && playwright install chromium
python3 scripts/check.py out.html
```
