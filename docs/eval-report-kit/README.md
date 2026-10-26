# Eval report kit

Everything needed to produce campaign reports that read and look like the five reference memos here, and a plan for converging `threetears.evals` on them.

## Start here

0. **`ANALYST.md`** is the foundation (2026-10-26): an expert analyst agent with tools, a skill and a report editor writes the report; code computes and verifies. It supersedes GENERATOR's shape table, spine and code-computed mode as rules.
1. **`GENERATOR.md`** is the complete spec: how to classify an eval, the page contract, gate logic, tradeoff mode, chart choice, words, the visual system, and the definition of done.
2. **`references/`** holds five finished pages. Open the `.html` files in a browser.
3. **`CONVERGENCE.md`** is the phased plan for the report module, with acceptance criteria.
4. **`UPSTREAM.md`** lists what the campaign bundle should carry so the adapter stops working around gaps.

## Files

| Path | What it is |
|---|---|
| `ANALYST.md` | The analyst foundation: roles, workflow, report document, tools, invariants, editor rubric |
| `GENERATOR.md` | Procedure and rules; its §1 shape table, spine and mode rules are superseded by ANALYST.md, the rest is guidance |
| `CONVERGENCE.md` | How the pipeline converges on the references, in phases P0–P7 |
| `UPSTREAM.md` | Bundle requirements U1–U15, each with evidence from Kairo round 2 |
| `BLIND-TEST.md` | What a fresh agent found with only this bundle, and how each gap was closed |
| `schema/report-input.schema.json` | Normalized input the planner reads |
| `schema/report-plan.schema.json` | What the planner writes; the renderer draws it |
| `adapters/bundle_to_report.py` | Campaign writer-input package → report input |
| `kit/kit.css`, `kit/kit.js`, `kit/API.md` | Visual system and chart helpers |
| `references/rN-*.src.html` + `.json` | Page source and its data; `.html` is the built page |
| `references/r5-kairo-round2.plan.json` | The plan the planner should write for Kairo round 2 |
| `scripts/build.py` | Inline the kit and data into a page |
| `scripts/check.py` | Render at 1200 and 390 px, light and dark; report overflow, label overlaps and errors |

## The references

| Page | Shape | Data | What it demonstrates |
|---|---|---|---|
| `r1-model-selection` | A: candidates under gates | synthetic | Gate matrix, cross-cutting gate blocking a routing plan, behavior strips, latency ranges, gate explorer |
| `r2-config-sweep` | B: factorial sweep | synthetic | Gate-space scatter with factor paths, lever effects, failure anatomy, efficiency, failing examples |
| `r3-classifier` | C: classifier over a target matrix | synthetic | Model × prompt lattice, frontier, per-cell heatmaps, per-axis confusion, difference explorer |
| `r4-tradeoff` | D, tradeoff mode | synthetic | No rule set in advance: price per point, flip bands, modeled hybrid, human anchor, long arm names |
| `r5-kairo-round2` | D, decide mode | **real** (Kairo round 2 bundle) | Forest against the margin, combined confidence flag, stratum leads, failure examples, margin explorer |

## Tracking

The convergence is tracked in #961 and the bundle-side requirements in #962.

## Commands

```bash
python3 -I adapters/bundle_to_report.py <writer-input.json> report-input.json
python3 scripts/build.py references/r5-kairo-round2.src.html report-input.json out.html
pip install playwright && playwright install chromium
python3 scripts/check.py out.html
```
