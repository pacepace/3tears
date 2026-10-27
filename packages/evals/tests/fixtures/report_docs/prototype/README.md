# W2 prototype: analyst and report editor runs (2026-10-26)

Each directory is one campaign snapshot. An Opus analyst wrote the report, working only from the skill (`packages/evals/src/threetears/evals/analysis/analyst/skill/`) and the `evals_analyst` tool, through the dev CLI (`tests/fixtures/report_plans/analyst_cli.py`). It was blind to the campaign's own exemplar note and reference page. A fresh Opus report editor then scored the report on ANALYST.md §7's rubric (7 items, 0–2; a report passes with no 0 and a total of at least 11). The editor never saw the analyst's notes. Between reviews the analyst revised the report in its own context, at most twice.

| Snapshot | Story | Confidence | Editor scores by round |
|---|---|---|---|
| kairo | decision (keep September) | high_if_immaterial | 11 → 13 |
| r1 | decision (Atlas-M everywhere) | high_if_immaterial | 12 → 13 |
| r2 | decision (Brisk-S low, one retry; re-test first) | low | 12 → 13 |
| r3 | decision (Brisk-S few-shot) | high_if_immaterial | 10 (fail) → 12 |
| r4 | tradeoff (prod vs f64; run the mix next) | high_if_immaterial | 10 (fail) → 12 → 13 |
| r2 without its gates (misfit) | tradeoff | high_if_immaterial | 11 → 13 |
| r1 reading only bars (misfit) | tradeoff | high_if_immaterial | 11 → 12 |

Files in each directory:
- `report.json`: the final report document (draft `report-doc/1`).
- `blocks.txt`: each evidence block's `block_render` text. It stands in for the page, since the claims renderer is W3's.
- `notes.md`: the analyst's reasoning note, the tools it used, and a section for each revision.
- `gaps.json`: every `capability_gap` call.
- `editor*.json`: each round's review.
- `round1/`, `round2/`: earlier report drafts.

Two caveats on these runs:
- Wave A (kairo, r1, r4, r2 without its gates) read `exemplars/INDEX.md`, whose one-line summaries give away each reference campaign's story. Those runs are less blind than wave B.
- The editor read values tables, not a rendered page.
