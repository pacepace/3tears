# W2 regression prototype: analyst and report editor runs (2026-10-27)

The third prototype, run after the W2-fix and W2b waves on report-kit/analyst @b5a586c07. It reran the campaigns the first prototype covered (Kairo round 2 and r1 to r4) to check that the fixes kept or improved what already passed. As before, an Opus analyst wrote each report from the skill (`docs/eval-report-kit/analyst/`) and the `evals_analyst` tool through the dev CLI (`tests/fixtures/report_plans/analyst_cli.py`), blind to its campaign's own exemplar, to `exemplars/INDEX.md` and to the test fixtures (the earlier prototype reports among them). A fresh Opus report editor scored each report on ANALYST.md §7's rubric: 7 items scored 0 to 2, and a report passes with no 0 and a total of at least 11. The analyst revised in its own context, at most twice.

| Snapshot | Story | Confidence | First prototype, by round | This run, by round |
|---|---|---|---|---|
| kairo | decision (keep September) | high_if_immaterial | 11 → 13 | 11 → 13 |
| r1 | decision (Atlas-M everywhere) | high_if_immaterial | 12 → 13 | 13 |
| r2 | decision (J, by 0.1 point) | high_if_immaterial | 12 → 13 | 12 → 13 |
| r3 | decision (Brisk-S few-shot) | high_if_immaterial | 10 (fail) → 12 | 13 → 14 → 14 |
| r4 | tradeoff (Production vs f64) | high_if_immaterial | 10 (fail) → 12 → 13 | 10 (fail) → 12 |

Every call matched the first prototype's. The analysts used 41 to 56 tool calls each (first prototype: 52 to 86) and logged 28 capability gaps across the five (first prototype: 66 for the same five). Every cited figure key read on its snapshot in every round. Each must_fix finding was an analyst misreading or overclaiming a correct tool result, except r3's last, where a rounding filter dropped a cost multiple's unit word.

Files in each directory:
- `report.json`: the final report document (draft `report-doc/1`).
- `blocks.txt`: each evidence block's `block_render` text. It stands in for the page, since the claims renderer is W3's.
- `notes.md`: the analyst's reasoning note, the tools it used, and a section for each revision.
- `gaps.json`: every `capability_gap` call.
- `editor.json`, `editor2.json`, `editor3.json`: each round's review.
- `round1/`, `round2/`: earlier drafts with the review they received.
- `work/`, `editor-work/`: the raw tool output each agent saved.

The editor read values tables, not a rendered page.

## Converted documents (`doc.json`)

Each directory's `doc.json` is its `report.json` converted to `report-doc/1` as `threetears.evals.kernel.report_doc.ReportDoc` stores it (`tests/test_report_doc_model.py` validates every one). `report.json` and the rest of the record are kept as the run left them. The conversion:
- adds `recommends`, read by hand from the call this README and the report's decision state: a tradeoff's is the arm its lean names, and a report that ships nothing yet recommends none;
- fills every evidence query's omitted fields with their empty values, as `block_render` returns a query;
- drops `snapshot_id` where the draft carried one: the stored record names its snapshot;
- writes every field the model defaults (`flag: null`, a claim's `caveats` and `durable`, a block's `reference` and `emphasis`).

This prototype's conversion also:
- tightens r4's headline (267 characters) and r3's lede (469) to the 240 and 420 the schema allows, keeping their keys and their findings;
- drops `point_estimate:overall` from r3: code no longer assigns it there, since an overall that is the plain mean of its cells now reads an interval derived from theirs. The analyst's notes logged that interval as a capability gap.
