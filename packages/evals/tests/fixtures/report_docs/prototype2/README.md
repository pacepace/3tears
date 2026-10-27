# W2 held-out prototype: analyst and report editor runs (2026-10-27)

The second prototype, run after the W2 fix wave on report-kit/analyst @6da6c055f. Each directory is one campaign snapshot that no earlier run or exemplar covered: the shape campaigns s1 to s3, and three misfit variants of r2 and r3. An Opus analyst wrote each report from the skill (`packages/evals/src/threetears/evals/analysis/analyst/skill/`) and the `evals_analyst` tool, through the dev CLI (`tests/fixtures/report_plans/analyst_cli.py`). It did not read `exemplars/INDEX.md`, and the variants did not read the exemplar of the campaign they are built from. A fresh Opus report editor then scored each report on ANALYST.md §7's rubric: 7 items scored 0 to 2, and a report passes with no 0 and a total of at least 11. The editor never saw the analyst's notes, and checked every cited key with `figure_keys_check`. Between reviews the analyst revised in its own context, at most twice.

| Snapshot | Story | Confidence | Editor scores by round |
|---|---|---|---|
| s1 | decision (keep production) | high_if_immaterial | 11 → 13 |
| s2 | decision (beta, provisionally) | low | 12 → 13 |
| s3 | status (don't rely on solo for the clarity cases yet) | high_if_immaterial | 11 → 12 |
| r2 with one gate | decision (J, by 0.1 point) | high_if_immaterial | 11 → 13 |
| r2 with its gates read in each task | decision (ship none yet) | high_if_immaterial | 12 → 13 → 13 |
| r3 without its configs | decision (Brisk-S few-shot) | high_if_immaterial | 11 → 13 |

Every first draft passed. Every cited figure key read on its snapshot in every round. The analysts made 49 to 56 tool calls each and logged 52 capability gaps in all.

Files in each directory:
- `report.json`: the final report document (draft `report-doc/1`).
- `blocks.txt`: each evidence block's `block_render` text. It stands in for the page, since the claims renderer is W3's.
- `notes.md`: the analyst's reasoning note, the tools it used, and a section for each revision.
- `gaps.json`: every `capability_gap` call.
- `editor.json`, `editor2.json`, `editor3.json`: each round's review.
- `round1/`, `round2/`: earlier drafts with the review they received.
- `work/`, `editor-work/`: the raw tool output each agent saved.

Two caveats on these runs:
- They ran with two bugs the runs found. The goal-state measures of s1 to s3 read as fractions rather than percents, so a price per point on them was 100 times too high (fixed since, on report-kit/goal-state-rates). And a bar declared per aspect reads the lead aspect's figure, so on s3 code's own call has the solo arm passing the clarity bar it misses (the next fix wave's first item). The analysts worked around both, and their notes say so.
- The editor read values tables, not a rendered page.

## Converted documents (`doc.json`)

Each directory's `doc.json` is its `report.json` converted to `report-doc/1` as `threetears.evals.kernel.report_doc.ReportDoc` stores it (`tests/test_report_doc_model.py` validates every one). `report.json` and the rest of the record are kept as the run left them. The conversion:
- adds `recommends`, read by hand from the call this README and the report's decision state. It holds only an arm the report recommends: a tradeoff's lean ("our lean, not a recommendation") stays in its prose and recommends none, as does a report that ships nothing yet;
- fills every evidence query's omitted fields with their empty values, as `block_render` returns a query;
- drops `snapshot_id` where the draft carried one: the stored record names its snapshot;
- writes every field the model defaults (`flag: null`, a claim's `caveats` and `durable`, a block's `reference` and `emphasis`).

This prototype's conversion also:
- rewrites s3's headline (252 characters) within the 240 the schema allows, keeping every key it cited and its finding;
- drops `point_estimate:overall` from r3 without its configs: code no longer assigns it there, since an overall that is the plain mean of its cells now reads an interval derived from theirs. The analyst's clause on it asked for that interval.
