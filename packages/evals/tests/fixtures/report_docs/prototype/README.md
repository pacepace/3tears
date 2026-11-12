# W2 prototype: analyst and report editor runs (2026-10-26)

Each directory is one campaign snapshot. An Opus analyst wrote the report, working only from the skill (`packages/evals/src/threetears/evals/analysis/analyst/skill/`) and the `evals_analyst` tool, through the dev CLI (`tests/fixtures/writer_cases/analyst_cli.py`). It was blind to the campaign's own exemplar note and reference page. A fresh Opus report editor then scored the report on ANALYST.md §7's rubric (7 items, 0–2; a report passes with no 0 and a total of at least 11). The editor never saw the analyst's notes. Between reviews the analyst revised the report in its own context, at most twice.

| Snapshot | Story | Confidence | Editor scores by round |
|---|---|---|---|
| kairo | decision (keep September) | high_if_immaterial | 11 → 13 |
| r1 | decision (Atlas-M everywhere) | high_if_immaterial | 12 → 13 |
| r2 | decision (Brisk-S low, one retry; re-test first) | low | 12 → 13 |
| r3 | decision (Brisk-S few-shot) | high_if_immaterial | 10 (fail) → 12 |
| r4 | tradeoff (prod vs f64; run the mix next) | high_if_immaterial | 10 (fail) → 12 → 13 |
| r2 without its gates (misfit) | tradeoff | high_if_immaterial | 11 → 13 |
| r1 reading only bars (misfit) | tradeoff | high_if_immaterial | 11 → 12 |

Each directory now holds only `doc.json`, the one file the tests read. The run's other files (`report.json`, `blocks.txt`, `notes.md`, `gaps.json`, each round's review, the earlier drafts) were removed, since nothing read them; git holds them at 83be8d3de.

Two caveats on these runs:
- Wave A (kairo, r1, r4, r2 without its gates) read `exemplars/INDEX.md`, whose one-line summaries give away each reference campaign's story. Those runs are less blind than wave B.
- The editor read values tables, not a rendered page.

## Converted documents (`doc.json`)

Each directory's `doc.json` is the run's `report.json` converted to `report-doc/1` as `threetears.evals.kernel.report_doc.ReportDoc` stores it (`tests/test_report_doc_model.py` validates every one). The conversion:
- adds `recommends`, read by hand from the call this README and the report's decision state. It holds only an arm the report recommends: a tradeoff's lean ("our lean, not a recommendation") stays in its prose and recommends none, as does a report that ships nothing yet;
- fills every evidence query's omitted fields with their empty values, as `block_render` returns a query;
- drops `snapshot_id` where the draft carried one: the stored record names its snapshot;
- writes every field the model defaults (`flag: null`, a claim's `caveats` and `durable`, a block's `reference` and `emphasis`).

This prototype's drafts predate the schema's bounds and its caveat ids. Its conversion also:
- rewrites three fields over their bounds within them: the headlines of r1 reading only bars (285 characters) and r2 without its gates (284), and r4's lede (488). Each keeps every key it cited and its finding;
- merges the first two of r1 reading only bars's four explore rows, both floors on judged behaviours, into one, since `explore` holds three.

Outside Kairo, its caveat ids were mostly made up by the analyst before code assigned ids: `judge_not_stated`, `resolves_borderline`, `cost_basis` and the rest (r1's `judge_unstated` is code's). They are kept as written, as planted negatives: the model accepts them, and validity on their snapshots refuses them.
