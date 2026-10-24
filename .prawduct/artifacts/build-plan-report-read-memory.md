# Build plan: one report read loads one analysis (#951)

Bugfix / optimization, medium. Branch `fix/report-read-one-analysis` from origin/develop @33d17e405 (0.78.0). No version bump.

## Goals

- A campaign's report read holds one analysis in memory, never every analysis the campaign ever had. Prod: one
  `report_read` of an 8-aspect campaign cost 400–500 MB, and a json-then-html pair OOM-killed a 2048 MB container.
- Peak memory does not grow with the number of archived analyses, pinned by a test.
- The default `report_read` is small (the toy 8-aspect campaign's html under 1 MB); the full per-trial records stay
  reachable, opt-in.
- The MCP reply carries the body once.

Decisions (each with what settled it):

- [DECISION] The report's analysis is chosen by one storage call, `load_reported_analysis(campaign_id, scope_id)`,
  on the `AnalysisStore` port and `EvalStorage`: a `by_doc_type` read filtered on `archived in (false, absent)`,
  newest `created_at` first, `limit=1`, so the store ships and the engine hydrates one document. No
  `DocumentStore` change: `field_in` with `None` and `limit` are already in its contract and its conformance kit, so a
  host's Postgres store needs nothing. `EvalStorage` is the only `AnalysisStore` implementation in the repo (the toy
  host's store implements the bundle's narrower read port, not this one). `record_campaign_call`'s check that a call
  names the campaign's report uses the same call, so "the campaign's report" has one definition.
- [DECISION] Step 2: no bundle is validated twice as a bundle. On load the analysis's `VerbatimJsonObject` fields get
  the JSON-shape check every load gets; `analysis_campaign_bundle` then does the typed read (`from_dict`), which is
  the validation that guards the read, and `from_frame` checks each bundle's fingerprint against the frame. What IS
  done twice is the fingerprint: `build_code_only_report` hashes each bundle again for its source line. A
  campaign page over a stored analysis hands the fingerprint it already checked to the fold builder instead
  (measured: 16 fingerprints → 8 per read). The JSON-shape check stays (it is the stored read's contract); step 1
  makes it run over one analysis instead of all.
- [DECISION] Step 3: a `trials` option, `"table"` (default) or `"all"`, on `report_read` (op and action),
  `campaign_report`, `analysis_report` and the CLI `report --trials`. Every campaign-page door defaults to `"table"`,
  so the one-page-every-door invariant (`test_report_doors`, `test_cli`) holds unchanged. `"table"`: the census line,
  the failed-and-lost table (each row now naming its stored result for `result_get`, and anchored so a trial id
  link lands on its row), the passed table; no per-trial records. `"all"`: the records too, as before. The
  one-aspect page builders (`build_report`, `build_code_only_report`) keep laying out every record by default:
  the quick path's `Result.report` and its tests read them. JSON carries no trial records in either mode.
- [DECISION] Step 4: the fastmcp transport leaves out of the structured content a top-level string field whose
  value IS the text block (`report_read`'s `body`, rendered verbatim). No other action renders a field verbatim
  (every other render is `render_model` or a composed text), so no other action's structured content changes.
- Step 5 (cache): decided after re-measuring; see Results.

## Baseline (before), toy 8-aspect campaign

Script: one serial process (scratchpad `measure_report.py`): 8 aspects, each a real quick-path run of 40 cases × 3
repeats × 2 arms (production passes 20 cases, the candidate 13), every failed answer 4000 bytes; N analyses
generated through the host with a fixtured writer, all but the newest archived. tracemalloc on, so times read high.

| phase | 1 analysis | 6 (5 archived) | 11 (10 archived) |
|---|---|---|---|
| list_analyses (every analysis hydrated) | 0.36 s, 6.9 MB | 1.57 s, 36.4 MB | 2.51 s, 65.9 MB |
| analysis_campaign_bundle | 0.34 s, 8.1 MB | 0.47 s, 8.1 MB | 0.33 s, 8.1 MB |
| _campaign_page | 0.56 s, 9.6 MB | 0.69 s, 9.6 MB | 0.65 s, 9.6 MB |
| campaign_report (whole) | 1.38 s, 20.2 MB | 5.33 s, 36.4 MB | 5.12 s, 65.7 MB |
| serialize html | 1.89 s, 11.6 MB | 3.73 s, 11.6 MB | 2.48 s, 11.6 MB |
| serialize markdown | 0.08 s, 3.6 MB | 0.12 s, 3.6 MB | 0.16 s, 3.6 MB |
| serialize json | 0.004 s | 0.006 s | 0.004 s |
| html body | 1.09 MB | 1.09 MB | 1.09 MB |
| markdown body | 0.62 MB | 0.62 MB | 0.62 MB |
| json body | 0.02 MB | 0.02 MB | 0.02 MB |

campaign_report's peak grows ~6 MB per archived analysis on this toy; prod's analyses are larger. Most of the html
serializer's time is `_links_in_sight` parsing the page it wrote, so it scales with the page's size.

## Chunks

1. `load_reported_analysis` on the port and `EvalStorage`; `campaign_report` and `_reported_analysis` use it.
   Tests: a counting store pins one analysis document read however many are archived; a tracemalloc test pins the
   peak flat across archived counts. Re-measure.
2. The fold builder takes the fingerprint already checked. Re-measure.
3. `trials` option through the campaign page; failed table gains a stored-result column and row anchors in
   `"table"`. Tests pin the new default exactly and that `"all"` lays out what the page laid out before. Docs:
   action detail, op docstring, CHANGELOG, MIGRATIONS. Re-measure.
4. Transport: body once. Test over the fastmcp client: report_read's structured content has no body and its text
   is the body; another action's structured content is unchanged.
5. Cache: measure; build or write up.
6. Docs: the GIL rule beside the executor guidance in the host-adoption docs.

## Results

(filled in as each chunk lands)
