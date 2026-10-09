# Reading reports

Read this when you have a campaign — runs you want compared — and want to read what it found, render it,
or build your own renderer. It covers the report document, how arms are named, results by kind of case
(strata), how far a judged score can be trusted (evidence tiers), and drawing charts. Terms such as
campaign, arm, cell and analysis are defined in [Concepts](concepts.md).

## What a report is, in plain words

Every campaign is read through one document, the **report**. It is a list of blocks in reading order:
text, tables, charts and disclosures. If someone has generated an **analysis** (a model reading the
campaign's numbers and writing findings), the report carries those findings beside the evidence. If not,
you still get a **code-only report**: every table and chart code can build, with no words from a model,
and a line saying no analysis was generated. Either way, every number comes from code; a model never
gets to state a figure or decide how much a judged score can be trusted.

[`examples/reports.py`](../examples/reports.py) takes a finished campaign to the files people read: the
contrasts' verdicts read off the `Report` as data, the report as Markdown and HTML, the evidence bundle
as JSON, and each chart as a Vega-Lite spec (plus an SVG with the `[vega]` extra). It runs offline.

## The report

A generated analysis is read through one document. `analysis_report(storage, analysis_id, scope_id)`
returns a `Report`: an ordered list of blocks — `text` (what the analysis's author wrote, with a role),
`table` (evidence, arms, decision surface), `chart` and `disclosure` (what code must add) — each linked
to the findings it belongs to or rests on. Serialize it three ways:

```python
from threetears.evals.analysis import analysis_report, report_html, report_markdown

report = analysis_report(host.storage, analysis_id, scope_id)
report.to_canonical_json()   # validated by the published schema, report/schema.json (see below)
report_markdown(report)      # for an agent, or to paste as a memo
report_html(report)          # a page that reads without a script
```

## The campaign's report

**The campaign's report** is `campaign_report(host, campaign_id, scope_id)` — the one answer the CLI's
`report` and the `report_read` action both give: the campaign's newest analysis that is not archived,
or, when it has none, a **code-only report** of its evidence (`build_code_only_report`).

A code-only report has `basis="code_only"` and no author's words — no headline, no findings, no text block,
which the published schema and the model both refuse. It holds:

- the arm table (every arm unresolved, since nothing decided);
- the decision surface;
- the contrasts the evidence tested against the control;
- a distribution chart per measure and judged dimension;
- for a classifier, one `labels` table of each label's precision, recall and F1, a row per label and arm:
  precision and recall with their 95% Wilson intervals, F1 with none (it has none by construction), and
  every figure with the n it is counted over;
- the results by stratum when the cases declare strata ([below](#results-by-kind-of-case-strata));
- every disclosure the evidence carries, opening with one line saying no analysis was generated.

**What an analysis would add.** A code-only report is what code computed; nothing in it reads the numbers.
An analysis adds that reading: findings, each with the evidence it rests on and the caveats that qualify it;
a decision per declared question, with its confidence; which arm won and why; and what to run next. No
headline, finding, decision or answer to a declared question appears in a code-only report.

`Report.basis` says which a report is; `REPORT_VERSION` is 4.

## Having a model write the analysis, over frozen evidence

An analysis is written from the campaign's **bundle** alone (`AnalysisContextBundle`), and every figure
in it is a reference code resolves against that bundle. Save the bundle (`bundle.to_json()`), reload it
(`AnalysisContextBundle.from_json`) and its `fingerprint()` is unchanged, so `generate_analysis` over the
saved file under a second prompt compares the two prompts and nothing else; each analysis records the
fingerprint it read on `generation.bundle_fingerprint`. The insight-ledger cutoff (`bundle_assembled_at`),
the generation time and its cost are on that provenance, not in the bundle.
[`examples/llm_analysis.py`](../examples/llm_analysis.py) does all of it in one file.

## How an arm is named

Every block that names an arm — decisions, evidence rows, the arm table, the decision surface, the
results by stratum, the contrasts and every chart group — prints one name for it, built once
(`threetears.evals.analysis.arms.arm_names`).

- An arm is named by the levers whose levels differ across the report's arms, each lever compared only
  among the arms it applies to; one that differs on none is named by its candidate kind and model.
- A lever that does not apply to the arm's kind is never named.
- A cell adds `@ rig <digest>` when its arm was measured under more than one rig.
- Each level is put on one line and cut in the middle past 48 characters, alike in Markdown, HTML and the
  chart intents.
- Two arms that would still read alike carry their variant keys' digests (`(arm <digest>)`), so no two
  arms of one report share a name.

What an arm ran in full is stated once, in the arm table's "Every lever it ran" column
(`ArmRow.settings`). A stored analysis's own charts keep the names they were drawn with at generation.

*Example:* a campaign comparing the triage classifier's prompt v1 and v2 on one model names its arms by the
prompt lever alone (`ticket_router.prompt_version=v1` and `ticket_router.prompt_version=v2`, as `lever=level` pairs), since the model is the same on both.

## What the schema checks, and what only the model does

`schema.json` holds the report's shape and every cross-field rule JSON Schema can state:

- a code-only report names no analysis or model and holds no headline, finding or text block;
- an analysis report names both;
- a report with no findings links no block to one;
- a finding's own words name their finding;
- a chart block carries exactly one of an intent and an error, the intent of its own type.

Three rules compare a value with a sibling's, which JSON Schema cannot: a block's finding positions are
below `finding_count`, a table's `total_rows` is at least the rows it shows, and a row keys only its
table's columns. Those only `Report.model_validate` holds, so a host that validates against the schema
alone accepts exactly those three malformations as well.

## Chart blocks carry an intent, not a library's spec

A chart block carries the chart's **intent** (`ChartIntent`, from `threetears.evals.analysis.viz`), never
a charting library's spec: its type from eval's eight, the rows it draws, what each field encodes
(identity, length, position, interval with what it varies over, level, class, ordinal), its axes with
their units and zero baselines, its order, the colour *slots* it uses and what it must disclose — plus its
values as drawn, which the HTML shows as a table.

How a chart looks is the host's: a renderer reads the intent and the host's palette —
`StyleProfile.chart_palette`, a renderer-neutral `ChartPalette` (the eight numbered series slots, slots
1-4 validated; a sequential ramp; background, ink, muted, grid, rule, context and on-fill), every colour
resolved `#rrggbb`. The presentation rules are checked on the intent (`check_intent`), so they hold for
any renderer, and the core ships no charting library.

## Results by kind of case: strata

A pooled accuracy can hide that a variant does well on easy tickets and badly on the hard ones you care
about. Strata let you see each kind of case separately.

A test case may declare a **stratum**, the kind of case it is (`EvalTestCase(stratum="lookalike")`), in
the author's own words. The engine puts it in no prompt; a kind, which is handed the whole case, must
leave it out of what it renders too, so the candidate is never told what kind of case it faces.

**How it is computed.** When any case of a cell declares one, the analysis reads that cell again per
stratum: every measure, a classifier's confusion matrix and per-label precision, recall and F1 included,
and every judged dimension, each over that stratum's cases alone, by the same rules as the cell's pooled
figure (`CellFacts.strata`, a list of `StratumFacts` on the decision surface). Cases declaring none in
such a cell are their own entry, so the strata add up to the cell. A cell none of whose cases declares a
stratum is not broken down, and its report reads as if strata did not exist.

**How it is shown.** Both reports carry it as a `strata` table: a row per arm and reading, the pooled
figure under `All cases`, then a column per stratum. Each arm opens with a `cases` row giving the cases
and observations behind each column. A stratum holding fewer than `STRATUM_MIN_CASES` (10) cases is still
shown, marked too few to read alone, and named in a `strata` disclosure below the table. A rate is shown
with its Wilson interval and a mean with its standard error, each with its n.

**Generated cases** take their stratum from the template: mark one `enum` or `sample` variation axis
`VariationAxis(..., stratum=True)` and each case generated takes that axis's value as its stratum. A case
generated before the axis was nominated keeps the stratum it was written with, which is none.

**Limits.** An analysis's writer reads the strata in its bundle, but a finding's evidence still cites a
cell's pooled figure: a reference names a cell and a measure, and there is no reference to one stratum of
a cell yet.

**Identity.** A stratum enters no identity key. A stored case never changes, so its id already pins its
stratum, and two arms run over the same cases share their strata. It is not part of `content_hash`
either, which digests what the candidate is given.

## How far a judged score can be leaned on: evidence tiers

A judged score is one model's opinion of another model's output. Before you act on one, you want to know
whether that judge has been shown to agree with people (best), or at least with itself when asked twice.
The engine measures both and labels every judged reading with the result — the **evidence tier**. It
never hides a reading for a weak tier; it tells you how much weight it can bear.

Every judged reading — each `judged_measures` arm, each judged reading on the decision surface, each
judged evidence row of a finding — carries an `evidence_tier` that code decides from what the judge's
reliability was measured to be (`threetears.evals.contracts.evidence_tiers`, owner ruling 2026-10-06):

| Tier | When |
|---|---|
| `calibrated` | the judge agrees with people: `judge_agreement` (person ratings only) at least `CALIBRATION_MIN_AGREEMENT` (0.6) over at least `CALIBRATION_MIN_RESULTS` (20) distinct results |
| `separation` | the judge agrees with itself: `judge_self_agreement` at least `SEPARATION_MIN_AGREEMENT` (0.8) over at least `SEPARATION_MIN_RESULTS` (20) distinct results |
| `incidental` | both measured over enough results, and both missed |
| `undetermined` | too little evidence to decide — never filed as incidental |

**How agreement is computed.** Agreement is one statistic computed by one rule for both — quadratic-weighted
kappa on 1-5, kappa on pass/fail, per rater (each person; each round of repeats) and pooled by result.
Every distinct result weighs 1, split across the raters that measured it, so the figure weighs what the
floor counts, distinct results, never pairs: neither a small rater nor many raters re-measuring a few
shared results (five annotators on the same three anchors; one result repeated thirty times) can carry it,
or the floor, over the bar. A repeat that answers "can't tell" where the judge had scored is a
disagreement, never set aside. A judge is a served model and a judge config, so a tier measured under one
prompt never sets another's.

**Where tiers appear.** The bundle lists each judge's tier per dimension with both criteria
(`judge_evidence_tiers`); a finding stands on the weakest tier among its rows
(`FindingResolution.evidence_tier`: `mechanical`, `calibrated`, `separation`, `undetermined`, `incidental`
or `none`), which every report states beside the finding; a code-only report also states each judge's tier
with the numbers behind it. Tiers are flagged, never a reason to drop a reading.

**Measuring self-agreement: judge repeats.** Self-agreement is measured by **repeating** a finished run's
judge scores: `repeat_judge_scores` (operation `judge_repeat`; `estimate_judge_repeat` /
`judge_repeat_estimate` price it without a call) asks the same judge the same question again from the
evidence its first judge read, under the apparatus the run recorded, and records each answer beside the
score it repeats (`EvalResult.judge_repeats`) without changing the scores. Every call it can make — parse
retries included — is priced and admitted against the host's out-of-run cap before the first is sent, and
each is ledgered under purpose `judge` with the run's id (see
[Cost and budgets](cost-and-budgets.md#spend-outside-any-run)).

## Drawing charts: the Vega-Lite adapter

The package's own renderer is an optional adapter, `threetears.evals.vega`. Install the extra for its
rasteriser (`vl-convert-python`):

```bash
pip install "3tears-evals[vega]"
```

```python
from pathlib import Path

from threetears.evals.analysis import finding_chart_intent
from threetears.evals.vega import VegaRenderer

# The host's declared palette, bound once; a host declaring none draws in the packaged "dark" palette.
renderer = VegaRenderer.for_style(host.profile.style, theme="dark", font_dir=Path("/srv/fonts"))
intent = finding_chart_intent(host.storage, analysis_id, scope_id, "0")  # or a chart block's `intent`
chart = renderer.draw(intent)   # a colourless Vega-Lite spec, chart.spec, for a browser to embed...
renderer.config()               # ...with this config beside it
renderer.png(chart)             # or rasterised, for a surface that cannot run a browser
```

Drawing a spec needs nothing past the core; only `png` and `svg` need the extra. Nothing in the core
imports the adapter.

## Bringing your own renderer

A host bringing its own renderer implements `ChartRenderer` (`draw(intent)`, and `drawn_data(drawing)`
reading its drawing back) and runs the one conformance check every renderer passes —
`assert_renderer_conforms(renderer, intents)`, from `threetears.evals.analysis.viz`: what it draws agrees
with the intent's marks (`data`), per identity, and the intent's values-as-drawn table agrees with those
marks (`table_disagreements`, policy rule 12) — so a drawing that passes agrees with the table beside it.
A table column spelled from a drawn number (a delta's `+72.7%`) or carried only by the table is the
builder's to spell, and is compared with nothing drawn.
