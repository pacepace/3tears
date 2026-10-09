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

- the arm table: each arm and every lever it ran, with no status column (every arm is unresolved, since
  nothing decided) and no finding column (there are no findings);
- the guardrails, each decided for each arm against the control ([below](#reading-the-guardrails));
- the decision surface;
- the contrasts the evidence tested against the control;
- a distribution chart per measure and judged dimension;
- for a classifier, one `labels` table of each label's precision, recall and F1, a row per label and arm:
  precision and recall with their 95% Wilson intervals over the cases, F1 with none (it has none by construction), and
  every figure with the n it is counted over;
- the results by stratum when the cases declare strata ([below](#results-by-kind-of-case-strata));
- every disclosure the evidence carries, opening with one line saying no analysis was generated.

**What an analysis would add.** A code-only report is what code computed; nothing in it reads the numbers.
An analysis adds that reading: findings, each with the evidence it rests on and the caveats that qualify it;
a decision per declared question, with its confidence; which arm won and why; and what to run next. No
headline, finding, decision or answer to a declared question appears in a code-only report.

`Report.basis` says which a report is; `REPORT_VERSION` is 5.

## Reading the arm table

An analysis's arm table gives each arm a status, read off the decisions that name its cells:

| Status | What it means |
|---|---|
| winner | an adopted decision names it |
| contradicted | one decision adopts it and another rejects it |
| ruled out | a rejected decision names it |
| replaced incumbent | the control, when some other arm won |
| unresolved | no decision reached a verdict on it; a deferred decision is not a verdict |

The analysis writer is refused when it adopts and rejects one arm, so only an analysis stored before that
refusal can carry **contradicted**. Neither verdict is shown as standing: the arm is never shown as the
winner, the control is not shown as replaced on its account, and a disclosure below the table names it.

## Reading the decision surface

The decision surface is a row per cell (an arm under one rig) with every bar's verdict and the cost and
latency figures. **Its row order is not a ranking**, and the table says so in its caption (`order` on the
table block):

- The control's rows come first, marked `(control: the reference)`: every other arm is read against them.
  They lead because they are the reference, not because they were picked.
- The other arms follow in alphabetical order of their names, and an arm measured under several rigs lists
  its rigs by id.
- With no control row, no row is the reference, and the caption says that instead.

Read which arm the evidence favours from the contrasts against the control and the analysis's decisions,
never from which row is on top. The results by stratum follow the same order, and so does the analysis
writer's `cell_measures`, so when the control was measured its first cell (`c1`) is the control's.

## Reading a comparison

The "Contrasts against the control" table (`multiple_comparisons` in the bundle) tests each arm against
the control on each reading, under one rig.

- **The test.** Per-case means (each case's repeats averaged first, because a case is the independent
  draw). When the arms share at least two cases, it is a paired t-test over the shared cases. Otherwise it is
  Welch's t statistic read on Hsu's `min(n_a, n_b) − 1` degrees of freedom. That is conservative at every
  size: on Welch–Satterthwaite's degrees of freedom, a 2-case side against 10 called false differences 10%
  of the time.
- **Delta, the means, and "Cases tested"** are all over the cases the test read. A case only one arm ran is
  left out of a paired test, and the row counts it, so a mean here can differ from the cell's own mean.
- **The interval on delta** comes from the same test, at `1 − 0.05/m` for a family of `m` tested rows
  (Bonferroni). All of a family's intervals cover together at least 95% of the time. One that excludes zero
  always comes with a separation. A separation Holm's later steps find can still touch zero.
- **Hedges' g** is the standardized difference, corrected for small samples. Cohen's d reads 0.88 for a
  true 0.5 at three paired cases; g reads 0.5. There is none at two pairs, where no unbiased estimate exists.
- **p (Holm-adjusted)** is corrected within the family: one per declared question (the readings on its
  merit axes), or, with no question, one campaign-wide family over every reading on a merit axis. A measure
  on no axis is never tested. That keeps the rig's own readings out: the judge's time (`judge_ms`), the
  drain wait, and `cost_usd` and `program_cost`, which include what the judge spent. The candidate's spend
  is tested as `production_replicating_cost`.

| Verdict | Means | Do |
|---|---|---|
| improved / regressed | adjusted p < 0.05, in that direction | act on it, unless the row says *immaterial* (below the measure's margin) |
| equivalent | the paired difference is shown inside ± the measure's margin by TOST, corrected in the same family | treat the arms as interchangeable on this reading |
| not separated | the cases could not tell the arms apart | add cases, or declare a margin; never read it as a tie |
| untested | no test could decide: too few cases, or every case moved by one amount over too few cases for an exact test to reach 0.05; the row says why | fix what it names (usually too few cases) |

`equivalent` needs a declared margin (`MetricDescriptor.materiality_threshold`) and a paired test. Its p
is corrected in the same Holm family as the separations, with the multiplier capped at the number of
compared rows (Shaffer's refinement: a difference cannot be both zero and at least the margin), so the
chance that any verdict in the family is wrong stays at 5%.

A delta-table chart states each row's change relative to the baseline only on a ratio scale. A judged 1–5
score (`MetricDescriptor.scale="interval"`) moves in points: one point up is +50% from 2 and +25% from 4,
so no percent is stated for it.

## Reading the guardrails

A guardrail is something an arm must not get worse on: a judged dimension on the `boundary` axis, or a
measure the host declared `guardrail`. The "Guardrails against the control" table (`guardrails` in the
bundle, its own section of both reports) decides each one for each arm, on its own 95% interval on
`arm − control`. Guardrails are never in the contrasts table, the composite or pass^k, so a gain there
cannot hide a loss here.

| Decision | Means | Do |
|---|---|---|
| held | the whole interval is on the good side of the margin | nothing; the arm did not get worse by more than you tolerate |
| breached | the whole interval is beyond the margin | do not adopt the arm, whatever it gained; the analysis writer is refused if it tries |
| undecided | the interval straddles the line, or no interval exists; the row says why | do not read it as safe. An arm can still be adopted, and the decision then carries a `Guardrails` fact naming it. To decide it, add cases or declare a margin |

The margin is the measure's `materiality_threshold`. A judged dimension declares none, so it is held at zero
change: `held` then needs the arm shown no worse at all, which a few cases rarely show. When every case
moved by the same amount (both arms pass every case, or every case flipped), a t interval has no width; the
row then reads `bounded` and uses the widest difference the scale allows for the cases that could still
move. An `undecided` guardrail does not block adoption because at a few cases and no margin almost every
guardrail is undecided, and a rule that blocked them all would block every adoption.

## Readings no question asked about: exploratory

A campaign's declared questions say what it set out to learn. A reading on no axis a live question names is
**exploratory**: worth reporting as a lead for the next campaign, never as a confirmed answer. Where
questions are declared, the bundle lists those readings (`reading_scope`), the code-only report names them
under the questions, and a finding resting only on them carries a `Scope` fact. Where none are declared,
every finding is exploratory, and the report says so once near the top rather than on every row. A
guardrail is never exploratory.

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
resolved `#rrggbb` — and, optionally, its typeface, `StyleProfile.chart_font`. A `ChartFont` is a CSS
family list plus the advance widths measured for its first face, because a renderer lays labels out from
those widths: a font declared without them is refused, rather than fitted against another face's widths.
The presentation rules are checked on the intent (`check_intent`), so they hold for any renderer, and the
core ships no charting library.

## Did a lever take effect: mechanism checks and observed mechanisms

A lever that changed nothing and a lever that never took effect read alike in every outcome measure. A
`Sweepable` lever may name the measure or covariate it is supposed to move — `acts_on="context_tokens_in"` on a
chunk-width lever, say; a kind's overlay field does the same with `ActsOn(...)` beside `Ordinal()` and
`Interval(...)`. Each coverage row of the analysis bundle then tests that measure across the lever's levels with
the same separation test the contrasts against the control use (per-case means, paired where the levels share
cases, Holm-corrected across the lever's pairs): `moved` when some pair separates; `inert` when every level was
observed, every pair could be tested and none separates — no measurable evidence the lever acted on its mechanism;
otherwise `unchecked` with the reason (`not_declared`, `not_swept`, `levels_unobserved`, `too_few_observations`).
Each level's mean and its number of cases sit beside the state. Every case shifting by the same amount is read
by an exact test, so it reads `moved` only from six shared cases (four a side where the levels share none); below
that it is `too_few_observations`, since a 0/1 measure under a lever that did nothing shifts two cases alike one
time in eight.

The profile accepts only a numeric measure that each result carries as a single value: a covariate, a measure
your kind reports per result, or one of the result's own fields. It refuses anything else where you declare it,
including a quantity recorded per usage row (`reasoning_tokens`: declare `reasoning_ratio`) or only over a whole
run (`p95_total_ms`). For a call cap, report the calls each case used as a host measure and name that. The
declaration enters no variant key. A lever naming no mechanism reads `unchecked`, never as having taken effect.

A knob your host also records resolved, as a lever of its own (`ResolvesInto(...)` on a kind's overlay field, or
`resolves_into` on a `Sweepable`), is reported as one lever where the runs show the resolved lever moved only with
the knob. The resolved lever then has no coverage row and is named in no confound, and the arm is named by the
knob, with the resolved lever listed in its variant-index entry's `folded`. Where it also moved while the knob was
held at one level by arms the comparison reads, it is reported as a lever of its own and named as a confound on
that comparison. Which arms that is depends on the lens: the knob's coverage row reads the arms that moved the knob,
so a drift in an arm that left the knob alone shows in the design's contrast for that arm, not on the knob's row.
Where a run did not record it, it is named as an `undecided` confound.

Only two arms at one level of the knob can show the resolved lever moving on its own; repeats of one arm cannot,
because every run of an arm resolves the same value. A fold nothing could have refuted is still applied, but it
carries an `unverified_fold` confound (`unverified_fold:<lever>`, explained in `confound_catalog`), and a code-only
report states it among its disclosures. Read it as an assumption these runs did not test: the knob's effect is not
separated from anything else written into that lever. A fold without the mark was tested at the levels two or more arms held, and held there; levels only one arm ran were not tested.

A comparison across candidate models can also differ in what the models did while no setting differed. A
reasoning effort is a word each vendor maps to its own budget, so two models at one effort setting can reason very
differently. The bundle reads each arm's mean reasoning share (`reasoning_ratio`) into `arm_mechanisms`. Where two
models' shares are at least `REASONING_SHARE_DIVERGENCE` (0.20) apart, the comparison names an
`observed_mechanism` confound carrying the threshold and both models' values: on the model coverage row and its
divergences (`confounded_by`), and on each pairwise contrast against the control (`mechanism_confounds` on the
design's contrast arms and on each family comparison). On any other lever the share moving is what the lever did,
so it is never named there. The confound qualifies the comparison; it never hides it. A share nothing measured is
said to be unmeasured and names no confound.

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
with its Wilson interval and a mean with its standard error, each over the cases and each with its n.

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
| `calibrated` | the judge agrees with people: the one-sided 95% lower bound on `judge_agreement` (person ratings only) is at or above `CALIBRATION_MIN_AGREEMENT` (0.6), over at least `CALIBRATION_MIN_RESULTS` (20) distinct results |
| `separation` | the judge agrees with itself: the one-sided 95% lower bound on `judge_self_agreement` is at or above `SEPARATION_MIN_AGREEMENT` (0.8), over at least `SEPARATION_MIN_RESULTS` (120) distinct results |
| `incidental` | both measured over enough results, and both upper bounds lie below their bars |
| `undetermined` | not shown either way: too few results, or bounds across a bar — never filed as incidental |

**A tier is decided on confidence bounds, never the point estimate.** A criterion is `met` when its one-sided
95% lower bound reaches the bar, `not_met` when its one-sided 97.5% upper bound is below it, and `undecided`
otherwise. An undecided criterion leaves the reading on the next tier down that is shown. At 20 results kappa's
sampling spread is about 0.2, and the point estimate awarded `calibrated` to a judge whose true agreement was 0.5
a third of the time. The tier sentence names the bounds, which way they fell, and how many more results the
criterion needs (`results_needed`): the rest of the floor when it is short, or, when undecided, about how many
more would carry the bounds clear of the bar if agreement held at its estimate. An analysis stored before this
rule has no `judged_tier_rule`, and its tiers are rendered as decided on the point estimate.

**Why these bounds.** Three ways to bound kappa were compared by seeded simulation over six marginals (1-5 flat,
peaked and skewed with quadratic weights, 1-5 with a "can't tell" answer, 1-5 unweighted, pass/fail 1:1 and 3:1),
500 replicates each. "Size" is how often a judge exactly at the bar is shown over it, at a one-sided 5% bound;
the target is at most 5%. Power is the range over marginals of how often a better judge earns the tier.

| Method | n | Size, 0.6 bar | Size, 0.8 bar | Power at 0.9, 0.6 bar | Power at 0.95, 0.8 bar | Coverage of a 95% interval |
|---|---|---|---|---|---|---|
| Score interval (chosen) | 20 | 3.0% | 3.2% | 12-80% | 0-39% | ≥ 94.0% |
| | 40 | 3.6% | 3.2% | 57-97% | 7-72% | ≥ 94.6% |
| | 60 | 4.0% | 3.6% | 71-100% | 37-90% | ≥ 94.8% |
| Analytic SE (Fleiss-Cohen-Everitt) on t | 20 | 16.6% | 33.6% | 72-91% | 63-76% | ≥ 51.7% |
| | 60 | 10.8% | 17.6% | 90-100% | 79-96% | ≥ 73.8% |
| Bootstrap over results, percentile | 20 | 12.6% | 25.8% | 64-89% | 54-73% | ≥ 51.3% |
| | 60 | 9.2% | 14.4% | 86-100% | 73-95% | ≥ 78.0% |
| Bootstrap over results, BCa | 20 | 12.0% | 24.2% | 50-86% | 51-70% | ≥ 50.7% |
| | 60 | 7.0% | 11.4% | 76-100% | 54-92% | ≥ 82.0% |

The analytic and bootstrap bounds read their spread off the estimate, so a sample that happens to agree looks
certain: their extra power is mostly false awards. The score interval holds each candidate kappa to the spread it
would have there. A score interval that read the disagreement size off the observed disagreements alone awarded
the tier at the bar 8-18% of the time, so the size used is the larger of the observed and the chance one. The
upper bound runs looser than the lower one (a one-sided 95% upper bound showed a judge at the bar below it up to
8.8% of the time), so a miss is decided on a 97.5% upper bound (1.4-3.7% at the floors). Raters of one result are a
cluster: their disagreements with the judge are added at the correlation they show on shared results. Two people
who copy the same truth, added as independent, awarded the tier at the bar 9.5% of the time; with the
correlation estimated it was at most 4.6%. A bootstrap over results would cluster them too, but it fails at
these sizes as the table shows.

**How many ratings a judge needs.** Seeded simulation, one rater per result, 1,500 replicates. The chance that a
judge earns the tier, by distinct results:

| Tier | True agreement | 20 | 40 | 60 | 80 | 100 | 120 | 140 |
|---|---|---|---|---|---|---|---|---|
| `calibrated` (bar 0.6) | 0.9 | 14-45% | 57-86% | 71-98% | 81-99% | 90-100% | 94-100% | 97-100% |
| `calibrated` (bar 0.6) | 0.8 | 4-16% | 23-42% | 35-58% | 46-76% | 53-86% | 62-90% | 69-95% |
| `separation` (bar 0.8) | 0.95 | below floor | below floor | below floor | below floor | below floor | 70-98% | 76-98% |

The low end of each range is a heavily skewed 1-5 marginal. Plan on about 50 person-rated results for a
`calibrated` reading of a good judge (80 on a skewed scale), and 120 repeated results for `separation`. A judge at
true self-agreement 0.9 needs more than 200 repeats for an 80% chance. Separation's floor is 120 rather than 20
because no valid bound can show a kappa of 0.8 from 20 results: even twenty perfect repeats bound it near 0.7.

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

# The host's declared palette and font, bound once; a host declaring none draws in the packaged "dark"
# palette and the packaged face. `font_dir` holds the files of a host's own face; the packaged one needs none.
renderer = VegaRenderer.for_style(host.profile.style, theme="dark", font_dir=Path("/srv/fonts"))
intent = finding_chart_intent(host.storage, analysis_id, scope_id, "0")  # or a chart block's `intent`
chart = renderer.draw(intent)   # a colourless Vega-Lite spec, chart.spec, for a browser to embed...
renderer.config()               # ...with this config beside it
renderer.png(chart)             # or rasterised, for a surface that cannot run a browser
```

Drawing a spec needs nothing past the core; only `png` and `svg` need the extra. Nothing in the core
imports the adapter.

The packaged palette is a brand-neutral default with a light and a dark variant (`packaged_palette("light")`,
`packaged_palette("dark")`). Its hues are a published, validated default categorical palette, used
unchanged, in an order chosen so that the four validated slots can be told apart as a set. The package's
tests measure these rules on it:

- text (`ink`, `muted`) clears 4.5:1 against the chart surface, and `on_fill` clears 4.5:1 over slot 1;
- slot 1, the colour of every single-series mark, clears 3:1 against the surface, as do `highlight` and
  `context`;
- every pair of slots 1-4 differs by at least OKLab ΔE 6 under simulated protanopia and deuteranopia and
  ΔE 15 under normal vision, and every pair of neighbouring slots across all eight by ΔE 8 and ΔE 15.

Several categorical slots fall below 3:1 on the light surface, so the chart vocabulary never relies on
colour alone to identify a category: it labels the marks directly, or names the level in the values table.
A host declaring its own `ChartPalette` is held to the contract's shape only; whether its colours separate
is for the host to measure.

The packaged face is Liberation Sans (`Liberation Sans, Arial, sans-serif`). The rasteriser embeds it, so
a PNG draws it on a machine with no fonts installed, and it is metric-compatible with Arial, so a browser
without it lays text out at the same widths. Its digits share one width, so numeric ticks line up. To
draw in your own face, measure it with the tool in a checkout of this repository (dev tooling, not
installed with the package; it needs the `[vega]` extra) and declare the result:

```bash
uv run python packages/evals/scripts/measure_font_metrics.py \
    --family "Inter, Arial, sans-serif" --font-dir /srv/fonts --out inter_metrics.json
```

```python
from threetears.evals.vega import load_chart_font

style = StyleProfile(chart_font=load_chart_font(Path("inter_metrics.json")))
```

## Bringing your own renderer

A host bringing its own renderer implements `ChartRenderer` (`draw(intent)`, and `drawn_data(drawing)`
reading its drawing back) and runs the one conformance check every renderer passes —
`assert_renderer_conforms(renderer, intents)`, from `threetears.evals.analysis.viz`: what it draws agrees
with the intent's marks (`data`), per identity, and the intent's values-as-drawn table agrees with those
marks (`table_disagreements`, policy rule 12) — so a drawing that passes agrees with the table beside it.
A table column spelled from a drawn number (a delta's `+72.7%`) or carried only by the table is the
builder's to spell, and is compared with nothing drawn.
