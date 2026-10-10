# Reading reports

**For** anyone with a campaign (runs to compare) who wants to read what it found, render it, or build a renderer.
**Answers:** what each part of a report means, which statistical method produced each number, how arms are
named, how far a judged score can be trusted (evidence tiers), and how charts are drawn. Terms such as
campaign, arm, cell and analysis are defined in [Concepts](concepts.md).

## The report

Every campaign is read through one document, the **report**: blocks in reading order, `text` (what an analysis's
author wrote, with a role), `table`, `chart` and `disclosure` (what code must add), each linked to the findings it
belongs to or rests on. With an **analysis** (a model reading the campaign's numbers and writing findings), the
report carries those findings beside the evidence; without one it is a **code-only report**, every table and
chart code can build and a line saying no analysis was generated. Either way the decision surface's charts lead its
table, and a finding's evidence that compares arms is drawn ahead of its table unless the author's own chart drew:
the chart is the reading form, the table the audit form. Either way every number comes from code, and a
model never decides how much a judged score can be trusted. [`examples/reports.py`](../examples/reports.py) writes
one to files, offline. `analysis_report(storage, analysis_id, scope_id)` returns a generated analysis's `Report`:

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

- the campaign's declared questions, when it declares any, with the readings no question names
  ([exploratory](#readings-no-question-asked-about-exploratory));
- the arm table: each arm and every lever it ran, with no status column (every arm is unresolved, since
  nothing decided) and no finding column (there are no findings);
- the guardrails, each decided for each arm against the control ([below](#reading-the-guardrails));
- the decision surface, led by a distribution chart per judged dimension and per measure with a better end,
  drawn across the arms, except a label's statistics (they are in the labels table), `match` where `accuracy` is
  charted, a cost no result reported, and a reading only one arm drew (nothing to compare);
- the contrasts the evidence tested against the control;
- for a classifier, one `labels` table of each label's precision, recall and F1, a row per label and arm:
  precision and recall with their 95% Wilson intervals over the cases, F1 with none (it has none by construction), and
  every figure with the n it is counted over;
- the results by stratum when the cases declare strata ([below](#results-by-kind-of-case-strata));
- every disclosure the evidence carries, opening with one line saying no analysis was generated.

**What an analysis would add.** A code-only report is what code computed; nothing in it reads the numbers.
An analysis adds that reading: findings, each with the evidence it rests on and the caveats that qualify it;
a decision per declared question, with its confidence; which arm won and why; and what to run next. No
headline, finding, decision or answer to a declared question appears in a code-only report.

**Coverage, joined.** An analysis's report opens What to run next with a `coverage` table: each lever of the
coverage map, its status, the findings whose `axes` name it (or "no finding") and the next steps whose `lever`
names it. A `thin` or `unswept` lever no step names reads "no next step names it". A step naming a lever with
no coverage row is a proposal, not a gap, and says so beside it. The methods count the levers no finding names;
a measured lever with no finding may simply have had nothing to say, so nothing acts on the count.

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

Only an analysis stored before the writer was refused for adopting and rejecting one arm can carry
**contradicted**; neither verdict is then shown as standing, and a disclosure names it.

## Reading the decision surface

The decision surface is a row per cell (an arm under one rig) with every bar's verdict and the cost and
latency figures. **Its row order is not a ranking**, and the table says so in its caption (`order` on the
table block):

- The control's rows come first, marked `(control: the reference)`: every other arm is read against them.
  They lead because they are the reference, not because they were picked.
- The other arms follow in alphabetical order of their names, and an arm measured under several rigs lists
  its rigs by id.
- With no control row, no row is the reference, and the caption says that instead.

A bar on a judged dimension is only as trustworthy as its judges, so its verdict carries
`judge_evidence_tier` and the cell states it beside the value (`3.5 ± 0.25 (n=6), judge tier separation`).
A bar on a measured quantity or a goal check has none, and neither does a verdict stored before the tier
was recorded. A bar on `pass_hat_k` gets no row verdict, because no cell carries pass^k. The frontier reads
it instead, decides each contestant's pass^k interval against it by the same three-valued rule, and names
the variants below it (`frontier.bar`, each point's `bar_decision`). A bar on any other measure is never
passed to the frontier: `frontier_bar_withheld` says which bars exist and that the frontier's `n_cleared_bar`
is then no count.

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
  (Bonferroni). All of a family's intervals cover together at least 95% of the time. A separation needs both
  an adjusted p below 0.05 and an interval that excludes zero, so the two never disagree: a row Holm's later
  steps would separate while its interval still touches zero reads not separated, its p below 0.05 notwithstanding.
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
| not separated | the cases could not tell the arms apart, or every shared case moved by exactly the same amount on a measure with no declared range, which no test of the mean can call (the row says so) | add cases, declare a margin, or declare `value_range`; never read it as a tie |
| untested | no test could decide: fewer than two cases on a side; the row says why | fix what it names (usually too few cases) |

Latency read while other cells or runs executed beside it is never in this table: the bundle withholds it
before anything reads it, a latency row whose arm has no other latency reads `untested` and says why, and
one line (`latency_contended`) names the arms. Launch with `measure_latency=True` to read latency clean.

A report from `compare` prints one line above this table when a tested measure declares no margin: "No margin
is declared on …, so no contrast on it can read equivalent", with how to declare one for each kind of measure:
a scorer's by its name (`compare(margins={"correct": 0.05})`), accuracy's the same way
(`compare(margins={"accuracy": 0.05})`), and for an engine measure such as cost, that none can be declared.

Every verdict here is also a typed value (`report.verdicts`, `Comparison.verdicts()`, the `outcome` key of
`Comparison.contrasts()`): its outcome, a reason code, the margin it read and where that came from
(`margin_source`: `measure`, `run` or, for a judged guardrail, `campaign`), its materiality, and whether it is a
guardrail. The printed verdict is rendered from it, so code branches on the outcome, never on the words. The
same values drive the CI gate ([The command line](command-line.md#gate)).

`equivalent` needs a margin and a paired test. A host declares a measure's margin
(`MetricDescriptor.materiality_threshold`); accuracy, whose description the engine owns, takes one declared on
the runs at launch (`compare(margins={"accuracy": ...})`, `start_run(margins=...)`), read only when every run of
the campaign declares the same one, and its verdict says "declared on the runs". On a
measure that declares its range (`value_range`), as every pass rate and 1–5 score does, each one-sided test
is a bounded test by betting, which holds 5% for any distribution of differences on that range at any number
of cases. Coarse scores need that: a regression that fails one case in ten leaves twelve agreeing cases 28%
of the time, and a t-test, or the exact sign-flip reading of a difference with no spread that the engine
once used, called such samples equivalent up to three times in four. The price is that equivalence on a small margin
takes many cases, whatever the test. Arms that give the same answer on every case and repeat show a pass rate
within 0.25 from 12 shared cases and within 0.1 from 33, and a 1–5 score within 0.5 from 26, when it is the
only reading the family tests (no valid test could do it in fewer than 11, 29 and 23). Each further reading
in the family raises that: with a second (accuracy from `expected=`, or a cost from `Answer`), a pass rate
takes 15 and 41. Arms that disagree on some cases need more: in a seeded simulation at k=2 where about one
answer in twelve departed from its case's usual one, a pass rate read `equivalent` within 0.25 in 18 of 20
runs at 24 cases, and within 0.1 in 16 of 20 runs at 100. Below that the row reads `not_separated`, which
claims nothing. A measure with a margin and no declared range
is not tested for equivalence at all, since no test of a mean holds 5% without one: its rows never read
`equivalent`, and the report names the measure once with the remedy, declare `value_range`. On the quick
path a scorer annotated `-> bool` is on 0 to 1 already, and one returning a number takes its range beside its
margin (`compare(margins={"rating": 0.5}, ranges={"rating": (1, 5)})`); a margin on it without one is refused
at the call. The p is corrected in the same Holm family as the separations, with the multiplier capped at the number of
compared rows (Shaffer's refinement: a difference cannot be both zero and at least the margin), so the
chance that any verdict in the family is wrong stays at 5%.

A delta-table chart states each row's change relative to the baseline only on a ratio scale. A judged 1–5
score (`MetricDescriptor.scale="interval"`) moves in points: one point up is +50% from 2 and +25% from 4,
so no percent is stated for it.

## Reading the guardrails

A guardrail is something an arm must not get worse on: a judged dimension on the `boundary` axis, or a
measure the host declared `guardrail`. On the quick path, `compare(guardrails=...)` declares either, each with
its margin and direction ([tutorial](tutorial.md#7-hold-a-guardrail)). The "Guardrails against the control" table
(`guardrails` in the bundle, its own section of both reports, `Comparison.guardrails()` in code) decides each one
for each arm, on its own 95% interval on `arm − control`. Guardrails are never in the contrasts table, the composite or pass^k, so a gain there
cannot hide a loss here.

| Decision | Means | Do |
|---|---|---|
| held | the whole interval is on the good side of the margin, and the reading declares its range | nothing; the arm did not get worse by more than you tolerate |
| breached | the whole interval is beyond the margin | do not adopt the arm, whatever it gained; the analysis writer is refused if it tries |
| undecided | the interval straddles the line, no interval exists, or the reading declares no range; the row says why | do not read it as safe. An arm can still be adopted, and the decision then carries a `Guardrails` fact naming it. To decide it, add cases, declare a margin, or declare the reading's range |

The margin is the measure's `materiality_threshold`, and a judged dimension's is the one its campaign declares
(`CampaignDesign.guardrail_margins`). A guardrail with no margin declared is held at zero change: `held` then
needs the arm shown no worse at all, which a few cases rarely show.

**`held` is a claim of safety, so it is read only off a test that holds its error rate.** On a declared range
(`value_range` on a measure, `ranges=` beside a quick scorer; a pass/fail and a judged scale have one already)
the interval is the bounded test by betting (Waudby-Smith & Ramdas), each end a one-sided 2.5% bound that holds
for any values on the range at any number of cases. The t interval it replaced read `held` up to 9.5% of the time
at the margin on coarse, skewed scores (a rare four-point drop on 1-5, pass/fail flips), against 2.5%. The price
is the truth about coarse data: a perfect record shows a pass/fail guardrail within 0.1 of the control only from
41 cases, and within 0.05 from 83, since fewer agreeing cases cannot rule out a rare drop. The bounded test
bets in a fixed pseudo-random order, so the order cases were listed in cannot steer it. **With no declared range a
guardrail is never `held`**: no test of a mean holds its rate there (an unbounded value can hide a rare large
move), so the row reads `undecided` and says to declare the range. It can still read `breached`, off the t interval,
which blocks an arm rather than clearing one; that claim's rate is not guaranteed on skewed values
([open problems](open-problems.md#a-guardrail-is-held-only-on-a-declared-range-accepted-limit)). The interval
column marks such a row `(t: no declared range)`. An `undecided` guardrail does not block adoption because at a few cases and no margin almost every
guardrail is undecided, and a rule that blocked them all would block every adoption.

**On the frontier.** The frontier holds each contestant's judged guardrails against the campaign's control by the
same rule. A contestant that breaches one is disqualified, and its row names the dimension (`disqualified_by`).
One that does not hold every guardrail is never the frontier's pick. With no control there is nothing to hold a
contestant against: the subject's `boundary_pillar` says the pillar was not checked, and a verdict lists the
dimensions it was not checked on (`boundary_unchecked`).

## Readings no question asked about: exploratory

A campaign's declared questions say what it set out to learn. A reading on no axis a live question names is
**exploratory**: worth reporting as a lead for the next campaign, never as a confirmed answer. Where
questions are declared, the bundle lists those readings (`reading_scope`), the code-only report names them
under the questions, and a finding resting only on them carries a `Scope` fact. Where none are declared,
every finding is exploratory, and the report says so once at the top rather than on every row. A campaign
that declared no design at all is an exploratory campaign, and that sentence says so: its readings confirm
nothing, and the design its comparisons read was inferred from the runs, not declared
([when to declare one](choosing-a-design.md)). A guardrail is never exploratory.

## Methods

Every test is two-sided at α = 0.05, and every interval is 95% unless a correction widens it. The case is the
unit of analysis: a case's repeats are averaged first, because they are not independent draws.

| Number | Method |
|---|---|
| Interval on a mean | t on `n_cases − 1` degrees of freedom with a cluster-robust standard error over cases (Miller 2024), clipped to the measure's scale, or at zero for a time, a spend or a count (`nonnegative`). One case gives no interval. |
| Interval on a rate (accuracy, precision, recall, any 0/1 measure) | Wilson, on the effective sample size the clustering of repeats leaves, with t on `n_cases − 1` df. F1 has none. |
| Mean composite | The mean of each result's capability dimensions put on 0–1, per case first. Every pooled composite (run summary, compare, pivot cell, frontier point, history point, a lever's dispersion) names the dimension sets it was meaned over, and a pool whose results carried different sets is marked *ragged*: its mean averages different questions. |
| pass^k | An attempt passes when every goal-state check passed and every capability criterion reached the behavior's pass threshold (3 of 5 unless declared; recorded as `rubric_threshold`, printed as `pass^k (k=3, criterion >= 4 of 5)`). An attempt with no goal-state check and no judge, such as a classifier scored only against its expected label, has nothing to pass: it is left out and counted (`n_no_criterion_excluded`), and an arm with none measurable has no pass^k (`pass_hat_k_unmeasured_reason`), never 0. Unbiased C(c, k) / C(n, k) per case, averaged over the cases with n ≥ k, pooled across the runs of one cell; its interval is Clopper–Pearson on an effective size. |
| A contrast against the control | Paired t-test on per-case means over the shared cases (two or more), else Welch's t on Hsu's `min(n_a, n_b) − 1` df; a gap with no spread (every shared case moved by one amount, or each side constant) is read by the bounded test by betting on the measure's declared range, with its own interval and no g, and with no range is not separated and says why: the exact sign-flip test asks whether a move is symmetric, not whether the mean moved. An analysis assembled before 0.66 read that gap by the sign-flip test and keeps its verdicts. Effect size Hedges' g (g_z when paired). Holm correction within each family; interval Bonferroni at 1 − α/m, clipped to the differences the measure's declared range allows (± its width). |
| `equivalent` | Paired TOST against the measure's `materiality_threshold`, in the same Holm family, capped at the number of compared rows (Shaffer); each one-sided test is the bounded test by betting (Waudby-Smith & Ramdas) on the measure's declared range, which holds α for any distribution on the range at any n; a measure with no declared range is not tested for equivalence. |
| A bar | Three-valued: the cell's interval against the threshold less the margin (cleared, missed, undecided). A seeded threshold is the incumbent's mean moved √2 − 1 of its half-width toward the permissive end. |
| A guardrail | Non-inferiority: the 95% interval on arm − control against zero change less the margin. On a declared range the interval is the bounded test by betting (each end a one-sided 2.5% bound; paired on the differences, unpaired each arm's mean at 1.25% and the gap between them); with no declared range it is the comparison's t interval, which can read `breached` but never `held`. |
| Scope divergence, mechanism checks | The difference tested directly, paired or Welch as for a contrast; a gap with no spread is read by the bounded test on the measure's declared range (a remainder's range from the whole's and the part's), and with none is never called: a mechanism reads `uniform_move_needs_range`, a divergence is counted untested. |
| Frontier | Dominance by the contrasts' test, Holm across the subject's pairs (a gap of one amount on cost or latency, which declare no range, is never shown); latency ranked on the mean; p95 median-unbiased (Hyndman–Fan type 8) from 13 observations; cost band a lognormal prediction band. |
| Run history | Paired test per adjacent pair of runs, uncorrected (every case moved by one amount: the bounded test on the declared range, or `not_separated` with the reason); `equivalent` by the same bounded TOST against the threshold, on the measure's declared range (with none, untested, and each step's flag says why). |
| Detectable difference (a launch estimate) | The smallest true difference the paired t-test on `n` cases finds with 80% power at α/m (Holm's first step over the `m` comparisons planned), by the noncentral t. The variance is measured on earlier runs of the template: repeat noise, plus how far two arms disagree about a case (or one arm's case spread, twice, where no earlier pair shares cases). Assumes near-normal per-case differences and one real difference in the family. |
| Judge agreement and evidence tiers | Cohen's κ, quadratic-weighted on 1–5; tiers decided on a score interval for κ (one-sided 95% lower bound to award, 97.5% upper bound to deny). |

The [simulation suite](measuring-soundly.md) checks each method's error rate against a known truth.

## Having a model write the analysis, over frozen evidence

An analysis is written from the campaign's **bundle** alone (`AnalysisContextBundle`), and every figure
in it is a reference code resolves against that bundle. Save the bundle (`bundle.to_json()`), reload it
(`AnalysisContextBundle.from_json`) and its `fingerprint()` is unchanged, so `generate_analysis` over the
saved file under a second prompt compares the two prompts and nothing else; each analysis records the
fingerprint it read on `generation.bundle_fingerprint`. The insight-ledger cutoff (`bundle_assembled_at`),
the generation time and its cost are on that provenance, not in the bundle.

The provenance also records the bundle `schema_version` the generation ran over and the
`host_declarations_digest`, a digest of the host's declared sweepables and world dimensions computed from its
registries. Re-assembling a stored analysis's bundle (`inspect_analysis_bundle`) compares fingerprints, and when they
differ `mismatch_cause` names why: `package_shape` (the bundle schema moved), `host_declarations` (the host added,
removed or renamed a declaration), `evidence` (neither moved, so the runs, results or insights did), or `cannot_say`
for an analysis stored before both were recorded.

Prior insights reach the bundle bounded: the newest live insight per claim, at most a fixed number of them, with
`prior_insights_omitted` counting the rest. `refused_merges` and `next_experiments` are capped the same way, with
`refused_merges_omitted` and `next_experiments_omitted`. A generation that restates a live insight's claim (same
words, ignoring case, spacing and a final period) replaces that insight in the ledger rather than adding a second
one, so regenerating over unchanged evidence leaves the ledger its size. Archiving the analysis that minted an
insight retracts it; each insight's `invalidation_trigger` states both rules.

The bundle stays closed: the generator has no tools to fetch more context, such as a `bisect_runs` or `pivot`
drill-down. A generator that fetched its own context would read different inputs on every call, so nothing could
be fingerprinted before generation and two prompts could no longer be compared on one bundle. A question the
bundle cannot answer is answered by adding a field to it.
[`examples/llm_analysis.py`](../examples/llm_analysis.py) does all of it in one file.

## How a measure and a question are named

A measure is headed by its reader-facing name (`MetricDescriptor.reader_name`, frozen on the decision surface as
`MeasureFacts.reader_name`) in every surface table header, chart title and axis label, every evidence,
contrast, guardrail and strata row, and the memo. A judged dimension is printed under its rubric name with
`(judged)` after it. The key is still there to cite: on `SurfaceColumn.measure_id`, on each `EvidenceRow`, and
as `measure_id` on each row `Comparison.contrasts()` returns (which filters on either the key or the heading).
An analysis frozen before measures had reader names prints an engine measure by the engine's name, and a host
measure by its key, since that is all the analysis recorded.

A declared question is printed as the words it was asked, both in the memo and in the contrasts table. Its id
is printed only where the declaration no longer holds that question. In the memo, each finding's evidence is
one table (Arm, Measure, Value, n, Spread), laid out like the report's evidence table.

## How an arm is named

Every block that names an arm — decisions, evidence rows, the arm table, the decision surface, the
results by stratum, the contrasts, the guardrails and labels tables, and every chart group — prints one name for it, built once
(`threetears.evals.analysis.arms.arm_names`).

- An arm is named by the levers whose levels differ across the report's arms, each lever compared only
  among the arms it applies to; one that differs on none is named by its candidate kind and model.
- A lever that does not apply to the arm's kind is never named.
- A cell adds `@ rig <digest>` when its arm was measured under more than one rig.
- Each level is put on one line and cut in the middle past 48 characters, alike in Markdown, HTML and the
  chart intents.
- Two arms that would still read alike carry their variant keys' digests (`(arm <digest>)`), so no two
  arms of one report share a name.

- A level the campaign declared a name for (the `display` of a value on its `SweptAxis`, such as "current text"
  for a prompt) is printed by that name rather than the host's display, which for a long text is a fingerprint
  (`cognitive_style: 2304 chars · 539ef3`). The name is matched on `(axis_id, content_hash)`; where one level
  is declared twice the first name wins, and an undeclared level keeps the host's display. Only the display
  changes, so no variant key moves.

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
(identity, length, position, interval with what it varies over, level, class, ordinal, count, label), its axes with
their units and zero baselines, its order, the colour *slots* it uses and what it must disclose — plus its
values as drawn, which the HTML shows as a table. An interval is drawn as a band only over 5 or more cases
(`SMALL_N_BAND_FLOOR`): below that a distribution draws each case's value as a point and says why, a timeseries
leaves a stated gap, and a null result is refused. The interval itself still appears in the tables.

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
otherwise `unchecked` with the reason (`not_declared`, `not_swept`, `levels_unobserved`, `too_few_observations`,
`uniform_move_needs_range`). Each level's mean and its number of cases sit beside the state. Every case shifting by
the same amount has no spread for a t-test, and the exact sign-flip test asks whether a shift is symmetric about
zero, not whether the mean moved. So such a shift is read by the bounded test on the measure's declared
`value_range`, and on a measure with no range it reads `uniform_move_needs_range`, never `moved`. An analysis
assembled before 0.66 read it by the sign-flip test and keeps the state it was assembled with.

The profile accepts only a numeric measure that each result carries as a single value: a covariate, a measure
your kind reports per result, or one of the result's own fields. It refuses anything else where you declare it,
including a quantity recorded per usage row (`reasoning_tokens`: declare `reasoning_ratio`) or only over a whole
run (`p95_total_ms`). For a call cap, report the calls each case used as a host measure and name that. The
declaration enters no variant key. A lever naming no mechanism reads `unchecked`, never as having taken effect.

A knob your host also records resolved as a lever of its own (`ResolvesInto`) is reported as one lever where
the runs show the resolved lever moved only with the knob: it then has no coverage row, names no confound, and
is listed in the arm's `folded`. Where it moved on its own it stays a lever and a confound, and a fold no two arms
at one knob level could have refuted carries an `unverified_fold` confound. The rule in full is in
[Adopting the engine](adopting-a-host.md#the-kind-what-you-are-evaluating).

A comparison across candidate models can also differ in what the models did while no setting differed. A
reasoning effort is a word each vendor maps to its own budget, so two models at one effort setting can reason very
differently. The bundle reads each arm's mean reasoning share (`reasoning_ratio`) into `arm_mechanisms`. Where two
models' shares are at least `REASONING_SHARE_DIVERGENCE` (0.20) apart, the comparison names an
`observed_mechanism` confound carrying the threshold and both models' values: on the model coverage row and its
divergences (`confounded_by`), and on each pairwise contrast against the control (`mechanism_confounds` on the
design's contrast arms and on each family comparison). On any other lever the share moving is what the lever did,
so it is never named there. The confound qualifies the comparison; it never hides it. A share nothing measured is
said to be unmeasured and names no confound.

A floating model alias (a "latest" pointer) is resolved by the provider, so two runs of one arm can have been
answered by different models. Each candidate usage row records the model the response named
(`RoleUsage.served_model`), and the bundle reads it into `arm_served_models`: per arm, `one`, `pooled` (its
numbers are a mixture) or `unrecorded`. Where one requested id was answered by more than one model across a
comparison's runs, the comparison names a `served_model:candidate` confound (`undecided` where some response
named no model). The arm still pools under its key, fixed at launch, so the mixture is disclosed, not split.
The read lenses outside the bundle use the same reading (`ServedModelReading`): each frontier point and its
verdict, each history series and each of its points, each pivot cell on a table grouped by `variant_key` or
`model` (with `served_model_disclosure` on the table), and each side of `compare_two_runs` (`served_models_a`,
`served_models_b`). `served_model` is also a coordinate: pivot on it to read each model alone, and the export
carries it as a column. A response that named no model reads `unrecorded`, never the alias.

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

Every judged reading carries a tier that code decides from what the judge's reliability was measured to be
(`threetears.evals.contracts.evidence_tiers`): `evidence_tier` on each `judged_measures` arm and each judged
reading on the decision surface, and `judged_tier` on a finding's resolved evidence row. Code decides it, never
the analysis writer, because how far a score can be leaned on depends on two measurements of the judge and on
nothing a report's author says:

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

**Why these bounds.** Seeded simulation compared a score interval for kappa with the analytic standard error
and two bootstraps over six marginals (`tests/test_simulated_agreement.py`). At 20 results the score interval
showed a judge exactly at the bar over it 3.0–3.2% of the time, against 12–34% for the others, which read their
spread off the estimate and so make a sample that happens to agree look certain. The disagreement size used is
the larger of the observed and the chance one, a miss is decided on the 97.5% upper bound because the upper
bound runs looser, and raters of one result are treated as a cluster.

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
Every distinct result weighs 1, split across the raters that measured it, so many raters re-measuring a few
shared results cannot carry the figure, or the floor, over the bar. A repeat that answers "can't tell" where the judge had scored is a
disagreement, never set aside. A judge is a served model, a judge config and the temperature its calls were
sent at, so a tier measured under one prompt or one temperature never sets another's.

**What temperature a judge samples at.** Every judge call is requested at `DEFAULT_JUDGE_TEMPERATURE` (0)
unless the dimension's `JudgeConfig` states another, so no dimension is judged at a provider's default beside
others at 0. A model that refuses a temperature is sent none, and its score records `model_default`. A quick `Judge`'s client is handed the temperature when its `generate` takes a `temperature` keyword. Each score
records what was sent (`RubricScore.judge_temperature`), and that is part of the judge's identity: agreement
groups, tiers and apparatus comparisons are keyed by it, and a repeat at another temperature is unpaired. A score
or run stored before this recorded none and reads as not recorded, never as 0; such a run is not re-judged or
repeated, and its cells keep their old ids.

**Where tiers appear.** The bundle lists each judge's tier per dimension with both criteria
(`judge_evidence_tiers`); a finding stands on the weakest tier among its rows
(`FindingResolution.evidence_tier`: `mechanical`, `calibrated`, `separation`, `undetermined`, `incidental`
or `none`), which every report states beside the finding; a code-only report also states each judge's tier
with the numbers behind it. Tiers are flagged, never a reason to drop a reading.

**Measuring self-agreement: judge repeats.** Self-agreement is measured by **repeating** a finished run's
judge scores: `repeat_judge_scores` (operation `judge_repeat`; `estimate_judge_repeat` /
`judge_repeat_estimate` price it without a call) asks the same judge the same question again from the
evidence its first judge read, under the apparatus the run recorded, and records each answer beside the
score it repeats (`EvalResult.judge_repeats`) without changing the scores. Its spend is out-of-run
([Cost and budgets](cost-and-budgets.md#spend-outside-any-run)).

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
`packaged_palette("dark")`), a published categorical palette ordered so the four validated slots can be told
apart as a set. The package's tests hold text to 4.5:1 against the chart surface, slot 1 (every single-series
mark) and `context` to 3:1, and slots 1-4 to an OKLab ΔE of at least 6 under simulated protanopia
and deuteranopia. Several categorical slots fall below 3:1 on the light surface, so a chart never relies on
colour alone to identify a category: it labels the marks or names the level in the values table. A host's own
`ChartPalette` is held to the contract's shape only.

The packaged face is Liberation Sans (`Liberation Sans, Arial, sans-serif`), embedded by the rasteriser and
metric-compatible with Arial, with digits of one width. To draw in your own face, measure it with the tool in a
checkout of this repository (it needs the `[vega]` extra) and declare the result:

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
