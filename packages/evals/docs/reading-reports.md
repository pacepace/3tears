# Reading reports

**For** anyone with a campaign (runs to compare) who wants to read what it found, put it in front of someone, or
embed it in a host. **Answers:** what each screen of a report says and where its numbers come from, which
statistical method produced each number, how arms are named, and how far a judged score can be trusted (evidence
tiers). Terms such as campaign, arm and analysis are defined in [Concepts](concepts.md).

## The report: one page, result first

Every campaign is read through one **report**, and every door into it returns the same page: the quick path's
`comparison.report`, the command line's `report`, the `report_read` operation behind the MCP action, and
`campaign_report(host, campaign_id, scope_id)`. Each door builds it with the one builder over the campaign's
evidence bundle, so none adds, drops or reorders anything. Most of what a reader needs is countable — who ran, what
got lost and why, which case failed, what the judge said — so code writes all of it; a model's analysis, when one
was written, adds only findings and recommendations, each pinned to the evidence it rests on.

```python
report = comparison.report                                   # the quick path
report = campaign_report(host, campaign_id, scope_id)        # a host
files = report.write("eval-report")    # report.html (one page, no script), report.md, bundle.json
```

`report.html` needs nothing beside it: no script, no stylesheet, no image, no URL but `#` links to places on the
page; its marks are inline SVG, and drill-down is `<details>`. `report.md` is the same page as Markdown, each mark
printed as the text it draws. `bundle.json` is the evidence bundle the page was computed from, with its
`schema_version`; the page names its fingerprint, and `AnalysisContextBundle.from_json(...).fingerprint()` over the
file returns the same one. **The bundle is the machine form**: there is no published JSON of the report, and a host
that wants its own pictures reads the bundle. `report_read(format="json")` and the command line's `--format json`
return it too. [`examples/3-invoices/extract.py`](../examples/3-invoices/extract.py) writes one, offline.

### The first screen

The invoices example (rung 3) renders this, verbatim:

```markdown
# invoices: does v2 do better than v1?
v1 (control) vs v2 · 15 Oct 2026 · 30 cases × 3 repeats

**Exploratory: nothing separated v2 from v1 on any reading. Not yet shown safe: v2 on no invented total.**

**Scoreboard**

| Arm | Ran | Usable | Passed | Failed by check | Failed by judge | Model failed | Cost per 1,000 replies |
|---|---|---|---|---|---|---|---|
| ▸ v1 | 90 | 90 | 73 | 17 | 0 | 0 | $0.0459 |
| v2 | 90 | 90 | 88 | 1 | 0 | 1 | $0.07036 |
```

A campaign with a question declared before the run (a host's two-model campaign, G2) opens the same way, with the question quoted
whole between the arms and the verdict:

```markdown
# G2: Kairo queue discipline
deepseek-v4.1-flash (production, control) vs deepseek-v4-flash (cheaper candidate) · 14 Oct 2026 · 4 cases × 3 repeats

Asked before the run: "On Kairo's queue-discipline regression set (dj_queue_discipline_v5), does deepseek-v4-flash hold every pass/fail queue-discipline dimension within 10 points of deepseek-v4.1-flash's pass rate, at lower cost per turn?"

**Keep production: deepseek-v4-flash passed 2 of 12 trials to deepseek-v4.1-flash's 10, and cost more per reply.**
```

The first screen carries the minimum a reader needs to grasp the result; every caveat and nuance is one step down.

- **Title**: a name — the campaign's (`G2: Kairo queue discipline`), with what it compares where no question was
  declared before the run (`invoices: does v2 do better than v1?`). A declared question is never cut down to a
  title.
- **Line 2**: each arm named in full once (`deepseek-v4.1-flash (production, control)`), then the date, then the
  design (`30 cases × 3 repeats`). After that an arm is its given label (`production`), the control marked `▸` in
  a table. Only line 2 and the verdict name a model.
- **Line 3**, where a question was declared before the run: the question quoted, `Asked before the run: "…"`. It
  is the contract the verdict reads, so it is never paraphrased, and it is quoted whole. Its decision rule is not
  words but typed clauses (`Question.decision`), so the Decision section states it in a sentence code writes.
- **Verdict**, written by code and never by a memo, result first: one sentence for the decision, a second only for
  a guardrail. The first rule that fires: the test setup lost too much (`No verdict: the test setup failed, not the
  models.`, the arm, the cause, the fix); one arm (`emails: 18/30 cases passed; 12 misses: double-charge,
  gdpr-delete, …`, each case by the name its author gave it, else `case N`); a declared decision rule, in
  plain counts with each arm by its model (`Keep production: deepseek-v4-flash passed 2 of 12 trials to
  deepseek-v4.1-flash's 10, and cost more per reply.`); which arm passed more, when an arm is under the case floor;
  a declared question's contrasts; `Exploratory:` (linked to the glossary) and what separated, or `nothing
  separated v2 from v1 on any reading`, never "no difference". A guardrail an arm is not shown safe on is the second sentence:
  `Not yet shown safe: v2 on no invented total.`
- **Scoreboard**: a bare table of counts and the cost per reply (per 1,000 replies when a reply costs under a cent).
  A trial passed when every check passed (a goal check, and on the quick path a `bool` grader or a field grader) and
  every judged dimension cleared its bar, and a case passed when every trial of it did, on the console as on the
  page. A failure cause beside checks and the judge heads its own column (`Model failed`) where it is the only one;
  the outcome bar under the table names each where there are several (in Markdown, its table only where it splits a
  cause the scoreboard lumps together). Under it, in normal weight, never bold, never first, the facts that change
  how the verdict is read and nothing else, in this order: the case floor; an arm's label that its own row
  contradicts (`'cheaper candidate' is not cheaper: it cost more per reply than production.`, latency only where it
  was read serially); an arm whose every trial failed. A "too few" line says what is withheld and why, in the
  reader's terms, never a bare count — and what is withheld is only the interval, never the gap, which is measured
  and stated: `4 cases per arm: keep follows from the counts above. Whether the gap holds on more cases is not
  estimated: an interval takes 5 independent cases.` where a declared rule settled keep, `Usable cases: v4.1-flash 3,
  v4-flash 1. Under 5 independent cases a rate is shown as a count, and whether the arms' gap holds on more cases is
  not estimated.` otherwise. Everything else is a step down: what was not charted and where a cost is a floor are in
  the Methods.

The floor counts independent cases, never trials or repeats. Below it no interval is estimated: no contrast is tested,
the verdict says too few, a pass rate (and pass^k) reads as a count with no estimate of how far off it could be, and no
gap between the arms is sized. From the floor up an interval is estimated, and its width says how precisely the gap is
known. A measure's mean is shown with its interval. A declared rule can still settle keep on what was observed, never
switch. So a host's seed gate covers every template whose results are compared across arms, whatever its declared
reading.

### The rest of the page

The sections follow in this order, each reached from a sticky nav; Cases, Trials, Not tested, Methods and the
Glossary start folded, and a nav link opens the one it lands on.

- **Decision**: the declared decision rule in one sentence code writes from its clauses, the arms by name (`Switch to
  v2 only if no pass rate drops by more than 2 points and v2 costs less per reply. Otherwise stay on v1.`; a clause
  naming one reading says `no tone rate drops by more than 2 points`), then one row per clause and reading, the
  clause's own sentence first, each arm's column headed by its name, then the clause's own outcome, `keep`,
  `switch` or `unsettled`, with the reason in words, the counts and the margin naming the arms (`keep: v2 passed 14
  of 60 to v1's 29, a drop of 25 points against a margin of 10`; `unsettled: v2's drop is not shown to stay within
  10 points (it could be as much as 53.6)`). One line under the table says what unsettled means: `Keep is the
  default: a switch needs every clause shown, so an unsettled rule keeps v1 for now. A clause reads the interval a
  switch reads, which can be wider than the contrasts table's.` So a clause can read keep or unsettled beside a
  contrast that is not separated. Keep settles on what the run observed: a drop past the margin, or a higher cost.
  Switch needs
  sampling evidence: every arm at the case floor, and for every rate the clause reads (the pass rate and each
  check's) the contrast's bounded interval keeping the drop within the margin; for cost, the interval wholly below
  zero. The bounded interval is not the one the Contrasts table prints, and is wider: it holds its error rate on a
  few coarse cases, where the t interval does not
  ([why](measuring-soundly.md#a-switch-reads-the-bounded-interval)). On a handful of cases only a sweep settles a
  10-point clause. A run that shows neither is unsettled ("the cost difference did not separate"); a bundle shaped
  before 55 reads "(this bundle predates the interval a switch needs)". A question stored with its rule written as
  prose in its words, and none typed, opens the section with `This question's rule was written as prose, which is
  no longer read; declare it as a rule.`, and its verdict is exploratory: no rule was applied.
- **What it means**: each finding as one sentence its writer wrote, unnumbered, and a table code built from the
  readings it chose (a check's pass rate as a count, `11 of 12`; money as `$`); its confidence, trials and caveats
  sit in a closed `Evidence` fold, one row with a column per fact so each header links its definition. Where a
  clause of a settled rule reads the finding's one reading, the sentence is code's: the two values plainly, then
  under the case floor one sentence of what is not estimated (`Cheaper candidate cost $3.301 per 1,000 replies to
  production's $3.092. Whether that gap holds on more cases is not estimated below 5 independent cases.`). Findings are
  evidence: one that names no reading is not a finding, so it is not laid out here but noted in one line under Not
  tested (`The analysis also noted: …`). Where a declared rule settled every clause, its question has no answer
  line: the verdict and the Decision table answer it; a clause left open is the one line (`Still open: whether …`).
  The memo's words are held to the page's style (below).
- **Contrasts**: one table headed by its control (`Against v1`), a row per reading: `Reading | ▸ control | arm | Change |
  Interval (99%) | Verdict | Cases`, each value with its interval, the change and its interval drawn in the row
  ([the marks](#the-marks)); one line instead when no contrast can resolve under the case floor. Then the effect
  size, pairing and adjusted p in a closed `Statistics` fold, the `Guardrails` table, and the `By stratum` and
  `Per label` tables, folded. A one-arm page contrasts nothing, so the section is headed **Readings** and lists
  each reading over all cases (its id stays `contrasts`).
- **Cases** and **Trials**: one row per case with each arm's marks (`✓ ✓ ✗`), its trial ids last and muted, the
  failed and lost ones first. The Flag column reads `regression` only where a candidate passed a smaller share of
  its usable trials on the case than the control (`✓ ✓ ✗` against `✓ ✓ ✓`); a tie or a gain never flags, and
  `no usable trial` marks a case an arm has none on. A case is named as its author named it (an `Eval`'s case `id`) where that reads as a
  name — one token, not a generated id, not a bare number — and `case N` otherwise. Then the trials' census in one
  line (`180 trials ran. Of them 161 passed and 19 failed.`) and only what went wrong: one row per failed or lost
  trial, the failed first, and each one's record under `T4 · double-charge · cheaper candidate · failed` (never the
  start of what the case asked), the first twelve open. A failed check is named by its expression, as code
  (``check: `call_count("music.queue_track") >= 2` ``), and a judge's fail by its dimensions (`judge: tone`), else
  `failed by the judge`. The passed trials are one closed `Passed trials` table
  (trial, case, arm, stored result), so each result id stays reachable with `result_get`; when every trial passed,
  the census line stands alone.
- **Not tested**: what the campaign could not speak to, one line each. A lever that ran at one level
  (`Lever model is unswept: one level ran.`) or that fewer than two results exercised
  (`research model: no research call was made on either arm.`), a case an arm has no usable trial of, a goal check
  not shown to beat doing nothing, the measures with no margin and how to declare one, each judge not yet calibrated,
  a finding that names no reading, the readings no declared question asked about, and, folded, a cost not measured.
  With none of these, `Nothing declared went untested.`
- **Methods**: a table (arms, judge, settings sent, test cost, the analysis's writer, the bundle); the coverage map
  in a closed `Coverage` fold (one row per lever: its status, how many findings name it, the next steps that do),
  which restates no finding; and the engine's own disclosures in a closed fold. Then the **Glossary**: every term of
  art the page links, in one plain sentence. Every column header that is a term of art links its entry, and in HTML a
  hover reads it.

### Style

`threetears.evals.analysis.style` holds every rendered sentence to one set of rules, `lint_prose`: no meta-prose
about the page ("rests on", "written by", "a count, not a ranking"), no hedge on a measured fact ("no measurable"),
no `key=value`, no chain of dashes, no sentence over 28 words or with more than three numbers (that is a table), no
header over three words, and the declared question only in the title. `humanize` is the one seam an identifier
passes on its way to a reader. A memo's sentence breaking a rule code can fix is dropped or replaced (a title
restating a measured number becomes the table's own row); one contradicting a settled decision rule becomes the
verdict line's sentence, and one saying the evidence cannot confirm a difference a settled clause decided becomes
code's counts on that reading, the size of the difference left to more cases; both are disclosed in Methods. What
is left is counted there (`style: N violations`) and listed.

### The marks

The page draws two marks, as inline SVG with no dependency. The **outcome bar**, under the scoreboard, is the page's
one chart: a census, so it is always drawn, one bar per arm with its label above it, passed, failed and lost to the
test setup cause by cause, each count written beside the bar, and a key naming only the states that occur. A
failure's causes share its one colour and differ in ink alone (solid, then a lighter tone), each alike on every bar,
and the key names each (`failed: check`, `failed: judge`); a lost cause's edge is dashed, the next dotted. An
**interval glyph** is drawn in the table row that states the figure (Tufte's table-graphic): the interval as a
whisker, the point as a dot, a zero line for a change, and a declared margin as a quiet band. Every cell of a column
in one unit shares one scale, written once under the table (`Change drawn from -7 to 36.7 pts`), so the rows compare
at a glance. There are no standalone interval charts: the tables carry the intervals.

No colour is the only encoding: a glyph is drawn in the page's ink and its verdict is the word in the next cell,
every figure is text in the row, the bar has a `<title>` and a `<desc>` stating its values, and a visually hidden
table beside it holds them for a screen reader. Colours come from the page's CSS variables, defined for light and
dark alike.

### Embedding the page in a host

Serve the directory `write` produces as static files; or keep a route that returns `report_read(format="html")`
with a strict content security policy (`default-src 'none'; style-src 'unsafe-inline'` holds: the page has no
script and inline SVG needs no `img-src`); or show that route in an `<iframe sandbox>` with no `allow-scripts`,
changing the host's own `frame-ancestors` to `'self'`. Linking to the route needs no change at all.

`Report.basis` says whether an analysis wrote the memo (`analysis`) or none did (`code_only`); `report.verdicts`
holds every verdict typed — each guardrail, each bar on each arm, each contrast — for a program to branch on.

## Reading a comparison

The Contrasts table (`multiple_comparisons` in the bundle) tests each arm against the control on each reading,
under one rig. How its columns, and the `Statistics` fold under it, are computed:

- **The test.** Per-case means (each case's repeats averaged first, because a case is the independent
  draw). When the arms share at least two cases, it is a paired t-test over the shared cases. Otherwise it is
  Welch's t statistic read on Hsu's `min(n_a, n_b) − 1` degrees of freedom. That is conservative at every
  size: on Welch–Satterthwaite's degrees of freedom, a 2-case side against 10 called false differences 10%
  of the time.
- **The change, the means, and `Cases`** (`30 paired`) are all over the cases the test read. A case only one arm
  ran is left out of a paired test, and the row counts it, so a mean here can differ from the cell's own mean.
- **The interval on the change** comes from the same test, at `1 − 0.05/m` for a family of `m` tested rows
  (Bonferroni), and the column's header states that level (`Interval (99%)` over five rows). All of a family's
  intervals cover together at least 95% of the time. A separation needs both
  an adjusted p below 0.05 and an interval that excludes zero, so the two never disagree: a row Holm's later
  steps would separate while its interval still touches zero reads not separated, its p below 0.05 notwithstanding.
- **Hedges' g** is the standardized difference, corrected for small samples. Cohen's d reads 0.88 for a
  true 0.5 at three paired cases; g reads 0.5. There is none at two pairs, where no unbiased estimate exists.
- **Pairing r** is the correlation between the two arms' per-case means over the paired cases: what pairing
  bought. Near 1, the cases' own differences cancelled and the paired test is far more sensitive than an
  unpaired one; near 0 pairing bought nothing; below 0 the paired test is less sensitive than an unpaired one
  would have been. Blank for an unpaired test, below three paired cases, and where an arm's values do not vary.
- **Adjusted p** is Holm-corrected within the family: one per declared question (the readings on its
  merit axes), or, with no question, one campaign-wide family over every reading on a merit axis. A measure
  on no axis is never tested. That keeps the rig's own readings out: the judge's time (`judge_ms`), the
  drain wait, and `cost_usd` and `program_cost`, which include what the judge spent. The candidate's spend
  is tested as `production_replicating_cost`.

Each verdict is printed naming both arms by label, never "the control", with a reading's better end in brackets
where it is the lower one: `v2 better than v1`, `v2 worse than v1 (lower is better)`, `v2 and v1 equivalent, within
the measure's margin`, `v2 and v1 not separated`. The verdict line names the reading too (`v2 better than v1 on
cost per reply (lower is better)`).

| Verdict | Means | Do |
|---|---|---|
| better / worse (typed `improved` / `regressed`) | adjusted p < 0.05, in that direction | act on it, unless the row says *immaterial* (below the measure's margin) |
| equivalent | the paired difference is shown inside ± the measure's margin by TOST, corrected in the same family | treat the arms as interchangeable on this reading |
| not separated | the cases could not tell the arms apart, or every shared case moved by exactly the same amount on a measure with no declared range, which no test of the mean can call (the row says so) | add cases, declare a margin, or declare `value_range`; never read it as a tie |
| untested | no test could decide: fewer than two cases on a side; the row says why | fix what it names (usually too few cases) |

Latency read while other cells or runs executed beside it is never in this table: the bundle withholds it
before anything reads it, a latency row whose arm has no other latency reads `untested` and says why, and
one line (`latency_contended`) names the arms. Launch with `measure_latency=True` to read latency clean.

Every report lists, under Not tested, each tested measure that declares no margin: one sentence, then a table
naming each once with how it takes one: a scorer's by its name (`grader(correct, margin=0.05)`), accuracy's on the
label's field grader (`grader("label", margin=0.05)`), and for an engine measure such as cost, that none can be
declared.

Every verdict here is also a typed value (`report.verdicts`, `Result.verdicts()`, the `outcome` key of
`Result.contrasts()`): its outcome, a reason code, the margin it read and where that came from
(`margin_source`: `measure`, `run` or, for a judged guardrail, `campaign`), its materiality, and whether it is a
guardrail. The printed verdict is rendered from it, so code branches on the outcome, never on the words. The
same values drive the CI gate ([`gate`](serving.md#gate)). The typed values are in the measure's own units; the
rows `Result.contrasts()` and `Result.guardrails()` return are the page's figures on the page's ruler (a share in
percent, its change in points, a cost per 1,000 replies where the page states it so), and each contrast row's
`unit` says which.

`equivalent` needs a margin and a paired test. A host declares a measure's margin
(`MetricDescriptor.materiality_threshold`); accuracy, whose description the engine owns, takes one declared on
the runs at launch (`grader("label", margin=...)`, `start_run(margins=...)`), read only when every run of
the campaign declares the same one, and its verdict says "declared on the runs". A host's own measures are read
as their runs were launched to read them: each run records its host's direction, guardrail, margin and range for
every measure (`EvalRun.declared_measures`), so a stored campaign read later through a host that declares them
otherwise reaches the same verdicts, and the report names the difference in one line. On a
measure that declares its range (`value_range`), as every pass rate and 1–5 score does, each one-sided test
is a bounded test by betting, which holds 5% for any distribution of differences on that range at any number
of cases. Coarse scores need that: a regression that fails one case in ten leaves twelve agreeing cases 28%
of the time, and a t-test, or the exact sign-flip reading of a difference with no spread that the engine
once used, called such samples equivalent up to three times in four. The price is that equivalence on a small margin
takes many cases, whatever the test. Arms that give the same answer on every case and repeat show a pass rate
within 0.25 from 12 shared cases and within 0.1 from 33, and a 1–5 score within 0.5 from 26, when it is the
only reading the family tests (no valid test could do it in fewer than 11, 29 and 23). Each further reading
in the family raises that: with a second (accuracy from a label's field grader, or a cost from `Answer`), a pass rate
takes 15 and 41. Arms that disagree on some cases need more: in a seeded simulation at k=2 where about one
answer in twelve departed from its case's usual one, a pass rate read `equivalent` within 0.25 in 18 of 20
runs at 24 cases, and within 0.1 in 16 of 20 runs at 100. Below that the row reads `not_separated`, which
claims nothing. A measure with a margin and no declared range
is not tested for equivalence at all, since no test of a mean holds 5% without one: its rows never read
`equivalent`, and the report names the measure once with the remedy, declare `value_range`. On the quick
path a scorer annotated `-> bool` is on 0 to 1 already, and one returning a number takes its range beside its
margin (`grader(rating, margin=0.5, range=(1, 5))`); a margin on it without one is refused
when the eval is built. The p is corrected in the same Holm family as the separations, with the multiplier capped at the number of
compared rows (Shaffer's refinement: a difference cannot be both zero and at least the margin), so the
chance that any verdict in the family is wrong stays at 5%.

No report states a change as a percent of the control: a Δ is in the reading's own unit. A reading that is a share
on 0–1 (a pass rate, an accuracy, a precision, a pass/fail grader or judged dimension) is stated as a percent to one
decimal, `98.9%`, and a change in it in points, `-1.1 pts`, with its interval `[-4.2, 2] pts`; the level every
interval holds at is said once, in the column's header (`Interval (99%)`). A cost per reply under a cent on every arm
is stated per 1,000 replies, and the heading says so (`Cost per 1,000 replies`, `Production cost ($/1,000 replies)`).

## Reading the guardrails

A guardrail is something an arm must not get worse on: a judged dimension on the `boundary` axis, or a
measure the host declared `guardrail`. On the quick path, `grader(fn, guard=margin)` declares one on a grader and
`Judge(guard={dimension: margin})` one on a judged dimension, every grade higher-is-better
([rung 3](ladder.md#rung-3-two-prompts-compared)). The Guardrails table
(`guardrails` in the bundle, under the contrasts on every report, `Result.guardrails()` in code) decides each one
for each arm, on its own 95% interval on `arm − control`. Guardrails are never in the contrasts table, the composite or pass^k, so a gain there
cannot hide a loss here.

| Decision | Means | Do |
|---|---|---|
| held | the whole interval is on the good side of the margin, and the reading declares its range | nothing; the arm did not get worse by more than you tolerate |
| breached | the whole interval is beyond the margin | do not adopt the arm, whatever it gained; the analysis writer is refused if it tries |
| undecided | the interval straddles the line, no interval exists, or the reading declares no range; the row says why | do not read it as safe; the contrasts open with a line naming the arm, and the verdict line says it. To decide it, add cases, declare a margin, or declare the reading's range |

The margin is the measure's `materiality_threshold`, and a judged dimension's is the one its campaign declares
(`CampaignDesign.guardrail_margins`). A guardrail with no margin declared is held at zero change: `held` then
needs the arm shown no worse at all, which a few cases rarely show.

**`held` is a claim of safety, so it is read only off a test that holds its error rate.** On a declared range
(`value_range` on a measure, `grader(..., range=)` on the quick path; a pass/fail and a judged scale have one already)
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
every finding is exploratory: line 2 of the report says so, every finding carries the `Scope` fact, and the analysis
writer may adopt nothing. A campaign
that declared no design at all is an exploratory campaign, and that sentence says so: its readings confirm
nothing, and the design its comparisons read was inferred from the runs, not declared
([when to declare one](measuring-soundly.md#start-from-the-question)). A guardrail is never exploratory.

**A question counts only if it was asked before the evidence.** A question confirms nothing when it was declared
after the campaign's first member run was created, since it may have been asked of results that already existed.
Each question's own `asked_at` is read against the earliest member run's `created_at`
(`preregistered_questions`), and every reader that scopes a reading by the questions (the comparison families, the
reading scope, the verdict order, the disclosure) reads that, so they cannot disagree. A late question builds no
family, its ids are listed as `reading_scope.post_hoc_questions` (and on the bundle as `post_hoc_questions`), and
`question_declared` is true only when a question was asked in time. When every live question came late, the one
disclosure says so and names them. To keep it from happening, a campaign's design is frozen once it holds a run:
`campaign_update` refuses `declared_design` in any form, and refuses re-pointing or clearing the control (which may
be named once, from a run, while none is declared); the name and description stay editable, and an amended design
is a new campaign, declared before launching into it.

## Excluded trials by cause

A trial is left out of an arm's measures for one of three causes (`trial_exclusion`): `infra`, the harness faulted
it; `truncated`, a judge reply was cut off at the output cap it was sent and was not bought again; and
`judge_cannot_tell`, the judge could not score a capability dimension, which leaves the trial out of pass^k and the
composite only. A candidate's own failure is never an exclusion: it fails, and is counted. The bundle's
`exclusions` gives every arm its trial count, its usable count and its exclusions by cause, with the caps the
truncated trials' judge calls were sent (`truncated_sent_caps`; `truncated_cap_unreported` when a client did not say,
never filled from the run's asked settings). A cell's `n_infra_excluded` still counts every trial its measures left
out, truncated ones included; `exclusions` says which cause each was.

The report's "lost to the test setup" is this classification, trial by trial: the scoreboard, the trials and the
verdict line read `trial_exclusion`, and the cap a truncated judge was sent comes from `exclusions`, never from the
run's asked settings. The scoreboard departs from `exclusions` in one place only, as pass^k does: a trial a failed
check or a sub-bar dimension already decided is counted failed, though the judge could not tell on another
dimension, while `exclusions` counts it under `judge_cannot_tell` because the measures needing every dimension
leave it out.

**Differential exclusion is flagged.** When one cause takes more than 10% of an arm's trials
(`DIFFERENTIAL_EXCLUSION_SHARE`), `differential_exclusions` names the arm, the cause and its share, beside every
other arm's share of the same cause under the same rig. Read it before any comparison of that arm: past the
threshold the arms are not compared over the same trials, and the lost ones were lost for a reason that may differ
by arm (a challenger whose long answers run its judge out of room, say). It is a flag, not a test.

## Methods

Every test is two-sided at α = 0.05, and every interval is 95% unless a correction widens it. The case is the
unit of analysis: a case's repeats are averaged first, because they are not independent draws.

| Number | Method |
|---|---|
| Interval on a mean | t on `n_cases − 1` degrees of freedom with a cluster-robust standard error over cases (Miller 2024), clipped to the measure's scale, or at zero for a time, a spend or a count (`nonnegative`). One case gives no interval. |
| Interval on a rate (accuracy, precision, recall, any 0/1 measure) | Wilson, on the effective sample size the clustering of repeats leaves, with t on `n_cases − 1` df. F1 has none. |
| Mean composite | The mean of each result's capability dimensions put on 0–1, per case first. Every pooled composite (run summary, compare, pivot cell, frontier point, history point, a lever's dispersion) names the dimension sets it was meaned over, and a pool whose results carried different sets is marked *ragged*: its mean averages different questions. |
| pass^k | An attempt passes when every goal-state check passed and every capability criterion reached the behavior's pass threshold (3 of 5 unless declared; recorded as `rubric_threshold`, printed as `pass^k (k=3, criterion >= 4 of 5)`). An attempt with no goal-state check and no judge, such as a classifier scored only against its expected label, has nothing to pass: it is left out and counted (`n_no_criterion_excluded`), and an arm with none measurable has no pass^k (`pass_hat_k_unmeasured_reason`), never 0. Unbiased C(c, k) / C(n, k) per case, averaged over the cases with n ≥ k, pooled across the runs of one cell; its interval is Clopper–Pearson on an effective size. |
| A contrast against the control | Paired t-test on per-case means over the shared cases (two or more), else Welch's t on Hsu's `min(n_a, n_b) − 1` df; a gap with no spread (every shared case moved by one amount, or each side constant) is read by the bounded test by betting on the measure's declared range, with its own interval and no g, and with no range is not separated and says why: the exact sign-flip test asks whether a move is symmetric, not whether the mean moved. Effect size Hedges' g (g_z when paired). Holm correction within each family; interval Bonferroni at 1 − α/m, clipped to the differences the measure's declared range allows (± its width). |
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
removed or renamed a declaration), or `evidence` (neither moved, so the runs, results or insights did).

Prior insights reach the bundle bounded: the newest live insight per claim, at most a fixed number of them, with
`prior_insights_omitted` counting the rest. `refused_merges` and `next_experiments` are capped the same way, with
`refused_merges_omitted` and `next_experiments_omitted`. A generation that restates a live insight's claim (same
words, ignoring case, spacing and a final period) replaces that insight in the ledger rather than adding a second
one, so regenerating over unchanged evidence leaves the ledger its size. Archiving the analysis that minted an
insight retracts it; each insight's `invalidation_trigger` states both rules. The `insights_list` and `insight_get`
actions show the ledger and each insight's standing (live, retracted or orphaned); `insight_delete` removes one wrong
insight.

**The memo is findings and recommendations, and code checks their structure.** The writer authors no headline, no
summary, no chart and no count: the first screen is code's. It reads the bundle's trials (their states, causes,
failed checks and the judge's reasons, without the excerpts) and names, in each finding's `trials`, the trials a
finding is drawn from. The generator refuses, through the one repair round, a recommendation that rests on no
finding, a finding naming a trial the bundle does not hold, and any `adopted` recommendation in a campaign that
declared no question before its first run — keeping the control included, since nothing in an exploratory campaign
is a decision. Adopting a non-control arm still needs an `improved` contrast.

**A finding's title never states a null its evidence did not establish.** "Halving the budget does not change
latency, cost or delivery" decides as surely as a measured result does, and over a contrast that did not separate it
asserts a result the campaign never measured. The bundle establishes a null in two places only: a contrast reading
`equivalent` (the equivalence test against a declared margin) and a guardrail reading `held`. The generator refuses
a title in a common null phrasing ("does not change", "no difference", "no effect", "no worse", "was unchanged",
"about the same", and a few more, listed in `analysis/null_claims.py`) when its finding cites no reading the bundle
established alike. The join is structural; only the phrase is read off the words, so a reworded null passes, and
the reporter eval measures the rest.

**The analysis keeps its evidence.** A stored analysis carries the bundle it was written over (`EvalAnalysis.bundle`),
so its report is laid out over that frozen evidence, whatever is archived or re-run afterwards.

The bundle stays closed: the generator has no tools to fetch more context, such as a `runs_bisect` or `scope_pivot`
drill-down. A generator that fetched its own context would read different inputs on every call, so nothing could
be fingerprinted before generation and two prompts could no longer be compared on one bundle. A question the
bundle cannot answer is answered by adding a field to it.
[`tests/fixtures/generate_analysis/llm_analysis.py`](../tests/fixtures/generate_analysis/llm_analysis.py) does all of it in one file.

## How a measure and a question are named

A measure is headed by its reader-facing name (`MetricDescriptor.reader_name`, frozen on the decision surface as
`MeasureFacts.reader_name`) in every table header and chart title, every evidence, contrast, guardrail and strata
row, and the memo. A judged dimension is printed under its rubric name with `(judged)` after it, and a reserved judge
axis as `transcript` or `outcome`. The key is still there to cite: on each `EvidenceRow`, and as `measure_id` on
each row `Result.contrasts()` returns (which filters on either the key or the heading).
A measure the surface froze no reader name for prints an engine measure by the engine's name, and a host
measure nothing describes by its key.

A declared question is printed as the words it was asked, both in the memo and in the contrasts table. Its id
is printed only where the declaration no longer holds that question. In the memo, each finding's evidence is
one table (Arm, Measure, Value, n, Spread), laid out like the report's evidence table.

## How an arm is named

Every block that names an arm — the title, the scoreboard, recommendations, evidence rows, the results by
stratum, the contrasts, the guardrails and labels tables, the cases and the outcome bar — prints one name for it,
built once (`threetears.evals.analysis.arms.arm_names`).

- An arm is named by the levers whose levels differ across the report's arms, each lever compared only
  among the arms it applies to; one that differs on none is named by its candidate kind and model.
- A lever that does not apply to the arm's kind is never named.
- An arm measured under more than one test setup adds `(test setup <digest>)` where a table names it per setup.
- Each level is put on one line and cut in the middle past 48 characters, alike in Markdown and HTML.
- Two arms that would still read alike carry their variant keys' digests (`(arm <digest>)`), so no two
  arms of one report share a name.

- A level the campaign declared a name for (the `display` of a value on its `SweptAxis`, such as "current text"
  for a prompt) is printed by that name rather than the host's display, which for a long text is a fingerprint
  (`cognitive_style: 2304 chars · 539ef3`). The name is matched on `(axis_id, content_hash)`; where one level
  is declared twice the first name wins, and an undeclared level keeps the host's display. Only the display
  changes, so no variant key moves.

What an arm ran in full is in the bundle's `variant_index`, not on the page: a report names an arm by what tells it
apart, never by every lever it carried.

*Example:* a campaign comparing the triage classifier's prompt v1 and v2 on one model names its arms by the
prompt lever alone (`ticket_router.prompt_version=v1` and `ticket_router.prompt_version=v2`, as `lever=level` pairs), since the model is the same on both.

## Did a lever take effect: mechanism checks and observed mechanisms

A lever that changed nothing and a lever that never took effect read alike in every outcome measure. A
`Sweepable` lever may name the measure or covariate it is supposed to move — `acts_on="context_tokens_in"` on a
chunk-width lever, say; a kind's overlay field does the same with `ActsOn(...)` beside `Ordinal()` and
`Interval(...)`, and an entry of a map field with `MemberActsOn({key: measure})` (an open family's
`member_acts_on`). Each coverage row of the analysis bundle then tests that measure across the lever's levels with
the same separation test the contrasts against the control use (per-case means, paired where the levels share
cases, Holm-corrected across the lever's pairs): `moved` when some pair separates; `inert` when every level was
observed, every pair could be tested and none separates — no measurable evidence the lever acted on its mechanism;
otherwise `unchecked` with the reason (`not_declared`, `not_swept`, `levels_unobserved`, `too_few_observations`,
`uniform_move_needs_range`). Each level's mean and its number of cases sit beside the state. Every case shifting by
the same amount has no spread for a t-test, and the exact sign-flip test asks whether a shift is symmetric about
zero, not whether the mean moved. So such a shift is read by the bounded test on the measure's declared
`value_range`, and on a measure with no range it reads `uniform_move_needs_range`, never `moved`.

The profile accepts only a numeric measure that each result carries as a single value: a covariate, a measure
your kind reports per result, or one of the result's own fields. It refuses anything else where you declare it,
including a quantity recorded per usage row (`reasoning_tokens`: declare `reasoning_ratio`) or only over a whole
run (`p95_total_ms`). For a call cap, report the calls each case used as a host measure and name that. The
declaration enters no variant key. A lever naming no mechanism reads `unchecked`, never as having taken effect.

A knob your host also records resolved as a lever of its own (`ResolvesInto`) is reported as one lever where
the runs show the resolved lever moved only with the knob: it then has no coverage row, names no confound, and
is listed in the arm's `folded`. Where it moved on its own it stays a lever and a confound, and a fold no two arms
at one knob level could have refuted, and no host residual reader checked, carries an `unverified_fold` confound.
The rule in full is in
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

## Which model answered

A run records the model id each call asked for. Which model answered is recorded only where the completion client
reported the model the provider's response named (`served_model`): on each candidate usage row and on each judge
score. Where it reported none, the record holds the requested id alone, and a provider may answer an id with a
model of its choosing. Nothing refuses such a run, and the package never guesses which ids float: that is the
provider's vocabulary. It says so instead, once, in one sentence every surface prints verbatim
(`model_identity_disclosure`): the run's summary (`run_get`, its `model_identity_warning` and `unconfirmed_models`),
`run_summary` (`model_identity_disclosure`), the analysis bundle (`model_identity_disclosure`, which the writer is
told never to read as the model that ran) and the campaign report's methods, whether an analysis wrote it or code
alone. Per role, the candidate then the judge, it counts the results whose candidate made a call no response named
a model for, and the scores no response named a model for, out of all of them, with the ids asked for and any
model the other responses did name:

```text
Model identity unconfirmed: the candidate (asked for vendor/model-latest) made calls whose response named no model
in 4 of 10 result(s) (the responses that named one named vendor/model-2026-05). The record keeps the model id asked
for, and a provider may answer an id with a model of its choosing (a floating alias), so these numbers cannot say
which model produced them; a completion client that reports the model its provider's response named
(served_model) confirms it.
```

A run whose responses all named their model prints nothing, and neither does one whose candidate called nothing.
The fix is in the host's completion client ([Adopting a host](adopting-a-host.md)); a quick candidate passes
`Answer(served_model=...)`.

## How the judge and the simulated user were sent

A run records what the engine asked its judge and its simulated user for (`judge_asked_settings`,
`simulator_asked_settings`). What each call actually carried is the client's to report (`sent_settings` on its
completion): a client may send the closest setting a model accepts (an effort word as a reasoning budget, to a model
that takes only budgets), or resend a strict call without enforcement when a router could not place it, and one that
reports nothing leaves the call unverified. Each score stores what its call was sent with, and each simulator usage row what its calls were. Cells
pool only when a role was both asked and sent alike, so unverified never pools with verified, and a judge sent other
settings is another judge for agreement and evidence tiers; each tier's sentence names what its judge was sent, or
says `request settings unconfirmed`.

Where a role was not shown to be sent as asked, one sentence says so, printed verbatim by the same surfaces as the
model-identity line (`request_settings_disclosure`): the run's summary (`run_get`, its `request_settings_warning`,
and `request_settings`, each role's reading, whose derived sent value is `as_asked`, `sent_otherwise`, `mixed` or
`unverified`), `run_summary`, the analysis bundle and the campaign report's methods. Per role, the judge (counted in
scores) then the simulated user (counted in results), it says, each where it applies, that the role was sent other
than asked, that its calls were sent more than one way with how many each way, that some calls did not report what
they were sent with, and that settings were sent without enforcement, so the provider may have ignored any the
serving model does not take:

```text
Request settings not confirmed as asked: the judge was asked for {max_tokens=10240, reasoning_effort=high, strict}
and sent {max_tokens=10240, reasoning_max_tokens=8192, strict} on 12 of 12 score(s). A client may send the closest
setting a model accepts, or resend a call without enforcement, and reports what each request carried
(sent_settings); runs sent differently, or one reported and one not, are not pooled as one apparatus.
```

It is never about the candidate: a host builds its candidate's client itself, and the engine asks it for nothing.
A run whose every judge and simulator call reported being sent, enforced, exactly as asked prints nothing.

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

**How it is shown.** Every report carries it as a `By stratum` table, folded under the contrasts: a row per arm and reading, the pooled
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

**On the quick path**, each case carries its kind under the reserved key `stratum`. The key is taken out of the
case before the runner, a grader or the judge is handed it, and stored as the test case's stratum. A report
comparing arms then carries the table above, which `Result.strata()` reads row by row, and every run's summary (`EvalSummary.strata`) reads each measure per
stratum, under `by stratum:` in `render()`, with the same too-few-cases warning.

**Identity.** A stratum enters no identity key, and two arms run over the same cases share their strata. It is
not part of `content_hash` either, which digests what the candidate is given. So re-labelling a case keeps the
case: an `Eval` stores a changed stratum on the same test case, and every run over it, earlier
ones included, is then read under the new one. A case's given `id` is stored on the same terms, as the test case's
`name`: the page names the case by it, and renaming a case keeps the case.

## How far a judged score can be leaned on: evidence tiers

A judged score is one model's opinion of another model's output. Before you act on one, you want to know
whether that judge has been shown to agree with people (best), or at least with itself when asked twice.
The engine measures both and labels every judged reading with the result — the **evidence tier**. It
never hides a reading for a weak tier; it tells you how much weight it can bear.

Every judged reading carries a tier that code decides from what the judge's reliability was measured to be
(`threetears.evals.kernel.evidence_tiers`): `evidence_tier` on each `judged_measures` arm and each judged
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
more would carry the bounds clear of the bar if agreement held at its estimate. Each analysis records the rule its
tiers were decided under (`judged_tier_rule`).

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
disagreement, never set aside. A judge is a served model, a judge config, the temperature its calls were
sent at and the request settings they were sent with, so a tier measured under one prompt, one temperature or one
set of sent settings never sets another's.

**What temperature a judge samples at.** Every judge call is requested at `DEFAULT_JUDGE_TEMPERATURE` (0)
unless the dimension's `JudgeConfig` states another, so no dimension is judged at a provider's default beside
others at 0. A model that refuses a temperature is sent none, and its score records `model_default`. A quick `Judge`'s client is handed the temperature when its `generate` takes a `temperature` keyword. Each score
records what was sent (`RubricScore.judge_temperature`), and that is part of the judge's identity: agreement
groups, tiers and apparatus comparisons are keyed by it, and a repeat at another temperature is unpaired. A score
made outside a judge phase (a host's own reviewers) records none and reads as not recorded, never as 0.

**Which prompt asked.** A judge is also the prompt that asked: a `JudgeConfig`'s id with a fingerprint of the
reply-format suffix the engine appended to it (`<config id>@<sha12>`), or, for a dimension with no config,
`builtin:<sha12>`, the identity of the engine's built-in wording. A release that rewords the built-in prompt or the
suffix makes a new judge, whose tiers and agreement are read apart from the old one's. A score made outside a
judge phase records no prompt identity, and reads as config `None`, a judge of its own
([judges and calibration](judges-and-calibration.md#step-4-run-the-judge)).

**Where tiers appear.** The bundle lists each judge's tier per dimension with both criteria
(`judge_evidence_tiers`); a finding stands on the weakest tier among its rows
(`FindingResolution.evidence_tier`: `mechanical`, `calibrated`, `separation`, `undetermined`, `incidental`
or `none`), which every report states beside the finding; a code-only report also states each judge's tier
with the numbers behind it. Tiers are flagged, never a reason to drop a reading.

**A tier read from a stored judge profile.** When a campaign's own evidence leaves a judge `undetermined` and a
judge campaign recorded a profile of that very judge and criterion that decides a tier, the tier is the profile's.
The entry's `from_profile` names the profile, when and on how many frozen cases it was measured, and the campaign's
own criteria, and the tier sentence says "read from the judge's stored profile". A changed model, prompt config,
temperature, sent request settings or criterion wording reads no profile
([storing a judge's profile](judges-and-calibration.md#storing-a-judges-profile-for-other-campaigns)).

**Measuring self-agreement: judge repeats.** Self-agreement is measured by **repeating** a finished run's
judge scores: `judge_repeat_work` (operation `judge_repeat`, a job; `estimate_judge_repeat` /
`judge_repeat_estimate` price it without a call) asks the same judge the same question again from the
evidence its first judge read, under the apparatus the run recorded, and records each answer beside the
score it repeats (`EvalResult.judge_repeats`) without changing the scores. Its spend is out-of-run
([Cost and budgets](cost-and-budgets.md#spend-outside-any-run)).
