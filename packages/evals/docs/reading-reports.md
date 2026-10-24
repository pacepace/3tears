# Reading reports

**For** anyone with a campaign (runs to compare) who wants to read what it found, put it in front of someone, or
embed it in a host. **Answers:** what each screen of a report says and where its numbers come from, which
statistical method produced each number, how arms are named, and how far a judged score can be trusted (evidence
tiers). Terms such as campaign, arm and analysis are defined in [Concepts](concepts.md).

## The report: one page, result first

Every campaign is read through one **page**, and every door into it returns the same one: the quick path's
`result.page`, the command line's `report`, the `report_read` operation behind the MCP action, and
`campaign_report(host, campaign_id, scope_id)`. It opens on the campaign's first screen, the whole decision at a
glance, and then holds one fold per aspect, each that aspect's own page. Each door builds it with the one builder over
the campaign's evidence, so none adds, drops or reorders anything. Most of what a reader needs is countable — who
ran, what got lost and why, which case failed, what the judge said — so code writes all of it; a model's analysis,
when one was written, adds only a summary across the aspects, findings and a recommendation, each finding and the
recommendation pinned to the evidence it rests on.

```python
page = result.page                                           # the quick path
page = campaign_report(host, campaign_id, scope_id)          # a host
files = page.write("eval-report")  # report.html, report.md, campaign.json, bundle-<aspect>.json
```

`report.html` needs nothing beside it: no script, no stylesheet, no image, no URL but `#` links to places on the
page; its marks are inline SVG, and drill-down is `<details>`. `report.md` is the same page as Markdown, each mark
printed as the text it draws. `campaign.json` is the page's machine form: code's readings, the recommendation's arm
(by label and variant key) and confidence, the call, and each aspect's bundle by its fingerprint
([its shape](serving.md#the-campaign-json)). Each aspect's evidence bundle is written whole beside it as `bundle-<aspect>.json`, with its `schema_version`;
`AnalysisContextBundle.from_json(...).fingerprint(like=named)` over the file returns the fingerprint the page names
(`like` computes it in the named one's version: see below). A host that
wants its own pictures reads the bundles. `report_read(format="json")` and the command line's `--format json` return
`campaign.json`. [`examples/3-invoices/extract.py`](../examples/3-invoices/extract.py) writes one, offline.

### The campaign's first screen

What a campaign can decide depends on its arms ([the three kinds](concepts.md#campaign-kind)), and so does its
first screen. Top to bottom:

| | A change (control and candidates) | A choice (arms, no control) | A measurement (one arm) |
|---|---|---|---|
| 1 | each missed bar, then each broken floor | each missed bar | each missed bar |
| 2 | title and question | title and question | title and question |
| 3 | the writer's summary | the writer's summary | the writer's summary |
| 4 | the writer's recommendation | the writer's recommendation, or no clear winner | **By aspect, measure by measure** |
| 5 | **By aspect**: each candidate against the control, then **To show no worse than the margin** | **By aspect, pair by pair** | **Bars** |
| 6 | **Bars**, then the call | **Bars**, then the call | |

The warnings are bold, above the title (in HTML each a `<p role="alert">`), and the writer's why sits beside each
one the recommendation answers. A warning only warns: nothing on the page picks an arm, and the bars and floors decide
nothing. The summary is the memo's one sentence across every aspect, written for every kind, so a measurement and a
choice with no clear winner open on what they found too. A page with no memo has no summary, and says where the
recommendation goes that none was written (a measurement, which recommends nothing, says nothing there).

`analyses_list` lists each stored analysis by the same warnings, word for word and without the writer's why, then
its verdict: the aspect page's verdict line for one aspect, what code read on each aspect for several
(`billing: candidate shown worse; tone: candidate shown better`). With no missed bar and no broken floor, the line is
the verdict alone.

A change of two aspects, one regressing past its floor and one improving, with a memo recommending production,
opens:

```markdown
**Bar missed on tone: candidate misses the bar on Measuring spend, all roles: its whole interval is on the bad side of the bar.**

**Floor broken on billing: candidate's pass rate falls past the 10-point margin on its whole interval, a drop of 24.6 to 88 points.**

# support queue
"Does the candidate answer support requests as well as production?"

The candidate answers in the house tone far more often but fails billing, so it is not yet the better arm.

**Recommended:** production (high confidence)

- Billing is where refunds are lost, and the candidate breaks its floor there.

**By aspect**

| Aspect | Arm | Cases | ▸ production | Arm passed | Change (pts) | Interval (pts) | Margin (pts) | Reading |
|---|---|---|---|---|---|---|---|---|
| billing | candidate | 20 | 54 of 60 | 18 of 60 | -60 | [-90.8, -19.9] | 10 | shown worse |
| tone | candidate | 20 | 18 of 60 | 57 of 60 | 65 | [24.1, 94.8] | 10 | shown better |

Each interval is read at 97.5%, corrected across each arm's aspects.

Each arm's gains and its drops are each shown wrongly at most 2.5% of the time.
```

**The readings.** A change reads each candidate on each aspect as **shown better** (its gain's interval lies wholly
above zero), **shown worse** (its drop's interval lies wholly past the aspect's margin), **no worse than the margin**
(neither, and its drop's interval at 95%, the floor's own, lies wholly within the margin: `within_margin` in the
JSON) or **unclear**, and an aspect declaring no margin is never shown worse or no worse than it. "Shown worse" is a
drop past the margin, which is worse than the control; "no worse than the margin" says the margin, since within it
the candidate may still be a little worse. A reading on an aspect the question does not ask about is
marked `a lead`: worth chasing, not yet shown. A choice reads each pair: one arm **ahead** where the whole interval
lies on one side of zero, else `no separation`; ahead is a reading, not a pick. A measurement reads each measure with
its interval, the pass rate as a count (`45 of 60`).

**One error rate per candidate.** A change tests each candidate's rows, one per aspect, as one family, with gains and
drops corrected apart (Holm), so each candidate's gains and its drops are each shown wrongly at most 2.5% of the time.
There is no correction across candidates: with several, one of them is shown better or worse by chance more often.
A choice's pairs have no direction, so every pair on every aspect is one family. Each row's interval is read at the
level its row was tested at, so a reading and its interval never disagree, and a campaign of one aspect reads at
95%, as a floor does, so a floor and the reading on the same margin cannot disagree
([why](measuring-soundly.md#one-family-per-candidate)). A row no worse than the margin is not corrected: it is the
floor's own reading of the pass rate at 95%, so a `RateDrop` floor on the pass rate at the margin's points and the
row read one interval and say one thing. Where a floor and a row still part, on the points they read against or on
a check's rate the floor also reads, a line under the table says which.

**To settle.** Each unclear row of a change says, in the `To show no worse than the margin` table, how many cases
would show the candidate no worse than the margin, if no difference and if as observed, with the cost. It is counted
at 95%, as the floor projection below is, so a floor on the same rate and points gives the same count.

**The call** is the last line: who decided, which arm, when and why (where runs joined after it, the line counts
them: `…, over 4 runs; 2 added since.`), then each broken floor and missed bar it overrides,
and each it leaves unnamed. An override the readings no longer support is kept and marked `(no longer broken)`. In a
choice, each pair where another arm is shown ahead of the decided one is a warning beside the call, never a refusal.
With no call: `No call is recorded yet: a person decides, over the readings above.`
([recording one](adopting-a-host.md#recording-the-call)).

**The folds.** Each aspect follows as a fold: one line of its reading, then that aspect's page in full, laid out as
the rest of this guide describes, its arms named by the campaign's labels. A campaign of one aspect has one fold.

### An aspect's first screen

In a fold, the campaign's first screen holds the title, the question and the warnings, so the fold opens on its
reading (`v2 unclear, a lead.`), then the arms line and the scoreboard. Laid out alone (`result.report`), an aspect's
page opens on its own first screen; the invoices example (rung 3) renders this, verbatim:

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

A campaign that asks a question opens on its name, then the question quoted on its own line,
then the arms line and the verdict. Where a candidate broke a floor, the warning comes first, above the title. A
declared triage campaign whose v2 broke its 5-point floor renders:

```markdown
**Floor broken on routing: v2's pass rate falls past the 5-point margin on its whole interval, a drop of 19.2 to 78.3 points.**

# triage v2
"Does v2 route tickets as well as v1?"

v1 (control) vs v2 · 18 Oct 2026 · 20 cases × 2 repeats

**Declared question: v2 worse than v1 on Accuracy; v2 worse than v1 on the pass rate.**
```

The first screen carries the minimum a reader needs to grasp the result; every caveat and nuance is one step down.

- **A broken floor**, where one is: the page's first line, bold, above the title (in HTML a
  `<p class="floor" role="alert">`), one per broken floor, naming the aspect, the arm, the margin and the interval.
  A floor warns and never decides: the verdict below it still says what the contrasts came to.
- **Title**: a name — the campaign's (`triage v2`), with what it compares where no question was asked
  (`invoices: does v2 do better than v1?`). A declared question is never cut down to a title.
- **The question**, where one is asked: quoted alone on the line under the title. It is the
  contract the verdict reads, so it is never paraphrased, and it is quoted whole: declare it short.
- **The arms line**: each arm named in full once (`deepseek-v4.1-flash (production, control)`), by the version its
  responses named where that is one model other than the one asked for (`deepseek-v4-flash-20261001`), then the
  date, then the design (`30 cases × 3 repeats`). After that an arm is its given label (`production`), the control
  marked `▸` in a table. Only the arms line and the verdict name a model.
- **Verdict**, written by code and never by a memo, result first: one sentence for the result, a second only for
  a guardrail. The first rule that fires: the test setup lost too much (`No verdict: the test setup failed, not the
  models.`, the arm, the cause, the fix; `No verdict: nothing was measured, because the test setup failed or the
  provider declined every call.` when every arm's run failed having measured nothing); one arm (`emails: 18/30 cases passed; 12 misses: double-charge,
  gdpr-delete, …`, each case by the name its author gave it, else `case N`); which arm passed more, when an arm is
  under the case floor; a declared question's contrasts (`Declared question: v2 worse than v1 on Accuracy; …`);
  `Exploratory:` (linked to the glossary) and what separated, or `nothing separated v2 from v1 on any reading`,
  never "no difference". A guardrail an arm is not shown safe on is the second sentence: `Not yet shown safe: v2 on
  no invented total.` No floor writes this line.
- **Scoreboard**: a bare table of counts and the cost per reply (per 1,000 replies when a reply costs under a cent).
  A trial passed when every check passed (a goal check, and on the quick path a `bool` grader or a field grader) and
  every judged dimension cleared its bar, and a case passed when every trial of it did, on the console as on the
  page. A failure cause beside checks and the judge heads its own column (`Model failed`) where it is the only one;
  the outcome bar under the table names each where there are several (in Markdown, its table only where it splits a
  cause the scoreboard lumps together). Under it, in normal weight, never bold, never first, the facts that change
  how the verdict is read and nothing else, in this order: the case floor; an arm whose every trial failed. A "too few" line says what is withheld and why, in the
  reader's terms, never a bare count — and what is withheld is only the interval, never the gap, which is measured
  and stated: `Usable cases: v4.1-flash 3, v4-flash 1. Under 5 independent cases a rate is shown as a count, and
  whether the arms' gap holds on more cases is not estimated.` Everything else is a step down: what was not charted
  and where a cost is a floor are in the Methods.

The case floor counts independent cases, never trials or repeats. Below it no interval is estimated: no contrast is
tested, the verdict says too few, a pass rate (and pass^k) reads as a count with no estimate of how far off it could
be, and no gap between the arms is sized. From the case floor up an interval is estimated, and its width says how
precisely the gap is known. A measure's mean is shown with its interval. Below the case floor a rate floor reads
neither broken nor held, since both need an interval; a cost floor can still read broken on a cost observed no
lower. So a host's seed gate covers every template whose results are compared across arms, whatever its declared
reading.

### The rest of the page

The sections follow in this order, each reached from a sticky nav; Cases, Trials, Not tested, Methods and the
Glossary start folded, and a nav link lands on the folded section's heading, which the reader opens. No link on the
page lands inside a closed fold the reader has not opened, since not every browser opens one for a link: a term lands
on the Glossary's heading (its definition is also the link's hover text), and a case's trial id on the Trials
heading.

- **Decision**: the aspect's floors, in the `Floors on <aspect>` table: one row per floor and reading, the floor's
  own sentence first (`no pass rate drops by more than 5 points`; a floor naming one reading says `no tone rate drops
  by more than 2 points`), each arm's column headed by its name, then that reading's own status with the reason in
  words about that reading alone. A floor is `broken` where its clause is shown crossed: a rate's drop past the
  margin on its whole interval, every arm at the case floor (`broken: v2 passed 20 of 40 to v1's 40, a drop that
  exceeds the margin of 5 points on its whole interval (at least 19.2)`), or a cost observed no lower. It is `held`
  where every rate's interval keeps the drop within the margin, or the cost interval is wholly below zero (`held:
  v2's drop is shown to stay within 10 points …`), and `unclear` otherwise (`unclear: the pass rate cannot separate
  the arms (both 16 of 16)`; a drop past the margin on its point estimate alone is unclear, and says so with its
  interval). One line under the table: `A floor warns and never decides: a broken one opens the page. A floor reads
  its own interval, uncorrected, which can differ from the contrasts table's.` A rate floor reads the bounded
  interval at the nominal 95%, wider than the t interval the Contrasts table prints: it holds its error rate on a
  few coarse cases, where the t interval does not
  ([why](measuring-soundly.md#a-floor-is-held-only-on-the-bounded-interval)). So a floor can read unclear beside a
  contrast that is not separated. A campaign that asks a question and declares no floor gets one line, `No floor
  was declared on routing.`; an exploratory comparison, which asks nothing and so declares no floor, gets none.
  **An unclear floor says what would show it holds**, computed, never the memo's guess: a `To show the floors hold`
  table gives each unclear floor's two counts with their cost, `If no difference` and `If as observed` (`121 cases,
  $3.1`; `none would` where the observed drop is at or past the margin; `re-run the lost trials` where the interval
  already clears and only lost trials keep it open; a cost floor's no-difference cell blank, since arms that do not
  differ are never shown cheaper). A rate's count is the bounded interval computed again over this run's cases
  repeated to that many, so it keeps their spread and the interval's small-sample excess; scaled as one over the
  square root of the cases, a dozen cases' bounded interval would overstate the need by up to about twice. It assumes
  new cases vary as these did, so the line under the table calls each count about. It counts cases: repeats of a case
  narrow an interval only by steadying each case's share, far less than new cases do. A count is always more than the
  run had, and where every count is, the line says so: `This run used 12 cases; showing the floors hold needs about
  121 to 223, more than it ran.` A floor broken or held projects nothing.
- **What it means**: each finding as one sentence its writer wrote, unnumbered, and a table code built from the
  readings it chose (a check's pass rate as a count, `11 of 12`; money as `$`); its confidence, trials and caveats
  sit in a closed `Evidence` fold, one row with a column per fact so each header links its definition. Findings are
  evidence: one that names no reading is not a finding, so it is not laid out here but noted in one line under Not
  tested (`The analysis also noted: …`). The memo's words are printed as written (below).
  A campaign's summary and recommendation are on its first screen, not in a fold; the writer's summary of the aspect
  opens the fold ([one memo per campaign](#having-a-model-write-the-analysis-over-frozen-evidence)).
  **Each finding traces to its evidence, in HTML.** Each figure in its table links to a cell printing the same
  figure of the same reading on the same arm: a contrast's arm or control, a guardrail's mean, a one-arm page's
  reading, an arm's cost per reply; a figure printed only in the closed `Bars` table links to that fold. A figure is
  never linked to a different number: a contrast that read fewer cases prints a paired mean, which is another figure.
  Each trial id in its `Evidence` fold links to the Trials heading; one the page does not hold is not linked. On a
  campaign's page each finding's `Evidence` fold lists what it cites with its figure, each linked to the heading of
  the aspect's fold that prints it. Code draws every link from the cells and trials the
  memo cites, never from its words; a finding none of whose figures or trials the page prints says
  `No contrast, guardrail, bar or cost on this page shows this finding's figures.` in its fold.
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
- **Methods**: a table (arms, judge, settings sent, test cost, the analysis's writer, the bundle); and the engine's own
  disclosures in a closed fold. The per-lever coverage map (status, findings that name it, next steps) is in the
  bundle's `coverage`, in the `bundle-<aspect>.json` written beside a saved page (the command line's `report --out`); the page lists each `thin` or `unswept` lever under Not tested.
  `threetears.evals.analysis.build_report` lays it out as a closed `Coverage` fold, one row per lever, for one analysis. Then the **Glossary**: every term of
  art the page links, in one plain sentence. Every column header that is a term of art links its entry, and in HTML a
  hover reads it.

### Style

`threetears.evals.analysis.style` holds every rendered sentence to one set of rules, `lint_prose`: no meta-prose
about the page ("rests on", "written by", "a count, not a ranking"), no hedge on a measured fact ("no measurable"),
no `key=value`, no chain of dashes, no sentence over 28 words or with more than three numbers (that is a table), no
header over three words, and the declared question only on its own quoted line. `humanize` is the one seam an
identifier passes on its way to a reader. Those rules hold the text code writes. The memo's words are printed as
written, whitespace aside: code checks code, and a model's prose is measured by a model, the reporter eval, calibrated
by people, in whatever language the memo is written. The writer's prompt states the same rules (no restated numbers,
no sample size or cost of its own, no dispute of a floor code settled, no discount of a reading for when its runs
ran), and the writer view carries the facts they rest on (`cases_to_settle`, each run's `k_runs`, `late_criteria`).
No floor rewrites a memo's recommendation: a floor only warns, and the page's first line is the warning.

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
- **Adjusted p** is Holm-corrected within the family: the aspect's, where the campaign asks it a question, whenever
  that was asked (the readings on the aspect's merit axes, and the pass rate where the aspect
  declares a margin), or else one exploratory family over every reading on a merit axis. A measure
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
| breached | the whole interval is beyond the margin | it warns: the writer is refused if it recommends the arm, whatever it gained; the person decides |
| undecided | the interval straddles the line, no interval exists, or the reading declares no range; the row says why | do not read it as safe; the contrasts open with a line naming the arm, and the verdict line says it. To decide it, add cases, declare a margin, or declare the reading's range |

The margin is the measure's `materiality_threshold`, and a judged dimension's is the one its campaign declares
(`AspectDesign.guardrail_margins`). A guardrail with no margin declared is held at zero change: `held` then
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

A campaign's question says what it set out to learn, and each aspect's merit axes say what it asks about there. A
reading on no merit axis of its aspect is **exploratory**: worth reporting as a lead for the next campaign, never as
a confirmed answer. Where a question was asked, the bundle lists those readings (`reading_scope`), the code-only
report names them under Not tested, and a finding resting only on them carries a `Scope` fact. Where none was,
every finding is exploratory: the verdict line says so, every finding carries the `Scope` fact, and the analysis
writer may adopt nothing. A campaign that declared no design at all is an exploratory campaign, and that sentence
says so: its readings confirm nothing, and the design its comparisons read was inferred from the runs, not declared
([when to declare one](measuring-soundly.md#start-from-the-question)). A guardrail is never exploratory.

**A criterion set after its readings is said, never demoted; data age is never said.** What pre-registration
guards against is a criterion chosen once the readings it judges could be seen: the aspect itself and its control
(where a question asks it), its margin, a bar, a judged guardrail's margin or a floor. Where no question is asked,
every reading is a lead and the control is kept, so when the aspect and its control were chosen decides nothing and is
not said; a margin, a bar, a guardrail's margin and a floor read or warn either way, so each is timed always. Under a
question a guardrail's margin carries the aspect's own time, which the aspect's line names. Each is read on its own time, where it is stored: the aspect's
`asked_at` (which also dates its bars, its guardrails' margins, and a margin or floor declared with it), `margin_declared_at`,
`Floor.declared_at` and the control's `named_at`, against the first stored result of the aspect's runs
(`first_result_at`). One set later reads exactly as it would had it been set first, because the operator is trusted
to have set it in good faith, and the line lets a reader judge. The bundle lists each (`reading_scope.late_criteria`:
`criterion`, `name`, `set_at`, null where no time is recorded), the page JSON carries them beside each aspect, and the
campaign's page, each aspect's page and the analysis listing say each once under the question, grouped across
aspects: "The margin on billing was set on 20 Oct 2026, after its readings were visible.", "Every aspect was declared
on …", "The control was named on 20 Oct 2026, after the readings on tone were visible.", or, for one stored by an
older build, "When the control was named is not recorded.". Each date is in its stamp's own offset, as every other
date on the page is. The writer names each once in its summary, with the page's dates. No line names the question or
when the runs began: a run whose measurement context is unchanged reads exactly like a fresh one, however old, and
arms measured minutes, hours or days apart are one experiment while nothing that decides their result changed.
`question_declared` is true wherever a question is asked, whenever. An aspect adopted from another campaign
(`aspects_from`) keeps its source's times, so a campaign adopting runs that are still valid reads as the one declared
before them; a margin, floor or arm added at adoption carries its own time, and is named where that is later than the
aspect's first result. Runs added to an aspect (`campaign_runs_add`) change no criterion's time, so runs whose
results were stored before the aspect's criteria were set name those criteria as set after their readings. A result
whose stored time cannot be ordered is left out of the comparison, and the page says whether a criterion came after it
is not known. One gap is known and not closed: an aspect adopted from a campaign that asked no question keeps its
source's time when the adopting campaign asks one, so its being put to the question after its readings were visible is
not named. A campaign's declaration is frozen once any aspect holds a run: `campaign_update` refuses
`question`, `aspects` and `floors` in any form, and refuses re-pointing or clearing the control (which may be named
once, from a run, while none is); the name and description stay editable, and an amended question is a new campaign
that adopts the aspects it keeps.

## Excluded trials by cause

A trial is left out of an arm's measures for one of three causes (`trial_exclusion`): `infra`, the harness faulted
it or the provider declined one of its calls (a 429 that is not a spent quota, or a 529 overloaded or 503
unavailable, which says nothing about the model: it needs no fix, only a rerun once the provider can serve it, and the
trial's error says which it was; but when an arm lost every trial this way and its run ended `failed` for it, the
page says nothing was measured and asks for the cause to be fixed before a rerun, not waited out; and when the
account stopped the run, the fix names the account);
`truncated`, a judge reply was cut off at the output cap it was sent and was not bought again; and
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

## A run a ceiling stopped

A run whose cost cap or metered-call ceiling was reached ends `budget_stopped`, and says so in one line written from
its stop records, never from prose:

> stopped by its metered-call ceiling: 40 of 40 at the stop, 47 once 3 running cells finished (+7); 12 cells not started

Read it left to right. *40 of 40 at the stop* is what the recorded cells had landed against the ceiling when the run
decided to start no new cell. *47 once 3 running cells finished (+7)* is where it ended: those cells were running at
the stop, were refused nothing, and finished whole, so the run lands past a ceiling by up to the width's worth of
cells and that overshoot is the design. *12 cells not started* is the part of the matrix never measured. Money reads
as `$` (`stopped by its cost cap: $10.20 of $10.00 at the stop, $12.35 once 1 running cell finished (+$2.35); ...`).
A run both ceilings stopped at one boundary reads both. The figures are `EvalRun.budget_stops`; a run stopped by its
wall-clock budget has none and gives its reason instead.

Every cell that ran is whole: a ceiling refuses no call inside a cell, so no cell's score was taken on a world it
could not fully reach because of the budget, and two identical runs never differ in which cell was cut. The one
refusal left is a host that declared no metered tools; a non-zero `metered_calls_refused` means that declaration is
wrong, not that the run ran out.

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
fingerprint it read on `generation.bundle_fingerprint`.

A fingerprint is `v2:` and the sha256 of the bundle's canonical JSON with every float rounded to twelve significant
figures. The bundle's intervals and multipliers come from C maths (`math.lgamma`, `NormalDist().inv_cdf`) whose last
bits differ between two builds of one Python, so the rounding is what makes one evidence fingerprint alike on a
laptop, in a container image and on another architecture, except, rarely, a value that falls exactly on a 12th-figure rounding boundary. The rounding is in the fingerprint only: the bundle keeps
its figures as computed. A fingerprint recorded before the rounding has no prefix, and every comparison computes the
fresh one in the recorded one's version (`fingerprint(like=recorded)`), so an analysis or a reporter case recorded then
still matches its evidence on the machine that recorded it. The insight-ledger cutoff (`bundle_assembled_at`),
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

**One memo per campaign: a summary, findings, one recommendation, a summary per aspect.** The writer authors no
headline, no chart and no count: the warnings and readings are code's. It reads the campaign bundle (each aspect's
readings, trials, their states, causes, failed checks and the judge's reasons) and writes one `summary`, its read
across every aspect in a sentence, required of every kind and laid out after the warnings and the question; findings,
each naming in `trials` the trials it is drawn from; at most one recommendation (`Recommendation`: an arm, reasons,
confidence, the findings it `rests_on`); one summary per aspect (what changed, how sure, what it means here), laid out
at the top of that aspect's fold; and next steps. The summary is prose like a finding's body: a number in it is a
figure reference code renders, an empty one is refused through the one repair round, and the page prints it as
written. Code holds the recommendation to the campaign's kind, refusing it through the one repair round:

| Kind | The writer must | The writer may not |
|---|---|---|
| measurement | write findings and summaries | recommend anything |
| choice | pick an arm, or say no clear winner | pick at more than `low` confidence when no pair separated its pick ahead; leave unanswered an aspect where another arm is shown ahead (`behind_on`) |
| change | recommend a switch or keeping the control | switch on exploratory evidence alone, to a candidate with no better reading or improved contrast on any asked aspect and not every floor held on them (each asked aspect declaring one), or to one that breached a guardrail |

Every kind answers its pick's warnings, each once and each with a why: a broken floor in `floors_overridden`, a
missed bar in `bars_missed`. A switch rests on at least one finding; keeping the control may rest on none. The page
sets each why beside the warning it answers, and a floor the memo overrides still opens the page.

**A finding's title never states a null its evidence did not establish, and names only arms its evidence reads.**
"Halving the budget does not change latency, cost or delivery" decides as surely as a measured result does, and over
a contrast that did not separate it asserts a result the campaign never measured; a title naming an arm none of its
finding's readings is taken at states a result the reader is shown nothing of. Both are the prompt's rule 9 and the
reporter eval's to measure. The generator once refused such titles by an English phrase list and by matching arm
names in the title; it reads no prose now (owner, 2026-10-24).

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

A declared question is printed as the words it was asked, both in the memo and in the contrasts table. In the memo,
each finding's evidence is
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
