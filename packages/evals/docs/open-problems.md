# Open problems

**For** anyone planning work on the engine, or wondering whether a weakness in a report is known. **Answers:**
what is missing today and why it matters. Each entry is a gap in today's code, or a limit chosen on purpose and marked as such: what is missing, why it
matters, and the fix where one is known. Each has a tracking issue; the change that closes it deletes its entry,
except a limit decided on purpose, which stays, marked "Decided in" its issue with the decision, so the record
survives. The figures behind several entries are in [measuring soundly](measuring-soundly.md), and the outside
sources in [prior art](design.md#prior-art).

## Measurement

### Equivalence needs a declared range (accepted limit)

Decided in [#695](https://github.com/pacepace/3tears/issues/695): a measure that declares a margin and no
`value_range` is never tested for equivalence, so it never reads `equivalent`.

With no declared range no test of a mean holds its error rate at a few cases (an unbounded value can hide a rare
large move). The paired t-test the engine used there claimed `equivalent` 11-13% of the time against 5% on skewed
coarse values. Every engine measure with a margin declares its range, so this binds host measures. Such a
comparison still reads `improved`, `regressed` or `not_separated`; its equivalence is untested, with the reason
"declare value_range on this measure to test equivalence", and a report says so once per measure. Fix, on the
host's side: declare the measure's range.

### On a few cases a rate floor reads unclear (accepted limit)

Decided with the fix that made a rate clause read the bounded interval (`.30`; no issue filed yet), and kept when
decision rules became floors (the campaign rewrite, durable v16).

A rate floor reads `held` and `broken` only on the bounded interval, for the overall pass rate and each check's
alike, at the nominal 95%, and that interval holds its error rate at every n
([why](measuring-soundly.md#a-floor-is-held-only-on-the-bounded-interval)). On a handful of cases it is wide. Even
when both arms pass every trial, it allows a drop of 46 points at 7 cases and 13 at 30 cases: no valid interval shows
a rare large drop is absent from a few agreeing cases. So at 7 cases a 10-point floor reads `held` only on a sweep
(the candidate passes every trial the control fails), and `broken` only on a drop far past its margin; otherwise it
reads `unclear`, saying how far the interval reaches, and its `To show the floors hold` row says how many cases
would settle it. Discodon's G2 re-ask at 7 cases should expect `unclear` on its rate floor unless the candidate
collapses. A floor only warns, so an unclear one blocks nothing; it is the reader who is left without the warning's
answer. Fix, on the host's side: more cases, or a wider margin the decision can live with. The t interval would
settle sooner, but at the margin it read `held` up to 14% of the time against the 2.5% it states.

### A guardrail is held only on a declared range (accepted limit)

Decided with the fix that made `held` a bounded test's claim (no issue filed yet; tracked with
[#697](https://github.com/pacepace/3tears/issues/697)'s guardrail pillar): a guardrail on a reading that declares no
`value_range` is never read `held`.

`held` is a claim of safety. With no declared range no test of a mean holds its error rate at a few cases, and the
t interval the engine read it off claimed `held` up to 9.5% of the time against 2.5% at the margin on skewed coarse
values. Such a guardrail now reads `undecided`, with the reason "declare value_range on this measure (on an eval,
grader(..., range=) beside the guard)". Fix, on the host's side: declare the reading's range. A pass/fail scorer and a judged
scale declare theirs already.

**The breach on such a reading is still the t interval's, and its rate is not guaranteed.** Simulated at 30 cases
with no margin, on 1-5 differences that are -1 on most cases and +4 on one in five (an arm no worse than the
control that looks worse in most samples), it read `breached` 4.1% of the time against 2.5%. A false breach blocks
a good arm, which is the lesser harm, and refusing every breach without a range would let a blatant regression on an
unbounded measure be adopted, so the t breach is kept and the gap recorded as a strict xfail in
`tests/test_simulated_guardrails.py`. On a declared range the breach is the bounded test's, and holds its rate.

## Reporting

### Latency read under concurrency is withheld rather than reported with a flag

No issue filed yet. A result whose latency was read while other cells or runs executed beside it
(`execution_mode: concurrent`) is left out of every latency comparison, bar and ranking, and the page says so once
in the methods' disclosures; a campaign whose runs were all concurrent reports no latency at all, so a question
asking "at lower cost per turn, and no slower?" gets no reading on speed and nothing on the first screen says why.
Open: whether withholding is right, or whether a contended latency should be reported with a flag that says what it
was read under (the way an unpriced cost is a floor the page names), and what a reader should see instead of
silence, on the first screen, when the one reading a question asks about was withheld. Discodon's G2 (2026-10) hit
this: 19 of 24 results were concurrent, and the only trace was a memo finding citing no reading.

### A reader cannot yet question a report's analyst (designed, not built)

Designed in [#1002](https://github.com/pacepace/3tears/issues/1002). A report's analyst had the whole context it
wrote from, and today that context is only an audit record. Chat would keep it available: a reader asks a stored
report's analyst a question, and the analyst answers from its own stored session, resumed, with read-only tools on
the same snapshot. A revision asked for in chat takes a draft's path (the invariants, then the editor) and stores a
new version, never an edit; and what readers teach it reaches the skill only as an owner-approved proposal in the
gap loop, never applied automatically. It would ship as an optional extra built on the agent framework, so the core
install stays as it is, and the host supplies the chat UI: a page or panel beside the report, never inside the
report's sandboxed frame. Open: how a reader conversation maps onto the framework's sessions, the turn contract
hosts build against, and whether a reader sees a question's cost before asking.

### Reporter cases judge memos and planned reports, not the analyst's report

No issue filed yet. The analysis reporter evaluates a writer as a subject: a case is a frozen campaign, and a judge
reads what the writer read and wrote. Its cases are memos and planned reports. The analyst's report is held instead
by its editor's rubric before storage, and by acceptance campaigns re-run when the skill changes. Moving the
reporter's cases onto report documents is calibration work: the editor's thresholds (no 0, at least 11 of 14) are
starting points until stored reports scored against people's labels set them.

## Judging

### An operator's new judge case joins the live revision

Since `.30` the boot seed never adds a case to a judge template that already holds cases: the case waits for
`reset_definitions`, which writes a new revision, and runs of two revisions never pool. An operator's own
`judge_case_author` or `judge_cases_freeze` into such a template still adds to the revision in place, so runs of that
judge template before and after the write share a revision id while their cases differ. Fix: an operator's write into
a judge template that holds cases writes the next revision, as the reset does.

### A tier's bounds are conservative on a 1-5 scale

A tier is decided on score bounds for kappa
([evidence tiers](reading-reports.md#how-far-a-judged-score-can-be-leaned-on-evidence-tiers)). On pass/fail and
for separation at its floor, a judge at the bar earns the tier 2.5-4.5% of the time, close to the 5% allowed. On a
1-5 scale at the 20-result calibration floor it earns it 0.2-1.9% of the time. That spends power: a true-0.9
judge calibrates only 14-45% of the time at 20 results. An exact test under a stated
disagreement model would earn tiers on fewer results. It would be valid only for that model, though, and a judge
that occasionally reverses the scale breaks it. Not built. The floors are set where the power is reasonable:
separation needs 120 results.

*Evidence:* seeded simulation, 20 to 140 results, six marginals, 2026-10, `tests/test_simulated_agreement.py`.

### Pairwise judging (declined for now)

Decided in [#599](https://github.com/pacepace/3tears/issues/599): declined for now. Every judged score stays a
single-trial score.

All judging scores one trial alone. For subjective comparisons (tone, style), practitioners find pairwise
judging with position swap more reliable. It was deliberately not adopted, but "is arm B better than arm
A" is exactly the pairwise case.

## Host contract

### A shared third-party quota can still be exhausted (accepted limit)

Decided in [#600](https://github.com/pacepace/3tears/issues/600): an accepted limit. The ceiling stays in memory
and per run, and a cross-run, host-declared per-provider quota is not planned.

The per-run metered-call ceiling (`run/metering.py`) refuses calls past a run's limit. It is in memory
and per run, so concurrent runs, out-of-run calls and the host's live traffic can exhaust one provider
quota together. Rejected when the ceiling shipped: a credit cap (provider-specific), folding credits into
the dollar cap (fails open with no rate card), a separate eval provider key. A fix needs a per-provider
quota the host declares, shared across runs.

### A price bound on rates the host does not guarantee is an estimate (accepted limit)

Decided by the owner, 2026-10-28; no issue filed. A cap is checked against each call's bound, computed from the rates the host's pricer answers. Where the host
cannot guarantee them (a router that may serve the call from another provider, the host's own key on a router), the
bound is an estimate, and a call it admits may cost more. Refusing such calls would refuse every routed call, so the
call is admitted and the estimate disclosed: a report says its cap was checked against estimates.

### Only the report writer has a token and turn budget (accepted limit)

Decided by the owner, 2026-10-28; no issue filed. A host with no price for a model can write reports on it under a budget in tokens and turns, and the page says the
report is unpriced. A run launch keeps its dollar rule: an arm nobody can price is refused under an enforced cap. No
host has yet needed an unmetered candidate.

### An aggregate over a scope past its read budget is refused (accepted limit)

Decided by the owner, 2026-10-27, and set per reader from measurement on 2026-10-30. The aggregating reads (a pivot,
a frontier, a history, a comparison grouping) and a campaign's bundle work in Python over every result they read, so
past their measured budget (`threetears.evals.analysis.reads`: about 60,000 results for a history, a frontier or a
grouping; about 27,000 for a pivot or a bundle) they refuse, naming the filters that narrow the call. The spend
lenses hold none. Answering such a scope in one
call needs the aggregation pushed down into the store's SQL
([#996](https://github.com/pacepace/3tears/issues/996)). No run has a size limit: its reads are paged, though a few
readers still read a run whole (the CHANGELOG lists them).

## Worlds

### A world with state has no recorded lookups

Not built (no issue filed yet). A `World` with state refuses `recording=`, since its tools move the state it
is graded on, and one world cannot also hold stateless lookups whose answers are recorded, while `as_host` refuses
evals on two worlds, so no host can pair the two. Rung 4 (`examples/4-booking/`) therefore has no recorded tool
lookup beside its calendar: every tool it calls runs live in each cell.
