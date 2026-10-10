# Open problems

**For** anyone planning work on the engine, or wondering whether a weakness in a report is known. **Answers:**
what is missing today and why it matters. Each entry is a gap in today's code, or a limit chosen on purpose and marked as such: what is missing, why it
matters, and the fix where one is known. Each has a tracking issue; the change that closes it deletes its entry,
except a limit decided on purpose, which stays, marked "Decided in" its issue with the decision, so the record
survives. The figures behind several entries are in [measuring soundly](measuring-soundly.md), and the outside
sources in [prior art](prior-art.md).

## Measurement

### The between-arm correlation is not reported

Tracked in [#595](https://github.com/pacepace/3tears/issues/595).

Pairing on the frozen case helps only as far as the arms' per-case results correlate; Miller recommends
reporting the correlation so a reader sees what pairing bought. The engine pairs but reports none.

### Factors that move together are not grouped

Tracked in [#596](https://github.com/pacepace/3tears/issues/596).

The bundle lists each varying factor as a
[confound](design-rationale.md#confounds-qualify-never-suppress) but not which moved together. In one campaign four factors moved in lockstep across all 22 runs, and
no comparison could separate them. The design: hash each factor's partition of the runs. Factors with identical
partitions are aliased and are reported as one group ("these four move together across all 22 runs; no
comparison separates them"). State that aliasing with an interaction (C = A⊕B) is not checked.

*Evidence:* agent with tools, 22 runs, 1 campaign, 2026-07, single campaign.

### The frontier does not read guardrails

Tracked in [#613](https://github.com/pacepace/3tears/issues/613).

The frontier, which ranks contestants against an absolute bar with no control, leaves boundary dimensions
out of pass^k and the composite but does not disqualify a contestant on one, and says so
(`TwoPillarDisclosure`). Fix: a frontier rule for the boundary pillar. (A judged guardrail's margin is now
declared on its campaign, `CampaignDesign.guardrail_margins`, [#697](https://github.com/pacepace/3tears/issues/697).)

### Equivalence needs a declared range (accepted limit)

Decided in [#695](https://github.com/pacepace/3tears/issues/695): a measure that declares a margin and no
`value_range` is never tested for equivalence, so it never reads `equivalent`.

With no declared range no test of a mean holds its error rate at a few cases (an unbounded value can hide a rare
large move). The paired t-test the engine used there claimed `equivalent` 11-13% of the time against 5% on skewed
coarse values. Every engine measure with a margin declares its range, so this binds host measures. Such a
comparison still reads `improved`, `regressed` or `not_separated`; its equivalence is untested, with the reason
"declare value_range on this measure to test equivalence", and a report says so once per measure. Fix, on the
host's side: declare the measure's range.

## Judging

### No check for judge drift across configurations

Tracked in [#597](https://github.com/pacepace/3tears/issues/597).

A judge is a model and its config, and evidence tiers are keyed that way (see
[evidence tiers](reading-reports.md#how-far-a-judged-score-can-be-leaned-on-evidence-tiers)). A re-judge
or repeat refuses to run under an apparatus the run did not record, so nothing re-scores stored evidence
under a new config to show how far the judge moved. The ruled design re-scores a frozen transcript set
whenever a judge config changes: it detects movement, not which judge is right. Without it a before/after
spanning a judge change cannot be answered. One judged dimension rose from 2.7 to 4.4 when a subject fix
and a judge swap landed together, while unchanged dimensions moved −0.3 to +0.5; "did the fix work" was
ruled permanently unanswerable.

*Evidence:* agent with tools, one before/after pair, 2026-07, 2.7→4.4 on the changed judge vs −0.3 to +0.5 on unchanged ones, single campaign.

### The judge temperature policy is not measured

Tracked in [#633](https://github.com/pacepace/3tears/issues/633).

Every judge call now asks for temperature 0 unless a config says otherwise, and the temperature
sent is part of the judge's identity, so the split the issue found is gone. What was not done is the measurement
the issue asked the policy to rest on: borderline-case score variance across repeats at each setting, which
`repeat_judge_scores` can produce for a judge at 0 and one at the provider's default. Until it is run, how much
temperature moves a judge's scores is unknown; what is known is that the two are never pooled.

*Evidence:* one probe in a private host application, 2026-09: scores stable across attempts at the provider default (a refusal scored
5, a detailed answer 4); says nothing about borderline cases.

### Human labels and judge scores are not combined

Tracked in [#598](https://github.com/pacepace/3tears/issues/598).

Calibration ratings (`CalibrationRating`) decide a judge's tier, but the estimate itself uses judge
scores alone. Prediction-powered inference (Angelopoulos et al. 2023) combines a small human-labelled set
with many judge scores into an estimate whose interval stays valid when the judge is biased. Not built.

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

### A fold with one arm per knob level is untested

Tracked in [#605](https://github.com/pacepace/3tears/issues/605).

`ResolvesInto` folds a subject component into the knob that writes it when, across the cohort, each level of
the knob carries one level of the component. The component is part of the variant key, so repeats, `k` or
further launches of one arm can never refute the fold; only two or more distinct arms at one knob level can. A
typical sweep has one arm per level, so the test cannot fail, and the engine marks the fold `unverified_fold`
rather than calling it a checked non-confound. Fix: an optional host reader for the component with the knob's
contribution removed, which makes a two-arm fold testable.

### A shared third-party quota can still be exhausted (accepted limit)

Decided in [#600](https://github.com/pacepace/3tears/issues/600): an accepted limit. The ceiling stays in memory
and per run, and a cross-run, host-declared per-provider quota is not planned.

The per-run metered-call ceiling (`run/metering.py`) refuses calls past a run's limit. It is in memory
and per run, so concurrent runs, out-of-run calls and the host's live traffic can exhaust one provider
quota together. Rejected when the ceiling shipped: a credit cap (provider-specific), folding credits into
the dollar cap (fails open with no rate card), a separate eval provider key. A fix needs a per-provider
quota the host declares, shared across runs.

## Testing the engine

### No paid analysis-generation lane

Tracked in [#602](https://github.com/pacepace/3tears/issues/602).

The analysis generator is exercised only against stubs, and a prompt, schema or fixture defect looks the
same until a real model writes an analysis. Fix: an
opt-in lane that generates one analysis with a real model whenever the generator or its prompt changes.

### No memory-bound probe

Tracked in [#603](https://github.com/pacepace/3tears/issues/603).

A run's peak memory should be bounded by its matrix, not its total trace volume. Fixes in the runner and
scoring hold this today; no test does. Fix: a probe that runs a large
synthetic matrix and asserts peak memory. Reinstate the accumulation to prove it fails.
