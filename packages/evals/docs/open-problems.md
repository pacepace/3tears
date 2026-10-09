# Open problems

Read this when you plan work on the engine, or want to know whether a weakness in a report is known.
Each entry is a gap in today's code, or a limit chosen on purpose and marked as such: what is missing, why it
matters, and the fix where one is known. Each has a tracking issue; the change that closes it deletes its entry. The figures behind several entries are in [measuring soundly](measuring-soundly.md),
and the outside sources in [prior art](prior-art.md).

## Measurement

### A reading's interval ignores clustering

Tracked in [#590](https://github.com/pacepace/3tears/issues/590).

A single reading's interval is a t-interval over every observation, so `k` repeats of one case count as
`k` independent draws. The reading says `N obs over M cases, interval too narrow`
(`analysis/references.py`) instead of computing the right width. Comparisons between arms already use
per-case means. In simulation a raw-observation 95% interval covered the
truth 70% of the time, against 94% for one computed over case means. Fix: the standard error over case means
with `n_cases − 1` degrees of freedom, or Miller's cluster-robust form, and drop the disclosure.

*Evidence:* simulation, 5 cases × k=3, between-case σ 1.0 and repeat σ 0.3, 2026-09, coverage 70.3% vs 94.3%.

### pass^k is the all-pass indicator, stored under the opposite name

Tracked in [#591](https://github.com/pacepace/3tears/issues/591).

`compute_pass_k` (`contracts/scoring.py`) is the all-pass indicator per case, averaged: a case passes when
every scored repeat passed. At uniform depth that is unbiased; a run stopped early leaves mixed depths, which flatter the shallow cases,
and the engine discloses the depth range rather than correcting. Grouping is per run, so repeat runs
cannot add depth. The stored key is `pass_at_k`, which elsewhere
means "at least one of k" (Chen et al. 2021): the opposite quantity. Fix: the τ-bench estimator
C(c,k)/C(n,k) per case with n ≥ k, pooled across the runs of one cell, reported as the pass^1..k curve,
and the field renamed.

*Evidence:* simulation, 5 cases, depths [1,3,2,3,1], 2026-09, 0.651 against a true 0.531.

### Bars compare means, not intervals

Tracked in [#593](https://github.com/pacepace/3tears/issues/593).

`propose_bars` (`analysis/bar_proposals.py`) seeds a bar at the incumbent's mean, and `BarVerdict.cleared`
compares the cell's mean with the threshold, so an unchanged incumbent misses its own bar about half the
time. The ruled design: a bar decides by the interval against a declared margin, and seeds from the
incumbent's measured interval.

*Evidence:* simulation of an unchanged incumbent, n = 3, 6 and 15, 2026-09, missed its own bar 49–50% of the time.

### No power pre-flight

Tracked in [#594](https://github.com/pacepace/3tears/issues/594).

A launch is priced before it runs (see [cost and budgets](cost-and-budgets.md)), but nothing says what
effect the campaign can detect, and power depends on variance components nobody measures in advance. Fix: a pre-flight beside the price ("with N cases
and k repeats this campaign can detect Δ ≥ x"), using variance from earlier runs of the same template.

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

### Human labels and judge scores are not combined

Tracked in [#598](https://github.com/pacepace/3tears/issues/598).

Calibration ratings (`CalibrationRating`) decide a judge's tier, but the estimate itself uses judge
scores alone. Prediction-powered inference (Angelopoulos et al. 2023) combines a small human-labelled set
with many judge scores into an estimate whose interval stays valid when the judge is biased. Not built.

### Pairwise judging (declined for now)

Tracked in [#599](https://github.com/pacepace/3tears/issues/599).

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

Tracked in [#600](https://github.com/pacepace/3tears/issues/600).

The per-run metered-call ceiling (`run/metering.py`) refuses calls past a run's limit. It is in memory
and per run, so concurrent runs, out-of-run calls and the host's live traffic can exhaust one provider
quota together. Rejected when the ceiling shipped: a credit cap (provider-specific), folding credits into
the dollar cap (fails open with no rate card), a separate eval provider key. A fix needs a per-provider
quota the host declares, shared across runs.

## Testing the engine

### No test checks the statistics against known answers

Tracked in [#601](https://github.com/pacepace/3tears/issues/601).

The statistics tests (`tests/test_stats.py`, `tests/test_multiple_comparisons.py`) pin outputs on fixed
inputs. None simulates data with a known truth and checks coverage, false-positive rate or power; this
page's figures came from a simulation outside the package. Fix: a seeded simulation suite that checks
interval coverage and test error rates at the sample sizes the engine actually sees.

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

### The report-writer prompt budget is unenforced

Tracked in [#604](https://github.com/pacepace/3tears/issues/604).

`analysis/gen_prompt.py` declares `PROMPT_CHAR_BUDGET = 28_000` and `PROMPT_RULE_BUDGET = 15`, and says
the budget only falls. Nothing reads either constant, and rules, each added to fix one failure, accrete. Fix: a test that measures the seed prompt against
both constants.
