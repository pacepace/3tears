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


### Equivalence needs a declared range (accepted limit)

Decided in [#695](https://github.com/pacepace/3tears/issues/695): a measure that declares a margin and no
`value_range` is never tested for equivalence, so it never reads `equivalent`.

With no declared range no test of a mean holds its error rate at a few cases (an unbounded value can hide a rare
large move). The paired t-test the engine used there claimed `equivalent` 11-13% of the time against 5% on skewed
coarse values. Every engine measure with a margin declares its range, so this binds host measures. Such a
comparison still reads `improved`, `regressed` or `not_separated`; its equivalence is untested, with the reason
"declare value_range on this measure to test equivalence", and a report says so once per measure. Fix, on the
host's side: declare the measure's range.

### A guardrail is held only on a declared range (accepted limit)

Decided with the fix that made `held` a bounded test's claim (no issue filed yet; tracked with
[#697](https://github.com/pacepace/3tears/issues/697)'s guardrail pillar): a guardrail on a reading that declares no
`value_range` is never read `held`.

`held` is a claim of safety. With no declared range no test of a mean holds its error rate at a few cases, and the
t interval the engine read it off claimed `held` up to 9.5% of the time against 2.5% at the margin on skewed coarse
values. Such a guardrail now reads `undecided`, with the reason "declare value_range on this measure (on compare(),
ranges= beside the scorer)". Fix, on the host's side: declare the reading's range. A pass/fail scorer and a judged
scale declare theirs already.

**The breach on such a reading is still the t interval's, and its rate is not guaranteed.** Simulated at 30 cases
with no margin, on 1-5 differences that are -1 on most cases and +4 on one in five (an arm no worse than the
control that looks worse in most samples), it read `breached` 4.1% of the time against 2.5%. A false breach blocks
a good arm, which is the lesser harm, and refusing every breach without a range would let a blatant regression on an
unbounded measure be adopted, so the t breach is kept and the gap recorded as a strict xfail in
`tests/test_simulated_guardrails.py`. On a declared range the breach is the bounded test's, and holds its rate.

## Judging

### The judge temperature policy is not measured

Tracked in [#633](https://github.com/pacepace/3tears/issues/633).

Every judge call now asks for temperature 0 unless a config says otherwise, and the temperature
sent is part of the judge's identity, so the split the issue found is gone. What was not done is the measurement
the issue asked the policy to rest on: borderline-case score variance across repeats at each setting.

The harness for it exists: `compare_judge_temperatures` (the `judge_temperature` action, and
`python -m threetears.evals judge-temperature RUN --max-cost-usd DOLLARS`) re-judges a run's borderline cases at 0
and at the provider's default and reports per-dimension variance and self-agreement side by side, with the case
count and the spend ([Step 10](judges-and-calibration.md#step-10-measure-what-temperature-does-to-the-judge)). It is
tested only against scripted judges. **Outstanding: the run itself, with real judges** on a host's real borderline
cases, which needs spend signed off. Until it is run, how much temperature moves a judge's scores is unknown; what
is known is that the two are never pooled. When it has been run, record the figures here and close the issue.

*Evidence:* one probe in a private host application, 2026-09: scores stable across attempts at the provider default (a refusal scored
5, a detailed answer 4); says nothing about borderline cases.

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
