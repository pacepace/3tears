# Reading the statistics

The pipeline computes every statistic. Your job is to read them correctly and say what they mean in one clause. Never recompute one the tools returned. If a figure you need isn't returned, call `capability_gap`.

## Intervals

- An interval is over **cases, not attempts**. Repeats of one case are not independent draws, and the tools' intervals already account for that. Five cases run three times each is five cases.
- Read the level on every interval you cite, and keep levels apart. A 95% interval on one confirmatory reading and a 99.69% interval on a corrected contrast answer different questions.
- **Separated** means the interval excludes zero (or the line). **Overlapping intervals don't prove two options alike.** Two options are alike only when a reading or contrast says *equivalent*. An *unclear* reading is unresolved, never "no change". An established null and a comparison too weak to resolve are different findings. Say which one you have.

## Margins and flip margins

A margin says how much worse a candidate may be and still count as acceptable (non-inferiority). For a candidate with change Δ and interval [lo, hi] against margin M:

- `hi < 0`: **worse**. Lead with the size, then whether the margin allows it.
- `lo > 0`: **better**.
- otherwise: **not separated**.
- It is within the margin when `-lo < M`. The **flip margin** is `-lo`: the margin at which the verdict changes. `decision_sensitivity` returns where the call flips, and the figure key `flip_margin.<candidate>` names it.

Why size first: "within the margin" is true of a candidate that is clearly worse by an amount the margin happens to allow. If that clause leads, a regression reads as a pass.

A margin larger than any plausible result could fail is uninformative. If the margin is more than about twice the worst observed drop, say so. When the margin was set after the results, check whether the call holds across every margin (`decision_sensitivity`, `parameter='margin:<aspect>'`). If it does, that caveat is checked, not open.

**An aspect left unsettled past its declared margin is an open caveat**, asked or not. If the interval reaches past the margin, a forbidden loss is still possible. Carry it in the confidence with its `material_if`.

## Gates and borderline results

A point estimate on the right side of a gate whose interval crosses the line is borderline. It passes for the call. Watch names it, with what would decide it: `settle_cases` sizes the run that would settle it and how often a run of today's size reads the other way. Its sizes against a line are one-sided; quote the two-sided size where the reader will compare it with a printed interval, and say whether it counts cases or trials. Decide which kind of "not settled" you have:

- **The interval crosses the line:** more cases can settle it. Name how many.
- **The interval is clear of the line on the wrong side:** more cases won't help. The candidate has to change.

Read each line one way. Say once whether a line is read on the point estimate or on the whole interval, and use that reading everywhere it appears: the call, Watch, the explore controls and Omitted.

Treat near options' noise even-handedly. If you size one option's thin clearance in standard errors, size the rival's thin miss the same way before leaning between them.

When you pick the one option out of many that clears a line, its estimate is optimistic. The best of eighteen configurations near a gate is partly the luckiest of eighteen. The thinner the clearance, the more this matters. Say so in Watch.

## Corrected versus uncorrected

Contrasts across many arms and measures are corrected for multiplicity, which widens their intervals relative to the same contrast uncorrected. Compare like with like: a reading against the control may be unpaired and so wider than a corrected paired contrast. Check the two intervals before you say which is wider. That is what lets you call one *improved* or *regressed* without being fooled by the number of comparisons. Breakdowns you request by stratum, task or cell (`measure_breakdown`) are uncorrected and exploratory. A separated stratum is a lead to follow up, not a finding. Phrase it that way once, then don't hedge every sentence.

Average effects over a crossed grid (what a retry buys, what more reasoning buys) are meaningful only when the factors don't interact. Check the grid before averaging. An interaction is a finding in itself, often the most important one.

## Unknown is not a fail

A gate read in each task, on a figure the snapshot holds only overall, has no figure in those places. The tools call such an option **undecided**, never failed or removed, and list its unknown places. Don't write that it fails the gate. Say what is unknown and what measuring it would settle. A labelled implied fact (for example, an overall p95 over the budget implies some task is over it) is an inference you may state as one, never a verdict.

Bars warn by default. Where the question makes a bar the condition, read it as a gate with `bars_as: "gates"` and say in Method that you did.

## Kept examples are a sample

The snapshot keeps a few failures per slice. `examples_list` says how many it kept of how many failed (`kept_from`). A sentence about all of a slice's failures needs that count, not the kept few; say how many you read out of how many there are.

## Small strata and zero counts

- A stratum with few cases (under about ten) produces deltas dominated by noise. Keep it in the full table and out of the leads, even when its delta is the largest.
- **Zero of n is not zero.** With no events in n cases, the 95% upper bound is roughly 3/n. Twenty clean cases still allow a 15% failure rate.
- A zero can also mean *not measured*. A cost of zero, or a rate with no trials behind it, is missing data, not a free or perfect arm. Check `n` before reading a zero.

## Depth, repeats and pass^k

Depth is how many times each arm ran each case. Arms can differ when old runs pool forward. **pass^k**, the chance that all k repeats of a case pass, measures reliability. It compares arms only at their common depth, the fewest repeats any arm gave a shared case. Above that, one arm is measured on more chances to fail. These figures describe the method. They go in Method or the appendix unless reliability is the question asked.

## Judges and calibration

A judged figure is only as good as the judge. Read its tier, calibration state, calibration ratings and agreement from the overview: each judged dimension's `judge` figures there (`judged.<dimension>.tier`, `.calibration`, `.calibration_ratings`, `.agreement`), which the `judge_uncalibrated` caveat in `caveats_list` cites too. A field the input leaves unstated is left out.

- An **uncalibrated** judge is a report-wide caveat. Its `material_if` names how a misreading would change the call ("only if it misreads in-character replies"). If the deciding finding is measured telemetry (cost, latency, a tool-call check), say the call doesn't rest on the judge.
- **Relative** comparisons on one judge are sturdier than **absolute** claims. "Lower than the control on the same cases" survives a biased judge better than "holds character 85% of the time". An absolute judged claim needs a calibrated judge.
- When labels come from humans (classifiers), human agreement is the ceiling. Errors concentrated where annotators also disagree are partly label noise.
- How many human ratings would turn a judge caveat into evidence: `settle_cases` sizes cases, not calibration labels, so log a `capability_gap` and say it in words.

## Derived figures

Some figures are arithmetic on point estimates: cost uplift against the control, a stratum's share of the loss, price per point. They carry no interval. Don't present them with an interval's authority, and don't test them. They describe; the readings and contrasts infer.
