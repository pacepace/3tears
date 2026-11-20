# Small-n reads: what the quick path's comparison costs a small test, and a fairer read

Report only: no product behaviour changed. Every number below comes from a seeded simulation run locally, as
pure-Python and numpy scratch. The scratch restates the package's bet over arrays of replicates. Before any table
is drawn, it is checked equal, replicate for replicate, to `bounded_separation_p(..., paired=True,
value_range=(0, 1), mixture=True)`: 2,400 replicates, to 1e-9 relative. Its interval is checked equal to
`bounded_difference_interval(..., mixture=True)` (to 2e-16), and its plug-in bet to
`bounded_mean_p(..., planned_n=n)` (exactly). Nothing ran on the pool, and nothing called a model.

The scratch is `scratch/` in the `bp-smalln` worktree, and it is not committed. Its seeds are fixed: the main
tables use 20261117 for the check, then 1 to 9 per part.

## 1. What the package does today, verified

| Claim | Verified at | Finding |
|---|---|---|
| The quick path reads a paired comparison by the hedged grid-mixture bet | `analysis/stats.py:2503` (`bounded_separation_p`), `:2382` (`_hedged_log_path`), `:2343` (`_mixture_log_path`), `:2332` (`_MIXTURE_STAKES = 10`), `:2103` (`_BET_CAP = 0.9`); chosen in `analysis/bundle/comparisons.py:547`, with `places` from `analysis/bundle/trials.py:429` (`bet_places`) | Confirmed. The capital is half the upward mixture plus half the downward one. Each mixture is the mean of 10 constant stakes, j/10 of the cap for j = 1..10, with equal weights. It is read over the longest complete prefix of the committed order. |
| Holm across the measures a comparison tests | `comparisons.py:900` (`holm_adjust(raw, max_true=family_size)`), `:907` (intervals at 1 − α/m) | Confirmed, with a correction (next row). |
| The on-ramp's reading holds about α/2 | `comparisons.py:1158-1159` adds the trial pass-rate reading to every family whose question asks it | **Confirmed, and worse than described.** With one boolean grader, the "trial passed" reading is the grader itself: the same per-case values and the same hypothesis. Probed through `Eval.run` with 8 cases, all wins: both rows read `p_raw 0.0410`, `p_adjusted 0.0820`. One hypothesis is counted twice, so every single-check eval with a question pays the factor 2 for nothing. This is not a check tested alongside a different pass rate. |
| The lead sentence when not separated | `analysis/style.py:611` (`CANNOT_TELL_APART`), `:618-635` (`cannot_tell_apart`) | Confirmed. The interval sentence is `analysis/report/verdict.py:579`. |
| The anytime price is paid even on a single first look | as above | Confirmed. It is paid even when the runs are kept in memory, where no continuation is possible (`quick/continuation.py:313`, `UNKEPT_SENTENCE`): the probe's in-memory run read the mixture p. The default store is a file (`quick/contract.py:585`), so a default run can be continued and the anytime guarantee buys something there. |
| The coordinator's numbers | T1; column A of T2-onramp | The all-wins p's (0.33, 0.12, 0.041, 0.013, 0.0042 at n = 4 to 12) reproduce to every digit, and so does the sign test's 0.0078. The on-ramp harness's separated-of-200 at +0.30 (1, 46, 124, 174 at 16, 24, 32, 40 apps) is 0.005, 0.23, 0.62 and 0.87. Column A of T2-onramp reads 0.01, 0.26, 0.65 and 0.87, which matches. The mean-0.3 sequence (0.53 at n = 8, and so on) depends on the coordinator's sample, so it is not re-derived; my n = 8 sample at +0.29 reads 0.62. |

Today's first lines, through the package (`Eval.run`, 8 cases, one boolean grader named `passed`, question asked):

- All wins: `Passed score: the evidence cannot tell v2 and v1 apart. Of 8 trials each, Passed score was met on 0 for v1 and 8 for v2. Over 8 paired cases, the interval allows v2 to be as much as 7 points worse or 100 better.`
- About +0.3 (k = 3): `... cannot tell v2 and v1 apart. Of 24 trials each, Passed score was met on 8 for v1 and 15 for v2. Over 8 paired cases, the interval allows v2 to be as much as 42 points worse or 70 better.`
- No effect (k = 3): `... cannot tell v2 and v1 apart. ... met on 8 for v1 and 9 for v2. ... as much as 53 points worse or 57 better.`

## 2. The diagnosis, tested

The coordinator's diagnosis is right about the stacking, and each guarantee's price is measurable. It needs
four corrections, and two of its proposed remedies fail.

1. **The Holm penalty on the on-ramp is a duplicate, not a family** (section 1). It is the cheapest fix, with no
   trade-off: test a reading once.
2. **The hedge and the duplicate each cost exactly a factor 2 in capital.** One-sided at α/2 (B) and two-sided at
   α (C) give the same decisions, to every digit, in T1 and T2. Together they are a factor 4, and they move the
   smallest all-wins n that decides from 10 to 8.
3. **The uniform grid is not "most weight on small fractions"; it is equal weight.** Half of that weight sits on
   stakes at or below half the cap, and over 8 cases those can grow the capital at most 1.45⁸ ≈ 20-fold.
   Weighting by j (D) helps everywhere at small n, and costs nothing measured up to n = 100 (T7). One example:
   one-sided at 0.05, binary, +0.3, n = 16, power goes from 0.19 to 0.28. At large n with a tiny effect, where only
   the smallest stake grows, it can cost up to a factor 5.5 in capital: that stake's weight falls from 1/10 to 1/55.
4. **Removing every stacked guarantee does not make n = 8 decisive.** That limit is information, not method. At
   n = 8 on pass/fail with +0.3, the best valid one-look tests decide 11-12% of the time (the exact sign test, and
   the package's own fixed-n bounded bet, T8), and F1 decides 4%. On the on-ramp's graded shares, a test valid for
   the mean on every distribution decides +0.3 at 8 apps 4% of the time even at one fixed look. Only tests that
   assume symmetry do better, and that assumption fails (point 5). So the larger part of the owner's fix is the
   coordinator's three-part read, not the threshold: the estimate, its interval and graded evidence lead, and a
   decision is reserved for when the evidence clears the bar.
5. **The exact first look (E) is not valid for the mean on graded values.** On pass/fail data (k = 1) the sign test
   is exact for the mean, because a difference in {-1, 0, 1} has mean 0 exactly when it is symmetric. On graded
   per-case values, take a null whose mean is 0 but which is not symmetric: the arm gains 1/3 on three cases in
   four and loses the full range on the fourth. There the sign-flip test separates toward the arm 17.7% of the
   time at n = 6 and 9.9% at n = 8 (T2-skewed), and 10.2% under continuation (F2, T3). This is the package's own
   #597 finding again. On pass/fail data, where E is valid, it adds nothing over F1 under continuation: at +0.3,
   F2 decides 0.755 and F1 0.808. **E is rejected.**
6. **"On these apps" (stratified by app over trials) is far more powerful, but holds α only when trials are
   independent given the app and arm.** At 8 apps, k = 3, +0.3 it decides 0.60, where F1 with apps as cases decides
   0.00 (T6). But the on-ramp's own model includes arm-by-app noise ("two runs of one design on one app that
   disagree", `onramp/reading.py` `simulated_rows`), and there the stratified test reads that noise as a
   difference: its false separation is 0.068 [0.063, 0.073] at k = 7, spread 0.1, 24 apps, and it grows with k. Its
   claim is "on these apps, in this run", not "with this model". It is offered below only as a scoped description.

## 3. Options measured

The base is the on-ramp's reading today (A): two-sided, hedged, uniform grid, Holm over the duplicated reading,
so α/2 in effect. Each option changes one thing:

| | Change | Valid for the mean on any distribution on the range, at every look? |
|---|---|---|
| A | today | yes (Ville) |
| B | one-sided when the direction is declared (the unhedged upward bet) | yes |
| C | the declared primary at full α (no Holm; for the on-ramp, the duplicate dropped) | yes |
| D | a stake grid weighted toward larger fractions (weights ∝ j) | yes: any fixed mixture of stakes is a test supermartingale |
| E | an exact sign-flip test at the first look at α/4, and the bet at α/4 at every look (union bound) | **pass/fail only**: the sign-flip tests symmetry |
| F1 | B + C + D | yes |
| F2 | B + C + E (first look α/2, bet α/2) | pass/fail only |
| F3 | B + C + D + E | pass/fail only |
| F4 | F1 with half its capital on the package's plug-in bet tuned to the first look's n (T9) | yes: the plug-in is predictable |

Data:
- **binary**: pass/fail once a case. The control passes at U(0.1, 0.5), and the arm at that plus the effect.
- **onramp**: the on-ramp model from `onramp/reading.py`. Apps are cases, with k = 3 trials an arm, a base share of
  U(0.1, 0.6) and an arm-by-app spread of 0.1. The effect is the nominal shift in share.
- **skewed**: the asymmetric mean-zero null above.

Power cells use 4,000 replicates: the Clopper-Pearson 95% half-width is at most ±0.016, stated here rather than per
cell. Null cells use 20,000 replicates, each with its Clopper-Pearson bounds. A two-sided method's power counts
only separations toward the arm; its null rate counts either direction.

**Continuation (T3).** The first look is at 8 cases, with up to 5 looks. After a look that does not decide, add 4
cases plus up to 16 more, sized from the bet's own p (the nearer it is to 0, the more are added). Stop at the
first decision. The naive row re-runs a fixed-n exact test at every look, to show what anytime validity
prevents.

## 4. Headline numbers

| | Today (A) | F1 (one-sided, declared primary at full α, j-weighted grid) |
|---|---|---|
| Smallest n where all wins decides (T1) | 10 | 6 (the exact sign test, one look: 5) |
| All wins at n = 8: p (T1) | 0.041, read against α/2: **not decided** | 0.0135: **decided** |
| On-ramp model, +0.3, one look, 16 / 24 / 32 apps (T2) | 0.01 / 0.26 / 0.65 | 0.36 / 0.77 / 0.93 |
| Pass/fail, +0.3, one look, n = 8 / 16 / 24 (T2) | 0.00 / 0.06 / 0.17 | 0.04 / 0.27 / 0.44 |
| False separation, one look, worst cell to n = 40 (T2) | 0.005 [0.004, 0.006] | 0.028 [0.026, 0.030] |
| False separation, continued to 5 looks: binary / on-ramp / skewed (T3) | 0.008 / 0.001 / 0.005 | 0.029 [0.027, 0.031] / 0.017 [0.015, 0.019] / 0.033 [0.031, 0.036] |
| The naive alternative (exact test at every look), same continuation (T3) | | 0.061 [0.054, 0.069] / 0.091 [0.083, 0.101] / 0.186 [0.174, 0.198]: over α |
| Continued from 8 cases, +0.3: decided by look 5, mean cases at stop (T3) | binary 0.59, 42 cases; on-ramp 0.95, 38 cases | binary 0.81, 38 cases; on-ramp 0.99, 29 cases |
| Graded evidence, on-ramp +0.3 at 8 apps: e ≥ 3 (T4) | 12% | 60% |
| Null: the e-value ever reached 10 within 100 cases (T4; Ville bound 0.10) | | binary 0.064 [0.061, 0.068], on-ramp 0.059 [0.056, 0.062] |
| Interval width at n = 8, pass/fail; coverage (T5) | 129 points (97.5%); 1.000 | 103 points (90% two-sided, each end 95%); 1.000 |

Every F1 false-separation rate has a Clopper-Pearson upper bound below α = 0.05, single look and continued, on every
null simulated, the skewed one included.

## 5. The tables

T0: the restated bet equals `bounded_separation_p(..., mixture=True)` on 2,400 replicates, to 1e-9 relative.

### T1. Every case a win (1 vs 0): the p each method reads, and whether it decides

| method | level it decides at | n=4 | n=5 | n=6 | n=7 | n=8 | n=10 | n=12 |
|---|---|---|---|---|---|---|---|---|
| A today (on-ramp: hedged, uniform, Holm 2) | bet 0.025 | 0.3292 | 0.2020 | 0.1208 | 0.0709 | 0.0410 | **0.0133** | **0.0042** |
| B one-sided | bet 0.025 | 0.1695 | 0.1025 | 0.0608 | 0.0356 | **0.0205** | **0.0066** | **0.0021** |
| C primary at full alpha | bet 0.05 | 0.3292 | 0.2020 | 0.1208 | 0.0709 | **0.0410** | **0.0133** | **0.0042** |
| D linear-weighted grid | bet 0.025 | 0.2540 | 0.1477 | 0.0847 | 0.0481 | 0.0271 | **0.0084** | **0.0026** |
| E exact first look + bet | bet 0.0125, exact 0.0125 at look 1 | 0.3292 / exact 0.1250 | 0.2020 / exact 0.0625 | 0.1208 / exact 0.0312 | 0.0709 / exact 0.0156 | **0.0410 / exact 0.0078** | **0.0133 / exact 0.0020** | **0.0042 / exact 0.0005** |
| F1 = B+C+D | bet 0.05 | 0.1282 | 0.0741 | **0.0424** | **0.0241** | **0.0135** | **0.0042** | **0.0013** |
| F2 = B+C+E | bet 0.025, exact 0.025 at look 1 | 0.1695 / exact 0.0625 | 0.1025 / exact 0.0312 | **0.0608 / exact 0.0156** | **0.0356 / exact 0.0078** | **0.0205 / exact 0.0039** | **0.0066 / exact 0.0010** | **0.0021 / exact 0.0002** |
| F3 = B+C+D+E | bet 0.025, exact 0.025 at look 1 | 0.1282 / exact 0.0625 | 0.0741 / exact 0.0312 | **0.0424 / exact 0.0156** | **0.0241 / exact 0.0078** | **0.0135 / exact 0.0039** | **0.0042 / exact 0.0010** | **0.0013 / exact 0.0002** |
| ref: exact sign test, one-/two-sided | fixed n, one look | 0.0625 / 0.1250 | 0.0312 / 0.0625 | 0.0156 / 0.0312 | 0.0078 / 0.0156 | 0.0039 / 0.0078 | 0.0010 / 0.0020 | 0.0002 / 0.0005 |

Bold: decided. Each bet p is the package's construction (T0 checks it against `bounded_separation_p`).

### T2-binary. One look at n paired cases: share decided toward the arm (power), and at effect 0 the share decided either way (false separation)

Rate [Clopper-Pearson 95% bounds]. Pass/fail once a case; control U(0.1, 0.5), arm + effect.

**effect +0.0** (20000 replicates a cell)

| n | A | B | C | D | E | F1 | F2 | F3 | ref: exact one-sided at 0.05, one fixed look |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| 6 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.001 [0.001, 0.002] |
| 8 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.001 [0.000, 0.001] | 0.001 [0.000, 0.001] | 0.001 [0.000, 0.001] | 0.005 [0.004, 0.006] |
| 10 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.003 [0.002, 0.004] | 0.003 [0.002, 0.003] | 0.003 [0.002, 0.003] | 0.009 [0.008, 0.010] |
| 12 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.001] | 0.001 [0.000, 0.001] | 0.000 [0.000, 0.000] | 0.001 [0.000, 0.001] | 0.006 [0.005, 0.007] | 0.004 [0.004, 0.006] | 0.005 [0.004, 0.006] | 0.012 [0.010, 0.013] |
| 16 | 0.000 [0.000, 0.001] | 0.001 [0.001, 0.002] | 0.002 [0.002, 0.003] | 0.000 [0.000, 0.001] | 0.002 [0.001, 0.002] | 0.012 [0.010, 0.013] | 0.008 [0.007, 0.009] | 0.009 [0.008, 0.010] | 0.018 [0.016, 0.020] |
| 20 | 0.002 [0.001, 0.002] | 0.002 [0.002, 0.003] | 0.004 [0.004, 0.005] | 0.002 [0.001, 0.002] | 0.003 [0.002, 0.004] | 0.016 [0.014, 0.018] | 0.009 [0.008, 0.011] | 0.012 [0.010, 0.013] | 0.021 [0.019, 0.023] |
| 24 | 0.002 [0.002, 0.003] | 0.004 [0.003, 0.004] | 0.007 [0.006, 0.008] | 0.002 [0.002, 0.003] | 0.004 [0.003, 0.005] | 0.020 [0.018, 0.022] | 0.012 [0.010, 0.013] | 0.015 [0.013, 0.016] | 0.024 [0.022, 0.026] |
| 30 | 0.004 [0.003, 0.005] | 0.005 [0.004, 0.006] | 0.009 [0.008, 0.011] | 0.004 [0.003, 0.005] | 0.004 [0.003, 0.005] | 0.024 [0.022, 0.026] | 0.014 [0.012, 0.016] | 0.018 [0.016, 0.019] | 0.028 [0.025, 0.030] |
| 32 | 0.004 [0.003, 0.005] | 0.005 [0.004, 0.006] | 0.010 [0.009, 0.011] | 0.004 [0.003, 0.005] | 0.005 [0.004, 0.006] | 0.025 [0.023, 0.027] | 0.015 [0.013, 0.016] | 0.018 [0.016, 0.020] | 0.028 [0.026, 0.031] |
| 40 | 0.005 [0.004, 0.006] | 0.006 [0.005, 0.008] | 0.013 [0.011, 0.015] | 0.006 [0.005, 0.007] | 0.005 [0.004, 0.007] | 0.028 [0.026, 0.030] | 0.017 [0.015, 0.018] | 0.021 [0.019, 0.023] | 0.030 [0.028, 0.033] |

**effect +0.1** (4000 replicates a cell)

| n | A | B | C | D | E | F1 | F2 | F3 | ref: exact one-sided at 0.05, one fixed look |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 6 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 8 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 |
| 10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 | 0.01 | 0.03 |
| 12 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.02 | 0.02 | 0.04 |
| 16 | 0.00 | 0.01 | 0.01 | 0.00 | 0.01 | 0.04 | 0.03 | 0.04 | 0.06 |
| 20 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.06 | 0.04 | 0.05 | 0.08 |
| 24 | 0.01 | 0.02 | 0.02 | 0.01 | 0.01 | 0.07 | 0.06 | 0.07 | 0.10 |
| 30 | 0.02 | 0.03 | 0.03 | 0.02 | 0.02 | 0.10 | 0.08 | 0.09 | 0.14 |
| 32 | 0.02 | 0.04 | 0.04 | 0.02 | 0.02 | 0.10 | 0.09 | 0.10 | 0.15 |
| 40 | 0.03 | 0.05 | 0.05 | 0.03 | 0.04 | 0.12 | 0.12 | 0.13 | 0.19 |

**effect +0.2** (4000 replicates a cell)

| n | A | B | C | D | E | F1 | F2 | F3 | ref: exact one-sided at 0.05, one fixed look |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 6 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 |
| 8 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 | 0.01 | 0.04 |
| 10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.04 | 0.03 | 0.04 | 0.08 |
| 12 | 0.00 | 0.01 | 0.01 | 0.00 | 0.01 | 0.07 | 0.06 | 0.06 | 0.11 |
| 16 | 0.01 | 0.03 | 0.03 | 0.01 | 0.03 | 0.12 | 0.11 | 0.11 | 0.18 |
| 20 | 0.03 | 0.06 | 0.06 | 0.03 | 0.05 | 0.16 | 0.15 | 0.16 | 0.23 |
| 24 | 0.05 | 0.08 | 0.08 | 0.05 | 0.07 | 0.21 | 0.19 | 0.21 | 0.30 |
| 30 | 0.08 | 0.13 | 0.13 | 0.09 | 0.10 | 0.26 | 0.27 | 0.29 | 0.38 |
| 32 | 0.09 | 0.15 | 0.15 | 0.10 | 0.11 | 0.28 | 0.29 | 0.31 | 0.42 |
| 40 | 0.13 | 0.20 | 0.20 | 0.15 | 0.18 | 0.34 | 0.39 | 0.40 | 0.51 |

**effect +0.3** (4000 replicates a cell)

| n | A | B | C | D | E | F1 | F2 | F3 | ref: exact one-sided at 0.05, one fixed look |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 6 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.04 |
| 8 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.04 | 0.04 | 0.04 | 0.11 |
| 10 | 0.00 | 0.01 | 0.01 | 0.00 | 0.01 | 0.10 | 0.09 | 0.09 | 0.18 |
| 12 | 0.01 | 0.03 | 0.03 | 0.01 | 0.03 | 0.16 | 0.15 | 0.15 | 0.24 |
| 16 | 0.06 | 0.11 | 0.11 | 0.06 | 0.10 | 0.27 | 0.26 | 0.28 | 0.36 |
| 20 | 0.12 | 0.18 | 0.18 | 0.12 | 0.17 | 0.36 | 0.36 | 0.38 | 0.47 |
| 24 | 0.17 | 0.25 | 0.25 | 0.17 | 0.22 | 0.44 | 0.45 | 0.46 | 0.58 |
| 30 | 0.25 | 0.36 | 0.36 | 0.27 | 0.32 | 0.54 | 0.58 | 0.59 | 0.70 |
| 32 | 0.28 | 0.39 | 0.39 | 0.30 | 0.36 | 0.56 | 0.62 | 0.63 | 0.74 |
| 40 | 0.40 | 0.50 | 0.50 | 0.42 | 0.52 | 0.67 | 0.73 | 0.74 | 0.83 |

**effect +0.5** (4000 replicates a cell)

| n | A | B | C | D | E | F1 | F2 | F3 | ref: exact one-sided at 0.05, one fixed look |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 6 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.03 | 0.03 | 0.03 | 0.16 |
| 8 | 0.00 | 0.01 | 0.01 | 0.00 | 0.01 | 0.19 | 0.19 | 0.19 | 0.37 |
| 10 | 0.03 | 0.08 | 0.08 | 0.03 | 0.08 | 0.39 | 0.38 | 0.38 | 0.54 |
| 12 | 0.11 | 0.23 | 0.23 | 0.11 | 0.23 | 0.53 | 0.53 | 0.54 | 0.67 |
| 16 | 0.36 | 0.49 | 0.49 | 0.36 | 0.50 | 0.72 | 0.75 | 0.76 | 0.84 |
| 20 | 0.55 | 0.67 | 0.67 | 0.56 | 0.69 | 0.85 | 0.88 | 0.88 | 0.93 |
| 24 | 0.71 | 0.82 | 0.82 | 0.73 | 0.81 | 0.91 | 0.93 | 0.94 | 0.97 |
| 30 | 0.85 | 0.91 | 0.91 | 0.87 | 0.91 | 0.96 | 0.98 | 0.98 | 0.99 |
| 32 | 0.88 | 0.93 | 0.93 | 0.90 | 0.94 | 0.97 | 0.99 | 0.99 | 0.99 |
| 40 | 0.96 | 0.98 | 0.98 | 0.96 | 0.98 | 0.99 | 1.00 | 1.00 | 1.00 |

### T2-onramp. One look at n paired cases: share decided toward the arm (power), and at effect 0 the share decided either way (false separation)

Rate [Clopper-Pearson 95% bounds]. The on-ramp model: apps as cases, k = 3 trials an arm, base U(0.1, 0.6), arm-by-app spread 0.1.

**effect +0.0** (20000 replicates a cell)

| n | A | B | C | D | E | F1 | F2 | F3 | ref: exact one-sided at 0.05, one fixed look |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| 6 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.001 [0.000, 0.001] | 0.001 [0.000, 0.001] | 0.009 [0.008, 0.011] |
| 8 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.001] | 0.000 [0.000, 0.000] | 0.006 [0.005, 0.007] | 0.006 [0.005, 0.007] | 0.018 [0.016, 0.020] |
| 10 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.001 [0.001, 0.002] | 0.000 [0.000, 0.000] | 0.009 [0.007, 0.010] | 0.009 [0.007, 0.010] | 0.023 [0.021, 0.026] |
| 12 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.004 [0.003, 0.005] | 0.000 [0.000, 0.000] | 0.010 [0.009, 0.011] | 0.010 [0.009, 0.011] | 0.028 [0.025, 0.030] |
| 16 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.005 [0.004, 0.007] | 0.001 [0.000, 0.001] | 0.013 [0.011, 0.015] | 0.013 [0.011, 0.015] | 0.032 [0.029, 0.034] |
| 20 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.005 [0.004, 0.006] | 0.001 [0.001, 0.002] | 0.014 [0.013, 0.016] | 0.014 [0.013, 0.016] | 0.036 [0.033, 0.038] |
| 24 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.006 [0.005, 0.007] | 0.002 [0.002, 0.003] | 0.015 [0.014, 0.017] | 0.015 [0.014, 0.017] | 0.036 [0.033, 0.039] |
| 30 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.001] | 0.001 [0.001, 0.001] | 0.000 [0.000, 0.001] | 0.007 [0.006, 0.008] | 0.005 [0.004, 0.006] | 0.018 [0.016, 0.020] | 0.018 [0.016, 0.020] | 0.037 [0.035, 0.040] |
| 32 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.001] | 0.001 [0.001, 0.002] | 0.000 [0.000, 0.001] | 0.007 [0.006, 0.009] | 0.006 [0.005, 0.007] | 0.017 [0.016, 0.019] | 0.017 [0.016, 0.019] | 0.038 [0.035, 0.040] |
| 40 | 0.000 [0.000, 0.001] | 0.001 [0.001, 0.002] | 0.002 [0.002, 0.003] | 0.001 [0.001, 0.001] | 0.008 [0.007, 0.010] | 0.010 [0.009, 0.011] | 0.018 [0.017, 0.020] | 0.018 [0.017, 0.020] | 0.038 [0.036, 0.041] |

**effect +0.1** (4000 replicates a cell)

| n | A | B | C | D | E | F1 | F2 | F3 | ref: exact one-sided at 0.05, one fixed look |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 6 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 | 0.04 |
| 8 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.03 | 0.03 | 0.08 |
| 10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 | 0.05 | 0.05 | 0.11 |
| 12 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.00 | 0.07 | 0.07 | 0.14 |
| 16 | 0.00 | 0.00 | 0.00 | 0.00 | 0.03 | 0.01 | 0.10 | 0.10 | 0.18 |
| 20 | 0.00 | 0.00 | 0.00 | 0.00 | 0.04 | 0.03 | 0.13 | 0.13 | 0.22 |
| 24 | 0.00 | 0.01 | 0.01 | 0.00 | 0.05 | 0.05 | 0.17 | 0.17 | 0.27 |
| 30 | 0.01 | 0.02 | 0.02 | 0.01 | 0.08 | 0.11 | 0.21 | 0.21 | 0.33 |
| 32 | 0.01 | 0.02 | 0.02 | 0.02 | 0.09 | 0.13 | 0.23 | 0.23 | 0.35 |
| 40 | 0.03 | 0.07 | 0.07 | 0.04 | 0.13 | 0.20 | 0.29 | 0.30 | 0.42 |

**effect +0.2** (4000 replicates a cell)

| n | A | B | C | D | E | F1 | F2 | F3 | ref: exact one-sided at 0.05, one fixed look |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 6 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 | 0.02 | 0.10 |
| 8 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 | 0.11 | 0.11 | 0.23 |
| 10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.04 | 0.00 | 0.18 | 0.18 | 0.31 |
| 12 | 0.00 | 0.00 | 0.00 | 0.00 | 0.08 | 0.01 | 0.25 | 0.25 | 0.39 |
| 16 | 0.00 | 0.00 | 0.00 | 0.00 | 0.17 | 0.09 | 0.37 | 0.37 | 0.51 |
| 20 | 0.01 | 0.04 | 0.04 | 0.02 | 0.25 | 0.21 | 0.47 | 0.47 | 0.61 |
| 24 | 0.03 | 0.10 | 0.10 | 0.07 | 0.33 | 0.35 | 0.56 | 0.56 | 0.70 |
| 30 | 0.12 | 0.25 | 0.25 | 0.19 | 0.46 | 0.51 | 0.68 | 0.68 | 0.80 |
| 32 | 0.16 | 0.30 | 0.30 | 0.24 | 0.50 | 0.56 | 0.72 | 0.72 | 0.83 |
| 40 | 0.33 | 0.49 | 0.49 | 0.42 | 0.65 | 0.73 | 0.83 | 0.84 | 0.91 |

**effect +0.3** (4000 replicates a cell)

| n | A | B | C | D | E | F1 | F2 | F3 | ref: exact one-sided at 0.05, one fixed look |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 6 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.07 | 0.07 | 0.23 |
| 8 | 0.00 | 0.00 | 0.00 | 0.00 | 0.03 | 0.00 | 0.26 | 0.26 | 0.44 |
| 10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.14 | 0.03 | 0.42 | 0.42 | 0.60 |
| 12 | 0.00 | 0.00 | 0.00 | 0.00 | 0.25 | 0.10 | 0.54 | 0.54 | 0.70 |
| 16 | 0.01 | 0.06 | 0.06 | 0.03 | 0.47 | 0.36 | 0.72 | 0.72 | 0.84 |
| 20 | 0.09 | 0.24 | 0.24 | 0.17 | 0.64 | 0.60 | 0.85 | 0.85 | 0.92 |
| 24 | 0.26 | 0.47 | 0.47 | 0.38 | 0.77 | 0.77 | 0.91 | 0.91 | 0.96 |
| 30 | 0.56 | 0.73 | 0.74 | 0.67 | 0.89 | 0.91 | 0.97 | 0.97 | 0.99 |
| 32 | 0.65 | 0.79 | 0.79 | 0.74 | 0.92 | 0.93 | 0.97 | 0.97 | 0.99 |
| 40 | 0.87 | 0.94 | 0.94 | 0.91 | 0.97 | 0.98 | 0.99 | 0.99 | 1.00 |

**effect +0.5** (4000 replicates a cell)

| n | A | B | C | D | E | F1 | F2 | F3 | ref: exact one-sided at 0.05, one fixed look |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 6 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.28 | 0.28 | 0.64 |
| 8 | 0.00 | 0.00 | 0.00 | 0.00 | 0.18 | 0.05 | 0.74 | 0.74 | 0.89 |
| 10 | 0.00 | 0.01 | 0.01 | 0.00 | 0.61 | 0.34 | 0.90 | 0.90 | 0.96 |
| 12 | 0.01 | 0.12 | 0.12 | 0.05 | 0.82 | 0.66 | 0.96 | 0.96 | 0.99 |
| 16 | 0.35 | 0.67 | 0.67 | 0.54 | 0.97 | 0.95 | 1.00 | 1.00 | 1.00 |
| 20 | 0.81 | 0.94 | 0.94 | 0.90 | 0.99 | 0.99 | 1.00 | 1.00 | 1.00 |
| 24 | 0.96 | 0.99 | 0.99 | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 30 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 32 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 40 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

### T2-skewed. A null on the mean that is not symmetric (graded, k = 3: +1/3 three cases in four, -1 the fourth; mean difference 0): false separation toward the arm, one look

| n | A | B | C | D | E | F1 | F2 | F3 | ref: exact sign-flip one-sided at 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.177 [0.172, 0.183] | 0.177 [0.172, 0.183] | 0.177 [0.172, 0.183] |
| 8 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.099 [0.095, 0.103] | 0.000 [0.000, 0.000] | 0.099 [0.095, 0.103] | 0.099 [0.095, 0.103] | 0.099 [0.095, 0.103] |
| 12 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.033 [0.030, 0.035] | 0.000 [0.000, 0.000] | 0.033 [0.030, 0.035] | 0.033 [0.030, 0.035] | 0.033 [0.030, 0.035] |
| 16 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.011 [0.009, 0.012] | 0.014 [0.013, 0.016] | 0.065 [0.061, 0.068] | 0.065 [0.061, 0.068] | 0.065 [0.061, 0.068] |
| 24 | 0.002 [0.001, 0.002] | 0.004 [0.003, 0.004] | 0.004 [0.003, 0.004] | 0.003 [0.002, 0.004] | 0.009 [0.008, 0.010] | 0.021 [0.019, 0.023] | 0.039 [0.036, 0.041] | 0.040 [0.037, 0.042] | 0.039 [0.036, 0.041] |
| 40 | 0.004 [0.003, 0.005] | 0.007 [0.006, 0.009] | 0.007 [0.006, 0.009] | 0.005 [0.004, 0.006] | 0.017 [0.016, 0.019] | 0.026 [0.024, 0.028] | 0.044 [0.041, 0.047] | 0.045 [0.042, 0.048] | 0.043 [0.040, 0.046] |

### T3. Continuation: first look at 8 cases, up to 5 looks, each add 4 + up to 16 cases sized from the bet's p, stopping at the first decision

**binary, effect +0.0 (null: decided either way)**, 20000 replicates

| method | decided by look 5 [CP 95%] | mean cases at stop | mean looks |
|---|---|---|---|
| A today (on-ramp: hedged, uniform, Holm 2) | 0.008 [0.007, 0.009] | 37.0 | 4.99 |
| B one-sided | 0.009 [0.008, 0.011] | 41.2 | 4.98 |
| C primary at full alpha | 0.016 [0.015, 0.018] | 36.7 | 4.97 |
| D linear-weighted grid | 0.009 [0.008, 0.011] | 39.3 | 4.99 |
| E exact first look + bet | 0.003 [0.002, 0.004] | 37.2 | 5.00 |
| F1 = B+C+D | 0.029 [0.027, 0.031] | 41.8 | 4.93 |
| F2 = B+C+E | 0.010 [0.008, 0.011] | 41.2 | 4.98 |
| F3 = B+C+D+E | 0.016 [0.014, 0.018] | 42.4 | 4.96 |
| naive: exact one-sided at 0.05 every look | 0.061 [0.054, 0.069] | 44.1 | 4.89 |

**binary, effect +0.2**, 3000 replicates

| method | decided by look 5 [CP 95%] | mean cases at stop | mean looks |
|---|---|---|---|
| A today (on-ramp: hedged, uniform, Holm 2) | 0.219 [0.204, 0.234] | 42.8 | 4.71 |
| B one-sided | 0.363 [0.346, 0.381] | 50.4 | 4.46 |
| C primary at full alpha | 0.276 [0.260, 0.292] | 40.5 | 4.58 |
| D linear-weighted grid | 0.234 [0.219, 0.250] | 44.4 | 4.67 |
| E exact first look + bet | 0.161 [0.148, 0.175] | 44.8 | 4.81 |
| F1 = B+C+D | 0.465 [0.447, 0.483] | 44.8 | 4.11 |
| F2 = B+C+E | 0.365 [0.347, 0.382] | 50.0 | 4.44 |
| F3 = B+C+D+E | 0.387 [0.370, 0.405] | 48.7 | 4.31 |

**binary, effect +0.3**, 3000 replicates

| method | decided by look 5 [CP 95%] | mean cases at stop | mean looks |
|---|---|---|---|
| A today (on-ramp: hedged, uniform, Holm 2) | 0.586 [0.568, 0.604] | 42.2 | 4.16 |
| B one-sided | 0.755 [0.740, 0.771] | 45.0 | 3.72 |
| C primary at full alpha | 0.628 [0.610, 0.645] | 39.0 | 3.99 |
| D linear-weighted grid | 0.606 [0.589, 0.624] | 42.7 | 4.08 |
| E exact first look + bet | 0.520 [0.502, 0.538] | 46.2 | 4.37 |
| F1 = B+C+D | 0.808 [0.793, 0.822] | 37.8 | 3.30 |
| F2 = B+C+E | 0.755 [0.740, 0.771] | 44.3 | 3.68 |
| F3 = B+C+D+E | 0.764 [0.748, 0.779] | 42.2 | 3.53 |

**onramp, effect +0.0 (null: decided either way)**, 20000 replicates

| method | decided by look 5 [CP 95%] | mean cases at stop | mean looks |
|---|---|---|---|
| A today (on-ramp: hedged, uniform, Holm 2) | 0.001 [0.001, 0.002] | 33.1 | 5.00 |
| B one-sided | 0.003 [0.003, 0.004] | 39.6 | 5.00 |
| C primary at full alpha | 0.004 [0.004, 0.005] | 33.0 | 5.00 |
| D linear-weighted grid | 0.003 [0.002, 0.004] | 35.5 | 5.00 |
| E exact first look + bet | 0.001 [0.000, 0.001] | 33.1 | 5.00 |
| F1 = B+C+D | 0.017 [0.015, 0.019] | 41.2 | 4.98 |
| F2 = B+C+E | 0.008 [0.006, 0.009] | 39.3 | 4.98 |
| F3 = B+C+D+E | 0.011 [0.009, 0.012] | 41.2 | 4.98 |
| naive: exact one-sided at 0.05 every look | 0.091 [0.083, 0.101] | 44.9 | 4.79 |

**onramp, effect +0.2**, 3000 replicates

| method | decided by look 5 [CP 95%] | mean cases at stop | mean looks |
|---|---|---|---|
| A today (on-ramp: hedged, uniform, Holm 2) | 0.595 [0.577, 0.613] | 44.9 | 4.43 |
| B one-sided | 0.822 [0.807, 0.835] | 47.7 | 3.86 |
| C primary at full alpha | 0.655 [0.638, 0.672] | 41.3 | 4.23 |
| D linear-weighted grid | 0.664 [0.646, 0.681] | 44.4 | 4.22 |
| E exact first look + bet | 0.529 [0.511, 0.547] | 47.5 | 4.56 |
| F1 = B+C+D | 0.895 [0.884, 0.906] | 39.5 | 3.32 |
| F2 = B+C+E | 0.824 [0.810, 0.837] | 44.4 | 3.67 |
| F3 = B+C+D+E | 0.860 [0.847, 0.872] | 42.4 | 3.46 |

**onramp, effect +0.3**, 3000 replicates

| method | decided by look 5 [CP 95%] | mean cases at stop | mean looks |
|---|---|---|---|
| A today (on-ramp: hedged, uniform, Holm 2) | 0.948 [0.939, 0.955] | 38.2 | 3.38 |
| B one-sided | 0.990 [0.986, 0.993] | 34.7 | 2.78 |
| C primary at full alpha | 0.957 [0.949, 0.964] | 34.0 | 3.17 |
| D linear-weighted grid | 0.961 [0.953, 0.968] | 36.1 | 3.15 |
| E exact first look + bet | 0.935 [0.926, 0.944] | 41.4 | 3.55 |
| F1 = B+C+D | 0.994 [0.991, 0.996] | 28.9 | 2.40 |
| F2 = B+C+E | 0.990 [0.986, 0.993] | 29.1 | 2.44 |
| F3 = B+C+D+E | 0.993 [0.989, 0.995] | 27.0 | 2.28 |

**skewed, effect +0.0 (null: decided toward the arm)**, 20000 replicates

| method | decided by look 5 [CP 95%] | mean cases at stop | mean looks |
|---|---|---|---|
| A today (on-ramp: hedged, uniform, Holm 2) | 0.005 [0.004, 0.006] | 38.8 | 4.99 |
| B one-sided | 0.011 [0.010, 0.013] | 44.9 | 4.98 |
| C primary at full alpha | 0.010 [0.009, 0.011] | 38.4 | 4.97 |
| D linear-weighted grid | 0.008 [0.007, 0.009] | 41.5 | 4.98 |
| E exact first look + bet | 0.098 [0.094, 0.103] | 33.4 | 4.61 |
| F1 = B+C+D | 0.033 [0.031, 0.036] | 45.9 | 4.93 |
| F2 = B+C+E | 0.102 [0.098, 0.106] | 38.8 | 4.60 |
| F3 = B+C+D+E | 0.103 [0.099, 0.108] | 40.5 | 4.60 |
| naive: exact one-sided at 0.05 every look | 0.186 [0.174, 0.198] | 43.1 | 4.45 |

### T4. Graded evidence: the e-value at the look (current capital, not its running maximum), strong >= 10, moderate 3 to 10, weak < 3

**binary**: F1's one-sided linear-grid e-value (today's hedged uniform e-value in brackets), share strong / moderate / weak

| effect | n=4 | n=6 | n=8 | n=12 | n=16 | n=24 | n=30 |
|---|---|---|---|---|---|---|---|
| +0.0 | 0.00/0.02/0.98 [0.00/0.00] | 0.00/0.04/0.96 [0.00/0.02] | 0.00/0.05/0.94 [0.00/0.04] | 0.01/0.05/0.93 [0.00/0.05] | 0.01/0.05/0.94 [0.00/0.04] | 0.01/0.03/0.96 [0.01/0.04] | 0.01/0.02/0.97 [0.01/0.04] |
| +0.1 | 0.00/0.05/0.95 [0.00/0.01] | 0.00/0.10/0.90 [0.00/0.03] | 0.02/0.12/0.87 [0.00/0.06] | 0.03/0.12/0.85 [0.00/0.07] | 0.04/0.11/0.85 [0.01/0.07] | 0.06/0.09/0.86 [0.02/0.07] | 0.07/0.08/0.85 [0.03/0.07] |
| +0.2 | 0.00/0.10/0.90 [0.00/0.02] | 0.02/0.17/0.81 [0.00/0.07] | 0.05/0.20/0.75 [0.00/0.12] | 0.10/0.20/0.70 [0.03/0.13] | 0.12/0.19/0.69 [0.05/0.14] | 0.21/0.15/0.64 [0.09/0.18] | 0.25/0.14/0.61 [0.13/0.18] |
| +0.3 | 0.00/0.17/0.83 [0.00/0.03] | 0.03/0.27/0.69 [0.00/0.14] | 0.11/0.28/0.61 [0.01/0.22] | 0.21/0.28/0.51 [0.07/0.23] | 0.28/0.25/0.47 [0.14/0.24] | 0.45/0.18/0.38 [0.27/0.26] | 0.54/0.15/0.31 [0.38/0.24] |
| +0.5 | 0.00/0.36/0.64 [0.00/0.09] | 0.14/0.46/0.40 [0.00/0.37] | 0.36/0.36/0.28 [0.06/0.48] | 0.60/0.26/0.14 [0.35/0.36] | 0.75/0.16/0.09 [0.55/0.29] | 0.93/0.04/0.03 [0.83/0.12] | 0.97/0.02/0.01 [0.93/0.05] |

Under the null, F1's e-value EVER reached 10 within 100 cases on 0.064 [0.061, 0.068] of runs (Ville: at most 0.10), and ever reached 3 on 0.216 [0.210, 0.221] (at most 0.33).

**onramp**: F1's one-sided linear-grid e-value (today's hedged uniform e-value in brackets), share strong / moderate / weak

| effect | n=4 | n=6 | n=8 | n=12 | n=16 | n=24 | n=30 |
|---|---|---|---|---|---|---|---|
| +0.0 | 0.00/0.00/1.00 [0.00/0.00] | 0.00/0.02/0.98 [0.00/0.00] | 0.00/0.03/0.97 [0.00/0.00] | 0.00/0.04/0.96 [0.00/0.01] | 0.00/0.05/0.95 [0.00/0.01] | 0.01/0.06/0.94 [0.00/0.03] | 0.01/0.05/0.94 [0.00/0.03] |
| +0.1 | 0.00/0.02/0.98 [0.00/0.00] | 0.00/0.07/0.93 [0.00/0.00] | 0.00/0.12/0.88 [0.00/0.00] | 0.01/0.19/0.80 [0.00/0.03] | 0.04/0.23/0.73 [0.00/0.06] | 0.11/0.25/0.64 [0.02/0.13] | 0.15/0.26/0.59 [0.05/0.15] |
| +0.2 | 0.00/0.06/0.94 [0.00/0.00] | 0.00/0.20/0.80 [0.00/0.00] | 0.01/0.31/0.68 [0.00/0.03] | 0.10/0.40/0.50 [0.00/0.16] | 0.24/0.40/0.37 [0.04/0.27] | 0.48/0.32/0.20 [0.22/0.35] | 0.63/0.24/0.13 [0.38/0.32] |
| +0.3 | 0.00/0.14/0.86 [0.00/0.00] | 0.00/0.41/0.59 [0.00/0.02] | 0.05/0.55/0.41 [0.00/0.12] | 0.34/0.48/0.18 [0.04/0.40] | 0.59/0.32/0.09 [0.22/0.48] | 0.88/0.10/0.02 [0.67/0.24] | 0.95/0.05/0.01 [0.84/0.13] |
| +0.5 | 0.00/0.44/0.56 [0.00/0.00] | 0.05/0.78/0.17 [0.00/0.13] | 0.36/0.58/0.05 [0.01/0.54] | 0.89/0.10/0.01 [0.43/0.51] | 0.99/0.01/0.00 [0.88/0.11] | 1.00/0.00/0.00 [1.00/0.00] | 1.00/0.00/0.00 [1.00/0.00] |

Under the null, F1's e-value EVER reached 10 within 100 cases on 0.059 [0.056, 0.062] of runs (Ville: at most 0.10), and ever reached 3 on 0.249 [0.243, 0.255] (at most 0.33).

### T5. The interval: coverage of the true mean difference and mean width, one look

I1 today's print on the on-ramp (hedged uniform CS, 97.5%); I2 the same at 95%; I3 hedged linear-grid CS at 90% (each side 5%, the partner of a one-sided 5% primary); I4 the package's fixed-n bounded interval at 95% (valid at one planned look only); I5 the t interval at 95% (clipped to the range); I6 percentile bootstrap 95%. Coverage [CP 95%] / mean width in points; t and bootstrap give no interval with no spread, counted as a miss of width 0 there.

| data | effect | n | I1 | I2 | I3 | I4 | I5 | I6 |
|---|---|---|---|---|---|---|---|---|
| binary | +0.0 | 8 | 1.000 [0.996] / 129 | 1.000 [0.996] / 120 | 1.000 [0.996] / 103 | 1.000 [0.996] / 115 | 0.942 [0.926] / 102 | 0.895 [0.874] / 77 |
| binary | +0.0 | 16 | 0.999 [0.994] / 94 | 0.998 [0.993] / 86 | 0.984 [0.974] / 73 | 0.986 [0.977] / 80 | 0.952 [0.937] / 67 | 0.944 [0.928] / 59 |
| binary | +0.0 | 24 | 0.999 [0.994] / 76 | 0.996 [0.990] / 69 | 0.965 [0.952] / 59 | 0.969 [0.956] / 64 | 0.944 [0.928] / 52 | 0.944 [0.928] / 48 |
| binary | +0.3 | 8 | 1.000 [0.996] / 131 | 1.000 [0.996] / 121 | 0.991 [0.983] / 105 | 0.991 [0.983] / 119 | 0.948 [0.932] / 104 | 0.921 [0.903] / 81 |
| binary | +0.3 | 16 | 0.998 [0.993] / 96 | 0.995 [0.988] / 87 | 0.971 [0.959] / 75 | 0.985 [0.975] / 82 | 0.949 [0.933] / 69 | 0.923 [0.905] / 61 |
| binary | +0.3 | 24 | 0.995 [0.988] / 78 | 0.989 [0.980] / 71 | 0.956 [0.941] / 61 | 0.975 [0.963] / 66 | 0.948 [0.932] / 55 | 0.941 [0.925] / 50 |
| onramp | +0.0 | 8 | 1.000 [0.996] / 113 | 1.000 [0.996] / 104 | 1.000 [0.996] / 86 | 1.000 [0.996] / 92 | 0.934 [0.917] / 62 | 0.864 [0.841] / 47 |
| onramp | +0.0 | 16 | 1.000 [0.996] / 74 | 1.000 [0.996] / 67 | 1.000 [0.996] / 54 | 1.000 [0.996] / 57 | 0.938 [0.921] / 40 | 0.905 [0.885] / 36 |
| onramp | +0.0 | 24 | 1.000 [0.996] / 57 | 1.000 [0.996] / 51 | 0.997 [0.991] / 41 | 0.997 [0.991] / 44 | 0.954 [0.939] / 33 | 0.940 [0.923] / 30 |
| onramp | +0.3 | 8 | 1.000 [0.996] / 115 | 1.000 [0.996] / 105 | 0.999 [0.994] / 87 | 0.999 [0.994] / 98 | 0.958 [0.944] / 64 | 0.887 [0.866] / 49 |
| onramp | +0.3 | 16 | 1.000 [0.996] / 75 | 0.999 [0.994] / 68 | 0.994 [0.987] / 55 | 0.996 [0.990] / 59 | 0.944 [0.928] / 41 | 0.909 [0.889] / 36 |
| onramp | +0.3 | 24 | 1.000 [0.996] / 58 | 0.996 [0.990] / 52 | 0.985 [0.975] / 42 | 0.990 [0.982] / 45 | 0.953 [0.938] / 32 | 0.933 [0.916] / 30 |

### T6. On these apps (stratified exact test over trials) versus beyond them (F1, apps as cases), one look, one-sided at 0.05

Share decided toward the arm. At effect 0 with spread 0.1 the arms' shares on each app still differ by the arm-by-app draw, so the stratified test's 'difference on these apps' is the noise of one run: its rate there is what it calls a difference when the arms do not differ.

| k | spread | effect | apps | stratified (these apps) | F1 (beyond: apps as cases) |
|---|---|---|---|---|---|
| 3 | 0.0 | +0.0 | 4 | 0.017 [0.015, 0.020] | 0.000 [0.000, 0.000] |
| 3 | 0.0 | +0.0 | 8 | 0.025 [0.022, 0.029] | 0.000 [0.000, 0.000] |
| 3 | 0.0 | +0.0 | 16 | 0.032 [0.029, 0.036] | 0.000 [0.000, 0.001] |
| 3 | 0.0 | +0.0 | 24 | 0.036 [0.033, 0.040] | 0.002 [0.001, 0.003] |
| 3 | 0.0 | +0.1 | 4 | 0.06 | 0.00 |
| 3 | 0.0 | +0.1 | 8 | 0.11 | 0.00 |
| 3 | 0.0 | +0.1 | 16 | 0.22 | 0.01 |
| 3 | 0.0 | +0.1 | 24 | 0.29 | 0.05 |
| 3 | 0.0 | +0.2 | 4 | 0.16 | 0.00 |
| 3 | 0.0 | +0.2 | 8 | 0.32 | 0.00 |
| 3 | 0.0 | +0.2 | 16 | 0.58 | 0.09 |
| 3 | 0.0 | +0.2 | 24 | 0.75 | 0.34 |
| 3 | 0.0 | +0.3 | 4 | 0.33 | 0.00 |
| 3 | 0.0 | +0.3 | 8 | 0.60 | 0.00 |
| 3 | 0.0 | +0.3 | 16 | 0.90 | 0.39 |
| 3 | 0.0 | +0.3 | 24 | 0.98 | 0.82 |
| 3 | 0.1 | +0.0 | 4 | 0.024 [0.021, 0.027] | 0.000 [0.000, 0.000] |
| 3 | 0.1 | +0.0 | 8 | 0.029 [0.026, 0.032] | 0.000 [0.000, 0.000] |
| 3 | 0.1 | +0.0 | 16 | 0.041 [0.037, 0.045] | 0.001 [0.000, 0.002] |
| 3 | 0.1 | +0.0 | 24 | 0.042 [0.038, 0.046] | 0.002 [0.001, 0.003] |
| 3 | 0.1 | +0.1 | 4 | 0.08 | 0.00 |
| 3 | 0.1 | +0.1 | 8 | 0.13 | 0.00 |
| 3 | 0.1 | +0.1 | 16 | 0.22 | 0.01 |
| 3 | 0.1 | +0.1 | 24 | 0.31 | 0.07 |
| 3 | 0.1 | +0.2 | 4 | 0.16 | 0.00 |
| 3 | 0.1 | +0.2 | 8 | 0.33 | 0.00 |
| 3 | 0.1 | +0.2 | 16 | 0.59 | 0.10 |
| 3 | 0.1 | +0.2 | 24 | 0.77 | 0.35 |
| 3 | 0.1 | +0.3 | 4 | 0.34 | 0.00 |
| 3 | 0.1 | +0.3 | 8 | 0.60 | 0.00 |
| 3 | 0.1 | +0.3 | 16 | 0.89 | 0.38 |
| 3 | 0.1 | +0.3 | 24 | 0.97 | 0.78 |
| 7 | 0.0 | +0.0 | 4 | 0.031 [0.028, 0.034] | 0.000 [0.000, 0.000] |
| 7 | 0.0 | +0.0 | 8 | 0.035 [0.031, 0.039] | 0.000 [0.000, 0.000] |
| 7 | 0.0 | +0.0 | 16 | 0.036 [0.033, 0.040] | 0.000 [0.000, 0.000] |
| 7 | 0.0 | +0.0 | 24 | 0.043 [0.039, 0.047] | 0.000 [0.000, 0.000] |
| 7 | 0.0 | +0.1 | 4 | 0.14 | 0.00 |
| 7 | 0.0 | +0.1 | 8 | 0.24 | 0.00 |
| 7 | 0.0 | +0.1 | 16 | 0.42 | 0.00 |
| 7 | 0.0 | +0.1 | 24 | 0.58 | 0.02 |
| 7 | 0.0 | +0.2 | 4 | 0.38 | 0.00 |
| 7 | 0.0 | +0.2 | 8 | 0.65 | 0.00 |
| 7 | 0.0 | +0.2 | 16 | 0.92 | 0.04 |
| 7 | 0.0 | +0.2 | 24 | 0.98 | 0.44 |
| 7 | 0.0 | +0.3 | 4 | 0.70 | 0.00 |
| 7 | 0.0 | +0.3 | 8 | 0.94 | 0.00 |
| 7 | 0.0 | +0.3 | 16 | 1.00 | 0.43 |
| 7 | 0.0 | +0.3 | 24 | 1.00 | 0.97 |
| 7 | 0.1 | +0.0 | 4 | 0.049 [0.045, 0.054] | 0.000 [0.000, 0.000] |
| 7 | 0.1 | +0.0 | 8 | 0.051 [0.047, 0.056] | 0.000 [0.000, 0.000] |
| 7 | 0.1 | +0.0 | 16 | 0.058 [0.054, 0.063] | 0.000 [0.000, 0.000] |
| 7 | 0.1 | +0.0 | 24 | 0.068 [0.063, 0.073] | 0.000 [0.000, 0.001] |
| 7 | 0.1 | +0.1 | 4 | 0.16 | 0.00 |
| 7 | 0.1 | +0.1 | 8 | 0.26 | 0.00 |
| 7 | 0.1 | +0.1 | 16 | 0.45 | 0.00 |
| 7 | 0.1 | +0.1 | 24 | 0.57 | 0.03 |
| 7 | 0.1 | +0.2 | 4 | 0.39 | 0.00 |
| 7 | 0.1 | +0.2 | 8 | 0.66 | 0.00 |
| 7 | 0.1 | +0.2 | 16 | 0.90 | 0.05 |
| 7 | 0.1 | +0.2 | 24 | 0.97 | 0.42 |
| 7 | 0.1 | +0.3 | 4 | 0.67 | 0.00 |
| 7 | 0.1 | +0.3 | 8 | 0.92 | 0.00 |
| 7 | 0.1 | +0.3 | 16 | 1.00 | 0.40 |
| 7 | 0.1 | +0.3 | 24 | 1.00 | 0.94 |

### T7. Choosing D: one-sided power at one look, n = 8 / 16 / 24 / 40 / 100, 3000 replicates a cell

| data | effect | uniform (today) | linear j (D) | quadratic j^2 | half on the top fraction | aGRAPA, prior 0.65 |
|---|---|---|---|---|---|---|
| binary | +0.0 | 0.00/0.00/0.01/0.01/0.02 | 0.00/0.01/0.02/0.02/0.03 | 0.00/0.01/0.02/0.02/0.03 | 0.00/0.01/0.02/0.02/0.02 | 0.00/0.03/0.04/0.04/0.04 |
| binary | +0.1 | 0.00/0.01/0.04/0.07/0.20 | 0.00/0.04/0.07/0.11/0.21 | 0.00/0.04/0.07/0.11/0.20 | 0.00/0.04/0.06/0.09/0.17 | 0.02/0.08/0.10/0.11/0.14 |
| binary | +0.2 | 0.00/0.07/0.15/0.31/0.71 | 0.02/0.13/0.22/0.35/0.70 | 0.02/0.13/0.22/0.35/0.67 | 0.02/0.13/0.21/0.30/0.63 | 0.05/0.17/0.21/0.27/0.52 |
| binary | +0.3 | 0.01/0.19/0.36/0.63/0.97 | 0.04/0.28/0.46/0.67/0.97 | 0.04/0.28/0.46/0.67/0.96 | 0.04/0.27/0.42/0.61/0.95 | 0.12/0.33/0.41/0.54/0.90 |
| binary | +0.5 | 0.06/0.62/0.88/0.99/1.00 | 0.19/0.73/0.92/0.99/1.00 | 0.19/0.73/0.92/0.99/1.00 | 0.19/0.73/0.90/0.98/1.00 | 0.40/0.74/0.86/0.97/1.00 |
| onramp | +0.0 | 0.00/0.00/0.00/0.00/0.01 | 0.00/0.00/0.00/0.01/0.02 | 0.00/0.00/0.00/0.01/0.03 | 0.00/0.00/0.01/0.01/0.03 | 0.00/0.00/0.01/0.02/0.02 |
| onramp | +0.1 | 0.00/0.00/0.03/0.14/0.53 | 0.00/0.01/0.06/0.19/0.57 | 0.00/0.02/0.07/0.22/0.58 | 0.00/0.02/0.08/0.22/0.55 | 0.00/0.04/0.12/0.23/0.48 |
| onramp | +0.2 | 0.00/0.04/0.23/0.67/0.99 | 0.00/0.08/0.33/0.73/0.99 | 0.00/0.12/0.39/0.77/0.99 | 0.00/0.13/0.40/0.76/0.99 | 0.01/0.21/0.47/0.76/0.98 |
| onramp | +0.3 | 0.00/0.23/0.69/0.97/1.00 | 0.00/0.36/0.78/0.98/1.00 | 0.00/0.43/0.81/0.99/1.00 | 0.01/0.46/0.82/0.98/1.00 | 0.04/0.56/0.84/0.98/1.00 |
| onramp | +0.5 | 0.00/0.89/1.00/1.00/1.00 | 0.04/0.95/1.00/1.00/1.00 | 0.11/0.97/1.00/1.00/1.00 | 0.15/0.97/1.00/1.00/1.00 | 0.34/0.98/1.00/1.00/1.00 |

### T8. One fixed look: the fixed-n bounded bet (valid for the mean on any distribution on the range, at the one n it was tuned to) beside F1 (valid at every look) and the exact sign-flip test; one-sided at 0.05, share decided, 2000 replicates a cell

All wins: fixed-n bounded bet p at n = 4 / 5 / 6 / 8 = 0.0767 / 0.0404 / 0.0213 / 0.0060

| data | effect | n | fixed-n bounded bet | F1 | exact sign-flip |
|---|---|---|---|---|---|
| binary | +0.0 | 8 | 0.005 [0.002, 0.009] | 0.000 [0.000, 0.002] | 0.005 [0.002, 0.009] |
| binary | +0.0 | 12 | 0.014 [0.009, 0.020] | 0.004 [0.002, 0.008] | 0.009 [0.006, 0.015] |
| binary | +0.0 | 16 | 0.029 [0.022, 0.037] | 0.013 [0.009, 0.019] | 0.018 [0.012, 0.024] |
| binary | +0.0 | 24 | 0.036 [0.028, 0.045] | 0.024 [0.018, 0.032] | 0.029 [0.022, 0.037] |
| binary | +0.3 | 8 | 0.12 | 0.04 | 0.11 |
| binary | +0.3 | 12 | 0.27 | 0.18 | 0.26 |
| binary | +0.3 | 16 | 0.36 | 0.28 | 0.39 |
| binary | +0.3 | 24 | 0.46 | 0.45 | 0.58 |
| binary | +0.5 | 8 | 0.39 | 0.20 | 0.38 |
| binary | +0.5 | 12 | 0.64 | 0.53 | 0.69 |
| binary | +0.5 | 16 | 0.77 | 0.72 | 0.84 |
| binary | +0.5 | 24 | 0.92 | 0.92 | 0.97 |
| onramp | +0.0 | 8 | 0.001 [0.000, 0.003] | 0.000 [0.000, 0.002] | 0.018 [0.012, 0.024] |
| onramp | +0.0 | 12 | 0.001 [0.000, 0.004] | 0.000 [0.000, 0.002] | 0.028 [0.021, 0.036] |
| onramp | +0.0 | 16 | 0.001 [0.000, 0.004] | 0.000 [0.000, 0.002] | 0.028 [0.021, 0.036] |
| onramp | +0.0 | 24 | 0.007 [0.004, 0.012] | 0.001 [0.000, 0.004] | 0.035 [0.027, 0.044] |
| onramp | +0.3 | 8 | 0.04 | 0.00 | 0.47 |
| onramp | +0.3 | 12 | 0.29 | 0.11 | 0.72 |
| onramp | +0.3 | 16 | 0.57 | 0.37 | 0.86 |
| onramp | +0.3 | 24 | 0.87 | 0.78 | 0.96 |
| onramp | +0.5 | 8 | 0.34 | 0.05 | 0.89 |
| onramp | +0.5 | 12 | 0.86 | 0.66 | 0.99 |
| onramp | +0.5 | 16 | 0.98 | 0.95 | 1.00 |
| onramp | +0.5 | 24 | 1.00 | 1.00 | 1.00 |

### T9. F4 = B + C + D + a first-look stake: one-sided at 0.05, capital half the package's fixed-n plug-in bet tuned to the first look's n0 and half the linear grid, valid at every look with no split

All wins, F4 p with n0 = n: n=4 0.0960, n=5 0.0523, n=6 0.0283, n=7 0.0153, n=8 0.0083, n=10 0.0025

One look at n (n0 = n), share decided toward the arm; at effect 0 with CP bounds

| data | effect | n | F1 | F4 |
|---|---|---|---|---|
| binary | +0.0 | 6 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| binary | +0.0 | 8 | 0.001 [0.000, 0.001] | 0.001 [0.000, 0.001] |
| binary | +0.0 | 10 | 0.002 [0.002, 0.003] | 0.002 [0.002, 0.003] |
| binary | +0.0 | 12 | 0.005 [0.004, 0.006] | 0.005 [0.004, 0.006] |
| binary | +0.0 | 16 | 0.009 [0.008, 0.010] | 0.009 [0.008, 0.011] |
| binary | +0.0 | 24 | 0.017 [0.016, 0.019] | 0.018 [0.016, 0.020] |
| binary | +0.2 | 6 | 0.00 | 0.00 |
| binary | +0.2 | 8 | 0.01 | 0.01 |
| binary | +0.2 | 10 | 0.04 | 0.04 |
| binary | +0.2 | 12 | 0.07 | 0.07 |
| binary | +0.2 | 16 | 0.12 | 0.13 |
| binary | +0.2 | 24 | 0.21 | 0.22 |
| binary | +0.3 | 6 | 0.00 | 0.00 |
| binary | +0.3 | 8 | 0.04 | 0.04 |
| binary | +0.3 | 10 | 0.10 | 0.10 |
| binary | +0.3 | 12 | 0.16 | 0.16 |
| binary | +0.3 | 16 | 0.28 | 0.29 |
| binary | +0.3 | 24 | 0.45 | 0.44 |
| binary | +0.5 | 6 | 0.03 | 0.03 |
| binary | +0.5 | 8 | 0.18 | 0.18 |
| binary | +0.5 | 10 | 0.38 | 0.38 |
| binary | +0.5 | 12 | 0.52 | 0.52 |
| binary | +0.5 | 16 | 0.73 | 0.74 |
| binary | +0.5 | 24 | 0.92 | 0.91 |
| onramp | +0.0 | 6 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| onramp | +0.0 | 8 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| onramp | +0.0 | 10 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.001] |
| onramp | +0.0 | 12 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.001] |
| onramp | +0.0 | 16 | 0.001 [0.000, 0.001] | 0.001 [0.001, 0.002] |
| onramp | +0.0 | 24 | 0.003 [0.002, 0.004] | 0.006 [0.005, 0.007] |
| onramp | +0.2 | 6 | 0.00 | 0.00 |
| onramp | +0.2 | 8 | 0.00 | 0.00 |
| onramp | +0.2 | 10 | 0.00 | 0.02 |
| onramp | +0.2 | 12 | 0.02 | 0.05 |
| onramp | +0.2 | 16 | 0.09 | 0.16 |
| onramp | +0.2 | 24 | 0.34 | 0.42 |
| onramp | +0.3 | 6 | 0.00 | 0.00 |
| onramp | +0.3 | 8 | 0.00 | 0.01 |
| onramp | +0.3 | 10 | 0.02 | 0.08 |
| onramp | +0.3 | 12 | 0.11 | 0.21 |
| onramp | +0.3 | 16 | 0.37 | 0.49 |
| onramp | +0.3 | 24 | 0.77 | 0.84 |
| onramp | +0.5 | 6 | 0.00 | 0.01 |
| onramp | +0.5 | 8 | 0.04 | 0.18 |
| onramp | +0.5 | 10 | 0.34 | 0.55 |
| onramp | +0.5 | 12 | 0.66 | 0.81 |
| onramp | +0.5 | 16 | 0.95 | 0.98 |
| onramp | +0.5 | 24 | 1.00 | 1.00 |
| skewed | +0.0 | 6 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| skewed | +0.0 | 8 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| skewed | +0.0 | 10 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| skewed | +0.0 | 12 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| skewed | +0.0 | 16 | 0.012 [0.011, 0.014] | 0.022 [0.020, 0.024] |
| skewed | +0.0 | 24 | 0.020 [0.018, 0.022] | 0.031 [0.029, 0.034] |

Continuation as T3 (first look at 8, up to 5 looks, data-sized adds)

| data | effect | F1 decided | F4 decided | F4 mean cases at stop |
|---|---|---|---|---|
| binary | +0.0 | 0.029 [0.027, 0.031] | 0.024 [0.022, 0.027] | 41.5 |
| binary | +0.2 | 0.472 | 0.398 | 46.0 |
| binary | +0.3 | 0.800 | 0.741 | 38.7 |
| onramp | +0.0 | 0.017 [0.015, 0.019] | 0.024 [0.022, 0.027] | 42.5 |
| onramp | +0.2 | 0.917 | 0.922 | 36.9 |
| onramp | +0.3 | 0.997 | 0.998 | 27.3 |
| skewed | +0.0 | 0.032 [0.029, 0.034] | 0.038 [0.036, 0.041] | 46.3 |

## 6. What each guarantee costs, and when it should apply

| Guarantee | What it costs (measured) | When it should apply |
|---|---|---|
| Valid for the mean on every distribution on the range (no t, no symmetry) | The largest cost at small n on graded data. At 8 apps, +0.3, one look: tests valid for the mean decide ≤ 0.04, while the symmetric sign-flip decides 0.47 and is invalid (T2-skewed: 0.099 against 0.05). On pass/fail it costs little: the fixed-n bounded bet equals the exact sign test (T8). | Always. It is why the package left the t test (#597, #695), and the skewed null shows the reason again. |
| Anytime validity (read again as cases are added) | At the first look only: pass/fail, +0.3, n = 8, the fixed-n bet decides 0.12 and F1 0.04. At n = 12 it is 0.27 against 0.18, and by n = 24 the cost is gone (T8). Interval width costs 4-15% (T5: I4 against I2). | On every stored comparison, which is the default store. It buys the re-read that lets a newcomer add cases. Kept in memory there is nothing to continue, and it buys nothing (option O1). |
| Two-sided hedge | A factor 2 in capital (B = C to the digit). On-ramp +0.3 at 24 apps: 0.26 against 0.47. | Only when no direction is declared. With a declared direction, tilt the capital 0.9 toward it and 0.1 away (R3). That costs 10% of capital (all wins at n = 8: p 0.0135 to 0.0150) and keeps "shown worse" for a gross regression (decided worse at down-capital ≥ 200: all losses decide from n = 10). |
| Holm across measures | A factor m. On a single-check eval with a question, m = 2 for one hypothesis counted twice. | Across readings that each get a verdict, and never on the declared primary. Never count a reading twice. |
| Equal-weight stake grid | T7, one-sided at 0.05: pass/fail +0.3, n = 16: 0.19 against 0.28 with weights ∝ j; on-ramp +0.3 at 16 apps: 0.23 against 0.36; no loss measured to n = 100. | Replace it with weights ∝ j. At large n with tiny effects it can need more cases (weight on j = 1 falls 5.5-fold). |

## 7. Recommended design

- **R1. The lead is the estimate, its interval and the direction.** The interval is read off the same capital
  as the decision: a 90% two-sided confidence sequence, each end a 95% one-sided bound, on a declared primary. It
  is 95% two-sided when no direction is declared. Above zero it is said as "at least X points better". At n = 8 it
  is 20-24% narrower than today's 97.5% print, and its coverage stayed ≥ 0.956 in every cell (T5, I3).
- **R2. Graded evidence is always shown.** It is the e-value for the declared direction, the current capital of
  the bet already computed, with no threshold. In words: weak below 3, moderate from 3, strong from 10 (the
  conventional steps for evidence ratios: Jeffreys; Kass and Raftery 1995). Under the null it reaches 10 at most
  1 time in 10, ever (Ville); measured 0.06. The reverse direction's e-value is named when it reaches 3. No
  surface says "no useful data".
- **R3. "Decided" means the capital ever reached 1/α = 20.** The capital is tilted 0.9 toward the declared
  direction and 0.1 away, so a decision "worse" needs the downward capital to reach 200. Once decided, it stays
  decided, as anytime validity allows.
- **R4. The declared primary is tested alone, at full α.** It is the question's measure: the first grader, with
  the direction the question asks (`quick/contract.py:406`). Secondary readings are shown as estimate, interval
  and e-value, and get no verdict, so they need no correction. If the owner wants verdicts on secondaries, gate
  them behind the primary (test them, Holm among themselves at α, only once the primary is decided). The primary
  then pays nothing. The trial pass rate never joins a family where it is the single grader's own values.
- **R5. The stake grid is weighted ∝ j.** Same 10 fractions, same cap.
- **R6. With no question asked:** read as today, two-sided and hedged with Holm across measures, but with R5, the
  duplicate removed, and R1 and R2 leading.
- **R7. Not recommended:** E (any first-look exact or permutation test of the mean on graded values), and the
  stratified "on these apps" test as a decision. The per-app tallies already printed are the scoped description:
  "on these apps, in this run".
- **Option O1 (owner's choice, not in the core).** A run kept in memory cannot be continued, so it can be read by
  the package's existing fixed-n bounded bet (`bounded_mean_p` with `planned_n` = n) instead of the mixture. Pass/fail
  +0.3 at n = 8: 0.12 against 0.04; on-ramp +0.5 at 8 apps: 0.34 against 0.05 (T8). It adds a second read rule, so
  it trades against "fewer concepts". F4 (T9), which mixes that bet into a stored comparison's capital, gains on
  graded data at one look (on-ramp +0.5 at 8 apps: 0.18 against 0.04). Under continuation it loses on pass/fail
  (+0.2: 0.40 against 0.47), so it is not recommended.

## 8. The first screen at n = 8, as a non-expert would see it

The numbers come from the scratch, for the same three samples as today's lines in section 1. A question is
asked, so "v2 passes more often than v1" is the declared primary.

**All wins** (v2 passed every case, v1 none):

> v2 did better than v1: it passed all 8 cases and v1 passed none, a gain of 100 points. We can be 95% sure the
> gain is at least 21 points. The evidence for v2 is strong (74; by chance alone a score of 10 comes up at most 1
> time in 10), and it clears the bar: v2 is better on cases like these.

**About +0.3** (k = 3: v1 passed 8 of 24 trials, v2 15):

> v2 looks better than v1: it passed 15 of 24 trials on 8 cases and v1 passed 8, a gain of 29 points. The
> evidence for v2 is moderate (3.6), not yet enough to decide. The true change could be anywhere from 24 points
> worse to 60 points better (90% interval). Add cases and run it again to narrow this. The read stays fair however
> often you look.

**No effect** (k = 3: v1 8 of 24, v2 9):

> v2 and v1 look about the same: v2 passed 9 of 24 trials on 8 cases and v1 passed 8, a gain of 4 points. The
> evidence either way is weak (1.1). The true change could be anywhere from 39 points worse to 44 points better
> (90% interval), so a difference that large is not ruled out yet.

## 9. What would change in the code (sketch only)

- `analysis/stats.py`:
  - `_mixture_log_path` (`:2343`) takes weights ∝ j over the same `_MIXTURE_STAKES` fractions.
  - `_hedged_log_path` (`:2382`) takes the tilt: 1/2 for two-sided, 0.9 toward a declared direction.
  - `bounded_separation_p` (`:2503`) gains the declared direction.
  - The current capital, the e-value, is returned beside the p. `SeparationTest` gains it.
  - `bounded_difference_interval` gains the same tilt, for the 90% sequence on a directional primary.
- `analysis/bundle/comparisons.py`:
  - `_corrected_family` (`:825`) leaves the declared primary out of Holm and reads it at α (`:900`, `:907`, `:952`).
  - Secondaries keep estimate, interval and e-value, with no verdict (or gated, R4).
  - `:1158-1159` stops adding the trial pass rate where it duplicates the single grader.
- Stored shape: `FamilyComparison` gains an optional `evidence` (the e-value). It is additive, with a MIGRATIONS
  entry. Rule 11's reach-back has to be checked against the base model: does a pre-change build ignore the field
  or refuse it?
- Wording:
  - The lead stops opening on `cannot_tell_apart` (`analysis/style.py:618`). It opens on the estimate and
    direction, the interval and an evidence word, plus "decided" where the bar is cleared.
  - `interval_words` (`analysis/report/verdict.py:542-579`) says "at least X better" for a one-sided bound.
  - `GLOSSARY` gains "evidence".
  - `CANNOT_TELL_APART` stays as the glossary's term for an undecided contrast, but no longer leads.
- Tests that will move (to report, not move, in the build chunk):
  - `tests/test_rereadable_bet.py`: `_reference_crossing` restates the uniform construction, and the pins 742/872
    and the null counts 66/24/53.
  - The oracle and presentation pages whose first lines change (`test_hot_path_oracle.py`,
    `test_presentation_one_reading.py`).
  - The on-ramp reading tests on the harness branch.
- New seeded simulations (`long_simulation`, each with a unit smoke shard):
  - F1's false separation under continuation, with the Clopper-Pearson upper bound ≤ α, on the three nulls here.
  - The graded e-value under the null, upper bound ≤ 1/10.
  - The tilted interval's coverage.
  - Each should fail for a wrong formula, for example a tilt that doesn't sum to 1, or a grid whose weights
    aren't normalised.

## 10. Literature each option follows

Only works I am sure of are cited.

- **The bet, its hedging, the predictable plug-in, aGRAPA and mixtures over stakes (A, B, D, F4).** Waudby-Smith,
  I. and Ramdas, A. (2024), "Estimating means of bounded random variables by betting", *JRSS-B* 86(1). The package
  already cites its §4.2 (hedged capital) and §5.6 (diversified stakes).
- **Validity at every look, and the e-value ever reaching 1/c at most 1/c of the time.** Ville, J. (1939),
  *Étude critique de la notion de collectif*. Howard, S. R., Ramdas, A., McAuliffe, J. and Sekhon, J. (2021),
  "Time-uniform, nonparametric, nonasymptotic confidence sequences", *Annals of Statistics* 49(2). The latter
  covers the intervals of R1.
- **E-values as graded evidence (R2).** Vovk, V. and Wang, R. (2021), "E-values: calibration, combination and
  applications", *Annals of Statistics* 49(3). Shafer, G. (2021), "Testing by betting: a strategy for statistical
  and scientific communication", *JRSS-A* 184(2): report the betting score itself. Grünwald, P., de Heide, R. and
  Koolen, W. (2024), "Safe testing", *JRSS-B*. Ramdas, A., Grünwald, P., Vovk, V. and Shafer, G. (2023),
  "Game-theoretic statistics and safe anytime-valid inference", *Statistical Science* 38(4).
- **The 3 and 10 steps.** Jeffreys' scale, as in Kass, R. E. and Raftery, A. E. (1995), "Bayes factors", *JASA*
  90(430).
- **Estimate and interval first (R1).** Wasserstein, R. L. and Lazar, N. A. (2016), "The ASA statement on
  p-values", *The American Statistician* 70(2).
- **Holm (C, R4).** Holm, S. (1979), "A simple sequentially rejective multiple test procedure", *Scandinavian
  Journal of Statistics* 6(2). Gating secondaries behind a primary is the fixed-sequence (gatekeeping) principle.
- **E's split.** α spent between looks by the union bound, as in alpha spending (Lan, K. K. G. and DeMets, D. L.
  (1983), *Biometrika* 70(3)). The sign-flip test is Fisher's randomisation test, *The Design of Experiments*
  (1935).
- **The stratified test (T6).** Mantel, N. and Haenszel, W. (1959), *JNCI* 22(4). The exact conditional form is
  Zelen, M. (1971), "The analysis of several 2×2 contingency tables", *Biometrika* 58(1).
- **No valid test of an unbounded mean.** Bahadur, R. R. and Savage, L. J. (1956), *Annals of Mathematical
  Statistics* 27(4), as the package already cites.
- **Clopper-Pearson.** Clopper, C. J. and Pearson, E. S. (1934), *Biometrika* 26(4).

## 11. Open questions for the owner

1. Adopt R1 to R5 (a one-sided declared primary at full α with the 0.9 tilt, the ∝ j grid, graded evidence always
   shown, and the estimate leading)?
2. Should secondaries get no verdict, or verdicts gated behind the primary (R4)?
3. Should O1 apply (a fixed-n read for runs kept in memory), or one rule everywhere?
4. The build chunk moves `test_rereadable_bet.py`'s construction and pins and the first-line goldens: is that
   sanctioned as a deliberate design change?
