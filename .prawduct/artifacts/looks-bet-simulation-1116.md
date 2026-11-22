# Looks: α/L against one prefix bet, simulated (2026-11-16)

Tests the condition in second-opinion-1116 §7 (prefix bet losing more than about 25% in cases-to-decision against α/L at a plan of 2 or 3), on `evals/main` @56edc697e.

## Setup

Two arms on the same cases pass independently, k trials per case (k = 1; k = 3 as a check), at base 0.5, 0.8 or 0.95; an alternative lowers the candidate 5, 10 or 20 points. Launches of 20 or 40, plans of L = 2, 3 or 5, stop at the first separation or the plan's end.

Extends are **fixed** (step = launch) or **planner**-sized: `todays_planner` of `tests/test_simulated_looks.py` (`cases_to_separate`, two-sided, power 0.8, at the next look's level, cap 100 a step; sizing nothing stops the campaign).

- **α/L (today).** Each look runs a fresh bounded bet on all cases held (tuned to n, order `bet_order(n)`), separated when its 1 − α/L interval clears zero: the campaign row's rule (`_rate_row_separates`) behind `FALSE_SEPARATION`'s bounded rows.
- **Prefix bet (proposal).** One bet fixed before the first look for N = launch + (L − 1) × step (the plan's most; the step is the cap of 100 under the planner), read in arrival order at α by the writer's rule (`writer_pairwise._separated`: p < α and the 95% interval clear of zero). Extends sized at α; a variant sizes them at α/L, giving both methods the same sizes.

The bet applies to a paired difference as implemented (differences on [−1, 1]; two-sided p = 2 × min of two one-sided bets at α/2). Unpaired, `_check_planned` refuses `planned_n`; nothing unpaired was simulated.

**Package code.** Imported and called as they are: `stats.bet_order`, `bounded_difference_interval`, `bounded_separation_p`, `bounded_mean_p`, `SIGNIFICANCE_ALPHA`, `significance.looks_alpha`, `separation.cases_to_separate`, `POWER`, `NoSpread`. Re-implemented in numpy for speed: the betting capital (plug-in stakes, cap 0.9, running maximum), the two decision rules and the planner wrapper. Checks against the imports (`validate.json`):
- Decisions agreed with the package in 300 of 300 samples per rule (about a third separations).
- The p-values matched `bounded_mean_p` with a relative error of 0.0.
- Two whole cells run on package calls only matched in 300 of 300 replicates.
- The α/L arm reproduced six rows pinned in `test_simulated_looks.py`, all within 4 SE.

## Null error

20,000 replicates per cell; every exact 95% half-width ≤ 0.24 points. Full table: `tables.md`.

| k | extends | α/L, largest rate [95% CI] | bet, largest rate [95% CI] |
|---|---|---|---|
| 1 | planner (data-sized) | 3.12 [2.89, 3.38] (0.5, 20, L=2) | 2.19 [1.99, 2.40] |
| 1 | fixed | 2.95 [2.72, 3.19] | 3.12 [2.88, 3.37] |
| 3 | planner | 2.30 [2.10, 2.52] | 2.05 [1.86, 2.26] |
| 3 | fixed | 1.99 [1.81, 2.20] | 2.94 [2.71, 3.18] |

- **α/L holds α with data-sized extends** in all 60 cells: every upper bound under 5% (measured here, not a theorem).
- **The bet holds α in every cell**, as Ville's inequality guarantees whatever the sizing.

## Alternatives, k = 1 (10,000 replicates per cell)

Ranges over base rate and launch. Power at the plan's end; E[N] = mean cases read at stop.

| extends | L | Δ | power α/L % | power bet % | bet power, same extend sizes % | E[N] bet/(α/L) | same sizes | N_max / E[N] |
|---|---|---|---|---|---|---|---|---|
| fixed | 2 | 5 | 2–8 | 3–11 | | 1.00 | | 1.0 |
| fixed | 2 | 10 | 6–30 | 8–36 | | 0.99–1.00 | | 1.0 |
| fixed | 2 | 20 | 20–82 | 25–86 | | 0.97–0.99 | | 1.0–1.3 |
| fixed | 3 | 5 | 3–11 | 4–17 | | 0.99–1.00 | | 1.0 |
| fixed | 3 | 10 | 8–44 | 10–55 | | 0.97–0.99 | | 1.0–1.1 |
| fixed | 3 | 20 | 30–95 | 40–98 | | 0.94–0.98 | | 1.1–1.6 |
| planner | 2 | 5 | 4–12 | 3–12 | 4–14 | 0.95–0.98 | 1.00–1.01 | 1.1–1.4 |
| planner | 2 | 10 | 13–46 | 11–38 | 12–46 | 0.93–0.98 | 1.00–1.01 | 1.2–1.5 |
| planner | 2 | 20 | 48–88 | 38–63 | 44–76 | 0.91–0.95 | 1.00–1.02 | 1.5–2.0 |
| planner | 3 | 5 | 7–24 | 6–24 | 8–30 | 0.91–0.97 | 1.00 | 1.2–1.5 |
| planner | 3 | 10 | 24–72 | 22–53 | 28–72 | 0.86–0.94 | 0.99–1.00 | 1.2–1.7 |
| planner | 3 | 20 | 76–97 | 43–59 | 61–80 | 0.85–0.88 | 1.00–1.03 | 1.9–3.0 |

Under the planner E[N] cannot show the cost: both methods stop at nearly the same n, so it shows as power, up to 30 points lost at Δ = 20, L = 3 (k = 3: 45), where the plan reads a third to a half of its declared maximum. At equal power instead:

**Equal power (80% at the plan's end), batches fixed at s, 2s, …, Ls, bet fixed for N = Ls** (4,000 replicates per grid point):

| L | base | Δ | N at 80%, α/L | N at 80%, bet | E[N], α/L | E[N], bet | E[N] ratio |
|---|---|---|---|---|---|---|---|
| 2 | 0.5 | 5/10/20 | 2383/602/147 | 2114/533/131 | 1885/476/119 | 1719/432/107 | 0.91/0.91/0.90 |
| 2 | 0.8 | 5/10/20 | 1654/457/130 | 1514/400/116 | 1304/363/104 | 1239/326/94 | 0.95/0.90/0.90 |
| 2 | 0.95 | 5/10/20 | 668/216/76 | 592/195/68 | 534/173/61 | 486/159/56 | 0.91/0.92/0.92 |
| 3 | 0.5 | 5/10/20 | 2478/622/157 | 2151/541/133 | 1815/460/117 | 1639/408/100 | 0.90/0.89/0.85 |
| 3 | 0.8 | 5/10/20 | 1793/485/137 | 1438/411/116 | 1320/359/103 | 1093/311/88 | 0.83/0.87/0.85 |
| 3 | 0.95 | 5/10/20 | 695/234/82 | 600/197/69 | 517/174/63 | 462/152/54 | 0.89/0.88/0.86 |

**A declared maximum larger than the cases read.** One read at n; cases for 80% power; the bet fixed for M = f × n. Ratios are against the fixed-n test at α/2 and α/3.

| base | Δ | n80 α/2 | n80 α/3 | f = 1 | f = 2 | f = 4 | f = 8 |
|---|---|---|---|---|---|---|---|
| 0.5 | 5 | 2485 | 2662 | 0.84/0.79 | 0.99/0.93 | 1.35/1.26 | 2.02/1.89 |
| 0.5 | 10 | 626 | 677 | 0.86/0.80 | 0.96/0.88 | 1.33/1.23 | 1.98/1.83 |
| 0.5 | 20 | 156 | 169 | 0.85/0.79 | 0.96/0.88 | 1.25/1.16 | 1.89/1.74 |
| 0.8 | 5 | 1711 | 1853 | 0.86/0.79 | 0.99/0.92 | 1.37/1.26 | 2.09/1.93 |
| 0.8 | 10 | 470 | 513 | 0.86/0.79 | 0.97/0.89 | 1.34/1.22 | 2.00/1.84 |
| 0.8 | 20 | 134 | 146 | 0.87/0.80 | 0.94/0.86 | 1.26/1.16 | 1.93/1.77 |
| 0.95 | 5 | 700 | 758 | 0.86/0.79 | 0.97/0.90 | 1.30/1.20 | 2.01/1.86 |
| 0.95 | 10 | 229 | 247 | 0.86/0.79 | 0.97/0.90 | 1.29/1.19 | 1.95/1.80 |
| 0.95 | 20 | 80 | 86 | 0.85/0.79 | 0.92/0.86 | 1.27/1.19 | 1.94/1.81 |

## Ratio against the 25% threshold (L = 2 and 3)

- **Declared maximum equal to what the plan reads** (fixed batches): at equal power the bet needs 5–10% fewer expected cases at L = 2, 10–17% fewer at L = 3. Never past the threshold.
- **Declared maximum f times the cases read:** break-even near f = 2. At f = 4 the bet needs 25–37% more cases than α/2, past the threshold in all 9 settings (the lowest at 1.253), and 16–26% more than α/3, past it in 2 of 9. At f = 8 it needs 1.7–2.1 times as many.
- **Planner extends, cap-based maximum:** E[N] equal or 3–15% lower, but less power wherever the plan reads under about half its declared N.

## Verdict

At L = 2 or 3 the package's prefix bet (stakes tuned to N) loses no more than 25% in cases-to-decision when its declared maximum is within about twice the cases the comparison reads: there it costs fewer cases than α/L and holds α by theorem. It crosses the threshold when the declared maximum is about 4 times the cases read or more. Today's planner, with its 100-case cap, puts large effects in that regime, where the bet's power at the plan's end falls by up to 30 points (k = 1) or 45 (k = 3).

**Updated after the mixture arm (section below).** A stake that does not depend on N removes most of that penalty when the stake is the grid mixture, but not in every cell:
- **Cases.** The grid mixture needs at most 25% more cases than α/L in 34 of 36 equal-power cells. It fails at k = 1, L = 2, Δ = 5 on base rates 0.5 (1.35) and 0.8 (1.28).
- **Error.** It holds α everywhere.
- **Power, same extend sizes as α/L.** It is within 12 points of α/L at the plan's end in every planner cell. Four cells, all k = 3 and L = 2, lose 10.6–11.8 points.
- **Power, extends sized at α.** It loses up to 24 points, because it then reads 1–16% fewer cases.

The time-varying plug-ins do not remove the penalty at either cap tried.

Limits: arms independent within a case (as in the package's own simulations; correlated cases not tested); k = 1 and 3 only; the test file's planner, not the caveat step's, which asks for more cases; the α/L arm on the campaign row's interval rule, not the bundle's t-p conjunction; only the package's plug-in stakes (stakes that do not depend on N, such as a mixture, might shrink the over-declaration penalty; not simulated).

## Reproduce

Master seed `20261116`; each cell's seed is `stable_seed(tag, …)` in the script. About 25 minutes.

```
cd /tmp/claude-1000/-opt-3tears/cf894fbb-3c9f-4187-bb67-8d3d05985b12/scratchpad/looks-sim
/opt/3tears/.claude/wt/evals-main/.venv/bin/python -I looks_sim.py all
/opt/3tears/.claude/wt/evals-main/.venv/bin/python -I analyze.py > tables.md
```

Raw JSON is beside the script.


## Mixture stakes

The arm added at the owner's request: stakes that do not depend on the declared maximum N. Script: `mix.py`; tables: `mix_analyze.py`; both sit beside `looks_sim.py`. Draws and seeds are those of the runs above, so the α/L and package-bet columns repeat the earlier numbers exactly.

**What was implemented.** Each arm runs on the paired differences rescaled to [0, 1], with null m = 1/2:
- **grid**, the mixture over a fixed grid of betting fractions ("diversified Kelly", Waudby-Smith & Ramdas 2024, JRSS-B 86(1), §5.6): K±_t = (1/D) Σ_d Π_{i≤t} (1 ± λ_d (z_i − m)), with λ_d = (d/10)·0.9/m and D = 10, so λ runs 0.18 to 1.8.
- **wsr**, WSR's predictable plug-in for confidence sequences (same paper, eq. 26–27): λ_t = sqrt(2 log(2/α) / (σ̂²_{t−1} t log(1+t))), truncated. It shrinks like 1/sqrt(t log t). Its μ̂ and σ̂² are WSR's, which are also the package's.
- **lit**, the brief's literal shape, not taken from a paper: λ_t = sqrt(2 log(2/α) log(1+t) / (σ̂²_{t−1} t)), truncated. It shrinks like sqrt(log t / t).
- The two plug-ins are truncated at the package's 0.9/m and again at WSR's recommended c/m with c = 1/2 (labelled c50).
- All new arms are two-sided by hedged capital (WSR §4.2, θ = 1/2): reject at the first t with (K⁺_t + K⁻_t)/2 ≥ 1/α. The package's bet stays 2 × min of two one-sided p-values.
- A scalar loop reproduced every crossing index on test rows (`mix.py check`).

**Null** (20,000 replicates a cell; 36 cells at k = 1, 24 at k = 3). No arm's lower 95% bound exceeds 5%.

| arm | largest rate, k = 1 [95% CI] | largest rate, k = 3 [95% CI] |
|---|---|---|
| α/L | 3.12 [2.89, 3.38] | 2.30 [2.10, 2.52] |
| bet (package) | 3.12 [2.88, 3.37] | 2.94 [2.71, 3.18] |
| grid | 3.07 [2.84, 3.32] | 1.93 [1.74, 2.13] |
| wsr | 4.11 [3.84, 4.39] | 2.62 [2.41, 2.86] |
| lit | 4.47 [4.19, 4.77] | 4.19 [3.92, 4.48] |
| wsr c50 / lit c50 (k = 1 and 3 together) | 2.89 [2.66, 3.13] / 4.15 [3.88, 4.44] | |

**Cases-to-decision at equal power** (80% at the plan's end, fixed batches s, …, Ls; E[N] at 80% over α/L's; 36 cells: k = 1 and 3 × L = 2 and 3 × 3 bases × 3 effects):

| arm | ratio range | cells > 1.25 |
|---|---|---|
| bet (package, N = Ls) | 0.78–0.93 | 0 / 36 |
| grid | 0.94–1.35 | 2 / 36: k=1 L=2 base 0.5 Δ=5 (1.35); k=1 L=2 base 0.8 Δ=5 (1.28) |
| wsr | 0.76–5.94 | 22 / 36 |
| lit | 0.75–22.9 | 21 / 35 (one never reached 80%) |
| wsr c50 | 1.01–2.70 | 14 / 36 |
| lit c50 | 0.81–12.8 | 12 / 36 |

At L = 3 the grid's largest ratio is 1.22. All cells are in `tables_mix.md`.

**The declared maximum f times the cases read** (one read at n; cases for 80% power over the fixed-n test at α/2 / α/3). The grid ignores N, so it has one ratio whatever f is:

| k | bet f=1 | f=2 | f=4 | f=8 | grid (any f) |
|---|---|---|---|---|---|
| 1 | 0.84–0.87 / 0.79–0.80 | 0.92–0.99 / 0.86–0.93 | 1.25–1.37 / 1.16–1.26 | 1.89–2.09 / 1.74–1.93 | 0.96–1.32 / 0.90–1.22 |
| 3 | 0.84–0.87 / 0.77–0.82 | 0.89–1.04 / 0.82–0.96 | 1.17–1.38 / 1.09–1.28 | 1.72–2.13 / 1.59–1.98 | 0.97–1.09 / 0.89–1.01 |

The grid is past 1.25 against α/2 only at k = 1 with Δ = 5 (base 0.5: 1.28; base 0.8: 1.32).

**Planner regime, power at the plan's end** (cap 100, N = launch + (L−1)·100, 10,000 replicates a cell, 72 cells per sizing).

Δ = 20 cells, extends sized at α/L for every arm (same sizes), power % ranges over base and launch:

| k | L | α/L | bet | grid | wsr | lit |
|---|---|---|---|---|---|---|
| 1 | 2 | 48–88 | 44–76 | 44–88 | 24–74 | 25–86 |
| 1 | 3 | 76–97 | 61–80 | 78–99 | 35–84 | 26–89 |
| 3 | 2 | 27–93 | 19–82 | 17–86 | 50–97 | 50–99 |
| 3 | 3 | 29–97 | 16–55 | 25–97 | 68–99 | 68–100 |

The ranges span different cells, so the per-cell difference decides. Arm minus α/L, worst cell, and the cells losing more than 10 points:

| arm | same extend sizes (at α/L): worst; cells < −10 | own sizes (bets at α): worst; cells < −10 |
|---|---|---|
| bet (package) | −45.1; 17 / 72 | −58.4; 32 / 72 |
| grid | −11.8; 4 / 72 (k=3, L=2: base 0.95 launch 40 Δ=20 and Δ=10; base 0.8 launch 20 Δ=20; base 0.5 launch 20 Δ=20) | −24.1; 40 / 72 |
| wsr | −49.8; 19 / 72 | −58.8; 31 / 72 |
| lit | −60.3; 15 / 72 | −62.1; 18 / 72 |
| wsr c50 | −42.8; 28 / 72 | −43.9; 52 / 72 |
| lit c50 | −42.8; 22 / 72 | −43.9; 30 / 72 |

- **Grid, same sizes.** Ten cells lose more than 5 points, all at k = 3, L = 2, or k = 1, L = 2 with Δ = 20 (−7.7 at worst there). At L = 3 and Δ = 20 it gains 2–3 points at k = 1.
- **Grid, own sizes.** E[N] is 0.84–0.99 of α/L's. The planner sized at α asks for fewer cases, so the extra loss comes from the sizing, not the test.
- **Why the plug-ins fail.** At the 0.9 cap they bet too hard early on pass/fail differences and spend the later cases recovering. At c = 1/2 they cannot bet hard enough on large effects at a high base rate (−43 points at base 0.95, Δ = 20).

**Answer.** No N-independent stake meets all three conditions in every cell. The grid mixture comes closest:
- **α held:** yes.
- **≤ 25% more cases:** yes, except k = 1, L = 2, Δ = 5 at bases 0.5 and 0.8 (1.35 and 1.28).
- **Power at the plan's end:** no large loss when it gets α/L's extend sizes (worst −11.8, k = 3, L = 2). Losses of up to 24 points appear when its extends are sized at α.

The WSR and literal plug-ins fail on cases and on power at either cap.

Limits: one grid was tried (linear, D = 10, λ from 0.18 to 1.8); a log-spaced grid or a smaller first λ might move the Δ = 5 cells, and was not run. The other limits above still apply.

Reproduce, from the scratchpad directory (about 2.5 hours, dominated by the k = 1 equal-power search):

```
P=/opt/3tears/.claude/wt/evals-main/.venv/bin/python
$P -I mix.py check; $P -I mix.py null; $P -I mix.py planner
$P -I mix.py power80 1; $P -I mix.py power80 3; $P -I mix.py overdeclare 1; $P -I mix.py overdeclare 3
$P -I mix.py c50; $P -I mix.py c50power 1; $P -I mix.py c50power 3
$P -I mix_analyze.py > tables_mix.md
```
