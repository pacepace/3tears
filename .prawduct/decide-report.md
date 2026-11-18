# Should I switch? Advice beside the proof

A design study: no product behaviour changed, no model was called, no test was run. Every number is from a seeded
simulation or from the package's own functions at evals/main 77137c804, run locally. The scratch is committed under
`.prawduct/artifacts/decide-scratch/`, and its restatement of the bet is checked equal to the package before any table is drawn
(appendix).

## Recommendation

Add one line above the proof: **advice**, "Switch to v2" or "Keep v1", read off the evidence figure the page already
prints. The advice says switch once the evidence that v2 is better has reached **moderate (3)**, while v2 is still
ahead and the evidence against it has not reached the same bar; when the user has said the switch is **hard to undo**,
the bar is **strong (10)**. Otherwise it says keep. The proof line keeps "decided", its α and its bar (about 22 on the
same figure), and `gate()` stays on the proof. The user declares one new thing, `undo="easy"` (the default) or
`"hard"`, before the results, as they declare the question. The advice keeps one promise however often anyone looks:
**if v2 is not really better, it says switch at most one time in three, and one time in ten when the switch is hard to
undo** (Ville's inequality on the e-value the package already computes; measured 0.22 to 0.27 and 0.06 to 0.07). On
held-out-1 it reads "Switch to docs@candidate" at either bar (evidence 17.6), while the proof reads "not yet". Against
today's "act only when decided", the worst expected regret over four data families falls from 30, 23 and 14 points to
19, 13 and 7 at 8, 24 and 100 cases, with no more than 2 points of that from wrong switches. The rules that switch more
often at 8 cases (always take the leader; a posterior probability of being better; an expected-loss threshold) pay for
it once people look repeatedly: up to 13 points of expected loss from wrong switches for the leader and 5 to 9 for the
Bayesian rules, with a promise only "if effects are like the prior"; and the posterior probability itself moves from
0.74 to 0.99 on the same 8 cases as the prior changes. "Not enough to say" never appears: with an incumbent a tie keeps it, which is a clear answer.

## 1. The two questions on one screen

**The rule of the screen.**

- Two labelled lines, the advice first, because it is what a reader acts on.
- The advice line names an action ("Switch to", "Keep") and never says "better", "shown" or "decided". The proof line
  names a status ("Proven", "not yet", "shown worse") and never says "switch" or "keep". With no shared words, the
  two cannot contradict each other, and a reader sees they answer different questions.
- "Undecided" leaves the lead. Beside "Switch to v2" it reads as "we don't know what to do". The proof line says
  "not yet", which is what "undecided" means there: not proven to the standard a claim needs.
- The proof line states its bar in the same figure the evidence is printed in: "the evidence is 18; a claim, a gate
  or a promotion needs 22". 22 is the figure at which the tilted capital (0.9 of it on the asked side) reaches 1/α =
  20. That answers the review note that "strong (18)" beside "undecided" is easy to misread.
- Everything below these two lines (the interval in full, the tallies, the disclosure) is unchanged.

**The on-ramp result (held-out-1, 24 apps), either bar:**

> **Advice: switch to docs@candidate.** Over these 24 apps it is 22 points ahead of docs@control, and at worst it is
> 1 point behind.
>
> **Proven: not yet.** The evidence that docs@candidate is better is strong (18); a claim, a gate or a promotion
> needs 22.

Under the plan, the result stays "no result; not promoted": the plan promotes on the proof, and the advice does not
change that. A consumer choosing which build to run reads the first line.

The other cases are drafted in section 4.

## 2. The statistic behind the advice

Six candidates were measured on the same simulated data (section 3). Each is described here with what it costs a
non-expert to learn and what it can promise.

### 2(a). The probability that v2 is better

A posterior probability is the most natural sentence ("a 97% chance v2 is better"), and it is what the coordinator
reached for. Its model here: the per-case difference has a Dirichlet-process prior (Ferguson 1973) whose base is
uniform on the difference's range, worth `a` cases. `a = 0` is the Bayesian bootstrap (Rubin 1981). Its first two
moments are exact in closed form, and a normal approximation to them gave the simulation's probabilities (checked
against 40,000-draw Monte Carlo on the named cases: 0.9734 against 0.9720 at 8 cases). Sceptical normal priors
(Spiegelhalter et al. 1994) and a uniform Dirichlet on the seven values a k = 3 difference takes are shown beside it.

**T-A. P(v2 better) on the named cases, by prior** (Monte Carlo, 40,000 draws; `named.txt`)

| case | Bayesian bootstrap | DP, a = 1 | DP, a = 2 | DP, a = 2 at the range's ends | DP, a = 8 | Dirichlet(1) on the lattice | sceptical N(0, 0.05²) / (0, 0.1²) / (0, 0.2²) |
|---|---|---|---|---|---|---|---|
| all wins, 8 cases | 1.000 | 1.000 | 1.000 | 0.998 | 0.998 | 0.996 | 0.999 / 1.000 / 1.000 |
| +0.3, 8 cases (8 vs 15 of 24 trials) | 0.990 | 0.972 | 0.956 | 0.908 | 0.879 | 0.867 | 0.739 / 0.868 / 0.945 |
| on-ramp, 24 apps | 1.000 | 0.998 | 0.997 | 0.989 | 0.986 | 0.981 | 0.943 / 0.989 / 0.997 |
| no effect, 24 cases | 0.479 | 0.484 | 0.479 | 0.490 | 0.491 | 0.494 | 0.500 / 0.500 / 0.500 |
| small regression (-5), 100 cases | 0.063 | 0.067 | 0.068 | 0.079 | 0.082 | 0.082 | 0.106 / 0.076 / 0.067 |

- **Sensitivity.** At 24 cases and beyond, every prior tried agrees to within a few points: the on-ramp reads 0.94
  to 1.00, matching the coordinator's 95 to 99%. At 8 cases it does not: the same +0.3 sample reads anywhere from
  0.74 to 0.99. A non-expert cannot choose among those priors, and the 8-case reading is the one the owner most wants
  to be useful.
- **What it promises.** The posterior stays calibrated under optional stopping only when true effects are drawn from
  the prior (Rouder 2014; Deng, Lu and Chen 2016). Section 3 shows that when the data's spread differs from what the
  model expects, a "switch at 0.9" rule, read at every look, advised switching to an arm 20 points worse 23 to 25% of
  the time (T-D).
- **Concepts.** A new number and a prior, both beside an e-value and an interval that answer differently, so "97%"
  and "undecided" would sit on one screen.

### 2(b). The worst plausible case against a declared margin

The lower end of the interval already printed (the current-capital 90% confidence sequence), switching when it is no
worse than `-m`. It is the owner's own reasoning about the on-ramp ("the worst plausible case is -1"), and its
promise is the cleanest: if v2 is worse than the margin, it says switch at most one time in twenty, however often you
look.

It is too strict to lead. At 8 cases the 90% interval is about 100 points wide, so it advises switching on +0.3
almost never (0.00 to 0.05 at margins 0 to 10 points, on the on-ramp model: `tables_4000.md`), and its worst expected
regret is about today's: 29.9 points at 8 cases and 18.5 to 26.6 at 24 (T-B and `tables_4000.md`). The worst plausible case is the right **caveat** for a switch, and the design uses it
there (section 5). It is not the right rule.

### 2(c). Expected gain, expected loss, regret

Regret is the yardstick for every rule here: the value of the better action less the value of the one taken. In
points, it is the arm's true gain when the advice kept a better arm, and its true loss when the advice switched to a
worse one. Expected loss is also a rule: switch when the posterior expected loss of switching, E[max(0, -δ)], falls
below a threshold of caring (1 point here). That is the stopping rule used in industry Bayesian A/B testing (Stucchio
2015). It shares 2(a)'s prior and its weakness under repeated looks (T-B, T-D).

### 2(d). What the decision-theory literature adds

- **At one look, "follow the leader" is near minimax-regret** (Manski 2004; Stoye 2009; Manski and Tetenov 2016), and
  a significance test is far from it. T-B reproduces both: at one look the leader's worst regret is 8.7, 4.3 and 1.9
  points at 8, 24 and 100 cases, and today's is 30, 23 and 14. Claxton (1999) makes the same case in health
  economics: decide on expected value; inference decides whether to collect more. Feit and Berman (2019) show
  profit-maximising tests are small, with "deploy the better posterior mean" at the end.
- **Under repeated looks, the leader is the worst rule here.** Read after every case, it switches to a no-better arm
  87 to 92% of the time, and to an arm 20 points worse 24 to 59% of the time (T-D). The industry answer to peeking is
  always-valid inference (Johari et al. 2022): the same family of guarantees as the package's bet.
- **E-values let the stakes set the bar.** An e-value's bound P(ever ≥ c) ≤ 1/c holds for any c fixed in advance
  (Ville 1939), so a cheap, reversible action can be taken at a lower c than a claim. That is the design: the bar is
  chosen by the cost of being wrong, before the data. (Grünwald 2024 goes further, allowing a level chosen after the
  data in an expected-risk sense; the design does not need it.)

### The choice

The **evidence ladder**: weak, moderate, strong, then proof, each rung a bar on the one figure the page already
prints.

| Rung | Evidence for v2 | What it allows |
|---|---|---|
| weak | below 3 | keep |
| moderate | 3 or more | switch, when the switch is easy to undo (the default) |
| strong | 10 or more | switch, when the switch is hard to undo |
| proven | the tilted capital reaches 20 (about 22 on this figure) | a claim, a publication, a promotion, a gate |

**Why this one.**

- **No new concept.** The words are already the page's (`style.evidence_grade`), and the number is already stored
  (`FamilyComparison.evidence_better`).
- **It is the only candidate with a distribution-free promise under repeated looks.** That is the package's own
  promise, so the advice can make it too.
- **It is not the regret-optimal rule under the most favourable data model.** On the on-ramp model, the
  expected-loss rule's worst regret at 24 cases is 1.8 points, against the advice's 4.8 (T-C). Over all four families,
  read at every look, it is 5.6 against 12.2 (T-B). That gap is the price of the promise, and it is the main open
  question for the owner.

**Read off the path, like the decision.** The advice reads whether the evidence has *reached* the bar at some count
so far, in the committed case order, not only where it stands now. This is how "decided" already works: a decision
stands from where the capital reached 1/α. Two conditions withdraw it: v2 is no longer ahead, or the evidence against
v2 has reached the same bar. Ville's bound covers the running maximum, so the promise is unchanged. The gain: a reader
who looks once gets the advice a reader who looked at every case would have got. Read off the current figure alone,
the worst regret at one look is 24.7, 19.1 and 13.8 points; read off the path, 18.5, 12.6 and 7.4 (T-B). The cost:
the advice can say switch while the current evidence has slipped below the bar, as the 100-case gate example shows
(section 4). That is open question 2.

## 3. Honesty under repeated looks

**The promise, in one plain sentence:**

> However often you look, if v2 is not really better, the advice says switch at most one time in three; one time in
> ten when you have said the switch is hard to undo.

It holds for every distribution on the declared range, at every count of cases. The upward mixture the package
already computes (`bet_evidence`) is a nonnegative supermartingale under any mean difference at or below zero, so its
running maximum reaches c with probability at most 1/c (Ville). The advice says switch only on that event. With a
margin, the same holds against "worse by more than the margin". With m arms, the bar is multiplied by m.

What it does not promise: anything about missed gains. Those are the cost, measured below.

**The simulation.**

- **Four data families**, each over paired per-case differences on a 0-1 measure:
  - **pass/fail**: one trial a case; the control passes at U(0.2, 0.7), and the arm at that plus the effect;
  - **on-ramp**: the on-ramp harness's own model (`onramp/reading.py`, `simulated_rows`): k = 3 trials an arm, a base
    share of U(0.1, 0.6), and an arm-by-app spread of 0.1;
  - **full swing**: every case a full win or a full loss, the most spread a difference on the range can have;
  - **skewed**: a third of the range gained on most cases and the whole range lost on the rest, a graded and
    asymmetric shape.
- **Effects** from -0.3 to +0.3 in steps of 0.025, 4,000 replicates a cell, seeds 1001 onward, one per cell. The
  realised mean effect is printed beside each nominal one.
- **Two protocols.**
  - *One look at n*: the advice as it reads at n cases.
  - *Every look to n*: the reader looks after every case from 2 to n and acts on the first "switch". This is the
    adversarial peeker; for a rule that is not anytime-valid, it is the worst case.
- **Regret** is in points, at no switching cost unless stated.

**T-D. Advised to switch to an arm that is no better** (the most over every effect at or below 0, every look to 100
cases; rate [Clopper-Pearson 95%])

| rule | pass/fail | on-ramp | full swing | skewed |
|---|---|---|---|---|
| today: act only when decided | 0.026 [0.022, 0.032] | 0.024 [0.020, 0.030] | 0.036 [0.031, 0.043] | 0.033 [0.028, 0.039] |
| **advice, easy to undo (bar 3)** | 0.215 [0.202, 0.228] | 0.257 [0.244, 0.271] | 0.233 [0.220, 0.246] | 0.272 [0.259, 0.286] |
| **advice, hard to undo (bar 10)** | 0.062 [0.054, 0.070] | 0.066 [0.058, 0.074] | 0.072 [0.065, 0.081] | 0.074 [0.066, 0.083] |
| always switch to the leader | 0.868 [0.857, 0.878] | 0.912 [0.903, 0.921] | 0.906 [0.897, 0.915] | 0.924 [0.916, 0.932] |
| P(better) ≥ 0.9 (DP, a = 1) | 0.470 [0.455, 0.486] | 0.474 [0.459, 0.490] | 0.538 [0.523, 0.554] | 0.617 [0.601, 0.632] |
| expected loss of switching ≤ 1 pt | 0.601 [0.586, 0.616] | 0.736 [0.722, 0.750] | 0.590 [0.575, 0.605] | 0.735 [0.721, 0.749] |

- The worst case for each rule is at an effect of exactly 0, where a switch costs nothing.
- At effects that cost something, the advice is far rarer:
  - **20 points worse**: bar 3 advised switching 0.03 / 0.00 / 0.10 / 0.05 of the time (pass/fail / on-ramp / full
    swing / skewed); bar 10, 0.00 / 0.00 / 0.02 / 0.00; P ≥ 0.9, 0.10 / 0.02 / 0.25 / 0.23; the leader, 0.35 / 0.24 /
    0.59 / 0.53.
  - **10 points worse**: bar 3, 0.08 / 0.02 / 0.15 / 0.11; bar 10, 0.01 / 0.00 / 0.03 / 0.02.
- Every Clopper-Pearson upper bound for the advice is below its promise: 0.286 against 1/3, and 0.083 against 1/10.

**T-B. Worst expected regret in points**, over all four families and every effect from -0.3 to +0.3. Each cell is the
worst from wrong switches / the worst from missed gains.

| rule | n=8, one look | n=8, every look | n=24, one look | n=24, every look | n=100, one look | n=100, every look |
|---|---|---|---|---|---|---|
| today: act only when decided | 0.1 / 30.0 | 0.1 / 30.0 | 0.1 / 23.1 | 0.1 / 23.1 | 0.1 / 13.9 | 0.1 / 13.9 |
| **advice, easy to undo (bar 3, the path)** | 1.4 / 18.5 | 1.9 / 18.1 | 0.9 / 12.6 | 2.0 / 12.2 | 0.5 / 7.4 | 2.0 / 7.2 |
| **advice, hard to undo (bar 10, the path)** | 0.3 / 29.9 | 0.3 / 29.9 | 0.2 / 19.2 | 0.4 / 19.1 | 0.1 / 11.7 | 0.4 / 11.6 |
| bar 3 on the current figure only | 0.4 / 24.7 | 1.9 / 18.1 | 0.1 / 19.1 | 2.0 / 12.2 | 0.0 / 13.8 | 2.0 / 7.2 |
| always switch to the leader | 3.9 / 8.7 | 10.8 / 4.0 | 2.7 / 4.3 | 12.7 / 1.3 | 1.6 / 1.9 | 13.0 / 0.3 |
| P(better) ≥ 0.9 (DP, a = 1) | 1.2 / 17.1 | 4.7 / 11.8 | 0.3 / 14.3 | 5.1 / 5.9 | 0.2 / 6.4 | 5.1 / 2.3 |
| P(better) ≥ 0.8 (DP, a = 1) | 2.3 / 17.1 | 8.6 / 8.5 | 0.8 / 10.1 | 8.7 / 3.5 | 0.4 / 4.4 | 8.7 / 1.1 |
| expected loss of switching ≤ 1 pt | 1.2 / 17.1 | 4.7 / 11.8 | 0.8 / 10.1 | 5.4 / 5.6 | 0.4 / 4.4 | 5.5 / 1.7 |
| worst plausible case ≥ -10 pts | 0.0 / 29.9 | 0.3 / 29.9 | 0.1 / 23.6 | 0.4 / 18.5 | 0.5 / 10.9 | 1.0 / 8.1 |
| evidence ≥ 2 (a lower bar, for scale) | 0.7 / 24.7 | 4.4 / 13.7 | 0.2 / 19.1 | 4.5 / 8.7 | 0.0 / 12.2 | 4.5 / 5.2 |

- With a switching cost of 5 points, the ranking holds: the advice is 1.7 / 15.4, 1.3 / 10.4 and 1.1 / 5.3 at one look,
  and today is 25.0, 19.2 and 10.6 (`tables_4000.md`).
- The worst-plausible-case rules at 0 and 5 points, the margin-evidence rules and the leader at every look are in
  `tables_4000.md`.

**T-C. Decision quality by effect, every look to n, on the on-ramp model.** Each cell is the share advised to switch /
the expected regret in points. The realised effect is in brackets.

| n | rule | -0.2 [-0.18] | -0.1 [-0.10] | 0 | +0.1 | +0.2 | +0.3 | worst regret, -0.3..+0.3 |
|---|---|---|---|---|---|---|---|---|
| 8 | today | 0.00 / 0.0 | 0.00 / 0.0 | 0.00 / 0.0 | 0.00 / 9.9 | 0.00 / 20.0 | 0.00 / 30.0 | 30.0 |
| 8 | **advice, easy** | 0.00 / 0.0 | 0.01 / 0.0 | 0.04 / 0.0 | 0.15 / 8.4 | 0.36 / 12.8 | 0.62 / 11.4 | 12.9 |
| 8 | **advice, hard** | 0.00 / 0.0 | 0.00 / 0.0 | 0.00 / 0.0 | 0.00 / 9.9 | 0.01 / 19.7 | 0.05 / 28.5 | 28.5 |
| 8 | the leader | 0.22 / 4.0 | 0.42 / 4.1 | 0.67 / 0.1 | 0.86 / 1.4 | 0.96 / 0.8 | 0.99 / 0.3 | 4.4 |
| 8 | P(better) ≥ 0.9 | 0.02 / 0.4 | 0.08 / 0.8 | 0.23 / 0.0 | 0.45 / 5.5 | 0.69 / 6.2 | 0.86 / 4.1 | 6.5 |
| 8 | expected loss ≤ 1 pt | 0.04 / 0.8 | 0.13 / 1.3 | 0.32 / 0.0 | 0.55 / 4.5 | 0.76 / 4.8 | 0.90 / 2.9 | 5.2 |
| 24 | today | 0.00 / 0.0 | 0.00 / 0.0 | 0.00 / 0.0 | 0.05 / 9.4 | 0.32 / 13.5 | 0.75 / 7.5 | 13.8 |
| 24 | **advice, easy** | 0.00 / 0.0 | 0.02 / 0.2 | 0.15 / 0.0 | 0.52 / 4.8 | 0.87 / 2.5 | 0.99 / 0.4 | 4.8 |
| 24 | **advice, hard** | 0.00 / 0.0 | 0.00 / 0.0 | 0.01 / 0.0 | 0.16 / 8.3 | 0.56 / 8.9 | 0.90 / 3.1 | 9.9 |
| 24 | the leader | 0.24 / 4.3 | 0.49 / 4.8 | 0.83 / 0.1 | 0.97 / 0.3 | 1.00 / 0.0 | 1.00 / 0.0 | 5.0 |
| 24 | P(better) ≥ 0.9 | 0.02 / 0.4 | 0.10 / 1.0 | 0.36 / 0.0 | 0.71 / 2.9 | 0.94 / 1.1 | 1.00 / 0.1 | 2.9 |
| 24 | expected loss ≤ 1 pt | 0.05 / 0.9 | 0.18 / 1.7 | 0.52 / 0.1 | 0.84 / 1.6 | 0.98 / 0.4 | 1.00 / 0.0 | 1.8 |
| 100 | today | 0.00 / 0.0 | 0.00 / 0.0 | 0.02 / 0.0 | 0.57 / 4.2 | 0.99 / 0.1 | 1.00 / 0.0 | 4.8 |
| 100 | **advice, easy** | 0.00 / 0.0 | 0.02 / 0.2 | 0.26 / 0.0 | 0.88 / 1.2 | 1.00 / 0.0 | 1.00 / 0.0 | 2.0 |
| 100 | **advice, hard** | 0.00 / 0.0 | 0.00 / 0.0 | 0.07 / 0.0 | 0.71 / 2.8 | 1.00 / 0.1 | 1.00 / 0.0 | 3.8 |
| 100 | the leader | 0.24 / 4.3 | 0.52 / 5.0 | 0.91 / 0.1 | 1.00 / 0.0 | 1.00 / 0.0 | 1.00 / 0.0 | 5.1 |
| 100 | P(better) ≥ 0.9 | 0.02 / 0.4 | 0.10 / 1.0 | 0.47 / 0.1 | 0.96 / 0.4 | 1.00 / 0.0 | 1.00 / 0.0 | 1.1 |
| 100 | expected loss ≤ 1 pt | 0.05 / 0.9 | 0.19 / 1.9 | 0.74 / 0.1 | 1.00 / 0.0 | 1.00 / 0.0 | 1.00 / 0.0 | 2.1 |

On the on-ramp model, a gentle one, the Bayesian rules look best. Their trouble is on the other families: in the full
swing family at 24 cases, read at every look, P ≥ 0.9 advised switching to an arm 20 points worse 24% of the time,
and the expected-loss rule 24% (`compact_every_look.md`). The pass/fail, full-swing and skewed versions of this table
are in `.prawduct/artifacts/decide-scratch/compact_every_look.md`.

**T-E. Switch, keep for now, or keep: what the advice says at one look** (on-ramp model; switch / ahead but below the
bar / level or behind)

| n | bar | -0.2 | -0.1 | 0 | +0.1 | +0.2 | +0.3 |
|---|---|---|---|---|---|---|---|
| 8 | 3 | 0.00 / 0.05 / 0.95 | 0.01 / 0.19 / 0.81 | 0.04 / 0.42 / 0.54 | 0.15 / 0.58 / 0.27 | 0.36 / 0.55 / 0.09 | 0.62 / 0.36 / 0.03 |
| 24 | 3 | 0.00 / 0.01 / 0.99 | 0.01 / 0.08 / 0.91 | 0.15 / 0.35 / 0.49 | 0.51 / 0.37 / 0.11 | 0.87 / 0.12 / 0.01 | 0.99 / 0.01 / 0.00 |
| 100 | 3 | 0.00 / 0.00 / 1.00 | 0.00 / 0.00 / 1.00 | 0.22 / 0.29 / 0.49 | 0.88 / 0.11 / 0.01 | 1.00 / 0.00 / 0.00 | 1.00 / 0.00 / 0.00 |
| 8 | 10 | 0.00 / 0.05 / 0.95 | 0.00 / 0.19 / 0.81 | 0.00 / 0.46 / 0.54 | 0.00 / 0.73 / 0.27 | 0.01 / 0.90 / 0.09 | 0.05 / 0.93 / 0.02 |
| 24 | 10 | 0.00 / 0.01 / 0.99 | 0.00 / 0.09 / 0.91 | 0.01 / 0.49 / 0.49 | 0.16 / 0.73 / 0.11 | 0.56 / 0.44 / 0.01 | 0.90 / 0.10 / 0.00 |
| 100 | 10 | 0.00 / 0.00 / 1.00 | 0.00 / 0.00 / 1.00 | 0.06 / 0.45 / 0.49 | 0.71 / 0.28 / 0.01 | 1.00 / 0.00 / 0.00 | 1.00 / 0.00 / 0.00 |

"Ahead but below the bar" is said as "Keep v1 for now", with the one caveat that says what the bar needs. It is a
clear "don't switch yet", never "not enough to say". At 8 cases it is the commonest reading for any effect between 0
and +0.2. That is the honest limit of 8 cases under a promise that survives peeking, and it is the case where adding
cases pays most. The pass/fail version is in `tables_4000.md`.

## 4. Read at the sizes that matter

The named cases go through the package's own functions (`bet_evidence`, `bounded_separation_p` with the 0.9 tilt,
`bet_decided_at`, and `bounded_difference_interval(..., mixture=True, current=True)` at 90%). The path figures come
from the checked restatement. Each drawn case is the first seed, from the one named, whose estimate lies within 1.5
points of its true effect (`named.txt`, `named_path.txt`).

| case | estimate | 90% interval | evidence for v2: now (most reached) | against | proof today | advice, easy | advice, hard |
|---|---|---|---|---|---|---|---|
| all wins, 8 cases, pass/fail | +100 | +21 to +100 | 73.8 (73.8) | 0.02 | decided better, at 6 cases (p 0.015) | **switch** | **switch** |
| +0.3, 8 cases, k = 3 (v1 8 of 24 trials, v2 15) | +29 | -25 to +67 | 3.3 (3.5) | 0.18 | not decided (p 0.31) | **switch** | **keep** |
| on-ramp held-out-1, 24 apps | +22 | -1 to +45 | 17.6 (17.6) | 0.05 | not decided (p 0.063) | **switch** | **switch** |
| no effect, 24 cases (seed 24) | 0 | -21 to +25 | 0.48 (0.96) | 0.38 | not decided | **keep** | **keep** |
| small regression -0.05, 100 cases (seed 107) | -5 | -14 to +4 | 0.04 (0.95) | 1.7 | not decided either way | **keep** | **keep** |
| gate: no worse than 5 points, 40 cases (seed 41) | -1 | -16 to +14 | within 5 points: 0.82 (1.9) | 0.45 | not decided; the default gate reads `undecided` | **keep** | **keep** |
| gate: no worse than 5 points, 100 cases (seed 140) | +1 | -10 to +13 | within 5 points: 1.2 (4.4, at case 13) | 0.09 | not decided; the default gate reads `undecided` | **switch** (path) / keep (current) | **keep** |

The advice says "switch" in four of these and "keep" in four, at the same bars. It never says "not enough to say".

**Draft first screens.** These are the exact sentences, each line from one owner (section 8).

All wins, 8 cases:

> **Advice: switch to v2.** It passed all 8 cases and v1 passed none; at worst it is 21 points ahead.
>
> **Proven: yes.** v2 is better, decided at 6 cases when the evidence reached 24.

+0.3 at 8 cases, easy to undo (the default):

> **Advice: switch to v2.** Over 8 cases it is 29 points ahead of v1; it could still be up to 25 points behind, so
> keep adding cases.
>
> **Proven: not yet.** The evidence that v2 is better is moderate (3.3); a claim, a gate or a promotion needs 22.

+0.3 at 8 cases, hard to undo:

> **Advice: keep v1 for now.** Over 8 cases v2 is 29 points ahead, but a switch that is hard to undo needs strong
> evidence (10), and this is moderate (3.3).
>
> **Proven: not yet.** The evidence that v2 is better is moderate (3.3); a claim, a gate or a promotion needs 22.

The on-ramp result: section 1.

No effect, 24 cases:

> **Advice: keep v1.** Over 24 cases v2 is level with it.
>
> **Proven: no difference shown.** The evidence that v2 is better is weak (0.5).

Small regression, 100 cases:

> **Advice: keep v1.** Over 100 cases v2 is 5 points behind it.
>
> **Proven: not shown worse.** The evidence that v2 is worse is weak (1.7).

Gate (no worse than 5 points), 40 cases:

> **Advice: keep v1.** Over 40 cases v2 is level with it (1 point behind), but it could be up to 16 points worse, past
> the 5 you allow.
>
> **Proven: not yet within 5 points.** The evidence that v2 is within 5 points is weak (0.8); the gate reads undecided,
> which is not a pass.

Gate, 100 cases, read off the path:

> **Advice: switch to v2.** Over 100 cases it is level with v1 (1 point ahead), and the evidence that it is within your
> 5 points reached moderate (4.4); it could still be up to 10 points worse.
>
> **Proven: not yet within 5 points.** The evidence that v2 is within 5 points is weak (1.2) now; the gate reads
> undecided, which is not a pass.

This last case is the one where reading the path and reading the current figure disagree. Section 3 shows why the path
is worth it in regret. It is open question 2, because a reader sees a moderate figure in the advice and a weak one in
the proof.

**When "not enough to say" appears.** In the core design, never. With a control in place, a tie keeps it ("Keep v1:
level"), and that is a clear answer: it costs nothing and is right whenever v2 is not better. The only case where the
costs of the two mistakes are truly symmetric is a choice between arms none of which is in place. There the right rule
is "pick the leader", and "Either will do: they are level" is said only when the estimate is level to the point
printed. That needs a declaration the package does not have (open question 4).

## 5. No hedging creep: one caveat, chosen by what matters

The advice line is three parts: the action, the fact behind it, and at most one caveat. The caveat is picked by the
first rule below that applies, and no other is added. Words like "may", "could possibly" and "it is possible that"
never appear in it.

| Advice | Situation | The one caveat | Example |
|---|---|---|---|
| Switch | the interval's low end is below zero (not proven) | the worst plausible case, and what settles it | "it could still be up to 25 points behind, so keep adding cases" |
| Switch | the interval's low end is at or above zero | none: the low end is said as a strength | "at worst it is 21 points ahead" |
| Keep | the arm is ahead but below the bar | what the bar needs | "a switch that is hard to undo needs strong evidence, and this is moderate (3.3)" |
| Keep | the arm is level or behind | none | "v2 is 5 points behind it over 100 cases" |
| Keep (margin question) | the worst plausible case is past the margin | the worst case against the margin | "it could be up to 16 points worse, past the 5 you allow" |

The scope ("on these 24 apps", "over 8 cases") is part of the fact clause, never a separate caveat. The model or the
data set is named only when the comparison declares one (the on-ramp's plan names its model tier), and then in the
fact clause too. Everything else the reader might want (the interval in full, the per-case tallies, the disclosure)
stays below the first two lines, where it is today.

The proof line has no caveat at all. It says "yes", "not yet" or "shown worse", the evidence figure in the words the
page already uses, and the bar a claim needs.

## 6. Inputs: what a user declares, and the defaults

**One new input: how easy the switch is to undo.** `Eval(..., undo="easy")` is the default; `undo="hard"` raises the
bar from moderate (3) to strong (10). Nothing else is new.

- **No switching cost, no prior, no loss function.** The two bars stand in for the cost: "easy" covers a prompt, a
  model, a docs build, a config flag, anything a revert undoes; "hard" covers a migration, a published claim, a
  customer-facing change that cannot be quietly rolled back. A non-expert can answer "is this easy to undo?"; few can
  answer "what is your loss function?". The simulations (section 3, cost 5 points) show the ranking of the rules does
  not move with a modest switching cost.
- **The margin is the one the package already has.** A question with `margin=` (a non-inferiority question, a gate)
  reads the advice on the evidence that the arm is no worse than the margin below the control, at the same bars. A
  question with no margin reads it against zero: "better".
- **Why the default is "easy".** Most eval-driven changes are reverts away, and a "hard" default would bring back
  exactly the tool the owner describes: at 8 cases it says switch on +0.3 about 5% of the time (T-C), barely above today's 0%.
- **Declared before the results, like the question.** A bar chosen after the evidence is seen is a second look: moving
  from 10 to 3 because the evidence came in at 4 breaks the one-in-three promise. So `undo` is stored with the
  question; a late change (through `Result.against(..., undo=...)`) is read, and the page says once that it was set
  after the results, as `LATE_QUESTION_SENTENCE` does for a late question. (Grünwald 2024 shows e-values tolerate a
  level chosen after the data in a weaker, expected-risk sense; the design does not lean on that.)
- **Several arms.** With m arms against one control, the bar is m times higher (3m, 10m): the mean of m e-values is an
  e-value, so the chance any no-better arm is advised stays at most one in three (one in ten). It mirrors the proof's
  Holm across arms. When several arms clear the bar, the advice names the one with the largest estimate.
- **No incumbent ("either").** The core design always has a control, the incumbent, and a tie goes to it: "keep". That
  is a clear answer, so the core never says "not enough to say". The only place "either" is earned is a choice
  between arms none of which is in place, where the costs of the two mistakes are equal: there the Bayes rule is "pick
  the leader" whatever the evidence, and "Either will do: they are level" is said only when the estimate is level to
  the point printed. This needs a declaration (`undo="none"`, say) and is an open question, not part of the core.

**How an agent consumes it.**

- **Typed, beside the verdict.** `Result.contrasts()` rows gain `advice` (`"switch"` or `"keep"`, on the question's
  reading only) and `advice_words` (the sentence). An agent branches on `advice` as it branches on `outcome` today,
  and never parses words.
- **The console and the page.** The advice line is the first line of the question's reading; the proof line second.
- **The gate.** Unchanged (section 7). `GateResult.render()` may print the advice line under its outcome as
  information; it never changes the outcome or the exit code.
- **The door.** The same two fields on the contrast rows the door serves.

## 7. Gates and CI

**Recommendation: `gate()` stays proof-based. No recommendation-based gate token, for now.**

- **A CI gate is a repeated look by design.** Every push re-runs it, often on the same comparison continued. A gate
  that passed a candidate at the "easy" bar would let a no-better candidate through up to one time in three, and the
  gate's whole value is that its pass means something to the next reader. The proof's α of 5% (and the decision that
  stands once made) is the right contract there.
- **A gate that failed on "keep" would block level changes.** In the 40-case gate example (section 4), a
  refactor that is truly level reads "keep": no anytime-valid rule shows "within 5 points" on that spread so soon.
  A gate that failed there would block a harmless change until enough cases arrived, and on a re-run it could flip.
- **The trade-off.** A proof-based gate misses some real gains and passes some changes it cannot clear: on a margin
  question today, a contrast that is not separated leaves the gate `undecided` (exit 0, printed as not a pass), while
  the advice says "keep". That difference is honest (two questions), but a host that wants the stricter behaviour
  already has `fail_on=("regressed", "not-separated")`. A recommendation token can be added later if a host asks; it
  would carry its one-in-three rate in its name and docs.

## 8. What changes in code (sketch)

**The rule and the sentences, each with one owner.**

- `analysis/stats.py`: `ADVICE_BARS = {"easy": EVIDENCE_STEPS[0], "hard": EVIDENCE_STEPS[1]}`, so the bars are the
  evidence words' own steps and cannot drift from them. One new public function, `no_worse_evidence(sample_a,
  sample_b, *, value_range, margin)`: the upward mixture (`_mixture_log_path`) against `-margin` instead of zero, the
  same construction `bet_evidence` reads at zero (with `margin=0` it returns `bet_evidence`'s upward figure exactly).
- `analysis/bundle/comparisons.py`: `advice_of(comparison, *, undo, arms)`, a pure function of stored fields
  (`evidence_better`, `evidence_worse`, the new `evidence_no_worse` where a margin is declared, `delta`, `interval`)
  returning `"switch"` or `"keep"`. It is derived on read, never stored, so it cannot disagree with the evidence it
  reads.
- `analysis/report/words.py`: `advice_sentence(...)` and `proof_sentence(...)`, the only writers of the two lines,
  beside `standing_words`, `estimate_sentence` and `evidence_sentence`, which they reuse. `GLOSSARY` gains "advice";
  the lead stops saying "undecided" (`_STANDING_WORDS`), which moves to the proof line as "not yet".

**Stored-shape additions, with their reach-back (rule 11).**

| Field | Where | Written | A build from before this change | Rollout and rollback |
|---|---|---|---|---|
| `undo: Literal["hard"] \| None` | `kernel/declaration.py` `Aspect`, beside `margin` | only when `"hard"` (`exclude_if` None); "easy" is the absence | reads every campaign that declares nothing (the default); **refuses** one declaring `undo` by name (`extra_forbidden`: checked by loading an `Aspect` with `undo` into this base's model) | upgrade every reader before a writer declares `hard`; after a rollback, such a campaign is unreadable until re-declared without it |
| `evidence_no_worse: float \| None` | `analysis/bundle/schema.py` `FamilyComparison`, beside `evidence_better` | only where a margin is declared and the bet read the contrast | refuses a stored analysis carrying it, by name (the same `extra_forbidden` the `evidence_better` addition met, MIGRATIONS 2026-11 entry) | readers first; regenerate such an analysis after a rollback. No engine bundle carries it, so no engine fingerprint moves |

No other stored field: `advice` and `advice_words` are served (on `contrasts()` rows and the door's rows), derived on
read. A served-shape model that gained them would refuse them in an older client that validates with `extra="forbid"`;
the rows are plain dicts today, so nothing refuses.

**Certificate claims** (seeded, on the real `advice_of` path, Clopper-Pearson bounds, failing for a wrong formula):

1. `advice.no-better.easy`: over the four null families of section 3 (pass/fail, the on-ramp model, the full-swing
   two-point, the skewed graded) at effect 0 and every effect down to -0.3, every look from 2 to 100 cases, the share
   ever advised "switch" has a CP upper bound at most 1/3. Fails for a bar below 3, or an e-value that is not one (a
   stake grid whose weights do not sum to 1).
2. `advice.no-better.hard`: the same at most 1/10.
3. `advice.margin`: with a margin m, effects at or below -m, at most 1/3 (easy) and 1/10 (hard).
4. `advice.arms`: three arms, all no better, bar 3 x 3: at most 1/3.

The regret figures of section 3 are description, not promises, and get no claim.

**Tests that would move** (to report, not move, in the build chunk): the first-line goldens of every quick-path
comparison with a question (`test_hot_path_oracle.py`, `test_presentation_one_reading.py`), and the on-ramp
reading's printed line on the harness branch. No statistical pin moves: no test, interval or decision changes.

## 9. Literature

Only works I am sure of are cited.

- **Treatment choice by regret, and "follow the leader" at one look.** Manski, C. F. (2004), "Statistical treatment
  rules for heterogeneous populations", *Econometrica* 72(4). Stoye, J. (2009), "Minimax regret treatment choice with
  finite samples", *Journal of Econometrics* 151(1). Manski, C. F. and Tetenov, A. (2016), "Sufficient trial size to
  inform clinical practice", *PNAS* 113(38). The empirical success rule (pick the better sample mean) is near
  minimax-regret for a single look, and hypothesis tests at conventional levels are far from it: T-B's one-look
  column reproduces both.
- **Decide on expected value, not significance.** Claxton, K. (1999), "The irrelevance of inference: a
  decision-making approach to the stochastic evaluation of health care technologies", *Journal of Health Economics*
  18(3).
- **Profit-maximising A/B tests.** Feit, E. M. and Berman, R. (2019), "Test & Roll: profit-maximizing A/B tests",
  *Marketing Science* 38(6): small tests and "deploy the better posterior mean" when the test is one look.
- **Expected-loss stopping in industry A/B testing.** Stucchio, C. (2015), "Bayesian A/B testing at VWO" (VWO
  whitepaper): stop when the expected loss of choosing the leader falls below a threshold of caring. Rule "expected
  loss of switching <= 1 pt" in section 3.
- **Bayesian posteriors under optional stopping.** Rouder, J. N. (2014), "Optional stopping: no problem for
  Bayesians", *Psychonomic Bulletin & Review* 21(2). Deng, A., Lu, J. and Chen, S. (2016), "Continuous monitoring of
  A/B tests without pain: optional stopping in Bayesian testing", IEEE DSAA. Both: calibration holds when effects are
  drawn from the prior; section 3 shows what happens when the data's spread is not what the model assumes.
- **Priors used.** Ferguson, T. S. (1973), "A Bayesian analysis of some nonparametric problems", *Annals of
  Statistics* 1(2) (the Dirichlet process). Rubin, D. B. (1981), "The Bayesian bootstrap", *Annals of Statistics*
  9(1). Spiegelhalter, D. J., Freedman, L. S. and Parmar, M. K. B. (1994), "Bayesian approaches to randomized trials",
  *JRSS-A* 157(3) (sceptical priors).
- **Always-valid A/B testing in industry.** Johari, R., Koomen, P., Pekelis, L. and Walsh, D. (2022), "Always valid
  inference: continuous monitoring of A/B tests", *Operations Research* 70(3): the same family of guarantees as the
  package's bet, at production scale.
- **E-values, the bound used, and the level chosen by the stakes.** Ville, J. (1939), *Étude critique de la notion de
  collectif*. Vovk, V. and Wang, R. (2021), "E-values: calibration, combination and applications", *Annals of
  Statistics* 49(3) (averaging e-values across arms). Grünwald, P. D. (2024), "Beyond Neyman-Pearson: e-values enable
  hypothesis testing with a data-driven alpha", *PNAS* 121(39). Waudby-Smith, I. and Ramdas, A. (2024), "Estimating
  means of bounded random variables by betting", *JRSS-B* 86(1) (the package's bet).
- **Clopper-Pearson.** Clopper, C. J. and Pearson, E. S. (1934), *Biometrika* 26(4).

## 10. Open questions for the owner

1. **The promise or the regret?** The advice keeps a distribution-free promise under peeking: one in three, one in
   ten. The expected-loss rule (Stucchio's) has lower worst regret over these families: 5.6 against 12.2 points at 24
   cases, read at every look. But it advised switching to a no-better arm 59 to 74% of the time, and to one 20 points
   worse up to 26%, with a promise only under its prior. The recommendation is the advice as designed. If you weigh
   usefulness above the promise, the alternative is the expected-loss rule, printed as "the expected cost of switching
   is under 1 point".
2. **Path or current figure?** Reading whether the evidence has reached the bar, as "decided" does, cuts the one-look
   worst regret from 24.7, 19.1 and 13.8 points to 18.5, 12.6 and 7.4, under the same promise. The cost is a screen
   where the advice cites a moderate figure the proof line no longer shows (the 100-case gate example). The
   recommendation is the path.
3. **The bar for an easy switch: 3, or lower?** Bar 2 cuts the worst missed gain at 24 cases from 12.2 to 8.7 points.
   It raises the worst wrong-switch regret from 2.0 to 4.5 points, makes the promise "one in two", and adds a step
   the evidence words do not have. The recommendation is 3, the existing "moderate".
4. **"Either" for a choice with no incumbent?** That needs a declaration (`undo="none"`) and a third advice value. The
   recommendation is to leave it out until a host asks.
5. **A margin question's default gate.** Today, a contrast on a non-inferiority question that is not separated leaves
   `gate()` `undecided` and exit 0, even when the interval reaches 16 points past the margin. The advice there says
   "keep". Should the default `fail_on` treat an undecided non-inferiority contrast as it treats an undecided
   guardrail (fail)? Outside this design, but the advice makes the gap visible.
6. **The proof line's bar.** The design prints "needs 22" for the upward figure, the evidence at which the tilted
   capital reaches 20. Printing the tilted capital itself instead ("18 of 20") would make the two match exactly. But
   the evidence word would then read a figure that is not the e-value, so the design keeps the e-value and the derived
   bar.

## Appendix: how the numbers were made

- **Code.** `.prawduct/artifacts/decide-scratch/core.py` restates the package's mixture bet over the counts of each lattice value the
  paired differences take (multiples of 1/3 on [-1, 1]). Its stakes are constant, so every current-capital quantity
  depends on the counts alone. `check_core.py` checks it, on 300 seeded random samples of 2 to 30 cases, against
  bp-decide's `analysis/stats.py`:
  - `bet_evidence`: worst relative gap 1.6e-15;
  - `bounded_difference_interval(..., confidence=0.9, mixture=True, current=True)`, low end: worst absolute gap
    4.4e-16;
  - `bet_decided_at(..., toward=0.9)`: the same decision point and side on 300 of 300.
- **Simulation.** `sim.py` (4,000 replicates a cell, seed 1000 + the cell's index), its tables from `tables.py`
  (`tables_4000.md`), and `compact_every_look.md`, the every-look extract. The named cases are `named.py`
  (`named.txt`, with `--path`: `named_path.txt`). Run with `PYTHONPATH=packages/evals/src` and the workspace's
  Python 3.14 and numpy; no pool, no model, no network.
- **Approximations.** The simulated P(better) and expected loss use the normal approximation to the Dirichlet-process
  posterior's exact first two moments. T-A's figures are 40,000-draw Monte Carlo.
- **Not run.** No test, lint or typecheck: no product or test file changed.
