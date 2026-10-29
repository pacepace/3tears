Choice · candidates compared · Research query planner · 18 arms · 4 tasks · Run 2026-10-23 · Synthetic example data

# Use Brisk-S · reasoning low · 1 retry (J): the cheapest configuration to clear the 97% pass-rate gate, though only by 0.1 points.

It costs $0.302 per 1k calls, least of the five that clear, but a fresh run of today's size reads it below the gate 38.6% of the time. If the gate must clear with confidence, high reasoning (L) is the pick, at $0.633 per 1k calls.

**Confidence: Low.**

- **Pass rate is cleared on a point estimate.** If J's true pass rate sits below the gate, which its clearance of 0.1 points cannot rule out, then L is the cheapest configuration that clears; settling J four times in five would take 111,255 cases, two-sided.
- **The pick is the best of several candidates.** J's figure is partly the luck of being picked from 18 configurations: a run on fresh cases tends to read it lower, which pushes it toward or under the gate and leaves L, with 1 points of room, as the pick.
- **No arm is the control.** Only for a claim that one configuration beats another on pass rate: every gap here is read unpaired and so wider than a paired reading; the call rests on each configuration's own gate reading and its cost.
- **Cost routing unknown.** Only if another provider's pricing for Brisk-S or Atlas-M reorders them: the snapshot names the model each arm served but no provider, so every cost is this run's routing's, not the model's own.

**Recommendation.** Ship Brisk-S · reasoning low · 1 retry (J). It reads 97.1% against the 97% gate at $0.302 per 1k calls, the least of any configuration that clears: about $1,208 a month if the declared volume counts calls, more if it counts tasks, by each arm's calls per task.

**Why not the others.** Brisk-S · reasoning high · 1 retry (L) clears with more room, 98%, but costs $0.633 per 1k calls, $1,324 more a month if the volume counts calls. Every Atlas-M that clears costs more again; the best, Atlas-M · reasoning high · 1 retry (F), reads 98.6% at $3.477 per 1k calls. No Nano-XS configuration clears; its best, Nano-XS · reasoning high · 1 retry (S), falls 2.2 points short.

**Watch.** The weakest margin is the pick's own: Brisk-S · reasoning low · 1 retry (J) reads 97.1% on Pass rate against 97%. A fresh run of today's size reads it below the line 38.6% of the time, against 0.9% for L, and as the best of 18 it likely reads high.

**Next.** Ship J now; in parallel the planner's owner declares whether the gate must clear on its interval. If it must, switch to L.

| Option | Role | Configuration | Pass rate | Cost per 1,000 calls |
|---|---|---|---|---|
| A · Atlas-M · reasoning off · no retry | Candidate | Atlas-M · Reasoning budget off · Retry on failure 0 | 94% | $1.000 |
| B · Atlas-M · reasoning off · 1 retry | Candidate | Atlas-M · Reasoning budget off · Retry on failure 1 | 96.4% | $1.060 |
| C · Atlas-M · reasoning low · no retry | Candidate | Atlas-M · Reasoning budget low · Retry on failure 0 | 96.6% | $1.600 |
| D · Atlas-M · reasoning low · 1 retry | Candidate | Atlas-M · Reasoning budget low · Retry on failure 1 | 97.9% | $1.655 |
| E · Atlas-M · reasoning high · no retry | Candidate | Atlas-M · Reasoning budget high · Retry on failure 0 | 97.7% | $3.400 |
| F · Atlas-M · reasoning high · 1 retry | Candidate | Atlas-M · Reasoning budget high · Retry on failure 1 | 98.6% | $3.477 |
| G · Brisk-S · reasoning off · no retry | Candidate | Brisk-S · Reasoning budget off · Retry on failure 0 | 91.1% | $0.180 |
| H · Brisk-S · reasoning off · 1 retry | Candidate | Brisk-S · Reasoning budget off · Retry on failure 1 | 94.6% | $0.196 |
| I · Brisk-S · reasoning low · no retry | Candidate | Brisk-S · Reasoning budget low · Retry on failure 0 | 95.3% | $0.288 |
| J · Brisk-S · reasoning low · 1 retry | Candidate (recommended) | Brisk-S · Reasoning budget low · Retry on failure 1 | 97.1% | $0.302 |
| K · Brisk-S · reasoning high · no retry | Candidate | Brisk-S · Reasoning budget high · Retry on failure 0 | 96.6% | $0.612 |
| L · Brisk-S · reasoning high · 1 retry | Candidate | Brisk-S · Reasoning budget high · Retry on failure 1 | 98% | $0.633 |
| M · Nano-XS · reasoning off · no retry | Candidate | Nano-XS · Reasoning budget off · Retry on failure 0 | 81% | $0.030 |
| N · Nano-XS · reasoning off · 1 retry | Candidate | Nano-XS · Reasoning budget off · Retry on failure 1 | 88.5% | $0.036 |
| O · Nano-XS · reasoning low · no retry | Candidate | Nano-XS · Reasoning budget low · Retry on failure 0 | 87.9% | $0.048 |
| Q · Nano-XS · reasoning low · 1 retry | Candidate | Nano-XS · Reasoning budget low · Retry on failure 1 | 92.7% | $0.054 |
| R · Nano-XS · reasoning high · no retry | Candidate | Nano-XS · Reasoning budget high · Retry on failure 0 | 91.3% | $0.102 |
| S · Nano-XS · reasoning high · 1 retry | Candidate | Nano-XS · Reasoning budget high · Retry on failure 1 | 94.8% | $0.111 |

## Only five of eighteen configurations clear the 97% gate, and J sits closest to the line at 97.1%.

| Row | Value | 95% computed interval |
|---|---|---|
| A · Atlas-M · reasoning off · no retry | 94.0% | 92.7% to 95.3% |
| B · Atlas-M · reasoning off · 1 retry | 96.4% | 95.3% to 97.4% |
| C · Atlas-M · reasoning low · no retry | 96.6% | 95.5% to 97.6% |
| D · Atlas-M · reasoning low · 1 retry | 97.9% | 97.1% to 98.7% |
| E · Atlas-M · reasoning high · no retry | 97.7% | 96.9% to 98.6% |
| F · Atlas-M · reasoning high · 1 retry | 98.6% | 98.0% to 99.3% |
| G · Brisk-S · reasoning off · no retry | 91.1% | 89.5% to 92.7% |
| H · Brisk-S · reasoning off · 1 retry | 94.6% | 93.3% to 95.9% |
| I · Brisk-S · reasoning low · no retry | 95.3% | 94.1% to 96.5% |
| J · Brisk-S · reasoning low · 1 retry | 97.1% | 96.2% to 98.1% |
| K · Brisk-S · reasoning high · no retry | 96.6% | 95.6% to 97.7% |
| L · Brisk-S · reasoning high · 1 retry | 98.0% | 97.2% to 98.8% |
| M · Nano-XS · reasoning off · no retry | 81.0% | 78.8% to 83.3% |
| N · Nano-XS · reasoning off · 1 retry | 88.5% | 86.7% to 90.3% |
| O · Nano-XS · reasoning low · no retry | 87.9% | 86.1% to 89.8% |
| Q · Nano-XS · reasoning low · 1 retry | 92.7% | 91.2% to 94.2% |
| R · Nano-XS · reasoning high · no retry | 91.3% | 89.7% to 92.9% |
| S · Nano-XS · reasoning high · 1 retry | 94.8% | 93.5% to 96.0% |
| Pass rate gate, at least 97% | 97.0% |  |

Each interval is computed here from the rate's binomial spread, by the normal approximation, held to 0 to 100%. A rate of none or all has none.

Each configuration's pass rate with a computed interval, against the gate at 97%; J is stressed.

## Of the configurations that clear, J costs least: $0.302 per 1k calls, against $0.633 for L and $1.655 for the cheapest Atlas-M.

_Caveats: Cost routing unknown._

| Row | Value |
|---|---|
| J · Brisk-S · reasoning low · 1 retry | $0.30 |
| L · Brisk-S · reasoning high · 1 retry | $0.63 |
| D · Atlas-M · reasoning low · 1 retry | $1.66 |
| E · Atlas-M · reasoning high · no retry | $3.40 |
| F · Atlas-M · reasoning high · 1 retry | $3.48 |

The bars are drawn on a log scale: their values span more than ten times, so each step is a multiple.

Cost per 1k calls of the configurations that clear the gate, on a log scale, at this run's provider routing, which the snapshot does not name.

### J's clearance is inside its noise: a fresh run of today's size would read it below the gate 38.6% of the time, against 0.9% for L. A two-sided run that settles it four times in five needs 111,255 cases for J and 1,660 for L.

_Caveats: Pass rate is cleared on a point estimate; The pick is the best of several candidates._

| Arm | Against | Today's figure | Line | Distance toward passing | Settles | Sizes count | Needed, one-sided | Needed, two-sided | Today | A fresh run reads the other side | Level |
|---|---|---|---|---|---|---|---|---|---|---|---|
| J · Brisk-S · reasoning low · 1 retry | Pass rate | 97.1% | 97.0% | +0.1 | clears | cases | 38,350 | 54,451 | 1,200 | 38.6% | 95% one-sided |
| L · Brisk-S · reasoning high · 1 retry | Pass rate | 98.0% | 97.0% | +1.0 | clears | cases | 572 | 812 | 1,200 | 0.9% | 95% one-sided |

In each row, "A fresh run reads the other side" is how often a fresh run of today's size would read the figure on the other side of its line, if the truth is today's estimate (95% one-sided); it plans a run and tests nothing.

Distance to the gate, how often a run of today's size reads the other side, and the cases that settle the line half the time; the four-in-five, two-sided sizes cited above are larger: 111,255 for J and 1,660 for L.

### High reasoning adds +0.8 points over low, an interval of -0.4 to +2.1 that does not exclude zero, for $0.633 against $0.302 per 1k calls.

_Caveats: No arm is the control._

| Row | Value | 95% unpaired interval |
|---|---|---|
| Overall | +0.8 | -0.4 to +2.1 |
| Single lookup | +0.1 |  |
| Multi-hop (2 to 3 calls) | +1.7 |  |
| Date and unit math | +0.9 |  |
| Ambiguous entity | +1.2 |  |

Each row is Brisk-S · reasoning high · 1 retry less Brisk-S · reasoning low · 1 retry on Pass rate, overall and in each place both hold it.

The interval is independent: each arm's own spread (a pass/fail rate's binomial spread, from the arm's cases), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same.

L minus J on pass rate, overall with an unpaired interval; the per-task rows state no case counts, so they carry no interval and are descriptive only.

#### The retry is what lifts J over the gate: it adds +1.9 points to Brisk-S at low reasoning, an interval of +0.3 to +3.4.

| Group | Row | Value | 95% unpaired interval |
|---|---|---|---|
| reasoning | Brisk-S · reasoning off · no retry → Low | +4.2 | +2.2 to +6.2 |
| reasoning | Brisk-S · reasoning off · 1 retry → Low | +2.5 | +0.9 to +4.1 |
| reasoning | Brisk-S · reasoning off · 1 retry → High | +3.3 | +1.8 to +4.9 |
| reasoning | Brisk-S · reasoning low · 1 retry → High | +0.8 | -0.4 to +2.1 |
| retry | Brisk-S · reasoning off · no retry → 1 retry | +3.5 | +1.5 to +5.6 |
| retry | Brisk-S · reasoning low · no retry → 1 retry | +1.9 | +0.3 to +3.4 |
| reasoning effect, across retry | (Brisk-S · reasoning off · 1 retry → Low) less (Brisk-S · reasoning off · no retry → Low) | -1.7 | -4.2 to +0.9 |

Each effect is one setting moved between two arms that differ in it alone, on Pass rate; each interaction is the same move's effect in one context less its effect in another, the two contexts differing in one other setting. Descriptive: read as two arms' own spreads, never a test.

What reasoning and the retry each add to Brisk-S's pass rate along the moves that lead to J and L, with unpaired intervals; the interaction shown is not separated from zero.

## What you could change

| What you could change | What it decides | What happens as it moves |
|---|---|---|
| The pass-rate gate's threshold | Which configuration is the cheapest that clears | J stays the pick for a gate above 95.3% up to 97.1%; L from there up to 98%; Atlas-M · reasoning high · 1 retry (F) up to 98.6%; above that nothing clears. Lower gates admit cheaper Brisk-S and Nano-XS configurations. |
| Reading the gate on the interval rather than the point estimate | Whether J counts as clearing | J's computed interval crosses the gate while L's sits above it, so on the interval L becomes the pick at $0.633 per 1k calls against $0.302; a month that is $1,324 more if the volume counts calls, more if it counts tasks. |

## Next steps

- The planner's owner declares whether the gate must clear on its interval or on its point estimate. _Whether J ships or L does: more cases cannot practically settle J, which would need 111,255 cases to settle four times in five, two-sided._
- Add worked multi-hop examples to Brisk-S's planning prompt, where J reads 95.6% (a figure with no stated case count, so no interval), then re-run J on 1,660 fresh cases. _Whether the change lifts J clear of the gate at its current cost: a run that size settles a clearance as large as L's, 1 points, four times in five; a smaller lift needs more cases._
- Re-run L on 1,660 fresh cases. _Whether L's clearance survives fresh cases. The size assumes today's estimate of 98%; the best-of-many selection likely reads that high, and a lower true rate needs more cases to settle four times in five._
- Declare each task's traffic share and case count, and whether the monthly volume counts calls or tasks. _Whether the overall pass rate reflects the production mix, and what the step from J to L costs a month._

## Full results

### The arms

Each arm's full configuration, once. The rest of the page names an arm by its letter tag and name.

| Arm | Role | Configuration | Served model |
|---|---|---|---|
| A · Atlas-M · reasoning off · no retry | Candidate | Atlas-M · Reasoning budget off · Retry on failure 0 | Atlas-M |
| B · Atlas-M · reasoning off · 1 retry | Candidate | Atlas-M · Reasoning budget off · Retry on failure 1 | Atlas-M |
| C · Atlas-M · reasoning low · no retry | Candidate | Atlas-M · Reasoning budget low · Retry on failure 0 | Atlas-M |
| D · Atlas-M · reasoning low · 1 retry | Candidate | Atlas-M · Reasoning budget low · Retry on failure 1 | Atlas-M |
| E · Atlas-M · reasoning high · no retry | Candidate | Atlas-M · Reasoning budget high · Retry on failure 0 | Atlas-M |
| F · Atlas-M · reasoning high · 1 retry | Candidate | Atlas-M · Reasoning budget high · Retry on failure 1 | Atlas-M |
| G · Brisk-S · reasoning off · no retry | Candidate | Brisk-S · Reasoning budget off · Retry on failure 0 | Brisk-S |
| H · Brisk-S · reasoning off · 1 retry | Candidate | Brisk-S · Reasoning budget off · Retry on failure 1 | Brisk-S |
| I · Brisk-S · reasoning low · no retry | Candidate | Brisk-S · Reasoning budget low · Retry on failure 0 | Brisk-S |
| J · Brisk-S · reasoning low · 1 retry | Candidate | Brisk-S · Reasoning budget low · Retry on failure 1 | Brisk-S |
| K · Brisk-S · reasoning high · no retry | Candidate | Brisk-S · Reasoning budget high · Retry on failure 0 | Brisk-S |
| L · Brisk-S · reasoning high · 1 retry | Candidate | Brisk-S · Reasoning budget high · Retry on failure 1 | Brisk-S |
| M · Nano-XS · reasoning off · no retry | Candidate | Nano-XS · Reasoning budget off · Retry on failure 0 | Nano-XS |
| N · Nano-XS · reasoning off · 1 retry | Candidate | Nano-XS · Reasoning budget off · Retry on failure 1 | Nano-XS |
| O · Nano-XS · reasoning low · no retry | Candidate | Nano-XS · Reasoning budget low · Retry on failure 0 | Nano-XS |
| Q · Nano-XS · reasoning low · 1 retry | Candidate | Nano-XS · Reasoning budget low · Retry on failure 1 | Nano-XS |
| R · Nano-XS · reasoning high · no retry | Candidate | Nano-XS · Reasoning budget high · Retry on failure 0 | Nano-XS |
| S · Nano-XS · reasoning high · 1 retry | Candidate | Nano-XS · Reasoning budget high · Retry on failure 1 | Nano-XS |

### Every arm by measure

| Arm | Pass rate | Calls per task | Cost per 1k calls | Latency |
|---|---|---|---|---|
| A · Atlas-M · reasoning off · no retry | 94% | 1.539 calls | $1.000 | 900 ms (p95) |
| B · Atlas-M · reasoning off · 1 retry | 96.4% | 1.599 calls | $1.060 | 1019 ms (p95) |
| C · Atlas-M · reasoning low · no retry | 96.6% | 1.469 calls | $1.600 | 1395 ms (p95) |
| D · Atlas-M · reasoning low · 1 retry | 97.9% | 1.504 calls | $1.655 | 1500 ms (p95) |
| E · Atlas-M · reasoning high · no retry | 97.7% | 1.43 calls | $3.400 | 2430 ms (p95) |
| F · Atlas-M · reasoning high · 1 retry | 98.6% | 1.452 calls | $3.477 | 2551 ms (p95) |
| G · Brisk-S · reasoning off · no retry | 91.1% | 1.579 calls | $0.180 | 300 ms (p95) |
| H · Brisk-S · reasoning off · 1 retry | 94.6% | 1.668 calls | $0.196 | 359 ms (p95) |
| I · Brisk-S · reasoning low · no retry | 95.3% | 1.492 calls | $0.288 | 465 ms (p95) |
| J · Brisk-S · reasoning low · 1 retry | 97.1% | 1.539 calls | $0.302 | 513 ms (p95) |
| K · Brisk-S · reasoning high · no retry | 96.6% | 1.442 calls | $0.612 | 810 ms (p95) |
| L · Brisk-S · reasoning high · 1 retry | 98% | 1.476 calls | $0.633 | 870 ms (p95) |
| M · Nano-XS · reasoning off · no retry | 81% | 1.698 calls | $0.030 | 140 ms (p95) |
| N · Nano-XS · reasoning off · 1 retry | 88.5% | 1.888 calls | $0.036 | 198 ms (p95) |
| O · Nano-XS · reasoning low · no retry | 87.9% | 1.559 calls | $0.048 | 217 ms (p95) |
| Q · Nano-XS · reasoning low · 1 retry | 92.7% | 1.68 calls | $0.054 | 275 ms (p95) |
| R · Nano-XS · reasoning high · no retry | 91.3% | 1.479 calls | $0.102 | 378 ms (p95) |
| S · Nano-XS · reasoning high · 1 retry | 94.8% | 1.566 calls | $0.111 | 450 ms (p95) |

### Every task

| Task | Cases | A · Atlas-M · reasoning off · no retry | B · Atlas-M · reasoning off · 1 retry | C · Atlas-M · reasoning low · no retry | D · Atlas-M · reasoning low · 1 retry | E · Atlas-M · reasoning high · no retry | F · Atlas-M · reasoning high · 1 retry | G · Brisk-S · reasoning off · no retry | H · Brisk-S · reasoning off · 1 retry | I · Brisk-S · reasoning low · no retry | J · Brisk-S · reasoning low · 1 retry | K · Brisk-S · reasoning high · no retry | L · Brisk-S · reasoning high · 1 retry | M · Nano-XS · reasoning off · no retry | N · Nano-XS · reasoning off · 1 retry | O · Nano-XS · reasoning low · no retry | Q · Nano-XS · reasoning low · 1 retry | R · Nano-XS · reasoning high · no retry | S · Nano-XS · reasoning high · 1 retry |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Single lookup | — | 97.2% | 98.3% | 98.6% | 99.2% | 98.7% | 99.2% | 94.3% | 96.6% | 97.4% | 98.4% | 97.6% | 98.5% | 86.8% | 92% | 91.7% | 95% | 93% | 95.8% |
| Multi-hop (2 to 3 calls) | — | 90.2% | 94.1% | 94.1% | 96.4% | 96.6% | 97.9% | 87.3% | 92.3% | 92.8% | 95.6% | 95.5% | 97.3% | 74.2% | 84.4% | 83.5% | 90% | 89.3% | 93.5% |
| Date and unit math | — | 93.7% | 96.2% | 96.4% | 97.8% | 97.7% | 98.6% | 90.8% | 94.4% | 95.1% | 97% | 96.6% | 97.9% | 80.5% | 88.2% | 87.6% | 92.5% | 91.1% | 94.7% |
| Ambiguous entity | — | 92.2% | 95.3% | 95.4% | 97.2% | 97.2% | 98.3% | 89.3% | 93.5% | 94.1% | 96.4% | 96.1% | 97.6% | 77.8% | 86.6% | 85.8% | 91.4% | 90.3% | 94.2% |

### Every gate, wherever it is read

| Where | Criterion | Line | A · Atlas-M · reasoning off · no retry | B · Atlas-M · reasoning off · 1 retry | C · Atlas-M · reasoning low · no retry | D · Atlas-M · reasoning low · 1 retry | E · Atlas-M · reasoning high · no retry | F · Atlas-M · reasoning high · 1 retry | G · Brisk-S · reasoning off · no retry | H · Brisk-S · reasoning off · 1 retry | I · Brisk-S · reasoning low · no retry | J · Brisk-S · reasoning low · 1 retry | K · Brisk-S · reasoning high · no retry | L · Brisk-S · reasoning high · 1 retry | M · Nano-XS · reasoning off · no retry | N · Nano-XS · reasoning off · 1 retry | O · Nano-XS · reasoning low · no retry | Q · Nano-XS · reasoning low · 1 retry | R · Nano-XS · reasoning high · no retry | S · Nano-XS · reasoning high · 1 retry |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Overall | Pass rate | at least 97% | Fails (94%) | Fails (96.4%) | Fails (96.6%) | Passes (97.9%) | Passes (97.7%) | Passes (98.6%) | Fails (91.1%) | Fails (94.6%) | Fails (95.3%) | Passes (97.1%) | Fails (96.6%) | Passes (98%) | Fails (81%) | Fails (88.5%) | Fails (87.9%) | Fails (92.7%) | Fails (91.3%) | Fails (94.8%) |

### Every arm's configuration

| Arm | Model | Reasoning budget | Retry on failure |
|---|---|---|---|
| A · Atlas-M · reasoning off · no retry | Atlas-M | Off | no retry |
| B · Atlas-M · reasoning off · 1 retry | Atlas-M | Off | 1 retry |
| C · Atlas-M · reasoning low · no retry | Atlas-M | Low | no retry |
| D · Atlas-M · reasoning low · 1 retry | Atlas-M | Low | 1 retry |
| E · Atlas-M · reasoning high · no retry | Atlas-M | High | no retry |
| F · Atlas-M · reasoning high · 1 retry | Atlas-M | High | 1 retry |
| G · Brisk-S · reasoning off · no retry | Brisk-S | Off | no retry |
| H · Brisk-S · reasoning off · 1 retry | Brisk-S | Off | 1 retry |
| I · Brisk-S · reasoning low · no retry | Brisk-S | Low | no retry |
| J · Brisk-S · reasoning low · 1 retry | Brisk-S | Low | 1 retry |
| K · Brisk-S · reasoning high · no retry | Brisk-S | High | no retry |
| L · Brisk-S · reasoning high · 1 retry | Brisk-S | High | 1 retry |
| M · Nano-XS · reasoning off · no retry | Nano-XS | Off | no retry |
| N · Nano-XS · reasoning off · 1 retry | Nano-XS | Off | 1 retry |
| O · Nano-XS · reasoning low · no retry | Nano-XS | Low | no retry |
| Q · Nano-XS · reasoning low · 1 retry | Nano-XS | Low | 1 retry |
| R · Nano-XS · reasoning high · no retry | Nano-XS | High | no retry |
| S · Nano-XS · reasoning high · 1 retry | Nano-XS | High | 1 retry |

### Left out, and why

- **Latency** No latency gate was declared, and latency does not reorder the two Brisk-S options; full figures in the appendix.
- **Calls per task** Varies little between the options a decider would weigh and changes no gate reading; it matters only to what a month costs if the volume counts tasks, which the cost lines say.
- **Extra calls per task and outcome** The snapshot holds no figures for either.
- **Per-task pass rates of the other configurations** The gate reads overall; per-task figures state no case counts, so they are leads, kept in the appendix.
- **Model and reasoning lever effects across Atlas-M and Nano-XS, and the high-reasoning retry move for Brisk-S** They explain why other configurations fall short or cost more but do not change the pick; all are in the appendix.

### Method

- **Cases** Each configuration ran 1,200 cases once each across the task types; none was lost or failed to run. Per-task case counts are not stated.
- **Pass** The input does not define what makes a planner run pass.
- **The gate** Pass rate at least 97%, declared before the run and read on each configuration's point estimate, as declared.
- **Intervals** Computed here from each configuration's cases as a binomial spread by the normal approximation, at the level each block prints. Gaps between configurations are unpaired, since no control was run, and uncorrected for the number of comparisons. Run sizes are quoted two-sided, sized to settle four times in five unless a table says otherwise. _Caveats: No arm is the control._
- **Cost** Cost per 1k calls as measured. Each arm names its model but no provider. Monthly figures use the declared volume of 4,000,000 read as calls; the input does not say whether it counts calls or tasks, and if it counts tasks each monthly figure grows by the arm's calls per task.
- **Selection** The pick is the cheapest of 18 configurations to clear the gate, so its pass rate likely reads high.
- **Source** Synthetic example data from docs/eval-report-kit/references/r2-config-sweep.json, run 2026-10-23.

_Written by claude-opus-5-5, served as claude-opus-5-5, for $2.47, with analyst skill ed7e2c60b5ad, on 2026-10-29._
