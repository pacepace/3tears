Choice · candidates compared · Research query planner · 18 arms · 4 tasks · Run 2026-10-23 · Synthetic example data

# Brisk-S with low reasoning and one retry is the knee: 97.1% pass rate for $1,208.00 a month; each point above costs far more.

Its lead over the best Nano-XS setup is +2.4 points, a gap whose interval excludes zero. Nothing declared a minimum pass rate or a price per point, so whether a higher setup is worth its cost is the planner owner's call.

**Confidence: High, if the caveats below aren't material.**

- **Nothing declares what is good enough.** Material for the pick itself: a declared minimum above 97.1% moves it to Brisk-S with high reasoning or to Atlas-M. The cost curve and its prices are measured and stand either way.
- **No arm is the control.** Only for gaps under about a point, such as high over low reasoning on Brisk-S with retry: every interval here is read as if the setups ran different cases, which errs wide, so a paired reading could separate it. The lean rests on cost, which is measured.
- **The pick is the best of several candidates.** Only if the lean's lead over Nano-XS with high reasoning and retry shrinks on fresh cases; its interval runs +0.8 to +4. Atlas-M with high reasoning and retry, the top reading of 18, is the one most likely to read lower.

**The trade.** Above Brisk-S with low reasoning and one retry, each step buys under a point of pass rate at a steeply rising price. The top step, to Atlas-M with high reasoning and retry, costs $13,908.00 a month for a gain over Brisk-S high that is not separated from zero.

**Where the answer flips.** Valuing a point of pass rate below $319.67 a month picks Nano-XS with high reasoning and retry; up to $1,595.18, Brisk-S low with retry; up to $17,236.36, Brisk-S high with retry; above that, Atlas-M high with retry.

**Our lean.** Ours, not a declared rule: Brisk-S, low reasoning, one retry. It holds while a point of pass rate is worth between the first two of those prices, and its slowest calls stay fast (p95 513 ms).

**Watch.** Brisk-S with high reasoning and retry reads +0.8 points higher, not separated (-0.4 to +2.1); a run of today's size would read that gap the wrong way 9.4% of the time.

**What would settle it.** The planner's owner declares a minimum pass rate or a monthly price per point. With either, the curve names one setup.

| Option | Role | Configuration | Cost per 1,000 calls |
|---|---|---|---|
| A · Atlas-M · reasoning off · no retry | Candidate | Atlas-M · reasoning off · retry 0 | $1.00 |
| B · Atlas-M · reasoning off · 1 retry | Candidate | Atlas-M · reasoning off · retry 1 | $1.06 |
| C · Atlas-M · reasoning low · no retry | Candidate | Atlas-M · reasoning low · retry 0 | $1.60 |
| D · Atlas-M · reasoning low · 1 retry | Candidate | Atlas-M · reasoning low · retry 1 | $1.66 |
| E · Atlas-M · reasoning high · no retry | Candidate | Atlas-M · reasoning high · retry 0 | $3.40 |
| F · Atlas-M · reasoning high · 1 retry | Candidate | Atlas-M · reasoning high · retry 1 | $3.48 |
| G · Brisk-S · reasoning off · no retry | Candidate | Brisk-S · reasoning off · retry 0 | $0.18 |
| H · Brisk-S · reasoning off · 1 retry | Candidate | Brisk-S · reasoning off · retry 1 | $0.196 |
| I · Brisk-S · reasoning low · no retry | Candidate | Brisk-S · reasoning low · retry 0 | $0.288 |
| J · Brisk-S · reasoning low · 1 retry | Candidate | Brisk-S · reasoning low · retry 1 | $0.302 |
| K · Brisk-S · reasoning high · no retry | Candidate | Brisk-S · reasoning high · retry 0 | $0.612 |
| L · Brisk-S · reasoning high · 1 retry | Candidate | Brisk-S · reasoning high · retry 1 | $0.633 |
| M · Nano-XS · reasoning off · no retry | Candidate | Nano-XS · reasoning off · retry 0 | $0.0300 |
| N · Nano-XS · reasoning off · 1 retry | Candidate | Nano-XS · reasoning off · retry 1 | $0.0360 |
| O · Nano-XS · reasoning low · no retry | Candidate | Nano-XS · reasoning low · retry 0 | $0.0480 |
| Q · Nano-XS · reasoning low · 1 retry | Candidate | Nano-XS · reasoning low · retry 1 | $0.0540 |
| R · Nano-XS · reasoning high · no retry | Candidate | Nano-XS · reasoning high · retry 0 | $0.102 |
| S · Nano-XS · reasoning high · 1 retry | Candidate | Nano-XS · reasoning high · retry 1 | $0.111 |

## Brisk-S with low reasoning and one retry sits where the cost curve bends: every setup that passes more costs at least twice as much.

| Point | Cost per 1k calls | Pass rate |
|---|---|---|
| Atlas-M · reasoning off · no retry | $1.000 | 94.0% |
| Atlas-M · reasoning off · 1 retry | $1.060 | 96.4% |
| Atlas-M · reasoning low · no retry | $1.600 | 96.6% |
| Atlas-M · reasoning low · 1 retry | $1.655 | 97.9% |
| Atlas-M · reasoning high · no retry | $3.400 | 97.7% |
| Atlas-M · reasoning high · 1 retry | $3.477 | 98.6% |
| Brisk-S · reasoning off · no retry | $0.180 | 91.1% |
| Brisk-S · reasoning off · 1 retry | $0.196 | 94.6% |
| Brisk-S · reasoning low · no retry | $0.288 | 95.3% |
| J · Brisk-S · reasoning low · 1 retry | $0.302 | 97.1% |
| Brisk-S · reasoning high · no retry | $0.612 | 96.6% |
| Brisk-S · reasoning high · 1 retry | $0.633 | 98.0% |
| Nano-XS · reasoning off · no retry | $0.030 | 81.0% |
| Nano-XS · reasoning off · 1 retry | $0.036 | 88.5% |
| Nano-XS · reasoning low · no retry | $0.048 | 87.9% |
| Nano-XS · reasoning low · 1 retry | $0.054 | 92.7% |
| Nano-XS · reasoning high · no retry | $0.102 | 91.3% |
| Nano-XS · reasoning high · 1 retry | $0.111 | 94.8% |

The horizontal axis is logarithmic: each step is a multiple.

Pass rate against cost per 1,000 calls for all 18 setups; cost on a log scale.

## The step that reaches Brisk-S low with retry costs $319.67 a month per point; the next costs $1,595.18.

| Step along the frontier | Price per point |
|---|---|
| Nano-XS · reasoning off · no retry → Nano-XS · reasoning off · 1 retry | $3.20 |
| Nano-XS · reasoning off · 1 retry → Nano-XS · reasoning low · 1 retry | $17.31 |
| Nano-XS · reasoning low · 1 retry → Nano-XS · reasoning high · 1 retry | $111.2 |
| Nano-XS · reasoning high · 1 retry → Brisk-S · reasoning low · 1 retry | $319.7 |
| Brisk-S · reasoning low · 1 retry → Brisk-S · reasoning high · 1 retry | $1,595 |
| Brisk-S · reasoning high · 1 retry → Atlas-M · reasoning high · 1 retry | $17,236 |

Monthly price of each extra point of pass rate between neighbouring setups on the curve, at the declared monthly volume; point estimates, no interval.

### A retry is the cheapest lever: on every model, low reasoning with one retry reads at least as high as high reasoning without one, at about half the cost. It helps weak setups most, adding +7.5 points to Nano-XS without reasoning but +0.9 to Atlas-M at high reasoning.

|  | Off · no retry | Off · 1 retry | Low · no retry | Low · 1 retry | High · no retry | High · 1 retry |
|---|---|---|---|---|---|---|
| Atlas-M | 94.0% · $1.000 per 1,000 calls · 900 ms | 96.4% · $1.060 per 1,000 calls · 1019 ms | 96.6% · $1.600 per 1,000 calls · 1395 ms | 97.9% · $1.655 per 1,000 calls · 1500 ms | 97.7% · $3.400 per 1,000 calls · 2430 ms | 98.6% · $3.477 per 1,000 calls · 2551 ms |
| Brisk-S | 91.1% · $0.180 per 1,000 calls · 300 ms | 94.6% · $0.196 per 1,000 calls · 359 ms | 95.3% · $0.288 per 1,000 calls · 465 ms | 97.1% · $0.302 per 1,000 calls · 513 ms | 96.6% · $0.612 per 1,000 calls · 810 ms | 98.0% · $0.633 per 1,000 calls · 870 ms |
| Nano-XS | 81.0% · $0.030 per 1,000 calls · 140 ms | 88.5% · $0.036 per 1,000 calls · 198 ms | 87.9% · $0.048 per 1,000 calls · 217 ms | 92.7% · $0.054 per 1,000 calls · 275 ms | 91.3% · $0.102 per 1,000 calls · 378 ms | 94.8% · $0.111 per 1,000 calls · 450 ms |

Pass rate, cost per 1,000 calls and p95 latency for each model under each reasoning level and retry setting.

### Neither step above the lean is separated: high reasoning adds +0.8 points on Brisk-S, and swapping Atlas-M for Brisk-S at high reasoning with retry moves pass rate by -0.7 points.

| Group | Row | Value | 95% unpaired interval | Note |
|---|---|---|---|---|
| model | Atlas-M · reasoning high · 1 retry → Brisk-S | -0.7 | -1.7 to +0.4 | -0.7 pts |
| reasoning | Brisk-S · reasoning low · 1 retry → High | +0.8 | -0.4 to +2.1 | +0.8 pts |

Each effect is one setting moved between two arms that differ in it alone, on Pass rate. Descriptive: read as two arms' own spreads, never a test.

Change in pass rate from moving one setting, each interval read as two independent setups.

#### Brisk-S low with retry trails most on multi-hop questions, where high reasoning adds +1.7 points; its kept failures collapse hops, slip a date by a day, or skip a clarifying question.

| Slice | Arm | Pass rate |
|---|---|---|
| Single lookup | S · Nano-XS · reasoning high · 1 retry | 95.8% |
| Single lookup | J · Brisk-S · reasoning low · 1 retry | 98.4% |
| Single lookup | L · Brisk-S · reasoning high · 1 retry | 98.5% |
| Single lookup | F · Atlas-M · reasoning high · 1 retry | 99.2% |
| Multi-hop (2 to 3 calls) | S · Nano-XS · reasoning high · 1 retry | 93.5% |
| Multi-hop (2 to 3 calls) | J · Brisk-S · reasoning low · 1 retry | 95.6% |
| Multi-hop (2 to 3 calls) | L · Brisk-S · reasoning high · 1 retry | 97.3% |
| Multi-hop (2 to 3 calls) | F · Atlas-M · reasoning high · 1 retry | 97.9% |
| Date and unit math | S · Nano-XS · reasoning high · 1 retry | 94.7% |
| Date and unit math | J · Brisk-S · reasoning low · 1 retry | 97.0% |
| Date and unit math | L · Brisk-S · reasoning high · 1 retry | 97.9% |
| Date and unit math | F · Atlas-M · reasoning high · 1 retry | 98.6% |
| Ambiguous entity | S · Nano-XS · reasoning high · 1 retry | 94.2% |
| Ambiguous entity | J · Brisk-S · reasoning low · 1 retry | 96.4% |
| Ambiguous entity | L · Brisk-S · reasoning high · 1 retry | 97.6% |
| Ambiguous entity | F · Atlas-M · reasoning high · 1 retry | 98.3% |

Pass rate by task for the four setups a decider would weigh; task figures state no case counts, so no intervals.

| Example | Details | Reason | Excerpt |
|---|---|---|---|
| TQ-0418 · Multi-hop (2 to 3 calls) · Brisk-S · reasoning low · 1 retry · missing call | Input: Who chaired the committee that approved the 2019 grid budget?; Expected: search(committee, grid budget 2019) → lookup(chair) | Collapsed two hops into one query; returned the budget author, not the chair. | search(grid budget 2019 chair) |
| TQ-0772 · Date and unit math · Brisk-S · reasoning low · 1 retry · wrong args | Input: Filings in the 90 days before the March 3 hearing; Expected: date_from=2026-12-03, date_to=2027-03-03 | Off-by-one on the window start. The most common wrong-args pattern (9 of 17). | date_from=2026-12-04, date_to=2027-03-03 |
| TQ-1103 · Ambiguous entity · Brisk-S · reasoning low · 1 retry · wrong tool | Input: Latest on Mercury's pricing; Expected: ask_clarify(company or planet?) | Picked the company without asking. Scored wrong because the gold answer requires clarification. | search(Mercury pricing) |

The three failures kept for Brisk-S low with retry, one per task; how many failed of each kind is not stated.

#### Brisk-S low with retry answers within 513 ms at p95, against 2551 ms for Atlas-M high with retry.

| Group | Row | From | To |
|---|---|---|---|
| Latency | S · Nano-XS · reasoning high · 1 retry | 170 ms | 450 ms |
| Latency | J · Brisk-S · reasoning low · 1 retry | 209 ms | 513 ms |
| Latency | L · Brisk-S · reasoning high · 1 retry | 364 ms | 870 ms |
| Latency | F · Atlas-M · reasoning high · 1 retry | 1094 ms | 2551 ms |

Latency from p50 to p95; no latency budget was declared.

## What you could change

| What you could change | What it decides | What happens as it moves |
|---|---|---|
| Monthly value of one point of pass rate | Which setup on the cost curve is worth picking | Below $319.67, Nano-XS high with retry; to $1,595.18, Brisk-S low with retry; to $17,236.36, Brisk-S high with retry; above, Atlas-M high with retry. |
| Minimum pass rate | The cheapest setup that clears it, read on point estimates | Up to 94.8%, Nano-XS high with retry at $444.00 a month; up to 97.1%, Brisk-S low with retry; up to 98%, Brisk-S high with retry at $2,532.00; above, only Atlas-M high with retry at $13,908.00. |
| Latency budget at p95 | Whether Atlas-M setups stay in play | A budget under 2551 ms rules out Atlas-M high with retry; every Brisk-S setup with retry stays within 870 ms. |

## Next steps

- The planner's owner declares a minimum pass rate or a monthly price per point of pass rate. _Turns the trade into a decision: the curve then names one setup._
- Re-run Brisk-S low and high reasoning, both with retry, on the same fresh cases, 5432 cases each. _Whether high reasoning's extra point is real; four runs in five of that size separate today's gap if it holds, and fresh cases also correct the pick's best-of-many optimism._
- Run a routed mix that sends multi-hop questions to Brisk-S high with retry and the rest to Brisk-S low with retry, measured like the other setups. _Whether it keeps most of high reasoning's gain at close to low reasoning's cost; nobody has run it, so no figure prices it yet._
- Record case counts per task in the next run. _Lets task-level gaps, such as the multi-hop one, be read with intervals instead of as point estimates._

## Full results

### The arms

Each arm's full configuration, once. The rest of the page names an arm by its letter tag and name.

| Arm | Role | Configuration | Served model |
|---|---|---|---|
| A · Atlas-M · reasoning off · no retry | Candidate | Atlas-M · reasoning off · retry 0 | Atlas-M |
| B · Atlas-M · reasoning off · 1 retry | Candidate | Atlas-M · reasoning off · retry 1 | Atlas-M |
| C · Atlas-M · reasoning low · no retry | Candidate | Atlas-M · reasoning low · retry 0 | Atlas-M |
| D · Atlas-M · reasoning low · 1 retry | Candidate | Atlas-M · reasoning low · retry 1 | Atlas-M |
| E · Atlas-M · reasoning high · no retry | Candidate | Atlas-M · reasoning high · retry 0 | Atlas-M |
| F · Atlas-M · reasoning high · 1 retry | Candidate | Atlas-M · reasoning high · retry 1 | Atlas-M |
| G · Brisk-S · reasoning off · no retry | Candidate | Brisk-S · reasoning off · retry 0 | Brisk-S |
| H · Brisk-S · reasoning off · 1 retry | Candidate | Brisk-S · reasoning off · retry 1 | Brisk-S |
| I · Brisk-S · reasoning low · no retry | Candidate | Brisk-S · reasoning low · retry 0 | Brisk-S |
| J · Brisk-S · reasoning low · 1 retry | Candidate | Brisk-S · reasoning low · retry 1 | Brisk-S |
| K · Brisk-S · reasoning high · no retry | Candidate | Brisk-S · reasoning high · retry 0 | Brisk-S |
| L · Brisk-S · reasoning high · 1 retry | Candidate | Brisk-S · reasoning high · retry 1 | Brisk-S |
| M · Nano-XS · reasoning off · no retry | Candidate | Nano-XS · reasoning off · retry 0 | Nano-XS |
| N · Nano-XS · reasoning off · 1 retry | Candidate | Nano-XS · reasoning off · retry 1 | Nano-XS |
| O · Nano-XS · reasoning low · no retry | Candidate | Nano-XS · reasoning low · retry 0 | Nano-XS |
| Q · Nano-XS · reasoning low · 1 retry | Candidate | Nano-XS · reasoning low · retry 1 | Nano-XS |
| R · Nano-XS · reasoning high · no retry | Candidate | Nano-XS · reasoning high · retry 0 | Nano-XS |
| S · Nano-XS · reasoning high · 1 retry | Candidate | Nano-XS · reasoning high · retry 1 | Nano-XS |

### Every arm by measure

| Arm | Pass rate | Calls per task | Cost per 1k calls | Latency |
|---|---|---|---|---|
| A · Atlas-M · reasoning off · no retry | 94% | 1.539 calls | $1.00 | 900 ms (p95) |
| B · Atlas-M · reasoning off · 1 retry | 96.4% | 1.599 calls | $1.06 | 1019 ms (p95) |
| C · Atlas-M · reasoning low · no retry | 96.6% | 1.469 calls | $1.60 | 1395 ms (p95) |
| D · Atlas-M · reasoning low · 1 retry | 97.9% | 1.504 calls | $1.66 | 1500 ms (p95) |
| E · Atlas-M · reasoning high · no retry | 97.7% | 1.43 calls | $3.40 | 2430 ms (p95) |
| F · Atlas-M · reasoning high · 1 retry | 98.6% | 1.452 calls | $3.48 | 2551 ms (p95) |
| G · Brisk-S · reasoning off · no retry | 91.1% | 1.579 calls | $0.18 | 300 ms (p95) |
| H · Brisk-S · reasoning off · 1 retry | 94.6% | 1.668 calls | $0.20 | 359 ms (p95) |
| I · Brisk-S · reasoning low · no retry | 95.3% | 1.492 calls | $0.29 | 465 ms (p95) |
| J · Brisk-S · reasoning low · 1 retry | 97.1% | 1.539 calls | $0.30 | 513 ms (p95) |
| K · Brisk-S · reasoning high · no retry | 96.6% | 1.442 calls | $0.61 | 810 ms (p95) |
| L · Brisk-S · reasoning high · 1 retry | 98% | 1.476 calls | $0.63 | 870 ms (p95) |
| M · Nano-XS · reasoning off · no retry | 81% | 1.698 calls | $0.0300 | 140 ms (p95) |
| N · Nano-XS · reasoning off · 1 retry | 88.5% | 1.888 calls | $0.0360 | 198 ms (p95) |
| O · Nano-XS · reasoning low · no retry | 87.9% | 1.559 calls | $0.0480 | 217 ms (p95) |
| Q · Nano-XS · reasoning low · 1 retry | 92.7% | 1.68 calls | $0.0540 | 275 ms (p95) |
| R · Nano-XS · reasoning high · no retry | 91.3% | 1.479 calls | $0.10 | 378 ms (p95) |
| S · Nano-XS · reasoning high · 1 retry | 94.8% | 1.566 calls | $0.11 | 450 ms (p95) |

### Every task

| Task | Cases | A · Atlas-M · reasoning off · no retry | B · Atlas-M · reasoning off · 1 retry | C · Atlas-M · reasoning low · no retry | D · Atlas-M · reasoning low · 1 retry | E · Atlas-M · reasoning high · no retry | F · Atlas-M · reasoning high · 1 retry | G · Brisk-S · reasoning off · no retry | H · Brisk-S · reasoning off · 1 retry | I · Brisk-S · reasoning low · no retry | J · Brisk-S · reasoning low · 1 retry | K · Brisk-S · reasoning high · no retry | L · Brisk-S · reasoning high · 1 retry | M · Nano-XS · reasoning off · no retry | N · Nano-XS · reasoning off · 1 retry | O · Nano-XS · reasoning low · no retry | Q · Nano-XS · reasoning low · 1 retry | R · Nano-XS · reasoning high · no retry | S · Nano-XS · reasoning high · 1 retry |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Single lookup | — | 97.2% | 98.3% | 98.6% | 99.2% | 98.7% | 99.2% | 94.3% | 96.6% | 97.4% | 98.4% | 97.6% | 98.5% | 86.8% | 92% | 91.7% | 95% | 93% | 95.8% |
| Multi-hop (2 to 3 calls) | — | 90.2% | 94.1% | 94.1% | 96.4% | 96.6% | 97.9% | 87.3% | 92.3% | 92.8% | 95.6% | 95.5% | 97.3% | 74.2% | 84.4% | 83.5% | 90% | 89.3% | 93.5% |
| Date and unit math | — | 93.7% | 96.2% | 96.4% | 97.8% | 97.7% | 98.6% | 90.8% | 94.4% | 95.1% | 97% | 96.6% | 97.9% | 80.5% | 88.2% | 87.6% | 92.5% | 91.1% | 94.7% |
| Ambiguous entity | — | 92.2% | 95.3% | 95.4% | 97.2% | 97.2% | 98.3% | 89.3% | 93.5% | 94.1% | 96.4% | 96.1% | 97.6% | 77.8% | 86.6% | 85.8% | 91.4% | 90.3% | 94.2% |

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

- **Calls per task** Retries add calls, but their cost is already inside cost per 1,000 calls; it changes no pick. In the appendix.
- **Extra calls and outcome measures** Listed for the campaign but hold no figures for any setup.
- **Model effects at every reasoning and retry level** Atlas-M leads Brisk-S at each level by under three points and Nano-XS trails both; the lattice shows it, and only the top pair bears on the trade.
- **Atlas-M setups other than high reasoning with retry** Each costs more than another setup that passes about as often or more; none is on the cost curve.
- **Interactions between retry, reasoning and model** Nano-XS gains more from a retry than Atlas-M does; no interaction changes the pick, since every setup on the curve above the cheapest already uses a retry.

### Method

- **Cases** 1200 cases per setup, one run each, none lost; 18 setups crossing three models, three reasoning levels and retry on or off. Whether every setup ran the same cases is not stated.
- **Pass** The campaign does not define what makes a task pass. The kept failures are scored against an expected sequence of calls: missing calls, wrong arguments and wrong tools.
- **Intervals** Computed at 95% from each setup's pass rate and case count, two setups combined as independent, with no correction across the many comparisons. Task-level figures state no case counts and carry no interval.
- **Cost** Cost per 1,000 calls as measured; monthly figures at the declared volume of 4000000 a month. Prices per point are arithmetic on point estimates.
- **Latency** p50 and p95 per setup; no budget was declared and run conditions are not stated.
- **No control or criteria** No setup was named the current one and no bar, floor or price was declared before the run, so every comparison is between setups' own figures.
- **Source** Synthetic example data from docs/eval-report-kit/references/r2-config-sweep.json.

### Review findings left open

- **Should fix** · The page: The minimum-pass-rate row in 'What you could change' says Brisk-S low with retry is the cheapest setup that clears any minimum up to 97.1%. For a minimum above 94.8% and up to 95.3%, Brisk-S with low reasoning and no retry is cheaper: 95.3% at $0.288 per 1,000 calls (about $1,152 a month, against $1,208). The hypothetical pass-rate sweep in decision_sensitivity confirms it. This row is the tool the owner will use once they declare a minimum, so it should be exact. Add that band, naming Brisk-S low without retry, or say that the row lists only setups on the cost curve.
- **Should fix** · The page: The trade line says the top step, to Atlas-M with high reasoning and retry, 'costs $13,908.00 a month'. That is Atlas-M's total monthly cost (cost_per_month.atlas-m.high.r1). The step itself adds $11,376.00 a month over Brisk-S high with retry (price.brisk-s.high.r1.atlas-m.high.r1.pass_rate.monthly_cost_gap). Read as the price of the step, the sentence overstates it. Either cite the step cost or write 'brings the monthly cost to $13,908.00'.
- **Should fix** · Claim 3: The claim rests on effect sizes ('the cheapest lever', '+7.5 points' against '+0.9'), but its block is a lattice of point estimates. The retry lever effects have intervals: +1.9 (+0.3 to +3.4) on Brisk-S low, +7.5 (+4.6 to +10.4) on Nano-XS with reasoning off, and +0.9 (-0.2 to +2) on Atlas-M high, which is not separated. Show them as a diverging bars block, or add them to the claim. Also say that the 'low with retry reads at least as high as high without' gaps (0.2, 0.5 and 1.4 points) are readings, not separated differences.
- **Should fix** · Claim 5: The caption says 'how many failed of each kind is not stated', but example TQ-0772's reason says off-by-one is 'the most common wrong-args pattern (9 of 17)'. The snapshot keeps no count of how many examples came from how many failures, so 9 of 17 is not anchored to any arm or base. Either say what 9 of 17 counts (which arm, out of what), or drop it, so that the caption and the example agree.
- **Should fix** · The page: The confidence line reads 'High, if the caveats below aren't material', and the first caveat begins 'Material for the pick itself'. A reader sees High confidence next to a caveat that calls itself material. Reword the caveat so it says it is material only if the owner's minimum sits above 97.1% or their price per point sits outside $319.67 to $1,595.18, which is what the rest of the caveat already says.

_Written by claude-opus-5-5, served as claude-opus-5-5, for $2.28, with analyst skill b70be6bcfa68, on 2026-10-28._
