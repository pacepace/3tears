Choice · candidates compared · Research query planner · 18 arms · 4 tasks · Run 2026-10-23 · Synthetic example data

# Ship Brisk-S with high reasoning and one retry: the cheapest configuration that clears the pass-rate gate by more than noise, at $2,532 a month.

It passes 98% of tasks, its whole interval above the 97% gate. The cheaper low-reasoning setting passes only on its estimate, by 0.1 points; if the gate is meant that way, it saves $1,324.00 a month.

**Confidence: High, if the caveats below aren't material.**

- **No arm is the control.** Only if the choice is meant relative to the planner now in production, which was not run; the gate is absolute and every configuration is read against it.
- **Pass rate is cleared on a point estimate.** Only for the low-reasoning alternative, whose interval 96.2–98.1% straddles the gate; the recommendation clears it on its whole computed interval.
- **The pick is the best of several candidates.** If the recommendation's clearance of 1 point is mostly selection luck among eighteen configurations, so a fresh case set reads it below the gate.

**Recommendation.** Ship Brisk-S with high reasoning and one retry. It passes 98% of tasks, clear of the 97% gate on its whole interval, at $0.633 per thousand calls ($2,532 a month).

**Why not the others.** Brisk-S with low reasoning and one retry is the cheapest configuration that passes on its estimate, and so the declared rule's pick, but it clears by 0.1 points: a fresh run this size would read it below the gate 38.6% of the time, and settling it would take 54451 cases. High reasoning instead costs $1,324.00 more a month.

**Why not the others.** Atlas-M: no setting is separated from Brisk-S at the same settings, and its cheapest passing setting costs $6,620 a month. Nano-XS: no setting reaches the gate; its best passes 94.8%.

**Watch.** The pick is the best of eighteen configurations near one line, so a fresh case set may read it lower; if today's estimate is the truth, a fresh run this size reads it below the gate 0.9% of the time. Multi-hop questions are its weakest task at 97.3%, and its slow tail reaches 870 ms.

**Next.** Confirm the pick on a fresh case set of today's size; if it holds above the gate, ship. In parallel, the gate's owner says whether the gate is read on the estimate, which decides whether the low-reasoning setting is eligible.

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
| L · Brisk-S · reasoning high · 1 retry | Candidate (recommended) | Brisk-S · reasoning high · retry 1 | $0.633 |
| M · Nano-XS · reasoning off · no retry | Candidate | Nano-XS · reasoning off · retry 0 | $0.0300 |
| N · Nano-XS · reasoning off · 1 retry | Candidate | Nano-XS · reasoning off · retry 1 | $0.0360 |
| O · Nano-XS · reasoning low · no retry | Candidate | Nano-XS · reasoning low · retry 0 | $0.0480 |
| Q · Nano-XS · reasoning low · 1 retry | Candidate | Nano-XS · reasoning low · retry 1 | $0.0540 |
| R · Nano-XS · reasoning high · no retry | Candidate | Nano-XS · reasoning high · retry 0 | $0.102 |
| S · Nano-XS · reasoning high · 1 retry | Candidate | Nano-XS · reasoning high · retry 1 | $0.111 |

## Of the eighteen configurations, five reach the 97% pass-rate gate, and the two Brisk-S ones with reasoning and a retry cost a fraction of the three Atlas-M ones.

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
| Brisk-S · reasoning low · 1 retry | $0.302 | 97.1% |
| Brisk-S · reasoning high · no retry | $0.612 | 96.6% |
| L · Brisk-S · reasoning high · 1 retry | $0.633 | 98.0% |
| Nano-XS · reasoning off · no retry | $0.030 | 81.0% |
| Nano-XS · reasoning off · 1 retry | $0.036 | 88.5% |
| Nano-XS · reasoning low · no retry | $0.048 | 87.9% |
| Nano-XS · reasoning low · 1 retry | $0.054 | 92.7% |
| Nano-XS · reasoning high · no retry | $0.102 | 91.3% |
| Nano-XS · reasoning high · 1 retry | $0.111 | 94.8% |
| Pass rate gate, at least 97% |  | 97.0% |

The horizontal axis is logarithmic: each step is a multiple.

Cost per thousand calls, on a log axis, against pass rate for every configuration; the gate is 97%.

## Only high reasoning clears the gate by more than noise: its interval's lower end is 97.2%, while low reasoning sits on the line at 97.1%.

| Row | Value | 95% computed interval |
|---|---|---|
| I · Brisk-S · reasoning low · no retry | 95.3% | 94.1% to 96.5% |
| K · Brisk-S · reasoning high · no retry | 96.6% | 95.6% to 97.7% |
| J · Brisk-S · reasoning low · 1 retry | 97.1% | 96.2% to 98.1% |
| L · Brisk-S · reasoning high · 1 retry | 98.0% | 97.2% to 98.8% |
| C · Atlas-M · reasoning low · no retry | 96.6% | 95.5% to 97.6% |
| D · Atlas-M · reasoning low · 1 retry | 97.9% | 97.1% to 98.7% |
| E · Atlas-M · reasoning high · no retry | 97.7% | 96.9% to 98.6% |
| F · Atlas-M · reasoning high · 1 retry | 98.6% | 98.0% to 99.3% |
| S · Nano-XS · reasoning high · 1 retry | 94.8% | 93.5% to 96.0% |
| Pass rate gate, at least 97% | 97.0% |  |

Each interval is computed here from the rate's binomial spread, by the normal approximation, held to 0 to 100%. A rate of none or all has none.

Pass rate with its computed interval against the gate, for the configurations nearest it.

| Arm | Against | Today's figure | Line | Distance toward passing | Settles | Sizes count | Needed, one-sided | Needed, two-sided | Today | A fresh run reads the other side | Level |
|---|---|---|---|---|---|---|---|---|---|---|---|
| J · Brisk-S · reasoning low · 1 retry | Pass rate | 97.1% | 97.0% | +0.1 | clears | cases | 38350 | 54451 | 1200 | 38.6% | 95% one-sided |
| L · Brisk-S · reasoning high · 1 retry | Pass rate | 98.0% | 97.0% | +1.0 | clears | cases | 572 | 812 | 1200 | 0.9% | 95% one-sided |

In each row, "A fresh run reads the other side" is how often a fresh run of today's size would read the figure on the other side of its line, if the truth is today's estimate (95% one-sided); it plans a run and tests nothing.

How often a fresh run of today's size would read each on the other side of the gate, and the cases it would take to settle.

### Atlas-M is not separated from Brisk-S at the same settings, so its extra cost buys nothing measurable: the best Atlas-M setting costs $11,376.00 more a month than the pick.

| Group | Row | Value | 95% unpaired interval | Note |
|---|---|---|---|---|
| model | Atlas-M · reasoning low · 1 retry → Brisk-S | -0.8 | -2.0 to +0.5 | -0.8 pts |
| model | Atlas-M · reasoning high · 1 retry → Brisk-S | -0.7 | -1.7 to +0.4 | -0.7 pts |
| reasoning | Atlas-M · reasoning low · 1 retry → High | +0.7 | -0.3 to +1.7 | +0.7 pts |
| reasoning | Brisk-S · reasoning low · 1 retry → High | +0.8 | -0.4 to +2.1 | +0.8 pts |
| model effect, across reasoning | (Atlas-M · reasoning high · 1 retry → Brisk-S) less (Atlas-M · reasoning low · 1 retry → Brisk-S) | +0.1 | -1.5 to +1.7 | +0.1 pts |
| reasoning effect, across model | (Brisk-S · reasoning low · 1 retry → High) less (Atlas-M · reasoning low · 1 retry → High) | +0.1 | -1.5 to +1.7 | +0.1 pts |

Each effect is one setting moved between two arms that differ in it alone, on Pass rate; each interaction is the same move's effect in one context less its effect in another, the two contexts differing in one other setting. Descriptive: read as two arms' own spreads, never a test.

Pass-rate change from moving the model or the reasoning level alone, with a retry on, with unpaired intervals; the step from low to high reasoning is not separated either.

#### Multi-hop questions are where low reasoning falls short, at 95.6% against 97.3% with high reasoning.

| Group | Row | Value | Note |
|---|---|---|---|
| Single lookup | J · Brisk-S · reasoning low · 1 retry | 98.4% | 98.4% |
| Single lookup | L · Brisk-S · reasoning high · 1 retry | 98.5% | 98.5% |
| Multi-hop (2 to 3 calls) | J · Brisk-S · reasoning low · 1 retry | 95.6% | 95.6% |
| Multi-hop (2 to 3 calls) | L · Brisk-S · reasoning high · 1 retry | 97.3% | 97.3% |
| Date and unit math | J · Brisk-S · reasoning low · 1 retry | 97.0% | 97.0% |
| Date and unit math | L · Brisk-S · reasoning high · 1 retry | 97.9% | 97.9% |
| Ambiguous entity | J · Brisk-S · reasoning low · 1 retry | 96.4% | 96.4% |
| Ambiguous entity | L · Brisk-S · reasoning high · 1 retry | 97.6% | 97.6% |

Pass rate by task for the two Brisk-S settings with a retry; task figures state no case counts, so they are descriptive.

| Example | Details | Reason | Excerpt |
|---|---|---|---|
| TQ-0418 · Multi-hop (2 to 3 calls) · Brisk-S · reasoning low · 1 retry · missing call | Input: Who chaired the committee that approved the 2019 grid budget?; Expected: search(committee, grid budget 2019) → lookup(chair) | Collapsed two hops into one query; returned the budget author, not the chair. | search(grid budget 2019 chair) |

A kept multi-hop failure of low reasoning: two hops collapsed into one search.

#### High reasoning lengthens the slow tail to 870 ms from 513 ms, still well under either Atlas-M setting.

| Group | Row | From | To |
|---|---|---|---|
| Latency | J · Brisk-S · reasoning low · 1 retry | 209 ms | 513 ms |
| Latency | L · Brisk-S · reasoning high · 1 retry | 364 ms | 870 ms |
| Latency | D · Atlas-M · reasoning low · 1 retry | 628 ms | 1500 ms |
| Latency | F · Atlas-M · reasoning high · 1 retry | 1094 ms | 2551 ms |

Latency from median to slow tail; no latency budget was declared.

## What you could change

| What you could change | What it decides | What happens as it moves |
|---|---|---|
| How the pass-rate gate is read | High or low reasoning on Brisk-S | Read on the estimate, low reasoning passes and saves $1,324.00 a month; read on the whole interval, only high reasoning among the Brisk-S settings clears it, its lower end at 97.2%. |
| The gate's threshold | Which configuration is cheapest to pass | Up to 92.7%, Nano-XS with low reasoning and a retry, and cheaper Nano-XS settings below that; to 94.8%, Nano-XS with high reasoning and a retry; to 95.3%, Brisk-S with low reasoning and no retry; to 97.1%, Brisk-S with low reasoning and a retry; to 98%, the pick; to 98.6%, Atlas-M with high reasoning and a retry; above that, nothing passes. |
| Price of a pass-rate point | How far up the cost frontier to go | Low to high reasoning on Brisk-S costs $1,595.18 a month per point; Brisk-S to Atlas-M at high reasoning costs $17,236.36 a month per point. Neither step's gain is separated from zero; the first buys distance from the gate, not a measured improvement. |

## Next steps

- Re-run Brisk-S with high reasoning and one retry, with the low-reasoning setting beside it, on a fresh set of 1200 tasks. _Whether the pick's clearance survives selection among many configurations; if it reads below the gate, Atlas-M with low reasoning and a retry is the next cheapest option clear of it._
- The gate's owner declares whether the 97% gate is read on the estimate or on the whole interval. _Whether low reasoning, $1,324.00 a month cheaper, is eligible._
- Teach the low-reasoning planner to split multi-hop questions into steps, then re-test Brisk-S with low reasoning and a retry. _Whether the cheaper setting can clear the gate with room; more cases alone would need 54451._
- Record case counts per task in the next run. _Whether the multi-hop shortfall is real or noise._

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

- **Retry effects** A retry lifts every model and reasoning level at a small cost, and every Brisk-S survivor already carries one; it doesn't change the call.
- **Interactions between settings** Retry and reasoning effects don't differ between Atlas-M and Brisk-S; Nano-XS gains more from a retry but still misses the gate.
- **Nano-XS settings in detail** None reaches the gate; they appear in the scatter and the appendix.
- **Calls per task** Small differences between configurations; describes planner behaviour, not a decision axis.
- **Extra calls and outcome measures** The snapshot holds no figures for them.
- **Atlas-M with high reasoning and no retry** Passes on its estimate with an interval crossing the gate, and costs more than either Brisk-S survivor.

### Method

- **Cases** Each configuration ran 1200 tasks once, with none lost. The snapshot doesn't say whether every configuration ran the same tasks, so differences are read unpaired.
- **Pass** The snapshot does not define what makes a task pass; its kept failures are scored for a missing call, wrong arguments or the wrong tool.
- **Gate** The one declared criterion: pass rate at least 97% overall. It is read on the estimate as declared; the recommendation is also read on its whole interval.
- **Intervals** 95% intervals computed from each rate's binomial spread over its cases. Differences between configurations combine two spreads as independent, with no correction for the many comparisons.
- **Fresh-run risk** How often a fresh run of today's size reads the other side of the gate, assuming today's estimate is the truth, at 95% one-sided.
- **Cost** Cost per thousand calls as measured; monthly cost at the declared volume of 4000000 a month.
- **Latency** Median and slow-tail latency per configuration; no budget was declared, so it is descriptive.
- **Not compared** No production planner was run, so nothing is read against current behaviour.
- **Source** Synthetic example data from docs/eval-report-kit/references/r2-config-sweep.json, run 2026-10-23.

### Review findings left open

- **Should fix** · Claim 2: The headline and this claim say high reasoning 'clears the gate by more than noise'. The basis is an uncorrected 95% interval whose lower end, 97.2%, is 0.2 points above the 97% gate, read among eighteen configurations. Say so where the claim is made: give the 0.2-point cushion and say the interval has no correction for the many comparisons. Or soften the wording to 'clears with its whole uncorrected interval'.
- **Should fix** · The page: The decision block never says that high reasoning is not measurably better than low reasoning on Brisk-S: +0.8 points, interval -0.4 to +2.1, and about 2,659 cases per arm needed to settle. Add one sentence to the Why-not entry for low reasoning saying that the extra $1,324 a month buys distance from the gate, not a measured gain. That sentence now appears only in the levers table.
- **Should fix** · The page: The headline says 'Ship Brisk-S with high reasoning…', but Next says 'Confirm the pick on a fresh case set… if it holds above the gate, ship.' Make the two agree. Either the headline recommends the pick subject to confirmation, or Next makes clear that shipping now and confirming in parallel is acceptable.
- **Should fix** · Claim 4: 'Multi-hop questions are where low reasoning falls short' overstates the data. Low reasoning with a retry is also below the gate on ambiguous entity (96.4% against 97.6% with high reasoning), and it sits exactly at 97.0% on date and unit math. Reword it to say that multi-hop is the largest of several gaps. Also say that the one kept failure shown is one of four kept examples, so it shows a failure type but not its share of failures.
- **Should fix** · The page: The next step 'Teach the low-reasoning planner to split multi-hop questions' rests on one kept failure and on task figures with no case counts. Make it depend on the per-task case-count step, or present it as a hypothesis to test, so a reader does not budget engineering work on an unmeasured gap.
- **Should fix** · The page: The Method says monthly cost uses 'the declared volume of 4000000 a month' but does not say 4 million of what. Say whether the volume counts planner calls or tasks. If it counts tasks, monthly cost should also be multiplied by calls per task (about 1.48 for the pick), which would raise every monthly figure by about 1.5 times. The ranking would not change.

_Written by claude-opus-5-5, served as claude-opus-5-5, for $2.07, with analyst skill b70be6bcfa68, on 2026-10-28._
