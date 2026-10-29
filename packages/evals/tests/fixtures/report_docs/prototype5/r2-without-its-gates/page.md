Choice · candidates compared · Research query planner · 18 arms · 4 tasks · Run 2026-10-23 · Synthetic example data

# No price per point was set: Brisk-S · reasoning low · 1 retry (J) is the knee; the top option passes 1.5 points more at 12 times its cost.

Every cheaper option passes fewer tasks; the nearest saves only $56 a month and trails by 1.9 points, clear of zero. Above the lean no single step is told apart from it; a declared floor or price per point would turn this trade into a decision.

**Confidence: High, if the caveats below aren't material.**

- **No arm is the control.** Only if pairing the shared cases separated the single steps above Brisk-S · reasoning low · 1 retry (J). Every interval here is unpaired; pairing would likely narrow them where outcomes correlate across shared cases, which could make the dearer options' small gains real and move the lean up the frontier. The pick-of-many caveat below pulls the other way.
- **Nothing declares what is good enough.** Only if the owner prices a point of pass rate below $319.67 or above $1,595.18 a month, or sets a floor or latency budget the lean misses; then another option is the better buy.
- **The pick is the best of several candidates.** Only if a fresh run reads Brisk-S · reasoning low · 1 retry (J) lower by about the low end of its lead over Brisk-S · reasoning low · no retry (I) (+0.3 to +3.4 points) or more; then nothing separates them and the cheaper one is the better buy. As the pick among eighteen options its estimate likely reads somewhat high.
- **What the monthly volume counts.** Only if the declared volume counts tasks rather than calls: every monthly figure and the price band would then scale by each option's calls per task, from 1.452 calls to 1.566 calls among the frontier options from Nano-XS · reasoning high · 1 retry (S) up, raising the band's dollars by about half. The order of the frontier steps, and so the knee, holds on those point figures.

**The trade.** Moving up from Nano-XS · reasoning high · 1 retry (S) to Brisk-S · reasoning low · 1 retry (J) costs $764 more a month for +2.4 points, clear of zero. Going on to the top option, Atlas-M · reasoning high · 1 retry (F), costs $12,700 more a month again for +1.5 points (+0.3 to +2.6), also clear of zero.

**Where the answer flips.** Brisk-S · reasoning low · 1 retry (J) is the best buy when a point of pass rate is worth between $319.67 and $1,595.18 a month, if the declared volume counts calls; both ends are point prices with no interval. Read as a floor on point estimates, it is the cheapest option for any pass-rate floor above 95.3% up to its own 97.1%.

**Our lean.** Ours, not a rule: run Brisk-S · reasoning low · 1 retry (J). It is the cheapest option that no single step up beats by a margin this run can separate, and it holds for any price per point in the band above.

**Watch.** No line was declared, so no margin is weakest. The lean's thinnest separated lead is over Brisk-S · reasoning low · no retry (I) (+0.3 to +3.4 points); a fresh run of today's size reads that pair the other way 0.8% of the time. Multi-hop questions are the lean's weakest task, and its p95 latency of 513 ms has no budget to read against.

**What would settle it.** The planner's owner declares a pass-rate floor, a price per point, a latency budget, and what the monthly volume counts. A paired re-run settles whether the single steps above the lean buy real points.

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
| J · Brisk-S · reasoning low · 1 retry | Candidate | Brisk-S · Reasoning budget low · Retry on failure 1 | 97.1% | $0.302 |
| K · Brisk-S · reasoning high · no retry | Candidate | Brisk-S · Reasoning budget high · Retry on failure 0 | 96.6% | $0.612 |
| L · Brisk-S · reasoning high · 1 retry | Candidate | Brisk-S · Reasoning budget high · Retry on failure 1 | 98% | $0.633 |
| M · Nano-XS · reasoning off · no retry | Candidate | Nano-XS · Reasoning budget off · Retry on failure 0 | 81% | $0.030 |
| N · Nano-XS · reasoning off · 1 retry | Candidate | Nano-XS · Reasoning budget off · Retry on failure 1 | 88.5% | $0.036 |
| O · Nano-XS · reasoning low · no retry | Candidate | Nano-XS · Reasoning budget low · Retry on failure 0 | 87.9% | $0.048 |
| Q · Nano-XS · reasoning low · 1 retry | Candidate | Nano-XS · Reasoning budget low · Retry on failure 1 | 92.7% | $0.054 |
| R · Nano-XS · reasoning high · no retry | Candidate | Nano-XS · Reasoning budget high · Retry on failure 0 | 91.3% | $0.102 |
| S · Nano-XS · reasoning high · 1 retry | Candidate | Nano-XS · Reasoning budget high · Retry on failure 1 | 94.8% | $0.111 |

## Pass rate rises with cost along a frontier of seven options, and Brisk-S · reasoning low · 1 retry (J) sits where the curve flattens.

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

Each option's cost per 1k calls (log axis) against its pass rate, the lean marked.

### Each step up the frontier costs more per point than the one before, reaching $17,236.36 a month per point for the last.

| Step along the frontier | Price per point |
|---|---|
| Nano-XS · reasoning off · no retry → Nano-XS · reasoning off · 1 retry | $3.20 |
| Nano-XS · reasoning off · 1 retry → Nano-XS · reasoning low · 1 retry | $17.31 |
| Nano-XS · reasoning low · 1 retry → Nano-XS · reasoning high · 1 retry | $111.22 |
| Nano-XS · reasoning high · 1 retry → Brisk-S · reasoning low · 1 retry | $319.67 |
| Brisk-S · reasoning low · 1 retry → Brisk-S · reasoning high · 1 retry | $1,595.18 |
| Brisk-S · reasoning high · 1 retry → Atlas-M · reasoning high · 1 retry | $17,236.36 |

Price of one point of pass rate a month at the declared volume, read as calls, for each step along the frontier; point figures with no interval.

### Above Brisk-S · reasoning low · 1 retry (J), neither raising reasoning nor swapping Brisk-S for Atlas-M moves pass rate by an amount this run can tell from zero; only both together, the top option, pass more by a separated +1.5 points.

| Group | Row | Value | 95% unpaired interval |
|---|---|---|---|
| model | Atlas-M · reasoning low · 1 retry → Brisk-S | -0.8 | -2.0 to +0.5 |
| model | Atlas-M · reasoning high · 1 retry → Brisk-S | -0.7 | -1.7 to +0.4 |
| reasoning | Atlas-M · reasoning low · 1 retry → High | +0.7 | -0.3 to +1.7 |
| reasoning | Brisk-S · reasoning low · 1 retry → High | +0.8 | -0.4 to +2.1 |
| model effect, across reasoning | (Atlas-M · reasoning high · 1 retry → Brisk-S) less (Atlas-M · reasoning low · 1 retry → Brisk-S) | +0.1 | -1.5 to +1.7 |

Each effect is one setting moved between two arms that differ in it alone, on Pass rate; each interaction is the same move's effect in one context less its effect in another, the two contexts differing in one other setting. Descriptive: read as two arms' own spreads, never a test.

Change in pass rate from moving one setting among the four options with a retry; the model rows read Brisk-S less Atlas-M, so a negative bar is Brisk-S behind. Unpaired intervals.

| Row | Value | 95% unpaired interval |
|---|---|---|
| Overall | +1.5 | +0.3 to +2.6 |
| Single lookup | +0.8 |  |
| Multi-hop (2 to 3 calls) | +2.3 |  |
| Date and unit math | +1.6 |  |
| Ambiguous entity | +1.9 |  |

Each row is Atlas-M · reasoning high · 1 retry less Brisk-S · reasoning low · 1 retry on Pass rate, overall and in each place both hold it.

The interval is independent: each arm's own spread (a pass/fail rate's binomial spread, from the arm's cases), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same.

The top option's pass rate less the lean's, overall with an unpaired interval and in each task, where no count gives an interval.

### Below it, the nearest cheaper option, Brisk-S · reasoning low · no retry (I), passes 1.9 points fewer, the lean's thinnest separated lead; the next frontier option down, Nano-XS · reasoning high · 1 retry (S), trails by 2.4.

| Row | Value | 95% unpaired interval |
|---|---|---|
| Overall | +1.9 | +0.3 to +3.4 |
| Single lookup | +1.0 |  |
| Multi-hop (2 to 3 calls) | +2.8 |  |
| Date and unit math | +1.9 |  |
| Ambiguous entity | +2.3 |  |

Each row is Brisk-S · reasoning low · 1 retry less Brisk-S · reasoning low · no retry on Pass rate, overall and in each place both hold it.

The interval is independent: each arm's own spread (a pass/fail rate's binomial spread, from the arm's cases), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same.

The lean's pass rate less the nearest cheaper option's, which sits off the price frontier; overall with an unpaired interval, and in each task, where no count gives an interval.

| Row | Value | 95% unpaired interval |
|---|---|---|
| Overall | +2.4 | +0.8 to +4 |
| Single lookup | +2.6 |  |
| Multi-hop (2 to 3 calls) | +2.1 |  |
| Date and unit math | +2.3 |  |
| Ambiguous entity | +2.2 |  |

Each row is Brisk-S · reasoning low · 1 retry less Nano-XS · reasoning high · 1 retry on Pass rate, overall and in each place both hold it.

The interval is independent: each arm's own spread (a pass/fail rate's binomial spread, from the arm's cases), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same.

The lean's pass rate less the frontier option below it, laid out the same way.

### A retry is the cheapest lift on every model, and it buys most where pass rate starts lowest: +7.5 points on Nano-XS · reasoning off · no retry (M) for $24 a month.

| Row | Value | 95% unpaired interval |
|---|---|---|
| Atlas-M · reasoning high · no retry → 1 retry | +0.9 | -0.2 to +2.0 |
| Brisk-S · reasoning low · no retry → 1 retry | +1.9 | +0.3 to +3.4 |
| Nano-XS · reasoning off · no retry → 1 retry | +7.5 | +4.6 to +10.4 |

Each effect is one setting moved between two arms that differ in it alone, on Pass rate. Descriptive: read as two arms' own spreads, never a test.

Change in pass rate from adding a retry, for one setting of each model; unpaired intervals.

#### The dearer options are also slower: Atlas-M · reasoning high · 1 retry (F) reaches a p95 latency of 2551 ms.

| Group | Row | From | To |
|---|---|---|---|
| Latency | S · Nano-XS · reasoning high · 1 retry | 170 ms | 450 ms |
| Latency | J · Brisk-S · reasoning low · 1 retry | 209 ms | 513 ms |
| Latency | L · Brisk-S · reasoning high · 1 retry | 364 ms | 870 ms |
| Latency | D · Atlas-M · reasoning low · 1 retry | 628 ms | 1500 ms |
| Latency | F · Atlas-M · reasoning high · 1 retry | 1094 ms | 2551 ms |

Latency from p50 to p95 for the options around the knee; no budget was declared.

#### Multi-hop questions are the weakest task for the lean and for the frontier options on either side of it.

|  | Nano-XS · reasoning high · 1 retry | Brisk-S · reasoning low · 1 retry | Atlas-M · reasoning high · 1 retry |
|---|---|---|---|
| Single lookup | 95.8% | 98.4% | 99.2% |
| Multi-hop (2 to 3 calls) | 93.5% | 95.6% | 97.9% |
| Date and unit math | 94.7% | 97.0% | 98.6% |
| Ambiguous entity | 94.2% | 96.4% | 98.3% |

Pass rate by task for the frontier option below the knee, the lean and the top option; tasks state no case counts, so these figures are descriptive.

#### The lean's kept failures are planning errors: a collapsed hop, an off-by-one date window, and a guess where it should have asked.

| Example | Details | Reason | Excerpt |
|---|---|---|---|
| TQ-0418 · Multi-hop (2 to 3 calls) · Brisk-S · reasoning low · 1 retry · missing call | Input: Who chaired the committee that approved the 2019 grid budget?; Expected: search(committee, grid budget 2019) → lookup(chair) | Collapsed two hops into one query; returned the budget author, not the chair. | search(grid budget 2019 chair) |
| TQ-0772 · Date and unit math · Brisk-S · reasoning low · 1 retry · wrong args | Input: Filings in the 90 days before the March 3 hearing; Expected: date_from=2026-12-03, date_to=2027-03-03 | Off-by-one on the window start. The most common wrong-args pattern (9 of 17). | date_from=2026-12-04, date_to=2027-03-03 |
| TQ-1103 · Ambiguous entity · Brisk-S · reasoning low · 1 retry · wrong tool | Input: Latest on Mercury's pricing; Expected: ask_clarify(company or planet?) | Picked the company without asking. Scored wrong because the gold answer requires clarification. | search(Mercury pricing) |

Every failure the snapshot keeps for the lean, one per task where kept; the run's total failures are not counted, so these show kinds, not shares.

## What you could change

| What you could change | What it decides | What happens as it moves |
|---|---|---|
| Price of a point of pass rate a month | Which frontier option is the best buy | Below $319.67 a cheaper option wins; up to $1,595.18 the lean holds; up to $17,236.36 Brisk-S · reasoning high · 1 retry (L) does; above that, Atlas-M · reasoning high · 1 retry (F). Point prices with no interval, on the volume read as calls. |
| A pass-rate floor, read on point estimates | The cheapest option that clears it | The lean is cheapest for floors above 95.3% up to 97.1%; Brisk-S · reasoning high · 1 retry (L) up to 98%; above that only Atlas-M · reasoning high · 1 retry (F) clears, up to 98.6%. |
| A p95 latency budget | Which options stay in | A budget under 513 ms rules the lean out; every Atlas-M option reads at or above 900 ms. |

## Next steps

- The planner's owner declares a pass-rate floor, a price per point of pass rate, a p95 latency budget, and what the monthly volume counts. _Turns the trade into a decision: each declaration picks one frontier option, and the volume fixes the dollars the price is set in._
- Re-run Brisk-S · reasoning low · 1 retry (J), Brisk-S · reasoning high · 1 retry (L) and Atlas-M · reasoning low · 1 retry (D) on fresh cases, paired. Read independently, telling the lean from Atlas-M · reasoning low · 1 retry (D) four times in five needs 6,044 cases per arm if today's gap is the truth; pairing needs fewer. _Whether the single steps above the lean buy real points, and how far the lean's estimate falls on fresh cases._
- Target multi-hop planning on Brisk-S and Nano-XS, then re-run the frontier options. _Whether cheaper options can close the gap on the weakest task and move the knee down the price curve._

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

- **The other options off the frontier** Each costs more than another option that passes at least as many tasks; Brisk-S with low reasoning and no retry, off only the price curve, is shown below the lean instead. All are in the full tables.
- **The other retry effects and their interaction with model** Every model and reasoning setting gains from a retry, all but Atlas-M with high reasoning clear of zero, and the gain is larger on Nano-XS than on Atlas-M; this agrees with the retry claim and changes no option's place.
- **Reasoning effects at off and low** Reasoning raises pass rate on every model but costs more than a retry; the frontier already prices it.
- **Calls per task as a result** Describes how options plan, not how often they pass; it enters only through the volume caveat.
- **Outcome and extra calls per task** The snapshot holds no figures for them.
- **Per-task pass rates of the other options** No task states its case count, so no task figure has an interval; they stay in the appendix.

### Method

- **Design** A factorial sweep of three models, three reasoning levels and retry on or off: eighteen options, each run on 1,200 research-planning tasks across four task types, none lost. No repeats are stated.
- **Pass** The input does not say what makes a task pass. Kept failures are scored against an expected plan of tool calls and their arguments.
- **Intervals** 95%, computed from each option's pass rate and case count, combined as if the options ran different cases. No correction for the many comparisons. Task figures state no counts and carry no interval.
- **Cost** Cost per 1k calls, a point figure with no spread. Monthly figures use the declared volume of 4,000,000 a month, read as calls; the input does not say what that volume counts.
- **Latency** p50 and p95 per option; no latency budget was declared.
- **Not compared** There is no control and nothing was declared in advance: no gate, floor, margin or exchange rate.
- **Source** Synthetic example data from docs/eval-report-kit/references/r2-config-sweep.json, run 2026-10-23.

### Review findings left open

- **Should fix** · The page: Add Brisk-S with low reasoning and no retry to the paired re-run in Next steps. Watch and the pick-of-many caveat both name the lean's lead over it (+1.9, +0.3 to +3.4) as the thinnest, and that is where the $56-a-month cheaper option could become the better buy. Today's design needs about 1,636 cases per arm to separate them four times in five when read unpaired (settle_cases); pairing needs fewer. Say that this re-run settles whether the lean's lead survives on fresh cases.
- **Should fix** · Claim 1: 'A frontier of seven options' counts the price curve, which is how price_tradeoff counts it. But Brisk-S with low reasoning and no retry ($0.288, 95.3%) is undominated, and the page calls it 'off only the price curve'. Say 'price curve' in the claim, and mark the seven curve options in the scatter so the block shows the frontier the sentence names.
- **Should fix** · Claim 5: The claim says a retry is 'the cheapest lift on every model' and quotes $24 a month for Nano-XS with reasoning off, but the block shows only pass-rate changes. Add each retry step's monthly cost gap, or its price per point, beside the rows, or cut 'cheapest' from the sentence. Also note that a retry raises calls per task (1.698 to 1.888 on Nano-XS with reasoning off), so the $24 grows if the volume counts tasks.
- **Should fix** · Claim 7: 'The frontier options on either side of it' implies the next step up, Brisk-S with high reasoning and one retry, but the heatmap shows the top option, Atlas-M with high reasoning and one retry. Either swap in Brisk-S with high reasoning and one retry, whose weakest task is also multi-hop at 97.3%, or reword to 'the frontier option below it and the top option'.

_Written by claude-opus-5-5, served as claude-opus-5-5, for $3.71, with analyst skill 1677ccb31cd6, on 2026-10-29._
