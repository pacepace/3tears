Choice · candidates compared · Research query planner · 18 arms · 4 tasks · Run 2026-10-23 · Synthetic example data

# Run every planner with a retry; past that, picking a model is an unpriced trade in which each extra point of pass rate costs more, steepest for Atlas-M.

With a retry, Brisk-S · reasoning low · 1 retry (J) passes 97.1% for $1,208 a month; Atlas-M · reasoning high · 1 retry (F) passes 1.5 points more for $12,700 more a month. A declared pass-rate floor or price per point would turn this trade into a decision.

**Nothing was declared in advance: no control, no floor, no price per point. The planner's owner makes this call, after settling what the monthly volume counts.**

**Confidence: High, if the caveats below aren't material.**

- **Nothing declares what is good enough.** only once the owner sets a floor or a price per point: that line then picks the option, and the lean below holds only if it lands in the lean's band.
- **No arm is the control.** only for gaps whose interval sits near zero: every gap here is read unpaired, which errs wide, so a paired re-read could separate Brisk-S's high reasoning from its low, but would not undo a gap already clear of zero.
- **The pick is the best of several candidates.** only for thin leads, and Atlas-M at high reasoning with a retry is the thin lead it means: the top of eighteen configurations, ahead of the lean by an uncorrected interval whose low end is +0.3 points, so its true lead may be smaller or nil. The lean holds either way, since it rests on what each point costs, not on beating Atlas-M.
- **Cost routing unknown.** only if another provider serves these models at other prices: no configuration names who served it, so the cost order, every cost multiple between models and every price on the page hold for this run's routing only.
- **Monthly volume's unit unstated.** only if the declared 4,000,000 a month counts tasks rather than calls: then every monthly cost, every price per point and every band in the explore rows grows by the configurations' calls per task, between 1.452 calls and 1.888 calls on the frontier, and an owner pricing a point against today's bands could land in the wrong one. The owner should settle the unit before pricing.

**The trade.** Cost against pass rate. The cheapest configuration, Nano-XS · reasoning off · no retry (M), passes 81% for $120 a month; the highest, Atlas-M · reasoning high · 1 retry (F), passes 98.6% for $13,908. Every option between them on the frontier runs with a retry.

**Where the answer flips.** Each step up the frontier costs more per point a month: $319.67 from Nano-XS · reasoning high · 1 retry (S) to Brisk-S · reasoning low · 1 retry (J), $1,595.18 on to Brisk-S · reasoning high · 1 retry (L), and $17,236.36 on to Atlas-M · reasoning high · 1 retry (F). These prices are point estimates; the last two steps buy gains this run cannot tell from zero. Every band is listed under What you could change.

**Our lean.** Ours, not a declared rule: Brisk-S · reasoning low · 1 retry (J), for a point worth between $319.67 and $1,595.18 a month. It is clearly ahead of Nano-XS · reasoning high · 1 retry (S) (interval +0.8 to +4 points), and the next step up buys a gain not separated from zero.

**Watch.** Atlas-M · reasoning high · 1 retry (F) is ahead of the lean, with an interval of +0.3 to +2.6 points that clears zero only narrowly, read unpaired and uncorrected across eighteen configurations, at 12 times its cost at this run's unnamed routing. No margin or floor was declared, so no option sits near a line.

**What would settle it.** A declaration first: the owner settles what the monthly volume counts, then sets a pass-rate floor or a price per point. Then a re-run of Brisk-S at high against low reasoning, both with a retry, at 5,432 cases each (the size for independent arms; fewer if paired on the same cases), tells them apart four times in five.

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

## Cost and pass rate trade along one frontier: from Nano-XS · reasoning off · no retry (M) to Atlas-M · reasoning high · 1 retry (F), cost per thousand calls rises 116 times at this run's unnamed routing, and seven configurations sit on it.

_Caveats: Cost routing unknown._

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

Cost per thousand calls, on a log scale, against pass rate for every configuration, at this run's unnamed provider routing; the table below names the frontier's configurations.

| Step along the frontier | Price per point |
|---|---|
| Nano-XS · reasoning off · no retry → Nano-XS · reasoning off · 1 retry | $3.20 |
| Nano-XS · reasoning off · 1 retry → Nano-XS · reasoning low · 1 retry | $17.31 |
| Nano-XS · reasoning low · 1 retry → Nano-XS · reasoning high · 1 retry | $111.22 |
| Nano-XS · reasoning high · 1 retry → Brisk-S · reasoning low · 1 retry | $319.67 |
| Brisk-S · reasoning low · 1 retry → Brisk-S · reasoning high · 1 retry | $1,595.18 |
| Brisk-S · reasoning high · 1 retry → Atlas-M · reasoning high · 1 retry | $17,236.36 |

The frontier configurations in order, each step with its price per point of pass rate a month; point estimates at the declared volume.

### A retry lifts pass rate for every model, most where the model is weakest: +7.5 points for Nano-XS · reasoning off · no retry (M), against +0.9 for Atlas-M · reasoning high · no retry (E), the only pairing whose interval reaches zero. At this run's unnamed routing it adds at most $0.077 per thousand calls.

_Caveats: Cost routing unknown._

| Row | Value | 95% unpaired interval |
|---|---|---|
| Atlas-M · reasoning high · no retry → 1 retry | +0.9 | -0.2 to +2.0 |
| Brisk-S · reasoning low · no retry → 1 retry | +1.9 | +0.3 to +3.4 |
| Nano-XS · reasoning off · no retry → 1 retry | +7.5 | +4.6 to +10.4 |

Each effect is one setting moved between two arms that differ in it alone, on Pass rate. Descriptive: read as two arms' own spreads, never a test.

Pass-rate gain from adding a retry, in points, with unpaired intervals, for one configuration per model.

| Row | Value | 95% unpaired interval |
|---|---|---|
| Atlas-M · reasoning off · no retry → 1 retry | +2.4 | +0.7 to +4.1 |
| Brisk-S · reasoning high · no retry → 1 retry | +1.3 | +0.03535 to +2.6 |
| Nano-XS · reasoning low · no retry → 1 retry | +4.8 | +2.4 to +7.1 |

Each effect is one setting moved between two arms that differ in it alone, on Pass rate. Descriptive: read as two arms' own spreads, never a test.

The same gain for another configuration per model; the remaining pairings' rates, without intervals, are in Every arm by measure.

### At the top, neither the bigger model nor more reasoning is told apart on its own: with a retry, Brisk-S trails Atlas-M by 0.8 points at low reasoning, and high reasoning adds +0.8 for Brisk-S; both intervals cross zero. At this run's unnamed routing, Atlas-M costs 5.5 times as much at the same settings.

_Caveats: No arm is the control; Cost routing unknown._

| Panel | Row | Value | 95% unpaired interval |
|---|---|---|---|
| model | Atlas-M · reasoning low · 1 retry → Brisk-S | -0.8 | -2.0 to +0.5 |
| model | Atlas-M · reasoning high · 1 retry → Brisk-S | -0.7 | -1.7 to +0.4 |
| reasoning | Atlas-M · reasoning low · 1 retry → High | +0.7 | -0.3 to +1.7 |
| reasoning | Brisk-S · reasoning low · 1 retry → High | +0.8 | -0.4 to +2.1 |
| model effect, across reasoning | (Atlas-M · reasoning high · 1 retry → Brisk-S) less (Atlas-M · reasoning low · 1 retry → Brisk-S) | +0.1 | -1.5 to +1.7 |

Each effect is one setting moved between two arms that differ in it alone, on Pass rate; each interaction is the same move's effect in one context less its effect in another, the two contexts differing in one other setting. Descriptive: read as two arms' own spreads, never a test.

Effect of switching model or raising reasoning among the retry configurations of Atlas-M and Brisk-S, in points, with unpaired intervals.

### Model and reasoning set the median latency; the retry leaves the median unchanged in every pair shown and lengthens the slow tail, Brisk-S · reasoning low · 1 retry (J) reading 513 ms at p95 against 465 ms without it. No latency budget was declared.

| Group | Row | From | To |
|---|---|---|---|
| Latency | M · Nano-XS · reasoning off · no retry | 63 ms | 140 ms |
| Latency | N · Nano-XS · reasoning off · 1 retry | 63 ms | 198 ms |
| Latency | O · Nano-XS · reasoning low · no retry | 98 ms | 217 ms |
| Latency | Q · Nano-XS · reasoning low · 1 retry | 98 ms | 275 ms |
| Latency | R · Nano-XS · reasoning high · no retry | 170 ms | 378 ms |
| Latency | S · Nano-XS · reasoning high · 1 retry | 170 ms | 450 ms |
| Latency | I · Brisk-S · reasoning low · no retry | 209 ms | 465 ms |
| Latency | J · Brisk-S · reasoning low · 1 retry | 209 ms | 513 ms |
| Latency | K · Brisk-S · reasoning high · no retry | 364 ms | 810 ms |
| Latency | L · Brisk-S · reasoning high · 1 retry | 364 ms | 870 ms |
| Latency | E · Atlas-M · reasoning high · no retry | 1094 ms | 2430 ms |
| Latency | F · Atlas-M · reasoning high · 1 retry | 1094 ms | 2551 ms |

Latency from median to p95 for every frontier configuration beside its twin with or without a retry.

#### Multi-hop questions are the weakest task for each of the three frontier options shown; the lean passes 95.6% there against 98.4% on single lookups.

| Slice | Arm | Pass rate |
|---|---|---|
| Single lookup | S · Nano-XS · reasoning high · 1 retry | 95.8% |
| Single lookup | J · Brisk-S · reasoning low · 1 retry | 98.4% |
| Single lookup | F · Atlas-M · reasoning high · 1 retry | 99.2% |
| Multi-hop (2 to 3 calls) | S · Nano-XS · reasoning high · 1 retry | 93.5% |
| Multi-hop (2 to 3 calls) | J · Brisk-S · reasoning low · 1 retry | 95.6% |
| Multi-hop (2 to 3 calls) | F · Atlas-M · reasoning high · 1 retry | 97.9% |
| Date and unit math | S · Nano-XS · reasoning high · 1 retry | 94.7% |
| Date and unit math | J · Brisk-S · reasoning low · 1 retry | 97.0% |
| Date and unit math | F · Atlas-M · reasoning high · 1 retry | 98.6% |
| Ambiguous entity | S · Nano-XS · reasoning high · 1 retry | 94.2% |
| Ambiguous entity | J · Brisk-S · reasoning low · 1 retry | 96.4% |
| Ambiguous entity | F · Atlas-M · reasoning high · 1 retry | 98.3% |

Pass rate by task; the snapshot states no case count per task, so these rows carry no interval and are descriptive.

#### The lean's kept failures each break differently: a multi-hop question collapsed into one search, a date window off by a day, and an ambiguous name searched instead of asked about. The snapshot does not say how many failures they were kept from.

| Example | Details | Reason | Excerpt |
|---|---|---|---|
| TQ-0418 · Multi-hop (2 to 3 calls) · Brisk-S · reasoning low · 1 retry · missing call | Input: Who chaired the committee that approved the 2019 grid budget?; Expected: search(committee, grid budget 2019) → lookup(chair) | Collapsed two hops into one query; returned the budget author, not the chair. | search(grid budget 2019 chair) |
| TQ-0772 · Date and unit math · Brisk-S · reasoning low · 1 retry · wrong args | Input: Filings in the 90 days before the March 3 hearing; Expected: date_from=2026-12-03, date_to=2027-03-03 | Off-by-one on the window start. The most common wrong-args pattern (9 of 17). | date_from=2026-12-04, date_to=2027-03-03 |
| TQ-1103 · Ambiguous entity · Brisk-S · reasoning low · 1 retry · wrong tool | Input: Latest on Mercury's pricing; Expected: ask_clarify(company or planet?) | Picked the company without asking. Scored wrong because the gold answer requires clarification. | search(Mercury pricing) |

Every failure the snapshot keeps for the lean, one per task where kept; a sample, not a count.

## What you could change

| What you could change | What it decides | What happens as it moves |
|---|---|---|
| What a point of pass rate is worth a month | Which configuration to run | Below $3.20 a point, Nano-XS · reasoning off · no retry (M); $3.20 to $17.31, Nano-XS · reasoning off · 1 retry (N); $17.31 to $111.22, Nano-XS · reasoning low · 1 retry (Q); $111.22 to $319.67, Nano-XS · reasoning high · 1 retry (S); $319.67 to $1,595.18, Brisk-S · reasoning low · 1 retry (J); $1,595.18 to $17,236.36, Brisk-S · reasoning high · 1 retry (L); above that, Atlas-M · reasoning high · 1 retry (F). Point estimates, read with the volume as calls; if it counts tasks, every band grows by the calls per task. |
| A pass-rate floor | The cheapest configuration that meets it | Read on point estimates: a floor up to 94.8% keeps a Nano-XS configuration; up to 98%, a Brisk-S one; above that, only Atlas-M · reasoning high · 1 retry (F). |

## Next steps

- The planner's owner settles what the monthly volume counts, then declares a pass-rate floor or a price per point of pass rate. _Which configuration to run: the explore bands then pick one._
- Re-run Brisk-S at low and high reasoning, both with a retry, at 5,432 cases each; that is the size for independent arms, and fewer suffice if both run the same cases paired. _Whether high reasoning's gain over the lean is real, which a run of today's size reads the wrong way 9.4% of the time._
- Target multi-hop planning on Brisk-S (explicit decomposition of the question into hops) and re-test, recording case counts per task so task figures carry intervals. _Whether the lean's gap to Atlas-M closes on its weakest task._
- Record the provider behind each configuration. _Whether the cost order and the monthly prices hold outside this run._

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

- **No-retry configurations other than the cheapest** Each passes less often than its own retry twin, which costs a little more; the retry claim shows the gain, and their rates are in Every arm by measure.
- **Calls per task** Varies little across configurations and decides nothing; it enters only the volume caveat.
- **Extra calls and outcome measures** The snapshot lists them but holds no figures for either.
- **Model and reasoning effects without a retry, and their interactions** Off the frontier; they don't change which options a decider would weigh. Each arm's rates are in Every arm by measure; the effects' intervals are not on the page.
- **Per-task gaps between configurations** No case counts per task, so no interval; descriptive only and the order matches overall.

### Method

- **Cases** Each of the eighteen configurations ran 1,200 cases with no lost or failed trials; repeats are not stated. Configurations cross three models, three reasoning levels, and a retry or none.
- **Pass** The pipeline's pass or fail grade on each planned query. Failures are classed as a missing call, wrong arguments, the wrong tool or malformed output; the snapshot defines pass no further. On ambiguous entities the expected answer is a clarifying question.
- **Intervals** Read at 95% from each configuration's binomial spread over its cases and combined as if the configurations ran different cases, which errs wide; uncorrected for the number of comparisons. Per-task figures carry none.
- **Cost** Cost per thousand calls as measured, at a provider routing no configuration names. Monthly figures use the declared 4,000,000 a month, whose unit the campaign does not state. _Caveats: Monthly volume's unit unstated._
- **Latency** Median and p95 per configuration; no budget was declared.
- **Not compared** No control and no criteria were declared, so no option is read against a line.
- **Source** Synthetic example data from docs/eval-report-kit/references/r2-config-sweep.json

### Review findings left open

- **Should fix** · Claim 1: Brisk-S at low reasoning without a retry (95.3% at $0.288 per 1k calls) is not dominated: no other configuration is both cheaper and better. It sits above the seven configurations the frontier claim counts, and a pass-rate floor between 94.9% and 95.3% makes it the cheapest option that meets the floor. Yet the headline says 'run every planner with a retry', and the floor row says only 'up to 98%, a Brisk-S one'. Say in the frontier claim or the floor row that the seven are the price-per-point frontier, and that this no-retry configuration is cheapest only for a floor of 94.9% to 95.3%, where it saves about $56 a month against the lean for 1.8 fewer points. Or soften the headline to 'every option on the price-per-point frontier runs with a retry'.
- **Should fix** · The page: The lede compares Brisk-S and Atlas-M costs ('$1,208 a month', '$12,700 more a month') with no routing qualifier. 'The trade' and 'Where the answer flips' in the decision block also price models without one. The confidence line covers 'every price on the page', so a reader is not misled. Still, add 'at this run's unnamed provider routing' to the lede's cost sentence, as the claims and Watch now do. Also, in the cost-routing confidence clause, say why the lean holds until the routing is known: for example, a routing change would have to shift the roughly 2.7 times cost gap between Nano-XS at high reasoning and Brisk-S at low reasoning enough to move the band edges. Watch's '12 times' is 11.5 times on the figures ($3.477 against $0.302).

_Written by claude-opus-5-5, served as claude-opus-5-5, for $3.62, with analyst skill 2a7662d928b0, on 2026-10-29._
