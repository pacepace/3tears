Choice · candidates compared · Research query planner · 18 arms · 4 tasks · Run 2026-10-23 · Synthetic example data

**Draft: its editor's review left 3 findings open.**

# No bar declared, so the pick is a price. Lean: Brisk-S, low reasoning, retry. Atlas-M's best is not told apart from Brisk-S at high reasoning, yet costs 5.5 times as much at this run's routing

Brisk-S at low reasoning with retry trails Atlas-M's best by 1.5 points; on point estimates, the next point past it costs about $1,595.18 a month. A declared bar or price per point from the planner's owner would turn this into a decision.

**Confidence: High, if the caveats below aren't material.**

- **Nothing declares what is good enough.** The high covers, at this run's routing, the frontier's shape, the price per point rising at every step, and Atlas-M's premium over Brisk-S at matched reasoning; it does not cover the lean, which stays open until the owner declares a bar or a price per point. The lean is cheapest only for a price between $319.67 and $1,595.18 a month on point estimates, and that upper edge rests on a gain of +0.8 points (-0.4 to +2.1)
- **No arm is the control.** Only if the owner needs the move from today's planner. Every option here is a candidate, so choosing among them needs no control; no configuration is marked as the one in production, so nothing here reads a change from it
- **The pick is the best of several candidates.** Fresh cases read the lean's lead over Nano-XS at high reasoning with retry near zero; that lead's interval is +0.8 to +4, and the best of many configurations tends to read lower on a re-run, which the paired re-run checks
- **Cost routing and volume unit unknown.** Another provider routing changes the models' prices per call relative to each other, or the declared monthly volume counts tasks rather than calls. The snapshot names no provider for any configuration. Atlas-M's premium over Brisk-S is 5.5 times at low reasoning and 5.5 times at high, so a re-routing would have to cut Atlas-M's price relative to Brisk-S's by about that factor to close it; the price edges between nearer options could move with far smaller shifts. If the volume counts tasks and a call is a tool call, each monthly cost scales by its calls per task (1.539 calls for the lean, 1.452 calls for Atlas-M at high reasoning with retry, 1.566 calls for Nano-XS at high reasoning with retry), which raises every price edge by about that factor and moves the lean's band up; no figure states the per-task prices

**The trade.** Atlas-M at high reasoning with retry passes 1.5 points more than Brisk-S at low reasoning with retry (+0.3 to +2.6), for $12,700 more a month at the declared volume, read as calls, at this run's routing. Brisk-S at high reasoning with retry is not told apart from Atlas-M on pass rate at $0.633 per 1k calls against $3.477. Nothing declared before the run ranks cost against pass rate; the planner's owner makes the call.

**Where the answer flips.** By price per point a month, on point estimates: below $319.67, a Nano-XS configuration; up to $1,595.18, Brisk-S at low reasoning with retry; up to $17,236.36, Brisk-S at high reasoning with retry; above it, Atlas-M at high reasoning with retry. These edges carry no interval, and both upper steps rest on gains not told apart from zero.

**Our lean.** Ours, not a rule: Brisk-S at low reasoning with retry, at $0.302 per 1k calls. It holds for a price per point between $319.67 and $1,595.18 a month.

**Watch.** The next step up, Brisk-S at high reasoning with retry, gains 0.8 points on the lean, not told apart from zero (-0.4 to +2.1); a fresh run of today's size reads it the other way 9.4% of the time. At the top of that interval the price of a point falls well below the lean's upper edge, so the band is soft. The lean is the best of eighteen configurations near its price, so its rate may read lower on fresh cases.

**What would settle it.** A pass-rate bar or a price per point, declared by the planner's owner, with the unit the monthly volume counts. Then a paired re-run of the two Brisk-S options with retry settles whether high reasoning's gain is real.

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

## Along the cost frontier of the eighteen configurations, pass rate climbs steeply across the cheapest and flattens past Brisk-S at low reasoning with retry, at 97.1%.

| Point | Cost per 1k calls | Pass rate |
|---|---|---|
| M · Nano-XS · reasoning off · no retry | $0.030 | 81.0% |
| N · Nano-XS · reasoning off · 1 retry | $0.036 | 88.5% |
| Q · Nano-XS · reasoning low · 1 retry | $0.054 | 92.7% |
| S · Nano-XS · reasoning high · 1 retry | $0.111 | 94.8% |
| J · Brisk-S · reasoning low · 1 retry | $0.302 | 97.1% |
| L · Brisk-S · reasoning high · 1 retry | $0.633 | 98.0% |
| F · Atlas-M · reasoning high · 1 retry | $3.477 | 98.6% |

The horizontal axis is logarithmic: each step is a multiple.

The seven configurations on the frontier, cost per 1k calls on a log axis against pass rate, the lean stressed; every other configuration is in the full results.

### The price of a pass-rate point rises at every step along the frontier, reaching $17,236.36 a month for the last step to Atlas-M.

| Step along the frontier | Price per point |
|---|---|
| Nano-XS · reasoning off · no retry → Nano-XS · reasoning off · 1 retry | $3.20 |
| Nano-XS · reasoning off · 1 retry → Nano-XS · reasoning low · 1 retry | $17.31 |
| Nano-XS · reasoning low · 1 retry → Nano-XS · reasoning high · 1 retry | $111.22 |
| Nano-XS · reasoning high · 1 retry → Brisk-S · reasoning low · 1 retry | $319.67 |
| Brisk-S · reasoning low · 1 retry → Brisk-S · reasoning high · 1 retry | $1,595.18 |
| Brisk-S · reasoning high · 1 retry → Atlas-M · reasoning high · 1 retry | $17,236.36 |

Monthly price of each pass-rate point gained along the frontier, at the declared volume read as calls and this run's routing; point estimates with no interval.

### At this run's unnamed provider routing, Atlas-M costs 5.5 times as much as Brisk-S at the same reasoning level with retry, yet the two are not told apart on pass rate (-1.7 to +0.4 at high reasoning).

| Panel | Row | Value | 95% unpaired interval |
|---|---|---|---|
| model | Atlas-M · reasoning low · 1 retry → Brisk-S | -0.8 | -2.0 to +0.5 |
| model | Atlas-M · reasoning high · 1 retry → Brisk-S | -0.7 | -1.7 to +0.4 |
| reasoning | Atlas-M · reasoning low · 1 retry → High | +0.7 | -0.3 to +1.7 |
| reasoning | Brisk-S · reasoning low · 1 retry → High | +0.8 | -0.4 to +2.1 |
| model effect, across reasoning | (Atlas-M · reasoning high · 1 retry → Brisk-S) less (Atlas-M · reasoning low · 1 retry → Brisk-S) | +0.1 | -1.5 to +1.7 |

Each effect is one setting moved between two arms that differ in it alone, on Pass rate; each interaction is the same move's effect in one context less its effect in another, the two contexts differing in one other setting. Descriptive: read as two arms' own spreads, never a test.

Change in pass rate from switching Atlas-M to Brisk-S, or low to high reasoning, all with retry; unpaired intervals.

| Row | Value |
|---|---|
| D · Atlas-M · reasoning low · 1 retry | $1.66 |
| J · Brisk-S · reasoning low · 1 retry | $0.30 |
| F · Atlas-M · reasoning high · 1 retry | $3.48 |
| L · Brisk-S · reasoning high · 1 retry | $0.63 |

The bars are drawn on a log scale: their values span more than ten times, so each step is a multiple.

Cost per 1k calls for the same four configurations, on a log scale.

### Retry buys most where pass rate starts lowest: on Brisk-S at low reasoning it adds +1.9 points for $0.014 more per 1k calls, with calls per task moving from 1.492 calls to 1.539 calls; it adds more on Nano-XS, and on Atlas-M at high reasoning a gain not told apart from zero.

| Panel | Row | Value | 95% unpaired interval |
|---|---|---|---|
| model | Atlas-M · reasoning high · no retry → Nano-XS | -6.4 | -8.2 to -4.6 |
| model | Atlas-M · reasoning high · 1 retry → Nano-XS | -3.9 | -5.3 to -2.5 |
| retry | Atlas-M · reasoning high · no retry → 1 retry | +0.9 | -0.2 to +2.0 |
| retry | Brisk-S · reasoning low · no retry → 1 retry | +1.9 | +0.3 to +3.4 |
| retry | Nano-XS · reasoning high · no retry → 1 retry | +3.4 | +1.4 to +5.5 |
| model effect, across retry | (Atlas-M · reasoning high · 1 retry → Nano-XS) less (Atlas-M · reasoning high · no retry → Nano-XS) | +2.5 | +0.3 to +4.8 |

Each effect is one setting moved between two arms that differ in it alone, on Pass rate; each interaction is the same move's effect in one context less its effect in another, the two contexts differing in one other setting. Descriptive: read as two arms' own spreads, never a test.

Change in pass rate from adding retry to each shortlisted option, beside the Atlas-M to Nano-XS gap with and without retry; unpaired intervals.

| Row | Value |
|---|---|
| R · Nano-XS · reasoning high · no retry | $0.10 |
| S · Nano-XS · reasoning high · 1 retry | $0.11 |
| I · Brisk-S · reasoning low · no retry | $0.29 |
| J · Brisk-S · reasoning low · 1 retry | $0.30 |
| E · Atlas-M · reasoning high · no retry | $3.40 |
| F · Atlas-M · reasoning high · 1 retry | $3.48 |

The bars are drawn on a log scale: their values span more than ten times, so each step is a multiple.

Cost per 1k calls without and with retry for the same three options, on a log scale.

#### Atlas-M at high reasoning with retry is the slowest option on the shortlist, its p95 latency at 2551 ms; no latency budget was declared, and retry lengthens the tail.

| Group | Row | From | To |
|---|---|---|---|
| Latency | S · Nano-XS · reasoning high · 1 retry | 170 ms | 450 ms |
| Latency | I · Brisk-S · reasoning low · no retry | 209 ms | 465 ms |
| Latency | J · Brisk-S · reasoning low · 1 retry | 209 ms | 513 ms |
| Latency | L · Brisk-S · reasoning high · 1 retry | 364 ms | 870 ms |
| Latency | F · Atlas-M · reasoning high · 1 retry | 1094 ms | 2551 ms |

Latency from p50 to p95 for the options a decider would weigh.

#### Multi-hop questions are the weakest task for every shortlisted option; the lean passes 95.6% of them. With no per-task counts this is a lead, not a separated finding.

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

Pass rate by task for the shortlisted options; point estimates.

| Example | Details | Reason | Excerpt |
|---|---|---|---|
| TQ-0418 · Multi-hop (2 to 3 calls) · Brisk-S · reasoning low · 1 retry · missing call | Input: Who chaired the committee that approved the 2019 grid budget?; Expected: search(committee, grid budget 2019) → lookup(chair) | Collapsed two hops into one query; returned the budget author, not the chair. | search(grid budget 2019 chair) |
| TQ-0772 · Date and unit math · Brisk-S · reasoning low · 1 retry · wrong args | Input: Filings in the 90 days before the March 3 hearing; Expected: date_from=2026-12-03, date_to=2027-03-03 | Off-by-one on the window start. The most common wrong-args pattern (9 of 17). | date_from=2026-12-04, date_to=2027-03-03 |
| TQ-1103 · Ambiguous entity · Brisk-S · reasoning low · 1 retry · wrong tool | Input: Latest on Mercury's pricing; Expected: ask_clarify(company or planet?) | Picked the company without asking. Scored wrong because the gold answer requires clarification. | search(Mercury pricing) |

The failures kept for the lean, one per task shown; the multi-hop card collapses two hops into one query. Illustrations, not a count.

## What you could change

| What you could change | What it decides | What happens as it moves |
|---|---|---|
| Price of a pass-rate point a month | Which frontier configuration is worth its cost | On point estimates, with the volume read as calls: below $319.67, a Nano-XS option; up to $1,595.18, Brisk-S at low reasoning with retry; up to $17,236.36, Brisk-S at high reasoning with retry; above, Atlas-M at high reasoning with retry. If the volume counts tasks, each edge rises by about the configurations' calls per task. |
| A pass-rate bar | The cheapest configuration that clears it, read on point estimates | Up to 94.8%, Nano-XS at high reasoning with retry; up to 95.3%, Brisk-S at low reasoning without retry; up to 97.1%, the lean; up to 98%, Brisk-S at high reasoning with retry; up to 98.6%, Atlas-M at high reasoning with retry; above that, none. |

## Next steps

- The planner's owner declares a pass-rate bar or a price per point of pass rate, and states what the monthly volume counts. _Turns the trade into a decision: the cheapest configuration that meets it, priced on the right unit._
- Re-run Brisk-S at low and at high reasoning, both with retry, on the same fresh cases, 5,432 per configuration. _Whether high reasoning's gain on Brisk-S is real, four times in five if today's gap holds; also checks the lean's rate against best-of-many optimism._
- Target hop-collapsing on multi-hop questions in the planner prompt, then re-run the lean on that task with stated case counts. _Whether the lean's weakest task can close on the costlier options without paying for them._
- Record each configuration's provider. _Whether the cost gaps, the price edges and Atlas-M's premium hold at production routing._

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

- **Atlas-M at no and low reasoning** Each costs more than a Brisk-S option that passes at least as often; none is on the frontier.
- **Calls per task as its own claim** Cost is read per call. Nano-XS and retry make more calls per task and Atlas-M fewer, so per task Nano-XS's cost advantage narrows a little and Atlas-M's premium shrinks a little; the shortlist order holds. How the volume's unit moves the price edges is carried in the confidence.
- **Outcome and extra-calls measures** The snapshot holds no figures for either.
- **The full lever table across all eighteen configurations** Shown only where it bears on the shortlist; the appendix holds every effect and interaction.
- **Per-task pass rates for configurations off the shortlist** Multi-hop is lowest throughout; the extra rows add no new pattern and carry no counts.

### Method

- **Configurations** eighteen configurations: three models crossed with three reasoning levels and retry on or off. With retry means 1 retry.
- **Cases** 1,200 cases per configuration, none lost; repeats are not stated, nor whether every configuration ran the same cases.
- **Pass** The input does not define what makes a task pass; failed cases are classed as missing call, wrong arguments or wrong tool.
- **Intervals** At 95%, from each rate's binomial spread over its cases, two configurations combined as independent, with no correction for the many comparisons. Where the configurations ran the same cases, a paired interval would be narrower. Task figures carry no counts and so no interval.
- **Control** No configuration is marked as production; every comparison is between two configurations' own figures. _Caveats: No arm is the control._
- **Cost** Cost per 1k calls as measured in this run; the provider routing is not named, and the input does not say whether a call is a planner request or a tool call. Monthly figures read the declared volume of 4,000,000 as calls, which the input does not state.
- **Latency** Median (p50) and p95 per configuration; no latency budget was declared.
- **Source** Synthetic example data from docs/eval-report-kit/references/r2-config-sweep.json, run 2026-10-23.

### Review findings left open

- **Must fix** · The page: The confidence clause on the pick being best of several says 'Fresh cases read the lean's lead over Nano-XS at high reasoning with retry near zero; that lead's interval is +0.8 to +4'. No fresh-case reading exists. The measured lead is +2.4 points (+0.8 to +4). A reader would think a re-run had erased the lean's edge over the option $764 a month cheaper. Rewrite it as a risk, not a result. For example: 'The lean's lead over Nano-XS at high reasoning with retry is +2.4 points (+0.8 to +4); as the best of eighteen, it may read lower on fresh cases, and the paired re-run checks it.'
- **Should fix** · Claim 2: 'The price of a pass-rate point rises at every step along the frontier' is true by construction. The frontier here is the convex hull, so its steps always get dearer, and the confidence's High covering 'the price per point rising at every step' claims nothing the data could have refuted. Say the frontier is the price-per-point hull and that its steps rise by construction. Put the finding in the size of the steps, for example the jump from $1,595.18 to $17,236.36 for the last step to Atlas-M. Then drop 'rising at every step' from what the High covers. In the frontier claim's caption, note that Brisk-S at low reasoning without retry (95.3% at $0.288) is not beaten by any cheaper option but lies under the hull. It is the pick for a bar between 94.8% and 95.3%, as the bar row in 'What you could change' already says.
- **Should fix** · Claim 6: The examples caption says 'one per task shown', but there are three cards for four tasks and none for single lookup. The snapshot keeps only three failures for the lean. Say 'one per task for the three tasks with a kept failure', so a reader doesn't look for a missing single-lookup card.

_Written by claude-opus-5-5, served as claude-opus-5-5, for $4.50, with analyst skill ed7e2c60b5ad, on 2026-10-29._
