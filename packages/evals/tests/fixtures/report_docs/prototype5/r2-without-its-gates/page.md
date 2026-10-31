Choice · candidates compared · Research query planner · 18 arms · 4 tasks · Run 2026-10-23 · Synthetic example data

_4 review findings open, listed at the end._

# Turn on a retry whichever planner is chosen; beyond Brisk-S at low reasoning, each extra point of pass rate costs far more and this run can't tell it apart. Nobody declared what a point is worth.

A retry raised the pass-rate estimate in every model and reasoning pairing, clear of zero in all but Atlas-M at high reasoning, for cents per thousand calls. Below $319.67 a month per point a Nano-XS setup is the best buy; above $1,595.18, Brisk-S at high reasoning.

**Confidence: High, if the caveats below aren't material.**

- **Nothing declares what is good enough.** It decides the pick. The high applies to the retry finding and to which configurations form the frontier; the lean is a low-confidence call that holds only for a price per point between $319.67 and $1,595.18 a month, and a declared price outside that band moves it.
- **No arm is the control.** Only if pairing would reorder close neighbours. Every interval here is read as independent, wider than a paired one; pairing could separate the two Brisk-S retry configurations. That would confirm the upper step's gain is real but not move the lean, which rests on the price band, not on whether a step clears zero.
- **The pick is the best of several candidates.** Where the lean leads its neighbour by little: the best of eighteen configurations reads high by chance, so a fresh run would likely read the lean lower. Its lead over Nano-XS at high reasoning with a retry is +0.8 to +4 points, so chance alone is unlikely to reverse it.
- **Cost routing unknown.** Only if another provider's prices would reorder the models. The input names no provider for any model, so every cost and price here belongs to this run's routing, not to the models.
- **Volume unit unnamed.** Only for the monthly figures and the band edges. The declared 4,000,000 a month names no unit; if it counts tasks rather than calls, every monthly cost and price per point scales by calls per task (1.566 calls for Nano-XS at high reasoning with a retry, 1.539 calls for the lean, 1.476 calls for Brisk-S at high reasoning), raising both edges of the lean's band. Costs per thousand calls and which configurations sit on the frontier do not change.

**The trade.** Nano-XS · reasoning high · 1 retry (S) costs $444 a month; Brisk-S · reasoning low · 1 retry (J) costs $1,208 and passes 2.4 points more, a gap clear of zero. Every step above that buys gains this run cannot separate from zero.

**Where the answer flips.** On point prices with no interval: below $319.67 a month per point, a Nano-XS configuration; up to $1,595.18, Brisk-S · reasoning low · 1 retry (J); above that, Brisk-S · reasoning high · 1 retry (L); Atlas-M only above $17,236.36.

**Our lean.** Ours, not a declared rule, and low-confidence: Brisk-S · reasoning low · 1 retry (J), the best buy while a point is worth between the two prices above. The step above it adds 0.8 points, interval -0.4 to +2.1, for $1,324 more a month; a rerun that separates the two would confirm that gain, and the price per point would still decide.

**Watch.** A retry lengthens the latency tail, not the median: on Brisk-S at low reasoning, p95 goes from 465 ms to 513 ms while p50 stays 209 ms. No latency budget was declared, so this binds nothing yet.

**What would settle it.** The planner owner declares a price per point of pass rate or a pass-rate floor. Telling the two Brisk-S configurations apart four times in five takes about 5,432 cases per arm; a run of today's size reads them the wrong way round 9.4% of the time.

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

## Brisk-S holds the middle of the frontier: every Atlas-M configuration but the most expensive is matched or beaten by a cheaper Brisk-S one.

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
| Brisk-S · reasoning high · 1 retry | $0.633 | 98.0% |
| Nano-XS · reasoning off · no retry | $0.030 | 81.0% |
| Nano-XS · reasoning off · 1 retry | $0.036 | 88.5% |
| Nano-XS · reasoning low · no retry | $0.048 | 87.9% |
| Nano-XS · reasoning low · 1 retry | $0.054 | 92.7% |
| Nano-XS · reasoning high · no retry | $0.102 | 91.3% |
| Nano-XS · reasoning high · 1 retry | $0.111 | 94.8% |

The horizontal axis is logarithmic: each step is a multiple.

Pass rate against cost per thousand calls, log cost axis, for all eighteen configurations; seven are the best buy at some price per point. Brisk-S · reasoning low · no retry (I) is beaten by nothing cheaper but is never the best buy at any single price.

## A retry buys more pass rate per dollar than more reasoning: on Brisk-S at low reasoning it adds 1.9 points, clear of zero, while cost per thousand calls moves only from $0.288 to $0.302; moving on to high reasoning adds 0.8, not clear of zero, and raises it to $0.633.

| Group | Row | Value | 95% unpaired interval |
|---|---|---|---|
| reasoning | Brisk-S · reasoning off · no retry → Low | +4.2 | +2.2 to +6.2 |
| reasoning | Brisk-S · reasoning off · no retry → High | +5.5 | +3.6 to +7.4 |
| reasoning | Brisk-S · reasoning off · 1 retry → Low | +2.5 | +0.9 to +4.1 |
| reasoning | Brisk-S · reasoning off · 1 retry → High | +3.3 | +1.8 to +4.9 |
| reasoning | Brisk-S · reasoning low · no retry → High | +1.4 | -0.2 to +2.9 |
| reasoning | Brisk-S · reasoning low · 1 retry → High | +0.8 | -0.4 to +2.1 |
| retry | Brisk-S · reasoning off · no retry → 1 retry | +3.5 | +1.5 to +5.6 |
| retry | Brisk-S · reasoning low · no retry → 1 retry | +1.9 | +0.3 to +3.4 |
| retry | Brisk-S · reasoning high · no retry → 1 retry | +1.3 | +0.03535 to +2.6 |
| reasoning effect, across retry | (Brisk-S · reasoning off · 1 retry → Low) less (Brisk-S · reasoning off · no retry → Low) | -1.7 | -4.2 to +0.9 |
| reasoning effect, across retry | (Brisk-S · reasoning off · 1 retry → High) less (Brisk-S · reasoning off · no retry → High) | -2.2 | -4.6 to +0.2 |
| reasoning effect, across retry | (Brisk-S · reasoning low · 1 retry → High) less (Brisk-S · reasoning low · no retry → High) | -0.5 | -2.5 to +1.5 |

Each effect is one setting moved between two arms that differ in it alone, on Pass rate; each interaction is the same move's effect in one context less its effect in another, the two contexts differing in one other setting. Descriptive: read as two arms' own spreads, never a test.

Within Brisk-S, the pass-rate change from moving one setting, in points, with intervals read as independent arms. At high reasoning a retry's interval only just clears zero at its low end. In the last rows reasoning's point estimate is smaller once a retry is on, but none of those differences is told apart from zero.

| Arm | Cost per 1k calls |
|---|---|
| M · Nano-XS · reasoning off · no retry | $0.030 |
| N · Nano-XS · reasoning off · 1 retry | $0.036 |
| G · Brisk-S · reasoning off · no retry | $0.180 |
| H · Brisk-S · reasoning off · 1 retry | $0.196 |
| I · Brisk-S · reasoning low · no retry | $0.288 |
| J · Brisk-S · reasoning low · 1 retry | $0.302 |
| K · Brisk-S · reasoning high · no retry | $0.612 |
| L · Brisk-S · reasoning high · 1 retry | $0.633 |

Cost per thousand calls for each Brisk-S configuration, and for Nano-XS with reasoning off, with and without a retry: a retry adds little, higher reasoning a lot.

### The retry gain is largest on the smallest model: 7.5 points for Nano-XS with reasoning off against 2.4 for Atlas-M, a difference clear of zero. On Atlas-M at high reasoning the retry gain is not clear of zero (-0.2 to +2).

| Setting | Move | Difference | Interval | Level | Verdict |
|---|---|---|---|---|---|
| model | Atlas-M · reasoning off · no retry → Nano-XS · reasoning off · no retry | -13.0 | -15.6 to -10.4 | 95% | — |
| model | Atlas-M · reasoning off · 1 retry → Nano-XS · reasoning off · 1 retry | -7.8 | -9.9 to -5.7 | 95% | — |
| retry | Atlas-M · reasoning off · no retry → Atlas-M · reasoning off · 1 retry | +2.4 | +0.7 to +4.1 | 95% | — |
| retry | Nano-XS · reasoning off · no retry → Nano-XS · reasoning off · 1 retry | +7.5 | +4.6 to +10.4 | 95% | — |
| model × retry | (Atlas-M · reasoning off · 1 retry → Nano-XS) less (Atlas-M · reasoning off · no retry → Nano-XS) | +5.1 | +1.8 to +8.5 | 95% | differs |

Retry and model effects among the reasoning-off configurations of Atlas-M and Nano-XS, in points, with intervals read as independent arms.

| Row | Value | 95% unpaired interval |
|---|---|---|
| Overall | +0.9 | -0.2 to +2 |

Each row is Atlas-M · reasoning high · 1 retry less Atlas-M · reasoning high · no retry on Pass rate, overall and in each place both hold it.

The interval is independent: each arm's own spread (a pass/fail rate's binomial spread, from the arm's cases), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same.

Atlas-M at high reasoning, with a retry less without, pass-rate points, interval read as independent arms.

### The price of a point climbs steeply along the frontier, from $3.20 a month for Nano-XS's retry to $17,236.36 for the last step to Atlas-M.

| Step along the frontier | Price per point |
|---|---|
| Nano-XS · reasoning off · no retry → Nano-XS · reasoning off · 1 retry | $3.20 |
| Nano-XS · reasoning off · 1 retry → Nano-XS · reasoning low · 1 retry | $17.31 |
| Nano-XS · reasoning low · 1 retry → Nano-XS · reasoning high · 1 retry | $111.22 |
| Nano-XS · reasoning high · 1 retry → Brisk-S · reasoning low · 1 retry | $319.67 |
| Brisk-S · reasoning low · 1 retry → Brisk-S · reasoning high · 1 retry | $1,595.18 |
| Brisk-S · reasoning high · 1 retry → Atlas-M · reasoning high · 1 retry | $17,236.36 |

Monthly price per extra point of pass rate at the declared 4,000,000 a month, a volume whose unit the input does not name, for each step between neighbouring best-buy configurations; point prices, no interval.

### Atlas-M's best configuration is not told apart from Brisk-S's best on pass rate and costs $3.477 per thousand calls against $0.633.

| Row | Value | 95% unpaired interval |
|---|---|---|
| Overall | +0.7 | -0.4 to +1.7 |

Each row is Atlas-M · reasoning high · 1 retry less Brisk-S · reasoning high · 1 retry on Pass rate, overall and in each place both hold it.

The interval is independent: each arm's own spread (a pass/fail rate's binomial spread, from the arm's cases), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same.

Atlas-M less Brisk-S, both at high reasoning with a retry, pass-rate points, interval read as independent arms.

| Arm | Cost per 1k calls |
|---|---|
| F · Atlas-M · reasoning high · 1 retry | $3.48 |
| L · Brisk-S · reasoning high · 1 retry | $0.63 |

Cost per thousand calls for the two, rounded to the cent, at this run's routing.

#### Multi-hop (2 to 3 calls) is the weakest task for each configuration shown, and high reasoning's gain on Brisk-S is largest there: 1.7 points against 0.1 on single lookups.

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

Pass rate by task for four best-buy configurations; the task figures state no case counts, so they carry no interval and read as descriptive leads.

#### Reasoning and model size move the whole latency range; a retry stretches only the tail, and every Atlas-M configuration is slower at p95 than any Brisk-S one.

| Group | Row | From | To |
|---|---|---|---|
| Latency | A · Atlas-M · reasoning off · no retry | 405 ms | 900 ms |
| Latency | B · Atlas-M · reasoning off · 1 retry | 405 ms | 1019 ms |
| Latency | C · Atlas-M · reasoning low · no retry | 628 ms | 1395 ms |
| Latency | D · Atlas-M · reasoning low · 1 retry | 628 ms | 1500 ms |
| Latency | E · Atlas-M · reasoning high · no retry | 1094 ms | 2430 ms |
| Latency | F · Atlas-M · reasoning high · 1 retry | 1094 ms | 2551 ms |
| Latency | G · Brisk-S · reasoning off · no retry | 135 ms | 300 ms |
| Latency | H · Brisk-S · reasoning off · 1 retry | 135 ms | 359 ms |
| Latency | I · Brisk-S · reasoning low · no retry | 209 ms | 465 ms |
| Latency | J · Brisk-S · reasoning low · 1 retry | 209 ms | 513 ms |
| Latency | K · Brisk-S · reasoning high · no retry | 364 ms | 810 ms |
| Latency | L · Brisk-S · reasoning high · 1 retry | 364 ms | 870 ms |
| Latency | M · Nano-XS · reasoning off · no retry | 63 ms | 140 ms |
| Latency | N · Nano-XS · reasoning off · 1 retry | 63 ms | 198 ms |
| Latency | O · Nano-XS · reasoning low · no retry | 98 ms | 217 ms |
| Latency | Q · Nano-XS · reasoning low · 1 retry | 98 ms | 275 ms |
| Latency | R · Nano-XS · reasoning high · no retry | 170 ms | 378 ms |
| Latency | S · Nano-XS · reasoning high · 1 retry | 170 ms | 450 ms |

Latency from p50 to p95 for each configuration, in milliseconds; no budget was declared.

#### The few kept failure cards show what failures look like, not how often each kind occurs. The lean configuration's cards each fail differently: the first collapses two hops into one query, the second misplaces a date window by a day, the third guesses an entity where the expected answer asks. The fourth, from Nano-XS at low reasoning with a retry already on, is malformed output that survived the retry; its own note says a retry recovers most such cases.

| Example | Details | Reason | Excerpt |
|---|---|---|---|
| TQ-0418 · Multi-hop (2 to 3 calls) · Brisk-S · reasoning low · 1 retry · missing call | Input: Who chaired the committee that approved the 2019 grid budget?; Expected: search(committee, grid budget 2019) → lookup(chair) | Collapsed two hops into one query; returned the budget author, not the chair. | search(grid budget 2019 chair) |
| TQ-0772 · Date and unit math · Brisk-S · reasoning low · 1 retry · wrong args | Input: Filings in the 90 days before the March 3 hearing; Expected: date_from=2026-12-03, date_to=2027-03-03 | Off-by-one on the window start. The most common wrong-args pattern (9 of 17). | date_from=2026-12-04, date_to=2027-03-03 |
| TQ-1103 · Ambiguous entity · Brisk-S · reasoning low · 1 retry · wrong tool | Input: Latest on Mercury's pricing; Expected: ask_clarify(company or planet?) | Picked the company without asking. Scored wrong because the gold answer requires clarification. | search(Mercury pricing) |
| TQ-0051 · Single lookup · Nano-XS · reasoning low · 1 retry · malformed | Input: Population of Bozeman; Expected · tool: lookup; Expected · entity: Bozeman, MT | Truncated JSON. Retry recovers 70% of these. | {"tool":"lookup","entity":"Bozeman, MT" |

Every failure card the snapshot kept; how many failures each configuration had in all is not stated.

## What you could change

| What you could change | What it decides | What happens as it moves |
|---|---|---|
| Price of a point of pass rate per month | Which configuration is the best buy | Below $319.67 a Nano-XS configuration; from there to $1,595.18, Brisk-S · reasoning low · 1 retry (J); to $17,236.36, Brisk-S · reasoning high · 1 retry (L); above it, Atlas-M · reasoning high · 1 retry (F). These edges assume the monthly volume counts calls. |
| A pass-rate floor | Which configurations are acceptable at all | A floor above 94.8% rules out every Nano-XS configuration, and the cheapest left is Brisk-S · reasoning low · no retry (I) at 95.3%; above 98%, only Atlas-M · reasoning high · 1 retry (F) remains. |
| A p95 latency budget | Whether Atlas-M or high reasoning can be used | A budget under 870 ms rules out every Atlas-M configuration and Brisk-S at high reasoning with a retry; the lean fits any budget from 513 ms up. |

## Next steps

- The planner owner declares a price per point of pass rate, or a pass-rate floor, and a p95 latency budget. _Turns the priced trade into a decision among the best-buy configurations._
- Rerun Brisk-S · reasoning low · 1 retry (J) against Brisk-S · reasoning high · 1 retry (L) on the same fresh cases, at about 5,432 cases per arm (fewer if read paired). _Whether high reasoning's gain on Brisk-S is real, so that a price above the upper band edge buys a confirmed gain; a fresh run also checks the lean against chance among many configurations._
- Run Brisk-S with high reasoning on multi-hop questions and low reasoning elsewhere, with a retry, as a measured configuration. _Whether a mix keeps most of high reasoning's gain for part of its cost; nobody has run it, so no figure prices it yet._
- Record which provider served each model and what the monthly volume counts. _Whether the costs belong to the models or to this run's routing, and where the band edges really sit._

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

- **Reasoning and model effects for Atlas-M and Nano-XS beyond the rows shown** They point the same way as Brisk-S's and don't change the frontier; the appendix holds all of them.
- **Calls per task as a claim** It moves within a narrow range and changes no per-thousand cost order; its one bearing, scaling every monthly figure and both band edges if the volume counts tasks, is carried in the confidence.
- **Outcome and extra-calls measures** The snapshot holds no figures for them.
- **Configurations that are never the best buy** Each is beaten by a cheaper one or, like Brisk-S at low reasoning with no retry, is never the best buy at any single price per point; that one matters only under a pass-rate floor, which the explore row names. All appear in the frontier chart and the appendix.
- **Per-task intervals and per-task routing** Task figures state no case counts, and no task carries a traffic share, so no routed mix can be read or priced.

### Method

- **Design** A full grid of three models, three reasoning levels and retry on or off: eighteen configurations, no control. Each ran 1,200 cases, none lost; repeats are not stated.
- **Pass** The input does not state what makes a trial pass. Failure cards name missing calls, wrong arguments, wrong tools and malformed output as failure classes.
- **Intervals** At 95%, computed from each pass rate's binomial spread over its cases, two configurations' spreads combined as if independent; no correction for the many comparisons. Task figures state no counts and carry no interval.
- **Best buy and frontier** A configuration is a best buy if, at some price per point of pass rate, no other gives more pass rate for its cost. The lean is read on that price band alone, never on whether a step's gain clears zero.
- **What we weighed and kept** The headline keeps 'turn on a retry' for every planner although on Atlas-M at high reasoning its gain is not clear of zero: its estimate is positive there and its cost small, and the page says so beside the figure. The confidence stays high only for the retry finding and the frontier's shape; the lean is stated as low-confidence and conditional on a price per point. Interactions between retry and reasoning are described by their estimates and never called real, since none is told apart from zero.
- **Cost** Cost per thousand calls as this run measured it; monthly figures at the declared 4,000,000, a volume whose unit the input does not name. No provider is named for any model.
- **Prices per point** Cost gap over pass-rate gain between neighbouring best-buy configurations, on point estimates; no interval.
- **Latency** p50 and p95 per configuration, in milliseconds; no budget declared.
- **Source** Synthetic example data from docs/eval-report-kit/references/r2-config-sweep.json, run 2026-10-23.

### Review findings left open

- **Should fix** · Claim 2: In the retry claim's diverging bars, the row 'Brisk-S · reasoning high · no retry → 1 retry' still prints its interval as +0.03535 to +2.6. Every other row uses one decimal. Print it as +0.04 to +2.6 so readers see a gain that only just clears zero, not a precision artefact. The other two parts of the earlier finding (the interaction caption and the three-decimal cost table) are now fixed.
- **Should fix** · Claim 5: The sentence cites $3.477 against $0.633 per thousand calls, but the block's cost table prints $3.48 and $0.63. Print three decimals in the table, as the retry claim's table and the options table do, so the block shows the figures the sentence quotes.
- **Should fix** · The page: The confidence line applies 'high' to 'which configurations form the frontier'. Brisk-S at high reasoning with a retry is on the frontier only through a +0.8 (-0.4 to +2.1) gain over the lean. Atlas-M at high reasoning with a retry is on it only through a +0.7 (-0.4 to +1.7) gain over Brisk-S, and the page says neither can be told from zero. Narrow the high to what holds at any value inside those intervals: Nano-XS holds the cheap end, Brisk-S at low reasoning with a retry clears Nano-XS (+0.8 to +4), and Brisk-S matches or beats every Atlas-M configuration but the top one. Then say the top two frontier members rest on point estimates.
- **Should fix** · Claim 6: The sentence says high reasoning's gain on Brisk-S is 'largest' on multi-hop questions (1.7 points against 0.1). The task figures state no case counts, so neither gain has an interval and the ranking across tasks is untested. Only the caption says so. Put the hedge in the sentence itself, e.g. 'on point estimates with no interval', since the next step's routed mix is motivated by this reading.

_Written by claude-opus-5-5, served as claude-opus-5-5, for $4.74, with analyst skill db31bf62aa68, on 2026-10-31._
