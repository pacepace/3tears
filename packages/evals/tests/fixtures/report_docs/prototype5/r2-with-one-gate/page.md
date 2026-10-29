Choice · candidates compared · Research query planner · 18 arms · 4 tasks · Run 2026-10-23 · Synthetic example data

**Draft: its editor's review left 1 finding open.**

# Use Brisk-S · reasoning low · 1 retry (J): the cheapest configuration to reach the 97% pass-rate gate, though it clears it by only 0.1 points.

It costs $0.302 per 1,000 calls as routed in this run, and the next passing option costs 2.1 times as much. A fresh run of today's size would read it below the gate 38.6% of the time; then Brisk-S · reasoning high · 1 retry (L) clears with room for $1,324 more a month.

**The confidence covers the cost order on declared point estimates only. Whether the pick truly clears the gate is open, near even odds, and the fresh read in Next settles it.**

**Confidence: High, if the caveats below aren't material.**

- **Pass rate is cleared on a point estimate.** Only if the pick's true pass rate sits below the gate, which today's figures leave open: its computed interval, 96.2–98.1%, crosses the line, and a fresh run of today's size reads it below 38.6% of the time. The high covers only that the pick is the cheapest configuration passing on point estimates as declared, with only Brisk-S · reasoning high · 1 retry (L) close behind; it does not cover whether the pick truly clears the gate. The fresh read in Next settles that, and a read below makes Brisk-S · reasoning high · 1 retry (L), whose interval 97.2–98.8% sits wholly above the line, the pick.
- **The pick is the best of several candidates.** Only if the pick's clearance is partly luck: it is the cheapest of eighteen configurations read against one line, so a fresh run tends to read it lower. This pushes the same way as the caveat above, toward Brisk-S · reasoning high · 1 retry (L), never toward a cheaper option.
- **No arm is the control.** Only for a claim that ranks two passing configurations against each other: the gap between Brisk-S · reasoning high · 1 retry (L) and the pick is read unpaired, and its interval, -0.4 to +2.1, crosses zero. The call reads each option against the gate, not against each other.
- **Provider routing unknown.** Only if another provider's prices reorder the models: no arm names who served it, so the cost order between Brisk-S, Atlas-M and Nano-XS holds for this run's routing. The pick and its fallback share one model and one routing, so the gap between them would likely hold.

**Recommendation.** Ship Brisk-S · reasoning low · 1 retry (J) on all traffic: of the five configurations that reach the gate on their point estimates it costs least, at $1,208 a month at the declared volume and this run's routing.

**Why not the others.** Brisk-S · reasoning high · 1 retry (L) clears the gate with room but costs 2.1 times as much. As routed in this run, with no provider recorded, every passing Atlas-M configuration costs more again, the cheapest 5.5 times as much as the pick. No Nano-XS configuration reaches the gate; the best reads 94.8%.

**Watch.** The weakest margin is the pick's own: 97.1% against 97%, 0.1 points clear, read as the figure as measured, on no interval. A run that settles it four times in five, read on a two-sided interval, needs 111,255 cases. If a fresh read puts it below the gate, move to Brisk-S · reasoning high · 1 retry (L), which a run of today's size reads on the wrong side only 0.9% of the time.

**Next.** Ship now and confirm in parallel: read the pick's pass rate on 111,255 fresh cases or live tasks, and switch to Brisk-S · reasoning high · 1 retry (L) if it lands below the gate.

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

## Of eighteen configurations, five reach the 97% gate on pass rate, and Brisk-S · reasoning low · 1 retry (J) costs least among them, at $0.302 per 1,000 calls as routed in this run; no provider is recorded for any arm, so the cost order across models is this routing's, not the models' own.

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
| Pass rate gate, at least 97% |  | 97.0% |

The horizontal axis is logarithmic: each step is a multiple.

Cost per 1,000 calls at this run's routing, on a log scale, against pass rate for every configuration, with the gate's line; the pick is marked.

### The pick clears the gate on its point estimate alone: its computed interval, 96.2–98.1%, crosses the line, while Brisk-S · reasoning high · 1 retry (L) sits wholly above it at 97.2–98.8%.

| Row | Value | 95% computed interval |
|---|---|---|
| J · Brisk-S · reasoning low · 1 retry | 97.1% | 96.2% to 98.1% |
| L · Brisk-S · reasoning high · 1 retry | 98.0% | 97.2% to 98.8% |
| D · Atlas-M · reasoning low · 1 retry | 97.9% | 97.1% to 98.7% |
| E · Atlas-M · reasoning high · no retry | 97.7% | 96.9% to 98.6% |
| F · Atlas-M · reasoning high · 1 retry | 98.6% | 98.0% to 99.3% |
| K · Brisk-S · reasoning high · no retry | 96.6% | 95.6% to 97.7% |
| C · Atlas-M · reasoning low · no retry | 96.6% | 95.5% to 97.6% |
| Pass rate gate, at least 97% | 97.0% |  |

Each interval is computed here from the rate's binomial spread, by the normal approximation, held to 0 to 100%. A rate of none or all has none.

Pass rate with its computed interval for the passing configurations and the two nearest misses, against the gate.

### A fresh run of today's size would read the pick below the gate 38.6% of the time, against 0.9% for Brisk-S · reasoning high · 1 retry (L). The table's sizes settle each reading only half the time; sized to settle it four times in five on a two-sided interval, a run needs 111,255 cases for the pick and 1,660 for the fallback.

| Arm | Against | Today's figure | Line | Distance toward passing | Settles | Sizes count | Needed, one-sided | Needed, two-sided | Today | A fresh run reads the other side | Level |
|---|---|---|---|---|---|---|---|---|---|---|---|
| J · Brisk-S · reasoning low · 1 retry | Pass rate | 97.1% | 97.0% | +0.1 | clears | cases | 38,350 | 54,451 | 1,200 | 38.6% | 95% one-sided |
| L · Brisk-S · reasoning high · 1 retry | Pass rate | 98.0% | 97.0% | +1.0 | clears | cases | 572 | 812 | 1,200 | 0.9% | 95% one-sided |

In each row, "A fresh run reads the other side" is how often a fresh run of today's size would read the figure on the other side of its line, if the truth is today's estimate (95% one-sided); it plans a run and tests nothing.

Distance to the gate, the cases a run needs to settle it half the time, one- and two-sided, and how often a run of today's size reads the other side; the four-in-five sizes are in the sentence above.

### Headroom over the gate is priced on gains not told apart from zero: moving from the pick to Brisk-S · reasoning high · 1 retry (L) gains +0.8 points, interval -0.4 to +2.1, at $1,595.18 a month per point; the step beyond, to an Atlas-M configuration as routed in this run with no provider recorded, costs $17,236.36 per point. Neither price has an upper end the data support.

| Step along the frontier | Price per point |
|---|---|
| Brisk-S · reasoning low · 1 retry → Brisk-S · reasoning high · 1 retry | $1,595 |
| Brisk-S · reasoning high · 1 retry → Atlas-M · reasoning high · 1 retry | $17,236 |

Each step along the cost and pass-rate frontier, priced per point a month at the declared volume, on point estimates; the second step crosses models at this run's unrecorded routing.

| Row | Value | 95% unpaired interval |
|---|---|---|
| Overall | +0.8 | -0.4 to +2.1 |
| Single lookup | +0.1 |  |
| Multi-hop (2 to 3 calls) | +1.7 |  |
| Date and unit math | +0.9 |  |
| Ambiguous entity | +1.2 |  |

Each row is Brisk-S · reasoning high · 1 retry less Brisk-S · reasoning low · 1 retry on Pass rate, overall and in each place both hold it.

The interval is independent: each arm's own spread (a pass/fail rate's binomial spread, from the arm's cases), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same.

The first step's gain: the high-reasoning fallback less the pick, overall with an unpaired interval that crosses zero, and by task with none.

#### The step to Atlas-M · reasoning high · 1 retry (F) gains +0.7 points over the fallback, interval -0.4 to +1.7, also not told apart from zero.

| Row | Value | 95% unpaired interval |
|---|---|---|
| Overall | +0.7 | -0.4 to +1.7 |
| Single lookup | +0.7 |  |
| Multi-hop (2 to 3 calls) | +0.6 |  |
| Date and unit math | +0.7 |  |
| Ambiguous entity | +0.7 |  |

Each row is Atlas-M · reasoning high · 1 retry less Brisk-S · reasoning high · 1 retry on Pass rate, overall and in each place both hold it.

The interval is independent: each arm's own spread (a pass/fail rate's binomial spread, from the arm's cases), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same.

The second step's gain: the Atlas-M configuration less the fallback, overall with an unpaired interval that crosses zero, and by task with none.

#### The pick reads lowest on multi-hop questions, at 95.6% against 97.3% for the fallback; task figures state no counts, so this ordering is descriptive, a lead for where a fix might aim.

| Slice | Arm | Pass rate |
|---|---|---|
| Single lookup | J · Brisk-S · reasoning low · 1 retry | 98.4% |
| Single lookup | L · Brisk-S · reasoning high · 1 retry | 98.5% |
| Multi-hop (2 to 3 calls) | J · Brisk-S · reasoning low · 1 retry | 95.6% |
| Multi-hop (2 to 3 calls) | L · Brisk-S · reasoning high · 1 retry | 97.3% |
| Date and unit math | J · Brisk-S · reasoning low · 1 retry | 97.0% |
| Date and unit math | L · Brisk-S · reasoning high · 1 retry | 97.9% |
| Ambiguous entity | J · Brisk-S · reasoning low · 1 retry | 96.4% |
| Ambiguous entity | L · Brisk-S · reasoning high · 1 retry | 97.6% |

Pass rate by task for the pick and its fallback, with no interval because no task states its count.

#### The retry is what lifts Brisk-S at low reasoning over the line: it adds +1.9 points, interval +0.3 to +3.4.

| Panel | Row | Value | 95% unpaired interval |
|---|---|---|---|
| reasoning | Brisk-S · reasoning off · no retry → Low | +4.2 | +2.2 to +6.2 |
| reasoning | Brisk-S · reasoning off · 1 retry → Low | +2.5 | +0.9 to +4.1 |
| retry | Brisk-S · reasoning off · no retry → 1 retry | +3.5 | +1.5 to +5.6 |
| retry | Brisk-S · reasoning low · no retry → 1 retry | +1.9 | +0.3 to +3.4 |
| reasoning effect, across retry | (Brisk-S · reasoning off · 1 retry → Low) less (Brisk-S · reasoning off · no retry → Low) | -1.7 | -4.2 to +0.9 |

Each effect is one setting moved between two arms that differ in it alone, on Pass rate; each interaction is the same move's effect in one context less its effect in another, the two contexts differing in one other setting. Descriptive: read as two arms' own spreads, never a test.

What reasoning and a retry each buy Brisk-S in pass rate at off and low reasoning, with unpaired intervals, and how the reasoning effect changes with a retry.

#### No latency budget was declared, but the fallback's tail is longer: 870 ms against 513 ms for the pick.

| Group | Row | From | To |
|---|---|---|---|
| Latency | J · Brisk-S · reasoning low · 1 retry | 209 ms | 513 ms |
| Latency | L · Brisk-S · reasoning high · 1 retry | 364 ms | 870 ms |
| Latency | D · Atlas-M · reasoning low · 1 retry | 628 ms | 1500 ms |
| Latency | F · Atlas-M · reasoning high · 1 retry | 1094 ms | 2551 ms |

Latency from median to tail for the pick, its fallback and the passing Atlas-M configurations with a retry.

#### The pick's kept failures are planning errors: a collapsed multi-hop query, an off-by-one date window, and a guess where it should have asked which entity was meant.

| Example | Details | Reason | Excerpt |
|---|---|---|---|
| TQ-0418 · Multi-hop (2 to 3 calls) · Brisk-S · reasoning low · 1 retry · missing call | Input: Who chaired the committee that approved the 2019 grid budget?; Expected: search(committee, grid budget 2019) → lookup(chair) | Collapsed two hops into one query; returned the budget author, not the chair. | search(grid budget 2019 chair) |
| TQ-0772 · Date and unit math · Brisk-S · reasoning low · 1 retry · wrong args | Input: Filings in the 90 days before the March 3 hearing; Expected: date_from=2026-12-03, date_to=2027-03-03 | Off-by-one on the window start. The most common wrong-args pattern (9 of 17). | date_from=2026-12-04, date_to=2027-03-03 |
| TQ-1103 · Ambiguous entity · Brisk-S · reasoning low · 1 retry · wrong tool | Input: Latest on Mercury's pricing; Expected: ask_clarify(company or planet?) | Picked the company without asking. Scored wrong because the gold answer requires clarification. | search(Mercury pricing) |

Failures the snapshot kept for the pick, from multi-hop, date math and ambiguous entities; it states no count of all its failures, so the cards show kinds, not shares.

## What you could change

| What you could change | What it decides | What happens as it moves |
|---|---|---|
| How the gate is read | Whether the pick or its fallback ships | Read on the point estimate, as declared, the pick passes. Read on the computed interval's lower end, the pick (96.2–98.1%) no longer clears and Brisk-S · reasoning high · 1 retry (L) (97.2–98.8%) is the cheapest that does. |
| The gate's level | Which configuration is cheapest to pass | On point estimates, the pick holds for any gate above 95.3% up to 97.1%; above that, Brisk-S · reasoning high · 1 retry (L) up to 98%, then Atlas-M · reasoning high · 1 retry (F) up to 98.6%; past that nothing passes. Lower gates admit cheaper Brisk-S or Nano-XS configurations. |
| Price of a point of pass rate | Whether to buy headroom over the gate | On point estimates, valuing a point at $1,595.18 a month or more favours Brisk-S · reasoning high · 1 retry (L); at $17,236.36 or more, as routed in this run with no provider recorded, Atlas-M · reasoning high · 1 retry (F). Neither gain is told apart from zero, so these prices have no supported upper end. |

## Next steps

- Ship the pick and, in parallel, read its pass rate on 111,255 fresh cases or live tasks, the size that settles it four times in five on a two-sided interval. _Whether its true pass rate holds at the gate; a read below it means switching to the high-reasoning fallback._
- The owner declares whether the gate must hold on the point estimate or on the interval's lower end. _Which of the pick and its fallback ships, without any further run._
- Teach the pick to plan each hop of multi-hop questions and to ask when an entity is ambiguous, then read it on fresh held-out cases, at least 1,660 of them: the size that settles a clearance like the fallback's four times in five. _Whether the cheapest model can clear the gate with room on cases it was not tuned on, at roughly the pick's cost._
- Record which provider served each model and what the monthly volume counts. _Whether the cost order between models and every monthly figure hold beyond this run's routing._

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

- **Calls per task** Descriptive; the pick and its fallback differ little and it decides nothing the gate and cost don't. It stays in the options table.
- **Extra calls per task and the outcome measure** The snapshot holds no figures for them.
- **Model and reasoning effects outside Brisk-S, and the retry and reasoning effects at high reasoning** They explain why other configurations lose or cost more but don't change the call; the high-reasoning rows also print an unrounded bound.
- **How the retry's effect differs by model** Retry helps Nano-XS more than Atlas-M, but no Nano-XS configuration reaches the gate, so it doesn't bear on the call.
- **The retry's cost per call, and the monthly cost gap of each price step** No block draws them beside the gains; the per-point prices carry the trade, and the pick-to-fallback monthly gap stays in the lede and the options table.
- **Per-task figures for configurations other than the pick and its fallback** No task gate was declared and tasks state no case counts; they stay in the task table.
- **A four-in-five column in the settle table** Weighed and kept as prose: the settle table holds only the half-the-time sizes, so the four-in-five sizes Watch and Next use are stated by figure in the settle claim's sentence, beside the table.
- **Reading the confidence as low** Weighed and not taken: the declared rule reads the gate on the point estimate, and every other part of the call (the cost order, the fallback, the miss of every Nano-XS configuration) is well separated, so the high is scoped to those and the open gate question is named in the flag and the first caveat.

### Method

- **Cases** Each configuration ran 1,200 research queries across four tasks (single lookup, multi-hop, date and unit math, ambiguous entity); no repeats are stated and none were lost.
- **Configurations** A full grid of three models, reasoning off, low or high, and with or without a retry; there is no control.
- **Pass** The input doesn't define what makes a query pass; the kept failures are missing calls, wrong arguments and wrong tools.
- **Gate** Pass rate of at least 97% overall, read on the point estimate as declared.
- **Intervals** Computed here from each rate's binomial spread on its cases, by the normal approximation, at the level each block prints; differences between configurations combine the two spreads as independent and are not corrected for the number of configurations. Task figures state no counts and carry no interval.
- **Run sizes** A run is sized to settle a reading four times in five on a two-sided interval; the settle table's own sizes settle it half the time.
- **Cost** Cost per 1,000 calls as measured; no arm names its provider, so the routing is unknown. Monthly figures use the declared volume of 4,000,000, whose unit the input doesn't state; they assume it counts what cost is priced per.
- **Latency** Median and tail percentiles per configuration; no budget was declared.
- **Source** Synthetic example data from docs/eval-report-kit/references/r2-config-sweep.json, run 2026-10-23.

### Review findings left open

- **Should fix** · Claim 3: The settle table still prints only the half-the-time sizes: 38,350 one-sided and 54,451 two-sided for the pick, 572 and 812 for the fallback. The 111,255 that Watch and the first next step use, and the 1,660 that the third next step uses, appear only in prose. The caption admits this, but the figures a reader is told to plan a run on are still missing from the evidence. Next to them, 54,451 reads as the run size. Add a column 'Needed, two-sided, four in five' from settle.criteria.pass_rate.<arm>.overall.cases_needed_two_sided_80: 111,255 for the pick, 1,660 for the fallback. Then drop or relabel the half-the-time columns so the run size is plain.

_Written by claude-opus-5-5, served as claude-opus-5-5, for $4.94, with analyst skill 2a7662d928b0, on 2026-10-29._
