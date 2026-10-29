Choice · candidates compared · Research query planner · 18 arms · 4 tasks · Run 2026-10-23 · Synthetic example data

# Run the planner on Brisk-S with low reasoning and one retry: the cheapest setting to clear the 97% pass-rate gate, though by only 0.1 points, inside the noise.

It passes 97.1% of its cases at $0.302 per thousand calls, but a fresh run of this size would read it below the gate 38.6% of the time. If the gate must hold on its interval, Brisk-S with high reasoning and one retry is the pick, at 2.1 times the cost.

**Confidence: Low.**

- **Pass rate is cleared on a point estimate.** Only if the pick's true pass rate sits below the gate: a fresh run of today's size would read it there 38.6% of the time, and then Brisk-S with high reasoning and one retry becomes the pick.
- **The pick is the best of several candidates.** Only if the pick reads high by luck, being the setting of eighteen that lands nearest above the line; a rerun on fresh cases would likely read it lower, toward or under the gate.
- **No arm is the control.** Only for the comparison between the two Brisk-S settings, which is read unpaired and so wider than a paired one; the gate reads each setting on its own figure and does not rest on it.
- **Task mix.** Only if live traffic carries more multi-hop or ambiguous-entity queries than the eval did: the pick sits below the gate on both, so a heavier share of them would pull its overall rate under the line.

**Recommendation.** Brisk-S, low reasoning, one retry. It is the cheapest of the five settings that clear the gate, at $1,208 a month at the declared volume, and the only one no other setting beats on both cost and pass rate.

**Why not the others.** Brisk-S with high reasoning and one retry clears the gate with room, but costs 2.1 times as much, and its lead of 0.8 points is not told apart from zero (-0.4 to +2.1). The Atlas-M settings that pass cost 5.5 times as much or more. No Nano-XS setting reaches the gate; its best reads 94.8%.

**Watch.** The weakest margin is the pick's own: Brisk-S · reasoning low · 1 retry (J) reads 97.1% against the 97% gate, 0.1 points clear, and its computed interval (96.2–98.1%) crosses the line. A run that settles it four times in five needs 111,255 cases, so more cases are not a practical way out.

**Next.** The planner owner declares whether the gate must hold on its interval or on its point estimate. On the interval, switch to Brisk-S with high reasoning and one retry, which costs $1,324 more a month.

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

## Only five of eighteen settings clear the gate, and Brisk-S with low reasoning and one retry is the cheapest of them.

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

Pass rate against cost per thousand calls for every setting, cost on a log axis; the gate sits at 97%.

## The pick's clearance is inside the noise: its computed interval (96.2–98.1%) crosses the gate, while Brisk-S with high reasoning and one retry sits wholly above it (97.2–98.8%).

| Row | Value | 95% computed interval |
|---|---|---|
| J · Brisk-S · reasoning low · 1 retry | 97.1% | 96.2% to 98.1% |
| L · Brisk-S · reasoning high · 1 retry | 98.0% | 97.2% to 98.8% |
| D · Atlas-M · reasoning low · 1 retry | 97.9% | 97.1% to 98.7% |
| E · Atlas-M · reasoning high · no retry | 97.7% | 96.9% to 98.6% |
| F · Atlas-M · reasoning high · 1 retry | 98.6% | 98.0% to 99.3% |
| Pass rate gate, at least 97% | 97.0% |  |

Each interval is computed here from the rate's binomial spread, by the normal approximation, held to 0 to 100%. A rate of none or all has none.

Pass rate of the settings that clear the gate, each with a computed interval, against the gate line.

| Arm | Against | Today's figure | Line | Distance toward passing | Settles | Sizes count | Needed, one-sided | Needed, two-sided | Today | A fresh run reads the other side | Level |
|---|---|---|---|---|---|---|---|---|---|---|---|
| J · Brisk-S · reasoning low · 1 retry | Pass rate | 97.1% | 97.0% | +0.1 | clears | cases | 38,350 | 54,451 | 1,200 | 38.6% | 95% one-sided |
| L · Brisk-S · reasoning high · 1 retry | Pass rate | 98.0% | 97.0% | +1.0 | clears | cases | 572 | 812 | 1,200 | 0.9% | 95% one-sided |

In each row, "A fresh run reads the other side" is how often a fresh run of today's size would read the figure on the other side of its line, if the truth is today's estimate (95% one-sided); it plans a run and tests nothing.

How far each Brisk-S setting sits from the gate, how often a fresh run of today's size would read it on the other side, and the cases a settling run needs.

### Moving up to high reasoning costs $1,324 more a month for a pass-rate lift not told apart from zero.

| Row | Value | 95% unpaired interval |
|---|---|---|
| Overall | +0.8 | -0.4 to +2.1 |
| Single lookup | +0.1 |  |
| Multi-hop (2 to 3 calls) | +1.7 |  |
| Date and unit math | +0.9 |  |
| Ambiguous entity | +1.2 |  |

Each row is Brisk-S · reasoning high · 1 retry less Brisk-S · reasoning low · 1 retry on Pass rate, overall and in each place both hold it.

The interval is independent: each arm's own spread (a pass/fail rate's binomial spread, from the arm's cases), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same.

High-reasoning Brisk-S minus low-reasoning Brisk-S, both with one retry, in pass-rate points; only the overall row carries an interval, unpaired.

### A retry lifts Brisk-S's pass rate at every reasoning level, by 1.9 points at low reasoning for $0.014 more per thousand calls; it is what carries the pick over the gate.

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

What each Brisk-S setting buys in pass-rate points when it alone changes, with unpaired intervals; the last rows show whether reasoning's lift differs with a retry.

#### The pick's misses sit in multi-hop and ambiguous-entity queries, where it reads 95.6% and 96.4%; this split is descriptive.

| Group | Row | Value |
|---|---|---|
| Single lookup | J · Brisk-S · reasoning low · 1 retry | 98.4% |
| Single lookup | L · Brisk-S · reasoning high · 1 retry | 98.5% |
| Multi-hop (2 to 3 calls) | J · Brisk-S · reasoning low · 1 retry | 95.6% |
| Multi-hop (2 to 3 calls) | L · Brisk-S · reasoning high · 1 retry | 97.3% |
| Date and unit math | J · Brisk-S · reasoning low · 1 retry | 97.0% |
| Date and unit math | L · Brisk-S · reasoning high · 1 retry | 97.9% |
| Ambiguous entity | J · Brisk-S · reasoning low · 1 retry | 96.4% |
| Ambiguous entity | L · Brisk-S · reasoning high · 1 retry | 97.6% |
|  | Pass rate gate, at least 97% overall | 97.0% |

Pass rate by kind of query for the two Brisk-S settings with one retry; the gate line applies overall, not per kind.

| Example | Details | Reason | Excerpt |
|---|---|---|---|
| TQ-0418 · Multi-hop (2 to 3 calls) · Brisk-S · reasoning low · 1 retry · missing call | Input: Who chaired the committee that approved the 2019 grid budget?; Expected: search(committee, grid budget 2019) → lookup(chair) | Collapsed two hops into one query; returned the budget author, not the chair. | search(grid budget 2019 chair) |
| TQ-0772 · Date and unit math · Brisk-S · reasoning low · 1 retry · wrong args | Input: Filings in the 90 days before the March 3 hearing; Expected: date_from=2026-12-03, date_to=2027-03-03 | Off-by-one on the window start. The most common wrong-args pattern (9 of 17). | date_from=2026-12-04, date_to=2027-03-03 |
| TQ-1103 · Ambiguous entity · Brisk-S · reasoning low · 1 retry · wrong tool | Input: Latest on Mercury's pricing; Expected: ask_clarify(company or planet?) | Picked the company without asking. Scored wrong because the gold answer requires clarification. | search(Mercury pricing) |

The failed cases kept for the pick, one per kind of query that misses; the snapshot does not say how many failures they were drawn from.

## What you could change

| What you could change | What it decides | What happens as it moves |
|---|---|---|
| How the gate is read: on the point estimate or on the whole interval | Which Brisk-S setting is picked | On the point estimate the pick is low reasoning with one retry, at $0.302 per thousand calls. On the interval its range (96.2–98.1%) crosses the line, and high reasoning with one retry, at $0.633, is the cheapest whose range clears it. |
| Where the gate's line sits | The pick, read on point estimates | The pick holds for any line above Brisk-S low reasoning without a retry (95.3%) up to its own 97.1%. Above that, Brisk-S high reasoning with one retry holds up to 98%, then Atlas-M high reasoning with one retry up to 98.6%; past it no setting passes. Each of these points carries its own interval, so each switch is a band, not a sharp line. |

## Next steps

- The planner owner declares whether the gate must hold on its interval or its point estimate. _Whether low or high reasoning ships; on the interval reading, high reasoning with one retry is the pick at $1,324 more a month._
- Fix the pick's multi-hop decomposition and date-window handling, then re-test it on fresh cases. _Whether low reasoning can clear the gate by more than noise; confirming today's thin clearance would need 111,255 cases, so a change to the setting is the practical route._
- Declare the live share of each kind of query and what the monthly volume counts. _Whether the eval's task mix matches live traffic, and so whether the overall pass rate and the monthly costs describe production._

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

- **Latency (median and tail per setting)** No latency budget was declared, so it gates nothing and does not change the call.
- **Calls per task** Varies little across the Brisk-S settings; describes planner behaviour and does not change the call.
- **Model and reasoning effects for Atlas-M and Nano-XS** In the full lever table; they explain the grid but neither model offers a cheaper passing setting.
- **Extra calls and outcome measures** Listed for the campaign but hold no figures in the snapshot.

### Method

- **Cases** Each setting ran 1,200 research queries across four kinds (single lookup, multi-hop, date and unit math, ambiguous entity), with no repeats stated and none lost. Per-kind case counts are not stated.
- **Pass** Each query is graded pass or fail; the campaign does not state the rule that makes a plan pass. Kept failures show a missing call, wrong arguments and a wrong tool.
- **Gate** One gate, declared for the campaign: overall pass rate at least 97%, read here on each setting's point estimate.
- **Intervals** The pipeline states no interval on pass rate. Intervals shown are computed here from each setting's binomial spread over its cases, by normal approximation. Comparisons between settings are unpaired, so wider than a paired reading of the same cases; there is no control arm and no correction across comparisons.
- **Cost** Cost per thousand planner calls as measured. Monthly figures multiply it by the declared volume of 4,000,000, whose unit the input does not state; if it counts tasks rather than calls, monthly costs run higher.
- **Source** Synthetic example data from docs/eval-report-kit/references/r2-config-sweep.json, run 2026-10-23.

### Review findings left open

- **Should fix** · The page: The Recommendation says the pick is 'the only one no other setting beats on both cost and pass rate'. As written this is false. High-reasoning Brisk-S with one retry (98.0%) and high-reasoning Atlas-M with one retry (98.6%) are beaten by no setting on both axes, and neither are the cheap Nano-XS settings. The pick is undominated only when gate-passing is treated as a filter and the passing settings are then compared on cost. That is what options_dominance computes (it weighs gates, then cost). Reword along these lines: 'among the settings that clear the gate, every other one costs more'.
- **Should fix** · Claim 2: Watch and next step 2 say settling the pick 'four times in five needs 111,255 cases'. That is the two-sided, 80%-power size (settle cases_needed_two_sided_80). The c-thin table prints only the 50% sizes, 38,350 one-sided and 54,451 two-sided. A reader sees three numbers for 'cases needed' and none is explained against the others. Add the 80% columns (cases_needed_80 = 87,635 and cases_needed_two_sided_80 = 111,255) to the c-thin table, or say in the caption that the 'Needed' columns are the size at which half of such runs settle it.
- **Should fix** · Claim 4: The claim prices the retry at '$0.014 more per thousand calls' ($0.288 to $0.302), but the block draws only pass-rate effects. Add the cost step to the block, or move the cost figure out of this claim. Also, the high-reasoning retry row prints its lower end as '+0.03535' while every other row is rounded to one decimal place. Print it as '+0.04' or similar so it doesn't read as false precision. Note that this interval only barely excludes zero, so the claim's 'at every reasoning level' rests on that row's thin margin.
- **Should fix** · Claim 5: 'The pick's misses sit in multi-hop and ambiguous-entity queries' overstates what the block shows. Date and unit math reads 97.0%, so it also misses, and one of the three kept failures (TQ-0772, the off-by-one date window, the most common wrong-args pattern at 9 of 17) is from date math. Say the pick sits below the gate line on multi-hop and ambiguous entity and fails in every kind. Next step 2 targets multi-hop decomposition and date-window handling but not the ambiguous-entity clarification failure (TQ-1103), although c-tasks and the task-mix caveat both name ambiguous entity as a weak kind. Add it, or say why it is left out.
- **Should fix** · Claim 2: The reference lines in the c-thin and c-tasks blocks are labelled 'Floor 97%', but the rows and the rest of the page call this criterion a gate ('Pass rate gate, at least 97%'). Use one name so a reader doesn't look for a second criterion.

_Written by claude-opus-5-5, served as claude-opus-5-5, for $1.99, with analyst skill 1677ccb31cd6, on 2026-10-29._
