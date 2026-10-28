Choice · candidates compared · Research query planner · 18 arms · 4 tasks · Run 2026-10-23 · Synthetic example data

# No planner configuration is shown to clear both gates: the two that pass in every task both overrun the 600 ms latency budget.

Brisk-S high reasoning with retry, the cheaper of those two, has a tail latency of 870 ms over all cases. Two things could change the answer: per-task latency, which the budget reads and nobody recorded, and per-task pass rates at the line, where both Brisk-S setups with retry sit within margins a fresh run could reverse.

**Confidence: Low.**

- **Per-task pass rates state no count.** Material wherever a gate reading sits near the line: Brisk-S low reasoning with retry, inside the budget over all cases, misses ambiguous-entity asks by 0.6 points, a miss a run of today's size would read as a pass 33.3% of the time, so it could yet clear both gates.
- **Latency not recorded per task.** Only if the tasks do not split the cases between them: the reading that some task overruns the budget rests on an overall tail over it, and the budget itself is read in each task.
- **The pick is the best of several candidates.** Only if Brisk-S high reasoning with retry is taken forward on its multi-hop clearance of +0.3 points: as the best of 18, a fresh run tends to read it lower.
- **No arm is the control.** Only for comparisons between configurations, which read unpaired and so wider than they would paired; each gate reading stands against its own line.

**What we know despite it.** On point estimates, only Atlas-M and Brisk-S, each with high reasoning and one retry, clear the pass-rate gate in all four tasks. Both overrun the latency budget over all cases, and Atlas-M costs $13,908.00 a month against Brisk-S's $2,532.00.

**What is tangled.** Reasoning effort buys the pass rate and spends the latency budget: every configuration inside the budget overall misses the pass-rate gate in at least one task, and the latency gate itself has no figure in any task.

**Watch.** Brisk-S high reasoning with retry clears each task by a point and a half or less. Reading each task as holding its traffic share of cases, a run of today's size would read it below the line on multi-hop 37.4% of the time, on ambiguous entity 29.9% and on date math 16.5%.

**Watch.** Brisk-S low reasoning with retry sits exactly on the line in date math, so a run reads it either side 50% of the time; the routed mix in the next steps leans on this leg and on the three thin legs above.

**What would settle it.** Tail latency per task and case-counted reruns of the legs at the line for the two Brisk-S setups with retry, then a measured run of the routed mix; failing that, the planner owner declares which line gives.

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

## On point estimates, only two of 18 configurations clear the pass-rate gate in every task, both with high reasoning and one retry; several others sit within a point of the line.

| Slice | Arm | Pass rate |
|---|---|---|
| Single lookup | F · Atlas-M · reasoning high · 1 retry | 99.2% |
| Single lookup | L · Brisk-S · reasoning high · 1 retry | 98.5% |
| Single lookup | E · Atlas-M · reasoning high · no retry | 98.7% |
| Single lookup | D · Atlas-M · reasoning low · 1 retry | 99.2% |
| Single lookup | K · Brisk-S · reasoning high · no retry | 97.6% |
| Single lookup | J · Brisk-S · reasoning low · 1 retry | 98.4% |
| Multi-hop (2 to 3 calls) | F · Atlas-M · reasoning high · 1 retry | 97.9% |
| Multi-hop (2 to 3 calls) | L · Brisk-S · reasoning high · 1 retry | 97.3% |
| Multi-hop (2 to 3 calls) | E · Atlas-M · reasoning high · no retry | 96.6% |
| Multi-hop (2 to 3 calls) | D · Atlas-M · reasoning low · 1 retry | 96.4% |
| Multi-hop (2 to 3 calls) | K · Brisk-S · reasoning high · no retry | 95.5% |
| Multi-hop (2 to 3 calls) | J · Brisk-S · reasoning low · 1 retry | 95.6% |
| Date and unit math | F · Atlas-M · reasoning high · 1 retry | 98.6% |
| Date and unit math | L · Brisk-S · reasoning high · 1 retry | 97.9% |
| Date and unit math | E · Atlas-M · reasoning high · no retry | 97.7% |
| Date and unit math | D · Atlas-M · reasoning low · 1 retry | 97.8% |
| Date and unit math | K · Brisk-S · reasoning high · no retry | 96.6% |
| Date and unit math | J · Brisk-S · reasoning low · 1 retry | 97.0% |
| Ambiguous entity | F · Atlas-M · reasoning high · 1 retry | 98.3% |
| Ambiguous entity | L · Brisk-S · reasoning high · 1 retry | 97.6% |
| Ambiguous entity | E · Atlas-M · reasoning high · no retry | 97.2% |
| Ambiguous entity | D · Atlas-M · reasoning low · 1 retry | 97.2% |
| Ambiguous entity | K · Brisk-S · reasoning high · no retry | 96.1% |
| Ambiguous entity | J · Brisk-S · reasoning low · 1 retry | 96.4% |

Pass rate by task for the six configurations nearest the gate of 97% in each task; per-task figures state no count, so none carries an interval.

### Both overrun the latency budget over all cases; only Brisk-S with reasoning off or low, and every Nano-XS setup, stay inside it.

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
|  | Latency budget (p95) gate, at most 600 ms in each task | 600 ms |  |

Median to tail latency over all cases; the budget applies in each task, and the campaign recorded no latency figure for any task.

### Inside the budget over all cases, the nearest miss is Brisk-S low reasoning with retry: short of the line on multi-hop and ambiguous-entity asks and exactly on it for date math, the tasks high reasoning lifts most.

| Slice | Arm | Pass rate |
|---|---|---|
| Single lookup | J · Brisk-S · reasoning low · 1 retry | 98.4% |
| Multi-hop (2 to 3 calls) | J · Brisk-S · reasoning low · 1 retry | 95.6% |
| Date and unit math | J · Brisk-S · reasoning low · 1 retry | 97.0% |
| Ambiguous entity | J · Brisk-S · reasoning low · 1 retry | 96.4% |

Brisk-S low reasoning with retry by task against the gate of 97%; its tail of 513 ms is over all cases and does not show it inside the budget in each task.

| Row | Value | 95% unpaired interval | Note |
|---|---|---|---|
| Overall | +0.8 | -0.4 to +2.1 | +0.8 pts |
| Single lookup | +0.1 |  | +0.1 pts |
| Multi-hop (2 to 3 calls) | +1.7 |  | +1.7 pts |
| Date and unit math | +0.9 |  | +0.9 pts |
| Ambiguous entity | +1.2 |  | +1.2 pts |

Each row is Brisk-S · reasoning high · 1 retry less Brisk-S · reasoning low · 1 retry on Pass rate, overall and in each place both hold it.

The interval is independent: each arm's own spread (a pass/fail rate's binomial spread, from the arm's cases), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same.

Brisk-S high reasoning minus low reasoning, both with retry, in points; only the overall row has an interval (-0.4 to +2.1), and it does not exclude zero.

#### The kept failures of Brisk-S low reasoning with retry are planning errors: two hops collapsed into one query, a date window off by a day, and an ambiguous name answered without asking.

| Example | Details | Reason | Excerpt |
|---|---|---|---|
| TQ-0418 · Multi-hop (2 to 3 calls) · Brisk-S · reasoning low · 1 retry · missing call | Input: Who chaired the committee that approved the 2019 grid budget?; Expected: search(committee, grid budget 2019) → lookup(chair) | Collapsed two hops into one query; returned the budget author, not the chair. | search(grid budget 2019 chair) |
| TQ-0772 · Date and unit math · Brisk-S · reasoning low · 1 retry · wrong args | Input: Filings in the 90 days before the March 3 hearing; Expected: date_from=2026-12-03, date_to=2027-03-03 | Off-by-one on the window start. The most common wrong-args pattern (9 of 17). | date_from=2026-12-04, date_to=2027-03-03 |
| TQ-1103 · Ambiguous entity · Brisk-S · reasoning low · 1 retry · wrong tool | Input: Latest on Mercury's pricing; Expected: ask_clarify(company or planet?) | Picked the company without asking. Scored wrong because the gold answer requires clarification. | search(Mercury pricing) |

Three kept failures, one each from multi-hop, date math and ambiguous-entity asks; the snapshot keeps a sample, not every failure.

#### One retry lifts Brisk-S's pass rate at both low and high reasoning, and it is what carries high reasoning over the line in the three tasks it otherwise misses.

| Group | Row | Value | 95% unpaired interval | Note |
|---|---|---|---|---|
| reasoning | Brisk-S · reasoning low · no retry → High | +1.4 | -0.2 to +2.9 | +1.4 pts |
| reasoning | Brisk-S · reasoning low · 1 retry → High | +0.8 | -0.4 to +2.1 | +0.8 pts |
| retry | Brisk-S · reasoning low · no retry → 1 retry | +1.9 | +0.3 to +3.4 | +1.9 pts |
| retry | Brisk-S · reasoning high · no retry → 1 retry | +1.3 | +0.03535 to +2.6 | +1.3 pts |
| reasoning effect, across retry | (Brisk-S · reasoning low · 1 retry → High) less (Brisk-S · reasoning low · no retry → High) | -0.5 | -2.5 to +1.5 | -0.5 pts |
| retry effect, across reasoning | (Brisk-S · reasoning high · no retry → 1 retry) less (Brisk-S · reasoning low · no retry → 1 retry) | -0.5 | -2.5 to +1.5 | -0.5 pts |

Each effect is one setting moved between two arms that differ in it alone, on Pass rate; each interaction is the same move's effect in one context less its effect in another, the two contexts differing in one other setting. Descriptive: read as two arms' own spreads, never a test.

Each Brisk-S setting moved alone, in points of overall pass rate; the lift from retry reads smaller at high reasoning, but the interaction is not separated, and the last two rows are that one interaction read from each side.

## What you could change

| What you could change | What it decides | What happens as it moves |
|---|---|---|
| Latency budget | Which configurations may carry traffic | No budget value yields a pick as measured: the budget is read in each task and no task has a latency figure, so every configuration stays out or undecided at any line. |
| Pass-rate gate | Which configurations pass on quality in each task | Lowering it admits more configurations on pass rate, but none becomes a pick while the latency gate has no per-task figure. |

## Next steps

- Record tail latency per task for Brisk-S low and high reasoning, both with retry, on the same cases. _Whether low reasoning stays inside the budget on single lookups and date math, and whether high reasoning's overrun spares multi-hop and ambiguous-entity asks, which a routed mix would need._
- Rerun the legs at the line with case counts recorded: multi-hop for high reasoning with retry at about 11214 cases and ambiguous entity at about 2500; date math for low reasoning with retry, on the line today, at about 1118 cases, enough to settle a true rate one point clear of the line. _Whether each leg the routed mix needs truly clears the pass-rate gate, and whether low reasoning's ambiguous-entity miss is real._
- Run the routed mix as a measured option: low reasoning with retry on single lookups and date math, high reasoning with retry on multi-hop and ambiguous-entity asks. _Its pass rate and tail latency in each task, so it can be read against both gates like the configurations that ran._
- Fix multi-hop decomposition and ambiguous-name clarification for Brisk-S low reasoning with retry, then rerun those two tasks with their case counts recorded. _Whether a configuration already inside the budget can clear the pass-rate gate on its own._
- The planner owner declares which line gives if no configuration or mix clears both: the latency budget or the pass-rate gate. _Turns the diagnosis into a decision between Brisk-S high reasoning with retry and Brisk-S low reasoning with retry._

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
| Single lookup | Pass rate | at least 97% | Passes (97.2%) | Passes (98.3%) | Passes (98.6%) | Passes (99.2%) | Passes (98.7%) | Passes (99.2%) | Fails (94.3%) | Fails (96.6%) | Passes (97.4%) | Passes (98.4%) | Passes (97.6%) | Passes (98.5%) | Fails (86.8%) | Fails (92%) | Fails (91.7%) | Fails (95%) | Fails (93%) | Fails (95.8%) |
| Multi-hop (2 to 3 calls) | Pass rate | at least 97% | Fails (90.2%) | Fails (94.1%) | Fails (94.1%) | Fails (96.4%) | Fails (96.6%) | Passes (97.9%) | Fails (87.3%) | Fails (92.3%) | Fails (92.8%) | Fails (95.6%) | Fails (95.5%) | Passes (97.3%) | Fails (74.2%) | Fails (84.4%) | Fails (83.5%) | Fails (90%) | Fails (89.3%) | Fails (93.5%) |
| Date and unit math | Pass rate | at least 97% | Fails (93.7%) | Fails (96.2%) | Fails (96.4%) | Passes (97.8%) | Passes (97.7%) | Passes (98.6%) | Fails (90.8%) | Fails (94.4%) | Fails (95.1%) | Passes (97%) | Fails (96.6%) | Passes (97.9%) | Fails (80.5%) | Fails (88.2%) | Fails (87.6%) | Fails (92.5%) | Fails (91.1%) | Fails (94.7%) |
| Ambiguous entity | Pass rate | at least 97% | Fails (92.2%) | Fails (95.3%) | Fails (95.4%) | Passes (97.2%) | Passes (97.2%) | Passes (98.3%) | Fails (89.3%) | Fails (93.5%) | Fails (94.1%) | Fails (96.4%) | Fails (96.1%) | Passes (97.6%) | Fails (77.8%) | Fails (86.6%) | Fails (85.8%) | Fails (91.4%) | Fails (90.3%) | Fails (94.2%) |
| Single lookup | Latency budget (p95) | at most 600 ms | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) |
| Multi-hop (2 to 3 calls) | Latency budget (p95) | at most 600 ms | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) |
| Date and unit math | Latency budget (p95) | at most 600 ms | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) |
| Ambiguous entity | Latency budget (p95) | at most 600 ms | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) | Fails (no figure) |

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

- **Nano-XS, all six setups** Each misses the pass-rate gate in every task, by the widest gaps of any model; it is cheapest and fastest but nowhere near the line.
- **Atlas-M with reasoning off or low** Each misses the pass-rate gate in at least one task and overruns the latency budget overall; neither changes the reading.
- **Model effects between Atlas-M and Brisk-S** At high reasoning the gap is not separated, and Atlas-M is out on latency at every setting, so the comparison does not bear on the call.
- **Calls per task and extra calls** They describe how the planner spends calls; no gate reads them.
- **Retry and reasoning effects with reasoning off** No setup with reasoning off comes near both lines, so its lever rows are left to the full tables.
- **A pick between the two Brisk-S setups with retry** Weighed and kept out: each rests on per-task readings at the line and an unmeasured per-task tail, so either pick could reverse on a fresh run.

### Method

- **Configurations** 18 configurations crossing three models, three reasoning levels and retry on or off; no control, so every comparison is between two configurations' own figures.
- **Cases** 1200 cases per configuration, none lost, spread over four tasks by traffic share; per-task figures state no case count.
- **Pass** The campaign does not state its pass rule; the kept failures are scored against an expected call plan and classed as a missing call, wrong arguments or the wrong tool.
- **Gates** Pass rate at least 97% and tail latency at most 600 ms, each read in every task on the point estimate.
- **Tail latency** The p95 latency, recorded only over all cases; the budget is declared per task, so it is read as unknown in each task, with an overall figure over the budget implying some task is over it too. Where the full gate grid marks a latency cell as failing with no figure, read it as unknown. _Caveats: Latency not recorded per task._
- **Intervals and run sizes** Overall pass-rate differences carry a binomial interval computed from case counts, at the level each table prints, combined unpaired, which errs wide where configurations ran the same cases. How often a rerun reads the other way, and the cases it needs, read each task as holding its traffic share of cases, an assumption; sizes are two-sided. _Caveats: Per-task pass rates state no count._
- **Cost** Cost per thousand planner calls, and per month at 4000000 calls.
- **Source** Synthetic example data for a research query planner config sweep.

### Review findings left open

- **Should fix** · The page: Rewrite the first confidence bullet so it gives every gap that stands between Brisk-S low reasoning with retry and a pass, not just one. Today it says this setup 'could yet clear both gates' and cites only its ambiguous-entity miss (-0.6, read as a pass 33.3% of the time). It also misses multi-hop by 1.4 points, which a run of today's size reads as a pass only 11.9% of the time. It sits exactly on the line in date math (50%), and its per-task latency is unknown. Say that clearing both gates would need all three pass-rate legs to go its way, plus per-task latency under 600 ms, so it is a long shot.
- **Should fix** · The page: The rerun step calls its case counts 'enough to settle'. They are the two-sided sizes at which only half of runs settle, assuming the truth equals today's figure. The sizes at which four in five settle are about double: about 22911 for multi-hop, 5108 for ambiguous entity and 2285 for date math (at one point clear). Print those, or say a run of the stated size settles the question only half the time. Also say that the multi-hop leg alone would be roughly nine times the arm's whole 1200-case run. That cost makes the owner's declaration a realistic alternative, not only a fallback.
- **Should fix** · Claim 5: The block still prints the one interaction twice: 'reasoning effect, across retry' and 'retry effect, across reasoning', both -0.5 (-2.5 to +1.5). The caption admits they are the same interaction. Drop one row so the block shows four effects and one interaction.

_Written by claude-opus-5-5, served as claude-opus-5-5, for $3.48, with analyst skill b70be6bcfa68, on 2026-10-28._
