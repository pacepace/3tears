Choice · candidates compared · Research query planner · 18 arms · 4 tasks · Run 2026-10-23 · Synthetic example data

# No planner configuration clears both gates: high reasoning is what lifts every task to the pass rate, and it is also what breaks the latency budget.

Brisk-S high+retry (L) holds the pass rate in every task at $2,532.00 a month, but its p95 over all cases is 870 ms against a 600 ms budget. A fix that lifts Brisk-S low+retry (J) on multi-hop and ambiguous queries, or a declaration of which gate gives, would settle the call.

**Confidence: High, if the caveats below aren't material.**

- **Latency measured only across all cases.** Material only if the four tasks do not split the cases between them; if they do, an overall p95 over budget means at least one task is over it too, so no high-reasoning configuration can pass the latency gate as declared.
- **Per-task case counts unstated.** Material to which near miss is closest, not to whether anything passes: per-task rates carry no count, so every gate is read on its point estimate and the noise on thin gaps is sized only under counts assumed from traffic shares.
- **The pick is the best of several candidates.** Material if a re-run reads the two high-reasoning configurations' thin per-task clearances lower, as the best of 18 tends to; that would leave fewer options on the pass-rate side, not more.
- **No arm is the control.** Material only to the comparisons between configurations, which are read unpaired and so err wide; the gate readings are absolute and do not rest on it.

**What is tangled.** The two gates pull opposite ways along one setting. High reasoning brings Brisk-S and Atlas-M to the pass rate in every task and pushes their p95 past the budget; low or no reasoning keeps Brisk-S inside the budget overall and leaves it short on multi-hop and ambiguous queries.

**What we know despite it.** Atlas-M high+retry (F) is out on every axis but pass rate, and there it is not separated from Brisk-S high+retry (L) while costing 5.5 times as much. Every Nano-XS configuration misses the pass rate in every task.

**Watch.** L clears multi-hop by 0.3 points; at an assumed 300 multi-hop cases a fresh run would read it below the line 37.4% of the time. J's date-math rate sits exactly on the line.

**What would settle it.** Either a Brisk-S low+retry (J) that clears multi-hop and ambiguous queries, or the planner's owner declaring whether the latency budget or the pass rate gives. Neither was declared in advance.

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

## Only 2 of the 18 configurations hold the pass rate in every task, and both run past the latency budget across all cases.

_Caveats: Latency measured only across all cases._

| Panel | Row | Value |
|---|---|---|
| Single lookup | L · Brisk-S · reasoning high · 1 retry | 98.5% |
| Single lookup | J · Brisk-S · reasoning low · 1 retry | 98.4% |
| Single lookup | F · Atlas-M · reasoning high · 1 retry | 99.2% |
| Single lookup | Pass rate gate, at least 97% | 97.0% |
| Multi-hop (2 to 3 calls) | L · Brisk-S · reasoning high · 1 retry | 97.3% |
| Multi-hop (2 to 3 calls) | J · Brisk-S · reasoning low · 1 retry | 95.6% |
| Multi-hop (2 to 3 calls) | F · Atlas-M · reasoning high · 1 retry | 97.9% |
| Multi-hop (2 to 3 calls) | Pass rate gate, at least 97% | 97.0% |
| Date and unit math | L · Brisk-S · reasoning high · 1 retry | 97.9% |
| Date and unit math | J · Brisk-S · reasoning low · 1 retry | 97.0% |
| Date and unit math | F · Atlas-M · reasoning high · 1 retry | 98.6% |
| Date and unit math | Pass rate gate, at least 97% | 97.0% |
| Ambiguous entity | L · Brisk-S · reasoning high · 1 retry | 97.6% |
| Ambiguous entity | J · Brisk-S · reasoning low · 1 retry | 96.4% |
| Ambiguous entity | F · Atlas-M · reasoning high · 1 retry | 98.3% |
| Ambiguous entity | Pass rate gate, at least 97% | 97.0% |

Pass rate by task for the two configurations that hold the gate everywhere and the fastest near miss, against the 97% gate.

| Group | Row | From | To |
|---|---|---|---|
| Latency | H · Brisk-S · reasoning off · 1 retry | 135 ms | 359 ms |
| Latency | I · Brisk-S · reasoning low · no retry | 209 ms | 465 ms |
| Latency | J · Brisk-S · reasoning low · 1 retry | 209 ms | 513 ms |
| Latency | K · Brisk-S · reasoning high · no retry | 364 ms | 810 ms |
| Latency | L · Brisk-S · reasoning high · 1 retry | 364 ms | 870 ms |
| Latency | F · Atlas-M · reasoning high · 1 retry | 1094 ms | 2551 ms |
|  | Latency budget (p95) gate, at most 600 ms in each task | 600 ms |  |

Latency from p50 to p95 across all cases against the 600 ms budget; no task's own p95 was recorded.

### On Brisk-S with a retry, high reasoning adds +0.8 points of pass rate over low, not separated from zero, while it moves p95 from 513 ms to 870 ms.

| Group | Row | Value | 95% unpaired interval | Note |
|---|---|---|---|---|
| reasoning | Brisk-S · reasoning off · no retry → Low | +4.2 | +2.2 to +6.2 | +4.2 pts |
| reasoning | Brisk-S · reasoning off · no retry → High | +5.5 | +3.6 to +7.4 | +5.5 pts |
| reasoning | Brisk-S · reasoning off · 1 retry → Low | +2.5 | +0.9 to +4.1 | +2.5 pts |
| reasoning | Brisk-S · reasoning off · 1 retry → High | +3.3 | +1.8 to +4.9 | +3.3 pts |
| reasoning | Brisk-S · reasoning low · no retry → High | +1.4 | -0.2 to +2.9 | +1.4 pts |
| reasoning | Brisk-S · reasoning low · 1 retry → High | +0.8 | -0.4 to +2.1 | +0.8 pts |
| retry | Brisk-S · reasoning off · no retry → 1 retry | +3.5 | +1.5 to +5.6 | +3.5 pts |
| retry | Brisk-S · reasoning low · no retry → 1 retry | +1.9 | +0.3 to +3.4 | +1.9 pts |
| retry | Brisk-S · reasoning high · no retry → 1 retry | +1.3 | +0.03535 to +2.6 | +1.3 pts |
| reasoning effect, across retry | (Brisk-S · reasoning off · 1 retry → Low) less (Brisk-S · reasoning off · no retry → Low) | -1.7 | -4.2 to +0.9 | -1.7 pts |
| reasoning effect, across retry | (Brisk-S · reasoning off · 1 retry → High) less (Brisk-S · reasoning off · no retry → High) | -2.2 | -4.6 to +0.2 | -2.2 pts |
| reasoning effect, across retry | (Brisk-S · reasoning low · 1 retry → High) less (Brisk-S · reasoning low · no retry → High) | -0.5 | -2.5 to +1.5 | -0.5 pts |
| retry effect, across reasoning | (Brisk-S · reasoning low · no retry → 1 retry) less (Brisk-S · reasoning off · no retry → 1 retry) | -1.7 | -4.2 to +0.9 | -1.7 pts |
| retry effect, across reasoning | (Brisk-S · reasoning high · no retry → 1 retry) less (Brisk-S · reasoning off · no retry → 1 retry) | -2.2 | -4.6 to +0.2 | -2.2 pts |
| retry effect, across reasoning | (Brisk-S · reasoning high · no retry → 1 retry) less (Brisk-S · reasoning low · no retry → 1 retry) | -0.5 | -2.5 to +1.5 | -0.5 pts |

Each effect is one setting moved between two arms that differ in it alone, on Pass rate; each interaction is the same move's effect in one context less its effect in another, the two contexts differing in one other setting. Descriptive: read as two arms' own spreads, never a test.

What each Brisk-S setting buys in pass rate, in points with unpaired intervals; a retry helps at every reasoning level.

### Atlas-M high+retry (F) buys no separated pass rate over Brisk-S high+retry (L) and costs 5.5 times as much.

_Caveats: No arm is the control._

| Place | First arm | Second arm | Difference | Interval | Level | Read as |
|---|---|---|---|---|---|---|
| Overall | 98.6% | 98.0% | +0.7 | -0.4 to +1.7 | 95.0% | independent: each arm's own spread (a pass/fail rate's binomial spread, from the arm's cases), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same |
| Single lookup | 99.2% | 98.5% | +0.7 | — | — | — |
| Multi-hop (2 to 3 calls) | 97.9% | 97.3% | +0.6 | — | — | — |
| Date and unit math | 98.6% | 97.9% | +0.7 | — | — | — |
| Ambiguous entity | 98.3% | 97.6% | +0.7 | — | — | — |

Atlas-M minus Brisk-S on pass rate; per-task rows carry no interval because no task states its count.

### Brisk-S low+retry (J) runs inside the budget across all cases, with L costing 2.1 times as much, and misses the pass rate only on multi-hop and ambiguous queries, where its kept failures are planning errors.

_Caveats: Per-task case counts unstated._

| Example | Details | Reason | Excerpt |
|---|---|---|---|
| TQ-0418 · Multi-hop (2 to 3 calls) · Brisk-S · reasoning low · 1 retry · missing call | Input: Who chaired the committee that approved the 2019 grid budget?; Expected: search(committee, grid budget 2019) → lookup(chair) | Collapsed two hops into one query; returned the budget author, not the chair. | search(grid budget 2019 chair) |
| TQ-0772 · Date and unit math · Brisk-S · reasoning low · 1 retry · wrong args | Input: Filings in the 90 days before the March 3 hearing; Expected: date_from=2026-12-03, date_to=2027-03-03 | Off-by-one on the window start. The most common wrong-args pattern (9 of 17). | date_from=2026-12-04, date_to=2027-03-03 |
| TQ-1103 · Ambiguous entity · Brisk-S · reasoning low · 1 retry · wrong tool | Input: Latest on Mercury's pricing; Expected: ask_clarify(company or planet?) | Picked the company without asking. Scored wrong because the gold answer requires clarification. | search(Mercury pricing) |

One kept failure per task for J: a collapsed two-hop query, an off-by-one date window, and a guess where a clarifying question was expected.

## What you could change

| What you could change | What it decides | What happens as it moves |
|---|---|---|
| Latency budget (p95) | Whether the high-reasoning configurations can pass the latency gate | No budget changes today's reading, because no task's p95 was recorded. If the tasks split the cases, L would need a budget no lower than its overall p95 of 870 ms. |
| Pass-rate gate | Whether the fast Brisk-S configurations can pass | J's weakest task reads 95.6%; a gate at or below that would admit it on pass rate, its per-task latency still unknown. |

## Next steps

- Re-record latency per task for the Brisk-S configurations, from the same runs if per-trial timings were kept. _Reads the latency gate where it was declared, and shows whether L misses on every task or only on the slow ones, which would make routing those tasks an option worth running._
- Fix multi-hop decomposition and ambiguous-entity clarification in J's planner, then re-run it on larger multi-hop and ambiguous sets; at an assumed 300 multi-hop cases today, about 825 would confirm today's miss with a two-sided interval. _Whether a configuration inside the budget can clear the pass rate in every task._
- The planner's owner declares which gate gives if no fix closes the gap: the latency budget or the per-task pass rate. _Turns the choice between L and J into a decision._

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

- **Nano-XS, all six configurations** Each misses the pass rate in every task by a wide gap; cheapest by far, but no setting brings it near the gate.
- **Atlas-M with reasoning off or low, and the no-retry high-reasoning arms** Each misses the pass rate on multi-hop; none changes the reading.
- **Calls per task, extra calls and outcome shares** Describe how plans fail; no gate reads them and they do not change the call.
- **Retry helps Nano-XS more than Atlas-M** A real interaction, but on a model that is out on every task.
- **Median latency** The gate reads the p95; the p50 is shown only as the start of each latency bar.

### Method

- **Cases** Each of the 18 configurations ran the same 1200 queries once, none lost, across four tasks with traffic shares; the snapshot states no per-task case count.
- **Pass** A query passes or fails; the campaign states no further definition, and kept failures are graded against an expected tool-call plan.
- **Gates** Pass rate at least 97% and p95 latency at most 600 ms, each in every task; both read on the point estimate. Only overall p95 was recorded, so the latency gate is unknown in every task, never failed.
- **Intervals** Binomial intervals at the 95% level, computed from each configuration's cases and combined as independent for differences, which reads wide where configurations ran the same cases; no correction across the many comparisons.
- **Noise on per-task gaps** Sized under an assumed count, each task's share of traffic times the configuration's cases, and labelled as such. _Caveats: Per-task case counts unstated._
- **No control** No production configuration was declared, so every comparison is between two candidates' own figures.
- **Cost** Cost per 1,000 planner calls, and per month at the declared volume of 4000000 calls.
- **Source** Synthetic example data, run 2026-10-23.

### Review findings left open

- **Must fix** · The page: The appendix gate table marks every arm 'Fails (no figure)' on the latency budget in every task. The Method says that gate is 'unknown in every task, never failed', and options_dominance reads it as unknown. As printed, the page says Brisk-S low+retry (513 ms over all cases) fails latency everywhere, which undercuts the 'fix low+retry' path the page recommends. Relabel those cells as unknown, or add a note at the table that 'Fails (no figure)' means no per-task figure was recorded.
- **Should fix** · The page: Add Brisk-S high+retry's other thin clearances to Watch. At the assumed counts, a rerun reads its ambiguous-entity rate below 97% 29.9% of the time and its date-math rate 16.5% of the time, on top of the 37.4% on multi-hop. Say that together these make a rerun more likely than not to show it failing some task. That matters to an owner deciding to let the latency budget give and take this configuration. Also add that low+retry's ambiguous miss (-0.6 points) has a 33.3% chance of reading the other way.
- **Should fix** · The page: Soften the headline's 'high reasoning is what lifts every task to the pass rate'. The reasoning claim shows the high-over-low effect with a retry is +0.8 points (-0.4 to +2.1), not separated from zero, and the per-task lifts have no intervals. Something like 'high reasoning is the setting under which Brisk-S and Atlas-M reach 97% in every task on point estimates' would match the evidence.
- **Should fix** · Claim 2: Drop the three 'retry effect, across reasoning' rows; they repeat the three 'reasoning effect, across retry' values (-1.7, -2.2, -0.5) under a second label. Either add the latency half of the claim (p95 513 ms vs 870 ms) to this block or point to the gates claim's latency bars. Round '+0.03535' to the precision of the other bounds.
- **Should fix** · Claim 4: 'Its kept failures are planning errors' rests on one kept example each on multi-hop (a missing call) and ambiguous queries (a wrong tool), with no stated share of all failures. Reword it as 'the kept examples show…'. Fix the caption: there are three examples for four tasks, none from single lookup. Say that the date-math example comes from a task low+retry passes, exactly on the line at 97.0%.
- **Should fix** · The page: Next step 2: size the rerun for what it is meant to settle, whether a fixed low+retry clears 97% on multi-hop and on ambiguous queries, not to 'confirm today's miss'. Note that the ambiguous set is the harder one (about 3,704 cases just to confirm today's -0.6-point miss). Finish the trailing fragment with what the step settles.
- **Should fix** · Claim 3: 'Atlas-M high+retry is out on every axis but pass rate' overstates its latency standing. On the declared per-task gate it is undecided; the miss is inferred from its 2551 ms p95 over all cases, assuming the tasks split the cases. Say so in one clause. Also note that its per-task clearances are wider than Brisk-S high+retry's (for example +0.9 vs +0.3 points on multi-hop), which bears on the choice if the latency budget gives.

_Written by claude-opus-5-5, served as claude-opus-5-5, for $1.87, with analyst skill b70be6bcfa68, on 2026-10-28._
