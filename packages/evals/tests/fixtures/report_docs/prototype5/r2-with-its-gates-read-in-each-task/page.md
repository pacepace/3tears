Choice · candidates compared · Research query planner · 18 arms · 4 tasks · Run 2026-10-23 · Synthetic example data

# Ship none yet: the two configurations that hold the 97% pass rate in every task run far past the 600 ms p95 latency budget.

Brisk-S · reasoning high · 1 retry (L), the cheaper of the two that hold every task, reads a p95 of 870 ms overall, so its slowest task is at least that slow. A budget covering that task, once read, or a Brisk-S fix that lifts multi-hop and ambiguous-entity planning within the budget would change the answer.

**Latency was read only over all tasks, never within each one, so the per-task latency gate is inferred from the overall p95, not read.**

**Confidence: High, if the caveats below aren't material.**

- **Latency read only overall.** only if a task's own p95 sits far below the overall one, enough that a configuration holding the pass rate everywhere comes under 600 ms in every task; with overall p95 readings of 870 ms and 2551 ms, at least one task must miss if the tasks split the cases between them.
- **Per-task pass rates state no count.** if the misses of Brisk-S · reasoning low · 1 retry (J) are noise: assuming each task held its traffic share of the cases, its multi-hop miss reads the wrong way 11.9% of the time and its ambiguous-entity miss 33.3%. Were both truly to clear, it would hold every task while under the budget overall and become the pick, pending per-task latency.
- **The pick is the best of several candidates.** only in the direction of the hold: Brisk-S · reasoning high · 1 retry (L) is the best of 18 configurations and clears multi-hop by 0.3 points, so a fresh run would tend to read it lower and leave only Atlas-M · reasoning high · 1 retry (F), which clears multi-hop by 0.9 points, holding the pass rate everywhere, at a slower p95.
- **No arm is the control.** only for the comparison of the two configurations that hold the pass rate, which reads their own spreads unpaired; the hold compares each configuration with the declared lines, not with a baseline.

**Recommendation.** Hold: zero of the 18 configurations clear both gates. The two left undecided are so only because latency was never read within a task, and both run past the budget overall.

**Why not the others.** Brisk-S · reasoning high · 1 retry (L) holds every task's pass rate but reads a p95 of 870 ms overall; Atlas-M · reasoning high · 1 retry (F) holds it too at 2551 ms and, as this run's unnamed provider routing served it, costs 5.5 times as much.

**Why not the others.** Brisk-S · reasoning low · 1 retry (J), the closest within the budget overall at 513 ms, misses the pass rate on multi-hop and on ambiguous entities.

**Watch.** Brisk-S · reasoning high · 1 retry (L) clears multi-hop by only 0.3 points, ambiguous entities by 0.6 and date math by 0.9. If each task held its traffic share of the cases, a run this size would read them on the wrong side 37.4%, 29.9% and 16.5% of the time.

**Watch.** On the other side of the hold, Brisk-S · reasoning low · 1 retry (J) misses multi-hop by 1.4 points and ambiguous entities by 0.6, and sits on the line in date math. On the same assumed counts these read the wrong way 11.9%, 33.3% and 50% of the time; a rerun there could turn the hold into a pick.

**What would settle it.** First, p95 latency read within each task for the configurations nearest both lines; then the latency owner declaring whether the 600 ms budget is firm.

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

## Only two of the 18 configurations hold the 97% pass rate in every task, both at high reasoning with a retry; every other configuration falls short on multi-hop.

_Caveats: Per-task pass rates state no count._

|  | Brisk-S · reasoning high · 1 retry | Atlas-M · reasoning high · 1 retry | Atlas-M · reasoning high · no retry | Atlas-M · reasoning low · 1 retry | Brisk-S · reasoning high · no retry | Brisk-S · reasoning low · 1 retry |
|---|---|---|---|---|---|---|
| Single lookup | 98.5% | 99.2% | 98.7% | 99.2% | 97.6% | 98.4% |
| Multi-hop (2 to 3 calls) | 97.3% | 97.9% | 96.6% | 96.4% | 95.5% | 95.6% |
| Date and unit math | 97.9% | 98.6% | 97.7% | 97.8% | 96.6% | 97.0% |
| Ambiguous entity | 97.6% | 98.3% | 97.2% | 97.2% | 96.1% | 96.4% |

Pass rate by task for the six configurations nearest the 97% gate; the rest sit further below it.

## Both run far past the 600 ms latency budget overall, and since a configuration's slowest task is at least as slow as the whole, each misses the budget in at least one task.

_Caveats: Latency read only overall._

| Group | Row | From | To |
|---|---|---|---|
| Latency | L · Brisk-S · reasoning high · 1 retry | 364 ms | 870 ms |
| Latency | F · Atlas-M · reasoning high · 1 retry | 1094 ms | 2551 ms |
| Latency | E · Atlas-M · reasoning high · no retry | 1094 ms | 2430 ms |
| Latency | D · Atlas-M · reasoning low · 1 retry | 628 ms | 1500 ms |
| Latency | K · Brisk-S · reasoning high · no retry | 364 ms | 810 ms |
| Latency | J · Brisk-S · reasoning low · 1 retry | 209 ms | 513 ms |
| Latency | I · Brisk-S · reasoning low · no retry | 209 ms | 465 ms |
| Latency | S · Nano-XS · reasoning high · 1 retry | 170 ms | 450 ms |
|  | Latency budget (p95) gate, at most 600 ms in each task | 600 ms |  |

Latency from p50 to p95 over all tasks, against the 600 ms budget the gate sets within each task.

### Every configuration under the budget overall misses the pass rate in at least one task; the closest, Brisk-S · reasoning low · 1 retry (J), misses multi-hop by 1.4 points and ambiguous entities by 0.6, and sits exactly on the line in date math.

_Caveats: Per-task pass rates state no count._

|  | Brisk-S · reasoning low · 1 retry | Brisk-S · reasoning low · no retry | Nano-XS · reasoning high · 1 retry | Brisk-S · reasoning off · 1 retry | Brisk-S · reasoning off · no retry | Nano-XS · reasoning high · no retry | Nano-XS · reasoning low · 1 retry | Nano-XS · reasoning low · no retry | Nano-XS · reasoning off · 1 retry | Nano-XS · reasoning off · no retry |
|---|---|---|---|---|---|---|---|---|---|---|
| Single lookup | 98.4% | 97.4% | 95.8% | 96.6% | 94.3% | 93.0% | 95.0% | 91.7% | 92.0% | 86.8% |
| Multi-hop (2 to 3 calls) | 95.6% | 92.8% | 93.5% | 92.3% | 87.3% | 89.3% | 90.0% | 83.5% | 84.4% | 74.2% |
| Date and unit math | 97.0% | 95.1% | 94.7% | 94.4% | 90.8% | 91.1% | 92.5% | 87.6% | 88.2% | 80.5% |
| Ambiguous entity | 96.4% | 94.1% | 94.2% | 93.5% | 89.3% | 90.3% | 91.4% | 85.8% | 86.6% | 77.8% |

Pass rate by task for the ten configurations whose overall p95 is under the 600 ms budget, against the 97% gate.

| Example | Details | Reason | Excerpt |
|---|---|---|---|
| TQ-0418 · Multi-hop (2 to 3 calls) · Brisk-S · reasoning low · 1 retry · missing call | Input: Who chaired the committee that approved the 2019 grid budget?; Expected: search(committee, grid budget 2019) → lookup(chair) | Collapsed two hops into one query; returned the budget author, not the chair. | search(grid budget 2019 chair) |
| TQ-0772 · Date and unit math · Brisk-S · reasoning low · 1 retry · wrong args | Input: Filings in the 90 days before the March 3 hearing; Expected: date_from=2026-12-03, date_to=2027-03-03 | Off-by-one on the window start. The most common wrong-args pattern (9 of 17). | date_from=2026-12-04, date_to=2027-03-03 |
| TQ-1103 · Ambiguous entity · Brisk-S · reasoning low · 1 retry · wrong tool | Input: Latest on Mercury's pricing; Expected: ask_clarify(company or planet?) | Picked the company without asking. Scored wrong because the gold answer requires clarification. | search(Mercury pricing) |

Kept failures of the closest configuration, one each from multi-hop, date math and ambiguous entities: the first collapses two hops into one query, the second is an off-by-one date window in a task it holds exactly on the line, the third answers an ambiguous name without asking. How many failures each was kept from is not stated; they show kinds of failure, not shares.

### Of the two that hold the pass rate, Atlas-M · reasoning high · 1 retry (F) costs 5.5 times as much as Brisk-S · reasoning high · 1 retry (L) as this run served them, with no provider named for either, so the ratio may not hold under another routing; their pass rates are not told apart.

_Caveats: No arm is the control._

| Row | Value |
|---|---|
| F · Atlas-M · reasoning high · 1 retry | $3.48 |
| L · Brisk-S · reasoning high · 1 retry | $0.63 |
| J · Brisk-S · reasoning low · 1 retry | $0.30 |

The bars are drawn on a log scale: their values span more than ten times, so each step is a multiple.

Cost per 1,000 calls at this run's unnamed routing. If the declared monthly volume counts calls, Atlas-M costs $11,376 more a month; if it counts tasks, which each make more than one call, the gap is larger.

| Row | Value | 95% unpaired interval |
|---|---|---|
| Atlas-M · reasoning high · 1 retry → Brisk-S | -0.7 | -1.7 to +0.4 |

Each effect is one setting moved between two arms that differ in it alone, on Pass rate. Descriptive: read as two arms' own spreads, never a test.

Pass rate change moving from Atlas-M to Brisk-S at high reasoning with a retry, computed before the rates are rounded, with its unpaired interval, which crosses zero.

## What you could change

| What you could change | What it decides | What happens as it moves |
|---|---|---|
| Latency budget (p95) | Which configurations can ship at all | Below 870 ms, no budget admits a configuration that holds the pass rate everywhere. To admit Brisk-S · reasoning high · 1 retry (L), the budget must cover its slowest task's p95, which is at least that overall figure and not yet read. |
| Pass rate gate | Whether a configuration within the budget qualifies | At a line of 95.6% or lower, Brisk-S · reasoning low · 1 retry (J) holds every task while staying under the budget overall; above it, no configuration within the budget does. |

## Next steps

- Read p95 latency within each task for Brisk-S · reasoning high · 1 retry (L), Atlas-M · reasoning high · 1 retry (F) and Brisk-S · reasoning low · 1 retry (J). _Whether the latency gate fails in each task as the overall p95 implies, how high a budget would have to go to admit the configurations that hold the pass rate, and whether single lookups, 40% of traffic, could go to a configuration within the budget while the harder tasks go elsewhere._
- The latency owner declares whether the 600 ms budget is firm. To admit Brisk-S · reasoning high · 1 retry (L) it would have to rise to at least 870 ms, its exact level pending the per-task read. _Whether Brisk-S · reasoning high · 1 retry (L) is a candidate at all; if it is, it costs less than the only other configuration that holds every task._
- Rerun Brisk-S · reasoning low · 1 retry (J) on multi-hop and ambiguous-entity cases: confirming its misses four times in five takes about 1,685 and 7,567 cases respectively, on assumed task counts. _Whether the closest configuration within the budget truly misses, or holds every task and becomes the pick._
- Fix Brisk-S's multi-hop decomposition and its clarifying question on ambiguous names at low reasoning with a retry, then rerun those two tasks. _Whether a configuration within the budget can hold the pass rate everywhere, which would make it the pick._
- If the budget rises, rerun Brisk-S · reasoning high · 1 retry (L) on fresh multi-hop cases: confirming its thin clearance four times in five takes about 22,911 cases, on assumed task counts. _Whether its multi-hop pass rate truly clears the gate or was the luck of the best of many._

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
| Single lookup | Pass rate | at least 97% | Passes (97.2%) | Passes (98.3%) | Passes (98.6%) | Passes (99.2%) | Passes (98.7%) | Passes (99.2%) | Fails (94.3%) | Fails (96.6%) | Passes (97.4%) | Passes (98.4%) | Passes (97.6%) | Passes (98.5%) | Fails (86.8%) | Fails (92%) | Fails (91.7%) | Fails (95%) | Fails (93%) | Fails (95.8%) |
| Multi-hop (2 to 3 calls) | Pass rate | at least 97% | Fails (90.2%) | Fails (94.1%) | Fails (94.1%) | Fails (96.4%) | Fails (96.6%) | Passes (97.9%) | Fails (87.3%) | Fails (92.3%) | Fails (92.8%) | Fails (95.6%) | Fails (95.5%) | Passes (97.3%) | Fails (74.2%) | Fails (84.4%) | Fails (83.5%) | Fails (90%) | Fails (89.3%) | Fails (93.5%) |
| Date and unit math | Pass rate | at least 97% | Fails (93.7%) | Fails (96.2%) | Fails (96.4%) | Passes (97.8%) | Passes (97.7%) | Passes (98.6%) | Fails (90.8%) | Fails (94.4%) | Fails (95.1%) | Passes (97%) | Fails (96.6%) | Passes (97.9%) | Fails (80.5%) | Fails (88.2%) | Fails (87.6%) | Fails (92.5%) | Fails (91.1%) | Fails (94.7%) |
| Ambiguous entity | Pass rate | at least 97% | Fails (92.2%) | Fails (95.3%) | Fails (95.4%) | Passes (97.2%) | Passes (97.2%) | Passes (98.3%) | Fails (89.3%) | Fails (93.5%) | Fails (94.1%) | Fails (96.4%) | Fails (96.1%) | Passes (97.6%) | Fails (77.8%) | Fails (86.6%) | Fails (85.8%) | Fails (91.4%) | Fails (90.3%) | Fails (94.2%) |
| Single lookup | Latency budget (p95) | at most 600 ms | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) |
| Multi-hop (2 to 3 calls) | Latency budget (p95) | at most 600 ms | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) |
| Date and unit math | Latency budget (p95) | at most 600 ms | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) |
| Ambiguous entity | Latency budget (p95) | at most 600 ms | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) | Unknown (no figure) |

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

- **The Nano-XS configurations as leads** Each misses the pass rate in every task; they appear only in the under-budget table.
- **Overall pass rate per configuration** The gate reads each task; the overall rate hides the multi-hop shortfall that decides it.
- **Retry and reasoning effects across the whole grid** A retry raises the pass rate in every configuration; the grid's effects describe how the two leading configurations were reached, not which ships. They stay in the appendix.
- **Calls per task and p50 latency** Neither is gated, and neither changes which configuration clears both lines.
- **Extra calls and outcome measures** The snapshot holds no figures for them.

### Method

- **Design** 18 configurations crossing three models, three reasoning levels and retry on or off, each run on 1,200 cases with none lost. There is no baseline configuration. _Caveats: No arm is the control._
- **Tasks** Single lookup, multi-hop, date and unit math, and ambiguous entity, with traffic shares of 40%, 25%, 20% and 15%. Cases per task are not stated; case counts and wrong-side odds quoted for reruns assume each task held its traffic share of the cases. _Caveats: Per-task pass rates state no count._
- **Pass** The campaign does not define a pass in words; the failure cards compare the planner's tool calls and arguments with the expected ones.
- **Gates** Pass rate at least 97% in each task, and p95 latency at most 600 ms in each task. Both are read on point estimates; per-task pass rates carry no interval.
- **Latency** Only p50 and p95 over all tasks were recorded, so the per-task latency gate has no figure for any configuration. A configuration's slowest task has a p95 at least as high as its overall one. _Caveats: Latency read only overall._
- **Intervals** The one interval shown is 95%, unpaired, from each configuration's binomial spread over its cases; it reads wider than a paired one would.
- **Cost** Cost per 1,000 calls as this run was served; no provider is named for any configuration, so the routing behind the costs is unknown and the cost order may not hold under another. Monthly figures use the declared volume of 4,000,000, read as calls; the campaign does not say what it counts. How many trials reported a cost is not stated.
- **Source** Synthetic example data from docs/eval-report-kit/references/r2-config-sweep.json, run 2026-10-23.

### Review findings left open

- **Should fix** · The page: The third next step reruns Brisk-S low with a retry on multi-hop and ambiguous-entity cases and says the result settles whether it 'holds every task and becomes the pick'. Two things that step does not settle also stand in the way. Its date-math pass rate is exactly on the 97% line, so on assumed counts it reads the wrong way half the time. And its overall p95 is 513 ms, so its slowest task is at least that slow and has not been read against 600 ms. Fix: add date-math cases to the rerun, and change the outcome to 'becomes the pick only if date math also holds and its per-task p95 stays under 600 ms'. Match the confidence clause's 'pending per-task latency' wording.

_Written by claude-opus-5-5, served as claude-opus-5-5, for $3.47, with analyst skill 2a7662d928b0, on 2026-10-29._
