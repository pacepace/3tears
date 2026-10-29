Choice · candidates compared · Research query planner · 18 arms · 4 tasks · Run 2026-10-23 · Synthetic example data

# No configuration is shown to pass both gates: only two, Atlas-M and Brisk-S with high reasoning and retry, hold 97% in every task, and both run over the 600 ms latency budget.

Multi-hop lookups are every configuration's weakest task; the configuration that comes closest inside the budget, Brisk-S with low reasoning and retry, falls short there by 1.4 points. Per-task latency, never recorded, could open a routed mix nobody has run.

**Confidence: High, if the caveats below aren't material.**

- **No arm is the control.** Only for a claim that one configuration beats another: the call reads each configuration against the declared gates.
- **The pick is the best of several candidates.** Only where a configuration clears a gate by little. Brisk-S with high reasoning and retry clears multi-hop by 0.3 points, best of eighteen; a fresh run would likely read it lower. That strengthens the headline but weakens the fallback and the routed mix, both of which lean on its thin clearances.
- **Task pass rates state no case count.** If the true per-task rates of the two Brisk-S configurations with retry sit on the other side of the gate from today's readings. High reasoning would lose the fallback if its multi-hop or ambiguous-entity pass flipped. Low reasoning would become a single-configuration candidate only if both its misses flipped, multi-hop and ambiguous entities, and its date math, exactly on the line, held.
- **Latency recorded only over all cases.** Only for a routed mix nobody ran: if each configuration's tail latency stays inside the budget in the tasks it would serve, the mix could clear both gates. It cannot rescue a single high-reasoning configuration, whose tail over all cases is already over budget.

**What is tangled.** On multi-hop lookups the two gates pull against each other: only high reasoning with retry reaches 97% there, and both such passers run over 600 ms on tail latency. Ship none under the gates as declared.

**What we know despite it.** If every case belongs to exactly one task, neither pass-rate passer can meet the latency gate on its own, since each one's tail over all cases is over budget. On today's readings every configuration inside the budget misses multi-hop, the nearest by 1.4 points, a miss not yet settled.

**Watch.** The Brisk-S readings nearest the gate are all thin and carry no case count. If each task held its traffic share of cases, a run of today's size would read high reasoning's multi-hop pass below the line 37.4% of the time and its ambiguous-entity pass 29.9%; it would read low reasoning's multi-hop miss above the line 11.9% and its ambiguous-entity miss 33.3%. Low reasoning's date math sits exactly on the line.

**What would settle it.** Per-task tail latency for a routed mix: Brisk-S with high reasoning and retry on multi-hop and ambiguous-entity lookups (25% and 15% of traffic), Brisk-S with low reasoning and retry on single lookups and date math. Its legs also need their thin pass rates settled. If the mix fails, the planner's owner has to say which gate gives way on multi-hop.

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

## Only two of the eighteen configurations reach 97% in every task, Atlas-M and Brisk-S with high reasoning and retry; multi-hop lookups are the weakest task for every configuration.

|  | Atlas-M · reasoning off · no retry | Atlas-M · reasoning off · 1 retry | Atlas-M · reasoning low · no retry | Atlas-M · reasoning low · 1 retry | Atlas-M · reasoning high · no retry | Atlas-M · reasoning high · 1 retry | Brisk-S · reasoning off · no retry | Brisk-S · reasoning off · 1 retry | Brisk-S · reasoning low · no retry | Brisk-S · reasoning low · 1 retry | Brisk-S · reasoning high · no retry | Brisk-S · reasoning high · 1 retry | Nano-XS · reasoning off · no retry | Nano-XS · reasoning off · 1 retry | Nano-XS · reasoning low · no retry | Nano-XS · reasoning low · 1 retry | Nano-XS · reasoning high · no retry | Nano-XS · reasoning high · 1 retry |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Single lookup | 97.2% | 98.3% | 98.6% | 99.2% | 98.7% | 99.2% | 94.3% | 96.6% | 97.4% | 98.4% | 97.6% | 98.5% | 86.8% | 92.0% | 91.7% | 95.0% | 93.0% | 95.8% |
| Multi-hop (2 to 3 calls) | 90.2% | 94.1% | 94.1% | 96.4% | 96.6% | 97.9% | 87.3% | 92.3% | 92.8% | 95.6% | 95.5% | 97.3% | 74.2% | 84.4% | 83.5% | 90.0% | 89.3% | 93.5% |
| Date and unit math | 93.7% | 96.2% | 96.4% | 97.8% | 97.7% | 98.6% | 90.8% | 94.4% | 95.1% | 97.0% | 96.6% | 97.9% | 80.5% | 88.2% | 87.6% | 92.5% | 91.1% | 94.7% |
| Ambiguous entity | 92.2% | 95.3% | 95.4% | 97.2% | 97.2% | 98.3% | 89.3% | 93.5% | 94.1% | 96.4% | 96.1% | 97.6% | 77.8% | 86.6% | 85.8% | 91.4% | 90.3% | 94.2% |

Pass rate by task and configuration; cells under the 97% gate are outlined.

## Both run over the latency budget on tail latency across all cases: 870 ms for Brisk-S and 2551 ms for Atlas-M, against 600 ms. Latency was kept only over all cases, so the per-task gate has no figure; but if every case belongs to exactly one task, each must miss the budget in some task, so per-task latency can rescue only a routed mix, not either configuration alone.

_Caveats: Latency recorded only over all cases._

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

Median to tail latency (p95) over all cases for each configuration, the 600 ms budget marked.

### Brisk-S with retry sits at the gate at both reasoning levels: with low reasoning (513 ms) it misses two tasks, multi-hop by 1.4 points and ambiguous entities by 0.6; with high reasoning it clears multi-hop by only 0.3 and ambiguous entities by 0.6. With no case counts, neither the misses nor the passes are settled.

_Caveats: Task pass rates state no case count; The pick is the best of several candidates._

| Panel | Row | Value |
|---|---|---|
| Single lookup | J · Brisk-S · reasoning low · 1 retry | 98.4% |
| Single lookup | L · Brisk-S · reasoning high · 1 retry | 98.5% |
| Single lookup | Pass rate gate, at least 97% | 97.0% |
| Multi-hop (2 to 3 calls) | J · Brisk-S · reasoning low · 1 retry | 95.6% |
| Multi-hop (2 to 3 calls) | L · Brisk-S · reasoning high · 1 retry | 97.3% |
| Multi-hop (2 to 3 calls) | Pass rate gate, at least 97% | 97.0% |
| Date and unit math | J · Brisk-S · reasoning low · 1 retry | 97.0% |
| Date and unit math | L · Brisk-S · reasoning high · 1 retry | 97.9% |
| Date and unit math | Pass rate gate, at least 97% | 97.0% |
| Ambiguous entity | J · Brisk-S · reasoning low · 1 retry | 96.4% |
| Ambiguous entity | L · Brisk-S · reasoning high · 1 retry | 97.6% |
| Ambiguous entity | Pass rate gate, at least 97% | 97.0% |

Per-task pass rate for Brisk-S with retry at low and at high reasoning, the gate marked; task rates state no case count, so no interval is drawn.

#### Of the two that hold the pass rate in every task, Atlas-M costs 5.5 times as much as Brisk-S, $11,376 more a month.

| Row | Value |
|---|---|
| F · Atlas-M · reasoning high · 1 retry | $3.48 |
| L · Brisk-S · reasoning high · 1 retry | $0.63 |
| J · Brisk-S · reasoning low · 1 retry | $0.30 |

The bars are drawn on a log scale: their values span more than ten times, so each step is a multiple.

Cost per thousand calls for the two pass-rate passers and the nearest miss inside the budget, on a log scale.

#### The kept multi-hop failure of Brisk-S with low reasoning and retry merges two hops into one search; the date and ambiguous-entity cards fail other ways, an off-by-one window and a missing clarifying question.

| Example | Details | Reason | Excerpt |
|---|---|---|---|
| TQ-0418 · Multi-hop (2 to 3 calls) · Brisk-S · reasoning low · 1 retry · missing call | Input: Who chaired the committee that approved the 2019 grid budget?; Expected: search(committee, grid budget 2019) → lookup(chair) | Collapsed two hops into one query; returned the budget author, not the chair. | search(grid budget 2019 chair) |
| TQ-0772 · Date and unit math · Brisk-S · reasoning low · 1 retry · wrong args | Input: Filings in the 90 days before the March 3 hearing; Expected: date_from=2026-12-03, date_to=2027-03-03 | Off-by-one on the window start. The most common wrong-args pattern (9 of 17). | date_from=2026-12-04, date_to=2027-03-03 |
| TQ-1103 · Ambiguous entity · Brisk-S · reasoning low · 1 retry · wrong tool | Input: Latest on Mercury's pricing; Expected: ask_clarify(company or planet?) | Picked the company without asking. Scored wrong because the gold answer requires clarification. | search(Mercury pricing) |

The failures kept for this configuration, one per task where a failure was kept; the input does not say how many failed in all.

## What you could change

| What you could change | What it decides | What happens as it moves |
|---|---|---|
| Latency budget | Which configurations pass the latency gate in each task | No budget value changes the call: latency has no per-task figure, so the gate reads nothing at any threshold. Read over all cases instead, a budget at or above 870 ms would admit Brisk-S with high reasoning and retry, subject to its thin multi-hop and ambiguous-entity passes. |
| Pass-rate gate | Which configurations hold the pass rate in every task | Lowering it admits more configurations on pass rate, Brisk-S with low reasoning and retry once the gate is lowered to 95.6%, but no configuration passes both gates while latency has no per-task figure. |

## Next steps

- Record tail latency per task for Brisk-S with high reasoning and retry on multi-hop and ambiguous-entity lookups, and for Brisk-S with low reasoning and retry on single lookups and date math, from existing traces if kept, else by re-running. _Whether the routed mix can meet the budget in each task it serves; the mix would then need its own run before shipping._
- Re-run Brisk-S with high reasoning and retry on multi-hop and ambiguous-entity lookups, stating per-task counts. As a what-if, if tasks held their traffic share of cases, a run sized to settle most of the time needs 22,911 multi-hop and 5,108 ambiguous-entity cases. _Whether it truly clears both tasks; the routed mix and any looser-budget fallback depend on it._
- Re-run Brisk-S with low reasoning and retry on multi-hop, ambiguous-entity and date-math lookups, stating per-task counts. On the same what-if, settling most of the time needs 1,685 multi-hop and 7,567 ambiguous-entity cases; date math sits exactly on the line, so no run settles it unless the true rate is away from the line. _Whether its misses are real and whether it holds date math, its leg of the routed mix; only if both misses prove false and date math holds is it a single fast candidate._
- Improve multi-hop planning for Brisk-S at low reasoning, where the kept failure merges hops into one search, and re-run multi-hop. _Whether a configuration inside the budget can reach the gate on multi-hop without high reasoning._
- The planner's owner declares which gate gives way on multi-hop if neither a single configuration nor the routed mix passes both: a looser latency budget, or a lower pass-rate gate for that task. _Turns the diagnosis into a decision; under a looser budget, Brisk-S with high reasoning and retry is the cheapest pass-rate passer, if its re-run confirms it._

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

- **Overall pass rate per configuration, and setting effects read on it** The gate reads each task; several configurations clear it overall yet miss multi-hop, so overall figures would mislead. The options table still prints it, labelled in Method; the reasoning step that matters is visible per task in the Brisk-S chart.
- **Calls per task and median latency** Neither bears on a declared gate.
- **Extra calls per task and outcome** The snapshot holds no figures for them.
- **Model and reasoning effects on Atlas-M and Nano-XS** Nano-XS reaches the gate in no task, and every Atlas-M configuration that reaches it on multi-hop is over the budget across all cases.
- **Other thin pass-rate misses (Atlas-M high reasoning without retry on multi-hop, Brisk-S high reasoning without retry on date math)** Both configurations are over the latency budget across all cases, so settling their misses would not produce a passer.
- **Any cost or gain for a routed mix** Nobody ran one; it appears only as a next step.
- **A lean toward Brisk-S with high reasoning and retry as the fallback** Weighed and kept as conditional only: its multi-hop and ambiguous-entity passes are as unsettled as the low-reasoning misses, so it rests on the re-run in Next.

### Method

- **Cases** 1,200 cases per configuration, with no repeats stated. The input does not say how cases split across tasks, only each task's share of traffic: 40% single lookups, 25% multi-hop, 20% date and unit math, 15% ambiguous entities.
- **Pass** The input does not define what makes a task pass; its failure classes are a missing call, wrong arguments and the wrong tool.
- **Gates** Pass rate at least 97% in each task, and tail latency (p95) at most 600 ms in each task, both declared before the run. Latency was recorded only over all cases, so the latency gate has no figure in any task; a configuration with no failing figure is undecided, not failed.
- **Options table** Its pass-rate column is the overall pass rate across all tasks. The gate reads each task, so an overall rate at or above the gate does not mean a configuration passes; read the per-task chart instead.
- **Assumed counts** Run sizes and wrong-side odds for per-task rates read each task as holding its traffic share of a configuration's cases, a what-if because the input states no per-task count; on that reading multi-hop holds 300 cases.
- **Intervals** Per-task pass rates state no case count and carry no interval. There is no control, so nothing is paired, and no multiplicity correction applies.
- **Cost** Cost per thousand calls as measured. Monthly figures multiply it by the declared volume of 4,000,000, and the input does not say what that volume counts.
- **Source** Synthetic example data from docs/eval-report-kit/references/r2-config-sweep.json

### Review findings left open

- **Should fix** · Claim 5: Next step 4 ('Improve multi-hop planning… where the kept failure merges hops into one search') rests on one kept card, TQ-0418. The snapshot doesn't say how many multi-hop failures there were for Brisk-S with low reasoning and retry, so hop-collapse may not be the main pattern. Fix: make step 4 conditional, e.g. first tally that configuration's multi-hop failures by pattern from the traces, then target planning only if hop-collapse dominates. c-fail could say in its sentence, not only its caption, that this is one example.

_Written by claude-opus-5-5, served as claude-opus-5-5, for $4.12, with analyst skill 1677ccb31cd6, on 2026-10-29._
