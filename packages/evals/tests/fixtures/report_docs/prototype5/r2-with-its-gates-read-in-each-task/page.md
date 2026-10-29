Choice · candidates compared · Research query planner · 18 arms · 4 tasks · Run 2026-10-23 · Synthetic example data

**Draft: its editor's review left 1 finding open.**

# No single configuration meets both gates: the two reaching 97% in every task exceed 600 ms p95 overall; only a split of tasks remains open.

With each case in one task, Brisk-S high + retry's overall p95 of 870 ms puts at least one task over budget, and every configuration inside the budget overall misses the pass-rate gate somewhere. Whether a split clears both, a fast configuration on some tasks and high reasoning on the rest, waits on latency read per task.

**Both gates are declared per task, but latency was recorded only over all tasks, so no configuration is shown to pass the latency budget in any task.**

**Confidence: Low.**

- **Latency read only over all tasks.** It decides whether a split of tasks is possible: until p95 is read per task, no task-level assignment can be shown to meet the budget. It does not reopen a single configuration unless a case can belong to more than one task, since an overall p95 above 600 ms otherwise puts at least one task over it.
- **Task sizes unstated.** Only where a configuration sits within about a point of the 97% line in a task, since no per-task interval can be read. Many cells do, among them Brisk-S high + retry on multi-hop, Brisk-S low + retry on date math (97%) and ambiguous entities (96.4%), Atlas-M high without retry on multi-hop (96.6%), Atlas-M low + retry on multi-hop (96.4%) and Brisk-S high without retry on date math (96.6%); the full set is in the gate table. Any of them could fall either side, which changes which configurations a split could use.
- **The pick is the best of several candidates.** It bears on Brisk-S high + retry, picked from eighteen configurations and clearing multi-hop by 0.3 points: a rerun on fresh cases tends to read the best of many lower, which would put it below the line there and leave multi-hop without its faster candidate even in a split.
- **No arm is the control.** Only for a claim that one configuration beats another: every comparison here is unpaired, so gaps between configurations read wider than paired ones would; no claim on this page rests on separating two of them overall.

**What is tangled.** The pass-rate gate and the latency budget pull opposite ways: the configurations that reach 97% on multi-hop use high reasoning, and high reasoning is what pushes p95 past 600 ms. The budget is declared per task, but latency was read only over all tasks, so whether a split of tasks clears both cannot be read.

**What we know despite it.** No single configuration clears both gates if each case belongs to one task. Of the two that clear pass rate everywhere, Brisk-S high + retry cost $2.84 less per thousand calls than Atlas-M high + retry at this run's serving, whose provider for either model is not named, and their overall pass rates are not told apart. Nano-XS misses the pass-rate gate in all four tasks even at its strongest setting.

**Watch.** Every leg of the split is thin. Read as if each task held its traffic share of the cases, the weakest is Brisk-S low + retry on date math, exactly on the line, so a rerun is as likely to read it below as above (50%); Brisk-S high + retry clears multi-hop by 0.3 points (below on a rerun 37.4% of the time) and ambiguous entities by 0.6 (29.9%). Date math could go to Brisk-S high + retry (97.9%) if its latency there allows.

**What would settle it.** Read p95 per task from this run's traces for Brisk-S low + retry and Brisk-S high + retry, and state each task's case count. A split is open only if Brisk-S high + retry is under budget on multi-hop and ambiguous entities while Brisk-S low + retry is under it on the other tasks.

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

## Only Atlas-M high + retry and Brisk-S high + retry reach 97% in all four tasks; multi-hop, 25% of traffic, is the task every other configuration misses.

_Caveats: Task sizes unstated._

| Slice | Arm | Pass rate |
|---|---|---|
| Single lookup | F · Atlas-M · reasoning high · 1 retry | 99.2% |
| Single lookup | L · Brisk-S · reasoning high · 1 retry | 98.5% |
| Single lookup | D · Atlas-M · reasoning low · 1 retry | 99.2% |
| Single lookup | E · Atlas-M · reasoning high · no retry | 98.7% |
| Single lookup | J · Brisk-S · reasoning low · 1 retry | 98.4% |
| Single lookup | K · Brisk-S · reasoning high · no retry | 97.6% |
| Multi-hop (2 to 3 calls) | F · Atlas-M · reasoning high · 1 retry | 97.9% |
| Multi-hop (2 to 3 calls) | L · Brisk-S · reasoning high · 1 retry | 97.3% |
| Multi-hop (2 to 3 calls) | D · Atlas-M · reasoning low · 1 retry | 96.4% |
| Multi-hop (2 to 3 calls) | E · Atlas-M · reasoning high · no retry | 96.6% |
| Multi-hop (2 to 3 calls) | J · Brisk-S · reasoning low · 1 retry | 95.6% |
| Multi-hop (2 to 3 calls) | K · Brisk-S · reasoning high · no retry | 95.5% |
| Date and unit math | F · Atlas-M · reasoning high · 1 retry | 98.6% |
| Date and unit math | L · Brisk-S · reasoning high · 1 retry | 97.9% |
| Date and unit math | D · Atlas-M · reasoning low · 1 retry | 97.8% |
| Date and unit math | E · Atlas-M · reasoning high · no retry | 97.7% |
| Date and unit math | J · Brisk-S · reasoning low · 1 retry | 97.0% |
| Date and unit math | K · Brisk-S · reasoning high · no retry | 96.6% |
| Ambiguous entity | F · Atlas-M · reasoning high · 1 retry | 98.3% |
| Ambiguous entity | L · Brisk-S · reasoning high · 1 retry | 97.6% |
| Ambiguous entity | D · Atlas-M · reasoning low · 1 retry | 97.2% |
| Ambiguous entity | E · Atlas-M · reasoning high · no retry | 97.2% |
| Ambiguous entity | J · Brisk-S · reasoning low · 1 retry | 96.4% |
| Ambiguous entity | K · Brisk-S · reasoning high · no retry | 96.1% |

Pass rate per task for the six configurations nearest the gate, which is 97% in each task; task sizes are unstated, so no per-task interval is shown.

### Both run past the 600 ms budget over all tasks, Brisk-S high + retry at 870 ms and Atlas-M high + retry at 2551 ms, so each has at least one task over budget; every configuration inside the budget overall misses the pass-rate gate in at least one task.

_Caveats: Latency read only over all tasks._

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

Median to p95 latency over all tasks for each configuration, against the per-task budget line.

### On Brisk-S with retry, high reasoning over low lifts multi-hop by +1.7 points, from 95.6% below the line to 97.3% above it, and takes overall p95 from 513 ms to 870 ms; overall, the gain is not separated (+0.8 points, interval -0.4 to +2.1).

_Caveats: Task sizes unstated._

| Row | Value | 95% unpaired interval |
|---|---|---|
| Overall | +0.8 | -0.4 to +2.1 |
| Single lookup | +0.1 |  |
| Multi-hop (2 to 3 calls) | +1.7 |  |
| Date and unit math | +0.9 |  |
| Ambiguous entity | +1.2 |  |

Each row is Brisk-S · reasoning high · 1 retry less Brisk-S · reasoning low · 1 retry on Pass rate, overall and in each place both hold it.

The interval is independent: each arm's own spread (a pass/fail rate's binomial spread, from the arm's cases), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same.

Brisk-S high + retry minus Brisk-S low + retry on pass rate, overall with its unpaired interval and per task as points with no interval, since task sizes are unstated.

| Group | Row | From | To |
|---|---|---|---|
| Latency | J · Brisk-S · reasoning low · 1 retry | 209 ms | 513 ms |
| Latency | L · Brisk-S · reasoning high · 1 retry | 364 ms | 870 ms |
|  | Latency budget (p95) gate, at most 600 ms in each task | 600 ms |  |

The same two configurations' median to p95 latency over all tasks, against the per-task budget line.

#### Brisk-S high + retry is not told apart from Atlas-M high + retry on overall pass rate (-0.7 points, interval -1.7 to +0.4) and cost 0.18 times as much per thousand calls at this run's serving; the snapshot names no provider for either model, so the ratio is this routing's, not the models'.

_Caveats: No arm is the control._

| Row | Value | 95% unpaired interval |
|---|---|---|
| Overall | -0.7 | -1.7 to +0.4 |
| Single lookup | -0.7 |  |
| Multi-hop (2 to 3 calls) | -0.6 |  |
| Date and unit math | -0.7 |  |
| Ambiguous entity | -0.7 |  |

Each row is Brisk-S · reasoning high · 1 retry less Atlas-M · reasoning high · 1 retry on Pass rate, overall and in each place both hold it.

The interval is independent: each arm's own spread (a pass/fail rate's binomial spread, from the arm's cases), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same.

Brisk-S high + retry minus Atlas-M high + retry on pass rate, overall with its unpaired interval and per task as points.

#### Brisk-S low + retry, the fastest configuration near the line, falls short on multi-hop and ambiguous entities and sits exactly on the line on date math; its kept failures are planning errors, a collapsed hop, an off-by-one date window and a missing clarifying question, not malformed output.

_Caveats: Task sizes unstated._

| Slice | Arm | Pass rate |
|---|---|---|
| Single lookup | J · Brisk-S · reasoning low · 1 retry | 98.4% |
| Multi-hop (2 to 3 calls) | J · Brisk-S · reasoning low · 1 retry | 95.6% |
| Date and unit math | J · Brisk-S · reasoning low · 1 retry | 97.0% |
| Ambiguous entity | J · Brisk-S · reasoning low · 1 retry | 96.4% |

Brisk-S low + retry's pass rate in each task against the 97% gate; with task sizes unstated, its smaller misses could fall either side on a rerun.

| Example | Details | Reason | Excerpt |
|---|---|---|---|
| TQ-0418 · Multi-hop (2 to 3 calls) · Brisk-S · reasoning low · 1 retry · missing call | Input: Who chaired the committee that approved the 2019 grid budget?; Expected: search(committee, grid budget 2019) → lookup(chair) | Collapsed two hops into one query; returned the budget author, not the chair. | search(grid budget 2019 chair) |
| TQ-0772 · Date and unit math · Brisk-S · reasoning low · 1 retry · wrong args | Input: Filings in the 90 days before the March 3 hearing; Expected: date_from=2026-12-03, date_to=2027-03-03 | Off-by-one on the window start. The most common wrong-args pattern (9 of 17). | date_from=2026-12-04, date_to=2027-03-03 |
| TQ-1103 · Ambiguous entity · Brisk-S · reasoning low · 1 retry · wrong tool | Input: Latest on Mercury's pricing; Expected: ask_clarify(company or planet?) | Picked the company without asking. Scored wrong because the gold answer requires clarification. | search(Mercury pricing) |

The failures kept for Brisk-S low + retry, one per task where it fell short or sat on the line; a sample, not a count.

## What you could change

| What you could change | What it decides | What happens as it moves |
|---|---|---|
| The pass-rate gate | Which configurations clear quality in each task | Lowering it admits more configurations on quality, but no pass-rate threshold yields a pick while latency per task is unmeasured. |
| The latency budget | Which configurations clear latency in each task | No budget value yields a pick either: with latency read only over all tasks, every configuration stays undecided on it in each task. |

## Next steps

- Read p95 per task from this run's traces for Brisk-S low + retry and Brisk-S high + retry, and record each task's case count. _Whether a split, Brisk-S high + retry on multi-hop and ambiguous entities (and date math, if its latency allows) and Brisk-S low + retry elsewhere, can meet the latency budget, and whether each leg's pass rate is borderline or clear._
- Declare whether tasks may be served by different configurations, and whether the latency budget is meant per task for multi-hop questions that make several calls by design. _Whether a split by task is an allowed option, and which latency line multi-hop is held to._
- Run Brisk-S low reasoning with a multi-hop decomposition fix, or a second retry, on multi-hop and ambiguous-entity cases. _Whether a configuration inside the latency budget overall can reach the pass-rate gate on the two tasks where it now falls short._

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

- **The remaining Atlas-M and Brisk-S configurations with reasoning off, or low without retry** Each misses the pass-rate gate in several tasks, and none is closer to clearing both gates than the rows shown; the appendix holds every figure.
- **Nano-XS in detail** It misses the pass-rate gate in every task under every setting, so it bears on no option; one line in the decision block.
- **What the retry buys, setting by setting** It raises pass rate at each reasoning level on Brisk-S but moves no configuration across both gates; the appendix holds every lever row.
- **Calls per task** No gate reads it and it does not separate the near-line configurations.
- **Cost per month** The call does not turn on cost, and the campaign does not say whether its monthly volume counts calls or tasks.
- **Outcome mix and extra calls** The snapshot holds no figures for either.
- **Model and reasoning effects on Atlas-M and Nano-XS** They point the same way as on Brisk-S and change no option.

### Method

- **Design** eighteen configurations: three models crossed with three reasoning levels and retry on or off, each run on 1,200 cases with none lost; no configuration is a control. _Caveats: No arm is the control._
- **Pass** The campaign does not define what makes a task pass; the kept failures compare the planner's calls with an expected plan. The gate is 97% in each of the four tasks, read on the point estimate.
- **Latency** Median and p95 per configuration over all tasks only; the 600 ms budget is declared per task. An overall p95 above the budget is read as at least one task over it, which assumes each case belongs to one task. _Caveats: Latency read only over all tasks._
- **Intervals** Overall pass-rate differences use each configuration's binomial spread, at the level each block prints, combined as independent and uncorrected for the number of comparisons. Per-task pass rates state no case count and carry no interval; the split's thin legs are sized as if each task held its traffic share of the cases, an assumption labelled as such. _Caveats: Task sizes unstated; The pick is the best of several candidates._
- **Cost** Cost per thousand calls as measured in this run; the provider routing behind each model is not named, so the cost order holds only for this run's serving.
- **Source** Synthetic example data from docs/eval-report-kit/references/r2-config-sweep.json, run 2026-10-23.

### Review findings left open

- **Should fix** · The page: Watch opens with 'Every leg of the split is thin.' The single-lookup leg is not: Brisk-S low + retry reads 98.4%, 1.4 points over the line, and at its traffic-share size would read below it only 0.7% of the time on a rerun. Change it to 'three of the four legs are thin', or name the three legs (date math, multi-hop, ambiguous entities) so the sentence matches the figures that follow it.

_Written by claude-opus-5-5, served as claude-opus-5-5, for $4.33, with analyst skill ed7e2c60b5ad, on 2026-10-29._
