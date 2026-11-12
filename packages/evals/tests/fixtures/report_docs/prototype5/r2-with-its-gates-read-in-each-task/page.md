Choice · candidates compared · Research query planner · 18 arms · 4 tasks · Run 2026-10-23 · Synthetic example data

_1 review finding open, listed at the end._

# No configuration clears both gates: the two that hold a 97% pass rate in every task run past the 600 ms latency budget, and every one inside the budget misses the pass rate, always on multi-hop.

Brisk-S with high reasoning and a retry holds the pass rate everywhere, its thinnest task 0.3 points over the gate, but reaches 870 ms at p95. A budget that admits it, or a low-reasoning configuration that fixes multi-hop, would change the answer.

**Pass rates over all tasks, as the options table prints them, are not the gate: it reads each task. Latency was measured only over all tasks, so every latency verdict here is inferred from the overall p95.**

**Confidence: High, if the caveats below aren't material.**

- **No arm is the control.** Only for the claim that Brisk-S and Atlas-M at high reasoning with a retry are not told apart: with no control they compare unpaired, which reads wider than a paired test would. The finding that nothing clears both gates compares no two configurations.
- **The pick is the best of several candidates.** Only for Brisk-S with high reasoning and a retry clearing multi-hop by 0.3 points: as the best of 18 configurations it likely reads high, so a fresh run would more likely put it under the gate, which still leaves nothing clearing both.
- **Per-task latency unmeasured.** Only if a configuration inside the budget over all tasks also cleared the pass rate; none does today. For the two that hold the pass rate, a p95 over all tasks past the budget means at least one task is past it too, provided each case belongs to one task.
- **Task pass rates state no count.** Only if Brisk-S with low reasoning and a retry truly clears multi-hop and ambiguous entities. Under counts assumed from traffic shares, a run of today's size reads its multi-hop miss the wrong way 11.9% of the time and its ambiguous-entity miss 33.3%, and its date-math figure sits exactly on the gate. To clear both gates it would need both misses and the date-math reading to land on the passing side together, and its unmeasured per-task latency to stay within 600 ms. The counts are assumed, so the true spread could be wider than these chances suggest.

**What is tangled.** The pass-rate gate and the latency budget pull against each other through the reasoning setting. Both configurations that hold the pass rate everywhere use high reasoning, which puts p95 past the budget. The closest one inside the budget, Brisk-S with low reasoning and a retry (513 ms at p95), misses the pass rate on multi-hop by 1.4 points and on ambiguous entities by 0.6.

**What we know despite it.** Of the two that hold the pass rate, Brisk-S with high reasoning and a retry is not told apart from Atlas-M with the same settings (-0.7 points, interval -1.7 to +0.4), and Atlas-M costs 5.5 times as much per 1,000 calls at this run's routing.

**Our lean.** Ours, not a call: if the latency owner can accept a per-task budget that covers Brisk-S with high reasoning and a retry in its slowest task, it is the configuration to confirm. That budget is at least 870 ms, its p95 over all tasks, and likely more on multi-hop; the exact figure stays unknown until per-task latency is measured. Under today's budget, nothing measured qualifies.

**Watch.** Brisk-S with high reasoning and a retry clears every task on its point estimate alone. Under counts assumed from traffic shares, a run of today's size would read it under the gate 37.4% of the time on multi-hop, 29.9% on ambiguous entities, 16.5% on date math and 0.3% on single lookups.

**What would settle it.** Per-task p95 latency for the two Brisk-S configurations with a retry, and a declaration from the latency owner on whether the per-task budget is firm.

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

## Only two of the eighteen configurations hold the pass rate in every task, both with high reasoning and a retry; multi-hop lookups are where every other configuration falls short.

| Panel | Row | Value |
|---|---|---|
| Single lookup | F · Atlas-M · reasoning high · 1 retry | 99.2% |
| Single lookup | L · Brisk-S · reasoning high · 1 retry | 98.5% |
| Single lookup | E · Atlas-M · reasoning high · no retry | 98.7% |
| Single lookup | D · Atlas-M · reasoning low · 1 retry | 99.2% |
| Single lookup | J · Brisk-S · reasoning low · 1 retry | 98.4% |
| Single lookup | K · Brisk-S · reasoning high · no retry | 97.6% |
| Single lookup | Pass rate gate, at least 97% | 97.0% |
| Multi-hop (2 to 3 calls) | F · Atlas-M · reasoning high · 1 retry | 97.9% |
| Multi-hop (2 to 3 calls) | L · Brisk-S · reasoning high · 1 retry | 97.3% |
| Multi-hop (2 to 3 calls) | E · Atlas-M · reasoning high · no retry | 96.6% |
| Multi-hop (2 to 3 calls) | D · Atlas-M · reasoning low · 1 retry | 96.4% |
| Multi-hop (2 to 3 calls) | J · Brisk-S · reasoning low · 1 retry | 95.6% |
| Multi-hop (2 to 3 calls) | K · Brisk-S · reasoning high · no retry | 95.5% |
| Multi-hop (2 to 3 calls) | Pass rate gate, at least 97% | 97.0% |
| Date and unit math | F · Atlas-M · reasoning high · 1 retry | 98.6% |
| Date and unit math | L · Brisk-S · reasoning high · 1 retry | 97.9% |
| Date and unit math | E · Atlas-M · reasoning high · no retry | 97.7% |
| Date and unit math | D · Atlas-M · reasoning low · 1 retry | 97.8% |
| Date and unit math | J · Brisk-S · reasoning low · 1 retry | 97.0% |
| Date and unit math | K · Brisk-S · reasoning high · no retry | 96.6% |
| Date and unit math | Pass rate gate, at least 97% | 97.0% |
| Ambiguous entity | F · Atlas-M · reasoning high · 1 retry | 98.3% |
| Ambiguous entity | L · Brisk-S · reasoning high · 1 retry | 97.6% |
| Ambiguous entity | E · Atlas-M · reasoning high · no retry | 97.2% |
| Ambiguous entity | D · Atlas-M · reasoning low · 1 retry | 97.2% |
| Ambiguous entity | J · Brisk-S · reasoning low · 1 retry | 96.4% |
| Ambiguous entity | K · Brisk-S · reasoning high · no retry | 96.1% |
| Ambiguous entity | Pass rate gate, at least 97% | 97.0% |

Pass rate per task for the six configurations nearest the gate, one panel per task, with the 97% gate line. Task figures state no count, so no interval is drawn.

## Both run past the latency budget: Brisk-S with high reasoning and a retry reaches 870 ms at p95 over all tasks and Atlas-M 2551 ms, while every configuration inside the budget misses the pass rate somewhere.

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

Latency from p50 to p95 over all tasks for all eighteen configurations, with the 600 ms budget line, which is declared per task.

### For Brisk-S with a retry, high reasoning is what carries multi-hop across the gate, from 95.6% at low reasoning to 97.3%, and it is also what takes p95 over all tasks from 513 ms to 870 ms and costs 2.1 times as much per 1,000 calls at this run's routing. Task figures carry no interval, and over all tasks the same move is not separated from zero (+0.8 points, interval -0.4 to +2.1).

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

Pass rate per task for Brisk-S with a retry at low and at high reasoning, with the 97% gate line.

| Group | Row | From | To |
|---|---|---|---|
| Latency | J · Brisk-S · reasoning low · 1 retry | 209 ms | 513 ms |
| Latency | L · Brisk-S · reasoning high · 1 retry | 364 ms | 870 ms |
|  | Latency budget (p95) gate, at most 600 ms in each task | 600 ms |  |

The same two configurations' latency from p50 to p95 over all tasks, with the 600 ms budget line.

### Brisk-S with high reasoning and a retry is not told apart from Atlas-M with the same settings on pass rate (-0.7 points), and Atlas-M costs 5.5 times as much per 1,000 calls; the snapshot names no provider for either, so that cost belongs to this run's unknown routing.

| Row | Value | 95% unpaired interval |
|---|---|---|
| Overall | -0.7 | -1.7 to +0.4 |
| Single lookup | -0.7 |  |
| Multi-hop (2 to 3 calls) | -0.6 |  |
| Date and unit math | -0.7 |  |
| Ambiguous entity | -0.7 |  |

Each row is Brisk-S · reasoning high · 1 retry less Atlas-M · reasoning high · 1 retry on Pass rate, overall and in each place both hold it.

The interval is independent: each arm's own spread (a pass/fail rate's binomial spread, from the arm's cases), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same.

Brisk-S less Atlas-M, both with high reasoning and a retry, in pass-rate points: overall with its unpaired interval, and per task as point differences.

#### Over all tasks, Atlas-M with low reasoning and a retry (97.9%), Atlas-M with high reasoning and no retry (97.7%) and Brisk-S with low reasoning and a retry (97.1%) read above the gate, yet each misses it on multi-hop (96.4%, 96.6% and 95.6%).

| Panel | Row | Value |
|---|---|---|
| Single lookup | D · Atlas-M · reasoning low · 1 retry | 99.2% |
| Single lookup | E · Atlas-M · reasoning high · no retry | 98.7% |
| Single lookup | J · Brisk-S · reasoning low · 1 retry | 98.4% |
| Single lookup | Pass rate gate, at least 97% | 97.0% |
| Multi-hop (2 to 3 calls) | D · Atlas-M · reasoning low · 1 retry | 96.4% |
| Multi-hop (2 to 3 calls) | E · Atlas-M · reasoning high · no retry | 96.6% |
| Multi-hop (2 to 3 calls) | J · Brisk-S · reasoning low · 1 retry | 95.6% |
| Multi-hop (2 to 3 calls) | Pass rate gate, at least 97% | 97.0% |
| Date and unit math | D · Atlas-M · reasoning low · 1 retry | 97.8% |
| Date and unit math | E · Atlas-M · reasoning high · no retry | 97.7% |
| Date and unit math | J · Brisk-S · reasoning low · 1 retry | 97.0% |
| Date and unit math | Pass rate gate, at least 97% | 97.0% |
| Ambiguous entity | D · Atlas-M · reasoning low · 1 retry | 97.2% |
| Ambiguous entity | E · Atlas-M · reasoning high · no retry | 97.2% |
| Ambiguous entity | J · Brisk-S · reasoning low · 1 retry | 96.4% |
| Ambiguous entity | Pass rate gate, at least 97% | 97.0% |

Pass rate per task for the configurations whose rate over all tasks reads above the gate without holding it in every task, with the 97% gate line.

## What you could change

| What you could change | What it decides | What happens as it moves |
|---|---|---|
| Latency budget (p95) | Whether the high-reasoning configurations, the only ones holding the pass rate, are allowed | Brisk-S with high reasoning and a retry needs a per-task budget of at least 870 ms, its p95 over all tasks, and Atlas-M at least 2551 ms; each likely needs more in its slowest task. Below those, nothing measured passes both. No budget value can be read per task, as declared, until per-task latency is measured. |
| Pass-rate gate | Whether the configurations inside the latency budget qualify | Brisk-S with low reasoning and a retry, the closest inside the budget, has its lowest task at 95.6%; a gate at or below that would pass it on pass rate, its per-task latency still unknown. |

## Next steps

- Measure p95 latency per task for Brisk-S with low reasoning and a retry and with high reasoning and a retry: re-slice the run's per-case timings by task if the run kept them (the snapshot holds none), otherwise rerun both with timing on the same cases. _Whether either meets the budget where it was declared; if the low-reasoning one stays inside it in every task, multi-hop and ambiguous-entity pass rate is its only obstacle._
- The latency owner declares whether the per-task budget is firm, or what per-task p95 high reasoning may take. _Whether Brisk-S with high reasoning and a retry becomes the pick; at this run's routing, no rerun of the same configuration would bring its p95 inside today's budget._
- Run Brisk-S with a retry, high reasoning on multi-hop and ambiguous-entity requests and low reasoning elsewhere, measured per task on pass rate and latency. _Whether any configuration clears both gates; nobody has run this mix, so no figure states its pass rate, latency or cost._
- If the high-reasoning Brisk-S is chosen, rerun its multi-hop set at 22,911 multi-hop cases, the size that settles its clearance four times in five under counts assumed from traffic shares. _Whether its clearance on multi-hop, 0.3 points, is real._

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

- **Monthly cost at the declared volume** The declared volume does not say what it counts, so a monthly figure could be off by the calls each unit holds; cost is given per 1,000 calls.
- **Nano-XS per-task figures** Every Nano-XS configuration misses the gate in every task; the best reads 94.8% over all tasks. They stay in the full tables, and none is among the configurations nearest the gate.
- **What the retry buys on its own** It lifts pass rate at every reasoning level, with intervals that barely clear zero on a descriptive read; both configurations near the gate already use it, so it does not choose between them.
- **Model and retry interactions across the grid (the retry helps Nano-XS more than Atlas-M)** Nano-XS is far from the gate in every configuration; the interaction does not change which configurations come close.
- **Calls per task, extra calls and outcome** Extra calls and outcome hold no figures; calls per task describes the planner's work, not either gate.
- **Failure examples** The snapshot keeps none.

### Method

- **Cases** 1,200 planner tasks per configuration, none lost, across four tasks: single lookup (40% of traffic), multi-hop (25%), date and unit math (20%) and ambiguous entity (15%). No repeats are stated.
- **Pass** The snapshot does not state what makes a planner task pass; pass rate is the share of tasks it counts as passed.
- **Gates** Declared: a pass rate of at least 97% in each task, and p95 latency of at most 600 ms in each task. Both are read as gates on the point estimate; a configuration with no figure at a place a gate reads is undecided there, not failed. A pass rate over all tasks is shown for reference only and is never the gate. _Caveats: Per-task latency unmeasured._
- **Intervals** Overall pass-rate intervals are computed from each configuration's cases (binomial, at the level the tables print); differences between configurations combine the two unpaired and are descriptive. Task pass rates state no count, so they carry no interval; run sizes for them assume each task holds its traffic share of the cases. _Caveats: Task pass rates state no count._
- **Latency** p50 and p95 over all tasks, with no interval; no per-task latency is held in the snapshot, and whether the run kept per-case timings is not stated.
- **Cost** Cost per 1,000 calls as measured in this run. The snapshot names each configuration's model but no provider, so the routing behind each cost, and behind each latency, is unknown. The declared volume, 4,000,000 a month, does not say what it counts.
- **Comparison** Eighteen configurations crossing three models, three reasoning levels and a retry; no control, so every comparison is between two configurations' own figures. _Caveats: No arm is the control._
- **Source** Synthetic example data from docs/eval-report-kit/references/r2-config-sweep.json

### Review findings left open

- **Should fix** · Claim 3: The claim rests its caution on the overall high-versus-low difference for Brisk-S with a retry, +0.8 points (interval -0.4 to +2.1, key compare.brisk-s.high.r1.brisk-s.low.r1.pass_rate.overall.*). It also cites a 2.1x cost ratio ($0.633 against $0.302). Neither figure is in either block, which show only the per-task points and latency. Add an overall row with its interval to the pass-rate block, or a small diverging block as in brisk-vs-atlas, so a reader can see the qualifier the claim leans on.

_Written by claude-opus-5-5, served as claude-opus-5-5, for $4.22, with analyst skill db31bf62aa68, on 2026-10-31._
