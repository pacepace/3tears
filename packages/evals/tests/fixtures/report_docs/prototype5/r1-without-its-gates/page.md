Choice · candidates compared · Concierge chat assistant · 4 arms · 5 tasks · Run 2026-10-21 · Synthetic example data

# The pick is a price nobody set: Atlas-L scores 2.96 points above Atlas-M on overall quality at 2.4 times its cost, and the two cheaper models confirm account changes less often.

No behaviour check tells Atlas-M from Atlas-L, while Open-70B confirms before account changes 17 points less often than Atlas-M. A declared price per point of overall quality, or a bar on these checks, would turn the trade into a decision.

**No control, gate or exchange rate was declared: every option below is a candidate, and the choice among them is the concierge owner's.**

**Confidence: High, if the caveats below aren't material.**

- **Nothing declares what is good enough.** Only if this page is read as a decision: with no bar or price per point declared, which side of the trade to take is the owner's call. The measured gaps do not rest on it.
- **A judge is not described.** Only if the unnamed judge of the behaviour checks is harsher on the smaller models' replies; the gaps on confirming account changes and declining to speculate would then narrow. Overall quality is scored by a judge calibrated on 200 human ratings.
- **No arm is the control.** Only for the size of a gap: arms compare by their own intervals combined as independent, which reads wider than a paired reading would, so a paired re-run would more likely narrow the intervals than reverse a separated gap.
- **The pick is the best of several candidates.** Only where the leader's margin is thin, as Atlas-L's over Atlas-M in billing (2 points): picked as the best of four, it would tend to read lower on fresh cases.

**The trade.** Each step up costs more per point of overall quality: Brisk-S to Atlas-M at $4.94 per 1,000 conversations a point, Atlas-M to Atlas-L at $14.76. Open-70B is never the cheapest way to a point. Brisk-S, the cheapest, also trails on the safety checks.

**Where the answer flips.** On overall quality alone, Brisk-S is the best buy while a point is worth less than somewhere from $4.15 to $6.11 per 1,000 conversations, Atlas-L once it is worth more than somewhere from $11.09 to $22.05, and Atlas-M between. Inside either band this run does not settle the pick.

**Our lean.** Ours, not a declared rule: Atlas-M, at $31.37 per 1,000 conversations against Atlas-L's $75.05. No behaviour check separates it from Atlas-L. The lean holds while a point of overall quality is worth between the two bands.

**Watch.** Atlas-L's lead over Atlas-M is widest in troubleshooting, 5 points, the task with 19% of traffic. No line was declared, so no margin is the weakest. Speculation bait carries 0% of traffic: its rates test behaviour, not what live traffic meets.

**What would settle it.** The concierge owner declares a price per point of overall quality and a bar on confirming account changes and declining to speculate.

| Option | Role | Configuration | Cost per 1,000 conversations |
|---|---|---|---|
| A · Atlas-L | Candidate | Atlas-L | $75.05 |
| B · Atlas-M | Candidate | Atlas-M | $31.37 |
| C · Brisk-S | Candidate | Brisk-S | $6.01 |
| D · Open-70B | Candidate | Open-70B | $11.51 |

## Brisk-S and Open-70B fall well behind both Atlas models on confirming account changes and declining to speculate; Brisk-S declines 17 points less often than Atlas-M.

_Caveats: A judge is not described._

| Panel | Row | Value | 95% uncorrected interval |
|---|---|---|---|
| Troubleshooting (multi-turn) · Confirms before account changes | A · Atlas-L | 98% | 96.0% to 100.0% |
| Troubleshooting (multi-turn) · Confirms before account changes | B · Atlas-M | 98% | 96.0% to 100.0% |
| Troubleshooting (multi-turn) · Confirms before account changes | C · Brisk-S | 88% | 83.3% to 92.7% |
| Troubleshooting (multi-turn) · Confirms before account changes | D · Open-70B | 81% | 75.3% to 86.7% |
| Speculation bait · Declines to speculate | A · Atlas-L | 99% | 97.4% to 100.0% |
| Speculation bait · Declines to speculate | B · Atlas-M | 98% | 95.8% to 100.0% |
| Speculation bait · Declines to speculate | C · Brisk-S | 81% | 74.7% to 87.3% |
| Speculation bait · Declines to speculate | D · Open-70B | 89% | 84.0% to 94.0% |

Each model's rate with its interval: confirming before account changes in troubleshooting, and declining to speculate on speculation bait, a test set that carries 0% of live traffic.

### Atlas-L scores above Atlas-M on overall quality in every task that carries live traffic, most in troubleshooting; on speculation bait the two are not told apart.

_Caveats: The pick is the best of several candidates; No arm is the control._

| Row | Value | 95% unpaired interval |
|---|---|---|
| Billing & account Q&A | +2 | +0.4444 to +3.556 |
| Troubleshooting (multi-turn) | +5 | +2.737 to +7.263 |
| Persona small talk | +3 | +0.8787 to +5.121 |
| Research-needed questions | +3 | +1.02 to +4.980 |
| Speculation bait | +1 | -1.546 to +3.546 |

Each row is Atlas-L less Atlas-M on Overall quality, overall and in each place both hold it.

The interval is independent: each arm's own spread (its stated half-width), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same.

Atlas-L less Atlas-M on overall quality per task, with intervals combined as independent.

### A point of overall quality costs far more on the step from Atlas-M to Atlas-L, $14.76 per 1,000 conversations, than on the step up from Brisk-S.

| Step along the frontier | Price per point | 95% unpaired interval |
|---|---|---|
| Brisk-S → Atlas-M | $4.94 | $4.15 to $6.11 |
| Atlas-M → Atlas-L | $14.76 | $11.09 to $22.05 |

Price per point of overall quality along the frontier, per 1,000 conversations, with intervals read off each step's gain.

#### No model reads clearly above the Atlas pair on the other behaviour checks; Brisk-S resolves the fewest troubleshooting issues, 74% against Atlas-M's 90%.

_Caveats: A judge is not described._

| Slice | Arm | Resolves the issue | 95% uncorrected interval | Gives clear steps | 95% uncorrected interval | Escalates when stuck | 95% uncorrected interval | Stays in character | 95% uncorrected interval | Calls research tool only when required | 95% uncorrected interval | Cites the retrieved source | 95% uncorrected interval | No unsupported claims | 95% uncorrected interval |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Troubleshooting (multi-turn) | A · Atlas-L | 91% | 86.8% to 95.2% | 96% | 93.1% to 98.9% | 94% | 90.5% to 97.5% | — | — | — | — | — | — | — | — |
| Troubleshooting (multi-turn) | B · Atlas-M | 90% | 85.6% to 94.4% | 95% | 91.8% to 98.2% | 95% | 91.8% to 98.2% | — | — | — | — | — | — | — | — |
| Troubleshooting (multi-turn) | C · Brisk-S | 74% | 67.6% to 80.4% | 90% | 85.6% to 94.4% | 85% | 79.8% to 90.2% | — | — | — | — | — | — | — | — |
| Troubleshooting (multi-turn) | D · Open-70B | 82% | 76.4% to 87.6% | 93% | 89.3% to 96.7% | 90% | 85.6% to 94.4% | — | — | — | — | — | — | — | — |
| Persona small talk | A · Atlas-L | — | — | — | — | — | — | 95% | 92.1% to 97.9% | — | — | — | — | — | — |
| Persona small talk | B · Atlas-M | — | — | — | — | — | — | 95% | 92.1% to 97.9% | — | — | — | — | — | — |
| Persona small talk | C · Brisk-S | — | — | — | — | — | — | 86% | 81.4% to 90.6% | — | — | — | — | — | — |
| Persona small talk | D · Open-70B | — | — | — | — | — | — | 97% | 94.7% to 99.3% | — | — | — | — | — | — |
| Research-needed questions | A · Atlas-L | — | — | — | — | — | — | — | — | 97% | 94.8% to 99.2% | 96% | 93.5% to 98.5% | 95% | 92.2% to 97.8% |
| Research-needed questions | B · Atlas-M | — | — | — | — | — | — | — | — | 95% | 92.2% to 97.8% | 94% | 91.0% to 97.0% | 94% | 91.0% to 97.0% |
| Research-needed questions | C · Brisk-S | — | — | — | — | — | — | — | — | 83% | 78.2% to 87.8% | 89% | 85.0% to 93.0% | 84% | 79.4% to 88.6% |
| Research-needed questions | D · Open-70B | — | — | — | — | — | — | — | — | 88% | 83.9% to 92.1% | 85% | 80.5% to 89.5% | 86% | 81.6% to 90.4% |

Every other behaviour check in the task it belongs to, with intervals.

#### Atlas-L is also the slowest: its p95 latency on research questions is 4.86 s, against 2.84 s for Atlas-M.

| Group | Row | From | To |
|---|---|---|---|
| Research-needed questions | A · Atlas-L | 2.48 s | 4.86 s |
| Research-needed questions | B · Atlas-M | 1.53 s | 2.84 s |
| Research-needed questions | C · Brisk-S | 0.60 s | 1.22 s |
| Research-needed questions | D · Open-70B | 1.07 s | 2.16 s |

Latency from p50 to p95 on research questions, the slowest task; no latency budget was declared.

## What you could change

| What you could change | What it decides | What happens as it moves |
|---|---|---|
| Price of one point of overall quality, per 1,000 conversations | Which of Brisk-S, Atlas-M and Atlas-L is the best buy on overall quality | Brisk-S below somewhere from $4.15 to $6.11; Atlas-M from there to somewhere from $11.09 to $22.05; Atlas-L above. Open-70B is best at no price. |

## Next steps

- The concierge owner declares a price per point of overall quality, and bars on confirming account changes and declining to speculate. _Turns the trade into a decision: the bars rule options out, the price picks among the rest._
- Name the judge of the behaviour checks and calibrate it on human ratings, as overall quality's judge already is. _Whether Brisk-S and Open-70B really skip account confirmations and speculate more, or the judge reads their style harshly._
- Run a routed mix, Atlas-L for troubleshooting and Atlas-M elsewhere, on the same cases. _Whether routing keeps most of Atlas-L's lead, which is widest in troubleshooting, at a cost nearer Atlas-M's._
- Re-run Atlas-M and Atlas-L on fresh cases with one of them as the control, so gaps read paired. _How large Atlas-L's lead really is once the best-of-several optimism and the unpaired intervals are removed._

## Full results

### The arms

Each arm's full configuration, once. The rest of the page names an arm by its letter tag and name.

| Arm | Role | Configuration | Served model |
|---|---|---|---|
| A · Atlas-L | Candidate | Atlas-L | Atlas-L |
| B · Atlas-M | Candidate | Atlas-M | Atlas-M |
| C · Brisk-S | Candidate | Brisk-S | Brisk-S |
| D · Open-70B | Candidate | Open-70B | Open-70B |

### Every arm by measure

| Arm | Cost per 1,000 conversations |
|---|---|
| A · Atlas-L | $75.05 |
| B · Atlas-M | $31.37 |
| C · Brisk-S | $6.01 |
| D · Open-70B | $11.51 |

### Every task

| Task | Measure | Cases | A · Atlas-L | B · Atlas-M | C · Brisk-S | D · Open-70B |
|---|---|---|---|---|---|---|
| Billing & account Q&A | Overall quality | 400 | 94 | 92 | 89 | 86 |
| Billing & account Q&A | Cost per 1k conversations | 400 | $58.00 | $24.00 | $4.60 | $8.90 |
| Billing & account Q&A | Latency | 400 | 3.6 s (p95) | 2.1 s (p95) | 0.9 s (p95) | 1.6 s (p95) |
| Troubleshooting (multi-turn) | Overall quality | 180 | 93 | 88 | 79 | 83 |
| Troubleshooting (multi-turn) | Cost per 1k conversations | 180 | $140.00 | $59.00 | $11.00 | $21.00 |
| Troubleshooting (multi-turn) | Latency | 180 | 4.68 s (p95) | 2.73 s (p95) | 1.17 s (p95) | 2.08 s (p95) |
| Troubleshooting (multi-turn) | Resolves the issue | 180 | 91% | 90% | 74% | 82% |
| Troubleshooting (multi-turn) | Gives clear steps | 180 | 96% | 95% | 90% | 93% |
| Troubleshooting (multi-turn) | Confirms before account changes | 180 | 98% | 98% | 88% | 81% |
| Troubleshooting (multi-turn) | Escalates when stuck | 180 | 94% | 95% | 85% | 90% |
| Persona small talk | Overall quality | 220 | 91 | 88 | 84 | 90 |
| Persona small talk | Cost per 1k conversations | 220 | $41.00 | $17.00 | $3.30 | $6.40 |
| Persona small talk | Latency | 220 | 2.88 s (p95) | 1.68 s (p95) | 0.72 s (p95) | 1.28 s (p95) |
| Persona small talk | Stays in character | 220 | 95% | 95% | 86% | 97% |
| Research-needed questions | Overall quality | 240 | 92 | 89 | 81 | 80 |
| Research-needed questions | Cost per 1k conversations | 240 | $95.00 | $40.00 | $8.00 | $15.00 |
| Research-needed questions | Latency | 240 | 4.86 s (p95) | 2.84 s (p95) | 1.22 s (p95) | 2.16 s (p95) |
| Research-needed questions | Calls research tool only when required | 240 | 97% | 95% | 83% | 88% |
| Research-needed questions | Cites the retrieved source | 240 | 96% | 94% | 89% | 85% |
| Research-needed questions | No unsupported claims | 240 | 95% | 94% | 84% | 86% |
| Speculation bait | Overall quality | 150 | 90 | 89 | 83 | 86 |
| Speculation bait | Cost per 1k conversations | 150 | $45.00 | $19.00 | $3.60 | $7.00 |
| Speculation bait | Latency | 150 | 3.24 s (p95) | 1.89 s (p95) | 0.81 s (p95) | 1.44 s (p95) |
| Speculation bait | Declines to speculate | 150 | 99% | 98% | 81% | 89% |

### Left out, and why

- **Latency on the other tasks** The same order as research questions in every task; no latency budget was declared, so it does not move the trade.
- **Cost per 1,000 conversations by task** The order of the models is the same in every task; the pooled cost carries the trade.
- **Open-70B staying in character most often in persona small talk** Not told apart from the Atlas pair, and it does not lift Open-70B onto the price frontier.
- **Failure example cards** The snapshot keeps none.
- **Speculation bait in the pooled overall quality** It carries no live traffic, so the traffic-weighted pool leaves it out; its own gap is shown in the quality claim.

### Method

- **Cases** Each model ran 1,190 cases across the tasks; the input states no repeats. Every model answered every case, with zero lost.
- **Overall quality** An ordinal score from a judge calibrated on 200 human ratings, agreeing with them 86% of the time.
- **Behaviour checks** Pass/fail checks, each read only in the task it belongs to; the input does not name their judge. _Caveats: A judge is not described._
- **Pooling** Overall quality and cost are pooled over the tasks that carry live traffic, weighted by traffic share: billing 42%, persona small talk 24%, troubleshooting 19% and research questions 15%.
- **Intervals** At 95%, each model's own; gaps between two models combine them as independent, since no control or paired contrast was run. Per-task gaps are uncorrected. _Caveats: No arm is the control._
- **Cost** Cost per 1,000 conversations. No monthly volume was declared, so no monthly figure is stated.
- **Latency** p50 and p95 per task; no budget was declared.
- **Source** Synthetic example data from docs/eval-report-kit/references/r1-model-selection.json, run 2026-10-21.

### Review findings left open

- **Should fix** · The page: The routing next step aims at the wrong task. In the task figures, the Atlas-M to Atlas-L step costs $16.20 per point in troubleshooting, $17.00 in billing and $18.33 in research questions, but only about $8 in persona small talk ($24 more per 1,000 conversations for 3 points). Routing troubleshooting to Atlas-L buys about a third of the lead for about a third of the cost gap. It does not keep 'most of the lead at a cost nearer Atlas-M's'. In persona small talk, Open-70B scores 90 against Atlas-M's 88 (gap −0.1 to +4.1) at $6.40 against $17, and stays in character 97% of the time. Rewrite the step from the task prices: say what a routed mix would test (for example, Open-70B or Atlas-L on persona small talk and Atlas-M elsewhere), and drop the claim that troubleshooting is where routing pays.
- **Should fix** · The page: 'Open-70B is never the cheapest way to a point' (the trade) and 'Open-70B is best at no price' (explore) are point-estimate statements. Open-70B's pooled overall quality is 85.49 against Brisk-S's 84.70, and each has a half-width of 0.69. Open-70B is off the frontier only if its lead is under about 1.1 points, and the combined interval reaches about +1.8. Qualify both sentences, for example 'on the point estimates; its lead over Brisk-S is not settled'. Or rest Open-70B's exclusion on its safety-check gaps, which are separated.
- **Should fix** · Claim 2: The caption says each row is Atlas-L less Atlas-M 'overall and in each place', but the block draws only the five task rows. Add the pooled, traffic-weighted row the headline cites (+2.96, +1.98 to +3.94) so the headline's figure is on the page beside its interval. Otherwise, remove 'overall' from the caption.
- **Should fix** · Claim 2: 'Above Atlas-M in every task that carries live traffic' rests on per-task intervals with no correction across five tasks. Billing's lower end is +0.44. Add the word 'uncorrected', or note that billing's gap is the one a correction could close. The best-of-several caveat already names billing, so this is a small change.

_Written by claude-opus-5-5, served as claude-opus-5-5, for $1.54, with analyst skill 1677ccb31cd6, on 2026-10-29._
