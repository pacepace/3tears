Choice · candidates compared · Concierge chat assistant · 4 arms · 5 tasks · Run 2026-10-21 · Synthetic example data

# Quality costs: Atlas-M trails Atlas-L by 2.96 points at $44 less per 1,000 conversations; Brisk-S is cheapest but slips on safety checks.

On the calibrated overall-quality score, each step up the cost ladder buys a separated gain: a point costs $4.94 per 1,000 conversations from Brisk-S to Atlas-M and $14.76 per 1,000 conversations from Atlas-M to Atlas-L. A declared price per point would pick the option, except inside the bands where the pick flips.

**Nothing was declared before the run: no control, gate, floor or price per point. The choice among these options is the assistant owner's to make.**

**Confidence: High, if the caveats below aren't material.**

- **Cost routing unknown.** The high covers the quality order and the separated quality gains, which do not depend on price. The dollar bands hold only at this run's provider routing, which names no provider: they stay open until production pricing is confirmed, and matter if any model is served at a different price there.
- **Nothing declares what is good enough.** Only if the owner's price per point lands inside one of the flip bands, where this run does not say which option is the better buy.
- **No arm is the control.** Only if an unpaired interval hides a reversal. Each model ran the same task types at the same counts, so paired intervals, were they computed, would likely read narrower and keep every gap called separated here separated; they could also settle where Open-70B sits against Brisk-S.
- **The pick is the best of several candidates.** Only if Atlas-L's lead is mostly chance; a re-run would likely read it lower, which makes the step from Atlas-M to Atlas-L dearer per point and favours the cheaper options.
- **A judge is not described.** Only if the unnamed judge of the task checks is harsher on Brisk-S's or Open-70B's style; the price ladder itself rests on the overall-quality judge, calibrated on 200 human ratings at 86% agreement.

**The trade.** Pooled over task types, Atlas-L reads 92.79 on overall quality at $75.05 per 1,000 conversations; Atlas-M reads 89.83 at $31.37 per 1,000 conversations; Brisk-S reads 84.7 at $6.01 per 1,000 conversations. Open-70B, at $11.51 per 1,000 conversations, is off the frontier on the point estimates, but its gain over Brisk-S (-0.1893 to +1.769 points) does not rule it out at low prices.

**Where the answer flips.** Brisk-S is the best buy if a point of overall quality is worth less than somewhere from $4.15 per 1,000 conversations to $6.11 per 1,000 conversations; Atlas-L if it is worth more than somewhere from $11.09 per 1,000 conversations to $22.05 per 1,000 conversations; Atlas-M in between. In the low band, within the intervals, Open-70B could also be the best buy. Inside either band this run does not settle the pick.

**Our lean.** Ours, not a recommendation: Atlas-M, for any price per point between the two bands. It is not told apart from Atlas-L on the task checks where Brisk-S and Open-70B slip: declining speculation, confirming before account changes, resolving troubleshooting.

**What would settle it.** The assistant's owner declares a price per point of overall quality, or floors on the safety checks; either turns this trade into a decision, at the routing confirmed for production.

**Watch.** Atlas-M's widest gap to Atlas-L is troubleshooting, 5 points (+2.737 to +7.263). Every dollar figure is this run's provider routing, which no arm names.

| Option | Role | Configuration | Cost per 1,000 conversations |
|---|---|---|---|
| A · Atlas-L | Candidate | Atlas-L | $75.05 |
| B · Atlas-M | Candidate | Atlas-M | $31.37 |
| C · Brisk-S | Candidate | Brisk-S | $6.01 |
| D · Open-70B | Candidate | Open-70B | $11.51 |

## Moving from Brisk-S to Atlas-M buys 5.13 points of overall quality at $4.94 per 1,000 conversations a point; moving on to Atlas-L buys 2.96 more at $14.76 per 1,000 conversations a point.

_Caveats: No arm is the control._

| Step along the frontier | Price per point | 95% unpaired interval |
|---|---|---|
| Brisk-S → Atlas-M | $4.94 | $4.15 to $6.11 |
| Atlas-M → Atlas-L | $14.76 | $11.09 to $22.05 |

Price per point of pooled overall quality at each step of the frontier, with unpaired intervals.

### Open-70B sits off the frontier only on its point estimates: a point bought from Brisk-S to Open-70B can cost as little as $3.11 per 1,000 conversations, and one bought on to Atlas-M as much as $5.91 per 1,000 conversations, so within the intervals it could be the best buy in the low band where Brisk-S and Atlas-M swap. By task type it reads above Brisk-S in troubleshooting, persona small talk and speculation bait, and below it in billing.

_Caveats: No arm is the control._

| Row | Value | 95% unpaired interval |
|---|---|---|
| Billing & account Q&A | -3 | -4.556 to -1.444 |
| Troubleshooting (multi-turn) | +4 | +1.737 to +6.263 |
| Persona small talk | +6 | +3.879 to +8.121 |
| Research-needed questions | -1 | -2.98 to +0.9799 |
| Speculation bait | +3 | +0.4544 to +5.546 |

Each row is Open-70B less Brisk-S on Overall quality, overall and in each place both hold it.

The interval is independent: each arm's own spread (its stated half-width), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same.

Open-70B less Brisk-S on overall quality in each task type, with unpaired intervals; task types only, no overall row is drawn.

### Atlas-L costs the most in every task type and Brisk-S the least; troubleshooting is the dearest task for every model, $140 per 1,000 conversations on Atlas-L.

_Caveats: Cost routing unknown._

| Group | Row | Value |
|---|---|---|
| Billing & account Q&A | A · Atlas-L | $58.00 |
| Billing & account Q&A | B · Atlas-M | $24.00 |
| Billing & account Q&A | C · Brisk-S | $4.60 |
| Billing & account Q&A | D · Open-70B | $8.90 |
| Troubleshooting (multi-turn) | A · Atlas-L | $140.00 |
| Troubleshooting (multi-turn) | B · Atlas-M | $59.00 |
| Troubleshooting (multi-turn) | C · Brisk-S | $11.00 |
| Troubleshooting (multi-turn) | D · Open-70B | $21.00 |
| Persona small talk | A · Atlas-L | $41.00 |
| Persona small talk | B · Atlas-M | $17.00 |
| Persona small talk | C · Brisk-S | $3.30 |
| Persona small talk | D · Open-70B | $6.40 |
| Research-needed questions | A · Atlas-L | $95.00 |
| Research-needed questions | B · Atlas-M | $40.00 |
| Research-needed questions | C · Brisk-S | $8.00 |
| Research-needed questions | D · Open-70B | $15.00 |
| Speculation bait | A · Atlas-L | $45.00 |
| Speculation bait | B · Atlas-M | $19.00 |
| Speculation bait | C · Brisk-S | $3.60 |
| Speculation bait | D · Open-70B | $7.00 |

The bars are drawn on a log scale: their values span more than ten times, so each step is a multiple.

Cost per 1,000 conversations by task type, on a log scale; no provider is named for any model.

### Atlas-L reads above Atlas-M in every task type but speculation bait, where the two are not told apart; the widest gap is troubleshooting at 5 points.

_Caveats: The pick is the best of several candidates._

| Row | Value | 95% unpaired interval |
|---|---|---|
| Billing & account Q&A | +2 | +0.4444 to +3.556 |
| Troubleshooting (multi-turn) | +5 | +2.737 to +7.263 |
| Persona small talk | +3 | +0.8787 to +5.121 |
| Research-needed questions | +3 | +1.02 to +4.980 |
| Speculation bait | +1 | -1.546 to +3.546 |

Each row is Atlas-L less Atlas-M on Overall quality, overall and in each place both hold it.

The interval is independent: each arm's own spread (its stated half-width), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same.

Atlas-L less Atlas-M on overall quality in each task type, with unpaired intervals; task types only, no overall row is drawn.

### Brisk-S's saving comes with drops on the checks that protect users: Atlas-M declines speculation bait more often by 17 points (+10.3 to +23.7). Both cheaper models confirm before account changes less often than the Atlas pair, Open-70B behind Atlas-M by 17 points; Brisk-S and Open-70B are not told apart there (-0.4 to +14.4).

_Caveats: A judge is not described._

| Row | Value | 95% unpaired interval |
|---|---|---|
| Speculation bait | +17 | +10.3 to +23.7 |

Each row is Atlas-M less Brisk-S on Declines to speculate, overall and in each place both hold it.

The interval is independent: each arm's own spread (its stated half-width), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same.

Atlas-M less Brisk-S on declining speculation bait, with its unpaired interval; no overall row is drawn.

| Slice | Arm | Resolves the issue | 95% uncorrected interval | Confirms before account changes | 95% uncorrected interval | Calls research tool only when required | 95% uncorrected interval | No unsupported claims | 95% uncorrected interval | Declines to speculate | 95% uncorrected interval |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Troubleshooting (multi-turn) | A · Atlas-L | 91% | 86.8% to 95.2% | 98% | 96.0% to 100.0% | — | — | — | — | — | — |
| Troubleshooting (multi-turn) | B · Atlas-M | 90% | 85.6% to 94.4% | 98% | 96.0% to 100.0% | — | — | — | — | — | — |
| Troubleshooting (multi-turn) | C · Brisk-S | 74% | 67.6% to 80.4% | 88% | 83.3% to 92.7% | — | — | — | — | — | — |
| Troubleshooting (multi-turn) | D · Open-70B | 82% | 76.4% to 87.6% | 81% | 75.3% to 86.7% | — | — | — | — | — | — |
| Research-needed questions | A · Atlas-L | — | — | — | — | 97% | 94.8% to 99.2% | 95% | 92.2% to 97.8% | — | — |
| Research-needed questions | B · Atlas-M | — | — | — | — | 95% | 92.2% to 97.8% | 94% | 91.0% to 97.0% | — | — |
| Research-needed questions | C · Brisk-S | — | — | — | — | 83% | 78.2% to 87.8% | 84% | 79.4% to 88.6% | — | — |
| Research-needed questions | D · Open-70B | — | — | — | — | 88% | 83.9% to 92.1% | 86% | 81.6% to 90.4% | — | — |
| Speculation bait | A · Atlas-L | — | — | — | — | — | — | — | — | 99% | 97.4% to 100.6% |
| Speculation bait | B · Atlas-M | — | — | — | — | — | — | — | — | 98% | 95.8% to 100.2% |
| Speculation bait | C · Brisk-S | — | — | — | — | — | — | — | — | 81% | 74.7% to 87.3% |
| Speculation bait | D · Open-70B | — | — | — | — | — | — | — | — | 89% | 84.0% to 94.0% |

Pass rates on the task checks, each in the task type it applies to, with uncorrected intervals; these are normal approximations, so near a perfect score an upper end can print past the ceiling.

#### In persona small talk Open-70B is not told apart from either Atlas model on overall quality, Atlas-L ahead by 1 (-1.121 to +3.121), and it reads highest on staying in character, 97%, at $6.40 per 1,000 conversations against $41.00 per 1,000 conversations for Atlas-L.

| Row | Value | 95% unpaired interval |
|---|---|---|
| Billing & account Q&A | -6 | -7.556 to -4.444 |
| Troubleshooting (multi-turn) | -5 | -7.263 to -2.737 |
| Persona small talk | +2 | -0.1213 to +4.121 |
| Research-needed questions | -9 | -10.98 to -7.02 |
| Speculation bait | -3 | -5.546 to -0.4544 |

Each row is Open-70B less Atlas-M on Overall quality, overall and in each place both hold it.

The interval is independent: each arm's own spread (its stated half-width), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same.

Open-70B less Atlas-M on overall quality in each task type, with unpaired intervals; task types only, no overall row is drawn.

#### Latency follows cost: Brisk-S has the shortest p95 in every task type, and Atlas-L's p95 reaches 4.86 s on research questions. No arm lost a case, so no short tail was bought by declining work.

| Group | Row | From | To |
|---|---|---|---|
| Billing & account Q&A | A · Atlas-L | 1.81 s | 3.60 s |
| Billing & account Q&A | B · Atlas-M | 1.03 s | 2.10 s |
| Billing & account Q&A | C · Brisk-S | 0.47 s | 0.90 s |
| Billing & account Q&A | D · Open-70B | 0.78 s | 1.60 s |
| Troubleshooting (multi-turn) | A · Atlas-L | 2.42 s | 4.68 s |
| Troubleshooting (multi-turn) | B · Atlas-M | 1.38 s | 2.73 s |
| Troubleshooting (multi-turn) | C · Brisk-S | 0.57 s | 1.17 s |
| Troubleshooting (multi-turn) | D · Open-70B | 1.07 s | 2.08 s |
| Persona small talk | A · Atlas-L | 1.39 s | 2.88 s |
| Persona small talk | B · Atlas-M | 0.86 s | 1.68 s |
| Persona small talk | C · Brisk-S | 0.35 s | 0.72 s |
| Persona small talk | D · Open-70B | 0.62 s | 1.28 s |
| Research-needed questions | A · Atlas-L | 2.48 s | 4.86 s |
| Research-needed questions | B · Atlas-M | 1.53 s | 2.84 s |
| Research-needed questions | C · Brisk-S | 0.60 s | 1.22 s |
| Research-needed questions | D · Open-70B | 1.07 s | 2.16 s |
| Speculation bait | A · Atlas-L | 1.70 s | 3.24 s |
| Speculation bait | B · Atlas-M | 1.03 s | 1.89 s |
| Speculation bait | C · Brisk-S | 0.42 s | 0.81 s |
| Speculation bait | D · Open-70B | 0.73 s | 1.44 s |

Latency p50 to p95 by task type; no latency budget was declared.

## What you could change

| What you could change | What it decides | What happens as it moves |
|---|---|---|
| Price of one point of overall quality, per 1,000 conversations | Which measured option is the best buy | Brisk-S below a band from $4.15 per 1,000 conversations to $6.11 per 1,000 conversations; Atlas-M between the bands; Atlas-L above a band from $11.09 per 1,000 conversations to $22.05 per 1,000 conversations. Open-70B is never best on the point estimates, but within the intervals it could be anywhere from $3.11 per 1,000 conversations to $5.91 per 1,000 conversations. Every dollar holds at this run's routing only. |

## Next steps

- The assistant's owner declares a price per point of overall quality, or floors on declining speculation and confirming before account changes. _Which option is the best buy; turns the trade into a decision._
- Confirm the provider and price each model would be served at in production. _Whether the dollar bands, which hold only at this run's routing, can be acted on._
- Re-read overall quality as paired per-case differences between the models, since they ran the same task types. _Where Open-70B sits against Brisk-S, and so whether it belongs on the frontier in the low band._
- Declare the monthly volume of conversations. _What each step costs per month, so the price per point reads against a budget._
- Run a routed option, Open-70B for persona small talk and Atlas-M elsewhere, on the same cases. _Whether routing lowers Atlas-M's cost without losing quality; nobody ran this mix, so no figure states it yet._
- Name the judge of the task checks and calibrate it against human ratings. _Whether Brisk-S's and Open-70B's drops on the safety checks are real or the judge's._

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

- **Gives clear steps, escalates when stuck, cites the retrieved source and stays in character, beyond the persona claim** Atlas-M and Atlas-L are not told apart on any of them, and Brisk-S or Open-70B reads lowest on each; none changes the trade. Full figures in the appendix.
- **Open-70B's lead over Brisk-S on resolving troubleshooting and declining speculation** It reads higher on both, but neither gap is separated on unpaired intervals; the paired re-read in Next would settle it alongside the overall-quality question.
- **A dominance call naming Brisk-S** It reads cost alone, since quality is held only per task type; the cheapest option is not shown good enough by anything declared.
- **Per-task price ladders beyond persona small talk** Only persona small talk was priced on its own, because it is where Open-70B moves onto the frontier; the pooled ladder carries the trade.
- **Failure examples** The snapshot keeps none.

### Method

- **Cases** 1,190 conversations per model, spread over the task types; no repeats are stated and no arm lost or failed a case.
- **Overall quality** A judged score per conversation, calibrated on 200 human ratings at 86% agreement. Pooled figures weight each task type by its traffic share as the input states it.
- **Task checks** Pass or fail per conversation, each read only in the task type it applies to. The input does not say who judged them. _Caveats: A judge is not described._
- **Intervals** Read at 95% per task type, as normal approximations, so a pass rate near a perfect score can show an upper end past the ceiling. Differences between models combine each model's own spread as if independent, uncorrected; with no control nothing is paired, so they read wider than paired intervals would. _Caveats: No arm is the control._
- **Cost** Cost per 1,000 conversations, pooled by traffic share. No provider is named for any model and no monthly volume is declared, so no monthly figure is stated.
- **Latency** p50 and p95 per task type; no budget was declared.
- **What was not compared** No control, gate, floor or exchange rate was declared, so no option is ruled out and nothing ranks the axes.
- **Source** Concierge model selection, run 2026-10-21; Synthetic example data from docs/eval-report-kit/references/r1-model-selection.json.

### Review findings left open

- **Should fix** · Claim 2: The claim's lead sentence rests on pooled figures its block does not draw. Those are Open-70B's pooled gain over Brisk-S, +0.79 (-0.1893 to +1.769), and the price ends $3.11 (Brisk-S to Open-70B) and $5.91 (Open-70B to Atlas-M). The block shows per-task differences only, so if the pooled figures moved, nothing on the block would change. Fix: add the pooled gain as a row, or a small table of the two pooled steps with their per-point price and unpaired interval. Also reword the explore row 'it could be anywhere from $3.11 to $5.91' to say it could be the best buy at a price per point somewhere in that range.
- **Should fix** · Claim 6: The claim's figures are not in its block. Those are Atlas-L ahead of Open-70B by 1 (-1.121 to +3.121), Open-70B's 97% on staying in character, and $6.40 vs $41.00 per 1,000 conversations. The block draws Open-70B less Atlas-M in all five task types, which is mostly about other tasks. Fix: draw persona small talk only, with Open-70B against both Atlas models, plus the in-character rates and the persona costs. Or narrow the sentence to what the block shows.
- **Should fix** · The page: The page says persona is the only task where Open-70B joins the frontier, and gives that as the reason for leaving out per-task price ladders. That is wrong on the point estimates. Open-70B is also on the frontier in troubleshooting (83 vs Brisk-S 79 at $21 vs $11: $2.50 a point, then $7.60 a point on to Atlas-M) and in speculation bait ($1.13 a point, then $4.00). In persona it beats Atlas-M outright (90 vs 88, $6.40 vs $17.00). Fix: correct the reason. Then either add a sentence to claim open-70b that it sits on the per-task frontier in three of five task types, or widen the routed-option next step to say which mix it would test.
- **Should fix** · Claim 1: The claim states the gains bought at each step, 5.13 points (Brisk-S to Atlas-M) and 2.96 (Atlas-M to Atlas-L). The lede calls both separated. The table shows only the price per point. Add a gain column with its unpaired interval (+4.151 to +6.109 and +1.981 to +3.939), so a reader sees the separated gains the High stands on.

_Written by claude-opus-5-5, served as claude-opus-5-5, for $2.87, with analyst skill 2a7662d928b0, on 2026-10-29._
