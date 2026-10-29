Choice · candidates compared · Concierge chat assistant · 4 arms · 5 tasks · Run 2026-10-21 · Synthetic example data

# Atlas-L scores 2.96 points higher than Atlas-M on overall quality at 2.4 times its cost; Brisk-S and Open-70B cost less but lapse on policy behaviours.

Atlas-M is not told apart from Atlas-L on any behaviour judged pass or fail, while the cheaper two trail it by up to 17 points on declining speculation and confirming account changes. Nothing declared what a point of quality is worth; above $22.05 per thousand conversations, Atlas-L is the better buy.

**Confidence: High, if the caveats below aren't material.**

- **Nothing declares what is good enough.** Only if the owner would accept the cheaper models' policy lapses, or values a point of overall quality above $11.09 per thousand conversations; either moves the pick off Atlas-M.
- **No arm is the control.** Only if pairing the arms on their shared cases would undo a separation; every gap here is read as if the arms ran different cases, which reads wider than paired, so pairing would narrow intervals rather than reverse a direction.
- **The pick is the best of several candidates.** Only if a fresh run read Atlas-M's policy rates below Atlas-L's; a lead picked from several candidates tends to shrink on re-run, which would weaken Atlas-L's case for its extra cost, not strengthen it.

**The trade.** Atlas-L scores 2.96 points higher than Atlas-M on overall quality for $44 more per thousand conversations. Brisk-S and Open-70B cost less still, but decline speculation, confirm account changes and avoid unsupported claims less often than either Atlas model.

**Where the answer flips.** A point of overall quality worth more than $22.05 per thousand conversations favours Atlas-L; worth less than $11.09, Atlas-M. Between the two, this run does not settle it.

**Our lean.** Ours, not a declared rule: Atlas-M, while a point of overall quality is worth less than $11.09 per thousand conversations and the cheaper models' policy lapses are not acceptable. It holds every policy behaviour at Atlas-L's level for $44 less per thousand conversations. The concierge owner makes the call.

**Watch.** Atlas-M's widest shortfall on overall quality is troubleshooting, 5 points behind Atlas-L, also the costliest task to serve. The pass-or-fail judges are unnamed, and the cheaper models' lapses rest on them.

**What would settle it.** The owner declares a price per point of overall quality and floors on declining speculation, confirming account changes and unsupported claims; then the frontier below names one option.

| Option | Role | Configuration | Cost per 1,000 conversations |
|---|---|---|---|
| A · Atlas-L | Candidate | Atlas-L | $75.05 |
| B · Atlas-M | Candidate | Atlas-M | $31.37 |
| C · Brisk-S | Candidate | Brisk-S | $6.01 |
| D · Open-70B | Candidate | Open-70B | $11.51 |

## A point of overall quality costs $14.76 per thousand conversations on the step from Atlas-M to Atlas-L, against $4.94 on the step from Brisk-S to Atlas-M; Open-70B is the best buy at no price.

| Step along the frontier | Price per point | 95% unpaired interval |
|---|---|---|
| Brisk-S → Atlas-M | $4.94 | $4.15 to $6.11 |
| Atlas-M → Atlas-L | $14.76 | $11.09 to $22.05 |

Each step along the price frontier on overall quality, pooled over tasks by traffic share, with the price per point read off the gain's unpaired interval.

### Brisk-S and Open-70B trail both Atlas models on declining speculation, confirming before account changes and avoiding unsupported claims, each with an interval clear of Atlas-M's; the widest lapses reach 17 points.

_Caveats: A judge is not described._

| Panel | Row | Value | 95% uncorrected interval |
|---|---|---|---|
| Troubleshooting (multi-turn) · Confirms before account changes | A · Atlas-L | 98% | 96.0% to 100.0% |
| Troubleshooting (multi-turn) · Confirms before account changes | B · Atlas-M | 98% | 96.0% to 100.0% |
| Troubleshooting (multi-turn) · Confirms before account changes | C · Brisk-S | 88% | 83.3% to 92.7% |
| Troubleshooting (multi-turn) · Confirms before account changes | D · Open-70B | 81% | 75.3% to 86.7% |
| Research-needed questions · No unsupported claims | A · Atlas-L | 95% | 92.2% to 97.8% |
| Research-needed questions · No unsupported claims | B · Atlas-M | 94% | 91.0% to 97.0% |
| Research-needed questions · No unsupported claims | C · Brisk-S | 84% | 79.4% to 88.6% |
| Research-needed questions · No unsupported claims | D · Open-70B | 86% | 81.6% to 90.4% |
| Speculation bait · Declines to speculate | A · Atlas-L | 99% | 97.4% to 100.0% |
| Speculation bait · Declines to speculate | B · Atlas-M | 98% | 95.8% to 100.0% |
| Speculation bait · Declines to speculate | C · Brisk-S | 81% | 74.7% to 87.3% |
| Speculation bait · Declines to speculate | D · Open-70B | 89% | 84.0% to 94.0% |

Each model's pass rate on the policy behaviours, in the task that tests each, with intervals; the judge behind these rates is unnamed.

### Atlas-L scores higher than Atlas-M on overall quality in every task but speculation bait, most in troubleshooting at 5 points.

| Row | Value | 95% unpaired interval |
|---|---|---|
| Billing & account Q&A | +2 | +0.4444 to +3.556 |
| Troubleshooting (multi-turn) | +5 | +2.737 to +7.263 |
| Persona small talk | +3 | +0.8787 to +5.121 |
| Research-needed questions | +3 | +1.02 to +4.980 |
| Speculation bait | +1 | -1.546 to +3.546 |

Each row is Atlas-L less Atlas-M on Overall quality, overall and in each place both hold it.

The interval is independent: each arm's own spread (its stated half-width), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same.

Atlas-L less Atlas-M on overall quality in each task, with unpaired intervals.

### On every behaviour judged pass or fail, Atlas-M is not told apart from Atlas-L: each pair of intervals overlaps.

_Caveats: A judge is not described._

| Panel | Row | Value | 95% uncorrected interval |
|---|---|---|---|
| Troubleshooting (multi-turn) · Resolves the issue | A · Atlas-L | 91% | 86.8% to 95.2% |
| Troubleshooting (multi-turn) · Resolves the issue | B · Atlas-M | 90% | 85.6% to 94.4% |
| Troubleshooting (multi-turn) · Gives clear steps | A · Atlas-L | 96% | 93.1% to 98.9% |
| Troubleshooting (multi-turn) · Gives clear steps | B · Atlas-M | 95% | 91.8% to 98.2% |
| Troubleshooting (multi-turn) · Confirms before account changes | A · Atlas-L | 98% | 96.0% to 100.0% |
| Troubleshooting (multi-turn) · Confirms before account changes | B · Atlas-M | 98% | 96.0% to 100.0% |
| Troubleshooting (multi-turn) · Escalates when stuck | A · Atlas-L | 94% | 90.5% to 97.5% |
| Troubleshooting (multi-turn) · Escalates when stuck | B · Atlas-M | 95% | 91.8% to 98.2% |
| Persona small talk · Stays in character | A · Atlas-L | 95% | 92.1% to 97.9% |
| Persona small talk · Stays in character | B · Atlas-M | 95% | 92.1% to 97.9% |
| Research-needed questions · Calls research tool only when required | A · Atlas-L | 97% | 94.8% to 99.2% |
| Research-needed questions · Calls research tool only when required | B · Atlas-M | 95% | 92.2% to 97.8% |
| Research-needed questions · Cites the retrieved source | A · Atlas-L | 96% | 93.5% to 98.5% |
| Research-needed questions · Cites the retrieved source | B · Atlas-M | 94% | 91.0% to 97.0% |
| Research-needed questions · No unsupported claims | A · Atlas-L | 95% | 92.2% to 97.8% |
| Research-needed questions · No unsupported claims | B · Atlas-M | 94% | 91.0% to 97.0% |
| Speculation bait · Declines to speculate | A · Atlas-L | 99% | 97.4% to 100.0% |
| Speculation bait · Declines to speculate | B · Atlas-M | 98% | 95.8% to 100.0% |

Atlas-L and Atlas-M on each pass-or-fail behaviour, in the task that tests it, with intervals.

#### Every model meets the 9 s latency bar in every task; Atlas-L is the slowest, its tail reaching 4.86 s on research questions.

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

Latency per model and task, each bar from the median to the tail percentile.

## What you could change

| What you could change | What it decides | What happens as it moves |
|---|---|---|
| Price of a point of overall quality, per thousand conversations | Which measured option is the best buy | Below $4.15, Brisk-S; from $6.11 to $11.09, Atlas-M; above $22.05, Atlas-L. Inside each step's own band this run does not settle the pick, and Open-70B is the pick at no price. |
| Whether the policy behaviours are guardrails | Whether Brisk-S and Open-70B stay in the trade | With floors near the Atlas models' rates on declining speculation, confirming account changes and avoiding unsupported claims, both cheaper models drop out and the choice narrows to Atlas-M or Atlas-L. |

## Next steps

- The concierge owner declares a price per point of overall quality, per thousand conversations, and floors on declining speculation, confirming account changes and unsupported claims. _Turns the tradeoff into a decision: the frontier then names one option._
- Run a routed mix on the same cases: Atlas-L on troubleshooting, where its lead over Atlas-M is largest, and Atlas-M elsewhere. _What such a mix keeps of Atlas-L's quality and Atlas-M's saving, so it can be priced beside the measured options._
- Name the judge behind the pass-or-fail behaviours and calibrate it against human ratings, as the overall-quality judge was. _Whether the cheaper models' policy lapses, and Atlas-M's parity with Atlas-L, hold under a trusted judge._
- Declare the monthly volume of conversations. _Turns each price per point into a monthly figure the owner can weigh against a budget._

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

- **Open-70B's persona small talk, where it reads highest on staying in character** Not told apart from either Atlas model there, and it lapses on the policy behaviours elsewhere; it changes no option a decider would weigh.
- **The cheaper models' rates on resolving troubleshooting issues, clear steps, escalation, tool calls and citations** They trail the Atlas models on these too; the policy behaviours already carry the case against them, and the full figures sit in the appendix.
- **Cost per task** The cost order is the same in every task; the pooled price frontier carries it.
- **Brisk-S against Open-70B, task by task** Mixed by task, and Open-70B is never the best buy at any price, so the comparison does not move the trade.
- **Failure example cards** The snapshot keeps no failed trials to show.

### Method

- **Cases** 1,190 conversations per model across the tasks, the same count for every model; no repeats are stated. No trial was lost (0 for Atlas-L, and likewise for the others).
- **Overall quality** An ordinal judged score per task, from a judge calibrated on 200 human ratings with 86% agreement. Overall figures pool the tasks, weighted by each task's traffic share as the input states it.
- **Pass-or-fail behaviours** Each judged pass or fail in the one task that tests it. Nothing states whose judgement these are or how well it agrees with people. _Caveats: A judge is not described._
- **Intervals** Intervals at the level each table prints, per task and uncorrected. No model is a control and no paired contrast is stated, so every gap between two models combines their own intervals as if they ran different cases, which reads wider than a paired interval.
- **Cost** Cost per thousand conversations as stated per model, read as exact: no interval and no count of trials reporting it. No monthly volume is declared, so no monthly figure is given.
- **Latency** Median and tail percentile per task, against a 9 s bar every model meets.
- **Declarations** No gate, floor, margin or price per point was declared; the policy behaviours are read here as what a concierge must not get wrong, which is our reading, not a rule.
- **Source** Synthetic example data from docs/eval-report-kit/references/r1-model-selection.json

### Review findings left open

- **Should fix** · Claim 1: The sentence says Open-70B is the best buy at no price, but the block draws only the Brisk-S → Atlas-M and Atlas-M → Atlas-L steps. Add Open-70B's two per-point prices: $6.96 from Brisk-S (price.brisk-s.open-70b.overall.per_point) and $4.58 on to Atlas-M (price.open-70b.atlas-m.overall.per_point). Those numbers show why it is passed over at every price. If you'd rather not add them, drop the Open-70B clause from the claim.
- **Should fix** · Claim 3: The caption says each row is Atlas-L less Atlas-M 'overall and in each place', but the block has only the five task rows. Add the pooled row (+2.96, interval +1.98 to +3.94, from price.atlas-m.atlas-l.overall.gain_interval_unpaired) so the headline's gap sits next to its task breakdown. Otherwise, take 'overall' out of the caption.
- **Should fix** · Claim 5: The claim and Method both measure latency against 'the 9 s latency bar', but the snapshot declares no criterion of any kind, and Method says nothing was declared. A reader will take 9 s for a declared requirement. Say where 9 s comes from and label it as such (for example 'a 9 s line we chose, not a declared bar'), or state latency descriptively. Atlas-L's p95 tops out at 4.86 s on research questions, against 2.84 s for Atlas-M.
- **Should fix** · The page: Fix the first confidence bullet: 'values a point of overall quality above $11.09 … moves the pick off Atlas-M' contradicts the decision block, which says the run doesn't settle the pick between $11.09 and $22.05. Write it as 'above $11.09 the pick is unsettled, and above $22.05 it moves to Atlas-L'. Separately, the confidence line says 'High' while the exclusion of the cheaper models rests entirely on pass-or-fail rates from an undescribed judge, and that caveat is missing from the confidence bullets. Add it there, or lower the stated confidence.
- **Should fix** · The page: In the lean, 'It holds every policy behaviour at Atlas-L's level' claims more than atlas-parity shows, which is only that the two can't be told apart. Each difference interval still allows a gap of about 3 points (on confirming account changes, -2.8 to +2.8). Reword to 'is not told apart from Atlas-L on any policy behaviour'.
- **Should fix** · The page: Widen the routed-mix next step. In persona small talk, Open-70B can't be told apart from Atlas-M on overall quality (90 against 88, difference -0.1 to +4.1). It has the highest staying-in-character rate (97%), costs $6.40 against $17.00 per thousand there, and that task tests no policy behaviour. Add Open-70B on persona small talk as a route to price beside Atlas-L on troubleshooting. Also revise the omitted-item reason that says it 'changes no option a decider would weigh'.

_Written by claude-opus-5-5, served as claude-opus-5-5, for $1.72, with analyst skill 1677ccb31cd6, on 2026-10-29._
