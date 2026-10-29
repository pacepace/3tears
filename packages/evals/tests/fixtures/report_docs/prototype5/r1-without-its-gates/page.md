Choice · candidates compared · Concierge chat assistant · 4 arms · 5 tasks · Run 2026-10-21 · Synthetic example data

# Atlas-M costs 0.42 times what Atlas-L does at this run's unnamed routing, with no behaviour check separating them; Atlas-L's extra overall quality is the owner's to price.

The cheaper models save more but confirm before account changes less often than Atlas-M, and Brisk-S resolves fewer troubleshooting issues. With no price set for a point of overall quality, Atlas-M is the better buy below $11.09 per point per 1,000 conversations, Atlas-L above $22.05.

**No control, gate or price per point was declared: this run prices the trade, and the concierge owner makes the call.**

**Confidence: High, if the caveats below aren't material.**

- **Nothing declares what is good enough.** It decides the pick itself, which waits on a price per point of overall quality or floors on the behaviour checks; it does not touch the measured gaps the trade is built from.
- **No arm is the control.** Only if scoring the arms paired on their shared cases would separate a behaviour-check gap the unpaired intervals leave open, such as Atlas-L's tool-call lead over Atlas-M (from -1.6 to +5.6). Pairing narrows intervals, so it could add a reason to pay for Atlas-L but would not undo a gap already clear of zero.
- **A judge is not described.** Only if the unnamed judge of the behaviour checks is harsher on the smaller models' style, which would shrink their shortfalls and make Brisk-S a stronger low-price option. The overall-quality judge is calibrated on 200 human ratings at 86% agreement, so the price per point does not rest on the unnamed one.
- **The pick is the best of several candidates.** Only if Atlas-L's lead in overall quality is partly the luck of the best of several: a fresh run would tend to read it lower, which would cheapen the case for Atlas-L, not Atlas-M. Its unpaired interval, from +1.981 to +3.939, sits clear of zero.
- **Cost routing and coverage unknown.** Only if another provider serving one of these models, or trials that reported no cost, would reorder the costs: the snapshot names no provider or pricing for any model and does not say how many trials reported a cost, so every price holds for this run's routing only. Open-70B, an open-weights model, is the most exposed, since its price depends on who hosts it.

**The trade.** Moving from Atlas-M to Atlas-L costs $44 more per 1,000 conversations and buys 2.96 points of overall quality (from +1.981 to +3.939); on the behaviour checks the two are not told apart. Brisk-S costs 0.19 times what Atlas-M does but trails it clearly on troubleshooting.

**Where the answer flips.** A point of overall quality worth less than $11.09 per 1,000 conversations favours Atlas-M, more than $22.05 favours Atlas-L; between the two this run does not settle it. On overall quality alone Brisk-S is the better buy below $4.15 a point. On point estimates Open-70B is never the better buy; inside the band from $4.15 to $6.11 this run does not separate it from Brisk-S and Atlas-M.

**Our lean.** Ours, not a recommendation: Atlas-M, which the price frontier favours from $6.11 to $11.09 a point; no behaviour check separates it from Atlas-L, and its tail latency is shorter in every task. Below that band we would still pass over Brisk-S: in troubleshooting it resolves the issue 74% of the time against 90%, and confirms before account changes 88% against 98%.

**What would settle it.** The concierge owner declares a price per point of overall quality, or floors on the behaviour checks that matter (confirming before account changes above all).

**Watch.** Every cost is this run's provider routing, which the snapshot does not name; another host for any of these models, Open-70B above all, could move the price bands.

| Option | Role | Configuration | Cost per 1,000 conversations |
|---|---|---|---|
| A · Atlas-L | Candidate | Atlas-L | $75.05 |
| B · Atlas-M | Candidate | Atlas-M | $31.37 |
| C · Brisk-S | Candidate | Brisk-S | $6.01 |
| D · Open-70B | Candidate | Open-70B | $11.51 |

## On this run's unnamed provider routing, each step up in overall quality costs more per point: Brisk-S to Atlas-M from $4.15 to $6.11 per point per 1,000 conversations, Atlas-M to Atlas-L from $11.09 to $22.05.

_Caveats: No arm is the control; Cost routing and coverage unknown._

| Step along the frontier | Price per point | 95% unpaired interval |
|---|---|---|
| Brisk-S → Atlas-M | $4.94 | $4.15 to $6.11 |
| Atlas-M → Atlas-L | $14.76 | $11.09 to $22.05 |

Price per point of overall quality along the frontier, per 1,000 conversations, with unpaired intervals; Open-70B sits off the frontier on point estimates only.

### Atlas-L leads Atlas-M on overall quality in every task that carries traffic, most in troubleshooting at +5 points; on speculation bait, the test slice, the two are not told apart.

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

Task rows only, no overall row: the gap pooled by traffic share is +2.96 (from +1.981 to +3.939).

### In troubleshooting Atlas-M is not told apart from Atlas-L on any check, while Brisk-S resolves the issue 16 points less often than Atlas-M.

_Caveats: A judge is not described._

| Slice | Arm | Resolves the issue | 95% uncorrected interval | Gives clear steps | 95% uncorrected interval | Confirms before account changes | 95% uncorrected interval | Escalates when stuck | 95% uncorrected interval |
|---|---|---|---|---|---|---|---|---|---|
| Troubleshooting (multi-turn) | A · Atlas-L | 91% | 86.8% to 95.2% | 96% | 93.1% to 98.9% | 98% | 96.0% to 100.0% | 94% | 90.5% to 97.5% |
| Troubleshooting (multi-turn) | B · Atlas-M | 90% | 85.6% to 94.4% | 95% | 91.8% to 98.2% | 98% | 96.0% to 100.0% | 95% | 91.8% to 98.2% |
| Troubleshooting (multi-turn) | C · Brisk-S | 74% | 67.6% to 80.4% | 90% | 85.6% to 94.4% | 88% | 83.3% to 92.7% | 85% | 79.8% to 90.2% |
| Troubleshooting (multi-turn) | D · Open-70B | 82% | 76.4% to 87.6% | 93% | 89.3% to 96.7% | 81% | 75.3% to 86.7% | 90% | 85.6% to 94.4% |

Troubleshooting pass rates on the behaviour checks read there, with their intervals.

### Both cheaper models confirm before account changes less often than Atlas-M: Open-70B by 17 points (from +11 to +23), Brisk-S by 10 (from +4.9 to +15.1).

_Caveats: A judge is not described._

| Row | Value | 95% unpaired interval |
|---|---|---|
| Troubleshooting (multi-turn) | +17 | +11 to +23 |

Each row is Atlas-M less Open-70B on Confirms before account changes, overall and in each place both hold it.

The interval is independent: each arm's own spread (its stated half-width), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same.

Atlas-M less Open-70B on confirming before account changes in troubleshooting, with its unpaired interval.

| Row | Value | 95% unpaired interval |
|---|---|---|
| Troubleshooting (multi-turn) | +10 | +4.9 to +15.1 |

Each row is Atlas-M less Brisk-S on Confirms before account changes, overall and in each place both hold it.

The interval is independent: each arm's own spread (its stated half-width), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same.

Atlas-M less Brisk-S on confirming before account changes in troubleshooting, with its unpaired interval.

### On research questions Atlas-M and Atlas-L are not told apart on tool use, citing or unsupported claims; Brisk-S and Open-70B score lower on all three, though Brisk-S's citing gap to Atlas-M is not separated from zero (from 0 to +10).

_Caveats: A judge is not described._

| Slice | Arm | Calls research tool only when required | 95% uncorrected interval | Cites the retrieved source | 95% uncorrected interval | No unsupported claims | 95% uncorrected interval |
|---|---|---|---|---|---|---|---|
| Research-needed questions | A · Atlas-L | 97% | 94.8% to 99.2% | 96% | 93.5% to 98.5% | 95% | 92.2% to 97.8% |
| Research-needed questions | B · Atlas-M | 95% | 92.2% to 97.8% | 94% | 91.0% to 97.0% | 94% | 91.0% to 97.0% |
| Research-needed questions | C · Brisk-S | 83% | 78.2% to 87.8% | 89% | 85.0% to 93.0% | 84% | 79.4% to 88.6% |
| Research-needed questions | D · Open-70B | 88% | 83.9% to 92.1% | 85% | 80.5% to 89.5% | 86% | 81.6% to 90.4% |

Research-question pass rates on the behaviour checks read there, with their intervals.

#### When baited, Brisk-S declines to speculate 17 points less often than Atlas-M; speculation bait carries 0% of traffic, so this is a test of behaviour, not a rate users will meet.

_Caveats: A judge is not described._

| Row | Value | 95% uncorrected interval |
|---|---|---|
| A · Atlas-L | 99% | 97.4% to 100.0% |
| B · Atlas-M | 98% | 95.8% to 100.0% |
| C · Brisk-S | 81% | 74.7% to 87.3% |
| D · Open-70B | 89% | 84.0% to 94.0% |

Declines to speculate on speculation bait, each model with its interval.

#### In persona small talk Open-70B is not told apart from Atlas-M on overall quality, 90 against 88 (gap from -0.1213 to +4.121), nor from either Atlas model on staying in character, at $6.40 per 1,000 conversations against $17.00 on this run's unnamed provider routing.

_Caveats: Cost routing and coverage unknown; A judge is not described._

| Slice | Arm | Overall quality | 95% uncorrected interval | Cost per 1k conversations | Stays in character | 95% uncorrected interval |
|---|---|---|---|---|---|---|
| Persona small talk | A · Atlas-L | 91 | 89.5 to 92.5 | $41.00 | 95% | 92.1% to 97.9% |
| Persona small talk | B · Atlas-M | 88 | 86.5 to 89.5 | $17.00 | 95% | 92.1% to 97.9% |
| Persona small talk | C · Brisk-S | 84 | 82.5 to 85.5 | $3.30 | 86% | 81.4% to 90.6% |
| Persona small talk | D · Open-70B | 90 | 88.5 to 91.5 | $6.40 | 97% | 94.7% to 99.3% |

Persona small talk: overall quality, cost per 1,000 conversations and staying in character for each model.

#### Atlas-L is the slowest model in every task; on research questions its tail latency reaches 4.86 s against 2.84 s for Atlas-M, and no model dropped a trial to get there.

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

Latency from median to tail for each model in every task.

## What you could change

| What you could change | What it decides | What happens as it moves |
|---|---|---|
| Price of one point of overall quality, per 1,000 conversations | Which model is the better buy: Brisk-S, Atlas-M or Atlas-L | On overall quality alone: below $4.15 Brisk-S; from $6.11 to $11.09 Atlas-M; above $22.05 Atlas-L. Inside each interval band between those, this run does not settle the pick; in the lower band it does not separate Open-70B from Brisk-S and Atlas-M either, though on point estimates Open-70B is never the better buy. Brisk-S's behaviour-check shortfalls are not priced in. |
| A floor on confirming before account changes | Whether the cheaper models stay options | A floor near Atlas-M's 98% would remove Brisk-S and Open-70B, whose intervals sit wholly below it; no floor was declared. |

## Next steps

- The concierge owner declares a price per point of overall quality, or floors on the behaviour checks. _Turns the trade into a decision between Atlas-M and Atlas-L, and says whether the cheaper models stay options._
- Re-run Atlas-M and Atlas-L on the same cases with paired scoring. _Whether the small behaviour-check gaps between them separate, and narrows the price band in which this run cannot choose._
- Run a routed mix as a measured option: Atlas-M for most traffic, Open-70B for persona small talk. _What the mix keeps of Atlas-M's quality and how much it saves, so it can be priced like the others; nobody has run it, so no figure states it yet._
- Name the judge of the behaviour checks and calibrate it against human ratings. _Whether the cheaper models' shortfalls on the checks are real or a judge's bias, and so whether Brisk-S belongs back on the table at low prices._
- Record the provider serving each model, the cost coverage and the monthly volume of conversations. _Whether the price bands hold beyond this run's routing, Open-70B's above all, and what the trade is worth a month._

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

- **Cost by task** Every task keeps the same cost order, so it adds nothing to the trade; the appendix holds it, and persona small talk's costs appear with the persona claim.
- **Overall quality per task for Brisk-S and Open-70B** Brisk-S sits below Atlas-M in every task and Open-70B in every task but persona small talk, where the two are not told apart (shown in the persona claim); the pooled figures behind the price table carry the rest.
- **An overall row in the per-task gap between Atlas-L and Atlas-M** Overall quality is held only per task, so no overall row can be drawn there; the caption states the gap pooled by traffic share with its interval instead, and the chart's own note promising an overall row should be read that way.
- **A dominance call weighing quality beside cost** With no model-level quality figure, the computed dominance rested on cost alone and would name Brisk-S; the price frontier weighs cost against quality instead.
- **Failure examples** None were kept with this run.

### Method

- **Cases** 1,190 conversations per model over the tasks listed under traffic shares, the same count for each; repeats are not stated. No trial was lost or failed for any model.
- **Overall quality** A judged score per conversation; the judge is calibrated on 200 human ratings with 86% agreement. Overall figures pool the tasks by traffic share.
- **Behaviour checks** Pass-or-fail judged checks, each read only in the task it applies to. The campaign does not define what passing each means beyond its name, nor who judges them. _Caveats: A judge is not described._
- **Traffic shares** Billing 42%, persona small talk 24%, troubleshooting 19%, research questions 15%, speculation bait 0%: speculation bait is a test slice and counts toward no overall figure.
- **Intervals** At the level each block prints, uncorrected. With no control and no paired contrasts, every gap between two models combines each model's own interval as if they ran different cases, which reads wider than a paired interval would. Price bands are read on these unpaired intervals throughout.
- **Cost** Cost per 1,000 conversations, weighted by traffic share. The provider serving each model and how many trials reported a cost are not stated, and no monthly volume was declared.
- **Latency** Median and tail percentile per task, in seconds.
- **Not compared** No model is a control and nothing was declared before the run: no gate, floor, margin or price per point.
- **Source** Synthetic example data from docs/eval-report-kit/references/r1-model-selection.json, run 2026-10-21.

_Written by claude-opus-5-5, served as claude-opus-5-5, for $4.45, with analyst skill ed7e2c60b5ad, on 2026-10-29._
