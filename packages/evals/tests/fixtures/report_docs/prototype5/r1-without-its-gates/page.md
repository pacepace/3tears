Choice · candidates compared · Concierge chat assistant · 4 arms · 5 tasks · Run 2026-10-21 · Synthetic example data

**Draft: its editor's review left 2 findings open.**

# Brisk-S costs least but trails on quality and on confirming account changes; Atlas-M closes most of the gap at a mid price; Atlas-L leads at the top price. No price per point of quality was set.

A point of overall quality costs $4.15 to $6.11 per thousand conversations from Brisk-S to Atlas-M, and $11.09 to $22.05 from Atlas-M to Atlas-L. A declared price per point would decide it.

**Confidence: High, if the caveats below aren't material.**

- **A judge is not described.** High applies to the priced bands, which rest on overall quality from a judge calibrated on 200 human ratings with 86% agreement. The lean's reasons rest on the behaviour checks, whose judge is not described: confirming account changes, declining to speculate, and Atlas-L not being told apart from Atlas-M on any check. Those reasons would move if that judge is lenient or harsh toward one model, and stay unconfirmed until the calibration step is done.
- **Nothing declares what is good enough.** Only for a reader who wants a call rather than a priced trade: with no price per point and no floor set, the choice among the three models on the price frontier stays the owner's.
- **No arm is the control.** Only for the gaps not told apart, such as Atlas-L against Atlas-M on the behaviour checks, or Open-70B against Brisk-S on overall quality: every gap here is read unpaired, which errs wide, so a paired read could separate them; the gaps already clear of zero would only narrow their intervals.
- **The pick is the best of several candidates.** Only if a fresh run reads Atlas-L's edge over Atlas-M lower, as the top of several candidates tends to: that would make the step up to Atlas-L dearer per point, which favours the lean rather than reversing it.
- **Cost routing unknown.** Only if another provider serves these models at different prices: the snapshot names no provider for any model, so every cost and price per point holds for this run's routing alone.

**The trade.** Moving up from Brisk-S ($6.01 per thousand conversations) to Atlas-M ($31.37) or Atlas-L ($75.05) buys overall quality in every task that carries traffic and better behaviour on the checks that guard users; each step up also adds latency.

**Where the answer flips.** If a point of overall quality is worth less than $4.15 to $6.11 per thousand conversations, Brisk-S is the buy; between that band and $11.09 to $22.05, Atlas-M; above it, Atlas-L. Inside either band this run does not settle the pick.

**Our lean.** Ours, not a declared rule: Atlas-M, for any price per point in the middle band. It avoids Brisk-S's misses on confirming account changes and declining to speculate, while Atlas-L costs 2.4 times what Atlas-M does and is not told apart from it on any behaviour check. Those reasons rest on an undescribed judge until it is calibrated.

**Why not the others.** Open-70B ($11.51) confirms before account changes less often than any other model (81%). On price it sits off the frontier on point estimates only: its pooled overall quality (85.49) is +0.79 above Brisk-S's, an interval of -0.1893 to +1.769 not told apart from zero, so a point bought moving to it costs $6.96 at the estimate and as little as $3.11 at the interval's top, inside the lower band.

**Watch.** Atlas-M's weak side is troubleshooting, where it trails Atlas-L by 5 points of overall quality. Speculation bait carries 0% of traffic, so declining to speculate is read on a test set, not on live conversations.

**What would settle it.** The concierge owner declares a price per point of overall quality, or a floor on confirming before account changes; either turns this trade into a decision.

| Option | Role | Configuration | Cost per 1,000 conversations |
|---|---|---|---|
| A · Atlas-L | Candidate | Atlas-L | $75.05 |
| B · Atlas-M | Candidate | Atlas-M | $31.37 |
| C · Brisk-S | Candidate | Brisk-S | $6.01 |
| D · Open-70B | Candidate | Open-70B | $11.51 |

## Each step up the price frontier buys overall quality, and the step from Atlas-M to Atlas-L costs more per point than the step from Brisk-S to Atlas-M. Pooled by traffic, overall quality reads 84.7 for Brisk-S, 89.83 for Atlas-M and 92.79 for Atlas-L: the first step gains +5.13 points (+4.151 to +6.109), the second +2.96 (+1.981 to +3.939).

| Step along the frontier | Price per point | 95% unpaired interval |
|---|---|---|
| Brisk-S → Atlas-M | $4.94 | $4.15 to $6.11 |
| Atlas-M → Atlas-L | $14.76 | $11.09 to $22.05 |

Price per point of overall quality at each step, per thousand conversations; intervals are unpaired.

### Atlas-M scores above Brisk-S on overall quality in every task, by the most in troubleshooting (9 points).

| Row | Value | 95% unpaired interval |
|---|---|---|
| Billing & account Q&A | +3 | +1.444 to +4.556 |
| Troubleshooting (multi-turn) | +9 | +6.737 to +11.26 |
| Persona small talk | +4 | +1.879 to +6.121 |
| Research-needed questions | +8 | +6.020 to +9.980 |
| Speculation bait | +6 | +3.454 to +8.546 |

Each row is Atlas-M less Brisk-S on Overall quality, overall and in each place both hold it.

The interval is independent: each arm's own spread (its stated half-width), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same.

Atlas-M minus Brisk-S on overall quality per task, with unpaired intervals.

### Brisk-S falls behind Atlas-M on the two behaviour checks that guard users most directly: Atlas-M confirms before account changes 10 points more often, and declines speculation bait more often too.

| Row | Value | 95% unpaired interval |
|---|---|---|
| Troubleshooting (multi-turn) | +10 | +4.9 to +15.1 |

Each row is Atlas-M less Brisk-S on Confirms before account changes, overall and in each place both hold it.

The interval is independent: each arm's own spread (its stated half-width), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same.

Atlas-M minus Brisk-S on confirming before account changes, read in troubleshooting, with its unpaired interval.

| Row | Value | 95% unpaired interval |
|---|---|---|
| Speculation bait | +17 | +10.3 to +23.7 |

Each row is Atlas-M less Brisk-S on Declines to speculate, overall and in each place both hold it.

The interval is independent: each arm's own spread (its stated half-width), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same.

Atlas-M minus Brisk-S on declining to speculate, read on speculation bait, a test set with no live traffic, with its unpaired interval.

### Atlas-L scores above Atlas-M on overall quality in every task that carries traffic, by the most in troubleshooting; on speculation bait the two are not told apart.

| Row | Value | 95% unpaired interval |
|---|---|---|
| Billing & account Q&A | +2 | +0.4444 to +3.556 |
| Troubleshooting (multi-turn) | +5 | +2.737 to +7.263 |
| Persona small talk | +3 | +0.8787 to +5.121 |
| Research-needed questions | +3 | +1.02 to +4.980 |
| Speculation bait | +1 | -1.546 to +3.546 |

Each row is Atlas-L less Atlas-M on Overall quality, overall and in each place both hold it.

The interval is independent: each arm's own spread (its stated half-width), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same.

Atlas-L minus Atlas-M on overall quality per task, with unpaired intervals.

#### Latency follows cost: Brisk-S answers fastest and Atlas-L slowest in every task that carries traffic, and no latency budget was declared.

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

Latency from p50 to p95, per task that carries traffic.

## What you could change

| What you could change | What it decides | What happens as it moves |
|---|---|---|
| Price of one point of overall quality, per thousand conversations | Which model on the price frontier is the buy | Brisk-S below $4.15 to $6.11; Atlas-M from there to $11.09 to $22.05; Atlas-L above. Open-70B is off the frontier on point estimates, but its gain over Brisk-S is not told apart from zero, so it could take part of the lower band. |
| A floor on confirming before account changes, read on the point estimate | Whether Brisk-S and Open-70B remain options | A floor above Brisk-S's 88% and below Atlas-M's 98% leaves Atlas-M the least costly option that meets it. |

## Next steps

- The concierge owner declares a price per point of overall quality, or a floor on confirming before account changes. _Turns the trade into a decision: the price picks a band, the floor removes the models below it._
- Name the judge of the behaviour checks and calibrate it against human ratings. _Whether the lean's reasons hold: Brisk-S's misses on confirming account changes and declining to speculate, and Atlas-L's lack of any edge over Atlas-M on those checks._
- Declare the monthly volume and what it counts. _Turns prices per thousand conversations into a monthly cost the owner can weigh._
- Score Atlas-L and Atlas-M on the same cases and read their gap on overall quality paired. _Narrows the upper band, which today's unpaired intervals leave wide. More cases will not flip the order: Atlas-L already leads in every task that carries traffic, and a fresh run reads billing, its thinnest lead, the wrong way 0.6% of the time._
- Measure a routed mix that sends billing questions to Brisk-S and the rest to Atlas-M, with confirming account changes checked on billing too. _Whether billing, the largest task (42% of traffic) and the one where Brisk-S trails Atlas-M least, can take the cheaper model; nobody has run this mix, so it has no price yet._

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

- **Per-task cost figures** The cost order is the same in every task, so the overall cost per model carries the point.
- **The dominance computation** It withheld its call because cost is the only declared axis; with quality axes added it weighed cost alone, since no quality figure is stated overall, so it shows nothing about the trade.
- **Resolves the issue, no unsupported claims, gives clear steps, escalates when stuck, cites the retrieved source, calls the research tool only when required, stays in character** Brisk-S's rate sits below Atlas-M's on each, and Atlas-L and Atlas-M are not told apart on any; none changes a band, and the appendix holds every rate.
- **A run sized to separate Atlas-L from Atlas-M on calling the research tool only when required** That check changes no band, so a run aimed at it would not move the pick.
- **Failure examples** None were kept with this run.

### Method

- **Cases** 1,190 cases per model, split over the tasks the tables list; no model lost or failed a case. Repeats are not stated.
- **Overall quality** A judged score on every task, from a judge calibrated on 200 human ratings with 86% agreement.
- **Behaviour checks** Pass or fail judgements, each read in the one task that tests it; the snapshot does not say who judged them.
- **Pooling** Overall quality across tasks is weighted by each task's traffic share; speculation bait carries 0%, so it tests behaviour without moving the pooled score. The full-results tables list per-task scores only, so each model's pooled score is stated in the lead claim, and Open-70B's under why not.
- **Intervals** At the level each table prints, uncorrected. With no control and no paired contrasts, every gap between two models combines their own intervals as independent, which errs wide.
- **Cost** Cost per thousand conversations as this run served each model; no provider is named and no monthly volume was declared.
- **Latency** p50 and p95 per task; no budget was declared.
- **Not compared** No routed mix of models was run, and nothing was declared before the run: no gate, floor, margin or exchange rate.
- **Source** Synthetic example data from docs/eval-report-kit/references/r1-model-selection.json, run 2026-10-21.

### Review findings left open

- **Should fix** · Claim 1: This is still open from last round. c1's block draws only the price per point, but its sentence and the whole frontier rest on pooled overall quality. Add a second block, or columns in this one, showing each frontier model's pooled overall quality (Brisk-S 84.7, Atlas-M 89.83, Atlas-L 92.79) and each step's gain with its unpaired interval (+5.13, +4.151 to +6.109; +2.96, +1.981 to +3.939). Also add a pooled overall quality column to 'Every arm by measure', including Open-70B's 85.49. That way a reader can find each model's quality in a table, not only in prose.
- **Should fix** · The page: 'Why not the others' says Open-70B costs 'as little as $3.11 at the interval's top, inside the lower band.' But $3.11 is below the lower band ($4.15 to $6.11), not inside it. At that end of its gain, the move from Open-70B on to Atlas-M would cost about $5.91 per point. So Open-70B could be the buy from about $3.11 to $5.91. That range overlaps the lower band and part of the region where 'Where the answer flips' calls Brisk-S the buy without condition. Change 'inside the lower band' to say it reaches below the band. Add a short clause to the flip line: below the lower band, Brisk-S is the buy on point estimates, and Open-70B could contest it there because its gain over Brisk-S is not told apart from zero. Open-70B's 81% on confirming account changes still argues against it.

_Written by claude-opus-5-5, served as claude-opus-5-5, for $3.76, with analyst skill 2a7662d928b0, on 2026-10-29._
