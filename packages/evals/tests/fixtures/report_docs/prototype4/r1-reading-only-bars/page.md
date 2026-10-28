Choice · candidates compared · Concierge chat assistant · 4 arms · 5 tasks · Run 2026-10-21 · Synthetic example data

# Atlas-M confirms and declines as reliably as Atlas-L at $31.37 against $75.05 per thousand conversations, but scores 2.96 points lower on quality.

Each point of overall quality bought from Atlas-M up to Atlas-L costs $14.76 per thousand conversations, against $4.94 for the step up from Brisk-S. A declared price per point, or a guardrail on confirming and declining, would turn this trade into a decision.

**No gate, guardrail, control or price per point was declared before the run, so the choice among the measured models is the concierge owner's.**

**Confidence: High, if the caveats below aren't material.**

- **Nothing declares what is good enough.** Material to any call: which model wins depends on a price per point of overall quality that nobody has set; the frontier itself is measured.
- **A judge is not described.** Only if the unnamed judge of the pass/fail checks favours the Atlas models; the price rests on overall quality, whose judge is calibrated on 200 human ratings at 86% agreement.
- **No arm is the control.** Only for a difference whose unpaired interval crosses zero, such as Atlas-L against Atlas-M on speculation bait; pairing the shared cases could only narrow the intervals shown.
- **The pick is the best of several candidates.** Only if a re-run shrinks Atlas-L's lead over Atlas-M by more than its interval; with 4 candidates the pull is small beside leads that exclude zero in four of five tasks.

**The trade.** Atlas-L buys 2.96 points of overall quality over Atlas-M for $43.68 more per thousand conversations. Atlas-M in turn buys 5.13 points over Brisk-S for $25.36 more, and keeps confirming and declining at Atlas-L's rate where Brisk-S does not.

**Where the answer flips.** On overall quality alone: below $4.94 a point, Brisk-S is the best buy; up to $14.76, Atlas-M; above it, Atlas-L. Open-70B is never the best buy.

**Our lean.** Ours, not a rule the campaign set: Atlas-M, at any price per point under $14.76. It is the cheapest model that confirms before account changes and declines to speculate as reliably as Atlas-L.

**Watch.** Atlas-M's largest shortfall is in multi-turn troubleshooting, 5 points behind Atlas-L.

**What would settle it.** The concierge owner declares a price per point of overall quality, or guardrails on confirming before account changes and declining to speculate.

| Option | Role | Configuration | Cost per 1,000 conversations |
|---|---|---|---|
| A · Atlas-L | Candidate | Atlas-L | $75.05 |
| B · Atlas-M | Candidate | Atlas-M | $31.37 |
| C · Brisk-S | Candidate | Brisk-S | $6.01 |
| D · Open-70B | Candidate | Open-70B | $11.51 |

## Three models sit on the price frontier, and each point gets dearer up it: Atlas-M to Atlas-L costs $14.76 per point per thousand conversations, against $4.94 from Brisk-S to Atlas-M.

| Step along the frontier | Price per point | 95% unpaired interval |
|---|---|---|
| Brisk-S → Atlas-M | $4.943 | $4.15 to $6.109 |
| Atlas-M → Atlas-L | $14.76 | $11.09 to $22.05 |

Price per point of overall quality, pooled over tasks by traffic share; intervals unpaired.

### Atlas-L leads Atlas-M on overall quality in four of five tasks, most in troubleshooting at 5 points; on speculation bait nothing separates them.

| Row | Value | 95% unpaired interval | Note |
|---|---|---|---|
| Billing & account Q&A | +2 | +0.4444 to +3.556 | +2 |
| Troubleshooting (multi-turn) | +5 | +2.737 to +7.263 | +5 |
| Persona small talk | +3 | +0.8787 to +5.121 | +3 |
| Research-needed questions | +3 | +1.02 to +4.980 | +3 |
| Speculation bait | +1 | -1.546 to +3.546 | +1 |

Each row is Atlas-L less Atlas-M on Overall quality, overall and in each place both hold it.

The interval is independent: each arm's own spread (its stated half-width), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same.

Atlas-L less Atlas-M on overall quality, per task; intervals unpaired.

### Atlas-M confirms before account changes and declines to speculate as often as Atlas-L; Open-70B confirms 17 points less often and Brisk-S declines 17 points less often.

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

Each check is judged only in the task that asks it; intervals uncorrected.

#### Open-70B is off the frontier on overall quality; its one strength is persona small talk, where it stays in character 97% of the time.

| Panel | Row | Value | 95% uncorrected interval |
|---|---|---|---|
| Billing & account Q&A | A · Atlas-L | 94 | 92.9 to 95.1 |
| Billing & account Q&A | B · Atlas-M | 92 | 90.9 to 93.1 |
| Billing & account Q&A | C · Brisk-S | 89 | 87.9 to 90.1 |
| Billing & account Q&A | D · Open-70B | 86 | 84.9 to 87.1 |
| Troubleshooting (multi-turn) | A · Atlas-L | 93 | 91.4 to 94.6 |
| Troubleshooting (multi-turn) | B · Atlas-M | 88 | 86.4 to 89.6 |
| Troubleshooting (multi-turn) | C · Brisk-S | 79 | 77.4 to 80.6 |
| Troubleshooting (multi-turn) | D · Open-70B | 83 | 81.4 to 84.6 |
| Persona small talk | A · Atlas-L | 91 | 89.5 to 92.5 |
| Persona small talk | B · Atlas-M | 88 | 86.5 to 89.5 |
| Persona small talk | C · Brisk-S | 84 | 82.5 to 85.5 |
| Persona small talk | D · Open-70B | 90 | 88.5 to 91.5 |
| Research-needed questions | A · Atlas-L | 92 | 90.6 to 93.4 |
| Research-needed questions | B · Atlas-M | 89 | 87.6 to 90.4 |
| Research-needed questions | C · Brisk-S | 81 | 79.6 to 82.4 |
| Research-needed questions | D · Open-70B | 80 | 78.6 to 81.4 |
| Speculation bait | A · Atlas-L | 90 | 88.2 to 91.8 |
| Speculation bait | B · Atlas-M | 89 | 87.2 to 90.8 |
| Speculation bait | C · Brisk-S | 83 | 81.2 to 84.8 |
| Speculation bait | D · Open-70B | 86 | 84.2 to 87.8 |

Overall quality per task; intervals uncorrected.

#### Latency follows cost: on research questions Atlas-L's p95 is 4.86 s, against 2.84 s for Atlas-M.

| Group | Row | From | To |
|---|---|---|---|
| Research-needed questions | A · Atlas-L | 2.48 s | 4.86 s |
| Research-needed questions | B · Atlas-M | 1.53 s | 2.84 s |
| Research-needed questions | C · Brisk-S | 0.60 s | 1.22 s |
| Research-needed questions | D · Open-70B | 1.07 s | 2.16 s |

p50 to p95 on research questions; no latency budget was declared.

## What you could change

| What you could change | What it decides | What happens as it moves |
|---|---|---|
| Price of one point of overall quality, per thousand conversations | Which measured model is the best buy on quality and cost | Below $4.94, Brisk-S; from there to $14.76, Atlas-M; above, Atlas-L. Open-70B is best at no price. This reads overall quality alone; Brisk-S's misses on confirming and declining sit outside it. |
| A guardrail on confirming before account changes and declining to speculate | Which models may be chosen at all | A line on both checks set above Brisk-S's and Open-70B's intervals and below the Atlas models' leaves only Atlas-M and Atlas-L, and the trade becomes theirs alone. |

## Next steps

- The concierge owner declares a price per point of overall quality, or guardrails on confirming before account changes and declining to speculate. _Turns the trade into a decision._
- Run a routed mix, Atlas-L on multi-turn troubleshooting and Atlas-M elsewhere, on the same cases. _What share of Atlas-L's lead it recovers, and at what cost, since troubleshooting carries Atlas-M's largest shortfall._
- Name the judge of the pass/fail checks and calibrate it on human ratings. _Whether Atlas-M's parity with Atlas-L on confirming and declining holds._
- Declare the monthly conversation volume. _Turns per-thousand prices into a monthly budget figure._

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
| Billing & account Q&A | Cost per 1k conversations | 400 | $58 | $24 | $4.60 | $8.90 |
| Billing & account Q&A | Latency | 400 | 3.6 s (p95) | 2.1 s (p95) | 0.9 s (p95) | 1.6 s (p95) |
| Troubleshooting (multi-turn) | Overall quality | 180 | 93 | 88 | 79 | 83 |
| Troubleshooting (multi-turn) | Cost per 1k conversations | 180 | $140 | $59 | $11 | $21 |
| Troubleshooting (multi-turn) | Latency | 180 | 4.68 s (p95) | 2.73 s (p95) | 1.17 s (p95) | 2.08 s (p95) |
| Troubleshooting (multi-turn) | Resolves the issue | 180 | 91% | 90% | 74% | 82% |
| Troubleshooting (multi-turn) | Gives clear steps | 180 | 96% | 95% | 90% | 93% |
| Troubleshooting (multi-turn) | Confirms before account changes | 180 | 98% | 98% | 88% | 81% |
| Troubleshooting (multi-turn) | Escalates when stuck | 180 | 94% | 95% | 85% | 90% |
| Persona small talk | Overall quality | 220 | 91 | 88 | 84 | 90 |
| Persona small talk | Cost per 1k conversations | 220 | $41 | $17 | $3.30 | $6.40 |
| Persona small talk | Latency | 220 | 2.88 s (p95) | 1.68 s (p95) | 0.72 s (p95) | 1.28 s (p95) |
| Persona small talk | Stays in character | 220 | 95% | 95% | 86% | 97% |
| Research-needed questions | Overall quality | 240 | 92 | 89 | 81 | 80 |
| Research-needed questions | Cost per 1k conversations | 240 | $95 | $40 | $8.00 | $15 |
| Research-needed questions | Latency | 240 | 4.86 s (p95) | 2.84 s (p95) | 1.22 s (p95) | 2.16 s (p95) |
| Research-needed questions | Calls research tool only when required | 240 | 97% | 95% | 83% | 88% |
| Research-needed questions | Cites the retrieved source | 240 | 96% | 94% | 89% | 85% |
| Research-needed questions | No unsupported claims | 240 | 95% | 94% | 84% | 86% |
| Speculation bait | Overall quality | 150 | 90 | 89 | 83 | 86 |
| Speculation bait | Cost per 1k conversations | 150 | $45 | $19 | $3.60 | $7.00 |
| Speculation bait | Latency | 150 | 3.24 s (p95) | 1.89 s (p95) | 0.81 s (p95) | 1.44 s (p95) |
| Speculation bait | Declines to speculate | 150 | 99% | 98% | 81% | 89% |

### Left out, and why

- **Resolving the issue, clear steps, escalating, calling the research tool only when needed, citing the source, avoiding unsupported claims** Same order as the checks shown: Atlas-M within noise of Atlas-L, Brisk-S and Open-70B below on point estimates; none changes the trade.
- **Latency outside research questions** Same order in every task; no budget was declared to read it against.
- **Any figure for a routed mix of models** Nobody ran one; it appears only as a next step.
- **Failure examples** None were kept with the evidence.

### Method

- **Cases** 1190 conversations per model over five tasks; repeats are not stated.
- **Overall quality** An ordinal judged score per conversation, held per task; its judge is calibrated on 200 human ratings at 86% agreement.
- **Pass/fail checks** Each judged only in the task that asks it; who judged them is not stated.
- **Intervals** At the level each table prints, uncorrected. With no control, model-to-model differences combine each model's own interval as independent, which reads wider than a paired interval.
- **Pooled quality** Each model's per-task scores weighted by traffic share; used for pricing only.
- **Cost** US dollars per thousand conversations; no monthly volume was declared.
- **Source** Synthetic example data from docs/eval-report-kit/references/r1-model-selection.json

### Review findings left open

- **Should fix** · Claim 4: The claim says Open-70B's 'one strength' is staying in character at 97%. Against Atlas-L's 95% the difference is +2 with an interval of -1.7 to +5.7, and that figure isn't in the block. Either add a staying-in-character panel and say it is level with the Atlas models, or point to what the block does show: on persona overall quality Open-70B scores 90, close to Atlas-L's 91 and above Atlas-M's 88 (interval -0.1 to +4.1).
- **Should fix** · The page: The 'Where the answer flips' line and the lean present $14.76 as a sharp line ('any price per point under $14.76'). The unpaired interval on that step runs from $11.09 to $22.05. Say that in the decision block, for example in Watch: a decider who prices a point anywhere from about $11 to $22 sits where the Atlas-M or Atlas-L call isn't settled by this run.
- **Should fix** · Claim 3: 'As often as Atlas-L' and the headline's 'as reliably' rest on differences whose intervals cross zero: declines +1 (-1.7 to +3.7) on 150 cases, and confirms 98% against 98% on 180. Reword as 'within noise of Atlas-L', or give the difference interval. Also say which model each 17-point gap is measured from: Brisk-S declines 17 points less than Atlas-M but 18 less than Atlas-L.
- **Should fix** · The page: The no-control caveat says 'pairing the shared cases could only narrow the intervals shown', but the snapshot never states that the arms ran the same cases. The caveat also names only speculation-bait quality as an example. Its most decision-relevant use is the Atlas-M/Atlas-L parity on confirming and declining that the lean rests on. State the shared-cases assumption as an assumption and name the parity as a claim this caveat affects.
- **Should fix** · Claim 2: The caption says 'overall and in each place', but there is no overall row. Add the pooled gap of +2.96 (+1.98 to +3.94) so the headline figure appears in its supporting block, and drop the Note column, which repeats Value. Also put a figure key on 'four of five' here and in the best-of-several caveat, as code flagged.

_Written by claude-opus-5-5, served as claude-opus-5-5, for $1.58, with analyst skill b70be6bcfa68, on 2026-10-28._
