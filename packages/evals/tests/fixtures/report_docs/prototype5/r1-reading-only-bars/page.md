Choice · candidates compared · Concierge chat assistant · 4 arms · 5 tasks · Run 2026-10-21 · Synthetic example data

**Draft: its editor's review left 2 findings open.**

# No price was set on quality: Atlas-M costs $44 less per 1,000 conversations than Atlas-L and trails it only on overall quality; Brisk-S is cheapest but slips on safety checks.

Each step up the price frontier buys overall quality at a rising price per point, and no check separates Atlas-M from Atlas-L. The pick flips with the price the owner puts on a point of overall quality, and with any floor set on the safety checks.

**Confidence: High, if the caveats below aren't material.**

- **Nothing declares what is good enough.** It moves the pick, not the figures: with no bar, floor or price per point declared, the option follows whatever price the owner sets, so the report prices the trade and leaves the call to the concierge owner.
- **A judge is not described.** Only if the unstated judge of the pass/fail checks reads Brisk-S or Open-70B more harshly than the Atlas models; the price frontier itself rests on overall quality, whose judge is calibrated on 200 human ratings at 86% agreement.
- **No arm is the control.** Only if a paired reading on the same cases would separate Atlas-L from Atlas-M on a check; every gap here combines each model's own interval as independent, which reads wider than a paired one, so pairing could only sharpen a gap, not reverse the frontier's order.
- **The pick is the best of several candidates.** Only if a fresh run reads Atlas-L's lead over Atlas-M lower than its +2.96 pooled points (unpaired interval +1.981 to +3.939); a smaller lead raises the price of each point bought on to Atlas-L and favours Atlas-M.
- **Cost routing unknown.** Only if another provider serving any of these models would reorder their costs; the run names no provider for any model, so each cost is this run's routing, not the model's own.

**The trade.** Atlas-L leads Atlas-M by +2.96 points of overall quality, pooled over traffic, and costs $44 more per 1,000 conversations. Atlas-M leads Brisk-S by +5.13 points, costs $25 more, and leads it on the safety checks too.

**Where the answer flips.** Brisk-S is the best buy if a point of overall quality is worth less than somewhere from $4.15 to $6.11 per 1,000 conversations; Atlas-L if it is worth more than somewhere from $11.09 to $22.05; Atlas-M in between. A price inside either band is not settled by this run.

**Our lean.** Ours, not a declared rule: Atlas-M, for any price per point in the middle band. No pass/fail check tells it apart from Atlas-L, while Brisk-S's saving comes with drops on declining to speculate and confirming before account changes.

**Why not the others.** Open-70B is off the price frontier on point estimates only: its pooled lead over Brisk-S is +0.79 points, interval -0.1893 to +1.769, so a point bought on to it could cost as little as $3.11, inside Brisk-S's band. What clearly counts against it: it confirms before account changes 17 points less often than Atlas-M, and trails Atlas-M on overall quality in every task but persona small talk.

**Watch.** No line was declared, so there is no weakest margin to watch. Speculation bait carries 0% of traffic: its decline rates say what each model does when baited, not what live traffic meets. Every cost is this run's provider routing, which is unnamed.

**What would settle it.** The concierge owner declares a price per point of overall quality, or a floor on declining to speculate and confirming before account changes; either turns the trade into a call.

| Option | Role | Configuration | Cost per 1,000 conversations |
|---|---|---|---|
| A · Atlas-L | Candidate | Atlas-L | $75.05 |
| B · Atlas-M | Candidate | Atlas-M | $31.37 |
| C · Brisk-S | Candidate | Brisk-S | $6.01 |
| D · Open-70B | Candidate | Open-70B | $11.51 |

## Each step up the price frontier buys overall quality at a rising price: a point costs $4.94 per 1,000 conversations from Brisk-S to Atlas-M, and $14.76 from Atlas-M on to Atlas-L.

_Caveats: Cost routing unknown._

| Row | Value | 95% unpaired interval |
|---|---|---|
| Brisk-S → Atlas-M | $4.94 | $4.15 to $6.11 |
| Atlas-M → Atlas-L | $14.76 | $11.09 to $22.05 |

Each step's price per point per 1,000 conversations: where the best option moves as a point's price rises.

Price per point of overall quality at each step of the frontier, per 1,000 conversations, with unpaired intervals; on point estimates Open-70B is never the best buy, though its interval against Brisk-S leaves that open.

### Brisk-S's saving comes with drops on the safety checks: it declines to speculate 17 points less often than Atlas-M, and Open-70B also falls short of both Atlas models.

_Caveats: A judge is not described._

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

Pass rates on the safety and research checks, each read in its own task, with intervals; speculation bait carries 0% of traffic, so its rates are a test.

### Atlas-L is clearly ahead of Atlas-M on overall quality in every task but speculation bait, where its +1 lead crosses zero, and most in troubleshooting at +5 points. On the checks, their largest gaps, tool calls and citing the source, both +2 points, are not told apart from zero.

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

Atlas-L less Atlas-M on overall quality, one row per task with unpaired intervals and no pooled row; only speculation bait crosses zero.

| Row | Value | 95% unpaired interval |
|---|---|---|
| Research-needed questions | +2 | -1.6 to +5.6 |

Each row is Atlas-L less Atlas-M on Calls research tool only when required, overall and in each place both hold it.

The interval is independent: each arm's own spread (its stated half-width), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same.

Atlas-L less Atlas-M on calling the research tool only when required; citing the source, the other gap of the same size, reads -1.9 to +5.9, and the remaining checks sit in the safety table.

#### Open-70B is not told apart from Atlas-M on persona small talk, at $6.40 per 1,000 conversations against Atlas-M's $17.00 there, but trails it in every other task.

_Caveats: Cost routing unknown._

| Row | Value | 95% unpaired interval |
|---|---|---|
| Billing & account Q&A | -6 | -7.556 to -4.444 |
| Troubleshooting (multi-turn) | -5 | -7.263 to -2.737 |
| Persona small talk | +2 | -0.1213 to +4.121 |
| Research-needed questions | -9 | -10.98 to -7.02 |
| Speculation bait | -3 | -5.546 to -0.4544 |

Each row is Open-70B less Atlas-M on Overall quality, overall and in each place both hold it.

The interval is independent: each arm's own spread (its stated half-width), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same.

Open-70B less Atlas-M on overall quality, one row per task with unpaired intervals and no pooled row; persona small talk is the one row that crosses zero.

| Slice | Arm | Overall quality | 95% uncorrected interval | Cost per 1k conversations |
|---|---|---|---|---|
| Persona small talk | B · Atlas-M | 88 | 86.5 to 89.5 | $17.00 |
| Persona small talk | D · Open-70B | 90 | 88.5 to 91.5 | $6.40 |

Overall quality and cost per 1,000 conversations in persona small talk, as served in this run.

#### Latency follows cost: in research-needed questions, the slowest task, Atlas-L's p95 latency is 4.86 s against Atlas-M's 2.84 s.

| Group | Row | From | To |
|---|---|---|---|
| Research-needed questions | A · Atlas-L | 2.48 s | 4.86 s |
| Research-needed questions | B · Atlas-M | 1.53 s | 2.84 s |
| Research-needed questions | C · Brisk-S | 0.60 s | 1.22 s |
| Research-needed questions | D · Open-70B | 1.07 s | 2.16 s |

Median to p95 latency in research-needed questions; no latency budget was declared.

## What you could change

| What you could change | What it decides | What happens as it moves |
|---|---|---|
| Price of one point of overall quality, per 1,000 conversations | Which of Brisk-S, Atlas-M and Atlas-L is the best buy | Below a band from $4.15 to $6.11, Brisk-S; above a band from $11.09 to $22.05, Atlas-L; Atlas-M between. Open-70B is never the best buy on point estimates; its interval against Brisk-S leaves a price from $3.11 upward open. |
| A floor on declining to speculate | Whether Brisk-S and Open-70B stay in the choice | A floor above Brisk-S's 81% removes it, and one above Open-70B's 89% removes it too; both Atlas models sit above either. |

## Next steps

- The concierge owner declares a price per point of overall quality, or floors on declining to speculate and confirming before account changes. _Turns the trade into a call: which band the price falls in names the model._
- Measure a routed option as its own arm: Open-70B on persona small talk, Atlas-M on everything else, on the same cases. _Whether routing the small talk keeps Atlas-M's safety checks at a lower cost; nobody ran this mix, so no figure states its cost or quality yet._
- State which judge scored the pass/fail checks and calibrate it against human ratings. _Turns the judge caveat on the safety checks into evidence, so Brisk-S's and Open-70B's drops can be read as absolute rates._
- Re-score Atlas-L and Atlas-M paired on the same cases. _Whether any check separates them; read independently, the tool-call check would need 1,555 cases per model to separate its gap four times in five._
- Name the provider serving each model in the cost records. _Whether the cost order is the models' or this run's routing._

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

- **Gives clear steps, escalates when stuck, cites the retrieved source and stays in character** Same order as the checks shown, with smaller gaps; none separates Atlas-L from Atlas-M, and Open-70B's in-character lead sits within persona small talk, where its overall quality is already shown.
- **Overall quality and cost for every model in every task** The pooled price frontier carries the call, and within billing, the largest task, the frontier keeps the same three models in the same order.
- **Median and p95 latency in the other tasks** Same order as research-needed questions in every task; no latency budget was declared, so latency doesn't bind.
- **A drawn row for Atlas-L's pooled lead over Atlas-M, and a drawn row for citing the source** No pooled gap can be drawn beside the per-task rows, and a gap block holds one check; the pooled lead and its interval are stated in the trade and the confidence line, and the citing gap's interval in the tool-call caption.
- **Failure examples** The run kept none.

### Method

- **Cases** Each model ran 1,190 cases over the same tasks; none was lost or failed. Repeats are not stated.
- **Overall quality** An ordinal score from a judge calibrated on 200 human ratings, at 86% agreement, read in every task.
- **Pass/fail checks** Each check is read only in its own task. The input says neither who judged them nor what makes a pass beyond each check's name.
- **Pooled overall quality** Task means weighted by traffic share: billing 42%, persona small talk 24%, troubleshooting 19%, research 15%; speculation bait carries 0%, so it enters no pooled figure.
- **Intervals** At the level each block prints, uncorrected. With no control and no paired contrast, a gap between two models combines each one's own interval as if they ran different cases, which reads wider than a paired interval would. The frontier is read on point estimates; where an interval leaves a step open, the page says so.
- **Cost basis** Cost per 1,000 conversations as served in this run, with no provider named for any model. No monthly volume was declared, so no monthly figure is stated.
- **Latency** Median and p95 per task; no budget declared, and whether the models shared a machine is not stated.
- **Not declared** No control, no gate, floor, margin or price per point; the call is the concierge owner's.
- **Source** Synthetic example data from docs/eval-report-kit/references/r1-model-selection.json, run 2026-10-21.

### Review findings left open

- **Must fix** · The page: The confidence reads High while the cost-routing caveat stays open, and its clause misstates what would make it matter. The clause says 'Only if another provider serving any of these models would reorder their costs'. But the call is a set of price-per-point bands built from cost gaps: price.atlas-m.atlas-l.overall.per_point is $14.76, which is $44 over +2.96 points. A provider charging differently for any model moves the bands without reordering anything. Rewrite the clause to say: the $4.15–$6.11 and $11.09–$22.05 bands hold for this run's unnamed routing; a different provider price for any model shifts where the pick flips, even with the cost order unchanged; the High applies to the quality figures and to the bands as priced in this run. Otherwise drop the form to low.
- **Should fix** · The page: 'Why not the others' says a point bought on to Open-70B 'could cost as little as $3.11, inside Brisk-S's band'. That $3.11 is price.brisk-s.open-70b.overall.per_point_low_unpaired, and it falls below the $4.15–$6.11 band, not inside it. A reader will look for $3.11 within $4.15–$6.11 and not find it. Write: 'as little as $3.11, below the $4.15–$6.11 band, where Brisk-S would otherwise be the best buy'. That matches what the explore row already says.

_Written by claude-opus-5-5, served as claude-opus-5-5, for $3.75, with analyst skill ed7e2c60b5ad, on 2026-10-29._
