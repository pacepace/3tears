Choice · candidates compared · Concierge chat assistant · 4 arms · 5 tasks · Run 2026-10-21 · Synthetic example data

# No bar was set, so the pick is a price: Atlas-L leads on quality at 2.4 times Atlas-M's cost, and Brisk-S is cheapest but slips on safety behaviours Atlas-M holds.

Moving from Brisk-S to Atlas-M buys overall quality at $4.94 a point per thousand conversations; moving on to Atlas-L costs $14.76 a point. A declared price per point, or a floor on the safety behaviours, would turn this into a decision.

**Confidence: High, if the caveats below aren't material.**

- **Nothing declares what is good enough.** Material to any call: no gate, floor, budget or price per point was declared, so which of Brisk-S, Atlas-M and Atlas-L to run is the concierge owner's choice; the price bands are measured, the choice among them is not.
- **A judge is not described.** Material only if the unstated judge of the safety behaviours is harsher on Brisk-S and Open-70B than on the Atlas models; the price bands rest on overall quality, whose judge was calibrated on 200 human ratings at 86% agreement.
- **No arm is the control.** Material only where a difference is unresolved: every gap the trade rests on already excludes zero read as independent arms, and pairing on shared cases can only narrow it; Atlas-L against Atlas-M on speculation bait, and Open-70B against Atlas-M on persona small talk, are where pairing could change a reading.
- **The pick is the best of several candidates.** Material if Atlas-L's lead, the top quality reading of four, reads lower on fresh cases: on billing, the busiest task, its lead's interval (+0.4444 to +3.556) nearly reaches zero.

**The trade.** On traffic-weighted overall quality Atlas-L scores 92.79, Atlas-M 89.83 and Brisk-S 84.7. Their costs per thousand conversations are $75.05, $31.37 and $6.01. Open-70B is the best buy at no price.

**Where the answer flips.** Below $4.94 a point per thousand conversations Brisk-S is the best buy; between that and $14.76, Atlas-M; above it, Atlas-L. The upper edge is the less certain: its interval runs from $11.09 to $22.05.

**Our lean.** Ours, not a declared rule: Atlas-M. It is not separated from Atlas-L on any safety behaviour checked, at less than half the cost, and it declines speculation bait more often than Brisk-S by +17 points. The lean holds for any price per point inside Atlas-M's band.

**What would settle it.** The concierge owner declares a price per point of overall quality, or floors for the safety behaviours. Either decides it on today's figures, with no new run.

**Watch.** Atlas-L's lead is thinnest on billing, the busiest task at 42% of traffic, and as the best of four options its readings run high by chance; a fresh run would tend to shrink the gap that justifies its price.

| Option | Role | Configuration | Cost per 1,000 conversations |
|---|---|---|---|
| A · Atlas-L | Candidate | Atlas-L | $75.05 |
| B · Atlas-M | Candidate | Atlas-M | $31.37 |
| C · Brisk-S | Candidate | Brisk-S | $6.01 |
| D · Open-70B | Candidate | Open-70B | $11.51 |

## Each step up in quality costs far more per point than the last: Atlas-M over Brisk-S at $4.94, Atlas-L over Atlas-M at $14.76.

| Row | Value | 95% unpaired interval | Note |
|---|---|---|---|
| Brisk-S → Atlas-M | $4.943 | $4.15 to $6.109 | $4.940 |
| Atlas-M → Atlas-L | $14.76 | $11.09 to $22.05 | $15.00 |

Each step's price per point per 1,000 conversations: where the best option moves as a point's price rises.

Price of one point of traffic-weighted overall quality, per thousand conversations, for each step along the frontier; Open-70B never lies on it.

### Atlas-L's quality lead over Atlas-M is small but clear of zero in every task that carries traffic, largest in multi-turn troubleshooting at +5 points.

| Row | Value | 95% unpaired interval | Note |
|---|---|---|---|
| Billing & account Q&A | +2 | +0.4444 to +3.556 | +2 |
| Troubleshooting (multi-turn) | +5 | +2.737 to +7.263 | +5 |
| Persona small talk | +3 | +0.8787 to +5.121 | +3 |
| Research-needed questions | +3 | +1.02 to +4.980 | +3 |
| Speculation bait | +1 | -1.546 to +3.546 | +1 |

Each row is Atlas-L less Atlas-M on Overall quality, overall and in each place both hold it.

The interval is independent: each arm's own spread (its stated half-width), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same.

Atlas-L minus Atlas-M on overall quality per task, intervals combined as independent arms; only speculation bait is not separated.

### Brisk-S and Open-70B slip on the safety behaviours where Atlas-M stays level with Atlas-L: Brisk-S declines speculation bait 81% of the time, and Open-70B confirms before account changes 81% of the time.

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

Pass rate on each safety behaviour, in the task that tests it, with intervals; no floor was declared for any of them.

#### Open-70B trails Atlas-M in every task except persona small talk, where the two are not separated and Open-70B costs $6.40 per thousand conversations against $17.00.

| Row | Value | 95% unpaired interval | Note |
|---|---|---|---|
| Billing & account Q&A | -6 | -7.556 to -4.444 | -6 |
| Troubleshooting (multi-turn) | -5 | -7.263 to -2.737 | -5 |
| Persona small talk | +2 | -0.1213 to +4.121 | +2 |
| Research-needed questions | -9 | -10.98 to -7.02 | -9 |
| Speculation bait | -3 | -5.546 to -0.4544 | -3 |

Each row is Open-70B less Atlas-M on Overall quality, overall and in each place both hold it.

The interval is independent: each arm's own spread (its stated half-width), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same.

Open-70B minus Atlas-M on overall quality per task, intervals combined as independent arms.

#### Atlas-L is also the slowest in every task, with a research-question p95 of 4.86 s against Atlas-M's 2.84 s.

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

Latency from median to p95 per task; no latency budget was declared.

## What you could change

| What you could change | What it decides | What happens as it moves |
|---|---|---|
| Price of one point of overall quality, per thousand conversations | Which of Brisk-S, Atlas-M and Atlas-L is the best buy | Brisk-S below $4.94, Atlas-M up to $14.76, Atlas-L above it; Open-70B at no price. |
| A floor on declining speculation bait, read on the point estimate | Whether Brisk-S and Open-70B stay options | A floor above Brisk-S's 81% removes it, and Open-70B becomes the best buy up to $4.58 a point; a floor above Open-70B's 89% removes both, leaving Atlas-M against Atlas-L. |

## Next steps

- The concierge owner declares a price per point of overall quality and the monthly volume. _Which of Brisk-S, Atlas-M and Atlas-L to run, and what it costs a month; no run is needed._
- Declare floors for declining speculation, confirming before account changes and avoiding unsupported claims. _Whether Brisk-S and Open-70B are options at all, which narrows the trade to Atlas-M against Atlas-L if both miss._
- State or calibrate the judge behind the safety behaviours with human ratings, as the overall-quality judge was. _Whether the safety gaps that rule out the cheap models are real, turning the judge caveat into evidence._
- Run a routed mix, persona small talk to Open-70B and the rest to Atlas-M, on the same cases including speculation bait. _What routing keeps of Atlas-M's quality at what cost, and whether Open-70B's weaker declining of speculation shows up inside persona traffic._

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

- **Resolves the issue, clear steps, escalation, citing the source and calling the research tool only when needed** Same ordering as the safety behaviours: Atlas-M within a few points of Atlas-L, a cheap model lowest; they add no new seam. Full tables in the appendix.
- **Cost per task** Every task keeps the same cost order; the overall cost carries the trade.
- **Stays in character in persona small talk** Open-70B leads it, which only reinforces the persona seam; full table in the appendix.
- **Failure examples** The evidence holds none for any arm.
- **Hypothetical floors on each behaviour read per task** A floor read where a behaviour is not tested has no figure, so every option is undecided outside that one task; the explore control states the reading that matters.

### Method

- **Cases** Each arm ran 1190 cases across five tasks, none lost or failed; no repeats are stated.
- **Traffic shares** Billing 42%, persona small talk 24%, troubleshooting 19%, research questions 15%; speculation bait carries 0%, so it enters no pooled figure and is read as a probe.
- **Overall quality** A judged ordinal score; the judge was calibrated on 200 human ratings with 86% agreement. Its model is not stated. Pooled quality is the traffic-weighted mean of the task scores, with an interval of about plus or minus 0.6925.
- **Safety and task behaviours** Pass or fail per trial, judged, each in the one task that tests it; nothing states whose judge scored them or how it was checked.
- **Intervals** At the level each table prints, uncorrected for the number of comparisons. With no control and no paired contrasts, a difference between two arms combines each arm's own spread as if they ran different cases, which reads wider than a paired interval would.
- **Cost** Dollars per thousand conversations, measured per task and overall. No monthly volume was declared, so no monthly figure is stated.
- **Latency** Median and p95 seconds per task; no budget was declared.
- **Not compared** No production or control arm; no gate, floor, budget or exchange rate was declared before or after the run; no routed mix was run.
- **Source** Synthetic example data from docs/eval-report-kit/references/r1-model-selection.json, run 2026-10-21.

### Review findings left open

- **Should fix** · Claim 1: The Note column disagrees with the Value column without explanation: $15.00 against $14.76 for Atlas-M → Atlas-L, and $4.940 against $4.943 for Brisk-S → Atlas-M. A reader of the lead claim can't tell whether the upper edge is $14.76 or $15. Drop the Note column or label what it is, and print the values and intervals at one precision ($4.94, $4.15 to $6.11, $14.76, $11.09 to $22.05), matching the prose.
- **Should fix** · The page: 'Confidence: High, if the caveats below aren't material' is followed straight away by a caveat that says it is 'Material to any call' (nothing declared). A reader may take High as confidence in a pick. Say what High covers, for example 'High in the measured price bands; the choice among them is unresolved until a price or floor is declared.'
- **Should fix** · The page: 'Open-70B is the best buy at no price' (decision block, The trade) and 'Open-70B at no price' (explore table) can be read as 'best if the price is zero'. The snapshot shows the opposite: Open-70B is never on the frontier, because a point bought from Brisk-S to Open-70B costs $6.96, more than the $4.58 a point from Open-70B on to Atlas-M. Write 'Open-70B is the best buy at no price per point' as 'Open-70B is never the best buy, whatever a point is worth' or similar.
- **Should fix** · Claim 3: The Brisk-S 81% and Open-70B 89% declines come from speculation bait, which carries 0% of traffic. They are probe rates, not rates on live conversations. Only Method says this, but the lean's '+17 points' and the explore floor both rely on it. Add a short clause to the claim or the lean saying that speculation bait is a probe with no traffic share, so a reader setting a floor knows what the rate means.
- **Should fix** · The page: 'The lean holds for any price per point inside Atlas-M's band' is true by definition and adds nothing. State the lean's actual condition instead: it holds only if the owner values a point of overall quality between $4.94 and $14.76 per thousand conversations, and the upper edge could be anywhere from $11.09 to $22.05.

_Written by claude-opus-5-5, served as claude-opus-5-5, for $1.70, with analyst skill b70be6bcfa68, on 2026-10-28._
