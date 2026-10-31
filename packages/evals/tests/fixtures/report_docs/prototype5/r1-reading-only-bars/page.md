Choice · candidates compared · Concierge chat assistant · 4 arms · 5 tasks · Run 2026-10-21 · Synthetic example data

_2 review findings open, listed at the end._

# No option wins outright: each step up in overall quality, from Brisk-S to Atlas-M to Atlas-L, costs more per point, and nothing declared says which price is worth paying.

Atlas-L scores higher than Atlas-M on overall quality in every live task, at 2.4 times its cost; Atlas-M holds the safety-type conduct checks that the cheaper models miss. A declared price per point, or a quality floor per task, would turn this into a decision.

**Confidence: High, if the caveats below aren't material.**

- **Nothing declares what is good enough.** Material to the pick whatever the figures say: no gate, floor or price per point was declared, so which option to run is the concierge owner's choice; the measured trade itself does not rest on it.
- **No arm is the control.** Only if pairing would reorder the options: every gap here is read unpaired, which is wider than a paired reading where models share cases. Pairing would likely narrow the price bands rather than reverse a step, but it could settle whether Open-70B joins the cheap end of the frontier.
- **The pick is the best of several candidates.** Only if a re-run on fresh cases pulled Atlas-L's lead over Atlas-M down far enough to move the price band of that step; the lead is clear of zero in each live task.
- **A judge is not described.** Only if the unnamed judge of the conduct checks is more lenient on the Atlas models than on Brisk-S or Open-70B; our lean toward Atlas-M rests on those checks, while overall quality comes from a calibrated judge.

**The trade.** Brisk-S is the cheapest at $6.01 per 1,000 conversations; Atlas-M costs 5.2 times as much and scores higher on overall quality; Atlas-L costs 2.4 times as much as Atlas-M and scores higher again. Open-70B is off the frontier on point estimates only: its gain over Brisk-S (-0.1893 to +1.769 points) is not told apart from zero.

**Where the answer flips.** Brisk-S is the pick while a point of overall quality is worth less than $4.15 to $6.11 per 1,000 conversations; Atlas-M above that band and below $11.09 to $22.05; Atlas-L above it. Inside the first band Open-70B is also a contender, and inside either band this run does not settle the pick. The bands treat cost as exact.

**Our lean.** Our lean, not a recommendation: Atlas-M, for a point priced between the bands. It is not told apart from Atlas-L on any conduct check, while Brisk-S and Open-70B fall behind it on confirming account changes, unsupported claims, resolving issues and declining to speculate.

**Watch.** Cost carries no interval, no count of trials that reported it, and no named provider, so the cost order holds for this run's unknown routing. The conduct checks come from a judge the campaign does not name. Speculation bait carries no traffic: its rates are a test of behaviour under bait, not what live traffic meets.

**What would settle it.** The concierge owner declares a price per point of overall quality, or a floor on it in every task, saying whether speculation bait counts; either picks one option from the figures already measured, unless the price lands inside a band.

| Option | Role | Configuration | Cost per 1,000 conversations |
|---|---|---|---|
| A · Atlas-L | Candidate | Atlas-L | $75.05 |
| B · Atlas-M | Candidate | Atlas-M | $31.37 |
| C · Brisk-S | Candidate | Brisk-S | $6.01 |
| D · Open-70B | Candidate | Open-70B | $11.51 |

## A point of overall quality costs more on the step from Atlas-M to Atlas-L than on the step from Brisk-S to Atlas-M, and the price bands of the two steps do not overlap while cost is read as exact.

_Caveats: No arm is the control._

| Row | Value | 95% unpaired interval |
|---|---|---|
| Brisk-S → Atlas-M | $4.94 | $4.15 to $6.11 |
| Atlas-M → Atlas-L | $14.76 | $11.09 to $22.05 |

Each step's price per point per 1,000 conversations: where the best option moves as a point's price rises.

Price per point per 1,000 conversations along the frontier, banded by the quality gain's unpaired interval only. Pooled overall quality reads 84.7 for Brisk-S, 85.49 for Open-70B, 89.83 for Atlas-M and 92.79 for Atlas-L, so the steps gain +5.13 (+4.151 to +6.109) and +2.96 (+1.981 to +3.939) points for $25 and $44 more. Cost carries no interval and is priced under this run's unnamed provider routing: the bands would meet only if the Atlas-M to Atlas-L cost gap shrank, or the Brisk-S to Atlas-M gap grew, until $6.11 and $11.09 touched.

### Atlas-L scores higher than Atlas-M on overall quality in every live task, by most in troubleshooting (5 points); on speculation bait, which carries no traffic, the two are not told apart.

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

Atlas-L less Atlas-M on overall quality, one row per task with unpaired intervals and no pooled row; pooled over the live tasks the gap is +2.96 (+1.981 to +3.939).

### Atlas-M is not told apart from Atlas-L on any conduct check; the cheaper models fall behind it on the safety-type checks, with Open-70B confirming before account changes 17 points less often, but not on every check.

_Caveats: A judge is not described._

| Panel | Row | Value | 95% uncorrected interval |
|---|---|---|---|
| Troubleshooting (multi-turn) · Confirms before account changes | A · Atlas-L | 98% | 96.0% to 100.0% |
| Troubleshooting (multi-turn) · Confirms before account changes | B · Atlas-M | 98% | 96.0% to 100.0% |
| Troubleshooting (multi-turn) · Confirms before account changes | C · Brisk-S | 88% | 83.3% to 92.7% |
| Troubleshooting (multi-turn) · Confirms before account changes | D · Open-70B | 81% | 75.3% to 86.7% |
| Troubleshooting (multi-turn) · Resolves the issue | A · Atlas-L | 91% | 86.8% to 95.2% |
| Troubleshooting (multi-turn) · Resolves the issue | B · Atlas-M | 90% | 85.6% to 94.4% |
| Troubleshooting (multi-turn) · Resolves the issue | C · Brisk-S | 74% | 67.6% to 80.4% |
| Troubleshooting (multi-turn) · Resolves the issue | D · Open-70B | 82% | 76.4% to 87.6% |
| Research-needed questions · No unsupported claims | A · Atlas-L | 95% | 92.2% to 97.8% |
| Research-needed questions · No unsupported claims | B · Atlas-M | 94% | 91.0% to 97.0% |
| Research-needed questions · No unsupported claims | C · Brisk-S | 84% | 79.4% to 88.6% |
| Research-needed questions · No unsupported claims | D · Open-70B | 86% | 81.6% to 90.4% |
| Speculation bait · Declines to speculate | A · Atlas-L | 99% | 97.4% to 100.0% |
| Speculation bait · Declines to speculate | B · Atlas-M | 98% | 95.8% to 100.0% |
| Speculation bait · Declines to speculate | C · Brisk-S | 81% | 74.7% to 87.3% |
| Speculation bait · Declines to speculate | D · Open-70B | 89% | 84.0% to 94.0% |

The checks where both cheaper models fall behind Atlas-M, each in the task that tests it, with intervals; speculation bait carries no traffic share.

| Panel | Row | Value | 95% uncorrected interval |
|---|---|---|---|
| Troubleshooting (multi-turn) · Gives clear steps | A · Atlas-L | 96% | 93.1% to 98.9% |
| Troubleshooting (multi-turn) · Gives clear steps | B · Atlas-M | 95% | 91.8% to 98.2% |
| Troubleshooting (multi-turn) · Gives clear steps | C · Brisk-S | 90% | 85.6% to 94.4% |
| Troubleshooting (multi-turn) · Gives clear steps | D · Open-70B | 93% | 89.3% to 96.7% |
| Troubleshooting (multi-turn) · Escalates when stuck | A · Atlas-L | 94% | 90.5% to 97.5% |
| Troubleshooting (multi-turn) · Escalates when stuck | B · Atlas-M | 95% | 91.8% to 98.2% |
| Troubleshooting (multi-turn) · Escalates when stuck | C · Brisk-S | 85% | 79.8% to 90.2% |
| Troubleshooting (multi-turn) · Escalates when stuck | D · Open-70B | 90% | 85.6% to 94.4% |
| Persona small talk · Stays in character | A · Atlas-L | 95% | 92.1% to 97.9% |
| Persona small talk · Stays in character | B · Atlas-M | 95% | 92.1% to 97.9% |
| Persona small talk · Stays in character | C · Brisk-S | 86% | 81.4% to 90.6% |
| Persona small talk · Stays in character | D · Open-70B | 97% | 94.7% to 99.3% |
| Research-needed questions · Calls research tool only when required | A · Atlas-L | 97% | 94.8% to 99.2% |
| Research-needed questions · Calls research tool only when required | B · Atlas-M | 95% | 92.2% to 97.8% |
| Research-needed questions · Calls research tool only when required | C · Brisk-S | 83% | 78.2% to 87.8% |
| Research-needed questions · Calls research tool only when required | D · Open-70B | 88% | 83.9% to 92.1% |
| Research-needed questions · Cites the retrieved source | A · Atlas-L | 96% | 93.5% to 98.5% |
| Research-needed questions · Cites the retrieved source | B · Atlas-M | 94% | 91.0% to 97.0% |
| Research-needed questions · Cites the retrieved source | C · Brisk-S | 89% | 85.0% to 93.0% |
| Research-needed questions · Cites the retrieved source | D · Open-70B | 85% | 80.5% to 89.5% |

The other checks: Open-70B is not told apart from Atlas-M on staying in character (-1.7 to +5.7), clear steps or escalating; Brisk-S on clear steps (-10.4 to +0.4) or citing the source, where its gap's interval ends at zero.

#### Atlas-L is also the slowest in every task: on research-needed questions its p95 latency is 4.86 s, against 2.84 s for Atlas-M.

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

Latency from median to p95, by task; no latency budget was declared.

## What you could change

| What you could change | What it decides | What happens as it moves |
|---|---|---|
| Price of one point of overall quality, per 1,000 conversations | Which frontier option is the best buy | On point estimates: Brisk-S below $4.15 to $6.11; Atlas-M from there to $11.09 to $22.05; Atlas-L above. Within the first band Open-70B could be the buy: a point bought from it on to Atlas-M costs $3.73 to $5.91. |
| A floor on overall quality that every task must clear, read on point estimates | The cheapest option that clears it everywhere | Counting speculation bait: Brisk-S up to 79; Open-70B up to 80; Atlas-M up to 88; Atlas-L up to 90; above that none. Live tasks only: the same, except Atlas-L holds up to 91. |

## Next steps

- The concierge owner declares a price per point of overall quality per 1,000 conversations, or a floor on it in every task, and says whether speculation bait counts toward the floor. _Which option to run, from the figures already measured, unless the price lands inside a band._
- State whether the models ran the same cases, and if so report every comparison paired. _How much narrower the price bands are, and whether Open-70B's gain over Brisk-S clears zero, which decides whether Open-70B joins the frontier._
- Name the judge of the conduct checks and calibrate it against human ratings, as the overall-quality judge was. _Whether Brisk-S and Open-70B really fall behind on the safety-type checks, which our lean rests on._
- Record each arm's provider routing, how many trials reported a cost, and a monthly volume of conversations. _Whether the cost order is the models' or this routing's, how far the price bands move with cost, and what each step costs a month._
- Run a routed mix as a measured option: Atlas-L on troubleshooting, where its lead is largest, and a cheaper model on persona small talk, where Open-70B reads close to Atlas-L. _Whether routing by task buys most of Atlas-L's quality at less of its cost, so it can be priced like the others._

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

- **Cost in each task** The cost order of the models is the same in every task; the appendix holds the figures.
- **The exact cost gap at which the two price bands would meet** No figure in the campaign states it, so the page gives the cost gaps and the band ends that would have to touch instead of a typed value.
- **Open-70B reading close to Atlas-L on overall quality in persona small talk** A per-task lead for a routed mix nobody ran; named as a next step, not priced.
- **Failure examples** The snapshot keeps none.

### Method

- **Cases** 1,190 cases per model across the tasks listed in the appendix; repeats are not stated. No model lost or failed a trial.
- **Overall quality** An ordinal score from a judge calibrated on 200 human ratings, agreeing 86%. Each model's overall figure is pooled over the four live tasks, weighted by traffic share; speculation bait carries 0% of traffic and is left out of it.
- **Conduct checks** Pass or fail checks, each read in the one task that tests it; the campaign does not say who judged them or how they were calibrated. Confirming account changes, resolving issues, unsupported claims and declining to speculate are called the safety-type checks here. _Caveats: A judge is not described._
- **Intervals** At the 95% level, uncorrected. With no control, every gap between two models combines each model's own interval as if they ran different cases, which reads wider than a paired interval. _Caveats: No arm is the control._
- **Cost** Cost per 1,000 conversations as measured, read as exact; no provider is named for any model, no count of trials reporting a cost is stated, and no monthly volume is declared.
- **Price per point** The cost gap between two neighbouring frontier options over their gain in pooled overall quality; its band comes from the gain's unpaired interval alone. Open-70B is left off the frontier on point estimates and named as a contender where its interval allows.
- **Latency** Median and p95 per task; no latency budget was declared.
- **What was declared** No control, gate, floor, margin or price per point; four models compared as equals. _Caveats: Nothing declares what is good enough; The pick is the best of several candidates._
- **Source** Synthetic example data from docs/eval-report-kit/references/r1-model-selection.json, run 2026-10-21.

### Review findings left open

- **Should fix** · Claim 3: The first block's caption says it holds the checks where both cheaper models fall behind Atlas-M. The second block's caption, 'The other checks', implies the cheaper models hold up on the rest. But on 'Calls research tool only when required', both fall behind: Brisk-S 83% (78.2% to 87.8%) and Open-70B 88% (83.9% to 92.1%), against 95% for Atlas-M (92.2% to 97.8%). The unpaired gaps are about -12 (-17.4 to -6.6) and -7 (-12.0 to -2.0). Either move that check into the first block, or reword the first caption as 'the safety-type checks, where both cheaper models fall behind' and add to the second caption that both also fall behind on calling the research tool. The lean only gets stronger, but as written a reader would misread where Brisk-S and Open-70B hold up.
- **Should fix** · Claim 2: The caption's first line says each row is Atlas-L less Atlas-M 'overall and in each place both hold it', but the block has no overall row. The last caption line says so and gives the pooled gap (+2.96, +1.981 to +3.939) in text. Drop 'overall and' from the first line so the caption describes only the task rows the block draws.

_Written by claude-opus-5-5, served as claude-opus-5-5, for $3.61, with analyst skill db31bf62aa68, on 2026-10-31._
