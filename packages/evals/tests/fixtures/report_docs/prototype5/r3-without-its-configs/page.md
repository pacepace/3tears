Choice · candidates compared · Turn router: complexity × domain · 9 arms · Run 2026-10-24 · Synthetic example data

# Use Brisk-S · Few-shot (12 ex.) (E): of nine router configurations it alone clears the balanced-accuracy gate, every cell floor and the latency budget.

It clears the 90% balanced-accuracy gate by 0.7 points, interval 90.4–91%. Nano-XS · Few-shot (12 ex.) (H) saves $1,143 a month at this run's unnamed routing (volume read as turns) but fails Complex × News and is borderline on balanced accuracy.

**Confidence: High, if the caveats below aren't material.**

- **Latency budget (p95) is cleared on a point estimate.** The high applies to the balanced-accuracy and cell-floor readings; the latency gate stays open until Brisk-S · Few-shot (12 ex.) (E)'s p95 is measured in production with an interval. It becomes material if that p95 runs more than 66 ms above today's 184 ms: then no configuration passes every gate and the choice becomes a trade between latency and accuracy.
- **The pick is the best of several candidates.** Only if a fresh run reads Brisk-S · Few-shot (12 ex.) (E)'s balanced accuracy more than 0.7 points lower: the one pick out of nine reads high by chance, though today's interval sits wholly above the gate.
- **No arm is the control.** Only for claims that one configuration beats another: with no control, each gap is read from the two configurations' own intervals combined as independent, which reads wider than a paired comparison would. The gates read each configuration on its own figures, so the call does not rest on it.

**Recommendation.** Ship Brisk-S · Few-shot (12 ex.) (E) as the turn router's classifier now: balanced accuracy 90.7%, p95 latency 184 ms, its weakest cell 82.9%, $1,200 a month at this run's unnamed routing with the volume read as turns.

**Why not the others.** Every Atlas-M prompt and Brisk-S · Rubric + reasoning (F) miss the 250 ms latency budget, though most are more accurate. Nano-XS · Few-shot (12 ex.) (H) is the cheapest near-miss but scores 76.7% on Complex × News, below the 80% floor with its interval clear of it, and its balanced accuracy is borderline. The other zero-shot and rubric prompts fall short on balanced accuracy or a cell floor.

**Watch.** The weakest margin is the recommendation's own balanced-accuracy gate: 90.7% against 90% for Brisk-S · Few-shot (12 ex.) (E), clear on its interval but picked out of nine. The latency budget is cleared on a point estimate only, 66 ms under it. Complex × News is the nearest cell, 2.9 points above the floor.

**Next.** Ship now and measure production p95 with an interval in parallel; also give Nano-XS · Few-shot (12 ex.) (H) Complex × News examples and re-run it, the one result that could move the call to a cheaper option.

| Option | Role | Configuration | Cost per 1,000,000 turns |
|---|---|---|---|
| A · Atlas-M · Zero-shot | Candidate | Atlas-M · Prompt Zero-shot | $180.00 |
| B · Atlas-M · Few-shot (12 ex.) | Candidate | Atlas-M · Prompt Few-shot (12 ex.) | $260.00 |
| C · Atlas-M · Rubric + reasoning | Candidate | Atlas-M · Prompt Rubric + reasoning | $610.00 |
| D · Brisk-S · Zero-shot | Candidate | Brisk-S · Prompt Zero-shot | $28.00 |
| E · Brisk-S · Few-shot (12 ex.) | Candidate (recommended) | Brisk-S · Prompt Few-shot (12 ex.) | $40.00 |
| F · Brisk-S · Rubric + reasoning | Candidate | Brisk-S · Prompt Rubric + reasoning | $95.00 |
| G · Nano-XS · Zero-shot | Candidate | Nano-XS · Prompt Zero-shot | $1.30 |
| H · Nano-XS · Few-shot (12 ex.) | Candidate | Nano-XS · Prompt Few-shot (12 ex.) | $1.90 |
| I · Nano-XS · Rubric + reasoning | Candidate | Nano-XS · Prompt Rubric + reasoning | $4.60 |

## Brisk-S · Few-shot (12 ex.) (E) is the only configuration that passes all three declared gates; each of the others fails at least one.

| Gate | Atlas-M · Zero-shot | Atlas-M · Few-shot (12 ex.) | Atlas-M · Rubric + reasoning | Brisk-S · Zero-shot | Brisk-S · Few-shot (12 ex.) | Brisk-S · Rubric + reasoning | Nano-XS · Zero-shot | Nano-XS · Few-shot (12 ex.) | Nano-XS · Rubric + reasoning |
|---|---|---|---|---|---|---|---|---|---|
| Overall accuracy (at least 90%) | Fails: Overall accuracy 89.8% < 90% | Passes | Passes | Fails: Overall accuracy 86.0% < 90% | Passes | Passes | Fails: Overall accuracy 78.2% < 90% | Borderline: Overall accuracy 90.1%, derived interval 89.8–90.4% | Fails: Overall accuracy 83.9% < 90% |
| Cell floor (at least 80% in every cell) | Passes | Passes | Passes | Fails: Cell floor 77.9% < 80% in Complex × News | Passes | Passes | Fails: Cell floor 77.8% < 80% in Medium × Science; Cell floor 73.5% < 80% in Medium × News; Cell floor 76.1% < 80% in Complex × Science; Cell floor 77.6% < 80% in Complex × Conversational; Cell floor 65.1% < 80% in Complex × News | Fails: Cell floor 76.7% < 80% in Complex × News | Fails: Cell floor 79.5% < 80% in Medium × News; Cell floor 70.9% < 80% in Complex × News |
| Latency budget (p95) (p95 at most 250 ms) | Fails: p95 420 ms > 250 ms | Fails: p95 483 ms > 250 ms | Fails: p95 1302 ms > 250 ms | Passes | Passes | Fails: p95 496 ms > 250 ms | Passes | Passes | Passes |

Each configuration against the balanced-accuracy gate (declared under the name Overall accuracy), the cell floor and the latency budget, with the figure that fails it.

### Every configuration with higher balanced accuracy than Brisk-S · Few-shot (12 ex.) (E) misses the latency budget; the most accurate, Atlas-M · Rubric + reasoning (C), runs at 1302 ms.

| Point | Latency (p95) | Balanced accuracy |
|---|---|---|
| Atlas-M · Zero-shot | 420 ms | 89.8% |
| Atlas-M · Few-shot (12 ex.) | 483 ms | 92.1% |
| Atlas-M · Rubric + reasoning | 1302 ms | 93.2% |
| Brisk-S · Zero-shot | 160 ms | 86.0% |
| Brisk-S · Few-shot (12 ex.) | 184 ms | 90.7% |
| Brisk-S · Rubric + reasoning | 496 ms | 91.5% |
| Nano-XS · Zero-shot | 70 ms | 78.2% |
| Nano-XS · Few-shot (12 ex.) | 80 ms | 90.1% |
| Nano-XS · Rubric + reasoning | 217 ms | 83.9% |
| Latency budget (p95) gate, at most 250 ms | 250 ms |  |
| Overall accuracy gate, at least 90% |  | 90.0% |

The Latency (p95) axis is on a log scale: its values span more than ten times.

Balanced accuracy against p95 latency, latency on a log scale; the horizontal line is the balanced-accuracy gate, printed under its declared name, and the region inside both gates is shaded.

### Nano-XS · Few-shot (12 ex.) (H) trails Brisk-S · Few-shot (12 ex.) (E) by only 0.6 points on balanced accuracy and stays close in most cells, but trails by 6.2 points on Complex × News, which puts it below the cell floor.

| Row | Value | 95% unpaired interval |
|---|---|---|
| Overall | +0.6 | +0.2 to +1 |

Each row is Brisk-S · Few-shot (12 ex.) less Nano-XS · Few-shot (12 ex.) on Balanced accuracy, overall and in each place both hold it.

The interval is independent: each arm's own spread (the standard error derived from its target-matrix cells, the figure being their plain mean: the root of the cells' squared standard errors over their count, the cells read as independent, at the cells' stated level; a size read from it counts cases at today's mix of cells), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same.

Brisk-S few-shot minus Nano-XS few-shot on balanced accuracy, overall, with its unpaired interval.

| Place | First arm | Second arm | Difference | Interval | Level | Read as |
|---|---|---|---|---|---|---|
| Simple × Science | 95.1% | 96.2% | -1.1 | -2.2 to -0.03699 | 95.0% | independent: each arm's own spread (its stated half-width), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same |
| Simple × Conversational | 97.1% | 98.2% | -1.1 | -1.5 to -0.7 | 95.0% | independent: each arm's own spread (its stated half-width), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same |
| Simple × News | 94.0% | 94.5% | -0.5 | -1.5 to +0.5 | 95.0% | independent: each arm's own spread (its stated half-width), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same |
| Medium × Science | 89.9% | 89.4% | +0.5 | -0.8 to +1.8 | 95.0% | independent: each arm's own spread (its stated half-width), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same |
| Medium × Conversational | 91.6% | 91.9% | -0.3 | -1.1 to +0.5 | 95.0% | independent: each arm's own spread (its stated half-width), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same |
| Medium × News | 87.5% | 86.0% | +1.5 | +0.2 to +2.8 | 95.0% | independent: each arm's own spread (its stated half-width), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same |
| Complex × Science | 88.5% | 88.2% | +0.3 | -1.3 to +1.9 | 95.0% | independent: each arm's own spread (its stated half-width), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same |
| Complex × Conversational | 89.4% | 89.8% | -0.4 | -2.3 to +1.5 | 95.0% | independent: each arm's own spread (its stated half-width), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same |
| Complex × News | 82.9% | 76.7% | +6.2 | +4.4 to +8 | 95.0% | independent: each arm's own spread (its stated half-width), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same |

Each cell's accuracy for Brisk-S few-shot (first) and Nano-XS few-shot (second), their gap and its unpaired interval; the Simple × Science bound prints unrounded and reads as just below zero.

### As this run's unnamed provider routing served them, Brisk-S · Few-shot (12 ex.) (E) costs $40.00 per 1,000,000 turns against $1.90 per 1,000,000 turns for Nano-XS · Few-shot (12 ex.) (H); reading the declared volume as turns, that is $1,143 more a month, the price of meeting the Complex × News floor.

| Row | Value |
|---|---|
| A · Atlas-M · Zero-shot | $180.00 |
| B · Atlas-M · Few-shot (12 ex.) | $260.00 |
| C · Atlas-M · Rubric + reasoning | $610.00 |
| D · Brisk-S · Zero-shot | $28.00 |
| E · Brisk-S · Few-shot (12 ex.) | $40.00 |
| F · Brisk-S · Rubric + reasoning | $95.00 |
| G · Nano-XS · Zero-shot | $1.30 |
| H · Nano-XS · Few-shot (12 ex.) | $1.90 |
| I · Nano-XS · Rubric + reasoning | $4.60 |

The bars are drawn on a log scale: their values span more than ten times, so each step is a multiple.

Cost per million turns for each configuration, on a log scale, as this run's provider routing served it; the routing is not named.

#### On Nano-XS, few-shot examples are what lift balanced accuracy: 11.9 points over zero-shot, with the rubric prompt 6.2 points below few-shot. On Brisk-S, the rubric edges few-shot by 0.8 points but takes p95 to 496 ms.

| Group | Row | Value | 95% unpaired interval |
|---|---|---|---|
| model | Brisk-S · Zero-shot → Nano-XS | -7.8 | -8.4 to -7.2 |
| model | Brisk-S · Few-shot (12 ex.) → Nano-XS | -0.6 | -1.0 to -0.2 |
| model | Brisk-S · Rubric + reasoning → Nano-XS | -7.6 | -8.1 to -7.1 |
| prompt | Brisk-S · Zero-shot → Few-shot (12 ex.) | +4.7 | +4.2 to +5.2 |
| prompt | Brisk-S · Zero-shot → Rubric + reasoning | +5.5 | +5.0 to +6.0 |
| prompt | Brisk-S · Few-shot (12 ex.) → Rubric + reasoning | +0.8 | +0.4 to +1.2 |
| prompt | Nano-XS · Zero-shot → Few-shot (12 ex.) | +11.9 | +11.4 to +12.4 |
| prompt | Nano-XS · Zero-shot → Rubric + reasoning | +5.7 | +5.1 to +6.3 |
| prompt | Nano-XS · Few-shot (12 ex.) → Rubric + reasoning | -6.2 | -6.7 to -5.7 |
| model effect, across prompt | (Brisk-S · Few-shot (12 ex.) → Nano-XS) less (Brisk-S · Zero-shot → Nano-XS) | +7.2 | +6.5 to +7.9 |
| model effect, across prompt | (Brisk-S · Rubric + reasoning → Nano-XS) less (Brisk-S · Zero-shot → Nano-XS) | +0.2 | -0.5 to +0.9 |
| model effect, across prompt | (Brisk-S · Rubric + reasoning → Nano-XS) less (Brisk-S · Few-shot (12 ex.) → Nano-XS) | -7.0 | -7.7 to -6.3 |
| prompt effect, across model | (Nano-XS · Few-shot (12 ex.) → Rubric + reasoning) less (Brisk-S · Few-shot (12 ex.) → Rubric + reasoning) | -7.0 | -7.7 to -6.3 |

Each effect is one setting moved between two arms that differ in it alone, on Balanced accuracy; each interaction is the same move's effect in one context less its effect in another, the two contexts differing in one other setting. Descriptive: read as two arms' own spreads, never a test.

Change in balanced accuracy from moving one setting, Brisk-S and Nano-XS only, with unpaired intervals; the model rows include the overall Brisk-S to Nano-XS gap at each prompt, and interactions follow the effects.

#### On balanced accuracy and all nine cell floors, Brisk-S · Few-shot (12 ex.) (E) clears its line on today's interval, Complex × News nearest at 2.9 points; the latency budget is cleared on a point estimate only, 66 ms under it.

| Arm | Against | Today's figure | Line | Distance toward passing | Settles | Sizes count | Needed, one-sided | Needed, two-sided | Today | A fresh run reads the other side | Level |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E · Brisk-S · Few-shot (12 ex.) | Overall accuracy | 90.7% | 90% | +0.7 | clears | cases | 6,682 | 9,487 | 48,210 | 0.0004972% | 95% one-sided |
| E · Brisk-S · Few-shot (12 ex.) | Cell floor · Simple × Science | 95.1% | 80% | +15.1 | clears | cases | 7 | 9 | 3,120 | < 0.000001% | 95% one-sided |
| E · Brisk-S · Few-shot (12 ex.) | Cell floor · Simple × Conversational | 97.1% | 80% | +17.1 | clears | cases | 4 | 5 | 14,880 | 0% | 95% one-sided |
| E · Brisk-S · Few-shot (12 ex.) | Cell floor · Simple × News | 94% | 80% | +14.0 | clears | cases | 8 | 12 | 4,410 | 0% | 95% one-sided |
| E · Brisk-S · Few-shot (12 ex.) | Cell floor · Medium × Science | 89.9% | 80% | +9.9 | clears | cases | 25 | 36 | 4,260 | < 0.000001% | 95% one-sided |
| E · Brisk-S · Few-shot (12 ex.) | Cell floor · Medium × Conversational | 91.6% | 80% | +11.6 | clears | cases | 14 | 20 | 7,350 | < 0.000001% | 95% one-sided |
| E · Brisk-S · Few-shot (12 ex.) | Cell floor · Medium × News | 87.5% | 80% | +7.5 | clears | cases | 53 | 75 | 5,190 | < 0.000001% | 95% one-sided |
| E · Brisk-S · Few-shot (12 ex.) | Cell floor · Complex × Science | 88.5% | 80% | +8.5 | clears | cases | 41 | 58 | 3,440 | < 0.000001% | 95% one-sided |
| E · Brisk-S · Few-shot (12 ex.) | Cell floor · Complex × Conversational | 89.4% | 80% | +9.4 | clears | cases | 31 | 44 | 1,960 | < 0.000001% | 95% one-sided |
| E · Brisk-S · Few-shot (12 ex.) | Cell floor · Complex × News | 82.9% | 80% | +2.9 | clears | cases | 435 | 617 | 3,600 | 0.0001087% | 95% one-sided |
| E · Brisk-S · Few-shot (12 ex.) | Latency budget (p95) | 184 ms | 250 ms | — | — | — | — | — | — | — | — |

In each row, "A fresh run reads the other side" is how often a fresh run of today's size would read the figure on the other side of its line, if the truth is today's estimate (95% one-sided); it plans a run and tests nothing.

Each line Brisk-S few-shot is read against: its distance, the cases a fresh run needs to settle it, and how often a run of today's size reads the other side; the latency row carries no interval, so nothing is sized for it.

## What you could change

| What you could change | What it decides | What happens as it moves |
|---|---|---|
| Cell floor | Which configuration survives | Read on point estimates: at a floor at or below 76.7%, Nano-XS · Few-shot (12 ex.) (H) becomes the pick, though its balanced accuracy stays borderline; above 82.9%, no configuration passes; in between the call is Brisk-S · Few-shot (12 ex.) (E). |
| Balanced-accuracy gate | Whether anything passes | Above 90.7% nothing passes; at any lower gate the call stays Brisk-S · Few-shot (12 ex.) (E), since Nano-XS · Few-shot (12 ex.) (H) still misses its cell floor. |
| Latency budget | How many configurations pass | Below 184 ms nothing passes. A looser budget admits more accurate, costlier configurations, and on cost the call stays Brisk-S · Few-shot (12 ex.) (E). |

## Next steps

- Add Complex × News examples to Nano-XS · Few-shot (12 ex.) (H)'s few-shot prompt and re-run every cell. _Whether it clears the cell floor. Its balanced accuracy also sits on the gate (a run of today's size reads it on the wrong side 26.5% of the time, and settling it four times in five takes 955,843 cases), so only a fix that lifts it settles the switch._
- Measure Brisk-S · Few-shot (12 ex.) (E)'s p95 latency on production hardware and load, with an interval. _Whether the latency budget holds outside the test rig, closing the one gate read on a point estimate._
- Run Brisk-S · Few-shot (12 ex.) (E) in shadow on fresh traffic, at least 19,383 turns, sampled by cell to keep today's per-cell proportions, so the rarer complex cells and Complex × News (today 3,600 cases) keep their share. _Whether its balanced accuracy holds above the gate once the pick-of-many optimism is gone; that size, at today's mix of cells, settles it four times in five if today's estimate is the truth._
- Have the provider routing and the unit of the monthly volume declared. _Whether the monthly cost gap is stated in the right units and at the routing production will use._

## Full results

### The arms

Each arm's full configuration, once. The rest of the page names an arm by its letter tag and name.

| Arm | Role | Configuration | Served model |
|---|---|---|---|
| A · Atlas-M · Zero-shot | Candidate | Atlas-M · Prompt Zero-shot | Atlas-M |
| B · Atlas-M · Few-shot (12 ex.) | Candidate | Atlas-M · Prompt Few-shot (12 ex.) | Atlas-M |
| C · Atlas-M · Rubric + reasoning | Candidate | Atlas-M · Prompt Rubric + reasoning | Atlas-M |
| D · Brisk-S · Zero-shot | Candidate | Brisk-S · Prompt Zero-shot | Brisk-S |
| E · Brisk-S · Few-shot (12 ex.) | Candidate | Brisk-S · Prompt Few-shot (12 ex.) | Brisk-S |
| F · Brisk-S · Rubric + reasoning | Candidate | Brisk-S · Prompt Rubric + reasoning | Brisk-S |
| G · Nano-XS · Zero-shot | Candidate | Nano-XS · Prompt Zero-shot | Nano-XS |
| H · Nano-XS · Few-shot (12 ex.) | Candidate | Nano-XS · Prompt Few-shot (12 ex.) | Nano-XS |
| I · Nano-XS · Rubric + reasoning | Candidate | Nano-XS · Prompt Rubric + reasoning | Nano-XS |

### Every arm by measure

| Arm | Balanced accuracy | Traffic-weighted accuracy | Macro F1 | Cost per million turns | Latency |
|---|---|---|---|---|---|
| A · Atlas-M · Zero-shot | 89.8% | 91.2% | 88.7% | $180.00 | 420 ms (p95) |
| B · Atlas-M · Few-shot (12 ex.) | 92.1% | 93.5% | 91% | $260.00 | 483 ms (p95) |
| C · Atlas-M · Rubric + reasoning | 93.2% | 94.6% | 92.1% | $610.00 | 1302 ms (p95) |
| D · Brisk-S · Zero-shot | 86% | 87.5% | 84.9% | $28.00 | 160 ms (p95) |
| E · Brisk-S · Few-shot (12 ex.) | 90.7% | 92.2% | 89.6% | $40.00 | 184 ms (p95) |
| F · Brisk-S · Rubric + reasoning | 91.5% | 93.1% | 90.4% | $95.00 | 496 ms (p95) |
| G · Nano-XS · Zero-shot | 78.2% | 80.2% | 77.1% | $1.30 | 70 ms (p95) |
| H · Nano-XS · Few-shot (12 ex.) | 90.1% | 92% | 89% | $1.90 | 80 ms (p95) |
| I · Nano-XS · Rubric + reasoning | 83.9% | 86% | 82.8% | $4.60 | 217 ms (p95) |

### Every gate, wherever it is read

| Where | Criterion | Line | A · Atlas-M · Zero-shot | B · Atlas-M · Few-shot (12 ex.) | C · Atlas-M · Rubric + reasoning | D · Brisk-S · Zero-shot | E · Brisk-S · Few-shot (12 ex.) | F · Brisk-S · Rubric + reasoning | G · Nano-XS · Zero-shot | H · Nano-XS · Few-shot (12 ex.) | I · Nano-XS · Rubric + reasoning |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Overall | Overall accuracy | at least 90% | Fails (89.8%) | Passes (92.1%) | Passes (93.2%) | Fails (86%) | Passes (90.7%) | Passes (91.5%) | Fails (78.2%) | Borderline (90.1%) | Fails (83.9%) |
| Simple × Science | Cell floor | at least 80% | Passes (93.8%) | Passes (96.3%) | Passes (97.5%) | Passes (90.1%) | Passes (95.1%) | Passes (95.5%) | Passes (84.4%) | Passes (96.2%) | Passes (89.5%) |
| Simple × Conversational | Cell floor | at least 80% | Passes (95.8%) | Passes (97.8%) | Passes (99.2%) | Passes (92.1%) | Passes (97.1%) | Passes (97.9%) | Passes (86.4%) | Passes (98.2%) | Passes (92.5%) |
| Simple × News | Cell floor | at least 80% | Passes (93.1%) | Passes (94.9%) | Passes (96.1%) | Passes (89.4%) | Passes (94%) | Passes (94.5%) | Passes (82.4%) | Passes (94.5%) | Passes (88.2%) |
| Medium × Science | Cell floor | at least 80% | Passes (88.8%) | Passes (90.8%) | Passes (92%) | Passes (85.1%) | Passes (89.9%) | Passes (91%) | Fails (77.8%) | Passes (89.4%) | Passes (83.6%) |
| Medium × Conversational | Cell floor | at least 80% | Passes (90.8%) | Passes (93.3%) | Passes (94.4%) | Passes (87.2%) | Passes (91.6%) | Passes (92.9%) | Borderline (80.5%) | Passes (91.9%) | Passes (85.7%) |
| Medium × News | Cell floor | at least 80% | Passes (86.9%) | Passes (89.3%) | Passes (90%) | Passes (83.1%) | Passes (87.5%) | Passes (88.2%) | Fails (73.5%) | Passes (86%) | Fails (79.5%) |
| Complex × Science | Cell floor | at least 80% | Passes (87.5%) | Passes (89.7%) | Passes (91.2%) | Passes (84.2%) | Passes (88.5%) | Passes (89.4%) | Fails (76.1%) | Passes (88.2%) | Passes (82%) |
| Complex × Conversational | Cell floor | at least 80% | Passes (88.8%) | Passes (91.5%) | Passes (92.2%) | Passes (85.1%) | Passes (89.4%) | Passes (90.9%) | Fails (77.6%) | Passes (89.8%) | Passes (83.3%) |
| Complex × News | Cell floor | at least 80% | Passes (82.9%) | Passes (85.5%) | Passes (86%) | Fails (77.9%) | Passes (82.9%) | Passes (83.6%) | Fails (65.1%) | Fails (76.7%) | Fails (70.9%) |
| Overall | Latency budget (p95) | at most 250 ms | Fails (420 ms) | Fails (483 ms) | Fails (1302 ms) | Passes (160 ms) | Passes (184 ms) | Fails (496 ms) | Passes (70 ms) | Passes (80 ms) | Passes (217 ms) |

### Every cell

| Cell | Cases | A · Atlas-M · Zero-shot | B · Atlas-M · Few-shot (12 ex.) | C · Atlas-M · Rubric + reasoning | D · Brisk-S · Zero-shot | E · Brisk-S · Few-shot (12 ex.) | F · Brisk-S · Rubric + reasoning | G · Nano-XS · Zero-shot | H · Nano-XS · Few-shot (12 ex.) | I · Nano-XS · Rubric + reasoning |
|---|---|---|---|---|---|---|---|---|---|---|
| Simple × Science | 3120 | 93.8% | 96.3% | 97.5% | 90.1% | 95.1% | 95.5% | 84.4% | 96.2% | 89.5% |
| Simple × Conversational | 14880 | 95.8% | 97.8% | 99.2% | 92.1% | 97.1% | 97.9% | 86.4% | 98.2% | 92.5% |
| Simple × News | 4410 | 93.1% | 94.9% | 96.1% | 89.4% | 94% | 94.5% | 82.4% | 94.5% | 88.2% |
| Medium × Science | 4260 | 88.8% | 90.8% | 92% | 85.1% | 89.9% | 91% | 77.8% | 89.4% | 83.6% |
| Medium × Conversational | 7350 | 90.8% | 93.3% | 94.4% | 87.2% | 91.6% | 92.9% | 80.5% | 91.9% | 85.7% |
| Medium × News | 5190 | 86.9% | 89.3% | 90% | 83.1% | 87.5% | 88.2% | 73.5% | 86% | 79.5% |
| Complex × Science | 3440 | 87.5% | 89.7% | 91.2% | 84.2% | 88.5% | 89.4% | 76.1% | 88.2% | 82% |
| Complex × Conversational | 1960 | 88.8% | 91.5% | 92.2% | 85.1% | 89.4% | 90.9% | 77.6% | 89.8% | 83.3% |
| Complex × News | 3600 | 82.9% | 85.5% | 86% | 77.9% | 82.9% | 83.6% | 65.1% | 76.7% | 70.9% |

### Left out, and why

- **Traffic-weighted accuracy and macro F1 from the body** Not gated, and neither reorders the survivor against the near-misses; both stay in the full tables.
- **Confusion tables by complexity and by domain** They describe which labels get confused; no gate reads them and they do not change the call.
- **Atlas-M rows of the prompt-and-model chart** Atlas-M is out on latency at every prompt; the Brisk-S and Nano-XS rows, model rows included, carry the point.
- **Atlas-M cell figures from the body** They pass every cell floor but miss the latency budget, so the cells do not bear on them; they stay in the full cell table.
- **Failure examples** The snapshot keeps none.

### Method

- **Cases** 48,210 labelled turns per configuration, spread across nine complexity × domain cells; none lost or failed. Repeats are not stated.
- **Pass** A turn passes when the predicted label matches the gold label. Balanced accuracy is the plain mean of the cells' accuracies, so every cell weighs the same whatever its traffic; it is not the traffic-weighted accuracy in the full tables.
- **Gates** All three were declared before the run: balanced accuracy at least 90% (declared under the name Overall accuracy, which the charts and tables print), accuracy at least 80% in every cell, p95 latency at most 250 ms. A gate is read on the point estimate; an interval crossing its line marks the option borderline.
- **Intervals** Cell accuracies carry stated half-widths at the 95% level. Balanced accuracy's interval is derived from its cells: the root of the cells' squared standard errors over their count, the cells read as independent. Gaps between configurations combine each one's own spread as independent, uncorrected for the number of comparisons.
- **Latency** p95 per configuration on the test setup, with no interval stated.
- **Cost** Cost per million turns as this run was served; the served model is named for each arm, the provider for none. Monthly figures read the declared volume of 30,000,000 as turns, which the input does not state.
- **Not compared** No control or production baseline.
- **Source** Synthetic example data from docs/eval-report-kit/references/r3-classifier.json, run 2026-10-24.

_Written by claude-opus-5-5, served as claude-opus-5-5, for $4.32, with analyst skill 2a7662d928b0, on 2026-10-29._
