Choice · candidates compared · Turn router: complexity × domain · 9 arms · Run 2026-10-24 · Synthetic example data

# Use Brisk-S · Few-shot (12 ex.) (E) for the turn router: of the nine configurations it is the only one inside every gate, at $1,200 a month.

It clears balanced accuracy by 0.7 points and its weakest cell, Complex × News, by 2.9 points, both beyond their intervals. The call moves only if its p95 latency, read without an interval, proves slower in production, or if the cell floor were lowered enough to admit the cheaper Nano-XS · Few-shot (12 ex.) (H).

**Confidence: High, if the caveats below aren't material.**

- **Latency budget (p95) is cleared on a point estimate.** Only if the recommended configuration's true p95 latency sits more than 66 ms above today's 184 ms: then nothing passes every gate and the latency budget itself needs a decision.
- **The pick is the best of several candidates.** Only if a fresh run reads the recommended configuration's balanced accuracy more than 0.7 points lower than today. Its interval clears the gate today, but being the pick of nine pulls its estimate up.
- **No arm is the control.** Only for the comparisons between configurations (the cell gaps to Nano-XS · Few-shot (12 ex.) (H), the effects of model and prompt), which are read unpaired and so wider than paired; the call reads each configuration against declared lines and does not rest on them.

**Recommendation.** Ship Brisk-S · Few-shot (12 ex.) (E) on all turns. Balanced accuracy is 90.7% (90.4% to 91%, derived), every cell holds its floor and p95 latency is 184 ms.

**Why not the others.** Nano-XS · Few-shot (12 ex.) (H) would cost $57 a month but misses the floor in Complex × News at 76.7%, clear of its interval. Atlas-M · Few-shot (12 ex.) (B), Atlas-M · Rubric + reasoning (C) and Brisk-S · Rubric + reasoning (F) are more accurate but blow the 250 ms latency budget. The rest miss the accuracy gate.

**Watch.** The weakest margin is Brisk-S · Few-shot (12 ex.) (E) on Overall accuracy: 90.7% against 90%, 0.7 points. Its interval clears the line, but the pick of nine reads high by chance. The latency headroom of 66 ms is relatively larger yet matters as much, because no interval says how firm it is.

**Next.** Ship now; in parallel, measure the recommended configuration's p95 latency on production traffic with an interval, and re-run it on fresh cases to confirm the accuracy clearance.

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

## Brisk-S · Few-shot (12 ex.) (E) is the only configuration that passes every gate; each other configuration fails at least one.

| Gate | Atlas-M · Zero-shot | Atlas-M · Few-shot (12 ex.) | Atlas-M · Rubric + reasoning | Brisk-S · Zero-shot | Brisk-S · Few-shot (12 ex.) | Brisk-S · Rubric + reasoning | Nano-XS · Zero-shot | Nano-XS · Few-shot (12 ex.) | Nano-XS · Rubric + reasoning |
|---|---|---|---|---|---|---|---|---|---|
| Overall accuracy (at least 90%) | Fails: Overall accuracy 89.8% < 90% | Passes | Passes | Fails: Overall accuracy 86.0% < 90% | Passes | Passes | Fails: Overall accuracy 78.2% < 90% | Borderline: Overall accuracy 90.1%, derived interval 89.8–90.4% | Fails: Overall accuracy 83.9% < 90% |
| Cell floor (at least 80% in every cell) | Passes | Passes | Passes | Fails: Cell floor 77.9% < 80% in Complex × News | Passes | Passes | Fails: Cell floor 77.8% < 80% in Medium × Science; Cell floor 73.5% < 80% in Medium × News; Cell floor 76.1% < 80% in Complex × Science; Cell floor 77.6% < 80% in Complex × Conversational; Cell floor 65.1% < 80% in Complex × News | Fails: Cell floor 76.7% < 80% in Complex × News | Fails: Cell floor 79.5% < 80% in Medium × News; Cell floor 70.9% < 80% in Complex × News |
| Latency budget (p95) (p95 at most 250 ms) | Fails: p95 420 ms > 250 ms | Fails: p95 483 ms > 250 ms | Fails: p95 1302 ms > 250 ms | Passes | Passes | Fails: p95 496 ms > 250 ms | Passes | Passes | Passes |

Each configuration against each gate, with the figure that binds where it fails or is borderline.

### The cheaper Nano-XS · Few-shot (12 ex.) (H) would save $1,143 a month if it passed, but it trails Brisk-S · Few-shot (12 ex.) (E) by 6.2 points in Complex × News, where it misses the floor.

| Row | Value | 95% unpaired interval |
|---|---|---|
| Simple × Science | -1.1 | -2.2 to -0.03699 |
| Simple × Conversational | -1.1 | -1.5 to -0.7 |
| Simple × News | -0.5 | -1.5 to +0.5 |
| Medium × Science | +0.5 | -0.8 to +1.8 |
| Medium × Conversational | -0.3 | -1.1 to +0.5 |
| Medium × News | +1.5 | +0.2 to +2.8 |
| Complex × Science | +0.3 | -1.3 to +1.9 |
| Complex × Conversational | -0.4 | -2.3 to +1.5 |
| Complex × News | +6.2 | +4.4 to +8 |

Each row is Brisk-S · Few-shot (12 ex.) less Nano-XS · Few-shot (12 ex.) on Accuracy, overall and in each place both hold it.

The interval is independent: each arm's own spread (its stated half-width), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same.

Accuracy, Brisk-S few-shot minus Nano-XS few-shot, in each cell, with unpaired intervals; positive rows favour Brisk-S few-shot.

### The most accurate configurations are out on latency alone: Atlas-M · Rubric + reasoning (C) reaches 93.2% balanced accuracy at a p95 of 1302 ms.

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

Balanced accuracy against p95 latency (log axis) for every configuration, with both gate lines. Nano-XS few-shot also sits inside both; the cell floor, not drawn here, removes it.

#### What a prompt buys depends on the model: few-shot examples lift Nano-XS by +11.9 points, while rubric + reasoning takes 6.2 points back from it yet adds accuracy for Atlas-M.

| Group | Row | Value | 95% unpaired interval |
|---|---|---|---|
| model | Atlas-M · Zero-shot → Nano-XS | -11.6 | -12.1 to -11.1 |
| model | Atlas-M · Few-shot (12 ex.) → Nano-XS | -2.0 | -2.4 to -1.6 |
| model | Atlas-M · Rubric + reasoning → Nano-XS | -9.3 | -9.8 to -8.8 |
| prompt | Atlas-M · Zero-shot → Few-shot (12 ex.) | +2.3 | +1.9 to +2.7 |
| prompt | Atlas-M · Zero-shot → Rubric + reasoning | +3.4 | +3.0 to +3.8 |
| prompt | Atlas-M · Few-shot (12 ex.) → Rubric + reasoning | +1.1 | +0.7 to +1.5 |
| prompt | Nano-XS · Zero-shot → Few-shot (12 ex.) | +11.9 | +11.4 to +12.4 |
| prompt | Nano-XS · Zero-shot → Rubric + reasoning | +5.7 | +5.1 to +6.3 |
| prompt | Nano-XS · Few-shot (12 ex.) → Rubric + reasoning | -6.2 | -6.7 to -5.7 |
| model effect, across prompt | (Atlas-M · Few-shot (12 ex.) → Nano-XS) less (Atlas-M · Zero-shot → Nano-XS) | +9.6 | +8.9 to +10.3 |
| model effect, across prompt | (Atlas-M · Rubric + reasoning → Nano-XS) less (Atlas-M · Zero-shot → Nano-XS) | +2.3 | +1.6 to +3.0 |
| model effect, across prompt | (Atlas-M · Rubric + reasoning → Nano-XS) less (Atlas-M · Few-shot (12 ex.) → Nano-XS) | -7.3 | -7.9 to -6.7 |
| prompt effect, across model | (Nano-XS · Few-shot (12 ex.) → Rubric + reasoning) less (Atlas-M · Few-shot (12 ex.) → Rubric + reasoning) | -7.3 | -7.9 to -6.7 |

Each effect is one setting moved between two arms that differ in it alone, on Balanced accuracy; each interaction is the same move's effect in one context less its effect in another, the two contexts differing in one other setting. Descriptive: read as two arms' own spreads, never a test.

Change in balanced accuracy from moving one setting, for Atlas-M and Nano-XS, and how each change differs across the other setting; unpaired intervals, descriptive.

#### Every accuracy clearance of Brisk-S · Few-shot (12 ex.) (E) is settled at today's size; the latency budget is the one line no interval reads.

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

Each line Brisk-S few-shot is read against: its distance, the cases that would settle it, and how often a fresh run of today's size would read the other side.

## What you could change

| What you could change | What it decides | What happens as it moves |
|---|---|---|
| The cell floor, declared at 80% | Whether the cheaper Nano-XS few-shot is admitted | Read on point estimates: at or below 76.7% Nano-XS few-shot becomes the call; up to 82.9% Brisk-S few-shot holds; above that nothing passes. |
| The latency budget, declared at 250 ms | Whether any configuration passes, and which | Below 184 ms nothing passes. Any looser budget admits slower, more accurate configurations, but Brisk-S few-shot stays the call: they cost more and nothing declared prices accuracy above the gate. |
| The gate on balanced accuracy, declared at 90% | Whether any configuration passes | Above 90.7% nothing passes; at any lower line Brisk-S few-shot stays the call. |

## Next steps

- Measure Brisk-S · Few-shot (12 ex.) (E)'s p95 latency on production traffic, repeated enough to put an interval on it. _Whether its 66 ms of headroom under the budget is real; if not, nothing passes every gate and the budget or the accuracy gate needs a decision._
- Re-run Brisk-S · Few-shot (12 ex.) (E) on 19,383 fresh cases, the size that settles its balanced-accuracy clearance four times in five. _Whether the clearance survives being the pick of nine configurations._
- Add Complex × News examples to Nano-XS · Few-shot (12 ex.) (H)'s prompt and re-run the full matrix at today's size. _Whether it clears that cell's floor and firms up its borderline balanced accuracy; only then is the saving of $1,143 a month on the table._

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

- **Traffic-weighted accuracy and macro F1** No gate reads them and the decision has no rule weighing them; they stay in the appendix.
- **Confusion tables for each configuration** They describe which labels get swapped; no gate reads them and they do not change the call.
- **Cost ranking among the configurations that fail a gate** A failed configuration is out before price; only Nano-XS few-shot is priced, as a labelled what-if.
- **Brisk-S rubric + reasoning's small accuracy gain over few-shot** It is out on latency, so the gain is unavailable.
- **Model and prompt effects for Brisk-S** They follow the same pattern as Atlas-M's; the full lever table is in the appendix.

### Method

- **Cases** Each configuration labelled 48,210 turns once, none lost; the matrix crosses complexity (Simple, Medium, Complex) with domain (Science, Conversational, News), and each cell holds the same count for every configuration.
- **Pass** A turn is correct when the predicted complexity and domain label matches the gold label. Accuracy is read per cell; balanced accuracy is the plain mean of the cells.
- **Intervals** Cell accuracy carries a stated half-width at 95%. Balanced accuracy's interval is derived from its cells (the root of their squared standard errors over their count), so it is called derived. Comparisons between configurations are unpaired and uncorrected for multiplicity.
- **Gates** None was set after the results: balanced accuracy at least 90%, accuracy at least 80% in every cell, p95 latency at most 250 ms. A figure on the passing side whose interval crosses its line passes and is called borderline.
- **Latency** One p95 per configuration, stated with no interval and no count of the requests behind it.
- **Cost** Measured cost per million routed turns. Monthly figures read the declared volume of 30,000,000 a month; the input does not say what that volume counts, and it is read here as turns.
- **Not compared** There is no production router or control, and no mix of configurations by cell was run.
- **Source** Synthetic example data from docs/eval-report-kit/references/r3-classifier.json.

### Review findings left open

- **Should fix** · The page: The same figure appears under two names. The gate is called 'Overall accuracy' (Watch: 'on Overall accuracy: 90.7% against 90%'; c1 and c5 rows). The measure is called 'Balanced accuracy' (recommendation, c3, appendix). The appendix also lists 'Traffic-weighted accuracy', 92.2% for the pick, and that is what most readers would take 'overall accuracy' to mean. Call the gate 'balanced-accuracy gate' or similar everywhere above the appendix.
- **Should fix** · The page: The lede says the call moves 'only if' p95 latency proves slower or the cell floor is lowered. The confidence line, Watch and the second next step all name a third way: a fresh run reads balanced accuracy more than 0.7 points lower because the pick was chosen from nine. Add that to the lede or drop 'only'.
- **Should fix** · Claim 5: c5 says every accuracy clearance is 'settled at today's size', and its fresh-run column (0.0005% for balanced accuracy) assumes the truth equals today's estimate. The best-of-nine caveat says that estimate reads high. Say in the sentence or caption that 'settled' does not account for selection, and point to the re-run in Next steps.
- **Should fix** · The page: The 'Left out' reason for the Brisk-S model and prompt effects says 'the full lever table is in the appendix', but the appendix has no lever table. Either add the Brisk-S rows to c4 or the appendix (Brisk-S zero-shot to few-shot is the step that takes the pick from failing to passing; Atlas-M to Brisk-S few-shot is -1.4, -1.8 to -1.0), or correct the reason.
- **Should fix** · Claim 2: The block note says each row is the difference 'overall and in each place', but the table has only the nine cells. Remove 'overall' or add the balanced-accuracy row (Nano-XS few-shot minus Brisk-S few-shot is -0.6, -1.0 to -0.2). Also round the Simple × Science upper bound, which prints as -0.03699.
- **Should fix** · Claim 2: The $1,143-a-month saving in c2 and the $1,200 in the headline both assume the declared 30,000,000-a-month volume counts turns, and only Method says so. Add a short qualifier where the monthly figures first appear.

_Written by claude-opus-5-5, served as claude-opus-5-5, for $2.03, with analyst skill 1677ccb31cd6, on 2026-10-29._
