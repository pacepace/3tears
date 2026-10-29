Choice · candidates compared · Turn router: complexity × domain · 9 arms · Run 2026-10-24 · Synthetic example data

# Route turns with Brisk-S · Few-shot (12 ex.) (E): it is the only one of nine configurations inside the balanced-accuracy gate, every cell floor and the latency budget.

Its balanced accuracy of 90.7% clears the 90% gate with its whole interval. The call would change if a fresh run read it below that gate, or if Nano-XS · Few-shot (12 ex.) (H), cheaper only as this run served it, lifted Complex × News above the cell floor and held the balanced-accuracy gate on a re-run.

**Confidence: High, if the caveats below aren't material.**

- **The pick is the best of several candidates.** Matters only if a run on fresh cases reads Brisk-S · Few-shot (12 ex.) (E) below the 90% balanced-accuracy gate. It was the one configuration of nine to clear every line, so its estimate reads high; its clearance is 0.7 points with the interval clear of the line, and selection could pull the truth down by an amount no figure states.
- **Latency budget (p95) is cleared on a point estimate.** Matters only if the p95 latency varies from run to run by as much as its 66 ms clearance under the 250 ms budget. No spread is stated, so this pass is unsized until the repeated latency runs report; the high confidence applies to the accuracy and cell-floor findings, whose intervals clear their lines, and not to the latency pass.
- **Provider and serving stack unnamed.** Matters only if production serves these models differently from this run: a slower stack could eat the latency clearance, and another provider could reorder the costs. The accuracy findings, and so the gate on balanced accuracy and every cell floor, do not depend on it.
- **No arm is the control.** Matters only for the comparisons between configurations, which are read unpaired and so err wide; the call reads each option against the declared gate lines, not against another arm, and no production router was measured beside them.

**Recommendation.** Use Brisk-S · Few-shot (12 ex.) (E): balanced accuracy 90.7% (90.4–91%), every cell at or above 80%, p95 latency 184 ms, at $1,200 a month at the declared volume as this run served it.

**Why not the others.** Every Atlas-M prompt and Brisk-S · Rubric + reasoning (F) miss the 250 ms latency budget (Brisk-S · Rubric + reasoning (F) reads 496 ms). Nano-XS · Few-shot (12 ex.) (H) misses the cell floor in Complex × News at 76.7%, its interval clear below the line, so more cases won't rescue it. The zero-shot Brisk-S and the other Nano-XS prompts miss the balanced-accuracy gate.

**Watch.** The weakest margin is the balanced-accuracy gate for Brisk-S · Few-shot (12 ex.) (E): 90.7% against 90%, a clearance of 0.7 points, on an option picked as the best of nine. Complex × News, at 82.9% against the 80% floor, is the thinnest cell; the latency clearance of 66 ms has no interval and was measured on an unnamed serving stack.

**Next.** Ship Brisk-S · Few-shot (12 ex.) (E) and, in parallel, re-run it on fresh cases and measure its p95 over repeated runs on the stack it will use; separately, fix news handling in Nano-XS · Few-shot (12 ex.) (H), the one change that could replace it.

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

## Brisk-S · Few-shot (12 ex.) (E) is the only configuration that lands inside both the balanced-accuracy gate and the latency budget and also holds every cell floor.

_Caveats: Provider and serving stack unnamed._

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

Balanced accuracy against p95 latency for every configuration, with the 90% gate and the 250 ms budget drawn; latency on a log axis. Nano-XS · Few-shot (12 ex.) (H) also lands in the box but misses a cell floor.

### News is the weak column for both few-shot models; Complex × News is where Brisk-S · Few-shot (12 ex.) (E) comes nearest the floor, at 82.9% with its interval clear of 80%.

| Panel |  | Science | Conversational | News |
|---|---|---|---|---|
| E · Brisk-S · Few-shot (12 ex.) | Simple | 95.1% (94.3% to 95.9%; n = 3120) | 97.1% (96.8% to 97.4%; n = 14880) | 94.0% (93.3% to 94.7%; n = 4410) |
| E · Brisk-S · Few-shot (12 ex.) | Medium | 89.9% (89.0% to 90.8%; n = 4260) | 91.6% (91.0% to 92.2%; n = 7350) | 87.5% (86.6% to 88.4%; n = 5190) |
| E · Brisk-S · Few-shot (12 ex.) | Complex | 88.5% (87.4% to 89.6%; n = 3440) | 89.4% (88.0% to 90.8%; n = 1960) | 82.9% (81.7% to 84.1%; n = 3600) |
| H · Nano-XS · Few-shot (12 ex.) | Simple | 96.2% (95.5% to 96.9%; n = 3120) | 98.2% (98.0% to 98.4%; n = 14880) | 94.5% (93.8% to 95.2%; n = 4410) |
| H · Nano-XS · Few-shot (12 ex.) | Medium | 89.4% (88.5% to 90.3%; n = 4260) | 91.9% (91.3% to 92.5%; n = 7350) | 86.0% (85.1% to 86.9%; n = 5190) |
| H · Nano-XS · Few-shot (12 ex.) | Complex | 88.2% (87.1% to 89.3%; n = 3440) | 89.8% (88.5% to 91.1%; n = 1960) | 76.7% (75.3% to 78.1%; n = 3600) |

Accuracy in each complexity × domain cell, one panel per configuration, with each cell's interval and case count; cells below the 80% floor are outlined.

### As this run served them, by providers the input does not name, Brisk-S · Few-shot (12 ex.) (E) costs $1,143 a month more than Nano-XS · Few-shot (12 ex.) (H), a gap another routing could change. What it buys is +0.6 points of balanced accuracy, chiefly from Complex × News, where it leads by +6.2; elsewhere the cell gaps are small or favour the cheaper model.

_Caveats: Provider and serving stack unnamed._

| Row | Value | 95% unpaired interval |
|---|---|---|
| Overall | +0.6 | +0.2 to +1 |

Each row is Brisk-S · Few-shot (12 ex.) less Nano-XS · Few-shot (12 ex.) on Balanced accuracy, overall and in each place both hold it.

The interval is independent: each arm's own spread (the standard error derived from its target-matrix cells, the figure being their plain mean: the root of the cells' squared standard errors over their count, the cells read as independent, at the cells' stated level; a size read from it counts cases at today's mix of cells), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same.

Brisk-S few-shot minus Nano-XS few-shot on balanced accuracy overall, with its unpaired interval.

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

The same difference on accuracy in each cell, with unpaired intervals; the Simple × Science lower bound prints unrounded and sits just below zero.

#### Worked examples matter most for the smallest model: moving from zero-shot to few-shot lifts Nano-XS by +11.9 points of balanced accuracy against +4.7 for Brisk-S, and rubric + reasoning on top of few-shot adds only +0.8 for Brisk-S.

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

What changing the model or the prompt alone does to balanced accuracy for Brisk-S and Nano-XS, and how each effect differs across the other setting, with unpaired intervals; descriptive.

## What you could change

| What you could change | What it decides | What happens as it moves |
|---|---|---|
| The cell floor | Which configurations survive, through Complex × News | At or below 76.7%, Nano-XS · Few-shot (12 ex.) (H) would pass the cell floor; it would take over only on cost as this run served it, and only if a re-run holds its balanced accuracy of 90.1% above the gate, which today's design reads the wrong way 26.5% of the time. From there up to 82.9%, Brisk-S · Few-shot (12 ex.) (E) holds; above that, nothing passes. Both ends are point estimates of cells whose intervals run 1.4% and 1.2% either side. |
| The balanced-accuracy gate | Whether the pick passes | Raised above 90.7%, nothing passes; lowered, the pick holds, because every option it would admit misses latency or a cell floor. |
| The latency budget | Whether the pick passes, and whether a more accurate option enters | Tightened below 184 ms, nothing passes. Loosened to 483 ms or more, Atlas-M few-shot (and from 496 ms, Brisk-S rubric) passes every gate with 1.4 (and 0.8) more points of balanced accuracy; the pick then holds only if its lower cost, which belongs to this run's unnamed routing, is preferred to those points. |

## Next steps

- Re-run Brisk-S · Few-shot (12 ex.) (E) on a fresh case set of at least 19,383 cases. _Whether its 0.7-point clearance of the balanced-accuracy gate survives being picked as the best of nine; that size settles it four times in five if today's estimate holds._
- Measure p95 latency for Brisk-S · Few-shot (12 ex.) (E) over repeated runs on the serving stack and provider it will use in production. _Whether its 66 ms latency clearance is more than run-to-run noise and holds off this run's unnamed stack; until then the latency pass is unsized._
- Add news examples to the Nano-XS few-shot prompt and re-run Complex × News, where it must gain at least 3.3 points, and the overall set, where it sits at 90.1%. _Whether a configuration that, as this run served both, cost 21 times less than the pick can pass every gate; its balanced-accuracy clearance reads the wrong way 26.5% of the time today, so the overall set needs re-reading too._
- Declare the unit the monthly volume counts and the provider each model is served by. _What the monthly cost figures and the cost gap to Nano-XS mean under production routing; the call does not rest on them._

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

- **Macro F1 and traffic-weighted accuracy** No gate reads them and they order the configurations as balanced accuracy does; the appendix holds them.
- **Cost of every configuration** Only one option survives the gates, so cost decides nothing; the one cost that frames a choice, against Nano-XS few-shot, is in the body with its routing caveat.
- **Atlas-M cell figures and lever rows** Every Atlas-M prompt misses the latency budget by a wide margin; its cell accuracies can't change the call.
- **Cells of the other configurations** Each is out on a gate already shown; the full cell table is in the appendix.
- **A lower confidence form over the latency pass** Weighed and kept high-if-immaterial: the accuracy and cell-floor findings that make the pick unique have intervals clear of their lines; the latency pass is scoped out of the high and sized by a next step instead.
- **A rounded Simple × Science bound in the per-cell comparison** The cell figures print as the data hold them, so that one bound shows extra digits; dropping the row would hide a cell that favours the cheaper model, so it is kept and its caption says the bound sits just below zero.

### Method

- **Cases** 48,210 labelled turns per configuration, each run once, spread over nine cells crossing complexity with domain; Complex × News holds 3,600.
- **Correct** A turn is correct when the predicted class matches its label. Accuracy is read per cell; balanced accuracy is the plain mean of the cell accuracies; traffic-weighted accuracy weights cells by traffic.
- **Intervals** Cell accuracies carry their stated intervals, at the level the cells state (95%). The interval on balanced accuracy is derived from the cells, reading them as independent; no multiplicity correction. Comparisons between configurations are unpaired, which errs wide where they ran the same turns.
- **Gates** The three gates were all declared before the run: balanced accuracy at least 90%, each cell's accuracy at least 80%, and p95 latency at most 250 ms. The gate tables label the first 'Overall accuracy'; it reads balanced accuracy, not traffic-weighted accuracy. Each gate is read on its figure; latency states only a p95, with no spread.
- **Cost and latency** Cost per million turns and p95 latency as this run served each model; no provider or serving stack is named for any configuration, so the routing behind both is unknown. Monthly figures use the declared volume of 30,000,000, which the input states without a unit; they read it as turns.
- **Not compared** No production router or control was run, and the run kept no failure examples or confusion counts, so what the Complex × News errors look like is not shown.
- **Source** Turn router classifier sweep, run 2026-10-24; Synthetic example data from docs/eval-report-kit/references/r3-classifier.json.

_Written by claude-opus-5-5, served as claude-opus-5-5, for $4.20, with analyst skill ed7e2c60b5ad, on 2026-10-29._
