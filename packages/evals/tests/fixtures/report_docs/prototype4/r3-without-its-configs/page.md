Choice · candidates compared · Turn router: complexity × domain · 9 arms · Run 2026-10-24 · Synthetic example data

# Route turns with Brisk-S and the few-shot prompt: it is the only one of the 9 configurations inside all three gates, at $1,200.00 a month.

It reaches 90.7% balanced accuracy, its whole interval above the gate, and holds every cell floor. What could change the call is latency: its p95 of 184 ms has no interval, so a production measurement above 250 ms would leave no configuration standing.

**Confidence: High, if the caveats below aren't material.**

- **Latency budget (p95) is cleared on a point estimate.** only if Brisk-S few-shot's true p95 is more than 66 ms slower than measured: then it misses the budget and no configuration passes every gate.
- **The pick is the best of several candidates.** only if a fresh run reads Brisk-S few-shot's balanced accuracy more than 0.7 points lower, or its Complex × News cell more than 2.9 points lower; as the one passer of 9, its figures read somewhat high.
- **No arm is the control.** only for claims that one configuration beats another; the call reads each configuration against fixed gates, not against a control.

**Recommendation.** Ship Brisk-S with the few-shot prompt on all turns. It clears overall balanced accuracy at 90.4–91%, every one of the nine cell floors, and the latency budget at 184 ms.

**Why not the others.** Every Atlas-M prompt and Brisk-S with the rubric prompt miss the latency budget (Brisk-S rubric at 496 ms). Nano-XS few-shot clears latency and, on its point estimate, overall accuracy, but misses the Complex × News cell at 76.7%, clear below the floor. The zero-shot prompts and Nano-XS rubric miss accuracy.

**Watch.** The thinnest margin is overall accuracy: 0.7 points above the gate, interval 90.4–91%, and above 90.7% nothing passes. Complex × News is 2.9 points above its floor; in noise the two are about equally safe, each more than twice its half-width. Latency headroom of 66 ms rests on a point estimate.

**Next.** Measure Brisk-S few-shot's p95 latency with an interval under production load; it is the one gate the call cannot yet show it clears beyond noise.

| Option | Role | Configuration | Cost per 1,000 turns |
|---|---|---|---|
| A · Atlas-M · Zero-shot | Candidate | Atlas-M · prompt Zero-shot | $0.18 |
| B · Atlas-M · Few-shot (12 ex.) | Candidate | Atlas-M · prompt Few-shot (12 ex.) | $0.26 |
| C · Atlas-M · Rubric + reasoning | Candidate | Atlas-M · prompt Rubric + reasoning | $0.61 |
| D · Brisk-S · Zero-shot | Candidate | Brisk-S · prompt Zero-shot | $0.0280 |
| E · Brisk-S · Few-shot (12 ex.) | Candidate (recommended) | Brisk-S · prompt Few-shot (12 ex.) | $0.0400 |
| F · Brisk-S · Rubric + reasoning | Candidate | Brisk-S · prompt Rubric + reasoning | $0.0950 |
| G · Nano-XS · Zero-shot | Candidate | Nano-XS · prompt Zero-shot | $0.00130 |
| H · Nano-XS · Few-shot (12 ex.) | Candidate | Nano-XS · prompt Few-shot (12 ex.) | $0.00190 |
| I · Nano-XS · Rubric + reasoning | Candidate | Nano-XS · prompt Rubric + reasoning | $0.00460 |

## Two configurations clear both the latency budget and the overall accuracy gate, Brisk-S and Nano-XS with the few-shot prompt; every Atlas-M prompt and the Brisk-S rubric fall out on latency, the rest on accuracy.

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

Balanced accuracy against p95 latency (log axis), with the gates at 90% and 250 ms; Nano-XS few-shot's overall interval, 89.8–90.4%, straddles its gate.

### The cell floor separates the two: at Complex × News, the weakest cell for both, Brisk-S few-shot clears the floor by 2.9 points, while Nano-XS few-shot fails there and nowhere else.

| Panel |  | Science | Conversational | News |
|---|---|---|---|---|
| E · Brisk-S · Few-shot (12 ex.) | Simple | 95.1% (94.3% to 95.9%; n = 3120) | 97.1% (96.8% to 97.4%; n = 14880) | 94.0% (93.3% to 94.7%; n = 4410) |
| E · Brisk-S · Few-shot (12 ex.) | Medium | 89.9% (89.0% to 90.8%; n = 4260) | 91.6% (91.0% to 92.2%; n = 7350) | 87.5% (86.6% to 88.4%; n = 5190) |
| E · Brisk-S · Few-shot (12 ex.) | Complex | 88.5% (87.4% to 89.6%; n = 3440) | 89.4% (88.0% to 90.8%; n = 1960) | 82.9% (81.7% to 84.1%; n = 3600) |
| H · Nano-XS · Few-shot (12 ex.) | Simple | 96.2% (95.5% to 96.9%; n = 3120) | 98.2% (98.0% to 98.4%; n = 14880) | 94.5% (93.8% to 95.2%; n = 4410) |
| H · Nano-XS · Few-shot (12 ex.) | Medium | 89.4% (88.5% to 90.3%; n = 4260) | 91.9% (91.3% to 92.5%; n = 7350) | 86.0% (85.1% to 86.9%; n = 5190) |
| H · Nano-XS · Few-shot (12 ex.) | Complex | 88.2% (87.1% to 89.3%; n = 3440) | 89.8% (88.5% to 91.1%; n = 1960) | 76.7% (75.3% to 78.1%; n = 3600) |

Accuracy in each complexity × domain cell, with its interval and count; the floor is 80%.

### On Brisk-S, few-shot examples lift balanced accuracy by +4.7 points and the rubric prompt adds only +0.8 more while breaking the latency budget; the prompt's effect depends on the model, and the rubric costs Nano-XS accuracy against few-shot.

|  | Zero-shot | Few-shot (12 ex.) | Rubric + reasoning |
|---|---|---|---|
| Atlas-M | 89.8% · $0.18 per 1,000 turns · 420 ms · Fails | 92.1% · $0.26 per 1,000 turns · 483 ms · Fails | 93.2% · $0.61 per 1,000 turns · 1302 ms · Fails |
| Brisk-S | 86.0% · $0.028 per 1,000 turns · 160 ms · Fails | 90.7% · $0.04 per 1,000 turns · 184 ms · Passes | 91.5% · $0.095 per 1,000 turns · 496 ms · Fails |
| Nano-XS | 78.2% · $0.0013 per 1,000 turns · 70 ms · Fails | 90.1% · $0.0019 per 1,000 turns · 80 ms · Fails | 83.9% · $0.0046 per 1,000 turns · 217 ms · Fails |

Each model × prompt tile: balanced accuracy, cost per 1,000 turns, p95 latency and whether it passes every gate. Few-shot gains more on Nano-XS than on Atlas-M by +9.6 points.

## What you could change

| What you could change | What it decides | What happens as it moves |
|---|---|---|
| Cell floor | Which configurations may route turns | Below Nano-XS few-shot's Complex × News accuracy of 76.7%, Nano-XS few-shot becomes the pick; above Brisk-S few-shot's 82.9%, nothing passes. |
| Overall accuracy gate | The minimum balanced accuracy | The call holds for any gate up to 90.7%; above it, nothing passes. |
| Latency budget | The p95 ceiling | The call holds for any budget at or above 184 ms; below it, nothing passes. |

## Next steps

- Measure Brisk-S few-shot's p95 latency, with an interval, under production load. _Whether the pick clears the latency budget beyond noise; if not, no configuration passes and the budget or the model needs a decision._
- Add Complex × News examples to the Nano-XS few-shot prompt and re-run that cell and the overall set. _Whether Nano-XS few-shot becomes eligible; the monthly cost gap to Brisk-S few-shot is +$1,143.00. Nano-XS must also clear the overall gate on its interval, today 89.8–90.4%._
- Run Brisk-S few-shot on a fresh holdout of about 19383 turns, the size at which four runs in five confirm the overall clearance if today's figure is the truth. _Whether the pick's thin overall clearance survives outside the set it was chosen on._

## Full results

### The arms

Each arm's full configuration, once. The rest of the page names an arm by its letter tag and name.

| Arm | Role | Configuration | Served model |
|---|---|---|---|
| A · Atlas-M · Zero-shot | Candidate | Atlas-M · prompt Zero-shot | Atlas-M |
| B · Atlas-M · Few-shot (12 ex.) | Candidate | Atlas-M · prompt Few-shot (12 ex.) | Atlas-M |
| C · Atlas-M · Rubric + reasoning | Candidate | Atlas-M · prompt Rubric + reasoning | Atlas-M |
| D · Brisk-S · Zero-shot | Candidate | Brisk-S · prompt Zero-shot | Brisk-S |
| E · Brisk-S · Few-shot (12 ex.) | Candidate | Brisk-S · prompt Few-shot (12 ex.) | Brisk-S |
| F · Brisk-S · Rubric + reasoning | Candidate | Brisk-S · prompt Rubric + reasoning | Brisk-S |
| G · Nano-XS · Zero-shot | Candidate | Nano-XS · prompt Zero-shot | Nano-XS |
| H · Nano-XS · Few-shot (12 ex.) | Candidate | Nano-XS · prompt Few-shot (12 ex.) | Nano-XS |
| I · Nano-XS · Rubric + reasoning | Candidate | Nano-XS · prompt Rubric + reasoning | Nano-XS |

### Every arm by measure

| Arm | Balanced accuracy | Traffic-weighted accuracy | Macro F1 | Cost per million turns | Latency |
|---|---|---|---|---|---|
| A · Atlas-M · Zero-shot | 89.8% | 91.2% | 88.7% | $180 | 420 ms (p95) |
| B · Atlas-M · Few-shot (12 ex.) | 92.1% | 93.5% | 91% | $260 | 483 ms (p95) |
| C · Atlas-M · Rubric + reasoning | 93.2% | 94.6% | 92.1% | $610 | 1302 ms (p95) |
| D · Brisk-S · Zero-shot | 86% | 87.5% | 84.9% | $28 | 160 ms (p95) |
| E · Brisk-S · Few-shot (12 ex.) | 90.7% | 92.2% | 89.6% | $40 | 184 ms (p95) |
| F · Brisk-S · Rubric + reasoning | 91.5% | 93.1% | 90.4% | $95 | 496 ms (p95) |
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

- **Traffic-weighted accuracy and macro F1** Neither is gated, and both rank the configurations as balanced accuracy does; they don't change the call.
- **Confusion tables on complexity and domain** Each axis on its own reads well for both few-shot configurations; they don't locate the Complex × News loss, which is a joint error.
- **The full lever and interaction table** Prompt and model interact clearly (c3 cites the largest interaction), but the call rests on gates read per configuration, not on averaged effects; the full table stays in the appendix.
- **Atlas-M rubric's top accuracy** It is out on latency by a wide margin, so its accuracy doesn't bear on the call.
- **Failure examples** No failed turns were kept for review in this campaign.

### Method

- **Cases** 48210 labelled turns per configuration, each run once, spread over nine complexity × domain cells.
- **Accuracy** The share of a cell's turns given the correct label, with an interval per cell at the level the cells state, uncorrected.
- **Balanced accuracy** The plain mean of the nine cell accuracies; its interval is derived from the cells' intervals, read as independent.
- **Gates** Balanced accuracy at least 90%, each cell at least 80%, p95 latency at most 250 ms; all three declared before the run, each read on the point estimate.
- **Latency** p95 only, with no interval stated. _Caveats: Latency budget (p95) is cleared on a point estimate._
- **Cost** Dollars per million turns, priced at 30000000 turns a month.
- **Comparisons** No control; prompt and model effects compare two configurations' own intervals as if independent, so they read wide.
- **Source** Synthetic example data from docs/eval-report-kit/references/r3-classifier.json

### Review findings left open

- **Should fix** · The page: The latency budget row in What you could change says the call holds for any budget at or above 184 ms. That is right: the sweep keeps Brisk-S few-shot as the pick at every higher budget, because it is the cheapest passer. But at 483 ms Atlas-M few-shot also passes all gates (92.1%, $0.26 per 1,000 turns), and at 496 ms so does Brisk-S rubric (91.5%, $0.095). From those points the headline's 'only one inside all three gates' no longer holds, and the pick rests on cost. Add a clause such as: 'from 483 ms, Atlas-M few-shot also passes at +1.4 points and 6.5× the cost; the pick then rests on cost.' A reader weighing a looser budget for more accuracy would then see the trade.
- **Should fix** · The page: In the second next step, 'the monthly cost gap to Brisk-S few-shot is +$1,143.00' doesn't say which way the gap runs. Nano-XS few-shot costs $57 a month against $1,200 for Brisk-S few-shot. Say plainly that Nano-XS few-shot would save about $1,143 a month. The cell-floor row could also say that below 76.7%, Nano-XS few-shot would become the pick only on a point estimate: its overall interval, 89.8–90.4%, straddles the 90% gate.

_Written by claude-opus-5-5, served as claude-opus-5-5, for $2.17, with analyst skill b70be6bcfa68, on 2026-10-28._
