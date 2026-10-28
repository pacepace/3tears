Choice · candidates compared · Turn router: complexity × domain · 9 arms · Run 2026-10-24 · Synthetic example data

# Use Brisk-S few-shot for the turn router: it is the only configuration inside the overall accuracy, cell floor and latency gates at once.

It clears overall accuracy with its whole interval and its weakest cell, complex news, by 2.9 points. The call would change if its p95 latency, measured only as a point estimate, ran over budget, or if the far cheaper Nano-XS few-shot were fixed on complex news.

**Confidence: High, if the caveats below aren't material.**

- **Latency budget (p95) is cleared on a point estimate.** Only if the true p95 of Brisk-S few-shot is over the budget: it reads 184 ms against 250 ms with no interval, so nothing says how much of the 66 ms clearance is noise. Accuracy, the other two gates, is measured with intervals clear of its lines.
- **The pick is the best of several candidates.** Only if the pick's thinnest clearance re-reads below its line on fresh cases: chosen from 9 configurations, its figures read somewhat high, but its overall interval (90.4–91%) and its complex-news cell both sit clear of their lines today.
- **No arm is the control.** Only for claims that one configuration beats another: those read each arm's own spread, unpaired and so wide. The call itself rests on each configuration against the declared gates, which needs no control.

**Recommendation.** Route turns with Brisk-S few-shot. It passes every declared gate, at $1,200.00 a month at the declared volume.

**Why not the others.** Nano-XS few-shot is cheaper but misses the complex-news floor at 76.7%, its whole interval below the line. Brisk-S rubric, Atlas-M few-shot and Atlas-M rubric are more accurate but break the latency budget; Brisk-S rubric reads 496 ms. Atlas-M zero-shot breaks the budget and misses overall accuracy too, and the remaining configurations miss overall accuracy.

**Watch.** Latency is the least-evidenced gate: a point estimate with no interval. Complex news is the thinnest accuracy clearance, though a fresh run of today's size would almost never read it below the floor (0.0001087%). Picked as the one of 9 that passes, its figures likely read a little high on a rerun.

**Next.** Re-measure the pick's p95 latency with an interval, then try to fix Nano-XS few-shot on complex news: if it clears the floor, a router at a fraction of the cost passes every gate.

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

## Brisk-S few-shot is the only configuration inside all three gates; its balanced accuracy of 90.7% clears the overall line with its whole interval.

_Caveats: Latency budget (p95) is cleared on a point estimate._

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

Balanced accuracy against p95 latency, latency on a log scale. Nano-XS few-shot also sits inside both lines here but fails the cell floor, which this view cannot show.

### Its thinnest clearance is complex news, the weakest cell for every configuration, where it reads 82.9% with its interval clear of the floor.

|  | Science | Conversational | News |
|---|---|---|---|
| Simple | 95.1% (94.3% to 95.9%; n = 3120) | 97.1% (96.8% to 97.4%; n = 14880) | 94.0% (93.3% to 94.7%; n = 4410) |
| Medium | 89.9% (89.0% to 90.8%; n = 4260) | 91.6% (91.0% to 92.2%; n = 7350) | 87.5% (86.6% to 88.4%; n = 5190) |
| Complex | 88.5% (87.4% to 89.6%; n = 3440) | 89.4% (88.0% to 90.8%; n = 1960) | 82.9% (81.7% to 84.1%; n = 3600) |

Brisk-S few-shot accuracy in each complexity and domain cell, with its interval and case count; the cell floor is 80%.

### Nano-XS few-shot, the cheap near-miss, is level with the pick or ahead of it outside the news cells, but trails it by 6.2 points in complex news, which puts it below the floor.

| Row | Value | 95% unpaired interval | Note |
|---|---|---|---|
| Simple × Science | -1.1 | -2.2 to -0.03699 | -1.1 pts |
| Simple × Conversational | -1.1 | -1.5 to -0.7 | -1.1 pts |
| Simple × News | -0.5 | -1.5 to +0.5 | -0.5 pts |
| Medium × Science | +0.5 | -0.8 to +1.8 | +0.5 pts |
| Medium × Conversational | -0.3 | -1.1 to +0.5 | -0.3 pts |
| Medium × News | +1.5 | +0.2 to +2.8 | +1.5 pts |
| Complex × Science | +0.3 | -1.3 to +1.9 | +0.3 pts |
| Complex × Conversational | -0.4 | -2.3 to +1.5 | -0.4 pts |
| Complex × News | +6.2 | +4.4 to +8 | +6.2 pts |

Each row is Brisk-S · Few-shot (12 ex.) less Nano-XS · Few-shot (12 ex.) on Accuracy, overall and in each place both hold it.

The interval is independent: each arm's own spread (its stated half-width), combined as if the arms ran different cases, which reads wider than a paired interval where they ran the same.

Brisk-S few-shot less Nano-XS few-shot in each cell, in points, with unpaired intervals (wider than pairing the same cases would give).

| Row | Value | Note |
|---|---|---|
| A · Atlas-M · Zero-shot | $180.0 | $180.0 |
| B · Atlas-M · Few-shot (12 ex.) | $260.0 | $260.0 |
| C · Atlas-M · Rubric + reasoning | $610.0 | $610.0 |
| D · Brisk-S · Zero-shot | $28.00 | $28.00 |
| E · Brisk-S · Few-shot (12 ex.) | $40.00 | $40.00 |
| F · Brisk-S · Rubric + reasoning | $95.00 | $95.00 |
| G · Nano-XS · Zero-shot | $1.30 | $1.30 |
| H · Nano-XS · Few-shot (12 ex.) | $1.90 | $1.90 |
| I · Nano-XS · Rubric + reasoning | $4.60 | $4.60 |

The bars are drawn on a log scale: their values span more than ten times, so each step is a multiple.

Cost per million turns on a log scale; Nano-XS few-shot would cost $57.00 a month against the pick's $1,200.00.

#### Prompt and model interact: rubric with reasoning adds accuracy over few-shot on Atlas-M and Brisk-S but costs Nano-XS 6.2 points, so no single prompt effect holds across models.

| Group | Row | Value | 95% unpaired interval | Note |
|---|---|---|---|---|
| model | Atlas-M · Few-shot (12 ex.) → Brisk-S | -1.4 | -1.8 to -1.0 | -1.4 pts |
| model | Atlas-M · Few-shot (12 ex.) → Nano-XS | -2.0 | -2.4 to -1.6 | -2.0 pts |
| model | Atlas-M · Rubric + reasoning → Brisk-S | -1.7 | -2.1 to -1.3 | -1.7 pts |
| model | Atlas-M · Rubric + reasoning → Nano-XS | -9.3 | -9.8 to -8.8 | -9.3 pts |
| model | Brisk-S · Few-shot (12 ex.) → Nano-XS | -0.6 | -1.0 to -0.2 | -0.6 pts |
| model | Brisk-S · Rubric + reasoning → Nano-XS | -7.6 | -8.1 to -7.1 | -7.6 pts |
| prompt | Atlas-M · Few-shot (12 ex.) → Rubric + reasoning | +1.1 | +0.7 to +1.5 | +1.1 pts |
| prompt | Brisk-S · Few-shot (12 ex.) → Rubric + reasoning | +0.8 | +0.4 to +1.2 | +0.8 pts |
| prompt | Nano-XS · Few-shot (12 ex.) → Rubric + reasoning | -6.2 | -6.7 to -5.7 | -6.2 pts |
| model effect, across prompt | (Atlas-M · Rubric + reasoning → Brisk-S) less (Atlas-M · Few-shot (12 ex.) → Brisk-S) | -0.3 | -0.9 to +0.3 | -0.3 pts |
| model effect, across prompt | (Atlas-M · Rubric + reasoning → Nano-XS) less (Atlas-M · Few-shot (12 ex.) → Nano-XS) | -7.3 | -7.9 to -6.7 | -7.3 pts |
| model effect, across prompt | (Brisk-S · Rubric + reasoning → Nano-XS) less (Brisk-S · Few-shot (12 ex.) → Nano-XS) | -7.0 | -7.7 to -6.3 | -7.0 pts |
| prompt effect, across model | (Brisk-S · Few-shot (12 ex.) → Rubric + reasoning) less (Atlas-M · Few-shot (12 ex.) → Rubric + reasoning) | -0.3 | -0.9 to +0.3 | -0.3 pts |
| prompt effect, across model | (Nano-XS · Few-shot (12 ex.) → Rubric + reasoning) less (Atlas-M · Few-shot (12 ex.) → Rubric + reasoning) | -7.3 | -7.9 to -6.7 | -7.3 pts |
| prompt effect, across model | (Nano-XS · Few-shot (12 ex.) → Rubric + reasoning) less (Brisk-S · Few-shot (12 ex.) → Rubric + reasoning) | -7.0 | -7.7 to -6.3 | -7.0 pts |

Each effect is one setting moved between two arms that differ in it alone, on Balanced accuracy; each interaction is the same move's effect in one context less its effect in another, the two contexts differing in one other setting. Descriptive: read as two arms' own spreads, never a test.

Balanced accuracy change from moving one setting between configurations that differ in it alone, in points, with unpaired intervals; the last rows are the interactions.

## What you could change

| What you could change | What it decides | What happens as it moves |
|---|---|---|
| Cell floor | Which configuration passes every cell | Below 76.7%, Nano-XS few-shot passes and, being cheapest, becomes the pick. From there up to 82.9%, Brisk-S few-shot holds. Above that, no configuration passes. |
| Overall accuracy line | Which configurations pass overall | Brisk-S few-shot holds at any line up to 90.7%; lowering the line admits no cheaper option, since each also fails the cell floor or the budget. Above it, nothing passes. |
| Latency budget | Which configurations pass on latency | Below 184 ms, nothing passes. Any budget above it keeps Brisk-S few-shot; a much looser one admits more accurate but costlier configurations, such as Brisk-S rubric, gaining +0.8 points, which would then be a price to weigh. |

## Next steps

- Re-measure Brisk-S few-shot's p95 latency with an interval, under production-like load. _Whether the latency clearance is real, turning the open latency caveat into evidence or reopening the choice._
- Add complex-news examples to Nano-XS few-shot's prompt and re-run that cell; more cases alone would not help, since its interval already sits below the floor. _Whether a configuration at a fraction of the pick's cost can pass every gate; if it does, it replaces the pick._
- Re-run Brisk-S few-shot on a fresh held-out sample of complex news turns. _Whether the pick's thinnest accuracy clearance survives the optimism of picking the best of many._

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

- **Traffic-weighted accuracy and macro F1** They rank the nine configurations in the same order as balanced accuracy and no gate reads them; full tables in the appendix.
- **Per-axis confusion tables** On complexity alone and on domain alone, the pick and Nano-XS few-shot look alike; their gap sits in one cell, which the cell comparison shows.
- **Zero-shot configurations and the full lever table** Every zero-shot configuration is out on the gates and does not bear on the call; the full lever and interaction table is in the appendix.
- **Failure examples** The snapshot keeps no failed turns to show.
- **The pick's other eight cells** Each clears the floor by a wide margin; they are in the cell heatmap and appendix.

### Method

- **Cases** 48210 turns per configuration, the same turns for each, run once; complex news holds 3600 of them.
- **Accuracy** In a cell, the share of turns whose true complexity and domain fall in that cell that the router labels correctly. Balanced accuracy is the plain mean of the nine cells' accuracies.
- **Gates** Balanced accuracy at least 90%; accuracy in every cell at least 80%; p95 latency at most 250 ms. No criterion is marked as set after the results. Accuracy gates are read on the whole interval where one exists, latency on its point estimate.
- **Intervals** Cell accuracies carry intervals as stated, at 95%. The interval on balanced accuracy is derived from the cells', read as independent. Comparisons between configurations combine each one's own spread as if independent, so they read wider than paired ones; no multiplicity correction applies to them, and they are descriptive.
- **Control** None: nine configurations, three models crossed with three prompts, compared on equal footing. _Caveats: No arm is the control._
- **Cost** Cost per million turns as measured, priced at the declared 30000000 turns a month.
- **Source** Turn router classifier sweep, synthetic example data.

### Review findings left open

- **Must fix** · Claim 2: c2 says the pick's thinnest clearance is complex news (82.9%, 2.9 points above the floor). The thinnest is actually overall balanced accuracy: 90.7% against 90%, a gap of 0.7 points (criteria.overall.brisk-s.few.gap), with the interval's low end at 90.4%. The chance of a fresh run reading it on the wrong side is also higher than for complex news (0.0004972% against 0.0001087%). Fix: say complex news is the weakest cell, and name overall accuracy as the thinnest gate clearance.
- **Must fix** · The page: Watch and the lede treat complex news as the accuracy risk and never mention the 0.7-point overall margin, the pick's weakest margin on any gate (the computed fact weakest_not_in_watch). Sweeping the overall line shows the call disappears above 90.7%, and a best-of-9 pick reads high by chance. Fix: add a Watch entry citing the overall value, line and gap (criteria.overall.brisk-s.few.value, .low, .gap), and name it in the lede next to the latency risk.
- **Should fix** · The page: The third next step reruns only complex news to test best-of-9 optimism. Extend it to a full fresh held-out run, so it also re-reads overall balanced accuracy, the clearance most at risk.
- **Should fix** · The page: Nano-XS few-shot's overall accuracy is borderline: 90.1%, interval 89.8–90.4% (criteria.overall.nano-xs.few.low/high). Yet 'Why not the others', the Nano-XS next step ('passes every gate') and the cell-floor row in 'What you could change' ('Nano-XS few-shot passes') treat it as clearing overall. Say it is borderline overall. Then make the Nano-XS next step settle both the complex-news floor and the overall line, perhaps noting that fixing complex news would also raise its balanced accuracy.
- **Should fix** · Claim 3: The difference table's caption says each row is 'overall and in each place', but the table has no overall row. Add Brisk-S few-shot minus Nano-XS few-shot on balanced accuracy, or drop 'overall' from the caption.

_Written by claude-opus-5-5, served as claude-opus-5-5, for $1.83, with analyst skill b70be6bcfa68, on 2026-10-28._
