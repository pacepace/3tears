# Reasoning note: turn router classifier sweep (r3)

**The eval.** The turn router's classifier, which labels each turn by complexity (simple, medium, complex) and domain (science, conversational, news). A full factorial grid: three models (Atlas-M, Brisk-S, Nano-XS) crossed with three prompt styles (zero-shot, few-shot with 12 examples, rubric + reasoning), so nine configurations, no control. 48,210 turns per configuration, the same turns for all, spread over the nine complexity × domain cells (Simple × Conversational is the largest at 14,880; Complex × Conversational the smallest at 1,960). Three gates declared with the campaign: balanced accuracy at least 90% overall, accuracy at least 80% in every cell, p95 latency at most 250 ms. Volume 30M turns a month. No caveats, no late criteria, no prior insights, no failure examples, and no interval on any figure.

**What I noticed first.**
- `options_dominance` returns one survivor: Brisk-S few-shot (90.7% balanced, 82.9% in its weakest cell, 184 ms p95, $40 per million turns, $1,200 a month).
- Its overall clearance is 0.7 points. Every gate figure is a bare point estimate; balanced accuracy carries `level: none`.
- Nano-XS few-shot is 20× cheaper ($57 a month), 80 ms p95, 90.1% overall, and fails only one cell: Complex × News at 76.7%.
- Everything more accurate than the pick (Brisk-S rubric 91.5%, Atlas-M few-shot 92.1%, Atlas-M rubric 93.2%) is out on latency (496, 483, 1302 ms).
- Rubric + reasoning helps Atlas-M (+1.1 over few-shot) and Brisk-S (+0.8) but hurts Nano-XS badly (−6.2, to 83.9%), with medium complexity collapsing from 88% to 80.5% correct.

**The trap.** Two of them. First, "the only configuration that passes" sounds decisive, but it passes the overall gate by 0.7 points, with no interval to say whether that is clear, and it is the one of nine picked because it passed. The recommendation stands (nothing else passes, and it is cheap), but the clearance is part of the story, on the first screen, not a footnote. Second, the grid invites lever averages ("what the rubric buys"), and the rubric's effect flips sign on Nano-XS. Averaging it would describe no model. A third, smaller one: the gate scatter shows Nano-XS few-shot inside both drawn gates; it is out only on the cell floor, which the scatter doesn't draw, so the caption has to say so.

**Story.** Decision. "Use Brisk-S few-shot: it is the only configuration of nine that clears every gate, though it clears overall accuracy by just 0.7 points." Confidence is high, if two caveats aren't material:
- **The overall clearance reads a point estimate.** Material only if Brisk-S few-shot's true balanced accuracy is below 90%; then nothing passes every gate and the choice becomes accuracy against latency. My own back-of-envelope (not on the page, since no tool returns it): on ~48k turns a binomial standard error is about 0.14 points, so 0.7 would be several errors clear, but turns may cluster by conversation and balanced accuracy weights small classes up, so I don't trust that arithmetic enough to type it.
- **No annotator agreement for the gold labels.** Material only if labels are noisy exactly where the errors and the binding cell sit (medium complexity, Complex × News); the gates are absolute lines, so noise moves the readings against them.
- Code proved no caveat immaterial; the caveat list was empty, so both caveats are mine, under ids I named.

**Lead evidence.** The lattice: one tile per model × prompt with balanced accuracy, cost, p95 and a pass/fail pill. One view shows that a single tile passes, why each neighbour fails, and the model × prompt interaction. Then the overall clearance as dots against the gate (dot_interval with no intervals: the caption says so); then the cell heatmap for the two few-shot configurations a decider would actually weigh; then the gate scatter for "everything more accurate costs latency"; then the interaction as detail.

**What I looked for and found.**
- Whether the call is sensitive to the declared lines (`decision_sensitivity` on each threshold). Overall: holds at any gate up to 90.7%. Cell floor: below 76.7% the call moves to Nano-XS few-shot; above 82.9% nothing passes. Latency: nothing passes below 184 ms; above 483 ms Atlas-M few-shot enters, and code still picks Brisk-S few-shot on cost because no accuracy axis ranks survivors. That became the Explore section.
- Where the errors live. Complex × News is the weakest cell for every one of the nine configurations, and medium complexity is the most-confused class (symmetric spill into simple and complex). Domain errors are small. I couldn't get confusion within the Complex × News cell, so I don't know if that cell fails on complexity or domain (gap).
- Whether the cheap near-miss is rescuable. Nano-XS few-shot misses by 3.3 points in one cell of 3,600 turns, so it is worth a targeted prompt fix; but its overall clearance would be 0.1 points, thinner still.
- Whether a routed mix helps. The obvious "send Complex × News to Brisk-S" is circular: this classifier is what produces the cell. Nobody ran a cascade, so it is omitted, with a reason, rather than priced.
- What the rubric arm actually changed. It bundles a rubric with a reasoning step, so neither the accuracy gain nor the latency cost can be credited to one; that became a next step.

**What I left out, and why.**
- Traffic-weighted accuracy and macro F1: not gated, same ordering near the decision.
- Domain-axis confusion and Brisk-S few-shot's own confusion: they describe the error shape and don't change the call.
- Per-cell accuracy of the seven other configurations: each is already out on a named gate.
- Scoreboard: nothing lost or failed.
- Latency distributions: only p95 is in the snapshot, so range_bars can't be drawn.
- The routed mix: circular as stated, never run.

**What I'd watch.**
- Overall accuracy: 0.7 points over the gate, unintervaled, on the configuration picked for passing.
- Complex × News: 2.9 points over the floor for the pick, and the weakest cell for everything.

**Next steps, each with what it settles.**
- Compute case-level intervals on the frozen cases for overall balanced accuracy and every cell: settles whether either clearance is real; if the overall interval crosses 90%, nothing passes robustly and the report becomes a tradeoff.
- Confirm Brisk-S few-shot on a fresh sample of production turns: settles whether the clearance survives outside the set it was chosen on.
- Add Complex × News examples to Nano-XS's few-shot prompt and re-run: settles whether a $57-a-month option can clear the floor (against $1,200).
- Run Brisk-S with the rubric but no reasoning step: settles whether the rubric's accuracy gain fits the latency budget.
- Double-label medium-complexity and Complex × News turns: settles how much of the error and the thin clearance is label noise.

**What doesn't transfer.** The interaction finding is this grid's: the rubric hurts the smallest model only. The general lesson is to check a lever's direction per model before averaging it. The thin clearance mattered because there was no interval at all; with intervals it would have been a borderline-or-not reading instead of an open caveat.

## Tools

**Calls.** 60 `evals_analyst` calls (primer and help counted), under the 80 budget; 31 tool calls of my own (Bash, Read, Write). By action: help ×7 (index, figure_keys twice after one refusal, figure_keys.cost, block_render, decision_sensitivity, price_tradeoff), campaign_overview ×1, options_dominance ×1, measure_read ×8 (five arm-level measures, three pages of cell accuracy), caveats_list, readings_list, contrasts_list, examples_list, block_catalog ×1 each, decision_sensitivity ×3, price_tradeoff ×2, block_render ×26 (exploration, three figure-key checks through block titles, one refused lines block run twice to read the refusal, and two passes rendering blocks.txt), capability_gap ×6.

**What I wished existed** (each also a capability_gap call):
- An interval on every gate figure (balanced accuracy, each cell's accuracy). Without one, dot_interval draws bare dots and the whole report's confidence rests on an unsized clearance.
- `settle`: the cases needed to separate the pick from the 90% gate and the 80% floor.
- Lever effects over a factorial grid: what each prompt style buys per model as signed deltas, with an interaction check. I read nine lattice tiles by eye instead; diverging_bars has no lever-delta source, and `lines` over prompt styles was refused (it only draws over aspects).
- Confusion within one target-matrix cell, to say whether Complex × News fails on complexity or domain.
- A figure-key check action. I validated every key by putting them all in a one-row table's `title` and reading back the resolved text, which works but is a hack.
- `confound_check`, for the rubric arm's bundled change and for whether latency was measured on a shared machine (the snapshot carries no latency caveat either way).

**Confusing in help, refusals or paging.**
- The brief's evidence shape is `{block, query, caption}`, but `reference` (the floor or budget line) and `emphasis` are block_render parameters outside `query`. I added them as sibling fields on the evidence entry so the block renders as drawn; the report-doc schema should say where they go.
- `help` refuses a `snapshot_id`; my wrapper injected one on every call, so the first figure_keys help was refused. The recovery line was clear.
- `measure_read` on `latency_ms` by arm returns rows with empty figures; the p95 figures live only under `criteria.p95.<arm>.value`. The overview lists `accuracy` with every arm in `arms_without`, though cell-level figures exist; both read as "no data" at first glance.
- The `bars` values table repeats the value in a `Note` column.
- `options_dominance` reports `axes: []` and still "dominates on cost"; it took the latency sweep to see that a looser budget would keep the call only because no accuracy axis ranks survivors. A note saying "with no ranking axis declared, cost alone decides among survivors" would have saved a call.
- `confidence.caveats` expects ids; with `caveats_list` empty, the open caveats are analyst-raised and their ids are mine. If the mechanical check only accepts snapshot caveat ids, these will be refused, and there is no action to register an analyst-raised caveat.
- `price_tradeoff` with one survivor says only that there is nothing to trade; pricing against `arm_filter` lists every gated option without figures, yet `price.<from>.<to>.<measure>.monthly_cost_gap` resolves for a gated pair through a title. It is unclear whether prose may cite a price between a survivor and a gated option.
- Paging was clear: every `next:` line was a ready-to-send call.

## Revision 1

The editor scored round one 10/14, not passing. Its one must-fix finding was right: round one said that no gate figure carries an interval, and that was false. Every editor finding is accepted and fixed below. None is rebutted. The round-one files are in `round1/`.

**How I looked for intervals in round one, and why I missed the cell intervals.**
- `campaign_overview` listed `accuracy` with `level: 95%`, `slices: [cell]`, and `arms_without` naming all nine arms. I read `arms_without` as "no data" and let it outweigh the level. The listing doesn't say which statistics a measure holds.
- `measure_read {"measure_key":"accuracy","by":"cell"}` (three pages, offsets 0, 40 and 80) returned only `cells.<cell>.<arm>.accuracy.mean`. I never passed `statistics`, and I never read `help topic='measure_read'`, so I didn't know the default is the mean alone. SKILL.md's table does list `statistics` as a key parameter. I skipped it. Nothing in the result said other statistics exist. A line such as "also held: half_width; pass statistics to read it" would have caught this.
- `measure_read {"measure_key":"balanced_accuracy","by":"arm"}` returned the mean only, and the overview gave balanced accuracy `level: none`. That is correct for that measure, and I wrongly generalised it to every gate figure.
- `block_render` heatmap over `cells` (with `reference: floor`) printed `82.9% (n = 3600)` per cell with `level:` empty. `dot_interval` over balanced accuracy also had an empty level. No block I rendered showed a cell interval, so the absence looked confirmed twice.
- `help topic='figure_keys'` gives the form `cells.<cell>.<arm>.<measure>.<statistic>` without listing which statistics exist. I didn't open `figure_keys.slices`.
- In round two, `measure_read` with `statistics: [mean, ci, half_width, sem, n]` returned `half_width` for all 81 arm × cell pairs. It silently dropped `ci`, `sem` and `n`, since each is returned "where the snapshot states it". The cell intervals were in the snapshot all along. Only an explicit statistics list reaches them.
- Discoverability finding: interval statistics are opt-in on `measure_read` and never drawn by the cell blocks. A default read gives no hint that they exist. Logged as a capability gap.

**What changed, finding by finding.**
- *Must-fix: the false "no interval" statement.* The flag, lede, captions, Method and Watch now say that only balanced accuracy lacks a stated interval. c3, Watch and why-not cite the Complex × News half-widths (Brisk-S few-shot 1.2, Nano-XS few-shot 1.4).
- *Must-fix: whether the 0.7-point overall clearance is inside the noise.* I checked the editor's derivation against the snapshot. The plain mean of Brisk-S few-shot's nine cell accuracies is 90.67, which shows as 90.7. Its square root of summed squared half-widths, divided by 9, is about 0.31, which puts the clearance at roughly two half-widths. Nano-XS few-shot comes out at 90.10 ± 0.31, straddling the gate. No key prints the derived interval, so the page states it in words: the cells are disjoint, and each is known to within the widest cell half-width (Complex × Conversational, 1.4, cited by key). "Just" is gone from the headline. The caveat is now `overall_interval_inferred`, material only if the cell intervals understate uncertainty, for example through clustering by conversation. It stays open, not checked, because no code proved it. The lead next step is now a fresh-sample confirmation. Round one's "compute intervals" step is gone, and its tool side is a gap.
- *c5 scope and cause.* The claim is narrowed to moving from few-shot to rubric + reasoning, and the caption now says few-shot beats zero-shot on every model. The "medium complexity" cause is dropped. The confusion block is replaced by a `diff_heatmap` showing the Nano-XS loss spread across every cell (5.7 to 6.7 points). The mismatch between the confusion tables and the cell accuracies goes into Method and a next step.
- *c3 overreach.* A second block, the full heatmap for every configuration, now carries "weakest cell for every configuration". The cells source can't be filtered to one cell, which is logged as a gap.
- *Domain-confusion omission reason.* Rewritten: confusion over all turns can't show which axis fails inside Complex × News. A new next step declares the definition of a correct turn and breaks that cell's errors down by axis.
- *Latency trade unpriced.* Explore now prices it from the `price.brisk-s.few.<to>.balanced_accuracy` keys. Atlas-M few-shot is +1.4 points for $6,600 more a month. Brisk-S rubric is +0.8 points for $1,650 more and is the cheaper step up.
- *Routed-mix reason.* "Circular" was wrong for a cascade routed on Nano-XS's own prediction. The reason is now that nobody ran it, and that Complex × News is too large a share of turns for a second call to stay out of the p95 tail (modelled, not measured).
- *Label-noise caveat.* Now says label noise shifts readings against the absolute gates and can reorder close configurations.
- *Nano-XS is closer than shown.* The why-not now says it is nearly as accurate on the production traffic mix, at its monthly cost. The traffic-accuracy omission says the same. The next step on Nano-XS says its own 0.1-point clearance sits inside the inferred noise.
- *Invariant concerns.* Spelled-out counts are gone ("every gate", "every cell", "all {n_candidates} are new"). The gate_scatter values table still labels points with raw arm ids. That is the renderer's labelling, and no other block kind draws both gates, so I kept the block and logged a gap. The repeated `Note` column on bars and "1302 ms" without a thousands separator are renderer issues I can't change from the report.

**Tools in revision 1.** 23 `evals_analyst` calls, within the 40 allowed: help ×1 (measure_read), measure_read ×3 (one refused for listing `interval`, which isn't a statistic, then cell accuracy and balanced accuracy with statistics), block_render ×13 (four trial blocks, two of them refused because the cells and measures sources take no cell id as a stratum, two figure-key checks, seven blocks for blocks.txt; one trial block is not in the final report), capability_gap ×6. New gaps:
- a derived interval for an averaged measure;
- interval statistics surfaced by default;
- a filter on the cells source for one cell;
- gate_scatter display names;
- the correct-turn definition and reconciling confusion with cell accuracy;
- registering analyst-raised caveats.
