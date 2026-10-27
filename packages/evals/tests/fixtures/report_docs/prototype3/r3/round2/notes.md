# Reasoning note: turn router classifier sweep (r3)

**The eval.** A turn router that labels each turn by complexity (simple, medium, complex) and domain (science, conversational, news): a nine-cell target matrix. A full factorial: three models (Atlas-M, Brisk-S, Nano-XS) by three prompts (zero-shot, few-shot with 12 examples, rubric with reasoning), so nine configurations, each run once on 48,210 labelled turns. No control. Three gates, all declared before the run: balanced ("overall") accuracy at least 90%, accuracy at least 80% in every cell, p95 latency at most 250 ms. Volume 30 million turns a month. No judge; labels are the reference. No failure examples were frozen.

**What I noticed first.**
- Exactly one configuration survives every gate: Brisk-S few-shot (90.7% overall, weakest cell 82.9%, 184 ms, $40 per million turns).
- Nano-XS few-shot is the obvious rival: 90.1% overall, 80 ms, $1.9 per million (21 times cheaper). It fails only one gate, in one cell: complex news at 76.7% (±1.4), wholly below the floor.
- Everything more accurate than the pick (Brisk-S rubric, Atlas-M few-shot and rubric) is far past the latency budget.
- The pick clears overall accuracy by 0.7 points, and that figure has no interval at all.

**The trap.** Two of them. First, the gate scatter (overall accuracy against p95) puts Nano-XS few-shot inside both gate lines, closer to the corner than the pick and far cheaper; a reader of that chart alone would choose it. The cell floor removes it, and that cell's miss is clear of noise, so the lead block has to be the gate matrix, not the scatter. Second, "the only configuration in the box" sounds decisive, but its overall clearance is 0.7 points on a point estimate, and it is the best of nine. If it slips, nothing passes, and the call becomes a trade between accuracy and the latency budget. The recommendation stands; the thin overall clearance goes in the lede and the confidence, not a footnote.

A smaller trap: the cell is the label the router predicts, so "route complex news to Brisk-S" is not a mix anyone can deploy by cell. A cascade on Nano-XS's predicted label might be, but nobody ran it, so it gets no figure.

**Story.** Decision. "Use Brisk-S few-shot: of nine configurations it is the only one inside every gate, at $1,200 a month." Confidence high_if_immaterial, with four open caveats: the overall point estimate (material if the true rate is under 90%), best-of-nine (material if a fresh run reads lower by more than the clearance), the p95 point estimate (wide clearance, least likely to bind), and no control (bears only on the two-arm comparison, not on the call, which reads each configuration against gates).

**Lead evidence.** The gate matrix: each configuration against all three gates, failures named with the figure and cell. It shows the decision and why each alternative is out in one view. Then the cell heatmap of the two few-shot small models with the floor plus the per-cell difference bars, which show the rival's loss sits in news cells and is clear of the floor; then the gate scatter for why nothing more accurate is eligible.

**What I looked for and found.**
- Whether the rival's miss could be noise: no. Its complex news interval tops out under the floor, so more cases won't help; the configuration has to change.
- Whether the pick's weakest cell is solid: yes. It clears complex news by 2.9 points; a two-sided read needs 617 cases and the cell holds 3,600.
- Where the two few-shot small models differ: only in the medium and complex news cells (Brisk-S ahead); Nano-XS is ahead on simple science and simple conversational; elsewhere not separated. Uncorrected, so a lead.
- What the prompt lever does: few-shot lifts every model, Nano-XS most (+11.9 overall, +11.6 in complex news). The rubric with reasoning helps Atlas-M and Brisk-S slightly over few-shot (not separated in complex news) but takes Nano-XS back down (−6.2 overall, −5.8 in complex news, clear of zero). The levers interact, so averaging the prompt effect over models would mislead. This points the Nano-XS fix at examples, not the rubric.
- Per-axis confusions for the two few-shot small models: nearly identical on each axis, so they don't locate the complex news gap.
- Sensitivity: the call flips to Nano-XS few-shot at a cell floor at or below 76.7%; nothing passes above 82.9%, above 90.7% overall, or under 184 ms p95. Raising the budget to 496 ms admits more accurate, costlier configurations; code's call stays with the pick on cost.

**What I left out, and why.**
- Traffic-weighted accuracy and macro F1: same ranking, no gate, a third name for one reading.
- Per-axis confusion tables: they don't show where the gap is.
- Atlas-M detail and the cost of the more accurate configurations: out on latency by far; one line in why_not and the latency control.
- Model-lever effects: the gates settle the model.
- Examples: none frozen.

**What I'd watch.** The overall accuracy clearance (0.7 points, no interval, best of nine). Complex news is the weakest cell for every configuration; the pick's clearance there is settled. The monthly saving at stake between the two few-shot small models is about $1,140, which is small next to engineering time; whether it is worth a second iteration is the owner's call, not mine.

**Next steps, each with what it settles.**
- Re-run Brisk-S few-shot on fresh held-out turns with a case-level interval on overall accuracy: settles whether the pick clears beyond noise and best-of-nine luck; if not, nothing passes and the owner chooses between accuracy and latency.
- Add complex news examples to Nano-XS few-shot (not the rubric) and re-run that cell: settles whether the cheaper, faster option clears the floor and takes the call.
- Owner declares whether the monthly difference is worth that iteration: settles whether the second step runs at all.

**What doesn't transfer.** The thin overall clearance mattered because it was a point estimate with no interval; with an interval it would be a settle_cases sizing. The rival lost only because one cell of a crossed target was weak; the general lesson is to check per-cell floors before reading a two-gate scatter.

## Tools

- **Calls:** 53 `evals_analyst` invocations (primer 1; help 3, including figure_keys and block_render; campaign_overview 1; options_dominance 1; caveats_list 1; measure_read 8, three of them pages of the cell table; levers_effects 2; settle_cases 2; decision_sensitivity 3; block_catalog 1; examples_list 1; arms_compare 1; figure_keys_check 6; block_render 13 while exploring plus 6 to write blocks.txt; capability_gap 3). Four were refusals. I also read the skill, its modules, four exemplars and ANALYST §5 as files.
- **Wished existed (each logged as capability_gap):** an interval on overall balanced accuracy and p95 so the pick's thinnest gate can be sized; cell ids accepted as places in block queries over settle and levers; a joint (cell-to-cell) confusion table for a crossed target.
- **Confusing:**
  - The levers block's interaction rows are labelled with arms that don't identify the contrast: "model effect, across prompt, at Brisk-S · Few-shot less at Brisk-S · Zero-shot" for the Brisk-S→Nano-XS effect, and two different rows both read "prompt effect, across model, at Nano-XS · Zero-shot less at Brisk-S · Zero-shot" (+7.2 and +0.2). A reader can't tell which interaction is which.
  - The settle table prints case counts with a decimal ("617.0") and carries six empty trial columns on a design with no repeats.
  - `cost.gap.<a>.<b>.multiple` prints "21.05 times", which only reads well as "costs 21.05 times as much"; a ratio key for the cheaper direction would help.
  - `weakest.*` fields are `margin, arm, criterion, value, threshold`, but help doesn't list them; I found them through a refusal.
  - `figure_keys_check` paged at 22 keys, so a 23-key draft took two calls.
  - A gate_scatter refuses `reference: budget` with two measures yet draws both gate lines without it; the refusal text suggested naming one measure, which would have lost the chart.
- **Refusals:** gate_scatter with a reference (above); `strata` naming a cell for settle (twice, `complex.news` and `cells.complex.news`) and for levers. Each recovery line pointed at the right parameter but not at a valid value for a cell.
- **Paging:** measure_read by cell needed three pages (81 rows); confusion over all arms ran 162 rows, so I filtered by arm and axis (`measures: ["domain"]` works as the axis filter, which help doesn't say).

## Revision 1

**What the editor found.** 13/14 and a pass, with one must_fix and seven should_fix items.

**What changed.**
- *Must fix, c2 caption.* I called the two small few-shot configurations "the two cheapest options inside the latency budget", which was wrong: inside the budget the pick is the costliest of five. The caption now says what is true: they are the only two inside both the latency budget and the overall accuracy gate.
- *Overall accuracy's noise.* I agree with the editor. Balanced accuracy is the plain mean of the nine disjoint cells (816.0/9 = 90.67), and the cells' half-widths combine to about ±0.31, which is wholly above 90%. I had treated an answerable question as open. No key prints that interval, so the page says it in words: in Method (Accuracy, Intervals), in the `point_estimate:overall` material_if, in the lede and in Watch. The open risk is now named for what it is: selection as the best of nine, plus drift on fresh turns. I kept the caveat open because only code may close it, but narrowed its clause. next[0] is now a fresh-turn re-run that answers the selection question, not a request for an interval. Logged as a capability gap.
- *c4.* The settle table is now filtered to the cell floor (`measures: ["accuracy"]`), so every row has a distance, and the claim is narrowed to "every cell clears and is settled; complex news is the narrowest". The overall line moved to the lede, Watch and the confidence. I can't drop the empty trials columns or fix the note's word "bar"; those are renderer defects.
- *c5.* The claim was wrong as stated: on Brisk-S the rubric lifts slightly more than few-shot (+5.5 vs +4.7). I restated it as "on Nano-XS few-shot lifts far more than the rubric (+11.9 vs +5.7); on Brisk-S the two are close, and the rubric is out there on latency". The evidence is now two levers blocks filtered to one model each, which removes the model effects and the mislabelled interaction rows. That makes the omitted entry for model levers true.
- *next[1].* This step now re-runs the whole Nano-XS configuration on fresh turns. It also says that a prompt tuned to one cell is a new pick and needs the same fresh-turn check.
- *Omitted, confusion.* I disagree in part. The snapshot does hold per-axis confusion figures, through `confusion.<axis>.<arm>.<expected>.<predicted>` keys (the editor's tool set has no block_render, which may be why it couldn't read them). The reason now cites four of them instead of asserting the fact, and says that no joint table is held.
- *Printing.* The overall gap now prints without a sign (`|abs`). The cost multiple is now `|int` ("21 times"). The scatter caption calls its horizontal line the overall accuracy line. The renderer still labels it "floor"; that is the kit's to fix.

**Disagreements.** One, a partial disagreement over the confusion figures, explained above. No other should_fix was declined.

**Tools this round.** 13 `evals_analyst` calls: block_render 2 to try the filtered settle and levers blocks, plus 7 to rewrite blocks.txt; figure_keys_check 3; capability_gap 1. `figure_keys_check` cut its first page at 20 keys this time, not 22. I sent offset 22 without reading the `next:` line, so two keys were skipped. Both had read earlier this round (5.8 and +11.9).

**New gap.** No figure key for the interval of a mean-of-cells measure, derived from the cells' stated spreads.
