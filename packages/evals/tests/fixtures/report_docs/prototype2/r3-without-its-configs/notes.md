# Reasoning note: turn router classifier sweep (snapshot "r3 without its configs")

**The eval.** A turn router that labels each turn by complexity (simple, medium, complex) and domain (science, conversational, news). A full 3×3 factorial: three models (Atlas-M, Brisk-S, Nano-XS) by three prompts (zero-shot, few-shot with 12 examples, rubric with reasoning), so nine configurations and no control. 48,210 turns per configuration over nine cells (smallest: complex conversational, 1,960). Three gates, all declared before the run: balanced accuracy at least 90% overall, accuracy at least 80% in every cell, p95 latency at most 250 ms. Volume 30 million turns a month. Synthetic data.

**What I noticed first.**
- One survivor: Brisk-S few-shot (E), at 90.7% balanced accuracy, 184 ms p95, $40 per million turns ($1,200 a month).
- Nano-XS few-shot (H) is the obvious temptation: $1.9 per million ($57 a month), 80 ms, 90.1% overall, and level with or ahead of E in most cells. It fails exactly one gate in one cell: complex news, 76.7% ± 1.4 against 80%.
- Atlas-M few-shot passes both accuracy gates but runs 483 ms. Rubric with reasoning blows the latency budget for Atlas-M and Brisk-S.
- The prompt effect is not one effect: few-shot adds +11.9 points to Nano-XS, +4.7 to Brisk-S, +2.3 to Atlas-M; rubric-vs-few-shot reverses for Nano-XS (−6.2) while helping the others slightly.

**The trap.** Two of them. First, the gate scatter (accuracy against p95, both lines) draws Nano-XS few-shot sitting comfortably inside both gates at a twentieth of the pick's cost. A reader would ask why we're paying more. The scatter hides the third gate, the per-cell floor, which is the only thing keeping H out. So the lead is the gate matrix, which shows all three gates and names the failing cell. Second, the cell that rules H out invites "route complex news elsewhere". But the router's cell is its own output: you don't know a turn is complex news until you have classified it. A cascade (H first, escalate turns it reads as complex or news) is a real idea, but nobody ran it, so it is a next step, never a priced option.

A smaller third: the pick clears the overall line by 0.7 points with no interval printed. That looks like a borderline result. settle_cases says the 48,210 turns are well past what a design needs to settle that clearance (13,512 at 80% power, two-sided) on a binomial basis. But that basis is an approximation for balanced accuracy, and code did not mark the caveat checked, so it stays open with a material_if clause that cites the settle figures rather than me waving it away.

**Story.** Decision. "Use Brisk-S few-shot as the turn router: it is the only one of the nine configurations that passes every gate, at $1,200 a month." Confidence high_if_immaterial, with four open caveats: the overall point estimate (material only if balanced accuracy is much noisier than the binomial sizing), best-of-nine selection (material only if a fresh run reads it under the line), the p95 point estimate (no spread at all; past the budget nothing passes), and no control (bears only on between-configuration comparisons, not on the gate call).

**Lead evidence.** The gate matrix: the one all-pass column, and each failure with its figure and line. Then the overall balanced accuracy dot chart against the 90% line, because the pick's weak axis has to be on the page early. Then the two few-shot heatmaps side by side, which show H's single failing cell and that it is clear below the floor, so more cases won't rescue it; a fix will. Then cost bars (log scale, 470× spread), which give the price of the floor.

**What I looked for and found.**
- Sensitivity of the call to each gate. Overall line: lowering it changes nothing; above 90.7% nothing passes. Cell floor: at 76.7% or lower the call flips to Nano-XS few-shot; above 82.9% nothing passes. Latency: below 184 ms nothing passes; widening it only admits pricier configurations. These went into explore. The cell floor is the business lever here: it is the line between $57 and $1,200 a month.
- Where H loses to E: arms_compare by cell. Complex news −6.2 (−8 to −4.4), medium news −1.5, ahead in two simple cells, unresolved elsewhere. The loss is a news-at-depth problem, which is what a targeted few-shot fix should aim at, and the seam a cascade would use.
- What kind of errors the pick makes: confusion. Almost all errors are between neighbouring complexity levels (medium splits evenly up and down); simple↔complex is rare; domain is read far better than complexity. That is a detail claim because it tells a fix where to aim and says misroutes are mostly near misses.
- Whether averaging the prompt lever is meaningful: levers_effects says no. The prompt effect differs across models and reverses for rubric on Nano-XS. Reported per model, as a detail claim.
- Failure examples: the snapshot froze none.

**What I left out, and why.**
- Traffic-weighted accuracy and macro F1: no gate reads them and they rank the same. Macro F1 puts the pick at 89.6%, under 90, which only matters if someone mistakes it for the gate; noted in omitted.
- The gate scatter: misleading for this campaign (hides the cell floor).
- Per-cell tables for the seven other configurations, the full lever list and interactions, lever effects on cost and latency (no spread), the price frontier (one survivor), Atlas-M zero-shot's 0.2-point near miss (it fails latency anyway).

**What I'd watch.** The overall margin of 0.7 points on the configuration picked as the one of nine that passed; its p95 margin of 66 ms with no spread and unstated serving conditions; complex news as its weakest cell (82.9%, interval 81.7 to 84.1, clear of the floor).

**Next steps, each with what it settles.**
- Aim Nano-XS's few-shot examples at complex news and re-test that cell: settles whether a $57-a-month configuration clears every gate. If it does, it replaces the pick. This is the step with the most leverage on the call.
- Run a cascade (Nano-XS few-shot first, hand turns it reads as complex or news to Brisk-S few-shot) as a measured arm: settles what a mix keeps of the cost and of the complex-news accuracy.
- Measure the pick's p95 under production serving with a spread: settles the latency clearance, the one gate with no uncertainty stated at all.
- Re-run the pick on fresh turns: settles the best-of-nine optimism on the overall clearance.

**What doesn't transfer.** The cheap option was kept out by one cell, so the floor became the price lever; in campaigns without a per-cell floor the same data would read as a tradeoff. The settle arithmetic made the thin overall clearance less worrying only because the turn count is large. And the routing trap is specific to a classifier whose output is the routing key.

## Tools

- **Calls.** About 40 tool calls in all (reads of the kit included), 56 invocations of `evals_analyst` (some chained in one shell call): primer and help ×4 (help, levers_effects, block_render, figure_keys), campaign_overview, caveats_list, options_dominance, measure_read ×8 (five per-arm measures plus three pages of accuracy by cell), readings_list, contrasts_list (both empty: no control), levers_effects ×2, settle_cases, decision_sensitivity ×3, price_tradeoff ×3, block_catalog, block_render ×16 (ten while choosing, six for blocks.txt), examples_list, arms_compare, figure_keys_check ×4, capability_gap ×6.
- **Refusals (3).** dot_interval over cells (it can't draw cells, and a cell id isn't a stratum); price_tradeoff with lists for arm_filter and what_if (both take one string; the recovery line was clear); lattice over configs (the snapshot holds no configs rows, which I take to be what "without its configs" means). Every recovery line was usable.
- **Wished existed (each a capability_gap).** The lattice over the model × prompt grid; an interval on overall balanced accuracy where measure_read and dot_interval read it (spread and settle compute an SE that the measure itself says it doesn't hold); figure keys for confusion cells; a spread and serving conditions for p95 latency (confound_check); a key for a monthly cost difference or cost multiple between two arms; per-panel column labels in the confusion values table.
- **Confusing.**
  - The confusion values table prints Simple/Medium/Complex as the column header for the domain panel too.
  - The bars values table repeats the value in its Note column.
  - price_tradeoff with `what_if` didn't show the what-if option on the first page I read. Nothing said where it went, so I priced Nano-XS through cost_per_month keys instead.
  - The `point_estimate:overall` caveat is listed as open while settle_cases says today's design all but never reads the wrong side. There's no route for code to mark it checked from that figure, so I couldn't tell whether I was allowed to treat it as settled. I left it open.
  - decision_sensitivity on the cell floor labels its upper boundary with Atlas-M zero-shot's complex-news figure, though the flip belongs to the pick, which has the same value.
- **Paging.** accuracy by cell took three pages (40 rows each, 81 in all); levers_effects two. Clear `next:` lines throughout.

## Revision 1

**Must-fix 1, Nano-XS's overall clearance.** Accepted. I had read H's overall pass as settled because it printed as a pass. It wasn't: it clears the line by 0.1 points, a fresh run of today's size reads it under the line 23.1% of the time, and a two-sided settle at 80% power needs about 700,000 turns. I hadn't run settle_cases on H at all, only on the pick, and that was the miss. Changes:
- The lede now says H clears the overall line only inside noise.
- why_not carries the wrong-side figure.
- The decision `next` slot and next[0] now ask for a complex-news fix followed by a full re-run on every gate with fresh turns, sized for the overall line. The held-out guard from should-fix 1 is folded in.

**Must-fix 2, c5's lever claim.** Accepted. "The lever that moves accuracy most" was wrong: rubric with reasoning gains more than few-shot on Brisk-S and Atlas-M. The claim now says what few-shot buys depends on the model (+11.9 on Nano-XS against +4.7 on Brisk-S). The caption now says rubric gains more on the two larger models but breaks the latency budget, and loses to few-shot on Nano-XS.

**Should-fix items.**
- **Held-out re-test:** fixed in next[0].
- **c6 column headers:** the renderer prints Simple/Medium/Complex over the domain panel and I can't change that, so a Method line tells the reader what those columns are. The block now holds both few-shot configurations, so the cascade step's dependence on Nano-XS flagging complex and news turns is visible: it does so about as well as the pick.
- **Gate label "Overall accuracy":** the label is the criterion's own name, so a Method line says it reads balanced accuracy, not traffic-weighted accuracy.
- **Caveat jargon:** "binomial spread code sized it on" is now "the spread expected from 48,210 turns".

**Invariant concerns.**
- "The seven other configurations" is now "the other configurations".
- Method's Source line no longer prints the provenance path. It says "Synthetic example data, run {date}".

**Disagreements.** None on substance. On c6, the header problem is a renderer limitation, so it is explained in Method rather than fixed in the block.

**Tools this round.** 5 tool calls: read the review, settle_cases on Nano-XS with a two-arm confusion render, the report.json edit, figure_keys_check on all 37 keys (all read) with the c6 re-render for blocks.txt, and this note. No new capability gap. The confusion column-header gap was already logged; the two-arm render also drops the axis names from the panel column, which makes the same gap worse.
