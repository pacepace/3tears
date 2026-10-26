# Reasoning note: query planner config sweep, no gates (prototype r2-no-gates)

**The eval.** A research query planner swept over a full factorial: three models (Atlas-M, Brisk-S, Nano-XS) by three reasoning levels (off, low, high) by retry (none, one). Eighteen configurations, 1,200 cases each, none lost. Four tasks: single lookup, multi-hop, date and unit math, ambiguous entity. Measures: pass rate, cost per call, p50 and p95 latency, calls per task. Volume is 4 million a month. Nothing else is declared: no criteria, no floor, no latency budget, no control, no caveats, no prior insights, no definition of pass. Pass rates carry no interval.

**What I noticed first.**
- Code's dominance call names Nano-XS, reasoning off, no retry (M) dominant. It is the cheapest configuration and has the lowest pass rate, 81%. With no criteria declared, cost is the only axis the computation weighs, and adding pass rate as an axis changes nothing in `dominated_by`.
- The pass-rate and cost frontier (from `price_tradeoff`) runs M, N, Q, S (all Nano-XS), then J (Brisk-S low, 1 retry), L (Brisk-S high, 1 retry) and F (Atlas-M high, 1 retry). The monthly price of a point climbs $3, $17, $111, $320, then jumps to $1,595 (J to L) and $17,236 (L to F).
- Every retry arm beats its no-retry pair, by 0.9 to 7.5 points, and gains most where the base is weakest. Retry leaves p50 unchanged and lengthens p95 a little.
- Five of the six Atlas-M settings are matched or beaten by a cheaper Brisk-S setting. Only F reaches the frontier, and it adds 0.7 points over L for $11.4k a month and a p95 of 2.55 s.
- Multi-hop is the weakest task for every configuration.

**The trap.** Two of them. The first is taking code's dominance call. "Nano-XS off, no retry dominates" is true only on the one axis the campaign gave the computation, and it would ship the worst planner in the sweep. The second is deciding anyway. With no floor, budget or value per point declared, "the knee" is a natural lean, but it is still a lean. The report prices the trade, names the band where each option wins, and says the owner makes the call.

**Story.** A tradeoff. Headline: pass rate is cheap up to Brisk-S low with one retry (J) and expensive after it, where a point costs $1,595 a month; with no floor or latency budget declared, where to stop is the owner's call. The lean, labelled as ours: J, which is best for any monthly value of a point between $320 and $1,595. Confidence is high if the caveats are immaterial. The trade's shape rests on large gaps and exact costs. The caveats are the missing intervals on the sub-point steps above J, best-of-eighteen optimism, the undefined pass, and the cost basis.

**Lead evidence.** The cost and pass-rate scatter over all eighteen configurations, on a log cost axis, because the shape is the finding: a steep climb, then a flat top. Then the lattice (model by reasoning and retry, with cost and p95 on each tile), which carries the retry finding and the full grid in one view. Then a scatter of Atlas-M and Brisk-S alone, which shows Atlas-M is dominated below its top setting. Then range bars for the four options a decider would weigh (S, J, L, F), because latency is the second axis of the trade and no budget was set. A task heatmap and the lean's failure cards are detail.

**What I looked for and found.**
- Whether retry's gain interacts with reasoning and model. It does: the gain shrinks as the base rises. So I state it cell by cell ("at every level, most where weakest") and never as an average.
- Whether a latency budget would move the lean. A p95 budget under 513 ms rules J out. Under 870 ms every Atlas-M setting is out as well.
- Where the lean fails. Its failure cards are a collapsed second hop, an off-by-one date window, and an ambiguous name answered without asking. All three are planning slips a fix could target, and they make a next step.
- Whether any tool could separate J from L (0.8 points apart at 1,200 cases each). None could. There are no intervals, no pairs and no `settle`.

**What I left out, and why.**
- Code's dominance call as a claim. It is stated once in Method with the reason the report disagrees, not argued in the body.
- Calls per task. It feeds cost, which is shown.
- The outcome and extra-calls measures. They are listed but hold no figures.
- Brisk-S low, no retry (I). It is on the Pareto edge but best at no price. It matters only under a narrow latency budget, which the explore controls cover.
- The Nano-XS malformed-JSON card. It belongs to an arm off the shortlist.

**What I'd watch.**
- J's multi-hop rate (95.6%), its weakest task.
- The lean as the knee picked from eighteen. Its estimate is the optimistic one, and its gap to the neighbouring frontier steps carries no interval.
- The ambiguous-entity scoring, where only a clarifying question passes. If the product would accept a confident pick, every arm's pass rate on that task is understated.

**Next steps, each with what it settles.**
- The owner declares a pass-rate floor and a p95 budget, or a value per point. Settles which band applies and turns the trade into a decision.
- A paired rerun of S, J and L on the same cases, with intervals on the differences. Settles whether the steps either side of the lean are real.
- Target J's slips (a date-window bounds check, per-hop planning) and rerun it. Settles whether the cheap option closes part of its gap to L.
- State whether the monthly volume counts tasks or calls. Settles the monthly dollar amounts, not the order of the options.

**What doesn't transfer.** The knee here is unusually sharp, a fivefold jump in price per point, which makes a lean easy to defend. A smoother frontier would leave no natural stopping point, and the report would be a pure priced menu. Retry being nearly free is a property of this planner's cost structure, not a general rule.

## Tools

**Calls.** 28 harness tool calls: 1 Read, 1 Write and 26 Bash, several of which batched multiple tool actions. About 55 `evals_analyst` actions:

| Action | Calls |
|---|---|
| `--primer`, `help` (general, `figure_keys`, `measure_read`, `options_dominance`, `price_tradeoff`, `block_render`, `capability_gap`) | 8 |
| `campaign_overview` | 1 |
| `measure_read` | 10 |
| `options_dominance` | 2 |
| `readings_list`, `contrasts_list`, `caveats_list`, `examples_list` | 4 |
| `block_catalog` | 1 |
| `price_tradeoff` | 1 |
| `decision_sensitivity` (refused) | 1 |
| `measure_breakdown` (refused) | 1 |
| `block_render` while choosing blocks | 9 |
| `block_render` again to write `blocks.txt` | 6 |
| `capability_gap` | 11 |

The budget was not a constraint.

**What I wished existed.** Each item is also a `capability_gap` call, listed in `gaps.json`.
- Intervals on pass rate and on paired differences between configurations. This is the biggest gap: the trade's upper steps are each under a point, and nothing says whether they are real.
- A dominance computation that doesn't rank on cost alone when no quality axis is declared, and whose `axes` actually feed `dominated_by`. Today it names the worst planner dominant, and `code_call.modes` says `decide`.
- A latency axis in dominance read on p95. It reads the mean, which latency doesn't carry, so every arm shows "figure: none".
- `decision_sensitivity` over a hypothetical, undeclared floor or budget. With nothing declared there is nothing to sweep, though "where would a floor flip it" is exactly the question.
- A cost unit that matches. The measure is "Cost per 1k calls", but the `usd_per_1k` filter and the lattice tiles print "per 1,000 replies". The digit rule allows "per 1,000 replies" but not "per 1,000 calls", so the correct unit is unwritable in prose.
- A within-cell factor effect (what retry adds in each cell).
- A block that draws `price_tradeoff`'s steps. The knee is the lead finding, and no block shows price per point as rows.
- Frontier membership and reader labels in `scatter_frontier`'s values table. Its rows are labelled `atlas-m.off.r0` and don't say which points are on the Pareto line.
- A slice comparison between two named arms without a control. `measure_breakdown` refuses with no control, and task case counts are not stated.
- `settle`.
- Caveat entries for what the snapshot plainly lacks: intervals, a pass definition, and selection optimism. `caveats_list` is empty, so the confidence caveat ids are self-made.

**Confusing in help, refusals, paging.**
- `measure_read` with the default statistics returned empty rows for latency, with no hint that latency holds only p50 and p95. I found that by guessing. A note like "this measure holds: p50, p95" would save a call. The same silent-empty result came back for `outcome` and `extra_calls`, where it means "no figures at all", which looks identical.
- Refusals were clear and the `recovery` lines were usable. I sent `ci_low` and `ci_high`, and the refusal listed the valid statistics.
- `price_tradeoff` paged off-frontier rows at 6 by default. I didn't need the rest.
- The examples' reason text carries typed figures ("9 of 17", "70%"). They render on cards but can't be cited in prose.
- The `table` block renders long format (arm, measure, value). A wide option-by-measure table, with monthly cost as a column, would have been the natural "four options" table, so I used range bars and the heatmap instead.
- There is no way to check that the report's figure keys resolve before the editor sees them. I used only keys the tools printed.
