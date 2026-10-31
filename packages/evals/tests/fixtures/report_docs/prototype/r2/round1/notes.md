# Reasoning note: query planner config sweep (r2)

**The eval.** A research query planner, swept over three models (Atlas-M, Brisk-S, Nano-XS), three reasoning levels (off, low, high) and retry (none or one): 18 configurations, fully crossed, 1,200 queries each, no repeats recorded. Four query types: single lookup, multi-hop, date and unit math, ambiguous entity. Two gates, both overall: pass rate at least 97%, p95 latency at most 600 ms. No control, no production configuration, no aspects, no caveats, no prior insights, nothing marked as set late. Volume 4M queries a month. The snapshot holds point estimates only: no interval on any figure.

**What I noticed first.**
- Exactly one configuration passes both gates: Brisk-S, low reasoning, one retry (J). Pass rate 97.1% against 97%, p95 513 ms against 600 ms, $0.30 per thousand queries ($1,208 a month).
- Its pass-rate clearance is 0.1 points. Nothing on the page can say whether that is real, because there is no interval. At 1,200 cases a binomial interval near 97% would be around a point wide each side (my arithmetic, not a tool figure, so not on the page).
- Everything that clears pass rate comfortably (Brisk-S high+retry at 98%, Atlas-M low+retry, Atlas-M high) misses the latency budget. Nano-XS never reaches 97%.
- A retry is the cheapest pass rate in the sweep. Within every model and reasoning level it adds roughly 1 to 7.5 points for at most a few cents per thousand; a step of reasoning costs roughly double and adds hundreds of ms. Retry gains shrink as reasoning rises (an interaction: they substitute).

**The trap.** Reading "the one survivor dominates" as a settled decision. `options_dominance` returns a clean single dominant arm, but it is the best of 18 configurations sitting right at one gate, cleared by a tenth of a point on a point estimate. That is the "best of eighteen is partly the luckiest of eighteen" case from statistics.md, with no interval to bound it. And if it falls below, `decision_sensitivity` shows the call goes to no_call: nothing else passes both gates. So the headline has to carry both the call and the re-test, and the first next step is a declaration of which gate gives, made before the re-test result is known.

A second, quieter trap: the gate reads the overall rate, and J is below 97% on multi-hop (95.6%) and ambiguous entity (96.4%). The overall is not the plain average of the four task rates, so the eval's case mix weights it. If production carries more multi-hop than the eval set, J's real rate is lower. The snapshot holds no task case counts, so this stays an open caveat.

**Story.** Decision, with a borderline gate. "Use Brisk-S low+retry, the only configuration inside both gates, and re-test it before relying on it: it clears the pass-rate gate by just 0.1 points." Confidence is high_if_immaterial, with three open caveats: the point-estimate clearance (material if the true rate is below 97%), selection among 18 (material if a fresh-case re-run lands lower), and case mix versus traffic (material if production has more multi-hop or ambiguous queries). I considered `low`, but if the clearance holds the pick is unambiguous (only survivor, cheapest near the gate, latency headroom); the weakness is one named caveat, which is what high_if_immaterial is for. The first-screen flag states that no interval exists.

**Lead evidence.** The gate scatter: pass rate against p95, both gates drawn, latency on a log scale. It shows in one view that one point sits inside the corner and how close it sits to the pass-rate edge. Then the lattice (model × reasoning·retry) with cost, latency and status per tile, which carries the retry-versus-reasoning finding and why each other option is out. Then cost bars for the fallbacks, because the decider needs to know what relaxing the latency budget would cost. Then per-task bars and the three frozen failure cards as detail.

**What I looked for and found.**
- Where the call flips. Pass-rate threshold: above 97.1% no call; at or below 95.3% the pick moves to Brisk-S low without retry. Latency budget: the pick holds at every budget from 513 ms up (slower passers cost more); below it nothing passes. Both went to explore.
- Where the pick fails. Multi-hop is weakest; the frozen multi-hop failure collapses two hops into one search. The date-math card's reason text says off-by-one window starts are the most common wrong-args pattern. Only three cards exist, one per type, so they illustrate, they don't measure.
- Whether a priced trade exists. It doesn't: one survivor. `price_tradeoff` prices survivors only, so the fallback cost came from cost bars and cost_per_month keys.
- Per-task breakdowns against a control: refused (no control). Read per-task rates with measure_read instead.

**What I left out, and why.**
- Calls per task: planner behavior, no gate, doesn't change the call.
- Extra calls and outcome measures: listed but empty in the snapshot.
- p50 latency: not gated; it is in the lattice.
- Per-task rates of the other configurations: the gate is overall; full table in the appendix.
- A routed per-task mix: no per-task gate, nobody ran it.
- No floor line on the per-task bars: the 97% gate is overall, and drawing it per task would invent a per-task gate. The first render drew it; I removed it.

**What I'd watch.**
- The pass-rate clearance, the weakest margin, on the pick itself.
- Latency headroom is wider (87 ms) but a retry adds tail latency; a heavier multi-hop mix would add both failures and retries.
- Multi-hop at 95.6%.

**Next steps, each with what it settles.**
- Re-run the pick on fresh cases, several times the current set, with intervals: settles whether the clearance is real or selection luck. (No `settle` to size it.)
- Owner declares which gate gives if the re-test fails (latency budget, where Brisk-S high+retry passes at $2,532 a month, or the pass-rate gate): settles the fallback before the result can bias it.
- A multi-hop planning directive, re-run on multi-hop and ambiguous queries: settles whether the margin widens without paying for reasoning.
- Declare the production query-type mix or reweight the eval: settles whether the overall rate the gate reads matches traffic.

**What doesn't transfer.** A single dominant survivor reads as a decision only when its gate clearance is clear of noise. Here it was a tenth of a point, with no interval and many configurations tried, so the decision carries its own re-test. The retry-versus-reasoning substitution is this sweep's finding; check the interaction before averaging factor effects elsewhere.

## Tools

**Calls.** 52 tool calls to `evals_analyst` (one `--primer`, then 51 JSON calls): help 5, campaign_overview 1, readings_list 1, contrasts_list 1, caveats_list 1, measure_read 11, options_dominance 1, decision_sensitivity 2, price_tradeoff 3, measure_breakdown 1 (refused), examples_list 1, block_catalog 1, block_render 15 (10 exploring and key-checking, 5 the final evidence renders written to blocks.txt), capability_gap 10. The full log is in work/calls.log.

**What I wished existed** (each a capability_gap call; gaps.json):
1. An interval on overall pass rate. `measure_read` with statistics `ci`, `half_width`, `n` returned only the mean; this snapshot holds no readings or contrasts, so no figure anywhere has an interval. This is the gap that shapes the report.
2. `settle`: the size of the re-test.
3. A selection-adjusted reading for the best of many configurations near one gate.
4. Pricing gated fallbacks against the survivor. `price_tradeoff` with `arm_filter` on the survivor returned an empty `against` list.
5. Figure keys for factor effects in a factorial (what a retry or a step of reasoning adds within a model). With no control, `measure_breakdown` and `arm_minus_control` refuse, so the main finding of the design can only be read off adjacent lattice tiles and stated without a number.
6. Task case counts or a declared traffic mix (`n_cases: none` on every task). Also `confound_check`.
7. A cost unit that fits: `usd_per_1k` and block cells print "per 1,000 replies" for a planner query measure the campaign names "Cost per 1k calls". I cited monthly cost in prose to avoid calling a query a reply. The cost bars caption says "per thousand planner queries" because the digit phrase "per 1,000 planner queries" isn't on the allowed list.
8. Reader names on every gate_scatter point. Only the emphasized point gets a name; the rest print raw arm ids, which writing.md bans above the appendix.
9. A prose resolver. I resolved every figure key by passing them as a `block_render` title, which works but is a side door.
10. A definition of pass, and a way to land analyst-raised caveats: `caveats_list` is empty, so my three caveat ids are my own and may not "land" for the mechanical check.

**Confusing or worth fixing.**
- The overview lists `extra_calls` and `outcome` as measures, but both read empty. A measure with no figures should say so in the overview.
- Arm tags skip P (Nano-XS low+retry is Q).
- Lattice cells print cost at three decimals ($1.655, $0.302) while the matching figure keys print two ($1.66, $0.30). The same figure would read differently in prose and chart.
- The cost bars span more than ten times ($0.30 to $3.48), but unlike the gate scatter the bars render notes no log axis.
- `bars` over `tasks` with `reference: floor` draws the overall gate as if it applied to each task; nothing warns that the criterion's scope is overall.
- The brief's evidence form is `{block, query, caption}`, but `reference` and `emphasis` are `block_render` parameters outside `query`. I carried them as extra fields on each evidence entry so the block renders exactly as drawn.
- The `measure_breakdown` refusal carried a useful remedy line but no `recovery:` block.
- No paging was needed; every list fit one page.
