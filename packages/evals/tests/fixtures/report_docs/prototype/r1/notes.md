# Reasoning note: Concierge model selection (r1)

**The eval.** A concierge chat assistant. Four models (Atlas-L, Atlas-M, Brisk-S, Open-70B), all challengers, no control, each run on the same 1,190 cases across five tasks: billing and account questions (400 cases, 42% of traffic), multi-turn troubleshooting (180, 19%), persona small talk (220, 24%), research-needed questions (240, 15%) and speculation bait (150, applies to all traffic). Eleven gates, declared with the campaign: an overall quality floor of 85 and a p95 latency budget of 3 s in every task, plus task gates (troubleshooting resolves, steps, confirms, escalates; persona stays in character; research tool use, citation, no unsupported claims; speculation declines at 95%). Every gate but latency is judged. No judge, calibration, repeats, monthly volume or prior insights are stated.

**What I noticed first.**
- `options_dominance` leaves exactly one survivor, Atlas-M, marked borderline on troubleshooting resolution (90% against 90%, ±4.4).
- Atlas-L is out on latency alone: p95 over 3 s in four of five tasks (4.86 s on research). It passes every quality gate.
- Brisk-S and Open-70B are far cheaper ($6.01 and $11.51 per thousand conversations against $31.37) and fail several task gates each. But each also passes some tasks outright: Brisk-S passes billing, the largest task; Open-70B passes persona.
- Both fail speculation, and the gate matrix marks speculation bait "applies to all traffic".

**The trap.** Routing. Billing is 42% of traffic, Brisk-S passes it at a fifth of Atlas-M's billing cost, and persona passes on Open-70B. A per-task "cheapest passing model" table invites a routed setup that looks like large savings. But speculation bait is not a task users choose; it can arrive inside any conversation, and Brisk-S (81%, interval top near 87%) and Open-70B (89%, top near 94%) both miss the 95% gate clear of their intervals. A model that fails it can't take any traffic, so the routed setup is not an option. The gate matrix already encodes this (its "Cheapest passing" mark sits on Atlas-M in billing and persona). `decision_sensitivity` on the speculation threshold, by contrast, reports "call unchanged" at every flip, because it reads only the single-model call, where Brisk-S and Open-70B are out on other gates anyway. Read naively, that output says the speculation gate doesn't matter; it is the gate that blocks the cheap route.

**Story.** Decision. "Use Atlas-M for all traffic: it is the only model that passes every gate in every task." Confidence high, if two caveats are immaterial:
- **The judge is unstated.** Every quality gate is an absolute judged rate, which needs a calibrated judge. Material only if the judge scores the Atlas models more kindly than the cheap ones; the cheap models miss by more than their intervals, so a uniform bias would not flip it.
- **Atlas-M's troubleshooting resolution is borderline.** Exactly on the line. Material if the true rate is below 90%: then nothing passes, and the decision becomes which gate to relax.
- First-screen flag: no control was named, so everything is read against gates, not against production.

**Lead evidence.** The gate matrix (model × task, pass/fail with the binding reason and cost per cell). It shows the call, all three rejections and the "applies to all traffic" note in one view. Then speculation declines as dots with intervals against the 95% floor, which is the routing blocker; then latency range bars against the budget, which is Atlas-L's only failure; then the troubleshooting resolve dots, the pick's weakest margin.

**What I looked for and found.**
- Whether the cheap models' speculation misses are borderline. They are not: both intervals sit wholly under 95%. More cases won't help them; only a change to the models would.
- Where Atlas-M is thin besides resolution. Latency on research (2.84 s, 0.16 s under budget) and troubleshooting (2.73 s), with no interval on p95 and no stated test conditions. Its speculation rate (98%) clears the floor at the interval's low end too (95.8%), so it is not borderline.
- Whether raising the latency budget changes the call. Above 4.86 s Atlas-L passes too, and it outscores Atlas-M on overall quality in every task; code still calls Atlas-M dominant because only cost is weighed and no quality axis was declared. I put that in Explore as a trade no declared rule ranks, not as code's verdict.
- Whether tightening the resolve gate admits anyone: no; above 90% nothing passes.

**What I left out, and why.**
- Per-task overall scores for all models, and the gates Atlas-M clears with room: none bind; the gate matrix holds them.
- Open-70B's borderline overall quality in billing: it is out on three other tasks' gates regardless.
- A cost chart: the spread is over 40 times, bars have no log axis, the gate matrix carries per-cell cost and the decision carries the all-traffic figures.
- Any figure for a routed mix: never run, and blocked by the speculation gate.
- Failure examples: the snapshot froze none.

**What I'd watch.**
- Troubleshooting resolution on Atlas-M, the weakest margin (gap 0).
- Atlas-M's research latency headroom, which has no interval; and the unstated latency conditions, which also decide Atlas-L's exclusion.

**Next steps, each with what it settles.**
- Improve Atlas-M's troubleshooting resolution (prompt, tools, escalation flow) and re-run that task: settles whether the weakest gate clears with room. With the estimate exactly on the line, a bigger sample only narrows the interval around the line.
- Human ratings on troubleshooting resolutions and speculation replies against the judge: turns the judge caveat into evidence or reopens the call.
- Declare how latency was measured, re-measure on dedicated capacity if shared: settles whether Atlas-L's exclusion and Atlas-M's thin headroom are real.
- If cost matters: fix Brisk-S speculation handling, re-run speculation bait, then run billing-to-Brisk-S as its own measured arm: settles whether a cheaper routed setup is allowable and what it saves.

**What doesn't transfer.** The routing trap existed because one gate measured a behavior that occurs inside every task, while being scored as a separate task. Where every gate is truly task-local, per-task routing to the cheapest passing model can be the right answer. The single survivor made this a decision; had Atlas-M's resolution estimate landed a point lower, the honest report would have been "nothing passes; here is which gate is nearest".

## Tools

**Calls.** 53 calls to `evals_analyst` (budget 80), inside roughly 40 tool calls of my own (several CLI calls batched per Bash call). By action: help and primer 10 (primer, help, figure_keys, measure_read, block_render, decision_sensitivity, figure_keys.gates/cost/measures/slices); campaign_overview 1; options_dominance 1; caveats_list 1; readings_list 1; contrasts_list 1; measure_read 7; decision_sensitivity 3; examples_list 1; block_catalog 1; block_render 18 (8 table renders used only to check figure keys through `title`, 4 of them refused; 6 exploratory renders, 1 refused; 4 final renders for blocks.txt); capability_gap 8. Five refusals in all, each with a clear reason. No `price_tradeoff` (one survivor, nothing to trade).

**What I wished existed** (each also a `capability_gap`):
- `settle` for the borderline resolve gate.
- Interval ends as figure keys: `criteria.<c>.<arm>.low/high` and `task.<t>.<arm>.<m>.ci` are refused as "not stated", although `dot_interval` draws the ends; I cite value plus half-width.
- A block source for an arm's all-traffic cost (`cost_per_1k.<arm>` exists as a key but `bars` over `measures` is refused as empty).
- A routing check that applies cross-cutting gates to a per-task assignment; `decision_sensitivity` reads only the single-model call.
- Code-raised caveats for an unstated judge and for the pick clearing a gate only on its point estimate; `caveats_list` is empty here.
- A "per 1,000 conversations" unit phrase: the measure is named per conversation, while `usd_per_1k` and the gate matrix print "per 1,000 replies".
- A log axis for bars spanning more than ten times.
- `confound_check` for the unstated latency conditions.

**Confusing in help, refusals or paging.**
- `measure_read` with the default `by: arm` returns zero rows for every measure, and the overview lists every arm under `arms_without`, although `options_dominance` cites `cost_per_1k.atlas-m`. The figures exist only by task here.
- `measure_read` on `latency_s` by task with default statistics returns rows with no figures at all; only an explicit `statistics: [p50, p95]` brings them. An empty-figures row should say which statistics the measure holds.
- No action resolves a figure key directly. I validated keys by passing them in a `block_render` table `title`, which works but costs a call per batch and fails on the first bad key only.
- `judge.tier` is refused with "which no judged dimension states": clear, but it is the only way to learn the judge is unstated; the overview doesn't say so.
- `cost_per_month` is refused for lack of `meta.volume_per_month`: clear and useful.
- The brief says evidence is `{block, query, caption}` "exactly as passed to block_render", but `reference` and `emphasis` are block_render parameters outside `query`; I carried them as sibling fields of the evidence entry. `emphasis` is not echoed in the rendered output, so blocks.txt can't show it took effect.
- Row labels differ by kind: `dot_interval` and `range_bars` print "B · Atlas-M", `bars` prints "Atlas-M" without the tag; `dot_interval`'s `level:` field is empty while its column header says "95% interval".
- No paging was needed; every result fit one page.

## Revision 1

Round-1 files are in `round1/`. The editor scored 12/14 (passes), with 1 must_fix, 5 should_fix and 4 invariant concerns. Each finding and what changed:

- **must_fix: the judge caveat misstated when the call reverses.** Accepted; I had it backwards for Atlas-M. Its resolve gap is 0, so any leniency on resolution, uniform or not, can remove it. The clause now says that, citing `criteria.trouble.resolves.atlas-m.gap`. I dropped "the cheaper models miss their gates by more than their intervals" and kept only the true part: each misses the speculation gate beyond its interval (c2's block shows both upper ends below 95%), and Brisk-S also misses troubleshooting resolution beyond its interval. So only a bias against them in particular would readmit them.
- **should_fix: the headline is unqualified.** It now ends "with troubleshooting resolution exactly on its line". The recommendation says that gate clears only on its point estimate.
- **should_fix: resolves_borderline left "which gate to relax" unanswered.** The clause now answers it from Explore: relaxing the resolve gate admits no other model; relaxing latency admits only Atlas-L, whose resolve rate (`criteria.trouble.resolves.atlas-l.value`) is just as borderline and which costs more in every task. So Atlas-M stays the practical pick unless latency is relaxed.
- **should_fix: "Atlas-L passes every quality gate".** It now says "on its point estimates, with a troubleshooting resolve rate as borderline as Atlas-M's".
- **should_fix: "more cases cannot clear it".** Accepted; the round-1 reasoning assumed the true rate equals the observed one. Both next texts now say more cases show which side of the line Atlas-M is on (the flip condition in the lede), and only a handling fix gives it room. The first next step now runs the fix and more cases together.
- **should_fix: cost basis unresolved.** I settled it on the campaign's own declaration: the measure is named Cost per 1k conversations. The prose now uses that name (a measure name may carry its digits), and drops "per thousand conversations". The gate matrix's "per 1,000 replies" is fixed by the renderer and no block parameter changes it, so I can't relabel the block. Rebuttal, which the editor can check: `measure_read` on `cost_per_1k` by task returns exactly the matrix's per-cell values (for example Atlas-M billing $24), so the label is a phrase on the same figures, not a second unit. The c1 caption and the Method entry both say this. Logged as a gap.
- **Invariant 1: number words.** Removed "four of five tasks" (why_not, c3, explore), "five tasks" (Method) and "the three research gates" (omitted). They are now phrased as the exception ("every task but persona small talk") or as a list of names. "single-model" in explore became "the call for a model on all traffic". The remaining "one" are pronouns ("a small uniform one", "no one else"). Logged a gap for count keys.
- **Invariant 3: analyst-raised caveats.** I can't confirm through the tool that the store accepts caveat ids it never raised; `caveats_list` is empty and no action registers a caveat. Both caveats stay as analyst-named, each `material_if` now resting on figure keys. Logged as a gap; this is the coordinator's or store's call to confirm.
- **Schema drift** (top-level `flag`, `evidence` as a list, `reference` and `emphasis` on evidence): this follows the brief's own decisions, which override ANALYST §5 for this prototype. I left it unchanged; ANALYST §5 should be updated, or the brief changed, and that is not mine to edit.
- **Invariant 6: appendix.** There is no page renderer yet (W3), so I can't produce the appendix. Method is present; the full-results appendix waits on the renderer.

**What I could not do.** I couldn't relabel the gate matrix's cost unit (renderer-fixed), confirm how analyst caveats are stored, produce the appendix, or cite interval ends as keys (still refused; the caveats cite the gap and the half-width instead).

**Tools this round.** 9 calls to `evals_analyst` (2 key checks through a `block_render` title, both clean; 4 final block renders; 3 `capability_gap`), inside about 9 tool calls of my own, well under the 40 allowed. No refusals. The four blocks' queries didn't change, and their rendered output is identical to round 1. Three new gaps were appended to `gaps.json`: count keys, registering analyst caveats, a block unit label from the measure name.
