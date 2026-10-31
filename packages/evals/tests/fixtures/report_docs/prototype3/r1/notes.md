# Reasoning note: Concierge model selection (r1)

**The eval.** Four candidate models for a concierge chat assistant (Atlas-L, Atlas-M, Brisk-S, Open-70B), no current model as a control. 1,190 cases each across five tasks: billing (400), multi-turn troubleshooting (180), persona small talk (220), research-needed questions (240) and speculation bait (150), the last marked as applying to all traffic. Gates, all declared before the run: an overall quality floor (85) and a p95 latency budget (3 s) in every task, plus each task's own pass-or-fail gates (resolves the issue, clear steps, confirms before account changes, escalates, stays in character, research tool use, cites, no unsupported claims, declines to speculate). Overall quality has a calibrated judge (200 ratings, 86% agreement). The pass-or-fail dimensions don't state their judge. No monthly volume was declared.

**What I noticed first.**
- Only one model survives the gates: Atlas-M. `options_dominance` agrees and calls it dominant on gates and cost.
- Its survival rests on one figure: it resolves 90% of troubleshooting issues against a 90% line, ±4.4. That is the estimate sitting exactly on the line, not a thin clearance. `settle_cases` refuses to size it: a fresh run lands on either side alike.
- Atlas-L is the best model on quality (higher overall quality than Atlas-M in four of five tasks, separated) but runs over the latency budget in four tasks, by up to 1.86 s on research. It also costs about 2.4 times as much.
- Brisk-S and Open-70B are far cheaper and fail many quality gates. Both fail declining to speculate, which applies to all traffic.
- Atlas-M's second-nearest line is latency: research p95 is 2.84 s against 3 s, and p95 carries no interval.

**The trap.** Two of them. First, "the only survivor" reads as decisive, and the gate matrix pill says "Borderline", which sounds like a near pass. In fact the estimate is on the line, a coin flip on a re-run, and the pick of four reads high by selection. The resolve gate has to be on the first screen, in the headline, not in a Watch footnote. Second, the gate matrix shows Brisk-S passing billing and Open-70B passing persona at a fraction of Atlas-M's cost. That invites a routed mix. Speculation bait applies to all traffic and both cheap models fail it, so no route exists. The caption has to say that, because the matrix doesn't.

**Story.** Decision. "Use Atlas-M: it is the only model inside every gate in every task, but it resolves troubleshooting issues exactly at the line." Confidence high if immaterial, with four open caveats: best_of_n (carrying the on-the-line resolve rate, since no caveat id exists for a borderline gate), judge_unstated (material because the deciding gate is a judged pass-or-fail figure with no stated judge), point_estimate:p95 (research latency), no_control (only touches what Atlas-M gives up against Atlas-L, not the call). I considered a diagnosis: if the resolve rate is truly under 90%, nothing passes and it's a tradeoff between resolving and latency. I kept the decision because the call as declared holds today, nothing else comes close on the declared gates, and the lede states what flips it.

**Lead evidence.** The gate matrix first: it shows in one table why every rival is out and carries cost per task. Then the resolve rate as dots with intervals against the 90% line, so the reader sees Atlas-M on the line with an interval reaching well below. Then latency range bars for Atlas-L and Atlas-M against the budget: why Atlas-L is out, and how close Atlas-M's research p95 sits. Then the Atlas-M minus Atlas-L comparison on overall quality per task: the recommended option's weak axis, what the latency budget costs in quality.

**What I looked for and found.**
- Whether the call moves with the latency budget. Tighten it below 2.84 s and nothing passes; loosen it to 4.86 s and Atlas-L passes too. Code's call doesn't change there because it weighs only gates and cost, but a decider would face a real trade: about three points of overall quality for about $44 more per 1,000 conversations, roughly $15 a point (unpaired, $11 to $22). That goes in explore, labelled as a what-if, since Atlas-L is gated today.
- Whether the call moves with the resolve line: any line above 90% leaves no model passing.
- Whether a routed mix helps: no, because of the all-traffic speculation gate. Code's per-task picks agree (Atlas-M in every routable task).
- Failure examples to show what an unresolved troubleshooting conversation looks like: none kept for any model.

**What I left out, and why.**
- Atlas-M's margins on its other gates: all clear the line by more than the resolve rate does.
- Single-task cheap routes (Brisk-S billing, Open-70B persona): ruled out by the all-traffic speculation gate.
- Open-70B's borderline billing quality: it's already out.
- A separate cost chart: the gate matrix carries cost per task, and no block draws pooled cost per model.
- Median latency, pooled overall quality, monthly cost (no volume declared).

**What I'd watch.**
- The resolve rate on its line (interval 85.6% to 94.4%), and that it was picked as the one of four that passes.
- Research latency clearance of 0.16 s with no interval; troubleshooting is 0.27 s.
- The resolve figure comes from a judge whose identity and calibration are unstated, exactly where the call is decided.

**Next steps, each with what it settles.**
- Improve Atlas-M's troubleshooting resolution and re-run that task on fresh cases: settles whether it clears with room; a miss turns the decision into a resolve-versus-latency trade.
- Run Atlas-M as it is on a larger troubleshooting set: settles the side of the line only if the true rate sits away from it.
- Name and calibrate the judge for the pass-or-fail dimensions: turns the judge caveat into evidence on the deciding figure.
- Measure p95 with an interval on research and troubleshooting: settles the thin latency clearance.
- Declare monthly volume, and a price per point of quality if the budget may relax: prices Atlas-L as a real option.

**What doesn't transfer.** The on-the-line estimate is an unusual case: nothing sizes it, so the next step leans on a change to the model rather than more cases. The routing trap depended on a task marked as applying to all traffic; where no such gate exists, a routed mix is a real next experiment.

## Tools

- **Calls.** About 40 tool calls in all (reads of the skill and exemplars, Writes, and Bash), carrying 41 invocations of `evals_analyst`, several batched per Bash call: primer, help (index, `figure_keys`, `block_render`), `campaign_overview`, `options_dominance`, `caveats_list`, `measure_read` ×6 (two refused), `decision_sensitivity` ×2, `settle_cases` (refused), `arms_compare` ×2, `price_tradeoff` ×3 (one refused), `examples_list` ×2 (one refused), `block_catalog`, `block_render` ×7 (two refused), `figure_keys_check` ×4, `capability_gap` ×6.
- **Wished for (each logged as a capability_gap).** A settle sizing for an estimate exactly on its line, against an assumed true rate. A caveat id for the pick's borderline gate. An interval on p95 latency. Failure examples for the borderline slice (none frozen). A gate-matrix state for "passes here, barred by an all-traffic gate". A bars block for each model's pooled cost.
- **Refusals.** `measure_read by=arm` on cost and resolves: held per task only; the recovery named `task`, clear. `examples_list` and `price_tradeoff` took `arm_filter`/`what_if` as a string, not a list; the SKILL table's plural-looking names (`arm_ids` is a list) led me to pass lists. `bars` over `scoreboard` refused for naming no scoreboard figure, but the query has no field to name one in; `bars` over `measures` with `cost_per_1k` selected nothing, with no hint that cost lives under `tasks`.
- **Confusing.** The gate matrix labels Atlas-M "Cheapest passing" in billing while Brisk-S's cell there says "Passes" at a fifth of the cost; the all-traffic rule is only visible on the speculation row. The compare block's Note column repeats the value, and its intervals print four decimals (-3.556 to -0.4444) where the table elsewhere prints one. Cost cells mix "$41.0" and "$17" formats. `decision_sensitivity` lists per-task picks for four tasks and silently omits speculation bait; I inferred it is non-routable. `settle_cases` refusing outright was right, but the `wrong_side` key's refusal text ("lands on either side alike") was the more useful statement and only reachable through `figure_keys_check`.
- **Paging.** No result needed a second page except one `figure_keys_check` over 31 keys (17 per page); passing `limit` avoided it the second time.
