# Reasoning note: query planner config sweep, one gate (r2 with one gate)

**The eval.** A research query planner swept over three models (Atlas-M, Brisk-S, Nano-XS) × three reasoning levels (off, low, high) × retry (none, one): eighteen configurations, 1,200 cases each across four tasks (single lookup, multi-hop, date and unit math, ambiguous entity). No control. One criterion declared: overall pass rate at least 97%, read on the point estimate. Cost per 1,000 calls and p50/p95 latency measured; volume 4M calls a month. No latency budget, no per-task floor, no price per point. The snapshot states no interval on any pass rate; the tools derive binomial spreads from the counts.

**What I noticed first.**
- Five configurations clear the gate. `options_dominance` picks Brisk-S · low reasoning · one retry (J) at $0.30 per 1,000 calls: it dominates on gates and cost.
- J clears by 0.1 point (97.1% against 97%). It is the weakest margin on the page.
- The next survivor up the cost frontier, Brisk-S · high · one retry (L), clears by 1.0 point at $0.63, about $1,324 a month more at the declared volume. Atlas-M's survivors cost $1.66 to $3.48.
- Retry raises the pass rate at every model and reasoning level, separated from zero in eight of nine places, for a few cents per 1,000 calls. It is what lifts Brisk-S over the line.
- Nano-XS fails everywhere; its best (94.8%) is a settled miss.

**The trap.** Reading "passes" as "passes". J clears a gate on a point estimate by 0.1 point, after being the cheapest of eighteen readings that landed above the line. `settle_cases` says a fresh run of the same size reads it under the gate 38.6% of the time, and settling it would take about 38,000 cases at even odds. Code's call (J) is right under the declared rule, but a page that says "J passes" without the size of its clearance would sell a coin flip as a gate. The second trap is the opposite: overriding the rule and recommending L. The principles say a borderline pass counts as passing for the call, and the choice between J and L is really a question about what the gate means, which only its owner can answer. So: recommend J per the declared rule, put the clearance on the first screen (headline and lede), name L as the fallback with its monthly price, and make the owner's declaration the first next step.

**Story.** Decision. "Use Brisk-S · reasoning low · one retry (J), the cheapest configuration over the 97% pass-rate gate, though it clears it by only 0.1 point." Lede: the re-run odds and L's price. Confidence `high_if_immaterial`: point_estimate (material if J's true rate is under 97%; then L is the pick; cost, the other half, is measured), best_of_n (same direction; L far less touched), no_control (only for claims of one configuration beating another). Nothing is code-checked immaterial. I considered `low`, but if the point-estimate caveat is immaterial the evidence for J is strong (cheapest by a wide margin), which is what `high_if_immaterial` describes.

**Lead evidence.** The model × reasoning × retry lattice: each tile carries pass rate, cost, p95 and gate status, so it shows the pick, why Atlas-M is out (cost) and Nano-XS out (gate), and the retry pattern in one grid. Second lead: the pass-rate dot plot of the five survivors and the two nearest misses against the gate line, which shows how tightly they pack around it. The re-run odds carry the uncertainty in the sentence, because the dot plot cannot draw the interval (gap logged).

**What I looked for and found.**
- Whether the clearance is more than noise: `settle_cases` on J (38.6% wrong side; ~38,350 cases to settle at even odds) and on L (0.9% wrong side). This decided the shape of the story.
- Whether L is actually better than J: `arms_compare` gives -0.8 points (interval -2.1 to +0.4), not separated, unpaired. So L buys clearance of the line, not a demonstrated gain. The claim is worded that way.
- What L costs beyond money: p95 870 ms against 513 ms. No budget, so it is a fact beside the price, not a gate.
- Where J is weak: multi-hop 95.6%, ambiguous entity 96.4%; L holds 97% or more in every task. Descriptive, no per-task floor; the one multi-hop example card shows two hops collapsed into one search.
- Where the call flips on the threshold (`decision_sensitivity`): up to 95.3% Brisk-S low without retry; to 97.1% J; to 98% L; to 98.6% Atlas-M high one retry; above, nothing. Goes in explore.
- Whether Atlas-M buys pass rate: the model lever, Atlas-M to Brisk-S at the same settings, is not separated wherever Atlas-M passes (e.g. -0.8, -2 to +0.5 at low reasoning, one retry). It costs about five times as much.
- Retry × model: retry helps Nano-XS more than Atlas-M (difference +5.1, separated). Real, but on a model that is out.

**What I left out, and why.** (Each in `omitted`.) Atlas-M's survivors as a lead (dominated on cost, not separated on pass rate); Nano-XS beyond one line (settled miss); near misses under the gate (none cheaper than J, so a true pass changes nothing); the reasoning lever as a main effect (the lattice shows it; the call turns on retry); the retry × Nano-XS interaction; calls per task (mechanism; cost prices it); outcome and extra-calls (no figures held); the cost-vs-pass-rate scatter (same points as the lattice); other example cards; per-task rates for other configurations.

**What I'd watch.**
- J's clearance, first. It is the weakest margin and sits on the recommendation itself. More cases cannot practically settle it, so it is a declaration, not a run.
- J's multi-hop rate. The overall gate averages it away; if multi-hop traffic grows as a share, the overall rate falls with it.

**Next steps, each with what it settles.**
- The gate's owner declares whether 97% must hold beyond noise or on the point estimate: settles J against L directly.
- Re-run J and L on fresh cases at today's size: removes the best-of-eighteen luck from J's reading; if it reads under 97%, L is the pick.
- Give low-reasoning Brisk-S an explicit hop-by-hop plan for multi-hop questions and re-run: settles whether the cheapest configuration can clear with room.

**What doesn't transfer.** The story here came from one thin clearance on one declared gate with no interval held; `settle_cases` turned it from a hunch into a figure. Where the pick clears by more than its noise, this is a plain decision. The cheap fallback existed because the cost frontier had a near neighbour one step up; with a wide gap to the next survivor, the declaration would carry a much larger price.

## Tools

**Calls.** 49 `evals_analyst` calls of the 80 budget: primer and 3 help topics (overview help, `block_render`, `figure_keys`); `campaign_overview` 1; `options_dominance` 1; `caveats_list` 1; `measure_read` 7 (pass rate by arm, by task over two pages, cost, latency, calls per task, outcome, extra calls); `decision_sensitivity` 1; `settle_cases` 4 (one refused); `arms_compare` 1; `levers_effects` 2; `price_tradeoff` 1; `block_catalog` 1; `examples_list` 2 (one refused); `block_render` 14 (one refused, 7 final renders for blocks.txt); `figure_keys_check` 3; `capability_gap` 4.

**What I wished existed** (each a `capability_gap` call):
- Intervals on `dot_interval`/`bars` over pass rates from their binomial counts. `spread.<arm>.pass_rate.overall.se` exists and `arms_compare` and `levers_effects` already use it, yet the block meant for "a rate against a floor with its uncertainty" draws a bare dot. On this campaign, where the whole story is a 0.1-point clearance, that is the biggest gap.
- A block source over `settle_cases` rows (wrong side, cases needed per arm). The decisive evidence lives only in prose.
- A block source over `levers_effects`. The retry claim rests on bars of arm means, not the effect and its interval.
- Task case counts, and a share-of-misses figure that works without a control.

**Confusing or rough.**
- Arm ids: outputs print `arm:brisk-s.low.r1`, but `settle_cases` refuses that form and wants `brisk-s.low.r1`. The refusal's `recovery` listed the valid values, so it cost one call.
- `examples_list.arm_filter` takes a string, while `block_render`'s `query.arms` takes a list. Another refused call.
- `scatter_frontier` with `reference: floor` over two measures is refused ("name that one measure"), but drawn without a reference it still adds a `floor 97%` row. It is unclear whether the drawn chart shows the gate line or not.
- `price_tradeoff` prints the J→L gain as +0.8 while the two rates read 97.1% and 98%, a rounding seam a reader will notice. I cite the monthly cost gap rather than the gain.
- `measure_read` for `outcome` and `extra_calls` returns 18 empty rows with `held: []`. One line saying "no figures held" would be clearer.
- `figure_keys_check` with ~29 keys appeared to stop partway (I filtered the output and could not see a `next:` line), so I re-checked the rest in a second call.
- No refusal I hit lacked a usable `recovery`. Paging on `measure_read` by task worked as described.
