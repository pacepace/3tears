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

## Revision 1

Editor round 1: 11/14, passes, one must_fix and six should_fix. Round 1 files are kept in `round1/`.

| Finding | Severity | What I did |
|---|---|---|
| The explore gate control's lowest band was wrong: at or below 94.8% Nano-XS settings are the cheapest that pass | must_fix | Fixed. I re-ran `decision_sensitivity` over 75–99 and the outcome now gives all nine bands, from Nano-XS off/no retry up to "nothing passes" above 98.6%. In round 1 I had swept only 95–99 and wrote the bottom band as if it were unbounded. That was my error. |
| next[1] claimed a same-size re-run settles best-of-eighteen luck | should_fix | Fixed. It now says what the run does give: a second reading for each Brisk-S survivor, free of the selection, which is the figure a point-estimate gate should be read on. It also says the run does not settle the pick against the line, because at that size it reads under the gate 38.6% of the time even if today's figure is true. |
| c2's block draws no intervals although binomial spreads exist | should_fix | Not fixable with today's blocks. `dot_interval` over `measures` draws a bare dot, and over `configs` it is refused. Already logged as a gap. I rewrote the caption so it no longer says spreads don't exist, and Method now says why the block has no interval. |
| c3 cites the $1,324 gap while its bars show $1,595 per point, and the priced gain isn't separated from zero | should_fix | Fixed. The price bars are replaced with cost-per-1,000 bars for the frontier configurations, which the sentence now cites. The sentence also states the monthly gap and that the high-reasoning option buys no pass-rate gain separated from the pick's (-0.4 to +2.1). The explore price control says neither step's gain is separated from zero, so the prices have no bounded ends. |
| c5 overstated "every model and reasoning level", and the block's rates differ by 1.8 while the prose said +1.9 | should_fix | Fixed. The claim is scoped to Brisk-S, which is what the block shows. It cites the two rates (95.3% to 97.1%) and the lever's interval instead of the effect, so prose and bars agree. The cross-model retry pattern moved to `omitted` with its one exception (Atlas-M high). |
| c4 named only multi-hop; "holds in every task" rested on per-task points with no counts | should_fix | Fixed. It names both tasks (multi-hop, ambiguous entity), and the sentence itself says it is a task-by-task reading with no counts or floor. The watch slot names both tasks too. |
| Never says the extra $1,324 a month buys no measurable gain | should_fix | Fixed in c3 and in the first watch slot ("buys clearance of the line, not a measured gain over the pick"). |
| Invariant: typed counts in captions and next | concern | Fixed. "five" is now a count key, "two" and "three" are removed, and "eighteen" is now `n_candidates`. |
| Invariant: confidence form | concern | Kept `high_if_immaterial`. See below. |
| Invariant: appendix and Source path | concern | Nothing for me to change. The appendix is the renderer's (W3), and Source sits in Method below the claims. |

**Where I disagreed.** The editor suggested "a conditional or unresolved form" of confidence. `high_if_immaterial` is the conditional form, and its material_if already says the caveat is live. `low` would claim the evidence is weak even if the caveat turns out immaterial, and that isn't true here: if J's true rate is at or over 97%, it is the cheapest passer by a wide margin. I kept the form and left the caveat explicitly open.

**Tools this round.** 7 `evals_analyst` calls: `decision_sensitivity` 1; `block_render` 3 (`dot_interval` over `configs`, sent twice because my output filter hid the first refusal: `dot_interval` cannot draw `configs`; and the new c3 cost bars); `figure_keys_check` 3 pages. All 43 cited keys read; none refused. No new gap: the missing-interval gap was already logged. One correction to the round 1 Tools note: `figure_keys_check` did not stop partway. It pages, with page size by output length (25, 14, 4 here), and its `next:` line carries the whole key list again, which is long but works.
