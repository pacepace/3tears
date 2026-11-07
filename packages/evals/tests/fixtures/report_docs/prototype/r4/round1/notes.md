# Reasoning note: persona model tradeoff (r4)

**The eval.** Production (Meridian-4.1, medium reasoning) against three challengers for a persona chat model: f64 and f16 (Meridian-4.1-flash with a 64k and a 16k output cap) and cor (Corvid-3.8-flash, high reasoning). The asked aspect is staying in character, which is judged and has no margin or floor. Two standing guardrails: declining to speculate at least 95% (checked by code) and music facts correct at least 90% (judged). Completing the requested action is checked but not gated. 300 cases and 2,400 trials per arm, with nothing lost. Volume is 12M replies a month. There are blind pairwise ratings of Production against f64 by conversation length. Nothing ranks cost against character.

**What I noticed first.**
- Every challenger is cheaper than Production and every one is clearly worse at staying in character. No option is better on the asked aspect, so the question is what the saving buys.
- Two challengers are out on a guardrail. cor fails speculation (92% against 95%, and clearly worse than Production at -5.3). f16 fails music facts on its point estimate (89.4% against 90%).
- That leaves Production and f64. `options_dominance` agrees: two survivors, no dominant option, mode tradeoff.

**The trap.** cor looks like the natural middle path. Its cost is between f64's and Production's, and it loses the least character (-2.2). A reader pricing "a little less character for a little less cost" would land on it. But speculation is a standing guardrail that cor misses clearly. The interval on its change sits well below zero and its rate is under the floor. It is not on the menu, and the page has to say so before anyone does that arithmetic. A second, quieter trap: f64 "passes" both guardrails only on point estimates. It clears speculation by 1.4 points and facts by 2.1, and its change against Production on each could be larger than that clearance (low ends -2.8 and -2.7). Without that, the trade reads as safer than it is. (I knew the exemplar index summary of this campaign's trap, since INDEX.md names it. I did not open r4-tradeoff.md.)

**Story.** Tradeoff. "Switching to f64 saves $7,560 a month but stays in character 6.1 points less often, with most of the loss in long conversations." The flag says nothing was declared in advance and names the persona owner as the one who decides. Confidence is high if the caveats aren't material:
- **f64's guardrail clearance:** material if either true rate sits under its floor, because then Production is the only survivor and the report becomes "keep Production".
- **The judge isn't described in the snapshot:** material only if it misreads character differently per arm. Cost is measured, and the blind ratings lean the same way.
- **Contrasts are uncorrected across three challengers:** material only if correction moved an in/out verdict, and the deciding intervals sit far from zero.

**Lead evidence.** Two blocks side by side: the forest of character change against Production, and cost bars. The trade has two axes, and I wanted them in one scatter_frontier. It refuses: the measures source has no judged persona, and the judged source has no cost. The price per point ($1,239 a month, band $1,036–$1,543) goes in prose by its price keys, because no block reads price_tradeoff.

**What I looked for and found.**
- **Where the loss lives.** Character by conversation length. f64's gap grows steadily: -2 at 1–2 turns, -4.5, -11, then -17.5 at 11+. Conversations of 6–10 and 11+ turns are 30% of cases and carry 64% of f64's loss. This is the seam for a mix. It is drawn as lines on an ordered axis, with Production and f64 only, and labelled exploratory.
- **Do listeners notice?** The blind ratings exist only for Production against f64. Raters prefer Production at 61% in 6+ turn conversations and 52% in 1–5. That supports the judged pattern from a different instrument. No interval was stored, so I could not say either share is separated from no preference. The claim states direction only.
- **Delivery beside the latency win.** f64 is faster (p95 1.6 s against 2.4 s) and not separated on task completion (+0.2, -0.1 to +0.5). The latency win isn't bought by declining work. It is a detail claim on f64's side of the trade.
- **How the call moves with the guardrail floors** (`decision_sensitivity` on both thresholds). Raising either floor past f64's value leaves Production alone. Loosening it far enough lets cor or f16 back in, and the result is still a tradeoff. These go in explore controls.

**What I left out, and why.** Each item is in `omitted`:
- The scenario strata: named, but no figures.
- Failure examples: none frozen.
- f64 against f16 as an output-cap lever: a lead for configuration, and f16 is out.
- cor's strengths on facts and character: gated, so they don't offset.
- Stratum half-widths: precision of an exploratory breakdown, so appendix.
- Gated arms' monthly cost.

**What I'd watch.** f64's two guardrail clearances, on point estimates with no stored interval. This is the weakest margin and it sits on the cheaper survivor. Also the mix's own risk: switching model mid-conversation may itself break character. That is in the mix's next step.

**Next steps, each with what it settles.**
1. Re-run f64's guardrail cases with more cases. This settles whether the trade exists at all, and could reverse it into "keep Production". I couldn't name a case count (`settle` isn't built).
2. Have the persona owner declare a price per point. This settles the trade against the price line with no further run.
3. Run the depth-routed mix (f64 early, Production from the 6–10 range). This settles its cost and whether it keeps long-conversation character. It isn't priced because nobody ran it.
4. Record the judge and calibrate it. This settles the size of the loss, which sets the price per point.

**Lean.** Mine, labelled as such: keep Production until the mix runs, because the loss raters notice is where the mix keeps Production. If the call can't wait, the price line decides it. I kept this out of the headline.

**What doesn't transfer.** The mix is suggestible only because the loss has a clean, routable seam (conversation length) and listener evidence points at the same seam. A loss spread evenly across strata would leave a plain two-option trade. The guardrail trap depended on cor missing the floor clearly. A challenger only borderline under a floor would need its own Watch line, not a flat exclusion. f16 is close to that case. Its miss is 0.6 points on a point estimate, so its claim says "on its point estimate".

## Tools

- **Calls.** 56 evals_analyst calls of the 80 budget: help ×7 (actions, figure_keys, two key groups, three action pages), campaign_overview ×1, readings_list ×1, contrasts_list ×1, caveats_list ×1, options_dominance ×1, measure_read ×10, measure_breakdown ×1, price_tradeoff ×1, decision_sensitivity ×2, examples_list ×1, block_catalog ×1, block_render ×17 (3 refused), capability_gap ×11. There was no paging: every listing fit one page.
- **Wished-for capabilities** (each is a capability_gap call, and in gaps.json):
  1. A scatter_frontier that pairs a judged dimension with measured cost. The obvious lead for a tradeoff is refused from both sources.
  2. A block that reads price_tradeoff. The deciding figure reaches the page only as prose.
  3. Intervals on each arm's guardrail rate against its floor. `measure_read` with ci/half_width returns nothing for arm-level rates, and dot_interval draws no interval. This is the gap that most limits the report, because the main caveat is f64's clearance.
  4. A figure key for human_pref slices, with an interval and count. The only listener evidence can't be cited in prose.
  5. A no_preference reference line on human_pref. It is refused, though the block's own note says 50% is no preference.
  6. Strata blocks that carry share, loss share and interval bands. The snapshot holds them, but lines and tables show means only.
  7. `settle`, for case or rating counts.
  8. Pricing or bounding a depth-routed mix.
  9. A figure-key resolver to check prose keys before submission.
  10. A judge description in the snapshot.
  11. Failure examples, since the pool is empty.
- **Confusing.**
  - `readings_list` returns nothing here, because there's no margin or floor on the asked aspect. The overview doesn't say that up front, so I spent a call finding out.
  - `measure_read` with no `statistics` returns no figures for latency (there is no mean). I needed a second call naming p50/p95.
  - `measure_read` on persona by stratum with `ci` returns half_width only. It silently omits ci and n_cases, with no note about what's missing.
  - The diverging_bars note says "Bold rows are separated", but the values table has no bold marker, so I can't tell which rows are bold.
  - `criteria.<c>.<arm>.low/high` are documented key fields, but I found no action that shows whether they exist here.
  - `caveats_list` is empty although the judge is undescribed and contrasts are uncorrected. I minted my own caveat ids, and I don't know whether the mechanical check accepts ids that caveats_list didn't issue.
- **Refusals.**
  - scatter_frontier, twice: on the measures source and on the judged source.
  - diverging_bars with reference no_preference.
  - Each recovery line named the faulty parameter clearly.
- **Environment.** The shared scratchpad held another session's helper script with the same name I first used, and it was overwritten mid-run. I moved to a uniquely named scratch folder. Two of the 56 calls ran through that first script.
