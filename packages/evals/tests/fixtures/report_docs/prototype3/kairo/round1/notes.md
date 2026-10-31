# Reasoning note: Kairo persona model, round 2

**The eval.** The production persona model (September) against two candidates, qwen3.8-flash and 1017, on one asked aspect: character adherence on the v2 set, with the promoted queue-wording and AI-lane directives. 299 conversations, three repeats each, paired. The question is a non-inferiority one: does either candidate keep Kairo in character at least as well, without being worse by more than 15 points? The aspect, control, margin and floor were all declared on 25 Oct, after the results. The judge is uncalibrated (no human ratings). No volume and no price per point were declared.

**What I noticed first.**
- Both candidates read "within_margin": qwen3.8-flash down 7.6 points (clearly worse, interval -10.8 to -3.4), 1017 down 3.0 (not separated, -5.8 to +1.1).
- Both candidates cost more, not less: 27% and 28% above September per reply, corrected contrasts clear of zero.
- `options_dominance` has September dominating both, and the margin sweep holds at every margin.
- One prior insight says qwen3.8-flash "significantly lowers production costs". This run says the opposite.

**The trap.** The question invites a yes: both candidates are "within the 15-point margin", so a report built from the verdicts reads as "either candidate is acceptable", and the prior insight then nudges toward qwen3.8-flash as the cheaper one. But clearing a non-inferiority margin only matters when the switch buys something. Here it buys nothing: both cost more and both are lower in character. The second trap is the size: "within the margin" said first would hide that qwen3.8-flash is clearly worse. The page leads with the size, then the margin.

**Story.** Decision. "Keep September: both candidates cost 27–28% more per reply and stay in character less often, so clearing the margin buys nothing." Confidence is high_if_immaterial. The deciding finding, cost, is measured telemetry, so the uncalibrated judge and the late declarations do not bear on the call. Code checked the late margin and floor as immaterial. The open caveats are the judge, the late aspect and control, the simulator-settings apparatus caveat, 1017's reasoning-share confound, the model confound, and qwen3.8-flash's two lost trials. Each has a material_if clause saying why it does not move the call.

**Lead evidence.** Two lead claims, side by side, because the call has two reasons. First, the readings forest with the margin band: size first, the margin second. Second, cost bars per 1,000 replies, where September's interval sits clear below both candidates. Then the corrected contrasts forest (in-character rate and pass rate at 99.69%). It shows 1017 clearly lower on the judge's in-character rate even though its pass-rate drop is not separated. Without it, "1017 not separated" would read as "no change".

**What I looked for and found.**
- Where the loss sits. Off-lane asks (non-music requests, the largest stratum at 38 cases) carry about half of each candidate's in-character loss (48% and 53%). It is the only one of the four largest strata where either drop is separated from zero, and only before correction. That is a lead, not a finding.
- What those failures look like. In the kept cards, the persona brushes off the non-music ask and then answers it anyway: jeans washing temperature, an opening line for a landlord email, phone-buying advice. September fails the same way: it holds character in only 65.8% of off-lane replies, with 39 failed trials there against 64 and 62. That is a watch item for production itself, not a model-choice finding.
- Whether more cases could change the call. They can't. The cost intervals are already clear, so only a price change or a cheaper candidate reopens the question.
- The reasoning confound on 1017. It spends 55% of its output reasoning against September's 76%. That could explain part of its in-character drop. But it costs more already, and matching the reasoning share would plausibly cost more still. Tokens are unreported, so I can't show that, and I say it only as a reason the confound doesn't reach the call.
- The floors read different measures: the pass rate for qwen3.8-flash, the in-character rate for 1017. Both hold, so this goes to Method and is not merged.

**What I left out, and why.**
- The ordinal outcome and transcript scores restate the same judge's reading.
- Booth-offstage and the delivery check are not separated for either candidate.
- Failed-reply notices, turns over budget, provider faults and items delivered are zero or near zero everywhere.
- Measuring spend is the eval's own cost, not serving cost.
- I left out the p95 cost tail, because the means decide.
- Strata beyond the four largest are mostly under ten cases.
- I didn't price a point of character, because there is no trade: the control wins both axes.
- Depth and repeats are uniform.
- The floor readings are immaterial, and their mismatch is in Method.

**What I'd watch.**
- September's off-lane weakness.
- The contradicted prior insight on qwen3.8-flash cost.
- 1017's reasoning share.

The weakest margin is qwen3.8-flash's worst-case drop of 10.8 points against the 15-point margin. It sits on a rejected option, so I carried it as the explore control (where each candidate stops counting as no worse) rather than in Watch. An editor may want it in Watch, and I'd accept that.

**Next steps, each with what it settles.**
- Reconcile the earlier qwen3.8-flash cost finding (its baseline and per-token prices) with this run's. This settles whether its cost advantage was against another control or an old price. Only a candidate cheaper than September can reverse the call, so this has the most leverage.
- Rewrite the non-music directive so the persona declines without answering, and re-run off-lane on September. This settles whether the weakest stratum for every arm can lift without a model change.
- Calibrate the judge on human ratings of off-lane and in-character replies. This settles whether the absolute rates and the candidates' drops hold under a human reading.
- Record the simulator's request settings on every run. This closes the apparatus caveat for later rounds.
- Run 1017 with a matched reasoning share, if it stays of interest. This settles whether its drop belongs to the model. It doesn't change today's call.

## Tools

- **Calls.** 35 tool calls in all (Read, Write and Bash). Inside them, 50 runs of the `evals_analyst` command: 2 primer and help, 6 action help topics, 1 overview, 1 readings, 1 contrasts, 1 caveats, 1 options_dominance, 1 decision_sensitivity, 6 measure_read, 2 measure_breakdown, 2 arms_compare, 1 examples_list, 1 settle_cases (refused), 1 block_catalog, 2 figure_keys_check, 13 block_render (1 refused) and 7 capability_gap. There were 2 refusals in all, and I never needed price_tradeoff or levers_effects.
- **What I wished existed.** Each of these is also a capability_gap call.
  - settle_cases on the aspect's pass rate for two arms. It refuses `passed`, and lists only per-trial measures.
  - A compare block drawing both candidates against the control in one chart. Today it takes one pair per block, so c4 needs two blocks.
  - Choosing which kept example cards a block shows. The first off-lane card is a different failure mode.
  - `confound_check`, for 1017's reasoning share.
  - Sizing human calibration ratings.
  - Per-trial token counts, to split cost into price and length.
  - `render_report`.
- **Confusing or worth fixing.**
  - The `campaign_overview` result said `total: 77, offset: 0` with no `next:` line. I couldn't tell whether that was all of it. It seemed complete.
  - `contrasts_list` shows identical failed-reply-notice deltas for both candidates. That is correct (both had zero against September's few), but it looks like a copy error at first glance.
  - The 1017 floor reads `persona.holds_character` while the qwen3.8-flash floor reads the pass rate. Nothing in the readings listing flags the mismatch. I found it only by reading `speaks_for` line by line.
  - `count.failures.<aspect>.<arm>` (126 for qwen3.8-flash) differs from `scoreboard.<arm>.failed` (200), with no word on why.
  - The diverging-bars block over `strata` shows no intervals and no bold rows, so I switched to `compare`.
  - The `compare` source refused a mix of "overall" and strata because of their different levels. The refusal was clear, though the recovery didn't say "drop overall".
- **Paging.** Only measure_breakdown paged (12 of 26 strata on the first page), and a `limit` of 14 fetched the rest in one call.
