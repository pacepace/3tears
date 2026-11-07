# Reasoning note: query planner config sweep, gates read in each task

**The eval.** A research query planner, 18 configurations: three models (Atlas-M, Brisk-S, Nano-XS) crossed with three reasoning levels (off, low, high) and a retry or none. 1,200 cases each, four tasks with traffic shares (single lookup 40%, multi-hop 25%, date math 20%, ambiguous entity 15%). No control. Two gates, both declared per task: pass rate at least 97% in each task, p95 latency at most 600 ms in each task. Pass is a check on the planned calls; no judge. Volume 4 million calls a month. No exchange rate between pass rate and latency.

**What I noticed first.**
- `options_dominance` returns no survivors, and every arm fails the latency gate as `no_figure` in every task. Latency was recorded only per arm, over all tasks.
- Two arms clear pass rate in all four tasks: Brisk-S high + retry (L) and Atlas-M high + retry (F). Both have pooled p95 over the budget (870 ms, 2,551 ms).
- Every arm under the budget pooled (all of Brisk-S off and low, all of Nano-XS) misses the pass line somewhere. The best of them in every task is Brisk-S low + retry (J): 95.6% multi-hop, 96.4% ambiguous entity.
- Multi-hop is the hardest task for every configuration.

**The trap.** Two traps. First, the tool's "no figure" reading makes the per-task latency gap look like the whole story, as if the result were blocked by missing data. It isn't. If every task's p95 were within 600 ms, the pooled p95 would be too. So a pooled p95 over the budget proves at least one task is over it. The two pass-rate survivors fail latency for certain, and everyone else fails pass rate. The call "nothing ships" doesn't depend on the missing figures; only the *where it flips* does. Second, the pooled pass rate: J reads 97.1% over all tasks, which looks like a pass, while it misses two tasks. The gate is per task, so I kept pooled pass rate out of every lead and labelled it where it shows up.

**Story.** Decision: "Ship none of the 18 configurations yet." This isn't hedging: the gates decide it. It carries tradeoff slots (where_it_flips, our_lean) because the way out is a declaration or a new configuration, and nobody declared which line gives. Confidence is high if immaterial: no_control matters only for the configuration-to-configuration comparisons, and best_of_n only for the lean. Code checked neither.

**Lead evidence.** Two blocks for one claim, because the call has two halves. Per-task pass rate for J, L and F against the 97% floor (dot_interval, which has no intervals because none were recorded; the caption says so). Then range bars of median-to-p95 latency for all 18 against the budget, which shows which configurations are inside it. I rejected the gate matrix: every cell reads "Fails: no figure", which buries where pass rate binds.

**What I looked for and found.**
- Whether Atlas-M is worth its cost at the top. It isn't: F vs L on pass rate is not separated (-0.4 to +1.7), and L is about a fifth of the monthly cost and a third of the tail. That narrows the real decision to Brisk-S.
- The Brisk-S ladder: the two lines cross between low and high reasoning. A retry at low lifts pass rate by 1.9 points (+0.3 to +3.4) for almost nothing. That's the seam for the next experiment (medium reasoning, or a second retry at low).
- Whether L's or J's per-task gate readings are settled. They can't be read: per-task pass rates carry no count or interval (`settle_cases` and `arms_compare` refuse per-task rows). L clears multi-hop by 0.3 points on a point estimate, as the best of 18. That goes in Watch.
- What J's failures look like: a two-hop question collapsed into one query, and an ambiguous name answered without asking. Planning errors, which suits a reasoning-level fix rather than a retry.
- Sensitivity: both the declared latency sweep and a hypothetical latency line return `no_call` across the range, because the per-task no-figure blocks every arm. Not informative, so where_it_flips is in words, conditional on per-task latency.

**What I left out, and why.** The gate matrix (unreadable, see above). Atlas-M off/low and high without retry, and all of Nano-XS in detail: none is nearer either line. Pooled pass rate as a gate. The full lever grid and the other retry interactions, which describe the grid. Calls per task (no line, nearly flat). Outcome and extra-calls (no figures held). The date-math off-by-one and Nano truncated-JSON examples (not where the call binds). Traffic shares, which only matter for routing, and nobody ran routing.

**What I'd watch.**
- L's multi-hop clearance: +0.3 points, point estimate only, best of 18. A re-run likely reads it lower. If it drops under, nothing clears pass rate everywhere, so the call holds, but the lean loses its candidate.
- J's latency is inside the budget only pooled. Multi-hop plans two or three calls and may run slower than the whole.
- Latency comparability across configurations (shared machine or not) isn't stated. `confound_check` isn't available.

**Next steps, each with what it settles.**
- Record p95 per task, from stored timings if they were kept: settles which line J and L actually miss, and what budget would admit L.
- Run Brisk-S between low and high reasoning, and low with a second retry, on multi-hop and ambiguous-entity cases: settles whether one configuration clears both lines with no line moved.
- Record per-task case counts so per-task pass rates carry intervals: settles whether L's thin clearance is real.
- The owner declares which line gives, or a price of pass rate in ms of p95: turns the dead end into a call.

**What doesn't transfer.** The pooled-to-per-task deduction only runs one way: a pooled percentile over a per-task budget proves a failure, but a pooled percentile under it proves nothing for any task. It worked here because the arms that cleared the other gate were exactly the ones over budget pooled.

## Tools

- **Calls:** 38 tool calls in this session; 55 `evals_analyst` invocations among them (several Bash calls ran more than one), well inside the 80 budget.
- **Actions used:** primer, help (index, block_render, price_tradeoff, figure_keys), campaign_overview, options_dominance (2 pages), caveats_list, measure_read (latency, cost, calls, pass rate by arm and by task, outcome, extra_calls), decision_sensitivity (threshold:p95, threshold:pass_rate, hypothetical:latency_ms/p95), levers_effects (pass rate all and retry; latency refused), settle_cases (2, both refused), arms_compare, examples_list, block_catalog, block_render (gate_matrix, dot_interval, range_bars, bars, examples, table; lattice refused), price_tradeoff (refused), figure_keys_check (3), capability_gap (8).
- **What I wished existed** (each logged as a capability_gap):
  1. Gate reading that uses a pooled percentile when it alone proves a per-task failure. This is the top gap: the tools' verdict (every arm fails on "no figure") is true but hides that the call is already decided.
  2. Per-task intervals and counts on pass rate.
  3. `levers_effects` on percentile measures.
  4. `price_tradeoff` across two gated options as a labelled what-if.
  5. `weakest.*` on per-task criteria, which is refused even though gaps exist.
  6. A sensitivity sweep that isn't blocked by no_figure on every place.
  7. Block column labels that name statistic and scope.
  8. `confound_check` and the page renderer.
- **Confusing in help or output.**
  - The hypothetical latency sweep's note says the snapshot "declares no gate on latency_ms" though the p95 criterion is one.
  - `measure_read`'s `arm_filter` takes one string; I first passed a list and was refused (the recovery was clear).
  - The overview prints no traffic shares; they first appeared in the gate_matrix rows, then read as `task.<task>.share`.
  - `measure_read` reports `held: [mean]` for pass rate, while `levers_effects` and `arms_compare` produce binomial intervals overall. It wasn't obvious that overall intervals exist when measure_read shows none.
  - The bars block's Note column repeats the value.
  - The configs table's "Latency" column is p95, unlabelled.
- **Paging.** `levers_effects` over pass rate pages 13 of 117 effects at a time; `setting_filter` made it usable. `options_dominance` pages 10 arms per page, and the repeated `criteria.p95.threshold` line on every no-figure gate makes the pages long.

## Revision 1

The editor scored round 1 at 12/14, a pass, with three must_fix findings. All three were right, and I fixed them.

- **best_of_n caveat.** I had written that if Brisk-S high retry (L) re-reads under the line, nothing clears pass rate everywhere. That was wrong: Atlas-M high retry (F) still clears all four tasks. The caveat now says the call holds because F is the slowest configuration (keyed p95), and that best_of_n touches F's clearances too. My error was reasoning about L alone after deciding F was out on latency.
- **c3.** The claim now reads "only high reasoning *with a retry*". K (high, no retry) misses three tasks. c3 now leads with a per-task dot plot of all six Brisk-S configurations against the floor. The pooled table moves second, with a caption saying the line is not read there.
- **Atlas-M omission.** I had written "none is nearer either line", which was false. Atlas-M high without a retry and Atlas-M low retry miss only multi-hop, by 0.4 and 0.6 points, which is nearer than J's 1.4. The real reason they stay out is that both are far over the budget, further than L. The omission now says that, by key.

Should_fix items, all taken:

- **Routing by task.** I had elided it under "traffic shares". It is a real option: on point estimates, J on single lookup and date math plus L on multi-hop and ambiguous entities clears the pass line everywhere. `where_it_flips` now names it, conditional on the task being known before planning and on per-task latency. The first next step now says it settles routing too. A new next step runs the split as one measured configuration, since a mix nobody ran is never priced. Traffic shares moved to Method. Note that J's date-math rate sits exactly on the line, so the split inherits that as a zero-margin cell; Watch says so.
- **c2.** It now has two blocks: cost bars for L and F only, and their per-task pass rates against the floor. The sentence says why cost matters while both are over budget: it decides between them only if the budget gives, which is the lean. The difference interval can't be drawn by any block, so a Method line says differences between two configurations are stated in the text. That is a new gap.
- **Fragility of L's multi-hop clearance.** I checked the editor's inference. The traffic-weighted mean of the task rates reproduces the pooled rate for both J and L, so cases were very likely drawn by share. On that assumption the clearance is about a third of one standard error. Watch now says this in words, labelled as an inference, with no typed figure, because no tool computes it. That is a new gap.
- **Typed counts in omitted.** "six" and "Two" are gone. "two or three calls" now reads "several calls".

The editor's other concern was that the report has no explicit appendix. I left it alone: per ANALYST.md the renderer supplies the appendix, and Method speaks to the reader, not about the renderer.

**Disagreements.** None with the findings. One point of my own: the routing option deserves its slot, but I kept it out of the headline. It is unmeasured on latency, and J's date-math cell sits exactly on the line.

**Tools this round.** 9 tool calls:
- 1 read of the editor file
- 1 render pair (Brisk-S per-task dots, L and F overall dots)
- 1 file read
- 1 scripted edit
- 3 figure_keys_check passes (34 of 34 keys read)
- 1 call that re-rendered all 7 blocks into blocks.txt and logged 2 capability_gap calls
- 1 gaps.json and notes update

Counted as `evals_analyst` invocations: 15. New gaps: a block for a two-arm difference with its interval, and settle or spread under an assumed per-task count.
