# Reasoning note: query planner config sweep (r2, prototype 3)

**The eval.** A research query planner swept across eighteen configurations: three models (Atlas-M, Brisk-S, Nano-XS) by three reasoning levels (off, low, high) by retry (none, one). There is no control, and each configuration ran 1,200 cases once. There are two overall gates, declared before the run: pass rate at least 97% and p95 latency at most 600 ms. Four tasks (single lookup, multi-hop, date math, ambiguous entity) carry pass rates but no case counts. Volume is 4M a month. The campaign does not say what makes a trial pass, and no prior insights exist.

**What I noticed first.**
- `options_dominance` has exactly one survivor: Brisk-S, low reasoning, one retry (J), at $0.30 per 1,000 calls (about $1.2k a month).
- J clears the pass-rate gate by 0.1 points (97.1% against 97%). Its computed interval is 96.2% to 98.1%, and `settle_cases` says a run of today's size reads it below the line 38.6% of the time.
- Every configuration whose pass-rate interval sits wholly above the line (D, F, L) misses the latency budget. The nearest is L (Brisk-S, high reasoning, one retry) at 870 ms.
- Retry is nearly free (one cent per 1,000 calls) and is what lifts J over the line: +1.9 points from I to J (0.3 to 3.4). High reasoning on top of J adds +0.8 (-0.4 to +2.1), which isn't separated.

**The trap.** "One survivor, and it dominates" reads like a clean decision. It isn't clean. The survivor is the best of eighteen, sitting on a line it clears by noise, and code read the gate on the point estimate. Read on the interval, nothing passes both gates. The second trap is the easy cure, "run more cases". Settling a 0.1-point clearance takes about 54k cases two-sided, roughly 45 times today's run, so more data is the wrong next step. Headroom has to come from changing the candidate, or from a declaration.

**Story.** A decision with a loud Watch. "Ship J, the only configuration inside both gates, knowing its pass rate sits on the line." I weighed calling it a diagnosis. Nothing is tangled, though: the declared rule picks J, and the risk is a borderline gate on the recommendation itself, which is exactly what Watch, the confidence and the flag exist for. Confidence is high_if_immaterial. Its four open caveats are all implied by code: the pass-rate point estimate, best of n, the p95 point estimate and no control. The first two push the same way, and their material_if clauses say plainly what happens if they bite: nothing measured passes both gates.

**Lead evidence.** A gate_scatter of pass rate against p95, with both lines and J emphasized. It shows in one view that J is alone in the passing corner and that the pass-rate-rich configurations sit to the right of the budget. Next comes a dot_interval of pass rate with computed intervals beside the settle table, which shows how thin the clearance is and why more cases won't fix it. Then range_bars of latency for the same configurations, which shows where the headroom went.

**What I looked for and found.**
- **Where the call flips.** `decision_sensitivity` on both thresholds. With the pass line at or below I's 95.3%, I becomes the pick. Past J's 97.1%, there is no call. Raising the latency budget never changes the pick, because everything it admits costs more. Below 513 ms there is no call.
- **What the levers do.** Retry helps everywhere and helps weak models most (a real interaction across model). It doesn't bear on the call, so it is omitted. The reasoning step J→L is unseparated, so L's case is its margin over the line, not a measured lead over J.
- **What the safer option would cost.** I priced L as a labelled what-if: +$1,324 a month over J. At this volume the money is small, so the real constraint is the latency budget, not cost.
- **Where J fails.** Multi-hop is its weakest task (95.6%). Tasks state no counts, so this is a descriptive lead. The three kept examples (one per task) show a collapsed hop, an off-by-one date window and an unasked clarification. kept_from is empty, so I don't know what share of failures they stand for.

**What I left out, and why.**
- A cost chart across all eighteen. With one survivor, cost decides nothing; the pick's cost is in text.
- All Nano-XS rows. None is near the line.
- The retry-by-model interaction, and the reasoning off→low effects. Both are real but don't move the call.
- Calls per task, p50 as a figure of its own, and per-task rates for the other configurations. No line is declared on them, and per-task rates invite a per-task gate nobody set.
- The outcome and extra-call measures. They hold nothing.

**What I'd watch.**
- J's pass rate. It is the weakest margin, sits on the recommendation itself, and is inflated by best-of-eighteen selection.
- J's p95 clearance of 87 ms. It has no stated spread, and a second retry would eat into it (one retry moved p95 from 465 ms to 513 ms).

**Next steps, each with what it settles.**
- Declare whether the pass-rate gate reads on the point or on the interval: settles whether J passes today. On the interval, nothing does, and the latency budget becomes the decision.
- Run Brisk-S low reasoning with a second retry, or with a multi-hop fix, on fresh cases, watching the tail: settles whether a cheap configuration can clear the line with its whole interval inside the budget, and corrects the best-of-eighteen optimism.
- The latency owner says whether the budget could rise to 870 ms: settles whether L becomes the safer pick, at about $2.5k a month.

**What doesn't transfer.** The sweep had one survivor only because two gates pull in opposite directions: reasoning buys pass rate and spends tail latency. In a sweep with a looser budget, several configurations survive and cost decides among them. The general lesson is that a lone survivor picked from many, sitting on a point-estimate gate, is a borderline result twice over. The fix is a declaration or a changed candidate, not a bigger run.

## Tools

**Calls.** About 46 tool calls in this session: 45 Bash or file calls plus the brief read. That covered 7 doc reads, the primer and help, and roughly 45 `evals_analyst` actions, some batched several to a Bash call: campaign_overview ×1, options_dominance ×1, caveats_list ×1, measure_read ×7 (pass rate by arm and by task, which took 2 pages; cost; latency; calls; extra calls; outcome), settle_cases ×2 (1 refused), decision_sensitivity ×2, levers_effects ×4 (1 refused), price_tradeoff ×2 (1 refused), examples_list ×1, block_catalog ×1, block_render ×9, figure_keys_check ×9 (paged), help ×4, capability_gap ×6.

**Wished for** (each logged as a capability_gap):
1. A gate read on its interval in options_dominance and decision_sensitivity. This campaign's whole question is "does J pass if the gate means the interval", and I had to infer the answer from two blocks.
2. Lever effects, comparisons and settle sizes on a p95.
3. Failure-mode counts (outcome holds nothing) and kept_from.
4. Task case counts, so task rates carry intervals.
5. render_report.
6. A two-gate scatter with intervals.

**Confusing in help or results.**
- Arm ids are inconsistent. campaign_overview and options_dominance print `arm:brisk-s.low.r1`, but settle_cases refuses that spelling and wants `brisk-s.low.r1`, while `{arm:brisk-s.low.r1}` is the figure-key form. The refusal's recovery was clear, but the first call is a coin toss.
- price_tradeoff takes `arm_filter` and `what_if` as single strings. I guessed lists, as in settle_cases' `arm_ids`. Its own notes then suggest naming arm_filter to price every option against one, which I didn't try.
- The levers diverging_bars renders unrounded values (`+0.03535`, `-0.4053`). The retry level prints as `→ 1.0` rather than "1 retry". With six arms, interaction rows read "at X less at Y", which is hard to parse, so I narrowed to three arms to get two clean rows.
- dot_interval on tasks draws the overall floor in every task panel, which reads like a per-task gate nobody declared. I switched to bars with no line.
- `figure_keys_check` pages at 11 keys, and each key repeats a paragraph-long `what:`. A 34-key draft took 4 pages and most of the output was the same definitions. The offsets also seemed to overlap: page 2 restarted a few keys before page 1 ended.
- `measure_read` lists `outcome` and `extra_calls` as measures with an empty statistics list. It would help if the overview said "holds no figures".

**Refusals.** Three, each with a usable recovery line: an arm-id spelling, list-versus-string on price_tradeoff, and levers_effects on latency (no mean).

## Revision 1

The editor scored round one 12/14 (passing) with 2 must_fix, 5 should_fix and 2 invariant concerns.

**Must fix, both done.**
- **c2 caption was false on its own block.** E (Atlas-M, high reasoning, no retry) has an interval of 96.9% to 98.6%, which also crosses the line. The caption now says that the pick's interval and E's both cross, and the rest clear. I kept E in the block because hiding the other crosser would flatter the pick.
- **Explore band for the pass-rate line was wrong.** In round one I swept only from 95 to 99. Re-swept from 80 to 99, the full ladder is:
  - above 97.1%: no call;
  - J down to 95.3%;
  - then I down to 94.8%;
  - then Nano-XS high reasoning with one retry (S) down to 92.7%, at $444 a month against J's $1,208;
  - then cheaper Nano-XS settings.

  The explore outcome now states the ladder and names S with its cost. This was my error: the tool did warn that flips outside the swept range are not listed. I logged a capability_gap asking for a full-ladder default.

**Should fix.**
- **Nano-XS omission.** Narrowed to "Nano-XS other than high reasoning with one retry". S is now named in the explore control and in next step 2, as a second-retry candidate: retry lifts Nano-XS most, which is also the reworded reason for omitting the retry-by-model interaction. S's interval (93.5% to 96%) is stated so nobody mistakes it for a near miss.
- **c3 set mismatch.** The range_bars now show J plus L, D and F, exactly the configurations whose interval clears the line. E is dropped from this block and the caption names the set.
- **38.6% without its condition.** The lede and c2 now say "if its true rate is today's".
- **One-sided vs two-sided cases.** The settle caption now names both readings. Watch keeps the two-sided 54,451 and says why: it is the size for an interval printed like the page's own to sit above the line. statistics.md says to quote two-sided where the reader compares with a printed interval.
- **Confidence form. Not changed: this is my one disagreement.** By the form definitions, `low` means the evidence is weak even with every caveat immaterial. Here the declared rule determines the call, and the doubt is exactly the open caveat. Turning it into a lower grade would hide what the doubt is. I added a reader-facing Method line explaining the grade and pointing to the near-even odds on the first screen. The editor's concern is fair: a near-coin-flip lead caveat makes "high if immaterial" read stronger than it is. If the house view is that a caveat this likely to bite should lower the form, the change is one field.

**Invariant concerns.**
- The c1 caption's "more than tenfold" is removed; the caption just says log scale.
- The settle table's "Each bar is…" note is renderer text on a table block. I can't change it from the report, so this is a tool issue: the settle source's note should match the block kind.

**Tools this round.** 10 tool calls, covering 9 `evals_analyst` actions:
- decision_sensitivity ×1;
- block_render ×1 (the c3 range_bars on J, L, D, F);
- figure_keys_check ×6 (new keys, then 40 keys over 4 pages, all read);
- capability_gap ×1.

blocks.txt is rebuilt, with c3 replaced by the new render. There was 1 new gap: sensitivity should walk the full ladder, or warn when the swept range stops short on the loosening side.
