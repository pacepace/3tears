# Reasoning note: s2 choice (held-out prototype)

**The eval.** Three arms (alpha, beta, gamma), no control, one asked aspect: does the reply get the total right. 20 cases, 3 repeats each, 60 trials per arm. Two checks per trial: the reply cites the policy, and its total matches the case total. Two bars were declared: mean spend at most $1.00 per 1,000 replies, and mean turn time at most 800 ms. The question: which arm answers within the cost and latency bars? No answer-rate floor, no price per point, no prior insights. Thirteen apparatus and world settings were not recorded.

**What I noticed first.**
- alpha answers best (88% right totals) but misses both bars by its whole interval: $4.00 per 1,000 replies against $1.00, and 1,001 ms against 800 ms.
- beta (65%, $0.80, 701 ms) and gamma (45%, $0.50, 401 ms) both clear both bars with room.
- The citing check passes 100% for every arm. Every failure is a wrong total, so "pass" here is the total check alone.
- The only separated pair is alpha ahead of gamma. beta against gamma is +20 points with a Holm-corrected interval of -9.3 to +47.6.

**The trap.** Two traps, pointing opposite ways.
1. The code's call. options_dominance lists all three arms as survivors (bars "warn and never gate") and finds gamma undominated, with alpha and beta each dominated by a cheaper arm that is not separated from it. Read as given, that makes this a cost tradeoff that includes alpha, or a quiet pick of gamma on cost. But the campaign's own question makes the bars the condition: "which arm answers *within* the bars". alpha is out, and the page has to say why it departs from code.
2. The cost axis inside the bars. gamma dominates beta on cost and turn time, and nothing separates them on answering. The kind's own rule ("a pick shown ahead on no pair rests on cost or simplicity") would point at gamma. But the question ranks answering inside the bars, and gamma gets the total right in under half its trials on its estimate. Picking gamma for a saving the bar already makes irrelevant would be optimizing an axis nobody asked about. Picking beta rests on a point estimate. Either way the confidence is low, and the page must say that a larger run could flip it to gamma.

**Story.** Decision, low confidence. "Use beta, provisionally: of the two arms inside both bars it gets the total right more often, but this run cannot separate it from the cheaper gamma." I weighed calling it a tradeoff (beta vs gamma) or a "no clear winner" diagnosis. The question does rank the axes (answers, subject to the bars), so it isn't a tradeoff in the stories.md sense. And a non-separated pick is exactly the low-confidence decision principles.md describes. The first-screen flag says only the bars were declared.

**Lead evidence.** Two leads, because the call has two reasons. c1: cost bars and turn-time bars, each with its declared bar line. These show alpha out and beta and gamma in. (One gate scatter with both lines would be better, but it refuses: bars aren't gates to it.) c2: answer rate per arm as dots with intervals, beta emphasized. The sentence carries the corrected pair difference and interval, because no block can draw a pair reading without a control.

**What I looked for and found.**
- **Whether the cost bar reads the right cost.** The bar reads "spend, all roles". Production cost per reply is identical for every arm, so the mismatch is immaterial. That goes in a Method line.
- **Sensitivity of the bars.** Code says moving either bar moves nothing, because it ignores bars. In words: the call flips to gamma if the turn-time bar tightens into beta's interval (673–729 ms), or the cost bar drops below $0.80. Admitting alpha needs both bars relaxed. These went into explore.
- **A hypothetical answer-rate floor.** Between gamma's and beta's estimates, the floor leaves beta alone inside the bars. Above beta's estimate it leaves nothing, since alpha misses both bars. Code's sweep names alpha as dominant there because it ignores the bars, so I left that reading out.
- **What would settle beta vs gamma.** settle_cases: 240 trials per arm for four-in-five separation (118 for even odds), read as independent arms. A paired design on shared cases needs fewer. A fresh run of today's size reads gamma ahead 8.1% of the time if the true gap is today's.
- **Price per point.** price_tradeoff reports $1.50 "per point" for gamma→beta. That divides a $0.30-per-1,000 cost gap by a gain of 0.2, so it is the price of 100 points. I left it out and logged it.
- **Failures.** The nine frozen failures carry no reply, excerpt or expected total. Each arm's three shown failures sit on different case ids (gamma tc-09/10/11, beta tc-12/13/14, alpha tc-17/18/19), but three per arm can't tell nesting from noise.
- **Weakest margin.** beta's turn time, 99 ms under the bar on its mean. Its p95 is 791.6 ms, near the bar's value, though the bar reads the mean.

**What I left out, and why.**
- Code's survivor/dominance listing: it ignores the bars that the question makes binding. Method states the departure.
- Price per point: the tool figure is off by a factor of 100 on a 0–1 measure, and no price per point was declared.
- Unpaired 95% intervals from arms_compare and price_tradeoff. One of them (beta→alpha) just clears zero, but it is uncorrected and independent. The corrected pair reading doesn't separate the two, and showing both would invite a false separation.
- p95 latency for alpha and gamma: the bars read the mean.
- Failure example cards: they hold no content.
- The hypothetical-floor sweep's "alpha dominant" segment, since it ignores the bars.
- Production cost: identical to measured spend.

**What I'd watch.**
- beta's turn time, the nearest line (99 ms on the mean), with its p95 near the bar's value.
- The best-of-n effect. beta was picked from the two arms inside the bars by a lead that isn't separated, so a fresh run will tend to read it lower.
- The unrecorded ingest backlog. The caveat says the extractor trades thoroughness for speed under backlog. If the backlog differed between arms, it could produce the exact pattern seen here: the fast arm answers worse. That is the confound I'd most want checked. With no confound_check, it stays open in the confidence.

**Next steps, each with what it settles.**
- Re-run beta and gamma on the same fresh cases, sized near 240 trials per arm (fewer if read paired): settles whether beta's lead is real, and flips the pick to gamma if it isn't.
- Declare a minimum answer rate: settles whether gamma is acceptable at all, and whether anything inside the bars is good enough. At today's estimates, a floor above 65% leaves nothing.
- Record apparatus and world settings, ingest backlog first: turns the thirteen open caveats into checked lines.
- Compare which cases each arm gets wrong: settles whether gamma's failures nest inside beta's (a capability gap) or scatter (noise).

**What doesn't transfer.** The bars bind here only because the question names them as the condition. Where bars are advisory, code's reading (bars warn) is right, and the cost tradeoff that includes alpha is the honest story. The beta-over-gamma lean depends on the question ranking answering above cost inside the bars. With a declared price per point, gamma's saving might buy enough to win.

## Tools

**Calls.** 50 `evals_analyst` invocations (about 70 tool calls overall, counting doc reads and file writes). Several invocations were batched into one Bash call. By action:
- reads: primer and help ×3, campaign_overview, readings_list, options_dominance, caveats_list, measure_read ×5, arms_compare, settle_cases, examples_list, block_catalog, price_tradeoff, decision_sensitivity ×3, measure_breakdown (refused)
- blocks: block_render ×16 (3 refused)
- checks and gaps: figure_keys_check ×3, capability_gap ×9

**What I wished existed** (each logged as a capability_gap):
1. **Bars as gates when the question makes them the condition.** This was the biggest gap: code's dominance and sensitivity contradict the plain reading of the question.
2. A gate_scatter over two declared bars.
3. A block for pair readings without a control.
4. A correct price per point on 0–1 measures.
5. Failure examples with content.
6. A per-case outcome read across arms without a control.
7. confound_check, and a paired settle for arms that share cases.
8. Goal-state rates printable as a percent.
9. A wide, reader-labelled scoreboard table.

**Confusing in help, refusals or paging.**
- **settle_cases counts.** `cases_needed` counts trials, not cases, on a repeated design. The help says so, but the key name invites "cases" in prose.
- **arms_compare basis.** It reads beta vs gamma as independent even though a corrected pair reading exists in readings_list, and its 95% interval differs from the pair's 98% Holm interval. Two intervals for the same difference are easy to mix.
- **Two units for the same difference.** The pair `change` is in points (+20), while arms_compare and settle give the same difference as a fraction (+0.2).
- **No campaign-wide level key.** `measure.level` is refused, so I borrowed `spread.beta.<measure>.overall.level` to print "95%" in Method.
- **The pct refusal.** It was clear, and its remedy pointed me to `pass_rate.<arm>`. The help could say that pass_rate is the percent form when pass equals one check.
- **gate_scatter refusals.** The first said to drop the reference or name one measure. The second (no reference) said "0 read 'cost_usd'". Together they meant "bars are not gates", but neither said so.
- **The overview's total line.** It ended with `total: 36` and no `next:` line, so it wasn't clear whether the overview was complete. It was.
- **Arm labels.** The bars block labels arms "alpha", while dot_interval and the tables label them "A · alpha". That is an inconsistent identity across blocks.

## Revision 1

**Editor round 1:** 12/14, passes. One must_fix, six should_fix, three invariant concerns.

**What changed.**
- **Must_fix (dominance misstated).** The omitted item is gone. Method's "Bars as the condition" now states code's dominance correctly: gamma dominates beta and beta dominates alpha, each on cost with answering not separated. gamma does not dominate alpha, which is ahead of it (+4.6 to +74.2). Method also says why the report departs from it: the bars remove alpha, and answering outranks cost inside them. The gamma why_not now says the pipeline favours gamma over beta on cost.
- **Kind rule.** The recommendation now says plainly that the campaign's rule for an unseparated pick falls back to cost, which would choose gamma. It departs from that rule because the question ranks answering inside the bars.
- **c3 evidence.** No block source draws settle sizing, so I dropped the claim. The old c4 is now c3. The settle figures stay in What settles it and Next. Method has a new "Settle sizing" line, and the trial counts moved to "Cases and repeats".
- **c2 evidence.** The claim text and caption now say the chart shows each arm's own spread, not the corrected gap. No pair-reading block exists (gap logged in round 0).
- **240 trials.** What settles it and Next step 1 now say fresh cases, the same for both arms, at today's 3 repeats each. They add that repeating today's cases more often doesn't count toward it.
- **Minimum-answer-rate control.** It now names its reading. On the estimates, a floor between 45% and 65% leaves beta alone. On whole intervals, as the bars are read, no floor separates beta from gamma. Next step 2 asks the owner to declare which reading the floor uses.
- **Turn-time control.** Now bounded. Between gamma's and beta's intervals, gamma is the only arm inside; below gamma's interval, none qualifies.
- **Typed numbers.** "the nine frozen failures" is now "the frozen failures", and "three pairs" is now "every pair of arms". "(a hundred points)" is removed, and "four times in five" is now "in most such runs". No count key exists for pairs or examples (new gap).
- **Next step 4.** Reworded: the gap sits either on a few hard cases, which a targeted fix could close, or spreads across cases, which only a larger run settles.

**Where I disagree, and why.**
- **The pick stays beta.** The editor asked for the departure to be stated, not reversed. I kept beta because the question asks which arm answers within the bars. A cost tie-break inside a bar that both arms clear optimizes an axis the question treats as satisfied. Confidence stays low, and the lede and the first next step say gamma becomes the pick if the lead is noise.
- **The 8.1% wrong-side rate stays in Watch.** The editor suggested moving it into the recommendation as the strongest reason. It is conditional on today's gap being the true one, so it sizes a rerun's risk and is not evidence that beta is ahead. Using it as support would assume the conclusion. Method's "Rerun risk" says so.
- **Invariant 5 is not fixable in the document.** The measure-key column headers in the checks table come from the renderer. The scoreboard labels left with the dropped claim. The c3 table still prints the check ids as headers. That is renderer behaviour, already logged as a gap for the scoreboard. I kept the table because the claim needs both checks side by side.

**Tools this round.** 4 `evals_analyst` calls: help figure_keys.counts, figure_keys_check over every key in the draft (all 28 read), and capability_gap ×2. I made no new block_render calls. The c1, c2 and c3 queries are unchanged from round 0, so blocks.txt reuses their exact outputs, with the dropped c3 removed and c4 renamed c3. Running total: 54.

**New gaps.**
- No block source for settle sizing.
- No count keys for pairs of arms or frozen examples.
