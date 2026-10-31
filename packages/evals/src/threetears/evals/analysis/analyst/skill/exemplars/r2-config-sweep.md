# Reasoning note: query planner config sweep (synthetic reference r2)

**The eval.** An inner tool model that plans research calls. A full factorial grid: three models, three reasoning budgets (off, low, high) and retry on failure (none or one), so 18 configurations. 1,200 test cases graded by exact match against gold call sequences. Gates: pass rate at least 97% and p95 at most 600 ms. Volume is about 4 million calls a month.

**What I noticed first.**
- Exactly one configuration lands inside both gates: Brisk-S, low reasoning, one retry, at 97.1% and 513 ms.
- It clears the pass gate by 0.14 points. Its interval is about ±0.9.
- The levers behave consistently across models. A retry adds about two points for about 5% more cost. More reasoning adds accuracy and a lot of latency, which pushes Atlas-M and high-reasoning Brisk-S out of the budget.

**The trap.** "The only configuration in the box" sounds decisive, and the scatter makes it look so. But it is in the box on its point estimate, by less than a sixth of its interval, and it was chosen as the one of eighteen that crossed the line, which makes its estimate optimistic. Reported as a clean win, a reader would lock it in. The recommendation stands, because nothing else passes and it is cheap, but the margin is part of the story, not a footnote.

**Story.** Decision. "Use Brisk-S with low reasoning and one retry." Confidence is high, if one caveat isn't material: the pass gate is unsettled for the recommendation. It is material if the true rate is under 97%, because then nothing passes both gates and the question becomes a tradeoff between accuracy and the latency budget. The near-miss alternatives belong in why_not: the same configuration without retry (95.3%), and high-reasoning Brisk-S (98.0%, 870 ms).

**Lead evidence.** The gate-space scatter: pass rate against p95 on a log axis (latency spans 18×), both gate lines, and a path per model through its reasoning levels. It shows all 18 options and both gates at once, and how close the pick sits to the edge. Next, lever-effect bars, because a complete grid makes average effects meaningful once you have checked that the levers don't interact.

**What I looked for and found.**
- What the retry fixes. It recovers about 70% of malformed calls and barely touches wrong arguments, which are now the largest remaining failure, mostly off-by-one date windows. That points the next step at argument handling, not more retries.
- Where the pick is weak. Multi-hop questions pass at 95.6%, below the overall gate within that task type. The gate is overall, so it doesn't bind, but it is where the margin will come from.

**What I left out, and why.**
- Extra-call efficiency in detail. It describes overhead and doesn't change the call. One line and the full table.
- Every configuration's per-type figures. Six configurations within the budget, plus the most accurate for reference, carry the point.

**What I'd watch.** The pass-rate margin, and what a re-test can actually settle. 3,000 more cases narrow the interval to about ±0.5 points. That settles the gate only if the true rate is at least half a point above it. A margin of 0.14 can't be settled by any affordable sample, so a fix that raises the rate is worth more than more cases.

**Next steps, each with what it settles.**
- Run 3,000 more cases on the recommended configuration: settles the gate if the rate is really half a point or more above 97%, and otherwise shows the margin is too thin to rest on.
- Fix date-window arguments in the planner prompt or an argument normalizer: settles whether the largest remaining failure falls, which would widen the margin without spending latency.

**What doesn't transfer.** Lever averages worked because the grid was complete and the effects were roughly the same for every model. On a sparse grid, or where the levers interact (see the classifier note), averages mislead. The 0.14-point margin is this run's. The lesson is to read the pick's clearance against its interval whenever one option is chosen out of many.
