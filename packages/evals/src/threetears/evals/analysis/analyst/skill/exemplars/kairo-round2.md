# Reasoning note: Kairo round 2

The first reasoning note. This is the form every exemplar takes. The point is the thinking, not the page.

**The eval.** Production persona model ("September") against two candidates on one aspect, character adherence. The question allowed a candidate to be up to 15 points worse. 299 cases, three repeats each, a judge with no human calibration, and four criteria set after the results were visible.

**What I noticed first.**
- Neither candidate is better on anything.
- Both cost more per reply, and each paired cost gap against September is clear of zero.
- That ends the decision before any judged evidence matters. The call rests on measured cost.
- Measured is not untouched. The snapshot attaches the reasoning-ratio confound to 1017's cost contrast, not only to its judged ones. 1017 spends less of its output reasoning than September (55% against 76%), so at a matched budget it would likely cost more, not less. The confound's likely direction favours the call, and the cost claim carries it and says so.
- Cost was read on most trials, not all. The cost claim says how many, and that the order holds even if every missing trial cost nothing.

**The trap.** The pipeline reads qwen3.8-flash as "no worse than the margin". That's true, but its interval (−10.8 to −3.4) excludes zero: it is clearly worse, by an amount the margin happens to allow. Leading with the margin verdict would make a regression read as a pass. So the lede states the size first, then the margin.

**Story.** Decision. "Keep September. Both candidates cost more per reply and stay in character less often." Confidence is high, if the caveats aren't material:
- **The judge is uncalibrated.** This only matters if it misreads in-character replies, and the deciding finding (cost) doesn't depend on it.
- **The aspect and control were chosen afterward.** This only matters if a different choice would have favored a candidate.
- **The late margin:** checked, not open. The call holds at every margin.

**Lead evidence.** A forest of change against September with the margin band, beside the paired cost gaps against September with their intervals. The forest shows the asked question is answered; the cost gaps show why the answer is "keep". The cost claim is about gaps, so its block draws the gaps, not each arm's cost level, and any percent beside them is read on the same paired basis. One chart alone would hide half the reason.

**What I looked for and found.**
- **Where the loss lives.** I broke in-character rate down by kind of listener message. Off-lane asks, the largest stratum, carry about half of each candidate's loss, and they are also September's weakest area (66%).
- That turned a "keep" memo into one with a real next step: fix off-lane handling, then re-test.

**What I left out, and why.**
- **Depth, pass^k and precision tables.** They describe the method; none changes the call. They go to the appendix.
- **Small strata** (under 10 cases) with noisy deltas: shown only in the full table, never as leads.
- **Latency.** Not comparable, because runs shared the machine. One line in Method.

**What I'd watch.** 1017 spends much less of its output reasoning (55% against 76%). Its in-character drop may partly be the effort budget, not the model, and its cost may be too: a matched budget would likely raise it. That goes in Watch beside both of 1017's readings, its in-character drop and its cost, not in the page-level flag. The next step that re-runs 1017 at a matched budget settles both.

**Next steps, each with what it settles.**
- Fix off-lane brush-offs in the directives: does that close most of the gap?
- Calibrate the judge with 20 human ratings: turns the judge caveat into evidence.
- Find out why qwen3.8-flash cost more than an earlier campaign found: reconciles a prior insight this run contradicts.

**What doesn't transfer.**
- The call rested on cost because the control was cheaper *and* nothing beat it. Where a candidate is cheaper, the judged evidence decides, and the uncalibrated judge becomes material.
- Off-lane asks mattered because they were both the largest stratum and the control's weakest. A large stratum that every arm handles well is not a lead.
- The late margin was checkable only because the call held at every margin. A late aspect or control can't be checked that way and stays open.
- The confound on 1017's cost pointed the call's way. A caveat on the deciding evidence that points the other way can reverse the call, and then the report may be a diagnosis.
