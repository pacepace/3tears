# Reasoning note: Kairo round 2

The first reasoning note. This is the form every exemplar takes. The point is the thinking, not the page.

**The eval.** Production persona model ("September") against two candidates on one aspect, character adherence. The question allowed a candidate to be up to 15 points worse. 299 cases, three repeats each, a judge with no human calibration, and four criteria set after the results were visible.

**What I noticed first.**
- Neither candidate is better on anything.
- Both cost 27–28% more per reply.
- That ends the decision before any judged evidence matters. The call rests on measured cost, which none of the caveats touch.

**The trap.** The pipeline reads qwen3.8-flash as "no worse than the margin". That's true, but its interval (−10.8 to −3.4) excludes zero: it is clearly worse, by an amount the margin happens to allow. Leading with the margin verdict would make a regression read as a pass. So the lede states the size first, then the margin.

**Story.** Decision. "Keep September. Both candidates cost 27–28% more per reply and stay in character less often." Confidence is high, if the caveats aren't material:
- **The judge is uncalibrated.** This only matters if it misreads in-character replies, and the deciding finding (cost) doesn't depend on it.
- **The aspect and control were chosen afterward.** This only matters if a different choice would have favored a candidate.
- **The late margin:** checked, not open. The call holds at every margin.

**Lead evidence.** A forest of change against September with the margin band, beside cost bars. The forest shows the asked question is answered; the cost bars show why the answer is "keep". One chart alone would hide half the reason.

**What I looked for and found.**
- **Where the loss lives.** I broke in-character rate down by kind of listener message. Off-lane asks, the largest stratum, carry about half of each candidate's loss, and they are also September's weakest area (66%).
- That turned a "keep" memo into one with a real next step: fix off-lane handling, then re-test.

**What I left out, and why.**
- **Depth, pass^k and precision tables.** They describe the method; none changes the call. They go to the appendix.
- **Small strata** (under 10 cases) with noisy deltas: shown only in the full table, never as leads.
- **Latency.** Not comparable, because runs shared the machine. One line in Method.

**What I'd watch.** 1017 spends much less of its output reasoning (55% against 76%). Its in-character drop may partly be the effort budget, not the model. That goes in Watch, beside 1017's claim, not in the page-level flag.

**Next steps, each with what it settles.**
- Fix off-lane brush-offs in the directives: does that close most of the gap?
- Calibrate the judge with 20 human ratings: turns the judge caveat into evidence.
- Find out why qwen3.8-flash cost more than an earlier campaign found: reconciles a prior insight this run contradicts.

**What doesn't transfer.**
- The call rested on cost because the control was cheaper *and* nothing beat it. Where a candidate is cheaper, the judged evidence decides, and the uncalibrated judge becomes material.
- Off-lane asks mattered because they were both the largest stratum and the control's weakest. A large stratum that every arm handles well is not a lead.
- The late margin was checkable only because the call held at every margin. A late aspect or control can't be checked that way and stays open.
