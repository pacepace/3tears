# Reasoning note: concierge model selection (synthetic reference r1)

**The eval.** Four candidate models for a concierge chat assistant, with no control. Five tasks: four routable traffic segments with declared shares, plus speculation bait (150 prompts), a test category that occurs inside every task and has no traffic share of its own. 1,190 conversations. Gates: overall quality at least 85, p95 latency at most 3.0 s, and per-task behavior floors (speculation declines at 95%). The judge was calibrated on 200 human-graded conversations (agreement 0.86).

**What I noticed first.**
- Read task by task, the cheapest passing model differs: Brisk-S on billing, Open-70B on small talk, Atlas-M elsewhere. Routing that way would cut blended cost by 34%.
- Both cheap models fail speculation bait (81% and 89% against 95%), and their intervals sit clear below the floor.
- Atlas-L has the best quality but misses the latency budget on four of five tasks.

**The trap.** The gate matrix invites reading one row at a time and routing each task to its cheapest passing cell. But speculation bait isn't a segment you can route around. It shows up inside billing and small talk too. Any model that receives traffic has to pass it, so the 34% routed plan is blocked, not available. The trap is treating a cross-cutting aspect as one more task. The headline says so before the reader can do the per-row arithmetic.

**Story.** Decision. "Ship Atlas-M on all traffic. It is the only candidate that clears every gate." Confidence is high, if one caveat isn't material: Atlas-M's issue resolution in troubleshooting is 90% at a floor of 90%, interval ±4.4. That is the recommendation clearing its own gate on its point estimate alone. It is material if the true rate is below the floor, since then no single model passes everywhere.

**Lead evidence.** The gate matrix: task by candidate, each cell's status with its binding reason, the cheapest passing cell marked, and the speculation row flagged as applying to all traffic. The matrix shows the decision and the reason the obvious saving is unavailable in one view. A score heatmap would hide which criterion failed.

**What I looked for and found.**
- Whether more data could rescue the cheap models on speculation. It can't: both upper bounds are under 95%. That made the next step a fix to the model, not more cases.
- Why the small models fail. It isn't overall quality. They fail on behaviors: Brisk-S speculates and calls the research tool when it isn't needed, and Open-70B makes account changes without confirming. That earns a support claim, because it says what a fix must target.
- Atlas-L's latency, as a range chart against the budget. It is the only reason the most accurate model is out, so it needs its own evidence.

**What I left out, and why.**
- An average-quality ranking. Quality above the floor isn't a decision axis here, and a ranking invites choosing on it.
- Behavior panels that bind for no one. They stay in the full table.

**What I'd watch.** Atlas-M's troubleshooting resolution. It is the weakest margin, and it sits on the recommendation itself. A re-test with 400 conversations narrows the interval to about ±3 points. That settles the gate only if the true rate is a few points from 90%. An estimate that stays at the floor means the floor itself needs a decision.

**Next steps, each with what it settles.**
- Add a speculation guard to Brisk-S and re-run the bait: if it reaches 95%, routing becomes legal and the 34% saving is real.
- Re-test troubleshooting on Atlas-M with 400 conversations: settles whether the recommendation passes its own floor, unless the rate really is at the floor.

**What doesn't transfer.** Routing exists as an option only because tasks had declared traffic shares and were routable. The cross-cutting test category is this campaign's design, not a general rule. The general lesson is to ask, for every per-segment saving, whether some gate applies to all traffic.
