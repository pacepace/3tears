# Reasoning note: persona model tradeoff (synthetic reference r4)

**The eval.** A persona chat model in production against three challengers: A (the flash model with 64k output), B (flash with 16k), and Corvid. 300 replayed conversations, 2,400 replies, paired. Four aspects: in character (judged, calibrated on 240 human ratings, agreement 0.82), completes the action, and two standing guardrails, music facts at least 90% and declines to speculate at least 95%. Volume is 12 million replies a month. No exchange rate between cost and personality was declared.

**What I noticed first.**
- At this run's routing, A saves about $7.6k a month (30%) and loses 6.1 points in character (interval 4.9 to 7.3). Every other aspect holds.
- Corvid loses only 2.2 points and saves the most per point lost of any full switch. But it breaks the speculation guardrail at 92%, and its drop there is clear of zero.
- B cost the least in this run and breaks the facts guardrail at 89.4%.
- Every saving is this run's provider routing's, not the models' own. The snapshot names no provider for any arm, so the routing is unknown, and the page says so beside the prices.
- A's loss grows with conversation depth: close for the first turns, 18 points behind past turn ten.

**The trap.** Corvid looks like the sensible middle path: a small personality loss for real savings. Pricing it beside A would put an option that breaks a standing policy into the trade. Guardrail failures remove options before any pricing. The second trap is deciding at all. With no declared price for a point of personality, any recommendation is a business choice. The report prices the trade and says who makes the call.

**Story.** Tradeoff. "Switching to A saves $7.6k a month but loses 6 points of personality, mostly in long conversations." The first screen says plainly that no rule was set in advance, and names the persona owner as the decider. The lean (labelled as ours): before switching, run a hybrid that hands conversations back to Production after turn five. The depth breakdown is the seam that suggests it: the loss sits in long conversations, and the savings sit in short ones. Nobody has run that mix, so the report prices only the options that were measured and names the hybrid as the next experiment.

**Lead evidence.** Change against Production on every aspect, as diverging bars with intervals, so the reader sees in one chart why B and Corvid are out and that A loses only personality. Then in-character rate by turn depth, which is the seam the recommended hybrid would exploit. Then the savings-against-loss map with a price line, which carries the trade itself.

**What I looked for and found.**
- What users notice. A blind preference test on 240 conversations: raters split evenly on short conversations and prefer Production in longer ones (61%, interval just clear of 50%). It is small, so it anchors direction, not size.
- Where in a conversation the loss happens. Banter and taking a correction lose the most. That is descriptive and uncorrected, so it is detail, not a lead.

**What I left out, and why.**
- Completion of the requested action. Every option holds it, so it doesn't bear on the trade.
- Any figure for the hybrid itself. It was never run, so its savings and loss would be modeled numbers no key states. It appears only as a next step.

**What I'd watch.**
- A's speculation rate is 96.4% against the 95% guardrail. Its drop from Production reaches 2.8 points at the interval's edge, so the guardrail isn't settled. That bears on every option that uses A, including the hybrid.
- The hybrid assumes Production's voice returns after the handoff. That is untested.

**Next steps, each with what it settles.**
- Run the hybrid (hand back to Production after turn five) as a measured option on the same cases: settles what it keeps of the savings and of the personality, so it can be priced like the others.
- Then a two-week live pilot on 10% of traffic, watching session length and thumbs-down past turn five: anchors points to user behavior.
- The persona owner declares a price per point: turns the tradeoff into a decision.
- More speculation bait on A: settles its guardrail.

**What doesn't transfer.** A hybrid existed because the loss concentrated by depth and cost varied with depth. Most tradeoffs have no such seam. The human preference anchor was a lucky extra. Without one, Watch says no evidence links points to what users notice.
