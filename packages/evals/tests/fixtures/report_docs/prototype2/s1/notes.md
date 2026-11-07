# Reasoning note: s1 change (support model)

**The eval.** Production against three candidates (v2, v3, v4) on three request sets, each declared as an asked aspect: billing, tone and refunds. The question: does a candidate pass more support requests than production without losing any aspect? Twenty cases per set, one run each (v3 ran the tone set twice). A 10-point margin on each set, plus two declared floors: a pass-rate floor on billing and a policy-citation floor on tone (15 points). Scoring is two scripted checks per reply: the total matches the case's total, and the reply cites policy. No judge and no human ratings. Cost is a fixed price per reply: $2 per 1,000 for every arm except v4 at $1.

**What I noticed first.**
- v4 heads the scoreboard: 19 of 20 billing passes at half the price. That scoreboard is the billing set alone.
- v2 is the only candidate that clearly beats production anywhere (tone, +40) and the only one that clearly loses on pass rate (billing, -60). Both survive correction.
- Production gets **no** refund total right (0 of 20). Every arm sits near zero there.
- v4 cites policy in 8 of 20 tone replies. Every other arm, on every set, cites it every time.

**The trap.** v4: cheapest, top of the scoreboard, no pass-rate reading separated against it. Read the scoreboard or the pass-rate charts alone and v4 looks like the answer. But its tone replies stop citing policy, and that breaks a floor the campaign declared on its whole interval (corrected contrast regressed, adjusted p 0.0004). The question's own rule, "without losing any aspect", removes it before price comes in. A second trap: `options_dominance` reads every arm as passing its gates (no criteria are declared, so the floors gate nothing) and names v2 as dominating production. Taking that at face value would recommend the arm that fails two-thirds of billing requests.

**Story.** Decision: keep production. v2 loses billing, v4 loses policy citation on tone, and v3 separates from production on nothing at the same price. Confidence is `high_if_immaterial`. Both rule-outs are clear after correction. Still open:
- **The unsettled-reading caveats**, which matter only if v4's citations get fixed;
- **best_of_n**, which bears on nothing because no candidate is picked;
- **thirteen unrecorded apparatus caveats.** These bear only if the grader, the simulated user or the seeded world differed between arms, and nothing records whether they did.

**Lead evidence.** Two lead claims side by side, one for each rule-out. The billing forest (corrected contrasts with the margin) shows v2's drop clear of zero and of the margin. The tone citation rates sit beside the floor table. Then v4's cost, as bars: what keeping production forgoes. Then the tone forest for v2's one real gain. Refunds goes last as detail.

**What I looked for and found.**
- Whether the call is margin-sensitive. `decision_sensitivity` on the billing margin and the tone margin (0 to 40) found no change. On the tone citation floor, v4 breaks it at any floor under 23.9 points and holds only above 78.5, so no plausible floor rescues it.
- Where v4's citation loss sits. Only on the tone set. It cites every time on billing and refunds. That makes it a targeted fix, which makes v4 the live next candidate.
- What "pass" means. The campaign doesn't say. On every set the pass counts equal the replies passing both checks. So the "tone" set is not graded on tone at all: v2's tone gain is a gain in getting totals right on those requests. I put this in Watch and Method, not in the headline.
- What the failures look like. The frozen examples carry only the failing check's name, with no reply, input or expected value. There was nothing to show.

**What I left out, and why.**
- The reading layer's own intervals. Same directions as the contrasts. I kept one interval system per chart, but the margin verdicts and flip margins I cite come from that layer, and Method says so.
- Policy citation on billing and refunds: every arm is at ceiling there.
- Cost contrasts read "not separated". That is an artifact: every reply costs the same, and no value range was declared.
- The production-cost measure. It duplicates cost.
- Failure examples. Empty.
- v3's doubled tone runs. Method only.

**What I'd watch.**
- v4's billing drop could reach 15.9 points against a margin of 10, and its tone pass drop 32.4. Twenty cases cannot clear it even if the citations are fixed.
- The disagreement with code's dominance reading, stated on the page with its reason.
- Refunds can't show a loss while production scores zero there.

**Next steps, each with what it settles.**
- Fix v4's policy citation on tone requests and re-run the tone set: settles whether the cheaper model can hold the floor.
- Re-run v4 on billing and tone with more cases: settles whether it stays within the margins, which decides between keeping production and halving the cost per reply. No case count, because `settle_cases` refuses floors and margins (gap).
- Hand-check the refund cases' expected totals and the total check: settles whether refunds is a real measure or a broken check.
- Declare a check or judge for tone: settles whether the tone set measures tone.
- Record grader, simulator and world settings per run: turns the apparatus caveats into checked lines.

**What doesn't transfer.** The call didn't need cost because every option that differs from production either broke a declared line or showed nothing. With a cheaper candidate that held its floors but sat unresolved on its margins, this would be a status or tradeoff story, and the 20-case design would be the headline.

## Tools

**Calls.** 29 tool calls in this session; 55 `evals_analyst` invocations within them (several batched per Bash call):
- primer and help ×5;
- campaign_overview, readings_list ×2 pages, contrasts_list ×3 pages, caveats_list, options_dominance, examples_list, measure_read ×3;
- decision_sensitivity ×3, settle_cases ×1 (refused), block_catalog;
- block_render ×16 (3 refused, 7 of them the final renders for blocks.txt);
- figure_keys_check ×3 (one accidental repeat);
- capability_gap ×9.

**What I wished existed** (each logged as a capability_gap, listed in gaps.json):
- settle_cases against a declared rate-drop floor or aspect margin. It refused: "no declared criterion reads a figure for arm:v4". That is exactly the next-step sizing question for v4.
- A cross-aspect block. `diverging_bars` over contrasts with `small_multiples_by: aspect` returned only the refunds rows, with no panel label. A `table` over contrasts with no aspect did the same. **This is the most dangerous finding**: it renders silently and looks valid, but shows one aspect as if it were all of them.
- A forest over readings. It was refused for mixing 97.5% and 98.33% levels, though `readings_list` prints 98% for every reading. I fell back to contrasts.
- Check rates printable as percents. `goal_state:*.mean|pct` is refused (plain scale), and `dot_interval` prints 0.4 with no unit.
- A floor reference line for a floor that speaks for a different measure than the reading.
- `options_dominance` honouring declared floors as gates. With no criteria declared, a broken floor still reads `gate_status: pass`.
- confound_check and render_report (known missing).
- Example excerpts, input and expected value. All of them came back `none`.

**Confusing in help, refusals or paging.**
- Readings are paged at offset 13 of 15 and contrasts at 17 and 33 of 36. The caveats list cites refunds readings for v3 and v4 that appear only on the second readings page. Easy to miss without following `next:`.
- `options_dominance` lists `dominated_by: [arm:v2]` for production while `code_call.dominant` is empty and the mode is tradeoff. The two fields read as contradictory.
- The floors' `broken_at` reads with a sign (+23.9), and v3's `broken_at` on billing is negative (-15.9). I needed `|abs` and a careful read of the help to use it in prose.
- The overview's thirteen apparatus caveats describe document extraction (OCR, scan quality, vendor template) in a support-chat campaign. They are stated, so I carried them, but they look like a template leak from another scenario.
- The cost contrast's "not separated" reason (declare a value_range) is correct but reads as a finding unless you know cost is a fixed price.

## Revision 1

The editor scored round 1 at 11/14 with 2 must_fix and 7 should_fix findings. All nine are fixed, so Method needs no "why not" entries.

**Must fix**
- **The Method sentence on margin verdicts was wrong.** It said the charted intervals were the wider ones. Fixed:
  - Method now says the margin verdict reads each arm's own reading interval.
  - It says neither interval is always the wider.
  - It says that for v4 on billing the reading interval reaches further below zero, which is why Watch cites 15.9 while c1's chart shows 11.1.
  - c1 gains a second block: a forest over the v3 and v4 billing readings with the margin, so the 15.9 is on the page. A forest over all three readings is refused because v2 reads at 97.5% and v3 and v4 at 98.33%, though every printed level says 98%. That is the new gap.
- **The unsettled:refunds caveat pointed the wrong way.** It now says the caveat matters only if the refund check is broken, and that a sound check means no candidate can fall below production.

**Should fix**
- **Headline overstated v4's loss.** It no longer says v4 "stops citing" policy; it now says v4 "cites policy on far fewer tone replies".
- **c2's evidence didn't match its claim.**
  - The uncorrected rate chart is replaced by the corrected citation contrast (99.55%, regressed), which is the evidence the lede rests on.
  - The floor table stays, captioned so that "Largest drop" reads as the far end of the interval.
  - The caption also explains why v2 and v3 read unclear: twenty cases cannot show a perfect rate holding within the floor.
  - c2's text no longer cites 23.9. That figure moved to Explore, where it is called the near end of the interval.
  - The raw rate chart is listed in omitted.
- **Production's weak tone result was hidden.** c4 now says production passes only 10 of 20 tone requests, through a key.
- **Explore's "binds v4 alone" was wrong.** It now says the floor decides only v4, and that v2 and v3 read unclear at the declared floor and hold it above 19.
- **The apparatus caveats repeated one sentence.**
  - Each world caveat now has its own clause.
  - vendor_template and grader_version are named as the facts most likely to explain production's zero refund totals, and are tied to the refund check in Next.
  - That next step now names the grading template and the grader version.
- **Next steps 1 and 2 overlapped.** They are merged into one. It says why no case count is given: the lines are margins and floors against production, and the sizing computation reads only declared criteria.
- **The page named a tool's internals.** "Code's dominance reading" and "Code's options_dominance reading" are replaced by a description of the reading itself, in Watch and in omitted.
- **Invariant concern on a typed count.** Method no longer says "two scripted checks".

**Disagreements:** none. Both must_fix findings were real errors on my part; the margin-verdict sentence came from assuming instead of comparing the two interval sets.

**Tools this round:**
- 6 tool calls in the session.
- 15 `evals_analyst` calls:
  - block_render ×13: 5 trials, 1 of them refused, and 8 final renders;
  - figure_keys_check ×1: all 19 keys read;
  - capability_gap ×1: readings print 98% while carrying 97.5% and 98.33%.
