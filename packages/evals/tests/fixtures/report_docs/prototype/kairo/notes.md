# Reasoning note: Kairo persona model, round 2 (prototype run)

**The eval.** Kairo's production persona model, September, against two candidates: qwen3.8-flash (A) and 1017 (B). 299 replayed cases, three repeats on every arm, one asked aspect: character adherence, read as a pass rate with a 15-point margin. The judge (gpt-6-luna) is not calibrated. The question, the control, the margin and the floor were all set on the run date, after the results. No monthly volume and no exchange rate between cost and character were declared. Two prior insights, one of which says qwen3.8-flash lowers production cost.

**What I noticed first.**
- qwen3.8-flash is down 7.6 points with an interval of -10.8 to -3.4. It is clearly worse. It is also "within margin", because a 15-point margin allows that much loss.
- 1017 is down 3.0 points (-5.8 to +1.1), not separated on the pass rate. On the judge's holds-character dimension, though, its corrected contrast is clearly negative (-4.7, -8.2 to -1.3 at 99.69%).
- Both candidates cost more per reply: $2.43 and $2.41 per 1,000 against $1.89, an uplift of 27 to 28%. The prior insight says qwen3.8-flash is the cheaper model. This run contradicts it.
- options_dominance: September dominates both. decision_sensitivity: the call holds at every margin, and code checks the late margin and floor as immaterial.

**The trap.** Two of them. First, the campaign's own verdict on both candidates is `within_margin`. If that verdict leads, qwen3.8-flash reads as a pass, but its interval sits wholly below zero. Size goes first: "clearly worse, by 7.6 points". The margin goes in the caption and the explore control. Second, the prior insight primes a reader to expect a cost saving from qwen3.8-flash and to read the character loss as the price of it. There is no saving to trade, so this is not a tradeoff. Pricing it would invent one: price_tradeoff confirms no price exists, because September is at least as good on both axes.

**Story.** Decision: "Keep September: both candidates cost 27–28% more per reply, and neither keeps Kairo in character better." The call doesn't rest on the judge, since cost is measured telemetry. If the judge were badly wrong, the worst case is a priced trade (pay more for better character), not a switch. Confidence is high_if_immaterial. The open caveats are the uncalibrated judge, the late aspect and control, the simulator-settings apparatus caveat, the reasoning-share confound on 1017, an unexplained model confound, and qwen3.8-flash's two lost trials. The late margin and late floor are code-checked as immaterial. A first-screen flag says that no rule was set in advance.

**Lead evidence.** Two leads, because the call has two reasons. The first is a forest of the pass-rate readings with the margin band, which answers the asked question with size first. The second is cost bars per 1,000 replies, which is what rules out 1017: by the campaign's rule it would pass the character question. Then come a corrected-contrast forest, showing that 1017 also holds character clearly less often on the judge's own dimension, and an off-lane stratum chart. Last is a detail card set showing what off-lane failures look like.

**What I looked for and found.**
- Where the loss concentrates. Off-lane asks, the largest stratum (38 cases), carry about half of each candidate's holds-character loss. September is weak there too, at 65.8%. The cards show the same two failure types on all three models: Kairo helps with the non-music task, or refuses like a help desk. That makes off-lane handling a prompt problem worth fixing for any future round, not a candidate-specific defect.
- Whether 1017's drop is the model. 1017 reasons a smaller share of its output (55%) than September (76%), which is the flagged confound. It bears on why 1017 holds character worse. It doesn't bear on the call, because 1017 costs more at the share it ran.
- Whether anything else separates. Action completion, failed-reply notices, provider faults and budget-ended turns are not separated or are near zero everywhere. Booth-stays-offstage is not separated for either candidate.
- Cost integrity. Production cost is recorded on 828 of qwen3.8-flash's 895 usable trials (September 864, 1017 879). price_tradeoff's cost gap (+$0.53 per 1,000) and the paired contrast (+$0.000616 per reply) differ, presumably because the contrast is paired on cases where both arms recorded cost. Both say "costs more" with intervals clear of zero, so the direction holds. Which number the page should quote is a tool question (gap logged). By my own arithmetic, even if every qwen3.8-flash trial with no recorded cost had cost nothing, its mean would still be above September's. No tool states that, so it stays out of the report.

**What I left out, and why.**
- The outcome and transcript grades. They are ordinal with no stated scale, and they move the same way as holding character.
- Small strata (next up, music tech, assistant shape). Their drops are large on very few cases, so they stay in the full table.
- Price per point and monthly cost. There is nothing to price, and no volume was declared.
- Latency, which was read concurrently. Depth and pass^k are identical across arms and describe the method.
- Measuring spend across all roles, and the test's own cost. Both price the eval, not production.

**What I'd watch.**
- The contradicted prior insight on qwen3.8-flash's cost. It is the only thing that could reopen the call. If the earlier saving was real and the promoted directives erased it, a different prompt could bring qwen3.8-flash back as a priced trade.
- Missing cost records on qwen3.8-flash, the largest gap of the three.
- September's own off-lane weakness, which any future candidate will be read against.

**Next steps, each with what it settles.**
- Reconcile qwen3.8-flash's cost with the earlier round on shared cases, and explain the missing cost records. This settles whether the saving the prior insight describes exists. Only a cheaper candidate reopens the call.
- Calibrate the judge on human ratings of holding character, sampled heavily from off-lane asks. This settles whether the judged drops are real. If the judge underrates the candidates, the report becomes a trade on cost.
- Tighten the off-lane directive and rerun off-lane asks on all three models. This settles whether the largest share of the loss is a shared prompt problem, and whether either candidate closes the gap.
- Rerun 1017 at a reasoning share matched to September's. This settles whether 1017's character loss belongs to the model. It can't reopen the call.
- Declare the question, control and margin before the next round. This removes the late-criteria caveats.

**What doesn't transfer.** This decision exists because the candidates were dearer on measured cost, which makes the judge caveat survivable. A round where a candidate saves money would be a tradeoff, and the uncalibrated judge would then sit on the deciding axis. "Within margin" on a clearly worse candidate is a trap whenever the margin is generous relative to the observed drop. Here the margin is about 1.4 times qwen3.8-flash's worst drop.

## Tools

**Calls.** 53 calls to the analyst tool: help 9 (primer and the help index included), campaign_overview 1, readings_list 1, contrasts_list 1, caveats_list 1, options_dominance 1, decision_sensitivity 1, measure_read 6, measure_breakdown 2 (1 refused), examples_list 1, price_tradeoff 1, block_catalog 1, block_render 14 (1 refused; 5 of them are the final renders for blocks.txt, 3 were key checks through a title), capability_gap 12. That is about 32 Bash calls in all.

**What I wished existed** (each is also a capability_gap call, listed in gaps.json):
- A figure-key resolver. I used block_render's `title`, which accepts figure keys, to check all 27 keys in one call. It worked, but it is a side door.
- measure_breakdown over the asked pass rate. It accepts only judged dimensions, so the stratum chart is on holding character, not on the measure the lead reading uses.
- A place for `reference` and `emphasis` in the report document's evidence. block_render takes them outside `query`, so I carried them as extra evidence fields.
- Stratum intervals, or a separated marker, in the diverging_bars values table. The notes say separated rows are bold, but nothing in the text table marks them.
- Example excerpts centred on the failing reply. The off-lane excerpts end in music-search tool output, so the judge's reason carries the card.
- confound_check and settle, as the skill anticipates.
- The definition of a passing trial, the lever behind `confound:model`, and why cost is missing on some trials.
- The earlier round's figures, to check the contradicted prior insight.
- A landing field for method-reach caveats in the report document.

**Confusing or rough edges.**
- The tool's verdict string `within_margin` sits beside "worse" in the forest note column. A writer copying the verdict would make exactly the trap the skill warns about.
- Floors read different measures per arm: the pass rate for qwen3.8-flash, holds_character for 1017. The floor figures carry `speaks_for`, but the inconsistency isn't explained. I named it in Method.
- `failed_reply_notices` contrasts are identical for both candidates (both have zero against September's two notices), which looks like a bug at first glance.
- `flip_margin|pts` prints a sign (+10.8), so I used `|abs|pts`.
- `cost_per_1k.<arm>` prints "$1.89" with no "per 1,000 replies", while `scoreboard.<arm>.reply_cost_usd|usd_per_1k` prints the words. Picking between them to avoid a doubled or missing unit took a test.
- A monthly key in a title refuses the whole block. That is reasonable, but it is only discoverable by trying.
- Paging never triggered: every listing fit on one page.

## Revision 1

The editor scored round 1 at 11/14 and passed it, with 1 must_fix, 6 should_fix and 2 invariant concerns. Round-1 files are in `round1/`. Every finding is fixed as described below; none is rebutted.

| Finding | Change |
|---|---|
| c5, must_fix: "look the same on every model" contradicted by 1017's card | c5 now states what the three cards show. September answers like a help desk, qwen3.8-flash helps with a landlord email, and 1017 queues a voice line that reads as a backstage recap. The caption says these are the only off-lane failures frozen, one per arm. They show kinds of slip, not frequency. The excerpts end in music-search output, so the judge's reason names the failing line. c5 stays as a detail claim, not a lead. |
| Answer the question in its own terms | The lede now leads with size: qwen3.8-flash is clearly worse by 7.6 points and 1017 is no better. It then says both stay within the 15-point margin the question allows. c1 adds "though within the margin". Its caption says both sit inside the band. The recommendation and qwen3.8-flash's why_not both say "within the margin". The order is still size first, then margin, per principles.md. |
| Omitted outcome and transcript grades: false reason for 1017 | Reworded. They fall with holding character for qwen3.8-flash. For 1017 they are not separated from September, and they don't bear on a call made on cost. |
| c2 should carry the corrected interval | c2 now has two blocks. The first is a forest of the corrected cost contrasts per 1,000 replies at 99.69%, both clear of zero. The second is the per-arm bars. The claim says both differences stay clear of zero after correction. Both why_not slots cite the per-reply contrast (`contrast.<arm>.production_replicating_cost.delta`) instead of the uplift. Method's cost basis now says the extra cost per reply is a corrected contrast with an interval, and that the percentage uplift is arithmetic on means with no interval. |
| Reasoning share has no figure | The confound caveat cites `reasoning_ratio.1017` (55%) and `reasoning_ratio.September` (76%). Next step 4 now says matching would raise 1017's reasoning, so it is *unlikely* to make 1017 cheaper. The old wording ("cannot") was stronger than a share supports. |
| Second short_run caveat inaccurate | Each short_run caveat now names its own single lost trial. The first is "this run's single trial lost to the harness". The second is "the other of qwen3.8-flash's two lost trials, on its second run". |
| Watch: "where every option is weakest" | Now reads "the weakest stratum large enough to read". Assistant shape reads lower on only 6 cases. |
| Invariant 5: caveat ids could print above the appendix | Every confidence caveat now carries a `label` in reader words, for example "A trial lost to the harness on one qwen3.8-flash run". The renderer has to print the label, not the id. I can't confirm that without a page renderer (W3), so a gap is logged. |
| Invariant 1: lead finding rests on an undefined pass | The tool doesn't expose the definition, so I can't supply it. Method's "Pass" entry now says the pass rate is the share of usable trials the campaign's grader marked passed, and that it is stricter than holding character for every arm. It adds that qwen3.8-flash's drop holds on holding character alone too (c3: down 6 points, clear of zero at 99.69%), so the lead finding doesn't rest on the undefined measure. |

**What I could not do.**
- Show the failing reply line on the c5 cards. The frozen excerpts end where they end. The gap was logged in round 1.
- Give the editor a way to check `reasoning_ratio` itself. No investigate action returns it, so it is a new gap. The figures are cited by key, so the renderer can resolve them.
- Confirm how the renderer prints caveats. There is no page renderer yet.
- Define "pass". The definition isn't in the snapshot.

**Tools, round 1.** 12 analyst-tool calls: block_render 10 (1 forest trial of the cost contrasts, 3 key checks through titles, 6 final renders for blocks.txt) and capability_gap 2. Two new gaps were appended to gaps.json. There were no refusals.
