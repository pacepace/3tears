# Reasoning note: s3 measurement (one arm against two bars)

**The eval.** One arm, solo, with no control and nothing to choose between. The question is how often solo is correct and how clear it is. Two aspects, accuracy and clarity, each ran its own 20 cases twice (40 trials each, separate trial ids). Both aspects carry a 90% bar on judged answer correctness, declared before the run. A bar warns and never gates. Two judged dimensions: answer correct (pass/fail, and it is the pass definition) and answer clear (an ordinal score with no stated scale). The judge has no human calibration ratings, an undetermined tier, and an unreported temperature. Cost is flat per trial.

**What I noticed first.**
- The same measure on the same arm reads very differently by aspect: answer correct is 95% on the accuracy cases and 60% on the clarity cases.
- The clarity bar is missed outright (interval 37.8 to 78.7%, wholly below 90%). The accuracy bar is unclear (95%, interval 75.3 to 99.2%, straddling 90%).
- The clarity aspect's bar reads correctness, not the clarity score. Nothing declares a line on clarity itself.

**The trap.** Two traps, one in the data and one in the tools.
- The tools. Every unscoped or criterion-level figure reads the clarity criterion on the accuracy cases' number. `criteria.clarity/bar/answer.correct.solo.value` reads 95%, `count.failing` reads 0, `options_dominance` calls solo a pass with nothing it borders, `settle_cases` sizes the clarity bar as if it clears, and the campaign-wide scoreboard reads 38 of 40 passed, which is the accuracy trials only. A report built from those would say solo meets both bars. Only the aspect-scoped bar readings and scoreboards show the clarity miss. I cited only `aspects.<aspect>.*` keys and logged the gap.
- The data. "Clarity" sounds as if it was measured against a line. It wasn't: its bar is a correctness bar on a different case set. The "how clear" half of the question has no line, no scale and an uncalibrated judge, so it can only be described, not judged.

**Story.** Status, with a small what-is-tangled on clarity. "Fix solo on the clarity cases before relying on it: it is judged correct on only 60% of them against a 90% bar, and its pass on the accuracy cases is not yet settled." Confidence is high_if_immaterial:
- judge_uncalibrated: material only if the judge marks right answers wrong on the clarity cases. The fix-first call rests entirely on it, and every figure is an absolute judged rate, which is the weakest kind of judged claim.
- nothing_declared: material only if the owner's real line sits below the top of the clarity interval. The bars warn and never gate.
- unreported judge temperature: material only if the judge sampled differently between the two runs.
- No caveat was checked immaterial by code.

I did not choose `low`: the clarity miss is clear of the line by more than ten points at the interval's top, so the evidence is strong if the judge is right. `options_dominance` says pass; my page says the clarity bar is missed. The disagreement comes from the tool reading the wrong aspect's figure, and the omitted list says so.

**Lead evidence.** The clarity-case correctness against its bar. No block draws a bar reading with its interval and line, so the claim carries two blocks: the judged dot_interval (value and the 90% line) and the pass-rate dot_interval (the interval). Then the accuracy-case pair the same way, then the clarity score tables per aspect.

**What I looked for and found.**
- Whether the accuracy bar can be settled. settle_cases (correct for the accuracy criterion, since its figure is the accuracy one): about 314 trials for the two-sided interval four times in five, against 40 today. That goes in Watch and Next.
- Whether more cases could rescue the clarity cases. They can't: the interval is wholly below the bar. The next step is a fix, not more cases.
- What the failures look like. The five frozen failures (one accuracy case twice, three clarity cases) carry no reply, reason or excerpt. Nothing to show; reading them is the first next step.
- Whether the two aspects share cases. Case ids look shared (tc-12 to tc-19 appear under both). If they are the same cases, the same items read correct far more often in one run than the other, which would point at the judge or the run setup. No tool can confirm it; it is a next step.
- Bar sensitivity. decision_sensitivity returns no_call for both bars because bars never gate. I wrote the status bands in words in the explore control.

**What I left out, and why.**
- Cost: flat per trial, one arm, decides nothing.
- The campaign-wide scoreboard and criterion-level figures: they read the wrong aspect.
- The clarity settle sizing: wrong figure, and the miss is already clear.
- Pass rate as its own measure: identical to correctness here.
- The production-replicating cost: identical to the measuring spend.
- Thirteen method-reach apparatus caveats: land once in Method.

**What I'd watch.** The accuracy bar, the nearest line: 95% against 90%, interval reaching 75.3%. It passes on its point estimate only. The gap between the two case sets' correctness, if they turn out to share cases.

**Next steps, each with what it settles.**
- Read the failed clarity-case replies, fix what they share, re-run the clarity cases: settles whether the shortfall is a fixable failure mode and whether a fix clears the bar.
- Human correctness ratings on the clarity cases: settles whether the miss is the model's or the judge's.
- Confirm whether the two aspects ran the same cases: settles whether the 95%/60% gap points at the judge or run setup.
- Declare a scale and a line for the clarity score, or point the clarity bar at it: settles the "how clear" half of the question.
- Run the accuracy cases to about 314 trials: settles the accuracy bar if the true rate is near today's.

**What doesn't transfer.** The tool defect (criterion figures read on the wrong aspect) appears only because two aspects carry a bar on the same measure for the same arm. The general lesson: when two aspects share a measure, check every unscoped figure against the aspect-scoped reading before citing it.

## Tools

**Calls.** 31 tool calls in this session, holding 49 `evals_analyst` invocations (several batched per Bash call): primer 1, help 3 (index, block_render, figure_keys), campaign_overview 1, readings_list 1, caveats_list 1, measure_read 5, examples_list 1, options_dominance 1, block_catalog 1, settle_cases 4 (2 refused), figure_keys_check 2, block_render 15 (4 refused, 6 of them the final renders for blocks.txt), decision_sensitivity 2, capability_gap 8. Not used: contrasts_list, measure_breakdown, arms_compare, price_tradeoff, levers_effects: one arm, no control, no slices.

**What I wished existed** (each also a capability_gap):
- Criterion-level figures, settle_cases, options_dominance and unscoped keys that read on the criterion's own aspect. This is the top gap: as built, the code's own figures say the opposite of the readings on the lead finding.
- A block that draws an arm against an absolute bar with the bar reading's own interval and line.
- small_multiples_by aspect that actually draws one panel per aspect (it silently drops the clarity panel).
- Failure examples with reply, reason and excerpt.
- A way to tell whether two aspects ran the same cases, and a paired comparison of one arm across aspects.
- A bar-threshold sweep that reports where a bar's status flips.
- The clarity score's scale and an interval at a level.
- The pass-rate measure in the bar's unit and on the bar's interval method (today 0.6 with no unit, and a slightly different interval than the bar reading).

**Confusing in help or results.**
- Arm ids: options_dominance and settle_cases' result print `arm:solo`, but settle_cases refuses `arm:solo` and wants `solo`. The recovery line fixed it at once.
- The judged dot_interval labels the bar's row "floor 90%", though the criterion is a bar.
- The `nothing_declared` caveat says no gate, floor or margin was declared, which reads oddly beside two declared bars until decision_sensitivity's note explains that a bar warns and never gates. Saying that in the caveat itself would help.
- `cost_per_1k.solo|usd_per_1k` prints "per 1,000 replies" though the cost counts trials.
- `weakest.criterion` reads "answer correct" without saying which aspect's bar, which is ambiguous here.

**Refusals and paging.** Six refusals, each with a usable recovery line: two on the arm id form, three on `reference: floor` (bars aren't a reference kind, and two criteria read the same measure even with the block narrowed to one aspect), one on an empty floors table. No result needed paging.

## Revision 1

**Editor round 1:** 11/14, passes, one must_fix and four should_fix items, plus three invariant concerns.

**Must fix: the remedy clause.** The recommendation no longer promises that a model or prompt change will fix the clarity miss. It now says solo is not ready on the clarity cases, keeps "more cases will not change that", and says the cause (model, judge or run setup) stays open until the case check and the judge calibration come back. The headline changed from "Fix solo..." to "Don't rely on solo for the clarity cases yet...", for the same reason.

**Should fix, all taken in substance:**
- **Case ids.** New detail claim c4 with one examples block per aspect: accuracy fails a single case (tc-19) on both repeats, and the clarity failures the snapshot keeps (tc-12 to tc-14) fall on other ids. Next step "confirm the two aspects ran the same cases" moved to first, with what each answer points to and which step comes next. The omitted entry now says the content is empty but the ids carry information. I did not adopt the editor's inference that the clarity failures are exactly 8 cases failing on every repeat. The snapshot keeps only three clarity failures, so that is arithmetic on a guess.
- **Sizing.** Watch and the last next step now say the trials have to come from new cases, and that the trial-level sizing is a lower bound. Method has a new "Sample sizes" entry.
- **c1/c2 blocks.** Partly declined, with the reason in Method under Intervals. No block draws a bar reading's interval beside its line: `reference: floor` is refused even when the block is narrowed to one aspect. I dropped the 0-to-1 pass-rate blocks, which carried the second, mismatching interval. Each claim now has one block (value against the bar, in percent), and its caption cites the bar reading's interval by key.
- **c3.** The text is softened: "somewhat lower ... a gap near its noise, on a score with no stated scale". Each table's caption now carries its standard error by key, and the tables show the clarity score only, so correctness isn't repeated. The single table with both aspects that the editor asked for can't be drawn: an unscoped table shows one aspect, and tables don't panel. Method says so.

**Invariant concerns:**
- **Typed counts.** Removed from omitted and Method ("five", "three", "eight").
- **Tool names and caveat ids in omitted.** Rewritten in words.
- **Bars entry in Method.** Each aspect now cites its own bar key.

**Disagreements.** One, partial: the c1/c2 single-block request can't be met with today's blocks. The editor's 8-case inference stays out of the page, because it rests on figures the snapshot doesn't hold.

**Keys and blocks.** `figure_keys_check` read all 20 cited keys; none were refused. blocks.txt was re-rendered for all 7 evidence blocks, with no refusals.

**Tools this round.** 8 tool calls holding 19 `evals_analyst` invocations:
- block_render: 8 trial renders (4 refused: twice because a table can't panel, twice because `floor` is refused even with an arm filter)
- block_render: 7 final renders for blocks.txt
- figure_keys_check: 1
- capability_gap: 3

**New gaps** (also appended to gaps.json):
- settle_cases should size in cases, not in trials treated as independent.
- A judged table should print its standard error and show one row per aspect.
- Example cards should name their aspect.
