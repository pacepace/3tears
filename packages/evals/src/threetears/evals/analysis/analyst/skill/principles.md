# Principles

## What a report is for

A reader with five minutes should leave knowing what to do, why, and the main risk. If the campaign can't support a call, they should know what is unresolved and what would resolve it. Everything else serves that. The first screen (headline, decision block, first claim) is where the reader decides whether to trust the rest, so it has to stand alone.

A report is not a tour of the data. Each claim earns its place by changing or qualifying the decision. A section that would show nothing new is dropped. A task with nothing specific to show gets a line, not an empty chart.

## Honest emphasis

- **Lead with the size of a change before its verdict against a line.** A candidate whose interval sits wholly below zero is worse, even when the margin allows that much loss. "Within the margin" said first makes a regression read as a pass. Say how much worse, then whether the margin allows it.
- **Show the recommended option's weak axes.** A recommendation whose weak side is hidden reads as stronger than the evidence. The weakest margin belongs in Watch. "Weakest" is code's measure, not yours: the gap smallest relative to its line (the gap over the line's value), which the `weakest.*` keys name. Cite those keys in Watch. If another margin matters more to the call (fewer points, or fewer half-widths), Watch still names the weakest and says why the other matters more, stating both in the same terms.
- **Status follows the direction of the change, not the margin.** A worse result stays marked as worse even when it is within the margin. Approval colors on a regression read as approval.
- **Frame nothing stronger than its interval supports.** A point estimate that clears a gate while its interval crosses it is *borderline*. It counts as passing for the call, but it goes in Watch with what would settle it: `settle_cases` sizes that run.
- **State findings in the direction the evidence supports.** A refuted claim at low confidence is still a refutation. Don't let a weak tier blur which way the evidence points.

## Elision

Leaving things out is most of the craft, and it has to be accountable. Everything substantial you saw and chose not to show goes in `omitted` with a one-line reason the editor can check: "doesn't change the call", "describes the method", "a stratum too small to read". The test: would a reader who saw it decide differently? If yes, it isn't elidable.

- Method statistics (repeats, depth, pass^k, precision tables) describe how the evidence was gathered. They go to Method or the appendix, never between the decision and the evidence.
- Small, noisy strata appear in the full tables, never as leads.
- If the data a comparison needs is missing, say so in one caption line. Never fill it in.

## Decisions: survivors and dominance

A call usually comes from two questions. Which options survive every gate and guardrail declared for them? Among those, does one dominate, at least as good as every other on every axis that matters? `options_dominance` computes both. You read the result and decide what it means for this campaign.

- **One survivor dominates:** a decision. The story says why it wins and what it gives up.
- **Several survive and none dominates:** if the campaign declared a rule that ranks the axes, the rule decides. If not, the honest report is a tradeoff. Price it and let the owner choose. You may state a lean, labelled as yours.
- **Nothing survives, one arm was measured, or nothing separates the options:** the report says what is missing and what would settle it.

If your call disagrees with `options_dominance`, the editor will see that. Disagree only for a reason the page states, such as an axis the computation can't weigh.

Gate reasoning that recurs:

- A gate with no figure for an option is not passed. Absence of evidence isn't a pass.
- An aspect that occurs inside every task (safety, speculation) gates every option that would receive any traffic, not one task. Reading it per task invents savings.
- Gate the asked aspect on its own readings. If a floor speaks for a different measure than the reading does, show it separately and name the mismatch in Method. Don't merge the two.
- "Pass" means whatever the campaign defined. If the campaign doesn't say what makes a trial pass, say so plainly. Don't guess.
- A latency or cost win needs the delivery rate beside it. An arm can buy a shorter tail by declining work.
- Rank on the scope that matches what was changed. A subsystem change read on whole-turn figures can pick the wrong arm.

## Confounds

A confound qualifies a claim. It never names a cause by itself. Check apparatus confounds (a shared machine, a changed simulator, a short run) before model ones, because they touch everything. For each confound, either show that it doesn't matter from the figures you have (`confound_check` is not available; log a `capability_gap`) or carry it into the story and the next steps.

A confound qualifies every finding whose evidence it is attached to, not only the one it was noticed on. `contrasts_list` names each contrast's confounds and `caveats_list` what each caveat touches. A setting that moved between arms (a reasoning budget, a retry) moves cost and latency as well as quality, so measured telemetry is not immune to it (`statistics.md`, Caveats on the evidence).

Say a caveat once, at the level it applies to. A caveat that touches the whole comparison (late criteria, an uncalibrated judge, a shared-machine latency) belongs in the confidence. One that touches one arm or one finding (that arm's reasoning budget, one short run) sits beside that claim, usually in Watch.

## Confidence

Confidence is conditional on the caveats, never capped by them. There are three forms:

- **`high`**: no open caveat that bears on the call remains.
- **`high_if_immaterial`**: the evidence is strong and caveats remain open. Each one gets a `material_if` clause saying what would make it matter ("only if the judge misreads in-character replies"). Where the deciding finding doesn't rest on the caveat, the clause says so ("cost, the deciding finding, is measured"), but only when no caveat is attached to that finding's own evidence. A caveat that is attached to it is said beside it, with the way it would likely move it.
- **`low`**: the evidence is weak even if every caveat is immaterial. A pick that nothing separates is the usual case.

Word each caveat by its condition. A caveat never calls itself "material" beside a confidence that reads "high, if the caveats aren't material": that reads as a contradiction. Say instead what would make it matter ("only if the owner's minimum sits above the pick's figure"). Where a caveat matters to the call whatever happens (nothing declared ranks the options), say what the confidence covers ("high in the measured figures; the choice among them waits on a declared price") rather than leaving a bare "high".

The form reads the method and the variance of the evidence, never the result alone. A lead's wrong-side risk (how often a fresh run of today's size would read the other way) is stated beside the call, and doesn't lower the form by itself.

Code removes the caveats it can prove immaterial, such as a late margin when the call is the same at every plausible margin, and shows each as a checked line. Nobody else attests a caveat away. **An aspect left unsettled past its declared margin is an open caveat**, even when it isn't the asked aspect, because a loss the margin forbids is still possible and would change the call. The same goes for a gate the recommended option clears only on its point estimate.

## Prior insights

When this run contradicts an earlier finding, say so in one sentence in Watch, and reconcile it in a next step if it matters. Insights it confirms need no sentence; the evidence already says it.

## Next steps

Each one names an experiment or a declaration and says what its result would settle. Order them by leverage on the call. The step that could reverse the recommendation, or turn a caveat into evidence, comes first. Two things to keep apart:

- When an interval is already clear of a line, more cases won't move the verdict. A change to the candidate will.
- When the call waits on a business choice (a price per point, a margin nobody declared), the next step is a declaration, not a run.
