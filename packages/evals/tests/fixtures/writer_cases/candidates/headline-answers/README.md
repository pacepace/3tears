# Candidate skill: headline-answers

A copy of the shipped analyst skill whose editor raises five more must_fix findings, each from the owner's reading of
the rung-5 pages on 2026-11-06 (findings H1 to H14 in the reset's presentation review). Every file the analyst or the
editor reads agrees with them: `SKILL.md`, the principles, the writing guide, the story forms and the exemplars state
the same rules and show headlines that follow them.

## Files changed

| File | Change |
|---|---|
| `README.md` | this record, in place of the shipped skill's notes for its maintainers; no session reads it |
| `SKILL.md` | the headline and decision rows, the Method row, the weakest-margin fact, and a paragraph on what the page says |
| `editor.md` | five must_fix findings, the weakest-margin fact, a caveat line answering an unrecorded provider, and code's own labels left alone |
| `exemplars/kairo-round2.md` | its headline, its cost and latency caveats, and its confound as a caveat line rather than Watch |
| `exemplars/r1-model-selection.md` | its headline and its cost caveat |
| `exemplars/r2-config-sweep.md` | its headline, and its near misses' rates with their counts |
| `exemplars/r3-classifier.md` | its headline and its cost caveat |
| `exemplars/r4-tradeoff.md` | its cost caveat, its undeclared rule, its Watch, and what does not transfer |
| `principles.md` | the weakest margin in Watch only when it could change the call; absences, an unrecorded provider and a contradicted prior insight as caveat lines |
| `stories.md` | every headline example one sentence with its figure, Watch's slot, the exploratory headline, and the tradeoff form's absences |
| `writing.md` | terms defined at first use, the headline, a volume's unnamed count, and Watch and absences before you submit |

Every other file is the shipped skill's, byte for byte.

## Rules

| Rule | Findings it answers |
|---|---|
| The headline answers the question asked (with none declared, the first grader's reading), in one sentence of at most 25 words, with the answer's figure | H1, H9 |
| An absence is a caveat line, anywhere on the page, never headline matter and never a sentence of its own | H9, H10; H5, H12 by analogy |
| Watch lists only what could change the call; when every alternative read worse and cost more, it says so in one line | H3 |
| A rate on the first screen carries its count | H11 by analogy |
| A term the reader did not bring is defined at first use or replaced | H2 |

H5, H11 and H12 were read on code pages, whose text the package writes and no skill reaches. This candidate carries
their rule to the writer's page; the code pages need their own package change. Code's own labels on a writer's page
(the confidence's caveat lines) are left alone: the rules apply to the analyst's text.

## Digests

| Skill | Skill digest | Analyst text digest | Editor text digest |
|---|---|---|---|
| shipped | `4e14e276b568647e68450b3b025204213d2bddcc3a2ff6b838550224149788b8` | `063768a69822c5655b17d434470f3a655506810d0cc5334d6f4c3216c3ba368b` | `07e15b95eed5b308c0986f019b0f88fb56bce14d18f99f0fa84b77cb295d5611` |
| headline-answers | printed by the run | `f6c737d08ce6327796d4e45137b2223ffcde186effb12fa95dee63110a049ef9` | `d383f0b1dd018afeb59c7c8042f80047ff0da7d09addf5d9ea4644d535eaac30` |

The candidate's skill digest covers this file, so this file cannot hold it: the acceptance readout prints it, and
`acceptance.json` records it under `candidate_skill`. The analyst's text is read with no note withheld.

## Running it

From the repository root, over the 25 held-out cases the held-out README gathers into `.prawduct/acceptance-cases`, at
the $1.50 report budget and $104 cap of the last no-editor record:

```bash
uv run python -m packages.evals.tests.fixtures.writer_cases.writer_acceptance \
  --candidate-skill packages/evals/tests/fixtures/writer_cases/candidates/headline-answers \
  --cases .prawduct/acceptance-cases --out .prawduct/acceptance-headline-answers \
  --held-out-fraction 0.99 --writer claude-sonnet-5-5 --effort medium \
  --budget-usd 1.5 --turns 30 --request-s 50 --cap-usd 104
```

## The diff

```diff
--- a/SKILL.md
+++ b/SKILL.md
@@ -19,24 +19,26 @@
 5. **Render and check.** Draw each claim's block with `block_render` and read its values table. Then send the whole draft to `report_check`: it returns every mechanical verdict the store would refuse it for, then the page as a reader and the editor will read it. Fix every verdict, reread the page as a reader and fix what is unclear, crowded or misleading, and check again. At most three rounds.
 6. **Submit to the editor.** Mechanical checks run first and return to you before any editor call. Address the editor's findings and resubmit, at most two rounds. Unresolved findings are stored and shown in Method.
 
-**Two kinds of feedback.** *Mechanical* verdicts you fix: an unresolved figure key, a typed digit, a block that doesn't render, a doubled unit, an id above the appendix, a caveat with no landing, a confidence form that doesn't match its caveats, a missing appendix or Method. *Interpretive* facts go to the editor, which judges them in context: the recommendation failed a declared gate, a warning on it is unanswered, your call disagrees with `options_dominance`, the weakest margin is missing from Watch, a caveat code attaches to a claim's evidence is carried nowhere, your confidence reads high over caveats it leaves open. Overriding one can be right; the page must say why.
+**Two kinds of feedback.** *Mechanical* verdicts you fix: an unresolved figure key, a typed digit, a block that doesn't render, a doubled unit, an id above the appendix, a caveat with no landing, a confidence form that doesn't match its caveats, a missing appendix or Method. *Interpretive* facts go to the editor, which judges them in context: the recommendation failed a declared gate, a warning on it is unanswered, your call disagrees with `options_dominance`, the weakest margin is missing from Watch (a caveat line citing it answers that when no move within its interval could change the call), a caveat code attaches to a claim's evidence is carried nowhere, your confidence reads high over caveats it leaves open. Overriding one can be right; the page must say why.
 
 ## The report document (`report-doc/1`)
 
 | Field | Purpose |
 |---|---|
 | `story` | The dominant form (`stories.md`). It describes the report; it selects nothing. |
-| `headline` | The call and its reason in one sentence, or the trade, or what is unresolved. Never the eval's question restated. |
+| `headline` | The answer to the question the campaign asked (where it declared none, to the reading of the first grader: the measure the page's lead reads, the first in the report input's grader order (`measure_order`) that a contrast tested), with the figure it turns on, in one sentence of at most 25 words: the call and its reason, or the trade, or what is unresolved. Never the eval's question restated, and never an absence. |
 | `lede` | At most two sentences: the strongest supporting fact, and what would change the answer. |
 | `confidence` | `high`, `high_if_immaterial` or `low`, and every open caveat with its `material_if` clause (`principles.md`). |
-| `decision` | Four slots from the shared vocabulary that fit the story: recommendation, why_not, watch, next; the_trade, where_it_flips, our_lean, what_settles_it; what_is_tangled, known_despite_it. |
+| `decision` | Four slots from the shared vocabulary that fit the story: recommendation, why_not, watch, next; the_trade, where_it_flips, our_lean, what_settles_it; what_is_tangled, known_despite_it. `watch` lists only what could change the call; when every alternative read worse and cost more, it says so in one line and lists nothing else, the weakest margin aside when it could change the call. |
 | `claims` | The story. Order is page order. Each has `text`, `weight` (lead, support, detail) and `evidence`: one block kind, a tool query, a caption. The first claim best carries the headline. |
 | `explore` | Controls for business choices (a margin, a floor, a price per point), what each binds, and in words where the outcome flips, or that it can't. |
 | `omitted` | Each substantial thing you saw and left out, with a one-line reason. The editor checks these. |
 | `next` | Each step names an experiment or a declaration, and `settles` says what its result would decide. |
-| `method` | Term and definition pairs: cases and repeats, what pass means, intervals and correction, the judge, cost basis, what wasn't compared, the source. |
+| `method` | Term and definition pairs: cases and repeats, what pass means, intervals and correction, the judge, cost basis, what was compared, the source. |
 
 A block's `query` uses the tools' query language, so the renderer resolves it from the data and a block can never show a figure that doesn't exist.
+
+**The page says what the report is, never what it is not.** An absence anywhere on the page (no margin declared, no recommendation, no question, no provider recorded, data a comparison needs) is a caveat line (in the confidence, or beside the one claim it qualifies), with its `material_if` clause: never the headline, never a sentence of its own. Code prints some caveat lines itself; leave them as they are. A rate on the first screen (the headline, lede, confidence and decision block) carries its count through its own key ("20 of 20", never "100%" alone). A term the reader did not bring is defined in plain words at its first use, or replaced by them. The editor raises each of these as a must_fix.
 
 ## Tools
 
--- a/editor.md
+++ b/editor.md
@@ -6,7 +6,7 @@
 
 - **The page**, as a reader sees it: the headline, lede and flag, the confidence line, the decision block, the claims in page order, each with its evidence block and that block's values table, then explore, next steps, the full-results appendix and Method. Every number on it was printed from the campaign's data through a figure key. Mechanical checks have already passed: every key resolves, every block renders, every caveat has a landing, the confidence form matches its caveats, and no internal id shows above the appendix. Don't spend your review on those.
 - **`omitted`**: each substantial thing the analyst saw and left out, with a one-line reason. Check that none of it would change a reader's decision.
-- **Computed facts**, when the message carries them: things code found in the data that bear on the call, such as a recommended arm failing a declared gate, an unanswered warning on it, a call that disagrees with `options_dominance`, the weakest margin missing from Watch, a caveat code attaches to a contrast a claim reads that the claim does not carry, who served the model calls of an arm whose cost the page cites, or a confidence that reads high over caveats it leaves open. Each is a fact for you to weigh, not a verdict. Overriding one can be right, but the page must say why. A fact the page neither addresses nor explains is a finding.
+- **Computed facts**, when the message carries them: things code found in the data that bear on the call, such as a recommended arm failing a declared gate, an unanswered warning on it, a call that disagrees with `options_dominance`, the weakest margin missing from Watch (answered by a caveat line citing it when no move within its interval could change the call), a caveat code attaches to a contrast a claim reads that the claim does not carry, who served the model calls of an arm whose cost the page cites, or a confidence that reads high over caveats it leaves open. Each is a fact for you to weigh, not a verdict. Overriding one can be right, but the page must say why. A fact the page neither addresses nor explains is a finding.
 - **Your earlier findings**, from the second round on, with the revised page, each with its id (`f1.2`). Check whether each was resolved, and mark it in your answer: a finding still open is continued by one of yours, a finding the page now answers goes under `resolved`. Don't re-raise a finding the page now answers, and don't hold back a new one because it wasn't raised before.
 
 ## Your tool
@@ -45,8 +45,15 @@
   - a finding said to be immune to the caveats, or impossible to reverse, while a caveat attached to its own evidence could move it;
   - a figure whose sign contradicts what its sentence says it measures (a "saving" printed below zero), since it changes what a reader concludes;
   - one measure under two names where a reader would take them for two measures;
-  - a claim that ranks models by cost with no provider named: a cost is the provider routing's, not the model's, so a cost comparison names who served each model (the `cost_served_by` fact states it, per arm, where the page cites a cost).
+  - a claim that ranks models by cost with no provider named: a cost is the provider routing's, not the model's, so a cost comparison names who served each model (the `cost_served_by` fact states it, per arm, where the page cites a cost). Where no provider is recorded, a caveat line saying the costs are what this run paid answers it.
   - a confidence that reads high over an open caveat it does not stand behind: one that could plausibly be material, or whose clause does not say what would make it matter and why the call holds until then. The fix is to say which findings the high applies to and what it leaves open, or to make the form low.
+  - a headline that does not answer the question the campaign asked, with the figure the answer turns on, in one sentence of at most 25 words. Where the campaign declared no question, the headline answers the reading of the first grader: the measure the page's lead reads, the first in the report input's grader order (`measure_order`) that a contrast tested, with that reading's figure. A headline that restates the question, leads with what is missing, or runs past 25 words fails the reader at the first sentence they read.
+  - an absence stated anywhere on the page as headline matter or as a sentence of its own: "nothing declares whether that pass rate is enough", "this report recommends nothing", "no recommendation was written", "no question was declared". What the campaign did not declare, run, record or decide is a caveat line (in the confidence, or beside the one claim it qualifies), with what would make it matter; that is its one form. The page says what it is, never what it is not.
+  - a Watch item that could not change the call. The weakest margin stays in Watch, by its `weakest.*` keys, when a move within its interval could change the call; when none could, it is a caveat line instead. When every alternative read worse and cost more, Watch says so in one line and lists nothing else, the weakest margin aside when it could change the call.
+  - a rate on the first screen without its count: "20 of 20", never "100%" alone, since a rate over 3 cases and one over 300 read alike without it.
+  - a term the reader did not bring, such as "unnamed provider routing", neither defined in plain words at first use nor replaced by them.
+
+  These five are about the analyst's text. The labels code prints on the page, such as the confidence's "Nothing declares what is good enough", "No arm is the control" or "Latency is not compared", are caveat lines already and the analyst cannot change them: never raise a finding on one.
 - **`should_fix`**: the page would be clearer, tighter or better supported, but a reader acting on it now would not be misled.
 
 What sends the report back, while revisions remain:
--- a/exemplars/kairo-round2.md
+++ b/exemplars/kairo-round2.md
@@ -7,14 +7,14 @@
 **What I noticed first.**
 - Neither candidate is better on anything.
 - Both cost more per reply as this run served them, and each paired cost gap against September is clear of zero.
-- Those costs are this run's provider routing's, not the models' own. The snapshot names no provider for any arm, so the routing is unknown, and the cost claim says so.
+- Those costs are what this run paid, not the models' own. The snapshot names no provider for any arm, so a caveat line beside the cost claim says the costs are this run's and another provider may charge differently.
 - That ends the decision before any judged evidence matters. The call rests on cost measured at this run's routing.
 - Measured is not untouched. The snapshot attaches the reasoning-ratio confound to 1017's cost contrast, not only to its judged ones. 1017 spends less of its output reasoning than September (55% against 76%), so at a matched budget it would likely cost more, not less. The confound's likely direction favours the call, and the cost claim carries it and says so.
 - Cost was read on most trials, not all. The cost claim says how many, and that the order holds even if every missing trial cost nothing.
 
 **The trap.** The pipeline reads qwen3.8-flash as "no worse than the margin". That's true, but its interval (−10.8 to −3.4) excludes zero: it is clearly worse, by an amount the margin happens to allow. Leading with the margin verdict would make a regression read as a pass. So the lede states the size first, then the margin.
 
-**Story.** Decision. "Keep September. Both candidates cost more per reply and stay in character less often." Confidence is high, if the caveats aren't material:
+**Story.** Decision. "Keep September: both candidates cost more per reply, and qwen3.8-flash stays in character about 7 points less often." Confidence is high, if the caveats aren't material:
 - **The judge is uncalibrated.** This only matters if it misreads in-character replies, and the deciding finding (cost) doesn't depend on it.
 - **The aspect and control were chosen afterward.** This only matters if a different choice would have favored a candidate.
 - **The late margin:** checked, not open. The call holds at every margin.
@@ -28,9 +28,9 @@
 **What I left out, and why.**
 - **Depth, pass^k and precision tables.** They describe the method; none changes the call. They go to the appendix.
 - **Small strata** (under 10 cases) with noisy deltas: shown only in the full table, never as leads.
-- **Latency.** Not comparable, because runs shared the machine. One line in Method.
+- **Latency.** Not comparable, because runs shared the machine. Code's own caveat line in the confidence carries it.
 
-**What I'd watch.** 1017 spends much less of its output reasoning (55% against 76%). Its in-character drop may partly be the effort budget, not the model, and its cost may be too: a matched budget would likely raise it. That goes in Watch beside both of 1017's readings, its in-character drop and its cost, not in the page-level flag. The next step that re-runs 1017 at a matched budget settles both.
+**What I'd watch.** 1017 spends much less of its output reasoning (55% against 76%). Its in-character drop may partly be the effort budget, not the model, and its cost may be too: a matched budget would likely raise it. Its likely direction favours the call, so it could not change it: it is a caveat line beside both of 1017's readings, its in-character drop and its cost, not a Watch item and not the page-level flag. The next step that re-runs 1017 at a matched budget settles both.
 
 **Next steps, each with what it settles.**
 - Fix off-lane brush-offs in the directives: does that close most of the gap?
--- a/exemplars/r1-model-selection.md
+++ b/exemplars/r1-model-selection.md
@@ -4,13 +4,13 @@
 
 **What I noticed first.**
 - Read task by task, the passing model that cost least in this run differs: Brisk-S on billing, Open-70B on small talk, Atlas-M elsewhere. Routing that way would cut blended cost by 34%.
-- Those costs are this run's provider routing's, not the models' own. The snapshot names no provider for any arm, so the routing is unknown, and the page says so wherever it compares them.
+- Those costs are what this run paid, not the models' own. The snapshot names no provider for any arm, so one caveat line says the costs are this run's and another provider may charge differently.
 - Both lower-cost models fail speculation bait (81% and 89% against 95%), and their intervals sit clear below the floor.
 - Atlas-L has the best quality but misses the latency budget on four of five tasks.
 
 **The trap.** The gate matrix invites reading one row at a time and routing each task to its cheapest passing cell. But speculation bait isn't a segment you can route around. It shows up inside billing and small talk too. Any model that receives traffic has to pass it, so the 34% routed plan is blocked, not available. The trap is treating a cross-cutting aspect as one more task. The headline says so before the reader can do the per-row arithmetic.
 
-**Story.** Decision. "Ship Atlas-M on all traffic. It is the only candidate that clears every gate." Confidence is high, if one caveat isn't material: Atlas-M's issue resolution in troubleshooting is 90% at a floor of 90%, interval ±4.4. That is the recommendation clearing its own gate on its point estimate alone. It is material if the true rate is below the floor, since then no single model passes everywhere.
+**Story.** Decision. "Ship Atlas-M on all traffic: it is the only one of 4 candidates that clears every gate." Confidence is high, if one caveat isn't material: Atlas-M's issue resolution in troubleshooting is 90% at a floor of 90%, interval ±4.4. That is the recommendation clearing its own gate on its point estimate alone. It is material if the true rate is below the floor, since then no single model passes everywhere.
 
 **Lead evidence.** The gate matrix: task by candidate, each cell's status with its binding reason, the cheapest passing cell marked, and the speculation row flagged as applying to all traffic. The matrix shows the decision and the reason the obvious saving is unavailable in one view. A score heatmap would hide which criterion failed.
 
--- a/exemplars/r2-config-sweep.md
+++ b/exemplars/r2-config-sweep.md
@@ -9,7 +9,7 @@
 
 **The trap.** "The only configuration in the box" sounds decisive, and the scatter makes it look so. But it is in the box on its point estimate, by less than a sixth of its interval, and it was chosen as the one of eighteen that crossed the line, which makes its estimate optimistic. Reported as a clean win, a reader would lock it in. The recommendation stands, because nothing else passes and it is cheap, but the margin is part of the story, not a footnote.
 
-**Story.** Decision. "Use Brisk-S with low reasoning and one retry." Confidence is high, if one caveat isn't material: the pass gate is unsettled for the recommendation. It is material if the true rate is under 97%, because then nothing passes both gates and the question becomes a tradeoff between accuracy and the latency budget. The near-miss alternatives belong in why_not: the same configuration without retry (95.3%), and high-reasoning Brisk-S (98.0%, 870 ms).
+**Story.** Decision. "Use Brisk-S with low reasoning and one retry: it alone of 18 configurations clears both gates, its pass rate 0.14 points above 97%." Confidence is high, if one caveat isn't material: the pass gate is unsettled for the recommendation. It is material if the true rate is under 97%, because then nothing passes both gates and the question becomes a tradeoff between accuracy and the latency budget. The near-miss alternatives belong in why_not, each rate with its count: the same configuration without retry (95.3%), and high-reasoning Brisk-S (98.0%, 870 ms).
 
 **Lead evidence.** The gate-space scatter: pass rate against p95 on a log axis (latency spans 18×), both gate lines, and a path per model through its reasoning levels. It shows all 18 options and both gates at once, and how close the pick sits to the edge. Next, lever-effect bars, because a complete grid makes average effects meaningful once you have checked that the levers don't interact.
 
--- a/exemplars/r3-classifier.md
+++ b/exemplars/r3-classifier.md
@@ -4,13 +4,13 @@
 
 **What I noticed first.**
 - Nano-XS with few-shot cost about a twenty-first of anything that passes in this run, with 90.1% balanced and 92.0% traffic-weighted accuracy. It fails one cell: complex news at 76.7%.
-- That ratio is this run's provider routing's, not the models' own. The snapshot names no provider for any configuration, so the routing is unknown, and every cost comparison on the page says so.
+- That ratio is what this run paid, not the models' own. The snapshot names no provider for any configuration, so one caveat line says the costs are this run's and another provider may charge differently.
 - Complex news is the weakest cell for every configuration.
 - The best prompt depends on the model. The rubric prompt helps Atlas-M and Brisk-S, and costs Nano-XS six points against few-shot.
 
 **The trap.** The overall number. Simple conversational turns are almost a third of the corpus, so a traffic-weighted average hides the cell that rules Nano-XS out. Lead with overall accuracy and the lowest-cost configuration looks like the answer. The campaign's gates already guard against this (balanced accuracy and a cell floor), but the analyst has to lead with the cell, not the average. A second trap: describing the prompt as a main effect ("few-shot helps", "the rubric helps"). Averaged over models, that would be true and wrong for Nano-XS.
 
-**Story.** Decision. "Use Brisk-S with the few-shot prompt. Of the configurations that stay above 80% in every cell, it cost the least in this run." The cost claim says the routing behind that order is unknown. Labels are human-adjudicated, so there is no judge caveat. Confidence is high.
+**Story.** Decision. "Use Brisk-S with the few-shot prompt: of the 9 configurations, it is the cheapest that stays above 80% in every cell." A caveat line beside the cost claim says the costs are what this run paid. Labels are human-adjudicated, so there is no judge caveat. Confidence is high.
 
 **Lead evidence.** The model × prompt lattice, with cost, latency and gate status per tile. It shows the pick, the interaction, and why the more accurate options are out (latency), in one grid. Then the target-matrix heatmaps for the pick, the lower-cost alternative and the most accurate, so the reader sees the cell that decides.
 
--- a/exemplars/r4-tradeoff.md
+++ b/exemplars/r4-tradeoff.md
@@ -6,12 +6,12 @@
 - At this run's routing, A saves about $7.6k a month (30%) and loses 6.1 points in character (interval 4.9 to 7.3). Every other aspect holds.
 - Corvid loses only 2.2 points and saves the most per point lost of any full switch. But it breaks the speculation guardrail at 92%, and its drop there is clear of zero.
 - B cost the least in this run and breaks the facts guardrail at 89.4%.
-- Every saving is this run's provider routing's, not the models' own. The snapshot names no provider for any arm, so the routing is unknown, and the page says so beside the prices.
+- Every saving is what this run paid, not the models' own. The snapshot names no provider for any arm, so one caveat line beside the prices says they are this run's and another provider may charge differently.
 - A's loss grows with conversation depth: close for the first turns, 18 points behind past turn ten.
 
 **The trap.** Corvid looks like the sensible middle path: a small personality loss for real savings. Pricing it beside A would put an option that breaks a standing policy into the trade. Guardrail failures remove options before any pricing. The second trap is deciding at all. With no declared price for a point of personality, any recommendation is a business choice. The report prices the trade and says who makes the call.
 
-**Story.** Tradeoff. "Switching to A saves $7.6k a month but loses 6 points of personality, mostly in long conversations." The first screen says plainly that no rule was set in advance, and names the persona owner as the decider. The lean (labelled as ours): before switching, run a hybrid that hands conversations back to Production after turn five. The depth breakdown is the seam that suggests it: the loss sits in long conversations, and the savings sit in short ones. Nobody has run that mix, so the report prices only the options that were measured and names the hybrid as the next experiment.
+**Story.** Tradeoff. "Switching to A saves $7.6k a month but loses 6 points of personality, mostly in long conversations." No rule set in advance prices a point of personality: that is a caveat line in the confidence, and the decision block names the persona owner as the decider. The lean (labelled as ours): before switching, run a hybrid that hands conversations back to Production after turn five. The depth breakdown is the seam that suggests it: the loss sits in long conversations, and the savings sit in short ones. Nobody has run that mix, so the report prices only the options that were measured and names the hybrid as the next experiment.
 
 **Lead evidence.** Change against Production on every aspect, as diverging bars with intervals, so the reader sees in one chart why B and Corvid are out and that A loses only personality. Then in-character rate by turn depth, which is the seam the recommended hybrid would exploit. Then the savings-against-loss map with a price line, which carries the trade itself.
 
@@ -25,7 +25,7 @@
 
 **What I'd watch.**
 - A's speculation rate is 96.4% against the 95% guardrail. Its drop from Production reaches 2.8 points at the interval's edge, so the guardrail isn't settled. That bears on every option that uses A, including the hybrid.
-- The hybrid assumes Production's voice returns after the handoff. That is untested.
+- The hybrid rests on Production's voice returning after the handoff, which the hybrid's run measures.
 
 **Next steps, each with what it settles.**
 - Run the hybrid (hand back to Production after turn five) as a measured option on the same cases: settles what it keeps of the savings and of the personality, so it can be priced like the others.
@@ -33,4 +33,4 @@
 - The persona owner declares a price per point: turns the tradeoff into a decision.
 - More speculation bait on A: settles its guardrail.
 
-**What doesn't transfer.** A hybrid existed because the loss concentrated by depth and cost varied with depth. Most tradeoffs have no such seam. The human preference anchor was a lucky extra. Without one, Watch says no evidence links points to what users notice.
+**What doesn't transfer.** A hybrid existed because the loss concentrated by depth and cost varied with depth. Most tradeoffs have no such seam. The human preference anchor was a lucky extra. Without one, a caveat line in the confidence says the trade is priced in judged points alone.
--- a/principles.md
+++ b/principles.md
@@ -9,7 +9,7 @@
 ## Honest emphasis
 
 - **Lead with the size of a change before its verdict against a line.** A candidate whose interval sits wholly below zero is worse, even when the margin allows that much loss. "Within the margin" said first makes a regression read as a pass. Say how much worse, then whether the margin allows it.
-- **Show the recommended option's weak axes.** A recommendation whose weak side is hidden reads as stronger than the evidence. The weakest margin belongs in Watch. "Weakest" is code's measure, not yours: the gap smallest relative to its line (the gap over the line's value), which the `weakest.*` keys name. Cite those keys in Watch. If another margin matters more to the call (fewer points, or fewer half-widths), Watch still names the weakest and says why the other matters more, stating both in the same terms.
+- **Show the recommended option's weak axes.** A recommendation whose weak side is hidden reads as stronger than the evidence. The weakest margin belongs in Watch when a move within its interval could change the call; when none could, it is a caveat line citing the same keys, since Watch lists only what could change the call. "Weakest" is code's measure, not yours: the gap smallest relative to its line (the gap over the line's value), which the `weakest.*` keys name. If another margin matters more to the call (fewer points, or fewer half-widths), Watch names the weakest and says why the other matters more, stating both in the same terms.
 - **Status follows the direction of the change, not the margin.** A worse result stays marked as worse even when it is within the margin. Approval colors on a regression read as approval.
 - **Frame nothing stronger than its interval supports.** A point estimate that clears a gate while its interval crosses it is *borderline*. It counts as passing for the call, but it goes in Watch with what would settle it: `settle_cases` sizes that run.
 - **State findings in the direction the evidence supports.** A refuted claim at low confidence is still a refutation. Don't let a weak tier blur which way the evidence points.
@@ -20,7 +20,7 @@
 
 - Method statistics (repeats, depth, pass^k, precision tables) describe how the evidence was gathered. They go to Method or the appendix, never between the decision and the evidence.
 - Small, noisy strata appear in the full tables, never as leads.
-- If the data a comparison needs is missing, say so in one caption line. Never fill it in.
+- If the data a comparison needs is missing, that is a caveat line. Never fill it in.
 
 ## Decisions: survivors and dominance
 
@@ -28,7 +28,7 @@
 
 - **One survivor dominates:** a decision. The story says why it wins and what it gives up.
 - **Several survive and none dominates:** if the campaign declared a rule that ranks the axes, the rule decides. If not, the honest report is a tradeoff. Price it and let the owner choose. You may state a lean, labelled as yours.
-- **Nothing survives, one arm was measured, or nothing separates the options:** the report says what is missing and what would settle it.
+- **Nothing survives, one arm was measured, or nothing separates the options:** the report states what was measured and what would settle the call.
 
 If your call disagrees with `options_dominance`, the editor will see that. Disagree only for a reason the page states, such as an axis the computation can't weigh.
 
@@ -37,9 +37,9 @@
 - A gate with no figure for an option is not passed. Absence of evidence isn't a pass.
 - An aspect that occurs inside every task (safety, speculation) gates every option that would receive any traffic, not one task. Reading it per task invents savings.
 - Gate the asked aspect on its own readings. If a floor speaks for a different measure than the reading does, show it separately and name the mismatch in Method. Don't merge the two.
-- "Pass" means whatever the campaign defined. If the campaign doesn't say what makes a trial pass, say so plainly. Don't guess.
+- "Pass" means whatever the campaign defined. If the campaign doesn't say what makes a trial pass, that is a caveat line (code often prints it already). Don't guess.
 - A latency or cost win needs the delivery rate beside it. An arm can buy a shorter tail by declining work.
-- **A cost belongs to the provider routing, not the model.** One model served by two providers (another host, another quantization) can cost several times as much, while its quality barely moves. A cost comparison between models names the provider each was served by (`campaign_overview` gives each arm's `served_by`), and never ranks models on cost alone as if the cost were the model's. Where an arm's pricing names no provider, say the cost's routing is unknown. A tool the model called (`tool_spend`, such as a web-search API) is spend in the cost, never a provider of the model.
+- **A cost belongs to the provider routing, not the model.** One model served by two providers (another host, another quantization) can cost several times as much, while its quality barely moves. A cost comparison between models names the provider each was served by (`campaign_overview` gives each arm's `served_by`), and never ranks models on cost alone as if the cost were the model's. Where an arm's pricing names no provider, that is one caveat line in plain words: the costs are what this run paid, and another provider may charge differently. On the page, name the provider that served a model; never write "routing". A tool the model called (`tool_spend`, such as a web-search API) is spend in the cost, never a provider of the model.
 - Rank on the scope that matches what was changed. A subsystem change read on whole-turn figures can pick the wrong arm.
 
 ## Confounds
@@ -48,7 +48,7 @@
 
 A confound qualifies every finding whose evidence it is attached to, not only the one it was noticed on. `contrasts_list` names each contrast's confounds and `caveats_list` what each caveat touches. A setting that moved between arms (a reasoning budget, a retry) moves cost and latency as well as quality, so measured telemetry is not immune to it (`statistics.md`, Caveats on the evidence).
 
-Say a caveat once, at the level it applies to. A caveat that touches the whole comparison (late criteria, an uncalibrated judge, a shared-machine latency) belongs in the confidence. One that touches one arm or one finding (that arm's reasoning budget, one short run) sits beside that claim, usually in Watch.
+Say a caveat once, at the level it applies to. A caveat that touches the whole comparison (late criteria, an uncalibrated judge, a shared-machine latency) belongs in the confidence. One that touches one arm or one finding (that arm's reasoning budget, one short run) sits beside that claim, in Watch only when it could change the call.
 
 ## Confidence
 
@@ -66,7 +66,7 @@
 
 ## Prior insights
 
-When this run contradicts an earlier finding, say so in one sentence in Watch, and reconcile it in a next step if it matters. Insights it confirms need no sentence; the evidence already says it.
+When this run contradicts an earlier finding, say so in one sentence beside the claim it bears on, in Watch when it could change the call, and reconcile it in a next step if it matters. Insights it confirms need no sentence; the evidence already says it.
 
 ## Next steps
 
--- a/stories.md
+++ b/stories.md
@@ -1,14 +1,16 @@
 # Story forms
 
 A story form is a way of answering, not a category the eval belongs to. Choose it after investigating, from what the evidence supports. The same campaign design can produce any of them: a control against two candidates can end in a decision, a tradeoff or a diagnosis, depending on what the numbers say. The `story` field names the dominant form so the editor knows what you meant to deliver. It selects nothing.
+
+Every headline below is one sentence of at most 25 words carrying the figure it turns on. The examples show each figure as a reader sees it; you write it through its key.
 
 ## Decision
 
 **When:** the evidence supports one call. Usually one option survives the gates and dominates the others, or a declared rule ranks the survivors.
 
-**Headline:** the call as an instruction, with its reason. "Keep the current router. The candidate costs more per request and resolves no more tickets."
+**Headline:** the call as an instruction, with its reason. "Keep the current router: the candidate costs 18% more per request and resolves no more tickets."
 
-**Slots:** recommendation (with the figures that make it), why_not (each rejected option and the one thing that rules it out), watch (the weakest margin, a confound scoped to one arm, a contradicted prior insight), next (the step that could change the call, and which way).
+**Slots:** recommendation (with the figures that make it), why_not (each rejected option and the one thing that rules it out), watch (only what could change the call: the weakest margin when a move within its interval could, a confound scoped to one arm, a contradicted prior insight), next (the step that could change the call, and which way).
 
 **Lead evidence:** whatever shows the reason for the call. If the call rests on two things (the asked question is answered, and cost decides), each needs its own claim, side by side. One chart alone would hide half the reason.
 
@@ -18,11 +20,11 @@
 
 **When:** two or more options survive, none dominates, and the campaign declared nothing that ranks the axes, such as an exchange rate or a priority.
 
-**Headline:** the trade, with both sides and where the loss sits. "Switching halves the cost but answers fewer questions correctly, mostly on the hardest ones."
+**Headline:** the trade, with both sides and where the loss sits. "Switching halves the cost per request but answers 6 points fewer questions correctly, mostly on the hardest ones."
 
 **Slots:** the_trade, where_it_flips (the price bands where the best option changes), our_lean (labelled yours, with the band it holds for), what_settles_it.
 
-**What it owes:** one axis priced in the other at real volume (`price_tradeoff`), over the two to four options a decider would actually weigh. Too many variants make the bands too thin to act on. If the loss concentrates in one stratum, a mix that avoids it (route that stratum to the stronger option) may be the most useful option, but nobody ran it, so no figure states its cost or gain. Never price it. Recommend running it as a next step, say what its measurement would settle, and name the seam that suggests it. Show any evidence of what users notice. If there is none, say so in Watch. Say plainly that nothing was declared in advance, and name who makes the call.
+**What it owes:** one axis priced in the other at real volume (`price_tradeoff`), over the two to four options a decider would actually weigh. Too many variants make the bands too thin to act on. If the loss concentrates in one stratum, a mix that avoids it (route that stratum to the stronger option) may be the most useful option, but nobody ran it, so no figure states its cost or gain. Never price it. Recommend running it as a next step, say what its measurement would settle, and name the seam that suggests it. Show any evidence of what users notice. Where there is none, and where nothing declared in advance ranks the axes, each is one caveat line in the confidence, never a sentence of its own, a Watch item or the headline. Name who makes the call.
 
 **Usual failure:** deciding anyway. A lean presented as a recommendation hides the fact that the call is a business choice.
 
@@ -30,7 +32,7 @@
 
 **When:** the result can't be read cleanly because two things are tangled. A model changed together with its reasoning budget. The judge is unreliable exactly where the arms differ. A short run touches one arm.
 
-**Headline:** what can't be concluded and why, plus what still holds. "The candidate's drop can't be credited to the model: its reasoning budget changed too. It still costs more."
+**Headline:** the tangle and what still holds. "The candidate's 9-point drop can't be credited to the model, since its reasoning budget changed too, and it costs 30% more."
 
 **Slots:** what_is_tangled, known_despite_it, what_settles_it, next.
 
@@ -42,7 +44,7 @@
 
 **When:** one arm, read against its own thresholds or its own history. There is nothing to choose between.
 
-**Headline:** where it stands against the lines that matter, led by the nearest one. "The service holds every floor; refunds are one point above their line."
+**Headline:** where it stands against the lines that matter, led by the nearest one. "The service holds all 4 floors; refunds sit 1 point above their line."
 
 **Slots:** usually recommendation (ship, hold, or fix first), watch, next, plus why_not when an alternative was implicitly on the table.
 
@@ -52,7 +54,7 @@
 
 **When:** no question was declared, or the campaign measured things without a decision in view.
 
-**Headline:** the most decision-relevant thing learned, stated as a finding, and the question worth asking next.
+**Headline:** the most decision-relevant thing learned, stated as a finding with its figure: the reading of the first grader, which `SKILL.md` defines. "Long tickets fail 3 times as often as short ones on every arm." The question worth asking next goes in `next`.
 
 **Slots:** known_despite_it (what is solid), what_is_tangled where relevant, what_settles_it, next. The next steps matter most here. They are usually declarations: the question, the margin, the control.
 
@@ -60,7 +62,7 @@
 
 ## Mixing forms
 
-Reports often mix. A decision on cost can carry a diagnosis on the judged aspect: "Keep the current router: the candidate costs more. Whether its answers also read worse can't be known until the judge is calibrated." The headline carries the dominant form, meaning the one the reader acts on. The secondary form gets its own claim and its own slot (often Watch or what_is_tangled), and never contradicts the headline unaddressed.
+Reports often mix. A decision on cost can carry a diagnosis on the judged aspect: "Keep the current router: the candidate costs 18% more, and whether its answers read worse waits on a calibrated judge." The headline carries the dominant form, meaning the one the reader acts on. The secondary form gets its own claim and its own slot (often Watch or what_is_tangled), and never contradicts the headline unaddressed.
 
 Pick the four decision slots that fit what you are actually saying. A decision with an unresolved secondary question might use recommendation, why_not, what_is_tangled and next.
 
--- a/writing.md
+++ b/writing.md
@@ -12,13 +12,13 @@
 - **No meta-commentary about the document.** Not "this report shows", not "the following section", not "it is worth noting". Say the thing.
 - **No `key=value` labels in human text.** Ids and field names are for machines. Write "cost per 1,000 requests", never `cost_per_1k=0.42`.
 - **Arms by their names, with the control marked.** Use the short names the tools return, the same name everywhere, and mark which arm is the control. A reader should never have to decode a configuration string.
-- **Define each term of art once, or replace it.** If a reader needs "non-inferiority" to follow the page, define it in Method once. Better, say "may be up to 15 points worse".
+- **Define each term of art in plain words at its first use, or replace it.** If a reader needs "non-inferiority" to follow the page, define it where it first appears, which is often the first screen. Better, say "may be up to 15 points worse".
 - **Statistics as one clause.** "Clearly worse (the interval excludes zero)", not a paragraph about coverage. Interval method and correction go in Method.
 - **Measured facts are stated flatly.** Uncertainty belongs to the interval, stated once. "Costs 28% more" is a measurement; don't soften it. Don't hedge a measured fact with "seems" or "appears".
 - **Every number comes through a figure key, never typed.** The renderer formats it from the data, so the prose can't drift from the tables and a rerun updates it. A figure no tool returns isn't written; record the gap. **A count is a number even when written as a word**: "three of the runs", "eight of the ten cells" each come from a key, spelled through the `words` filter (`{count.…|words}`).
 - **A change is cited by its gap, never by a level.** A step's price, a saving or an uplift is the gap between two arms: cite the gap's key (a `price.*` step's cost gap, a `compare.*` difference, a contrast's delta), never one arm's total, which a reader takes for what the move costs.
 - **A saving reads as a saving.** Write a saving or a cost increase as its size (the `abs` filter) with its direction in words: "saves {gap|abs} a month", "costs {gap|abs} more". A signed gap says which arm it is measured from. A sign that contradicts its sentence (a "saving" printed below zero) tells the reader the opposite.
-- **A volume says what it counts.** A declared monthly volume is a count of something: calls, conversations, tasks. Say which wherever the page cites it, Method included. Where the input doesn't say, write that it doesn't and log a `capability_gap`; a monthly figure on an unnamed count can be off by whatever each unit holds.
+- **A volume says what it counts.** A declared monthly volume is a count of something: calls, conversations, tasks. Say which wherever the page cites it, Method included. Where the input doesn't say, that is a caveat line, and log a `capability_gap`; a monthly figure on an unnamed count can be off by whatever each unit holds.
 - **No internal id, hash or measure key above the appendix.** Trial and case ids may appear on example cards as references. The appendix may hold ids.
 - **The page talks about the campaign, never the tools.** No word about tools, blocks, renders, revisions or review rounds anywhere on the page, Method included. If no block can show something, pick another block or state the figure in a sentence through its key, and log the gap with `capability_gap`.
 
@@ -26,7 +26,7 @@
 
 A claim's text states what is true, with a subject and a direction. "Refund requests, the largest slice, carry most of the candidate's loss." Never a topic: "Failure analysis". A reader who reads only the claim sentences should get the argument.
 
-The headline is the call (or the trade, or what is unresolved) with its reason. It never restates the question, and it never quotes the campaign's own question text, which is often written in ids.
+The headline answers the question the campaign asked (where it declared none, the reading of its first grader, which `SKILL.md` defines): the call, the trade or what is unresolved, with the figure it turns on, in one sentence of at most 25 words. It never restates the question, never quotes the campaign's own question text, which is often written in ids, and never states an absence.
 
 ## Names
 
@@ -57,7 +57,8 @@
 Then check the draft against its own evidence:
 
 - **Read the headline and each claim against its block's values table, row by row and option by option.** "Every option undercuts the baseline", "no option misses the floor", "the gain grows with the budget": find each row that would make the sentence false. If one does, rewrite the sentence. A page its own chart contradicts loses the reader's trust in every other line.
-- **Watch carries the weakest margin by its `weakest.*` keys** (`principles.md`), or says why another margin matters more, in the same terms.
+- **Watch lists only what could change the call.** It carries the weakest margin by its `weakest.*` keys when a move within its interval could change the call (`principles.md`), or says why another margin matters more, in the same terms; when none could, a caveat line cites those keys. When every alternative read worse and cost more, Watch says so in one line and lists nothing else, the weakest margin aside when it could change the call.
+- **No sentence states an absence.** What was not declared, run or recorded is a caveat line.
 - **Every caveat on the deciding evidence is carried** with its direction (`statistics.md`), and no line calls a finding immune or irreversible while one is attached to it.
 - **The headline's action and Next's first step agree.** A headline that says "ship" beside a Next that says "confirm first" leaves the reader two instructions. Say which: ship now and confirm in parallel, or confirm, then ship.
 - **Every sign matches its words**, every measure has one name, and every count comes from a key.
```
