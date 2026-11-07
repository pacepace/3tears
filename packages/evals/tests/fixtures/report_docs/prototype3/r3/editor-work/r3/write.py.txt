import json, sys
out = sys.argv[1]
resolutions = [
  {"finding": "report must_fix: why_not printed 'the pick costs 21 as much' (|int dropped 'times')",
   "status": "resolved",
   "evidence": "why_not now reads 'the pick costs {cost.gap.brisk-s.few.nano-xs.few.multiple|int} times as much'; the key reads 21, so the page prints 'costs 21 times as much'. Checked against cost_per_1m: $40 against $1.9 per 1M turns (21.05x)."},
  {"finding": "report should_fix: best-of-nine selection kept open without sizing it, beside the statement that sampling noise is well inside the 0.7-point clearance; lede's 'but as the best of nine' unclear",
   "status": "resolved",
   "evidence": "best_of_n material_if now says the pick 'was picked on three gates, not on overall accuracy alone, and any lift from the pick is bounded by the same small sampling noise, so selection alone is unlikely to take the clearance back'. watch says 'Neither sampling noise nor being picked as the best of nine is likely to take that back; the open risk is fresh traffic that differs from this set.' The lede's 'but as the best of nine' clause is gone; it now ends on the fresh-traffic risk. The lede's replacement wording ('more than its sampling noise allows to be chance') is awkward but readable (see findings)."},
  {"finding": "report should_fix: 'well inside the clearance' uncheckable; Method should name the arithmetic for the overall interval",
   "status": "resolved",
   "evidence": "Method 'Intervals' now says the nine cell half-widths in the cell chart are 'squared, summed, square-rooted and divided by nine' and that 'no figure for it is printed'. Recomputed from c2's heatmap for the pick: half-widths 0.8, 0.3, 0.7, 0.9, 0.6, 0.9, 1.1, 1.4, 1.2 give sqrt(7.81)/9 = 0.31 points at 95%, well inside 0.7. The plain mean of the nine cells is 90.67%, matching the 90.7% gate figure. The kit gap (no readable combined interval) remains, outside the analyst's reach."},
  {"finding": "c4 should_fix: settle table padded and misprinted (four all-'—' columns, constant Arm and Level, '435.0'-style counts, 'Each bar' note)",
   "status": "not resolved",
   "evidence": "blocks.txt c4/1 is unchanged on every point: Trials needed one-/two-sided, Repeats of a case and Trials today are '—' on all nine rows; Arm and Level repeat one value; Needed prints '7.0', '435.0', '617.0'; the note still says 'Each bar'. A 'Sizes count: cases' column was added, also constant. These are renderer defects; the analyst narrowed the caption to the columns that carry values, which helps."},
  {"finding": "c3 should_fix: gate_scatter labels the overall line 'floor 90%' and the axis 'Balanced accuracy' while prose says 'overall accuracy'",
   "status": "resolved",
   "evidence": "The finding allowed either fix; Method 'Accuracy' now names the equivalence once: 'Charts label it balanced accuracy, and mark its line as a floor; the page keeps \"floor\" for the per-cell line.' c3's caption also calls it 'the overall accuracy line'. The chart labels themselves (renderer) are unchanged."},
  {"finding": "c5 should_fix: both levers block notes describe interactions neither block shows",
   "status": "not resolved",
   "evidence": "blocks.txt c5/1 and c5/2 notes still read 'each interaction is how that effect changes across another setting' over three effect rows each. Renderer note text; nothing in the report doc can change it."},
]
items = [
  {"n": 1, "name": "First screen", "score": 2,
   "reason": "Headline, decision and c1 give what to do (ship Brisk-S few-shot), why (the only one of nine inside all three gates; options_dominance lists it as the sole survivor), the cost ($1,200 a month, $40 per 1M) and the main risk (a 0.7-point overall clearance, with fresh traffic, not noise, named as the open risk). why_not disposes of every rival with checked figures: Nano-XS few-shot at 76.7% on complex news (interval 75.3-78.1, wholly under 80%), Brisk-S rubric and Atlas-M past 250 ms (closest 420 ms), and the zero-shot pair and Nano-XS rubric under 90% overall (86.0, 78.2, 83.9). The round-2 broken '21 as much' now prints '21 times as much'."},
  {"n": 2, "name": "Coherent story", "score": 2,
   "reason": "c1 (sole survivor) -> c2 (the cheap rival fails only complex news, by 6.2 points) -> c3 (every more accurate option, 91.5-93.2%, is past latency, nearest 483 ms) -> c4 (the pick's cells settled, narrowest +2.9) -> c5 (few-shot is Nano-XS's lever, pointing next[1]). Explore matches decision_sensitivity exactly: cell floor flips to Nano-XS few-shot at or below 76.7% and to no call above 82.9%; overall no call above 90.7%; latency no call below 184 ms, and raising it never moves the call. One wrinkle: explore and next[1] say Nano-XS few-shot 'takes the call' if it clears the floor, which rests on cost deciding between gate-passers (options_dominance weighs [gates, cost]), while the flag says 'no rule beyond the gates' was declared and the page never states that cost decides (finding)."},
  {"n": 3, "name": "Honest emphasis", "score": 2,
   "reason": "The pick's weak axes lead: the 0.7-point overall clearance in lede, flag, watch and two caveats, and complex news at 82.9% (weakest cell for all nine arms, verified by measure_read) in the recommendation, watch and c4. c2's compare bars are captioned unpaired and uncorrected, 'a lead, not a confirmed finding'. c2's 'level with the pick or ahead' covers medium science (+0.5, interval -0.8 to +1.8) and complex science (+0.3, -1.3 to +1.9), fair. The overall-noise claim is now checkable from Method. A small asymmetry: the latency explore notes that the pricier configurations' extra accuracy is an unpriced choice, but the cell-floor explore and next[1] don't note that Nano-XS few-shot, if it passes, is 0.6 points less accurate overall (90.1 vs 90.7)."},
  {"n": 4, "name": "Sound elision", "score": 2,
   "reason": "Each omission checked: traffic-weighted accuracy and macro F1 are not gates; the confusion figures read as cited (92.7 vs 92.3, 95.8 vs 95.5); examples_list returns none (kept_from empty), so 'the snapshot holds none' holds; Atlas-M misses latency at 420-1302 ms against 250; the model-lever omission matches c5's per-model blocks; the more accurate configurations' cost ($95, $260, $610 per 1M) can't change the call, since decision_sensitivity shows no latency budget makes them win over the pick on the declared criteria."},
  {"n": 5, "name": "Confounds handled", "score": 2,
   "reason": "All four caveats caveats_list holds (no_control, point_estimate:overall, point_estimate:p95, best_of_n) appear with material_if clauses tied to the story. point_estimate:overall and best_of_n are now sized against the derived 0.31-point half-width and carried into watch and next[0] as a fresh-traffic risk. no_control is scoped to the unpaired c2 comparison, and Method says those intervals read wide and uncorrected. point_estimate:p95 notes the 66 ms clearance with no interval. next[1] flags that a cell-tuned prompt is a new selection needing the same fresh-turn check."},
  {"n": 6, "name": "Actionable next steps", "score": 2,
   "reason": "next[0] (fresh held-out run of the pick) settles whether the 0.7 clearance survives outside the selection set and names the fallback (an accuracy-vs-latency trade). next[1] (complex news examples for Nano-XS few-shot, not the rubric, which cost that cell 5.8 points, verified 76.7 -> 70.9; whole configuration re-run on fresh turns) settles whether the cheaper, faster option takes the call. next[2] (owner declares whether the $1,143-a-month saving is worth a second iteration) settles whether next[1] runs. The decision 'next' slot orders them."},
  {"n": 7, "name": "Evidence fits claims", "score": 2,
   "reason": "Counterfactuals hold: c1's gate_matrix would show a second all-pass column; c2's heatmap and compare bars put the gap at complex news (+6.2, +4.4 to +8.0) with Nano-XS's interval under 80%; c3's scatter would show a point above 90.7% inside 250 ms; c4's settle table shows all nine cells 'clears', complex news nearest at +2.9 needing 617 of 3,600; c5's per-model lever blocks show +11.9 vs +5.7 and +4.7 vs +5.5. Remaining defects are renderer presentation and don't break any claim: c4's padded table, c5's interaction note, and c2's compare intervals printed at four significant figures ('-2.163 to -0.03699') beside one-decimal values, with a Note column that repeats the value."},
]
findings = [
  {"claim": "report", "severity": "should_fix",
   "finding": "The cell-floor explore ('the call flips to Nano-XS few-shot') and next[1] ('if both, it takes the call') rely on cost deciding between configurations that pass every gate; options_dominance reads the declared weighing as [gates, cost]. The page never says so, and the flag says 'no rule beyond the gates were declared', which a reader takes to mean nothing decides between two passers. Say once (Method 'Gates' or the flag) that among configurations inside every gate the cheapest is the call. Then, as the latency explore does for the pricier options, note that Nano-XS few-shot would win while reading 0.6 points lower overall (90.1% vs 90.7%), a trade the owner may want to weigh in next[2]."},
  {"claim": "report", "severity": "should_fix",
   "finding": "The lede calls Nano-XS few-shot 'the nearest cheaper rival'. On cost, Brisk-S zero-shot ($28 per 1M against the pick's $40) is nearer; Nano-XS few-shot ($1.9) is the cheaper rival nearest to passing, the only cheaper one inside both the overall accuracy gate and the latency budget. Say which sense of 'nearest' is meant."},
  {"claim": "report", "severity": "should_fix",
   "finding": "Two phrases read awkwardly on the first screen and in c5: the lede's 'by only 0.7 points, more than its sampling noise allows to be chance' (say: 'more than sampling noise can explain') and c5's 'the rubric is out there on latency' (say: 'the rubric fails the latency budget')."},
  {"claim": "c4", "severity": "should_fix",
   "finding": "Carried from round 2, renderer: the settle table has four all-'—' columns (Trials needed one-/two-sided, Repeats of a case, Trials today), constant Arm, Sizes count and Level columns, counts printed as '7.0', '435.0', '617.0', a wrong-side column mixing '0%' and '< 0.000001%', and a note that says 'Each bar' on a table. Drop columns that are constant or empty and print counts as integers."},
  {"claim": "c2", "severity": "should_fix",
   "finding": "Renderer: the compare bars' intervals print at four significant figures ('-2.163 to -0.03699', '+0.2272 to +2.773') beside one-decimal values, and the Note column repeats the Value column ('-1.1' / '-1.1 pts'). Print intervals at the value's precision and drop the duplicate column."},
  {"claim": "c5", "severity": "should_fix",
   "finding": "Carried from round 2, renderer: both levers block notes say 'each interaction is how that effect changes across another setting', but neither block shows an interaction. The note should describe only the rows present."},
]
invariant_concerns = [
  "Invariant 1: all 34 distinct figure keys in report.json read through figure_keys_check, none refused. The prose numbers I checked by hand (0.31-point derived half-width, 0.6-point Nano-XS overall gap) are not printed, so they are not breaches; the overall-accuracy spread is still unreadable as a figure (kit gap).",
  "Invariant 2: every block renders. Renderer text defects remain: c4's note says 'bar' on a table and its counts print with '.0'; c5's notes describe interactions not shown; c1's row label reads 'Latency budget (p95) (p95 at most 250 ms)'; c2's compare intervals print at four significant figures.",
  "Invariant 3/4: the four caveats caveats_list holds all appear with material_if clauses; form high_if_immaterial matches open caveats.",
  "Invariant 5: no internal ids or hashes in prose; the meta.provenance path is not cited. Charts show 'Balanced accuracy', which Method now ties to 'overall accuracy'.",
  "Invariant 6: Method present; report.json has no explicit full-results appendix entry, assumed page-generated.",
]
total = sum(i["score"] for i in items)
doc = {"resolutions": resolutions, "items": items, "total": total,
       "passes": all(i["score"] > 0 for i in items) and total >= 11,
       "findings": findings, "invariant_concerns": invariant_concerns, "calls": 16}
json.dump(doc, open(out, "w"), indent=2, ensure_ascii=False)
print(total, doc["passes"])
