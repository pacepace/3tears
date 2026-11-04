# Blind read of the acceptance pairs

These are read-only verdicts, made without opening KEY.json. Reasons, page stats and slips for each case are in `BLIND-READ.json`.

Each page cell gives words / tables / decision on the first screen / statistics read correctly.

| Case | Better | Confidence | Page A | Page B | The deciding difference |
|---|---|---|---|---|---|
| booking | A | lean | 1216 / 6 / yes / mostly | 1163 / 5 / yes / partly | A names the busy-slot risk and gives intervals per check. B wrongly suspects the interval ignores clustering, and it points to an appendix list that does not exist. |
| invoices | B | sure | 1403 / 6 / no / mostly | 2165 / 9 / yes / mostly | B makes a call (keep v1, do not switch yet) and surfaces v2's new failure kind. A states no decision and leaves four must-fix findings open. |
| kairo | B | coin flip | 3415 / 7 / yes / partly | 3267 / 7 / yes / partly | B's lede answers the non-inferiority question ("inside the 15-point margin, yet September is cheaper"). A names the wrong arm as having the thinnest margin. B in turn misstates 1017's character result. |
| concierge-model-selection-1 | B | coin flip | 2202 / 8 / yes / mostly | 1952 / 7 / yes / mostly | B says when each caveat matters. A's Open-70B dominance claim sits on a cost-only chart. |
| concierge-model-selection-2 | B | coin flip | 1748 / 7 / yes / mostly | 2596 / 8 / yes / mostly | B reads the price bands correctly and gives a routing lead. Its lead claim "quality rises with cost in every task" is false by its own table. |
| query-planner-config-sweep-1 | A | lean | 4749 / 9 / yes / mostly | 5540 / 10 / yes / partly | A commits and lays out both ways the owner could bend the gates. B treats per-task latency as open, though an overall p95 above 600 ms already implies a per-task miss, and it labels unknown latency "Fails". |
| query-planner-config-sweep-2 | B | sure | 3776 / 8 / yes / no | 3979 / 10 / yes / mostly | A claims High confidence on a 0.1-point clearance whose interval spans the gate. B picks the arm whose interval clears the gate and keeps confidence Low. |
| query-planner-config-sweep-3 | B | sure | 6625 / 10 / yes / mostly | 3917 / 10 / yes / yes | B gives the pick in monthly dollars and shows the one pairwise gap that matters, with its interval. A buries that gap in an 80-row effects table and claims latency "does not separate" options that run from 140 to 2551 ms. |
| turn-router-classifier-sweep-1 | A | lean | 2962 / 10 / yes / mostly | 2505 / 7 / yes / partly | A's heatmaps show the cell floors behind its "only one passes" claim, and it names fallbacks. B's block cannot show that claim. Both pages overstate how fragile a 0.7-point clearance is at 48,210 turns. |
| support-replies | B | lean | 874 / 6 / yes / mostly | 901 / 6 / yes / yes | B says the interval allows a materially lower rate and covers sampling noise only. A says "a few percent" for a bound that permits about 16% failure. |
| support-triage | A | lean | 1390 / 5 / yes / mostly | 1079 / 5 / yes / no | A's failure accounting holds: some failures must be label-only. B calls the urgent-versus-label gap "within noise" because two paired measures' marginal intervals overlap. |

Tally: B is better in 7 cases and A in 4. Three of B's wins are coin flips. There are no ties.

## Overall observations

Length did not decide the outcome: the better page was shorter in five cases and longer in six. The one large gap, sweep-3 (6.6k against 3.9k words), favoured the shorter, more focused page. The better page differed in four ways.

- **A call and the risk behind it on the first screen.** It says what to do, or what not to do yet, and names the main risk on that screen. Hiding the decision (invoices) or committing weakly (sweep-1) lost.
- **Blocks that show exactly what the claim says.** Winners used a heatmap for a cell-floor claim and the single relevant pairwise row for a "not told apart" claim. Losing pages more often had a cost chart under a quality claim, a dump of every interaction, or a table missing the gate that does the eliminating.
- **Correct reading of intervals.** The decisive errors were statistical:
  - treating overlapping marginal intervals as a test, even for paired measures;
  - calling an untested gap "unseparated";
  - confusing non-inferiority with being "worse";
  - missing that an overall p95 above budget implies a per-task miss;
  - High confidence on a clearance whose interval spans the gate;
  - understating how far an interval reaches.
- **Confidence that matches the caveats.** Pages that kept confidence consistent with their own caveats read better. A page claiming "High" beside three or more open caveats was the most common defect on both sides.

Several weaknesses showed up on both sides of nearly every pair, so they did not separate the pages:

- caveat lines phrased as asserted facts ("the judge is lenient");
- unexplained sample-size figures;
- duplicate pass-rate columns;
- the unnamed-provider cost caveat;
- treating thin point clearances as fragile without reasoning from n.
