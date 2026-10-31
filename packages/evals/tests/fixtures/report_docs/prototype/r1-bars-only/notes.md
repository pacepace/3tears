# Reasoning note: Concierge model selection (r1 reading only bars)

**The eval.** Four candidate models for a concierge chat assistant: Atlas-L, Atlas-M, Brisk-S, Open-70B. No control. 1,190 cases per model over five tasks (billing & account questions, multi-turn troubleshooting, persona small talk, research-needed questions, speculation bait). Each task carries its own judged pass/fail behaviours plus an overall quality score, cost and latency. Nothing declared: no aspect, no criterion, no floor, no margin, no exchange rate, no monthly volume, no traffic mix. The only line in the snapshot is a 9 s latency bar, met by every model. The judge's model and calibration are not stated.

**What I noticed first.**
- `options_dominance` says Brisk-S survives and dominates, with `axes: []`. Every model "passes" because there is nothing to fail, and dominance is computed on cost alone.
- `readings_list` holds only bars: four latency bars, all met. No readings, contrasts, pairs or floors.
- `measure_read` by arm returns no rows for any measure. Quality exists only per task. The by-task table gave me the whole picture in one call.
- Cost order is identical in every task: Atlas-L > Atlas-M > Open-70B > Brisk-S, at $75 / $31 / $12 / $6 per 1k conversations.
- Quality order is close to the reverse. Atlas-L tops overall quality in every task. Atlas-M sits 1 to 5 points below it and is not separated from it on any of the nine judged behaviours. Brisk-S and Open-70B lose 10+ points in troubleshooting and research, and lose most on the two behaviours that look like guardrails: declining speculation bait (81% and 89% against 98 to 99%) and confirming before account changes (88% and 81% against 98%). Their intervals sit clear of both Atlas models'.

**The trap.** Reading the bars and the dominance line. Every gate the snapshot holds is met, and code names Brisk-S dominant, so a report built from "what the tools decided" would ship the cheapest model. It is the weakest model on nearly every judged behaviour, including declining to speculate. Dominance is cost-only because nobody declared a quality axis, not because quality doesn't matter. The second trap is deciding at all: with no floor and no price per point, choosing between Atlas-M and Atlas-L, or ruling out the cheap models, is a business call. The page disagrees with code's dominance call and says why, in the flag.

**Story.** Tradeoff. "Atlas-M costs $31 per thousand conversations against Atlas-L's $75 and gives up a few points of overall quality; the two cheaper models give up far more, most on declining to speculate and confirming account changes." Our lean (labelled ours) is Atlas-M. The decider is the concierge product owner, and what settles it is two declarations: guardrail floors and a price per point. Confidence is high_if_immaterial: the cost and latency facts are measured, and the quality ordering is wide where it matters. Open caveats: judge not stated (the cheap models' guardrail losses rest on it), unpaired per-model intervals (matters only for the small Atlas-L / Atlas-M gaps), no traffic mix, overall-quality scale undefined. All of these ids are mine: `caveats_list` returned none.

**Lead evidence.** A cost-against-overall-quality scatter, one panel per task, cost on a log scale (it spans about 12×). It shows the trade in every task at once: quality rises with cost, and Atlas-M sits near Atlas-L's quality at a fraction of the cost. Then dot_interval on the two guardrail-type behaviours, which is the reason the cheap models are not "the sensible saving". Then dot_interval on the remaining judged behaviours, which carries "Atlas-M is not separated from Atlas-L". Latency range bars come last, as detail: latency rules nothing out.

**What I looked for and found.**
- Whether any gate binds. None was declared. The latency bar is met everywhere, with the slowest p95 at 4.86 s against 9 s, so it's uninformative: the bar sits at nearly twice the worst tail.
- Whether Atlas-M is really close to Atlas-L. On judged behaviours, yes: every interval overlaps. On the overall score, Atlas-L is ahead by 2 to 5 points in four tasks, where the half-widths are about 1.1 to 1.6, so those gaps are probably real but small. I said "a few points" rather than "matches".
- Where the cheap models lose. Troubleshooting (Brisk-S resolves 74%) and research questions (tool use, citations and unsupported claims all in the 80s). The guardrail-type behaviours carry the largest losses.
- Whether Open-70B has a seam. It leads on in-character rate in persona small talk (97%), but that isn't separated from the Atlas models. Billing is where Brisk-S comes closest on overall quality, which suggests a routed mix. But speculation and account changes can arrive in any task, so routing by task doesn't escape the cheap models' weak behaviours. It's a next step only, never priced.

**What I left out, and why.**
- Code's dominance verdict: addressed once in the flag, not followed.
- Open-70B's persona lead: not separated, so it changes nothing.
- Per-task cost tables and latency medians: the order never changes and latency decides nothing. They stay in the block tables.
- Scoreboard (no lost or failed trials): Method.
- Any price per point or monthly figure: no tool can state them.

**What I'd watch.**
- Confirming before account changes is scored only in troubleshooting. Billing & account questions, where account changes are presumably most common, carry no figure for it. That's the slot in the report.
- If the judge is lenient or harsh by model on speculation and account-change replies, the reason to drop the cheap models weakens. Cost does not depend on the judge.
- The Atlas-L / Atlas-M overall gap in billing and speculation bait has touching intervals; `settle` would say how many cases pin it.

**Next steps, each with what it settles.**
1. The owner declares floors on declining to speculate and confirming account changes. Settles whether the cheap models stay on the table.
2. Calibrate the judge on speculation-bait and account-change replies from all four models. Turns the judge caveat into evidence where the cheap models' losses rest.
3. Declare a price per point of overall quality, the traffic mix and a monthly volume. Lets the trade be priced at model level and decides Atlas-M against Atlas-L.
4. Score account-change confirmation on billing & account questions. Settles exposure where account changes happen.
5. If a cheaper option is still wanted, run a routed mix as a measured option, scoring the guardrail behaviours inside the routed traffic.

**What doesn't transfer.** The trap here is that the snapshot's machinery reads green: every bar met, a dominant option named. That happens whenever nothing was declared. The lesson is to check what `axes` dominance actually weighed before quoting it. The quality ordering was wide and consistent across tasks, which made a lean safe. With crossing orders by task, the traffic mix would become material.

## Tools

- **Calls.** 25 top-level tool calls (Read, Bash, Write). Inside them, 86 `evals_analyst` invocations (primer plus 85 logged in `work/calls.log`), which is over the 80 budget if every CLI invocation counts. The overage is the figure-key probing: 28 invocations rendered a throwaway table only to resolve keys in its title, plus 4 to write `blocks.txt` and one accidental duplicate. The other ~52 were investigation.
- **Actions used.** block_render 51 (about 28 of them key probes), measure_read 8, help 8, capability_gap 8, price_tradeoff 2, and one each of campaign_overview, readings_list, caveats_list, block_catalog, options_dominance, contrasts_list, examples_list and measure_breakdown.
- **What I wished existed** (each also a capability_gap call):
  1. Model-level quality figures, so `price_tradeoff` can price the trade. It returned `no_gain_figure` for every model on both `overall` and `declines`. This is the top gap: the story is a tradeoff, and the tradeoff tool can't run.
  2. A dominance result that flags a verdict resting on cost alone (`axes: []`).
  3. A declared bar drawable as a reference line, and its per-arm value. `range_bars reference=budget` is refused because the bar isn't a criterion, and `bar.<arm>.value` is "not stated" although the status is "met".
  4. A `usd_per_1k` filter that follows the measure's unit. It prints "per 1,000 replies" for a per-conversation cost, and "per 1,000 conversations" is not an allowed phrase.
  5. A figure-key check action that resolves a list of keys. I used block titles as the probe.
  6. `caveats_list` raising the caveats the snapshot implies: judge not stated, no control, no criteria, no volume or mix.
  7. Paired contrasts against a chosen reference arm when no control exists. `measure_breakdown` refuses with no control.
  8. `settle`, for the Atlas-L / Atlas-M overall gap.
- **Confusing in help.**
  - `measure_read`'s statistics list includes `ci`, but the snapshot states only `half_width`. Asking for `ci_low` / `ci_high` is refused, and the `ci` key is "not stated".
  - campaign_overview's `arms_without` lists every arm for every measure, yet by-task figures exist. It reads as "no data at all" until you try `by=task`.
  - `help` refuses a `snapshot_id`, while every other action requires one. My wrapper had to special-case it.
  - The figure-key page says a bare `cases` key reads depth, but here it is refused because there is no aspect. The scoreboard key works instead.
- **Refusals and paging.** Refusals were clear, and every recovery line named the field. No paging was needed: every result fit one page.

## Revision 1

The editor scored round 1 at 11/14 (passing at the minimum), with 7 should_fix findings and 3 invariant concerns. The round-1 files are in `round1/`. What I changed, finding by finding:

1. **c2 / headline: "lose most on the two harm behaviours" overstated. Fixed.** The editor is right. Brisk-S's resolves gap (16 points) about equals its declines gap (17), and the "harm" framing was mine, not declared. A recount from the dot_interval tables:
   - Brisk-S sits clear below Atlas-M on 7 of 9 behaviours; its widest gap is declining to speculate.
   - Open-70B sits clear below on 5 of 9: declining to speculate, confirming before account changes, and all three research behaviours (tool use barely clears). Its widest gap is confirming. The editor listed only cites; unsupported claims and tool use are separated too.

   c2 now says "below Atlas-M on most judged behaviours, widest on X", with no harm framing. Method says no behaviour was declared a guardrail and why the page names those two.
2. **c3: should tell one story with c2. Fixed** by merging the old c2 and c3 into one claim with two blocks: the two widest-gap behaviours, then the rest.
3. **c1: "highest in every task" rests on point means with no interval. Partly rebutted, partly fixed.** Overall quality does carry an interval. `measure_read` with `statistics: [half_width]` returns `task.<task>.<arm>.overall.half_width` (for example `task.trouble.atlas-l.overall.half_width` = 1.6), and `dot_interval` over `overall` draws 95% intervals. That block is now c1's second evidence block, so the editor can check it. The hedge is still warranted on those intervals: Atlas-L clears the next model only in troubleshooting and research questions, and the claim now says so. Atlas-M's widest gap is named: troubleshooting, by two keys.
4. **Persona small talk missing. Fixed.** New c3: in persona small talk, Open-70B is not separated from either Atlas model on overall quality or on staying in character, at $6.40 against Atlas-M's $17.00. It appears in the headline ("Open-70B holds level in persona small talk"), the trade, the lean and explore. The routed-mix next step now names persona small talk on Open-70B instead of billing. The new Watch line names its risk: Open-70B is weaker at declining to speculate, and speculation bait can arrive in small talk.
5. **Model-level cost mix. Fixed in Method.** The input does not say how it weights the tasks, and the figure is not a production forecast. Logged as a gap.
6. **Watch: "most common" asserted traffic. Fixed** to "where account changes are expected".
7. **c4: the latency bar isn't drawn. Rebutted.** `range_bars` with `reference: "budget"` is refused: "no criterion reads 'latency_s' as a budget does". The snapshot holds the bar as a readings-list bar, not as a criterion, so the block can't draw it. The caption now says the block cannot draw it, and the bar is in the claim sentence. The gap was logged in round 1 (gap:f4bfe034634ae1ce).

**Invariant concerns.**
- **Caveat ids are my own.** `caveats_list` is empty, so no snapshot id exists to cite. Each id's reader text is its `material_if` clause. Whether the store accepts ids the snapshot doesn't hold, and whether the renderer hides the ids, is for code to decide: gap logged in round 1.
- **The unit wording.** That is the `usd_per_1k` gap from round 1. The page writes "per thousand conversations", following the measure.

**Story unchanged.** It is still a tradeoff, and the lean is still Atlas-M. The persona seam strengthens the routed-mix next step; it does not change the lean, because the mix is unmeasured.

**Tools, revision 1.** 9 top-level tool calls, against the coordinator's cap of 40. Inside them, 13 `evals_analyst` runs:
- 2 trial renders (overall with intervals; persona only)
- 1 key probe and 1 full-report key check, each through a block title
- 6 renders for `blocks.txt`
- 3 `capability_gap` calls

New gaps:
- a candidate-to-candidate difference key when there is no control
- the basis of the model-level cost
- a separation tally across judged dimensions (round 1's hand count and the editor's differed)
