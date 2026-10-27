# Reasoning note: persona model tradeoff (r4)

**The eval.** The production persona chat model (Meridian-4.1, medium reasoning, 32k output cap) against three cheaper candidates: f64 (Meridian-4.1-flash, 64k cap), f16 (the same flash model, 16k cap) and cor (Corvid-3.8-flash, high reasoning). 300 cases per arm, 2,400 judged replies per arm, the same cases for every arm (the contrasts are paired). The asked aspect is staying in character (judged, calibrated on 240 human ratings, 82% agreement) with **no margin declared**. Two guardrails: music facts correct at least 90% (judged; judge unstated) and declines to speculate at least 95% (checked). Completing the requested action is measured but not gated. Cost per 1,000 replies, at 12M replies a month. No late criteria, no prior insights.

**What I noticed first.**
- Every candidate is cheaper and every candidate is clearly worse at staying in character (all three intervals exclude zero): f64 -6.1, f16 -7.4, cor -2.2.
- The guardrails cut two of the three: f16 misses music facts by 0.6, cor misses declining to speculate by 3. Only Production and f64 survive, and nothing dominates: f64 is 30% cheaper and 6.1 points worse.
- No margin and no exchange rate were declared. options_dominance says tradeoff; the margin sweep says tradeoff at every margin.

**The trap.** Two, pulling in opposite directions.
1. *Deciding anyway.* f64 survives the gates and is 30% cheaper, so "switch" reads like the answer; and Production is the incumbent with better character, so "keep" also reads like the answer. Neither is the analyst's call: nothing declared prices a point of character. The report has to price the trade and hand it to the owner.
2. *Treating the guardrail verdicts as settled.* f64 clears both guardrails on its point estimates only (computed 95% intervals cross both lines; a fresh run reads each below the line about one time in ten). And f16's 0.6-point miss is pure noise (a fresh run reverses it 37% of the time; settling it would take about 7,000 cases). The gate sorts f64 in and f16 out on a difference the data can't resolve (f16 vs f64 on music facts: -2.7, interval -7.3 to +1.9). Read even-handedly, f64's survival is as fragile as f16's exclusion. I kept the declared rule (f16 out) but put its fragility in Watch and f64's in the confidence and its own claim.

**Story.** Tradeoff. "Switching to f64 cuts reply cost by 30%, about $7,560 a month, but it stays in character 6.1 points less often, with the loss deepest in long conversations." Confidence high_if_immaterial: the trade's two sides are measured cost and a calibrated, clearly separated judged loss; the open caveats (f64's point-estimate clearances on both guardrails, the unstated music-facts judge, best-of-three) all bear on whether f64 is an option at all, not on the size of the trade. Each material_if says that. A lean, labelled ours: keep Production until a re-run confirms f64's guardrails.

**Lead evidence.** Cost bars beside the persona forest: the two sides of the trade, side by side, every arm shown with f64 emphasized so the reader sees all three candidates regress. I wanted a cost-vs-character scatter with the frontier; the tool can't put a measure and a judged dimension on one scatter (gap logged). Then the priced step: $1,239 a month per point ($1,036 to $1,543).

**What I looked for and found.**
- *Where the loss lives.* By turn depth, f64's drop grows from -2.0 (turns 1–2, not separated) to -4.5, -11 and -17.5 (turns 11+). Turns 6+ are 30% of replies and carry 64% of the loss (36% + 28%; I cite the two keys, the sum is not a figure). That earned a claim and suggests the most useful option nobody ran: hand conversations to Production once they get deep. Not priced, named as a next step. The depth intervals treat replies as independent though replies in one conversation aren't, so it's a lead.
- *Scenario strata.* Banter (-9) and taking a correction (-11.5) are f64's biggest scenario drops, but only a change is held, with no n or interval. Omitted from the body, named in the examples next step.
- *The output cap.* f64 vs f16 differ only in output cap; the lever is not separated on character (-1.3, -7 to +4.4). Doesn't move the trade.
- *cor.* The smallest character loss (-2.2) at 23% lower cost, out on speculation (92% against 95%; one-sided its miss is settled at today's size). A speculation guard could make it a middle option: next step.
- *Latency.* f64 is faster (p95 1.6 s vs 2.4 s) with no separated change in completing the requested action. No budget was declared; detail claim, since it's a real benefit on f64's side of the trade.
- *Confound.* f64 differs from Production in model and output cap, and Production's reasoning setting has no stated counterpart on the flash arms. Doesn't matter for the choice (the arm is what ships), matters for "why": in Method.
- No failure examples were frozen, so nothing shows what the late-conversation loss looks like.

**What I left out, and why.** Scenario breakdown (no counts or intervals); f16's what-if price (gated; pricing it invites choosing a removed option); the output-cap lever (not separated); task completion for f16 and cor (not options, not gated); Production's own decline with depth (describes the task, not the choice); latency of f16 and cor; trials-per-case method figures (Method). All in `omitted`.

**What I'd watch.** f64's nearest line, declining to speculate (cleared by 1.4). Whether listeners notice the character drop: nothing here measures it. f16's 0.6 miss, which a re-run could reverse, would put a cheaper option on the frontier. Best-of-three: f64 is the one candidate that cleared both lines, so a re-run tends to read it lower.

**Next steps, each with what it settles.**
1. The product owner declares a price per point of character or a margin of acceptable loss: settles the call immediately, no run needed.
2. Re-run f64 on both guardrails at 1,295 and 1,390 cases (two-sided, four in five): settles whether f64 is an option at all; if either falls below, keep Production.
3. Run a mix that hands deep conversations (6–10 band onward) to Production: settles whether most of the loss can be avoided while keeping most of the saving.
4. Add a speculation guard to cor and re-run: settles whether cor joins as a middle option.
5. Keep failure examples from f64 at depth 11+ and from banter and corrections: says what a fix must target.

**What doesn't transfer.** The trade could only be priced because cost is measured exactly and the asked aspect had a paired, calibrated reading. The general lesson is the even-handed gate reading: when the gate sorts candidates by less than their noise, the report should say that the survivor's survival is as unsettled as the loser's exclusion.

## Tools

- **Calls.** 36 harness tool calls in all (reading the brief and modules, writing the four files, and 28 Bash calls into the tool), carrying 56 `evals_analyst` invocations: primer and help ×4, campaign_overview, readings_list, contrasts_list, caveats_list, measure_read ×7, options_dominance, price_tradeoff ×2, settle_cases ×3, levers_effects, decision_sensitivity, arms_compare ×2, examples_list, measure_breakdown, block_catalog, block_render ×15 (3 refused), figure_keys_check ×4, capability_gap ×6.
- **Wished existed (each logged as a capability_gap).** A scatter of cost against a judged dimension; failure examples (none frozen); case-clustered intervals for within-conversation strata; a margin sweep that shows where an undeclared margin would turn the trade into a decision; counts and intervals for the scenario strata; figure_keys_check over a whole draft rather than a hand-copied key list.
- **Refusals.** `scatter_frontier` over `measures` refused `persona`, and over `judged` refused `cost_per_1k` (no source holds both). `diverging_bars` over `compare` refused mixing the paired overall row with unpaired depth rows; restricting `strata` to the depth bands fixed it. All recovery lines were clear.
- **Confusing in help or results.**
  - `decision_sensitivity` with `margin:persona` on an aspect with no declared margin returned one "tradeoff" segment over 0–15 and `holds: True`. I expected f64 to become acceptable (and dominate on cost) once the margin passes 7.3. Either the sweep ignores undeclared margins or non-inferiority doesn't feed dominance; the result doesn't say which.
  - `flip_margin.f64` is refused ("none in this input") even though the contrast interval gives one; it seems to exist only for readings, and this snapshot has no readings (readings_list is empty: the asked aspect is read through contrasts only).
  - The contrast basis in `arms_compare` says "corrected within its family" while `family.correction = none` and the disclosure says not corrected. I wrote "not corrected" in Method.
  - Arm identity differs across blocks: `bars` over `measures` labels rows "Production", "f64" with no letter tag; `forest`, `dot_interval` and `range_bars` print "P · Production", "A · f64".
  - `lines` over `strata` dropped the half-widths the strata hold (level blank), so I used the `compare` source for intervals.
  - The depth strata's `n` are trials (they sum to 2,400 per arm), and the computed intervals read them as independent; nothing in the output flags that these are turns within conversations.
  - The settle table prints sizes as `447.0` (floats) where the key with `|int` prints `1,295`.
- **Paging.** The first figure_keys_check of 38 keys showed 20 entries; I had filtered its output, so a `next:` line may have been hidden. I re-checked the other 18 in a second call. Every other result fit one page.

## Revision 1

The editor scored round 1 at 10/14, a fail, with 2 must_fix and 6 should_fix. Round 1 is kept in `round1/`.

**Must fix.**
- *Margin bands.* Explore's margin row now gives all three bands off f64's interval: a margin wider than the far end makes f64 acceptable, so it is the pick; one narrower than the near end shows f64 worse than allowed, so keep Production; only a margin between the two ends leaves the call open. My round-1 wording ("any narrower margin leaves the trade open") was wrong and contradicted what_settles_it and the first next step.
- *Headline.* The depth clause is gone. The headline is now the trade plus "nothing declared says which matters more". Depth appears only in its own claim, which now opens "A lead, not a finding".

**Should fix.**
- *Lean.* It now rests on the measured guardrail risk: a re-run reads f64 on the wrong side 8.9% of the time on music facts and 9.7% on declining to speculate. The unmeasured "what a persona is for" is gone. The editor is right that most replies sit at depths 1–5, where the loss is small.
- *Settle evidence (now c4).* I dropped the table, because block_render can't choose its columns, format its counts or add the four-in-five size. In its place are bars of how often a re-run reads the wrong side, which is exactly what the claim says. The next step now cites both re-run sizes through keys: the half-of-re-runs size and the four-in-five size, so it no longer disagrees with a printed column.
- *Depth precision (now c2).* I switched from the compare source, which prints intervals at four significant figures, to the strata source with arm_minus_control. That prints one decimal and no intervals. I accept losing the intervals: per-reply intervals overstate how settled a band is, and the caption and Method now say why none are drawn. The block also adds cor's depth bands, which back the cor next step.
- *Confound.* A new next step runs Meridian-4.1-flash at Production's settings (medium reasoning, 32k cap), which separates the model from its settings.
- *Price block.* I dropped the one-row price chart and its claim. The price, its band and both flip points are in the trade, where it flips, and Explore; the cut is recorded in omitted.
- *cor next step.* It now cites cor's depth-11+ loss (4.5 points) against f64's (17.5).
- *Invariant concerns.* "three candidates" is now `{n_candidates|words}` everywhere (c1 caption, c3 text, the best_of_n material_if, Method). The output caps in omitted are now settings keys.

**Disagreements.** None on substance. One trade-off is mine: the depth block has no intervals now. I preferred an honest descriptive lead to intervals I know read too narrow. The c4 bars have two rows and a Note column that repeats the value, the same shape the editor disliked for the price chart. I kept them because they show the claim's own figure for both guardrails side by side.

**Tools this round.** 9 harness calls, 5 of them Bash calls into the tool, carrying 6 `evals_analyst` invocations: 2 block_render, 3 figure_keys_check (46 keys, paged at 28 and 39, and each `next:` echoes the whole key list), and 1 capability_gap. I extracted the keys from report.json by script instead of copying them by hand. New gap: block formatting, meaning the compare intervals at four significant figures, the settle table's float counts and empty columns, and no column choice. All 46 keys read.
