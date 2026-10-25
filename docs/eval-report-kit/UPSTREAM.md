# Upstream requirements: what the campaign bundle should carry

Building the Kairo round 2 reference from `kairo-round2-writer-input.json` worked, but only with an adapter that joins, guesses or works around the gaps below. Each item names the evidence in that bundle, what the page needs, and a proposed change. Items are independent and ordered by impact on the page.

### U1. Arm display names and settings in `bundle.arms`

**Seen:** `bundle.arms[].label` is `null` for all three arms. The names a reader knows ("September", "1017") exist only in the writer message's `arms.arms[].arm`, with `levels.model` set to the alias, not the served model. The adapter has to join alias → `disclosures.arm_served_models` → `levers.model.display` → `variant_key`.
**Need:** for each arm, a display name, a role, the served model, and the levers that differ from the control (`display` values). The page derives short names and the arms key from these (GENERATOR.md §6.1).
**Proposal:** populate `label`; add `differs_from_control: {lever: display}` per candidate arm, computed from `levers` content hashes.

### U2. One interval system per comparison, and consistent point estimates

**Seen:** the asked reading for qwen3.8-flash is −7.58 [−10.79, −3.38] at 95% (confirmatory), while the contrast on the same `passed` measure is −7.58 [−12.59, −2.57] at 99.69% (Holm, family of 16). Both are correct, but a page that shows both without labels contradicts itself. Separately, the cost contrast delta for qwen3.8-flash is +$0.000616 per reply while the difference of the cell means is +$0.000535. The populations differ (n = 828 vs 864 priced replies), but the bundle doesn't say so.
**Need:** each interval tagged with its level and family; a note when a contrast's point estimate isn't the difference of the reported means, and why.
**Proposal:** add `level` and `family_id` to every interval-bearing object; add `population` (n per side) to contrasts.

### U3. Don't trim the comparisons the page is built on

**Seen:** `trimmed` drops 8 contrasts (least separated first) to fit the writer budget. The page then lacks the `passed` contrast for 1017 and both `persona.booth_stays_offstage` contrasts, so the summary forest has holes. The page says so in a caption, which is correct but avoidable.
**Need:** every contrast on the asked aspect's measures and on every judged dimension, for every arm.
**Proposal:** trim excerpts and trials before any contrast. Contrasts are small (about 200 bytes each).

### U4. Per-stratum figures with paired deltas

**Seen:** `cell_measures[].strata[]` has per-stratum judged means and SEMs per arm (26 strata), but no per-stratum pass rate and no paired difference. The stratum chart on the reference (the off-lane finding, about half of each candidate's in-character loss) has to compute unpaired, uncorrected deltas itself.
**Need:** per stratum, per candidate: the pass-rate change and judged-dimension changes against the control, paired by case, with intervals and an exploratory flag.
**Proposal:** a `strata_contrasts` list alongside `contrasts`, marked exploratory and excluded from the confirmatory family.

### U5. One computed disclosure for late criteria and judge tiers

**Seen:** all four criteria (aspect, control, margin, floor) were set after the readings were visible, and every judge tier is `undetermined` (0 of 20 calibration ratings). The writer has to assemble that disclosure from `late_criteria` and `judge_evidence_tiers` itself.
**Need:** one disclosure the page states once, near the top: its reasons, the findings it weakens and the findings that survive (here, the cost regression is measured, not judged). It does not cap the recommendation's confidence (owner ruling, 2026-10-25).
**Proposal:** `disclosure: {reasons[], weakens[], survives[]}`, computed in the bundle.

### U6. Margin diagnostics

**Seen:** the 15-point margin was set after the readings; the worst plausible drops are 10.8 and 5.8 points, so the margin couldn't fail.
**Need:** per arm, the margin at which the reading flips (`-lo`), and a flag when the declared margin is uninformative.
**Proposal:** add `flip_margin` to each reading, and `margin_informative: bool` with its rule to the aspect.

### U7. Volume for cost at scale

**Seen:** cost is per reply only, so the page can't say what a switch costs per month.
**Need:** a host-declared volume (replies or calls per month) per subject.
**Proposal:** an optional `volume_per_month` on the campaign declaration. The page shows monthly figures only when it is present.

### U8. A way to declare how to weigh axes

**Seen:** `merit_axes` and `merit_priority` are empty. When a candidate is cheaper but worse, the page has to enter tradeoff mode with no declared exchange rate (references/r4).
**Need:** optional declarations, either a priority order or an exchange rate (dollars per point of an aspect), plus standing guardrails from policy.
**Proposal:** extend `merit_priority` with `exchange_rates: [{axis, per, value}]` and `guardrails: [{measure, floor, source: policy}]`.

### U9. Human anchors for judged dimensions

**Seen:** the calibration state is `insufficient` (0 of 20 results) for every dimension.
**Need:** when even a small human preference or calibration set exists, the page shows it next to judged deltas.
**Proposal:** carry calibration results (agreement, n, by stratum if available) in `judge_evidence_tiers`.

### U10. Declared display names for strata and measures

**Seen:** stratum ids (`off_lane`, `assistant_shape`) and measure keys (`production_replicating_cost`, `goal_state:call_count(...)`) reach the writer as-is.
**Need:** reader names ("Off-lane asks", "Cost per 1,000 replies", "Sent a message or voice line").
**Proposal:** extend #956 chunk 01's declared short names to strata and measures.

### U11. Structured judge verdicts on trials

**Seen:** `trials[].judge` arrives sometimes as a list of dicts and sometimes as a truncated repr string, so pulling one dimension's reason needs a regex.
**Need:** `judge: [{dim, score, scale, reason}]` always, with `reason` truncated only at its own end.

### U12. Floors that speak for a different measure than their reading

**Seen:** in `floors`, qwen3.8-flash's floor `speaks_for` "the pass rate" and matches its reading. 1017's floor on the same aspect `speaks_for: persona.holds_character`, with interval [−7.11, −1.70] (worse), while its asked reading is on the pass rate, [−5.8, +1.1] (not separated). Two arms on one aspect are judged on different measures.
**Need:** every floor on an aspect speaks for the same measure as the aspect's reading, or the bundle says why it doesn't.
**Proposal:** a validation in bundle assembly that refuses mixed `speaks_for` within an aspect, or adds an explicit `speaks_for_reason`.

### U13. What "pass" means, as data

**Seen:** the scoreboard splits failures into `judged`, `check` and `model_failed`, but nothing lists which judged dimensions a trial must pass. September's pass rate (85.3%) is below its in-character rate (91.5%), so other dimensions count, but the page can't say which.
**Need:** the pass rule per aspect: which checks and which judged dimensions, at what threshold.
**Proposal:** `aspects[].pass_rule: {checks: [...], judged: [{dim, at_least}]}`.

### U14. Short runs mapped to arms

**Seen:** `disclosures.short_runs` is keyed by run id only. The adapter maps run ids to arms through `design.contrasts[].run_ids`.
**Proposal:** include `arm` on each short-run entry.

### U15. Prior insights checked against the new data

**Seen:** a prior insight says qwen3.8-flash "significantly lowers production costs" (confidence high). In this campaign it costs 28% more per reply. Nothing marks the conflict.
**Need:** each prior insight tagged as confirmed, contradicted or untested by this campaign, with the figure that decides it.
**Proposal:** `prior_insights[].status` plus `evidence` (a contrast reference).
