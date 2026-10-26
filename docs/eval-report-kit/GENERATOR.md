# Eval report generator: procedure and rules

This is the complete spec for turning one campaign's results into a decision memo like the references in `references/`. It assumes no other context. Read it top to bottom once; after that, sections 3–6 are the working reference.

The generator has three stages. Each one has a typed output that is stored and testable:

```
campaign bundle ──adapter──▶ report input ──planner──▶ report plan ──renderer──▶ HTML page (+ Markdown)
  (pipeline)                 schema/report-input       schema/report-plan        kit/ + references/
```

- **Adapter** (deterministic code). Copies statistics through; never recomputes one the pipeline produced. `adapters/bundle_to_report.py` is a working example for the current bundle format.
- **Planner** (LLM, validated). Reads the report input and writes a report plan: the shape, the mode, the sections, the charts, the explore controls, and every sentence of prose. Prose refers to figures by key, never by typed numbers (§6.2).
- **Renderer** (deterministic code). Draws the plan with the kit. It owns layout, typography, chart geometry, color, responsiveness and accessibility. The planner never writes markup.

The planner is the part that makes the structure fit the data. A fixed template per eval kind is what overfits; a planner choosing from a fixed vocabulary does not.

---

## 1. Classify the eval (code, before the planner)

Code reads these signals from the report input, applies the shape lookup below, and writes `plan.signals` and `plan.shape` before the planner runs; the planner receives them as facts and never writes them. The shape's spine is a fact too: code looks it up below and hands it to the planner, and the plan's first evidence section draws it. The same holds for the masthead, arm colours, and each chart's interval level, unit and axis range (owner ruling, 2026-10-25: code computes facts; the planner chooses the mode, the sections, the charts and every sentence).

| Signal | Read from | Values |
|---|---|---|
| Candidate structure | `arms[].role`, `meta.design_shape` | unordered set · ordered versions · factorial grid · one control plus candidates |
| Baseline | an arm with `role: control` | none · control |
| Score type | `judged[].scale`, `measures` | graded (ordinal judge) · pass/fail · class label · mixed (pass/fail dimensions plus ordinal grades) |
| Strata | `strata`, or named tasks | none · 2–6 tasks · many strata (show the largest 12) · a target matrix |
| Criteria | `aspects[].margin`, `floors`, `bars`, `guardrails` | hard gates · margin (non-inferiority) · none declared |
| Cross-cutting aspects | aspects that occur inside every task (safety, speculation) | list |
| Spread and n | cost range across arms; n per cell | log axis when > 10×; borderline flags |

Map them to a shape with this lookup. It is a lookup, not a judgment call. When signals conflict, the criteria signal wins.

| Shape | Signals | Spine (first evidence chart) | Reference |
|---|---|---|---|
| A. Candidate selection under gates | unordered, tasks, hard gates | Gate matrix: task × candidate, pass / borderline / fail with the binding reason, cheapest passing cell marked | `r1-model-selection` |
| B. Factorial config sweep | factorial grid, pass/fail, efficiency | Gate-space scatter with one path per factor level, then lever-effect bars | `r2-config-sweep` |
| C. Classifier over a target matrix | two factors, class labels, cell matrix | Config lattice (factor × factor tiles), then per-cell heatmaps | `r3-classifier` |
| D. Control vs candidates | a control, paired cases, a margin or no rule | Forest plot of change from control with intervals and the margin band | `r5-kairo-round2` (decided), `r4-tradeoff` (tradeoff mode) |
| E. Version lineage | ordered versions, same cases | Step chart per task across versions, then cases that newly fail | none yet: build from the rules |

A prompt eval can be A, D or E. A model eval can be A, B or D. The shape follows the structure of the comparison, not what varied.

**The spine is code's fact** (owner ruling, 2026-10-25: code computes facts). Code reads the spine chart's kind off this table, and the parts of its query the input settles (`report_plan_checks.spine`): A, a gate matrix over the tasks; B, a gate-space scatter over the configs, its axes the two measures the gates read; C, a config lattice over the configs, its metric the measure of the first gate read over the whole comparison; D, a forest of the asked aspect's readings, with the margin line where it declares a margin; on a measured question, where no surviving candidate is separated by an asked reading and at least one is separated by a corrected contrast on an asked aspect (the separation §1.1's no-call reads, over the same survivors), a forest of those contrasts, since that is the evidence the call rests on: the first deciding contrast's aspect and the deciding measures there in its unit and interval level, so the chart keeps one of each; else a forest of the contrasts, its measures the planner's. A forest of contrasts draws no margin line: the margin is on the readings' scale; E, lines over the tasks (else the strata). It hands the spine to the planner with the other facts, and the plan's first evidence section copies the kind, every query field code fixed and the reference line. A plan whose first evidence section is not the spine is refused. What the input does not settle is the planner's: a source the input holds none of the above for, the measures where no rule names them, and on every spine the arms and their order, the strata and the emphasis. Some of what a spine draws is no query field at all (a gate matrix's binding reasons and cheapest passing cell, a gate scatter's gate region and paths per factor level, a lattice's row and column factors); the renderer reads those off the criteria and the configs.

### 1.1 Mode: decide or trade off (code computes the allowed modes)

Code computes the surviving options and whether one dominates, and passes the planner the modes it allows (`report_decision.allowed_modes`); the planner picks within them, and a plan in any other mode is refused (review, owner-endorsed, 2026-10-25). The same holds for the arithmetic of tradeoff mode (§4: price per point, flip bands, hybrids).

- **Survivors.** An option survives when it passes every gate (each `gate` or `guardrail` criterion, read on its point estimate where its scope says: overall, in its task, in every task or in every cell) and no guardrail the pipeline read on it is breached. A gate with no figure for the option is not passed. Borderline counts as passing (§3). Floors and bars warn and remove nothing; the margin is the asked aspect's own axis.
- **Dominance.** A survivor dominates when it is at least as good as every other survivor on every axis that matters. *Gates*: every survivor passes them, so they tie. *Cost*: cost per 1,000 replies (else the one dollar measure where lower is better, else that measure blended over tasks by traffic share), lower is better; two arms with no cost tie, and an arm with no cost is not comparable with one that has a cost. *The asked aspect*: in a change, a candidate shown better (a confirmatory `better` reading, or an `improved` contrast on an asked aspect) ranks above the control, one shown no worse (every floor on every asked aspect held, and within the margin on each asked reading that declares one) ties it, any other ranks below it. In a choice, an arm is at least as good as another unless a pair reading shows the other ahead.
- **Decide** when exactly one survivor dominates; the plan recommends it. Kairo round 2 is this: the control is cheaper and no candidate is shown better, so it dominates. Also decide when several survive, none dominates alone, and the campaign declared a rule that ranks the axes (`merit_priority`): the rule picks.
- **Trade off** when two or more survive, none dominates, and the campaign declared no rule that ranks them (`merit_priority` empty, no exchange rate). §4 covers this mode.
- **No call** when the report measures one arm, when no option survives, or, in a change, when every surviving candidate's asked readings and its asked aspects' corrected contrasts can't separate it (each reading's interval crosses zero and it is outside the margin, or no margin is declared, and no contrast on an asked aspect is `improved`, `regressed` or `equivalent`). The headline says what is missing and what would settle it.

---

## 2. The page contract (every shape)

Top to bottom. The renderer enforces the order; the planner fills the slots and picks the middle.

1. **Masthead.** One line: kind · system under test · counts (arms, aspects or tasks, cases × repeats) · run date · `Synthetic example data` when it is. Never a paragraph.
2. **Headline: the decision as an instruction with its reason.** "Keep September. Both candidates cost 27–28% more per reply and stay in character less often." Never the eval question restated. In tradeoff mode it states the trade (§4).
3. **Lede.** At most two sentences: the strongest supporting fact, and what would change the answer.
4. **Flag** (only when needed). One amber line for the report-wide disclosures the decision's confidence does not carry: "No rule set in advance" in tradeoff mode (§4), and, only under a "Low" confidence, a report-wide caveat the plan chose to disclose here instead of with a clause. It says which findings it weakens and which survive. Report-wide caveats are criteria set after the results, an uncalibrated judge, and any confound or short run that touches the whole comparison. A confound that touches one arm or one finding (an arm's reasoning budget, one short run) is said once beside the claim it qualifies, usually in Watch, and never in the flag: a caveat is stated once, at the level it applies to (owner ruling, 2026-10-24).
5. **Decision block.** Four slots and a confidence. Decide mode: Recommendation (with its numbers) · Why not the others · Watch (the weakest margin, a confound, what wasn't measured) · Next (an experiment whose result could change the call, and which way). Which margin is weakest is code's fact (`report_margins.weakest_margin`): the figure nearest the line it is read against, as a share of that line, among a surviving option's gate, floor and bar figures, readings and floors, and a rejected option's one missed gate figure when that alone keeps it out. Watch cites it by its `weakest.*` keys, and where code found one, a decide plan whose Watch cites none is refused (an input that reads no figure against a line has no weakest margin, and nothing is asked of Watch). Tradeoff mode: The trade · Where the answer flips · Our lean · What would settle it. **Confidence is conditional on the caveats, never capped by them** (review, owner-endorsed, 2026-10-25). Exactly three forms:
   - **"High"**: no open report-wide caveat remains.
   - **"High, if the caveats below aren't material"**: the evidence is strong, and open caveats remain. Each gets one clause saying what would make it material ("The judge is uncalibrated. Material only if it misreads in-character replies"), or, where the deciding finding doesn't rest on it, saying so ("cost, the deciding finding, is measured and doesn't depend on it").
   - **"Low"**: the evidence is weak even if every caveat is immaterial, for example a pick nothing separated. A choice whose declared gates leave its pick the one dominant survivor (§1.1) is separated by the gates and cost, not by pair readings, so it may state the confidence those earn; a choice with no gate, picked where no pair reading shows it ahead, is "low".

   **Code removes the caveats it can prove immaterial** and renders one muted checked line for each, which counts as nothing open: a late margin, floor or bar is immaterial when the decision (the allowed modes and the dominant option) is the same across its whole plausible range. A margin's range is every margin from zero up, cut at each reading's flip margin (`-lo`, §3); a floor's, every margin it could take, cut where each read flips between held, unclear and broken; a bar only warns, so the call holds at any bar. "The late margin was checked: the call holds at every margin." A late aspect, control or guardrail margin can't be checked this way and stays open. Nobody attests a caveat away: there are no acceptance records.

   Code enforces the form: plain "High" is refused while an open report-wide caveat remains, and "High, if…" is refused unless it lists every open report-wide caveat, each with its clause. "Low" is always allowed. Kairo round 2: "Keep September. Confidence: high, if the caveats below aren't material." · the judge · the aspect, chosen after the results · the control, chosen after the results · "The late margin was checked: the call holds at every margin." (and the same for its late floor).
6. **Options table.** 3–5 rows: the recommendation first, then the near-miss, the most accurate, and the cheapest (skip duplicates). Columns are the criteria plus cost, at real volume when the volume is known. Code writes the columns, each where the input holds its figure, from one closed list of families (`report_plan_checks.OPTION_COLUMN_FAMILIES`); a plan holding any other column is refused when it is generated:

   | Column | When code writes it | Cell |
   |---|---|---|
   | `pass_rate` | a pass rate resolves for some arm | the percent, with "(k of n)" from the scoreboard |
   | `reading.change` | an aspect is asked and read | the change in points, with its interval |
   | `reading.verdict` | the same | the reading's pill ("Control"; "Worse, within margin"), by the §3 status rule |
   | `judged:<dim>` | the primary judged dimension, and a bar's or guardrail's judged dimension | the percent or score |
   | `gate` | a gate is declared | the gate status (§3) with its first binding reason |
   | `measure:<key>` | a bar's or guardrail's measure | the mean, in its unit |
   | `cost_per_1k` | a cost per 1,000 replies resolves for some arm | dollars, with "(+28%)" against the control |
7. **Summary charts.** One or two (at most three), beside or under the options table, in the same row order and colors (§5.3).
8. **Arms key** whenever any arm's full configuration is longer than about 24 characters, or any setting other than the model varied (§6.1).
9. **Evidence sections.** One per question the decision raises: eyebrow · headline stating the finding · one line on how to read the chart · the chart. The first draws the shape's spine, which code gives (§1); add supporting sections only if they change or qualify the decision.
10. **Explore part.** Under a divider. 1–3 controls for business choices (a margin, a floor, a price per point, a weighting, a lever level). It recomputes with the same gate function as the decision and says in words where the answer flips. When no setting can flip the decision (a dominated choice), it says so in one sentence and shows where each arm's *reading* flips instead (for example, the margin below which a candidate stops holding).
11. **Full results.** Every arm × measure, contrasts with their level, and strata, as sortable tables. These double as the table view of every chart.
12. **Method.** 4–8 short term/definition pairs: cases and repeats, what "pass" means, the interval method and level, correction, the judge and its tier, cost basis, what wasn't compared, and the source.

Rules for every section:

- One gate function decides pass / borderline / fail (§3). The headline, the options table, the matrix, the charts and the explore part all call it, so they cannot disagree.
- Every number in prose is bound to data (§6.2). If a figure isn't in the input, the sentence doesn't say it.
- Every chart the plan names draws on the plan's own input. When a plan is generated, code draws each chart through the renderer's own resolvers and refuses one that does not draw, at its field, for the planner to repair: one rule at both ends, so no stored plan names a chart its page cannot draw.
- Drop a section that would show nothing new. A task with no task-specific behaviors gets a one-line note, not an empty chart.
- Method statistics (depth, repeats, pass^k, precision decomposition) go in Method or Full results. They never sit between the decision and the evidence.
- When the input was trimmed and a comparison the page wants is missing, say so in one caption line. Never fill it in.
- **Prior insights that the new data contradicts** go in the Watch slot, in one sentence ("the earlier finding that X lowers cost does not hold here"). Ones it confirms go nowhere; the evidence already says it.

---

## 3. Gate logic (shared by every part of the page)

```text
check(arm, stratum, criteria) -> {status, fails[], borders[]}
  for each criterion c (quality floor, behavior floor, latency budget, margin, guardrail):
    v  = point estimate; lo, hi = its interval at the level the criterion declares
    if v fails c                        -> fails += reason        # e.g. "p95 3.6 s > 3.0 s"
    elif the interval crosses c's limit -> borders += reason      # e.g. "90% ±4.4"
  status = "fail" if fails else "borderline" if borders else "pass"
```

- Borderline counts as passing for the decision, but it must appear in the Watch slot with the sample needed to settle it.
- A cross-cutting aspect gates every candidate that would receive routed traffic, not one task.
- **Non-inferiority reading** (shape D). For a candidate with change Δ and interval [lo, hi] against margin M:
  - `hi < 0`: **worse**. Lead with the size ("worse by 3.4 to 10.8 points"), then say whether it is within the margin.
  - `lo > 0`: **better**.
  - otherwise: **not separated**.
  - Within the margin when `-lo < M`. The margin at which the reading flips is `-lo`. Show it in the explore part.
  - A margin that no plausible result could fail (larger than twice the worst observed drop) gets flagged as uninformative.
- **Status color follows the direction of the change, not the margin.** Worse is ✕ fail-red even when it is within the margin ("Worse, within margin"); not separated is neutral gray; better is ✓ green. The margin verdict goes in the words. A green pill on a regression reads as approval.
- **Gate the asked aspect on its readings** (`aspects[].readings`). If a floor `speaks_for` a different measure than its reading (a judged dimension instead of the pass rate), show that floor separately and name the mismatch in Method. Never merge the two.
- **"Pass" is whatever the bundle defines.** If the input doesn't say which checks and judged dimensions make a trial pass, Method says so plainly ("a trial passes when its checks and judged dimensions pass; the bundle doesn't list which") instead of guessing.
- **Routing.** For each routable task, the cheapest passing candidate (among those that also pass every cross-cutting aspect). Blended cost uses declared traffic shares only.
- **A chart's floor or budget line is a criterion's own threshold** (§5.2): the one criterion that reads the chart's measure, so the line a chart draws and the status the gate function gives come from the same declared rule.

---

## 4. Tradeoff mode

Use it when §1.1 says so. The page prices the trade; it doesn't pretend to decide it.

1. **Headline states the trade, with both sides and where the loss sits.** "Switching to A (flash · 64k out) saves $7.6k a month but loses 6 points of personality, mostly in long conversations."
2. **Flag:** "No rule set in advance," naming who makes the call.
3. **Price one axis in the other.** Savings per point lost, per month, at real volume: `ppp = (cost_control - cost_option) × volume / (points lost)`.
4. **Flip points.** For a price p per point, net value = savings − p × points lost. Sweep p and record where the best option changes. Compute this over the 2–4 options an exec would choose between, not every variant, or the bands get too thin to act on.
5. **Middle path.** If the loss concentrates in one stratum, model a hybrid that avoids it (for example, hand off after turn 5). Label it *modeled, not measured*, and state the assumption it rests on.
6. **Anchor points to people.** Show any evidence of what users notice (blind preference, CSAT, retention). If there is none, say so in Watch.
7. **Lean.** Allowed, labeled "Our lean", with the price band it holds for and the experiment that would settle it.
8. **Explore** = a price-per-point slider, a net-value ranking, the flip bands, and a savings-against-loss map with the price line.

---

## 5. Charts

### 5.1 Choosing a chart

| Data job | Use (kit function) | Never |
|---|---|---|
| Pass/fail against several criteria, per candidate × task | Gate matrix (table cells with pill, reason, cost; cheapest passing highlighted) | A score heatmap that hides which criterion failed |
| Change from a control, with interval and margin | `forest()` | A sentence saying "no worse than the margin" |
| One rate per candidate against a floor | Dot + interval strips, floor dashed (r1 `dims`) | Bars from zero hiding a 3-point gap |
| Two measures that trade off | Scatter with gate region or floor, Pareto line, hollow marks for failures, `placeLabels()` | A dual-axis chart |
| Magnitudes per option (cost, error rate) | `barsH()` from zero, reference line for the budget | Pass rates on a truncated axis: chart failure rate from zero instead |
| Deltas per stratum or lever | `barsH()` diverging about zero; muted when not separated | A before-and-after table |
| Ordered x (turn depth, version, date, input length) | `lineChart()`, bands on at most two series, end labels | Lines over unordered categories |
| Latency distributions | p50–p95 range bars with the budget line | Mean only |
| Two crossed factors | Lattice of tiles with metric, cost, latency, pill | Grouped bars hiding the interaction |
| Target matrix of cells | Sequential heatmap, n in each cell, red outline below the floor | One overall number |
| Errors between classes | Row-normalized confusion, one per label axis | A 9 × 9 matrix on a phone |
| Difference between two configs | Diverging heatmap, gray within ±0.5 | Two heatmaps for the reader to subtract |
| What failures look like | Example cards: id, case, stratum, arm, judge reason, excerpt | A wall of transcripts |

### 5.2 Mechanics

- **One unit per chart, one interval system per chart, and the level is printed on it** ("95% intervals", "99.69% intervals"). Don't mix confirmatory 95% readings and family-corrected contrasts in one chart.
- **Log axes** when a measure spans more than 10×; the caption says so.
- **A reference line sits at a declared line, and code fills its value.** The planner asks for the kind; code fills the figure key at completion:
  - a **margin** on a forest, at the chart's aspect's margin (`aspect.margin`, or `aspects.<id>.margin`);
  - a **floor** or a **budget**, at `criteria.<id>.threshold`, the threshold of the one criterion that reads the chart's one measure (the measure its query names, else the one its rows read). A floor's criterion is higher-is-better and a budget's lower-is-better, as the kit draws them (pass at or above a floor, at or below a budget). Its statistic is one the chart draws (range bars draw p50 to p95, every other kind the mean). It reads where the chart does: over the whole comparison or each task, within the one task the chart keeps, or each cell on a chart of cells.

  A line no criterion gives, or that several give, is refused naming why; the planner draws no line or the line a declared criterion gives. Never a line at a value the input does not declare.
- **Small multiples** (`small_multiples_by`) only on a kind the kit draws as panels: bars and diverging bars, the two scatters, dot intervals, heatmaps and confusion matrices. On any other kind they are refused when the plan is generated.
- **Identity.** One color per arm for the whole page, plus a shape or letter tag, so color is never the only cue. The control is ink-gray; candidates take series colors in fixed order.
- **Status colors** (good / borderline / fail) always come with ✓ ! ✕ and a word, and never reuse a series color.
- **Labels.** Use `placeLabels()`, which avoids other labels, markers and reference text. On phone widths, label only the recommended point and the extremes, and use letter tags.
- **Hover adds detail but is never the only way to read a value;** Full results carries everything.
- **Charts render at the container's real pixel width** (`chart()`), so text stays readable on a phone.
- **Descriptive versus inferential.** Per-stratum deltas computed by the generator (not the pipeline) are descriptive: say "uncorrected", bold only rows separated at 95%, and call them leads.

### 5.3 Summary charts at the top

- Use the same rows as the options table, in the same order and colors. Fade options that fail a gate; bold the recommended one.
- Bars start at zero. For rates near a ceiling, chart the error or failure rate from zero with the budget as a reference line.
- One chart per decision question: cost, the gating metric, and the losing axis if there is a trade. At most three.

---

## 6. Words

### 6.1 Naming arms

- **Store arms as structured settings**, never only as strings: `model` plus every setting that varied, and an optional owner `alias`.
- **Full configuration once,** in the arms key, in mono type, with the role.
- **Short name** = the short name the campaign declares for the arm (`arms[].short_name`, at most 32 characters, unique among the campaign's arms) if set; else the alias if set; else only what differs from the control, with the model trimmed of the prefix it shares with the control's model. "Meridian-4.1-flash · 64k max output · reasoning medium" against "Meridian-4.1 · 32k max output · reasoning medium" becomes "flash · 64k out". The control is named by its alias, or by its role ("Production").
- **Collisions** append the next attribute that separates them. A declared short name never collides, since the campaign refuses two arms named alike; only a computed one can.
- **Letter tags** (A, B, C; the control is P) in colored squares go in tight places; on phones, the tag alone.
- **Factorial configs** are named by their factor levels: "Brisk-S · reasoning low · 1 retry".
- `armLabels()` in `kit/kit.js` implements this.

### 6.2 Numbers in prose

The planner writes prose with figure keys, and the renderer formats the values.

Grammar: `{path|filter|filter}`, in single braces. (The memo's `{{cell|…}}` references are a different grammar; plan prose never holds a double brace.) `{arm:<id>}` prints the arm's short name with its tag and takes no filter. `figure_keys.py` (`threetears.evals.analysis.report`) parses and renders keys, and writes the planner prompt's section from the same definitions.

**Paths are matched, never split at dots.** Ids carry dots and spaces (`qwen3.8-flash`, `persona.holds_character`, `character adherence (v2)`), so a path is matched against a closed vocabulary built from the input: a family keyword, then the ids that family takes, then a statistic or field from a fixed set. Exactly one complete reading is the figure. Two readings are refused, naming both. No reading is refused, naming what could come next where the path stopped matching. An unknown path is a validation error, never a blank.

| Path | Reads |
|---|---|
| `meta.<field>` | campaign, question, behavior, kind, kind_means, design_shape, run_date, provenance, volume_per_month |
| `control.<field>` | the control's id, name, served_model, description |
| `n_candidates`, `cases`, `repeats` | candidates; the asked aspect's cases and repeats per case |
| `aspect.<field>`, `aspects.<aspect>.<field>` | the one asked aspect, or any aspect: id, name, margin, description |
| `reading.<arm>.<statistic>`, `aspects.<aspect>.reading.<arm>.<statistic>` | change, interval, low, high, margin, level, verdict, control_passed, arm_passed, why |
| `contrast.<arm>.<measure>.<statistic>` | delta, interval, low, high, verdict, p_adjusted |
| `judged.<dim>.<arm>` | the arm's mean on a judged dimension |
| `judged.<dim>.<field>`, `judge.<field>` | label (dimension only), model, tier, calibration, calibration_ratings, agreement; `judge.*` only where every dimension agrees |
| `pass_rate.<arm>` | passed over usable (scoreboard), else the `pass_rate` measure |
| `task.<task>.<arm>.<measure>.<statistic>`, `task.<task>.<field>` | an arm's figure on a measure or judged dimension within one task (mean, ci, half_width, sem, n, n_cases, p50, p95, change); the task's name, n_cases, share |
| `criteria.<id>.<field>` | a declared criterion: threshold (in its measure's units), measure (its reader name), statistic, direction ("higher is better") |
| `cost_per_1k.<arm>` | a `usd_per_1k` measure, else a `usd_per_1m` measure divided by 1,000, else the one `usd` measure on the cost merit axis, else the scoreboard's cost per reply, per 1,000 |
| `measure.<key>.<arm>.<statistic>` | mean, ci, half_width, sem, n, n_cases, p50, p95, change |
| `scoreboard.<arm>.<statistic>` | ran, usable, passed, cases, lost, failed, reply_cost_usd, test_cost_usd |
| `reasoning_ratio.<arm>` | the share of output spent reasoning |
| `late_criteria.set_at`, `family.<field>`, `caveats.latency`, `strata.<id>.<field>` | when late criteria were set; the one contrast family (disclosure, correction, alpha, family_size, level); the latency caveat; a stratum's name, n_cases, share |
| `weakest.<field>` | the weakest margin (§2 item 5): arm, criterion (what it is read against, in reader words), value, threshold, margin (the gap, in the line's units: points for a rate or a change) |

**Derived values** are computed by the renderer from the input's point figures. They are descriptive: no interval, no test. `cost.uplift.<arm>` is the candidate's cost per 1,000 over the control's, minus one. `cost.uplift_range` spans the candidates' uplifts. `strata.<id>.loss_share.<arm>` is the stratum's part of the candidate's loss against the control on the primary judged dimension, weighted by cases (or by traffic share), within its stratification; `strata.<id>.loss_share` is the least of those across candidates ("about X of each candidate's loss"). `flip_margin.<arm>` is the asked reading's worst drop, the margin above which it holds. `weakest.*` reads the weakest margin. A `.name` or `.label` the input leaves undeclared prints the id's reader name (its underscores and dots as spaces), never the id.

**Filters**, one spelling at most plus `abs` and `interval`, in any order: `pct` (a rate as a percent: 0.853 → 85.3%; a descriptive ratio in whole percents), `pts` (a change in points, signed, one decimal at most), `abs` (the magnitude: `abs|pts` → 7.6), `usd` (dollars and cents), `int` (nearest whole number, its thousands separated: 12000000 → 12,000,000; prose only, since the package's number rule writes no separator inside tables' intervals and lists), `pct10` (nearest 10%), `interval` (`[lo, hi]` → "-10.8 to -3.4"), `usd_per_1k` ("$1.89 per 1,000 replies"). Without a filter a rate reads as a percent, a change in points, money in dollars, and a measure stated in dollars per 1,000,000 replies (unit `usd_per_1m`) with its words ("$180 per 1,000,000 replies"; `int` reads it as a whole number, `usd` as the amount in dollars, `usd_per_1k` converted to dollars per 1,000 replies). An unsigned interval reads with its unit once ("27–28%"); one with a sign reads "lo to hi". Numbers follow the package's number rule (`numbers.py`), so a whole value drops its decimal (-6.0 reads -6) and minus is the hyphen-minus. A filter that does not suit the figure (a percent of money, points of a count, a percent of a change) is refused.

**D3: no digit outside a key.** Keys are set aside; then every name the input holds is set aside where a non-word character or the text's edge bounds it on both sides: arm ids and names, served models, settings as displayed (`64k`), aspect, measure, dimension, stratum, task and factor ids and names, the campaign's name, trial and case ids. So are, under the same bound, two fixed package vocabularies: the percentile names `p50`, `p90`, `p95` and `p99` (`p95 latency`, never `p-ninety-five`), and the package's own unit phrases, each matched whole (`per 1,000 replies`, `($/1,000 replies)`, `per 1,000,000 replies`), which are the templates rendered figures are joined with. Any Unicode decimal digit left (`95%`, `95 percent`, `1,000 cases`, `٣`, the `95` in `xp95`) is an error. An id made only of digits is never set aside: write `{arm:1017}`, never `1017`.

Until the renderer exists (CONVERGENCE.md P3–P4), a page is hand-built with the kit. Write the plan first and keep it beside the page, then compute every prose number in the page script from the same data the keys name. That keeps the two honest. For example: `"{arm:qwen3.8-flash} is clearly worse: down {reading.qwen3.8-flash.change|abs|pts} points ({reading.qwen3.8-flash.interval|pts})"`. That makes three things true by construction: no number is invented, rounding matches the tables, and a re-run updates the prose. A sentence whose figure is missing from the input is dropped.

### 6.3 Style

- **Headlines are findings,** with subject and direction: "Off-lane asks, the largest stratum, carry about half of each candidate's loss." Never a topic label ("Failure analysis").
- **Use reader names,** not identifiers: "cost per 1,000 replies", not `production_replicating_cost`; "off-lane asks", not `off_lane`. Ids appear only in Full results.
- **One name per measure.** A judged rate and a pass rate on the same aspect get different names ("in-character rate (judge)" vs "pass rate (asked)").
- **Say each caveat once,** next to the claim it qualifies.
- **A column whose value is the same in every row** moves into the heading.
- **No statistician's grammar above the fold.** "Each arm's gains and its drops are each shown wrongly at most 2.5% of the time" becomes, in Method: "95% intervals, not corrected across arms."
- **Sentences under 25 words.** Active voice. No hedging adverbs, no "worth noting".

---

## 7. Visual system

The renderer applies the kit (`kit/kit.css`, `kit/kit.js`). Don't restyle per report.

- **Host theming.** A host overrides only `--host-accent` (light and dark values). Series, status and sequential ramps are fixed so a brand color can never make a fail look like a pass.
- **Both themes** are designed: tokens on `:root`, dark values under `prefers-color-scheme` and under `[data-theme="dark"]`. Heatmap cell text uses `--seq-tN` tokens because the dark ramp runs the other way.
- **Type.** IBM Plex Sans Condensed for headlines, IBM Plex Sans for body and charts, IBM Plex Mono for ids and payloads. Tabular figures in tables and on axes.
- **Layout.** A 44rem prose column; figures and tables break out to 68rem. Small multiples use auto-fill grids that collapse to one column. Wide tables scroll in their own container; the page never scrolls sideways.
- **Restraint.** Borders only on figures. No shadows or gradients. One emphasis per chart.

---

## 8. Definition of done

A page is done when all of these pass. `scripts/check.py` automates the first five.

1. It renders at 1200 px and 390 px, in light and dark, with no console errors.
2. `document.documentElement.scrollWidth` equals the viewport width at both widths (nothing scrolls sideways).
3. No two chart labels overlap, and no label overlaps a marker or leaves its chart (inspect the screenshots; `check.py` lists any element past the right edge).
4. Every chart has a table view (Full results) and states its interval level.
5. The explore controls change the outcome text and the chart, and the text says where the answer flips (or that it can't, and where each reading flips).
6. The headline is a decision (or a trade, or a stated no-call) and matches the options table and the first chart.
7. Every number in prose appears in the input or is computed by the renderer from it.
8. Every entry in `caveats` (late criteria, the uncalibrated judge, confounds, apparatus confounds, short runs, latency, unreported observations) lands exactly once: checked immaterial by code (one muted line); or open and listed in the decision's confidence with its clause (report-wide ones; under "Low", the flag may carry them instead); beside a named claim (scoped ones); or in Method (apparatus confounds, latency and unreported observations only). The confidence takes one of the three forms (§2 item 5) and plain "High" appears only with nothing open. The plan records each landing structurally (caveat ids on the confidence, a decision slot, a section or a Method pair), so the coverage check never searches prose.
9. No internal id, hash or measure key appears above Full results. Trial and case ids may appear on example cards, in small mono type, as references a reader can look up. Code supplies each id's one reader name to the planner, and refuses above Full results an id that reads as one (an underscore, dot, colon, slash, equals sign, bracket or quote in it) where it differs from its reader name, and any quote of the campaign's question (`{meta.question}`): the campaign writes its question in its own ids (Kairo's `dj_character_v2`), which no reader name can replace, so the page says what it asks in reader words.
10. A reviewer who sees only the first screen can say what to do, why, and what the main risk is.
