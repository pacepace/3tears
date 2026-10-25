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

Code reads these signals from the report input, applies the shape lookup below, and writes `plan.signals` and `plan.shape` before the planner runs; the planner receives them as facts and never writes them. The same holds for the masthead, arm colours, and each chart's interval level, unit and axis range (owner ruling, 2026-10-25: code computes facts; the planner chooses the mode, the sections, the charts and every sentence).

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

### 1.1 Mode: decide or trade off

After applying guardrails and gates, compute the surviving options. Then choose the mode:

- **Decide** when one surviving option is at least as good on every axis that matters (cost, the asked aspect, any gate). Kairo round 2 is this: the control is cheaper and better, so it dominates.
- **Trade off** when two or more survive and none dominates, and the campaign declared no rule that ranks them (`merit_priority` empty, no exchange rate). §4 covers this mode.
- **No call** when no option survives, or the evidence can't separate them (every interval crosses zero and the margin). The headline says what is missing and what would settle it.

---

## 2. The page contract (every shape)

Top to bottom. The renderer enforces the order; the planner fills the slots and picks the middle.

1. **Masthead.** One line: kind · system under test · counts (arms, aspects or tasks, cases × repeats) · run date · `Synthetic example data` when it is. Never a paragraph.
2. **Headline: the decision as an instruction with its reason.** "Keep September. Both candidates cost 27–28% more per reply and stay in character less often." Never the eval question restated. In tradeoff mode it states the trade (§4).
3. **Lede.** At most two sentences: the strongest supporting fact, and what would change the answer.
4. **Flag** (only when needed). One amber line that combines every report-wide disclosure: criteria set after the results, an uncalibrated judge, and any confound or short run that touches the whole comparison. It says which findings it weakens and which survive. It is said once, here, and not repeated beside each finding. A confound that touches one arm or one finding (an arm's reasoning budget, one short run) is said once beside the claim it qualifies, usually in Watch, and not in the flag: a caveat is stated once, at the level it applies to (owner ruling, 2026-10-24). Neither a late criterion nor an uncalibrated judge lowers the recommendation's confidence: the flag discloses them, and the reader weighs them (owner ruling, 2026-10-25).
5. **Decision block.** Four slots. Decide mode: Recommendation (with its numbers) · Why not the others · Watch (the weakest margin, a confound, what wasn't measured) · Next. Tradeoff mode: The trade · Where the answer flips · Our lean · What would settle it.
6. **Options table.** 3–5 rows: the recommendation first, then the near-miss, the most accurate, and the cheapest (skip duplicates). Columns are the criteria plus cost, at real volume when the volume is known.
7. **Summary charts.** One or two (at most three), beside or under the options table, in the same row order and colors (§5.3).
8. **Arms key** whenever any arm's full configuration is longer than about 24 characters, or any setting other than the model varied (§6.1).
9. **Evidence sections.** One per question the decision raises: eyebrow · headline stating the finding · one line on how to read the chart · the chart. The shape picks the spine; add supporting sections only if they change or qualify the decision.
10. **Explore part.** Under a divider. 1–3 controls for business choices (a margin, a floor, a price per point, a weighting, a lever level). It recomputes with the same gate function as the decision and says in words where the answer flips. When no setting can flip the decision (a dominated choice), it says so in one sentence and shows where each arm's *reading* flips instead (for example, the margin below which a candidate stops holding).
11. **Full results.** Every arm × measure, contrasts with their level, and strata, as sortable tables. These double as the table view of every chart.
12. **Method.** 4–8 short term/definition pairs: cases and repeats, what "pass" means, the interval method and level, correction, the judge and its tier, cost basis, what wasn't compared, and the source.

Rules for every section:

- One gate function decides pass / borderline / fail (§3). The headline, the options table, the matrix, the charts and the explore part all call it, so they cannot disagree.
- Every number in prose is bound to data (§6.2). If a figure isn't in the input, the sentence doesn't say it.
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
- **Short name** = alias if set; else only what differs from the control, with the model trimmed of the prefix it shares with the control's model. "Meridian-4.1-flash · 64k max output · reasoning medium" against "Meridian-4.1 · 32k max output · reasoning medium" becomes "flash · 64k out". The control is named by its alias, or by its role ("Production").
- **Collisions** append the next attribute that separates them.
- **Letter tags** (A, B, C; the control is P) in colored squares go in tight places; on phones, the tag alone.
- **Factorial configs** are named by their factor levels: "Brisk-S · reasoning low · 1 retry".
- `armLabels()` in `kit/kit.js` implements this.

### 6.2 Numbers in prose

The planner writes prose with figure keys, and the renderer formats the values.

Grammar: `{path|filter|filter}`. The path is a dotted path into the report input, with arm ids as keys (`reading.<arm>.change`, `contrast.<arm>.<measure>.delta`, `judged.<dim>.<arm>`, `cost_per_1k.<arm>`, `pass_rate.<arm>`, `meta.question`), or a derived value the renderer defines (`cost.uplift.<arm>`, `cost.uplift_range`, `strata.<id>.loss_share`, `flip_margin.<arm>`). `{arm:<id>}` prints the arm's short name with its tag. Filters: `pct` (0.853 → 85.3%), `pts` (signed, one decimal), `abs`, `usd` (two decimals), `int`, `pct10` (rounds to the nearest 10%), `interval` (`[lo, hi]` → "−10.8 to −3.4"). An unknown path is a validation error, never a blank.

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
8. Every late criterion, uncalibrated judge and confound in `caveats` is disclosed exactly once: report-wide ones in the flag, arm- or finding-scoped ones beside the claim they qualify.
9. No internal id, hash or measure key appears above Full results. Trial and case ids may appear on example cards, in small mono type, as references a reader can look up.
10. A reviewer who sees only the first screen can say what to do, why, and what the main risk is.
