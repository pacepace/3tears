# Design: track T3 "Structure" (evals big-picture program)

Branch `evals/main`, measured at `7b2e0afeb`, revised against `9dc3882ea`. Read-only design. Paths are under
`packages/evals/src/threetears/evals/` unless they start with `packages/` or `tests/`.

## Revision 2 (2026-11-14)

Folded in: the binding constraint document (`information-architecture.md`, cited below as IA with its rule ids), the
judges track's field list (`design-judges.md` §9.3, chunk B0), and decisions since revision 1. Read against
`evals/main` at `9dc3882ea`.

| Change | Where |
|---|---|
| Status: S1 is merged (the direction test, ten known pairs). S2 and S3 are being built. Q3 is answered yes: v3 lands before the shape change | §7, §9 |
| S5a is re-specified as a build spec and split in three: S5a-1 (the shape, the closed sets and the first-screen function; moves no digest), then S5a-2 (the answer read moves down; caveats raised and ranked) and S5a-3 (the structure fields filled), built in parallel and merged as one | §2.6, §7 |
| The report input carries reading states and reasons as typed values, one typed caveat list, and one first-screen value of four slots; S5b is its single rendering on both pages | §2.6, §2.7 |
| The judges' B0 fields, amended to the IA's names, and the four truth fixes' fields land in the one shape move | §2.6.5 |
| What S5a and S5b subtract, as numbers | §2.8 |
| S5c: an absent interval has one representation. S7: `why.kind` is two enums, and an arm id can name one cell | §2.4, §3.3, §3.4 |
| Seven places where the IA does not match the code as it is, each with the line taken | §2.9 |
| Module moves are safe for the production host, which imports only from public roots; shims in S6 and S9 are for tests alone | §7 |

Sections 0, 1, 3.1, 3.2, 4, 5, 6 and 8 are unchanged from revision 1 except where a row above names them.

Terms, used one way throughout:

- **Numbers**: code that computes, decides or assembles (statistics, verdicts, the bundle, the lenses, the report
  input and the readings over it).
- **Presentation**: code that turns numbers into something read (pages, forms, charts, keys, the analyst's tools, the
  writer, the harness, command-line text).
- **Boundary shape**: the one typed value presentation reads numbers through.
- **Identity**: anything stored or compared to answer "is this the same thing": a fingerprint, a digest, a recording
  key, an arm id.

## 0. Measurements

All counts are from an `ast` walk of the tree at `7b2e0afeb` (441 modules, 10,496 intra-package imported names;
relative imports resolved; `TYPE_CHECKING` imports counted; there are no function-level imports in the package).

### 0.1 Sizes

| Area | Modules | Lines |
|---|---|---|
| `analysis/` whole | 191 | 116,721 |
| `analysis/report/` | 100 | 56,724 |
| `analysis/analyst/` | 13 | 8,130 |
| `analysis/bundle/` | 22 | 15,473 |
| `analysis/lenses/` | 11 | 6,982 |
| `analysis/*.py` (top level) | 45 | 29,412 |
| `ops/report_writer/` + `harness/` | 24 | about 12,800 |

### 0.2 Import edges inside `analysis` (module pairs / imported names)

| From | To | Module pairs | Names |
|---|---|---|---|
| top level | `report/` | 34 | 62 |
| `lenses/` | `report/` | 2 | 3 |
| `bundle/` | `report/` or `analyst/` | 0 | 0 |
| top level | `analyst/` | 0 | 0 |
| `report/` | `analyst/` | 20 | 37 |
| `analyst/` | `report/` | 33 | 87 |
| `report/` | top level | 205 | 575 |
| `report/` | `bundle/` | 44 | 79 |
| `bundle/` + `lenses/` + top level | `analysis/reporting.py` | 24 | 112 |

The audit's "16 numbers modules import `report`" is 13 top-level modules, 2 lens modules and `analysis/__init__`.

The 34 top-level pairs come from 13 modules: `agreement`, `analyst_gaps`, `bar_proposals`, `campaign_readings`,
`doc_analysis`, `gate`, `judge_temperature`, `lens_text`, `out_of_run_spend`, `references`, `service`, `significance`,
`summary`. `service` alone is 11 pairs and 21 names; `summary` is 6 pairs.

### 0.3 Two findings that change the plan

1. **`analysis/reporting.py` is misnamed, not misplaced.** It is the score projection (runs and results into flat
   rows; 2,025 lines) and imports one analysis module (`completeness`). The 24 importing modules are numbers
   reading numbers. Nothing moves; the module is renamed late in the track (S9).
2. **An enforcement test caused eleven of the edges.** `packages/evals/tests/test_one_owner_of_number_forms.py`
   (2,475 lines) rules that above the engine trees "`PageForms` is the only route, through `PageForms.off_page()`".
   So `significance.py`, `agreement.py`, `bar_proposals.py`, `judge_temperature.py`, `out_of_run_spend.py`,
   `references.py`, `summary.py`, `lens_text.py`, `lenses/cost_estimate.py` and two more import the 3,428-line
   `report/forms.py` to spell a count or a dollar amount in a command-line sentence. `off_page()` is
   `PageForms("off", {})`: it holds no page and no measure.

### 0.4 The two page pipelines, by which shape a module reads

`report/` and `analyst/` modules, by whether they import the bundle (`bundle/*`, `campaign_bundle`,
`campaign_readings`) or the report input (`report_input`, `report_margins`):

| Reads | Modules | Lines | Where |
|---|---|---|---|
| Bundle only | 26 | 16,078 | `report/aspect`, `report/campaign`, half of `report/` top |
| Report input only | 34 | 19,201 | `analyst/`, `report/doc_page`, most of `report/keys` |
| Both | 15 | 16,831 | `forms.py`, `number_forms/sets.py`, `call.py`, `parts/charting.py`, `keys/reads.py`, `keys/resolve.py`, `keys/settle.py`, `input_build.py` and 7 more |
| Neither | 38 | 12,744 | blocks, serializers, SVG, vocabulary |

`PageForms` has four constructors over the two shapes (`of`, `of_campaign`, `of_bundle`, `of_aspects`), and
`number_forms/code_page.py` (210 lines) and `number_forms/written.py` (361) decide the same thing twice.

### 0.5 Tests

| Measure | Count |
|---|---|
| Python files under `packages/evals/tests` | 709 (620 `test_*.py`), 262,395 lines |
| `from threetears.evals...` import statements | 4,617 |
| ... naming a module below a public root | 3,052 (66%) |
| Test files importing below the root in `analysis/report` | 202 (707 statements) |
| ... in `analysis/analyst` | 84 |
| ... in `analysis/bundle` | 79 |
| ... `analysis/report_input` | 90 |
| Test files that import `ast` | 47; 56 read the source tree at all |
| Asserts holding an English literal of three or more words | 2,086 in 345 files (heuristic; the audit's stricter count was about 1,500) |
| Test files still importing `report.figure_keys` (a 222-line re-export shim since the #1141 split) | 59 |

The last row matters: the #1141 split of an 8,210-line module moved no test, because a shim kept the old path. That is
the pattern section 4 generalises.

## 1. The boundary

### 1.1 The rule

> Numbers never import presentation, directly or through another module. Presentation reads numbers through the
> boundary shape and the typed values it holds.

### 1.2 Which side each module is on

**Numbers** (70 modules, 45,696 lines):

- `analysis/bundle/**`, `analysis/lenses/**`;
- top level: `agreement`, `arms`, `aspects`, `bar_proposals`, `campaign_bundle`, `campaign_readings`, `campaigns`,
  `cassette_mode`, `cell_mean`, `cells`, `completeness`, `confusion`, `contention`, `depth`, `depth_reading`,
  `errors`, `generation_tally`, `judge_drift`, `judge_kind_readings`, `judge_profiles`, `judge_temperature`, `kinds`,
  `latency_partition`, `measurement_windows`, `out_of_run_spend`, `quantities`, `reads`, `references`, `remainder`,
  `reporting`, `revisions`, `separation`, `significance`, `stats`;
- the boundary shape and the readings over it: `report_input`, `report_margins`;
- `style.py` for now. It imports nothing, and numbers call it to mint names (`arm_name`, `check_names`,
  `lever_spelled`). Section 3.3 splits it.

**Presentation** (120 modules in `analysis`, 70,162 lines; plus `harness/`, `ops/report_writer/`, `quick/` printing):

- `analysis/report/**`, `analysis/analyst/**`;
- seven top-level modules that sit above the page or only print: `service` (builds reports and reads both sides),
  `doc_analysis`, `gate` (reads a built `Report`'s verdicts), `summary` (a run's printed summary), `lens_text`,
  `campaign_title`, `analyst_gaps`. No numbers module imports any of the seven (checked), so naming them
  presentation costs nothing today. They move under `report/` in S9.

`analysis/__init__.py` is the public root: it re-exports both sides and is not followed, as
`test_report_layer_direction.py` already treats it.

### 1.3 Every violating edge under that placement

With the seven surface modules placed, 10 module pairs (23 names) from 9 numbers modules break the rule. Through
them, **35 of the 70 numbers modules reach presentation transitively today**, the whole bundle among them
(`bundle/*` imports `agreement`, which imports `report/forms.py`).

| # | Numbers module | Imports | Why | Where the code should live |
|---|---|---|---|---|
| 1 | `significance.py` | `report.forms.PageForms` | `off_page().lone(p).text` in one disclosure line | Spell through the leaf (`kernel/numbers.py`); see 1.4 |
| 2 | `bar_proposals.py` | `PageForms` | A proposal's reason sentence | Same |
| 3 | `judge_temperature.py` | `PageForms` | `render()` and `_side_text` | Same |
| 4 | `out_of_run_spend.py` | `PageForms` | Ledger lines (`money_ledger`) | Same |
| 5 | `references.py` | `PageForms` | Interval and sem text | Same |
| 6 | `lenses/cost_estimate.py` | `PageForms` | Band sentences | Same |
| 7 | `agreement.py` | `PageForms` | `tier_sentence_in(tier, forms)`, `_criterion_words` (about 90 lines) | The wording moves up into `lens_text.py`, which already prints a judge's tier; `analysis.tier_sentence` keeps its root export |
| 8 | `campaign_readings.py` | `report.decision` (12 names) | Floors are evaluated in `report/decision.py` (1,623 lines) | The decision moves down: `report/decision.py` splits into `analysis/floors.py` (the typed `FloorDecision`, `FloorStatus`, `evaluate_floor`, `read_floor`, `project_rate`: numbers) and what stays (the 70-odd lines that word a projection with `PageForms`) |
| 9 | `campaign_readings.py`, `lenses/frontier.py` | `report.said` (`Said`, `SaidFigure`) | A sentence held as words plus typed figures | `said.py` (163 lines, no analysis imports) moves down to `analysis/said.py` unchanged; section 3.4 then shrinks it |

Edges 8 and 9 are the two cases of "a decision both sides need moves down". Edges 1 to 7 are "a wording helper moves
up" or "a number is spelled by the leaf".

The 26 raw pairs not in this table are the seven surface modules' own imports of `report/`, which are legal once they
are named presentation.

### 1.4 Spelling a figure on the numbers side

Numbers modules still author short operator sentences (a refusal, a ledger line, a disclosure). Moving every one of
those up would split dozens of public functions and methods off stable types (`EvalSummary.render`, `tier_sentence`, the lens texts). The pragmatic line:

- A numbers module may spell a figure **only** through `kernel/numbers.py` (671 lines, the existing leaf:
  `format_number`, `format_signed`, `spell_dollars`, `spell_count`, `spell_p`, `spell_effect`, `spell_at`).
- The fixed forms that `PageForms.off_page()` adds today and the leaf lacks move down as leaf functions, each a few
  lines: a share (`f"{value:.0%}"`), ledger dollars (`f"${amount:.4f}"`), `in_unit` for a declared unit.
  `PageForms.share`, `money_ledger`, `in_unit` and `lone` then call the leaf, so a page's ledger is unchanged.
- `test_one_owner_of_number_forms.py` is re-pointed, not weakened: its "engine" line, today "trees whose imports
  never reach `threetears.evals.analysis`" (`_ENGINE_TREES`, ceiling 6), becomes "modules on the numbers side". Its
  walk still refuses a format spec, `round()` and `quantize()` everywhere; only the route for a fixed form changes
  from `PageForms.off_page()` to the leaf. Its `_PERMANENT` list should shrink by the entries that exist only because
  `forms.py` imports a helper it "cannot route through".

Output is byte-identical (the same primitives run). This is owner question Q1, because that test's line is an owner
ceiling.

### 1.5 Inside presentation: one cycle to break, one tangle to leave

`report/` and `analyst/` import each other (20 pairs one way, 33 the other). The cause is that `analyst/` holds two
different things:

- **Analyses over the input** (`cited`, `compare`, `levers`, `settle`, `analyses`, `facts`, `price`, `keys`: 7,019
  lines). They compute rows from the report input, and `report/rows.py`, `report/block_spec.py`,
  `report/parts/resolve_*.py` and `report/doc_page/facts.py` draw those rows.
- **The analyst's own surface** (`blocks`, `snapshot`, `skill`, `skill_ledger`: 1,092 lines).

Design: the eight analysis modules move to `report/analyses/`; `analyst/blocks.values_table` moves beside the two
`doc_page` modules that call it; `analyst/snapshot.frozen_input` moves next to `report_input.py` (it checks a digest;
it is numbers). `analyst/` then imports `report/` and nothing in `report/` imports `analyst/`. Old module paths keep
one-line shims for one minor (section 4.3).

**What this track does not untangle.** `report/keys/` (8,795 lines) fuses three things: the key grammar (parse,
vocabulary, grammar text), computations over the input (`separation`, `settle`, `prices`, `levers`, `reads`,
`counts`, `weakest`: about 5,200 lines that derive figures) and formatting. `report/call.py` and `report/gates.py`
(1,668 lines) decide materiality, dominance and gate findings. By the rule those computations are numbers. Pulling
them out means rewriting how a key resolves (each family computes inside its resolver), which is a rewrite of the
package's most-tested 9,000 lines for no reader-visible gain. The pragmatic line for this track: **the enforced
boundary is between evidence numbers (everything up to and including the report input) and everything that reads the
report input.** Section 2 narrows the gap from the other side, by moving each derived fact that both pages need
(flip margin, caveats, paired baseline) into the input, so `keys/` computes less over time. A later track can lift
what remains.

### 1.6 The enforcement test

Generalise `packages/evals/tests/test_report_layer_direction.py`, which already has the machinery (direct imports,
`TYPE_CHECKING`, relative imports, dotted-string patch targets, laundered reaches through a third module, synthetic
trees that prove the checker fires). Today its facts side is three modules by file-name pattern. The change:

- The **presentation side is a path set**, written once: `analysis/report/**`, `analysis/analyst/**`, and a tuple
  `_SURFACES` of the seven top-level modules. Everything else under `analysis/` is numbers, so a new module is held
  to the rule the day it lands and nobody has to remember to classify it.
- `test_no_numbers_module_imports_presentation` (direct) and
  `test_no_numbers_module_reaches_presentation_through_another` (laundered), over the whole numbers side.
- `test_presentation_outside_analysis_is_never_imported_by_analysis`: nothing under `analysis/` imports `ops`,
  `harness`, `quick`, `actions` or `transports` (zero edges today; `test_package_matrix.py` holds the package level,
  this pins it per module).
- `test_the_report_does_not_import_the_analyst` after S6.
- The existing non-vacuity and planted-violation tests, re-pointed at the wider side.

**Exemptions.** The first chunk lands the test with a ratchet: a frozen set of exactly the 10 module pairs in 1.3,
keyed `(module, target)`, with two more asserts: the set may only shrink (a pair not found in the tree fails the
test), and its size is pinned. S2 removes all ten and deletes the set, so the ratchet lives for two chunks. `_SURFACES`
is not an exemption list: it is a placement, and S9 empties it by moving the files.

## 2. One shape for both pages

### 2.1 Decision

The report input stays, and becomes strictly **a projection of the campaign bundle computed by one function on the
numbers side**. The bundle carries the facts; the input is how presentation reads them. Rules:

1. **One builder.** `report/input_build.py` (1,286 lines) moves down to `analysis/report_input_build.py` (where it
   lived before #1141 moved it up for three wording imports). Its three presentation imports go: `report.words`
   names become structured (`ArmSpelling`, 2.3), `report.campaign.names` likewise, and `report.decision` is already
   down after S2. The analyst's "counted" input stays one function with one flag; no second builder exists.
2. **The builder computes nothing.** A fact is computed where the bundle is assembled and copied through. Held by a
   test: the builder module imports no function from `stats`, `separation` or `cell_mean` (types only), in the style
   of the source-walking gates.
3. **A fact both pages state exists once.** When the code page states something the input lacks, the fix is a bundle
   field plus its projection, never a second computation in `report/`.
4. **The code page reads the same forms.** The code page keeps reading the bundle for layout in this track (16,078
   lines read only the bundle; re-plumbing them is not affordable here), but it takes every number form from the
   input (2.4).

Rejected: making the bundle the only shape. The analyst's tools, the frozen snapshots, the 42 frozen writer cases and
Discodon's reads are all report inputs, and the input is the smaller, documented, content-addressed shape. Also
rejected for now: moving the code page onto the input wholesale. It is the right end state and is listed under
"later".

### 2.2 The parity guard: what "by construction" means here

The class of bug (#1125, #1129, #1067) is "the code page can state it, the written page cannot". A structural test
closes the class rather than each instance:

- For each hot-path fixture (`tests/fixtures/hot_path_oracle`: `toyhost`, `courierhost`, `scaled`,
  `scaled_two_aspects`), build the code page and collect the typed figures it issued. `PageForms` already keeps a
  ledger of every figure it hands out, by measure, kind and role (`PageForms.ledger`, `issued`).
- Build the report input from the same bundle and assert that every `(measure, arm or contrast, role)` the code page
  issued resolves from the input to the same value.
- A figure with no counterpart fails the test and names the measure and role.

It asserts values and roles, never text. The first run will fail on exactly the open issues, which is the work list
for S5.

### 2.3 What the shape gains, and what each addition closes

All fields optional, left out of the document when unset (the convention `report_input.py` already follows in 17
places, so an input frozen earlier digests as it did).

| Addition (bundle field, projected to the input) | Closes |
|---|---|
| A contrast's paired baseline and the cases it is the mean of (`FamilyComparison` → `Contrast.baseline`, `baseline_n_cases`) | #1125 |
| `Factor.ordered: bool = False`, set by the host's lever declaration or the bundle builder | #1126 |
| Each arm's spread per measure, or `no_variation` with the common value | #1129 |
| Each arm's structured spelling (`ArmSpelling`) on `Arm`; delete `report/words.reader_arm_name` (the text matcher) | #1141 last comment |
| A contrast's family id; a note with its cause when a contrast's delta is not the difference of the means beside it | #1142 U2 |
| One caveat list, each entry typed with class, reach, state and reason, assembled on the numbers side (§2.6; the list IA F1 names) | #1142 U5 |
| `Reading.flip_margin`, `Aspect.margin_informative` (the arithmetic is `report_margins.flip_margin` today) | #1142 U6 |
| A declared volume per subject on the campaign declaration → `meta.volume_per_month` | #1142 U7 |
| One cost reading per arm, with the cost-per-reply fallback decided once | #1067 |

Closed by construction (the parity guard then holds them shut): #1125, #1129, #1067, #1141 c3, #1142 U2 (note), U5,
U6. Closed by a field with no code-page counterpart, so by the field and its own test: #1126, #1142 U7, U2 (family
id). Not closed here: #1142 U4, U8, U9, U12, U15 (each needs an owner ruling or new statistics).

This is the burn-down's package E, with U5 and #1067 added and the builder moved down. Revision 2: the same shape
move also carries the reading states, the first-screen value, the judges' fields and the truth fixes' fields; §2.6
is the build spec and supersedes this table where they differ.

### 2.4 One owner of number forms at the boundary

The policy says a measure's form is decided once, by the measure. Today it is decided by four constructors over two
shapes, in two parallel implementations. Design:

- A pure function on the numbers side of the boundary: `measure_forms(report_input) -> MeasureForms`, a table of one
  `MeasureForm` per `(measure key, kind)`: unit, places, money form, set scope. It reads only the input. It is the
  one place rule 2 of the policy (two significant figures of the narrowest interval's width) runs.
- `MeasureForms` is **derived on read, never stored and in no digest**. The owner changed the precision rule on
  2026-11-14; had forms been stored, that ruling would have staled every report.
- `PageForms` becomes a printer over a `MeasureForms`: `PageForms(measure_forms(input))`. `of_bundle`, `of_campaign`
  and `of_aspects` are deleted; the code page builds its campaign's input (it has the bundle) and uses the same
  table. `number_forms/code_page.py` and the bundle-reading half of `number_forms/sets.py` retire (about 450 lines);
  `forms.py` loses its 370 lines of constructors and bundle plumbing.
- The policy's `_SET_SCOPE` line survives unchanged, in one place.
- **An absent interval has one representation** (IA §5.1, S5c): the table reads `Stat.state` and `reason`
  (`not_available`, with `too_few` or `no_variation`) and one printer method spells it. The per-surface absent
  texts go. It reaches the first screen only as a class E caveat on the deciding measure (F8); otherwise it prints
  in its table cell.

Risk, stated plainly: where the two implementations disagree today, a code-page figure changes its printed precision.
By policy rule 1 any such difference is already a defect, but it moves hot-path goldens. They are regenerated
deliberately, with the diff reviewed and each moved figure explained, per the goldens ruling.

### 2.5 Stored compatibility

| Stored thing | Effect | Why |
|---|---|---|
| A frozen report input (snapshot, writer case) | Reads unchanged; digest unchanged | New fields are optional and absent; `frozen_input` checks the digest on the raw document before any model reads it |
| A stored report document over an old input | Renders; degrades | A page element whose field is absent prints as it does today (a change chart's zero label states the arm table's level; Full results shows no "no variation"; no line across configurations). Absence reads "not recorded", never a guess |
| `report_input_digest` of a campaign frozen again | Moves once | The shape lands unset in S5a-1 and fills in one merge (S5a-2 with S5a-3); `first_screen` is outside the digest (§2.6.6). `schema` stays `report-input/1` |
| The bundle's `schema_version` | Not bumped | Additive optional fields. Today any bump makes every stored bundle unreadable (`ASPECT_BUNDLE_SHAPE` refusal, "no earlier shape is read as stored since 66"). This track adopts: an additive optional field never bumps; a bump is reserved for a field whose meaning changed |
| The v2 bundle fingerprint | Must not move | It hashes the bundle's whole dict, so a new field would move it. Fixed by 3.1: v3 lands first and v2's pre-image is frozen to today's field set |

`packages/evals/MIGRATIONS.toml` gets one note for S5 naming the fields and the degraded reads.

### 2.6 S5a: build spec

S5a is the one place the first-screen value is computed (IA §5.3). A builder can start S5a-1 the day S2 merges; it
merges after S3, because S3 freezes the v2 pre-image that keeps new bundle fields out of stored fingerprints.

#### 2.6.1 Modules, all on the numbers side

| Module | New or moved | Holds |
|---|---|---|
| `analysis/reading_states.py` | New, about 150 lines | The shared closed sets (IA V1, V2, T4) and nothing else. Imports nothing from `analysis` |
| `analysis/report_input.py` | Edited | The typed fields of 2.6.5, the caveat entry and the first-screen value, as `_Part` models |
| `analysis/caveats.py` | New in S5a-1 (types, class table, subsumption, ranking); filled in S5a-2 | The one list: raisers, `CLASS_OF_KIND`, `REMEDY_OF_REASON` |
| `analysis/first_screen.py` | New, about 250 lines | `first_screen(...)`, the single function |
| `analysis/report_input_build.py` | Moved down from `report/input_build.py` (1,286 lines) | The one builder; attaches the caveat list and the first-screen value in S5a-2 |
| `analysis/answer.py` | Moved down from `report/call.py` in S5a-2 (about 600 of its 1,220 lines) | Standing, dominance, materiality: `asked_standing`, `surviving_options`, `dominant_options`, `caveat_materiality` and the private read they share |
| `analysis/criteria.py` | Split out of `report/gates.py` in S5a-2 (about 200 of 448 lines) | A criterion's status per arm, as arithmetic over the input. The line that words it stays in `report/gates.py` |

#### 2.6.2 The closed sets (`analysis/reading_states.py`)

```python
ReadingState = Literal["counted", "set_aside", "withheld", "not_available"]

SetAsideReason = Literal["instrument_unchecked", "instrument_below_bar", "instrument_leans",
                         "people_disagree", "identical_arms_separated"]
WithheldReason = Literal["unfinished", "split", "identity_unrecorded", "measured_under_concurrency"]
NotAvailableReason = Literal["too_few", "no_variation", "not_recorded", "replayed",
                             "not_drawn_at_random", "skipped_past_limit"]
Reason = SetAsideReason | WithheldReason | NotAvailableReason

IdentityState = Literal["one", "split", "unrecorded"]            # T4, one per coordinate
Coordinate = Literal["served_model", "measure_definition"]       # extended here and nowhere else
CaveatClass = Literal["A", "B", "C", "D", "E"]
CaveatReach = Literal["report", "scoped", "method"]              # moves down from report/call.py
Maker = Literal["person", "model"]                               # M4
```

`REASONS_OF: Mapping[ReadingState, frozenset[str]]` pairs them, and one validator, `check_state(state, reason,
coordinate)`, is what every model below calls: a reason outside its state's set is refused, `coordinate` is required
exactly when the reason is `split`, and `counted` carries no reason. `CriterionState` stays in
`kernel/evidence_tiers.py` (it exists; a check result is one). Tests, one per rule: every `(state, reason)` pair of
IA §2.3 validates and every cross pair is refused; a source walk in the style of the existing gates finds no other
`Literal` in the package holding two or more of these state words, so a second vocabulary cannot appear unseen (V1).

#### 2.6.3 The typed structures on the report input

All optional; each is left out of the document when unset or empty (`exclude_if`, used 20 times in the file today).

```python
class StateOf(_Part):                 # on a Reading, a Contrast, a Stat's interval, a PairRead
    state: ReadingState               # absent means counted
    reason: Reason | None
    coordinate: Coordinate | None

class CaveatEntry(_Part):             # one row of the one list (IA F1)
    id: str                           # stable; every id code assigns today keeps its spelling (2.9, item 5)
    kind: str
    cls: CaveatClass | None           # alias "class"; None only for a method-reach kind
    reach: CaveatReach
    state: ReadingState | None        # the non-counted state that raised it, where one did
    reason: Reason | None
    coordinate: Coordinate | None
    arms: list[str]                   # arm ids it touches; empty for all
    readings: list[str]               # "<aspect>/<measure>" addresses it is scoped to; empty for the whole report
    asked: bool                       # whether any of `readings` is on an asked aspect
    deciding: bool                    # whether the answer turns on a reading it names (F8)
    immaterial: bool | None           # True once code proved it (materiality); None where code cannot say
    subsumed_by: str | None           # the caveat id that silences it (F7)

class GateRun(_Part):
    run: str
    status: str
    ended_because: str | None         # the run record's own words; free text, never a 2.3 reason

class Gate(_Part):                    # S0; absent when the campaign can be read
    rule: Literal["setup", "unfinished", "too_small"]
    runs: list[GateRun]
    waiting: int                      # open report-reach caveats silenced below it (F5)

class CallRead(_Part):                # S3's call, as of the freeze
    arm: str                          # arm id
    made_by: Maker | None             # None for a call recorded before the field existed
    model: str | None
    on_behalf_of: str | None
    runs_joined_since: int | None
    identity_changed: bool | None     # the evidence fingerprint differs from the one the call was read against

class FirstScreen(_Part):
    gate: Gate | None                 # S0
    reasons: list[str]                # S2: caveat ids, ranked, at most three
    remainder: int                    # S2: eligible caveats past the cap
    open_report_caveats: int          # S2 on a code-only page; F11 reads it
    called_arms: list[str]            # the arms code's own dominance read calls; the F3 second key
    call: CallRead | None             # S3
```

`ReportInput` gains `caveat_list: list[CaveatEntry]` and `first_screen: FirstScreen | None`. The old `Caveats` part
(seven parallel lists and the `unfinished` sentences) stays filled, for stored readers and the figure keys that cite
it (`{caveats.latency}`). S1 holds no field: it is the verdicts and warnings the input already carries, in their
existing order. S3's remedy is not stored: it is `REMEDY_OF_REASON[reason]`, a closed kind per reason.

#### 2.6.4 The one function and its rules

```python
def first_screen(
    caveats: Sequence[CaveatEntry],
    *,
    gate: Gate | None,
    called_arms: Collection[str],
    call: CallRead | None = None,
) -> FirstScreen
```

Pure; no clock, no store, no text. It reads no campaign kind, no measure name and no prose, and a source-walk test
holds that: the module imports only `reading_states` and `report_input`, and no comparison in it names a string
other than a member of the closed sets.

| Rule | Code-level statement | Test (on constructed lists, never on text) | IA |
|---|---|---|---|
| R1 eligible | A caveat is eligible when `reach == "report"`, `immaterial is not True` and `subsumed_by is None` | One caveat per excluded condition is absent from `reasons` and from `remainder` | F1, F6 |
| R2 gate | With a gate: `reasons == []`, `remainder == 0`, `gate.waiting == len(eligible)` | A three-caveat list under an `unfinished` gate | F5 |
| R3a withheld | For each caveat whose `state == "withheld"` on reading r, every other caveat whose `readings == [r]` gets `subsumed_by` its id | A withheld reading with an instrument caveat scoped only to it; and one scoped to two readings, which survives | F7 |
| R3b set aside | A caveat `set_aside` on r with reason x subsumes a caveat on r whose `reason == x` | "set aside: unchecked" beside "unchecked" yields one reason | F7 |
| R3c A/A | A caveat with reason `identical_arms_separated` on r subsumes every class C caveat on r | One per instrument reason | F7 |
| R3d people over model | A check against a model on criterion c is subsumed when a check against people exists on c | Both orders of insertion | F7 |
| R4 class | Sort key 1: `"ABCDE".index(cls)`; a caveat with no class (an analyst's) sorts after E | A shuffled list of five comes back A to E, across 50 seeded shuffles | F3 |
| R5 touch | Sort key 2: `0 if asked and (not arms or set(arms) & called_arms) else 1` | Two class B caveats, one on a called arm | F3 |
| R6 id | Sort key 3: `id`, ascending | Two equal on keys 1 and 2 | F3 |
| R7 cap | `reasons = ranked[:3]`, `remainder = len(ranked) - 3` floored at 0; `reasons` plus the rest is exactly the eligible set | Lists of 0, 3, 4 and 9 | F4 |
| R8 precision | A class E caveat with `deciding == False` is raised at reach `scoped`, so R1 excludes it. This is the raiser's rule, asserted here on the raiser's output | A cost interval `not_available` off the deciding measure | F8 |
| R9 call | `call` is copied through untouched and never enters `reasons`; with a call, `made_by` is `person`, `model` or None, and None is rendered as not recorded | A call with nine caveats | F9 |
| R10 count | `open_report_caveats == len(eligible)` whether or not a gate stands | With and without a gate | F2, F11 |
| R11 stable | The function is deterministic under any input order | Same 50 shuffles as R4 | F10 |

**The class of each kind** is one table, `CLASS_OF_KIND`, in `analysis/caveats.py`. The IA fixes the first four
rows; the rest are this design's defaults, marked, because the IA names no class for kinds that exist today (2.9,
item 4).

| Class | Kinds |
|---|---|
| A | `set_aside:<reading>`, `withheld:<reading>` |
| B | `late:*`, `confound:*` and `apparatus:*` at report reach, identity unrecorded; default: `short_run:*`, and the code page's "no ordered time" and "request settings not confirmed" |
| C | `judge_uncalibrated`, `judge_partly_uncalibrated`, `judge_unstated` (an instrument unchecked, or checked against a model only) |
| D | `looks`; default: `best_of_n` (a pick among several is the same loosening) |
| E | Precision missing on the deciding measure; default: `unsettled:*`, the implied point-estimate and borderline caveats |
| none | `latency`, `unreported:*`: method reach, never first-screen |

**Raising** (S5a-2). `analysis/caveats.py` holds one raiser per source, each a pure function of the bundle and the
input under construction: a non-counted reading or contrast (one caveat per reading, kind from its state); a check
that is not `met`; late criteria; confounds; short runs; looks. A track adds a raiser, never a page line (IA F1, T1).
`_judge_uncalibrated` in `report/call.py`, which recomputes from tier strings a fact the bundle holds, is deleted.

#### 2.6.5 Every optional field added in the one shape move

**Structure's own** (revision 1, §2.3), unchanged in name:

| On | Field |
|---|---|
| Bundle `FamilyComparison`; input `Contrast` | `baseline`, `baseline_n_cases`; `family` (id); `delta_note` (a closed cause where the delta is not the difference of the means beside it) |
| Input `Factor`; bundle design axis | `ordered: bool` |
| Bundle arm measures; input `Stat` | `spread`, `no_variation` |
| Input `Arm` | `spelling` (`ArmSpelling`), `arm_id`, `cell` (S7) |
| Input `Reading`, `Aspect` | `flip_margin`; `margin_informative` |
| Declaration; input `Meta` | `volume_per_month` per subject |
| Input `ScoreboardRow` | `cost` (the one cost reading, fallback decided once) |

**The IA's** (§2.3, §4, F10):

| On | Field |
|---|---|
| Input `Reading`, `Contrast`, `PairRead`, `Stat` | `state`, `reason`, `coordinate` (`StateOf`) |
| Input `Arm`; bundle arm | `identity: {served_model: IdentityState, measure_definition: {measure: IdentityState}}` |
| Input `ReportInput`; bundle (campaign level) | `caveat_list`, `first_screen` |
| Kernel `CampaignCall` | `made_by`, `model`, `on_behalf_of`, `evidence_fingerprint` (filled by judges A2, #1178; S5a adds them empty only if A2 has not landed) |

**The judges' B0**, with the names changed where the IA rules:

| Judges §9.3 name | Lands as | Why |
|---|---|---|
| `blind_reads[].standing` | `blind_reads[].sample` | "Standing" is an arm's only (IA §2.2) |
| `anchor_standing {state: five strings, ...}` | `anchor_check {result: CriterionState, reason, double_read_items, lean, lean_interval, kappa, kappa_interval, reads, reviewer_family_of}` | T3; and see 2.9, item 6 for the name |
| `prediction_powered` and `reference_corrected` on a contrast | one `corrected {against: "people" | "model", reviewer?, estimate, interval, level, rectifier_difference, rectifier_interval, labelled_cases, identical_cases, read_id, check: CriterionState, state, reason}` | IA §1.3, B1 |
| `decision_eligible` | dropped; derived on read from `check` and `state` | B1 |
| `unavailable_reason` (three places), `void_reason` | dropped; `state: "not_available"` with a 2.3 reason (`replayed` for a void A/A) | V2, C1 |
| Input `Contrast.corrected.kind: "people" | "reference"` | `against: "people" | "model"` | M4 |
| `null_contrasts` (bundle), `Contrast.null` (input) | `aa_checks`, `Contrast.aa` | "Null" is ruled internal (V6); the reader word is A/A check. The IA gives no field name: this one is mine |
| Caveat kinds `judge_short`, `judge_leans`, `people_disagree`, `null_separated` | one kind, `set_aside:<reading>`, with its reason | IA §2.3, E1 |
| Caveat kind `reference_family` | no kind; `anchor_check.reviewer_family_of` | M3, R1 |
| Caveat kinds `looks_undeclared`, `looks_exceeded` | one kind, `looks` | D1 |

Unchanged from judges §9.3: `anchor`, `reviewer` on a tier entry and `Judged`; `blind_reads[]` (its other fields);
`person_preference`, `reference_preference` (projected one step down; in no first-screen value); `judge_noise`
(less its reason string), `judge_share`; `accrual_batches`, `planned_looks`, `looks_rule` on the reading scope;
input `Aspect.looks {planned, taken}`; `HumanPreference`'s five optional fields; `AspectDesign.planned_looks`.

**The four truth fixes'.** Named here from the coordinator's summary; the builder reconciles spellings with each
branch before adding a field, and adds none that a branch already added.

| Fix | Its fact | Absorbed as |
|---|---|---|
| D, merged (`analysis/run_ending.py`, `Caveats.unfinished`) | An unfinished run's ending and the `stopped_short` decision | `FirstScreen.gate` with `GateRun.ended_because`. `Caveats.unfinished` (sentences, in a digest) stays filled and is no longer read by a page that holds a gate |
| C, in review | A split arm's withheld arm-level reading | `Reading.state = "withheld"`, `reason = "split"`, `coordinate = "served_model"`; the arm's cells one step down (`ScoreboardRow.cells`) |
| A, rework | Measure definitions recorded or stamped on a run | Bundle `RunSummary.measure_definitions {measure: {digest, source: "recorded" | "stamped"}}`; input `RunRig` the same; the arm's `identity.measure_definition` |
| B, in review | An absent cost or latency interval, with a reason | `Stat.state = "not_available"`, `reason = "too_few" | "no_variation"` |

#### 2.6.6 Which digests move

| Digest | S5a-1 | S5a-2 + S5a-3 (one merge) | After |
|---|---|---|---|
| A stored `report_input_digest`, snapshot id, frozen writer case | No | No | Never: stored documents are not rebuilt |
| `report_input_digest` of an input built fresh from the test fixtures | **No**: every new field is unset. Done-when asserts each built fixture's digest equals the one pinned before | Yes, once: the typed states, the caveat list and the structure fields fill | Only when a judges filler applies to that fixture (next row) |
| Bundle fingerprint v2 (stored) | No (frozen pre-image, S3) | No | No |
| Bundle fingerprint v3 | No | No | No: it hashes evidence, not the bundle |
| Campaign fingerprint | No in S5a-1 | No: the new campaign-level fields are added to the frame's excluded set until S7's v4 | S7 |
| Recording keys | No | No | No (S4) |

Two rules make "once" true beyond S5a:

- **`first_screen` is left out of `report_input_digest`'s pre-image.** It is a pure function of the rest of the input
  plus the call, and the call can be made after the freeze. Hashing it would re-key a snapshot when someone decides,
  or when the ranking table changes. It is stored on the snapshot so a stored report keeps the slots it showed.
- **A judges filler leaves its field unset in the default state**: one look, no blind read, no replicate, no
  corrected contrast. Then a fixture's digest moves again only when the fixture actually has that fact.

#### 2.6.7 Stored compatibility

- Every field is optional and absent on a stored document; `schema` stays `report-input/1`; the bundle's
  `schema_version` is not bumped (Q5).
- **A stored report degrades on read and is never regenerated.** An input with no `caveat_list` gets one on read from
  its old `Caveats` part by the legacy raisers (the body of today's `report_caveats`, kept as
  `caveats.from_legacy`), with the same ids; classes come from `CLASS_OF_KIND`; states and reasons are absent, which
  reads `counted`. `first_screen` is then computed by the same function. Nothing is written back.
- A stored call with no `made_by` reads as not recorded.
- `Caveats.unfinished`, `ServedModelReading.state` (`pooled`) and every stored reason string stay as written.
- One MIGRATIONS note for the three S5a parts, naming the fields and the degraded reads.

#### 2.6.8 The split

| Part | What | Lines | Moves digests or goldens |
|---|---|---|---|
| **S5a-1** | `reading_states.py`; the typed structures of 2.6.3 and every field of 2.6.5, all unset; `caveats.py` with the class table, subsumption and `from_legacy`; `first_screen.py` with R1 to R11; the builder moved down, presentation imports cut; the parity guard, with its allowance listing today's gaps | About 2,100 changed, 1,286 moved | Nothing. No page changes |
| S5a-2 | The answer read down (`analysis/answer.py`, `analysis/criteria.py`); raisers fill the caveat list with class, reach, state and reason; reading and contrast states filled from the truth fixes' facts; the builder attaches `first_screen`; `report_caveats`, `every_caveat`, `implied_caveats` and `caveat_materiality` become thin reads of the list | About 2,300 changed, 800 moved | Built-fixture input digests. No page changes: pages still call the thin reads |
| S5a-3 | The structure fields filled: baseline, spread, `ordered`, spelling (and `reader_arm_name` deleted), family id and note, flip margin, volume, the one cost reading | About 1,500 | Built-fixture input digests, in the same merge as S5a-2 |

S5a-1 is the minimal chunk that unblocks the judges: after it, R1, B1, B2, C1 and D1 fill their fields and add their
raisers with no page work. E1 waits for S5a-2, because it edits the answer read where S5a-2 leaves it. S5a-2 and
S5a-3 have disjoint files (`report/call.py`, `report/gates.py`, `caveats.py`, `answer.py` against
`bundle/comparisons.py`, `report_margins.py`, `report/words.py`, `report/keys/names.py`, `kernel/declaration.py`);
both edit `report_input_build.py`, in separate methods of `_Builder`. The coordinator merges them together and
regenerates the built fixtures once.

### 2.7 S5b: the single rendering

S5b is structural work inside the presentation freeze; the owner's exception is asked for exactly this (Q7).

**One renderer.** A new `report/first_screen.py` (about 250 lines): `first_screen_blocks(value, caveats, labels) ->
list[ReportBlock]`, called by the written page (`doc_page/build.py`), the campaign code page (`campaign/page.py`) and
the aspect page (`build.py`). It returns the S0 block or the S2 block, and the S3 lines. Nothing else in `report/`
may build a first-screen qualification, held by a test: the only importers of `FirstScreen` under `report/` are that
module and the three page builders, and the page builders pass its blocks through unread.

**What it replaces** (IA §6, subtraction 2):

| Route today | Where | Becomes |
|---|---|---|
| The code page's row of report-wide caveats, computed from the bundles, apart from the input's list | `campaign/provenance.py` `caveat_blocks` (called from `campaign/page.py:233` and the aspect head) | S2. The kinds only this route raises join the one list in S5a-2 |
| The written page's Confidence caveats, unbounded, "checked" ones included | `doc_page/build.py` (`_code_label`, the label and landing code, about 200 lines) | S2: the form, then at most three reasons and the count. The full list, checked lines included, prints once in the appendix |
| The analyst's `flag` line above the decision | `ReportDoc.flag` | A stored flag prints as an analyst caveat (id `analyst:flag`) in the one list, ranked after class E. The field stays readable; the skill is not edited in S5b (2.9, item 5) |
| A split arm's line among the warnings | `campaign/warnings.py` (fix C) | A class A reason. A warning is a declared line crossed, and a split is not one |
| The unfinished-run opener, built per surface | `verdict.py`, `build.py`, `campaign/page.py`, `doc_page/build.py` | S0 from `FirstScreen.gate`. The quick print and the CLI (`summary.py`, `quick/`) keep reading `stopped_short`, the same decision, since they print one run and no page |
| Scoped caveat lines under a scoreboard | `verdict.caveat_lines` | Unchanged: scoped reach is already one step down |

**Not re-worded, not re-styled.** S5b reuses every existing string through its existing function: caveat labels
(`CAVEAT_LABELS`, `_code_label`), the checked lines (`_CHECKED`), the unfinished sentences
(`StoppedShort.sentences`, `covers`, `pooled`), the confidence form's words, the "runs joined since" line. Blocks keep
their types and CSS classes. It adds strings only where the contract has no words today, in one module
(`report/first_screen_words.py`), each listed in the PR for the owner:

1. the remainder line (a count and a link);
2. the code-only trust statement (a count of open caveats);
3. the call's maker ("by a person", "by a model (`<id>`)", not recorded), and "the evidence's identity changed since";
4. one remedy phrase per reason that can reach S2 and has none today (at most 15; `unfinished` already has one).

**Printed output that moves.**

| Page | Change |
|---|---|
| Written, with more than three open report-wide caveats, or any checked one | The Confidence block shrinks to three reasons and a count; the rest and the checked lines move to the appendix. Kairo's seven bullets become three and a line |
| Written, with a flag | The flag's sentence moves from above the decision into the reasons, by rank |
| Code, campaign and aspect | The caveat row is replaced by the S2 block: a count, up to three reasons in rank order (today: every one, in source order) |
| Any page with a recorded call | The call line states its maker |
| Any page over an unfinished run | Same sentences, from one value; the count of qualifications waiting is new |
| A clean campaign | Code page gains the "zero open caveats" statement; nothing else moves |

**Goldens.** The four hot-path oracle pages (`toyhost`, `courierhost`, `scaled`, `scaled_two_aspects`) and
`toyhost_doc`; `test_report_golden_g2.py`; the built pages under `tests/fixtures/report_docs` (`prototype4`,
`prototype5`, 14 pages). All regenerated deliberately, the diff confined to the first screen and the appendix's
caveat list. **Recordings:** none, because S4 keys the judge on the report document, not the page.

**Parity (F10).** The guard of §2.2 gains two asserts per fixture: both pages hold the same `gate`, and the same
`reasons` in the same order. The written page adds only the analyst's form and the analyst's own caveats.

**Done when.** The importer test above passes; the parity asserts pass; every stored `report_docs` fixture and all 14
recorded rung-5 reports still pass `doc_page/checks.py` with no caveat id unassigned; F11 is read from
`open_report_caveats`.

### 2.8 What S5a and S5b subtract

| Chunk | Subtracted | Count |
|---|---|---|
| S5a | Caveat computations | 3 → 1 (`report_caveats` over the input, `caveat_blocks` over the bundles, `_judge_uncalibrated` over tier strings) |
| S5a | Public functions that answer "which caveats" | 4 → 1 list (`report_caveats`, `implied_caveats`, `every_caveat`, `caveat_materiality` become reads of it; removed after S5b) |
| S5a | Caveat types | 3 → 1 (`Caveat`, `ImpliedCaveat`, `Materiality` → `CaveatEntry`) |
| S5a | Closed vocabularies for "can this be used" that would otherwise be added | 6 → 0: the anchor's five standings, `void_reason`, three `unavailable_reason` strings, `decision_eligible`, a served-model-only split flag, a definition state |
| S5a | Judges fields | Two corrected fields → one; seven reason and eligibility members dropped (`unavailable_reason` in three shapes, `void_reason`, `decision_eligible` in two, the five-string `state`) |
| S5a | Judges caveat kinds | 7 → 2 |
| S5a | Identity state vocabularies in new code | `ServedModelState`, `DefinitionState`, "stands alone" → one `IdentityState` (stored `pooled` stays readable) |
| S5a | `report/call.py` | 1,220 → about 350 lines (the rest moves down, not away); `report/gates.py` 448 → about 250 |
| S5a | `reader_arm_name` (text matcher) and `input_build`'s three presentation imports | About 70 lines; 3 edges |
| S5a | Stored sentences read by a page | `Caveats.unfinished` no longer read where a gate exists |
| S5b | First-screen qualification routes | 5 → 1 |
| S5b | First-screen reasons on the worst real page | 7 → 3 and a count |
| S5b | First-screen layouts by campaign kind for qualifications | 3 → 1 slot order |
| S5b | `doc_page/build.py` caveat labelling and landing, `provenance.caveat_blocks`, the split line in `warnings.py`, four opener sites | About 450 lines net, after the 250-line renderer |
| S5b | Public names | `caveat_blocks` and the four functions above leave the `analysis` root's provisional exports; no declared name (503) moves |

### 2.9 Where the constraint document does not match the code

| # | The IA says | The code is | Line taken |
|---|---|---|---|
| 1 | `report_caveats` moves down with its ranking as one pure function (its §7 flags this as unverified) | `report_caveats` calls `_outcome` (dominance), which calls `gate_check` in `report/gates.py`, which resolves figure keys and takes a `PageForms`. It cannot move alone | S5a-2 splits a criterion's status (arithmetic, already in `report_margins.py`) from the line that words it, and replaces `resolve("cost_per_1k.<arm>")` with the one cost reading. This is why S5a is three parts |
| 2 | F3 ranks by whether a reason touches "the decided or recommended arm"; F10 puts the value in the report input | The input is frozen before the analyst recommends, and the call is recorded later still. The call is on the campaign document, not in the bundle or the input, and only the code page prints it | The second key is the arms code's own dominance read calls (`called_arms`), as `report_caveats` uses today for `unsettled`. The input carries the call as of the freeze; a renderer holding a later call passes it to the same function, which replaces S3's call and never re-ranks. `first_screen` is outside the digest |
| 3 | T4: identity is `one`, `split`, `unrecorded`, "the values `ServedModelState` nearly has" | `ServedModelState` is `one`, `pooled`, `unrecorded`, stored on bundles | New fields use `IdentityState`. The stored `pooled` stays and maps to `split` on read |
| 4 | F3 gives five classes by example | Eight caveat kinds exist with no class named: `short_run`, `unsettled`, `apparatus`, `latency`, `unreported`, the implied `best_of_n`, point-estimate and borderline caveats, the analyst's own (`analyst:*`), and two the code page alone raises (no ordered time, request settings not confirmed) | Defaults in 2.6.4's table. An analyst's caveat ranks after E unless the IA's author rules otherwise (Q8) |
| 5 | Kinds merge into `set_aside:<reading>`; "flag" merges into caveat | Caveat ids are identity: a stored report document and every recorded writer submission name them, and `doc_page/checks.py` refuses an id code does not assign. `flag` is written because the skill says so, and the skill's digest keys the writer recordings ($6.90 to record again) | No existing id is renamed; `judge_uncalibrated` keeps its id and gains class C, and F7 handles its overlap with a new `set_aside`. S5b renders a stored flag as an analyst caveat and leaves the skill; the skill edit rides the next paid batch |
| 6 | §5.1 writes `anchor_standing.state`; §2.2 rules that "standing" names only an arm's standing | The field does not exist yet | Named `anchor_check`, with `result` |
| 7 | The gate's reason is `unfinished`; a run's ending has a "reason" | `RunEnding.reason` is the run record's free text | The typed field is `ended_because`, so "reason" means one thing in the input |

## 3. Identity over inputs only

### 3.1 Fingerprint v3 (#1062)

**Constructed, not subtracted.** `evidence_digest` today starts from the whole bundle and removes presentation fields
one family at a time (`_NOT_EVIDENCE`, `_NOT_ARM_EVIDENCE`, `_TRIAL_PRESENTATION_FIELDS`, five more). Each of the 15
identity bumps was one more subtraction. v3 lists what it hashes:

| In the v3 pre-image | Source |
|---|---|
| Each member run: id, variant key, context key, identity version, template revision digest, completeness record | Stored run |
| Each member result: id and its stored evaluative values (outcome, scores, goal states, judge evidence, exclusion cause), as one digest per result, the list sorted by id | Stored result, hashed as stored JSON |
| The campaign's declaration: aspects, control, design by content hash, margins, floors, bars | Stored campaign, presentation fields excluded by the existing `*_PRESENTATION_FIELDS` sets |
| `host_declarations_digest` | Exists |
| The case set's case ids; judge profile ids and prior insight ids the bundle consulted | Stored |

Out, by construction: every computed statistic, every name and label, pricing (ruled 2026-10-28 and #1011), reason
text, `schema_version`, `package_version`. Stored JSON parses to the same doubles everywhere, so v3 is
platform-independent and no formula can move it.

**Where.** A new `bundle/evidence.py` builds an `EvidenceManifest` (the digests and counts above, a few hundred
bytes) from the reads `assemble_context_bundle` already makes, and the bundle carries it as an optional field.
`fingerprint()` returns `v3:` over the manifest.

**Migration.**

- `fingerprint(like=<stored prefix>)`: a record holding `v2:` is compared the v2 way; one holding `v3:` the v3 way.
  The inspection's `mismatch_cause` compares like with like and never reports a version difference as moved
  evidence.
- **v2's pre-image is frozen** to an explicit list of today's bundle fields, so the additive fields of S5 never enter
  it. This is why v3 lands **before** the shape change, reversing the order decided on 2026-11-14. That order assumed
  v3 would hash the settled shape; a v3 over inputs does not hash the shape at all, and without the freeze S5 would
  move every stored v2 fingerprint. Owner question Q3.
- `evals/hedges-j-exact` lands in the same PR, as ruled. For stored v2 records to stay unmoved, the v2 path keeps the
  old correction as a private `_hedges_j_v2` (about ten lines) used only to build the v2 pre-image.
- Reads never backfill: a v2 record is never rewritten to v3 on read. A regenerated analysis stores v3.
- v2 stays platform-dependent on `scaled_two_aspects` for stored v2 records. The per-platform golden goes away
  because the oracle pins v3.
- Naming trap: the campaign bundle's own fingerprint prefix is already `v3:` (`_CAMPAIGN_FINGERPRINT_PREFIX`). Its
  next version (over its aspects' v3 fingerprints and the declaration, no readings and no reason text) is `v4:`.
- A Discodon handover note: reporter cases and `reporter_case_freeze` hold v2 and keep matching.

### 3.2 Recordings keyed on structured inputs

There are two recorders, and only one has the problem.

| Recorder | Keyed by | Problem? |
|---|---|---|
| `testing/recorded_turns.py` (writer sessions) | Path: arm, skill digest, model, effort, revisions, snapshot id. Replies only | No. A wording change in a tool's text moves nothing. It moves when a case is frozen again or a skill file is edited, which are input changes |
| `testing/recorded_completions.py` (every `Model` call) | sha256 of `system`, `user`, `response_format`, model, temperature, settings | Only where the prompt embeds package-rendered text |

For a candidate's reply or a rubric judge over a candidate's output, the prompt **is** the input, and keying on it is
right. The defect is narrow: the writer-eval judge (`harness/writer_eval.py`, `ops/report_writer/writer_pairwise.py`,
`rubric.py`) puts rendered report pages in its prompt, so `examples/5-writer/recordings/judge.jsonl` (112 calls)
re-keys on any wording change. #1162 made that loud; it still costs a re-record each time.

Design: a call may carry an **input key** that replaces the prompt in the recording key.

- `RecordedCompletions` accepts an optional `input_key: Mapping[str, JSON]` per call, bound the way the cell identity
  already is (`bind_cell_identity` / `current_cell_identity` in `schema`). When present, the recording key is the
  digest of `{input_key, response_format, model, temperature, settings}`, and the prompt is stored on the line for
  inspection only. When absent, behaviour is exactly today's. Recording format 3 → 4, additive; format 3 files still
  read.
- The writer-eval judge binds, for each call: the rubric's digest, the dimension, the judge template's digest, and
  for each page shown its **report document digest** and snapshot id (the stored document: keys and prose as the
  writer submitted them, before any figure is rendered), and the order shown.
- The judge template's own text stays in the key through its digest: changing what the judge is asked is an input
  change. The package's rendering of the page is not.
- A package-authored prompt that embeds no rendered page keeps prompt keying. No other caller changes.

**Carry-over without paying.** A one-off script re-keys `judge.jsonl` in place: it replays rung 5 once under format 3
(every prompt still matches at the commit before any wording moves), computes each call's input key, and rewrites the
line under the new key. Zero paid calls. If the owner prefers a clean record instead, the re-record is the judge
alone: 112 calls on `google/gemini-3.8-flash`, $0.28 at the recorded prices; the writer's turns ($6.90) replay free
either way. The seven other example recordings (`1-triage` to `4-booking`, 808 calls) embed no rendered page and do
not change.

**What must be re-recorded in this track.** Nothing, if S4 lands before S5. If S5 landed first, #1125 alone changes
the written page's text and forces the judge re-record ($0.28) plus re-pinning seven rung-5 tests, which is what
happened on 2026-11-14.

**Honest limit.** Under input keying, a replayed judgment was made on the page as it was rendered when recorded. A
replay test therefore checks the plumbing and the arithmetic, not the judge's opinion of today's rendering. That is
what a replay test is for; the paid lane is where today's rendering is judged.

### 3.3 Labels separate from ids (#1024 item 3)

Today an arm's printed label is its id: `campaign_bundle` validates `spelling.label() == arm.label`, the bundle keys
readings by label, and `style.arm_name` parses a display string (`", "` and `=` splitting) to recover levers. So the
ruled rename (`2` → `Fanout 2`) is a re-key with a migration note.

Design:

- An arm's **id** is its variant key (it already has one; a declared arm also has its declared id). Every by-arm map
  in the bundle and the input is keyed by id.
- An arm's **label** is a presentation field computed on read from its `ArmSpelling` and the levers' declared reader
  names. It is stored for old readers, excluded from every fingerprint (it joins `ARM_PRESENTATION_FIELDS`), and
  never parsed.
- `style.py` splits: `analysis/naming.py` keeps what mints or compares names as data (numbers); `style.py` keeps the
  glossary, `humanize` and the page's wording rules (presentation, and it leaves the numbers side's imports).
- Migration: additive. New bundles carry `arm_id` beside `label` in each by-arm row; a stored bundle without it reads
  `label` as the id (they were equal when it was written). #1024.3's rename then moves labels only. Under v3 the
  fingerprint holds neither, so nothing re-keys and the MIGRATIONS note becomes "labels changed; no stored key
  moved".

Sizing risk: by-arm maps keyed by label run through `report_input.py` (`per_arm`, `scoreboard`), 13 `keys/` modules
and the analyst's tools. The input keeps label keys in this track (figure keys cite arms by label, and frozen reports
hold those keys); the id rides beside. So this track delivers "a label change moves no identity", not "labels are
gone from keys".

Revision 2 (IA M1, §5.1): an arm id can name **one cell** of a split arm. `Arm` gains an optional `cell` (the
coordinate and its value, by content hash), and a withheld arm's `ScoreboardRow.cells` lists its parts by that id, so
the figures beneath a `withheld` reading are addressable one step down and by a figure key.

### 3.4 Stored reasons as typed values (#1136)

`FloorStatus` already carries its "why" as facts and words it on the page. `AspectReading.reason` and the bar and
pair reasons (`campaign_readings.py`: `_row_reason`, `_pair_reason`, `_bar_reason`) are still stored English, and
the campaign fingerprint reads them.

Design: each gets an optional typed `why` beside the string: a closed `kind` (for a row: `too_few_cases`,
`lost_trials`, `interval_spans_both`, `within_margin`, `outside_margin`, `gain_shown` and the rest of today's
branches, one per branch of `_row_reason`) plus the figures it rests on, as `SaidFigure` already types them. The page
words `why`. `reason` stays filled for stored readers and old pages, and is out of the v3/v4 pre-image. A stored
reading without `why` prints its stored `reason` byte for byte (degrade on read). `Said` shrinks to what a page
needs to word a `why`.

No code inspects the wording, so this adds no prose check.

Revision 2 (IA V2): `why` is two enums, never mixed. Where the reading is not counted, `why` is absent and the
reading's `state` and `reason` (the closed set in `analysis/reading_states.py`) say why. Where it is counted, `why.kind`
is a verdict-explaining kind (`interval_spans_both`, `within_margin`, `outside_margin`, `gain_shown`). So
`too_few_cases` and `lost_trials` leave the list above: they are `not_available`/`too_few` and a `withheld` or
`set_aside` state. No new spelling of "undecided" is added (V3).

### 3.5 The invariant, as a test

`packages/evals/tests/test_wording_moves_no_identity.py`:

- Run rung 5's replay and the four hot-path fixtures twice in one process: once as is, once with a **wording
  perturbation** installed through the front door. The perturbation is a host vocabulary and name overlay that
  changes every reader-facing name (measure names, lever names, arm labels, stratum names) plus a fixture-local
  subclass of the page serializer that appends a marker word to every text block. No monkeypatching of private names.
- Assert equal across the two runs: every bundle fingerprint (v3), the campaign fingerprint, every
  `report_input_digest` computed from stored evidence, every recording key asked, every replayed number (each arm's
  measures, the preference share and its interval), every gate verdict token.
- Assert the rendered pages differ, so the test is not vacuous.

It will not pass until S3, S4 and S7 land; each of those chunks turns on the part it makes true.

## 4. Tests at the boundary

### 4.1 Which wording asserts are contracts

| Kind | Rule | Rough share of the 2,086 |
|---|---|---|
| **Host-matched text**: a refusal's message and remedy, a CLI exit line, a constant the package exports for a host to match (`NO_LONGER_READS`, `JUDGE_WORDING_CHANGE_SENTENCE`, `SEED_REMEDY`), error messages a host's logs alert on | A contract. Keeps its literal. Where the literal is an exported constant, the assert names the constant so wording changes in one place | About a quarter |
| **Golden pages**: the hot-path oracle, `test_report_golden_g2.py`, the `report_docs` fixtures | A contract on the whole page, changed only by deliberate regeneration. Stays as is: one golden catches what 50 substring asserts do | Small count, large coverage |
| **A sentence used to prove a value**: `assert "85.3%" in page`, `assert "no variation" in html`, `assert "too few cases" in reason` | Re-pin to the typed value at the boundary: the `Printed` value and role from the forms ledger, the block's typed cell (`TableCell.value`, `value_text`), the reading's `why.kind`, the input field. The behaviour asserted is unchanged; the English is no longer the handle | About half |
| **Prose style pins**: asserting a phrase the package authors but no host matches | Stays until touched; when a wording change breaks one, it is re-pinned to the typed value, never deleted | The rest |

No assert is weakened or removed. A re-pin replaces "this string appears" with "this value, in this role, in this
block", which is stricter.

To make re-pinning cheap, presentation exposes what production already needs: the forms ledger, typed blocks
(`ChartBlock`, `TableBlock` and their cells), `why` values. No accessor is added for tests alone.

### 4.2 Source-walking tests

47 test files import `ast`. They fall into three groups:

| Group | Examples | Disposition |
|---|---|---|
| Structural gates over the whole tree by rule | `test_report_layer_direction`, `test_package_matrix`, `test_one_owner_of_number_forms`, `test_public_surface_is_closed`, `test_no_process_global_state` | Keep. They name sides and rules, not files, so a move does not break them. Each takes its side definitions from `tests/package_placement.py` so there is one answer to "where does this module live" |
| Gates that name specific modules or functions | `test_*_exports.py` (five files), `test_significance_single_renderer`, `test_one_interval_join`, `test_writer_prompt_retired`, `test_retired_vocabulary` | Re-point when their subject moves: the module path is read from one table in `package_placement.py` (`MODULE_HOMES`), so a move edits one line. Their assertions do not change |
| Tests that parse source to check behaviour they could observe | To be listed per chunk by walking each test | Re-pin to behaviour when the chunk touches the subject. Each is walked individually; none is dropped |

### 4.3 Making moves cheap

- **Stable import roots for tests.** A test imports from a public root (`threetears.evals.analysis`,
  `threetears.evals.kernel`, ...) or from a subpackage's `__init__` (`analysis.report`, `analysis.bundle`,
  `analysis.report.keys`). Subpackage inits export what their tests and siblings use. This is not "API for a test":
  each name is already imported by production code in another module, which is what makes it a module-level export.
- **A move leaves a shim for one minor**, exactly as `report/figure_keys.py` did: the old path re-exports. Tests are
  re-pointed in a separate, mechanical commit, so a move PR's diff shows the move alone.
- **The #928 rename pass (693 cross-module private imports, 393 in `report/keys/`) runs after the moves**, in S8. A
  name is promoted once, at its final home. Doing it first would rename 393 names in a package S6 and S9 then move.
- A new gate, `test_tests_import_from_package_roots.py`, with a ratchet: the count of below-root import statements
  per area may only fall. This one is a long-lived ratchet (3,052 today); it does not reach zero in the track and is
  not meant to. Owner question Q6.

### 4.4 Test files each chunk touches (estimate)

| Chunk | Test files edited | New | Basis |
|---|---|---|---|
| S1 | 1 | 0 | `test_report_layer_direction.py` |
| S2 | about 25 | 1 | 6 name `agreement`, 2 `significance`, 22 `campaign_readings`, plus `test_one_owner_of_number_forms.py` and the `decision` importers; most survive on shims |
| S3 | about 55 | 2 | 52 files mention a fingerprint; 13 pin a digest literal |
| S4 | about 12 | 2 | The recorder's own tests, rung-5 tests, `test_harness_writer_eval.py` |
| S5 | about 60 | 3 | 28 name `input_build`, 90 import `report_input` (most unaffected by additive fields), 69 name `forms`; goldens regenerate |
| S6 | 0 to 84 | 1 | 84 files import `analyst` below the root; zero if shims stay, all 84 in the mechanical re-point commit |
| S7 | about 40 | 2 | Arm label pins and reason-string asserts; 25 import `style` |
| S8 | about 120 | 0 | The rename pass; tests import no private name today (0 found), so edits follow promoted names only |
| S9 | 0 to 230 | 1 | Same shim choice as S6 |

## 5. What to stop carrying (owner questions; nothing is built without a ruling)

| # | Candidate | Size | Recommendation | Saves | Risk |
|---|---|---|---|---|---|
| Q-A | Number-forms machinery: `forms.py` 3,428 + `number_forms/` 2,072; `PageForms` is a 2,963-line class with about 120 methods; its gate is 2,475 lines with a 119-entry permanent list | 5,500 src + 2,475 test | **Simplify**, as 2.4: one `MeasureForms` table from the input; `PageForms` a printer over it. Then split the printer by surface (table cell, chart, key, off-page) into four modules | About 800 src lines now (constructors, the code-page duplicate); the gate's permanent list shrinks | Code-page goldens move where the two implementations disagreed |
| Q-B | Figure-key grammar, `report/keys/` | 8,795 | **Keep.** It is the writer's whole interface and the "code renders every number" rule. Simplify only by attrition: each fact S5 moves into the input removes a computing resolver | 0 now; perhaps 1,500 over time | Any rewrite stales stored reports' keys |
| Q-C | Hand-rolled static SVG charts: `chart_svg.py` 2,103 + `charts.py` 2,007 + `svg.py` 148, beside `kit.js` 687 that draws the same charts in the browser | 4,258 | **Ask.** Default: keep the static SVG only for the Markdown page and email-safe output; if the owner confirms HTML with `kit.js` is the only page hosts show, retire `chart_svg.py` and render Markdown charts as tables | Up to 2,100 | A host embedding static SVG loses pictures. Needs a host census first |
| Q-D | The provisional tier: 1,262 names exported and promised nothing, against 503 declared | n/a | **Simplify.** Stop re-exporting provisional names from public roots unless a declared signature reaches them; subpackage inits (4.3) carry the rest | Root `__init__` files shrink (the `analysis` root is 863 lines, 409 names); "is this public" has one answer | Discodon imports some provisional names; it accepted on 2026-11-06 that they may move. Needs its import list first |
| Q-E | `:ptype` / `:rtype` lines restating annotations | 9,924 lines by grep (the audit counted 7,335) | **Retire going forward, strip mechanically once.** mypy strict already holds every annotation. One scripted commit, no logic | About 7,000 to 9,900 lines; every later diff is smaller | Zero behaviour risk; a one-time conflict with every open branch, so it lands at a quiet point (after #1163 merges) |
| Q-F | History narration in source ("until 2026-11-14 the rule read...", "owner ruling, ...", "retired") | 525 lines match a narrow grep; more in prose | **Simplify.** A docstring says what the code does now; history goes to `docs/design.md` rationale or CHANGELOG. Do it per module as chunks touch them, never as a sweep | Several hundred lines; fewer stale docstrings | Low. A ruling's date in a comment has saved rework before, so keep one line citing the ruling where a rule looks arbitrary |
| Q-G | Retired-writer fixtures and prototypes: `tests/fixtures/report_docs` is 155 files and 15 MB, with `prototype` through `prototype5`; `tests/fixtures/retired_writers` | 15 MB | **Ask.** Default: keep what a live test reads; list the rest for the owner. Walk each test, never a directory | Repository weight | A fixture may be the only witness of a degrade-on-read path |

## 6. Giant units in the engine

`execute_run` (834 lines) and `_run_one_result` (544) in `run/runner.py`, `launch_run` (442), `engine_actions`
(1,297, `actions/engine.py`), `EvalStorage` (2,874, `kernel/storage.py`), `EvalRun` (981 lines, 81 fields).

**Later, not in this track.** Reasons:

- None blocks the boundary. `run/`, `kernel/`, `schema/` and `storage/` import nothing from `analysis` (zero edges),
  and the boundary work touches none of these files.
- They are host contract (`execute_run`, `launch_run`, `EvalStorage`, `EvalRun` are all declared names). Decomposing
  them risks the pinned host; this track's promise is that the host is never broken mid-track.
- Package B, I and I2 of the burn-down just changed `runner.py`, `launch.py` and `EvalRun`. They deserve a quiet
  period and their own design.

One exception to watch, not to act on: `EvalStorage` gains nothing in this track. If S3's evidence manifest needs a
read the store lacks, it uses what `assemble_context_bundle` already reads, and no method is added.

The presentation side has its own giant, `PageForms`; it is covered by Q-A because it sits on the boundary.

## 7. Build chunks

Each chunk is one PR-sized unit folded into the evals area PR line, no version bump. "Lines" is changed source lines,
tests apart. The production host imports only from public roots, so a module move cannot break it; the shims in S6
and S9 exist for tests, and may be dropped in the same PR if the mechanical test re-point rides along.

| Chunk | What | Lines | Files (disjoint from parallel chunks) | Needs | State |
|---|---|---|---|---|---|
| S1 | The direction test with its ten-pair ratchet | 0 src, about 250 test | `tests/test_report_layer_direction.py` | — | **Merged** |
| S2 | Ratchet to zero | about 2,400 (1,600 moved) | `kernel/numbers.py`, the 9 modules of §1.3, `report/decision.py`, `report/said.py`, `report/forms.py` | S1 | Building |
| S3 | Fingerprint v3 + `hedges-j-exact` + v2 frozen pre-image | about 1,200 | `bundle/schema.py`, `bundle/evidence.py`, `bundle/assemble.py`, `schema/hashing.py`, `stats.py`, `campaign_bundle.py` | — | Building |
| S4 | Input-keyed recordings; `judge.jsonl` re-keyed | about 700 | `testing/recorded_completions.py`, `schema` (the binding), `harness/writer_eval.py`, `ops/report_writer/writer_pairwise.py`, `rubric.py` | — | Can start now |
| **S5a-1** | The shape, the closed sets, the first-screen function (§2.6.8) | about 2,100 + 1,286 moved | `analysis/reading_states.py`, `caveats.py`, `first_screen.py` (new); `report_input.py`; `report/input_build.py` → `analysis/report_input_build.py`; `bundle/schema.py` (fields only); `kernel/declaration.py` (fields only) | S2 to start; S3 to merge | **First after S2** |
| S5a-2 | The answer read down; caveats raised and ranked; `first_screen` attached | about 2,300 + 800 moved | `report/call.py`, `report/gates.py`, `analysis/answer.py`, `analysis/criteria.py`, `analysis/caveats.py`, `report_input_build.py` (the caveat and state methods), `campaign/provenance.py` (its kinds' raisers move down) | S5a-1 | |
| S5a-3 | The structure fields filled | about 1,500 | `bundle/comparisons.py`, `bundle/measures.py`, `report_margins.py`, `report/words.py`, `report/keys/names.py`, `kernel/declaration.py` (volume), `report_input_build.py` (the arm, contrast and measure methods) | S5a-1 | Parallel with S5a-2; one merge |
| S5b | The single rendering of the four slots (§2.7), and the written page reads the S5a-3 fields (#1125, #1126, #1129, #1067) | about 1,900 | New `report/first_screen.py`, `first_screen_words.py`; `doc_page/build.py`, `doc_page/checks.py`, `campaign/page.py`, `campaign/provenance.py`, `campaign/warnings.py`, `report/build.py`, `report/verdict.py` (openers); `parts/charting.py`, `parts/full_results.py`, `parts/tables.py` | S5a-2, S5a-3, S4, Q7 | |
| S5c | `MeasureForms` at the boundary; `PageForms` one constructor; one representation of an absent interval | about 2,500 | `report/forms.py`, `report/number_forms/*`, `report/aspect/*`, `campaign/scorecard.py`, `campaign/aggregate.py` | S5a-3; after S5b where they share `campaign/` | |
| S6 | Break `report` ⇄ `analyst` | about 7,100 moved, under 300 edited | `analyst/*`, `report/rows.py`, `block_spec.py`, `parts/resolve_*.py`, `doc_page/{facts,markdown,text,store}.py` | S5b | |
| S7 | Arm ids (naming one cell where an arm is split) beside labels; `style.py` split; typed `why`; campaign fingerprint v4 | about 2,600 | `campaign_readings.py`, `campaigns.py`, `arms.py`, `style.py`, `naming.py`, `campaign_bundle.py`, `report/aspect/scoreboard.py` | S3, S5a-2 | |
| S8 | #928 module-scope privacy | about 3,000, in 3 PRs | Nearly every evals module | S6 | |
| S9 | Surfaces move under `report/`; `reporting.py` → `projection.py` | about 9,500 moved | The seven surface modules, `reporting.py` | S8 | |

### Order and parallelism

```
S1 ✔ ─► S2 ─┐
S3 ─────────┼─► S5a-1 ─┬─► S5a-2 ─┐                    ┌─► S5c ─► S6 ─► S8 ─► S9
            │          └─► S5a-3 ─┴─(one merge)─► S5b ─┘
S4 ─────────────────────────────────────────────────┘   S7 (after S5a-2; beside S5b)

judges:  S5a-1 ─► R1, B1, B2, C1, D1 (fields and raisers, no page)      S5a-2 ─► E1 ─► E2
```

- Running now, in parallel: S2, S3, S4.
- S5a-1 is the first chunk a builder starts when S2 merges. It moves no digest and no page, so it can merge the day
  it passes.
- When S5a-1 merges, six builders can run at once: S5a-2, S5a-3, and the judges' R1, B1/B2, C1, D1. Files are
  disjoint except `analysis/caveats.py`, where each judges chunk adds one raiser function and one row of
  `CLASS_OF_KIND`: append-only, so merges are trivial.
- S5b waits for the S5a-2 and S5a-3 merge and for S4. S7 runs beside it (numbers files and the scoreboard only).
- E1 follows S5a-2 and raises caveats only (IA §5.1). Any first-screen line from R1, C1, D1, E1 or #1178 waits for
  S5b; a merge that adds one outside the caveat list is refused.
- S6, S8 and S9 stay serial and mechanical.

### Done-when, goldens, recordings, stored schema

| Chunk | Done when (a test holds it) | Goldens moved | Recordings moved | Stored schema |
|---|---|---|---|---|
| S2 | The known-pairs set is deleted and the direction tests pass with no exemption; the one-owner gate passes re-pointed, its ceiling not raised | None | None | None |
| S3 | v3 pinned for the four oracle fixtures from one file on both platforms; swapping `hedges_j` moves no v3; a stored v2 fixture still matches through `like=` | The per-platform `scaled_two_aspects` golden is deleted; oracle fingerprints re-pinned once | None | Optional `evidence` manifest; `v3:` strings |
| S4 | Rung 5 replays with zero misses after a planted wording change; a format-3 file still replays | None | `judge.jsonl` re-keyed, 112 lines, $0 | Recording format 4, reads 3 |
| S5a-1 | R1 to R11 each pass; the closed-set tests of 2.6.2 pass; **every built fixture's `report_input_digest` and every bundle fingerprint equals its value before the chunk**; a bundle and an input with no new field set dump byte-identically; the builder imports nothing from `report/`; the parity guard runs with its allowance listing today's gaps | None | None | Additive optional fields, all unset; MIGRATIONS note |
| S5a-2 | `caveats.from_legacy(old input)` returns exactly the ids `report_caveats` returned before, for every fixture; a fixture per reason of IA §2.3 raises one caveat with its class, reach, state and reason; the IA's three applied campaigns (§3.3) produce the stated slot contents as typed values; `analysis/answer.py` and `criteria.py` import nothing from `report/`; all 14 recorded rung-5 reports and every `report_docs` fixture still pass their checks | None on pages. Built `report_inputs` fixtures regenerate, once, with S5a-3 | None | Fields fill |
| S5a-3 | The parity guard's allowance lists only what S5b renders; `reader_arm_name` is gone and `page_arm_names` parses no text; every frozen input still verifies its digest | As S5a-2, same merge | None | Fields fill |
| S5b | §2.7's done-when; r2's reasoning budget draws its line and an unordered factor none; a two-arm no-variation fixture prints it on Full results; the parity guard passes with an empty allowance | Four hot-path oracle pages and `toyhost_doc`; G2; the built `report_docs` pages. Diff confined to the first screen, the appendix's caveat list, a change chart's zero label, no-variation cells, configuration lines | None | None |
| S5c | `PageForms` has one constructor; one text per `(measure, role, value)` across both pages of each oracle fixture; an absent interval prints through one path, from `Stat.state` and `reason` | Oracle pages where the two implementations disagreed, each listed | None | None |
| S6 | `test_the_report_does_not_import_the_analyst` passes | None | None | None |
| S7 | `test_wording_moves_no_identity` passes whole; #1024.3's rename moves no fingerprint; a split arm's cells are addressable by id | Pages printing a one-knob derived arm (`2` → `Fanout 2`) | None | `arm_id`, `cell`, `why`; campaign fingerprint `v4:` |
| S8 | The walker refuses a cross-module private import; the surface ceiling stays 503 | None | None | None |
| S9 | `_SURFACES` is gone; `test_package_matrix` gains the row | None | None | None |

### The pinned host

Discodon pins `evals/main` by SHA. Through the whole track:

- No declared name (503) moves, is renamed or changes signature. `tests/test_api_tiers.py`,
  `test_public_surface_is_closed.py` and `tests/fixtures/host_imports` hold that per chunk.
- Every moved module leaves a shim at its old path for one minor.
- Every stored-shape change is an optional field that reads absent.
- The two host-visible changes are S3 (new analyses store `v3:` fingerprints; reporter cases holding `v2:` keep
  matching) and S7 (a one-knob derived arm's label). Each gets a handover note before its SHA is offered as a pin.

## 8. A later split into a second distribution

After S9 the presentation side is `analysis/report/`, `analysis/analyst/`, `ops/report_writer/` and `harness/`, and
numbers import none of it. A second distribution (say `3tears-evals-report`) would then take:

1. One directory move: `threetears/evals/analysis/report` and `analyst` to a root that is not under `analysis/`
   (`threetears.evals.report`), because one import package cannot be split across two distributions without making
   `analysis` a namespace package. About 65,000 lines moved, no logic. Shims in the core cannot point up at an
   optional distribution, so this is the breaking step: hosts' imports of `analysis.report.*` and the 109 report
   names the `analysis` root re-exports change, in a minor with a migration note.
2. `ops` split along the same line (`ops/report_writer`, `ops/analyst*.py`, `ops/analysis.py` go with the report),
   and `actions`' report tools registered by the report distribution rather than imported by the core.
3. `quick`'s printing (`Result.__str__`, the CLI summary) either stays on the leaf's fixed forms (it can, after S2) or
   takes the report as an optional extra.
4. The family lockstep rule applies: the new distribution pins the core to the same minor.

What makes it a move and not a rewrite is exactly S2 (numbers print without `PageForms`), S5a (the builder and the
shape are in the core) and S9. Nothing else in the track is done for the split's sake.

## 9. Owner questions, with defaults

| # | Question | Default |
|---|---|---|
| Q1 | May the numbers side spell fixed-form figures through `kernel/numbers.py`, as the engine trees do, instead of `PageForms.off_page()`? This re-points the one-owner gate's line from "never reaches `analysis`" to "the numbers side" | Yes. Fallback if no: move every off-page sentence up (dozens of functions), roughly twice the size of S2 |
| Q2 | Does the code page keep reading the bundle for layout in this track, taking only its forms from the input? | Yes; moving it onto the input is a later track |
| Q3 | Reverse the 2026-11-14 order: fingerprint v3 lands before the report-input shape change, with v2's pre-image frozen? | **Answered yes (2026-11-14).** S3 is being built |
| Q4 | `judge.jsonl`: re-key in place for $0, or re-record for $0.28? | Re-key |
| Q5 | An additive optional bundle field never bumps `schema_version`? | Yes |
| Q6 | A long-lived ratchet on below-root test imports (3,052 today), falling as chunks touch tests? | Yes |
| Q7 | Lift the presentation freeze for S5b alone: one rendering of the four slots, existing strings reused, new strings limited to the four groups listed in §2.7? | Yes; otherwise five tracks each add a first-screen line |
| Q8 | The classes this design assigns to caveat kinds the IA does not name (`short_run`, the two code-page kinds: B; `best_of_n`: D; `unsettled` and the implied ones: E; an analyst's own: after E)? | As listed in §2.6.4 |
| Q9 | `first_screen` stored on the input but left out of `report_input_digest`, so a call or a ranking change never re-keys a snapshot? | Yes |
| Q10 | A stored `flag` prints as an analyst caveat, and the skill is not edited until the next paid re-record batch? | Yes; editing the skill now costs $6.90 of writer re-records |
| Q-A to Q-G | Section 5 | As stated in each row |
