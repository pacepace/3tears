---
artifact: build-plan
version: 1
scope: evals-senior-report
branch: evals/senior-report
depends_on: []
governed_by: []
partition: "delegated in waves, isolated worktrees, at most 3 at once: A = 00 || 02; B = 01 || 03 || 04; C = 05 || 06; D = 07 then 08. Chunk 02 splits packages/evals/src/threetears/evals/analysis/report/campaign_page.py and packages/evals/tests/test_campaign_page.py so 03/04/05/06 own disjoint modules, meeting only at named seams; packages/evals/src/threetears/evals/analysis/report/campaign/common.py and packages/evals/src/threetears/evals/analysis/report/model.py are frozen after 02"
last_validated: 2026-10-24
---

# Build plan: a senior campaign report, for any eval shape

## Goals

**Response taken:** Proceed, citing it. The owner said "Fable's review is great. I agree with it all" (relayed
2026-10-24), and gave scope and limits in the same message (quoted under Constraints). Two more rulings arrived while
this plan was drafted, also 2026-10-24: reporter label form (chunk 00), and caveats stated once (rule 4b).

**Product.** The campaign page, and the memo the writer produces over it, should read like a senior analyst's work for
whatever subject a host evaluates. That covers a model swap, a judge calibration, a latency study, one arm or many,
with or without a control, floors or k>1. The evidence is one review of one memo
(`memo-review.md`, round 1 of the Kairo persona campaign). It is evidence, never the target: no change here may only
make sense for that campaign.

**Architecture: what must hold.**
- Code reports, the writer recommends, a person decides. The page keeps the three apart.
- Code checks the **structure** of LLM output, never its prose. Writer judgment is fixed through prompt, view and
  calibration. No word lists, no prose gates (memory: no-code-checking-prose).
- Package-authored page text stays one template per sentence, so a later locale can supply it. No casing or
  word-count rules over host strings, because they break French and Arabic hosts.
- Every number on the page has one owner and one spelling. A sentence points at the number. It does not restate it.
- Names are declared, never derived. The fallback is the full label or expression.

**Constraints (owner, verbatim, relayed).**
- "we can't templatize the memo at a detailed level; other memos will be about wildly different topics. To the
  extent there's a template or fixed requirements it needs to be at a very high level of abstraction that can adapt
  to wildly different subject matters and eval shapes."
- Use "one or more subagents to address fable's issues, focusing on generalizability and not optimizing for this one
  report at all."
- Aggregates such as "production spend" can be produced, with stated assumptions about how scenarios are weighted:
  equal weight for now ("in the future we might measure weighting from logs"). "That also unblocks charts for
  aggregate values, like cost/quality frontier."
- 2026-10-24: "Any date concerns should be flagged as a caveat to the whole report, not repeated over and over to add
  'but what if we're cheating ourselves' uncertainty each individual finding." Also: "Recommended: September (high
  confidence)" passes on confidence. A post-hoc declaration does not by itself lower a recommendation's confidence.
- 2026-10-24, reporter labels: a label's form matches its dimension's scale. The owner rejected a 5-to-2 mapping
  and rejected moving the dimensions to ordinal.
- Standing rulings that bind:
  - evals-only (nothing outside `packages/evals`, CHANGELOG excepted);
  - no version bumps in PRs;
  - calibration is mechanism only;
  - no pooling by assumption (a silent field is a difference);
  - the report style rules (short headers, numbers in tables, one sentence of commentary, no meta-prose, terms
    defined once), tested on package-authored text only;
  - onboarding joy (every change removes more than it adds where it can).

**Tradeoffs accepted.**
- The writer prompt is at 24,297 of 24,300 characters (`PROMPT_CHAR_BUDGET`, which only falls). Every prompt clause
  added here is paid for by one removed. *Inferred:* the owner would rather cut than raise the budget, because the
  view now carries facts the prompt used to describe. Open question Q1.
- The cost/quality chart is descriptive (points with intervals). It is not a tested dominance reading. Revisit when
  an owner wants a tested aggregate frontier (Q3).
- Reconciling A19 explains two interval machines in place. It does not unify them (Q2).

**Level:** Medium. The renderer and view changes are well understood from the code. Two things are not settled. One
is the prompt budget (Q1). The other is whether the aggregate needs a tested reading (Q3). Settling Q1 and Q3 raises
this to High. The cheapest step is the owner answering them before Wave C.

**Wave B integration to-do (main session, when 01, 03 and 04 merge):**
- Route the remaining goal-check sites through the declared name: report/decision.py:668 (floor readings), report/words.py:333 (evidence_rows), and 04's scorecard.py:521 (`_reconciled`: pass `declared=reports[aspect].bundle.check_name(comparison.name)`) and :1213 (`named()` becomes `measure_name_of`). Add a test that every page site naming a goal check goes through it, then fix the CHANGELOG overclaim.
- test_one_memo_per_decision.py:438-440 raises once 04 moves "By aspect" under Scorecard; reconcile 03's and 04's hunks.
- test_campaign_skeleton.py `_UNFILLED` empties after the merge; drop the parametrised guard or give it its remaining rows so it never runs over nothing.
- page.py: the textual conflict between 04's settle_blocks call and 03's removal of behind_on_blocks.
- `_MATRIX_TABLES={"scorecard"}` in serialize_html depends on 04's private `_SCORECARD_TABLE`; give it a public seam.
- The model.py CampaignReport docstring still describes code's table on the first screen.
- 04's docs/reading-reports.md shows the old first screen; 03 rewrites the same file.

**Wave B integration, done** (merges 6b11d2c89, 3ae3ef26e, db1b43467): every item above is closed; each is a
decision line below, prefixed `B:`.

**Main-session read of the merged S1 page (after wave B), to fix in wave C:**
- To settle prints two identical rows where a floor reads the pass rate at the margin's own points (e.g. billing/v3). One quantity is rendered once, so merge them into one row that says both what it shows and that the floor holds.
- The level and correction explanation under the matrix is seven one-sentence paragraphs. Make it one short block, saying the same mechanism with no recitation.

**From the #951 merge review (rev-20261025T010626Z-1b0c28e4), for wave C:**
- On the default page (`trials="table"`), a check with a declared name appears only by that name, and its expression appears nowhere. Give declared check names a legend (name — expression), once, as aspects have, and test it over campaign_report's default with names declared. Fix CHANGELOG:23-24 and reading-reports.md:333 to match.
- Prose notes to fix in passing: the `_Trace._trial` docstring (memo_blocks.py:411-413) still says the fold holds the record; the Unreleased CHANGELOG's "byte for byte as before" (line 78) against the trials default (lines 99-102).

**Decisions made mid-build:** (append here, one line each)
- [DECISION: runs made at 00107a5b7 stay poolable with every later pin of this plan, and no chunk may move a pooling or identity input: IDENTITY_VERSION, CELL_MODEL_VERSION, context/variant keys, apparatus_class_id, measured_template_digest, the content digest, case ids | main session 2026-10-24, before Discodon spends about $100 on runs at 00107a5b7; an investigation verified this with file:line evidence, and chunk 01 pins every such digest to its 00107a5b7 value]
- [DECISION: Chunk 06's CAMPAIGN_BUNDLE_SHAPE move must keep a reporter case frozen at an earlier shape readable and calibratable (reporter_kind.py:554 refuses it today) | frozen labels are human work; Discodon freezes labels at 00107a5b7, so a shape bump must never void them. 06 adds a read path for earlier frozen shapes or freezes the frame shape-independently, with a test freezing at shape 2 and reading after 06]
- [DECISION: open questions Q1-Q8 take the plan's recommendations | main session 2026-10-24, owner may veto; Q8 = exact substring after collapsing whitespace runs and trimming the ends]
- [DECISION: a re-freeze whose excerpts differ from the stored case's is refused, naming both, and a correction goes through `supersedes` | Critic R-10: a silent drop with a success receipt is a lie to the caller; case identity still excludes the excerpt]
- [DECISION: the 14 drafted labels ran as a scratch check (all 14 validate once reshaped to verdicts; the draft's direction-plus-verdict shape is refused), not a committed fixture | the drafts are campaign-specific owner text, which this plan keeps out of tests; script kept at the main session's scratchpad `s/draft_labels.py`]
- [DECISION: `schema/versioning.py`'s general additivity rule (L181-185, L335-336) gains the opaque-payload exception in Chunk 01, which edits the same ledger | Critic R-6 half-fixed in 00; rides 01 rather than buying a round]
- 02: `packages/evals/src/threetears/evals/analysis/report/campaign/common.py` is a ninth module, holding what several parts share (`as_written`, `named_arms`,
  `floor_of`, `rate_name`, `points`). The underscore rule bars binding another module's private name, and a module
  owned by one chunk should not hold helpers four chunks call. **Frozen after 02:** no chunk edits it; a chunk that
  needs a new shared helper keeps it in its own module, or asks the main session. Settled by: the underscore contract
  (CLAUDE.md) and the partition.
- 02: `late_criteria_lines` and `AspectNames` join `campaign_page.__all__`. `packages/evals/src/threetears/evals/analysis/report/build.py` imports both, and strict mypy
  needs a re-exported name listed. Settled by: mypy `implicit_reexport`.
- 02: `FIRST_SCREEN` gains `answer` and `warnings`, so the first screen's membership is set in 02 rather than 03,
  because `packages/evals/src/threetears/evals/analysis/report/model.py` is frozen after 02 and R1 puts both on the first screen. Settled by: R1, and `packages/evals/src/threetears/evals/analysis/report/model.py` being
  frozen.
- 02: `ReportSection` lists every section in reading order, with the campaign's own interleaved. The new
  `CampaignReport.section_titles` reads that order. `SECTION_TITLES`, and with it every aspect page's headings, is
  unchanged, because `test_one_memo_per_decision` and `packages/evals/tests/test_report.py` pin those headings and their order. Settled
  by: those tests.
- 02: `CostQualityPoint` and `ChartBlock.points` are modelled in 02, ahead of 06, because 06 does not own
  `packages/evals/src/threetears/evals/analysis/report/model.py`. `test_public_api_has_a_caller` stays red for `CostQualityPoint` until 06 constructs points: an
  **accepted red**, closed by 06. Settled by: the ownership table.
- 02: `test_api_growth_requires_a_minor_bump` is red because the unreleased surface grew since v0.77.0 with no bump.
  This chunk's new public names add to that growth: the campaign/* part functions, `AspectNames`, `CostQualityPoint`.
  It is an **accepted red**, and nobody bumps the version in this PR. Settled by: the no-version-bumps-in-PRs ruling.
- 02: S1-S4 are fabricated runs written case by case and assembled with `ToyhostStorage`. They reuse only
  `campaign_support`'s constants. Its real runs give every arm one k, judge nothing without a model call, and draw
  new ids on each run, so pages could not be diffed byte for byte. Settled by: the plan's shapes (per-arm k, judged
  dimensions) and its byte-identity acceptance.
- 02: every aspect name a part prints goes through `packages/evals/src/threetears/evals/analysis/report/campaign/names.py`: `aspect_name(bundle)`, and `AspectNames`, built once
  per page from the bundles. `campaign_warnings` and `campaign_verdict` take the names, and so does `aspect_depth` for
  its sentences (`name=`). Identifiers (floor keys, fold ids, links) keep the label. That way 01 edits only
  `packages/evals/src/threetears/evals/analysis/report/campaign/names.py` and the declaration files. Settled by: the partition.
- 02: both serializers lay sections out in the page's own order, a campaign's up to `provenance` before the folds
  (`CampaignReport.headed_sections`). An aspect's `Report` refuses a campaign-only section, and both pages refuse a
  `glossary` block. A block is never silently dropped, and 04/05/06 need not wait on 03 for their sections to render.
  Settled by: the Critic's dropped-section finding.
- 02: the seams between parallel chunks are named stubs, each with one filler and one placer.
  `aggregate.equal_weight_cells` (06 fills) gives the row `scorecard` places (04) under `EQUAL_WEIGHT_ROW`.
  `aggregate.aggregate_blocks` (06) has its slot in `packages/evals/src/threetears/evals/analysis/report/campaign/page.py`. `provenance.provenance_blocks` (03 fills) and
  `depth_blocks.depth_provenance_blocks` (05 fills) both have slots in `packages/evals/src/threetears/evals/analysis/report/campaign/page.py`'s provenance part. Settled by: the
  partition.
- 01: [DECISION: no identity, digest or fingerprint moves with a declared name, and the bundle shape stays 64 (not 65), so `CORE_PINNED` gains no row | coordinator 2026-10-24: runs made at 00107a5b7 must stay poolable; `test_declared_names.py` pins every key to its 00107a5b7 value with and without names. A shape bump moves every fingerprint, and a name field written only when set leaves an unnamed bundle byte-identical to shape 64]
- 01: [DECISION: a check's display name rides on the bundle (`goal_check_names`), never in `measure_catalog` | names in the catalog flow into `campaign_readings`' baked text and move the campaign fingerprint; `report.campaign.names.measure_name_of` is the door 03/04 call, since the readings keep printing a check as its expression until a part names it]
- 01: [DECISION: `template_digest` (seed drift) leaves `goal_check_names` out as well as `measured_template_digest` | a drift in names alone would read a seeded template stale and a reset would write a new revision, whose results never pool with the old; a stored template is named in place with `template_update`]
- 01: [DECISION: the short name sits on `AspectView` and the bundle's top level, not on `AspectDeclaration` | an exploratory aspect has no declaration and still has a name]
- 01: [DECISION: `campaign_update` gains `short_names` (label to name or null), outside `_DECLARATION_FIELDS`, so a campaign holding runs can be named | otherwise naming an existing campaign means re-creating it; a name asks nothing of the runs, like `name`]
- 01: [DECISION: printed names are unique per campaign (a short name may not equal another aspect's short name or label, whether or not that aspect is named), refused on `EvalCampaign`; check names are unique per template; both under one 32-character cap (`DECLARED_NAME_CHARS`) | two aspects printing alike cannot be told apart in a table; Q5's cap applied to both names, as the brief asks for a length cap]
- 01: [DECISION: the campaign fingerprint and `campaign_evidence_digest` drop `declaration.aspects[].short_name` from their pre-image (`campaign_bundle._evidence_of`) | the frame carries the declaration snapshot, and a reporter case or inspection must not read a rename as moved evidence]
- 01: [DECISION: a `template_update` writing only `TEMPLATE_PRESENTATION_FIELDS` leaves `updated_at` | Critic BLOCKING: `updated_at` decides judge-criterion readability, rejudging, take-up and intent for every earlier run, so a names-only stamp cut them all off and moved the fingerprint through the judge profile; the readability rule for measured edits is unchanged]
- 01: [DECISION: `style.measure_name` takes a check's name as `declared=` and never reads a goal check's reader name; the aspect page, the campaign bars table, its timing lines and the evidence trace pass it | Critic: no page printed a declared name; one explicit door, no guessed second path. A check site 03/04 add passes `declared=bundle.check_name(measure)` or calls `names.measure_name_of`]
- 01: [DECISION: two templates naming one check differently print it whole and log `eval.check_names_differ`; the stored read drops a name whose check is gone (logged) | never a silent first pick; a v16.1 pod renaming or dropping a named check would otherwise leave a template this build refuses on every read]
- 01: [DECISION: presentation fields are named once per model (`ASPECT_PRESENTATION_FIELDS`, `TEMPLATE_PRESENTATION_FIELDS`), and every exclusion (`_NOT_EVIDENCE`, `_evidence_of`, both template digests, the `updated_at` rule) is derived from them; the printed-name rule is `printed_aspect_name`, read by the uniqueness check and `names.aspect_name` | Critic: three hand-kept lists and two copies of one rule]

- 04: [DECISION: the scorecard (matrix, folded rows, its lines, To settle, bars, frontier lines) sits in the headed
  `scorecard` section, off the first screen | R1, and the frozen model: `FIRST_SCREEN` excludes `scorecard`, headed
  "Scorecard". Seven assertions in 03's `test_campaign_first_screen.py` and `test_campaign_memo_blocks.py` that put
  code's table on the first screen go red, for 03 or the integrator to move]
- 04: [DECISION: a matrix cell's reading word rides `flag`, with any differing floor status after it ("unclear; floor
  broken") | a cell holds one value, the number its glyph draws, and `model.py` is frozen; Markdown draws no glyph]
- 04: [DECISION: the R8 mark sits once on the control's cell per aspect, and "cannot show a drop" only where a margin is
  declared | it is the control's property, said once at its level (R4b); with no margin the page already says no arm
  is shown worse there. The check-level R8 case stays in the "Why unclear" reason, as it is not in 04's list]
- 04: [DECISION: a cell's floor mark reads rate floors only, combined as floors are read (broken if any, held if all)
  against the status its reading implies | a cost floor reads no pass rate, and its breach is a warning already]
- 04: [DECISION: To settle is one row per aspect, arm and what the count would show (the margin, or a floor holding),
  the floor's row naming the rate deciding its as-observed count | "each with its kind" read as what the count shows;
  built from the fold's own functions, so a floor's counts equal the fold's]
- 04: [DECISION: A19 reconciles any fold contrast on a higher-is-better reading whose per-case values equal the row's
  and whose verdict differs (shown better/worse against improved/regressed), skipping untested ones; a contrast
  regressed beside a row not worse adds the margin | the review's case in general form; S1 has none, so the test builds
  one]
- 04: [DECISION: the equal-weight seam refuses (ValueError naming the key) a cell for an arm the matrix has no column
  for, and is called through the `aggregate` module | never drop silently; a test stands a figure in where 06 will]
- 04: [DECISION: `settle_blocks(campaign, readings, reports, *, names)`, one call line in `page.py` | a floor's count
  reads each aspect's evidence. `decision.py` is unchanged: `projected_cell` already says each kind]
- 04: [DECISION: one assertion each in `test_campaign_page.py` and `test_one_memo_per_decision.py` reads the Scorecard
  instead of the first screen | neither file is a parallel chunk's; the behaviour moved per R1]

- 04: [DECISION: the control's base rate and n sit in their own `▸ <control> (%)` and `Cases` columns, not in the row
  header | a cell holds one value, and a number in its own column sorts and reads as a number; a "63% · n=60" header
  cell would be a key=value-shaped label, which the style rules bar. Recorded late: the Critic flagged it unrecorded]
- 04: [DECISION: `decision.rate_projections` (new) gives each unclear rate's projection, which `floor_projection`
  joins and To settle names its deciding rate from | one owner for a floor's count and its decider (Critic R-5);
  this goes past "projection kinds and projected_cell only" in 04's ownership of `decision.py`]
- 04: [DECISION: an equal-weight cell states its unit (`TableCell.unit`), which the matrix checks against its
  column and then drops, since the header names it: a change's control column `%`, candidates `pts`, a choice's
  arms `%`; a cell where no matrix holds arms is refused | `aggregate.py` is 06's, so the contract rides the cell it
  already returns (Critic R-6)]
- 04: [DECISION: "cannot show a drop" is decided on the control's per-case shares the row's values read (every row's
  mean at or below the margin), the gain mark on its trials (exact at 100%) | the drop is read on shares; trials
  and shares part where cases ran uneven trials (Critic R-1)]

- 03: the first screen runs title, question (`answer`), summary, recommendation and its whys (`answer`, a choice's `behind_on` why included), warnings then report caveats (`warnings`); the call stays `header`, worded "over code's readings" so it holds wherever 04's scorecard sits. Settled by: R1 and the owner's "title, question, answer, warnings".
- 03: warnings are one line per arm per kind, led by the consequence ("Adopting <arm> overrides the <m>-point floor(s) it breaks on <where>"; a measurement's bar "falls short of"), naming aspects but never the rate, check or interval, which the tables print once. Settled by: R1, R3, R5 (no expression above the fold).
- 03: the plan's "S2 has no warnings section" is false for S2 as built (alpha misses both bars); the test holds it on S4, which nothing warns on. Settled by: the shapes.
- 03: the first-screen rule holds on every page (main-session ruling on Critic R-17, 2026-10-24): an aspect's own page opens on its title, its floor line under its verdict in the campaign's consequence wording (`warnings.floor_line`), no interval, no `role="alert"`; `decision.floor_warning` is removed. The earlier line here, keeping the aspect page's layout because "item 1 names the campaign builder", was wrong: Chunk 03 owns `_floor_warning_blocks` and its deliverable drops the above-title placement. Settled by: R1, the ruling.
- 03: an aspect's own page states each report-wide disclosure as the campaign does, once: its caveat line under the verdict, its detail in its methods (`criteria_set`, and the latency and request-settings lines). Settled by: R4b (line plus row), Critic R-10/R-18.
- 03: Design & provenance holds `criteria_set` and `test_setup` only; margins, floors and bars stay where they are printed once (scorecard, folds, Bars table). Settled by: R3.
- 03: each report-wide disclosure has one construction (`campaign/provenance.py`: `_lines`, `_setup_detail`, `_criteria_table`); its first-screen line carries an id per kind (`caveat-*`), the aspect's full methods line `setup-*`, its tables their names, and `for_the_campaign` drops those from each fold once, in `page.py`, by mark, never by words. The test_setup row's detail is each aspect's own sentence verbatim (latency's with the arms it holds for), so the exact role and settings, and a partial latency's share, stay on the page. Settled by: R4b, Critic R-2/R-3/R-9.
- 03: aspect links cover a campaign table's `aspect` column only; a sentence names aspects unlinked, since a disclosure carries no span structure (model frozen) and scanning code's text for names would be a derivation. Settled by: R5, the frozen model.
- 03: long expressions (>60 chars) collapse to a visible ellipsis in HTML table cells and headers only; sentences keep them whole, since `<details>` cannot sit inside `<p>`, and none remains above the fold. Settled by: Q4, HTML content rules.
- 03: a `scorecard` matrix's glyph columns share one scale per unit (04's request). Settled by: R2 comparability.
- 03: package sentences built with count forms from fragments (an existing `result{'s'}` style) are accepted, since the package's own sentences are English; the lines this chunk added pick one whole template per count form instead. Settled by: main-session ruling on Critic R-19.
- 03: a guardrail line names each guardrail with its own aspects (`the A guardrail on x and the B guardrail on y`), never every guardrail on every aspect. Settled by: Critic R-7.

- B: [DECISION: a page names a goal check through `floor_readings(check_names=)`/`floor_statement(check_names=)`, `AspectNames.check_names` and `names.measure_name_of`; `campaign_readings` passes no names | routing decision.py:668 through the bundle moved `FloorStatus.reason`, which is baked into the campaign readings and so the campaign fingerprint (01's no-fingerprint-moves rule); `test_check_names_on_the_page` pins that naming moves no reading and sweeps S1 (with a call), S2, S4 and a checked measurement for a stray expression, folds included]
- B: [DECISION: a trial's failed-check line keeps the name beside the expression, the name still code-quoted in Markdown | it is the one place a check's text is evidence, and the code span keeps a host's name inert (`test_report_presentation` pins it)]
- B: [DECISION: `common.rate_name` is deleted in the frozen module | 03 removed its last caller, and it printed a check as code with no declared name]
- B: [DECISION: "No question was declared, so every reading is a lead." stands once in the question's place (`provenance.question_blocks`), not under each kind's Scorecard table | R4b, it is report-wide; `test_campaign_create_declares` pinned it on the first screen, and 04's own test moved back to pin it once there]
- B: [DECISION: campaign depth blocks (`depth_blocks.py`) sit in `scorecard`, not the first screen's `scoreboard` | R1: after 04 they were left between the warnings and the call; 05 still owns their collapse and REPEATS_FLOOR's move to Design & provenance]
- B: [DECISION: Critic R-13 as the Critic recommended: `TableBlock.shared_scale` (model, default false) and fold links set where an aspect cell is built (`AspectNames.cell`, `AspectNames.folds` from `fold_ids`); the serializer's `_MATRIX_TABLES`, `_fold_places`, `_linked` go | S1-S4 render byte for byte as before; a trace's "reaches nothing" line now reads only figure and trial links, since the aspect cell links its fold whatever it cites]
- B: [DECISION: `_UNFILLED` and its parametrised guard are deleted | 03 and 04 fill every campaign section, and a guard over an empty table runs nothing]
- B: page.py merged textually clean (04's `settle_blocks` call kept, 03's move of `behind_on_blocks` into the answer kept); verified by reading the merged builder and the first-screen tests.
- B: `FLOORS_WARN_ONLY` said "a broken one opens the page", false since 03; it says "warns on the first screen". Settled by: 03's first-screen ruling.
- B: [DECISION: Q7 is deferred: one plan through 08, since the chunks share seams (names, the Scorecard, the provenance and aggregate slots); archive it at the end, when 08 lands | the plan's Q7 recommendation, the Advisory and the after-Wave-B checkpoint said to archive and redraw here; Wave C's to-dos and Status already continue in this file, and splitting now would leave the seams described in two plans. This line overrides those three for now]
- B: [DECISION: the declared names are durable v16.3 (`DURABLE_SCHEMA_MINOR` 3), in a ledger line of their own, with the rolling-deploy rule "declare no name until every pod runs 16.3" | coordinator 2026-10-24: 0.78.0 shipped 16.2 without the names, and two builds must never claim one minor with different fields. Verified before committing: every identity pin in `test_declared_names` holds (context and variant keys, case-set fingerprint, case content hash, both template digests, cells, aspect and campaign fingerprints, evidence digests, readings, writer message); `schema_minor` is in none of them (`definition_digests` excludes it). Only the `durable version` entry and the durable shape pins move, and a template or campaign naming nothing differs from 00107a5b7's bytes in its minor stamp alone, which the test now checks]
- B: [DECISION: MIGRATIONS' unreleased section is `since = "v0.78.0"` | coordinator 2026-10-24: never back-dated. Expected reds until the maintainer tags v0.78.0: `test_api_surface::test_the_installed_surface_is_the_snapshot_and_its_record`, `::test_each_section_is_checked_against_its_own_tags`, `::test_the_newest_release_the_tree_follows_has_its_own_section`; `generate_api_snapshot.py` refuses for the same reason, so `surface.json` is regenerated after the tag]
- C: [DECISION: the cost/quality chart is drawn where a campaign compares more than one pair of figures: two or more aspects, or three or more arms | 06 builder; the plan contradicted itself (principle 4 says 2+ aspects, chunk 06's tests have S2, one aspect and three arms, drawing and S4 not); this rule meets both tests]
- C: [DECISION: a shape-2 campaign bundle is read as shape 3 (`from_dict`/`from_frame` compute the aggregate from its own readings and aspect bundles), and the fingerprint reads a shape-3 bundle as shape 2 without `readings.aggregate` | binding decision on frozen reporter cases; the aggregate is a pure function of what the fingerprint already covers, so every recorded fingerprint still matches and no identity pin moves]
- C: [DECISION: S2 "tables kept" means its Pass^k depth and Precision tables; its Depth table, every row alike, is one sentence | 05 builder, rule 2]
- C: [DECISION: the level paragraph drops "Correcting across arms would lower it, at the price of wider intervals" and "the reading beside it always agrees with it" as recitation | c3 builder; owner may want the first back]
- C: [DECISION: the "seams are empty today" skeleton test is removed at integration, since 05 and 06 fill all three seams and their own tests pin what fills them | main session]
- D: [DECISION: a disclosure moves to the view's campaign-level `disclosures` when two or more aspects carry it alike; the report-wide kinds (late_criteria, latency_contended, request_settings_disclosure) always move; a one-aspect campaign keeps its own in its part. Shared confound explanations move the same way (`confound_catalog`) | 07 builder, R4b and "terms defined once"]
- D: [DECISION: the view's `aggregate` is handed on two or more aspects only; on one aspect it repeats that aspect's figures | 07 builder, matching 06's page rule]
- D: [DECISION: the per-arm floor rollup replaces the flat `floors` list rather than sitting beside it | 07 builder; one place per floor]
- D: [DECISION: the R8 control rule moves to report/decision.py (`control_cannot_show`, `ControlBound`), which the scorecard and the view both call | 07 builder; campaign_bundle cannot import scorecard (cycle)]
- D: [FOLLOW-UP: the check-level R8 case (a check every arm passes or none does) is flagged neither on the page nor in the view; not in 07's list, filed for after this plan]

## Advisory: what I would do differently

- **Split into two plans at the Wave C boundary.** Wave A-B (schema, names, page structure, scorecard) is renderer and
  schema work. Wave C-D (aggregates, writer view, prompt) needs a re-recording and an owner answer on Q1 and Q3. Nine
  chunks will not ship in three sessions, and the planning guide calls that a program. I kept them in one file so the
  owner sees the whole item map once. *Recommendation:* when Wave B lands, archive this plan and draw
  `.prawduct/artifacts/build-plan-senior-report-writer.md` for chunks 05-08.
- **Cut the review's named-set mechanism and its Markdown casing rule** (see Rejected). Both would add machinery to
  fix what consolidation (rule 3) already removes, and the casing rule is English-only.
- **The biggest risk the review does not price** is that two of its "writer" failures are the package's own bugs.
  - A12: the view's `cases_to_settle` is the run's case count, not a sizing.
  - A6: the view's `change` is spelled at four figures, beside a page that prints one decimal.
  These are fixed in code (chunk 07). Any reporter label that blames the writer for them should be revisited on the
  Discodon side.

## General principles (subject-agnostic)

The skeleton below holds for every campaign kind. Each section appears only when its content exists (rule 2).

1. **Answer.** Title, the question, code's verdict, the writer's summary and recommendation. A measurement has no
   recommendation.
2. **Warnings.** One plain line per arm per warning kind (bars missed, floors broken, guardrails), each a consequence.
   Absent when nothing warns.
3. **Report caveat.** At most one line per report-wide disclosure (rule 4b).
4. **Scorecard.** The form follows the kind's shape (rule 2):
   - change: aspects × candidates;
   - choice: aspects × arms, with the arm shown ahead marked;
   - measurement: aspects × measures for the one arm.
   An equal-weight aggregate row and the cost/quality chart are added where there are two or more aspects and two or
   more arms.
5. **Findings.** The writer's, each with its Evidence fold.
6. **Next.** The writer's steps, sized by code.
7. **Design & provenance.** Arms and control, margins, floors, depth, when each criterion was set, test-setup
   disclosures. All of it as data.
8. **Detail.** One fold per aspect, as now.
9. **Methods and Glossary.**

The rules that produce it. Rules 1-7 are the review's, restated for any shape. Rules 4b and 8-10 generalise owner
rulings and review findings.

- **R1. The first screen is the decision.** Title, question, answer, warnings as consequences. Nothing a
  decision-maker does not need. This holds for every kind: a measurement's answer is its standing against its bars.
- **R2. Form follows the content's shape.**
  - Two categorical axes and one number make a matrix.
  - Intervals against a threshold make glyphs with the threshold drawn.
  - One fact is a sentence.
  - A table whose rows all carry one value set collapses to one sentence. A column empty on every row is dropped.
  - A chart is never drawn for one point.
- **R3. Every quantity is rendered once, in one place.** Prose points at it and says what it means. Two tables never
  print different numbers for one cell unless the page says why, beside them. A figure handed to the writer is spelled
  as the page spells it.
- **R4. A caveat is a consequence, not a fact.** "Because X, the page cannot Y; Z would fix it." Dates, levels, k and
  corrections are data in Design & provenance.
- **R4b. A caveat is stated once, at the level it applies to** (owner, 2026-10-24).
  - A report-wide disclosure gets at most one report-level caveat line plus its Design & provenance row. Examples:
    declaration timing, latency not compared, settings undecided.
  - It is never recited per aspect or per finding, in the page or in the writer's view.
  - It does not by itself lower a recommendation's confidence.
- **R5. One name per entity, defined once, used everywhere.**
  - An aspect has a declared short name, and a check has a declared display name.
  - The full label appears once, in a legend or a header.
  - Code font is for identifiers and expressions only.
  - The fallback is the full label or expression, never a derivation.
- **R6. Progressive disclosure.** Summary, then finding, then evidence fold, then per-aspect detail, then trials.
  Nothing heavy above the fold.
- **R7. Two readings of one quantity that disagree are shown with their reason, in one place.** A disagreement is
  never left for a reader to find by scrolling.
- **R8. A reading that cannot move is said to be uninformative, structurally.**
  - A control at or below the margin cannot show a drop past it.
  - A control at 100% cannot show a gain.
  - A check every arm passes, or none does, cannot separate them.
  Code states these as facts, never as prose judgments. Generalises A20.
- **R9. The writer's view hands facts at the altitude the writer makes claims.**
  - Campaign-wide and per-arm rollups beside per-aspect cells.
  - Each field named for what it holds.
  - Each sizing carried with its kind.
  - Each report-wide disclosure handed once.
  Generalises review item 11 and the two view bugs.
- **R10. Aggregates state their weighting.** A cross-aspect figure carries its weighting basis (`equal` today).
  - It is withheld, with the aspects that lack it named, when any aspect lacks the figure. Never a partial mean
    (no pooling by assumption).
  - It never decides a reading, a floor or a bar.

## Review claims checked against the code

| Item | Review said | Code says | Result |
|---|---|---|---|
| A6 | RENDERER, `prose_refs` printed 53.33 | `prose_number` spells ≥10 whole, so a figure reference renders 53, and no figure reference resolves a campaign reading's change. The view carries `change` through `_figure` at 4 significant figures (`packages/evals/src/threetears/evals/analysis/campaign_bundle.py`, `pass_rate_verdicts` and `readings`) | **Wrong source.** The writer typed a figure from the view. Fix is WRITER-VIEW (R3: view spells as the page), chunk 07 |
| A12 | WRITER invented 115 and 60 | `readings[].cases_to_settle` is `reading.projection.cases`, the usable cases the interval was read over (115 and 60 for qwen), not `if_none`/`if_observed` | **Wrong source.** The view handed the writer those exact numbers under a sizing name. WRITER-VIEW bug, chunk 07 |
| A2 | WRITER-VIEW, medium: no rollup | Confirmed: `writer_view` carries per-aspect `scoreboard` cost only. No per-arm or campaign figure exists | Confirmed |
| A16 | `_which` returns "pass rate" | Confirmed: `_floor_beside_lines` first loop prints "reads its pass rate … while its row reads only the pass rate" when `status.reading is None` | Confirmed |
| A17 | mixed denominators | Confirmed: `cases` is `min(control_cases, arm_cases)` and `passed` is `arm_passed of arm_trials` | Confirmed |
| A18 | note false for multi-rate floors | Confirmed. The row projects the pass rate alone (`project_rate`). `floor_projection` joins every unclear rate (`SampleProjection.joined`). The note's "same count" holds only for a floor reading the pass rate alone | Confirmed |
| A19 | medium on cause | Confirmed in `report.html`, 63.3%→78.3% on the same 60 paired cases. The row is a bounded interval, Holm across the arm's aspects (99.2%): [3.1, 37.6], shown better. The fold contrast is a t interval within the aspect's family (99.8%): [−7.7, 37.7], not separated | Confirmed |
| §4 rewrite of "Nothing corrects across arms" | "the floors and the recommendation do not rest on it" | The generator refuses a candidate with no confirmatory `better` reading unless every floor held, so a recommendation **can** rest on a corrected gain | **Rewrite is wrong in one clause.** Use the rest and drop that clause, chunk 04 |
| A14 | view hands latency once per aspect | Confirmed: `_DISCLOSURES` (incl. `latency_contended`, `request_settings_disclosure`) are read per aspect in `_untrimmed` | Confirmed |
| A10 dup | page prints timing four times plus memo | Confirmed: `late_criteria_lines` in the campaign header, the aspect header (`packages/evals/src/threetears/evals/analysis/report/build.py` `_heading_blocks`) and the listing line (`analysis_verdict`), plus rule 11 asking the summary to say it | Confirmed. Superseded by R4b |
| Chart count | 8 figures and glyphs | Only `ChartKind = outcome_bar \| interval_rows` exist. Glyphs are table cells | Consistent with review |
| §3.2 humanize path exists | yes | `style.measure_name` returns `code(expression)` for any goal check, never a reader name. The catalog's `reader_name` exists but is ignored for checks | Path exists for measures, not for checks. Chunk 01 adds it |
| `packages/evals/src/threetears/evals/analysis/lenses/frontier.py` | (owner asked) | It ranks one subject's contestants per aspect bundle on pass^k × production cost × latency, with dominance decided by test, and draws no chart. The campaign page prints one line per aspect (`_frontier_blocks`, choice kind only). Nothing aggregates across aspects | The aggregate and chart are new (chunk 06). Reuse the `WEIGHTING_EQUAL_PER_SCENARIO` vocabulary of `packages/evals/src/threetears/evals/analysis/lenses/aggregation.py` |

## Item → chunk map

Chunks: 00 reporter labels · 01 names · 02 split and scaffold · 03 first screen and provenance · 04 scorecard ·
05 depth collapse · 06 aggregates and chart · 07 writer view · 08 writer prompt.

### Accuracy findings

| Item | Rule | Chunk / disposition |
|---|---|---|
| A1 title/evidence aspect mismatch | R5 (near-collision names), R9 | 01 (short names), 08 (rule 9 clause names the aspect it cites), Discodon labels |
| A2 one aspect's cost as the arm's | R9, R10 | 06 (equal-weight cost per arm), 07 (rollup in view, cheaper/dearer counts) |
| A3 garbled term | R5 | 07: the view names cost by its page name, not `production_replicating_cost` (the key appears verbatim in the recorded bundle). Writer otherwise, Discodon labels |
| A4 omitted unclear floor | R9 | 07 (per-arm floor rollup). Writer otherwise |
| A5 "queueing" too wide | R5 | 01 (distinct short names), writer, Discodon labels |
| A6 53.33 | R3 | **07** (correction above) |
| A7 "safety" floors | R5 | Writer only, Discodon labels. No code change |
| A8 firmer than an unclear floor | R9 | 07 (floor status beside each check reading cited). Writer otherwise |
| A9 "gain" on within-margin | R9 | 08 (rule 11: "gain"/"improves" only for `better`/`improved`). Discodon labels |
| A10 overbroad late clause | **R4b** | 03 (one report caveat line), 07 (handed once), 08 (rule 11 clause removed). Supersedes the review's "names the aspects it covers" |
| A11 object of confidence | **R4b** | **No change.** The owner ruled the line passes, and a post-hoc declaration does not lower confidence. 08 checks the prompt for any timing-based confidence rule (rule 6 has none today, rule 11's clause is removed) |
| A12 115/60 | R9 | **07** (bug fix: carry `if_none`/`if_observed` with kind) and 08 (rule 12: `never` means a candidate change) |
| A13 next step with no stated effect | R9 | 08 (rule 12 already asks; tighten wording within budget). Discodon labels |
| A14 latency next step filler | R4b, R9 | 07 (report-wide disclosures handed once, campaign level) |
| A15 raw expressions in headline | R1, R5 | 01 (check display names), 03 (warnings under the title, short names) |
| A16 self-contradictory floor-beside line | R7 | 04 (Floor mark in the scorecard; sentences deleted) |
| A17 mixed denominators | R3 | 04 (cases and trials stated apart where they differ) |
| A18 two sizing tables | R3, R7 | 04 (one table, "pass rate" and "every check" side by side, the deciding check named) |
| A19 shown better vs not separated | R7 | 04 (one reconciliation line where a fold contrast's per-case values equal the row's pass-rate values and the verdicts differ). Unification is Q2 |
| A20 degenerate control | R8 | 04 (scorecard marks the cell "cannot show a drop"), 07 (view flag) |

### Prioritised change list

| Item | Rule | Chunk / disposition |
|---|---|---|
| 1 Title first; plain warning line per arm; drop `role="alert"` | R1 | 03 |
| 2 Timing lines to Design & provenance; drop rule-11 clause | R4, **R4b** | 03 (renderer: one caveat line plus a provenance table; per-aspect recitations dropped from campaign header, aspect header and listing line), 08 (clause removed, not replaced) |
| 3 Collapse degenerate sections | R2 | 05 |
| 4 Floor-beside fix or Floor column | R2, R7 | 04 (Floor column, sentences deleted) |
| 5 Rewrite level/within/settle notes; remove "same count" | R3, R4 | 04 (§4 rewrites, parametrised by counts; the one wrong clause dropped) |
| 6 Reconcile sizing tables | R3 | 04 |
| 7 Scorecard matrix | R2 | 02 (cell fields), 04 (matrix per kind) |
| 8 Reconcile shown better vs not separated | R7 | 04 |
| 9 Prose figures formatted as tables | R3 | 07 (the source was the view, see A6). `prose_refs` unchanged |
| 10 Short names, display names | R5 | 01 (fields, bundle, `measure_name(declared=)`, the aspect page and the campaign page's existing check sites), 03/04 pass `declared=` at any check site they add, 07 hands to writer |
| 11a cost rollup | R9, R10 | 06 computes, 07 hands |
| 11b per-arm floor rollup | R9 | 07 |
| 11c campaign-wide disclosures once | **R4b** | 07, generalised: every report-wide disclosure, late criteria included, handed once at campaign level, never per aspect |
| 11d degenerate control flag | R8 | 07 (flag), 04 (page) |
| 11e sizing with its kind | R9 | 07 |
| 12 Prompt clauses | R9 | 08, within budget. The late-criteria sentence clause is **dropped** per R4b, not added |
| 13 Reporter calibration cases | - | **Discodon side** |

### Other review sections and owner additions

| Item | Rule | Chunk / disposition |
|---|---|---|
| §2 diagnosis: machinery order | R1, R6 | 02 (section ids), 03 (order) |
| §2 "senior version of THIS memo" content | - | **Rejected as a target**: campaign-specific. Used as evidence only |
| §2 subject-agnostic skeleton | R1-R7 | 02/03 (sections appear by content, per kind) |
| §2 per-aspect summary template "How sure" restates the row | R3 | 08 (rule 11 aspect-summary wording, within budget). The three slots stay structural |
| §3.1 aspect short name | R5 | 01. Discodon declares the names on seeds |
| §3.2 check display name | R5 | 01 (field on the template). Fallback is the full expression (Q4 for long ones) |
| §3.3 code font only for identifiers; aspect links in HTML | R5 | 03 |
| §3.3 Markdown casing rule (Title Case arms, lowercase aspects) | - | **Rejected**: an English casing rule over host strings (no-code-checking-prose: French/Arabic). Names print as declared |
| §3.4 named sets | R3 | **Rejected as a mechanism.** Naming a set automatically is a derivation, and the recurrences the review counted (timing ×4, floors ×3, depth ×8) are removed by consolidation in 03/04/05 |
| §3 "≤3 words" | R5 | **Modified**: a character cap (32), since a word count is language-bound (Q5) |
| §3 the campaign's own short names | - | **Discodon side** |
| §4 statistical rewrites | R4 | 04 (levels, within, floor-beside, settle), 05 (depth sentences), 03 (timing becomes one caveat line, superseding §4's timing rewrite per R4b). Each rewrite is parametrised by the page's counts, never "eight" or "four" |
| §2 lessons from Claude artifact generation | R1, R2, R6 | 02-05 (conditional sections, form by content, legend) |
| Page weight (7 MB trials, 1,775 folds) | R6 | **Not in this plan, flagged**: no review item asks for it. It belongs in the backlog |
| Owner: equal-weight aggregates incl. cost | R10 | 06 (computed in readings), 04 consumes (scorecard aggregate row), 07 (view) |
| Owner: aggregate cost/quality frontier chart | R2, R10 | 06 (new chart kind with hidden table). Per-aspect `packages/evals/src/threetears/evals/analysis/lenses/frontier.py` verdicts unchanged |
| Owner 2026-10-24: caveats once (R4b) | R4b | 03 (page), 07 (view), 08 (prompt). Mapped above under A10, A11, A14, 2, 11c, 12 |
| Owner 2026-10-24: reporter label form | - | 00 |
| Owner 2026-10-24: `memo_excerpt` | - | 00 |

## Shape-bearing constants (what the consumer re-pins on)

| Constant | Where | Moves in | To |
|---|---|---|---|
| `DURABLE_SCHEMA_MINOR` (+ v16.2 ledger line) | `packages/evals/src/threetears/evals/schema/versioning.py` | 00 (reporter labels), extended by 01 (`Aspect.short_name`, `EvalTemplate.goal_check_names`) | 2. v16.1 shipped in 0.75.0, so this is a new minor |
| Aspect bundle `schema_version` (+ `CORE_PINNED` row, shape-only) | `packages/evals/src/threetears/evals/analysis/bundle/schema.py` `AnalysisContextBundle.schema_version`; `packages/evals/tests/test_hostneutral_sweepables.py` | **does not move in 01**: `short_name` and `goal_check_names` are written only when set and kept out of every fingerprint, so an unnamed bundle is shape 64 byte for byte; `measure_catalog` is unchanged | 64/14, no new `CORE_PINNED` row |
| `CAMPAIGN_JSON_SHAPE` | `packages/evals/src/threetears/evals/analysis/report/model.py` | 02 (section ids, cell fields, chart kind), extended by 03-06 within the same unreleased shape | 5 |
| `CAMPAIGN_BUNDLE_SHAPE` | `packages/evals/src/threetears/evals/analysis/campaign_bundle.py` | 06 (`readings.aggregate`) | 3 |
| `EVAL_ANALYSIS_GEN_DEFAULT` (seed; hosts `eval_prompt_reset`) | `packages/evals/src/threetears/evals/analysis/gen_prompt.py`; ledger `packages/evals/src/threetears/evals/analysis/GEN_PROMPT_RULES.md` | 08 | moves; `PROMPT_CHAR_BUDGET` only falls |
| Writer recording `packages/evals/tests/fixtures/generate_analysis/recordings/` (`llm_analysis.jsonl`, `.bundle.json`, `.prompt.txt`) | replayed by `packages/evals/tests/test_generate_analysis_fixture.py` | stale after 01 (bundle), 06 (readings), 07 (view), 08 (prompt) | **re-recorded once, by the main session, after 08** |
| `packages/evals/api/surface.json`, `packages/evals/docs/reference.md` | regenerated | 00 (`ReporterLabel` is host-contract), 01, 06 | the main session regenerates at integration, never a builder |

No other `*_SHAPE` constant exists in the package (`ASPECT_BUNDLE_SHAPE = 59` is a floor for refusal, not the current
shape).

## Chunks

Builder brief, every chunk:
- Read `/prawduct:methodology building`; work in an isolated worktree off this branch; evals-only.
- Tight single-file or `-k` test selections, one at a time, never a suite or `check-all.sh` (memory:
  subagents-no-heavy-tests). Lint and typecheck by reading.
- Never touch `.prawduct/`, `packages/evals/api/surface.json` or `packages/evals/docs/reference.md`.
- Append to `CHANGELOG.md` and the `unreleased` MIGRATIONS notes. The main session reconciles them.
- No version bumps. No agent attribution anywhere.
- A sentence a test pins may change only where this plan changes that behaviour, with the reason in the commit.
  Tests are contracts.

**Shared fixtures.** Chunk 02 creates new `packages/evals/tests/report_shapes_support.py`, using `packages/evals/tests/campaign_support.py`'s constants. It provides
four campaign shapes that every later chunk's tests run against. A behaviour test passes on at least two of them:

- **S1 change.** A control and three candidates, three aspects, k=1 with one arm at k=2 on one aspect. Floors include
  a check-reading floor. One aspect has the control at 0%.
- **S2 choice.** Three arms, no control, one aspect, k=3, bars, no floors. Priced and timed, latency-study-like.
- **S3 measurement.** One arm, two aspects, judged dimensions, a bar.
- **S4 change.** One candidate, one aspect, no floors, no margin.

### Chunk 00: Reporter label form and memo excerpt

Landed first: Discodon holds a label freeze on it. Independent of every report file.

**Owns:**
- `packages/evals/src/threetears/evals/analysis/reporter_kind.py` (`ReporterLabel`, `LabelDirection`, `LABEL_BANDS`, `reporter_case_payload`)
- `packages/evals/src/threetears/evals/analysis/reporter_bank.py` (`label_agrees`, `LabelReading`, the calibration read)
- `packages/evals/src/threetears/evals/analysis/service.py`: only `freeze_reporter_case` and `_label_set`
- `packages/evals/src/threetears/evals/ops/reporter.py`
- the label description in `packages/evals/src/threetears/evals/actions/engine.py`
- `packages/evals/src/threetears/evals/schema/versioning.py` (v16.2)
- tests: `packages/evals/tests/test_hostneutral_reporter_bank.py`, `packages/evals/tests/test_reporter_case_actions.py`, `packages/evals/tests/test_reporter_case_freeze_attested.py`,
  `packages/evals/tests/reporter_case_support.py`, and new `packages/evals/tests/test_reporter_label_form.py`

**Deliverables.**
- `ReporterLabel` gains `verdict: Literal["pass", "fail"] | None`. `direction` becomes optional.
  - Exactly one of the two is set. Model-level, structural.
  - A label stored before reads as it did: direction set, no verdict.
  - `verdict` is written to the payload only when set, as `reader`/`collected_by` are. So a direction label is
    stored byte-identically to v16.1.
- `label_agrees(label, score)`:
  - an ordinal score reads `direction` against `LABEL_BANDS`, as now;
  - a pass/fail score reads `verdict == ("pass" if score passes else "fail")`;
  - a form that does not match its score's scale (a stored legacy direction on pass/fail) stays `None`, as today.
  - The owner rejected any 5-to-2 mapping, so none is added.
- The freeze (`freeze_reporter_case`, and the `reporter_case_freeze` operation through it) refuses a label whose form
  does not match its dimension's scale. The scale is read from the criterion the freeze stamps (`LabelCriterion.scale`:
  `pass_fail` needs `verdict`, `ordinal` needs `direction`). Validation error, nothing written. Never applied on read,
  so a stored mismatched label keeps reading.
- `memo_excerpt: str | None` on `ReporterLabel`.
  - The freeze refuses it unless it is a substring of the case's `recorded_memo` (Q8: exact, after collapsing runs of
    whitespace on both sides).
  - A label with an excerpt on a case pinning no memo is refused.
  - It is written only when set, and dropped in `_label_set` beside `reader`/`collected_by`, so it is not part of case
    identity.
  - `LabelReading` gains `verdict` and `memo_excerpt`, so the calibration read shows the excerpt beside the quote.
- Durable v16.2 ledger line (ends with `MINOR_ADDITIVITY_SENTENCE`). The rolling-deploy note follows the v16.1
  pattern: a v16.1 pod cannot read a case with a verdict or excerpt label, so the template stops there until it
  upgrades.

**Behavioural tests.**
- A pass/fail case with verdict labels yields agreement `True`/`False`, not `None`.
- The ordinal path is unchanged.
- The freeze refuses: a direction on pass/fail, a verdict on ordinal, both, neither, an excerpt not in the memo, an
  excerpt with no memo.
- `packages/evals/tests/fixtures/reporter_cases/labelled_case_v16_0.json` still reads, and re-stores byte-identically.
- Re-freezing the same verdicts with a different excerpt names the stored case.
- Run against the 14 labels in `reporter-labels-draft.json` reshaped. Note: in that draft `quote` holds memo text; the
  memo passage belongs in `memo_excerpt` and the reader's words in `quote`. Tell Discodon.

**Acceptance.** Every refusal is structural (form, substring), never semantic. No bundle or campaign shape moves.

### Chunk 01: Declared names (aspect short name, check display name)

Depends on 00 (`packages/evals/src/threetears/evals/schema/versioning.py`, v16.2 entry).

**Owns:**
- `packages/evals/src/threetears/evals/kernel/declaration.py` (`Aspect`)
- `packages/evals/src/threetears/evals/schema/models.py` (`EvalTemplate`)
- `packages/evals/src/threetears/evals/run/definition_digests.py` (`_UNMEASURED_TEMPLATE_FIELDS`)
- `packages/evals/src/threetears/evals/analysis/aspects.py`
- `packages/evals/src/threetears/evals/analysis/bundle/schema.py` (shape stays 64: `short_name`, `goal_check_names`, `check_name`), plus the bundle assembly
  (`packages/evals/src/threetears/evals/analysis/bundle/assemble.py`, `_check_names`); `measure_catalog` is not touched
- `packages/evals/src/threetears/evals/analysis/style.py` (`measure_name`)
- `packages/evals/src/threetears/evals/analysis/report/campaign/names.py`, the one naming helper (02 created it as a pass-through, and every part already names
  aspects through it). 01 fills `aspect_name` from the declared short name and adds `legend`. It edits no other
  part's module
- the campaign declare/adopt actions' argument descriptions
- new `packages/evals/tests/test_declared_names.py` (no `CORE_PINNED` row: the shape does not move)

**Deliverables.**
- `Aspect.short_name: str | None`.
  - Unique within the campaign, at most 32 characters (Q5), not blank.
  - Default None, written only when set.
  - Copied by adoption (`aspects_from`).
  - Not in any identity, fingerprint or staleness digest. The builder verifies a short-name edit marks no analysis
    stale.
- `EvalTemplate.goal_check_names: dict[str, str]`.
  - Expression to display name, keys must be declared checks.
  - Excluded from `measured_template_digest` and `template_digest`, because presentation does not change what a run
    measures; a `template_update` of it alone leaves `updated_at`.
  - It rides on the bundle (`AnalysisContextBundle.goal_check_names`, `.check_name(measure)`), never in
    `measure_catalog`. `style.measure_name(..., declared=)` prints it when passed, and otherwise `code(expression)` in
    full; `names.measure_name_of(bundle, measure)` is the door for a part holding a bundle.
- The short name carried into the bundle (`AspectView.short_name`, `AnalysisContextBundle.short_name`). Bundle shape
  stays 64, no `CORE_PINNED` row.
- `packages/evals/src/threetears/evals/analysis/report/campaign/names.py`: `aspect_name(bundle)` returns `kernel.declaration.printed_aspect_name`, the one statement of
  "the short name, else the full label". `AspectNames.of` reads it for every
  part, so every table cell and sentence follows without other edits, depth's sentences included (via
  `aspect_depth(name=)`). `legend(...)` gives the pairs for one legend row, and 03 places it.

**Tests.**
- S1 with and without names: the page uses short names in every table cell, and the full label appears once in the
  legend and fold header.
- An undeclared name falls back to the full label or expression unchanged, never truncated.
- A duplicate or over-long short name is refused at declare.
- A template with names has an unchanged measured digest.

**Acceptance.** No derivation anywhere. Floor keys and every identifier still use `label`.

### Chunk 02: Split the campaign page and scaffold the skeleton

A cleanup plus a small schema change. No behaviour change except the JSON shape.

**Owns:**
- `packages/evals/src/threetears/evals/analysis/report/campaign_page.py`, which becomes a facade re-exporting today's `__all__` (plus `late_criteria_lines` and
  `AspectNames`)
- new `packages/evals/src/threetears/evals/analysis/report/campaign/` package. Each module is owned by one later chunk:

  | Module | Holds | Owner |
  |---|---|---|
  | `packages/evals/src/threetears/evals/analysis/report/campaign/page.py` | `build_campaign_report`, section order, one call per part, the provenance slot | 02 (03 patches the slot list only) |
  | `packages/evals/src/threetears/evals/analysis/report/campaign/warnings.py` | warning lines | 03 |
  | `packages/evals/src/threetears/evals/analysis/report/campaign/provenance.py` | question, late criteria, `provenance_blocks` (Design & provenance, empty today) | 03 |
  | `packages/evals/src/threetears/evals/analysis/report/campaign/memo_blocks.py` | memo, trace, summaries, recommendation, call; the `findings` and `next` sections' blocks | 03 |
  | `packages/evals/src/threetears/evals/analysis/report/campaign/scorecard.py` | table, pair, measure and bar blocks; notes; settle; places `EQUAL_WEIGHT_ROW` | 04 |
  | `packages/evals/src/threetears/evals/analysis/report/campaign/depth_blocks.py` | depth, pass^k, model, precision; `depth_provenance_blocks` (empty today) | 05 |
  | `packages/evals/src/threetears/evals/analysis/report/campaign/aggregate.py` | `equal_weight_cells` and `aggregate_blocks` stubs, `EQUAL_WEIGHT_ROW` | 06 |
  | `packages/evals/src/threetears/evals/analysis/report/campaign/names.py` | `aspect_name`, `AspectNames`: every aspect name a part prints | 01 |
  | `packages/evals/src/threetears/evals/analysis/report/campaign/common.py` | `as_written`, `named_arms`, `floor_of`, `rate_name`, `points` | **frozen after 02** |

- `packages/evals/src/threetears/evals/analysis/report/model.py` (**frozen after 02**):
  - section ids `answer`, `warnings`, `scorecard`, `findings`, `next`, `provenance` in `ReportSection`, headed by
    `CampaignReport.section_titles` (titles of at most 3 words; `answer` and `warnings` sit on the first screen);
  - an aspect page refuses them, and every page refuses a block in `glossary`;
  - `CampaignReport.headed_sections`, the order both serializers lay sections out in;
  - `TableCell.flag` (a short code-authored marker, such as the floor status or "cannot show a drop", rendered
    after the value);
  - `ChartKind` gains `cost_quality` (`ChartBlock.points`, `CostQualityPoint`);
  - `CAMPAIGN_JSON_SHAPE` 5.
- `packages/evals/src/threetears/evals/analysis/report/serialize_html.py` and `packages/evals/src/threetears/evals/analysis/report/serialize_md.py`: only the section loops, which read the page's own section
  order. The rest of both stays 03's.
- `packages/evals/src/threetears/evals/analysis/report/svg.py`: only the `cost_quality` refusal in `chart_svg`. The file is 06's.
- `packages/evals/src/threetears/evals/analysis/depth_reading.py`: only `aspect_depth(name=)`. The file is 05's.
- `packages/evals/src/threetears/evals/analysis/report/build.py`: only `analysis_verdict`'s `AspectNames` argument. The function is 03's.
- split `packages/evals/tests/test_campaign_page.py` along the same modules, tests moved verbatim:
  - `packages/evals/tests/test_campaign_warnings.py`, `packages/evals/tests/test_campaign_provenance.py`, `packages/evals/tests/test_campaign_memo_blocks.py`
    (03);
  - `packages/evals/tests/test_campaign_scorecard.py` (04);
  - `packages/evals/tests/test_campaign_first_screen.py`, the first-screen order, owned by 03. Each part's marker sits in a row tagged
    by its chunk, so 04 edits only its `_SCORECARD` rows;
  - `packages/evals/tests/test_campaign_page.py`, the page's folds, JSON and invariants (02).
  Their shared fixtures are in `packages/evals/tests/campaign_page_support.py`.
- new `packages/evals/tests/report_shapes_support.py` (S1-S4) and `packages/evals/tests/test_report_shapes.py`
- new `packages/evals/tests/test_campaign_skeleton.py`. Its `_UNFILLED` table names, per section, the chunk that fills it. Each
  chunk deletes only its own row.

**Acceptance.**
- Every moved test passes unchanged, apart from `CAMPAIGN_JSON_SHAPE == 5`.
- `git diff` of rendered HTML and Markdown for S1-S4 before and after is empty.

Verification: render S1-S4 to HTML and Markdown and diff.

### Chunk 03: First screen, caveats once, Design & provenance

Depends on 01, 02.

**Owns:**
- `packages/evals/src/threetears/evals/analysis/report/campaign/warnings.py`, `packages/evals/src/threetears/evals/analysis/report/campaign/provenance.py` (fills `provenance_blocks`)
- `packages/evals/src/threetears/evals/analysis/report/campaign/memo_blocks.py`: moves the memo's findings and next steps into the `findings` and `next` sections.
  The writer's words in them are still `memo_blocks`'s, as written
- `packages/evals/src/threetears/evals/analysis/report/serialize_html.py`, `packages/evals/src/threetears/evals/analysis/report/serialize_md.py`: the layout. They already lay out every section the page
  declares, in its order
- `packages/evals/src/threetears/evals/analysis/report/build.py`: only `_heading_blocks`, `_floor_warning_blocks` and `analysis_verdict`
- the `answer`/`warnings`/`findings`/`next`/`provenance` order in `packages/evals/src/threetears/evals/analysis/report/campaign/page.py`, by patch to the slot list only
- `packages/evals/tests/test_campaign_warnings.py`, `packages/evals/tests/test_campaign_provenance.py`, `packages/evals/tests/test_campaign_memo_blocks.py`,
  `packages/evals/tests/test_campaign_first_screen.py` (the order; 04 edits only its `_SCORECARD` rows), and the `answer`,
  `warnings`, `findings`, `next` and `provenance` rows of `_UNFILLED` in `packages/evals/tests/test_campaign_skeleton.py`

**Deliverables.**
- The title comes first. `role="alert"` and the above-title placement go.
- Warnings sit under the question, one plain line per arm per kind, named with short and display names. Example
  shape: "<arm> breaks the <m>-point floor on <n> of <N> aspects: <short names>".
- Report-wide disclosures (R4b) become **one** caveat line each at report level, worded as a consequence, plus rows in
  a Design & provenance table (Criterion · Set on · Aspects whose results existed). Applies to:
  - criteria set after their readings;
  - latency not compared;
  - settings undecided;
  - untimed results.
- `late_criteria_lines` per-aspect recitations are removed from the campaign header, the aspect header and the listing
  line (`analysis_verdict` keeps at most the one caveat line).
- HTML: every aspect mention links to its fold. Code font only for expressions. An expression past a length renders
  as a visible-ellipsis summary with the full text in its fold (Q4).

**Tests** (S1, S2, S4):
- The first element after `<h1>` is the question.
- No header block precedes the title.
- A campaign with late criteria on k of N aspects prints exactly one timing line, and the provenance table lists the
  k aspects.
- An aspect page and the listing line carry no per-aspect timing recitation.
- S2 has no warnings section.

**Acceptance.** Style lint (`style.lint_page`) passes on every new package sentence.

### Chunk 04: Scorecard, floors beside it, sizing, reconciliation

Depends on 01, 02.

**Owns:**
- `packages/evals/src/threetears/evals/analysis/report/campaign/scorecard.py`, including where the `EQUAL_WEIGHT_ROW` sits in the matrix. Its cells come from
  `aggregate.equal_weight_cells` (06), and 04 does not edit `packages/evals/src/threetears/evals/analysis/report/campaign/aggregate.py`
- `packages/evals/src/threetears/evals/analysis/report/decision.py` (projection kinds and `projected_cell` only)
- not `packages/evals/src/threetears/evals/analysis/report/svg.py`, which is 06's. A matrix cell's glyph is the existing cell glyph (`TableCell` `low`/`high`/
  `margin`, drawn by the HTML serializer). A glyph that needs more is a new function in a new module of 04's, never
  an edit to `packages/evals/src/threetears/evals/analysis/report/svg.py`
- `packages/evals/tests/test_campaign_scorecard.py`, `packages/evals/tests/test_report_projected_sample.py`, the `_SCORECARD` rows of
  `packages/evals/tests/test_campaign_first_screen.py`, and the `scorecard` row of `_UNFILLED` in
  `packages/evals/tests/test_campaign_skeleton.py`

**Deliverables.**
- The matrix per kind (R2):
  - change: rows are aspects, columns candidates. Each cell is the change in points, the glyph against the margin
    band, the reading word, and `flag` for a floor whose status differs from the row. The row header carries the
    control's base rate and n.
  - choice: rows are aspects, columns arms, with the ahead-of marks.
  - measurement: rows are aspects, columns measures.
  - The long per-row table moves to a fold.
- Cases and trials stated apart where they differ (A17).
- `_floor_beside_lines` deleted for the Floor flag (A16).
- One sizing table per unclear arm and aspect: the pass rate, every check (with the deciding check named), each with
  its kind. The false "same count" sentence goes (A18).
- §4 rewrites of the level and within notes, parametrised by arm and aspect counts. The clause "the recommendation
  does not rest on it" is dropped (wrong).
- One reconciliation line per aspect where a fold contrast's per-case values equal the row's pass-rate values and
  their verdicts differ, naming both methods (A19).
- R8 flags: a control rate at or below the margin, or at 100%, marks the cell "cannot show a drop" or "cannot show a
  gain".
- An S4-shaped single aspect and single candidate renders as a one-row matrix, never a chart.

**Tests:**
- Matrix dimensions per kind on S1-S4.
- A16, A17, A18 regressions on S1.
- The A19 line appears only on matching case values.
- The R8 flag on S1's 0% aspect, and not when control > margin.

### Chunk 05: Degenerate sections collapse

Depends on 02.

**Owns:**
- `packages/evals/src/threetears/evals/analysis/report/campaign/depth_blocks.py`, including `depth_provenance_blocks`, whose blocks `packages/evals/src/threetears/evals/analysis/report/campaign/page.py` places in Design &
  provenance; 05 does not edit `packages/evals/src/threetears/evals/analysis/report/campaign/provenance.py`
- `packages/evals/src/threetears/evals/analysis/depth.py` (sentence templates), `packages/evals/src/threetears/evals/analysis/depth_reading.py` (keeping `aspect_depth(name=)`)
- `packages/evals/tests/test_depth_surfaces.py`, `packages/evals/tests/test_depth_disclosure.py`, `packages/evals/tests/test_depth_review.py`, `packages/evals/tests/depth_support.py`

**Deliverables:**
- The Depth table collapses to one sentence where every arm's repeats are equal on every aspect but a few, which the
  sentence names (§4 wording).
- No Pass^k table at a common depth of 1 on every aspect: one campaign-level sentence instead of one per aspect.
- A Precision column empty on every row is dropped.
- Precision at k=1 on every arm moves under Methods.
- `REPEATS_FLOOR` is said once, in Design & provenance, as a block `depth_provenance_blocks` returns in the
  `provenance` section.

**Tests:**
- S1 (mostly k=1): one sentence and no table.
- S2 (k=3): tables kept.
- The writer view's `depth.said` is unchanged for 07 to consume.

### Chunk 06: Equal-weight aggregates and the cost/quality chart

Depends on 02. Answers to Q3 shape the reading.

**Owns:**
- `packages/evals/src/threetears/evals/analysis/campaign_readings.py` (new `ArmAggregate` / `CampaignAggregate` on `CampaignReadings`)
- `packages/evals/src/threetears/evals/analysis/report/campaign/aggregate.py`: fills `equal_weight_cells` (each arm's cell in the scorecard's `EQUAL_WEIGHT_ROW`,
  by variant key) and `aggregate_blocks` (the chart, in its slot in `packages/evals/src/threetears/evals/analysis/report/campaign/page.py`)
- `packages/evals/src/threetears/evals/analysis/report/svg.py`, sole owner: replaces the `cost_quality` refusal in `chart_svg` with the drawing. A
  `CostQualityPoint` construction closes the accepted `test_public_api_has_a_caller` red
- `packages/evals/src/threetears/evals/analysis/campaign_bundle.py`: only `CAMPAIGN_BUNDLE_SHAPE`
- new `packages/evals/tests/test_campaign_aggregate.py`

**Deliverables.**
- Per arm, over aspects, with `weighting: {"basis": "equal", "weights": {aspect: w}}`:
  - the mean pass rate and, with a control, the mean paired change, each with an interval (aspects assumed
    independent, stated);
  - the mean production cost per reply, with an interval from each aspect's cost SEM;
  - counts of aspects where the arm is cheaper or dearer.
- Withheld, naming the missing aspects, when any aspect lacks the figure (R10). No reading word.
- The chart: one point per arm, cost on x, aggregate pass rate on y, both intervals, with a visually hidden table of
  every figure.
  - Drawn for 2+ arms.
  - On one aspect it is that aspect's values, with no aggregate row.
  - Never for a measurement.
- The scorecard's `EQUAL_WEIGHT_ROW` cells come from `equal_weight_cells`, once 2+ aspects exist. 04 places the row.

**Shape:** `CAMPAIGN_BUNDLE_SHAPE` 3.

**Tests:**
- S1 aggregate equals the hand mean.
- An unpriced arm on one aspect withholds cost and names it.
- S2 draws a chart.
- S3 and S4 draw none.
- The weights sum to 1 and are stated in the JSON.

### Chunk 07: The writer's view

Depends on 01 and 06.

**Owns:**
- `packages/evals/src/threetears/evals/analysis/campaign_bundle.py` (`writer_view`, `_untrimmed`, `_DISCLOSURES` handling)
- `packages/evals/tests/test_campaign_bundle_budget.py`, `packages/evals/tests/test_bundle_growing_lists.py`
- new `packages/evals/tests/test_writer_view_rollups.py`

**Deliverables:**
- `aggregate` per arm (from 06), with the weighting.
- A per-arm floor rollup: the arm, then the aspects broken, unclear and held, each with the check it read.
- Report-wide disclosures handed **once** at campaign level (R4b, 11c generalised). This covers `late_criteria`,
  `latency_contended`, `request_settings_disclosure` and any disclosure equal across every aspect. None of them is
  repeated per aspect.
- Per-aspect `control_floor` flags (R8).
- Each reading's `cases_to_settle` is replaced by `to_settle: {if_none: {kind, cases}, if_observed: {kind, cases}}`
  for the pass rate, and the same per floor. **This fixes the A12 bug.**
- `change` and the other point figures are spelled as the page prints them (A6).
- `short_name` per aspect, and check display names in `cells`/`measures`.
- Cost named by its page name (A3).

**Tests** (S1 and S2):
- No disclosure key repeats across aspect parts.
- `to_settle` carries kinds.
- `never` appears for an observed drop past the margin.
- The budget tests hold.

### Chunk 08: The writer prompt

Depends on 07.

**Owns:** `packages/evals/src/threetears/evals/analysis/gen_prompt.py`, `packages/evals/src/threetears/evals/analysis/GEN_PROMPT_RULES.md`, `packages/evals/tests/test_gen_prompt_budget.py`.

**Deliverables**, within `PROMPT_CHAR_BUDGET` (it only falls):
- **Remove** rule 11's "`summary` also says once that each `late_criteria` entry was set after…" clause and its
  ledger reason (R4b).
- Keep "Never discount a reading for when its runs ran".
- Confirm no rule lowers confidence on timing alone (rule 6 has none).
- The "WHAT THE BUNDLE HOLDS" paragraph names the new fields (`aggregate`, the floor rollup, `to_settle`, campaign
  disclosures, `short_name`) and drops the descriptions they replace.
- Rule 11: "gain"/"improves" name only `better`/`improved` readings. An arm's unclear floors are listed from the floor
  rollup.
- Rule 12: quote `to_settle` with its kind; `never` means a change to the candidate, not more cases.
- Rule 9: a control the view flags (R8) is a validity finding raised first. Name an aspect by its `short_name`.
- Every added clause names the words it replaces in the ledger row.

**Tests:** the budget test and ledger match.

**Main-session step, not a builder's:** re-record the writer fixture once, here, with `RECORD=1`.
- `packages/evals/tests/fixtures/generate_analysis/llm_analysis.py` is wired to `claude-haiku-5-5` via `ANTHROPIC_API_KEY` from env. That is the ceiling the memory
  rule allows for a Claude-specific path: provider anthropic, no temperature.
- If the fixture is not meant to prove a Claude path, record on a cheaper OpenRouter model (GLM / DeepSeek / Gemini
  flash-lite) instead.
- Hosts push the new seed (`eval_prompt_reset`).

## Verification strategy

Each builder:
- renders S1-S4 to HTML and Markdown and reads the first screen as a decider would;
- checks R1-R10 by hand against its chunk;
- runs its own tight test selections.

The main session:
- integrates;
- regenerates `packages/evals/api/surface.json` and `packages/evals/docs/reference.md`;
- runs `check-all.sh` once per integration head, in the background, telling Discodon first;
- runs `test-integration.sh -rs` before any PR;
- re-records the writer fixture after 08;
- re-renders the Kairo campaign as a smoke test only, never as a test target.

## Governance checkpoints

- After Wave A (00 and 02): the label freeze is unblocked for Discodon, and the split is proven behaviour-neutral.
- After Wave B: whole-trajectory review of the page on S1-S4 against R1-R7. Archive the plan and redraw Waves C-D
  (advisory).
- Before PR: `/prawduct:critic` cumulative.

## Open questions for the owner (recommendation first)

1. **Q1 prompt budget.** *Recommend:* stay within 24,300 by cutting what the view and page now carry (the late
   criteria clause, descriptions the new fields replace). Raise the budget only if 08 cannot fit rule 12's sizing
   clause.
2. **Q2 A19.** *Recommend:* explain the two methods in one line where they disagree. Do not unify the
   campaign-row and contrast interval machines in this plan, because that is a statistics change with its own
   review.
3. **Q3 aggregate reading.** *Recommend:* descriptive only, with intervals, no "shown better/worse" word and no
   tested dominance line on the chart. The per-aspect frontier stays the tested reading.
4. **Q4 long expressions with no display name.** *Recommend:* full expression in code font. Past a length, HTML
   shows its start with a visible ellipsis and the full text one click away. That is not a silent derivation.
   Markdown prints it in full.
5. **Q5 short-name cap.** *Recommend:* 32 characters, not "≤3 words", because a word count is language-bound.
6. **Q6 home of check display names.** *Recommend:* the template (`goal_check_names`), so a name follows its check
   across campaigns. The alternative is per-aspect on the campaign.
7. **Q7 plan split.** *Recommend:* archive this plan after Wave B and draw a writer-side plan for 05-08.
8. **Q8 excerpt matching.** *Recommend:* exact substring after collapsing whitespace runs on both sides, since the
   page prints memo text whitespace-collapsed. No other normalisation.

## Status

- [x] Chunk 00: reporter label form and memo excerpt
- [x] Chunk 01: declared names
- [x] Chunk 02: split and scaffold
- [x] Chunk 03: first screen, caveats once, provenance
- [x] Chunk 04: scorecard and reconciliation
- [x] Chunk 05: degenerate collapse
- [x] Chunk 06: aggregates and chart
- [x] Chunk 07: writer view
- [ ] Chunk 08: writer prompt (then the main session re-records)
