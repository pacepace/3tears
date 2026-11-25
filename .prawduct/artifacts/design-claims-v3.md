# Design: claim-level factuality grading, revision 3: an engine judged instrument (#1287)

Status: revision 3, 2026-11-21. It follows the owner's acceptance (#1287) of `design-claims-boundary.md` §5 and his
two additions that day: a decision register (§13) and DoW built as the second consumer (§12). It replaces §3 to §7 of
`design-claims.md`, revision 2. §1 of that file (A1, the key's context tier) is kept: built on `evals/bp-ctxkey`
`1afd937d4`, durable 17.8. §2 (A2, set grading) is kept and unaffected. Paths are under
`packages/evals/src/threetears/evals/` unless they start with `tests/`, or with `dow:` (the DoW clone, branch
`evals-ladder`). Line numbers are at `0bce2461a`.

## 0. Rulings followed

| Source | Ruling, as this revision applies it |
|---|---|
| D1 | Per unit; no pass/fail, no composite; claim readings never enter a trial's pass |
| D3a / D3b | A false statement is a rate, never a veto. Hedged claims are graded for truth; the ranked rate counts asserted claims |
| D4 | Truth is primary; the candidate's own findings are never evidence (§2). `in_findings` is deferred to its own issue |
| D7 + boundary 5 | Case evidence (key facts, case documents) is authoritative; `run` pages and model memory only nominate; the web decides nominated claims by a verbatim quote; web-confirmed reads `true_ungrounded` (+ `evidence_conflict` if case evidence contradicted it) |
| Boundary 1-8 | Engine judge phase; engine-typed verdicts; "unit"; the four verdicts with `graded_against`/`nominated_by`; truth and groundedness apart; the rubric ranking rule; no key needed; web gated, recorded, never re-searched |
| Review, carried | Repair not refusal (R6); verbatim fold (R8); cite by index (R10); bounded quotes with spans (R9); page budgets, truncation recorded (R7); regime in identity (R3); web budget, `skipped_cap` never a raise (R11); restatements heard, original's verdict (R12); key omission contradicts nothing (R21); per-cell rates over cells (I1); populations stated (R13, R23); built controls |
| Dropped | The ellipsis refusal (B's Critic: it only rejected real ellipses); the `key`/`key+sources` literal (provenances replace it) |

## 1. The instrument: `ClaimCriterion`, on the template

`EvalTemplate.claim_criteria: list[ClaimCriterion]` sits beside `rubric` (`schema/models.py:1828`), `exclude_if`
empty, so a template without one stores and digests as today.

| Field | Meaning |
|---|---|
| `name` | Namespaced like a dim (`rules.claims`); refused on collision with a rubric dim or a host measure prefix |
| `unit` | The reader's noun ("stop", "answer", "entry"). Presentation: outside the fingerprint (`kernel/metrics.py:405`'s rule) |
| `guidance` | The host's notes and examples of what counts as a claim; in the prompts and the fingerprint |
| `web` | `WebBudget(max_queries_per_claim, max_checked_per_cell)`, or `None`: web off, a first-class regime (`web none`) |
| `groundedness` | Declares the groundedness reading; requires `documents` on the evidence |
| `pages` | `PageBudget(per_document_chars, total_chars, window_chars)` |
| `repairs` | `RepairBudget(rounds=2, max_items_per_cell)`; past it is a judge error, never a drop |
| `judge` | `JudgeDeclaration`, `builtin` only in v3 (configured prompts: a later issue) |

`fingerprint_claim_criterion` hashes every field but `unit`, the revision (`claim grade v1`), every prompt and reply
schema (extract, repair, verify, web query, web verdict), attempts, the batch rule, and the fold identity:
`VERBATIM_FOLD` + sha256 of the fold's tables (`_REMOVED`, `_TO_ASCII`, every table it reads) +
`unicodedata.unidata_version`, so a moved Unicode database is a changed definition. Judge config id:
`claims:<name>@<sha12>`, the shape `kernel/evidence_tiers.py:64-70` keys judges by. **Quick-path door:**
`Claims(model, name, *, unit, segments, documents=None, guidance="", web=None, groundedness=False, ...)` in
`quick/claims.py`, a grader in `Eval(grader=[...])` as `Judge` is (`quick/judged.py:244`); its `segments(case, answer)`
and `documents(case, answer)` callables render the evidence the callable kind stores (`quick/one_call.py:740`).

## 2. Evidence: what the kind renders, what the engine reads, the web port

`JudgeEvidence` (`schema/models.py:5797`) gains three fields, each `exclude_if=lambda v: v is None`;
`fingerprint_judged_output` (`:1000`, a dump excluding `inputs`) is then unchanged for every existing evidence (pinned
before the fields land).

| Field | Shape and rule |
|---|---|
| `segments` | `tuple[Segment, ...] \| None`; `Segment(text, unit: int \| None, tags: dict[str, str] = {})`. Units numbered 0..n-1, validated. `unit=None` (a message, a summary line): claims counted, no denominator. `tags`: the host's, opaque, for derived readings (§9), never in a prompt |
| `segmentation` | `str \| None`, the host's name for its cut; required with `segments`; inside the fingerprint, so a changed cut is another output |
| `documents` | `tuple[EvidenceDocument, ...] \| None`; `EvidenceDocument(id, provenance: Literal["case", "run"], title, url \| None, text)`. `()` = the arm fetched nothing (recorded); `None` = not declared (a `groundedness` criterion refuses it); over the stored cap is refused by name (the kind truncates and says so in the title) |

**Kind contract** (docstrings): documents are pages and sources the arm or case author supplied, never the candidate's
own findings (D4); `artifact` is still rendered for rubric dims. **Key facts** are read by the engine, never rendered:
`reference_facts(case)` in `kernel/reference.py`, key-type-neutral (an `entity_set` key yields `fact_entities`' facts,
A1; no key yields none). R20 (a key needs one core entity) stays a rule for set keys only.

**`WebSearch`**, a host port on `RunnerOptions` (`run/runner.py:185`) as `web_search: WebSearch | None`: `identity`,
`async search(query) -> tuple[WebPage(url, title, text, snippet_only), ...]`, `price_ceiling(query)`. A criterion with
`web` and no port is refused at launch, before spend; the quick path takes `web=` on `Claims`. Each search is a
`judge`-role usage row with its cost and no tokens (C4 confirms `CallUsage` carries one, or adds the field).

## 3. The judge-phase flow

The judge phase (`run/runner.py:1947`) runs each criterion after the rubric dims, on the same context
(`build_judge_context`, `:3402`) and judge client factory (`judge_clients_for_run`).

**Priced up front.** `_judging_ceiling` (`:923`) adds each criterion's ceiling; `_JudgingGate` (`:804`) admits the
phase whole. Ceiling = extraction (3 attempts) + verification (one batch per unit plus one for unitless segments, 3
attempts, evidence at `pages.total_chars`, claims at the extraction reply cap) + repairs (`max_items_per_cell` ×
`rounds`) + web (`max_checked_per_cell` × (query call + `max_queries_per_claim` × the port's ceiling + verdict call)).
Any unpriced part takes the existing one-call-at-a-time path (`_judge_one_call_at_a_time`, `:2056`).

| Step | What happens |
|---|---|
| 1 Extract (chunk B reworked, §11) | Sees the segments only: never key, documents, findings or web. Per claim: `claim` (one proposition), `excerpt` with offsets in its segment (the segment's own slice, found by `verbatim_span`), `specific`, `hedged`, `restates`, and new **`attributed`**: the text reports someone else's claim (a character's rumour, "the SRD says"), found by the DoW proof (§12), graded like `hedged`: for truth, outside the asserted counts. A failing item is repaired, two rounds, then a judge error; never dropped |
| 2 Verify | One call per unit batch plus one for unitless segments; a restatement is batched with its original and takes its verdict in code (R12). Evidence numbered: key facts `K1..` (entity, claim, source_url), passages `S1..` labelled by provenance, chosen by code under `pages` (windows around folded key names, aliases and the batch's excerpts, document order; truncation recorded). Per claim the verifier returns `citations: [{ref, stance: supports\|contradicts, quote, subject_span, value_span}]` and `suspect: bool` (nothing covers it and it doubts it), never a verdict. A quote is one sentence or table row, ≤ 300 chars, verbatim in the full stored text, both spans verbatim inside (R9); a contradiction quotes the conflicting value; a `K` citation needs none. The prompt says a fact the key omits contradicts nothing (R21). Code checks refs and quotes (`verbatim_span`); a failure is a repair item |
| Step-1 finding (code) | Case contradiction → `nominated_by=case`; only a run contradiction → `run`; no citation and `suspect` → `judge`; case support, no contradiction → `supported`; case support and contradiction → nominated, `evidence_conflict`; else `unverifiable`. `grounded` = a `run` document supports it (read by groundedness only) |
| 3 Web | Nominated claims only, in claim order, up to `max_checked_per_cell`; the judge writes ≤ `max_queries_per_claim` queries; verdict `true`/`false`/`inconclusive`; a decisive one needs a quote with spans verbatim in a returned page (its snippet when `snippet_only`), else a repair item |

| Nominated by | Web `true` | Web `false` | Inconclusive, `skipped_cap`, or web off |
|---|---|---|---|
| `case` | `true_ungrounded`, `evidence_conflict` | `contradicted`, against `case` | `contradicted`, against `case` |
| `run` or `judge` | `true_ungrounded` | `contradicted`, against `web` | `unverifiable` |

Past the cap is `skipped_cap`, counted, never an apparatus raise (R11: that would drop the most false-heavy cells).
**Failures** (cut short, unreadable after retries, repair budget spent, port error) are `ClaimGradingFailed`, a judge
error on that criterion: the cell is excluded as `judge_error`, which a re-judge recovers without re-buying the
candidate (`:482-486`). Never an `ApparatusError` in the phase, which fails the run (`:2031`); a replay miss stays one.

## 4. Stored shapes and schema steps

**`ClaimGrade`** on `EvalTrace.claim_grades[criterion]` (bulk): criterion fingerprint; per call its role (extract,
verify, repair, web), `served_model`, `judge_temperature`, `judge_sent_settings`, as on `RubricScore`
(`schema/models.py:1143`); segmentation; the evidence index (ref → provenance, document id or fact, passage offsets);
truncation, sources count, characters read; `claims: [GradedClaim]` (segment, unit, claim, excerpt, start, end, the
four tags, citations with quote offsets in the full document, `suspect`, `nominated_by`, `verdict`, `graded_against`,
`grounded`, `evidence_conflict`, `web: checked|skipped_cap|none`); `web: [WebCheckRecord(claim, queries, urls per
query, outcome, deciding url, quote, spans)]`. No page text is copied in; it is on the evidence.

**`ClaimTally`** on `EvalResult.claim_tallies[criterion]` (small; every bundle walk reads it): criterion fingerprint;
the verifying calls' judge-key fields (`served_model`, temperature; `None` when calls disagree, matching nothing);
`run_documents_declared`; counts (units, claims, asserted, asserted_specific, asserted_contradicted,
hedged_contradicted, attributed_contradicted, contradicted, ungrounded_specific, unverifiable, true_ungrounded,
evidence_conflicts, web_checked, web_skipped). One pure function derives it from the grade; a re-judge rewrites both.
A failed criterion lands on the existing judge-error fields under its name.

| Schema step (ledger line, MIGRATIONS, reach-back as 17.7/17.8: refused at seed by name; kept opaquely when newer) | Fields | Chunk |
|---|---|---|
| Durable 17.9 | `EvalTemplate.claim_criteria` | C1 |
| Core 15.24 (`schema/versioning.py:509`) | `JudgeEvidence.segments`/`segmentation`/`documents`; `EvalTrace.claim_grades`; `EvalResult.claim_tallies` | C1 |
| Durable 17.10 | `CalibrationRating.claim_ref`; `BlindOutput.claim` | G |

## 5. Readings and their family

A seventh engine family, `claims` (`graded_by="judge"`), joins `ENGINE_FAMILIES` (`kernel/metrics.py:380`), outside
`CODE_GRADED_FAMILIES`, so the bundle's measure surfaces never rank it (`analysis/bundle/measures.py:182`). This is the
"decision to record" `metrics.py:2345` asks for; `tests/test_open_family.py:89`'s pinned set grows by it on that
record. `describe_claim_reading(criterion, reading)` describes each (population, evidence, a cell's value is a point),
as `describe_rubric_dim` does; `claim_readings(tally)` computes them on read, so a new diagnostic needs no re-measure.

| Reading | Definition | Population | Direction |
|---|---|---|---|
| R1 `<c>_false_per_unit` (truth) | Asserted contradicted claims per unit, each restatement counted as heard | Cells with ≥ 1 unit | Lower; ranked |
| R2 `<c>_ungrounded_share` (only with `groundedness`) | Of asserted specific claims, the share no `run` document supports | Cells with an asserted specific claim and documents declared | Lower; ranked |
| Diagnostics | `_false`, `_false_hedged`, `_false_attributed`, `_false_share`, `_claims`, `_units`, `_hedged_share`, `_unverifiable`, `_true_ungrounded`, `_evidence_conflicts`, `_web_checked`, `_web_skipped` | Stated per reading | None |

Neither enters a pass. R1's range is the instrument's error against people, from its checks (§7), not the interval over
cells. **Built controls** (scripted clients, hand-built grades), each moving only its own reading: a planted lie, a
hedged lie, an attributed lie, a bland set, a repetitive set, an invented vivid detail, and a canon contradiction with
web off (DoW's regime).

## 6. Ranking through the rubric rule, keyed to the instrument

Claim readings rank as judged readings do: counted, qualified by the instrument's checks, set aside when a check is
`not_met` or it leans. None enters a trial's pass (`kernel/scoring.py:410`). The instrument's `JudgeKey`
(`analysis/agreement.py:116`): `dim` = `<c>.true` or `<c>.grounded` (§7's questions), `scale` = `pass_fail`,
`config` = `claims:<c>@<sha12>`, served model, temperature and settings from the tally. What must learn it:

| Where | Change |
|---|---|
| `kernel/campaign.py:89` `ReadingKind` | Add `"claims"` |
| `analysis/bundle/comparisons.py` `_per_case_values` (`:160-200`), family builder (`:330-345`) | `("claims", <reading>)`: per-case mean of per-cell readings over non-faulted results; described by `describe_claim_reading`; diagnostics skipped |
| `analysis/bundle/cell_reads.py` `judged_measures` (`:151-226`) | Per-cell claim summaries carrying the instrument's key |
| `analysis/report_input.py` `Judged` (`:590`) | `instrument: str \| None` (`exclude_if` None); one entry per ranked reading; `checks` are the instrument's |
| `analysis/caveats.py` `_checks` (`:858-885`), implied `judge_unstated`, `judge_partly_uncalibrated` | Raised once per instrument, reaching its readings |
| `analysis/first_screen.py` `RANKS` (`:97-112`) | Unchanged; a test pins a claims caveat at rank 5 or 6 |
| `instrument_*` set-aside producer (`schema/reading_states.py:54-57`; design-judges §5.5, unbuilt) | Keys on `JudgeKey`, so it covers claims by construction |
| `kernel/declaration.py` `resolve_bar_name` (`:1425-1470`) | A claims reading resolves after judged dims, so a bar may name `<c>_false_per_unit` |
| `ops/regressions.py:20-25` | Fires on a claims reading only when its instrument is calibrated |

## 7. Calibration: a claim item in the check record and the blind read

Two pass/fail questions per claim, so the per-rater kappa (`kernel/evidence_tiers.py`) and the 2×2
`BlindVerdictTable` (`schema/blind_read.py:199`) apply unchanged: `<c>.true` (pass = not false; a person may search
the web) and `<c>.grounded` (pass = the documents shown support it). The instrument's answers come from the stored
verdict (`contradicted` fails `<c>.true`; `grounded` answers `<c>.grounded`). A third, `<c>.is_claim`, checks the
extractor's precision. **Blind read:** `BlindOutput.claim: ClaimItemView | None` (`schema/blind_read.py:100`) carries
the claim, excerpt and offsets, the segment text and the numbered evidence the verifier read: no verdict, citation or
arm. `draw_blind_read` (`analysis/blind_read.py`) draws claims at random under its seed with `planned_items`. The label
key's output fingerprint is the view's digest, so a label carries to a byte-identical item; `CalibrationRating.claim_ref`
(grade digest + position) names the claim. **Checks** (`analysis/checks.py:37`) keep their shape: `instrument` = the
criterion, `criterion` = the question, `against` = people, a model, or itself; the reference-reviewer guards (addendum
§5) apply as written. `run/judge_repeat.py` learns criteria: it re-verifies stored claims, extraction fixed, web
verdict copied: the check against itself. Discodon's agreement run (#3526) stays host-side and writes engine ratings.
**Left out, named in the CHANGELOG:** an extraction recall check (raters list misses; not checked), judge cases and
campaigns for claims, stratified draws, a corrected contrast.

## 8. Re-judge and replay

`reproducible_judge_inputs` (`run/rejudge.py:103`) learns criteria: it loads the stored evidence and refuses when
today's criterion fingerprints otherwise than the tally recorded. A failed criterion re-runs whole, web included
(nothing it replaces was a verdict); a completed grade is never re-searched (repeats and second judges copy its web
record). Extract, verify, repair and web calls are judge-client completions, so `RecordedCompletions`
(`testing/recorded_completions.py:369`) records and replays them; a new `testing/` `RecordedWebSearch` replays the port
by (identity, query), a miss being a `ReplayMiss`. Checks over replayed grades read `not available: replayed`
(`schema/reading_states.py:74`).

## 9. Derived host readings (`derived_from`)

A host reading over the grade (DoW's NPC contradictions; discodon's R3 track-relevant specifics and R4 worst
repetition) is a `MetricDescriptor` with `derived_from: <criterion>` (`exclude_if` None), computed by a pure
`ClaimReader` on the `HostProfile`, `(ClaimGrade, segments) -> Mapping[str, float | bool]`, synchronous like
`MeasureRereader` (`kernel/host/measures.py:69`), reading `Segment.tags`. The engine calls it after the grade lands and
on re-judge and re-measure; the values go on `host_measures`. The registry refuses `derived_from` in a code-graded
family; the reader's identity enters the descriptor's `formula`. Its values take the `"claims"` reading kind and the
instrument's key, so they rank and are qualified exactly as R1 is.

## 10. The `failed_checks` / `graded_by` fix (independent; first)

`failed_checks` (`kernel/scoring.py:380-389`) and `is_check` (`kernel/metrics.py:598`) ignore `graded_by`, so a
judge-graded host boolean or pass/fail measure with a direction fails trials today and moves the ranked pass rate.
Fix: `MeasureRegistry` refuses one at registration (`MeasureRegistrationError`, naming the rule), and `failed_checks`
also skips any measure `is_code_graded` rejects. Tests: the refusal, and pass^k unmoved by a judged host boolean. For
hosts: such a host is refused at startup; declare the family `graded_by="code"` if code made the number, or drop the
direction.

## 11. What chunk B (`evals/bp-claimx` `f350cd5f0`) must change

| Change | Detail |
|---|---|
| Move, hide | `kernel/claim_extraction.py` → `kernel/claims/extraction.py`, engine-internal; off the host surface (`surface_tiers.json`, `api/surface.json`, `tests/enforcement/_public_api_consumers.txt`); the vocabulary registers follow the path; the door is `Claims` |
| Rename | `Utterance` → `Segment` (into `schema`, beside `JudgeEvidence`); `stop_index` → `unit` (on `ExtractedClaim` too); `_check_stops` → `_check_units`; "stops" → "units", "speaker"/"what a speaker said" → "the text" in docstrings, prompts (`:89`, `:100-101`) and errors |
| Generic rules (`:64-84`) | Keep the claim definition, one proposition per claim, implied framing and sweeping causal claims. Into discodon's `guidance`: "who made or took part… a first, a record", "introducing a work as someone's", "how something sounds", "talk about the programme itself", "echoes of the request", "a credit", the hedge phrases |
| Add | The `attributed` tag, its prompt line and tests |
| Drop | The ellipsis refusal (`is_elided` in the excerpt checks and its fault text); the fold itself is unchanged |
| Fold identity | `VERBATIM_FOLD` + sha256 of the fold tables + `unicodedata.unidata_version`, computed once, in the identity, pinned by a test that changes a table |
| Error type | `ClaimExtractionFailed(ApparatusError)` → `ClaimGradingFailed(Exception)`, recorded by the phase (§3) |
| Keep | The repair loop, offsets, `judge` ledger and digest. Rewrite the CHANGELOG entry; provisional until C4 |

## 12. The second consumer: Dungeons of Wagons, designed in and built

DoW consumes the quick path (`Eval(cases, runner=..., grader=[...])`, `dow:evals/forge.py:112`,
`dow:evals/router.py:76`) with recordings replayed offline in pytest. Two surfaces, opposite regimes:

| | (a) Rules answers: `rules.claims` | (b) Scout drafts: `scout.claims` |
|---|---|---|
| Cases | Out-of-character rules questions with the asker's sheet | Briefs and a starting place over The Silver Hollow |
| Runner | `Answerer.answer` (`dow:src/dow/gm/answering.py:83-109`), two arms: A as today (player view only); B runs `SrdIndex.search(query, limit=3)` (`dow:src/dow/lookup/index.py:238`, `rules_lookup`'s search) first and adds the passages. The product question: should an answer check the SRD first? | The scout's draft call (`dow:src/dow/gm/scouting.py:114`) on `scout_context(...)` (`:126`) |
| Segments | One unit per answer (noun "answer", `segmentation="answer-v1"`) | From the structured `ScoutDraft` (`:31-80`): summary `unit=None`; place, each area, each NPC a unit (noun "entry"); each description and fact a segment; `tags` carry the entity id |
| Documents | `case`: the SRD sections the case names (index chunks, heading path as title), the asker's player view, the house rules in force (`render_house_rules`, so a house-rule truth is case-supported before any web check). `run`: arm B's passages; `()` for arm A | `case` only: the module (`dow:content/modules/silver-hollow.yaml`) and the context the scout was shown |
| Web | On (the SRD is public); DoW supplies a `WebSearch` adapter; the rung replays a recording; the one live recording needs owner spend approval | Off (`web none`): a fictional world has no web evidence; suspects read `unverifiable` |
| Ranked | `rules.claims_false_per_unit`, `rules.claims_ungrounded_share` (arm A reads high, as it should) | `scout.claims_false_per_unit`, described as canon contradictions per entry; groundedness not declared (invention is the scout's job); derived `scout.npc_contradictions` via `derived_from` |
| Guidance | 5e rules statements, numbers and conditions are claims | In-world statements are claims about the fiction; a character's rumour is attributed |

**Boundary defects DoW exposed, fixed in the engine shape above:** the `attributed` tag; web off as a first-class
regime; a truth description that names its evidence ("contradicted by the case's evidence or the web"), so it reads as
canon in fiction; segments from structured output; a case with case documents and no key. DoW needs no discodon
concept.

## 13. Decision register

★ marks a decision that is hard to reverse, for the owner. The reversibility codes:

- **S:** stored shape (a migration);
- **N:** public name (a breaking change);
- **I:** measure identity (old and new runs never pool);
- **D:** doctrine (every reader changes).

| # | Decision | Effect on value and maintainability | Rev. | A non-discodon consumer sees | Rejected |
|---|---|---|---|---|---|
| 1★ | The engine runs claim grading in the judge phase | Ranking, re-judge, repeats and checks come with it; the judge phase gains a second instrument type | D,S | A criterion; no grading code | Helper in `invoke`: never ranks; a failure loses the output |
| 2★ | `ClaimCriterion` is its own type, not a `RubricDim` | A clean fingerprint and budgets; one more durable type | S,N | One more grader | Optional claim fields on `RubricDim` |
| 3★ | Segments and documents live on `JudgeEvidence`, fingerprinted | Re-judge is reproducible; labels key on what was read; traces grow by the page text | S | Two render callables | Documents kept apart: labels carry across changed evidence |
| 4★ | Verdicts: supported, contradicted, unverifiable, true_ungrounded | One vocabulary for every host; renaming one later is a migration | S,N | Four words for RAG, fiction and research | Host-defined verdicts: no shared readings or checks |
| 5★ | Provenance is `case`, `run` or `web` | Arm-dependent evidence is visible; a fourth would be additive | S | Retrievals are `run`; a source document is `case` | One "sources" bag |
| 6★ | Truth and groundedness are separate readings, and `run` only nominates | Arms cannot buy truth by fetching more | I,D | Picks either or both | One unsupported share over mixed provenance |
| 7 | Units, with unitless segments counted outside the denominator | Per-unit rates for any cut | N | "answer", "entry" | "stop" |
| 8★ | Per-cell rates averaged over cells (I1) | Wide intervals at few cases | I | Same | Claims pooled across cells, though they cluster |
| 9★ | Readings derived on read from a stored tally | A diagnostic is free; a changed definition is a new identity | S,I | — | Readings stored on `host_measures` |
| 10★ | The `claims` engine family, judge-graded | One ranking rule for judged numbers | D | Ranks like its rubric dims | A host family naming its judge |
| 11★ | The rubric rule: counted, qualified, set aside | No second rule; an unchecked reading ranks, with a caveat | D | Same rule as its dims | Rank only once a check is met |
| 12★ | Calibration by two pass/fail questions per claim | Reuses kappa, the 2×2 and the blind read; four-way agreement is lost | S | The existing sheet | A four-class nominal scale and new statistics |
| 13 | `derived_from` with a pure `ClaimReader` | Domain readings inherit the instrument's checks | N | DoW's NPC reading | Counts computed in `invoke` |
| 14★ | Web search is a host port, recorded, never re-searched | Grades stay stationary, and replay works | N,S | A port, or web off | Model memory decides; re-searching |
| 15 | Failures are judge errors, not apparatus errors | Recoverable without re-buying the candidate; rubric readings wait for the re-judge | D | Same | `ApparatusError`: the output is lost |
| 16 | A whole-phase ceiling and a repair budget | The cap holds; a very long output can hit the budget | I | Sets budgets | Admission call by call |
| 17 | The fold identity includes its tables and the Unicode version | A Python upgrade splits runs, honestly | I | Invisible | The fold's name alone |
| 18 | No key required; facts read through a type-neutral accessor | Serves cases with no answer set | — | Summarizers, RAG | An `entity_set` key required (R20) |
| 19 | An `attributed` tag | Rumours and quotes are not asserted claims | I,S | Fiction and RAG need it | Folded into `hedged` |
| 20 | A judge-graded host check is refused at registration | Closes a ranking hole; may refuse an existing measure | D | A clear startup error | A silent filter |

## 14. Chunks (one risk surface each)

| Chunk | Risk surface | After |
|---|---|---|
| F0 | The `failed_checks` / `graded_by` fix (§10) | — |
| B′ | Chunk B rework (§11): library only, nothing calls it | — |
| C1 | Schema: the criterion (durable 17.9); evidence, grade and tally (core 15.24); fingerprint pins; reach-back. No behaviour | A1 |
| C2 | Readings: the `claims` family, `describe_claim_reading`, `claim_readings`, the built controls | C1 |
| C3 | Verifier library: numbering, page selection, citations and spans, repair, the step-1 finding, restatements | B′, C1 |
| C4 | The judge phase: pricing, the gate, errors, spend rows, re-judge, recorded replay, the quick path's `Claims` | C2, C3 |
| D | Web: the port, the step, budgets, `skipped_cap`, `RecordedWebSearch`, pricing | C4 |
| E1 | Ranking: `ReadingKind`, comparisons, cell reads, report input, caveats, bars, regressions, the first-screen pin | C2, C4 |
| E2 | Derived readings: `derived_from`, `ClaimReader`, re-measure | E1 |
| G | Calibration: claim items in the blind read, `claim_ref` (durable 17.10), checks, judge repeats | C4, E1 |
| R | DoW rungs on `evals-ladder`: `evals/rules_claims.py` and `evals/scout_claims.py`, with cases and recordings, replayed offline in pytest like rungs 0 to 3, a ladder-doc section, and the family pinned to one exact version | D, E2, G |
| H | Discodon adoption (host-side) | R |

**Nothing is done until R is green.** Both DoW rungs replay offline, rank their arms, and raise their check caveats,
all without importing anything the engine took from discodon. Any such need is fixed in the engine first.

Each chunk gets one builder (Opus), scripted clients and no spend. It carries its CHANGELOG and MIGRATIONS, and a
Critic reviews it.

## 15. Not verified

- Whether kappa on `<c>.true`, drawn at random at 8 to 12% prevalence, is informative at realistic item counts, or
  whether stratified draws must come first. This is the statistician's question.
- Whether `CallUsage` can carry a search row (C4).
- Whether the stored document cap fits discodon's pages (C1).
