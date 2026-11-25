# Design: claim-level factuality grading, revision 4: an engine judged instrument (#1287)

Status: revision 4, 2026-11-21. It answers `design-claims-v3-review.md` (5 blocking, 9 should-fix, 6 notes; all
dispositioned in §14) and two owner rulings that day: **run pages never nominate** (truth uses only what every arm
shares), and **groundedness is a ranked reading**, higher is better. `design-claims-v3.md` is unchanged. This replaces
§3 to §7 of `design-claims.md` (rev 2); its §1 (A1, the context tier, `evals/bp-ctxkey` `1afd937d4`, durable 17.8) and
§2 (A2) stand. Paths are under `packages/evals/src/threetears/evals/` unless they start `tests/` or `dow:` (DoW clone,
`evals-ladder` at `ffd982f`). Line numbers at `0bce2461a`. **∆S** marks the session-scope addition (discodon #3526); **∆V** the fixes from the review's "v4 verification" (V1 to V8).

## 1. Rulings followed

| Source | Ruling, as applied |
|---|---|
| D1 | Per unit; no pass/fail, no composite. Claim readings never enter a trial's pass, and their loss never moves it (§4) |
| D3a / D3b | A false statement is a rate, never a veto. Hedged and attributed claims are graded for truth, outside the asserted counts |
| D4 | Truth is primary; the candidate's own findings are never evidence. `in_findings` is deferred to its own issue |
| D7, as ruled 2026-11-21 | Case evidence (key facts, case documents) is authoritative and **never overruled**: case support reads `supported`, case contradiction reads `contradicted`. The arm's own (`run`) pages never nominate and are never shown to the truth step. Nomination is the grader's own suspicion only, and the web decides a nominated claim by a verbatim quote. Model memory never decides |
| Owner, 2026-11-21 | Groundedness is ranked, higher is better: an arm whose own research supports what it says ranks above one that must be checked after the fact. It is a separate reading, never a composite |
| Boundary 1-8 | The engine judge phase; engine-typed verdicts; "unit"; four verdicts with `graded_against` and `nominated_by`; truth and groundedness apart; the rubric ranking rule; no key needed; web gated by the cap, recorded, never re-searched |
| Carried from the rev-2 review | Repair, not refusal (R6); the verbatim fold (R8); cite by index (R10); quotes bounded with spans (R9); page budgets with truncation recorded (R7); regime in the identity (R3); web budget, `skipped_cap` never a raise (R11); restatements counted as heard, taking their original's verdict (R12); a key omission contradicts nothing (R21); per-cell rates over cells (I1); populations stated; built controls |
| Dropped | The ellipsis refusal, from `verbatim_span` itself (§10); v3's run nomination and the web overruling case evidence |

## 2. The instrument: `ClaimCriterion`, on the template

`EvalTemplate.claim_criteria: list[ClaimCriterion]` sits beside `rubric` (`schema/models.py:1828`), `exclude_if` empty.

| Field | Meaning |
|---|---|
| `name` | Namespaced (`rules.claims`). Its readings live in the judged namespace (§7), so the name and every reading name are refused on collision with a rubric dim, another criterion or a host measure |
| **∆V6** unit noun | Not a criterion field: a top-level presentation dict `EvalTemplate.claim_unit_names: dict[criterion, str]` ("stop", "answer", "entry"), listed in both `TEMPLATE_PRESENTATION_FIELDS` and `_UNMEASURED_TEMPLATE_FIELDS` (`run/definition_digests.py:60-73`), the `goal_check_names` precedent (`schema/models.py:1829-1840`): a nested field is unreachable by that set, and a renamed noun must not read as drift |
| `guidance` | The host's notes and examples; in the prompts and the fingerprint |
| `grounded_in` | `frozenset[Provenance] \| None`. `None`: no groundedness reading. Else the documents the **candidate itself was given**: `{run}` for research and RAG, `{case}` for a summarizer's source. Never case evidence the candidate did not see (kind contract) |
| `web` | `WebBudget(max_queries_per_claim, max_checked_per_cell, results_per_query, page_chars, window_chars)`, or `None`: web off, a first-class regime (`web none`) |
| `pages` / `repairs` | `PageBudget(per_document_chars, total_chars, window_chars)` / `RepairBudget(rounds=2, max_items_per_cell)`, past which is the criterion's loss, never a drop |
| `judge` | `JudgeDeclaration`, `builtin` only in v4 |

The fingerprint (built in C4, after B′) hashes every field (∆V6: the unit noun is no longer one), the revision `claim grade v1`, every prompt and
reply schema, attempts, the batch rule and the fold identity (`VERBATIM_FOLD` + sha256 of its tables +
`unicodedata.unidata_version`). Judge config id: `claims:<name>@<sha12>`. **Quick-path door** (chunk Q):
`Claims(model, name, *, unit, segments, case_documents=None, grounded_in=None, web=None, guidance="", ...)`, a grader
in `Eval(grader=[...])` as `Judge` is (`quick/judged.py:244`). `run` documents reach it only through a new side
channel, `Answer.documents` (`quick/answer.py:25`), so fetched pages stay out of `Answer.value`, the output every grader and judge reads.

## 3. Evidence, beside `JudgeEvidence`

**Verified:** `fingerprint_judged_output` hashes `evidence.model_dump(mode="json", exclude={"inputs"})`
(`schema/models.py:1021`), while the rubric judge reads only `subject`, `case_material` and `artifact`
(`run/judge_service.py:1036-1039`). A set field added to `JudgeEvidence` would move every rubric label key of an
adopting kind, differ across builds (a kept-opaque field is in no dump, `schema/base.py:354-356`), and ship pages
through `load_judge_evidence` (`kernel/storage.py:141-150`, `:2865`) to every bundle and blind read. **So
`JudgeEvidence` is untouched**, and the claims evidence is its own record: `ClaimEvidence(segmentation, segments,
documents)`, rendered on `CandidateOutput.claim_evidence`, stored on `EvalTrace.claim_evidence` outside
`_JUDGE_EVIDENCE_FIELDS`, loaded by trace id only by the claims phase, the re-judge, the claim read and the derived readings. The grade stores `claim_evidence_digest`.

| Shape | Rule |
|---|---|
| **∆S** `ClaimEvidence.session: VerbatimJsonObject \| None` | The run's own record (a play order, a game state), host-shaped, stored, in `claim_evidence_digest`; the engine never reads it and no prompt sees it. Only a `ClaimReader` does (§9) |
| `Segment(text, unit: int \| None, tags: dict[str, str] = {})` | Units numbered 0..n-1; `unit=None` claims counted, no denominator; `tags` the host's, opaque, never in a prompt |
| `EvidenceDocument(id, provenance: Literal["case", "run"], title, url \| None, text)` | `documents=()` = the arm fetched nothing (recorded); a `grounded_in` criterion refuses undeclared documents |
| Cap (**∆V5**) | The serialized claims payload of one trace (documents + grades + web windows) is capped at `CLAIM_TRACE_BYTES` = 3 MiB, under the store row limit hosts must provide (named in "For hosts"); documents get at most 1 MiB of it. `WebBudget` validation refuses a budget whose `max_checked_per_cell × max_queries_per_claim × results_per_query × page_chars` exceeds the window share. The kind truncates documents to fit and says so in the title; still over is the criterion's loss `evidence_over_cap`, counted per arm; the engine never truncates. **A failed trace write** for a claim-bearing cell is raised into the criterion, not only logged (`kernel/storage.py:2782-2797`): the trace is written first, and on failure the result carries `claim_failures[c] = trace_lost` and no tally, so no reading rests on a grade that is not stored |
| Key facts | Read by the engine, never rendered: `reference_facts(case)` (`kernel/reference.py`), key-type-neutral (`entity_set` yields `fact_entities`; no key, none). A1 refuses a context-only key (`_one_entity_per_name`), so a fact-only case uses case documents |
| `WebSearch` port on `RunnerOptions` (`run/runner.py:185`) | `identity`; `async search(query) -> tuple[WebPage(url, title, text, snippet_only), ...]`; `price_ceiling() -> float \| None` per query (the gate prices before any query exists). No port with `web` set, or an unpriced port under an enforced cap without a declared `usd_per_query`, is refused at launch (an unpriced phase judges the run serially, `run/runner.py:2063-2087`). Searches land as a `RoleUsage` row: role `judge` (never `external`, which counts as production-replicating, `schema/models.py:5345-5348`), provider = port identity, `provider_unit="query"`, `provider_units`, `call_count`, no tokens |

## 4. The judge-phase flow

**∆V1 Entry.** A cell carrying `claim_evidence` enters the claims phase whatever its `judged_artifact`: today judging starts only on `judge_evidence` (`run/runner.py:1886-1887`) and the quick path declares `UNJUDGED` with no `Judge` (`quick/one_call.py:880`), so a `grader=[Claims(...)]` eval would grade nothing. C1 makes the trace validator (`schema/models.py:5925-5937`), `hold_to_declaration` (`run/runner.py:2172`) and `judge_witnessed_output` admit `claim_evidence` without `judge_evidence`; a run with a criterion requires a judge client at launch, refused by name otherwise. The flow runs after the rubric dims (if any), on the same context and judge client factory (`run/runner.py:1947`, `:3402`).
**Separate holds** (`_JudgingGate`, `:804`): the rubric dims are one hold, each criterion its own; a criterion the cap
sets aside is its own loss (`cost_cap`), never the rubric dims'. Each criterion's ceiling (`_judging_ceiling`, `:923`)
is priced from known bounds: extraction (3 attempts) + truth and ground batches (3 attempts each, at the page budget
and the extraction reply cap) + repairs (`max_items_per_cell × rounds`) + web (`max_checked_per_cell` × (query call +
per-query price × `max_queries_per_claim` + a verdict call over `results_per_query × page_chars`)).

| Step | What it sees | What happens |
|---|---|---|
| 1 Extract (B′, §10) | Segments only | Per claim: `claim`, `excerpt` with offsets (the segment's own slice, found by `verbatim_span`), and the tags `specific`, `hedged`, `attributed` (∆A: the text reports a claim it does not vouch for: a rumour, a character's or another party's statement, "some say". A statement the text makes and backs with a source, such as "the SRD says" or "according to its liner notes", is **asserted**, since citing research is what groundedness rewards), `restates`, and **∆S `scope`**: `world` or `session` (§4, below). A failing item is repaired, up to two rounds |
| 2a Truth verify | Key facts `K1..` and `case` passages `S1..`: the same for every arm. Never `run` pages | One call per unit batch, plus one for unitless segments; a restatement rides with its original (R12). Per claim: `citations [{ref, stance, quote, subject_span, value_span}]` and `suspect` (nothing covers it, and the grader doubts it). A quote is one sentence or row, ≤ 300 chars, verbatim in the full text, with spans verbatim inside it (R9). A contradiction quotes the value. The prompt says an omission contradicts nothing (R21). Passages are chosen by code under `pages`. Bad refs or quotes become repair items |
| 2b Ground verify (when `grounded_in`) | Only the `grounded_in` documents | The same call shape, stance `supports` only. `grounded` = a cited, verbatim-checked support |
| 3 Web | Nominated claims, in claim order, up to `max_checked_per_cell` | ≤ `max_queries_per_claim` queries, `results_per_query` pages each, windows cut by code to `page_chars`. Verdict `true` / `false` / `inconclusive`. A decisive verdict needs a quote with spans, verbatim in a window, else it is a repair item. The windows read are stored |

**Truth verdict**, derived in code from step 2a (and step 3):

| Step 2a | Verdict | `graded_against` |
|---|---|---|
| Any case support | `supported`; `evidence_conflict` if a case citation also contradicts | `case` |
| A case contradiction and no support | `contradicted` | `case` |
| No case citation, `suspect` (`nominated_by=judge`) | web `true`: `true_ungrounded`; `false`: `contradicted`; inconclusive, `skipped_cap` or off: `unverifiable` | `web` (none if unverifiable) |
| Anything else | `unverifiable` | none |

The web never overrules case evidence, and truth sees no arm's pages: one detector for every arm, whose only arm-dependent part, the grader's suspicion, calibration measures per arm (§8).

**∆S Session claims.** A `session` claim is about the output or run itself: a back-announce ("That was X, Y, off Z"
→ "Y just played"), DoW's "you rolled a 14" or "you're at 7 hit points", a RAG answer's "as noted above". It skips
steps 2a, 2b and 3, stays out of R1, R1b, R2 and every world diagnostic, and stays on the grade with its excerpt for a
host's derived reading against `ClaimEvidence.session` (§9). The extractor splits a mixed line ("That was X, Y, off Z"
→ `session` "Y just played" + `world` "Y is on Z"; the world part is graded). A tag, because only the extractor sees
inside the sentence; an optional opaque record, because a derived reading must recompute from stored inputs and a play
log or game state does not fit per-segment tags.

**Failures** (`ClaimGradingFailed`: cut short, unreadable after retries, repair budget spent, port error, cost cap,
evidence cap) are recorded per criterion on `EvalResult.claim_failures[criterion]`, **never** as `judge_error`, so
`trial_exclusion`, the pass rate and the rubric readings keep the cell (`run/runner.py:482-486` untouched); the cell
leaves only that criterion's readings, and bundles count each criterion's losses per arm. A replay miss stays an `ApparatusError` and fails the run (#1162).

## 5. Stored shapes and schema steps

**`ClaimGrade`** on `EvalTrace.claim_grades[criterion]`: criterion fingerprint, `claim_evidence_digest`; every call
(role, `served_model`, `judge_temperature`, `judge_sent_settings`); the evidence index (ref → provenance, document id
or fact, offsets) and truncation; `claims: [GradedClaim]` (segment, unit, claim, excerpt, offsets, tags, truth and
ground citations, `scope` (∆S), `suspect`, `nominated_by`, `verdict`, `graded_against`, `grounded`, `evidence_conflict`,
`web: checked|skipped_cap|none`, and `decided_by`, the call whose answer fixed the verdict); `web: [WebCheckRecord(claim, queries, urls, windows read, outcome, deciding url, quote, spans)]`.

**`ClaimTally`** on `EvalResult.claim_tallies[criterion]`: criterion fingerprint, `documents_declared`, and
`by_judge: [{judge key fields, counts}]`, one entry per deciding call's key, so a fallback-served retry is one more key,
not an unknown judge. Counts (world claims; ∆S plus `session_claims`; ∆V2 plus `units_with_false`): units, claims, asserted, asserted_specific, asserted_contradicted, hedged/attributed
contradicted, grounded_specific, unverifiable, true_ungrounded, web_decided, evidence_conflicts, web_checked,
web_skipped. One pure function derives it from the grade; a cell whose entries hold two keys reads its arm's judge identity as `split`.

| Step (ledger line, MIGRATIONS, reach-back as 17.7/17.8) | Fields | Chunk |
|---|---|---|
| Durable 17.9 | `EvalTemplate.claim_criteria` | C1 |
| Core 15.24 (`schema/versioning.py:509`) | `EvalTrace.claim_evidence`, `claim_grades`; `EvalResult.claim_tallies`, `claim_failures` | C1 |
| Durable 17.10 | `ClaimRead` (§8); `CalibrationRating.claim_ref` | G |

An older build keeps newer fields opaquely and **refuses to launch** from a template carrying them, naming them (it
would otherwise run without grading claims, under a template digest the newer build does not share). C1 pins, before
the fields land and again after (unchanged): `fingerprint_judged_output` of a stored evidence, `measured_template_digest`
of a template without criteria, and a stored result and trace round-tripped by the 15.23 build.

## 6. Readings

A seventh engine family, `claims` (`graded_by="judge"`), joins `ENGINE_FAMILIES` (`kernel/metrics.py:380`): the
recorded decision `metrics.py:2345` asks for (`tests/test_open_family.py:89`'s set grows by one). `claim_readings(tally)` computes the readings on read.

| Reading | Definition | Population | Direction |
|---|---|---|---|
| R1 `<c>_false_per_unit` (truth) | Asserted contradicted claims per unit, each restatement counted as heard | Cells with ≥ 1 unit, criterion not lost | Lower; ranked |
| **∆V2** R1b `<c>_false_unit_share` (truth) | Units holding ≥ 1 asserted contradicted claim, over units; `value_range` (0, 1), so it takes the mixture bet | Cells with ≥ 1 unit, not lost | Lower; ranked |
| R2 `<c>_grounded_share` (with `grounded_in`) | Of asserted specific claims, the share a `grounded_in` document supports | Cells with ≥ 1 asserted specific claim, documents declared, not lost | **Higher**; ranked |
| Diagnostics | `_false`, `_false_hedged`, `_false_attributed`, `_false_share` (on [0,1]), `_claims`, `_units`, `_hedged_share`, `_unverifiable`, `_true_ungrounded`, **`_web_settled_share`** (asserted claims the web decided), `_evidence_conflicts`, `_web_checked`, `_web_skipped`, `_lost` | Stated per reading | None |

**R2, precisely.** *Population:* asserted specific claims, since a specific detail (a name, number, date or rule) is
what research can source; general statements would reward padding and penalise framing no page states; hedged and
attributed claims are out, as in R1. *Grounded:* supported by a `grounded_in` document, one the candidate itself was
given. Case support does **not** count unless `grounded_in` names `case` (the summarizer, whose source it is shown);
otherwise an arm reciting the key from memory reads grounded and R2 collapses into truth. *Ungrounded:*
`true_ungrounded`, `unverifiable`, case-supported-only, and contradicted unless its own page supports it
(false-but-grounded is possible; R1 catches it). *No research* (DoW arm A, `documents=()`) reads **0**, not "no
reading": it asserted specifics with no source, which is what R2 ranks, and dropping it would hide the contrast the owner asked for. A cell with no asserted specific claim has no R2 value.

**The third reading, `_web_settled_share`, stays a diagnostic.** It moves with the grader's suspicion and the case's
coverage as much as with the arm, so as a merit it would rank the instrument. R2 already ranks the arm that sourced
its claims above one checked after the fact. Promoting it later is a descriptor change. **R1's test:** unbounded and
zero-inflated, R1 declares no `value_range`, takes no mixture bet, and is read on the comparison's t-path over per-case
means (`analysis/bundle/comparisons.py:532-543`), as its description says. **∆V2** R1b, ranked by owner ruling, is its bounded companion: a lie in a unitless segment
counts in R1's numerator but in no unit's share, so the two can disagree on a summary-line lie, and each says so. Up to
three ranked claim readings per criterion join the aspect's Holm family with its rubric dims, so each added reading widens
every corrected interval in that family a little; a host declaring several criteria should know it. **Built controls** (scripted clients), each moving only its own reading: a planted,
a hedged and an attributed lie; a bland, a repetitive set; an invented vivid detail; a canon contradiction with web
off; a house-rule truth with a contradicting `run` passage and web page (reads `supported`); a source-faithful summary
claim the web contradicts (`supported`); no research (R2 = 0); full research (R2 = 1); ∆V2 one lie in each of two units against two lies in one unit (R1 equal, R1b differs); **∆S** a shuffled set (back-announces out of order, every world claim right): R1 and R2 do not move, `_session_claims` counts them, and a session-order derived reading catches it.

## 7. Ranking: claim readings are judged readings

Claim readings go in the **`judged` namespace**, keyed by reading name. `ReadingKind` keeps its two values
(`kernel/campaign.py:89`). The writer's `ReadingRef`, `resolve_reading` (`analysis/references.py:97-99`, `:211-236`),
`report/aspect/common.py:226`, `report/aspect/guardrails.py:144` and `CellFacts` therefore need no change: a claim reading is a judged reading that is not a rubric dim.

| Where | Change |
|---|---|
| `analysis/reporting.py` `project_score_records` (`:1597`) | Emits one judged row per claim reading per result, `rubric_scale=None`, `instrument=<c>` |
| `describe_rubric_dim` callers (`analysis/bundle/comparisons.py:330-345`, cell reads) | Dispatch through `describe_judged_reading(name)`: a rubric dim, or `describe_claim_reading` |
| `analysis/agreement.py` `judge_key` (`:190`) | For a claim reading, the key comes from the tally: `dim` = the question it rests on (R1 `<c>.true`, R2 `<c>.grounded`), `scale` = `pass_fail`, `config` = `claims:<c>@<sha12>` |
| `analysis/bundle/cell_reads.py` `judged_measures` (`:151-226`) | Accepts a scale-less claim row |
| `analysis/report_input.py` `Judged` (`:590`) | `instrument: str \| None` (`exclude_if` None) |
| `analysis/caveats.py` `_checks` (`:858-885`) and implied judge raisers | De-duplicated by `(instrument, question, against)` |
| `analysis/first_screen.py` `RANKS` | Unchanged; a pinned test places a claims caveat at rank 5 or 6 |
| `kernel/declaration.py` `resolve_bar_name` (`:1425-1470`) | Step 2 (judged) resolves claim readings |
| `ops/regressions.py:20-25` | Fires only when the instrument is calibrated |
| Per-criterion losses | A reading whose arms lost cells at shares that differ beyond their interval is `withheld: trials_lost` (`schema/reading_states.py`); any loss raises a class-B caveat |

**The set-aside half of the rubric rule has no producer yet** (`schema/reading_states.py:54-57`; design-judges §5.5).
Until it exists, a claims reading ranks counted, with its class-C caveat. E1 does not wait for it.

## 8. Calibration: a claim read

**`ClaimRead`, its own durable type** (17.10), not a `BlindRead`, which is case- and pair-shaped
(`schema/blind_read.py:74`, `:113-121`). It reuses `Maker`, the sample rules and the reviewer guards (addendum §5). An
item is one claim's view (the claim, the excerpt with offsets, the segment, and the numbered evidence and windows the
instrument read), with no verdict, citation or arm. The questions, all pass/fail: `<c>.true` (pass = not false; the
rater may search), `<c>.grounded` (the documents shown support it), `<c>.is_claim` (extractor precision). The label key is the view's digest; `CalibrationRating.claim_ref` names the claim.

**The draw** is stratified by the instrument's verdict (truth: `contradicted`, `true_ungrounded`, `unverifiable`,
`supported`; groundedness: `grounded`, `ungrounded`), takes at most two claims per cell, counts cells toward the floor,
and has a seed and `planned_items` per stratum. **Reported per question:** the population 2×2, inverse-probability-
weighted (stratum share ÷ stratum draws), with a stratified bootstrap interval; the per-stratum agreement beside it;
weighted kappa. **Per arm:** recall (detection recall), precision (PPV) and rate bias (the instrument's false rate ÷
people's, on the same items). When per-arm recall or bias differs beyond its interval, the check is `not_met`, reason
`instrument_leans`. That is the owner's requirement: suspicion may differ by arm. A weighted 2×2 is new code (G): today's kappa is per rater and unweighted (`kernel/evidence_tiers.py:20-35`).

**Sizing the labelling.** From the review's seeded simulation: kappa's 2.5 to 97.5% spread at about 10% prevalence, against the 0.6 bar (`evidence_tiers.py:100-101`).

| Labels | Random: width at true κ 0.71 / 0.82 | P(`met`) at 0.71 / 0.82 | Stratified 50/50: width at 0.71 / 0.82 |
|---|---|---|---|
| 100 to 120 | 0.50 / 0.42 | 0.29 / 0.65 | 0.34 / 0.28 (120) |
| 200 | 0.34 / 0.28 | 0.39 / 0.85 | 0.27 / 0.22 |
| 400 | 0.24 / 0.19 | 0.58 / 0.98 | 0.19 / 0.16 |
| 800 | 0.16 / 0.14 | 0.81 / 1.00 | — |

At 8% prevalence the random widths are 0.56, 0.38, 0.27 and 0.19; at 12%, 0.45, 0.31, 0.22 and 0.15. Stratifying buys
the precision of 1.5 to 2× as many random labels and pins PPV tightly. People agree with each other at about κ 0.73 to
0.81 on `<c>.true` (the host's second label), so plan **≥ 200 stratified labels per question**, and that per arm for
the per-arm figures. **Judge repeats** (`run/judge_repeat.py`) re-verify the stored claims with the extraction fixed
and re-ask the web verdict over the stored windows without re-searching, so the A/A check and a second judge both test
the web step. **Left out** (CHANGELOG): an extraction recall check (misses are listed, not checked), judge campaigns for claims, a corrected contrast.

## 9. Re-judge, replay, derived readings, and the F0 fix

**Re-judge and replay.** `reproducible_judge_inputs` (`run/rejudge.py:103`) learns criteria: it loads `claim_evidence`, refuses when today's
criterion fingerprints otherwise than the tally recorded, and re-runs each criterion in `claim_failures` whole, web
included. A completed grade is never re-searched. Claims calls bind no `inputs`, so they are keyed by their prompt,
which carries the windows: a changed document is a different call (`testing/recorded_completions.py:16-31`).
`RecordedWebSearch` replays the port by (identity, query); a miss is a `ReplayMiss`; checks over replayed grades read `not available: replayed`.

**Derived host readings (`derived_from`).** A `MetricDescriptor` with `derived_from: <criterion>` is computed by a pure, synchronous `ClaimReader(ClaimGrade,
ClaimEvidence) -> Mapping[str, float | bool]` on the `HostProfile`, reading `Segment.tags` and (∆S) `ClaimEvidence.session` against the grade's session claims, after the grade lands and
on re-judge and re-measure. It reads in the judged namespace under the instrument's key (§7), qualified as R1 is; the
registry refuses `derived_from` in a code-graded family. Examples: discodon's R3, R4 and back-announce order correctness; DoW's `npc_contradictions` and game-state accuracy ("you rolled a 14" against the roll log).

**F0, the `failed_checks` / `graded_by` fix (independent).** `failed_checks` (`kernel/scoring.py:380-389`) and `is_check` (`kernel/metrics.py:598-601`) ignore `graded_by`, so a
judge-graded directional host check fails trials today. Fix: `MeasureRegistry` refuses one at registration, naming the
rule, and `failed_checks` skips anything `is_code_graded` rejects. For hosts: declare such a family `graded_by="code"` if code made the number, or drop the direction.

## 10. What chunk B (`evals/bp-claimx` `f350cd5f0`) must change

| Change | Detail |
|---|---|
| Move, hide | `kernel/claims/extraction.py`, and `kernel/verbatim.py` into `kernel/claims/`, both engine-internal; off `surface_tiers.json`, `api/surface.json`, `tests/enforcement/_public_api_consumers.txt`; vocabulary registers follow |
| Rename | `Utterance` → `Segment` (into `schema`); `stop_index` → `unit`; `_check_stops` → `_check_units`; "stops" → "units", "speaker" → "the text" in docstrings, prompts (`:89`, `:100-101`) and errors |
| Generic rules (`:64-84`) | Into discodon's `guidance`: "who made or took part… a first, a record", "introducing a work as someone's", "how something sounds", "talk about the programme itself", "echoes of the request", "a credit", the hedge phrases |
| Add | The `attributed` tag; **∆S** the `scope` tag (`world`/`session`) with its prompt rule (a claim about this output or session itself: what was just presented or played, what comes next, a roll, a hit-point count, "as noted above"), the rule that a mixed line splits into a session and a world claim, `restates` never crossing scopes, and tests for each, the back-announce split among them |
| Drop | The ellipsis refusal in `verbatim_span` itself (`kernel/verbatim.py:112`) and in the excerpt checks: a contract change, so the fold becomes `verbatim-fold-v2` |
| Fold identity | `VERBATIM_FOLD` + sha256 of the tables + `unicodedata.unidata_version`, pinned by a test that changes a table |
| Error | `ClaimExtractionFailed(ApparatusError)` → `ClaimGradingFailed(Exception)` |
| Keep | The repair loop, offsets, `judge` ledger and digest; rewrite the CHANGELOG entry |

## 11. Consumers, designed in

| | DoW (a) `rules.claims` | DoW (b) `scout.claims` | RAG answer bot | Summarizer |
|---|---|---|---|---|
| Output, segments | `Answerer.answer` (`dow:src/dow/gm/answering.py:83-109`); one unit per answer | `Scout._draft` (`dow:src/dow/gm/scouting.py:193-195`) on `scout_context` (`:126`); `ScoutDraft` (`:59`) fields: summary `unit=None`, place, areas and NPCs as units ("entry"); `tags` = entity id | The answer; one unit, or per paragraph | Per paragraph |
| Arms | A: as today. B: `SrdIndex.search(q, limit=3)` (`dow:src/dow/lookup/index.py:238`) passages added first | Scout prompt or model | Retrievers, prompts | Models, prompts |
| `case` | Named SRD sections, the asker's view, the house rules in force (`render_house_rules`) | The module (`dow:content/modules/silver-hollow.yaml`) and the scout's context | Key facts or gold passages, if any | The source |
| `run` (via `Answer.documents`) | B's passages; `()` for A | — | The retrieved passages | — |
| `grounded_in` / web | `{run}` / on (the SRD is public; a recorded `WebSearch`) | `None` / off (fiction has no web) | `{run}` / on, or off for a private corpus | `{case}` / off |
| Ranked | R1 truth; R2 (arm A reads 0) | R1 (canon contradictions per entry); derived `npc_contradictions` | R2, faithfulness to retrieval; R1 against case and web. With web off and no case evidence, R1 detects nearly nothing, **by ruling** (run pages never nominate), so R2 is its reading | R1 (contradicts the source); R2 (supported by the source) |

House rules as case evidence make arm B's correct house-rule answer read `supported` (blocker 1). No column needs a
discodon concept; what DoW exposed is fixed in the engine: `attributed`, web off, structured segments, no key,
`Answer.documents`, `grounded_in`, and (∆S) session-scoped claims graded against the session record.

## 12. Decision register

★ = hard to reverse, for the owner. Reversibility: S stored shape, N public name, I measure identity, D doctrine. ∆ = changed since v3.

| # | Decision | Long-term effect | Rev. | A non-discodon consumer | Rejected |
|---|---|---|---|---|---|
| 1★ | The engine runs claims in the judge phase | Ranking, re-judge, repeats and checks come built in | D,S | Declares; writes no grader | A helper in `invoke` |
| 2★ | `ClaimCriterion` as its own type | A clean fingerprint and budgets | S,N | One more grader | Fields on `RubricDim` |
| 3★∆ | Claims evidence beside `JudgeEvidence`, on the trace, outside its fingerprint and projection; 1 MiB cell cap | No rubric label key moves; bundles stay light; a second evidence record to maintain. **The hardest storage call** | S | Two render callables plus `Answer.documents` | Fields on `JudgeEvidence` (v3) |
| 4★ | Four verdicts plus `graded_against` and `nominated_by` | One vocabulary | S,N | Same words in every regime | Host verdicts |
| 5★ | Provenance `case`, `run`, `web` | Arm-dependent evidence is visible | S | Retrieval = `run`; source = `case` | One bag |
| 6★∆ | Truth sees only case evidence; run pages never nominate; the web never overrules case | One detector for every arm; RAG with web off and no case gets little R1 | I,D | Truth means the same everywhere | Run nomination (v3) |
| 7★∆ | Units, unitless segments counted outside the denominator | Per-unit rates for any cut | N,S | "answer", "entry" | "stop" |
| 8★ | Per-cell rates averaged over cells | Wide intervals at few cases | I | Same | Pooled claims |
| 9★ | Readings derived on read from a tally | A diagnostic is free | S,I | — | Stored readings |
| 10★ | The `claims` engine family | One rule for judged numbers | D | Ranks like its dims | A host family naming its judge |
| 11★ | The rubric rule; counted with a caveat until set-aside exists | No second rule | D | Same | Rank only once a check is met |
| 12★∆ | `ClaimRead`, stratified by verdict, ≤ 2 per cell, IPW 2×2, per-arm recall, PPV and bias | Calibration that sizes and detects leans; a weighted 2×2 to maintain | S | Labels claims on its own sheet | Claim items on `BlindRead`; random draws |
| 13 | `derived_from` plus a pure `ClaimReader` | Domain readings inherit the checks | N | DoW's NPC reading | Counts in `invoke` |
| 14★∆ | A priced `WebSearch` port, recorded with its windows, never re-searched | Stationary and checkable | N,S | A port, or web off | Re-searching; unpriced ports |
| 15★∆ | Failures per criterion; never `judge_error` | The pass rate and rubric readings never move with claims | D,S | Same | A cell-level judge error (v3) |
| 16 | Separate gate holds per criterion | A tight cap drops claims, not rubric dims | — | Same | One whole-phase hold |
| 17 | The fold identity includes its tables and the Unicode version | An honest split on a Python upgrade | I | Invisible | Name only |
| 18 | No key required | Serves cases with no answer set | — | Summarizer, RAG | `entity_set` required |
| 19★∆ | The `attributed` tag | Changes R1's numerator | I,S | Fiction and RAG need it | Folded into `hedged` |
| 20 | A judge-graded host check refused at registration | Closes a ranking hole | D | A startup error | A silent filter |
| 26★ ∆V1 | A cell with `claim_evidence` is claim-graded whatever its declaration | Claims-only evals work; one more judging entry to maintain | D,S | DoW and RAG evals with no rubric | Forcing a judged declaration |
| 27★ ∆V2 | Two ranked truth readings (R1, R1b) plus R2 | A bounded reading takes the mixture bet; wider Holm families | I | Same | R1 alone |
| 28 ∆V5/V6 | `CLAIM_TRACE_BYTES` with `trace_lost`; unit nouns in `claim_unit_names` | No reading without a stored grade; nouns are presentation | S,N | Same | Logging a lost trace; a nested `unit` |
| 21★ new | Claim readings in the `judged` namespace; `ReadingKind` unchanged | Writer, resolver and charts untouched; the namespace now holds non-rubric readings | N,S,D | Same | A third `ReadingKind` |
| 22★ new | Groundedness ranked, higher is better, over asserted specific claims supported by `grounded_in` documents; no research reads 0 | Rewards sourcing; arm-dependent by design | I,D | RAG: faithfulness to retrieval. Summarizer: faithfulness to its source (`{case}`), or not declared | Unranked diagnostic; case support counting |
| 23 new | `_web_settled_share` as a diagnostic, not ranked | Avoids ranking the grader's suspicion | I | Visible, unranked | A third ranked reading |
| 25★ ∆S | A `scope` tag (`world`/`session`) on every claim; session claims out of R1, R2 and world diagnostics, never web-checked, graded only by derived readings against an optional opaque `ClaimEvidence.session` | Continuity defects stop reading as false statements; the extractor carries one more judgment that calibration's `<c>.is_claim` question should also check (scope right?) | I,S | DoW game-state claims, RAG self-reference, discodon back-announces | A host-side filter (cannot split a sentence); grading session claims as world claims |
| 24★ new | `grounded_in` declared per criterion | One field serves research, RAG and summarizers | S,N | Declares its sources | `run`-only groundedness |

## 13. Chunks (one risk surface each)

| Chunk | Risk surface | After |
|---|---|---|
| F0 | The `failed_checks` / `graded_by` fix | — |
| B′ | Chunk B rework (§10) | — |
| C1 | Stored shapes only (durable 17.9, core 15.24), the launch refusal, the pins; ∆V1 the claims-only declaration rule; ∆V2 `units_with_false`; ∆V5 the trace cap and `trace_lost`; ∆V6 `claim_unit_names`; ∆S `scope`, `session`, `session_claims` | A1 |
| C2 | The family, readings, `describe_claim_reading`, the built controls | C1 |
| C3 | Verifier library: truth and ground calls, numbering, pages, citations, repair, the verdict table | B′, C1 |
| C4 | Judge phase: the criterion fingerprint, separate holds (rubric first, then each criterion, each released before the next), pricing, per-criterion failures, spend, re-judge, replay, a test that rubric-bound `inputs` never reach a claims call. **∆V3:** a deadline inside a criterion is that criterion's loss (`deadline`, a `_DEADLINE_CHARGE` arm, `run/runner.py:487-493`, `:4312-4330`), never `judge_error`; settled rubric dims stand; criteria run inside the cell timeout, and "For hosts" tells hosts to raise `cell_timeout`. **∆V8:** `judge_witnessed_output` (`run/runner.py:2124-2125`) carries `claim_evidence` | C2, C3 |
| Q | The quick-path `Claims` door and `Answer.documents` | C4 |
| R0 | **DoW spike:** both rungs on scripted clients, no spend, to catch shape defects early | Q |
| D | Web: port, budgets, windows, `skipped_cap` caveat with a per-arm flag, `RecordedWebSearch` | C4 |
| E1 | The judged-namespace ranking path (§7), loss withholding. **∆V4** also: `judged_values` (`analysis/bundle/cell_reads.py:119-132`) yields claim rows; the `EvalRun.rubric_scales` readers (`analysis/bundle/trials.py:311`, `analysis/bundle/telemetry.py:217`, `analysis/analyst/facts.py:473`) and the ~15 `rubric_scale` readers (`analysis/bundle/bars.py:270`, `analysis/lenses/pivot.py:753` among them) treat a scale-less claim row as a rate; `resolve_bar_name` step 2 (`kernel/declaration.py:1463-1470`) stops keying judged names to a `RubricScale` alone; pins that a claim row never enters a trial's pass or a threshold share, and that writer prose calls a per-unit rate a rate, never a score | C2, C4 |
| E2 | `derived_from`, `ClaimReader` | E1 |
| G | `ClaimRead` (17.10), stratified IPW calibration, per-arm figures, judge repeats over windows. **∆V7:** allocate draws by expected misses, oversampling `unverifiable` (and `supported`), not 50/50; the lean is one homogeneity test across arms, not pairwise intervals | C4, E1 |
| R | **DoW rungs on `evals-ladder`**: live recordings once, under one owner-approved spend ceiling for every R call; DoW pins the integration ref by git rev (`dow:pyproject.toml:35`); the DoW branch is merged by the owner, not by this program | R0, D, E2, G |
| H | Discodon adoption | R |

C1 through G sit unreleased on one integration branch (`evals/next`) until R is green: both DoW rungs replay offline,
rank their arms and raise their caveats, importing no discodon name. Each chunk: one builder, scripted clients, a Critic.

## 14. Review dispositions (`design-claims-v3-review.md`)

| # | Finding | Disposition |
|---|---|---|
| 1 | Case support could be overruled | Fixed: case support always reads `supported`; the web never overrules case evidence (§4); house-rule and summary controls (§6) |
| 2 | Documents on `JudgeEvidence` move label keys and ship pages | Fixed: the fingerprint claim was verified at `schema/models.py:1021`; `ClaimEvidence` sits beside, outside the projection; 1 MiB cap; ★3 re-rated (§3) |
| 3 | Run nomination is arm-dependent | Fixed by owner ruling: run pages never nominate, and truth never sees them; per-arm recall and a lean check (§8) |
| 4 | Third `ReadingKind` | Fixed: the judged namespace, register row 21, sites listed (§7) |
| 5 | A failure excludes the whole cell | Fixed: per-criterion failures and separate holds (§4); row 15 ★ |
| 6 | Blind read does not fit | Fixed: `ClaimRead` durable type; row 12 ★ |
| 7 | Web bounds, pricing, spend row, checkability, `skipped_cap` | Fixed: `WebBudget` bounds; `price_ceiling()` per query; unpriced port refused under a cap; `RoleUsage` row; windows stored and re-asked; a `skipped_cap` caveat with a per-arm flag (§3, §4, §8, D) |
| 8 | Summarization and RAG served badly | Fixed: `grounded_in`; the consumer table (§11); the A1 context-only refusal is stated |
| 9 | No quick-path channel for run documents | Fixed: `Answer.documents` (§2, chunk Q) |
| 10 | Calibration statistics | Fixed: stratified, ≤ 2 per cell, IPW 2×2, per-arm recall, PPV and bias, the n-versus-CI table (§8) |
| 11 | R1's range, test and set-aside | Fixed: t-path stated, a bounded companion shown, counted-with-caveat until set-aside exists, de-dup key named (§6, §7). The companion is now ranked by owner ruling (V2) |
| 12 | Chunk dependencies and the R gate | Fixed: fingerprint into C4; chunk Q split out; R0 spike; one spend approval for R; R owner named (§13) |
| 13 | Reach-back gaps | Fixed: launch refused on kept newer fields; `unit` unmeasured; pins named (§2, §5) |
| 14 | A fallback model makes the key `None` | Fixed: per-verdict `decided_by`, tally `by_judge` (§5) |
| 15 | Line references | Fixed (§11 DoW refs) |
| 16 | The ellipsis lives in `verbatim_span` | Fixed: dropped there, `verbatim-fold-v2`, `verbatim.py` hidden too (§10) |
| 17 | Recordings keyed by inputs | Fixed: claims calls bind no inputs (§9) |
| 18 | The whole-phase gate | Fixed: separate holds (§4); row 16 |
| 19 | Register re-ratings | Applied: 3, 7, 15, 19 ★; rows 21 and 24 added; 16 and 17 unchanged |
| 20 | What holds up | Kept |
| V1 | Claims-only cell never judged | Fixed (§4 entry rule; C1; row 26) |
| V2 | Both truth readings ranked | Fixed (R1b, `units_with_false`, a control; row 27) |
| V3 | Deadline during claims | Assigned to C4 |
| V4 | More judged-namespace readers | Assigned to E1, sites listed |
| V5 | Cap omits windows; trace loss logged | Fixed (§3 cap; `trace_lost`) |
| V6 | Nested `unit` | Fixed (`claim_unit_names`) |
| V7 | Allocation, lean test | Assigned to G; cost is owner question 2 |
| V8 | Witnessed path | Assigned to C4 |

## 15. For the owner, and not verified

1. **Groundedness's population and zero rule** (row 22). Recommended: asserted specific claims; no research reads 0;
   case support counts only where `grounded_in` names it.
2. ∆V7 **Calibration cost.** At ≥ 200 labels per question per arm, a three-arm campaign needs about 1,800 labels for the
   three questions. Recommended: full stratified labels for `<c>.true`; `<c>.grounded` and `<c>.is_claim` on a smaller unstratified sample (about 100 per arm).
3. The web audit of case contradictions: deferred to #1297 (owner). Ranking R1b: ruled (both truth readings ranked).

Not verified: the 1 MiB cap against discodon's pages (C1 measures it), and the stratified bootstrap's coverage at two claims per cell (G simulates it).

## ∆C1: corrections from building C1 (2026-11-21)
- An older build (17.8) launches a template carrying claim criteria and grades no claims. It doesn't refuse it; refusing unknown fields is new in C1. §5's reach-back is corrected to that.
- `schema.Segment` is B′'s, and C1 adopted it. **A unit may span several segments** (decided by the coordinator for structured outputs, e.g. a DoW scout entry): units are 0..n-1, each in at least one segment; an excerpt lies within one segment; per-unit counts count distinct units.
- `ClaimTally.units`, `units_with_false` and `session_claims` are per cell, not per judge key, so two keys never double-count a unit.
- `grounded_in` is stored as a sorted tuple. `judge_witnessed_output` belongs to C4 (§13 wins over §4).
- Carried to C2/E1: refusing criterion names that collide with host measures. Carried to C4: **V5's behaviour** (C1 built only the `trace_lost` value). A failed trace write must lose the criterion, not just be logged, but `kernel/storage.py` writes the result BEFORE the trace (`:2782-2797`), the opposite of what V5 assumes, so C4 must settle the order or record the loss after the fact. Also carried to C4: `CandidateOutput.claim_evidence`, the judge client required at launch, and the tally derived from a grade. Carried to C2–D: the callers ledger entries for the new public names, removed as callers land.

## ∆C3: from building the verifier (2026-11-21)
- **Separate page budgets for case evidence and run documents.** With one shared `PageBudget.total_chars`, an arm's fetched pages would change which case passages are read, and the truth reading must use the same detector for every arm. A test holds this. C1's `PageBudget` description must say it (a fix in the C3/C1 integration).
- `reference_facts(case)` was not built. The verifier reads `key.fact_entities` (A1). A key-type-neutral accessor waits for a second key type.
- `ClaimGradingFailed` carries a `loss` (`cut_short` / `unreadable` / `repair_budget_spent`), which C4 records.
- Carried to D: a restatement's web outcome. `ClaimGrade` requires a web record for every claim marked web-checked, and `graded_claims` doesn't yet take web outcomes.
- Carried to C4: the extractor's two repair rounds counting against `RepairBudget.max_items_per_cell`.
- "One sentence per quote" is enforced by the prompt only. Code enforces the 300-character cap and that the spans sit inside the quote.
- **∆C3 Critic (BLOCKED, fixing):**
  - **Truth and groundedness are fully independent:** separate repair budgets, and a failure on the arm's pages loses only groundedness, never truth. Otherwise arm-dependence comes back through repair.
  - **Subject and value spans:** they must sit in one sentence or table row, with no sentence break between them.
  - **Shown passages:** a quote must come from a passage the call was shown.
  - **Identity:** the fault templates enter it.
  - **Docstrings:** they must say code narrows the neighbouring-value error but cannot prove binding, and that calibration measures what remains.
- ∆C3 fixed at 7b3400b33. Carried to C4: a per-reading loss slot, since `claim_failures` holds one loss per criterion and `Verification.ground_loss` must be recorded so that a lost groundedness never reads as lost truth. Fix at integration or in C4: C1's `RepairBudget` description must say "per step (truth and ground separately)", and `PageBudget` must say the case and run budgets are separate.
- ∆C3 verify passed at 7b3400b33. **Carried to C4 (arm fairness):**
  - A provider error or `CompletionProtocolDefect` in a GROUND call still raises out of `verify` and loses the truth findings. C4 must catch it at the ground step and record it as a ground loss, so truth stands. After that, only the cost cap stays arm-tied.
  - Store `ground_loss` per reading.
  - Price the repair budget per step, which doubles the worst case.

## ∆C4: from building the judge-phase integration (2026-11-21)
- C4 was split, as the builder proposed and the coordinator accepted. C4 (bp-claimphase 5351b246a) covers the flow, failures, per-criterion per-step pricing, V1, V3, V5, V8, `CandidateOutput.claim_evidence`, ground-call provider errors becoming a ground loss, and extractor repairs counting against the budget. **C4b:** §9 re-judge and replay; witnessed claims-only cells of unjudged kinds; pricing each criterion per arm before launch; the account-refusal stop for claims calls.
- Corrections:
  - **Repair pricing:** the budget counts items across rounds, so the bound is `max_items` per step × 3 steps, not `max_items × rounds`.
  - **Write order:** a run cell writes its trace second. V5 is handled by writing the row as lost first and restoring its tallies once the trace lands.
  - **Losses:** they include `provider_error`.
  - **V1:** it needed launch to allow, and require, a judge for an unjudged kind with claim criteria.
  - **Protocol defects:** one in an extraction or truth call fails the run, as it does for a rubric dimension. That's a programming defect, not an arm outcome.
  - **Web:** a criterion with a web budget is refused until D lands.
- Open: the verifier's echo allowance (3× text + 2000 chars), for the C4 Critic to weigh.
- C4 Critic (PASS WITH WARNINGS). All four warnings are being fixed before merge, including **a deadline in the ground step keeping completed truth** (arm fairness). Its nine observations are added to **C4b's scope**: they include a guard refusal mislabelled `cost_cap`, and the "No judge produced output" disclosure on claims-only cells, which goes to E1. The full list is in `bp-claimphase/.prawduct/critic-findings.json`.
- **C4 PASS at e59eb4ebb** (bp-claimphase). `trace_pending` is a non-loss state, and a result becomes `trace_lost` only on a reported trace-write failure. **Carried to E1:** exclude `trace_pending` from every loss count and disclose pending cells. **Top of C4b:** recovering a pending row by recomputing its tally from the stored grade (the grade must first store its ground-step loss).

## ∆D: from building the web check (2026-11-22)
- Built on bp-claimweb f273fc73d. The port is `WebSearch`/`WebPage`, provisional, on `RunnerOptions.web_search` and `KindWiring.web_search`. The web step has its own repair budget, and restatements share their original's web record.
- Builder decisions sent to the Critic: a port error is `port_error`, not `provider_error`. An unpriced port is billed at its own `price_ceiling`. A cap or deadline inside the web step leaves the rest `skipped_cap` and truth stands. **Open:** an unquoted web verdict after repairs loses the whole criterion, truth included (arm-fairness question).
- **Carried to E1:** the per-arm `skipped_cap` caveat. **Carried, open:** web criteria are refused on the witnessed path. **Port gap:** it can state only a per-query ceiling, not the actual cost; an optional cost field would be an additive change later.
- **C4b built** at bp-claimrejudge 7735eb0cc. It added the grade's `ground_loss`, `grade_tally`, `claim_fingerprints`, `ClaimRescore`, and `hold_exceeded`, which replaces the mislabelled `cost_cap`. The claims re-judge check is a sibling, `reproducible_claim_inputs`. The core shape pins are re-pinned under the unshipped 15.24. **Open, to the Critic:** whether take-up should recover pending rows automatically (today only a re-judge does); a ground-call protocol defect stays a ground loss, per ∆C3.
- **D Critic BLOCKED** at f273fc73d. Coordinator rulings for the fix:
  - Every web-step failure is a web-only loss (`web_loss`, mirroring `ground_loss`): truth stands, and the undecided claims read `unverifiable`. Only the cap and the deadline give `skipped_cap`. This follows from the D7 and arm-fairness rulings, so no new owner ruling is needed.
  - The port returns `WebSearchResult(pages, usd: float | None, replayed: bool)`: the actual cost when known, and replay declared, not probed. It's decided now because widening a bare tuple later would be a breaking change.
  - The "not available: replayed" readers are carried to G.
  - The web budget limits get tests.
- **New chunk Dw** (scheduled 2026-11-22, at discodon's request): web on the witnessed path (`judge_witnessed_output` takes a `WebSearch` port, with the same refusals, spend row and replay). It starts after D and C4b land. **Why:** discodon's first claims work is witnessed: #3526 grades its 24 story-set fixtures and the aired transcripts (owner D5 ordering), and owner D2 grades live sessions later. Until then discodon runs web-off, labelled as such. Claims the case contradicts still read `contradicted`.
- **C4b Critic: PASS WITH WARNINGS** at 7735eb0cc, with arm fairness confirmed. One fix round before merge:
  - the program budget reads `claim_rescores`;
  - the trace-conflict message is made true, plus an etag guard if it is small;
  - **pending rows are recovered on their own at run settle and at take-up**, never refused for template or setting drift;
  - ground-step protocol defects get their own loss name;
  - the CHANGELOG reach-back is corrected (no 15.23 build was ever released).
- **D fixed** at 6c22d84d8: `web_loss` (web state `lost`; truth stands); `WebSearchResult(pages, usd, replayed)`, billed reported → declared rate → ceiling, with a replay free. Verify running. **Carried to Dw:** re-judge redoes a web step whose grade carries `web_loss`, since it needs both D and C4b. Also a test for a search refused at admission. **To E1:** the caveat names `web_loss` and `web_skipped` separately.
- **D verify PASS WITH WARNINGS** at 6c22d84d8. The four observations go into **Dw's scope** rather than an issue:
  - a recording run that repeats a query marks its own grades replayed;
  - a reported cost can overshoot the hold by at most one search;
  - a grade over the trace cap because of web windows loses the whole criterion (it should be a web loss);
  - no test covers a cut-short web reply.
- D integrates with C4b, as one batch, once C4b's fix round passes.
- **E1 built** at bp-claimrank a0c44cc0d:
  - judged rows with `ScoreRecord.instrument`, no scale, and one dispatch;
  - a lost criterion withholds all of a cell's readings, and a lost ground step only R2;
  - a reading is withheld on differential loss (interval on per-case loss shares excludes zero; 42/1000 at equal 10%, 180/200 at 0 vs 30%);
  - class-B caveats per arm for lost, pending, web-skipped and web-lost, plus the `web_loss` seam;
  - a new stored `ClaimTally.grounded`, amending 15.24.

  The Critic is running, weighing multiplicity across more than two arms and whether `grounded` is needed. Open: the aspect page heads claim rows with the raw name.
- **E1 Critic BLOCKED** at a0c44cc0d, and one fix round was sent:
  - claim readings are kept out of floors and bars (R1b was read as an unflipped pass rate);
  - differential-loss withholding becomes one homogeneity test across all arms, since the pairwise tests reached 12.6–25% false withholds at 3–4 arms; seeded 2/3/4-arm simulations are pinned;
  - an arm that loses every cell counts as lost in R2;
  - the CHANGELOG states the `grounded` reach-back;
  - the aspect page uses reader names, prints the rate, marks withheld readings, and gains a claims-page presentation test.
  
  **Carried to Dw:** the `n_web_lost` producer, and re-grading an `evidence_over_cap` loss, which pays to fail again.
- **Batch 28 stopped** on two seams between C4b and D (nothing pushed; local evals/next 7f00660fb holds C4b only). Coordinator rulings:
  1. The priced-hold guard is `hold_exceeded` everywhere, and D's `_admit_amount` raises it for calls and searches alike. In the web step, `hold_exceeded` and `cost_cap` are both budget stops: `skipped_cap`, truth stands, counted in `web_skipped`.
  2. A re-judge or take-up of a lost web criterion searches live through the port (§9: re-run whole, web included). The re-judge or take-up refuses by name, up front, when the port is missing. `refuse_unbuilt_claim_steps` is deleted.
  
  The D builder merges C4b into bp-claimweb with these rulings, then integration takes bp-claimweb alone.
- **E1 verify PASS** at f6050da69:
  - claim readings are refused from bars (`ranks_only`);
  - withholding uses one CMH test of arm × loss, stratified by case: 4.8, 3.2 and 2.2% at 2, 3 and 4 arms, and 195/200 detection;
  - `ClaimTally.grounded` stays required, so a3035f640 claim stores are unreadable after E1 (pre-release, no back-compat). Discodon has been told they are throwaway.
  
  Before integration, seeded rows go in at 3% loss, one repeat, and case-clustered loss, with an exact test at small totals if any row exceeds 5%.
- **E1 final PASS** at 56b07f51a, with no new findings:
  - below 40 lost cells, the CMH p comes from a conditional permutation (999 within-case draws, fixed seed);
  - 13 `claim-loss.*` certified claims: equal-loss rows at most 0.05, detection at least 0.75;
  - `n_web_lost` is wired to `web_loss`.
  
  **Carried to E2:** a certified row above 40 losses, since a reading near p 0.05 can flip as a campaign crosses the switch. Integrating as batch 29.

## ∆E2: derived readings (2026-11-22)
- Built at bp-claimderive a2e9e4778. It adds `MetricDescriptor.derived_from`, `HostProfile.claim_readers`, `ClaimReader`, and a stored `ClaimTally.derived`. The readings are ranked and withheld as R1 is. Deviation (accepted): the registry is shared by all templates, so a reading must be named `<criterion>_…` and resolves as `not_carried` elsewhere.
- **Critic BLOCKED.** Coordinator rulings:
  1. Withholding always uses the conditional permutation, with no 40-loss switch, unless it costs more than about 1 s per reading at 10k cells. Every `claim-loss.*` row is re-certified, plus a stress row above 40 losses.
  2. **Host contract (decision register):** a reader receives a frozen `ClaimView` (per claim: text, excerpt, scope, verdict, tags, unit, segment; the segments; a deep-frozen `session`), never `ClaimGrade` or `ClaimEvidence`. The grade's internals stay free to change. This is the reversible choice, and the owner is told.
  3. A reader is checked at launch on a synthetic view. At run time a raising reader leaves the grade stored and its derived readings unread, with a caveat, and never fails the run (the good-intent principle). Re-measure recovers.
- E2 fixes at 505956a0c:
  - the permutation runs always (1.09 s per reading at 4 arms × 10k cells and 10% lost);
  - a frozen `ClaimView` (`ViewedClaim`/`ViewedSegment`);
  - readers are checked at launch, and a raising reader leaves `derived_unread` with a `claims_unread` caveat.

  **Certificate ruling:** an exact test's size is α by construction, so a 95% upper bound below α is uncertifiable. Every claim-loss null row is restated as at_most α + 0.01 (upper bound ≤ 0.06 on about 4,000 draws), certifying that it is not anti-conservative. Withholding stays at p ≤ 0.05. The permutation cost at the largest campaigns goes to an issue draft.

## ∆Q: the quick-path door (2026-11-22)
- Built at bp-claimquick f01546f15: `Claims`, `Document` and `Answer.documents` (all provisional), the `segments=` and `case_documents=` callables, and `search=`. Both DoW rungs can be expressed on the quick path; derived readings and `session` need `as_host`.
- **Critic BLOCKED.** Rulings:
  - Refuse a `Claims` with no truth source and no `grounded_in`, and refuse `grounded_in="case"` with no case documents.
  - With `grounded_in="run"` and no truth source, R1 carries a "not checked" caveat and never reads as a clean 0.
  - `_fitted` must terminate.
  - `unit` is dropped from the case-set id (∆V6).
  - **The aspect page prints every claims class-B caveat**, built here because it must land before R.
  - `Answer.documents` is stated provisional.
- Filed: claim readings as `measure_order` leads.

## ∆G: calibration (2026-11-22)
- Built at bp-claimcal a8cdbdc27: `ClaimRead` (17.10), the stratified draw (floor, then spread by expected misses), IPW figures, `repeat_grade` over stored windows, "not available: replayed". Departures accepted: a smoothed stratified bootstrap; the lean test on recall and false fails; no `claim_ref` (the claim read's record traces every label).
- **Critic BLOCKED:**
  - the unweighted arm permutation reads a fair grader as leaning when the arms' verdict mixes differ under the stratified draw. Fix: permute within strata, and add null rows with differing mixes;
  - claim-read rows not certified on the pool;
  - **no host door.** Split into **G2** (draw, sheet, answer, host-readable figures, host-run repeats), before H.
- Corpus paused: "Jules" in two claim excerpts, possibly a discodon persona; asked.

## ∆R0: the DoW spike (2026-11-22)
- DoW branch `evals-claims-r0` b0ce60506, pinned to aed87e72c. Both rungs run offline (scripted judge, replayed web) and every planted claim reads as intended. Twelve shape defects are listed in the R0 report.
- **Owner ruling, case precedence:** explicit overrides. `EvidenceDocument.overrides: tuple[str, ...]` (case documents only). Support quoted from an overridden document, where an overriding document contradicts the claim, reads `contradicted`, not supported with a conflict. Generic cases: house rules, errata, policy updates. Additive and optional.
- **Owner ruling, R spend:** a $3 ceiling for every R call.
- Fix chunks:
  - **S1** (engine and page): #2 the invented model call, #6 the uncalibrated disclosure, #7 the meaningless pass rate, #8 the derived-reading output, #9 the web-failure and replay disclosure, #12 runner bugs blamed on the model;
  - **S2** (quick-path doors): #3 derived readings, #4 `session=`, #5 a public read of claim results;
  - **S3** (verification): the overrides ruling, plus #10, a semicolon breaking a citation;
  - #11 (small-case guidance) is filed as an issue.
- **S3 built** at bp-claimoverride 4291b5dc5. `EvidenceDocument.overrides` is optional and case-only, with refusals for run documents, self-overrides, unknown ids and cycles.
  - Per citation, support from a document is overruled when a document that overrides it takes the opposite stance. Precedence follows the chain, and key facts sit outside it.
  - The call is shown one line per override pair, and code applies the precedence.
  - A semicolon no longer breaks a sentence for the span rule.
  - The base build refuses traces that carry overrides.
  - **Follow-up for S2:** `case_documents` mappings carry `id` and `overrides`, and `Answer.documents` refuses `overrides`.
- **S2 built** at bp-claimdoors 12b155c5f:
  - `Claims(readers={name: (read, direction[, bounds])})`;
  - `Claims(session=field or (case, answer) -> mapping)`;
  - `Result.claim_cells()` returning `ClaimCell`.

  All provisional, in `threetears.evals.quick`.

## ∆S4: host-path questions from discodon (2026-11-22)
- **Silence:** `ClaimEvidence.segments=()` means the candidate said nothing. There is no judge call, and zero claims and units. The cell is left out of every per-unit and share reading, is not a loss, is counted per arm, and gets a class-B caveat. `claim_evidence=None` with output stays refused, since it can't be told from a kind that forgot.
- **Undeclared documents:** `documents=None` under `grounded_in` run is a per-cell ground loss, `documents_undeclared`. Truth stands and differential-loss withholding applies. The quick path keeps its run-level refusal.
- **Export:** `CLAIM_DOCUMENT_BYTES` is exported, provisional. Every claims name stays provisional until R is green; promotion is the decision after that.

## ∆G2: the claim-read host door (2026-11-22)
- Built at bp-claimreaddoor 05f613afd: seven actions on the blind-read pattern (create, sheet, answer, close, repeat, get, list). There is no caller seed, answers require an attested caller, and repeats are priced at the worst case under the repair budget, never search, and are stored on the read.
- **Critic BLOCKED.** Rulings:
  - the grader's web windows are dropped from the sheet, since they mark which claims the grader doubted, and the rater searches;
  - **figures, the Check and exclusions are frozen at close** and stored on the read;
  - the repeat guards get tests.
- **Merge condition:** the report page reads a closed claim read's Check, so the S1 "uncalibrated" line resolves. Whichever of S1 and G2 lands second wires it.
- **S4 Critic BLOCKED** at 5c6b6b023 (untested `result_regrade` refusal). **Coordinator ruling, amending silence:** silent cells (and, for derived readings, unread cells) enter E1's differential test as lost-or-silent against read cells, so a silent share that differs by arm withholds the reading. Silence stays out of the loss counts and every mean. This closes #1301 item 2. `trace_pending` is never due a paid re-grade.
