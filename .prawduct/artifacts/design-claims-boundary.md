# Claim grading: where the boundary goes (architect's read, 2026-11-21)

Read against `evals/main` at `0bce2461a`, chunk A1 at `origin/evals/bp-ctxkey` (`1afd937d4`) and chunk B at
`origin/evals/bp-claimx` (`f350cd5f0`). Paths are under `packages/evals/src/threetears/evals/` unless noted. Nothing
was run. This note answers one question: which side of the engine/host seam claim grading lives on, given that a
second consumer must be able to use it and that stored shapes are hard to take back.

## 0. The answer

**Make claim grading an engine judged instrument, run in the judge phase beside rubric dimensions.** Today it is a
helper a kind calls inside `invoke`. The engine owns the claim, its verdict, the evidence citations and the per-unit
readings. The host owns three things: how its output is cut into units, its domain guidance text, and the readings
only it can name (track relevance, persona). Those host readings are computed in code from the stored claim grade
and declared as derived from the engine instrument, so they rank under the same rule.

The deciding fact is that the helper path cannot meet its own success criterion. It lands numbers in a judge-graded
host family. Nothing ranks those (§1.6). A host-named judge would not fix that either, because the engine cannot see
the verdicts it would need to check.

## 1. How judging works today (facts)

1. **The criterion.** `RubricDim` (`schema/models.py:925`) has a name, description, scale (`ordinal` 1-5 or
   `pass_fail`), scoring guide and axis (`capability`/`boundary`), and declares its judge (`builtin`/`configured`,
   `:956`). Its *criterion fingerprint* hashes name, description, scale and guide (`fingerprint_criterion`, `:1025`).
2. **Who runs it: the engine, after the cell.** The kind only renders `JudgeEvidence` (`subject`, `case_material`,
   `artifact`, optional `inputs`; `schema/models.py:5797`), declared by `JudgedArtifact` (`:5777`). The runner's judge
   phase (`run/runner.py:1947-2040`) builds the context (`build_judge_context`, `:3402`), prices it against the cost
   cap through a gate, and asks each dim (`judge_dims`, `:3466`) through `JudgeService` (`run/judge_service.py:625`)
   → `run_judge_llm` (`run/judge.py:388`). The kind never calls the judge.
3. **Stored per result.** Each `RubricScore` (`schema/models.py:1143`) holds the score, reasoning, `served_model`,
   `judge_temperature` and `judge_sent_settings` as the call reported them, and the label key: an
   `output_fingerprint` over the evidence (`fingerprint_judged_output`, `:1000`) plus the `criterion_fingerprint`.
   The evidence is kept on the trace, so a re-judge re-sends exactly it. Also stored: `judge_config_ids`, errors,
   cannot-tell and cut-short per dim (`run/runner.py:628-645`), and `judge_rescores` (`:6565`).
4. **Failure.** A judge error on a dim excludes the cell as `judge_error`, and a re-judge recovers it without
   re-buying the candidate (`run/runner.py:482-486`; `run/rejudge.py`, `run/lifecycle.py:270-320`). An
   `ApparatusError` *in the judge phase* fails the run (`run/runner.py:2031`). One raised out of `invoke` excludes
   the cell as `apparatus_failed`, and the cell is never judged (`run/runner.py:38-45`).
5. **Re-score and checks.** The engine re-asks the same stored evidence and criterion in four ways: judge repeats for
   self-agreement (`run/judge_repeat.py`), a second judge (`run/judge_second.py`), temperature sweeps
   (`run/judge_temperature.py`) and judge campaigns over frozen cases (`kernel/judge_cases.py:1-30`,
   `run/judge_kind.py:1-35`). A `JudgeCase` is one (output, criterion, labels). People's and models' ratings pair
   with a score by `LabelKey` (`schema/models.py:986`). Agreement is per-rater kappa (`kernel/evidence_tiers.py:1-60`).
   A judge is keyed by dimension, scale, served model and config digest (`evidence_tiers.py:64-70`).
6. **How a judged reading becomes rankable.**
   - Rubric dims rank: per-case means enter the comparison family as `("judged", dim)`
     (`analysis/bundle/comparisons.py:195-199`), and capability dims under the threshold fail trials, which moves the
     campaign's pass rate (`kernel/scoring.py:410-440`, `analysis/campaign_readings.py`).
   - Checks qualify a judged reading but do not gate it. An unchecked judge raises a class-C caveat
     (`analysis/caveats.py:170`, `:858-885`). A check that is `not_met` sets the reading aside:
     `instrument_unchecked`, `instrument_below_bar`, `instrument_leans`, `people_disagree`
     (`schema/reading_states.py:54-57`, ranked at `analysis/first_screen.py:103-107`). I found the vocabulary and the
     ranks but no producer of those set-aside states yet (design-judges §5.5 is not fully built).
   - The owner removed the evidence tier from every decision (design-judges §9.4). Regressions alone still require
     `calibrated` (`ops/regressions.py:20-25`).
   - Host measures rank only when code-graded (`kernel/metrics.py:2361-2381`; `analysis/bundle/measures.py:182`;
     bar names at `kernel/declaration.py:1463`; pinned by `tests/test_open_family.py:70`). `regressions.py:24`
     states the reason in one line: *"A judge-graded host measure names no rubric dimension the engine can look a
     judge up by, so it never fires."*
   - **A gap to fix either way.** `failed_checks` (`kernel/scoring.py:380-389`) and `is_check` (`kernel/metrics.py:598`)
     never consult `graded_by`. A judge-graded host boolean or pass/fail measure with a direction would therefore
     fail trials and move the ranked pass rate, with no judge identity and no check behind it.
7. **Spend.** The roles are `candidate|judge|simulator|inner_agent|external` (`schema/models.py:5303`). The judge
   phase's calls are `judge` rows on the cell, admitted by the cost-cap gate. The extractors called in `invoke` bill
   the cell's `judge` ledger too (`kernel/entity_extraction.py:40`; chunk B `claim_extraction.py:59`), but no gate
   prices them up front. Repeats and second judges are out-of-run spend.
8. **Identity of host-graded numbers.** A host measure's descriptor `formula` carries its identity (`set_grade.py`
   formula). The `measure_rereader` is synchronous (`kernel/host/measures.py:69`), so a re-measure can re-derive
   numbers from stored data but cannot re-ask a model.

## 2. The three placements

| | (a) Kernel helper in `invoke` (current) | (b) Engine judged instrument (recommended) | (c) Helper plus a "host family names its judge" link |
|---|---|---|---|
| Who runs it | The kind, inside `invoke` | The runner's judge phase, after the cell | The kind |
| What a second consumer writes | A kind that calls the extractor and grader, builds `Utterance`s, holds a `ReferenceKey` with at least one core entity (`schema/reference.py:74`; R20) even with no answer set, declares the family and descriptors, and stores the grade on `kind_payload` | A criterion on its template (`ClaimCriterion`: name, unit noun, guidance, evidence regime, budgets), plus segments and documents on the `JudgeEvidence` it already renders. Nothing else | Everything in (a), plus a judge declaration the engine cannot verify |
| What is stored | Numbers on `host_measures`, and the grade on `kind_payload`, which the engine never reads (`kernel/candidate_kind.py:386-404`) | Typed claims and verdicts with citations on the result or trace, the evidence on the trace, and per-call `served_model`/temperature as on `RubricScore` | As (a) |
| Changing it later | A host-private shape: every host forks it, and the engine cannot migrate or re-read it | An engine durable shape (schema minor, MIGRATIONS). Costly once, then one migration path for every host | As (a), plus a doctrine exception to unwind |
| Calibration | Host-side only (#3526). The engine sees no item, label or check | The engine's check record: per-claim items keyed like `LabelKey`, blind reads, agreement, A/A, the "corrected" contrast | The host reports a tier the engine cannot reproduce |
| Ranking | Never: the family is judge-graded (§1.6) | Under the rubric rule: counted, qualified and set aside by the instrument's checks (§4) | Ranks on a host assertion: a second rule for one kind of number |
| Evidence path | The host passes the key, sources and web port into its own call | Key facts read by the engine from `EvalTestCase.reference`; run documents rendered by the kind onto the evidence (so a re-judge is reproducible); web search through a host port on the run options, every query recorded | As (a) |
| A grader failure | `ApparatusError` out of `invoke` throws away the candidate's paid output for every measure, rubric dims included, and cannot be recovered | A `judge_error` on that criterion alone, recoverable by re-judge | As (a) |
| Re-grade with a new judge | Re-run the candidate (the rereader is synchronous) | Re-judge from the stored evidence, as rubric dims are | As (a) |

The second consumers, concretely:

- **A RAG answer bot.** One unit per answer; documents are the retrieved passages (provenance `run`); no key; web
  check off.
- **A summarizer.** One unit per summary, or per paragraph; the document is the source (provenance `case`); no key.

Under (b) each one declares a criterion and renders evidence. Under (a) each writes the discodon integration again,
with `stop_index=0` and an invented answer set. And still gets no ranking.

What (b) costs. The judge phase gains a second instrument type. `JudgeEvidence` gains optional `segments` and
`documents`, declared with `exclude_if` so existing fingerprints do not move, the same mechanism A1 used. The check
record and the blind read must also accept an item that is a claim, not a (result, dim) pair. That last part is the
real build cost. It is also exactly what calibration needs, wherever the grader sits.

## 3. General versus discodon-specific

**General (engine):**

- **Units and per-unit rates.** A *unit* is a host-cut segment that counts in the denominator. A segment with no
  unit still has its claims counted. The host names the unit noun ("stop", "answer", "section") for reader text.
- **Atomic claims** with a verbatim excerpt and offsets, using the verbatim fold (`kernel/verbatim.py`).
- **Tags:** `specific`, `hedged`, `restates`.
- **The verdict taxonomy:** supported, contradicted, unverifiable, and true-but-ungrounded after a web check, with
  `graded_against ∈ {key, document, web}` and `nominated_by`.
- **Evidence sources with provenance.** `case` covers key facts in any tier and case documents. `run` covers
  documents the arm itself fetched. `web` covers what the port returned.
- **The web check:** a host port, budgets, recorded queries and deciding quotes, never re-searched on re-score.
- **The readings** "false per unit" and "unsupported share of specific claims", plus the diagnostics.

**Discodon's (host):**

- segues as stops, and messages and voice lines as unitless segments
- track relevance (R3) and the worst-repeated fact (R4)
- persona, the radio-programme guidance, music-credit examples
- the labelled set and its agreement run (private data)

**Where chunk B and the design bake in a discodon shape:**

| Where | Discodon shape | General form |
|---|---|---|
| B `claim_extraction.py:130-142`, `:151` (`Utterance.stop_index`, `ExtractedClaim.stop_index`) | stop | `Segment(text, unit: int \| None)`; `ExtractedClaim.unit` |
| B `:131` docstring "a stop's introduction"; module doc `:3-5` "stops (the numbered items it presents in turn)" | introduces a stop | "a segment that is one unit of the output" |
| B `:204-215` `_check_stops` | stops 0..n-1 | units 0..n-1 (keep the rule, rename it) |
| B `:65-67` rules: "who made or took part in something, what something belongs to... a first, a record" | music trivia | "a name, date, place, quantity, attribution or relation stated as fact" |
| B `:69` "introducing a work as someone's asserts that it is theirs" | track introduction | move to discodon `guidance`; keep "a framing's implied claim counts" |
| B `:71-73` "how something sounds"; "talk about the programme itself (what was just presented, what comes next)"; "echoes of the request" | radio DJ | "descriptions and opinions are not claims; text about the output itself is not a claim" (move the rest to guidance) |
| B `:79` "a credit"; `:80-81` hedge phrases ("if memory serves") | music, spoken | keep hedging generic; put the examples in guidance |
| B `:89` "what a speaker said" (prompts `:100-101`) | spoken output | "the text" / "the author" (a summary is not spoken) |
| B `:121-126` `ClaimExtractionFailed(ApparatusError)` | runs inside `invoke` | under (b) a judge-error type: an `ApparatusError` in the judge phase fails the whole run (`runner.py:2031`) |
| Design §6 names `_false_per_stop`, `_stops`; "per segue stop" | stop | `<c>_false_per_unit`, `<c>_units`; the host's `reader_name` says "per stop" |
| Design §1 / R20: a key needs one core or acceptable entity | an answer set always exists | claim grading needs no key; R20 stays for set keys only |
| Design §4: evidence `K` read from `ReferenceKey` | entity_set key | a key-type-neutral `reference_facts(case)`; `entity_set` yields `fact_entities` (A1) |
| Design §4 `Literal["key", "key+sources"]` | discodon's two regimes | a set of provenances, `{case, run, web}` |
| Design §6 one "judge-graded claim family" of host measures | host family | an engine family, `claims`, judge-graded, whose judge identity the engine records |
| Design §7 "judge sampling parameters are the host client's to keep fixed" | host-run | recorded per call, as on `RubricScore` |

**One general distinction the design does not draw.** Evidence of provenance `run` depends on the arm. An arm that
fetches more pages gets more claims supported. Against `run` documents, "unsupported share" therefore measures how
grounded the reply is in the arm's own research, not whether it is true. A false rate against `case`+`web` evidence
does measure truth. For discodon this is the research arms' main confound, so the grade should record provenance on
every citation, and each reading should declare which provenances it counts.

## 4. When a judge-derived count may rank

The rule already in the code, written generally: **a number a judge stands behind may rank only when the engine
itself recorded which judge instrument produced every judgment the number counts, keyed so that the instrument's
checks attach, and only when the engine can compute those checks from items it stored.** Such a number then ranks as
a rubric dim does. It is counted, it is qualified when unchecked, and it is set aside when a check is `not_met` or
the instrument leans (§1.6). A count derived from per-claim verdicts meets this rule under (b). The checks then
belong to the claim instrument: verdict agreement per claim, extraction recall and precision, and the A/A check.
Host readings over the same grade declare `derived_from=<criterion>` and inherit those checks.

The proposal "a judge-graded host family names its judge and ranks only at a calibration tier" is not coherent with
this rule, for four reasons:

1. **The tier is out of decisions.** The owner removed it (design-judges §9.4). Gating on it brings back the thing
   §9.4 deleted.
2. **Two rules for one kind of number.** Rubric readings rank unchecked, with a caveat. This would gate ranking
   instead.
3. **A name is not an identity.** A host-written name is not the per-call `served_model`/temperature/config identity
   the engine keys a judge by (`evidence_tiers.py:64-70`).
4. **The engine cannot check it.** The verdicts would sit in `kind_payload`, unread, so the "tier" would be host
   self-report.

The cleaner existing path is the rubric one, generalised from a "dim" to an "instrument". Close the
`failed_checks`/`graded_by` gap at the same time.

## 5. Recommendation

**Boundary.** Build `kernel/claims/` as an engine instrument. The criterion declared on the template carries: name,
unit noun, the host's guidance (inside the criterion fingerprint), provenances read, page caps, and web budgets.

- **Kind:** renders `JudgeEvidence.segments` (text, unit, opaque host tags such as a track id) and
  `JudgeEvidence.documents` (id, url, title, text, provenance).
- **Run:** the host supplies a `WebSearch` port on the run options.
- **Judge phase:** extract, then verify, then web-check. The result is stored as a typed `ClaimGrade` on the result
  or trace, with per-call identity.
- **Readings:** engine measures per criterion, judge-graded and keyed to the instrument.
- **Host readings** (R3, R4): code over the stored grade, `derived_from` the criterion.
- **The library stays callable on its own.** The quick path and tests can use it outside a run, but no stored path
  bypasses the phase.

**Chunks.**

- **A1 (bp-ctxkey): keep as built.** It is general for `entity_set` keys and additive (`exclude_if`, durable 17.8).
  `fact_entities` is the accessor the grader's case facts should read, behind a key-type-neutral function. Do not
  extend R20 to claim grading.
- **A2 (#1293, #1292):** unaffected. It is set grading.
- **B (bp-claimx): keep the mechanics and rename the surface before anything pins it.**
  - Keep: `verbatim.py` unchanged, the repair loop, the offsets, the identity digest, the `judge` ledger.
  - Rename: `Utterance`→`Segment`, `stop_index`→`unit`, "stops"→"units", "speaker"→"text".
  - Rewrite: the rules at `:64-84`, generically; the domain examples move to discodon's `guidance`.
  - Change: `ClaimExtractionFailed` becomes a judge-error type that the phase records against the criterion.
  - Leave it `provisional`. Its CHANGELOG entry describes stops and utterances, so rewrite it.
- **C1, C2, D:** redesign as engine judge-phase chunks, then a calibration chunk (claim items in the check record and
  blind read). Discodon's agreement run stays host-side on private data, but it writes engine checks.
- **Fix now, independent of claims:** `failed_checks`/`is_check` must ignore, or refuse at registration, a
  directional check measure in a judge-graded host family.

**Owner decisions, each hard to reverse:**

1. *Who runs claim grading, the engine's judge phase or the host's kind?* **Engine.** Only then can it rank, be
   re-judged and be checked.
2. *Where do per-claim verdicts live: an engine-typed record (durable schema) or host `kind_payload`?* **Engine-typed**,
   on the trace, with counts on the result.
3. *What is the denominator noun in shared names?* **"unit"**, with each host's reader name on top ("per stop" for
   discodon).
4. *Which closed verdict set ships, and what is the owner's "ungrounded, not untrue" called?* **`supported`,
   `contradicted`, `unverifiable`, `true_ungrounded`, plus `graded_against` and `nominated_by`.** Renaming a stored
   verdict later is a migration.
5. *Does a ranked reading mix arm-fetched evidence with case and web evidence?* **No.** The false rate counts
   case+web contradictions (run documents may nominate). A separate groundedness share is declared against run
   documents. Discodon chooses which of the two it ranks.
6. *Does a judge-derived count rank by the rubric rule (counted, caveated when unchecked, set aside on a failed
   check) or only after a check is met?* **The rubric rule.** One rule for all judged numbers. Discodon may still
   decide only on checked readings.
7. *Does claim grading require a reference key?* **No.** Key facts are one optional evidence source, and summarizer
   and RAG cases have none.
8. *How are web searches spent and recorded?* **They are billed and capped per cell inside the judge phase's
   cost-cap gate, with every query and deciding quote stored and replayed, never re-searched.**
