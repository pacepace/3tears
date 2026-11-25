# Review: design-claims-v3.md (claim grading as an engine judged instrument, #1287)

Independent design review, 2026-11-21. Read: the v3 design, rev 2 (`design-claims.md`), the rev-2 review, the boundary
note, the #1287 thread, the code at `evals/main` `0bce2461a`, `origin/evals/bp-ctxkey` (`1afd937d4`, `392a3e3c1`),
`origin/evals/bp-claimx` (`f350cd5f0`), and the DoW clone on `evals-ladder` (`ffd982f`). Nothing was run except a
local seeded simulation of kappa's sampling spread (finding 10). Paths are under `packages/evals/src/threetears/evals/`
unless they start with `tests/` or `dow:`. No DoW content is copied here, only paths and lines.

Severity: **blocking** (resolve before the chunk it names is built), **should-fix** (resolve in the design or its
chunk), **note**.

---

## Blocking

### 1. Case support can be overruled by the web, and a case-supported claim can read `true_ungrounded` (blocking, C3)

The step-1 rules (§3) send "case support and contradiction" to nomination with `evidence_conflict`. The contradiction
may come from a `run` document, and then `nominated_by=run`. The verdict table then gives:

- web `false`: `contradicted`, against `web`. A web page has overruled a case document that **supports** the claim.
- web `true`: `true_ungrounded` + `evidence_conflict`. The claim is grounded, in the case.

That breaks D7 / boundary 5 ("case evidence is authoritative"). It also breaks the three regimes where the case is the
truth by definition:

- **DoW rules, arm B.** A house rule in force (a `case` document, §12) supports a claim. The SRD passage arm B fetched
  (`run`) states the public rule. The web (public SRD) agrees with the passage. The answer that correctly applies the
  table's house rule is graded **false**. §12 says the house rules are case documents "so a house-rule truth is
  case-supported before any web check". The table defeats exactly that.
- **Summarization against a source.** The source is `case`. Any outside page that disagrees with the source flips a
  faithful summary sentence to false.
- **Fiction** (DoW scout) is safe only because web is off there.

Recommendation: add the missing row. A claim that case evidence supports reads `supported`, whatever `run` documents or
the web say. A contradiction elsewhere sets `evidence_conflict` as a diagnostic and never changes the verdict. Web
checks it only to record the conflict, and only under a declared budget. Then two-case-documents-disagree is the
remaining conflict, and it needs its own ruling (today it reads nominated by `case`, web decides). Add built controls:
a house-rule truth with a contradicting `run` passage and a contradicting web page, and a source-faithful summary claim
that the web contradicts.

### 2. Documents on `JudgeEvidence`: rubric label keys move, builds disagree on fingerprints, and every batch evidence reader ships page text (blocking, C1; re-rate ★3)

Evidence:

- `fingerprint_judged_output` hashes the **whole** evidence except `inputs`
  (`schema/models.py:1004-1021`: `evidence.model_dump(mode="json", exclude={"inputs"})`). The rubric judge reads only
  `subject`, `case_material` and `artifact` (`run/judge_service.py:1036-1039`). So once a kind renders `segments` and
  `documents`, every rubric dim's `LabelKey.output_fingerprint` for that kind moves. It moves again whenever a fetched
  page's text differs, though the rubric judge never read the page. A label then fails to carry to a byte-identical
  rubric question, for example from a judge case frozen before adoption. That contradicts the fingerprint's own
  doctrine ("the digest of the evidence a judge reads"). §2's "unchanged for every existing evidence" is true only of
  evidence that never adopts claims.
- **Mixed builds disagree.** A 15.23 build reads a 15.24 trace's new fields as kept-opaque fields, and those "appear in
  no `model_dump` … so no digest … can reach them" (`schema/base.py:354-356`). The two builds therefore stamp
  **different** rubric label keys on the same stored evidence during any rollback or rolling-deploy window, for example
  on a re-judge by the older build.
- **Weight on the light path.** `load_judge_evidence` projects a trace down to `judge_evidence` precisely because it is
  light (`kernel/storage.py:137-150`, `:2864-2865`; addressing pinned at `schema/versioning.py:652-654`). Its readers
  are the bundle's trials, up to `MAX_TRIALS = 500` (`analysis/bundle/trials.py:333`, `analysis/bundle/caps.py:37`),
  the writer pairwise (`ops/report_writer/writer_pairwise.py:555`) and blind reads (`ops/blind_reads.py:680`). With
  pages on the evidence, each of these ships every page of every trial. An illustration, not a measurement: 15 pages
  of 30k characters is about 450 kB of evidence per research cell, against the ~120 kB whole-result figure in
  `EvalTrace`'s docstring (`schema/models.py:5876-5880`). That is about 225 MB per bundle assembly at 500 trials.
  `BlindOutput.evidence` is a required whole `JudgeEvidence` (`schema/blind_read.py:113`), so a blind read stores the
  pages durably once per drawn item.
- §2's table puts `segmentation` "inside the fingerprint". Which fingerprint? The rubric output fingerprint (above), the
  claim view digest (§7), or the criterion's is never said.

Recommendation, before C1 pins anything:

- Exclude `segments`, `segmentation` and `documents` from `fingerprint_judged_output`, as `inputs` is. Then no rubric
  label key moves under adoption or across builds. Claims carry their own label key: §7's view digest, which already
  includes the evidence the verifier read.
- Store documents **beside** the evidence, not in it: an `EvalTrace.judge_documents` field outside
  `_JUDGE_EVIDENCE_FIELDS`, or its own `eval_trace_documents` doc type. Keep a digest of them on the grade. The claims
  phase, the re-judge and the claim read load them by id. Nothing else does.
- State a per-cell stored cap in bytes now (§15 leaves it open) and what the kind does past it.
- Re-rate ★3. Where the page text lives is the hardest-to-reverse storage choice in the design.

### 3. `run` nomination makes the truth reading's sensitivity depend on the arm, in the direction that penalizes fetching (blocking, E1/G; needs an owner line)

A false claim that contradicts the arm's own fetched page is always nominated and then web-checked. The same false
claim from an arm that fetched nothing is nominated only if the judge happens to flag it `suspect`. So R1's
**detection rate differs by arm**: arms that research have more of their own errors found. Decision 6 says "arms cannot
buy truth by fetching more". That is true, but the converse is the live risk: an arm *pays* for fetching in R1.

This is DoW's exact product question: arm A (no SRD) against arm B (SRD passages), §12. It is also discodon's main
contrast (research arms). The calibration plan (§7) draws claims at random over all arms and reports one agreement. A
pooled kappa can be fine while recall differs by arm, and that difference is what a contrast reads.

Recommendation, at least the first two:

- (a) Say it in decision 6 and in R1's description.
- (b) G reports the instrument's recall on human-false claims **per arm**, or per `run_documents_declared` regime, and
  E1 raises `instrument_leans` (already in `SetAsideReason`, `schema/reading_states.py:54-57`) when the per-arm
  recalls differ beyond their interval.
- (c) Optionally, make nomination arm-symmetric: grade each cell against the case's documents plus **every** arm's
  `run` documents for that case, as a post-run re-judge. That is costly and couples cells, so put it to the owner rather
  than build it now.

### 4. A third `ReadingKind` is a hard-to-reverse decision missing from the register, and E1 lists too few sites (blocking, E1)

`ReadingKind` is "measure | judged" (`kernel/campaign.py:89`). The two-namespace assumption runs deeper than §6's
table:

- `ReadingRef.reading` is model-authored ("which of a cell's two namespaces", `analysis/references.py:97-99`), so the
  writer's reply schema and its recordings move.
- `resolve_reading` raises `ValueError` on any third value (`analysis/references.py:211-236`).
- `report/aspect/common.py:226` maps an unknown kind to `None`.
- `report/aspect/guardrails.py:144` infers judged-or-measure from the name.
- The `CellFacts` measures/judged namespaces and `ResolvedReading.judged_tier` assume two kinds.
- A stored analysis whose findings name `claims` is refused by an older build.

The alternative is cheaper and as honest: claim readings are judged readings. Put them in the `judged` namespace, keyed
by reading name, with the instrument's `JudgeKey` and tier. That way the writer, the resolver, the tier code and the
charts need no change. Recommendation: choose one explicitly as a ★ (N,S,D) register row. If `claims` stays a third
kind, list every site above in E1.

### 5. A claims failure excludes the whole cell, so claim grading moves the ranked pass rate's population (blocking, C4; re-rate decision 15)

`ClaimGradingFailed` becomes a judge error, and a judge error on any dim excludes the cell as `judge_error`
(`run/runner.py:482-486`; boundary §1.4). The trial then leaves the pass rate and every rubric reading until a re-judge
runs. Under D1 claim readings never enter a trial's pass. Yet a claims failure removes the trial, and it does so
differentially: claim-rich, page-heavy cells hit the repair budget and the output cap more often (rev-2 review 6).
Decision 15 says "rubric readings wait for the re-judge" but does not say the pass rate waits too.

Recommendation: a claims failure is recorded **per criterion**. It excludes the cell from that criterion's readings only.
It never sets the cell's `judge_error` exclusion, and the bundle counts it as the criterion's own loss, with a
differential-loss flag per arm. If the engine cannot express a per-criterion exclusion, that is the C4 build item, and
decision 15 becomes ★ D.

---

## Should-fix

### 6. The blind read does not fit a claim item; the register calls it reuse (should-fix, G; add a ★ S row)

`BlindRead` is case- and pair-shaped. `BlindDraw.population` counts "cases every chosen arm gave a judged, usable
answer on" (`schema/blind_read.py:74`). A `BlindItem` is one case with one or two slotted outputs for a preference
(`:116-121`). `BlindOutput.evidence` is required and whole (`:113`). A claim item is one claim of one arm's output,
drawn from a claim population, with no pair and no preference. Hanging `claim: ClaimItemView | None` on `BlindOutput`
(§7) leaves `population`, `population_digest`, slots, the preference and `evidence` meaning something else per item.
It also stores every page once per drawn claim (finding 2).

Recommendation: a claim read with its own draw unit. That is either a `ClaimRead` durable type or a `unit: case|claim`
discriminator that `BlindDraw` validates. Its item carries only the view (claim, excerpt, segment, the numbered
evidence the verifier read). Add it to §13 as ★ S: this shape is what every later claim label keys on.

### 7. The web step: spend is bounded per cell, but three parts of it are not, and its checks cannot test it (should-fix, D)

- **Verdict-call input is unbounded.** `PageBudget` governs documents. Nothing caps results per query or the characters
  of returned pages fed to the web-verdict call. The ceiling (§3) prices "verdict call" with no input bound. Add
  `results_per_query` and `page_chars` to `WebBudget`, and select windows by the same code rule as `pages`.
- **`price_ceiling(query)` cannot be asked before the query exists**, and the gate prices up front. It must be
  `price_ceiling()` per query.
- **An unpriced port serializes the run.** An unpriced phase "holds the run's judging alone, one call at a time"
  (`run/runner.py:2063-2087`, `:2002-2005`). Web search providers mostly report no price (`RoleUsage` docstring,
  `schema/models.py:5366-5376`). A claims criterion on such a port would then judge every cell of the run serially.
  Refuse an unpriced port at launch where a cap is enforced, or require a declared rate.
- **The spend row (§15 open question).** `CallUsage` cannot carry it, but `RoleUsage` already can: `provider`,
  `provider_unit`, `provider_units` and `call_count` with `None` tokens. Write role `judge` (program spend) with
  provider = the port's identity and unit `query`. Never use `external`, which counts as production-replicating
  (`schema/models.py:5345-5348`).
- **Reproducibility is right, checkability is not.** Never re-searching keeps grades stationary, and
  `RecordedWebSearch` keyed by (identity, query) replays deterministically, because the query comes from a replayed
  judge call. But `WebCheckRecord` keeps no returned text. Judge repeats and second judges therefore *copy* web
  verdicts (§7, §8), so neither A/A nor a second judge ever tests the step that decides every `run`- and
  `judge`-nominated R1 numerator. Store the bounded windows the verdict call read (or the snippets) with the record.
  Then a repeat re-asks the verdict over the recorded text without re-searching.
- **`skipped_cap` biases R1 low where it matters most.** Past the cap, `run`- and `judge`-nominated claims read
  `unverifiable`. R11 rightly refused an apparatus raise, but the reading is then a lower bound for false-heavy cells.
  Raise a caveat when any cell skipped, and flag it when the skip rate differs by arm.

### 8. Generality: summarization and RAG are still served badly (should-fix, C1/C2)

- **Groundedness counts only `run` documents** (R2, step 1 "`grounded` = a `run` document supports it"). A summarizer's
  source is a `case` document (§13 row 5 says so). Its core question is "is each sentence supported by the source"
  (faithfulness), and R2 cannot ask it. Labelling the source `run` instead makes it nominate-only for truth, so a
  sentence the source contradicts reads `unverifiable` with web off.
- **RAG over a private corpus, web off.** Retrieved passages are `run`, so they never decide. R1 can only come from key
  facts and case documents. A false claim the retrieval contradicts reads `unverifiable`, and R1 reads near zero for
  every arm. The design should say this is by ruling and give such a host its reading.

Recommendation: the criterion declares `grounded_in: frozenset[Provenance]` (default `{run}`), so a summarizer declares
`{case}`. State in §12's style a one-paragraph third-consumer walk-through, summarizer and RAG, naming its readings.
Also note that A1 still refuses a context-only key (`bp-ctxkey` `schema/reference.py` `_one_entity_per_name`), so a
fact-only case must use case documents. That is fine, but say it.

### 9. Quick-path `run` documents have no channel but the answer value (should-fix, C4; an N decision)

The quick path's `documents(case, answer)` sees only the runner's return. `Answer.value` is "what is graded, and what
the run stores as the cell's output" (`quick/answer.py:24-30`). For DoW arm B to declare `run` passages, its runner
must put them in the value. They then reach every rubric judge and grader as part of the output. Add a side channel
(`Answer.documents`, or the runner returning a typed `ClaimEvidence`) and name it in §1. It is DoW's only path to
`run` documents.

### 10. Calibration statistics: random draws at 8-12% prevalence are weak at realistic counts; stratify, weight, and report the rate's bias per arm (should-fix, G; answers §15)

Simulation: seeded, 20,000 draws per point. Judge sensitivity 0.75 and specificity 0.97 at prevalence 0.10 gives true
kappa 0.71 and a judge false rate of 0.102. That sits close to the owner's 8-12% and the host's human pair. The
interval is the 2.5-97.5% sampling spread of kappa:

| Labels | Random draws, ~positives | Kappa spread (true 0.71) | Width | P(`met` at 0.6) | Same at true 0.82 (Se .85, Sp .98) |
|---|---|---|---|---|---|
| 100 | ~10 | 0.42-0.92 | 0.50 | ~0.29 | width 0.42, P ~0.65 |
| 200 | ~20 | 0.52-0.86 | 0.34 | ~0.39 | width 0.28, P ~0.85 |
| 400 | ~40 | 0.58-0.82 | 0.24 | ~0.58 | width 0.19, P ~0.98 |
| 800 | ~80 | 0.63-0.79 | 0.16 | ~0.81 | width 0.14, P ~1.00 |

At 8% prevalence the widths are 0.56, 0.38, 0.27 and 0.19. At 12% they are 0.45, 0.31, 0.22 and 0.15. "P(met)" uses a
large-sample SE with the one-sided 95% lower bound against `CALIBRATION_MIN_AGREEMENT = 0.6`
(`kernel/evidence_tiers.py:100-101`, `:407-433`). The engine's score interval will differ somewhat, so read these as
orders of magnitude.

**Stratified by the instrument's verdict** (equal draws from instrument-contradicted and instrument-not, then
reweighted to the population's 2×2 by the stratum shares, which are known exactly for every graded claim):

| Labels | Width (true 0.71) | Width (true 0.82) |
|---|---|---|
| 120 (60+60) | 0.34 | 0.28 |
| 200 (100+100) | 0.27 | 0.22 |
| 400 (200+200) | 0.19 | 0.16 |

So stratification buys about the precision of 1.5-2× as many random labels. More importantly, it pins the
instrument's **precision** (PPV on contradicted) tightly. With weighting it still recovers recall.

Answer and recommendations:

- Stratify by the **instrument's** verdict (known for every claim), never by a human label. Report population agreement
  as the inverse-probability-weighted 2×2 (weights = stratum share ÷ stratum draws), with a stratified-bootstrap or
  weighted score interval. Show the per-stratum agreement beside it, as the host already does (supported 60/60,
  contradicted 60/74, unverifiable 21/46). Today's kappa is per rater and unweighted (`kernel/evidence_tiers.py:20-35`),
  so weighted draws need a weighted 2×2. That is a G build item, not "unchanged".
- Kappa is the wrong headline for a ranked rate. Report alongside it the instrument's false rate against the people's on
  the same items (bias, as a ratio), recall and PPV, **per arm** (finding 3). Two offsetting errors (recall 0.75,
  false-positive rate 0.03) give an unbiased overall rate with kappa 0.71, but they do not stay offset across arms that
  differ in prevalence.
- **Clustering.** The floor counts distinct "results" (`evidence_tiers.py:36-39`). If each claim is a result, 400
  claims from 15 cells are not 400 draws. Draw at most two claims per cell, or count cells toward the floor.
- **The human ceiling.** From the host's stratified second label, people agree with each other on the binary
  `<c>.true` at roughly kappa 0.73-0.81. That range depends on where the 25 moved unverifiable labels went, and it is
  lower if the supported stratum hides disagreement. A 0.6 bar is attainable. A judge cannot be expected to clear it at
  fewer than ~400 random or ~200 stratified labels.
- Move "stratified draws" out of §7's "left out" list. At this prevalence random draws are the expensive choice.

### 11. Ranking R1 through the comparison machinery: no range, zero-inflated, and "set aside" has no producer (should-fix, E1)

- A per-unit rate has no upper bound, so `describe_claim_reading` can declare no `value_range`. A reading with no range
  never gets the mixture bet. The quick path's comparisons use that bet, and DoW is the quick path
  (`analysis/bundle/comparisons.py:532-543`, `:407-410`). It falls to the t-test, and "with no range a gap with no
  spread is not separated". Per-case R1 values at 8-12% false and few claims per unit are mostly zero. State the
  test R1 is read by. Consider ranking a bounded companion (`_false_share` on [0,1], or "units with a false claim",
  a share) alongside R1, or instead of it.
- The set-aside half of "the rubric rule" has no producer yet (`schema/reading_states.py:54-57`; design-judges §5.5
  unbuilt). Until it exists, a claims reading ranks counted with a class-C caveat only. Say so in §6, and say that E1
  does not wait for that producer.
- `ReportInput.judged` is keyed by dimension (`analysis/caveats.py:858-862`). "Raised once per instrument" needs a
  de-duplication key. Name it.

### 12. Chunking: dependencies and the R gate (should-fix)

- **C1 needs B′.** `fingerprint_claim_criterion` hashes the fold identity and the prompts (§1). Both are B′'s. So
  either C1 follows B′, or the criterion fingerprint lands in C3/C4 and C1 pins only the stored shapes.
- **R needs spend, not only for the web.** DoW's rungs were recorded once live (`dow:docs/evals-ladder.md:39`). R's
  extract, verify and repair calls for both rungs need a live recording too. §12 asks owner approval only for the web
  recording. Ask once, for all of R's recordings, with a ceiling.
- **R needs a pushed 3tears ref.** DoW pins `3tears-evals` by git rev (`dow:pyproject.toml:35`), not by a released
  exact version, so R needs an integration ref carrying D, E2 and G. "Nothing is done until R is green" then means C1-G
  sit unmerged on one integration branch, which fits "one evals PR" and keeps the three schema steps unreleased until R
  proves them. That is the right property. But DoW is a separate private repo, and the design names no one who merges
  R there. Name the R owner, and say that R's DoW branch is not merged by this program.
- **Shift a DoW spike left.** A scripted-client DoW rung after C4, before D/E/G, would catch shape defects (findings 1,
  8, 9) while they are still cheap. As written, the first consumer contact is the last chunk.
- One risk surface each: yes, except C4, which holds pricing, errors, spend rows, re-judge, replay **and** the quick
  path door. Split the quick-path `Claims` door, with finding 9's channel, into its own chunk.

### 13. Reach-back gaps in §4 (should-fix, C1)

- A 17.8 build reading a 17.9 template keeps `claim_criteria` opaquely. If it **launches** a run from that template,
  the run grades no claims and says nothing. Its `measured_template_digest` (by exclusion over a dump,
  `run/definition_digests.py:87-103`) also leaves the kept field out, so its digest differs from the newer build's.
  State the outcome. Better, refuse a launch from a template carrying kept newer-minor fields.
- `unit` is "presentation, outside the fingerprint" (§1), but it sits inside `measured_template_digest` unless it is
  added to `_UNMEASURED_TEMPLATE_FIELDS`. Say which.
- Name the fingerprint pins C1 captures before the fields land: `fingerprint_judged_output` of an evidence without them
  (and, per finding 2, of one with them), `measured_template_digest` of a template without criteria, and a stored
  result and trace round-tripped by the 15.23 build.

### 14. `ClaimTally`'s judge key goes `None` on any fallback model (should-fix, C2/C4)

"`None` when calls disagree, matching nothing" (§4). One verify retry served by another model (a router fallback)
makes the whole cell's claims an unknown judge, permanently uncheckable. Either key each verdict by the call that
decided it, and count verdicts by key, or record the majority key with a disagreement count, and say which.

---

## Notes

### 15. Line references that are off

- `dow:src/dow/gm/scouting.py:114` is `DRAFT_FORMAT`. The draft call is `Scout._draft` (`:193-195`). `ScoutDraft` is
  at `:59`, not `:31-80` (`:30` is `DraftFact`). `scout_context` is at `:126` (correct).
- Other spot-checked references hold: `schema/models.py:1828`, `:1000`, `:1143` area, `:5797`; `run/runner.py:185`,
  `:482-486`, `:804`, `:923`, `:1947`, `:2031`, `:2056`, `:3402`; `kernel/metrics.py:380`, `:405`, `:598`, `:2345`;
  `tests/test_open_family.py:89`; `kernel/scoring.py:380-389`; `analysis/first_screen.py:97-112`;
  `ops/regressions.py:20-25`; `kernel/declaration.py:1425-1470`; `schema/versioning.py:509`;
  `kernel/evidence_tiers.py:64-70`; `quick/judged.py:244`; `quick/one_call.py:740`; `kernel/host/measures.py:69`.
- F0 (§10) is correctly diagnosed: `failed_checks` reads `descriptor.check_failed` (`kernel/scoring.py:385-389`) and
  `is_check` ignores `graded_by` (`kernel/metrics.py:598-601`).

### 16. Dropping the ellipsis refusal touches `verbatim.py`, not only the excerpt checks (note, B′)

`verbatim_span` itself refuses an elided needle (`bp-claimx` `kernel/verbatim.py:112`). §11 says "drop `is_elided` in
the excerpt checks; the fold itself is unchanged". Dropping it there is a change to `verbatim_span`'s contract, and it
belongs in the fold identity. §11 also does not say whether `kernel/verbatim.py` (public on `bp-claimx`:
`surface.json`, `_public_api_consumers.txt`) moves and hides with the extractor.

### 17. Recordings: claim calls must be keyed by their prompt (note, C4)

A call that binds inputs is replayed by its inputs, not its prompt (`testing/recorded_completions.py:16-31`). If a claims
call inherits the evidence's `inputs` binding, a changed document replays a stale verdict. Say that claims calls bind
no inputs, or add the documents' digest to them.

### 18. The whole-phase gate can set aside rubric dims that fit before (note, C4)

The phase is admitted whole or set aside whole (`run/runner.py:2002-2011`). A claims ceiling is large: 3 attempts
everywhere, the full repair budget, the full web budget. Under a tight cap, adding a criterion can set aside cells
whose rubric dims alone would have run. Consider admitting rubric dims and each criterion as separate holds.

### 19. Register re-ratings

- **7 (units): ★ N,S.** `unit` is stored on `GradedClaim` and in reading names (`_per_unit`).
- **15: ★ D** (finding 5).
- **19 (`attributed`): ★.** It is stored on every claim and changes R1's numerator (I,S).
- **3: ★, the hardest storage call** (finding 2).
- **Add:** the third `ReadingKind` (finding 4), the claim read's shape (finding 6), and `grounded_in` (finding 8).
- 16 is rightly not ★.
- 17 (fold identity) is rightly I, not ★: splits on a Python upgrade are honest and cheap to explain.

### 20. What holds up

- The boundary (★1) and the separate criterion type (★2) are right.
- So are the four verdicts plus `graded_against`/`nominated_by` (★4), and truth apart from groundedness (★6, with
  finding 3's caveat).
- Per-cell rates over cells (★8), readings derived from a stored tally (★9) and the rubric ranking rule (★11) hold too.
- The DoW section needs no discodon concept in its readings or evidence. Its gaps are the house-rule verdict (finding 1)
  and the quick-path document channel (finding 9), both general. Web off as a first-class regime and
  segments from structured output are real generalizations, and `attributed` is one too.

---

## v4 verification

Read on 2026-11-21: `design-claims-v4.md` (346 lines, written 19:41) against the owner rulings on #1287 since this
review, and against the code at `0bce2461a`. Read-only. v4's new citations were spot-checked and hold. They are
`schema/models.py:1021`, `run/judge_service.py:1036-1039`, `schema/base.py:354-356`, `kernel/storage.py:141-150` and
`:2865`, `run/runner.py:2063-2087`, `schema/models.py:5345-5348`, `quick/answer.py:25`, `run/definition_digests.py:73`,
`analysis/reporting.py:1597`, `analysis/agreement.py:190`, and the DoW references in §11
(`dow:src/dow/gm/scouting.py:59`, `:126`, `:193-195`; `dow:src/dow/gm/answering.py:83-109`).

### Findings 1-20

| # | Status | Note |
|---|---|---|
| 1 | Resolved | §4's table: any case support reads `supported`, and the web decides only judge-suspected claims that no case citation covers. The house-rule and summary controls are in §6 |
| 2 | Resolved, with a new defect | `JudgeEvidence` is untouched, and `ClaimEvidence` sits on the trace outside `_JUDGE_EVIDENCE_FIELDS`. The cap omits the stored web windows (V5) |
| 3 | Resolved | By the owner's ruling: run pages never nominate and are never shown to truth (§4). Per-arm recall, PPV and bias are in §8. See V7 on whether the lean test can fire |
| 4 | Partially resolved | The judged-namespace choice is right (row 21), but the site list misses where judged rows are recognised and scaled (V4) |
| 5 | Resolved, with one gap | `claim_failures` per criterion, never `judge_error`. A cell deadline during claims is unaddressed (V3) |
| 6 | Resolved | `ClaimRead` is its own durable type (17.10) |
| 7 | Resolved | `WebBudget` bounds, `price_ceiling()` per query, refusal of an unpriced port under a cap, a `RoleUsage` judge row, windows stored and re-asked, and the `skipped_cap` caveat with a per-arm flag |
| 8 | Resolved | `grounded_in` (row 24) and the RAG and summarizer columns in §11. The A1 context-only refusal is stated |
| 9 | Partially resolved | `Answer.documents` is added, but a quick eval whose only grader is `Claims` never reaches the judge phase (V1) |
| 10 | Resolved, with a caution | Stratified by verdict, ≤ 2 claims per cell, cells counted, an IPW 2×2, per-arm figures. The allocation and the per-arm test need care (V7) |
| 11 | Resolved, then superseded | The t-path is stated and set-aside is "counted with caveat". The owner since ruled that the bounded share is ranked too, and v4 still lists it as a diagnostic (V2) |
| 12 | Resolved | The fingerprint moves to C4; Q is split out; R0 is a spike; one spend approval for R; the DoW branch is the owner's to merge |
| 13 | Partially resolved | The launch refusal and the pins are stated. The way `unit` is to be left unmeasured does not work for a nested field (V6) |
| 14 | Resolved | Per-verdict `decided_by` and `by_judge` in the tally. A cell holding two keys reads `split`, visible but still matching no check, which is acceptable if stated |
| 15 | Resolved | The DoW references hold |
| 16 | Resolved | The ellipsis refusal is dropped in `verbatim_span` itself, as `verbatim-fold-v2`; `verbatim.py` is hidden too |
| 17 | Resolved | Claims calls bind no inputs. C4 should pin it with a test that a rubric-bound `inputs` context never reaches a claims call (`testing/recorded_completions.py:16-31`) |
| 18 | Resolved | Separate holds per criterion. The order of the holds within one cell (rubric first, then each criterion, each released before the next is asked) is unstated, so say it in C4 |
| 19 | Resolved | The ★ ratings are applied. Rows 21 to 24 are added |
| 20 | n/a | — |

### Owner rulings

- **Run pages never nominate:** followed (§1, §4, row 6).
- **Groundedness ranked, higher is better:** followed. The population is asserted specific claims, case support counts
  only under `grounded_in={case}`, and no research reads 0 (`documents=()` declared). "True but confirmed only by the
  web" counts as ungrounded (`true_ungrounded`, §6).
- **Both truth readings ranked: not followed** (V2). v4 predates the ruling and still lists `_false_unit_share` as a
  diagnostic, with ranking it as owner question 2.
- **Key audit deferred to #1297:** followed (§15 item 3).

### New findings

**V1. A claims-only cell never reaches the judge phase (blocking, C1/C4/Q).** The runner enters judging only when
`candidate.judge_evidence is not None` (`run/runner.py:1886-1887`). The quick path declares `UNJUDGED` whenever no
`Judge` grader is given (`quick/one_call.py:880`). An `UNJUDGED` kind renders no evidence: `hold_to_declaration`
(`run/runner.py:2172`) and the trace validator (`schema/models.py:5925-5937`) both refuse one that does. A DoW rung or a
RAG eval with `grader=[Claims(...)]` and no rubric judge therefore grades no claims and says nothing. Pick one:

- (a) A cell carrying `claim_evidence` enters a claims-only phase whatever its `judged_artifact`. The trace validator,
  `hold_to_declaration` and `judge_witnessed_output` admit `claim_evidence` without `judge_evidence`, and a run with a
  criterion requires a judge service at launch.
- (b) A criterion forces a judged declaration.

(a) is the general one. Either is a C1 shape rule.

**V2. Ruling 2 is not applied, and the tally cannot compute the unit share (blocking, C1/C2).** Rank
`_false_unit_share`: lower is better, `value_range` (0, 1), so it takes the mixture bet; its population and description
need stating; add it to the register and the built controls. `ClaimTally`'s counts (§5) hold `units` and
`asserted_contradicted` but no count of **units with an asserted contradicted claim**, so `claim_readings(tally)` cannot
derive it. Add `units_with_false` (or per-unit counts) to the tally in C1. Also state that a false claim in a unitless
segment counts in R1's numerator but in no unit's share, so the two truth readings can disagree on a summary-line lie.
Name the family-correction effect of three ranked claim readings per criterion.

**V3. A cell deadline during claims grading has no rule (should-fix, C4).** The cell timeout covers judging (the
`_CellSink` judging and pending fields, `run/runner.py:972`). A deadline in the judge phase is charged "to each dim it
cut off" (`run/runner.py:4312-4330`, `_DEADLINE_CHARGE` `:487-493`). Claims grading, with extraction, batches, repairs
and web, is likely the cell's longest stage. State that a deadline striking inside a criterion is that criterion's loss
(`deadline`), never `judge_error`, and that the rubric dims already settled stand. Say whether hosts should raise
`cell_timeout`, or whether the criterion runs under its own deadline.

**V4. The judged namespace has more readers than §7 lists (should-fix, E1).**

- `judged_values` yields only `score` rows with a `rubric_dim`, or the dual axes (`analysis/bundle/cell_reads.py:119-132`).
- `EvalRun.rubric_scales` is read as "the scale each dim was answered on" (`analysis/bundle/trials.py:311`,
  `analysis/bundle/telemetry.py:217`, `analysis/analyst/facts.py:473`).
- `rubric_scale` is read in about 15 analysis modules (`analysis/bundle/bars.py:270`, `analysis/lenses/pivot.py:753`
  among them).
- `resolve_bar_name` step 2 keys judged names to a `RubricScale` (`kernel/declaration.py:1463-1470`).

List these. Pin that a claim row never enters a trial's pass or a rubric threshold share. Make sure the writer's prose
for a judged reading does not describe a per-unit rate as a 1-5 or pass/fail score.

**V5. The cap omits the stored web windows, and an oversized trace is lost quietly (should-fix, C1).** The 1 MiB cap
covers document text. `WebCheckRecord` now stores the windows read: up to
`max_checked_per_cell × max_queries_per_claim × results_per_query × page_chars` on the same trace, plus the grade. The
trace is one row in the host's store. A failed trace write is logged, never raised (`kernel/storage.py:2782-2797`), so
the result then keeps its tallies with no grade or evidence behind them, and re-judge and the claim read are refused.

- Cap the serialized trace, documents and windows together, against a named store limit.
- Make `WebBudget` validation refuse a budget whose window product exceeds what is left.
- Say what a reader shows for tallies without a grade.

**V6. `unit` cannot be left unmeasured by `_UNMEASURED_TEMPLATE_FIELDS` (should-fix, C1).** That set holds **top-level**
template field names, applied as `model_dump(include=...)` (`run/definition_digests.py:73`, `:100-103`). A `unit` nested
inside `claim_criteria[]` is not reachable by it. The stored-content `template_digest` would also see a renamed noun as
drift and write a new revision that never pools (`run/definition_digests.py:60-68`, `TEMPLATE_PRESENTATION_FIELDS`).
Follow the `goal_check_names` precedent (`schema/models.py:1829-1840`): a top-level presentation dict,
`claim_unit_names: dict[criterion, str]`, listed in both `TEMPLATE_PRESENTATION_FIELDS` and
`_UNMEASURED_TEMPLATE_FIELDS`. This is a 17.9 shape choice, so settle it before C1.

**V7. Allocation and the per-arm lean test (should-fix, G).** Stratifying by the instrument's verdict pins precision.
The instrument's **misses** live in its `unverifiable` and `supported` strata: the host's second label moved 25 of 46
unverifiable claims. So the per-arm recall that row 12 relies on is only as good as the draws from those strata.

- Allocate draws to them by expected misses, not 50/50, and oversample `unverifiable`.
- Make the lean a single homogeneity test across arms, not a pairwise "beyond its interval".
- At "≥ 200 per question per arm", a three-arm discodon campaign needs about 1,800 labels for the three questions. Put
  that cost to the owner, or label `<c>.is_claim` unstratified on a smaller sample.

**V8. The witnessed path (note, C4).** `judge_witnessed_output` gates on `output.judge_evidence`
(`run/runner.py:2124-2125`) and must also carry `claim_evidence`, or witnessed cells are never claim-graded.

### Can C1 onward start?

- **F0 and B′:** yes, now. Nothing above touches them.
- **C1:** after four text fixes to v4, all stored-shape rules, none needing new analysis:
  - V1's trace and declaration rule;
  - V2's tally count;
  - V5's trace cap including windows;
  - V6's placement of `unit`.
- **C2:** also needs V2's reading definition.
- **C3:** can start beside C1. It is library-only, and the verdict table is settled.
- **C4 and later:** carry V3, V4, V7 and V8 into their chunks.
