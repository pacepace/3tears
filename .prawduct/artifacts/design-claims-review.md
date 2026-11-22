# Review: design-claims.md (claim-level factuality grading, #1287 + context tier + #1292/#1293)

Reviewer: independent design review, 2026-11-21. Read: the design; `kernel/set_grade.py`, `kernel/entity_extraction.py`,
`kernel/reference.py`, `schema/reference.py`, `kernel/metrics.py`, `run/definition_digests.py`, `schema/base.py`,
`schema/versioning.py`, `analysis/bundle/measures.py`, `ops/regressions.py`, the two Unreleased CHANGELOG sections; the
host's labelling guide and the host's own rulings text (discodon `documentation/evals/decisions.md` and its storytelling
goals doc, readings table R1-R6). No host data is copied here; host figures are cited only as aggregates.

Severity: **blocking** (resolve before the chunk it governs is built), **should-fix** (resolve in the design or its
chunk), **note**.

---

## Blocking

### 1. The context tier's "keeps its digest exactly" has no mechanism, and the obvious implementation breaks it on three surfaces (blocking, chunk A)

Evidence:
- `run/definition_digests.py:144-146`: `_case_digest` dumps the key with
  `case.model_dump(mode="json", include={"reference"})`, **without** `exclude_defaults`. `_content`'s
  "a field at its default is left out" rule (`:57`) does not apply to template cases. So a field
  `context: tuple[ReferenceEntity, ...] = ()` serialises as `"context": []` inside the key, and every keyed case's
  `content_hash`, its seeded id (`run/definition_seed.py:306-315`) and its `seed_digest` move. Every keyed case then
  reads stale in drift and resets under a new id.
- The stored document (`to_dict`) gains `"context": []` on every keyed case too, so "stored exactly as before" fails.
- `kernel/reference.py:48-54`: `reference_material` renders the key for every judge grading against it, and that text
  enters the judge's label fingerprint. `_part_lines` prints `"<heading> none."` for an empty part (`:33-34`). A builder
  following that pattern for a context heading would move every keyed judge fingerprint (discodon's hedged grader calls
  it: `discodon/evals/kinds/persona/hedged.py:189`). The design does not mention `reference_material` at all.
- The only pinned digest today is a case **without** a key (`tests/test_reference_key.py:213-218`). Nothing pins a
  keyed case, so nothing would catch the drift except the new test the design promises, and that test covers only the
  first surface.

Recommendation: state the mechanism in §1. Declare `context` with `exclude_if=lambda value: not value`, the precedent
`EvalTestCase.reference` (`schema/models.py:2607-2609`) and `MetricDescriptor.pass_fail` (`kernel/metrics.py:565-567`)
set. `reference_material` emits no context lines when the tier is empty. When the tier is present it uses a heading
that says the entities are **not** part of the answer, e.g. "Context (facts only; not answers):". The non-exhaustive
line at `kernel/reference.py:15-18` otherwise invites a judge to treat a context entity as an acceptable pick. Pin a
keyed case three ways, captured from the current build **before** the field lands: `content_hash` and seeded id,
`seed_digest`, and the exact `reference_material` text.

### 2. I2 likely misreads D7, and it makes `ungrounded_not_untrue` mean something other than the owner's words (blocking, chunks C and D; needs an owner ruling)

Evidence: design lines 24-26 and 35-37, §3 lines 83-88. The owner's text: the key and fetched pages are
authoritative. Before a claim is graded untrue, the grader searches the web. If the web shows it correct, it is
"ungrounded but not untrue". The check runs only on claims "about to be graded untrue", with spend about 8 to 12% of
claims.

Under I2 a claim is contradicted only when a key fact or a fetched page states otherwise. Three consequences:
- **"Ungrounded" becomes a misnomer.** Under I2 the web rescue fires only when the key or a page *contradicts* the
  claim and the web disagrees with them: a conflict inside the authoritative evidence, i.e. a key or page defect. The
  owner's word "ungrounded" describes a claim the grounding does not support, which then turns out true. That is a
  claim the grader suspected false with no contradicting evidence. Under I2 that claim is unreachable: it is graded
  `unverifiable` and never checked.
- **The spend figure the owner gave is the overall false rate.** The 8 to 12% is the human labellers' false share, and
  the labellers verified every claim on the web. Bounding the check to that share implies the owner expects it to reach
  roughly all false claims. Under I2 it reaches only those the key or pages cover. Until discodon #3443 lands, sources
  are empty and that is the key alone.
- **Calibration will read low recall on contradicted by construction.** The human labels are web-derived, so most
  human-contradicted claims the key does not cover come out `unverifiable`. The agreement target (I4, line 41) is
  undefined until the design says which claims it is measured on.

Reading the design misses: **memory nominates, the web decides.** Step 1 may flag a claim with no key or page evidence
as `suspect_false`. That flag is not a verdict. Step 2 searches, and a verbatim quote from a returned page then
decides: contradicted (`graded_against: web`), ungrounded_not_untrue (web supports), or unverifiable (inconclusive). A
key or page contradiction still goes through the same web check, and a web rescue there is recorded as an evidence
conflict (a key defect the host should see, and the ledger's first input). This reading never grades from memory. Its
spend is bounded by the nominated count, which is about the false rate plus false alarms, and it can be capped (see 11).

Recommendation: put both readings to the owner as one question before C. Shape the verdict record so either reading is
expressible (`graded_against ∈ {key, source, web, none}`, plus a `nominated_by` field), so that D does not reshape C.

### 3. Whether the grade had page text is not part of the measure's identity, so key-only and key-plus-pages grades pool (blocking, chunk C)

Evidence: §3 line 74 ("empty until discodon #3443") and §5 lines 102-103. The formula names the extractor, judge, web
check and revision, but not the evidence regime. With sources empty, every claim the key does not cover is
`unverifiable`. That is most of them, since a key bounds a few facts per entity. Once #3443 lands, the same claims
become supported or contradicted on page quotes. `false_per_stop` and `unsupported_share` move a lot under the same
formula, so runs from before and after #3443 pool and pair. That is the "changed definition that never pools" rule
(set_grade.py:21-22) broken by an input the host can switch.

Recommendation: the grader is constructed with the evidence it reads, `evidence="key"` or `"key+sources"`, and that
enters the formula. A key-only grader refuses non-empty sources, and a key-plus-sources grader refuses a cell whose
sources are absent. A cell where research fetched nothing passes an explicit empty tuple, which is recorded, not
refused. Also record `sources_count` and the total characters on the grade, and state the regime in each descriptor's
description.

### 4. Owner readings dropped or contradicted without saying so (blocking, chunk C; "never silently drop a requirement")

The host's storytelling goals doc, which the D-rulings commit names as "the design behind these rulings", lists readings
R1-R6 and says "R1-R6 are the readings an arm is ranked on". Against §4 (lines 91-99):
- **D4 diagnostic dropped.** "Came from the findings is a diagnostic" (line 23) maps to nothing. §3 line 75 keeps the
  findings out of the judge's evidence, which is right. But no `in_findings` diagnostic exists. The labelling guide
  records it per claim, and it is the one provenance reading the owner kept. Either add it as a separate code or model
  step that never feeds the verdict (the host passes `findings` text; a per-claim `in_findings: yes/partial/no`; a
  diagnostic share), or say explicitly that it is deferred, with an issue.
- **R1 counts asserted claims, and the design counts hedged ones too.** Line 95 puts hedged claims "inside `_false`".
  The goals doc's R1 is "asserted claims the key or a fetched page contradicts and a web check does not rescue", and it
  orders severity as contradicted asserted > contradicted hedged > unverifiable vivid > unverifiable generic. Split
  `_false` (asserted) from `_false_hedged`, so the ordering is readable without a composite.
- **R1 "read as a range".** The design produces a point per cell and the engine's interval over cells. The owner's range
  (8 to 12%) is grader error, from calibration, not sampling error. Say which one R1's range is, and if it is the
  calibration range, where it is stated (descriptor description, or a disclosure fed from discodon #3526).
- **R3 substance and R4 repetition are ranked readings, and the design makes them directionless.** Lines 96-97 make
  `specifics_per_stop` and `restatements_per_stop` diagnostics "unless the owner says", but the owner's table already
  says. R3 is also "specific **track-relevant** claims per stop; stops with nothing track-specific". The extractor has
  no track-relevance tag and no empty-stop count. R4 adds "the worst-repeated fact and its count". Either build them
  (`track_relevant: bool` on a claim, needing the stop's track identity in `Stop`; `<p>_stops_without_specifics`;
  `<p>_worst_restated`) or list each as deferred.

---

## Should-fix

### 5. Utterances that are not segues have no home (should-fix, chunk B)

Evidence: line 61 (`Stop(index, text)`, "one stop for prose with none"). The labelling guide counts every on-air
utterance: messages, voice and segues. In the host's labelled set about 9% of claims sit on non-segue utterances, and so
do about the same share of the contradicted ones. If the host drops them, those false statements vanish. If it makes
each one a stop, the per-stop denominator is inflated by chatter.

Recommendation: `Stop(index: int | None, text, kind)`, or a separate `preamble` list. Claims from every utterance are
counted, and the per-stop denominator is the segue stops only. Say so in each per-stop descriptor. Validate stop indices
as 0..n-1 with no duplicates.

### 6. One bad excerpt or quote fails the whole cell, and failures correlate with the thing being ranked (should-fix, chunks B and C)

Evidence: line 59 (unreadable retried twice, then `ApparatusError`), line 63 (an excerpt miss is unreadable), lines
79-80 (a quote miss or missing evidence is unreadable). `EntityExtractor` re-asks the **whole** request
(`entity_extraction.py:164-176`). A cell has about 30 claims (host set: median about 30, up to 40), each with an
excerpt, plus a quote on every supported or contradicted verdict. At even a 1% per-item miss rate most cells miss
something, three full retries triple spend, and `apparatus_failed` excludes the cell. Exclusion grows with claim count,
specificity and fetched-page volume: claim-rich, research-heavy arms lose more cells, which is survivorship on exactly
the readings being compared. A cut-short reply raises at once (`entity_extraction.py:167-169`) and is more likely with
long pages (see 7).

Recommendation:
- Re-ask **only the failing items**, with the failure named ("excerpt for claim 12 is not in stop 3"). Two repair rounds,
  then raise. Never degrade a failing item to `unverifiable`: that would lower the false count, which is perverse.
- Grade in fixed batches (per stop, or N claims), each with its own retry budget. The batch rule enters the identity.
- State in the host notes that the runner's per-arm apparatus-exclusion count must be read beside these measures.

### 7. Long fetched pages: context length and determinism (should-fix, chunk C)

Evidence: §3 lines 73-76: every key fact plus every page, for every claim batch, in one call. A research-heavy cell can
fetch many pages of tens of thousands of tokens each. Nothing caps or selects them.

Recommendation: decide a deterministic page budget in code: a per-source and a total character cap, plus a passage
selection (for example, windows around occurrences of folded entity names and aliases from the key and the claim's
excerpt). The quote check still runs against the **full** stored text. Record truncation on the grade. The caps and the
selection rule enter the judge identity, since changing them changes verdicts. Never copy page text into the grade, only
quotes and offsets (keeps `kind_payload` small).

### 8. Verbatim checks: the normalisation must be specified, and `fold_name` is not enough (should-fix, chunks B and C)

Evidence: `fold_name` (`schema/reference.py:12-14`) is NFKC, casefold and whitespace runs. NFKC leaves `’` `“` `–` `—`,
the soft hyphen (U+00AD) and the zero-width space (U+200B) unchanged (checked). Models routinely straighten curly
quotes, turn dashes into hyphens, drop soft hyphens, change case, and elide with `…`. Fetched page text carries
entities (`&amp;`), NBSP and soft hyphens. Spoken text renders numbers as words.

Recommendation: one helper, `verbatim_span(needle, haystack) -> (start, end) | None`. It applies the same fold to both
sides (NFKC, casefold, typographic quotes and dashes mapped to ASCII, soft-hyphen and zero-width characters removed,
whitespace runs collapsed) and maps the match back to offsets in the original. Store the offsets; the excerpt shown is
the original slice, not the model's string. Refuse an ellipsis inside a needle (an elided quote is not verbatim). Test
each character class. The fold enters the extractor's and judge's identity, so changing it is a new definition.

### 9. The "binds the value to this claim's subject" rule is prompt-only, and scripted tests cannot test a prompt (should-fix, chunk C)

Evidence: lines 79-82. Code checks substring presence only. A scripted client returns whatever the test scripts, so the
neighbouring-value regression tests prove the code path and nothing about the model. The failure discodon found (a
neighbouring value on the same page) passes every code check.

Recommendation: add code-checkable structure. The judge returns `subject_span` and `value_span`, both verbatim inside
its quote, and the quote is bounded (one sentence or table row, ≤ ~300 characters). That makes "a value from another
row, 2,000 characters away" unrepresentable rather than discouraged. Call the scripted tests what they are, code-path
tests, and name the neighbouring-value shapes as a calibration stratum for discodon #3526.

### 10. Cite evidence by index, not by restating it (should-fix, chunk C)

Evidence: line 77 ("the key fact (entity, claim) or the source url"). A key fact has no id. The judge would have to
reproduce the fact's claim text exactly, and long URLs with query strings get mangled. Each mismatch is an unreadable
reply (see 6).

Recommendation: number every key fact (`K7`) and every source (`S3`) in the prompt, and have the judge cite indices.
Code resolves the indices to (entity, fact) and url, and the grade stores the resolved values.

### 11. The web step needs the same evidence discipline, a budget, and the confirming quote (should-fix, chunk D)

Evidence: lines 83-87. The web verdict is "one model step reads the pages", and no quote is required. The queries are
"queries the judge writes", with no count bound. The record keeps query, url and verdict.

Recommendation:
- A rescue (`ungrounded_not_untrue`), or under finding 2's reading a web contradiction, requires a verbatim quote from
  a returned page (finding 8 helper), or it is refused. Otherwise model memory rescues claims.
- `max_queries_per_claim` and `max_checked_per_cell` are constructor arguments and enter the formula. The host's notes
  show search effort moved the human labels substantially between two labellers, so effort is part of the measure.
- If a cell's checks exceed the cap, the grade raises (apparatus). It never grades silently past the cap.
- Record the confirming quote with its url, so a later ledger entry (§6) has evidence, not just a verdict.
- A host `WebCheck` returning snippets only must say so. The quote is checked against what was returned.

### 12. Restatements: count them, and keep their verdicts consistent (should-fix, chunks B and C)

Evidence: line 66 (`restates`), line 97. The design does not say whether a restated false claim counts once or twice in
`_false`. The owner's notes say "one wrong fact restated is two falsehoods heard", and repeats are contradicted at twice
the base rate. In the host's labelled set every restatement carries its original's label.

Recommendation: count each utterance in `_false` (as heard), and say so in the descriptor. A restatement either takes
its original's verdict in code, or the grade refuses a disagreement as unreadable. Grading the same fact twice is
needless spend and an inconsistency waiting to happen. Batch the original and its restatements together.

### 13. `unsupported_share`: population and numerator under-specified, hedging moves claims out of it (should-fix, chunk C)

Evidence: line 94.
- The numerator is "not supported", and it is not stated whether `ungrounded_not_untrue` counts. Under D7 the key and
  pages are authoritative, so it should count as unsupported. Say so.
- "Cells with at least one" asserted specific claim: an arm that hedges more moves claims out, and an arm with all-hedged
  cells has no reading on them. D3b sanctions hedged colour, so this is owner-intended. But the owner's notes also say
  the hedge has become a verbal tic, not a signal. Add a `<p>_hedged_share` diagnostic, so a reader can see an arm
  buying this reading with hedges.
- Equal weight per cell: a cell with one asserted specific claim reads 0 or 1 and weighs as much as one with 30.
  Acceptable under I1, but the description should state the population, and the host notes should state how a contrast
  treats cells missing the reading. Arms are otherwise compared over different case subsets.

### 14. `_false` as a ranked raw count rewards short sets; a false share is missing (should-fix, chunk C)

Evidence: line 93. A count with `higher_is_better=False` ranks an arm that delivers fewer stops above one that delivers
more, at the same rate. `false_per_stop` also ranks an arm that says no specifics as best. R2 (the share) is the
counterweight, which is why R3 must carry a direction (finding 4).

Recommendation: make `<p>_false` a diagnostic (a count, no direction). Add `<p>_false_share` (contradicted of claims
graded supported or contradicted, or of all claims; pick and state one) as a diagnostic at least. It is the owner's own
framing ("8 to 12% of them are false"), and it is what calibration compares against.

### 15. The reach-back statement is half true, and the durable minor bump is missing (should-fix, chunk A)

Evidence: line 52 ("a build before this refuses a key carrying `context` by name (`extra_forbidden`)"). True of the
corpus path (`SeedTemplateCase`, boot seed). False of a stored read. `ReferenceKey` is a `CoreShapeModel`
(`schema/base.py:341-359`), so a 17.7 build reading a stored case at a newer durable minor **keeps** `context` opaquely
and writes it back. Its drift digests the case without the tier, which does not match the recorded `seed_digest`, so
the case reads `customized`. That is the same shape the 17.7 CHANGELOG entry documents (`CHANGELOG.md:24-32`).
`EvalTestCase` is durable, so `tests/test_durable_schema_pins.py` goes red until `DURABLE_SCHEMA_MINOR` moves to 8 with a
ledger line carrying the additivity sentence (`schema/versioning.py:982-990`).

Recommendation: §1 states the durable 17.8 step, both reach-back paths and the drift outcome after a rollback, mirroring
the 17.7 entry.

### 16. Name the surface that ranks arms on a judge-graded family (should-fix, chunk C)

Evidence: line 91 puts every reading in a judge-graded family like `EXTRACTED_SET_FAMILY`. The bundle admits only
code-graded families (`analysis/bundle/measures.py:136-140,182`), so the analyst generator never sees these readings.
Regression alerts never fire on a judge-graded host measure (`ops/regressions.py:25`). The 17.7 CHANGELOG says an
extracted set's measures are "judge-graded and never ranked on" (`CHANGELOG.md:64-65`). The goal (line 14) is readings
"that rank arms".

Recommendation: state which surface ranks arms on these readings (the campaign contrast over host measures with a
direction), confirm it reads a host judge family, and correct the CHANGELOG sentence if "never ranked on" means only
"never in the bundle". If regressions should fire on `false_per_stop`, that is a feature this design must add or defer
by name.

### 17. #1293 breaks discodon's matcher, and context hits need a reason (should-fix, chunk A)

Evidence: line 111. `EntityMatcher.match` returning a match-or-refusal-with-reason changes the protocol
(`set_grade.py:112-119`), and discodon implements it (`discodon/evals/kinds/persona/set_matching.py:111`). A host matcher
that reads all tiers and returns a context entity's name raises at `set_grade.py:434-438`.

Recommendation: the host notes name the breaking change and the migration. One of the refusal reasons is "names a
context entity" (`FoldingMatcher` can detect it through the new all-tiers accessor). Counts stay unchanged per line
113. Whether a context pick should ever count as wrong is an owner question, not a builder's.

### 18. What changes the grade without changing the formula (should-fix, chunks B to D)

Beyond finding 3:
- **The host's stop segmentation.** How prose becomes `Stop`s sets every per-stop denominator, and nothing names it.
  Have the host pass a `segmentation` identity, or at least record stops per cell on the grade.
- **Batch composition** (finding 6), **page caps and selection** (finding 7), **the verbatim fold** (finding 8) and
  **the web budget** (finding 11): each enters the identity.
- **Judge sampling parameters.** The `EntityExtractor` precedent names model plus prompt digest only
  (`entity_extraction.py:125-138`). Temperature and output cap are not in it. Name them, or state that the client's
  configuration is the host's to keep fixed.
- **Fact ledger (§6).** "Each grade records the ledger revision it read" (line 107). A recorded revision outside the
  formula means grades read against two ledgers pool. Inside the formula it splits readings on each growth, which is the
  objection §6 raises against the key. The later issue must choose, and the choice should be named now.
- **The web is not stationary.** A re-grade reproduces nothing. Say that the grade object is the record, and that a
  re-measure reuses it and never re-searches.

### 19. Chunk A bundles three unrelated risks; C is the largest (should-fix)

Evidence: lines 118-122. A holds a durable schema step (digest pins, reach-back, `reference_material`), a breaking
protocol change discodon implements, and API promotions with a fixture regeneration. C holds the prompt, evidence
checks, grade object, readings, descriptors and family.

Recommendation:
- Split A into **A1**, the context tier (schema, durable 17.8, digest and material pins), and **A2**, #1293 then #1292.
  Promote after the protocol change, in the same chunk, so a promoted name is not changed immediately afterward.
- Split C into **C1**, the grade object, readings and descriptors, fed by hand-built verdicts. That is pure code, and
  where the owner's built controls land as scripted tests: each of planted lie, hedged lie, bland, repetitive and
  invented vivid detail moves only its own reading. **C2** is the judge step and its evidence checks.
- Fix the full formula, including the `web none` segment and the evidence regime, in C, so that D does not change C's
  definition. B can run in parallel with A as stated: the extractor never sees the key.

---

## Notes

### 20. A context-only key is refused (note, chunk A)
`schema/reference.py:73-74` requires an entity in core or acceptable. A storytelling case whose ask has no answer set
cannot carry facts in context alone. Decide whether that case exists, and keep or relax the rule on purpose.

### 21. A key's facts are non-exhaustive too (note, chunk C)
The judge prompt must say that a fact the key omits contradicts nothing. Only a stated conflicting value contradicts.
A line-up fact naming three members does not contradict a claim of a fourth. Finding 9's `value_span` makes this
checkable: a contradiction must cite the conflicting value.

### 22. I1 is sound as an estimand, with caveats worth stating (note)
A per-cell rate averaged over cells is the clustered estimand: the cell is the unit, and claims within a cell are not
independent. Stating it is right. Caveats for the descriptions:
- Per-cell counts are small (host set: a median of one or two contradicted claims over about five or six stops), so the
  per-cell rates are coarse and the interval is wide at a handful of cases.
- The same few cases recur across replicates, so the effective n is cases, not cells, if the engine's interval does not
  already cluster by case.
- Mean-of-ratios weights a 3-stop set like a 12-stop set. That is defensible ("per set heard") but should be said.

### 23. Zero-stop cells make no reading (note)
Line 92. An arm that delivers no set leaves no reading on these measures, so it reads no worse on truth. Fit and recall
catch it elsewhere, but add a `<p>_stops` diagnostic so the absence is visible beside the rates.

### 24. Under I2, the D7 bound holds trivially (note)
Under I2, step-1 contradicted claims are a subset of key- and page-covered claims, so the web spend is bounded and
reachable whenever a key fact conflicts. Until keys are deep and sources populated, the readings mostly measure key
coverage. The owner's "6 to 10 deep cases" plan depends on that, so it should be named in the descriptions.

### 25. Extraction tags (note, chunk B)
`specific` is not in the labelling guide, which labels every checkable proposition as a claim. So the "specific" cut
has no human label to calibrate against. Either label a sample host-side, or say the tag is uncalibrated.
`hedged` matches the guide's definition. Keep the guide's examples generic in the prompt (I4).
