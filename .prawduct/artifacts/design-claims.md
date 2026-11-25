# Design: claim-level factuality grading (#1287), with the key's context tier and the set-grade follow-ups (#1292, #1293)

Status: revision 2, 2026-11-21, after the independent review (`design-claims-review.md`, 25 findings, every one
dispositioned below as R<n>). Owner go 2026-11-21 (direct, this session).

## Goals

What this is for: discodon's Kairo research and storytelling evals read `research.grounded` 0/5 on every arm, so a
reply-level pass/fail cannot rank arms. Kairo says checkable things on air, and 8 to 12% of them are false (discodon's
second-labelled set). The owner wants readings of how true a persona's segues are, per stop, that rank arms.

Success: a host grades a cell's utterances into atomic claims, each with a verdict and the evidence behind it, and
lands R1 and R2 (below) as ranked host measures; the grade object carries every claim, verdict, quote and search for
audit, and the per-stop detail discodon counts R3 and R4 from.

Out of scope: the quick path (follow-up); R3/R4 counts and the track-relevance tag (discodon's, per the goals doc's
"who does what"; this grade exposes what they count); pairwise R6 (#1295); live feedback (#1296); the fact ledger
(§7, designed for, filed as its own issue); the D4 `in_findings` diagnostic (R4: deferred to its own issue, named in
the CHANGELOG, because reading the findings in the same judge call risks contaminating verdicts).

### Owner rulings followed (2026-11-21)
- D1 per stop, no set pass/fail, no composite. D3a a false statement is a rate, never a veto. D3b hedged colour is
  acceptable; hedged claims are graded for truth. D4 truth primary; findings are never evidence.
- D7 (refined by the owner in this session, R2): the key and fetched pages are authoritative. The web is searched for
  (a) claims the key or a page contradicts and (b) claims no evidence covers that the judge nominates as suspect.
  Model memory only nominates; a verbatim quote from a returned page decides. Web shows the claim correct ->
  `ungrounded_not_untrue`. Spend is about the false rate plus false alarms, capped per cell.
- Context tier: facts about entities outside the answer set live on a third key tier set grading never counts.
- Goals doc readings: R1 false statements = asserted claims contradicted and not rescued, listed verbatim, a rate per
  stop. R2 unsupported specifics as a share of asserted specific claims. R1-R6 are ranked; no composite.

### Inferences (stated, for the boundary review)
- I1 Per-stop readings are per-cell rates averaged over cells (the cell is the unit; claims cluster in cells). A 3-stop
  set weighs like a 12-stop set ("per set heard"). Per-cell counts are small, so intervals are wide at few cases.
- I4 3tears is a public repo: no prod transcript, listener request or host-labelled claim is committed. Tests are
  synthetic, on the labelled error classes. The agreement measurement against discodon's set runs host-side (#3526),
  with owner spend approval.

## 1. Key: a context tier (chunk A1)
`ReferenceKey.context: tuple[ReferenceEntity, ...] = ()`, declared with `exclude_if=lambda value: not value` (the
`EvalTestCase.reference` and `MetricDescriptor.pass_fail` precedent), so a key without context serialises and digests
exactly as today (R1: `_case_digest` dumps the key without excluding defaults, `run/definition_digests.py:144-146`).
- `entities` keeps meaning the answer set (core + acceptable), which `SetGrader` and `FoldingMatcher` read. A new
  accessor returns every fact-bearing entity (three tiers). Folded names and aliases unique across all three tiers.
- `reference_material` renders no context lines when the tier is empty; when present, under a heading saying they are
  not answers ("Context (facts only; not answers):").
- Pins captured from the current build BEFORE the field lands, on a keyed case: `content_hash` and seeded id,
  `seed_digest`, and the exact `reference_material` text. All three unchanged after.
- Durable step: `EvalTestCase` is durable, so `DURABLE_SCHEMA_MINOR` moves to 8 with its ledger line (R15). Reach-back,
  both paths, mirroring the 17.7 entry: a pre-change build refuses a corpus key carrying `context` at seed
  (`extra_forbidden`, by name); reading a stored case at the newer minor it keeps `context` opaquely and leaves it out of
  the digest, so after a rollback a case differing only in context reads `current` (verified on the base build by A1
  and its Critic); one differing in other content too reads `customized`, as in 17.7. Host rollout: upgrade before shipping a corpus with context; roll the corpus back with
  the package.
- A key still needs one core or acceptable entity (R20: storytelling cases have an answer set; kept on purpose).

## 2. Set-grade follow-ups (chunk A2, after A1)
- #1293: `EntityMatcher.match` may return a match with a reason (a tier label) or a refusal with a reason. `SetGrade`
  records per-pick provenance (written name -> key entity, how) and refusals with reasons, apart from names the key
  lacks. Counts unchanged: a refusal is an unverified pick; `set grade v1` stays; a test pins the counts. A plain
  `str | None` return still works if that costs nothing; otherwise the breaking change and the migration are named in
  "For hosts" (R17: discodon implements the protocol). `FoldingMatcher` refuses a name that is a context entity, with
  reason `context`. Whether a context pick ever counts as wrong is an owner question, not built.
- #1292: promote the seven names discodon imports to stable after the protocol change, in the same chunk; regenerate
  the discodon host-imports fixture with `record_host_imports.py` from discodon's source (`/opt/venv`, its develop).

## 3. Claim extraction (chunk B, parallel with A)
`ClaimExtractor(client, ...)`, shaped like `EntityExtractor`: judge ledger, a cut-short reply raises, identity = model +
digest of prompts, schema and the verbatim fold.
- Input: the cell's utterances in order, `Utterance(text, stop_index: int | None)`: a segue carries its stop's index
  (0..n-1, unique, validated), any other utterance (a message, a voice line) carries None (R5). The host names its
  segmentation with an identity string that enters the formula (R18). The extractor never sees key, sources or findings.
- Output per claim: the utterance it came from, `claim` (one atomic proposition), `excerpt`, `specific: bool`,
  `hedged: bool`, `restates: int | None` (an earlier claim of the cell stating the same fact).
- Verbatim: one helper `verbatim_span(needle, haystack) -> (start, end) | None` folds both sides the same way (NFKC,
  casefold, typographic quotes and dashes to ASCII, soft hyphen and zero-width characters removed, whitespace runs
  collapsed) and maps the match back to offsets in the original. An ellipsis inside a needle is refused. The stored
  excerpt is the original slice at those offsets (R8). Each character class is tested.
- Repair, not refusal (R6): items that fail a check (an excerpt not found, a `restates` that does not point earlier) are
  re-asked alone with the failure named, two repair rounds, then `ClaimExtractionFailed(ApparatusError)`. A failing
  item is never dropped or downgraded.
- Prompt rules restated generically (no host examples): descriptions, opinions, station chatter and echoes of the
  request are not claims; compounds are split; a framing's implied claim counts.

## 4. Claim grading: the evidence step (chunk C2, after A1, B and C1)
`ClaimGrader(prefix, *, reader_label, extractor, judge_client, evidence: Literal["key", "key+sources"],
page_budget=..., web_check=None, web_budget=...)` declares and grades, like `SetGrader`.
`grade(utterances, key, sources, *, ledger) -> ClaimGrade`.
- Evidence regime in the identity (R3): a `key` grader refuses non-empty sources; a `key+sources` grader takes an
  explicit tuple, empty when research fetched nothing (recorded, not refused). The grade records sources count and
  characters read.
- Prompt evidence is numbered: key facts `K1..` (entity, claim, source_url, from all three tiers) and passages `S1..`
  (R10). Pages are cut by a deterministic budget in code: per-source and total character caps and a passage selection
  around folded key names, aliases and the claim's excerpt. Caps and selection enter the identity; truncation is
  recorded; quotes are checked against the full stored text; the grade stores quotes and offsets, never page text (R7).
- Batches: one call per stop (non-segue utterances as one batch), each with its own repair budget. A restatement is
  batched with its original and takes the original's verdict in code (R12).
- Per claim the judge returns a step-1 finding: `supported` (cites K or S), `contradicted` (cites K or S), `suspect`
  (no evidence covers it, and the judge doubts it), or `unverifiable`. A cited S carries a quote of at most one sentence
  or table row (≤ 300 chars) with `subject_span` and `value_span` verbatim inside it (R9), so a value from another row
  is unrepresentable, and a contradiction must cite the conflicting value. The prompt states that a fact the key omits
  contradicts nothing (R21).
- Code checks: cited indices exist; quote and spans verbatim (the helper); a supported or contradicted finding without
  a citation is a repair item. Entity names resolve by `fold_name`.

## 5. The web check (chunk D, after C2)
Runs for step-1 `contradicted` and `suspect` claims only. `WebCheck` is a host port: `search(query) -> pages (url, title,
text, snippet_only: bool)`; the judge writes the queries.
- `max_queries_per_claim` and `max_checked_per_cell` enter the formula (R11). Claims beyond the cell cap are not
  searched: their verdict is their step-1 finding mapped (contradicted stays contradicted, suspect -> unverifiable),
  marked `web: skipped_cap`, and counted (`<p>_web_skipped`). Never an apparatus raise, which would exclude the most
  false-heavy cells (survivorship).
- A web verdict requires a verbatim quote with spans from a returned page (the helper), or it is a repair item. Model
  memory never decides.
- Final verdicts: web shows correct -> `ungrounded_not_untrue` (with `evidence_conflict` set when step 1 was a K or S
  contradiction: a key or page defect the host should see); web shows false -> `contradicted`, graded against `web`
  (or `key`/`source` when step 1 cited one); inconclusive -> step-1 contradicted stays contradicted, suspect becomes
  `unverifiable`. No web check configured: `web none` in the formula, and suspects are unverifiable.
- Recorded per checked claim: every query, every url returned, the deciding quote. The grade object is the record; a
  re-measure reuses it and never re-searches (the web is not stationary, R18).

## 6. The grade object and readings (chunk C1, code only, before C2)
`ClaimGrade`: every claim (utterance, stop, excerpt and offsets, tags, verdict, graded_against, citations, quotes and
spans, web record), the false list (asserted contradicted claims, verbatim), per-stop counts, stops, sources read.
Readings, all in one judge-graded claim family; no pass/fail measure:
- R1 `<p>_false_per_stop`: asserted contradicted claims per segue stop, each restatement counted as heard (R12).
  Lower is better; ranked.
- R2 `<p>_unsupported_share`: of asserted specific claims, the share not supported (`ungrounded_not_untrue` counts as
  unsupported, R13); population = cells with at least one. Lower is better; ranked.
- Diagnostics (no direction): `<p>_false` (count), `<p>_false_hedged` (hedged contradicted, R4), `<p>_false_share`
  (contradicted of all claims, R14), `<p>_claims`, `<p>_stops` (R23), `<p>_hedged_share` (R13),
  `<p>_unverifiable`, `<p>_ungrounded_not_untrue`, `<p>_web_checked`, `<p>_web_skipped`.
- Each description states its population and that the per-cell value is a point; R1's "range" is the grader's
  calibration error, which the host states from #3526 (R4), not this interval.
- C1 builds these from hand-built verdicts and lands the owner's built controls as scripted tests: a planted lie, a
  hedged lie, a bland set, a repetitive set and an invented vivid detail each move only their own reading.
- C1 confirms which surface ranks arms on a judge-graded host family (R16: the bundle admits code-graded families only
  and regressions never fire on judge-graded measures; the 17.7 CHANGELOG says extracted-set measures are "never ranked
  on"). If the campaign contrast does not rank R1/R2, the builder stops and reports before building around it.

## 7. Identity and pooling (R3, R18)
The formula names: the grading revision, the extractor's identity, the judge's (model + prompt digest), the
evidence regime, page caps and selection, the batch rule, the verbatim fold, the host's segmentation identity, the web
check's identity (host-named, or `none`) and its budgets. Judge sampling parameters are the host client's to keep
fixed; the CHANGELOG says so. Later, the fact ledger (owner idea): web-confirmed facts written to a versioned document
outside case identity; whether its revision enters the formula (splitting on growth) or is recorded beside it (pooling
across growth) is that issue's decision, named now.

## Chunks and order
A1 context tier -> A2 #1293 + #1292. B extraction (parallel with A). C1 grade object and readings (after B's claim
type). C2 evidence step (after A1, C1). D web check (after C2). One builder each (Opus), scripted clients only, no
spend, CHANGELOG + MIGRATIONS where shapes move, Critic per chunk, integration lead merges on evals/next with the
wide gate once per batch. Calibration against discodon's set: host-side, after D, owner spend approval.
