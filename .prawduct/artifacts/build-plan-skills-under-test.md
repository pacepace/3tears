---
artifact: build-plan
version: 2
scope: skills-under-test
branch: evals/skills-under-test
base: report-kit/analyst
working-branches: []
last_validated: 2026-10-31
partition: 01 then 02 serial (02 builds on 01's seam); 03 and 04 delegated in isolated worktrees off 02 (03 owns the scoring judge, 04 the case bank and sets; disjoint files); 05 serial in the main session; 06 is a live spend, main session only, after the owner names a cap
---

## Goals

**Response taken:** Ask, in one batch (2026-10-31). The owner answered four questions. Issue #1009 (P3, parked 2026-10-28) is now taken up by the owner as "the last barrier to self-improving evals".

**Product.** Any change to how the eval system works becomes a measured decision, made by the engine's own corrected comparison, never a judgement call. Wave 1 (this plan) builds the gate. Wave 2 (its own plan, drawn when it starts) builds the search that feeds the gate.

**Owner rulings (2026-10-31, direct):**
- Scope: gate first, then search. The search must cover **arm settings** (model, reasoning effort, quantization and so on) as well as skill text: "self improving evals as a system", not "evolutionary skill writing". So the gate is generic over a kind's levers, and the report writer is its first subject, not its only one.
- Judge: a **fixed scoring judge**. Its rubric text and model hold for the whole campaign, from a different model family than the writer. The in-loop editor stays part of the arm under test. Its scores carry the existing uncalibrated-judge disclosure.
- Model: the A/B campaigns run the writer on **claude-haiku-5-5** (provider anthropic, ANTHROPIC_API_KEY from env, no temperature).
- Infra first (2026-10-31): "we need to do the infra first, and not try to use our limited data for actual changes to the eval system." No skill or setting is changed on today's cases. More cases and diversity come from integrating evals into metallm, so cases arrive from hosts, not only from fixtures.
- Promotion: **package files plus a recorded decision**. The skill files stay the source of truth. Promotion is a commit that changes them and cites the campaign decision, and a test checks that the shipped digest has a recorded decision. Hosts get an override seam only to run candidates.

**Inferred (not owner words):**
- Cases: Kairo plus the six misfits, and r1–r5 rebuilt as writer cases. That is about 12 cases, split into a dev set and a held-out set. With this few, corrected contrasts will read "not separated" unless the effect is large. Inferred: we accept that, and the decision says so rather than pooling repeats as cases.
- Scoring judge model: gemini flash via OpenRouter (cheap, and not the writer's family). The owner can swap it; the point is that it is fixed per campaign.
- Withholding: each writer case records its own exemplar note to withhold, so blind withholding is automatic per case instead of chosen by hand on the CLI.
- Regression: "the shipped skill against the last promoted one" is the same A/B mechanism. Wiring `RegressionWatch` to a host's periodic run is the host's job.

**Architecture.**
- The writer becomes a candidate kind, `report_writer`, beside `analysis_reporter` (same shape: frozen cases, a judge on a rubric, a spend seat).
- Its overlays are its levers: the skill texts (by content digest, carried as subject components so identity refuses an uncarried text), reasoning effort, and the writer limits. The model is the arm's candidate model.
- The kind reports objective measures from the run itself: ending, bounces by kind, open must_fix and should_fix counts, editor rounds, turns and cost. The scoring judge adds the seven rubric items.
- The promotion read wraps the existing family verdicts and `gate_verdicts`. It adds no new statistics.
- Must stay possible (wave 2): a search proposes lever maps for ANY kind (host kinds included, e.g. a quantization lever a host declares), and scores them through this same campaign path. Nothing in wave 1 may assume the lever under test is text.

**Constraints:** no new PR (all evals work merges into report-kit/analyst, #1006). Report text the package writes follows the report-style rulings. Live spend only in chunk 06, under a cap the owner names. Subagents run one test file at a time; package suites go on the testament pool.

**Tradeoffs accepted:**
- Small case set, so low power. Revisit when hosts contribute writer cases (Discodon snapshots).
- The scoring judge is uncalibrated until humans label pages (an app's job, per the 2026-10-14 ruling). Its scores are disclosed as uncalibrated, never hidden.

**Decisions made mid-build:**
- 01: both roles' texts are built before the freeze, so a withheld note the skill lacks is refused with nothing frozen (this was also true of the shipped skill before).
- 02 step 0: the recorded Opus runs took 682–985 s of turn time over 36–44 turns, past the engine's 600 s default cell deadline. The kind sets its own deadline (`writer_cell_timeout_s`, from the writer's worst case plus the judge phase).
- 02: the kind lives in `ops/report_writer/`, since `analysis` may not import `ops`.
- 02: an arm with no skill named records the shipped skill's digest, not None, so two builds with different shipped files never pool. This departs from the brief, which said None.
- 02: effort is in, through a host model factory (`WriterModels`) called with the effort word. Levers: `skill`, `reasoning_effort`, `revisions`.
- 02: a `WriterCase` embeds the whole snapshot and restores it into the run's scope. Snapshot ids come from content, so the restored snapshot is the frozen one. Caveat: analyst tools that read other stored documents still read the host's store.
- 02: `invoke` drives `write_rounds` directly; a cell stores no report document and no gap.
- 04: `writer_case_mint` takes a campaign, an analysis or a frozen snapshot. A case is keyed on its source: an exact re-mint returns the stored case, and a different one is refused, because stored cases never change.
- 04: the split goes by group. A host names a group for related campaigns, or the source key is the group. A group keeps its recorded side; a new group's side comes from the sha256 of `writer-split:<group>` against `held_out_fraction` (default 0.3). Seed: 5 held-out (Kairo, the four r2 cases), 7 dev. Inferred defaults (the fraction and the prefix), not owner words.
- 04: `WriterCase.case_version = 1` with `origin` (host, minted_by, source, campaign or analysis, group, side, set, fraction, the skill digest the notes were checked against).
- 02b: the engine knows models only by id, so the judge-family rule takes a host hook (`ModelFamily`: model id → family or None). It refuses the same family, and records "unchecked" on the run when either model can't be placed. A writer launch must name its judge model. The failure-owner routing is one helper (`kernel/failed_call.py`) for the writer, reporter and judge kinds. Each of the five bounce kinds is a measure, read through `Tally.bounces()`.
- 05: `writer_promotion(host, campaign_id, scope_id, *, candidate)` reads `campaign_verdicts` through `gate_verdicts`: a failing gate → keep, any undecided → undecided, everything decided with an improvement → promote. It refuses a moved judge, a run with no case set, and any non-held-out case, and it never writes (a person records the call). The ledger is `analysis/analyst/skill/PROMOTIONS.toml`, read by `skill_ledger.py`; the shipped skill is seeded unmeasured. A measured entry also records `scope`, since a campaign id can't be looked up without it.
- 03: the scoring rubric's one source is the shipped editor.md, read once (`ops/report_writer/rubric.py`, `shipped_rubric()`); editor.py's item names come from the same read. `writer_scoring_rubric()` makes seven 1–5 dims `report_writer.<item>` on the built-in judge prompt (editor 2/1/0 → 5/3/1). A host seeds the template from it at creation, and the stored template then holds the text, frozen per run in `template_digest`. Consequence: promoting a new editor.md moves the scoring rubric for templates created after it; campaigns already created keep theirs.
- 03: the engine reports a judge change between arms but does not refuse one, so `refuse_unfixed_scoring_judge(runs)` refuses a campaign whose arms differ in template, rubric revision, judge model, configs or temperature.

**Level:** Medium. The kind's shape is clear from `analysis_reporter`. Unknown: whether a full writer run (a multi-session tool loop) fits a single cell's invoke under the existing cell timeout and cost seats. Chunk 02's first step checks that against one replayed prototype5 case.

## Chunks

### Chunk 01: Skill seam: a skill version is a value
Type: code
- `AnalystSkill` can be built from files and published (today only `load_skill` and `SkillText` are exported).
- `report_write` and `write_report` take an optional skill; `None` keeps the shipped one.
- A skill version's identity is its content digest. The generation record already stores it; a run under a supplied skill records it the same way.
- Done when: a write with a supplied skill records that skill's digests; the default write is unchanged (replay tests green); an incomplete skill is refused before anything is frozen or spent.

### Chunk 02: The `report_writer` candidate kind
Type: code
- `ops/report_writer/writer_kind.py` and `ops/report_writer/writer_contract.py`, mirroring `reporter_kind.py` and `reporter_contract.py` (in `ops/`, because `analysis` may not import `ops`).
- Overlays: `skill` (the whole skill's content digest; an arm naming none records the shipped digest), `reasoning_effort` and `revisions`. The skill digest travels as a subject component.
- Case: a frozen writer case, meaning the report snapshot plus the exemplar note it withholds.
- `invoke` runs `write_rounds` and returns the page as the output, with the objective measures.
- Step 0: check one prototype5 case's recorded turn time through the kind and check that it fits the cell timeout and the spend seat.
- Done when: a replayed campaign of two arms (shipped skill against a one-line-edited skill) over two scripted cases produces runs, both arms keyed apart by the skill lever, and the objective measures land on results.

### Chunk 02b: Writer launcher and the review fixes
Type: code
- A launcher for the writer kind, the counterpart of `ops/reporter.py`. It wires `KindWiring.cell_timeout_s` from `writer_cell_timeout_s`, so a real-speed cell is never cut off at the 600 s default.
- Every bounce kind the analyst records (unparsed, shape, validity, invariant, max_tokens) is a measure, read through a typed accessor on `Tally`, not bare strings.
- The revisions cap has one public source.
- One shared failure-owner mapping replaces the three copies (reporter, judge and writer kinds).
- The launcher calls chunk 03's `refuse_undeliverable_writer_template` at every launch. The judge-family rule (owner: the judge comes from a different model family than the writer) is refused at launch wherever the engine can tell a model's family; otherwise the gap is recorded.
- The launcher admits a launch over a named case set (chunk 04's sets), as the reporter's does over its bank.
- A writer-case retire op, so a moved campaign can be re-minted (chunk 04's refusal names it).
- New tests: the failure routing, prepare's refusals, and the launcher's deadline.
- Done when: rev-20261031T213525Z-a63c4302's warnings R-2, R-3, R-4, R-6, R-7, R-8, R-9 and R-10 are fixed, and a scripted launch through the launcher runs under the writer's own deadline.

### Chunk 05b: Writer actions for hosts
Type: code
- Writer case mint, retire and restore, writer template create, and the promotion read become engine actions in the catalogue, so a host (Discodon, metallm) calls them as it calls the reporter's. Each refusal's remedy names a real action (`test_refusal_remedies`).
- Done when: test_refusal_remedies passes, the awaiting-a-caller lists for these names are empty, and an action-level test mints a case, creates a template and reads a decision.

### Chunk 03: Fixed scoring judge (delegated)
Type: code
- The editor rubric's seven items become rubric dims of a writer template's judge. That judge reads the final page and the snapshot facts the editor reads.
- Its rubric text and model are fixed per campaign. A launch whose arms differ in the scoring rubric or judge model is refused.
- Its scores disclose that the judge is uncalibrated, through the existing judge-profile path.
- Done when: a replayed campaign carries all seven judged dims per result, and the refusal is tested.

### Chunk 04: Writer case bank, dev and held-out sets (delegated)
Type: code
- The mechanism is the product: any host (Discodon, then metallm) mints a writer case from one of its own campaigns, and the case joins a named set. The fixtures are only the first seed.
- A builder freezes writer cases from Kairo, the six misfits and r1–r5, each recording its withheld note.
- The cases are minted into two named `CaseSet`s, `writer-dev` and `writer-held-out`. The split is fixed and recorded, and a case never sits in both.
- Done when: both sets resolve; a case in both is refused; withholding is per case with no CLI choice.

### Chunk 05: Promotion decision and the shipped-digest check
Type: code
- `skill_promotion(campaign)` reads the held-out campaign's corrected family verdicts through `gate_verdicts`. It answers `promote`, `keep` or `undecided`, with the candidate's variant key, the per-measure verdicts, the cost and the case count.
- A promotion is recorded as a `CampaignCall`, and in a new package ledger, `analysis/analyst/skill/PROMOTIONS.toml` (skill digest, campaign id, decision, date). The current skill is the baseline entry.
- Every shipped analyst and editor digest has a ledger entry, and the entry states its basis: `measured` (a campaign decision) or `unmeasured` (an edit shipped without one, with the reason). An enforcement test fails on a shipped digest with no entry. It does not block edits while the case set is too small to measure: an unmeasured entry is the honest record. Inferred: the owner may want a later switch that requires `measured` once host cases exist.
- Done when: the ledger test trips on an edited skill file and passes on a recorded one; `undecided` never promotes.

### Chunk 06: Live mechanism smoke on Haiku (main session, owner-capped)
Type: code
- This proves the pipeline end to end; it is not a decision. Run one small campaign on claude-haiku-5-5: the shipped skill against a deliberately inert variant, over two or three dev cases, scored by the fixed judge.
- First quote it with `ArmQuote`, and get the owner's cap before sending a turn.
- Nothing is promoted and nothing is changed on its result. The ledger is untouched.
- Done when: every arm has results with the objective and judged measures; the promotion read answers (expected: `undecided`); the cumulative Critic is clean.
Critic mode: cumulative

## Verification strategy
Chunks 01–05 are proven on recorded and replayed turns (prototype5 and the hand-scripted fixtures). Chunk 06 is the live proof. Gates: lint and typecheck locally; the evals suite on the testament pool; the integration suite before merging into report-kit/analyst.

## Wave 2 (sketch, not this plan)
- A search over a kind's lever space. A genome is a lever map: model, effort, a host's quantization lever, skill texts.
- Operators: enum and numeric lever moves; text mutation by an LLM proposer seeded from the gap digest and editor-score gaps; crossover between strong arms.
- Each generation is a campaign on `writer-dev` with cheap screening (fewer repeats, a cheap judge). Survivors go to the wave 1 gate on `writer-held-out`. A human promotes.
- A budget per generation, and generic over kinds, so Discodon's own kinds can be searched too.
