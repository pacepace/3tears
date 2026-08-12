# Convergence Sequencing: Search + Evals

**Status:** Working sequence — 2026-08-04
**Scope:** the cross-repo execution order for the search capability
([`search-spec.md`](search-spec.md)), its enablers, and the eval extraction
([`family-convergence.md` §4.2](family-convergence.md#42-evals--3tears-eval-contractsrungenanalysis-new-from-discodon)).
Detailed in-repo sequencing for the search build stays in
[`search-spec.md` §7](search-spec.md#7-sequencing); this document is the layer
above it — who can start what, and when, across five repos.
**Not sequenced here:** the rest of the convergence program
([chat-kit build §4.11](family-convergence.md#411-chat-ui--a-headless-typescript-kit-protocol-from-3tears-seeds-from-scriob-and-metallm),
[tiled images §4.12](family-convergence.md#412-tiledzoomable-images--a-slim-acquisition-package-new-from-samsungs-design),
[scrape consolidation §4.13](family-convergence.md#413-scraping--consolidate-on-3tears-scrape-exists-faidh-lineage),
the observe/models/iam adoptions) — except where an item below explicitly
gates one of them.

## How to read this

- Items **within a phase run in parallel** — no ordering between them.
- A phase **closes at its checkpoint**: every gating item done. Nothing in a
  later phase starts first. This deliberately trades a little idle time for a
  coordination model that stays simple across many repos and many sessions.
- Items marked **(non-gating)** ride their product's own schedule; the phase
  closes without them.
- Repo prefix says who does the work; the arrow says where the detail lives.

## Phase 1 — Foundations

All items independent; all can start today.

- **3tears:** build the search leaf — contracts, SearXNG + Tavily adapters,
  Call, Bind, standalone transport, in-process limiter, the
  `test_no_bespoke_reuse` widening.
  → [`search-spec.md` §7 Phase 1](search-spec.md#7-sequencing)
- **3tears:** add the three `media-contracts` facet fields (rights status,
  pixel dimensions, direct-file vs containing-page).
  → [`search-spec.md` §4](search-spec.md#4-changes-elsewhere-in-3tears)
- **discodon:** adopt Python 3.14 (interpreter switch + verify; its floor is
  already `>=3.12`).
  → [`family-convergence.md` open question 1](family-convergence.md#6-open-questions)
- **discodon:** land the in-flight character-eval / cassette-delivery work.
  → its own branch; seam rules in
  [`search-spec.md` D28](search-spec.md#1-decisions-taken)
- **discodon:** carve the eval **StorageProtocol**, migrate `EvalStorage` and
  the analysis bundle onto it (table migrations may trail); shape it per the
  store-shape rule.
  → [`family-convergence.md` §4.2 (verified corrections)](family-convergence.md#42-evals--3tears-eval-contractsrungenanalysis-new-from-discodon),
  [open question 22](family-convergence.md#6-open-questions)
- **discodon:** reshape budget refusal to the
  `check(estimate)`/`record(spend)` port shape, copied from
  `threetears.search.contracts`, not invented.
  → [`search-spec.md` §7 Phase 5](search-spec.md#7-sequencing);
  [`family-convergence.md` §4.2](family-convergence.md#42-evals--3tears-eval-contractsrungenanalysis-new-from-discodon)
- **metallm:** close the family version lag.
  → [`family-convergence.md` §5 (metallm)](family-convergence.md#5-implications-per-family-member)

**Checkpoint:** search-leaf **Gate A** passed (contract review); discodon on
3.14 with port-shaped storage and budgets; metallm current.

*Status 2026-08-11:* the two **3tears** items are done — the leaf shipped
(both adapters, budgets, pacing, wiring) and Gate A passed with its findings
landed the same night; the `media-contracts` facets rode the keystone
commits. Both landed in
[#303](https://github.com/pacepace/3tears/pull/303), Gate A findings
included. The discodon and metallm items are outstanding and owned outside
this repo: discodon still declares `>=3.12`, and metallm's lock resolves the
family at 0.10.6. Because the phase gate is a coordination convention rather
than a build dependency, 3tears carried on into Phase 2 work that does not
consume either — see the Phase 2 note.

## Phase 2 — In-family integration (3tears)

- **3tears:** search Phase 2 — Extract's web path; gut
  `WebSearchTool`/`WebFetchTool`; serve wiring; context-save fix;
  `ToolExecutor` artifact fix; MCP `structuredContent`; the two envelope
  asks. → [`search-spec.md` §7 Phase 2](search-spec.md#7-sequencing),
  [§4](search-spec.md#4-changes-elsewhere-in-3tears)
- **3tears:** search Phase 3 — replay, aggregate, select; the replay record
  schema is elicited against discodon's pipeline-eval needs and carved
  storage port (why Phase 1 precedes this).
  → [`search-spec.md` §7 Phase 3](search-spec.md#7-sequencing),
  [§3.10](search-spec.md#310-replaypy-phase-3),
  [D26–D28](search-spec.md#1-decisions-taken)
- **3tears (design only):** decide the stream-protocol channel for
  structured tool results — chat-kit workstream input; it gates metallm's
  *frontend* convergence later, not this sequence's next phase, but deciding
  it while attention is on structure is the point.
  → [`family-convergence.md` §5 (3tears obligations)](family-convergence.md#5-implications-per-family-member)

*Status 2026-08-11 — most of this phase has landed.* Extract's web path is
built ([#316](https://github.com/pacepace/3tears/pull/316)), on ground laid
by [#307](https://github.com/pacepace/3tears/pull/307) (the `FetchTransport`
implementation and the connection lifecycle it needs),
[#310](https://github.com/pacepace/3tears/pull/310) (five rulings recorded
before the build) and [#315](https://github.com/pacepace/3tears/pull/315)
(the `extraction_status` constants it records into). Alongside it: the
`ToolExecutor` artifact fix ([#318](https://github.com/pacepace/3tears/pull/318),
which check 4 needed) and the MCP `structuredContent` face
([#319](https://github.com/pacepace/3tears/pull/319)).

**Item 7 was pulled forward out of last place**
([#317](https://github.com/pacepace/3tears/pull/317)), because the two
envelope asks turn out to be additive in different directions. §10.9
populates a field that already existed, so it has no rollout constraint at
all; §10.10 adds one to a model with `extra="forbid"`, where a client
sending it to an older server is *rejected*, not degraded. Only the
accepting half shipped. It is the one item in this phase needing two
release cycles, which is why it should not have been last.

**What remains here:** item 5 (gut both builtins, serve wiring, the NATS
metadata end-to-end test) and the rest of item 6 (`page_finder` structure,
the context-save node). Item 5 carries the phase's only user-visible
behavior change — an Extract-backed `WebFetchTool` makes robots binding for
callers it was never binding for — so it owes a stated rollout of its own.
See [`search-spec.md` §7 Phase 2](search-spec.md#7-sequencing) for the
per-item table.

*Note 2026-08-11 — what Phase 1's outstanding items actually hold up.* Only
one thing here consumes them: the **replay record schema** (search Phase 3
item 8) is elicited against discodon's carved storage port and its
pipeline-eval needs, which is the whole reason "Why this order" puts Phase 1
first. Cutting that schema before the port exists would be cutting it
against an imagined consumer. Everything else in this phase — the Extract
web path, gutting the builtins, the serve wiring, the envelope asks, and
Phase 3's `aggregate`/`select` — depends on nothing outside this repo and
proceeds.

**Checkpoint:** **Gate B** — success checks verified in-repo, SearXNG score
semantics confirmed against a live instance, decisions/vetoes propagated
back into
[`search-requirements.md` §13](search-requirements.md#13-decisions-needing-an-owner).
→ [`search-spec.md` §7](search-spec.md#7-sequencing)

## Phase 3 — Release

- **3tears:** lockstep family release — bump → PR into develop → PR
  develop→main → **push the annotated tag from main**; verify the GitHub
  Release exists.
  → [`search-spec.md` §7 Phase 4](search-spec.md#7-sequencing);
  [`CLAUDE.md` release rules](../CLAUDE.md)

**Checkpoint:** `3tears-search` and the bumped family on PyPI; tag verified
via `git ls-remote --tags origin`.

**Ran early — v0.24.0, 2026-08-11, with Phase 2 not started.** Bump in
[#304](https://github.com/pacepace/3tears/pull/304), develop→main in
[#305](https://github.com/pacepace/3tears/pull/305), tag `v0.24.0` pushed
from main; the ruling it forced is
[#306](https://github.com/pacepace/3tears/pull/306) (D29). The
checkpoint is met on its own terms: tag on origin, Release present, all 30
packages at 0.24.0 including `3tears-search`. But it published the leaf
*alone*, so it does not open Phase 4 — the consumer migrations below need
the Phase 2 surface (metallm the gutted `WebSearchTool`, discodon the Phase
3 replay piece), and a consumer release carrying migration work must pin a
version that has it. Phase 2 still closes at Gate B, which now guards the
**next** release rather than the first. The contracts stay re-cuttable
meanwhile: nobody has bound them
([`search-spec.md` D29](search-spec.md#1-decisions-taken)).

## Phase 4 — Consumer search migrations (parallel per repo)

- **metallm:** `feature/new-search` — pin the released family, delete both
  side-steps (check 1).
  → [`search-spec.md` §7 Phase 5](search-spec.md#7-sequencing)
- **discodon:** `feature/new-search` — collapse its two implementations, wire
  the eval cost cap onto `BudgetPort` (check 3, D27), adopt pipeline-eval
  replay over its own store, re-capture web_search cassettes.
  → [`search-spec.md` §7 Phase 5](search-spec.md#7-sequencing)
- **samsung (non-gating):** build phase-2 image search on the leaf when that
  work schedules (checks 2, 5, 9).
  → [`search-spec.md` §7 Phase 5](search-spec.md#7-sequencing)
- **all migrating repos:** record acceptance of what binds them.
  → [`search-requirements.md` SR-M3](search-requirements.md#m-lifecycle)

**Checkpoint:** metallm and discodon merged and green; acceptance recorded.
(**Gate C** — the wire-compatibility promise and released envelope asks —
fires whenever the first *pod-resident* search deploys, which follows
discodon's NATS convergence and sits outside this sequence.)

## Phase 5 — Eval extraction

- **3tears + discodon:** extract `3tears-eval-contracts` first (identity,
  measures, spend/replay composition per
  [D27/D28](search-spec.md#1-decisions-taken)), then `run` and `analysis`;
  the generator ships package prompts plus the narrow completion protocol
  satisfied by `3tears-models`, and may extract last.
  → [`family-convergence.md` §4.2 (incl. verified corrections)](family-convergence.md#42-evals--3tears-eval-contractsrungenanalysis-new-from-discodon),
  [open questions 2, 3, 22](family-convergence.md#6-open-questions)
- **discodon:** consume its own extraction — first consumer, green before
  anyone else binds.
  → [`family-convergence.md` §5 (discodon)](family-convergence.md#5-implications-per-family-member)

**Checkpoint:** discodon running on the extracted packages; the eval
contracts ratified per
[SR-M3's per-repo-acceptance pattern](search-requirements.md#m-lifecycle).

## Phase 6 — Eval adoption across the family

All parallel:

- **metallm:** adopt `eval-*` — today it has zero eval infrastructure, the
  family's biggest gap.
  → [`family-convergence.md` §5](family-convergence.md#5-implications-per-family-member)
- **scriob:** adopt `eval-{contracts,analysis}` for its continuity corpus.
  → [`family-convergence.md` §5](family-convergence.md#5-implications-per-family-member)
- **samsung:** adopt `eval-run` for its MCP-driver eval.
  → [`family-convergence.md` §5](family-convergence.md#5-implications-per-family-member)
- **hallucinote:** programmatic runner over its scenario briefs and canonical
  verdicts.
  → [`family-convergence.md` §5](family-convergence.md#5-implications-per-family-member)

**Checkpoint:** the eval workstream of
[`family-convergence.md` §4.2](family-convergence.md#42-evals--3tears-eval-contractsrungenanalysis-new-from-discodon)
is delivered; the remaining §4 workstreams sequence on their own.

## Why this order

Phase 1 front-loads everything that is independent *and* everything later
phases are elicited against: the storage port and budget shapes must exist
before search's replay record schema and `BudgetPort` wiring are cut against
them, and 3.14 must land before discodon can consume anything the family
ships. Phases 2–3 are single-repo and mechanical. Phase 4 deliberately
precedes the extraction so the eval seams (budget, replay identity) are
extracted in their *post-migration* shape — extract-don't-invent applied to
sequencing: lift proven code, never code that is about to change. Phase 6
rides packages that are released and discodon-proven.
