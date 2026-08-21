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
- **discodon:** ~~reshape budget refusal to the
  `check(estimate)`/`record(spend)` port shape, copied from
  `threetears.search.contracts`, not invented.~~ **Withdrawn 2026-08-19 —
  this asked for the wrong shape.** See the correction below.
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

***Correction 2026-08-19 — both outstanding items above have moved, and the
note asserting otherwise was five days stale.*** discodon adopted 3.14 on
2026-08-14 (`build: take the interpreter from uv, and move it to CPython
3.14`); it now declares `requires-python = ">=3.14"`, not `>=3.12`. metallm
closed its lag the same day, `v0.17.7 -> v0.24.4`, so it is current and no
longer resolves the family at 0.10.6. **Neither landed because of this
sequence** — each repo did it on its own clock, and nothing here was watching,
which is the same failure mode Phase 2's third bullet recorded from the other
direction: a status note goes stale in whichever direction nobody is checking.

**What actually remains of discodon's two items is narrower than "outstanding",
and one of them is subtler than not-done.**

- **The storage carve is half-present, and the present half is not the half
  this sequence needs.** `discodon/eval/storage.py` does carve a `Protocol` —
  `EvalRunDocumentStore`, two methods, `load_eval_run_with_etag` and
  `save_eval_run` — but it was carved for its own local reason (stating the
  conditional-write retry policy over the protocol rather than over Postgres),
  and it covers only that pair. `EvalStorage` does not declare it, and the
  analysis bundle is not on it. So the seam Phase 3's replay record schema is
  meant to be elicited against still does not exist. **A protocol in the file
  is not the protocol the item names**, and grepping for the item's own word —
  `StorageProtocol` — finds nothing, which is how this reads as untouched from
  outside.
- **The budget reshape was withdrawn on 2026-08-19, and the paragraph that
  called discodon's shape a defect was wrong.** It read: `EvalRunCostCap`
  exposes `check()` and `record(cost_usd)`, matching
  [`threetears.search.contracts.BudgetPort`](../packages/search/src/threetears/search/contracts/budget.py)
  by name, but `check()` takes no estimate and reports `accumulated > ceiling`,
  so it can only stop a run *after* the ceiling is crossed — filed as the shape
  most likely to be mistaken for done at wiring time.

  **The two are solving different problems, and both shapes are right.** The
  search port wraps exactly one provider call — `check(estimate)` before it,
  `record(spend)` after it, both below the retry boundary — so it can refuse a
  spend that has not happened yet. discodon's cap fires *between eval cells*,
  where the cell's spend is already booked and the cell contains an unknown
  number of LLM calls; there is no honest pre-estimate to give, and the port's
  own "an estimate a run may freely exceed is not an estimate" is the argument
  against inventing one. Requiring one shape for both was the error.

  The eval budget contract will be shaped from what discodon runs in
  production. Recorded as R6 of
  [`eval-extraction-discodon-requirements.md`](eval-extraction-discodon-requirements.md),
  stated there as a requirement precisely because an agent reading the
  withdrawn instruction would "fix" a working production control.

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
  → [`family-convergence.md` §5 (3tears obligations)](family-convergence.md#5-implications-per-family-member);
  **designed 2026-08-14**, two days after this phase was declared complete
  without it — [`stream-protocol-structured-results.md`](stream-protocol-structured-results.md);
  **approved and built 2026-08-18** (§8 there). The item said *design only*;
  the build came with it because the decision the design could not take alone
  turned out to be one of five, and the other four were answerable from the two
  consumers' own code. **D-S2 was reversed by that code**: the new frame type it
  recommended has no caller — metallm mints its own ws message types and scriob
  mints frame types from event discriminators — so the structure rides the two
  *events* instead, and scriob's existing `tool_call_end` frame carries it with
  no rollout order to respect. One branch stays unbuilt on a consumer decision:
  what store an over-the-bound handle resolves against.

*Status 2026-08-12 — this phase is complete; the entries below run in build
order, and the closing one is item 6.* Extract's web path is
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

*Correction 2026-08-14 — the conclusion held, the premise did not.* Shipping
only the accepting half does **not** keep the field off the wire: every
declared field is serialized, so an optional nobody set still crosses as an
explicit `null`, and `extra="forbid"` refuses an unknown null exactly as it
refuses an unknown value. A 0.23.11 tool pod refused every call from a 0.24.1
registry for three days on `deadline_seconds`, a field no caller populated,
while registering and heartbeating normally throughout. Fixed in 0.24.3 by
pruning unset top-level optionals from the registry→pod envelope, and by making
the pod say why it was refusing. This is the same pattern this program keeps
recording — item 5's untested host configuration, the sweep's untested
independence — now between two *versions*: both sides correct alone, a prose
rollout note asserting a wire fact, and no test parsing real forwarded bytes
with a model predating the newest field. Detail at
[`search-spec.md` §7 Phase 2 item 7](search-spec.md#7-sequencing); §13 rows
`§10.10` and `SR-M1` updated.

**Item 5 landed 2026-08-11**, and took a correction pass before merging
([#321](https://github.com/pacepace/3tears/pull/321)). Both builtins run on
the leaf, `serve.py` wires them, and check 8 is pinned end-to-end — a real
pod dispatch, read back from the published bytes, on the failure path as well
as the success one. Two things came out of the build that reach past this
phase. Gate A's expectation that one host adapter would satisfy **both**
transport protocols was wrong: `TracedHttpClient` is per-upstream and buffers
bodies, so the search half is a thin adapter over it and the fetch half is
`StandaloneTransport` — the split is ruled and explained in the spec. And the
phase's user-visible change is **two** changes, not one: robots became
binding (foreseen), and extraction now refuses rather than falling back to
stripping tags with a regex, so a consumer driving `web_fetch` must declare
`3tears-agent-tools[fetch]` or get nothing back.

*What the correction pass adds, because it generalises past this item.* Review
found seven defects, the worst of which would have shipped as a widespread
`web_fetch` failure: the host's fetch transport inherited the leaf's
zero-redirect default, so every URL that canonicalises via a 301 came back
empty. All seven shared one cause — the tool and the transport were each
tested alone and the seam between them was not tested at all, so the
configuration production actually runs had no test and the suite proved values
were *passed* rather than that behaviour *held*. The generalisable lesson for
the phases still to come: **a host's choice of configuration needs its own
test, driven end-to-end, separate from the tests for the thing being
configured**. Phase 3's `aggregate`/`select` and the Phase 5 consumer
migrations each stand up new wiring of exactly this kind. Detail and rulings
in [`search-spec.md` §7 Phase 2](search-spec.md#7-sequencing).

**Item 6 closed 2026-08-12, and with it the phase.** `page_finder` reads the
typed projection off `ToolMessage.artifact`
([#326](https://github.com/pacepace/3tears/pull/326)) rather than re-parsing the
prose the LLM read; every new fact arrives as a defaulted field, so check 4's
"without its callers changing" clause is the literal shape of the change. The
context-save node ([#327](https://github.com/pacepace/3tears/pull/327)) had been
inert in production — its default tool set held bare names while the adapter
binds every tool under `mcp_name()` — and now binds on result *type* before tool
name, which is what C8 asked for, retaining structure beside the prose so
SR-A3's re-checkability survives the truncation.

Both builds turned up an adjacent defect one layer over, and the pattern is the
same one item 5 recorded: a unit test that asserts a value was *passed* rather
than that behaviour *held*. `_verify_candidate_page` fetched unbounded (19 MiB
of HTML peaking at ~1.5 GiB of heap), and `chunker.py` registered its only
default strategy under the same bare name the node was being fixed for.

**Phase 2 is complete.** One item spun *out* rather than
in: asking why nothing in the stack sends a conditional request produced
**SR-M4 / D30**, ruled 2026-08-12, whose build sequence is
[`search-task-01-conditional-revalidation.md`](search-task-01-conditional-revalidation.md).
It blocks nothing and is blocked by nothing, and its step 1 is a
`media-contracts` change, so it moves the family bound when it lands.

***Correction 2026-08-14 — it was not complete, and the way it was wrong is
worth more than the item.*** This phase lists **three** items. The status notes
above close out the search build (items 4-9) and the note declaring the phase
complete was written against those. The third bullet — *decide the
stream-protocol channel for structured tool results* — **was never taken**, and
nothing anywhere recorded a decision. It went unnoticed for two days because it
is the only item here that gates nothing in the search sequence: everything that
consumes it is in the chat-kit workstream and metallm's *frontend*, neither of
which was waiting on this phase's checkpoint.

That is the general shape to watch for. **A phase closes on its gating items and
then reads as closed for its non-gating ones too** — the very items most likely
to be forgotten, because nothing downstream complains. The design is now written
([`stream-protocol-structured-results.md`](stream-protocol-structured-results.md)),
and it is design-only as the bullet always said, so nothing that shipped is
wrong; the record was.

*Status 2026-08-19 — the design-only bullet was built, and it grew a workstream.*
[#355](https://github.com/pacepace/3tears/pull/355) shipped the channel on both
streaming faces on 2026-08-18, four days after the design was written. The
bullet said *design only* and the build came with it, for the reason recorded
above: the decision the design could not take alone turned out to be one of
five, and the other four were answerable from the two consumers' own code.
**With that, every item in this phase is both decided and built** — the
2026-08-14 correction stands as the record of how it was nearly missed.

What followed is **not** part of this sequence and should not be counted
against it. Building the channel surfaced a question the design had deferred —
what a client may ask for, and what happens when the answer is too big — ruled
in [`structured-result-tiers.md`](structured-result-tiers.md)
([#367](https://github.com/pacepace/3tears/pull/367), 2026-08-18) and carved
into three tasks. Task-03 is in flight
([#368](https://github.com/pacepace/3tears/pull/368)); tasks 01 and 02 are
unbuilt. It belongs to the chat-kit workstream
([§4.11](family-convergence.md#411-chat-ui--a-headless-typescript-kit-protocol-from-3tears-seeds-from-scriob-and-metallm)),
which this document explicitly does not sequence.

See [`search-spec.md` §7 Phase 2](search-spec.md#7-sequencing) for the per-item
table and the item 5 rulings.

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

*Status 2026-08-12:* the SearXNG half is discharged — the formula and its four
consequences are recorded at SR-A4
([#322](https://github.com/pacepace/3tears/pull/322)), and
[#328](https://github.com/pacepace/3tears/pull/328) gave it a container to check
itself against **and closed the residue**: multi-engine fusion was observed
live (a fused score of 4.64 across two engines, matching the formula to
floating point), so the unbounded claim no longer rests on the formula alone.

*Status 2026-08-13 — the other two parts ran, and Gate B is one decision from
closing.* Search Phase 3 item 9 (`aggregate`/`select`) landed
([#333](https://github.com/pacepace/3tears/pull/333),
[#335](https://github.com/pacepace/3tears/pull/335)), so the sweep the gate asks
for became possible and was done. **Eight of the nine in-repo success checks
pass**; the other five checks are consumer-repo and excluded by the gate's own
wording. §13 is propagated — every row now carries a status, and no vetoes were
taken during Phases 1–3.

Two things came out of it that reach past the gate. **Check 12 did not pass
until the sweep wrote the test it was missing**: egress was pinned in several
places but *independence* nowhere, so the requirement's own hard case — SearXNG
and Tavily on different exits in the same process — had never been driven. That
is the fourth appearance of the pattern this program keeps recording, and the
first draft of the new test reproduced it exactly, comparing each side against
the constant it was configured from until the pins were rewritten to compare the
two sides to each other.

**Check 14 does not pass, and it is a decision rather than a build.** The two
builtins leave `face_api` and `face_mcp` at their `False` defaults, so two of the
three faces are unreachable and "no second result shape per face" has nothing to
hold against. The mechanism is built; the reach is off. Because it is ACL-visible
surface, and because §13 already lists the adjacent `skill_eligible` / `web`
alias question as needing an owner, the sweep recorded it rather than flipping
the flags. **Gate B closes when that decision is taken** — flags on and a pin
written, or the check amended with the reason. Detail at
[`search-spec.md` §7 Gate B sweep](search-spec.md#7-sequencing).

**GATE B IS CLOSED — 2026-08-14, and the decision above turned out to be a
false choice.** Both options assumed check 14 needs the reach turned on. It does
not. The face flags govern reach, and **nothing in 3tears reads them** — the
surfaces that act on them (the API namespace stamp, the MCP export, the face-flip
re-stamp) are all hub-side. Check 14 is about *rendering*, and two of the three
renderings happen here: the platform tool face and the MCP face. The third, the
HTTP API face, renders in the hub and is verified there, alongside the five
consumer-repo checks the gate's own wording already excluded.

So the check was **split rather than flipped or amended**, and its in-repo half
now passes: one candidate set driven through both renderings and compared *to
each other*, plus an enforcement guard holding the projection to a single
construction site — because a comparison between known faces cannot see a face
nobody has added yet. Building it surfaced two pre-existing second-shape sites,
one closed (`web_fetch` reimplemented the projection instead of calling it) and
one exempted with a rationale (the context-save node narrows on purpose).
**Nine of nine in-repo checks pass.** Detail and the two findings at
[`search-spec.md` §7, "Check 14, resolved by splitting it"](search-spec.md#7-sequencing).

§5.5's `skill_eligible` / `web`-alias question stays open in §13, on its own
merits as a reach decision. It never gated this.

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

**Correction 2026-08-14: "the next release" was the wrong object, and four
releases have since walked past it.** v0.24.1 through v0.24.4 all shipped with
Gate B open, and they carry the surface it was said to guard — **0.24.1**
Extract's web path, the gutted builtins and both envelope asks; **0.24.2** the
rest of Phase 2 (`page_finder`, the context-save node), making it the first tag
carrying Phase 2 whole; **0.24.3** `aggregate`/`select` plus the Gate B sweep
itself; **0.24.4** nothing search-side. None of that was a bypass. The family
releases on its own cadence for unrelated reasons — 0.24.3 went out to fix the
envelope outage above — so a gate phrased as "the next release" was always
going to be overtaken by the next unrelated fix.

What Gate B actually protects is what D29 already named: **the first consumer
release that pins a version carrying search.** Publication does not bind the
contracts, and a tag does not bind them; a consumer's pin does. The gate stands
in front of Phase 4, not in front of the release button.

**One consequence for Phase 4, which the phase's single "the released version"
hides: its two consumers are no longer symmetric.** metallm needs the gutted
`WebSearchTool`, which is *released* — 0.24.2 and later — so its migration has
a pin it could name today and waits only on Gate B and its own version lag.
discodon needs the replay piece (search Phase 3 item 8), which is **not built
at all**, and which is elicited against discodon's own carved storage port, an
outstanding Phase 1 item. No 3tears release will unblock discodon; a build
will, and that build waits on discodon.

## Phase 4 — Consumer search migrations (parallel per repo)

- **metallm:** `feature/new-search` — pin the released family, delete both
  side-steps (check 1).
  → [`search-spec.md` §7 Phase 5](search-spec.md#7-sequencing)
- **discodon:** `feature/new-search` — collapse its two implementations,
  ~~wire the eval cost cap onto `BudgetPort`~~ (**withdrawn 2026-08-19** — the
  eval budget contract is shaped from what discodon runs; check 3's spend half
  is R5 of
  [`eval-extraction-discodon-requirements.md`](eval-extraction-discodon-requirements.md)),
  adopt pipeline-eval replay over its own store, re-capture web_search
  cassettes.
  → [`search-spec.md` §7 Phase 5](search-spec.md#7-sequencing)
- **samsung (non-gating):** build phase-2 image search on the leaf when that
  work schedules (checks 2, 5, 9).
  → [`search-spec.md` §7 Phase 5](search-spec.md#7-sequencing)
- **all migrating repos:** record acceptance of what binds them.
  → [`search-requirements.md` SR-M3](search-requirements.md#m-lifecycle)

*2026-08-14:* the first two are **not** blocked on the same thing — metallm's
surface is released (0.24.2+), discodon's replay piece is unbuilt and waits on
its own Phase 1 storage port. See the correction under Phase 3.

*Status 2026-08-19 — nothing here has started, and the reasons differ per repo,
as the note above predicted.* The family has released six more times since
(v0.24.5 through v0.26.1, all tagged; 0.27.0 is bumped and in flight on
[#368](https://github.com/pacepace/3tears/pull/368), not yet tagged), none of it
search-side. **metallm is unblocked on every axis this document tracks**: it is
current at 0.24.4, its lag item is closed, and Gate B closed on 2026-08-14. What
it lacks is the branch — `feature/new-search` does not exist. **discodon is
still hard-blocked, and the chain runs back through discodon itself**: it needs the
replay piece (search Phase 3 item 8), which is unbuilt here —
`packages/search/src/threetears/search/` has `aggregate`, `select`, `extract`,
`bind`, `call`, `limiter` and `standalone`, and no `replay.py` — and which is
elicited against the storage port that is still uncarved *in discodon*. Not a
cycle — a three-link chain, discodon → 3tears → discodon — but one whose first
link nobody outside discodon can pull. Neither end has moved in the five days
since this was last written.

*Status 2026-08-19 (same day, later) — metallm's migration is up, and it is
almost entirely not a migration.* **Check 1 already passed**, by a deletion
that happened on 2026-06-23: metallm's `f0b6903` removed
`api/src/services/web_fetch_utils.py` as a file and replaced the SearXNG
scrape in `lookup_details` with `threetears.models.lookup_price`, as a side
effect of unrelated pricing work. No wrapper, no re-export;
`grep -rin searxng api/src/` is empty. Step 4's clause has no target either —
metallm deliberately refuses to sniff result text for structure and says so in
`_execute_service_tool`'s docstring. What remained was step 2, the pin:
[metallm#288](https://github.com/pacepace/metallm/pull/288) pins the whole
family at `v0.26.1`, which under D29 is the event that **binds these
contracts** — the first consumer pin against a version carrying search, which
is exactly what Gate B was standing in front of. It also unblocks
[metallm#287](https://github.com/pacepace/metallm/pull/287), a live production
defect (the model reading stringified `(content, artifact)` tuples) that is the
fourth site of what [#318](https://github.com/pacepace/3tears/pull/318)
and [#326](https://github.com/pacepace/3tears/pull/326) closed here.

**Neither metallm PR waits on anything in 3tears.** Both need only released
tags — `v0.26.1` for the pin, `v0.24.7` for what #287 consumes — so
[#368](https://github.com/pacepace/3tears/pull/368) and the unreleased 0.27.0
are not in their path. Merge order between them is #288 then #287.

**What this costs to know, and the rule it argues for.** Establishing the above
meant re-deriving a three-month-old deletion, because
[`search-spec.md` §7 Phase 5](search-spec.md#7-sequencing) states check 1 as
*instructions to change two named files* rather than as the property it wants.
Consumer-repo checks rot at the consumer's commit rate and nothing here
notices. Checks 2, 3, 9 and 10 are written the same way and are owed the same
re-verification before anyone works from them. The correction is recorded at
that section.

*Status 2026-08-19 (same day, later still) — the other four consumer checks were
re-verified against their trees, and the picture per repo is now firm.*

**samsung's phase-2 image search is built and is not on the leaf**, which is a
decision rather than a gap: `curation/src/curation/discovery/` carries
`phase_two.py`, an `ImageSearch` seam and an Art Institute provider, solving a
different problem from the leaf's (collection identity lookup, zero spend,
provider relevance scores refused as evidence). Its recorded reason for keeping
3tears out of the default install is transitive weight on a `MemoryMax=2G` Pi —
so **the bar the leaf must clear there is install size, not capability** — and
measured against that bar the leaf clears it: `3tears-search[standalone]` is 14
packages and 10 MB, of which curation already carries all but three wheels and
~684 KB, because both family leaves under it declare `dependencies = []`. The
exclusion samsung recorded was about `3tears-models`' closure and does not
transfer. `[extract]` is the extra to keep off that Pi (+16 packages, +65 MB,
babel and lxml dominating), and phase 2 has no use for it. Check 2 never asked
about size, so nobody measured it. Check 9's premise expired outright: the plane
that would call search is a long-lived uvicorn process whose sync code runs in a
threadpool, not the one-shot `asyncio.run()` the check describes.

**discodon has no 3tears dependency at all** — nothing in `pyproject.toml` or
`uv.lock`, no `threetears` import — so its steps 1 and 2 have no referent, and
its position is a first adoption rather than a migration. Check 3's spend half
is *already largely true by other means*: research-tool search dollars enter
`EvalRunCostCap` through a per-run rate card whenever the operator declared a
credit rate, while the standalone `web_search` tool's credits are deliberately
excluded until cassette withholding reaches that seam. Its replay half holds at
the two seams D28 gave discodon and is unbuilt at the one this program proposes,
on both sides. **Check 10 has no site**: no `nats` anywhere in discodon, 49
modules on `zmq`, no leaf consumed — there is no "before" for the switch to
preserve.

Detail and evidence for all four at
[`search-spec.md` §7 Phase 5](search-spec.md#7-sequencing); the check list at
[`search-requirements.md` §3](search-requirements.md#3-how-we-would-know-it-worked)
now carries a pointer to it.

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
- **discodon (first, and it is the only thing this phase currently waits on):**
  put the package boundary inside its own tree — the eval modules destined for
  `3tears-eval-contracts` stop reaching into the host, asserted by a test.
  Requirements, measured and stated as properties, at
  [`eval-extraction-discodon-requirements.md`](eval-extraction-discodon-requirements.md).
  Nothing moves out of discodon and no package is published; the lift becomes
  mechanical once it holds.

  **Held as of 2026-08-20.** In place: R1's boundary test; the `DocumentStore`
  port with its YugabyteDB adapter; the narrow completion protocol
  (`CompletionClient` / `CompletionResult`); R10's fidelity axis; an installable
  chart/spec package with a host-supplied palette.

  Outstanding — the first item was this phase's pair of open decisions and both
  are now ruled, so what remains of it is build:

  - **`SweepableValue`'s `display` and `scale`** (R9) and **the pooling boundary**
    (R11) are ruled (discodon Wave 2 design §5.2; C6) and unbuilt. Both are cheaper
    before the cut than after, so they still belong to this phase.
  - **`universe` → `scope`** — discodon Wave 2; touches the MCP surface. Binding
    eval-contracts before it lands means binding under provisional vocabulary.
  - **The package still imports `discodon.llm`** — discodon #2402–#2407.
  - **R8's input registry** — not built, and **R10's authoring-time gate cannot be
    built without it**, since controllability derives from the registry. One of
    R10's four axes exists.

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
before search's replay record schema is cut against them, and 3.14 must land
before discodon can consume anything the family ships. (This sentence used to
name `BudgetPort` wiring as the other thing elicited against those shapes;
that wiring was withdrawn on 2026-08-19 and the eval budget contract is shaped
from discodon's own instead.) Phases 2–3 are single-repo and mechanical. Phase 4 deliberately
precedes the extraction so the eval seams (budget, replay identity) are
extracted in their *post-migration* shape — extract-don't-invent applied to
sequencing: lift proven code, never code that is about to change. Phase 6
rides packages that are released and discodon-proven.
