---
artifact: build-plan
version: 2
scope: search-answers
branch: search/answers-slice-2
depends_on:
  - artifact: answers-requirements-design   # docs/answers-requirements-design.md
  - artifact: search-requirements           # docs/search-requirements.md
  - artifact: search-spec                   # docs/search-spec.md
partition: "serial -- each slice stacks on the last (contracts, then the adapter against the fixture, then the agent-tools builtin, then discodon), and slice 1 is one risk surface (the transport seam) small enough that a second agent would spend more on coordination than it saves"
last_validated: 2026-10-24
---

## Requirements Confidence

**Level:** Medium

**Why:** The shape is settled: AD1-AD14 in `docs/answers-requirements-design.md` §8 rest on
code read the same day in both repositories, and every seam slice 1 extends already exists
(`SearchTransport`, `FetchTransport`, the standalone retry loop and SSRF guard, the
capabilities registry, the `SearchFailure` taxonomy). The provider is the weak side: every
Brave fact in design §4.2 comes from documentation, not from a live stream.

Problem: the search leaf has no way to read an open-ended streamed response, and no contract
for a provider's cited synthesis, so a cited Brave answer cannot be carried at all.

Success (slice 1): a host can inject a `StreamingTransport`; the standalone transport streams
a response chunk by chunk under a whole-stream byte cap, retrying only before the first byte,
through the same SSRF guard and connect pinning as its other two seams; and the answer
contract types exist, round-trip, canonicalise and register their capabilities.

Out of scope for slice 1: the Brave adapter, the SSE and tag parsers, `answer.py`,
`bind_answer.py`, `ANSWER_METADATA_KEY`, the `web_answer` builtin, and every discodon change.

**Open assumptions / unknowns:**
- [ASSUMPTION: `start_index`/`end_index` units (characters or bytes) and whether they index the
  text before or after tag stripping are UNVERIFIED | HIGH impact on `Anchor` | resolved by slice
  0's live capture. `Anchor` ships documented as provisional, its units not fixed]
- [ASSUMPTION: whether Brave's citation `snippet` is page text or model-written is UNVERIFIED |
  MED impact on AR-G1 | resolved by slice 0]
- [ASSUMPTION: the usage-report key names and the credit-limit refusal shape are UNVERIFIED |
  MED impact on slice 2's spend reader and error mapping | resolved by slice 0; slice 1 does not
  depend on them]

**What would raise confidence:** slice 0's live capture with discodon's provisioned key.

**Decisions taken in slice 1** (the design left these open):
- [DECISION: `StreamedResponse` is a runtime-checkable Protocol, not a frozen dataclass | it owns
  a live connection, so it cannot be a value; `stream()` is an `async def` returning it once the
  headers are in, and the caller releases it with `async with` or `aclose()` | user can veto]
- [DECISION: `stream()` takes `max_bytes` as a required per-call cap, and the effective cap is
  the lesser of it and the transport's configured `max_response_bytes` | the `fetch` precedent:
  a per-call cap can tighten the deployment cap, never loosen it | user can veto]
- [DECISION: the per-call timeout bounds the whole stream, body included, not only the wait for
  headers | a stream with no total bound can hold a caller forever on a trickle; Brave's own
  guidance is a total-answer timeout | user can veto]
- [DECISION: the capabilities registry is keyed by `(kind, provider)`; both declarations carry a
  `kind` literal (`"search"` / `"answer"`), and `get_capabilities(provider, kind="search")` keeps
  its one-call shape with the default preserving every existing caller | one provider (Brave)
  sells both kinds under one name, so keying by name alone would make the second registration
  replace the first | user can veto]
- [DECISION: `GroundedAnswer` refuses zero sources with zero notices | AG2 made structural: an
  answer with no sources and nothing saying so cannot be constructed | user can veto]
- [DECISION: `AnswerCapabilities` declares no `local_criteria` | a synthesised paragraph cannot be
  filtered after the fact, so a criterion is pushed down or unsatisfied | user can veto]
- [DECISION: what both capability declarations share -- the `<provider>:<unit>` composition, the
  disposition ladder and the one-bucket-per-key refusal -- lives once in the private
  `contracts/_declarations.py`, which both declaration modules import | one owner per rule, no
  import cycle between `capabilities.py` and `answer.py`, and no public name that exists only to
  be shared | user can veto]
- [DECISION: the streaming transport pins live in `packages/search/tests/test_conformance_streaming.py`,
  not as a shipped suite in `threetears.search.testing` | the provider suite ships because
  consumers write providers; no consumer implements a streaming transport, so a shipped suite
  would be public API with no caller (`test_public_api_has_a_caller`, CLAUDE.md). It moves to
  `testing/` when a host writes a second implementation | user can veto]
- [DECISION: `Anchor`, `AnswerRequest`, `GroundedAnswer`, `AnswerProvider` and `StreamingTransport`
  have no `src` caller until Chunk 02 (the adapter takes a `StreamingTransport` and returns a
  `GroundedAnswer`), so `test_public_api_has_a_caller` is red for them on this branch; no ledger entry
  is filed, because the ledger names callers outside the repo and these have none]
- [RULING (owner, 2026-10-23): slice 1 stacks with slice 2. Chunk 01 merges onto PR #930's area
  branch now; #930 is not mergeable to `develop` until Chunk 02 (Brave adapter, `answer.py`,
  `bind_answer.py`) is on it too, so the `test_public_api_has_a_caller` red is held on the area
  branch and never reaches `develop`. Design §12 says so]
- [RULING (owner, 2026-10-23): the standalone transport carries a per-upstream circuit breaker and
  a per-call span for `request`, `fetch` and `stream`, so every obligation the three protocols
  list is met by their one implementer. Breaker keyed by `host[:port]`, threshold and cooldown
  configurable on the transport; span through `threetears.observe.traced`, zero-cost without
  OpenTelemetry; an open breaker refuses before connecting as `TransportFailed`]
- [DECISION: an open breaker raises the existing `TransportFailed`, not a new class | its
  documented meaning is "the request could not complete at the transport -- give up rather than
  retry", which is what an open breaker says; a new class would break the seven-class taxonomy
  (SR-J1) and every consumer's one mapping of it (AD9) | user can veto]
- [DECISION: two release-failure paths in `packages/search/src/threetears/search/standalone.py` stay
  unpinned by a test, as an explicit descope: `_StandaloneStream._failed` containing a release that
  raises, and `StandaloneTransport._release_quietly` logging a 5xx head whose close raises before a
  retry | no release the transport performs can be made to raise from outside without a test-only
  seam. Tried against a real socket: the server resetting the connection mid-body (SO_LINGER 0,
  then abort) before the client's close, and the same over TLS, each with and without a connection
  scope -- every close completed cleanly, because anyio's socket close swallows OSError, httpcore
  wraps TLS with `standard_compatible=False` (so a close sends no close_notify that could fail),
  and httpcore shields its pool bookkeeping. Both handlers stay as defence: they cost nothing, and
  a future httpx/httpcore whose close can raise would otherwise put an untyped exception where a
  typed failure belongs. The cancellation half of the same finding is pinned
  (`test_a_close_cancelled_part_way_is_finished_by_the_next_close`) | user can veto]
- [DECISION: a stream's release runs as its own task, awaited shielded, so a cancelled `aclose`
  leaves it running and the next `aclose` waits on the same one | found while pinning the
  cancellation path: a cancellation landing inside httpcore's response close marked the response
  closed before its pool slot was handed back, leaving the slot taken for good, so "finish it on
  the next close" cannot be done after the fact -- only by never cutting the release off | user
  can veto]
- [DECISION: the breaker holds an entry only for an upstream with failures counted, open, or
  half-open; a call that returns an upstream to clean closed drops its entry | a long-lived fetch
  transport reaches whatever host each candidate names, and a map keeping every one would grow
  without bound | user can veto]
- [DECISION: the breaker judges a call, not an attempt: a call is a failure when it ends on a 5xx
  or on connect/timeout attempts that never got an answer, a success when it ends below 500, and
  nothing when it ends on a refusal of the transport's own (guard, cap) or a cancellation | a
  call that recovered on retry is the upstream working | user can veto]

**Decisions taken in slices 2 and 3** (the design left these open):
- [DECISION: `AnswerProvider` gains `estimate(request) -> Spend`, the figure `answer.py` hands
  `BudgetPort.check` | AR-C3's money rates are the configured instance's, which no capability
  declaration carries; recorded in design §10.1 | user can veto]
- [DECISION: the money estimate is the search at `usd_per_search`, plus the completion-token cap
  at `usd_per_million_tokens` when the request sets one; the two rates are given together or not
  at all | prompt tokens are unknowable before the call, so the estimate is a floor | user can veto]
- [DECISION: Brave's namespaced criteria are `brave:country`, `brave:safesearch`,
  `brave:search-context-size`, `brave:max-completion-tokens`; language rides the well-known
  `language` key | the criterion key grammar refuses underscores, and one name per parameter]
- [DECISION: a tag is `<name>` followed by `{`; any other `<` is answer text | prose holds `<`, and
  a tag reader that waited on every one would hold the answer hostage to a closer never sent]
- [DECISION: `RATE_LIMITED` and `QUOTA_LIMITED` map by Brave's published codes, marked in code as
  documentation rather than capture | the credit limit could not be provoked live]
- [DECISION: a stream ending without `[DONE]` is `MalformedResponse` | a cut-off answer must not
  read as complete]
- [DECISION: `bind_answer` declares `RenderedAnswer` and `bind_answer` only; it builds
  `AnswerMetadata` itself, stamping the version | the key's one construction site owns the shape]
- [DECISION: the answer key's literal guard covers dict-literal keys in `packages/search` and
  `packages/agent` | its value is the English word "answer", which evals uses for its own payloads]
- [DECISION: `create_web_answer_tool(config, description)` takes `api_key`, `transport`, `budget`,
  `limiter`, `timeout_seconds` from `config` | the `web_search` factory's shape, which the
  registry calls]
- [DECISION: the built-in pod serves `web_answer` when `THREETEARS_BRAVE_ANSWERS_API_KEY` is set and
  reports the skip otherwise | `web_search`'s pod path, mirrored; the leaf itself reads no
  environment | user can veto]
- [DECISION: `ANSWER_TOOLS` and `create_web_answer_tool` stay out of their modules' `__all__` | the
  alias map and the registry's by-name import are their only callers]
- [RULING (owner, 2026-10-24): slices 1-3 reach `develop` together on this area branch; the answer
  conformance suite and the stream replay live in `packages/search/tests/`; the `answers` alias
  stands; research mode stays out of v1; no trigger is tracked for LLM Context; `fact_check`
  grounds on citations alone. Design §14]

## Status

- [x] Chunk 01: Contracts + StreamingTransport + standalone streaming (design §12 slice 1)
- [x] Chunk 02: BraveAnswersAdapter + answer.py + bind_answer.py + enforcement rows (slice 2)
- [x] Chunk 03: agent-tools `threetears.web_answer` builtin (slice 3)
- [ ] Chunk 04: discodon `fact_check` on Answers (slice 4, discodon's repo)
- [ ] Chunk 05: discodon `research__ask_web` behind its flag (slice 5, discodon's repo)
Context: Chunk 01 merged onto `docs/answers-requirements-design` (PR #930) as #935. Slice 0
ran 2026-10-24 (design §4.2). Chunks 02 and 03 are built on `search/answers-slice-2`, which
merges onto the area branch; slices 1-3 then reach `develop` together (design §14). Every name
this area adds has a `src` caller. Next: Critic on Chunks 02-03, the area PR, then Chunk 04 in
discodon after the MINOR release.

## Scaffolding

### Project Initialization

None: the work extends `packages/search` in the existing uv workspace.

### Dependencies

None new. httpx stays the `[standalone]` extra's one weight (D24, AR-H1).

### Build & Test Configuration

`TESTAMENT=off ./scripts/test.sh search -q`, `./scripts/lint.sh`, `./scripts/typecheck.sh search`,
and the enforcement files named under Chunk 01, each run on its own.

### Scaffold Verification

The search package suite is green at the branch point.

### Verification Strategy

The standalone stream is pinned against a real loopback socket (`testing/http_server.py`),
never a mocked httpx, because the obligations are properties of what the transport does with a
socket. The live Brave stream is slice 0's, and Chunk 02 does not merge without it.

## Project Structure

```
packages/search/src/threetears/search/
  contracts/answer.py      # new: AnswerRequest, Anchor, GroundedAnswer, AnswerCapabilities, AnswerProvider
  contracts/transport.py   # + StreamingTransport, StreamedResponse
  contracts/capabilities.py# + kind-keyed registry
  contracts/_declarations.py # new: the rules both capability declarations share
  standalone.py            # + stream(), the per-upstream breaker, the per-call span
  testing/http_server.py   # + chunked, gated replies
packages/search/tests/
  test_conformance_streaming.py  # new: the streaming seam's pins
  test_answer_contracts.py       # new: the answer contract pins
```

### Module Boundaries

Only `standalone.py` imports httpx (AR-F3, check 10). SSE framing is the adapter's, never the
transport's (AR-F2). The answer contracts import nothing beyond the existing contracts.

## Build Chunks

### Chunk 01: Contracts + StreamingTransport + standalone streaming

- **Description:** The third transport protocol and its standalone implementation, and the
  answer contract types, so slice 2's adapter has a seam to read through and types to return.
  Ships in a MINOR release (AR-H3); the PR itself bumps nothing.
- **Depends on:** none
- **Artifacts consumed:** design §8 (AD5-AD7, AD9, AD11, AD12), §9 A, B, D, E, F, H, K, §10,
  §10.1, §10.6, §12, §13
- **Deliverables:**
  - `StreamingTransport` and `StreamedResponse` in `packages/search/src/threetears/search/contracts/transport.py`, with the module
    docstring's protocols account extended to three;
  - `StandaloneTransport.stream` in `packages/search/src/threetears/search/standalone.py`, sharing the retry loop, deadline,
    guard, connect pinning, byte cap, breaker and span with `request` and `fetch`;
  - new `packages/search/src/threetears/search/contracts/answer.py`; the kind-keyed registry in
    `packages/search/src/threetears/search/contracts/capabilities.py`; exports;
  - the streaming transport pins (new `packages/search/tests/test_conformance_streaming.py`) and a
    parity-declared fake streaming transport;
  - the httpx-only-in-standalone pin covering the stream path;
  - design §13's rider on `docs/search-spec.md` §5, and the spec's protocol list and §3.8.
- **Tests:** behavioural, against a real socket: chunks arrive incrementally; status and headers
  before the body; the byte cap refuses an overrun mid-stream holding no more than the cap, and
  a declared overrun before the body; retry on connect failure and 5xx before the first byte and
  never after; the timeout bounds the stream; early close releases the connection; the SSRF
  guard refuses a private address and pins the connection to the checked one. Contract pins:
  `extra` forbidden, round-trip, canonical form, registry `kind`, zero sources valid.
- **Acceptance criteria:** the search suite, lint, `typecheck.sh search`, and the enforcement
  files pass, except the accepted reds named in the handoff (the minor-bump gate, and
  `test_public_api_has_a_caller` for the answer names whose first caller is Chunk 02).
- **Done when:**
  1. Acceptance criteria met and tests pass
  2. `/prawduct:critic` run and blocking findings resolved
  3. Committed and chunk marked `[x]` in Status

### Chunk 02: BraveAnswersAdapter + answer.py + bind_answer.py

- **Description:** The one provider, the call layer and the prose/metadata render (design §10.2-
  §10.4), with `ANSWER_METADATA_KEY` and its enforcement row (§10.6). Ships in a MINOR release
  (AR-H3); the PR bumps nothing.
- **Depends on:** Chunk 01, and slice 0's recorded fixture (AR-I2), both on the area branch.
- **Foreign API:** brave-answers. Slice 0 ran 2026-10-24; its facts are design §4.2 "Verified
  live", and the recordings are `packages/search/src/threetears/search/testing/recordings/`.
- **Artifacts consumed:** design §4.2, §8 (AD5-AD10, AD12), §9 A-E, G, I-K, §10.2-§10.4, §10.6.
- **Deliverables:**
  - `contracts/metadata.py`: `ANSWER_METADATA_KEY = "answer"`, `ANSWER_SCHEMA_VERSION`,
    `AnswerMetadata`, beside the search key's;
  - `adapters/brave_answers.py`: `BraveAnswersAdapter` (an `AnswerProvider`) per §10.2: the SSE
    reader and the tag state machine (outside / `<citation>` / `<usage>` / unknown tag, buffered
    across SSE events and chunk boundaries), the spend reader (AD10; cost numbers decoded as
    `Decimal`), the error mapping (one `ErrorResponse` envelope keyed by `error.code`; a 422
    `SUBSCRIPTION_TOKEN_INVALID` is an auth refusal, not a malformed request; 429 `RateLimited`;
    the credit-limit code `QuotaExhausted`), capabilities registered with `kind="answer"`;
  - `answer.py`: the call layer, mirroring `call.search` (§10.3);
  - `bind_answer.py`: `bind_answer` and `RenderedAnswer` (§10.4), the answer key's one
    construction site, AR-G2's first line, AR-A3's zero-source first line;
  - the recordings shipped as package data under `testing/recordings/` and read through
    `importlib.resources`; a recorded streaming transport (the `recorded.py` shape) that replays
    them, and an `AnswerConformanceSuite` the Brave adapter passes against them, both in
    `packages/search/tests/` (owner ruling, design §14: no consumer writes an answers provider,
    and nothing in `src` replays a stream);
  - `GroundedAnswer` refuses an anchor past the end of `text`, or one naming no source;
  - `tests/enforcement/test_one_search_result_shape.py` generalised to `(key, sanctioned site)`
    rows with the answer key's row; exports in `__init__.py`; `docs/search-spec.md` names the
    new modules.
- **Tests:** against the recordings: text equals the stripped stream, anchors are the recorded
  points in code points (the non-ASCII capture pins character, not byte, units), sources in
  citation order with `content=None` and the snippet carried, `provider_ids` keeping `number`
  and the non-text facts, spend equal to the usage tag's own total. Synthetic streams for the
  edges: a tag split across events and mid-UTF-8 chunk, an unknown tag (AR-A5), no `<usage>`
  (AR-C1), unparseable tag JSON and missing cost keys (AR-E2), zero citations (AR-A3), a break
  after the first byte (AR-E1, never retried), each error envelope. Call layer: budget check
  before, record after, pacing, dispositions. Bind: first line, sources list, metadata
  round-trip, failure metadata.
- **Acceptance criteria:** the search suite, lint, `typecheck.sh search`, and the enforcement
  files pass. `test_public_api_has_a_caller` is green for every name this area adds once Chunk
  03 is on the branch. The one held red is `test_api_growth_requires_a_minor_bump`.
- **Done when:**
  1. Acceptance criteria met and tests pass
  2. `/prawduct:critic` run and blocking findings resolved
  3. Committed and chunk marked `[x]` in Status

### Chunk 03: agent-tools `threetears.web_answer`

- **Description:** Design §10.5 and AD13 (the `answers` alias; the owner confirmed the name).
  Built on the same branch as Chunk 02, because Chunk 02's names have their first caller here.
- **Depends on:** Chunk 02
- **Deliverables:**
  - `builtin/web_answer.py`: `WebAnswerTool(TearsTool)` mirroring `web_search` -- `question` in,
    `RenderedAnswer.content` on `content`, the projection on `metadata[ANSWER_METADATA_KEY]`, a
    failed `ToolResult` carrying the failure record -- and `create_web_answer_tool(config,
    description)` taking `api_key`, `transport`, `budget`, `limiter`, `timeout_seconds`, and
    optionally `usd_per_search` / `usd_per_million_tokens` (both or neither; a finite number of 0
    or more, refused naming the key) to price the budget's pre-call estimate;
  - registered in `builtin/__init__.py`'s list, and served by the built-in pod when
    `THREETEARS_BRAVE_ANSWERS_API_KEY` is set, with the skip reported otherwise. Known limit: the
    pod reads no rates (as `web_search`'s pod reads no optional settings), so a pod-served
    `web_answer` cannot refuse on money before the call;
  - `aliases.py`: `answers` → `{"threetears.web_answer"}`; `web` unchanged;
  - the agent-tools README's builtin list and alias table.
- **Tests:** the tool over a parity-declared fake streaming transport replaying the live
  capture (first line, projection, sent body), a refusal as a failed result with its record,
  an empty question refused before sending, the factory's key requirement, registry and pod
  registration, and the alias (`answers` grants it, `web` does not).
- **Acceptance criteria:** `TESTAMENT=off ./scripts/test.sh agent/tools -q`,
  `./scripts/typecheck.sh agent/tools`, lint, and the enforcement files pass;
  `test_public_api_has_a_caller` is green for every name this area adds.
- **Done when:** tests pass; `/prawduct:critic` run; committed and marked.

### Chunk 04: discodon `fact_check` on Answers

- **Description:** Design §11.1; AC1 and AC2 measured on prod. Built in discodon's repo.
- **Depends on:** Chunk 03 released

### Chunk 05: discodon `research__ask_web`

- **Description:** Design §11.2-§11.3, behind a flag, with the stated hypothesis measured
  before it leaves the flag. Built in discodon's repo.
- **Depends on:** Chunk 04
- **Type:** cumulative-final

## Early Feedback Milestone

**Milestone chunk:** 04
**What the user can do:** see a persona's `fact_check` answer cited, and Brave spend in
discodon's budget rows.

## Governance Checkpoints

**Commit & PR cadence:** one PR per slice into `develop`, stacked (design §12). Work PRs do not
bump versions; the maintainer bundles the MINOR bump.

- After Chunk 01: the transport seam reviewed before an adapter is written against it.
- Before Chunk 02: slice 0's live capture.
- After Chunk 03: one Critic over Chunks 02-03, then the area PR carrying slices 1-3.
