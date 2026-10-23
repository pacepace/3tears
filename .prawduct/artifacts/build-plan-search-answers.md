---
artifact: build-plan
version: 2
scope: search-answers
branch: search/answers-slice-1
depends_on:
  - artifact: answers-requirements-design   # docs/answers-requirements-design.md
  - artifact: search-requirements           # docs/search-requirements.md
  - artifact: search-spec                   # docs/search-spec.md
partition: "serial -- each slice stacks on the last (contracts, then the adapter against the fixture, then the agent-tools builtin, then discodon), and slice 1 is one risk surface (the transport seam) small enough that a second agent would spend more on coordination than it saves"
last_validated: 2026-10-23
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

## Status

- [ ] Chunk 01: Contracts + StreamingTransport + standalone streaming (design §12 slice 1)
- [ ] Chunk 02: BraveAnswersAdapter + answer.py + bind_answer.py + enforcement rows (slice 2)
- [ ] Chunk 03: agent-tools `threetears.web_answer` builtin (slice 3)
- [ ] Chunk 04: discodon `fact_check` on Answers (slice 4, discodon's repo)
- [ ] Chunk 05: discodon `research__ask_web` behind its flag (slice 5, discodon's repo)
Context: Plan cut 2026-10-23 on `search/answers-slice-1`, stacked on
`docs/answers-requirements-design` (PR #930). Slice 0 (the live capture) is a prerequisite of
Chunk 02, not of Chunk 01, and has not run. Next: Chunk 01.

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
  §10.4), with `ANSWER_METADATA_KEY` and its enforcement row (§10.6). Ships in a MINOR release (AR-H3).
- **Depends on:** Chunk 01, and slice 0's recorded fixture (AR-I2)
- **Foreign API:** brave-answers (live capture first)
- **Done when:**
  0. verify-api: slice 0's capture settles the three (verify live) items; `Anchor` finalised
  1. Acceptance criteria met and tests pass
  2. `/prawduct:critic` run and blocking findings resolved
  3. Committed and chunk marked `[x]` in Status

### Chunk 03: agent-tools `threetears.web_answer`

- **Description:** Design §10.5 and AD13 (opt-in alias, owner may veto the name).
- **Depends on:** Chunk 02
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
