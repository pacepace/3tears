# Search: The Specification

**Status:** Draft for build planning — 2026-08-04
**Scope:** the next level down from `search-architecture.md`: decisions taken,
package and module breakdown, MUST/SHOULD/MAY/MUST NOT requirements, and
broad-stroke sequencing for the build and for migrating consuming apps. Not a
build plan — the build plan derives from this, under prawduct, and lives in the
planning session's local `.prawduct/` (gitignored here by design), so **this
document is the durable record** of what was decided and why.

**Companions** — read in the order *direction → need → shape → spec*:

| Document | Carries |
|---|---|
| [`family-convergence.md` §4.14](family-convergence.md#414-web-search--one-contract-staged-pipeline-searxng-from-metallm-budgets-from-discodon) | the **direction** |
| [`search-requirements.md`](search-requirements.md) | the **need** — evidence, requirement IDs (`G*`, `P*`, `SR-*`), success checks |
| [`search-architecture.md`](search-architecture.md) | the **shape** — six layers, the seams, what each consumer does |
| **this document** | the **spec** — the buildable statement |

This is the newest of the five search documents and the authority for anything
build-facing. Requirement IDs cited here (`SR-*`, `G*`, `P*`, "check N") are
defined in `search-requirements.md` and are not restated; the builder reads
that document once, then works from this one.

Everything below was written against code verified on 2026-08-04 (post-merge
from develop at 0.23.0), including a correction pass on the three older
documents made the same day. **Re-verified 2026-08-10** against develop at
0.23.11: every build-facing claim held — the gutting targets, check 4's
consumer, the mcp serializer, and `media-contracts` are all unchanged — and
the two envelope deltas that landed meanwhile (`CallRequest.result_subject`;
manifest-level `timeout_seconds` with server-side enforcement) are recorded
at D18 and §4.6.

---

## 1. Decisions taken

Each entry adopts the recommendation recorded in `search-requirements.md` §13
unless noted, as a **vetoable ruling**: `[DECISION: … | why | user can veto]`.
A veto is recorded in the requirements doc and propagated here. Rationale
lives with the requirement; only the ruling and its build consequence appear
here.

| # | Ruling | Build consequence |
|---|---|---|
| D1 (SR-A4) | Named, provenanced scores; **no single `score` field, ever** | Result core carries a set of score entries — name, value, scale semantics, source (provider or stage), cross-provider comparability flag. A comparable relevance exists only if Select produced one. |
| D2 (SR-A5) | Call returns a candidate set; the corpus is Aggregate's named type | Two types, two dedup/merge stories; Call never accumulates. |
| D3 (SR-B5, OQ21) | Model-mediated search is out of Adapter and Call, in at **Aggregate** as a candidate producer | Provenance carries a `producer` distinction from day one; the producer seam is designed in Phase 3, implemented when samsung pulls. Its token cost is owned by the models usage tracker — the producer seam records a *reference* to that spend and MUST NOT re-price it into search spend (no double counting). |
| D4 (SR-D4) | Budget follows the bill | The budget increment and the transport retry sit on the same side of the seam: a retried attempt that never billed never counts. C2's fail-closed retry bound moves into the transport's bounded-retry config in the same change. |
| D5 (SR-D5) | Both refusal authorities, distinct roles | Local caps bound a run's *shape* (overrun is a defect); provider refusal bounds *money*. Neither substitutes for the other. |
| D6 (SR-E6) | Self-hosted cost is zero | The rate/quota spend dimensions carry the real constraint (SR-D6); no synthetic infrastructure pricing. |
| D7 (SR-F5) | Replay recordings go through a **consumer-supplied store port** | Embedded: the consumer passes a port object. Pod-resident: the consumer passes a store *reference* the pod resolves to its own implementation. The port follows `media-contracts`' `ObjectStore` shape. |
| D8 (SR-H4, SR-N4) | Pace, don't just react; keyed on `(provider instance, egress)` | Two mechanisms: an in-process limiter shipped in the leaf, and a distributed-limiter port `core`'s `TokenBucket` satisfies where a bus exists. |
| D9 (SR-I4) | Return records, emit nothing | The capability owns no sink; hosts wire `observe` where they have it. |
| D10 (SR-J3) | Typed exceptions carrying spend; prose at Bind; **Bind converts before the wire** | Nothing raises across the NATS hop — a failed call arrives as a failed `ToolResult` with spend on `metadata`. This is what makes SR-E3 hold pod-resident today despite §10.9. |
| D11 (SR-K2) | Queries are user content | The capability makes the query available for redaction; redaction policy stays with the consumer. |
| D12 (SR-K4) | Family stance, enforced per adapter *(needs cross-repo ratification — flagged, not silently ruled)* | Proposed stance: provider API calls are governed by provider terms, documented per adapter; Extract's direct fetches honor robots.txt by default, with a per-deployment override that is *recorded config, never code*; retention of recorded content follows the consumer's policy (D7 puts the bytes in the consumer's store, which is what makes that dischargeable). |
| D13 (SR-M1) | In-family versioning is lockstep (already ruled); the **wire payload carries a schema version** | The metadata payload and replay record embed `schema_version`; changes are additive within a family minor. Formal wire-compatibility promise is a **gate before the first pod-resident deployment**, not before first release. *Gate A rider (2026-08-10):* with `extra="forbid"` on every contract type and consumers pinning exact versions, the additive promise is scoped to **exact-version pairs** until that gate; the gate's promise MUST include flipping wire-*read* payload types (`FailureRecord`, `SearchResultsMetadata`, `Candidate` and nested) to ignore-unknown, keeping strict rejection for caller-constructed inputs — and the metadata reader already checks `schema_version` before structural validation so a version refusal names versions, never surfaces as a pydantic error. |
| D14 (SR-M2) | No response caching in v1 | MUST NOT, beyond whatever a provider does upstream. Revisit after replay ships, in its light — a cache here has two different legal shapes (SR-O3). |
| D15 (SR-M3, OQ13) | Ratification home is `search-requirements.md` | discodon, metallm, samsung each record acceptance of what binds them, in their own repos, pointing at it. |
| D16 (§5.4) | The v1 wire hop is the existing `TearsTool` envelope, at Bind | No new wire protocol. Every contract type still JSON-round-trips (SR-L4) so a future intra-stack hop stays open — paid in design discipline, not in v1 machinery. |
| D17 (§5.5) | One tool, one contract, all faces; search stays in the `web` alias; `skill_eligible = False` initially | Image/carrier scoping is a *criteria* parameter of the one tool, not a second tool — so an agent granted `web` gets exactly what it got before. Samsung's image search is embedded and never enters the tool surface. |
| D18 (§10.9, §10.10) | Both envelope asks are accepted as in-repo work | (a) exception-path metadata carry; (b) an optional per-call deadline field. Sequenced in Phase 2 with an explicit rollout order — the server must accept the field a release before any client sends it, because `extra="forbid"` on an old server rejects unknown fields. *Re-verified 2026-08-10 (0.23.11):* both asks still open; the envelope meanwhile gained `result_subject` additively — live precedent for exactly this rollout — and a manifest-level `timeout_seconds` the deadline field must compose with (§4.6). |
| D19 (SR-N1) | The no-bespoke-client norm **widens**; no exemption is filed | Verified mechanism: `_SANCTIONED_HTTPX_SITES` is a path frozenset, and the walker only flags raw httpx clients stored on `self` — a protocol-typed transport field never trips it. The widening = add the leaf's standalone-transport module path to the sanctioned set + restate the norm prose. Lands in the same PR as that module (check 11). |
| D20 (SR-N2) | Egress is per-upstream input at Adapter and provenance on every result; `direct` is a named value | Rate/ban budgets key on it (D8); replay comparability depends on it. |
| D21 (SR-K3, SR-N3) | The SSRF ruling binds at the transport seam | Provider base URLs come from deployment config only — MUST NOT accept a caller-supplied base URL. Redirect policy and private-address guards live in the transport implementations, not per call site. |
| D22 (§10.12) | Structured results ride `ToolResult.metadata` under a named key | `SEARCH_RESULTS_METADATA_KEY = "search_results"`, following the `OBJECT_HANDLE_METADATA_KEY` precedent, defined in the leaf's contracts. |
| D23 (packaging) | **One package, `3tears-search`**, import root `threetears.search`; contracts as an import-clean module, not a separate package | See §2. The alternative (a separate `3tears-search-contracts`, the eval precedent) is not taken because the whole package already sits at the contracts-leaf floor; import paths are chosen so a later split is a non-breaking move (the OQ3 discipline). Split trigger: a consumer that needs the types but must refuse even `observe` + `media-contracts` + pydantic — none exists or is foreseen. |
| D24 (leaf floor) | Hard deps: `pydantic`, `3tears-media-contracts`, `3tears-observe`. Extras: `[standalone]` = httpx (the bare transport impl), `[extract]` = trafilatura | Matches SR-L7's permitted floor exactly. Provider adapters ship in the base package — they are pure logic over the injected transport and weigh nothing; extras carry *weight*, and the only weights are httpx (only for hosts that don't inject their own transport) and trafilatura. |
| D25 (Python floor) | The leaf declares `requires-python = ">=3.14"` today, avoids gratuitous 3.14-only surface | The workspace is 3.14. **OQ1 ruled in principle 2026-08-04: discodon adopts 3.14** — its declared floor is already `>=3.12`, so this is an interpreter switch plus verification, owned by discodon. The avoid-3.14-only-surface intent stays as cheap insurance until that lands; the per-module-floor fallback is retired unless adoption hits a wall. |
| D26 (replay durability, 2026-08-04) | Recordings outlive the stack that made them: replay records the **typed result** and keys on the **canonical caller request** | Three rules, detailed in §3.10: the key hashes explicitly-set caller parameters (never resolved defaults) plus a key-derivation version; replay short-circuits at the Call boundary and never touches an adapter, so removing a provider strands no recordings; payload readability is promised within a family major, refused loudly across, matching the cascade-delete lifetime recordings actually have (SR-F5). |
| D27 (replay spend, 2026-08-04) | Replay reports **both** spends, never one field: the recording's original spend rides inside the replayed payload; the replay's own execution spend rides where spend always rides | Budgets bind on execution spend; cost-model analyses read recorded spend, so a replayed baseline never looks free. P7 applied to spend. Detail in §3.10. |
| D28 (recorder composition, 2026-08-04) | Multiple freezing seams coexist under one rule: **the outermost active recorder wins** | Search replay is the innermost seam and the only one reaching non-Tool callers; character/agent evals freeze coarser (discodon's action- and delivery-seam cassettes), correctly. **No replay engine enters the `TearsTool` base class.** Detail in §3.10. |

Two §13 rows are *not* ruled here because nothing in Phases 1–4 needs them:
**one NATS bus or two** (gates only how much distributed pacing the client side
can carry, after convergence), and the final **wire-boundary placement** beyond
D16 (kept survivable by SR-L4).

---

## 2. The package

**PyPI `3tears-search` · import `threetears.search` · `packages/search/`** in
this monorepo, standard shape (hatchling, `src/threetears/search/`, `tests/`,
`py.typed`, no `__init__.py` above the leaf). It joins the lockstep family:
version = family version, intra-family deps carry the enforced
`>=<major>.<minor>.0,<major>.<minor+1>.0` bound.

```
threetears/search/
  contracts/        # the leaf within the leaf — types, protocols, errors, keys
  adapters/
    searxng.py      # Adapter: SearXNG
    tavily.py       # Adapter: Tavily (ported from discodon — extract, don't invent)
  call.py           # Call
  aggregate.py      # Aggregate  (Phase 3)
  extract.py        # Extract    (web path Phase 2; carrier dispatch Phase 3)
  select.py         # Select     (Phase 3)
  bind.py           # Bind helpers: prose render + metadata projection
  standalone.py     # bare-httpx transport impl   [standalone] — the sanctioned path (D19)
  limiter.py        # in-process pacing + distributed-limiter port (D8)
  replay.py         # record/replay over the store port (Phase 3)
  testing/          # provider-conformance suite + parity-declared fakes (SR-O5)
```

Layer names (Adapter, Call, Aggregate, Extract, Select, Bind) are **module
vocabulary, not type names** — the requirements doc's own warning (§12): the
fewer of them that appear in the contract as types, the cheaper a re-cut stays.
Contract types are named for what they are (`SearchRequest`, `Candidate`,
`Corpus`, `Spend`…), never for the layer that makes them.

**Package-level requirements**

- MUST keep `contracts/` import-clean: importing it pulls nothing beyond
  stdlib, pydantic, and `media-contracts` types. Enforced by the package's
  import-cost test (§6).
- MUST NOT import `threetears.core` from anywhere in the package (SR-L7), nor
  `threetears.agent.*`, `langchain*`, or NATS from anywhere. The package
  depends downward only.
- MUST have every contract type wire-serialisable — JSON round-trip with no
  callables, open files, or port objects in any result/record type (SR-L4).
  Port objects are *parameters*, never *payload*.
- MUST be usable from a one-shot `asyncio.run()` — no ambient loop, no
  long-lived client, no background task required for a single call (SR-L5,
  check 9).
- MUST ship safe-unturned concurrency defaults (SR-L6): defaults that hold
  under a `MemoryMax` cap with nothing tuned.
- MUST NOT read environment variables or ambient config anywhere (SR-K1): the
  host passes base URLs, secret references (`scheme://locator`) or resolved
  values, and transport.

---

## 3. Modules

For each module: what it turns (from the architecture doc), then its binding
requirements. "Caller is told" always means: in the typed response, not in a
log.

### 3.1 `contracts/`

The lingua franca. Types (name-level; fields only where a requirement forces
them):

- **`SearchRequest`** — query text, criteria, requested fidelity, opt-in
  record flag, budget scope tags. MUST treat query as user content (D11).
- **Criteria** — one open vocabulary (P6, SR-B1): well-known criteria ship as
  typed constructors (time range, domains include/exclude, language, carrier,
  min resolution, rights class, …), unknown criteria as namespaced keys. MUST
  NOT be a closed enum. The response MUST carry a per-criterion disposition —
  `pushdown | local | unsatisfied | ignored-unknown` (SR-B2, SR-B3, P8); an
  unsatisfiable criterion is named, never dropped.
- **`Candidate`** — the carrier-neutral result core (SR-C1): identity,
  locator(s), provenance, scores, fidelity available/achieved, an optional
  content slot recording whether content arrived with the response or from a
  later fetch (SR-A2), and **facets** — additive, keyed by the
  `media-contracts` vocabulary, ignorable by consumers that don't recognise
  them (SR-C2, SR-C3). MUST NOT define a closed carrier union.
- **Provenance** (on every candidate and every spend/replay record, P2,
  SR-A3): query, provider instance, provider-native identifiers, retrieval
  time, **egress name with `direct` as a value** (D20), and producer class
  (API provider now; model-mediated later, D3).
- **Scores** — per D1. MUST mark provider-native scores non-comparable across
  providers.
- **`Corpus`** — Aggregate's accumulation type with a stated dedup key and
  merge rule (D2, SR-A5).
- **`Spend`** — every resource a call consumed (SR-E1): money (Decimal),
  wall-clock, call count, weighted provider units (SR-E4), bytes. MUST survive
  the failure path (SR-E3); the count a cap enforces and the count a bill
  prices MUST be the same number (SR-E2); per-request (not per-result) pricing
  must be representable (SR-E5).
- **Typed errors** — the seven distinguishable failure classes of SR-J1, each
  carrying `Spend`; remediation text where the cause is known (the SearXNG
  403-json-formats teaching error). Zero results is a success value, not an
  error (SR-J2). The wire record carries provenance enough to rebuild D8's
  pacing key consumer-side — provider instance, egress, occurrence time —
  because pod-resident it is the only fact that survives the wire (D10, P2;
  Gate A, 2026-08-10).
- **Protocols** (structural, injected — P9): `SearchTransport` (shaped so
  `core.http_client.TracedHttpClient` satisfies it via a thin host-side
  adapter: configurable timeout, bounded retry, circuit-breaking, per-call
  span, egress selection — SR-N1, SR-G1, SR-G4, SR-D3), `FetchTransport`
  (the streamed, byte-capped, content-type-gated read Extract requires —
  declared as a *second* protocol at Gate A, 2026-08-10, so Phase-1
  `SearchTransport` implementers are never retroactively non-conformant;
  the standalone transport and the host adapter implement the union from
  Phase 2), `SearchProvider` (the provider seam Call consumes and the
  conformance suite parametrizes over — named here at Gate A; it is the
  one seam-vocabulary addition §3.1's original field list did not carry),
  `BudgetPort` (`check(estimate)` / `record(spend)` with plural scopes —
  SR-D1, SR-D2), `RateLimiterPort` (D8), `RecordingStore` (D7,
  `ObjectStore`-shaped, streaming), `HeavyFetcher` (implemented by
  `3tears-scrape`, never imported).
- **Replay record** — typed envelope (id — UUIDv7-compatible, created-at,
  provider, key, size, `schema_version`) over a payload that can rebuild the
  corpus (SR-F4); the key is derived by search (SR-F8).
- **Canonical serialization is a public contract feature, not a replay
  internal.** One canonical form, two consumers that must agree: the D26
  replay key and eval run identity (SR-F1 — search parameters already
  participate in discodon's `canonical_digest`). MUST be exposed on the
  request/parameter types. Only the *semantic* parameters participate —
  query, criteria, fidelity; the operational fields (`record`,
  `budget_scope_tags`) MUST NOT enter the canonical form: a recording is
  made with `record=True` by definition (SR-F6) and replayed without it
  (SR-F7), and scope tags carry per-run identity (SR-D2), so keying either
  strands recordings and gives every eval run a unique digest *(Gate A
  finding, 2026-08-10)*.
- **`SEARCH_RESULTS_METADATA_KEY`** and the metadata projection schema, with
  `schema_version` (D13, D22).

MUST version additively within a family minor (D13). SHOULD keep every type
constructible with defaults-off — no hidden globals.

### 3.2 `adapters/` — SearXNG, Tavily

One provider's API each, through the injected transport only. Each adapter:

- MUST declare capabilities queryably (SR-B4), following the
  `3tears-models` capability-metadata pattern — SearXNG: categories, engines,
  language, safesearch, paging, time range; Tavily: depth, domains, topic,
  dates.
- MUST keep everything the provider returns that P2 protects — scores, engine
  attribution, published dates — in typed form, not a disclaimed `raw` blob.
- MUST attach `Spend` to every call including failures; Tavily MUST weight
  units correctly (`advanced` = 2 credits — the SR-E4 live defect must be
  impossible to reproduce here).
- MUST map provider failures onto the typed error taxonomy (SR-J1, SR-D3 —
  quota exhaustion short-circuits distinctly from a local cap).
- MUST take base URL and credentials from the host (D21, SR-K1); MUST NOT
  default them from env.
- SHOULD implement pushdown for every criterion the provider can express and
  report `local`/`unsatisfied` for the rest (§7 of the requirements doc).
- Tavily MUST be ported from discodon's wrapper (principle: extract, don't
  invent), preserving its hard-won semantics — depth/credit coupling, domain
  scoping, score coercion, absolute-dates-beat-time_range (SR-B3's RES-T4M9
  precedent).

Conformance: both pass the shared suite in `testing/` (§6).

### 3.3 `call.py`

A query → one candidate set, through one adapter. Owns criteria negotiation
with the adapter's declared capabilities, failure mapping, spend attachment,
budget consultation (D4, D5), pacing (D8), and replay record/replay hooks
(Phase 3). MUST be the layer where "budget follows the bill" is enforced —
below the retry boundary, so retried-but-unbilled attempts don't count (D4).
MUST apply safe default bounds when the caller tunes nothing (SR-L6).

### 3.4 `aggregate.py` *(Phase 3)*

Many calls → one set. Owns the dedup key, the merge rule, fan-out accounting
(SR-H2: within-batch and cross-run bounds; SR-H3: one failure never poisons
siblings), and the `Corpus` type. MUST accept candidates from an external
producer (D3's model-mediated seam) without them impersonating a provider —
provenance keeps the classes distinct. MAY implement reciprocal-rank fusion
across engines/providers (prior art: `Lombey/Local-Web-Search-MCP`); MUST NOT
require it.

### 3.5 `extract.py` *(web path Phase 2; carrier dispatch Phase 3)*

A carrier → the information in it. Carrier-dispatched (SR-C4); a consumer MUST
be able to take search with no extraction at all. Requirements:

- MUST no-op (and cost nothing) when the provider already supplied content
  (SR-A2 — the Tavily case).
- MUST stream with a byte cap and a content-type gate; MUST NOT hold an
  unbounded `resp.text` (SR-G5). This is the acute `MemoryMax` case. The
  seam that carries this is `FetchTransport` (§3.1) — Extract's fetches go
  through it, never through `SearchTransport.request`, whose fully-buffered
  response shape cannot express the cap (Gate A, 2026-08-10).
- MUST honor the D12 robots stance; the enforcement point is here and in the
  transports, not per call site.
- MUST record extraction method and status on the result (fidelity achieved,
  SR-B6), using `media-contracts`' `extraction_status` vocabulary.
- Escalation to hostile targets goes through the `HeavyFetcher` protocol slot;
  `3tears-scrape` implements it; this package MUST NOT import scrape. SHOULD
  make escalation explicit (a caller choice), not automatic — silent
  escalation multiplies cost (shared_search OQ3, resolved conservative).
- MAY add a Wayback fallback tier later (prior art: `TadMSTR/searxng-mcp`);
  not in v1.

### 3.6 `select.py` *(Phase 3)*

Candidates + criteria → an ordered, filtered subset. Owns local criteria
application and the cull; exposes a **ranker slot** and never a ranking
implementation (§4.14's ruling — MMR lives in `agent-memory`, rerank metadata
in `3tears-models`, a cross-encoder arrives as a models provider). MUST mark
unranked output as unranked (SR-L2, P8). MUST satisfy P4's acceptance test: a
consumer supplying its own ranker can still constrain carrier type; a
consumer wanting the cull pays for no reranker.

### 3.7 `bind.py`

Candidates → what the caller consumes. Two bindings ship: prose-for-a-model
(the LLM rendering, migrated from `_format_results` but structure-preserving
underneath) and the metadata projection under `SEARCH_RESULTS_METADATA_KEY`
(D22, explicit border projection à la `ObjectHandle.to_metadata`). MUST catch
every typed exception and render a failed result carrying spend — nothing
raises across the wire (D10). MUST NOT import `agent-tools` — the `TearsTool`
gutting consumes these helpers, not the reverse. One binding path serves all
three faces (check 14); a face-specific response shape is a regression by
definition.

### 3.8 `standalone.py` — `[standalone]`

The bare-httpx `SearchTransport` implementation for hosts without core
(samsung; any embedded consumer). Carries the same obligations the injected
core transport gives for free: configurable timeout, bounded retry with
backoff, per-attempt accounting visible to spend, SSRF guards
(private-address and redirect policy per D21), streamed reads with caps. This
module's path is the one added to `_SANCTIONED_HTTPX_SITES` (D19). MUST NOT
be imported by anything else in the package at module level — it is an
implementation a host chooses. When Extract lands (Phase 2) this module
implements `FetchTransport` alongside `SearchTransport` — the union is the
declared shape (Gate A, 2026-08-10), and its per-request-client lifecycle is
revisited in the same change (right for one search, wrong for Extract's
many-fetch path).

### 3.9 `limiter.py`

In-process token-bucket pacing keyed `(provider instance, egress)` (D8, D20),
plus the port a distributed implementation satisfies (`core`'s NATS
`TokenBucket`, host-injected, where a bus exists). The in-process limiter's
state is the argued SR-O2 allowlist entry — argued in the build plan, not
assumed. MUST be on by default with safe rates (SR-L6); the shared SearXNG's
own server-side limiter remains the backstop that covers non-cooperating
deployments (SR-H4's honest layering).

### 3.10 `replay.py` *(Phase 3)*

Record/replay attached at Adapter/Call (SR-F3), writing through the
consumer's `RecordingStore` (D7). Opt-in per call (SR-F6); a replay miss is a
typed error, never a silent live call (SR-F7); recordings rebuild the corpus,
not just rendered text (SR-F4); ids are UUIDv7 (SR-O4), generated by the
writer. Retention, purge, and redaction belong to the store's owner (D7,
D12).

**Durability against stack evolution (D26).** The replay key is an opaque
digest used for equality lookup only — nobody ever parses it — so the risks
are derivation drift and payload readability, and each gets a rule:

- **Canonical-request keying.** MUST derive the key from the caller's request
  in canonical form — explicitly-set *semantic* parameters only (operational
  fields like the record flag and budget scope tags never participate; Gate A,
  2026-08-10), stably serialized, with absent and defaulted canonically
  identical — plus provider-instance identity and profile digest (SR-F8);
  MUST NOT derive it from the resolved provider wire request. Adding a
  parameter with a default therefore shifts no existing key. The record envelope MUST carry a key-derivation version;
  a genuinely incompatible derivation change bumps it, and the resulting
  miss names both versions instead of being mysterious.
- **Adapter-free replay.** The recorded payload is the contract-shaped typed
  result (candidates, scores, dispositions, spend) per SR-F4, so replay MUST
  short-circuit at the Call boundary — deserialize and return — and MUST NOT
  require the recording's provider adapter to be installed. Removing a
  provider strands no recordings; provenance keeps naming it as historical
  fact. Layers above the frozen exchange run live, which is the point: a
  replayed eval measures pipeline changes against frozen web input.
- **Versioned refusal.** Payload readability follows D13 — `schema_version`,
  additive within a family minor — and is promised **within a family major**;
  a reader meeting a version it cannot read MUST refuse with a typed error
  naming both versions, never best-effort misread. The promise is scoped to
  the lifetime recordings actually have (cascade-delete with the owning run,
  SR-F5) — this is not an archival format and MUST NOT be priced as one.

**Dual spend on replay (D27).** A replayed result MUST carry the original
call's spend inside the replayed payload — it is part of what SR-F4
preserves — AND the replay's own execution spend in the ordinary place, and
MUST NOT merge them. Budget ports are consulted with execution spend only (a
replay debits no provider quota; wall-clock bounds still bind); cost-model
analyses read the recorded spend. Provenance MUST mark the result replayed so
the two readings can never be mixed silently.

**Recorder composition (D28).** Verified against discodon's live cassette
work: an action seam wraps tools by name, and a delivery seam (landed
2026-08-04) freezes research payloads — its design record rejects per-query
freezing for character evals. (Since then, scrape grew a request-payload
capture so POST-read APIs can be replayed — 0.23.2 — one more in-family
freezing seam; the multiplication these rules assume is already happening.)
The rules:

- **Outermost active recorder wins.** Under outer replay, inner code never
  runs, so inner recorders are inert by construction. On capture, seams are
  independent and opt-in.
- **Run identity names the active seams.** Discodon already hashes cassette
  mode/version into eval identity; the search-replay flag joins identically.
- **No replay engine in the `TearsTool` base class.** Wrap-by-name at
  existing choke points is the uniform mechanism; hierarchy-attached replay
  is the documented failure this plan exists not to repeat. The one
  tool-level convention evals do need is already ruled: structure on
  `metadata` (SR-A1), which makes an action-seam recording semantically
  complete.

### 3.11 `testing/`

The provider-conformance suite (SR-O5): one parametrized suite every adapter
must pass — contract shape, spend-on-failure, error taxonomy, criterion
disposition honesty, zero-results-is-success — plus parity-declared fakes for
the transport, the store port, and the limiter (`test_fake_protocol_parity`
compliance). A live tier per provider, env-gated: SearXNG against a
self-hosted instance (also settles SR-A4's unverified score-semantics
assumption), Tavily behind explicit credentials.

---

## 4. Changes elsewhere in 3tears

Same repo, same PRs where noted; none of these is optional garnish — each is
load-bearing for a success check.

1. **`media-contracts`**: three facet fields — rights status, pixel
   dimensions, direct-file-versus-containing-page (SR-C3, check 13). Stdlib
   dataclass discipline; the contract-purity pin already enforces the
   package's floor.
2. **`agent-tools` — gut `WebSearchTool`** (check 8): keeps
   `threetears.web_search`, the `TearsTool` ABC, the `ToolResult` shape;
   `execute` becomes async over the leaf (Call + Bind); prose unchanged for
   existing callers; structure on `metadata` under the named key. The 15s
   hardcode, the sync client in `async execute`, and string-prefix errors all
   die here (§10 defects 2, 8).
3. **`agent-tools` — gut `WebFetchTool`**: same identity, Extract-backed;
   streamed + capped + typed; `time.sleep` and unbounded `resp.text` die
   (§10 defects 6, 7). Its `[fetch]` extra forwards to
   `3tears-search[extract]`.
4. **`agent-tools` — `serve.py` wiring**: hosts build the leaf's transport
   from `TracedHttpClient` via a thin adapter (lives here, where core is
   already a hard dep); the skip-with-reason pattern extends to the new
   configuration.
5. **`agent-tools` — context-save node** (C8): fix the name-grain defect
   (match bound names), read structure off `metadata`, and state the retention
   posture in the module docstring *before* wiring it anywhere (§10 defect
   11, as corrected 2026-08-04 — the node is inert today, so this is new
   wiring, not a behavior change).
6. **`agent-tools` — envelope asks** (D18): exception-path metadata carry
   (§10.9); optional per-call deadline on `CallRequest` (§10.10, SR-G2), with
   the server-accepts-first rollout order stated in the PR. *Elicit against
   the 0.23.11 envelope, not 0.23.0's* — it has moved twice since this spec
   was cut: `CallRequest.result_subject` + `CallAccepted` now give long calls
   a durable delivery path (a standing subject that survives connection
   refresh), and `ToolManifestEntry.timeout_seconds` plus server-side
   hard-timeout of runaways (0.23.2) give every tool a declared ceiling. The
   deadline field carries a different quantity — the *caller's remaining
   budget* — and composes with that ceiling: the effective bound is the min
   of the two. `result_subject`'s own rollout is the pattern to copy.
7. **`agent-tools` — `ToolExecutor` keeps the artifact** (audited
   2026-08-04): `executor.py` stringifies tool output and rebuilds
   `ToolMessage` without the artifact — and that is `page_finder`'s actual
   execution path, so **check 4 fails without this fix**. The in-process
   `langchain_adapter` already does it right (`content_and_artifact` →
   `ToolMessage.artifact`); the executor matches it. Additive.
8. **`mcp` — structure on the MCP face**: the result serializer flattens
   everything to `TextContent`; emit spec-sanctioned `structuredContent`
   alongside, and add the missing `metadata` field to the MCP client's
   `McpToolResult`. Both additive. Two adjacent gaps are *named asks
   elsewhere*, not in-repo work:
   - the SDK-side remote wrapper that rebuilds `ToolMessage`s from
     `CallResponse` drops metadata — re-attaching it as the artifact rides
     each consumer's migration;
   - the stream protocol carries no tool structure (`ToolCallEndEvent` is
     name + timing; `Frame.payload` is a string) — that channel is designed
     in the chat-kit workstream (`family-convergence.md` §5, 3tears
     obligations), before metallm's frontend converges.
9. **`scrape`**: `page_finder` reads structure off `metadata`; its callers
   unchanged (check 4, together with item 7). Later, scrape implements
   `HeavyFetcher`.
10. **`enforcement` / `tests/enforcement`**: (a) the D19 norm widening —
   sanctioned-path addition + prose restatement; (b) add `packages/search` to
   `test_dict_state_detection`'s root list; (c) a floor pin for the leaf —
   the `test_contracts_packages_stay_dependency_free` mechanism extended with
   an allowed-floor variant pinning `3tears-search`'s hard deps to exactly
   D24's list; (d) the package's own import-cost/lazy-init test.
11. **`core`**: nothing moves (the aiosqlite removal and the flush.py extras
    refactor are that repo's separate backlog, not this workstream).

---

## 5. Explicit non-goals

MUST NOT, restated from §2 of the requirements doc as build guardrails: crawl
or index; own a ranking implementation; own a telemetry sink; own a replay
store backend; summarise, judge, or conclude; cache responses in v1 (D14);
accept caller-supplied base URLs (D21); import `threetears.core`,
`agent-tools`, langchain, or NATS from the leaf; grow a second result shape
for any face (check 14); let an exception cross the wire (D10).

---

## 6. Testing requirements

- **Conformance** (§3.11) green for both adapters; live tiers env-gated.
- **Wire round-trip**: every contract type JSON round-trips; the metadata
  projection survives `CallResponse` end-to-end (asserted at agent-tools,
  where the envelope lives).
- **Enforcement**: all of §4.8; the leaf passes `test_no_bespoke_reuse`
  **without an exemption** (check 11), `test_no_silent_swallow`,
  `test_uuidv7_enforcement`, fake parity, dependency alignment, and the
  intra-family bounds check on day one.
- **Behavioral pins**: zero-results-is-success; spend-on-failure; budget
  follows the bill (retried-unbilled attempts don't count); replay miss
  raises; per-criterion disposition honesty; degradation marks (unranked is
  known-unranked).
- **The embedded smoke**: a test that runs a full search from a one-shot
  `asyncio.run()` with the standalone transport and no broker (checks 5, 9 in
  spirit; the Pi install-weight check itself runs in samsung's CI, not here).

---

## 7. Sequencing

The cross-repo layer above this section — who starts what, when, across all
five repos plus the eval extraction — is
[`convergence-sequencing.md`](convergence-sequencing.md); this section remains
the detailed in-repo sequence it links into.

Build in this repo first, release, then migrate consumers. Phases are
PR-sized groupings, not calendar units; each lands green through
`./scripts/check-all.sh`. Git discipline throughout is the house rule set:
feature branches into `develop`, stacked PRs where a phase has internal
order, **merge commits only, never squash, never force-push**; release =
version bump → PR into develop → PR develop→main (no second bump) → **push an
annotated tag from main** (the tag push is the trigger; a green run without a
tag is not a release).

### Phase 1 — the leaf (branch `feature/search-leaf`, stacked PRs into develop)

1. **Keystone slice** *(the first chunk of the prawduct build plan; prove the
   architecture before widening)*: `contracts/` core (request, candidate,
   criteria + disposition, spend, errors, transport protocol, metadata key)
   + SearXNG adapter + `call.py` + prose Bind + `standalone.py` + the D19
   norm widening in the same PR + wire round-trip test + conformance
   skeleton. Done when: a live SearXNG query returns typed candidates and
   renders prose, from a one-shot `asyncio.run()`, with enforcement green.
2. Tavily adapter (ported from discodon) + capability declarations + budget
   port + weighted spend + limiter + egress provenance. Done when: both
   adapters pass conformance; the SR-E4 defect is unreproducible.
3. Enforcement deliverables not already landed (floor pin, dict-state root,
   import-cost test) + `media-contracts` facet fields.

**Gate A (architecture checkpoint):** contracts reviewed against SR-A/B/C by
a fresh pass before Phase 2 widens — the contract is the lock-in surface
(every future consumer's queries are its requirements); this is where a re-cut
is still cheap.

#### Gate A findings — 2026-08-10

The independent review returned **pass-with-findings**: the shapes honor
D1/D2/D20/D22/D23, the naming rule, open criteria, honest dispositions, and
Spend's pricing model; one canonical-form mistake had to be re-cut before
Phase 2/3 widens, and no structural re-cut was needed. All findings landed
the same night, while no consumer had bound. Taken:

- **BLOCKING — operational fields left the canonical form.** `record` and
  `budget_scope_tags` participated in the replay/eval digest, which made
  replay structurally unusable (recorded with `record=True`, replayed
  without — SR-F6/SR-F7) and gave every eval run a unique identity
  (SR-D2 tags carry run identity, defeating SR-F1). Fixed via
  `ContractModel.CANONICAL_EXCLUDED`; the semantic parameters — query,
  criteria, fidelity — are the canonical form. `CANONICAL_FORM_VERSION`
  stays 1: no digest had been persisted, which is the point of catching it
  here.
- **`FetchTransport` declared** as a second protocol beside
  `SearchTransport` (§3.1, §3.5, §3.8), so Extract's byte-capped,
  content-type-gated read never forces a widening that would retroactively
  invalidate Phase-1 transport implementers. Implementations arrive with
  Extract in Phase 2.
- **`FailureRecord` carries `egress` and `occurred_at`** (optional,
  additive), stamped by the standalone transport and the SearXNG adapter —
  a consumer-side pacing/ban tracker can now rebuild D8's
  `(provider instance, egress)` key from the one record that survives the
  wire.
- **`Criterion.time_range` refuses naive datetimes and normalizes to UTC**,
  matching the Provenance stance, so equal instants cannot canonicalize
  unequally.
- **D13 skew stance ruled** (see the D13 rider in §1): the additive promise
  is scoped to exact-version pairs while `extra="forbid"` stands; Gate C's
  wire-compatibility promise must include the ignore-unknown flip for
  wire-read payload types. The flip itself is deliberately deferred to that
  gate — strict rejection is the safer default while every reader shares a
  venv.

Dispositions on the build's carried questions, so they stop being open:

- **`ProviderCapabilities` stays in `contracts/`** — capability
  declarations are consumed before an adapter is constructed (SR-B4), which
  makes them contract vocabulary, not adapter internals. No move.
- **`SEARCH_RESULTS_SCHEMA_VERSION` stays 1** despite the additive
  `published_at`/`notices`/`failure` fields — nothing is released, so there
  is no older reader to protect; the version starts counting at first
  release.
- **`SearchProvider` is accepted contract vocabulary** (now named in
  §3.1) — Call's dependency and the conformance suite's parametrization
  axis; it is a seam name, not a layer name, so the §2 naming rule holds.

### Phase 2 — in-family consumers (branches stacked on Phase 1)

4. Extract's web path (streamed, capped, robots stance, no-op on
   provider-supplied content).
5. Gut `WebSearchTool` + `WebFetchTool`; serve.py wiring; metadata key
   end-to-end test over NATS (check 8).
6. `page_finder` structure (check 4); context-save node fix + retention
   posture (§4.5).
7. Envelope asks, as their own PRs with the rollout-order note (D18).

### Phase 3 — pull-driven depth (may start parallel to Phase 2 after Gate A)

8. `replay.py` + `RecordingStore` port + record schema (versioned envelope),
   under D26's durability rules (canonical-request keying, adapter-free
   replay, versioned refusal). The record format is a lock-in decision: its
   consumers' future queries (discodon's eval replay; samsung's re-search)
   are elicited in the build plan before fields are cut.
9. `aggregate.py` (corpus, dedup, producer seam) and `select.py` (criteria
   application, cull, ranker slot, degradation marks).

**Gate B (pre-release):** all §3 success checks that can be verified in-repo
are; SR-A4's SearXNG score semantics confirmed against a live instance;
requirements doc's §13 updated with any vetoes taken during build.

### Phase 4 — release

10. Family minor bump (lockstep — the bounds test names every edge), PR into
    develop, PR develop→main, tag pushed from main. `3tears-search` appears
    on PyPI with the rest of the family.

### Phase 5 — consumer migrations (parallel, per repo; each pins the whole family to the one released version)

**metallm** — precondition: close its family version lag *first*, as its own
change (every adoption assumes a current pin).
1. `git checkout -b feature/new-search` off its default branch.
2. Bump the family pin to the released version — whole family, one exact
   version, per the consumer-pinning rule.
3. Replace the raw SearXNG helper (`admin/models.py`) with leaf Call;
   replace `web_fetch_utils` with Extract; **delete both side-steps** — check
   1 is "deleted, not wrapped".
4. Its frontend/agent callers keep consuming the builtin unchanged; where it
   filtered raw streams for structure, read `metadata` instead.
5. PR, merge per its own workflow.

**discodon** — embedded mode (it is pre-NATS-convergence; check 10 says the
switch later costs no consumer rewrite).
1. Any generalization it needs lands **upstream first** (convergence
   principle 4) — e.g. gaps found while porting its budget semantics onto
   `BudgetPort`.
2. `git checkout -b feature/new-search`; pin the family version.
3. Collapse `tools/web_search_tool.py` and `tools/research/web_search.py`
   onto the leaf: persona path = Call + prose Bind; research path = Call +
   Aggregate + both bindings (prose and corpus).
4. Budget hooks move onto `BudgetPort` — the 2× advanced under-billing dies
   (SR-E4); timeouts get wired to config (SR-G1 defect 3).
5. Search-internal replay lands where the coarse seams cannot reach:
   research-*pipeline* evals (grounding gate and cull re-run against a
   frozen web), through a `RecordingStore` over its existing store.
   Character-eval freezing stays on its action- and delivery-seam cassettes
   (D28), already wired discodon-side.
6. Eval cost caps include search spend via the `BudgetPort` (check 3, under
   D27's execution-spend rule). Recommended pre-work, doable any time while
   discodon is still sole owner of its evals: reshape `EvalRunCostCap` and
   the daily-budget mixin's refusal contract to the port's
   `check(estimate)`/`record(spend)` shape, copied from
   `threetears.search.contracts` rather than invented — the protocol is
   structural, so this gates on nothing shipping.
7. Existing web_search cassettes are keyed on the current wrapper's
   parameter hash; the rebuilt tool reshapes parameters, so this migration
   includes cassette re-capture (or a recorded key mapping) — never silent
   reuse.
8. PR, merge; record acceptance of what binds it (D15).

**samsung** — rides its planned phase-2 image-search work, not a
migration-for-migration's-sake.
1. Branch per its own conventions when that work starts.
2. Take `3tears-search[standalone]` (or supply its own transport), `[extract]`
   if needed; supply a `RecordingStore` over its SQLite plane.
3. Build image search on Call + Select with deep criteria and the new
   `media-contracts` facets; verify checks 2, 5, 9 (no fork, no torch,
   one-shot `asyncio.run()`).
4. Record acceptance (D15).

**Ordering within Phase 5:** metallm first (smallest, exercises the gutted
builtins), discodon second (deepest, exercises budgets + replay + corpus),
samsung when its image work schedules. In-family consumers (scrape,
context-save) already landed in Phase 2.

**Gate C (before the first pod-resident deployment of search):** the D13 wire
compatibility promise ruled formally; envelope asks (D18) released; identity
scope carried for search calls pod-side (the "stateless tool" classification
is invalidated by budgets/replay — §5.4).

---

## 8. Deliberately open

- **OQ1 (Python floor)** — ruled in principle 2026-08-04: discodon adopts
  3.14 (its declared floor is already `>=3.12`, so this is an interpreter
  switch plus verification, owned by discodon). Tracked until that lands;
  D25's avoid-3.14-only-surface intent stays as cheap insurance meanwhile,
  and the per-module-floor fallback is retired unless adoption hits a wall.
- **One bus or two** — decides distributed-pacing reach post-convergence.
- **Final wire-boundary placement** — D16 is the v1 answer; SR-L4 keeps the
  rest open.
- **Model-mediated producer detail** (D3) — designed when samsung pulls.
- **D12 ratification** — the robots/terms stance needs per-repo acceptance,
  not just this spec's proposal.

## 9. Requirements confidence

**High** for Phases 1–2: every requirement is traced to verified code
(2026-08-04), the consumers are known call sites, and the migration path
(metadata under a named key, identity preserved) is confirmed to survive the
wire. **Medium** for Phase 3: SR-F5's "a wiring line per consumer" is
estimated, not measured — the cheapest raise is wiring a `RecordingStore`
over discodon's existing store as a spike before the record schema is cut —
and Aggregate/Select depth depends on samsung's phase-2 requirements holding
as written. **Open assumptions carried:** SR-A4's SearXNG score semantics
(settled at Gate B); the six-layer cut is proposed vocabulary, not ratified
type structure (mitigated by D23's naming rule); OQ1.
