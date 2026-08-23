# Search: The Specification

**Status:** Draft for build planning -- 2026-08-04
**Scope:** the next level down from `search-architecture.md`: decisions taken,
package and module breakdown, MUST/SHOULD/MAY/MUST NOT requirements, and
broad-stroke sequencing for the build and for migrating consuming apps. Not a
build plan -- the build plan derives from this, under prawduct, and lives in the
planning session's local `.prawduct/` (gitignored here by design), so **this
document is the durable record** of what was decided and why.

**Companions** -- read in the order *direction → need → shape → spec*:

| Document | Carries |
|---|---|
| [`family-convergence.md` §4.14](family-convergence.md#414-web-search--one-contract-staged-pipeline-searxng-from-metallm-budgets-from-discodon) | the **direction** |
| [`search-requirements.md`](search-requirements.md) | the **need** -- evidence, requirement IDs (`G*`, `P*`, `SR-*`), success checks |
| [`search-architecture.md`](search-architecture.md) | the **shape** -- six layers, the seams, what each consumer does |
| **this document** | the **spec** -- the buildable statement |

This is the newest of the five search documents and the authority for anything
build-facing. Requirement IDs cited here (`SR-*`, `G*`, `P*`, "check N") are
defined in `search-requirements.md` and are not restated; the builder reads
that document once, then works from this one.

Everything below was written against code verified on 2026-08-04 (post-merge
from develop at 0.23.0), including a correction pass on the three older
documents made the same day. **Re-verified 2026-08-10** against develop at
0.23.11: every build-facing claim held -- the gutting targets, check 4's
consumer, the mcp serializer, and `media-contracts` are all unchanged -- and
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
| D1 (SR-A4) | Named, provenanced scores; **no single `score` field, ever** | Result core carries a set of score entries -- name, value, scale semantics, source (provider or stage), cross-provider comparability flag. A comparable relevance exists only if Select produced one. *Evidence-backed 2026-08-12:* SearXNG's weight is unbounded above (two agreeing engines score 4.0, three 9.0) while Tavily's relevance is `[0,1]`, so one shared field would corrupt a mixed-corpus ranking silently. Select's cull MUST NOT read `score > 0` as "relevant" -- a `priority: low` engine scores everything 0. |
| D2 (SR-A5) | Call returns a candidate set; the corpus is Aggregate's named type | Two types, two dedup/merge stories; Call never accumulates. |
| D3 (SR-B5, OQ21) | Model-mediated search is out of Adapter and Call, in at **Aggregate** as a candidate producer | Provenance carries a `producer` distinction from day one; the producer seam is designed in Phase 3, implemented when samsung pulls. Its token cost is owned by the models usage tracker -- the producer seam records a *reference* to that spend and MUST NOT re-price it into search spend (no double counting). |
| D4 (SR-D4) | Budget follows the bill | The budget increment and the transport retry sit on the same side of the seam: a retried attempt that never billed never counts. C2's fail-closed retry bound moves into the transport's bounded-retry config in the same change. |
| D5 (SR-D5) | Both refusal authorities, distinct roles | Local caps bound a run's *shape* (overrun is a defect); provider refusal bounds *money*. Neither substitutes for the other. |
| D6 (SR-E6) | Self-hosted cost is zero | The rate/quota spend dimensions carry the real constraint (SR-D6); no synthetic infrastructure pricing. |
| D7 (SR-F5) | Replay recordings go through a **consumer-supplied store port** | Embedded: the consumer passes a port object. Pod-resident: the consumer passes a store *reference* the pod resolves to its own implementation. The port follows `media-contracts`' `ObjectStore` shape. |
| D8 (SR-H4, SR-N4) | Pace, don't just react; keyed on `(provider instance, egress)` | Two mechanisms: an in-process limiter shipped in the leaf, and a distributed-limiter port `core`'s `TokenBucket` satisfies where a bus exists. |
| D9 (SR-I4) | Return records, emit nothing | The capability owns no sink; hosts wire `observe` where they have it. |
| D10 (SR-J3) | Typed exceptions carrying spend; prose at Bind; **Bind converts before the wire** | Nothing raises across the NATS hop -- a failed call arrives as a failed `ToolResult` with spend on `metadata`. This is what makes SR-E3 hold pod-resident today despite §10.9. |
| D11 (SR-K2) | Queries are user content | The capability makes the query available for redaction; redaction policy stays with the consumer. |
| D12 (SR-K4) | Family stance, enforced per adapter *(needs cross-repo ratification -- flagged, not silently ruled)* | Proposed stance: provider API calls are governed by provider terms, documented per adapter; Extract's direct fetches honor robots.txt by default, with a per-deployment override that is *recorded config, never code*; retention of recorded content follows the consumer's policy (D7 puts the bytes in the consumer's store, which is what makes that dischargeable). |
| D13 (SR-M1) | In-family versioning is lockstep (already ruled); the **wire payload carries a schema version** | The metadata payload and replay record embed `schema_version`; changes are additive within a family minor. Formal wire-compatibility promise is a **gate before the first pod-resident deployment**, not before first release. *Gate A rider (2026-08-10):* with `extra="forbid"` on every contract type and consumers pinning exact versions, the additive promise is scoped to **exact-version pairs** until that gate; the gate's promise MUST include flipping wire-*read* payload types (`FailureRecord`, `SearchResultsMetadata`, `Candidate` and nested) to ignore-unknown, keeping strict rejection for caller-constructed inputs -- and the metadata reader already checks `schema_version` before structural validation so a version refusal names versions, never surfaces as a pydantic error. |
| D14 (SR-M2) | No response caching in v1 | MUST NOT, beyond whatever a provider does upstream. Revisit after replay ships, in its light -- a cache here has two different legal shapes (SR-O3). |
| D15 (SR-M3, OQ13) | Ratification home is `search-requirements.md` | discodon, metallm, samsung each record acceptance of what binds them, in their own repos, pointing at it. |
| D16 (§5.4) | The v1 wire hop is the existing `TearsTool` envelope, at Bind | No new wire protocol. Every contract type still JSON-round-trips (SR-L4) so a future intra-stack hop stays open -- paid in design discipline, not in v1 machinery. |
| D17 (§5.5) | One tool, one contract, all faces; search stays in the `web` alias; `skill_eligible = False` initially | Image/carrier scoping is a *criteria* parameter of the one tool, not a second tool -- so an agent granted `web` gets exactly what it got before. Samsung's image search is embedded and never enters the tool surface. |
| D18 (§10.9, §10.10) | Both envelope asks are accepted as in-repo work | (a) exception-path metadata carry; (b) an optional per-call deadline field. Sequenced in Phase 2 with an explicit rollout order -- the server must accept the field a release before any client sends it, because `extra="forbid"` on an old server rejects unknown fields. *Re-verified 2026-08-10 (0.23.11):* both asks still open; the envelope meanwhile gained `result_subject` additively -- live precedent for exactly this rollout -- and a manifest-level `timeout_seconds` the deadline field must compose with (§4.6). ***Corrected 2026-08-13, by production rather than by review:*** the rollout order is necessary and was **not sufficient**. Pydantic serializes every *declared* field, so `deadline_seconds` reached the wire as an explicit `null` from the moment it was declared -- "no client sends it yet" was never true on the wire, whatever the callers did. A 0.23.11 pod refused **every** call from a 0.24.1 registry for three days on a field no caller populated. The order only became true in 0.24.3, when the registry proxy started pruning unset top-level optionals; see §7 Phase 2 item 7. |
| D19 (SR-N1) | The no-bespoke-client norm **widens**; no exemption is filed | Verified mechanism: `_SANCTIONED_HTTPX_SITES` is a path frozenset, and the walker only flags raw httpx clients stored on `self` -- a protocol-typed transport field never trips it. The widening = add the leaf's standalone-transport module path to the sanctioned set + restate the norm prose. Lands in the same PR as that module (check 11). |
| D20 (SR-N2) | Egress is per-upstream input at Adapter and provenance on every result; `direct` is a named value | Rate/ban budgets key on it (D8); replay comparability depends on it. |
| D21 (SR-K3, SR-N3) | The SSRF ruling binds at the transport seam | Provider base URLs come from deployment config only -- MUST NOT accept a caller-supplied base URL. Redirect policy and private-address guards live in the transport implementations, not per call site. |
| D22 (§10.12) | Structured results ride `ToolResult.metadata` under a named key | `SEARCH_RESULTS_METADATA_KEY = "search_results"`, following the `OBJECT_HANDLE_METADATA_KEY` precedent, defined in the leaf's contracts. |
| D23 (packaging) | **One package, `3tears-search`**, import root `threetears.search`; contracts as an import-clean module, not a separate package | See §2. The alternative (a separate `3tears-search-contracts`, the eval precedent) is not taken because the whole package already sits at the contracts-leaf floor; import paths are chosen so a later split is a non-breaking move (the OQ3 discipline). Split trigger: a consumer that needs the types but must refuse even `observe` + `media-contracts` + pydantic -- none exists or is foreseen. |
| D24 (leaf floor) | Hard deps: `pydantic`, `3tears-media-contracts`, `3tears-observe`. Extras: `[standalone]` = httpx (the bare transport impl), `[extract]` = trafilatura | Matches SR-L7's permitted floor exactly. Provider adapters ship in the base package -- they are pure logic over the injected transport and weigh nothing; extras carry *weight*, and the only weights are httpx (only for hosts that don't inject their own transport) and trafilatura. |
| D25 (Python floor) | The leaf declares `requires-python = ">=3.14"` today, avoids gratuitous 3.14-only surface | The workspace is 3.14. **OQ1 ruled in principle 2026-08-04: discodon adopts 3.14** -- its declared floor is already `>=3.12`, so this is an interpreter switch plus verification, owned by discodon. The avoid-3.14-only-surface intent stays as cheap insurance until that lands; the per-module-floor fallback is retired unless adoption hits a wall. |
| D26 (replay durability, 2026-08-04) | Recordings outlive the stack that made them: replay records the **typed result** and keys on the **canonical caller request** | Three rules, detailed in §3.10: the key hashes explicitly-set caller parameters (never resolved defaults) plus a key-derivation version; replay short-circuits at the Call boundary and never touches an adapter, so removing a provider strands no recordings; payload readability is promised within a family major, refused loudly across, matching the cascade-delete lifetime recordings actually have (SR-F5). |
| D27 (replay spend, 2026-08-04) | Replay reports **both** spends, never one field: the recording's original spend rides inside the replayed payload; the replay's own execution spend rides where spend always rides | Budgets bind on execution spend; cost-model analyses read recorded spend, so a replayed baseline never looks free. P7 applied to spend. Detail in §3.10. |
| D28 (recorder composition, 2026-08-04) | Multiple freezing seams coexist under one rule: **the outermost active recorder wins** | Search replay is the innermost seam and the only one reaching non-Tool callers; character/agent evals freeze coarser (discodon's action- and delivery-seam cassettes), correctly. **No replay engine enters the `TearsTool` base class.** Detail in §3.10. |
| D29 (freeze window, 2026-08-11) | Publication does not freeze the contracts. They stay **re-cuttable until the first consumer release pins a version carrying search** | 0.24.0 shipped `3tears-search` to PyPI after Phase 1, not at Phase 4 (§7). What makes a re-cut cheap was never "unpublished" -- it is that nobody has bound: no consumer pins it, and exact-version family pinning (the D13 rider) means a changed shape cannot reach an installed reader. So Phases 2-3 may still re-cut contract types. The window closes at the first consumer *release* naming a version that carries search; past that, a change to a wire-read payload type is a compatibility event, not an edit. Two counters therefore begin at 0.24.0 rather than "first release": `SEARCH_RESULTS_SCHEMA_VERSION` and `CANONICAL_FORM_VERSION` both stay 1 while changes remain additive, and any non-additive change now MUST be spelled as a bump rather than absorbed. **The consumers this window is measured against are ours** -- metallm and discodon, the two the sequencing tracks. A third party can `pip install 3tears-search==0.24.0` today and is outside that definition entirely; what covers them is the alpha policy the root README states, that the public API may shift between minor versions until 1.0.0, and a re-cut here lands in 0.25.0. That is the whole of their protection and it is deliberate: a package published three phases before its consumers exist has no installed base to protect, and pretending otherwise would freeze a contract nobody is holding. Past 1.0.0 this row does not apply -- the freeze is then whatever semver promises, and "nobody has bound" stops being knowable. |
| D30 (SR-M4, 2026-08-12) | The fetch path carries **caller-supplied validators** and reports *not modified*; D14 is untouched | D14 forbids the capability **holding** a response; a conditional request holds nothing -- the caller's `If-None-Match` / `If-Modified-Since` go out as request headers and a `304` comes back. Same carve-out the robots memo already took, with less to argue since nothing survives even the call. Ruled REQUIRED rather than deferred behind replay because D7/D12 put the bytes in the consumer's store and then left it no way to spend the validator they arrived with -- an incoherence, not a deferral. **Fetch path only** (Extract's carrier read and re-reads of a known URL); conditionalising a provider *query* is worthless and mis-scopes the work. Additive and opt-in, so it binds no consumer and needs no D15 ratification. The transport seam already suffices -- `FetchTransport.fetch` takes `headers`, `TransportResponse` carries `status_code` + `headers`, and a `304` verifiably survives `StandaloneTransport` today. Build sequence in `search-task-01-conditional-revalidation.md`. |

Two §13 rows are *not* ruled here because nothing in Phases 1-4 needs them:
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
  contracts/        # the leaf within the leaf -- types, protocols, errors, keys
  adapters/
    searxng.py      # Adapter: SearXNG
    tavily.py       # Adapter: Tavily (ported from discodon -- extract, don't invent)
  call.py           # Call
  aggregate.py      # Aggregate  (Phase 3)
  extract.py        # Extract    (web path Phase 2; carrier dispatch Phase 3)
  select.py         # Select     (Phase 3)
  bind.py           # Bind helpers: prose render + metadata projection
  standalone.py     # bare-httpx transport impl   [standalone] -- the sanctioned path (D19)
  limiter.py        # in-process pacing (D8; the port is contract vocabulary, §3.1)
  replay.py         # record/replay over the store port (Phase 3)
  testing/          # provider-conformance suite + parity-declared fakes (SR-O5)
```

Layer names (Adapter, Call, Aggregate, Extract, Select, Bind) are **module
vocabulary, not type names** -- the requirements doc's own warning (§12): the
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
- MUST have every contract type wire-serialisable -- JSON round-trip with no
  callables, open files, or port objects in any result/record type (SR-L4).
  Port objects are *parameters*, never *payload*.
- MUST be usable from a one-shot `asyncio.run()` -- no ambient loop, no
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

- **`SearchRequest`** -- query text, criteria, requested fidelity, opt-in
  record flag, budget scope tags. MUST treat query as user content (D11).
- **Criteria** -- one open vocabulary (P6, SR-B1): well-known criteria ship as
  typed constructors (time range, domains include/exclude, language, carrier,
  min resolution, rights class, …), unknown criteria as namespaced keys. MUST
  NOT be a closed enum. The response MUST carry a per-criterion disposition --
  `pushdown | local | unsatisfied | ignored-unknown` (SR-B2, SR-B3, P8); an
  unsatisfiable criterion is named, never dropped.
- **`Candidate`** -- the carrier-neutral result core (SR-C1): identity,
  locator(s), provenance, scores, fidelity available/achieved, an optional
  content slot recording whether content arrived with the response or from a
  later fetch (SR-A2), and **facets** -- additive, keyed by the
  `media-contracts` vocabulary, ignorable by consumers that don't recognise
  them (SR-C2, SR-C3). MUST NOT define a closed carrier union.
- **Provenance** (on every candidate and every spend/replay record, P2,
  SR-A3): query, provider instance, provider-native identifiers, retrieval
  time, **egress name with `direct` as a value** (D20), and producer class
  (API provider now; model-mediated later, D3).
- **Scores** -- per D1. MUST mark provider-native scores non-comparable across
  providers.
- **`Corpus`** -- Aggregate's accumulation type with a stated dedup key and
  merge rule (D2, SR-A5).
- **`Spend`** -- every resource a call consumed (SR-E1): money (Decimal),
  wall-clock, call count, weighted provider units (SR-E4), bytes. MUST survive
  the failure path (SR-E3); the count a cap enforces and the count a bill
  prices MUST be the same number (SR-E2); per-request (not per-result) pricing
  must be representable (SR-E5).
- **Typed errors** -- the seven distinguishable failure classes of SR-J1, each
  carrying `Spend`; remediation text where the cause is known (the SearXNG
  403-json-formats teaching error). Zero results is a success value, not an
  error (SR-J2). The wire record carries provenance enough to rebuild D8's
  pacing key consumer-side -- provider instance, egress, occurrence time --
  because pod-resident it is the only fact that survives the wire (D10, P2;
  Gate A, 2026-08-10).
- **Protocols** (structural, injected -- P9): `SearchTransport` (shaped so
  `core.http_client.TracedHttpClient` satisfies it via a thin host-side
  adapter: configurable timeout, bounded retry, circuit-breaking, per-call
  span, egress selection -- SR-N1, SR-G1, SR-G4, SR-D3), `FetchTransport`
  (the streamed, byte-capped, content-type-gated read Extract requires --
  declared as a *second* protocol at Gate A, 2026-08-10, so Phase-1
  `SearchTransport` implementers are never retroactively non-conformant.
  *Corrected 2026-08-11 (Phase 2 item 5):* Gate A expected the standalone
  transport **and the host adapter** to implement the union. Only the
  standalone one does, and cannot be otherwise -- `TracedHttpClient` is
  constructed per upstream and buffers whole bodies, while a fetch is an
  arbitrary candidate URL read under a per-call cap. A host injects the
  traced adapter for search and the standalone transport for fetch; the
  ruling is in §7 Phase 2), `SearchProvider` (the provider seam Call consumes and the
  conformance suite parametrizes over -- named here at Gate A; it is the
  one seam-vocabulary addition §3.1's original field list did not carry),
  `BudgetPort` (`check(estimate)` / `record(spend)` with plural scopes --
  SR-D1, SR-D2), `RateLimiterPort` (D8), `RecordingStore` (D7,
  `ObjectStore`-shaped, streaming), `HeavyFetcher` (implemented by
  `3tears-scrape`, never imported).
- **Replay record** -- typed envelope (id -- UUIDv7-compatible, created-at,
  provider, key, size, `schema_version`) over a payload that can rebuild the
  corpus (SR-F4); the key is derived by search (SR-F8).
- **Canonical serialization is a public contract feature, not a replay
  internal.** One canonical form, two consumers that must agree: the D26
  replay key and eval run identity (SR-F1 -- search parameters already
  participate in discodon's `canonical_digest`). MUST be exposed on the
  request/parameter types. Only the *semantic* parameters participate --
  query, criteria, fidelity; the operational fields (`record`,
  `budget_scope_tags`) MUST NOT enter the canonical form: a recording is
  made with `record=True` by definition (SR-F6) and replayed without it
  (SR-F7), and scope tags carry per-run identity (SR-D2), so keying either
  strands recordings and gives every eval run a unique digest *(Gate A
  finding, 2026-08-10)*.
- **`SEARCH_RESULTS_METADATA_KEY`** and the metadata projection schema, with
  `schema_version` (D13, D22).

MUST version additively within a family minor (D13). SHOULD keep every type
constructible with defaults-off -- no hidden globals.

### 3.2 `adapters/` -- SearXNG, Tavily

One provider's API each, through the injected transport only. Each adapter:

- MUST declare capabilities queryably (SR-B4), following the
  `3tears-models` capability-metadata pattern -- SearXNG: categories, engines,
  language, safesearch, paging, time range; Tavily: depth, domains, topic,
  dates.
- MUST keep everything the provider returns that P2 protects -- scores, engine
  attribution, published dates -- in typed form, not a disclaimed `raw` blob.
- MUST attach `Spend` to every call including failures; Tavily MUST weight
  units correctly (`advanced` = 2 credits -- the SR-E4 live defect must be
  impossible to reproduce here).
- MUST map provider failures onto the typed error taxonomy (SR-J1, SR-D3 --
  quota exhaustion short-circuits distinctly from a local cap).
- MUST take base URL and credentials from the host (D21, SR-K1); MUST NOT
  default them from env.
- SHOULD implement pushdown for every criterion the provider can express and
  report `local`/`unsatisfied` for the rest (§7 of the requirements doc).
- Tavily MUST be ported from discodon's wrapper (principle: extract, don't
  invent), preserving its hard-won semantics -- depth/credit coupling, domain
  scoping, score coercion, absolute-dates-beat-time_range (SR-B3's RES-T4M9
  precedent).

Conformance: both pass the shared suite in `testing/` (§6).

### 3.3 `call.py`

A query → one candidate set, through one adapter. Owns criteria negotiation
with the adapter's declared capabilities, failure mapping, spend attachment,
budget consultation (D4, D5), pacing (D8), and replay record/replay hooks
(Phase 3). MUST be the layer where "budget follows the bill" is enforced --
below the retry boundary, so retried-but-unbilled attempts don't count (D4).
MUST apply safe default bounds when the caller tunes nothing (SR-L6).

### 3.4 `aggregate.py` *(Phase 3)*

Many calls → one set. Owns the dedup key, the merge rule, fan-out accounting
(SR-H2: within-batch and cross-run bounds; SR-H3: one failure never poisons
siblings), and the `Corpus` type. MUST accept candidates from an external
producer (D3's model-mediated seam) without them impersonating a provider --
provenance keeps the classes distinct. MAY implement reciprocal-rank fusion
across engines/providers (prior art: `Lombey/Local-Web-Search-MCP`); MUST NOT
require it.

### 3.5 `extract.py` *(web path Phase 2; carrier dispatch Phase 3)*

A carrier → the information in it. Carrier-dispatched (SR-C4); a consumer MUST
be able to take search with no extraction at all. Requirements:

- MUST no-op (and cost nothing) when the provider already supplied content
  (SR-A2 -- the Tavily case).
- MUST stream with a byte cap and a content-type gate; MUST NOT hold an
  unbounded `resp.text` (SR-G5). This is the acute `MemoryMax` case. The
  seam that carries this is `FetchTransport` (§3.1) -- Extract's fetches go
  through it, never through `SearchTransport.request`, whose fully-buffered
  response shape cannot express the cap (Gate A, 2026-08-10).
- MUST honor the D12 robots stance; the enforcement point is here and in the
  transports, not per call site.
- MUST record extraction method and status on the result (fidelity achieved,
  SR-B6), using `media-contracts`' `extraction_status` vocabulary.
- Escalation to hostile targets goes through the `HeavyFetcher` protocol slot;
  `3tears-scrape` implements it; this package MUST NOT import scrape. SHOULD
  make escalation explicit (a caller choice), not automatic -- silent
  escalation multiplies cost (shared_search OQ3, resolved conservative).
- MAY add a Wayback fallback tier later (prior art: `TadMSTR/searxng-mcp`);
  not in v1.

#### Rulings taken before the build -- 2026-08-11

Recorded ahead of the build rather than after it, per the Gate A precedent
and for the reason that precedent exists: a ruling that lives only in the
session that took it is a ruling the next session re-litigates. Each is
vetoable; a veto lands here and in `search-requirements.md` §13.

- **Robots is fetched through the same `FetchTransport`**, parsed with
  stdlib `urllib.robotparser`, and memoized for the life of one Extract
  call and no longer. Fetching it through the injected seam means it
  inherits the guards, the byte cap, the pacing and the egress provenance
  rather than acquiring a second, weaker path to the same hosts. It costs
  one extra fetch per host per call; the alternative is honoring robots
  without reading it, which is not honoring it. **This is not a response
  cache and D14 is untouched** -- D14 governs *search responses*, and the
  memo dies with the call rather than outliving it.
- **A missing `[extract]` extra is a typed refusal naming the extra**, never
  a silently degraded extraction. A caller handed prose has to be able to
  tell whether a real extractor produced it; a crude tag-strip fallback
  would be indistinguishable at the call site and wrong in a way that only
  shows up in a model's output.
- **The `extraction_status` vocabulary gets named constants in
  `media-contracts` first.** Today it is a bare `str | None` with a comment
  listing `"pending"` / `"complete"` -- no constants, and no value for
  *refused* (robots said no) or *failed* (the fetch died), both of which
  Extract produces. Additive constants land beside the facet vocabulary,
  in the package that owns the neighbourhood, rather than a second status
  vocabulary being invented in the search leaf (SR-C3's rule applied to
  status rather than to facets).
- **The shipped DDL is that vocabulary's canonical statement, not the
  contract's comment** -- the two already disagree, and the constants have
  to land on one side of it. `MediaInfo.extraction_status` documents
  `"pending"` / `"complete"` / `None`; migration v021 declares the column
  `TEXT NOT NULL DEFAULT 'none'` and names `'none'` / `'pending'` /
  `'complete'` / `'failed'`, and v022 builds a partial index
  `WHERE extraction_status = 'pending'`. The DDL wins because it is the
  side with rows in it: a spelling in a column default and an index
  predicate is changed by a migration and an index rebuild, not by an
  edit. Two consequences. **`failed` already exists** -- of the two values
  the ruling above says Extract produces, only `refused` is new, so the
  constants are `none` / `pending` / `complete` / `failed` / `refused`.
  And **the field stays `str | None`** -- no `Literal`, no `StrEnum`.
  Narrowing it would break consumers that assign a bare `str`
  (`analyze_media.py` compares against these values today) and would force
  a ruling on the dataclass default `None` versus the column default
  `'none'`, which is a data question wearing a typing costume. That split
  is **recorded, not fixed**: both spellings mean *no extraction
  attempted*, every consumer today falls through both branches
  identically, and reconciling them is a migration, not a side effect of
  adding a string constant.
- **Escalation to `HeavyFetcher` is a caller choice**, never automatic --
  the conservative resolution `shared_search` OQ3 already reached, restated
  here because "SHOULD" above left it open and silent escalation multiplies
  cost by a factor nobody sees until the bill.

**Not Extract's to open: the transport's connection scope.** The
`StandaloneTransport` gained `connection_scope()` with the fetch seam (§3.8),
and Extract deliberately opens none of its own: a transport is an injected
port, and how long its connections live belongs to the deployment that
constructed it. A host that wants pooling wraps its own work in the scope.

#### Built 2026-08-11 -- what the web path ruled

The five rulings above held; four more were forced by writing it, recorded
here for the same reason.

- **Per-candidate outcomes are recorded on the candidate, not raised.** A
  404, a cap refusal, a robots file saying no: each returns a candidate
  whose `extraction_status` facet says what happened. One unreadable page
  must never take down the extraction of a set -- SR-H3's rule for
  fan-out siblings, applied to carriers. The single exception is the
  missing `[extract]` extra, which raises because it would refuse *every*
  candidate identically, and marking a hundred `refused` one at a time
  hides one fixable fault behind a hundred plausible ones.
- **The missing extra raises `LocalCapExceeded` with scope
  `extractor-unavailable`.** It is a local refusal the provider never saw,
  which is that class's definition, and `scope` already doubles as refusal
  identity here (`query-length`, `response-bytes`, `content-type`). **This
  is the ruling most worth a veto**: the honest alternative is an eighth
  taxonomy class, `CapabilityUnavailable`, and it was not taken mid-build
  because a taxonomy change should be somebody's deliberate decision
  rather than a side effect of writing this module. D29's window makes the
  re-cut cheap for now.
- **Robots follows RFC 9309 on its own failures.** A 4xx for `robots.txt`
  means no rules exist and the fetch proceeds; a 5xx or a transport failure
  means the rules are unknown, and unknown rules are honored as *deny*.
  Reading "allowed" out of a server error is how a polite fetcher turns
  impolite for exactly as long as the origin is having a bad day.
- **Escalation is a parameter, not a fallback.** Passing `heavy_fetcher`
  says *use it for this candidate*; Extract never reaches for it after an
  ordinary fetch fails, so "a caller choice" means the caller wrote the
  policy. `HeavyFetcher` is declared beside `FetchTransport` in §3.1 with
  a deliberately distinct method name (`fetch_rendered`), so an ordinary
  transport cannot satisfy it by accident and be used as one.

Two smaller dispositions, so they stop being open: the robots memo is
**not** offered as a batch-scoped parameter -- a memo outliving one call is
a wider scope than the ruling sanctions, and it belongs with Phase 3's
carrier dispatch where something owns the set; and `extraction_method` is
recorded under a **search-owned** facet key, because `media-contracts`
names the status vocabulary but not the method one, and promoting the key
belongs with the second producer that needs it rather than the first.

### 3.6 `select.py` *(Phase 3)*

Candidates + criteria → an ordered, filtered subset. Owns local criteria
application and the cull; exposes a **ranker slot** and never a ranking
implementation (§4.14's ruling -- MMR lives in `agent-memory`, rerank metadata
in `3tears-models`, a cross-encoder arrives as a models provider). MUST mark
unranked output as unranked (SR-L2, P8). MUST satisfy P4's acceptance test: a
consumer supplying its own ranker can still constrain carrier type; a
consumer wanting the cull pays for no reranker.

### 3.7 `bind.py`

Candidates → what the caller consumes. Two bindings ship: prose-for-a-model
(the LLM rendering, migrated from `_format_results` but structure-preserving
underneath) and the metadata projection under `SEARCH_RESULTS_METADATA_KEY`
(D22, explicit border projection à la `ObjectHandle.to_metadata`). MUST catch
every typed exception and render a failed result carrying spend -- nothing
raises across the wire (D10). MUST NOT import `agent-tools` -- the `TearsTool`
gutting consumes these helpers, not the reverse. One binding path serves all
three faces (check 14); a face-specific response shape is a regression by
definition.

### 3.8 `standalone.py` -- `[standalone]`

The bare-httpx `SearchTransport` implementation for hosts without core
(samsung; any embedded consumer). Carries the same obligations the injected
core transport gives for free: configurable timeout, bounded retry with
backoff, per-attempt accounting visible to spend, SSRF guards
(private-address and redirect policy per D21), streamed reads with caps. This
module's path is the one added to `_SANCTIONED_HTTPX_SITES` (D19). MUST NOT
be imported by anything else in the package at module level -- it is an
implementation a host chooses. When Extract lands (Phase 2) this module
implements `FetchTransport` alongside `SearchTransport` -- the union is the
declared shape (Gate A, 2026-08-10), and its per-request-client lifecycle is
revisited in the same change (right for one search, wrong for Extract's
many-fetch path).

### 3.9 `limiter.py`

In-process token-bucket pacing keyed `(provider instance, egress)` (D8, D20).
The port a distributed implementation satisfies (`core`'s NATS `TokenBucket`,
host-injected, where a bus exists) is contract vocabulary and lives in
`contracts/` per §3.1's protocol list -- `threetears.search.contracts.limiter`
declares the shape, this module ships the in-process implementation, and the
two module names mirror each other on purpose *(placement settled at build,
2026-08-11, following §3.1 where the two sections disagreed)*. The
in-process limiter's state is the argued SR-O2 allowlist entry -- the
argument is written into the enforcement allowlist entry itself. MUST be on
by default with safe rates (SR-L6); the shared SearXNG's own server-side
limiter remains the backstop that covers non-cooperating deployments
(SR-H4's honest layering). *Build ruling (2026-08-11):* "on by default" is
discharged by the implementation's own no-argument construction -- 1
token/second, burst 3, chosen so too-slow is latency a host can tune away at
construction while too-fast is a shared-instance ban nobody can tune away
after the fact. Call mints no implicit instance when none is injected: a
per-call limiter paces nothing (buckets unshared), and a hidden process-wide
one is exactly what the ports discipline (P9) and the SR-O2 stance forbid.
The host constructs one limiter per process and passes it; the Phase-2 pod
wiring is the construction site that owns the default.

### 3.10 `replay.py` *(Phase 3)*

Record/replay attached at Adapter/Call (SR-F3), writing through the
consumer's `RecordingStore` (D7). Opt-in per call (SR-F6); a replay miss is a
typed error, never a silent live call (SR-F7); recordings rebuild the corpus,
not just rendered text (SR-F4); ids are UUIDv7 (SR-O4), generated by the
writer. Retention, purge, and redaction belong to the store's owner (D7,
D12).

**Durability against stack evolution (D26).** The replay key is an opaque
digest used for equality lookup only -- nobody ever parses it -- so the risks
are derivation drift and payload readability, and each gets a rule:

- **Canonical-request keying.** MUST derive the key from the caller's request
  in canonical form -- explicitly-set *semantic* parameters only (operational
  fields like the record flag and budget scope tags never participate; Gate A,
  2026-08-10), stably serialized, with absent and defaulted canonically
  identical -- plus provider-instance identity and profile digest (SR-F8);
  MUST NOT derive it from the resolved provider wire request. Adding a
  parameter with a default therefore shifts no existing key. The record envelope MUST carry a key-derivation version;
  a genuinely incompatible derivation change bumps it, and the resulting
  miss names both versions instead of being mysterious.
- **Adapter-free replay.** The recorded payload is the contract-shaped typed
  result (candidates, scores, dispositions, spend) per SR-F4, so replay MUST
  short-circuit at the Call boundary -- deserialize and return -- and MUST NOT
  require the recording's provider adapter to be installed. Removing a
  provider strands no recordings; provenance keeps naming it as historical
  fact. Layers above the frozen exchange run live, which is the point: a
  replayed eval measures pipeline changes against frozen web input.
- **Versioned refusal.** Payload readability follows D13 -- `schema_version`,
  additive within a family minor -- and is promised **within a family major**;
  a reader meeting a version it cannot read MUST refuse with a typed error
  naming both versions, never best-effort misread. The promise is scoped to
  the lifetime recordings actually have (cascade-delete with the owning run,
  SR-F5) -- this is not an archival format and MUST NOT be priced as one.

**Dual spend on replay (D27).** A replayed result MUST carry the original
call's spend inside the replayed payload -- it is part of what SR-F4
preserves -- AND the replay's own execution spend in the ordinary place, and
MUST NOT merge them. Budget ports are consulted with execution spend only (a
replay debits no provider quota; wall-clock bounds still bind); cost-model
analyses read the recorded spend. Provenance MUST mark the result replayed so
the two readings can never be mixed silently.

**Recorder composition (D28).** Verified against discodon's live cassette
work: an action seam wraps tools by name, and a delivery seam (landed
2026-08-04) freezes research payloads -- its design record rejects per-query
freezing for character evals. (Since then, scrape grew a request-payload
capture so POST-read APIs can be replayed -- 0.23.2 -- one more in-family
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
must pass -- contract shape, spend-on-failure, error taxonomy, criterion
disposition honesty, zero-results-is-success -- plus parity-declared fakes for
the transport, the store port, and the limiter (`test_fake_protocol_parity`
compliance). A live tier per provider, env-gated: SearXNG against a
self-hosted instance (which settled SR-A4's score-semantics assumption on
2026-08-12), Tavily behind explicit credentials. The SearXNG live test
tolerates zero results, because zero results is a success (SR-J2) -- so it
takes `SEARXNG_REQUIRE_RESULTS=1` to make an empty run fail, without which
every per-candidate assertion sits in a loop that ran zero times.

---

## 4. Changes elsewhere in 3tears

Same repo, same PRs where noted; none of these is optional garnish -- each is
load-bearing for a success check.

1. **`media-contracts`**: three facet fields -- rights status, pixel
   dimensions, direct-file-versus-containing-page (SR-C3, check 13). Stdlib
   dataclass discipline; the contract-purity pin already enforces the
   package's floor.
2. **`agent-tools` -- gut `WebSearchTool`** (check 8): keeps
   `threetears.web_search`, the `TearsTool` ABC, the `ToolResult` shape;
   `execute` becomes async over the leaf (Call + Bind); prose unchanged for
   existing callers; structure on `metadata` under the named key. The 15s
   hardcode, the sync client in `async execute`, and string-prefix errors all
   die here (§10 defects 2, 8).
3. **`agent-tools` -- gut `WebFetchTool`**: same identity, Extract-backed;
   streamed + capped + typed; `time.sleep` and unbounded `resp.text` die
   (§10 defects 6, 7). Its `[fetch]` extra forwards to
   `3tears-search[extract]`.
4. **`agent-tools` -- `serve.py` wiring**: hosts build the leaf's transport
   from `TracedHttpClient` via a thin adapter (lives here, where core is
   already a hard dep); the skip-with-reason pattern extends to the new
   configuration. *As built (item 5):* the adapter serves **search** only --
   the fetch half is `StandaloneTransport`, for the structural reason ruled
   in §7 Phase 2 -- and the skip-with-reason had to become a probe rather
   than a caught `ImportError`, because Extract imports its extractor lazily.
5. **`agent-tools` -- context-save node** (C8): fix the name-grain defect
   (match bound names), read structure off `metadata`, and state the retention
   posture in the module docstring *before* wiring it anywhere (§10 defect
   11, as corrected 2026-08-04 -- the node is inert today, so this is new
   wiring, not a behavior change).
6. **`agent-tools` -- envelope asks** (D18): exception-path metadata carry
   (§10.9); optional per-call deadline on `CallRequest` (§10.10, SR-G2), with
   the server-accepts-first rollout order stated in the PR. *Elicit against
   the 0.23.11 envelope, not 0.23.0's* -- it has moved twice since this spec
   was cut: `CallRequest.result_subject` + `CallAccepted` now give long calls
   a durable delivery path (a standing subject that survives connection
   refresh), and `ToolManifestEntry.timeout_seconds` plus server-side
   hard-timeout of runaways (0.23.2) give every tool a declared ceiling. The
   deadline field carries a different quantity -- the *caller's remaining
   budget* -- and composes with that ceiling: the effective bound is the min
   of the two. `result_subject`'s own rollout is the pattern to copy.
7. **`agent-tools` -- `ToolExecutor` keeps the artifact** (audited
   2026-08-04): `executor.py` stringifies tool output and rebuilds
   `ToolMessage` without the artifact -- and that is `page_finder`'s actual
   execution path, so **check 4 fails without this fix**. The in-process
   `langchain_adapter` already does it right (`content_and_artifact` →
   `ToolMessage.artifact`); the executor matches it. Additive.
8. **`mcp` -- structure on the MCP face**: the result serializer flattens
   everything to `TextContent`; emit spec-sanctioned `structuredContent`
   alongside, and add the missing `metadata` field to the MCP client's
   `McpToolResult`. Both additive. Two adjacent gaps are *named asks
   elsewhere*, not in-repo work:
   - the SDK-side remote wrapper that rebuilds `ToolMessage`s from
     `CallResponse` drops metadata -- re-attaching it as the artifact rides
     each consumer's migration;
   - the stream protocol carries no tool structure (`ToolCallEndEvent` is
     name + timing; `Frame.payload` is a string) -- that channel is designed
     in the chat-kit workstream (`family-convergence.md` §5, 3tears
     obligations), before metallm's frontend converges.
9. **`scrape`**: `page_finder` reads structure off `metadata`; its callers
   unchanged (check 4, together with item 7). Later, scrape implements
   `HeavyFetcher`.
10. **`enforcement` / `tests/enforcement`**: (a) the D19 norm widening --
   sanctioned-path addition + prose restatement; (b) add `packages/search` to
   `test_dict_state_detection`'s root list; (c) a floor pin for the leaf --
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

The cross-repo layer above this section -- who starts what, when, across all
five repos plus the eval extraction -- is
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

### Phase 1 -- the leaf (branch `feature/search-leaf`, stacked PRs into develop)

**Landed 2026-08-11 in [#303](https://github.com/pacepace/3tears/pull/303)**
-- all three items plus the Gate A findings, which the branch absorbed the
same night the review returned them.

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
   import-cost test) + `media-contracts` facet fields. *(Landed with the
   keystone commits, before item 2 -- the facet fields ship in
   `media-contracts`' `facets.py`, the three tests in their §6 homes.)*

**Gate A (architecture checkpoint):** contracts reviewed against SR-A/B/C by
a fresh pass before Phase 2 widens -- the contract is the lock-in surface
(every future consumer's queries are its requirements); this is where a re-cut
is still cheap.

#### Gate A findings -- 2026-08-10

The independent review returned **pass-with-findings**: the shapes honor
D1/D2/D20/D22/D23, the naming rule, open criteria, honest dispositions, and
Spend's pricing model; one canonical-form mistake had to be re-cut before
Phase 2/3 widens, and no structural re-cut was needed. All findings landed
the same night, while no consumer had bound. Taken:

- **BLOCKING -- operational fields left the canonical form.** `record` and
  `budget_scope_tags` participated in the replay/eval digest, which made
  replay structurally unusable (recorded with `record=True`, replayed
  without -- SR-F6/SR-F7) and gave every eval run a unique identity
  (SR-D2 tags carry run identity, defeating SR-F1). Fixed via
  `ContractModel.CANONICAL_EXCLUDED`; the semantic parameters -- query,
  criteria, fidelity -- are the canonical form. `CANONICAL_FORM_VERSION`
  stays 1: no digest had been persisted, which is the point of catching it
  here.
- **`FetchTransport` declared** as a second protocol beside
  `SearchTransport` (§3.1, §3.5, §3.8), so Extract's byte-capped,
  content-type-gated read never forces a widening that would retroactively
  invalidate Phase-1 transport implementers. Implementations arrive with
  Extract in Phase 2.
- **`FailureRecord` carries `egress` and `occurred_at`** (optional,
  additive), stamped by the standalone transport and the SearXNG adapter --
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
  gate -- strict rejection is the safer default while every reader shares a
  venv.

Dispositions on the build's carried questions, so they stop being open:

- **`ProviderCapabilities` stays in `contracts/`** -- capability
  declarations are consumed before an adapter is constructed (SR-B4), which
  makes them contract vocabulary, not adapter internals. No move.
- **`SEARCH_RESULTS_SCHEMA_VERSION` stays 1** despite the additive
  `published_at`/`notices`/`failure` fields -- nothing is released, so there
  is no older reader to protect; the version starts counting at first
  release. *Superseded 2026-08-11:* first release happened early (0.24.0,
  §7 Phase 4), so the counting has started. What the disposition was
  actually resting on -- no bound reader -- still holds and is now ruled
  explicitly as **D29**, which is what governs a re-cut from here.
- **`SearchProvider` is accepted contract vocabulary** (now named in
  §3.1) -- Call's dependency and the conformance suite's parametrization
  axis; it is a seam name, not a layer name, so the §2 naming rule holds.

#### Phase 1 item 2 -- built 2026-08-11

Both done-when conditions hold: both adapters pass the shared conformance
suite, and the SR-E4 defect is pinned unreproducible at three levels -- the
adapter's own coupling (`_Plan.set_depth` moves the wire parameter and the
billed weight as one operation; an unknown depth bills the *highest* known
weight), a generic conformance pin read off the provider's pricing
declaration, and the env-gated live tier. Rulings taken during the build,
recorded here per the Gate A precedent:

- **Budget refusal is a returned decision, not a raise.** `BudgetPort.check`
  returns a `BudgetDecision`; the taxonomy mapping stays in the consuming
  layer (Call raises `LocalCapExceeded` from it), so a local refusal can
  never be spelled `QuotaExhausted` (SR-D3) and the port stays satisfiable
  purely by shape. The estimate is denominated in `Spend` itself -- a second
  estimate vocabulary would let the cap and the bill drift (SR-E2).
- **Call's estimate is a floor, not a quote:** `calls=1`, plus one weighted
  unit when -- and only when -- the provider's own capability declaration says
  per-weighted-unit pricing. What *this* request weighs is the adapter's
  planning knowledge; re-deriving it in Call would be a second tally.
- **Ordering in Call:** check → acquire → provider call → record, all below
  the retry boundary; a budget refusal short-circuits the limiter too (a
  call that will not be made takes no pacing slot); the pacing wait is paid
  out of the caller's bound, not on top of it (SR-G2); `record` fires
  exactly once per *attempted* call -- success, typed failure, or unmapped
  defect -- and never for a refusal or denial, with the same `Spend` the
  caller receives. An explicit `timeout_seconds=0.0` now means zero.
- **Egress reaches Call as a parameter** defaulting to `direct` -- nothing on
  `SearchProvider` carries the transport's egress today. If contracts later
  let a provider declare its egress, the parameter becomes derivable; that
  is a contracts change, not taken here.
- **`bind_search` forwards the ports** -- deferred at first build to the
  Phase-2 consumers (§4), then landed same-branch on 2026-08-11 after
  review: an entry point the tool envelope reaches that could not carry
  `budget`/`limiter`/`egress` enforced nothing for the caller most able to
  search in a loop. They pass through verbatim; a refusal from either
  authority renders as a failed result like any other typed failure (D10).
- **Contract gaps carried, worked around with precedent, not patched
  mid-chunk:** `Spend` cannot distinguish *unpriced* from *free* (the
  capability declaration's `pricing_model` carries the difference; an
  additive `money_known` flag would close it); the taxonomy has no
  invalid-request class (a provider 400 maps to `TransportFailed` with
  teaching remediation, matching the SearXNG else-branch);
  `LocalCapExceeded.scope` doubles as cap identity (`query-length`,
  `response-bytes`) beyond SR-D2's scope-tag reading -- existing `standalone`
  precedent, now shared by the Tavily adapter and Call's budget refusals.

### Phase 2 -- in-family consumers (branches stacked on Phase 1)

*Status 2026-08-11 -- most of this phase has landed.* The ground first:
[#307](https://github.com/pacepace/3tears/pull/307) implemented
`FetchTransport` on `StandaloneTransport` and settled §3.8's
per-request-client condition with an opt-in `connection_scope()`;
[#310](https://github.com/pacepace/3tears/pull/310) recorded §3.5's five
Extract rulings before the build; and
[#315](https://github.com/pacepace/3tears/pull/315) landed the
`extraction_status` constants Extract records into, ruled off the shipped
DDL rather than proposed.

Then the phase itself:

| Item | State |
|---|---|
| 4 -- Extract's web path | **Done** -- [#316](https://github.com/pacepace/3tears/pull/316), with four more rulings in §3.5 |
| 5 -- gut the two builtins; serve wiring; NATS metadata test | **Done** -- with the transport-split ruling below, plus a correction pass ([#321](https://github.com/pacepace/3tears/pull/321)) that fixed seven review findings and the seam gap that hid them |
| 6 -- `page_finder` structure; context-save node | **Done** -- `ToolExecutor` in [#318](https://github.com/pacepace/3tears/pull/318); `page_finder` reads structure (check 4) and the context-save node's C8 fix, both below |
| 7 -- envelope asks | **Done, pulled forward** -- [#317](https://github.com/pacepace/3tears/pull/317) |
| §4.7 executor artifact | **Done** -- [#318](https://github.com/pacepace/3tears/pull/318) |
| §4.8 MCP `structuredContent` | **Done** -- [#319](https://github.com/pacepace/3tears/pull/319) |

**Phase 2 is complete.** What remains before Gate B is Phase 3's
`aggregate`/`select`, plus the gate's own sweep of the in-repo success checks.

#### Phase 2 item 6, `page_finder` -- built 2026-08-12

Check 4 is discharged, and its "without its callers changing" clause is the
literal shape of the change: every new fact arrives as a defaulted
`PageFinderResult` field, so existing construction and existing readers are
untouched. `page_finder` reads the typed projection off `ToolMessage.artifact`
-- the artifact item 7 stopped the executor from stringifying -- rather than
re-parsing the prose the LLM read.

Three things structure buys that prose could not, and one it deliberately does
not:

- **A URL the search never returned is now visible as such.** The coercion step
  can name a page the loop reached by following a fetched link, or one it
  invented outright; `url_was_a_search_result` tells those from a real find.
  The URL is still the LLM's choice -- structure qualifies the answer, it does
  not replace it.
- **A refused search stops reading as a fruitless one.** Every empty run used
  to report "exhausted its turn budget"; a typed `rate-limited` now says so,
  class first, because that is the fact an operator acts on. The verdict needs
  *every* turn to have failed, not merely one: a run whose first search was
  rate-limited and whose next four searched fine did not fail for want of
  searching, and blaming the provider would send an operator after a quota
  problem that had already cleared. The first failure is reported on its own
  field either way.
- **Provider degradations survive** (SR-L2, P8) -- a page found over a search
  that lost two engines is still a finding, just one whose thinness has a
  stated cause.
- **No ranking is implied.** `candidates_seen` is provider order across turns,
  deduplicated by identity; ordering them is Select's business, not this
  module's.

Two seam facts worth keeping, because both are the kind that pass a unit test
and fail in production. `web_fetch` writes its projection under the *same*
metadata key as `web_search`, so the reader filters by bound tool name --
`threetears.web_search`, never the bare string, the identical name-grain bug
`_extract_search_queries` already carries a regression test for. And
`from_metadata` refuses a newer `schema_version` loudly (D13), which is right
for a reader that may fail and wrong inside a function that promises never to
raise, so here that refusal degrades to a warning and the prose path.

`scrape` now declares `3tears-search` directly. It had been arriving only
transitively through `agent-tools[fetch]`, which is not a dependency a package
may lean on for its own imports.

**One adjacent defect, found by testing the module rather than the change.**
`_verify_candidate_page` fetched unbounded: `client.get` buffered the whole
body and BeautifulSoup then built a parse tree from it, measured at **77x** the
served size -- 19 MiB of HTML peaked at ~1.5 GiB of heap. It fetches a URL an
LLM picked out of search results, so that size was never this process's to
choose. This is the same defect class as §10 defect 7, which the gutting
removed from `web_fetch`; it survived because this is scrape's own fetch rather
than the leaf's. Now streamed under a 2 MiB cap matching `extract.py`'s
`DEFAULT_MAX_BYTES` (SR-G5), same peak measurement down to 157 MiB, and the
"nothing found" note distinguishes *nothing in what I read* from *nothing on
the page* -- a note that conflated them would send the next reader hunting a
structure bug that is really a size cap.

Encoding behaviour is pinned rather than assumed, because the decode moved from
`response.text` to an explicit one: declared Shift-JIS, windows-1256 and UTF-16
all decode; an undeclared or self-contradicting charset degrades with
`errors="replace"` instead of raising; and a body sliced mid-multibyte at the
cap does the same. A charset Python has *never heard of* falls back to UTF-8
with a warning: `httpx` returns the `charset=` parameter verbatim without
checking it against the codec registry, so a server declaring `utf8mb4` (a real
MySQL-ism) or any typo raises `LookupError` -- which is **not** a `ValueError`,
so it escaped the fetch's own guard and left `find_target_page`, whose contract
is that it never raises. Third-party input on the strength of a string match. Structure detection survives all of it for a reason worth
stating -- every marker it looks for (`<table>`, `<tr>`, `href`) is ASCII, so
finding structure never depended on rendering the text correctly. A bidi
override that makes a link *render* as `.pdf` is also pinned as not verifying,
since the extension check reads the real characters.

#### Phase 2 item 6, the context-save node (C8) -- built 2026-08-12

The last of Phase 2. Three deliverables, and the third is the one with a
deadline attached.

**The name-grain defect is fixed, and the fix is not the interesting part.**
`_DEFAULT_SAVEABLE_TOOLS` held bare `web_search` / `web_fetch` matched by exact
equality against `ToolMessage.name`, while the adapter binds every tool under
`mcp_name()` -- so the default set matched nothing and the node was inert in
production. The interesting part is *why the suite could not see it*: every
test passed its own bare names explicitly, so the node's logic was asserted
against names the tests chose rather than the name production binds. The
regression pin therefore reads the real tools' real `mcp_name()` values rather
than restating a string, and a seam test drives the real adapter end to end and
lets it pick the name.

**The node binds on result type before tool name**, which is what C8 actually
asked for: *"it binds on the tool name as a string, not on the result type --
and the failure class this predicts has already fired."* A message carrying
search structure is saved whatever the tool is called, so renaming or splitting
a tool per carrier can no longer silently change what is retained. The name set
remains, for tools with no structure to recognise.

**Structure is retained beside the prose**, making this the consumer C8 said
should *"get better rather than merely unbroken."* The stored row previously
held a flattened 4000-character truncation with no provenance -- a claim nobody
could trace back. It now carries the query, candidate identities and titles,
notices, and any typed failure class, so SR-A3's re-checkability survives the
truncation. The query also becomes the dedup fingerprint, so asking the same
thing twice refreshes a row instead of stacking a second copy. A failed search
is recorded *as* a failure, which prose alone could not distinguish from a thin
answer.

**The retention posture is stated in the module docstring, before any wiring**
-- the ordering C8 required and the reason it was still available: the node has
been shipped-but-inert since it was written, so the posture could still precede
the first retained byte. It records what is kept and the rules that govern it
(D11/SR-K2 queries are user content, redaction is the host's; D7/D12 retention
follows the consumer's policy; SR-K4 fetched page text is third-party content),
and it states that none of those are this module's to decide. Wiring remains
deliberately undone.

Two adjacent corrections, both the same defect class one layer over:

- **`chunker.py` registered its only default strategy under the bare
  `web_fetch`.** The save node passes a message's tool name as the strategy
  hint, so fixing the node's names alone would have silently downgraded header
  chunking to line chunking. `chunk_content` now falls back from a namespaced
  hint to its bare tail, which fixes it for every namespaced tool rather than
  just this one.
- **An empty `saveable_tools` set meant its own opposite.** `saveable_tools or
  _DEFAULT_SAVEABLE_TOOLS` made `frozenset()` falsy and fell back to the
  defaults, so a caller asking for "no tools by name" got the defaults instead.
  Now `None` means defaults and empty means empty, which also makes
  "structure only" expressible.

Review corrected three things, one of which falsified a claim made for this
node:

- **The dedup fingerprint never deduped.** `save_tool_result` derives the row
  key as `tool_name:sha256(fingerprint)`, and the node passed
  `tool_name=f"{name}:{tool_call_id}"` -- so the per-call id was baked into the
  key and two identical queries could never collide. The node was duplicating
  uniqueness logic the store already provides; it now passes the bare bound
  name and lets the store key it. The original test asserted only that the
  kwarg was *passed*, which is precisely why it passed; the replacement drives
  the real key derivation and asserts two identical queries land on one key.
- **Structure-first retention had no off switch**, and `web_fetch` projects
  under the same metadata key as `web_search` -- so it also captured fetched
  page text, which SR-K4/D12 make the deployment's own agreement with a site.
  "Stop using the node" is not a lever. `save_structured` is, and it defaults
  to the C8 posture.
- **`short_desc` could exceed its documented 200 characters**, and degraded a
  fetched page's summary from a content preview to `1 result(s) for
  'https://...'`. The structured summary is now bounded, and reserved for a
  genuine result *set* rather than the single candidate `web_fetch` projects.

**Conditional requests: nothing today, and the question it raised is now ruled.**
No `If-None-Match` or `If-Modified-Since` exists anywhere in the leaf, scrape, or
core's traced client. For `_verify_candidate_page` specifically that is correct
and stays correct -- it runs once per candidate at discovery, holds no copy, and
would have nothing to inspect on a 304.

Asking why led somewhere larger, and the answer is **SR-M4 / D30**, ruled
2026-08-12. D14 was never the obstacle: it forbids the capability *holding* a
response, and a conditional request holds nothing. What the gap actually was is
that D7/D12 put the bytes in the consumer's store and then left the consumer no
way to spend the validator they arrived with, so every re-read of an unchanged
page pays full freight -- on the scrape path, a render and an LLM extraction. The
build sequence is `search-task-01-conditional-revalidation.md`; the transport
seam already suffices and the work sits above it.

#### Phase 2 item 5 -- built 2026-08-11

Both builtins now run on the leaf, `serve.py` wires them, and check 8 is
pinned end-to-end: a real `ToolServer` dispatch of the real `WebSearchTool`
publishes a `CallResponse` whose bytes a consumer rebuilds with
`SearchResultsMetadata.from_metadata` -- on the failure path as well as the
success one, which is the half a success-only carry would have left on prose.
The defects the gutting was scheduled to retire are gone with the bodies that
held them: the 15-second hardcode and the sync client inside `async execute`
(§10 defect 2), `time.sleep` and unbounded `resp.text` (§10 defects 6, 7), and
`[TOOL ERROR]` string-prefix error detection on both tools (§10 defect 8).

Rulings taken during the build, recorded here per the Gate A precedent:

- **The two transport protocols get two implementations, and §3.8's
  expectation that one host adapter would satisfy the union was wrong.**
  Gate A declared `FetchTransport` beside `SearchTransport` and predicted the
  `TracedHttpClient` adapter would implement both. It cannot, for a reason
  that is structural rather than unfinished: `TracedHttpClient` is
  constructed **per upstream** -- one `upstream_base_url` that request paths
  join onto, one circuit breaker guarding it -- and it buffers whole bodies.
  A search call is one configured upstream and a small buffered response, so
  the adapter fits it exactly. Extract fetches a *candidate-derived* URL: a
  different host every call, under a per-call byte cap, refused on content
  type before the body. There is no upstream to construct a client for, no
  breaker whose state would mean anything, and no way to cap a buffered read.
  So `agent-tools` injects `TracedSearchTransport` for search and
  `StandaloneTransport` for fetch. This does not reopen D19: what that norm
  forbids is a host hand-rolling a client, and `standalone` is the sanctioned
  single-purpose transport module it explicitly widened to admit. The
  module's own docstring, which said hosts with core would not need it, has
  been corrected rather than left to contradict the code.
- **The traced client gained a per-call timeout and a visible attempt
  count**, both additive, both required for the adapter to be a conformant
  transport rather than one that quietly ignores its obligations. SR-G1/G2
  make the per-call bound the mechanism a caller's deadline travels through,
  and item 7's `deadline_seconds` has nowhere to land without it. D4 needs
  the attempts: retry lives *inside* the client, so a caller billing per
  exchange sees one where there were three, which is the SR-E4 under-billing
  class this package exists to retire. The count rides `Response.extensions`
  -- httpx's own channel for transport-level facts -- so no existing caller
  learns a new shape to ignore. §4.11's "core: nothing moves" is unaffected:
  nothing moved, and the two additions are this workstream's, not core's
  separate backlog.
- **A search transport refuses a URL off its configured base** (D21). The
  guard is cheap and it is the only seam that can enforce "base URLs come
  from deployment config" for a per-upstream client; it keys on
  scheme/host/**port**, because same-host-different-port is exactly what a
  host-only check waves through.
- **`web_fetch` projects its one candidate under the search-results key**
  rather than inventing a second border vocabulary (D22). A consumer reading
  structure off a tool result reads one shape whether the tool searched or
  fetched, and learns *why* an empty fetch was empty from the typed
  `extraction_status` facet. The projection's `query` slot carries the URL
  asked for, since a direct fetch has no query behind it and inventing one
  would be worse than naming what was actually requested.
- **`credential_resolver` was dropped, not ported.** Its only path in was a
  `_credential_resolver` key in the tool's config dict that nothing in the
  family ever set; carrying an unexercised credential-injection seam through
  the rewrite would have meant re-deriving its failure semantics for no
  caller.
- **The skip-with-reason pattern had to be extended by probing, not by
  catching.** Extract imports trafilatura lazily, so the extra's absence no
  longer raises `ImportError` at registration -- the tool constructs fine and
  refuses every call. `serve.py` therefore probes for the extractor and skips
  with a reason naming `3tears-agent-tools[fetch]`, because a pod that
  registers a tool it cannot serve is the exact failure the pattern exists to
  prevent. *Corrected below: the probe went into `serve.py` only, and
  `register_builtins` needed the same one.*
- **The two transports want opposite redirect defaults, and the split is the
  reason why** (added in the correction pass). `build_fetch_transport` took
  the leaf's `DEFAULT_MAX_REDIRECTS` of 0, which is right for the half it was
  named after and wrong for the half it serves. A search upstream is a
  deployment-configured host answering its own API: a healthy one does not
  redirect, so refusing to follow is a signal. A fetch is a candidate-derived
  URL, where http->https, `www` and trailing-slash canonicalisation are how a
  large share of the real web answers -- so refusing turned ordinary pages
  into `extraction_status: failed`, and, because robots is read through the
  same transport, a `robots.txt` that had merely moved refused its page
  outright. `DEFAULT_FETCH_MAX_REDIRECTS` names the fetch stance at 5, which
  is what the hand-rolled body this replaced allowed; every hop stays
  re-guarded. This is the *second* time Gate A's one-adapter expectation cost
  something: the first was the transport split itself, and this is the same
  assumption surviving in a default.

**The correction pass, and the test gap that made it necessary.** A review of
this item found seven defects, the redirect default above being the one that
would have shipped as a widespread `web_fetch` failure. The others: the
whole-run refusal reached the border with `failure: null` (built through
`from_candidate_set`, which has no slot for a record, where `from_failure`
exists for exactly this); a per-call bound bounded each *attempt* rather than
the call, so a caller with 0.3s remaining could fund three attempts plus
backoff, which is not what SR-G2 says the argument carries and not what
`StandaloneTransport._perform` already does with it; `register_builtins` still
skipped only on `ImportError`; a `max_chars` under the truncation marker's own
length made the slice index negative and returned the *tail* of the page; a
caller-supplied `http_transport` was closed by the per-call client; and
`FetchTransport`'s docstring still named `TracedHttpClient` as an implementer
of the union this item declared it could not be.

They share one cause, which is the finding worth keeping. The tool and the
transport were each tested alone and the seam between them was not tested at
all: every `WebFetchTool` test injected a stub that hardcodes 200 and cannot
answer a 3xx, and every `build_fetch_transport` test asserted conformance and
refusals without ever serving a response. So `WebFetchTool()` with no injected
transport -- the shape a pod runs -- had no test, and the suite proved values
were *passed* rather than that behaviour *held*. `test_standalone.py`
compounded it by pinning `max_redirects=0` as correct, which it is, for
search. The seam now has its own class driving a real socket through the
transport the tool builds for itself, and `LocalHttpServer` moved from
`packages/search/tests` into `threetears.search.testing` to make that possible
-- a published addition, justified by that module's existing rationale that a
host injecting its own transport must be able to pin its own wiring.

**Two user-visible behaviour changes, and the rollout they need.** Both are
`web_fetch`'s, both were foreseen in kind, and neither needs a second release
cycle -- they need *saying*, because a caller cannot discover either from a
signature:

1. **Robots became binding** for callers it was never binding for (D12). A
   page whose rules refuse `3tears-search` now returns a refusal instead of
   content. The stance is deployment config, not a per-call parameter, so a
   deployment with its own agreement with a site sets `respect_robots=False`
   where it constructs the tool.
2. **Extraction refuses instead of degrading.** The old body fell back to
   stripping tags with a regex when trafilatura was absent; Extract refuses
   with a typed `LocalCapExceeded` naming the extra. Regex tag-stripping is
   not extraction, and returning it as though it were is how a caller ends up
   reasoning over navigation chrome -- but the consequence is that an install
   without the extra stops returning *anything*. `3tears-scrape` declares
   `3tears-agent-tools[document,fetch]` as a result; any other consumer
   driving `web_fetch` owes itself the same line.

4. Extract's web path (streamed, capped, robots stance, no-op on
   provider-supplied content). **Done 2026-08-11** -- `extract.py` plus the
   `HeavyFetcher` slot in §3.1; rulings in §3.5.
5. Gut `WebSearchTool` + `WebFetchTool`; serve.py wiring; metadata key
   end-to-end test over NATS (check 8).
6. `page_finder` structure (check 4); context-save node fix + retention
   posture (§4.5).
7. Envelope asks, as their own PRs with the rollout-order note (D18).
   **Done 2026-08-11, pulled forward.** Both landed early for a reason the
   phase ordering hid: they are additive in *different* directions. §10.9
   populates `CallResponse.metadata`, a field that already existed, so it
   has no ordering constraint at all. §10.10 adds
   `CallRequest.deadline_seconds` to a model with `extra="forbid"`, where a
   client that sends it to an older server gets its call **rejected**, not
   degraded -- so only the accepting half shipped, and a caller may not be
   taught to populate it until a release carrying this one is deployed.
   That is the only item in Phase 2 that needs two release cycles, which is
   why it stopped being last.

   **What production then found, 2026-08-13 (0.24.3).** The conclusion above
   was right and its premise was wrong, which is the more useful half. Shipping
   only the accepting side does not keep the field off the wire: Pydantic
   serializes every *declared* field, so an optional nobody set still crosses
   as an explicit `null`, and `extra="forbid"` on the receiver treats a null it
   does not know exactly like a value it does not know -- the WHOLE call is
   refused. A field can therefore break every lagging pod in the fleet without
   a single caller ever populating it. Observed on cobalt-dev: a 0.23.11 tool
   pod refused every call from a 0.24.1 registry for three days on
   `deadline_seconds`. The receiving half had shipped exactly as this item
   specified; the pod registered, heartbeated and answered reachability probes
   throughout, so nothing about it looked unhealthy while it refused
   everything.

   Fixed in 0.24.3 by pruning unset **top-level** optionals from the
   registry→pod envelope -- the registry proxy is the only sender of
   `CallRequest` -- with the scope deliberately not recursive: `CallContext`
   does not forbid extras and needs no protection, and pruning it would strip
   identity dimensions the proxy stamps as null on purpose, including the
   verified-`user_id`-is-None case that must override a claimed user. The same
   release named the refusal in the pod's own log (it had been silent, the
   reason travelling only in a reply read by a model) and gave the pod an
   arrival record, so "did the call get here" stops being an inference from two
   timestamps.

   **The generalisable lesson, and it is this document's recurring one in a new
   costume.** Phase 2 item 5 recorded that a host's *configuration* needs its
   own end-to-end test; the Gate B sweep recorded that *independence* was
   pinned nowhere because every test drove one provider. Here the untested seam
   was between two **versions**: both sides were correct alone, the rollout note
   asserted a wire fact, and nothing ever parsed real forwarded bytes with a
   model predating the newest field. `test_forward_is_version_tolerant.py` now
   does. A prose rollout note is not a compatibility mechanism, and any future
   additive field on an `extra="forbid"` envelope should be read as needing a
   test that a *lagging* reader accepts the bytes, not a sentence promising no
   one sends it.

### Phase 3 -- pull-driven depth (may start parallel to Phase 2 after Gate A)

8. `replay.py` + `RecordingStore` port + record schema (versioned envelope),
   under D26's durability rules (canonical-request keying, adapter-free
   replay, versioned refusal). The record format is a lock-in decision: its
   consumers' future queries (discodon's eval replay; samsung's re-search)
   are elicited in the build plan before fields are cut.
9. `aggregate.py` (corpus, dedup, producer seam) and `select.py` (criteria
   application, cull, ranker slot, degradation marks).

| Item | State |
|---|---|
| 8 -- `replay.py` + `RecordingStore` + record schema | **Not started, deliberately** -- the record format is elicited against discodon's carved storage port, which is a Phase 1 item still outstanding outside this repo. Cutting it now would be cutting it against an imagined consumer |
| 9 -- `aggregate.py` / `select.py` | **Done** -- steps 1-3 of the task doc in [#333](https://github.com/pacepace/3tears/pull/333) and [#335](https://github.com/pacepace/3tears/pull/335); the producer seam (step 4) is sketched and deliberately unbuilt, below |

#### Phase 3 item 9 -- built 2026-08-12

Rulings, tests owed and build sequence are
[`search-task-02-aggregate-and-select.md`](search-task-02-aggregate-and-select.md);
what belongs here is the state and the two things that reached past the item.

`contracts/corpus.py`, `contracts/ranker.py`, `contracts/shortlist.py`,
`aggregate.py` and `select.py` shipped with the tests §6 of that document owes.
The ruling worth re-reading is **R1**: the corpus is a set of *groups*, not a bag
of merged candidates, because `Candidate.provenance` is singular and a dedup that
collapses two providers' hits into one candidate must discard one origin --
destroying exactly the per-result grounding SR-A3 exists to keep. Contributions
stay whole.

**Step 4 -- the producer seam -- split off rather than built**, and the argument
that split it is worth keeping because it reversed itself under checking. The
case for building now was that samsung is available as a second producer, so D3's
deferral no longer applied. samsung's session then reported that its active build
plan contains no search producer and that `3tears-core` is deliberately absent
from that repo. So samsung is available for **contract elicitation** -- which
already paid, two of its answers contradicting what the document would otherwise
have specified -- and *not* available to consume a seam. Building it now would
produce a seam whose only caller is a test, which is the exact failure the
argument invoked to justify building it. The sketch is
[`search-task-03-producer-seam-sketch.md`](search-task-03-producer-seam-sketch.md)
([#334](https://github.com/pacepace/3tears/pull/334)), reviewable now and
approved to build when a consumer can drive it.

**Success check 2 is designable now and not satisfiable now**, and it does not
block Gate B -- the gate's wording is *"success checks that can be verified
in-repo"*, and check 2 lives in samsung's repo. It was never waiting on this.

**Gate B (pre-release):** all §3 success checks that can be verified in-repo
are; SR-A4's SearXNG score semantics confirmed against a live instance
(**done 2026-08-12** -- the formula, its four consequences and the one residue
are recorded at SR-A4 in the requirements doc; discharging the last of it wants
an instance whose engines are not rate-limited, run with
`SEARXNG_REQUIRE_RESULTS=1`); requirements doc's §13 updated with any vetoes
taken during build. *Rider
2026-08-11:* a release overtook this gate (below), so Gate B now gates the
**next** release -- the one carrying Phase 2-3 -- not the leaf's first
appearance on PyPI. Nothing in it is discharged by 0.24.0 having shipped.

***Rider 2026-08-14 -- the rider above is retired, because it describes a
brake that does not exist.*** Four further releases shipped while Gate B stayed
open, and they carry the surface it was said to guard: **0.24.1** has Extract's
web path, the gutted builtins and both envelope asks; **0.24.2** completes
Phase 2 with `page_finder` and the context-save node; **0.24.3** adds
`aggregate`/`select` and the Gate B sweep itself. None of this was a bypass.
The family releases on its own cadence for reasons that have nothing to do with
search -- 0.24.3 went out to fix the envelope-null outage above -- and a gate
phrased as "the next release" was always going to be overtaken by the next
unrelated fix.

So state what Gate B actually protects, which D29 already named and this rider
merely stops contradicting: **the first consumer release that pins a version
carrying search**. That is the moment the contracts freeze and re-cutting them
stops being free. Publication does not bind them; a tag does not bind them; a
consumer's pin binds them. Gate B is the check standing in front of Phase 5,
not in front of the release button, and nothing about it is discharged by
0.24.1-0.24.4 having shipped.

#### Gate B sweep -- 2026-08-13

The gate has three parts. The SearXNG half was discharged 2026-08-12
([#322](https://github.com/pacepace/3tears/pull/322),
[#328](https://github.com/pacepace/3tears/pull/328)). This is the other two: the
in-repo success-check sweep, and the §13 propagation.

**Which checks are in-repo.** Nine of the fourteen: 4, 5, 6, 7, 8, 11, 12, 13,
14. The other five are consumer-repo by construction -- 1 (metallm's side-steps),
2 (samsung's image search), 3 (discodon's cost cap and replay), 9 (samsung's
one-shot `asyncio.run()`), 10 (discodon before and after NATS). The gate's
wording excludes them and always did; they belong to Phase 5 and Gate C.

| Check | Verdict | Evidence |
|---|---|---|
| 4 -- `page_finder` structured, callers unchanged | **PASS** | [#326](https://github.com/pacepace/3tears/pull/326), [#335](https://github.com/pacepace/3tears/pull/335). Every new fact is a defaulted `PageFinderResult` field, so the "without its callers changing" clause is the literal shape of the change |
| 5 -- installs without torch | **PASS** | `test_import_cost.py` -- an *allowlist* over everything a fresh interpreter loads, covering the stage modules as well as the contracts, with both halves confirmed to fail on real cases before being relied on |
| 6 -- a new provider without touching a consumer | **PASS, with its limit stated** | `SearchProvider` + one conformance suite run against both adapters; no consumer names a concrete provider. The limit: both providers are ours. A third-party adapter is the check's real proof and arrives with a third-party author |
| 7 -- a new carrier without a coordinated release | **PASS in-repo; its wire half is Gate C's** | `Candidate` carries no carrier union, `facets` is an open mapping, `Criterion.carrier` takes an open `media_category`. A new carrier is facet *values*, not a schema change. A reader on an older version is D13's `extra="forbid"` stance, deliberately deferred to Gate C |
| 8 -- builtin keeps its name and shape | **PASS, end-to-end** | `test_search_metadata_over_nats.py`: named key on the wire, a consumer rebuilding the projection from the published bytes, typed details not flattened, schema version riding along, and the failure path carried as well as the success one |
| 11 -- `test_no_bespoke_reuse` without an exemption | **PASS, and the exemptions file does not contradict it** | `standalone.py` is in `_SANCTIONED_HTTPX_SITES` in the walker -- the D19 *widening* SR-N1 asked for, not a waiver -- and files no exemption. Worth stating because a reader who greps `_no_bespoke_reuse_exemptions.txt` finds a search line and would reasonably conclude otherwise: that line is `LocalHttpServer`, a scripted test HTTP server, not the Adapter and not a transport |
| 12 -- egress routed independently, result names its exit | **PASS as of this sweep** -- it did not before | See below |
| 13 -- facets found in `media-contracts`, not added here | **PASS, structurally** | `contracts/facets.py` keys are `MediaInfo`'s and `MediaFacets`' own field names, checked **at import**, so a drift fails loudly rather than the two vocabularies diverging silently. The three genuinely new facets landed in `media-contracts` itself -- check 13 passing is the observable fact that they did |
| 14 -- three faces, one contract | **SPLIT, and the in-repo half PASSES** (2026-08-14) | `test_three_faces_one_contract.py` drives one candidate set through both renderings that exist here and compares them *to each other*; `tests/enforcement/test_one_search_result_shape.py` holds the shape at its source. The API face renders in the hub and is verified there, with checks 1, 2, 3, 9 and 10. See below |

**Check 12 was the sweep's find, and it is the pattern this document keeps
recording.** Egress was pinned in several places and *independence* was pinned
nowhere: every existing test drove one provider through one exit, asserting that
a value was carried. The requirement names its own hard case -- *"the self-hosted
SearXNG and the Tavily API can take different exits in the same process"* -- and
nothing drove two. `test_egress_independence.py` now does, through both real
adapters, and asserts the exits stay disjoint on the results, in the pacing
buckets, and after Aggregate has merged them.

The first draft of that file was itself the bug it was written to catch: its
assertions compared each side against the constant it had been configured from,
so collapsing the two exits to one value left all four tests green. A deployment
routing both providers through one exit would have been reported as passing check
12. The pins now compare the two sides *to each other* -- `!=`, `isdisjoint` --
with a guard test on the fixture itself, and the collapse was confirmed to fail
four of the five before the file was relied on.

Two facts about egress that the sweep surfaced and that belong in the record.
A result's exit comes from the **transport** (`TransportResponse.egress`, stamped
onto `Provenance` by the adapter), while Call's `egress=` parameter keys pacing
and stamps refusals. They are two independent sources of one fact, and nothing
reconciles them -- a deployment that passes one and injects a transport reporting
another gets pacing keyed on a value its own results contradict. This is not a
new defect: it is the gap Phase 1 item 2 already recorded (*"nothing on
`SearchProvider` carries the transport's egress today... if contracts later let a
provider declare its egress, the parameter becomes derivable; that is a contracts
change, not taken here"*). It stays not-taken, now with a test standing on the
side of it that a consumer can observe.

**Check 14 does not pass, and it is a decision rather than a build.**
`WebSearchTool` and `WebFetchTool` leave `face_api` and `face_mcp` at their
`False` defaults, so two of the three faces are unreachable and the property the
check is actually about -- *no second result shape per face* -- has nothing to
hold against. The mechanism is built and tested (Bind renders; `agent-tools`'
`mcp.py` carries `structuredContent`); the reach is not turned on. Turning it on
is two class attributes, but it is **ACL-visible surface** and §13 already lists
the adjacent question as needing an owner (`skill_eligible`, and whether search
stays in the `web` group alias). So this sweep records the gap rather than
closing it by fiat, and the decision is now the one thing between here and a
clean Gate B.

**Gate B status: two of three parts discharged.** SearXNG done; the sweep done
with eight of nine in-repo checks passing and check 14 held on a decision; §13
propagated (see the requirements doc's §13, updated the same day). Gate B closes
when check 14's face decision is taken -- either the flags go on and the pin is
written, or the check is amended with the reason, which is what §13's
"decisions needing an owner" is for.

#### Check 14, resolved by splitting it -- 2026-08-14

**The decision was framed as flip-or-amend, and both options were wrong**,
because both assumed the check needs the reach turned on. It does not. The face
flags govern reach and **nothing in this repository reads them**: verified by
search -- `mcp.py` never consults them, no in-repo HTTP surface consults them,
and the only in-repo consumer is `publish_registration`, which copies them onto
`ToolManifestEntry`. From there they land in the ACL `namespaces` table, whose
own comment names the surfaces that act on them: the API namespace stamp
(gu-task-24), the MCP export (gu-task-25) and the face-flip re-stamp
(gu-task-26). All hub-side.

So flipping the flags would change **what the hub is told**, not what this
repository renders -- and check 14 is about rendering. Its parts divide the way
the other five out-of-repo checks already do:

| Face | Renders | Verified |
|---|---|---|
| platform tool | here (`TearsTool.execute` → `ToolResult`) | in-repo, this sweep |
| MCP | here (`threetears.mcp.server`, prose `content` + `structuredContent`) | in-repo, this sweep |
| HTTP API | in the hub (`map_tool_result_to_http` over `metadata["http"]`) | where it is exposed, with checks 1, 2, 3, 9, 10 |

**What was built.** `test_three_faces_one_contract.py` renders one candidate set
through both in-repo faces and asserts they agree -- comparing the faces **to
each other**, never to the constant they were built from, which is the
`test_egress_independence.py` lesson applied before rather than after. It
carries guard tests that reproduce a reshaped face and a truncated one and
confirm the comparison notices, plus a pin that no rendering module reads a face
flag: if one ever does, the faces can disagree, this split stops being honest,
and the pin says so in its failure message.

`tests/enforcement/test_one_search_result_shape.py` holds the other half, which
the unit test structurally cannot: a comparison between known faces cannot see a
face nobody added yet. So the shape is protected at its source -- the border key
may be read anywhere and **constructed in exactly one place**, `bind`, which
owns the schema version and the field names. A second guard forbids spelling the
key as a literal, without which the first is defeated by anyone who did not know
the constant existed.

**Two findings from building it, both pre-existing.**

`web_fetch` was already a second construction site: it reimplemented the
projection -- `SearchResultsMetadata.from_candidate_set(...)` then
`{SEARCH_RESULTS_METADATA_KEY: ...}` -- once for success and once for refusals.
Output identical to `bind`'s, which is exactly why nothing failed, and identical
only until somebody edited one copy. Both now call `project_metadata` /
`project_failure_metadata`; the latter is new, exported so `bind_failure` and
`web_fetch` share one owner rather than one shape twice.

The context-save node writes a **deliberately narrowed** record under the same
key -- candidate identity and title only, plus `candidate_count` and
`candidates_truncated`, because a context row must not become a second copy of
the corpus. That is intended and documented at the site, so it is exempted with
a rationale rather than closed. It does leave an asymmetry worth naming: the
narrowed record carries `schema_version` from the full projection, so a reader
that finds this key on a context row and hands it to
`SearchResultsMetadata.from_metadata` gets something that parses while
under-reporting. Not a defect today -- nothing does that -- but it is the shape
of one, and it is now written down where the next reader will find it.

**Gate B closes on this.** Nine of nine in-repo checks pass; the reach half of
14 joins the five consumer-repo checks that were always Gate C's and Phase 5's.
§5.5's `skill_eligible` / `web`-alias question stays open in §13 on its own
merits -- it is a reach decision, and this check never depended on it.

### Phase 4 -- release

10. Family minor bump (lockstep -- the bounds test names every edge), PR into
    develop, PR develop→main, tag pushed from main. `3tears-search` appears
    on PyPI with the rest of the family.

#### Taken early -- v0.24.0, 2026-08-11

The family shipped its lockstep minor with Phase 1 complete and Phases 2-3
not started, so the release step ran three phases ahead of where this
section puts it. Bump in
[#304](https://github.com/pacepace/3tears/pull/304), develop→main in
[#305](https://github.com/pacepace/3tears/pull/305), D29 ruled in
[#306](https://github.com/pacepace/3tears/pull/306). Verified: tag `v0.24.0`
on origin, the GitHub Release exists, and all 30 packages --
`3tears-search` among them -- are on PyPI at 0.24.0. Three consequences,
none of which reorder the remaining build:

- **The contracts are published but not bound.** Ruled as D29: Phases 2-3
  may still re-cut them; the freeze is the first consumer *release* that
  pins a version carrying search, not this publication.
- **0.24.0 is not the version Phase 5 migrations pin.** It carries the leaf
  but not Extract, not the gutted builtins, not replay. A consumer release
  containing migration work still gates on a later tag -- the one Gate B
  clears. Consumer *development* is unaffected: it tracks develop (Phase 5
  preamble).
- **Gate B did not move to the past.** It kept its content and changed which
  release it guards (rider above).

**Which tag carries what -- recorded 2026-08-14, because the bullets above go
stale silently and a consumer pinning the wrong one gets a build that compiles.**

| Tag | First carries |
|---|---|
| v0.24.0 | the leaf alone -- contracts, both adapters, Call, Bind, standalone transport, limiter |
| v0.24.1 | Extract's web path ([#316](https://github.com/pacepace/3tears/pull/316)), both envelope asks ([#317](https://github.com/pacepace/3tears/pull/317)), the `ToolExecutor` artifact fix ([#318](https://github.com/pacepace/3tears/pull/318)), MCP `structuredContent` ([#319](https://github.com/pacepace/3tears/pull/319)), the gutted builtins and their correction pass ([#321](https://github.com/pacepace/3tears/pull/321)), SR-A4 ([#322](https://github.com/pacepace/3tears/pull/322)) |
| v0.24.2 | `page_finder` structure ([#326](https://github.com/pacepace/3tears/pull/326)) and the context-save node ([#327](https://github.com/pacepace/3tears/pull/327)) -- **the first tag carrying all of Phase 2** |
| v0.24.3 | `aggregate`/`select` ([#333](https://github.com/pacepace/3tears/pull/333), [#335](https://github.com/pacepace/3tears/pull/335)), the Gate B sweep and `test_egress_independence.py`, and the envelope-null fix |
| v0.24.4 | nothing search-side (L1 column projection) |

**The two consumers are no longer symmetric, and the Phase 5 preamble's single
"the released version" hides it.** metallm needs the gutted `WebSearchTool`
(Phase 2 item 5): that is **released, in 0.24.2 and later**, so its migration
has a pin it could name today and is held only by Gate B and its own version
lag. discodon needs the replay piece (Phase 3 item 8), which is **not built at
all** -- no tag can carry it -- and is itself elicited against discodon's own
carved storage port, an outstanding Phase 1 item. No release will unblock
discodon; a build will, and that build waits on discodon.

### Phase 5 -- consumer migrations (parallel, per repo; each pins the whole family to the one released version)

*Consumption modes (recorded 2026-08-11).* metallm's and discodon's
**develop** branches track 3tears **develop**; their **releases** pin the
whole family to one exact released version. Two consequences the steps below
inherit: migration *development* in those repos starts when the surface it
needs merges to 3tears develop (metallm's needs the gutted `WebSearchTool`,
Phase 2 item 5; discodon's replay piece needs Phase 3 item 8) and does not
wait for a tag -- but a consumer *release* containing migration work gates on
the Phase 4 tag, because the release pin must name a released version that
carries that surface. The modes MUST NOT hybridise inside one environment: a
develop checkout of one family member beside pinned-PyPI siblings is exactly
the mixed-family install the pinning rule exists to prevent.

**metallm** -- precondition: close its family version lag *first*, as its own
change (every adoption assumes a current pin).
1. `git checkout -b feature/new-search` off its default branch.
2. Bump the family pin to the released version -- whole family, one exact
   version, per the consumer-pinning rule.
3. Replace the raw SearXNG helper (`admin/models.py`) with leaf Call;
   replace `web_fetch_utils` with Extract; **delete both side-steps** -- check
   1 is "deleted, not wrapped".
4. Its frontend/agent callers keep consuming the builtin unchanged; where it
   filtered raw streams for structure, read `metadata` instead.
5. PR, merge per its own workflow.

***Correction 2026-08-19 -- steps 3 and 4 describe a metallm tree three months
stale, and both were already discharged before this document named them.***
The steps are left above as written; what follows is what is actually true, and
**check 1 passes by prior deletion rather than by migration work**.

metallm's commit `f0b6903` (2026-06-23, *"consume the 3tears price-lookup
primitive; drop the bespoke scraper"*) deleted **both** side-steps as a side
effect of unrelated pricing work:

- `api/src/services/web_fetch_utils.py` was **deleted as a file** -- 210 lines,
  zero importers.
- `lookup_details` in `api/src/api/v1/admin/models.py` had its SearXNG
  web-search-plus-LLM-extraction scrape replaced by
  `threetears.models.lookup_price`. That commit's own message records
  "adversarially verified: NO remnant of the bespoke pricing lookup remains".

`grep -rin searxng api/src/` in metallm returns nothing today. There is no
forwarding shim, no re-export, and no module of either name, so the check's
"deleted, not wrapped" clause holds in its strong form. The only surviving
`searxng` strings in that repo are its self-hosted service's own
`searxng/settings.yml` and two `base_url` fixtures in an integration test.

**Step 4 has no target either.** metallm does not filter raw streams for
structure and deliberately refuses to: `_execute_service_tool`'s docstring
(`api/src/services/tool_loop.py`) states that sniffing result text for an error
prefix "both misses real failures and risks false positives", and status comes
from an authoritative bool instead.

**So metallm's migration reduces to step 2**, the pin -- up as
[metallm#288](https://github.com/pacepace/metallm/pull/288), pinning the whole
family at the released tag `v0.26.1`. Under D29 that pin is what *binds* these
contracts: publication does not, and a tag does not. It also unblocks
[metallm#287](https://github.com/pacepace/metallm/pull/287), which fixes
`_execute_service_tool` invoking `ainvoke(tool_args)` by args -- the fourth site
of the defect this repo closed in
[#318](https://github.com/pacepace/3tears/pull/318) and
[#326](https://github.com/pacepace/3tears/pull/326), and one that has been
feeding the model stringified `(content, artifact)` tuples in production.

**The generalisable point, and the reason this is recorded rather than quietly
fixed.** A consumer-repo check written as *instructions to change a named file*
rots at the consumer's commit rate, not ours, and nothing here notices -- the
whole cost of this one was paid re-deriving a June deletion in August. The five
consumer-repo checks (1, 2, 3, 9, 10) all have this shape. **State a
consumer-repo check as the property it wants, and verify it against the
consumer's tree at the time of asking**; a filename in a step is a snapshot,
not a requirement.

**discodon** -- embedded mode (it is pre-NATS-convergence; check 10 says the
switch later costs no consumer rewrite).
1. Any generalization it needs lands **upstream first** (convergence
   principle 4). *(The example that stood here -- gaps found while porting
   its budget semantics onto `BudgetPort` -- is withdrawn with step 4.)*
2. `git checkout -b feature/new-search`; pin the family version.
3. Collapse `tools/web_search_tool.py` and `tools/research/web_search.py`
   onto the leaf: persona path = Call + prose Bind; research path = Call +
   Aggregate + both bindings (prose and corpus).
4. ***Withdrawn 2026-08-19, same ruling as step 6:*** budget hooks do **not**
   move onto `BudgetPort`. What survives of this step is the pair of defects
   it named, which were never about the port: the 2× advanced under-billing
   dies (SR-E4), and timeouts get wired to config (SR-G1 defect 3). SR-E4's
   fix is R5 of
   [`eval-extraction-discodon-requirements.md`](eval-extraction-discodon-requirements.md)
   -- the caller reports what it consumed, so the code that knows the depth is
   the code that counts the credits.
5. Search-internal replay lands where the coarse seams cannot reach:
   research-*pipeline* evals (grounding gate and cull re-run against a
   frozen web), through a `RecordingStore` over its existing store.
   Character-eval freezing stays on its action- and delivery-seam cassettes
   (D28), already wired discodon-side.
6. Eval cost caps include search spend (check 3, under D27's execution-spend
   rule). ***The pre-work this step used to recommend -- reshaping
   `EvalRunCostCap` and the daily-budget mixin onto the port's
   `check(estimate)`/`record(spend)` shape -- is withdrawn as of
   2026-08-19.*** `BudgetPort` wraps one provider call and refuses a spend that
   has not happened; the eval cap fires between cells, where the spend is
   booked and the cell holds an unknown number of LLM calls, so there is no
   honest estimate to pass. Two problems, two correct shapes. The eval budget
   contract is shaped from what discodon runs -- R6 of
   [`eval-extraction-discodon-requirements.md`](eval-extraction-discodon-requirements.md).

   **What check 3 actually needs is the other half, and it generalises past
   search.** Eval must hold no provider-specific parameter-to-cost table: the
   caller reports what it consumed in dimensions that name no provider --
   calls, weighted provider units *and the name of that unit*, money and
   currency. `Spend` here is already that shape (SR-E4), and eval owning an
   equivalent vocabulary rather than importing this one is deliberate: eval
   accounts for every metered external call -- image generation, Wolfram,
   YouTube -- not only search. R5 of the same document.
7. Existing web_search cassettes are keyed on the current wrapper's
   parameter hash; the rebuilt tool reshapes parameters, so this migration
   includes cassette re-capture (or a recorded key mapping) -- never silent
   reuse.
8. PR, merge; record acceptance of what binds it (D15).

***Correction 2026-08-19 -- verified against discodon's tree. Check 3's spend
half is largely satisfied already, by code that never touched this family;
check 10 has no site at all.***

**discodon has no 3tears dependency of any kind.** `3tears` appears in neither
its `pyproject.toml` nor its `uv.lock`, and no module imports `threetears`. The
Phase 5 preamble's consumption modes -- develop tracking develop, releases
pinning the whole family -- are true of metallm and have **no referent here**:
step 2 is not a pin to bump but a first adoption, and step 1's "lands upstream
first" governs a relationship that does not exist yet.

**Check 3, first half -- search spend inside the eval cost ceiling -- is already
true for the research path, and deliberately false for the standalone tool.**
Built without `BudgetPort`, and the route is worth stating because a migrator
reading step 6 would not expect to find it:
`external_pricing_for` (`discodon/eval/usage_capture.py`) resolves one
`ExternalCallPricing` per run from the research tool's depth
(`SEARCH_CREDITS_BY_DEPTH`) and the operator-declared
`config.tavily.usd_per_credit`; `blended_cost_roles` puts `external` into the
roles `EvalResult.cost_usd` sums **exactly when that card is priced**;
`runner.py` calls `on_cost(result.cost_usd)` after each saved cell, and
`EvalService` wires that to `EvalRunCostCap.record`. So a run whose operator
declared a credit rate stops on a ceiling that already counts its research
search dollars.

What is outside is outside on purpose: the standalone `web_search` tool's calls
land as `external` rows carrying credits and **no dollars**
(`RoleUsageLedger.add_unpriced_calls`), contributing nothing to the total, and
discodon names its own blocker -- action-seam dollars need the cassette
withholding widened to that seam first. Volume rather than dollars is bounded
separately, by `EvalRun.max_metered_calls`. One inconsistency a migrator will
hit: `discodon/eval/models.py` still describes `max_cost_usd` as capping "the
run's LLM cost", written before `external` joined the blend.

**Step 6's reshape is still owed, and for the reason already recorded** at
[`convergence-sequencing.md` Phase 1](convergence-sequencing.md#phase-1--foundations):
`EvalRunCostCap.check()` takes no estimate and reports `accumulated > ceiling`,
so it can only stop a run *after* the ceiling is crossed. Same names as
`BudgetPort`, post-hoc semantics. Re-verified today, unchanged.

**Check 3, second half -- replay -- is satisfied at the two seams D28 assigned to
discodon and unbuilt at the one this document proposes.** Action-seam
(`web_search`) and delivery-seam (research) cassettes are wired discodon-side.
The search-internal replay of step 5 exists on neither side:
`packages/search/src/threetears/search/` has no `replay.py`, and `RecordingStore`
appears nowhere in discodon.

**Check 10 has no site, and nothing on discodon's current course will give it
one.** `nats` appears nowhere in its code or its compose files; 49 modules
import `zmq`; no leaf is consumed. A check phrased as *before and after the
switch* needs a before, and there is none to preserve.

So discodon's honest position is not "not started": **step 6's property is
mostly met by other means, steps 3, 4, 5 and 7 have no started work, and steps
1 and 2 have no referent.**

**samsung** -- rides its planned phase-2 image-search work, not a
migration-for-migration's-sake.
1. Branch per its own conventions when that work starts.
2. Take `3tears-search[standalone]` (or supply its own transport), `[extract]`
   if needed; supply a `RecordingStore` over its SQLite plane.
3. Build image search on Call + Select with deep criteria and the new
   `media-contracts` facets; verify checks 2, 5, 9 (no fork, no torch,
   one-shot `asyncio.run()`).
4. Record acceptance (D15).

***Correction 2026-08-19 -- samsung's phase-2 image search is BUILT, it is not
built on the leaf, and that is a decision with a price tag rather than a gap.***

The work this step forecasts as "when that work schedules" shipped:
`curation/src/curation/discovery/phase_two.py` ranks image instances,
`discovery/images.py` is the provider seam (`ImageSearch`, `ImageQuery`,
`FoundImage`), and `discovery/artic.py` is its first provider.

**It does not use `3tears-search`, and not by forking it -- the two solve
different problems.** Phase 2 asks a museum's collection for instances of a
*named* work and settles identity by comparing title and artist above the seam.
It refuses to treat the provider's relevance score as evidence at all, having
measured a nonsense query returning ten real works scoring in the fifties, and
it spends nothing -- so no budget, no carriers, no ranking cross that seam.
Check 2's "without forking" clause has nothing to bite on. What it should have
asked is whether the leaf was **considered and rejected**, and it was.

**The rejection is recorded, and its currency is install weight rather than
capability.** `curation/pyproject.toml` states 3tears core is "deliberately
absent and is not expected", and confines `3tears-models` to an opt-in `eval`
group because the default install is the plane running beside `display` under
`MemoryMax=2G` (`deploy/curation.service`) and that one package would bring
`3tears-observe`, `3tears-media-contracts`, `anthropic`, three `langchain-*`
packages and `jsonschema` with it. Web-shaped traffic goes to one provider -- 
OpenRouter, via its web plugin -- behind a first-party client on a narrow seam.
**So the bar `3tears-search` must clear at this consumer is the size of its
transitive closure**, and no phrasing of check 2 says so. Its pin is also two
minors below anything adoptable: `3tears-models>=0.22.5,<0.23` against a family
at 0.26.1.

***And the leaf clears that bar with room to spare, which nobody had
measured.*** Installed from PyPI at 0.26.1 on CPython 3.14:

| Install | Packages | `site-packages` |
|---|---|---|
| `3tears-search` | 8 | 8.3 MB |
| `3tears-search[standalone]` | 14 | 10 MB |
| `3tears-search[standalone,extract]` | 30 | 75 MB |

The base closure is pydantic plus the two dependency-free family leaves -- both
`3tears-observe` and `3tears-media-contracts` declare `dependencies = []`, which
is D24's permitted floor doing exactly what it was written for. curation already
declares `pydantic` **and** `httpx`, so the marginal cost of
`3tears-search[standalone]` at that consumer is **three wheels, ~684 KB, and no
new transitive dependency at all**.

**The exclusion samsung recorded is sound for the package it was written about
and does not transfer.** `3tears-models` brings anthropic, three `langchain-*`
and jsonschema; the search leaf shares only the two empty leaves with it. What
kept the leaf out of that plane was never the leaf's weight -- it was that
nobody had asked, and check 2 gave no reason to.

**`[extract]` is the extra that must stay off a Pi: +16 packages, +65 MB**, of
which babel is 32 MB and lxml 19 MB, arriving `trafilatura -> courlan -> babel`.
Phase 2 has no use for it -- museum JSON, no HTML-to-text -- and the leaf refuses
correctly in its absence: importing `threetears.search.extract` succeeds because
`_load_extractor` defers the import, and calling it raises `LocalCapExceeded`
naming the extra and its remediation rather than an `ImportError` from the
middle of a pipeline.

Where the leaf would plug in is named in samsung's own code and is unbuilt:
`phase_two.py` records that nothing produces a `contemporary_web` candidate -- 
every instance it stores comes from a museum API and is `institutional`.

***Check 9 describes a caller samsung no longer is, and the real shape is the
harder one.*** The one-shot `asyncio.run()` callers are the legacy 2024 scripts
(`tvart.py`, `tv_api_check.py`, `display/tools/power_probe.py`) -- the modules
`curation` exists to retire. The plane that would call search is a long-lived
uvicorn process: `curation/src/curation/app.py` builds a FastAPI app with a
mounted MCP session manager, its HTTP routes are plain `def` so Starlette runs
them in its threadpool, and MCP tool dispatch crosses into sync code through
`asyncio.to_thread` (`curation/src/curation/mcp/server.py`). The calling code is
genuinely synchronous, which is the half that matters and it holds -- but "no
ambient event loop, no long-lived client" is **false of the process**. A leaf
proved only against a bare `asyncio.run()` would not have proved what this
consumer needs: a synchronous entry point reached from a worker thread while a
loop runs on the main one.

Check 5 passes vacuously and should not be counted: `torch` appears in no
manifest or lock in that repo, but nothing has driven the check, because the
leaf is not installed.

***What the four re-verifications have in common.*** Three months of consumer
commits moved every one of checks 1, 2, 3, 9 and 10, and none of the movement
was toward the step as written -- 1 and half of 3 came true by unrelated work, 2
was answered in the negative for a reason recorded only in the consumer's own
`pyproject.toml`, 9's premise expired, and 10 lost the "before" it compares
against. A consumer-repo check is a claim about a tree this repo does not watch;
it has to be re-read against that tree at the time of asking, and its cost is
paid by whoever asks late.

**Ordering within Phase 5:** metallm first (smallest, exercises the gutted
builtins), discodon second (deepest, exercises budgets + replay + corpus),
samsung when its image work schedules. In-family consumers (scrape,
context-save) already landed in Phase 2.

**Gate C (before the first pod-resident deployment of search):** the D13 wire
compatibility promise ruled formally; envelope asks (D18) released; identity
scope carried for search calls pod-side (the "stateless tool" classification
is invalidated by budgets/replay -- §5.4).

*2026-08-14 -- D13 now has a price tag.* The `extra="forbid"` stance was
deferred here as a considered trade, on the reading that a mixed-version fleet
is a deployment concern rather than a contract one. It has since been paid
once, in production and not by search: three days of total refusal on
cobalt-dev, from a field no caller populated, diagnosed only after the pod was
made to say why it was refusing (D18's correction). The 0.24.3 pruning closes
the *sender* half for the one sender there is, which is a mitigation and not
the ruling -- an ignore-unknown receiver would have degraded instead of
refusing, and that is the choice Gate C still owes. **Envelope asks "released"
also remains half true:** `deadline_seconds` is accepted everywhere and sent
by nobody, so a caller may now safely be taught to populate it against a fleet
at 0.24.3+, and Gate C should confirm that floor rather than assume it.

---

## 8. Deliberately open

- **OQ1 (Python floor)** -- ruled in principle 2026-08-04: discodon adopts
  3.14 (its declared floor is already `>=3.12`, so this is an interpreter
  switch plus verification, owned by discodon). Tracked until that lands;
  D25's avoid-3.14-only-surface intent stays as cheap insurance meanwhile,
  and the per-module-floor fallback is retired unless adoption hits a wall.
- **One bus or two** -- decides distributed-pacing reach post-convergence.
- **Final wire-boundary placement** -- D16 is the v1 answer; SR-L4 keeps the
  rest open.
- **Model-mediated producer detail** (D3) -- designed when samsung pulls.
- **D12 ratification** -- the robots/terms stance needs per-repo acceptance,
  not just this spec's proposal.
- ~~**A weighted provider unit has no name.**~~ **Closed the same day it was
  opened, 2026-08-19, by fixing it rather than carrying it.**
  `Spend.provider_units` was a bare `Decimal` while `Spend.__add__` refused to
  sum `money` across currencies -- the identical fabrication, unguarded, in the
  dimension right beside the one the guard was written for. `Spend` now carries
  `provider_unit` as `"<provider>:<unit>"`, composed once from
  `ProviderCapabilities.metered_unit` so a second spelling cannot appear and
  compare unequal, and the sum refuses a mismatch. Qualified by provider on
  purpose: two providers may both meter "credits" without those being one
  fungible unit.

  **Taken as a breaking change, deliberately.** `ContractModel` is
  `extra="forbid"`, so an added field is a wire break for any reader on an
  older exact version -- the exposure `_base.py` already documents. It was
  taken because 0.27.0 is bumped and untagged, the only pinned consumer does
  not call this surface, and a spend figure that silently sums two providers'
  units fails in the direction nobody audits: it reads as a smaller, plausible
  bill rather than as an error.

## 9. Requirements confidence

**High** for Phases 1-2: every requirement is traced to verified code
(2026-08-04), the consumers are known call sites, and the migration path
(metadata under a named key, identity preserved) is confirmed to survive the
wire. **Medium** for Phase 3: SR-F5's "a wiring line per consumer" is
estimated, not measured -- the cheapest raise is wiring a `RecordingStore`
over discodon's existing store as a spike before the record schema is cut --
and Aggregate/Select depth depends on samsung's phase-2 requirements holding
as written. **Open assumptions carried:** ~~SR-A4's SearXNG score semantics~~
(**settled 2026-08-12**, ahead of Gate B); the six-layer cut is proposed
vocabulary, not ratified type structure (mitigated by D23's naming rule); OQ1.
