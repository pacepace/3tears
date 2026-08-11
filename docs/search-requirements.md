# Search: What the Family Needs

**Status:** Draft for ratification — 2026-08-02, second pass 2026-08-03,
coherence pass 2026-08-03
**Companions:** `family-convergence.md` §4.14 records the *direction*;
`search-architecture.md` derives the *shape* and what each consumer adopts;
`search-spec.md` (2026-08-04) is the *spec* — it takes the §13 decisions as
vetoable rulings and is the buildable statement; `shared_search.md` sketches an
earlier *mechanism*, superseded where the two disagree. This states the *need*,
and per SR-M3 it is the cross-repo record.
**Relates to:** §4.13 (scraping), §4.2 (evals), open questions 1, 13, 15, 16 and 21

The second pass read the 3tears source rather than the consuming apps, asking
what the family already ships that this must reuse, obey, or ask to change. It
added section N (transport and egress), principle P9 and goal G13, corrected two
reuse claims that hold only pod-resident, closed the `media-contracts` question
with evidence, and found one consumer and one deployment axis that were missing.
Citations from that pass are `3tears/packages/...` and read 2026-08-03.

The coherence pass propagated two corrections back from
`search-architecture.md`, which is the newest of the four documents. **Spend is
every resource a call consumed, not only money** — section E is renamed and
SR-E1 widened, resolving a tension this document already carried between SR-E1's
"in money" and SR-D1/SR-D6/SR-I3. And **the Python floor is live rather than
decided**: §5.4, SR-L6 and §12 no longer assert it, because open question 1 has
changed shape — a blanket relaxation of core to ≥3.12 is off the table, and the
live options are moving discodon to 3.14 or a per-module minimum with a relaxed
subset. Nothing in this document turns on which way it goes; SR-L7 is why.

## Summary

Our products need information from the outside world, and six of them are
solving that separately. This document says what the family needs from a shared
search capability, the principles any answer has to hold to, and — in Part II —
the requirements that follow, traced to code in the six repos.

The one reframe worth reading for: **we are building information retrieval, not
web page lookup.** Consumers want a fact, an image that meets a spec, a source
that says a thing. A URL is a locator and a page is a carrier; neither is what
was asked for. That shifts extraction from an optional stage to part of the
deliverable, makes provenance a requirement rather than a nicety, and gives
model-mediated search an obvious home instead of an awkward one.

**What changes, by package** — the whole picture in one table, so nobody has to
reach §9 to learn whether their code moves:

| Repo / package | What happens |
|---|---|
| `3tears` `agent-tools` | `WebSearchTool` keeps its `threetears.web_search` name, its `TearsTool` ABC and its `ToolResult` shape. `_format_results` stops being the primitive: it becomes a Bind over structured results and carries them on `metadata`. Two envelope gaps close with it (§10, items 9-10). |
| `3tears` `scrape` | `page_finder` gets structured results with no change at its call sites — success check 4. |
| `metallm` | Both side-steps deleted, not wrapped: the raw SearXNG helper and the app-side trafilatura wrapper — success check 1. |
| `discodon` | Two internal implementations collapse to one; search spend enters the eval cost cap and research evals become replayable — success check 3. |
| `samsung` | Phase 2 image search is built on this rather than forked — success check 2 — from a synchronous caller on a `MemoryMax`-capped Pi. |
| `3tears` `core` | Nothing moves. Two shipped seams are consumed rather than reimplemented — `http_client` (traced, retried, circuit-broken outbound transport) and `egress` (which exit a call leaves by) — and the enforced no-bespoke-httpx rule gains a protocol form so a leaf can satisfy it without importing `core` (SR-N1, SR-L7). |
| `3tears` `media-contracts` | Nothing moves, and nothing is invented. It is already dependency-free and already carries the carrier taxonomy, `extraction_status`, and the `ObjectHandle`/`ObjectStore` pair; SR-C3's facets and SR-F5's port pin here. |
| a new leaf | The contract types: dependency-free and wire-serialisable — SR-L3, SR-L4 — and reaching no further down than `3tears-observe`, which is itself dependency-free (SR-L7). |

This proposes no contract, fields, packages, or sequencing. Part I is the whole
picture — a reader who needs the direction can stop at the end of it. Part II is
the derivation, and it is where the arguments are checkable.

---

# Part I — What we need

## 1. What we are trying to achieve

Goals in three groups: what we deliver, who we serve, how we run it.

### What we deliver

**G1. The unit is information, not a link.** A consumer asks for a fact, a
source, an image meeting a spec. What comes back may be a bare URL, a snippet,
extracted page text, or a structured record — four points on a **fidelity
ladder**, and the consumer says which rung it needs. Today every implementation
stops at "here are ten links and a snippet," which is why the two consumers that
wanted information built their own extraction.

**G2. Retrieval quality is the outcome.** Recall and precision of *information*
is what we are buying — the source that actually says it, the highest-resolution
rights-clear image, the paper rather than the press release about the paper.
Providers, extraction, fusion and reranking are means. Saying it this way is what
tells us when a stage earns its place and when it is machinery for its own sake.

**G3. Any carrier.** Pages, images, PDFs, video, datasets — carriers of
information, not separate products. **Ruled 2026-08-02:** images and arbitrary
data types are in scope, so the result core has to be carrier-neutral and the
carrier facets open (SR-C1). Samsung's image search is the near-term proof —
designed, records and lifecycle built, only the search itself missing.

The fidelity ladder is carrier-dependent. A page runs locator → snippet →
extracted text; a dataset runs locator → schema → sample → full download; video
probably runs locator → metadata → transcript. The rungs differ, the principle
does not — the consumer says how far to go, and Extract does whatever that
carrier's version of "get the information out" is.

**G4. Every piece of information carries a source you can re-check.**
Attributable to where it came from, groundable against what was actually
retrieved, and replayable later. Several of our products state things they
learned from the web; without this they are confident and unfalsifiable, and
their evals measure the web's drift rather than their own changes.

### Who we serve

**G5. One capability the family relies on.** Every 3tears module and every
consuming app gets search from one place, so an improvement lands everywhere and
a fix happens once. Today: eight call sites, four implementations, two of them
side-steps written *because* the shared one did not fit.

**G6. Humans, programs, and agents are equally first-class.** A person typing a
query, a program acquiring data, and an LLM invoking a tool are three real
callers. Every implementation today is shaped for the agent case — which is
exactly why both programmatic consumers hand-rolled their own.

**G7. Value at every layer; integrate at any of them.** A consumer takes one
provider call, or the whole pipeline, or anything between, and gets full fidelity
at whatever depth it stopped. Nobody rewrites to benefit, and nobody carries
weight they do not use.

### How we run it

**G8. Providers are pluggable, and self-hosting is first-class.** SearXNG on our
own hardware and a paid API are the same capability with different economics.
Two consequences worth naming: no product's decisions get made by a search
vendor, and no product is stranded when one changes its terms.

**G9. Spend, evals and telemetry are built into the seam.** A run's cost cap
includes what it spent searching — and "spent" covers the unpriced resources
too, since the constraints that bite on a self-hosted provider are quota and
wall-clock rather than money (§E). An eval can attribute a score change to a
search-config change. A slow turn is explainable.

**G10. Safe to depend on.** Every call site in §5.2 — across four repos — binds
to this.
It must never be the thing that stops an app shipping — bounded weight (the Pi is
the honest constraint), a stated versioning promise, explicit degradation when an
optional piece is absent.

**G11. Operable by a person.** Someone can see what it costs, why it is slow,
what it is doing and what broke, without reading the source.

**G12. Two deployment modes, neither privileged.** The same leaf runs
**embedded** — one synchronous process on a `MemoryMax`-capped Pi, no broker,
where the SQLite file *is* the store — and **pod-resident**, as a `TearsTool`
served over NATS through the registry proxy. This is not a concession granted to
the smallest consumer: the embedded mode is what discodon occupies for the whole
of its NATS convergence, so the library path carries the family's largest
consumer before it carries its most constrained one. §5.4 states the two modes
and which requirements each conditions, and §5.5 the reach axis that crosses
them.

**G13. Nothing new where the family already has it.** Most of what the
requirements below ask for is shipped somewhere in 3tears already — retry,
circuit breaking, tracing, a rate limiter, an exit selector, a secret resolver, a
store port, a wire-descriptor pattern, a dynamic tool-pod lifecycle. Several are
enforced, not merely available: `tests/enforcement/test_no_bespoke_reuse.py`
fails a build that hand-rolls an HTTP client, and
`test_cache_primitive_usage.py` one that hand-rolls a cache. So "reuse" here is
a constraint with a gate behind it, not an aspiration — and where a primitive
sits above the smallest consumer's dependency floor, reuse has to arrive as a
shape rather than an import (P9). §6 maps each layer to its existing home and
says, per row, whether that home is reachable in **both** deployment modes; two
of them are not.

## 2. What we are not trying to achieve

Naming these is what keeps a shared capability from turning into a platform.

- **Not an answer engine.** We retrieve information and say where it came from.
  Deciding what it *means* stays with the consumer — discodon's research tool
  synthesises, samsung's model proposes works. We do not summarise, judge, or
  conclude.
- **Not a search engine.** We do not crawl or index the web.
- **Not RAG.** Retrieval over the family's *own* content is `agent-memory`'s job.
- **Not an agent.** We do not decide what to search for. `scrape`'s `page_finder`
  is an agent that *uses* search, and it sits above this.
- **Not a scraper.** Hostile targets and heavy fetch belong to `3tears-scrape`
  (§4.13); we reach for that rather than grow one.
- **Not a home for app-specific ranking policy.** Criteria come from consumers.
- **Not a UI.**

## 3. How we would know it worked

Checks, not sentiments. They are numbered for reference and the numbers are
stable — new checks append rather than insert:

1. metallm's two side-steps are **deleted**, not wrapped.
2. samsung's image search is built on it **without forking** — the real test,
   because it is the consumer least like the ones that shaped today's code.
3. discodon's eval cost cap includes search spend, and a research eval can be
   **replayed** instead of re-issued against a changed web.
4. `scrape`'s `page_finder` gets structured results **without its callers
   changing**.
5. A Pi deployment installs it **without torch**.
6. A new provider is added **without touching a consumer**.
7. A new carrier type ships **without a coordinated release** across consumers.
8. The 3tears `WebSearchTool` builtin keeps its MCP name and its `ToolResult`
   shape while its formatting stops being the primitive — **an existing caller
   sees no break**, and structure arrives on `metadata`.
9. samsung calls it **synchronously, from a one-shot `asyncio.run()`** — no
   broker, no ambient event loop, no long-lived client.
10. The same leaf serves discodon **before and after** its NATS convergence —
    embedded first, pod-resident after — with no consumer-side rewrite at the
    switch.
11. The Adapter **passes `test_no_bespoke_reuse` without an exemption** and
    still installs on the Pi — one transport seam, two implementations, no
    waiver. Today those two demands contradict each other (SR-N1).
12. A deployment routes search egress **independently of the rest of its
    traffic**, and a result says which exit it left by (SR-N2). The check that
    bites: the self-hosted SearXNG and the Tavily API can take different exits
    in the same process.
13. A carrier facet a consumer needs is **found in `media-contracts`, not added
    to the search leaf** — or the reason it could not be is written down
    (SR-C3).
14. Search is reachable as a platform mesh tool, an HTTP API operation and an
    external MCP tool from **one contract and one binding** — no second result
    shape per face (§5.5).

## 4. Principles

These are about which direction things flow. They are the ones most likely to be
broken by accident, because every violation is locally reasonable.

### Flow direction

**P1 — No upward vocabulary.** A layer never requires a type defined above it.
The search call must not know what a corpus is; the provider adapter must not
know what a rerank criterion is.

**P2 — No lossy upward projection.** A fact knowable only lower down is preserved
upward *in usable form*, even where that layer has no use for it. Samsung records
the general case: `acquisition_method` rides the image record "because it is
knowable only where the instance was found… nothing downstream can recover which"
(`discovery_records.py:258-263`). This is P1's mirror and the harder one to
catch — nothing fails at the seam; it fails three layers up, later. An untyped
`raw` passthrough disclaimed as "never load-bearing" does not satisfy it.

### Independence

**P3 — Downward independence.** Every layer is usable without any layer above it,
at full fidelity. G7 stated as a constraint.

**P4 — Feature orthogonality.** No capability is conditioned on an unrelated one.
The acceptance test: *a consumer supplying its own rerank criteria must still be
able to constrain carrier type.* A pairing that cannot compose is a defect
needing written justification. Fused "pipeline" helpers are where this usually
breaks — bundle dedup, rerank and fetch, and wanting one means taking three.

**P5 — Cross-cutting concerns attach at their source.** Cost, telemetry, budget
and replay attach where the fact arises, not at a chosen layer. Spend arises at
the provider call and must be visible to a consumer that goes no further.

### Honesty

**P6 — Open vocabularies.** No layer closes the set of criteria a consumer can
express. Each understands the subset it can act on and passes the rest through
intact. A closed enum makes every future consumer's novel criterion a library
change — samsung's `rights_status` would have been exactly that a year ago.

**P7 — No collapsed scores.** Where several ranking dimensions exist they stay
separate and the consumer combines them. Samsung says why: `confidence` and
`quality_score` are separate because "a museum's own page is maximum confidence
and may be lower resolution than a gigapixel scan elsewhere. Collapsing them into
one number makes the trade invisible and the choice unexplainable"
(`discovery_records.py:240-244`).

**P8 — Explicit degradation.** When an optional layer is absent or a request
cannot be honored, the caller is told. Unranked results are *known* to be
unranked; an unsatisfiable criterion is named. Silence turns a missing stage into
a wrong answer. The family already has the shape: the tool pod's registration
path skips a tool whose configuration is absent and accumulates a *reason* per
skip rather than a silent omission
(`packages/agent/tools/.../serve.py:113-129`, written after
`THREETEARS_SEARXNG_URL` dropped `web_search` with no counter and no reason).

### Reuse

**P9 — Reuse arrives by protocol, not by import.** The smallest target sets the
dependency floor; a family primitive living above that floor is reused by
declaring its *shape* in the leaf and letting the host inject an implementation.
This is not a concession invented here — it is the move 3tears already makes
with itself, twice. `core.http_client` guards its upstreams with the breaker
from `threetears.models` while never importing it, by declaring
`CircuitBreakerLike` as a structural protocol: *"the injection keeps that
layering seam intact"* (`packages/core/.../http_client.py:16-21`).
`media-contracts` publishes `ObjectStore` and `MediaStorage` as
`runtime_checkable` protocols from a package with `dependencies = []`, *"so that
implementing or accepting a contract never inherits a feature package's
dependency closure"*.

P9 is what makes G13 and G10 compatible instead of opposed. Without it, every
reuse row in §6 that points into `threetears.core` — the transport, the exit
selector, the limiter, the cache primitive, the secret resolver — is a
dependency the Pi refuses and a norm the leaf breaks, and the capability has to
pick which rule to violate. With it, the enforced rule and the constrained
consumer are satisfied by the same design.

## 5. Who consumes this

### 5.1 Four axes that vary independently

Consumers do not divide by provider. They divide four ways at once, which is why
the capability has to decompose rather than take a shape fitted to a
representative caller.

| Axis | Values seen or anticipated |
|------|---------------------------|
| **Caller** | program · LLM via a tool call · person typing a query |
| **Carrier** | web page · image · PDF / document · video · dataset — open by ruling, not a fixed list |
| **Criteria depth** | none ("top 5") · shallow (recency, domain) · deep (resolution, rights, provenance class, publication type) |
| **Fidelity** | locator · snippet · extracted content · structured record |

Every cell is reachable. A person wanting rights-clear 4K images as domain
objects is one cell; an LLM issuing an unscoped query and reading prose is
another.

### 5.2 The call sites

| # | Consumer | Caller | Carrier | Criteria | Fidelity wanted |
|---|----------|--------|---------|----------|-----------------|
| C1 | discodon persona `web_search` | LLM | web | none | snippet |
| C2 | discodon research sub-tool | LLM | web | shallow | extracted content + corpus |
| C3 | **3tears `agent-tools` `WebSearchTool` builtin** (shared; metallm and C7 consume it) | LLM | web | none | snippet |
| C4 | metallm admin price lookup | program | web | none | extracted content |
| C5 | samsung discovery phase 1 | program (model-mediated) | web | shallow | structured record |
| C6 | samsung discovery phase 2 | program | **image** | **deep** | structured record |
| C7 | `3tears-scrape` `page_finder` | LLM agent (in-family) | web | shallow | locator, then content |
| C8 | **3tears `agent-tools` LangGraph context-save node** (shared; consumes C3's output) | program (post-turn) | web | none | extracted content, persisted |

Evidence: C1 `discodon/tools/web_search_tool.py`; C2
`discodon/tools/research/web_search.py`; C3
`3tears/packages/agent/tools/.../builtin/web_search.py`; C4
`metallm/api/src/api/v1/admin/models.py:948`; C5
`samsung/curation/src/curation/discovery/phase_one.py`; C7
`3tears/packages/scrape/src/threetears/scrape/page_finder.py:32,237-241`; C8
`3tears/packages/agent/tools/.../graph_nodes.py:126,129-168`.

**C8 was missed on the first pass and it is the one that touches §2's hardest
boundary.** `_DEFAULT_SAVEABLE_TOOLS = frozenset({"web_search", "web_fetch"})`
— a post-response graph node scans `ToolMessage`s from those two tools and
persists their content to the conversation context store, chunked, truncated at
4000 characters. Verified 2026-08-04, and the finding sharpened: **the node
is shipped but inert**, twice over —

- it matches `ToolMessage.name` by exact equality against those **bare**
  names, while the adapter binds tools under their namespaced names
  (`langchain_adapter.py:131` sets `name=tool.mcp_name()`, which returns
  `threetears.web_search`) — so the default set never matches;
- nothing in production wires `create_context_save_node` at all. Its only
  callers are its own tests, which pass bare names and therefore cannot see
  the mismatch.

Three consequences at requirements altitude:

- It binds on the **tool name as a string**, not on the result type — and the
  failure class this predicts has already fired, in the silent-off direction:
  written against bare names, bound namespaced, retention quietly does nothing.
  Anything that changes what search is called, or splits it per carrier,
  changes what gets remembered — invisibly, in either direction. A rename is a
  data-retention change.
- It is the seam where retrieved third-party content becomes *our own*
  content — which §2 assigns to `agent-memory` and declares out of scope. The
  boundary is real and the doc should keep it, but it is crossed by an existing
  in-family node, so "not RAG" is a statement about what search *owns*, not a
  claim that no path exists. It also widens SR-K4: the moment this
  node is wired correctly, web text is retained — the posture should be stated
  before that wiring, not after it.
- It reads `content` — the flattened string — so it inherits exactly the
  destruction SR-A1 exists to stop, and truncates at 4000 chars with no
  provenance. Under SR-A1 it becomes the second consumer that should read
  structure off `metadata` instead, and the first that gets *better* rather than
  merely unbroken.

Read the fidelity column: only two of the eight want what the shared builtin
returns. Five want extracted content or a structured record — and today four of
those five get there alone, while the fifth (C8) does not get there at all, and
its retention path turns out to be inert besides (above).

Three rows carry more weight than their size suggests. **C3 is the 3tears
builtin itself** — not metallm's, though metallm consumes it — so it is at once
a consumer of this capability and the thing `family-convergence.md` §4.14 has
already ruled must change: "the builtin `WebSearchTool` becomes a consumer that
renders results for LLMs — formatting is presentation, and it stops destroying
structure." It is listed as a consumer here because after that change it is
precisely one: a Bind. What it keeps is its identity — `threetears.web_search`,
the `TearsTool` ABC, the `ToolResult` shape — so this is a gutting, not a
replacement (success check 8). **C7** is a shared package consuming that same
builtin, so whatever it forecloses, scrape inherits — today, flattened text.
**C6** is not yet built, which is what
makes it informative: its lifecycle exists (`services/discovery.py:62,168-219`),
its result record is fully designed
(`persistence/discovery_records.py:236-282`), and phase 1 states the split — it
asks for works "and never for images, which is phase 2's job and a different
search entirely" (`phase_one.py:3-5`). A capability shaped around today's text
callers forecloses the consumer standing at the door with its requirements
already written down.

**Ruled 2026-08-02** *(was assumption A1)*: C6 is this capability. *Finding* an
image is search; *fetching* it is acquisition (§4.12). Arbitrary carriers —
video, datasets — are in scope on the same basis.

**A2 — settled 2026-08-03** *(was a vetoable assumption)*: person-typed queries
are in scope. It was inferred from the family having web UIs, with no call site
to point at. It no longer needs inferring: `TearsTool.face_api` makes an
external HTTP API surface a class attribute the family already ships (§5.5), so
the question is which faces this capability turns on, not whether the caller
could exist. "Queries are untrusted user content" is therefore the stated
posture rather than cheap insurance (SR-K2).

### 5.3 What the plot already shows

C2 wants prose *and* structure at once: text to its inner agent, plus a typed
per-URL corpus accumulated on the side (`research/web_search.py:301-321`) that
its grounding gate and relevance cull later read. C6 wants deep criteria from
three unrelated families at once — technical (resolution), legal (rights),
provenance (source class) — which samsung has already recorded as conflicting and
un-collapsible (P7).

No single opinionated result shape serves C1 and C6. That is the requirement
generating most of Part II.

### 5.4 Where they run — two modes

**Ruled 2026-08-02:** deployment shape is a requirements axis, not a design
detail deferred to the contract. C6 is why. samsung is a Raspberry Pi running two
systemd planes under a `MemoryMax` cap (`samsung/README.md:5`), and it has
already recorded package rejections on exactly these grounds — `3tears-core`,
because core's L1 is an in-memory SQLite *cache* while samsung's SQLite file must
*be* the store (`family-convergence.md`, "Where the framework doesn't fit"); and
`3tears-models`, because it "would pull 3tears-observe, 3tears-media-contracts,
anthropic, langchain-anthropic/openai/openrouter and jsonschema into the DEFAULT
install — the plane that runs on the Pi beside display under a MemoryMax"
(`samsung/curation/pyproject.toml:60-70`). A capability that does not state its
deployment constraints gets refused the same way, on the same evidence, after it
is built.

**Ruled 2026-08-02 (north star):** discodon converges off ZMQ onto NATS, and
search becomes a `TearsTool` in a pod there as it already can be in 3tears;
3tears and metallm run Yugabyte at scale. So the three topologies visible today
are not three permanent modes. They are two, plus one the family is leaving —
and requirements written against the one being left would age out with it.

| | **Embedded** | **Pod-resident** |
|---|---|---|
| Consumers | samsung (permanent); discodon (until convergence) | 3tears, metallm; discodon (after) |
| Topology | one process, systemd, `MemoryMax` | k8s, NATS, registry proxy, tool pods |
| Concurrency | **synchronous caller** | async |
| Broker | none | NATS |
| Store | the SQLite file *is* the store | Yugabyte + object-store |
| Reached as | direct in-process call | `TearsTool` dispatched over NATS |
| Holds provider credentials | the consumer | the pod |
| May depend on `threetears.core` | **no** — refused, on the record | yes |

**The last row is the one that decides the design.** It was implicit on the
first pass and it should not be: `core` is where most of the family primitives
this capability wants actually live — the traced HTTP transport, the egress
selector, the token bucket, the cache primitive, the secret resolver — and it
hard-requires `sqlalchemy`, `asyncpg`, `aiosqlite`, `cryptography`,
`pyjwt[crypto]` and `httpx`, at `requires-python = ">=3.14"`
(`packages/core/pyproject.toml`). Samsung's refusal of core is settled and on the
record, and it is a refusal on **shape** — core's L1 is an in-memory cache while
samsung's SQLite file must *be* the store. The weight argument about core is this
document's own inference, not something samsung recorded; its recorded weight
rejection was of `3tears-models`. Keeping the two apart matters, because the
weight half is the half that turns out to be softer than it reads (below).
The version floor points the same way today — discodon is on 3.12 —
but is *not* settled: open question 1 is live, between moving discodon to 3.14
and making the minimum a per-module statement with a relaxed subset. Either
outcome leaves the design here unchanged, because it rests on the rule that a
leaf four repos bind to cannot inherit the heaviest package's dependency
closure. P9 is the answer; SR-L7 is the requirement.

*Core's weight was checked on 2026-08-03 and re-checked by running on
2026-08-04, and it is an install cost rather than a runtime one.*
`import threetears.core` pulls none of sqlalchemy, asyncpg, aiosqlite, httpx,
cryptography, pyjwt or pydantic (a PEP 562 lazy surface); `egress` and
`coordination.token_bucket` import clean, and `http_client` eagerly imports
only `httpx`, its own subject. So the `MemoryMax` framing does not apply to
importing core, only to installing it.

And the install list is *partly* softer than it reads — corrected 2026-08-04:

- `aiosqlite` is unused anywhere in the monorepo (L1 uses stdlib `sqlite3`)
  and can simply go;
- `collections/flush.py` — the sole importer of `asyncpg` and one of three
  module-level `sqlalchemy` sites — is **not** test-only as the first check
  claimed. `collections/__init__.py` and `collections/base.py` import it,
  and six downstream packages reach it through them. Both libraries sit on
  the live `BaseCollection` path, so demoting them to extras requires
  refactoring those imports first.

`search-architecture.md` piece 5 carries the corrected per-dependency ruling. SR-L7 survives this, on the layering rule and the
unresolved Python floor rather than on runtime weight — which is the ground it
should be defended on, since a `pyproject` cleanup in 3tears would dissolve the
weight argument and leave the layering one untouched.

Three consequences worth stating rather than discovering:

**The wire boundary is real and unplaced.** The north star puts a NATS hop
somewhere inside §6's layer stack. *Which* boundary is a design question this
document does not answer — so the requirement is that any of them could be it
(SR-L4). That costs nothing at runtime on the embedded path; it costs design
freedom, which is the cheap thing to spend now and the expensive thing to
retrofit.

**Search stops being a stateless tool.** `agent-tools` classifies it as one
today — the call envelope's identity scope is optional because "pure stateless
tools (math, web search) do not require identity scope"
(`packages/agent/tools/src/threetears/agent/tools/server.py:389-391`). Under
SR-D2 (per-persona-per-day budget), SR-E1 (per-call spend), SR-F8 (replay key)
and SR-I1 (per-call telemetry) it is not stateless, and the pod path has to carry
scope for it. That is a stated assumption in 3tears this work invalidates, named
here rather than discovered during implementation.

**Two of §6's reuse rows are pod-only, and one of those was stated as closed.**
`TokenBucket` — §6's answer to SR-H4, and the reason that row says "so we do not
build a limiter" — is *"a distributed token-bucket rate limiter over NATS
JetStream KV"* whose constructor takes a `nats_client`
(`packages/core/.../coordination/token_bucket.py:1,16-18`). The embedded mode
has no broker by definition, so the primitive is unavailable exactly where
SR-D6's zero-cost-provider bound matters most. `3tears-epoch` has the same shape
and the first pass caught it; this one was missed. Both are now marked in §6's
new **Embedded?** column, which exists so a third one cannot hide.

Requirements this axis conditions, each flagged at its own entry: SR-E3 (spend on
the failure path), SR-F5 (who stores a recording), SR-G2 (deadline propagation),
SR-H1 (tuning without a restart), SR-H4 (rate pacing), SR-I4 (telemetry
delivery), SR-K1 (credential resolution), SR-M1 (versioning), SR-N1 (transport),
SR-N2 (egress).

### 5.5 How they reach it — three faces, orthogonal to the two modes

§5.4 asks *where the code runs*. `TearsTool` already carries a second, unrelated
question — *who may call it* — as three independent class-level flags
(`packages/agent/tools/.../base_tool.py:110-127`):

| Face | Default | What it means |
|---|---|---|
| `face_platform_tool` | `True` | reachable over the internal NATS mesh as a native platform tool |
| `face_api` | `False` | reachable as an external HTTP API operation |
| `face_mcp` | `False` | reachable as an external MCP tool |

*"The face flags govern reach only — ACL still governs authorization."* Two
things follow that change requirements rather than design:

**Assumption A2 is settled, not assumed.** Person-typed queries were vetoable on
the grounds that no current call site is one. `face_api` is the mechanism by
which one arrives, it already exists, and turning it on is a class attribute
rather than a project. So "queries are untrusted user content" (SR-K2) and
"queries may be sensitive" stop being cheap insurance against a hypothetical and
become the stated posture for a reach the family ships today. A2 moves from
ASSUMPTION to REQUIRED on that basis.

**One contract, not one per face.** Success check 14. An HTTP API operation and
an MCP tool are two more renderings of the same candidate set, which is what
SR-A1 already says — structure is the primitive, rendering is a Bind. The risk
is the ordinary one: a face gets added, someone shapes a response for it, and
the second result shape is born. Naming the axis is what makes that visible as a
regression instead of a feature.

Two smaller reach decisions the design will have to take, listed here so they are
not discovered: whether search is `skill_eligible` (surfaced in the skills
catalog, not just the default tool surface), and whether it stays inside the
`web` group alias — `WEB_TOOLS = {threetears.web_search, threetears.web_fetch}`,
*"opt-in because they hit the network"*
(`packages/agent/tools/.../aliases.py:15-17,70`). Both are ACL-visible surface,
and a per-carrier split (an image search tool) would change what an agent
granted `web` actually got.

## 6. Where value sits — the layers

Named rather than numbered, because `shared_search.md` uses L0–L3 for a different
cut and two numbering schemes in one directory is its own trap. Mapping given so
the two documents read together. Per G7 and P3, a consumer may stop at any row —
and where it stops is the fidelity it gets.

| Layer | Turns | Owns | `shared_search.md` |
|-------|-------|------|--------------------|
| **Adapter** | a family request → one provider's API | transport, auth, provider-native params and quirks, that call's spend record | L1 providers |
| **Call** | one query → one candidate set | request/result shape, capability declaration, criteria negotiation, failure classes | L0 contracts |
| **Aggregate** | many calls → one candidate set | dedup key, merge, fusion across queries and providers, fan-out accounting | L2 pipeline (part) |
| **Extract** | a carrier → the information in it | carrier-appropriate extraction, content provenance, probing | L1b fetch cascade |
| **Select** | candidates + criteria → an ordered, filtered subset | filtering, criteria negotiation, the cull, and a *slot* a ranker plugs into — never a ranking implementation (§4.14) | L2 pipeline (rerank slot) |
| **Bind** | candidates → what the caller consumes | typed domain objects, or prose for a model | L3 presentation |

Cross-cutting, attaching where the fact arises (P5): **spend**, **budget**,
**telemetry**, **concurrency and rate control**, **record/replay**.

Consumers use different subsets. C4 uses Adapter–Extract and binds itself; C2
uses everything and both bindings; C1 uses Adapter–Call–Bind; C6 needs all six
with the deepest Select; C7 reaches them through an agent loop. A design that
only works end-to-end serves one of them.

**What each layer reuses.** The names above are new; almost none of the
machinery under them is. Naming the existing homes is how this stays a contract
plus composition rather than a second platform.

The **Embedded?** column is the addition this pass forced. A home the Pi cannot
reach is not a reuse answer for the embedded mode, it is a reuse answer for half
the family — and the first pass recorded two of those as though they were whole.
"Via P9" means the leaf declares the shape and the host injects the family
implementation where it can; that is a real answer, not a hedge, but it is a
different one from "import it".

| Layer or concern | Existing home | Embedded? | So we do not build |
|---|---|---|---|
| Outbound transport (retry, breaker, tracing, timeout) | `threetears.core.http_client.TracedHttpClient` | via P9 | a retry loop, a breaker, a span, a timeout constant (SR-N1, SR-G1, SR-G4, SR-D3) |
| Which exit a call leaves by | `threetears.core.egress` — `EgressDriver`, `Direct/Proxy/Socks/Warp` | via P9 | an exit flag per call site (SR-N2) |
| Adapter capability declaration | `3tears-models` `capabilities.py` | via P9 | a second capability-metadata scheme (SR-B4) |
| Rate pacing | `threetears.core.coordination.token_bucket.TokenBucket` | **no — NATS KV** | a *distributed* limiter (SR-H4); an in-process one is still owed |
| Credential resolution | `threetears.core.security.secret_refs` — `scheme://locator` | via P9 | a second secret convention (SR-K1) |
| Carrier facets, object handles | `3tears-media-contracts` — `dependencies = []` | **yes** | image/document facets, a wire-descriptor pattern (SR-C3, SR-L4) |
| Bind, pod-resident | `agent-tools` `TearsTool` / `ToolServer` / registry proxy | n/a | a dispatch or discovery mechanism |
| Provider-per-spec pod lifecycle | `agent-tools` `DynamicToolPod` — load specs, build tools, register, republish, hot add/remove | n/a | a pod skeleton, or a second answer to SR-H1 pod-side |
| Bind, tool-call audit | `agent-audit`'s shared envelope | n/a | a second tool-call history |
| Contract types | the ratified contracts-leaf pattern | **yes** | a shared-types package with dependencies (SR-L3) |
| Replay store port | `media-contracts` `ObjectStore` / `MediaStorage` protocols; `DurableStore` direction (OQ16) | **yes** | a store protocol (SR-F5) |
| Replay store, embedded | samsung's `SqliteDurableStore` | **yes** | a store (SR-F5) |
| Replay store, pod-resident | `3tears-object-store` — streaming, S3-compatible, built for large artifacts | n/a | a blob path (SR-F5, SR-G5) |
| Telemetry, tracing, bounded retry | `3tears-observe` — `dependencies = []`, OTel behind `[otel]` | **yes** | a sink, a span, a backoff (SR-I4, SR-G4) |
| Tuning without a restart, pod-resident | `3tears-epoch` | **no — NATS** | a config-broadcast mechanism (SR-H1) |
| Heavy or hostile fetch | `3tears-scrape` | n/a | a scraper (§2) |
| Retrieval over our own content | `agent-memory` | n/a | RAG (§2) |
| Principle enforcement | `3tears-enforcement` scanners | n/a | P1-P9 living only in a reviewer's head |

**The `media-contracts` question is closed, in favour.** The first pass left it
open — samsung refused it as a transitive of `3tears-models`, not on its merits,
and nobody had checked. Checked 2026-08-03: `dependencies = []`, it is *already*
a hard dependency of `agent-tools`, and it already ships the facets SR-C3 asks
for. `MediaInfo` carries `media_category` (`"image" | "audio" | "video" |
"document"`), `mime_type`, `extraction_status` and `has_downloadable_data`;
`ObjectHandle` carries `object_id`, `s3_key`, `mime_type`, `size_bytes`,
`summary`, `category` plus `to_metadata()`/`from_metadata()`. So the honest
statement is stronger than "a direct pin may be fine": a direct pin costs
samsung nothing, and inventing image and document facets in a search leaf would
be building a second one of something the family already has.

**`3tears-observe`'s row was justified on the wrong evidence.** SR-I4 cites
samsung's rejection of `3tears-models` — which names `3tears-observe` among the
transitive weight — as *"direct evidence that owning a sink would have cost this
capability its most constrained consumer."* It is not: `observe` declares
`dependencies = []` and puts OTel itself behind an `[otel]` extra. It was
collateral in that list, not the weight. The recommendation survives on P1 and
P5 (a capability that owns a sink forces every consumer onto it, and both
existing consumers have their own), and the weight argument should be dropped
rather than repeated. The practical upside: `observe` is a dependency the leaf
can take outright, which is where `traced`, `retry_with_backoff` and
`spawn_background` come from at zero cost.

**One row was drifting from a ratified direction and has been corrected.** As
first written, Select owned "filtering, reranking, scoring, cull".
`family-convergence.md` §4.14 rules the opposite: *"Rerank is a stage with
existing homes, not part of search"* — `agent-memory` ships MMR, `3tears-models`
carries rerank capability metadata and pricing, and a cross-encoder arrives as a
models provider when a consumer pulls for it. §4.14 is the direction and this is
the derivation, so the derivation yields rather than the reverse: Select owns
the criteria negotiation and the cull and exposes a *slot*; the ranker is
composed in. The correction is recorded in §13 rather than made silently,
because a derivation quietly widening its own scope past the direction it
derives from is exactly the drift worth catching. P4 is the reason it matters in
practice: a consumer that wants the cull without paying for a reranker must be
able to have it.

## 7. What the consumer asks for, and what it is told

One mechanism the goals depend on, and it cannot wait for design because it
decides whether P4 survives contact. Consumers state two things, and get an
honest answer about each.

**Criteria — what to find.** Stated once, in one open vocabulary (P6),
regardless of which layer ends up satisfying them. Each criterion is then met by
**pushdown** (the provider filters), **local application** (a layer filters or
ranks), or **not at all** — and the caller is told which, per criterion (P8).
Pushdown becomes an optimisation — cheaper, fewer discarded results, sometimes
better recall — rather than a precondition for expressing the criterion. It is
what stops "you may filter by document type only if you skip reranking," and why
a consumer never has to know that `time_range` is a provider parameter while
`min_resolution` is a local filter. Those facts change per provider and per year.

**Fidelity — how far to go.** Extraction costs money and seconds, so the
consumer says which rung of the ladder it needs, and is told which it got. A
provider that already returned page text means the extraction step does nothing
and costs nothing (SR-A2). A consumer that only wanted locators never pays for
extraction it will not read.

## 8. What is still open

The decisions that most change the answer. Full list with recommendations in
§13.

- **Who stores a replay recording?** Replay itself is ruled in (SR-F3); where the
  bytes live and who expires them is not. Recommended: a store port the consumer
  supplies, with the hand-back bundle as one implementation of it rather than the
  primitive (SR-F5).
- **Local caps or provider refusal?** Samsung and discodon hold reasoned,
  recorded, opposite positions (SR-D5). This needs a ruling, not a merge.
- **How many score dimensions?** One `score` field forecloses C6 (SR-A4).
- **Model-mediated search — in or out?** Open question 21 (SR-B5).
- **Long-term retention of retrieved content.** Recording turns transit into
  storage, and the posture differs by carrier (SR-K4).
- **Where the wire boundary falls** in §6's layer stack, once search is
  pod-resident. Not answered here; SR-L4 is what makes any answer survivable.
- **One NATS bus or two?** Whether discodon and metallm converge onto one bus or
  keep separate ones decides whether SR-H4's pacing recommendation can actually
  close.
- **Does the family's no-bespoke-HTTP-client rule widen, or does search file an
  exemption?** The enforced rule points at `threetears.core.http_client`; the
  smallest consumer refuses core. Recommended: widen the rule to accept a
  declared transport protocol, which is the family's own `CircuitBreakerLike`
  move (SR-N1, P9). This is the only item here that asks 3tears to change a rule
  rather than to answer a question.
- **Does Select own ranking or compose it?** §4.14 already ruled rerank lives in
  `agent-memory` and `3tears-models`; §6 as first written gave Select ownership.
  Recommended: compose (§6, §13).

Ruled 2026-08-02: carriers are open, including images, video and datasets (G3);
searches must be replayable (SR-F3); deployment shape is a requirements axis
with two permanent modes (G12, §5.4); discodon converges onto NATS with search
pod-resident, and 3tears and metallm run Yugabyte at scale (§5.4).

Settled by the 2026-08-03 pass and no longer open: retries (the family shipped
the answer, SR-N1/SR-G4); whether person-typed queries are in scope (`face_api`,
§5.5); whether `media-contracts` is the home for carrier facets (yes, §6).

---

# Part II — What that cashes out to

Requirements traced to evidence, read 2026-08-02 and 2026-08-03, cited as
`repo/path:line`. Each is **REQUIRED** (a consumer regresses or breaks without
it), **DECISION** (consumers disagree or nobody has ruled — recommendation
given, owner picks), or **ASSUMPTION** (inferred, vetoable). Where
implementations disagree, the disagreement *is* the finding: the requirement was
never stated, so each site guessed.

Sections A–M are the consumer-derived requirements: what the eight call sites
need. **N** (transport and egress) and **O** (conformance to enforced norms) are
derived the other way — from what 3tears already ships and already gates — and
they exist because a capability that satisfies every consumer and fails the
family's own rules does not ship either.

## 9. Requirements

### A. What comes back

**SR-A1 (REQUIRED, Call/Bind — G1, G6).** Structured results are the primitive;
rendering for a model is one binding. C4 exists as a hand-rolled side-step
precisely because the shared builtin returns only formatted text
(`builtin/web_search.py:27-44`), and C7 inherits the same flattening.

The named deliverable this implies, since §4.14 already ruled it: the 3tears
`WebSearchTool` becomes that binding. `_format_results` moves behind Bind,
structure rides `ToolResult.metadata`, and the tool keeps its MCP name, ABC and
result shape so no existing caller breaks (success check 8). The metadata channel
is confirmed to survive the pod path end-to-end — `server.py:2053` passes
`tool_result.metadata` into the `CallResponse` — so this migration works
identically embedded and pod-resident.

**SR-A2 (REQUIRED, Call↔Extract — G1).** A result must be able to carry the
information itself, recording whether it arrived with the search response or came
from a later fetch. Tavily returns page text *in the search response* at no extra
credit versus `advanced` (`research/web_search.py:206`). A shape that puts
content only in a fetch result forces a Tavily consumer to re-fetch what it
already bought. Extract must be a **no-op when the provider already supplied the
content**.

**SR-A3 (REQUIRED, P2 — G4).** Results carry provenance: query, provider, the
provider's own identifiers, retrieval time. C2's grounding gate answers a
per-result question — "does this claim appear on the page it was cited from"
(`research/web_search.py:105-110`) — that no aggregate can answer.

**SR-A4 (DECISION, P7 — G2).** How many score dimensions, and whose?
Tavily returns relevance ∈ [0,1] and C2's cull ranks on it
(`research/web_search.py:27-44`). SearXNG returns an engine-fusion weight on a
different scale — *unverified; confirm against a live instance before ruling*.
C6 needs three orthogonal judgments.
*Recommendation:* a set of named, provenanced scores. Provider scores marked
non-comparable across providers; a comparable relevance exists only if Select
produced one.

**SR-A5 (DECISION, Aggregate).** Candidate set or corpus?
C2 and C4 built the same accumulation independently — C2 keyed by URL,
concatenating across searches and keeping the best score seen
(`research/web_search.py:104-111`, `:314-321`); C4 dedups URLs across concurrent
queries preserving order (`admin/models.py:967-975`).
*Recommendation:* Call returns a set; the corpus is Aggregate's named type with a
stated dedup key and merge rule.

### B. What goes in

**SR-B1 (REQUIRED — §7).** Criteria are stated once, in one open vocabulary,
regardless of which layer satisfies them.

**SR-B2 (REQUIRED — §7, P8).** Each criterion is met by pushdown, local
application, or not at all, and the caller is told which, per criterion.

**SR-B3 (REQUIRED).** An unsatisfiable criterion is reported, never silently
dropped. Precedent: RES-T4M9, a Tavily 400 from sending `time_range` with
absolute dates, fixed by stating absolute-wins precedence rather than silently
suppressing either (`research/web_search.py:218-234`).

**SR-B4 (REQUIRED, Adapter/Call).** Provider capability differences are
declarable and queryable, so a consumer branches before sending rather than after
failing — the pattern `3tears-models` already uses
(`packages/models/.../capabilities.py`). SearXNG has no `search_depth` and no
domain allow-list; it has `categories`, `engines`, `language`, `safesearch`,
`pageno`. Tavily has depth, domains, topic, dates
(`research/web_search.py:54-103`, `:196-234`).

**SR-B5 (DECISION — open question 21).** Is model-mediated search inside this?
Under an information-retrieval frame the answer gets easier than it looked. C5
retrieves information and cannot produce a provider result list at all — no
per-result scores, cost folded into token spend it already tracks
(`discovery/engine.py:74-88`). It is a retrieval path with different provenance,
which is a thing the contract already has to represent (SR-A3).
*Recommendation:* out of Adapter and Call, where a result list is the unit; in at
**Aggregate**, as a candidate producer — which is where its output already lands.
A consumer can then fuse model-mediated and API candidates without either
pretending to be the other.

**SR-B6 (REQUIRED — §7, G1).** The consumer states the fidelity it needs and is
told what it got. Extraction that a consumer will not read is money and latency
spent for nothing; extraction a consumer needed and did not get is a silent
partial answer.

### C. Carriers

**SR-C1 (REQUIRED, P6 — G3).** The result core is carrier-agnostic — identity,
provenance, scores, what fidelity is available. Carrier facets are additive and
open. A closed union of "web | image | pdf" is prohibited: the 2026-08-02 ruling
puts video and datasets in scope, and the list is explicitly not finite.

**SR-C2 (REQUIRED).** A consumer that does not recognise a facet ignores it
rather than failing, and adding a carrier requires no change at Adapter or Call
for consumers that do not use it. This is success check 7, and it is the check
that a closed union would fail.

**SR-C3 (REQUIRED).** Known facet needs, from real records — image: dimensions,
rights status, direct-file versus containing-page URL, and how the bytes are
fetchable (`discovery_records.py:274-281`, with `acquisition_method` load-bearing
per P2). Document/PDF: at minimum extraction status, since getting the
information out of a PDF is a different operation from getting it out of HTML.

*Facets are found, not invented (G13, §6).* `3tears-media-contracts` already
carries most of this list, dependency-free: `MediaInfo.media_category` is the
carrier taxonomy this section calls open, `extraction_status` is the
document/PDF facet named above, `has_downloadable_data` and `ObjectHandle` are
"how the bytes are fetchable". The requirement is therefore two-sided — the
result core stays carrier-agnostic (SR-C1), *and* a facet the family already
publishes is pinned rather than redeclared. Success check 13. What is genuinely
absent there and would be new: rights status, pixel dimensions, and the
direct-file-versus-containing-page distinction — three fields, in the package
that already owns the neighbourhood, rather than a fourth media vocabulary in a
search leaf.

**SR-C4 (REQUIRED, Extract, P3).** Extraction is carrier-dispatched, and a
consumer must be able to take search without any extraction — the Pi case and
C4's case are the same case.

### D. Budget controls

**SR-D1 (REQUIRED).** Budgets are expressible in **calls**, not only money.
Samsung says why: "A monthly credit limit cannot bound a single run that has
decided to search forever, and an estimate a run may freely exceed is not an
estimate" (`discovery/engine.py:42-49`).

**SR-D2 (REQUIRED).** Budget scopes are plural and not interchangeable:
per-persona-per-day (`web_search_tool.py:148-152`), per-invocation
(`research/web_search.py:186-192`), per-run (`engine.py:52`).

**SR-D3 (REQUIRED).** Provider quota exhaustion is distinguishable from a local
cap, and short-circuits. C2 trips a per-invocation breaker on HTTP 432/433 and
logs at ERROR because "a dead search backend is an outage, not a per-query
warning" (`research/web_search.py:249-265`).

**SR-D4 (DECISION — currently contradictory).** Does a failed search consume
budget? One repo answers both ways: C2 increments before the request
(`research/web_search.py:192`); C1 only after `raise_for_status()`
(`web_search_tool.py:271-272`). Neither records a decision.
*Recommendation:* budget follows the bill. But C2's fail-closed behavior is what
currently bounds retries against a degraded provider, so that bound has to move
somewhere explicit (SR-G4) in the same change.

**SR-D5 (DECISION).** Local refusal or provider refusal?
Samsung: "An engine reports what it spent; it never decides whether it may… never
a local sum crossing a number, because a local tally that fails open is
indistinguishable from one that works" (`engine.py:14-18`). Discodon does the
opposite. Two reasoned positions in conflict.
*Recommendation:* both, with distinct roles — local caps bound a run's *shape*
(overrun is a defect); the provider's refusal bounds *money*. Needs a ruling.

**SR-D6 (REQUIRED — G8).** A zero-cost provider still needs bounding. SearXNG's
failure mode is upstream rate-limiting or a ban, not spend, and a mechanism keyed
only on cost never fires for it (SR-H4). Self-hosting is first-class only if it
is protected too.

### E. Spend accounting

*Spend is every resource a call consumed, not only money* — dollars, wall-clock,
provider quota and call count, bytes moved. Modelling it as currency is the trap,
because the two constraints that bite hardest here are unpriced: a self-hosted
SearXNG costs nothing and fails by ban rather than by bill (SR-D6), and a call
that returned nothing against an already-spent budget is pure latency with zero
coverage gain (SR-I3). SR-E5 and SR-E6 are the money-specific entries; the rest
of this section holds for any resource, and so does SR-D1's "budgets in calls,
not only money".

**SR-E1 (REQUIRED, P5 — G9).** Spend is attributable per call and observable
from any layer — in money where the call is priced, and in wall-clock, provider
quota and call count whether it is priced or not. Discodon's ledger has moved
since the first pass (verified 2026-08-04) and now proves the requirement
rather than motivating it:

- search calls are counted always, and priced when the operator declares a
  rate — `ExternalCallPricing`, per-depth credit weights, configured
  `usd_per_credit` (`eval/usage_capture.py:61-127`), fed by the count at
  `research_tool.py:2869-2871`;
- its eval surface still warns that "max_cost_usd does not bound external
  search quota" (`web/mcp/eval/runs.py:184`) — closing that residual gap is
  success check 3.

Replay adds a second spend fact, and the two must never share a field:
execution spend binds budgets, recorded spend feeds cost models
(`search-spec.md` D27).

**SR-E2 (REQUIRED).** The count a cap enforces and the count a bill prices are
one number. Samsung derives `searches_used` from the priced records because "a
`searches_used` field beside a priced spend record is two tallies of the same
event, free to disagree, and the disagreement would surface as a cap that held
while the bill said otherwise" (`engine.py:106-115`).

**SR-E3 (REQUIRED).** Spend survives the failure path — "a run that broke halfway
still incurred whatever it incurred, and a failure path that dropped it would
under-report the month by exactly the amount the failures cost"
(`engine.py:118-127`).

*Mode-conditioned (§5.4).* This does not hold pod-resident today. The tool
server's exception branch builds `CallResponse(success=False, content="",
error=...)` with **no metadata** (`server.py:2071-2076`), so whatever the adapter
spent before the failure is discarded at the wire. Embedded, the consumer can
still see it. This is an ask on `agent-tools`, not a search-side fix — §10, item
9 — and it is cheap now and expensive once every §5.2 consumer has bound.

**SR-E4 (REQUIRED — live defect).** Weighted units must be accounted. Discodon's
persona tool bills every search as one unit (`_check_budget` at its `cost=1`
default, `web_search_tool.py:224`; `tools/base.py:1390`) while
`search_depth="advanced"` spends two Tavily credits — and its docstring says the
budget exists "to manage shared API credits" (`web_search_tool.py:7,57`). The
weighted primitive exists and `youtube_tool.py:216` uses it. Bite scope,
verified 2026-08-04: the tool's own default depth is `basic`, so the 2× fires
only where an operator sets `advanced` per-entity. And the eval-side ledger
now weights correctly (`SEARCH_CREDITS_BY_DEPTH`), so the daily budget and
the ledger disagree exactly there.

**SR-E5 (REQUIRED).** Cost granularity is per-request for some providers, and the
model must not imply per-result pricing. Samsung: "The fee is charged per search
*request*, not per result: one, three, five and ten results all bill identically.
A deployment that lowers the result count therefore saves nothing and sees less"
(`phase_one.py:17-21`) — a knob that looks like a cost control and is not.

**SR-E6 (DECISION).** Self-hosted cost: zero, or amortised infrastructure?
*Recommendation:* zero for spend attribution, with SR-D6's rate/quota dimension
carrying the real constraint. A synthetic per-query infrastructure cost corrupts
cross-provider comparison in the other direction.

### F. Evals and reproducibility

**SR-F1 (REQUIRED — G9).** Search parameters participate in eval identity so a
score delta is attributable to a config change — already true and load-bearing
(`canonical_digest` at `eval/identity.py:126`, applied to the variant key with
`resolved_tool_configs` at `:209-228`). The parameter object must therefore be
canonically serialisable — and that canonical form has two consumers that must
agree, this identity and the D26 replay key, so it is a public contract
feature, not a replay internal (`search-spec.md` §3.1).

**SR-F2 (REQUIRED).** Eval runs against a quota separable from production's, with
sharing explicit rather than a fallback. Discodon designed exactly this (EVL-TQ7K,
`discodon/config/sections/tavily.py`): an optional eval-scoped key so "an eval
search burst can never exhaust the quota or trip the breaker protecting live
personas' research," with unset meaning a *documented shared* quota.

**SR-F3 (REQUIRED — ruled 2026-08-02; G4).** A search must be replayable.
Record/replay attaches as a cross-cutting concern at Adapter and Call (P5), not
at whatever layer happens to be a `Tool`. That placement is the lesson of the
gap as first found: discodon's cassette layer records and replays at
`Tool.act()` (`eval/cassette_proxy.py:12-18`; the proxy's `act` at
`:313-360`), C1 is a `Tool` and is replayable, and C2's sub-tool deliberately
is not — "no Tool ABC overhead" (`research/web_search.py:3-5`). Replay was
attached to a class hierarchy, so a component that left the hierarchy silently
left reproducibility.

*Updated 2026-08-04.* Discodon has since closed the character-eval half of
that hole from above: a **delivery seam** freezes what research handed to the
character world (wired the same day this update was written), and its design
record explicitly rejects freezing the sub-tool's individual queries for that
use. The lesson stands unchanged — and it scopes this requirement honestly.
Search-internal replay serves what the coarse seams cannot:

- re-running the research *pipeline* (grounding gate, cull) against a frozen
  web;
- provenance re-checks and re-search diffs;
- programmatic determinism for consumers that re-issue the same requests.

Character-eval freezing is better served above. `search-spec.md` D28 records
how the seams compose.

**SR-F4 (REQUIRED).** What is recorded must rebuild the corpus, not merely the
rendered text — otherwise a replayed run cannot re-run its grounding gate.

**SR-F5 (DECISION — who stores the recording).** Search knows *what* to record
and *how to key it*; it does not know how long to keep it. That question belongs
to whatever needed the recording: an eval run's recording should live exactly as
long as the eval run, and search cannot see that lifecycle. The same argument
runs for privacy — a recording holds user-supplied queries (SR-K2) and retrieved
third-party content, so retention and redaction are the consumer's policy, and a
capability that persists them inherits an obligation it cannot discharge (P1).

Three shapes, and the difference matters:

- **Search owns a store.** Rejected. It forces every consumer onto one backend,
  contradicting SR-I4's decision on the same question for telemetry, and it puts
  the TTL choice in the one place that cannot make it.
- **An opaque bundle handed back and later handed in.** Right instinct, wrong
  primitive. A research run makes ~20 searches across nested inner-agent rounds
  (`research_tool.py:83-90`); a bundle has to be accumulated and returned through
  layers that have no other reason to carry it, which is P5 inverted — the fact
  arises at the provider call and would travel up through everything.
- **A store port the consumer supplies.** *Recommended.* Search defines the
  record type and the key, and writes through a port; the consumer wires its own
  store. The recording is written where it happens, and lifecycle sits with the
  owner. All three candidate consumers already have a durable store, so the
  burden is a wiring line, and the family already has a `DurableStore` protocol
  direction (open question 16) for the port to follow.

  *And a shipped port to follow it with.* `3tears-media-contracts` publishes
  `ObjectStore` as a `runtime_checkable` Protocol from a package with
  `dependencies = []` — *"streaming by contract: writes consume an async byte
  stream and reads yield one, so a multi-GB artifact never has to sit whole in a
  pod's memory,"* with *"the key opaque here"* and tenant scoping built above it.
  That is the port this requirement describes, already dependency-free, already
  satisfied by `3tears-object-store` pod-side, and satisfiable by samsung's
  SQLite plane embedded. Its streaming shape also discharges SR-G5 for
  recordings, which is where the byte problem is worst — a replay recording of an
  image or a video search is the largest thing this capability will ever write.
  Preferring it over inventing a search-local port is G13; preferring it over
  `DurableStore` is that one is published and conformance-testable today while
  the other is still open question 16.

The bundle is then one *implementation* of the port — an in-memory store the
consumer serialises — which is what you want anyway for out-of-process or
portable replay. Keeping it as an implementation rather than the primitive means
the in-process family pattern does not pay for the portable case.

One shape decision inside this: the envelope should be **typed and the payload
versioned**. The consumer sees id, created-at, provider, key, size and schema
version — enough to expire, purge, index and account for it — while the payload
stays search's business. That gives lifecycle management without schema coupling,
and makes schema evolution search's problem (SR-M1) rather than a shared one.

*Mode-conditioned (§5.4).* A Python protocol object cannot cross a NATS hop, so
"the consumer supplies a port" is literally true only embedded. Pod-resident, the
consumer supplies a store *reference* — a bucket, a table, a URI — which the pod
resolves to its own implementation of the same port. The abstraction survives the
mode change; only the wiring differs. Embedded, samsung's `SqliteDurableStore` is
the implementation; pod-resident, `3tears-object-store` is. That is the
recommendation's strength rather than a hole in it: one decision, two
implementations, no consumer rewrite at the switch (success check 10). What it
does require is that the reference and the record type are wire-serialisable —
SR-L4.

*On the specific questions:* how long to keep it — as long as the thing that
needed it, which usually means cascade-delete with the owning run rather than a
clock. When to purge — same event; a TTL is a backstop for orphans, not the
primary mechanism (discodon already runs one for research traces,
`research_tool.py:351`).

**SR-F6 (REQUIRED).** Recording is opt-in per call. A recording carries full
retrieved content (SR-F4), most calls will never be replayed, and paying that
cost on every call is waste that shows up as storage and bytes moved.

**SR-F7 (REQUIRED).** A replay miss is an error, never a silent live call.
Discodon has the precedent — a `CassetteMiss` raised on lookup failure
(`eval/cassette_proxy.py:137`, raised at `:330`). Falling through to the network would let an eval
go live without saying so, and its trend line would then be measuring the web.

**SR-F8 (REQUIRED, P2).** The replay key is derived by search, because only
search knows what varies — provider, query, resolved parameters, profile digest.
Discodon already computes the analogous digest for eval variant identity
(`eval/identity.py:126`). A consumer-derived key would go stale the first time a
provider parameter was added.

### G. Performance

**SR-G1 (REQUIRED — G11).** Timeouts are configurable, not constants. Four
implementations, three values, none operator-tunable: 30s class attribute
(`web_search_tool.py:39`); 15s constructor default never wired to config
(`research/web_search.py:59`, with construction at `research_tool.py:1412-1414`
passing depth and caps but no timeout); 15s hardcoded
(`builtin/web_search.py:53`); 15s hardcoded (`admin/models.py:951`).

**SR-G2 (REQUIRED).** A search timeout is derivable from the caller's remaining
deadline, not fixed independently of it. C2's caller runs under a per-call LLM
bound and a run-level conclusion deadline explicitly floored at the per-call
timeout (`research_tool.py:647-707`, `:762`).

*Mode-conditioned (§5.4).* Unimplementable pod-resident as the envelope stands.
`CallRequest` carries `tool_name`, `tool_version`, `arguments`, `context` and
`proxy_assertion` and nothing else (`server.py:376-417`), and it is declared
`extra="forbid"`, so a deadline cannot even be smuggled through `arguments`
without lying about the tool's own input schema. `MCPToolDefinition.timeout_seconds`
is the *tool's* declared ceiling, not the caller's remaining budget — a different
quantity. Honouring SR-G2 pod-side means amending the shared envelope: an ask on
`agent-tools`, stated here rather than assumed free (§10, item 10).
*Corrected 2026-08-10:* at 0.23.11 `CallRequest` also carries `result_subject`
(durable long-call delivery) — still no deadline field, so the finding stands —
and `ToolManifestEntry.timeout_seconds` has joined `MCPToolDefinition`'s in the
tool-ceiling category, now server-enforced. Same different-quantity distinction;
§10 item 10 carries the composition consequence.

**SR-G3 (REQUIRED).** No blocking IO on an async path. The builtin's `execute()`
is `async` but calls a synchronous `httpx.Client` (`builtin/web_search.py:50-56`,
called at `:122`); `web_fetch.py` has the same defect plus a `time.sleep` in its
retry loop (`shared_search.md:38-40`).

**SR-G4 (REQUIRED — was DECISION; ruled by reuse, SR-N1).** Bounded transport
retry at Adapter. None of the four implementations retry today; C2 tells the
model to retry in prose (`research/web_search.py:280`), spending an LLM round to
redo an HTTP call. This was written as an open decision on the first pass. It is
not one: `threetears.core.http_client` already implements the recommendation as
the family's sanctioned transport — finite `max_attempts` (*"forever-retry is
wrong for a request"*), exponential backoff between `initial_backoff` and
`max_backoff`, retrying connect errors, timeouts and 5xx while 4xx does not.
Adopting it is SR-N1. Interacts with SR-D4: if budget follows the bill, a
retried attempt that never billed must not count, so the retry boundary and the
budget increment have to sit on the same side of the transport seam.

**SR-G5 (REQUIRED).** Resource bounds are part of the contract — byte caps,
content-type gates, streamed downloads, not an unbounded `resp.text`, which is a
memory incident on a `MemoryMax`-capped host (`shared_search.md:41-42`) and
applies with more force to images and PDFs than to the HTML the current code
assumed.

### H. Concurrency and rate control

**SR-H1 (REQUIRED).** Concurrent calls are boundable and tunable without a
restart. C2 dispatches under an `asyncio.Semaphore` rebuilt per run so a hot
update lands (`research_tool.py:83-90`, `:768-771`); C4 fans out with an
unbounded `asyncio.gather` (`admin/models.py:967`).

*Mode-conditioned (§5.4).* `3tears-epoch` is the family's answer pod-resident,
and serves both pod deployments after convergence. It serves neither embedded
one, so this needs a second mechanism — a config reload on signal, or an explicit
ruling that a restart is acceptable on a single-process deployment. A default
matters as much as the tuning: whatever ships must be safe under a `MemoryMax`
cap without anyone tuning it first (SR-L6).

*Pod-resident there is a cheaper answer than epoch, and it is already written.*
`agent-tools` ships `DynamicToolPod`, a generic base whose whole purpose is
"load a set of specs, build one or more `TearsTool` per spec, register them on a
`ToolServer`, publish the registration manifest, serve, and — at runtime — add /
remove a spec's tools and re-publish"
(`packages/agent/tools/.../dynamic_pod.py:1-20`). A search pod is one tool per
configured provider *instance*, which is exactly a spec; retuning or adding a
provider is a spec change with a republish, not a config broadcast and not a
restart. It composes existing primitives by declared intent — *"it does NOT
reimplement a serve loop, a manifest publish, or a registry handshake"* — so
naming it here costs nothing and forecloses a bespoke pod skeleton. epoch stays
the answer for values that are not per-spec.

**SR-H2 (REQUIRED).** Two bound scopes, both real: within one batch, and across
simultaneous runs (`research_tool.py:328,376`).

**SR-H3 (REQUIRED).** One call's failure must not poison its siblings in a
concurrent batch — handled and reasoned in C2 (`research_tool.py:106-114`).

**SR-H4 (DECISION — G8).** Rate limiting: pace, or react to 429s?
All react; none pace. An unbounded fan-out at a shared self-hosted SearXNG (C4)
is the case most likely to get our own instance blocked upstream.
*Recommendation:* client-side pacing per provider *instance*, at Adapter. The
shared instance is what is at risk, and no single consumer sees the aggregate
load.

*Correction to the first pass.* That pass named
`threetears.core.coordination.token_bucket.TokenBucket` as the primitive and
recorded §6's row as "so we do not build a limiter". Read 2026-08-03, it is *"a
distributed token-bucket rate limiter over NATS JetStream KV"* — the constructor
takes a `nats_client` and every claim is a CAS read-modify-write against a KV
bucket. It is the right primitive for the *shared* bound and the wrong one for
the embedded mode, which §5.4 defines as having no broker. So the requirement is
two mechanisms, not one: a **distributed** bucket on `TokenBucket` where a bus
exists, and an **in-process** limiter that holds on a single process with no
NATS. The second is small, but it is genuinely owed rather than reused, and
recording it as reuse would have produced a Pi that paces nothing.

*Keyed on the exit, not only the provider (SR-N4).* A ban is issued against an
address. Two deployments sharing a SearXNG instance but leaving by different
exits are two rate-limit subjects; one deployment reaching two providers through
one exit is one. The bound is per `(provider instance, egress)`, which the first
pass could not state because egress was not in scope.

*Mode-conditioned (§5.4), and this one does not fully close.* Client-side pacing
assumes one process sees the aggregate. Deployments that share no state cannot
pace collectively, and the shared SearXNG runs with its limiter **off** —
recorded as an ops knob already correct (`family-convergence.md` §4.14). After
convergence two of the three consumers are pod-resident and can share a bucket
*if they share a bus*; the embedded Pi never can. So the honest answer is
layered: pace where a bus makes it possible, and turn the shared instance's own
limiter on, because that is the only bound that covers every consumer. Whether
discodon and metallm land on one bus or two is open (§8) and decides how much of
this the client side can carry.

### I. Telemetry

**SR-I1 (REQUIRED — G11).** Every call is individually recorded: query, scoping
parameters, result count, duration, error — so "the facet that failed is visible
per trace, not just in Loki" (`ResearchSearchRecord`,
`discodon/logging/models.py:232-247`).

**SR-I2 (REQUIRED).** Search wall-clock is separable from model wall-clock in any
run that mixes them (`ResearchRoundRecord`, `logging/models.py:250-275`).

**SR-I3 (REQUIRED).** Calls that returned nothing because a budget was already
spent are counted separately from calls that did work — `exhausted_calls`, "pure
latency waste with zero coverage gain" (`logging/models.py:258-261`).

**SR-I4 (DECISION, P5).** Emit telemetry, or return records?
C2 persists a trace document with a TTL to its own store
(`research_tool.py:351,794-806`); C5 returns spend records and holds no sink
(`engine.py:90-104`).
*Recommendation:* return records, emit nothing. A capability that owns a sink
forces every consumer onto it; both existing consumers already have their own,
and `observe` integration then belongs to the host, per the zero-dep-core
pattern. samsung's rejection of `3tears-models` names `3tears-observe` among the
transitive weight it refused (`samsung/curation/pyproject.toml:60-70`).

*One leg of that argument does not hold and should not be repeated (§6).*
`3tears-observe` declares `dependencies = []` and puts OTel behind an `[otel]`
extra, so it was collateral in samsung's list rather than the weight — taking it
directly would have cost samsung nothing. The recommendation stands on P1 and P5
alone, which is enough: a capability owning a sink forces every consumer onto
it, both existing consumers already have their own, and a sink is an opinion
from above. The corrected reading also *helps* — `observe` is a dependency the
leaf may take outright, and it is where `traced`, `retry_with_backoff` and
`spawn_background` come from.

*Mode-conditioned (§5.4).* Pod-resident, "return records" means they ride
`ToolResult.metadata` — confirmed to survive the hop (`server.py:2053`) — and
they must therefore be wire-serialisable (SR-L4). Note also that `agent-audit`
already envelopes every tool call pod-side, so the returned records are a second
account of the same event: the two need a stated relationship rather than both
growing independently, which is SR-E2's argument applied to telemetry.

### J. Failure semantics

**SR-J1 (REQUIRED — G10).** Failure classes are distinguishable, not merged:
rate-limited, quota-exhausted, auth-failed, timeout, transport error, malformed
response, zero results. C2 distinguishes all seven and gives each a different
instruction (`research/web_search.py:246-292`), including the deliberate split
between "retry" for timeouts and "give up" for transport failures. Errors carry
remediation where the cause is known and fixable — `shared_search.md:104-107`
makes this point for SearXNG's 403-when-`json`-missing, the #1 setup failure.

**SR-J2 (REQUIRED).** Zero results is a success. All implementations agree; pin
it so no future one disagrees.

**SR-J3 (DECISION, P1).** Errors as values or exceptions?
C2 returns instructional prose to a model; C1 returns
`ActionResult(success=False)` (`web_search_tool.py:243-259`); C5 raises, carrying
spend (`engine.py:118-127`); the builtin sniffs a string prefix
(`builtin/web_search.py:123`).
*Recommendation:* typed exceptions carrying spend (SR-E3), with prose at Bind.
The prose is prompt engineering — tuned per persona and per inner agent — and a
lower layer emitting it has taken an opinion from above.

*With a clause the first pass left out, because without it this recommendation
destroys SR-E3.* §10's item 9 is narrower than stated: the tool server carries
`tool_result.metadata` into `CallResponse` on the **handled**-failure branch —
a `ToolResult(success=False, metadata=…)` arrives intact
(`server.py:2050-2056`) — and drops it only when an exception escapes
`tool.run()` (`server.py:2071-2076`). So an exception that reaches the pod
boundary is precisely what discards the spend. **Bind must catch every typed
exception and render it as a failed `ToolResult` carrying the spend on
`metadata`; an exception must never cross the wire.** With that clause, SR-E3
holds pod-resident today and §10.9 stops blocking it *for search* — it remains a
real gap for every other pod-served tool, and the ask on `agent-tools` stands on
that broader ground rather than on this capability's need.

### K. Security, privacy, and conduct

**SR-K1 (REQUIRED).** Credentials resolve through the consumer's secret handling;
the capability must not read environment variables itself
(`web_search_tool.py:186-192`).

*Mode-conditioned (§5.4) — and once the family's own mechanism is named, the
modes stop disagreeing.* The first pass read this as an inversion: embedded the
consumer supplies, pod-resident the pod holds, because holding the provider
credential so callers never see it is a large part of what a tool pod is *for*.
Both halves are true and neither needs a different rule.
`threetears.core.security.secret_refs` is the family's canonical answer — *"a
secret is referenced by a `scheme://locator` string, never by value; the value is
resolved at use time by the backend the scheme selects — so the secret never
lands in a config file or DB and never sits in a long-lived process variable"*
— with `env://NAME` as the dev backend, `k8s://rel/path` as the prod shape over
a projected Secret volume, `vault://` and the cloud managers reserved, and
`register_scheme` for an app's own (scriob registers one for an
encrypted-at-rest deploy key). Resolution failures raise naming the *reference*,
never the value.

So the requirement restates cleanly and identically in both modes: **the
capability accepts a secret reference or an already-resolved value from its
host, and never reads ambient configuration itself.** Embedded, samsung passes
whatever its plane resolves. Pod-resident, the pod passes a ref from its own
deployment config and no secret crosses the bus. The pod is a consumer that
supplies from deployment config — which is what the first pass concluded, but
the mechanism makes it one rule instead of two, and `env://` shows the real line
is *who resolves*, not *whether an environment variable is ever involved*. This
is the current builtin's shape already: the host reads
`THREETEARS_SEARXNG_URL` and passes `base_url` in
(`packages/agent/tools/.../serve.py:100-129`); the tool reads nothing.

**SR-K2 (DECISION — no longer gated on A2).** Are queries sensitive?
A query can carry user-supplied conversational content and is recorded verbatim
today (`logging/models.py:243`). metallm ships a PII sanitisation wrapper it is
contributing. The first pass gated this on assumption A2 (person-typed queries
in scope, inferred from the family having web UIs). §5.5 removes the gate:
`TearsTool.face_api` makes an externally-reachable HTTP surface a class
attribute rather than a hypothesis, so this is a live posture question about a
reach the family already ships, not insurance against a future one. C8 sharpens
it differently than first written: the persistence path is shipped but inert
today (§5.2 — the bare-name/namespaced-name mismatch), so the posture can
still be stated *before* the first byte is retained rather than after.
*Recommendation:* treat queries as user content — retention governed by the
consumer's policy, the capability required only to make the query available for
redaction rather than to redact on its own (P1: redaction policy is an opinion
from above).

**SR-K3 (REQUIRED).** A self-hosted base URL is an internal endpoint. SSRF-shaped
risks — consumer-supplied base URLs, redirect following during extraction — need
a ruling before the capability accepts a URL from anywhere but deployment config.

**SR-K4 (DECISION).** robots.txt and provider terms — a family stance, or
adapter-side? Unaddressed everywhere today (`shared_search.md:176-177`), and the
exposure differs by carrier: image search touches rights-bearing works, which C6
already models as `rights_status`.
*Recommendation:* a stated family stance, enforced per adapter. A per-app answer
means the first app to get our shared SearXNG banned decides for everyone.

Replay widens this. A recording (SR-F3) turns transit into storage — we would be
keeping third-party page text, images, and eventually video, for as long as an
eval run lives. That is a different posture from fetching and discarding, it
varies by carrier and by provider terms, and it is worth ruling on alongside the
retention mechanism rather than after somebody notices.

### L. Packaging, weight, and deployment shape

**SR-L1 (REQUIRED, P3 — G10).** A consumer must take Adapter and Call without a
cross-encoder, torch, or a fetch stack — success check 5.

**SR-L2 (REQUIRED, P8).** Absent optional layers degrade explicitly: a consumer
without rerank gets provider order and *knows* it.

**SR-L3 (REQUIRED, P1).** Types crossing package boundaries ship as a
dependency-free leaf, per the ratified contracts-leaf pattern.

**SR-L4 (REQUIRED, G12 — §5.4).** Types crossing a *layer* boundary are
wire-serialisable, not merely dependency-free. SR-L3 is the packaging rule; this
is its wire twin, and the two are not the same — a dataclass holding a callable,
an open file, or a supplied port satisfies SR-L3 and fails this. The north star
places a NATS hop somewhere in §6's stack and does not say where, so the
survivable requirement is that any boundary could carry one. Two things already
depend on it: SR-F5's store reference and SR-I4's returned records. The cost is
paid in design freedom, not at runtime, so the embedded path carries none of it.

*The family has a shipped shape for this and the design should copy it rather
than invent one.* `media-contracts`' `ObjectHandle` is *"the small descriptor
that crosses NATS in place of the bytes"*, with an explicit
`to_metadata()`/`from_metadata()` pair that stringifies UUIDs *"at this border so
the descriptor survives the NATS/JSON round-trip intact"*, and a named metadata
key (`OBJECT_HANDLE_METADATA_KEY`) so a producer and a consumer agree on where it
rides. Three things worth taking whole: an explicit border projection rather than
"whatever the serialiser does", the descriptor/bytes split (which is also the
§4.12 line between finding and acquiring), and a named key on
`ToolResult.metadata` — because `metadata` is typed `dict[str, Any] | None`
(`base_tool.py:26-41`), so it is a shared namespace with no schema, and two tools
writing bare top-level fields into it is a collision waiting to happen. Search's
structured results need a key of their own for the same reason.

**SR-L5 (REQUIRED, G12 — §5.4).** The leaf is usable from a **one-shot
`asyncio.run()`**: no ambient event loop, no long-lived client, no background
task required to make a single call. This is what lets a synchronous consumer
call an async capability without adopting an event loop it has deliberately
avoided — samsung's discovery plane is synchronous throughout, with async
confined to its HTTP and MCP surfaces. It is deliberately narrower than a sync
API: samsung searches one work at a time and needs none of the fan-out machinery,
so the bridge is a call wrapper rather than a second surface. It also scopes
open question 15 out of this capability's path — the family's sync-subscript
bridge stays unproven, and search does not become the thing that proves it.

**SR-L6 (REQUIRED, G10, G12).** Bounds that hold on the smallest target: a
steady-state footprint that fits beside another plane under a `MemoryMax` cap
(SR-G5's byte caps are the acute case; this is the resting one), concurrency
**defaults** — not just limits — that are safe unturned there (SR-H1), `arm64`
wheels for anything carrying a native extension, and a Python floor spanning the
family's spread — samsung is 3.14, discodon is 3.12, and open question 1 is what
decides how that gap closes. Success check 5 states
the install-weight half of this; the runtime half was unstated until now, and it
is the half a `MemoryMax` cap actually enforces.

*The Python floor is live, and this requirement does not wait on it.* `3tears`
core declares `requires-python = ">=3.14"` today, so a leaf depending on core
would exclude discodon at 3.12 as well as samsung on weight. Open question 1 is
open between two shapes — move discodon to 3.14, or make the minimum a
per-module statement and find the subset that can hold a relaxed floor — with no
recommendation recorded and nobody having checked how large that subset is.
SR-L7 is what makes this survivable under either: if the leaf takes no core
dependency, the floor question stops gating it and starts gating only the hosts
that inject core-backed implementations.

**SR-L7 (REQUIRED, P9, G13 — §5.4).** **No path from the contract leaf or the
Adapter into `threetears.core`.** Core is where most of the primitives this
capability wants actually live, and it is refused on the record by the smallest
consumer and excluded by version from the largest (§5.4). Everything the leaf
needs from core is therefore taken as a *shape*: the transport (SR-N1), the exit
(SR-N2), the limiter (SR-H4), the secret reference (SR-K1), the replay store
(SR-F5). The permitted dependency floor is the contracts-leaf set —
`3tears-observe` (`dependencies = []`), `3tears-media-contracts`
(`dependencies = []`), `pydantic`, and a provider's own transport behind an
extra. Anything heavier is a host concern, wired in.

This is the requirement most likely to be argued away one import at a time, so
the acceptance test must be mechanical rather than cultural — and *(checked
2026-08-04)* the test this pass first cited does not yet do that job:

- `tests/enforcement/test_dependency_alignment.py` verifies imports match
  declarations in both directions — an undeclared `threetears.*` import
  fails, and a declared-but-unimported workspace dep fails. That catches
  *"the drift class where the uv workspace masks undeclared cross-package
  dependencies until a standalone `pip install` of one package ImportErrors
  in a consumer."*
- It pins no package's dependency *list*: a new hard dep that is genuinely
  imported self-satisfies with no reviewed change.
- The pin that actually exists is its sibling
  `test_contracts_packages_stay_dependency_free`, which holds
  `media-contracts` to stdlib-only by walker.

The leaf therefore needs its own floor pin — the same mechanism, allowing
exactly the floor above — and that pin is a deliverable of the build, not
something the suite already provides.

### M. Lifecycle

**SR-M1 (DECISION).** How do the types version, and what is the compatibility
promise across lockstep releases? Every §5.2 consumer, across four repos, will
bind, and discodon carries an open advisory that it exposes an API with no
recorded versioning scheme. *Recommendation:* rule before the first consumer
binds.

*Half of this is already ruled, and the doc should say which half.* Inside the
family the answer is the lockstep rule — every `3tears*` package releases at the
same version and every intra-family dependency carries the bound
`>=<major>.<minor>.0,<major>.<minor+1>.0`, mechanically enforced by
`tests/enforcement/test_intra_family_version_bounds.py` and stated in
`3tears/CLAUDE.md` as a hard rule with two named production incidents behind it.
A new leaf inherits that on arrival; there is nothing to decide. What is
genuinely open is everything the lockstep rule does not reach: the wire contract
below, and any consumer outside the family's release train.

*Escalated by §5.4.* Pod-resident this stops being a Python API promise and
becomes a **wire contract** between independently deployed pods and consumers,
where the two sides are not upgraded together and the family's lockstep-version
rule does not reach across the bus. SR-C1's open carrier facets need a schema
evolution rule to match — "additive and open" is a Python-typing statement, not a
serialisation one. That moves this from "rule before the first consumer binds"
toward gating the first pod-resident deployment.

**SR-M2 (DECISION).** Response caching — where, and is it in scope?
`shared_search.md:173-175` raises it and notes core collections have no TTL
semantics. It interacts with SR-F3: a cache and a replay store solve adjacent
problems, and building one without deciding the other tends to produce a cache
that is *almost* a replay store.
*Recommendation:* decide replay first; caching after, in its light.

**SR-M3 (DECISION — open question 13).** Ratification home.
*Recommendation:* this file is the cross-repo record; discodon, metallm and
samsung each record acceptance of what binds them. Otherwise the next session in
any of those repos re-derives all of it.

### N. Transport and egress

Absent entirely from the first pass, and from `shared_search.md`, which says
"httpx-only" three times. Every layer above Adapter is about *what* to retrieve;
this section is about *how the request leaves*, which the family already treats
as a first-class seam and which several requirements above silently assume an
answer to.

**SR-N1 (REQUIRED, P9, G13).** Adapter reaches an upstream through **one
injected transport seam**, never a client it opens itself.

This is a norm with a gate behind it: `tests/enforcement/test_no_bespoke_reuse.py`
fails any class under a 3tears package holding *"a raw, long-lived
`httpx.AsyncClient` / `httpx.Client` as a field outside the sanctioned traced
HTTP-client wrapper (`threetears.core.http_client`)"*. That wrapper already
supplies, in one place, four things Part II asks for separately:

| Asked for here | Already in `TracedHttpClient` |
|---|---|
| SR-G1 — timeouts configurable, not constants | `timeout` parameter, defaulting to `DEFAULT_HTTP_TIMEOUT_SECONDS` from core config |
| SR-G4 — bounded transport retry at Adapter | `max_attempts` / `initial_backoff` / `max_backoff` over `observe.retry_with_backoff`; 5xx and connect/timeout retry, 4xx does not |
| SR-D3 — provider exhaustion short-circuits | injected `CircuitBreakerLike`, per upstream |
| SR-I1/SR-I2 — per-call telemetry, duration | `observe.traced` span per request, zero-cost when OTel is absent |

**So SR-G4 is not an open decision.** The family has ruled it — bounded retry at
the transport, finite by design (*"forever-retry is wrong for a request"*) — and
shipped it. It moves from DECISION to REQUIRED, satisfied by reuse.

The collision, stated plainly because it is the sharpest one in this document:
the wrapper lives in `threetears.core`, and SR-L7 forbids the leaf from
depending on core. **Both rules are right and neither bends.** The resolution is
P9, and it is the move `http_client` itself already makes: it guards upstreams
with `threetears.models`' circuit breaker *without importing it*, by declaring
`CircuitBreakerLike` as a structural protocol and taking the breaker injected —
*"the injection keeps that layering seam intact"*. Applied one level out: the
leaf declares the transport shape, `TracedHttpClient` satisfies it by shape for
every host that has core, and samsung supplies a bare-httpx implementation.
Success check 11 is that this passes enforcement **without an exemption** —
`_no_bespoke_reuse_exemptions.txt` requires a specific `# rationale:` line, and
"the Pi refused core" is a design smell to fix rather than a rationale to file.

*Recommendation to 3tears (a norm improvement, not a waiver):* the sanctioned
list in `test_no_bespoke_reuse` should recognise a declared transport protocol
as a sanctioned target, so the rule reads "no bespoke client" rather than "no
client outside core". That widens the norm's reach — it would then bind
lightweight leaves that today escape it entirely by not being able to comply.

*Mechanism, checked 2026-08-04 — and it prices the ask.* The sanction is a
path allowlist: `_SANCTIONED_HTTPX_SITES`, a frozenset of file paths the
walker skips (core's and mcp's `http_client.py` today). The walker only flags
a raw `httpx.AsyncClient`/`Client` stored on `self` — an Adapter holding a
*protocol-typed* transport field never trips it at all. What would trip it is
the leaf's shipped bare-httpx default transport. So the widening is a one-line
frozenset edit plus prose: add that module's path to the sanctioned set, and
restate the norm as "no bespoke client outside a sanctioned transport
implementation". Reviewed, no exemption filed.

**SR-N2 (REQUIRED, P2, P5 — G8, G11).** **Which exit a call leaves by is an
input at Adapter and provenance on the result.**

`threetears.core.egress` is the family's shipped answer — `EgressDriver` as a
`runtime_checkable` Protocol with `DirectEgress`, `ProxyEgress`, `SocksEgress`
and `WarpEgress`, written because *"every app on this framework eventually wants
a request to leave by something other than the container's default route"*, and
deliberately a driver rather than a flag so *"adding the fourth is one class and
no change to any caller."* `TracedHttpClient` already takes `egress=` and
exposes `egress_name`; `3tears-scrape` already records `last_egress` on its
circuit and health rows, with a comment pointing at `DirectEgress` for why
"direct" is a named exit rather than an absence.

Three consequences, each of which is a requirement rather than a design note:

- **P2 applies.** Which exit a retrieval left by is knowable only at Adapter and
  unrecoverable above it — the same argument samsung records for
  `acquisition_method`. It belongs on the result, in usable form, and `direct` is
  a value rather than a null. Without it, a replayed run (SR-F3) cannot know
  whether it is comparable to the original, and a blocked provider cannot be
  attributed to the address that got blocked.
- **Exits differ per provider, in the same process.** SR-K3 already observes
  that a self-hosted SearXNG base URL is an internal endpoint. It follows that a
  deployment routing external search through a proxy for non-attribution must
  *not* route its internal SearXNG the same way, so egress is per-upstream
  configuration, not a process-wide setting. Success check 12.
- **Pacing keys on it** — SR-N4.

**SR-N3 (REQUIRED — extends SR-K3).** The SSRF surface is the transport's, and
so is its answer. A consumer-supplied base URL, a redirect followed during
extraction, and a fetch of a result URL are three ways for a caller to choose
where our process connects — and under SR-N2 they may be connecting *through a
configured exit*, which is a stronger capability than plain outbound HTTP.
Whatever ruling SR-K3 takes has to bind at the transport seam rather than at
each call site, or the third call site written will be the one that skipped it.

**SR-N4 (REQUIRED — G8, extends SR-H4, SR-D6).** Rate and ban budgets are keyed
on `(provider instance, egress)`. A ban is issued against an address, not
against a configuration. Two deployments sharing our SearXNG behind different
exits are two subjects; one deployment reaching two providers through one exit
is one. SR-H4's "per provider instance" was the closest statement available
before egress was in scope.

### O. Conformance to the family's enforced norms

3tears enforces several house rules mechanically over `packages/*/src/` trees —
some workspace-wide (`test_no_silent_swallow`, `test_uuidv7_enforcement`), some
over an explicit root list a new package must be *added to*
(`test_dict_state_detection` scans five named roots today), and some
per-package (the import-cost / lazy-init gates, which a new package must bring
its own copy of). A new leaf is inside the workspace-wide scope on the day it
lands and joins the listed and per-package scopes as build deliverables, so
these are requirements on the deliverable rather than review preferences. Listed
because each one *already answers* a question Part II asks, and because
discovering them during implementation is how exemption files grow.

**SR-O1 (REQUIRED — pins SR-B3, SR-J1, SR-J2).** No silent swallow.
`test_no_silent_swallow.py`: an exception handler logs, re-raises, or carries a
`# NOSILENT: <reason>` marker. What that rule's own review kept finding is
precisely this capability's failure mode — *"the handler that returns a
plausible value: a memory search answering 'nothing found' when three of its
four branches had failed."* SR-J2 pins zero results as a success; SR-O1 is what
stops a *failed* search from arriving as one.

**SR-O2 (REQUIRED — conditions SR-A5, SR-M2, SR-H4).** Persistent state goes in
a backend, not a dict. `test_dict_state_detection.py`: state assigned in an
`__init__` that outlives a request belongs in an L1 backend or NATS KV, because
*"a dict is per-process, so two pods disagree and a restart forgets."* Three
things in this document are candidates the moment they are built —
Aggregate's accumulated corpus, a response cache, and a local rate limiter — and
the third is the one SR-H4 now says must exist embedded. The rule has an
ALLOWLIST for state that genuinely cannot live in a backend; a per-process
limiter on a broker-less Pi is a plausible entry, and it is an entry to be
argued at design time, not assumed. (Scope note, checked 2026-08-04: this
scanner runs over an explicit five-root list — core, registry, agent/memory,
agent/tools, langgraph — so adding the search package to that list is itself
part of the deliverable.)

**SR-O3 (REQUIRED — conditions SR-M2).** A cache is a `BaseCollection`.
`test_cache_primitive_usage.py` and `test_no_bespoke_reuse.py` check (c) both
fail *"a KV / counter that stores an `SQLiteBackend` and exposes cache-style
verbs without subclassing `BaseCollection`."* This sharpens SR-M2's
recommendation to defer caching: pod-resident, response caching is a collection
or it is a violation; embedded, `BaseCollection` is in core and SR-L7 forbids
it. Deciding replay first (SR-M2) is therefore doubly right — a cache here is
not a small thing to add later, it is a thing with two different legal shapes.

**SR-O4 (REQUIRED — conditions SR-F8).** Persisted identifiers are UUIDv7.
`test_uuidv7_enforcement.py` statically bans importing `uuid4` anywhere under
`packages/*/src/` and dynamically verifies version 7, because a random id
*"poisons cursor-paged ordering."* A replay recording's id (SR-F5's typed
envelope) is a persisted identifier. The leaf can declare `UUID`-typed fields
without a dependency — `media-contracts` does exactly this — and generation
stays with whatever writes the record.

**SR-O5 (REQUIRED).** Provider conformance is a shared suite, and test doubles
declare their parent. `shared_search.md`'s per-provider conformance tests are
the right instinct and have a house precedent in the `DurableStore` conformance
direction (OQ16). Additionally, `test_fake_protocol_parity.py` requires every
`Fake<Name>` under any `tests/` directory to declare the production protocol it
stands in for — which for this capability means the injected transport (SR-N1),
the egress driver (SR-N2) and the store port (SR-F5) each get a declared fake
rather than an ad-hoc stub, and drift in any of the three breaks loudly.

## 10. Defects found while gathering this

True today, whether or not any convergence happens. Items 6–8 are
`shared_search.md`'s findings, kept here so one list is complete. Items 9–10 are
in `agent-tools`' call envelope rather than in any search implementation — they
affect every pod-served tool, and surfaced only because §5.4 put the pod path in
scope.

1. **Discodon persona search under-bills by 2× on `advanced`** — SR-E4.
2. **The builtin blocks the event loop** — SR-G3.
3. **Research search timeout is unconfigurable in practice** — the constructor
   parameter is never wired to config — SR-G1.
4. **Research searches sat outside the cassette layer**, because replay was
   attached to a class hierarchy the sub-tool deliberately left — SR-F3.
   *Stale as of 2026-08-04:* discodon wired a delivery seam freezing research
   payloads for character evals the day this list was re-verified; the
   sub-tool's individual queries stay unfrozen by recorded design. What
   remains true: nothing can replay the research *pipeline's own* searches —
   the consumer search-internal replay exists for.
5. **The two discodon implementations disagree on whether a failed search costs
   budget**, and neither records a decision — SR-D4.
6. **`time.sleep(1)` in the fetch retry loop** — same blocking class as 2.
7. **Unbounded download** — `resp.text` with no byte cap or content-type gate —
   SR-G5.
8. **Errors detected by string prefix** — `not content.startswith("[TOOL ERROR]")`
   — SR-J3.
9. **The tool-call envelope drops metadata when an exception escapes the tool.**
   `CallResponse(success=False, content="", error=...)` carries no metadata
   (`server.py:2071-2076`), so any spend a tool incurred before raising is
   discarded at the wire — SR-E3. True today for every pod-served tool, not only
   search; search is what makes it cost money. *Scope corrected 2026-08-03:* the
   **handled**-failure branch does carry it — a `ToolResult(success=False,
   metadata=…)` reaches `CallResponse` intact (`server.py:2050-2056`) — so this
   is an exception-path defect, not a failed-call defect. Search can route around
   it by never letting an exception cross the wire (SR-J3); tools that raise
   cannot, which is why the ask stands. *Re-verified 2026-08-10 (0.23.11):*
   still true after the durable-results work more than doubled `server.py` —
   every error-path `CallResponse` is still built without metadata (the
   constructions now sit near `server.py:2219`, `:2302`, `:2330`, `:2354`;
   line references above are the 0.23.0 tree's).
10. **The envelope has no per-call deadline** (`server.py:376-417`, and
    `extra="forbid"` closes the workaround), so no pod-served tool can derive its
    timeout from the caller's remaining budget — SR-G2. A gap rather than a
    defect: it was never asked to carry one. Items 9 and 10 are both asks on
    `agent-tools` rather than search-side fixes. *Re-verified 2026-08-10
    (0.23.11):* still no per-call deadline, and `extra="forbid"` stands. Two
    adjacent things landed meanwhile, neither closing the gap:
    `CallRequest.result_subject` (durable long-call delivery — and live proof
    the envelope takes additive fields under the server-accepts-first rollout
    D18 prescribes), and `ToolManifestEntry.timeout_seconds` with server-side
    hard-timeout of runaways (0.23.2) — the *tool's* declared ceiling again,
    not the caller's remaining budget. When the deadline field lands it
    composes with that ceiling: the effective bound is the min of the two.
11. **The LangGraph context-save node is silently inert** — it matches
    `ToolMessage.name` against bare `web_search`/`web_fetch` by exact equality
    while the adapter binds the namespaced `threetears.*` names
    (`graph_nodes.py:126,156-168`; `langchain_adapter.py:131`), so the default
    set never matches; no production code wires the node, and its tests pass
    bare names, so the suite cannot see it — C8. When wired, it persists
    flattened, 4000-char-truncated `content` with no provenance: a rename is a
    silent retention change in either direction, and the saved text carries
    nothing SR-A3 would let a reader re-check.
12. **`ToolResult.metadata` is an unkeyed shared namespace** —
    `dict[str, Any] | None` with no schema (`base_tool.py:26-41`). The family
    already handles this for one payload by convention
    (`OBJECT_HANDLE_METADATA_KEY`), but nothing enforces that two tools writing
    top-level keys do not collide — a latent problem that SR-A1 makes load-bearing
    the moment structured results start riding there. Not a defect in any
    implementation; a convention that needs stating before a second payload
    arrives.

## 11. Reading `shared_search.md` against this

The sketch is right about layering, provider extras, capability metadata,
per-provider conformance tests, `ToolResult.metadata` as the non-breaking
migration path, and packaging option A. The `metadata` call is better than it
looked when the sketch made it: verified end-to-end, that field survives the NATS
hop into `CallResponse` (`server.py:2053`), so the same migration works embedded
and pod-resident. It is the one choice the north star promotes from convenience
to foundation. Five of its choices are contradicted, all
in the same direction — they assume the deliverable is a web page:

1. **`category: str | None = None  # searxng categories; tavily ignores`**
   (`:59`). A silent per-provider drop, written into the contract sketch itself —
   SR-B2/SR-B3/P8, and the seed of exactly the coupling §7 exists to prevent.
2. **A single `score: float | None`** (`:68`). Keeping the provider's score rather
   than discarding it improves on today; collapsing every ranking judgment into
   one number is P7/SR-A4, and forecloses C6, whose conflicting
   `confidence`/`quality_score` split is already designed.
3. **Content lives in `FetchResult`, not `SearchResult`** (`:64-88`). Tavily
   returns page text with the search response, so this shape cannot express it
   and a Tavily consumer re-fetches what it already bought — SR-A2, and a
   capability regression under convergence principle 4.
4. **`SearchResult` is page-shaped** — url/title/snippet/published_at, no carrier
   facets (`:64-71`). C6 cannot bind to it — G3, SR-C1/SR-C3.
5. **`raw: dict[str, Any]  # provider passthrough, never load-bearing`** (`:71`).
   Untyped and disclaimed is not preservation: it is where `acquisition_method`
   would land and then be unusable — P2.

Two structural notes:

6. **The L2 "pipeline" fuses aggregate, select and extract** — "search → dedupe →
   optional rerank → bounded-concurrency fetch of top-k" (`:130-131`). A consumer
   wanting dedup without rerank, or rerank without fetch, takes all three — P4.
   Composable stages, not one helper.
7. **Record/replay appears nowhere** — SR-F3, and the one that most affects G4.
8. **"httpx-only" is the one phrase to drop** (`:103`, `:150`; the raw httpx
   path is also `:114`). It reads as a virtue — a slim provider with no heavy
   transitive — and under `test_no_bespoke_reuse` it is a build failure, because
   the sanctioned transport is `threetears.core.http_client` and a stored
   `httpx.AsyncClient` is exactly what the rule names. The sketch's instinct is
   right and its wording is a norm violation; SR-N1 is what makes both true at
   once. Egress (SR-N2) is likewise absent from the sketch, and the same seam
   carries it.

None of this argues against the sketch's direction. It argues that the contract
should be cut after §13 has answers, and that the answers change five fields —
plus a transport seam the sketch does not have.

## 12. Open assumptions

- **SR-A4** — SearXNG's score semantics are stated from general knowledge, not
  measured. Confirm against a live instance before ruling.
- The layer cut in §6 is proposed here, not derived from any owner's recorded
  position. Every requirement is attributed to it, so re-cutting it ripples.
  *This is also the document's one violation of convergence principle 2*
  ("extract, don't invent") — six invented layer names over almost entirely
  existing machinery. The §6 reuse table is what keeps that honest; the fewer of
  those names that survive into the contract as *types*, the better.
- **SR-F5's storage burden is estimated, not measured.** The claim that all three
  candidate consumers can satisfy a store port with a wiring line rests on each
  already having a durable store, not on anyone having tried it. Narrowed by the
  second pass, not closed: `ObjectStore` being dependency-free and already
  implemented twice makes the port cheaper than assumed, but nobody has wired it.
- **The NATS convergence has no date.** §5.4's north star is ruled but not
  scheduled, and every requirement written for the pod path is weight the
  embedded path carries meanwhile. SR-L4 is designed to make that weight
  design-time rather than runtime; if convergence slips far enough, revisit
  whether the rest of the pod-conditioned work should have waited.
- **SR-L6's Python floor** is asserted from the family's current spread, not from
  a check of what the leaf's own dependencies actually support on 3.12 — the same
  unchecked quantity open question 1 now turns on, since a per-module floor is
  only as good as the subset that can actually hold it. Under SR-L7 the floor
  stops gating the leaf either way, and gates only the hosts that inject
  core-backed implementations.
- **SR-N1's norm amendment is proposed, not agreed.** Widening
  `test_no_bespoke_reuse`'s sanctioned set to include a declared transport
  protocol is a change to a 3tears enforcement rule, and 3tears has not been
  asked. If it is refused, the leaf needs a filed exemption with a rationale —
  which is the outcome success check 11 exists to prevent.
- **The in-process limiter SR-H4 now requires has no home.** Named as owed, not
  designed; SR-O2 says where the argument about its state will happen.

Closed: **A1** (carriers, ruled in), the replay question (SR-F3, ruled in), the
deployment axis and its two modes (G12/§5.4, ruled in), the sync/async fork —
resolved by SR-L5 rather than ruled, since a one-shot `asyncio.run()` contract
serves the synchronous consumer without settling open question 15 for the family
— **A2** (person-typed queries: settled by `face_api` existing rather than
assumed, §5.5), and **`3tears-media-contracts` as a direct pin** (evaluated
2026-08-03: `dependencies = []`, already a hard dep of `agent-tools`, already
carries most of SR-C3's facets — §6).

## 13. Decisions needing an owner

**2026-08-04:** `search-spec.md` §1 takes each build-gating decision below as a
vetoable ruling, adopting this table's recommendation unless noted there. This
table remains the evidence record; a veto lands there and propagates here.

| ID | Decision | Recommendation |
|----|----------|----------------|
| SR-A4 | How many score dimensions, whose | Named provenanced scores; never one `score` |
| SR-A5 | Candidate set vs corpus | Call returns a set; corpus is Aggregate's named type |
| SR-B5 | Model-mediated search in or out (OQ21) | Out of Adapter/Call; in at Aggregate |
| SR-D4 | Does a failed search consume budget | Follow the bill; move the retry bound in the same change |
| SR-D5 | Local vs provider refusal authority | Both, distinct roles — two recorded positions conflict |
| SR-E6 | Self-hosted cost: zero or amortised | Zero, plus a separate rate/quota dimension |
| SR-F5 | Who stores a replay recording | A store port the consumer supplies; the bundle is one implementation, not the primitive |
| SR-H4 | Rate limiting: pace or react | Pace per `(provider instance, egress)`; `TokenBucket` where a bus exists, an in-process limiter where none does |
| SR-I4 | Emit telemetry or return records | Return records |
| SR-J3 | Errors as values or exceptions | Typed exceptions carrying spend; prose at Bind; **Bind converts before the wire** |
| SR-K2 | Are queries sensitive | User content; capability exposes, consumer redacts |
| SR-K4 | robots.txt / provider terms | A family stance, enforced per adapter |
| SR-M1 | Versioning and compatibility promise | Lockstep already covers the in-family Python API; rule the **wire** contract before the first pod-resident deployment |
| SR-M2 | Response caching | Decide replay first, cache in its light — and note it has two legal shapes (SR-O3) |
| SR-M3 | Ratification home (OQ13) | This file, per-repo acceptance |
| §5.4 | Where the wire boundary falls in the layer stack | Not answered here; SR-L4 makes any answer survivable |
| §5.4 | One NATS bus for discodon and metallm, or two | Gates how much of SR-H4's pacing the client side can carry |
| §10.9 | Exception-path metadata on the tool-call envelope | An ask on `agent-tools`; search routes around it via SR-J3, other pod tools cannot |
| §10.10 | A per-call deadline on the tool-call envelope | An ask on `agent-tools`; blocks SR-G2 pod-resident |
| **SR-N1** | Does `test_no_bespoke_reuse` recognise a declared transport protocol as sanctioned | **Yes — a norm widening, asked of 3tears.** Otherwise the leaf needs a filed exemption, which success check 11 exists to prevent |
| **SR-N2** | Is egress per-upstream configuration on this capability | Yes; and the exit is provenance on the result, `direct` included |
| **SR-K3/N3** | Where the SSRF ruling binds | At the transport seam, not per call site |
| **§6** | Select owning ranking was drift from §4.14; corrected to "composes a ranker through a slot" | Ratify the correction, or amend §4.14 — one of the two, not neither |
| **§5.5** | `skill_eligible`, and whether search stays in the `web` group alias | Both are ACL-visible surface; decide before a per-carrier tool split makes `web` mean something new |
| **§10.12** | A named key for search results on `ToolResult.metadata` | Yes — follow `OBJECT_HANDLE_METADATA_KEY`; an unkeyed shared dict is a collision waiting for a second payload |

Closed by the second pass, recorded so they are not reopened: **SR-G4** (retries)
— the family ruled it and shipped it in `core.http_client`, so it is REQUIRED by
reuse rather than a decision; **A2** (person-typed queries) — settled by
`face_api`; **`3tears-media-contracts` as the home for image facets** — evaluated
and taken.

**Ruled 2026-08-02:** carriers are open, images and arbitrary data types in scope
(G3, SR-C1); searches must be replayable, attached at Adapter and Call (SR-F3);
deployment shape is a requirements axis with two permanent modes, embedded and
pod-resident, neither privileged (G12, §5.4); discodon converges off ZMQ onto
NATS with search pod-resident there, while 3tears and metallm run Yugabyte at
scale — so the ZMQ topology is transitional and requirements are written against
the two modes that persist (§5.4).
