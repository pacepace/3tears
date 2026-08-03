# Search: What the Family Needs

**Status:** Draft for ratification — 2026-08-02
**Companions:** `family-convergence.md` §4.14 records the *direction*;
`shared_search.md` sketches a *mechanism*. This states the *need*.
**Relates to:** §4.13 (scraping), §4.2 (evals), open questions 1, 13, 15, 16 and 21

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
| a new leaf | The contract types: dependency-free and wire-serialisable — SR-L3, SR-L4. |

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
a fix happens once. Today: seven call sites, four implementations, two of them
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

**G9. Cost, evals and telemetry are built into the seam.** A run's cost cap
includes what it spent searching. An eval can attribute a score change to a
search-config change. A slow turn is explainable.

**G10. Safe to depend on.** Seven consumers across four repos will bind to this.
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
and which requirements each conditions.

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
a wrong answer.

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

### 5.2 The seven call sites

| # | Consumer | Caller | Carrier | Criteria | Fidelity wanted |
|---|----------|--------|---------|----------|-----------------|
| C1 | discodon persona `web_search` | LLM | web | none | snippet |
| C2 | discodon research sub-tool | LLM | web | shallow | extracted content + corpus |
| C3 | **3tears `agent-tools` `WebSearchTool` builtin** (shared; metallm and C7 consume it) | LLM | web | none | snippet |
| C4 | metallm admin price lookup | program | web | none | extracted content |
| C5 | samsung discovery phase 1 | program (model-mediated) | web | shallow | structured record |
| C6 | samsung discovery phase 2 | program | **image** | **deep** | structured record |
| C7 | `3tears-scrape` `page_finder` | LLM agent (in-family) | web | shallow | locator, then content |

Evidence: C1 `discodon/tools/web_search_tool.py`; C2
`discodon/tools/research/web_search.py`; C3
`3tears/packages/agent/tools/.../builtin/web_search.py`; C4
`metallm/api/src/api/v1/admin/models.py:948`; C5
`samsung/curation/src/curation/discovery/phase_one.py`; C7
`3tears/packages/scrape/src/threetears/scrape/page_finder.py:32,237-241`.

Read the fidelity column: only two of seven want what the shared builtin returns.
Four want extracted content or a structured record, and today they each get there
alone.

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

**Assumption A2** *(vetoable)*: person-typed queries are in scope. No current
call site is one; inferred from the family having web UIs. Cheap now (mostly
"queries are untrusted user content"), expensive to retrofit.

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

Two consequences worth stating rather than discovering:

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

Requirements this axis conditions, each flagged at its own entry: SR-E3 (spend on
the failure path), SR-F5 (who stores a recording), SR-G2 (deadline propagation),
SR-H1 (tuning without a restart), SR-H4 (rate pacing), SR-I4 (telemetry
delivery), SR-K1 (credential resolution), SR-M1 (versioning).

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
| **Select** | candidates + criteria → an ordered, filtered subset | filtering, reranking, scoring, cull | L2 pipeline (rerank slot) |
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

| Layer or concern | Existing home | So we do not build |
|---|---|---|
| Adapter capability declaration | `3tears-models` `capabilities.py` | a second capability-metadata scheme (SR-B4) |
| Rate pacing | `threetears.core.coordination.token_bucket.TokenBucket` | a limiter (SR-H4) |
| Bind, pod-resident | `agent-tools` `TearsTool` / `ToolServer` / registry proxy | a dispatch or discovery mechanism |
| Bind, tool-call audit | `agent-audit`'s shared envelope | a second tool-call history |
| Contract types | the ratified contracts-leaf pattern | a shared-types package with dependencies (SR-L3) |
| Replay store, embedded | samsung's `SqliteDurableStore`; the `DurableStore` direction (OQ16) | a store (SR-F5) |
| Replay store, pod-resident | `3tears-object-store` — streaming, S3-compatible, built for large artifacts | a blob path (SR-F5, SR-G5) |
| Telemetry | `3tears-observe`, wired by the host | a sink (SR-I4) |
| Tuning without a restart, pod-resident | `3tears-epoch` | a config-broadcast mechanism (SR-H1) |
| Heavy or hostile fetch | `3tears-scrape` | a scraper (§2) |
| Retrieval over our own content | `agent-memory` | RAG (§2) |
| Principle enforcement | `3tears-enforcement` scanners | P1-P8 living only in a reviewer's head |

Two rows need analysis this document does not have.
**`3tears-media-contracts`** is the obvious home for SR-C3's image facets, and
samsung named it among the dependencies it refused — as a *transitive* of
`3tears-models`, not on its own merits, so a direct dependency-free pin may be
fine. Nobody has checked, and the family's only image consumer is the one that
would carry it. **`3tears-epoch`** serves both pod-resident deployments after
convergence and neither embedded one, so SR-H1 needs a second answer rather than
a single mechanism.

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

Ruled 2026-08-02: carriers are open, including images, video and datasets (G3);
searches must be replayable (SR-F3); deployment shape is a requirements axis
with two permanent modes (G12, §5.4); discodon converges onto NATS with search
pod-resident, and 3tears and metallm run Yugabyte at scale (§5.4).

---

# Part II — What that cashes out to

Requirements traced to evidence, read 2026-08-02, cited as `repo/path:line`.
Each is **REQUIRED** (a consumer regresses or breaks without it), **DECISION**
(consumers disagree or nobody has ruled — recommendation given, owner picks), or
**ASSUMPTION** (inferred, vetoable). Where implementations disagree, the
disagreement *is* the finding: the requirement was never stated, so each site
guessed.

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

### E. Cost tracking

**SR-E1 (REQUIRED, P5 — G9).** Spend is attributable per call, in money,
observable from any layer. Discodon counts and explicitly declines to price:
"Paid non-LLM calls this run made (web search). Counted, never priced"
(`research_tool.py:2503`), with its eval surface warning that "max_cost_usd does
not bound external search quota" (`web/mcp/eval/runs.py:190`). Closing this is
success check 3.

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
error=...)` with **no metadata** (`server.py:2070-2074`), so whatever the adapter
spent before the failure is discarded at the wire. Embedded, the consumer can
still see it. This is an ask on `agent-tools`, not a search-side fix — §10, item
9 — and it is cheap now and expensive once seven consumers bind.

**SR-E4 (REQUIRED — live defect).** Weighted units must be accounted. Discodon's
persona tool bills every search as one unit (`_check_budget` at its `cost=1`
default, `web_search_tool.py:224`; `tools/base.py:1412`) while
`search_depth="advanced"` spends two Tavily credits — and its docstring says the
budget exists "to manage shared API credits" (`web_search_tool.py:7,57`). The
weighted primitive exists and `youtube_tool.py:216` uses it.

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
(`eval/identity.py:221`, via `canonical_digest`). The parameter object must
therefore be canonically serialisable.

**SR-F2 (REQUIRED).** Eval runs against a quota separable from production's, with
sharing explicit rather than a fallback. Discodon designed exactly this (EVL-TQ7K,
`discodon/config/sections/tavily.py`): an optional eval-scoped key so "an eval
search burst can never exhaust the quota or trip the breaker protecting live
personas' research," with unset meaning a *documented shared* quota.

**SR-F3 (REQUIRED — ruled 2026-08-02; G4).** A search must be replayable.
Record/replay attaches as a cross-cutting concern at Adapter and Call (P5), not
at whatever layer happens to be a `Tool`. That placement is the lesson of the
current gap: discodon's cassette layer records and replays at `Tool.act()`
(`eval/cassette_proxy.py:16-18,185-207`), C1 is a `Tool` and is replayable, and
C2's sub-tool deliberately is not — "no Tool ABC overhead"
(`research/web_search.py:5`). Replay was attached to a class hierarchy, so a
component that left the hierarchy silently left reproducibility, and every
research eval now re-issues live searches against a changing web.

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
(`eval/cassette_proxy.py:120`). Falling through to the network would let an eval
go live without saying so, and its trend line would then be measuring the web.

**SR-F8 (REQUIRED, P2).** The replay key is derived by search, because only
search knows what varies — provider, query, resolved parameters, profile digest.
Discodon already computes the analogous digest for eval variant identity
(`eval/identity.py:221`). A consumer-derived key would go stale the first time a
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

**SR-G3 (REQUIRED).** No blocking IO on an async path. The builtin's `execute()`
is `async` but calls a synchronous `httpx.Client` (`builtin/web_search.py:50-56`,
called at `:122`); `web_fetch.py` has the same defect plus a `time.sleep` in its
retry loop (`shared_search.md:38-40`).

**SR-G4 (DECISION).** Retries: capability or consumer?
None retry today; C2 tells the model to retry in prose
(`research/web_search.py:280`), spending an LLM round to redo an HTTP call.
*Recommendation:* bounded transport retry at Adapter. Interacts with SR-D4.

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

**SR-H2 (REQUIRED).** Two bound scopes, both real: within one batch, and across
simultaneous runs (`research_tool.py:328,376`).

**SR-H3 (REQUIRED).** One call's failure must not poison its siblings in a
concurrent batch — handled and reasoned in C2 (`research_tool.py:106-114`).

**SR-H4 (DECISION — G8).** Rate limiting: pace, or react to 429s?
All react; none pace. An unbounded fan-out at a shared self-hosted SearXNG (C4)
is the case most likely to get our own instance blocked upstream. The primitive
exists — `threetears.core.coordination.token_bucket.TokenBucket`.
*Recommendation:* client-side pacing per provider *instance*, at Adapter. The
shared instance is what is at risk, and no single consumer sees the aggregate
load.

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
transitive weight it refused (`samsung/curation/pyproject.toml:60-70`) — direct
evidence that owning a sink would have cost this capability its most constrained
consumer.

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

### K. Security, privacy, and conduct

**SR-K1 (REQUIRED).** Credentials resolve through the consumer's secret handling;
the capability must not read environment variables itself
(`web_search_tool.py:186-192`).

*Mode-conditioned (§5.4), and the two modes disagree.* Embedded, the rule stands
as written: the library reads nothing and the consumer supplies. Pod-resident it
inverts — holding the provider credential so that consumers never see it is a
large part of what a tool pod is *for*, and a pod that demanded the credential
from each caller would be pushing a secret across the bus on every request. The
requirement is therefore about the **library never reading ambient
configuration**, not about where the secret ultimately lives; the pod is a
consumer that happens to supply it from its own deployment config.

**SR-K2 (DECISION — A2).** Are queries sensitive?
A query can carry user-supplied conversational content and is recorded verbatim
today (`logging/models.py:243`). metallm ships a PII sanitisation wrapper it is
contributing.
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
family's spread — samsung is 3.14, and open question 1 proposes relaxing core to
≥3.12 precisely because it "serves discodon and the Pi", which is this
requirement's two modes under another name. Success check 5 states
the install-weight half of this; the runtime half was unstated until now, and it
is the half a `MemoryMax` cap actually enforces.

### M. Lifecycle

**SR-M1 (DECISION).** How do the types version, and what is the compatibility
promise across lockstep releases? Seven consumers in four repos will bind, and
discodon carries an open advisory that it exposes an API with no recorded
versioning scheme. *Recommendation:* rule before the first consumer binds.

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
4. **Research searches are unreplayable by the cassette layer**, because replay
   was attached to a class hierarchy the sub-tool deliberately left — SR-F3.
5. **The two discodon implementations disagree on whether a failed search costs
   budget**, and neither records a decision — SR-D4.
6. **`time.sleep(1)` in the fetch retry loop** — same blocking class as 2.
7. **Unbounded download** — `resp.text` with no byte cap or content-type gate —
   SR-G5.
8. **Errors detected by string prefix** — `not content.startswith("[TOOL ERROR]")`
   — SR-J3.
9. **The tool-call envelope drops everything but the error on a failed call.**
   `CallResponse(success=False, content="", error=...)` carries no metadata
   (`server.py:2070-2074`), so any spend a tool incurred before failing is
   discarded at the wire — SR-E3. True today for every pod-served tool, not only
   search; search is what makes it cost money.
10. **The envelope has no per-call deadline** (`server.py:376-417`, and
    `extra="forbid"` closes the workaround), so no pod-served tool can derive its
    timeout from the caller's remaining budget — SR-G2. A gap rather than a
    defect: it was never asked to carry one. Items 9 and 10 are both asks on
    `agent-tools` rather than search-side fixes.

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

None of this argues against the sketch's direction. It argues that the contract
should be cut after §13 has answers, and that the answers change five fields.

## 12. Open assumptions

- **A2** — person-typed queries are in scope. Gates SR-K2.
- **SR-A4** — SearXNG's score semantics are stated from general knowledge, not
  measured. Confirm against a live instance before ruling.
- The layer cut in §6 is proposed here, not derived from any owner's recorded
  position. Every requirement is attributed to it, so re-cutting it ripples.
- **SR-F5's storage burden is estimated, not measured.** The claim that all three
  candidate consumers can satisfy a store port with a wiring line rests on each
  already having a durable store, not on anyone having tried it.
- **The NATS convergence has no date.** §5.4's north star is ruled but not
  scheduled, and every requirement written for the pod path is weight the
  embedded path carries meanwhile. SR-L4 is designed to make that weight
  design-time rather than runtime; if convergence slips far enough, revisit
  whether the rest of the pod-conditioned work should have waited.
- **`3tears-media-contracts` has not been evaluated for a direct pin** (§6).
  samsung refused it transitively, which is not the same as refusing it.
- **SR-L6's Python floor** is asserted from the family's current spread, not from
  a check of what the leaf's own dependencies actually support on 3.12.

Closed: **A1** (carriers, ruled in), the replay question (SR-F3, ruled in), the
deployment axis and its two modes (G12/§5.4, ruled in), and the sync/async fork —
resolved by SR-L5 rather than ruled, since a one-shot `asyncio.run()` contract
serves the synchronous consumer without settling open question 15 for the family.

## 13. Decisions needing an owner

| ID | Decision | Recommendation |
|----|----------|----------------|
| SR-A4 | How many score dimensions, whose | Named provenanced scores; never one `score` |
| SR-A5 | Candidate set vs corpus | Call returns a set; corpus is Aggregate's named type |
| SR-B5 | Model-mediated search in or out (OQ21) | Out of Adapter/Call; in at Aggregate |
| SR-D4 | Does a failed search consume budget | Follow the bill; move the retry bound in the same change |
| SR-D5 | Local vs provider refusal authority | Both, distinct roles — two recorded positions conflict |
| SR-E6 | Self-hosted cost: zero or amortised | Zero, plus a separate rate/quota dimension |
| SR-F5 | Who stores a replay recording | A store port the consumer supplies; the bundle is one implementation, not the primitive |
| SR-G4 | Retries: capability or consumer | Bounded, at Adapter |
| SR-H4 | Rate limiting: pace or react | Pace per provider instance, on core's `TokenBucket` |
| SR-I4 | Emit telemetry or return records | Return records |
| SR-J3 | Errors as values or exceptions | Typed exceptions carrying spend; prose at Bind |
| SR-K2 | Are queries sensitive | User content; capability exposes, consumer redacts |
| SR-K4 | robots.txt / provider terms | A family stance, enforced per adapter |
| SR-M1 | Versioning and compatibility promise | Rule before the first consumer binds |
| SR-M2 | Response caching | Decide replay first, cache in its light |
| SR-M3 | Ratification home (OQ13) | This file, per-repo acceptance |
| A2 | Are person-typed queries in scope | Assumed yes; cheap now, expensive to retrofit |
| §5.4 | Where the wire boundary falls in the layer stack | Not answered here; SR-L4 makes any answer survivable |
| §5.4 | One NATS bus for discodon and metallm, or two | Gates how much of SR-H4's pacing the client side can carry |
| §10.9 | Failure-path metadata on the tool-call envelope | An ask on `agent-tools`; blocks SR-E3 pod-resident |
| §10.10 | A per-call deadline on the tool-call envelope | An ask on `agent-tools`; blocks SR-G2 pod-resident |
| §6 | `3tears-media-contracts` as the home for image facets | Evaluate a direct dependency-free pin before inventing facets (SR-C3) |

**Ruled 2026-08-02:** carriers are open, images and arbitrary data types in scope
(G3, SR-C1); searches must be replayable, attached at Adapter and Call (SR-F3);
deployment shape is a requirements axis with two permanent modes, embedded and
pod-resident, neither privileged (G12, §5.4); discodon converges off ZMQ onto
NATS with search pod-resident there, while 3tears and metallm run Yugabyte at
scale — so the ZMQ topology is transitional and requirements are written against
the two modes that persist (§5.4).
