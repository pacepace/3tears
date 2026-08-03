# Search: The Architecture

**Status:** Draft — 2026-08-03
**Scope:** structural shape and adoption. No fields, no signatures, no sequence.

**Companions** — read in the order *direction → need → shape*:

| Document | Carries |
|---|---|
| `family-convergence.md` §4.14 | the **direction** |
| `search-requirements.md` | the **need** — the evidence, and the cross-repo record (SR-M3) |
| **this document** | the **shape** |
| `shared_search.md` | an earlier *mechanism* sketch, overtaken by this one |

This is the newest of the four. Where it and an older one disagree, this is
current, and the disagreement should be propagated rather than left standing.

## Summary

Eight call sites across four repos reach the web today, through four
implementations, two of them written *because* the shared one did not fit.

```
today — four implementations, and the shared one destroys structure

  3tears    agent-tools WebSearchTool ──▶ flattened numbered string
                ├── C3  the builtin's own callers
                ├── C7  scrape page_finder     (inherits the flattening)
                └── C8  context-save node      (persists it, truncated at 4000 chars)

  discodon  web_search_tool.py       ──▶ C1  persona search
            research/web_search.py   ──▶ C2  research sub-tool — prose + a typed corpus

  metallm   raw SearXNG helper       ──┬─▶ C4  admin price lookup
            + extraction wrapper       └── side-steps, written because the shared one
                                           returns only formatted text (SR-A1)

  samsung   model-mediated               C5  discovery phase 1
            not built                    C6  discovery phase 2  ◀── the load-bearing test
```

If one line survives: **structure is the primitive; rendering is a binding.**

```
after — one leaf, six layers, and each consumer stops where its need ends

   Adapter ──▶ Call ──▶ Aggregate ──▶ Extract ──▶ Select ──▶ Bind
      │                                             │          │
      │                                             │          └─ prose for a model, or a
      │                                             │             typed domain object
      │                                             └─ a slot a ranker plugs into
      └─ transport arrives injected                    (composed, never owned)
         (piece 6)

   full fidelity wherever you stop · every boundary wire-serialisable, because
   nobody has decided yet which one carries the NATS hop
```

The rest is what that costs once you hold to it across four repos, two deployment
modes, and a carrier list that is open by ruling. This says what changes in the
shape of the code, what each consumer does about it, and what drives each piece.
Drivers are cited as requirements-doc IDs — `G*` goals, `P*` principles, `SR-*`
requirements, and the numbered success checks in §3.

---

## What changes

### 1. Formatting moves to the edge

Today the builtin flattens ten results into a numbered string, and everything
downstream either accepts that or reimplements search.

```
today   provider ──▶ _format_results ──▶ "1. Title — url\n2. …" ──▶ every consumer
                                          └─ structure destroyed once, for everyone

after   provider ──▶ structured candidates
                          ├──▶ Bind ──▶ prose ──▶ ToolResult.content    (callers unchanged)
                          └─────────────────────▶ ToolResult.metadata   (structure, own key)
```

The structured result becomes the thing that exists; the LLM-shaped string
becomes one rendering, produced at Bind. `WebSearchTool` keeps all three parts of
its identity — its `threetears.web_search` name, its `TearsTool` ABC, its
`ToolResult` shape. This is a gutting, not a replacement.

Structure rides `ToolResult.metadata` under a key of its own, and that channel is
verified to survive the NATS hop, so one migration covers both deployment modes.

*Driven by: G1, G6 · SR-A1 · check 8 · §4.14's ruling that formatting is
presentation.*

### 2. Six layers, each a legitimate stopping point

| Layer | Turns |
|---|---|
| **Adapter** | a request → one provider's API |
| **Call** | a query → one candidate set |
| **Aggregate** | many calls → one set |
| **Extract** | a carrier → the information in it |
| **Select** | candidates + criteria → an ordered subset |
| **Bind** | candidates → what the caller consumes |

A consumer stops wherever it likes and gets full fidelity there:

| Consumer | Takes | Gets |
|---|---|---|
| discodon persona (C1) | Adapter · Call · Bind | prose snippets |
| metallm price lookup (C4) | Adapter → Extract, then binds itself | extracted content |
| discodon research (C2) | all six, both bindings | prose *and* a typed corpus |
| samsung phase 2 (C6) | all six, with the deepest Select | structured records |
| scrape `page_finder` (C7) | reaches them through an agent loop | locators, then content |

A fused pipeline helper that bundles dedup, rerank and fetch means wanting one
costs you three — which is how the current code got here.

**Select owns the criteria negotiation and the cull, and exposes a *slot* a
ranker plugs into.** It never owns a ranking implementation, because:

- `agent-memory` already ships MMR
- `3tears-models` already carries rerank metadata
- a cross-encoder drags torch onto a Pi that cannot carry it

*Driven by: G7, P3, P4 · SR-L1 · §4.14's rerank ruling.*

### 3. The result core is carrier-neutral, and its facets are found rather than invented

Images, PDFs, video and datasets are in scope by ruling, so a closed
`web | image | pdf` union is prohibited:

- the **core** carries identity, provenance, scores and available fidelity
- **facets are additive**, and a consumer that does not recognise one ignores it

The facets come from `3tears-media-contracts`, which is already dependency-free,
already a hard dep of `agent-tools`, and already carrying the carrier taxonomy
and `extraction_status`.

Three fields are genuinely missing — **rights status**, **pixel dimensions**,
**direct-file versus containing-page** — and they belong there rather than in a
search leaf inventing a fourth media vocabulary.

*Driven by: G3 (ruled 2026-08-02), G13 · SR-C1, SR-C2, SR-C3 · checks 7 and 13.*

### 4. Cross-cutting concerns attach where the fact arises, not to a class

Spend, budget, telemetry and record/replay attach at Adapter and Call.

**The evidence is sharp.** discodon attached replay to `Tool.act()`. Its research
sub-tool then left the `Tool` hierarchy for good reasons, and reproducibility
silently left with it — every research eval now re-issues live searches and
measures the web's drift.

**One consequence to carry rather than discover:** search stops being the
stateless tool `agent-tools` classifies it as, since per-call spend, per-day
budget scope and a replay key are all state.

#### Spend is every resource a call consumed, not only money

Dollars, wall-clock, provider quota and call count, bytes moved. Modelling it as
currency is the trap, because the two constraints that actually bite here are
unpriced:

- **Self-hosted SearXNG costs nothing**, and its failure mode is a ban. A
  mechanism keyed on cost never fires for it.
- **A call that burned twelve seconds and returned nothing** because a budget was
  already spent is pure latency with zero coverage gain. The run that paid for it
  needs that as its own number, not as a zero.

All of it has to survive the failure path too: a search that timed out still
spent time and quota, and dropping that under-reports by exactly what the
failures cost. So **Bind catches every typed exception and renders it as a failed
result carrying the spend** — nothing raises across the wire.

*Driven by: P5 · SR-D1, SR-D6, SR-E1, SR-E3, SR-F3, SR-I1, SR-I2, SR-I3 ·
check 3.*

### 5. The leaf depends on nothing the smallest consumer refuses

`threetears.core` holds most of the primitives this capability wants — traced
transport, egress selection, the token bucket, secret resolution — and also
hard-requires sqlalchemy, asyncpg, cryptography and pyjwt. So the permitted floor
is `3tears-observe`, `3tears-media-contracts`, pydantic, and a provider's
transport behind an extra, and there is **no path from the contract leaf or the
Adapter into `threetears.core`**.

```
what the leaf must not do

    leaf ──▶ threetears.core ──▶ sqlalchemy · asyncpg · cryptography · pyjwt

what it does instead

    leaf ──▶ declares a transport protocol ──┬──▶ host injects core.http_client  (pod)
                                             └──▶ host injects bare httpx        (Pi)
```

Everything core provides arrives as a shape the leaf declares and the host
injects — the move 3tears already makes with itself twice, in
`core.http_client`'s injected circuit breaker and in `media-contracts`'
dependency-free store protocols.

**One leaf then serves both modes:**

- **embedded** — on a `MemoryMax`-capped Pi with no broker, callable from a
  one-shot `asyncio.run()`
- **pod-resident** — as a `TearsTool` over NATS

Where the wire hop falls in the stack is undecided, so every layer boundary is
wire-serialisable and any of them could be it — paid in design freedom, not at
runtime.

#### Why core sits on the far side of the seam

The rule underneath is that a leaf four repos bind to cannot inherit the
dependency closure of the heaviest package in the family. Two facts point the
same way, and only one of them is settled:

- **Samsung refused core on the record, on *shape*** — core's L1 is an in-memory
  SQLite cache while samsung's SQLite file must *be* the store. That refusal is
  settled. Its separate weight rejection was of `3tears-models`, not of core;
  conflating the two overstates what samsung actually recorded.
- **The Python floor points the same way today** (core declares `>=3.14`,
  discodon is on 3.12) but is *not* settled. Open question 1 is live between
  moving discodon to 3.14 and making the minimum a per-module statement with a
  relaxed subset, with no recommendation recorded. Either outcome is compatible
  with what follows.

#### Core's weight was checked rather than assumed, and the check moves this argument

Read 2026-08-03: `import threetears.core` pulls none of sqlalchemy, asyncpg,
aiosqlite, httpx, cryptography, pyjwt or pydantic, and `http_client`, `egress`
and `coordination.token_bucket` — the three primitives this capability wants
most — each import clean.

So core's cost is an **install** cost, not a runtime one, and the `MemoryMax`
framing does not apply to importing it. Reading the hard-dependency list against
actual use gives a ruling per entry:

| Dependency | Ruling |
|---|---|
| `aiosqlite` | **Unused — remove.** Zero references anywhere in the monorepo outside the line declaring it. L1 uses stdlib `sqlite3`, synchronously. Added 2026-03-13 in a commit about unrelated packages. |
| `sqlalchemy` | **Optional — make it an extra.** Four of its seven users already lazy-import it inside function bodies. Of the three module-level sites, `testing/sqla_parity.py` sits under `core.testing`, which the pyproject already treats as extras territory; the other two — `models.py` (self-described optional ORM mixins) and `collections/flush.py` — are imported by nothing but their own tests. It is a SQL builder and type mapper here, not an ORM on any live path. |
| `asyncpg` | **Optional — make it an extra.** Imported exactly once, in that same unreachable `flush.py`. L3 arrives as an injected `L3Backend` / `DurableStore` protocol, so core generates SQL and the host supplies the pool. |
| `httpx`, `pydantic`, `uuid-utils` | **Required**, and light. |
| `cryptography`, `pyjwt[crypto]` | **Required by `core.security`** — a real subsystem, not dead weight, though it would take an extra cleanly. |

Two consequences:

- **For 3tears** — one dead dependency and two extras. A `packaging` change with
  a real payoff for every constrained consumer, and it belongs in that repo's
  backlog rather than in this document.
- **For search** — **SR-L7 survives but on narrower grounds than it was written
  on**: the layering rule and the unresolved Python floor, not runtime weight.

Worth knowing before somebody argues the seam is unnecessary. It is a layering
decision and should be defended as one, not as a weight workaround that a
`pyproject` cleanup would dissolve.

*Driven by: G10, G12, G13, P9 · SR-L3 through SR-L7 · checks 5, 9, 10.*

### 6. One injected transport seam, and the exit is part of the contract

Adapter never opens a client. It reaches upstream through a declared transport
protocol, which `core.http_client` satisfies by shape wherever core exists and a
bare-httpx implementation satisfies on the Pi.

That one seam supplies four things Part II otherwise asks for separately:

- configurable timeouts
- bounded retry
- circuit-breaking on provider exhaustion
- a per-call span

**Which exit a call leaves by is an input at Adapter and provenance on the
result** — `direct` included as a named value rather than an absence. It is
per-upstream, because a deployment routing external search through a proxy must
not route its own internal SearXNG the same way. Rate and ban budgets key on
`(provider instance, egress)`, since a ban is issued against an address.

**One rule change asked of 3tears.** This is the one place the architecture asks
3tears to change a rule rather than answer a question: `test_no_bespoke_reuse`
should accept a declared transport protocol as sanctioned, so the norm reads "no
bespoke client" rather than "no client outside core" — which *widens* its reach
to lightweight leaves that today escape it by being unable to comply.

*Driven by: G8, G11, G13, P2, P9 · SR-N1–SR-N4, SR-G4 · checks 11 and 12.*

### 7. Packaging

- Contract types ship as a **dependency-free leaf**, per the ratified
  contracts-leaf pattern.
- Provider adapters sit **behind extras** — `[searxng]`, `[tavily]`.
- `agent-tools` **consumes** the leaf rather than growing it. Its hard
  dependencies on langchain-core, memory and NATS are exactly what the
  constrained consumers refuse.

*Driven by: G7, G10 · SR-L3, SR-L4 · check 6.*

---

## What consumers do

High level. Every row is a code change; none is a rewrite.

| Repo / package | What it does |
|---|---|
| `3tears` `agent-tools` | Gut `WebSearchTool` into a Bind — same name, ABC and result shape, structure on `metadata` under a named key. Two envelope gaps land here as asks from *every* pod-served tool: metadata is dropped when an exception escapes, and there is no per-call deadline. |
| `3tears` `scrape` | Nothing at its call sites. `page_finder` starts receiving structure instead of flattened text (check 4). |
| `3tears` context-save node | Stop binding search on a bare tool-name string and persisting truncated flattened text; read structure off `metadata`. This is where retrieved content becomes retained content, so the retention posture gets stated here. |
| `3tears` `media-contracts` | Three fields added; nothing moves out. |
| `3tears` `core` | Nothing moves. One norm widening asked (piece 6). |
| `metallm` | Delete both side-steps — the raw SearXNG helper and the app-side extraction wrapper — rather than wrapping them (check 1). |
| `discodon` | Two implementations collapse to one, and its budget semantics move upward. Search spend enters the eval cost cap; research search becomes replayable. Embedded first, pod-resident after convergence, no consumer change at the switch (checks 3, 10). |
| `samsung` | Builds phase 2 image search on this rather than forking (check 2), supplying its own transport and replay store, calling synchronously from a one-shot `asyncio.run()` (checks 5, 9). |

Samsung is the load-bearing test, and not because it is the hardest engineering.
It is the consumer least like the ones that shaped today's code — a program not
an agent, images not pages, deep criteria not none, a record not prose — and its
requirements are written down while its search is not yet built. If the shape
serves it without a fork, the decomposition was real.

## What this is not

Boundaries with existing owners, stated in full in `search-requirements.md` §2.

- **Not a ranker** — Select composes one.
- **Not a scraper** — that is `3tears-scrape`.
- **Not RAG** — that is `agent-memory`.
- **Not a telemetry sink** — it returns records; the host emits them.
- **Not an answer engine, and not an agent** — deciding what a result means, and
  deciding what to search for, both stay above.

## What is still open

Stable against most of `search-requirements.md` §13. Four items change the shape
rather than the detail:

- **How many score dimensions** (SR-A4). One `score` field forecloses samsung's
  phase 2, whose `confidence`/`quality_score` split is already designed and
  reasoned. Until it is ruled, the core's ranking shape is provisional.
- **Model-mediated search, in or out** (SR-B5 / open question 21). Recommended
  out of Adapter and Call, in at Aggregate as a candidate producer. Ruling
  otherwise adds a layer.
- **Who stores a replay recording** (SR-F5). A consumer-supplied store port is
  recommended; search owning a store pulls a backend into the leaf and
  contradicts piece 5.
- **Whether the no-bespoke-client norm widens** (SR-N1). A refusal leaves the seam
  intact but ships it with a filed exemption — the outcome check 11 exists to
  prevent.

And one assumption worth flagging rather than burying: the six layer names are
proposed in the requirements doc, not derived from any owner's recorded position,
and every requirement is attributed to that cut. Re-cutting it ripples. The fewer
of those names that survive into the contract as *types* rather than as
vocabulary, the cheaper that stays.
