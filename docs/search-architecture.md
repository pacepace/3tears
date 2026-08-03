# Search: The Architecture

**Status:** Draft — 2026-08-03
**Companions:** `search-requirements.md` states the need and carries the
evidence; `family-convergence.md` §4.14 records the direction; `shared_search.md`
is the earlier mechanism sketch, superseded where the two disagree.
**Scope:** structural shape and adoption. No fields, no signatures, no sequence.

## Summary

Eight call sites across four repos reach the web today, through four
implementations, two of them written *because* the shared one did not fit. This
says what changes in the shape of the code, what each consumer does about it, and
what drives each piece.

If one line survives: **structure is the primitive; rendering is a binding.** The
rest is what that costs once you hold to it across four repos, two deployment
modes, and a carrier list that is open by ruling. Drivers are cited as
requirements-doc IDs — `G*` goals, `P*` principles, `SR-*` requirements, and the
numbered success checks in §3.

---

## What changes

**1. Formatting moves to the edge.** Today the builtin flattens ten results into
a numbered string and everything downstream either accepts that or reimplements
search. The structured result becomes the thing that exists; the LLM-shaped
string becomes one rendering, produced at Bind. `WebSearchTool` keeps its
`threetears.web_search` name, its `TearsTool` ABC and its `ToolResult` shape —
this is a gutting, not a replacement — and structure rides `ToolResult.metadata`
under a key of its own. That channel is verified to survive the NATS hop, so one
migration covers both deployment modes.
*Driven by: G1, G6 · SR-A1 · check 8 · §4.14's ruling that formatting is
presentation.*

**2. Six layers, each a legitimate stopping point.** **Adapter** (a request → one
provider's API) · **Call** (a query → one candidate set) · **Aggregate** (many
calls → one set) · **Extract** (a carrier → the information in it) · **Select**
(candidates + criteria → an ordered subset) · **Bind** (candidates → what the
caller consumes). A consumer stops wherever it likes and gets full fidelity
there — metallm's price lookup takes Adapter through Extract and binds itself;
samsung's phase 2 needs all six. A fused pipeline helper that bundles dedup,
rerank and fetch means wanting one costs you three, which is how the current
code got here. Select owns criteria
negotiation and the cull and exposes a *slot* a ranker plugs into; it never owns
a ranking implementation, because `agent-memory` ships MMR, `3tears-models`
carries rerank metadata, and a cross-encoder drags torch onto a Pi that cannot
carry it.
*Driven by: G7, P3, P4 · SR-L1 · §4.14's rerank ruling.*

**3. The result core is carrier-neutral, and its facets are found rather than
invented.** Images, PDFs, video and datasets are in scope by ruling, so a closed
`web | image | pdf` union is prohibited: the core carries identity, provenance,
scores and available fidelity, facets are additive, and a consumer that does not
recognise one ignores it. The facets come from `3tears-media-contracts` — already
dependency-free, already a hard dep of `agent-tools`, already carrying the
carrier taxonomy and `extraction_status`. Three fields are genuinely missing —
rights status, pixel dimensions, direct-file versus containing-page — and they
belong there rather than in a search leaf inventing a fourth media vocabulary.
*Driven by: G3 (ruled 2026-08-02), G13 · SR-C1, SR-C2, SR-C3 · checks 7 and 13.*

**4. Cross-cutting concerns attach where the fact arises, not to a class.**
Spend, budget, telemetry and record/replay attach at Adapter and Call. The
evidence is sharp: discodon attached replay to `Tool.act()`, its research sub-tool
left the `Tool` hierarchy for good reasons, and reproducibility silently left with
it — every research eval now re-issues live searches and measures the web's
drift. One consequence to carry rather than discover: search stops being the
stateless tool `agent-tools` classifies it as, since per-call spend, per-day
budget scope and a replay key are all state.

**Spend is every resource a call consumed, not only money** — dollars,
wall-clock, provider quota and call count, bytes moved. Modelling it as currency
is the trap, because the two constraints that actually bite here are unpriced.
Self-hosted SearXNG costs nothing and its failure mode is a ban, so a mechanism
keyed on cost never fires for it. And a call that burned twelve seconds and
returned nothing because a budget was already spent is pure latency with zero
coverage gain — the run that paid for it needs that as its own number, not as a
zero. All of it has to survive the failure path too: a search that timed out
still spent time and quota, and dropping that under-reports by exactly what the
failures cost. So Bind catches every typed exception and renders it as a failed
result carrying the spend — nothing raises across the wire.
*Driven by: P5 · SR-D1, SR-D6, SR-E1, SR-E3, SR-F3, SR-I1, SR-I2, SR-I3 ·
check 3.*

**5. The leaf depends on nothing the smallest consumer refuses.**
`threetears.core` holds most of the primitives this capability wants — traced
transport, egress selection, the token bucket, secret resolution — and also
hard-requires sqlalchemy, asyncpg, cryptography and pyjwt. Samsung has refused it
on the record, on weight, and that refusal is settled. The Python floor points
the same way today (core declares `>=3.14`, discodon is on 3.12) but is *not*
settled: the family is actively weighing whether to standardise on 3.14 or set
minimums per module, and either outcome is compatible with what follows. The
design does not rest on it — it rests on the weight, and on the general rule that
a leaf four repos bind to cannot inherit the dependency closure of the heaviest
package in the family. So there is **no path from the contract leaf or the
Adapter into `threetears.core`**: everything core provides arrives as a shape the
leaf declares
and the host injects — the move 3tears already makes with itself twice, in
`core.http_client`'s injected circuit breaker and in `media-contracts`'
dependency-free store protocols. The permitted floor is `3tears-observe`,
`3tears-media-contracts`, pydantic, and a provider's transport behind an extra.
One leaf then serves both modes: embedded on a `MemoryMax`-capped Pi with no
broker, callable from a one-shot `asyncio.run()`, and pod-resident as a
`TearsTool` over NATS. Where the wire hop falls in the stack is undecided, so
every layer boundary is wire-serialisable and any of them could be it — paid in
design freedom, not at runtime.
*Driven by: G10, G12, G13, P9 · SR-L3 through SR-L7 · checks 5, 9, 10.*

**6. One injected transport seam, and the exit is part of the contract.** Adapter
never opens a client; it reaches upstream through a declared transport protocol,
which `core.http_client` satisfies by shape wherever core exists and a bare-httpx
implementation satisfies on the Pi. That one seam supplies four things Part II
otherwise asks for separately: configurable timeouts, bounded retry,
circuit-breaking on provider exhaustion, and a per-call span. Which exit a call
leaves by is an input at Adapter and provenance on the result, `direct` included
as a named value rather than an absence — and it is per-upstream, because a
deployment routing external search through a proxy must not route its own
internal SearXNG the same way. Rate and ban budgets key on
`(provider instance, egress)`, since a ban is issued against an address. This is
the one place the architecture asks 3tears to change a rule rather than answer a
question: `test_no_bespoke_reuse` should accept a declared transport protocol as
sanctioned, so the norm reads "no bespoke client" rather than "no client outside
core" — which *widens* its reach to lightweight leaves that today escape it by
being unable to comply.
*Driven by: G8, G11, G13, P2, P9 · SR-N1–SR-N4, SR-G4 · checks 11 and 12.*

**7. Packaging.** Contract types ship as a dependency-free leaf per the ratified
contracts-leaf pattern; provider adapters sit behind extras (`[searxng]`,
`[tavily]`). `agent-tools` consumes the leaf rather than growing it — its hard
dependencies on langchain-core, memory and NATS are exactly what the constrained
consumers refuse.
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
Not a ranker (Select composes one), not a scraper (`3tears-scrape`), not RAG
(`agent-memory`), not a telemetry sink (it returns records; the host emits them),
not an answer engine and not an agent — deciding what a result means, and what to
search for, stay above.

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
