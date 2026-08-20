# Shared Web Search and Fetch — Design Exploration

**Status:** Exploration — 2026-08-02. Not a proposal; nothing here is ratified.
**Superseded in part:** `search-requirements.md` §11 reads this sketch against
the requirements and contradicts five of its field choices plus two structural
ones. **Read that section before building anything here.** In particular, do not
take "httpx-only" (`:103`, `:150`) at face value: a stored `httpx.AsyncClient`
fails `tests/enforcement/test_no_bespoke_reuse.py`, whose sanctioned target is
`threetears.core.http_client`. The requirements doc's §N reconciles that with the
Pi's refusal of `core`, and adds the egress seam this sketch does not mention.
**Companion to:** `family-convergence.md` §4.14, which records the *direction*
(one contract, staged pipeline). This document sketches the *mechanism* for
evolving the common tools, so the thinking survives until someone cuts a real
proposal. Open questions 20–21 there still govern sequencing and the
model-mediated-search seam.
**Overtaken by:** `search-architecture.md`, which states the structural shape
this sketch was reaching for — six layers, a carrier-neutral core, one injected
transport seam, and no path from the leaf into `threetears.core`. Where this
document and that one disagree, that one is current.
**Fully superseded 2026-08-04:** `search-spec.md` is now the buildable
statement — decisions taken, modules, sequencing. Nothing here should be built
from directly. Read the five in the order direction → need → shape → spec, and
treat this as the exploration that got there first.

---

## 1. What exists today

Both builtins live in `3tears-agent-tools` (`builtin/web_search.py`,
`builtin/web_fetch.py`) and share the same architecture problem: each fuses
provider, contract, and presentation into one module.

**`WebSearchTool`** (~160 lines) is a SearXNG passthrough: `GET
{base_url}/search?q=…&format=json`, flatten the top 10 into numbered text
(title / URL / snippet), return the string. One input (`query`); none of
SearXNG's knobs (category, time range, language, paging, engines) are
exposed; relevance scores and engine attribution are discarded at
`_format_results`. No cost or budget concept; SearXNG is hardcoded.

**`WebFetchTool`** is further along: credential-resolver hook with
non-leaking failure logging, meta-refresh and JS-redirect following, a crude
429/403 retry, trafilatura behind the `[fetch]` extra with a regex fallback,
stub rejection (`_validate_content`, <50 chars), 15k-char output cap.

The consumers tell the story. metallm side-steps both builtins for
structured access (`_searxng_query` in `admin/models.py`,
`web_fetch_utils.py`); discodon built its Tavily wrapper twice; scrape's
`page_finder` composes on the search builtin and inherits its flattened
output.

### Bugs worth fixing now, independent of any redesign

- **Sync httpx clients inside `async execute`** — both tools. A slow search
  blocks the event loop for up to 15s; a slow fetch, 30s.
- **`time.sleep(1)` in the fetch retry loop** — same blocking problem.
- **Unbounded download** — `resp.text` with no byte cap or content-type
  gate; a huge page is a memory incident on a `MemoryMax`-capped host.
- **Errors by string prefix** — success is `not
  content.startswith("[TOOL ERROR]")`.

## 2. The shape: four layers

Split what today is fused. Each layer is consumable without the ones above
it.

### Layer 0 — contracts (typed, dependency-free)

Per the contracts-as-a-leaf pattern. Pydantic-only sketch:

```python
class SearchQuery(BaseModel):
    query: str
    max_results: int = 10
    category: str | None = None        # searxng categories; tavily ignores
    time_range: str | None = None
    include_domains: list[str] | None = None   # discodon lineage
    exclude_domains: list[str] | None = None

class SearchResult(BaseModel):
    url: str
    title: str
    snippet: str
    score: float | None                # kept, not discarded
    engine: str | None                 # searxng attribution
    published_at: datetime | None = None
    raw: dict[str, Any]                # provider passthrough, never load-bearing

class SearchResponse(BaseModel):
    results: list[SearchResult]
    provider: str
    cost: Cost                         # Cost(amount=0) for searxng; credits for tavily
    took_ms: int

class FetchResult(BaseModel):
    url: str
    final_url: str
    status: int
    text: str | None
    title: str | None
    method: str        # "trafilatura" | "regex-fallback" | "heavy" | "wayback"
    truncated: bool
    cost: Cost
```

Alongside: typed errors (`ProviderError`, `BudgetExceeded`, `FetchBlocked`)
replacing string-sniffing, and a `Budget` protocol —
`check(estimate)` / `record(cost)` — generalizing discodon's daily +
per-invocation discipline so it is provider-independent. Cost is in the
contract because credits are real money; SearXNG's zero and Tavily's credits
are the same field.

### Layer 1 — providers behind extras

A `SearchProvider` protocol (`async def search(SearchQuery) ->
SearchResponse`) with capability metadata, mirroring how `3tears-models`
describes models (supports_time_range, supports_domains, cost_model).

- **`SearxngProvider`** (`[searxng]`, httpx-only): exposes the knobs the
  current tool hides; keeps scores and engine attribution. One teaching
  error from the hallucinote conventions: when SearXNG returns 403 because
  `json` is missing from `formats` in settings.yml — the #1 setup failure,
  today a bare HTTP 403 — the error names the line to add.
- **`TavilyProvider`** (`[tavily]`): lifted from discodon's wrapper, which
  owns the hard-won semantics — `search_depth` with credit costs surfaced
  in `Cost`, domain scoping, score coercion.

### Layer 1b — fetch as a cascade, heavy tier injected

- **Tier 1 (default):** today's httpx + trafilatura path made properly
  async, with a streamed download and byte cap, a content-type gate, and
  retry-with-backoff replacing `time.sleep`.
- **Tier 2 (optional, dependency-inverted):** a `HeavyFetcher` protocol
  slot that `3tears-scrape` *implements*. agent-tools never imports scrape;
  camoufox/playwright arrive only where an app wires them in. Escalation
  trigger: tier 1 fails `_validate_content` (the stub detector already
  exists).
- **Tier 3 (optional):** Wayback fallback — from the `TadMSTR/searxng-mcp`
  prior art (§4.14's investigation list).
- **Per-domain politeness:** rate limiting via
  `threetears.core.coordination.token_bucket` — the primitive already
  exists in core; dogfood, not new machinery.

### Layer 2 — pipeline

One composition helper: search → URL canonicalization/dedupe → optional
rerank → bounded-concurrency fetch of top-k → typed entries. Rerank stays a
protocol slot — MMR from `agent-memory` now; a cross-encoder or Voyage
reranker later, arriving through models (a cross-encoder drags torch, so it
is never a default). Budget and observe spans thread through; the
circuit-breaker pattern comes from models.

### Layer 3 — presentation: the existing tools become renderers

`WebSearchTool` / `WebFetchTool` keep their MCP names and grow their input
schemas (category, time range, max_results). `ToolResult.content` stays the
LLM-friendly text render; **`ToolResult.metadata` — a field that already
exists — carries the structured `SearchResponse` / `FetchResult`**. That is
the migration path: metallm's two side-steps retire without any consumer
breaking.

## 3. The packaging decision

Where do layers 0–1 live?

- **Option A — a new slim package** (contract + providers; httpx-only base;
  `[searxng]` / `[tavily]` extras), with agent-tools consuming it.
- **Option B — grow inside agent-tools** behind extras. Fewer packages, but
  agent-tools hard-requires langchain-core (and, per the coupling audit,
  memory and NATS).

Recommendation: **A.** The consumer that most wants bare search is samsung's
discovery, which cannot take agent-tools' hard deps. This would be the first
new package built under "offer everything, require a bare minimum" from day
one, and it should look like it.

## 4. Migration

- Builtins keep names and MCP identity; schemas widen, output gains
  structure via `metadata`.
- metallm retires `_searxng_query` and `web_fetch_utils`; discodon's
  research tools adapt onto the contract, contributing the budget hooks.
- Conformance tests per provider (the same idea as the `DurableStore`
  conformance direction): one shared suite every provider must pass, plus a
  live-credentials tier for Tavily.

## 5. Open questions

1. **Response caching.** Where does it live? Core collections have an opt-in
   L1 max age (`CollectionRegistry.set_l1_max_age`), but it is not TTL: it
   bounds how long an L1 row cached *from a lower tier* is served, and a
   collection with no L3 pool is refused a bound outright, because an expired
   row there reads as "does not exist" rather than as a miss that repairs.
   A response cache has no lower tier to pull through from, so it is exactly
   the shape that refusal excludes — don't force the dogfood; an httpx-level
   or app-side cache may be the honest answer.
2. **robots.txt.** A stance, or adapter-side? Currently unaddressed
   everywhere in the family.
3. **Heavy-tier escalation.** Auto-escalate on stub detection is convenient
   but can silently multiply the cost of a fetch; explicit mode is safer and
   more annoying. Pick one and say why.
4. **Model-mediated search** — family-convergence open question 21; the
   contract seam decision lands there, not here.
5. **Rerank protocol home** — models (it is a model call) vs. the pipeline
   package (it is a stage). Leaning models; undecided.
