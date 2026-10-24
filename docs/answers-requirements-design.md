# Grounded Answers: What the Family Needs, and the Shape

**Status:** Draft for stakeholder review -- 2026-10-22. Requirements and design
in one document, because the capability is small enough that splitting them
would be two documents saying one thing. Decisions below are **vetoable
rulings** in the `search-spec.md` sense: each adopts a recommendation and says
why; a veto is recorded here and propagated.
**Companions:** [`search-requirements.md`](search-requirements.md) states the
need for search and owns the requirement IDs (`G*`, `P*`, `SR-*`) and success
checks cited here; [`search-spec.md`](search-spec.md) is the buildable statement
for the search leaf this extends. This document adds **one amendment** to
`search-requirements.md` §2 (owner ruling, 2026-10-22; text in §13 below) and
otherwise cites rather than restates.
**Owner rulings taken 2026-10-22, before this draft:** paid search comes back
as an *option*, and discodon will use it immediately; and the charter amendment
in §13 is accepted. Both are recorded in §8 as AD1 and AD2.

Everything below was written against `3tears` at develop `938ebdbdd`
(2026-10-22, post-#911) and discodon at `fix/tool-start-budget` `678e54cca`
the same day, and against Brave's published documentation read the same day.
Where a provider fact could not be verified without a live key it is marked
**(verify live)** and listed in §12.

## Summary

Brave sells three things under one key. Two are search-shaped and already fit
the search leaf's contract -- Web Search (which discodon's in-cluster SearXNG
already fronts) and LLM Context (search plus extracted page text in one call).
The third, **Answers**, is not search: one question in, a cited,
LLM-synthesised paragraph out. It is OpenAI-compatible on the wire and not a
chat model in substance (exactly one user message, no history, no tools), and
its citations exist only when streaming, as inline tags in the token stream.

The question this document answers is where a provider's *synthesis* belongs
in a family whose search charter says "we do not summarise, judge, or
conclude", and the answer is: **in the search leaf, as a second contract beside
the candidate set, carried but never produced.** The answer text is a new
type; the sources it cites are `Candidate`s, so everything the family already
does with a candidate -- page reads, grounding, provenance, spend, failure
taxonomy, budget and pacing ports -- applies unchanged. Nothing a search-only
consumer installs gets heavier.

The immediate consumer is discodon, and reading its research tool changed the
design in specific ways (§6). Brave Answers must not replace that tool's inner
agent: the tool's value is a typed findings contract and a grounding gate
computed against retrieved page text, neither of which a synthesised paragraph
supplies. It fits *underneath*: first as the engine of `fact_check`, which runs
inline on the persona's turn today, and then as a candidate sub-tool the inner
agent may ask a question of, whose citations flow into the same page reads and
gate as a search result does.

What is deliberately not here: Brave's research mode (a loop with fewer levers
than discodon's own, and an undocumented citation format), a chat-model
adapter in `3tears-models`, and a separate package. §7 says why each was set
aside.

---

# Part I -- What we need

## 1. What we are trying to achieve

**AG1. A cited answer is information with its sources attached, and we carry
both.** G1 says the unit is information, not a link; G4 says every piece of it
carries a source you can re-check. A provider that returns a paragraph with
citations is delivering a fifth rung of G1's fidelity ladder -- above extracted
text, a synthesis *of* extracted text -- and we take it on the same terms as
the other four: the text as the provider gave it, the sources as candidates,
the spend as spend. We do not write the paragraph. We do not edit it.

**AG2. Never an answer without its sources.** An answer rendered without at
least one citation, or without an explicit statement that the provider
returned none, is the silent-partial-answer defect P8 exists to prevent, in
its most dangerous form: fluent, confident, and unfalsifiable. Brave's blocking
mode returns exactly that, so the capability never uses it (AD5).

**AG3. The bill is the bill.** Brave meters two things per call -- searches
and tokens -- and the search charge is roughly half of a typical single
answer. A spend that counts tokens alone under-bills by about 2x, which is the
SR-E4 defect (Tavily `advanced` = 2 credits) in a new coat. Spend is read from
what the provider says it billed, never recomputed from a published formula
(D4: budget follows the bill). Today Brave spend made through discodon shows
only on Brave's dashboard; after this it shows in discodon's budget rows and
eval cost (AC2).

**AG4. The consumer that synthesises itself keeps doing so.** discodon's
research tool synthesises on purpose, with a model it chose, steered by prompts
its operator tunes. This capability gives it one more gathering move, not a
replacement brain (AD3).

**AG5. One more provider costs nothing to the consumers who do not want it.**
The leaf's dependency floor is pinned (D24) and stays pinned. An adapter is
pure logic over an injected transport; the one new weight, a streaming read,
rides the `[standalone]` extra that already carries httpx.

## 2. What we are not trying to achieve

- **Not a replacement for discodon's research tool.** §6 is the evidence.
- **Not a chat model.** Brave Answers takes one user message. Presenting it as
  a `BaseChatModel` makes `bind_tools`, `with_structured_output` and multi-turn
  silently wrong (§7, Shape 1).
- **Not Brave's research mode, in v1.** It is a loop with fewer levers than
  the consumer's own, its in-answer citation format is undocumented, and it
  excludes `enable_citations`. The contract is written so that adding it later
  is additive (AD6).
- **Not a second result shape.** Check 14 holds. The answer's sources are
  `Candidate`s; the answer payload rides its own metadata key with one
  construction site, never the search key (AD8).
- **Not a cache.** D14 stands. Brave's own per-answer freshness is the
  provider's business.
- **Not synthesis.** The §2 amendment (§13) permits carrying a provider's
  synthesis; it does not permit producing one. No stage here calls a model.
- **Not a UI.** Streaming an answer *to a person* as it arrives is a later,
  additive face on the same provider seam (§8, AD12). v1 is request/response,
  because every named consumer is a tool.

## 3. How we would know it worked

Checks, not sentiments. Numbered for reference; new checks append.

1. **discodon's `fact_check` runs on it**, and its inline wall-clock drops
   from a multi-round inner-agent loop to one provider call (Brave's stated
   single-search p50 is under 4.5 s to first token). Measured on prod, not
   assumed.
2. **Brave spend appears in discodon's budget rows and eval cost**, in money
   and in searches, from the provider's own usage report -- not only on
   Brave's dashboard.
3. **No answer renders without a citation or an explicit notice** that the
   provider returned none. Pinned by the conformance suite.
4. **A stream cut off before the usage report fails typed**, carrying the
   spend it can state (one call) and naming the bill it cannot, never a
   fabricated estimate (SR-E3, D4).
5. **A second answers provider is added without touching a consumer** (the
   search family's check 6, restated). Perplexity Sonar, Exa `/answer`, and
   Tavily's own `answer` field are the candidates the contract is shaped for.
6. **The search-only install is unchanged in weight.** D24's floor test stays
   green with no edit to the ruled list.
7. **One construction site for the answer metadata key**, held by the same
   enforcement mechanism that holds the search key.
8. **discodon's eval seams are untouched.** A cell that substitutes a seeded
   research delivery, or replays a cassette, behaves exactly as before.
9. **A citation that survives into discodon's grounding corpus is page text or
   a SERP-grade snippet, never the provider's prose.** The gate's
   anti-fabrication contract is not weakened by the arrival of a synthesising
   provider (§9 G).
10. **The `[standalone]` SSE path is the only new socket-owning code**, in the
    module D19 already sanctions, with no exemption filed.

## 4. The provider, as it is

Read from Brave's documentation on 2026-10-22; field names verbatim.

### 4.1 Three products, one key

| Product | Endpoint | Shape | Fits today's contract? |
|---|---|---|---|
| Web Search | `/res/v1/web/search` | SERP: titles, URLs, snippets | Yes -- and discodon's in-cluster SearXNG already runs the paid `braveapi` engine behind a one-hour result cache, so a direct adapter is redundant for that consumer |
| LLM Context | `/res/v1/llm/context` | `grounding.generic[{url, title, snippets[]}]` plus `sources` keyed by URL; token-budgeted | Yes -- `Candidate` with `ContentSlot(origin="provider-response")`; Extract is already ruled a no-op when the provider supplied content (SR-A2) |
| **Answers** | `/res/v1/chat/completions` | OpenAI-compatible chat completion; cited, synthesised answer | **No** -- the subject of this document |

### 4.2 Answers: the facts that shape the design

- **One user message.** `messages` must contain exactly one user message. No
  system prompt, no history, no tools.
- **Citations only when streaming.** `enable_citations=true` requires
  `stream=true`; the blocking response is bare text. Citations arrive *inline
  in the content stream* as `<citation>{...}</citation>` tags whose JSON carries
  `start_index`, `end_index`, `number`, `url`, `favicon`, `snippet`. A tag can
  straddle chunk boundaries.
- **The usage report closes the stream** as a `<usage>{...}</usage>` tag:
  `X-Request-Requests`, `X-Request-Queries`, `X-Request-Tokens-In`,
  `X-Request-Tokens-Out`, the per-dimension `*-Cost` keys and
  `X-Request-Total-Cost`. In blocking mode the same keys are response headers.
- **Pricing** is published and flat: $4 per 1,000 searches plus $5 per million
  tokens in and out. A worked example in Brave's own docs -- 2 searches, 1,234
  in, 300 out -- bills $0.01567, of which the searches are $0.008.
- **Rate limit** 2 requests per second by default. A monthly credit limit is
  "checked before answering": a refusal here is the provider's own about
  money, which is `QuotaExhausted` territory (SR-D3), distinct from a local
  cap.
- **Knobs:** `country`, `language`, `safesearch` (`off|moderate|strict`),
  `max_completion_tokens`, `web_search_options.search_context_size`
  (`low|medium|high`).
- **Research mode** (`enable_research=true`): streaming only, incompatible
  with `enable_citations`, up to 50 queries, 5 iterations and 300 s, emitting
  `<queries>`, `<analyzing>`, `<thinking>`, `<progress>`, `<blindspots>`,
  `<answer>` and `<usage>` tags. Citations are "built into the `<answer>`
  tag" in a format the documentation does not specify. Brave's own p99 on
  SimpleQA is 53 queries and about 300 s.
- **Timeouts** Brave recommends: at least 30 s single mode, at least 300 s
  research.
- **Auth:** `X-Subscription-Token` header (or Bearer). discodon's SearXNG
  secret already holds `BRAVE_ANSWERS_API_KEY`, provisioned and unused.

**(verify live)** Whether `start_index`/`end_index` are character or byte
offsets, and whether they index the answer text before or after the tags are
stripped; and whether `snippet` is the cited page's own text or model-written.
Both change a contract field and both are a five-minute check with the key
discodon already holds.

## 5. Who consumes this

The same two deployment modes and three faces as search (`search-requirements.md`
§5.4-5.5). The named consumers, in the order they will arrive:

| Consumer | What it wants | Face |
|---|---|---|
| discodon `fact_check` | one claim → a cited verdict, inline on the persona's turn, fast | embedded, in-process |
| discodon research inner agent (`research__ask_web`, experimental) | a synthesised lead in one call for asks keyword search is bad at; citations into the grounding corpus | embedded, in-process |
| `agent-tools` `threetears.web_answer` builtin | prose on `content`, structure on `metadata`, `ToolResult` shape; opt-in because every call costs money | mesh tool / HTTP operation / MCP (check 14) |
| chat UIs (later) | the answer streamed as it arrives, citations as cards | a streaming face on the same provider seam, D-S1..D-S5 vocabulary |

`scrape`'s `page_finder` is a plausible fourth: "find the page that holds X"
is a question an answers provider answers in one call. Not designed for here;
it is a search-family consumer and gains the tool for free through
`agent-tools`.

## 6. What discodon's research tool taught us

Read on 2026-10-22: `discodon/tools/research_tool.py` (4.9k lines), its
sub-tools, the grounding gate, the eval seams, the target-design record
(`.prawduct/artifacts/research-target-design.md`, 2026-07-14) and the
September campaign (`campaign-research-agent-2026-09.md`), plus answers from
the discodon coordinator the same day. Each finding below changed a decision in
§8.

**6.1 It is an autonomous inner agent, not a search wrapper.** A larger model
(prod Kairo: `google/gemini-3.5-flash-lite`; 3 rounds, 12 searches max, 16,384
output tokens) runs a bounded tool loop: `research__web_search` (SearXNG, then
Extract reads each of the top 8 results' pages), then
`research__submit_findings`. The persona is deliberately out of the loop: it
passes `query`, an `intake` brief, structured `fields` with `required` flags
that gate stopping, and a checkable `done_when`. Budgets are a backstop; the
settled design says sufficiency is the stop. → AD3, AD6.

**6.2 Findings are a typed contract consumed structurally.** `ResearchFindings`
= `summary`, `sources`, `items[name, kind, fields{}, note, source_url]`,
`discoveries[insight, sources]`, `confidence`, `coverage`. The music consumer
reads `items`; six rubric dimensions judge the reply (grounded, calibrated
confidence, relays findings, answers facets, goes past canon, in voice). A
synthesised paragraph carries none of `items`, `discoveries`, `confidence` or
`coverage`; reconstructing them means a second model pass over the provider's
prose. → AD3.

**6.3 Grounding is computed, never self-reported.** After submit, code checks
every item: `source_url` must canonically match a URL the run retrieved, and
`name` plus every `fields` value must appear in *that page's* corpus text, by a
Unicode-correct match. The corpus text per URL is the SERP title and snippet
plus the page text Extract read. **A name appearing only in the snippet already
passes the gate**; the trace counts how often an item stood on a snippet alone
(`cited_sources_snippet_only`) because it is the degraded case. The coordinator
expects the owner to want cited text to be the page's own words, not the
provider's synthesis. → AD7, requirement AR-G1: a Brave citation `snippet`
enters the corpus with exactly the standing a SERP snippet has today, and the
cited URL is page-read through the existing path. No new grounding ruling is
needed, and the contract is not weakened.

**6.4 Latency is rounds-bound, not search-bound.** Pooled over 60 campaign
results, stop cause was `max_rounds` 39 / `end_turn` 13 / `submitted` 8: the
round budget binds about two thirds of the time and the search cap is never
reached (§12.5, §17.4). Dropping `max_rounds` 3→2 bought −8.1 s (−34%) per
call and a third of search spend, deferred pending eval repair (§16, §18). On
the new SearXNG path, one search plus 8 page reads took 7.5 s. So a faster
*search* does little; what would move latency is fewer rounds or an earlier
voluntary submit. → LLM Context is demoted (§7, Shape 4); `research__ask_web`
is an experiment whose hypothesis is "a synthesised lead ends the loop
sooner", to be measured, not assumed.

**6.5 `fact_check` is inline.** It runs the same loop synchronously on the
universe worker's turn loop (page reads hop to their own loop for that reason);
its latency adds directly to the persona's turn. One cited provider call is
the clearest win on the list. → AD4.

**6.6 Brave is already in the stack, and its spend is invisible.** The
in-cluster SearXNG runs the paid Brave Search engine on every uncached search;
the owner's recorded stance is "I won't top up tavily, we're moving to
searxng" (`build-plan-searxng-search.md`). Brave spend shows on Brave's
dashboard and nowhere in discodon's budget. → AG3, AC2; and the Web Search
product needs no adapter (§4.1).

**6.7 The seams are already shaped for a second sub-tool.** The inner loop
takes a *list* of `ResearchSubTool`s and merges their corpora by URL; each
declares a `MeteredUnit(provider, unit, units_per_call)`; `SearchFailure`
subclasses map onto `ProviderFaultKind` through one function; failure
ownership (candidate / provider / rig) is typed; evals substitute at the
*delivery* seam. One wrinkle: `units_per_call` is a constant and Brave's
searches per call vary (`X-Request-Queries`). → AD9 keeps the `SearchFailure`
taxonomy for answers so discodon's mapping is reused as-is; §11 names the
per-call units change on the consumer side.

## 7. Alternatives considered

**Shape 1 -- a `brave` provider in `3tears-models`.** `create_openai_chat(base_url=…,
extra_body={…})` works on the wire (confirmed: `ChatOpenAI` 1.3.5 exposes
`extra_body`), and a reply-observing wrapper in the `openrouter.py` pattern
could strip the tags. Rejected: a one-message endpoint posing as a chat model;
cost derived from per-token capabilities alone under-bills ~2x (AG3); no
`Spend`, no budget port, no typed refusal taxonomy, no provenance; and a
consumer wanting a cited paragraph as a *tool* would have to install langchain
and the heaviest package in the family to get it (AG5).

**Shape 3 -- a separate leaf, `3tears-answers`.** Same contract, own package,
depending on `3tears-search` for `Spend`, `Candidate`, the ports and the
errors. Rejected: the modularity gain is nominal -- by D24's own argument an
adapter weighs nothing, and the package would import search's contracts
anyway -- while the cost is a 31st lockstep package, a floor pin, a README row
and a release surface; and it splits the transport story, since the streaming
protocol belongs beside the two the leaf already declares (Gate A). It would
also hand discodon a second error vocabulary to map (6.7).

**Replacement -- Brave research mode instead of discodon's inner agent.**
Rejected on §6.1-6.3: it drops the findings contract, hollows the grounding
gate to snippets, and removes `done_when`, `required` fields, the
operator-tunable prompts and model choice -- levers the target design settled
and an active model bake-off depends on. Its latency is not obviously better
(p99 300 s, 53 queries). Revisit only if the owner abandons the structured
findings contract, which is a product decision, not a provider one.

**Shape 4 -- Brave LLM Context as the research gathering provider.** One call
returns up to 20 URLs with extracted text chunks; it fits the search contract
today and would retire the page-read step. Demoted, not rejected: latency is
rounds-bound (6.4); the owner just chose the SearXNG path, which LLM Context
would bypass along with its cache; and no post-SearXNG prod run exists to
compare against. Trigger to revisit: page-read failures or
`cited_sources_snippet_only` rising on prod traces. If it is built, it is a
`SearchProvider` in `adapters/`, with the Brave auth, spend and pacing wiring
from this document shared.

---

# Part II -- What that cashes out to

## 8. Decisions

| # | Ruling | Why | Build consequence |
|---|---|---|---|
| AD1 | **Paid search providers return, as an option** (owner, 2026-10-22) | discodon will use it immediately; SearXNG stays the self-hosted default | Nothing in the leaf changes stance -- paid providers were always representable (Tavily, D6); the consumer wiring in §11 is what makes it an option rather than a default |
| AD2 | **`search-requirements.md` §2 is amended** (owner, 2026-10-22) -- text in §13 | The leaf still produces no synthesis; it may carry a provider's, typed | The amendment lands with this document on the same branch; `search-spec.md` §5 takes the same rider when the build plan is cut |
| AD3 | **Answers is a layer under discodon's research tool, never its replacement** | §6.1-6.3 | Consumer wiring in §11 names two integration points and no change to the findings contract or the gate |
| AD4 | **`fact_check` is the first consumer; `research__ask_web` is an experiment** | 6.5 vs 6.4 | Sequencing in §12: the adapter ships for `fact_check`; the sub-tool lands behind a config flag with a measured hypothesis |
| AD5 | **The adapter always streams; blocking mode is not exposed** | Citations exist only when streaming (4.2); AG2 | `StreamingTransport` is a hard requirement of the adapter, not an optimisation; a host that injects no streaming transport cannot construct it, and is told why |
| AD6 | **v1 has no `mode` field; single mode only** | 6.1; research mode's citation format is undocumented | `AnswerRequest` gains `mode` additively later; `AnswerCapabilities.modes` declares `("single",)` so a consumer can branch before sending (SR-B4) |
| AD7 | **Sources are `Candidate`s; the answer text is the new type** | Everything that consumes a candidate -- page reads, grounding, provenance, spend -- applies unchanged (6.3, 6.7). Tavily's `answer` objection (a different shape from a candidate) is honoured by *containing* candidates rather than impersonating one | `GroundedAnswer.sources: tuple[Candidate, ...]`; Brave's citation `snippet` → `Candidate.snippet`, `content=None`, so Extract runs |
| AD8 | **Own metadata key, one construction site** | Check 14; `test_one_search_result_shape` guards the search key at its source and must do the same for this one | `ANSWER_METADATA_KEY = "answer"`, projection owned by `bind_answer`'s module, enforcement test extended per key |
| AD9 | **The search error taxonomy is the answers error taxonomy** | 6.7; SR-J1; one mapping on the consumer side | `AnswerProvider.answer` raises `SearchFailure` subclasses only; no `AnswerFailure` |
| AD10 | **Spend is read from the provider's usage report, never computed** | D4, AG3; Tavily precedent for unpriced-is-not-free | `money` = `X-Request-Total-Cost`; `provider_units` = `X-Request-Queries` as `"brave:searches"`; `calls` = 1; tokens typed on the answer, not folded into spend. The pre-call estimate counts one call and one search; money is estimated only when the host supplies rates |
| AD11 | **A third transport protocol, `StreamingTransport`, beside `SearchTransport` and `FetchTransport`** | Gate A: different obligations are different protocols; widening `SearchTransport` would invalidate every implementer | Declared in `contracts/transport.py`; implemented by `standalone.py` (httpx streams; same D19-sanctioned module, no exemption); a host-side `TracedHttpClient` adapter cannot implement it today (it buffers), which is the same situation `FetchTransport` records |
| AD12 | **v1 is request/response; streaming-to-UI is a later additive face** | Every named consumer is a tool (§5) | `AnswerProvider.answer` returns a whole `GroundedAnswer`; an `AnswerProvider.stream` method is reserved by name, not declared |
| AD13 | **The `web_answer` builtin is opt-in by its own alias, never inside `web`** | Every call costs money; a persona that opted into `web` for a free SearXNG search must not acquire a paid tool silently | `aliases.py` gains `answers` → `{"threetears.web_answer"}`; `web` is unchanged. **Owner may veto** the alias name |
| AD14 | **Research mode, LLM Context and a chat-model adapter are out of v1** | §2, §7 | None; recorded so nobody builds them by accident |

## 9. Requirements

`MUST` / `SHOULD` / `MAY` / `MUST NOT`. IDs are `AR-<section><n>`. Requirements
the search family already states are cited, not restated.

### A. What comes back

**AR-A1 (MUST).** `GroundedAnswer` carries: `text` (the provider's answer with
every tag removed), `sources: tuple[Candidate, ...]` in citation order,
`anchors: tuple[Anchor, ...]` (`start`, `end`, `source_index`; may be empty),
`provenance` (the provider instance, query, retrieval time, egress, producer
class -- SR-A3, on the answer itself as well as on each source), `spend`,
`notices`, `dispositions`, and typed token counts (`prompt_tokens`,
`completion_tokens`) as optional ints. `extra="forbid"` like every contract
type.

**AR-A2 (MUST).** Each source is a `Candidate` whose `identity` is the cited
URL, `locators` the canonical locator, `snippet` the provider's citation
snippet, `provenance.provider_ids` carrying at least `number` and the raw
citation JSON's non-text facts, `content=None`, `fidelity_available` =
`snippet`. P2: nothing the provider said about a citation is dropped.

**AR-A3 (MUST).** An answer with zero sources is a success value carrying a
notice that says so (SR-J2 shape; AG2). The prose rendering states it in the
first line, not a footnote.

**AR-A4 (MUST).** `anchors` are empty when the provider gave no usable offsets,
and a notice says so. They are never inferred by string search.
**(verify live)** before the offset semantics are fixed (§12).

**AR-A5 (SHOULD).** Unknown `<name>{json}</name>` tags in single mode are
removed from `text`, kept verbatim under `provenance.provider_ids`, and named
in a notice. Treating them as answer text would put provider machinery in a
persona's mouth; dropping them silently would break P2.

### B. What goes in

**AR-B1 (MUST).** `AnswerRequest`: `question: str` (one string, composed by the
caller; never `messages`), `criteria` using the search criteria vocabulary for
what overlaps (`safesearch`, locale) and namespaced keys for the provider's
own (`brave:country`, `brave:language`, `brave:search_context_size`,
`brave:max_completion_tokens`), `budget_scope_tags` as on `SearchRequest`.
Dispositions are owed per criterion (SR-B2, SR-B3).

**AR-B2 (MUST).** `AnswerCapabilities` declares `modes`, the pushdown /
unsatisfiable criteria, `citations_in_modes`, the pricing model
(`PRICING_PER_WEIGHTED_UNIT`, `metered_unit="searches"`) and the provider's
stated minimum timeout, so a consumer branches before sending (SR-B4).

**AR-B3 (MUST NOT).** No caller-supplied base URL (D21). The Brave base URL is
a deployment fact with a default; the API key is host-supplied (SR-K1), never
read from the environment.

### C. Spend

**AR-C1 (MUST).** Spend follows AD10. A stream that ends without a `<usage>`
tag records `calls=1`, zero money and zero units, and the failure record (or
the success notices, if the answer text arrived whole) names the bill as
unknown. Never an estimate wearing an observation's clothes.

**AR-C2 (MUST).** Money is `Decimal`, parsed from the provider's string;
`currency="USD"`.

**AR-C3 (SHOULD).** The pre-call estimate for `BudgetPort.check` is
`Spend(calls=1, provider_units=1, provider_unit="brave:searches")` plus money
only when the host passed `usd_per_search` and `usd_per_million_tokens`;
unpriced is not free (Tavily precedent).

### D. Budget and pacing

**AR-D1 (MUST).** `answer()` consults the same `BudgetPort` and
`RateLimiterPort` as `search()` does, in the same order and below the retry
boundary (D4, D5, D8). Pacing keys on `(provider_instance, egress)`; Brave's 2
rps default is the limiter's configuration, not the adapter's constant.

**AR-D2 (MUST).** A provider refusal on the monthly credit limit maps to
`QuotaExhausted`; a 429 to `RateLimited`; a local budget refusal to
`LocalCapExceeded`. The three stay distinguishable (SR-D3).

### E. Failures

**AR-E1 (MUST).** AD9. Every raise carries the spend consumed so far (SR-E3).
A stream broken after the first byte is `TransportFailed` with
`attempts` as the transport counted them; it is never retried after the first
byte, because the provider has billed the search (AR-C1).

**AR-E2 (MUST).** A tag whose JSON does not parse, or a `<usage>` whose cost
keys are absent, is `MalformedResponse`, carrying the raw tag in the failure
record's detail. Fail closed; a usage report we could not read is a bill we
cannot state.

### F. Transport and streaming

**AR-F1 (MUST).** `StreamingTransport` declares `egress_name` and
`stream(method, url, *, headers, json_body, timeout_seconds) -> StreamedResponse`,
where `StreamedResponse` exposes `status_code`, `headers`, `final_url`,
`egress` before the body, an async iterator of `bytes` chunks, and `elapsed_seconds`
/ `attempts` after close. Obligations as `SearchTransport` states them
(configurable timeout, bounded retry *before first byte*, breaker, span,
egress, SSRF rulings), plus: **a byte cap** on the whole stream (SR-G5's
rule, applied to a response that is open-ended by construction).

**AR-F2 (MUST).** The SSE framing (`data:` lines, `[DONE]`) is the adapter's
to parse, not the transport's; the transport moves bytes. The tag parser
buffers across chunks and across SSE events.

**AR-F3 (MUST).** `standalone.py` implements `StreamingTransport` alongside
the two it already implements. No other module opens a socket (check 10).

### G. Grounding posture (the discodon contract)

**AR-G1 (MUST).** A consumer that grounds against sources treats
`Candidate.snippet` from an answer exactly as it treats a SERP snippet: it may
seed a corpus entry, and the URL is page-read by the consumer's existing path
before anything is grounded against page text. Nothing in `text` -- the
provider's prose -- is ever grounding text.

**AR-G2 (MUST).** The prose rendering (`bind_answer`) marks the answer as the
provider's synthesis in its first line ("Answer from brave, N sources:"), so
a model reading it cannot mistake it for retrieved text.

### H. Packaging, weight, deployment

**AR-H1 (MUST).** D24's floor is unchanged. New modules are pure logic over
the injected transport. The SSE reader lives in `standalone.py` under the
`[standalone]` extra.

**AR-H2 (MUST).** Nothing in the leaf imports `threetears.core`,
`agent-tools`, langchain or NATS (SR-L7). The `web_answer` builtin lives in
`agent-tools`, which already depends on `3tears-search[standalone]`.

**AR-H3 (MUST).** An intra-family API addition ships in a MINOR bump
(`test_api_growth_requires_a_minor_bump`).

### I. Evals and reproducibility

**AR-I1 (MUST).** Canonical serialisation covers `AnswerRequest` (SR-F1), so
a future replay key and an eval run identity agree. Search-internal replay
(D26) is still unshipped; an answer in a live eval is live spend, as a search
is today, and D27's two-spend rule applies when replay lands.

**AR-I2 (MUST).** A recorded live stream is the fixture for the tag parser and
the spend reader, in `testing/recorded.py`'s shape. The fixture is captured
once with a real key and committed; the conformance suite runs against it.

### J. Telemetry

**AR-J1 (MUST).** One span per answer call (SR-I1), with the spend dimensions,
`provider_instance`, `egress`, source count, and whether anchors arrived, as
attributes. Token counts are attributes, not metrics labels.

### K. Security and conduct

**AR-K1 (MUST).** The question is untrusted user content (SR-K2). It is sent
as the one user message and nowhere else; no template wraps it.

**AR-K2 (MUST).** Citation URLs are provider output and go through the same
SSRF rulings as any other candidate locator when a consumer fetches them
(D21); the adapter does not fetch them.

## 10. The shape

Module placement in `packages/search/src/threetears/search/`:

```
contracts/
  answer.py          # AnswerRequest, Anchor, GroundedAnswer, AnswerCapabilities, AnswerProvider
  transport.py       # + StreamingTransport, StreamedResponse (AD11)
  metadata.py        # + ANSWER_METADATA_KEY, ANSWER_SCHEMA_VERSION, AnswerMetadata
adapters/
  brave_answers.py   # the one provider; SSE + tag parser + spend reader
answer.py            # the call layer: ports consulted, dispositions completed, one answer
bind_answer.py       # prose + the metadata projection (the key's ONE construction site)
standalone.py        # + StreamingTransport implementation
testing/             # + AnswerConformanceSuite, the recorded Brave stream
```

Layer names stay module vocabulary (`search-spec.md` §2): no type is called
`Adapter`, `Call` or `Bind`.

### 10.1 `contracts/answer.py`

- `AnswerRequest(ContractModel)`: `question`, `criteria`, `budget_scope_tags`.
  Canonical form registered beside `SearchRequest`'s.
- `Anchor(ContractModel)`: `start: int`, `end: int`, `source_index: int`.
  Units fixed after §12's live check; documented on the field.
- `GroundedAnswer(ContractModel)`: per AR-A1.
- `AnswerCapabilities(ContractModel)`: per AR-B2, registered through the same
  module-level registry pattern as `ProviderCapabilities` (one registry,
  keyed by provider name, with a `kind` discriminator so `get_capabilities`
  stays one call).
- `AnswerProvider(Protocol)`: `provider`, `provider_instance`,
  `capabilities`, `async answer(request, *, timeout_seconds) -> GroundedAnswer`.

### 10.2 `adapters/brave_answers.py`

Constructor: `api_key`, `transport: StreamingTransport`, `provider_instance`,
`base_url` (defaulted), `usd_per_search` / `usd_per_million_tokens` (optional,
AR-C3), `timeout_seconds` (default 45; Brave's floor is 30).

Request body: `model="brave"`, `messages=[{"role":"user","content":question}]`,
`stream=true`, `enable_citations=true`, plus pushed-down criteria. The plan
answers every criterion (SR-B2) through the shared `_DispositionPlan`.

Stream handling: SSE events → `delta.content` strings → a tag parser that is a
small state machine over the concatenated content (outside a tag / inside
`<citation>` / inside `<usage>` / inside an unknown tag), tolerant of tags split
across events. Output: `text`, citations, usage, unknown tags. `[DONE]` without
`<usage>` → AR-C1.

Spend: AD10. Errors: AR-D2, AR-E1, AR-E2; HTTP status mapping as the Tavily
adapter's, with Brave's credit-limit refusal mapped to `QuotaExhausted`.

### 10.3 `answer.py`

Mirrors `call.search`: bounded criteria, pacing (`_pace`), budget `check`
before and `record` after, dispositions completed from capabilities, one
`GroundedAnswer` out or one typed failure. No accumulation, no retry of its
own.

### 10.4 `bind_answer.py`

`bind_answer(request, *, provider, timeout_seconds, budget, limiter, egress,
zone) -> RenderedAnswer` (success / content / error / metadata, the
`RenderedSearch` shape). Prose: AR-G2's first line, the text, then a numbered
sources list in citation order using `result_facts`. `project_metadata` and
`project_failure_metadata` own `ANSWER_SCHEMA_VERSION`. The reader checks
`schema_version` before structural validation (D13).

### 10.5 `agent-tools`: `threetears.web_answer`

`WebAnswerTool(TearsTool)`: `question` in, `RenderedAnswer.content` on
`content`, the projection on `metadata[ANSWER_METADATA_KEY]`, a failed
`ToolResult` carrying the failure record. Factory `create_web_answer_tool`
takes `api_key`, `transport`, `budget`, `limiter`, `timeout_seconds`. Registered
in `builtin/__init__.py`'s list; alias per AD13.

### 10.6 Enforcement changes

- `test_one_search_result_shape.py`: generalised to a table of
  `(border key, sanctioned site)`; the answer key gets its own row. The
  exemptions file format is unchanged.
- `test_dependency_alignment.py`: no change (AR-H1 is the check).
- `tests/test_package_boundaries.py` in the search package: pins that no
  module but `standalone.py` imports httpx, now including the SSE path.
- `test_fake_protocol_parity.py`: the suite's `_FakeStreamingTransport`
  declares `# parity-with: threetears.search.contracts.transport.StreamingTransport`.

## 11. Consumer wiring: discodon

Recorded here because this document is the cross-repo record (SR-M3); the
build itself is discodon's.

**11.1 `fact_check` on Answers.** The `mode="fact_check"` path stops running
the inner-agent loop. It composes the claim and optional context into one
question, calls `bind_answer` through discodon's existing transport seam
(`SearxngTransport`'s sibling for a public upstream; it already has the
guarded route), and renders the verdict as the answer text with its sources
list. No page reads: a fact-check is a verdict the persona hedges on, not
findings that reach the queue, and it is on the turn loop. Spend: the
delivery's `cost_usd` adds `Spend.money`; `metered_call_spend` reports
`ExternalSpend(provider="brave", calls=1, provider_units=<queries>,
provider_unit="brave:searches", money=<total>)`.

**11.2 `research__ask_web` (experiment, behind a research config flag).** A
second `ResearchSubTool`: the inner agent asks a question; the sub-tool
returns the answer text with its sources rendered, and each source enters
`_corpus_by_url` as a `CorpusEntry` seeded from title+snippet, then goes
through the existing `_read_pages` so grounding has page text (AR-G1). The
hypothesis is 6.4's: a synthesised lead lets the model submit a round earlier.
Measure `stop_cause`, `synthesis_path`, elapsed and spend against the anchor
before it leaves the flag.

**11.3 `MeteredUnit.units_per_call`** becomes per-call: the sub-tool reports
the units each call actually consumed (from `Spend.provider_units`) rather
than a constant. Small, and it fixes an SR-E4-class under-count that would
otherwise be invisible with one provider and arrive with the second.

**11.4 The failure mapping is reused.** `search_fault` already maps
`SearchFailure` subclasses to `ProviderFaultKind`; AD9 is what makes this a
no-op.

**11.5 Spend visibility.** The budget section's rate table gains
`brave:searches`; `ExternalSpend.money` carries the provider's own total, so
eval cost rows and the budget cap see Brave money for the first time (AC2).

## 12. Sequencing, and what to verify live first

Slices, each a PR into `develop`, stacked:

0. **Live capture (prerequisite, half a day).** With discodon's provisioned
   key: one single-mode streamed answer with citations, saved as the recorded
   fixture (AR-I2). Settles the three **(verify live)** items: offset units,
   pre/post-strip indexing, and whether `snippet` is page text. The `Anchor`
   field and AR-A4 are finalised from this, not before.
1. **Contracts + `StreamingTransport` + standalone SSE** (AD11, AR-F1..F3),
   with the conformance suite's transport pins. MINOR bump.
2. **`BraveAnswersAdapter` + `answer.py` + `bind_answer.py`** against the
   fixture; enforcement rows (10.6). MINOR bump.
3. **`agent-tools` `web_answer`** (10.5, AD13).
4. **discodon `fact_check`** (11.1) -- the first production use; AC1 and AC2
   measured there.
5. **discodon `research__ask_web`** behind its flag (11.2, 11.3) -- an
   experiment with a stated hypothesis and the anchor to measure against.

Gates: slice 2 does not merge without the fixture from slice 0; slice 5 does
not leave its flag without the measurement in 11.2.

## 13. The amendment to `search-requirements.md` §2

Applied on this branch, verbatim:

> **Not an answer engine.** We retrieve information and say where it came
> from. Deciding what it *means* stays with the consumer -- discodon's research
> tool synthesises, samsung's model proposes works. We do not summarise, judge,
> or conclude. *Amended 2026-10-22 (owner ruling; `answers-requirements-design.md`):*
> this capability still produces no synthesis, but it **may carry a
> provider's**, when the provider delivers it with citations -- the text as
> given, the citations as candidates, the spend as spend, and the rendering
> saying whose synthesis it is. Carrying is not concluding.

`search-spec.md` §5 ("summarise, judge, or conclude") takes the same rider
when the build plan for slice 1 is cut.

## 14. Decisions needing an owner

Beyond AD1 and AD2, already taken:

1. **AD13's alias name** (`answers`) -- or no alias, by-name only.
2. **AD6** -- confirm research mode stays out of v1 rather than shipping as
   "declared unsatisfied".
3. **Shape 4's trigger** -- is "page-read failures or `cited_sources_snippet_only`
   rising on prod" the right condition to revisit LLM Context, or is there a
   cost condition too (Brave Web Search spend through SearXNG is currently
   invisible to discodon, 6.6).
4. **11.1's no-page-reads stance for `fact_check`** -- a verdict grounded on the
   provider's citations alone. The alternative, reading the cited pages
   inline, puts the latency back on the turn.

## 15. Requirements confidence

**Medium-high on the shape, medium on the provider.** The contract decisions
(AD7-AD11) rest on code read in both repositories the same day and on seams
that already exist. The provider facts (§4.2) are from documentation, not a
live stream: §12 slice 0 exists because three of them cannot be settled any
other way, and nothing in §9 that depends on them is final until it runs.
The consumer hypotheses (AC1, 11.2) are stated as measurements, not
expectations.
