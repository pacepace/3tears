# Search: Requirements

**Status:** Draft for ratification — 2026-08-02
**Scope:** the family's search capability, ahead of any contract or package
**Companions:** `family-convergence.md` §4.14 records the *direction*;
`shared_search.md` sketches a *mechanism*. This document states the
*requirements* neither has — what consumers need, and what must stay
decomposable — so the sketch can be judged against something.
**Relates to:** §4.13 (scraping), §4.2 (evals), open questions 13 and 21

---

## 1. Why this document exists

`family-convergence.md` §4.14 proposes a result shape — url, title, snippet,
score, cost — and a stage pipeline. `shared_search.md` refines that into layers,
typed contracts, provider extras and a packaging recommendation. Both may be
right. Neither is checkable, because no document states what the consumers
actually need. §9 uses these requirements to check the sketch; five of its
choices come out contradicted.

Two things drive the shape:

- **The consumer set is wider than the current call sites.** Apps and models both
  call search; queries originate from code, from a tool an LLM invoked, and from
  a person typing. The things searched for are web pages, images, PDFs and
  documents. Criteria run from "top 5" to "≥3840×2160, rights-clear, from an
  institutional source." Requirements derived only from today's text callers
  would bake text-and-shallow into the family's floor.
- **Layer coupling is the failure mode to design against.** The stated anti-goal:
  *you must never have to accept a higher layer's opinions to get a lower layer's
  benefit.* If asking for consumer-supplied rerank criteria means you can no
  longer say "PDFs only," the layers are welded and the capability serves whoever
  the author imagined and nobody else. §4 makes this testable rules rather than
  an aspiration — `shared_search.md` asserts the principle in one line ("Each
  layer is consumable without the ones above it") and then, in three places,
  breaks it.

**Non-goals.** Rerank algorithm selection; heavy scraping (§4.13 owns it);
model-native search *implementation* (SR-B5 covers the boundary only); which repo
migrates when.

## 2. How to read this

Each item is **REQUIRED** (a consumer regresses or breaks without it, evidence
cited), **DECISION** (consumers disagree or nobody has ruled — a recommendation
is given, the owner picks), or **ASSUMPTION** (inferred from code, not stated by
an owner; vetoable — if wrong, the requirement above it probably changes).

Evidence is cited as `repo/path:line`, read 2026-08-02. Where implementations of
the same thing disagree, the disagreement *is* the finding: the requirement was
never stated and each site guessed.

## 3. The consumer space

### 3.1 Four independent axes

Consumers do not divide by provider. They divide along four axes that vary
independently — the first reason the capability must decompose rather than take a
shape fitted to a representative caller.

| Axis | Values seen or anticipated |
|------|---------------------------|
| **Caller** | program · LLM via a tool call · person typing a query |
| **Target media** | web page · image · PDF / document · (video, dataset — anticipated) |
| **Criteria depth** | none ("top 5") · shallow (recency, domain) · deep (resolution, rights, provenance class, publication type) |
| **Binding** | text for a model · typed domain object · a persisted corpus |

Every cell is reachable. A person typing a query in a web UI wanting rights-clear
4K images bound to domain objects is one cell; an LLM issuing an unscoped web
query and reading prose is another.

### 3.2 Today's call sites, plotted

| # | Consumer | Caller | Media | Criteria | Binding |
|---|----------|--------|-------|----------|---------|
| C1 | discodon persona `web_search` | LLM | web | none | text |
| C2 | discodon research sub-tool | LLM | web | shallow (recency, domains) | text **and** corpus |
| C3 | metallm agent builtin | LLM | web | none | text |
| C4 | metallm admin price lookup | program | web | none | typed |
| C5 | samsung discovery phase 1 | program (model-mediated) | web | shallow | typed |
| C6 | samsung discovery phase 2 | program | **image** | **deep** | typed |
| C7 | `3tears-scrape` `page_finder` | LLM agent (in-family) | web | shallow | typed (a chosen URL) |

Evidence: C1 `discodon/tools/web_search_tool.py`; C2
`discodon/tools/research/web_search.py`; C3
`3tears/packages/agent/tools/.../builtin/web_search.py`; C4
`metallm/api/src/api/v1/admin/models.py:948`; C5
`samsung/curation/src/curation/discovery/phase_one.py`; C7
`3tears/packages/scrape/src/threetears/scrape/page_finder.py:32,237-241`.

C7 matters disproportionately: it is a *shared package* consuming another shared
package's builtin, so whatever the builtin's output forecloses, scrape inherits —
today, flattened text.

C6 is the one **not yet built**, and is the most informative row for that reason.
Its run lifecycle exists — approve-before-phase-2 gating, an `unresolved` bucket,
a finish path (`services/discovery.py:62,168-219`) — and its result record is
fully designed (`persistence/discovery_records.py:236-282`). Only the search
behind it is missing. Phase 1 states the split: it asks for works "and never for
images, which is phase 2's job and a different search entirely"
(`phase_one.py:3-5`). A capability shaped around today's text callers forecloses
the one consumer standing at the door with its requirements already written down.

**ASSUMPTION A1.** C6 is the same capability rather than a permanently separate
path. §4.12 places tiled-image *acquisition* in its own package; this assumes
*finding* an image is search and *fetching* it is acquisition. If the owner
disagrees, §5.C can be dropped — but then this is web-text search and should be
named that.

**ASSUMPTION A2.** Person-typed queries are in scope. No current call site is
one; inferred from the family having web UIs. Cheap to honor now (mostly "queries
are untrusted user content" — SR-K2), expensive to retrofit.

### 3.3 What the plot already shows

C2 wants text *and* structure at once: prose to its inner agent, plus a typed
per-URL corpus accumulated on the side (`research/web_search.py:301-321`) that
the grounding gate and relevance cull later read (`:346-356`). C6 wants deep
criteria from three unrelated families at once — technical (resolution), legal
(rights), provenance (source class) — and samsung has recorded that these
**conflict and must not be collapsed**: `confidence` and `quality_score` are
separate fields because "a museum's own page is maximum confidence and may be
lower resolution than a gigapixel scan elsewhere. Collapsing them into one number
makes the trade invisible and the choice unexplainable"
(`discovery_records.py:240-244`).

No single opinionated result shape serves C1 and C6. That is the requirement.

## 4. Layers, and the rules against coupling them

### 4.1 The layers

Named, not numbered, because `shared_search.md` already uses L0–L3 for a
different cut and two numberings in one docs directory is a coherence trap. The
mapping is given so the documents can be read together.

| Layer | Turns | Owns | `shared_search.md` |
|-------|-------|------|--------------------|
| **Adapter** | a family request → one provider's API | transport, auth, provider-native params and quirks, that call's spend record | L1 providers |
| **Call** | one query → one result set | request/result shape, capability declaration, criteria negotiation, failure classes | L0 contracts |
| **Aggregate** | many calls → one candidate set | dedup key, merge, fusion across queries/providers, fan-out accounting | L2 pipeline (part) |
| **Enrich** | a candidate → its content and metadata | media-appropriate extraction, content provenance, probing | L1b fetch cascade |
| **Select** | candidates + criteria → an ordered/filtered subset | filtering, reranking, scoring, cull | L2 pipeline (rerank slot) |
| **Bind** | candidates → what the caller consumes | typed domain objects, or prose for a model | L3 presentation |

Cross-cutting, attaching where the fact arises rather than at a chosen layer:
**spend**, **budget**, **telemetry**, **concurrency and rate control**,
**record/replay**.

Consumers use different subsets. C4 uses Adapter–Aggregate and binds itself. C2
uses everything and both bindings. C1 uses Adapter–Call–Bind. C6 needs all six
with the deepest Select. C7 reaches them through an agent loop. A design that
only works end-to-end serves one of them.

### 4.2 The non-coupling rules

Requirements, stated so they can be tested rather than admired.

**NC1 — Downward independence (REQUIRED).** Every layer is usable without any
layer above it, at full fidelity. A consumer wanting one provider call and its own
everything-else must be able to have exactly that.

**NC2 — No upward vocabulary (REQUIRED).** A layer never requires a type defined
above it. Call must not know what a corpus is; Adapter must not know what a rerank
criterion is.

**NC3 — Feature orthogonality (REQUIRED).** No capability is conditioned on an
unrelated capability. The acceptance test, from the stated anti-goal: *a consumer
supplying its own rerank criteria (Select) must still be able to constrain media
type (Call).* Any pairing that cannot compose is a defect needing written
justification, not a limitation. This is also the rule a fused "pipeline" helper
tends to break: bundling dedup + rerank + fetch means a consumer wanting dedup
without rerank takes all three.

**NC4 — Open criteria vocabularies (REQUIRED).** No layer closes the set of
criteria a consumer can express. Each understands the subset it can act on and
passes the rest through intact. A closed enum in the criteria path makes every
future consumer's novel criterion a library change — C6's `rights_status` would
have been exactly that a year ago.

**NC5 — No lossy upward projection (REQUIRED).** A fact knowable only at a lower
layer is preserved upward *in usable form*, even where that layer has no use for
it. Samsung records the general case: `acquisition_method` rides the image record
"because it is knowable only where the instance was found: the search reached it
through a provider that either offers tiles, a direct file, or an API, and nothing
downstream can recover which. Without it, promoting an instance into a source
would have to guess" (`discovery_records.py:258-263`). This is NC2's mirror, and
harder to notice: nothing fails at the seam, it fails three layers up, later. An
untyped `raw` passthrough declared "never load-bearing" does not satisfy this.

**NC6 — No collapsed scores (REQUIRED).** Where several ranking dimensions exist
they stay separate and the consumer combines them. Derived independently twice:
samsung for images (`discovery_records.py:240-244`), and the same holds for web
results where provider relevance, a reranker's score and a recency preference are
three different judgments.

**NC7 — Cross-cutting concerns attach at their source (REQUIRED).** Taking cost
tracking, telemetry, budget or replay must not require taking a particular layer.
Spend arises at Adapter and must be observable by a consumer that stops at Call.

**NC8 — Degradation is explicit (REQUIRED).** When an optional layer is absent or
a criterion cannot be honored, the caller is told: unranked results are *known* to
be unranked; an unsatisfiable criterion is named. Silence turns a missing stage
into a wrong answer.

## 5. Requirements

### A. Results

**SR-A1 (REQUIRED, Call/Bind).** Structured results are the primitive; rendering
for a model is one binding. C4 exists as a hand-rolled side-step precisely because
the shared builtin returns only formatted text (`builtin/web_search.py:27-44`,
`admin/models.py:948`), and C7 inherits the same flattening.

**SR-A2 (REQUIRED, Call↔Enrich).** A result must be able to carry retrieved
content, recording whether it came from the search response or a later fetch.
Tavily returns page text *in the search response* at no extra credit versus
`advanced` (`research/web_search.py:206`). A shape that puts content only in a
fetch result forces a Tavily consumer to re-fetch what it already paid for.
Enrich must therefore be a **no-op when the provider already supplied content**,
not a mandatory second call.

**SR-A3 (REQUIRED, NC5).** Results carry provenance: query, provider, the
provider's own identifiers, and retrieval time. C2's grounding gate answers a
per-result question — "does this claim appear on the page it was cited from"
(`research/web_search.py:105-110`) — that no aggregate can answer.

**SR-A4 (DECISION, Call/Select, NC6).** How many score dimensions, and whose?
Tavily returns relevance ∈ [0,1] and C2's cull ranks on it
(`research/web_search.py:27-44`); SearXNG returns an engine-fusion weight on
another scale (*unverified — general knowledge; confirm against a live instance*);
C6 needs at least three orthogonal judgments.
*Recommendation:* a set of named, provenanced scores — never one `score` field.
Provider scores marked non-comparable across providers; a comparable relevance
exists only if Select produced it.

**SR-A5 (DECISION, Aggregate).** Result set or corpus?
C2 and C4 independently built the same accumulation — C2 keyed by URL,
concatenating across searches, keeping the best score seen
(`research/web_search.py:104-111`, `:314-321`); C4 dedups URLs across concurrent
queries preserving order (`admin/models.py:967-975`).
*Recommendation:* Call returns a result set; the corpus is Aggregate's named type
with a stated dedup key and merge rule. Two consumers is the family's own
threshold.

### B. Requests and criteria

This answers "must search own deep criteria, or does the consumer?" Neither
exclusively — and the mechanism is what keeps NC3 true.

**SR-B1 (REQUIRED).** Criteria are stated **once**, by the consumer, in one open
vocabulary (NC4), regardless of which layer satisfies them. A consumer must not
have to know that `time_range` is a provider parameter while `min_resolution` is
a local filter — an implementation fact that changes per provider and per year.

**SR-B2 (REQUIRED).** Each criterion is satisfied by **pushdown** (provider
filters), **local application** (a layer filters or ranks), or **not at all** —
and the caller is told which, per criterion (NC8). Pushdown is an optimization:
cheaper, fewer discarded results, sometimes better recall. Never a precondition
for expressing the criterion.

**SR-B3 (REQUIRED).** An unsatisfiable criterion is reported, never silently
dropped. Precedent: RES-T4M9, a Tavily 400 from sending `time_range` with
absolute dates, fixed by stating absolute-wins precedence rather than silently
suppressing either (`research/web_search.py:218-234`). Discodon's project rules
forbid silent-default shims outright.

**SR-B4 (REQUIRED, Adapter/Call).** Provider capability differences are
declarable and queryable, so a consumer branches before sending rather than after
failing — the pattern `3tears-models` already uses
(`packages/models/.../capabilities.py`). SearXNG has no `search_depth` and no
domain allow-list; it has `categories`, `engines`, `language`, `safesearch`,
`pageno`. Tavily has depth, domains, topic, dates (`research/web_search.py:54-103`,
`:196-234`).

**SR-B5 (DECISION — open question 21).** Is model-mediated search inside this?
C5 has no per-result API scores, no provider result list, and folds cost into
token spend it already tracks (`discovery/engine.py:74-88`).
*Recommendation:* out of Adapter/Call; a `3tears-models` capability flag. It
enters at **Aggregate** as a candidate producer — where its output already lands —
so a consumer can fuse model-mediated and API results without either pretending to
be the other.

### C. Media polymorphism

**SR-C1 (REQUIRED, NC4).** The result core is media-agnostic — identity,
provenance, scores, content availability. Media facets are **additive and open**,
not a closed union of "web | image | pdf".

**SR-C2 (REQUIRED).** A consumer that does not recognize a facet ignores it
rather than failing. This is what lets a new media type ship without a
coordinated release across every consumer.

**SR-C3 (REQUIRED).** Known facet needs, from real records — image: dimensions,
rights status, direct-file versus containing-page URL, and how the bytes are
fetchable (`discovery_records.py:274-281`, `acquisition_method` load-bearing per
NC5). Document/PDF: at minimum extraction status, since "content" for a PDF is a
different operation from HTML extraction.

**SR-C4 (REQUIRED, Enrich).** Enrichment is media-dispatched, and a consumer must
be able to take search without any enrichment (NC1) — the Pi case and C4's case
are the same case.

### D. Budget controls

**SR-D1 (REQUIRED).** Budgets are expressible in **calls**, not only money.
Samsung states why: "A monthly credit limit cannot bound a single run that has
decided to search forever, and an estimate a run may freely exceed is not an
estimate" (`discovery/engine.py:42-49`). C1, C2 and C5 all bound in calls.

**SR-D2 (REQUIRED).** Budget scopes are plural and not interchangeable:
per-persona-per-day (`web_search_tool.py:148-152`), per-invocation
(`research/web_search.py:186-192`), per-run (`engine.py:52`).

**SR-D3 (REQUIRED).** Provider quota exhaustion is distinguishable from a local
cap and short-circuits. C2 trips a per-invocation breaker on HTTP 432/433 and logs
at ERROR because "a dead search backend is an outage, not a per-query warning"
(`research/web_search.py:249-265`).

**SR-D4 (DECISION — currently contradictory).** Does a failed search consume
budget? One repo answers both ways: C2 increments before the request, so timeouts
consume budget (`research/web_search.py:192`); C1 increments only after
`raise_for_status()` (`web_search_tool.py:271-272`). Neither records a decision.
*Recommendation:* budget follows the bill. But C2's fail-closed behavior is what
currently bounds retries against a degraded provider, so that bound must move
somewhere explicit (SR-G4) in the same change.

**SR-D5 (DECISION).** Local refusal or provider refusal?
Samsung: "An engine reports what it spent; it never decides whether it may… never
a local sum crossing a number, because a local tally that fails open is
indistinguishable from one that works" (`engine.py:14-18`). Discodon does the
opposite. Two reasoned positions in conflict, not an oversight.
*Recommendation:* both, distinct roles — local caps bound a run's *shape*
(overrun is a defect); the provider's refusal bounds *money*. Needs a ruling.

**SR-D6 (REQUIRED).** A zero-cost provider still needs bounding. SearXNG's failure
mode is upstream rate-limiting or a ban, not spend; a mechanism keyed only on cost
never fires for it (SR-H4).

### E. Cost tracking

**SR-E1 (REQUIRED, NC7).** Spend is attributable per call, in money, observable
from any layer. Discodon counts and explicitly declines to price: "Paid non-LLM
calls this run made (web search). Counted, never priced"
(`research_tool.py:2503`), with its eval surface warning that "max_cost_usd does
not bound external search quota" (`web/mcp/eval/runs.py:190`). Closing this is the
concrete payoff — an eval cost cap that finally includes search.

**SR-E2 (REQUIRED).** The count a cap enforces and the count a bill prices are one
number. Samsung derives `searches_used` from the priced records because "a
`searches_used` field beside a priced spend record is two tallies of the same
event, free to disagree, and the disagreement would surface as a cap that held
while the bill said otherwise" (`engine.py:106-115`).

**SR-E3 (REQUIRED).** Spend survives the failure path — "a run that broke halfway
still incurred whatever it incurred, and a failure path that dropped it would
under-report the month by exactly the amount the failures cost"
(`engine.py:118-127`).

**SR-E4 (REQUIRED — live defect).** Weighted units must be accounted. Discodon's
persona tool bills every search as one unit (`_check_budget` at its `cost=1`
default, `web_search_tool.py:224`; `tools/base.py:1412`) while
`search_depth="advanced"` spends two Tavily credits — and its docstring says the
budget exists "to manage shared API credits" (`web_search_tool.py:7,57`). The
weighted primitive exists and is used by `youtube_tool.py:216`.

**SR-E5 (REQUIRED).** Cost granularity is per-request for some providers; the
model must not imply per-result pricing. Samsung: "The fee is charged per search
*request*, not per result: one, three, five and ten results all bill identically.
A deployment that lowers the result count therefore saves nothing and sees less"
(`phase_one.py:17-21`) — a knob that looks like a cost control and is not.

**SR-E6 (DECISION).** Self-hosted cost: zero, or amortized infrastructure?
*Recommendation:* zero for spend attribution, with SR-D6's rate/quota dimension
carrying the real constraint. A synthetic per-query infrastructure cost corrupts
cross-provider comparison in the other direction.

### F. Evals and reproducibility

**SR-F1 (REQUIRED).** Search parameters participate in eval identity so a score
delta is attributable to a config change — already true and load-bearing
(`eval/identity.py:221`, via `canonical_digest`). This requires that whatever
parameter object exists be **canonically serializable**.

**SR-F2 (REQUIRED).** Eval runs against a quota separable from production's, with
sharing explicit rather than a fallback. Discodon designed exactly this (EVL-TQ7K,
`discodon/config/sections/tavily.py`): an optional eval-scoped key so "an eval
search burst can never exhaust the quota or trip the breaker protecting live
personas' research," with unset meaning a *documented shared* quota.

**SR-F3 (DECISION — the largest gap found).** Must a search be replayable?
Discodon's cassette layer records and replays at `Tool.act()`
(`eval/cassette_proxy.py:16-18,185-207`). C1 is a `Tool` and is replayable. C2's
sub-tool deliberately is not — "no Tool ABC overhead"
(`research/web_search.py:5`) — so it sits *below* the replay seam and its searches
cannot be replayed. Every research eval re-issues live searches against a changing
web.
*Recommendation:* promote to REQUIRED, and place record/replay as a cross-cutting
concern at Adapter/Call (NC7) rather than at whatever layer happens to be a
`Tool`. That placement is the lesson of the current gap: replay was attached to a
class hierarchy, so a component that opted out of the hierarchy silently opted out
of reproducibility. Without it, research eval trend lines confound model changes
with web drift — undermining §4.2, the family's highest-value workstream.

**SR-F4 (REQUIRED).** What is recorded must rebuild the corpus, not merely the
rendered text — else a replayed run cannot re-run its grounding gate (SR-A2,
SR-A3).

### G. Performance

**SR-G1 (REQUIRED).** Timeouts are configurable, not constants. Four
implementations, three values, none operator-tunable: 30s class attribute
(`web_search_tool.py:39`); 15s constructor default never wired to config
(`research/web_search.py:59`, with the construction at
`research_tool.py:1412-1414` passing depth and caps but no timeout); 15s hardcoded
(`builtin/web_search.py:53`); 15s hardcoded (`admin/models.py:951`).

**SR-G2 (REQUIRED).** A search timeout is derivable from the caller's remaining
deadline, not fixed independently of it. C2's caller runs under a per-call LLM
bound and a run-level conclusion deadline explicitly floored at the per-call
timeout (`research_tool.py:647-707`, `:762`) — so a 30s search inside a shorter
remaining budget is expressible today.

**SR-G3 (REQUIRED).** No blocking IO on an async path. The builtin's `execute()`
is `async` but calls a synchronous `httpx.Client` (`builtin/web_search.py:50-56`,
called at `:122`), blocking the event loop for up to 15s; `web_fetch.py` has the
same defect plus a `time.sleep` in its retry loop (`shared_search.md:38-40`).

**SR-G4 (DECISION).** Retries: capability or consumer?
None retry today; C2 tells the model to retry in prose
(`research/web_search.py:280`), spending an LLM round to redo an HTTP call.
*Recommendation:* bounded transport retry at Adapter. Interacts with SR-D4.

**SR-G5 (REQUIRED).** Resource bounds are part of the contract: byte caps,
content-type gates and streamed downloads, not unbounded `resp.text`. An
unbounded read is a memory incident on a `MemoryMax`-capped host
(`shared_search.md:41-42`) and applies with more force to image and PDF media
(§5.C) than to the HTML the current code assumed.

### H. Concurrency and rate control

**SR-H1 (REQUIRED).** Concurrent calls are boundable and tunable without a
restart. C2 dispatches under an `asyncio.Semaphore` rebuilt per run so a hot
update lands (`research_tool.py:83-90`, `:768-771`); C4 fans out with an unbounded
`asyncio.gather` (`admin/models.py:967`).

**SR-H2 (REQUIRED).** Two bound scopes, both real: within one batch, and across
simultaneous runs (research caps at 3 concurrent, `research_tool.py:328,376`).

**SR-H3 (REQUIRED).** One call's failure must not poison its siblings in a
concurrent batch — handled and reasoned in C2 (`research_tool.py:106-114`).

**SR-H4 (DECISION).** Rate limiting: pace, or react to 429s?
All react; none pace. An unbounded fan-out at a shared self-hosted SearXNG (C4) is
the case most likely to get the family's own instance blocked upstream. The
primitive exists — `threetears.core.coordination.token_bucket.TokenBucket`.
*Recommendation:* client-side pacing per provider *instance*, at Adapter — the
shared instance is what is at risk, and no single consumer sees the aggregate load.

### I. Telemetry

**SR-I1 (REQUIRED).** Every call is individually recorded: query, scoping
parameters, result count, duration, error. Discodon states why — so "the facet
that failed is visible per trace, not just in Loki" (`ResearchSearchRecord`,
`discodon/logging/models.py:232-247`).

**SR-I2 (REQUIRED).** Search wall-clock is separable from model wall-clock in any
run that mixes them (`ResearchRoundRecord`, `logging/models.py:250-275`).

**SR-I3 (REQUIRED).** Calls returning nothing because a budget was already spent
are counted separately from calls that did work — `exhausted_calls`, "pure latency
waste with zero coverage gain, the over-search tail this profile exists to
measure" (`logging/models.py:258-261`).

**SR-I4 (DECISION, NC7).** Emit telemetry, or return records?
C2 persists a trace document with a TTL to its own store
(`research_tool.py:351,794-806`); C5 returns spend records and holds no sink
(`engine.py:90-104`).
*Recommendation:* return records, emit nothing. A capability owning a sink forces
every consumer onto it; both existing consumers already have their own, and
`observe` integration then belongs to the host per the zero-dep-core pattern.

### J. Failure semantics

**SR-J1 (REQUIRED).** Failure classes are distinguishable, not merged:
rate-limited, quota-exhausted, auth-failed, timeout, transport error, malformed
response, zero results. C2 distinguishes all seven and gives each a different
instruction (`research/web_search.py:246-292`), including the deliberate split
between "retry" for timeouts and "give up" for transport failures. Errors carry
remediation where the cause is known and fixable — `shared_search.md:104-107`
makes exactly this point for SearXNG's 403-when-`json`-missing, the #1 setup
failure.

**SR-J2 (REQUIRED).** Zero results is a success. All implementations agree; pin it
so no future one disagrees.

**SR-J3 (DECISION, NC2).** Errors as values or exceptions?
C2 returns instructional prose to a model (`research/web_search.py:246-292`); C1
returns `ActionResult(success=False)` (`web_search_tool.py:243-259`); C5 raises,
carrying spend (`engine.py:118-127`); the builtin sniffs a string prefix
(`builtin/web_search.py:123`).
*Recommendation:* typed exceptions carrying spend (SR-E3), with prose at Bind. The
prose is prompt engineering — tuned per persona and per inner agent — and a lower
layer emitting it has taken an opinion from above (NC2).

### K. Security, privacy, and conduct

**SR-K1 (REQUIRED).** Credentials resolve through the consumer's secret handling;
the capability must not read environment variables itself. Discodon passes the key
in explicitly and holds it as `SecretStr` (`web_search_tool.py:186-192`).

**SR-K2 (DECISION).** Are queries sensitive?
A query can carry user-supplied conversational content and is recorded verbatim
today (`logging/models.py:243`); under A2 it is directly user input. metallm ships
a PII sanitization wrapper it is contributing.
*Recommendation:* treat queries as user content — retention governed by the
consumer's policy, the capability required only to make the query available for
redaction rather than to redact on its own (NC2: redaction policy is the
consumer's opinion).

**SR-K3 (REQUIRED).** A self-hosted base URL is an internal endpoint. SSRF-shaped
risks — consumer-supplied base URLs, redirect following during enrichment — must
be ruled on before the capability accepts a URL from anywhere but deployment
config.

**SR-K4 (DECISION).** robots.txt and provider terms — a family stance, or
adapter-side? Currently unaddressed everywhere
(`shared_search.md:176-177`), and the exposure differs by media: image search
touches rights-bearing works, which C6 already models as `rights_status`.
*Recommendation:* a stated family stance with per-adapter enforcement, because a
per-app answer means the first app to get the shared SearXNG banned decides for
everyone.

### L. Packaging and weight

**SR-L1 (REQUIRED, NC1).** A consumer must take Adapter+Call without a
cross-encoder, torch, or a fetch stack. The Pi is the binding constraint; §5's
"offer everything, require a bare minimum" is the stance.

**SR-L2 (REQUIRED, NC8).** Absent optional layers degrade explicitly: a consumer
without rerank gets provider order and *knows* it.

**SR-L3 (REQUIRED, NC2).** Types crossing package boundaries ship as a
dependency-free leaf, per the ratified contracts-leaf pattern.

### M. Lifecycle

**SR-M1 (DECISION).** How do the types version, and what is the compatibility
promise across lockstep releases? Seven consumers in four repos will bind;
discodon carries an open advisory that it exposes an API with no recorded
versioning scheme. *Recommendation:* rule before the first consumer binds.

**SR-M2 (DECISION).** Response caching — where, and is it in scope?
`shared_search.md:173-175` raises it and notes core collections have no TTL
semantics. It interacts with SR-F3: a cache and a replay store are different
things solving adjacent problems, and building one without deciding about the
other tends to produce a cache that is *almost* a replay store.
*Recommendation:* decide replay first (SR-F3); caching after, in its light.

**SR-M3 (DECISION — open question 13).** Ratification home.
*Recommendation:* this file is the cross-repo record; discodon, metallm and
samsung each record acceptance of what binds them. Otherwise the next session in
any of those repos re-derives all of it.

## 6. Decisions needing an owner

| ID | Decision | Recommendation |
|----|----------|----------------|
| SR-A4 | How many score dimensions, whose | Named provenanced scores; never one `score` |
| SR-A5 | Result set vs corpus | Call returns a set; corpus is Aggregate's named type |
| SR-B5 | Model-mediated search in or out (OQ21) | Out of Adapter/Call; enters at Aggregate |
| SR-D4 | Does a failed search consume budget | Follow the bill; move the retry bound in the same change |
| SR-D5 | Local vs provider refusal authority | Both, distinct roles — two recorded positions conflict |
| SR-E6 | Self-hosted cost: zero or amortized | Zero, plus a separate rate/quota dimension |
| SR-F3 | Is per-search replay required | **Yes — promote to REQUIRED**, attached at Adapter/Call |
| SR-G4 | Retries: capability or consumer | Bounded, at Adapter |
| SR-H4 | Rate limiting: pace or react | Pace per provider instance, on core's `TokenBucket` |
| SR-I4 | Emit telemetry or return records | Return records |
| SR-J3 | Errors as values or exceptions | Typed exceptions carrying spend; prose at Bind |
| SR-K2 | Are queries sensitive | User content; capability exposes, consumer redacts |
| SR-K4 | robots.txt / provider terms | A family stance, enforced per adapter |
| SR-M1 | Versioning and compatibility promise | Rule before the first consumer binds |
| SR-M2 | Response caching | Decide replay first, cache in its light |
| SR-M3 | Ratification home (OQ13) | This file, per-repo acceptance |
| A1 | Is image search the same capability | Assumed yes; if no, this is web-text search and should be named so |
| A2 | Are person-typed queries in scope | Assumed yes; cheap now, expensive to retrofit |

## 7. Defects found while gathering this

True today, independent of whether any convergence happens. Items 6–8 are
`shared_search.md`'s findings, kept here so one list is complete.

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

## 8. What this document deliberately does not do

It does not propose a contract, choose fields, name packages, pick providers, or
sequence a migration.

## 9. Reading `shared_search.md` against these requirements

The sketch is right about layering, provider extras, capability metadata,
conformance tests per provider, `ToolResult.metadata` as the non-breaking
migration path, and packaging option A. Five choices in it are contradicted by the
requirements above, all in the same direction — they assume a text-web caller:

1. **`category: str | None = None  # searxng categories; tavily ignores`**
   (`:59`). A silent per-provider drop, written into the contract sketch itself —
   SR-B2/SR-B3/NC8. This is the seed of exactly the coupling the anti-goal names.
2. **A single `score: float | None`** (`:68`). Keeping the provider's score
   instead of discarding it is an improvement over today; collapsing every ranking
   judgment into one number is NC6/SR-A4, and forecloses C6, whose conflicting
   `confidence`/`quality_score` split is already designed.
3. **Content lives in `FetchResult`, not `SearchResult`** (`:64-88`). Tavily
   returns page text with the search response; this shape cannot express that, so
   a Tavily consumer re-fetches what it already bought — SR-A2, and a capability
   regression under convergence principle 4.
4. **`SearchResult` is web-shaped** — url/title/snippet/published_at, no media
   facets (`:64-71`). C6 cannot bind to it — SR-C1/SR-C3.
5. **`raw: dict[str, Any]  # provider passthrough, never load-bearing`** (`:71`).
   Untyped and disclaimed is not preservation: it is where `acquisition_method`
   would land and then be unusable — NC5.

Two structural notes:

6. **The L2 "pipeline" fuses aggregate, select and enrich** — "search → dedupe →
   optional rerank → bounded-concurrency fetch of top-k" (`:130-131`). A consumer
   wanting dedup without rerank, or rerank without fetch, takes all three — NC3.
   Composable stages, not one helper.
7. **Record/replay appears nowhere** — SR-F3, and the one that most affects §4.2.

None of this argues against the sketch's direction. It argues that the contract
should be cut after §6 has answers, not before — and that the answers change five
fields.
