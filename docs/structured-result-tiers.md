# Structured results: let the client say what it wants

**Status:** proposal, 2026-08-18. Nothing built.
**Input wanted from:** metallm, scriob, the chat-kit workstream (§4.11), and
discodon — whose research flow produced §2.1 and half of §4.
**Background:** the channel that carries a tool's structure to a client is
[`stream-protocol-structured-results.md`](stream-protocol-structured-results.md),
built in [#355](https://github.com/pacepace/3tears/pull/355), with consumer
halves in [scriob#180](https://github.com/pacepace/scriob/pull/180) and
[metallm#287](https://github.com/pacepace/metallm/pull/287).

## Summary

Structure reaches clients now. When it doesn't fit the frame we drop all of it
and say so — because nobody told us what mattered, and choosing fields ourselves
would invent a second result shape.

So let the client say. It declares a **tier** once per connection: `citations`
or `full`, with an optional `max_body_chars` on `full`. Two tiers is enough
because content bodies are the only unbounded thing in a projection. And when
something is still too large, we hand back a **handle** rather than dropping it
— by that point we've already paid for the content, so throwing it away refunds
nothing.

## 1. The problem

A tool's typed result — candidates, provenance, scores, dispositions — rides its
completion event, and a frontend can draw a citation card instead of regexing
the model's prose. That works until the result gets big. Measured against real
projections:

| Result | JSON size |
|---|---|
| 10 search results, metadata only | 6,670 chars |
| 20 search results, metadata only | 13,110 chars |
| 50 search results, metadata only | 32,430 chars |
| 1 result carrying 20 KB of extracted page text | 21,264 chars |
| 1 result carrying 100 KB of extracted page text | 103,184 chars |

The sizes are bimodal. A `web_search` turn runs a few hundred bytes per
candidate and fits fine; anything carrying extracted page text — `web_fetch`,
`page_finder`, Tavily configured for raw content — is one or two orders of
magnitude larger, and no sane frame budget covers it.

The per-candidate figure is a function, not a constant, and the table above is
its low end.
[`scripts/measure-structured-result-sizes.py`](../scripts/measure-structured-result-sizes.py)
re-measures all of this through the real projection, and a candidate with
every field an adapter fills — facets, provider ids, a publication date, a
provenanced score — runs nearer 950 chars, rising with snippet length alone to
1,594 at an 800-char snippet. Which way the estimate should lean depends on
the adapter; what matters here is that both ends of that range are small, and
stay small, next to one content body.

Today we send an `omitted` record over the bound: the reason, the size, the
bound it missed. Nothing is lost silently, but nothing is delivered either. The
platform drops everything because it's the only safe move available — deciding
for ourselves which fields to keep invents a second result shape, and a narrowed
payload wearing the full projection's key parses while under-reporting, which is
the defect D-S4 already refuses.

Two things make this fixable rather than fundamental:

- **Which mode you're in is knowable before the call.** It's a property of the
  tool and its config: the SearXNG adapter sets `content=None` unconditionally,
  Tavily fills content only when the plan asked (`include_raw_content`), and
  `web_fetch` and Extract carry it by definition.
- **What the client wants is knowable even earlier.** A chat surface wants
  citations on every turn of every conversation. It does not want 100 KB of
  extracted document text in a websocket frame, ever.

## 2. The proposal

The client declares what it wants, once, and we honor it. Two tiers:

| Tier | Carries | Typical size |
|---|---|---|
| `citations` | the projection **minus content bodies** — every candidate, score, facet and disposition, with `content` marked withheld | ~650 chars per candidate |
| `full` | the projection, bodies included | unbounded |

Declared once per connection or subscription — not per tool call, which no
client can predict. A client that declares nothing gets today's behavior.

The tier is defined by one axis, **bodies in or out**, and not by a list of
fields to keep. That distinction earns its keep immediately: samsung's image
search carries its answer in `Candidate.facets` (rights status, pixel
dimensions, direct-file versus containing-page), and a `citations` tier
enumerated as "url, title, snippet, scores, provenance" would silently drop
exactly the payload. Defined subtractively, facets ride in both tiers for free,
and so does every field nobody has added yet.

It also settles how many tiers we need. Content bodies are the only unbounded
thing in the projection — everything else is a few hundred bytes per candidate
and stays that way — so there's no third tier to find.

### An amount, on the tier that has one

`full` takes an optional `max_body_chars`. This is the honest version of "a
middle tier with content for the top few results," which isn't actually bounded
(three pages at 100 KB is still 300 KB) while a body cap is. It's a parameter
rather than a tier because it's the same shape in a different amount.

It stays honest for the same reason the tiers do: the client named the budget,
so we're not guessing what to drop. Each body says what happened to it:

```json
"content": {"text": "Counts rose 12% across …", "truncated": true,
            "size_chars": 102400, "delivered_chars": 4000}
```

Practically, this is what keeps handles rare — a preview renders inline, and the
handle only gets fetched when a reader expands one.

### On the wire

A chat client that declared `citations`, after a `web_fetch` that pulled a
100 KB page:

```json
{
  "type": "tool_call_end",
  "tool_name": "threetears.web_fetch",
  "structured_kind": "inline",
  "structured_tier": "citations",
  "structured": {
    "search_results": {
      "schema_version": 1,
      "query": "otter population survey 2026",
      "candidates": [
        {
          "identity": "https://example.test/survey",
          "title": "2026 otter population survey",
          "snippet": "Counts rose 12% across …",
          "scores": [{"name": "relevance", "value": 0.82, "source": "tavily"}],
          "content": {"withheld": true, "size_chars": 102400, "mime_type": "text/plain"}
        }
      ]
    }
  }
}
```

`content` is present and marked, never simply absent — a reader has to be able
to tell "nothing was extracted" from "it was extracted and withheld." That
distinction is the whole reason this is a tier and not a trim.

Same turn, a client that declared `full`, where the body didn't fit:

```json
{
  "structured_kind": "handle",
  "structured_tier": "full",
  "structured": {
    "handle": "ctx:9f2c…",
    "summary": "1 result, 100 KB extracted",
    "citations": { "…the citations tier, inline, as above…" }
  }
}
```

It gets the citations tier immediately — always renderable, never a spinner
waiting on a fetch — plus a handle for the part that didn't fit.

### Three rules

1. **The payload MUST name its own tier.** A narrowing that doesn't say it
   narrowed is the defect, not the fix.
2. **The `citations` tier MUST be complete in itself.** Every path delivers
   something renderable; no path delivers nothing.
3. **An unknown tier or kind MUST be skipped, never rejected.** The vocabulary
   is open on purpose, so a client meeting a third tier degrades instead of
   failing.

### 2.1 Where a tier applies, and where it must not

**Only at the client boundary.** A tier says what to put on a wire to a
renderer. It says nothing about what a tool produces, and an in-process consumer
always gets everything.

That's a rule rather than an obvious truth because the failure is silent.
discodon's research tool runs an inner agent whose findings are checked by a
grounding gate, matching every finding name and field value against the
retrieved page text it accumulates per URL. When that inner search moves onto
the 3tears leaf, that corpus *is* `Candidate.content.text`. A `citations` tier
applied at Call or Aggregate — one layer too early — would strip exactly the
text the gate verifies against, and grounding would go on returning answers.
Not a rendering regression; a verification one, failing quietly.

So we read the tier where the frame is built, and nowhere else.

## 3. What this means for clients

**Do nothing and nothing changes.** This is additive, and a client that never
declares a tier keeps today's behavior.

**Declare, and it's one field** at connect or subscribe. In exchange, a
content-bearing tool stops blowing your frame budget and starts rendering.

**Want `full`?** You need a way to resolve a handle — an HTTP GET, the shape
metallm's frontend already runs for media (`GET /api/v1/media/{id}/url`). A
client without that reach should declare `citations`, where the inline bound
becomes a ceiling rather than a fallback.

**Broadcast rooms need one decision.** The declaration is per connection, but
the frame is built once per turn and fanned out to every viewer of the room. Two
viewers who connected with different declarations are asking one payload to be
two things — not because they searched differently (nobody searched; a turn
produced a result and the room watched) but because they declared at different
times, in different clients.

It's latent rather than live: scriob and metallm each have one frontend today,
which would declare one value product-wide, so every viewer agrees by
construction. It stops being latent the first time a second client exists — a
phone beside a desktop, or an embedded read-only view.

Proposal, and it's cheap to settle now: **on a broadcast channel the tier
belongs to the room, not the viewer.** The frame carries `citations` plus a
handle, and a viewer wanting more resolves it. Per-viewer appetite gets served
by the handle instead of by per-viewer frames — which is one of the better
arguments for having a handle at all.

### 3.1 What the bound is measured in, and what it cannot exceed

A tier says what goes on the wire. How much fits is a separate question, and
nobody has written the answer down. It has a hard part.

**The bound and the wire are in different units.** `structure_for_stream`
bounds the artifact's own JSON encoding — 16,384 chars, which #355 calls a
placeholder metallm owns. What gets *published* is that JSON nested in an
event, nested in a `Frame`, nested in a `RoomFrame`, each level escaping the
quotes of the one below. Measured through the real types
([`scripts/measure-structured-result-sizes.py`](../scripts/measure-structured-result-sizes.py)),
the nesting costs 1.20× on a body-heavy payload and 1.34× on a
metadata-heavy one. So a bound expressed in artifact characters is about a
third smaller than the frame it produces.

**And there's a ceiling above it that isn't a budget.** On a shared room the
frame crosses NATS: `RoomFanout.broadcast` publishes it and every pod fans out
on receive. nats-py refuses an oversized publish client-side, before anything
leaves the process, against the server's advertised `max_payload` — 1 MB on a
broker nobody has tuned, the same figure `threetears.nats.pipe` already sizes
its chunks against. Working back through the nesting, that's about **780,000
artifact characters**.

| Payload | Artifact chars | Published bytes |
|---|---|---|
| 20 results, metadata only | 19,150 | 25,221 |
| 1 result, 100 KB extracted text | 105,697 | 127,052 |
| 8 results, 100 KB each | 843,616 | 1,010,599 |
| 20 results, 100 KB each | 2,108,650 | 2,525,281 — refused |

`citations` never comes near it: a 50-result projection is 47,500 chars, so
the ceiling sits about 16× above the widest citation set anyone renders, and
48× above the inline bound #355 ships with. `full` does come near it. Eight
results carrying 100 KB each publishes at 1,010,599 bytes — under the limit,
with less than 4% to spare — and §4's own research corpus is "half a megabyte
to two megabytes," the top of which is refused outright.

**The failure is quiet, and it is quiet on both consumers.** scriob delivers
to the author's own socket first and fans out to the rest of the room after —
that fanout is best-effort, and a failure there is caught and logged rather
than raised, deliberately, so it can never be mistaken for the author's own
connection dying. metallm's cross-worker fanout falls back to local-only
delivery when a publish fails, with a warning. Either way: the person who ran
the search sees their citations, everyone else watching sees nothing, and the
only trace is a log line. Single-pod tests pass.

So the inline bound has an upper limit that is not the client's to declare,
and "raise it" stops working before `full` gets interesting. The answer for
anything past it is the handle §4 already proposes — which is a second
argument for the room-level tier above. On a broadcast channel, `citations`
plus a handle is the only shape that always publishes.

## 4. When it's too large anyway

By the time a projection is too big, the expensive part is already bought. The
provider call was made, the bytes were fetched, and on the scrape path an LLM
extraction already ran. Dropping the result refunds nothing, and the store that
would keep it costs a rounding error against what we spent producing it.

It's only waste if the client wanted it, which is exactly what the declaration
tells us:

- A `citations` client never wanted the body. Not sending it isn't waste — the
  money was spent for the model's benefit, not the client's.
- A `full` client did want it. For that client, `omitted` is the wrong default:
  we'd be throwing away something bought, in hand, and asked for.

**So for a client that asked for `full`, over the bound MUST be a handle, never
an omission.** Storing what we already have turns waste into a cache hit.

There are two reasons to keep what we bought, and only one of them is about
size. The other is *expensive to produce, cheap to keep*, and it stands on its
own. discodon's research tool is the clean case: it buys full page text for up
to eight results across three searches, accumulates half a megabyte to two
megabytes of corpus, grounds against it once, and drops it. Nothing was ever too
large for a frame, because none of it was ever going to a frame — and a
follow-up question on a later turn buys it again. A handle is worth having for
that alone; it's what lets a later turn re-check a claim against the pages the
answer was grounded in, rather than re-searching and hoping the corpus comes
back the same.

The step is shorter than it looks. The offload seam
(`threetears.langgraph.offload`) already moves a tool result over 8,192 chars
out of band into the three-tier store for the model's context window — so on
exactly these turns, the content is usually already stored. What isn't stored is
the structured projection, and storing that is an increment on a path that
exists. The real gap is the client-facing resolve surface: the `[ctx:<id>]`
handle has never crossed to a frontend.

### Do we need pagination?

Not on the stream. Possibly on the resolve endpoint, later.

There are two different "too big," and only one is a paging problem. **Many
candidates** fits offset/limit naturally — show twenty, fetch more as the user
scrolls. **One huge body** doesn't page usefully into frames; the client wants
it when the user expands the citation, and not before.

Paging the *stream* is the option to refuse. It puts reassembly state on a
channel that's transient, broadcast, and reconnectable: a dropped frame leaves a
half-built object, a late joiner gets a fragment, a reconnect starts over.
That's a store with extra steps and worse failure modes — and a stream holding
partial state has quietly become the store D14 says it isn't. A handle plus a
request/response fetch gets retries, caching, conditional requests and ranges
for free, from infrastructure we already run.

One design constraint follows, and it's cheap to honor now: **a handle MUST
address a result set, not an opaque blob**, so `?offset=&limit=` can be added
later without moving the wire. "The projection for tool call X" can grow paging;
"blob 9f2c" can't.

## 5. Out of scope: summaries

Summarization is deliberately not part of this proposal, in any form — not one
the platform writes at the delivery boundary, not one `bind` writes into the
projection. It's a possible future enhancement **to search**, and it should be
taken there or not at all.

Three things worth carrying forward to whoever picks it up:

- **The cheap version already ships and costs nothing.** Both providers return a
  per-result snippet and both adapters already map it to `Candidate.snippet`
  (`adapters/tavily.py:1225`, `adapters/searxng.py:1275`). That's most of what a citation card
  renders, and it rides both tiers today.
- **Two query-level summaries exist and get dropped.** Tavily's `answer` is
  never requested (`include_answer` appears nowhere in the plan body), and
  SearXNG's `answers` / `infoboxes` arrive unrequested and unmapped. Both
  adapters refuse them for the same stated reason — a different shape from a
  candidate — and SearXNG's note leaves them "for a layer that has somewhere
  honest to put them."
- **Anything we carry must be provenanced, and never presented as grounded.** A
  provider synthesis and an instant-answer fact are different objects, only one
  provider produces each, and neither can be checked against the corpus the
  projection carries. Absence would mean "this provider doesn't do that," not
  "there was nothing to say."

On cost: Tavily's published credit table prices a search by depth alone (basic
1, advanced 2) and says nothing about `include_answer`, which suggests it's
free. [not verified — that's a documented absence, not a documented price. One
call with the eval key would settle it.]

## 6. Open questions

- **Where is the declaration made** in each product — a subscribe frame, a
  connect parameter, or per turn in the request?
- **Who owns the resolve endpoint, and what is a handle's lifetime** — the turn,
  the conversation, a TTL? One floor is known: discodon's research delivery is
  asynchronous, landing on a later turn than the one that asked, so a
  turn-scoped handle would be dead before its first reader. Conversation-scoped
  is the minimum that works for a consumer we already have.
- **Who owns the size guard, and where does it go?** (§3.1) Nothing on
  either fanout path measures a frame before publishing it, so the ceiling
  is discovered as a caught exception and a log line. `RoomFanout.broadcast`
  is the one place both scriob's rooms and any future consumer pass through.
- **Does anything but chat consume this?** If a non-chat consumer wants
  structure, the tier vocabulary may need a name that isn't about citations.
- **scriob: is a room-level tier acceptable?** (§3) It's the only answer that
  keeps one frame per event.
