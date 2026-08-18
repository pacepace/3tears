# Structured results: let the client say what it wants

**Status:** proposal, for input — 2026-08-18. Nothing here is built.
**Input wanted from:** metallm, scriob, the chat-kit workstream (§4.11), discodon
(whose research flow is the case that produced §2.1 and half of §4).
**Context:** the channel that carries a tool's structure to a client is
[`stream-protocol-structured-results.md`](stream-protocol-structured-results.md),
built in [#355](https://github.com/pacepace/3tears/pull/355) with consumer
halves in [scriob#180](https://github.com/pacepace/scriob/pull/180) and
[metallm#287](https://github.com/pacepace/metallm/pull/287). This proposal is
about the one case that channel leaves unanswered: what happens when the
structure is too big for the frame.

---

## 1. The problem

Structure now reaches clients. A tool's typed result — candidates, provenance,
scores, dispositions — rides its completion event, and a frontend can draw a
citation card instead of regexing the model's prose.

Until it can't. Real measurements, against real projections:

| Result | JSON size |
|---|---|
| 10 search results, metadata only | 6,670 chars |
| 20 search results, metadata only | 13,110 chars |
| 50 search results, metadata only | 32,430 chars |
| 1 result carrying 20 KB of extracted page text | 21,264 chars |
| 1 result carrying 100 KB of extracted page text | 103,184 chars |

**The distribution is bimodal, not continuous.** A `web_search` turn is a few
hundred bytes per candidate and fits comfortably. Anything carrying extracted
page text — `web_fetch`, `page_finder`, a Tavily call configured for raw
content — is one or two orders of magnitude larger, and no plausible frame
budget accommodates it.

Today, over the bound, the event carries an honest `omitted` record: the reason,
the size, the bound it missed. Nothing is lost silently. But nothing is
delivered either, and **the platform drops everything because nobody told it
what mattered.** Dropping is the only safe move available: a platform that
decides for itself which fields to keep has invented a second result shape, and
a narrowed payload wearing the full projection's key is a payload that *parses
while under-reporting* — the defect the family already refuses (D-S4).

Two facts make this solvable rather than fundamental:

- **Which mode you are in is knowable before the call.** It is a property of the
  tool and its configuration, not a surprise: the SearXNG adapter sets
  `content=None` unconditionally; Tavily fills content only when the plan asked
  (`include_raw_content`); `web_fetch` and Extract carry it by definition.
- **What the client wants is knowable even earlier.** A chat surface wants
  citations on every turn of every conversation. It does not want 100 KB of
  extracted document text in a websocket frame, on any turn, ever.

## 2. What we propose

**The client declares what it wants, once, and the platform honours it.** Not a
byte budget — an intent.

Two tiers to start:

| Tier | Carries | Typical size |
|---|---|---|
| `citations` | the projection **minus content bodies** — every candidate, every score, every facet, every disposition, with `content` marked as withheld | ~650 chars per candidate |
| `full` | the projection, bodies included | unbounded |

Declared once per connection or subscription — not per tool call, which no
client can predict. A client that declares nothing gets today's behaviour
exactly.

**The tier is defined by one axis — bodies in or out — and not by a list of
fields to keep.** That distinction matters more than it looks. A field
allowlist would already be wrong for the second consumer: samsung's image search
carries its answer in `Candidate.facets` (rights status, pixel dimensions,
direct-file versus containing-page), and a `citations` tier enumerated as
"url, title, snippet, scores, provenance" silently drops exactly the payload.
Defined subtractively, facets ride in both tiers for free, and so does every
field nobody has added yet.

It also explains why two tiers is the right number rather than a compromise:
**content bodies are the only unbounded thing in the projection.** Everything
else is a few hundred bytes per candidate and stays that way. There is no third
tier to find, because there is no second unbounded field.

### 2.1 An amount, on the tier that has one

`full` takes an optional `max_body_chars`. This is the answer to "what about a
middle tier that carries content for the top few results" — that idea is not
bounded (three pages at 100 KB is still 300 KB), while a body cap is, and it is
a *parameter* rather than a tier because the client is asking for the same
shape in a different amount.

It is honest for the same reason the tiers are: the client named the budget, so
the platform is not guessing what to drop. And it must say so, per body:

```json
"content": {"text": "Counts rose 12% across …", "truncated": true,
            "size_chars": 102400, "delivered_chars": 4000}
```

Practically, this is what keeps handles rare: a preview renders inline, and the
handle is fetched only when a reader expands one.

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

Note `content` — **present and marked, not absent.** A reader can tell "no
content was extracted" from "content was extracted and withheld". That
distinction is the whole reason this is a tier and not a trim.

Same turn, a client that declared `full`:

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
waiting on a fetch — plus a handle for the part that did not fit.

### 2.2 Where a tier applies — and where it must not

**Only at the client boundary.** A tier is a statement about what to put on a
wire to a renderer. It is never a statement about what a tool produces, and an
**in-process consumer always gets everything.**

This is a rule rather than an obvious truth because the failure it prevents is
silent. discodon's research tool runs an inner agent whose findings are checked
by a grounding gate that matches every finding name and field value against the
retrieved page text — `CorpusEntry.text`, accumulated per URL across the
invocation's searches. When that inner search moves onto the 3tears leaf, that
corpus *is* `Candidate.content.text` in the projection. A `citations` tier
applied at Call or Aggregate — one layer too early — would strip exactly the
text the grounding gate verifies against, and grounding would keep returning
answers. Not a rendering regression: a verification one, failing quietly.

So the tier is read where the frame is built, and nowhere else.

### Three rules that keep it honest

1. **The payload names its own tier.** A narrowing that does not say it narrowed
   is the defect, not the fix.
2. **The citations tier is complete in itself.** Every path delivers something
   renderable; no path delivers nothing.
3. **An unknown tier or kind is skipped, never rejected.** The vocabulary is
   open on purpose — a client meeting a third tier must degrade, not fail.

## 3. Implications for clients

**If you do nothing:** nothing changes. This is additive, and the current
default is preserved for every client that never declares a tier.

**If you declare:** one field, at connect or subscribe. In exchange, a
content-bearing tool stops blowing your frame budget and starts rendering.

**If you want `full`:** you need a way to resolve a handle — an HTTP GET, the
same shape metallm's frontend already runs for media
(`GET /api/v1/media/{id}/url`). A client without that reach should declare
`citations`, where the inline bound becomes a ceiling rather than a fallback.

**Broadcast rooms need one decision, and the mechanism is worth stating
plainly.** The declaration is per connection; the frame is built once per turn
and fanned out to every viewer of the room. So two viewers who connected with
different declarations are asking one payload to be two things — not because
they searched differently (nobody searched; a turn produced a result and the
room watched) but because they declared at different times, in different
clients.

It is latent rather than live: scriob and metallm each have one frontend today,
which would declare one value product-wide, so every viewer agrees by
construction. It stops being latent the first time a second client exists — a
phone alongside a desktop, or an embedded read-only view.

**Proposal, and it is cheap to settle now: on a broadcast channel the tier
belongs to the room, not the viewer.** The frame carries `citations` plus a
handle, and a viewer wanting more resolves it. Per-viewer appetite is served by
the handle rather than by per-viewer frames — which is one of the better
arguments for having a handle at all.

## 4. When it is too large anyway — and the fact that we already bought it

At the moment a projection is too big, **the expensive part is already paid
for.** The provider call was made, the bytes were fetched, and on the scrape
path an LLM extraction already ran. Dropping the result refunds nothing. It is
pure waste — and the store that would keep it costs a rounding error against
what was already spent to produce it.

But it is only waste **if the client wanted it**, and that is exactly what the
declaration tells us. This is the argument for tiers stated in cost terms:

- A `citations` client never wanted the body. Not sending it is not waste; it is
  correctness, and the money was spent for the model's benefit, not the
  client's.
- A `full` client did want it. For that client, `omitted` is the wrong default —
  we are throwing away something bought, in hand, and asked for.

**So: for a client that asked for `full`, over the bound is a handle, never an
omission.** Storing what we already have converts waste into a cache hit.

**And there are two reasons to keep what we bought, not one.** The argument
above is *too big to send*. The other is *expensive to produce, cheap to keep* —
and it stands on its own, without a frame anywhere in sight. discodon's research
tool is the clean case: it buys full page text for up to eight results across
three searches (Tavily `include_raw_content="text"`), accumulates roughly half a
megabyte to two megabytes of corpus, uses it once for a grounding pass, and
drops it. Nothing was ever too large for a frame, because none of it was ever
going to a frame — and a follow-up question on a later turn buys it again.

A handle is worth having for that alone. It is what lets a later turn re-check a
claim against the pages the answer was actually grounded in, rather than
re-searching and hoping the corpus comes back the same.

Worth noting how short that step is: the offload seam
(`threetears.langgraph.offload`) *already* moves a tool result over 8,192 chars
out of band, into the three-tier store, for the model's context window — so on
exactly the turns this proposal is about, the content is usually **already
stored**. What is not stored is the structured projection, and storing it is an
increment on a path that exists rather than new infrastructure. The gap is the
client-facing resolve surface: the `[ctx:<id>]` handle has never crossed to a
frontend.

### Do we need pagination?

**Not on the stream. Possibly on the resolve endpoint, later.**

There are two different "too big", and only one of them is a paging problem:

- **Many candidates.** A 500-result corpus is a natural fit for offset/limit —
  the client shows twenty and fetches more as the user scrolls.
- **One huge body.** A 100 KB document does not usefully page into frames. The
  client wants it when the user expands the citation, and not before.

Paging the *stream* is the option to refuse. It puts reassembly state on a
channel that is transient, broadcast, and reconnectable: a dropped frame leaves
a half-built object, a late joiner gets a fragment, and a reconnect starts over.
That is a store with extra steps and worse failure modes — and a stream that
holds partial state has quietly become the store D14 says it is not.

A handle plus a request/response fetch gets retries, caching, conditional
requests and ranges for free, from infrastructure that already exists.

**One design constraint follows, and it is cheap to honour now:** make the
handle address a *result set*, not an opaque blob, so `?offset=&limit=` can be
added later without moving the wire. A handle that resolves to "the projection
for tool call X" can grow paging. One that resolves to "blob 9f2c" cannot.

## 5. Out of scope: summaries

**Summarization is deliberately not part of this proposal**, in any of its
forms — not a platform-authored one at the delivery boundary, not one `bind`
writes into the projection. It is a possible future enhancement *to search*,
and it should be taken there or not at all.

Three things are worth carrying forward to whoever picks it up:

- **The cheap version already ships and costs nothing.** Both providers return
  a per-result snippet and both adapters already map it to `Candidate.snippet`
  (`tavily.py:1225`, `searxng.py:1275`). That is most of what a citation card
  renders, and it rides both tiers today.
- **Two query-level summaries exist and are dropped.** Tavily's `answer` is
  never requested (`include_answer` appears nowhere in the plan body), and
  SearXNG's `answers` / `infoboxes` arrive unrequested and unmapped. Both
  adapters refuse them for the same stated reason — they are a different shape
  from a candidate — and SearXNG's note ends by leaving them "for a layer that
  has somewhere honest to put them." Tavily's published credit table prices a
  search by depth alone (basic 1, advanced 2) and says nothing about
  `include_answer`, which suggests it is free; that is a documented absence
  rather than a documented price, so it wants one empirical call before anyone
  relies on it.
- **Whatever is carried must be provenanced, and never presented as grounded.**
  A provider synthesis and an instant-answer fact are different objects, only
  one provider produces each, and neither can be checked against the corpus the
  projection carries. Absence of the field would mean "this provider does not do
  that", not "there was nothing to say" — which is precisely the kind of field
  that gets misread.

## 6. Open questions

1. **Where is the declaration made** in each product: a subscribe frame, a
   connect parameter, or per turn in the request?
2. **Who owns the resolve endpoint, and what is a handle's lifetime** — the
   turn, the conversation, a TTL? One floor is already known: discodon's
   research delivery is **asynchronous**, landing on a later turn than the one
   that asked, so a turn-scoped handle would be dead before its first reader.
   Conversation-scoped is the minimum that works for an existing consumer.
3. **Does anything but chat consume this?** If a non-chat consumer wants
   structure, the tier vocabulary may need a name that is not about citations.
4. **scriob: is a room-level tier acceptable?** (§3) It is the only answer that
   keeps one frame per event.
