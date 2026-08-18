# Stream protocol: the channel for structured tool results

**Status:** **APPROVED 2026-08-18**, and the in-repo half is **built** — see
§8. The five decisions below stand as written except **D-S2, which the
consumers' own code reversed**: no new frame type is needed, and adding one
would have broken the mapping scriob's transport already uses. The correction
is recorded in place at D-S2 rather than by editing the recommendation away.
**Still needed from outside this repo:** metallm's inline bound (open question
1), and the handle branch's producer (§8, the one thing D-S3 asks for that
nothing here can supply without becoming a store).
**Why now:** this is
[`convergence-sequencing.md` Phase 2's third item](convergence-sequencing.md),
the design-only one. Phase 2 was declared complete on 2026-08-12 having closed
its search-build items; **this item was never taken.** It is recorded as
outstanding here rather than quietly carried.

**What it gates:** metallm's *frontend* convergence, and the headless chat kit
(`family-convergence.md` §4.11) that is meant to serve every product's chat
surface. It gates nothing in the search sequence, which is why it was safe to
miss — and why it is now the oldest undone thing in the program.

---

## 1. The gap, precisely

Three layers carry a tool result outward. Structure exists at the first, dies at
the second, and has nowhere to live at the third.

| Layer | Type | Carries structure? |
|---|---|---|
| in-process | `ToolMessage.artifact` | **Yes** — the typed projection, since [#318](https://github.com/pacepace/3tears/pull/318) / [#326](https://github.com/pacepace/3tears/pull/326) |
| agent runtime | `ToolCompletedEvent` (`langgraph/events.py:133`) | **No** — `tool_name`, `tool_status`, `tool_duration_ms` |
| wire to client | `Frame` (`channels/frames.py:70`) | **No** — `payload: str \| None`, "opaque body broadcast verbatim" |

**A fourth row, missing until 2026-08-18** (D-S2's correction): the agent-runtime
line is *two* events, not one. `ToolCompletedEvent` is the in-process custom
event metallm builds; `ToolCallEndEvent`
(`packages/langgraph/src/threetears/langgraph/streaming.py`) is the NATS
streaming envelope scriob rides, and it carried the same three fields and the
same nothing. Both are fixed together below.

So a search runs, produces a fully typed candidate set with provenance, scores
and per-criterion dispositions, and a streaming client is told: *a tool named
`threetears.web_search` finished, status completed, 840ms*. Everything that
makes the answer re-checkable (SR-A3) is dropped one layer below the consumer
that would render it.

The consequence is the one `search-spec.md` §4.8 already recorded for MCP before
it was fixed: **a structured payload reaches its consumer as prose it must
re-parse, or not at all.** A frontend wanting to draw citation cards has to
regex the model's rendered text — which is exactly the defect
`page_finder` was rebuilt to stop doing ([#326](https://github.com/pacepace/3tears/pull/326)).

## 2. What already exists, and must not be re-derived

Verified against the code on 2026-08-14.

**`ToolMessage.artifact` already carries it.** `ToolExecutor` invokes with the
whole tool call so LangChain builds the `ToolMessage` and a tool registered
`response_format="content_and_artifact"` keeps its artifact
(`executor.py:115-139`). The structure is present in-process at the moment the
tool completes. **Nothing needs to be re-plumbed to make it available** — the
question is only what to do with it.

**The event layer is additive-safe, and this is the one place that is true.**
`FrameworkEvent` sets no `model_config`, and its docstring states the contract
outright: *"adding a new field to a framework event is a non-breaking change for
older consumers that ignore the new attribute."* Events cross via
`adispatch_custom_event` as `model_dump(mode="json")` dicts (`events.py:321`).

This matters enormously in light of what the family learned on 2026-08-13:
`CallRequest` is `extra="forbid"`, an unset optional still crossed the wire as
an explicit `null`, and a 0.23.11 pod refused **every** call from a 0.24.1
registry for three days. **That hazard does not apply here.** Say so explicitly
so the next reader does not import the fear along with the lesson — and see
§3's rule 5 for the one place it *does* apply.

**`Frame` is forward-compatible too**, deliberately: `extra="ignore"`, *"a newer
client may send fields an older handler does not understand; dropping them
silently keeps the migration additive"*.

**Fat events are already the norm.** `ResponseCompletedEvent` carries
`system_prompt`, `jailbreak_text`, `tools_used`, six token counters and two
invocation ids. Any argument that the event layer must stay thin has to explain
that event first.

**The family has ruled this question twice already, for other faces.** D22:
structure rides a **named key** on `ToolResult.metadata`, following
`OBJECT_HANDLE_METADATA_KEY`, *"built by a named method that owns its schema
version, not dumped at the call site where the shape would drift per caller"*.
MCP: prose in `content`, structure in `structuredContent`, **one message, two
registers** (`mcp/server.py:292-305`).

**A summary-plus-handle idiom already exists, one layer up.** The model's own
context window has the identical size problem D-S3 is solving for the wire, and
the family already shipped an answer: `threetears.langgraph.offload` defines
the contract (`packages/langgraph/src/threetears/langgraph/offload.py`), a tool
result over `DEFAULT_OFFLOAD_THRESHOLD_CHARS` (8192 chars, `offload.py:38`) is
stored whole in the three-tier `ContextItemCollection` via `save_tool_result`
(`packages/agent/tools/src/threetears/agent/tools/context.py:290`), and the
model sees `summary + [ctx:<id>]` instead of the dump. `context_recall`
(`packages/agent/tools/src/threetears/agent/tools/builtin/context_recall.py`)
is the paired tool that pulls the full content back on demand, same turn. This
is a *different* layer than D-S3 — it bounds what reaches the model, not what
reaches the client, and the `[ctx:<id>]` handle never crosses to the frontend
today (verified: no `ctx:` or `context_recall` reference anywhere in metallm's
frontend). But it is the family's own working precedent for rule 3's shape —
small reference in-band, full content one call away — and D-S3 should be read
as choosing the *wire* analogue of a pattern already proven at the *model*
layer, not inventing the pattern itself.

**A client-fetchable handle already has a live implementation, not just a
contract.** `ObjectHandle` (`packages/media-contracts/src/threetears/media/contracts/protocols.py:290`)
is D22's named-key precedent, but it is more than that: metallm already
resolves one to bytes for a real browser client today.
`GET /api/v1/media/{id}/url` (`metallm/api/src/api/v1/media.py:1267`,
`get_media_url`) takes a small id, looks up its `s3_key`, and returns a
presigned S3 URL the frontend fetches directly — no bytes ever cross the
websocket. **This is D-S3(b)'s "endpoint and object-store reach," built and in
production**, for every image and attachment metallm renders. It is not a cost
the design would be introducing; it is a cost already paid by an existing
consumer. See D-S3 below — this changes that option's calculus.

## 3. Rules any answer must obey

1. **One contract, N faces** — success check 14, closed
   [#344](https://github.com/pacepace/3tears/pull/344). The stream is the
   *fourth* face after platform tool, MCP and HTTP API. It renders the same
   candidate set; it does not get a result shape of its own.
2. **One construction site.** `threetears.search.bind` owns the projection, and
   `tests/enforcement/test_one_search_result_shape.py` now enforces it. A
   stream-specific serializer that assembles the payload itself is a guard
   failure, not a design choice.
3. **Re-checkability is the point** (SR-A3). Prose is bounded; structure is what
   turns a truncation back into a citation. A stream that carries only a summary
   has not solved the problem it was asked to solve.
4. **The stream is not a store** (D7, D12, D14). Frames are transient. Nothing
   here may become a cache, and a client that wants to keep what it saw keeps it
   itself.
5. **Where the forbid hazard DOES apply.** Adding a *field* to a
   `FrameworkEvent` or to `Frame` is safe (§2). Adding a new **frame `type`** is
   safe for handlers that dispatch on known types and ignore the rest — but a
   handler with an exhaustive match will reject it. Any new type ships
   receiver-first, and, per 2026-08-13, "receiver-first" is a claim that must be
   **tested against a reader that predates the change**, not asserted in a
   docstring.

## 4. The decisions

Five, each with a recommendation. They are separable; a reviewer may take some
and veto others.

### D-S1 — Which event carries the structure?

| Option | For | Against |
|---|---|---|
| **(a) Extend `ToolCompletedEvent`** with an optional named field | Matches the family's "one message, two registers" (D22, MCP). No correlation problem — the structure arrives with the completion it belongs to. Additive-safe by the base class's own contract | Every consumer of the event now receives the payload, wanted or not |
| (b) A new `ToolStructuredEvent` | Consumers opt in by subscribing | Re-creates a pairing problem the started/completed pair already solved; a consumer must correlate two events by tool name + ordering, and nothing today gives an invocation id to correlate *on* |
| (c) Both | — | Two shapes for one fact, which is precisely what check 14 forbids |

**Recommendation: (a).** The family answered this for two other faces and both
times the answer was one message carrying both registers. Option (b)'s opt-in
appeal is real but it buys the wrong thing: the cost that matters is bytes on a
websocket, and (b) does not reduce them for a consumer that wants the structure
— it only defers them. Bound the payload instead, which is D-S3.

**Note the missing primitive that (b) exposes.** There is no per-invocation id
on `ToolStartedEvent` / `ToolCompletedEvent` — they pair by name and order. That
is fine today and would not survive concurrent tool calls. Out of scope here,
worth recording: if the family ever wants (b), it needs that id first.

### D-S2 — How does structure cross `Frame.payload: str | None`?

The payload is a string by contract and there is no reason to widen it. So the
projection is JSON-encoded into it either way, and the real question is **which
`type` the frame carries**.

| Option | For | Against |
|---|---|---|
| **(a) A new frame type**, e.g. `tool.result`, payload = the JSON projection | A client dispatches on a name instead of sniffing a body. No change to `Frame` at all | New type; ships receiver-first (rule 5) |
| (b) Reuse the existing chat/message frame and put structure inside its payload | No new type | Makes every existing message handler parse a body it did not expect; the "opaque, broadcast verbatim" contract stops being true |

**Recommendation: (a).** It needs no contract change, which makes it the
cheapest correct answer, and it keeps `payload`'s opacity honest — the frame
*type* is what says how to read the body, which is what a discriminator is for.

**Overtaken 2026-08-18 — the answer is (c), "neither", and this decision should
not have been framed as a choice between two frame types.** Both options assume
this document's author gets to choose the frame. Neither consumer lets them,
and they do not even disagree in the same direction:

| Consumer | Runtime event it builds | What reaches the browser |
|---|---|---|
| metallm | `ToolCompletedEvent` (`api/src/services/tool_loop.py:1942`, `converged_tool_loop.py:245`) | its **own** ws dataclass — `ToolInvocationEndMessage`, `type: "tool_invocation_end"` (`api/src/ws/protocol.py:409`), broadcast at `api/src/ws/handler.py:2483`. It does not use `threetears.channels` for chat at all; its only channels import is the webhook receiver |
| scriob | `ToolCallEndEvent` (`server/src/scriob_server/chat/turn.py:519`, from an `awrap_tool_call` middleware holding the `ToolMessage`) | `threetears.channels.Frame` — but minted generically: `WsStreamTransport.publish` forwards each serialized `StreamEvent` as one frame **whose `type` is the event's own discriminator** (`server/src/scriob_server/chat/streaming.py:17`) |

So a `tool.result` frame type would have been a 3tears surface with **no
caller** — metallm mints its own message types and scriob mints frame types
from event discriminators, which means a hand-added type is one scriob would
never produce. Worse, it would have needed the receiver-first rollout of rule
5, and paid it for nothing.

Widening the **event** instead makes the frame follow for free: scriob's
existing `tool_call_end` frame carries the structure inside the payload it
already forwards, no new type, no rollout order, no exhaustive-match hazard.
Rule 5 does not engage at all.

**What this costs elsewhere:** the design's three-layer table (§1) is missing a
row. `ToolCallEndEvent` is not the same object as `ToolCompletedEvent` — it is
the NATS streaming vocabulary in `threetears.langgraph.streaming`, which calls
itself "the single source of truth for the wire envelopes any 3tears app sends
or receives on a streaming token channel", and it is the face scriob rides.
D-S1(a) applied to one event and not the other would have dropped the structure
at precisely the hop it was meant to cross, for the consumer whose wire this
document reasoned about. **Both faces grew the field, in one commit, held
together by a shared mixin and a test that compares them to each other.**

### D-S3 — How much structure, and what happens when it is large?

This is the substantive decision and the one most needing outside input.

A corpus of fifty candidates with content slots is not a websocket frame. But a
summary is not re-checkable (rule 3).

| Option | For | Against |
|---|---|---|
| (a) The full projection, bounded, with an explicit truncation mark | Re-checkable in the common case; one shape; nothing new to build | A large result is still truncated, and the client cannot get the rest |
| (b) Always a handle (`ObjectHandle`-style) the client fetches | Unbounded results work; reuses a family primitive D22 already follows, and — for metallm — an *implementation* it already runs (§2) | A round-trip before anything renders, even for a three-candidate result that would have fit inline |
| **(c) Inline when small, handle when large** | Best of both | Two paths — but for metallm this is not two paths to *build*: the inline path is (a) below the bound, and the handle path is the existing `ObjectHandle` + presigned-URL primitive (§2), which the frontend already implements for media |

**Recommendation, revised: (c) from v1, not deferred to an escalation.** §2 was
added after this section was first drafted, and it changes the answer. The
original reasoning for (a)-only leaned on the `search-task-03` rule against
building a seam whose only caller is a test
([`search-task-03`](search-task-03-producer-seam-sketch.md)) — correct in
general, but it assumed the handle side of (c) was new infrastructure with no
proven caller. It is not: `ObjectHandle` + `GET .../url`-style resolution is
already built, already in production, and metallm is already its caller — for
media today, for a large search projection tomorrow is the same primitive, not
a new one. Deferring (c) behind "let the first consumer that hits the bound
justify the handle" makes sense when the handle is speculative cost. It does
not make sense when the handle is a `git grep` away from already shipping.
Ship (a)'s bounded projection under the frame's normal size, and fall back to a
small `ObjectHandle`-shaped reference (id + summary, no bytes) plus the same
resolve-by-id call metallm's frontend already makes for media, once the
projection exceeds it. The exact bound is still metallm's number to pick (open
question 1) — what changes is that picking it wrong no longer strands a large
result with no way to get the rest.

**The bound must be explicit in the payload, not implicit in the sender.** A
truncated payload that does not say it was truncated is the silent-partial-
answer defect P8 exists to prevent, and `CandidateSet.notices` already exists
for exactly this class of statement.

### D-S4 — If it narrows, does it reuse the projection's key and `schema_version`?

**No, and this is the one hard "must not" in the document.**

The context-save node already writes a deliberately narrowed record under the
full projection's key, carrying `schema_version` copied from the full
projection. Found and documented while closing check 14
([#344](https://github.com/pacepace/3tears/pull/344)): a reader that hands that
record to `SearchResultsMetadata.from_metadata` gets something that **parses
while under-reporting**. Nothing does that today. It is the shape of a defect,
not one yet.

A stream payload that narrows and keeps the key would make the same latent
defect real, on a surface with far more readers. So: **either the payload is the
full projection under its own key, or it is a differently-named thing that no
reader will mistake for one.** Not a narrowed impostor wearing the same
`schema_version`.

### D-S5 — Ordering and pairing

**Recommendation:** the structure rides the completion event (D-S1a), so there
is nothing to order. Stated anyway because if D-S1 is vetoed toward (b), a
structure event MUST NOT precede its `ToolCompletedEvent` — a client that
renders results before it has been told the tool finished will draw a completed
card under a spinner.

## 5. What the recommended shape looks like

Sketch, not a patch. Nothing here is built.

```python
class ToolCompletedEvent(FrameworkEvent):
    type: Literal["tool_completed"] = "tool_completed"
    tool_name: str
    tool_status: str = "completed"
    tool_duration_ms: int = 0
    #: the typed projection the tool put on ``ToolMessage.artifact``, under
    #: the same named key it rides everywhere else (D22). None when the tool
    #: produced no structure -- which is most tools, and stays free.
    structured: dict[str, Any] | None = None
```

On the wire, one frame — **and not a new one; see D-S2's correction.** What
scriob's transport actually sends, given the widened `ToolCallEndEvent`, is the
frame it already sent, with the structure inside the payload it already
forwards:

```json
{
  "type": "tool_call_end",
  "room": "acme:story:chat:{session}",
  "payload": "{\"type\":\"tool_call_end\",\"tool_name\":\"threetears.web_search\",\"structured\":{\"search_results\":{ ... }},\"structured_kind\":\"inline\", ...}"
}
```

Three properties fall out, and they are the reasons to prefer it:

- **The key inside `payload` is the same one the other three faces use.** A
  consumer that already reads `ToolResult.metadata` or MCP `structuredContent`
  reads this with the same code.
- **`structured=None` costs nothing.** Most tools have no structure; they emit
  what they emit today, one JSON null in an already-fat event.
- **`bind` remains the only construction site.** Nothing in the stream path
  assembles a payload; it forwards one.

The sketch above is the inline branch (D-S3 under the bound). Over the bound,
`structured` carries an `ObjectHandle.to_metadata()`-shaped dict (id + summary,
no bytes, §2) instead of the projection — a shape that already exists and needs
no new pydantic model, only a discriminator (e.g. a `structured_kind: Literal["inline", "handle"]`
sibling field) so the client knows which one it got without probing the shape.

## 6. Open questions — these need stakeholders, not this document

1. **What is the practical size bound for the inline path?** Narrower than
   originally framed: with D-S3 now recommending (c) from v1, this number no
   longer decides *whether* a large result is reachable at all (it always is,
   via the handle) — it only sizes where inline stops and the handle starts.
   Still metallm's number to pick.
2. **Do chat-kit's other planned clients (scriob, samsung) have the same
   object-store reach metallm already does?** Answered for metallm: yes —
   `GET /api/v1/media/{id}/url` (§2). Open for the others. Where a future
   client lacks it, D-S3(c)'s handle branch is off the table *for that
   client*, and its inline bound becomes a hard ceiling rather than a
   default — a per-client capability, not a protocol-wide one.
3. **Does any product need structure from a tool other than search?**
   `page_finder` and `web_fetch` already produce it. If the answer is "many",
   the `structured` field wants a per-tool schema story rather than search's one
   named key — which is a bigger design than this one.
4. **Concurrent tool calls.** If any product runs tools concurrently within a
   turn, the missing per-invocation id (§D-S1) stops being theoretical, and it
   should be added before this ships rather than after.
5. **Who owns the chat-kit side?** §4.11 is unstarted. This design is its input,
   and an input with no consumer schedule is a thing that goes stale.

## 7. Out of scope

- **Any change to `Frame`.** The recommendation needs none.
- **A stored transcript of structured results.** D14 stands (rule 4).
- **The search-side contracts.** They are built, released and unchanged by this.
- **metallm's frontend work itself.** This decides the channel; consuming it is
  metallm's, after its `feature/new-search` migration.

## 8. The build — 2026-08-18

Approved as written, with D-S2 corrected by its own consumers. What shipped is
the channel; what did not is the one branch that cannot be built here without
this package becoming a store.

**Shipped** (`packages/langgraph`):

| Piece | Where | Why it is that shape |
|---|---|---|
| `structure_for_stream(artifact, *, max_chars)` | `tool_structure.py` | the whole decision — inline, omitted, or absent — in one function, so neither face makes it twice and the answer cannot differ per emitter |
| `StructuredToolResultFields` mixin (`structured`, `structured_kind`) | `tool_structure.py` | the two faces inherit the pair rather than declaring it twice; a field added to the channel lands on both in the same commit, which is check 14's rule applied to the channel itself |
| the pair on `ToolCompletedEvent` | `events.py` | D-S1(a) — metallm's face, one message and two registers |
| the pair on `ToolCallEndEvent` | `streaming.py` | D-S1(a) applied to the hop §1's table missed — scriob's face |
| `emit_tool_call_end(..., artifact=..., structured_max_chars=...)` | `streaming.py` | the caller already holds the `ToolMessage` when it stops the clock; it hands the artifact over and the emitter projects. A caller that passes nothing emits exactly what it emitted before |
| 25 pins | `tests/test_tool_structure.py` | including the two the program's own scars demand, below |

**An inline payload is the artifact verbatim.** Not a re-key, not a narrowing,
not a projection this package builds: `bind` stays the only construction site
(rule 2, `tests/enforcement/test_one_search_result_shape.py`), and the stream
forwards what it was handed. That is also what generalises the channel past
search for free — `page_finder` and `web_fetch` structure rides it with no
per-tool code, which is open question 3 answered by construction rather than by
a schema story.

**Two pins earn their place beyond ordinary coverage.**

*A reader that predates the field.* Both events grew a declared optional, and a
declared optional **serializes** — `"structured":null` crosses the wire from the
first emit, which is the exact shape that refused every call from a 0.24.1
registry for three days on 2026-08-13. The argument that these models tolerate
it is sound (neither sets `extra="forbid"`; `FrameworkEvent`'s docstring
promises additive safety) and the outage was what an argument is worth. So the
tests hand real emitted bytes to hand-written models that have never heard of
`structured` — populated and null alike — rather than asserting the property in
prose.

*The faces compared to each other.* Not each to a constant. That is the Gate B
sweep's finding about `test_egress_independence`, where two sides were each
pinned against the value they were configured from and the requirement's own
hard case had never been driven.

**`structured_kind` is a `str`, not a `Literal`.** The design sketched
`Literal["inline", "handle"]`. A closed vocabulary on a wire model means a
reader predating a *fourth* kind rejects the event instead of ignoring the
value — the 2026-08-13 lesson one level down, in the value rather than the
field. The constants are the vocabulary; an unknown kind is a thing to skip.

**Three kinds, not two.** `inline` and `handle` as designed, plus `omitted`:
over the bound with no host store available, the event says so, in the payload,
carrying its size and the bound it missed. D-S3's own rule — "the bound must be
explicit in the payload, not implicit in the sender" — has to survive the case
where the handle is unavailable, and silence there would be the
silent-partial-answer defect wearing a null. The omission record is written
under its own `omitted` key, never the projection's: D-S4's one hard *must not*
applies to this payload as much as to a narrowed projection.

### An adjacent finding, one layer over

**The one in-repo emitter of `ToolCompletedEvent` cannot populate the channel,
for the reason [#318](https://github.com/pacepace/3tears/pull/318) already
fixed elsewhere.** The Claude-CLI provider
(`packages/models/src/threetears/models/providers/_claude_cli.py:374-390`)
invokes each wrapped tool as `tool.ainvoke(args)` — plain args, not the whole
tool call — so LangChain never builds a `ToolMessage` and a
`content_and_artifact` tool hands back the raw `(content, artifact)` tuple,
which the wrapper then renders as `str(result)`. There is no artifact to read
because the artifact was never separated, and the model sees a stringified
tuple. That is the same defect `ToolExecutor` carried until #318 and
`page_finder` carried until [#326](https://github.com/pacepace/3tears/pull/326)
— its third appearance, in the third place that invokes a tool by args.

Left unfixed here deliberately: it is a provider-path behaviour change (the
tool text a running agent sees), it belongs with the two prior fixes rather
than with a wire channel, and nothing about this design depends on it. Both
consumers that will actually drive this channel invoke correctly today —
metallm through its tool loops, scriob through an `awrap_tool_call` middleware
that is handed the finished `ToolMessage`.

### What is NOT built, and the decision it waits on

**The handle branch has a discriminator and no producer.** D-S3(c) was approved
from v1 on the strength of §2 — `ObjectHandle` + `GET /api/v1/media/{id}/url`
is built, in production, and metallm is already its caller. That is true of
*resolution*. It is not true of *production*: nothing anywhere stores a search
projection and mints an id for it, and this package must not become the thing
that does (D7 / D12 / D14, rule 4 — the stream is not a store).

The family already owns the right-shaped port, and it is one line from this
seam: `ToolResultOffloader`
(`packages/langgraph/src/threetears/langgraph/offload.py`), injected on
`config["configurable"]`, already moves oversized tool *content* out-of-band and
returns a summary plus a handle. The open question is not how to build the
producer; it is **which store the client resolves against**, and that is a
consumer-side decision with two live answers:

1. **Reuse the offload store.** Zero new infrastructure on the producing side —
   the host that already injects an offloader gets the handle branch by
   configuration. Cost: the `[ctx:<id>]` handle has **no client-facing resolve
   surface** today (verified in §2: no `ctx:` reference anywhere in metallm's
   frontend), so metallm or scriob owes an endpoint.
2. **Mint an `ObjectHandle` into the object store.** Reuses the resolve path
   metallm's frontend already implements for media — no new client work at all.
   Cost: a producing-side store this seam does not have, and a lifetime
   question the media path answers for images and does not answer for a
   transient search projection.

Until one is taken, an oversized projection is an honest `omitted` rather than
a silent truncation, and the wire does not move when the answer arrives.
