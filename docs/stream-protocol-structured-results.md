# Stream protocol: the channel for structured tool results

**Status:** design, for review — 2026-08-14. Nothing here is built.
**Ruling needed from:** 3tears (owns the contracts), metallm (first frontend
consumer), chat-kit workstream (§4.11 — the design's actual client), scriob and
samsung (planned chat surfaces).
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

### D-S3 — How much structure, and what happens when it is large?

This is the substantive decision and the one most needing outside input.

A corpus of fifty candidates with content slots is not a websocket frame. But a
summary is not re-checkable (rule 3).

| Option | For | Against |
|---|---|---|
| **(a) The full projection, bounded, with an explicit truncation mark** | Re-checkable in the common case; one shape; nothing new to build | A large result is still truncated, and the client cannot get the rest |
| (b) Always a handle (`ObjectHandle`-style) the client fetches | Unbounded results work; reuses a family primitive D22 already follows | Needs an endpoint and object-store reach the client may not have; a round-trip before anything renders |
| (c) Inline when small, handle when large | Best of both | Two paths, and the client must implement both to be correct |

**Recommendation: (a) for v1, with (c) recorded as the escalation.** The
program's own rule against building a seam whose only caller is a test
([`search-task-03`](search-task-03-producer-seam-sketch.md), where the producer
seam was split off for exactly this reason) applies: no client exists yet that
has proven it needs unbounded results in a stream frame. Ship the bounded form,
and let the first consumer that hits the bound justify the handle.

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

On the wire, one frame:

```json
{
  "type": "tool.result",
  "room": "acme:story:main:chat",
  "payload": "{\"search_results\": { ... the projection ... }}"
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

## 6. Open questions — these need stakeholders, not this document

1. **What is the frontend's actual payload budget per frame?** D-S3's bound is a
   number this repo cannot pick. metallm's answer decides whether (a) is
   sufficient or (c) is needed immediately.
2. **Will chat-kit clients have object-store reach?** If not, D-S3(b) and (c)
   are off the table for browser clients regardless of size, and the bound in
   (a) becomes a hard ceiling rather than a default.
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
