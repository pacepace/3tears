# structured-result-tiers-task-02 -- over the bound is a handle, never an omission

**Ruling:** `structured-result-tiers.md` §4, 2026-08-18. **Status:** not built.
**Blocks:** nothing in this repo. The resolve surface it implies is the
consumers' (§3, §6).
**Blocked by:** task-01 -- "a client that asked for `full`" is not expressible
until a client can ask.

Read `structured-result-tiers.md` §4 for *why*. This document is *what to
build*.

---

## 1. The one-paragraph version

By the time a projection is too big the expensive part is already bought: the
provider call was made, the bytes were fetched, and on the scrape path an LLM
extraction ran. For a client that declared `full`, dropping that and sending a
receipt is throwing away something we paid for, hold in hand, and were asked
for. This task makes over-the-bound at `full` mint a handle instead -- the
citations tier inline so something always renders, plus a reference to the part
that didn't fit. `citations` clients are untouched: they never wanted the body,
so not sending it was never waste.

## 2. What already works, and must not be re-done

- **The kind already exists.** `STRUCTURED_KIND_HANDLE` shipped in #355 with
  nothing producing it, deliberately, so that turning it on later does not move
  the wire. This task is that later. Do not add a fourth kind.
- **The store port exists and is wired.** `ToolResultOffloader`
  (`offload.py:153`) is injected on `config["configurable"]`, already moves
  oversized tool *content* out of band, and is already conversation-scoped -- 
  it takes a verified `conversation_id` / `user_id` and never sees the store
  itself, which is what keeps `langgraph` free of a context-store dependency.
  Copy that posture exactly; do not import a store into `langgraph`.
- **The threshold precedent exists.** `DEFAULT_OFFLOAD_THRESHOLD_CHARS = 8192`,
  and the inline bound is deliberately twice it. Two different questions with
  two different numbers, both in characters. Keep them separate.
- **The marker shape has one owner.** `format_offload_handle`
  (`offload.py:102`) is the single source of truth for `[ctx:<id>]`. It is the
  *model's* marker, appended to prose the model reads. See R4.
- **On exactly these turns the content is usually already stored.** A tool
  result over 8,192 chars has been through `ToolResultOffloadMiddleware`
  already. What is missing is not a store; it is that the *projection* was
  never the thing stored.

## 3. Rulings taken before the build

### R1 -- A handle when a store is reachable; an omission only when none is

`full` + over the bound + an offloader on the config → handle. `full` + over
the bound + no offloader → today's omission record, unchanged. The omission is
not deprecated and its reason codes do not change; it becomes the answer to
"nowhere to put it" rather than the answer to "too big".

`citations` never takes this path. If a citations payload is over the bound -- 
50 results is 47,500 characters against a 16,384 bound, so this is not
hypothetical -- it omits, exactly as today. A handle there would be solving a
problem the client did not have with a fetch the client may not be able to
make.

### R2 -- What gets stored is the projection, and the handle addresses a result set

The design's one forward-compatibility constraint (§4, *Do we need
pagination?*): a handle MUST address a result set, not an opaque blob, so
`?offset=&limit=` can be added later without moving the wire. "The projection
for tool call X" can grow paging; "blob 9f2c" cannot.

Concretely: what is stored is the **full-tier projection** -- the artifact
`structure_for_stream` was handed, JSON-encoded, whole. Not the prose, not the
bodies alone, and not a narrowing. A resolve that returns something the client
would then have to merge with what it already rendered is a second result
shape with extra steps.

### R3 -- A handle frame always carries the citations tier inline

Never a bare reference. The handle payload is:

```json
{"handle": "…", "summary": "1 result, 100 KB extracted",
 "citations": { "…the citations tier, inline…" }}
```

so every path delivers something renderable and no client shows a spinner
waiting on a fetch it may never make. This is rule 2 of the design, and it is
also what makes the room-level tier in task-03 work at all.

It costs one extra projection on the over-bound path -- task-01's
`withhold_bodies` over the same artifact -- and that path is by definition rare
and already expensive.

### R4 -- A new port, not a grown one, and not the model's marker

Two temptations, both refused.

**Do not extend `ToolResultOffloader.offload` with a `structured=` parameter.**
`search-task-01` §3.1 recorded the correction the hard way: a default keeps an
existing implementer conformant only if the caller does not pass the argument,
and an implementer written before the parameter exists has no such parameter,
so passing it is a `TypeError`. `ToolResultOffloader` is a published Protocol
with implementers outside this repo. It gets a sibling:

```python
class StructuredResultStore(Protocol):
    async def store(self, *, tool_name: str, projection: str,
                    conversation_id: UUID, user_id: UUID | None) -> str | None: ...
```

`None` means "declined to store" and falls back to R1's omission, matching
`offload`'s own `None`-means-fall-back contract.

**Do not reuse `[ctx:<id>]` on the wire.** That marker is prose, appended to
content the model reads, and `has_offload_handle` exists to detect it in text.
The client's handle is a field in a JSON payload. One store may back both, but
the two readers are a language model and a browser, and giving them one syntax
means every future change to either has to be safe for both.

### R5 -- Conversation scope, because a turn-scoped handle is already known to be too short

§6 asks for a lifetime and one floor is already known: discodon's research
delivery is asynchronous, landing on a later turn than the one that asked, so a
turn-scoped handle would be dead before its first reader. Conversation-scoped is
the minimum that works for a consumer we already have, and it is the scope the
offload port is already built around -- so this is the cheap answer as well as
the right one.

A TTL below the conversation's own lifetime is a deployment's business and is
not decided here. A handle whose target is gone must resolve to a clean *gone*,
not a 500 and not an empty projection: the client has already rendered
citations and is expanding one, so "it expired" is a renderable answer and an
empty result set is a wrong one.

### R6 -- Resolve is an authorisation boundary the offload path never had

This is the ruling most likely to be skipped, because the store looks like the
same store.

`context_recall` is reached by the model, from inside a conversation whose
identity was already verified before the graph ran. The resolve surface is
reached by a browser, over HTTP, holding a string it got in a frame -- and on a
broadcast channel (§3) that frame went to *every viewer of the room*, not only
to the author. So a handle is an identifier, never a bearer token.

Two obligations follow, and they belong in this repo's contract even though the
endpoint is the host's:

1. The resolve surface MUST authorise the caller against the conversation (or
   room) the handle belongs to, on every request. Possession proves nothing.
2. The handle MUST be unguessable. It rides the existing stored-item id, so
   this is a property to check rather than to build -- but check it, because a
   sequential id that was fine when only the model could name it stops being
   fine the moment a frame carries it to a browser.

Write both into `StructuredResultStore`'s docstring, where an implementer will
meet them.

### R7 -- The cheap-to-keep case is real, and it is not this task

§4's second argument -- *expensive to produce, cheap to keep* -- stands on its
own: discodon buys half a megabyte to two megabytes of page text, grounds
against it once, drops it, and re-buys it on the follow-up question. None of
that was ever too large for a frame, because none of it was ever going to a
frame.

That is a retention decision with a retention owner (D12, SR-K4 -- a
deployment's agreement with a site governs whether it may keep a page's text),
and wiring it to a frame-size trigger would be storing third-party content as a
side effect of a rendering budget. Same port, different caller, separate
decision. Note it in the module docstring so the next reader finds the argument
rather than re-deriving it.

## 4. What is missing (the build)

**`packages/langgraph` -- `offload.py`**

- `StructuredResultStore` protocol (R4), with R5 and R6 in its docstring.

**`packages/langgraph` -- `tool_structure.py`**

- The handle payload builder: `handle` + `summary` + `citations` (R3), and the
  key names are the wire, so fix them here and reference them from the design
  doc rather than the other way round.
- `structure_for_stream` becomes able to *return* `STRUCTURED_KIND_HANDLE`.
  Note that it acquires an async caller for the first time: storing is I/O.
  Either the store call happens in `emit_tool_call_end` and the decision
  function stays synchronous, taking a pre-minted handle, or the function goes
  async. **Prefer the first** -- the decision stays a pure function of
  (artifact, tier, bound), which is what makes task-01's tests as cheap as they
  are, and the I/O stays where the `await` already is.

**`packages/langgraph` -- `streaming.py`**

- `emit_tool_call_end` grows the store dependency, mints the handle on the
  over-bound `full` path, and falls back to the omission when the store
  declines or raises. A store failure MUST NOT fail the emit: the client is
  owed its citations either way.

**`packages/langgraph` -- `middleware_offload.py`**

- Nothing, and confirm that. The two paths share a store, not a code path, and
  the model-visible content offload must keep working identically for a tool
  whose projection also got a handle.

## 5. Sequencing

1. The protocol (R4). Nothing depends on it yet.
2. The handle payload shape + the citations-inline rule (R3), tested against a
   pre-minted handle string, no store in sight.
3. `emit_tool_call_end` wiring, including the decline and raise paths.
4. The docstring obligations (R5, R6, R7) -- last only in the sense that they
   land with the code they describe, not that they are optional.

## 6. Tests the build owes

- **`full` over the bound with a store mints a handle**, and the payload carries
  a renderable citations block, not a bare reference.
- **`full` over the bound with no store omits**, byte-identical to today.
- **`citations` over the bound omits** even with a store present (R1).
- **A store that returns `None` falls back to the omission**; a store that
  *raises* also falls back, and the emit still happens. Two tests, because the
  second one is the one that gets forgotten.
- **What was stored is the full projection**, asserted by round-tripping the
  stored string back through `SearchResultsMetadata.from_metadata` -- which
  succeeds only if it really is the untiered payload (task-01 R4 makes a tiered
  one refuse, so this assertion is load-bearing in both directions).
- **The handle is not the model's marker**: the payload's handle field does not
  match `_OFFLOAD_HANDLE_RE`, and `has_offload_handle` is not involved.
- **The offload middleware is unaffected** on a turn that produced both.

## 7. Explicitly out of scope

- **The resolve endpoint itself.** It is HTTP, it is per-product, and metallm
  already runs the shape it should follow (`GET /api/v1/media/{id}/url`). This
  task ships the handle and the obligations on whoever resolves it.
- **Pagination.** R2 keeps it possible; nothing implements it. Do not add
  `offset` / `limit` to a surface with no reader.
- **Retention on the cheap-to-keep argument** (R7).
- **A handle for `citations`** (R1).
