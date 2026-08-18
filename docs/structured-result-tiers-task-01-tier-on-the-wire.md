# structured-result-tiers-task-01 — the declared tier reaches the wire

**Ruling:** `structured-result-tiers.md` §2, §2.1, 2026-08-18. **Status:** not
built.
**Blocks:** task-02 (a handle is only the right answer for a client that asked
for bodies), task-03 (the room-level decision is a decision about tiers).
**Blocked by:** nothing. [#355](https://github.com/pacepace/3tears/pull/355)
landed the channel this attaches to; it is on `develop` at `d27e3875`.

Read `structured-result-tiers.md` for *why*. This document is *what to build*.

---

## 1. The one-paragraph version

#355 put a tool's structure on the wire and gave it one bound: over 16,384
characters of JSON the client gets an omission record instead of a result. This
task adds the client's half of that exchange — a tier, `citations` or `full`,
declared once, that decides what the payload *contains* before the bound
measures it. The tier is a subtraction of content bodies, never a list of fields
to keep, and it is read where the frame is built and nowhere else.

Nothing here persists anything, and nothing here decides what a tool produces.
The handle path is task-02; this task's `full` tier behaves exactly like today
when it doesn't fit.

## 2. What already works, and must not be re-done

Verified against `develop` at `d27e3875`.

- **The channel is complete on both faces.** `StructuredToolResultFields`
  (`tool_structure.py`) is inherited by `ToolCompletedEvent`
  (`events.py:135`) and `ToolCallEndEvent` (`streaming.py:315`). That mixin is
  the reason a field lands on both faces in one commit. Add `structured_tier`
  **to the mixin**, never to either class.
- **There is one decision point.** `structure_for_stream` is called from exactly
  one production site, `emit_tool_call_end` (`streaming.py:678`). The tier
  threads through one signature, not a dozen emitters.
- **The open-vocabulary posture is already set.** `structured_kind` is a `str`
  and not a `Literal`, so a reader meeting an unknown value skips it instead of
  rejecting the event. `structured_tier` takes the same treatment for the same
  reason; the design's rule 3 is this package's existing practice, not a new
  ask.
- **The bound is already a placeholder with an owner.**
  `DEFAULT_STRUCTURED_INLINE_MAX_CHARS = 16384`, read from one place so that
  answering it moves one line. Do not fork it into a per-tier constant.
- **The projection and its body slot exist.** `ContentSlot`
  (`contracts/candidate.py:52`) carries `text` (required), `origin`,
  `mime_type`, `size_bytes`, and the two HTTP validators. `Candidate.content`
  is the only unbounded field in the projection, which is what makes "bodies in
  or out" a complete axis.
- **The border key has exactly one writer, and a test holds it there.**
  `SEARCH_RESULTS_METADATA_KEY` may be read anywhere and written only in
  `threetears.search.bind` — `tests/enforcement/test_one_search_result_shape.py`.
  A citations payload rides that key, so this task is that test's business
  before it is anything else's.

## 3. Rulings taken before the build

### R1 — The withholding is a projection built in `bind`, and nowhere else

The tempting shortcut is a generic walk in `tool_structure`: find any mapping
under `"content"` that has a `"text"`, replace it with a marker. It needs no
new dependency and it works on the artifact `emit_tool_call_end` already holds.

Refuse it. `packages/langgraph` does not depend on `packages/search` (check the
`dependencies` list — `3tears`, `media-contracts`, `observe`, langchain), and
that is not an accident of packaging: a walk that recognises the search
projection's field names *is* knowledge of the search schema, written in a
package that cannot import it and therefore cannot be checked against it. It
would also be a second writer of the border payload wearing the border key,
which is exactly the shape `test_one_search_result_shape.py` exists to catch —
and it would sit where that test cannot see it, because at that layer the key
is a runtime string rather than a dict literal.

So the withholding is a function in `bind`, beside `project_metadata`
(`bind.py:156`), which already owns the schema version and the field names:

```python
def withhold_bodies(payload: Mapping[str, Any]) -> dict[str, Any]: ...
def cap_bodies(payload: Mapping[str, Any], *, max_chars: int) -> dict[str, Any]: ...
```

They take the already-projected border mapping and return a new one. That is
not a second construction site: it is bind editing the payload bind built,
under bind's own roof, and the enforcement test's allowlist already names that
module.

### R2 — `langgraph` learns the tier, never the schema

`tool_structure` gains a port, in the shape `offload.py:153` already
established for exactly this problem — a `Protocol` here, an implementation
elsewhere, injected by the host:

```python
class StructureTierProjector(Protocol):
    def project(self, artifact: Mapping[str, Any], *, tier: str,
                max_body_chars: int | None = None) -> Mapping[str, Any] | None: ...
```

`None` means *this projector does not know how to narrow this artifact* — an
artifact from some other tool, or a tier it does not implement. It is not an
error and must not be logged as one; most artifacts on most turns are not
search projections.

The implementation for the search key lives with the code that already imports
both sides (`packages/agent/tools`, whose dependencies carry `3tears-langgraph`
and `3tears-search`), and dispatches on `SEARCH_RESULTS_METADATA_KEY` into R1's
`bind` functions.

### R3 — A tier that cannot be honoured is not claimed

No projector wired, or the projector returned `None`: emit what today emits —
the artifact measured whole against the bound, an omission if it is over — and
leave `structured_tier` unset.

Both halves matter. Sending bodies while the payload says `citations` is the
lie rule 1 forbids. Claiming a tier we applied nothing for is the same lie in
the other direction, and it is the more likely one to ship, because it looks
like bookkeeping. An unhonoured `citations` declaration degrades to today's
behaviour, which is the failure the client declared to avoid — that is a
degradation and it is allowed to happen, but it is worth one log line at the
host, once per process, not per call.

### R4 — The withheld body is a wire shape, and the contract parser must refuse it

`ContentSlot.text` stays required. Do not relax it to express withholding.
Every in-process reader depends on it — `extract.py:210` and `extract.py:354`,
`corpus.py:150`, `web_fetch.py:313` — and §2.1 is the rule that a tier must not
reach them. A contract relaxation would reach all of them at once, which is
precisely the quiet verification failure §2.1 was written about.

The withheld body is therefore a shape that only exists on the wire:

```json
"content": {"withheld": true, "size_chars": 102400, "mime_type": "text/plain"}
```

No `text`, no `origin`. Because `ContractModel` sets `extra="forbid"` and
`text` / `origin` are required, `SearchResultsMetadata.from_metadata` **raises**
on a citations payload rather than reconstructing a result that reports no
content. That refusal is the property to test, and it is a feature: pydantic's
`ValidationError` subclasses `ValueError`, so the one existing caller that
guards this parse (`graph_nodes.py:223`) already catches it and logs rather
than failing a turn.

It leaves the border key carrying a payload that does not round-trip — the
asymmetry `_one_search_result_shape_exemptions.txt` warns about in its note on
the context-save node. The difference is rule 1: this payload's event names its
tier, so a reader knows before it parses. Which is only true if the tier
travels with it, hence R5.

### R5 — The tier belongs to the envelope, and anything that copies `structured` out must carry it

`structured_tier` is a sibling field of `structured`, not a key inside it. That
is the right place — the payload is the artifact's own shape, and burying a
platform field inside it would be inventing the second result shape by a
different route.

The cost is that `event["structured"]` handed to a parser on its own has lost
the tier. So: any consumer that lifts `structured` out of its event MUST carry
`structured_tier` with it, and any *producer* that re-wraps one (a proxy, a
replay, a test fixture) MUST move both fields or neither. Say it in the mixin's
docstring, where a reader of either face will meet it.

### R6 — `max_body_chars` is a parameter of `full`, and a capped body says so

`citations` takes no amount; withholding has no dial. `full` takes an optional
`max_body_chars`, and a body that was cut says what happened to it:

```json
"content": {"text": "Counts rose 12% across …", "truncated": true,
            "size_chars": 102400, "delivered_chars": 4000}
```

This is a wire shape for the same reason R4's is, and `from_metadata` refuses it
for the same reason (`truncated` and `delivered_chars` are unknown fields under
`extra="forbid"`). Cap on characters, not bytes, matching the unit the bound and
the offload seam are already in.

A body that fits its cap is emitted unchanged — no `truncated: false` marker on
untouched content. The marker's presence is the signal, and marking every body
would put a platform field on every result that never needed one.

### R7 — The tier is applied at the frame, and an enforcement test says so

§2.1's failure mode is silent: a tier applied one layer too early strips the
corpus a grounding gate verifies against, and grounding keeps returning
answers. Nothing about that failure shows up in a rendering test.

So bound it structurally. R1's `withhold_bodies` / `cap_bodies` may be called
from exactly one place — the projector implementation R2 describes — and the
projector port may be invoked from exactly one place, `structure_for_stream`.
An AST test in the shape of `test_one_search_result_shape.py` holds both, with
an exemptions file that makes any third site a written decision rather than an
import.

## 4. What is missing (the build)

**`packages/langgraph` — `tool_structure.py`**

- `STRUCTURED_TIER_CITATIONS = "citations"`, `STRUCTURED_TIER_FULL = "full"`,
  exported, documented as an open vocabulary (R2/rule 3).
- `structured_tier: str | None` on `StructuredToolResultFields`, with R5 in the
  docstring.
- `StructureTierProjector` protocol (R2).
- `StreamStructure` grows `structured_tier`; `as_fields()` returns three keys.
- `structure_for_stream(artifact, *, max_chars=..., tier=None, max_body_chars=None,
  projector=None)`. Order of operations, and it is the whole point: **project,
  then encode, then measure.** A tier applied after the bound has been missed
  changes nothing.

**`packages/langgraph` — `streaming.py`**

- `emit_tool_call_end` grows `tier`, `max_body_chars`, and the projector; the
  projector is a construction-time dependency of the publisher, the tier is
  per-emit (it comes from the connection). Callers that pass nothing emit what
  they emit today plus one more null.

**`packages/search` — `bind.py`**

- `withhold_bodies` and `cap_bodies` (R1), each returning a new mapping and
  mutating nothing — the artifact belongs to the caller and the in-process
  readers are still using it.

**`packages/agent/tools`**

- The projector implementation: dispatch on `SEARCH_RESULTS_METADATA_KEY`,
  call into `bind`, return `None` for anything else.

**`tests/enforcement`**

- The call-site test from R7, plus its exemptions file.
- `test_one_search_result_shape.py` needs a decision recorded either way: the
  new `bind` functions write the key from the sanctioned module, so the test
  should keep passing untouched — confirm that, and if the allowlist is
  function-scoped rather than module-scoped, widen it deliberately and say why.

## 5. Sequencing

1. `bind.withhold_bodies` / `cap_bodies` + their tests. Pure functions over a
   mapping; nothing else moves.
2. The vocabulary, the mixin field, and the port in `tool_structure`. Additive,
   no behaviour change — every existing test keeps passing with three nulls
   instead of two.
3. `structure_for_stream` applies the tier before it measures.
4. `emit_tool_call_end` threads it.
5. The projector in `agent/tools`.
6. The enforcement tests (R7), last, so they are written against the shape that
   actually landed.

Steps 1–2 are independently landable and version-bump-visible: the mixin gains
a public field, so `tests/enforcement/test_api_growth_requires_a_minor_bump.py`
applies and the family minor moves. Do not try to slip this in as a patch.

## 6. Tests the build owes

- **Project-then-measure.** A projection whose full encoding is over the bound
  and whose citations encoding is under it delivers inline at `citations` and
  omits at `full`. This is the whole task in one test.
- **Withholding is subtractive.** A candidate carrying an unrecognised facet
  keeps that facet at `citations`. Written against `facets` specifically,
  because a field-list implementation passes every other test in this file.
- **A withheld body is distinguishable from an absent one.** A candidate with
  no content at all and a candidate whose 100 KB body was withheld produce
  different payloads, and neither produces a bare missing key.
- **The contract refuses a tiered payload.** `from_metadata` raises on both
  R4's and R6's shapes. Assert the raise, not a message.
- **Tier claimed only when applied.** No projector → no `structured_tier`, and
  the payload is byte-identical to today's.
- **Unknown tier degrades.** `tier="turnip"` with a projector that returns
  `None` behaves as R3 says, and nothing raises anywhere.
- **Both faces carry the field.** The mixin makes this structural, so the test
  is over the mixin's fields, not two hand-written assertions.
- **`max_body_chars` marks what it cut, and leaves alone what it didn't.**
- **The call-site bound** (R7), including a deliberate violation in a fixture to
  prove the test fails.

## 7. Explicitly out of scope

- **The handle.** Over the bound at `full` stays an omission until task-02.
- **The ceiling above the bound** and who measures a frame before publishing it
  — task-03.
- **Where a client declares its tier.** A connect parameter, a subscribe frame,
  or a per-turn field is each product's decision (§6, open question 1); this
  task ends at a `tier=` argument the host fills in.
- **Summaries.** §5, and they belong to search if they happen at all.
- **`_provenance_of`'s narrowing** (`graph_nodes.py:227`). It writes a
  narrowed record under the border key with no tier to name it — the same
  asymmetry, in the context store rather than on a wire, already exempted with
  a rationale. Leave it; it is a store row for prompt-side traceability, not a
  frame to a renderer, and folding it into this vocabulary would put a wire
  concept in the retention path.
