# structured-result-tiers-task-03 — the ceiling above the bound, and its quiet failure

**Ruling:** `structured-result-tiers.md` §3, §3.1, 2026-08-18. **Status:** not
built.
**Blocks:** nothing. **Blocked by:** nothing in the code — it is independent of
task-01 and task-02, and worth landing whether or not tiers ship, because the
failure it closes exists today.

Read `structured-result-tiers.md` §3.1 for *why*. This document is *what to
build*.

---

## 1. The one-paragraph version

The inline bound is a budget somebody chose. Above it sits a limit nobody chose:
a room frame crosses NATS, and nats-py refuses an oversized publish client-side
against the broker's advertised `max_payload` — 1 MB on an untuned broker, which
works back through the frame nesting to roughly 780,000 characters of artifact.
`citations` sits about 16× under it; `full` does not. The failure is silent on
both consumers by deliberate design on each side, so the person who ran the
search sees their citations, the room sees nothing, and single-pod tests pass.
This task gives that refusal a name, puts the check where a caller cannot fix it
from outside, and writes down which frame shape is the one that always
publishes.

## 2. What already works, and must not be re-done

- **There is exactly one funnel for core publishes.** `NatsClient.publish`,
  `publish_raw`, `publish_reply` and the positional shorthand all end in
  `_publish_bytes` (`client.py:1650`), which already wraps every raised
  exception into a typed `PublishError` and already counts one specific
  sub-case (`_note_if_outbound_overflow`). Add nothing to the four callers.
- **nats-py already refuses.** The check is client-side, before anything leaves
  the process, and it raises `nats.errors.MaxPayloadError`. Do not reimplement
  the limit; the broker's advertised value is the truth and it is already being
  enforced. What is missing is that it arrives as a generic `PublishError`
  carrying a stringified cause.
- **The broker's value is readable.** `nats.aio.client.Client` exposes
  `max_payload`. `NatsClient` does not surface it — that is the gap, not the
  measurement.
- **The reasoning is already written down once**, in
  `pipe.py:169-182`, where `DEFAULT_MAX_CHUNK_BYTES` is 64 KiB precisely
  because it sits under an untuned 1 MB with room for headers. Cross-reference
  it; do not write a second copy of the argument.
- **The precedent for where a bound lives is `_publish.py`.** Its docstring
  makes the argument this task reuses: a caller cannot fix, from outside, a
  failure mode that lives inside the publish path. That is why `publish_bounded`
  is in this package, and it is why the size check belongs beside it rather
  than in each fanout.
- **`RoomFanout.broadcast` (`fanout.py:157`) is the one place both room
  consumers pass through**, and it does not catch. Per-socket delivery in
  `_deliver` is best-effort by design and stays that way; that is a different
  failure with a different correctness story.

## 3. Rulings taken before the build

### R1 — The refusal gets a type, and it names both numbers

`PayloadTooLargeError(PublishError)`, raised from `_publish_bytes` when the
cause is `MaxPayloadError`, carrying the subject, the payload size in bytes,
and the broker's `max_payload`. Catch-and-retype, not a pre-check: the limit
has one source of truth and it is the connected server, so re-deriving it above
the client invites the two to disagree exactly when it matters.

Why a type at all, when the message already says it: because both consumers
*catch* here, deliberately, and a consumer that catches `PublishError` cannot
tell "the frame was too big" from "the broker went away" without matching on
string content. One of those is a bug in what we built; the other is an outage.
They deserve different log lines and different responses, and today they cannot
have them.

### R2 — Expose `max_payload`, so a caller can choose a smaller payload instead of failing

`NatsClient.max_payload` (property, `int | None` — `None` before connect).
This is the half that makes R1 more than a nicer error. A frame builder that
can ask how much fits can pick the shape that fits — task-01's `citations`, or
task-02's handle — instead of building something large, publishing it, and
discovering the answer as an exception.

`None` before connect is deliberate and must not be papered over with a 1 MB
default. A default here would be a second source of truth for the one number
this task exists to stop guessing, and it would read as authoritative at exactly
the moment nobody has asked the broker yet.

### R3 — The policy is at the frame, the guard is at the wire, and they meet at a type

Do not put a size *policy* in `_publish_bytes`. It cannot know whether the right
answer for an oversized frame is a handle, a narrower tier, a chunked pipe, or a
refusal — those are different for a room frame, a KV update, and a token stream.

The split:

- **`packages/nats`** owns the guard: measure nothing, catch precisely, raise a
  type that names the numbers (R1), and answer the question when asked (R2).
- **The frame builder** owns the policy, and for a broadcast room the design
  already picked it (§3): the tier belongs to the room, the frame carries
  `citations` plus a handle, and a viewer wanting more resolves it. That is the
  only shape that always publishes, which is a better argument for it than the
  per-viewer-appetite one the design leads with.

### R4 — `RoomFanout.broadcast` states its failure mode, and does not grow a catch

`broadcast` lets `PublishError` propagate today, and that is correct — this
package is not where the decision about a lost room frame is made. What it owes
is a docstring that says so, including the fact that its two known callers each
catch (scriob so a fanout failure cannot be mistaken for the author's own socket
dying, metallm falling back to local-only delivery with a warning), and that
`PayloadTooLargeError` is the one they must not treat as transient. A retry of
an oversized publish is an infinite loop that logs.

This is a documentation change with a real target: `tests/enforcement/test_no_silent_swallow.py`
already holds this repo's handlers to logging or re-raising, and the equivalent
handlers on the consumer side are outside its reach. A named type in a
docstring is the only lever this repo has on them.

### R5 — Raising the inline bound is not the answer, and the numbers say where it stops

Measured through the real types by
[`scripts/measure-structured-result-sizes.py`](../scripts/measure-structured-result-sizes.py):
the event → `Frame` → `RoomFrame` nesting costs 1.20× on a body-heavy payload
and 1.34× on a metadata-heavy one, so an artifact-character bound produces a
frame about a third larger. Eight results carrying 100 KB each publishes at
1,010,599 bytes — under an untuned 1 MB with less than 4% to spare — and §4's
own research corpus, at the top of its range, is refused outright.

So "raise the bound" stops working before `full` gets interesting, and anything
past it is task-02's handle. Write the ceiling into
`DEFAULT_STRUCTURED_INLINE_MAX_CHARS`'s docstring as an upper limit on the
answer metallm is expected to give, so the open question is answered inside a
range rather than into open air.

### R6 — Convert with the script, never with the ratio

1.20× and 1.34× are two measurements of one payload shape each, not a constant.
Anyone turning a frame budget into an artifact bound re-runs the script against
the shape they actually send. The two figures are in this document to show that
the multiplier is neither 1.0 nor stable, which is the only claim they support.

## 4. What is missing (the build)

**`packages/nats` — `errors.py`**

- `PayloadTooLargeError(PublishError)` with `subject`, `size_bytes`,
  `max_payload` as attributes, not only interpolated into the message.

**`packages/nats` — `client.py`**

- `_publish_bytes` recognises `MaxPayloadError` and raises R1's type. Keep the
  existing `_note_if_outbound_overflow` call and the generic wrap for everything
  else; this is one new branch, not a rewrite of the handler.
- `max_payload` property (R2).
- The JetStream path deserves the same treatment where it shares the failure —
  check whether `publish_bounded`'s caller can hit `MaxPayloadError` before the
  ack wait, and if so, retype it there too rather than leaving one publish path
  typed and the other not.

**`packages/channels` — `presence/fanout.py`**

- `broadcast`'s docstring gains R4: what propagates, who catches it downstream,
  and which failure must not be retried.

**`packages/langgraph` — `tool_structure.py`**

- One line in `DEFAULT_STRUCTURED_INLINE_MAX_CHARS`'s docstring (R5): the
  placeholder has an owner *and* a ceiling.

## 5. Sequencing

1. The error type. Nothing depends on it.
2. `_publish_bytes` retypes; `max_payload` is exposed. One commit — the property
   without the type is a question with no useful answer, and the type without
   the property leaves the caller unable to act on it.
3. The two docstrings.

None of this is a wire change and none of it is a new public model, so this
lands as a patch to `packages/nats` plus docs — but `max_payload` is public API
growth, so check `tests/enforcement/test_api_growth_requires_a_minor_bump.py`
before assuming a patch bump is available.

## 6. Tests the build owes

- **An oversized publish raises `PayloadTooLargeError`, not a bare
  `PublishError`**, and the exception carries both numbers as attributes. Drive
  it with a fake raw client that raises `MaxPayloadError`; do not require a
  broker.
- **Every other publish failure still raises plain `PublishError`.** The
  narrowing must not swallow the general case — that is the regression this
  shape invites.
- **All four public publish entry points** reach the new type, since they share
  one funnel. Cheap to assert, and it is the property that makes the funnel
  worth having.
- **`max_payload` is `None` before connect** and reflects the broker after
  (fake client, advertised value).
- **The outbound-overflow counter still fires** on the path it already covered.
- **An integration test at the real limit is not owed.** It would need a broker
  with a known `max_payload` and a megabyte of traffic to assert something the
  unit tests already pin; say so here so the next reader does not add one out of
  diligence.

## 7. Explicitly out of scope

- **Chunking a room frame.** `threetears.nats.pipe` already exists for payloads
  that must be split, and a broadcast room frame is not one: reassembly state on
  a transient, broadcast, reconnectable channel is the option §4 refuses.
- **Tuning the broker.** 1 MB is the untuned default and raising it is a
  deployment decision with its own memory costs. This task assumes nothing about
  it, which is exactly why R2 reads the value rather than hardcoding it.
- **Per-viewer frames on a broadcast channel.** §3 settles this the other way:
  the tier belongs to the room.
- **The consumers' catch sites.** They are in other repos and they are correct
  as designed; what they get from this task is a type to branch on.
