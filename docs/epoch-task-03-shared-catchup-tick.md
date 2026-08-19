# epoch-task-03: A shared catch-up tick body, driven by the consumer

**Status:** BUILT, not shipped -- no PR, not merged, not released. Landed across
`3335a97f`..`ab93534b`; the branch pointer that carried them may move or go away. Reshaped after review; see "Why the tick does not live in core".
**Scope:** `3tears-epoch` (new tick body), `3tears-mcp` (`auth.py`, its loop body shrinks
to a call).
**Depends on:** epoch-task-02, which must not release without this. The recovery action a
detected reset triggers is the consumer's `on_reset` callback, defined in epoch-task-02.

---

## Objective

Give the epoch catch-up pass one implementation that every consumer calls, so bucket-reset
detection is uniform instead of depending on whether a given consumer hand-rolled a loop.

## Why

`EpochListener.catch_up` is documented as "a public hook for periodic catch-up ticks",
and exactly one caller in the codebase drives it: `packages/mcp/src/threetears/mcp/auth.py:537-563`.
Every other consumer of `EpochListener` has no catch-up at all, so only pods that happen
to be publishing notice anything. After epoch-task-02, a pod with no catch-up never
detects a recreated bucket, which is precisely the failure this series exists to fix.

Note the narrow claim: no framework component runs an *epoch* catch-up. The framework
does run periodic loops elsewhere (`packages/channels/src/threetears/channels/presence/sweeper.py:142`,
`packages/registry/src/threetears/registry/health.py:301`). `packages/mcp/src/threetears/mcp/rbac.py`
and `packages/channels/src/threetears/channels/websocket.py` reference `EpochListener` in
docstrings only and drive no loop; `auth.py` is genuinely the sole driver.

## Why the tick does not live in core, and does not own a task

An earlier draft put a background tick task on `CollectionRegistry`. That is wrong three
times over, and each reason is independently sufficient:

1. **It inverts the package dependency.** `packages/epoch/pyproject.toml:24` declares
   `3tears>=0.26.0,<0.27.0`; `packages/core/pyproject.toml:24-36` declares no epoch
   dependency and no extra that adds one. Epoch imports core, never the reverse. Under
   the lockstep-bounds rule in `CLAUDE.md`, adding the back-edge makes the family
   unresolvable rather than merely wrong.
2. **It contradicts the house position on cadence.**
   `packages/scheduled-jobs/src/threetears/scheduled_jobs/tick.py:1-33` states it
   directly: "Pure-async, one tick per call. No internal polling. The consumer's
   scheduler drives cadence." A framework-owned loop here would be a new precedent set
   by accident.
3. **`CollectionRegistry` has no lifecycle to hang a task on.** Its only teardown is
   `clear()` at `registry.py:428`, documented "(for tests)". And it is constructed
   per-request in production, at `packages/channels/src/threetears/channels/webhook.py:321`
   and `packages/agent/wake/src/threetears/agent/wake/dispatch.py:350` among eight sites,
   so a task started in the registry would multiply per request.

## Shape

A pure-async one-pass function in `3tears-epoch`, matching `scheduled_jobs/tick.py`:

- It takes an `EpochListener`, the set of `(subject, on_bump)` pairs to check, and runs
  one `catch_up` pass over them. No sleeping, no task, no state.
- Errors from one subject do not abort the pass; the pass is best-effort by nature,
  because the next one will retry.
- The consumer schedules it. `auth.py`'s existing loop stays and becomes the caller,
  with its body reduced to one call.

The value here is a single correct pass body, not a new scheduler.

## Reuse (verified present, use these rather than writing new)

- Any consumer that does spawn a task uses `spawn_background` at
  `packages/observe/src/threetears/observe/background.py:34`. Core and epoch both depend
  on `3tears-observe`, so it is available today, and none of the four existing loops use
  it.
- Jitter already exists twice, with the full-jitter rationale already argued:
  `_full_jitter_backoff` at `packages/nats/src/threetears/nats/client.py:283-330`, with a
  non-zero floor at `:370` to stop a near-zero draw hot-spinning. Do not restate the
  argument; call the helper, and keep the floor.
- The `start/stop/_loop` shell is written three times already
  (`presence/sweeper.py:112-165`, `registry/health.py:198,301`, `mcp/auth.py:537-563`),
  and `sweeper.py:161` says it mirrors the second. This task does not add a fourth. If a
  later task wants a framework loop, lifting that shell into one primitive is the work,
  and it belongs in `3tears-observe` beside `spawn_background`, not here.

## Contract with epoch-task-02

The tick's only obligation is to call `catch_up` on every registered subject at the
consumer's interval. Identity detection and local reset happen inside the listener and
are epoch-task-02's business. The tick neither knows nor cares that a reset occurred.

## What this does NOT do

It does not give consumers catch-up for free. A consumer that schedules nothing still
gets nothing, and after epoch-task-02 that consumer will not detect a recreated bucket.
Closing that gap means either a framework-owned loop (rejected above) or an explicit
wiring requirement on each consumer. **Record which, here, before building**, because
leaving it implicit is how `EpochListener` ended up with one driver in the first place.

## Acceptance

- One tick body in `3tears-epoch`, pure-async, one pass per call, no internal sleep.
- It carries multiple `(subject, callback)` pairs; a single-subject API repeats the gap
  it is meant to close.
- A failing subject does not prevent the others in the same pass from running.
- `auth.py:537-563` keeps its loop and its `stop()` contract, with the body replaced by a
  call. Its interval default of `60.0` (`auth.py:449`) and the
  `catchup_interval_seconds=3600.0` knob used by
  `packages/mcp/tests/integration/test_multi_pod_rbac.py:177-194` both survive unchanged.
- No new `create_task` outside `spawn_background`, and no fourth copy of the loop shell.
