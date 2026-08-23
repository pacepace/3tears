# coll-task-01: Subscription Lifecycle on `CollectionRegistry` (3tears core)

## Objective

Give `CollectionRegistry.start_invalidation_listener` a real lifecycle: retain
the subscription handle, make a second call a no-op, and add a stop method.

This exists because `coll-task-02` needs the hub processes to subscribe **and
tear down**, and today teardown is not expressible. It also lets the agent SDK
delete a locally-invented guard rather than every consumer inventing its own.

---

## The gap, from source

`packages/core/src/threetears/core/collections/registry.py:307-401`:

```python
await nats_client.subscribe_typed(
    subject=Subjects.cache_invalidate(),
    message_type=CacheInvalidationMessage,
    cb=_on_invalidation,
)
```

`subscribe_typed` returns a `Subscription`
(`packages/nats/src/threetears/nats/client.py:1825-1834`). The return value is
**discarded**, the method returns `None`, and there is no
`stop_invalidation_listener`. `CollectionRegistry.clear()` (`registry.py:505`) is
documented "(for tests)" and unsubscribes nothing.

**The consequence in the SDK.** `subscribe_collection_invalidations`
(`14-eng-ai-bot-agents/src/aibots_agents/runtime/three_tier_stack.py:316-351`)
invents a re-entry guard locally and stores **the L2 client itself** as a
stand-in handle, because no real handle is available. Its `close()`
(`:353-387`) then tears down `_membership_sub`, `_assignment_sub` and
`_role_sub` -- and never the collection-invalidation subscription, which it
cannot.

So the only existing "pattern to copy" does not do the thing a consumer needs,
and copying it four more times would spread a workaround rather than a solution.

---

## Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| SUB-01 | `start_invalidation_listener` retains the `Subscription` returned by `subscribe_typed` | P0 |
| SUB-02 | `stop_invalidation_listener()` unsubscribes and clears the handle | P0 |
| SUB-03 | A second `start` call while one is live is a no-op -- not a second consumer | P0 |
| SUB-04 | `stop` with no live subscription is a no-op, not an error | P0 |
| SUB-05 | Start after stop works -- the registry is reusable, not one-shot | P1 |
| SUB-06 | The SDK's local guard and stand-in handle are **deleted**, not left alongside | P0 |

SUB-06 is not tidying. Leaving the SDK's guard in place next to the new one is
a parallel code path, which CLAUDE.md's no-shims rule forbids, and it is how the
two drift.

---

## Patterns to Follow

The registry already owns other subscription lifecycles worth mirroring rather
than inventing against. `packages/registry/src/threetears/registry/rbac_stack.py:140-210`
subscribes three ACL subjects and tears all three down in `close()`; that
teardown loop is the shape `stop_invalidation_listener` should be callable from.

Do not model the guard on the SDK's sentinel. Store the `Subscription`.

---

## Files to Modify

- `packages/core/src/threetears/core/collections/registry.py` -- retain the
  handle; add `stop_invalidation_listener`; guard re-entry.
- `packages/core/tests/test_registry.py` -- the lifecycle cases below. (This is
  the registry test module; it also holds every `set_l1_max_age` call.
  `test_cache_coherence.py` covers the eviction behaviour and should not need
  changing.)
- `14-eng-ai-bot-agents/src/aibots_agents/runtime/three_tier_stack.py` -- delete
  the local guard, call the new API, add the teardown to `close()`'s existing
  loop.

The SDK change lands in the same landing as the library change. It is a separate
repo, so see the sequence doc's landing mechanics -- this is the first shard that
crosses a repo boundary and it is the one that proves the version dance works
before the security shards depend on it.

---

## Implementation Notes

1. **The handle is a `Subscription`, not a bool and not the client.** A sentinel
   is what the SDK had to use and it is why teardown was impossible.
2. **`stop` must be safe during shutdown**, when the NATS connection may already
   be draining. Unsubscribing a subscription on a closed connection should not
   raise out of a `close()` path -- catch the specific transport error, not
   `Exception`.
3. **Do not add an L1 max age as part of this.** It bounds staleness by accident
   and would mask exactly the defect `coll-task-02` closes.

---

## Anti-patterns

- DO NOT leave the SDK's `_collection_invalidation_sub` sentinel in place beside the new handle. One owner.
- DO NOT make `stop` raise when nothing is subscribed. Callers run it from `finally` blocks.
- DO NOT swallow the unsubscribe error broadly. Name the transport exception.
- DO NOT change the invalidation message shape or subject here. This shard is lifecycle only; `CacheInvalidationMessage` and `threetears.cache.invalidate` are untouched.

---

## Success Criteria

- [x] `start` twice creates one subscription
- [x] `stop` unsubscribes; a second `stop` is a no-op
- [x] `start` → `stop` → `start` works
- [x] `stop` during a draining connection does not raise
- [x] The SDK's local guard is gone and its `close()` tears the subscription down
- [x] `./scripts/test.sh core` green
- [x] `./scripts/typecheck.sh` and `./scripts/lint.sh` clean

---

## Verification

```bash
cd 3tears
./scripts/test.sh core -v
./scripts/check-all.sh
cd ../14-eng-ai-bot-agents
uv run pytest tests/unit/ -q
```

The SDK half is modified by this shard and has no test script of its own -- run
its suite directly.

The core suite should be green before and after this shard. A draft
`test_l2_key_scoping.py` existed and was deleted -- it specified a superseded
two-tier design and failed to collect. `coll-task-03` creates it fresh. If you
find it present and erroring, it has been resurrected from somewhere; do not
work around it.

Note that `check-all.sh` deliberately excludes integration tests. Cross-pod
behaviour lives there, so before the PR:

```bash
./scripts/test-integration.sh
```

Behavioural: with two processes subscribed, stop one and confirm the other still
receives invalidations -- a teardown that takes the queue group down with it is
the failure this shard could plausibly introduce.

---

## Enforcement Test Suggestions

- [ ] **A process that calls `start_invalidation_listener` also calls `stop_invalidation_listener` in its shutdown path.** Suggested: an AST walker pairing the two per module. This is the gate that makes `coll-task-02`'s INV-02 checkable rather than aspirational, and it is the reason this shard is worth its own number.

  **Deliberately NOT built here -- carried to `coll-task-02` / `-06a`.** At the
  close of this shard the only 3tears caller of `start_invalidation_listener` is
  `registry/server.py`, whose teardown `coll-task-06a` adds; the hub callers
  `coll-task-02` adds do not exist yet. A pairing walker landed now would fail on
  its first and only subject, so it would have to ship pre-exempted -- an
  enforcement test that starts life switched off, which is the shape this repo
  has been burned by before. It lands with the first shard that gives it a
  population to check.
