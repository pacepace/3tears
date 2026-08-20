# epoch-task-04: Do not build a pending-invalidation queue

**Status:** CLOSED, decided 2026-08-18. The queue this shard specified will not be built.
Option 1 below (accept the bound from epoch-task-05) is the accepted resolution. This
document is kept as the record of why, so the queue is not re-proposed.
**Scope:** none to build. Two small corrections, listed at the end.
**Depends on:** nothing.

---

## What this shard originally proposed, and why it was wrong

The proposal: when NATS is unreachable, a write still reaches L3 but its invalidation is
lost, so peers serve that row stale forever. Queue the failed invalidations locally,
dedupe on `(table, entity_id)`, replay on reconnect.

**The premise is false.** `publish_invalidation` reaches `NatsClient.publish` →
`_publish_bytes` → core NATS `publish`. nats-py does not raise when disconnected: it
appends to a pending buffer and raises `OutboundBufferLimitError` only once that buffer is
full (`nats/aio/client.py:898-908`), then flushes the buffer to the wire on reconnect.
This repo sets that buffer explicitly to 4 MiB
(`packages/nats/src/threetears/nats/client.py:196`, `DEFAULT_PENDING_SIZE_BYTES`), which at
roughly 150 bytes per envelope is on the order of tens of thousands of buffered
invalidations replayed automatically.

So an ordinary outage already loses nothing. The queue would sit empty through exactly the
scenario it was designed for.

## The remaining states, and why a local queue reaches none of them

**Buffer overflow.** The only realistic `PublishError` here is `OutboundBufferLimitError`.
Three consecutive overflows with no intervening success flip `is_healthy` to `False`
(`client.py:1238-1250`, `_OUTBOUND_OVERFLOW_UNHEALTHY_THRESHOLD` at `:186`), which is wired
to the supervised restart from resilience-task-02. The pod is deliberately restarted at
almost exactly the moment the queue starts filling, and no L1 backend is disk-backed
(`cache/sqlite.py:104-113` uses `vfs=memdb`, `cache/duckdb.py:76` is `:memory:`), so the
queue dies with it. The queue's trigger condition is the system's kill condition.

**No NATS client.** `registry.py:384-385` returns before the try block when
`nats_client is None`. That is a single-pod dev or test run, not an outage, and enqueueing
there would accumulate an unbounded queue with nothing to drain it.

**Subscriber-side loss.** The invalidation subject is core NATS pub/sub, which is
at-most-once. A pod whose *subscription* is partitioned while its peers stay healthy misses
every invalidation published in that window and knows nothing failed. No publisher-side
queue can reach this, and it is the larger half of the failure surface.

## What actually closes the gap

Two options, and they are not equivalent:

1. **Accept the bound from epoch-task-05.** L1 max-age bounds staleness from any cause
   including the subscriber-side case, which is the only mechanism in this series that
   touches it. Costs nothing new. Weak in the sense that the bound is the max-age, not
   zero.
2. **Move invalidations onto JetStream.** `packages/nats/src/threetears/nats/oplog.py` is
   an existing sanctioned JetStream append-only log with terminating `replay(from_seq=...)`,
   `Nats-Msg-Id` dedup and a CAS fence. A per-pod durable consumer replays from its last
   acked sequence on reconnect, survives pod restart, and fixes the subscriber-side case
   too. Real costs: `OpLog.open` is keyed `(repo, branch)` and would need generalising,
   per-pod durable consumer state, stream retention sizing, and an ack wait on a path that
   is currently fire-and-forget (`_propagate_write` at `collections/base.py:813-815`).

**Recommendation: take option 1 now.** Ship epoch-task-05, observe whether the staleness
bound is actually a problem in practice, and file option 2 as its own design if it is.
Option 2 is a transport change to the coherence path and should not ride in as a
side effect of a bug that turned out not to exist.

## Corrections worth making while this is fresh

Both are in `packages/core/src/threetears/core/collections/registry.py`, and neither
depends on anything above.

- The `except PublishError` handler at `:397-409` justifies swallowing with "a read will
  pull a stale-but-still-correct row from L3". That does not hold: peers keep serving from
  their own L1 and never re-read. The swallow is still correct given the pending buffer,
  but the stated reason is not.
- The `ValidationError` handler at `:410-426` claims failures "surface as a real failure in
  CI because tests assert on this counter". There is no counter, and nothing in
  `packages/core/tests/test_cache_coherence.py` asserts on one.

## Rejected alternatives, kept because they are still worth not re-deriving

Anyone revisiting durable invalidation delivery will consider reconstructing the stale set
from L3 instead. Three ways, all rejected:

- **The epoch itself.** A row could carry the epoch it was written under, and a pod could
  ask L3 for rows above its last-seen. It fails on exactly the window that matters: the
  epoch only advances when NATS bumps it, and NATS is down, so writes during the outage
  carry the pre-outage epoch and are invisible.
- **`xmin` / `pg_current_xact_id()`.** Commit order from L3 with no clock. Not available:
  YSQL rejects system columns ("System column with id -3 is not supported yet") because
  DocDB orders by hybrid logical clock rather than a global xid sequence. This repo's only
  YSQL-aware artefact is the `migration_yugabyte_safety` check at
  `tests/enforcement/test_codebase_conventions.py:57`.
- **`date_updated` with a margin.** Works where it exists, and the margin makes skew
  irrelevant. But it does not exist everywhere: the L1+L2-only collections
  (`packages/registry/src/threetears/registry/heartbeat_collection.py:115`,
  `packages/channels/src/threetears/channels/presence/collection.py:119`) have no L3 table
  to query at all.
