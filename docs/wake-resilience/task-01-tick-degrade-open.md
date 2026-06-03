# wake-task-01: Wake tick degrades open when the cross-pod lock is unavailable

## Objective

A NATS JetStream wipe (single-node restart on ephemeral storage) must not silence the
wake heartbeat. Today it does: `wake_tick_job` only catches `LockHeld`, so when the
distributed lock's bucket vanishes the resulting `KvError` propagates and the entire tick
body never runs — for hours, until the API restarts. Fix: degrade open.

## Root cause

`threetears/agent/wake/tick.py:131-135`:

```python
try:
    async with nats_distributed_lock(nats_client, _LOCK_KEY):
        await _run_tick_body(pool, dispatch_callback)
except LockHeld:
    log.debug("wake_tick: lock held by another pod, skipping")
```

The cross-pod lock is an **optimization**, not a correctness requirement. Per-schedule
mutual exclusion is already the Postgres optimistic-CAS in
`WakeScheduleCollection.claim_and_reschedule` (tick.py:224, `expected_next_fire` anchor):
two pods both reach a due schedule, exactly one CAS wins, the loser logs `SKIPPED_BUSY`.
So `_run_tick_body` needs **zero NATS** to be correct.

`nats_distributed_lock` deliberately raises `KvError` (lock infra failed — bucket/stream
gone, NATS unreachable) as a type *distinct* from `LockHeld` (another pod holds it) —
see `distributed_lock.py:66-73`. The `except LockHeld` does not catch `KvError`, so an
**optional optimization's infrastructure failure kills a tick whose correctness never
depended on it.**

## The fix (prescriptive)

Catch `KvError` separately and run the body anyway:

```python
from threetears.nats import LockHeld, nats_distributed_lock  # noqa: PLC0415
from threetears.nats.errors import KvError  # noqa: PLC0415

try:
    async with nats_distributed_lock(nats_client, _LOCK_KEY):
        await _run_tick_body(pool, dispatch_callback)
except LockHeld:
    log.debug("wake_tick: lock held by another pod, skipping")
except KvError as exc:
    # Lock INFRASTRUCTURE failed (bucket/stream gone, NATS unreachable) — distinct
    # from LockHeld. The per-schedule Postgres CAS in claim_and_reschedule is the
    # real mutual exclusion, so the cross-pod lock is a redundant-work optimization,
    # not a correctness requirement. Degrade open: run the tick body without the
    # lock rather than silently dropping every tick until a process restart.
    log.warning(
        "wake_tick: cross-pod lock unavailable; proceeding without it (CAS still guards fires)",
        extra={"extra_data": {"error_type": type(exc).__name__, "error": str(exc)}},
    )
    await _run_tick_body(pool, dispatch_callback)
```

Worst case under a NATS outage: every pod runs the due-scan each tick and contends on the
CAS — already the handled `SKIPPED_BUSY` path. No double-fires, no data loss, heartbeat
survives.

## TDD

Write the test first. `packages/agent/wake/tests/unit/test_tick.py`:

- **Regression test (the bug):** patch the locally-imported `nats_distributed_lock` so its
  `__aenter__` raises `KvError("no response from stream")`. Assert `_run_tick_body` STILL
  ran — e.g. `list_due_for_tick` was queried and the `dispatch_callback` fired for a due
  schedule. Before the fix this test fails (body never runs); after, it passes.
- **Held lock unchanged:** a `LockHeld` still skips the body (existing behavior preserved).
- **Healthy lock unchanged:** body runs once inside the lock.

## Anti-patterns

- DO NOT catch bare `Exception` here — that would also swallow real bugs in the tick body.
  Catch `KvError` specifically (lock-infra) and let everything else propagate.
- DO NOT remove the lock or the `LockHeld` path — the optimization is still worth having
  when NATS is healthy.
- DO NOT add a new schedule/claim object — the durable three-tier scheduler
  (`WakeScheduleCollection` / `WakeFireCollection` / `claim_and_reschedule` / `wake_fires`)
  already exists and already survives shutdown. This shard changes ~6 lines.

## Success criteria

- [ ] `test_tick.py` regression test: a `KvError`-raising lock does not suppress the tick body.
- [ ] `LockHeld` still skips; healthy lock still runs once.
- [ ] Full wake suite green: `pytest packages/agent/wake -q`
- [ ] Ruff clean: `ruff check . && ruff format . --check`

## Verification

```bash
cd /Users/pace/crypt/pub/dev-wsl/vscode/3tears/3tears
pytest packages/agent/wake/tests/unit/test_tick.py -q
pytest packages/agent/wake -q
ruff check . && ruff format . --check
```

## Out of scope (separate follow-up)

`threetears.nats` KV self-heal (recreate a vanished bucket on `KvError`, flush the
`NatsClient._buckets` cache on reconnect) — that *restores the lock optimization* and helps
the cache/transport paths, but it is NOT what keeps the heartbeat alive. Track it as its own
shard so this correctness fix ships small and standalone.
