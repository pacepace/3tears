# agent-wake-02: Tick engine + `nats_distributed_lock` primitive

## 2026-05-19 revision deltas (apply BEFORE implementing)

Canonical source: `<metallm>/docs/long_running/PLACEMENT.md`.

**Behavioral additions:**
- **Missed-fire policy handling.** The tick body honors `agent_wake_schedules.missed_fire_policy` (`'coalesce'` default | `'catch_up'`). When `next_fire_at <= now` and the policy is `'coalesce'`, fire ONCE and recompute `next_fire_at` forward. When `'catch_up'`, fire once per missed tick. PLACEMENT §1.7.
- **Drift recording.** Every fire writes `wake_fires.actual_fired_at = now` AND `wake_fires.scheduled_fire_at = schedule.next_fire_at`. No drift-skip rule v1. PLACEMENT §1.8.
- **Per-conv active-schedule cap enforced at create write (shard 04) AND as defense-in-depth at tick.** Default 10 per conversation. PLACEMENT §1.9.

**`_compute_next_fire_at` signature** gains awareness of `missed_fire_policy` — pass it through and decide between coalesce-forward vs. per-missed-tick.

## Objective

Two cohesive deliverables in one shard because both lift from the
same metallm primitive (`api/src/services/scheduler.py`) and the tick
consumes the lock directly:

1. **Lift `scheduler_lock` to `3tears-nats`** as the new
   `nats_distributed_lock(client, key, ttl, heartbeat)` async context
   manager. metallm's existing `scheduler_lock` becomes a one-line
   re-export.
2. **Land the tick engine** in `3tears-agent-wake`: pure-async
   `wake_tick_job(pool, nats_client, dispatch_callback)` that polls
   `agent_wake_schedules` for due rows under the lock, computes
   `next_fire_at` per schedule_type, claims via optimistic-CAS,
   inserts the initial `wake_fires` row, and invokes the dispatch
   callback. Plus the pure helper `_compute_next_fire_at`.

Consumers (metallm) register the tick body as their own APScheduler
job; the platform doesn't impose an APScheduler dependency on
consumers.

---

## Requirements

### Part A — `nats_distributed_lock` in `3tears-nats`

| ID | Requirement | Priority |
|----|-------------|----------|
| LOCK-01 | New module `packages/nats/src/threetears/nats/distributed_lock.py` exporting `nats_distributed_lock(client: NatsClient, key: str, *, bucket_name: str = "scheduler-locks", ttl: timedelta = timedelta(seconds=60), heartbeat: timedelta = timedelta(seconds=20)) -> AsyncIterator[None]` async context manager. | P0 |
| LOCK-02 | `LockHeld` exception raised when the lock is already held by another holder. Distinct from `KvError` (transport / bucket failures). | P0 |
| LOCK-03 | `bucket_name` defaults to `"scheduler-locks"` and is automatically namespace-prefixed via the existing `NatsClient.kv_bucket(name=...)` path. | P0 |
| LOCK-04 | Heartbeat task refreshes the KV entry every `heartbeat` seconds. On normal exit OR exception, heartbeat is cancelled cleanly and the key is deleted. On pod crash, the TTL expires the key within `ttl` seconds. | P0 |
| LOCK-05 | `nats_distributed_lock(client=None, key=...)` yields immediately without acquiring — safe for single-pod dev environments. Matches the existing metallm `scheduler_lock(None, ...)` behavior. | P0 |
| LOCK-06 | Acquired via `bucket.create(key, value=b"1")` — atomic put-if-absent. Returns False when the key already exists; in that case raise `LockHeld`. | P0 |
| LOCK-07 | Unit tests against a fake `NatsClient`: acquire-and-release, lock-held, heartbeat-refreshes-key, cancellation-during-body-cleans-up. | P0 |
| LOCK-08 | Documented in the package `README.md` under the existing "NatsKvBucket" section. Add a "Distributed locks" sub-section with the canonical usage shape. | P0 |

### Part B — Tick engine in `3tears-agent-wake`

| ID | Requirement | Priority |
|----|-------------|----------|
| TICK-01 | New module `packages/agent/wake/src/threetears/agent/wake/tick.py` exporting `wake_tick_job(pool, nats_client, dispatch_callback) -> None`. Acquires `nats_distributed_lock(nats_client, "agent_wake_tick")`; on `LockHeld`, returns at debug. | P0 |
| TICK-02 | Per tick: SELECT all schedules with `status='active' AND next_fire_at <= now()` ordered by `next_fire_at ASC`, then for each: compute `next_fire_at` → claim via optimistic-CAS → INSERT initial `wake_fires` row → invoke `await dispatch_callback(trigger, fire_id, pool)`. | P0 |
| TICK-03 | Pure helper `_compute_next_fire_at(schedule_type: str, schedule_config: dict, last_fired_at: datetime, now: datetime) -> datetime | None` covering every documented schedule_type. Returns `None` for `one_shot_at` and `relative_delay` after fire (caller marks `expired`). | P0 |
| TICK-04 | TZ-aware computation for `daily_at` / `random_within_window` via `zoneinfo.ZoneInfo(config['tz'])` (stdlib, no new dep). DST transitions handled correctly (verify via test). | P0 |
| TICK-05 | `random_within_window` overnight-window wrap: when `start_hour > end_hour`, pick uniform time in wrapped window. When `start_hour == end_hour`, fail upstream (validation lives in shard 04). | P0 |
| TICK-06 | `WakeScheduleCollection.claim_and_reschedule(schedule_id, expected_next_fire, computed_next_fire, new_status, now)` method on the collection from shard 01. Optimistic-CAS UPDATE: returns `True` on claim, `False` when another tick beat us. | P0 |
| TICK-07 | The tick body does NOT execute pre-checks — that's the dispatch handler's responsibility (shard 03). Tick only claims, reschedules, and invokes the callback. Keeps tick body fast. | P0 |
| TICK-08 | Tick observability: structured INFO log at start (`"wake_tick: N due"`), structured INFO log per fire, structured ERROR with stack-trace on callback exception, completion summary log. Use `threetears.observe.get_logger(__name__)`. | P0 |
| TICK-09 | Failure isolation: per-schedule dispatch wrapped in `try/except Exception`. Log + record a failed fire (UPDATE the `wake_fires` row to `status='failed', error=str(exc)`); continue to next row. | P0 |
| TICK-10 | When the dispatch callback returns `status='skipped_gate'`, the tick still advances `next_fire_at` per the schedule's normal cadence — a gate-skip does NOT pause the schedule. | P0 |
| TICK-11 | The tick body is consumer-callable as `await wake_tick_job(pool, nats_client, dispatch_callback)`. The consumer (metallm) registers it as their APScheduler IntervalTrigger job; the platform does not own APScheduler. | P0 |

---

## Design Context

### Part A — Why this lift is overdue

metallm has run `scheduler_lock` in production for the backup job for
months. It's a NATS KV TTL lock with heartbeat — the canonical
distributed-lock shape, not metallm-specific. The tick adds a second
consumer in the same product; lifting it to `3tears-nats` ahead of
adding the wake tick is the right ordering ("put it in the right place
to start with").

After this shard lands, metallm's
`api/src/services/scheduler.py:scheduler_lock` becomes:

```python
from threetears.nats.distributed_lock import nats_distributed_lock, LockHeld

# Old internal symbols are kept for backward compatibility during the
# bump; deprecation warning + scheduled removal in the next release.
SchedulerLockHeld = LockHeld
scheduler_lock = nats_distributed_lock
```

The `BUCKET_SCHEDULER_LOCKS = "scheduler-locks"` bucket name stays the
same so existing prod state continues to work.

### Part B — Why one tick job, not one-per-schedule

From the metallm v1 design notes:

> DO NOT register one APScheduler job per `agent_wake_schedules` row.
> That couples schedule-storage to APScheduler's in-memory state and
> creates a synchronization problem on tool-driven schedule creation.
> One tick job that polls is simpler and restart-safe.

Preserved exactly. The tick polls; APScheduler holds one job per pod;
the lock serializes pods.

### Part B — Why `_compute_next_fire_at` is pure

Pure function = unit-testable without DB or APScheduler infrastructure.
DST transitions, end-of-month cron, overnight windows — all
deterministic with pinned `now`. The reschedule branches stay clean.

---

## API specifications

### `nats_distributed_lock` (Part A)

```python
# packages/nats/src/threetears/nats/distributed_lock.py
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Final

from threetears.observe import get_logger

from threetears.nats.client import NatsClient
from threetears.nats.errors import KvError

__all__ = ["LockHeld", "nats_distributed_lock"]

log = get_logger(__name__)


class LockHeld(Exception):
    """Raised by nats_distributed_lock when another holder owns the lock."""


_DEFAULT_TTL: Final[timedelta] = timedelta(seconds=60)
_DEFAULT_HEARTBEAT: Final[timedelta] = timedelta(seconds=20)
_DEFAULT_BUCKET: Final[str] = "scheduler-locks"


@asynccontextmanager
async def nats_distributed_lock(
    client: NatsClient | None,
    key: str,
    *,
    bucket_name: str = _DEFAULT_BUCKET,
    ttl: timedelta = _DEFAULT_TTL,
    heartbeat: timedelta = _DEFAULT_HEARTBEAT,
) -> AsyncIterator[None]:
    """Acquire a TTL-based distributed NATS lock.

    Usage::

        try:
            async with nats_distributed_lock(nats, "my_job"):
                ... # job body
        except LockHeld:
            return  # another holder owns this key

    When client is None, yields immediately without acquiring.
    """
    if client is None:
        yield
        return
    if heartbeat >= ttl:
        msg = f"heartbeat {heartbeat} must be less than ttl {ttl}"
        raise ValueError(msg)

    bucket = await client.kv_bucket(name=bucket_name)
    acquired = await bucket.create(key=key, value=b"1")
    if not acquired:
        raise LockHeld(f"lock already held: {key}")

    async def _heartbeat() -> None:
        try:
            while True:
                await asyncio.sleep(heartbeat.total_seconds())
                await bucket.put(key=key, value=b"1")
        except asyncio.CancelledError:
            pass

    hb_task = asyncio.create_task(_heartbeat())
    try:
        yield
    finally:
        hb_task.cancel()
        await asyncio.gather(hb_task, return_exceptions=True)
        try:
            await bucket.delete(key=key)
        except KvError:
            log.debug("lock cleanup: key already deleted: %s", key)
```

### Tick body (Part B)

```python
# packages/agent/wake/src/threetears/agent/wake/tick.py
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Final
from uuid import UUID

from asyncpg import Pool
from uuid_utils import uuid7

from threetears.nats import NatsClient
from threetears.nats.distributed_lock import LockHeld, nats_distributed_lock
from threetears.observe import get_logger

from threetears.agent.wake.collections import WakeScheduleCollection, WakeFireCollection
from threetears.agent.wake.types import WakeTrigger, WakeDispatchResult

DispatchCallback = Callable[[WakeTrigger, UUID, Pool], Awaitable[WakeDispatchResult]]

_LOCK_KEY: Final[str] = "agent_wake_tick"

log = get_logger(__name__)


async def wake_tick_job(
    pool: Pool,
    nats_client: NatsClient | None,
    dispatch_callback: DispatchCallback,
) -> None:
    """Run one tick of the agent-wake scheduler.

    Acquires the cross-pod tick lock, polls due schedules, and dispatches each
    via dispatch_callback. Returns silently if the lock is held by another pod.
    """
    try:
        async with nats_distributed_lock(nats_client, _LOCK_KEY):
            await _run_tick_body(pool, dispatch_callback)
    except LockHeld:
        log.debug("wake_tick: lock held by another pod, skipping")


async def _run_tick_body(pool: Pool, dispatch_callback: DispatchCallback) -> None:
    schedules = await WakeScheduleCollection(pool).list_due_for_fire(now=datetime.now(UTC))
    log.info("wake_tick: %d due", len(schedules), extra={"extra_data": {"due_count": len(schedules)}})
    fires = WakeFireCollection(pool)
    for schedule in schedules:
        await _dispatch_one(pool, schedule, fires, dispatch_callback)


async def _dispatch_one(pool, schedule, fires, dispatch_callback) -> None:
    now = datetime.now(UTC)
    new_fire_at = _compute_next_fire_at(schedule.schedule_type, schedule.schedule_config, now, now)
    new_status = "active" if new_fire_at is not None else "expired"

    claimed = await WakeScheduleCollection(pool).claim_and_reschedule(
        schedule_id=schedule.schedule_id,
        expected_next_fire=schedule.next_fire_at,
        computed_next_fire=new_fire_at,
        new_status=new_status,
        now=now,
    )
    if not claimed:
        log.debug("wake_tick: claim lost on schedule", extra={"extra_data": {"schedule_id": str(schedule.schedule_id)}})
        return

    fire_id = uuid7()
    trigger = WakeTrigger(
        schedule_id=schedule.schedule_id,
        user_id=schedule.user_id,
        conversation_id=schedule.conversation_id,
        fire_source="scheduled_tick",
        execution_mode=schedule.execution_mode,
        schedule_type=schedule.schedule_type,
        task_prompt=schedule.task_prompt,
        schedule_name=schedule.name,
        fired_at=now,
        no_agent=schedule.no_agent,
        pre_check_type=schedule.pre_check_type,
        pre_check_config=schedule.pre_check_config,
        context_from_schedule_id=schedule.context_from_schedule_id,
        delivery_target=schedule.delivery_target,
        delivery_config=schedule.delivery_config,
        attached_skill_ids=await _load_attached_skill_ids(pool, schedule.schedule_id),
    )
    await fires.create_dispatching(
        fire_id=fire_id,
        schedule_id=schedule.schedule_id,
        webhook_subscription_id=None,
        conversation_id=schedule.conversation_id,
        fired_at=now,
        fire_source="scheduled_tick",
        execution_mode=schedule.execution_mode,
        delivery_target_resolved=schedule.delivery_target,
    )

    try:
        await dispatch_callback(trigger, fire_id, pool)
    except Exception as exc:
        log.exception("wake_tick: dispatch failed for schedule %s", schedule.schedule_id)
        await fires.finalize_failed(fire_id, error=str(exc), duration_ms=None)
```

### `_compute_next_fire_at` (Part B)

Pure. No DB, no I/O. Branches on `schedule_type`:

| schedule_type | Computation |
|---|---|
| `daily_at` | `next_in_tz = today_at_hh_mm_in_tz` if `next_in_tz > now` else `tomorrow_at_hh_mm_in_tz`. Resolved in `zoneinfo.ZoneInfo(config['tz'])`. |
| `every_n_hours` | `now + timedelta(hours=config['n'])`. |
| `random_within_window` | Pick random `H:M` in the window (overnight-wrapped if `start > end`). |
| `one_shot_at` | If `config['fire_at_iso'] > now`: return that; if already passed and this is the first compute: return same value (immediate fire). After fire: return `None`. |
| `cron` | `CronTrigger.from_crontab(config['expr']).get_next_fire_time(previous_fire_time=last_fired_at, now=now)`. Note: this is the only branch that uses APScheduler — and only as a utility, not as scheduler infrastructure. The function imports it locally. |
| `relative_delay` | `now + timedelta(parse_delay(config['delay']))`. After fire: return `None`. |

---

## Patterns to Follow

- Distributed-lock body shape: existing metallm `scheduler_lock` body (the source of truth for this lift) at `api/src/services/scheduler.py:48-93`.
- TableSchema enrichment for the `agent_wake_schedules` declaration: v0.8.0 patterns.
- Tick body shape: the original metallm shard-02 design notes — preserve the optimistic-CAS UPDATE pattern.
- Logger context: `threetears.observe.get_logger(__name__)`. Use `extra={"extra_data": {...}}` for structured fields.

---

## Files to Create

### Part A — `3tears-nats`

- `packages/nats/src/threetears/nats/distributed_lock.py` — `LockHeld`, `nats_distributed_lock`.
- `packages/nats/tests/unit/test_distributed_lock.py` — acquire-release, lock-held, heartbeat, cancellation tests against a fake bucket.
- `packages/nats/src/threetears/nats/__init__.py` — export `LockHeld`, `nats_distributed_lock`.

### Part B — `3tears-agent-wake`

- `packages/agent/wake/src/threetears/agent/wake/tick.py` — `wake_tick_job` + `_dispatch_one` + `_load_attached_skill_ids`.
- `packages/agent/wake/src/threetears/agent/wake/reschedule.py` — `_compute_next_fire_at` pure helper (split out for cleaner unit testing).
- Add `claim_and_reschedule` method on `WakeScheduleCollection` (defined in shard 01; this shard's modification).
- Add `create_dispatching`, `finalize_*` methods on `WakeFireCollection` (defined in shard 01; this shard's modification — fire lifecycle methods).
- `packages/agent/wake/tests/unit/test_reschedule.py` — table-driven tests for `_compute_next_fire_at` across every schedule_type, including DST transitions and overnight-window wrap.
- `packages/agent/wake/tests/integration/test_wake_tick_loop.py` — seed 3 schedules (one due, one not due, one paused), run tick body once with a stub `dispatch_callback`, assert exactly the due one was dispatched + the row's `next_fire_at` advanced + `wake_fires` got one row.

---

## Implementation Notes

1. **`heartbeat < ttl` invariant.** Enforce in the lock entry — raise `ValueError` if violated. Existing metallm hard-codes `60` / `20`; we make these parameters but keep the same defaults.

2. **Bucket name namespacing.** `NatsClient.kv_bucket(name=...)` already auto-prefixes with the client's namespace. So `"scheduler-locks"` becomes `"<ns>-scheduler-locks"`. metallm bumping to the platform version: their existing bucket retains its name because the client's namespace doesn't change.

3. **`asyncio.gather(hb_task, return_exceptions=True)` in finally.** Matches existing metallm pattern. Ensures heartbeat is reaped cleanly even on body exception.

4. **Optimistic-CAS shape on `claim_and_reschedule`.**

   ```python
   async def claim_and_reschedule(
       self,
       *,
       schedule_id: UUID,
       expected_next_fire: datetime,
       computed_next_fire: datetime | None,
       new_status: str,
       now: datetime,
   ) -> bool:
       claimed = await self._pool.fetchval(
           """
           UPDATE agent_wake_schedules
           SET next_fire_at = $1, last_fired_at = $2, date_updated = $2, status = $3
           WHERE schedule_id = $4 AND next_fire_at = $5
           RETURNING schedule_id
           """,
           computed_next_fire,
           now,
           new_status,
           schedule_id,
           expected_next_fire,
       )
       return claimed is not None
   ```

5. **Per-schedule wrapping in `try/except Exception`.** Wake feature must be robust to one bad row poisoning the tick.

6. **`asyncio.create_task` from inside the dispatch callback.** The tick awaits `dispatch_callback`, which awaits `inject_conversation_event` (in metallm's callback), which internally `asyncio.create_task`'s the LLM call. So the tick returns once the message is inserted, not once the LLM responds. Document this in `wake_tick_job`'s docstring.

7. **`NatsClient is None` graceful degradation.** Inherited from existing metallm `scheduler_lock`. Single-pod dev environments without NATS still work.

8. **Test fixture for time.** Use `freezegun` (already a dev dep in metallm; check `3tears/pyproject.toml` for the same). The reschedule tests are time-sensitive; pinning `now` is the only way to make them deterministic.

9. **DST transition test cases.** Specifically:
   - `daily_at 09:00 America/Los_Angeles` across spring-forward day → advances by 23h not 24h.
   - `daily_at 09:00 America/Los_Angeles` across fall-back day → advances by 25h not 24h.
   - zoneinfo handles these naturally; the test verifies.

10. **`cron` branch imports APScheduler locally.** The function does `from apscheduler.triggers.cron import CronTrigger` inside the cron branch only. Keeps APScheduler an optional-utility dep, not a platform requirement on `3tears-agent-wake`.

---

## Anti-patterns

- DO NOT acquire the lock outside the `async with` block. The heartbeat is fire-and-forget and must be cancelled by the contextmanager's exit.
- DO NOT register one APScheduler job per schedule row. Documented in TICK-01.
- DO NOT process schedules in parallel within a single tick (`asyncio.gather` across rows). Sequential keeps the `wake_fires` inserts ordered without write-contention.
- DO NOT call the dispatch callback from inside the lock acquisition block at module level. Callback is injected for testability.
- DO NOT compute `next_fire_at` using `apscheduler.triggers.IntervalTrigger.get_next_fire_time` for the typed shortcuts — pure helper only. APScheduler's helpers are stateful.
- DO NOT skip the optimistic-CAS UPDATE. Two pods that briefly disagree about NATS lock ownership could double-fire a schedule otherwise.
- DO NOT remove `scheduler_lock` from metallm in the same release. Keep it as a re-export deprecated-shim for one minor; remove in the next.

---

## Success Criteria

### Part A

- [ ] `nats_distributed_lock` exported from `3tears-nats`.
- [ ] `LockHeld` exception type distinct from `KvError`.
- [ ] Unit tests pass: acquire-release, lock-held, heartbeat refreshes, cancellation cleans up.
- [ ] metallm dry-run: `scheduler_lock` becomes a re-export of `nats_distributed_lock`; existing backup-job test suite passes.

### Part B

- [ ] `_compute_next_fire_at` returns correct values for every schedule_type across the reschedule test table.
- [ ] DST transition tests pass.
- [ ] Integration test: tick runs, dispatches the due schedule, records a `wake_fires` row, advances `next_fire_at`, leaves the paused + not-due schedules untouched.
- [ ] Lock-held simulation test: two concurrent tick bodies, only one acquires the lock and processes the schedules.
- [ ] Failure isolation test: one schedule's callback raises, the next schedule still dispatches.
- [ ] `./scripts/check-all.sh` clean across `packages/agent/wake/` and `packages/nats/`.

---

## Verification

```bash
cd /Users/pace/crypt/pub/dev-wsl/vscode/3tears/3tears
uv run --directory packages/nats pytest tests/unit/test_distributed_lock.py -v
uv run --directory packages/agent/wake pytest tests/ -v
./scripts/check-all.sh
```
