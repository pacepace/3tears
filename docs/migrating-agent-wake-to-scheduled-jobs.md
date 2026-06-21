# Migrating agent-wake onto the scheduled-jobs core (S-2)

`3tears-agent-wake`'s tick engine now **delegates** its cross-pod tick pump
(lock acquire / degrade-open, due-scan, optimistic-CAS claim, per-fire
isolation, drift) and its reschedule math to the generic `3tears-scheduled-jobs`
core. `threetears.agent.wake.tick` is a thin adapter over
`threetears.scheduled_jobs.scheduled_tick_job`.

**The good news:** the wake-facing contract is unchanged. `wake_tick_job(pool,
nats_client, dispatch_callback)`, the wake-shaped `DispatchCallback`,
`WakeTrigger`, `WakeDispatchResult`, `FireStatus`, the schedule/fire schema, and
the webhook / `[SILENT]` handling all stay put. The cross-pod lock key stays
`"agent_wake_tick"`. **For most consumers this is a no-op upgrade** — except one
deleted module and some renamed metrics.

## Breaking — change this one import (mechanical, no behavior change)

`threetears.agent.wake.reschedule` is **gone**. Its private `_compute_next_fire_at`
is now the public `threetears.scheduled_jobs.compute_next_fire_at` — **identical
positional signature** (`schedule_type, schedule_config, missed_fire_policy,
last_fired_at=, now=`), identical behavior.

```python
# before
from threetears.agent.wake.reschedule import _compute_next_fire_at
next_fire = _compute_next_fire_at(schedule_type, config, policy, last_fired_at=..., now=...)

# after
from threetears.scheduled_jobs import compute_next_fire_at
next_fire = compute_next_fire_at(schedule_type, config, policy, last_fired_at=..., now=...)
```

Find every site (in metallm this is exactly one — `api/src/api/v1/wake_schedules.py`):

```sh
grep -rn "agent\.wake\.reschedule\|_compute_next_fire_at" .
# verify: the grep above returns empty afterwards.
```

## Unchanged — nothing to do

- **`wake_tick_job` registration.** Its signature is preserved, so your tick
  driver (e.g. metallm's `api/src/services/scheduler.py`, which calls
  `await wake_tick_job(pool=..., nats_client=..., dispatch_callback=...)`) and
  any test that patches `threetears.agent.wake.tick.wake_tick_job` keep working
  as-is.
- **The dispatch callback.** Still `(WakeTrigger, fire_id, pool) ->
  WakeDispatchResult`. The adapter rebuilds the `WakeTrigger` from the generic
  envelope internally; your handler never sees the generic shape.
- **Schema / tables / migrations.** `agent_wake_schedules`, `wake_fires`,
  `webhook_subscriptions` are untouched. If you mirror the wake schema in your
  own migrations (metallm's alembic `096`–`099`), no new migration is needed.
- **`config` / `collections` / `entities` / `api_models` / `tools`.** Unchanged.

## Dependency

`3tears-agent-wake` now declares `3tears-scheduled-jobs` and **drops its direct
`3tears-nats` dependency** (the cross-pod lock belongs to the scheduled-jobs
core; wake reaches NATS only transitively). `uv sync` resolves the new edge
automatically — nothing to declare unless you pinned `3tears-nats` *solely* for
agent-wake, in which case it is now transitive via `3tears-scheduled-jobs`.

## Observability — update dashboards / alerts

The tick's per-fire metrics moved from the `threetears_agent_wake_*` family to
the generic `threetears_scheduled_jobs_*` family:

| Before (`threetears_agent_wake_*`) | After (`threetears_scheduled_jobs_*`) | Note |
|---|---|---|
| `fires_total{status,schedule_type,execution_mode}` | `fires_total{status,schedule_type}` | **`execution_mode` label dropped** |
| `failures_total{reason="conv_busy"}` (CAS miss) | `failures_total{reason="claim_lost"}` | reason renamed |
| `drift_seconds` | `drift_seconds` | family renamed |
| `tick_duration_seconds` | `tick_duration_seconds` | family renamed |

Preserved on the `threetears_agent_wake_*` family (still emitted): the
wake-specific `yield_duration_seconds`, plus all webhook / rate-limit /
schedule-cap metrics (those are emitted outside the tick path).

The `EVENT_FIRE_SKIPPED_BUSY` structured-log `extra_data` keys also changed
(`conversation_id` / `fire_source` / `execution_mode` → `job_id` /
`partition_key`); update any log-based alerts that key on the old fields.

## Verify after upgrading

```sh
grep -rn "agent\.wake\.reschedule\|_compute_next_fire_at" .   # → empty
python -c "from threetears.agent.wake.tick import wake_tick_job"  # still resolves
```

Then run your wake integration tests (the tick lifecycle, failure isolation, and
concurrent-claim race assertions are unchanged and should stay green).
