# 3tears-scheduled-jobs

A generic, payload-agnostic, multipod-safe **scheduled-jobs core** —
extracted and generalized from `3tears-agent-wake`'s scheduling
machinery with every agent/skill/webhook/conversation-specific concept
stripped out.

What it gives you:

- **`scheduled_tick_job(...)`** — one cross-pod-locked tick pump. Acquire
  the `nats_distributed_lock` at a caller-supplied key; on `LockHeld`
  skip silently; on `KvError` degrade open (the per-row optimistic-CAS
  is the real guard); enumerate due rows; per-row CAS-claim + reschedule;
  invoke an injected dispatch callback; drift / missed-fire accounting;
  per-row failure isolation. Takes the store(s), the dispatch callback,
  and the NATS client as parameters — no domain knowledge.
- **`compute_next_fire_at(...)`** — the pure reschedule math for every
  schedule type (`daily_at`, `every_n_hours`, `random_within_window`,
  `one_shot_at`, `cron`, `relative_delay`, `interval`) and both
  missed-fire policies (`coalesce`, `catch_up`). The `cron` branch
  imports APScheduler lazily, so non-cron consumers pay nothing.
- **`ScheduleStore` / `FireStore` Protocols** — the exact surface the
  tick engine calls. The engine depends only on these, so a typed
  consumer collection can implement them.
- **A default store** — `ScheduledJobEntity` / `JobFireEntity` +
  collections + `scheduled_jobs` / `job_fires` table factories + a v001
  migration, keyed on an opaque `kind` (TEXT) + `payload` (JSONB). A
  simple consumer can use it as-is with no table of its own.
- **Generic config / events / metrics** — a `JobConfig` protocol, the
  tick / fire / drift event-name constants, and cardinality-bounded
  Prometheus instruments.

The engine is **pure-async, one tick per call** — no internal polling.
Drive cadence with whatever scheduler you like (an APScheduler
`IntervalTrigger`, a `while True: await asyncio.sleep(...)`, …). The
platform does not own the scheduler.
