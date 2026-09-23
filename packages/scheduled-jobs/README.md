# 3tears-scheduled-jobs

A generic, payload-agnostic, multipod-safe **scheduled-jobs core**. Every
agent/skill/webhook/conversation-specific concept is stripped out, leaving
only the scheduling machinery.

What it gives you:

- **`scheduled_tick_job(...)`** -- one cross-pod-locked tick pump. Acquire
  the `nats_distributed_lock` at a caller-supplied key; on `LockHeld`
  skip silently; on `KvError` degrade open (the per-row optimistic-CAS
  is the real guard); sweep abandoned in-flight fires per kind group;
  enumerate due rows of the routed kinds; per-row CAS-claim + reschedule;
  invoke the handler registered for that row's `kind`; drift /
  missed-fire accounting; per-row failure isolation. Takes the store(s),
  a `DispatchRoutes` table, and the NATS client as parameters, with no
  domain knowledge.
- **Per-`kind` routing, and an unrouted kind is inert.** The pump's
  `kind -> handler` table is matched EXACTLY -- no wildcard, no default
  handler, no fall-through -- and it also scopes the due-row scan (a SQL
  predicate, not a Python filter) and the reaper sweep. A row whose kind
  has no handler is refused with an `unrouted_kind` failure metric and an
  ERROR event, and is deliberately not claimed, so the occurrence
  survives until its handler is registered. Silent misdelivery is the
  failure this prevents: a row absorbed by another kind's dispatcher does
  that dispatcher's work and records it as a success.
- **Per-`kind` reap thresholds.** `dispatch_reap_after_seconds_by_kind`
  overrides `DEFAULT_DISPATCH_REAP_AFTER_SECONDS` per kind, so a kind
  whose work legitimately runs for hours is not reaped on the 15-minute
  baseline. Kinds sharing a threshold sweep in one query. Note the age is
  measured from dispatch start, not last activity, so the threshold alone
  only moves the false-reap cliff -- pair it with progress-conditioned
  renewal in the executor.
- **A distinct lock key per pump.** `JobConfig.tick_lock_key` defaults to
  `DEFAULT_TICK_LOCK_KEY`; consumers running more than one pump in a
  process MUST vary it, or the pumps serialise against each other for no
  reason.
- **`BackgroundDispatch` -- for fires that take long.** The pump awaits
  each handler inline, so a tick lasts as long as all its fires together
  and a kind due every minute waits behind every slow one. Wrap the
  handlers (`routes = {kind: background.wrap(h) ...}`) and each fire is
  handed off (`JobFireResult.handed_off`: the tick leaves the row
  `'dispatching'` and moves on), runs as its own task, and finalizes its
  own row with its real outcome. One fire per kind is in flight at a
  time: in-process, and across pods through `nats_distributed_lock` on
  `in_flight_lock_key(kind)`. A fire that finds its kind still running is
  recorded as a success whose output carries `IN_FLIGHT_SKIP_OUTPUT_KEY`.
  `max_concurrent` bounds how many run at once. The reaper counts from the
  tick, so each fire runs under the smaller of its own timeout and the time
  left before its kind's reap threshold (less `REAP_MARGIN_SECONDS`), and a
  timeout that could never fit is refused at construction; pass the pump's
  own `config`. Each fire is recorded exactly once: a process that dies
  mid-fire leaves the row to the reaper, as before, and `aclose()` records
  what it cancels (a fire whose body already finished keeps its result).
  Opt-in: a pump that does not wrap its handlers behaves exactly as before.
  **Exclusion groups** keep kinds that share something (one client object,
  one file handle) off each other: `exclusion_groups={kind: group}`. Kinds
  in one group never run at the same time; they take turns in the order
  they reach the group. A fire waits for its group's turn *before* it takes
  a slot and before its timeout starts, so waiting costs neither; the reap
  clock does keep running, so the wait is bounded by it, and a fire still
  waiting when its time runs out is recorded as failed without running,
  naming its group. `EVENT_FIRE_WAITING_EXCLUSION_GROUP` logs each wait with
  the kind holding the turn. Groups hold within one `BackgroundDispatch`
  (one process). A waiting fire holds its kind's cross-pod in-flight lock,
  so another pod records that kind as skipped instead of running it twice.
- **`compute_next_fire_at(...)`** -- the pure reschedule math for every
  schedule type (`daily_at`, `every_n_hours`, `random_within_window`,
  `one_shot_at`, `cron`, `relative_delay`, `interval`) and both
  missed-fire policies (`coalesce`, `catch_up`). The `cron` branch
  imports APScheduler lazily, so non-cron consumers pay nothing.
- **`ScheduleStore` / `FireStore` Protocols** -- the exact surface the
  tick engine calls. The engine depends only on these, so a typed
  consumer collection can implement them.
- **A default store** -- `ScheduledJobEntity` / `JobFireEntity` +
  collections + `scheduled_jobs` / `job_fires` table factories + a v001
  migration, keyed on an opaque `kind` (TEXT) + `payload` (JSONB). A
  simple consumer can use it as-is with no table of its own.
- **Generic config / events / metrics** -- a `JobConfig` protocol, the
  tick / fire / drift event-name constants, and cardinality-bounded
  Prometheus instruments.

The engine is **pure-async, one tick per call** with no internal polling.
Drive cadence with whatever scheduler you like (an APScheduler
`IntervalTrigger`, a `while True: await asyncio.sleep(...)`, and so on). The
engine does not own the scheduler.
