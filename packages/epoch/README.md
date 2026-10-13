# 3tears-epoch

Generation-stamped configuration epochs with NATS broadcast and per-message echo for cross-pod cache-reload coherence.

## Why

Multiple in-memory configuration caches across the platform need to stay coherent across pods on admin writes:

- a model capabilities registry (registered at startup from the `models` table)
- a catalog cache (`gateway_models` + `gateway_providers` + `gateway_credit_rates`)
- per-tool MCP RBAC grants

Pure NATS broadcast (push) ships with a missed-message hole: a pod that didn't receive the broadcast (subscriber blip, pod just started during the window, JetStream redelivery edge) stays stale. Pure polling (pull) is correct but expensive on hot paths.

This package combines both: a strictly-monotonic generation number (epoch) per *subject*, broadcast best-effort via NATS and echoed in every relevant response message so consumers detect staleness on the next read and lazy-pull. Push for speed, pull for correctness.

The counter itself lives in a memory-backed NATS KV bucket for every epoch whose value stays inside the cluster. That is deliberate: an epoch is a coherence signal, not a durable fact, and a restart that loses the counter also loses every cache it was sequencing. The one exception is an epoch whose value *escapes* -- a tile epoch is the `v{n}` in a tile URL and reaches browser and CDN caches nothing here can reach -- and that family keeps a durable `config_epochs` row.

This is the standard pattern from etcd `mod_revision` + watch, K8s `resourceVersion` + informer, Envoy xDS `version_info` + ACK, DNS SOA serial + secondary refresh.


## Deployment obligation: the KV grant

Every principal that bumps or reads an epoch must be granted the `{ns}-epochs` KV bucket.
`threetears.nats.subject_permissions` is the list: today the hub and the gateway hold it
writable, and the agent pod, the tool pod and the standalone registry hold it read only (a
pod follows epochs and write generations and never advances one). **A missing KV grant does
not raise.** A refused JetStream request is never answered, so the call blocks to
its deadline and returns a timeout, indistinguishable by shape from an unreachable broker.

The log tells them apart. `threetears.nats` reads the server's `permissions violation` frame
(which leaves the connection open, so nothing else reports it), names the bucket, and states
the `js_resources` entry to add; the deadline path and a failed bucket open say the same.
`tests/enforcement/test_kv_bucket_grant_naming.py` pins the grant against the bucket the
client actually opens, which catches a wrong name before a deploy rather than in a log.

Consumers also register an `on_reset` callback and schedule a
`threetears.epoch.catchup_tick` pass. A consumer that subscribes and schedules neither still
receives broadcasts, and misses everything a broadcast can lose -- including a counter
replaced by a broker restart, which every KV operation survives silently.

## The durable family: `advance_to` and `versions`

A durable subject (a tile epoch) has two operations an ephemeral one does not:

- `EpochClient.advance_to(subject, epoch, payload=None) -> DurableEpoch` moves the epoch
  forward to a target and never back, in one statement under the row lock, then broadcasts.
  A subject already at or past the target is left alone and nothing is broadcast. Use it
  where the epoch names something outside the counter (a tile version equal to the
  generation of the rows it reads), where `bump`'s "one more" would let two writers carry
  it past the value both meant.
- `EpochClient.versions(subject, *, cached=False) -> DurableEpoch` reads the epoch and the
  one its latest move replaced (`DurableEpoch.epoch`, `DurableEpoch.previous`). `previous`
  is recorded by every move, never inferred as `epoch - 1`; it is `None` until the subject
  has moved twice.

`versions(subject, cached=True)` answers from a per-process copy, for a hot path. The copy
has no age. It answers until the epoch system says the subject moved: this process's own
`advance_to` writes it, its `bump` drops it, a broadcast naming a later epoch drops it, and
a missed broadcast is repaired by the catch-up pass, whose read of the row
(`EpochClient.current`) drops a copy the row has moved away from. A copy is kept only for a
subject an `EpochListener` over the same client is subscribed to, by its concrete subject or
a wildcard covering it; for any other subject `cached=True` reads the row every time. So a
process that reads cached versions:

1. subscribes the subject, or a wildcard covering it, on an `EpochListener` built on the
   same `EpochClient`;
2. schedules `catchup_tick(listener, [(subject, on_bump) for subject in
   client.cached_subjects])`, as every epoch consumer schedules a pass.

A cached value may lag the row between a missed broadcast and the pass that repairs it. Use
it to choose what to serve or advertise, and read without `cached` before reading source
rows at a version.

**A stand-in for `EpochClient` handed to `EpochListener` must have `observe_subscription`
and `observe_broadcast`.** The listener calls the first once per successful subscribe and
the second for every broadcast it receives, before its dedupe. No-ops are enough for a stub
that keeps no copy.

## Following collection write generations

The epoch bucket also holds each switched-on table's write generation
(`EpochGenerationSource` advances it, `EpochGenerationReader` reads and watches it). A
registry that follows a table drops it in this process when the bucket shows an advance it
did not hear every row of. Three ways to follow, all exported from `threetears.epoch`:

- `generation_catchup_tick(registry, reader)` -- one pass over every table the registry
  follows; the consumer schedules it.
- `follow_generation_key(registry, reader, table_name, *, grace=...)` -- one table, by
  watching its key, until cancelled.
- `follow_tables(registry, reader, tables)` -- returns a started `TableFollower`: one
  supervised watch per table, restarted with a backoff when it ends or fails, with each
  table's `WatchHealth` readable and `stop()` bounded. It tells the registry whether each
  watch is running (`CollectionRegistry.watched_by`), which is what a cache derived from
  the table checks before it serves what it holds. `threetears.agent.acl.generation_follow`
  uses it for the access tables.

## Identity

The unit of identity is the **NATS subject path**. Each consumer:

1. Defines or uses an existing `Subject` builder for the configuration domain it owns (e.g. `Subjects.capabilities_epoch()` -> `capabilities.epoch`).
2. Calls `EpochClient.bump(subject, payload=...)` after committing the row mutation that motivates the reload.
3. Subscribes via `EpochListener.subscribe(subject, on_bump=...)` from sibling pods.

For the durable family, the hub's `config_epochs` row PK is the subject path string and Postgres is the source of truth. For everything else the KV counter is, keyed on the subject path (digested when the path falls outside the KV key grammar). The NATS broadcast is best-effort either way. A subscriber that missed every broadcast still catches up on the next request whose response echoes the higher epoch (per-message echo is consumer-side wiring; the framework supplies the building blocks).

## Wire envelope

`EpochBumpMessage` is a frozen Pydantic v2 model:

- `subject_path: str` is the namespaced subject the bump targets (the KV counter key, or the row PK for the durable tile family)
- `epoch: int` is the new strictly-monotonic value
- `payload: dict[str, Any] | None` is an opaque hint for the consumer's reload callback (e.g. `{"model_id": "...", "action": "create"}`)

The framework never inspects `payload`. Consumers parse if useful, ignore if not. The hint exists so a domain that only changes one row can avoid reloading the entire derived view.

## Postgres

The durable family's row lives in `config_epochs`: `subject_path` (the primary key), `epoch`, `previous_epoch`, `payload` and `date_updated`. The migrations are the definition (`threetears.epoch.migrations`: v001 creates the table, v002 adds the nullable `previous_epoch`), and they ship as a PLATFORM-scope `PackageMigrations` registration so consumers wire them via the canonical `MigrationRunner` alongside the rest of their platform tables.

`bump(subject, payload)` increments the subject's counter and returns the new value. For an ephemeral subject that is a `DistributedCounter` CAS loop over NATS KV; for the durable family it is one upsert serialized on the row lock, which also records the epoch it replaced in `previous_epoch` (the statements are `_BUMP_SQL` and `_ADVANCE_SQL` in `threetears.epoch.client`). Both are atomic and both hand back a per-subject contiguous number, so `0` means "never bumped" and the first bump returns `1`.

`previous_epoch` is written by each writer's own statement, so it is right only while every writer of a subject runs a release that names it (0.66.0 or later).
