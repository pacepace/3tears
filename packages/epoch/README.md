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

Every principal that bumps or reads an epoch must be granted the `{ns}-epochs` KV bucket
(`AGENT_POD`, `HUB`, `GATEWAY` in `threetears.nats.subject_permissions`). **A missing KV
grant does not raise.** A refused JetStream request is never answered, so the call blocks to
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

```
CREATE TABLE IF NOT EXISTS config_epochs (
    subject_path TEXT PRIMARY KEY,
    epoch BIGINT NOT NULL DEFAULT 0,
    payload JSONB,
    date_updated TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

`bump(subject, payload)` increments the subject's counter and returns the new value. For an ephemeral subject that is a `DistributedCounter` CAS loop over NATS KV; for the durable family it is `INSERT ... ON CONFLICT (subject_path) DO UPDATE SET epoch = config_epochs.epoch + 1, payload = $2, date_updated = now() RETURNING epoch`, serialized on the row lock. Both are atomic and both hand back a per-subject contiguous number, so `0` means "never bumped" and the first bump returns `1`.

Migration ships as a PLATFORM-scope `PackageMigrations` registration so consumers wire it via the canonical `MigrationRunner` alongside the rest of their platform tables.
