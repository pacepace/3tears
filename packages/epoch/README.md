# 3tears-epoch

Generation-stamped configuration epochs with NATS broadcast and per-message echo for cross-pod cache-reload coherence.

## Why

Multiple in-memory configuration caches across the 3tears platform need to stay coherent across pods on admin writes:

- metallm's `ModelCapabilities` registry (registered at startup from the `models` table)
- 14-eng-ai-bot gateway's catalog cache (`gateway_models` + `gateway_providers` + `gateway_credit_rates`)
- per-tool MCP RBAC grants (when MCP shared framework lands)

Pure NATS broadcast (push) ships with a missed-message hole: a pod that didn't receive the broadcast (subscriber blip, pod just started during the window, JetStream redelivery edge) stays stale. Pure polling (pull) is correct but expensive on hot paths.

This package combines both: a strictly-monotonic generation number (epoch) per *subject*, durable in Postgres, broadcast best-effort via NATS, and echoed in every relevant response message so consumers detect staleness on the next read and lazy-pull. Push for speed, pull for correctness.

This is the standard pattern from etcd `mod_revision` + watch, K8s `resourceVersion` + informer, Envoy xDS `version_info` + ACK, DNS SOA serial + secondary refresh.

## Identity

The unit of identity is the **NATS subject path**. Each consumer:

1. Defines or uses an existing `Subject` builder for the configuration domain it owns (e.g. `Subjects.metallm_capabilities_epoch()` -> `metallm.capabilities.epoch`).
2. Calls `EpochClient.bump(subject, payload=...)` after committing the row mutation that motivates the reload.
3. Subscribes via `EpochListener.subscribe(subject, on_bump=...)` from sibling pods.

The `platform.config_epochs` row PK is the subject path string. Postgres is the source of truth; the NATS broadcast is best-effort. A subscriber that missed every broadcast still catches up on the next request whose response echoes the higher epoch (per-message echo is consumer-side wiring; the framework supplies the building blocks).

## Wire envelope

`EpochBumpMessage` is a frozen Pydantic v2 model:

- `subject_path: str` -- the namespaced subject the bump targets (matches the row PK)
- `epoch: int` -- the new strictly-monotonic value
- `payload: dict[str, Any] | None` -- opaque hint for the consumer's reload callback (e.g. `{"model_id": "...", "action": "create"}`)

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

`bump(subject, payload)` runs `INSERT ... ON CONFLICT (subject_path) DO UPDATE SET epoch = config_epochs.epoch + 1, payload = $2, date_updated = now() RETURNING epoch`. Atomic; concurrent bumps from different writers serialize on the row lock.

Migration ships as a PLATFORM-scope `PackageMigrations` registration so consumers wire it via the canonical `MigrationRunner` alongside the rest of their platform tables.
