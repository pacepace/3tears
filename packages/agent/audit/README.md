# 3tears-agent-audit

Unified audit envelope + fire-and-forget publish helper for the 3tears platform.

## Purpose

Single `AuditEvent` envelope + single `publish_audit` helper used by every
domain (workspace, rbac, memory, custom tools) so the audit pipeline is
one subject tree, one consumer, one table, one admin query API. Replaces
domain-specific envelopes (`WorkspaceAuditEnvelope`,
`RbacAuditEnvelope`) that produced slightly-different wire shapes per domain
and made cross-domain audit queries require a UNION.

Publishing is the package's core: `AuditEvent` and `publish_audit` import nothing
beyond the NATS client. On a hub deployment the HUB persists every event into its
platform table, and the package's persistence code is never used.

A deployment with NO hub (one application owning its own control plane) owns its
audit table, and `threetears.agent.audit.persist` is its persister:

- `ensure_audit_events_table(db)` creates the table, with every envelope field,
  `ip_address`, and both idempotency anchors.
- `start_audit_persister(nats, db, durable=...)` runs a shared durable pull
  consumer. The stream is file-backed with a dead-letter subject; a malformed event
  is dropped and a database fault is retried, then dead-lettered.
- `prune_audit_events(db, older_than=...)` is retention.
- `anonymize_audit_rows(db, actor_user_ids=...)` is erasure: every row and id is
  kept, and `details` and `ip_address` are rewritten by the platform's rule. It is
  this deployment's own erasure and never answers the hub's anonymize subject.

`db` is anything with asyncpg's `execute`/`fetch`, e.g. a pool.

## Public API

```python
from threetears.agent.audit import AuditEvent, publish_audit
```

- `AuditEvent` -- pydantic `BaseModel` with `extra='forbid'`, timezone-aware
  `timestamp` validator, closed `event_type` string family (dotted verb,
  e.g. `workspace.fs_write`, `rbac.assignment.create`). All common identity
  fields are typed columns on the envelope; event-type-specific extras live
  in `details: dict[str, Any]`.
- `publish_audit(event, nats_client, namespace)` -- fire-and-forget async
  helper. Serializes the envelope via `model_dump_json()` and awaits one
  `nats_client.publish` on `{namespace}.audit.{event_type}`. On any publish
  failure logs at WARN and returns; never raises.

## Design commitments

- **Fire-and-forget.** Audit publish failures must never break the producing
  call. The helper catches every exception, logs at WARN, and returns.
- **Typed wire contract.** `extra='forbid'` + timezone-aware validator catch
  publisher-side drift at construction time, not at the consumer's decode.
- **No domain-specific envelope types.** Every domain publishes the same
  model; `event_type` conveys the domain.
- **No dual-emit.** There is no legacy envelope the consumer still accepts in
  parallel. Emission sites migrate in the same PR that deletes the legacy
  envelope module.

## Subject naming

`{namespace}.audit.{event_type}` where `event_type` is the dotted event
name (e.g. `{namespace}.audit.workspace.fs_write`). The consumer
subscribes to `{namespace}.audit.>` so new event types route automatically
without consumer-side changes.
