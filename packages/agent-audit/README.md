# 3tears-agent-audit

Unified audit envelope + fire-and-forget publish helper for the 3tears platform.

## Purpose

Single `AuditEvent` envelope + single `publish_audit` helper used by every
domain (workspace, rbac, memory, custom tools) so the hub's audit pipeline is
one subject tree, one consumer, one table, one admin query API. Replaces the
pre-audit-task-01 domain-specific envelopes (`WorkspaceAuditEnvelope`,
`RbacAuditEnvelope`) that produced slightly-different wire shapes per domain
and made cross-domain audit queries require a UNION.

The package is pure Python with no NATS consumer code and no Postgres code.
Publish is the only direction: the hub-side `unified_audit_consumer` owns
persistence to `platform_audit.audit_events`.

## Public API

```python
from threetears.agent.audit import AuditEvent, publish_audit
```

- `AuditEvent` — pydantic `BaseModel` with `extra='forbid'`, timezone-aware
  `timestamp` validator, closed `event_type` string family (dotted verb,
  e.g. `workspace.fs_write`, `rbac.assignment.create`). All common identity
  fields are typed columns on the envelope; event-type-specific extras live
  in `details: dict[str, Any]`.
- `publish_audit(event, nats_client, namespace)` — fire-and-forget async
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
- **No dual-emit.** There is no legacy envelope the hub still accepts in
  parallel. Emission sites migrate in the same PR that deletes the legacy
  envelope module.

## Subject naming

`{namespace}.audit.{event_type}` where `event_type` is the dotted event
name (e.g. `{namespace}.audit.workspace.fs_write`). The hub consumer
subscribes to `{namespace}.audit.>` so new event types route automatically
without consumer-side changes.
