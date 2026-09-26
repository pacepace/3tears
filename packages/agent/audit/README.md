# 3tears-agent-audit

Unified audit envelope + fire-and-forget publish helper for the 3tears platform.

## Purpose

Single `AuditEvent` envelope + single `publish_audit` helper used by every
domain (workspace, rbac, memory, custom tools) so the audit pipeline is
one subject tree, one consumer, one table, one admin query API. Replaces
domain-specific envelopes (`WorkspaceAuditEnvelope`,
`RbacAuditEnvelope`) that produced slightly-different wire shapes per domain
and made cross-domain audit queries require a UNION.

The package is pure Python with no NATS consumer code and no Postgres code.
Publish is the only direction: a consumer-side audit consumer owns
persistence to the audit events table.

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

## Erasure: anonymize, never delete

```python
from threetears.agent.audit import anonymize_details, anonymize_ip

row.details = anonymize_details(row.details, event_type=row.event_type)
row.ip_address = anonymize_ip(row.ip_address)
```

An audit record is never deleted by erasure and no id on it changes. Every
erasure path (a GDPR request, a principal anonymization, a customer
offboarding) routes the record's content through this one rule instead of
writing its own:

- `anonymize_details(details, *, event_type)` keeps every key of `details`,
  keeps the value under a safe key (a dict inside it is judged key by key),
  and replaces the WHOLE value under any other key with `ANONYMIZED_MARKER`
  (`"[anonymized]"`) -- a subtree and the keys a user chose inside it alike.
  `None` stays `None`. Pure and idempotent.
- `anonymize_ip(value)` returns `None`: an address is removed, not truncated.
- `SAFE_DETAIL_KEYS` is the explicit safe list. A key not on it is masked, so
  a field nobody classified fails safe. `PERSONAL_DETAIL_KEYS` records the keys
  classified as able to carry personal data.
- `declare_safe_detail_keys(event_type_prefix, keys)` widens the safe set for
  one event family, and `safe_detail_keys_for(event_type)` is the single
  lookup. A declaration is visible only in the process that makes it, so a
  family whose events the hub erases from the platform audit table is declared
  in `threetears/agent/audit/anonymize.py` itself.
- `is_classified_detail_key(key, *, event_type)` answers whether a key was
  classified. The gate that holds producers to the classification is the
  `threetears.enforcement.audit_details` domain: every producing repo runs it
  over its own `src/` with `safe_keys_for=safe_detail_keys_for,
  personal_keys=PERSONAL_DETAIL_KEYS`, and 3tears runs it as
  `tests/enforcement/test_audit_details_keys_are_classified.py`.

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
