# 3tears-agent-audit

Unified audit envelope, fire-and-forget publish helper, and the erasure rule and
hub request for anonymizing audit records, for the 3tears platform.

## Purpose

Single `AuditEvent` envelope + single `publish_audit` helper used by every
domain (workspace, rbac, memory, custom tools) so the audit pipeline is
one subject tree, one consumer, one table, one admin query API. Replaces
domain-specific envelopes (`WorkspaceAuditEnvelope`,
`RbacAuditEnvelope`) that produced slightly-different wire shapes per domain
and made cross-domain audit queries require a UNION.

Publishing is the package's core: `AuditEvent` and `publish_audit` import nothing
beyond the NATS client. On a hub deployment the HUB persists every event into its
platform table, and the package's persistence code is never used; an agent there
erases a person from that table by asking the hub (`request_audit_anonymization`),
and the hub applies the erasure rule.

A deployment with NO hub (one application owning its own control plane) owns its
audit table, and `threetears.agent.audit.persist` is its persister:

- `ensure_audit_events_table(db)` creates the table, with every envelope field and
  `ip_address`, and migrates an existing one: every column the insert names beyond
  `id` and the four required fields (`timestamp`, `event_type`, `action`,
  `correlation_id`) is added if missing, with the default the CREATE gives it.
  Idempotency is on the envelope `id`.
- `start_audit_persister(nats, db, durable=..., storage="memory")` runs a shared
  durable pull consumer with a dead-letter subject. A malformed event is dropped;
  a database fault is retried, then dead-lettered.
  - `storage` must match every other declarer of the `audit` stream.
  - `durable` must be unique per table.
- `prune_audit_events(db, older_than=...)` is batched retention.
- `anonymize_audit_rows(db, actor_user_ids=...)` is erasure: every row and id is
  kept, and `details` and `ip_address` are rewritten by the platform's rule and no
  other -- it takes no replacement anonymizer. A family whose keys are safe in this
  deployment declares them with `declare_safe_detail_keys`. It returns the same
  `AuditAnonymization` the hub path does, and never answers the hub's anonymize
  subject.

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
- `anonymize_ip(value) -> str | None` returns `None`: an address is removed, not
  truncated. It is typed as the column, so its result assigns back without an ignore.
- `SAFE_DETAIL_KEYS` is the explicit safe list. A key not on it is masked, so
  a field nobody classified fails safe. `PERSONAL_DETAIL_KEYS` records the keys
  classified as able to carry personal data.
- `declare_safe_detail_keys(event_type_prefix, keys)` widens the safe set for
  one event family, and `safe_detail_keys_for(event_type)` is the single
  lookup. A declaration is visible only in the process that makes it, so a
  family whose events the hub erases from the platform audit table is declared
  in `threetears/agent/audit/anonymize.py` itself.
- `is_classified_detail_key(key, *, event_type)` answers, in process, whether a key
  was classified. The gate that holds producers to the classification carries its
  own copy of that predicate (it cannot import this package): the
  `threetears.enforcement.audit_details` domain, which every producing repo runs
  over its own `src/` with `safe_keys_for=safe_detail_keys_for,
  personal_keys=PERSONAL_DETAIL_KEYS`, and 3tears runs it as
  `tests/enforcement/test_audit_details_keys_are_classified.py`.

## An agent erasing a person: the hub's copy of its audit rows

```python
from threetears.agent.audit import request_audit_anonymization

result = await request_audit_anonymization(
    nats_client,
    identity_token=current_identity_token(),
    agent_id=my_agent_id,
    actor_user_ids=[respondent_admission_user_id],
)
# result.rows_matched, result.rows_changed
```

The hub holds the audit rows an agent's events became. This asks it to anonymize the
ones that agent published about those actors, on `{ns}.hub.audit.anonymize`
(`Subjects.hub_audit_anonymize()`), with the same rule as above: rows kept, ids kept,
`details` and `ip_address` anonymized. The hub takes the agent from the verified identity
token and touches only rows whose agent is the caller. A refusal (`INVALID_REQUEST`,
`IDENTITY_UNVERIFIED`, `AGENT_MISMATCH`) raises `AuditAnonymizeRefusedError` with the
hub's `error_code`, and retrying meets it again -- including a refusal that carries no
correlation id, since a hub that could not decode the body had none to echo. No token, a
timeout, a reply that does not decode or carries another request's correlation id, and
the hub's `ANONYMIZE_FAILED` raise `AuditAnonymizeUnavailableError`, which is safe to
retry. The contract, including every
obligation of the hub's responder, is the docstring of `threetears/agent/audit/erasure.py`.

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
