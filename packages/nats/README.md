# 3tears-nats

Typed NATS client wrapper, subject builders, and JetStream KV bucket primitives for 3tears applications.

## What this package provides

- `NatsClient` — single canonical wrapper around `nats-py`. Handles connect (with bounded startup-timeout + bounded runtime reconnect ceiling), graceful shutdown/drain, typed publish, kw-only subscribe with optional Pydantic validation, request/reply with `timedelta` timeouts, and JetStream KV bucket access.
- `Subject` + `Subjects` — opaque subject dataclass and factory of every canonical subject family used by aibots / 3tears apps. Replaces ad-hoc `f"{namespace}.tools.call"` string-concatenation across the platform.
- `NatsKvBucket` — operations against one JetStream KV bucket (`get` / `put` / `delete` / `create` / `update` / `get_entry`). Bucket name auto-prefixed by the connected client's `nats_subject_namespace`.
- `StreamTransport` — narrow Protocol used by streaming consumers; lets test fakes substitute for the live client.
- Errors — `NatsClientError`, `SubscribeError`, `PublishError`, `RequestError`, `KvError`.

## Why a separate package

The wrapper is consumed by both the aibots platform (hub, gateway, registry, channel adapters, agent SDK) and any future 3tears-based application (metallm, etc.). Keeping it in `3tears-nats` avoids forcing those apps to depend on the aibots hub repo just for a NATS primitive.

## Mistake-proofed API

- Subscribe is keyword-only after `self`. The 2026-04-25 production bug (`nc.subscribe(subject, callback)` silently treating the callback as a queue group in `nats-py` 2.10+) is impossible to reproduce against this wrapper.
- Publish accepts `BaseModel` instances. Raw bytes go through the explicit escape hatch `publish_raw`.
- Subjects are typed `Subject` objects, not strings. The factory owns subject formatting; callers cannot accidentally interpolate the wrong shape.
- Default `deadletter_on_error=True` — uncaught subscribe-callback exceptions auto-republish to `{ns}.deadletter.{path}`.

## Usage

```python
from datetime import timedelta

from threetears.nats import NatsClient, Subjects

nc = await NatsClient.connect(
    nats_url="nats://localhost:4222",
    nats_subject_namespace="aibots",
    client_name="my-service",
)

# Typed publish
await nc.publish(
    subject=Subjects.audit_event("workspace.doc_set"),
    message=AuditEvent(...),
)

# Typed request/reply
response = await nc.request(
    subject=Subjects.tools_call(),
    message=ToolCallRequest(...),
    response_type=ToolCallResponse,
    timeout=timedelta(seconds=5),
)

# Subscribe with Pydantic validation
sub = await nc.subscribe_typed(
    subject=Subjects.audit_wildcard(),
    cb=on_audit_event,
    message_type=AuditEvent,
    queue="audit-consumer",
)

# JetStream KV
bucket = await nc.kv_bucket(name="agent_config", ttl=timedelta(hours=2))
await bucket.put(key="agent-1", value=b"config-payload")

await nc.shutdown()
```

## Enforcement

Direct `from nats import` / `from nats.aio` imports are flagged by the per-repo enforcement walker `tests/enforcement/test_nats_wrapper_usage.py`. Strict mode by default; exemptions require a `# rationale: ...` line.
