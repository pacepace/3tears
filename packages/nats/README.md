# 3tears-nats

Typed NATS client wrapper, subject builders, and JetStream KV bucket primitives for 3tears applications.

## What this package provides

- `NatsClient` -- single canonical wrapper around `nats-py`. Handles connect (with bounded startup-timeout + bounded runtime reconnect ceiling), graceful shutdown/drain, typed publish, kw-only subscribe with optional Pydantic validation, request/reply with `timedelta` timeouts, and JetStream KV bucket access.
- `Subject` + `Subjects` -- opaque subject dataclass and factory of every canonical subject family used by 3tears applications. Replaces ad-hoc `f"{namespace}.tools.call"` string-concatenation across the platform.
- `NatsKvBucket` -- operations against one JetStream KV bucket (`get` / `put` / `delete` / `create` / `update` / `get_entry`). Bucket name auto-prefixed by the connected client's `nats_subject_namespace`.
- `nats_distributed_lock` -- TTL-based distributed lock primitive built on `NatsKvBucket.create` (put-if-absent). Atomic acquisition + background heartbeat + automatic cleanup; on holder death the TTL expires the key.
- `StreamTransport` -- narrow Protocol used by streaming consumers; lets test fakes substitute for the live client.
- Errors -- `NatsClientError`, `SubscribeError`, `PublishError`, `RequestError`, `KvError`.

## Why a separate package

The wrapper is consumed by the platform services (broker, gateway, registry, channel adapters, agent SDK) and any 3tears-based application. Keeping it in `3tears-nats` avoids forcing those apps to depend on a host application repo just for a NATS primitive.

## Mistake-proofed API

- Subscribe is keyword-only after `self`. A common production bug (`nc.subscribe(subject, callback)` silently treating the callback as a queue group in `nats-py` 2.10+) is impossible to reproduce against this wrapper.
- Publish accepts `BaseModel` instances. Raw bytes go through the explicit escape hatch `publish_raw`.
- Subjects are typed `Subject` objects, not strings. The factory owns subject formatting; callers cannot accidentally interpolate the wrong shape.
- Default `deadletter_on_failure=True`. Uncaught subscribe-callback exceptions auto-republish to `{ns}.deadletter.{path}`.

## Usage

```python
from datetime import timedelta

from threetears.nats import NatsClient, Subjects

nc = await NatsClient.connect(
    nats_url="nats://localhost:4222",
    nats_subject_namespace="myapp",
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

## Distributed locks

`nats_distributed_lock` is a TTL-backed lock for "only one pod should run this body" patterns (scheduled jobs, periodic ticks, exclusive resource access). It is layered on top of `NatsKvBucket.create` (atomic put-if-absent), with a background heartbeat that refreshes the entry while the body runs and a TTL that bounds the orphan-lock window after a pod death.

```python
from datetime import timedelta

from threetears.nats import LockHeld, nats_distributed_lock

try:
    async with nats_distributed_lock(
        nc,
        "backup-job",
        bucket_name="scheduler-locks",     # default; override per consumer
        ttl=timedelta(seconds=60),         # KV entry TTL
        heartbeat=timedelta(seconds=20),   # heartbeat MUST be < ttl
    ):
        await run_backup()
except LockHeld:
    # another pod owns the lock; skip this run cleanly
    return
```

`client=None` is a graceful no-op that yields immediately. Single-pod dev environments without NATS work unchanged. `LockHeld` is distinct from `KvError` (transport / bucket failures); callers should treat the former as the expected "another pod is running" branch and surface the latter separately.


## Enforcement

Direct `from nats import` / `from nats.aio` imports are flagged by the per-repo enforcement walker `tests/enforcement/test_nats_wrapper_usage.py`. Strict mode by default; exemptions require a `# rationale: ...` line.
