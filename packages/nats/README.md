# 3tears-nats

Typed NATS client wrapper, subject builders, and JetStream KV bucket primitives for 3tears applications.

## What this package provides

- `NatsClient` -- single canonical wrapper around `nats-py`. Handles connect (with a bounded startup timeout, then unbounded runtime reconnect -- it rides out an outage of any length rather than closing itself), graceful shutdown/drain, typed publish, kw-only subscribe with optional Pydantic validation, request/reply with `timedelta` timeouts, JetStream KV bucket access, and stream declaration (`ensure_jetstream_stream`). Streams and KV buckets the client declared (`ensure_jetstream_stream`, `ensure_kv_bucket`), on memory or file storage, and the durable consumers it bound on them, are created again after every reconnect, so a NATS restart that wipes them -- memory storage always, file storage when the restart lost its volume, as it can on Kubernetes -- heals without a process restart. They come back empty; contents are their owner's to write again. To have one refilled, declare it with `ensure_kv_bucket(on_restored=...)`, which the client runs with the live handle each time it creates the bucket again, or let `PersistedCopyBucket` own it.
- `Subject` + `Subjects` -- opaque subject dataclass and factory of every canonical subject family used by 3tears applications. Replaces ad-hoc `f"{namespace}.tools.call"` string-concatenation across the platform.
- `NatsKvBucket` -- operations against one JetStream KV bucket (`get` / `put` / `delete` / `create` / `update` / `get_entry` / `list_keys`). Bucket name auto-prefixed by the connected client's `nats_subject_namespace`, unless it was declared with `ensure_kv_bucket(prefix_namespace=False)`, which uses the name exactly. Every operation follows the client onto its current connection, so a handle survives a credential renewal or a move off a lame-duck server.
- `PersistedCopyBucket` -- the one owner of a KV bucket that is the persisted copy of an in-memory structure. It declares the bucket under its exact name in the background, retrying until the declaration lands; runs `load` (bucket into memory) once, at the first declaration; and runs `write_back` (memory into bucket) then and every time the client creates the bucket again empty. Use it rather than hand-rolling a declare/load/refill loop.
- `forward` / `serve_owner` -- payload-agnostic owner-routed request/reply: send a request to whichever pod currently serves a key and get its reply back. A separate election mechanism decides who owns the key; this only carries the message.
- `attach_pipe` / `serve_pipe` / `open_pipe` -- a payload-agnostic byte pipe to whichever pod owns a key, for reaching a process that has no inbound network path. Rendezvous rides `forward`; the stream then moves to its own subjects with a sequenced framing (a lost frame raises rather than being skipped) and a credit window that stops the producer reading its source when the consumer falls behind.
- `StreamTransport` -- narrow Protocol used by streaming consumers; lets test fakes substitute for the live client.
- Errors -- `NatsClientError`, `SubscribeError`, `PublishError`, `RequestError`, `KvError`, and `KvBucketNotFoundError` (a `KvError`). See [KV errors](#kv-errors).
- `is_bucket_not_found` / `is_key_not_found` / `is_nats_error` -- classify the failures of a RAW nats-py handle by type, so a consumer that keeps `nats.*` imports out of its code never matches nats-py class names as strings.

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

The cross-pod job lock is not in this package: `nats_distributed_lock` (with `LockHeld`, `LockHold`, `LockLost`) lives in core, `threetears.core.coordination`, beside the `KVLease` it runs on. It takes the `NatsClient` from this package as its `client`. Only `LockLossReason`, the reason a lost hold reports, is defined here (`threetears.nats.errors`).

## KV errors

A KV failure the wrapper raises is a `KvError`, except the two refusals in the table, which are deliberately not: the L2 accessors catch `KvError` and degrade, and a refusal must stop the process instead. One `KvError` has its own type, because it needs a different response from the rest:

| Raised | Means | Respond by |
|---|---|---|
| `KvBucketNotFoundError` (a `KvError`; `.bucket` names it) | The server answered that the bucket's stream does not exist. | Waiting for the declarer, which has not declared it yet or not since a NATS restart wiped it. |
| any other `KvError` | The call failed for another reason. A request this principal is not granted is never answered, so a missing grant and an unreachable broker both arrive here as a deadline. | Checking the grant the message names, or the broker. |
| `KvConfigMismatch` (NOT a `KvError`) | A bind-only open found the live bucket carrying a configuration it refuses. | Running the declarer, which reconciles it. A bucket-wide expiry other than the opener's lifetime is reconciled only by a declarer that owns the bucket: `ensure_kv_bucket(ttl=None, owns_bucket=True)` from the bucket's one declarer. A bucket found live on file storage is recreated on memory, dropping its entries, only with the second opt-in `drop_file_storage=True`; without it the owner raises this same error and leaves the bucket untouched. |
| `StreamSubjectsOverlapError` (NOT a `KvError`) | A declaring open, or an operation recreating its bucket, found another stream owning the bucket's subjects. | Removing the conflicting stream, usually a client on the wrong subject namespace. |

`KvBucketNotFoundError` is raised by a bind-only open (`kv_bucket` / `ensure_kv_bucket` with `create_if_missing=False`) once its wait for the declarer is spent, by a declaring open whose create went unanswered and whose bind found nothing, and by an operation on a bound handle -- `list_keys` included -- whose stream vanished and could not be bound again. An existing `except KvError` still catches it.

A consumer holding a RAW nats-py handle (a `KeyValue` reached through `jetstream_context()`) gets nats-py's own exceptions. Classify them with the predicates instead of importing nats-py:

```python
from threetears.nats import is_bucket_not_found, is_key_not_found, is_nats_error

try:
    entry = await raw_kv.get(key)
except Exception as exc:
    if is_key_not_found(exc):
        entry = None          # the bucket exists; the key does not
    elif is_bucket_not_found(exc):
        ...                   # the bucket's stream is gone: wait for, or ask, its declarer
    elif is_nats_error(exc):
        ...                   # any other failure the bus reported
    else:
        raise
```

`is_bucket_not_found` is also true of `KvBucketNotFoundError`, so one predicate serves a consumer holding both kinds of handle. None of the three reads message text.


## Enforcement

Direct `from nats import` / `from nats.aio` imports are flagged by the per-repo enforcement walker `tests/enforcement/test_nats_wrapper_usage.py`. Strict mode by default; exemptions require a `# rationale: ...` line.
