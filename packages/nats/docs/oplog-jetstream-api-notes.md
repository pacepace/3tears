# Op-log JetStream API notes (live-probed)

`verify-api` evidence for the `threetears.nats.oplog` primitive. Every signature
and behaviour below was **confirmed against a live JetStream-enabled NATS
testcontainer** (`testcontainers.nats.NatsContainer(jetstream=True)`), not read
from docs or assumed. nats-py is the version pinned in this repo's `.venv`
(Python 3.14). Re-probe if nats-py is bumped.

## Stream creation

```python
from nats.js.api import StorageType, StreamConfig

StreamConfig(
    name=<sanitized stream name, no dots>,
    subjects=[<one subject per (repo, branch)>],
    storage=StorageType.MEMORY,   # durability rides R3 replication, not disk
    num_replicas=3,               # R3
    duplicate_window=<seconds, float>,  # dedup window; we use 300.0 (5 min)
)
```

- `js.add_stream(config=...)` creates; on an already-existing stream nats-py
  raises (the create-or-bind idiom mirrors `kv.py`: try `add_stream`, fall back to
  `js.stream_info(name)` to bind). Confirmed `info.config.duplicate_window` comes
  back as a **float seconds** value (`300.0`), so we pass an int/float of seconds.
- **Non-clustered fallback (live-probed).** A single-node broker does **not**
  silently clamp R3 — it *rejects* it: `js.add_stream(num_replicas=3)` against a
  single-node testcontainer raises `nats.js.errors.ServerError` with `code=500`,
  `err_code=10074`, `description="replicas > 1 not supported in non-clustered
  mode"`. `OpLog.open` therefore tries R3 first and, **only on err_code 10074**,
  retries with `num_replicas=1` (logging a WARNING). R3 is the production intent;
  a dev/test single node runs at 1 replica. Any other `add_stream` error is a real
  failure (then a create-or-bind `stream_info` is attempted in case it pre-exists).

## Append — publish with CAS + dedup headers

```python
from nats.js.api import Header
ack = await js.publish(
    subject, payload,
    headers={
        Header.EXPECTED_LAST_SEQUENCE: str(expected_last_seq),  # "Nats-Expected-Last-Sequence"
        Header.MSG_ID:                 op_id,                    # "Nats-Msg-Id"
    },
    stream=<stream name>,
)
# ack: PubAck(stream, seq, domain, duplicate)
```

- `PubAck.seq` — assigned stream sequence (1-based, monotonic). Confirmed
  3 successive appends return seq 1, 2, 3.
- `PubAck.duplicate` — `True` when the `Nats-Msg-Id` was already in the dedup
  window AND no second message was stored; otherwise `None`/`False`. We coerce to
  `bool`.

### CRITICAL ORDERING: the server checks expected-last-seq BEFORE dedup

This is the load-bearing finding. With **both** headers present, the server
evaluates the CAS (`Nats-Expected-Last-Sequence`) **first**. Consequences,
all live-probed:

| Scenario | Headers on retry | Result |
|---|---|---|
| Dup `op_id`, **no** CAS header | `Msg-Id` only | `PubAck(duplicate=True)`, original seq, no 2nd msg |
| Dup `op_id`, CAS = **current** last seq | `Msg-Id` + `Expected-Last-Seq=1` | `PubAck(duplicate=True)`, original seq |
| Dup `op_id`, CAS = **stale** (e.g. 0 after stream advanced) | `Msg-Id` + `Expected-Last-Seq=0` | **raises** `BadRequestError` err_code 10071 |
| Fresh `op_id`, CAS = stale | new `Msg-Id` + stale `Expected-Last-Seq` | **raises** `BadRequestError` err_code 10071 |

The proof test's `test_duplicate_op_id_is_at_most_once` resends the identical op
with the **same** `expected_last_seq=0` after the stream advanced to seq 1 — i.e.
the third row above: the CAS is now stale, so the publish **raises** rather than
returning `duplicate=True`. A naive "trust `PubAck.duplicate`" implementation
fails this test.

### The conflict exception

```python
from nats.js.errors import APIError, BadRequestError
# raised: nats.js.errors.BadRequestError (subclass of APIError)
#   .code      == 400
#   .err_code  == 10071           <-- discriminator
#   .description like "wrong last sequence: 1"
```

`err_code == 10071` is the **only** safe discriminator — catch `APIError` and
re-raise anything whose `err_code != 10071` so unrelated 400s (bad stream, bad
subject) are never swallowed. (KV's `KeyWrongLastSequenceError` is a *KV-specific*
subclass and is **not** raised by a raw JetStream `publish`; the publish path
surfaces the bare `BadRequestError`.)

### Discriminating a stale-CAS duplicate-retry from a genuine conflict

On `err_code == 10071`, the append is either (a) a client/transport **retry** of
an op already logged (its CAS went stale because the original landed), or (b) a
genuine **fencing** conflict (a different writer with a stale view). Discriminate
by the dedup key: scan the stream for a stored message whose `Nats-Msg-Id` header
equals `op_id`.

- **Found** → it is a retry; return `AppendResult(seq=<found seq>,
  deduplicated=True)`. No second message was written (the publish was rejected).
- **Not found** → genuine fence; raise `OpLogSequenceConflict`.

Stored messages **preserve the `Nats-Msg-Id` header** (confirmed: replayed
messages carry `m.headers["Nats-Msg-Id"]`), so the scan can recover the original
sequence. The scan is on the rare conflict path only; the happy path is a single
publish.

## Replay — ordered, terminating

```python
from nats.js.api import ConsumerConfig, DeliverPolicy, AckPolicy

si = await js.stream_info(stream)
last = si.state.last_seq         # current end of log
if from_seq > last:              # past-end: yield nothing, return cleanly
    return
cc = ConsumerConfig(
    deliver_policy=DeliverPolicy.BY_START_SEQUENCE,
    opt_start_seq=from_seq,
    ack_policy=AckPolicy.NONE,
)
psub = await js.pull_subscribe(subject, stream=stream, config=cc)
# loop: psub.fetch(batch, timeout=...) until the last delivered seq >= last,
# then unsubscribe. A FetchTimeoutError ends the loop too (empty tail).
```

- `m.metadata.sequence.stream` — the stream sequence of a delivered message.
- `m.data` — payload bytes; `m.headers["Nats-Msg-Id"]` — the op_id.
- **Termination** is by reading `stream_info().state.last_seq` *once* up front and
  stopping as soon as a delivered seq reaches it. This is what makes
  `[r async for r in replay(...)]` terminate instead of hanging on a live
  subscription. Confirmed: full replay `[1..5]`, tail-from-4 `[4,5]`, past-end
  (from 99) `[]`, all terminate cleanly.
- `AckPolicy.NONE` — replay is read-only; nothing to ack. The ephemeral pull
  consumer is unsubscribed at the end.

## Constants used

- `duplicate_window` = **300 s (5 min)** — "generous (≥ minutes)" per write-path.md
  Constants; the expected-last-seq CAS is the unbounded backstop.
- `num_replicas` = **3** (R3).
- `storage` = **MEMORY**.
