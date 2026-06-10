# channels-task-01: Cross-pod room broadcast backplane (NATS-backed)

**Status:** READY. Foundational — `channels-task-02` (presence roster) and `channels-task-03`
(typed frames + resume) build on the subject family and injection seam this lands.
**Scope:** `3tears-channels` (`websocket.py` — `ConnectionRegistry`, `WebSocketHandler`) + a new
`room` subject family in `3tears-nats` (`subjects.py`). Net-new framework code; **additive only**.
**Origin:** `docs/channels-cross-pod-design.md` (D1, D5). Today `ConnectionRegistry.broadcast_to_room`
reaches sockets on *this pod only* (in-process dicts, no NATS). A socket on pod A cannot reach a
room member on pod B. This task makes room broadcast **cross-pod** while keeping every existing
in-process consumer byte-for-byte unchanged.

---

## Objective

Make `ConnectionRegistry` room broadcast reach room members **on every pod**, via NATS fanout —
publish a room message to a per-room NATS subject; each pod that has a local member of that room
re-broadcasts to its **own** sockets. The cross-pod path is **opt-in**: an injected NATS backplane
turns it on; without one, behaviour is exactly today's in-process broadcast (existing chat consumers
untouched). This is the **fast-notify transient fanout** only — durable ordering/resume is the
op-log's job (`channels-task-03` / the consuming app), not this layer.

---

## Design constraints (the cross-pod fanout must stay single-purpose)

- **Additive, opt-in, backward-compatible.** `ConnectionRegistry()` and `WebSocketHandler(...)` keep
  working with no NATS. The backplane is an *optional* constructor parameter (default `None`).
  **No backplane ⇒ today's in-process `broadcast_to_room`, unchanged.** A consumer opts into
  cross-pod by passing a backplane; its message-handling code does not change.
- **One delivery path — publish, then every pod fans locally** (mirror the `epoch` pattern:
  `EpochClient.bump` publishes, sibling `EpochListener`s act locally). `broadcast_to_room` **publishes**
  to the room subject; the per-pod subscriber (including the sender's own pod) receives it and calls
  the **local** in-process fanout. Do **not** also fan locally at publish time — that double-delivers
  to local members. One path = consistent ordering, no dupes.
- **Subscribe on first local member, unsubscribe on last.** A pod subscribes to a room's NATS subject
  only while it holds ≥1 local member of that room (on `join_room`'s first add); it unsubscribes when
  the last local member leaves (`leave_room`). Pods never subscribe to rooms they have no one in.
- **`exclude` survives the hop.** `broadcast_to_room(room_id, message, exclude=ws)` excludes a local
  socket today. Cross-pod, the excluded socket lives on the sender's pod, so the published frame must
  carry an **exclude connection-id** that each pod honours against its own members (assign a stable id
  per registered socket). A `None` exclude broadcasts to everyone.
- **Transient, best-effort.** A dropped room message is **not** recovered here — the durable layer
  (op-log resume, `channels-task-03`) does that, exactly as `epoch` recovers a missed bump via the
  next tick/echo. Do not add durability/replay to this fanout.
- **Reuse, don't reinvent.** Subjects go through `threetears.nats.Subjects` (never an f-string).
  NATS pub/sub through the `NatsClient` wrapper (`publish_raw`/`subscribe`), never raw `nats` (the
  enforcement walker forbids it). Sanitize the room-id into the subject (`_sanitize`, dots→dashes).

---

## Files to Create / Modify

### Modify
- `packages/nats/src/threetears/nats/subjects.py` — add `Subjects.room(room_id) -> Subject`
  (`{ns}.channels.room.{sanitized}`) + a wildcard if a pod ever needs it. Mirror `Subjects.oplog`.
- `packages/channels/src/threetears/channels/websocket.py` —
  - a `RoomBackplane` (new class, this module or a sibling `rooms.py`): holds a `NatsClient`;
    `async publish(room_id, payload: bytes)`, `async subscribe(room_id, on_message)` /
    `async unsubscribe(room_id)` (idempotent, ref-counted per room).
  - `ConnectionRegistry.__init__(self, *, backplane: RoomBackplane | None = None)` — optional.
    `join_room` (first local member → `await backplane.subscribe(room_id, self._local_fanout)`),
    `leave_room` (last local member → `await backplane.unsubscribe(room_id)`),
    `broadcast_to_room` (backplane present → `await backplane.publish(room_id, frame)`; absent →
    today's local loop). Assign each registered socket a stable connection-id for `exclude`.
  - `WebSocketHandler.__init__(..., backplane: RoomBackplane | None = None)` — thread it into the
    `ConnectionRegistry` it creates (default `None`).
- `packages/channels/src/threetears/channels/__init__.py` — export `RoomBackplane`.

### Create
- `packages/channels/tests/unit/test_rooms.py` — backplane absent ⇒ in-process unchanged; subscribe
  ref-counting (first join subscribes, last leave unsubscribes); exclude-id honoured; frame
  encode/decode.
- `packages/channels/tests/integration/test_rooms_cross_pod.py` — **the proof**.

---

## Implementation Notes

1. **Two registries, one broker = "two pods."** The cross-pod proof needs no real second process:
   build **two `ConnectionRegistry` instances, each with its own `RoomBackplane` over the same NATS
   testcontainer**, each holding a fake local socket. Registry A `broadcast_to_room` → registry B's
   local socket receives. That *is* cross-pod (the message crossed the broker between two independent
   registries), and it's deterministic.
2. **The local fanout callback** the backplane subscribes with is `ConnectionRegistry`'s existing
   in-process `broadcast_to_room` body — factor it into `_local_fanout(room_id, frame)` so both the
   no-backplane path and the NATS-subscriber path call the same code.
3. **Frame on the wire** = the room message bytes + the optional `exclude` connection-id (a small
   JSON envelope, or headers). Keep it minimal; this is transient transport.
4. **Ref-count subscriptions** per room so N joins / N-1 leaves keeps the subscription, and the Nth
   leave drops it. Guard with a lock (the registry is touched from the WS loop + the NATS callback).

---

## Anti-patterns

- DO NOT fan locally **and** publish — publish-only; every pod (incl. sender) fans on receive. Double
  fan = duplicate delivery.
- DO NOT keep any per-process sequence/ordering here — this is transient fanout; ordering/resume is
  the op-log's job (`channels-task-03`).
- DO NOT break the no-backplane path — absent a backplane, `broadcast_to_room` MUST be exactly
  today's in-process loop (existing consumers depend on it).
- DO NOT import `nats` directly or f-string a subject — `NatsClient` + `Subjects` only (enforcement
  walker).
- DO NOT subscribe every pod to every room — subscribe only while holding a local member.

---

## Acceptance Criteria

- [ ] `Subjects.room(...)` exists, namespace-prefixed + sanitized; unit-tested.
- [ ] `ConnectionRegistry`/`WebSocketHandler` take an **optional** `backplane` (default `None`); with
      `None`, `broadcast_to_room` behaviour is **identical to before** (a regression test proves it).
- [ ] **Cross-pod proof (integration, real NATS):** registry A `broadcast_to_room("r", msg)` is
      received by registry B's local member of room "r"; a non-member of "r" on B does NOT receive it;
      `exclude` omits the named connection. (Real testcontainer NATS, not a fake.)
- [ ] Subscribe ref-counting: first `join_room` subscribes, last `leave_room` unsubscribes (asserted).
- [ ] `channels` enforcement (no direct `nats` import outside the wrapper) green; existing
      `channels` tests green unchanged.
- [ ] mypy strict + ruff clean.

---

## Verification

```bash
cd /Users/pace/crypt/pub/dev-wsl/vscode/3tears/3tears-scriob
uv run pytest packages/channels/tests/ -m "not integration" -q          # unit + backward-compat
uv run pytest -m integration packages/channels/tests/integration/test_rooms_cross_pod.py -q  # Docker up
./scripts/lint.sh && ./scripts/typecheck.sh
```

---

## Enforcement Test Suggestions

- [ ] "no per-process seq in the room fanout" — a test asserting the backplane carries no ordering
      state, so the durable-source-of-truth rule can't silently regress.
