# channels-task-02: Cross-pod room fanout (NATS-backed live delivery)

**Status:** READY. Builds on `channels-task-01` (the presence state layer: `PresenceCollection`
+ `RoomState`). `channels-task-03` (authorization + typed frames + reconnect-resume) builds on the
`Subjects.room(...)` family and the fanout seam this lands.
**Scope:** a new `room` subject family in `3tears-nats` (`subjects.py`) + a new thin fanout
backplane in `3tears-channels` (`presence/` package), composing the already-built `RoomState`.
Net-new framework code.
**Origin:** `docs/channels-cross-pod-design.md` — **D1** (cross-pod rooms via NATS fanout) and
**Shard B**. This is the **one genuinely net-new mechanism** in the cross-pod design: no 3tears
primitive provides live room message delivery (the collections give cross-pod *state*; this gives
cross-pod *fanout*).

> **Premise correction (vs. the pre-task-01 draft of this shard).** There is **no existing
> in-process `broadcast_to_room` to preserve.** `ConnectionRegistry` (in `websocket.py`) is only
> `WebSocketHandler`'s pod-local *per-user* live-handle map (`user_id → [sockets]`); it has no
> room membership, join/leave, or room broadcast. Room state now lives in `RoomState`
> (task-01: `local_sockets(room_id) -> list[socket]`, the synchronized handle map) backed by
> `PresenceCollection`. So this task is **net-new room delivery built on `RoomState`**, not an
> "opt-in backplane wrapped around a legacy dict path." Single-pod / NATS-free remains a
> deliberate *config* (no fanout wired), per design **D5** — not a per-call fallback to keep alive.

---

## Objective

Deliver a room message to every member of a room **on every pod**, via NATS fanout: publish the
message once to a per-room NATS subject; **every pod that holds ≥1 local member of that room**
receives it and delivers to **its own** live sockets (resolved through `RoomState.local_sockets`).
This is **transient fast-notify fanout only** — durable ordering/replay is the op-log's job
(scriob), never this layer.

**Out of scope (other shards — reuse, do NOT hand-roll here):** membership/presence roster is
task-01 (done — `PresenceCollection`, a `BaseCollection`, **not** a raw KV roster or dict);
**authorization** of join/broadcast is task-03 (`agent-acl` `authorize_on_entity`); durable
ordering/resume is the op-log (scriob). Subjects go through `Subjects.room(...)`.

---

## Design constraints (the fanout must stay single-purpose and correct)

- **Build on `RoomState`, don't duplicate state.** The fanout holds **no** room membership of its
  own. It resolves local delivery targets through `RoomState.local_sockets(room_id)` and the
  membership it already maintains. The fanout's only own state is its per-room **subscription
  ref-count** (transport bookkeeping, not domain state).
- **One delivery path — publish, then every pod fans locally (incl. the sender's pod).** Mirror
  the `epoch` pattern (`EpochClient.bump` publishes → sibling `EpochListener`s act locally).
  `broadcast(room_id, …)` **only publishes** to `Subjects.room(room_id)`. The per-pod subscriber
  (including the **sender's own pod**, via NATS echo) receives it and delivers to that pod's local
  sockets. **Do NOT also fan locally at publish time** — that double-delivers to local members.
  > **No-echo trap:** same-pod delivery to the sender's *other* local members relies on the
  > sender's own NATS subscription receiving its own published message. Verify `NatsClient` does
  > not suppress echo on this subscription; the integration proof asserts a same-pod non-sender
  > member receives (below), which catches a no-echo regression.
- **Subscribe on first local member, unsubscribe on last (ref-counted, lock-guarded).** A pod
  subscribes to a room's subject only while it holds ≥1 local member (ref `0→1` on join);
  it unsubscribes when the last local member leaves (ref `1→0`). Pods never subscribe to rooms
  they have no one in. The ref-count + subscription handle map is guarded by an `asyncio.Lock`
  (touched from the WS loop and the NATS callback).
- **`exclude` survives the cross-pod hop, by connection-id.** `broadcast(room_id, payload,
  exclude=connection_id)` must omit exactly that one connection wherever it lives. The published
  envelope carries the **exclude connection-id**; each pod honours it against its own members.
  A `None` exclude delivers to everyone. (The typical caller excludes the originating socket so the
  author does not receive their own frame.)
- **Transient, best-effort.** A dropped room message is **not** recovered here — the durable layer
  (op-log resume, task-03 / scriob) does that, exactly as `epoch` recovers a missed bump via the
  next tick. **No durability/replay, and no per-process `seq`/ordering** in this fanout (design
  **D4**: the seq is the durable op-log sequence).
- **Tenancy-ready (design D7).** The room id is `{customer}:{story}:{branch}:{file}`; it is the
  unit the subject is derived from, so rooms are disjoint across customer/story/resource. The
  subject is also `{ns}.`-prefixed (env isolation) by `Subjects`.
- **Reuse, don't reinvent.** Subjects through `threetears.nats.Subjects` (**never** an f-string).
  NATS pub/sub through the `NatsClient` wrapper using the **same typed pub/sub the `epoch` module
  uses** (`subscribe_typed` + the wrapper's typed publish) — **never** raw `nats` (the enforcement
  walker forbids it).

---

## Files to Create / Modify

### Modify
- `packages/nats/src/threetears/nats/subjects.py` — add `Subjects.room(room_id: str) -> Subject`,
  `{ns}.channels.room.{token}`. **The room id is arbitrary** (`:` separators; app-supplied
  branch/file segments may contain `.`, spaces, `*`, `>` — all illegal or ambiguous in a NATS
  subject token). Derive the token as a **SHA-256 hex digest of the room id** (subject-safe:
  `[0-9a-f]` only; collision-resistant; deterministic) — the same robustness move task-01 used for
  the KV key. The raw room id rides in the wire envelope, so reversibility is not needed. Mirror the
  classmethod shape + namespace-prefix resolution of `Subjects.oplog` / `Subjects.agent_heartbeat`.
  Unit-test it.
- `packages/channels/src/threetears/channels/presence/room_state.py` — if needed for a precise
  exclude, add `async local_member_sockets(room_id: str) -> list[tuple[str, Any]]` returning
  `(connection_id, socket)` pairs for this pod's members (snapshot under the existing `_lock`), so
  the fanout can skip the excluded connection-id without socket-identity guesswork. (If
  `local_sockets` + `local_socket(exclude)` identity-skip is cleaner, that is acceptable — but the
  pair-returning method is the recommended, directly-testable shape.)
- `packages/channels/src/threetears/channels/__init__.py` — export the new fanout class + wire model.

### Create
- `packages/channels/src/threetears/channels/presence/fanout.py` — `RoomFanout` (thin backplane):
  - `__init__(self, room_state: RoomState, nats_client: NatsClient)`.
  - `async join_room(room_id, connection_id, user_id, customer_id)` — `await room_state.join(...)`;
    ref `0→1` ⇒ `await self._subscribe(room_id)`.
  - `async leave_room(room_id, connection_id)` — `await room_state.leave(...)`; ref `1→0` ⇒
    `await self._unsubscribe(room_id)`.
  - `async broadcast(room_id, payload: str, *, exclude: str | None = None)` — publish a `RoomFrame`
    to `Subjects.room(room_id)`. **Publish-only** (no local fan here).
  - `async _deliver(frame: RoomFrame)` — the subscription callback: resolve this pod's local member
    sockets for `frame.room_id`, send `frame.payload` to each, **skipping** `frame.exclude`.
  - `_subscribe` / `_unsubscribe` — idempotent, ref-counted, `asyncio.Lock`-guarded; hold the
    subscription handle so unsubscribe actually tears down.
- `packages/channels/src/threetears/channels/presence/wire.py` (or alongside `fanout.py`) —
  `RoomFrame` pydantic model (mirror `epoch/wire.py:EpochBumpMessage`): `room_id: str`,
  `payload: str`, `exclude: str | None = None`, `origin_pod: str` (diagnostics). Keep it minimal —
  transient transport.
- `packages/channels/tests/unit/channels/presence/test_fanout.py` — ref-count subscribe/unsubscribe
  (first `join_room` subscribes, last `leave_room` unsubscribes; N joins / N−1 leaves keeps it);
  `broadcast` publishes exactly once and does **not** locally fan at publish time; `_deliver` skips
  the excluded connection-id; `RoomFrame` encode/decode round-trip. (L1-only / fake-NATS unit
  level — the cross-pod truth is the integration proof.)
- `packages/channels/tests/integration/channels/presence/test_room_fanout_cross_pod.py` —
  **the proof** (real NATS testcontainer, the task-01 `two_pods` fixture pattern).

---

## Implementation Notes

1. **Two pods, one broker (reuse task-01's `two_pods` fixture).** The cross-pod proof needs no
   second process: two `RoomState` + `RoomFanout` triples over one NATS testcontainer (extend the
   existing `conftest.py` `Pod` to carry a `RoomFanout`). Pod A `broadcast` → pod B's local socket
   receives. That *is* cross-pod (the frame crossed the broker between two independent registries).
2. **Local delivery resolves through task-01.** `_deliver` calls `room_state.local_member_sockets(
   frame.room_id)` (or `local_sockets` + exclude resolution) — it does **not** re-query the
   collection itself or keep its own member set.
3. **Fake socket** = a tiny object with an awaitable `send_text(str)` capturing delivered frames
   (same shape task-01's tests use for live handles).
4. **Ref-count under the lock** so concurrent join/leave on the same pod can't double-subscribe or
   tear down a still-needed subscription.
5. **Typed wire**: publish/subscribe `RoomFrame` via the wrapper's typed pub/sub exactly as
   `epoch` does (`subscribe_typed(subject, message_type=RoomFrame, cb=_deliver)`); do not hand-roll
   JSON or touch `nats` directly.

---

## Anti-patterns

- **DO NOT fan locally AND publish** — publish-only; every pod (incl. sender, via echo) fans on
  receive. Double fan = duplicate delivery.
- **DO NOT keep room membership in the fanout** — it composes `RoomState` (task-01). The fanout's
  only state is the per-room subscription ref-count.
- **DO NOT keep any per-process `seq`/ordering** here — transient fanout; ordering/resume is the
  op-log's job (design D4 / task-03 / scriob).
- **DO NOT subscribe every pod to every room** — subscribe only while holding a local member.
- **DO NOT import `nats` directly or f-string a subject** — `NatsClient` typed pub/sub + `Subjects`
  only (enforcement walker).
- **DO NOT reintroduce a dict for shared/queryable room state** (design D6) — the live-socket handle
  map (task-01, lock-synchronized) is the *only* in-process structure.

---

## Acceptance Criteria

- [ ] `Subjects.room(room_id)` exists: `{ns}`-prefixed, SHA-256-token (subject-safe for arbitrary
      room ids — colons, dots, spaces), deterministic; unit-tested (incl. two distinct room ids →
      distinct subjects; an out-of-grammar room id → a valid subject).
- [ ] `RoomFanout.broadcast` **publishes once** and does **not** locally fan at publish time
      (asserted: a unit test proves no local `send_text` happens on the publish path).
- [ ] **Cross-pod proof (integration, real NATS):** with pod A holding members `a1` (the sender,
      `exclude=a1`) and `a2`, and pod B holding member `b1` of the same room, and a non-member `b2`
      on B in a *different* room — `fanout_A.broadcast(room, payload, exclude="a1")` results in:
      `a2` **receives** (same-pod delivery via echo), `b1` **receives** (cross-pod), `a1` does
      **NOT** (excluded), `b2` does **NOT** (non-member). (Real testcontainer NATS, not a fake.)
- [ ] **Subscribe ref-counting:** first `join_room` on a pod subscribes to the room subject; the
      last `leave_room` unsubscribes (asserted, including N-join / N−1-leave keeps the subscription).
- [ ] `channels` enforcement (no direct `nats` import outside the wrapper) green; **all existing
      `channels` tests (task-01 unit + integration) green unchanged**.
- [ ] mypy strict + ruff clean.

---

## Verification

```bash
cd /Users/pace/crypt/pub/dev-wsl/vscode/3tears/3tears-scriob
uv run pytest packages/channels/tests/ -m "not integration" -q          # unit incl. task-01 unchanged
uv run pytest -m integration packages/channels/tests/integration/channels/presence/ -q  # Docker up
uv run pytest -m integration packages/nats/tests/ -q                     # Subjects.room, if integration-touched
./scripts/lint.sh && ./scripts/typecheck.sh
```

---

## Enforcement Test Suggestions

- [ ] "no per-process seq in the fanout" — a test asserting `RoomFrame` / `RoomFanout` carry no
      ordering/sequence state, so the durable-source-of-truth rule (D4) can't silently regress.
- [ ] "publish-only" — a unit test asserting `broadcast` performs exactly one publish and zero
      local deliveries, so the no-double-fan rule can't regress.
