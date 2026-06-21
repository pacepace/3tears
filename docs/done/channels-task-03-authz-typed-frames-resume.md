# channels-task-03: Authorization + typed frames + reconnect-resume (the integration)

**Status:** SHIPPED. Final channels shard (design-doc **Shard C**). Builds on task-01 (presence
state) + task-02 (room fanout). After this, scriob consumes the whole channels stack.
**Scope:** `packages/channels/src/threetears/channels/websocket.py` (+ a small `frames.py` for the
typed-frame model + injected-seam protocols). Modifies `WebSocketHandler`. Additive to existing
chat consumers (the room/authz/resume behaviour activates only when the new seams are injected).
**Origin:** `docs/channels-cross-pod-design.md` — **D4** (ordering from the durable op-log), **D8**
(authorization via `agent-acl`), and **Shard C**.

> **Premise correction (from grounding research).** task-01 (presence) and task-02 (fanout) are
> **not yet wired** to `WebSocketHandler`. Today the handler authenticates (`auth_validator` →
> `user_id`), then runs a hardcoded `if msg_type != "message": continue` (other frame types are
> **silently dropped** — a latent bug vs. "never silently drop"). It mints no `connection_id` and
> never calls `register`/`join`/`broadcast`. So this shard is **four concerns as one unit**:
> **(1) integrate** the handler ↔ presence/fanout, **(2)** a **typed-frame router**, **(3)**
> **authorization**, **(4)** **reconnect-resume**.

---

## Objective

Turn `WebSocketHandler` into the cross-pod collaboration entry point: authenticate once, mint a
stable `connection_id`, register the live socket, **authorize** every join/broadcast/op through
`agent-acl`, **route typed frames** (`message`/`join`/`leave`/`editor.op`/`cursor`/`typing`/
`presence`/`resume`) instead of the single hardcoded `message`, fan room frames cross-pod via
task-02, and **resume from the durable op-log sequence** on reconnect. Existing chat consumers
(no rooms) keep working unchanged when the new seams are absent.

---

## The 3tears / scriob boundary (decides every seam here)

channels owns the **mechanism + transport**; scriob owns the **policy + durable source**. So every
scriob-specific capability enters as an **injected seam**, never hardcoded in channels:

| Capability | channels (this shard) | scriob (later) |
|---|---|---|
| Authz **call** | invokes `agent-acl` `authorize_on_entity` (the mechanism — 3tears owns it) | provides the **`AclCache`** (built from its membership/grant loaders) + the **room→`ns_entity` resolver** (room id → the story/branch namespace entity = its policy) |
| Durable op append | routes `editor.op` to an injected **`op_handler`**, broadcasts the result | the `op_handler` = the **op-log append** (authoritative `seq`) |
| Resume replay | runs the resume handshake; pulls from an injected **`replay_source`** | the `replay_source` = **`OpLog.replay(from_seq)`** |
| Transport/state | presence (task-01), room fanout (task-02), typed frames, the seams | — |

**Backward-compatible / deliberate config (design D5):** all four seams are **optional**. With none
injected, `WebSocketHandler` behaves as today (chat `message` → router; no rooms/authz/resume). A
room-capable, authorized, resumable deployment injects them. This is a *config*, not a preserved
legacy path.

---

## Decisions (bank these; T3-D*)

- **T3-D1 — Authz = `agent-acl` mechanism in channels, policy injected from scriob.** The handler
  takes optional `acl_cache: AclCache` + `ns_resolver: Callable[[str], Awaitable[NsEntity]]`
  (room id → namespace entity). Before a **join** it calls `authorize_on_entity(ns_entity=
  await ns_resolver(room), action="room.join", user_id=<uuid>, agent_id=None, cache=acl_cache)`;
  before a **broadcast/op**, `action="entry.write"`. `AccessDenied` → an `error` frame to the
  sender; **no** presence row / **no** broadcast. Identity: `user_id` (and `customer_id`) come from
  the `auth_validator` payload as strings and are **border-converted to `UUID`** at the authz call
  (per the UUID-boundary rule). Actions are the canonical strings `agent-acl` already takes; the
  two defaults (`room.join`, `entry.write`) are overridable via handler config. **No new
  authorization path** (anti-pattern) — channels calls `agent-acl`, full stop.
- **T3-D2 — Typed-frame router replaces the hardcoded `message` check.** Inbound frames are parsed
  to a typed `Frame` (a `type` discriminator + payload) and dispatched:
  - `message` → the existing chat router path (`route_inbound`/streaming) — **unchanged**.
  - `join` / `leave` → room membership via `RoomFanout.join_room`/`leave_room` (authz `room.join`).
  - `editor.op` → the injected `op_handler` (durable), then broadcast carrying the returned `seq`.
  - `cursor` / `typing` / `presence` → **transient** room broadcast (no durability, no `seq`),
    authz `entry.write`.
  - `resume` → the resume handshake (below).
  - **unknown type → an `error` frame** (never a silent drop — fixes today's latent `continue`).
- **T3-D3 — Durable `editor.op` goes through the injected `op_handler`; transport stays channels'.**
  channels does **not** append to any op-log (that is scriob's). `op_handler(room, user_id, frame)
  -> OpResult(seq=int, ...)`; channels then `broadcast`s the op frame carrying that `seq`. The
  `seq` is the **op-log's** (D4) — channels keeps **no** counter of its own.
- **T3-D4 — Resume = a channels handshake over the injected `replay_source`.** On connect (or a
  `resume` frame) carrying `{room, last_seq}`, the handler streams `replay_source(room, last_seq)`
  to the socket (durable tail) **before** going live, so a reconnect to any pod loses nothing. The
  resume cursor is the op-log `seq` (D4), never an in-process counter. With no `replay_source`,
  resume is a no-op (chat config).
- **T3-D5 — `connection_id` is minted once per socket** (a `uuid7` string), used as the presence
  pk (task-01) and the broadcast `exclude` (so the author never receives their own frame). It lives
  only in the handler's local scope + the synchronized socket map — never a shared dict (D6).

---

## Files to Create / Modify

### Create
- `packages/channels/src/threetears/channels/frames.py`:
  - `Frame` — a typed inbound/outbound envelope (pydantic): `type: str`, `room: str | None`,
    `payload: str | None`, `seq: int | None`, plus parsing from raw JSON text and an `error`-frame
    constructor. Keep it minimal + forward-compatible (`extra="ignore"`).
  - The injected-seam **protocols** (runtime-checkable): `NsResolver`
    (`async (room_id) -> NsEntity`), `OpHandler` (`async (room_id, user_id, frame) -> OpResult`),
    `ReplaySource` (`async (room_id, from_seq) -> AsyncIterator[str]`). `OpResult` carries `seq`.
    `NsEntity` is a structural protocol exposing `id`/`customer_id`/`namespace_type`/
    `owner_agent_id` (matches what `authorize_on_entity` reads) so channels need not import a scriob
    type.

### Modify
- `packages/channels/src/threetears/channels/websocket.py` — `WebSocketHandler.__init__` gains
  optional `room_state: RoomState | None`, `room_fanout: RoomFanout | None`,
  `acl_cache: AclCache | None`, `ns_resolver: NsResolver | None`, `op_handler: OpHandler | None`,
  `replay_source: ReplaySource | None`, and overridable action strings. The connect path mints
  `connection_id`, registers the socket, optionally runs resume. `_message_loop` parses each raw
  message into a `Frame` and dispatches via the typed router (authz at each gate). The disconnect
  path leaves all joined rooms + unregisters. The chat `message` path is preserved verbatim when no
  room seams are present.
- `packages/channels/src/threetears/channels/__init__.py` — export `Frame`, the seam protocols,
  `OpResult` (sorted `__all__`).

---

## Implementation Notes

1. **Authz call shape** (T3-D1): `from threetears.agent.acl import authorize_on_entity, AccessDenied,
   AclCache`. Wrap each gate: `try: authorize_on_entity(...) except AccessDenied: send error frame;
   return`. Catch **`AccessDenied` specifically** (no broad except).
2. **One room per connection (scriob shape) but model N.** A connection's room set is tracked in the
   handler's local scope (the connection's joined rooms) so disconnect can leave each. Use the
   handler-local structure only for this connection's own rooms (not shared/queryable state — that
   is the collection).
3. **Broadcast carries the author's `connection_id` as `exclude`** so the author's own socket does
   not echo their frame back (task-02 honours it cross-pod).
4. **Resume before live.** Stream the replay tail fully, then enter `_message_loop`. If
   `replay_source` raises, surface an `error` frame and continue live (best-effort; the client can
   re-resume) — do **not** crash the socket.
5. **Reuse, don't reinvent:** presence via task-01, fanout via task-02, authz via `agent-acl`,
   subjects via `Subjects`. No raw `nats`, no f-string subjects, no new authz path, no in-process
   `seq`.

---

## Anti-patterns

- **DO NOT** hardcode `agent-acl` policy/loaders/ns-resolution into channels — inject them
  (boundary). channels calls the mechanism; scriob owns the policy.
- **DO NOT** keep an in-process `seq`/cursor — the resume cursor is the durable op-log `seq` (D4).
- **DO NOT** silently drop an unknown frame type — emit an `error` frame (fixes today's `continue`).
- **DO NOT** append to / own any op-log in channels — durability is the injected `op_handler`'s
  (scriob's) job.
- **DO NOT** broadcast only in-process — room frames fan cross-pod via task-02.
- **DO NOT** use a shared dict for any room/membership state (D6) — only the connection-local room
  set + the task-01 synchronized socket map.
- **DO NOT** break the chat path — with no room seams injected, `message` routing is byte-identical
  to today.

---

## Acceptance Criteria

- [ ] **Chat unchanged:** with no room seams injected, a `message` frame routes through the existing
      router exactly as before (regression test); unknown frame types now get an `error` frame
      (not silently dropped).
- [ ] **Typed routing:** `join`/`leave` drive presence+fanout; `cursor`/`typing`/`presence`
      transient-broadcast; `editor.op` calls the injected `op_handler` then broadcasts with its
      `seq`; `message` hits the chat router; unknown → `error`. (Unit-tested with fakes.)
- [ ] **Authorization (integration, real NATS where membership is involved):** an allowed user joins
      and broadcasts; a denied user's `join` yields an `error` frame, writes **no** presence row,
      and triggers **no** broadcast; a denied `editor.op`/broadcast is refused. (Injected authorizer
      backed by `agent-acl` `authorize_on_entity` + an `AclCache`.)
- [ ] **End-to-end cross-pod vertical slice (integration, real NATS):** two handlers over one NATS
      container (two pods), each with a fake socket; client A connects→auth→`join`→`editor.op`;
      client B (same room, other pod) **receives** the op frame (carrying the `op_handler` `seq`);
      A does **not** receive its own (exclude); a client in another room does not. This is the
      whole stack through the handler.
- [ ] **Reconnect-resume (integration):** a client reconnecting with `last_seq=N` receives the
      replayed tail from the injected `replay_source` (fed by a fake op-log) **before** any live
      frame, in `seq` order.
- [ ] **Disconnect cleanup:** dropping a socket leaves every joined room (presence row gone on both
      pods) and unregisters the local handle.
- [ ] `channels` enforcement green (no direct `nats`, no f-string subject); **all task-01 + task-02
      tests green unchanged**; mypy strict + ruff clean.

---

## Verification

```bash
cd /Users/pace/crypt/pub/dev-wsl/vscode/3tears/3tears-scriob
uv run pytest packages/channels/tests/ -m "not integration" -q          # unit incl. task-01/02 unchanged
uv run pytest -m integration packages/channels/tests/integration/ -q    # the vertical slice + authz + resume (Docker up)
./scripts/lint.sh && ./scripts/typecheck.sh
```

---

## Enforcement Test Suggestions

- [ ] "no in-process seq" — assert neither `WebSocketHandler` nor `Frame` keeps a sequence counter
      (the resume cursor is the durable op-log `seq`, D4).
- [ ] "unknown frame is not silently dropped" — assert an unknown `type` produces an `error` frame,
      so the old silent `continue` cannot regress.
- [ ] "no-seam config is the chat path" — assert that with all seams `None`, a `message` frame's
      dispatch is identical to the pre-task-03 behaviour.
