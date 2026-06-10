# Channels: cross-pod, NATS-backed WebSocket — design + decisions

**Status:** DECIDED (2026-06-10) — design captured; build sharded below (`channels-task-01..03`).
**Driver:** Scriob (live collaboration — editor-ops + presence + co-typing on one socket), but the
enhancement is **generic** and must continue to serve existing chat consumers (aibots/metallm)
unchanged. This is a 3tears framework decision; Scriob consumes it.

> Captured here because the conclusion previously lived only in a design chat, evaporated, and got
> re-litigated. Two failures it records against, both with standing learnings: **(1)** we designed
> against a 3tears capability (`channels` "rooms with broadcast") without verifying the code — it is
> *in-process*, not cross-pod; **(2)** the decision was never written down. This file is the fix.

---

## The requirement (decided — user direction 2026-06-10)

> "The WebSocket should connect to a pod, and then be able to interact with WebSockets on other
> pods. Yes, the WebSocket is a stateful connection, but **everything else should be NATS-backed**.
> That pod goes down? The WebSocket reconnects, **nothing is lost**, things go on, **the user never
> knows**."

Distilled:
- **The socket is the only pod-local, stateful thing.** A client connects to *one* pod (via the load
  balancer) and holds a stateful WebSocket there.
- **Everything else is NATS-backed** — room membership, broadcast, presence, message ordering — so a
  socket on pod A interacts with sockets on pod B, and **any pod can serve any client**.
- **Pod loss is invisible.** Pod dies → the socket drops → the client reconnects (to *any* pod) →
  re-auths, re-joins its rooms, **resumes from its last-seen sequence** → no data lost → the user
  sees at most a reconnect blip.

---

## Current state (VERIFIED in the code, not assumed)

`packages/channels/src/threetears/channels/websocket.py`, as committed:
- **No `nats` import anywhere in the `channels` package** (grepped the whole package — zero hits).
- `ConnectionRegistry` is two in-process dicts: `self._connections: dict[str, list[ws]]`,
  `self._rooms: dict[str, list[ws]]`.
- `broadcast_to_room(room_id, message)` is `for ws in self._rooms.get(room_id, []): await
  ws.send_text(message)` — it reaches sockets **on this process only**.
- No 3tears package wires a WebSocket to NATS.

So **channels today is single-pod**: a client on pod A cannot see a client on pod B in the same room,
and a pod death loses that pod's sockets with no resume. `api.md`'s claim that channels "already give
us … rooms with broadcast" is true only for *in-process* rooms — it over-claims the cross-pod fanout
the multi-pod design needs (corrected in scriob `docs/arch/api.md`).

The cross-pod *substrate* DOES exist in 3tears, just not for sockets: `core/coordination` (NATS-KV
lease), `core/collections` (cross-pod cache-invalidation pub/sub), `epoch` (config-epoch coherence),
`nats` (KV CAS + `Subjects`). The design below composes these patterns — it does not invent new NATS
substructure.

---

## The model

```
   client ── stateful WS ──▶ POD A            POD B ◀── stateful WS ── client
                              │  local ConnectionRegistry (sockets)  │
                              │            (pod-local)               │
                              ▼                                      ▼
        ┌───────────────────────── NATS (the shared, durable plane) ─────────────────────────┐
        │  room fanout (publish room.{id} → every pod re-broadcasts to its LOCAL members)     │
        │  presence roster (KV, ephemeral, TTL+heartbeat — "who is in room X", any pod)       │
        │  ordering/durability: the op-log (R3) is the source of truth; NATS is the fast notify│
        └────────────────────────────────────────────────────────────────────────────────────┘
```

- **Cross-pod rooms = NATS fanout.** `broadcast_to_room` publishes to a room NATS subject; *every*
  pod (including the sender) subscribes to the rooms it has local members in, and on receipt fans out
  to its **local** `ConnectionRegistry` members. Mirrors the `epoch` pattern exactly: publish to a
  subject, sibling pods act locally; the durable source of truth recovers a missed notify.
- **Presence = NATS-KV roster.** A reconcile-anywhere roster keyed by room records *who* (user_id +
  pod) is in `(branch,file)` room X, with a **TTL + heartbeat** so a dead pod's members expire
  automatically (self-healing). Queryable from any pod ("who's here"). This is the net-new primitive
  `registry-task-01` already names.
- **Ordering/resume from the durable source, never in-process.** The sequence a client resumes from
  is the **op-log sequence** (editor-ops) / the durable chat cursor — NOT a per-process counter
  (that resets on restart and can't order across pods; it is the exact trap Brooks's spike fell into).
- **Reconnect is the failover story.** Socket drop → client reconnects to any pod → re-auth →
  re-join rooms (re-assert membership to the KV roster) → **resume from last-seen seq** (replay the
  op-log tail for editor-ops; re-fetch + seq for awareness). Nothing is lost because the durable state
  (op-log R3 + KV roster + NATS) outlives the pod.

---

## Backward compatibility — existing consumers MUST keep working

The enhancement is **additive and opt-in**, never a breaking change:
- `ConnectionRegistry` and `WebSocketHandler` gain an **optional NATS room-backplane** parameter
  (default `None`). **No backplane → exactly today's in-process behavior** — every current chat
  consumer (aibots/metallm) is byte-for-byte unaffected.
- **Backplane injected → cross-pod.** An existing consumer opts into multi-pod chat by wiring a NATS
  client at construction; **its message-handling code does not change** (`broadcast_to_room` just
  reaches farther). New consumers (Scriob) wire it from day one.
- The existing chat envelope (`ChannelMessage`/`ChannelResponse`/`StreamingChannelRouter`) is
  untouched; the typed editor-event frames (`channels-task-03`) are a *disjoint* frame family that
  multiplexes over the same socket, so chat and editor-ops coexist.

---

## The 3tears / scriob boundary (decided)

- **3tears `channels` owns the generic, reusable collaborative *transport*:** the stateful socket
  lifecycle + auth (`auth_validator`), **cross-pod rooms** (NATS fanout backplane), the **presence
  roster** (KV), a **typed frame/envelope** that carries a `version`/`seq` field, and the
  reconnect/resume scaffolding. App-agnostic; every 3tears app gets it.
- **Scriob owns what is ours and cannot be generic:** the op *semantics* (replace-range over
  markdown, stable position IDs — `A14`), the **op-log** (which supplies the authoritative `seq`
  value and is the durable resume source), git-as-L3, and provenance markup.
- Clean line: **channels = typed cross-pod transport + presence; scriob = op semantics + the
  authoritative durable sequencer.** The envelope carries the `seq` *field*; the op-log supplies the
  *value*.

---

## Decisions

- **D1 — Cross-pod rooms via NATS fanout.** `broadcast_to_room` publishes to a per-room NATS subject;
  each pod fans out to its local members. New `Subjects.room(...)`/`presence(...)` family in
  `3tears-nats`. Additive, backward-compatible.
- **D2 — Presence = NATS-KV roster**, ephemeral (TTL + heartbeat), keyed by room, queryable from any
  pod; self-heals on pod death.
- **D3 — The socket is the only pod-local state.** All shared state is NATS-backed; reconnect to any
  pod resumes from the durable sequence, so pod loss is transparent to the user.
- **D4 — Ordering/seq comes from the durable source** (the op-log), never an in-process counter.
- **D5 — Additive & opt-in.** Optional NATS backplane injection; absent it, behavior is exactly
  today's in-process rooms. Existing consumers are untouched and may opt in without changing their
  message handling.

---

## Build (task shards — 3tears `<area>-task-NN` convention; in this `docs/` dir)

- **`channels-task-01`: Cross-pod room broadcast backplane** — the `room.*` NATS subject family in
  `3tears-nats`; a backplane that publishes room messages and, per pod, subscribes to its rooms and
  re-fans to local `ConnectionRegistry` members; optional injection into `ConnectionRegistry` /
  `WebSocketHandler` (default `None` = in-process). Proven with a **real-NATS, two-registry**
  integration test (registry A publishes; registry B's local member receives).
- **`channels-task-02`: Cross-pod presence roster** — a KV-backed, TTL+heartbeat roster keyed by
  room; join/leave/heartbeat/query; self-heal on expiry. Real-NATS integration test: two pods, query
  sees both; kill one's heartbeat, its members expire.
- **`channels-task-03`: Typed frame dispatch + editor-event envelope + reconnect/resume** — a
  frame-type router so `editor.op`/`presence`/`cursor`/`typing` are first-class (chat unchanged); a
  generic event envelope carrying `version`/`seq`; the resume-from-seq handshake on (re)connect.

---

## Anti-patterns (the traps — do NOT repeat the spike)

- **DO NOT** keep an in-process `seq` counter — it resets on restart and cannot order across pods. The
  seq is the durable op-log sequence.
- **DO NOT** broadcast only in-process — a room message must NATS-fan to every pod with a member.
- **DO NOT** model write-authority as a per-process `threading.Lock` — that is single-pod fiction; the
  cross-pod authority is the per-repo NATS-KV lease + the op-log CAS.
- **DO NOT** break existing consumers — the backplane is optional; absent it, behavior is unchanged.
- **DO NOT** make reconnect mean "re-fetch everything over HTTP and hope" — resume is replay-from-seq
  off the durable op-log; that is the whole point of having one.
