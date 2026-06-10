# Channels: cross-pod, NATS-backed, 3tears-native WebSocket — design + decisions

**Status:** DECIDED (2026-06-10) — design captured; build sharded below (`channels-task-01..03`).
**Driver:** Scriob (live collaboration — editor-ops + presence + co-typing on one socket), but the
enhancement is **generic** and must keep serving existing chat consumers (aibots/metallm) unchanged.
This is a 3tears framework decision; Scriob consumes it.

> Captured here because the conclusion previously lived only in a design chat, evaporated, and got
> re-litigated. Two recorded failures, both with standing learnings: **(1)** we designed against a
> 3tears capability (`channels` "rooms with broadcast") without verifying the code — it is
> *in-process*, not cross-pod; **(2)** the decision was never written down. This file is the fix.
>
> **Design rule for this whole effort:** *compose existing 3tears primitives; build only what
> genuinely does not exist.* No bespoke dict-based reinventions of things 3tears already ships
> (collections, RBAC, typed NATS pub/sub, subjects, leases). Every claim below is grounded in the
> actual code (file:line) — see "3tears primitives composed".

---

## The requirement (decided — user direction 2026-06-10)

> "The WebSocket should connect to a pod, and then be able to interact with WebSockets on other
> pods. Yes, the WebSocket is a stateful connection, but **everything else should be NATS-backed**.
> That pod goes down? The WebSocket reconnects, **nothing is lost**, things go on, **the user never
> knows**." — *and it must use 3tears objects everywhere they exist (collections, RBAC, …), and
> respect namespacing + RBAC.*

Distilled:
- **The socket is the only pod-local, stateful thing.** A client connects to *one* pod (via the LB)
  and holds a stateful WebSocket there. Live sockets are the one thing that genuinely cannot be
  moved off the pod.
- **Everything else is NATS-backed and lives in a 3tears primitive** — room membership/presence in a
  `BaseCollection` (L1+L2), broadcast over typed NATS pub/sub, authorization in `agent-acl`,
  ordering from the durable op-log. Any pod can serve any client.
- **Pod loss is invisible.** Pod dies → socket drops → client reconnects to *any* pod → re-auths,
  re-joins its rooms, **resumes from its last-seen sequence** → no data lost → at most a reconnect blip.

---

## Current state (VERIFIED in the code, not assumed)

`packages/channels/src/threetears/channels/websocket.py`, as committed:
- **No `nats` import anywhere in the `channels` package** (grepped the whole package — zero hits).
- `ConnectionRegistry` is two in-process dicts (`_connections`, `_rooms`); `broadcast_to_room`
  iterates `self._rooms.get(room_id, [])` and `send_text`s — **this pod's sockets only**.
- No 3tears package wires a WebSocket to NATS.

So **channels today is single-pod**: a client on pod A cannot reach a member on pod B, and a pod
death loses that pod's sockets with no resume. api.md's "channels … already give us rooms with
broadcast" is true only for *in-process* rooms — it over-claims the cross-pod fanout the multi-pod
design needs (corrected in scriob `docs/arch/api.md`).

The cross-pod substrate already exists in 3tears — just not wired to sockets. We compose it.

---

## 3tears primitives composed (use what exists; build only the gap)

| Need | 3tears primitive (file) | Reuse or new |
|---|---|---|
| Pod-local live sockets | `channels.ConnectionRegistry` (dicts of *sockets*) | **reuse** — sockets are legitimately pod-local |
| Socket lifecycle + auth | `channels.WebSocketHandler` + `auth_validator` | **reuse** |
| Cross-pod presence / room-membership roster | `core.collections.BaseCollection` (L1+L2-only), mirroring `registry/heartbeat_collection.py` `HeartbeatCollection` | **reuse the framework**; new `PresenceCollection` subclass |
| Cross-pod state coherence | the collection **invalidation envelope** (`core.collections.registry.INVALIDATION_SUBJECT`) — built into `BaseCollection`'s L2 write path | **reuse** (free with the collection) |
| Self-heal on pod death | `date_last_heartbeat` timestamp + a sweep loop, exactly like `registry/health.py` `HeartbeatSubscriber` (NOT raw KV-TTL) | **reuse the pattern** |
| Cross-pod room **message fanout** (live delivery to sockets) | thin pub/sub on the `NatsClient` wrapper + a new `Subjects.room(...)` family | **NEW** — the one piece that genuinely does not exist |
| Subjects / namespace prefixing | `nats.Subjects` (`{ns}.…`) + `NatsClient.kv_bucket` (`{ns}-…`) | **reuse** |
| Authorization (join / broadcast / op) | `agent.acl.authorize_on_entity(ns_entity, action, user_id, agent_id, cache)` → `AccessDenied` | **reuse** |
| Cross-pod ACL coherence | `agent.acl.invalidation` + the `Subjects.acl_invalidate(...)` family | **reuse** |
| Logging / tracing | `observe.get_logger` / `@traced` | **reuse** |

**The only net-new mechanism is the room *fanout*** (live message delivery to sockets) — it is not
cache invalidation (that's the collection's job) and no 3tears primitive provides it, so it is built,
thinly, *in channels*. Everything else is composition.

---

## The model

```
   client ── stateful WS ──▶ POD A                        POD B ◀── stateful WS ── client
                              │ ConnectionRegistry (LOCAL live sockets — the only pod-local state)
                              ▼                                          ▼
   ┌──────────────────────────────── NATS (the shared plane) ─────────────────────────────────┐
   │  • room fanout      Subjects.room(ns, room) → publish; every pod with a local member       │
   │                     subscribes + delivers to ITS sockets (thin pub/sub; the only new bit)  │
   │  • presence roster  PresenceCollection : BaseCollection (L1+L2) keyed by room — who is in   │
   │                     room X across pods; heartbeat-timestamp + sweep self-heals on pod death │
   │  • authorization    agent-acl authorize_on_entity(story/branch ns, action, user) gates      │
   │                     join / broadcast / op — RBAC + namespace, cross-pod coherent            │
   │  • ordering/resume   the op-log (R3) is the durable source of truth; NATS is the fast notify │
   └───────────────────────────────────────────────────────────────────────────────────────────┘
```

- **Cross-pod rooms = NATS fanout.** `broadcast_to_room` publishes to `Subjects.room(...)`; every pod
  with a local member subscribes (on first local join, ref-counted) and delivers to *its* sockets.
  One delivery path (publish-only; every pod incl. the sender fans on receive — no double delivery).
  Mirrors the `epoch` publish→siblings-act-locally pattern.
- **Presence/membership = a `BaseCollection` (L1+L2), not a hand-rolled roster.** A `PresenceCollection`
  subclass keyed by room, modelled exactly on `HeartbeatCollection` (L1+L2-only, `l3_pool=None`,
  L3 methods raise). Cross-pod coherence rides the collection's invalidation envelope; "who is in
  room X" is a collection query, answerable from any pod. **Self-heal** is the heartbeat pattern: each
  membership row carries `date_last_heartbeat`, the socket lifecycle refreshes it, and a presence
  sweep loop (modelled on `HeartbeatSubscriber`) evicts rows whose heartbeat went stale — so a dead
  pod's members disappear automatically.
- **Authorization is `agent-acl`, woven through.** Identity comes from the `auth_validator` (user_id,
  established once on connect). **Every** join, broadcast target, and editor-op is gated by
  `authorize_on_entity(ns_entity=<story/branch>, action=<canonical>, user_id=…, cache=AclCache)` —
  `AccessDenied` → error frame / refused op. A presence row is only written *after* a successful
  `room.join` authorization, so the roster cannot contain an unauthorized member.
- **Ordering/resume from the durable source, never in-process.** The resume sequence is the **op-log
  sequence** (editor-ops) / the durable chat cursor — never a per-process counter (resets on restart,
  cannot order across pods — the spike's exact trap).
- **Reconnect is the failover story.** Socket drop → reconnect to any pod → re-auth → re-authorize +
  re-join rooms (re-assert the presence row) → **resume from last-seen seq** (replay the op-log tail).
  Nothing is lost: the durable state (op-log R3 + the PresenceCollection's L2 + NATS) outlives the pod.

---

## Namespacing (structural isolation, via the primitives — not bolted on)

Isolation is enforced at three composed layers, none bespoke:
- **Environment / deployment** — every NATS subject is namespace-prefixed by `Subjects` (`{ns}.…`,
  bound at `NatsClient.connect`); every KV bucket is `{ns}-…`; the `PresenceCollection`'s L2 bucket
  inherits the prefix. Two envs on one NATS cluster **cannot** cross streams.
- **Tenant** — authorization and presence are scoped to the **ACL namespace** (`customer_id` +
  `namespace_type` on the `ns_entity`); `agent-acl` is namespace-native. The tenant boundary and the
  authorization boundary are the **same object**, so a user in tenant A can never be authorized into,
  or appear present in, tenant B.
- **Resource** — the room id encodes story/workspace + branch + file (`{story}:{branch}:{file}`), so
  story A's `(branch,file)` rooms are disjoint from story B's even within one tenant + env.

So a room id is qualified `{ns}` (subject/bucket) · tenant (ACL namespace) · resource
(story/branch/file). Leakage is impossible by construction, not by remembering to filter.

---

## RBAC / authorization (CO-5, via `agent-acl` — not a new check)

- **Identity** is established once, on connect, by the shared `auth_validator` (the same callable the
  HTTP side uses — one validation path, ID-3) → `user_id` (+ claims). No per-message re-auth of
  identity; the socket carries the authenticated principal.
- **Authorization** is per-action, every time, via `agent.acl.authorize_on_entity`:
  - `room.join` (presence) — may this user be present in this `(branch,file)` room?
  - `entry.read` / `entry.write` — gates which editor-ops/broadcasts a user may send/receive (CO-5
    per-story/branch ACL; the same gate the write path already names, api.md §6).
  - Deny → `AccessDenied` → an `error` frame to that socket / a refused op; never a silent allow.
- **Cross-pod coherence** — a revoked grant propagates to every pod via `agent.acl.invalidation`
  (`Subjects.acl_invalidate(...)`) before the next check, so authorization is consistent fleet-wide.
- **The roster is authorization-clean by construction** — a member is only written to the
  `PresenceCollection` after `room.join` passed, so "who is in room X" never lists an unauthorized user.

---

## Backward compatibility — existing consumers MUST keep working

Additive and opt-in, never breaking:
- The cross-pod machinery (room backplane + presence + authz) is injected; **absent it,
  `ConnectionRegistry`/`WebSocketHandler` behave exactly as today** (in-process rooms). Every current
  chat consumer (aibots/metallm) is byte-for-byte unaffected.
- A consumer opts into cross-pod by wiring the NATS-backed pieces at construction; **its
  message-handling code does not change** — `broadcast_to_room` just reaches farther. New consumers
  (Scriob) wire them from day one.
- The chat envelope (`ChannelMessage`/`ChannelResponse`/`StreamingChannelRouter`) is untouched; typed
  editor-event frames (`channels-task-03`) are a *disjoint* frame family multiplexed over the same
  socket, so chat + editor-ops coexist.

---

## The 3tears / scriob boundary (decided)

- **3tears owns the generic, reusable collaborative *transport + state*:** the socket + auth
  (`channels`), cross-pod **room fanout** (`channels` + `nats`), the **presence/membership roster**
  (`core.collections`), **authorization** (`agent-acl`), namespacing (`nats.Subjects`/`kv`), and the
  reconnect/resume scaffolding. App-agnostic; every 3tears app gets it.
- **Scriob owns what is ours and cannot be generic:** the op *semantics* (replace-range over
  markdown, stable position IDs — `A14`), the **op-log** (the authoritative `seq` value + durable
  resume source), git-as-L3, provenance markup, and the per-story ACL *policy* (the roles/grants;
  the *mechanism* is `agent-acl`).
- Clean line: **channels/collections/acl = generic cross-pod transport + state + authz; scriob = op
  semantics + the authoritative durable sequencer + the policy.**

---

## Decisions

- **D1 — Cross-pod rooms via NATS fanout.** `broadcast_to_room` publishes to `Subjects.room(...)`;
  each pod delivers to its local members. Additive, backward-compatible.
- **D2 — Presence/membership = `BaseCollection` (L1+L2), modelled on `HeartbeatCollection`.** Cross-pod
  coherence via the invalidation envelope; self-heal via heartbeat-timestamp + a sweep loop. **No
  bespoke KV roster, no dicts for shared state.**
- **D3 — The socket is the only pod-local state.** Everything shared is in a 3tears primitive backed by
  NATS; reconnect to any pod resumes from the durable sequence — pod loss is transparent.
- **D4 — Ordering/seq comes from the durable source** (the op-log), never an in-process counter.
- **D5 — Additive & opt-in.** Absent the injected NATS pieces, behavior is exactly today's in-process
  rooms; existing consumers untouched.
- **D6 — Namespacing is structural** via `Subjects`/`kv` prefixes (env) + the ACL namespace (tenant) +
  the scoped room id (resource). Not a filter we remember to apply.
- **D7 — Authorization is `agent-acl`** (`authorize_on_entity`) on every join/broadcast/op (CO-5),
  cross-pod coherent via `agent-acl.invalidation`. Identity from the shared `auth_validator`.

---

## Build (task shards — 3tears `<area>-task-NN` convention; in this `docs/` dir)

- **`channels-task-01`: Cross-pod room fanout** — `Subjects.room(...)` + the thin publish→per-pod-local-
  deliver backplane + optional injection into `ConnectionRegistry`/`WebSocketHandler` (default `None`
  = in-process). Proven with a real-NATS, two-registry cross-pod integration test. *(The one net-new
  mechanism.)*
- **`channels-task-02`: `PresenceCollection` (BaseCollection L1+L2) + presence sweep** — subclass
  `BaseCollection`, L1+L2-only, mirroring `HeartbeatCollection`; keyed by room; join/leave/heartbeat/
  query; a sweep loop (modelled on `HeartbeatSubscriber`) evicts stale members. Real-NATS integration
  test: two pods see each other; kill one's heartbeat, its members expire.
- **`channels-task-03`: Authorization + typed frames + reconnect-resume** — gate join/broadcast/op with
  `authorize_on_entity`; a frame-type router so `editor.op`/`presence`/`cursor`/`typing` are first-class
  (chat unchanged); a generic event envelope carrying `version`/`seq`; the resume-from-seq handshake.

---

## Anti-patterns (the traps — verified against the spike + this study)

- **DO NOT hand-roll the roster** with dicts or a raw `NatsKvBucket` — it is a `BaseCollection`
  (L1+L2), modelled on `HeartbeatCollection`. Reinventing it is the exact "bespoke dict thing" to avoid.
- **DO NOT write a new authorization path** — join/broadcast/op go through `agent-acl`
  `authorize_on_entity`. CO-5 is policy on top of that mechanism, not a fresh check.
- **DO NOT keep an in-process `seq`** — it resets on restart and cannot order across pods. The seq is
  the durable op-log sequence.
- **DO NOT broadcast only in-process** — a room message must NATS-fan to every pod with a member.
- **DO NOT skip namespacing** — every subject/bucket/room id is namespace + tenant + resource scoped.
- **DO NOT break existing consumers** — the cross-pod pieces are optional; absent them, behavior is
  unchanged.
- **DO NOT make reconnect "re-fetch everything over HTTP and hope"** — resume is replay-from-seq off
  the durable op-log; that is the whole point of having one.
