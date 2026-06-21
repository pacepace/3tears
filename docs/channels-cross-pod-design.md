# Channels: cross-pod, NATS-backed, 3tears-native WebSocket — design + decisions

**Status:** DECIDED (2026-06-10) — design captured; build sharded below.
**Driver:** Scriob (live collaboration — editor-ops + presence + co-typing on one socket), but the
enhancement is **generic** 3tears framework work; Scriob is its first full consumer, and existing chat
consumers (aibots/metallm) migrate to it. This is a 3tears framework decision.

> Captured because the conclusion previously lived only in a design chat, evaporated, and got
> re-litigated. Standing failures it records against: **(1)** designed against a 3tears capability
> (`channels` "rooms with broadcast") without verifying the code — it is *in-process*, not cross-pod;
> **(2)** the decision was never written down. This file is the fix.

> **Design rules for this whole effort:**
> 1. **Compose existing 3tears primitives; build only what genuinely does not exist.** No bespoke
>    reinventions of things 3tears already ships (collections, RBAC, typed NATS pub/sub, subjects,
>    leases). Every claim below is grounded in actual code.
> 2. **Do the structurally-best thing — backward-compat is a *consideration, not a constraint*.** If
>    the right shape changes `channels` for aibots/metallm, change it and migrate them (we own the
>    consumers). Change because it's right, never gratuitously.
> 3. **No dicts for shared/queryable state.** This stack is **async *and* threaded** (asyncio +
>    `run_in_threadpool`); a `dict` for shared state races (iterate-while-mutate across awaits +
>    threadpool) and is slow at scale. Shared/queryable/growable state goes in the concurrency-safe
>    store (the collection: **L1 SQLite + L2 NATS**, indexed). The one exception is a minimal,
>    synchronized map of `connection_id → live socket` (a non-serializable handle).
> 4. **Build tenancy-ready.** Thread the tenant dimension (`customer_id`, the ACL namespace) through
>    everything now, with all users under **one fixed customer**; multi-tenancy later = issue real
>    customer ids, **zero re-architecture**.

---

## The requirement (decided — user direction 2026-06-10)

> "The WebSocket should connect to a pod, and then be able to interact with WebSockets on other pods.
> Yes, the WebSocket is a stateful connection, but **everything else should be NATS-backed**. That pod
> goes down? The WebSocket reconnects, **nothing is lost**, things go on, **the user never knows**."

Distilled:
- **The socket is the only pod-local, stateful thing** — a live connection genuinely cannot be moved
  off its pod. Everything else lives in a concurrency-safe, NATS-backed 3tears primitive.
- **Any pod serves any client.** Membership/presence in a `BaseCollection` (L1 SQLite + L2 NATS),
  broadcast over typed NATS pub/sub, authorization in `agent-acl`, ordering from the durable op-log.
- **Pod loss is invisible.** Pod dies → socket drops → client reconnects to *any* pod → re-auths,
  re-joins, **resumes from its last-seen sequence** → no data lost → at most a reconnect blip.

---

## Current state (VERIFIED in the code, not assumed)

`packages/channels/src/threetears/channels/websocket.py`, as committed:
- **No `nats` import anywhere in the `channels` package.**
- `ConnectionRegistry` is two in-process **dicts** (`_connections`, `_rooms`). `broadcast_to_room`
  iterates `self._rooms.get(room_id, [])` **with `await` inside the loop** while `join_room`/`leave_room`
  mutate the same dict — single-pod **and** concurrency-unsafe (iterate-while-mutate race).
- No 3tears package wires a WebSocket to NATS.

So channels today is single-pod, racy, and unresumable. api.md's "channels … already give us rooms
with broadcast" is true only for *in-process* rooms — it over-claims the cross-pod fanout the design
needs (corrected in scriob `docs/arch/api.md`). We **replace** this state layer (not preserve it) and
**compose** the cross-pod substrate 3tears already ships.

---

## 3tears primitives composed (use what exists; build only the gap)

| Need | 3tears primitive | Reuse / new |
|---|---|---|
| Cross-pod room-membership + presence state | `core.collections.BaseCollection` (L1 SQLite + L2 NATS, **pk-keyed**), modelled on `registry/heartbeat_collection.py` `HeartbeatCollection` | **reuse the framework**; new pk-keyed `PresenceCollection` (per-connection entry + room-index) — concurrency-safe, **replaces the dicts** |
| Cross-pod state coherence | the collection **invalidation envelope** (`core.collections.registry.INVALIDATION_SUBJECT`) | **reuse** (free with the collection) |
| Self-heal on pod death | `date_last_heartbeat` + a sweep loop, like `registry/health.py` `HeartbeatSubscriber` (NOT raw KV-TTL) | **reuse the pattern** |
| Pod-local live-socket handles | a **minimal, synchronized** `connection_id → live socket` map (asyncio.Lock; snapshot-iterate) | **NEW, minimal** — the one unavoidable in-process structure (non-serializable handles); **not a dict for shared state** |
| Cross-pod room **message fanout** (live delivery) | thin pub/sub on the `NatsClient` wrapper + a new `Subjects.room(...)` family | **NEW** — the only mechanism that genuinely does not exist |
| Socket lifecycle + auth | `channels.WebSocketHandler` + `auth_validator` | **reuse** (reshaped as needed) |
| Subjects / namespace prefixing | `nats.Subjects` (`{ns}.…`) + `kv_bucket` (`{ns}-…`) | **reuse** |
| Authorization (join / broadcast / op) | `agent.acl.authorize_on_entity(ns_entity, action, user_id, agent_id, cache)` | **reuse** |
| Cross-pod ACL coherence | `agent.acl.invalidation` + `Subjects.acl_invalidate(...)` | **reuse** |
| Logging / tracing | `observe.get_logger` / `@traced` | **reuse** |

Net-new: the **live room fanout** (message delivery to sockets — not cache invalidation, which the
collection owns) and the **minimal synchronized socket-handle map**. Everything else is composition.

---

## The model

```
   client ── stateful WS ──▶ POD A                        POD B ◀── stateful WS ── client
                              │ minimal synced connection_id→socket map (live handles; the only
                              │ in-process state — NOT a dict for shared/queryable data)
                              ▼                                          ▼
   ┌──────────────────────────── NATS (the shared plane) + the collections ─────────────────────────┐
   │  • membership + presence   PresenceCollection : BaseCollection (L1 SQLite + L2 NATS), keyed by    │
   │                            (customer, story, branch, file) room — concurrency-safe, indexed,      │
   │                            queryable from any pod; heartbeat + sweep self-heals on pod death      │
   │  • room fanout             Subjects.room(...) → publish; every pod with a local member subscribes │
   │                            + delivers to its sockets (thin pub/sub; the only net-new mechanism)   │
   │  • authorization           agent-acl authorize_on_entity(ns=story/branch under customer, action,  │
   │                            user) gates join / broadcast / op — RBAC + tenant, cross-pod coherent  │
   │  • ordering / resume       the op-log (R3) is the durable source of truth; NATS is the fast notify │
   └────────────────────────────────────────────────────────────────────────────────────────────────┘
```

- **Membership/presence/rooms = `BaseCollection`(s), not dicts — and PK-KEYED (verified constraint).**
  `BaseCollection` is **pk-keyed only** (no secondary-field query — verified in the code), its L2
  (NATS-KV) coherence is **eviction-based**, and the KV wrapper exposes **no key-listing**; the
  secondary-query variant (`SchemaBackedCollection`) is **L3/postgres-backed** (wrong for ephemeral
  presence). So "who is in room X" must be a **pk-get, never a secondary scan.** Model it pk-keyed:
  a **per-connection entry** (`pk = connection_id`; carries the heartbeat — no contention) plus a
  **room-index entry** (`pk = room` = `{customer}:{story}:{branch}:{file}`; the member set), updated
  only on join/leave. "Who's in room X" = `get(room-index)` → its members (each then a pk-get). Both
  are L1+L2-only (`l3_pool=None`, like `HeartbeatCollection`): concurrency-safe (SQLite L1, no
  iterate-while-mutate race), cross-pod via the invalidation envelope. CAS contention is on
  join/leave only (low churn), not on heartbeats. **Self-heal**: each connection entry carries
  `date_last_heartbeat`; a sweep loop (modelled on `HeartbeatSubscriber`) evicts stale connections
  and prunes them from the room-index — a dead pod's members vanish automatically.
- **The only pod-local structure is a minimal, synchronized `connection_id → live socket` map.** Live
  sockets are non-serializable handles that must live on the pod. The map is bounded by concurrent
  connections on this pod, guarded by an `asyncio.Lock` (event-loop-confined), and **iterated over a
  snapshot** — never mutated while awaiting. Delivery: query the collection for the local connection-ids
  in room X, then resolve handles from this map.
- **Cross-pod rooms = NATS fanout.** `broadcast_to_room` publishes to `Subjects.room(...)`; every pod
  with a local member subscribes (on first local join, ref-counted) and delivers to *its* sockets.
  One delivery path (publish-only; every pod incl. sender fans on receive — no double delivery), the
  `epoch` pattern.
- **Authorization is `agent-acl`, woven through.** Identity from the shared `auth_validator` (user_id,
  once on connect). Every join/broadcast-target/op is gated by `authorize_on_entity(ns_entity=<story/
  branch under the customer>, action=<canonical>, user_id=…, cache=AclCache)` → `AccessDenied` → error
  frame / refused op. A presence row is written only **after** `room.join` passes — the roster is
  authorization-clean by construction.
- **Ordering/resume from the durable source.** The resume sequence is the **op-log sequence** /
  durable chat cursor — never an in-process counter (resets on restart, can't order cross-pod).
- **Reconnect is the failover story.** Drop → reconnect to any pod → re-auth → re-authorize + re-join
  (re-assert the presence row) → **resume from last-seen seq** (replay the op-log tail). Nothing is
  lost: the durable state (op-log R3 + the collection's L2 + NATS) outlives the pod.

---

## Namespacing + tenancy (structural; built tenancy-ready under one customer)

Isolation is enforced at composed layers, none bespoke, and the tenant dimension is plumbed from day one:
- **Environment / deployment** — every subject is `{ns}.…` (`Subjects`, bound at connect); every bucket
  is `{ns}-…`; the collection's L2 inherits both. Two envs on one NATS cluster cannot cross.
- **Tenant** — authorization and presence are scoped to the **ACL namespace** (`customer_id` +
  `namespace_type` on `ns_entity`). `agent-acl` is namespace-native, so **the tenant boundary and the
  authorization boundary are the same object**. **Scriob runs all users under one fixed `customer_id`
  now** — but `customer_id` is threaded through room ids, the `PresenceCollection` keys, and every authz
  call from the start. **Adding multi-tenancy later = issue real `customer_id`s; no schema or
  architecture change.** (This is the right seam, built now because it's nearly free and murder to
  retrofit — not a deferred corner.)
- **Resource** — the room id is `{customer}:{story}:{branch}:{file}`, so rooms are disjoint across
  customer, story, and resource by construction.

---

## RBAC / authorization (CO-5, via `agent-acl` — not a new check)

- **Identity** is established once, on connect, by the shared `auth_validator` (the same callable HTTP
  uses — one validation path, ID-3) → `user_id` (+ claims). The socket carries the authenticated principal.
- **Authorization** is per-action, every time, via `agent.acl.authorize_on_entity`: `room.join`
  (presence), `entry.read`/`entry.write` (which ops/broadcasts a user may send/receive — CO-5). Deny →
  `AccessDenied` → error frame / refused op; never a silent allow.
- **Cross-pod coherence** — a revoked grant propagates fleet-wide via `agent.acl.invalidation` before
  the next check.
- **The roster is authorization-clean by construction** — a member is written only after `room.join`
  passes, so "who is in room X" never lists an unauthorized user.

---

## Existing consumers (aibots/metallm) — migrate, don't preserve

Backward-compat is a *consideration, not a constraint*. The dict-based `ConnectionRegistry` is
concurrency-unsafe and single-pod; the structurally-correct replacement is the collection-backed
membership + the synchronized socket map + the fanout. **We change `channels` to the right shape and
migrate aibots/metallm** (we own them). Where a pure single-pod, NATS-free deployment is genuinely
wanted (tests, tiny installs), it remains a deliberate **config** (no fanout/collection-L2 wired) — not
a preserved legacy dict path kept alive to avoid touching anyone. Change because it's right; never
gratuitously.

---

## The 3tears / scriob boundary (decided)

- **3tears owns the generic cross-pod transport + state + authz:** the socket + auth (`channels`),
  membership/presence (`core.collections`), room fanout (`channels` + `nats`), authorization
  (`agent-acl`), namespacing/tenancy (`nats.Subjects`/`kv` + the ACL namespace), reconnect/resume
  scaffolding. App-agnostic.
- **Scriob owns what is ours and cannot be generic:** op semantics (replace-range, stable position IDs
  — `A14`), the **op-log** (the authoritative `seq` + durable resume source), git-as-L3, provenance,
  and the per-story ACL *policy* (roles/grants; the *mechanism* is `agent-acl`).

---

## Decisions

- **D1 — Cross-pod rooms via NATS fanout** (`Subjects.room(...)`; each pod delivers to its local
  members).
- **D2 — Membership/presence = `BaseCollection` (L1 SQLite + L2 NATS)**, modelled on
  `HeartbeatCollection`; cross-pod coherence via the invalidation envelope; self-heal via heartbeat +
  sweep. **Concurrency-safe + indexed — never dicts.**
- **D3 — The socket is the only pod-local state**; the only in-process structure is a minimal,
  `asyncio.Lock`-synchronized `connection_id → live socket` map (snapshot-iterated). Reconnect resumes
  from the durable sequence — pod loss is transparent.
- **D4 — Ordering/seq from the durable source** (the op-log), never an in-process counter.
- **D5 — Structurally-best over backward-compat.** Change `channels` to the right shape and migrate
  existing consumers; backward-compat is a consideration, not a constraint. Single-pod/NATS-free stays
  a deliberate config, not a preserved legacy path.
- **D6 — No dicts for shared/queryable state.** Async + threaded ⇒ dicts race + slow; shared state goes
  in the concurrency-safe store (the collection). Only the live-socket handle map is in-process, synchronized.
- **D7 — Namespacing is structural** (`Subjects`/`kv` env prefix + ACL namespace tenant + scoped room
  id), and **tenancy-ready**: `customer_id` threaded everywhere, one fixed customer now, real
  customer ids later with zero re-architecture.
- **D8 — Authorization is `agent-acl`** (`authorize_on_entity`) on every join/broadcast/op (CO-5),
  cross-pod coherent; identity from the shared `auth_validator`.

---

## Build (task shards — 3tears `<area>-task-NN` convention; in this `docs/` dir)

Dependency order (the state layer is foundational; the existing `channels-task-01` fanout shard will be
renumbered/reframed to match this):

- **A — Membership/presence state layer** — `PresenceCollection : BaseCollection` (L1 SQLite + L2 NATS,
  modelled on `HeartbeatCollection`), keyed by `(customer, story, branch, file)` room; the
  `connection_id → live socket` synchronized map; the heartbeat + sweep self-heal (modelled on
  `HeartbeatSubscriber`). **Replaces the dict `ConnectionRegistry`.** Proof: real-NATS two-registry —
  membership written on A is queryable on B; a stale heartbeat expires.
- **B — Cross-pod room fanout** — `Subjects.room(...)` + the thin publish→per-pod-local-deliver
  backplane, delivering to local members resolved via layer A. Proof: real-NATS two-registry broadcast.
- **C — Authorization + typed frames + reconnect-resume** — gate join/broadcast/op with
  `authorize_on_entity`; a frame-type router so `editor.op`/`presence`/`cursor`/`typing` are first-class
  (chat unchanged); a generic event envelope carrying `version`/`seq`; the resume-from-seq handshake.

---

## Anti-patterns

- **DO NOT use dicts for room/membership/presence/shared state** — concurrency-unsafe (iterate-while-
  mutate across awaits/threads) + slow at scale; use the `BaseCollection`. The only in-process map is the
  minimal synchronized `connection_id → live socket` handle, snapshot-iterated.
- **DO NOT hand-roll the roster** with a raw `NatsKvBucket` — it is a `BaseCollection`.
- **DO NOT write a new authorization path** — join/broadcast/op go through `agent-acl`.
- **DO NOT keep an in-process `seq`** — the seq is the durable op-log sequence.
- **DO NOT broadcast only in-process** — a room message NATS-fans to every pod with a member.
- **DO NOT hardcode away the tenant dimension** — thread `customer_id` everywhere now (one fixed
  customer) so multi-tenancy is a config change later, not a rewrite.
- **DO NOT preserve the legacy dict path to avoid touching aibots/metallm** — migrate them; keep
  single-pod only as a deliberate config.
- **DO NOT make reconnect "re-fetch over HTTP and hope"** — resume is replay-from-seq off the op-log.
