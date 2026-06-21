# channels-task-01: Presence/membership state layer (PK-keyed `PresenceCollection` + synced socket map)

**Status:** SHIPPED. **Foundational** — `channels-task-02` (room fanout) and `channels-task-03`
(authz + frames + resume) build on the membership + socket map this lands.
**Scope:** `3tears-channels` — a new presence/state module + a reshaped connection registry — using
`threetears.core.collections` (L1 SQLite + L2 NATS) and `threetears.nats`. Net-new framework code.
**Origin:** `docs/channels-cross-pod-design.md` (D2, D3, D6, D7) + `registry-task-01` (which names the
"queryable cross-pod presence roster" as net-new and defers it to this design). Replaces the racy,
single-pod dict `ConnectionRegistry`.

---

## Objective

Build the concurrency-safe, cross-pod, tenancy-ready **state layer** for live rooms: who is connected,
who is in which `(customer, story, branch, file)` room, across pods — plus the minimal pod-local map of
live socket handles. Membership/presence lives in a `BaseCollection` (L1 SQLite + L2 NATS), **not
dicts**; the only in-process structure is a synchronized `connection_id → live socket` map. This is the
layer the fanout (task-02) delivers through and authz (task-03) gates.

---

## Design constraints (the hard, verified ones — do NOT reinvent around them)

- **PK-keyed only — `BaseCollection` has no secondary-field query** (verified: `base.py` exposes only
  pk get/save/delete; `SchemaBackedCollection`'s `WHERE`/list queries hit **L3/postgres**, wrong for
  ephemeral presence; the **NATS-KV wrapper exposes no key-listing**). So "who is in room X" is a
  **pk-get**, never a scan. Model it as **two pk-keyed entry types**:
  - **per-connection entry** — `pk = connection_id`; fields: `room_id`, `user_id`, `pod_id`,
    `customer_id`, `date_last_heartbeat` (+ optional cursor/selection later). Heartbeats refresh THIS
    entry (no contention).
  - **room-index entry** — `pk = room_id` = `{customer}:{story}:{branch}:{file}`; value = the set of
    member `connection_id`s. Updated only on **join/leave** (low churn). "Who's in room X" =
    `get(room_index)` → member ids → (optionally) pk-get each connection entry.
- **Ephemeral L1+L2 (no L3).** Mirror `HeartbeatCollection`: set `self.l3_pool = None`; override
  `fetch_from_store`/`save_to_store`/`delete_from_store` to raise. Presence is transient.
- **Cross-pod coherence = the collection's invalidation envelope** (`INVALIDATION_SUBJECT`) — free with
  `BaseCollection`'s L2 write path; peers evict L1 and refill from L2 on the next pk-get.
- **Self-heal = heartbeat + sweep**, NOT KV-TTL. Each connection entry carries `date_last_heartbeat`;
  a sweep loop (modelled on `registry/health.py` `HeartbeatSubscriber`) evicts connections whose
  heartbeat is older than a threshold and prunes them from their room-index — a dead pod's members
  vanish automatically.
- **The only in-process structure is a synchronized `connection_id → live socket` map** (live,
  non-serializable handles). Guard with an `asyncio.Lock` (event-loop-confined); **iterate over a
  snapshot, never mutate while awaiting.** No dicts for shared/queryable state.
- **Tenancy-ready under one customer.** `customer_id` is a first-class field + part of the room key;
  scriob passes one fixed customer now. No code path may assume single-tenant.
- **Reuse the wrappers** — collections via `BaseCollection`, NATS via the `NatsClient` wrapper +
  `Subjects` (never raw `nats`, never f-string subjects; the enforcement walker forbids it).
- **Replace, don't preserve.** This supersedes the dict-based `ConnectionRegistry`; aibots/metallm
  migrate. A pure single-pod/NATS-free mode stays a deliberate config (no L2/collection wired), not a
  kept-alive legacy dict path.

---

## Patterns to follow (file:line in the worktree)

- L1+L2-only collection: `packages/registry/src/threetears/registry/heartbeat_collection.py`
  (`HeartbeatCollection` — `l3_pool=None`, L3 methods raise, L1+L2 get/save/delete, serialize/deserialize).
- The sweep/health loop: `packages/registry/src/threetears/registry/health.py` (`HeartbeatSubscriber`).
- BaseEntity + L1 schema declaration: `packages/registry/src/threetears/registry/entities.py` +
  `.../registry/l1_cache.py` (how `HeartbeatEntity` + the `pod_heartbeats` table are defined).
- Collection construction + invalidation: `packages/core/src/threetears/core/collections/base.py`,
  `.../collections/registry.py` (`CollectionRegistry`, `INVALIDATION_SUBJECT`).
- Subjects: `packages/nats/src/threetears/nats/subjects.py` (add a presence/room key family here if one
  is needed for the room key; the cross-pod fanout subject is task-02).

---

## Files to Create / Modify (the build settles exact module placement)

### Create (in `3tears-channels`)
- a presence module: `PresenceCollection(BaseCollection[...])` (pk-keyed, per-connection + room-index
  entry types) + the entities + the L1 schema for the presence tables.
- a `RoomState` / reshaped registry: the synchronized `connection_id → live socket` map (asyncio.Lock)
  + the public surface `register/unregister(connection_id, socket)`, `join/leave(room_id, connection_id,
  user_id, customer_id, pod_id)`, `heartbeat(connection_id)`, `members(room_id) -> list[...]`,
  `local_sockets(room_id) -> list[socket]` (for the fanout).
- a presence sweeper (modelled on `HeartbeatSubscriber`).
- unit + integration tests.

### Modify / Retire
- the dict-based `ConnectionRegistry` in `websocket.py` — superseded by the above (migrate consumers).

---

## Implementation Notes

1. **Two collections (or one collection with two entry types)** — keep the per-connection heartbeat
   write off the room-index hot path so heartbeats never CAS the room set. Join/leave do both (the
   connection entry + a CAS update of the room-index member set, retry on revision conflict).
2. **`members(room_id)`** = `get(room_index)` → member set; cross-pod-complete because the room-index
   is pk-keyed and L2-coherent (a join on pod A invalidates pod B's L1 copy; B refills on next get).
3. **Sweeper** runs periodically: list known connection entries (the sweeper maintains the id set, like
   `HeartbeatSubscriber`), evict any whose `date_last_heartbeat` is stale, prune from the room-index.
4. **Snapshot iteration** for the socket map: copy the member-handle list under the lock, release, then
   `await send` — never hold the lock across `await`, never iterate the live dict.

---

## Anti-patterns

- DO NOT add a secondary-field scan — `BaseCollection` has none, the KV wrapper has no listing, and
  `SchemaBackedCollection` is L3-bound. Stay pk-keyed (per-connection + room-index).
- DO NOT use a dict for membership/presence/rooms — that is the racy, slow thing being replaced.
- DO NOT hold the socket-map lock across `await`, or iterate the live map — snapshot then send.
- DO NOT wire L3 — presence is ephemeral (`l3_pool=None`, raise on L3 paths, like `HeartbeatCollection`).
- DO NOT assume single-tenant — `customer_id` is first-class everywhere.
- DO NOT import `nats` directly or f-string subjects — `NatsClient` + `Subjects`.

---

## Acceptance Criteria

- [ ] `PresenceCollection` is L1+L2-only (pk-keyed; L3 paths raise), per-connection + room-index entries.
- [ ] **Cross-pod proof (integration, real NATS):** join on registry/collection A → `members(room)`
      queried on B returns the member (L2 + invalidation); leave on A → B no longer lists it.
- [ ] **Self-heal proof:** a connection whose heartbeat goes stale is evicted by the sweep and pruned
      from its room-index (asserted across two instances).
- [ ] The `connection_id → live socket` map is `asyncio.Lock`-synchronized and snapshot-iterated (a
      concurrency test: join/leave churn during a `local_sockets` read does not race).
- [ ] `customer_id` threaded through the room key + entries (a two-customer test shows isolation even
      with one customer configured).
- [ ] `channels` enforcement (no direct `nats`) green; mypy strict + ruff clean; existing `channels`
      tests updated for the migration (not weakened).

---

## Verification

```bash
cd /Users/pace/crypt/pub/dev-wsl/vscode/3tears/3tears-scriob
uv run pytest packages/channels/tests/ -m "not integration" -q
uv run pytest -m integration packages/channels/tests/integration/ -q   # Docker up
./scripts/lint.sh && ./scripts/typecheck.sh
```

---

## Enforcement Test Suggestions

- [ ] "no dict-typed attribute holds room/membership state in channels" — an AST walker, given how
      reflexively dicts get reached for, to keep shared state in the collection.
- [ ] "no secondary-field query against `PresenceCollection`" — keep it pk-keyed.
