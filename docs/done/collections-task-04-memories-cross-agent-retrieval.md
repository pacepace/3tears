# collections-task-04: Memories Cross-Agent Retrieval

**Status:** Ready for implementation (depends on collections-task-02; framework already in place)
**Scope:** `3tears/packages/agent-memory/`. `(3tears)` label.

---

## Objective

Apply the partition-column primitive to every memory-related table (`memories`, `media`, `media_content`, `memory_chunks`, `memory_refs`), and build the canonical cross-partition retrieval pattern: `@spans_partitions` Collection method that takes a resolved tuple of authorized agent ids, paired with an ACL-integrated service-layer caller that uses the unified evaluator + namespace_collection to resolve the authorized list.

This shard is the FIRST USER of the cross-partition retrieval pattern. The pattern it establishes generalizes to other domains (workspaces, conversations, audit events) in future shards. Get it right.

---

## Locked design decisions (do not re-question)

These were decided during the collections-task-02 walkthrough; this shard inherits them:

- **Memories partition column = `agent_id`.** Agent is the primary isolation boundary; customer + user are sub-scopes within an agent's data. Matches `project_tenant_means_agent.md` line 17 doctrine.
- **Restore `(agent_id, customer_id, user_id)` NOT NULL triplet on `memories`.** agent-memory v003 loosened agent_id/customer_id to nullable; this shard reverts that via a new vN migration. v003 stays untouched (no editing historical migrations). Pre-GA, no data backfill needed.
- **Memory child tables (`media`, `media_content`, `memory_chunks`) get composite FK `(agent_id, memory_id) REFERENCES memories(agent_id, memory_id)`.** Each child table gets a new `agent_id UUID NOT NULL` column; composite PK becomes `(agent_id, <existing_pk>)`. Children declare `partition_column='agent_id'`.
- **`memory_refs` already has composite PK `(conversation_id, item_id)` from phase 8.5l-2.** Just declare partition (`agent_id`? or `conversation_id`?) — see Design Context below.
- **Cross-partition retrieval pattern**: `@spans_partitions` Collection method takes `tuple[UUID, ...]` of partition values. ACL resolution lives at the service layer, NOT on the Collection. The Collection method takes a RESOLVED list. The service layer calls the unified evaluator to resolve "which agents has this caller been authorized to read memory from?" before calling the Collection.
- **Pre-GA = drop and recreate.** No dual paths. No back-compat shims. Single migration. All callers update in same commit.

---

## Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| MEM-01 | New agent-memory migration `vNNN_restore_memories_agent_customer_not_null.py`: ALTER `memories.agent_id` and `memories.customer_id` to NOT NULL. Drop existing `memories_pkey` (on `memory_id`) and replace with composite `PRIMARY KEY (agent_id, memory_id)`. UNIQUE (memory_id) constraint preserved alongside for FK targets that reference memories by memory_id alone. | P0 |
| MEM-02 | New migration: ALTER `media` to add `agent_id UUID NOT NULL`. Drop `media_pkey`. Add `PRIMARY KEY (agent_id, media_id)`. Add composite FK `(agent_id, memory_id) REFERENCES memories(agent_id, memory_id)` (replaces any existing simple FK on `memory_id`). UNIQUE (media_id) preserved. | P0 |
| MEM-03 | New migration: ALTER `media_content` analogous to MEM-02 (add `agent_id NOT NULL`, composite PK on `(agent_id, content_id)`, composite FK to memories). UNIQUE (content_id) preserved. | P0 |
| MEM-04 | New migration: ALTER `memory_chunks` analogous (add `agent_id NOT NULL`, composite PK on `(agent_id, chunk_id)`, composite FK to memories). UNIQUE (chunk_id) preserved. | P0 |
| MEM-05 | `MemoriesCollection`, `MediaCollection`, `MediaContentCollection`, `MemoryChunkCollection`, `MemoryRefsCollection` all declare `partition_column='agent_id'` in their `TableSchema`. The collections-task-02 framework's `__init_subclass__` enforcement fires; every public method must accept `agent_id` as a required first non-self arg, OR be `@spans_partitions`-decorated, OR be in `_partition_exempt_methods`. | P0 |
| MEM-06 | Refactor `_build_user_scope_clause` and every method that calls it (`find_for_user`, `find_similar_for_dedup`, `search_by_ids`, `search_by_semantic`, `search_by_fts`, `fetch_content_for_recall`, hybrid-search composers) to require `agent_id` as a positional argument. The `agent_id IS NULL` and `customer_id IS NULL` optional-filter branches in the legacy clause are deleted; agent_id is the partition predicate, customer_id is a required sub-scope, user_id is a required sub-scope. | P0 |
| MEM-07 | All 14 cache-bypass hybrid-search SQL strings in `MemoriesCollection` get `agent_id = $N` predicates added (one per joined table where applicable: memories on the parent side, media/media_content/memory_chunks on the child side using their composite FK). The cache-bypass annotations remain (these are genuine join queries the Collection API can't express through `BaseCollection.get`); the partition predicate is added; the AST walker now passes for these strings. | P0 |
| MEM-08 | New `@spans_partitions` method `MemoriesCollection.find_for_user_in_agents(*, user_id: UUID, agent_ids: tuple[UUID, ...], customer_id: UUID | None = None) -> list[MemoryEntity]`. Single SQL with `WHERE user_id = $1 AND agent_id = ANY($2)`. Empty `agent_ids` raises `ValueError` (must be a deliberately resolved set, never empty). | P0 |
| MEM-09 | New service-layer module `3tears/packages/agent-memory/src/threetears/agent/memory/access.py` exposing `MemoryAccessService`. Takes `acl_cache`, `namespace_collection`, `memories_collection` at construction. Method `find_for_user_across_authorized_agents(*, user_id: UUID, caller_user_id: UUID, customer_id: UUID) -> list[MemoryEntity]` that: (a) finds candidate `memory` namespaces under the customer, (b) filters to namespaces where `caller_user_id` has `memory.read` via the unified evaluator, (c) extracts `owner_agent_id` from each authorized namespace, (d) calls `memories_collection.find_for_user_in_agents(user_id=user_id, agent_ids=tuple(authorized_agent_ids), customer_id=customer_id)` with the resolved list. | P0 |
| MEM-10 | Multi-agent integration test in `tests/integration/test_memories_cross_agent_retrieval.py` (testcontainers Postgres + minimal NATS): seed two agents (A, B) with memories for the same user; grant agent A's user `memory.read` on agent B's memory namespace; call `find_for_user_across_authorized_agents`; assert results include both A's and B's memories for the user. Negative test: revoke the grant on agent B; assert results no longer include B's memories. | P0 |
| MEM-11 | Documentation in `3tears/docs/partition-column-pattern.md` describing the canonical cross-partition retrieval shape: Collection method with `@spans_partitions` taking `tuple[UUID, ...]`, paired with ACL-integrated service-layer caller. Memories is the worked example. Includes a "when to use this pattern" section + the principle that ACL resolution lives at the service layer, never on the Collection. | P0 |
| MEM-12 | Existing memory tests pass without modification to assertions. Test fixtures may need `agent_id` added where they constructed bare entity dicts; that's a fixture change, not an assertion change. | P0 |
| MEM-13 | AST partition walker (already in strict mode from collections-task-02) flips memories-table allowlist entries from `"deferred to collections-task-04"` to non-allowlisted (regular strict enforcement). Verify zero memories-table violations remain in strict mode. | P0 |

---

## Design Context

### `memory_refs` partition column choice

`memory_refs` has composite PK `(conversation_id, item_id)` from phase 8.5l-2. It tracks which memories are referenced from which conversation. Two valid partition choices:

- **conversation_id**: matches the existing PK shape; queries are typically "what memories does this conversation reference?"
- **agent_id**: matches the rest of the memory tables in this shard; would require adding `agent_id` column

Decision: **partition on `conversation_id`** for `memory_refs`. Reason: the existing PK is already correct (composite on conversation_id); adding agent_id would be redundant since conversation_id implies agent_id (a conversation belongs to one agent). Declaring `partition_column='conversation_id'` aligns with the existing schema and doesn't require a migration.

This is a deliberate variation: most memory tables partition on agent_id; memory_refs partitions on conversation_id because its access pattern is conversation-centric, not agent-centric. Document this in the new `partition-column-pattern.md`.

### Why `tuple[UUID, ...]` not `list[UUID]` for `find_for_user_in_agents`

The type signal matters. `list` connotes "any agents you want." `tuple` connotes "an immutable resolved set." The `@spans_partitions` decorator validates this at call time — passes `tuple`, fails `list`. Anyone reading the code sees the intent: "this is a deliberately-scoped, ACL-resolved partition list."

### ACL integration boundary

The Collection layer knows about partitions; it does NOT know about authorization. The service layer (`MemoryAccessService`) knows about authorization; it composes the partition Collection. This separation:

- Keeps Collection layer domain-pure and testable in isolation (mock ACL is irrelevant).
- Reuses the unified evaluator (no parallel ACL machinery in the Collection).
- Makes the authorization step visible at the call site that uses it (the service layer).

Pattern:
```python
class MemoryAccessService:
    def __init__(self, *, acl_cache, namespace_collection, memories_collection):
        self.acl_cache = acl_cache
        self.namespace_collection = namespace_collection
        self.memories_collection = memories_collection

    async def find_for_user_across_authorized_agents(
        self, *, user_id, caller_user_id, customer_id,
    ) -> list[MemoryEntity]:
        # 1. Resolve candidate memory namespaces under customer
        candidate = await self.namespace_collection.find_by_type_and_customer(
            namespace_type="memory", customer_id=customer_id,
        )
        # 2. Filter to authorized agents via unified evaluator
        authorized: list[UUID] = []
        for ns in candidate:
            ctx = EvaluationContext(
                namespace=AclNamespace(
                    id=ns.id,
                    customer_id=ns.customer_id,
                    namespace_type=ns.namespace_type,
                    owner_agent_id=ns.owner_agent_id,
                ),
                action=ACTION_MEMORY_READ,
                user_id=caller_user_id,
            )
            allowed = await evaluate_decision(
                ctx,
                membership_loader=self.acl_cache.membership_loader,
                grant_loader=self.acl_cache.grant_loader,
            )
            if allowed and ns.owner_agent_id is not None:
                authorized.append(ns.owner_agent_id)
        # 3. Single fan-out to Collection
        if not authorized:
            return []
        return await self.memories_collection.find_for_user_in_agents(
            user_id=user_id,
            agent_ids=tuple(authorized),
            customer_id=customer_id,
        )
```

### NamespaceCollection.find_by_type_and_customer

This method may not exist yet; verify. If not, add it as part of this shard — it's a natural Collection query method ("memory namespaces in customer X"). Single-partition (the customer is the partition for the namespace_collection — actually `namespace_collection` partitions on `scope_type` per collections-task-03, but this shard runs in parallel; use the existing API if collections-task-03 hasn't landed yet, with a follow-up note).

If `find_by_type_and_customer` doesn't exist:
```python
async def find_by_type_and_customer(
    self, *, namespace_type: str, customer_id: UUID,
) -> list[NamespaceEntity]:
    """find all namespaces of a given type within a customer."""
    rows = await self.l3_pool.fetch(
        """
        SELECT * FROM namespaces
        WHERE namespace_type = $1 AND customer_id = $2
        """,
        namespace_type, customer_id,
    )
    return [self.entity_class(dict(r), is_new=False, collection=self) for r in rows]
```

### Hybrid-search SQL refactor — example

Current shape (one of the 14, in `MemoriesCollection.search_by_semantic` or similar):
```sql
SELECT m.*, mc.content as chunk_content, ...
FROM memories m
LEFT JOIN memory_chunks mc ON mc.memory_id = m.memory_id
WHERE m.user_id = $1
  AND m.embedding <=> $2 < $3
ORDER BY m.embedding <=> $2
LIMIT $4
```

Refactored:
```sql
SELECT m.*, mc.content as chunk_content, ...
FROM memories m
LEFT JOIN memory_chunks mc ON mc.agent_id = m.agent_id AND mc.memory_id = m.memory_id
WHERE m.agent_id = $1
  AND m.user_id = $2
  AND m.embedding <=> $3 < $4
ORDER BY m.embedding <=> $3
LIMIT $5
```

Two changes: composite JOIN on `(agent_id, memory_id)`, and `agent_id` predicate added at the partition position. Every method's signature gains `agent_id` as a required first arg.

---

## Files to Create / Modify

### Create
- `3tears/packages/agent-memory/src/threetears/agent/memory/migrations/v00N_restore_memories_agent_customer_not_null.py` — schema migration (where N is the next version after current head)
- `3tears/packages/agent-memory/src/threetears/agent/memory/migrations/v00N+1_media_composite_fk.py`
- `3tears/packages/agent-memory/src/threetears/agent/memory/migrations/v00N+2_media_content_composite_fk.py`
- `3tears/packages/agent-memory/src/threetears/agent/memory/migrations/v00N+3_memory_chunks_composite_fk.py`
- `3tears/packages/agent-memory/src/threetears/agent/memory/access.py` — `MemoryAccessService`
- `3tears/packages/agent-memory/tests/integration/test_memories_cross_agent_retrieval.py`
- `3tears/docs/partition-column-pattern.md` — canonical pattern documentation

### Modify
- `3tears/packages/agent-memory/src/threetears/agent/memory/collections.py` — partition declarations + signature changes on every read/write method + 14 hybrid-search SQL strings + `find_for_user_in_agents` method
- Existing memory tests — fixture-only changes (add `agent_id` to constructed entities); no assertion changes per MEM-12

### Retire
- The `agent_id IS NOT NULL` and `customer_id IS NOT NULL` optional-filter branches in `_build_user_scope_clause` — agent_id and customer_id are now required predicates, not optional
- The "deferred to collections-task-04" allowlist entries in the AST partition walker

---

## Implementation Notes

1. **Migrations first.** Land the schema changes (MEM-01 through MEM-04) as one logical commit (or four small commits, your call). Tests against testcontainers Postgres verify the migration chain stays clean. Run the full agent-memory test suite after each migration to catch fixture breakage early.
2. **Schema declarations second.** Declare `partition_column='agent_id'` on each Collection's `TableSchema`. The `SchemaBackedCollection.__init_subclass__` enforcement fires immediately on import; expect class-definition errors that point at every method that doesn't take `agent_id` first. That's your punch list for MEM-06.
3. **Method signature refactor third.** Update each method to take `agent_id` as a required first arg. Update its callers in the same commit (no dual paths). Use ruff/mypy to verify call-site compatibility.
4. **Hybrid-search SQL fourth.** The 14 strings need careful work — composite JOINs change cardinality if done wrong. Test each affected method against testcontainers data after the change.
5. **Cross-agent retrieval method fifth.** `find_for_user_in_agents` is small; `MemoryAccessService` composes it with the evaluator. Both have unit tests against mocked dependencies + an integration test against real Postgres + real evaluator.
6. **Documentation last.** Once the pattern is proven through the integration test, write `partition-column-pattern.md` with the worked example. Future shards (workspaces cross-agent, conversations cross-agent if those use cases emerge) reference this doc as the canonical pattern.

---

## Anti-patterns

- **DO NOT** add ACL resolution to the `MemoriesCollection`. The Collection takes a resolved list; it does not know about ACL. ACL lives at the service layer.
- **DO NOT** make `find_for_user_in_agents` accept a `list[UUID]` "for ergonomics." `tuple[UUID, ...]` is the type contract. The `@spans_partitions` decorator validates it.
- **DO NOT** allow `agent_ids=()` (empty tuple). The service layer returns `[]` early when no authorized agents resolve; the Collection method raises `ValueError` if it ever sees an empty tuple — that's a bug surface (caller should have short-circuited).
- **DO NOT** keep the `agent_id IS NULL` optional-filter branches in `_build_user_scope_clause`. agent_id is now mandatory; the legacy "optional scoping tag" semantics are retired with v003's reversion.
- **DO NOT** edit historical migrations. v003 stays untouched. New vN migration restores NOT NULL.
- **DO NOT** ship a back-compat shim for `find_for_user(user_id, agent_id=None)`. agent_id is required, period.
- **DO NOT** widen `find_for_user_in_agents` to also be the single-partition path "to avoid two methods." `find_for_user(agent_id=...)` is the partition-bound workhorse; `find_for_user_in_agents(agent_ids=...)` is the deliberate cross-partition method. Two methods, two intents, both clear.

---

## Success Criteria

- [ ] Memories schema: agent_id NOT NULL, customer_id NOT NULL, composite PK `(agent_id, memory_id)`, UNIQUE (memory_id) preserved.
- [ ] Child tables (`media`, `media_content`, `memory_chunks`): `agent_id NOT NULL`, composite PK on `(agent_id, <pk>)`, composite FK to memories, UNIQUE (`<pk>`) preserved.
- [ ] All 5 memory collections declare `partition_column='agent_id'` (memory_refs declares `partition_column='conversation_id'` — variation documented).
- [ ] Every method on every memory Collection accepts `agent_id` (or is `@spans_partitions`-decorated).
- [ ] All 14 hybrid-search SQL strings carry `agent_id` predicates + composite JOINs.
- [ ] `MemoriesCollection.find_for_user_in_agents` exists and is `@spans_partitions`-decorated.
- [ ] `MemoryAccessService.find_for_user_across_authorized_agents` exists, integrates with `acl_cache` + `namespace_collection`, resolves authorized agents via the unified evaluator before fan-out.
- [ ] Multi-agent integration test passes (positive + negative grant scenarios).
- [ ] AST partition walker in strict mode: zero memories-table violations.
- [ ] Existing memory tests pass without assertion changes.
- [ ] `3tears/docs/partition-column-pattern.md` documents the pattern with the memories worked example.
- [ ] mypy + ruff clean on touched files.

---

## Verification

```bash
cd /Users/pace/crypt/pub/dev-wsl/vscode/3tears/3tears

# new migrations apply cleanly
uv run --directory packages/agent-memory pytest tests/integration/ -v -k migration

# Collection schema declarations + signature changes
uv run --directory packages/agent-memory pytest tests/unit/ -v
uv run --directory packages/agent-memory ruff check src/ tests/
uv run --directory packages/agent-memory mypy src/

# integration: cross-agent retrieval positive + negative
uv run --directory packages/agent-memory pytest tests/integration/test_memories_cross_agent_retrieval.py -v

# AST walker strict — zero memories violations
PARTITION_ENFORCEMENT_MODE=strict uv run --directory packages/core pytest tests/enforcement/test_partition_column_enforcement.py -v

# downstream consumers stay green
cd /Users/pace/crypt/pub/dev-wsl/vscode/3tears/14-eng-ai-bot
uv run pytest tests/enforcement/ -q
uv run pytest tests/integration/test_audit_pipeline.py tests/integration/test_namespace_invariants.py -v
```
