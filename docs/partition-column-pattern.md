# The Partition Column Pattern

> Status: canonical pattern as of `collections-task-04` (memories cross-agent retrieval).
>
> Worked example: `MemoriesCollection` (single-partition) +
> `MemoryAccessService` (cross-partition).

This document describes the platform's pattern for **isolating data per
partition** and the **canonical shape for cross-partition retrieval**
when authorization permits it.

It is the single source of truth for:

- How to declare a partition column on a `BaseCollection` /
  `SchemaBackedCollection`.
- How every method signature on a partitioned Collection must
  acknowledge the partition.
- How to compose **authorization** (service layer) with
  **partition fan-out** (Collection layer) without weakening either.
- When to reach for the cross-partition pattern, and when not to.

---

## What is a partition?

A **partition** is the structural unit of isolation between tenants of
shared infrastructure. In the platform's vocabulary the primary
isolation boundary is the **agent**: every agent occupies a logical
partition of the per-customer data surface, and one agent's runtime
must never observe another agent's rows by accident.

Partitioning is enforced through the **partition column** declared on
each table's schema. Every row carries a partition value (typically
`agent_id`), and every read / write predicate includes the partition
column. When a Collection method forgets the partition predicate, the
SQL touches every partition — that is the **cross-partition bleed**
class of bug, and the partition primitive exists to prevent it
structurally.

Partition columns scale beyond `agent_id`:

| Table | Partition column | Rationale |
|---|---|---|
| `memories`, `media`, `media_content`, `memory_chunks` | `agent_id` | agent owns the memory namespace |
| `conversations`, `workspaces` | `agent_id` | agent owns conversations and workspaces |
| `context_items` | `conversation_id` | conversation is the narrower isolation unit for live agent state |
| `workspace_files`, `workspace_file_versions` | `workspace_id` | workspace is the isolation unit for files |
| `conversation_memory_refs` | `conversation_id` | the table tracks per-conversation surfacing of items; conversation is the natural query key |

The choice of partition column is **not arbitrary**: it is the column
the dominant access pattern always filters on. Tables whose access
pattern is narrowly conversation-scoped partition on `conversation_id`
even if `agent_id` is also present (a conversation belongs to one
agent, so `conversation_id` implies `agent_id`).

---

## Declaring a partition

Add `partition=True` to one column in the `TableSchema`:

```python
from threetears.core.collections.schema_backed import (
    Column, SchemaBackedCollection, TableSchema, UUID_TYPE, STRING_TYPE,
)


class MemoriesCollection(SchemaBackedCollection[MemoryEntity]):
    primary_key_column: str | tuple[str, ...] = ("agent_id", "memory_id")
    schema = TableSchema(
        name="memories",
        primary_key=("agent_id", "memory_id"),
        columns=[
            Column("memory_id", UUID_TYPE),
            Column("agent_id", UUID_TYPE, partition=True),
            Column("user_id", UUID_TYPE, immutable=True),
            Column("content", STRING_TYPE),
            # ... other columns
        ],
        cas_column="date_updated",
    )
```

Constraints enforced by `TableSchema.__post_init__`:

- **Exactly one** column per table may have `partition=True`. Multiple
  partition columns is an error.
- The partition column **must be part of the primary key**. The
  composite PK enforces row uniqueness through the partition.
- Partition columns are **automatically immutable**. Mutating the
  partition value would corrupt the partition map; the schema validator
  coerces `immutable=True` even if the declaration omits it.

---

## The `__init_subclass__` guard

When a Collection's schema declares a partition column,
`SchemaBackedCollection.__init_subclass__` enforces — at class
definition time — that every public method on the subclass either:

1. **accepts the partition column as a parameter** (positional or
   keyword-only), so the SQL generator can include the partition
   predicate naturally, OR
2. **is decorated with `@spans_partitions`**, declaring "this method
   deliberately fans out across multiple partition values supplied as
   a tuple", OR
3. **is listed in `_partition_exempt_methods`** with a documented
   rationale (a narrow allowlist for genuine framework overrides).

A subclass that violates the guard fails at import time with
`PartitionEnforcementError`. There is no runtime fallback — the bug
surfaces during CI, before any cross-partition leak can ship.

---

## The cross-partition retrieval pattern

There are legitimate cases where a caller needs to read across
multiple partitions: a user logged into one agent wants to see
memories the same user has accumulated across other agents in the
same customer they have been authorized to read from.

The pattern composes two layers:

### Layer 1: Collection method with `@spans_partitions`

Decorated method on the Collection that takes a **resolved tuple**
of partition values:

```python
from threetears.core.collections.schema_backed import spans_partitions


class MemoriesCollection(SchemaBackedCollection[MemoryEntity]):
    # ... schema declaration above ...

    @spans_partitions
    async def find_for_user_in_agents(
        self,
        *,
        user_id: UUID,
        agent_ids: tuple[UUID, ...],
        customer_id: UUID | None = None,
    ) -> list[MemoryEntity]:
        """fetch memories for ``user_id`` across an authorized set of agents."""
        if self.l3_pool is None:
            return []
        # cache-bypass: cross-partition fan-out by ANY($N::uuid[]) is
        # not primary-key-addressable; method on Collection preserves
        # single entry point and is decorated @spans_partitions so the
        # AST walker accepts the deliberate fan-out shape.
        rows = await self.l3_pool.fetch(
            "SELECT * FROM memories "
            "WHERE agent_id = ANY($1::uuid[]) AND user_id = $2 "
            "AND is_deleted = false "
            "ORDER BY date_created DESC",
            list(agent_ids),
            user_id,
        )
        # ... materialize entities ...
```

**Contract enforced by `@spans_partitions`:**

- The plural-shaped argument (any parameter name ending in `_ids`)
  **must be a `tuple`**. Passing a `list` raises `TypeError`. The
  tuple shape is the type signal that the caller has deliberately
  resolved the authorized set upstream.
- The tuple **must be non-empty**. Passing `()` raises `TypeError`.
  An empty fan-out is a bug surface — the service layer is
  responsible for short-circuiting to `[]` when no partitions
  authorize.

The Collection method **knows nothing about authorization**. It accepts
the resolved set and trusts the caller. This separation keeps the
Collection layer domain-pure and testable in isolation against a mock
pool — no ACL machinery is in the way.

### Layer 2: Service composes ACL with the Collection

The **service layer** composes the unified RBAC evaluator with the
`@spans_partitions` Collection method:

```python
from threetears.agent.acl import (
    EvaluationContext, Namespace as AclNamespace, evaluate_decision,
)


class MemoryAccessService:
    def __init__(
        self, *,
        acl_cache,
        namespace_collection,
        memories_collection,
    ):
        self.acl_cache = acl_cache
        self.namespace_collection = namespace_collection
        self.memories_collection = memories_collection

    async def find_for_user_across_authorized_agents(
        self, *, user_id, caller_user_id, customer_id,
    ):
        # 1. enumerate candidate memory namespaces under the customer
        candidate = await self.namespace_collection.find_by_type_and_customer(
            namespace_type="memory", customer_id=customer_id,
        )

        # 2. evaluate memory.read on each via the unified evaluator
        authorized: list[UUID] = []
        for ns in candidate:
            ctx = EvaluationContext(
                namespace=AclNamespace(
                    id=ns.id,
                    customer_id=ns.customer_id,
                    namespace_type=ns.namespace_type,
                    owner_agent_id=ns.owner_agent_id,
                ),
                action="memory.read",
                user_id=caller_user_id,
            )
            allowed = await evaluate_decision(
                ctx,
                membership_loader=self.acl_cache.membership_loader,
                grant_loader=self.acl_cache.grant_loader,
            )
            if allowed and ns.owner_agent_id is not None:
                authorized.append(ns.owner_agent_id)

        # 3. short-circuit when no agents authorize — the @spans_partitions
        #    decorator refuses an empty tuple and that refusal is correct,
        #    but the service contracts at the same boundary so the noise
        #    stays out of the call stack.
        if not authorized:
            return []

        # 4. single fan-out to the Collection
        return await self.memories_collection.find_for_user_in_agents(
            user_id=user_id,
            agent_ids=tuple(authorized),
            customer_id=customer_id,
        )
```

The service layer is where **authorization meets partitioning**. It is
the only place that:

- knows about ACL (loaders, evaluator, decision contexts).
- knows about candidate namespace enumeration.
- transforms a customer-scoped namespace catalogue into a
  partition-scoped tuple.

The service is testable in isolation against a mock evaluator + mock
Collection. The Collection is testable in isolation against a real or
mock pool. **Neither layer carries the other layer's responsibility.**

---

## Why `tuple[UUID, ...]` and not `list[UUID]`?

The type signal matters. `list` reads as "any agents you want." `tuple`
reads as "an immutable, deliberately resolved set." The
`@spans_partitions` decorator validates the tuple shape at call time,
so a stray `list` argument fails loud rather than silently fanning out
across whatever the caller happened to pass.

This is the same discipline as `frozenset` vs `set`: the immutable
form signals "I am not modifying this" at the type level. For
cross-partition fan-out, immutability is the deliberateness signal.

---

## When to use this pattern

Reach for the cross-partition pattern **only** when the access pattern
is genuinely cross-partition:

- A user logged into agent A asks "what do I know about Foo?" and the
  caller is authorized to read memories from agents A, B, and C in the
  same customer.
- An admin queries audit events across every agent in their customer.
- A workspace tool surfaces files from every workspace the user has
  read access to.

**DO NOT** reach for it when the access pattern is naturally
single-partition. If the caller already knows which partition they are
in, pass `agent_id=...` to the partition-bound method. Two methods,
two intents, both clear:

| Use case | Method | Argument shape |
|---|---|---|
| "give me this user's memories within this agent" | `find_by_user(...)` | `agent_id: UUID` (single value) |
| "give me this user's memories across every agent the caller can read" | `find_for_user_in_agents(...)` | `agent_ids: tuple[UUID, ...]` (resolved set) |

The partition-bound method is the workhorse. The cross-partition
method is the deliberate fan-out. **The two methods coexist; do not
collapse them into a single nullable-aware variant** — that erases the
partition contract and reintroduces the very bug class the partition
primitive exists to prevent.

---

## Variation: when the partition column differs

`memory_refs` is the canonical example: it tracks which memories have
been surfaced in which conversation, with composite PK
`(conversation_id, item_id)`. The partition column is
**`conversation_id`**, not `agent_id`, even though the rest of the
memory tables partition on `agent_id`.

The reason: `memory_refs` is queried per-conversation
(`find_by_conversation`). A conversation belongs to exactly one agent,
so `conversation_id` implies `agent_id`; the access pattern is
narrower (conversation-scoped) and the partition column reflects that.
Adding `agent_id` would be redundant.

When a table's access pattern is narrower than the agent boundary,
**partition on the narrower column**. Document the variation. The
cross-partition retrieval pattern still applies — just substitute
`agent_id` with whatever the partition column is for that table.

---

## Anti-patterns

These are **rejected on code review**:

- **Adding ACL evaluation inside a Collection method.** The Collection
  layer is domain-pure. ACL belongs at the service layer.
- **Accepting `list[UUID]` for the cross-partition argument "for
  ergonomics."** `tuple[UUID, ...]` is the type contract. The
  decorator validates it.
- **Passing `agent_ids=()` to the cross-partition method.** Service
  layers must short-circuit to `[]` before invoking the Collection.
  An empty tuple is a bug surface.
- **Dropping the partition column from the SQL "because the caller
  knows what partition they're in."** Future callers will not. The
  partition predicate is structural defense, not redundancy.
- **Adding `_partition_exempt_methods` entries with rationales like
  "internal helper" or "tests need this."** Exemptions are last-resort
  with **specific** rationales. The first two resolution paths are:
  (1) add the partition column to the signature; (2) decorate with
  `@spans_partitions`.
- **Translating `tuple[UUID, ...]` to `list[UUID]` inside the
  Collection method "to make ANY work."** asyncpg accepts both;
  `list(agent_ids)` at the SQL boundary is the right place to coerce.
  The tuple-shape contract on the API is what matters.

---

## Test surface

Two enforcement tests guard the partition primitive:

1. **`tests/enforcement/test_partition_column_enforcement.py`** —
   AST walker that scans every package's source files for SQL string
   literals touching a partitioned table without including the
   partition column name in the WHERE / SET / VALUES context. Mode is
   strict by default; `PARTITION_ENFORCEMENT_MODE=report` available
   during cleanup windows.
2. **`SchemaBackedCollection.__init_subclass__`** — runtime guard at
   class-definition time that flags any public method on a partitioned
   Collection that neither accepts the partition column nor opts into
   `@spans_partitions`.

Both run in under 15s. CI runs the AST walker; the `__init_subclass__`
guard fires on every test that imports the package.

For cross-partition retrieval surfaces, write integration tests that:

- seed rows in two or more partitions.
- exercise the service-layer composition with positive (authorize all)
  and negative (deny one) grant scenarios.
- assert the negative case **does not** surface rows from the denied
  partition.

`tests/integration/test_memories_cross_agent_retrieval.py` is the
worked example.

---

## Summary

- **Single column per table** carries `partition=True`. It is part of
  the PK and is automatically immutable.
- **Every public method on the Collection** acknowledges the partition
  column — as a parameter, via `@spans_partitions`, or via a narrow
  documented exemption.
- **Cross-partition retrieval** is a **two-layer composition**:
  service evaluates ACL + resolves authorized partitions; Collection
  fans out via `@spans_partitions` over the resolved tuple.
- **Tuple, not list, for the fan-out argument.** `@spans_partitions`
  enforces this at call time.
- **Two methods, two intents:** partition-bound workhorse +
  cross-partition deliberate fan-out. Never collapse them.

Future shards apply this pattern to other domains as cross-partition
access cases emerge. Reference this document in the shard so the
reviewer evaluates the proposal against the canonical shape.
